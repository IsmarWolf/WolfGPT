"""WolfGPT Phase 1 - the model, written by hand. Read top to bottom: each class is one idea.

Data flow for one batch of text (B sequences, T characters each):
  ids (B,T) -> token embedding + position embedding -> [Block x n_layer] -> LayerNorm -> logits (B,T,vocab)
Each Block = attention (tokens look at earlier tokens) + MLP (each token thinks on its own).
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65     # how many different characters exist
    block_size: int = 256    # context length: how many characters the model can look at
    n_layer: int = 6         # how many Blocks are stacked
    n_head: int = 6          # attention heads per block
    n_embd: int = 384        # width: how many numbers describe each character
    dropout: float = 0.2     # randomly zero some values while training (fights memorizing)


class CausalSelfAttention(nn.Module):
    """Every position looks at itself and EARLIER positions, and blends what it finds.

    q (query)  = "what am I looking for?"
    k (key)    = "what do I contain?"
    v (value)  = "what do I hand over if you pick me?"
    score = q . k, softmax turns scores into weights that sum to 1, output = weights @ v.
    The causal mask hides the future so the model cannot cheat when predicting the next char.
    """
    def __init__(self, c: GPTConfig):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.n_head, self.n_embd = c.n_head, c.n_embd
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)   # one matmul makes q, k and v
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)       # mixes the heads back together
        self.attn_drop = c.dropout
        self.resid_drop = nn.Dropout(c.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # split the width into heads: (B,T,C) -> (B,n_head,T,C/n_head)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Fused GPU kernel computing softmax(q k^T / sqrt(d)) v with the causal mask.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.attn_drop if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # glue the heads back together
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Per-position 'thinking': widen 4x, apply a non-linearity (GELU), narrow back."""
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """LayerNorm -> attention -> add;  LayerNorm -> MLP -> add.
    'add' = residual connection: the block only has to learn a small CORRECTION to its input,
    which is what makes deep stacks trainable."""
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(c.n_embd), CausalSelfAttention(c)
        self.ln2, self.mlp = nn.LayerNorm(c.n_embd), MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)   # char id -> vector of n_embd numbers
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)   # position 0..T-1 -> vector (order matters!)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)  # vector -> one score per character
        self.head.weight = self.tok_emb.weight   # weight tying: reading and writing chars share a table
        self.apply(self._init)
        for n, p in self.named_parameters():     # scale down residual outputs so deep stacks start stable
            if n.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())   # tied weights are counted once

    def forward(self, ids, targets=None):
        B, T = ids.shape
        assert T <= self.c.block_size
        pos = torch.arange(T, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))                    # (B,T,vocab)
        loss = None
        if targets is not None:                             # cross-entropy: -log(prob given to the true next char)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None):
        """Predict one char, append it, repeat. temperature<1 = safer, >1 = wilder."""
        for _ in range(max_new_tokens):
            logits, _ = self(ids[:, -self.c.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
