"""WolfGPT Phase 3 model: the Phase 1 GPT, upgraded the way modern open models do it.

What changed vs Phase 1 (and why):
  - RMSNorm instead of LayerNorm: simpler (no mean subtraction), slightly faster, works as well.
  - RoPE (rotary position embedding) instead of a learned position table: position is encoded by
    ROTATING q and k, so attention scores depend on how far apart two tokens are. Generalises better.
  - SwiGLU MLP instead of GELU MLP: a gated MLP (silu(gate) * up) that gives better loss per parameter.
  - No dropout: with billions of tokens the model cannot memorise the data, so dropout only slows learning.
  - Everything else (causal self-attention, residual blocks, weight tying) is unchanged.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 32768
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    @property
    def mlp_hidden(self):                    # ~8/3 * width, rounded to a multiple of 64 (GPU-friendly)
        return int(8 * self.n_embd / 3 / 64 + 0.5) * 64


SIZES = {  # "pilot" is the quick ~49M test model; "main" is the ~110M-class model for the long run
    "pilot": dict(n_layer=10, n_head=8, n_embd=512),
    "main": dict(n_layer=12, n_head=12, n_embd=768),
}


def rope_tables(T, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))   # one frequency per pair of dims
    ang = torch.outer(torch.arange(T).float(), inv)                           # (T, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate each (x1, x2) pair of channels by an angle that grows with position."""
    T, half = x.size(2), x.size(-1) // 2
    cos, sin = cos[:T][None, None], sin[:T][None, None]                       # (1,1,T,half)
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        assert c.n_embd % c.n_head == 0
        self.h, self.d = c.n_head, c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.qkv.weight.muon_split = 3       # q, k, v stacked: Muon orthogonalises each separately
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, self.d).transpose(1, 2) for t in (q, k, v))   # (B,h,T,d)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)                        # position enters here
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.up = nn.Linear(c.n_embd, 2 * c.mlp_hidden, bias=False)    # gate and value in one matmul
        self.up.weight.muon_split = 2        # gate, value stacked: Muon orthogonalises each separately
        self.down = nn.Linear(c.mlp_hidden, c.n_embd, bias=False)

    def forward(self, x):
        gate, val = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * val)


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.attn = nn.RMSNorm(c.n_embd, eps=1e-6), Attention(c)
        self.n2, self.mlp = nn.RMSNorm(c.n_embd, eps=1e-6), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = nn.RMSNorm(c.n_embd, eps=1e-6)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                          # weight tying
        cos, sin = rope_tables(c.block_size, c.n_embd // c.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():                            # keep the residual stream stable at depth
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, T):
        """Training FLOPs per token: 6 per parameter (2 forward + 4 backward) plus the attention term.
        Attention term is 6*L*C*T (not 12): with causal masking FlashAttention skips the upper half."""
        return 6 * self.num_params() + 6 * self.c.n_layer * self.c.n_embd * T

    def forward(self, ids, targets=None):
        x = self.tok_emb(ids)
        for blk in self.blocks:
            x = blk(x, self.cos, self.sin)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
