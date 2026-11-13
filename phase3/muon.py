"""Muon optimizer, written out in full.

AdamW updates every weight independently. Muon treats each weight MATRIX as a whole:
  1. keep a momentum buffer of the gradient,
  2. replace the update by its nearest 'orthogonal' matrix (all singular values pushed to ~1),
     computed with 5 cheap Newton-Schulz iterations,
  3. step in that direction.
Intuition: ordinary gradients are dominated by a few directions; orthogonalising makes the update
use all directions evenly, which tends to learn faster per step. Muon is only for 2-D hidden weight
matrices; embeddings and norm gains stay on AdamW.
"""
import torch


def newton_schulz(G, steps=5):
    """Works on one matrix (r, c) or a batch of them (k, r, c): each matrix is orthogonalised separately."""
    a, b, c = 3.4445, -4.7750, 2.0315          # tuned coefficients (K. Jordan) for a fast quintic iteration
    X = G.bfloat16()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)   # singular values must start <= 1 (per matrix)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.mT if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("buf", torch.zeros_like(p.grad))
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"])           # Nesterov momentum
                # Fused weights (qkv = 3 matrices, SwiGLU up = gate+value) carry p.muon_split = k:
                # orthogonalise each of the k row-blocks on its own, otherwise one block (e.g. q) dominates.
                k = getattr(p, "muon_split", 1)
                r, c = upd.shape
                upd = newton_schulz(upd.view(k, r // k, c), g["ns_steps"]).view(r, c)
                upd = upd * max(1.0, (r // k) / c) ** 0.5            # keep step size comparable across shapes
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd.to(p.dtype), alpha=-g["lr"])
