from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class AlphaOmegaReport:
    l2_norm: torch.Tensor          # (T+1,) mean over batch
    cosine: torch.Tensor           # (T+1,) mean over batch
    score: float
    best_return_step: int


def _flatten(x: torch.Tensor) -> torch.Tensor:
    # (B,H,W,C) -> (B, H*W*C)
    return x.reshape(x.size(0), -1)


def alpha_omega_metrics(x0: torch.Tensor, xt: torch.Tensor, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns per-batch metrics:
      - l2_norm: ||Xt - X0|| / (||X0|| + eps)
      - cosine: cosine_similarity(Xt, X0)
    """
    f0 = _flatten(x0)
    ft = _flatten(xt)

    diff = ft - f0
    l2 = torch.norm(diff, dim=-1) / (torch.norm(f0, dim=-1) + eps)

    dot = (f0 * ft).sum(dim=-1)
    denom = (torch.norm(f0, dim=-1) * torch.norm(ft, dim=-1) + eps)
    cos = dot / denom
    return l2, cos


@torch.no_grad()
def run_alpha_omega_spatial(grid_module, x0: torch.Tensor, *, steps: int = 12) -> AlphaOmegaReport:
    """
    grid_module: typically SpiralGrid (callable as grid_module(x, steps=1))
    x0: (B,H,W,C)

    Computes series over t=0..steps:
      l2_norm[t] = mean_B normalized L2 distance between Xt and X0
      cosine[t]  = mean_B cosine similarity between Xt and X0

    Score heuristic:
      maximize (cosine - l2_norm). High when we are both aligned and close to origin.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if x0.dim() != 4:
        raise ValueError("x0 must be (B,H,W,C)")

    l2_series: List[torch.Tensor] = []
    cos_series: List[torch.Tensor] = []

    xt = x0
    for t in range(steps + 1):
        l2, cos = alpha_omega_metrics(x0, xt)
        l2_series.append(l2.mean())
        cos_series.append(cos.mean())

        if t < steps:
            xt = grid_module(xt, steps=1)

    l2_t = torch.stack(l2_series)    # (T+1,)
    cos_t = torch.stack(cos_series)  # (T+1,)

    signal = cos_t - l2_t
    best_step = int(torch.argmax(signal).item())
    score = float(signal[best_step].item())

    return AlphaOmegaReport(
        l2_norm=l2_t,
        cosine=cos_t,
        score=score,
        best_return_step=best_step,
    )

