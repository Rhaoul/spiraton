from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, List, Tuple, Optional

import torch
import torch.nn as nn

from .neighborhood import neighbors, Neighborhood


SpiralOrder = Literal["outward", "inward"]
AggMode = Literal["mean", "sum"]


def spiral_indices(h: int, w: int, order: SpiralOrder = "outward") -> List[Tuple[int, int]]:
    """
    Deterministic spiral traversal of an h×w grid.

    outward:
      - start at center (h//2, w//2)
      - then ring by ring (perimeter of growing squares), clockwise

    inward:
      - reverse of outward
    """
    if h <= 0 or w <= 0:
        raise ValueError("h and w must be positive")

    cy, cx = h // 2, w // 2
    seen = set()
    seq: List[Tuple[int, int]] = []

    def add(y: int, x: int) -> None:
        if 0 <= y < h and 0 <= x < w and (y, x) not in seen:
            seen.add((y, x))
            seq.append((y, x))

    add(cy, cx)

    max_r = max(cy, h - 1 - cy, cx, w - 1 - cx)
    for r in range(1, max_r + 1):
        top = cy - r
        bot = cy + r
        left = cx - r
        right = cx + r

        # top row: left -> right
        for x in range(left, right + 1):
            add(top, x)
        # right col: top+1 -> bot
        for y in range(top + 1, bot + 1):
            add(y, right)
        # bottom row: right-1 -> left
        for x in range(right - 1, left - 1, -1):
            add(bot, x)
        # left col: bot-1 -> top+1
        for y in range(bot - 1, top, -1):
            add(y, left)

    # Safety: ensure full coverage
    if len(seq) != h * w:
        # fill any missed (shouldn't happen, but keep it robust)
        for y in range(h):
            for x in range(w):
                add(y, x)

    if order == "inward":
        seq = list(reversed(seq))
    elif order != "outward":
        raise ValueError(f"Unknown spiral order: {order}")

    return seq


@dataclass(frozen=True)
class SpiralGridConfig:
    channels: int
    neighborhood: Neighborhood = "von_neumann"
    aggregator: AggMode = "mean"
    spiral: SpiralOrder = "outward"
    steps_default: int = 1


class SpiralGrid(nn.Module):
    """
    SpiralGrid: apply a Spiraton-like cell across a (B,H,W,C) grid with spiral propagation.

    Core idea:
      - For each cell (y,x), compute neighbor aggregate vector (C,)
      - Combine local + neighbor into u = concat(local, neigh) => (B,2C)
      - Feed to `cell(u)` which must return scalar (B,)
      - Update the local vector via y_to_vec(y) (scalar->C) with residual add

    This keeps the grid update expressive while remaining simple and testable.
    """

    def __init__(
        self,
        cell: nn.Module,
        *,
        channels: int,
        neighborhood: Neighborhood = "von_neumann",
        aggregator: AggMode = "mean",
        spiral: SpiralOrder = "outward",
        steps_default: int = 1,
    ) -> None:
        super().__init__()
        self.cell = cell
        self.cfg = SpiralGridConfig(
            channels=channels,
            neighborhood=neighborhood,
            aggregator=aggregator,
            spiral=spiral,
            steps_default=steps_default,
        )
        # scalar -> vector update
        self.y_to_vec = nn.Linear(1, channels, bias=False)

    def _neighbor_agg(self, grid: torch.Tensor, y: int, x: int) -> torch.Tensor:
        # grid: (B,H,W,C)
        B, H, W, C = grid.shape
        coords = neighbors(y, x, H, W, self.cfg.neighborhood)

        if not coords:
            return torch.zeros(B, C, device=grid.device, dtype=grid.dtype)

        neigh = torch.stack([grid[:, yy, xx, :] for (yy, xx) in coords], dim=0)  # (N,B,C)
        if self.cfg.aggregator == "mean":
            return neigh.mean(dim=0)  # (B,C)
        if self.cfg.aggregator == "sum":
            return neigh.sum(dim=0)
        raise ValueError(f"Unknown aggregator: {self.cfg.aggregator}")

    def forward(self, x: torch.Tensor, *, steps: Optional[int] = None) -> torch.Tensor:
        """
        x: (B,H,W,C)
        returns: (B,H,W,C)
        """
        if x.dim() != 4:
            raise ValueError("x must be (B,H,W,C)")
        B, H, W, C = x.shape
        if C != self.cfg.channels:
            raise ValueError(f"last dim must be channels={self.cfg.channels}")

        K = steps if steps is not None else self.cfg.steps_default
        if K < 1:
            raise ValueError("steps must be >= 1")

        path = spiral_indices(H, W, order=self.cfg.spiral)

        grid = x
        for _ in range(K):
            # sequential propagation: updates are visible to subsequent cells
            g = grid.clone()
            for (y, xx) in path:
                local = g[:, y, xx, :]                 # (B,C)
                neigh = self._neighbor_agg(g, y, xx)   # (B,C)
                u = torch.cat([local, neigh], dim=-1)  # (B,2C)

                y_scalar = self.cell(u)                # (B,)
                delta = self.y_to_vec(y_scalar.unsqueeze(-1))  # (B,C)
                g[:, y, xx, :] = local + delta
            grid = g

        return grid

