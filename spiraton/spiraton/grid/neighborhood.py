from __future__ import annotations

from typing import Iterable, List, Tuple, Literal

Neighborhood = Literal["von_neumann", "moore"]


def neighbor_offsets(kind: Neighborhood) -> List[Tuple[int, int]]:
    """
    Returns neighbor offsets (dy, dx) excluding (0,0).
    """
    if kind == "von_neumann":
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if kind == "moore":
        return [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    raise ValueError(f"Unknown neighborhood: {kind}")


def in_bounds(y: int, x: int, h: int, w: int) -> bool:
    return 0 <= y < h and 0 <= x < w


def neighbors(y: int, x: int, h: int, w: int, kind: Neighborhood) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for dy, dx in neighbor_offsets(kind):
        yy, xx = y + dy, x + dx
        if in_bounds(yy, xx, h, w):
            out.append((yy, xx))
    return out

