import torch

from spiraton.core.cell import SpiratonCell
from spiraton.grid import SpiralGrid, spiral_indices


def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def test_spiral_indices_cover_all_unique() -> None:
    h, w = 5, 4
    seq = spiral_indices(h, w, order="outward")
    assert len(seq) == h * w
    assert len(set(seq)) == h * w

    seq2 = spiral_indices(h, w, order="inward")
    assert len(seq2) == h * w
    assert len(set(seq2)) == h * w
    assert seq2[0] == seq[-1]


def test_spiral_grid_shapes_and_finiteness() -> None:
    torch.manual_seed(0)
    B, H, W, C = 2, 3, 4, 5

    # SpiralGrid concatenates local+neigh => 2C input to cell
    cell = SpiratonCell(input_size=2 * C)
    grid = SpiralGrid(cell, channels=C, neighborhood="von_neumann", aggregator="mean", spiral="outward")

    x = torch.randn(B, H, W, C)
    y = grid(x, steps=2)

    assert y.shape == (B, H, W, C)
    assert _finite(y)


def test_spiral_grid_deterministic_given_seed() -> None:
    torch.manual_seed(0)
    B, H, W, C = 1, 3, 3, 4

    cell = SpiratonCell(input_size=2 * C)
    grid = SpiralGrid(cell, channels=C)

    x = torch.randn(B, H, W, C)
    y1 = grid(x, steps=1)
    y2 = grid(x, steps=1)

    assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)

