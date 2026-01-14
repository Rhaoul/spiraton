import torch

from spiraton.core.cell import SpiratonCell
from spiraton.grid import SpiralGrid
from spiraton.diagnostics import run_alpha_omega_spatial


def test_alpha_omega_runs_and_shapes() -> None:
    torch.manual_seed(0)
    B, H, W, C = 2, 5, 5, 4

    cell = SpiratonCell(input_size=2 * C)
    grid = SpiralGrid(cell, channels=C, neighborhood="von_neumann", aggregator="mean", spiral="outward")

    x0 = torch.randn(B, H, W, C)
    rep = run_alpha_omega_spatial(grid, x0, steps=6)

    assert rep.l2_norm.shape == (7,)
    assert rep.cosine.shape == (7,)
    assert 0 <= rep.best_return_step <= 6
    assert torch.isfinite(rep.l2_norm).all().item()
    assert torch.isfinite(rep.cosine).all().item()

