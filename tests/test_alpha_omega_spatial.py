import torch

from spiraton.core.cell import SpiratonCell, MatrixSpiratonCell
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


def test_double_dynamic_asymmetry() -> None:
    torch.manual_seed(0)
    B, H, W, C = 2, 5, 5, 4

    cell = MatrixSpiratonCell(input_size=2 * C)
    grid_out = SpiralGrid(cell, channels=C, spiral="outward", cell_out_size=2 * C)
    grid_in = SpiralGrid(cell, channels=C, spiral="inward", cell_out_size=2 * C)

    x0 = torch.randn(B, H, W, C)
    
    # L circ D (outward then inward)
    x_d = grid_out(x0, steps=2)
    x_ld = grid_in(x_d, steps=2)

    # D circ L (inward then outward)
    x_l = grid_in(x0, steps=2)
    x_dl = grid_out(x_l, steps=2)

    diff = torch.norm(x_ld - x_dl)
    assert float(diff.item()) > 1e-4, "Double dynamic should be asymmetric!"
