import torch

from spiraton.core.cell import SpiratonCell
from spiraton.grid import SpiralGrid
from spiraton.diagnostics import run_alpha_omega_spatial


def main() -> None:
    torch.manual_seed(0)
    B, H, W, C = 1, 7, 7, 6

    cell = SpiratonCell(input_size=2 * C)
    grid = SpiralGrid(cell, channels=C, neighborhood="moore", aggregator="mean", spiral="outward")

    x0 = torch.randn(B, H, W, C)
    rep = run_alpha_omega_spatial(grid, x0, steps=12)

    print("best_return_step:", rep.best_return_step)
    print("score:", rep.score)
    print("l2_norm:", [round(float(v), 6) for v in rep.l2_norm.tolist()])
    print("cosine:", [round(float(v), 6) for v in rep.cosine.tolist()])


if __name__ == "__main__":
    main()

