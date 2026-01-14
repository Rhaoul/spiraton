import torch

from spiraton.core.cell import SpiratonCell
from spiraton.grid import SpiralGrid


def main() -> None:
    torch.manual_seed(0)

    B, H, W, C = 1, 5, 5, 6

    cell = SpiratonCell(input_size=2 * C)
    grid = SpiralGrid(
        cell,
        channels=C,
        neighborhood="moore",
        aggregator="mean",
        spiral="outward",
        steps_default=1,
    )

    x = torch.randn(B, H, W, C)
    y = grid(x, steps=2)

    print("input:", x.shape)
    print("output:", y.shape)
    print("sample center vec:", y[0, H // 2, W // 2, :])


if __name__ == "__main__":
    main()

