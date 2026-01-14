import torch

from spiraton.core.cell import SpiratonCell
from spiraton.recursion import RecursiveSpiraton


def main() -> None:
    torch.manual_seed(0)

    x_size = 6
    state_size = 4

    # concat(state, x) => input_size = state_size + x_size
    cell = SpiratonCell(input_size=state_size + x_size)

    rec = RecursiveSpiraton(
        cell,
        x_size=x_size,
        state_size=state_size,
        combine="concat",
        update="gated",
        steps_default=5,
    )

    x = torch.randn(3, x_size)
    y, st, tr = rec(x, return_trace=True)

    print("y:", y)
    print("state:", st)
    print("steps:", tr["steps"])
    print("last output:", tr["outputs"][-1])


if __name__ == "__main__":
    main()

