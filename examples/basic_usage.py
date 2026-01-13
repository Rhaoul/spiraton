import torch

from spiraton import SpiratonCell, GatedSpiratonCell


def main() -> None:
    torch.manual_seed(0)

    input_size = 8
    x_single = torch.randn(input_size)
    x_batch = torch.randn(4, input_size)

    core = SpiratonCell(input_size=input_size)
    gated = GatedSpiratonCell(input_size=input_size)

    print("=== Single vector ===")
    print("x_single:", x_single)
    print("core(x_single):", core(x_single).item())
    print("gated(x_single):", gated(x_single).item())

    print("\n=== Batch ===")
    print("x_batch:\n", x_batch)
    print("core(x_batch):", core(x_batch))
    print("gated(x_batch):", gated(x_batch))

    print("\n=== Adaptation demo (gated) ===")
    err1 = torch.tensor([1.0, 0.5, 0.25])
    err2 = torch.tensor([1.5, 1.0, 0.75])

    a1 = gated.second_order_adjust(err1)
    a2 = gated.second_order_adjust(err2)

    print("adaptation after err1:", a1)
    print("adaptation after err2:", a2)


if __name__ == "__main__":
    main()
