import torch

from spiraton.core.cell import SpiratonCell
from spiraton.experimental.gated_cell import GatedSpiratonCell
from spiraton.recursion import RecursiveSpiraton


def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def test_recursive_shapes_and_finiteness_core_concat() -> None:
    torch.manual_seed(0)
    x_size, state_size = 5, 3
    cell = SpiratonCell(input_size=x_size + state_size)
    rec = RecursiveSpiraton(cell, x_size=x_size, state_size=state_size, combine="concat", steps_default=3)

    x = torch.randn(4, x_size)
    y = rec(x)
    assert y.shape == (4,)
    assert _finite(y)


def test_recursive_return_trace() -> None:
    torch.manual_seed(0)
    x_size, state_size = 4, 4
    cell = SpiratonCell(input_size=x_size + state_size)
    rec = RecursiveSpiraton(cell, x_size=x_size, state_size=state_size)

    x = torch.randn(2, x_size)
    y, st, tr = rec(x, steps=5, return_trace=True)

    assert y.shape == (2,)
    assert st.shape == (2, state_size)
    assert tr["steps"] == 5
    assert len(tr["states"]) == 5
    assert len(tr["outputs"]) == 5
    assert _finite(y)
    assert _finite(st)


def test_recursive_grad_flow() -> None:
    torch.manual_seed(0)
    x_size, state_size = 6, 2
    cell = SpiratonCell(input_size=x_size + state_size)
    rec = RecursiveSpiraton(cell, x_size=x_size, state_size=state_size, steps_default=2)

    x = torch.randn(3, x_size, requires_grad=True)
    y = rec(x)
    y.sum().backward()

    assert x.grad is not None
    assert float(x.grad.abs().sum().item()) > 0.0
    assert any(p.grad is not None for p in cell.parameters())


def test_recursive_update_modes() -> None:
    torch.manual_seed(0)
    x_size, state_size = 3, 3
    cell = SpiratonCell(input_size=x_size + state_size)
    x = torch.randn(2, x_size)

    for update in ["residual", "gated", "momentum"]:
        rec = RecursiveSpiraton(cell, x_size=x_size, state_size=state_size, update=update, steps_default=3)
        y, st, tr = rec(x, return_trace=True)
        assert _finite(y)
        assert _finite(st)
        assert tr["steps"] == 3
        assert tr["update"] == update


def test_recursive_works_with_gated_cell() -> None:
    torch.manual_seed(0)
    x_size, state_size = 4, 2
    cell = GatedSpiratonCell(input_size=x_size + state_size)
    rec = RecursiveSpiraton(cell, x_size=x_size, state_size=state_size, steps_default=3)

    x = torch.randn(5, x_size)
    y = rec(x)
    assert y.shape == (5,)
    assert _finite(y)

