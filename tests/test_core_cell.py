import torch
import pytest

from spiraton.core.cell import SpiratonCell
from spiraton.core.operators import additive, subtractive, multiplicative, divisive
from spiraton.core.modes import dextro_mask


def _no_nan_inf(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("input_size,batch", [(8, 1), (8, 4), (16, 7)])
def test_core_shapes_and_finiteness(input_size: int, batch: int) -> None:
    torch.manual_seed(0)
    cell = SpiratonCell(input_size=input_size)

    x = torch.randn(batch, input_size)
    y = cell(x)

    assert y.shape == (batch,)
    assert _no_nan_inf(y)

    # singleton path
    xs = torch.randn(input_size)
    ys = cell(xs)
    assert ys.shape == ()
    assert _no_nan_inf(ys)


def test_core_mode_rule_matches_mask() -> None:
    torch.manual_seed(0)
    input_size = 6
    cell = SpiratonCell(input_size=input_size)

    # Construct two samples: one strictly positive mean, one strictly negative mean
    x = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],      # mean > 0 => dextro True
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0] # mean < 0 => dextro False
    ])
    m = dextro_mask(x)
    assert m.dtype == torch.bool
    assert m.tolist() == [True, False]

    y = cell(x)
    assert y.shape == (2,)
    assert _no_nan_inf(y)


def test_core_matches_manual_formula_when_params_forced() -> None:
    """
    Force parameters to simple constants and check the forward formula exactly.
    Core definition:
      add = sum(inputs * w_add)
      sub = sum(inputs - w_sub)
      log_mul = sum(w_mul * log(|x|+eps))
      log_div = -sum(w_div * log(|x|+eps))
      mul = tanh(log_mul)
      div = tanh(log_div)
      raw_dextro = add + mul - div
      raw_levogyre = sub + div - mul
      z = raw + bias
      out = tanh(z) if dextro else atan(z)
    """
    torch.manual_seed(0)
    input_size = 5
    eps = 1e-6
    cell = SpiratonCell(input_size=input_size, eps=eps)

    with torch.no_grad():
        cell.w_add.fill_(1.0)
        cell.w_sub.fill_(1.0)
        cell.w_mul.fill_(1.0)
        cell.w_div.fill_(1.0)
        cell.bias.fill_(0.0)

    x = torch.tensor([
        [2.0, 2.0, 2.0, 2.0, 2.0],         # mean > 0 => dextro
        [-2.0, -2.0, -2.0, -2.0, -2.0],    # mean < 0 => levo
    ])

    m = dextro_mask(x)

    add = additive(x, cell.w_add)
    sub = subtractive(x, cell.w_sub)
    log_mul = multiplicative(x, cell.w_mul, eps=eps)
    log_div = divisive(x, cell.w_div, eps=eps)

    mul = torch.tanh(log_mul)
    div = torch.tanh(log_div)

    raw_dextro = add + mul - div
    raw_levogyre = sub + div - mul
    raw = torch.where(m, raw_dextro, raw_levogyre)
    z = raw + cell.bias
    expected = torch.where(m, torch.tanh(z), torch.atan(z))

    y = cell(x)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_core_gradients_flow() -> None:
    torch.manual_seed(0)
    input_size = 10
    cell = SpiratonCell(input_size=input_size)

    x = torch.randn(8, input_size, requires_grad=True)
    y = cell(x).sum()
    y.backward()

    # Gradients should exist
    assert cell.w_add.grad is not None
    assert cell.w_sub.grad is not None
    assert cell.w_mul.grad is not None
    assert cell.w_div.grad is not None

    # And at least one should be non-zero in normal conditions
    total = (
        cell.w_add.grad.abs().sum()
        + cell.w_sub.grad.abs().sum()
        + cell.w_mul.grad.abs().sum()
        + cell.w_div.grad.abs().sum()
    )
    assert float(total.item()) > 0.0
