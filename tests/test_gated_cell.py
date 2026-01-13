import torch
import pytest

from spiraton.experimental.gated_cell import GatedSpiratonCell
from spiraton.core.operators import additive, subtractive, multiplicative, divisive
from spiraton.core.modes import dextro_mask


def _no_nan_inf(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("input_size,batch", [(8, 1), (8, 4), (16, 7)])
def test_gated_shapes_and_finiteness(input_size: int, batch: int) -> None:
    torch.manual_seed(0)
    cell = GatedSpiratonCell(input_size=input_size)

    x = torch.randn(batch, input_size)
    y = cell(x)

    assert y.shape == (batch,)
    assert _no_nan_inf(y)

    xs = torch.randn(input_size)
    ys = cell(xs)
    assert ys.shape == ()
    assert _no_nan_inf(ys)


def test_gated_matches_manual_formula_when_params_forced() -> None:
    """
    Force parameters to easy constants and check exact forward formula.

    Gated:
      gate_mul = sigmoid(sum(x*w_gate_mul) + b_gate_mul)
      mul = tanh(log_mul) * gate_mul
      gate_div = sigmoid(sum(x*w_gate_div) + b_gate_div)
      div = tanh(log_div) * gate_div
      c_* = sigmoid(coeffs)
      raw_dextro = c_add*add + c_mul*mul - c_div*div
      raw_levogyre = c_sub*sub + c_div*div - c_mul*mul
      out = tanh(z) if dextro else atan(z)
    """
    torch.manual_seed(0)
    input_size = 4
    eps = 1e-6
    cell = GatedSpiratonCell(input_size=input_size, eps=eps)

    with torch.no_grad():
        # operator weights
        cell.w_add.fill_(1.0)
        cell.w_sub.fill_(1.0)
        cell.w_mul.fill_(1.0)
        cell.w_div.fill_(1.0)

        # gates: keep w_gate at zero => sum=0; b=0 => sigmoid(0)=0.5
        cell.w_gate_mul.zero_()
        cell.b_gate_mul.fill_(0.0)
        cell.w_gate_div.zero_()
        cell.b_gate_div.fill_(0.0)

        # coeffs: 0 => sigmoid(0)=0.5
        cell.coeffs.zero_()

        cell.bias.fill_(0.0)

    x = torch.tensor([
        [2.0, 2.0, 2.0, 2.0],          # dextro
        [-2.0, -2.0, -2.0, -2.0],      # levo
    ])

    m = dextro_mask(x)

    add = additive(x, cell.w_add)
    sub = subtractive(x, cell.w_sub)
    log_mul = multiplicative(x, cell.w_mul, eps=eps)
    log_div = divisive(x, cell.w_div, eps=eps)

    gate_mul = torch.sigmoid(torch.sum(x * cell.w_gate_mul, dim=-1) + cell.b_gate_mul)  # 0.5
    gate_div = torch.sigmoid(torch.sum(x * cell.w_gate_div, dim=-1) + cell.b_gate_div)  # 0.5

    mul = torch.tanh(log_mul) * gate_mul
    div = torch.tanh(log_div) * gate_div

    c_add, c_sub, c_mul, c_div = torch.sigmoid(cell.coeffs)

    raw_dextro = c_add * add + c_mul * mul - c_div * div
    raw_levogyre = c_sub * sub + c_div * div - c_mul * mul
    raw = torch.where(m, raw_dextro, raw_levogyre)

    z = raw + cell.bias
    expected = torch.where(m, torch.tanh(z), torch.atan(z))

    y = cell(x)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_gated_second_order_adjust_updates_buffer() -> None:
    torch.manual_seed(0)
    cell = GatedSpiratonCell(input_size=6)

    with torch.no_grad():
        cell.adaptation.fill_(0.1)

    e1 = torch.tensor([1.0, 1.0, 1.0])
    a1 = cell.second_order_adjust(e1)
    assert 0.001 <= a1 <= 0.2

    # bigger error -> should go down (typically)
    e2 = torch.tensor([2.0, 2.0, 2.0])
    a2 = cell.second_order_adjust(e2)
    assert 0.001 <= a2 <= 0.2
    assert a2 <= a1 + 1e-12  # allow equality if edge-case


def test_gated_gradients_flow() -> None:
    torch.manual_seed(0)
    input_size = 10
    cell = GatedSpiratonCell(input_size=input_size)

    x = torch.randn(8, input_size, requires_grad=True)
    y = cell(x).sum()
    y.backward()

    # essential grads
    assert cell.w_add.grad is not None
    assert cell.w_mul.grad is not None
    assert cell.coeffs.grad is not None

    total = (
        cell.w_add.grad.abs().sum()
        + cell.w_sub.grad.abs().sum()
        + cell.w_mul.grad.abs().sum()
        + cell.w_div.grad.abs().sum()
        + cell.coeffs.grad.abs().sum()
    )
    assert float(total.item()) > 0.0
