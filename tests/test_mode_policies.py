import torch
import pytest

from spiraton.core.mode_policy import MeanThreshold, EnergyThreshold, LearnedGateMode
from spiraton.core.cell import SpiratonCell
from spiraton.experimental.gated_cell import GatedSpiratonCell


def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def test_mean_threshold_hard_mask() -> None:
    pol = MeanThreshold(0.0)
    x = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    m = pol(x)
    assert m.dtype == torch.bool
    assert m.tolist() == [True, False]


def test_energy_threshold_hard_mask() -> None:
    pol = EnergyThreshold(threshold=1.0)
    x = torch.tensor([[0.1, 0.1], [2.0, 0.0]])  # rms ~0.1 and ~1.414
    m = pol(x)
    assert m.dtype == torch.bool
    assert m.tolist() == [False, True]


def test_learned_gate_mode_soft_in_range_and_grad() -> None:
    torch.manual_seed(0)
    gate = LearnedGateMode(input_size=4)
    x = torch.randn(5, 4, requires_grad=True)
    g = gate(x)

    assert g.dtype != torch.bool
    assert g.shape == (5,)
    assert float(g.min().item()) >= 0.0
    assert float(g.max().item()) <= 1.0

    # grad flows
    loss = g.sum()
    loss.backward()
    assert gate.w.grad is not None
    assert gate.b.grad is not None
    assert float(gate.w.grad.abs().sum().item()) > 0.0


@pytest.mark.parametrize("CellCls", [SpiratonCell, GatedSpiratonCell])
def test_soft_mode_mixing_matches_hard_extremes(CellCls) -> None:
    """
    If the soft gate is forced to 1 -> should behave like hard dextro for that sample.
    If forced to 0 -> should behave like hard levogyre for that sample.
    We simulate this with a dummy policy returning constant gates.
    """
    class GateOnes:
        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.ones(inputs.size(0), device=inputs.device, dtype=torch.float32)

    class GateZeros:
        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(inputs.size(0), device=inputs.device, dtype=torch.float32)

    torch.manual_seed(0)
    input_size = 6

    cell_soft_1 = CellCls(input_size=input_size, mode_policy=GateOnes())
    cell_soft_0 = CellCls(input_size=input_size, mode_policy=GateZeros())
    cell_hard = CellCls(input_size=input_size)  # default hard mean>=0

    # Force deterministic params by copying from hard cell
    with torch.no_grad():
        for name, p in cell_hard.named_parameters():
            if name in dict(cell_soft_1.named_parameters()):
                dict(cell_soft_1.named_parameters())[name].copy_(p)
                dict(cell_soft_0.named_parameters())[name].copy_(p)

    # Two samples: one positive mean, one negative mean
    x = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],      # hard dextro
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # hard levo
    ])

    y_hard = cell_hard(x)
    y_soft_1 = cell_soft_1(x)
    y_soft_0 = cell_soft_0(x)

    assert _finite(y_hard)
    assert _finite(y_soft_1)
    assert _finite(y_soft_0)

    # Gate=1 should match "always dextro" behavior, so it should equal hard output on sample 0,
    # but not necessarily on sample 1 (because hard sample 1 is levo).
    assert torch.allclose(y_soft_1[0], y_hard[0], atol=1e-6, rtol=1e-6)

    # Gate=0 should match "always levo", so it should equal hard output on sample 1.
    assert torch.allclose(y_soft_0[1], y_hard[1], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("CellCls", [SpiratonCell, GatedSpiratonCell])
def test_cells_accept_mode_policy(CellCls) -> None:
    torch.manual_seed(0)
    input_size = 8
    pol = MeanThreshold(0.0)
    cell = CellCls(input_size=input_size, mode_policy=pol)

    x = torch.randn(4, input_size)
    y = cell(x)
    assert y.shape == (4,)
    assert _finite(y)

