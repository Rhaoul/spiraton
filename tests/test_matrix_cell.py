import torch
import pytest

from spiraton.experimental.matrix_cell import (
    MatrixSpiratonCell,
    commutator_norm,
    OPS,
)
from spiraton.core.modes import dextro_mask


def _no_nan_inf(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


# --- Quatuor de tests (cf. CLAUDE.md) --------------------------------------


@pytest.mark.parametrize("input_size,batch", [(8, 1), (8, 4), (16, 7)])
def test_matrix_shapes_and_finiteness(input_size: int, batch: int) -> None:
    torch.manual_seed(0)
    cell = MatrixSpiratonCell(input_size=input_size)

    x = torch.randn(batch, input_size)
    y = cell(x)
    # endomorphie de l'espace d'états : sortie de même dimension que l'entrée
    assert y.shape == (batch, input_size)
    assert _no_nan_inf(y)

    # chemin singleton
    xs = torch.randn(input_size)
    ys = cell(xs)
    assert ys.shape == (input_size,)
    assert _no_nan_inf(ys)


def test_matrix_mode_rule_matches_mask() -> None:
    torch.manual_seed(0)
    input_size = 6
    cell = MatrixSpiratonCell(input_size=input_size)

    x = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],       # mean > 0 => dextro True
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # mean < 0 => dextro False
    ])
    m = dextro_mask(x)
    assert m.dtype == torch.bool
    assert m.tolist() == [True, False]

    y = cell(x)
    assert y.shape == (2, input_size)
    assert _no_nan_inf(y)

    # La ligne dextro doit suivre la branche tanh(compose(dextro_order)),
    # la ligne levo la branche atan(compose(levo_order)).
    raw_d = cell.compose(x, cell.cfg.dextro_order) + cell.bias
    raw_l = cell.compose(x, cell.cfg.levo_order) + cell.bias
    expected = torch.where(m.unsqueeze(-1), torch.tanh(raw_d), torch.atan(raw_l))
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_matrix_matches_manual_formula_when_params_forced() -> None:
    """Force les matrices et vérifie exactement la composition + l'activation."""
    torch.manual_seed(0)
    d = 3
    cell = MatrixSpiratonCell(
        input_size=d,
        dextro_order=("add", "mul"),  # ordre court pour une vérif lisible
    )
    # levo_order = renversement => ("mul", "add")
    assert cell.cfg.levo_order == ("mul", "add")

    with torch.no_grad():
        # Deux matrices simples qui NE commutent PAS.
        Wa = torch.tensor([[1.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        Wm = torch.tensor([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 1.0, 1.0]])
        cell.W_add.copy_(Wa)
        cell.W_mul.copy_(Wm)
        cell.W_sub.copy_(torch.eye(d))
        cell.W_div.copy_(torch.eye(d))
        cell.bias.zero_()

    x = torch.tensor([
        [2.0, 2.0, 2.0],      # mean > 0 => dextro
        [-2.0, -2.0, -2.0],   # mean < 0 => levo
    ])
    m = dextro_mask(x)

    # compose dextro = add puis mul : Wm @ (Wa @ x)
    raw_d = (x @ Wa.t()) @ Wm.t()
    # compose levo = mul puis add : Wa @ (Wm @ x)
    raw_l = (x @ Wm.t()) @ Wa.t()
    expected = torch.where(m.unsqueeze(-1), torch.tanh(raw_d), torch.atan(raw_l))

    y = cell(x)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_matrix_gradients_flow() -> None:
    torch.manual_seed(0)
    d = 10
    cell = MatrixSpiratonCell(input_size=d)

    x = torch.randn(8, d, requires_grad=True)
    y = cell(x).sum()
    y.backward()

    for name in OPS:
        g = getattr(cell, f"W_{name}").grad
        assert g is not None, f"pas de gradient sur W_{name}"
    assert cell.bias.grad is not None

    total = sum(getattr(cell, f"W_{name}").grad.abs().sum() for name in OPS)
    assert float(total.item()) > 0.0


# --- Test spécifique chantier 1 : l'ordre de composition change la sortie ----


def test_composition_order_changes_output() -> None:
    """Preuve de non-commutativité : permuter l'ordre change le résultat."""
    torch.manual_seed(1)
    d = 5
    cell = MatrixSpiratonCell(input_size=d, init_scale=0.5)  # init plus bruité

    x = torch.randn(4, d)
    forward = cell.compose(x, ("add", "mul"))
    backward = cell.compose(x, ("mul", "add"))
    # Avec des matrices génériques non commutantes, les deux diffèrent nettement.
    assert not torch.allclose(forward, backward, atol=1e-4)

    # Et la sortie complète dextro != levo (branches miroir d'ordre inverse).
    raw_d = cell.compose(x, cell.cfg.dextro_order)
    raw_l = cell.compose(x, cell.cfg.levo_order)
    assert not torch.allclose(raw_d, raw_l, atol=1e-4)


def test_commutator_norm_properties() -> None:
    torch.manual_seed(2)
    d = 4
    I = torch.eye(d)
    A = torch.randn(d, d)
    B = torch.randn(d, d)

    # L'identité commute avec tout => commutateur nul.
    assert float(commutator_norm(I, A)) == pytest.approx(0.0, abs=1e-6)
    # Antisymétrie en norme : ‖[A,B]‖ == ‖[B,A]‖.
    assert float(commutator_norm(A, B)) == pytest.approx(float(commutator_norm(B, A)), abs=1e-6)
    # Une matrice commute avec elle-même.
    assert float(commutator_norm(A, A)) == pytest.approx(0.0, abs=1e-6)


def test_commutators_dict_and_total() -> None:
    torch.manual_seed(3)
    cell = MatrixSpiratonCell(input_size=6, init_scale=0.3)
    comms = {k: v.detach() for k, v in cell.commutators().items()}
    # 4 opérateurs => C(4,2) = 6 paires.
    assert len(comms) == 6
    for v in comms.values():
        assert torch.isfinite(v).all()
        assert float(v) >= 0.0
    total = cell.total_noncommutativity().detach()
    assert float(total) == pytest.approx(sum(float(v) for v in comms.values()), rel=1e-5)
    # Avec init bruité, la non-commutativité est strictement positive.
    assert float(total) > 0.0


def test_commutators_are_differentiable() -> None:
    """Le diagnostic est différentiable : utilisable comme terme de perte."""
    torch.manual_seed(4)
    cell = MatrixSpiratonCell(input_size=5, init_scale=0.3)
    loss = cell.total_noncommutativity()
    loss.backward()
    # Le gradient remonte vers au moins une matrice d'opérateur.
    grads = [getattr(cell, f"W_{name}").grad for name in OPS]
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


def test_invalid_order_raises() -> None:
    cell = MatrixSpiratonCell(input_size=4)
    with pytest.raises(ValueError):
        cell.compose(torch.randn(4), ("add", "bogus"))
    with pytest.raises(ValueError):
        MatrixSpiratonCell(input_size=4, dextro_order=())
