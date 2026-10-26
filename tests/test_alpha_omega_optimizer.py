"""Tests ``AlphaOmegaOpt`` (experimental/alpha_omega_optimizer.py).

Porte 0 (organe ``regulate_step`` importé), validation, pivot AdamW bit-à-bit,
pivot η=0, pivot repos, formule exacte du retour (ω, ω̄, log g), signe de la
rétroaction (pas trop court ⇒ g monte, pas trop long ⇒ g descend), bornes du gain,
groupes ajoutés après coup, reprise ``state_dict`` au bit, déterminisme float64,
fenêtre de lr élargie sur une quadratique mal conditionnée.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Dict

import pytest
import torch

from spiraton.experimental.chrono import ChronoSpiraton
from spiraton.experimental.edge_controller import regulate_step as canon_regulate_step
from spiraton.experimental import alpha_omega_optimizer
from spiraton.experimental.alpha_omega_optimizer import AlphaOmegaOpt
from spiraton.training.aba_loss import alpha_omega_loss


def _seq(n: int = 40, shape=(7, 5), seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g, dtype=torch.float64)
    return w, [torch.randn(*shape, generator=g, dtype=torch.float64) for _ in range(n)]


def _hash(state: Dict[str, torch.Tensor]) -> str:
    blob = b"".join(state[k].detach().cpu().numpy().astype("<f8").tobytes() for k in sorted(state))
    return hashlib.sha256(blob).hexdigest()


def _quadratique(lr: float, steps: int, make) -> float:
    """f(x) = ½ xᵀAx, A diagonale de conditionnement 1e2, départ x = 1."""
    a = torch.logspace(0, 2, 10, dtype=torch.float64)
    x = torch.nn.Parameter(torch.ones(10, dtype=torch.float64))
    opt = make([x], lr)
    for _ in range(steps):
        opt.zero_grad()
        (0.5 * (a * x * x).sum()).backward()
        opt.step()
    return float(0.5 * (a * x.detach() ** 2).sum())


def test_gate0_regulate_step_is_edge_controller_organ() -> None:
    assert alpha_omega_optimizer.regulate_step is canon_regulate_step


def test_argument_validation() -> None:
    p = [torch.nn.Parameter(torch.ones(3, dtype=torch.float64))]
    with pytest.raises(ValueError, match="lr"):
        AlphaOmegaOpt(p, lr=0.0)
    with pytest.raises(ValueError, match="betas"):
        AlphaOmegaOpt(p, betas=(1.0, 0.999))
    with pytest.raises(ValueError, match="weight_decay"):
        AlphaOmegaOpt(p, weight_decay=-1.0)
    with pytest.raises(ValueError, match="cible"):
        AlphaOmegaOpt(p, cible=1.0)
    with pytest.raises(ValueError, match="eta"):
        AlphaOmegaOpt(p, eta=-0.1)
    with pytest.raises(ValueError, match="beta_q"):
        AlphaOmegaOpt(p, beta_q=1.0)
    with pytest.raises(ValueError, match="g_min"):
        AlphaOmegaOpt(p, g0=1e-5, g_min=1e-4)


@pytest.mark.parametrize("wd", [0.0, 1e-2])
def test_pivot_adamw_bit_for_bit(wd: float) -> None:
    """η=0, g0=1 : trajectoire identique AU BIT à torch.optim.AdamW."""
    w, grads = _seq()
    a, b = torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.clone())
    oa = torch.optim.AdamW([a], lr=3e-3, weight_decay=wd)
    ob = AlphaOmegaOpt([b], lr=3e-3, weight_decay=wd, eta=0.0, g0=1.0)
    for g in grads:
        a.grad, b.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
        assert torch.equal(a.data, b.data)
    assert ob.param_groups[0]["n_regul"] == len(grads) - 1  # le retour est lu, pas appliqué


def test_pivot_eta_zero_keeps_gain() -> None:
    w, grads = _seq()
    p = torch.nn.Parameter(w.clone())
    opt = AlphaOmegaOpt([p], lr=1e-2, g0=0.3, eta=0.0)
    for g in grads:
        p.grad = g.clone()
        opt.step()
        assert opt.param_groups[0]["g"] == 0.3


def test_pivot_zero_gradient_from_rest() -> None:
    p0 = torch.tensor([0.25, -0.5, 1.0, -2.0], dtype=torch.float64)
    p = torch.nn.Parameter(p0.clone())
    opt = AlphaOmegaOpt([p], lr=1e-2, weight_decay=0.0)
    for _ in range(9):
        p.grad = torch.zeros_like(p)
        opt.step()
    assert torch.equal(p.data, p0)
    assert opt.param_groups[0]["g"] == 1e-2
    assert opt.param_groups[0]["n_regul"] == 0  # ‖∇‖ = 0 : rien à comparer


def test_return_formula_exact() -> None:
    """Pas 2 : ω = cos(−∇L(θ_1), Δθ_0), ω̄ = (1−β_q)ω, log g = log g0 + η(ω̄ − cible)."""
    w, grads = _seq(n=2)
    p = torch.nn.Parameter(w.clone())
    opt = AlphaOmegaOpt([p], lr=1e-2, weight_decay=1e-2, eta=0.2, beta_q=0.7, g0=0.5, cible=0.1)
    p.grad = grads[0].clone()
    before = p.detach().clone()
    opt.step()
    delta0 = p.detach() - before
    assert torch.equal(opt.state[p]["delta"], delta0)
    p.grad = grads[1].clone()
    opt.step()
    omega = float(-(grads[1] * delta0).sum() / (grads[1].norm() * delta0.norm()))
    omega_bar = (1 - 0.7) * omega
    g_attendu = math.exp(min(math.log(1e2), max(math.log(1e-4), math.log(0.5) + 0.2 * (omega_bar - 0.1))))
    grp = opt.param_groups[0]
    assert math.isclose(grp["omega_last"], omega, rel_tol=1e-12)
    assert math.isclose(grp["omega_bar"], omega_bar, rel_tol=1e-12)
    assert math.isclose(grp["g"], g_attendu, rel_tol=1e-12)


def test_feedback_sign_short_step_grows_long_step_shrinks() -> None:
    """Pas trop court (lr=1e-5) : le retour reste aligné ⇒ g monte.
    Pas trop long (SGD-like sur f = 50x², lr=1) : le retour s'oppose ⇒ g descend."""
    x = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    opt = AlphaOmegaOpt([x], lr=1e-5, weight_decay=0.0, g0=1.0)
    for _ in range(30):
        opt.zero_grad()
        (0.5 * (x * x).sum()).backward()
        opt.step()
    assert opt.param_groups[0]["omega_bar"] > 0.5
    assert opt.param_groups[0]["g"] > 1.0

    y = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    opt = AlphaOmegaOpt([y], lr=1.0, weight_decay=0.0, g0=1.0)
    for _ in range(30):
        opt.zero_grad()
        (50.0 * (y * y).sum()).backward()
        opt.step()
    assert opt.param_groups[0]["omega_bar"] < 0.0
    assert opt.param_groups[0]["g"] < 1.0


def test_gain_bounded() -> None:
    x = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    opt = AlphaOmegaOpt([x], lr=1e-9, weight_decay=0.0, eta=1.0, g0=1.0, g_max=3.0)
    hist = []
    for _ in range(80):
        opt.zero_grad()
        (0.5 * (x * x).sum()).backward()
        opt.step()
        hist.append(opt.param_groups[0]["g"])
    assert max(hist) == 3.0  # atteint, jamais dépassé
    assert all(1e-4 <= h <= 3.0 for h in hist)


def test_added_param_group_has_return_state() -> None:
    a = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    b = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    opt = AlphaOmegaOpt([a], lr=1e-2)
    opt.add_param_group({"params": [b], "lr": 1e-3})
    assert opt.param_groups[1]["g"] == 1e-2 and opt.param_groups[1]["n_regul"] == 0
    for _ in range(3):
        a.grad, b.grad = torch.ones_like(a), torch.ones_like(b)
        opt.step()
    assert opt.param_groups[1]["n_regul"] == 2


def test_state_dict_resume_bit_for_bit() -> None:
    w, grads = _seq(n=23)
    p_ref = torch.nn.Parameter(w.clone())
    o_ref = AlphaOmegaOpt([p_ref], lr=1e-2)
    for g in grads:
        p_ref.grad = g.clone()
        o_ref.step()

    p1 = torch.nn.Parameter(w.clone())
    o1 = AlphaOmegaOpt([p1], lr=1e-2)
    for g in grads[:10]:
        p1.grad = g.clone()
        o1.step()
    saved_p, saved_o = p1.detach().clone(), copy.deepcopy(o1.state_dict())
    p2 = torch.nn.Parameter(saved_p)
    o2 = AlphaOmegaOpt([p2], lr=1e-2)
    o2.load_state_dict(saved_o)
    for g in grads[10:]:
        p2.grad = g.clone()
        o2.step()
    assert torch.equal(p2.data, p_ref.data)
    assert o2.param_groups[0]["g"] == o_ref.param_groups[0]["g"]


def test_lr_window_wider_than_adamw_on_ill_conditioned_quadratic() -> None:
    """lr=1e-3 est trop court pour AdamW en 500 pas ; le retour allonge le pas."""
    def adamw(p, lr):
        return torch.optim.AdamW(p, lr=lr, weight_decay=0.0)

    def ao(p, lr):
        return AlphaOmegaOpt(p, lr=lr, weight_decay=0.0)

    assert _quadratique(1e-3, 500, adamw) > 1.0
    assert _quadratique(1e-3, 500, ao) < 1e-6
    for lr in (1e-2, 1e-1, 1.0):
        assert _quadratique(lr, 500, ao) < 1e-6


def test_determinism_float64_on_chrono_spiraton() -> None:
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        def _run() -> str:
            torch.manual_seed(64)
            model = ChronoSpiraton(state_size=19, init_scale=0.1, bounded=False).double()
            a = torch.randn(32, 19, generator=torch.Generator().manual_seed(1064), dtype=torch.float64)
            opt = AlphaOmegaOpt(model.parameters(), lr=1e-2)
            for _ in range(10):
                opt.zero_grad()
                alpha_omega_loss(model(a, steps=2), a, target_dist=0.3).loss.backward()
                opt.step()
            return _hash(model.state_dict())

        assert _run() == _run()
    finally:
        torch.use_deterministic_algorithms(prev)
