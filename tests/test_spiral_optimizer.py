"""Tests ``SpiralClosureOpt`` (experimental/spiral_optimizer.py).

Porte 0 (organe ``regulate_step`` importé), pivot AdamW bit-à-bit (le zéro de la
mesure), pivot η=0, pivot repos, formule exacte de la clôture A′, borne du pas sous
pic de gradient, actionneur ρ̂ vivant et borné, reprise ``state_dict`` au bit,
déterminisme float64, stabilité sur ``ChronoSpiraton(19, init_scale=0.6)`` sans clip.
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
from spiraton.experimental import spiral_optimizer
from spiraton.experimental.spiral_optimizer import SpiralClosureOpt
from spiraton.training.aba_loss import alpha_omega_loss


def _seq(n: int = 40, shape=(7, 5), seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g, dtype=torch.float64)
    return w, [torch.randn(*shape, generator=g, dtype=torch.float64) for _ in range(n)]


def _hash(state: Dict[str, torch.Tensor]) -> str:
    blob = b"".join(state[k].detach().cpu().numpy().astype("<f8").tobytes() for k in sorted(state))
    return hashlib.sha256(blob).hexdigest()


def test_gate0_regulate_step_is_edge_controller_organ() -> None:
    assert spiral_optimizer.regulate_step is canon_regulate_step


def test_argument_validation() -> None:
    p = [torch.nn.Parameter(torch.ones(3, dtype=torch.float64))]
    with pytest.raises(ValueError, match="lr"):
        SpiralClosureOpt(p, lr=0.0)
    with pytest.raises(ValueError, match="betas"):
        SpiralClosureOpt(p, betas=(1.0, 0.999))
    with pytest.raises(ValueError, match="k"):
        SpiralClosureOpt(p, k=0)
    with pytest.raises(ValueError, match="alpha"):
        SpiralClosureOpt(p, alpha=0.0)  # copie : F5−
    with pytest.raises(ValueError, match="alpha"):
        SpiralClosureOpt(p, alpha=1.5)
    with pytest.raises(ValueError, match="g_min"):
        SpiralClosureOpt(p, g_min=1.3, g_max=1.2)
    with pytest.raises(ValueError, match="weight_decay"):
        SpiralClosureOpt(p, weight_decay=-1.0)


@pytest.mark.parametrize("wd", [0.0, 1e-2])
def test_pivot_adamw_bit_for_bit(wd: float) -> None:
    """α=1, k=1, borne=False, η=0 : trajectoire identique AU BIT à torch.optim.AdamW."""
    w, grads = _seq()
    a, b = torch.nn.Parameter(w.clone()), torch.nn.Parameter(w.clone())
    oa = torch.optim.AdamW([a], lr=3e-3, weight_decay=wd)
    ob = SpiralClosureOpt([b], lr=3e-3, weight_decay=wd, alpha=1.0, k=1, borne=False, eta_rho=0.0)
    for g in grads:
        a.grad, b.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
        assert torch.equal(a.data, b.data)


def test_pivot_eta_zero_keeps_gain() -> None:
    w, grads = _seq()
    p = torch.nn.Parameter(w.clone())
    opt = SpiralClosureOpt([p], lr=1e-2, k=3, g0=0.9, eta_rho=0.0)
    for g in grads:
        p.grad = g.clone()
        opt.step()
        assert opt.state[p]["g_ctrl"] == 0.9


def test_pivot_zero_gradient_from_rest() -> None:
    p0 = torch.tensor([0.25, -0.5, 1.0, -2.0], dtype=torch.float64)
    p = torch.nn.Parameter(p0.clone())
    opt = SpiralClosureOpt([p], lr=1e-2, weight_decay=0.0, k=2)
    for _ in range(9):
        p.grad = torch.zeros_like(p)
        opt.step()
    assert torch.equal(p.data, p0)
    assert torch.equal(opt.state[p]["anchor"], p0)


@pytest.mark.parametrize("alpha", [0.25, 0.5])
def test_closure_formula_exact(alpha: float) -> None:
    """Après k pas : φ ← φ + α(θ_B − φ) et θ ← φ, avec θ_B l'excursion sans retour."""
    k = 4
    w, grads = _seq(n=k)
    p = torch.nn.Parameter(w.clone())
    twin = torch.nn.Parameter(w.clone())
    opt = SpiralClosureOpt([p], lr=1e-2, k=k, alpha=alpha, eta_rho=0.0)
    opt_twin = SpiralClosureOpt([twin], lr=1e-2, k=10_000, alpha=alpha, eta_rho=0.0)
    for j, g in enumerate(grads, start=1):
        p.grad, twin.grad = g.clone(), g.clone()
        opt.step()
        opt_twin.step()
        if j < k:
            assert torch.equal(p.data, twin.data)  # SEG_B : même excursion
    expected = w + alpha * (twin.detach() - w)
    assert torch.allclose(p.data, expected, rtol=0.0, atol=1e-15)
    assert torch.equal(opt.state[p]["anchor"], p.data)
    assert opt.state[p]["cycle_pos"] == 0


def test_step_bounded_under_gradient_spike() -> None:
    """Action tanh : |Δθ| ≤ lr·g par coordonnée hors clôture, même sous un pic ×1e12."""
    w, grads = _seq(n=30, shape=(50,), seed=5)
    p = torch.nn.Parameter(w.clone())
    opt = SpiralClosureOpt([p], lr=1e-2, weight_decay=0.0, k=10_000)
    for t, g in enumerate(grads):
        before = p.detach().clone()
        p.grad = g * (1e12 if t == 15 else 1.0)
        opt.step()
        assert float((p.detach() - before).abs().max()) <= 1e-2 * opt.state[p]["g_ctrl"]


def test_rho_actuator_alive_and_bounded() -> None:
    """ρ̂ n'existe qu'à partir de la 2e clôture (pas 2k) ; ensuite g réagit, borné."""
    g = torch.Generator().manual_seed(1920)
    p = torch.nn.Parameter(torch.randn(16, generator=g, dtype=torch.float64))
    opt = SpiralClosureOpt([p], lr=5e-2, k=2, eta_rho=0.5, borne=False)
    hist = []
    for t in range(60):
        p.grad = torch.randn(16, generator=g, dtype=torch.float64) * (1.0 + 0.3 * t)
        opt.step()
        hist.append(opt.state[p]["g_ctrl"])
    assert hist[:3] == [1.0] * 3  # k=2 : clôtures aux pas 2, 4, … ; ρ̂ dès le pas 4
    assert any(h != 1.0 for h in hist[3:])
    assert all(0.80 <= h <= 1.20 for h in hist)


def test_state_dict_resume_bit_for_bit() -> None:
    """Interrompre puis reprendre via state_dict reproduit la trajectoire au bit."""
    w, grads = _seq(n=23)
    p_ref = torch.nn.Parameter(w.clone())
    o_ref = SpiralClosureOpt([p_ref], lr=1e-2, k=4)
    for g in grads:
        p_ref.grad = g.clone()
        o_ref.step()

    p1 = torch.nn.Parameter(w.clone())
    o1 = SpiralClosureOpt([p1], lr=1e-2, k=4)
    for g in grads[:10]:
        p1.grad = g.clone()
        o1.step()
    saved_p, saved_o = p1.detach().clone(), copy.deepcopy(o1.state_dict())
    p2 = torch.nn.Parameter(saved_p)
    o2 = SpiralClosureOpt([p2], lr=1e-2, k=4)
    o2.load_state_dict(saved_o)
    for g in grads[10:]:
        p2.grad = g.clone()
        o2.step()
    assert torch.equal(p2.data, p_ref.data)


def test_quadratic_descends() -> None:
    g = torch.Generator().manual_seed(1)
    xs = torch.randn(100, generator=g, dtype=torch.float64)
    x = torch.nn.Parameter(torch.zeros(100, dtype=torch.float64))
    opt = SpiralClosureOpt([x], lr=5e-2, weight_decay=0.0)
    f0 = float(0.5 * (xs ** 2).sum())
    for _ in range(600):
        opt.zero_grad()
        f = 0.5 * ((x - xs) ** 2).sum()
        f.backward()
        opt.step()
    assert float(0.5 * ((x.detach() - xs) ** 2).sum()) < 1e-3 * f0


def test_determinism_float64_on_chrono_spiraton() -> None:
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        def _run() -> str:
            torch.manual_seed(64)
            model = ChronoSpiraton(state_size=19, init_scale=0.1, bounded=False).double()
            a = torch.randn(32, 19, generator=torch.Generator().manual_seed(1064), dtype=torch.float64)
            opt = SpiralClosureOpt(model.parameters(), lr=1e-3, k=3)
            for _ in range(10):
                opt.zero_grad()
                alpha_omega_loss(model(a, steps=2), a, target_dist=0.3).loss.backward()
                opt.step()
            return _hash(model.state_dict())

        assert _run() == _run()
    finally:
        torch.use_deterministic_algorithms(prev)


def test_non_divergence_high_init_scale_without_clip() -> None:
    """ChronoSpiraton(19, init_scale=0.6, bounded=False), SANS clip_grad_norm_ :
    finitude, descente franche, copy_rate == 0 (refus F5−)."""
    torch.manual_seed(7)
    model = ChronoSpiraton(state_size=19, init_scale=0.6, bounded=False).double()
    a = torch.randn(48, 19, generator=torch.Generator().manual_seed(1007), dtype=torch.float64)
    opt = SpiralClosureOpt(model.parameters(), lr=5e-3, k=5)
    with torch.no_grad():
        loss_init = alpha_omega_loss(model(a, steps=2), a, target_dist=0.3).metrics["loss"]
    for _ in range(40):
        opt.zero_grad()
        res = alpha_omega_loss(model(a, steps=2), a, target_dist=0.3)
        assert torch.isfinite(res.loss)
        res.loss.backward()
        opt.step()
    with torch.no_grad():
        final = alpha_omega_loss(model(a, steps=2), a, target_dist=0.3)
    assert math.isfinite(final.metrics["loss"])
    assert final.metrics["loss"] < 0.8 * loss_init
    assert final.metrics["copy_rate"] == 0.0
