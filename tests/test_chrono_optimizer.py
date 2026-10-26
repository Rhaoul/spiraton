"""Tests chantier OPTIMISEUR_ABA — ``ChronoSpiratonOpt`` (experimental/chrono_optimizer.py).

Portes de la spec (CHANTIER_OPTIMISEUR_ABA.md §5, étape 3) : Porte 0 (identité
d'organe), pivot η=0, pivot repos, pivot Tour 3 ``c_outside``, co-régulation g+ω,
déterminisme P0-c float64, non-divergence ``init_scale=0.6`` sans clip.

Plus, pour CHAQUE correction de la v0, un test qui montre le défaut sous
``legacy_v0=True`` (la spec exacte) et sa disparition en v1 : le défaut reste
démontrable, la correction reste traçable.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict

import pytest
import torch

from spiraton.experimental.chrono import ChronoSpiraton
from spiraton.experimental.edge_controller import regulate_step as canon_regulate_step
from spiraton.experimental.chrono_optimizer import (
    ChronoOptimizerConfig,
    ChronoSpiratonOpt,
    regulate_step as opt_regulate_step,
)
from spiraton.training.aba_loss import alpha_omega_loss


@pytest.fixture
def deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


def _hash_state_dict(state: Dict[str, torch.Tensor]) -> str:
    """Empreinte sha256 canonique '<f8' identique à return_training.hash_etat."""
    blob = b"".join(
        state[k].detach().cpu().numpy().astype("<f8").tobytes() for k in sorted(state)
    )
    return hashlib.sha256(blob).hexdigest()


def _quadratic_losses(legacy_v0: bool, steps: int = 600):
    """f(x) = ½‖x − x*‖², x ∈ R^100 depuis 0 : (perte initiale, perte finale)."""
    g = torch.Generator().manual_seed(1)
    xs = torch.randn(100, generator=g, dtype=torch.float64)
    x = torch.nn.Parameter(torch.zeros(100, dtype=torch.float64))
    opt = ChronoSpiratonOpt([x], lr=1e-2, legacy_v0=legacy_v0)
    f0 = float(0.5 * ((x.detach() - xs) ** 2).sum())
    for _ in range(steps):
        opt.zero_grad()
        f = 0.5 * ((x - xs) ** 2).sum()
        f.backward()
        opt.step()
    return f0, float(0.5 * ((x.detach() - xs) ** 2).sum())


# --- Porte 0 et configuration -------------------------------------------------


def test_gate0_regulate_step_is_edge_controller_organ() -> None:
    """Porte 0 : l'optimiseur réutilise l'organe exact d'edge_controller (aucune copie)."""
    assert opt_regulate_step is canon_regulate_step
    assert opt_regulate_step(1.0, 1.10, 1.0, 0.5, 0.80, 1.20) == canon_regulate_step(
        1.0, 1.10, 1.0, 0.5, 0.80, 1.20
    )


def test_config_dataclass_frozen_and_validation() -> None:
    cfg = ChronoOptimizerConfig(lr=1e-3, window_w=10)
    assert cfg.lr == 1e-3 and cfg.window_w == 10 and cfg.legacy_v0 is False
    with pytest.raises(Exception):
        cfg.lr = 1e-2  # type: ignore[misc]

    p = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
    opt = ChronoSpiratonOpt([p], eta_rho=0.4, window_w=8)
    assert opt.cfg.eta_omega == 0.4 / 8.0  # loi T20 : η_ω = η_ρ / W
    assert opt.param_groups[0]["eta_omega"] == opt.cfg.eta_omega
    with pytest.raises(ValueError, match="lr"):
        ChronoSpiratonOpt([p], lr=-1e-3)
    with pytest.raises(ValueError, match="window_w"):
        ChronoSpiratonOpt([p], window_w=0)
    with pytest.raises(ValueError, match="g_min"):
        ChronoSpiratonOpt([p], g_min=1.30, g_max=1.20)
    with pytest.raises(ValueError, match="omega_min"):
        ChronoSpiratonOpt([p], omega_min=0.5, omega_max=0.314)
    with pytest.raises(ValueError, match="b_quad"):
        ChronoSpiratonOpt([p], b_quad=1.0)
    with pytest.raises(ValueError, match="weight_decay"):
        ChronoSpiratonOpt([p], weight_decay=-0.1)


# --- Pivots ---------------------------------------------------------------------


@pytest.mark.parametrize("legacy_v0", [False, True])
def test_pivot_eta_zero_bit_for_bit(deterministic, legacy_v0: bool) -> None:
    """η_ρ = η_ω = 0 : g ≡ g0, ω ≡ ω0, et trajectoire BIT-À-BIT égale aux gains figés."""
    g = torch.Generator().manual_seed(20260720)
    w_init = torch.randn(8, 8, generator=g, dtype=torch.float64)
    grads = [torch.randn(8, 8, generator=g, dtype=torch.float64) for _ in range(25)]

    common = dict(lr=5e-3, g0=0.97, eta_rho=0.0, window_w=5, omega0=0.0795,
                  eta_omega=0.0, legacy_v0=legacy_v0)
    p_zero = torch.nn.Parameter(w_init.clone())
    opt_zero = ChronoSpiratonOpt([p_zero], **common)
    p_fixed = torch.nn.Parameter(w_init.clone())
    opt_fixed = ChronoSpiratonOpt([p_fixed], g_min=0.97, g_max=0.97,
                                  omega_min=0.0795, omega_max=0.0795, **common)

    for grad in grads:
        p_zero.grad = grad.clone()
        p_fixed.grad = grad.clone()
        opt_zero.step()
        opt_fixed.step()
        assert opt_zero.state[p_zero]["g_ctrl"] == 0.97
        assert opt_zero.state[p_zero]["omega_ctrl"] == 0.0795
        assert torch.equal(p_zero.data, p_fixed.data)


def test_pivot_zero_gradient_keeps_rest_state_exact() -> None:
    """Depuis le repos (s=0), un gradient nul ne déplace rien, même avec ω actif."""
    p0 = torch.tensor([0.25, -0.5, 1.0, -2.0], dtype=torch.float64)
    p = torch.nn.Parameter(p0.clone())
    opt = ChronoSpiratonOpt([p], lr=1e-2, window_w=3)
    for _ in range(15):
        p.grad = torch.zeros_like(p)
        opt.step()
    assert torch.equal(p.data, p0)
    assert torch.equal(opt.state[p]["s_cur"], torch.zeros_like(p0))
    assert torch.equal(opt.state[p]["s_prev"], torch.zeros_like(p0))


def test_pivot_weight_decay_zero_and_decoupled() -> None:
    """wd=0 : bit-à-bit identique au défaut ; wd>0 : θ·(1−lr·wd) découplé du pas."""
    g = torch.Generator().manual_seed(11)
    w0 = torch.randn(6, generator=g, dtype=torch.float64)
    grad = torch.randn(6, generator=g, dtype=torch.float64)

    p_def, p_wd0, p_wd = (torch.nn.Parameter(w0.clone()) for _ in range(3))
    o_def = ChronoSpiratonOpt([p_def], lr=1e-2)
    o_wd0 = ChronoSpiratonOpt([p_wd0], lr=1e-2, weight_decay=0.0)
    o_wd = ChronoSpiratonOpt([p_wd], lr=1e-2, weight_decay=0.5)
    for p, o in ((p_def, o_def), (p_wd0, o_wd0), (p_wd, o_wd)):
        p.grad = grad.clone()
        o.step()
    assert torch.equal(p_def.data, p_wd0.data)
    # Formule exacte au premier pas (ring incomplet ⇒ pas de rappel ω).
    s1 = o_def.state[p_def]["s_cur"]
    expected = (w0 * (1.0 - 1e-2 * 0.5)) - 1e-2 * 1.0 * s1
    assert torch.allclose(p_wd.data, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("legacy_v0", [False, True])
def test_pivot_c_outside_position_tour3(legacy_v0: bool) -> None:
    """Pivot Tour 3 : à c_inhib=0, c_outside True/False identiques au bit sur tout
    l'horizon ; à c_inhib>0, identiques aux pas 1 ET 2 (depuis le repos s_{−1}=s_0=0,
    le terme C(s_{t−1}) est nul tant que t−1 ≤ 0), séparés dès le pas 3.

    (La spec annonçait la séparation au pas 2 : décalage d'un pas, corrigé ici.)
    """
    g = torch.Generator().manual_seed(303)
    w0 = torch.randn(12, generator=g, dtype=torch.float64)
    grads = [torch.randn(12, generator=g, dtype=torch.float64) for _ in range(4)]

    p_in0, p_out0 = torch.nn.Parameter(w0.clone()), torch.nn.Parameter(w0.clone())
    o_in0 = ChronoSpiratonOpt([p_in0], lr=1e-2, c_inhib=0.0, c_outside=False, legacy_v0=legacy_v0)
    o_out0 = ChronoSpiratonOpt([p_out0], lr=1e-2, c_inhib=0.0, c_outside=True, legacy_v0=legacy_v0)
    for grad in grads:
        p_in0.grad, p_out0.grad = grad.clone(), grad.clone()
        o_in0.step()
        o_out0.step()
        assert torch.equal(p_in0.data, p_out0.data)

    p_in, p_out = torch.nn.Parameter(w0.clone()), torch.nn.Parameter(w0.clone())
    o_in = ChronoSpiratonOpt([p_in], lr=1e-2, c_inhib=0.15, c_outside=False, legacy_v0=legacy_v0)
    o_out = ChronoSpiratonOpt([p_out], lr=1e-2, c_inhib=0.15, c_outside=True, legacy_v0=legacy_v0)
    for k, grad in enumerate(grads[:3], start=1):
        p_in.grad, p_out.grad = grad.clone(), grad.clone()
        o_in.step()
        o_out.step()
        if k <= 2:
            assert torch.equal(p_in.data, p_out.data), f"pas {k} : s_{{t−1}} = 0"
        else:
            assert not torch.equal(p_in.data, p_out.data), "pas 3 : s_{t−1} ≠ 0"
    assert torch.isfinite(p_in.data).all() and torch.isfinite(p_out.data).all()


# --- Défauts de la v0 démontrés, puis corrigés ------------------------------------


def test_correction1_first_step_follows_negative_gradient() -> None:
    """v0 : le terme B retourne le signe du pas (cos(Δθ, −g) < 0). v1 : cos ≈ 1."""
    g = torch.Generator().manual_seed(0)
    w0 = torch.randn(361, generator=g, dtype=torch.float64)
    grad = torch.randn(361, generator=g, dtype=torch.float64)
    cos = {}
    for v0 in (True, False):
        p = torch.nn.Parameter(w0.clone())
        p.grad = grad.clone()
        ChronoSpiratonOpt([p], lr=1e-2, legacy_v0=v0).step()
        d = p.detach() - w0
        cos[v0] = float(torch.dot(d, -grad) / (d.norm() * grad.norm()))
    assert cos[True] < -0.5
    assert cos[False] > 0.99


def test_correction1_quadratic_descends_only_in_v1() -> None:
    """v0 ne descend pas une quadratique (perte finale ≥ 95 % de l'initiale) ; v1 si."""
    f0, f_v0 = _quadratic_losses(legacy_v0=True)
    _, f_v1 = _quadratic_losses(legacy_v0=False)
    assert f_v0 > 0.95 * f0
    assert f_v1 < 0.05 * f0


def _run_regulators(legacy_v0: bool, steps: int = 40):
    g = torch.Generator().manual_seed(1920)
    p = torch.nn.Parameter(torch.randn(16, generator=g, dtype=torch.float64))
    opt = ChronoSpiratonOpt([p], lr=5e-2, window_w=4, legacy_v0=legacy_v0)
    g_hist, om_hist = [], []
    for t in range(steps):
        p.grad = torch.randn(16, generator=g, dtype=torch.float64) * (1.0 + 0.3 * t)
        opt.step()
        g_hist.append(opt.state[p]["g_ctrl"])
        om_hist.append(opt.state[p]["omega_ctrl"])
    return g_hist, om_hist


def test_correction2_rho_hat_actuator_is_alive_only_in_v1() -> None:
    """v0 : ρ̂ compare un tenseur à lui-même ⇒ g ≡ g0. v1 : g réagit, dans ses bornes."""
    g_v0, _ = _run_regulators(legacy_v0=True)
    g_v1, _ = _run_regulators(legacy_v0=False)
    assert all(gv == 1.0 for gv in g_v0)
    assert g_v1[0] == 1.0 and g_v1[1] == 1.0  # ρ̂ défini à partir de t = 2
    assert any(gv != 1.0 for gv in g_v1[2:])
    assert all(0.80 <= gv <= 1.20 for gv in g_v1)


def test_correction3_omega_feedback_sign() -> None:
    """Dans l'espace des poids cos(θ_t, θ_{t−W}) ≈ 1 > 0,7 : v0 (boucle positive) sature
    à ω_max ; v1 (boucle négative) ramène ω à ω_min — le rappel s'éteint de lui-même.
    Avant W pas, ω reste ω0 (fenêtre incomplète, convention T19/T20)."""
    _, om_v0 = _run_regulators(legacy_v0=True)
    _, om_v1 = _run_regulators(legacy_v0=False)
    for om in (om_v0, om_v1):
        assert all(o == 0.0795 for o in om[:4])
        assert all(0.0 <= o <= 0.314 for o in om)
    assert om_v0[-1] == 0.314
    assert om_v1[-1] == 0.0


def test_correction4_step_scale_independent_of_tensor_size() -> None:
    """v1 : pas RMS par coordonnée ≈ indépendant de n ; v0 : décroît comme 1/√n.

    ``b_quad=0`` isole la normalisation (en v0, le terme B défectueux masquerait l'effet).
    """
    rms = {}
    for v0 in (True, False):
        for n in (16, 4096):
            g = torch.Generator().manual_seed(n)
            p = torch.nn.Parameter(torch.zeros(n, dtype=torch.float64))
            p.grad = torch.randn(n, generator=g, dtype=torch.float64)
            ChronoSpiratonOpt([p], lr=1e-2, b_quad=0.0, legacy_v0=v0).step()
            rms[(v0, n)] = float(p.detach().pow(2).mean().sqrt())
    assert rms[(True, 16)] / rms[(True, 4096)] > 4.0
    assert 0.8 < rms[(False, 16)] / rms[(False, 4096)] < 1.25


# --- Déterminisme, bornes, stabilité ----------------------------------------------


def test_step_is_bounded_under_gradient_spike() -> None:
    """Un pic de gradient ×1e12 ne produit pas un pas plus grand que la borne
    ``lr·g_max·(d + l·π/2 + c·|s_{t−1}|)`` : le rôle du clip_grad_norm_ externe."""
    g = torch.Generator().manual_seed(5)
    p = torch.nn.Parameter(torch.randn(50, generator=g, dtype=torch.float64))
    opt = ChronoSpiratonOpt([p], lr=1e-2)
    bound = 1e-2 * 1.20 * (0.7 + 0.3 * math.pi / 2 + 0.05 * 2.0) * (1.0 + 1e-9)
    for t in range(30):
        scale = 1e12 if t == 15 else 1.0
        before = p.detach().clone()
        p.grad = torch.randn(50, generator=g, dtype=torch.float64) * scale
        opt.step()
        assert float((p.detach() - before).abs().max()) <= bound


def test_determinism_p0c_float64_on_chrono_spiraton(deterministic) -> None:
    """Parité P0-c de return_training.py : deux runs froids donnent le même sha256."""

    def _run_cold() -> str:
        torch.manual_seed(64)
        model = ChronoSpiraton(state_size=19, init_scale=0.1, bounded=False, c_outside=False).double()
        gen = torch.Generator().manual_seed(1064)
        a = torch.randn(32, 19, generator=gen, dtype=torch.float64)
        opt = ChronoSpiratonOpt(model.parameters(), lr=1e-3, window_w=5)
        for _ in range(10):
            opt.zero_grad()
            alpha_omega_loss(model(a, steps=2), a, target_dist=0.3).loss.backward()
            opt.step()
        return _hash_state_dict(model.state_dict())

    assert _run_cold() == _run_cold()


def test_non_divergence_and_spiral_closure_on_high_init_scale() -> None:
    """Porte 6 : ChronoSpiraton(19, init_scale=0.6, bounded=False), SANS clip_grad_norm_ :
    100 % de finitude, descente franche de alpha_omega_loss, copy_rate == 0 (refus F5−)."""
    torch.manual_seed(7)
    model = ChronoSpiraton(state_size=19, init_scale=0.6, bounded=False, c_outside=False).double()
    gen = torch.Generator().manual_seed(1007)
    a = torch.randn(48, 19, generator=gen, dtype=torch.float64)
    opt = ChronoSpiratonOpt(model.parameters(), lr=5e-3, window_w=6)

    with torch.no_grad():
        loss_init = alpha_omega_loss(model(a, steps=2), a, target_dist=0.3).metrics["loss"]

    for _ in range(25):
        def closure() -> torch.Tensor:
            opt.zero_grad()
            res = alpha_omega_loss(model(a, steps=2), a, target_dist=0.3)
            res.loss.backward()
            return res.loss

        loss_t = opt.step(closure)
        assert loss_t is not None and torch.isfinite(loss_t)

    with torch.no_grad():
        final = alpha_omega_loss(model(a, steps=2), a, target_dist=0.3)
    assert math.isfinite(final.metrics["loss"])
    assert final.metrics["loss"] < 0.8 * loss_init
    assert final.metrics["copy_rate"] == 0.0
