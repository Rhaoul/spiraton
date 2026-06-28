"""Tests Tour 15 — EdgeController (gain auto-régulé) + edge_maintenance (f_edge, T_survie).

Quatuor canon (finitude, formes simple/batch au sens applicable, formule exacte sous
paramètres forcés, flux de gradient au sens applicable) + contrôle de cohérence DUR
``η=0`` reproduit bit-à-bit le g fixe initial (parallèle CTRL D=L Tour 1 / commutateur=0
Tour 6) + déterminisme bit-à-bit + perturbation entièrement seedée.
"""
import math

import torch
import pytest

from spiraton.experimental.edge_controller import (
    ControlTrace,
    EdgeController,
    GainDrift,
    run_fixed_gain,
)
from spiraton.diagnostics.edge_maintenance import (
    EdgeReport,
    edge_report,
    run_edge_sweep,
    _band_mask,
    _seed_s0,
    W_WINDOW,
    COS_THRESH,
    R_FLOOR,
    R_CEIL,
)


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# --- formes / finitude -------------------------------------------------------

@pytest.mark.parametrize("steps", [40, 120])
def test_run_shapes_and_finite(steps: int) -> None:
    ctrl = EdgeController(omega=math.pi / 5, g0=1.0, eta=0.5)
    s0 = torch.tensor([1.0, 0.0])
    ct = ctrl.run(s0, steps=steps, drift=GainDrift())
    assert ct.trace.shape == (steps + 1, 2)
    assert ct.radius.shape == (steps + 1,)
    assert ct.g_ctrl.shape == (steps + 1,)
    assert ct.rho_hat.shape == (steps + 1,)
    assert ct.g_drift.shape == (steps + 1,)
    assert _finite(ct.trace)
    assert _finite(ct.radius)
    assert _finite(ct.g_ctrl)


def test_g_ctrl_stays_within_bounds() -> None:
    """Le clip est respecté : g_min ≤ g_t ≤ g_max à tous les pas."""
    ctrl = EdgeController(omega=math.pi / 5, g0=1.0, eta=2.0, g_min=0.8, g_max=1.2)
    s0 = torch.tensor([0.6, -0.8])
    ct = ctrl.run(s0, steps=120, drift=GainDrift(start=0.90, end=1.20))
    g = ct.g_ctrl
    assert float(g.min()) >= 0.8 - 1e-9
    assert float(g.max()) <= 1.2 + 1e-9


# --- formule exacte sous paramètres forcés -----------------------------------

def test_drift_ramp_exact() -> None:
    """g_drift(t) interpole linéairement start→end sur T pas (extrêmes exacts)."""
    d = GainDrift(start=0.95, end=1.10)
    T = 100
    assert abs(d.at(0, T) - 0.95) < 1e-12
    assert abs(d.at(T - 1, T) - 1.10) < 1e-12
    # milieu : moyenne des bornes (à la demi-fraction près)
    mid = d.at((T - 1) // 2, T)
    assert 0.95 < mid < 1.10


def test_control_law_exact_one_step() -> None:
    """g_t = clip(g_{t-1} − η(ρ̂_t − 1)) et A_t = g_t·g_drift(t)·R(ω), valeur exacte.

    On vérifie le PREMIER pas régulé (t=1) à la main : ρ̂_1 = r_1/r_0, puis la mise à
    jour de g, puis la transition appliquée à s_1.
    """
    omega, eta, g0 = 0.4, 0.5, 1.0
    ctrl = EdgeController(omega=omega, g0=g0, eta=eta, g_min=0.5, g_max=1.5)
    s0 = torch.tensor([1.0, 0.0])
    drift = GainDrift(start=0.95, end=1.10)
    ct = ctrl.run(s0, steps=5, drift=drift)

    R = torch.tensor(
        [[math.cos(omega), -math.sin(omega)], [math.sin(omega), math.cos(omega)]]
    )
    # pas t=0 : g=g0 (pas de correction), drift(0)
    A0 = (g0 * drift.at(0, 5)) * R
    s1_expected = s0 @ A0.t()
    assert torch.allclose(ct.trace[1], s1_expected, atol=1e-6)

    # pas t=1 : ρ̂_1 = r_1/r_0, g_1 = g0 − η(ρ̂_1 − 1), A_1 = g_1·drift(1)·R
    r0 = float(torch.linalg.vector_norm(s0))
    r1 = float(torch.linalg.vector_norm(s1_expected))
    rho1 = r1 / r0
    g1 = g0 - eta * (rho1 - 1.0)
    g1 = min(1.5, max(0.5, g1))
    assert abs(float(ct.g_ctrl[1]) - g1) < 1e-6
    A1 = (g1 * drift.at(1, 5)) * R
    s2_expected = s1_expected @ A1.t()
    assert torch.allclose(ct.trace[2], s2_expected, atol=1e-6)


# --- CONTRÔLE DE COHÉRENCE DUR : η=0 reproduit bit-à-bit le g fixe -------------

def test_eta_zero_reproduces_fixed_gain_bit_for_bit() -> None:
    """À η=0, le contrôleur EST l'oscilloscope à g fixe g0 sous la MÊME dérive.

    Parallèle CTRL D=L (Tour 1) / commutateur=0 (Tour 6) : la loi de mise à jour
    inerte doit reproduire EXACTEMENT (bit-à-bit) la baseline g fixe. Différent ⇒ bug.
    """
    drift = GainDrift(start=0.95, end=1.10)
    for g_fixed in (0.94, 1.00, 1.06):
        s0 = torch.tensor([0.3, -0.7])
        ctrl = EdgeController(omega=math.pi / 5, g0=g_fixed, eta=0.0)
        ct_ctrl = ctrl.run(s0, steps=80, drift=drift)
        ct_fixed = run_fixed_gain(s0, steps=80, g_fixed=g_fixed,
                                  omega=math.pi / 5, drift=drift)
        assert torch.equal(ct_ctrl.trace, ct_fixed.trace)
        assert torch.equal(ct_ctrl.radius, ct_fixed.radius)
        assert torch.equal(ct_ctrl.g_ctrl, ct_fixed.g_ctrl)


def test_eta_zero_g_ctrl_is_constant() -> None:
    """À η=0, g_t reste exactement g0 à tous les pas (std nul)."""
    ctrl = EdgeController(g0=1.03, eta=0.0)
    s0 = torch.tensor([1.0, 0.0])
    ct = ctrl.run(s0, steps=60, drift=GainDrift())
    assert torch.allclose(ct.g_ctrl, torch.full_like(ct.g_ctrl, 1.03))
    rep = edge_report(ct)
    assert rep.g_std < 1e-12  # plat = pas une régulation (issue c démontrée sur η=0)


# --- déterminisme bit-à-bit ---------------------------------------------------

def test_determinism_bit_for_bit() -> None:
    def run():
        ctrl = EdgeController(omega=math.pi / 5, g0=1.0, eta=0.5)
        s0 = _seed_s0(7)
        return ctrl.run(s0, steps=120, drift=GainDrift())

    a = run()
    b = run()
    assert torch.equal(a.trace, b.trace)
    assert torch.equal(a.g_ctrl, b.g_ctrl)
    assert torch.equal(a.radius, b.radius)


# --- bande PROGRESSION : sanité ----------------------------------------------

def test_band_mask_radius_collapse_out_of_band() -> None:
    """Un rayon effondré (r < r_floor·r_0) est HORS bande."""
    # transition fortement contractante via η=0, g0 bas, drift bas : r → 0.
    ctrl = EdgeController(g0=0.5, eta=0.0)
    s0 = torch.tensor([1.0, 0.0])
    ct = ctrl.run(s0, steps=120, drift=GainDrift(start=0.90, end=0.92))
    mask = _band_mask(ct, window=W_WINDOW, cos_thresh=COS_THRESH,
                      r_floor=R_FLOOR, r_ceil=R_CEIL)
    # en fin de trace, le rayon est sous le plancher → hors bande
    assert mask[-1] is False
    rep = edge_report(ct)
    assert rep.f_edge < 0.5  # majorité hors bande


def test_band_mask_diverging_out_of_band() -> None:
    """Un rayon divergent (r > r_ceil·r_0) est HORS bande."""
    ctrl = EdgeController(g0=1.10, eta=0.0)
    s0 = torch.tensor([1.0, 0.0])
    ct = ctrl.run(s0, steps=120, drift=GainDrift(start=1.05, end=1.10))
    rep = edge_report(ct)
    # divergence garantie (g·drift > 1 partout) → quitte la bande
    assert rep.t_survie < ct.trace.size(0) - 1
    assert rep.f_edge < 1.0


def test_f_edge_in_unit_interval() -> None:
    """f_edge ∈ [0, 1] et T_survie ∈ [0, T] pour le contrôleur."""
    ctrl = EdgeController(g0=1.0, eta=0.5)
    s0 = _seed_s0(3)
    ct = ctrl.run(s0, steps=200, drift=GainDrift())
    rep = edge_report(ct)
    assert 0.0 <= rep.f_edge <= 1.0
    assert 0 <= rep.t_survie <= 200


# --- flux de gradient (la transition est différentiable au sens batch) --------

def test_gradient_flows_through_fixed_transition() -> None:
    """Sanité de différentiabilité : un pas g·drift·R(ω) laisse passer le gradient.

    Le contrôleur lui-même est un chemin de DIAGNOSTIC (@torch.no_grad, lecture de
    rayons) ; le flux de gradient se vérifie sur la transition linéaire sous-jacente,
    comme pour l'oscilloscope (chemin batch différentiable).
    """
    omega = math.pi / 5
    R = torch.tensor(
        [[math.cos(omega), -math.sin(omega)], [math.sin(omega), math.cos(omega)]]
    )
    s0 = torch.tensor([0.5, 0.5], requires_grad=True)
    s = s0
    for t in range(8):
        A = (1.03 * GainDrift().at(t, 8)) * R
        s = s @ A.t()
    s.sum().backward()
    assert s0.grad is not None
    assert float(s0.grad.abs().sum()) > 0.0
