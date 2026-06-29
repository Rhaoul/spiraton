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


# =============================================================================
# Tour 16 — VRAIE variance de population : perturbation SEEDÉE PAR GRAINE.
#
# H16 : sous perturbation non-isotrope tirée indépendamment par graine, l'avantage
# du contrôleur sur le meilleur g fixe PERSISTE-T-IL en distribution ? P1 (non-
# stationnaire, SeededDrift) vs P2 (proche-stationnaire AR(1), ProcessNoise).
# Geste : SUB · dextro · out (creuser un écart entre graines confondues, l.341-342).
# =============================================================================

from spiraton.experimental.edge_controller import (  # noqa: E402
    SeededDrift,
    ProcessNoise,
)
from spiraton.diagnostics.edge_maintenance import (  # noqa: E402
    run_edge_sweep,
    run_variance_sweep,
    p1_drift_factory,
    p2_noise_factory,
    wilcoxon_signed_rank,
    _median,
    _max_pairwise_radius_spread,
)


# --- quatuor adapté : finitude / formes des nouvelles perturbations -----------

@pytest.mark.parametrize("steps", [40, 120])
def test_seeded_drift_run_shapes_and_finite(steps: int) -> None:
    ctrl = EdgeController(omega=math.pi / 5, g0=1.0, eta=0.5)
    s0 = _seed_s0(2)
    ct = ctrl.run(s0, steps=steps, drift=SeededDrift.from_seed(2))
    assert ct.trace.shape == (steps + 1, 2)
    assert _finite(ct.trace) and _finite(ct.radius) and _finite(ct.g_ctrl)


@pytest.mark.parametrize("steps", [40, 120])
def test_process_noise_run_shapes_and_finite(steps: int) -> None:
    ctrl = EdgeController(omega=math.pi / 5, g0=1.0, eta=0.5)
    s0 = _seed_s0(2)
    ct = ctrl.run(s0, steps=steps, drift=ProcessNoise.from_seed(2, steps=steps))
    assert ct.trace.shape == (steps + 1, 2)
    assert _finite(ct.trace) and _finite(ct.radius) and _finite(ct.g_ctrl)


# --- formule exacte sous paramètres forcés -----------------------------------

def test_seeded_drift_reduces_to_gaindrift_formula() -> None:
    """À offset/pente fixés, SeededDrift.at == GainDrift.at (même expression, à l'identique)."""
    sd = SeededDrift(start=0.95, end=1.10)
    gd = GainDrift(start=0.95, end=1.10)
    for T in (5, 100, 200):
        for t in range(T):
            assert sd.at(t, T) == gd.at(t, T)  # égalité FLOAT exacte, pas allclose


def test_process_noise_zero_sigma_is_constant_base() -> None:
    """À σ=0, ProcessNoise.at ≡ base ∀t (série nulle) — pivot P2."""
    pn = ProcessNoise.from_seed(7, steps=50, sigma=0.0, base=1.0)
    for t in range(50):
        assert pn.at(t, 50) == 1.0
    # degenerate() équivaut à σ=0
    deg = ProcessNoise.degenerate(base=1.0)
    for t in range(50):
        assert deg.at(t, 50) == 1.0


def test_process_noise_ar1_recursion_exact() -> None:
    """e_t = φ·e_{t-1} + w_t, vérifié à la main sur les innovations seedées."""
    seed, steps, phi, sigma = 4, 6, 0.5, 0.04
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(steps, generator=g) * sigma
    e_prev, expected = 0.0, []
    for t in range(steps):
        e = phi * e_prev + float(w[t])
        expected.append(e)
        e_prev = e
    pn = ProcessNoise.from_seed(seed, steps=steps, phi=phi, sigma=sigma, base=1.0)
    for t in range(steps):
        assert abs(pn.at(t, steps) - (1.0 + expected[t])) < 1e-12


# --- déterminisme par graine -------------------------------------------------

def test_seeded_perturbations_are_deterministic_per_seed() -> None:
    assert SeededDrift.from_seed(3) == SeededDrift.from_seed(3)
    assert SeededDrift.from_seed(3) != SeededDrift.from_seed(4)
    assert ProcessNoise.from_seed(3, steps=30).series == ProcessNoise.from_seed(3, steps=30).series
    assert ProcessNoise.from_seed(3, steps=30).series != ProcessNoise.from_seed(4, steps=30).series


# --- PIVOT ANTI-ARTEFACT DUR : variance=0 ⇒ T16 reproduit T15 bit-à-bit -------

def test_pivot_p1_degenerate_reproduces_t15_bit_for_bit() -> None:
    """SeededDrift.degenerate (offset/pente fixés T15) ≡ GainDrift(0.95,1.10) bit-à-bit.

    Transposition du contrôle η=0 du Tour 15 à la variance : à amplitude NULLE, la
    perturbation par graine DOIT coïncider EXACTEMENT avec la dérive T15 partagée.
    Différent ⇒ bug d'injection, pas un résultat.
    """
    for seed in range(8):
        s0 = _seed_s0(seed)
        ct_t15 = EdgeController(g0=1.0, eta=0.5).run(s0, steps=160, drift=GainDrift(0.95, 1.10))
        ct_t16 = EdgeController(g0=1.0, eta=0.5).run(s0, steps=160, drift=SeededDrift.degenerate())
        assert torch.equal(ct_t15.trace, ct_t16.trace)
        assert torch.equal(ct_t15.radius, ct_t16.radius)
        assert torch.equal(ct_t15.g_ctrl, ct_t16.g_ctrl)


def test_pivot_p1_degenerate_sweep_reproduces_t15() -> None:
    """Au niveau SWEEP : variance=0 ⇒ f_edge contrôleur ET best_fixed identiques à T15.

    Et Δf_edge est une VALEUR UNIQUE répétée (le « 40 angles, valeur unique » du T15) :
    l'isotropie persiste (radius_distinct == False) puisque la perturbation ne dépend
    plus de la graine.
    """
    n, steps = 12, 120
    t15 = run_edge_sweep(n_seeds=n, steps=steps, drift=GainDrift(0.95, 1.10))
    t16 = run_variance_sweep(
        lambda s: SeededDrift.degenerate(start=0.95, end=1.10), n_seeds=n, steps=steps
    )
    assert t15.ctrl_f_edge == t16.ctrl_f_edge
    assert t15.best_fixed_gain == t16.best_fixed_gain
    # Δf_edge identique pour toutes les graines (isotropie)
    assert len(set(round(x, 9) for x in t16.delta_vs_fixed)) == 1
    assert t16.radius_distinct is False


def test_pivot_p2_zero_sigma_is_native_constant_gain() -> None:
    """P2 à σ=0 ≡ oscilloscope à gain natif CONSTANT base : isotropie, trace identique."""
    for seed in range(6):
        s0 = _seed_s0(seed)
        ct_const = run_fixed_gain(s0, steps=160, g_fixed=1.0, drift=GainDrift(1.0, 1.0))
        ct_p2 = run_fixed_gain(s0, steps=160, g_fixed=1.0, drift=ProcessNoise.degenerate(base=1.0))
        assert torch.equal(ct_const.trace, ct_p2.trace)


# --- GARDE-FOU : séries r_t distinctes sous variance>0, confondues sous variance=0 --

def test_guardrail_radius_distinct_under_variance() -> None:
    """Pré-condition de validité H16 : sous P1/P2 (variance>0) les séries r_t DIFFÈRENT.

    C'est l'INVERSE du constat T15 (où la même dérive partagée rendait r_t invariant
    par graine). Si elles ne diffèrent pas, le test de population serait VIDE.
    """
    n, steps = 16, 150

    def radii(factory):
        out = []
        for seed in range(n):
            s0 = _seed_s0(seed)
            ct = EdgeController(g0=1.0, eta=0.5).run(s0, steps=steps, drift=factory(seed))
            out.append(ct.radius)
        return out

    p1_spread = _max_pairwise_radius_spread(radii(p1_drift_factory()))
    p2_spread = _max_pairwise_radius_spread(radii(p2_noise_factory(steps=steps)))
    assert p1_spread > 1e-3   # vraie variance de population (P1)
    assert p2_spread > 1e-3   # vraie variance de population (P2)

    # variance NULLE ⇒ isotropie (séries r_t confondues, comme T15)
    deg_spread = _max_pairwise_radius_spread(
        radii(lambda s: SeededDrift.degenerate())
    )
    assert deg_spread < 1e-5


# --- Wilcoxon signed-rank maison : sanité contre cas connus -------------------

def test_wilcoxon_all_positive_is_significant() -> None:
    """40 différences toutes positives ⇒ W+ maximal, p très petit."""
    deltas = [0.1 * (i + 1) for i in range(40)]  # toutes > 0
    w_plus, p, n = wilcoxon_signed_rank(deltas)
    assert n == 40
    assert w_plus == 40 * 41 / 2  # somme de tous les rangs
    assert p < 1e-6


def test_wilcoxon_symmetric_is_not_significant() -> None:
    """Différences symétriques autour de 0 ⇒ pas de significativité."""
    deltas = [(-1) ** i * (i % 5 + 1) * 0.1 for i in range(40)]
    _, p, _ = wilcoxon_signed_rank(deltas)
    assert p > 0.05


def test_wilcoxon_drops_zeros() -> None:
    """Les différences nulles sont écartées (n_effectif les exclut)."""
    deltas = [0.0, 0.0, 0.3, -0.1, 0.2]
    _, _, n = wilcoxon_signed_rank(deltas)
    assert n == 3


# --- déterminisme bit-à-bit du balayage T16 (relance ×2) ---------------------

def test_variance_sweep_determinism_bit_for_bit() -> None:
    """run_variance_sweep est reproductible : relance ×2 ⇒ distributions identiques."""
    f = p2_noise_factory(steps=80)
    a = run_variance_sweep(f, n_seeds=12, steps=80)
    b = run_variance_sweep(f, n_seeds=12, steps=80)
    assert a.ctrl_f_edge == b.ctrl_f_edge
    assert a.best_fixed_f_edge == b.best_fixed_f_edge
    assert a.delta_vs_fixed == b.delta_vs_fixed
    assert a.wilcoxon_p == b.wilcoxon_p
    assert a.best_fixed_gain == b.best_fixed_gain


# =============================================================================
# Tour 17 — LOI DE RÉPONSE au mélange convexe P_α = α·P1 + (1−α)·P2.
#
# H17 : Δf_edge(α) suit la DÉRIVE NETTE (composante DC ∝ α), PAS la variation
# totale ∫|dg/dt|. Pivot anti-artefact : α=1 ≡ P1, α=0 ≡ P2 bit-à-bit.
# Geste : ADD · dextro · out (agrégation pondérée, dual du SUB du T16).
# =============================================================================

from spiraton.experimental.edge_controller import (  # noqa: E402
    MixedPerturbation,
)
from spiraton.diagnostics.edge_maintenance import (  # noqa: E402
    run_alpha_mix_sweep,
    mix_factory,
    spearman_rho,
    _rankdata,
)


# --- quatuor adapté : finitude / formes du mélange ---------------------------

@pytest.mark.parametrize("steps", [40, 120])
def test_mixed_perturbation_run_shapes_and_finite(steps: int) -> None:
    ctrl = EdgeController(omega=math.pi / 5, g0=1.0, eta=0.5)
    s0 = _seed_s0(2)
    mix = MixedPerturbation.from_seed(2, alpha=0.5, steps=steps)
    ct = ctrl.run(s0, steps=steps, drift=mix)
    assert ct.trace.shape == (steps + 1, 2)
    assert _finite(ct.trace) and _finite(ct.radius) and _finite(ct.g_ctrl)


def test_mixed_perturbation_convex_formula_exact() -> None:
    """p_α.at == α·p1.at + (1−α)·p2.at, à l'identique (égalité FLOAT, pas allclose)."""
    steps = 60
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        mix = MixedPerturbation.from_seed(3, alpha=alpha, steps=steps)
        for t in range(steps):
            expected = alpha * mix.p1.at(t, steps) + (1.0 - alpha) * mix.p2.at(t, steps)
            assert mix.at(t, steps) == expected


def test_mixed_perturbation_deterministic_per_seed() -> None:
    a = MixedPerturbation.from_seed(5, alpha=0.4, steps=50)
    b = MixedPerturbation.from_seed(5, alpha=0.4, steps=50)
    assert a.p1 == b.p1 and a.p2.series == b.p2.series and a.alpha == b.alpha
    c = MixedPerturbation.from_seed(6, alpha=0.4, steps=50)
    assert not (a.p1 == c.p1 and a.p2.series == c.p2.series)


# --- PIVOT ANTI-ARTEFACT DUR : α=1 ≡ P1, α=0 ≡ P2 bit-à-bit (à exécuter d'abord) --

def test_pivot_alpha_one_is_seeded_drift_bit_for_bit() -> None:
    """À α=1, p_α.at == SeededDrift.from_seed(seed).at bit-à-bit sur tous t/graines.

    Le terme (1−α)·p2 vaut 0.0·p2 = 0.0 et p1 + 0.0 == p1 en IEEE754 : identité
    exacte, pas approchée. Un seul écart ⇒ bug d'injection (issue v).
    """
    steps = 200
    for seed in range(40):
        mix = MixedPerturbation.from_seed(seed, alpha=1.0, steps=steps)
        p1 = SeededDrift.from_seed(seed)
        for t in range(steps):
            assert mix.at(t, steps) == p1.at(t, steps)


def test_pivot_alpha_zero_is_process_noise_bit_for_bit() -> None:
    """À α=0, p_α.at == ProcessNoise.from_seed(seed, steps).at bit-à-bit (tous t/graines)."""
    steps = 200
    for seed in range(40):
        mix = MixedPerturbation.from_seed(seed, alpha=0.0, steps=steps)
        p2 = ProcessNoise.from_seed(seed, steps=steps)
        for t in range(steps):
            assert mix.at(t, steps) == p2.at(t, steps)


def test_pivot_alpha_borders_reproduce_t16_traces_bit_for_bit() -> None:
    """Au niveau TRACE : α=1 reproduit la trace P1, α=0 la trace P2 (bit-à-bit)."""
    steps = 160
    for seed in range(8):
        s0 = _seed_s0(seed)
        ctrl = lambda: EdgeController(g0=1.0, eta=0.5)  # noqa: E731
        # α=1 ≡ SeededDrift
        ct_p1 = ctrl().run(s0, steps=steps, drift=SeededDrift.from_seed(seed))
        ct_m1 = ctrl().run(s0, steps=steps,
                           drift=MixedPerturbation.from_seed(seed, alpha=1.0, steps=steps))
        assert torch.equal(ct_p1.trace, ct_m1.trace)
        assert torch.equal(ct_p1.g_ctrl, ct_m1.g_ctrl)
        # α=0 ≡ ProcessNoise
        ct_p2 = ctrl().run(s0, steps=steps, drift=ProcessNoise.from_seed(seed, steps=steps))
        ct_m0 = ctrl().run(s0, steps=steps,
                           drift=MixedPerturbation.from_seed(seed, alpha=0.0, steps=steps))
        assert torch.equal(ct_p2.trace, ct_m0.trace)
        assert torch.equal(ct_p2.g_ctrl, ct_m0.g_ctrl)


def test_mixed_degenerate_returns_pure_component_at_borders() -> None:
    """degenerate() renvoie p1 à α=1, p2 à α=0, et lève hors des bornes."""
    m1 = MixedPerturbation.from_seed(0, alpha=1.0, steps=50)
    m0 = MixedPerturbation.from_seed(0, alpha=0.0, steps=50)
    assert m1.degenerate() is m1.p1
    assert m0.degenerate() is m0.p2
    with pytest.raises(ValueError):
        MixedPerturbation.from_seed(0, alpha=0.5, steps=50).degenerate()


# --- Spearman maison : sanité contre cas connus ------------------------------

def test_spearman_perfect_monotone() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [10.0, 20.0, 30.0, 40.0, 50.0]   # monotone croissant
    assert abs(spearman_rho(a, b) - 1.0) < 1e-12
    c = [50.0, 40.0, 30.0, 20.0, 10.0]   # monotone décroissant
    assert abs(spearman_rho(a, c) + 1.0) < 1e-12


def test_spearman_handles_ties_via_mean_ranks() -> None:
    """Ex æquo gérés par rangs moyens (pas la formule 6Σd² qui les ignore)."""
    ranks = _rankdata([3.0, 1.0, 1.0, 2.0])  # deux ex æquo en 1.0 -> rangs 1.5,1.5
    assert ranks == [4.0, 1.5, 1.5, 3.0]


def test_spearman_constant_is_zero() -> None:
    assert spearman_rho([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) == 0.0


# --- balayage T17 : bornes reproduisent T16, monotonie, invariant dérive-nette --

def test_alpha_mix_sweep_borders_reproduce_t16() -> None:
    """α=1 reproduit Δf_edge(P1) et α=0 reproduit Δf_edge(P2) du T16 (mêmes valeurs).

    Le mélange aux bornes EST la perturbation T16 (pivot trace) ⇒ même best_fixed,
    même distribution Δf_edge ⇒ même médiane. On compare sur une grille réduite pour
    la vitesse (paramètres identiques au sweep complet).
    """
    n, steps = 16, 120
    mix = run_alpha_mix_sweep([0.0, 1.0], n_seeds=n, steps=steps)
    p1 = run_variance_sweep(p1_drift_factory(), n_seeds=n, steps=steps)
    p2 = run_variance_sweep(p2_noise_factory(steps=steps), n_seeds=n, steps=steps)
    by_a = {p.alpha: p for p in mix.points}
    assert by_a[1.0].delta_median == p1.delta_median
    assert by_a[1.0].delta_vs_fixed == p1.delta_vs_fixed
    assert by_a[0.0].delta_median == p2.delta_median
    assert by_a[0.0].delta_vs_fixed == p2.delta_vs_fixed


def test_alpha_mix_sweep_is_monotone_and_drift_driven() -> None:
    """H17 : Δf_edge MONTE avec α (Spearman α ≥ 0.85), suit net_drift PAS total_var.

    Test discriminant central : ρ_s(net_drift, Δ) ≥ 0.85 ET |ρ_s(total_var, Δ)| < 0.3
    serait l'idéal, mais comme net_drift ∝ α et total_var DÉCROÎT en α, total_var est
    fortement ANTI-corrélé. L'assertion honnête : Δ corrèle POSITIVEMENT α/net_drift
    et NÉGATIVEMENT total_var (la dérive nette gouverne, pas la variation totale —
    sinon Δ monterait avec total_var). Grille réduite pour la vitesse.
    """
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    res = run_alpha_mix_sweep(alphas, n_seeds=20, steps=120)
    assert res.spearman_alpha_delta >= 0.85          # monotonie en α
    assert res.spearman_netdrift_delta >= 0.85       # suit la dérive nette
    assert res.spearman_totalvar_delta < 0.3         # NE suit PAS (anti-)corrélé total_var
    assert res.n_inversions <= 1                     # ≤ 1 inversion hors-ε
    # net_drift croît en α, total_var décroît (P2 h.f. perd du poids)
    ordered = sorted(res.points, key=lambda q: q.alpha)
    assert ordered[-1].net_drift > ordered[0].net_drift
    assert ordered[-1].total_var < ordered[0].total_var


def test_alpha_mix_sweep_all_radius_distinct() -> None:
    """Garde-fou : à chaque α>0 les séries r_t diffèrent (vraie variance de population)."""
    res = run_alpha_mix_sweep([0.0, 0.5, 1.0], n_seeds=12, steps=100)
    for p in res.points:
        assert p.radius_distinct is True


def test_alpha_mix_sweep_determinism_bit_for_bit() -> None:
    """run_alpha_mix_sweep reproductible : relance ×2 ⇒ médianes et Spearman identiques."""
    a = run_alpha_mix_sweep([0.0, 0.5, 1.0], n_seeds=10, steps=80)
    b = run_alpha_mix_sweep([0.0, 0.5, 1.0], n_seeds=10, steps=80)
    assert [p.delta_median for p in a.points] == [p.delta_median for p in b.points]
    assert [p.net_drift for p in a.points] == [p.net_drift for p in b.points]
    assert [p.total_var for p in a.points] == [p.total_var for p in b.points]
    assert a.spearman_alpha_delta == b.spearman_alpha_delta
    assert a.alpha_star == b.alpha_star


# =============================================================================
# TOUR 18 — DriftPlusHFSine + run_hf_amplitude_sweep (DISJONCTION net_drift/total_var)
#
# H18 : sous net_drift CONSTANT (P1 par graine FIXE) + total_var CROISSANT (sinus
# moyenne-nulle d'amplitude A), Δf_edge(A) reste PLAT ⇒ net_drift gouverne. Le geste
# DIV·lévo·in (⊘ SÉPARER ce qui était confondu en α, l.200) DISJOINT les deux variables
# que le mélange T17 confondait. Pivot anti-artefact : A=0 ≡ P1 du T16 bit-à-bit (et
# +0.6556, PAS +0.6623 de .degenerate() mono-série). Garde auto-protectrice : si le
# sinus FUIT dans net_drift (mauvaise phase), la pré-condition netdrift_is_flat échoue.
# =============================================================================

from spiraton.experimental.edge_controller import (  # noqa: E402
    DriftPlusHFSine,
    SeededDrift as _SeededDrift,
)
from spiraton.diagnostics.edge_maintenance import (  # noqa: E402
    run_hf_amplitude_sweep,
    run_variance_sweep,
    hf_factory,
    p1_drift_factory,
    _net_drift_and_total_var,
)


# --- quatuor adapté : finitude / formes / formule exacte / pivot -------------

@pytest.mark.parametrize("steps", [40, 120])
def test_hf_sine_run_shapes_and_finite(steps: int) -> None:
    ctrl = EdgeController(omega=math.pi / 5, g0=1.0, eta=0.5)
    s0 = _seed_s0(2)
    hf = DriftPlusHFSine.from_seed(2, amplitude=0.05, k_periods=10, steps=steps)
    ct = ctrl.run(s0, steps=steps, drift=hf)
    assert ct.trace.shape == (steps + 1, 2)
    assert _finite(ct.trace) and _finite(ct.radius) and _finite(ct.g_ctrl)


def test_hf_sine_additive_formula_exact() -> None:
    """p.at == p1.at + A·sin(2π·k·t/(N−1)), à l'identique (égalité FLOAT, pas allclose)."""
    steps = 80
    for A in (0.02, 0.06, 0.12):
        for k in (5, 20, 50):
            hf = DriftPlusHFSine.from_seed(4, amplitude=A, k_periods=k, steps=steps)
            for t in range(steps):
                sine = math.sin(2.0 * math.pi * k * t / (steps - 1))
                expected = hf.p1.at(t, steps) + A * sine
                assert hf.at(t, steps) == expected


def test_hf_sine_vanishes_at_sampled_endpoints() -> None:
    """Le sinus s'annule (à ~5e-15) aux deux extrémités ÉCHANTILLONNÉES t∈{0, N−1}.

    C'est la condition qui empêche le sinus de FUIR dans net_drift = |p(N−1)−p(0)|.
    À t=0 : sin(0)=0 exact ; à t=N−1 : sin(2πk) ≈ 0 (résidu flottant). On vérifie que
    p.at coïncide avec p1.at aux extrémités à 1e-12 près malgré une amplitude non nulle.
    """
    steps = 200
    for k in (5, 20, 50):
        hf = DriftPlusHFSine.from_seed(0, amplitude=0.12, k_periods=k, steps=steps)
        assert abs(hf.at(0, steps) - hf.p1.at(0, steps)) < 1e-12
        assert abs(hf.at(steps - 1, steps) - hf.p1.at(steps - 1, steps)) < 1e-12


# --- PIVOT anti-artefact : A=0 ≡ P1 du T16 bit-à-bit -------------------------

def test_pivot_amplitude_zero_is_seeded_drift_bit_for_bit() -> None:
    """À A=0, DriftPlusHFSine ≡ SeededDrift.from_seed bit-à-bit (.at, 40 graines)."""
    steps = 200
    for seed in range(40):
        hf = DriftPlusHFSine.from_seed(seed, amplitude=0.0, k_periods=20, steps=steps)
        sd = _SeededDrift.from_seed(seed)
        for t in range(steps):
            assert hf.at(t, steps) == sd.at(t, steps)


def test_hf_degenerate_returns_p1_at_zero_and_raises_otherwise() -> None:
    """degenerate() renvoie l'objet p1 (SeededDrift) à A=0, et lève hors de A=0."""
    hf0 = DriftPlusHFSine.from_seed(7, amplitude=0.0, steps=200)
    assert hf0.degenerate() is hf0.p1
    assert isinstance(hf0.degenerate(), _SeededDrift)
    with pytest.raises(ValueError):
        DriftPlusHFSine.from_seed(7, amplitude=0.05, steps=200).degenerate()


def test_pivot_amplitude_zero_trace_reproduces_t16_bit_for_bit() -> None:
    """À A=0, la trace SOUS CONTRÔLE coïncide bit-à-bit avec P1 du T16 (SeededDrift)."""
    steps = 200
    for seed in (0, 3, 11):
        s0 = _seed_s0(seed)
        ct_hf = EdgeController(g0=1.0, eta=0.5).run(
            s0, steps=steps, drift=DriftPlusHFSine.from_seed(seed, amplitude=0.0, steps=steps))
        ct_sd = EdgeController(g0=1.0, eta=0.5).run(
            s0, steps=steps, drift=_SeededDrift.from_seed(seed))
        assert torch.equal(ct_hf.trace, ct_sd.trace)
        assert torch.equal(ct_hf.g_ctrl, ct_sd.g_ctrl)
        assert torch.equal(ct_hf.radius, ct_sd.radius)


def test_pivot_amplitude_zero_sweep_matches_t16_p1() -> None:
    """Le sweep HF à A=0 reproduit le point P1 du T16 (delta_vs_fixed bit-à-bit)."""
    n, steps = 16, 120
    hf0 = run_variance_sweep(hf_factory(0.0, k_periods=20, steps=steps), n_seeds=n, steps=steps)
    p1 = run_variance_sweep(p1_drift_factory(), n_seeds=n, steps=steps)
    assert hf0.delta_vs_fixed == p1.delta_vs_fixed
    assert hf0.delta_median == p1.delta_median
    assert hf0.best_fixed_gain == p1.best_fixed_gain


# --- PRÉ-CONDITION de validité : net_drift plat / total_var croissant --------

def test_precondition_net_drift_is_flat_total_var_increases() -> None:
    """net_drift(A) plat à <1e-6 (le sinus NE fuit PAS) ET total_var(A) croissant.

    Si net_drift n'était pas plat ⇒ BUG (issue iv), pas résultat : le test l'attrape.
    """
    amps = (0.0, 0.02, 0.06, 0.12)
    steps = 120
    nets, tvs = [], []
    for A in amps:
        nd, tv = _net_drift_and_total_var(
            hf_factory(A, k_periods=20, steps=steps), n_seeds=16, steps=steps)
        nets.append(nd)
        tvs.append(tv)
    assert (max(nets) - min(nets)) < 1e-6                  # net_drift plat
    assert all(tvs[i] > tvs[i - 1] for i in range(1, len(tvs)))  # total_var croissant


def test_hf_sweep_precondition_flags_set() -> None:
    """run_hf_amplitude_sweep reporte netdrift_is_flat=True et totalvar_increasing=True."""
    res = run_hf_amplitude_sweep((0.0, 0.04, 0.08, 0.12), k_periods=20, n_seeds=12, steps=100)
    assert res.netdrift_is_flat is True
    assert res.totalvar_increasing is True
    assert res.netdrift_range < res.netdrift_flat_tol


# --- GARDE-FOU : radius_distinct à chaque A ----------------------------------

def test_hf_sweep_all_radius_distinct() -> None:
    """À chaque A les séries r_t diffèrent entre graines (vraie variance de population)."""
    res = run_hf_amplitude_sweep((0.0, 0.06, 0.12), k_periods=20, n_seeds=12, steps=100)
    for p in res.points:
        assert p.radius_distinct is True


# --- DÉTERMINISME bit-à-bit ---------------------------------------------------

def test_hf_amplitude_sweep_determinism_bit_for_bit() -> None:
    """run_hf_amplitude_sweep reproductible : relance ×2 ⇒ tout identique."""
    a = run_hf_amplitude_sweep((0.0, 0.06, 0.12), k_periods=20, n_seeds=10, steps=80)
    b = run_hf_amplitude_sweep((0.0, 0.06, 0.12), k_periods=20, n_seeds=10, steps=80)
    assert [p.delta_median for p in a.points] == [p.delta_median for p in b.points]
    assert [p.delta_vs_fixed for p in a.points] == [p.delta_vs_fixed for p in b.points]
    assert [p.total_var for p in a.points] == [p.total_var for p in b.points]
    assert [p.net_drift for p in a.points] == [p.net_drift for p in b.points]
    assert a.spearman_amp_delta == b.spearman_amp_delta
    assert a.spearman_amp_totalvar == b.spearman_amp_totalvar
