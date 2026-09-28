"""Tests Tour 21 — régulation de l'organe ``regulate_step`` sur dérive RÉELLE (33D ABA).

Confronte ``regulate_step`` (INCHANGÉ) à une dérive issue des vecteurs 33D d'un cycle
ABA réel (voie A de Ra). Deux familles de tests :

  * SANS ``.so`` (toujours exécutés) : pivots et propriétés de ``RealAbaDrive`` /
    ``regulate_step`` qui ne dépendent QUE de facteurs de gain fabriqués à la main —
    pivot ``η=0`` ≡ g-fixe (``torch.equal``), pivot dégénéré (drive constant ⇒ Δ=0),
    porte (0) sur dérives synthétiques (rampe passe, marche stationnaire échoue),
    shuffle déterministe, déterminisme bit-à-bit.

  * AVEC ``.so`` (skip propre si absent, modèle ``test_vector33d_bridge.py``) : le
    pipeline complet sur ``dataset_aba.txt`` — construction du drive depuis le pont
    natif, exécution de l'ordre lexicographique, finitude, déterminisme du verdict.

L'organe et le canon ne sont JAMAIS touchés ; tout est seedé et déterministe.
"""
import math
from pathlib import Path

import pytest
import torch

from spiraton.experimental.edge_controller import (
    EdgeController,
    GainDrift,
    ProcessNoise,
    SeededDrift,
    regulate_step,
    run_fixed_gain,
)
from spiraton.data.tokenizer_bridge import is_available, NativeTokenizer33D
from spiraton.diagnostics.aba_regulation import (
    AbaRegulationReport,
    RealAbaDrive,
    BASE_GAIN,
    GAIN_SPAN,
    OMEGA,
    G_MIN,
    G_MAX,
    G0,
    ETA,
    DELTA_DRIFT,
    RATIO_DRIFT,
    _make_projector,
    _project_to_gain,
    _seed_s0_aba,
    _run_pair,
    run_aba_regulation,
)


_DATASET = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
# Repli portable : si le chemin absolu n'existe pas, chercher à la racine du dépôt.
if not _DATASET.is_file():
    _DATASET = Path(__file__).resolve().parents[2] / "dataset_aba.txt"

native_available = is_available()


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _hand_drive(factors, n_seg=None) -> RealAbaDrive:
    """Drive fabriqué à la main (sans ``.so``) — facteurs de gain explicites."""
    n = len(factors)
    bounds = n_seg if n_seg is not None else (n, n, n)
    return RealAbaDrive(factors=tuple(factors), seg_bounds=bounds, n_tokens=n)


# =============================================================================
# Famille SANS .so : pivots et propriétés structurelles (toujours exécutés)
# =============================================================================

# --- PIVOT (i) : η = 0 ≡ g-fixe bit-à-bit ------------------------------------

def test_pivot_eta_zero_equals_fixed_gain_bit_for_bit() -> None:
    """``η=0`` (organe inerte) reproduit ``run_fixed_gain`` BIT-À-BIT sous la même dérive."""
    drive = _hand_drive([0.95, 1.02, 0.98, 1.05, 1.01, 0.99, 1.03, 0.97])
    s0 = _seed_s0_aba(0)
    steps = drive.n_tokens
    ctrl0 = EdgeController(omega=OMEGA, g0=1.0, eta=0.0, g_min=G_MIN, g_max=G_MAX)
    t_organe0 = ctrl0.run(s0, steps=steps, drift=drive).trace
    t_fixed = run_fixed_gain(s0, steps=steps, g_fixed=1.0, omega=OMEGA, drift=drive).trace
    assert torch.equal(t_organe0, t_fixed)


def test_regulate_step_byte_identity_preserved() -> None:
    """``regulate_step`` reste la loi T15 mot pour mot (clip(prev − η(obs−target)))."""
    assert regulate_step(1.0, 1.2, 1.0, 0.5, 0.8, 1.2) == 1.0 - 0.5 * (1.2 - 1.0)
    # clip haut et bas
    assert regulate_step(1.19, 0.0, 1.0, 0.5, 0.8, 1.2) == 1.2
    assert regulate_step(0.81, 2.0, 1.0, 0.5, 0.8, 1.2) == 0.8
    # η=0 ⇒ inerte
    assert regulate_step(0.93, 5.0, 1.0, 0.0, 0.8, 1.2) == 0.93


# --- PIVOT dégénéré : drive constant ⇒ Δ = 0 ---------------------------------

def test_pivot_constant_drive_gives_zero_delta() -> None:
    """Drive constant (un seul vecteur 33D répété) ⇒ Δf_edge = 0 (rien à réguler)."""
    drive = _hand_drive([BASE_GAIN] * 12)
    s0 = _seed_s0_aba(3)
    r_ctrl, r_fixed = _run_pair(drive, s0, BASE_GAIN)
    assert r_ctrl.f_edge - r_fixed.f_edge == 0.0
    assert drive.net_drift() == 0.0
    assert drive.total_var() == 0.0


# --- PORTE (0) sur dérives SYNTHÉTIQUES : ancrage des baselines ---------------

def test_gate_ratio_ramp_is_one() -> None:
    """Une rampe pure (régime P1/T15-T18) a un ratio net/total = 1.0 (non-stationnaire)."""
    T = 60
    g = GainDrift(0.95, 1.10)
    vals = [g.at(t, T) for t in range(T)]
    drive = _hand_drive(vals)
    assert drive.total_var() > 0.0
    ratio = drive.net_drift() / drive.total_var()
    assert abs(ratio - 1.0) < 1e-9
    assert ratio > RATIO_DRIFT   # une rampe PASSE la porte


def test_gate_ratio_ar1_zero_mean_is_stationary() -> None:
    """Un AR(1) moyenne-nulle (P2/T16, régime RÉFUTÉ) a un ratio net/total ≪ seuil."""
    T = 200
    p = ProcessNoise.from_seed(0, steps=T, phi=0.5, sigma=0.04)
    vals = [p.at(t, T) for t in range(T)]
    drive = _hand_drive(vals)
    ratio = drive.net_drift() / drive.total_var()
    assert ratio < RATIO_DRIFT   # une marche stationnaire ÉCHOUE à la porte (NULL honnête)


# --- shuffle déterministe (ordre détruit) ------------------------------------

def test_shuffle_is_deterministic_and_permutation() -> None:
    """Le shuffle est seedé (reproductible) et est une vraie permutation (multiset égal)."""
    drive = _hand_drive([0.9, 0.95, 1.0, 1.05, 1.1, 1.0, 0.92, 0.97])
    sh1 = drive.shuffled(123)
    sh2 = drive.shuffled(123)
    assert sh1.factors == sh2.factors
    assert sorted(sh1.factors) == sorted(drive.factors)
    # une graine différente donne (en général) un ordre différent
    sh3 = drive.shuffled(124)
    assert sh3.factors != sh1.factors or len(set(drive.factors)) <= 1


def test_shuffle_preserves_net_and_total_endpoints_invariants() -> None:
    """Le shuffle conserve la variation TOTALE bornée mais change la dérive nette/ordre."""
    drive = _hand_drive([0.90, 0.93, 0.96, 1.00, 1.05, 1.10])  # monotone : net = total
    sh = drive.shuffled(7)
    # multiset conservé ⇒ min/max conservés ⇒ net_drift ≤ (max−min) des deux côtés
    span = max(drive.factors) - min(drive.factors)
    assert drive.net_drift() <= span + 1e-9
    assert sh.net_drift() <= span + 1e-9


# --- déterminisme bit-à-bit de la projection canon ---------------------------

def test_projector_is_deterministic_and_bounded() -> None:
    """La cellule canon de projection est seedée (reproductible) et bornée par GAIN_SPAN.

    NB : un vecteur NUL ne donne PAS BASE_GAIN exactement — les branches mul/div du
    canon sont en log-domaine (``log(|x|+eps)``), non nulles même à x=0 (point de
    vigilance documenté du canon). On vérifie donc le déterminisme et la BORNE, pas un
    centrage exact. Le seul gain BASE_GAIN exact est garanti via ``RealAbaDrive`` à
    facteurs constants (cf. pivot dégénéré), pas via l'input nul.
    """
    c1 = _make_projector(20210601)
    c2 = _make_projector(20210601)
    assert torch.equal(c1.w_add, c2.w_add)
    assert torch.equal(c1.w_mul, c2.w_mul)
    # déterminisme de la projection elle-même
    v0 = torch.arange(15, dtype=torch.float32) * 0.1
    assert _project_to_gain(c1, v0) == _project_to_gain(c2, v0)
    # sortie bornée dans [BASE_GAIN−GAIN_SPAN, BASE_GAIN+GAIN_SPAN] (tanh borné)
    g = torch.Generator().manual_seed(5)
    for _ in range(20):
        v = torch.randn(15, generator=g) * 3.0
        val = _project_to_gain(c1, v)
        assert BASE_GAIN - GAIN_SPAN - 1e-6 <= val <= BASE_GAIN + GAIN_SPAN + 1e-6


# =============================================================================
# Famille AVEC .so : pipeline complet (skip propre si lib absente)
# =============================================================================

@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_real_drive_build_and_finite() -> None:
    """Le drive réel se construit (dims 8-22), facteurs finis et bornés."""
    from spiraton.diagnostics.aba_regulation import build_real_drive, collect_real_drives

    tok = NativeTokenizer33D()
    cell = _make_projector()
    drives = collect_real_drives(str(_DATASET), tok, cell, n_cycles=5)
    assert len(drives) == 5
    for d in drives:
        assert d.n_tokens >= 4
        for f in d.factors:
            assert math.isfinite(f)
            assert BASE_GAIN - GAIN_SPAN - 1e-6 <= f <= BASE_GAIN + GAIN_SPAN + 1e-6
        # seg_bounds croissants et cohérents avec n_tokens
        assert 0 < d.seg_bounds[0] <= d.seg_bounds[1] <= d.seg_bounds[2] == d.n_tokens


@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_real_pivot_eta_zero_bit_for_bit() -> None:
    """Sur un drive RÉEL : ``η=0`` ≡ g-fixe bit-à-bit (le pivot tient sur données réelles)."""
    from spiraton.diagnostics.aba_regulation import build_real_drive
    from spiraton.data.aba import iter_aba_cycles

    tok = NativeTokenizer33D()
    cell = _make_projector()
    cycle = next(iter_aba_cycles(str(_DATASET)))
    drive = build_real_drive(cycle, tok, cell)
    s0 = _seed_s0_aba(0)
    steps = drive.n_tokens
    ctrl0 = EdgeController(omega=OMEGA, g0=G0, eta=0.0, g_min=G_MIN, g_max=G_MAX)
    t0 = ctrl0.run(s0, steps=steps, drift=drive).trace
    tf = run_fixed_gain(s0, steps=steps, g_fixed=G0, omega=OMEGA, drift=drive).trace
    assert torch.equal(t0, tf)


@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_run_aba_regulation_deterministic_and_gate_stationary() -> None:
    """Le rapport complet est déterministe ET la porte (0) déclare la dérive stationnaire.

    Résultat MESURÉ (jamais forcé) : la dérive 33D réelle est stationnaire (ratio
    net/total ≈ 0.11 ≪ 0.5), donc le verdict est NULL-stationnaire — cohérent T16/P2.
    On vérifie le déterminisme (deux runs identiques) et la cohérence interne (pivot
    dégénéré présent : delta_shuffle fini). Le verdict exact est documenté, pas exigé
    comme "succès".
    """
    tok = NativeTokenizer33D()
    r1 = run_aba_regulation(str(_DATASET), tok, n_cycles=40)
    r2 = run_aba_regulation(str(_DATASET), tok, n_cycles=40)
    assert isinstance(r1, AbaRegulationReport)
    # déterminisme bit-à-bit des quantités décisionnelles
    assert r1.ratio_drift_median == r2.ratio_drift_median
    assert r1.delta_real_median == r2.delta_real_median
    assert r1.delta_shuffle_median == r2.delta_shuffle_median
    assert r1.verdict == r2.verdict
    # finitude des médianes
    assert math.isfinite(r1.ratio_drift_median)
    assert math.isfinite(r1.delta_real_median)
    assert math.isfinite(r1.ao_l2_ctrl_median)
    # la dérive réelle est stationnaire (mesure) ⇒ porte non franchie ⇒ NULL-stationnaire
    assert r1.ratio_drift_median < RATIO_DRIFT
    assert not r1.gate_passed
    assert r1.verdict == "NULL-stationnaire"


@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_gate_verdict_robust_across_projection_seeds() -> None:
    """Le verdict NULL-stationnaire ne dépend PAS de la graine de projection (anti-artefact).

    Le net_drift ABSOLU varie avec proj_seed (amplitude de projection), mais le RATIO
    sans échelle reste ≪ 0.5 pour toute graine ⇒ verdict invariant (honnêteté du gate).
    """
    tok = NativeTokenizer33D()
    for ps in (20210601, 1, 12345):
        r = run_aba_regulation(str(_DATASET), tok, n_cycles=40, proj_seed=ps)
        assert r.ratio_drift_median < RATIO_DRIFT
        assert r.verdict == "NULL-stationnaire"
