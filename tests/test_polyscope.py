"""Tests Tour 6 — Polyscope2D (alternance non-commutative) + corner_signature + angular_separability.

Quatuor canon adapté (seeds fixés, finitude, formes simple/batch, formule exacte sous
matrices forcées, flux de gradient) + tests ANTI-ARTEFACT DURS :
  * ‖[W_a, W_b]‖ = 0  ⇒  cornerness S'EFFONDRE au niveau orbite-lisse (le coin est PORTÉ
    par la non-commutativité, pas par la rotation seule — parallèle au CTRL D=L du Tour 1) ;
  * baseline linéaire NE trace PAS d'anguleux-vrai ;
  * CTRL-PERM : l'AUC réelle dépasse le 95e pct sous H0.
+ déterminisme bit-à-bit. La formule α-ω, ShapeSignature et spectral_signature ne sont PAS
touchées (corner_signature est ADDITIF).

REFUS incarné : aucune forme dessinée à la main (émergence de l'alternance) ; commutator_norm
(dormant) réveillé ; les baselines (linéaire + commutateur=0) sont obligatoires ; si une
issue NULL sort par forme, on l'assert telle quelle.
"""
import math

import torch
import pytest

from spiraton.experimental.polyscope import (
    Polyscope2D,
    anisotropic_shear,
    reflection,
)
from spiraton.experimental.matrix_cell import commutator_norm
from spiraton.diagnostics.shape_signature import (
    CornerSignature,
    corner_signature,
    # intacts — on vérifie qu'ils existent encore :
    ShapeSignature,
    shape_signature,
    spectral_signature,
)
from spiraton.diagnostics.angular_separability import (
    run_angular_separability,
    build_population,
    _s0_unit,
)
from spiraton.diagnostics.spectral_separability import _rank_auc, _abs_dir


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# --- formes simple / batch + finitude ----------------------------------------

@pytest.mark.parametrize("steps", [12, 40])
def test_trace_shape_and_finite(steps: int) -> None:
    cell = Polyscope2D.angular(gain=1.05)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=steps)
    assert tr.shape == (steps + 1, 2)
    assert _finite(tr)


def test_forward_single_and_batch() -> None:
    cell = Polyscope2D.angular(gain=1.05)
    s0 = torch.tensor([1.0, 0.0])
    sf, tr = cell(s0, steps=20, return_trace=True)
    assert sf.shape == (2,)
    assert tr.shape == (21, 2)
    assert torch.allclose(sf, tr[-1])
    # batch (état final seulement)
    s0b = torch.randn(5, 2)
    out = cell(s0b, steps=20)
    assert out.shape == (5, 2)
    assert _finite(out)


def test_batch_matches_single_trajectory() -> None:
    """La trajectoire batch d'un point == la trajectoire singleton (cohérence d'alternance)."""
    cell = Polyscope2D.angular(gain=1.05)
    s0 = torch.tensor([0.3, -0.7])
    out_single = cell(s0, steps=15)
    out_batch = cell(s0.unsqueeze(0), steps=15)
    assert torch.allclose(out_single, out_batch.squeeze(0), atol=1e-5)


# --- formule exacte sous matrices forcées ------------------------------------

def test_alternation_program_exact() -> None:
    """Le programme W(t) applique W_a sur k_a pas puis W_b sur k_b pas (période k_a+k_b)."""
    W_a = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    W_b = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    cell = Polyscope2D(W_a, W_b, k_a=2, k_b=1)
    s0 = torch.tensor([1.0, 1.0])
    tr = cell.trace(s0, steps=3)
    # t=0 (W_a) : (2,1) ; t=1 (W_a) : (4,1) ; t=2 (W_b) : (4,3)
    assert torch.allclose(tr[1], torch.tensor([2.0, 1.0]))
    assert torch.allclose(tr[2], torch.tensor([4.0, 1.0]))
    assert torch.allclose(tr[3], torch.tensor([4.0, 3.0]))


def test_anisotropic_shear_is_area_preserving() -> None:
    """shear(gain, φ) = R(φ)·diag(gain,1/gain)·R(φ)ᵀ : det=1, valeurs propres gain et 1/gain."""
    for gain in (1.1, 1.5, 2.0):
        for phi in (0.0, 0.5, 1.2):
            W = anisotropic_shear(gain, axis_angle=phi)
            assert abs(float(torch.det(W)) - 1.0) < 1e-5
            ev = torch.linalg.eigvals(W).real.sort().values
            assert abs(float(ev[0]) - 1.0 / gain) < 1e-4
            assert abs(float(ev[1]) - gain) < 1e-4


def test_commutator_zero_iff_same_axis() -> None:
    """Deux shears commutent SSI mêmes axes propres : même axe ⇒ ‖[·,·]‖=0 ; axes ≠ (≠π/2) ⇒ >0."""
    W0 = anisotropic_shear(1.3, axis_angle=0.0)
    W0b = anisotropic_shear(1.7, axis_angle=0.0)  # même axe, gain différent
    assert float(commutator_norm(W0, W0b)) < 1e-5      # commutent
    W1 = anisotropic_shear(1.3, axis_angle=1.0)        # axe différent (≠ 0, ≠ π/2)
    assert float(commutator_norm(W0, W1)) > 1e-3       # ne commutent pas


def test_factory_commutators() -> None:
    """angular : ‖·‖>0 ; commuting : ‖·‖=0 exact ; cross D4 : ‖·‖>>0 ; ngon : ‖·‖=0 (vs I)."""
    assert Polyscope2D.angular(gain=1.05).commutator() > 0.0
    assert Polyscope2D.commuting(gain=1.05).commutator() == 0.0
    assert Polyscope2D.cross().commutator() > 1.0          # diédral non-abélien
    assert Polyscope2D.ngon_vertices(n=4).commutator() == 0.0


def test_reflection_is_involution_det_minus_one() -> None:
    """Réflexion : det=−1 et Ref² = I (involution)."""
    for phi in (0.0, 0.4, math.pi / 4):
        R = reflection(phi)
        assert abs(float(torch.det(R)) + 1.0) < 1e-6
        assert torch.allclose(R @ R, torch.eye(2), atol=1e-6)


# --- flux de gradient (l'alternance de cartes linéaires est différentiable) ---

def test_gradient_flows_through_alternation() -> None:
    cell = Polyscope2D.angular(gain=1.03)
    s0 = torch.tensor([[0.5, 0.5]], requires_grad=True)
    out = cell(s0, steps=8)
    out.sum().backward()
    assert s0.grad is not None
    assert float(s0.grad.abs().sum()) > 0.0


# --- corner_signature : sanité géométrique -----------------------------------

def test_corner_signature_straight_line_no_corner() -> None:
    """Une dilatation pure (orbite droite) : cornerness ≈ 0, straight_edge_fraction = 1."""
    W = torch.tensor([[1.05, 0.0], [0.0, 1.05]])  # isotrope : ligne radiale droite
    cell = Polyscope2D(W, torch.eye(2), k_a=1, k_b=0)
    s0 = torch.tensor([1.0, 0.3])
    cs = corner_signature(cell.trace(s0, steps=40))
    assert cs.cornerness < 1.0
    assert cs.straight_edge_fraction > 0.9


def test_corner_signature_ngon_vertices_uniform_curvature() -> None:
    """n-gone-de-SOMMETS = R(2π/n) : courbure UNIFORME ⇒ P95=médiane ⇒ cornerness ≈ 0 (H6a)."""
    cs = corner_signature(Polyscope2D.ngon_vertices(n=4).trace(torch.tensor([1.0, 0.0]), steps=40))
    assert cs.cornerness < 1.0                  # pas d'anguleux-VRAI (pas d'arête droite)
    assert abs(cs.median_kappa - cs.p95_kappa) < 0.1   # courbure constante = π/2 par pas


def test_corner_signature_angular_is_truly_angular() -> None:
    """angular (double-shear axes ≠) : cornerness >> 5 (arêtes droites + coins francs)."""
    cs = corner_signature(Polyscope2D.angular(gain=1.05).trace(torch.tensor([1.0, 0.0]), steps=40))
    assert cs.cornerness >= 5.0
    assert cs.n_corners >= 1


def test_corner_signature_finite_fields() -> None:
    cs = corner_signature(Polyscope2D.angular(gain=1.05).trace(torch.tensor([1.0, 0.0]), steps=40))
    assert math.isfinite(cs.cornerness)
    assert 0.0 <= cs.straight_edge_fraction <= 1.0
    assert cs.self_intersections >= 0


# --- ANTI-ARTEFACT DUR : ‖[W_a,W_b]‖=0 ⇒ cornerness s'effondre ----------------

def test_commutator_zero_collapses_cornerness() -> None:
    """LE test du Tour 6 : à commutateur nul exact, la cornerness s'effondre au niveau lisse.

    MÊME geste (double-shear, même gain, même k_edge), seule différence : axes alignés
    (commutent) vs axes désalignés (ne commutent pas). L'anguleux-vrai a C >> 5 ; le
    contrôle commutateur=0 a C d'un ORDRE DE GRANDEUR plus bas. Preuve que c'est la
    NON-COMMUTATIVITÉ qui porte le coin (et non la simple présence de deux opérateurs).
    """
    s0 = torch.tensor([1.0, 0.0])
    ang = corner_signature(Polyscope2D.angular(gain=1.05).trace(s0, steps=40))
    com = corner_signature(Polyscope2D.commuting(gain=1.05).trace(s0, steps=40))
    assert Polyscope2D.angular(gain=1.05).commutator() > 0.0
    assert Polyscope2D.commuting(gain=1.05).commutator() == 0.0
    assert ang.cornerness >= 5.0                       # anguleux-vrai
    assert com.cornerness < 0.2 * ang.cornerness       # effondrement (ordre de grandeur)
    assert com.straight_edge_fraction > 0.9            # le contrôle est une orbite droite


def test_baseline_linear_rarely_truly_angular() -> None:
    """La baseline linéaire (matrice unique) trace RAREMENT un anguleux-vrai (C>=5).

    Une seule carte linéaire à λ complexes a une courbure UNIFORME : elle ne peut pas
    avoir le contraste arête-droite/coin-franc. Sur 40 graines, peu passent C>=5, bien
    en-deçà du réglage anguleux (toujours)."""
    n_pass = 0
    for seed in range(40):
        from spiraton.experimental.oscilloscope import Oscilloscope2D, InputSignal
        g = torch.Generator().manual_seed(seed)
        cell = Oscilloscope2D.random(g, scale=1.05)
        cs = corner_signature(cell.trace(_s0_unit(seed), steps=40, signal=InputSignal(kind="zero")))
        if cs.cornerness >= 5.0 and not cs.degenerate_median:
            n_pass += 1
    assert n_pass <= 8   # << 40 (le réglage anguleux passe systématiquement)


# --- ANTI-ARTEFACT : CROIX D4 passe par le centre et se recroise --------------

def test_cross_passes_through_center_and_recrosses() -> None:
    """La croix D4 (s0 sur l'axe) : passe plusieurs fois près du centre et se recroise (H6c)."""
    cs = corner_signature(Polyscope2D.cross().trace(torch.tensor([1.0, 0.0]), steps=80))
    assert cs.min_center_ratio < 0.2          # passe TRÈS près du centre
    assert cs.n_center_passes >= 2            # passages-centre récurrents
    assert cs.self_intersections >= 1         # se recroise


def test_cross_is_bounded() -> None:
    """La croix (|val propre|<1 + rotation isométrique) est BORNÉE — figure fermée-stable."""
    tr = Polyscope2D.cross().trace(torch.tensor([1.0, 0.0]), steps=80)
    assert _finite(tr)
    radii = torch.linalg.vector_norm(tr, dim=-1)
    assert float(radii.max()) < 5.0


# --- RAPPORT COMPLET : verdicts, AUC, CTRL-PERM ------------------------------

def test_report_polygon_issue_a_cornerness_carried_by_noncommutativity() -> None:
    """VERDICT POLYGONE = (a) : cornerness élevée ET s'effondre au CTRL comm=0, > CTRL-PERM."""
    rep = run_angular_separability()
    assert rep.corn_angular[1] >= 5.0                      # anguleux-vrai (médiane)
    assert rep.corn_commuting[1] < rep.corn_angular[1]     # le contrôle s'effondre
    assert _abs_dir(rep.auc_ang_vs_commuting.auc) >= 0.95
    assert rep.auc_ang_vs_commuting.above_ctrl is True     # > 95e pct CTRL-PERM
    assert rep.auc_ang_vs_baseline.above_ctrl is True      # bat la baseline linéaire
    assert rep.issue_polygon == "a"


def test_report_commutators_separated() -> None:
    """Covariable axiome 1 : angular ‖·‖>0, commuting ‖·‖=0 exact, cross ‖·‖>>0."""
    rep = run_angular_separability()
    assert rep.comm_angular > 0.0
    assert rep.comm_commuting == 0.0
    assert rep.comm_cross > 1.0


def test_report_cross_recrosses_above_baseline() -> None:
    """CROIX : recroisement + passage-centre dépassent la baseline linéaire (selfX médian >> 0)."""
    rep = run_angular_separability()
    # selfX médian de la croix >> baseline (la magnitude du recroisement est le discriminant)
    assert rep.cross_self_intersections[1] > rep.base_self_intersections[1]
    assert rep.frac_cross_recrosses > rep.frac_base_recrosses
    assert rep.issue_cross == "a-non-comm"


def test_report_closure_axis_reported_separately() -> None:
    """La fermeture/stabilité est mesurée À PART : angular dérive (bornée), cross fermée-stable."""
    rep = run_angular_separability()
    # angular : polygone-spirale (dérive bornée) — final_norm fini, frac bornée = 1.
    assert rep.frac_angular_bounded == 1.0
    # cross : bornée (figure fermée-stable, final_norm petit).
    assert rep.frac_cross_bounded == 1.0
    assert rep.final_norm_cross[1] < rep.final_norm_angular[2]


def test_population_size_and_finiteness() -> None:
    members = build_population(n_seeds=40)
    assert len(members) == 200  # 40 × 5 origines
    for m in members:
        assert math.isfinite(m.sig.cornerness)
        assert m.commutator >= 0.0


# --- DÉTERMINISME bit-à-bit (2 runs) -----------------------------------------

def test_report_deterministic_bit_for_bit() -> None:
    r1 = run_angular_separability()
    r2 = run_angular_separability()
    assert r1.auc_ang_vs_commuting.auc == r2.auc_ang_vs_commuting.auc
    assert r1.auc_ang_vs_baseline.auc == r2.auc_ang_vs_baseline.auc
    assert r1.auc_cross_selfX_vs_baseline.auc == r2.auc_cross_selfX_vs_baseline.auc
    assert r1.corn_angular == r2.corn_angular
    assert r1.issue_polygon == r2.issue_polygon
    assert r1.issue_cross == r2.issue_cross


def test_trace_deterministic_bit_for_bit() -> None:
    s0 = torch.tensor([0.3, -0.4])
    a = Polyscope2D.angular(gain=1.05).trace(s0, steps=40)
    b = Polyscope2D.angular(gain=1.05).trace(s0, steps=40)
    assert torch.equal(a, b)


# --- intégrité : shape_signature / spectral_signature NON touchés -------------

def test_shape_and_spectral_signature_still_intact() -> None:
    """corner_signature est ADDITIF : ShapeSignature et spectral_signature restent inchangés."""
    from spiraton.experimental.oscilloscope import Oscilloscope2D, InputSignal
    s0 = torch.tensor([1.0, 0.0])
    tr = Oscilloscope2D.circle(omega=math.pi / 5).trace(s0, steps=120, signal=InputSignal(kind="zero"))
    sig = shape_signature(tr, s0)
    assert sig.cv_r < 0.05
    assert sig.passes_circle_guards()
    spec = spectral_signature(Oscilloscope2D.circle(omega=math.pi / 5).A)
    assert abs(spec.rho - 1.0) < 1e-6
