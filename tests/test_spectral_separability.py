"""Tests Tour 5 (E2) — signature spectrale + séparabilité des classes de forme.

Quatuor canon adapté à un DIAGNOSTIC spectral (seeds fixés, finitude, formes
simple/compagnon, formule exacte des valeurs propres sous matrices forcées) +
tests anti-artefact DURS (CTRL-PERM : l'AUC réelle dépasse le 95e pct sous H0 ;
CTRL-SIG-RAND ≈ 0.5 ; l'AUC sur étiquettes permutées NE dépasse PAS le seuil) +
déterminisme bit-à-bit. La formule α-ω et ``ShapeSignature`` ne sont PAS touchées.

REFUS incarné : aucune étiquette de forme ni (ω, g) ne touche le calcul de
signature ; le seuil/AUC vit sur le nuage mélangé ; l'issue (b) est un résultat
légitime, pas un échec — on l'assert telle quelle.
"""
import math

import torch
import pytest

from spiraton.experimental.oscilloscope import Oscilloscope2D, rotation_matrix, InputSignal
from spiraton.diagnostics.shape_signature import (
    SpectralSignature,
    companion_matrix,
    spectral_signature,
    shape_signature,  # non modifié — on vérifie qu'il existe encore
)
from spiraton.diagnostics.spectral_separability import (
    _rank_auc,
    _abs_dir,
    _ctrl_perm_distribution,
    _ctrl_sig_rand_auc,
    build_population,
    collect_baseline_spirals,
    run_spectral_separability,
)

OMEGA = math.pi / 5


def _finite(x: float) -> bool:
    return math.isfinite(x)


# --- formule exacte des valeurs propres sous matrices forcées ----------------

def test_spectral_signature_circle_is_unit_modulus_complex() -> None:
    """A = R(ω) : valeurs propres e^{±iω} ⇒ ρ=1, α_eig=ω, complexes conjuguées.

    Tolérance 1e-6 : ``rotation_matrix`` (et le buffer ``model.A``) sont en float32 ;
    le rounding float32 (~1e-7) précède la diagonalisation float64. C'est honnête :
    on lit la matrice TELLE QU'ELLE EST stockée, pas une version recalculée en double.
    """
    A = rotation_matrix(OMEGA)
    sig = spectral_signature(A)
    assert abs(sig.rho - 1.0) < 1e-6
    assert abs(sig.alpha_eig - OMEGA) < 1e-6
    assert sig.is_complex is True


def test_spectral_signature_spiral_is_gain_modulus() -> None:
    """A = g·R(ω) : ρ=g exactement, α_eig=ω, complexes (le gain ne tue pas la rotation)."""
    g = 1.06
    A = g * rotation_matrix(OMEGA)
    sig = spectral_signature(A)
    assert abs(sig.rho - g) < 1e-6
    assert abs(sig.alpha_eig - OMEGA) < 1e-6
    assert sig.is_complex is True


def test_spectral_signature_real_diagonal_is_real() -> None:
    """A diagonale réelle : valeurs propres réelles ⇒ α_eig=0, is_complex=False."""
    A = torch.diag(torch.tensor([0.9, 1.3]))
    sig = spectral_signature(A)
    assert abs(sig.rho - 1.3) < 1e-6
    assert sig.alpha_eig < 1e-6
    assert sig.is_complex is False


def test_spectral_signature_pure_expansion_no_rotation() -> None:
    """ω=0, gain>1 : A = g·I, valeurs propres réelles g (pas de rotation)."""
    cell = Oscilloscope2D(omega=0.0, gain=1.1)
    sig = spectral_signature(cell.A)
    assert abs(sig.rho - 1.1) < 1e-6
    assert sig.alpha_eig < 1e-6
    assert sig.is_complex is False


# --- matrice compagnon (leçon Tour 1 : A seule n'est pas la bonne matrice) ----

def test_companion_reduces_to_A_when_memory_zero() -> None:
    """memory=0 ⇒ compagnon = A (le bloc nilpotent découplé est inutile)."""
    A = 1.06 * rotation_matrix(OMEGA)
    M = companion_matrix(A, 0.0)
    assert M.shape == (2, 2)
    assert torch.allclose(M.to(torch.float64), A.to(torch.float64))


def test_companion_shape_and_block_structure() -> None:
    """memory≠0 ⇒ compagnon 4×4 = [[A, −cI], [I, 0]] (structure exacte)."""
    A = 1.06 * rotation_matrix(OMEGA)
    c = 0.3
    M = companion_matrix(A, c).to(torch.float64)
    assert M.shape == (4, 4)
    Ad = A.to(torch.float64)
    assert torch.allclose(M[:2, :2], Ad)
    assert torch.allclose(M[:2, 2:], -c * torch.eye(2, dtype=torch.float64))
    assert torch.allclose(M[2:, :2], torch.eye(2, dtype=torch.float64))
    assert torch.allclose(M[2:, 2:], torch.zeros(2, 2, dtype=torch.float64))


def test_companion_spectrum_governs_second_order_growth() -> None:
    """ρ(compagnon) prédit le taux de croissance de s_{t+1}=A·s_t−c·s_{t−1}.

    Lire A seule (ρ=1.06) sous-estimerait : c'est l'erreur du Tour 1 que la
    compagnon corrige. On déroule la vraie récurrence et compare le taux empirique
    au ρ de la matrice compagnon.
    """
    g, c = 1.06, 0.3
    A = (g * rotation_matrix(OMEGA)).to(torch.float64)
    sig = spectral_signature(A, memory=c)
    s_prev = torch.zeros(2, dtype=torch.float64)
    s = torch.tensor([1.0, 0.0], dtype=torch.float64)
    norms = []
    for _ in range(200):
        nxt = A @ s - c * s_prev
        s_prev, s = s, nxt
        norms.append(float(s.norm()))
    emp_growth = (norms[-1] / norms[-50]) ** (1.0 / 50.0)
    assert abs(emp_growth - sig.rho) < 1e-2
    # et ρ(compagnon) > ρ(A seule) : la mémoire AJOUTE au gain ici (preuve que A
    # seule serait fausse).
    assert sig.rho > spectral_signature(A, memory=0.0).rho


# --- finitude de toutes les signatures de la population ----------------------

def test_population_signatures_are_finite() -> None:
    members = build_population(n_seeds=40)
    assert len(members) == 120  # 40 cercles + 40 spirales + 40 baselines
    for m in members:
        assert _finite(m.sig.rho)
        assert _finite(m.sig.alpha_eig)
        assert m.sig.rho >= 0.0
        assert 0.0 <= m.sig.alpha_eig <= math.pi + 1e-9


def test_signature_reads_only_model_A_same_status() -> None:
    """circle/spiral/random : signature lue sur model.A au MÊME statut (pas (ω,g))."""
    g = torch.Generator().manual_seed(0)
    for cell in (
        Oscilloscope2D.circle(omega=OMEGA),
        Oscilloscope2D.spiral(omega=OMEGA, gain=1.06),
        Oscilloscope2D.random(g, scale=0.6),
    ):
        sig = spectral_signature(cell.A, memory=cell.cfg.memory)
        # même appel, mêmes champs, aucune branche par origine.
        assert isinstance(sig, SpectralSignature)
        assert _finite(sig.rho)


# --- AUC maison : sanité (rang = Mann-Whitney) -------------------------------

def test_rank_auc_perfect_and_chance() -> None:
    """Séparation parfaite ⇒ AUC=1 ; groupes identiques ⇒ AUC=0.5."""
    assert abs(_rank_auc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-12
    assert abs(_rank_auc([0, 1, 2], [3, 4, 5]) - 0.0) < 1e-12  # direction conservée
    a = [1.0, 2.0, 3.0, 4.0]
    assert abs(_rank_auc(a, list(a)) - 0.5) < 1e-12


# --- ANTI-ARTEFACT DUR 1 : CTRL-PERM — l'AUC permutée ne sépare pas ----------

def test_ctrl_perm_under_null_is_near_half() -> None:
    """Sous permutation des étiquettes, la force discriminante reste basse (~0.5).

    Avec des scores RÉELLEMENT séparés mais des étiquettes PERMUTÉES, l'AUC ne doit
    PLUS séparer : la médiane de la force |auc−0.5|+0.5 est proche de 0.5 et le 95e
    pct reste bien en-deçà de 1. C'est le test anti-artefact dur : une AUC réelle ne
    compte que si elle dépasse ce 95e pct.
    """
    scores = [float(i) for i in range(40)]          # parfaitement ordonnés
    labels = [1] * 20 + [0] * 20                      # vraie séparation
    # AUC réelle (non permutée) = parfaite
    real = _rank_auc(scores[:20], scores[20:])
    assert _abs_dir(real) > 0.99
    med, p95, dist = _ctrl_perm_distribution(scores, labels, n_perms=50, seed0=123)
    assert med < 0.65          # médiane sous H0 proche de 0.5
    assert p95 < 0.85          # 95e pct bien en-deçà de la séparation parfaite
    assert _abs_dir(real) > p95  # l'AUC réelle DÉPASSE le 95e pct de H0


# --- ANTI-ARTEFACT DUR 2 : CTRL-SIG-RAND ≈ 0.5 ------------------------------

def test_ctrl_sig_rand_is_near_half() -> None:
    """Discriminateur sur signatures aléatoires de même support ⇒ AUC≈0.5 (moyennée)."""
    auc = _ctrl_sig_rand_auc(20, 40, (0.0, 1.06), seed=999, n_rep=300)
    assert abs(auc - 0.5) < 0.05


# --- baseline-qui-spirale : collecte déterministe ----------------------------

def test_collect_baseline_spirals_deterministic_and_shape_based() -> None:
    """Les baselines-qui-spiralent sont rares, déterministes, et qualifiées par FORME."""
    a = collect_baseline_spirals(scale=0.6, target=5, reservoir=5000)
    b = collect_baseline_spirals(scale=0.6, target=5, reservoir=5000)
    assert len(a) == 5
    assert [m.seed for m in a] == [m.seed for m in b]          # déterministe
    assert all(m.is_spiral_shape for m in a)                    # qualifiées par forme
    # toutes ces baselines spiralent ⇒ valeurs propres complexes et ρ>1 (rotation
    # sortante) : pont de cohérence avec la signature de forme.
    for m in a:
        assert m.sig.is_complex
        assert m.sig.rho > 1.0


# --- RAPPORT COMPLET : verdict, contrôles, issue (b) -------------------------

def test_full_report_test1_separates_above_ctrl() -> None:
    """TEST 1 : ρ sépare cercle/spirale (AUC≥0.90) ET dépasse le 95e pct CTRL-PERM."""
    rep = run_spectral_separability()
    assert _abs_dir(rep.auc_test1) >= 0.90
    assert rep.test1_above_ctrl is True
    assert _abs_dir(rep.auc_test1) > rep.ctrl_perm_p95_test1
    # CTRL-SIG-RAND du test 1 ≈ 0.5 (le protocole est non-biaisé).
    assert abs(rep.ctrl_sig_rand_test1 - 0.5) < 0.05


def test_full_report_test2_spiral_not_separable_issue_b() -> None:
    """TEST 2 : spirale-réglage vs baseline-qui-spirale NON séparable (issue b).

    RÉSULTAT LÉGITIME, PAS UN ÉCHEC (REFUS) : l'AUC vaut ≈0.5 et ne dépasse PAS le
    95e pct de CTRL-PERM. On l'assert TELLE QUELLE — surtout si elle vaut 0.5.
    """
    rep = run_spectral_separability()
    assert rep.n_baseline_spiral >= 5
    # AUC≈0.5 (±0.15 : petit échantillon) et NE dépasse PAS CTRL-PERM.
    assert abs(_abs_dir(rep.auc_test2) - 0.5) < 0.15
    assert rep.test2_above_ctrl is False
    assert abs(rep.ctrl_sig_rand_test2 - 0.5) < 0.05
    # verdict : issue (b) — cercle séparable, spirale générique non privilégiée.
    assert rep.issue == "b"


def test_full_report_test3_baseline_mostly_real() -> None:
    """TEST 3 : la majorité des baselines ont des valeurs propres RÉELLES.

    Pont de cohérence avec le Tour 4 : une baseline réelle ne peut pas spiraler
    (pas de rotation) ⇒ taux spirale-baseline très faible. Fraction réelle élevée.
    """
    rep = run_spectral_separability()
    assert rep.frac_baseline_real > 0.5
    assert abs(rep.frac_baseline_real + rep.frac_baseline_complex - 1.0) < 1e-9


# --- DÉTERMINISME bit-à-bit (2 runs) -----------------------------------------

def test_report_deterministic_bit_for_bit() -> None:
    r1 = run_spectral_separability()
    r2 = run_spectral_separability()
    assert r1.auc_test1 == r2.auc_test1
    assert r1.auc_test2 == r2.auc_test2
    assert r1.ctrl_perm_p95_test1 == r2.ctrl_perm_p95_test1
    assert r1.ctrl_perm_p95_test2 == r2.ctrl_perm_p95_test2
    assert r1.frac_baseline_real == r2.frac_baseline_real
    assert r1.n_baseline_spiral == r2.n_baseline_spiral
    assert r1.issue == r2.issue


# --- intégrité : la formule α-ω et ShapeSignature ne sont pas touchées --------

def test_shape_signature_still_intact() -> None:
    """shape_signature (Tour 4) reste appelable et inchangé (cercle = répétition)."""
    s0 = torch.tensor([1.0, 0.0])
    tr = Oscilloscope2D.circle(omega=OMEGA).trace(
        s0, steps=120, signal=InputSignal(kind="zero")
    )
    sig = shape_signature(tr, s0)
    assert sig.cv_r < 0.05
    assert sig.passes_circle_guards()
