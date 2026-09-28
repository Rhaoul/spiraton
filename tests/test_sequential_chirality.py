"""Tests Tour 9 (E3 re-tentative) — pente du flux ordonne ↔ chiralite ABA.

Quatuor canon adapte a un DIAGNOSTIC sequentiel, PLUS le test non-negociable de l'ablation :
  * deterministe / seeds fixes (split par cycle, permutations seedees) ;
  * finitude (pas de NaN/Inf dans g_bar ni l'AUC quand la sequence est definie) ;
  * formes simple/batch (un segment → un point ; un cycle → 3 points, ratio 2:1) ;
  * « formule exacte » de la fonctionnelle UNIQUE : g_bar = (seq[-1] - seq[0])/(L-1) ;
  * mapping FIXE A PRIORI (montee→ouverture), JAMAIS fitte ;
  * L'ABLATION VERROUILLEE PAR TEST : sur un signal qui lit VRAIMENT l'ordre, l'AUC d'ordre
    permute s'effondre vers 0.5 (le gain emerge) ; sur un signal ordre-independant (somme
    deguisee), l'AUC SURVIT a la permutation (le gain ne s'effondre pas) — on prouve que le
    diagnostic DISTINGUE les deux.

Le tokenizer natif est requis pour le pipeline complet : sinon SKIP propre (jamais de
substitution etiquette→donnee). Les tests de la fonctionnelle et de l'ablation NE dependent
PAS du tokenizer (sequences synthetiques) — ils tournent toujours.
"""
import math

import pytest
import torch

from spiraton.data import vector33d as v33
from spiraton.data.tokenizer_bridge import is_available, NativeTokenizer33D
from spiraton.diagnostics.sequential_chirality import (
    SIG_FLUX,
    SegmentFluxPoint,
    SequentialChiralityReport,
    _abs_dir,
    _auc_closure_descends,
    _ctrl_perm_order_distribution,
    _order_perm_force_p95,
    _token_flux_prefix,
    mean_signed_gradient,
    permuted_gradient,
    run_sequential_chirality,
    split_by_cycle,
)

native = is_available()
needs_native = pytest.mark.skipif(
    not native, reason="tokenizer natif (.so/.dll) indisponible — diagnostic Tour 9 skippe"
)


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


# --- slice sig_flux : verifie contre la source de verite C --------------------

def test_sig_flux_slice_matches_c_header() -> None:
    """sig_flux = dims 13-17 (5 valeurs) — VERIFIE contre tokenizer.c (vector33d[13+i])."""
    assert SIG_FLUX == slice(13, 18)
    assert SIG_FLUX.stop - SIG_FLUX.start == 5
    # le slice flux est DANS la zone phonemique 8-22 (PHONEME_SIG), apres l'impedance (8-12).
    assert v33.PHONEME_SIG.start <= SIG_FLUX.start
    assert SIG_FLUX.stop <= v33.PHONEME_SIG.stop


# --- la fonctionnelle UNIQUE : formule exacte de g_bar ------------------------

def test_mean_signed_gradient_exact_formula() -> None:
    """g_bar = (seq[-1] - seq[0])/(L-1) — telescopage de la moyenne des gradients."""
    # montee lineaire 0,1,2,3 : g_bar = (3-0)/3 = 1.0
    assert abs(mean_signed_gradient([0.0, 1.0, 2.0, 3.0]) - 1.0) < 1e-9
    # descente 3,2,1,0 : g_bar = (0-3)/3 = -1.0 (fermeture attendue)
    assert abs(mean_signed_gradient([3.0, 2.0, 1.0, 0.0]) + 1.0) < 1e-9
    # plat : g_bar = 0
    assert mean_signed_gradient([2.0, 2.0, 2.0]) == 0.0
    # egal a la moyenne des gradients elementaires (definition d'origine, non simplifiee)
    seq = [0.5, 2.0, 1.0, 4.0]
    grads = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    assert abs(mean_signed_gradient(seq) - sum(grads) / len(grads)) < 1e-9
    # longueur < 2 ⇒ pente non definie ⇒ nan (jamais imputee)
    assert math.isnan(mean_signed_gradient([5.0]))
    assert math.isnan(mean_signed_gradient([]))


def test_mapping_is_fixed_not_fitted() -> None:
    """Le mapping montee→ouverture / descente→fermeture est FIXE, lisible dans le signe de g_bar."""
    # une sequence montante => g_bar > 0 => ouverture (dextro) ; jamais l'inverse selon les data.
    assert mean_signed_gradient([0.0, 5.0]) > 0
    assert mean_signed_gradient([5.0, 0.0]) < 0


# --- preprocessing : retrait du zero-padding du C -----------------------------

def test_token_flux_prefix_strips_zero_padding() -> None:
    """_token_flux_prefix tronque sig_flux au nombre reel de phonemes (dim 31 = nb_phon/10)."""
    row = torch.zeros(v33.DIM)
    # 3 phonemes reels ; sig_flux = [a,b,c,0,0] ; dim 31 = 3/10
    row[13], row[14], row[15] = 1.0, 2.0, 3.0
    row[16], row[17] = 0.0, 0.0  # padding C
    row[31] = 3.0 / 10.0
    assert _token_flux_prefix(row) == [1.0, 2.0, 3.0]
    # 5 phonemes : tout est reel
    row2 = torch.zeros(v33.DIM)
    for i, val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        row2[13 + i] = val
    row2[31] = 5.0 / 10.0
    assert _token_flux_prefix(row2) == [1.0, 2.0, 3.0, 4.0, 5.0]
    # mot > 5 phonemes : sig_flux ne porte QUE les 5 premiers (borne a 5)
    row3 = torch.zeros(v33.DIM)
    for i in range(5):
        row3[13 + i] = float(i + 1)
    row3[31] = 8.0 / 10.0
    assert _token_flux_prefix(row3) == [1.0, 2.0, 3.0, 4.0, 5.0]


# --- AUC orientee : convention documentee (pos = A' descend) ------------------

def test_auc_orientation_closure_descends() -> None:
    """AUC>0.5 ⇔ A' (cloture) a g_bar plus NEGATIF que A/B (descend plus). Direction preservee."""
    pts = [
        SegmentFluxPoint(0, "SEG_A", False, 3, (0.0, 1.0, 2.0), +1.0),       # ouverture, monte
        SegmentFluxPoint(0, "SEG_B", False, 3, (0.0, 1.0, 2.0), +1.0),
        SegmentFluxPoint(0, "SEG_A_PRIME", True, 3, (2.0, 1.0, 0.0), -1.0),  # cloture, descend
    ]
    auc = _auc_closure_descends(pts)
    assert auc == 1.0  # separation parfaite dans le sens attendu


# --- split par cycle (verrou 3) ----------------------------------------------

def test_split_by_cycle_disjoint_and_deterministic() -> None:
    class FakeCycle:
        def __init__(self, i): self.i = i
    cycles = [FakeCycle(i) for i in range(20)]
    tr1, hd1 = split_by_cycle(cycles, holdout_frac=0.3, seed=7)
    tr2, hd2 = split_by_cycle(cycles, holdout_frac=0.3, seed=7)
    assert [c.i for c in tr1] == [c.i for c in tr2]
    assert [c.i for c in hd1] == [c.i for c in hd2]
    s_tr, s_hd = {c.i for c in tr1}, {c.i for c in hd1}
    assert s_tr.isdisjoint(s_hd)
    assert s_tr | s_hd == set(range(20))
    assert len(hd1) == 6


# --- permuted_gradient : meme ENSEMBLE, gradient DIFFERENT --------------------

def test_permuted_gradient_changes_with_order() -> None:
    """La permutation conserve l'ensemble des valeurs mais change g_bar (gradient genuine)."""
    seq = [0.0, 1.0, 2.0, 3.0, 4.0]
    real = mean_signed_gradient(seq)
    # plusieurs permutations donnent des g_bar varies (pas tous egaux au reel)
    perms = [permuted_gradient(seq, seed=s) for s in range(30)]
    assert all(math.isfinite(p) for p in perms)
    assert any(abs(p - real) > 1e-6 for p in perms)  # l'ordre compte pour g_bar
    # deterministe : meme seed => meme resultat
    assert permuted_gradient(seq, seed=3) == permuted_gradient(seq, seed=3)


# --- L'ABLATION VERROUILLEE : signal qui lit l'ordre vs somme deguisee --------

def _ordered_signal_points(n_cycles: int = 40):
    """Signal SYNTHETIQUE ou l'ordre PORTE le label : ouverture=montant, cloture=descendant.

    Meme ENSEMBLE de valeurs {0,1,2,3} pour les deux classes — SEUL l'ordre differe. Une
    statistique d'ensemble (somme, variance, etendue) ne peut donc PAS separer : seul le
    gradient ordonne le peut. C'est le cas-test ideal de l'ablation.
    """
    pts = []
    for c in range(n_cycles):
        pts.append(SegmentFluxPoint(c, "SEG_A", False, 4, (0.0, 1.0, 2.0, 3.0), 1.0))
        pts.append(SegmentFluxPoint(c, "SEG_B", False, 4, (0.0, 1.0, 2.0, 3.0), 1.0))
        pts.append(SegmentFluxPoint(c, "SEG_A_PRIME", True, 4, (3.0, 2.0, 1.0, 0.0), -1.0))
    return pts


def test_ablation_gain_collapses_on_ordered_signal() -> None:
    """CŒUR DU TOUR : sur un signal qui lit l'ordre, la FORCE d'ordre permute S'EFFONDRE vers 0.5.

    L'ensemble {0,1,2,3} est commun aux deux classes : seul l'ordre separe. AUC reel = 1.0 (force
    1.0). Sous permutation de l'ordre, la force doit retomber pres de 0.5, et la force reelle doit
    depasser le 95e pct de la force sous H0-ordre (gain significatif).
    """
    pts = _ordered_signal_points()
    auc_real = _auc_closure_descends(pts)
    assert auc_real == 1.0  # ordre reel : separation parfaite

    op_med, _op5, _op95, dist = _ctrl_perm_order_distribution(pts, n_perms=25, seed0=1234)
    assert dist
    # la FORCE d'ordre permute s'effondre nettement vers 0.5 (le gain emerge)
    assert _abs_dir(op_med) < 0.70
    # gain = effondrement de force, franc
    gain = _abs_dir(auc_real) - _abs_dir(op_med)
    assert gain > 0.25

    force_med, force_p95, forces = _order_perm_force_p95(pts, n_perms=25, seed0=5678)
    assert forces
    # la FORCE reelle (1.0) depasse le 95e pct de la force sous H0-ordre → gain significatif
    assert _abs_dir(auc_real) > force_p95


def test_ablation_isolates_order_from_ensemble_statistic() -> None:
    """CONTRE-EPREUVE (anti-DISSIPATION) : une statistique d'ENSEMBLE survit a la permutation,
    la pente ordonnee NON.

    On prend le MEME signal ordonne que ci-dessus (ensemble {0,1,2,3} commun aux deux classes :
    AUCUNE statistique d'ensemble ne peut le separer). On compare deux scoreurs :
      * la SOMME (somme deguisee, ordre-invariante) : AUC ≈ 0.5 deja en ordre reel, et reste 0.5
        sous permutation → un scoreur d'ensemble ne lit RIEN ici ;
      * la PENTE ordonnee g_bar : AUC = 1.0 en ordre reel, qui S'EFFONDRE sous permutation.
    Cela prouve que (1) la separation vient bien de l'ORDRE (l'ensemble ne porte rien), et (2)
    l'ablation par permutation est le bon test : elle laisserait passer une somme deguisee (qui
    ne separe deja pas) mais detruit le vrai signal d'ordre. Si jamais une « signature de
    gradient » survivait a la permutation, ce serait une statistique d'ensemble masquee — a
    REJETER (issue c).
    """
    pts = _ordered_signal_points()
    labels = [1 if p.is_closure else 0 for p in pts]

    # scoreur d'ENSEMBLE : la somme de la sequence (ordre-invariante par construction)
    sums = [sum(p.seq) for p in pts]
    pos_s = [s for s, l in zip(sums, labels) if l == 1]
    neg_s = [s for s, l in zip(sums, labels) if l == 0]
    from spiraton.diagnostics.spectral_separability import _rank_auc
    auc_sum = _rank_auc(pos_s, neg_s)
    # l'ensemble {0,1,2,3} est commun aux deux classes ⇒ la somme ne separe PAS (AUC ≈ 0.5)
    assert abs(auc_sum - 0.5) < 1e-9

    # scoreur d'ORDRE : g_bar separe parfaitement en ordre reel, s'effondre sous permutation
    auc_order = _auc_closure_descends(pts)
    assert auc_order == 1.0
    op_med, _o5, _o95, _d = _ctrl_perm_order_distribution(pts, n_perms=25, seed0=4321)
    assert _abs_dir(op_med) < 0.70  # s'effondre
    # le diagnostic isole donc bien l'ordre : seul un scoreur sensible a l'ordre y voit le signal.


# --- pipeline complet (natif requis) -----------------------------------------

@needs_native
def test_run_pipeline_finite_ratio_deterministic() -> None:
    rep1 = run_sequential_chirality(max_cycles=80)
    rep2 = run_sequential_chirality(max_cycles=80)
    assert isinstance(rep1, SequentialChiralityReport)
    assert rep1.available is True
    assert _finite(rep1.auc_held)
    assert 0.0 <= rep1.auc_held <= 1.0
    assert _finite(rep1.majority_baseline)
    # ratio 2:1 par cycle dans les effectifs bruts
    assert rep1.n_open == 2 * rep1.n_closure
    # deterministe bit-a-bit
    assert rep1.auc_held == rep2.auc_held
    assert rep1.gain_sequentiel == rep2.gain_sequentiel
    assert rep1.issue == rep2.issue


@needs_native
def test_pipeline_verdict_named_and_leak_guard() -> None:
    rep = run_sequential_chirality(max_cycles=80)
    assert rep.issue in ("a", "b", "c", "skip")
    strength = abs(rep.auc_held - 0.5) + 0.5
    if math.isfinite(strength) and strength > 0.95:
        assert rep.issue == "c"
    # cohérence (a) : progression EXIGE l'effondrement du gain sous permutation
    if rep.issue == "a":
        assert rep.above_ctrl_perm is True
        assert rep.gain_significant is True
        assert rep.gain_collapses is True


def test_skip_clean_when_native_absent() -> None:
    if native:
        pytest.skip("natif disponible : la branche skip est couverte par construction")
    rep = run_sequential_chirality()
    assert rep.available is False
    assert rep.issue == "skip"
    assert "indisponible" in rep.skipped_reason
