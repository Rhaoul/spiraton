"""Tests Tour 7 (E3) — séparabilité forme-géométrique ↔ chiralité ABA.

Quatuor canon adapté à un DIAGNOSTIC de pont 33D↔géométrie↔ABA :
  * seeds fixés / déterminisme bit-à-bit (projection fixe, split par cycle seedé) ;
  * finitude (pas de NaN/Inf dans les slopes ni l'AUC) ;
  * formes simple/batch (un segment → un point ; un cycle → 3 points, ratio 2:1) ;
  * « formule exacte » sous paramètres forcés (gain = couplage sémantique fixé ;
    courant = projection fixe ; AUC de rang = Mann-Whitney) ;
  * flux : ici le diagnostic est @no_grad par construction (trace), on vérifie donc
    plutôt la NON-FUITE (verrou 2) et la discipline anti-fuite (CTRL-PERM, held-out).

ANTI-FUITE — le test de NON-FUITE du verrou 2 est OBLIGATOIRE (kickoff) : le 33D ne
dépend QUE du texte, jamais de la position du segment. REFUS incarné : l'issue (b)
(null légitime) est un résultat ACCEPTABLE, asserté tel quel ; une AUC>0.95 doit
lever l'issue (c) (suspicion de fuite). Le tokenizer natif est requis : sinon les
tests qui en dépendent se SKIPPENT proprement (jamais de substitution étiquette→donnée).
"""
import math

import pytest
import torch

from spiraton.data import vector33d as v33
from spiraton.data.tokenizer_bridge import is_available, NativeTokenizer33D
from spiraton.diagnostics.semantic_shape_separability import (
    GAIN_CEIL,
    GAIN_FLOOR,
    KAPPA,
    PhonemeCurrent,
    SemanticShapeReport,
    phoneme_projection,
    run_semantic_shape_separability,
    segment_gain,
    split_by_cycle,
    verrou2_nonleak_holds,
    _segment_current,
)

native = is_available()
needs_native = pytest.mark.skipif(
    not native, reason="tokenizer natif (.so/.dll) indisponible — diagnostic E3 skippe"
)


def _finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


# --- projection FIXE 15→2 : déterminisme + forme (coordonnée du geste) --------

def test_phoneme_projection_deterministic_and_shape() -> None:
    P1 = phoneme_projection()
    P2 = phoneme_projection()
    assert P1.shape == (2, 15)
    assert torch.equal(P1, P2)              # graine fixe ⇒ bit-à-bit
    # graine différente ⇒ projection différente (c'est bien un tirage, pas une constante)
    assert not torch.equal(P1, phoneme_projection(seed=1))


# --- gain : couplage sémantique FIXÉ (formule exacte sous scores forcés) ------

def test_segment_gain_formula_and_bounds() -> None:
    """g = 1 + κ·(MUL + dextro − DIV − lévo), moyenné, borné à [FLOOR, CEIL]."""
    # tout MUL (dim 2) : drive = +1 ⇒ g = 1 + κ.
    s = torch.zeros(3, 6); s[:, 2] = 1.0
    assert abs(segment_gain(s) - (1.0 + KAPPA)) < 1e-6
    # tout dextro (dim 4) : drive = +1 ⇒ g = 1 + κ (centrifuge).
    s = torch.zeros(3, 6); s[:, 4] = 1.0
    assert abs(segment_gain(s) - (1.0 + KAPPA)) < 1e-6
    # DIV (dim 3) + lévo (dim 5) : drive = −2 ⇒ g clampé au plancher.
    s = torch.zeros(3, 6); s[:, 3] = 1.0; s[:, 5] = 1.0
    g = segment_gain(s)
    assert g == GAIN_FLOOR
    # bornes respectées sur un drive extrême positif.
    s = torch.zeros(3, 6); s[:, 2] = 10.0; s[:, 4] = 10.0
    assert segment_gain(s) == GAIN_CEIL
    # segment vide ⇒ gain neutre 1.0 (ni amplifie ni contracte).
    assert segment_gain(torch.zeros(0, 6)) == 1.0


# --- courant phonémique : projection puis régime libre ------------------------

def test_phoneme_current_projects_then_zeroes() -> None:
    """u_t = P·phon_t pour t < N, puis 0 (régime libre) — verrou compat InputSignal."""
    phon = torch.tensor([[1.0] + [0.0] * 14, [0.0, 1.0] + [0.0] * 13])  # (2, 15)
    P = phoneme_projection()
    cur = _segment_current(phon, P)
    assert isinstance(cur, PhonemeCurrent)
    u0 = cur.at(0)
    assert torch.allclose(u0, (phon[0] @ P.t()).to(torch.float32), atol=1e-5)
    # au-delà du dernier token : courant nul (régime libre).
    assert torch.equal(cur.at(2), torch.zeros(2))
    assert torch.equal(cur.at(99), torch.zeros(2))
    # segment vide ⇒ courant nul partout.
    empty = _segment_current(torch.zeros(0, 15), P)
    assert torch.equal(empty.at(0), torch.zeros(2))


# --- split par cycle (verrou 3 : jamais un segment d'un cycle des deux côtés) --

def test_split_by_cycle_is_disjoint_and_deterministic() -> None:
    class FakeCycle:
        def __init__(self, i): self.i = i
    cycles = [FakeCycle(i) for i in range(20)]
    tr1, hd1 = split_by_cycle(cycles, holdout_frac=0.3, seed=7)
    tr2, hd2 = split_by_cycle(cycles, holdout_frac=0.3, seed=7)
    # déterminisme
    assert [c.i for c in tr1] == [c.i for c in tr2]
    assert [c.i for c in hd1] == [c.i for c in hd2]
    # partition disjointe et exhaustive PAR CYCLE
    s_tr = {c.i for c in tr1}
    s_hd = {c.i for c in hd1}
    assert s_tr.isdisjoint(s_hd)
    assert s_tr | s_hd == set(range(20))
    assert len(hd1) == 6  # round(0.3*20)


# --- AUC de rang (la « formule exacte » du scoring : Mann-Whitney) ------------

def test_rank_auc_reused_from_spectral_is_directional() -> None:
    from spiraton.diagnostics.spectral_separability import _rank_auc
    assert abs(_rank_auc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-12
    assert abs(_rank_auc([0, 1, 2], [3, 4, 5]) - 0.0) < 1e-12  # direction préservée


# --- VERROU 2 (NON-FUITE PHONÉTIQUE) — test OBLIGATOIRE du kickoff ------------

@needs_native
def test_verrou2_nonleak_same_text_same_vector() -> None:
    """vectors(texte) IDENTIQUE quelle que soit la position prétendue (A vs A′).

    C'est le cœur anti-fuite : si le 33D différait selon que le texte est « segment A »
    ou « segment A′ », la chiralité serait lisible dans l'entrée (fuite). On présente
    le MÊME texte deux fois et on exige l'égalité bit-à-bit. Le tokenizer ne reçoit que
    le texte nettoyé (aba._clean_text) — il n'a aucun moyen de connaître la position.
    """
    import numpy as np

    tok = NativeTokenizer33D()
    for text in ("la spirale revient transformee", "sans dissoudre ce qui demeure", "Joie"):
        a = tok.vectors(text)
        b = tok.vectors(text)
        assert np.array_equal(a, b)
        # le 33D ne porte aucune dim « position » : re-vectoriser ne change rien.
        assert a.shape[1] == v33.DIM


@needs_native
def test_verrou2_helper_holds_on_small_corpus() -> None:
    """Le helper de verrou 2 confirme l'absence de fuite sur un petit lot de cycles."""
    from spiraton.data.aba import iter_aba_cycles
    from spiraton.diagnostics.semantic_shape_separability import _default_corpora

    paths = _default_corpora()
    assert paths, "au moins un corpus ABA doit etre resolu"
    cycles = list(iter_aba_cycles(paths[0]))[:20]
    assert cycles
    tok = NativeTokenizer33D()
    assert verrou2_nonleak_holds(tok, cycles) is True


# --- pipeline complet : finitude, ratio 2:1, déterminisme, discipline ---------

@needs_native
def test_run_pipeline_finite_and_ratio_and_deterministic() -> None:
    rep1 = run_semantic_shape_separability(max_cycles=60)
    rep2 = run_semantic_shape_separability(max_cycles=60)
    assert isinstance(rep1, SemanticShapeReport)
    assert rep1.available is True

    # finitude des mesures clés
    assert _finite(rep1.auc_held)
    assert _finite(rep1.slope_aprime_med)
    assert _finite(rep1.slope_ab_med)
    assert 0.0 <= rep1.auc_held <= 1.0

    # ratio 2:1 par cycle (OUT A,B : IN A′) ⇒ neg ≈ 2·pos en held-out
    assert rep1.n_neg_held == 2 * rep1.n_pos_held

    # verrou 2 OK (sinon le verdict serait (c) STOP)
    assert rep1.verrou2_nonleak is True

    # déterminisme bit-à-bit
    assert rep1.auc_held == rep2.auc_held
    assert rep1.ctrl_perm_p95 == rep2.ctrl_perm_p95
    assert rep1.slope_aprime_med == rep2.slope_aprime_med
    assert rep1.issue == rep2.issue


@needs_native
def test_ctrl_sig_rand_near_half() -> None:
    """CTRL-SIG-RAND ≈ 0.5 : le protocole AUC est non-biaisé (garde-fou)."""
    rep = run_semantic_shape_separability(max_cycles=60)
    assert _finite(rep.ctrl_sig_rand)
    assert abs(rep.ctrl_sig_rand - 0.5) < 0.05


@needs_native
def test_verdict_is_a_b_or_c_and_leak_guard() -> None:
    """Le verdict est une issue nommée ; une AUC>0.95 lèverait (c) (fuite), pas (a)."""
    rep = run_semantic_shape_separability(max_cycles=60)
    assert rep.issue in ("a", "b", "c")
    strength = abs(rep.auc_held - 0.5) + 0.5
    # garde anti-fuite : si la force discriminante > 0.95, l'issue DOIT être (c).
    if strength > 0.95:
        assert rep.issue == "c"
    # cohérence (a) : si issue (a), l'AUC dépasse bien le 95e pct CTRL-PERM.
    if rep.issue == "a":
        assert rep.above_ctrl is True
        assert strength >= 0.70


def test_skip_is_clean_when_native_absent() -> None:
    """Sans tokenizer natif, le rapport SKIP proprement (jamais de substitution)."""
    if native:
        pytest.skip("natif disponible : la branche skip est couverte par construction")
    rep = run_semantic_shape_separability()
    assert rep.available is False
    assert rep.issue == "skip"
    assert "indisponible" in rep.skipped_reason
