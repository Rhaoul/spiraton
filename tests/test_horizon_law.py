"""Tests Tour 27 — loi de l'horizon sur grille synthétique (N, k) (``horizon_law``).

H27 : les quatre lois gelées (L_{N−k} primaire 3-classes, L_N, L_k, L_side)
classent-elles le signe de Δ_nom sur la grille exhaustive N ∈ [6,24], k ∈ [2,N−2] ?
Le signe est calculé en ``Fraction`` EXACTE (moteur rationnel indépendant du chemin
float de l'instrument gelé) ; la carte dérivée est gelée dans le docstring du module
AVANT la grille float (forme forte du mandat). Deux familles :

  * SANS dataset (toujours exécutées) : quatuor adapté (protocole/formes de la
    grille, exactitude/finitude, formule EXACTE sur profils connus et cellules-
    témoins rationnelles, déterminisme/pureté du diagnostic), portes 0a/0b/1,
    P-gate sur la grille, concordance dérivation↔instrument 228/228 en signe,
    et le verdict porte 2 MESURÉ (aucune loi à 100 % — gravé, jamais forcé).

  * AVEC corpus (skip propre si absents — aucun ``.so`` requis) : porte 3, retour
    aux Δ réels des runners T24/T25/T26 re-joués tels quels (aucun paramètre neuf).

L'instrument ``structural_gap.py`` reste GELÉ byte-à-byte ; ``regulate_step`` et le
canon ``core/`` ne sont JAMAIS touchés ; tout est déterministe (shuffles seedés).
"""
import math
from fractions import Fraction
from pathlib import Path

import pytest

from spiraton.diagnostics.horizon_law import (
    GATE0A_CELL_HIGH,
    GATE0A_CELL_LOW,
    GRID_K_MIN,
    GRID_N_MAX,
    GRID_N_MIN,
    K_STAR_FAMILY,
    MIN_DECISIVE_CELLS,
    N_STAR_CANDIDATES,
    WITNESS_CELLS,
    band_edge_touches,
    delta_nom_exact,
    delta_nom_float,
    excursion_x3,
    f_edge_exact,
    flip_side,
    gate0a,
    gate0b,
    gate1,
    grid_cells,
    grid_report,
    is_high_excursion,
    oriented_tokens_of,
    predict_l_k,
    predict_l_n,
    predict_l_side,
    predict_primary,
    profile_of,
    real_return,
    trace_e_exact,
)
from spiraton.diagnostics.structural_regulation import (
    BLOCK26_LINES,
    N_CYCLES_BLOCK26,
    N_CYCLES_CLAUDE,
    excursion,
)

_DATASET = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
if not _DATASET.is_file():
    _DATASET = Path(__file__).resolve().parents[2] / "dataset_aba.txt"

_CORPUS_CLAUDE = Path("F:/code/claude/spiraton-enhanced/corpus_claude_aba.txt")
if not _CORPUS_CLAUDE.is_file():
    _CORPUS_CLAUDE = Path(__file__).resolve().parents[2] / "corpus_claude_aba.txt"


# =============================================================================
# Quatuor 1/4 — protocole et formes : la grille gelée
# =============================================================================

def test_grid_cells_frozen_bounds() -> None:
    """228 cellules exactement, N ∈ [6,24], k ∈ [2,N−2], les DEUX côtés du flip."""
    cells = grid_cells()
    assert len(cells) == 228
    assert all(GRID_N_MIN <= n <= GRID_N_MAX and GRID_K_MIN <= k <= n - 2
               for (n, k) in cells)
    sides = {flip_side(n, k) for (n, k) in cells}
    assert sides == {"précoce", "tardif", "exact"}
    # déterminisme d'ordre (le rapport est apparié cellule à cellule)
    assert cells == grid_cells()


def test_profile_and_tokens_forms() -> None:
    """Profil = [+1]*k + [−1]*(N−k) ; tokens synthétiques tracés ; erreurs franches."""
    assert profile_of(7, 4) == [+1] * 4 + [-1] * 3
    toks = oriented_tokens_of(7, 4)
    assert [tk.orientation for tk in toks] == profile_of(7, 4)
    assert [tk.text for tk in toks[:2]] == ["t0", "t1"]
    with pytest.raises(ValueError):
        profile_of(6, 6)          # pas de flip
    with pytest.raises(ValueError):
        profile_of(6, 0)


def test_excursion_integer_identity_and_sharp_threshold() -> None:
    """``excursion_x3 = |3k−2N|`` entier ; identique au descripteur T26 (/3) ; le
    seuil band = 0.5 ne coupe AUCUNE cellule entière (|3k−2N| ∈ ℕ, jamais 1.5) :
    la P-gate sépare proprement bas (≤1) et haut (≥2)."""
    for (n, k) in [(7, 4), (8, 6), (9, 6), (13, 8), (24, 22)]:
        assert excursion_x3(n, k) == pytest.approx(3 * excursion(profile_of(n, k)))
        assert is_high_excursion(n, k) == (excursion(profile_of(n, k)) > 0.5)
    # une seule cellule basse par N (arithmétique mod 3) ⇒ 19 cellules basses
    lows = [(n, k) for (n, k) in grid_cells() if not is_high_excursion(n, k)]
    assert len(lows) == 19
    assert sorted({n for (n, _) in lows}) == list(range(6, 25))


# =============================================================================
# Quatuor 2/4 — formule EXACTE sous entrées forcées
# =============================================================================

def test_trace_exact_formula_k2_n4() -> None:
    """Le moteur rationnel reproduit en Fractions EXACTES le profil-témoin T24
    (k=2, N=4, lecteur fixe) : e = [1, 2/3, 1/3, 2/3, 1], f_edge = 3/4."""
    e = trace_e_exact(4, 2, Fraction(0))
    assert e == [Fraction(1), Fraction(2, 3), Fraction(1, 3), Fraction(2, 3), Fraction(1)]
    assert f_edge_exact(e) == Fraction(3, 4)


def test_witness_cells_exact_rationals() -> None:
    """Les 6 cellules-témoins rationnelles T26 sont reproduites EXACTEMENT
    (Fraction == Fraction, pas approx) — dont le contre-exemple (8,6) → 0."""
    for cell, expected in WITNESS_CELLS.items():
        assert delta_nom_exact(*cell) == expected
    # et l'instrument float gelé concorde à 1e-12 sur ces cellules
    for cell, expected in WITNESS_CELLS.items():
        assert delta_nom_float(*cell) == pytest.approx(float(expected), abs=1e-12)


# =============================================================================
# Quatuor 3/4 — exactitude / finitude
# =============================================================================

def test_exact_engine_returns_fractions_and_float_finite() -> None:
    """Δ_nom exact est une Fraction bornée (|Δ| ≤ 1) ; le float est fini."""
    for (n, k) in [(6, 2), (8, 6), (13, 8), (24, 12), (24, 22)]:
        d = delta_nom_exact(n, k)
        assert isinstance(d, Fraction)
        assert abs(d) <= 1
        assert math.isfinite(delta_nom_float(n, k))


def test_sign_concordance_derivation_vs_instrument_228() -> None:
    """Concordance dérivation Fraction ↔ instrument float GELÉ, cellule par cellule :
    SIGNES 228/228 identiques ; 2 divergences de MAGNITUDE seulement, toutes deux
    sur un point EXACTEMENT au bord de bande |e−1| = 1/2 (gravées) :
    (9,7) float 1/3 vs exact 2/9 ; (15,11) float 2/5 vs exact 1/3."""
    mag_div = []
    for (n, k) in grid_cells():
        d_exact = delta_nom_exact(n, k)
        d_float = delta_nom_float(n, k)
        assert ((d_float > 0) - (d_float < 0)) == ((d_exact > 0) - (d_exact < 0))
        if abs(d_float - float(d_exact)) > 1e-9:
            mag_div.append((n, k))
    assert mag_div == [(9, 7), (15, 11)]
    assert band_edge_touches(9, 7) >= 1 and band_edge_touches(15, 11) >= 1
    assert delta_nom_float(9, 7) == pytest.approx(1.0 / 3.0)
    assert delta_nom_exact(9, 7) == Fraction(2, 9)
    assert delta_nom_float(15, 11) == pytest.approx(0.4)
    assert delta_nom_exact(15, 11) == Fraction(1, 3)


def test_pgate_holds_on_grid() -> None:
    """P-gate commune (gelée) : excursion ≤ 0.5 ⟹ Δ = 0 EXACT — 19/19, 0 violation."""
    lows = [(n, k) for (n, k) in grid_cells() if not is_high_excursion(n, k)]
    assert len(lows) == 19
    assert all(delta_nom_exact(n, k) == 0 for (n, k) in lows)


# =============================================================================
# Quatuor 4/4 — déterminisme / pureté (l'analogue du flux de gradient)
# =============================================================================

def test_determinism_and_purity() -> None:
    """Diagnostic pur et déterministe : deux appels identiques bit-à-bit (exact ET
    float), aucune source aléatoire non seedée dans les portes."""
    for (n, k) in [(7, 4), (13, 8), (15, 11)]:
        assert delta_nom_exact(n, k) == delta_nom_exact(n, k)
        assert delta_nom_float(n, k) == delta_nom_float(n, k)   # égalité float stricte
    g1a = gate0a()
    g2a = gate0a()
    assert g1a.primary_high.gap == g2a.primary_high.gap
    assert g1a.primary_low.gap == g2a.primary_low.gap
    assert gate0b() == gate0b()


# =============================================================================
# PORTE 0a — re-jeu de la pré-validation d'instrument sur les 2 cellules gelées
# =============================================================================

def test_gate0a_measured() -> None:
    """Porte 0a MESURÉE : primaire order-sensible sur les DEUX cellules-témoins
    (haute (13,8) gap = 2.186e-1 ; basse (7,5) gap = 8.095e-2, tous ≥ δ_min = 1e-4),
    multiset vacuous (gap = 0.0 exact) sur les deux — la porte est saine."""
    g = gate0a()
    assert g.cell_high == GATE0A_CELL_HIGH == (13, 8)
    assert g.cell_low == GATE0A_CELL_LOW == (7, 5)
    assert excursion_x3(*g.cell_high) == 2      # excursion 2/3 > band
    assert excursion_x3(*g.cell_low) == 1       # excursion 1/3 ≤ band, ≠ 0 (non dégénérée)
    assert g.primary_high.is_order_sensitive
    assert g.primary_high.gap == pytest.approx(0.2185897, abs=1e-6)
    assert g.primary_low.is_order_sensitive
    assert g.primary_low.gap == pytest.approx(0.0809524, abs=1e-6)
    assert g.multiset_high.is_vacuous and g.multiset_high.gap == 0.0
    assert g.multiset_low.is_vacuous and g.multiset_low.gap == 0.0
    assert g.passes


# =============================================================================
# PORTE 0b — cellules décisives par paire (AVANT toute lecture de signe)
# =============================================================================

def test_gate0b_decisive_counts_measured() -> None:
    """Porte 0b MESURÉE : ≥ 5 cellules d'arbitrage pour chaque paire de lois —
    L_{N−k}↔L_N : 34/36/39 (N* = 8/9/10) ; L_{N−k}↔L_k : min 33 sur la famille
    k* ∈ [2,23] (k* n'étant ajusté qu'après lecture, la garantie couvre la
    famille) ; L_{N−k}↔L_side : 26. L'arbitrage n'est pas vacuous."""
    g = gate0b()
    assert g.ln_counts == {8: 34, 9: 36, 10: 39}
    assert set(g.lk_counts) == set(K_STAR_FAMILY)
    assert g.lk_min == 33
    assert g.lside_count == 26
    assert all(c >= MIN_DECISIVE_CELLS for c in g.ln_counts.values())
    assert g.lk_min >= MIN_DECISIVE_CELLS and g.lside_count >= MIN_DECISIVE_CELLS
    assert g.passes


# =============================================================================
# PORTE 1 — pivots + cellules-témoins
# =============================================================================

def test_gate1_pivots_and_witnesses() -> None:
    """Porte 1 : η=0 ≡ g-fixe exact, profil sans flip Δ = 0, les 6 témoins exacts."""
    g = gate1()
    assert g.pivot_eta0_exact
    assert g.pivot_noflip == 0.0
    assert all(g.witness_exact.values())
    assert all(g.witness_float_ok.values())
    assert g.passes


# =============================================================================
# PORTE 2 — la grille : verdict MESURÉ (gravé, jamais forcé)
# =============================================================================

def test_grid_report_verdict_documented() -> None:
    """Verdict porte 2 MESURÉ (2026-07-11) : AUCUNE des quatre lois gelées ne
    classe 100 % de la grille — issue honnête n° 3 de l'émission, la CARTE EXACTE
    est le livrable.

      * P-gate : 0 violation (19 cellules basses, Δ = 0 exact partout) ;
      * concordance : 0 divergence de SIGNE float↔exact ; 2 divergences de
        magnitude (bord de bande), gravées au test de concordance ;
      * L_{N−k} 3-classes : 119/228 mal classées — la loi primaire est RÉFUTÉE
        telle qu'énoncée (l'« escalier de N » T26 était un fait de la droite
        |3k−2N| = 2 flip-précoce ; la grille qui dissocie N, k, N−k le dissout) ;
      * L_N : N* retenu = 10 (mal classées {8: 86, 9: 84, 10: 83}) ;
      * L_k : k* retenu = 7 (ex æquo [7, 8]), 17/228 — meilleure rivale, pas 100 % ;
      * L_side : 144/228 ;
      * B_sweep_global : g = 1.0 (le nominal gagne le sweep — aucun décalage) ;
      * paysage : côté TARDIF 56 Δ>0 + 1 Δ=0 (le contre-exemple (8,6) gravé T26
        est le SEUL zéro tardif) ; côté PRÉCOCE {−: 43, 0: 47, +: 62} — structure
        arithmétique de l'oscillateur MARGINAL (u_{t+1} = (3/2)u_t − u_{t−1},
        |λ| = 1, cos θ = 3/4), pas une loi monotone de N, k ou N−k.
    """
    gr = grid_report()
    assert (gr.n_cells, gr.n_low, gr.n_high) == (228, 19, 209)
    assert gr.pgate_violations == []
    assert gr.sign_mismatches == []
    assert [(c.n, c.k) for c in gr.magnitude_divergences] == [(9, 7), (15, 11)]
    # les quatre lois, comptes gravés
    assert len(gr.mis_primary) == 119
    assert gr.n_star_retained == 10
    assert {ns: len(v) for ns, v in gr.mis_l_n.items()} == {8: 86, 9: 84, 10: 83}
    assert gr.k_star_retained == 7 and gr.k_star_ties == [7, 8]
    assert len(gr.mis_l_k_retained) == 17
    assert len(gr.mis_l_side) == 144
    assert not gr.any_law_100
    # baseline secondaire : le nominal gagne le sweep global de la grille
    assert gr.sweep_global_gain == 1.0
    # paysage gravé : côté tardif quasi uniformément positif, (8,6) seul zéro
    high = [c for c in gr.cells if c.excursion_x3 >= 2]
    tardif = [c for c in high if c.side == "tardif"]
    assert len(tardif) == 57
    assert sum(1 for c in tardif if c.sign > 0) == 56
    zeros_tardif = [(c.n, c.k) for c in tardif if c.sign == 0]
    assert zeros_tardif == [(8, 6)]
    precoce = [c for c in high if c.side == "précoce"]
    counts = {s: sum(1 for c in precoce if c.sign == s) for s in (-1, 0, 1)}
    assert counts == {-1: 43, 0: 47, 1: 62}
    # cohérence interne : chaque mal-classée primaire est bien une divergence réelle
    for m in gr.mis_primary[:5]:
        assert predict_primary(m.n, m.k) == m.predicted != m.actual


def test_rival_predictions_are_pure_functions() -> None:
    """Les lois sont des prédictions PURES sur (N, k) — aucune mesure requise."""
    assert predict_primary(7, 4) == -1          # N−k = 3, haute
    assert predict_primary(13, 8) == +1         # N−k = 5, haute
    assert predict_primary(8, 6) == 0           # N−k = 2, haute
    assert predict_primary(9, 6) == 0           # excursion 0 : P-gate
    assert predict_l_n(13, 8, 10) and not predict_l_n(9, 4, 10)
    assert predict_l_k(13, 8, 7) and not predict_l_k(13, 6, 7)
    assert predict_l_side(13, 8) and not predict_l_side(11, 8)   # (11,8) tardif


# =============================================================================
# PORTE 3 — retour aux Δ réels (runners re-joués tels quels, aucun paramètre neuf)
# =============================================================================

@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_real_return_t24_documented() -> None:
    """Porte 3 MESURÉE (T24-défaut, 40 cycles) : la carte exacte classe 100 % des
    Δ réels en SIGNE et en VALEUR (best_fixed sweep = 1.0 ⟹ mêmes chemins float) ;
    la loi primaire (info) y classe aussi 40/40 — ces cellules vivent sur la
    droite compatible."""
    rr = real_return(str(_DATASET), n_cycles=40, label="T24-defaut")
    assert rr.n_cycles == 40
    assert rr.best_fixed == 1.0
    assert rr.n_sign_match_map == 40 and rr.sign_mismatches_map == []
    assert rr.n_value_match_map == 40
    assert rr.n_correct_primary == 40 and rr.mis_primary == []


@pytest.mark.skipif(not _CORPUS_CLAUDE.is_file(), reason="corpus_claude_aba.txt indisponible")
def test_real_return_t25_claude_documented() -> None:
    """Porte 3 MESURÉE (T25-claude, 76 cycles) : carte exacte 76/76 en signe ET en
    valeur. La loi primaire L_{N−k} (info) : 75/76 — le cycle 27 (N=11, k=8, flip
    TARDIF, N−k=3, prédit −, observé +) la réfutait DÉJÀ dans le réel : la carte
    ((11,8) = '+') classe ce cycle correctement, la loi non."""
    rr = real_return(str(_CORPUS_CLAUDE), n_cycles=N_CYCLES_CLAUDE, label="T25-claude")
    assert rr.n_cycles == 76
    assert rr.best_fixed == 1.0
    assert rr.n_sign_match_map == 76 and rr.sign_mismatches_map == []
    assert rr.n_value_match_map == 76
    assert rr.n_correct_primary == 75
    assert [(m[1], m[2], m[3], m[4]) for m in rr.mis_primary] == [(11, 8, -1, +1)]
    assert delta_nom_exact(11, 8) == Fraction(2, 11)   # la carte, elle, dit '+'


@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_real_return_t26_bloc26_documented() -> None:
    """Porte 3 MESURÉE (T26-bloc26, 2000 cycles) : carte exacte 2000/2000 en signe
    ET en valeur ; loi primaire (info) 2000/2000 (le bloc vit sur la droite
    |3k−2N| = 2 flip-précoce, là où la loi et la carte coïncident)."""
    rr = real_return(str(_DATASET), n_cycles=N_CYCLES_BLOCK26,
                     line_range=BLOCK26_LINES, label="T26-bloc26")
    assert rr.n_cycles == 2000
    assert rr.best_fixed == 1.0
    assert rr.n_sign_match_map == 2000 and rr.sign_mismatches_map == []
    assert rr.n_value_match_map == 2000
    assert rr.n_correct_primary == 2000 and rr.mis_primary == []
