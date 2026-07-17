"""Tests Tour 33 — chasse à l'index post-hoc, split vierge (``index_hunt``).

H33 : ∃ un index compact structurel I(o) (exploration libre de la dérivation,
gelé PAR ÉCRIT avant validation, règle §8 : ni Δ ni C_organe — profil + lecteur
FIXE η=0 seulement) sound 100 % par-η + compressif strict + raffinant (N,k) sur
le périmètre de VALIDATION VIERGE (V-GRAMMAIRE N∈[25,40], V-STAR (15,9),
V-TOTAL N∈[13,14], V-TOTAL-15 optionnel — joué, mur de budget tenu).

VERDICT MESURÉ (2026-07-17, one-shot, gravé ici) : **LES DEUX CARTOUCHES GELÉS
SONT MORTS NETTEMENT** — P (« T_bande », trajectoire fixe tronquée à la bande)
et S (« T_bande_runs », son raffinement strict) sont UNSOUND aux 3 η sur CHACUN
des 4 sous-périmètres vierges (toujours compressifs et raffinants : la mort est
purement de soundness). Paires-témoins exactes gravées ci-dessous (Fraction,
jamais arrondies). La BASELINE tautologique (C_organe, C_fixe) montre qu'il
restait de la place (425 paires / 8158 profils sur le total de dérivation) :
la granularité de la carte Δ résiste au nommage structurel bon marché — branche
(ii) du verdict gelé de l'émission, recevable UNE FOIS.

Stratégie de gel des chiffres DÉCLARÉE : la validation entière (46 297 profils
× 3 η + V-TOTAL-15) coûte ~2 min — les tests ne la rejouent PAS ; ils gèlent
des ÉCHANTILLONS-TÉMOINS exacts (paires même-I à Δ différents sur chaque
sous-périmètre × η) + les comptes de compression sur V-STAR (5005 index, sans
Δ) + les portes de dérivation. Corpus gelé T31 hors dépôt : skip propre s'il
est absent (convention T24-T32).
"""
import inspect
from fractions import Fraction

import pytest

from spiraton.diagnostics.horizon_law import profile_of
from spiraton.diagnostics.spectral_map import ETAS_FROZEN
from spiraton.diagnostics.profile_exact import (
    census_horscanon,
    corpus_horscanon_path,
    delta_nom_exact_profile,
)
from spiraton.diagnostics.arrangement_map import (
    ETAS_T32,
    STAR_CELL,
    cell_of,
    delta_map,
    enumerate_grammar,
    enumerate_total,
    is_order_degenerate_cell,
)
from spiraton.diagnostics.index_hunt import (
    BUDGET_WALL_S,
    CARTRIDGES_T33,
    ETA_LABELS_T33,
    ETA_PRIMARY_T33,
    VGRAM_N_MAX,
    VGRAM_N_MIN,
    VSTAR_CELL,
    VTOTAL15_N,
    VTOTAL_N_MAX,
    VTOTAL_N_MIN,
    baseline_count,
    enumerate_cell,
    enumerate_v_grammar,
    enumerate_v_star,
    enumerate_v_total,
    fixed_trace,
    gate0a_t33,
    gate0h_t33,
    gate1_t33,
    index_trunc_band,
    index_trunc_band_runs,
    index_verdict,
    pick_0a_witness_grammar,
    pick_0a_witness_star,
    validation_cells,
)

_CORPUS = corpus_horscanon_path()
_HAVE_CORPUS = _CORPUS.is_file()


# =============================================================================
# Quatuor 1/4 — protocole gelé
# =============================================================================

def test_frozen_protocol_t33() -> None:
    """η gelés (lecture seule T28), bornes de validation vierges (émission §6),
    mur de budget déclaré, 2 cartouches exactement dans l'ordre de tir P puis S,
    cellule star fraîche NON dégénérée (pré-validée, doctrine T32 (iii))."""
    assert ETA_LABELS_T33 == ("1/2", "1", "4") and ETA_PRIMARY_T33 == "1/2"
    for lbl in ETA_LABELS_T33:
        assert ETAS_T32[lbl] == ETAS_FROZEN[lbl]
    assert (VGRAM_N_MIN, VGRAM_N_MAX) == (25, 40)
    assert (VTOTAL_N_MIN, VTOTAL_N_MAX) == (13, 14)
    assert VTOTAL15_N == 15 and BUDGET_WALL_S == 1800.0
    assert VSTAR_CELL == (15, 9)
    assert not is_order_degenerate_cell(*VSTAR_CELL)       # 27 ≠ 30
    # aucun N de la validation n'a été énuméré par T32 (fraîcheur au sens fort)
    assert VGRAM_N_MIN > 24 and VTOTAL_N_MIN > 12 and VSTAR_CELL[0] > 12
    assert [name for name, _ in CARTRIDGES_T33] == ["P_trunc_band",
                                                    "S_trunc_band_runs"]


def test_rule_8_admissibility_lexical_guard() -> None:
    """Garde anti-tautologie (§8) : la matière des cartouches est le lecteur
    FIXE (η = 0 codé en dur dans ``fixed_trace``) ; les définitions de P et S
    ne référencent NI delta NI un η libre (C_organe interdit)."""
    src_ft = inspect.getsource(fixed_trace)
    assert "Fraction(0)" in src_ft
    for fn in (index_trunc_band, index_trunc_band_runs):
        src = inspect.getsource(fn)
        assert "delta" not in src.lower()
        assert "eta" not in src.replace("η", "").lower()
        assert "fixed_trace" in src


# =============================================================================
# Quatuor 2/4 — énumérateurs de validation : comptes EXACTS, déterminisme
# =============================================================================

def test_validation_enumerators_exact_counts() -> None:
    """V-GRAMMAIRE : Σ_{25..40} N(N−1) = 16 720 ; V-STAR : C(15,9) = 5005 ;
    V-TOTAL : (2^13−2)+(2^14−2) = 24 572 ; 530 cellules canoniques ;
    déterminisme (double appel)."""
    vg = enumerate_v_grammar()
    assert len(vg) == sum(n * (n - 1) for n in range(25, 41)) == 16720
    assert all(25 <= len(p) <= 40 for p in vg)
    vs = enumerate_v_star()
    assert len(vs) == 5005
    assert all(cell_of(p) == VSTAR_CELL for p in vs)
    assert vs[0] == tuple(profile_of(15, 9))               # 1er = canonique
    assert enumerate_cell(15, 9) == vs                     # déterminisme
    vt = enumerate_v_total()
    assert len(vt) == (2 ** 13 - 2) + (2 ** 14 - 2) == 24572
    cells = validation_cells()
    assert len(cells) == 504 + 25 + 1 == 530
    assert VSTAR_CELL in cells


# =============================================================================
# Quatuor 3/4 — index calculés à la main (marche fermée du lecteur fixe)
# =============================================================================

def test_indexes_hand_computed() -> None:
    """profile_of(6,2) = ++−−−− : trace fixe (0, −1, −1/2, 0, 1/2, 1) (gravée
    T31/T32 ; pas −1 sur +1, +1/2 sur −1, marche FERMÉE e_N = 1) ⟹
    P = (1/2, 1/2, 1/2, 1/2, 1/2, 1) (clamp) et
    S = (6, ((−1, 4), 1/2, 1)) (run hors-bande bas de longueur 4, puis les
    valeurs EN BANDE exactes 1/2 — bord distingué du clamp — et 1).
    (+1,−1,+1,+1) : trace (10/9, 7/9, 8/9, 1) toute en bande ⟹ P = la trace,
    S = (4, trace)."""
    p = tuple(profile_of(6, 2))
    assert fixed_trace(p) == (Fraction(0), Fraction(-1), Fraction(-1, 2),
                              Fraction(0), Fraction(1, 2), Fraction(1))
    assert index_trunc_band(p) == (Fraction(1, 2),) * 5 + (Fraction(1),)
    assert index_trunc_band_runs(p) == (6, ((-1, 4), Fraction(1, 2), Fraction(1)))

    o = (+1, -1, +1, +1)
    tr = (Fraction(10, 9), Fraction(7, 9), Fraction(8, 9), Fraction(1))
    assert fixed_trace(o) == tr
    assert index_trunc_band(o) == tr
    assert index_trunc_band_runs(o) == (4, tr)


def test_fixed_trace_closed_walk_and_s_refines_p() -> None:
    """La marche fixe est FERMÉE (e_N = 1, dérivé : Σ des pas = 0) ; S raffine
    STRICTEMENT P (même-S ⟹ même-P ; l'inverse est faux — le bord en bande vs
    clampé), vérifié exhaustivement sur le total N = 6."""
    small = enumerate_total(6, 6)
    assert len(small) == 62
    p_by_s = {}
    for p in small:
        assert fixed_trace(p)[-1] == Fraction(1)
        s_val, p_val = index_trunc_band_runs(p), index_trunc_band(p)
        assert p_by_s.setdefault(s_val, p_val) == p_val    # même-S ⟹ même-P
    assert len({index_trunc_band_runs(p) for p in small}) >= len(
        {index_trunc_band(p) for p in small})


# =============================================================================
# Quatuor 4/4 — types exacts / déterminisme
# =============================================================================

def test_index_types_and_determinism() -> None:
    """Fractions exactes partout ; deux appels = même valeur (aucun aléa)."""
    for p in (tuple(profile_of(7, 3)), (+1,) * 8 + (-1,) + (+1,) + (-1,) * 5):
        assert index_trunc_band(p) == index_trunc_band(p)
        assert index_trunc_band_runs(p) == index_trunc_band_runs(p)
        for x in index_trunc_band(p):
            assert isinstance(x, Fraction)
            assert Fraction(1, 2) <= x <= Fraction(3, 2)


# =============================================================================
# PORTE 0h gravée — héritages + non-régression fraîche au terrain vierge
# =============================================================================

@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gate0h_t33_engraved() -> None:
    """PASSE (gravé 2026-07-17) : gate1_t32 (témoins T31) True via import
    lecture seule ; gate0b_t32 zéro divergence ; non-régression FRAÎCHE
    530 cellules × 3 η = 1590 égalités Fraction, 0 divergence (couverture
    100 % — delta_nom_exact_eta défini pour tout 1 ≤ k ≤ N−1)."""
    grammar_t32 = enumerate_grammar()
    gmaps_t32 = {lbl: delta_map(grammar_t32, ETAS_T32[lbl])
                 for lbl in ETA_LABELS_T33}
    cen = census_horscanon(str(_CORPUS))
    g0h = gate0h_t33(cen, gmaps_t32)
    assert g0h.passes
    assert g0h.t32_gate1_passes and g0h.t32_gate0b_mismatches == 0
    assert (g0h.fresh_cells, g0h.fresh_checks) == (530, 1590)
    assert g0h.fresh_mismatches == 0 and g0h.coverage_full


# =============================================================================
# PORTE 0a gravée — order-sensibilité au périmètre VIERGE (témoins a priori)
# =============================================================================

def test_gate0a_t33_engraved() -> None:
    """PASSE (gravé 2026-07-17), témoins choisis PAR CRITÈRE A PRIORI (premier
    ≥ 2 flips dans l'ordre d'énumération, cellule hors 3k = 2N — aucun incident
    ce tour) : star (15,9) `++++++++-+-----` gap primaire 3.148148e-01 ;
    grammaire (25,1) `(-)²³ + -` gap 2.402222e+00 ; multiset VACUOUS (0.0) ;
    pivot η = 0 bit-à-bit. 0a valide L'INSTRUMENT au terrain vierge, n'explore
    aucune classe."""
    assert pick_0a_witness_star() == (+1,) * 8 + (-1,) + (+1,) + (-1,) * 5
    assert pick_0a_witness_grammar() == (-1,) * 23 + (+1,) + (-1,)
    assert cell_of(pick_0a_witness_grammar()) == (25, 1)
    g0a = gate0a_t33()
    assert g0a.passes
    assert g0a.star_primary.is_order_sensitive
    assert g0a.star_primary.gap == pytest.approx(3.148148e-01, rel=1e-5)
    assert g0a.grammar_primary.is_order_sensitive
    assert g0a.grammar_primary.gap == pytest.approx(2.402222e+00, rel=1e-5)
    assert g0a.star_multiset.is_vacuous and g0a.star_multiset.gap == 0.0
    assert g0a.grammar_multiset.is_vacuous and g0a.grammar_multiset.gap == 0.0
    assert g0a.pivots_exact


# =============================================================================
# PORTE 1 gravée — compression sur DÉRIVATION (gate dure, avant gel)
# =============================================================================

def test_gate1_t33_compression_engraved() -> None:
    """PASSE (gravé) : #I < #profils STRICT sur CHACUN des 3 périmètres de
    dérivation — P : 3699/4560, 6215/8158, 218/220 ; S : 3762/4560, 6414/8158,
    218/220. (La trace fixe ENTIÈRE, elle, était 4462/4560, 7567/8158 et
    220/220 sur star = BIJECTIVE ⟹ barrée, leçon I3 — gravé au relais §α.)"""
    grammar = enumerate_grammar()
    total = enumerate_total()
    star = [p for p in total if cell_of(p) == STAR_CELL]
    g1 = gate1_t33((("grammaire", grammar), ("total", total), ("star", star)))
    assert g1[("P_trunc_band", "grammaire")] == (3699, 4560)
    assert g1[("P_trunc_band", "total")] == (6215, 8158)
    assert g1[("P_trunc_band", "star")] == (218, 220)
    assert g1[("S_trunc_band_runs", "grammaire")] == (3762, 4560)
    assert g1[("S_trunc_band_runs", "total")] == (6414, 8158)
    assert g1[("S_trunc_band_runs", "star")] == (218, 220)
    assert all(n_i < n for (n_i, n) in g1.values())


# =============================================================================
# PORTE 2 gravée — soundness sur dérivation (exploratoire) : paires-témoins
# =============================================================================

def test_gate2_derivation_witnesses_engraved() -> None:
    """Unsound dès la dérivation (prédiction de mort DÉCLARÉE avant tir,
    relais §β) — paires-témoins exactes : P : `++----` Δ=−1/6 vs `+-----` Δ=0
    (η=1/2, même I_P) ; S : `+++----` Δ=−2/7 vs `++-+---` Δ=0 (η=1/2, même I_S).
    Et la baseline tautologique sur star : 35/30/28 paires (jamais cartouche)."""
    h = ETAS_T32["1/2"]
    a, b = (+1, +1, -1, -1, -1, -1), (+1, -1, -1, -1, -1, -1)
    assert index_trunc_band(a) == index_trunc_band(b)
    assert delta_nom_exact_profile(list(a), h) == Fraction(-1, 6)
    assert delta_nom_exact_profile(list(b), h) == Fraction(0)
    c, d = (+1, +1, +1, -1, -1, -1, -1), (+1, +1, -1, +1, -1, -1, -1)
    assert index_trunc_band_runs(c) == index_trunc_band_runs(d)
    assert delta_nom_exact_profile(list(c), h) == Fraction(-2, 7)
    assert delta_nom_exact_profile(list(d), h) == Fraction(0)
    star = enumerate_cell(*STAR_CELL)
    base = {lbl: baseline_count(star, ETAS_T32[lbl]) for lbl in ETA_LABELS_T33}
    assert base == {"1/2": 35, "1": 30, "4": 28}


# =============================================================================
# PORTE 3 gravée — VALIDATION ONE-SHOT : mort NETTE des 2 cartouches
# (échantillons-témoins exacts ; la validation entière n'est pas rejouée)
# =============================================================================

# (profil_a, profil_b, η, Δ_a, Δ_b) — même index P ET même index S, Δ différents.
_VALIDATION_KILL_WITNESSES = [
    # V-GRAMMAIRE[25,40] — les 3 η (mort à chaque η ⟹ mort au sens strict (ii))
    ((-1,) * 24 + (+1,), (-1,) * 20 + (+1,) * 5,
     "1/2", Fraction(-1, 25), Fraction(0)),
    ((-1,) * 17 + (+1,) + (-1,) * 7, (-1,) * 16 + (+1,) * 2 + (-1,) * 7,
     "1", Fraction(-1, 25), Fraction(0)),
    ((-1,) * 17 + (+1,) + (-1,) * 7, (-1,) * 16 + (+1,) * 2 + (-1,) * 7,
     "4", Fraction(-1, 25), Fraction(0)),
    # V-STAR (15,9)
    ((+1,) * 8 + (-1,) * 5 + (+1,) + (-1,),
     (+1,) * 7 + (-1,) + (+1,) + (-1,) * 4 + (+1,) + (-1,),
     "1/2", Fraction(1, 3), Fraction(2, 5)),
    ((+1,) * 9 + (-1,) * 6, (+1,) * 8 + (-1,) * 2 + (+1,) + (-1,) * 4,
     "4", Fraction(7, 15), Fraction(1, 3)),
    # V-TOTAL[13,14]
    ((+1,) * 10 + (-1,) * 2 + (+1,), (+1,) * 9 + (-1,) + (+1,) + (-1,) + (+1,),
     "1/2", Fraction(6, 13), Fraction(7, 13)),
    # V-TOTAL-15 (optionnel, joué : mur tenu — 66 s < 1800 s)
    ((+1,) * 13 + (-1,) * 2, (+1,) * 12 + (-1,) + (+1,) + (-1,),
     "1/2", Fraction(2, 3), Fraction(3, 5)),
]


def test_gate3_validation_kill_witnesses_engraved() -> None:
    """MORT NETTE gravée (one-shot 2026-07-17, aucune retouche après tir) :
    sur CHAQUE sous-périmètre vierge il existe une paire même-I à Δ différents
    — pour P ET pour S, aux η gravés (V-GRAMMAIRE : les 3 η ⟹ aucun η commun
    sound, critère de survie strict (ii) violé pour les deux cartouches)."""
    for a, b, lbl, da, db in _VALIDATION_KILL_WITNESSES:
        eta = ETAS_T32[lbl]
        assert index_trunc_band(a) == index_trunc_band(b), (a, b)
        assert index_trunc_band_runs(a) == index_trunc_band_runs(b), (a, b)
        assert delta_nom_exact_profile(list(a), eta) == da
        assert delta_nom_exact_profile(list(b), eta) == db
        assert da != db


def test_gate3_vstar_full_engraved() -> None:
    """V-STAR (15,9) rejoué intégralement (le seul sous-périmètre assez petit) :
    compression tenue (P : 4935/5005 ; S : 4975/5005) mais soundness FAUSSE aux
    3 η — fractions exactes gravées : P 26/108, 38/108, 16/108 ; S 18/42,
    32/42, 9/42 (paires même-I à Δ égal / paires même-I). #Δ = 16/15/15."""
    vs = enumerate_v_star()
    expected = {
        ("P_trunc_band", "1/2"): ((26, 108), 45, 16),
        ("P_trunc_band", "1"): ((38, 108), 40, 15),
        ("P_trunc_band", "4"): ((16, 108), 49, 15),
        ("S_trunc_band_runs", "1/2"): ((18, 42), 16, 16),
        ("S_trunc_band_runs", "1"): ((32, 42), 10, 15),
        ("S_trunc_band_runs", "4"): ((9, 42), 20, 15),
    }
    n_i = {"P_trunc_band": 4935, "S_trunc_band_runs": 4975}
    for lbl in ETA_LABELS_T33:
        dm = delta_map(vs, ETAS_T32[lbl])
        for cname, fn in CARTRIDGES_T33:
            v = index_verdict(vs, dm, fn, name=cname, perimeter="V-STAR",
                              eta_label=lbl)
            pairs, bad, nd = expected[(cname, lbl)]
            assert not v.report.sound
            assert v.report.soundness_pairs == pairs
            assert v.report.n_unsound_groups == bad
            assert v.report.n_delta_values == nd
            assert v.report.n_i_values == n_i[cname] < 5005   # compressif
            assert v.report.refines_nk
            assert v.witness is not None


# =============================================================================
# PORTE 4 — conditionnelle : SKIP DÉCLARÉ (aucun survivant)
# =============================================================================

def test_gate4_skip_declared() -> None:
    """Aucun cartouche ne survit la porte 3 (mort aux 3 η sur chaque
    sous-périmètre) ⟹ la porte 4 est SKIP DÉCLARÉ — rien à prédire, zéro
    donnée neuve consommée. (Le témoin V-GRAMMAIRE aux 3 η ci-dessus est la
    preuve exacte de non-survie au sens strict (ii).)"""
    etas_with_kill = {lbl for _a, _b, lbl, _da, _db in _VALIDATION_KILL_WITNESSES}
    assert etas_with_kill == set(ETA_LABELS_T33)
