"""Tests Tour 31 — moteur exact par PROFIL + échappement à (N, k) (``profile_exact``).

H31a : ``delta_nom_exact_profile`` (récurrence Fraction, p_ref incrémenté par le
SIGNE o_t — la spec EXACTE de ``phi_ref_increments`` gelée) reproduit l'instrument
float GELÉ sur les 44 profils du corpus hors-canon × 3 η, hors bord-exact
pré-déclarés (offset prédit exactement, doctrine T29/T30 généralisée).
H31b : ∃ profil non-canonique tel que Δ_profile ≠ Δ_canon(N, k_total) — TRANCHÉ
(mesure 2026-07-13, gravée ici). Portes : A recensement, A′ prédictions Fraction,
0a pré-validation généralisée (T22/T23 applicable : le tour porte sur l'ordre),
B non-régression 228 × 3, C reproduction d'instrument, D escape par forme × η.

Le corpus ``corpus_horscanon_aba.txt`` vit HORS du dépôt git spiraton (racine du
répertoire de travail, comme dataset_aba/corpus_claude) : skip propre s'il est
absent (même convention que ``test_eta_runners``). AUCUN ``.so`` requis ; tout
déterministe (seuls les shuffles seedés de la porte 0a, infra T23).
"""
import math
from fractions import Fraction
from pathlib import Path

import pytest

from spiraton.diagnostics.horizon_law import profile_of, trace_e_exact
from spiraton.diagnostics.spectral_map import ETAS_FROZEN, delta_nom_exact_eta
from spiraton.diagnostics.eta_runners import FROZEN_OFFSETS, OFFSETS_ETA_INVARIANT
from spiraton.diagnostics.profile_exact import (
    CORPUS_HORSCANON,
    ETA_LABELS_T31,
    ETAS_T31,
    EXPECTED_FORM_COUNTS,
    N_MEASURABLE_EXPECTED,
    band_edge_touches_profile,
    boundary_offset_analysis,
    census_horscanon,
    classify_form,
    corpus_horscanon_path,
    delta_canon_exact,
    delta_nom_exact_profile,
    delta_nom_float_profile,
    escapes_canon,
    gate0a_t31,
    gateB_nonregression,
    gateC_instrument,
    gateD_escape,
    k_total,
    profile_predictions,
    trace_e_exact_profile,
)

_CORPUS = corpus_horscanon_path()
_HAVE_CORPUS = _CORPUS.is_file()

# lignes de fixture (quatuor 4/4 — recensement déterministe SANS corpus réel)
_LINE_F2 = (
    "<SEG_A> <ADD><DX><OUT><ALPHA> un deux trois </SEG_A> "
    "<SEG_B> <ADD><LV><IN><OMEGA> quatre cinq </SEG_B> "
    "<SEG_A_PRIME> <ADD><DX><OUT><A_PRIME> six sept huit<EOL> </SEG_A_PRIME> <EOL>"
)
_LINE_F4 = (
    "<SEG_A> <SUB><DX><OUT><ALPHA> un deux trois quatre </SEG_A> "
    "<SEG_B> <SUB><DX><OUT><OMEGA> </SEG_B> "
    "<SEG_A_PRIME> <SUB><LV><IN><A_PRIME> cinq six sept huit neuf<EOL> </SEG_A_PRIME> <EOL>"
)
_LINE_F5 = (
    "<SEG_A> <MUL><DX><OUT><ALPHA> un deux </SEG_A> "
    "<SEG_B> <MUL><DX><OUT><OMEGA> trois quatre </SEG_B> "
    "<SEG_A_PRIME> <MUL><DX><OUT><A_PRIME> cinq six<EOL> </SEG_A_PRIME> <EOL>"
)


# =============================================================================
# Quatuor 1/4 — protocole gelé : η, formes attendues, effectifs
# =============================================================================

def test_frozen_protocol() -> None:
    """Les 3 η gelés (sous-ensemble des gravures T28), le recensement annoncé §3."""
    assert ETA_LABELS_T31 == ("1/2", "1", "4")
    assert ETAS_T31 == {"1/2": Fraction(1, 2), "1": Fraction(1), "4": Fraction(4)}
    for lbl, eta in ETAS_T31.items():
        assert eta == ETAS_FROZEN[lbl]            # lecture seule, jamais redéfinis
        assert float(eta) == eta                  # dyadiques exacts au float
    assert EXPECTED_FORM_COUNTS == {
        "F0": 11, "F2": 14, "F3": 8, "F1": 5, "F1b": 2, "F4": 4, "F5": 2}
    assert sum(EXPECTED_FORM_COUNTS.values()) == 46
    assert N_MEASURABLE_EXPECTED == 46 - EXPECTED_FORM_COUNTS["F5"] == 44
    assert CORPUS_HORSCANON == "corpus_horscanon_aba.txt"


def test_classify_form_frozen_taxonomy() -> None:
    """La taxonomie F0-F5 de l'émission §1 ; triplet inconnu ⟹ ValueError (ARRÊT)."""
    assert classify_form((+1, +1, -1), (5, 6, 5)) == "F0"
    assert classify_form((+1, -1, +1), (5, 6, 6)) == "F2"
    assert classify_form((-1, +1, -1), (5, 5, 6)) == "F3"
    assert classify_form((-1, +1, +1), (7, 6, 7)) == "F1"
    assert classify_form((-1, -1, +1), (3, 5, 6)) == "F1b"
    assert classify_form((+1, +1, -1), (4, 0, 5)) == "F4"    # SEG_B vide d'abord
    assert classify_form((+1, +1, +1), (2, 3, 4)) == "F5"
    assert classify_form((-1, -1, -1), (3, 4, 5)) == "F5"
    with pytest.raises(ValueError):
        classify_form((+1, -1, -1), (4, 4, 5))               # hors taxonomie gelée


# =============================================================================
# Quatuor 2/4 — formule exacte sous profil forcé (récurrence vérifiée à la main)
# =============================================================================

def test_trace_exact_profile_hand_computed() -> None:
    """o = [+1,−1,+1,+1] (N=4, n_plus=3 ⟹ inc+ = 8/9, inc− = 4/3), η = 1/2 :
    e = [1, 10/9, 13/18, 11/12, 83/72] ; fixe (η=0) : e = [1, 10/9, 7/9, 8/9, 1].
    Les deux f_edge valent 1 ⟹ Δ_profile = 0 (calcul longhand, gravé)."""
    o = [+1, -1, +1, +1]
    e_organ = trace_e_exact_profile(o, Fraction(1, 2))
    assert e_organ == [Fraction(1), Fraction(10, 9), Fraction(13, 18),
                       Fraction(11, 12), Fraction(83, 72)]
    e_fixed = trace_e_exact_profile(o, Fraction(0))
    assert e_fixed == [Fraction(1), Fraction(10, 9), Fraction(7, 9),
                       Fraction(8, 9), Fraction(1)]
    assert delta_nom_exact_profile(o, Fraction(1, 2)) == 0
    assert k_total(o) == 3


def test_canonical_profile_equals_grid_engine() -> None:
    """Sur ``profile_of(N, k)`` la trace PROFIL == la trace (N, k) de T27,
    Fraction par Fraction (le signe o_t et ``t < k`` coïncident)."""
    for (n, k) in [(7, 4), (13, 8), (16, 12), (9, 4)]:
        prof = profile_of(n, k)
        for eta in (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(4)):
            assert trace_e_exact_profile(prof, eta) == trace_e_exact(n, k, eta)


def test_canonical_arrangement_never_escapes() -> None:
    """Arrangement canonique (mono-flip) ⟹ Δ_profile == Δ_canon PAR CONSTRUCTION
    — la garde structurelle derrière « F4 ne doit pas échapper »."""
    for (n, k) in [(9, 4), (12, 6), (11, 5), (16, 11)]:
        prof = profile_of(n, k)
        for lbl in ETA_LABELS_T31:
            assert not escapes_canon(prof, ETAS_T31[lbl])
            assert delta_canon_exact(prof, ETAS_T31[lbl]) == delta_nom_exact_eta(
                n, k, ETAS_T31[lbl])


def test_delta_canon_requires_a_flip() -> None:
    """Profil sans flip (F5) : Δ_canon indéfini — ValueError, jamais un 0 silencieux."""
    with pytest.raises(ValueError):
        delta_canon_exact([+1, +1, +1, +1], Fraction(1, 2))
    with pytest.raises(ValueError):
        delta_canon_exact([-1, -1, -1], Fraction(1))


# =============================================================================
# Quatuor 3/4 — finitude / types exacts
# =============================================================================

def test_finite_and_exact_types() -> None:
    """Δ_profile est une Fraction bornée ; le float de l'instrument est fini."""
    o = [+1, +1, +1, +1, -1, -1, +1, +1, +1, +1, +1, -1, -1, -1]   # un F2 synthétique
    for lbl in ETA_LABELS_T31:
        eta = ETAS_T31[lbl]
        d = delta_nom_exact_profile(o, eta)
        assert isinstance(d, Fraction) and abs(d) <= 1
        f = delta_nom_float_profile(o, float(eta))
        assert math.isfinite(f)
        assert isinstance(band_edge_touches_profile(o, eta), int)
        bo = boundary_offset_analysis(o, eta)
        assert isinstance(bo.offset, Fraction)


# =============================================================================
# Quatuor 4/4 — déterminisme / recensement sur fixture (sans corpus réel)
# =============================================================================

def test_census_deterministic_on_fixture(tmp_path) -> None:
    """Deux recensements identiques ; F5 exclue par le filtre, F2/F4 mesurables ;
    cohérence segment→profil et accord avec collect_profiles vérifiés."""
    p = tmp_path / "mini_horscanon.txt"
    p.write_text("# provenance : fixture de test\n" + _LINE_F2 + "\n"
                 + _LINE_F4 + "\n" + _LINE_F5 + "\n", encoding="utf-8")
    c1 = census_horscanon(str(p))
    c2 = census_horscanon(str(p))
    assert c1 == c2
    assert c1.n_cycles == 3
    assert c1.form_counts == {"F2": 1, "F4": 1, "F5": 1}
    assert not c1.counts_match_emission              # fixture ≠ corpus gelé : honnête
    assert c1.n_measurable == 2
    assert c1.profiles_coherent and c1.measurable_match_collect
    r_f2, r_f4, r_f5 = c1.records
    assert (r_f2.form, r_f2.n, r_f2.k, r_f2.arrangement) == ("F2", 8, 6, "+3|-2|+3")
    assert (r_f4.form, r_f4.n, r_f4.k, r_f4.arrangement) == ("F4", 9, 4, "+4|+0|-5")
    assert (r_f5.form, r_f5.measurable) == ("F5", False)


# =============================================================================
# PORTE B — non-régression : moteur PROFIL == moteur (N, k), 228 cellules × 3 η
# =============================================================================

def test_gateB_nonregression_full_grid() -> None:
    """``delta_nom_exact_profile(profile_of(N,k), η) == delta_nom_exact_eta(N,k,η)``
    sur TOUTES les cellules de la grille T27 × les 3 η gelés — égalité de
    Fraction exacte, zéro divergence (un écart serait un ARRÊT, bug du moteur)."""
    assert gateB_nonregression() == []


# =============================================================================
# PORTE A gravée — recensement du corpus gelé (mesuré 2026-07-13)
# =============================================================================

@pytest.fixture(scope="module")
def census():
    """Le recensement du corpus gelé, calculé UNE fois (déterministe)."""
    if not _HAVE_CORPUS:
        pytest.skip("corpus_horscanon_aba.txt indisponible")
    return census_horscanon(str(_CORPUS))


_FROZEN_CENSUS = (
    # (index, forme, N, k_total) — gravé depuis la mesure porte A (2026-07-13)
    (0, "F0", 16, 11), (1, "F0", 16, 11), (2, "F0", 15, 11), (3, "F0", 15, 10),
    (4, "F0", 15, 11), (5, "F0", 17, 11), (6, "F0", 16, 11), (7, "F0", 16, 12),
    (8, "F0", 13, 8), (9, "F0", 15, 10), (10, "F0", 12, 9),
    (11, "F2", 17, 11), (12, "F2", 17, 12), (13, "F2", 16, 11), (14, "F2", 14, 10),
    (15, "F2", 15, 10), (16, "F2", 16, 11), (17, "F2", 16, 10), (18, "F2", 15, 11),
    (19, "F2", 14, 9), (20, "F2", 18, 12), (21, "F2", 12, 9), (22, "F2", 15, 9),
    (23, "F2", 16, 11), (24, "F2", 17, 11),
    (25, "F3", 16, 5), (26, "F3", 17, 5), (27, "F3", 16, 5), (28, "F3", 15, 5),
    (29, "F3", 20, 7), (30, "F3", 18, 6), (31, "F3", 18, 6), (32, "F3", 18, 7),
    (33, "F1", 20, 13), (34, "F1", 15, 10), (35, "F1", 16, 11), (36, "F1", 16, 10),
    (37, "F1", 18, 12),
    (38, "F1b", 14, 6), (39, "F1b", 13, 6),
    (40, "F4", 9, 4), (41, "F4", 11, 5), (42, "F4", 11, 5), (43, "F4", 12, 6),
    (44, "F5", 9, 9), (45, "F5", 12, 0),
)


@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gateA_census_frozen(census) -> None:
    """Recensement GELÉ (gravé) : 46 cycles, formes == émission §3, 44 mesurables,
    F5 exclues par le filtre, profils cohérents et alignés sur collect_profiles."""
    assert census.n_cycles == 46
    assert census.form_counts == EXPECTED_FORM_COUNTS
    assert census.counts_match_emission
    assert census.n_measurable == N_MEASURABLE_EXPECTED == 44
    assert census.profiles_coherent
    assert census.measurable_match_collect
    assert tuple((r.index, r.form, r.n, r.k) for r in census.records) == _FROZEN_CENSUS
    for r in census.records:
        assert r.measurable == (r.form != "F5")     # SEULES les F5 tombent au filtre
    # arrangements-témoins (l'objet du tour est l'ARRANGEMENT, pas les comptes)
    by_idx = {r.index: r for r in census.records}
    assert by_idx[12].arrangement == "+5|-5|+7"     # F2 : repart au lieu de revenir
    assert by_idx[25].arrangement == "-5|+5|-6"     # F3 : repli, émission, clôture
    assert by_idx[38].arrangement == "-3|-5|+6"     # F1b : double repli puis jaillir
    assert by_idx[40].arrangement == "+4|+0|-5"     # F4 : le déploiement absent


# =============================================================================
# PORTE 0a gravée — pré-validation généralisée (T22/T23 applicable ce tour)
# =============================================================================

@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gate0a_generalized(census) -> None:
    """PASSE (gravé) : primaire order-sensible sur F2 #11 (gap 3.0006e-2) ET F3 #25
    (gap 5.0227e-1), tous deux ≥ δ_min = 1e-4 ; multiset VACUOUS (0.0) sur les
    deux ; pivot η = 0 bit-à-bit sur les 29 profils non-canoniques."""
    g0 = gate0a_t31(census)
    assert g0.passes
    assert g0.cell_f2 == (11, 17) and g0.cell_f3 == (25, 16)
    assert g0.primary_f2.is_order_sensitive and g0.primary_f3.is_order_sensitive
    assert g0.primary_f2.gap == pytest.approx(3.000594e-02, rel=1e-5)
    assert g0.primary_f3.gap == pytest.approx(5.022727e-01, rel=1e-5)
    assert g0.multiset_f2.is_vacuous and g0.multiset_f2.gap == 0.0
    assert g0.multiset_f3.is_vacuous and g0.multiset_f3.gap == 0.0
    assert g0.n_noncanonical == 29                  # 14 F2 + 8 F3 + 5 F1 + 2 F1b
    assert g0.pivot_eta0_all_exact


# =============================================================================
# PORTE C gravée — reproduction d'instrument (H31a), 44 × 3 η
# =============================================================================

@pytest.fixture(scope="module")
def checks(census):
    """Les 132 comparaisons float ↔ exact, calculées UNE fois (déterministes)."""
    return gateC_instrument(census)


@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gateC_instrument_reproduced(checks) -> None:
    """H31a TIENT (gravé) : 132/132 ok — hors bord-exact, |Δ_float − Δ_exact| ≤
    1e-12 ; sur les 28 comparaisons à bord-exact, l'offset MESURÉ == l'offset
    PRÉDIT par l'émulation aux seuls points de bord, et AUCUN point hors bord
    n'est classé différemment par le float (``nonboundary_agree`` partout)."""
    assert len(checks) == 44 * 3 == 132
    assert all(c.ok for c in checks)
    assert all(c.nonboundary_agree for c in checks)
    n_edge = sum(1 for c in checks if c.band_touches > 0)
    assert n_edge == 28
    # hors bord-exact, l'offset prédit est 0 par construction
    for c in checks:
        if c.band_touches == 0:
            assert c.offset_predicted == 0


@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gateC_nonzero_offsets_frozen(checks) -> None:
    """Les 10 offsets NON NULS mesurés (gravés) — et leur COHÉRENCE T28 : sur les
    profils à arrangement canonique, l'offset retrouve EXACTEMENT la gravure
    T28/T30 de la cellule ((15,11) → +1/15 η-invariant ; (16,12), (12,9), (9,4)
    → −1/16, −1/12, −1/9 à η = 4). #36 (F1, non-canonique) est un bord-exact
    NEUF : −1/16 à η = 4, hors de toute gravure antérieure."""
    nonzero = {(c.index, c.eta_label): c.offset_predicted
               for c in checks if c.offset_predicted != 0}
    assert nonzero == {
        (2, "1/2"): Fraction(1, 15), (2, "1"): Fraction(1, 15), (2, "4"): Fraction(1, 15),
        (4, "1/2"): Fraction(1, 15), (4, "1"): Fraction(1, 15), (4, "4"): Fraction(1, 15),
        (7, "4"): Fraction(-1, 16),
        (10, "4"): Fraction(-1, 12),
        (36, "4"): Fraction(-1, 16),
        (40, "4"): Fraction(-1, 9),
    }
    # cohérence lecture-seule avec les gravures T28/T30 (arrangements canoniques)
    assert OFFSETS_ETA_INVARIANT[(15, 11)] == Fraction(1, 15)         # #2, #4 (F0)
    assert FROZEN_OFFSETS["4"][(16, 12)] == Fraction(-1, 16)          # #7 (F0)
    assert FROZEN_OFFSETS["4"][(12, 9)] == Fraction(-1, 12)           # #10 (F0)
    assert FROZEN_OFFSETS["4"][(9, 4)] == Fraction(-1, 9)             # #40 (F4)
    assert (16, 10) not in FROZEN_OFFSETS["4"]                        # #36 : NEUF


# =============================================================================
# PORTE D gravée — escape (H31b) : le réel ÉCHAPPE à (N, k_total)
# =============================================================================

@pytest.fixture(scope="module")
def summary(census):
    """Prédictions A′ (Fraction pur) + résumé D, calculés UNE fois."""
    preds = profile_predictions(census)
    return preds, gateD_escape(preds)


@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gateD_escape_counts_frozen(summary) -> None:
    """H31b VRAIE (gravé) : comptes d'échappés par forme × η —
    η=1/2 : F2 6/14, F3 6/8, F1b 2/2, F1 0/5 ; η=1 : F2 6/14, F3 8/8, F1b 2/2,
    F1 0/5 ; η=4 : F2 8/14, F3 7/8, F1b 1/2, F1 2/5. F0 et F4 : 0 partout
    (arrangement canonique ⟹ jamais d'échappement, garde structurelle)."""
    _, summ = summary
    assert summ.escape_found
    assert summ.f4_escaped == []                    # sinon ARRÊT (bug)
    expected = {
        ("F0", "1/2"): (0, 11), ("F0", "1"): (0, 11), ("F0", "4"): (0, 11),
        ("F1", "1/2"): (0, 5), ("F1", "1"): (0, 5), ("F1", "4"): (2, 5),
        ("F1b", "1/2"): (2, 2), ("F1b", "1"): (2, 2), ("F1b", "4"): (1, 2),
        ("F2", "1/2"): (6, 14), ("F2", "1"): (6, 14), ("F2", "4"): (8, 14),
        ("F3", "1/2"): (6, 8), ("F3", "1"): (8, 8), ("F3", "4"): (7, 8),
        ("F4", "1/2"): (0, 4), ("F4", "1"): (0, 4), ("F4", "4"): (0, 4),
    }
    assert summ.by_form_eta == expected


@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gateD_witness_values_frozen(summary) -> None:
    """Les Δ-témoins les plus parlants (gravés, Fraction exacte) :
    #21 F2 (12,9) à η=1/2 : Δ_profile = −1/2 vs Δ_canon = +1/2 — même (N,k) que
    #10 F0 (Δ = +1/2) : l'arrangement RENVERSE le signe à excursion maximale.
    #28 F3 (15,5) à η=1 : −1/15 (canon) → +1/15 (profil) — bascule de signe.
    #38 F1b (14,6) à η=1/2 : 3/7 vs 0 — le canon dit « rien », le profil régule."""
    preds, _ = summary
    d = {(p.index, p.eta_label): p for p in preds}
    p21 = d[(21, "1/2")]
    assert (p21.delta_profile, p21.delta_canon) == (Fraction(-1, 2), Fraction(1, 2))
    assert p21.escapes
    p10 = d[(10, "1/2")]
    assert (p10.delta_profile, p10.delta_canon) == (Fraction(1, 2), Fraction(1, 2))
    assert not p10.escapes and (p10.n, p10.k) == (p21.n, p21.k) == (12, 9)
    p28 = d[(28, "1")]
    assert (p28.delta_profile, p28.delta_canon) == (Fraction(1, 15), Fraction(-1, 15))
    p38 = d[(38, "1/2")]
    assert (p38.delta_profile, p38.delta_canon) == (Fraction(3, 7), Fraction(0))
    # F4 : Δ_profile == Δ_canon EXACT sur les 4 profils, aux 3 η
    for idx in (40, 41, 42, 43):
        for lbl in ETA_LABELS_T31:
            p = d[(idx, lbl)]
            assert p.form == "F4" and p.delta_profile == p.delta_canon


@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gateD_witness_pairs_same_cell(summary) -> None:
    """Paires témoins mêmes (N, k_total) à Δ distincts : présentes aux 3 η —
    dont (12,9) et (15,11) à chaque η (gravé). La variable hors (N, k) est
    visible SUR PAIRE CONCRÈTE, pas seulement contre le canon synthétique."""
    _, summ = summary
    cells_by_eta = {}
    for lbl, n, k, _members in summ.witness_pairs:
        cells_by_eta.setdefault(lbl, set()).add((n, k))
    for lbl in ETA_LABELS_T31:
        assert (12, 9) in cells_by_eta[lbl]
        assert (15, 11) in cells_by_eta[lbl]
        assert (16, 10) in cells_by_eta[lbl]
