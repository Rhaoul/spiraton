"""Tests Tour 32 — cartographie des arrangements (``arrangement_map``).

H32 : ∃ un invariant compact I(o) dans la liste CLOSE {I1, I2, I3} (+I1b si I1
échoue) tel que, à η fixé, I(o) = I(o′) ⟹ Δ(o;η) = Δ(o′;η) (soundness 100 %),
raffinant strictement (N,k). VERDICT MESURÉ (2026-07-16, gravé ici) : **AUCUN
invariant compact ne classe** — I1 (excursion-fixe), I2 (positions de flips) et
I1b (multiset des paliers) sont FAUX tels qu'énoncés (fractions de soundness
gravées, jamais arrondies) ; I3 (RLE) est sound mais TRIVIALEMENT (encodage
bijectif du profil : compression nulle, #I == #profils partout) — c'est la
borne haute « profil entier », pas un invariant compact. La CARTE EXACTE des
classes est le livrable (issue honnête n°3 de l'émission) : granularité gravée,
dont la ligne dégénérée 3k = 2N (inc+ = inc− = 1) à granularité 1 sur les DEUX
périmètres à TOUS les η — le seul lieu où (N,k) reste suffisant.

INCIDENT 0a GRAVÉ : le premier témoin N=24 gelé, (24,16), vivait sur la ligne
dégénérée 3k=2N — order-invariant PAR CONSTRUCTION (vacuité dérivable a priori,
doctrine T23) ; remplacé par (24,17) AVANT toute dérivation de carte (portes
lexicographiques), incident déclaré au module et testé ici.

Corpus gelé T31 hors du dépôt git : skip propre s'il est absent (convention
T24-T31). Les runs porte 4 exigent aussi dataset_aba/corpus_claude. Tout est
déterministe (seuls les shuffles seedés de la porte 0a, infra T23).
"""
import math
from fractions import Fraction
from pathlib import Path

import pytest

from spiraton.diagnostics.horizon_law import profile_of
from spiraton.diagnostics.spectral_map import ETAS_FROZEN
from spiraton.diagnostics.profile_exact import census_horscanon, corpus_horscanon_path
from spiraton.diagnostics.arrangement_map import (
    DERIV_N_MAX,
    ETA_LABELS_T32,
    ETA_PRIMARY_T32,
    ETAS_T32,
    FROZEN_T31_OFFSETS,
    GRAMMAR_N_MAX,
    GRAMMAR_N_MIN,
    GRAMMAR_TRIPLETS,
    GRAMMAR24_NONCANON_T32,
    INVARIANT_I1B,
    INVARIANTS_T32,
    STAR_CELL,
    STAR_NONCANON_T32,
    TOTAL_N_MAX,
    TOTAL_N_MIN,
    VALID_N_MIN,
    cell_of,
    classes_of_cell,
    delta_map,
    enumerate_grammar,
    enumerate_total,
    gate0a_t32,
    gate0b_t32,
    gate1_t32,
    gate4_t32,
    granularity_by_cell,
    inv_flip_positions,
    inv_palier_multiset,
    inv_peak_fixed,
    inv_rle,
    invariant_report,
    is_order_degenerate_cell,
    perimeter_cells,
)

_CORPUS = corpus_horscanon_path()
_HAVE_CORPUS = _CORPUS.is_file()
_ROOT = Path("F:/code/claude/spiraton-enhanced")
if not (_ROOT / "dataset_aba.txt").is_file():
    _ROOT = Path(__file__).resolve().parents[2]
_HAVE_RUNNERS = ((_ROOT / "dataset_aba.txt").is_file()
                 and (_ROOT / "corpus_claude_aba.txt").is_file())


# =============================================================================
# fixtures de module — cartes calculées UNE fois (déterministes)
# =============================================================================

@pytest.fixture(scope="module")
def grammar():
    return enumerate_grammar()


@pytest.fixture(scope="module")
def total():
    return enumerate_total()


@pytest.fixture(scope="module")
def gmaps(grammar):
    return {lbl: delta_map(grammar, ETAS_T32[lbl]) for lbl in ETA_LABELS_T32}


@pytest.fixture(scope="module")
def tmaps(total):
    return {lbl: delta_map(total, ETAS_T32[lbl]) for lbl in ETA_LABELS_T32}


# =============================================================================
# Quatuor 1/4 — protocole gelé
# =============================================================================

def test_frozen_protocol_t32() -> None:
    """η gelés (lecture seule des gravures T28), périmètres, split, star, triplets."""
    assert ETA_LABELS_T32 == ("1/2", "1", "4") and ETA_PRIMARY_T32 == "1/2"
    for lbl, eta in ETAS_T32.items():
        assert eta == ETAS_FROZEN[lbl]
        assert float(eta) == eta                  # dyadiques exacts au float
    assert (GRAMMAR_N_MIN, GRAMMAR_N_MAX) == (6, 24)
    assert (TOTAL_N_MIN, TOTAL_N_MAX) == (4, 12)
    assert (DERIV_N_MAX, VALID_N_MIN) == (18, 19)
    assert STAR_CELL == (12, 9)
    assert len(GRAMMAR_TRIPLETS) == 6
    assert (+1, -1, -1) in GRAMMAR_TRIPLETS       # le 6e triplet, hors taxonomie F0-F5
    assert (+1, +1, +1) not in GRAMMAR_TRIPLETS
    assert (-1, -1, -1) not in GRAMMAR_TRIPLETS


def test_gate0a_witnesses_not_degenerate() -> None:
    """INCIDENT T32 gravé : le témoin (24,16) initial vivait sur 3k = 2N (inc+ =
    inc− = 1 : order-invariant par construction, vacuité T23 dérivable a priori) ;
    les témoins retenus (12,9) et (24,17) sont HORS ligne dégénérée."""
    assert cell_of(STAR_NONCANON_T32) == (12, 9)
    assert cell_of(GRAMMAR24_NONCANON_T32) == (24, 17)
    assert not is_order_degenerate_cell(*cell_of(STAR_NONCANON_T32))
    assert not is_order_degenerate_cell(*cell_of(GRAMMAR24_NONCANON_T32))
    # la ligne dégénérée elle-même (le témoin rejeté y vivait)
    assert is_order_degenerate_cell(24, 16)
    assert is_order_degenerate_cell(12, 8)
    assert is_order_degenerate_cell(6, 4)
    assert not is_order_degenerate_cell(12, 9)


# =============================================================================
# Quatuor 2/4 — énumération : comptes EXACTS, déterminisme, formes
# =============================================================================

def test_enumeration_exact_counts(grammar, total) -> None:
    """Grammaire : N(N−1) profils distincts par N (2 blocs : 2(N−1) ; 3 blocs :
    (N−1)(N−2)), 4560 au total ; dérivation 1898 / validation 2662. Total :
    Σ(2^N − 2) = 8158 ; star (12,9) : C(12,9) = 220. Dédup déterministe."""
    assert len(grammar) == sum(n * (n - 1) for n in range(6, 25)) == 4560
    by_n = {}
    for p in grammar:
        by_n[len(p)] = by_n.get(len(p), 0) + 1
    for n in range(6, 25):
        assert by_n[n] == n * (n - 1)
    assert len([p for p in grammar if len(p) <= DERIV_N_MAX]) == 1898
    assert len([p for p in grammar if len(p) >= VALID_N_MIN]) == 2662
    assert len(total) == sum(2 ** n - 2 for n in range(4, 13)) == 8158
    assert len([p for p in total if cell_of(p) == STAR_CELL]) == 220
    assert enumerate_grammar() == grammar         # déterminisme (double appel)
    for p in grammar:
        assert (+1 in p) and (-1 in p)            # deux signes présents (strict (i))
        assert len(inv_rle(p)) in (2, 3)          # ≤ 3 blocs, ≥ 1 flip
    # les canoniques [+]^k[−]^{N−k} sont dans la grammaire (repli (N,k))
    gset = set(grammar)
    for (n, k) in [(6, 2), (12, 9), (24, 1), (24, 23)]:
        assert tuple(profile_of(n, k)) in gset


# =============================================================================
# Quatuor 3/4 — invariants : valeurs exactes calculées à la main
# =============================================================================

def test_invariants_hand_computed() -> None:
    """o = (+1,−1,+1,+1) : trace fixe [1, 10/9, 7/9, 8/9, 1] (gravée T31) ⟹
    I1 = 2/9, I1b = () (aucun palier hors bande) ; flips en 1 et 2 ⟹ I2 = (1,2) ;
    I3 = ((+1,1),(−1,1),(+1,2)).  profile_of(6,2) = ++−−−− : trace fixe
    [1, 0, −1, −1/2, 0, 1/2, 1] ⟹ I1 = 2, I1b = (−1, −1/2, 0, 0),
    I2 = (2,2), I3 = ((+1,2),(−1,4))."""
    o = (+1, -1, +1, +1)
    assert inv_peak_fixed(o) == Fraction(2, 9)
    assert inv_flip_positions(o) == (1, 2)
    assert inv_rle(o) == ((+1, 1), (-1, 1), (+1, 2))
    assert inv_palier_multiset(o) == ()

    p = tuple(profile_of(6, 2))
    assert inv_peak_fixed(p) == Fraction(2)
    assert inv_palier_multiset(p) == (Fraction(-1), Fraction(-1, 2),
                                      Fraction(0), Fraction(0))
    assert inv_flip_positions(p) == (2, 2)
    assert inv_rle(p) == ((+1, 2), (-1, 4))


# =============================================================================
# Quatuor 4/4 — finitude / types exacts / déterminisme de la carte
# =============================================================================

def test_delta_map_exact_types_and_determinism() -> None:
    """Δ est une Fraction bornée ; deux constructions de carte sont identiques."""
    profs = [tuple(profile_of(7, 3)), STAR_NONCANON_T32, (+1, -1, +1, +1)]
    for lbl in ETA_LABELS_T32:
        m1 = delta_map(profs, ETAS_T32[lbl])
        m2 = delta_map(profs, ETAS_T32[lbl])
        assert m1 == m2
        for d in m1.values():
            assert isinstance(d, Fraction) and abs(d) <= 1
            assert math.isfinite(float(d))


# =============================================================================
# PORTE 0a gravée — order-sensibilité au périmètre (δ_min = 1e-4)
# =============================================================================

def test_gate0a_t32_engraved() -> None:
    """PASSE (gravé 2026-07-16) : primaire order-sensible sur la star (12,9)
    non canonique (gap 1.018519e-1) ET sur la grammaire N=24 (24,17)
    (gap 1.453081e-1), tous deux ≥ δ_min = 1e-4 ; multiset VACUOUS (0.0) sur
    les deux ; pivot η = 0 bit-à-bit sur les deux profils-témoins."""
    g0 = gate0a_t32()
    assert g0.passes
    assert g0.star_primary.is_order_sensitive
    assert g0.star_primary.gap == pytest.approx(1.018519e-01, rel=1e-5)
    assert g0.n24_primary.is_order_sensitive
    assert g0.n24_primary.gap == pytest.approx(1.453081e-01, rel=1e-5)
    assert g0.star_multiset.is_vacuous and g0.star_multiset.gap == 0.0
    assert g0.n24_multiset.is_vacuous and g0.n24_multiset.gap == 0.0
    assert g0.pivots_exact


# =============================================================================
# PORTE 0b — non-régression moteur profil == moteur (N,k) au périmètre
# =============================================================================

def test_gate0b_t32_nonregression() -> None:
    """273 cellules (union grammaire ∪ total) × 3 η : zéro divergence Fraction."""
    assert len(perimeter_cells()) == 273
    assert gate0b_t32() == []


# =============================================================================
# PORTE 1 gravée — témoins T31 retrouvés (énumérateur + corpus gelé)
# =============================================================================

@pytest.fixture(scope="module")
def census():
    if not _HAVE_CORPUS:
        pytest.skip("corpus_horscanon_aba.txt indisponible")
    return census_horscanon(str(_CORPUS))


@pytest.mark.skipif(not _HAVE_CORPUS, reason="corpus_horscanon_aba.txt indisponible")
def test_gate1_t32_witnesses(census, gmaps) -> None:
    """PASSE (gravé) : (12,9) η=1/2 {+1/2, −1/2} en deux classes ; (16,10) η=4
    {−1/8, +1/8} ; (14,6) η=1/2 {0, 3/7} ; valeurs corpus exactes ; les 10
    offsets bord-exact T31 retrouvés à l'identique."""
    g1 = gate1_t32(census, gmaps)
    assert g1.passes
    assert g1.star_ok and g1.mirror_ok and g1.zero_ok and g1.corpus_ok
    assert g1.offsets == FROZEN_T31_OFFSETS
    assert len(g1.offsets) == 10


# =============================================================================
# PORTE 2 gravée — granularité de la carte (grammaire ≠ total, jamais fusionnés)
# =============================================================================

_EXPECTED_MULTI_GRAMMAR = {"1/2": 247, "1": 248, "4": 259}   # cellules ≥ 2 classes /266
_EXPECTED_MULTI_TOTAL = {"1/2": 53, "1": 57, "4": 60}        # /63
_EXPECTED_N_DELTA_TOTAL = {"1/2": 59, "1": 61, "4": 66}      # # valeurs Δ distinctes

# la cellule star (12,9), périmètre TOTAL, η = 1/2 : les 12 classes gravées
_STAR_CLASSES_HALF = {
    Fraction(-1, 12): 42, Fraction(-1, 2): 1, Fraction(-1, 3): 24,
    Fraction(-1, 4): 34, Fraction(-1, 6): 26, Fraction(-5, 12): 7,
    Fraction(0): 58, Fraction(1, 12): 7, Fraction(1, 2): 2,
    Fraction(1, 3): 5, Fraction(1, 4): 3, Fraction(1, 6): 11,
}


def test_gate2_granularity_engraved(gmaps, tmaps) -> None:
    """La carte brute (gravée 2026-07-16) : 266 cellules grammaire dont
    247/248/259 à ≥ 2 classes (η = 1/2, 1, 4) ; 63 cellules total dont
    53/57/60 ; star (12,9) : 12 classes / 220 arrangements aux 3 η, classes
    η = 1/2 gravées intégralement ; grammaire (12,9) : 10 classes / 12."""
    for lbl in ETA_LABELS_T32:
        gg = granularity_by_cell(gmaps[lbl])
        tt = granularity_by_cell(tmaps[lbl])
        assert len(gg) == 266 and len(tt) == 63
        assert sum(1 for c, _m in gg.values() if c >= 2) == _EXPECTED_MULTI_GRAMMAR[lbl]
        assert sum(1 for c, _m in tt.values() if c >= 2) == _EXPECTED_MULTI_TOTAL[lbl]
        assert len(set(tmaps[lbl].values())) == _EXPECTED_N_DELTA_TOTAL[lbl]
        star = classes_of_cell(tmaps[lbl], STAR_CELL)
        assert len(star) == 12 and sum(star.values()) == 220
    assert classes_of_cell(tmaps["1/2"], STAR_CELL) == _STAR_CLASSES_HALF
    assert granularity_by_cell(gmaps["1/2"])[(12, 9)] == (10, 12)


def test_gate2_degenerate_line_single_class(gmaps, tmaps) -> None:
    """POST-HOC étiqueté, avec dérivation a priori : sur la ligne 3k = 2N les
    incréments valent inc+ = inc− = 1 ⟹ la trace est INDÉPENDANTE de
    l'arrangement ⟹ granularité 1. Vérifié sur les DEUX périmètres, aux 3 η —
    le SEUL lieu structurel où (N,k) reste suffisant."""
    for lbl in ETA_LABELS_T32:
        for maps, cells in ((gmaps, range(GRAMMAR_N_MIN, GRAMMAR_N_MAX + 1)),
                            (tmaps, range(TOTAL_N_MIN, TOTAL_N_MAX + 1))):
            gran = granularity_by_cell(maps[lbl])
            for n in cells:
                if (2 * n) % 3 == 0:
                    k = 2 * n // 3
                    if (n, k) in gran:
                        assert gran[(n, k)][0] == 1


# =============================================================================
# PORTE 3 gravée — le verdict des invariants (fractions EXACTES, jamais arrondies)
# =============================================================================

def test_gate3_i1_faux_engraved(grammar, gmaps) -> None:
    """I1 (excursion-fixe) est FAUX tel qu'énoncé (gravé) : en dérivation
    (grammaire N ∈ [6,18], η = 1/2), 2652/12869 paires même-I à Δ égal
    (fraction exacte 156/757 ≈ 20.6 %, 286 groupes non uniformes) — loin des
    100 % exigés. Il raffine (N,k) mais ne classe pas ; l'ajout de (N,k)
    (intra-cellule) ne le sauve pas : 700/1390."""
    deriv = [p for p in grammar if len(p) <= DERIV_N_MAX]
    r = invariant_report(deriv, gmaps["1/2"], inv_peak_fixed,
                         name="I1_peak_fixed", perimeter="deriv", eta_label="1/2")
    assert not r.sound
    assert r.soundness_pairs == (2652, 12869)
    assert Fraction(*r.soundness_pairs) == Fraction(156, 757)
    assert r.n_unsound_groups == 286
    assert r.refines_nk
    assert not r.sound_within_cell
    assert r.within_cell_pairs == (700, 1390)
    assert (r.n_profiles, r.n_i_values, r.n_delta_values) == (1898, 391, 121)


def test_gate3_i2_faux_engraved(grammar, gmaps) -> None:
    """I2 (positions de flips) est FAUX (gravé) : dérivation η = 1/2,
    1378/15353 paires même-I à Δ égal (≈ 9 %), 153 groupes non uniformes."""
    deriv = [p for p in grammar if len(p) <= DERIV_N_MAX]
    r = invariant_report(deriv, gmaps["1/2"], inv_flip_positions,
                         name="I2_flip_positions", perimeter="deriv", eta_label="1/2")
    assert not r.sound
    assert r.soundness_pairs == (1378, 15353)
    assert r.n_unsound_groups == 153
    assert r.refines_nk and not r.sound_within_cell


def test_gate3_i3_sound_mais_trivial(grammar, total, gmaps, tmaps) -> None:
    """I3 (RLE) est sound PARTOUT mais TRIVIALEMENT : l'encodage (signe,
    longueur)* est BIJECTIF — #I == #profils sur chaque périmètre (compression
    NULLE : c'est la borne haute « profil entier », pas un invariant compact).
    Incomplet partout (des profils distincts partagent Δ)."""
    deriv = [p for p in grammar if len(p) <= DERIV_N_MAX]
    for profs, dmap, peri in ((deriv, gmaps["1/2"], "deriv"),
                              (total, tmaps["1/2"], "total")):
        r = invariant_report(profs, dmap, inv_rle,
                             name="I3_rle", perimeter=peri, eta_label="1/2")
        assert r.sound                            # vacuous : groupes singletons
        assert r.soundness_pairs == (0, 0)        # AUCUNE paire même-I distincte
        assert r.n_i_values == r.n_profiles       # compression nulle (bijection)
        assert not r.complete
        assert r.refines_nk


def test_gate3_i1b_active_et_faux(grammar, gmaps) -> None:
    """I1 ayant échoué, la variante I1b (multiset des paliers hors bande) est
    activée (clause émission §4) — et FAUSSE aussi (gravé) : dérivation
    η = 1/2, 14777/23031 paires (≈ 64 %), 88 groupes non uniformes."""
    deriv = [p for p in grammar if len(p) <= DERIV_N_MAX]
    r = invariant_report(deriv, gmaps["1/2"], inv_palier_multiset,
                         name="I1b_palier_multiset", perimeter="deriv",
                         eta_label="1/2")
    assert not r.sound
    assert r.soundness_pairs == (14777, 23031)
    assert r.n_unsound_groups == 88
    assert not r.sound_within_cell


def test_gate3_verdict_aucun_invariant_compact(grammar, total, gmaps, tmaps) -> None:
    """VERDICT H32 gravé : parmi les candidats CLOS, aucun invariant COMPACT ne
    classe à 100 % (I1/I2/I1b faux ; I3 = identité déguisée). L'issue honnête
    n°3 de l'émission s'applique : le profil est irréductiblement une séquence
    au regard de cette liste ; la carte exacte des classes EST le livrable."""
    deriv = [p for p in grammar if len(p) <= DERIV_N_MAX]
    compact = [("I1_peak_fixed", inv_peak_fixed),
               ("I2_flip_positions", inv_flip_positions),
               ("I1b_palier_multiset", inv_palier_multiset)]
    for lbl in ETA_LABELS_T32:
        for name, fn in compact:
            r = invariant_report(deriv, gmaps[lbl], fn, name=name,
                                 perimeter="deriv", eta_label=lbl)
            assert not r.sound, f"{name} sound à η={lbl} : contredit la gravure"
    # I3 : sound mais sans compression — sur le total aussi (bijection)
    r3 = invariant_report(total, tmaps["4"], inv_rle, name="I3_rle",
                          perimeter="total", eta_label="4")
    assert r3.sound and r3.n_i_values == r3.n_profiles


# =============================================================================
# PORTE 4 gravée — retour au réel (zéro donnée neuve)
# =============================================================================

@pytest.mark.skipif(not (_HAVE_CORPUS and _HAVE_RUNNERS),
                    reason="corpus gelés T24-T31 indisponibles")
def test_gate4_t32_engraved(census, gmaps) -> None:
    """PASSE (gravé) : les 44 profils mesurables du corpus hors-canon sont TOUS
    dans l'énumération grammaire, Δ dans les classes 132/132 ; les 2116 profils
    canoniques T24/T25/T26 (tous mono-flip) sont TOUS dans l'énumération,
    Δ dans les classes 6348/6348. Seul I3 (= le profil) est éligible comme
    table de prédiction — trivialement 132/132 et 6348/6348."""
    g4 = gate4_t32(census, gmaps, {"I3_rle": inv_rle})
    assert g4.passes
    assert g4.n_corpus_measurable == g4.n_corpus_in_grammar == 44
    assert g4.corpus_delta_ok == (132, 132)
    assert g4.n_real == g4.n_real_mono == g4.n_real_in_grammar == 2116
    assert g4.real_delta_ok == (6348, 6348)
    assert g4.corpus_pred["I3_rle"] == (132, 132)
    assert g4.real_pred["I3_rle"] == (6348, 6348)
