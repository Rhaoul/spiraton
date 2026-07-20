"""Tests Tour 35 — grammaire ABA étendue (``spiraton/data/aba_forms.py``).

Portes gelées (émission T35, ordre strict, premier échec = verdict) :

* P1 TOTALITÉ : ``classify_form_extended`` ne lève sur AUCUN profil parseable
  (8 triplets × compositions N ∈ [6, 24], + variantes B-vide et longueurs
  limites) — zéro exception, comptes par forme exacts.
* P2 COHÉRENCE RÉTROGRADE : sur tout le domaine où le ``classify_form`` gelé
  T31 ne lève pas, nouveau == ancien (identité 100 %) ; l'UNIQUE différence de
  domaine est (+,−,−)-B-non-vide → « F0b » (l'ancien levait ValueError).
* P3 NON-RÉGRESSION : les 2116 cycles canoniques des 3 runners → F0 (100 %) ;
  les 46 cycles hors-canon T31 → étiquettes T31 identiques (skip propre si les
  corpus, hors dépôt git, sont absents — convention ``test_eta_runners``).
* P4 ROUND-TRIP : forme → triplet → classification = identité pour les 6
  formes non uniformes, sur un éventail de longueurs GELÉ a priori
  (``LENS_FROZEN``). F4/F5 : dégénérescence documentée (hors porte).
* P5 DOCS : chaque forme de ``GRAMMAIRE_ABA.md`` a son test nommé ici
  (``test_form_*`` — table §5 du document).

Quatuor adapté : pureté/déterminisme (fonction pure, aucune sortie aléatoire),
formes simples et limites (B vide, segments vides), valeurs exactes (tables
gelées), cohérence rétrograde exhaustive. AUCUNE mesure de Δ ce tour (le tour
NOMME des arrangements, il ne re-mesure pas leur dynamique — émission §5) ;
la pré-validation shuffle T22/T23 est N-A, déclarée telle.
"""
from typing import Dict, List, Tuple

import pytest

from spiraton.data.aba_forms import (
    ALL_FORMS,
    FORM_TRIPLETS,
    FORMS,
    POLE_TRIPLETS,
    TRIPLET_FORMS,
    classify_form_extended,
    describe_form,
    form_profile,
    form_to_signs,
    pole_of,
    roundtrip_form,
)
from spiraton.diagnostics.profile_exact import (
    EXPECTED_FORM_COUNTS,
    census_horscanon,
    classify_form,
    corpus_horscanon_path,
)
from spiraton.diagnostics.arrangement_map import (
    GRAMMAR_N_MAX,
    GRAMMAR_N_MIN,
    enumerate_grammar,
)

_CORPUS_HC = corpus_horscanon_path()
_HAVE_HC = _CORPUS_HC.is_file()

# --- constantes GELÉES a priori (déclarées AVANT toute exécution) -------------

#: Les 8 triplets de signes (2³) — l'espace complet, uniformes inclus.
ALL_TRIPLETS: Tuple[Tuple[int, int, int], ...] = (
    (+1, +1, +1), (+1, +1, -1), (+1, -1, +1), (+1, -1, -1),
    (-1, +1, +1), (-1, +1, -1), (-1, -1, +1), (-1, -1, -1),
)

#: Éventail de longueurs GELÉ pour la porte 4 (choisi a priori, jamais ajusté) :
#: minimal, hétérogène, symétrique, homogène, B dominant, B minimal.
LENS_FROZEN: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 1), (2, 3, 4), (6, 5, 6), (8, 8, 8), (1, 9, 1), (10, 1, 10),
)

#: Table de VALEURS EXACTES gelée : triplet → forme (les 6 non uniformes).
FROZEN_SIX: Dict[Tuple[int, int, int], str] = {
    (+1, +1, -1): "F0",
    (+1, -1, -1): "F0b",
    (-1, +1, +1): "F1",
    (-1, -1, +1): "F1b",
    (+1, -1, +1): "F2",
    (-1, +1, -1): "F3",
}


def _compositions(n: int) -> List[Tuple[int, int, int]]:
    """Toutes les compositions (a, b, c) de n avec a ≥ 1, b ≥ 0, c ≥ 1."""
    return [(a, b, n - a - b) for a in range(1, n) for b in range(0, n - a)]


# =============================================================================
# Quatuor 1/4 — tables gelées, valeurs exactes
# =============================================================================

def test_frozen_tables() -> None:
    """Bijection triplet ↔ nom EXACTE (6 formes) + pôles + les 9 noms."""
    assert FORM_TRIPLETS == {f: t for t, f in FROZEN_SIX.items()}
    assert TRIPLET_FORMS == FROZEN_SIX
    assert len(TRIPLET_FORMS) == 6                 # bijection : aucune collision
    assert POLE_TRIPLETS == {"F5+": (+1, +1, +1), "F5-": (-1, -1, -1)}
    assert ALL_FORMS == ("F0", "F0b", "F1", "F1b", "F2", "F3", "F4", "F5+", "F5-")
    # les 8 triplets = 6 non uniformes + 2 pôles, sans recouvrement
    assert set(ALL_TRIPLETS) == set(FROZEN_SIX) | set(POLE_TRIPLETS.values())


def test_classifier_pure_deterministic() -> None:
    """Fonction PURE : mêmes entrées ⇒ même sortie, entrées non mutées."""
    signs, lens = [+1, -1, -1], [3, 2, 4]
    out1 = classify_form_extended(signs, lens)
    out2 = classify_form_extended(tuple(signs), tuple(lens))
    assert out1 == out2 == "F0b"
    assert signs == [+1, -1, -1] and lens == [3, 2, 4]   # jamais mutées


# =============================================================================
# Quatuor 2/4 — chaque forme, individuellement (porte 5 : symbole ⇒ test)
# =============================================================================

def test_form_F0_canonical() -> None:
    """F0 (+,+,−) : la clôture canonique — mono-flip tardif, finit repliée."""
    assert classify_form_extended((+1, +1, -1), (5, 6, 5)) == "F0"
    info = describe_form("F0")
    assert info.family == "mono-flip" and info.flip_position == "tardif"
    assert info.ending == "repliée" and info.measurable


def test_form_F0b_new_citizen() -> None:
    """F0b (+,−,−) : la NOUVELLE citoyenne — l'ancien classificateur levait.

    Test SYNTHÉTIQUE (aucune ligne F0b en corpus — étiquette honnête T35) :
    triplet construit, pas parsé. L'ancien ``classify_form`` gelé DOIT lever
    ici (c'est la lacune que F0b ferme), le nouveau DOIT nommer.
    """
    assert classify_form_extended((+1, -1, -1), (3, 4, 3)) == "F0b"
    with pytest.raises(ValueError):
        classify_form((+1, -1, -1), (3, 4, 3))    # instrument T31, gelé
    info = describe_form("F0b")
    assert info.family == "mono-flip" and info.flip_position == "précoce"
    assert info.ending == "repliée" and info.measurable
    assert "non encore attestée" in info.attestation   # honnêteté gravée


def test_form_F1_receive_first() -> None:
    """F1 (−,+,+) : le retour précède l'aller — mono-flip précoce, finit émise."""
    assert classify_form_extended((-1, +1, +1), (4, 4, 4)) == "F1"
    info = describe_form("F1")
    assert info.flip_position == "précoce" and info.ending == "émise"


def test_form_F1b_double_intake() -> None:
    """F1b (−,−,+) : double repli puis émission — mono-flip tardif, finit émise."""
    assert classify_form_extended((-1, -1, +1), (4, 4, 4)) == "F1b"
    info = describe_form("F1b")
    assert info.flip_position == "tardif" and info.ending == "émise"
    # miroir de F0b : renversement TEMPOREL (triplet lu à rebours), émission §2
    assert tuple(reversed(info.triplet)) == describe_form("F0b").triplet


def test_form_F2_reopens() -> None:
    """F2 (+,−,+) : repart au lieu de revenir — bi-flip, cycle ouvert."""
    assert classify_form_extended((+1, -1, +1), (3, 5, 3)) == "F2"
    info = describe_form("F2")
    assert info.family == "bi-flip" and info.n_flips == 2 and info.ending == "émise"


def test_form_F3_speaks_once() -> None:
    """F3 (−,+,−) : le repli qui émet une fois puis se referme — bi-flip."""
    assert classify_form_extended((-1, +1, -1), (3, 5, 3)) == "F3"
    info = describe_form("F3")
    assert info.family == "bi-flip" and info.n_flips == 2 and info.ending == "repliée"


def test_form_F4_empty_B() -> None:
    """F4 (+, ⟨vide⟩, −) : déploiement absent — |B| = 0, o_B fantôme.

    Règle 1 (identique à l'instrument T31) : (o_A=+1, o_A′=−1, |B|=0) ⇒ F4
    QUEL QUE SOIT le signe fantôme o_B — (+,+,−) comme (+,−,−).
    """
    assert classify_form_extended((+1, +1, -1), (4, 0, 5)) == "F4"
    assert classify_form_extended((+1, -1, -1), (4, 0, 5)) == "F4"
    info = describe_form("F4")
    assert info.b_vide and info.triplet is None and info.family == "modificateur"


def test_form_F5_plus_dissipation() -> None:
    """F5+ (+,+,+) : pôle dissipation — nommé, NON mesurable, agrégé « F5 »."""
    assert classify_form_extended((+1, +1, +1), (3, 3, 3)) == "F5"
    assert pole_of((+1, +1, +1)) == "F5+"
    info = describe_form("F5+")
    assert info.pole == "dissipation" and not info.measurable and info.n_flips == 0


def test_form_F5_minus_verrouillage() -> None:
    """F5− (−,−,−) : pôle verrouillage — nommé, NON mesurable, agrégé « F5 »."""
    assert classify_form_extended((-1, -1, -1), (3, 3, 3)) == "F5"
    assert pole_of((-1, -1, -1)) == "F5-"
    info = describe_form("F5-")
    assert info.pole == "verrouillage" and not info.measurable and info.n_flips == 0


def test_invalid_inputs_raise() -> None:
    """Hors espace parseable : signes ∉ {±1} ou longueurs < 0 ⇒ ValueError."""
    with pytest.raises(ValueError):
        classify_form_extended((+1, 0, -1), (1, 1, 1))
    with pytest.raises(ValueError):
        classify_form_extended((+1, +1), (1, 1, 1))
    with pytest.raises(ValueError):
        classify_form_extended((+1, +1, -1), (1, -1, 1))
    with pytest.raises(ValueError):
        pole_of((+1, -1, +1))                      # non uniforme : pas un pôle
    with pytest.raises(ValueError):
        describe_form("F9")


# =============================================================================
# Quatuor 3/4 — PORTES 1 et 2 : totalité + cohérence rétrograde (exhaustives)
# =============================================================================

def test_gate1_totality() -> None:
    """P1 : zéro exception sur l'espace-grammaire seg-level COMPLET.

    Compositions (a ≥ 1, b ≥ 0, c ≥ 1) de N ∈ [6, 24] × les 8 triplets
    (= 2280 × 8 = 18 240 classifications, variantes B-vide incluses), plus le
    balayage limite lens ∈ [0, 3]³ × 8 triplets (segments vides A/B/A′ —
    l'espace parseable d'``aba.py`` admet un texte vide). Comptes par forme
    EXACTS (dérivés a priori, gravés).
    """
    counts: Dict[str, int] = {}
    for n in range(GRAMMAR_N_MIN, GRAMMAR_N_MAX + 1):
        for lens in _compositions(n):
            for trip in ALL_TRIPLETS:
                form = classify_form_extended(trip, lens)   # ne doit JAMAIS lever
                counts[form] = counts.get(form, 0) + 1
    assert sum(counts.values()) == 18240
    assert counts == {
        "F0": 2014, "F0b": 2014, "F1": 2280, "F1b": 2280,
        "F2": 2280, "F3": 2280, "F4": 532, "F5": 4560,
    }
    # balayage limite : TOUTES les longueurs de [0,3]³ (y compris (0,0,0))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for trip in ALL_TRIPLETS:
                    assert classify_form_extended(trip, (a, b, c)) in (
                        "F0", "F0b", "F1", "F1b", "F2", "F3", "F4", "F5")


def test_gate1_grammar_enumerator_coverage() -> None:
    """P1 (couverture) : l'expansion des 6 formes == espace-grammaire T32 (4560).

    ``enumerate_grammar`` (arrangement_map, LECTURE SEULE) est exactement
    l'ensemble des profils plats ``form_profile(f, (a,b,c))`` à deux signes
    présents — la grammaire seg-level et la carte T32 décrivent le même monde.
    """
    grammar = set(enumerate_grammar())
    assert len(grammar) == 4560
    expanded = set()
    for n in range(GRAMMAR_N_MIN, GRAMMAR_N_MAX + 1):
        for lens in _compositions(n):
            for f in FORM_TRIPLETS:
                prof = form_profile(f, lens)
                if (+1 in prof) and (-1 in prof):
                    expanded.add(prof)
    assert expanded == grammar


def test_gate2_retro_coherence() -> None:
    """P2 : identité 100 % avec le ``classify_form`` GELÉ T31 sur son domaine.

    Sur les 18 240 entrées seg-level : là où l'ancien ne lève pas, nouveau ==
    ancien (aucune divergence) ; là où il lève, l'entrée est EXACTEMENT
    (+,−,−)-B-non-vide et le nouveau rend « F0b ». Comptes exacts des deux
    populations : 16 226 identiques + 2 014 F0b = 18 240.
    """
    n_same = n_new = 0
    for n in range(GRAMMAR_N_MIN, GRAMMAR_N_MAX + 1):
        for lens in _compositions(n):
            for trip in ALL_TRIPLETS:
                new = classify_form_extended(trip, lens)
                try:
                    old = classify_form(trip, lens)
                except ValueError:
                    assert trip == (+1, -1, -1) and lens[1] > 0   # l'UNIQUE lacune
                    assert new == "F0b"
                    n_new += 1
                else:
                    assert new == old                             # identité pure
                    n_same += 1
    assert n_same == 16226
    assert n_new == 2014
    assert n_same + n_new == 18240


# =============================================================================
# Quatuor 4/4 — PORTES 3 et 4 : non-régression réelle + round-trip
# =============================================================================

@pytest.mark.skipif(not _HAVE_HC, reason="corpus_horscanon_aba.txt absent (hors dépôt git)")
def test_gate3_horscanon_labels() -> None:
    """P3 (hors-canon) : les 46 cycles T31 gardent leurs étiquettes, à 100 %.

    ``census_horscanon`` (instrument T31, lecture seule) fournit forme gelée +
    signes + longueurs par cycle ; le nouveau classificateur doit coïncider
    cycle par cycle (46/46 — aucun (+,−,−) dans ce corpus : le 6ᵉ triplet n'a
    jamais été écrit), comptes == recensement T31 (44 mesurables + 2 F5).
    """
    cen = census_horscanon(str(_CORPUS_HC))
    assert cen.n_cycles == 46
    for r in cen.records:
        assert classify_form_extended(r.seg_signs, r.seg_lens) == r.form
    assert cen.form_counts == EXPECTED_FORM_COUNTS
    assert cen.n_measurable == 44
    f5 = [r for r in cen.records if r.form == "F5"]
    assert len(f5) == 2 and all(not r.measurable for r in f5)
    for r in f5:
        assert pole_of(r.seg_signs) in ("F5+", "F5-")


def test_gate3_canonical_all_F0() -> None:
    """P3 (canonique) : les 2116 cycles des 3 runners T24/T25/T26 ⇒ F0, 100 %.

    Réplique EXACTE de la sélection des runners (``_iter_cycles`` +
    filtre N ≥ MIN_TOKENS et deux orientations, cap n_cycles) — lecture seule.
    Skip propre si les corpus (hors dépôt git) sont absents.
    """
    from pathlib import Path

    from spiraton.diagnostics.eta_runners import corpus_runs
    from spiraton.diagnostics.structural_regulation import MIN_TOKENS, _iter_cycles
    from spiraton.experimental.structural_gap import orientation_profile
    from spiraton.diagnostics.profile_exact import segment_sign

    runs = corpus_runs()
    if not all(Path(p).is_file() for (_, p, _, _) in runs):
        pytest.skip("corpus canoniques absents (hors dépôt git)")
    total = 0
    per_run = {}
    for label, path, n_cycles, line_range in runs:
        n_used = 0
        for cycle in _iter_cycles(path, line_range):
            toks = orientation_profile(cycle)
            orients = {tk.orientation for tk in toks}
            if len(toks) >= MIN_TOKENS and orients == {+1, -1}:
                segs = (cycle.seg_a, cycle.seg_b, cycle.seg_a_prime)
                signs = tuple(segment_sign(s) for s in segs)
                lens = tuple(len(s.text.split()) for s in segs)
                assert classify_form_extended(signs, lens) == "F0"
                n_used += 1
            if n_used >= n_cycles:
                break
        per_run[label] = n_used
        total += n_used
    assert per_run == {"T24-defaut": 40, "T25-claude": 76, "T26-bloc26": 2000}
    assert total == 2116


def test_gate4_roundtrip() -> None:
    """P4 : forme → triplet → classification = IDENTITÉ pour les 6 non uniformes.

    Éventail de longueurs GELÉ a priori (``LENS_FROZEN``, |B| ≥ 1) :
    6 formes × 6 longueurs = 36 round-trips, identité 100 %. Le profil plat
    associé a la bonne longueur et les bons comptes.
    """
    n_ok = 0
    for form in FORM_TRIPLETS:
        for lens in LENS_FROZEN:
            assert roundtrip_form(form, lens) == form
            prof = form_profile(form, lens)
            assert len(prof) == sum(lens)
            signs = FORM_TRIPLETS[form]
            assert sum(1 for o in prof if o == +1) == sum(
                l for s, l in zip(signs, lens) if s == +1)
            n_ok += 1
    assert n_ok == 36


def test_roundtrip_degenerate_documented() -> None:
    """Dégénérescences HORS porte 4, documentées (GRAMMAIRE_ABA.md §4).

    F4 perd o_B (signe fantôme), F5± sont agrégées : ``form_to_signs`` lève.
    F0/F0b avec |B| = 0 retombent sur F4 (règle 1) : la dégénérescence B-vide.
    """
    for name in ("F4", "F5+", "F5-"):
        with pytest.raises(ValueError):
            form_to_signs(name)
    assert roundtrip_form("F0", (3, 0, 3)) == "F4"
    assert roundtrip_form("F0b", (3, 0, 3)) == "F4"
    # les 4 autres restent elles-mêmes à B vide (nom d'en-tête conservé)
    for name in ("F1", "F1b", "F2", "F3"):
        assert roundtrip_form(name, (3, 0, 3)) == name


def test_describe_form_all() -> None:
    """P5 : les 9 descripteurs existent, cohérents avec les tables gelées."""
    assert set(FORMS) == set(ALL_FORMS)
    for name in ALL_FORMS:
        info = describe_form(name)
        assert info.name == name
        assert info.semantics and info.attestation and info.corpus_refs
        if name in FORM_TRIPLETS:
            assert info.triplet == FORM_TRIPLETS[name]
            assert info.measurable
        if name in POLE_TRIPLETS:
            assert info.triplet == POLE_TRIPLETS[name]
            assert not info.measurable
    # familles : 4 mono-flip, 2 bi-flip, 1 modificateur, 2 pôles
    fams = [describe_form(n).family for n in ALL_FORMS]
    assert fams.count("mono-flip") == 4 and fams.count("bi-flip") == 2
    assert fams.count("modificateur") == 1 and fams.count("pôle") == 2
    # la famille mono-flip est COMPLÈTE : {précoce, tardif} × {repliée, émise}
    mono = [describe_form(n) for n in ALL_FORMS if describe_form(n).family == "mono-flip"]
    assert {(i.flip_position, i.ending) for i in mono} == {
        ("tardif", "repliée"), ("précoce", "repliée"),
        ("précoce", "émise"), ("tardif", "émise"),
    }
