from __future__ import annotations

"""Cartographie des arrangements — classes d'équivalence de Δ(o; η) (Tour 32).

H32 (émission linguiste, GELÉE avant toute dérivation — ``TOUR32_EMISSION.md``) :
il existe un invariant compact ``I(o)`` — pris dans la liste CLOSE {I1, I2, I3}
(+ variante I1b si I1 échoue) — tel que, À η FIXÉ,

    I(o) = I(o′)  ⟹  Δ(o; η) = Δ(o′; η)          (SOUNDNESS, critère 100 %)

sur le périmètre d'énumération gelé, et qui RAFFINE STRICTEMENT (N, k) (∃ une
cellule où I prend ≥ 2 valeurs — sinon I ≡ (N,k), réfuté T31). La COMPLÉTUDE
(Δ égal ⟹ I égal) est rapportée SÉPARÉMENT. Un invariant qui classe 99 % est
FAUX tel qu'énoncé — rapporté tel, jamais arrondi. Bornes de référence : basse
= (N,k) seul (réfutée T31) ; haute = le profil entier (aucune compression).

MOTEUR (garde-fou (b), LECTURE SEULE, zéro octet modifié) :
:func:`profile_exact.delta_nom_exact_profile` (T31, récurrence ``Fraction``
exacte, spec de l'instrument GELÉ ``structural_gap.py``) ; ``spectral_map.
delta_nom_exact_eta`` (T28) pour la non-régression ; ``boundary_offset_analysis``
(doctrine bord-exact T29/T30) pour les témoins offsets.

PÉRIMÈTRES D'ÉNUMÉRATION — BORNES EXACTES GELÉES (émission §5) :

  * GRAMMAIRE (là où vit le réel) : profils en ≤ 3 blocs — composition
    ``N = a + b + c`` (a ≥ 1, c ≥ 1, b ≥ 0) × 6 triplets de signes non uniformes
    ``(+,+,−) (+,−,+) (+,−,−) (−,+,+) (−,+,−) (−,−,+)`` — construits DIRECTEMENT
    (le 6e triplet (+,−,−) est HORS taxonomie F0-F5 de ``classify_form``, qui
    lèverait ValueError : jamais utilisée ici). ``N ∈ [6, 24]`` COMPLET,
    profils DÉDUPLIQUÉS. Compte exact attendu : N(N−1) profils distincts par N
    (2 blocs : 2(N−1) ; 3 blocs : (N−1)(N−2)) — rapporté à la porte 2.
  * TOTAL (sonde hors-grammaire) : toutes les séquences ±1 à DEUX signes
    présents, ``N ∈ [4, 12]`` COMPLET ; cellule star (12, 9) : C(12,9) = 220.

INTERPRÉTATIONS STRICTES DÉCLARÉES (points laissés ouverts par l'émission ;
choix les plus stricts, gravés ici AVANT toute mesure) :

  (i)   Les profils UNIFORMES issus de b = 0 avec triplet (+,−,+) ou (−,+,−)
        (un seul signe, aucun flip) sont EXCLUS du domaine : la relation
        d'équivalence est posée « à (N, k) et η fixés » avec 1 ≤ k ≤ N−1,
        Δ_canon y est indéfini (analogue F5, exclu par le filtre T31).
  (ii)  SOUNDNESS/COMPLÉTUDE jugées au périmètre GLOBAL (par η, par périmètre),
        la formule H32 ne restreignant pas à la cellule ; la version
        INTRA-CELLULE (I conditionné à (N,k), i.e. l'invariant ((N,k), I)) est
        rapportée en INFORMATION SECONDAIRE, jamais substituée au verdict.
  (iii) I1 : ``peak_fixed(o) = max_{t=1..N} |e_t^{fixe} − 1|`` (e_0 = 1 exclu,
        convention f_edge ; sans effet sur le max, qui est ≥ 0).
        I2 : positions 0-based ``i ∈ [1, N−1]`` avec ``o_i ≠ o_{i−1}`` ;
        couple (premier, dernier). I3 : tuple ordonné des blocs (signe,
        longueur) — encodage BIJECTIF du profil (borne haute rendue explicite).
        I1b (si I1 échoue seulement) : multiset trié des ``e_t^{fixe}`` (t ≥ 1)
        HORS bande (``|e_t − 1| > 1/2``).
  (iv)  Porte 0b : « cellules (N,k) du périmètre » = UNION des cellules
        touchées par les deux énumérations (273 cellules × 3 η).

η GELÉS : {1/2} PRIMAIRE + {1, 4} robustesse (sous-ensemble d'``ETAS_FROZEN``
T28, lecture seule). Invariant testé PAR-η ; η-uniformité = bonus rapporté.

SPLIT GELÉ A PRIORI : DÉRIVATION = grammaire N ∈ [6, 18] ; VALIDATION
(disjointe, invariants figés seulement) = grammaire N ∈ [19, 24] ∪ TOTAL
N ∈ [4, 12] en entier. Grammaire et total JAMAIS fusionnés (garde-fou (d)).

PORTES (ordre lexicographique STRICT, premier échec = verdict — émission §7) :

  0a. Order-sensibilité AU PÉRIMÈTRE (patron T22/T23/T31) : primaire
      ``obs_struct_frozen`` gap ≥ δ_min = 1e-4 sur un arrangement NON canonique
      de la star (12,9) ET un profil grammaire à N = 24 (gelés ci-dessous) ;
      multiset ``obs_struct_multiset`` gap = 0 VACUOUS requis ; pivot η = 0
      bit-à-bit (``reconstruct_profile(o,0) == reconstruct_fixed(o,g0)``).
  0b. Non-régression : ``delta_nom_exact_profile(profile_of(N,k), η) ==
      delta_nom_exact_eta(N,k,η)`` sur les 273 cellules × 3 η, zéro divergence.
  1.  Témoins T31 gravés retrouvés EXACTEMENT (Fraction) : (12,9) η=1/2 →
      {+1/2, −1/2} en deux classes ; (16,10) η=4 → {−1/8, +1/8} ; (14,6)
      η=1/2 → {0, 3/7} ; les 10 offsets bord-exact T31 (corpus gelé).
  2.  Carte brute : énumération COMPLÈTE, partition exacte par Δ (Fraction),
      granularité (nb classes / nb profils par cellule) rapportée AVANT toute
      interprétation, grammaire et total SÉPARÉS.
  3.  Invariants : soundness (100 % ou FAUX) / complétude / raffinement strict
      de (N,k), dérivation PUIS validation ; survie hors grammaire = information.
  4.  Retour au réel (JOUÉE, zéro donnée neuve) : les 44 profils mesurables du
      corpus hors-canon T31 + les 2116 canoniques T24-T26 tombent dans les
      classes prédites (Δ recalculés par le moteur, aucune mesure neuve).

GARDE-FOUS REFUS : (a) tout gelé avant énumération, aucun ajustement en cours
de mesure ; (b) moteur/instrument lecture seule ; (c) la relation est définie
sur Δ = f_edge(organe η) − f_edge(fixe g=1), pas d'oracle ; (d) grammaire ≠
total ; (e) anti-circularité — profils ±1 construits directement ou via
``aba.py``, JAMAIS les dims 0-5 du 33D, ``op_from_vector`` interdit.
DÉTERMINISME TOTAL : énumérations exhaustives ordonnées, aucun aléa hors
shuffles seedés de la porte 0a (infra T23).
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..experimental.structural_gap import (
    OrientedToken,
    obs_struct_frozen,
    obs_struct_multiset,
    shuffle_tokens,
)
from .horizon_law import BAND_EXACT, TARGET_EXACT, profile_of
from .instrument_validation import OrderSensitivityReport, assert_order_sensitive
from .spectral_map import ETAS_FROZEN, delta_nom_exact_eta
from .structural_regulation import collect_profiles, pivot_eta0_is_exact
from .eta_runners import corpus_runs, is_mono_flip
from .profile_exact import (
    HorsCanonCensus,
    boundary_offset_analysis,
    delta_nom_exact_profile,
    trace_e_exact_profile,
)


# --- constantes GELÉES A PRIORI (émission T32 — jamais ajustées après mesure) --------

Profile = Tuple[int, ...]

ETA_LABELS_T32: Tuple[str, ...] = ("1/2", "1", "4")   # {1/2} primaire, {1,4} robustesse
ETA_PRIMARY_T32 = "1/2"
ETAS_T32: Dict[str, Fraction] = {lbl: ETAS_FROZEN[lbl] for lbl in ETA_LABELS_T32}

GRAMMAR_N_MIN, GRAMMAR_N_MAX = 6, 24                  # périmètre grammaire, COMPLET
DERIV_N_MAX = 18                                      # dérivation : N ∈ [6, 18]
VALID_N_MIN = 19                                      # validation : N ∈ [19, 24]
TOTAL_N_MIN, TOTAL_N_MAX = 4, 12                      # périmètre total, COMPLET
STAR_CELL: Tuple[int, int] = (12, 9)                  # C(12,9) = 220 arrangements

# les 6 triplets de signes NON uniformes (émission §5 ; (+,−,−) inclus, construit
# DIRECTEMENT — jamais via classify_form qui lèverait ValueError sur ce triplet)
GRAMMAR_TRIPLETS: Tuple[Tuple[int, int, int], ...] = (
    (+1, +1, -1), (+1, -1, +1), (+1, -1, -1),
    (-1, +1, +1), (-1, +1, -1), (-1, -1, +1),
)

# profils-témoins de la porte 0a, GELÉS a priori (non canoniques, périmètre du tour).
# INCIDENT DÉCLARÉ (doctrine T23, pré-validation) : le premier témoin N = 24 choisi,
# (+1)¹⁰(−1)⁸(+1)⁶, a k = 16 = φ*·24 EXACTEMENT ⟹ inc(+1) = inc(−1) = 1 : sur la
# ligne dégénérée 3k = 2N la trace du lecteur fixe est ORDER-INVARIANTE PAR
# CONSTRUCTION (tout profil de la cellule) — témoin VACUOUS, rejeté comme test
# d'ordre A PRIORI (dérivation, pas mesure de carte : aucune carte n'était encore
# dérivée, portes lexicographiques). Remplacé par (24, 17), hors ligne dégénérée
# (inc+ = 16/17 ≠ inc− = 8/7). Les cellules 3k = 2N restent order-insensibles
# LOCALEMENT — le cas prévu par l'émission §11, acceptable ; un TÉMOIN de porte 0a
# ne doit simplement pas y vivre.
STAR_NONCANON_T32: Profile = (+1,) * 5 + (-1,) * 3 + (+1,) * 4          # (12, 9)
GRAMMAR24_NONCANON_T32: Profile = (+1,) * 10 + (-1,) * 7 + (+1,) * 7    # (24, 17)


def is_order_degenerate_cell(n: int, k: int) -> bool:
    """Cellule à incréments égaux : 3k = 2N ⟺ inc(+1) = inc(−1) = 1 (φ* = 2/3).

    Sur ces cellules, p_ref avance uniformément quel que soit l'ordre des
    tokens : TOUT observable du lecteur y est order-invariant par construction
    (vacuité dérivable a priori — doctrine T23). Un témoin de porte 0a ne doit
    pas y vivre ; la carte, elle, les inclut (order-insensibilité LOCALE,
    émission §11).
    """
    return 3 * k == 2 * n

# les 10 offsets bord-exact T31 (gravure test_gateC_nonzero_offsets_frozen, lecture
# seule) : (index corpus, η) → offset prédit — la porte 1 doit les RETROUVER.
FROZEN_T31_OFFSETS: Dict[Tuple[int, str], Fraction] = {
    (2, "1/2"): Fraction(1, 15), (2, "1"): Fraction(1, 15), (2, "4"): Fraction(1, 15),
    (4, "1/2"): Fraction(1, 15), (4, "1"): Fraction(1, 15), (4, "4"): Fraction(1, 15),
    (7, "4"): Fraction(-1, 16),
    (10, "4"): Fraction(-1, 12),
    (36, "4"): Fraction(-1, 16),
    (40, "4"): Fraction(-1, 9),
}


# --- énumérateurs (exhaustifs, dédupliqués, ordre déterministe) ----------------------

def enumerate_grammar(n_min: int = GRAMMAR_N_MIN, n_max: int = GRAMMAR_N_MAX
                      ) -> List[Profile]:
    """Espace-GRAMMAIRE : profils ≤ 3 blocs, dédupliqués, DEUX signes présents.

    Composition ``N = a + b + c`` (a ≥ 1, c ≥ 1, b ≥ 0) × 6 triplets non
    uniformes, construits directement en ±1. Les doublons (b = 0 notamment)
    sont fusionnés ; les profils UNIFORMES issus de b = 0 avec (+,−,+)/(−,+,−)
    sont EXCLUS (interprétation stricte (i) : pas de flip, hors carte (N,k)).
    Compte exact attendu : N(N−1) profils distincts par N. Ordre déterministe
    (longueur, puis ordre lexicographique du tuple).
    """
    out = set()
    for n in range(n_min, n_max + 1):
        for a in range(1, n):                 # a ≥ 1
            for b in range(0, n - a):         # b ≥ 0, c = n − a − b ≥ 1
                c = n - a - b
                for (s1, s2, s3) in GRAMMAR_TRIPLETS:
                    prof = (s1,) * a + (s2,) * b + (s3,) * c
                    if (+1 in prof) and (-1 in prof):
                        out.add(prof)
    return sorted(out, key=lambda p: (len(p), p))


def enumerate_total(n_min: int = TOTAL_N_MIN, n_max: int = TOTAL_N_MAX
                    ) -> List[Profile]:
    """Espace TOTAL : toutes les séquences ±1 à deux signes présents, N ∈ [4, 12].

    Exhaustif (2^N − 2 par N), ordre déterministe d'``itertools.product``.
    """
    out: List[Profile] = []
    for n in range(n_min, n_max + 1):
        for bits in product((+1, -1), repeat=n):
            if (+1 in bits) and (-1 in bits):
                out.append(bits)
    return out


def cell_of(profile: Profile) -> Tuple[int, int]:
    """La cellule (N, k_total) du profil : N = longueur, k = # de +1."""
    return len(profile), sum(1 for o in profile if o == +1)


# --- la carte : Δ exact par profil, classes par cellule ------------------------------

def delta_map(profiles: Sequence[Profile], eta: Fraction) -> Dict[Profile, Fraction]:
    """Δ(o; η) EXACT (Fraction) pour chaque profil — moteur T31 lecture seule."""
    return {p: delta_nom_exact_profile(list(p), eta) for p in profiles}


def granularity_by_cell(dmap: Dict[Profile, Fraction]
                        ) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Par cellule (N, k) : (nb de classes Δ distinctes, nb de profils)."""
    deltas_by_cell: Dict[Tuple[int, int], set] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for p, d in dmap.items():
        c = cell_of(p)
        deltas_by_cell.setdefault(c, set()).add(d)
        counts[c] = counts.get(c, 0) + 1
    return {c: (len(deltas_by_cell[c]), counts[c]) for c in sorted(counts)}


def classes_of_cell(dmap: Dict[Profile, Fraction], cell: Tuple[int, int]
                    ) -> Dict[Fraction, int]:
    """Les classes Δ d'une cellule : Δ → effectif (tri déterministe à l'affichage)."""
    out: Dict[Fraction, int] = {}
    for p, d in dmap.items():
        if cell_of(p) == cell:
            out[d] = out.get(d, 0) + 1
    return out


# --- invariants candidats — LISTE CLOSE (émission §4) --------------------------------

def inv_peak_fixed(profile: Profile) -> Fraction:
    """I1 — excursion-fixe : ``max_{t=1..N} |e_t^{fixe} − 1|`` (Fraction, η-indép.).

    Trace du LECTEUR FIXE (η = 0, g = 1) par ``trace_e_exact_profile`` (T31,
    lecture seule). Généralisation par profil du ``|k − φ*N|`` de T26/T29 —
    née de la mesure, GELÉE ICI a priori (recevabilité = ordre, garde-fou (a)).
    """
    e = trace_e_exact_profile(list(profile), Fraction(0))
    return max(abs(x - TARGET_EXACT) for x in e[1:])


def inv_flip_positions(profile: Profile) -> Tuple[int, int]:
    """I2 — (position du 1er flip, position du dernier flip), 0-based i ∈ [1, N−1].

    Un flip en position i = ``o_i ≠ o_{i−1}``. Les deux signes étant présents,
    au moins un flip existe. Capte OÙ la séquence dévie du bloc.
    """
    flips = [i for i in range(1, len(profile)) if profile[i] != profile[i - 1]]
    return flips[0], flips[-1]


def inv_rle(profile: Profile) -> Tuple[Tuple[int, int], ...]:
    """I3 — signature de bloc RLE : tuple ordonné des (signe, longueur).

    Encodage BIJECTIF du profil (borne haute « profil entier » rendue
    explicite) : sound par construction, compression nulle — mesuré tel quel.
    """
    blocks: List[Tuple[int, int]] = []
    cur, cnt = profile[0], 1
    for s in profile[1:]:
        if s == cur:
            cnt += 1
        else:
            blocks.append((cur, cnt))
            cur, cnt = s, 1
    blocks.append((cur, cnt))
    return tuple(blocks)


def inv_palier_multiset(profile: Profile) -> Tuple[Fraction, ...]:
    """I1b — multiset trié des paliers ``e_t^{fixe}`` (t ≥ 1) HORS bande.

    Variante de I1 (émission §4) : testée SEULEMENT si I1 échoue. Paliers
    « au-dessus de la bande » = ``|e_t − 1| > 1/2`` (strict, complément exact
    du comptage f_edge).
    """
    e = trace_e_exact_profile(list(profile), Fraction(0))
    return tuple(sorted(x for x in e[1:] if abs(x - TARGET_EXACT) > BAND_EXACT))


INVARIANTS_T32: Tuple[Tuple[str, Callable[[Profile], object]], ...] = (
    ("I1_peak_fixed", inv_peak_fixed),
    ("I2_flip_positions", inv_flip_positions),
    ("I3_rle", inv_rle),
)
INVARIANT_I1B: Tuple[str, Callable[[Profile], object]] = (
    "I1b_palier_multiset", inv_palier_multiset)


# --- porte 3 : soundness / complétude / raffinement ----------------------------------

@dataclass(frozen=True)
class InvariantReport:
    """Verdict d'un invariant sur un périmètre × un η (tout exact, rien d'arrondi).

    * ``sound`` : 100 % des paires même-I à Δ égal (critère H32) ;
      ``soundness_pairs`` = (paires même-I à Δ égal, paires même-I) — la
      fraction EXACTE est rapportée si ≠ 1, jamais arrondie.
    * ``complete`` : Δ égal ⟹ I égal (I induit exactement la partition Δ).
    * ``refines_nk`` : ∃ cellule (N,k) où I prend ≥ 2 valeurs.
    * ``sound_within_cell`` : soundness de l'invariant AUGMENTÉ ((N,k), I) —
      information secondaire (interprétation (ii)), jamais le verdict H32.
    * ``n_i_values`` vs ``n_profiles`` : la compression (I3 : aucune).
    """

    name: str
    perimeter: str
    eta_label: str
    n_profiles: int
    n_i_values: int
    n_delta_values: int
    sound: bool
    soundness_pairs: Tuple[int, int]
    n_unsound_groups: int
    complete: bool
    n_incomplete_groups: int
    refines_nk: bool
    sound_within_cell: bool
    within_cell_pairs: Tuple[int, int]
    n_unsound_cells: int


def _pair_soundness(groups: Iterable[List[Profile]],
                    dmap: Dict[Profile, Fraction]) -> Tuple[int, int, int]:
    """(paires même-groupe à Δ égal, paires même-groupe, groupes non uniformes)."""
    eq = tot = bad = 0
    for members in groups:
        m = len(members)
        tot += m * (m - 1) // 2
        counts = Counter(dmap[p] for p in members)
        eq += sum(c * (c - 1) // 2 for c in counts.values())
        if len(counts) > 1:
            bad += 1
    return eq, tot, bad


def invariant_report(profiles: Sequence[Profile], dmap: Dict[Profile, Fraction],
                     inv_fn: Callable[[Profile], object], *, name: str,
                     perimeter: str, eta_label: str) -> InvariantReport:
    """Teste un invariant candidat contre la partition Δ d'un périmètre à un η."""
    ivals = {p: inv_fn(p) for p in profiles}

    by_i: Dict[object, List[Profile]] = {}
    for p in profiles:
        by_i.setdefault(ivals[p], []).append(p)
    eq, tot, bad = _pair_soundness(by_i.values(), dmap)

    by_d: Dict[Fraction, set] = {}
    for p in profiles:
        by_d.setdefault(dmap[p], set()).add(ivals[p])
    incomplete = sum(1 for vs in by_d.values() if len(vs) > 1)

    by_cell_i: Dict[Tuple[Tuple[int, int], object], List[Profile]] = {}
    i_per_cell: Dict[Tuple[int, int], set] = {}
    for p in profiles:
        c = cell_of(p)
        by_cell_i.setdefault((c, ivals[p]), []).append(p)
        i_per_cell.setdefault(c, set()).add(ivals[p])
    eq_c, tot_c, bad_c = _pair_soundness(by_cell_i.values(), dmap)
    refines = any(len(vs) >= 2 for vs in i_per_cell.values())

    return InvariantReport(
        name=name, perimeter=perimeter, eta_label=eta_label,
        n_profiles=len(profiles), n_i_values=len(by_i), n_delta_values=len(by_d),
        sound=(bad == 0), soundness_pairs=(eq, tot), n_unsound_groups=bad,
        complete=(incomplete == 0), n_incomplete_groups=incomplete,
        refines_nk=refines,
        sound_within_cell=(bad_c == 0), within_cell_pairs=(eq_c, tot_c),
        n_unsound_cells=bad_c,
    )


# --- PORTE 0a : order-sensibilité au périmètre du tour -------------------------------

@dataclass(frozen=True)
class Gate0aT32:
    """Order-sensibilité sur les 2 profils-témoins gelés + pivots η = 0."""

    star_primary: OrderSensitivityReport
    star_multiset: OrderSensitivityReport
    n24_primary: OrderSensitivityReport
    n24_multiset: OrderSensitivityReport
    pivots_exact: bool
    passes: bool


def _tokens_of(profile: Profile) -> List[OrientedToken]:
    """Profil nu → OrientedToken synthétiques (textes t0..tN−1, patron T27)."""
    return [OrientedToken(text=f"t{i}", orientation=o)
            for i, o in enumerate(profile)]


def gate0a_t32() -> Gate0aT32:
    """Porte 0a : l'instrument bouge sous shuffle LÀ où la carte sera jugée.

    Primaire ``obs_struct_frozen`` gap ≥ δ_min = 1e-4 sur ``STAR_NONCANON_T32``
    ((12,9), non canonique) ET ``GRAMMAR24_NONCANON_T32`` (grammaire N = 24) ;
    multiset gap = 0 VACUOUS requis (sinon la carte serait un artefact de
    comptage) ; pivot η = 0 bit-à-bit sur les deux profils-témoins.
    Pré-condition dérivée a priori (doctrine T23) : aucun témoin sur la ligne
    dégénérée 3k = 2N (vacuité par construction — incident gravé ci-dessus).
    """
    for prof in (STAR_NONCANON_T32, GRAMMAR24_NONCANON_T32):
        n, k = cell_of(prof)
        if is_order_degenerate_cell(n, k):
            raise ValueError(
                f"témoin 0a sur la ligne dégénérée 3k=2N ({n},{k}) : "
                "order-invariant par construction, vacuous a priori (T23)")
    reports = {}
    for label, prof in (("star", STAR_NONCANON_T32),
                        ("n24", GRAMMAR24_NONCANON_T32)):
        toks = _tokens_of(prof)
        reports[label] = (
            assert_order_sensitive(obs_struct_frozen, toks, shuffle_fn=shuffle_tokens),
            assert_order_sensitive(obs_struct_multiset, toks, shuffle_fn=shuffle_tokens),
        )
    ps, ms = reports["star"]
    p24, m24 = reports["n24"]
    pivots = (pivot_eta0_is_exact(list(STAR_NONCANON_T32))
              and pivot_eta0_is_exact(list(GRAMMAR24_NONCANON_T32)))
    return Gate0aT32(
        star_primary=ps, star_multiset=ms, n24_primary=p24, n24_multiset=m24,
        pivots_exact=pivots,
        passes=(ps.is_order_sensitive and p24.is_order_sensitive
                and ms.is_vacuous and m24.is_vacuous and pivots),
    )


# --- PORTE 0b : non-régression sur les cellules du périmètre -------------------------

def perimeter_cells() -> List[Tuple[int, int]]:
    """Union des cellules (N,k) touchées par les deux énumérations (273 cellules)."""
    cells = {cell_of(p) for p in enumerate_grammar()}
    cells |= {cell_of(p) for p in enumerate_total()}
    return sorted(cells)


def gate0b_t32() -> List[Tuple[int, int, str]]:
    """Porte 0b : moteur PROFIL == moteur (N,k) sur les cellules du périmètre × 3 η.

    Égalité de ``Fraction`` EXACTE ; retourne les divergences (attendu : vide ;
    toute entrée ⟹ ARRÊT, bug).
    """
    mismatches: List[Tuple[int, int, str]] = []
    for (n, k) in perimeter_cells():
        prof = profile_of(n, k)
        for lbl in ETA_LABELS_T32:
            eta = ETAS_T32[lbl]
            if delta_nom_exact_profile(prof, eta) != delta_nom_exact_eta(n, k, eta):
                mismatches.append((n, k, lbl))
    return mismatches


# --- PORTE 1 : témoins T31 retrouvés par l'énumérateur + corpus gelé -----------------

@dataclass(frozen=True)
class Gate1T32:
    """Témoins T31 : valeurs EXACTES retrouvées (énumérateur ET corpus gelé).

    * ``star_ok``    : {+1/2, −1/2} ⊆ classes de la cellule (12,9), η = 1/2.
    * ``mirror_ok``  : {−1/8, +1/8} ⊆ classes de (16,10), η = 4.
    * ``zero_ok``    : {0, 3/7} ⊆ classes de (14,6), η = 1/2 (canon 0 → profil 3/7).
    * ``corpus_ok``  : #10 F0 → +1/2, #21 F2 → −1/2 (η=1/2, (12,9)) ; F2/(16,10)
      → −1/8 et F1/(16,10) → +1/8 (η=4) ; #38 F1b (14,6) → 3/7 (η=1/2).
    * ``offsets``    : les offsets bord-exact non nuls recalculés sur les 44
      profils × 3 η ; ``offsets_ok`` ⟺ == FROZEN_T31_OFFSETS (les 10).
    """

    star_ok: bool
    mirror_ok: bool
    zero_ok: bool
    corpus_ok: bool
    offsets: Dict[Tuple[int, str], Fraction]
    offsets_ok: bool
    passes: bool


def gate1_t32(census: HorsCanonCensus,
              gmaps: Dict[str, Dict[Profile, Fraction]]) -> Gate1T32:
    """Porte 1 : l'énumérateur et le corpus gelé retrouvent les témoins T31."""
    half = Fraction(1, 2)
    c129 = classes_of_cell(gmaps["1/2"], (12, 9))
    star_ok = (half in c129) and (-half in c129)
    c1610 = classes_of_cell(gmaps["4"], (16, 10))
    mirror_ok = (Fraction(1, 8) in c1610) and (Fraction(-1, 8) in c1610)
    c146 = classes_of_cell(gmaps["1/2"], (14, 6))
    zero_ok = (Fraction(0) in c146) and (Fraction(3, 7) in c146)

    by_idx = {r.index: r for r in census.records}
    by_form_cell = {(r.form, r.n, r.k): r for r in census.records if r.measurable}
    d = delta_nom_exact_profile
    corpus_ok = (
        d(list(by_idx[10].orientations), ETAS_T32["1/2"]) == half
        and d(list(by_idx[21].orientations), ETAS_T32["1/2"]) == -half
        and d(list(by_form_cell[("F2", 16, 10)].orientations),
              ETAS_T32["4"]) == Fraction(-1, 8)
        and d(list(by_form_cell[("F1", 16, 10)].orientations),
              ETAS_T32["4"]) == Fraction(1, 8)
        and d(list(by_idx[38].orientations), ETAS_T32["1/2"]) == Fraction(3, 7)
        and delta_nom_exact_eta(14, 6, ETAS_T32["1/2"]) == 0
    )

    offsets: Dict[Tuple[int, str], Fraction] = {}
    for r in census.records:
        if not r.measurable:
            continue
        for lbl in ETA_LABELS_T32:
            bo = boundary_offset_analysis(list(r.orientations), ETAS_T32[lbl])
            if bo.offset != 0:
                offsets[(r.index, lbl)] = bo.offset
    offsets_ok = offsets == FROZEN_T31_OFFSETS

    return Gate1T32(
        star_ok=star_ok, mirror_ok=mirror_ok, zero_ok=zero_ok,
        corpus_ok=corpus_ok, offsets=offsets, offsets_ok=offsets_ok,
        passes=(star_ok and mirror_ok and zero_ok and corpus_ok and offsets_ok),
    )


# --- PORTE 4 : retour au réel (zéro donnée neuve) ------------------------------------

@dataclass(frozen=True)
class Gate4T32:
    """Les Δ réels T31 (44 profils) et T24-T26 (2116 canoniques) dans les classes.

    ``corpus_pred`` / ``real_pred`` : par invariant SOUND retenu, (prédictions
    de Δ correctes via la table I → Δ construite sur la carte grammaire, total).
    Aucune mesure physique neuve : les Δ sont recalculés par le moteur exact.
    """

    n_corpus_measurable: int
    n_corpus_in_grammar: int
    corpus_delta_ok: Tuple[int, int]
    n_real: int
    n_real_mono: int
    n_real_in_grammar: int
    real_delta_ok: Tuple[int, int]
    corpus_pred: Dict[str, Tuple[int, int]]
    real_pred: Dict[str, Tuple[int, int]]
    passes: bool


def gate4_t32(census: HorsCanonCensus,
              gmaps: Dict[str, Dict[Profile, Fraction]],
              sound_invariants: Dict[str, Callable[[Profile], object]]
              ) -> Gate4T32:
    """Porte 4 : le réel tombe dans les classes prédites (recompute, zéro donnée).

    (a) chaque profil mesurable du corpus hors-canon est PRÉSENT dans
    l'énumération grammaire et son Δ == la classe de la carte ; (b) idem pour
    les 2116 profils canoniques des trois runners T24/T25/T26 (mono-flip,
    profil canonique ∈ grammaire) ; (c) pour chaque invariant SOUND retenu, la
    table I → Δ construite sur la carte grammaire PRÉDIT le Δ de chaque profil
    réel (la valeur d'invariant prédite est portée).
    """
    grammar_set = set(gmaps[ETA_PRIMARY_T32])
    lookup: Dict[str, Dict[str, Dict[object, Fraction]]] = {}
    for name, fn in sound_invariants.items():
        lookup[name] = {}
        for lbl in ETA_LABELS_T32:
            tbl: Dict[object, Fraction] = {}
            for p, dv in gmaps[lbl].items():
                tbl[fn(p)] = dv          # bien défini ssi l'invariant est sound
            lookup[name][lbl] = tbl

    corpus_profiles = [tuple(r.orientations) for r in census.records if r.measurable]
    n_in = sum(1 for p in corpus_profiles if p in grammar_set)
    ok = tot = 0
    pred_counts: Dict[str, List[int]] = {name: [0, 0] for name in sound_invariants}
    for p in corpus_profiles:
        for lbl in ETA_LABELS_T32:
            tot += 1
            dv = delta_nom_exact_profile(list(p), ETAS_T32[lbl])
            if p in gmaps[lbl] and gmaps[lbl][p] == dv:
                ok += 1
            for name, fn in sound_invariants.items():
                pred_counts[name][1] += 1
                if lookup[name][lbl].get(fn(p)) == dv:
                    pred_counts[name][0] += 1
    corpus_delta_ok = (ok, tot)
    corpus_pred = {n: (c[0], c[1]) for n, c in pred_counts.items()}

    real_profiles: List[Profile] = []
    for (_label, path, n_cycles, line_range) in corpus_runs():
        for prof in collect_profiles(path, n_cycles=n_cycles, line_range=line_range):
            real_profiles.append(tuple(tk.orientation for tk in prof))
    n_mono = sum(1 for p in real_profiles if is_mono_flip(list(p)))
    n_real_in = sum(1 for p in real_profiles if p in grammar_set)
    ok_r = tot_r = 0
    pred_r: Dict[str, List[int]] = {name: [0, 0] for name in sound_invariants}
    for p in real_profiles:
        for lbl in ETA_LABELS_T32:
            tot_r += 1
            dv = delta_nom_exact_profile(list(p), ETAS_T32[lbl])
            if p in gmaps[lbl] and gmaps[lbl][p] == dv:
                ok_r += 1
            for name, fn in sound_invariants.items():
                pred_r[name][1] += 1
                if lookup[name][lbl].get(fn(p)) == dv:
                    pred_r[name][0] += 1
    real_delta_ok = (ok_r, tot_r)
    real_pred = {n: (c[0], c[1]) for n, c in pred_r.items()}

    passes = (
        n_in == len(corpus_profiles)
        and corpus_delta_ok[0] == corpus_delta_ok[1]
        and n_mono == len(real_profiles)
        and n_real_in == len(real_profiles)
        and real_delta_ok[0] == real_delta_ok[1]
        and all(c[0] == c[1] for c in corpus_pred.values())
        and all(c[0] == c[1] for c in real_pred.values())
    )
    return Gate4T32(
        n_corpus_measurable=len(corpus_profiles), n_corpus_in_grammar=n_in,
        corpus_delta_ok=corpus_delta_ok,
        n_real=len(real_profiles), n_real_mono=n_mono, n_real_in_grammar=n_real_in,
        real_delta_ok=real_delta_ok,
        corpus_pred=corpus_pred, real_pred=real_pred, passes=passes,
    )


# --- runner de mesure (ordre STRICT 0a → 0b → 1 → 2 → 3 → 4) -------------------------

def _fmt_frac(x: Fraction) -> str:
    return str(x)


def _print_granularity(dmap: Dict[Profile, Fraction], label: str) -> None:
    gran = granularity_by_cell(dmap)
    by_n: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {}
    for (n, k), v in gran.items():
        by_n.setdefault(n, []).append((k, v))
    total_cells = len(gran)
    multi = sum(1 for v in gran.values() if v[0] >= 2)
    print(f"--- granularité {label} : {total_cells} cellules, "
          f"{multi} à ≥ 2 classes ---")
    for n in sorted(by_n):
        row = " ".join(f"k{k}:{c}/{m}" for k, (c, m) in sorted(by_n[n]))
        print(f"  N={n:2d} : {row}")


if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    from .profile_exact import census_horscanon, corpus_horscanon_path

    print("=== ÉNUMÉRATION (bornes gelées) ===")
    grammar = enumerate_grammar()
    total = enumerate_total()
    g_by_n = Counter(len(p) for p in grammar)
    print(f"grammaire N∈[6,24] : {len(grammar)} profils dédupliqués "
          f"(attendu Σ N(N−1) = {sum(n * (n - 1) for n in range(6, 25))})")
    print(f"  par N : {dict(sorted(g_by_n.items()))}")
    print(f"total N∈[4,12] : {len(total)} profils "
          f"(attendu Σ(2^N−2) = {sum(2 ** n - 2 for n in range(4, 13))})")
    star = [p for p in total if cell_of(p) == STAR_CELL]
    print(f"cellule star {STAR_CELL} : {len(star)} arrangements (attendu 220)")

    print("\n=== PORTE 0a : ORDER-SENSIBILITÉ AU PÉRIMÈTRE (δ_min = 1e-4) ===")
    g0a = gate0a_t32()
    print(f"star {STAR_CELL} non-canon : PRIMAIRE gap={g0a.star_primary.gap:.6e} "
          f"sensible={g0a.star_primary.is_order_sensitive} | "
          f"MULTISET gap={g0a.star_multiset.gap:.6e} vacuous={g0a.star_multiset.is_vacuous}")
    print(f"grammaire N=24 non-canon : PRIMAIRE gap={g0a.n24_primary.gap:.6e} "
          f"sensible={g0a.n24_primary.is_order_sensitive} | "
          f"MULTISET gap={g0a.n24_multiset.gap:.6e} vacuous={g0a.n24_multiset.is_vacuous}")
    print(f"pivot η=0 bit-à-bit (2 profils-témoins) : {g0a.pivots_exact}")
    print(f"porte 0a : {'PASSE' if g0a.passes else 'ECHEC — ARRÊT'}")
    assert g0a.passes, "PORTE 0a EN ÉCHEC — ARRÊT"

    print("\n=== PORTE 0b : NON-RÉGRESSION (cellules du périmètre × 3 η) ===")
    cells = perimeter_cells()
    mm = gate0b_t32()
    print(f"cellules : {len(cells)} | divergences : {len(mm)} {mm if mm else ''}")
    print(f"porte 0b : {'PASSE' if not mm else 'ECHEC — ARRÊT (bug moteur)'}")
    assert not mm, "PORTE 0b EN ÉCHEC — ARRÊT"

    print("\n=== CARTES Δ (Fraction exacte, 3 η × 2 périmètres) ===")
    gmaps = {lbl: delta_map(grammar, ETAS_T32[lbl]) for lbl in ETA_LABELS_T32}
    tmaps = {lbl: delta_map(total, ETAS_T32[lbl]) for lbl in ETA_LABELS_T32}

    print("\n=== PORTE 1 : TÉMOINS T31 (corpus gelé + énumérateur) ===")
    cpath = corpus_horscanon_path()
    print(f"corpus gelé : {cpath}")
    cen = census_horscanon(str(cpath))
    g1 = gate1_t32(cen, gmaps)
    print(f"(12,9) η=1/2 {{+1/2, −1/2}} : {g1.star_ok} | classes de la cellule : "
          f"{sorted((str(d), c) for d, c in classes_of_cell(gmaps['1/2'], (12, 9)).items())}")
    print(f"(16,10) η=4 {{−1/8, +1/8}} : {g1.mirror_ok}")
    print(f"(14,6) η=1/2 {{0, 3/7}}    : {g1.zero_ok}")
    print(f"valeurs corpus (#10, #21, F2/F1 (16,10), #38) : {g1.corpus_ok}")
    print(f"offsets bord-exact non nuls recalculés : "
          f"{sorted(((i, l), str(v)) for (i, l), v in g1.offsets.items())}")
    print(f"== les 10 gravés T31 : {g1.offsets_ok}")
    print(f"porte 1 : {'PASSE' if g1.passes else 'ECHEC — ARRÊT'}")
    assert g1.passes, "PORTE 1 EN ÉCHEC — ARRÊT"

    print("\n=== PORTE 2 : LA CARTE BRUTE (granularité par cellule, AVANT interprétation) ===")
    for lbl in ETA_LABELS_T32:
        print(f"\n----- η = {lbl} -----")
        _print_granularity(gmaps[lbl], f"GRAMMAIRE (η={lbl})")
        _print_granularity(tmaps[lbl], f"TOTAL (η={lbl})")
        cstar = classes_of_cell(tmaps[lbl], STAR_CELL)
        print(f"  star {STAR_CELL} (total) : {len(cstar)} classes : "
              f"{sorted((str(d), c) for d, c in cstar.items())}")

    print("\n=== PORTE 3 : INVARIANTS (dérivation PUIS validation, par η) ===")
    deriv = [p for p in grammar if len(p) <= DERIV_N_MAX]
    valid_g = [p for p in grammar if len(p) >= VALID_N_MIN]
    print(f"dérivation grammaire N∈[6,18] : {len(deriv)} profils ; "
          f"validation grammaire N∈[19,24] : {len(valid_g)} ; total : {len(total)}")
    reports: Dict[Tuple[str, str, str], InvariantReport] = {}

    def _run_candidate(name: str, fn) -> None:
        for lbl in ETA_LABELS_T32:
            for peri, profs, dm in (("deriv", deriv, gmaps[lbl]),
                                    ("valid_g", valid_g, gmaps[lbl]),
                                    ("total", total, tmaps[lbl]),
                                    ("full_g", grammar, gmaps[lbl])):
                # "full_g" = garde technique pour la table I → Δ de la porte 4
                # (dérivation ∪ validation ne somme pas paire à paire) — INFO.
                reports[(name, peri, lbl)] = invariant_report(
                    profs, dm, fn, name=name, perimeter=peri, eta_label=lbl)

    for name, fn in INVARIANTS_T32:
        _run_candidate(name, fn)
    # I1b : évalué SEULEMENT si I1 échoue en dérivation (liste close, émission §4)
    i1_fails = any(not reports[("I1_peak_fixed", "deriv", lbl)].sound
                   for lbl in ETA_LABELS_T32)
    candidates = list(INVARIANTS_T32)
    if i1_fails:
        print("I1 ÉCHOUE en dérivation à ≥ 1 η → la variante I1b est activée "
              "(clause de l'émission §4).")
        _run_candidate(*INVARIANT_I1B)
        candidates.append(INVARIANT_I1B)
    else:
        print("I1 sound en dérivation aux 3 η → I1b NON évalué (liste close).")
    for name, _fn in candidates:
        print(f"\n--- {name} ---")
        for peri in ("deriv", "valid_g", "total", "full_g"):
            for lbl in ETA_LABELS_T32:
                r = reports[(name, peri, lbl)]
                eq, tot_p = r.soundness_pairs
                frac = "1" if tot_p == 0 else str(Fraction(eq, tot_p))
                eqc, totc = r.within_cell_pairs
                fracc = "1" if totc == 0 else str(Fraction(eqc, totc))
                print(f"  [{peri:7s}] η={lbl:3s} : sound={r.sound} "
                      f"({eq}/{tot_p} paires, frac={frac}, groupes≠ {r.n_unsound_groups}) "
                      f"| complet={r.complete} (groupes Δ mixtes {r.n_incomplete_groups}) "
                      f"| raffine(N,k)={r.refines_nk} "
                      f"| intra-cellule sound={r.sound_within_cell} "
                      f"({eqc}/{totc}, frac={fracc}, cellules≠ {r.n_unsound_cells}) "
                      f"| #I={r.n_i_values} #Δ={r.n_delta_values} n={r.n_profiles}")

    print("\n=== PORTE 4 : RETOUR AU RÉEL (zéro donnée neuve, recompute moteur) ===")
    sound_all: Dict[str, Callable[[Profile], object]] = {}
    for name, fn in candidates:
        if all(reports[(name, peri, lbl)].sound
               for peri in ("deriv", "valid_g", "total", "full_g")
               for lbl in ETA_LABELS_T32):
            sound_all[name] = fn
    print(f"invariants sound partout (dérivation + validation + grammaire "
          f"entière, 3 η) : {sorted(sound_all) if sound_all else 'AUCUN'}")
    g4 = gate4_t32(cen, gmaps, sound_all)
    print(f"corpus hors-canon : {g4.n_corpus_measurable} mesurables, "
          f"{g4.n_corpus_in_grammar} dans l'énumération grammaire ; "
          f"Δ dans les classes : {g4.corpus_delta_ok[0]}/{g4.corpus_delta_ok[1]}")
    print(f"runners T24/T25/T26 : {g4.n_real} profils, mono-flip {g4.n_real_mono}, "
          f"dans l'énumération {g4.n_real_in_grammar} ; "
          f"Δ dans les classes : {g4.real_delta_ok[0]}/{g4.real_delta_ok[1]}")
    for name in sorted(sound_all):
        print(f"  prédiction via table {name} : corpus {g4.corpus_pred[name]} ; "
              f"réel {g4.real_pred[name]}")
    print(f"porte 4 : {'PASSE' if g4.passes else 'ECHEC'}")

    print("\nISSUE (provisoire, l'ingénieur statue) : portes exécutées dans "
          "l'ordre 0a → 0b → 1 → 2 → 3 → 4 ; voir soundness ci-dessus.")
