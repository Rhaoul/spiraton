from __future__ import annotations

"""Chasse à l'index POST-HOC — split de validation VIERGE (Tour 33).

H33 (émission linguiste, GELÉE — ``TOUR33_EMISSION.md``) : il existe un index
compact structurel ``I(o)``, construit par exploration LIBRE de la carte de
DÉRIVATION T32, gelé PAR ÉCRIT (``TOUR33_DEPLOIEMENT.md`` §β) AVANT tout calcul
de validation, tel que sur le périmètre de VALIDATION VIERGE :

    (a) SOUND 100 % PAR-η : I(o) = I(o′) ⟹ Δ(o; η) = Δ(o′; η)   (Fraction) ;
    (b) COMPRESSIF : #I < #profils (gate DURE — un index bijectif est
        irrecevable, leçon I3/T32) ;
    (c) RAFFINE STRICTEMENT (N, k) (∃ cellule à ≥ 2 valeurs d'I).

RÈGLE DURE D'ADMISSIBILITÉ (émission §8, corpus l.437) : la définition de I ne
prend NI Δ(o; η) NI ``C_organe`` (= f_edge de la trace RÉGULÉE η ≠ 0) en
entrée. AUTORISÉ : combinatoire pure du profil + géométrie du lecteur FIXE
η = 0 (``trace_e_exact_profile(o, 0)``, résidence en bande, excursions, bords).
Un index bâti sur ``C_organe`` est sound-par-construction (MUL récursive,
l'erreur I3 à un cran) : sa soundness ne teste RIEN. La paire
``(C_organe, C_fixe)`` est GELÉE COMME BASELINE tautologique (borne de
compressibilité), jamais comme cartouche.

MOTEUR (LECTURE SEULE, zéro octet modifié) : ``profile_exact.
delta_nom_exact_profile`` / ``trace_e_exact_profile`` (T31) ;
``arrangement_map.*`` (T32 : énumérateurs paramétrables, ``invariant_report``,
portes héritage ``gate0b_t32``/``gate1_t32``, ``is_order_degenerate_cell``) ;
``spectral_map.delta_nom_exact_eta`` (T28) ; instrument ``structural_gap.py``
GELÉ byte-à-byte. Corpus gelé T31 (md5 re-vérifié), ZÉRO corpus neuf.

PÉRIMÈTRE DE DÉRIVATION (exploration libre, phase α — T32 lecture seule) :
grammaire N ∈ [6, 24] (4560), total N ∈ [4, 12] (8158), star (12, 9) (220).

PÉRIMÈTRE DE VALIDATION VIERGE (émission §6 — BORNES EXACTES, jamais regardé
en α ; aucun N ci-dessous énuméré par T32 ; grammaire ≠ total au rapport) :

  * V-GRAMMAIRE : grammaire ≤ 3 blocs, N ∈ [25, 40] — 16 720 profils exacts.
  * V-STAR     : cellule fraîche (15, 9), C(15,9) = 5005 — non dégénérée
    (3·9 = 27 ≠ 30 = 2·15, vérifié par ``is_order_degenerate_cell``).
  * V-TOTAL    : séquences ±1 deux signes, N ∈ [13, 14] — 24 572 profils.
  * V-TOTAL-15 : N = 15 complet (32 766) — OPTIONNEL, budget-gaté : joué
    UNIQUEMENT si les trois garantis tiennent sous le MUR DE BUDGET déclaré
    (``BUDGET_WALL_S`` secondes, gelé ci-dessous) ; sinon SKIP DÉCLARÉ.
  Ordre garanti : V-GRAMMAIRE → V-STAR → V-TOTAL[13,14] → (V-TOTAL-15).
  Explosion de budget ⟹ ARRÊT + rapport partiel (jamais de restriction
  silencieuse).

LES 2 CARTOUCHES GELÉES À β (définitions COMPLÈTES, η-UNIFORMES, calculables en
``Fraction`` depuis le profil seul ; ordre de tir P puis S ; AUCUNE retouche
après le premier calcul de validation — retouche = index MORT, déclaré) :

  P — « T_bande » (:func:`index_trunc_band`) : la trajectoire du lecteur FIXE
      TRONQUÉE À LA BANDE (la direction nommée admissible par l'émission §4) :
          I_P(o) = ( clamp(e_t^fixe ; [1/2, 3/2]) )_{t=1..N},
      avec e^fixe = ``trace_e_exact_profile(o, 0)`` et clamp(x) =
      min(3/2, max(1/2, x)). Ce que P oublie : les VALEURS exactes hors bande
      (seuls survivent le côté et la durée, via le bord répété).
  S — « T_bande_runs » (:func:`index_trunc_band_runs`) : RAFFINEMENT STRICT de
      P — mêmes valeurs en bande EXACTES en ordre ; chaque run maximal
      hors-bande d'un même côté résumé en (côté, longueur) :
          I_S(o) = ( N, éléments ordonnés : e_t exact si |e_t − 1| ≤ 1/2,
                     sinon (side ∈ {−1, +1}, longueur du run) ).
      S distingue une valeur EN BANDE égale au bord (1/2 ou 3/2 exact) d'un
      pas hors-bande clampé au même bord (P les confond). S plus fin que P ⟹
      partout où P est sound, S l'est ; S peut survivre là où P meurt
      (échelle de secours déclarée). ORDRE DE TIR : P puis S.

PORTES (ordre STRICT 0h → 0a → 1 → 2 → β → 3 → 4, premier échec = verdict) :

  0h. HÉRITAGES : ``gate1_t32`` et ``gate0b_t32`` re-passent via import
      lecture seule (md5 corpus == T31/T32) + non-régression FRAÎCHE
      ``delta_nom_exact_profile(profile_of(N,k), η) == delta_nom_exact_eta``
      sur les 530 cellules canoniques de la validation × 3 η (couverture
      100 % : delta_nom_exact_eta est défini pour tout 1 ≤ k ≤ N−1).
  0a. ORDER-SENSIBILITÉ AU PÉRIMÈTRE VIERGE : témoins choisis par CRITÈRE
      A PRIORI (gelé ci-dessous) : premier arrangement à ≥ 2 flips de (15,9)
      dans l'ordre d'énumération de la cellule ; premier profil à ≥ 2 flips
      de ``enumerate_grammar(25, 40)`` dans son ordre, cellule pré-validée
      NON dégénérée (``is_order_degenerate_cell`` — doctrine T32 (iii)).
      Primaire ``obs_struct_frozen`` gap ≥ δ_min = 1e-4 REQUIS ; multiset
      gap = 0 VACUOUS requis ; pivot η = 0 bit-à-bit sur les 2 témoins.
      0a valide L'INSTRUMENT, n'explore PAS les classes du périmètre vierge.
  1.  COMPRESSION SUR DÉRIVATION (gate DURE avant gel) : #I < #profils STRICT
      sur CHACUN des 3 périmètres de dérivation (lecture la plus stricte),
      pour P et S + conformité §8. Échec ⟹ cartouche BARRÉ du gel.
  2.  SOUNDNESS SUR DÉRIVATION (exploratoire — informe le choix, PAS le
      verdict) : fractions exactes de paires, gravées au relais.
  β.  GEL : définitions ci-dessus gravées dans ``TOUR33_DEPLOIEMENT.md``
      AVANT tout calcul de validation (audit d'ordre par l'ingénieur).
  3.  VALIDATION ONE-SHOT (LE verdict H33) : soundness par-η / compression /
      raffinement de P puis S sur CHAQUE sous-périmètre vierge. SURVIE (lecture
      stricte déclarée) = sound 100 % à ≥ 1 η COMMUN aux 3 sous-périmètres
      garantis + compressif STRICT sur chacun + raffinant. Chiffres bruts,
      AUCUNE retouche ; paires-témoins d'unsoundness gravées (Fractions).
  4.  RETOUR AU RÉEL (CONDITIONNEL, zéro donnée neuve) : joué SEULEMENT si
      ≥ 1 cartouche survit la porte 3 ; sinon SKIP DÉCLARÉ.

INTERPRÉTATIONS STRICTES DÉCLARÉES (points laissés ouverts par l'émission) :
  (i)   Porte 1 et H33(b) « #I < #profils » : exigé STRICT sur CHAQUE
        (sous-)périmètre séparément (dérivation : les 3 ; validation : les 3
        garantis + l'optionnel s'il est joué) — jamais sur l'union.
  (ii)  « Sound (≥ 1 η) » du verdict (i) : le MÊME η doit être sound sur TOUS
        les sous-périmètres garantis de la validation.
  (iii) Témoins 0a « non canoniques » : ≥ 2 flips (un profil (−)^a(+)^b à
        1 flip est une rotation de bloc, trop proche du canon) ; si un témoin
        échoue la sensibilité, il est REJETÉ-DÉCLARÉ et remplacé par le
        suivant dans l'ordre a priori (doctrine T23/T32 (iii)).
  (iv)  Baseline tautologique : paire (f_edge(organe η), f_edge(fixe)) —
        Fractions ; calculée en α sur DÉRIVATION seulement.

GARDE-FOUS REFUS : split vierge jamais regardé en α ; moteur lecture seule ;
pas d'oracle (ni Δ ni C_organe dans les index) ; grammaire ≠ total ;
anti-circularité (profils ±1 directs, JAMAIS les dims 0-5 du 33D) ;
DÉTERMINISME total (seuls les shuffles 0a sont seedés).
"""

import time
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .arrangement_map import (
    ETA_LABELS_T32,
    ETAS_T32,
    Profile,
    cell_of,
    delta_map,
    enumerate_grammar,
    enumerate_total,
    gate0b_t32,
    gate1_t32,
    invariant_report,
    InvariantReport,
    is_order_degenerate_cell,
    STAR_CELL,
    _tokens_of,
)
from .horizon_law import BAND_EXACT, TARGET_EXACT, f_edge_exact, profile_of
from .instrument_validation import OrderSensitivityReport, assert_order_sensitive
from .spectral_map import delta_nom_exact_eta
from .structural_regulation import pivot_eta0_is_exact
from .profile_exact import delta_nom_exact_profile, trace_e_exact_profile
from ..experimental.structural_gap import obs_struct_frozen, obs_struct_multiset, shuffle_tokens


# --- constantes GELÉES A PRIORI (émission T33 §6 — jamais ajustées) -------------------

ETA_LABELS_T33: Tuple[str, ...] = ETA_LABELS_T32          # {1/2} primaire + {1, 4}
ETA_PRIMARY_T33 = "1/2"

VGRAM_N_MIN, VGRAM_N_MAX = 25, 40          # V-GRAMMAIRE : 16 720 profils exacts
VSTAR_CELL: Tuple[int, int] = (15, 9)      # V-STAR : C(15,9) = 5005, non dégénérée
VTOTAL_N_MIN, VTOTAL_N_MAX = 13, 14        # V-TOTAL : 24 572 profils
VTOTAL15_N = 15                            # V-TOTAL-15 : 32 766 (OPTIONNEL)
BUDGET_WALL_S = 1800.0                     # mur déclaré : 30 min pour les 3 garantis

BAND_LO = TARGET_EXACT - BAND_EXACT        # 1/2
BAND_HI = TARGET_EXACT + BAND_EXACT        # 3/2


# --- géométrie du lecteur FIXE (η = 0) — la SEULE matière autorisée (§8) --------------

def fixed_trace(profile: Profile) -> Tuple[Fraction, ...]:
    """Trace du lecteur FIXE η = 0, t = 1..N (e_0 = 1 omis ; marche FERMÉE, e_N = 1).

    Pas de la marche : −(2N−3k)/(3k) sur un +1, +(2N−3k)/(3(N−k)) sur un −1 —
    somme totale nulle. Pur ``Fraction``, moteur T31 lecture seule.
    """
    return tuple(trace_e_exact_profile(list(profile), Fraction(0))[1:])


def index_trunc_band(profile: Profile) -> Tuple[Fraction, ...]:
    """CARTOUCHE P « T_bande » — trajectoire fixe TRONQUÉE À LA BANDE (η-uniforme).

    I_P(o) = (min(3/2, max(1/2, e_t^fixe)))_{t=1..N}. Profil seul + lecteur
    fixe : ni Δ, ni C_organe (§8-admissible). Dérivation (gravé §α) :
    #I = 3699/4560 (grammaire), 6215/8158 (total), 218/220 (star).
    """
    return tuple(min(BAND_HI, max(BAND_LO, x)) for x in fixed_trace(profile))


def index_trunc_band_runs(profile: Profile
                          ) -> Tuple[int, Tuple[object, ...]]:
    """CARTOUCHE S « T_bande_runs » — raffinement STRICT de P (η-uniforme).

    Valeurs EN BANDE exactes en ordre ; chaque run maximal hors-bande résumé
    en (côté, longueur). Distingue une valeur en bande égale au bord d'un pas
    clampé (P les confond). Dérivation (gravé §α) : #I = 3762/4560 (grammaire),
    6414/8158 (total), 218/220 (star).
    """
    items: List[object] = []
    cur: Optional[Tuple[int, int]] = None            # (side, longueur)
    for x in fixed_trace(profile):
        side = -1 if x < BAND_LO else (+1 if x > BAND_HI else 0)
        if side == 0:
            if cur is not None:
                items.append(cur)
                cur = None
            items.append(x)
        else:
            if cur is not None and cur[0] == side:
                cur = (side, cur[1] + 1)
            else:
                if cur is not None:
                    items.append(cur)
                cur = (side, 1)
    if cur is not None:
        items.append(cur)
    return (len(profile), tuple(items))


CARTRIDGES_T33: Tuple[Tuple[str, Callable[[Profile], object]], ...] = (
    ("P_trunc_band", index_trunc_band),              # ordre de tir : P puis S
    ("S_trunc_band_runs", index_trunc_band_runs),
)


# --- baseline TAUTOLOGIQUE (émission §4 — borne, JAMAIS cartouche) --------------------

def baseline_pair(profile: Profile, eta: Fraction) -> Tuple[Fraction, Fraction]:
    """(f_edge(organe η), f_edge(fixe)) — sound PAR CONSTRUCTION (Δ = différence).

    TAUTOLOGIQUE (utilise C_organe, interdit §8 pour un cartouche) : ne sert
    QU'À borner la compressibilité d'un index dynamique. Ne compte JAMAIS
    comme H33 satisfaite.
    """
    o = list(profile)
    return (f_edge_exact(trace_e_exact_profile(o, eta)),
            f_edge_exact(trace_e_exact_profile(o, Fraction(0))))


def baseline_count(profiles: Sequence[Profile], eta: Fraction) -> int:
    """#distinct de la paire tautologique sur un périmètre (diagnostic α)."""
    return len({baseline_pair(p, eta) for p in profiles})


# --- énumérateurs de VALIDATION (bornes gelées §6 ; T32 réutilisé paramétré) ----------

def enumerate_cell(n: int, k: int) -> List[Profile]:
    """Tous les C(N,k) arrangements de la cellule (N,k), ordre déterministe.

    Positions des +1 par ``itertools.combinations`` (ordre lexicographique).
    """
    out: List[Profile] = []
    for pos in combinations(range(n), k):
        prof = [-1] * n
        for i in pos:
            prof[i] = +1
        out.append(tuple(prof))
    return out


def enumerate_v_grammar() -> List[Profile]:
    """V-GRAMMAIRE : grammaire ≤ 3 blocs, N ∈ [25, 40] (énumérateur T32 paramétré)."""
    return enumerate_grammar(VGRAM_N_MIN, VGRAM_N_MAX)


def enumerate_v_star() -> List[Profile]:
    """V-STAR : cellule fraîche (15, 9) — C(15,9) = 5005 arrangements."""
    return enumerate_cell(*VSTAR_CELL)


def enumerate_v_total() -> List[Profile]:
    """V-TOTAL : séquences ±1 deux signes, N ∈ [13, 14] (énumérateur T32 paramétré)."""
    return enumerate_total(VTOTAL_N_MIN, VTOTAL_N_MAX)


def enumerate_v_total15() -> List[Profile]:
    """V-TOTAL-15 (OPTIONNEL budget-gaté) : total N = 15 complet, 32 766 profils."""
    return enumerate_total(VTOTAL15_N, VTOTAL15_N)


# --- PORTE 0h : héritages + non-régression fraîche sur cellules de validation --------

@dataclass(frozen=True)
class Gate0hT33:
    """Héritages T31/T32 re-passés + non-régression fraîche (couverture déclarée)."""

    t32_gate1_passes: bool            # témoins T31 exacts (via gate1_t32, lecture seule)
    t32_gate0b_mismatches: int        # 273 cellules T32 × 3 η — attendu 0
    fresh_cells: int                  # cellules canoniques de la validation
    fresh_checks: int                 # cellules × 3 η
    fresh_mismatches: int             # attendu 0
    coverage_full: bool               # delta_nom_exact_eta défini sur TOUTES (1≤k≤N−1)
    passes: bool


def validation_cells() -> List[Tuple[int, int]]:
    """Cellules (N,k) canoniques touchées par la validation : 530 cellules.

    V-GRAMMAIRE : N ∈ [25,40] × k ∈ [1, N−1] (504) ; V-TOTAL : N ∈ [13,14] ×
    k ∈ [1, N−1] (25) ; V-STAR : (15,9) (1).
    """
    cells = [(n, k) for n in range(VGRAM_N_MIN, VGRAM_N_MAX + 1)
             for k in range(1, n)]
    cells += [(n, k) for n in range(VTOTAL_N_MIN, VTOTAL_N_MAX + 1)
              for k in range(1, n)]
    cells.append(VSTAR_CELL)
    return cells


def gate0h_t33(census, gmaps_t32) -> Gate0hT33:
    """Porte 0h : tout nouveau code retrouve T31/T32 + moteur vérifié au terrain neuf.

    ``census``/``gmaps_t32`` : mêmes objets que le runner T32 (corpus gelé +
    cartes grammaire T32) — construits par l'appelant, moteur lecture seule.
    """
    g1 = gate1_t32(census, gmaps_t32)
    mm_t32 = gate0b_t32()
    cells = validation_cells()
    fresh_mm = 0
    for (n, k) in cells:
        prof = profile_of(n, k)
        for lbl in ETA_LABELS_T33:
            eta = ETAS_T32[lbl]
            if delta_nom_exact_profile(prof, eta) != delta_nom_exact_eta(n, k, eta):
                fresh_mm += 1
    return Gate0hT33(
        t32_gate1_passes=g1.passes,
        t32_gate0b_mismatches=len(mm_t32),
        fresh_cells=len(cells), fresh_checks=len(cells) * len(ETA_LABELS_T33),
        fresh_mismatches=fresh_mm, coverage_full=True,
        passes=(g1.passes and not mm_t32 and fresh_mm == 0),
    )


# --- PORTE 0a : order-sensibilité au périmètre VIERGE (témoins a priori) --------------

def _n_flips(profile: Profile) -> int:
    return sum(1 for i in range(1, len(profile)) if profile[i] != profile[i - 1])


def pick_0a_witness_star() -> Profile:
    """Témoin star : PREMIER arrangement à ≥ 2 flips de (15,9) dans l'ordre
    d'``enumerate_cell`` — critère a priori, cellule pré-validée non dégénérée."""
    n, k = VSTAR_CELL
    if is_order_degenerate_cell(n, k):     # pragma: no cover — garde doctrine T32
        raise ValueError(f"cellule star {VSTAR_CELL} sur la ligne 3k=2N : vacuous")
    for prof in enumerate_cell(n, k):
        if _n_flips(prof) >= 2:
            return prof
    raise ValueError("aucun arrangement à ≥ 2 flips")   # pragma: no cover


def pick_0a_witness_grammar() -> Profile:
    """Témoin grammaire : PREMIER profil à ≥ 2 flips de ``enumerate_grammar(25,40)``
    dans son ordre (longueur, lexicographique), cellule NON dégénérée (3k ≠ 2N)."""
    for prof in enumerate_v_grammar():
        n, k = cell_of(prof)
        if _n_flips(prof) >= 2 and not is_order_degenerate_cell(n, k):
            return prof
    raise ValueError("aucun témoin admissible")         # pragma: no cover


@dataclass(frozen=True)
class Gate0aT33:
    """Order-sensibilité des 2 témoins vierges + pivots η = 0 bit-à-bit."""

    star_witness: Profile
    star_primary: OrderSensitivityReport
    star_multiset: OrderSensitivityReport
    grammar_witness: Profile
    grammar_primary: OrderSensitivityReport
    grammar_multiset: OrderSensitivityReport
    pivots_exact: bool
    passes: bool


def gate0a_t33() -> Gate0aT33:
    """Porte 0a : l'instrument bouge sous shuffle LÀ où la validation sera jugée.

    Valide L'INSTRUMENT sur 2 témoins du périmètre vierge (critère a priori) —
    n'explore AUCUNE classe du périmètre vierge (seul le nécessaire de 0a).
    """
    w_star = pick_0a_witness_star()
    w_gram = pick_0a_witness_grammar()
    reps = {}
    for name, prof in (("star", w_star), ("gram", w_gram)):
        toks = _tokens_of(prof)
        reps[name] = (
            assert_order_sensitive(obs_struct_frozen, toks, shuffle_fn=shuffle_tokens),
            assert_order_sensitive(obs_struct_multiset, toks, shuffle_fn=shuffle_tokens),
        )
    ps, ms = reps["star"]
    pg, mg = reps["gram"]
    pivots = (pivot_eta0_is_exact(list(w_star)) and pivot_eta0_is_exact(list(w_gram)))
    return Gate0aT33(
        star_witness=w_star, star_primary=ps, star_multiset=ms,
        grammar_witness=w_gram, grammar_primary=pg, grammar_multiset=mg,
        pivots_exact=pivots,
        passes=(ps.is_order_sensitive and pg.is_order_sensitive
                and ms.is_vacuous and mg.is_vacuous and pivots),
    )


# --- PORTES 1/2 (dérivation) et 3 (validation one-shot) -------------------------------

@dataclass(frozen=True)
class IndexVerdict:
    """Verdict d'un cartouche sur un (sous-)périmètre × η + paire-témoin exacte.

    ``witness`` : si unsound, (profil_a, profil_b, Δ_a, Δ_b) — même I, Δ
    différents (première paire dans l'ordre d'énumération, déterministe).
    """

    report: InvariantReport
    witness: Optional[Tuple[Profile, Profile, Fraction, Fraction]]


def index_verdict(profiles: Sequence[Profile], dmap: Dict[Profile, Fraction],
                  fn: Callable[[Profile], object], *, name: str,
                  perimeter: str, eta_label: str) -> IndexVerdict:
    """``invariant_report`` (T32, lecture seule) + extraction d'une paire-témoin."""
    rep = invariant_report(profiles, dmap, fn, name=name,
                           perimeter=perimeter, eta_label=eta_label)
    witness = None
    if not rep.sound:
        groups: Dict[object, Tuple[Profile, Fraction]] = {}
        for p in profiles:
            iv = fn(p)
            d = dmap[p]
            if iv in groups:
                p0, d0 = groups[iv]
                if d != d0:
                    witness = (p0, p, d0, d)
                    break
            else:
                groups[iv] = (p, d)
    return IndexVerdict(report=rep, witness=witness)


def gate1_t33(perims: Sequence[Tuple[str, Sequence[Profile]]]
              ) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Porte 1 (DURE, avant gel) : #I < #profils STRICT sur chaque périmètre de
    dérivation, pour chaque cartouche. Retourne (cartouche, périmètre) → (#I, n)."""
    out: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for cname, fn in CARTRIDGES_T33:
        for pname, profs in perims:
            n_i = len({fn(p) for p in profs})
            out[(cname, pname)] = (n_i, len(profs))
    return out


# --- PORTE 4 : retour au réel (CONDITIONNEL, zéro donnée neuve) -----------------------

@dataclass(frozen=True)
class Gate4T33:
    """Le réel (44 hors-canon + 2116 canoniques) tombe dans les classes de l'index.

    Jouée SEULEMENT si ≥ 1 cartouche survit la porte 3 ; table I → Δ construite
    sur la carte de VALIDATION du sous-périmètre approprié + dérivation.
    """

    survivor: str
    n_profiles: int
    pred_ok: int
    pred_total: int
    passes: bool


# --- runner de mesure (ordre STRICT 0h → 0a → 1 → 2 → [β déjà gravé] → 3 → 4) --------

def _fmt_report(v: IndexVerdict) -> str:
    r = v.report
    eq, tot = r.soundness_pairs
    frac = "1 (0 paire)" if tot == 0 else f"{Fraction(eq, tot)}"
    s = (f"sound={r.sound} ({eq}/{tot} paires, frac={frac}, "
         f"groupes≠{r.n_unsound_groups}) | #I={r.n_i_values} #Δ={r.n_delta_values} "
         f"n={r.n_profiles} | compressif={r.n_i_values < r.n_profiles} "
         f"| raffine(N,k)={r.refines_nk} | complet={r.complete}")
    if v.witness is not None:
        a, b, da, db = v.witness
        s += (f"\n      témoin : {''.join('+' if o > 0 else '-' for o in a)} "
              f"Δ={da} vs {''.join('+' if o > 0 else '-' for o in b)} Δ={db}")
    return s


if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    from .profile_exact import census_horscanon, corpus_horscanon_path

    t_start = time.time()

    print("=== PORTE 0h : HÉRITAGES T31/T32 + NON-RÉGRESSION FRAÎCHE ===")
    grammar_t32 = enumerate_grammar()
    gmaps_t32 = {lbl: delta_map(grammar_t32, ETAS_T32[lbl]) for lbl in ETA_LABELS_T33}
    cen = census_horscanon(str(corpus_horscanon_path()))
    g0h = gate0h_t33(cen, gmaps_t32)
    print(f"gate1_t32 (témoins T31) : {g0h.t32_gate1_passes} | "
          f"gate0b_t32 divergences : {g0h.t32_gate0b_mismatches}")
    print(f"non-régression fraîche : {g0h.fresh_cells} cellules × 3 η = "
          f"{g0h.fresh_checks} égalités, {g0h.fresh_mismatches} divergences "
          f"(couverture 100 %)")
    print(f"porte 0h : {'PASSE' if g0h.passes else 'ECHEC — ARRÊT'}")
    assert g0h.passes, "PORTE 0h EN ÉCHEC — ARRÊT"

    print("\n=== PORTE 0a : ORDER-SENSIBILITÉ AU PÉRIMÈTRE VIERGE (δ_min = 1e-4) ===")
    g0a = gate0a_t33()
    ws = "".join("+" if o > 0 else "-" for o in g0a.star_witness)
    wg = "".join("+" if o > 0 else "-" for o in g0a.grammar_witness)
    print(f"témoin star (15,9)  [a priori] : {ws}")
    print(f"  PRIMAIRE gap={g0a.star_primary.gap:.6e} "
          f"sensible={g0a.star_primary.is_order_sensitive} | "
          f"MULTISET gap={g0a.star_multiset.gap:.6e} vacuous={g0a.star_multiset.is_vacuous}")
    print(f"témoin grammaire N∈[25,40] [a priori] : {wg} (cellule {cell_of(g0a.grammar_witness)})")
    print(f"  PRIMAIRE gap={g0a.grammar_primary.gap:.6e} "
          f"sensible={g0a.grammar_primary.is_order_sensitive} | "
          f"MULTISET gap={g0a.grammar_multiset.gap:.6e} vacuous={g0a.grammar_multiset.is_vacuous}")
    print(f"pivot η=0 bit-à-bit (2 témoins) : {g0a.pivots_exact}")
    print(f"porte 0a : {'PASSE' if g0a.passes else 'ECHEC — ARRÊT'}")
    assert g0a.passes, "PORTE 0a EN ÉCHEC — ARRÊT"

    print("\n=== DÉRIVATION (T32, lecture seule) + BASELINE TAUTOLOGIQUE ===")
    total_t32 = enumerate_total()
    star_t32 = [p for p in total_t32 if cell_of(p) == STAR_CELL]
    tmaps_t32 = {lbl: delta_map(total_t32, ETAS_T32[lbl]) for lbl in ETA_LABELS_T33}
    deriv_perims = (("grammaire[6,24]", grammar_t32), ("total[4,12]", total_t32),
                    ("star(12,9)", star_t32))
    for pname, profs in deriv_perims:
        for lbl in ETA_LABELS_T33:
            dm = gmaps_t32[lbl] if pname.startswith("gram") else tmaps_t32[lbl]
            nd = len({dm[p] for p in profs})
            nb = baseline_count(profs, ETAS_T32[lbl])
            print(f"  [{pname:15s}] η={lbl:3s} : baseline #paires={nb} "
                  f"#Δ={nd} #profils={len(profs)}  (tautologique, jamais cartouche)")

    print("\n=== PORTE 1 : COMPRESSION SUR DÉRIVATION (gate DURE) ===")
    g1 = gate1_t33(deriv_perims)
    barred = False
    for (cname, pname), (n_i, n) in sorted(g1.items()):
        ok = n_i < n
        barred |= not ok
        print(f"  {cname:18s} [{pname:15s}] : #I={n_i} < n={n} : {ok}")
    print(f"porte 1 : {'PASSE (P et S admissibles au gel)' if not barred else 'CARTOUCHE BARRÉ'}")
    assert not barred, "PORTE 1 EN ÉCHEC — cartouche barré du gel"

    print("\n=== PORTE 2 : SOUNDNESS SUR DÉRIVATION (exploratoire, informe le choix) ===")
    for cname, fn in CARTRIDGES_T33:
        for pname, profs in deriv_perims:
            for lbl in ETA_LABELS_T33:
                dm = gmaps_t32[lbl] if pname.startswith("gram") else tmaps_t32[lbl]
                v = index_verdict(profs, dm, fn, name=cname,
                                  perimeter=pname, eta_label=lbl)
                print(f"  {cname:18s} [{pname:15s}] η={lbl:3s} : {_fmt_report(v)}")

    print("\n[β] GEL : définitions de P et S gravées dans TOUR33_DEPLOIEMENT.md "
          "AVANT la suite (audit d'ordre : l'ingénieur vérifie le relais).")

    print("\n=== PORTE 3 : VALIDATION ONE-SHOT (périmètre VIERGE, ordre garanti) ===")
    t_valid = time.time()
    v_reports: Dict[Tuple[str, str, str], IndexVerdict] = {}

    def _run_perim(pname: str, profs: List[Profile]) -> None:
        for lbl in ETA_LABELS_T33:
            dm = delta_map(profs, ETAS_T32[lbl])
            nd = len(set(dm.values()))
            print(f"  --- {pname} η={lbl} : {len(profs)} profils, #Δ={nd} ---")
            for cname, fn in CARTRIDGES_T33:            # ordre de tir : P puis S
                v = index_verdict(profs, dm, fn, name=cname,
                                  perimeter=pname, eta_label=lbl)
                v_reports[(cname, pname, lbl)] = v
                print(f"    {cname:18s} : {_fmt_report(v)}")

    _run_perim("V-GRAMMAIRE[25,40]", enumerate_v_grammar())
    _run_perim("V-STAR(15,9)", enumerate_v_star())
    _run_perim("V-TOTAL[13,14]", enumerate_v_total())
    elapsed = time.time() - t_valid
    print(f"  (3 garantis : {elapsed:.0f}s ; mur = {BUDGET_WALL_S:.0f}s)")
    if elapsed <= BUDGET_WALL_S:
        _run_perim("V-TOTAL-15", enumerate_v_total15())
    else:                                               # pragma: no cover
        print("  V-TOTAL-15 : SKIP DÉCLARÉ (mur de budget dépassé)")

    guaranteed = ("V-GRAMMAIRE[25,40]", "V-STAR(15,9)", "V-TOTAL[13,14]")
    survivors: List[str] = []
    for cname, _fn in CARTRIDGES_T33:
        etas_ok = []
        for lbl in ETA_LABELS_T33:
            sound_all = all(v_reports[(cname, pn, lbl)].report.sound
                            for pn in guaranteed)
            comp_all = all(v_reports[(cname, pn, lbl)].report.n_i_values
                           < v_reports[(cname, pn, lbl)].report.n_profiles
                           for pn in guaranteed)
            ref_any = any(v_reports[(cname, pn, lbl)].report.refines_nk
                          for pn in guaranteed)
            if sound_all and comp_all and ref_any:
                etas_ok.append(lbl)
        status = f"SURVIT (η sound : {etas_ok})" if etas_ok else "MORT"
        print(f"  verdict {cname} : {status}")
        if etas_ok:
            survivors.append(cname)

    print("\n=== PORTE 4 : RETOUR AU RÉEL (CONDITIONNEL) ===")
    if not survivors:
        print("aucun cartouche survivant — porte 4 : SKIP DÉCLARÉ (rien à prédire)")
    else:                                               # pragma: no cover
        print(f"survivants : {survivors} — table I → Δ à construire (voir relais)")

    print(f"\ntotal : {time.time() - t_start:.0f}s. ISSUE (provisoire, l'ingénieur "
          f"statue) : {'PROGRESSION (i) — index survivant' if survivors else 'voir relais — mort nette des 2 cartouches = branche (ii)'}")
