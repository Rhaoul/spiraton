from __future__ import annotations

"""Moteur exact par PROFIL — le réel hors-canon échappe-t-il à (N, k) ? (Tour 31).

H31 (émission linguiste, gelée AVANT toute mesure — ``TOUR31_EMISSION.md``) sur le
corpus GELÉ ``corpus_horscanon_aba.txt`` (46 cycles écrits pour que la FORME non
canonique soit le sens ; 44 mesurables, 2 F5 exclues par le filtre du runner) :

  * **H31a (P-profil, reproduction d'instrument)** : un moteur exact GÉNÉRALISÉ
    :func:`delta_nom_exact_profile` — récurrence ``Fraction`` IDENTIQUE à
    ``trace_e_exact`` (T27) mais ``p_ref`` incrémenté PAR LE SIGNE réel ``o_t``
    (non par ``t < k``) — reproduit l'instrument float GELÉ
    (``structural_gap.reconstruct_profile``/``reconstruct_fixed``) sur chaque
    profil du corpus, signe ET valeur (≤ 1e-12), SAUF profils à bord-exact
    pré-déclarés (offset prédit exactement, doctrine T29/T30 GÉNÉRALISÉE, voir
    :func:`boundary_offset_analysis`).
  * **H31b (échappement — LA question du tour)** : soit
    ``Δ_canon(o, η) := delta_nom_exact_eta(N, k_total(o), η)`` avec
    ``k_total = n_plus``. Prédicat gelé : ∃ profil non-canonique ``o`` tel que
    ``delta_nom_exact_profile(o, η) ≠ Δ_canon(o, η)`` (à un η gelé). VRAI ⟹ le
    réel porte une variable HORS (N, k_total) — candidat (3) T27 réalisé. FAUX ⟹
    (N, k) suffit même hors-canon. Attendu déclaré : F2/F3 échappent ; F4
    (arrangement canonique, SEG_B vide) NE DOIT PAS (sinon ARRÊT, bug).

SPEC EXACTE DE L'INSTRUMENT GELÉ (vérifiée dans ``structural_gap.py``, jamais
idéalisée) : ``phi_ref_increments`` calcule ``inc(+1) = φ*·N/n_plus`` et
``inc(−1) = (1−φ*)·N/n_minus`` à partir des COMPTES GLOBAUX du profil réel
(``n_plus`` = # de +1), et chaque token avance ``p_ref`` de l'incrément DE SON
SIGNE — pour le profil canonique ``[+1]*k + [−1]*(N−k)`` cela coïncide avec le
``t < k`` de ``trace_e_exact`` (non-régression, porte B). Profil SANS flip :
incrément uniforme 1 (cas dégénéré de la spec — jamais atteint ici, les F5 sont
exclues par le filtre). Le lecteur un-flip (``o_hat``/``gap_binary``) mésestime
les multi-flip par construction : c'est L'OBJET MESURÉ, pas un défaut —
``gap_binary`` n'entre pas dans Δ, seule la trajectoire ``e_t`` compte.

DOCTRINE BORD-EXACT GÉNÉRALISÉE (T29/T30). Les seules divergences float ↔ exact
jamais observées (T27-T30) vivent aux points ``|e_t − 1| = 1/2`` EXACTS. Ici les
profils sont neufs (aucune gravure antérieure) : la table d'offsets est DÉRIVÉE
a priori par :func:`boundary_offset_analysis` — les points de bord sont localisés
par la trace ``Fraction`` (pur exact), puis le comportement du comparateur float
de l'instrument GELÉ y est émulé point par point (calcul déterministe, déclaré) ;
le contenu falsifiable de H31a est que la divergence GLOBALE ``Δ_float − Δ_exact``
est ENTIÈREMENT expliquée par ces points (``nonboundary_agree`` ET égalité à
l'offset prédit, ≤ 1e-12). Tout écart hors bord-exact ⟹ ARRÊT/dissection.

η GELÉS : {1/2 (ancre historique), 1 (Niven, périodicité-6), 4 (stress
clip-dominé)} — dyadiques exacts au float, sous-ensemble des ``ETAS_FROZEN`` T28
(lecture seule). Parcimonie déclarée : l'échappement est une propriété
d'ARRANGEMENT, largement η-robuste.

PORTES (ordre lexicographique, premier échec = verdict — émission §8) :

  A.  Recensement a priori GELÉ AVANT tout float (:func:`census_horscanon`) :
      46 lignes → ``aba.py`` → profils, comptes (N, k_total), arrangements,
      classification F0-F5, F5 confirmées exclues, cohérence avec
      ``collect_profiles`` (anti-circularité : profils depuis ``aba.py``
      EXCLUSIVEMENT — jamais les dims 0-5 du 33D).
  A′. Prédictions GELÉES AVANT float (:func:`profile_predictions`, Fraction
      pur) : Δ_profile, Δ_canon, verdict d'échappement, points de bord-exact ;
      puis table d'offsets pré-déclarée (:func:`boundary_offset_analysis`).
  0a. Pré-validation GÉNÉRALISÉE (T22/T23 APPLICABLE : le tour porte sur
      l'ordre/l'arrangement) : ``assert_order_sensitive`` sur ≥ 1 F2 + ≥ 1 F3
      frais (primaire ≥ δ_min = 1e-4, multiset VACUOUS — sinon l'échappement
      serait un artefact de comptage, PORTE-SUSPECTE) ; pivot η = 0 :
      ``reconstruct_profile(o, 0) == reconstruct_fixed(o, g0)`` bit-à-bit sur
      TOUS les profils non-canoniques (:func:`gate0a_t31`).
  B.  Non-régression (:func:`gateB_nonregression`) :
      ``delta_nom_exact_profile(profile_of(N,k), η) == delta_nom_exact_eta(N,k,η)``
      sur les 228 cellules × 3 η. Écart ⟹ ARRÊT (bug).
  C.  Reproduction d'instrument (H31a, :func:`gateC_instrument`) : 44 profils
      × 3 η, hors pré-déclarés valeur ≤ 1e-12, pré-déclarés offset == prédit.
  D.  Escape (H31b, :func:`gateD_escape`) : compter les non-canoniques où
      Δ_profile ≠ Δ_canon, par forme et par η ; paires témoins mêmes
      (N, k_total) à Δ différents.

INTOUCHABLES : ``structural_gap.py`` GELÉ byte-à-byte (le lecteur un-flip sur
multi-flip EST l'objet mesuré) ; ``edge_controller.py`` ; ``aba.py`` (toutes les
formes retenues parsent tel quel) ; ``horizon_law``/``spectral_map``/
``phase_diagram``/``eta_runners`` en LECTURE SEULE ; le corpus gelé n'est jamais
modifié. Zéro degré de liberté après gel ; tout est déterministe (seuls les
shuffles seedés de la porte 0a, infra T23).
"""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..data.aba import AbaSegment
from ..experimental.structural_gap import (
    G0_STRUCT,
    OrientedToken,
    f_edge_struct,
    obs_struct_frozen,
    obs_struct_multiset,
    orientation_profile,
    reconstruct_fixed,
    reconstruct_profile,
    shuffle_tokens,
)
from .horizon_law import (
    BAND_EXACT,
    G_MAX_EXACT,
    G_MIN_EXACT,
    PHI_STAR_EXACT,
    TARGET_EXACT,
    f_edge_exact,
    grid_cells,
    profile_of,
)
from .instrument_validation import OrderSensitivityReport, assert_order_sensitive
from .spectral_map import ETAS_FROZEN, delta_nom_exact_eta
from .structural_regulation import (
    MIN_TOKENS,
    _iter_cycles,
    collect_profiles,
    pivot_eta0_is_exact,
)
from .eta_runners import VALUE_TOL, in_grid, is_mono_flip, transition_counts


# --- constantes GELÉES A PRIORI (émission T31 — jamais ajustées après mesure) -------

CORPUS_HORSCANON = "corpus_horscanon_aba.txt"     # racine du répertoire de travail
ETA_LABELS_T31: Tuple[str, ...] = ("1/2", "1", "4")   # émission §4 (dyadiques exacts)
ETAS_T31: Dict[str, Fraction] = {lbl: ETAS_FROZEN[lbl] for lbl in ETA_LABELS_T31}
# recensement ANNONCÉ par l'émission (§3) — vérifié à la porte A, jamais forcé
EXPECTED_FORM_COUNTS: Dict[str, int] = {
    "F0": 11, "F2": 14, "F3": 8, "F1": 5, "F1b": 2, "F4": 4, "F5": 2,
}
N_MEASURABLE_EXPECTED = 44                        # 46 − 2 F5 (exclues par le filtre)


def corpus_horscanon_path() -> Path:
    """Chemin du corpus GELÉ — même résolution que les runners T24/T25/T26.

    Le corpus vit à la racine du RÉPERTOIRE DE TRAVAIL (au-dessus du dépôt git
    spiraton), comme ``dataset_aba.txt`` et ``corpus_claude_aba.txt`` : chemin
    absolu d'abord, repli sur ``parents[3]`` de ce fichier (modèle
    ``corpus_runs``/``_default_dataset`` de ``eta_runners``/``structural_regulation``).
    """
    p = Path("F:/code/claude/spiraton-enhanced") / CORPUS_HORSCANON
    if not p.is_file():
        p = Path(__file__).resolve().parents[3] / CORPUS_HORSCANON
    return p


# --- moteur exact GÉNÉRALISÉ par profil (H31a — spec exacte de l'instrument) --------

def trace_e_exact_profile(orientations: Sequence[int], eta: Fraction) -> List[Fraction]:
    """Trace EXACTE de l'écart de phase e_t pour un profil ARBITRAIRE (Fractions).

    Récurrence IDENTIQUE à ``trace_e_exact`` (T27, gelée) mais ``p_ref`` avance
    de l'incrément DU SIGNE de chaque token — la SPEC EXACTE de
    ``phi_ref_increments`` (instrument gelé) : ``inc(+1) = φ*·N/n_plus``,
    ``inc(−1) = (1−φ*)·N/n_minus`` avec ``n_plus`` = # de +1 DU PROFIL RÉEL ;
    profil sans flip ⇒ incrément uniforme 1 (cas dégénéré de la spec). Sur le
    profil canonique ``[+1]*k + [−1]*(N−k)`` (``n_plus = k``), signe et ``t < k``
    coïncident : ``trace_e_exact_profile == trace_e_exact`` (porte B). À η = 0,
    g reste 1 : lecteur fixe nominal. Aucun float.
    """
    n = len(orientations)
    if n < 1:
        raise ValueError("profil vide : au moins 1 token requis")
    n_plus = sum(1 for o in orientations if o == +1)
    n_minus = n - n_plus
    if n_plus == 0 or n_minus == 0:
        inc_plus = inc_minus = Fraction(1)        # spec : horloge parfaite sans flip
    else:
        inc_plus = PHI_STAR_EXACT * n / n_plus
        inc_minus = (1 - PHI_STAR_EXACT) * n / n_minus
    p_read, p_ref, g = Fraction(1), Fraction(0), Fraction(1)
    e: List[Fraction] = [p_read - p_ref]
    for o in orientations:
        e_t = p_read - p_ref
        g = g - eta * (e_t - TARGET_EXACT)
        g = min(G_MAX_EXACT, max(G_MIN_EXACT, g))
        p_read += g
        p_ref += inc_plus if o == +1 else inc_minus
        e.append(p_read - p_ref)
    return e


def delta_nom_exact_profile(orientations: Sequence[int], eta: Fraction) -> Fraction:
    """Δ_nom EXACT d'un profil arbitraire : f_edge(organe η) − f_edge(fixe g=1)."""
    return f_edge_exact(trace_e_exact_profile(orientations, eta)) - f_edge_exact(
        trace_e_exact_profile(orientations, Fraction(0)))


def k_total(orientations: Sequence[int]) -> int:
    """``k_total = n_plus`` : le compte de +1 du profil (émission §H31b)."""
    return sum(1 for o in orientations if o == +1)


def delta_canon_exact(orientations: Sequence[int], eta: Fraction) -> Fraction:
    """Δ_canon(o, η) := delta_nom_exact_eta(N, k_total(o), η) — la carte (N, k).

    Le profil canonique de mêmes comptes : ``[+1]*k_total + [−1]*(N−k_total)``.
    Exige un flip (``1 ≤ k_total ≤ N−1``) — garanti par le filtre du runner
    (les deux orientations présentes).
    """
    n = len(orientations)
    k = k_total(orientations)
    if not (1 <= k <= n - 1):
        raise ValueError("profil sans flip : Δ_canon n'est pas défini (F5 exclues)")
    return delta_nom_exact_eta(n, k, eta)


def escapes_canon(orientations: Sequence[int], eta: Fraction) -> bool:
    """H31b : le profil ÉCHAPPE-t-il à (N, k_total) ? (Δ_profile ≠ Δ_canon, exact).

    Pour un arrangement canonique (mono-flip), Δ_profile == Δ_canon PAR
    CONSTRUCTION (même récurrence, mêmes incréments, même ordre) : F0/F4
    n'échappent jamais — si F4 échappait, c'est un ARRÊT (bug de classification).
    """
    return delta_nom_exact_profile(orientations, eta) != delta_canon_exact(
        orientations, eta)


def band_edge_touches_profile(orientations: Sequence[int], eta: Fraction) -> int:
    """Points EXACTEMENT sur le bord de bande |e−1| = 1/2 (organe η + fixe g=1).

    Généralisation directe de ``band_edge_touches_eta`` (T28) au profil
    arbitraire : là où le comptage float de l'instrument peut différer du
    comptage exact (doctrine bord-exact). Pur ``Fraction`` — gelable porte A′.
    """
    c = 0
    for e_run in (trace_e_exact_profile(orientations, eta),
                  trace_e_exact_profile(orientations, Fraction(0))):
        c += sum(1 for x in e_run[1:] if abs(x - TARGET_EXACT) == BAND_EXACT)
    return c


def delta_nom_float_profile(orientations: Sequence[int], eta: float) -> float:
    """Δ_nom mesuré par l'INSTRUMENT FLOAT GELÉ (``structural_gap`` byte-à-byte).

    Chemin de code de l'instrument tel quel : ``reconstruct_profile(o, eta,
    g0=1)`` vs ``reconstruct_fixed(o, g=1)`` — baseline NOMINALE (le nominal a
    gagné tous les sweeps T27-T30). η passe par le paramètre d'appel ; les 3 η
    gelés sont dyadiques exacts (float(Fraction) sans perte).
    """
    fe_o = f_edge_struct(reconstruct_profile(orientations, eta=eta, g0=G0_STRUCT))
    fe_f = f_edge_struct(reconstruct_fixed(orientations, g_fixed=G0_STRUCT))
    return fe_o - fe_f


# --- doctrine bord-exact GÉNÉRALISÉE (offsets pré-déclarés, T29/T30) -----------------

@dataclass(frozen=True)
class BoundaryOffset:
    """Analyse de bord-exact d'un profil à un η (la table d'offsets pré-déclarée).

    * ``n_touches``  : points ``|e_t − 1| = 1/2`` EXACTS (localisés en Fraction).
    * ``offset``     : offset PRÉDIT ``Δ_float − Δ_exact`` = (miscount float aux
      seuls points de bord, signé organe − fixe) / N — émulation point par point
      du comparateur de l'instrument GELÉ (déterministe, déclarée).
    * ``nonboundary_agree`` : TOUS les points hors bord classés identiquement
      par le float et l'exact — le contenu falsifiable de la doctrine (un
      désaccord hors bord ⟹ ARRÊT/dissection, jamais toléré).
    """

    n_touches: int
    offset: Fraction
    nonboundary_agree: bool


def boundary_offset_analysis(orientations: Sequence[int], eta: Fraction) -> BoundaryOffset:
    """Dérive l'offset bord-exact PRÉDIT d'un profil à un η (doctrine T29/T30).

    Les points de bord sont localisés par la trace ``Fraction`` (pur exact) ;
    le comparateur float de l'instrument gelé (``|e_t − 1.0| ≤ 0.5``, spec de
    ``f_edge_struct``) y est émulé sur la trace float de l'instrument lui-même.
    L'offset global prédit = Σ signé des miscounts de bord / N. À T29 ce
    mécanisme expliquait (9,7) : e_8 fixe == 3/2 exact ⟹ float 4/9 vs exact
    5/9 ⟹ offset +1/9 — ici il est appliqué a priori aux profils neufs.
    """
    n = len(orientations)
    eta_f = float(eta)                            # dyadique exact (η gelés)
    runs = (
        (trace_e_exact_profile(orientations, eta),
         reconstruct_profile(orientations, eta=eta_f, g0=G0_STRUCT).e, +1),
        (trace_e_exact_profile(orientations, Fraction(0)),
         reconstruct_fixed(orientations, g_fixed=G0_STRUCT).e, -1),
    )
    touches = 0
    miscount = 0                                  # en hits, signé organe − fixe
    agree = True
    for e_exact, e_float, sgn in runs:
        for x_ex, x_fl in zip(e_exact[1:], e_float[1:]):
            hit_ex = abs(x_ex - TARGET_EXACT) <= BAND_EXACT
            hit_fl = abs(x_fl - 1.0) <= 0.5       # le comparateur de f_edge_struct
            if abs(x_ex - TARGET_EXACT) == BAND_EXACT:
                touches += 1
                miscount += sgn * (int(hit_fl) - int(hit_ex))
            elif hit_ex != hit_fl:
                agree = False
    return BoundaryOffset(n_touches=touches, offset=Fraction(miscount, n),
                          nonboundary_agree=agree)


# --- PORTE A : recensement a priori (46 lignes → aba.py → formes F0-F5) --------------

def segment_sign(seg: AbaSegment) -> int:
    """Signe d'orientation d'un SEGMENT : DX/OUT = +1, LV/IN = −1 (grammaire).

    Même application que ``_orientation_of`` de l'instrument gelé (dupliquée en
    lecture, jamais éditée là-bas) ; toute autre combinaison est HORS PÉRIMÈTRE.
    """
    key = (seg.chirality, seg.direction)
    if key == ("DX", "OUT"):
        return +1
    if key == ("LV", "IN"):
        return -1
    raise ValueError(f"orientation non canonique {key} : hors périmètre T31")


def classify_form(seg_signs: Tuple[int, int, int], seg_lens: Tuple[int, int, int]) -> str:
    """Classification F0-F5 GELÉE (taxonomie de l'émission §1, dérivée du code).

    F4 d'abord (SEG_B VIDE, arrangement canonique (+, [], −)) ; F5 = une seule
    orientation (exclue par le filtre) ; sinon le triplet de signes tranche.
    """
    if seg_lens[1] == 0 and seg_signs[0] == +1 and seg_signs[2] == -1:
        return "F4"
    if len(set(seg_signs)) == 1:
        return "F5"
    table = {
        (+1, +1, -1): "F0",
        (+1, -1, +1): "F2",
        (-1, +1, -1): "F3",
        (-1, +1, +1): "F1",
        (-1, -1, +1): "F1b",
    }
    form = table.get(seg_signs)
    if form is None:
        raise ValueError(f"triplet hors taxonomie gelée : {seg_signs}")
    return form


@dataclass(frozen=True)
class ProfileRecord:
    """Une ligne du corpus recensée : forme, comptes, arrangement, profil.

    ``orientations`` vient d'``orientation_profile`` (parseur ``aba.py``
    EXCLUSIVEMENT — anti-circularité) ; ``arrangement`` est la notation par
    segment ``±longueur`` (ex. ``+6|-5|+6``). ``measurable`` = retenu par le
    filtre du runner (N ≥ 4 tokens ET les deux orientations) — les F5 tombent là.
    """

    index: int                        # ordre du fichier parmi les cycles parsés
    form: str
    op: str
    n: int
    k: int                            # k_total = n_plus
    seg_signs: Tuple[int, int, int]
    seg_lens: Tuple[int, int, int]
    arrangement: str
    orientations: Tuple[int, ...]
    tokens: Tuple[OrientedToken, ...]
    measurable: bool


@dataclass(frozen=True)
class HorsCanonCensus:
    """Porte A complète : les 46 lignes classées, F5 exclues, cohérence vérifiée."""

    path: str
    n_cycles: int
    records: Tuple[ProfileRecord, ...]
    form_counts: Dict[str, int]
    counts_match_emission: bool       # == EXPECTED_FORM_COUNTS (annonce §3)
    n_measurable: int
    measurable_match_collect: bool    # profils mesurables == collect_profiles (ordre+contenu)
    profiles_coherent: bool           # profil == expansion des signes de segment


def census_horscanon(path: str) -> HorsCanonCensus:
    """Porte A : recense les 46 lignes — ``aba.py`` exclusivement, AUCUN float.

    Les lignes ``#`` de provenance sont inertes (``AbaParseError`` sautée par
    l'itérateur non-strict, comportement gelé du chargement T24-T30). Cohérences
    vérifiées, jamais forcées : (a) profil == expansion des signes de segment
    par leurs longueurs ; (b) les profils mesurables coïncident EXACTEMENT (ordre
    et contenu) avec ``collect_profiles`` — le recensement décrit la population
    que les portes C/D jugeront.
    """
    records: List[ProfileRecord] = []
    counts: Dict[str, int] = {}
    coherent = True
    for i, cycle in enumerate(_iter_cycles(path, None)):
        segs = (cycle.seg_a, cycle.seg_b, cycle.seg_a_prime)
        signs = tuple(segment_sign(s) for s in segs)
        lens = tuple(len(s.text.split()) for s in segs)
        toks = tuple(orientation_profile(cycle))
        orients = tuple(tk.orientation for tk in toks)
        expansion = tuple(s for s, l in zip(signs, lens) for _ in range(l))
        if orients != expansion:
            coherent = False
        form = classify_form(signs, lens)
        counts[form] = counts.get(form, 0) + 1
        n = len(orients)
        measurable = n >= MIN_TOKENS and set(orients) == {+1, -1}
        arrangement = "|".join(f"{'+' if s > 0 else '-'}{l}" for s, l in zip(signs, lens))
        records.append(ProfileRecord(
            index=i, form=form, op=cycle.op, n=n, k=sum(1 for o in orients if o == +1),
            seg_signs=signs, seg_lens=lens, arrangement=arrangement,
            orientations=orients, tokens=toks, measurable=measurable,
        ))
    measurable_profiles = [list(r.orientations) for r in records if r.measurable]
    collected = collect_profiles(path, n_cycles=len(records) + 1)
    collected_orients = [[tk.orientation for tk in p] for p in collected]
    return HorsCanonCensus(
        path=path,
        n_cycles=len(records),
        records=tuple(records),
        form_counts=counts,
        counts_match_emission=(counts == EXPECTED_FORM_COUNTS),
        n_measurable=len(measurable_profiles),
        measurable_match_collect=(measurable_profiles == collected_orients),
        profiles_coherent=coherent,
    )


# --- PORTE A′ : prédictions gelées AVANT float (Fraction pur) ------------------------

@dataclass(frozen=True)
class ProfilePrediction:
    """Prédiction EXACTE d'un profil mesurable à un η — gravable avant tout float."""

    index: int
    form: str
    n: int
    k: int
    eta_label: str
    delta_profile: Fraction
    delta_canon: Fraction
    escapes: bool
    band_touches: int


def profile_predictions(census: HorsCanonCensus) -> List[ProfilePrediction]:
    """Porte A′ (volet Fraction) : Δ_profile, Δ_canon, échappement, bord-exact.

    ENTIÈREMENT dérivable a priori (aucun float d'instrument) ; ordre
    déterministe (lignes du fichier × η gelés).
    """
    preds: List[ProfilePrediction] = []
    for r in census.records:
        if not r.measurable:
            continue
        o = list(r.orientations)
        for lbl in ETA_LABELS_T31:
            eta = ETAS_T31[lbl]
            dp = delta_nom_exact_profile(o, eta)
            dc = delta_canon_exact(o, eta)
            preds.append(ProfilePrediction(
                index=r.index, form=r.form, n=r.n, k=r.k, eta_label=lbl,
                delta_profile=dp, delta_canon=dc, escapes=(dp != dc),
                band_touches=band_edge_touches_profile(o, eta),
            ))
    return preds


# --- PORTE 0a : pré-validation généralisée (T22/T23 applicable ce tour) --------------

@dataclass(frozen=True)
class Gate0aT31:
    """Order-sensibilité sur F2/F3 frais + pivots η = 0 sur TOUS les non-canoniques."""

    cell_f2: Tuple[int, int]          # (index, N) du 1er F2
    primary_f2: OrderSensitivityReport
    multiset_f2: OrderSensitivityReport
    cell_f3: Tuple[int, int]
    primary_f3: OrderSensitivityReport
    multiset_f3: OrderSensitivityReport
    n_noncanonical: int
    pivot_eta0_all_exact: bool        # reconstruct_profile(o,0) == reconstruct_fixed(o,g0)
    passes: bool


def gate0a_t31(census: HorsCanonCensus) -> Gate0aT31:
    """Porte 0a : l'instrument bouge sous shuffle LÀ où l'échappement sera jugé.

    Primaire ``obs_struct_frozen`` order-sensible (gap ≥ δ_min = 1e-4) sur le
    1er F2 ET le 1er F3 frais ; multiset VACUOUS sur les deux (sinon
    l'échappement serait un artefact de comptage — PORTE-SUSPECTE) ; pivot
    η = 0 bit-à-bit sur TOUS les profils non-canoniques mesurables.
    """
    first: Dict[str, ProfileRecord] = {}
    for r in census.records:
        if r.measurable and r.form in ("F2", "F3") and r.form not in first:
            first[r.form] = r
    if "F2" not in first or "F3" not in first:
        raise ValueError("corpus sans F2/F3 mesurable : porte 0a inapplicable")
    reports = {}
    for form, r in first.items():
        toks = list(r.tokens)
        reports[form] = (
            assert_order_sensitive(obs_struct_frozen, toks, shuffle_fn=shuffle_tokens),
            assert_order_sensitive(obs_struct_multiset, toks, shuffle_fn=shuffle_tokens),
            (r.index, r.n),
        )
    p2, m2, c2 = reports["F2"]
    p3, m3, c3 = reports["F3"]
    non_canon = [r for r in census.records
                 if r.measurable and not is_mono_flip(list(r.orientations))]
    pivots_ok = all(pivot_eta0_is_exact(list(r.orientations)) for r in non_canon)
    return Gate0aT31(
        cell_f2=c2, primary_f2=p2, multiset_f2=m2,
        cell_f3=c3, primary_f3=p3, multiset_f3=m3,
        n_noncanonical=len(non_canon), pivot_eta0_all_exact=pivots_ok,
        passes=(p2.is_order_sensitive and p3.is_order_sensitive
                and m2.is_vacuous and m3.is_vacuous and pivots_ok),
    )


# --- PORTE B : non-régression du moteur généralisé sur la grille T27 -----------------

def gateB_nonregression() -> List[Tuple[int, int, str]]:
    """Porte B : le moteur PROFIL == le moteur (N, k) sur les 228 cellules × 3 η.

    Égalité de ``Fraction`` EXACTE exigée cellule par cellule ; retourne les
    divergences (attendu : liste vide ; toute entrée ⟹ ARRÊT, bug du moteur).
    """
    mismatches: List[Tuple[int, int, str]] = []
    for (n, k) in grid_cells():
        prof = profile_of(n, k)
        for lbl in ETA_LABELS_T31:
            eta = ETAS_T31[lbl]
            if delta_nom_exact_profile(prof, eta) != delta_nom_exact_eta(n, k, eta):
                mismatches.append((n, k, lbl))
    return mismatches


# --- PORTE C : reproduction d'instrument (H31a) --------------------------------------

@dataclass(frozen=True)
class InstrumentCheck:
    """Un profil × un η : float GELÉ vs moteur exact (offset bord-exact inclus)."""

    index: int
    form: str
    n: int
    k: int
    eta_label: str
    delta_exact: Fraction
    delta_float: float
    band_touches: int
    offset_predicted: Fraction        # 0 hors bord-exact
    nonboundary_agree: bool
    ok: bool                          # |Δ_float − float(Δ_exact + offset)| ≤ 1e-12


def gateC_instrument(census: HorsCanonCensus) -> List[InstrumentCheck]:
    """Porte C (H31a) : l'instrument float GELÉ reproduit-il le moteur exact ?

    Hors bord-exact : ``|Δ_float − float(Δ_exact)| ≤ 1e-12`` (l'offset prédit
    est alors 0 par construction). Sur les profils à bord-exact PRÉ-DÉCLARÉS
    (porte A′) : ``Δ_float == float(Δ_exact + offset_prédit)`` ≤ 1e-12 ET
    ``nonboundary_agree`` (la divergence vit AUX SEULS points de bord).
    """
    checks: List[InstrumentCheck] = []
    for r in census.records:
        if not r.measurable:
            continue
        o = list(r.orientations)
        for lbl in ETA_LABELS_T31:
            eta = ETAS_T31[lbl]
            d_exact = delta_nom_exact_profile(o, eta)
            d_float = delta_nom_float_profile(o, float(eta))
            bo = boundary_offset_analysis(o, eta)
            ok = (bo.nonboundary_agree
                  and abs(d_float - float(d_exact + bo.offset)) <= VALUE_TOL)
            checks.append(InstrumentCheck(
                index=r.index, form=r.form, n=r.n, k=r.k, eta_label=lbl,
                delta_exact=d_exact, delta_float=d_float,
                band_touches=bo.n_touches, offset_predicted=bo.offset,
                nonboundary_agree=bo.nonboundary_agree, ok=ok,
            ))
    return checks


# --- PORTE D : escape (H31b) — comptes par forme × η + paires témoins ----------------

@dataclass(frozen=True)
class EscapeSummary:
    """H31b tranchée : comptes d'échappés par forme × η + paires témoins.

    ``by_form_eta`` : (forme, η) → (échappés, total). ``witness_pairs`` : à un
    η, deux profils de MÊMES (N, k_total) mais Δ_profile DIFFÉRENTS — la
    variable hors (N, k) rendue visible sur une paire concrète.
    ``escape_found`` = ∃ un non-canonique échappé (le prédicat gelé) ;
    ``f4_escaped`` DOIT rester vide (sinon ARRÊT).
    """

    by_form_eta: Dict[Tuple[str, str], Tuple[int, int]]
    escape_found: bool
    f4_escaped: List[Tuple[int, str]]             # (index, η) — attendu vide
    witness_pairs: List[Tuple[str, int, int, Tuple[Tuple[int, str, str], ...]]]
    # (η, N, k, ((index, forme, Δ_profile str), ...)) — groupes à Δ non uniforme


def gateD_escape(preds: Sequence[ProfilePrediction]) -> EscapeSummary:
    """Porte D (H31b) : compte les échappés par forme × η, extrait les paires témoins."""
    by: Dict[Tuple[str, str], Tuple[int, int]] = {}
    f4_escaped: List[Tuple[int, str]] = []
    for p in preds:
        esc, tot = by.get((p.form, p.eta_label), (0, 0))
        by[(p.form, p.eta_label)] = (esc + (1 if p.escapes else 0), tot + 1)
        if p.form == "F4" and p.escapes:
            f4_escaped.append((p.index, p.eta_label))
    escape_found = any(p.escapes for p in preds)
    # paires témoins : mêmes (N, k_total) au même η, Δ_profile différents
    groups: Dict[Tuple[str, int, int], List[ProfilePrediction]] = {}
    for p in preds:
        groups.setdefault((p.eta_label, p.n, p.k), []).append(p)
    witness: List[Tuple[str, int, int, Tuple[Tuple[int, str, str], ...]]] = []
    for (lbl, n, k), members in sorted(groups.items()):
        deltas = {p.delta_profile for p in members}
        if len(deltas) > 1:
            witness.append((lbl, n, k, tuple(
                (p.index, p.form, str(p.delta_profile)) for p in members)))
    return EscapeSummary(by_form_eta=by, escape_found=escape_found,
                         f4_escaped=f4_escaped, witness_pairs=witness)


# --- runner de mesure (ordre lexicographique A → A′ → 0a → B → C → D) ----------------

if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    path = corpus_horscanon_path()
    print(f"corpus gelé : {path}")

    print("\n=== PORTE A : RECENSEMENT A PRIORI (aba.py exclusivement, aucun float) ===")
    cen = census_horscanon(str(path))
    print(f"cycles parsés : {cen.n_cycles} | mesurables : {cen.n_measurable} "
          f"(attendu {N_MEASURABLE_EXPECTED})")
    print(f"formes : {dict(sorted(cen.form_counts.items()))} "
          f"| == émission : {cen.counts_match_emission}")
    print(f"profils cohérents (segment→profil) : {cen.profiles_coherent} | "
          f"mesurables == collect_profiles : {cen.measurable_match_collect}")
    print("--- table de recensement (ligne par ligne) ---")
    for r in cen.records:
        grid = "" if not r.measurable else ("" if in_grid(r.n, r.k) else " HORS-GRILLE")
        excl = "" if r.measurable else "  EXCLU (filtre)"
        ud, du = transition_counts(list(r.orientations))
        print(f"  #{r.index:2d} {r.form:3s} {r.op} N={r.n:2d} k_total={r.k:2d} "
              f"arr={r.arrangement:12s} flips(+→−,−→+)=({ud},{du}){grid}{excl}")
    gateA_ok = (cen.counts_match_emission and cen.profiles_coherent
                and cen.measurable_match_collect
                and cen.n_measurable == N_MEASURABLE_EXPECTED)
    print(f"porte A : {'PASSE' if gateA_ok else 'ECHEC — ARRÊT'}")

    print("\n=== PORTE A' : PRÉDICTIONS GELÉES AVANT FLOAT (Fraction pur) ===")
    preds = profile_predictions(cen)
    for lbl in ETA_LABELS_T31:
        print(f"--- η = {lbl} ---")
        for p in preds:
            if p.eta_label != lbl:
                continue
            mark = " ESCAPE" if p.escapes else ""
            edge = f" bord-exact×{p.band_touches}" if p.band_touches else ""
            print(f"  #{p.index:2d} {p.form:3s} (N={p.n:2d},k={p.k:2d}) "
                  f"Δ_profile={str(p.delta_profile):>7s} Δ_canon={str(p.delta_canon):>7s}"
                  f"{mark}{edge}")
    edge_profiles = sorted({(p.index, p.eta_label) for p in preds if p.band_touches})
    print(f"profils à bord-exact PRÉ-DÉCLARÉS : {edge_profiles if edge_profiles else 'AUCUN'}")
    print("--- table d'offsets pré-déclarée (émulation float aux points de bord) ---")
    by_idx = {r.index: r for r in cen.records}
    if not edge_profiles:
        print("  (vide)")
    for idx, lbl in edge_profiles:
        bo = boundary_offset_analysis(list(by_idx[idx].orientations), ETAS_T31[lbl])
        print(f"  #{idx} η={lbl} : touches={bo.n_touches} offset_prédit={bo.offset} "
              f"nonboundary_agree={bo.nonboundary_agree}")

    print("\n=== PORTE 0a : PRÉ-VALIDATION GÉNÉRALISÉE (T22/T23, δ_min=1e-4) ===")
    g0 = gate0a_t31(cen)
    print(f"F2 #{g0.cell_f2[0]} (N={g0.cell_f2[1]}) : PRIMAIRE gap={g0.primary_f2.gap:.6e} "
          f"sensible={g0.primary_f2.is_order_sensitive} | "
          f"MULTISET gap={g0.multiset_f2.gap:.6e} vacuous={g0.multiset_f2.is_vacuous}")
    print(f"F3 #{g0.cell_f3[0]} (N={g0.cell_f3[1]}) : PRIMAIRE gap={g0.primary_f3.gap:.6e} "
          f"sensible={g0.primary_f3.is_order_sensitive} | "
          f"MULTISET gap={g0.multiset_f3.gap:.6e} vacuous={g0.multiset_f3.is_vacuous}")
    print(f"pivot η=0 bit-à-bit sur les {g0.n_noncanonical} non-canoniques : "
          f"{g0.pivot_eta0_all_exact}")
    print(f"porte 0a : {'PASSE' if g0.passes else 'ECHEC — ARRÊT'}")

    print("\n=== PORTE B : NON-RÉGRESSION moteur profil == moteur (N,k), 228 × 3 η ===")
    mm = gateB_nonregression()
    print(f"divergences : {len(mm)} {mm if mm else ''}")
    print(f"porte B : {'PASSE' if not mm else 'ECHEC — ARRÊT (bug moteur)'}")

    print("\n=== PORTE C : REPRODUCTION D'INSTRUMENT (H31a), 44 profils × 3 η ===")
    checks = gateC_instrument(cen)
    n_ok = sum(1 for c in checks if c.ok)
    n_edge = sum(1 for c in checks if c.band_touches > 0)
    print(f"comparaisons : {len(checks)} | ok : {n_ok} | à bord-exact : {n_edge}")
    for c in checks:
        if c.band_touches > 0 or not c.ok:
            status = "ok" if c.ok else "ECHEC"
            print(f"  #{c.index:2d} {c.form:3s} η={c.eta_label} (N={c.n},k={c.k}) "
                  f"Δ_float={c.delta_float:+.6f} Δ_exact={c.delta_exact} "
                  f"offset_prédit={c.offset_predicted} touches={c.band_touches} "
                  f"nonbord_ok={c.nonboundary_agree} [{status}]")
    gateC_ok = n_ok == len(checks)
    print(f"porte C : {'PASSE' if gateC_ok else 'ECHEC — ARRÊT/dissection'}")

    print("\n=== PORTE D : ESCAPE (H31b) — le réel échappe-t-il à (N, k_total) ? ===")
    summ = gateD_escape(preds)
    forms_order = ("F0", "F1", "F1b", "F2", "F3", "F4")
    header = "forme | " + " | ".join(f"η={lbl}" for lbl in ETA_LABELS_T31)
    print(header)
    for form in forms_order:
        cells = []
        for lbl in ETA_LABELS_T31:
            esc, tot = summ.by_form_eta.get((form, lbl), (0, 0))
            cells.append(f"{esc}/{tot}")
        print(f"{form:5s} | " + " | ".join(f"{c:>5s}" for c in cells))
    print(f"échappement trouvé (prédicat H31b) : {summ.escape_found}")
    print(f"F4 échappés (attendu VIDE, sinon ARRÊT) : {summ.f4_escaped}")
    print("--- paires témoins : mêmes (N, k_total), Δ_profile différents ---")
    if not summ.witness_pairs:
        print("  aucune (aucun couple de profils partageant (N, k_total) à Δ distincts)")
    for lbl, n, k, members in summ.witness_pairs:
        detail = ", ".join(f"#{i} {f} Δ={d}" for (i, f, d) in members)
        print(f"  η={lbl} (N={n},k={k}) : {detail}")

    verdict_ok = gateA_ok and g0.passes and not mm and gateC_ok and not summ.f4_escaped
    print(f"\nISSUE (provisoire, l'ingénieur statue) : "
          f"{'portes A/0a/B/C tenues, D tranché — H31b ' + ('VRAIE (le réel échappe à (N,k))' if summ.escape_found else 'FAUSSE (le réel replie sur (N,k))') if verdict_ok else 'ARRÊT — voir la première porte en échec'}")
