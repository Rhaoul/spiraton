from __future__ import annotations

"""Régulation STRUCTURELLE : l'organe ``regulate_step`` sur ``gap_struct`` réel (Tour 24).

H24 (émission linguiste, revue Fable 5 intégrée) : la généricité de l'organe
``regulate_step`` — établie T19/T20 sur DEUX instances numériques disjointes (ρ̂
rayon, cos de phase) — s'étend-elle au substrat STRUCTUREL non-commutatif ? Le
substrat, l'observable gradué ``e_t`` et la boucle fermée vivent dans
``experimental/structural_gap.py`` (choix d'implémentation gelés a priori dans son
en-tête). Ce module est le SQUELETTE LEXICOGRAPHIQUE T21 (``aba_regulation.py`` =
modèle, laissé INTACT comme référence géométrique) branché sur ``gap_struct``.

ORDRE LEXICOGRAPHIQUE GRAVÉ A PRIORI (premier échec = verdict ; seuils GELÉS AVANT
exécution — émission §5) :

  (0) PRÉ-VALIDATION D'INSTRUMENT [PORTE, acquis méta T23]. ``assert_order_sensitive``
      (δ_min = 1e-4, n_shuffle = 8) sur le PRIMAIRE (``obs_struct_frozen`` :
      reconstruction FIGÉE en interne, seul l'ordre bouge ; le shuffle permute les
      tokens, le tag voyage avec son token) ET sur la variante-contrôle MULTISET
      (``obs_struct_multiset``, comptes seuls). EXIGÉ : primaire order-sensible ET
      multiset vacuous (la seconde valide la porte elle-même — prédiction
      falsifiable de Fable 5). Primaire vacuous ⇒ ARRÊT : 7e null, localisé à la
      construction de l'observable. Multiset non-vacuous ⇒ PORTE-SUSPECTE.
  (i) PIVOTS. ``η=0`` ≡ lecteur à g fixe : égalité float EXACTE des traces (deux
      chemins de code — ``reconstruct_profile(eta=0)`` vs ``reconstruct_fixed``).
      Pivot dégénéré : profil SANS flip (tout-DX synthétique) ⇒ Δ = 0 (l'offset
      initial = target rend l'organe exactement inerte : rien à réguler).
      ``regulate_step`` byte-identique (importé, jamais copié).
  (ii) EFFET vs best_fixed. ``Δf_edge = f_edge(organe) − f_edge(best_fixed)`` sur
       ≥ 40 cycles réels DÉTERMINISTES (ordre du fichier ``dataset_aba.txt``),
       médiane appariée + Wilcoxon + compte de signes. Seuils hérités T19/T20 :
       > 0.15 ACTIVE / [0.05, 0.15] INERTE-déclarée / < 0.05 MORTE.
       ``f_edge`` = fraction des tokens avec ``e_t`` dans [target−band, target+band]
       (target = 1 token, band = 0.5 — gelés depuis la grammaire, cf. structural_gap).
  (iii) BASELINE SHUFFLE (leçon T14/T22). Refaire (ii) sur profils SHUFFLÉS (même
        best_g, shuffles seedés). ``Δf_edge(réel) − Δf_edge(shuffle) < 0.05`` ⇒
        NULL-ordre-détruit (l'avantage est la dynamique générique de l'organe, pas
        la structure A→B→A′). Sinon ACTIVE-structurelle.

DIAGNOSTIC CENTRAL α-ω : NON APPLICABLE tel quel — le substrat est une phase
SCALAIRE strictement monotone (``cos`` entre scalaires ∈ {−1, +1}, dégénéré ;
``best_return_step`` suppose un état multidim qui peut revenir près de s_0, ce
qu'une phase monotone ne fait jamais par construction). L'ANALOGUE de la clôture
est rapporté à la place : ``e_N`` = écart de phase à la CLÔTURE du cycle
(``p_ref(N) = N`` exactement) — le « retour transformé » vaut ``e_N ≈ target``
(proche-aligné-non-identique : le lecteur referme le cycle avec son avance d'un
token), ``e_N = 0`` serait la copie, ``|e_N| ≫ target`` la dissipation.

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/`` NI ``regulate_step`` NI
``aba_regulation.py`` (référence T21). Tout déterministe : cycles dans l'ordre du
fichier, shuffles seedés, AUCUNE source aléatoire non seedée. AUCUN ``.so`` requis
(le profil d'orientation est structurel — aucun skip de test).

TOUR 25 (H25, émission linguiste — MÊME instrument, AUTRE corpus). Extension
MINIMALE, l'instrument ``structural_gap.py`` est GELÉ BYTE-À-BYTE (condition de
comparabilité T24↔T25) et les seuils/portes ci-dessus sont INCHANGÉS :

  * ``CORPUS_CLAUDE`` : 2e corpus réel (76 cycles, prose de Claude) ; le défaut du
    module reste ``dataset_aba.txt`` (le run T24 doit rester reproductible tel quel).
  * :func:`population_descriptor` : le DESCRIPTEUR DE POPULATION est rapporté AVANT
    toute porte (leçon T24 : le null vivait dans la statistique du corpus, pas dans
    l'organe). σ(k/N) = écart-type ÉCHANTILLON (ddof=1 ; sur les 40 cycles T24 il
    vaut 0.0382 → la référence publiée « σ=0.038 »). Critère GELÉ a priori
    (émission T25 §2a) : ``σ(k/N) ≥ SIGMA_KN_MATERIAL = 2×0.038 = 0.076`` décide
    quelle PRÉDICTION est testable (P-a variance / P-b horizon) — JAMAIS le verdict.
  * :func:`length_strata` : lecture PARTITIONNÉE des MÊMES ``Δf_edge`` déjà calculés
    (court < 10 tokens / long ≥ 12, bornes pré-déclarées §2a) — pas un instrument
    neuf, aucun recalcul.

TOUR 26 (H26, émission linguiste — consolider le vivant T25). Le descripteur
d'EXCURSION, proposé au journal APRÈS la mesure T25, est ici GELÉ A PRIORI et
confronté à un bloc frais de ``dataset_aba.txt`` (lignes 1001-3000, indices gelés
AVANT toute lecture de contenu). L'instrument ``structural_gap.py`` reste GELÉ
byte-à-byte ; seuls s'ajoutent le descripteur, la partition et le chargement :

  * :func:`excursion` : ``|k − φ*·N|`` (tokens du parseur ``aba.py`` ; k = |A|+|B|).
    Ancrage a priori : c'est le PIC EXACT de ``|e_t − target|`` subi par le lecteur
    fixe nominal (g=1) au token de flip — dérivé de l'algèbre de l'instrument,
    jamais des données. ALGÉBRIQUEMENT IDENTIQUE à ``obs_struct_multiset`` : c'est
    une propriété VOULUE, pas un accident — une étiquette de partition doit être
    shuffle-INVARIANTE (chaque cycle reste dans sa strate sous la porte 3) et ne
    peut pas être circulaire avec l'observable order-sensible régulé (porte 0).
    Étiquette de partition, JAMAIS un observable régulé.
  * Seuil de strate = ``BAND_LEAD = 0.5`` (dérivé de la bande d'instrument, PAS du
    0.667 mesuré T25) : excursion > 0.5 ⟺ le flip SORT de la bande pour le lecteur
    fixe ⟺ il existe une erreur de phase réelle à corriger.
  * :func:`excursion_strata` : lecture PARTITIONNÉE des MÊMES ``Δf_edge`` (modèle
    ``length_strata`` — aucun recalcul, aucun best_fixed par strate). Plancher
    d'interprétabilité : ≥ 20 cycles à Δ non-nul par strate (critère de PUISSANCE,
    jamais un bouton de verdict).
  * Chargement par OFFSET de lignes gelé (``line_range``, 1-based inclusif) dans
    :func:`collect_profiles`/:func:`population_descriptor`/
    :func:`run_structural_regulation` — extension du CHARGEMENT seulement, jamais
    de l'instrument ; ``line_range=None`` reproduit T24/T25 à l'identique.

TOUR 30 (émission linguiste — η sur runners, clôture lévo·in). Extension MINIMALE
par PARAMÈTRE : :func:`run_structural_regulation` accepte un kwarg ``eta``
(défaut ``ETA_STRUCT``) propagé aux appels ``reconstruct_profile`` des portes
(ii)/(iii). ``ETA_STRUCT`` reste FIGÉ, ``edge_controller`` INTACT, et le chemin
par défaut est BYTE-IDENTIQUE aux runs T24/T25/T26 (le défaut du kwarg EST la
constante : mêmes suites d'opérations flottantes — invariant de comparabilité,
testé explicitement dans ``tests/test_eta_runners.py``). Le croisement avec la
carte(η) dérivée T28/T29 vit dans ``diagnostics/eta_runners.py``, jamais ici.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from ..data.aba import AbaCycle, AbaParseError, is_terminator_line, iter_aba_cycles, parse_aba_line
from ..experimental.structural_gap import (
    ETA_STRUCT,
    G0_STRUCT,
    G_MAX_STRUCT,
    G_MIN_STRUCT,
    BAND_LEAD,
    OrientedToken,
    PHI_STAR,
    STRUCT_GAIN_SWEEP,
    TARGET_LEAD,
    f_edge_struct,
    flip_fraction,
    obs_struct_frozen,
    obs_struct_multiset,
    orientation_profile,
    reconstruct_fixed,
    reconstruct_profile,
    shuffle_orientations,
    shuffle_tokens,
)
from .edge_maintenance import _median, wilcoxon_signed_rank
from .memory_inhibition_scan import spearman_rho, spearman_t_pvalue
from .instrument_validation import (
    DELTA_MIN_DEFAULT,
    N_SHUFFLE_DEFAULT,
    OrderSensitivityReport,
    assert_order_sensitive,
)

# Seuils de verdict HÉRITÉS T19/T20/T21 (continuité décisionnelle, gelés — émission §5).
# Importés du module T21 (source unique) plutôt que redéclarés.
from .aba_regulation import THRESHOLD_ACTIVE, THRESHOLD_INERTE, THRESHOLD_STRUCTURE

# Cycles utilisables : au moins MIN_TOKENS tokens (même a priori que T21 : ~un token
# par segment plus un) ET les deux orientations présentes (un flip existe — un profil
# sans flip est le pivot dégénéré, pas un cycle mesurable).
MIN_TOKENS = 4
N_CYCLES_DEFAULT = 40
SHUFFLE_SEED_BASE = 70000       # même base que T21 (continuité, déterminisme)
N_PROFILES_GATE0_INFO = 5       # gaps porte 0 rapportés sur les premiers profils (info)

# --- Tour 25 : constantes GELÉES A PRIORI (émission T25 §2a) -----------------------
SIGMA_KN_REF_T24 = 0.038        # σ(k/N) de référence (40 cycles dataset_aba, T24)
SIGMA_KN_MATERIAL = 2 * SIGMA_KN_REF_T24   # = 0.076 : « variance matériellement plus
                                # grande » ⇒ P-a testable. Décide la TESTABILITÉ
                                # d'une prédiction, JAMAIS le verdict des portes.
N_CYCLES_CLAUDE = 76            # les 76 cycles COMPLETS (émission : ne pas sous-échantillonner)
SHORT_MAX_TOKENS = 10           # strate courte : N < 10 tokens (pré-déclarée §2a)
LONG_MIN_TOKENS = 12            # strate longue : N ≥ 12 tokens (pré-déclarée §2a)

# --- Tour 26 : constantes GELÉES A PRIORI (émission T26 §1) -------------------------
EXCURSION_THRESHOLD = BAND_LEAD  # = 0.5 : seuil de strate DÉRIVÉ de la bande
                                 # d'instrument (excursion > band ⟺ le flip sort de
                                 # la bande pour le lecteur fixe) — PAS du 0.667 T25.
MIN_NONZERO_STRATUM = 20         # plancher d'interprétabilité (puissance, pas verdict)
BLOCK26_LINES = (1001, 3000)     # bloc frais de dataset_aba.txt, indices gelés AVANT
                                 # toute lecture de contenu (émission §0)
N_CYCLES_BLOCK26 = 2000          # cap = taille du bloc ; le FILTRE décide l'effectif


def _iter_cycles(path: str, line_range: Optional[Tuple[int, int]]) -> Iterator[AbaCycle]:
    """Itère les cycles, éventuellement restreints à ``line_range`` (1-based inclusif).

    ``line_range=None`` ⇒ délégation PURE à :func:`iter_aba_cycles` (comportement
    T24/T25 byte-identique). Sinon : mêmes règles non-strictes (vides, terminateurs
    et lignes illisibles sautés), sur la tranche de lignes gelée SEULEMENT.
    Extension du CHARGEMENT, jamais de l'instrument.
    """
    if line_range is None:
        yield from iter_aba_cycles(path)
        return
    lo, hi = line_range
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if i < lo:
                continue
            if i > hi:
                break
            stripped = line.strip()
            if not stripped or is_terminator_line(stripped):
                continue
            try:
                yield parse_aba_line(stripped)
            except AbaParseError:
                continue


def collect_profiles(
    path: str,
    *,
    n_cycles: int = N_CYCLES_DEFAULT,
    min_tokens: int = MIN_TOKENS,
    line_range: Optional[Tuple[int, int]] = None,
) -> List[List[OrientedToken]]:
    """Les ``n_cycles`` premiers profils utilisables, dans l'ORDRE du fichier.

    Déterministe : ordre du fichier, aucun tirage. Utilisable = ``N ≥ min_tokens``
    ET les deux orientations présentes. Source du tag : parseur ``aba.py``
    exclusivement (garde anti-circularité (a) de l'émission). ``line_range``
    (T26, 1-based inclusif) restreint le CHARGEMENT à un bloc de lignes gelé.
    """
    profiles: List[List[OrientedToken]] = []
    for cycle in _iter_cycles(path, line_range):
        toks = orientation_profile(cycle)
        orients = {tk.orientation for tk in toks}
        if len(toks) >= min_tokens and orients == {+1, -1}:
            profiles.append(toks)
        if len(profiles) >= n_cycles:
            break
    return profiles


# --- Tour 25 : descripteur de population (rapporté AVANT toute porte) --------------

def _sample_std(xs: Sequence[float]) -> float:
    """Écart-type ÉCHANTILLON (ddof=1) — définition gelée de σ(k/N) (en-tête T25)."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5


@dataclass(frozen=True)
class PopulationDescriptor:
    """Statistique du corpus AVANT jugement de l'organe (leçon T24).

    ``variance_material`` applique le critère GELÉ ``kn_sigma ≥ SIGMA_KN_MATERIAL``
    (décide quelle prédiction P-a/P-b est TESTABLE, jamais le verdict).
    ``n_at_phi_star_exact`` compte les profils à ``k/N = φ*`` EXACT — cas limite
    d'instrument documenté T24 (incréments uniformes ⇒ order-invariance ponctuelle
    ET Δ = 0 exact pour ces cycles : rien à réguler par construction).
    """

    n_cycles: int
    tokens_min: int
    tokens_median: float
    tokens_max: int
    kn_min: float
    kn_median: float
    kn_max: float
    kn_sigma: float
    variance_material: bool          # kn_sigma ≥ SIGMA_KN_MATERIAL (P-a testable ?)
    n_at_phi_star_exact: int
    # T26 : distribution du descripteur d'excursion GELÉ (rapportée AVANT les portes)
    exc_min: float
    exc_median: float
    exc_max: float
    n_exc_high: int                  # cycles à excursion > EXCURSION_THRESHOLD (= band)


def population_descriptor(
    path: str,
    *,
    n_cycles: int = N_CYCLES_DEFAULT,
    min_tokens: int = MIN_TOKENS,
    line_range: Optional[Tuple[int, int]] = None,
) -> PopulationDescriptor:
    """Descripteur de population d'un corpus ABA (mesure, jamais cible).

    Mêmes cycles utilisables que :func:`collect_profiles` (même filtre, même ordre
    de fichier — le descripteur décrit EXACTEMENT la population jugée aux portes).
    """
    profiles = collect_profiles(
        path, n_cycles=n_cycles, min_tokens=min_tokens, line_range=line_range
    )
    if not profiles:
        raise ValueError(f"aucun cycle utilisable dans {path!r}")
    lens = [len(p) for p in profiles]
    orientation_lists = [[tk.orientation for tk in p] for p in profiles]
    kns = [flip_fraction(o) for o in orientation_lists]
    excs = [excursion(o) for o in orientation_lists]
    sigma = _sample_std(kns)
    return PopulationDescriptor(
        n_cycles=len(profiles),
        tokens_min=min(lens),
        tokens_median=_median([float(x) for x in lens]),
        tokens_max=max(lens),
        kn_min=min(kns),
        kn_median=_median(kns),
        kn_max=max(kns),
        kn_sigma=sigma,
        variance_material=sigma >= SIGMA_KN_MATERIAL,
        n_at_phi_star_exact=sum(1 for k in kns if abs(k - PHI_STAR) < 1e-12),
        exc_min=min(excs),
        exc_median=_median(excs),
        exc_max=max(excs),
        n_exc_high=sum(1 for e in excs if e > EXCURSION_THRESHOLD),
    )


# --- Tour 25 : strate longueur (lecture PARTITIONNÉE des mêmes Δf_edge) -------------

@dataclass(frozen=True)
class LengthStrata:
    """Médiane appariée de ``Δf_edge`` par strate de longueur (pré-déclarée §2a).

    Lecture partitionnée des sorties DÉJÀ calculées (aucun recalcul, aucun
    instrument neuf). Une strate vide est rapportée telle quelle : effectif 0 et
    médiane NaN — un fait de population, jamais masqué.
    """

    n_short: int                     # cycles N < SHORT_MAX_TOKENS
    delta_short_median: float        # NaN si strate vide
    n_long: int                      # cycles N ≥ LONG_MIN_TOKENS
    delta_long_median: float         # NaN si strate vide


def length_strata(
    delta_real: Sequence[float],
    tokens_per_cycle: Sequence[int],
    *,
    short_max: int = SHORT_MAX_TOKENS,
    long_min: int = LONG_MIN_TOKENS,
) -> LengthStrata:
    """Partitionne les ``Δf_edge`` appariés d'un rapport par longueur de cycle.

    S'applique aux champs ``delta_real``/``tokens_per_cycle`` d'un
    :class:`StructuralRegulationReport` (séquences alignées par cycle).
    """
    if len(delta_real) != len(tokens_per_cycle):
        raise ValueError("delta_real et tokens_per_cycle doivent être alignés")
    short = [d for d, n in zip(delta_real, tokens_per_cycle) if n < short_max]
    long_ = [d for d, n in zip(delta_real, tokens_per_cycle) if n >= long_min]
    return LengthStrata(
        n_short=len(short),
        delta_short_median=_median(short) if short else float("nan"),
        n_long=len(long_),
        delta_long_median=_median(long_) if long_ else float("nan"),
    )


# --- Tour 26 : descripteur d'excursion GELÉ + strate (lecture partitionnée) --------

def excursion(orientations: Sequence[int]) -> float:
    """Descripteur GELÉ a priori (T26) : ``excursion = |k − φ*·N|`` en tokens.

    ``k`` = nombre de tokens à +1 (= |SEG_A|+|SEG_B| pour un cycle canonique),
    ``N`` = longueur du profil, ``φ* = 2/3`` (constante d'instrument T24).

    Ancrage algébrique (émission T26 §1-i) : c'est le PIC EXACT de
    ``|e_t − target|`` que subit le lecteur fixe nominal (g=1) au token de flip —
    dérivé de l'instrument seul, jamais des données. Identique à
    ``obs_struct_multiset`` par construction : shuffle-INVARIANT (un cycle reste
    dans sa strate sous la porte 3) et non-circulaire avec l'observable
    order-sensible de la porte 0. Étiquette de PARTITION, jamais observable régulé.
    """
    n = len(orientations)
    k = sum(1 for o in orientations if o == +1)
    return abs(k - PHI_STAR * n)


@dataclass(frozen=True)
class ExcursionStrata:
    """Lecture PARTITIONNÉE des MÊMES ``Δf_edge`` par excursion (modèle T25).

    Aucun recalcul, aucun best_fixed par strate. ``*_interpretable`` applique le
    plancher de PUISSANCE gelé (≥ ``MIN_NONZERO_STRATUM`` cycles à Δ non-nul) —
    critère d'interprétabilité, jamais un bouton de verdict. Une strate vide est
    rapportée telle quelle (effectif 0, médiane NaN).
    """

    threshold: float                 # = EXCURSION_THRESHOLD (band, dérivé a priori)
    n_high: int                      # cycles à excursion > threshold
    delta_high_median: float         # NaN si strate vide
    n_high_nonzero: int              # cycles de la strate haute à Δ ≠ 0
    high_interpretable: bool
    n_low: int                       # cycles à excursion ≤ threshold
    delta_low_median: float          # NaN si strate vide
    n_low_nonzero: int
    low_interpretable: bool
    contrast: float                  # delta_high_median − delta_low_median (NaN si vide)


def excursion_strata(
    delta_real: Sequence[float],
    excursions: Sequence[float],
    *,
    threshold: float = EXCURSION_THRESHOLD,
    min_nonzero: int = MIN_NONZERO_STRATUM,
) -> ExcursionStrata:
    """Partitionne les ``Δf_edge`` appariés d'un rapport par excursion de cycle.

    S'applique aux champs ``delta_real`` d'un :class:`StructuralRegulationReport`
    et aux :func:`excursion` des MÊMES cycles (séquences alignées par cycle).
    """
    if len(delta_real) != len(excursions):
        raise ValueError("delta_real et excursions doivent être alignés")
    high = [d for d, e in zip(delta_real, excursions) if e > threshold]
    low = [d for d, e in zip(delta_real, excursions) if e <= threshold]
    med_high = _median(high) if high else float("nan")
    med_low = _median(low) if low else float("nan")
    n_high_nz = sum(1 for d in high if d != 0.0)
    n_low_nz = sum(1 for d in low if d != 0.0)
    return ExcursionStrata(
        threshold=threshold,
        n_high=len(high),
        delta_high_median=med_high,
        n_high_nonzero=n_high_nz,
        high_interpretable=n_high_nz >= min_nonzero,
        n_low=len(low),
        delta_low_median=med_low,
        n_low_nonzero=n_low_nz,
        low_interpretable=n_low_nz >= min_nonzero,
        contrast=med_high - med_low,
    )


# --- pivots (porte 1) -------------------------------------------------------------

def pivot_eta0_is_exact(orientations: Sequence[int], *, g0: float = G0_STRUCT) -> bool:
    """``η=0`` ≡ g-fixe : égalité float EXACTE des traces (deux chemins de code)."""
    tr0 = reconstruct_profile(orientations, eta=0.0, g0=g0)
    trf = reconstruct_fixed(orientations, g_fixed=g0)
    return (
        tr0.e == trf.e
        and tr0.p_read == trf.p_read
        and tr0.p_ref == trf.p_ref
        and tr0.g == trf.g
        and tr0.o_hat == trf.o_hat
        and tr0.gap_binary == trf.gap_binary
    )


def pivot_noflip_delta(n_tokens: int = 12) -> float:
    """Pivot dégénéré : profil SANS flip (tout-DX synthétique) ⇒ Δf_edge attendu = 0.

    Incréments uniformes ⇒ à g nominal ``e ≡ target`` ⇒ l'organe ne corrige jamais
    (exactement inerte) ; le meilleur g fixe du sweep est le nominal (f_edge = 1).
    """
    orientations = [+1] * n_tokens
    fe_organ = f_edge_struct(reconstruct_profile(orientations, eta=ETA_STRUCT, g0=G0_STRUCT))
    fe_best = max(
        f_edge_struct(reconstruct_fixed(orientations, g_fixed=g)) for g in STRUCT_GAIN_SWEEP
    )
    return fe_organ - fe_best


# --- baseline dure : meilleur rythme fixe global (porte 2) -------------------------

def best_fixed_gain(
    orientation_lists: Sequence[Sequence[int]],
    *,
    fixed_gains: Sequence[float] = STRUCT_GAIN_SWEEP,
) -> float:
    """Le g fixe qui MAXIMISE le f_edge MÉDIAN sur les cycles (baseline dure, modèle T21)."""
    best_g = fixed_gains[0]
    best_med = -1.0
    for g in fixed_gains:
        fes = [f_edge_struct(reconstruct_fixed(o, g_fixed=g)) for o in orientation_lists]
        med = _median(fes)
        if med > best_med:
            best_med = med
            best_g = g
    return best_g


# --- rapport complet ---------------------------------------------------------------

@dataclass(frozen=True)
class StructuralRegulationReport:
    """Verdict T24 complet : porte 0, pivots, effet (ii), shuffle (iii), clôture e_N."""

    n_cycles: int
    tokens_per_cycle: List[int]
    flip_fracs: List[float]           # k/N par cycle (la variation structurelle, informatif)
    # PORTE 0 : pré-validation d'instrument (T23) sur le 1er profil réel
    order_primary: OrderSensitivityReport
    order_multiset: OrderSensitivityReport
    gate0_primary_sensitive: bool
    gate0_multiset_vacuous: bool
    order_gaps_info: List[float]      # gaps du primaire sur les 1ers profils (informatif)
    # PORTE 1 : pivots
    pivot_eta0_exact: bool
    pivot_noflip: float               # attendu 0.0
    # PORTE 2 : effet vs best_fixed (réel)
    best_fixed: float
    ctrl_f_edge: List[float]
    fixed_f_edge: List[float]
    delta_real: List[float]
    delta_real_median: float
    wilcoxon_p_real: float
    sign_pos_real: int
    sign_neg_real: int
    # PORTE 3 : shuffle (ordre détruit, même best_fixed)
    ctrl_f_edge_shuffle: List[float]
    fixed_f_edge_shuffle: List[float]
    delta_shuffle: List[float]
    delta_shuffle_median: float
    real_minus_shuffle: float
    # analogue de clôture (α-ω non applicable : phase scalaire monotone — en-tête)
    e_final_ctrl_median: float        # e_N organe (attendu ≈ target : retour transformé)
    e_final_fixed_median: float
    gap_binary_ctrl_median: float     # moyenne du gap binaire par cycle, médiane (organe)
    gap_binary_fixed_median: float
    verdict: str


def run_structural_regulation(
    path: str,
    *,
    n_cycles: int = N_CYCLES_DEFAULT,
    fixed_gains: Sequence[float] = STRUCT_GAIN_SWEEP,
    shuffle_seed_base: int = SHUFFLE_SEED_BASE,
    delta_min: float = DELTA_MIN_DEFAULT,
    n_shuffle: int = N_SHUFFLE_DEFAULT,
    line_range: Optional[Tuple[int, int]] = None,
    eta: float = ETA_STRUCT,
) -> StructuralRegulationReport:
    """Exécute l'ordre lexicographique (0)→(i)→(ii)→(iii) sur ``n_cycles`` cycles réels.

    DÉTERMINISTE : cycles dans l'ordre du fichier, shuffles seedés
    (``shuffle_seed_base + i``), aucune source aléatoire non seedée. Les portes
    (ii)/(iii) sont calculées même si une porte amont tombe (à titre informatif,
    modèle T21) ; le VERDICT, lui, suit strictement l'ordre lexicographique.
    ``line_range`` (T26) restreint le CHARGEMENT au bloc de lignes gelé — portes,
    seuils et instrument STRICTEMENT inchangés.
    ``eta`` (T30) paramètre l'ORGANE des portes (ii)/(iii) via l'argument d'appel
    de ``reconstruct_profile`` — ``ETA_STRUCT`` n'est jamais édité ; le défaut
    (``eta=ETA_STRUCT``) est BYTE-IDENTIQUE aux runs T24/T25/T26 (testé).
    """
    profiles = collect_profiles(path, n_cycles=n_cycles, line_range=line_range)
    n = len(profiles)
    if n == 0:
        raise ValueError(f"aucun cycle utilisable dans {path!r}")
    orientation_lists: List[List[int]] = [[tk.orientation for tk in p] for p in profiles]

    # --- PORTE 0 : pré-validation d'instrument (1er profil réel) -----------------
    order_primary = assert_order_sensitive(
        obs_struct_frozen, profiles[0], shuffle_fn=shuffle_tokens,
        delta_min=delta_min, n_shuffle=n_shuffle,
    )
    order_multiset = assert_order_sensitive(
        obs_struct_multiset, profiles[0], shuffle_fn=shuffle_tokens,
        delta_min=delta_min, n_shuffle=n_shuffle,
    )
    gate0_primary = order_primary.is_order_sensitive
    gate0_multiset = order_multiset.is_vacuous
    order_gaps_info = [
        assert_order_sensitive(
            obs_struct_frozen, p, shuffle_fn=shuffle_tokens,
            delta_min=delta_min, n_shuffle=n_shuffle,
        ).gap
        for p in profiles[:N_PROFILES_GATE0_INFO]
    ]

    # --- PORTE 1 : pivots ---------------------------------------------------------
    pivot_exact = pivot_eta0_is_exact(orientation_lists[0])
    pivot_noflip = pivot_noflip_delta()

    # --- PORTE 2 : organe vs best_fixed sur le réel -------------------------------
    best_g = best_fixed_gain(orientation_lists, fixed_gains=fixed_gains)

    ctrl_f, fixed_f, delta_real = [], [], []
    e_fin_c, e_fin_f, gb_c, gb_f = [], [], [], []
    for orients in orientation_lists:
        tr_c = reconstruct_profile(orients, eta=eta, g0=G0_STRUCT)
        tr_f = reconstruct_fixed(orients, g_fixed=best_g)
        fe_c = f_edge_struct(tr_c)
        fe_f = f_edge_struct(tr_f)
        ctrl_f.append(fe_c)
        fixed_f.append(fe_f)
        delta_real.append(fe_c - fe_f)
        e_fin_c.append(tr_c.e[-1])
        e_fin_f.append(tr_f.e[-1])
        gb_c.append(sum(tr_c.gap_binary) / len(tr_c.gap_binary))
        gb_f.append(sum(tr_f.gap_binary) / len(tr_f.gap_binary))

    delta_real_median = _median(delta_real)
    _, p_real, _ = wilcoxon_signed_rank(delta_real)
    sign_pos = sum(1 for d in delta_real if d > 0)
    sign_neg = sum(1 for d in delta_real if d < 0)

    # --- PORTE 3 : shuffle (ordre détruit ; même best_g, shuffles seedés) ---------
    ctrl_f_sh, fixed_f_sh, delta_sh = [], [], []
    for i, orients in enumerate(orientation_lists):
        sh = shuffle_orientations(orients, shuffle_seed_base + i)
        tr_c = reconstruct_profile(sh, eta=eta, g0=G0_STRUCT)
        tr_f = reconstruct_fixed(sh, g_fixed=best_g)
        fe_c = f_edge_struct(tr_c)
        fe_f = f_edge_struct(tr_f)
        ctrl_f_sh.append(fe_c)
        fixed_f_sh.append(fe_f)
        delta_sh.append(fe_c - fe_f)
    delta_shuffle_median = _median(delta_sh)
    real_minus_shuffle = delta_real_median - delta_shuffle_median

    # --- verdict lexicographique GELÉ ----------------------------------------------
    if not gate0_primary:
        verdict = "NULL-instrument-vacuous"       # 7e null localisé à l'observable
    elif not gate0_multiset:
        verdict = "PORTE-SUSPECTE"                # la porte elle-même est à réexaminer
    elif (not pivot_exact) or pivot_noflip != 0.0:
        verdict = "PIVOT-CASSE"                   # bug, pas résultat
    elif delta_real_median < THRESHOLD_INERTE:
        verdict = "MORTE"
    elif delta_real_median < THRESHOLD_ACTIVE:
        verdict = "INERTE-déclarée"
    elif real_minus_shuffle < THRESHOLD_STRUCTURE:
        verdict = "NULL-ordre-détruit"
    else:
        verdict = "ACTIVE-structurelle"

    return StructuralRegulationReport(
        n_cycles=n,
        tokens_per_cycle=[len(o) for o in orientation_lists],
        flip_fracs=[flip_fraction(o) for o in orientation_lists],
        order_primary=order_primary,
        order_multiset=order_multiset,
        gate0_primary_sensitive=gate0_primary,
        gate0_multiset_vacuous=gate0_multiset,
        order_gaps_info=order_gaps_info,
        pivot_eta0_exact=pivot_exact,
        pivot_noflip=pivot_noflip,
        best_fixed=best_g,
        ctrl_f_edge=ctrl_f,
        fixed_f_edge=fixed_f,
        delta_real=delta_real,
        delta_real_median=delta_real_median,
        wilcoxon_p_real=p_real,
        sign_pos_real=sign_pos,
        sign_neg_real=sign_neg,
        ctrl_f_edge_shuffle=ctrl_f_sh,
        fixed_f_edge_shuffle=fixed_f_sh,
        delta_shuffle=delta_sh,
        delta_shuffle_median=delta_shuffle_median,
        real_minus_shuffle=real_minus_shuffle,
        e_final_ctrl_median=_median(e_fin_c),
        e_final_fixed_median=_median(e_fin_f),
        gap_binary_ctrl_median=_median(gb_c),
        gap_binary_fixed_median=_median(gb_f),
        verdict=verdict,
    )


def _default_dataset() -> Path:
    p = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
    if not p.is_file():
        p = Path(__file__).resolve().parents[3] / "dataset_aba.txt"
    return p


def _corpus_claude() -> Path:
    p = Path("F:/code/claude/spiraton-enhanced/corpus_claude_aba.txt")
    if not p.is_file():
        p = Path(__file__).resolve().parents[3] / "corpus_claude_aba.txt"
    return p


# Chemin du 2e corpus réel (T25). Le DÉFAUT du module reste ``dataset_aba.txt``.
CORPUS_CLAUDE = _corpus_claude()


if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    import sys

    # T25 : ``python -m …structural_regulation claude`` ⇒ corpus_claude, 76 cycles,
    # descripteur de population AVANT toute porte, strate longueur après.
    # T26 : ``python -m …structural_regulation bloc26`` ⇒ dataset_aba.txt, bloc frais
    # GELÉ lignes 1001-3000, descripteur (avec distribution d'excursion) AVANT toute
    # porte, strate d'excursion + Spearman + porte 3 partitionnée + lentille oracle.
    # Sans argument : run T24 inchangé (dataset_aba.txt, 40 cycles).
    on_claude = "claude" in sys.argv[1:]
    on_block26 = "bloc26" in sys.argv[1:]
    ds = CORPUS_CLAUDE if on_claude else _default_dataset()
    n_run = N_CYCLES_CLAUDE if on_claude else (N_CYCLES_BLOCK26 if on_block26 else N_CYCLES_DEFAULT)
    lr = BLOCK26_LINES if on_block26 else None

    print(f"corpus            : {ds}")
    if on_block26:
        print(f"bloc gelé         : lignes {BLOCK26_LINES[0]}-{BLOCK26_LINES[1]} (indices gelés AVANT lecture)")
    if on_claude or on_block26:
        d = population_descriptor(str(ds), n_cycles=n_run, line_range=lr)
        print("--- DESCRIPTEUR DE POPULATION (rapporté AVANT toute porte — leçon T24) ---")
        print(f"n utilisables     : {d.n_cycles}")
        print(f"tokens/cycle      : min={d.tokens_min} med={d.tokens_median} max={d.tokens_max}")
        print(f"k/N               : min={d.kn_min:.4f} med={d.kn_median:.4f} max={d.kn_max:.4f} sigma={d.kn_sigma:.4f}")
        print(f"profils k/N=phi* exact : {d.n_at_phi_star_exact} (cas limite d'instrument T24 : Δ=0 par construction)")
        print(f"critère σ ≥ {SIGMA_KN_MATERIAL:.3f} : {'PASS (P-a testable)' if d.variance_material else 'FAIL (seule P-b en jeu)'}")
        if on_block26:
            print(f"excursion |k−φ*N| : min={d.exc_min:.4f} med={d.exc_median:.4f} max={d.exc_max:.4f}")
            print(f"strates (seuil=band={EXCURSION_THRESHOLD}) : HAUTE n={d.n_exc_high} | BASSE n={d.n_cycles - d.n_exc_high}")

    r = run_structural_regulation(str(ds), n_cycles=n_run, line_range=lr)
    print(f"n_cycles          : {r.n_cycles}")
    print(f"tokens/cycle      : min={min(r.tokens_per_cycle)} med={_median([float(x) for x in r.tokens_per_cycle])} max={max(r.tokens_per_cycle)}")
    print(f"k/N (flip_frac)   : min={min(r.flip_fracs):.4f} med={_median(r.flip_fracs):.4f} max={max(r.flip_fracs):.4f}")
    print("--- PORTE 0 (pré-validation d'instrument, δ_min=1e-4, n_shuffle=8) ---")
    print(f"PRIMAIRE  obs_réel={r.order_primary.obs_real:.6f} obs_shuffle={r.order_primary.obs_shuffle:.6f} gap={r.order_primary.gap:.6e} order_sensitive={r.order_primary.is_order_sensitive}")
    print(f"MULTISET  obs_réel={r.order_multiset.obs_real:.6f} obs_shuffle={r.order_multiset.obs_shuffle:.6f} gap={r.order_multiset.gap:.6e} vacuous={r.order_multiset.is_vacuous}")
    print(f"gaps primaire (5 premiers profils, info) : {[f'{g:.4e}' for g in r.order_gaps_info]}")
    print("--- PORTE 1 (pivots) ---")
    print(f"pivot η=0 exact   : {r.pivot_eta0_exact}")
    print(f"pivot sans-flip Δ : {r.pivot_noflip}")
    print("--- PORTE 2 (organe vs best_fixed, réel) ---")
    print(f"best_fixed        : {r.best_fixed}")
    print(f"f_edge organe méd : {_median(r.ctrl_f_edge):.4f} | f_edge fixe méd : {_median(r.fixed_f_edge):.4f}")
    print(f"Δf_edge médian    : {r.delta_real_median:+.4f}  (Wilcoxon p={r.wilcoxon_p_real:.3e}, signes +{r.sign_pos_real}/−{r.sign_neg_real}/{r.n_cycles})")
    print("--- PORTE 3 (shuffle, ordre détruit) ---")
    print(f"Δf_edge shuffle   : {r.delta_shuffle_median:+.4f}")
    print(f"réel − shuffle    : {r.real_minus_shuffle:+.4f}")
    print("--- clôture (analogue α-ω : e_N, cible = retour avec avance d'1 token) ---")
    print(f"e_N organe médian : {r.e_final_ctrl_median:+.4f} | e_N fixe médian : {r.e_final_fixed_median:+.4f}")
    print(f"gap_binaire méd   : organe {r.gap_binary_ctrl_median:.4f} | fixe {r.gap_binary_fixed_median:.4f}")
    if on_claude:
        st = length_strata(r.delta_real, r.tokens_per_cycle)
        print("--- strate longueur (lecture partitionnée pré-déclarée §2a) ---")
        print(f"COURTS (N<{SHORT_MAX_TOKENS})  : n={st.n_short} Δ méd={st.delta_short_median:+.4f}")
        print(f"LONGS  (N≥{LONG_MIN_TOKENS}) : n={st.n_long} Δ méd={st.delta_long_median:+.4f}")

    if on_block26:
        profiles = collect_profiles(str(ds), n_cycles=n_run, line_range=lr)
        ol = [[tk.orientation for tk in p] for p in profiles]
        excs = [excursion(o) for o in ol]

        # PORTE 0 re-jouée sur le 1er profil à excursion HAUTE (émission §5)
        idx_high = next(
            (i for i, e in enumerate(excs) if e > EXCURSION_THRESHOLD), None
        )
        print("--- PORTE 0 bis : 1er profil à excursion HAUTE ---")
        if idx_high is None:
            print("aucun profil à excursion > seuil dans le bloc (rapporté tel quel)")
        else:
            rep_h = assert_order_sensitive(
                obs_struct_frozen, profiles[idx_high], shuffle_fn=shuffle_tokens
            )
            rep_hm = assert_order_sensitive(
                obs_struct_multiset, profiles[idx_high], shuffle_fn=shuffle_tokens
            )
            print(f"profil #{idx_high} (N={len(ol[idx_high])}, exc={excs[idx_high]:.4f})")
            print(f"PRIMAIRE  obs_réel={rep_h.obs_real:.6f} obs_shuffle={rep_h.obs_shuffle:.6f} gap={rep_h.gap:.6e} order_sensitive={rep_h.is_order_sensitive}")
            print(f"MULTISET  gap={rep_hm.gap:.6e} vacuous={rep_hm.is_vacuous}")

        # STRATE D'EXCURSION : lecture partitionnée des MÊMES Δ (aucun recalcul)
        st26 = excursion_strata(r.delta_real, excs)
        d_high = [d for d, e in zip(r.delta_real, excs) if e > st26.threshold]
        d_low = [d for d, e in zip(r.delta_real, excs) if e <= st26.threshold]
        _, p_high, n_eff_high = wilcoxon_signed_rank(d_high)
        _, p_low, n_eff_low = wilcoxon_signed_rank(d_low)
        print(f"--- STRATE D'EXCURSION (seuil=band={st26.threshold}, plancher {MIN_NONZERO_STRATUM} non-nuls) ---")
        print(f"HAUTE : n={st26.n_high} Δ méd={st26.delta_high_median:+.4f} "
              f"signes +{sum(1 for d in d_high if d > 0)}/−{sum(1 for d in d_high if d < 0)}/{st26.n_high} "
              f"non-nuls={st26.n_high_nonzero} interprétable={st26.high_interpretable} "
              f"Wilcoxon p={p_high:.3e} (n_eff={n_eff_high})")
        print(f"BASSE : n={st26.n_low} Δ méd={st26.delta_low_median:+.4f} "
              f"signes +{sum(1 for d in d_low if d > 0)}/−{sum(1 for d in d_low if d < 0)}/{st26.n_low} "
              f"non-nuls={st26.n_low_nonzero} interprétable={st26.low_interpretable} "
              f"Wilcoxon p={p_low:.3e} (n_eff={n_eff_low})")
        print(f"CONTRASTE Δ_haute − Δ_basse : {st26.contrast:+.4f}")
        rho_s = spearman_rho(r.delta_real, excs)
        p_s = spearman_t_pvalue(rho_s, len(excs))
        print(f"Spearman(Δ, excursion) bloc entier : rho={rho_s:+.4f} p={p_s:.3e} (n={len(excs)})")

        # PORTE 3 PARTITIONNÉE : mêmes Δ shuffle, lus par strate (excursion
        # shuffle-invariante ⇒ chaque cycle reste dans sa strate)
        dsh_high = [d for d, e in zip(r.delta_shuffle, excs) if e > st26.threshold]
        dsh_low = [d for d, e in zip(r.delta_shuffle, excs) if e <= st26.threshold]
        med_sh_high = _median(dsh_high) if dsh_high else float("nan")
        med_sh_low = _median(dsh_low) if dsh_low else float("nan")
        print("--- PORTE 3 PARTITIONNÉE (shuffle par strate) ---")
        print(f"HAUTE : Δ shuffle méd={med_sh_high:+.4f} | réel−shuffle={st26.delta_high_median - med_sh_high:+.4f}")
        print(f"BASSE : Δ shuffle méd={med_sh_low:+.4f} | réel−shuffle={st26.delta_low_median - med_sh_low:+.4f}")

        # LENTILLE D'ÉQUITÉ héritée T25 (REFUS) : dominance vs ORACLE fixe par
        # cycle (grille fine 0.500..2.000 pas 0.005, hors protocole gelé — contrôle)
        if d_high:
            fine = [0.5 + 0.005 * i for i in range(301)]
            ol_high = [o for o, e in zip(ol, excs) if e > st26.threshold]
            fe_ctrl_high = [c for c, e in zip(r.ctrl_f_edge, excs) if e > st26.threshold]
            fe_oracle = [
                max(f_edge_struct(reconstruct_fixed(o, g_fixed=g)) for g in fine)
                for o in ol_high
            ]
            d_or = [c - f for c, f in zip(fe_ctrl_high, fe_oracle)]
            _, p_or, n_or = wilcoxon_signed_rank(d_or)
            print("--- LENTILLE D'ÉQUITÉ (strate haute vs ORACLE fixe par cycle, grille fine) ---")
            print(f"Δ oracle : méd={_median(d_or):+.4f} min={min(d_or):+.4f} "
                  f"signes +{sum(1 for d in d_or if d > 0)}/−{sum(1 for d in d_or if d < 0)}/{len(d_or)} "
                  f"Wilcoxon p={p_or:.3e} (n_eff={n_or})")

    print(f"VERDICT (provisoire, ingénieur statue) : {r.verdict}")
