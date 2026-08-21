from __future__ import annotations

"""Carte spectrale de l'organe — déformation de la carte(η) par rotation (Tour 28).

H28 (émission linguiste, gelée AVANT toute mesure) : la déformation de la carte
signe(Δ_nom)(N, k) quand η varie est une ROTATION de fréquence, pas un
amortissement. Tant que le clip est inactif, la boucle fermée satisfait (u = e − 1)

    u_{t+1} = (2 − η)·u_t − u_{t−1}

dont les racines sont sur le CERCLE UNITÉ pour η ∈ (0, 4) : cos θ(η) = (2 − η)/2,
|λ| = 1. Chaque η est un oscillateur marginal de fréquence différente ; la carte(η)
est dérivable EXACTEMENT en ``Fraction`` (clip inclus) AVANT le float. Réfutations
gelées : divergence de SIGNE dérivation ↔ float = BUG (arrêt technique) ; carte
monotone/simple émergeant à un η = mécanisme faux ; carte η-invariante = résonance
fausse.

PROTOCOLE GELÉ (émission T28) :

  * η GELÉS : {1/4, 1, 3/2, 4} + ancre 1/2 en CONTRÔLE DUR (doit reproduire
    byte-pour-signe la carte T27 gravée dans ``horizon_law``). Tous rationnels,
    trace ``Fraction`` exacte clip inclus (``trace_e_exact`` T27, déjà générique).
  * Grille : la même grille T27 gelée — N ∈ [6, 24], k ∈ [2, N−2], 228 cellules.
  * Baseline : Δ_nom(η) = f_edge(organe η, g0=1) − f_edge(fixe g=1) ;
    ``reconstruct_fixed`` est η-INVARIANTE (un calcul par cellule, une fois,
    mis en cache). Oracle par cellule INTERDIT. B_sweep_global secondaire non
    requis ce tour (le nominal a gagné le sweep de la grille au T27).
  * FORME FORTE par η : la carte(η) complète est dérivée en Fraction et GELÉE
    ci-dessous (``FROZEN_MAPS``/``FROZEN_WITNESSES``) AVANT d'exécuter
    l'instrument float à ce η. Critère : concordance de SIGNE 228/228 par η ;
    divergences de MAGNITUDE au bord exact |e−1| = 1/2 tolérées et rapportées
    (patron T27). L'instrument ``experimental/structural_gap.py`` reste GELÉ
    byte-à-byte ; ``edge_controller.py`` INTACT (η passe par le PARAMÈTRE
    d'appel ``reconstruct_profile(eta=…)``, jamais par ``ETA_STRUCT``).
  * Seuils T24 hérités, ZÉRO degré de liberté : band = 0.5, φ* = 2/3, g0 = 1,
    target = 1, g_min = 0.5, g_max = 2.0, δ_min = 1e-4.
  * PAS DE VOLET RÉEL ce tour (déclaré à l'émission) : les Δ réels historiques
    sont à η = 1/2 seulement ; appliquer carte(η) aux profils réels mono-flip
    serait 100 % par construction. La clôture lévo·in reviendra au tour
    « η sur runners ».

CARTES DÉRIVÉES ET GELÉES (Fraction exacte, clip inclus, 2026-07-11, AVANT le
float) : voir ``FROZEN_MAPS`` (lignes N = 6..24, colonnes k = 2..N−2 ; '.' =
cellule basse à Δ = 0 ; une cellule basse à Δ ≠ 0 — violation P-gate — apparaît
avec son signe, elle N'EST PAS masquée). Les 6 cellules-témoins T27 prennent à
chaque η leurs NOUVELLES valeurs ``Fraction`` (``FROZEN_WITNESSES``), gelées
avant float ; à η = 1/2 elles doivent coïncider avec ``WITNESS_CELLS`` T27.

PRÉDICATS QUALITATIFS — TRANCHÉS PAR LA DÉRIVATION (gravés, jamais forcés) :

  1. P-gate par η : TIENT à η ∈ {1/4, 1/2, 1, 3/2} (0 violation sur les 19
     cellules basses). VIOLÉE à η = 4 : (8,5) → Δ = −1/8 et (11,7) → Δ = −1/11
     (toutes deux basses-précoces, |3k−2N| = 1). η_crit = 4 dans la famille
     gelée : l'organe sur-corrige et sort de bande en basse excursion — la
     prédiction « à η assez grand P-gate violée » est CONFIRMÉE à η = 4, et le
     clip protège la bande jusqu'à η = 3/2 inclus.
  2. Zéro (8,6) : Δ = 0 à η ∈ {1/4, 1/2} ; Δ = +1/4 à η ∈ {1, 3/2, 4}. Le zéro
     N'est PAS propre à η = 1/2 (il tient aussi à 1/4) : la prédiction « quitte
     zéro pour η ≠ 1/2 » est RÉFUTÉE à 1/4 et confirmée à {1, 3/2, 4} — le zéro
     est un plateau de basse fréquence, pas une coïncidence ponctuelle.
  3. Témoins : (7,4) : −1/7 → −2/7 → 0 → +1/7 → +1/7 — CHANGE DE SIGNE ; la
     bascule passe par Δ = 0 EXACT à η = 1 (le η de bascule est 1, dérivé).
     (13,8) : +1/13 → +3/13 → +3/13 → +3/13 → +3/13 — ne change JAMAIS de
     signe. « Au moins un témoin bascule » : CONFIRMÉ par (7,4).
  4. N_plus_tardif(η) (comptes de '+' côté tardif k > 2N/3, strate haute,
     57 cellules) : 55 (η=1/4), 56 (1/2), 57 (1), 57 (3/2), 57 (4). Il CROÎT
     quand η monte — la direction mécaniste attendue à l'émission (décroît
     dans (0, ~2)) est RÉFUTÉE par la dérivation. Rapporté, jamais forcé.
  5. η = 1 (θ = π/3) : périodicité-6 de u = e − 1 sur les fenêtres de plateau
     non clipées : 884/884 fenêtres exactes (125 cellules éligibles, k ≥ 6,
     pas t+1..t+5 non clipés) — CONFIRMÉE, structure de bande mod-6.
  6. η = 4 (cos θ = −1, racine double défective) : clip-dominée — fraction de
     pas aux bornes {min 0, MÉDIANE 2/3, max 23/24}, 61/228 cellules avec TOUS
     les pas t ≥ 1 aux bornes. MAIS le signe est majoritairement POSITIF
     ({+: 142, 0: 59, −: 27} ; Δ ≤ 0 : 86/228) : la moitié « majoritairement
     Δ≤0 » de la prédiction est RÉFUTÉE. Qualitativement distincte : OUI —
     seule carte à violer la P-gate, Hamming ≥ 12 vs toutes les autres.

H28 (rotation vs amortissement) — LA DÉRIVATION TRANCHE : la carte N'est PAS
η-invariante (Hamming inter-cartes 8..76 sur 228 : 1/4↔1/2 = 50, 1/2↔1 = 24,
1↔3/2 = 8, 3/2↔4 = 12, 1/4↔4 = 76) et AUCUN η ne fait émerger une carte
monotone/simple : le côté précoce reste un mélange arithmétique {+, 0, −} à
tout η ({45,51,56} à 1/4 ; {62,47,43} à 1/2 ; {72,43,37} à 1 ; {75,45,32} à
3/2 ; {85,42,25} à 4) — un amortissement donnerait une carte de plus en plus
uniforme, une rotation déplace les zéros/négatifs sans les faire disparaître.
Les deux réfutations gelées (monotone-simple, η-invariante) sont écartées.

VERDICT MESURÉ AU FLOAT (2026-07-11, APRÈS gel des cartes — gravé) :

  * Porte 0a re-confirmée : PASSE ((13,8) gap primaire = 2.185897e-1, (7,5)
    gap = 8.095238e-2, ≥ δ_min = 1e-4 ; multiset vacuous 0.0 sur les deux).
  * Concordance dérivation ↔ instrument float gelé, par η : SIGNES identiques
    5 × 228/228 — AUCUN arrêt technique. Divergences de MAGNITUDE (> 1e-9,
    TOUTES avec ≥ 1 point EXACTEMENT au bord de bande |e−1| = 1/2) :
      η=1/4 : 2 — (9,7) float 1/3 vs exact 2/9 ; (15,11) float 2/5 vs 1/3 ;
      η=1/2 : 2 — (9,7) 1/3 vs 2/9 ; (15,11) 2/5 vs 1/3 (== T27, contrôle) ;
      η=1   : 6 — (9,7) 4/9 vs 1/3 ; (14,7) 9/14 vs 5/7 ; (15,11) 8/15 vs
              7/15 ; (16,8) 1/2 vs 9/16 ; (18,9) 5/9 vs 11/18 ; (20,10)
              13/20 vs 7/10 ;
      η=3/2 : 2 — (9,7) 4/9 vs 1/3 ; (15,11) 8/15 vs 7/15 ;
      η=4   : 10 — (8,6) 1/8 vs 1/4 ; (9,4) 1/9 vs 2/9 ; (9,7) 4/9 vs 1/3 ;
              (12,9) 5/12 vs 1/2 ; (15,11) 2/5 vs 1/3 ; (16,12) 1/2 vs 9/16 ;
              (20,15) 3/5 vs 13/20 ; (21,10) 3/7 vs 10/21 ; (24,11) 5/12 vs
              11/24 ; (24,18) 5/8 vs 2/3.
    Le signe, seule quantité décisionnelle, est partout celui de la Fraction.
    NOTE : à η = 4 la cellule-témoin (8,6) fait partie des divergences de
    magnitude (float 1/8 vs Fraction 1/4, un point au bord exact) ⟹
    ``witnesses_float_ok`` vaut False à η = 4 SEULEMENT (signe préservé,
    valeurs Fraction gelées reproduites exactement) — True aux 4 autres η.
  * Contrôle dur η = 1/2 : carte dérivée == carte T27 gravée byte-pour-signe ;
    témoins T27 inchangés (Fraction exacte ET float ≤ 1e-12) ; divergences de
    magnitude identiques à T27 ((9,7) et (15,11)).

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/``, NI ``regulate_step``,
NI ``structural_gap.py`` (gelé), NI ``horizon_law.py`` (T27 gravé — importé en
lecture seule). Anti-circularité : profils construits depuis (N, k) directement ;
jamais les dims 0-5 du 33D. Tout est déterministe (aucune source aléatoire hors
shuffles seedés de la porte 0a, infra T23).
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from ..experimental.structural_gap import (
    G0_STRUCT,
    f_edge_struct,
    reconstruct_fixed,
    reconstruct_profile,
)
from .horizon_law import (
    BAND_EXACT,
    G_MAX_EXACT,
    G_MIN_EXACT,
    GRID_K_MIN,
    GRID_N_MAX,
    GRID_N_MIN,
    PHI_STAR_EXACT,
    TARGET_EXACT,
    WITNESS_CELLS,
    f_edge_exact,
    flip_side,
    gate0a,
    grid_cells,
    is_high_excursion,
    profile_of,
    trace_e_exact,
)


# --- les η GELÉS (émission T28 — rationnels, jamais ajustés) ------------------------

ETA_LABELS_FROZEN: Tuple[str, ...] = ("1/4", "1/2", "1", "3/2", "4")
ETAS_FROZEN: Dict[str, Fraction] = {
    "1/4": Fraction(1, 4),
    "1/2": Fraction(1, 2),      # ancre = CONTRÔLE DUR (carte T27)
    "1": Fraction(1),
    "3/2": Fraction(3, 2),
    "4": Fraction(4),
}
T27_CONTROL_LABEL = "1/2"


def cos_theta_exact(eta: Fraction) -> Fraction:
    """cos θ(η) = (2 − η)/2 — la fréquence de rotation de la boucle fermée (H28).

    Racines de x² − (2−η)x + 1 : |λ| = 1 ⟺ |cos θ| ≤ 1 ⟺ η ∈ [0, 4]. À η = 4,
    cos θ = −1 : racine double défective λ = −1 (bord du domaine oscillant).
    """
    return (2 - eta) / 2


# --- CARTES(η) DÉRIVÉES EN FRACTION ET GELÉES AVANT LE FLOAT (forme forte) ----------
#
# Lignes N = 6..24, colonnes k = 2..N−2. '.' = cellule basse (|3k−2N| ≤ 1) à Δ = 0 ;
# les cellules basses à Δ ≠ 0 (violations P-gate, η = 4 seulement) portent leur signe.

FROZEN_MAPS: Dict[str, Tuple[str, ...]] = {
    "1/4": (
        "--.",
        "---.",
        "0--.0",
        "0--+.+",
        "-----.+",
        "00--+.0+",
        "00--0+.++",
        "--0--++.++",
        "000--++.+++",
        "-----0++.+++",
        "0000-0+++.+++",
        "0000-0+++.++++",
        "------0+++.++++",
        "00000-0++++.++++",
        "-0000-00+++.+++++",
        "0------0++++.+++++",
        "000000-+0++++.+++++",
        "----00-+0++++.++++++",
        "0000----++++++.++++++",
    ),
    "1/2": (                    # == carte T27 gravée dans horizon_law (contrôle dur)
        "-0.",
        "0--.",
        "--+.0",
        "0--+.+",
        "0--++.+",
        "00-0+.++",
        "00-0++.++",
        "----0++.++",
        "000-0++.+++",
        "-00-++++.+++",
        "0----0+++.+++",
        "0000-++++.++++",
        "--00-+++++.++++",
        "00---0+++++.++++",
        "-0000-+++++.+++++",
        "0--00-++++++.+++++",
        "000---0++++++.+++++",
        "--0000-++++++.++++++",
        "00--00-+++++++.++++++",
    ),
    "1": (
        "-0.",
        "0-0.",
        "--+.+",
        "0-++.+",
        "000++.+",
        "0--++.++",
        "00-+++.++",
        "--0++++.++",
        "00--+++.+++",
        "-00-++++.+++",
        "0--0+++++.+++",
        "000--++++.++++",
        "--00-+++++.++++",
        "00--0++++++.++++",
        "-000--+++++.+++++",
        "0--00-++++++.+++++",
        "000--0+++++++.+++++",
        "--000-0++++++.++++++",
        "00--00-+++++++.++++++",
    ),
    "3/2": (
        "-0.",
        "00+.",
        "--+.+",
        "0-++.+",
        "00+++.+",
        "0-0++.++",
        "00-+++.++",
        "--0++++.++",
        "00-0+++.+++",
        "-00-++++.+++",
        "0--0+++++.+++",
        "000-0++++.++++",
        "--00-+++++.++++",
        "00--0++++++.++++",
        "-000-0+++++.+++++",
        "0--00-++++++.+++++",
        "000--0+++++++.+++++",
        "--000-+++++++.++++++",
        "00--00-+++++++.++++++",
    ),
    "4": (
        "+0.",
        "0++.",
        "--+-+",                 # (8,5) : violation P-gate, Δ = −1/8 (visible)
        "0+++.+",
        "00+++.+",
        "0-0++-++",              # (11,7) : violation P-gate, Δ = −1/11 (visible)
        "00++++.++",
        "--0++++.++",
        "0+-0+++.+++",
        "-00+++++.+++",
        "0--0+++++.+++",
        "00+-0++++.++++",
        "--00++++++.++++",
        "00--0++++++.++++",
        "-000-0+++++.+++++",
        "0--00+++++++.+++++",
        "000--0+++++++.+++++",
        "--000-+++++++.++++++",
        "00--00++++++++.++++++",
    ),
}

# les 6 cellules-témoins T27, valeurs Fraction PAR η, gelées avant float ------------
FROZEN_WITNESSES: Dict[str, Dict[Tuple[int, int], Fraction]] = {
    "1/4": {(7, 4): Fraction(-1, 7), (8, 6): Fraction(0), (10, 6): Fraction(-1, 10),
            (13, 8): Fraction(1, 13), (16, 10): Fraction(1, 4), (19, 12): Fraction(4, 19)},
    "1/2": {(7, 4): Fraction(-2, 7), (8, 6): Fraction(0), (10, 6): Fraction(1, 5),
            (13, 8): Fraction(3, 13), (16, 10): Fraction(1, 4), (19, 12): Fraction(4, 19)},
    "1": {(7, 4): Fraction(0), (8, 6): Fraction(1, 4), (10, 6): Fraction(1, 5),
          (13, 8): Fraction(3, 13), (16, 10): Fraction(1, 4), (19, 12): Fraction(4, 19)},
    "3/2": {(7, 4): Fraction(1, 7), (8, 6): Fraction(1, 4), (10, 6): Fraction(1, 5),
            (13, 8): Fraction(3, 13), (16, 10): Fraction(1, 4), (19, 12): Fraction(4, 19)},
    "4": {(7, 4): Fraction(1, 7), (8, 6): Fraction(1, 4), (10, 6): Fraction(1, 10),
          (13, 8): Fraction(3, 13), (16, 10): Fraction(3, 16), (19, 12): Fraction(3, 19)},
}

# violations P-gate dérivées (prédicat 1) : uniquement η = 4, gelées ----------------
FROZEN_PGATE_VIOLATIONS: Dict[str, Tuple[Tuple[int, int, Fraction], ...]] = {
    "1/4": (), "1/2": (), "1": (), "3/2": (),
    "4": ((8, 5, Fraction(-1, 8)), (11, 7, Fraction(-1, 11))),
}

# N_plus_tardif (strate haute, 57 cellules tardives) par η, gelé (prédicat 4) ------
FROZEN_N_PLUS_TARDIF: Dict[str, int] = {"1/4": 55, "1/2": 56, "1": 57, "3/2": 57, "4": 57}


# --- moteur exact paramétré par η (réutilise trace_e_exact T27, clip inclus) -------

@lru_cache(maxsize=None)
def _fixed_f_edge_exact(n: int, k: int) -> Fraction:
    """f_edge EXACT du lecteur fixe g = 1 — η-INVARIANT : un calcul par cellule."""
    return f_edge_exact(trace_e_exact(n, k, Fraction(0)))


@lru_cache(maxsize=None)
def delta_nom_exact_eta(n: int, k: int, eta: Fraction) -> Fraction:
    """Δ_nom(η) EXACT : f_edge(organe η, g0=1) − f_edge(fixe g=1) (baseline cachée)."""
    return f_edge_exact(trace_e_exact(n, k, eta)) - _fixed_f_edge_exact(n, k)


def trace_full_exact(n: int, k: int, eta: Fraction
                     ) -> Tuple[List[Fraction], List[Fraction], List[bool]]:
    """Trace EXACTE enrichie : (e, g, clipped) — même récurrence que trace_e_exact.

    ``clipped[t]`` = le clip a MODIFIÉ g au pas t (pré-clip ≠ post-clip). Sert aux
    prédicats 5 (fenêtres non clipées) et 6 (clip-dominance). L'égalité e ==
    trace_e_exact(n, k, eta) est testée (deux chemins, mêmes Fractions).
    """
    inc_plus = PHI_STAR_EXACT * n / k
    inc_minus = (1 - PHI_STAR_EXACT) * n / (n - k)
    p_read, p_ref, g = Fraction(1), Fraction(0), Fraction(1)
    e: List[Fraction] = [p_read - p_ref]
    gs: List[Fraction] = []
    clips: List[bool] = []
    for t in range(n):
        e_t = p_read - p_ref
        g_raw = g - eta * (e_t - TARGET_EXACT)
        g = min(G_MAX_EXACT, max(G_MIN_EXACT, g_raw))
        clips.append(g != g_raw)
        gs.append(g)
        p_read += g
        p_ref += inc_plus if t < k else inc_minus
        e.append(p_read - p_ref)
    return e, gs, clips


def band_edge_touches_eta(n: int, k: int, eta: Fraction) -> int:
    """Points EXACTEMENT sur le bord de bande |e−1| = 1/2 (organe η + fixe).

    Généralisation directe de ``band_edge_touches`` T27 (qui est le cas η = 1/2) :
    là où le comptage float peut différer d'un hit du comptage exact.
    """
    c = 0
    for e_run in (trace_e_exact(n, k, eta), trace_e_exact(n, k, Fraction(0))):
        c += sum(1 for x in e_run[1:] if abs(x - TARGET_EXACT) == BAND_EXACT)
    return c


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def sign_char(d: Fraction) -> str:
    return "+" if d > 0 else ("-" if d < 0 else "0")


def derived_map_exact(eta: Fraction) -> Tuple[str, ...]:
    """Carte signe(Δ_nom(η))(N, k) DÉRIVÉE par le moteur rationnel (clip inclus).

    Même convention d'affichage que ``FROZEN_MAPS`` : '.' = cellule basse à Δ = 0 ;
    toute cellule basse à Δ ≠ 0 porte son signe (une violation P-gate est VISIBLE).
    Doit reproduire la gravure : ``derived_map_exact(η) == FROZEN_MAPS[label]``.
    """
    rows: List[str] = []
    for n in range(GRID_N_MIN, GRID_N_MAX + 1):
        row = ""
        for k in range(GRID_K_MIN, n - 1):
            d = delta_nom_exact_eta(n, k, eta)
            if not is_high_excursion(n, k) and d == 0:
                row += "."
            else:
                row += sign_char(d)
        rows.append(row)
    return tuple(rows)


# --- chemin FLOAT : l'instrument gelé, η par PARAMÈTRE d'appel uniquement ----------

@lru_cache(maxsize=None)
def _fixed_f_edge_float(n: int, k: int) -> float:
    """f_edge float du lecteur fixe g = 1 — η-invariant, UN calcul par cellule."""
    return f_edge_struct(reconstruct_fixed(profile_of(n, k), g_fixed=G0_STRUCT))


def delta_nom_float_eta(n: int, k: int, eta: float) -> float:
    """Δ_nom(η) mesuré par l'INSTRUMENT FLOAT GELÉ (``structural_gap`` byte-à-byte).

    η passe par le paramètre d'appel de ``reconstruct_profile`` — ``ETA_STRUCT``
    n'est JAMAIS édité, ``edge_controller`` reste intact. Les 5 η gelés sont des
    dyadiques exacts en binaire (0.25, 0.5, 1.0, 1.5, 4.0) : float(Fraction) est
    sans perte. Contre-épreuve de la dérivation (concordance de SIGNE exigée).
    """
    fe_o = f_edge_struct(reconstruct_profile(profile_of(n, k), eta=eta, g0=G0_STRUCT))
    return fe_o - _fixed_f_edge_float(n, k)


# --- concordance dérivation ↔ float, par η -----------------------------------------

@dataclass(frozen=True)
class EtaCellRecord:
    """Une cellule à un η : Δ exact (autorité du signe), Δ float, descripteurs."""

    n: int
    k: int
    side: str
    high: bool
    delta_exact: Fraction
    delta_float: float
    sign: int
    band_touches: int


@dataclass(frozen=True)
class EtaMapReport:
    """La carte complète à un η : gel vérifié, concordance float, comptes, prédicats."""

    eta_label: str
    eta: Fraction
    cos_theta: Fraction
    cells: List[EtaCellRecord]
    map_matches_frozen: bool                    # dérivation == gravure (gel honoré)
    sign_mismatches: List[EtaCellRecord]        # float ≠ exact en SIGNE (attendu : 0)
    magnitude_divergences: List[EtaCellRecord]  # |float − exact| > 1e-9 (bord de bande)
    pgate_violations: List[EtaCellRecord]       # cellules basses à Δ ≠ 0
    counts_by_side_high: Dict[Tuple[str, str], Dict[str, int]]
    n_plus_tardif_high: int                     # prédicat 4
    witness_deltas: Dict[Tuple[int, int], Fraction]
    witnesses_match_frozen: bool
    witnesses_float_ok: bool                    # |float − Fraction| ≤ 1e-12 sur les 6


def eta_map_report(label: str) -> EtaMapReport:
    """Mesure la grille entière à ``ETAS_FROZEN[label]`` : exact PUIS float (ordre gelé)."""
    eta = ETAS_FROZEN[label]
    eta_f = float(eta)          # dyadique exact (sans perte pour les 5 η gelés)

    # 1) dérivation exacte, vérifiée contre la gravure (forme forte)
    map_ok = derived_map_exact(eta) == FROZEN_MAPS[label]

    # 2) instrument float gelé, cellule par cellule
    records: List[EtaCellRecord] = []
    for (n, k) in grid_cells():
        d_exact = delta_nom_exact_eta(n, k, eta)
        d_float = delta_nom_float_eta(n, k, eta_f)
        records.append(EtaCellRecord(
            n=n, k=k, side=flip_side(n, k), high=is_high_excursion(n, k),
            delta_exact=d_exact, delta_float=d_float, sign=_sign(d_exact),
            band_touches=band_edge_touches_eta(n, k, eta),
        ))

    sign_mm = [c for c in records if _sign(c.delta_float) != c.sign]
    mag_div = [c for c in records if abs(c.delta_float - float(c.delta_exact)) > 1e-9]
    pgate_viol = [c for c in records if not c.high and c.delta_exact != 0]

    counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for c in records:
        key = (c.side, "haute" if c.high else "basse")
        counts.setdefault(key, {"+": 0, "0": 0, "-": 0})
        counts[key][sign_char(c.delta_exact)] += 1

    npt = sum(1 for c in records
              if c.high and c.side == "tardif" and c.sign > 0)

    wit = {cell: delta_nom_exact_eta(*cell, eta) for cell in WITNESS_CELLS}
    wit_ok = wit == FROZEN_WITNESSES[label]
    wit_float = all(
        abs(delta_nom_float_eta(*cell, eta_f) - float(wit[cell])) <= 1e-12
        for cell in wit
    )

    return EtaMapReport(
        eta_label=label, eta=eta, cos_theta=cos_theta_exact(eta),
        cells=records, map_matches_frozen=map_ok,
        sign_mismatches=sign_mm, magnitude_divergences=mag_div,
        pgate_violations=pgate_viol, counts_by_side_high=counts,
        n_plus_tardif_high=npt,
        witness_deltas=wit, witnesses_match_frozen=wit_ok,
        witnesses_float_ok=wit_float,
    )


# --- prédicat 5 : η = 1 (θ = π/3), périodicité-6 sur plateaux non clipés ------------

@dataclass(frozen=True)
class Period6Report:
    """u_{t+6} == u_t (u = e − 1) sur toutes les fenêtres de plateau non clipées.

    Fenêtre éligible en (N, k), t : t ∈ [0, k−6] (récurrence à inc constant :
    rangs s = t+1..t+5, incréments s−1 et s dans le plateau) ET pas t+1..t+5
    non clipés. À η = 1 : u_{t+1} = u_t − u_{t−1} ⟹ période 6 EXACTE attendue.
    """

    n_cells_eligible: int
    n_windows: int
    n_windows_ok: int
    failures: List[Tuple[int, int, int]]        # (N, k, t) — attendu : []


def period6_report() -> Period6Report:
    """Vérifie la périodicité-6 exacte de u = e − 1 à η = 1 (prédicat 5)."""
    eta1 = Fraction(1)
    n_cells = n_win = n_ok = 0
    fails: List[Tuple[int, int, int]] = []
    for (n, k) in grid_cells():
        if k < 6:
            continue
        e, _gs, clips = trace_full_exact(n, k, eta1)
        u = [x - TARGET_EXACT for x in e]
        cell_has = False
        for t in range(0, k - 5):
            if any(clips[s] for s in range(t + 1, t + 6)):
                continue
            cell_has = True
            n_win += 1
            if u[t + 6] == u[t]:
                n_ok += 1
            else:
                fails.append((n, k, t))
        if cell_has:
            n_cells += 1
    return Period6Report(n_cells_eligible=n_cells, n_windows=n_win,
                         n_windows_ok=n_ok, failures=fails)


# --- prédicat 6 : η = 4 (racine double défective), clip-dominance ------------------

@dataclass(frozen=True)
class ClipDominanceReport:
    """Mesures de saturation du clip à η = 4 sur la grille (prédicat 6).

    ``frac_at_bound`` par cellule = #{t : g_t ∈ {1/2, 2}} / N (post-clip). Une
    cellule est « saturée » si TOUS ses pas t ≥ 1 sont aux bornes (le pas 0 part
    de g0 = 1 : e_0 = target, l'organe ne corrige pas encore).
    """

    frac_min: Fraction
    frac_median: Fraction
    frac_max: Fraction
    n_saturated_after_first: int    # cellules avec tous les g_t (t ≥ 1) aux bornes
    sign_counts: Dict[str, int]     # {+, 0, −} sur les 228 cellules
    n_delta_le0: int                # Δ ≤ 0 (la prédiction « majoritaire » à tester)


def clip_dominance_report() -> ClipDominanceReport:
    """Mesure la clip-dominance à η = 4 (dérivation exacte, aucun float)."""
    eta4 = Fraction(4)
    fracs: List[Fraction] = []
    n_sat = 0
    sc = {"+": 0, "0": 0, "-": 0}
    n_le0 = 0
    for (n, k) in grid_cells():
        _e, gs, _clips = trace_full_exact(n, k, eta4)
        at_bound = sum(1 for g in gs if g in (G_MIN_EXACT, G_MAX_EXACT))
        fracs.append(Fraction(at_bound, len(gs)))
        if all(g in (G_MIN_EXACT, G_MAX_EXACT) for g in gs[1:]):
            n_sat += 1
        d = delta_nom_exact_eta(n, k, eta4)
        sc[sign_char(d)] += 1
        if d <= 0:
            n_le0 += 1
    fr = sorted(fracs)
    return ClipDominanceReport(
        frac_min=fr[0], frac_median=fr[len(fr) // 2], frac_max=fr[-1],
        n_saturated_after_first=n_sat, sign_counts=sc, n_delta_le0=n_le0,
    )


# --- distance de Hamming entre cartes (réfutation « η-invariante ») ----------------

def hamming_between_maps(label_a: str, label_b: str) -> int:
    """# cellules dont le SIGNE exact diffère entre deux η gelés (0 ⟹ invariance)."""
    ea, eb = ETAS_FROZEN[label_a], ETAS_FROZEN[label_b]
    return sum(1 for (n, k) in grid_cells()
               if _sign(delta_nom_exact_eta(n, k, ea)) != _sign(delta_nom_exact_eta(n, k, eb)))


# --- runner de mesure ---------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    print("=== PORTE 0a re-confirmée (héritée T27, δ_min=1e-4) ===")
    g0a = gate0a()
    print(f"(13,8) primaire gap={g0a.primary_high.gap:.6e} sensible={g0a.primary_high.is_order_sensitive} | "
          f"multiset gap={g0a.multiset_high.gap:.6e} vacuous={g0a.multiset_high.is_vacuous}")
    print(f"(7,5)  primaire gap={g0a.primary_low.gap:.6e} sensible={g0a.primary_low.is_order_sensitive} | "
          f"multiset gap={g0a.multiset_low.gap:.6e} vacuous={g0a.multiset_low.is_vacuous}")
    print(f"porte 0a : {'PASSE' if g0a.passes else 'ECHEC'}")

    reports = {}
    for lbl in ETA_LABELS_FROZEN:
        r = eta_map_report(lbl)
        reports[lbl] = r
        print(f"\n=== eta = {lbl}  (cos theta = {r.cos_theta}) ===")
        print(f"carte derivee == gravure gelée : {r.map_matches_frozen}")
        print(f"concordance SIGNE float↔exact : "
              f"{228 - len(r.sign_mismatches)}/228 (mismatches={len(r.sign_mismatches)})")
        print(f"divergences magnitude >1e-9   : {len(r.magnitude_divergences)} "
              f"{[(c.n, c.k, c.delta_float, str(c.delta_exact), c.band_touches) for c in r.magnitude_divergences]}")
        print(f"P-gate violations             : "
              f"{[(c.n, c.k, str(c.delta_exact)) for c in r.pgate_violations]}")
        for key in sorted(r.counts_by_side_high):
            print(f"  {key}: {r.counts_by_side_high[key]}")
        print(f"N_plus_tardif (haute, /57)    : {r.n_plus_tardif_high} "
              f"(gelé : {FROZEN_N_PLUS_TARDIF[lbl]})")
        print(f"témoins : {[(c, str(v)) for c, v in r.witness_deltas.items()]}")
        print(f"témoins == gelés : {r.witnesses_match_frozen} ; float ≤1e-12 : {r.witnesses_float_ok}")
        print("--- carte (exacte ; '.' = basse à Δ=0) ---")
        for i, row in enumerate(FROZEN_MAPS[lbl]):
            print(f"N={GRID_N_MIN + i:2d} : {row}")

    print("\n=== contrôle dur eta=1/2 : témoins T27 inchangés ===")
    r12 = reports[T27_CONTROL_LABEL]
    print(f"witness_deltas == WITNESS_CELLS T27 : {r12.witness_deltas == dict(WITNESS_CELLS)}")

    print("\n=== prédicat 3 : bascule des témoins ===")
    for cell in [(7, 4), (13, 8), (8, 6)]:
        vals = " ; ".join(f"η={lbl}: {reports[lbl].witness_deltas.get(cell, delta_nom_exact_eta(*cell, ETAS_FROZEN[lbl]))}"
                          for lbl in ETA_LABELS_FROZEN)
        print(f"Δ{cell} : {vals}")

    print("\n=== prédicat 5 : η=1, périodicité-6 (plateaux non clipés) ===")
    p6 = period6_report()
    print(f"cellules éligibles={p6.n_cells_eligible} fenêtres={p6.n_windows} "
          f"ok={p6.n_windows_ok} échecs={p6.failures}")

    print("\n=== prédicat 6 : η=4, clip-dominance ===")
    cd = clip_dominance_report()
    print(f"fraction de pas aux bornes : min={cd.frac_min} médiane={cd.frac_median} max={cd.frac_max}")
    print(f"cellules saturées (tous t≥1 aux bornes) : {cd.n_saturated_after_first}/228")
    print(f"signes : {cd.sign_counts} ; Δ≤0 : {cd.n_delta_le0}/228")

    print("\n=== Hamming entre cartes (réfutation « η-invariante ») ===")
    for i, la in enumerate(ETA_LABELS_FROZEN):
        for lb in ETA_LABELS_FROZEN[i + 1:]:
            print(f"  {la} vs {lb} : {hamming_between_maps(la, lb)}/228")
