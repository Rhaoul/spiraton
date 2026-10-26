from __future__ import annotations

"""Loi de l'horizon sur grille synthétique (N, k) — arbitrage de lois gelées (Tour 27).

H27 (émission linguiste, gelée AVANT toute mesure) : le bloc T26 confond N, k et
N−k (sa strate haute vit sur UNE droite, excursion = 2/3, 100 % flip-précoce).
Ce module dissocie ces variables sur une GRILLE SYNTHÉTIQUE exhaustive de profils
d'orientation ``[+1]*k + [−1]*(N−k)`` et départage QUATRE lois candidates gelées,
puis rattache le résultat aux Δ réels DÉJÀ mesurés (T24/T25/T26 — aucun paramètre
neuf). L'instrument ``experimental/structural_gap.py`` est GELÉ byte-à-byte
(condition de comparabilité T24→T27) ; l'organe ``regulate_step`` est INTOUCHÉ ;
η = 0.5, target = 1, band = 0.5, g0 = 1, bornes [0.5, 2.0] — tous hérités T24.

PROTOCOLE GELÉ (émission T27, ordre lexicographique, premier échec = verdict) :

  * Grille : N ∈ [6, 24], k ∈ [2, N−2] (228 cellules ; les DEUX côtés du flip).
  * Δ_nom = f_edge(organe η=0.5, g0=1) − f_edge(lecteur fixe g=1) — baseline
    PRIMAIRE B_nominal = g fixe 1.0 par cellule. Le SIGNE de Δ_nom est calculé
    en ``Fraction`` EXACTE de bout en bout (le moteur rationnel ci-dessous
    réimplémente la récurrence de l'instrument depuis sa spec a priori, sans
    importer sa logique float — deux chemins de code indépendants).
  * Baseline SECONDAIRE B_sweep_global : le g de ``STRUCT_GAIN_SWEEP`` qui
    maximise le f_edge MÉDIAN de toute la grille — rapportée À PART (si g ≠ 1.0 :
    décalage rapporté comme information, rien n'est corrigé). L'oracle par
    cellule est INTERDIT comme baseline (lentille d'équité hors verdict).
  * P-gate (commune) : excursion = |k − (2/3)N| ≤ 0.5 ⟹ Δ = 0 exact — TESTÉE
    sur la grille, pas supposée. Pour k, N entiers : excursion = |3k−2N|/3 ne
    vaut JAMAIS 0.5 exactement ⟹ le seuil sépare proprement (|3k−2N| ≤ 1 bas,
    ≥ 2 haut).
  * Lois candidates sur la strate excursion > 0.5 (gelées AVANT la grille) :
      L_{N−k} (PRIMAIRE, 3-classes) : N−k ≥ 4 ⟹ Δ>0 ; N−k = 3 ⟹ Δ<0 ;
                                       N−k ≤ 2 ⟹ Δ=0.
      L_N (2-classes)   : N ≥ N* ⟹ Δ>0, sinon Δ≤0 ; N* ∈ {8, 9, 10}, retenu =
                          argmin des mal-classées.
      L_k (2-classes)   : k ≥ k* ⟹ Δ>0, sinon Δ≤0 ; k* ajusté par la grille
                          (rivale nulle à battre).
      L_side (jointe)   : Δ>0 ⟺ (k < 2N/3) ∧ (N−k ≥ 4) ; sinon Δ≤0.
  * Portes : 0a (re-jeu order-sensibilité du primaire + contrôle multiset sur
    2 cellules-témoins, haute (13,8) et basse (7,5) excursion, gelées ici) ;
    0b (≥ 5 cellules décisives par paire de lois AVANT toute lecture de signe ;
    pour L_k, exigé pour TOUTE la famille k* ∈ [2, 23] puisque k* n'est ajusté
    qu'après lecture) ; 1 (pivots η=0 exact, sans-flip Δ=0, cellules-témoins
    rationnelles) ; 2 (critère 100 % sur les 228 cellules — L_{N−k} en 3-classes
    exactes P-gate incluse, rivales en 2-classes projetées {+, ≤0}) ; 3 (retour
    aux Δ réels des runners défaut/claude/bloc26, critère 100 %).

FORME FORTE — DÉRIVATION ANALYTIQUE GELÉE AVANT LA GRILLE (mandat pré-enregistré).
La dynamique est exactement calculable en rationnels. Tant que le clip est inactif,
la boucle fermée sur le plateau satisfait (u = e − target) :

    u_{t+1} = (3/2)·u_t − u_{t−1}        (η = 1/2)

dont les valeurs propres sont sur le CERCLE UNITÉ (cos θ = 3/4, |λ| = 1) : l'organe
à η = 0.5 est un oscillateur MARGINAL non amorti autour de la cible. La présence
d'un e_t dans la bande est donc une condition ARITHMÉTIQUE sur (N, k) (échantillon-
nage quasi-périodique), pas une loi monotone — c'est le mécanisme qui rend les
quatre lois candidates fausses sur la grille large. Carte signe(Δ_nom)(N, k)
DÉRIVÉE par la récurrence Fraction exacte et GELÉE ICI AVANT l'exécution de la
grille float (lignes N, colonnes k = 2..N−2 ; '.' = strate basse, Δ = 0) :

    N= 6 : -0.                    N=16 : 0----0+++.+++
    N= 7 : 0--.                   N=17 : 0000-++++.++++
    N= 8 : --+.0                  N=18 : --00-+++++.++++
    N= 9 : 0--+.+                 N=19 : 00---0+++++.++++
    N=10 : 0--++.+                N=20 : -0000-+++++.+++++
    N=11 : 00-0+.++               N=21 : 0--00-++++++.+++++
    N=12 : 00-0++.++              N=22 : 000---0++++++.+++++
    N=13 : ----0++.++             N=23 : --0000-++++++.++++++
    N=14 : 000-0++.+++            N=24 : 00--00-+++++++.++++++
    N=15 : -00-++++.+++

VERDICT MESURÉ (2026-07-11, gravé — jamais forcé ; l'ingénieur statue) :

  * Concordance dérivation ↔ instrument float gelé : SIGNES 228/228 identiques ;
    2 cellules divergent en MAGNITUDE seulement — (9,7) : float 1/3 vs exact 2/9 ;
    (15,11) : float 2/5 vs exact 1/3 — toutes deux avec un point EXACTEMENT sur le
    bord de bande |e−1| = 1/2 (52 cellules ont un tel point ; la quantisation float
    n'a d'effet de comptage que sur ces 2). Le signe, seule quantité décisionnelle,
    est partout celui de la Fraction. Pas d'arrêt technique.
  * Porte 0a : PASSE. (13,8) primaire gap = 2.186e-1, (7,5) gap = 8.095e-2, tous
    deux ≥ δ_min = 1e-4 ; multiset vacuous (gap = 0.0) sur les deux.
  * Porte 0b : PASSE. Décisives L_{N−k}↔L_N : 34/36/39 (N* = 8/9/10) ;
    L_{N−k}↔L_k : min 33 (k* = 2), max 180, sur k* ∈ [2,23] ;
    L_{N−k}↔L_side : 26. Tout ≥ 5.
  * Porte 1 : PASSE. η=0 ≡ g-fixe exact ; sans-flip Δ = 0 ; les 6 cellules-témoins
    rationnelles reproduites EXACTEMENT ((7,4)→−2/7, (8,6)→0, (10,6)→+1/5,
    (13,8)→+3/13, (16,10)→+1/4, (19,12)→+4/19), en Fraction ET au float ≤ 1e-12.
  * P-gate : TIENT 19/19 (0 violation — toutes les cellules basses à Δ = 0 exact).
  * Porte 2 : ÉCHEC DE TOUTES LES LOIS (critère 100 %) — c'est LE résultat :
      L_{N−k} 3-classes : 119/228 mal classées (109/228 = 47.8 % correct) — RÉFUTÉE ;
      L_N (N* retenu = 10 ; {8: 86, 9: 84, 10: 83}) : 83/228 mal classées ;
      L_k (k* retenu = 7, ex æquo avec 8, argmin) : 17/228 mal classées —
        meilleure rivale (92.5 %), mais pas 100 % ;
      L_side : 144/228 mal classées.
    Le paysage réel (carte gelée ci-dessus) : côté TARDIF (k > 2N/3) 56 Δ>0 et
    1 Δ=0 (le contre-exemple (8,6) gravé au T26 est le SEUL zéro tardif) ; côté
    PRÉCOCE : mélange arithmétique {−: 43, 0: 47, +: 62} sans loi monotone en
    N, k ou N−k — cohérent avec l'oscillateur marginal (|λ| = 1) dérivé ci-dessus.
    L'« escalier de N » du T26 était un fait de la droite |3k−2N| = 2 flip-précoce :
    la grille qui dissocie les variables le dissout. Issue honnête n° 3 de
    l'émission : la CARTE EXACTE est le livrable.
  * B_sweep_global : g = 1.0 (le nominal GAGNE le sweep sur la grille — aucun
    décalage à rapporter) ; Δ médian vs sweep = +0.0893 (info).
  * Porte 3 (INFO — aucune loi survivante, le verdict est tombé porte 2) : la
    CARTE exacte, elle, classe 100 % des Δ réels, en SIGNE ET en VALEUR
    (best_fixed sweep = 1.0 sur les TROIS runs ⟹ mêmes chemins float, écart
    ≤ 1e-12) : T24-défaut 40/40, T25-claude 76/76, T26-bloc26 2000/2000.
    La loi primaire L_{N−k} (info) : T24 40/40, T26 2000/2000 (leurs cellules
    vivent sur la droite compatible), MAIS T25-claude 75/76 — cycle 27
    (N=11, k=8, flip TARDIF, N−k=3, prédit −, observé +) : le réel contenait
    déjà une cellule qui réfute la loi et confirme la carte ((11,8) = '+').

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/``, NI ``regulate_step``,
NI ``structural_gap.py`` (gelé), NI ``structural_regulation.py`` (les runs
T24/T25/T26 restent byte-identiques). Anti-circularité : profils construits
depuis (N, k) directement ; jamais les dims 0-5 du 33D ; ``op_from_vector``
interdit. Tout est déterministe (shuffles seedés via l'infra T23).
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

from ..experimental.structural_gap import (
    ETA_STRUCT,
    G0_STRUCT,
    OrientedToken,
    STRUCT_GAIN_SWEEP,
    f_edge_struct,
    obs_struct_frozen,
    obs_struct_multiset,
    reconstruct_fixed,
    reconstruct_profile,
    shuffle_tokens,
)
from .instrument_validation import OrderSensitivityReport, assert_order_sensitive
from .structural_regulation import (
    best_fixed_gain,
    collect_profiles,
    pivot_eta0_is_exact,
    pivot_noflip_delta,
    run_structural_regulation,
)
from .edge_maintenance import _median


# --- constantes GELÉES A PRIORI (émission T27 — jamais ajustées après mesure) ------

GRID_N_MIN = 6
GRID_N_MAX = 24
GRID_K_MIN = 2                    # k ∈ [2, N−2] : les DEUX côtés du flip
N_STAR_CANDIDATES = (8, 9, 10)    # famille gelée de L_N
K_STAR_FAMILY = tuple(range(2, 24))   # famille de seuils balayée par L_k (rivale nulle)
RECOVERY_MIN_POSITIVE = 4         # L_{N−k} : N−k ≥ 4 ⟹ Δ>0
RECOVERY_NEGATIVE = 3             # L_{N−k} : N−k = 3 ⟹ Δ<0 (N−k ≤ 2 ⟹ Δ=0)
MIN_DECISIVE_CELLS = 5            # porte 0b : plancher de cellules décisives par paire
GATE0A_CELL_HIGH = (13, 8)        # cellule-témoin haute excursion (2/3, aussi témoin rationnel)
GATE0A_CELL_LOW = (7, 5)          # cellule-témoin basse excursion (1/3, ≠ 0 : primaire non dégénéré)

# cellules-témoins rationnelles (porte 1) — valeurs T26 vérifiées en Fraction, gravées
WITNESS_CELLS: Dict[Tuple[int, int], Fraction] = {
    (7, 4): Fraction(-2, 7),
    (8, 6): Fraction(0),          # le contre-exemple flip-tardif gravé au T26
    (10, 6): Fraction(1, 5),
    (13, 8): Fraction(3, 13),
    (16, 10): Fraction(1, 4),
    (19, 12): Fraction(4, 19),
}

# --- moteur rationnel EXACT (indépendant du chemin float de l'instrument) ----------
#
# Réimplémente la récurrence depuis la SPEC a priori de structural_gap.py (choix
# 3-7 de son en-tête), en Fraction de bout en bout : le signe de Δ_nom est exact.
# Prototype : les vérifications rationnelles de l'ingénieur T26
# (test_structural_regulation.py). Constantes = celles de l'instrument, en rationnels.

TARGET_EXACT = Fraction(1)        # TARGET_LEAD = 1.0
BAND_EXACT = Fraction(1, 2)       # BAND_LEAD = 0.5
ETA_EXACT = Fraction(1, 2)        # ETA_STRUCT = 0.5
G_MIN_EXACT = Fraction(1, 2)      # G_MIN_STRUCT = 0.5
G_MAX_EXACT = Fraction(2)         # G_MAX_STRUCT = 2.0
PHI_STAR_EXACT = Fraction(2, 3)   # PHI_STAR = 2/3


def grid_cells() -> List[Tuple[int, int]]:
    """Les 228 cellules gelées : N ∈ [6, 24], k ∈ [2, N−2] (ordre déterministe)."""
    return [(n, k) for n in range(GRID_N_MIN, GRID_N_MAX + 1)
            for k in range(GRID_K_MIN, n - 1)]


def profile_of(n: int, k: int) -> List[int]:
    """Profil d'orientation canonique de la cellule (N, k) : [+1]*k + [−1]*(N−k)."""
    if not (1 <= k <= n - 1):
        raise ValueError("cellule invalide : il faut 1 <= k <= N-1 (un flip existe)")
    return [+1] * k + [-1] * (n - k)


def oriented_tokens_of(n: int, k: int) -> List[OrientedToken]:
    """Le même profil en ``OrientedToken`` synthétiques (textes t0..tN−1, porte 0a)."""
    return [OrientedToken(text=f"t{i}", orientation=o)
            for i, o in enumerate(profile_of(n, k))]


def excursion_x3(n: int, k: int) -> int:
    """``3·excursion = |3k − 2N|`` — entier EXACT (excursion = |k − φ*N| = ce/3).

    Strate haute ⟺ excursion > band = 1/2 ⟺ |3k−2N| ≥ 2 (jamais égal à 3/2 :
    le seuil ne coupe aucune cellule entière — la P-gate est nette).
    """
    return abs(3 * k - 2 * n)


def is_high_excursion(n: int, k: int) -> bool:
    """Strate haute de la cellule (excursion > 0.5), en arithmétique entière exacte."""
    return excursion_x3(n, k) >= 2


def flip_side(n: int, k: int) -> str:
    """Côté du flip : 'précoce' (k < φ*N), 'tardif' (k > φ*N), 'exact' (k = φ*N)."""
    d = 3 * k - 2 * n
    return "précoce" if d < 0 else ("tardif" if d > 0 else "exact")


def trace_e_exact(n: int, k: int, eta: Fraction) -> List[Fraction]:
    """Trace EXACTE de l'écart de phase e_t (Fractions), organe (η) ou fixe (η=0).

    Récurrence identique à ``reconstruct_profile`` (spec gelée) : e_t = p_read−p_ref ;
    g ← clip(g − η(e_t − 1), 1/2, 2) ; p_read += g ; p_ref += inc_t, avec
    inc(+1) = φ*N/k et inc(−1) = (1−φ*)N/(N−k). À η = 0, g reste 1 : c'est le
    lecteur fixe nominal. Aucun float : le signe de Δ_nom en sort exact.
    """
    inc_plus = PHI_STAR_EXACT * n / k
    inc_minus = (1 - PHI_STAR_EXACT) * n / (n - k)
    p_read, p_ref, g = Fraction(1), Fraction(0), Fraction(1)
    e: List[Fraction] = [p_read - p_ref]
    for t in range(n):
        e_t = p_read - p_ref
        g = g - eta * (e_t - TARGET_EXACT)
        g = min(G_MAX_EXACT, max(G_MIN_EXACT, g))
        p_read += g
        p_ref += inc_plus if t < k else inc_minus
        e.append(p_read - p_ref)
    return e


def f_edge_exact(e: Sequence[Fraction]) -> Fraction:
    """f_edge EXACT : #{t = 1..N : |e_t − 1| ≤ 1/2} / N (comparaisons rationnelles)."""
    n = len(e) - 1
    hits = sum(1 for x in e[1:] if abs(x - TARGET_EXACT) <= BAND_EXACT)
    return Fraction(hits, n)


def delta_nom_exact(n: int, k: int) -> Fraction:
    """Δ_nom EXACT de la cellule : f_edge(organe η=1/2) − f_edge(fixe g=1)."""
    return f_edge_exact(trace_e_exact(n, k, ETA_EXACT)) - f_edge_exact(
        trace_e_exact(n, k, Fraction(0)))


def band_edge_touches(n: int, k: int) -> int:
    """Points EXACTEMENT sur le bord de bande |e−1| = 1/2 (organe + fixe).

    Là où le comptage float de l'instrument peut différer d'un hit du comptage
    exact (cas limite documenté ; le SIGNE reste jugé sur la Fraction).
    """
    c = 0
    for eta in (ETA_EXACT, Fraction(0)):
        c += sum(1 for x in trace_e_exact(n, k, eta)[1:]
                 if abs(x - TARGET_EXACT) == BAND_EXACT)
    return c


def delta_nom_float(n: int, k: int) -> float:
    """Δ_nom mesuré par l'INSTRUMENT FLOAT GELÉ (baseline B_nominal g = 1).

    Chemin de code de ``structural_gap.py`` byte-à-byte — sert de contre-épreuve
    de la dérivation exacte (concordance de SIGNE exigée cellule par cellule).
    """
    orients = profile_of(n, k)
    fe_o = f_edge_struct(reconstruct_profile(orients, eta=ETA_STRUCT, g0=G0_STRUCT))
    fe_f = f_edge_struct(reconstruct_fixed(orients, g_fixed=G0_STRUCT))
    return fe_o - fe_f


def _sign(x: Fraction) -> int:
    return (x > 0) - (x < 0)


# --- les QUATRE lois gelées (prédiction pure sur (N, k), aucune mesure) ------------

def predict_primary(n: int, k: int) -> int:
    """L_{N−k}, 3-classes, P-gate incluse : renvoie le signe prédit ∈ {+1, −1, 0}."""
    if not is_high_excursion(n, k):
        return 0                              # P-gate commune : rien à réguler
    leg = n - k
    if leg >= RECOVERY_MIN_POSITIVE:
        return +1
    if leg == RECOVERY_NEGATIVE:
        return -1
    return 0


def predict_primary_positive(n: int, k: int) -> bool:
    """L_{N−k} PROJETÉE 2-classes {+, ≤0} (comparabilité avec les rivales, porte 0b)."""
    return predict_primary(n, k) > 0


def predict_l_n(n: int, k: int, n_star: int) -> bool:
    """L_N 2-classes : Δ>0 prédit ⟺ strate haute ∧ N ≥ N*."""
    return is_high_excursion(n, k) and n >= n_star


def predict_l_k(n: int, k: int, k_star: int) -> bool:
    """L_k 2-classes : Δ>0 prédit ⟺ strate haute ∧ k ≥ k* (rivale nulle à battre)."""
    return is_high_excursion(n, k) and k >= k_star


def predict_l_side(n: int, k: int) -> bool:
    """L_side jointe : Δ>0 prédit ⟺ strate haute ∧ (k < 2N/3) ∧ (N−k ≥ 4)."""
    return is_high_excursion(n, k) and (3 * k - 2 * n) < 0 and (n - k) >= RECOVERY_MIN_POSITIVE


# --- PORTE 0a : re-jeu de la pré-validation d'instrument (T23) sur 2 cellules ------

@dataclass(frozen=True)
class Gate0aReport:
    """Order-sensibilité du primaire + contrôle multiset sur les 2 cellules gelées."""

    cell_high: Tuple[int, int]
    primary_high: OrderSensitivityReport
    multiset_high: OrderSensitivityReport
    cell_low: Tuple[int, int]
    primary_low: OrderSensitivityReport
    multiset_low: OrderSensitivityReport
    passes: bool                  # primaire sensible ET multiset vacuous, sur les 2


def gate0a() -> Gate0aReport:
    """Porte 0a : l'instrument bouge sous shuffle là où la grille sera jugée."""
    reports = {}
    for label, cell in (("high", GATE0A_CELL_HIGH), ("low", GATE0A_CELL_LOW)):
        toks = oriented_tokens_of(*cell)
        reports[label] = (
            assert_order_sensitive(obs_struct_frozen, toks, shuffle_fn=shuffle_tokens),
            assert_order_sensitive(obs_struct_multiset, toks, shuffle_fn=shuffle_tokens),
        )
    ph, mh = reports["high"]
    pl, ml = reports["low"]
    return Gate0aReport(
        cell_high=GATE0A_CELL_HIGH, primary_high=ph, multiset_high=mh,
        cell_low=GATE0A_CELL_LOW, primary_low=pl, multiset_low=ml,
        passes=(ph.is_order_sensitive and pl.is_order_sensitive
                and mh.is_vacuous and ml.is_vacuous),
    )


# --- PORTE 0b : cellules décisives par paire de lois (AVANT toute lecture de signe) -

@dataclass(frozen=True)
class Gate0bReport:
    """Comptes de cellules où les PRÉDICTIONS divergent (aucun signe mesuré requis).

    ``lk_counts`` couvre TOUTE la famille k* ∈ [2, 23] (k* n'est ajusté qu'après
    lecture des signes : la garantie porte sur la famille entière).
    """

    ln_counts: Dict[int, int]         # N* → # cellules décisives L_{N−k} ↔ L_N
    lk_counts: Dict[int, int]         # k* → # cellules décisives L_{N−k} ↔ L_k
    lk_min: int
    lside_count: int
    passes: bool


def gate0b() -> Gate0bReport:
    """Porte 0b : chaque paire de lois a ≥ 5 cellules d'arbitrage sur la grille."""
    cells = grid_cells()
    ln_counts = {
        ns: sum(1 for (n, k) in cells
                if predict_primary_positive(n, k) != predict_l_n(n, k, ns))
        for ns in N_STAR_CANDIDATES
    }
    lk_counts = {
        ks: sum(1 for (n, k) in cells
                if predict_primary_positive(n, k) != predict_l_k(n, k, ks))
        for ks in K_STAR_FAMILY
    }
    lside_count = sum(1 for (n, k) in cells
                      if predict_primary_positive(n, k) != predict_l_side(n, k))
    lk_min = min(lk_counts.values())
    passes = (
        all(c >= MIN_DECISIVE_CELLS for c in ln_counts.values())
        and lk_min >= MIN_DECISIVE_CELLS
        and lside_count >= MIN_DECISIVE_CELLS
    )
    return Gate0bReport(ln_counts=ln_counts, lk_counts=lk_counts,
                        lk_min=lk_min, lside_count=lside_count, passes=passes)


# --- PORTE 1 : pivots + cellules-témoins rationnelles ------------------------------

@dataclass(frozen=True)
class Gate1Report:
    """Pivots (η=0 exact, sans-flip Δ=0) + les 6 témoins rationnels EXACTS."""

    pivot_eta0_exact: bool
    pivot_noflip: float               # attendu 0.0
    witness_exact: Dict[Tuple[int, int], bool]      # Fraction == valeur gravée
    witness_float_ok: Dict[Tuple[int, int], bool]   # |float − Fraction| ≤ 1e-12
    passes: bool


def gate1() -> Gate1Report:
    """Porte 1 : l'instrument et le moteur exact reproduisent les pivots/témoins."""
    piv_eta0 = pivot_eta0_is_exact(profile_of(*GATE0A_CELL_HIGH))
    piv_noflip = pivot_noflip_delta()
    w_exact = {cell: delta_nom_exact(*cell) == expected
               for cell, expected in WITNESS_CELLS.items()}
    w_float = {cell: abs(delta_nom_float(*cell) - float(WITNESS_CELLS[cell])) <= 1e-12
               for cell in WITNESS_CELLS}
    return Gate1Report(
        pivot_eta0_exact=piv_eta0,
        pivot_noflip=piv_noflip,
        witness_exact=w_exact,
        witness_float_ok=w_float,
        passes=(piv_eta0 and piv_noflip == 0.0
                and all(w_exact.values()) and all(w_float.values())),
    )


# --- PORTE 2 : la grille — carte exacte, concordance float, classement des lois ----

@dataclass(frozen=True)
class CellRecord:
    """Une cellule (N, k) : Δ_nom exact (autorité du signe), float, descripteurs."""

    n: int
    k: int
    excursion_x3: int                 # |3k − 2N| (entier ; excursion = /3)
    side: str                         # 'précoce' | 'tardif' | 'exact'
    delta_exact: Fraction
    delta_float: float
    sign: int                         # signe EXACT ∈ {−1, 0, +1}
    band_touches: int                 # points |e−1| = 1/2 exacts (cas limite float)


@dataclass(frozen=True)
class Misclassified:
    """Une cellule mal classée par une loi : localisation + prédit/observé."""

    n: int
    k: int
    side: str
    predicted: int                    # 3-classes : {−1, 0, +1} ; 2-classes : {+1, 0}
    actual: int
    delta_exact: Fraction


@dataclass(frozen=True)
class GridReport:
    """Porte 2 complète : P-gate, concordance dérivation↔float, classement des 4 lois."""

    cells: List[CellRecord]
    n_cells: int
    n_low: int
    n_high: int
    pgate_violations: List[CellRecord]        # cellules basses à Δ ≠ 0 (attendu : 0)
    sign_mismatches: List[CellRecord]         # signe float ≠ signe exact (attendu : 0)
    magnitude_divergences: List[CellRecord]   # |float − exact| > 1e-9 (bord de bande)
    mis_primary: List[Misclassified]          # L_{N−k} 3-classes, P-gate incluse
    n_star_retained: int
    mis_l_n: Dict[int, List[Misclassified]]   # par N* candidat
    k_star_retained: int
    k_star_ties: List[int]                    # k* ex æquo à l'argmin (rapportés)
    mis_l_k_retained: List[Misclassified]
    lk_counts: Dict[int, int]                 # # mal-classées par k* (famille entière)
    mis_l_side: List[Misclassified]
    sweep_global_gain: float                  # B_sweep_global (rapportée à part)
    delta_vs_sweep_median: float              # info : Δ médian vs B_sweep_global
    any_law_100: bool


def _mis3(cells: Sequence[CellRecord]) -> List[Misclassified]:
    out = []
    for c in cells:
        pred = predict_primary(c.n, c.k)
        if pred != c.sign:
            out.append(Misclassified(n=c.n, k=c.k, side=c.side, predicted=pred,
                                     actual=c.sign, delta_exact=c.delta_exact))
    return out


def _mis2(cells: Sequence[CellRecord], pred_fn) -> List[Misclassified]:
    out = []
    for c in cells:
        pred = 1 if pred_fn(c.n, c.k) else 0
        actual = 1 if c.sign > 0 else 0       # projection {+, ≤0}
        if pred != actual:
            out.append(Misclassified(n=c.n, k=c.k, side=c.side, predicted=pred,
                                     actual=actual, delta_exact=c.delta_exact))
    return out


def grid_report() -> GridReport:
    """Mesure la grille entière (exact + float), classe les 4 lois, critère 100 %."""
    records: List[CellRecord] = []
    for (n, k) in grid_cells():
        d_exact = delta_nom_exact(n, k)
        d_float = delta_nom_float(n, k)
        records.append(CellRecord(
            n=n, k=k, excursion_x3=excursion_x3(n, k), side=flip_side(n, k),
            delta_exact=d_exact, delta_float=d_float, sign=_sign(d_exact),
            band_touches=band_edge_touches(n, k),
        ))

    low = [c for c in records if c.excursion_x3 < 2]
    high = [c for c in records if c.excursion_x3 >= 2]
    pgate_viol = [c for c in low if c.delta_exact != 0]
    sign_mm = [c for c in records
               if ((c.delta_float > 0) - (c.delta_float < 0)) != c.sign]
    mag_div = [c for c in records if abs(c.delta_float - float(c.delta_exact)) > 1e-9]

    mis_primary = _mis3(records)
    mis_l_n = {ns: _mis2(records, lambda n, k, ns=ns: predict_l_n(n, k, ns))
               for ns in N_STAR_CANDIDATES}
    n_star = min(N_STAR_CANDIDATES, key=lambda ns: (len(mis_l_n[ns]), ns))
    lk_counts = {ks: len(_mis2(records, lambda n, k, ks=ks: predict_l_k(n, k, ks)))
                 for ks in K_STAR_FAMILY}
    best_count = min(lk_counts.values())
    ties = [ks for ks in K_STAR_FAMILY if lk_counts[ks] == best_count]
    k_star = ties[0]                          # ex æquo → plus petit k* (règle déterministe)
    mis_l_k = _mis2(records, lambda n, k: predict_l_k(n, k, k_star))
    mis_l_side = _mis2(records, predict_l_side)

    # B_sweep_global (baseline SECONDAIRE, rapportée à part — jamais le verdict)
    profiles = [profile_of(n, k) for (n, k) in grid_cells()]
    g_sweep = best_fixed_gain(profiles, fixed_gains=STRUCT_GAIN_SWEEP)
    d_sweep = []
    for orients in profiles:
        fe_o = f_edge_struct(reconstruct_profile(orients, eta=ETA_STRUCT, g0=G0_STRUCT))
        fe_s = f_edge_struct(reconstruct_fixed(orients, g_fixed=g_sweep))
        d_sweep.append(fe_o - fe_s)

    any100 = (len(mis_primary) == 0 or len(mis_l_n[n_star]) == 0
              or len(mis_l_k) == 0 or len(mis_l_side) == 0)

    return GridReport(
        cells=records, n_cells=len(records), n_low=len(low), n_high=len(high),
        pgate_violations=pgate_viol, sign_mismatches=sign_mm,
        magnitude_divergences=mag_div,
        mis_primary=mis_primary,
        n_star_retained=n_star, mis_l_n=mis_l_n,
        k_star_retained=k_star, k_star_ties=ties, mis_l_k_retained=mis_l_k,
        lk_counts=lk_counts,
        mis_l_side=mis_l_side,
        sweep_global_gain=g_sweep, delta_vs_sweep_median=_median(d_sweep),
        any_law_100=any100,
    )


# --- PORTE 3 : retour aux Δ réels DÉJÀ mesurés (runners T24/T25/T26) ---------------

@dataclass(frozen=True)
class RealReturnReport:
    """Classement des Δ réels d'un runner par la loi primaire ET par la carte exacte.

    ``delta_real`` vient du rapport du runner (vs best_fixed de SON sweep, gelé) ;
    la carte est à baseline NOMINALE g = 1. Quand ``best_fixed == 1.0`` les deux
    chemins float coïncident : la concordance en VALEUR est alors exigible ; sinon
    seule la concordance en SIGNE avec la carte est rapportée (information, jamais
    lissée). Aucun paramètre neuf : (N, k) sortent des profils du runner.
    """

    label: str
    n_cycles: int
    best_fixed: float
    n_sign_match_map: int                    # signe(Δ_réel) == signe exact carte
    sign_mismatches_map: List[Tuple[int, int, int, float, str]]   # (i, N, k, Δ, exact)
    n_value_match_map: int                   # |Δ_réel − Δ_nom_float| ≤ 1e-12
    n_correct_primary: int                   # classes L_{N−k} 3-classes (info)
    mis_primary: List[Tuple[int, int, int, int, int]]  # (i, N, k, prédit, observé)


def real_return(path: str, *, n_cycles: int, label: str,
                line_range: Optional[Tuple[int, int]] = None) -> RealReturnReport:
    """Re-joue un runner T24/T25/T26 tel quel et classe ses Δ par cycle (porte 3)."""
    r = run_structural_regulation(path, n_cycles=n_cycles, line_range=line_range)
    profiles = collect_profiles(path, n_cycles=n_cycles, line_range=line_range)
    sign_mm: List[Tuple[int, int, int, float, str]] = []
    mis_prim: List[Tuple[int, int, int, int, int]] = []
    n_sign = n_value = n_prim = 0
    for i, (prof, d_real) in enumerate(zip(profiles, r.delta_real)):
        n = len(prof)
        k = sum(1 for tk in prof if tk.orientation == +1)
        d_map_exact = delta_nom_exact(n, k)
        s_real = (d_real > 0) - (d_real < 0)
        if s_real == _sign(d_map_exact):
            n_sign += 1
        else:
            sign_mm.append((i, n, k, d_real, str(d_map_exact)))
        if abs(d_real - delta_nom_float(n, k)) <= 1e-12:
            n_value += 1
        pred = predict_primary(n, k)
        if pred == s_real:
            n_prim += 1
        else:
            mis_prim.append((i, n, k, pred, s_real))
    return RealReturnReport(
        label=label, n_cycles=r.n_cycles, best_fixed=r.best_fixed,
        n_sign_match_map=n_sign, sign_mismatches_map=sign_mm,
        n_value_match_map=n_value,
        n_correct_primary=n_prim, mis_primary=mis_prim,
    )


# --- runner de mesure ----------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    from pathlib import Path

    def _root() -> Path:
        p = Path("F:/code/claude/spiraton-enhanced")
        return p if (p / "dataset_aba.txt").is_file() else Path(__file__).resolve().parents[3]

    print("=== PORTE 0a (2 cellules-témoins, δ_min=1e-4, n_shuffle=8) ===")
    g0a = gate0a()
    for lbl, cell, pr, mr in (("HAUTE", g0a.cell_high, g0a.primary_high, g0a.multiset_high),
                              ("BASSE", g0a.cell_low, g0a.primary_low, g0a.multiset_low)):
        print(f"{lbl} {cell} exc={excursion_x3(*cell)}/3 : "
              f"PRIMAIRE gap={pr.gap:.6e} sensible={pr.is_order_sensitive} | "
              f"MULTISET gap={mr.gap:.6e} vacuous={mr.is_vacuous}")
    print(f"porte 0a : {'PASSE' if g0a.passes else 'ECHEC'}")

    print("=== PORTE 0b (cellules décisives par paire, plancher 5) ===")
    g0b = gate0b()
    print(f"L_(N-k)<->L_N   : {g0b.ln_counts}")
    print(f"L_(N-k)<->L_k   : min={g0b.lk_min} sur k* in [2,23] "
          f"(détail : {g0b.lk_counts})")
    print(f"L_(N-k)<->L_side: {g0b.lside_count}")
    print(f"porte 0b : {'PASSE' if g0b.passes else 'ECHEC'}")

    print("=== PORTE 1 (pivots + 6 témoins rationnels) ===")
    g1 = gate1()
    print(f"pivot η=0 exact : {g1.pivot_eta0_exact} | pivot sans-flip Δ : {g1.pivot_noflip}")
    for cell, ok in g1.witness_exact.items():
        print(f"témoin {cell} → {WITNESS_CELLS[cell]} : exact={ok} "
              f"float={g1.witness_float_ok[cell]}")
    print(f"porte 1 : {'PASSE' if g1.passes else 'ECHEC'}")

    print("=== PORTE 2 (grille 228 cellules, critère 100 %) ===")
    gr = grid_report()
    print(f"cellules : {gr.n_cells} (basses {gr.n_low} / hautes {gr.n_high})")
    print(f"P-gate violations           : {len(gr.pgate_violations)}")
    print(f"signe float ≠ signe exact   : {len(gr.sign_mismatches)}")
    print(f"divergences magnitude >1e-9 : {len(gr.magnitude_divergences)} "
          f"{[(c.n, c.k, c.delta_float, str(c.delta_exact), c.band_touches) for c in gr.magnitude_divergences]}")
    print(f"L_(N-k) 3-classes : {len(gr.mis_primary)}/228 mal classées")
    for m in gr.mis_primary[:10]:
        print(f"  N={m.n} k={m.k} ({m.side}) prédit={m.predicted:+d} observé={m.actual:+d} Δ={m.delta_exact}")
    if len(gr.mis_primary) > 10:
        print(f"  ... (+{len(gr.mis_primary) - 10})")
    print(f"L_N : N* retenu = {gr.n_star_retained} ; mal classées par N* : "
          f"{ {ns: len(v) for ns, v in gr.mis_l_n.items()} }")
    print(f"L_k : k* retenu = {gr.k_star_retained} (ex æquo {gr.k_star_ties}) ; "
          f"mal classées = {len(gr.mis_l_k_retained)}/228")
    print(f"L_side : {len(gr.mis_l_side)}/228 mal classées")
    print(f"B_sweep_global : g = {gr.sweep_global_gain} "
          f"(décalage vs nominal : {'AUCUN' if gr.sweep_global_gain == 1.0 else 'OUI — rapporté, non corrigé'}) ; "
          f"Δ médian vs sweep = {gr.delta_vs_sweep_median:+.4f}")
    print(f"une loi à 100 % ? {gr.any_law_100}")
    print("--- carte des signes (exacte ; '.' = strate basse) ---")
    by_cell = {(c.n, c.k): c for c in gr.cells}
    for n in range(GRID_N_MIN, GRID_N_MAX + 1):
        row = "".join(
            ("." if by_cell[(n, k)].excursion_x3 < 2 else
             {1: "+", -1: "-", 0: "0"}[by_cell[(n, k)].sign])
            for k in range(GRID_K_MIN, n - 1))
        print(f"N={n:2d} : {row}")

    print("=== PORTE 3 (retour aux Δ réels — runners re-joués tels quels) ===")
    root = _root()
    runs = [
        (str(root / "dataset_aba.txt"), 40, None, "T24-defaut"),
        (str(root / "corpus_claude_aba.txt"), 76, None, "T25-claude"),
        (str(root / "dataset_aba.txt"), 2000, (1001, 3000), "T26-bloc26"),
    ]
    for path, n_run, lr, label in runs:
        rr = real_return(path, n_cycles=n_run, line_range=lr, label=label)
        print(f"{label} : n={rr.n_cycles} best_fixed(sweep)={rr.best_fixed}")
        print(f"  carte exacte : signes {rr.n_sign_match_map}/{rr.n_cycles} ; "
              f"valeurs exactes (vs Δ_nom float) {rr.n_value_match_map}/{rr.n_cycles}")
        for mm in rr.sign_mismatches_map[:10]:
            print(f"    divergence signe : cycle {mm[0]} N={mm[1]} k={mm[2]} Δ={mm[3]:+.4f} carte={mm[4]}")
        print(f"  L_(N-k) 3-classes (info) : {rr.n_correct_primary}/{rr.n_cycles} bien classés")
        for mp in rr.mis_primary[:6]:
            print(f"    mal classé : cycle {mp[0]} N={mp[1]} k={mp[2]} prédit={mp[3]:+d} observé={mp[4]:+d}")
        if len(rr.mis_primary) > 6:
            print(f"    ... (+{len(rr.mis_primary) - 6})")
