from __future__ import annotations

"""Contrôleur du « bord du chaos » — un gain AUTO-RÉGULÉ sur l'oscilloscope (Tour 15).

PORTEUR. On reprend l'oscilloscope des Tours 4-6 (``Oscilloscope2D``, positif franc)
et on lui ajoute UNE capacité neuve : au lieu d'un gain radial ``g`` FIXE, un gain
``g_t`` qui SE RÉGULE en lisant la trace en cours. La « température » du système est
le rayon spectral effectif ``ρ`` (= gain par tour) ; le « bord du chaos » est ``ρ≈1``
(orbite marginale : ni effondrement vers 0, ni divergence). Les trois régimes sont
nos trois verdicts :

    ρ < 1  →  r_t → 0     →  RÉPÉTITION   (cycle sans mise à jour, l.867)
    ρ > 1  →  r_t → ∞     →  DISSIPATION  (cycle ouvert/incohérent, l.826)
    ρ ≈ 1  →  r borné, cos→1  →  PROGRESSION (proche-aligné non identique, l.290)

ON NE MESURE PAS LA CONSCIENCE. On mesure une propriété dynamique OBSERVABLE : le gain
``g_t`` suit-il le signe de ``(ρ̂_t − 1)`` ? — c'est-à-dire le système SE RÉGULE-t-il
vers son bord ? Aucune prétention de cognition (l.300 reste une métaphore non testée
ici).

ÉQUATION (par pas). L'oscilloscope est PASSIF ; le contrôleur PILOTE le gain :

    ρ̂_t      = r_t / r_{t-1}                         (estimation en ligne du gain réalisé)
    g_t      = clip( g_{t-1} − η·(ρ̂_t − 1), g_min, g_max )   (loi proportionnelle bornée)
    A_t      = g_t · g_drift(t) · R(ω)                (transition effective du pas t)
    s_{t+1}  = A_t · s_t   (+ W_in·u_t si signal)

où ``r_t = ‖s_t‖``. NOTE D'HONNÊTETÉ STRUCTURELLE (le cœur du test) : le contrôleur ne
voit JAMAIS ``g_drift(t)`` — il ne lit QUE le rapport de rayons ``r_t/r_{t-1}`` (=
``ρ̂_t`` = le gain RÉALISÉ, dérive comprise) et corrige ``g_t`` pour ramener ``ρ̂_t``
vers 1. Un ``g`` FIXE ne peut PAS suivre une cible mobile ``g_drift(t)`` par
construction : c'est le seul régime où « régule vraiment » se sépare franchement de
« meilleur g moyen » (issue (a) vs issue (b) de l'émission H15).

PERTURBATION DÉCLARÉE A PRIORI (option 3 = dérive de gain natif), FIXÉE AVANT TOUTE
MESURE, jamais réglée pour favoriser le contrôleur (PIÈGE ANTI-TRIVIAL, REFUS) :

    g_drift(t) = g_drift_start + (g_drift_end − g_drift_start) · t/(T−1)

avec ``g_drift_start = 0.95``, ``g_drift_end = 1.10`` (rampe lente sur tout
l'horizon). Sans cette perturbation, un oscilloscope linéaire à g=1 reste un cercle
parfait et le contrôleur est inutile (issue (b) garantie par construction) : c'est
pourquoi la dérive est la perturbation principale et ses bornes sont posées ici, en
DUR, indépendamment du résultat.

CONTRÔLE DE COHÉRENCE DUR (parallèle CTRL D=L Tour 1 / commutateur=0 Tour 6) : à
``η = 0``, la loi de mise à jour est inerte (``g_t = g_0`` pour tout t) ⇒ le contrôleur
REPRODUIT BIT-À-BIT l'oscilloscope à ``g`` fixe ``g_0`` SOUS LA MÊME dérive. Différent
⇒ bug, pas résultat (testé).

STRICTEMENT EXPÉRIMENTAL. Ne touche NI le canon ``core/`` NI le défaut de
``oscilloscope.py`` (le ``g`` fixe reste la baseline). Réutilise ``rotation_matrix`` et
la convention de pas de ``Oscilloscope2D`` (lecture seule).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from .oscilloscope import InputSignal, rotation_matrix


# --- perturbation : dérive de gain natif (déclarée a priori) ------------------

@dataclass(frozen=True)
class GainDrift:
    """Dérive de gain natif ``g_drift(t)`` imposée SOUS le contrôleur (perturbation 3).

    Rampe linéaire ``start → end`` sur l'horizon ``T`` pas. C'est une cible MOBILE
    qu'un ``g`` fixe ne peut suivre par construction (l.605 « revient à l'équilibre
    après perturbation »). Les bornes sont posées A PRIORI (0.95 → 1.10) et ne sont
    JAMAIS réglées sur le résultat.

    start, end : gain natif au premier et au dernier pas.
    Le facteur appliqué au pas ``t`` (0 ≤ t < T) est interpolé linéairement.
    """

    start: float = 0.95
    end: float = 1.10

    def at(self, t: int, T: int) -> float:
        """Facteur de dérive au pas ``t`` sur un horizon de ``T`` pas."""
        if T <= 1:
            return self.start
        frac = t / (T - 1)
        return self.start + (self.end - self.start) * frac


# --- perturbations PAR GRAINE (Tour 16, déclarées a priori) ------------------
#
# H16 (linguiste) : sous une VRAIE variance de population, chaque graine est un
# tirage INDÉPENDANT d'une perturbation NON-ISOTROPE seedée — non plus la même
# dérive partagée pour toutes les graines (artefact d'isotropie du Tour 15). Le
# geste : SUB · dextro · out (creuser un écart entre graines confondues, l.341-342).
#
# Toutes les perturbations exposent le MÊME protocole que ``GainDrift`` :
#   ``.at(t, T) -> float`` (facteur de gain natif appliqué au pas t).
# Ainsi ``EdgeController.run`` et ``run_fixed_gain`` les acceptent sans changement.
#
# PIVOT ANTI-ARTEFACT (le ``η=0`` du T15 transposé à la variance) : à amplitude
# de perturbation NULLE, chaque perturbation T16 DOIT reproduire bit-à-bit la
# dérive T15. Les deux fabriques ``.degenerate()`` ci-dessous incarnent ce pivot.


@dataclass(frozen=True)
class SeededDrift:
    """P1 — dérive NON-STATIONNAIRE à offset ET pente ALÉATOIRES PAR GRAINE.

    Généralise ``GainDrift`` : la rampe ``start → end`` n'est plus fixe mais TIRÉE
    par graine via ``torch.Generator().manual_seed(seed)`` dans des intervalles
    posés A PRIORI (jamais réglés sur le résultat) :

        start ~ U[start_lo, start_hi]   (défaut [0.93, 0.97])
        end   ~ U[end_lo,   end_hi]     (défaut [1.07, 1.13])

    À offset/pente FIXÉS (``start_lo==start_hi``, ``end_lo==end_hi``) c'est
    EXACTEMENT ``GainDrift`` : la formule d'interpolation est reproduite à
    l'identique (même expression flottante) ⇒ pivot variance=0 bit-à-bit.

    Le tirage est figé à la construction (``from_seed``) : ``start``/``end`` sont
    des floats concrets, ``.at`` est alors identique mot pour mot à ``GainDrift.at``.
    """

    start: float
    end: float

    def at(self, t: int, T: int) -> float:
        """Facteur de dérive au pas ``t`` (formule IDENTIQUE à ``GainDrift.at``)."""
        if T <= 1:
            return self.start
        frac = t / (T - 1)
        return self.start + (self.end - self.start) * frac

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        start_lo: float = 0.93,
        start_hi: float = 0.97,
        end_lo: float = 1.07,
        end_hi: float = 1.13,
    ) -> "SeededDrift":
        """Tire (start, end) par graine dans les intervalles a priori (seedé).

        Tirage déterministe via ``torch.Generator().manual_seed(seed)``. Deux
        ``rand()`` consécutifs (start puis end) : l'ordre est figé, donc la même
        graine donne toujours le même couple.
        """
        g = torch.Generator().manual_seed(seed)
        u = torch.rand(2, generator=g)  # (2,) dans [0,1)
        start = start_lo + (start_hi - start_lo) * float(u[0])
        end = end_lo + (end_hi - end_lo) * float(u[1])
        return cls(start=start, end=end)

    @classmethod
    def degenerate(cls, *, start: float = 0.95, end: float = 1.10) -> "SeededDrift":
        """P1 DÉGÉNÉRÉE (amplitude=0) : offset/pente FIXÉS aux valeurs T15.

        Reproduit ``GainDrift(start, end)`` bit-à-bit (cible du pivot anti-artefact).
        """
        return cls(start=start, end=end)


@dataclass(frozen=True)
class ProcessNoise:
    """P2 — bruit de process AR(1) gaussien seedé, MOYENNE NULLE, ajouté au gain.

    Le facteur de gain natif au pas ``t`` est ``base + e_t`` où ``e_t`` est un
    processus AR(1) à moyenne nulle :

        e_0 = w_0
        e_t = φ·e_{t-1} + w_t,     w_t ~ N(0, σ²)   (i.i.d. seedés)

    ``φ`` (corrélation) et ``σ`` (échelle) sont posés A PRIORI. ``base = 1.0``
    (gain natif neutre : sans bruit, oscilloscope = cercle). Le processus est
    PROCHE-STATIONNAIRE (pas de tendance) : c'est le CONTRASTE qui tranche
    l'issue (d) — si l'avantage T15 venait de la non-stationnarité (rampe), il
    doit s'EFFONDRER sous P2.

    Le tirage est figé à la construction (``from_seed`` pré-calcule toute la série
    ``e_0…e_{T-1}``). À ``σ = 0`` ⇒ ``e_t = 0`` ∀t ⇒ facteur ≡ ``base`` ∀t :
    pivot variance=0 (P2 dégénérée = oscilloscope à g natif constant ``base``).
    """

    series: Tuple[float, ...]   # e_0 … e_{T-1} pré-calculés (la trajectoire du bruit)
    base: float                 # gain natif autour duquel oscille le bruit

    def at(self, t: int, T: int) -> float:
        """Facteur ``base + e_t`` au pas ``t`` (série pré-calculée, déterministe)."""
        if not self.series:
            return self.base
        idx = t if t < len(self.series) else len(self.series) - 1
        return self.base + self.series[idx]

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        steps: int,
        phi: float = 0.5,
        sigma: float = 0.04,
        base: float = 1.0,
    ) -> "ProcessNoise":
        """Tire la trajectoire AR(1) ``e_0…e_{steps-1}`` par graine (seedée).

        Innovations ``w_t ~ N(0, σ²)`` via ``torch.randn(steps, generator=...)``.
        À ``σ = 0`` la série est nulle (pivot). ``φ ∈ [0,1)`` posé a priori.
        """
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(steps, generator=g) * float(sigma)
        e: List[float] = []
        prev = 0.0
        for t in range(steps):
            cur = phi * prev + float(w[t])
            e.append(cur)
            prev = cur
        return cls(series=tuple(e), base=base)

    @classmethod
    def degenerate(cls, *, base: float = 1.0) -> "ProcessNoise":
        """P2 DÉGÉNÉRÉE (σ=0) : série nulle ⇒ facteur ≡ ``base`` ∀t (pivot)."""
        return cls(series=(), base=base)


# --- perturbation MÉLANGÉE (Tour 17, mélange convexe au niveau du gain) -------
#
# H17 (linguiste) : on interpole CONVEXEMENT P1 (non-stationnaire) et P2 (proche-
# stationnaire moyenne-nulle) AU NIVEAU DU FACTEUR DE GAIN :
#
#     p_α(t) = α·p1.at(t,T) + (1−α)·p2.at(t,T)
#
# où p1 = SeededDrift.from_seed(seed) et p2 = ProcessNoise.from_seed(seed, steps)
# PARTAGENT la même graine (appariement T16 préservé). Geste : ADD·dextro·out
# (agrégation pondérée = dual du SUB du T16 qui distinguait les graines).
#
# PIVOT ANTI-ARTEFACT (à exécuter EN PREMIER) — garanti par construction IEEE754 :
#   * à α=1.0 : ``1.0·p1.at + 0.0·p2.at`` == ``p1.at`` bit-à-bit (1.0·x exact,
#     0.0·y == 0.0 pour y fini, x + 0.0 == x). ⇒ p_α ≡ SeededDrift.from_seed.
#   * à α=0.0 : ``0.0·p1.at + 1.0·p2.at`` == ``p2.at`` bit-à-bit (idem, symétrie
#     de l'addition à un opérande nul à droite). ⇒ p_α ≡ ProcessNoise.from_seed.
# L'ordre des opérandes (p1 d'abord, p2 ensuite) est figé pour que cette identité
# tienne sur TOUS les t/graines. Un seul écart au pivot ⇒ bug (issue v), on
# s'arrête.
#
# H17 prédit que ``Δf_edge`` suit la composante DC (dérive nette ∝ α dans le
# mélange : P2 moyenne-nulle ne contribue pas aux bornes en espérance), PAS la
# variation totale ∫|dg/dt| (qui décroît en α car la h.f. de P2 perd du poids).


@dataclass(frozen=True)
class MixedPerturbation:
    """P_α — mélange CONVEXE direct de deux perturbations au niveau du gain.

    Facteur de gain natif au pas ``t`` : ``α·p1.at(t,T) + (1−α)·p2.at(t,T)``.
    ``p1`` et ``p2`` sont des perturbations quelconques satisfaisant le protocole
    ``.at(t,T)->float`` (ici P1=SeededDrift et P2=ProcessNoise construites depuis
    la MÊME graine via ``from_seed``). ``alpha ∈ [0,1]``.

    Le mélange est DIRECT (pas de re-tirage) : l'aléa vit entièrement dans p1/p2,
    déjà figés à la construction. ``MixedPerturbation`` ne fait qu'une combinaison
    linéaire déterministe de leurs sorties ⇒ aucune source aléatoire ici, pivot
    bit-à-bit aux bornes garanti par IEEE754 (cf. en-tête de section).
    """

    p1: Perturbation        # composante de poids α   (P1 = SeededDrift, non-stationnaire)
    p2: Perturbation        # composante de poids 1−α (P2 = ProcessNoise, moyenne-nulle)
    alpha: float            # poids convexe de p1 (∈ [0,1])

    def at(self, t: int, T: int) -> float:
        """Facteur ``α·p1.at + (1−α)·p2.at`` au pas ``t`` (ordre des opérandes figé)."""
        a = self.alpha
        return a * self.p1.at(t, T) + (1.0 - a) * self.p2.at(t, T)

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        alpha: float,
        steps: int,
        # bornes P1 (SeededDrift) — défauts T16, posés A PRIORI
        start_lo: float = 0.93,
        start_hi: float = 0.97,
        end_lo: float = 1.07,
        end_hi: float = 1.13,
        # paramètres P2 (ProcessNoise) — défauts T16, posés A PRIORI
        phi: float = 0.5,
        sigma: float = 0.04,
        base: float = 1.0,
    ) -> "MixedPerturbation":
        """Construit p1=SeededDrift.from_seed et p2=ProcessNoise.from_seed (MÊME graine).

        Les défauts reproduisent EXACTEMENT les perturbations T16 ⇒ à α=1 le mélange
        coïncide bit-à-bit avec ``SeededDrift.from_seed(seed)`` et à α=0 avec
        ``ProcessNoise.from_seed(seed, steps=steps)`` (pivot anti-artefact).
        """
        p1 = SeededDrift.from_seed(
            seed, start_lo=start_lo, start_hi=start_hi, end_lo=end_lo, end_hi=end_hi
        )
        p2 = ProcessNoise.from_seed(seed, steps=steps, phi=phi, sigma=sigma, base=base)
        return cls(p1=p1, p2=p2, alpha=alpha)

    def degenerate(self) -> "Perturbation":
        """Cas dégénérés EXACTS aux bornes (renvoie la composante pure, sans mélange).

        À α=1 ⇒ ``p1`` (SeededDrift) ; à α=0 ⇒ ``p2`` (ProcessNoise). Sert d'oracle
        de comparaison pour le test de pivot : la composante pure et le mélange à la
        borne doivent coïncider bit-à-bit. Hors bornes ⇒ ValueError (pas de
        composante « pure » bien définie).
        """
        if self.alpha == 1.0:
            return self.p1
        if self.alpha == 0.0:
            return self.p2
        raise ValueError(
            "degenerate() n'est défini qu'aux bornes α∈{0,1} ; "
            f"reçu α={self.alpha!r} (mélange strict, pas de composante pure)"
        )


# --- perturbation DÉRIVE + SINUS H.F. (Tour 18, DISJONCTION net_drift / total_var) ---
#
# H18 (linguiste) : le mélange convexe α du T17 CONFOND deux variables (net_drift
# CROÎT en α, total_var DÉCROÎT) le long d'un seul axe — on ne peut pas dire
# laquelle pilote ``Δf_edge``. Le geste DIV·lévo·in (⊘ SÉPARER ce qui était
# confondu, l.200) impose ici une perturbation où l'une est CONSTANTE et l'autre
# VARIE INDÉPENDAMMENT :
#
#     p(t) = p1.at(t, T) + A·sin(2π·k·t/(N−1))
#
# où ``p1 = SeededDrift.from_seed(seed)`` est la dérive PAR GRAINE du T16 (FIXE,
# net_drift constant en A) et le second terme est un sinus moyenne-nulle dont
# l'AMPLITUDE ``A`` fait varier total_var SANS toucher net_drift.
#
# DISJONCTION net_drift / total_var (le cœur du test) :
#   * net_drift = |p(N−1) − p(0)|. Le sinus s'annule aux DEUX extrémités
#     ÉCHANTILLONNÉES quand ``k`` est ENTIER et la phase porte sur ``N−1`` :
#         sin(2π·k·0/(N−1)) = sin(0) = 0          (exact)
#         sin(2π·k·(N−1)/(N−1)) = sin(2π·k) ≈ 0    (résidu flottant ~5e-15)
#     ⇒ net_drift NE FUIT PAS dans le sinus (plat à ~1e-6, garde auto-protectrice :
#     une phase sur ``N`` au lieu de ``N−1`` ferait fuir net_drift et FAIT ÉCHOUER
#     la pré-condition — le bug se révèle, il ne se cache pas).
#   * total_var = Σ_t |p(t+1) − p(t)| CROÎT linéairement avec A (le sinus ajoute
#     de la variation totale ∝ A·k indépendamment de net_drift).
#
# PIVOT ANTI-ARTEFACT (à exécuter EN PREMIER) — garanti par construction IEEE754 :
#   à A=0, ``A·sin(…)`` = ``0.0·x`` == 0.0 (x fini) et ``p1.at + 0.0`` == ``p1.at``
#   bit-à-bit. ⇒ ``DriftPlusHFSine(p1, amplitude=0.0)`` ≡ ``SeededDrift.from_seed``.
#   Le pivot A=0 DOIT reproduire le point P1 du T16 / α=1 du T17 = Δf_edge médian
#   +0.6556 (PAS +0.6623 de .degenerate() mono-série) car la base est ``from_seed``
#   PAR GRAINE (population de 40 tirages), pas la dérive fixe partagée du T15.
#
# NOTE D'HONNÊTETÉ (ρ̂ large bande) : ``ρ̂_t = r_t/r_{t-1}`` est un ratio à UN pas
# ⇒ le contrôleur VOIT la h.f. dans son estimée et y réagira. L'issue (i) « Δ plat »
# n'est PAS acquise d'avance : le contrôleur peut tracker/annuler la h.f. (Δ plat
# ou amélioré) ou sur-corriger/chattering (Δ dégradé). On mesure, on ne présume pas.


@dataclass(frozen=True)
class DriftPlusHFSine:
    """P1 (dérive par graine) + sinus h.f. moyenne-nulle d'amplitude ``A`` (T18).

    Facteur de gain natif au pas ``t`` : ``p1.at(t, T) + amplitude·sin(2π·k·t/(N−1))``
    où ``N = steps`` (le nombre d'échantillons sur lesquels la perturbation est lue,
    indices 0..N−1) et ``k = k_periods`` est ENTIER (nombre de périodes complètes sur
    l'horizon). La phase sur ``N−1`` annule le sinus aux deux extrémités échantillonnées
    ⇒ ``net_drift = |p(N−1) − p(0)|`` reste celui de ``p1`` (à ~5e-15 près).

    p1        : la dérive de base (ici ``SeededDrift.from_seed(seed)``, MÊME tirage T16).
    amplitude : ``A`` ≥ 0, amplitude du sinus (0 ⇒ ``p1`` pur, pivot anti-artefact).
    k_periods : ``k`` entier, nombre de périodes complètes du sinus sur ``N−1``.
    steps     : ``N``, l'horizon d'échantillonnage (doit valoir le ``steps`` du run).

    L'aléa vit ENTIÈREMENT dans ``p1`` (déjà figé) ; le sinus est déterministe. Aucune
    source aléatoire ici ⇒ pivot bit-à-bit à A=0 garanti par IEEE754.
    """

    p1: "SeededDrift"
    amplitude: float
    k_periods: int
    steps: int

    def at(self, t: int, T: int) -> float:
        """Facteur ``p1.at(t,T) + A·sin(2π·k·t/(N−1))`` (ordre des opérandes figé)."""
        base = self.p1.at(t, T)
        if self.amplitude == 0.0:
            return base  # pivot : 0.0·sin == 0.0, base + 0.0 == base (redondant mais explicite)
        n = self.steps
        if n <= 1:
            return base
        sine = math.sin(2.0 * math.pi * self.k_periods * t / (n - 1))
        return base + self.amplitude * sine

    @classmethod
    def from_seed(
        cls,
        seed: int,
        *,
        amplitude: float,
        k_periods: int = 20,
        steps: int = 200,
        # bornes P1 (SeededDrift) — défauts T16, posés A PRIORI (MÊME tirage)
        start_lo: float = 0.93,
        start_hi: float = 0.97,
        end_lo: float = 1.07,
        end_hi: float = 1.13,
    ) -> "DriftPlusHFSine":
        """Construit ``p1 = SeededDrift.from_seed(seed)`` (MÊME tirage T16) + sinus A.

        À ``amplitude=0`` ⇒ ``p1`` pur ⇒ coïncide bit-à-bit avec
        ``SeededDrift.from_seed(seed)`` (pivot anti-artefact : point P1 du T16). Les
        défauts P1 reproduisent EXACTEMENT le tirage T16 (population de 40 graines).
        """
        p1 = SeededDrift.from_seed(
            seed, start_lo=start_lo, start_hi=start_hi, end_lo=end_lo, end_hi=end_hi
        )
        return cls(p1=p1, amplitude=amplitude, k_periods=k_periods, steps=steps)

    def degenerate(self) -> "SeededDrift":
        """Cas dégénéré EXACT à A=0 : renvoie ``p1`` (SeededDrift) — oracle de pivot.

        Sert d'oracle de comparaison bit-à-bit pour le test de pivot (comme
        ``MixedPerturbation.degenerate`` aux bornes). À ``amplitude≠0`` ⇒ ValueError
        (pas de composante « pure » bien définie hors A=0).
        """
        if self.amplitude == 0.0:
            return self.p1
        raise ValueError(
            "degenerate() n'est défini qu'à A=0 (sinus nul) ; "
            f"reçu amplitude={self.amplitude!r} (perturbation stricte)"
        )


# --- protocole de perturbation -----------------------------------------------
#
# Toute perturbation acceptée par ``EdgeController.run`` / ``run_fixed_gain``
# expose ``.at(t, T) -> float`` (facteur de gain natif au pas t). ``GainDrift``
# (T15, partagée), ``SeededDrift`` (P1, par graine), ``ProcessNoise`` (P2, par
# graine), ``MixedPerturbation`` (P_α, mélange convexe T17) et ``DriftPlusHFSine``
# (P1 + sinus h.f., disjonction T18) satisfont ce protocole — d'où l'union de type
# ci-dessous (purement documentaire : la boucle n'appelle QUE ``.at``).
Perturbation = Union[
    GainDrift, "SeededDrift", "ProcessNoise", "MixedPerturbation", "DriftPlusHFSine"
]


# --- l'ORGANE générique : la loi de gain abstraite (Tour 19, H1) -------------
#
# H1 (linguiste) : la loi de mise à jour de ``g`` du contrôleur T15 (l.548-549
# ci-dessous, inchangées sémantiquement) ne dépend PAS de la nature de l'observable.
# Sa structure est « percevoir l'écart à une cible / corriger proportionnellement /
# borner ». On l'EXTRAIT ici en une fonction pure, indépendante de ``ρ̂`` :
#
#     regulate_step(g_prev, obs, target, eta, g_min, g_max)
#       = clip( g_prev − η·(obs − target), g_min, g_max )
#
# À ``obs = ρ̂``, ``target = 1.0`` c'est EXACTEMENT la loi T15 (les deux lignes du
# contrôleur appellent désormais cette fonction). Geste SUB·lévo·in : abstraire
# (distinguer la STRUCTURE de l'instance) — l.341-342, « la diversité est une
# perturbation contrôlée ». Un seul écart au pivot bit-à-bit ⇒ bug, pas résultat.


def regulate_step(
    g_prev: float,
    obs: float,
    target: float,
    eta: float,
    g_min: float,
    g_max: float,
) -> float:
    """Une mise à jour de gain proportionnelle bornée — la loi T15, abstraite.

    ``g ← clip( g_prev − η·(obs − target), g_min, g_max )``.

    g_prev : gain courant (avant ce pas).
    obs    : observable lu sur la trace (ρ̂ pour T15 ; cos de phase pour la 2e instance).
    target : valeur-cible de l'observable (1.0 pour ρ̂ = bord du chaos ; 0.7 pour la phase).
    eta    : gain de rétroaction η (0 ⇒ loi inerte, g reste g_prev).
    g_min, g_max : bornes du clip.

    Identité de structure : à ``obs=ρ̂``, ``target=1.0`` cette expression est mot pour
    mot les lignes 548-549 d'``EdgeController.run`` ⇒ pivot bit-à-bit (IEEE754 : même
    suite d'opérations flottantes, même résultat). Aucune source aléatoire.
    """
    g_cur = g_prev - eta * (obs - target)
    g_cur = min(g_max, max(g_min, g_cur))
    return g_cur


# --- le contrôleur -----------------------------------------------------------

@dataclass(frozen=True)
class EdgeControllerConfig:
    omega: float        # angle de rotation par pas (couplage anti-symétrique D↔L)
    g0: float           # gain de contrôle initial g_0
    eta: float          # gain de rétroaction η (0 ⇒ contrôleur inerte = g fixe)
    g_min: float        # borne basse du clip
    g_max: float        # borne haute du clip


@dataclass(frozen=True)
class ControlTrace:
    """Trace d'un déroulé sous contrôle (toutes les séries alignées par pas)."""

    trace: torch.Tensor       # (T+1, 2) : positions s_0 … s_T
    radius: torch.Tensor      # (T+1,)   : r_t = ‖s_t‖
    g_ctrl: torch.Tensor      # (T+1,)   : g_t appliqué au pas t (g_ctrl[0]=g0, inutilisé)
    rho_hat: torch.Tensor     # (T+1,)   : ρ̂_t = r_t/r_{t-1} (rho_hat[0]=nan : pas défini)
    g_drift: torch.Tensor     # (T+1,)   : g_drift(t) appliqué au pas t


class EdgeController:
    """Enveloppe l'oscilloscope d'un gain ``g_t`` AUTO-RÉGULÉ par ``r_t/r_{t-1}``.

    L'oscilloscope est passif : à chaque pas, la transition est
    ``A_t = g_ctrl(t)·g_drift(t)·R(ω)``. Le contrôleur lit le rapport de rayons réalisé
    ``ρ̂_t = r_t/r_{t-1}`` (le gain EFFECTIF du pas précédent, dérive comprise) et met à
    jour ``g_ctrl`` par la loi proportionnelle bornée

        g_ctrl(t) = clip( g_ctrl(t-1) − η·(ρ̂_t − 1), g_min, g_max ).

    Le contrôleur ne voit PAS ``g_drift`` : il ne corrige que sur la trace observée.

    À ``η = 0`` la loi est inerte (``g_ctrl(t) = g0`` ∀t) : le contrôleur reproduit
    bit-à-bit un oscilloscope à ``g`` fixe ``g0`` sous la même dérive (anti-bug).
    """

    def __init__(
        self,
        *,
        omega: float = math.pi / 5,
        g0: float = 1.0,
        eta: float = 0.5,
        g_min: float = 0.80,
        g_max: float = 1.20,
    ) -> None:
        self.cfg = EdgeControllerConfig(
            omega=omega, g0=g0, eta=eta, g_min=g_min, g_max=g_max
        )
        self.R = rotation_matrix(omega)  # (2,2), float32
        self.W_in = torch.eye(2, dtype=torch.float32)

    @torch.no_grad()
    def run(
        self,
        s0: torch.Tensor,
        *,
        steps: int,
        drift: Perturbation,
        signal: Optional[InputSignal] = None,
    ) -> ControlTrace:
        """Déroule ``steps`` pas sous la dérive ``drift`` et retourne la trace complète.

        s0     : état initial ``(2,)``.
        steps  : horizon ``T`` (la dérive est échantillonnée sur ``T`` pas).
        drift  : perturbation ``GainDrift`` (cible mobile imposée sous le contrôleur).
        signal : courant d'entrée optionnel (défaut : aucun — régime libre, le porteur
                 de la forme est la transition seule, comme aux Tours 4-6).

        ``@torch.no_grad`` : chemin de DIAGNOSTIC (on mesure une trajectoire). Tout est
        déterministe : aucune source aléatoire dans la boucle.
        """
        if s0.dim() != 1 or s0.size(0) != 2:
            raise ValueError("s0 doit être de forme (2,)")
        if steps < 1:
            raise ValueError("steps doit valoir >= 1")

        s0 = s0.to(torch.float32)
        cur = s0
        r_prev = float(torch.linalg.vector_norm(cur))

        pts: List[torch.Tensor] = [cur]
        radii: List[float] = [r_prev]
        g_list: List[float] = [self.cfg.g0]          # g appliqué au pas t
        rho_list: List[float] = [float("nan")]       # ρ̂_t : non défini à t=0
        drift_list: List[float] = []

        g_cur = self.cfg.g0
        for t in range(steps):
            # --- estimation en ligne du gain réalisé ρ̂_t = r_t / r_{t-1} ---
            if t == 0:
                rho_hat = 1.0  # pas de pas précédent : on ne corrige pas encore
            else:
                r_t = radii[-1]
                rho_hat = r_t / r_prev if r_prev > 0 else 1.0
                # --- loi proportionnelle bornée (l'ORGANE générique, obs=ρ̂, target=1) ---
                # Identique bit-à-bit aux anciennes l.548-549 : regulate_step n'est que
                # l'extraction littérale de cette expression (REFUS : API/défaut inchangés).
                g_cur = regulate_step(
                    g_cur, rho_hat, 1.0,
                    self.cfg.eta, self.cfg.g_min, self.cfg.g_max,
                )

            g_drift_t = drift.at(t, steps)
            A_t = (g_cur * g_drift_t) * self.R

            u = (signal.at(t) if signal is not None else None)
            nxt = cur @ A_t.t()
            if u is not None:
                nxt = nxt + u @ self.W_in.t()

            # journalise (g_cur et rho_hat valent pour CE pas t)
            if t > 0:
                g_list.append(g_cur)
                rho_list.append(rho_hat)
            drift_list.append(g_drift_t)

            r_prev = radii[-1]
            cur = nxt
            r_now = float(torch.linalg.vector_norm(cur))
            pts.append(cur)
            radii.append(r_now)

        # alignements de longueur : trace/radius ont T+1 points ; g_ctrl/rho_hat aussi
        # (g_list[0]=g0, rho_list[0]=nan, puis un par pas t>=1). drift a T entrées →
        # on aligne en répétant la dernière (le pas T n'applique aucune transition).
        while len(g_list) < steps + 1:
            g_list.append(g_cur)
        while len(rho_list) < steps + 1:
            r_t = radii[len(rho_list)]
            r_pm = radii[len(rho_list) - 1]
            rho_list.append(r_t / r_pm if r_pm > 0 else 1.0)
        drift_list.append(drift.at(steps - 1, steps))

        return ControlTrace(
            trace=torch.stack(pts, dim=0),
            radius=torch.tensor(radii, dtype=torch.float64),
            g_ctrl=torch.tensor(g_list, dtype=torch.float64),
            rho_hat=torch.tensor(rho_list, dtype=torch.float64),
            g_drift=torch.tensor(drift_list, dtype=torch.float64),
        )


# --- baseline : oscilloscope à g FIXE sous la MÊME dérive --------------------

@torch.no_grad()
def run_fixed_gain(
    s0: torch.Tensor,
    *,
    steps: int,
    g_fixed: float,
    omega: float = math.pi / 5,
    drift: Optional[Perturbation] = None,
    signal: Optional[InputSignal] = None,
) -> ControlTrace:
    """Baseline : oscilloscope à gain de contrôle CONSTANT ``g_fixed`` sous la dérive.

    Strictement équivalent à ``EdgeController(g0=g_fixed, eta=0.0).run(...)`` : c'est le
    ``g`` fixe du balayage {0.94…1.09}, soumis à la MÊME perturbation. Implémenté
    explicitement (sans la loi de mise à jour) pour servir de baseline indépendante ET
    de cible du test d'équivalence bit-à-bit ``η=0`` (les deux chemins doivent coïncider
    exactement).
    """
    if drift is None:
        drift = GainDrift()
    ctrl = EdgeController(omega=omega, g0=g_fixed, eta=0.0)
    return ctrl.run(s0, steps=steps, drift=drift, signal=signal)


# --- l'ORGANE générique paramétré par un observable quelconque (Tour 19, H1) --
#
# H1 prouvée par CONSTRUCTION+MESURE : ``GenericRegulator`` est l'``EdgeController``
# dont l'observable et la cible sont DÉCOUPLÉS de ρ̂. Il déroule la MÊME mécanique
# (rotation R(ω), gain g·drift, journalisation) mais pilote ``g`` par
# ``regulate_step(g_prev, obs, target, …)`` où ``obs`` est calculé par un ``obs_fn``
# fourni. À ``obs_fn = obs_rho_hat`` et ``target = 1.0`` il REPRODUIT bit-à-bit le
# contrôleur (pivot (i)). À ``obs_fn = obs_phase_coherence`` et ``target = 0.7`` il
# régule une quantité STRUCTURELLEMENT DISJOINTE (angle, pas échelle) — 2e instance.

ObsFn = "Callable[[List[torch.Tensor], List[float], int], float]"


def obs_rho_hat(pts: List[torch.Tensor], radii: List[float], t: int) -> float:
    """Observable ρ̂_t = r_t / r_{t-1} (l'instance T15 : RATIO de rayons = échelle).

    Lit le rapport des deux derniers rayons journalisés (le gain RÉALISÉ au pas
    précédent, dérive comprise). À ``t=0`` (pas de pas antérieur) retourne ``target``
    par convention via l'appelant — ici on renvoie 1.0 (neutre vis-à-vis de la cible 1).
    Identique au calcul en ligne de ``EdgeController.run`` (pivot bit-à-bit).
    """
    if t == 0:
        return 1.0
    r_prev = radii[t - 1]
    r_t = radii[t]
    return r_t / r_prev if r_prev > 0 else 1.0


def make_obs_phase_coherence(window: int) -> "Callable":
    """Fabrique l'observable cos(s_t, s_{t-W}) (2e instance : ANGLE, pas échelle).

    Cohérence de phase locale : l'alignement entre l'état courant et l'état ``window``
    pas plus tôt — exactement la quantité ``cos(s_t, s_{t-W})`` de la bande PROGRESSION
    (``_band_mask`` d'``edge_maintenance``, cos_thresh=0.7, W=10). C'est l'ANGLE de la
    rotation cumulée sur une fenêtre, DISJOINT du ratio de rayons ρ̂ (échelle).

    Avant qu'une fenêtre complète soit disponible (``t < window``), l'observable n'est
    pas défini : on renvoie ``None`` pour signaler à l'organe « ne corrige pas encore »
    (parallèle exact du ``t==0`` de ρ̂). L'organe laisse alors ``g`` inchangé.
    """
    def obs_phase(pts: List[torch.Tensor], radii: List[float], t: int):
        if t < window:
            return None  # fenêtre incomplète : observable indéfini (pas de correction)
        a = pts[t]
        b = pts[t - window]
        na = float(torch.linalg.vector_norm(a))
        nb = float(torch.linalg.vector_norm(b))
        if na <= 0.0 or nb <= 0.0:
            return -1.0  # un point nul n'est pas aligné (cohérent avec _band_mask)
        return float((a @ b) / (na * nb))
    return obs_phase


@dataclass(frozen=True)
class RegulatorTrace:
    """Trace d'un déroulé sous ``GenericRegulator`` (séries alignées par pas).

    Identique à ``ControlTrace`` mais le champ ``obs`` remplace ``rho_hat`` (généralisé) :
    ``obs[t]`` est la valeur de l'observable régulé au pas t (``nan`` quand indéfini).
    On expose AUSSI ``rho_hat`` (recalculé = r_t/r_{t-1}) pour que ``edge_report`` —
    qui lit ``rho_hat`` pour ``reg_corr`` — reste applicable tel quel.
    """

    trace: torch.Tensor       # (T+1, 2)
    radius: torch.Tensor      # (T+1,)
    g_ctrl: torch.Tensor      # (T+1,)
    obs: torch.Tensor         # (T+1,) : observable régulé (nan si indéfini au pas)
    rho_hat: torch.Tensor     # (T+1,) : ratio de rayons (recalculé, pour edge_report)
    g_drift: torch.Tensor     # (T+1,)

    def as_control_trace(self) -> ControlTrace:
        """Vue ``ControlTrace`` (obs droppé) : permet de réutiliser ``edge_report`` tel quel."""
        return ControlTrace(
            trace=self.trace, radius=self.radius, g_ctrl=self.g_ctrl,
            rho_hat=self.rho_hat, g_drift=self.g_drift,
        )


class GenericRegulator:
    """L'ORGANE ``regulate(observable → cible)`` — l'``EdgeController`` dé-spécialisé.

    Même mécanique de transition que ``EdgeController`` (``A_t = g_t·g_drift(t)·R(ω)``,
    propagation séquentielle, @torch.no_grad de diagnostic) mais la loi de gain lit un
    observable ARBITRAIRE :

        obs_t = obs_fn(pts_so_far, radii_so_far, t)
        g_t   = regulate_step(g_{t-1}, obs_t, target, η, g_min, g_max)   si obs_t défini
        g_t   = g_{t-1}                                                  si obs_t is None

    ``obs_fn`` reçoit l'historique disponible (états + rayons jusqu'à t) et renvoie un
    float, ou ``None`` quand l'observable n'est pas encore défini (fenêtre incomplète,
    premier pas) — l'organe laisse alors ``g`` inchangé (parallèle du ``t==0`` de ρ̂).

    À ``obs_fn = obs_rho_hat``, ``target = 1.0`` : REPRODUIT ``EdgeController`` bit-à-bit
    (la transition, la journalisation et l'ordre des opérations flottantes coïncident).
    """

    def __init__(
        self,
        obs_fn,
        *,
        target: float,
        omega: float = math.pi / 5,
        g0: float = 1.0,
        eta: float = 0.5,
        g_min: float = 0.80,
        g_max: float = 1.20,
    ) -> None:
        self.obs_fn = obs_fn
        self.target = float(target)
        self.cfg = EdgeControllerConfig(
            omega=omega, g0=g0, eta=eta, g_min=g_min, g_max=g_max
        )
        self.R = rotation_matrix(omega)
        self.W_in = torch.eye(2, dtype=torch.float32)

    @torch.no_grad()
    def run(
        self,
        s0: torch.Tensor,
        *,
        steps: int,
        drift: Perturbation,
        signal: Optional[InputSignal] = None,
    ) -> RegulatorTrace:
        """Déroule ``steps`` pas en régulant ``obs_fn`` vers ``target``. Cf. EdgeController.run."""
        if s0.dim() != 1 or s0.size(0) != 2:
            raise ValueError("s0 doit être de forme (2,)")
        if steps < 1:
            raise ValueError("steps doit valoir >= 1")

        s0 = s0.to(torch.float32)
        cur = s0
        r0 = float(torch.linalg.vector_norm(cur))

        pts: List[torch.Tensor] = [cur]
        radii: List[float] = [r0]
        g_list: List[float] = [self.cfg.g0]
        obs_list: List[float] = [float("nan")]   # observable au pas t (nan à t=0)
        drift_list: List[float] = []

        g_cur = self.cfg.g0
        for t in range(steps):
            # --- observable lu sur l'historique disponible (pts/radii : s_0 … s_t) ---
            obs_val = self.obs_fn(pts, radii, t)
            if t == 0 or obs_val is None:
                obs_t = float("nan")  # indéfini : on ne corrige pas (g inchangé)
            else:
                obs_t = float(obs_val)
                g_cur = regulate_step(
                    g_cur, obs_t, self.target,
                    self.cfg.eta, self.cfg.g_min, self.cfg.g_max,
                )

            g_drift_t = drift.at(t, steps)
            A_t = (g_cur * g_drift_t) * self.R

            u = (signal.at(t) if signal is not None else None)
            nxt = cur @ A_t.t()
            if u is not None:
                nxt = nxt + u @ self.W_in.t()

            if t > 0:
                g_list.append(g_cur)
                obs_list.append(obs_t)
            drift_list.append(g_drift_t)

            cur = nxt
            pts.append(cur)
            radii.append(float(torch.linalg.vector_norm(cur)))

        # alignements de longueur (T+1 points), mêmes conventions qu'EdgeController.run
        while len(g_list) < steps + 1:
            g_list.append(g_cur)
        while len(obs_list) < steps + 1:
            obs_list.append(float("nan"))
        drift_list.append(drift.at(steps - 1, steps))

        # rho_hat recalculé (r_t/r_{t-1}) pour que edge_report.reg_corr reste applicable
        rho_list: List[float] = [float("nan")]
        for t in range(1, len(radii)):
            r_pm = radii[t - 1]
            rho_list.append(radii[t] / r_pm if r_pm > 0 else 1.0)

        return RegulatorTrace(
            trace=torch.stack(pts, dim=0),
            radius=torch.tensor(radii, dtype=torch.float64),
            g_ctrl=torch.tensor(g_list, dtype=torch.float64),
            obs=torch.tensor(obs_list, dtype=torch.float64),
            rho_hat=torch.tensor(rho_list, dtype=torch.float64),
            g_drift=torch.tensor(drift_list, dtype=torch.float64),
        )
