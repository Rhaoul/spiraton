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
from typing import List, Optional, Tuple

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
        drift: GainDrift,
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
                # --- loi proportionnelle bornée ---
                g_cur = g_cur - self.cfg.eta * (rho_hat - 1.0)
                g_cur = min(self.cfg.g_max, max(self.cfg.g_min, g_cur))

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
    drift: Optional[GainDrift] = None,
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
