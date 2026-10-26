from __future__ import annotations

"""Oscilloscope 2D — les OPÉRATEURS COMME GÉNÉRATEURS DE FORME (Tour 4, porteur géométrique).

CHANGEMENT DE PORTEUR. Les tours précédents lisaient ``chrono.py`` comme sonde de
*stabilité* (la norme croît-elle ?). Ici le porteur est la FORME GÉOMÉTRIQUE 2D :
on regarde la trajectoire ``s_t ∈ R²`` à l'oscilloscope et on demande quelle FIGURE
elle trace. La question H4 : un réglage des opérateurs (ADD/SUB/MUL/DIV) + orientation
D/L produit-il un CERCLE et une SPIRALE, mieux que des poids aléatoires de même échelle ?

GESTE OPÉRATOIRE (émission du linguiste, incarnée telle quelle) :

- ROTATION = couplage ANTI-SYMÉTRIQUE D↔L. Miroir canon : le dextrogyre exprime
  vers le dehors, le lévogyre reçoit vers le dedans ; leur couplage croisé est une
  rotation. Une matrice de bloc anti-symétrique ``G(ω) = [[0, −ω], [ω, 0]]`` est le
  GÉNÉRATEUR d'une rotation pure ; son exponentielle est exactement la matrice de
  Givens ``R(ω) = [[cos ω, −sin ω], [sin ω, cos ω]]`` (rotation d'angle ω, det = 1,
  norme préservée). On l'incarne directement par ``R(ω)``.
- RAYON↑ (centrifuge) = MUL / dextro / out. Gain radial ``g > 1`` : amplification,
  l'état s'éloigne du centre.
- RAYON↓ (centripète) = DIV / lévo / in. Gain radial ``g < 1`` : contraction.
  Le gain radial isotrope est ``g · I₂`` (scalaire fois identité) : il ne déforme
  pas la figure, il dilate/contracte le rayon — c'est le sens géométrique de
  « amplifier / ramifier » réduit à 2D.
- INERTIE / oscillation = terme MÉMOIRE ``s_{t−1}`` (terme ``−C`` du second ordre,
  §3.2). Reprend la mémoire explicite de ``ChronoSpiraton`` : l'état précédent entre
  dans la mise à jour, ce qui donne au système une « durée » (l.290 : la boucle
  revient à un point modifié, pas au même).
- SIGNAL D'ENTRÉE ``u_t`` = le « courant » de l'oscilloscope. Forme documentée
  (cf. ``InputSignal``) : impulsion initiale (Dirac à t=0, défaut — la figure est
  alors le régime libre), ou sinusoïde entretenue (excitation continue).

ÉQUATION (par pas). Avec gain radial ``g``, rotation ``R(ω)``, mémoire ``c`` :

    s_{t+1} = g · R(ω) · s_t  −  c · s_{t−1}  +  W_in · u_t

Le bloc ``g·R(ω)`` EST l'opérateur composé « tourner-puis-dilater » : c'est lui qui
porte tout le geste. Mettre ``c = 0`` redonne une récurrence du premier ordre (carte
linéaire pure) ; ``c > 0`` ajoute l'inertie du second ordre.

REFUS — ce réglage DÉCOULE DE LA SÉMANTIQUE, il n'est PAS fitté sur la forme cible :

  * Réglage CERCLE = rotation pure, gain radial = 1, mémoire = 0.
    ``g = 1`` ⇒ le rayon est exactement conservé pas après pas (``R`` est une
    isométrie) ⇒ la trajectoire est un cercle. AUCUN paramètre n'est ajusté en
    regardant la trace ; ``g = 1`` est la valeur sémantique « ni amplifier ni
    contracter » (MUL et DIV s'équilibrent), ``ω`` est l'angle de rotation libre.
  * Réglage SPIRALE = rotation + gain radial > 1 (centrifuge MUL/dextro/out).
    ``g > 1`` ⇒ le rayon croît d'un facteur ``g`` constant par tour ⇒
    ``log r`` est AFFINE en ``θ`` : une spirale logarithmique. Là encore la pente
    est posée par ``g`` (la sémantique « amplifier »), pas lue sur la cible.

Le seul paramètre laissé libre est l'angle ``ω`` (vitesse de rotation) et le gain
``g`` (sens et force du geste radial). Aucun des deux n'est optimisé contre une
figure mesurée : ce sont les COORDONNÉES du geste opératoire dans le plan.

STRICTEMENT EXPÉRIMENTAL. Ne touche pas le canon ``core/`` ni le défaut de
``chrono.py``. Réutilise la forme « opérateurs = applications linéaires » et la
mémoire ``s_{t−1}`` de ``ChronoSpiraton`` (lecture seule du squelette).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# --- signaux d'entrée u_t (le « courant » de l'oscilloscope) -----------------

@dataclass(frozen=True)
class InputSignal:
    """Forme du signal d'entrée ``u_t ∈ R²`` injecté à chaque pas.

    - ``kind="impulse"`` (défaut) : impulsion (Dirac) à ``t = 0`` valant ``amp``
      sur la première coordonnée, nulle ensuite. La trajectoire est alors le
      RÉGIME LIBRE de la récurrence (la figure « propre » des opérateurs). C'est
      le réglage de référence pour lire la forme géométrique sans excitation.
    - ``kind="sine"`` : excitation entretenue ``u_t = amp · [cos(Ωt), sin(Ωt)]``.
      Le « courant alternatif » de l'oscilloscope ; teste la réponse à un forçage.
    - ``kind="zero"`` : aucun courant (utile si ``s_0`` porte seul l'amplitude).
    """

    kind: str = "impulse"
    amp: float = 1.0
    omega: float = 0.0  # pulsation du forçage sinusoïdal (kind="sine")

    def at(self, t: int) -> torch.Tensor:
        """Valeur ``u_t`` (forme ``(2,)``) au pas ``t`` (t >= 0)."""
        if self.kind == "zero":
            return torch.zeros(2)
        if self.kind == "impulse":
            if t == 0:
                return torch.tensor([self.amp, 0.0])
            return torch.zeros(2)
        if self.kind == "sine":
            return torch.tensor(
                [self.amp * math.cos(self.omega * t), self.amp * math.sin(self.omega * t)]
            )
        raise ValueError(f"signal kind inconnu: {self.kind!r}")


def rotation_matrix(omega: float) -> torch.Tensor:
    """Matrice de Givens ``R(ω) = exp([[0,−ω],[ω,0]])`` : rotation pure d'angle ω.

    C'est l'EXPONENTIELLE du générateur anti-symétrique D↔L. ``det R = 1`` et
    ``R`` est une isométrie (préserve la norme) — d'où le cercle quand le gain
    radial vaut 1.
    """
    c, s = math.cos(omega), math.sin(omega)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


# --- la cellule oscilloscope -------------------------------------------------

@dataclass(frozen=True)
class OscilloscopeConfig:
    omega: float           # angle de rotation par pas (couplage anti-symétrique D↔L)
    gain: float            # gain radial g : >1 centrifuge (MUL/out), <1 centripète (DIV/in)
    memory: float          # coefficient c du terme d'inertie −c·s_{t−1} (second ordre)
    win_scale: float       # échelle de l'injection du signal d'entrée W_in


class Oscilloscope2D(nn.Module):
    """Cellule 2D dont les opérateurs sont des applications linéaires explicites.

    state_size est fixé à 2 (le plan de l'oscilloscope). La mise à jour est

        s_{t+1} = gain · R(ω) · s_t  −  memory · s_{t−1}  +  W_in · u_t

    où ``R(ω)`` porte la ROTATION (couplage anti-symétrique D↔L), ``gain`` le geste
    RADIAL (MUL centrifuge / DIV centripète), ``−memory·s_{t−1}`` l'INERTIE du
    second ordre, et ``W_in·u_t`` l'injection du courant d'entrée.

    Deux fabriques sémantiques (réglages qui DÉCOULENT du geste, jamais fittés) :
      * :meth:`circle`  — rotation pure, gain = 1, mémoire = 0  → cercle.
      * :meth:`spiral`  — rotation + gain > 1                   → spirale log.

    Et une fabrique de CONTRÔLE :
      * :meth:`random`  — poids ``A`` aléatoires (carte linéaire non structurée) de
        MÊME ÉCHELLE, regénérée par graine. Baseline obligatoire (REFUS).
    """

    state_size = 2

    def __init__(
        self,
        *,
        omega: float = 0.4,
        gain: float = 1.0,
        memory: float = 0.0,
        win_scale: float = 1.0,
        transition: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cfg = OscilloscopeConfig(
            omega=omega, gain=gain, memory=memory, win_scale=win_scale
        )

        # Opérateur de transition A = gain · R(ω) (rotation + dilatation isotrope).
        # Pour la baseline aléatoire, `transition` est fourni explicitement et
        # remplace la structure rotation×gain par une matrice quelconque.
        if transition is None:
            A = self.cfg.gain * rotation_matrix(self.cfg.omega)
        else:
            if transition.shape != (2, 2):
                raise ValueError("transition doit être (2,2)")
            A = transition.to(dtype=torch.float32)
        self.register_buffer("A", A)

        # Injection du signal d'entrée (identité mise à l'échelle : le courant
        # arrive « tel quel » dans le plan).
        self.register_buffer("W_in", self.cfg.win_scale * torch.eye(2, dtype=torch.float32))

    # -- fabriques sémantiques ------------------------------------------------

    @classmethod
    def circle(cls, *, omega: float = 0.4) -> "Oscilloscope2D":
        """Réglage CERCLE : rotation pure (gain = 1, mémoire = 0).

        ``gain = 1`` ⇒ ``A = R(ω)`` est une isométrie ⇒ rayon conservé ⇒ cercle.
        Posé par la SÉMANTIQUE (MUL et DIV s'équilibrent : ni dilatation ni
        contraction), pas par un fit. ``ω`` est l'angle libre.
        """
        return cls(omega=omega, gain=1.0, memory=0.0)

    @classmethod
    def spiral(cls, *, omega: float = 0.4, gain: float = 1.06) -> "Oscilloscope2D":
        """Réglage SPIRALE : rotation + gain radial > 1 (centrifuge MUL/dextro/out).

        ``gain > 1`` ⇒ rayon ×gain par pas ⇒ ``log r`` affine en ``θ`` ⇒ spirale
        logarithmique. La pente vient du gain (sémantique « amplifier »), pas de
        la cible. ``gain`` modéré (≈1.06) pour rester non-divergent sur l'horizon
        mesuré tout en bouclant ≥ 2 tours.
        """
        if gain <= 1.0:
            raise ValueError("spiral exige gain > 1 (centrifuge)")
        return cls(omega=omega, gain=gain, memory=0.0)

    @classmethod
    def random(cls, generator: torch.Generator, *, scale: float) -> "Oscilloscope2D":
        """Baseline CONTRÔLE : transition aléatoire non structurée, MÊME échelle.

        ``A`` est une matrice 2×2 gaussienne d'écart-type ``scale`` (l'échelle de
        référence = la norme spectrale typique des réglages ciblés). Aucune
        structure rotation×gain : c'est le « poids aléatoire » contre lequel H4
        doit gagner. Regénérée par graine via ``generator`` (déterminisme).
        """
        A = torch.randn(2, 2, generator=generator) * scale
        return cls(transition=A)

    # -- déroulé --------------------------------------------------------------

    def step(self, s_t: torch.Tensor, s_prev: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        """Un pas : ``s_{t+1} = A · s_t − memory · s_prev + W_in · u_t``.

        Formes ``(B, 2)`` (ou ``(2,)``). ``A`` et ``W_in`` agissent à droite sur
        les vecteurs-lignes (convention nn.Linear : ``s @ A.t()``).
        """
        out = s_t @ self.A.t() + u_t @ self.W_in.t()
        if self.cfg.memory != 0.0:
            out = out - self.cfg.memory * s_prev
        return out

    @torch.no_grad()
    def trace(
        self,
        s0: torch.Tensor,
        *,
        steps: int = 60,
        signal: Optional[InputSignal] = None,
        s_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Déroule ``steps`` pas et retourne la TRACE ``(steps+1, 2)`` (incluant s0).

        Trajectoire d'un seul point (forme ``(2,)`` attendue pour ``s0``). C'est
        l'écran de l'oscilloscope : la suite des positions ``s_0, s_1, …, s_steps``.

        ``@torch.no_grad`` à dessein : c'est un chemin de DIAGNOSTIC (on mesure une
        figure, on n'entraîne pas). Pour un déroulé différentiable (flux de
        gradient), passer par ``forward`` en mode batch ``(B, 2)``.
        """
        if signal is None:
            signal = InputSignal(kind="impulse", amp=1.0)
        if s0.dim() != 1 or s0.size(0) != 2:
            raise ValueError("s0 doit être de forme (2,)")
        if s_prev is None:
            s_prev = torch.zeros(2)

        prev, cur = s_prev, s0
        pts: List[torch.Tensor] = [cur]
        for t in range(steps):
            u = signal.at(t)
            nxt = self.step(cur.unsqueeze(0), prev.unsqueeze(0), u.unsqueeze(0)).squeeze(0)
            pts.append(nxt)
            prev, cur = cur, nxt
        return torch.stack(pts, dim=0)

    def forward(
        self,
        s0: torch.Tensor,
        *,
        steps: int = 60,
        signal: Optional[InputSignal] = None,
        s_prev: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        """Déroule la dynamique. Accepte ``(2,)`` ou ``(B,2)``.

        Avec ``return_trace=True`` et entrée ``(2,)``, retourne ``(s_final, trace)``
        où ``trace`` est ``(steps+1, 2)``. Le mode batch retourne seulement l'état
        final (la signature de forme se calcule trajectoire par trajectoire).
        """
        if signal is None:
            signal = InputSignal(kind="impulse", amp=1.0)

        single = s0.dim() == 1
        if single:
            tr = self.trace(s0, steps=steps, signal=signal, s_prev=s_prev)
            return (tr[-1], tr) if return_trace else tr[-1]

        if s0.size(-1) != 2:
            raise ValueError("dernière dim doit valoir 2")
        if s_prev is None:
            s_prev = torch.zeros_like(s0)
        prev, cur = s_prev, s0
        for t in range(steps):
            u = signal.at(t).expand(s0.size(0), 2)
            cur, prev = self.step(cur, prev, u), cur
        return cur
