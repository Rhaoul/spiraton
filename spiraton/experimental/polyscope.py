from __future__ import annotations

"""Polyscope 2D — la NON-COMMUTATIVITÉ (axiome 1) comme condition d'une forme ANGULEUSE VRAIE (Tour 6).

CHANGEMENT DE GESTE (continuité du porteur géométrique des Tours 4-5). L'oscilloscope
(Tour 4) trace une figure avec UNE seule carte linéaire ``s_{t+1} = A·s_t`` : à valeurs
propres complexes ``A`` ne sait engendrer qu'une COURBURE UNIFORME (cercle, spirale).
Le n-gone *de sommets* lui-même (carré/triangle vus comme suite de sommets) n'est qu'une
rotation discrète ``R(2π/n)`` — donc LINÉAIRE, comme le cercle = 10-gone dense du Tour 4.

H6 (émission du linguiste, incarnée telle quelle) — une forme ANGULEUSE VRAIE (arêtes
DROITES multi-pas + COINS FRANCS, ou croix qui se recroise par le centre) n'est PAS
engendrable par une seule matrice à λ complexes (sa courbure est toujours uniforme),
mais l'est par une **alternance ordonnée de deux opérateurs ``W_a, W_b`` qui NE
COMMUTENT PAS** (``‖[W_a, W_b]‖ > 0``) :

  * le long de l'ARÊTE on applique UN SEUL opérateur (droit, courbure nulle) ;
  * au COIN on BASCULE d'opérateur (braquage franc) — c'est l'axiome 1 GÉOMÉTRISÉ.

GESTE OPÉRATOIRE par régime (DÉCOULE de la sémantique, n'est PAS dessiné à la main) :

- ANGULEUX-VRAI = MUL/dextro/out ANISOTROPE (dilatation selon UN axe) alterné avec une
  ROTATION ``R(ω)`` d'angle franc, AXES DIFFÉRENTS ⇒ non-commutatif. Sur ``k_a`` pas un
  cisaillement/dilatation pousse l'état le long d'une direction (arête droite) ; sur
  ``k_b`` pas la rotation braque (coin). La FORME (carré-ish, triangle-ish) ÉMERGE du
  rapport ``ω·k_b`` ≈ angle extérieur ; aucun sommet n'est posé.
- CROIX (+) = RÉFLEXION (SUB/lévo/in) × ROTATION π/2 = générateurs du groupe diédral
  ``D₄`` (NON-ABÉLIEN). L'alternance réflexion/quart-de-tour rabat la trajectoire À
  TRAVERS le centre : elle se recroise (≥1 self-intersection, ≥2 passages près du centre).
- N-GONE-DE-SOMMETS (contrôle linéaire « anguleux apparent ») = ``R(2π/n)`` PURE, un pas
  par sommet. C'est une rotation discrète ⇒ LINÉAIRE ⇒ ne teste PAS l'axiome 1 (H6a).
- CONTRÔLE COMMUTATEUR=0 = deux opérateurs qui COMMUTENT (``W_b = I``, ou ``W_a, W_b``
  co-diagonaux). À commutateur nul la composition dégénère en UNE seule application ⇒
  la cornerness doit S'EFFONDRER au niveau orbite-lisse. Preuve anti-artefact que c'est
  bien la non-commutativité qui PORTE le coin (parallèle au CTRL D=L du Tour 1).

ÉQUATION (par pas, premier ordre, mémoire=0 — on isole l'effet de l'ALTERNANCE) :

    s_{t+1} = W(t) · s_t ,   W(t) = W_a si (t mod (k_a+k_b)) < k_a, sinon W_b

Le PROGRAMME d'alternance ``(W_a, k_a, W_b, k_b)`` EST le geste : il n'y a aucune cible
de forme dans le code. Ce qui est mesuré (cornerness, passages-centre, fermeture) est lu
SUR la trace par ``corner_signature`` (diagnostics), jamais imposé.

REFUS — discipline. Aucun sommet hard-codé : la forme émerge du rapport angle/longueur.
Baseline linéaire (oscilloscope, Tour 4) et contrôle ``‖[·,·]‖=0`` obligatoires
(``angular_separability``). ``commutator_norm`` (matrix_cell, dormant) réveillé comme
covariable. STRICTEMENT EXPÉRIMENTAL : ne touche ni le canon, ni ``oscilloscope.py``,
ni la théorie.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .matrix_cell import commutator_norm  # réveil du diagnostic dormant (axiome 1)
from .oscilloscope import rotation_matrix  # R(ω) partagé avec le Tour 4


# --- briques d'opérateurs 2×2 (DÉCOULENT de la sémantique) --------------------


def anisotropic_shear(gain: float, *, axis_angle: float = 0.0) -> torch.Tensor:
    """Cisaillement ANISOTROPE area-preserving (det=1) selon un axe tourné de ``axis_angle``.

    MUL / dextro / out réduit à 2D mais ANISOTROPE : on amplifie d'un facteur ``gain``
    le long d'UN axe (direction ``axis_angle``) et on CONTRACTE de ``1/gain`` selon l'axe
    orthogonal. En base canonique : ``R(φ) · diag(gain, 1/gain) · R(φ)ᵀ`` — matrice
    symétrique, valeurs propres réelles ``gain`` et ``1/gain``, DÉTERMINANT = 1 (préserve
    l'aire). Appliquée seule et répétée, elle pousse l'état le long de la direction propre
    dominante (axe ``φ``) : une ARÊTE quasi-droite.

    POINT CLÉ pour l'axiome 1 : deux cisaillements d'AXES DIFFÉRENTS ne partagent pas
    leurs vecteurs propres ⇒ ils NE COMMUTENT PAS (``‖[W_a, W_b]‖ > 0``), SAUF le cas
    dégénéré ``Δφ = π/2`` (axes orthogonaux : les deux matrices sont co-diagonales et
    commutent — un piège documenté, à éviter pour le régime non-commutatif). Aux MÊMES
    axes, ils commutent (mêmes vecteurs propres) : la composition dégénère en un seul
    cisaillement ⇒ pas de coin. C'est ce contraste que le contrôle commutateur=0 exploite.
    """
    R = rotation_matrix(axis_angle)
    D = torch.tensor([[gain, 0.0], [0.0, 1.0 / gain]], dtype=torch.float32)
    return R @ D @ R.t()


def reflection(axis_angle: float) -> torch.Tensor:
    """Réflexion à travers la droite d'angle ``axis_angle`` (det = −1).

    SUB / lévo / in = DISTINCTION/retournement : une réflexion renverse l'orientation.
    ``Ref(φ) = [[cos 2φ, sin 2φ], [sin 2φ, −cos 2φ]]``. Avec une rotation π/2 elle
    engendre le groupe diédral ``D₄`` (non-abélien) — le geste de la croix.
    """
    c, s = math.cos(2 * axis_angle), math.sin(2 * axis_angle)
    return torch.tensor([[c, s], [s, -c]], dtype=torch.float32)


# --- la cellule polyscope ----------------------------------------------------


@dataclass(frozen=True)
class PolyscopeConfig:
    k_a: int          # nombre de pas consécutifs sous W_a (longueur d'arête « a »)
    k_b: int          # nombre de pas consécutifs sous W_b (longueur d'arête « b »)
    label: str        # nom du régime (jamais lu par les diagnostics ; trace seulement)


class Polyscope2D(nn.Module):
    """Cellule 2D qui ALTERNE deux applications linéaires ``W_a`` (k_a pas) / ``W_b`` (k_b pas).

    Mise à jour PAR MORCEAUX (premier ordre, mémoire=0) :

        s_{t+1} = W(t) · s_t ,  W(t) = W_a sur les k_a premiers pas du cycle, W_b ensuite.

    Le cycle a pour période ``k_a + k_b``. La FORME émerge du programme d'alternance ;
    aucune cible géométrique n'est codée. ``W_a`` et ``W_b`` sont des buffers (diagnostic,
    pas d'entraînement ici), ``commutator()`` expose ``‖[W_a, W_b]‖`` (covariable axiome 1).

    Fabriques sémantiques (réglages qui DÉCOULENT du geste) :
      * :meth:`angular`     — dilatation anisotrope × rotation, axes ≠ ⇒ NON-commutatif
                              (arête droite + coin franc : anguleux VRAI).
      * :meth:`cross`       — réflexion × rotation π/2 (générateurs de D₄, non-abélien) :
                              la croix qui se recroise par le centre.
      * :meth:`ngon_vertices` — CONTRÔLE LINÉAIRE : R(2π/n) PURE, 1 pas/sommet. Une seule
                              matrice ⇒ ``W_b`` confondu (k_b=0) ⇒ commutateur trivial.
      * :meth:`commuting`   — CONTRÔLE ‖[·,·]‖=0 : W_a, W_b co-diagonaux (ou W_b=I) qui
                              COMMUTENT. L'alternance dégénère ⇒ cornerness doit s'effondrer.
    """

    state_size = 2

    def __init__(
        self,
        W_a: torch.Tensor,
        W_b: torch.Tensor,
        *,
        k_a: int = 1,
        k_b: int = 1,
        label: str = "",
    ) -> None:
        super().__init__()
        if W_a.shape != (2, 2) or W_b.shape != (2, 2):
            raise ValueError("W_a et W_b doivent être (2, 2)")
        if k_a < 0 or k_b < 0 or (k_a + k_b) <= 0:
            raise ValueError("k_a, k_b >= 0 et k_a + k_b > 0")
        self.cfg = PolyscopeConfig(k_a=k_a, k_b=k_b, label=label)
        self.register_buffer("W_a", W_a.to(dtype=torch.float32))
        self.register_buffer("W_b", W_b.to(dtype=torch.float32))

    # -- fabriques sémantiques ------------------------------------------------

    # Réglages canon du régime anguleux-vrai (fixés après calibrage, JAMAIS fittés sur
    # une cible de forme : ils DÉCOULENT du geste « deux cisaillements d'axes différents »).
    ANGULAR_AXIS_A = 0.0
    ANGULAR_AXIS_B = 1.2   # ≠ 0 et ≠ π/2 : axes désalignés ⇒ ‖[W_a, W_b]‖ > 0 (vrai axiome 1)

    @classmethod
    def angular(
        cls,
        *,
        gain: float = 1.1,
        axis_a: Optional[float] = None,
        axis_b: Optional[float] = None,
        k_edge: int = 3,
    ) -> "Polyscope2D":
        """ANGULEUX-VRAI : alternance de DEUX cisaillements anisotropes d'AXES DIFFÉRENTS.

        ``W_a = shear(gain, axis_a)``, ``W_b = shear(gain, axis_b)``. Le long de chaque
        arête (``k_edge`` pas sous un seul opérateur) l'état suit la direction propre
        dominante de ce cisaillement (segment quasi-droit) ; au passage A→B la direction
        propre dominante CHANGE (axe ``axis_a`` → ``axis_b``) ⇒ la trajectoire BRAQUE : un
        COIN FRANC. Le braquage est PORTÉ par le désalignement des axes propres, c'est-à-dire
        par la NON-COMMUTATIVITÉ (``‖[W_a, W_b]‖ > 0`` dès que ``axis_a ≠ axis_b`` et
        ``≠ π/2``). C'est l'axiome 1 géométrisé.

        FERMETURE/STABILITÉ (mesurée à part, REFUS) : un produit de cisaillements
        non-commutants CROÎT (chaque shear amplifie selon son grand axe) ⇒ la figure est un
        POLYGONE-SPIRALE (coins francs + dérive bornée-mais-croissante), PAS une figure
        fermée. C'est le résultat central accepté par l'orchestrateur : on ne force pas la
        fermeture, on rapporte la dérive. ``gain`` modéré (≈1.1) garde la dérive FINIE sur
        l'horizon de mesure ; ``gain`` plus grand ⇒ commutateur plus net mais dérive plus
        forte. Aucun sommet posé : la forme émerge du rapport ``k_edge``/désalignement.
        """
        if axis_a is None:
            axis_a = cls.ANGULAR_AXIS_A
        if axis_b is None:
            axis_b = cls.ANGULAR_AXIS_B
        W_a = anisotropic_shear(gain, axis_angle=axis_a)
        W_b = anisotropic_shear(gain, axis_angle=axis_b)
        return cls(W_a, W_b, k_a=k_edge, k_b=k_edge, label=f"angular_g{gain}")

    @classmethod
    def cross(cls, *, contract: float = 0.8) -> "Polyscope2D":
        """CROIX (+) : FLIP à valeur propre RÉELLE NÉGATIVE × ROTATION π/2 — groupe diédral D₄ (non-abélien).

        ``W_a = diag(−contract, 1)`` : valeur propre RÉELLE NÉGATIVE ``−contract`` (|·|<1)
        le long de x. Une valeur propre réelle négative ENVOIE l'état de l'AUTRE CÔTÉ du
        centre (x → −contract·x) à chaque pas : la trajectoire TRAVERSE le centre — c'est
        le bras de la croix. ``W_b = R(π/2)`` réoriente le bras d'un quart de tour. ``Flip``
        et ``R(π/2)`` sont les générateurs du groupe diédral ``D₄``, NON-ABÉLIEN
        (``‖[W_a, W_b]‖ > 0``, valant ici ≈2.5).

        Résultat : la trace passe PLUSIEURS FOIS près du centre (``n_center_passes ≥ 2``,
        ``min_center_ratio`` proche de 0) et se RECROISE abondamment
        (``self_intersections ≥ 1``) — la signature géométrique de la croix. ``|valeur
        propre| < 1`` garde la figure BORNÉE (isométrie de la rotation + contraction du
        flip) : la croix est FERMÉE-stable (contraste avec le polygone-spirale d'``angular``).
        Aucune matrice à λ COMPLEXES ne trace une croix (orbite convexe) : H6c.
        """
        if not (0.0 < contract < 1.0):
            raise ValueError("contract doit être dans (0, 1) (contraction bornée)")
        W_a = torch.tensor([[-contract, 0.0], [0.0, 1.0]], dtype=torch.float32)  # flip x + contraction
        W_b = rotation_matrix(math.pi / 2)
        return cls(W_a, W_b, k_a=1, k_b=1, label="cross_D4")

    @classmethod
    def ngon_vertices(cls, *, n: int = 4) -> "Polyscope2D":
        """CONTRÔLE LINÉAIRE : n-gone DE SOMMETS = R(2π/n) PURE, 1 pas/sommet (H6a).

        Une SEULE matrice de rotation discrète : chaque pas est un sommet du n-gone. C'est
        LINÉAIRE (une rotation à λ complexes), donc ne teste PAS l'axiome 1 — exactement ce
        que H6a prédit. On le code comme polyscope avec ``W_b = I`` et ``k_b = 0`` : il n'y a
        qu'un opérateur effectif ⇒ ``‖[W_a, W_b]‖ = 0`` (commute avec l'identité). Chaque pas
        est un « coin » (κ = 2π/n constant) MAIS il n'y a PAS d'arête droite multi-pas :
        ``straight_edge_fraction ≈ 0`` et cornerness ≈ 0 (P95 = médiane, courbure uniforme).
        """
        W_a = rotation_matrix(2.0 * math.pi / n)
        W_b = torch.eye(2, dtype=torch.float32)
        return cls(W_a, W_b, k_a=1, k_b=0, label=f"ngon_vertices_n{n}")

    @classmethod
    def commuting(
        cls,
        *,
        gain: float = 1.1,
        axis: Optional[float] = None,
        k_edge: int = 3,
    ) -> "Polyscope2D":
        """CONTRÔLE ‖[·,·]‖=0 : MÊME programme qu':meth:`angular`, mais W_a, W_b COMMUTENT.

        On garde le MÊME geste (deux cisaillements anisotropes, même ``gain``, même
        ``k_edge``) mais on aligne les DEUX axes sur la MÊME direction ``axis`` : ``W_a`` et
        ``W_b`` partagent alors leurs vecteurs propres ⇒ ils COMMUTENT exactement
        (``‖[W_a, W_b]‖ = 0``). La composition se réduit à un SEUL cisaillement
        ``shear(gain)^total`` : la direction propre dominante ne change JAMAIS ⇒ pas de
        braquage ⇒ la cornerness doit S'EFFONDRER au niveau orbite-lisse (``C ≈ 0``,
        ``straight_edge_fraction ≈ 1``). C'est l'anti-artefact DUR du Tour 6 (parallèle au
        CTRL D=L du Tour 1 : à commutateur nul exact, l'effet d'intérêt disparaît).
        """
        if axis is None:
            axis = cls.ANGULAR_AXIS_A
        W_a = anisotropic_shear(gain, axis_angle=axis)
        W_b = anisotropic_shear(gain, axis_angle=axis)  # MÊME axe ⇒ commute
        return cls(W_a, W_b, k_a=k_edge, k_b=k_edge, label=f"commuting_g{gain}")

    # -- diagnostic non-commutativité (réveil de commutator_norm) -------------

    def commutator(self) -> float:
        """``‖[W_a, W_b]‖`` (Frobenius) — covariable axiome 1, via ``commutator_norm`` (dormant)."""
        return float(commutator_norm(self.W_a, self.W_b))

    # -- déroulé --------------------------------------------------------------

    def _which(self, t: int) -> torch.Tensor:
        """Opérateur actif au pas ``t`` selon le programme d'alternance (période k_a+k_b)."""
        period = self.cfg.k_a + self.cfg.k_b
        phase = t % period
        return self.W_a if phase < self.cfg.k_a else self.W_b

    @torch.no_grad()
    def trace(self, s0: torch.Tensor, *, steps: int = 60) -> torch.Tensor:
        """Déroule ``steps`` pas d'alternance et retourne la TRACE ``(steps+1, 2)`` (incluant s0).

        ``@torch.no_grad`` à dessein : chemin de DIAGNOSTIC (on mesure une figure). Pour un
        déroulé différentiable, voir :meth:`forward` (mode batch).
        """
        if s0.dim() != 1 or s0.size(0) != 2:
            raise ValueError("s0 doit être de forme (2,)")
        cur = s0
        pts: List[torch.Tensor] = [cur]
        for t in range(steps):
            cur = cur @ self._which(t).t()
            pts.append(cur)
        return torch.stack(pts, dim=0)

    def forward(
        self, s0: torch.Tensor, *, steps: int = 60, return_trace: bool = False
    ):
        """Déroule la dynamique. Accepte ``(2,)`` ou ``(B, 2)`` (différentiable en batch).

        Avec ``return_trace=True`` et entrée ``(2,)``, retourne ``(s_final, trace)``.
        Le mode batch (état final) reste différentiable : la composition de cartes
        linéaires laisse remonter le gradient (test de flux de gradient).
        """
        single = s0.dim() == 1
        if single:
            tr = self.trace(s0, steps=steps)
            return (tr[-1], tr) if return_trace else tr[-1]
        if s0.size(-1) != 2:
            raise ValueError("dernière dim doit valoir 2")
        cur = s0
        for t in range(steps):
            cur = cur @ self._which(t).t()
        return cur
