"""Observables effectives d'une trajectoire dans un champ de référence.

Définitions opérationnelles (protocole ``docs/SEMANTIC_THERMO_PROTOCOLE.md`` §2).
Un « champ » est un nuage fixe R de points (lignes uniques) ; une
« trajectoire » est une suite ordonnée de points x_1 … x_n (un segment).

Par point x, avec d_1 ≤ … ≤ d_k ses distances aux k plus proches voisins de R :

    densité      ρ(x) = 1 / (mean_j d_j + eps)
    entropie     S(x) = −Σ_j p_j log p_j,  p = softmax(−d / h)   (h gelé)
    température  T(x) = Var_j(d_j)

Par trajectoire :

    vitesse          v_t = x_t − x_{t−1}
    flux             J_t = ρ_t v_t
    divergence       div = − pente OLS de ln ρ_t contre t
                     (continuité lagrangienne : d ln ρ/dt = −∇·v le long d'une
                     trajectoire ; > 0 ⇔ la trajectoire gagne des régions
                     plus rares du champ = expansion)
    divergence^J     moyenne_t (J_t − J_{t−1})·û, û = direction x_1 → x_n
                     (définition littérale du doc §9.10, rapportée en secondaire)
    dispersion       moyenne_t ‖x_t − centroïde‖
    retour           ‖x_n − x_1‖ / longueur du chemin  (0 = boucle refermée)

Aucune de ces quantités n'est présumée physique. Le module ne connaît aucune
étiquette ABA.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .state import SemanticThermoState

# Distances exactes (pas de la forme ‖x‖²+‖y‖²−2xy, bruitée près de 0) : les
# invariances (rotation, CTRL-7) doivent tenir au bruit flottant près.
_EXACT = "donot_use_mm_for_euclid_dist"


# ---------------------------------------------------------------------------
# Champ de référence
# ---------------------------------------------------------------------------

class ReferenceField:
    """Nuage de référence R (lignes uniques) + bande passante h gelée.

    ``h`` : si non fourni, médiane sur R de la distance moyenne aux k voisins
    (en excluant le point lui-même). Recalculé à partir de R seul, donc
    covariant avec toute isométrie ou homothétie appliquée à R.
    """

    def __init__(
        self,
        points: torch.Tensor,
        *,
        k: int = 8,
        eps: float = 1e-8,
        h: Optional[float] = None,
        dedupe: bool = True,
    ) -> None:
        pts = torch.as_tensor(points, dtype=torch.float64)
        if pts.ndim != 2 or pts.size(0) < 2:
            raise ValueError("le champ doit être une matrice (N ≥ 2, d)")
        if dedupe:
            pts = torch.unique(pts, dim=0)
        if pts.size(0) <= k:
            raise ValueError(f"champ trop petit ({pts.size(0)} points) pour k={k}")
        self.points = pts
        self.k = int(k)
        self.eps = float(eps)
        if h is None:
            # k+1 voisins de chaque point de R, le premier étant lui-même (distance 0,
            # lignes uniques) ; par blocs pour borner la mémoire.
            knn = self._knn(pts, self.k + 1)[:, 1:]
            h = float(knn.mean(dim=1).median())
        self.h = float(h)
        if not (self.h > 0 and math.isfinite(self.h)):
            raise ValueError(f"bande passante h invalide : {self.h}")

    @property
    def dim(self) -> int:
        return int(self.points.size(1))

    def _knn(self, x: torch.Tensor, k: int, block: int = 1024) -> torch.Tensor:
        out = [torch.cdist(x[i:i + block], self.points, compute_mode=_EXACT)
               .topk(k, dim=1, largest=False).values
               for i in range(0, x.size(0), block)]
        return torch.cat(out) if out else torch.zeros(0, k, dtype=torch.float64)

    def knn_distances(self, x: torch.Tensor) -> torch.Tensor:
        """(n, k) distances triées aux k plus proches voisins de R."""
        x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, self.dim)
        return self._knn(x, self.k)

    def local(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Observables locales (ρ, S, T) de chaque ligne de x."""
        d = self.knn_distances(x)
        density = 1.0 / (d.mean(dim=1) + self.eps)
        p = torch.softmax(-d / self.h, dim=1)
        entropy = -(p * torch.log(p.clamp_min(1e-300))).sum(dim=1)
        temperature = d.var(dim=1, unbiased=False)
        return {"density": density, "entropy": entropy, "temperature": temperature}


# ---------------------------------------------------------------------------
# Trajectoire
# ---------------------------------------------------------------------------

def semantic_velocity(x: torch.Tensor) -> torch.Tensor:
    """(n−1, d) : v_t = x_t − x_{t−1}."""
    x = torch.as_tensor(x, dtype=torch.float64)
    return x[1:] - x[:-1]


def semantic_flux(density: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """J_t = ρ_t v_t, avec ρ pris au point d'arrivée du pas (n−1, d)."""
    return density[1:, None] * velocity


def continuity_divergence(density: torch.Tensor) -> float:
    """div = − pente OLS de ln ρ_t contre t = 0 … n−1. 0.0 si n < 2."""
    n = int(density.numel())
    if n < 2:
        return 0.0
    y = torch.log(density.to(torch.float64))
    t = torch.arange(n, dtype=torch.float64)
    tc = t - t.mean()
    slope = float((tc * (y - y.mean())).sum() / (tc * tc).sum())
    return -slope


def literal_flux_divergence(x: torch.Tensor, density: torch.Tensor) -> float:
    """Doc §9.10 : moyenne_t (J_t − J_{t−1})·û, û = (x_n − x_1)/‖·‖. 0.0 si n < 3."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.size(0) < 3:
        return 0.0
    disp = x[-1] - x[0]
    norm = float(disp.norm())
    if norm == 0.0:
        return 0.0
    u = disp / norm
    J = semantic_flux(density, semantic_velocity(x))
    return float(((J[1:] - J[:-1]) @ u).mean())


def return_ratio(x: torch.Tensor) -> float:
    """‖x_n − x_1‖ / longueur du chemin ; 0 = boucle refermée, 1 = ligne droite."""
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.size(0) < 2:
        return 0.0
    path = float(semantic_velocity(x).norm(dim=1).sum())
    if path == 0.0:
        return 0.0
    return float((x[-1] - x[0]).norm()) / path


def segment_observables(
    x: torch.Tensor, field: ReferenceField, *, label: Optional[str] = None
) -> SemanticThermoState:
    """Agrège une trajectoire (n ≥ 1, d) en ``SemanticThermoState``."""
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, field.dim)
    n = int(x.size(0))
    if n == 0:
        raise ValueError("trajectoire vide")
    loc = field.local(x)
    rho = loc["density"]
    if n >= 2:
        v = semantic_velocity(x)
        speeds = v.norm(dim=1)
        speed = float(speeds.mean())
        flux_norm = float(semantic_flux(rho, v).norm(dim=1).mean())
        energy = float((0.5 * speeds ** 2).mean())
    else:
        speed = flux_norm = energy = 0.0
    dispersion = float((x - x.mean(dim=0)).norm(dim=1).mean())
    return SemanticThermoState(
        density=float(rho.mean()),
        entropy=float(loc["entropy"].mean()),
        temperature=float(loc["temperature"].mean()),
        speed=speed,
        flux_norm=flux_norm,
        divergence=continuity_divergence(rho),
        dispersion=dispersion,
        energy=energy,
        n_tokens=n,
        phase_label=label,
    )
