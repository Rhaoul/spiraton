from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AlphaOmegaLossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


def copy_rate(pred: torch.Tensor, a: torch.Tensor, *, tol: float = 1e-3) -> float:
    """Fraction d'échantillons où A′ prédit est (quasi) une copie exacte de A.

    La copie exacte est le mode dégénéré (répétition) que le projet refuse :
    on la mesure pour pouvoir la surveiller, pas seulement la pénaliser.
    """
    rel = (pred - a).norm(dim=-1) / (a.norm(dim=-1) + 1e-8)
    return float((rel < tol).float().mean().item())


def alpha_omega_loss(
    pred_aprime: torch.Tensor,
    a: torch.Tensor,
    *,
    target_dist: float = 0.3,
    w_align: float = 1.0,
    w_dist: float = 1.0,
    eps: float = 1e-8,
) -> AlphaOmegaLossOutput:
    """Perte de clôture spirale : A′ *proche-et-aligné* avec A, mais non identique.

    Traduction opératoire du diagnostic central du projet (CLAUDE.md,
    ``run_alpha_omega_spatial`` : « le retour proche et aligné mais non
    identique ») en une perte d'entraînement (chantier 5). Trois exigences,
    deux termes :

    1. **Aligné** : la direction de A′ doit suivre celle de A
       → ``align = mean(1 − cos(A′, A))`` (minimal quand colinéaires).
    2. **Proche mais transformé** : la distance relative ``‖A′−A‖/‖A‖`` doit
       viser une *bande* ``target_dist > 0`` — ni 0 (copie : la répétition,
       mode dégénéré), ni grande (le système se perd)
       → ``band = mean((rel_dist − target_dist)²)``.

    La tension entre « proche » (petite distance) et « non identique »
    (distance ≥ cible) est *voulue* : elle pousse vers le retour transformé.
    ``target_dist`` est l'ampleur de transformation recherchée.

    Retourne la perte scalaire et des métriques (alignement moyen, distance
    relative moyenne, taux de copie) pour le suivi.
    """
    if pred_aprime.shape != a.shape:
        raise ValueError(f"formes incompatibles : {tuple(pred_aprime.shape)} vs {tuple(a.shape)}")

    cos = F.cosine_similarity(pred_aprime, a, dim=-1, eps=eps)  # (N,)
    align = (1.0 - cos).mean()

    rel_dist = (pred_aprime - a).norm(dim=-1) / (a.norm(dim=-1) + eps)  # (N,)
    band = ((rel_dist - target_dist) ** 2).mean()

    loss = w_align * align + w_dist * band

    metrics = {
        "loss": float(loss.item()),
        "align": float(align.item()),          # 0 = parfaitement aligné
        "mean_rel_dist": float(rel_dist.mean().item()),
        "target_dist": float(target_dist),
        "copy_rate": copy_rate(pred_aprime.detach(), a.detach()),
    }
    return AlphaOmegaLossOutput(loss=loss, metrics=metrics)
