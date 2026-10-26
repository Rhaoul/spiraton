from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .aba_loss import alpha_omega_loss
from ..data.loader import AbaTriples, iter_batches


@dataclass(frozen=True)
class TrainReport:
    loss_history: List[float]
    final_metrics: Dict[str, float]

    @property
    def improved(self) -> bool:
        return len(self.loss_history) >= 2 and self.loss_history[-1] < self.loss_history[0]


class AbaPredictor(nn.Module):
    """Prédicteur minimal A,B → A′ (concatène A et B puis projette).

    Volontairement simple : la boucle d'entraînement démontre le *signal*
    alpha-oméga, pas une architecture. Remplaçable par toute cellule mappant
    ``(B, 2·dim) → (B, dim)``.
    """

    def __init__(self, dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        h = hidden if hidden is not None else dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, h),
            nn.Tanh(),
            nn.Linear(h, dim),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([a, b], dim=-1))


def make_aba_predictor(dim: int, hidden: Optional[int] = None) -> AbaPredictor:
    return AbaPredictor(dim, hidden=hidden)


def train_aba(
    model: nn.Module,
    triples: AbaTriples,
    *,
    epochs: int = 100,
    lr: float = 1e-2,
    batch_size: Optional[int] = None,
    target_dist: float = 0.3,
    w_align: float = 1.0,
    w_dist: float = 1.0,
    shuffle_seed: Optional[int] = 0,
) -> TrainReport:
    """Boucle d'entraînement minimale sous perte alpha-oméga (chantier 5).

    ``model(a, b) -> pred_aprime``. La perte pousse A′ prédit à être
    proche-et-aligné avec A sans le copier (cf. :func:`alpha_omega_loss`).

    Déterministe à seed fixée. Retourne l'historique de perte et les dernières
    métriques. Aucune cible chiffrée n'est « visée » : on mesure que le signal
    est apprenable, c'est tout.
    """
    if len(triples) == 0:
        raise ValueError("triples vide")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = batch_size if batch_size is not None else len(triples)

    history: List[float] = []
    last_metrics: Dict[str, float] = {}

    for epoch in range(epochs):
        epoch_losses: List[float] = []
        seed = (shuffle_seed + epoch) if shuffle_seed is not None else None
        for a, b, _a_prime in iter_batches(triples, bs, shuffle_seed=seed):
            pred = model(a, b)
            out = alpha_omega_loss(
                pred, a, target_dist=target_dist, w_align=w_align, w_dist=w_dist
            )
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            epoch_losses.append(out.metrics["loss"])
            last_metrics = out.metrics
        history.append(sum(epoch_losses) / len(epoch_losses))

    return TrainReport(loss_history=history, final_metrics=last_metrics)
