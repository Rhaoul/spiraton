from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SecondOrderAdaptation:
    """
    Règle simple : adapte un scalaire 'adaptation' en fonction de la tendance d'erreur.
    - si l'erreur augmente -> on diminue l'adaptation
    - sinon -> on augmente doucement
    """
    init: float = 0.1
    min_value: float = 0.001
    max_value: float = 0.2
    down_factor: float = 0.9
    up_factor: float = 1.05
    tol: float = 1e-6

    _last_error: Optional[torch.Tensor] = None

    def step(self, adaptation: torch.Tensor, error: torch.Tensor) -> float:
        """
        adaptation: tensor buffer scalaire (ex: register_buffer("adaptation", tensor(...)))
        error: tensor quelconque
        return: nouvelle valeur float
        """
        with torch.no_grad():
            current = error.detach().abs().mean().cpu()
            if self._last_error is None:
                self._last_error = current.detach().clone()
                adaptation.fill_(self.init)
                return float(adaptation.item())

            prev = float(self._last_error.item())
            if float(current) > prev + self.tol:
                new = max(self.min_value, float(adaptation.item()) * self.down_factor)
            else:
                new = min(self.max_value, float(adaptation.item()) * self.up_factor)

            adaptation.fill_(new)
            self._last_error = current.detach().clone()
            return float(new)
