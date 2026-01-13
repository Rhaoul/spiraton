from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

import torch
import torch.nn as nn


# Hard mode: bool mask (True=dextro, False=levo)
HardMode = torch.Tensor  # dtype=bool, shape (batch,)
# Soft mode: float gate in [0,1], shape (batch,)
SoftMode = torch.Tensor  # dtype=float, shape (batch,)

ModeTensor = Union[HardMode, SoftMode]


class ModePolicy(Protocol):
    """
    A ModePolicy returns either:
      - a hard mask (bool) where True=dextro, False=levogyre
      - OR a soft gate (float in [0,1]) used to mix dextro/levo branches
    """

    def __call__(self, inputs: torch.Tensor) -> ModeTensor:
        ...


@dataclass(frozen=True)
class MeanThreshold:
    """
    Hard mode: dextro iff mean(inputs) >= threshold.
    """
    threshold: float = 0.0

    def __call__(self, inputs: torch.Tensor) -> HardMode:
        return inputs.mean(dim=-1) >= self.threshold


@dataclass(frozen=True)
class EnergyThreshold:
    """
    Hard mode: dextro iff RMS energy >= threshold.
    RMS = sqrt(mean(x^2))
    """
    threshold: float = 1.0

    def __call__(self, inputs: torch.Tensor) -> HardMode:
        rms = inputs.pow(2).mean(dim=-1).sqrt()
        return rms >= self.threshold


class LearnedGateMode(nn.Module):
    """
    Soft mode: learned gate in [0,1].
    gate = sigmoid(sum(inputs * w) + b)

    Notes:
    - returning a soft gate keeps the branch selection differentiable.
    - you can push it to hard with thresholds during evaluation if desired.
    """

    def __init__(self, input_size: int, init_scale: float = 0.01) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(input_size) * init_scale)
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> SoftMode:
        logits = torch.sum(inputs * self.w, dim=-1) + self.b
        return torch.sigmoid(logits)

