from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .operators import additive, subtractive, multiplicative, divisive
from .modes import dextro_mask


@dataclass
class SpiratonCoreConfig:
    """
    Core config = contrat stable.
    - pas de gates
    - pas de coeffs globaux learnés
    - pas d'heuristique d'adaptation
    """
    input_size: int
    init_scale: float = 0.1
    eps: float = 1e-6


class SpiratonCell(nn.Module):
    """
    SpiratonCell (CORE)
    - Définition canonique des 4 opérateurs (add/sub/mul/div)
    - Mode dextro/levogyre basé sur dextro_mask(inputs)
    - Composition symétrique minimale
    - Activations dépendantes du mode (tanh vs atan)
    """

    def __init__(self, input_size: int, init_scale: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.cfg = SpiratonCoreConfig(input_size=input_size, init_scale=init_scale, eps=eps)

        # Poids opérateurs (canon)
        self.w_add = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_sub = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_mul = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))
        self.w_div = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))

        # Bias (canon)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (..., input_size) ou (input_size,)
        returns: (...,) scalaire par vecteur d'entrée
        """
        squeeze = False
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze = True

        if inputs.size(-1) != self.cfg.input_size:
            raise ValueError(f"last dim must equal input_size={self.cfg.input_size}")

        # Mode
        m = dextro_mask(inputs)  # (batch,) bool

        # Opérateurs canoniques
        add = additive(inputs, self.w_add)
        sub = subtractive(inputs, self.w_sub)
        log_mul = multiplicative(inputs, self.w_mul, eps=self.cfg.eps)
        log_div = divisive(inputs, self.w_div, eps=self.cfg.eps)

        # Stabilisation minimale (core)
        mul = torch.tanh(log_mul)
        div = torch.tanh(log_div)

        # Composition canon (simple, symétrique)
        raw_dextro = add + mul - div
        raw_levogyre = sub + div - mul
        raw = torch.where(m, raw_dextro, raw_levogyre)

        z = raw + self.bias

        # Activation dépendante du mode
        out = torch.where(m, torch.tanh(z), torch.atan(z))

        return out.squeeze(0) if squeeze else out
