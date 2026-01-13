from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from spiraton.core.operators import additive, subtractive, multiplicative, divisive
from spiraton.core.modes import dextro_mask
from .adaptation import SecondOrderAdaptation


class GatedSpiratonCell(nn.Module):
    """
    Version EXPERIMENTAL (ma version actuelle) :
    - mul/div en log-domain (operators.multiplicative/divisive)
    - gates par opérateur (sigmoid)
    - coeffs globaux (sigmoid) pour pondérer add/sub/mul/div
    - buffer adaptation + règle second-order séparée
    """

    def __init__(self, input_size: int, init_scale: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.input_size = input_size
        self.eps = eps

        # Poids opérateurs
        self.w_add = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_sub = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_mul = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))
        self.w_div = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))

        # Gates (projections simples)
        self.w_gate_mul = nn.Parameter(torch.zeros(input_size))
        self.b_gate_mul = nn.Parameter(torch.tensor(0.0))
        self.w_gate_div = nn.Parameter(torch.zeros(input_size))
        self.b_gate_div = nn.Parameter(torch.tensor(0.0))

        # Coeffs globaux (add, sub, mul, div)
        self.coeffs = nn.Parameter(torch.zeros(4))

        # Bias + adaptation
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("adaptation", torch.tensor(0.1))

        self._adapt_rule = SecondOrderAdaptation()
        self._last_error: Optional[torch.Tensor] = None  # gardé si tu veux debug local

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze = True

        if inputs.size(-1) != self.input_size:
            raise ValueError(f"last dim must equal input_size={self.input_size}")

        # Mode
        m = dextro_mask(inputs)

        # Opérateurs canoniques
        add = additive(inputs, self.w_add)
        sub = subtractive(inputs, self.w_sub)

        log_mul = multiplicative(inputs, self.w_mul, eps=self.eps)
        log_div = divisive(inputs, self.w_div, eps=self.eps)

        # Gates
        gate_mul = torch.sigmoid(torch.sum(inputs * self.w_gate_mul, dim=-1) + self.b_gate_mul)
        gate_div = torch.sigmoid(torch.sum(inputs * self.w_gate_div, dim=-1) + self.b_gate_div)

        # Stabilisation + gating
        mul = torch.tanh(log_mul) * gate_mul
        div = torch.tanh(log_div) * gate_div

        # Coeffs
        c_add, c_sub, c_mul, c_div = torch.sigmoid(self.coeffs)

        # Compose
        raw_dextro = c_add * add + c_mul * mul - c_div * div
        raw_levogyre = c_sub * sub + c_div * div - c_mul * mul
        raw = torch.where(m, raw_dextro, raw_levogyre)

        z = raw + self.bias
        out = torch.where(m, torch.tanh(z), torch.atan(z))

        return out.squeeze(0) if squeeze else out

    def second_order_adjust(self, error: torch.Tensor) -> float:
        """
        Compat API: délègue à SecondOrderAdaptation (experimental/adaptation.py)
        """
        return self._adapt_rule.step(self.adaptation, error)
