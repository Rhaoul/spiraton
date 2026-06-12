from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from spiraton.core.operators import additive, subtractive, multiplicative, divisive
from spiraton.core.modes import dextro_mask
from spiraton.core.mode_policy import ModePolicy
from .adaptation import SecondOrderAdaptation


class GatedSpiratonCell(nn.Module):
    """
    Experimental gated cell:
    - log-domain mul/div + tanh stabilization
    - learned gates per operator
    - learned global coeffs (sigmoid)
    - optional mode_policy: hard mask or soft gate
    """

    def __init__(
        self,
        input_size: int,
        init_scale: float = 0.1,
        eps: float = 1e-6,
        mode_policy: Optional[ModePolicy] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.eps = eps
        self.mode_policy = mode_policy

        self.w_add = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_sub = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_mul = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))
        self.w_div = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))

        self.w_gate_mul = nn.Parameter(torch.zeros(input_size))
        self.b_gate_mul = nn.Parameter(torch.tensor(0.0))
        self.w_gate_div = nn.Parameter(torch.zeros(input_size))
        self.b_gate_div = nn.Parameter(torch.tensor(0.0))

        self.coeffs = nn.Parameter(torch.zeros(4))

        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("adaptation", torch.tensor(0.1))

        self._adapt_rule = SecondOrderAdaptation()

    def _mode(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode_policy is None:
            return dextro_mask(inputs)
        return self.mode_policy(inputs)

    @staticmethod
    def _is_soft(mode: torch.Tensor) -> bool:
        return mode.dtype != torch.bool

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze = True

        if inputs.size(-1) != self.input_size:
            raise ValueError(f"last dim must equal input_size={self.input_size}")

        mode = self._mode(inputs)

        add = additive(inputs, self.w_add)
        sub = subtractive(inputs, self.w_sub)

        log_mul = multiplicative(inputs, self.w_mul, eps=self.eps)
        log_div = divisive(inputs, self.w_div, eps=self.eps)

        gate_mul = torch.sigmoid(torch.sum(inputs * self.w_gate_mul, dim=-1) + self.b_gate_mul)
        gate_div = torch.sigmoid(torch.sum(inputs * self.w_gate_div, dim=-1) + self.b_gate_div)

        mul = torch.tanh(log_mul) * gate_mul
        div = torch.tanh(log_div) * gate_div

        c_add, c_sub, c_mul, c_div = torch.sigmoid(self.coeffs)

        raw_dextro = c_add * add + c_mul * mul - c_div * div
        raw_levogyre = c_sub * sub + c_div * div - c_mul * mul

        if self._is_soft(mode):
            g = torch.clamp(mode, 0.0, 1.0).to(dtype=raw_dextro.dtype)
            raw = g * raw_dextro + (1.0 - g) * raw_levogyre
            z = raw + self.bias
            out = g * torch.tanh(z) + (1.0 - g) * torch.atan(z)
        else:
            m = mode
            raw = torch.where(m, raw_dextro, raw_levogyre)
            z = raw + self.bias
            out = torch.where(m, torch.tanh(z), torch.atan(z))

        return out.squeeze(0) if squeeze else out

    def second_order_adjust(self, error: torch.Tensor) -> float:
        return self._adapt_rule.step(self.adaptation, error)

