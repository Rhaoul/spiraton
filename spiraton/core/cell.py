from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from .operators import additive, subtractive, multiplicative, divisive
from .modes import dextro_mask
from .mode_policy import ModePolicy


@dataclass
class SpiratonCoreConfig:
    input_size: int
    init_scale: float = 0.1
    eps: float = 1e-6


class SpiratonCell(nn.Module):
    """
    SpiratonCell (CORE)
    - Canon operators: add/sub/mul/div
    - Mode selection:
        - default: dextro_mask(inputs) (mean>=0)
        - optional: mode_policy(inputs) returning hard mask (bool) or soft gate (float in [0,1])
    """

    def __init__(
        self,
        input_size: int,
        init_scale: float = 0.1,
        eps: float = 1e-6,
        mode_policy: Optional[ModePolicy] = None,
    ) -> None:
        super().__init__()
        self.cfg = SpiratonCoreConfig(input_size=input_size, init_scale=init_scale, eps=eps)
        self.mode_policy = mode_policy

        self.w_add = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_sub = nn.Parameter(torch.randn(input_size) * init_scale)
        self.w_mul = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))
        self.w_div = nn.Parameter(torch.randn(input_size) * (init_scale / 2.0))

        self.bias = nn.Parameter(torch.zeros(1))

    def _mode(self, inputs: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        if self.mode_policy is None:
            return dextro_mask(inputs)
        return self.mode_policy(inputs)

    @staticmethod
    def _is_soft(mode: torch.Tensor) -> bool:
        # bool => hard, float => soft
        return mode.dtype != torch.bool

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze = True

        if inputs.size(-1) != self.cfg.input_size:
            raise ValueError(f"last dim must equal input_size={self.cfg.input_size}")

        mode = self._mode(inputs)

        # Operators
        add = additive(inputs, self.w_add)
        sub = subtractive(inputs, self.w_sub)

        log_mul = multiplicative(inputs, self.w_mul, eps=self.cfg.eps)
        log_div = divisive(inputs, self.w_div, eps=self.cfg.eps)

        mul = torch.tanh(log_mul)
        div = torch.tanh(log_div)

        raw_dextro = add + mul - div
        raw_levogyre = sub + div - mul

        if self._is_soft(mode):
            gate = torch.clamp(mode, 0.0, 1.0).to(dtype=raw_dextro.dtype)
            raw = gate * raw_dextro + (1.0 - gate) * raw_levogyre
            z = raw + self.bias
            # soft activation mix (keeps differentiability)
            out = gate * torch.tanh(z) + (1.0 - gate) * torch.atan(z)
        else:
            m = mode
            raw = torch.where(m, raw_dextro, raw_levogyre)
            z = raw + self.bias
            out = torch.where(m, torch.tanh(z), torch.atan(z))

        return out.squeeze(0) if squeeze else out

