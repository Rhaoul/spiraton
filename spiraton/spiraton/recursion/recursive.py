from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn


CombineMode = Literal["concat", "add"]
UpdateMode = Literal["residual", "gated", "momentum"]


@dataclass(frozen=True)
class RecursiveConfig:
    x_size: int
    state_size: int
    steps_default: int = 2
    combine: CombineMode = "concat"
    update: UpdateMode = "residual"
    alpha: float = 0.1  # residual step
    beta: float = 0.9   # momentum decay


class RecursiveSpiraton(nn.Module):
    """
    A → B → A' recursion wrapper around a Spiraton-like cell.

    - Maintains an internal state vector A (state).
    - Combines (state, x) into an input vector B for `cell`.
    - Updates state into A' according to an update rule.

    Requirements:
    - `cell(u)` must return a scalar per sample: shape (B,)
      This matches SpiratonCell / GatedSpiratonCell.
    """

    def __init__(
        self,
        cell: nn.Module,
        *,
        x_size: int,
        state_size: int,
        combine: CombineMode = "concat",
        update: UpdateMode = "residual",
        alpha: float = 0.1,
        beta: float = 0.9,
        steps_default: int = 2,
    ) -> None:
        super().__init__()
        self.cell = cell
        self.cfg = RecursiveConfig(
            x_size=x_size,
            state_size=state_size,
            steps_default=steps_default,
            combine=combine,
            update=update,
            alpha=alpha,
            beta=beta,
        )

        # If combine == "add", we need x -> state projection
        self.x_to_state: Optional[nn.Module]
        if self.cfg.combine == "add":
            self.x_to_state = nn.Linear(self.cfg.x_size, self.cfg.state_size, bias=False)
        else:
            self.x_to_state = None

        # y is scalar -> project to state space
        self.y_to_state = nn.Linear(1, self.cfg.state_size, bias=False)

        # gated update uses a gate on state
        self.gate: Optional[nn.Module]
        if self.cfg.update == "gated":
            self.gate = nn.Linear(self.cfg.state_size, self.cfg.state_size, bias=True)
        else:
            self.gate = None

        # momentum buffer will be created on first forward per batch/device
        self.register_buffer("_v", torch.zeros(0, self.cfg.state_size), persistent=False)

    def init_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch, self.cfg.state_size, device=device, dtype=dtype)

    def _combine(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.cfg.combine == "concat":
            return torch.cat([state, x], dim=-1)
        # add
        assert self.x_to_state is not None
        return state + self.x_to_state(x)

    def _ensure_momentum(self, batch: int, device, dtype) -> None:
        if self._v.numel() == 0 or self._v.size(0) != batch or self._v.device != device or self._v.dtype != dtype:
            self._v = torch.zeros(batch, self.cfg.state_size, device=device, dtype=dtype)

    def _update_state(self, state: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: (B,) -> (B,1) -> (B,state)
        y_state = self.y_to_state(y.unsqueeze(-1))

        if self.cfg.update == "residual":
            return state + (self.cfg.alpha * y_state)

        if self.cfg.update == "gated":
            assert self.gate is not None
            g = torch.sigmoid(self.gate(state))
            return g * state + (1.0 - g) * y_state

        # momentum
        self._ensure_momentum(state.size(0), state.device, state.dtype)
        self._v = self.cfg.beta * self._v + (1.0 - self.cfg.beta) * y_state
        return state + self._v

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        return_trace: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        x: (B, x_size) or (x_size,)
        state: optional (B, state_size) or (state_size,)
        steps: number of unrolled recursion steps
        return_trace: if True, returns (y, state, trace)
        """
        squeeze = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True

        if x.size(-1) != self.cfg.x_size:
            raise ValueError(f"x last dim must be x_size={self.cfg.x_size}")

        B = x.size(0)

        if state is None:
            state = self.init_state(B, device=x.device, dtype=x.dtype)
        else:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.size(0) != B or state.size(-1) != self.cfg.state_size:
                raise ValueError("state must be (B, state_size) with same batch size as x")

        K = steps if steps is not None else self.cfg.steps_default
        if K < 1:
            raise ValueError("steps must be >= 1")

        trace_states: List[torch.Tensor] = []
        trace_outputs: List[torch.Tensor] = []

        y = torch.zeros(B, device=x.device, dtype=x.dtype)

        for _ in range(K):
            u = self._combine(x, state)
            y = self.cell(u)  # expected (B,)
            state = self._update_state(state, y)

            if return_trace:
                trace_states.append(state)
                trace_outputs.append(y)

        if squeeze:
            y = y.squeeze(0)
            state = state.squeeze(0)

        if not return_trace:
            return y

        trace: Dict[str, Any] = {
            "states": trace_states,
            "outputs": trace_outputs,
            "steps": K,
            "combine": self.cfg.combine,
            "update": self.cfg.update,
            "alpha": self.cfg.alpha,
            "beta": self.cfg.beta,
        }
        return y, state, trace

