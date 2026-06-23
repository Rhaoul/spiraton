from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ChronoConfig:
    state_size: int
    init_scale: float = 0.1
    bounded: bool = False   # tanh terminal pour stabiliser (s'écarte de l'équation pure)


class ChronoSpiraton(nn.Module):
    """Dynamique du second ordre du Logos (THEORIE_LOGOS §3.2, chantier 6).

    Incarne *exactement* l'équation fondatrice, là où ``RecursiveSpiraton`` est
    de premier ordre :

        s_{t+1} = D( A(s_t) + B(s_t²) − C(s_{t−1}) ) + L(s_t)

    Deux nouveautés par rapport au premier ordre :

    1. **Mémoire explicite** de l'état précédent ``s_{t−1}`` via le terme
       inhibiteur ``−C(s_{t−1})`` : la durée entre dans la mise à jour, pas
       seulement l'instant courant.
    2. **Terme quadratique** ``B(s_t²)`` (carré élément-à-élément, puis
       application linéaire B) : la dynamique est non linéaire au sens propre.

    Les cinq applications sont linéaires (sans biais), conformément à leur statut
    d'opérateurs : ``A`` agrégation, ``B`` amplification, ``C`` inhibition,
    ``D`` dextrogyre (action), ``L`` lévogyre (mémoire). D et L sont des
    paramètres distincts — ils ne sont pas inverses l'un de l'autre (axiome §2.3).

    Stabilité : l'équation pure (``bounded=False``) est non bornée et peut
    diverger ; c'est attendu (chaos déterministe possible, cf. §3.2). Pour les
    usages nécessitant une trajectoire bornée, ``bounded=True`` applique un
    ``tanh`` terminal — au prix d'une fidélité moindre à l'équation. Voir
    :meth:`stability_scan` et ``examples/chrono_stability.py``.
    """

    def __init__(self, state_size: int, *, init_scale: float = 0.1, bounded: bool = False) -> None:
        super().__init__()
        self.cfg = ChronoConfig(state_size=state_size, init_scale=init_scale, bounded=bounded)

        def op() -> nn.Linear:
            lin = nn.Linear(state_size, state_size, bias=False)
            with torch.no_grad():
                lin.weight.mul_(init_scale)
            return lin

        self.A = op()  # agrégation
        self.B = op()  # amplification (sur le carré)
        self.C = op()  # inhibition (sur la mémoire s_{t-1})
        self.D = op()  # dextrogyre (action)
        self.L = op()  # lévogyre (mémoire)

    def step(self, s_t: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        """Un pas de la récurrence du second ordre. Formes ``(B, d)`` ou ``(d,)``."""
        inner = self.A(s_t) + self.B(s_t * s_t) - self.C(s_prev)
        s_next = self.D(inner) + self.L(s_t)
        if self.cfg.bounded:
            s_next = torch.tanh(s_next)
        return s_next

    def forward(
        self,
        s0: torch.Tensor,
        *,
        steps: int = 4,
        s_prev: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        """Déroule la dynamique ``steps`` pas à partir de ``s0``.

        s_prev : état précédent initial s_{−1}. Par défaut, le repos (zéros) :
            au premier pas, le terme de mémoire ``−C(s_{t−1})`` est nul.
        return_trace : si True, retourne ``(s_final, trace)`` où ``trace`` liste
            les états s_1..s_steps.
        """
        if steps < 1:
            raise ValueError("steps must be >= 1")

        squeeze = False
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
            squeeze = True
        if s0.size(-1) != self.cfg.state_size:
            raise ValueError(f"dernière dim doit être state_size={self.cfg.state_size}")

        if s_prev is None:
            s_prev = torch.zeros_like(s0)
        elif s_prev.dim() == 1:
            s_prev = s_prev.unsqueeze(0)

        prev, cur = s_prev, s0
        trace: List[torch.Tensor] = []
        for _ in range(steps):
            nxt = self.step(cur, prev)
            trace.append(nxt)
            prev, cur = cur, nxt

        out = cur.squeeze(0) if squeeze else cur
        if not return_trace:
            return out
        if squeeze:
            trace = [t.squeeze(0) for t in trace]
        return out, trace

    # -- diagnostics de stabilité -----------------------------------------

    @torch.no_grad()
    def stability_scan(
        self,
        s0: torch.Tensor,
        *,
        steps: int = 50,
        s_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Mesure la croissance de la norme de l'état le long d'un déroulé.

        Retourne ``{final_norm, max_norm, diverged}`` — benchmark obligatoire
        pour cette cellule expérimentale (CLAUDE.md, chantier 6).
        """
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        if s_prev is None:
            s_prev = torch.zeros_like(s0)

        prev, cur = s_prev, s0
        max_norm = float(cur.norm(dim=-1).max().item())
        diverged = False
        for _ in range(steps):
            cur, prev = self.step(cur, prev), cur
            n = float(cur.norm(dim=-1).max().item())
            max_norm = max(max_norm, n)
            if not torch.isfinite(cur).all() or n > 1e6:
                diverged = True
                break
        return {
            "final_norm": float(cur.norm(dim=-1).max().item()),
            "max_norm": max_norm,
            "diverged": diverged,
        }
