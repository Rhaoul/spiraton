"""Trajectoires synthétiques de l'expérience E0 (validation instrumentale).

Champ : nuage gaussien isotrope N(0, I_d) seedé. Toutes les trajectoires sont
déterministes à graine fixée. Le « centre » est l'origine, région la plus
dense du champ ; s'en éloigner = aller vers des régions rares (expansion).
"""
from __future__ import annotations

import math
from typing import Dict

import torch


def gaussian_field(n: int = 2000, d: int = 8, *, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def _unit(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(d, generator=g, dtype=torch.float64)
    return u / u.norm()


def radial(d: int, r0: float, r1: float, n: int = 12, *, seed: int = 1) -> torch.Tensor:
    """Ligne radiale de rayon r0 à r1 (expansion si r1 > r0, contraction sinon)."""
    u = _unit(d, seed)
    r = torch.linspace(r0, r1, n, dtype=torch.float64)
    return r[:, None] * u[None, :]


def stationary(d: int, n: int = 12, *, r: float = 1.0, seed: int = 1) -> torch.Tensor:
    return (r * _unit(d, seed)).expand(n, d).clone()


def random_walk(d: int, n: int = 12, *, step: float = 0.3, seed: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    steps = step * torch.randn(n, d, generator=g, dtype=torch.float64) / math.sqrt(d)
    steps[0] = 0.0
    return torch.cumsum(steps, dim=0)


def diffusion(d: int, n: int = 12, *, scale: float = 1.0, seed: int = 3) -> torch.Tensor:
    """Particule diffusante depuis le centre : rayon ∝ √t, direction aléatoire à chaque pas."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n, d, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    r = scale * torch.sqrt(torch.arange(n, dtype=torch.float64))
    return r[:, None] * dirs


def closed_loop(d: int, n: int = 13, *, r: float = 1.5, seed: int = 4) -> torch.Tensor:
    """Cercle de rayon r dans un plan aléatoire ; x_n = x_1 exactement."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, 2, generator=g, dtype=torch.float64))
    th = torch.linspace(0.0, 2 * math.pi, n, dtype=torch.float64)
    x = r * (torch.cos(th)[:, None] * q[:, 0] + torch.sin(th)[:, None] * q[:, 1])
    x[-1] = x[0]
    return x


def e0_suite(d: int = 8) -> Dict[str, torch.Tensor]:
    return {
        "expansion": radial(d, 0.2, 3.0),
        "contraction": radial(d, 3.0, 0.2),
        "stationary": stationary(d),
        "random_walk": random_walk(d),
        "diffusion": diffusion(d),
        "closed_loop": closed_loop(d),
    }
