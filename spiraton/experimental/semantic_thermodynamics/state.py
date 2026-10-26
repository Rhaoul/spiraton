"""Dataclasses d'état (doc directeur §8). Immuables, validées finies."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional


def _check_finite(obj) -> None:
    for f in fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"{type(obj).__name__}.{f.name} non fini : {v!r}")


@dataclass(frozen=True)
class ThermoProbeConfig:
    """Hyperparamètres gelés de l'instrument (protocole §2).

    neighborhood_k : nombre de voisins dans le champ de référence (densité,
        température, entropie).
    eps : plancher numérique de la densité.
    normalize : z-score des dimensions sur le pool de calibration.
    seed : graine (partage calibration/test, permutations, bootstrap).
    """

    neighborhood_k: int = 8
    eps: float = 1e-8
    normalize: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.neighborhood_k < 1:
            raise ValueError("neighborhood_k doit être ≥ 1")


@dataclass(frozen=True)
class SemanticThermoState:
    """Agrégat d'un segment (trajectoire de tokens) — variables effectives.

    density, entropy, temperature : moyennes sur les tokens des observables
        locales dans le champ de référence (voir ``observables.ReferenceField``).
    speed : moyenne de ‖x_t − x_{t−1}‖ (0 si un seul token).
    flux_norm : moyenne de ‖ρ_t v_t‖.
    divergence : divergence primaire (équation de continuité), cf.
        ``observables.continuity_divergence`` ; 0 si n < 2.
    dispersion : moyenne des distances des tokens au centroïde du segment.
    energy : énergie cinétique moyenne ½‖v‖².
    n_tokens : longueur de la trajectoire.
    """

    density: float
    entropy: float
    temperature: float
    speed: float
    flux_norm: float
    divergence: float
    dispersion: float
    energy: float
    n_tokens: int
    phase_label: Optional[str] = None

    def __post_init__(self) -> None:
        _check_finite(self)


@dataclass(frozen=True)
class SemanticTransition:
    """Passage d'un segment à un autre (centroïdes) — doc §8.2, sous-ensemble mesuré."""

    src_index: int
    dst_index: int
    delta_density: float
    delta_entropy: float
    displacement: float

    def __post_init__(self) -> None:
        _check_finite(self)
