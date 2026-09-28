"""Thermodynamique sémantique — instrument expérimental, NON canonique.

Document directeur : ``brainstorming/SEMANTIC_THERMODYNAMICS.md`` ; protocole
pré-enregistré : ``docs/SEMANTIC_THERMO_PROTOCOLE.md``.

Les noms (densité, entropie, température, flux, divergence) sont des
**variables effectives** définies opérationnellement dans ``observables.py`` ;
aucune n'est présentée comme thermodynamique au sens physique. Aucune
dépendance aux étiquettes ABA : ce module ne voit que des vecteurs.
"""
from .state import SemanticThermoState, SemanticTransition, ThermoProbeConfig
from .observables import (
    ReferenceField,
    segment_observables,
    semantic_velocity,
    semantic_flux,
    continuity_divergence,
    literal_flux_divergence,
    return_ratio,
)

__all__ = [
    "SemanticThermoState",
    "SemanticTransition",
    "ThermoProbeConfig",
    "ReferenceField",
    "segment_observables",
    "semantic_velocity",
    "semantic_flux",
    "continuity_divergence",
    "literal_flux_divergence",
    "return_ratio",
]
