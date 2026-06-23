"""Chargement d'un corpus ABA en triplets de tenseurs (A, B, A′).

Transforme chaque cycle du corpus en trois vecteurs via un featurizer, prêts
pour la boucle d'entraînement (chantier 5). Le featurizer par défaut est le
substitut de développement ; passer un featurizer adossé au tokenizer 33D dès
qu'il est disponible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .aba import iter_aba_cycles, AbaCycle
from .featurizers import Featurizer, HashingFeaturizer


@dataclass(frozen=True)
class AbaTriples:
    """Triplets vectorisés d'un corpus ABA + métadonnées de traçabilité."""

    a: torch.Tensor          # (N, dim) — segment A (origine)
    b: torch.Tensor          # (N, dim) — segment B (déploiement)
    a_prime: torch.Tensor    # (N, dim) — segment A′ (retour transformé)
    ops: List[str]           # (N,) opérateur dominant par cycle
    featurizer_is_fallback: bool

    def __len__(self) -> int:
        return self.a.size(0)

    @property
    def dim(self) -> int:
        return self.a.size(1)


def triples_from_cycles(
    cycles: List[AbaCycle],
    featurizer: Optional[Featurizer] = None,
) -> AbaTriples:
    """Vectorise une liste de cycles déjà parsés."""
    feat = featurizer if featurizer is not None else HashingFeaturizer()
    if not cycles:
        z = torch.zeros(0, feat.dim)
        return AbaTriples(z, z, z, [], bool(getattr(feat, "is_fallback", False)))

    a = torch.stack([feat(c.seg_a.text) for c in cycles], dim=0)
    b = torch.stack([feat(c.seg_b.text) for c in cycles], dim=0)
    ap = torch.stack([feat(c.seg_a_prime.text) for c in cycles], dim=0)
    ops = [c.op for c in cycles]
    return AbaTriples(a, b, ap, ops, bool(getattr(feat, "is_fallback", False)))


def load_aba_triples(
    path: str,
    featurizer: Optional[Featurizer] = None,
    *,
    limit: Optional[int] = None,
    skip_fixed_points: bool = False,
) -> AbaTriples:
    """Charge un corpus ABA en :class:`AbaTriples`.

    limit : nombre maximal de cycles (utile pour des tests rapides).
    skip_fixed_points : ignore les cycles dégénérés (A=B=A′, l'écho du vide) —
        ils sont la répétition que l'on cherche justement à pénaliser.
    """
    cycles: List[AbaCycle] = []
    for cyc in iter_aba_cycles(path):
        if skip_fixed_points and cyc.is_fixed_point:
            continue
        cycles.append(cyc)
        if limit is not None and len(cycles) >= limit:
            break
    return triples_from_cycles(cycles, featurizer)


def iter_batches(triples: AbaTriples, batch_size: int, *, shuffle_seed: Optional[int] = None):
    """Itère des sous-lots ``(a, b, a_prime)``.

    shuffle_seed : si fourni, mélange déterministe (jamais d'aléa non seedé).
    """
    n = len(triples)
    if n == 0:
        return
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        order = torch.randperm(n, generator=g)
    else:
        order = torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield triples.a[idx], triples.b[idx], triples.a_prime[idx]
