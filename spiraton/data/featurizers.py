"""Featurizers texte → vecteur, pour alimenter les cellules.

La source de vérité des traits est le **tokenizer 33D** (dépôt voisin). Quand
il n'est pas disponible (pas de compilateur C, CI minimale…), on a besoin d'un
featurizer de **substitution déterministe** pour développer et tester tout
l'aval (loader, perte alpha-oméga, boucle d'entraînement) sans le ``.so``.

``HashingFeaturizer`` est ce substitut, et il est **explicitement étiqueté comme
tel** (``is_fallback = True``). Il ne prétend pas porter la physique
articulatoire : c'est un encodage stable (hachage) du texte, suffisant pour que
la mécanique d'apprentissage soit réelle et testable. Ne pas le confondre avec
les vrais traits phonémiques (chantier 7).
"""
from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol, runtime_checkable

import torch

from . import vector33d
from .tokenizer_bridge import NativeTokenizer33D


@runtime_checkable
class Featurizer(Protocol):
    """Transforme un texte en vecteur ``(dim,)`` déterministe."""

    dim: int
    is_fallback: bool

    def __call__(self, text: str) -> torch.Tensor:
        ...


def _stable_hash(token: str) -> int:
    """Hachage stable inter-process (contrairement à ``hash()`` randomisé)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class HashingFeaturizer:
    """Substitut déterministe du tokenizer 33D (DEV/TEST seulement).

    Encode un texte par le « hashing trick » sur ses bigrammes de caractères :
    chaque bigramme est haché vers un index et un signe, puis accumulé. Le
    vecteur est L2-normalisé. Déterministe (même texte → même vecteur, au bit
    près, entre processus).
    """

    is_fallback = True

    def __init__(self, dim: int = 33, *, lowercase: bool = True) -> None:
        if dim < 1:
            raise ValueError("dim doit être >= 1")
        self.dim = dim
        self.lowercase = lowercase

    def __call__(self, text: str) -> torch.Tensor:
        s = text.lower() if self.lowercase else text
        s = s.strip()
        vec = torch.zeros(self.dim)
        if not s:
            return vec
        padded = f" {s} "
        for i in range(len(padded) - 1):
            bigram = padded[i:i + 2]
            h = _stable_hash(bigram)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = vec.norm()
        if float(norm) > 0:
            vec = vec / norm
        return vec

    def batch(self, texts) -> torch.Tensor:
        """Empile les vecteurs de plusieurs textes en ``(N, dim)``."""
        if not texts:
            return torch.zeros(0, self.dim)
        return torch.stack([self(t) for t in texts], dim=0)


# --- Featurizer RÉEL : les traits articulatoires du tokenizer 33D ------------


class PhonemeFeaturizer:
    """Featurizer adossé au tokenizer 33D natif — le livrable du chantier 7.

    Là où :class:`HashingFeaturizer` est un *substitut* (``is_fallback=True``,
    encodage de hachage sans physique), celui-ci fait circuler les **vrais
    traits articulatoires mesurés** par le tokenizer C. Le 33D porte, en
    dims 8-22 (contrat ``vector33d.PHONEME_SIG``), exactement les signatures que
    le chantier 7 du CLAUDE.md demande d'*exploiter comme canaux d'entrée des
    cellules plutôt que d'en recalculer* :

        dims 8-12  : signature d'impédance (5 premiers phonèmes) ;
        dims 13-17 : signature de flux (5 premiers phonèmes) ;
        dims 18-22 : séquence de spins syllabiques.

    Le contrat ``Featurizer`` est ``texte -> (dim,)``. Un segment ABA est donc
    tokenisé en ``N`` tokens (chacun un 33D réel) puis **agrégé** en un seul
    vecteur 33D. Agrégations disponibles (``pool``) :

      - ``"mean"`` (défaut) : moyenne sur les tokens — profil du segment ;
      - ``"sum"``           : somme (garde l'intensité cumulée) ;
      - ``"first"``         : premier token (mot-tête, sans mélange).

    On **ne renormalise pas** globalement le 33D (contrairement au substitut) :
    les dims 0-3 (scores opérateurs) et 4-5 (chiralité) ont une échelle
    sémantique propre que le consommateur (``OperatorEmbedding``) lit pour
    router la représentation. Les écraser par une norme L2 globale brouillerait
    ce routage.

    Pour un usage **token-par-token** (canaux séquentiels, p. ex. nourrir une
    cellule récurrente token après token), utiliser :meth:`sequence`. Pour la
    seule sous-bande phonémique (dims 8-22), :meth:`signature`.

    Le tokenizer natif est **requis** : sans lui, la construction lève
    :class:`TokenizerUnavailable`. C'est volontaire — la frontière réel/substitut
    reste explicite, l'appelant choisit son repli (``HashingFeaturizer``) en
    connaissance de cause. Pour les tests d'agrégation hors ``.so``, injecter un
    ``tokenizer`` factice exposant ``vectors(text, max_tokens) -> (N, 33)``.
    """

    is_fallback = False

    _POOLS = ("mean", "sum", "first")

    def __init__(
        self,
        tokenizer: Optional[object] = None,
        *,
        pool: str = "mean",
        max_tokens: int = 128,
    ) -> None:
        if pool not in self._POOLS:
            raise ValueError(f"pool inconnu : {pool!r} (attendu {self._POOLS})")
        self.dim = vector33d.DIM        # 33 — contrat ABI
        self.pool = pool
        self.max_tokens = int(max_tokens)
        # Création paresseuse du natif si non injecté (peut lever TokenizerUnavailable).
        self._tok = tokenizer if tokenizer is not None else NativeTokenizer33D()

    def sequence(self, text: str) -> torch.Tensor:
        """``(N, 33)`` — un vecteur 33D réel par token, **sans** agrégation."""
        import numpy as np

        arr = np.asarray(self._tok.vectors(text, max_tokens=self.max_tokens), dtype=np.float32)
        if arr.ndim != 2 or (arr.size and arr.shape[1] != self.dim):
            raise ValueError(f"tokenizer a renvoyé une forme {arr.shape}, attendu (N, {self.dim})")
        return torch.tensor(arr, dtype=torch.float32)

    def __call__(self, text: str) -> torch.Tensor:
        seq = self.sequence(text)               # (N, 33)
        if seq.size(0) == 0:                    # texte vide / aucun token
            return torch.zeros(self.dim)
        if self.pool == "first":
            return seq[0].clone()
        if self.pool == "sum":
            return seq.sum(dim=0)
        return seq.mean(dim=0)

    def signature(self, text: str) -> torch.Tensor:
        """``(15,)`` — uniquement les canaux phonémiques (dims 8-22) agrégés.

        La quantité *littérale* du chantier 7 : impédance + flux + spins.

        Honnêteté : côté C, ces bandes ne couvrent que les **5 premiers
        phonèmes/spins** du mot (la tête articulatoire), pas l'intégralité d'un
        mot long. Voir docs/CHANTIER7_PHONETIQUE_SEMANTIQUE.md.
        """
        return self(text)[vector33d.PHONEME_SIG]

    def batch(self, texts: List[str]) -> torch.Tensor:
        """Empile les vecteurs agrégés de plusieurs textes en ``(M, 33)``."""
        if not texts:
            return torch.zeros(0, self.dim)
        return torch.stack([self(t) for t in texts], dim=0)
