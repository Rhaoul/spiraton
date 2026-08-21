from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from spiraton.data import vector33d


FeatureSource = Literal["phoneme", "full"]


class OperatorEmbedding(nn.Module):
    """Représentation conditionnée par l'opérateur × chiralité (chantier 4).

    Au lieu d'un vocabulaire *plat* (un token → un vecteur), on tient **huit
    sous-espaces** : 4 opérateurs (ADD/SUB/MUL/DIV) × 2 chiralités (DX/LV).
    Chaque sous-espace est une application linéaire ``W[op,chi]`` des traits
    d'entrée vers l'espace d'embedding.

    Le point théorique (CLAUDE.md, chantier 4) : *l'opérateur conditionne la
    représentation, il ne s'y ajoute pas*. Ici les scores des dims 0-5 du
    vecteur 33D (0-3 = ADD/SUB/MUL/DIV, 4-5 = dextro/lévo) ne sont pas
    concaténés aux traits : ils **pondèrent** (par une combinaison convexe issue
    de deux softmax) le choix du sous-espace. Une représentation routée
    entièrement vers ADD diffère donc de la même entrée routée vers MUL — la
    modulation est multiplicative (sélection), pas additive.

    Entrée : vecteur 33D ``(.., 33)`` (le module lit lui-même les poids op/chi
    dans les dims 0-5). Traits projetés selon ``source`` :
      - ``"phoneme"`` : dims 8-22 (15 traits articulatoires — voir chantier 7) ;
      - ``"full"``    : les 33 dims.
    Sortie : embedding ``(.., embed_dim)``.

    ``temperature`` contrôle la dureté du routage : →0 sélectionne un seul
    sous-espace (one-hot), →∞ moyenne uniformément les huit.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        source: FeatureSource = "phoneme",
        temperature: float = 1.0,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature doit être > 0")
        if source == "phoneme":
            self.feat_slice = vector33d.PHONEME_SIG
        elif source == "full":
            self.feat_slice = slice(0, vector33d.DIM)
        else:
            raise ValueError(f"source inconnue : {source!r} (attendu 'phoneme' ou 'full')")

        self.source = source
        self.embed_dim = embed_dim
        self.temperature = float(temperature)
        self.feature_size = self.feat_slice.stop - self.feat_slice.start
        self.n_ops = len(vector33d.OPERATORS)        # 4
        self.n_chi = 2                                # DX, LV
        self.n_sub = self.n_ops * self.n_chi         # 8 sous-espaces

        # Une application linéaire par sous-espace : (8, feature_size, embed_dim).
        self.W = nn.Parameter(torch.randn(self.n_sub, self.feature_size, embed_dim) * init_scale)
        self.b = nn.Parameter(torch.zeros(self.n_sub, embed_dim))

    def opchi_weights(self, v33: torch.Tensor) -> torch.Tensor:
        """Combinaison convexe sur les 8 sous-espaces, issue des dims 0-5.

        Deux softmax indépendants (sur les 4 opérateurs, sur les 2 chiralités)
        puis produit extérieur. Index aplati ``k = op * 2 + chi``. Somme à 1.
        """
        op_w = torch.softmax(v33[..., 0:4] / self.temperature, dim=-1)        # (..,4)
        chi_w = torch.softmax(v33[..., 4:6] / self.temperature, dim=-1)       # (..,2)
        w = op_w.unsqueeze(-1) * chi_w.unsqueeze(-2)                          # (..,4,2)
        return w.reshape(*w.shape[:-2], self.n_sub)                           # (..,8)

    def forward(self, v33: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if v33.dim() == 1:
            v33 = v33.unsqueeze(0)
            squeeze = True
        if v33.size(-1) != vector33d.DIM:
            raise ValueError(f"dernière dim doit être {vector33d.DIM} (vecteur 33D)")

        feats = v33[..., self.feat_slice]                # (.., feature_size)
        weights = self.opchi_weights(v33)                # (.., 8)
        # Projeter par chacun des 8 sous-espaces : (.., 8, embed_dim).
        proj = torch.einsum("...f,kfe->...ke", feats, self.W) + self.b
        out = (weights.unsqueeze(-1) * proj).sum(dim=-2)  # (.., embed_dim)
        return out.squeeze(0) if squeeze else out

    def subspace_index(self, op: str, chi: str) -> int:
        """Index aplati k du sous-espace (op, chi) — utile pour inspecter W."""
        op = op.upper()
        if op not in vector33d.OPERATORS:
            raise ValueError(f"opérateur inconnu : {op!r}")
        chi_idx = 0 if chi.upper() in ("DX", "DEXTRO") else 1
        return vector33d.OPERATORS.index(op) * 2 + chi_idx
