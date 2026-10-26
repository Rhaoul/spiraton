from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from spiraton.core.modes import dextro_mask
from spiraton.core.mode_policy import ModePolicy


# Les quatre opérateurs primitifs du Logos, dans l'ordre canonique de THEORIE_LOGOS.md.
OPS: Tuple[str, ...] = ("add", "sub", "mul", "div")


def commutator_norm(w_a: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
    """Norme de Frobenius du commutateur ‖W_a W_b − W_b W_a‖.

    Définition opératoire de la *non-commutativité acquise* (axiome de
    contextualité, THEORIE_LOGOS.md §2.4). Vaut 0 ssi les deux applications
    linéaires commutent ; croît à mesure que l'ordre d'application des deux
    opérateurs change le résultat. À suivre pendant l'entraînement : un cœur
    qui « réfléchit » développe des opérateurs qui ne commutent pas.
    """
    if w_a.shape != w_b.shape or w_a.dim() != 2 or w_a.size(0) != w_a.size(1):
        raise ValueError("commutator_norm attend deux matrices carrées de même forme")
    comm = w_a @ w_b - w_b @ w_a
    return torch.linalg.norm(comm)


@dataclass
class MatrixSpiratonConfig:
    input_size: int
    init_scale: float = 0.1
    # Ordre de composition dextrogyre (appliqué de gauche à droite dans le temps :
    # le premier nom est appliqué en premier au vecteur d'entrée).
    dextro_order: Tuple[str, ...] = OPS
    # Ordre lévogyre = miroir exact du dextrogyre. None => renversement automatique.
    levo_order: Optional[Tuple[str, ...]] = None


class MatrixSpiratonCell(nn.Module):
    """Cellule Spiraton à poids matriciels — incarnation de la contextualité.

    Différence avec le canon (`SpiratonCell`) : ici chaque opérateur n'est plus
    un vecteur de poids mais une **application linéaire** ``W_op ∈ R^{d×d}``.
    Les vecteurs-poids du canon commutent trivialement (un produit terme-à-terme
    puis somme est invariant par permutation des opérateurs) ; des matrices, non.

    Mécanisme : la sortie est la **composition** des quatre opérateurs appliqués
    dans un ordre donné. Comme le produit matriciel ne commute pas en général,
    changer l'ordre change la sortie — c'est la traduction testable de l'axiome
    de contextualité (THEORIE_LOGOS.md §2.4 : « les opérations ne sont pas
    nécessairement commutatives »).

    Le miroir dextro/lévo du canon (« les deux branches sont des miroirs — ce
    retournement est structurel ») devient ici un **renversement de l'ordre de
    composition** : la branche lévogyre applique les mêmes opérateurs que la
    dextrogyre, mais dans l'ordre inverse. À noter : c'est l'*ordre* qui est
    renversé (``W_a W_m`` → ``W_m W_a``), PAS une inversion algébrique des
    matrices (``W_a⁻¹ W_m⁻¹``) — le retour rejoue la séquence à rebours, il ne
    la défait pas.

    Conventions conservées depuis le canon :
      - activation ``tanh`` en mode dextrogyre, ``atan`` en mode lévogyre ;
      - sélection de mode par ``dextro_mask`` (moyenne ≥ 0) ou ``mode_policy``.

    Contrat de forme : entrée ``(B, d)`` ou ``(d,)`` → sortie ``(B, d)`` ou
    ``(d,)``. La cellule est une endomorphie de l'espace d'états S (state→state),
    pas un réducteur scalaire ; pour l'enchaîner dans la grille/récursion
    (qui attendent un scalaire), la composer avec une projection ``d→1``.

    Initialisation : ``W_op = I + init_scale · N(0,1)``. Près de l'identité, les
    opérateurs commutent presque (commutateurs en O(init_scale²)) : la
    non-commutativité est alors une propriété *acquise* que ``commutators()``
    permet de suivre.
    """

    def __init__(
        self,
        input_size: int,
        init_scale: float = 0.1,
        dextro_order: Sequence[str] = OPS,
        levo_order: Optional[Sequence[str]] = None,
        mode_policy: Optional[ModePolicy] = None,
    ) -> None:
        super().__init__()
        dextro = tuple(dextro_order)
        self._validate_order(dextro)
        levo = tuple(levo_order) if levo_order is not None else tuple(reversed(dextro))
        self._validate_order(levo)

        self.cfg = MatrixSpiratonConfig(
            input_size=input_size,
            init_scale=init_scale,
            dextro_order=dextro,
            levo_order=levo,
        )
        self.mode_policy = mode_policy

        eye = torch.eye(input_size)
        self.W_add = nn.Parameter(eye + torch.randn(input_size, input_size) * init_scale)
        self.W_sub = nn.Parameter(eye + torch.randn(input_size, input_size) * init_scale)
        self.W_mul = nn.Parameter(eye + torch.randn(input_size, input_size) * init_scale)
        self.W_div = nn.Parameter(eye + torch.randn(input_size, input_size) * init_scale)

        self.bias = nn.Parameter(torch.zeros(input_size))

    # -- utilitaires -------------------------------------------------------

    @staticmethod
    def _validate_order(order: Tuple[str, ...]) -> None:
        if not order:
            raise ValueError("l'ordre de composition ne peut pas être vide")
        for name in order:
            if name not in OPS:
                raise ValueError(f"opérateur inconnu : {name!r} (attendu parmi {OPS})")

    def _mat(self, name: str) -> torch.Tensor:
        return getattr(self, f"W_{name}")

    def _mode(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.mode_policy is None:
            return dextro_mask(inputs)
        return self.mode_policy(inputs)

    @staticmethod
    def _is_soft(mode: torch.Tensor) -> bool:
        return mode.dtype != torch.bool

    # -- composition -------------------------------------------------------

    def compose(self, inputs: torch.Tensor, order: Sequence[str]) -> torch.Tensor:
        """Applique les opérateurs (linéaires) dans ``order`` puis retourne le brut.

        Le premier opérateur de ``order`` agit en premier sur ``inputs``. Aucune
        activation ni biais : c'est le brut composé, ``W_{o_k} … W_{o_1} · x``.
        Méthode publique pour pouvoir tester l'effet de l'ordre indépendamment
        du choix de branche/activation.
        """
        order = tuple(order)
        self._validate_order(order)
        squeeze = False
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze = True
        if inputs.size(-1) != self.cfg.input_size:
            raise ValueError(f"last dim must equal input_size={self.cfg.input_size}")

        t = inputs
        for name in order:
            # t: (B, d) ; W: (d, d) ; application linéaire t @ Wᵀ
            t = t @ self._mat(name).t()
        return t.squeeze(0) if squeeze else t

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze = True
        if inputs.size(-1) != self.cfg.input_size:
            raise ValueError(f"last dim must equal input_size={self.cfg.input_size}")

        mode = self._mode(inputs)  # (B,) bool ou float

        raw_dextro = self.compose(inputs, self.cfg.dextro_order) + self.bias
        raw_levo = self.compose(inputs, self.cfg.levo_order) + self.bias

        if self._is_soft(mode):
            gate = torch.clamp(mode, 0.0, 1.0).to(dtype=raw_dextro.dtype).unsqueeze(-1)
            out = gate * torch.tanh(raw_dextro) + (1.0 - gate) * torch.atan(raw_levo)
        else:
            m = mode.unsqueeze(-1)  # (B,1) broadcast sur d
            out = torch.where(m, torch.tanh(raw_dextro), torch.atan(raw_levo))

        return out.squeeze(0) if squeeze else out

    # -- diagnostic --------------------------------------------------------

    def commutators(self) -> Dict[str, torch.Tensor]:
        """Norme du commutateur pour chaque paire d'opérateurs (non ordonnée).

        Retourne un dict ``{"add,mul": ‖[W_add, W_mul]‖, ...}``. Diagnostic
        central du chantier 1 : à tracer pendant l'entraînement pour voir la
        non-commutativité *émerger*.
        """
        out: Dict[str, torch.Tensor] = {}
        for i in range(len(OPS)):
            for j in range(i + 1, len(OPS)):
                a, b = OPS[i], OPS[j]
                out[f"{a},{b}"] = commutator_norm(self._mat(a), self._mat(b))
        return out

    def total_noncommutativity(self) -> torch.Tensor:
        """Somme des normes de commutateurs — scalaire résumant la contextualité."""
        vals = list(self.commutators().values())
        return torch.stack(vals).sum()
