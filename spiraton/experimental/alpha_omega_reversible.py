"""alpha_omega_reversible.py — ``AlphaOmegaReversible`` : un pas raté est défait.

Variante d'``AlphaOmegaOpt`` qui prend au mot l'exigence C5 du dépôt ``cybernetique``
(« chaque révision A → A′ est jugée sur des épisodes indépendants, et annulée si elle
échoue »), à l'échelle d'un pas d'optimisation :

* le juge est le même : ``ω = cos(−∇L(θ_{t+1}), Δθ_t)``, le gradient d'un lot que le
  pas n'a pas vu, mesuré au point d'arrivée ;
* **échec** ⇔ ``ω < seuil_annulation`` (par défaut −0,5) : le gradient d'arrivée
  s'oppose franchement au pas, qui a franchi le creux au lieu d'y descendre ;
* sur échec, le pas est **défait** : ``θ ← θ_t`` au bit (copie gardée, pas une
  soustraction), le gain est coupé (``g ← max(g_min, κ·g)``, κ = 0,5 par défaut) et le
  gradient du lot courant n'est pas déployé : il a été mesuré en un point rejeté. Les
  moments m et v restent : ils portent des gradients mesurés en θ_t, où l'on revient.
  On défait le pas, pas la mémoire ;
* le pas suivant part de θ_t avec le gain réduit ; comme aucun pas n'a été fait, il
  n'y a rien à juger avant lui (Δθ est remis à zéro).

La boucle continue d'AlphaOmegaOpt (ω̄ → g) s'applique d'abord, telle quelle ; la
coupure ne s'y ajoute que sur échec. C'est l'analogue d'une région de confiance dont
le test d'acceptation serait le signe du gradient d'arrivée plutôt que la baisse de la
perte : il ne coûte aucune passe avant supplémentaire.

MÉMOIRE : 4 tampons par paramètre (m, v, Δθ, θ_t), un de plus qu'``AlphaOmegaOpt``.

PIVOTS (testés) :
  * ``seuil_annulation=None`` : trajectoire BIT-À-BIT celle d'``AlphaOmegaOpt``.
  * un pas défait rend θ_t au bit, laisse m, v et le compteur de pas inchangés.

Compteur : ``param_groups[i]["n_annul"]`` (pas défaits), sauvé par ``state_dict``.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch

from .alpha_omega_optimizer import AlphaOmegaOpt

__all__ = ["AlphaOmegaReversible"]


class AlphaOmegaReversible(AlphaOmegaOpt):
    """``AlphaOmegaOpt`` dont un pas jugé raté (``ω < seuil_annulation``) est défait.

    Paramètres propres :
      seuil_annulation : cosinus en dessous duquel le pas est défait, dans [−1, 0) ;
        ``None`` désactive l'annulation (pivot ``AlphaOmegaOpt``).
      kappa : facteur de coupure du gain sur annulation, dans (0, 1).
    Les autres paramètres sont ceux d'``AlphaOmegaOpt``.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        seuil_annulation: Optional[float] = -0.5,
        kappa: float = 0.5,
        **kwargs,
    ) -> None:
        if seuil_annulation is not None and not -1.0 <= seuil_annulation < 0.0:
            raise ValueError(f"seuil_annulation doit être dans [-1, 0) ou None, reçu {seuil_annulation}")
        if not 0.0 < kappa < 1.0:
            raise ValueError(f"kappa doit être dans (0, 1), reçu {kappa}")
        self._propres = dict(seuil_annulation=seuil_annulation, kappa=kappa)
        super().__init__(params, lr, betas, eps, weight_decay, **kwargs)
        self.defaults.update(self._propres)

    def add_param_group(self, param_group: dict) -> None:
        for cle, val in self._propres.items():
            param_group.setdefault(cle, val)
        super().add_param_group(param_group)
        self.param_groups[-1].setdefault("n_annul", 0)

    def _apres_retour(self, group: dict, params: List[torch.Tensor],
                      omega: Optional[float]) -> bool:
        seuil = group["seuil_annulation"]
        if omega is not None and seuil is not None and omega < seuil:
            for p in params:
                state = self.state[p]
                p.copy_(state["theta_prec"])
                state["delta"].zero_()  # aucun pas fait : rien à juger au pas suivant
            group["g"] = max(group["g_min"], group["kappa"] * group["g"])
            group["n_annul"] += 1
            return True
        for p in params:  # le pas va être fait : θ_t est gardé pour pouvoir le défaire
            state = self.state[p]
            if "theta_prec" not in state:
                state["theta_prec"] = torch.empty_like(p)
            state["theta_prec"].copy_(p)
        return False
