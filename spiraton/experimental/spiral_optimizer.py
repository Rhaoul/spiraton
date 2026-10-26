"""spiral_optimizer.py — ``SpiralClosureOpt`` : le cycle A → B → A′ comme règle de mise à jour.

Conception libre (chantier OPTIMISEUR_ABA, accord de Ra : « l'idée est de recréer un
optimiseur à partir de la logique A-B-A′ et des spiratons »), distincte de la
transposition littérale ``chrono_optimizer.ChronoSpiratonOpt``. Ici la spirale vit
au niveau des CYCLES, pas des pas : la suite des ancrages φ_0, φ_1, … est la
trajectoire spirale ; chaque cycle est une forme F0 (+, +, −).

Un cycle, pour chaque tenseur de poids :

* **SEG_A (+, <DX><OUT><ALPHA>) — ancrage.** φ = poids lents, figés pendant le cycle.
* **SEG_B (+, <DX><OUT><OMEGA>) — déploiement.** ``k`` pas d'excursion des poids
  rapides θ depuis φ, chacun porté par l'action dextrogyre BORNÉE de la cellule
  (``tanh``, branche ``raw_dextro`` de ``core/cell.py``) appliquée au rapport
  « mul − div » en log-espace, c'est-à-dire au quotient ``m̂/√v̂`` :
  ``θ ← θ − lr·g·D(m̂/(√v̂+ε))``, ``D = tanh`` (ou identité si ``borne=False``).
* **SEG_A′ (−, <LV><IN><A_PRIME>) — retour transformé.** L'excursion
  ``Δ = θ − φ`` est intégrée partiellement : ``φ ← φ + α·Δ``, puis ``θ ← φ``.
  « Proche et aligné mais non identique » devient une contrainte sur α :
  ``α = 0`` serait la copie (F5−, le système ne bouge plus), ``α = 1`` l'excursion
  sans retour (F5+, c'est AdamW). Le constructeur exige ``0 < α ≤ 1`` ; ``α = 1``
  n'est admis que comme pivot de référence.

Maintien au bord (``regulate_step`` d'``edge_controller.py``, importé tel quel) :
à chaque clôture, l'observable est le rapport des RAYONS de deux excursions
successives ``ρ̂_k = ‖Δ_k‖/‖Δ_{k−1}‖`` (le pas de la spirale), régulé vers 1.0 : une
spirale qui s'ouvre (ρ̂ > 1, dissipation) baisse le gain g ; une spirale qui se
referme sur elle-même (ρ̂ < 1, extinction/répétition) le remonte, dans
``[g_min, g_max]``. Observable croissant en g ⇒ rétroaction négative stable.

Oubli : décroissance découplée comme AdamW (``θ ← θ·(1 − lr·g·wd)``), sur les poids
rapides ; elle passe dans φ par la clôture.

MÉMOIRE : 3 tampons par paramètre (m, v, φ) contre 2 pour AdamW.

PIVOTS (testés) :
  * ``alpha=1, k=1, borne=False, eta_rho=0`` : trajectoire BIT-À-BIT égale à
    ``torch.optim.AdamW`` (même suite d'opérations flottantes) — le zéro de la mesure.
  * ``eta_rho = 0`` : g ≡ g0 au bit.
  * gradient nul depuis le repos (m = v = 0, wd = 0) : Δθ = 0 au bit.
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

import torch
from torch.optim import Optimizer

from .edge_controller import regulate_step

__all__ = ["SpiralClosureOpt"]


class SpiralClosureOpt(Optimizer):
    """Optimiseur à clôture spirale A → B → A′ (ancrage, excursion bornée, retour α).

    Paramètres :
      lr, betas, eps, weight_decay : comme ``torch.optim.AdamW``.
      k : longueur du déploiement SEG_B (pas d'excursion par cycle).
      alpha : fraction de l'excursion intégrée au retour A′ (0 < α ≤ 1).
      borne : applique l'action dextrogyre bornée ``tanh`` au quotient m̂/√v̂.
      g0, eta_rho, g_min, g_max : actionneur de bord sur ρ̂ (T15), cible 1.0.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        k: int = 5,
        alpha: float = 0.5,
        borne: bool = True,
        g0: float = 1.0,
        eta_rho: float = 0.1,
        g_min: float = 0.80,
        g_max: float = 1.20,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr doit être > 0, reçu {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas doivent être dans [0, 1), reçu {betas}")
        if k < 1:
            raise ValueError(f"k doit valoir >= 1, reçu {k}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha doit être dans (0, 1] (α=0 : copie F5−), reçu {alpha}")
        if g_min > g_max:
            raise ValueError(f"g_min ({g_min}) > g_max ({g_max})")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay doit être >= 0, reçu {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, k=k,
                        alpha=alpha, borne=borne, g0=g0, eta_rho=eta_rho,
                        g_min=g_min, g_max=g_max)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            wd: float = group["weight_decay"]
            k: int = group["k"]
            alpha: float = group["alpha"]
            borne: bool = group["borne"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["anchor"] = p.detach().clone()  # φ : SEG_A
                    state["cycle_pos"] = 0
                    state["r_prev"] = 0.0
                    state["g_ctrl"] = group["g0"]

                state["step"] += 1
                step = state["step"]
                g_cur: float = state["g_ctrl"]
                lr_g = lr * g_cur
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # --- SEG_B : un pas d'excursion borné (ordre des opérations = AdamW) ---
                if wd != 0.0:
                    p.mul_(1 - lr_g * wd)
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr_g / bias_correction1
                denom = (v.sqrt() / bias_correction2 ** 0.5).add_(eps)
                if borne:
                    # D = tanh sur le quotient m̂/√v̂ : |pas| ≤ lr·g par coordonnée.
                    u = torch.tanh(m / bias_correction1 / denom)
                    p.add_(u, alpha=-lr_g)
                else:
                    p.addcdiv_(m, denom, value=-step_size)

                # --- SEG_A′ : clôture du cycle (retour transformé + maintien au bord) ---
                state["cycle_pos"] += 1
                if state["cycle_pos"] >= k:
                    state["cycle_pos"] = 0
                    anchor: torch.Tensor = state["anchor"]
                    if alpha != 1.0:
                        delta = p - anchor
                        r = float(torch.linalg.vector_norm(delta).item())
                        anchor.add_(delta, alpha=alpha)
                        p.copy_(anchor)
                    else:
                        r = float(torch.linalg.vector_norm(p - anchor).item())
                        anchor.copy_(p)
                    r_prev: float = state["r_prev"]
                    if r_prev > 0.0:
                        state["g_ctrl"] = regulate_step(
                            g_cur, r / r_prev, 1.0, group["eta_rho"],
                            group["g_min"], group["g_max"],
                        )
                    state["r_prev"] = r

        return loss
