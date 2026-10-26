"""chrono_optimizer.py — Optimiseur à clôture spirale A-B-A′ et co-régulation g + ω.

Chantier OPTIMISEUR_ABA (``CHANTIER_OPTIMISEUR_ABA.md``) : transposer la dynamique
A → B → A′ dans l'espace des PARAMÈTRES pour remplacer ``torch.optim.Adam`` /
``AdamW`` (``return_training.py``, ``train_run``). Fusionne sans les modifier :

  1. La récurrence du second ordre de ``ChronoSpiraton`` (``experimental/chrono.py``),
     portée par l'état ``s`` de l'optimiseur (pas par le modèle) :
     ``s_{t+1} = D( A(s_t) + B(s_t²) ) + L(s_t) − C(s_{t−1})``   [``c_outside=True``]
     ``s_{t+1} = D( A(s_t) + B(s_t²) − C(s_{t−1}) ) + L(s_t)``   [``c_outside=False``]
     avec D = ``d_dextro·tanh`` (action, <DX><OUT>) et L = ``l_levo·atan`` (mémoire,
     <LV><IN>), les deux activations des branches chirales de ``core/cell.py``.
  2. L'organe ``regulate_step`` d'``edge_controller.py``, IMPORTÉ tel quel (Porte 0),
     câblé en co-régulation :
       - Actionneur 1 (gain g, T15) : ``ρ̂_t = ‖s_t‖/‖s_{t−1}‖`` → cible 1.0.
       - Actionneur 2 (rappel ω, T20) : ``cos(θ_t, θ_{t−W})`` → cible ``phase_target``.

Le pas appliqué (SEG_A_PRIME) : ``θ ← θ − lr·g·s_{t+1}``, puis un rappel centripète
de poids ``sin(ω)·lr`` vers la direction de ``θ_{t−W}`` (retour vers l'ancrage).

DÉFINITIONS OPÉRATIONNELLES (chaque nom → une quantité) :

* ``u_t`` (entrée B) : gradient normalisé en RMS, ``u = g·√n/‖g‖`` — direction du
  gradient du tenseur entier (géométrie angulaire préservée, pas de division
  coordonnée par coordonnée), coordonnées d'ordre 1 quelle que soit la taille n.
* ``x = a_agg·s_t + (1−a_agg)·u_t`` : agrégation A (moyenne mobile, rôle du m_t d'Adam).
* Terme B : ``b_quad·x·tanh(log x²)`` — l'énergie x² module le gain de chaque
  coordonnée dans ``(1−b_quad, 1+b_quad)`` : amplifie |x|>1, atténue |x|<1, sans
  jamais retourner le signe (``b_quad < 1``) ; nul en x = 0.
* Pas borné : ``|s_{t+1}| ≤ d_dextro + l_levo·π/2 + c_inhib·|s_{t−1}|`` coordonnée
  par coordonnée ⇒ aucun pic de gradient ne peut produire un pas non borné (le rôle
  du ``clip_grad_norm_`` externe, porté par la forme même de la mise à jour).

CORRECTIONS MESURÉES de la spécification v0 (``legacy_v0=True`` reproduit la spec
EXACTE, conservée pour que chaque défaut reste démontrable par un test) :

  (1) Terme B. v0 : ``b·sign(x)·tanh(log x²)`` ≈ ``−b·sign(x)`` dès |x| < 1 : il
      retourne le signe du pas (cos(Δθ, −g) ≈ −0,76 au premier pas) et la perte
      d'une quadratique ne descend pas. v1 : ``b·x·tanh(log x²)`` (signe préservé).
  (2) Observable ρ̂. v0 stocke ``‖s_{t+1}‖`` puis le compare à ``‖s_cur‖`` — le même
      tenseur un pas plus tard — donc ρ̂ ≡ 1 et g reste g0. v1 : ``‖s_t‖/‖s_{t−1}‖``.
  (3) Signe de la boucle ω. v0 lit ``−cos(θ_t, θ_{t−W})`` (croissant en ω au T20, où
      ω est une vitesse de rotation). Ici ω est un RAPPEL vers θ_{t−W} : il AUGMENTE
      cos, donc ``−cos`` décroît en ω ⇒ rétroaction positive, ω sature à ``omega_max``.
      v1 lit ``+cos`` (croissant en ω : rétroaction négative, règle même du T20).
  (4) Échelle de u. v0 : ``g/‖g‖`` ⇒ coordonnées ~1/√n, pas par coordonnée ~√n fois
      plus petit qu'Adam au même lr (un 4096×4096 serait 4096× plus lent). v1 : RMS.

PIVOTS ANTI-ARTEFACT (garantis par construction IEEE754, v0 comme v1) :
  * ``eta_rho = 0.0`` et ``eta_omega = 0.0`` : ``g ≡ g0`` et ``ω ≡ omega0`` au bit.
  * ``c_inhib = 0.0`` : ``c_outside=True`` et ``c_outside=False`` coïncident au bit.
  * Depuis le repos (``s = 0``) sous gradient nul : ``Δθ = 0`` au bit.
  * ``weight_decay = 0.0`` (défaut) : aucune opération de décroissance n'est exécutée.

MÉMOIRE : 2 tampons (``s_t``, ``s_{t−1}``) comme AdamW, PLUS l'anneau de W copies
de θ pour l'observable de phase (W=10 ⇒ 10× la taille du modèle). Acceptable sur
``ChronoSpiraton(19)``, rédhibitoire à grande échelle : point ouvert, documenté.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
from torch.optim import Optimizer

from .edge_controller import regulate_step

__all__ = ["ChronoOptimizerConfig", "ChronoSpiratonOpt", "regulate_step"]


@dataclass(frozen=True)
class ChronoOptimizerConfig:
    """Hyperparamètres gelés d'une instance (lecture seule, publiés dans les rapports)."""

    lr: float = 1e-3
    a_agg: float = 0.9
    b_quad: float = 0.1
    c_inhib: float = 0.05
    d_dextro: float = 0.7
    l_levo: float = 0.3
    c_outside: bool = True
    g0: float = 1.0
    eta_rho: float = 0.5
    g_min: float = 0.80
    g_max: float = 1.20
    window_w: int = 10
    phase_target: float = 0.7
    omega0: float = 0.0795
    eta_omega: float = 0.05
    omega_min: float = 0.0
    omega_max: float = 0.314
    weight_decay: float = 0.0
    eps: float = 1e-12
    legacy_v0: bool = False


class ChronoSpiratonOpt(Optimizer):
    """Optimiseur A → B → A′ à dynamique Chrono du second ordre et maintien au bord.

    Remplaçant direct de ``torch.optim.Adam`` / ``AdamW`` : même protocole
    (``zero_grad`` / ``backward`` / ``step(closure=None)``), état par tenseur.
    ``weight_decay`` est découplé comme dans AdamW (``θ ← θ·(1 − lr·wd)`` avant le pas).
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        lr: float = 1e-3,
        a_agg: float = 0.9,
        b_quad: float = 0.1,
        c_inhib: float = 0.05,
        d_dextro: float = 0.7,
        l_levo: float = 0.3,
        c_outside: bool = True,
        g0: float = 1.0,
        eta_rho: float = 0.5,
        g_min: float = 0.80,
        g_max: float = 1.20,
        window_w: int = 10,
        phase_target: float = 0.7,
        omega0: float = 0.0795,
        eta_omega: Optional[float] = None,
        omega_min: float = 0.0,
        omega_max: float = 0.314,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
        legacy_v0: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr doit être > 0, reçu {lr}")
        if window_w < 1:
            raise ValueError(f"window_w doit valoir >= 1, reçu {window_w}")
        if g_min > g_max:
            raise ValueError(f"g_min ({g_min}) > g_max ({g_max})")
        if omega_min > omega_max:
            raise ValueError(f"omega_min ({omega_min}) > omega_max ({omega_max})")
        if not 0.0 <= b_quad < 1.0:
            raise ValueError(f"b_quad doit être dans [0, 1) (signe du pas préservé), reçu {b_quad}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay doit être >= 0, reçu {weight_decay}")

        if eta_omega is None:
            # Loi de ré-échelonnement T20 : η_ω = η_ρ / W
            eta_omega = eta_rho / float(window_w)

        self.cfg = ChronoOptimizerConfig(
            lr=lr, a_agg=a_agg, b_quad=b_quad, c_inhib=c_inhib, d_dextro=d_dextro,
            l_levo=l_levo, c_outside=c_outside, g0=g0, eta_rho=eta_rho, g_min=g_min,
            g_max=g_max, window_w=window_w, phase_target=phase_target, omega0=omega0,
            eta_omega=eta_omega, omega_min=omega_min, omega_max=omega_max,
            weight_decay=weight_decay, eps=eps, legacy_v0=legacy_v0,
        )
        defaults = {k: getattr(self.cfg, k) for k in self.cfg.__dataclass_fields__}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            a_agg: float = group["a_agg"]
            b_quad: float = group["b_quad"]
            c_inhib: float = group["c_inhib"]
            d_dextro: float = group["d_dextro"]
            l_levo: float = group["l_levo"]
            c_outside: bool = group["c_outside"]
            g0: float = group["g0"]
            eta_rho: float = group["eta_rho"]
            g_min: float = group["g_min"]
            g_max: float = group["g_max"]
            W: int = group["window_w"]
            phase_target: float = group["phase_target"]
            omega0: float = group["omega0"]
            eta_omega: float = group["eta_omega"]
            omega_min: float = group["omega_min"]
            omega_max: float = group["omega_max"]
            wd: float = group["weight_decay"]
            eps: float = group["eps"]
            v0: bool = group["legacy_v0"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["s_cur"] = torch.zeros_like(p)
                    state["s_prev"] = torch.zeros_like(p)
                    state["r_prev"] = 0.0
                    state["g_ctrl"] = g0
                    state["anchor_ring"] = []
                    state["omega_ctrl"] = omega0

                t: int = state["step"]
                s_cur: torch.Tensor = state["s_cur"]
                s_prev: torch.Tensor = state["s_prev"]
                g_cur: float = state["g_ctrl"]
                om_cur: float = state["omega_ctrl"]
                ring: List[torch.Tensor] = state["anchor_ring"]

                # --- 1. SEG_A : observables et co-régulation (regulate_step) ---
                # v1 : r_prev = ‖s_{t−1}‖ (correction 2) ; v0 : ‖s_t‖ du pas d'avant (ρ̂ ≡ 1).
                r_t = float(torch.linalg.vector_norm(s_cur).item())
                r_prev: float = state["r_prev"]
                if t > 0 and r_prev > eps:
                    rho_hat = r_t / r_prev
                    g_cur = regulate_step(g_cur, rho_hat, 1.0, eta_rho, g_min, g_max)

                theta_anchor_w: Optional[torch.Tensor] = None
                if len(ring) >= W:
                    theta_anchor_w = ring[0]
                    na = float(torch.linalg.vector_norm(p).item())
                    nb = float(torch.linalg.vector_norm(theta_anchor_w).item())
                    if na > eps and nb > eps:
                        cos_w = float(torch.sum(p * theta_anchor_w).item() / (na * nb))
                    else:
                        cos_w = -1.0  # sentinelle T20 : point nul = non aligné
                    if v0:
                        om_cur = regulate_step(
                            om_cur, -cos_w, -phase_target, eta_omega, omega_min, omega_max
                        )
                    else:
                        # Correction 3 : observable croissant en ω (rappel) ⇒ boucle stable.
                        om_cur = regulate_step(
                            om_cur, cos_w, phase_target, eta_omega, omega_min, omega_max
                        )

                ring.append(p.detach().clone())
                if len(ring) > W:
                    ring.pop(0)

                # --- 2. SEG_B : excursion du second ordre (Chrono + Cell) ---
                grad_norm = float(torch.linalg.vector_norm(grad).item())
                if grad_norm > eps:
                    if v0:
                        grad_unit = grad / (grad_norm + eps)
                    else:
                        # Correction 4 : normalisation RMS (coordonnées d'ordre 1).
                        grad_unit = grad * (math.sqrt(grad.numel()) / grad_norm)
                else:
                    grad_unit = torch.zeros_like(grad)

                s_input = a_agg * s_cur + (1.0 - a_agg) * grad_unit

                log_sq = torch.tanh(torch.log(torch.clamp(s_input * s_input, min=eps)))
                if v0:
                    sq_term = b_quad * s_input.sign() * log_sq
                else:
                    # Correction 1 : modulation de gain dans (1−b, 1+b), signe préservé.
                    sq_term = b_quad * s_input * log_sq

                if c_outside:
                    inner = s_input + sq_term
                    s_next = (
                        d_dextro * torch.tanh(inner)
                        + l_levo * torch.atan(s_cur)
                        - c_inhib * s_prev
                    )
                else:
                    inner = s_input + sq_term - c_inhib * s_prev
                    s_next = d_dextro * torch.tanh(inner) + l_levo * torch.atan(s_cur)

                # --- 3. SEG_A_PRIME : clôture spirale (rayon g_cur + rappel om_cur) ---
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(s_next, alpha=-lr * g_cur)

                if theta_anchor_w is not None and om_cur > 0.0 and grad_norm > eps:
                    pull = math.sin(om_cur) * lr
                    norm_p = float(torch.linalg.vector_norm(p).item())
                    norm_w = float(torch.linalg.vector_norm(theta_anchor_w).item())
                    if norm_p > eps and norm_w > eps:
                        aligned_ref = theta_anchor_w * (norm_p / (norm_w + eps))
                        p.lerp_(aligned_ref, weight=pull)

                if v0:
                    state["r_prev"] = float(torch.linalg.vector_norm(s_next).item())
                else:
                    state["r_prev"] = r_t
                state["s_prev"].copy_(s_cur)
                state["s_cur"].copy_(s_next)
                state["g_ctrl"] = g_cur
                state["omega_ctrl"] = om_cur
                state["step"] = t + 1

        return loss
