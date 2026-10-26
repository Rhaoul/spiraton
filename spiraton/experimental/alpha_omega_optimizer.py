"""alpha_omega_optimizer.py — ``AlphaOmegaOpt`` : comparer le résultat à l'intention.

Troisième lecture du chantier OPTIMISEUR_ABA, tirée de ``corpus_eve_clean.txt``
plutôt que de la mécanique des cellules :

    « Le système apprend quand il compare sa sortie à son intention. »
    « Alpha encode une direction avant l'action. Oméga mesure l'effet réel après l'action. »
    « Une récursion de second ordre modifie la définition de B. »
    « La validation est un miroir externe. »

Les deux premières lectures (``ChronoSpiratonOpt``, ``SpiralClosureOpt``) bouclaient
la sortie sur la sortie précédente (rayons successifs ρ̂, phase des pas) : avec un
pas normalisé à la Adam, ces observables restent près de leur cible quoi qu'il
arrive, d'où des régulateurs inertes (``docs/CHANTIER_OPTIMISEUR_ABA_RESULTATS.md``).
Ici le retour compare l'action à ce qu'elle visait, et il corrige la RÈGLE (le gain
du pas), pas les poids : c'est une récursion du second ordre au sens du corpus.

Un cycle par pas, sur tout le groupe de paramètres (sommes sur tous les tenseurs) :

* **A (alpha, l'intention).** La direction du pas que l'optimiseur vient de faire,
  ``Δθ_t = θ_{t+1} − θ_t`` (gardée en mémoire jusqu'au retour).
* **B (le déploiement).** Le pas AdamW de gain ``g`` :
  ``Δθ_t = −lr·g·(m̂/(√v̂+ε) + wd·θ_t)`` (même suite d'opérations que AdamW).
* **Ω (oméga, le résultat).** Au pas suivant, le gradient du lot SUIVANT, mesuré au
  point d'arrivée, juge le pas : ``ω_t = cos(−∇L(θ_{t+1}), Δθ_t)``.
* **A′ (le retour transformé).** ``ω̄`` (moyenne exponentielle de taux ``beta_q``)
  corrige le gain : ``log g ← regulate_step(log g, −ω̄, −cible, eta, log g_min,
  log g_max)`` (``regulate_step`` d'``edge_controller.py``, importé tel quel ;
  appliqué au logarithme du gain pour que le lr puisse traverser des décades).

Lecture de ω (la question « le retour est-il proche et aligné mais non identique ? ») :
  * ``ω ≈ +1`` : à l'arrivée, la pente est encore celle du départ, le pas n'a rien
    changé (répétition) ⇒ pas trop court ⇒ g monte ;
  * ``ω < 0`` : le gradient d'arrivée s'oppose à l'intention, le pas a franchi le
    creux (le système ne revient pas, il se perd) ⇒ pas trop long ⇒ g descend ;
  * ``ω = 0`` : le pas a consommé exactement la descente disponible le long de sa
    direction (condition de la recherche linéaire exacte). C'est la cible.
L'opposition ``−ω̄`` croît avec la longueur du pas (sur une coupe quadratique, la
pente d'arrivée vaut ``σ_0·(1 − h/h*)``) : la boucle est une rétroaction NÉGATIVE.

Pourquoi un miroir EXTERNE (mesuré à la calibration, ``docs/PROTOCOLE_FENETRE_LR.md``) :
une première version jugeait le rapport ``q = ω/α`` avec ``α = −m̂·Δθ``, l'intention
lue sur le moment m̂. Or Δθ est fait de m̂ : l'intention contient le même bruit que
l'action et se confirme elle-même (``α`` est gonflé par la variance du gradient,
``q`` tend vers un rapport signal/bruit). Le gain descendait alors jusqu'à
``g_min`` et l'apprentissage s'arrêtait (verrouillage). Le gradient du lot suivant
est indépendant de Δθ : le signe de ω est sans biais.

Pourquoi ``cible = 0`` : quand le bruit domine, ω tend vers 0 quel que soit le pas ;
toute cible > 0 ferait alors descendre g jusqu'à ``g_min``.

Pourquoi ``g0 = 0,01`` (un premier pas court) : avant le premier pas il n'existe
aucun retour. À la calibration, les échecs à grand lr se décidaient dans les tout
premiers pas, avant que la boucle se ferme. Le premier B est donc court, et c'est
le retour qui l'allonge (au plus ``×e^{eta}`` par pas).

Conditions de la récursion (« sans condition, la récursion diverge ») : ω est un
cosinus, donc ``|Δ log g| ≤ eta·(1 + |cible|)`` par pas, et ``g ∈ [g_min, g_max]``.

Parenté déclarée : c'est la descente par hypergradient (Baydin et al., 2018,
``∂L/∂lr = −∇L(θ_t)·u_{t−1}``), dans sa variante multiplicative normalisée.
Aucune nouveauté n'est revendiquée ; ce qui est testé, c'est si cette lecture du
cycle A → B → A′ élargit la plage de lr où l'on apprend sur le banc T64.

MÉMOIRE : 3 tampons par paramètre (m, v, Δθ), comme ``SpiralClosureOpt``. Coût :
trois produits scalaires par pas, aucune passe avant supplémentaire. Sur GPU, ces
scalaires sont rapatriés en **une seule synchronisation par pas** (par appareil), quel
que soit le nombre de tenseurs et de groupes : sommés dans le même ordre que tenseur par
tenseur, ils donnent la même trajectoire au bit (testé).

PIVOTS (testés) :
  * ``eta = 0, g0 = 1`` : trajectoire BIT-À-BIT celle de ``torch.optim.AdamW``.
  * ``eta = 0`` : g ≡ g0 au bit.
  * gradient nul depuis le repos (m = v = 0, wd = 0) : Δθ = 0 au bit, g inchangé.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer

from .edge_controller import regulate_step

__all__ = ["AlphaOmegaOpt"]


def _vers_hote(scalaires: List[torch.Tensor]) -> List[float]:
    """Rapatrie des scalaires 0-d en flottants Python : une synchronisation par appareil.

    Chaque scalaire est converti sans perte (float64, ou float32 sur MPS qui n'a pas de
    float64 : exact pour float32, float16 et bfloat16), puis empilé : la valeur lue est
    celle que donnerait ``float(s.item())``, mais en un seul transfert."""
    par_appareil: Dict[torch.device, List[int]] = {}
    for k, s in enumerate(scalaires):
        par_appareil.setdefault(s.device, []).append(k)
    out = [0.0] * len(scalaires)
    for appareil, idx in par_appareil.items():
        dtype = torch.float32 if appareil.type == "mps" else torch.float64
        valeurs = torch.stack([scalaires[k].to(dtype) for k in idx]).tolist()
        for k, v in zip(idx, valeurs):
            out[k] = float(v)
    return out


class AlphaOmegaOpt(Optimizer):
    """AdamW dont le gain de pas est corrigé par l'alignement du retour sur l'intention.

    Paramètres :
      lr, betas, eps, weight_decay : comme ``torch.optim.AdamW`` (le lr effectif est
        ``lr·g``).
      cible : alignement visé ``ω̄`` (un cosinus dans (−1, 1) ; 0 = recherche linéaire).
      eta : gain de rétroaction sur ``log g`` (0 ⇒ loi inerte).
      beta_q : taux de la moyenne exponentielle de ω (0 ≤ beta_q < 1).
      g0, g_min, g_max : gain initial et bornes (``0 < g_min ≤ g0 ≤ g_max``).

    Après chaque ``step``, ``param_groups[i]`` expose ``g`` (gain courant),
    ``omega_bar``, ``omega_last`` (dernier ω lu, ``nan`` avant la première lecture) et
    ``n_regul`` (nombre de retours appliqués) ; ``state_dict`` les sauve.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        cible: float = 0.0,
        eta: float = 0.1,
        beta_q: float = 0.9,
        g0: float = 1e-2,
        g_min: float = 1e-4,
        g_max: float = 1e2,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr doit être > 0, reçu {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"betas doivent être dans [0, 1), reçu {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"weight_decay doit être >= 0, reçu {weight_decay}")
        if not -1.0 < cible < 1.0:
            raise ValueError(f"cible doit être dans (-1, 1) (un cosinus), reçu {cible}")
        if eta < 0.0:
            raise ValueError(f"eta doit être >= 0, reçu {eta}")
        if not 0.0 <= beta_q < 1.0:
            raise ValueError(f"beta_q doit être dans [0, 1), reçu {beta_q}")
        if not 0.0 < g_min <= g0 <= g_max:
            raise ValueError(f"il faut 0 < g_min <= g0 <= g_max, reçu {g_min}, {g0}, {g_max}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, cible=cible,
                        eta=eta, beta_q=beta_q, g0=g0, g_min=g_min, g_max=g_max)
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict) -> None:
        """Ajoute un groupe et l'état du retour qui lui est propre (gain, ω̄)."""
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        group.setdefault("g", group["g0"])
        group.setdefault("omega_bar", 0.0)
        group.setdefault("omega_last", float("nan"))
        group.setdefault("n_regul", 0)

    def _apres_retour(self, group: dict, params: List[torch.Tensor],
                      omega: Optional[float]) -> bool:
        """Point d'extension après le retour, avant le déploiement : ``omega`` est le
        cosinus lu à ce pas (``None`` si aucun pas n'était à juger). Renvoyer ``True``
        saute le déploiement de ce groupe. Ici : rien, le pas suit toujours."""
        return False

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        groupes = []
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if params:
                groupes.append((group, params, all("delta" in self.state[p] for p in params)))

        # --- Ω : le gradient d'arrivée rencontre le pas précédent. Les 3 produits scalaires
        # de chaque tenseur jugé sont lus ensemble : une synchronisation par pas.
        scalaires: List[torch.Tensor] = []
        for _, params, juge in groupes:
            if juge:
                for p in params:
                    d = self.state[p]["delta"]
                    scalaires += [torch.sum(p.grad * d), torch.sum(p.grad * p.grad),
                                  torch.sum(d * d)]
        lus = iter(_vers_hote(scalaires))

        for group, params, juge in groupes:
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            wd: float = group["weight_decay"]

            # --- A′ : le retour change la règle (sommes dans l'ordre des tenseurs) ---
            omega: Optional[float] = None
            if juge:
                dot, n_g, n_d = 0.0, 0.0, 0.0
                for _ in params:
                    dot -= next(lus)
                    n_g += next(lus)
                    n_d += next(lus)
                if n_g > 0.0 and n_d > 0.0:
                    omega = dot / math.sqrt(n_g * n_d)
                    bq = group["beta_q"]
                    group["omega_bar"] = bq * group["omega_bar"] + (1 - bq) * omega
                    log_g = regulate_step(
                        math.log(group["g"]), -group["omega_bar"], -group["cible"], group["eta"],
                        math.log(group["g_min"]), math.log(group["g_max"]),
                    )
                    # exp∘log n'est pas l'identité au bit : la borne est réappliquée sur g
                    group["g"] = min(group["g_max"], max(group["g_min"], math.exp(log_g)))
                    group["omega_last"] = omega
                    group["n_regul"] += 1

            if self._apres_retour(group, params, omega):
                continue  # le pas jugé a été défait : pas de déploiement sur ce gradient

            # --- A puis B : l'intention (la direction du pas) et le déploiement AdamW de gain g ---
            lr_g = group["lr"] * group["g"]
            for p in params:
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["delta"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                m, v, delta = state["exp_avg"], state["exp_avg_sq"], state["delta"]
                delta.copy_(p)

                # ordre des opérations = torch.optim.AdamW (pivot bit-à-bit à g = 1)
                if wd != 0.0:
                    p.mul_(1 - lr_g * wd)
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr_g / bias_correction1
                denom = (v.sqrt() / bias_correction2 ** 0.5).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

                delta.sub_(p).neg_()  # Δθ = θ_{t+1} − θ_t : l'intention, gardée pour le retour

        return loss
