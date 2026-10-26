# CHANTIER_OPTIMISEUR_ABA.md — De la logique A-B-A′ à l'optimiseur `ChronoSpiratonOpt`

> **Destinataires** : Agents d'ingénierie et de recherche (Claude, Gemini, Codex) et mainteneurs du dépôt `spiraton-chantiers-phonetique-et-mesures`.
> **Statut** : Spécification d'implémentation et protocole de confrontation expérimentale (suite logique des Tours 3, 15, 19, 20, 21 et 64).
> **Règle d'or** : Respect absolu de `MANIFESTE.md`, `REFUS.md`, `GRAMMAIRE_ABA.md` et `CLAUDE.md`. **Aucun fichier du canon `spiraton/core/` ne doit être modifié.** `spiraton/experimental/chrono.py` et `spiraton/experimental/edge_controller.py` restent **INCHANGÉS** (réutilisés par import direct). Aucun symbole sans test.

---

## 1. Contexte et problématique

Jusqu'au Tour 64 (`spiraton/experimental/return_training.py`), la logique **A → B → A′** de Spiraton est utilisée dans :
1. **La grammaire et les données** (`GRAMMAIRE_ABA.md`, `spiraton/data/aba_forms.py`) : distinction entre la clôture canonique **F0 (+,+,−)**, les formes d'échappement/maturation (**F0b, F1, F1b, F2, F3**) et l'exclusion des pôles dégénérés **F5+ (+,+,+ : dissipation)** et **F5− (−,−,− : verrouillage/répétition)**.
2. **L'architecture dynamique** (`spiraton/experimental/chrono.py`) : récurrence du second ordre `ChronoSpiraton` :
   $$s_{t+1} = D\big( A(s_t) + B(s_t^2) - C(s_{t-1}) \big) + L(s_t)$$
   et sa variante expérimentale du Tour 3 (`c_outside=True`) qui déplace l'inhibition mémoire $-C(s_{t-1})$ hors de $D$ :
   $$s_{t+1} = D\big( A(s_t) + B(s_t^2) \big) + L(s_t) - C(s_{t-1})$$
3. **La fonction de perte** (`spiraton/training/aba_loss.py` et `perte_alpha_omega` dans `return_training.py`) : exigence d'un retour $A'$ **proche et aligné mais non identique** à $A$ ($\cos(A', A)$ élevé mais distance relative $\Vert{}A' - A\Vert{}/\Vert{}A\Vert{}$ maintenue au-dessus d'un plancher $m > 0$ pour interdire la copie exacte).
4. **La régulation au bord du chaos** (`spiraton/experimental/edge_controller.py` et `spiraton/diagnostics/aba_regulation.py`) : loi générique `regulate_step(prev, obs, target, eta, min, max)` maintenant le rayon spectral $\hat{\rho}_t = r_t / r_{t-1}$ autour de $1.0$ (Actionneur 1, Tour 15/19) et la cohérence de phase retournée $-\cos(s_t, s_{t-W})$ autour de $-0.7$ (Actionneur 2, Tour 20).

**L'angle mort actuel** : Dans `return_training.py` (l. 359), l'entraînement des poids $\theta$ de `ChronoSpiraton` est encore confié à `torch.optim.Adam` standard (assisté d'un `clip_grad_norm_` externe à $1.0$ et d'une clause de repli `bounded=True` si $\ge 6/24$ runs divergent).
**L'objectif de ce chantier** : Transposer la dynamique $A \rightarrow B \rightarrow A'$ directement dans l'espace des paramètres pour créer l'optimiseur natif du projet (`ChronoSpiratonOpt`), capable de remplacer `Adam` et `AdamW`.

---

## 2. Fondement théorique : l'isomorphisme AdamW ↔ ChronoSpiraton + EdgeController

La comparaison terme à terme entre AdamW et les briques existantes du dépôt montre que `spiraton` possède déjà une formulation supérieure à AdamW :

| Mécanisme d'optimisation | Dans Adam / AdamW | Limite structurelle d'AdamW | Dans `ChronoSpiratonOpt` (briques Spiraton) |
| :--- | :--- | :--- | :--- |
| **Direction du pas (1er ordre)** | $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ | Moyenne mobile linéaire aveugle à la chiralité. | Couplage de l'agrégation $A(s_t)$ et des branches chirales de `cell.py` : action centrifuge **D** ($\tanh$, `<DX><OUT>`) et mémoire centripète **L** ($\arctan$, `<LV><IN>`). |
| **Échelle quadratique (2nd ordre)** | Division coordonnée par coordonnée par $\sqrt{v_t} = \sqrt{\text{EMA}(g_t^2)}$ | Détruit la géométrie angulaire globale (le cosinus M1) du tenseur de poids. | Terme quadratique additif $B(s_t^2)$ de `chrono.py` combiné au passage en log-espace borné $\tanh(\log(\vert{}s_t\vert{}^2))$ de `operators.py` et `cell.py`. |
| **Oubli / Régularisation** | *Weight Decay* découplé $-\eta \lambda \theta_{t-1}$ | Tire aveuglément les poids vers $0$ (pousse vers le pôle dégénéré **F5−**). | Inhibition mémoire du second ordre $-C(s_{t-1})$ placée hors de $D$ (`c_outside=True`, Tour 3 de `chrono.py`), complétée par la clôture angulaire $B \to A'$. |
| **Contrôle radial du pas** | Aucun (exige `clip_grad_norm_` externe) | Explosion possible sur surfaces non bornées (`bounded=False`). | **Actionneur 1 (Tour 15)** : `regulate_step` ajuste le gain $g_t \in [0.80, 1.20]$ en lisant $\hat{\rho}_t = r_t / r_{t-1}$ vers la cible $1.0$ ($\eta_\rho = 0.5$). |
| **Contrôle angulaire (Phase)** | Aucun | Dérive libre loin de l'ancrage (**F5+**, dissipation). | **Actionneur 2 (Tour 20)** : `regulate_step` ajuste $\omega_t \in [0, 0.314]$ en lisant $-\cos(\theta_t, \theta_{t-W})$ vers $-0.7$ avec le gain ré-échelonné $\eta_\omega = \eta_\rho / W = 0.05$. |

> **Note sur la co-régulation $g + \omega$** : `edge_controller.py` (l. 453) mentionnait explicitement : *« UN SEUL ACTIONNEUR RÉGULÉ CE TOUR : g reste FIXE à 1.0 (la co-régulation g+ω est un tour ultérieur) »*. `ChronoSpiratonOpt` réalise cette co-régulation $g + \omega$ en utilisant **mot pour mot** la fonction `regulate_step` importée de `spiraton.experimental.edge_controller`.

---

## 3. Spécification du module à créer : `spiraton/experimental/chrono_optimizer.py`

L'agent doit créer le fichier `spiraton/experimental/chrono_optimizer.py` avec le code exact ci-dessous.
**Contraintes d'architecture respectées** :
- Import direct de `regulate_step` depuis `.edge_controller` (zéro duplication de l'organe).
- Compatibilité totale avec `torch.use_deterministic_algorithms(True)` et `float64`.
- Respect des pivots bit-à-bit IEEE754 à `eta_rho = 0.0`, `eta_omega = 0.0` et `c_inhib = 0.0`.

```python
from __future__ import annotations

"""chrono_optimizer.py — Optimiseur à clôture spirale A-B-A′ et co-régulation g + ω.

Fusionne sans les modifier :
  1. La dynamique du second ordre de ``ChronoSpiraton`` (``experimental/chrono.py``) :
     s_{t+1} = D( A(s_t) + B(s_t²) ) + L(s_t) − C(s_{t−1})   [c_outside=True par défaut].
  2. Les branches chirales bornées de ``SpiratonCell`` (``core/cell.py``) et le passage
     logarithmique de ``core/operators.py`` : action dextrogyre ``tanh`` (<DX><OUT>) et
     mémoire lévogyre ``atan`` (<LV><IN>).
  3. L'organe générique ``regulate_step`` (``experimental/edge_controller.py``, importé
     tel quel) cablé en CO-RÉGULATION :
       - Actionneur 1 (échelle g_t, T15) : lit ρ̂_t = r_t / r_{t−1} → cible 1.0.
       - Actionneur 2 (phase ω_t, T20)   : lit −cos(θ_t, θ_{t−W}) → cible −0.7.

PIVOTS ANTI-ARTEFACT (garantis par construction IEEE754) :
  * À ``eta_rho = 0.0`` et ``eta_omega = 0.0`` : ``g_ctrl == g0`` et ``omega_ctrl == omega0``
    bit-à-bit sur tout l'horizon.
  * À ``c_inhib = 0.0`` : ``c_outside=True`` et ``c_outside=False`` coïncident bit-à-bit.
  * Depuis le repos (``s_cur = 0, s_prev = 0``) sous gradient nul : ``Δθ = 0`` au bit.
"""

import math
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
from torch.optim import Optimizer

from .edge_controller import regulate_step


@dataclass(frozen=True)
class ChronoOptimizerConfig:
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
    eps: float = 1e-12


class ChronoSpiratonOpt(Optimizer):
    """Optimiseur A → B → A′ à dynamique Chrono du second ordre et maintien au bord."""

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
        eps: float = 1e-12,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"lr doit être > 0, reçu {lr}")
        if window_w < 1:
            raise ValueError(f"window_w doit valoir >= 1, reçu {window_w}")
        if g_min > g_max:
            raise ValueError(f"g_min ({g_min}) > g_max ({g_max})")
        if omega_min > omega_max:
            raise ValueError(f"omega_min ({omega_min}) > omega_max ({omega_max})")

        if eta_omega is None:
            # Loi de ré-échelonnement T20 : η_ω = η_ρ / W
            eta_omega = eta_rho / float(window_w)

        self.cfg = ChronoOptimizerConfig(
            lr=lr,
            a_agg=a_agg,
            b_quad=b_quad,
            c_inhib=c_inhib,
            d_dextro=d_dextro,
            l_levo=l_levo,
            c_outside=c_outside,
            g0=g0,
            eta_rho=eta_rho,
            g_min=g_min,
            g_max=g_max,
            window_w=window_w,
            phase_target=phase_target,
            omega0=omega0,
            eta_omega=eta_omega,
            omega_min=omega_min,
            omega_max=omega_max,
            eps=eps,
        )

        defaults = dict(
            lr=lr,
            a_agg=a_agg,
            b_quad=b_quad,
            c_inhib=c_inhib,
            d_dextro=d_dextro,
            l_levo=l_levo,
            c_outside=c_outside,
            g0=g0,
            eta_rho=eta_rho,
            g_min=g_min,
            g_max=g_max,
            window_w=window_w,
            phase_target=phase_target,
            omega0=omega0,
            eta_omega=eta_omega,
            omega_min=omega_min,
            omega_max=omega_max,
            eps=eps,
        )
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
            neg_phase_target: float = -group["phase_target"]
            omega0: float = group["omega0"]
            eta_omega: float = group["eta_omega"]
            omega_min: float = group["omega_min"]
            omega_max: float = group["omega_max"]
            eps: float = group["eps"]

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

                # --- 1. SEG_A : Observables et co-régulation (regulate_step) ---
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
                        neg_cos_obs = -cos_w
                    else:
                        neg_cos_obs = 1.0
                    om_cur = regulate_step(
                        om_cur, neg_cos_obs, neg_phase_target, eta_omega, omega_min, omega_max
                    )

                ring.append(p.detach().clone())
                if len(ring) > W:
                    ring.pop(0)

                # --- 2. SEG_B : Excursion du second ordre (Chrono + Cell) ---
                grad_norm = float(torch.linalg.vector_norm(grad).item())
                if grad_norm > eps:
                    grad_unit = grad / (grad_norm + eps)
                else:
                    grad_unit = torch.zeros_like(grad)

                s_input = a_agg * s_cur + (1.0 - a_agg) * grad_unit

                # Terme quadratique modulé en log-espace borné (s'annule exactement si s_input == 0)
                log_sq = torch.tanh(torch.log(torch.clamp(s_input * s_input, min=eps)))
                sq_term = b_quad * s_input.sign() * log_sq

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

                # --- 3. SEG_A_PRIME : Clôture spirale (rayon g_cur + phase om_cur) ---
                p.add_(s_next, alpha=-lr * g_cur)

                if theta_anchor_w is not None and om_cur > 0.0 and grad_norm > eps:
                    pull = math.sin(om_cur) * lr
                    norm_p = float(torch.linalg.vector_norm(p).item())
                    norm_w = float(torch.linalg.vector_norm(theta_anchor_w).item())
                    if norm_p > eps and norm_w > eps:
                        aligned_ref = theta_anchor_w * (norm_p / (norm_w + eps))
                        p.lerp_(aligned_ref, weight=pull)

                state["s_prev"].copy_(s_cur)
                state["s_cur"].copy_(s_next)
                state["r_prev"] = float(torch.linalg.vector_norm(s_next).item())
                state["g_ctrl"] = g_cur
                state["omega_ctrl"] = om_cur
                state["step"] = t + 1

        return loss
```

---

## 4. Protocole d'évaluation face à Adam / AdamW (Sur le banc d'essai T64)

Une fois `spiraton/experimental/chrono_optimizer.py` et `tests/test_chrono_optimizer.py` en place et passés au vert (`pytest`), l'agent devra conduire l'évaluation comparative sur le protocole exact de `spiraton/experimental/return_training.py` sans modifier les splits ni les seuils :

1. **Splits gelés (`dataset_aba.txt`, md5 `a0f9fad13db6db551a23b56b46f5748b`)** :
   - `TRAIN` : lignes 101 à 1000
   - `VALIDATION` : lignes 3001 à 3500
   - `TEST` (scellé, une seule ouverture) : lignes 3501 à 5001
2. **Grille d'entraînement close (24 runs)** :
   - `INIT_SCALES = (0.1, 0.6)` × `LRS = (1e-3, 1e-2)` × `GRAINES = (0, 1, 2)` × `CARTOUCHES = ("S", "P")`.
3. **Hypothèses mesurables (Confrontation `ChronoSpiratonOpt` vs `Adam` / `AdamW`)** :
   - **Stabilité intrinsèque (H_stab)** : Sans `clip_grad_norm_` externe et avec `bounded=False`, `ChronoSpiratonOpt` maintient `n_diverges < SEUIL_DIVERGENCES_REPLI` ($< 6/24$), en particulier sur `init_scale = 0.6` où les méthodes non bornées explosent.
   - **Gardes anti-collapse (Q6 / Q7)** : Sur la validation, la configuration retenue respecte $R_{\text{disp}} \ge 0.20$ (pas d'effondrement en un point fixe) et $\bar{d} \ge 0.05$ (pas de copie déguisée).
   - **Porteur $\Delta_S$ (C1–C4)** : Comparaison appariée par cycle sur le split `TEST` de $\Delta_i(\Phi_{\text{ChronoOpt}}) - \Delta_i(\Phi_{\text{AdamW}})$ via `sign_test_exact` ($p < 0.01$).

---

## 5. Feuille de route séquentielle pour l'Agent (Checklist d'exécution)

- [ ] **Étape 1** : Vérifier l'état initial de la suite de tests existante (`pytest -q`).
- [ ] **Étape 2** : Créer `spiraton/experimental/chrono_optimizer.py` (section 3 ci-dessus) et exposer `ChronoSpiratonOpt`, `ChronoOptimizerConfig` dans `spiraton/experimental/__init__.py` si pertinent (sans toucher à `spiraton/core/`).
- [ ] **Étape 3** : Créer `tests/test_chrono_optimizer.py` et exécuter `pytest tests/test_chrono_optimizer.py -v`. Les 7 tests (Porte 0, Pivot $\eta=0$, Pivot repos, Pivot Tour 3 `c_outside`, Co-régulation $g+\omega$, Déterminisme P0-c `float64`, Non-divergence `init_scale=0.6`) doivent passer à 100 %.
- [ ] **Étape 4** : Créer un script de benchmark non destructif (ex. `examples/benchmark_chrono_opt_vs_adamw.py`) qui appelle `train_run` de `return_training.py` en opposant à graine identique `torch.optim.Adam`, `torch.optim.AdamW(weight_decay=1e-2)` et `ChronoSpiratonOpt` (avec `c_outside=True` et `c_outside=False`), et consigner les résultats dans un rapport horodaté et scellé par `.sha256`.