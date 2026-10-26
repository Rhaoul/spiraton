"""chrono_duration.py — T65 : chantier 6, la durée, et qui la tient.

Mesure le RÉGIME DE DURÉE de la dynamique apprise au T64 (``ChronoSpiraton(19)``,
poids gelés ``TOUR64_POIDS``) : les poids entraînés à K=2 sont itérés à K=64 et
qualifiés par trois familles d'instruments gelés (TOUR65_EMISSION.md §4.1) :

* **(I-1)** ``ρ₀`` — rayon spectral de la matrice compagnon 38×38 de la
  linéarisation à l'origine ``J₀ = [[DA+L, −DC], [I, 0]]`` (AVEUGLE à ``B(s²)``
  par construction, dit d'avance — portée Q1) ;
* **(I-2)** ``λ_i`` / ``Λ`` — exposant empirique de durée par point (médiane sur
  les 900 points TRAIN, l.717 : jamais la moyenne) ;
* **(I-3)** ``f_div`` / ``H₅₀`` — fraction divergée dans les 64 pas et horizon
  médian des divergents.

Attribution (C2/C3) : ablation de ``−C(s_{t−1})`` et de ``B(s²)`` sous spec
STRICTEMENT identique au T64 (TOUR64_SPEC.json ; seule déviation déclarée :
50 graines au lieu de 3), boucle d'entraînement RE-DÉRIVÉE de
``return_training.train_run`` (l'ablation s'insère entre l'init seedée et
l'optimiseur — pour la variante FULL la trajectoire d'optimisation est
BIT-IDENTIQUE, et la gate C4 le prouve au sha256). ``chrono.py`` et
``return_training.py`` ne sont PAS modifiés : ils sont importés en lecture.

Rivaux obligatoires (§4.3) : ridge λ=1 itérée (``ρ(W)`` exact, ``Λ_ridge``),
premier ordre (= variante ``MINUS_C``), figé T63 (12 graines, repère).

CLI à étapes MATÉRIELLES (chaque étape post-gel vérifie le jeton β
``TOUR65_FREEZE.json`` et le sceau de l'étape précédente) :
``etalonnage`` / ``p0b`` / ``p0c`` / ``p1`` / ``gel`` / ``regime`` /
``ablation`` / ``sigma`` / ``attribution`` / ``info``.

Gate Q11 (matérielle) : aucune route de ce module vers le split de verdict
brûlé du T64 ; seuls TRAIN et VALIDATION sont matérialisés ; la liste des
artefacts T64 lus est close (§2.2) et auto-vérifiée (``scan_q11``).

Déterministe : seeds fixés partout, float64, CPU,
``torch.use_deterministic_algorithms(True)`` dans le runner. Les verdicts
vivent dans TOUR65_DEPLOIEMENT.md, pas ici.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..diagnostics import textual_return_dynamics as trd
from ..diagnostics.freeze_token import freeze_token, read_token, verify_freeze, write_token
from . import return_training as rt
from .chrono import ChronoSpiraton

# --- Constantes gelées (TOUR65_EMISSION.md §4) --------------------------------

K_MAX = 64                    #: horizon gelé (§4.1).
THETA_DIV = 1e6               #: seuil de divergence (valeur de chrono.stability_scan).
EPS = rt.EPS                  #: 1e-12 — le même objet que M1/T64.
INIT_SCALE = 0.1              #: config unique gelée (TOUR64_SELECTION, S).
LR_VERDICT = 0.01             #: config unique gelée.
LR_INFO = 1e-3                #: INFO-robustesse (§4.4), hors-verdict.

ZONE_NEUTRE_LAMBDA = 0.005    #: |Λ| < 0,005 ⇒ régime neutre (§4.1 I-2).
ZONE_MORTE_RHO = 0.01         #: |ρ₀ − 1| < 0,01 ⇒ composante ρ₀ NON-MESURE (§3.3).
SEUIL_F_DIV = 0.90            #: DIVERGENT si f_div ≥ 0,90 (§4.8).
SEUIL_F_BORNE = 0.10          #: BORNÉ si f_div ≤ 0,10 (et norme bornée ≤ 10×).
FACTEUR_BORNE = 10.0
SEUIL_POINT_FIXE = 1e-6       #: garde point-fixe (§4.2, branche BORNÉ).

DELTA_LAMBDA_MIN = 0.05       #: |médiane_g d| ≥ 0,05 (C2/C3, dérivé §4.5).
SEUIL_P = 0.01                #: sign-test exact bilatéral.
SIGMA_D_MAX = 0.1034          #: σ̂_d > 0,1034 ⇒ magnitude NON-MESURE (§4.5).
Z_SUM = 3.418                 #: z_{0.995} + z_{0.80} (dimensionnement).

GRAINES_ABLATION: Tuple[int, ...] = tuple(range(50))   #: {0..49}, seule déviation.
GRAINES_INFO_LR: Tuple[int, ...] = tuple(range(10))    #: INFO lr=1e-3.
GRAINES_C4: Tuple[int, ...] = (0, 1, 2)                #: gate C4 (parité au bit).
VARIANTES: Tuple[str, ...] = ("FULL", "MINUS_C", "MINUS_B", "MINUS_CB")
ABLATIONS: Dict[str, Tuple[str, ...]] = {
    "FULL": (), "MINUS_C": ("C",), "MINUS_B": ("B",), "MINUS_CB": ("B", "C"),
}
N_DESCENTES_VERDICT = 200     #: 4 variantes × 50 graines.
QUORUM_DESCENTE = 167         #: perte descendue ≥ 20 % sur ≥ 167/200 (C5).
SEUIL_REPLI_DIV = 50          #: ≥ 50/200 divergences ⇒ repli bounded (INFO, I8).

FIGE_SEEDS: Tuple[int, ...] = tuple(range(12))         #: repère T63 (§4.3).
RIDGE_LAMBDA = 1.0            #: ridge λ=1, recette T64 exacte.

R_STAR_SEED = 65              #: sous-échantillon gelé (I-5).
R_STAR_N = 100
R_STAR_GRID: Tuple[float, ...] = tuple(2.0 ** k for k in range(-6, 4))

MUR_PLANCHER_S = 600.0        #: mur = max(4·T̂, 600 s), plafond dur 3600 s (§4.9).
MUR_PLAFOND_S = 3600.0
MUR_FACTEUR = 4.0

#: Parités de chaîne SANS ouverture du split brûlé (§2.3, valeurs gelées au bit).
PARITE_RELOAD_DELTA_VAL: Tuple[float, ...] = (
    0.3486412467388898, 0.3435198556206523, 0.34861286042047285,
)
PARITE_M_S = 0.3656003930703504
PARITE_RIDGE_VAL = 0.22270548365817838

#: Artefacts du tour (racine spiraton-enhanced, hors git).
ART: Dict[str, str] = {
    "etalonnage": "TOUR65_ETALONNAGE.json",
    "p0b": "TOUR65_P0B.json",
    "p0c": "TOUR65_P0C.json",
    "p1": "TOUR65_P1.json",
    "instruments": "TOUR65_INSTRUMENTS.json",
    "spec": "TOUR65_SPEC.json",
    "predictions": "TOUR65_PREDICTIONS.json",
    "gel": "TOUR65_GEL.json",
    "freeze": "TOUR65_FREEZE.json",
    "regime": "TOUR65_REGIME.json",
    "ablation": "TOUR65_ABLATION.json",
    "courbes": "TOUR65_COURBES.json",
    "sigma": "TOUR65_SIGMA.json",
    "attribution": "TOUR65_ATTRIBUTION.json",
    "info": "TOUR65_INFO.json",
}
POIDS_DIR = "TOUR65_POIDS"
POIDS_T64 = "TOUR64_POIDS"
RUN_T64_PREFIX = "S_is0.1_lr0.01_seed"   #: porteur T64 (cartouche S SEULE, §2.4).

#: Les 19 chemins du digest β (§4.11, liste close), relatifs à la racine.
FREEZE_PATHS: Tuple[str, ...] = (
    "TOUR65_EMISSION.md",
    "TOUR65_SPEC.json",
    "TOUR65_PREDICTIONS.json",
    "TOUR65_INSTRUMENTS.json",
    "TOUR65_P1.json",
    "TOUR64_SPEC.json",
    "TOUR64_SELECTION.json",
    "TOUR64_TRAIN.json",
    "TOUR64_POIDS/S_is0.1_lr0.01_seed0.bin",
    "TOUR64_POIDS/S_is0.1_lr0.01_seed1.bin",
    "TOUR64_POIDS/S_is0.1_lr0.01_seed2.bin",
    "dataset_aba.txt",
    "spiraton/spiraton/experimental/chrono.py",
    "spiraton/spiraton/experimental/return_training.py",
    "spiraton/spiraton/diagnostics/textual_return_dynamics.py",
    "spiraton/spiraton/data/tokenizer_bridge.py",
    "Tokenizer/python/spiraton_tokenizer/alpha_omega_text.py",
    "Tokenizer/bin/libspiratontokenizer.so",
    "spiraton/spiraton/experimental/chrono_duration.py",
)

#: Liste CLOSE des artefacts T64 lisibles par ce tour (§2.2, gate Q11 volet ii).
ARTEFACTS_T64_LICITES: Tuple[str, ...] = (
    "TOUR64_SPEC.json", "TOUR64_SELECTION.json", "TOUR64_TRAIN.json",
)


# --- (I-1) Matrice compagnon et rayon spectral --------------------------------

def matrices_np(model: ChronoSpiraton) -> Dict[str, np.ndarray]:
    """Les 5 matrices float64 (agissant à gauche : ``sortie = W @ s``)."""
    return {nom: getattr(model, nom).weight.detach().cpu().numpy().astype(np.float64)
            for nom in ("A", "B", "C", "D", "L")}


def compagnon(mats: Dict[str, np.ndarray], s: Optional[np.ndarray] = None) -> np.ndarray:
    """Matrice compagnon 38×38 de la linéarisation en ``s`` (§4.1 I-1).

    ``J(s) = [[D(A + 2·B·diag(s)) + L, −DC], [I, 0]]`` ; ``s=None`` ⇒ origine
    (``∂B(s⊙s)/∂s|₀ = 0`` : la linéarisation à 0 est AVEUGLE au quadratique).
    """
    a, b, c, d_, l = mats["A"], mats["B"], mats["C"], mats["D"], mats["L"]
    dim = a.shape[0]
    haut_gauche = d_ @ a + l
    if s is not None:
        haut_gauche = haut_gauche + 2.0 * (d_ @ b @ np.diag(np.asarray(s, dtype=np.float64)))
    j = np.zeros((2 * dim, 2 * dim), dtype=np.float64)
    j[:dim, :dim] = haut_gauche
    j[:dim, dim:] = -(d_ @ c)
    j[dim:, :dim] = np.eye(dim)
    return j


def rho_compagnon(mats: Dict[str, np.ndarray], s: Optional[np.ndarray] = None) -> float:
    """``ρ(s)`` = max |valeurs propres| de la compagnon (numpy eigvals, float64)."""
    return float(np.max(np.abs(np.linalg.eigvals(compagnon(mats, s)))))


# --- (I-2/I-3) Trajectoires K=64 : λ_i, Λ, f_div, H₅₀ -------------------------

@dataclass(frozen=True)
class Trajectoires:
    """Sortie du déroulé K=64 sur n points (convention T63/T64 : s_prev = repos).

    ``T`` = premier pas hors-borne (‖s_t‖ > Θ ou non fini), sinon K_max ;
    ``t_prime`` = dernier pas FINI ; ``lam`` = exposant empirique par point ;
    ``norme_0/derniere/max`` = ‖s_0‖, ‖s_{T'}‖, max des normes finies ;
    ``etats_63/64`` = états aux pas 63/64 (garde point-fixe) ; ``etats`` =
    trace complète (n, K, d) si demandée (best_return_step, INFO).
    """

    n: int
    T: np.ndarray
    t_prime: np.ndarray
    diverged: np.ndarray
    lam: np.ndarray
    norme_0: np.ndarray
    norme_derniere: np.ndarray
    norme_max: np.ndarray
    etats_63: Optional[np.ndarray]
    etats_64: Optional[np.ndarray]
    etats: Optional[np.ndarray]

    @property
    def f_div(self) -> float:
        return float(np.mean(self.diverged))

    @property
    def h50(self) -> Optional[float]:
        """Médiane des T_i sur les points divergents SEULEMENT (N-A sinon)."""
        if not bool(self.diverged.any()):
            return None
        return float(np.median(self.T[self.diverged]))

    @property
    def n_tprime_zero(self) -> int:
        """Points non finis dès le pas 1 (λ_i := log(Θ/‖s_0‖), comptés à part)."""
        return int((self.t_prime == 0).sum())


@torch.no_grad()
def derouler(step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             s0_np: np.ndarray, *, k_max: int = K_MAX, theta: float = THETA_DIV,
             garder_etats: bool = False) -> Trajectoires:
    """Déroule ``s_{t+1} = step_fn(s_t, s_{t−1})`` depuis chaque ligne de s0.

    Les lignes sont INDÉPENDANTES (opérations par lot élément/matrice) : la
    divergence d'un point ne contamine pas les autres. ``s_{−1} = 0``.
    """
    if s0_np.ndim == 1:
        s0_np = s0_np[None, :]
    s0_np = np.ascontiguousarray(s0_np.astype(np.float64))
    s0 = torch.from_numpy(s0_np)
    n = s0.shape[0]
    prev, cur = torch.zeros_like(s0), s0
    # MÊME route numpy que les normes de pas (exactitude E1 : ratio = 1 au bit).
    norme_0 = np.linalg.norm(s0_np, axis=1)
    T = np.full(n, k_max, dtype=np.int64)
    t_prime = np.zeros(n, dtype=np.int64)
    diverged = np.zeros(n, dtype=bool)
    done = np.zeros(n, dtype=bool)
    norme_derniere = norme_0.copy()
    norme_max = norme_0.copy()
    etats_63 = etats_64 = None
    trace: List[np.ndarray] = []
    for t in range(1, k_max + 1):
        nxt = step_fn(cur, prev)
        arr = nxt.numpy()
        fini = np.isfinite(arr).all(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            nrm = np.linalg.norm(arr, axis=1)
        actifs = ~done
        maj = actifs & fini
        norme_derniere[maj] = nrm[maj]
        t_prime[maj] = t
        norme_max[maj] = np.maximum(norme_max[maj], nrm[maj])
        sortie = actifs & (~fini | (nrm > theta))
        T[sortie] = t
        diverged |= sortie
        done |= sortie
        if garder_etats:
            trace.append(arr.copy())
        if t == k_max - 1:
            etats_63 = arr.copy()
        if t == k_max:
            etats_64 = arr.copy()
        prev, cur = cur, nxt
    lam = np.empty(n, dtype=np.float64)
    zero_tp = t_prime == 0
    with np.errstate(divide="ignore"):
        lam[zero_tp] = np.log(theta / norme_0[zero_tp])
        ok = ~zero_tp
        lam[ok] = (np.log((norme_derniere[ok] + EPS) / (norme_0[ok] + EPS))
                   / t_prime[ok])
    return Trajectoires(
        n=n, T=T, t_prime=t_prime, diverged=diverged, lam=lam,
        norme_0=norme_0, norme_derniere=norme_derniere, norme_max=norme_max,
        etats_63=etats_63, etats_64=etats_64,
        etats=np.stack(trace, axis=1) if garder_etats else None,
    )


def pas_chrono(model: ChronoSpiraton) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Le pas de ``chrono.py`` INCHANGÉ (méthode ``step``), en lecture."""
    return model.step


def pas_ridge(theta: np.ndarray) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Rival linéaire itéré : ``s_{t+1} = W s_t + b`` (recette T64, §4.3)."""
    w = torch.from_numpy(np.ascontiguousarray(theta[:-1].astype(np.float64)))
    b = torch.from_numpy(np.ascontiguousarray(theta[-1].astype(np.float64)))

    def f(cur: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        return cur @ w + b
    return f


def stats_trajectoires(tr: Trajectoires) -> Dict[str, object]:
    """Le cartouche de chiffres publié pour UN modèle (médianes, l.717)."""
    return {
        "n": tr.n,
        "Lambda": float(np.median(tr.lam)),
        "lam_std": float(tr.lam.std()),
        "lam_min": float(tr.lam.min()),
        "lam_max": float(tr.lam.max()),
        "lam_moyenne_INFO": float(tr.lam.mean()),
        "f_div": tr.f_div,
        "H50": tr.h50,
        "n_tprime_zero": tr.n_tprime_zero,
        "T_mediane": float(np.median(tr.T)),
        "mediane_norme_0": float(np.median(tr.norme_0)),
        "mediane_norme_max": float(np.median(tr.norme_max)),
        "mediane_norme_derniere": float(np.median(tr.norme_derniere)),
        "finitude_lam_100pct": bool(np.isfinite(tr.lam).all()),
    }


def garde_point_fixe(tr: Trajectoires) -> Dict[str, object]:
    """Garde point-fixe (§4.2, branche BORNÉ) : médiane ‖s_64−s_63‖/(‖s_63‖+eps)."""
    if tr.etats_63 is None or tr.etats_64 is None:
        return {"mesurable": False}
    fini = (np.isfinite(tr.etats_63).all(axis=1)
            & np.isfinite(tr.etats_64).all(axis=1))
    if not fini.any():
        return {"mesurable": False, "n_finis": 0}
    d63 = np.linalg.norm(tr.etats_64[fini] - tr.etats_63[fini], axis=1)
    n63 = np.linalg.norm(tr.etats_63[fini], axis=1)
    ratio = float(np.median(d63 / (n63 + EPS)))
    return {
        "mesurable": True,
        "n_finis": int(fini.sum()),
        "ratio_median": ratio,
        "seuil": SEUIL_POINT_FIXE,
        "mord": bool(ratio < SEUIL_POINT_FIXE),
        "mediane_norme_s64": float(np.median(np.linalg.norm(tr.etats_64[fini], axis=1))),
    }


def regime_partition(f_div: float, mediane_norme_max: float,
                     mediane_norme_0: float) -> str:
    """Partition close du régime (§4.2 C1) : DIVERGENT / BORNÉ / BASSIN."""
    if f_div >= SEUIL_F_DIV:
        return "DIVERGENT"
    if f_div <= SEUIL_F_BORNE and mediane_norme_max <= FACTEUR_BORNE * mediane_norme_0:
        return "BORNE"
    return "BASSIN"


# --- Sign-test exact : k*, puissances (récurrence Fraction, §4.5) -------------

def binom_pmf_fraction(n: int, p: Fraction) -> List[Fraction]:
    """pmf binomiale EXACTE par récurrence (aucun flottant avant la fin)."""
    q = 1 - p
    pmf = [q ** n]
    for k in range(n):
        pmf.append(pmf[-1] * (n - k) * p / ((k + 1) * q))
    return pmf


def k_star_sign_test(n: int, seuil: Fraction = Fraction(1, 100)) -> int:
    """Plus petit k tel que le sign-test bilatéral (½) rende p < seuil."""
    pmf = binom_pmf_fraction(n, Fraction(1, 2))
    tail = Fraction(0)
    tails = [Fraction(0)] * (n + 2)
    for k in range(n, -1, -1):
        tail += pmf[k]
        tails[k] = tail
    for k in range(0, n + 1):
        if 2 * tails[k] < seuil:
            return k
    raise RuntimeError("aucun k n'atteint le seuil (n trop petit)")


def puissance_sign_test(n: int, k_star: int, p: Fraction) -> float:
    """P(X ≥ k*) sous Bin(n, p) — puissance exacte, convertie en float à la fin."""
    pmf = binom_pmf_fraction(n, p)
    return float(sum(pmf[k_star:], Fraction(0)))


# --- Boucle d'ablation RE-DÉRIVÉE de train_run (T64) — gate C4 ----------------

def entrainer_variante(m1, variante: str, seed: int, lr: float,
                       a_tr: np.ndarray, ap_tr: np.ndarray,
                       a_val: np.ndarray, ap_val: np.ndarray,
                       s_raw_val: np.ndarray, m: float,
                       *, epochs_max: int = rt.EPOCHS_MAX,
                       bounded: bool = False) -> "rt.RunResult":
    """UN run sous spec T64 EXACTE, ablation insérée entre init et optimiseur.

    Miroir terme à terme de ``return_training.train_run`` (NON MODIFIÉ, importé
    en lecture) ; la SEULE insertion est la mise à zéro + gel
    (``requires_grad=False``) des matrices de la variante APRÈS l'init seedée —
    mathématiquement identique à la suppression du terme, et NEUTRE pour le RNG
    (aucun tirage). Pour ``variante='FULL'`` la trajectoire d'optimisation est
    BIT-IDENTIQUE à ``train_run`` (gate C4, prouvée au sha256 des poids).
    """
    if variante not in ABLATIONS:
        raise ValueError(f"variante inconnue : {variante}")
    torch.manual_seed(seed)
    model = ChronoSpiraton(state_size=rt.STATE_SIZE, init_scale=INIT_SCALE,
                           bounded=bounded, c_outside=False).double()
    for nom in ABLATIONS[variante]:
        lin = getattr(model, nom)
        with torch.no_grad():
            lin.weight.zero_()
        lin.weight.requires_grad_(False)
    gen = torch.Generator()
    gen.manual_seed(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=rt.ADAM_BETAS,
                           eps=rt.ADAM_EPS, weight_decay=0.0)
    ta = torch.from_numpy(np.ascontiguousarray(a_tr))
    tap = torch.from_numpy(np.ascontiguousarray(ap_tr))
    n = ta.shape[0]
    pairs = rt.paires_dispersion(a_val.shape[0])

    with torch.no_grad():
        loss_init = float(rt.perte_alpha_omega(model, ta, tap, m).item())

    courbes: Dict[str, List[float]] = {
        "perte_train": [], "delta_med_val": [], "r_disp": [], "dbar": [],
    }
    best_val = -math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = -1
    sans_amelioration = 0
    diverged = False
    early = False
    epochs_run = 0

    for epoch in range(1, epochs_max + 1):
        perm = torch.randperm(n, generator=gen)
        for k0 in range(0, n, rt.BATCH):
            idx = perm[k0:k0 + rt.BATCH]
            opt.zero_grad()
            loss = rt.perte_alpha_omega(model, ta[idx], tap[idx], m)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rt.CLIP_NORM)
            opt.step()
        if diverged:
            break
        epochs_run = epoch
        with torch.no_grad():
            lt = float(rt.perte_alpha_omega(model, ta, tap, m).item())
        if not math.isfinite(lt):
            diverged = True
            break
        x_val = rt.appliquer_chrono(model, a_val)
        if not np.isfinite(x_val).all():
            diverged = True
            break
        dv = rt.deltas(m1, x_val, ap_val, s_raw_val)
        dmed = float(np.median(dv))
        r_disp, dbar = rt.gardes_collapse(x_val, a_val, pairs)
        courbes["perte_train"].append(lt)
        courbes["delta_med_val"].append(dmed)
        courbes["r_disp"].append(r_disp)
        courbes["dbar"].append(dbar)
        if dmed > best_val:
            best_val = dmed
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            sans_amelioration = 0
        else:
            sans_amelioration += 1
            if sans_amelioration >= rt.PATIENCE:
                early = True
                break

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        best_val = float("nan")
    loss_min = min(courbes["perte_train"]) if courbes["perte_train"] else float("nan")
    descente = ((loss_init - loss_min) / abs(loss_init)
                if courbes["perte_train"] and abs(loss_init) > 0 else float("nan"))
    return rt.RunResult(
        cartouche="S", init_scale=INIT_SCALE, lr=lr, seed=seed,
        diverged=diverged, epochs_run=epochs_run, best_epoch=best_epoch,
        best_delta_val=best_val, loss_init=loss_init, loss_min=loss_min,
        descente_rel=descente, early_stopped=early, courbes=courbes,
        state=best_state, state_sha256=rt.hash_etat(best_state),
    )


def modele_variante_depuis_etat(state: Dict[str, torch.Tensor]) -> ChronoSpiraton:
    """Reconstruit un modèle (équation pure) depuis un state gelé (matrices
    ablatées = zéros dans le state ⇒ équation effective identique)."""
    return rt.modele_depuis_poids(INIT_SCALE, state)


# --- Chargement des poids T64 (cartouche S SEULE, §2.4) -----------------------

def charger_poids_t64(root: Path, seed: int) -> Dict[str, torch.Tensor]:
    """Relit un poids T64 du porteur S ``is0.1_lr0.01`` (JAMAIS réécrit)."""
    return rt.charger_poids(root / POIDS_T64 / f"{RUN_T64_PREFIX}{seed}.bin")


def modele_t64(root: Path, seed: int, *, bounded: bool = False) -> ChronoSpiraton:
    """Modèle T64 rechargé ; ``bounded=True`` = lecture INFO (tanh terminal)."""
    state = charger_poids_t64(root, seed)
    torch.manual_seed(0)
    model = ChronoSpiraton(state_size=rt.STATE_SIZE, init_scale=INIT_SCALE,
                           bounded=bounded, c_outside=False).double()
    model.load_state_dict(state)
    return model


def modele_chirurgical(root: Path, seed: int, matrices: Tuple[str, ...]) -> ChronoSpiraton:
    """C7 (INFO) : poids T64 avec matrice(s) mise(s) à zéro SANS ré-entraînement."""
    model = modele_t64(root, seed)
    with torch.no_grad():
        for nom in matrices:
            getattr(model, nom).weight.zero_()
    return model


# --- Gardes matérielles (ordre α-β-γ) -----------------------------------------

def verifier_gel(root: Path) -> None:
    """Vérifie le jeton β T65 — dérive ⇒ ARRÊT (retouche = candidat mort)."""
    token = read_token(root / ART["freeze"])
    report = verify_freeze(token)
    if not report.ok:
        raise RuntimeError(f"gel β dérivé : {report.drifts} — ARRÊT")


def exiger(root: Path, *cles: str) -> None:
    for cle in cles:
        if not (root / ART[cle]).is_file():
            raise RuntimeError(
                f"artefact {ART[cle]} absent : l'étape refuse de s'exécuter "
                f"(ordre MATÉRIEL du protocole)")


def sceller(root: Path, cle: str) -> str:
    p = root / ART[cle]
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    (root / (ART[cle] + ".sha256")).write_text(sha + "\n", encoding="utf-8")
    return sha


def verifier_sceau(root: Path, cle: str) -> None:
    p = root / ART[cle]
    seal = root / (ART[cle] + ".sha256")
    if not seal.is_file():
        raise RuntimeError(f"sceau {seal.name} absent — ARRÊT")
    attendu = seal.read_text(encoding="utf-8").strip()
    reel = hashlib.sha256(p.read_bytes()).hexdigest()
    if reel != attendu:
        raise RuntimeError(f"sceau {seal.name} brisé : {reel} != {attendu} — ARRÊT")


def budget_verifier(root: Path, t_consomme: float) -> None:
    """Mur de budget (§4.9) : dépassement ⇒ ARRÊT (I9), jamais de dégradation."""
    etal = json.loads((root / ART["etalonnage"]).read_text(encoding="utf-8"))
    mur = float(etal["mur_s"])
    if t_consomme > mur:
        raise RuntimeError(
            f"MUR DE BUDGET dépassé : {t_consomme:.1f}s > {mur:.1f}s — ARRÊT (I9)")


# --- Gate Q11 : scan matériel du code neuf ------------------------------------

def scan_q11(fichiers: Sequence[Path]) -> Dict[str, object]:
    """Volet (i) de Q11 : aucun des jetons interdits dans le code neuf.

    Les jetons sont construits par concaténation pour que CE scanner ne se
    déclenche pas lui-même (le scan est bit-à-bit sur le contenu des fichiers).
    """
    interdits = ["35" + "01", "50" + "01", '"TE' + 'ST"', "'TE" + "ST'"]
    occurrences: List[Dict[str, str]] = []
    for f in fichiers:
        texte = Path(f).read_text(encoding="utf-8")
        for jeton in interdits:
            if jeton in texte:
                occurrences.append({"fichier": str(f), "jeton": jeton})
    return {"fichiers_scannes": [str(f) for f in fichiers],
            "jetons_interdits_construits": interdits,
            "occurrences": occurrences,
            "PASS": len(occurrences) == 0}


# --- Collecte commune ----------------------------------------------------------

def _instrument():
    return trd.charger_instrument()


def _splits(tok, m1, root: Path) -> Tuple["rt.SplitAgg", "rt.SplitAgg"]:
    tr = rt.charger_split(tok, m1, root, "TRAIN", rt.SPLITS["TRAIN"])
    va = rt.charger_split(tok, m1, root, "VALIDATION", rt.SPLITS["VALIDATION"])
    return tr, va


def _ridge_train(tr: "rt.SplitAgg") -> np.ndarray:
    """Ridge λ=1 refittée sur TRAIN S (recette T64 : lstsq augmentée, float64)."""
    return rt.ajuster_ridge(tr.a["S"], tr.ap["S"], RIDGE_LAMBDA)


# --- Étapes du protocole --------------------------------------------------------

def etape_p0b(root: Path) -> dict:
    """P0-b : substrats EXACTS du quatuor (E1/E2/E3 + ablation + gradient).

    Doublé par ``tests/test_chrono_duration.py`` (pytest) ; l'étape écrit la
    preuve dans un artefact (100 % ou FAUX).
    """
    d = rt.STATE_SIZE
    # E1 : A=B=C=D=0, L=I ⇒ s_t = s_0 au bit ⇒ λ=0 EXACT, f_div=0.
    model = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D"):
            getattr(model, nom).weight.zero_()
        model.L.weight.copy_(torch.eye(d, dtype=torch.float64))
    rng = np.random.default_rng(650)
    pts = rng.standard_normal((16, d))
    tr = derouler(pas_chrono(model), pts, garder_etats=True)
    e1_etats = bool((tr.etats[:, -1, :] == pts).all())
    e1 = bool((tr.lam == 0.0).all() and tr.f_div == 0.0 and e1_etats)
    # E2 : A=I, B=C=0, D=αI, L=0 ⇒ s_t = α^t s_0 ⇒ λ = log α ; ρ₀ = α exact.
    alpha = 1.5
    model2 = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    with torch.no_grad():
        model2.A.weight.copy_(torch.eye(d, dtype=torch.float64))
        model2.B.weight.zero_()
        model2.C.weight.zero_()
        model2.D.weight.copy_(alpha * torch.eye(d, dtype=torch.float64))
        model2.L.weight.zero_()
    tr2 = derouler(pas_chrono(model2), pts)
    e2_lam = bool(np.max(np.abs(tr2.lam - math.log(alpha)) / math.log(alpha)) <= 1e-12)
    rho0_e2 = rho_compagnon(matrices_np(model2))
    e2 = bool(e2_lam and rho0_e2 == alpha)
    # E3 : compagnon vs racines de la récurrence linéaire (route indépendante).
    rng3 = np.random.default_rng(651)
    diag_a = rng3.uniform(-0.9, 0.9, d)
    diag_c = rng3.uniform(-0.5, 0.5, d)
    diag_l = rng3.uniform(-0.9, 0.9, d)
    model3 = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    with torch.no_grad():
        model3.A.weight.copy_(torch.diag(torch.from_numpy(diag_a)))
        model3.B.weight.zero_()
        model3.C.weight.copy_(torch.diag(torch.from_numpy(diag_c)))
        model3.D.weight.copy_(torch.eye(d, dtype=torch.float64))
        model3.L.weight.copy_(torch.diag(torch.from_numpy(diag_l)))
    racines = []
    for j in range(d):
        mm, cc = diag_a[j] + diag_l[j], diag_c[j]
        disc = complex(mm * mm - 4.0 * cc)
        r = np.sqrt(disc)
        racines.extend([abs((mm + r) / 2.0), abs((mm - r) / 2.0)])
    rho_racines = float(max(racines))
    rho_comp = rho_compagnon(matrices_np(model3))
    e3 = bool(abs(rho_comp - rho_racines) <= 1e-10)
    # Ablation exacte : forward MINUS_C bit-identique au calcul sans le terme C.
    torch.manual_seed(7)
    m4 = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    with torch.no_grad():
        m4.C.weight.zero_()
    s = torch.from_numpy(rng.standard_normal((8, d)))
    sp = torch.from_numpy(rng.standard_normal((8, d)))
    ref = m4.D(m4.A(s) + m4.B(s * s)) + m4.L(s)
    abl_exacte = bool(torch.equal(m4.step(s, sp), ref))
    payload = {
        "tour": 65, "etape": "P0B",
        "E1_identite": {"PASS": e1, "lam_tous_zero_exact": bool((tr.lam == 0.0).all()),
                        "f_div": tr.f_div, "etats_au_bit": e1_etats},
        "E2_geometrique": {"PASS": e2, "alpha": alpha, "rho0": rho0_e2,
                           "rho0_exact": bool(rho0_e2 == alpha),
                           "lam_tol_rel": 1e-12},
        "E3_compagnon_vs_racines": {"PASS": e3, "rho_compagnon": rho_comp,
                                    "rho_racines": rho_racines,
                                    "ecart": abs(rho_comp - rho_racines)},
        "ablation_exacte_bit": {"PASS": abl_exacte},
        "PASS": bool(e1 and e2 and e3 and abl_exacte),
    }
    if not payload["PASS"]:
        raise RuntimeError(f"P0-b FAIL : {payload}")
    return payload


def etape_p0c(root: Path) -> dict:
    """P0-c : UNE descente complète FULL graine 0 (preuve de déterminisme,
    à exécuter DEUX fois en processus froids — poids JETÉS, proof-only)."""
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    rr = entrainer_variante(m1, "FULL", 0, LR_VERDICT, tr.a["S"], tr.ap["S"],
                            va.a["S"], va.ap["S"], s_raw_val, m)
    return {
        "tour": 65, "etape": "P0C",
        "config": {"variante": "FULL", "seed": 0, "lr": LR_VERDICT,
                   "init_scale": INIT_SCALE},
        "m_train_S": m,
        "state_sha256": rr.state_sha256,
        "best_epoch": rr.best_epoch, "best_delta_val": rr.best_delta_val,
        "epochs_run": rr.epochs_run, "early_stopped": rr.early_stopped,
        "diverged": rr.diverged,
        "courbes_md5": hashlib.md5(json.dumps(rr.courbes, sort_keys=True)
                                   .encode()).hexdigest(),
        "note": "poids JETÉS (preuve de déterminisme) — le verdict n'en dépend pas",
    }


def etape_etalonnage(root: Path) -> dict:
    """Étalonnage MESURÉ du mur (§4.9) : 2 descentes, 1 passe K=64, 50 eigvals,
    1 split ; extrapolation gelée AVEC le mur (T̂ = 240·t_desc + 256·t_traj +
    956·t_eig + 3·t_split ; mur = max(4·T̂, 600), plafond 3600)."""
    tok, m1 = _instrument()
    t0 = time.perf_counter()
    tr, va = _splits(tok, m1, root)
    t_split = time.perf_counter() - t0
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])

    t0 = time.perf_counter()
    entrainer_variante(m1, "FULL", 0, LR_VERDICT, tr.a["S"], tr.ap["S"],
                       va.a["S"], va.ap["S"], s_raw_val, m)
    entrainer_variante(m1, "MINUS_C", 0, LR_VERDICT, tr.a["S"], tr.ap["S"],
                       va.a["S"], va.ap["S"], s_raw_val, m)
    t_desc = (time.perf_counter() - t0) / 2.0

    fige = trd.make_chrono(0)
    t0 = time.perf_counter()
    derouler(pas_chrono(fige), tr.a["S"])
    t_traj = time.perf_counter() - t0

    mats = matrices_np(fige)
    t0 = time.perf_counter()
    for _ in range(50):
        rho_compagnon(mats)
    t_eig = (time.perf_counter() - t0) / 50.0

    t_hat = 240 * t_desc + 256 * t_traj + 956 * t_eig + 3 * t_split
    mur = min(max(MUR_FACTEUR * t_hat, MUR_PLANCHER_S), MUR_PLAFOND_S)
    return {
        "tour": 65, "etape": "ETALONNAGE",
        "t_desc_s": t_desc, "t_traj_s": t_traj, "t_eig_s": t_eig,
        "t_split_s": t_split,
        "T_hat_s": t_hat, "mur_s": mur,
        "facteur": MUR_FACTEUR, "plancher_s": MUR_PLANCHER_S,
        "plafond_s": MUR_PLAFOND_S,
        "note": "2 descentes pré-gel (poids jetés, étalonnage seulement) — "
                "le premier poids du VERDICT ne bouge qu'à P2, après β",
    }


def etape_p1(root: Path) -> dict:
    """P1 : (a) C4 parité au bit ; (b) parités de chaîne SANS le split brûlé ;
    (c) publications (k*, puissances, règle σ̂_d, portées, figé, ridge,
    effectifs) ; (d) finitude / non-vacuité ; (f) Q11 scan.
    Écrit AUSSI TOUR65_INSTRUMENTS.json (portées Q1-Q11, digest β)."""
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    t64_train = json.loads((root / "TOUR64_TRAIN.json").read_text(encoding="utf-8"))
    t64_sel = json.loads((root / "TOUR64_SELECTION.json").read_text(encoding="utf-8"))

    # (a) C4 — ré-entraînement FULL graines 0-2 ⇒ sha256 fichier ET state au bit.
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    attendus_state = {r["run_id"]: r["state_sha256"] for r in t64_train["runs"]}
    attendus_fichier = {g["seed"]: g["poids_sha256"]
                        for g in t64_sel["selection_par_cartouche"]["S"]["par_graine"]}
    poids_dir = root / POIDS_DIR
    poids_dir.mkdir(exist_ok=True)
    c4_par_graine = []
    for seed in GRAINES_C4:
        rr = entrainer_variante(m1, "FULL", seed, LR_VERDICT, tr.a["S"], tr.ap["S"],
                                va.a["S"], va.ap["S"], s_raw_val, m)
        fichier = poids_dir / f"P1C4_FULL_seed{seed}.bin"
        sha_fichier = rt.sauver_poids(rr.state, fichier)
        rid = f"{RUN_T64_PREFIX}{seed}"
        c4_par_graine.append({
            "seed": seed,
            "state_sha256": rr.state_sha256,
            "state_sha256_attendu": attendus_state[rid],
            "state_identique": bool(rr.state_sha256 == attendus_state[rid]),
            "fichier_sha256": sha_fichier,
            "fichier_sha256_attendu": attendus_fichier[seed],
            "fichier_identique": bool(sha_fichier == attendus_fichier[seed]),
            "best_epoch": rr.best_epoch, "best_delta_val": rr.best_delta_val,
        })
    c4_pass = bool(all(g["state_identique"] and g["fichier_identique"]
                       for g in c4_par_graine))

    # (b) Parités de chaîne (§2.3) : rechargement, m_S, ridge — au bit, 100 %.
    reload_par_graine = []
    for i, seed in enumerate(GRAINES_C4):
        model = modele_t64(root, seed)
        x = rt.appliquer_chrono(model, va.a["S"])
        med = float(np.median(rt.deltas(m1, x, va.ap["S"], s_raw_val)))
        reload_par_graine.append({
            "seed": seed, "delta_med_val": med,
            "attendu": PARITE_RELOAD_DELTA_VAL[i],
            "identique_au_bit": bool(med == PARITE_RELOAD_DELTA_VAL[i]),
        })
    theta = _ridge_train(tr)
    dv_ridge = rt.deltas(m1, rt.predire_ridge(theta, va.a["S"]), va.ap["S"], s_raw_val)
    med_ridge = float(np.median(dv_ridge))
    parites_b = {
        "reload": reload_par_graine,
        "m_S": {"mesure": m, "attendu": PARITE_M_S,
                "identique_au_bit": bool(m == PARITE_M_S)},
        "ridge_lambda1_val": {"mesure": med_ridge, "attendu": PARITE_RIDGE_VAL,
                              "identique_au_bit": bool(med_ridge == PARITE_RIDGE_VAL)},
    }
    p1b_pass = bool(all(g["identique_au_bit"] for g in reload_par_graine)
                    and parites_b["m_S"]["identique_au_bit"]
                    and parites_b["ridge_lambda1_val"]["identique_au_bit"])

    # (c) Publications AVANT gel : k*, puissances, règle σ̂, figé, ridge, effectifs.
    n_g = len(GRAINES_ABLATION)
    k_star = k_star_sign_test(n_g)
    puissances = {
        "contre_75pct": puissance_sign_test(n_g, k_star, Fraction(3, 4)),
        "contre_70pct": puissance_sign_test(n_g, k_star, Fraction(7, 10)),
    }
    fige_pub = []
    for seed in FIGE_SEEDS:
        fige = trd.make_chrono(seed)
        trj = derouler(pas_chrono(fige), tr.a["S"])
        fige_pub.append({"seed": seed, "rho0": rho_compagnon(matrices_np(fige)),
                         "Lambda": float(np.median(trj.lam)),
                         "f_div": trj.f_div,
                         "lam_std": float(trj.lam.std())})
    rho_w = float(np.max(np.abs(np.linalg.eigvals(theta[:-1].astype(np.float64)))))
    trj_ridge = derouler(pas_ridge(theta), tr.a["S"])
    normes_a = np.linalg.norm(tr.a["S"], axis=1)

    # (d) Finitude / non-vacuité (sur le repère figé — le porteur attend β).
    fige0 = trd.make_chrono(0)
    trj0 = derouler(pas_chrono(fige0), tr.a["S"])
    non_vacuite = float(trj0.lam.std())
    finitude = bool(np.isfinite(tr.a["S"]).all() and np.isfinite(tr.ap["S"]).all()
                    and np.isfinite(va.a["S"]).all() and np.isfinite(va.ap["S"]).all()
                    and np.isfinite(trj0.lam).all())

    # (f) Q11 — scan du code neuf + liste close des artefacts T64 lus.
    ici = Path(__file__).resolve()
    test_file = ici.parents[2] / "tests" / "test_chrono_duration.py"
    q11 = scan_q11([ici, test_file])
    q11["artefacts_T64_lus"] = list(ARTEFACTS_T64_LICITES) + [
        f"{POIDS_T64}/{RUN_T64_PREFIX}{s}.bin (S seule)" for s in GRAINES_C4]

    payload = {
        "tour": 65, "etape": "P1",
        "C4_parite_au_bit": {"par_graine": c4_par_graine, "PASS": c4_pass,
                             "regle": "100 % ou FAUX (99 % = FAUX)"},
        "P1b_parites_chaine": {**parites_b, "PASS": p1b_pass},
        "sign_test_n50": {
            "k_star": k_star, "n": n_g,
            "p_a_k_star": float(2 * sum(binom_pmf_fraction(n_g, Fraction(1, 2))[k_star:],
                                        Fraction(0))),
            "puissances": puissances,
        },
        "regle_sigma_d": {"seuil": SIGMA_D_MAX,
                          "MDE_formule": "3.418 * sigma_d / sqrt(50) = 0.4834 * sigma_d",
                          "delta_lambda_min": DELTA_LAMBDA_MIN},
        "fige_T63_12_graines": fige_pub,
        "fige_T63_medianes": {
            "rho0": float(np.median([g["rho0"] for g in fige_pub])),
            "Lambda": float(np.median([g["Lambda"] for g in fige_pub])),
            "f_div": float(np.median([g["f_div"] for g in fige_pub])),
        },
        "ridge_iteree": {"lambda": RIDGE_LAMBDA, "rho_W_exact": rho_w,
                         "Lambda_ridge": float(np.median(trj_ridge.lam)),
                         "f_div_ridge": trj_ridge.f_div,
                         "delta_med_val_K2": med_ridge},
        "effectifs": {"n_TRAIN": tr.n, "n_VALIDATION": va.n,
                      "norme_a_min": float(normes_a.min()),
                      "norme_a_max": float(normes_a.max()),
                      "norme_a_mediane": float(np.median(normes_a))},
        "finitude_non_vacuite": {"finitude_100pct": finitude,
                                 "std_lam_fige_seed0": non_vacuite,
                                 "PASS": bool(finitude and non_vacuite > 0.0)},
        "Q11_scan": q11,
        "PASS": bool(c4_pass and p1b_pass and finitude and non_vacuite > 0.0
                     and q11["PASS"]),
    }
    if not payload["PASS"]:
        raise RuntimeError("P1 FAIL — STOP net, consigner, ne rien réparer en silence")
    # TOUR65_INSTRUMENTS.json — portées Q1-Q11 (T60), publiées AVANT toute
    # mesure de verdict (le gel β les scelle).
    trd.ecrire_artefact(portees_instruments(), root / ART["instruments"])
    return payload


def portees_instruments() -> dict:
    """PORTÉE par instrument (émission §4.6) — quantité qui borne + 2 branches."""
    return {
        "tour": 65, "artefact": "INSTRUMENTS",
        "Q1_rho0": {"role": "PORTEUR C1",
                    "borne": "linéarisation à l'origine, AVEUGLE à B(s²) (dérivée nulle en 0) ; précision eigvals ~1e-12 ; zone morte |rho0-1| < 0.01",
                    "branches": "rho0>1 atteignable (perte pousse le gain 2-pas vers 1) ; rho0<1 atteignable (figé T63 ≈ 0,1)"},
        "Q2_Lambda": {"role": "PORTEUR C1/C2/C3",
                      "borne": "plancher ≈ -0.45 (eps 1e-12, 64 pas, ‖s0‖~2.6) ; plafond ≈ +12.8 (censure Θ=1e6 à t=1) ; ne distingue pas deux contractions sous le plancher",
                      "branches": "Λ>0 (poids appris) et Λ<0 (figé, ridge) mesurés ce tour même"},
        "Q3_fdiv_H50": {"role": "PORTEUR C1",
                        "borne": "détecte une croissance ≥ (1e6/‖s0‖)^(1/64) ≈ 1.216/pas ; non-divergent ≠ stable (dit avant)",
                        "branches": "f_div→1 et f_div→0 (figé) atteignables"},
        "Q4_sign_test": {"role": "PORTEUR C2/C3",
                         "borne": "p<0.01 ⇔ k ≥ k* (recalculé P1) ; puissance 0.84 contre 75 %, 0.56 contre 70 %",
                         "branches": "50 % (sans effet) et ~100 % (effet) atteignables"},
        "Q5_magnitude": {"role": "PORTEUR C2/C3",
                         "borne": "MDE = 0.4834·sigma_d ; NON-MESURE si sigma_d > 0.1034 (règle gelée, scellée avant lecture des signes)",
                         "branches": "deux branches atteignables ; delta=0.05 dérivé (facteur 20 sur 64 pas)"},
        "Q6_C4_parite": {"role": "GATE substrat EXACT",
                         "borne": "toute divergence de spec au dernier bit ; 100 % ou FAUX",
                         "branches": "PASS (déterminisme 15×) et FAIL (re-dérivation infidèle) atteignables"},
        "Q7_C5_optimisation": {"role": "GATE",
                               "borne": "perte ≥ 20 % sur ≥ 167/200 ; finitude 100 % ; std_i(lam_i)>0",
                               "branches": "les deux atteignables (T64 : 24/24 convergés)"},
        "Q8_ridge_iteree": {"role": "RIVAL OBLIGATOIRE (repère C1)",
                            "borne": "borne affine ; rho(W) exact au spectre 19×19",
                            "branches": "contracte (λ=1) / diverge atteignables"},
        "Q9_fige_T63": {"role": "repère (INFO)",
                        "borne": "étalonne l'axe contraction-explosion ; rho0_figé > 0.5 = défaut de chaîne à publier",
                        "branches": "les deux atteignables"},
        "Q10_best_return_step": {"role": "INFO",
                                 "borne": "quasi-corollaire de C1 en branche DIVERGENT (argmax trivial) ; informatif en BORNÉ/BASSIN seulement",
                                 "branches": "les deux atteignables"},
        "Q11_gate_clos": {"role": "GATE matérielle",
                          "borne": "détecte les jetons de route interdits et la lecture des 4 artefacts exclus ; ne détecte pas une fuite par mémoire humaine (interdit écrit)",
                          "branches": "détection / non-détection atteignables"},
        "order_sensibilite": "N-A MOTIVÉE : Φ^K est une puissance d'une seule application ; aucune revendication d'ordre n'est portée (émission §4.6)",
        "agregat": "MÉDIANES partout (T40) ; moyennes en INFO ; cartouche S (POINTE) seule",
    }


def specs_gelees() -> Tuple[dict, dict]:
    """TOUR65_SPEC.json / TOUR65_PREDICTIONS.json (transcription émission §3-§4)."""
    spec = {
        "tour": 65, "artefact": "SPEC",
        "objet": "régime de durée des poids T64 (S is0.1_lr0.01, 3 graines) itérés K=64 + attribution par ablation sous spec T64 identique",
        "base": "TOUR64_SPEC.json repris à l'identique ; SEULE déviation : graines {0..49} (dimensionnement §4.5), appliquée à tous les bras",
        "K_max": K_MAX, "theta_div": THETA_DIV,
        "config_unique": {"init_scale": INIT_SCALE, "lr": LR_VERDICT},
        "variantes": {v: list(ABLATIONS[v]) for v in VARIANTES},
        "ablation": "mise à zéro + gel (requires_grad=False) APRÈS init seedée — A,B,D,L bit-identiques entre bras à graine fixée",
        "n_descentes_verdict": N_DESCENTES_VERDICT,
        "info_robustesse": {"lr": LR_INFO, "graines": list(GRAINES_INFO_LR)},
        "perte": "identique T64 (perte_alpha_omega, m_S au bit)",
        "grandeurs": "rho0 (compagnon 38x38), Lambda (médiane 900 lam_i), f_div, H50, d(g) = Lambda_FULL(g) - Lambda_V(g)",
        "seuils": {"f_div_divergent": SEUIL_F_DIV, "f_div_borne": SEUIL_F_BORNE,
                   "facteur_borne": FACTEUR_BORNE, "zone_neutre_Lambda": ZONE_NEUTRE_LAMBDA,
                   "zone_morte_rho": ZONE_MORTE_RHO, "delta_lambda_min": DELTA_LAMBDA_MIN,
                   "p": SEUIL_P, "sigma_d_max": SIGMA_D_MAX,
                   "quorum_descente": QUORUM_DESCENTE, "repli_div": SEUIL_REPLI_DIV,
                   "point_fixe": SEUIL_POINT_FIXE},
        "rivaux": "ridge λ=1 itérée (rho(W) exact) ; premier ordre (=MINUS_C) ; figé T63 12 graines",
        "precision": "float64, CPU, torch.manual_seed, use_deterministic_algorithms(True)",
    }
    predictions = {
        "tour": 65, "artefact": "PREDICTIONS",
        "H65_A": "rho0 > 1 (3/3), Lambda_FULL > 0, f_div >= 0.90 — transport à deux coups, pas une durée",
        "H65_B": "couper B(s²) et/ou -C(s_prev) déplace Lambda de >= 0.05 en médiane (50 graines appariées), sign-test p < 0.01",
        "cotes": {"P_A_rho0": "65/35", "P_B_Lambda": "70/30",
                  "P_C_moins_B_reduit": "60/40", "P_D_moins_C_augmente": "55/45",
                  "H65_globale": "75/25"},
        "refutation": "(A) tombe si rho0 < 1 ET f_div <= 0.10 ; (B) tombe si |médiane d| < 0.05 OU p >= 0.01 pour les DEUX ablations",
        "directions_non_porteuses": "P-C/P-D sont des paris déclarés ; le porteur C2/C3 est direction-agnostique",
        "conjoncts": {
            "C1": "PORTEUR — régime tranché (DIVERGENT/BORNÉ/BASSIN) + rho0 + Lambda + contrastes figé/ridge",
            "C2": "PORTEUR — ablation -C : |médiane_g d_C| >= 0.05 ET sign-test p < 0.01",
            "C3": "PORTEUR — ablation B(s²) : idem d_B",
            "C4": "GATE — parité au bit des 3 FULL (chute => C2/C3 NON-MESURE, C1 intact)",
            "C5": "GATE — optimisation/finitude (chute => C2/C3 NON-MESURE, C1 intact)",
            "C6": "INFO — ablation conjointe -C-B", "C7": "INFO — chirurgicale",
            "C8": "INFO — best_return_step, r*, bounded, delta K=2 val, rho0 12 poids S, lr=1e-3",
        },
    }
    return spec, predictions


def etape_gel(root: Path) -> dict:
    """β : écrit SPEC/PREDICTIONS puis scelle le jeton (19 chemins, §4.11)
    AVANT la première trajectoire de verdict et avant tout poids de verdict."""
    exiger(root, "p1", "instruments", "etalonnage", "p0b")
    spec, predictions = specs_gelees()
    trd.ecrire_artefact(spec, root / ART["spec"])
    trd.ecrire_artefact(predictions, root / ART["predictions"])
    token = freeze_token([root / p for p in FREEZE_PATHS])
    write_token(token, root / ART["freeze"])
    return {"tour": 65, "etape": "GEL", "digest": token.digest,
            "n_chemins": len(token.artifacts),
            "mtimes_ns": {a.path: a.mtime_ns for a in token.artifacts}}


def etape_regime(root: Path) -> dict:
    """P2-regime — C1 : rho0, Lambda, f_div, H50 sur les 3 poids T64 + les DEUX
    contrastes obligatoires (figé T63, ridge itérée). PREMIÈRE mesure de verdict."""
    exiger(root, "freeze", "etalonnage")
    verifier_gel(root)
    t0 = time.perf_counter()
    tok, m1 = _instrument()
    tr, _ = _splits(tok, m1, root)
    a_s = tr.a["S"]
    par_graine = []
    for seed in GRAINES_C4:
        model = modele_t64(root, seed)
        mats = matrices_np(model)
        rho0 = rho_compagnon(mats)
        trj = derouler(pas_chrono(model), a_s)
        st = stats_trajectoires(trj)
        # rho aux points de fonctionnement (sous-échantillon gelé graine 65).
        rng = np.random.default_rng(R_STAR_SEED)
        idx = rng.choice(a_s.shape[0], size=R_STAR_N, replace=False)
        rho_pts = np.asarray([rho_compagnon(mats, a_s[i]) for i in idx])
        par_graine.append({
            "seed": seed, "rho0": rho0,
            "rho0_zone_morte": bool(abs(rho0 - 1.0) < ZONE_MORTE_RHO),
            **st,
            "Lambda_zone_neutre": bool(abs(st["Lambda"]) < ZONE_NEUTRE_LAMBDA),
            "regime": regime_partition(st["f_div"], st["mediane_norme_max"],
                                       st["mediane_norme_0"]),
            "garde_point_fixe": garde_point_fixe(trj),
            "rho_points_fonctionnement": {
                "n": R_STAR_N, "rng": f"default_rng({R_STAR_SEED})",
                "mediane": float(np.median(rho_pts)),
                "min": float(rho_pts.min()), "max": float(rho_pts.max()),
                "fraction_sup_1": float(np.mean(rho_pts > 1.0)),
            },
        })
    regimes = sorted({g["regime"] for g in par_graine})
    regime_3sur3 = regimes[0] if len(regimes) == 1 else "MIXTE"
    # Contrastes obligatoires (repris de P1, re-mesurés ici pour l'artefact C1).
    p1 = json.loads((root / ART["p1"]).read_text(encoding="utf-8"))
    contrastes = {
        "fige_T63": p1["fige_T63_medianes"],
        "fige_T63_12_graines": p1["fige_T63_12_graines"],
        "ridge_iteree": p1["ridge_iteree"],
    }
    payload = {
        "tour": 65, "etape": "REGIME",
        "grandeur": "régime de durée des poids T64 (config S is0.1_lr0.01, 3 graines), 900 points TRAIN comme conditions initiales, K=64",
        "par_graine": par_graine,
        "regime_3sur3": regime_3sur3,
        "rho0_3graines": [g["rho0"] for g in par_graine],
        "Lambda_3graines": [g["Lambda"] for g in par_graine],
        "f_div_3graines": [g["f_div"] for g in par_graine],
        "contrastes_obligatoires": contrastes,
        "duree_s": time.perf_counter() - t0,
    }
    return payload


def _table_ablation(root: Path, m1, tr, va, s_raw_val, m, graines: Sequence[int],
                    lr: float, poids_prefix: str, t_debut: float,
                    mur_actif: bool) -> Tuple[List[dict], Dict[str, dict]]:
    """Descentes appariées 4 variantes × graines + Λ de chaque modèle retenu."""
    runs: List[dict] = []
    courbes: Dict[str, dict] = {}
    poids_dir = root / POIDS_DIR
    poids_dir.mkdir(exist_ok=True)
    for variante in VARIANTES:
        for seed in graines:
            if mur_actif:
                budget_verifier(root, time.perf_counter() - t_debut)
            rr = entrainer_variante(m1, variante, seed, lr, tr.a["S"], tr.ap["S"],
                                    va.a["S"], va.ap["S"], s_raw_val, m)
            rid = f"{poids_prefix}{variante}_seed{seed}"
            fichier = poids_dir / f"{rid}.bin"
            sha_fichier = rt.sauver_poids(rr.state, fichier)
            model = modele_variante_depuis_etat(rr.state)
            trj = derouler(pas_chrono(model), tr.a["S"])
            st = stats_trajectoires(trj)
            runs.append({
                "run_id": rid, "variante": variante, "seed": seed, "lr": lr,
                "diverged_train": rr.diverged, "epochs_run": rr.epochs_run,
                "best_epoch": rr.best_epoch, "early_stopped": rr.early_stopped,
                "best_delta_val_K2": rr.best_delta_val,
                "loss_init": rr.loss_init, "loss_min": rr.loss_min,
                "descente_rel": rr.descente_rel,
                "descente_ge_20pct": bool(math.isfinite(rr.descente_rel)
                                          and rr.descente_rel >= rt.SEUIL_PERTE_DESCENTE),
                "state_sha256": rr.state_sha256,
                "poids_fichier": fichier.name, "poids_sha256": sha_fichier,
                **{f"traj_{k}": v for k, v in st.items()},
            })
            courbes[rid] = rr.courbes
    return runs, courbes


def etape_ablation(root: Path) -> dict:
    """P2-ablation : les 200 descentes de verdict (4 variantes × 50 graines)
    + Λ de chaque modèle ; gate C5 ; courbes publiées (TOUR65_COURBES.json)."""
    exiger(root, "freeze", "regime", "etalonnage")
    verifier_gel(root)
    verifier_sceau(root, "regime")
    t_debut = time.perf_counter()
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    runs, courbes = _table_ablation(root, m1, tr, va, s_raw_val, m,
                                    GRAINES_ABLATION, LR_VERDICT, "", t_debut, True)
    n_descendus = sum(1 for r in runs if r["descente_ge_20pct"])
    n_diverges = sum(1 for r in runs if r["diverged_train"])
    finitude = bool(all(r["traj_finitude_lam_100pct"] for r in runs))
    non_vacuite = bool(all(r["traj_lam_std"] > 0.0 for r in runs))
    c5_pass = bool(n_descendus >= QUORUM_DESCENTE and finitude and non_vacuite)
    trd.ecrire_artefact({"tour": 65, "artefact": "COURBES",
                         "axes": ["perte_train", "delta_med_val", "r_disp", "dbar"],
                         "par_run": courbes}, root / ART["courbes"])
    return {
        "tour": 65, "etape": "ABLATION",
        "n_runs": len(runs),
        "C5": {"n_descendus_ge_20pct": n_descendus, "quorum": QUORUM_DESCENTE,
               "finitude_lam_100pct": finitude, "non_vacuite_std_pos": non_vacuite,
               "PASS": c5_pass},
        "repli_bounded": {"n_diverges_train": n_diverges,
                          "seuil": SEUIL_REPLI_DIV,
                          "requis": bool(n_diverges >= SEUIL_REPLI_DIV)},
        "runs": runs,
        "duree_s": time.perf_counter() - t_debut,
    }


def _lambdas_par_variante(runs: Sequence[dict], graines: Sequence[int]
                          ) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for v in VARIANTES:
        vals = {r["seed"]: r["traj_Lambda"] for r in runs if r["variante"] == v}
        out[v] = np.asarray([vals[g] for g in graines], dtype=np.float64)
    return out


def etape_sigma(root: Path) -> dict:
    """P2-sigma : σ̂_d et MDE réalisés, ÉCRITS ET SCELLÉS avant le sign-test
    (§4.5, ordre matériel — la règle est mécanique et gelée d'avance)."""
    exiger(root, "freeze", "ablation")
    verifier_gel(root)
    verifier_sceau(root, "ablation")
    abl = json.loads((root / ART["ablation"]).read_text(encoding="utf-8"))
    lams = _lambdas_par_variante(abl["runs"], GRAINES_ABLATION)
    d_c = lams["FULL"] - lams["MINUS_C"]
    d_b = lams["FULL"] - lams["MINUS_B"]
    n = math.sqrt(len(GRAINES_ABLATION))
    payload = {
        "tour": 65, "etape": "SIGMA",
        "regle": "gelée §4.5 : magnitude NON-MESURE si sigma_d > 0.1034 ; ddof=0 (convention T64)",
        "d_C": {"sigma_d": float(d_c.std()),
                "MDE_realise": float(Z_SUM * d_c.std() / n),
                "magnitude_NON_MESURE": bool(d_c.std() > SIGMA_D_MAX)},
        "d_B": {"sigma_d": float(d_b.std()),
                "MDE_realise": float(Z_SUM * d_b.std() / n),
                "magnitude_NON_MESURE": bool(d_b.std() > SIGMA_D_MAX)},
        "seuil_sigma": SIGMA_D_MAX, "delta_lambda_min": DELTA_LAMBDA_MIN,
    }
    return payload


def etape_attribution(root: Path) -> dict:
    """P2-attribution — C2/C3 : médianes appariées et sign-tests exacts."""
    exiger(root, "freeze", "ablation", "sigma")
    verifier_gel(root)
    verifier_sceau(root, "ablation")
    verifier_sceau(root, "sigma")
    abl = json.loads((root / ART["ablation"]).read_text(encoding="utf-8"))
    sig = json.loads((root / ART["sigma"]).read_text(encoding="utf-8"))
    lams = _lambdas_par_variante(abl["runs"], GRAINES_ABLATION)
    k_star = k_star_sign_test(len(GRAINES_ABLATION))
    resultats = {}
    for nom, cible in (("C2_moins_C", "MINUS_C"), ("C3_moins_B", "MINUS_B")):
        d = lams["FULL"] - lams[cible]
        med = float(np.median(d))
        st = rt.sign_test_exact(d)
        cle_sigma = "d_C" if cible == "MINUS_C" else "d_B"
        non_mesure = bool(sig[cle_sigma]["magnitude_NON_MESURE"])
        magnitude_ok = bool(abs(med) >= DELTA_LAMBDA_MIN)
        sign_ok = bool(st["p"] < SEUIL_P)
        resultats[nom] = {
            "variante": cible,
            "d_par_graine": d.tolist(),
            "mediane_d": med, "moyenne_d_INFO": float(d.mean()),
            "sign_test": st, "k_star_n50": k_star,
            "concordants": int(max(st["pos"], st["neg"])),
            "magnitude_ge_seuil": magnitude_ok,
            "magnitude_NON_MESURE_sigma": non_mesure,
            "sign_test_p_lt_001": sign_ok,
            "verdict": bool(magnitude_ok and sign_ok and not non_mesure),
            "verdict_sign_seul_si_sigma_mord": sign_ok,
        }
    return {
        "tour": 65, "etape": "ATTRIBUTION",
        "grandeur": "d(g) = Lambda_FULL(g) − Lambda_V(g), appariée par graine et par init, 50 graines",
        "resultats": resultats,
        "directions_INFO": {
            "P_C_predit": "Lambda_FULL − Lambda_moins_B > 0 (couper le quadratique réduit la croissance)",
            "P_D_predit": "Lambda_FULL − Lambda_moins_C < 0 (couper l'inhibition augmente la croissance)",
            "note": "paris déclarés, PAS le porteur — direction inverse = INFO, jamais un échec",
        },
    }


def etape_info(root: Path) -> dict:
    """P2-info — C6/C7/C8 (INFO, jamais opposables) : −C−B, chirurgicale,
    best_return_step, r*, bounded en lecture, Δ K=2 val, ρ₀ des 12 poids S,
    robustesse lr=1e-3 (40 descentes)."""
    exiger(root, "freeze", "ablation", "attribution")
    verifier_gel(root)
    verifier_sceau(root, "attribution")
    t_debut = time.perf_counter()
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    a_s = tr.a["S"]
    abl = json.loads((root / ART["ablation"]).read_text(encoding="utf-8"))
    lams = _lambdas_par_variante(abl["runs"], GRAINES_ABLATION)

    # C6 — ablation conjointe −C−B (direction et magnitude, INFO).
    d_cb = lams["FULL"] - lams["MINUS_CB"]
    c6 = {"mediane_d": float(np.median(d_cb)),
          "sign_test": rt.sign_test_exact(d_cb),
          "Lambda_MINUS_CB_mediane": float(np.median(lams["MINUS_CB"]))}

    # C7 — chirurgicale : C ou B (ou les deux) mis à zéro SANS ré-entraînement.
    c7 = {}
    for nom, mats in (("moins_C", ("C",)), ("moins_B", ("B",)), ("moins_CB", ("B", "C"))):
        par_seed = []
        for seed in GRAINES_C4:
            model = modele_chirurgical(root, seed, mats)
            trj = derouler(pas_chrono(model), a_s)
            par_seed.append({"seed": seed, "Lambda_surg": float(np.median(trj.lam)),
                             "f_div": trj.f_div})
        c7[nom] = par_seed

    # C8a — best_return_step (argmax_t score_retour(s_t, a'_i), M1 inchangé).
    brs = {}
    for seed in GRAINES_C4:
        model = modele_t64(root, seed)
        trj = derouler(pas_chrono(model), a_s, garder_etats=True)
        argmaxes = []
        for i in range(trj.n):
            fin = trj.t_prime[i]
            if fin < 1:
                argmaxes.append(0)
                continue
            scores = [m1.score_retour(trj.etats[i, t - 1, :], tr.ap["S"][i])
                      for t in range(1, fin + 1)]
            argmaxes.append(int(np.argmax(scores)) + 1)
        arr = np.asarray(argmaxes)
        brs[str(seed)] = {
            "mediane": float(np.median(arr)),
            "fraction_le_2": float(np.mean(arr <= 2)),
            "max": int(arr.max()), "min": int(arr.min()),
            "note": "quasi-corollaire de C1 en branche DIVERGENT (portée Q10)",
        }

    # C8b — r* : sous-échantillon gelé (graine 65), grille λ = 2^-6..2^3.
    rng = np.random.default_rng(R_STAR_SEED)
    idx = rng.choice(a_s.shape[0], size=R_STAR_N, replace=False)
    sous = a_s[idx]
    rstar = {}
    for seed in GRAINES_C4:
        model = modele_t64(root, seed)
        div_par_lam = {}
        r_star_i = np.full(R_STAR_N, np.nan)
        for lam_ech in R_STAR_GRID:
            trj = derouler(pas_chrono(model), lam_ech * sous)
            div_par_lam[str(lam_ech)] = float(trj.f_div)
            nouveaux = trj.diverged & np.isnan(r_star_i)
            r_star_i[nouveaux] = lam_ech
        ok = ~np.isnan(r_star_i)
        rstar[str(seed)] = {
            "f_div_par_lambda": div_par_lam,
            "n_avec_r_star": int(ok.sum()),
            "r_star_mediane": float(np.median(r_star_i[ok])) if ok.any() else None,
        }

    # C8c — bounded=True en lecture (tanh terminal, s'écarte de l'équation pure).
    bounded = {}
    for seed in GRAINES_C4:
        model = modele_t64(root, seed, bounded=True)
        trj = derouler(pas_chrono(model), a_s)
        bounded[str(seed)] = {**stats_trajectoires(trj),
                              "garde_point_fixe": garde_point_fixe(trj)}

    # C8d — Δ K=2 des variantes sur VALIDATION (best_delta_val des 200 runs).
    delta_k2 = {}
    for v in VARIANTES:
        vals = np.asarray([r["best_delta_val_K2"] for r in abl["runs"]
                           if r["variante"] == v], dtype=np.float64)
        delta_k2[v] = {"mediane": float(np.median(vals)),
                       "min": float(vals.min()), "max": float(vals.max())}

    # C8e — ρ₀ des 12 poids S du T64 (4 configs × 3 graines ; S SEULE, §2.4).
    t64_train = json.loads((root / "TOUR64_TRAIN.json").read_text(encoding="utf-8"))
    rho0_12 = []
    for r in t64_train["runs"]:
        if not r["run_id"].startswith("S_"):
            continue
        state = rt.charger_poids(root / POIDS_T64 / r["poids_fichier"])
        torch.manual_seed(0)
        model = ChronoSpiraton(state_size=rt.STATE_SIZE, init_scale=INIT_SCALE,
                               bounded=False, c_outside=False).double()
        model.load_state_dict(state)
        rho0_12.append({"run_id": r["run_id"],
                        "rho0": rho_compagnon(matrices_np(model))})

    # C8f — robustesse lr=1e-3 (10 graines, 4 variantes, 40 descentes, INFO).
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    runs_lr, _ = _table_ablation(root, m1, tr, va, s_raw_val, m,
                                 GRAINES_INFO_LR, LR_INFO, "INFO_lr1e-3_",
                                 t_debut, False)
    lams_lr = {v: np.asarray([r["traj_Lambda"] for r in runs_lr
                              if r["variante"] == v]) for v in VARIANTES}
    robustesse = {
        "lr": LR_INFO, "n_graines": len(GRAINES_INFO_LR),
        "Lambda_medianes": {v: float(np.median(lams_lr[v])) for v in VARIANTES},
        "d_C_mediane": float(np.median(lams_lr["FULL"] - lams_lr["MINUS_C"])),
        "d_B_mediane": float(np.median(lams_lr["FULL"] - lams_lr["MINUS_B"])),
        "runs": runs_lr,
    }
    return {
        "tour": 65, "etape": "INFO",
        "etiquette": "INFO / hors-verdict — aucune de ces mesures n'altère un conjonct",
        "C6_ablation_conjointe": c6,
        "C7_chirurgicale": c7,
        "C8_best_return_step": brs,
        "C8_r_star": {"rng": f"default_rng({R_STAR_SEED})", "grille": list(R_STAR_GRID),
                      "par_graine": rstar},
        "C8_bounded_lecture": bounded,
        "C8_delta_K2_validation": delta_k2,
        "C8_rho0_12_poids_S": rho0_12,
        "C8_robustesse_lr": robustesse,
        "duree_s": time.perf_counter() - t_debut,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("etape", choices=[
        "etalonnage", "p0b", "p0c", "p1", "gel", "regime", "ablation",
        "sigma", "attribution", "info"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    torch.use_deterministic_algorithms(True)
    fn: Dict[str, Callable[[Path], dict]] = {
        "etalonnage": etape_etalonnage, "p0b": etape_p0b, "p0c": etape_p0c,
        "p1": etape_p1, "gel": etape_gel, "regime": etape_regime,
        "ablation": etape_ablation, "sigma": etape_sigma,
        "attribution": etape_attribution, "info": etape_info,
    }
    t0 = time.perf_counter()
    payload = fn[args.etape](root)
    dt = time.perf_counter() - t0
    out = Path(args.out) if args.out else root / ART[args.etape]
    sha = trd.ecrire_artefact(payload, out)
    print(f"{args.etape}: {out} sha256={sha} duree={dt:.3f}s")
    if args.etape in ("regime", "ablation", "sigma", "attribution"):
        print(f"sceau {args.etape}: {sceller(root, args.etape)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée mesure
    raise SystemExit(main())
