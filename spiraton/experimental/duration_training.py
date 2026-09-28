"""duration_training.py — T67 : chantier 6, la durée DANS la perte.

Le T65 a mesuré que la dynamique apprise à K=2 (poids T64) explose à K=64
(DIVERGENT, Λ_FULL médian +1,0845). Ce module déploie l'hypothèse H67
(TOUR67_EMISSION.md) : « la durée s'apprend si elle entre dans la perte — et
elle se paie ». Quatre bras de perte, liste close (§4.4) :

* **A0** (parité)  : ``ℓ(2)`` — EXACTEMENT la perte T64 (gate C6, graines 0-2) ;
* **A1** (MULTI-K) : ``L = (1/3)·[ℓ(2)+ℓ(4)+ℓ(8)]`` — la durée par la tâche ;
* **A2** (STAB)    : ``L = ℓ(2) + λ_stab·mean relu((1/8)·log(‖s_8‖/‖a‖))²`` ;
* **WD**           : ``ℓ(2)`` + ``weight_decay=1e-2`` (rival anti-shrinkage).

avec ``ℓ(K) = −mean_i s_i^{(K)} + λ_copy·mean_i relu(m − d_i^{(K)})²`` où
``s^{(K)} = cos−l2`` (parité M1) et ``d^{(K)} = ‖Φ(a,K)−a‖/(‖a‖+eps)``.

Verdict de durée à K=64 (hors horizon entraîné) : partition CLOSE en cinq
classes — DIVERGENT / BASSIN / EXTINCTION / VERROUILLÉ / BORNÉ-VIVANT — via
``Λ``, ``f_div`` (instruments T65 réutilisés SANS modification,
``chrono_duration.py``), la garde d'extinction (l.717) et l'instrument NEUF
``R_disp^{64}`` (l.341 : « le verrouillage est une récursion sans diversité »
— la dispersion des états à l'horizon rapportée à celle des entrées).

Transport jamais-vu : rôles d'``aba_v2`` GELÉS par index de cycle parsé
(§2.2) — VAL_V2 = cycles 1-1024 (gates/portées/sélecteur g*), RESERVE_V2 =
1025-2048 JAMAIS MATÉRIALISÉE (gate Q12 : parse structurel et md5 d'octets
seulement, exigés par la découpe ; AUCUN agrégat, AUCUN score), TEST_V2 =
2049-4096 (UNE ouverture, en dernier, après scellement de la durée).

Rivaux : identité, poids T64 rechargés (repère apparié), ridge λ=1 re-fittée
(transport ET durée itérée), constante c̄, RESCALE-γ (10 γ × 3 graines,
oracle du split de verdict DONNÉ AU RIVAL), bras WD ; figé T63 cité.

CLI à étapes MATÉRIELLES (chaque étape post-gel vérifie le jeton β
``TOUR67_FREEZE.json`` et le sceau de l'étape précédente) : ``etalonnage`` /
``p0b`` / ``p0c`` / ``p1`` / ``gel`` / ``train`` / ``duree`` / ``sigma`` /
``contraste`` / ``transport`` / ``info``. La durée est SCELLÉE avant
l'ouverture du corpus de verdict (ordre matériel §5).

``chrono.py``, ``return_training.py``, ``chrono_duration.py`` sont importés
en LECTURE, jamais modifiés. Déterministe : seeds fixés, float64, CPU,
``torch.use_deterministic_algorithms(True)`` dans le runner. Les verdicts
vivent dans TOUR67_DEPLOIEMENT.md, pas ici.
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

from ..data import aba, aba_forms
from ..diagnostics import textual_return_dynamics as trd
from ..diagnostics.freeze_token import freeze_token, read_token, verify_freeze, write_token
from . import chrono_duration as cd
from . import return_training as rt
from .chrono import ChronoSpiraton

# --- Constantes gelées (TOUR67_EMISSION.md §4) --------------------------------

EPS = rt.EPS                       #: 1e-12 — le même objet que M1/T64/T65.
K_EVAL = 2                         #: K de transport, gelé pour TOUS les modèles.
K_MAX = 64                         #: horizon de durée (§4.1).
K_PEN = 8                          #: horizon de la pénalité A2.
LAMBDA_STAB = 1.0                  #: gelé sans balayage (§4.4 écart 7).
WEIGHT_DECAY_WD = 1e-2             #: bras WD (écart 6).
INIT_SCALE = 0.1                   #: config unique T64 (aucune grille).
LR = 0.01

BRAS_VERDICT: Tuple[str, ...] = ("A1", "A2", "WD")
BRAS_TOUS: Tuple[str, ...] = ("A0", "A1", "A2", "WD")
GRAINES_50: Tuple[int, ...] = tuple(range(50))
GRAINES_A0: Tuple[int, ...] = (0, 1, 2)
K_MULTI: Tuple[int, ...] = (2, 4, 8)          #: 𝒦 de A1, close en extension.
WD_PAR_BRAS: Dict[str, float] = {"A0": 0.0, "A1": 0.0, "A2": 0.0,
                                 "WD": WEIGHT_DECAY_WD}

#: RESCALE-γ (anti-shrinkage 1) : grille CLOSE en extension (§4.3).
GAMMAS: Tuple[float, ...] = (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

#: Seuils de recevabilité (§4.8) — aucune prose ne lie, les chiffres lient.
C1_QUORUM = 40                     #: BORNÉ-VIVANT ≥ 40/50.
C2_D_MIN = 0.25                    #: médiane_g d ≥ 0,25 (δ_Λ, dérivé §4.5).
C2_SIGMA_D_MAX = 0.5172            #: σ̂_d > 0,5172 ⇒ magnitude NON-MESURE.
C3_E_MIN = -0.05                   #: médiane E ≥ −0,05 (non-infériorité).
C7_SIGMA_E_MAX = 0.662             #: σ̂_E > 0,662 ⇒ transport NON-MESURE.
SEUIL_P = 0.01
C8_QUORUM = 43                     #: descente ≥ 20 % sur ≥ 43/50 par bras.
SEUIL_REPLI_DIVERGES = 25          #: ≥ 25/50 DIVERGED ⇒ survivants + effectif.
Z_SUM = 3.418

#: Partition close (§4.1) — seuils identiques T65 + gardes neuves.
SEUIL_F_DIV = cd.SEUIL_F_DIV                 #: 0,90.
SEUIL_F_BORNE = cd.SEUIL_F_BORNE             #: 0,10.
FACTEUR_BORNE = cd.FACTEUR_BORNE             #: 10×.
FACTEUR_EXTINCTION = 1e-3                    #: I-6 (l.717).
R_DISP64_MIN = 0.20                          #: I-4 (l.341) — VERROUILLÉ si <.
R_DISP64_FRAC_FINIE_MIN = 0.90               #: sinon N-A publié.
SEUIL_POINT_FIXE = cd.SEUIL_POINT_FIXE       #: 1e-6 — INFO, jamais seul (§1).
CLASSES: Tuple[str, ...] = ("DIVERGENT", "BASSIN", "EXTINCTION",
                            "VERROUILLE", "BORNE_VIVANT", "NA_DISP")

#: Gardes de transport (I-8) et de réception (I-9) — paires gelées rng(67).
PAIRES_SEED = 67
N_PAIRES = 200
R_DISP_K2_MIN = rt.R_DISP_MIN                #: 0,20.
DBAR_MIN = rt.DBAR_MIN                       #: 0,05.
R_IN_FACTEUR = 0.50                          #: R_in(VAL_V2) ≥ 0,50·R_in(TRAIN).

#: Contre-épreuve de dépendance macro-récursive (§2.2 piège 4).
STRIDE = 4
STRIDE_OFFSET = 0

#: Corpus jamais-vu — découpe par INDEX DE CYCLE PARSÉ, 1-indexé, inclus.
ABA_V2 = "aba_v2_lab_pack/dataset_aba_v2.txt"
ABA_V2_MD5 = "d7ab47c8a974f06b017ac2d6432f30e0"
TRANCHES_V2: Dict[str, Tuple[int, int]] = {
    "VAL_V2": (1, 1024),
    "TEST_V2": (2049, 4096),
}
#: RESERVE_V2 : bornes publiées pour la découpe/md5 SEULEMENT — toute
#: matérialisation en données (agrégats, scores) est un ValueError (gate Q12).
RESERVE_V2_BORNES: Tuple[int, int] = (1025, 2048)

#: Parités de chaîne au bit (§4.8 i-iv, valeurs T64/T65 gelées).
PARITE_M_S = cd.PARITE_M_S
PARITE_RIDGE_VAL = cd.PARITE_RIDGE_VAL
PARITE_RELOAD_DELTA_VAL = cd.PARITE_RELOAD_DELTA_VAL
RIDGE_LAMBDA = 1.0

#: Mur de budget (§4.9) : mur = max(3·T̂, 1200 s), plafond dur 5400 s.
MUR_PLANCHER_S = 1200.0
MUR_PLAFOND_S = 5400.0
MUR_FACTEUR = 3.0

#: Bande de TENUE dérivée (§3.1), publiée d'avance : Λ ∈ ]ln(1e-3)/64 ; ln(10)/64[.
BANDE_TENUE = (math.log(1e-3) / 64.0, math.log(10.0) / 64.0)

#: Artefacts du tour (racine spiraton-enhanced, hors git).
ART: Dict[str, str] = {
    "etalonnage": "TOUR67_ETALONNAGE.json",
    "p0b": "TOUR67_P0B.json",
    "p0c": "TOUR67_P0C.json",
    "p1": "TOUR67_P1.json",
    "splits_v2": "TOUR67_SPLITS_V2.json",
    "instruments": "TOUR67_INSTRUMENTS.json",
    "spec": "TOUR67_SPEC.json",
    "predictions": "TOUR67_PREDICTIONS.json",
    "gel": "TOUR67_GEL.json",
    "freeze": "TOUR67_FREEZE.json",
    "train": "TOUR67_TRAIN.json",
    "courbes": "TOUR67_COURBES.json",
    "duree": "TOUR67_DUREE.json",
    "sigma": "TOUR67_SIGMA.json",
    "contraste": "TOUR67_CONTRASTE.json",
    "transport": "TOUR67_TRANSPORT.json",
    "info": "TOUR67_INFO67.json",
}
POIDS_DIR = "TOUR67_POIDS"
POIDS_T64 = "TOUR64_POIDS"
RUN_T64_PREFIX = "S_is0.1_lr0.01_seed"

#: Les 22 chemins du digest β (§4.11, liste close), relatifs à la racine.
FREEZE_PATHS: Tuple[str, ...] = (
    "TOUR67_EMISSION.md",
    "TOUR67_SPEC.json",
    "TOUR67_PREDICTIONS.json",
    "TOUR67_INSTRUMENTS.json",
    "TOUR67_P1.json",
    "TOUR67_SPLITS_V2.json",
    "TOUR64_SPEC.json",
    "TOUR64_SELECTION.json",
    "TOUR64_TRAIN.json",
    "TOUR64_POIDS/S_is0.1_lr0.01_seed0.bin",
    "TOUR64_POIDS/S_is0.1_lr0.01_seed1.bin",
    "TOUR64_POIDS/S_is0.1_lr0.01_seed2.bin",
    "TOUR65_ABLATION.json",
    "TOUR65_ATTRIBUTION.json",
    "dataset_aba.txt",
    "aba_v2_lab_pack/dataset_aba_v2.txt",
    "spiraton/spiraton/experimental/chrono.py",
    "spiraton/spiraton/experimental/return_training.py",
    "spiraton/spiraton/experimental/chrono_duration.py",
    "spiraton/spiraton/experimental/duration_training.py",
    "spiraton/spiraton/data/tokenizer_bridge.py",
    "Tokenizer/bin/libspiratontokenizer.so",
)

#: Artefacts T64/T65 lisibles par ce tour (liste CLOSE, §2.3).
ARTEFACTS_ANTERIEURS_LICITES: Tuple[str, ...] = (
    "TOUR64_SPEC.json", "TOUR64_SELECTION.json", "TOUR64_TRAIN.json",
    "TOUR64_P1.json", "TOUR64_SPLITS.json", "TOUR64_ETALONNAGE.json",
    "TOUR64_BASELINES.json", "TOUR64_COURBES.json",
    "TOUR65_ABLATION.json", "TOUR65_ATTRIBUTION.json", "TOUR65_REGIME.json",
    "TOUR65_SIGMA.json", "TOUR65_INFO.json", "TOUR65_P1.json",
    "TOUR65_SPEC.json",
)


# --- Pertes des 4 bras (§4.4, liste close) ------------------------------------

def ell_k(model: ChronoSpiraton, a: torch.Tensor, ap: torch.Tensor,
          m: float, k: int) -> torch.Tensor:
    """``ℓ(K)`` — miroir terme à terme de ``rt.perte_alpha_omega`` à K libre.

    Pour ``k=2`` la séquence d'opérations est IDENTIQUE à
    ``return_training.perte_alpha_omega`` (gate C6 / substrat exact §4.8-vii :
    valeur ET gradients bit-identiques, prouvé par test).
    """
    x = model(a, steps=k)
    nx = torch.linalg.vector_norm(x, dim=-1)
    nap = torch.linalg.vector_norm(ap, dim=-1)
    cos = (x * ap).sum(dim=-1) / (nx * nap + EPS)
    l2 = torch.linalg.vector_norm(ap - x, dim=-1) / (nx + EPS)
    s = cos - l2
    d = torch.linalg.vector_norm(x - a, dim=-1) / (
        torch.linalg.vector_norm(a, dim=-1) + EPS)
    return -s.mean() + rt.LAMBDA_COPY * torch.relu(m - d).pow(2).mean()


def perte_bras(bras: str, model: ChronoSpiraton, a: torch.Tensor,
               ap: torch.Tensor, m: float) -> torch.Tensor:
    """La perte GELÉE de chaque bras (§4.4). A0 ≡ WD ≡ ℓ(2) (WD agit dans
    l'optimiseur, pas dans la perte). A1 somme 3 horizons ; A2 ajoute la
    pénalité de croissance à K_pen=8 (relu : côté positif seulement)."""
    if bras in ("A0", "WD"):
        return ell_k(model, a, ap, m, K_EVAL)
    if bras == "A1":
        return (ell_k(model, a, ap, m, 2) + ell_k(model, a, ap, m, 4)
                + ell_k(model, a, ap, m, 8)) / 3.0
    if bras == "A2":
        base = ell_k(model, a, ap, m, K_EVAL)
        x8 = model(a, steps=K_PEN)
        g = torch.log((torch.linalg.vector_norm(x8, dim=-1) + EPS)
                      / (torch.linalg.vector_norm(a, dim=-1) + EPS)) / K_PEN
        return base + LAMBDA_STAB * torch.relu(g).pow(2).mean()
    raise ValueError(f"bras inconnu : {bras}")


# --- Boucle d'entraînement RE-DÉRIVÉE de train_run (T64) — gate C6 ------------

def entrainer_bras(m1, bras: str, seed: int,
                   a_tr: np.ndarray, ap_tr: np.ndarray,
                   a_val: np.ndarray, ap_val: np.ndarray,
                   s_raw_val: np.ndarray, m: float,
                   *, epochs_max: int = rt.EPOCHS_MAX) -> "rt.RunResult":
    """UN run sous spec T64 EXACTE (TOUR64_SPEC.json), perte du bras (§4.4).

    Miroir terme à terme de ``return_training.train_run`` (importé en lecture,
    NON modifié) : mêmes init seedée, générateur de mélange 1000+seed, Adam
    (0.9, 0.999) eps 1e-8 lr 0.01, clip 1.0, batch 64, early-stop patience 20
    sur ``médiane Δ`` VALIDATION à K=2 (transport-seul : AUCUN bras n'est
    arrêté sur un critère de durée — écrit contre l'hypothèse, §4.4).
    Seuls écarts (énumérés §4.4) : la perte (A1/A2), le ``weight_decay`` de
    l'optimiseur (WD). Pour ``bras='A0'`` la trajectoire d'optimisation est
    BIT-IDENTIQUE à ``train_run`` — gate C6, prouvée au sha256.
    """
    if bras not in BRAS_TOUS:
        raise ValueError(f"bras inconnu : {bras}")
    torch.manual_seed(seed)
    model = ChronoSpiraton(state_size=rt.STATE_SIZE, init_scale=INIT_SCALE,
                           bounded=False, c_outside=False).double()
    gen = torch.Generator()
    gen.manual_seed(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=rt.ADAM_BETAS,
                           eps=rt.ADAM_EPS, weight_decay=WD_PAR_BRAS[bras])
    ta = torch.from_numpy(np.ascontiguousarray(a_tr))
    tap = torch.from_numpy(np.ascontiguousarray(ap_tr))
    n = ta.shape[0]
    pairs = rt.paires_dispersion(a_val.shape[0])

    with torch.no_grad():
        loss_init = float(perte_bras(bras, model, ta, tap, m).item())

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
            loss = perte_bras(bras, model, ta[idx], tap[idx], m)
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
            lt = float(perte_bras(bras, model, ta, tap, m).item())
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
        cartouche="S", init_scale=INIT_SCALE, lr=LR, seed=seed,
        diverged=diverged, epochs_run=epochs_run, best_epoch=best_epoch,
        best_delta_val=best_val, loss_init=loss_init, loss_min=loss_min,
        descente_rel=descente, early_stopped=early, courbes=courbes,
        state=best_state, state_sha256=rt.hash_etat(best_state),
    )


# --- Instruments neufs : R_disp^64 (l.341), extinction (l.717), partition -----

def paires_67(n: int) -> List[Tuple[int, int]]:
    """200 paires (i, j), i≠j, gelées sous ``default_rng(67)`` (I-4/I-8/I-9)."""
    rng = np.random.default_rng(PAIRES_SEED)
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < N_PAIRES:
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.append((int(i), int(j)))
    return pairs


def r_disp_64(trj: "cd.Trajectoires", a: np.ndarray,
              pairs: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    """I-4 — instrument NEUF : diversité des états à l'horizon (l.341).

    ``R_disp^64 = médiane_(i,j) ‖s64_i − s64_j‖ / médiane_(i,j) ‖a_i − a_j‖``
    sur les paires dont les DEUX trajectoires sont finies à t=64 (num ET den
    sur le MÊME sous-ensemble — interprétation déclarée). Si < 90 % des
    paires sont exploitables ⇒ N-A publié.
    """
    if trj.etats_64 is None:
        return {"mesurable": False, "motif": "etats_64 absents"}
    fini = np.isfinite(trj.etats_64).all(axis=1)
    ii = np.asarray([p[0] for p in pairs])
    jj = np.asarray([p[1] for p in pairs])
    ok = fini[ii] & fini[jj]
    frac = float(np.mean(ok))
    if frac < R_DISP64_FRAC_FINIE_MIN:
        return {"mesurable": False, "fraction_paires_finies": frac,
                "seuil_fraction": R_DISP64_FRAC_FINIE_MIN}
    num = float(np.median(np.linalg.norm(
        trj.etats_64[ii[ok]] - trj.etats_64[jj[ok]], axis=1)))
    den = float(np.median(np.linalg.norm(a[ii[ok]] - a[jj[ok]], axis=1)))
    val = num / den if den > 0 else float("nan")
    return {"mesurable": True, "R_disp_64": val,
            "fraction_paires_finies": frac, "num": num, "den": den}


def extinction_mesure(trj: "cd.Trajectoires") -> Dict[str, object]:
    """I-6 (l.717) : ``médiane_i ‖s_64‖ < 1e-3 · médiane_i ‖s_0‖`` — la
    médiane de ‖s_64‖ est prise sur les lignes finies à t=64 (déclaré)."""
    if trj.etats_64 is None:
        return {"mesurable": False}
    fini = np.isfinite(trj.etats_64).all(axis=1)
    med0 = float(np.median(trj.norme_0))
    if not fini.any():
        return {"mesurable": False, "n_finis": 0, "mediane_norme_0": med0}
    med64 = float(np.median(np.linalg.norm(trj.etats_64[fini], axis=1)))
    return {"mesurable": True, "n_finis": int(fini.sum()),
            "mediane_norme_s64": med64, "mediane_norme_0": med0,
            "seuil": FACTEUR_EXTINCTION * med0,
            "mord": bool(med64 < FACTEUR_EXTINCTION * med0)}


def classer_regime(trj: "cd.Trajectoires", a: np.ndarray,
                   pairs: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    """Partition CLOSE, ordre d'évaluation GELÉ (§4.1) :

    1. ``f_div ≥ 0,90`` ⇒ DIVERGENT ; 2. ``f_div > 0,10`` ou
    ``médiane max_t‖s_t‖ > 10·médiane‖s_0‖`` ⇒ BASSIN ; 3. extinction ⇒
    EXTINCTION ; 4. ``R_disp^64 < 0,20`` ⇒ VERROUILLÉ ; 5. BORNÉ-VIVANT.
    Si R_disp^64 est N-A au stade 4 (cas limite) : classe ``NA_DISP``,
    comptée NON BORNÉ-VIVANT (lecture défavorable à l'hypothèse, déclarée).
    """
    st = cd.stats_trajectoires(trj)
    ext = extinction_mesure(trj)
    rd = r_disp_64(trj, a, pairs)
    pf = cd.garde_point_fixe(trj)
    if st["f_div"] >= SEUIL_F_DIV:
        classe = "DIVERGENT"
    elif (st["f_div"] > SEUIL_F_BORNE
          or st["mediane_norme_max"] > FACTEUR_BORNE * st["mediane_norme_0"]):
        classe = "BASSIN"
    elif ext.get("mord", False):
        classe = "EXTINCTION"
    elif not rd.get("mesurable", False):
        classe = "NA_DISP"
    elif rd["R_disp_64"] < R_DISP64_MIN:
        classe = "VERROUILLE"
    else:
        classe = "BORNE_VIVANT"
    return {"classe": classe, "stats": st, "extinction": ext,
            "R_disp_64": rd, "point_fixe_INFO": pf}


def r_in(a: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> float:
    """I-9 : ``R_in = médiane_(i,j) ‖a_i − a_j‖ / médiane_i ‖a_i‖``."""
    ii = np.asarray([p[0] for p in pairs])
    jj = np.asarray([p[1] for p in pairs])
    num = float(np.median(np.linalg.norm(a[ii] - a[jj], axis=1)))
    den = float(np.median(np.linalg.norm(a, axis=1)))
    return num / den if den > 0 else float("nan")


def mediane_basse(valeurs: Dict[int, float]) -> int:
    """Sélecteur g* (§4.2) : la graine dont la valeur est la MÉDIANE de la
    liste triée, rang (n−1)//2 0-indexé = « rang 25, convention basse » à
    n=50 — JAMAIS le maximum. Départage déterministe par graine croissante."""
    tri = sorted(valeurs.items(), key=lambda kv: (kv[1], kv[0]))
    return tri[(len(tri) - 1) // 2][0]


# --- Corpus aba_v2 : découpe par index de cycle parsé (§2.2) ------------------

def cycles_aba_v2(root: Path):
    """Parse structurel de TOUT le fichier (nécessaire : la découpe est PAR
    INDEX DE CYCLE PARSÉ). Retourne ``(cycles, numeros_ligne, n_lignes,
    lignes_octets)``. md5 DUR du fichier vérifié avant toute lecture."""
    path = root / ABA_V2
    md5 = trd.md5_fichier(path)
    if md5 != ABA_V2_MD5:
        raise RuntimeError(f"md5 {ABA_V2} = {md5} != gelé {ABA_V2_MD5} — ARRÊT")
    lignes = path.read_bytes().splitlines(keepends=True)
    cycles, numeros = [], []
    for off, lb in enumerate(lignes):
        line = lb.decode("utf-8")
        try:
            c = aba.try_parse_aba_line(line)
        except aba.AbaParseError:
            continue
        if c is not None:
            cycles.append(c)
            numeros.append(off + 1)
    return cycles, numeros, len(lignes), lignes


def decoupe_aba_v2(root: Path) -> dict:
    """Découpe STRUCTURELLE publiée à P1 : md5 par tranche (octets bruts des
    lignes de la tranche), effectifs, composition par opérateur — pour les
    TROIS tranches (le md5 et l'opérateur sont du parse structurel, exigés
    §2.2/P1-c ; AUCUNE grandeur géométrique n'est extraite de RESERVE_V2)."""
    cycles, numeros, n_lignes, lignes = cycles_aba_v2(root)
    bornes = dict(TRANCHES_V2)
    bornes["RESERVE_V2"] = RESERVE_V2_BORNES
    tranches = {}
    for nom in ("VAL_V2", "RESERVE_V2", "TEST_V2"):
        i1, i2 = bornes[nom]
        cyc = cycles[i1 - 1:i2]
        nums = numeros[i1 - 1:i2]
        raw = b"".join(lignes[n - 1] for n in nums)
        ops = {op: 0 for op in aba.OPERATORS}
        for c in cyc:
            ops[c.op] += 1
        tranches[nom] = {
            "cycles": [i1, i2], "n": len(cyc),
            "lignes_physiques": [nums[0], nums[-1]] if nums else [],
            "md5_octets_bruts": hashlib.md5(raw).hexdigest(),
            "composition_operateurs": ops,
        }
    return {"fichier": ABA_V2, "md5": ABA_V2_MD5, "n_lignes": n_lignes,
            "n_cycles_parses": len(cycles), "tranches": tranches}


@dataclass(frozen=True)
class TrancheV2:
    """Une tranche d'aba_v2 MATÉRIALISÉE (cartouche S, tranche MAIN 19d)."""

    nom: str
    cycles_bornes: Tuple[int, int]
    n: int
    md5_octets: str
    numeros: Tuple[int, ...]
    operateurs: Tuple[str, ...]
    a: np.ndarray
    ap: np.ndarray
    exclusions: Tuple[Dict[str, object], ...]


def charger_tranche_v2(tok, m1, root: Path, nom: str) -> TrancheV2:
    """Matérialise VAL_V2 ou TEST_V2 en paires (a, a′) S/MAIN float64.

    GATE Q12 STRUCTURELLE : toute autre tranche (RESERVE_V2 incluse) est un
    ``ValueError`` — la réserve n'est JAMAIS matérialisée ce tour.
    Chaîne T63/T64 inchangée : segments isolés → 33D (heuristic OFF) →
    cartouche S (pointe), tranche MAIN {6..22, 31, 32} (dims 0-5 EXCLUES).
    """
    if nom not in TRANCHES_V2:
        raise ValueError(
            f"tranche {nom!r} non matérialisable (gate Q12 : seules "
            f"{sorted(TRANCHES_V2)} s'ouvrent ce tour)")
    cycles, numeros, _, lignes = cycles_aba_v2(root)
    i1, i2 = TRANCHES_V2[nom]
    cyc = cycles[i1 - 1:i2]
    nums = numeros[i1 - 1:i2]
    raw = b"".join(lignes[n - 1] for n in nums)
    a_list: List[np.ndarray] = []
    ap_list: List[np.ndarray] = []
    numeros_ret: List[int] = []
    ops: List[str] = []
    exclusions: List[Dict[str, object]] = []
    for c, num in zip(cyc, nums):
        forme = m1.forme_cycle(c, aba_forms)
        if forme == "F5":
            exclusions.append({"ligne": num, "motif": "F5"})
            continue
        seg_a = m1.agreger_segment(tok, c.seg_a.text)
        seg_ap = m1.agreger_segment(tok, c.seg_a_prime.text)
        if ("S", "MAIN") not in seg_a.aggs or ("S", "MAIN") not in seg_ap.aggs:
            exclusions.append({"ligne": num, "motif": "agrégat absent"})
            continue
        a_list.append(np.asarray(seg_a.aggs[("S", "MAIN")], dtype=np.float64))
        ap_list.append(np.asarray(seg_ap.aggs[("S", "MAIN")], dtype=np.float64))
        numeros_ret.append(num)
        ops.append(c.op)
    return TrancheV2(
        nom=nom, cycles_bornes=(i1, i2), n=len(a_list),
        md5_octets=hashlib.md5(raw).hexdigest(),
        numeros=tuple(numeros_ret), operateurs=tuple(ops),
        a=np.stack(a_list), ap=np.stack(ap_list),
        exclusions=tuple(exclusions),
    )


# --- Rivaux (§4.3) -------------------------------------------------------------

def modele_t64(root: Path, seed: int) -> ChronoSpiraton:
    """Poids T64 rechargés (JAMAIS réécrits) — le repère central."""
    state = rt.charger_poids(root / POIDS_T64 / f"{RUN_T64_PREFIX}{seed}.bin")
    return rt.modele_depuis_poids(INIT_SCALE, state)


def modele_rescale(root: Path, seed: int, gamma: float) -> ChronoSpiraton:
    """RESCALE-γ : poids T64 avec ``D ← γD, L ← γL`` (grille close §4.3)."""
    model = modele_t64(root, seed)
    with torch.no_grad():
        model.D.weight.mul_(gamma)
        model.L.weight.mul_(gamma)
    return model


def modele_depuis_fichier(root: Path, fichier: str) -> ChronoSpiraton:
    return rt.modele_depuis_poids(
        INIT_SCALE, rt.charger_poids(root / POIDS_DIR / fichier))


# --- Statistique : sign-test stride-4 (contre-épreuve de dépendance) -----------

def contre_epreuve_stride(d: np.ndarray) -> Dict[str, object]:
    """Sign-test exact sur ``d[offset::stride]`` (n=512) — contrôle de la
    macro-récursion ``A′(n)=A(n+1)``. Concordance = même signe de médiane ET
    pas de bascule significative (p<0,01) en direction OPPOSÉE (déclaré)."""
    sous = d[STRIDE_OFFSET::STRIDE]
    st_full = rt.sign_test_exact(d)
    st_sub = rt.sign_test_exact(sous)
    signe_full = float(np.sign(np.median(d)))
    signe_sub = float(np.sign(np.median(sous)))
    inverse_significatif = bool(
        st_sub["p"] < SEUIL_P
        and np.sign(st_sub["pos"] - st_sub["neg"])
        == -np.sign(st_full["pos"] - st_full["neg"])
        and (st_full["pos"] - st_full["neg"]) != 0)
    concordant = bool(signe_full == signe_sub and not inverse_significatif)
    return {"stride": STRIDE, "offset": STRIDE_OFFSET, "n_sous": int(sous.size),
            "mediane_full": float(np.median(d)),
            "mediane_sous": float(np.median(sous)),
            "sign_test_full": st_full, "sign_test_sous": st_sub,
            "bascule_inverse_significative": inverse_significatif,
            "concordant": concordant}


# --- Gardes matérielles (ordre α-β-γ, sceaux) ----------------------------------

def verifier_gel(root: Path) -> None:
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


def budget_verifier(root: Path, t_consomme_etape: float) -> None:
    """Mur §4.9 : cumul P2 (durées scellées des étapes P2 passées + étape en
    cours) > mur ⇒ ARRÊT (I9), aucune dégradation de périmètre en vol."""
    etal = json.loads((root / ART["etalonnage"]).read_text(encoding="utf-8"))
    mur = float(etal["mur_s"])
    cumul = t_consomme_etape
    for cle in ("train", "duree", "sigma", "contraste", "transport"):
        p = root / ART[cle]
        if p.is_file():
            try:
                cumul += float(json.loads(p.read_text(encoding="utf-8"))
                               .get("duree_s", 0.0))
            except Exception:
                pass
    if cumul > mur:
        raise RuntimeError(
            f"MUR DE BUDGET dépassé : {cumul:.1f}s > {mur:.1f}s — ARRÊT (I9)")


# --- Gate Q12 : scan matériel du code neuf -------------------------------------

def scan_q12(fichiers: Sequence[Path]) -> Dict[str, object]:
    """Q12 : aucun jeton de route interdit dans le code neuf. Jetons construits
    par concaténation (le scanner ne se déclenche pas lui-même) : bornes du
    split brûlé de dataset_aba, littéraux du split de verdict T64, noms des 4
    artefacts T64 interdits en lecture."""
    interdits = ["35" + "01", "50" + "01", '"TE' + 'ST"', "'TE" + "ST'",
                 "TOUR64_POR" + "TEUR", "TOUR64_DELTAS" + "_T",
                 "TOUR64_CARTOUCHE" + "_P", "TOUR64_IN" + "FO"]
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


def _splits(tok, m1, root: Path):
    tr = rt.charger_split(tok, m1, root, "TRAIN", rt.SPLITS["TRAIN"])
    va = rt.charger_split(tok, m1, root, "VALIDATION", rt.SPLITS["VALIDATION"])
    return tr, va


def _ridge_train(tr) -> np.ndarray:
    return rt.ajuster_ridge(tr.a["S"], tr.ap["S"], RIDGE_LAMBDA)


def _lambda_full_t65(root: Path) -> Dict[int, float]:
    """``Λ_FULL^{T65}(g)`` recopié de TOUR65_ABLATION.json (liste close §2.3),
    jamais re-mesuré (voie close)."""
    abl = json.loads((root / "TOUR65_ABLATION.json").read_text(encoding="utf-8"))
    out = {r["seed"]: float(r["traj_Lambda"]) for r in abl["runs"]
           if r["variante"] == "FULL"}
    if sorted(out) != list(GRAINES_50):
        raise RuntimeError("Λ_FULL^{T65} : 50 graines attendues — ARRÊT")
    return out


def _classer_modele(model: ChronoSpiraton, a: np.ndarray,
                    pairs: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    trj = cd.derouler(cd.pas_chrono(model), a)
    res = classer_regime(trj, a, pairs)
    res["rho0"] = cd.rho_compagnon(cd.matrices_np(model))
    return res


def _resume_classement(res: Dict[str, object]) -> Dict[str, object]:
    """Cartouche plat publié par modèle (les gros tableaux restent internes)."""
    st = res["stats"]
    rd = res["R_disp_64"]
    return {
        "classe": res["classe"], "rho0": res["rho0"],
        "Lambda": st["Lambda"], "f_div": st["f_div"], "H50": st["H50"],
        "lam_std": st["lam_std"],
        "mediane_norme_0": st["mediane_norme_0"],
        "mediane_norme_max": st["mediane_norme_max"],
        "mediane_norme_derniere": st["mediane_norme_derniere"],
        "finitude_lam_100pct": st["finitude_lam_100pct"],
        "R_disp_64": rd.get("R_disp_64"),
        "R_disp_64_mesurable": rd.get("mesurable"),
        "extinction_mord": res["extinction"].get("mord"),
        "extinction_mediane_norme_s64": res["extinction"].get("mediane_norme_s64"),
        "point_fixe_ratio_INFO": res["point_fixe_INFO"].get("ratio_median"),
        "dans_bande_tenue": bool(BANDE_TENUE[0] < st["Lambda"] < BANDE_TENUE[1]),
    }


# --- Étapes du protocole --------------------------------------------------------

def etape_p0b(root: Path) -> dict:
    """P0-b : substrats EXACTS §4.8 (v)-(vii) + flux de gradient + WD 1 pas.

    Doublé par ``tests/test_duration_training.py`` (quatuor pytest complet) ;
    l'étape écrit la preuve dans un artefact (100 % ou FAUX).
    """
    d = rt.STATE_SIZE
    rng = np.random.default_rng(670)
    pts = rng.standard_normal((16, d))
    # (v) A=B=C=D=0, L=I ⇒ s_t = s_0 ⇒ λ_i = 0 EXACT et Δ_i = 0 EXACT.
    model = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D"):
            getattr(model, nom).weight.zero_()
        model.L.weight.copy_(torch.eye(d, dtype=torch.float64))
    trj = cd.derouler(cd.pas_chrono(model), pts, garder_etats=True)
    ev_lam = bool((trj.lam == 0.0).all())
    x2 = rt.appliquer_chrono(model, pts)
    ev_phi_id = bool(np.array_equal(x2, pts))
    ev = bool(ev_lam and ev_phi_id and trj.f_div == 0.0)
    # (vi) A=I, B=C=0, D=αI, L=0 ⇒ λ_i = log α (1e-12 rel) ; ρ₀ = α exact.
    alpha = 1.5
    model2 = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    with torch.no_grad():
        model2.A.weight.copy_(torch.eye(d, dtype=torch.float64))
        model2.B.weight.zero_()
        model2.C.weight.zero_()
        model2.D.weight.copy_(alpha * torch.eye(d, dtype=torch.float64))
        model2.L.weight.zero_()
    trj2 = cd.derouler(cd.pas_chrono(model2), pts)
    evi_lam = bool(np.max(np.abs(trj2.lam - math.log(alpha)) / math.log(alpha)) <= 1e-12)
    rho0 = cd.rho_compagnon(cd.matrices_np(model2))
    evi = bool(evi_lam and rho0 == alpha)
    # (vii) perte A0 bit-identique à rt.perte_alpha_omega (batch témoin gelé).
    torch.manual_seed(670)
    temoin = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    ta = torch.from_numpy(rng.standard_normal((8, d)))
    tap = torch.from_numpy(rng.standard_normal((8, d)))
    l_a0 = perte_bras("A0", temoin, ta, tap, PARITE_M_S)
    l_t64 = rt.perte_alpha_omega(temoin, ta, tap, PARITE_M_S)
    evii_val = bool(torch.equal(l_a0, l_t64))
    g_a0 = torch.autograd.grad(l_a0, list(temoin.parameters()), retain_graph=False)
    l_t64b = rt.perte_alpha_omega(temoin, ta, tap, PARITE_M_S)
    g_t64 = torch.autograd.grad(l_t64b, list(temoin.parameters()))
    evii_grad = bool(all(torch.equal(x, y) for x, y in zip(g_a0, g_t64)))
    evii = bool(evii_val and evii_grad)
    # Flux de gradient : les 5 matrices sous chaque perte de bras.
    flux = {}
    for bras in BRAS_TOUS:
        torch.manual_seed(671)
        mm = ChronoSpiraton(state_size=d, init_scale=0.1).double()
        loss = perte_bras(bras, mm, ta, tap, PARITE_M_S)
        grads = torch.autograd.grad(loss, list(mm.parameters()))
        flux[bras] = bool(all(g is not None and float(g.abs().sum()) > 0.0
                              for g in grads))
    # WD analytique 1 pas : Adam(wd) ≡ Adam(0) sur perte + (wd/2)·‖θ‖².
    torch.manual_seed(672)
    m_wd = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    torch.manual_seed(672)
    m_l2 = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    o_wd = torch.optim.Adam(m_wd.parameters(), lr=LR, betas=rt.ADAM_BETAS,
                            eps=rt.ADAM_EPS, weight_decay=WEIGHT_DECAY_WD)
    o_l2 = torch.optim.Adam(m_l2.parameters(), lr=LR, betas=rt.ADAM_BETAS,
                            eps=rt.ADAM_EPS, weight_decay=0.0)
    o_wd.zero_grad()
    perte_bras("WD", m_wd, ta, tap, PARITE_M_S).backward()
    o_wd.step()
    o_l2.zero_grad()
    (perte_bras("WD", m_l2, ta, tap, PARITE_M_S)
     + (WEIGHT_DECAY_WD / 2.0)
     * sum(p.pow(2).sum() for p in m_l2.parameters())).backward()
    o_l2.step()
    ecarts = [float((p.detach() - q.detach()).abs().max())
              for p, q in zip(m_wd.parameters(), m_l2.parameters())]
    wd_ok = bool(max(ecarts) <= 1e-15)
    payload = {
        "tour": 67, "etape": "P0B",
        "Ev_identite": {"PASS": ev, "lam_zero_exact": ev_lam,
                        "phi2_identite_bit": ev_phi_id, "f_div": trj.f_div},
        "Evi_geometrique": {"PASS": evi, "alpha": alpha, "rho0": rho0,
                            "rho0_exact": bool(rho0 == alpha)},
        "Evii_perte_A0_bit": {"PASS": evii, "valeur_bit": evii_val,
                              "gradients_bit": evii_grad},
        "flux_gradient_par_bras": flux,
        "WD_analytique_1_pas": {"PASS": wd_ok, "ecart_max": max(ecarts)},
        "PASS": bool(ev and evi and evii and all(flux.values()) and wd_ok),
    }
    if not payload["PASS"]:
        raise RuntimeError(f"P0-b FAIL : {payload}")
    return payload


def etape_p0c(root: Path) -> dict:
    """P0-c : UNE descente A1 complète graine 0 (preuve de déterminisme, à
    exécuter DEUX fois en processus froids — poids JETÉS, proof-only)."""
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    rr = entrainer_bras(m1, "A1", 0, tr.a["S"], tr.ap["S"],
                        va.a["S"], va.ap["S"], s_raw_val, m)
    return {
        "tour": 67, "etape": "P0C",
        "config": {"bras": "A1", "seed": 0, "lr": LR, "init_scale": INIT_SCALE},
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
    """§4.9 : étalonnage MESURÉ (~2 % du périmètre), extrapolation gelée AVEC
    le mur : T̂ = 50·(t_A1+t_A2+t_WD) + 3·t_A0 + 182·t_traj + 31·t_tok100 +
    3·t_split ; mur = max(3·T̂, 1200 s), plafond dur 5400 s. ``t_A0 := t_WD``
    (même perte ℓ(2), seul le wd de l'optimiseur diffère — déclaré)."""
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    temps = {}
    for bras in ("A1", "A2", "WD"):
        t0 = time.perf_counter()
        entrainer_bras(m1, bras, 0, tr.a["S"], tr.ap["S"],
                       va.a["S"], va.ap["S"], s_raw_val, m)
        temps[bras] = time.perf_counter() - t0
    fige = trd.make_chrono(0)
    pairs = paires_67(tr.a["S"].shape[0])
    t0 = time.perf_counter()
    _classer_modele(fige, tr.a["S"], pairs)
    t_traj = time.perf_counter() - t0
    cycles, numeros, _, lignes = cycles_aba_v2(root)
    t0 = time.perf_counter()
    for c in cycles[:100]:
        m1.agreger_segment(tok, c.seg_a.text)
        m1.agreger_segment(tok, c.seg_a_prime.text)
    t_tok100 = time.perf_counter() - t0
    t0 = time.perf_counter()
    charger_tranche_v2(tok, m1, root, "VAL_V2")
    t_split = time.perf_counter() - t0
    t_a0 = temps["WD"]
    t_hat = (50.0 * (temps["A1"] + temps["A2"] + temps["WD"]) + 3.0 * t_a0
             + 182.0 * t_traj + 31.0 * t_tok100 + 3.0 * t_split)
    mur = min(max(MUR_FACTEUR * t_hat, MUR_PLANCHER_S), MUR_PLAFOND_S)
    return {
        "tour": 67, "etape": "ETALONNAGE",
        "t_A1_s": temps["A1"], "t_A2_s": temps["A2"], "t_WD_s": temps["WD"],
        "t_A0_s_declare_egal_t_WD": t_a0,
        "t_traj_s": t_traj, "t_tok100_s": t_tok100, "t_split_s": t_split,
        "T_hat_s": t_hat, "mur_s": mur, "facteur": MUR_FACTEUR,
        "plancher_s": MUR_PLANCHER_S, "plafond_s": MUR_PLAFOND_S,
        "arret_probable_si_T_hat_sup": 1800.0,
        "note": "3 descentes pré-gel (3/153 = 2,0 %, poids jetés) ; "
                "le premier poids du VERDICT ne bouge qu'à P2, après β",
    }


def _sigma_e_val_v2(m1, va2: TrancheV2, modeles_t64, theta) -> Dict[str, object]:
    """Estimateur de ``σ̂_E`` GELÉ (déclaré aux instruments AVANT mesure) :
    max(σ des 3 contrastes appariés inter-graines T64, σ du contraste
    T64−ridge) sur VAL_V2, ddof=0 — lecture défavorable (le max)."""
    scores = {seed: rt.scores_m1(m1, rt.appliquer_chrono(mod, va2.a), va2.ap)
              for seed, mod in modeles_t64.items()}
    s_ridge = rt.scores_m1(m1, rt.predire_ridge(theta, va2.a), va2.ap)
    candidats = {}
    seeds = sorted(scores)
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            dif = scores[seeds[i]] - scores[seeds[j]]
            candidats[f"std_E_T64_{seeds[i]}v{seeds[j]}"] = float(dif.std())
    for seed in seeds:
        candidats[f"std_D_T64s{seed}_vs_ridge"] = float(
            (scores[seed] - s_ridge).std())
    sigma_e = max(candidats.values())
    return {"estimateur": "max des contrastes appariés disponibles à P1 "
                          "(inter-graines T64 et T64−ridge), ddof=0",
            "candidats": candidats, "sigma_E": sigma_e,
            "seuil": C7_SIGMA_E_MAX,
            "NON_MESURE_si_sup": bool(sigma_e > C7_SIGMA_E_MAX)}


def etape_p1(root: Path) -> dict:
    """P1 (§5), dans cet ordre : (a) C6 parité au bit (3 poids A0) ;
    (b) parités de chaîne (rechargement ×3, m_S, ridge — au bit) ;
    (c) découpe aba_v2 (md5 par tranche, composition par opérateur) ;
    (d) C7 sur VAL_V2 (parse, finitude, R_in, R_disp^K2, d̄, σ̂_E) ;
    (e) publications (k*, puissances, Λ_FULL^{T65}, bande TENUE, portées) ;
    (g) Q12 scan. Écrit AUSSI TOUR67_SPLITS_V2.json et TOUR67_INSTRUMENTS.json.
    """
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    t64_train = json.loads((root / "TOUR64_TRAIN.json").read_text(encoding="utf-8"))
    t64_sel = json.loads((root / "TOUR64_SELECTION.json").read_text(encoding="utf-8"))

    # (a) C6 — bras A0 graines {0,1,2} ⇒ sha256 state ET fichier au bit T64.
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    attendus_state = {r["run_id"]: r["state_sha256"] for r in t64_train["runs"]}
    attendus_fichier = {g["seed"]: g["poids_sha256"]
                        for g in t64_sel["selection_par_cartouche"]["S"]["par_graine"]}
    poids_dir = root / POIDS_DIR
    poids_dir.mkdir(exist_ok=True)
    c6_par_graine = []
    for seed in GRAINES_A0:
        rr = entrainer_bras(m1, "A0", seed, tr.a["S"], tr.ap["S"],
                            va.a["S"], va.ap["S"], s_raw_val, m)
        fichier = poids_dir / f"A0_seed{seed}.bin"
        sha_fichier = rt.sauver_poids(rr.state, fichier)
        rid = f"{RUN_T64_PREFIX}{seed}"
        c6_par_graine.append({
            "seed": seed,
            "state_sha256": rr.state_sha256,
            "state_sha256_attendu": attendus_state[rid],
            "state_identique": bool(rr.state_sha256 == attendus_state[rid]),
            "fichier_sha256": sha_fichier,
            "fichier_sha256_attendu": attendus_fichier[seed],
            "fichier_identique": bool(sha_fichier == attendus_fichier[seed]),
            "best_epoch": rr.best_epoch, "best_delta_val": rr.best_delta_val,
        })
    c6_pass = bool(all(g["state_identique"] and g["fichier_identique"]
                       for g in c6_par_graine))

    # (b) Parités de chaîne au bit (§4.8 i-iv).
    reload_par_graine = []
    modeles_t64 = {}
    for i, seed in enumerate(GRAINES_A0):
        model = modele_t64(root, seed)
        modeles_t64[seed] = model
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

    # (c) Découpe aba_v2 : md5 par tranche, effectifs, composition opérateurs.
    decoupe = decoupe_aba_v2(root)
    ops_ok = {}
    for nom, tin in decoupe["tranches"].items():
        ops = tin["composition_operateurs"]
        total = sum(ops.values())
        ops_ok[nom] = {op: (cnt / total if total else 0.0)
                       for op, cnt in ops.items()}
    operateurs_sous_10pct = {
        nom: [op for op, frac in fr.items() if frac < 0.10]
        for nom, fr in ops_ok.items()}

    # (d) C7 sur VAL_V2 (gates armées AVANT toute promesse).
    va2 = charger_tranche_v2(tok, m1, root, "VAL_V2")
    parse_100 = bool(va2.n + len(va2.exclusions)
                     == TRANCHES_V2["VAL_V2"][1] - TRANCHES_V2["VAL_V2"][0] + 1
                     and len(va2.exclusions) == 0)
    finitude_v2 = bool(np.isfinite(va2.a).all() and np.isfinite(va2.ap).all())
    pairs_tr = paires_67(tr.a["S"].shape[0])
    pairs_v2 = paires_67(va2.n)
    r_in_train = r_in(tr.a["S"], pairs_tr)
    r_in_val2 = r_in(va2.a, pairs_v2)
    # Repère T64 : graine de médiane Δ MÉDIANE sur VAL_V2 (parmi {0,1,2}).
    s_raw_v2 = rt.scores_m1(m1, va2.a, va2.ap)
    deltas_t64_v2 = {}
    for seed, model in modeles_t64.items():
        x = rt.appliquer_chrono(model, va2.a)
        deltas_t64_v2[seed] = float(np.median(
            rt.deltas(m1, x, va2.ap, s_raw_v2)))
    seed_repere = mediane_basse(deltas_t64_v2)
    x_rep = rt.appliquer_chrono(modeles_t64[seed_repere], va2.a)
    r_disp_k2, dbar = rt.gardes_collapse(x_rep, va2.a, pairs_v2)
    sigma_e = _sigma_e_val_v2(m1, va2, modeles_t64, theta)
    sigma_delta_info = float(rt.deltas(
        m1, x_rep, va2.ap, s_raw_v2).std())
    # Production gelée §4.7 (garde contournée) : histogramme des ‖a_i‖ et
    # chiffres de dégénérescence — publiés QUEL QUE SOIT le verdict de C7.
    normes_v2 = np.linalg.norm(va2.a, axis=1)
    hist, edges = np.histogram(normes_v2, bins=8)
    ii = np.asarray([p[0] for p in pairs_v2])
    jj = np.asarray([p[1] for p in pairs_v2])
    d_paires = np.linalg.norm(va2.a[ii] - va2.a[jj], axis=1)
    degenerescence = {
        "n_uniques_a": int(np.unique(va2.a, axis=0).shape[0]),
        "n_uniques_ap": int(np.unique(va2.ap, axis=0).shape[0]),
        "n_paires_gelees_dist_zero": int((d_paires == 0.0).sum()),
        "histogramme_normes_a": {"comptes": hist.tolist(),
                                 "bords": [float(e) for e in edges]},
        "mediane_norme_a": float(np.median(normes_v2)),
    }
    c7 = {
        "parse_100pct": parse_100,
        "finitude_100pct": finitude_v2,
        "R_in_TRAIN": r_in_train, "R_in_VAL_V2": r_in_val2,
        "R_in_ratio": r_in_val2 / r_in_train if r_in_train > 0 else float("nan"),
        "R_in_seuil": R_IN_FACTEUR,
        "R_in_PASS": bool(r_in_val2 >= R_IN_FACTEUR * r_in_train),
        "repere_T64": {"delta_med_VAL_V2_par_graine":
                       {str(k): v for k, v in deltas_t64_v2.items()},
                       "graine_repere": seed_repere},
        "R_disp_K2_repere": r_disp_k2, "R_disp_K2_seuil": R_DISP_K2_MIN,
        "R_disp_K2_PASS": bool(r_disp_k2 >= R_DISP_K2_MIN),
        "dbar_repere": dbar, "dbar_seuil": DBAR_MIN,
        "dbar_PASS": bool(dbar >= DBAR_MIN),
        "sigma_E": sigma_e,
        "sigma_E_PASS": bool(not sigma_e["NON_MESURE_si_sup"]),
        "sigma_Delta_repere_INFO": sigma_delta_info,
        "degenerescence_geometrique": degenerescence,
        "PASS": bool(parse_100 and finitude_v2
                     and r_in_val2 >= R_IN_FACTEUR * r_in_train
                     and bool(np.isfinite(r_disp_k2))
                     and r_disp_k2 >= R_DISP_K2_MIN and dbar >= DBAR_MIN
                     and not sigma_e["NON_MESURE_si_sup"]),
        "chute": "C7 FAIL => C3/C4/C5 NON-MESURE, TEST_V2 JAMAIS OUVERT "
                 "(économisé, issue I7) ; C1/C2 intacts — partage de sort "
                 "ÉCRIT (émission §3.3/§4.2)",
    }

    # (e) Publications : k*, puissances, Λ_FULL^{T65}, bande TENUE, ‖s_0‖.
    n_g = len(GRAINES_50)
    k_star = cd.k_star_sign_test(n_g)
    puissances = {
        "contre_75pct": cd.puissance_sign_test(n_g, k_star, Fraction(3, 4)),
        "contre_70pct": cd.puissance_sign_test(n_g, k_star, Fraction(7, 10)),
    }
    lam_full = _lambda_full_t65(root)
    normes_a = np.linalg.norm(tr.a["S"], axis=1)
    publications = {
        "k_star_n50": k_star,
        "p_a_k_star": float(2 * sum(cd.binom_pmf_fraction(n_g, Fraction(1, 2))[k_star:],
                                    Fraction(0))),
        "puissances": puissances,
        "regle_sigma_d": {"seuil": C2_SIGMA_D_MAX,
                          "MDE_formule": "3.418·sigma_d/sqrt(50) = 0.4834·sigma_d",
                          "delta_Lambda_min": C2_D_MIN},
        "regle_sigma_E": {"seuil": C7_SIGMA_E_MAX,
                          "MDE_formule": "3.418·sigma/sqrt(2048) = 0.07553·sigma"},
        "Lambda_FULL_T65_provenance": "TOUR65_ABLATION.json, champ traj_Lambda "
                                      "des 50 runs variante=FULL (voie close : "
                                      "cité, jamais re-mesuré)",
        "Lambda_FULL_T65_par_graine": {str(k): lam_full[k] for k in sorted(lam_full)},
        "Lambda_FULL_T65_mediane": float(np.median(list(lam_full.values()))),
        "mediane_norme_s0_TRAIN": float(np.median(normes_a)),
        "bande_TENUE_Lambda": list(BANDE_TENUE),
        "sign_test_n2048": {"seuil_detection": "52.85 %",
                            "stride4_n512": "55.7 %"},
    }

    # (g) Q12 — scan du code neuf + liste close des artefacts lus.
    ici = Path(__file__).resolve()
    test_file = ici.parents[2] / "tests" / "test_duration_training.py"
    q12 = scan_q12([ici, test_file])
    q12["artefacts_anterieurs_licites"] = list(ARTEFACTS_ANTERIEURS_LICITES)
    q12["poids_t64_lus"] = [f"{POIDS_T64}/{RUN_T64_PREFIX}{s}.bin (S seule)"
                            for s in GRAINES_A0]

    payload = {
        "tour": 67, "etape": "P1",
        "C6_parite_au_bit": {"par_graine": c6_par_graine, "PASS": c6_pass,
                             "regle": "100 % ou FAUX (99 % = FAUX)"},
        "P1b_parites_chaine": {**parites_b, "PASS": p1b_pass},
        "decoupe_aba_v2": decoupe,
        "operateurs_fractions": ops_ok,
        "operateurs_sous_10pct": operateurs_sous_10pct,
        "C7_gates_corpus": c7,
        "publications": publications,
        "Q12_scan": q12,
        "PASS_toutes_gates": bool(c6_pass and p1b_pass and c7["PASS"]
                                  and q12["PASS"]),
        "partages_de_sort": {
            "C6_FAIL": "C2/C3/C4/C5 NON-MESURE ; C1 opposable en absolu (I6)",
            "C7_FAIL": "C3/C4/C5 NON-MESURE ; TEST_V2 jamais ouvert (I7) ; "
                       "C1/C2 intacts",
            "Q12_FAIL": "DISSIPATION, rollback (I10) — seule chute qui ARRÊTE",
        },
    }
    # TOUR67_SPLITS_V2.json (découpe + md5 par tranche) — entre au digest β.
    trd.ecrire_artefact({"tour": 67, "artefact": "SPLITS_V2", **decoupe},
                        root / ART["splits_v2"])
    # TOUR67_INSTRUMENTS.json — portées Q1-Q13 (T60), AVANT toute mesure de
    # verdict (le gel β les scelle).
    trd.ecrire_artefact(portees_instruments(), root / ART["instruments"])
    if not q12["PASS"]:
        raise RuntimeError("Q12 FAIL — DISSIPATION (I10) : STOP net, consigner")
    return payload


def portees_instruments() -> dict:
    """PORTÉE par instrument (émission §4.6, Q1-Q13) — quantité qui borne ce
    que l'instrument détecte + atteignabilité des DEUX branches."""
    return {
        "tour": 67, "artefact": "INSTRUMENTS",
        "Q1_Lambda": {
            "role": "PORTEUR C1/C2",
            "borne": "plancher ≈ −0.45 (eps 1e-12, 64 pas, ‖s0‖≈2.68) ; plafond ≈ +12.8 "
                     "(censure Θ=1e6 dès t=1) ; ne distingue pas deux contractions plus "
                     "rapides que le plancher — d'où I-6 en garde séparée",
            "branches": "Λ>0 (poids T64, cité T65) et Λ<0 (figé/ridge, cités et ridge "
                        "re-mesurée ce tour)"},
        "Q2_f_div": {
            "role": "PORTEUR C1",
            "borne": "détecte une croissance ≥ (1e6/2.68)^(1/64) ≈ 1.216/pas ; "
                     "« non divergent » ≠ « stable »",
            "branches": "f_div→1 (T64, cité) et f_div→0 (ridge, re-mesurée)"},
        "Q3_rho0": {
            "role": "INFO (support C1)",
            "borne": "compagnon 38×38 AVEUGLE à B(s²) (dérivée nulle en 0) ; zone morte "
                     "|rho0−1| < 0.01 ; précision eigvals ~1e-12",
            "branches": ">1 et <1 toutes deux atteintes au T65"},
        "Q4_R_disp_64": {
            "role": "PORTEUR de la garde de diversité (C1) — instrument NEUF (l.341)",
            "borne": "détecte une perte de diversité ≥ 5× (seuil 0.20) ; ne détecte pas "
                     "une contraction < 5× ; N-A si < 90 % des paires finies",
            "branches": "R→0 atteignable (ridge itérée, P-RIDGE 85/15 — branche FAIL "
                        "exercée CE tour) ; R≈1 atteignable (identité, prouvé au test)"},
        "Q5_sign_test_n50": {
            "role": "PORTEUR C2",
            "borne": "k* = 35 ; puissance 0.84 vs 75 %, 0.56 vs 70 %",
            "branches": "50 % et ~100 % atteignables"},
        "Q6_sign_test_n2048": {
            "role": "PORTEUR C3/C4/C5",
            "borne": "seuil de détection 52.85 % ; effectif effectif réduit par la "
                     "macro-récursion A′(n)=A(n+1) — borne traitée par la contre-épreuve "
                     "stride-4 (n=512, 55.7 %)",
            "branches": "les deux atteignables (identité ⇒ 50 %)"},
        "Q7_mediane_E": {
            "role": "PORTEUR C3 (non-infériorité)",
            "borne": "MDE = 0.07553·sigma_E ; NON-MESURE si sigma_E > 0.662 (estimateur "
                     "gelé : max des contrastes appariés disponibles à P1 sur VAL_V2 — "
                     "inter-graines T64 et T64−ridge, ddof=0 ; scellé avant TEST_V2)",
            "branches": "E>0 et E<−0.05 atteignables"},
        "Q8_parite_bit": {
            "role": "GATE C6, substrat EXACT",
            "borne": "toute divergence de spec au dernier bit ; 100 % ou FAUX",
            "branches": "PASS (déterminisme prouvé 16×) et FAIL (re-dérivation "
                        "infidèle) atteignables"},
        "Q9_gates_corpus": {
            "role": "GATE C7",
            "borne": "R_in détecte une concentration ≥ 2× vs TRAIN ; R_disp_K2 ≥ 0.20 "
                     "détecte un effondrement ≥ 5× ; d̄ ≥ 0.05 détecte l'identité "
                     "déguisée, PAS une copie à 5-10 %",
            "branches": "les deux atteignables ; valeurs absolues TRAIN et VAL_V2 publiées"},
        "Q10_ridge": {
            "role": "RIVAL OBLIGATOIRE C4 + repère de durée",
            "borne": "bornée par le meilleur transport affine ; rho(W) exact",
            "branches": "« ridge gagne » et « ridge perd » atteignables (T64 : elle a "
                        "perdu de +0.0708)"},
        "Q11_rescale_oracle": {
            "role": "RIVAL C5, AVANTAGÉ",
            "borne": "borné par le meilleur des 10 γ (grille close) AVEC l'oracle du "
                     "split de verdict ⇒ borne supérieure du shrinkage trivial",
            "branches": "« aucun γ ne tient » (C5 par défaut, publié) et « un γ tient "
                        "et gagne » atteignables"},
        "Q12_gate_materielle": {
            "role": "GATE",
            "borne": "détecte les routes vers le split brûlé de dataset_aba, la "
                     "matérialisation de RESERVE_V2 (ValueError structurel + scan), la "
                     "lecture des 4 artefacts T64 interdits ; ne détecte pas une fuite "
                     "par mémoire humaine — d'où l'interdit écrit",
            "branches": "détection / non-détection atteignables"},
        "Q13_point_fixe": {
            "role": "INFO",
            "borne": "détecte l'immobilité quasi exacte (< 1e-6) ; ne qualifie RIEN à "
                     "lui seul (§1 : l'immobilité n'est pas le verrouillage)",
            "branches": "les deux atteignables (T65 : a mordu sous bounded)"},
        "substrat": "numérique-géométrique (s ∈ R^19, normes, spectres, cos−l2, float64)",
        "order_sensibilite": "N-A MOTIVÉE : Φ^K est une puissance d'une seule "
                             "application, le segment B n'entre pas (s_prev=0) — aucune "
                             "revendication d'ordre (règle T22/T23)",
        "agregat": "MÉDIANE partout pour le verdict ; moyennes en INFO ; POINTE "
                   "(cartouche S) gelée, unique",
    }


def specs_gelees() -> Tuple[dict, dict]:
    """TOUR67_SPEC.json / TOUR67_PREDICTIONS.json (transcription émission)."""
    spec = {
        "tour": 67, "artefact": "SPEC",
        "objet": "la durée DANS la perte : 4 bras (A0 parité, A1 MULTI-K, A2 STAB, WD), "
                 "50 graines/bras (+3 A0), verdict de durée K=64 en partition close à 5 "
                 "classes + transport jamais-vu sur aba_v2 TEST_V2",
        "base": "TOUR64_SPEC.json repris à l'identique ; 7 écarts énumérés §4.4 "
                "(pertes A1/A2, K d'entraînement, graines 0-49, grille supprimée, "
                "cartouche S seule, weight_decay WD, lambda_stab gelé)",
        "pertes": {
            "A0": "ell(2) — exactement T64 (gate C6)",
            "A1": "(1/3)·[ell(2)+ell(4)+ell(8)], K={2,4,8} close en extension",
            "A2": f"ell(2) + {LAMBDA_STAB}·mean relu((1/{K_PEN})·log((‖s_{K_PEN}‖+eps)/(‖a‖+eps)))²",
            "WD": f"ell(2) + weight_decay {WEIGHT_DECAY_WD} (optimiseur)",
        },
        "ell_K": "ell(K) = −mean_i s_i^(K) + lambda_copy·mean_i relu(m − d_i^(K))² ; "
                 "s^(K) = cos−l2 (parité M1) ; d^(K) = ‖Φ(a,K)−a‖/(‖a‖+eps)",
        "config_unique": {"init_scale": INIT_SCALE, "lr": LR},
        "m": PARITE_M_S,
        "graines": {"A0": list(GRAINES_A0), "A1": list(GRAINES_50),
                    "A2": list(GRAINES_50), "WD": list(GRAINES_50)},
        "early_stop": "médiane Δ VALIDATION (K=2, dataset_aba 3001-3500) — "
                      "transport-seul, AUCUN bras arrêté sur un critère de durée",
        "K_eval": K_EVAL, "K_max": K_MAX, "theta_div": cd.THETA_DIV,
        "partition": {"ordre": ["DIVERGENT", "BASSIN", "EXTINCTION",
                                "VERROUILLE", "BORNE_VIVANT"],
                      "f_div_divergent": SEUIL_F_DIV, "f_div_borne": SEUIL_F_BORNE,
                      "facteur_borne": FACTEUR_BORNE,
                      "extinction": FACTEUR_EXTINCTION,
                      "R_disp64_verrouille": R_DISP64_MIN,
                      "R_disp64_frac_finie_min": R_DISP64_FRAC_FINIE_MIN},
        "roles_aba_v2": {nom: list(b) for nom, b in TRANCHES_V2.items()},
        "reserve_v2": {"bornes": list(RESERVE_V2_BORNES),
                       "statut": "JAMAIS MATÉRIALISÉE (parse structurel/md5 seulement)"},
        "rivaux": "identité ; poids T64 (transport neuf, durée citée) ; ridge λ=1 "
                  "re-fittée (transport + itérée) ; c̄ ; rescale-γ 10×3 (oracle au "
                  "rival) ; WD ; figé T63 cité",
        "gammas": list(GAMMAS),
        "selecteur_g_star": "médiane basse (rang (n−1)//2) de médiane Δ sur VAL_V2, "
                            "parmi les graines BORNÉ-VIVANT si ≥5, sinon règles §4.2",
        "repli_instabilite": f">= {SEUIL_REPLI_DIVERGES}/50 DIVERGED dans un bras => "
                             "survivants avec effectif publié ; AUCUN bounded=True",
        "precision": "float64, CPU, torch.manual_seed, use_deterministic_algorithms(True)",
    }
    predictions = {
        "tour": 67, "artefact": "PREDICTIONS",
        "H67": "(A) TENUE : >= 1 bras de {A1, A2} BORNÉ-VIVANT >= 40/50 ; (B) TRANSPORT : "
               "médiane E >= −0.05, médiane Δ > 0 p < 0.01, bat la ridge ; (C) NON-TRIVIALITÉ : "
               "bat rescale-oracle ET weight-decay",
        "cotes": {"P_A1": "35/65", "P_A1_prime_verrouille": "40/60", "P_A2": "50/50",
                  "P_H67A_disjonction": "65/35", "P_B": "55/45", "P_C": "65/35",
                  "P_RIDGE_verrouillee": "85/15", "P_T64GEN": "70/30",
                  "H67_globale": "40/60"},
        "conjoncts": {
            "C1": "PORTEUR principal — #BORNÉ-VIVANT >= 40/50 par bras ; distribution "
                  "sur les 5 classes publiée dans tous les cas",
            "C2": "PORTEUR — d(g)=Λ_FULL^{T65}(g)−Λ_arm(g) : médiane >= 0.25 ET "
                  "sign-test p < 0.01 (k >= 35/50) ; sigma_d > 0.5172 => magnitude NON-MESURE",
            "C3": "PORTEUR — TEST_V2, g* : médiane E >= −0.05 ET médiane Δ > 0 p < 0.01 "
                  "ET stride-4 concordante ; sigma_E > 0.662 => NON-MESURE",
            "C4": "PORTEUR — médiane F > 0, p < 0.01 (vs ridge re-fittée)",
            "C5": "PORTEUR conditionnel — médiane G > 0 ET médiane H > 0, p < 0.01, "
                  "restreint aux rivaux de shrinkage qui TIENNENT ; aucun ne tient => "
                  "SATISFAIT PAR DÉFAUT, publié chiffré",
            "C6": "GATE — parité au bit A0 (3 sha256 fichier + state), 100 % ou FAUX",
            "C7": "GATE — gates corpus VAL_V2 (parse, finitude, R_in, R_disp_K2, dbar, sigma_E)",
            "C8": "GATE par bras — descente >= 20 % sur >= 43/50 ; lam finis 100 % ; std > 0",
            "C9": "INFO liste close — K intermédiaires, transport K 8/64, rho0, "
                  "best_return_step, table gammas, corrélation Λ↔Δ, Δ 50 graines, "
                  "point-fixe, variante spectrale NON JOUÉE (déclarée)",
        },
        "partages_de_sort": "C6 -> C2/C3/C4/C5 NON-MESURE (C1 opposable en absolu) ; "
                            "C7 -> C3/C4/C5 NON-MESURE (C1/C2 intacts) ; C8 -> le bras",
        "issues": "I1-I15 de l'émission §7, chacune portant son verdict de tour",
    }
    return spec, predictions


def etape_gel(root: Path) -> dict:
    """β : écrit SPEC/PREDICTIONS puis scelle le jeton (22 chemins, §4.11)
    AVANT la première descente de verdict."""
    exiger(root, "p1", "splits_v2", "instruments", "etalonnage", "p0b")
    spec, predictions = specs_gelees()
    trd.ecrire_artefact(spec, root / ART["spec"])
    trd.ecrire_artefact(predictions, root / ART["predictions"])
    token = freeze_token([root / p for p in FREEZE_PATHS])
    write_token(token, root / ART["freeze"])
    return {"tour": 67, "etape": "GEL", "digest": token.digest,
            "n_chemins": len(token.artifacts),
            "mtimes_ns": {a.path: a.mtime_ns for a in token.artifacts}}


def etape_train(root: Path) -> dict:
    """P2-train : les 153 descentes (A1 50, A2 50, WD 50, A0 3) ; C8 partiel
    (descente de perte) ; courbes publiées (TOUR67_COURBES.json)."""
    exiger(root, "freeze", "etalonnage")
    verifier_gel(root)
    t_debut = time.perf_counter()
    tok, m1 = _instrument()
    tr, va = _splits(tok, m1, root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    poids_dir = root / POIDS_DIR
    poids_dir.mkdir(exist_ok=True)

    runs: List[dict] = []
    courbes: Dict[str, dict] = {}
    plan: List[Tuple[str, int]] = ([("A1", g) for g in GRAINES_50]
                                   + [("A2", g) for g in GRAINES_50]
                                   + [("WD", g) for g in GRAINES_50]
                                   + [("A0", g) for g in GRAINES_A0])
    for bras, seed in plan:
        budget_verifier(root, time.perf_counter() - t_debut)
        rr = entrainer_bras(m1, bras, seed, tr.a["S"], tr.ap["S"],
                            va.a["S"], va.ap["S"], s_raw_val, m)
        rid = f"{bras}_seed{seed}"
        fichier = poids_dir / f"{rid}.bin"
        sha_fichier = rt.sauver_poids(rr.state, fichier)
        runs.append({
            "run_id": rid, "bras": bras, "seed": seed,
            "diverged_train": rr.diverged, "epochs_run": rr.epochs_run,
            "best_epoch": rr.best_epoch, "early_stopped": rr.early_stopped,
            "best_delta_val_K2": rr.best_delta_val,
            "loss_init": rr.loss_init, "loss_min": rr.loss_min,
            "descente_rel": rr.descente_rel,
            "descente_ge_20pct": bool(math.isfinite(rr.descente_rel)
                                      and rr.descente_rel >= rt.SEUIL_PERTE_DESCENTE),
            "state_sha256": rr.state_sha256,
            "poids_fichier": fichier.name, "poids_sha256": sha_fichier,
        })
        courbes[rid] = rr.courbes
    c8_partiel = {}
    for bras in BRAS_VERDICT:
        rs = [r for r in runs if r["bras"] == bras]
        n_desc = sum(1 for r in rs if r["descente_ge_20pct"])
        n_div = sum(1 for r in rs if r["diverged_train"])
        c8_partiel[bras] = {
            "n_descendus_ge_20pct": n_desc, "quorum": C8_QUORUM,
            "descente_PASS": bool(n_desc >= C8_QUORUM),
            "n_diverges_train": n_div,
            "repli_survivants": bool(n_div >= SEUIL_REPLI_DIVERGES),
        }
    trd.ecrire_artefact({"tour": 67, "artefact": "COURBES",
                         "axes": ["perte_train", "delta_med_val", "r_disp", "dbar"],
                         "par_run": courbes}, root / ART["courbes"])
    return {
        "tour": 67, "etape": "TRAIN",
        "n_runs": len(runs),
        "C8_partiel_descente": c8_partiel,
        "runs": runs,
        "duree_s": time.perf_counter() - t_debut,
    }


def etape_duree(root: Path) -> dict:
    """P2-duree — C1 : classification des 150 modèles neufs (A1/A2/WD × 50)
    + ridge itérée + 30 rescale-γ, partition close à 5 classes. AUCUNE
    quantité d'aba_v2 n'entre ici (la durée est scellée AVANT le transport).
    Le régime K=64 des poids T64 et du figé T63 est CITÉ (voie close)."""
    exiger(root, "freeze", "train")
    verifier_gel(root)
    verifier_sceau(root, "train")
    t_debut = time.perf_counter()
    tok, m1 = _instrument()
    tr, _ = _splits(tok, m1, root)
    a_s = tr.a["S"]
    pairs = paires_67(a_s.shape[0])
    train_art = json.loads((root / ART["train"]).read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in train_art["runs"]}

    par_bras: Dict[str, dict] = {}
    for bras in BRAS_VERDICT:
        # Repli instabilité (§4.4, GELÉ) : les descentes DIVERGED sont
        # EXCLUES de la mesure, comptées et publiées ; le bras est mesuré
        # sur les SURVIVANTS avec l'effectif publié.
        par_graine = {}
        survivants: List[int] = []
        for seed in GRAINES_50:
            r = runs[f"{bras}_seed{seed}"]
            if r["diverged_train"]:
                par_graine[str(seed)] = {"classe": "EXCLU_DIVERGED_TRAIN",
                                         "diverged_train": True,
                                         "epochs_avant_divergence": r["epochs_run"]}
                continue
            budget_verifier(root, time.perf_counter() - t_debut)
            model = modele_depuis_fichier(root, r["poids_fichier"])
            res = _classer_modele(model, a_s, pairs)
            par_graine[str(seed)] = {**_resume_classement(res),
                                     "diverged_train": False}
            survivants.append(seed)
        classes = {c: sum(1 for g in par_graine.values() if g["classe"] == c)
                   for c in CLASSES}
        classes["EXCLU_DIVERGED_TRAIN"] = len(GRAINES_50) - len(survivants)
        lams = np.asarray([par_graine[str(s)]["Lambda"] for s in survivants])
        finitude = bool(all(par_graine[str(s)]["finitude_lam_100pct"]
                            for s in survivants))
        non_vacuite = bool(all(par_graine[str(s)]["lam_std"] > 0.0
                               for s in survivants))
        c8_train = train_art["C8_partiel_descente"][bras]
        c8_pass = bool(c8_train["descente_PASS"] and finitude and non_vacuite)
        par_bras[bras] = {
            "par_graine": par_graine,
            "classes": classes,
            "survivants": survivants,
            "n_survivants": len(survivants),
            "repli_survivants_actif": bool(classes["EXCLU_DIVERGED_TRAIN"]
                                           >= SEUIL_REPLI_DIVERGES),
            "C1_borne_vivant": classes["BORNE_VIVANT"],
            "C1_quorum": C1_QUORUM,
            "C1_PASS": bool(classes["BORNE_VIVANT"] >= C1_QUORUM),
            "Lambda_mediane_survivants": float(np.median(lams)) if len(lams) else None,
            "Lambda_q25": float(np.percentile(lams, 25)) if len(lams) else None,
            "Lambda_q75": float(np.percentile(lams, 75)) if len(lams) else None,
            "C8": {**c8_train, "finitude_lam_100pct": finitude,
                   "non_vacuite_std_pos": non_vacuite, "PASS": c8_pass},
        }

    # Ridge itérée (rival de durée, P-RIDGE : VERROUILLÉE, 85/15).
    theta = _ridge_train(tr)
    trj_r = cd.derouler(cd.pas_ridge(theta), a_s)
    res_r = classer_regime(trj_r, a_s, pairs)
    ridge_pub = {**_resume_classement({**res_r, "rho0": float(np.max(np.abs(
        np.linalg.eigvals(theta[:-1].astype(np.float64)))))}),
        "note_rho": "rho(W) exact du 19×19 (pas de compagnon : premier ordre)"}

    # RESCALE-γ : 3 graines × 10 γ, classés par la même partition.
    rescale_pub = []
    for seed in GRAINES_A0:
        for gamma in GAMMAS:
            budget_verifier(root, time.perf_counter() - t_debut)
            model = modele_rescale(root, seed, gamma)
            res = _classer_modele(model, a_s, pairs)
            rescale_pub.append({"seed": seed, "gamma": gamma,
                                **_resume_classement(res)})
    n_rescale_vivants = sum(1 for r in rescale_pub if r["classe"] == "BORNE_VIVANT")

    payload = {
        "tour": 67, "etape": "DUREE",
        "grandeur": "classe de régime K=64 (partition close 5 classes) aux points de "
                    "fonctionnement TRAIN 101-1000 (R-INT : ces poids-ci, aucune "
                    "généralisation)",
        "par_bras": par_bras,
        "ridge_iteree": ridge_pub,
        "rescale_gamma": rescale_pub,
        "n_rescale_borne_vivant": n_rescale_vivants,
        "cites_voie_close": {
            "poids_T64_FULL": {"rho0": [1.4140, 1.4177, 1.3555],
                               "Lambda": [0.7104, 0.8318, 1.7475],
                               "f_div": [1.0, 1.0, 1.0], "H50": [26, 20, 12],
                               "classe": "DIVERGENT",
                               "source": "TOUR65_REGIME.json (cité, non re-mesuré)"},
            "fige_T63": {"Lambda": -0.4472, "f_div": 0.0,
                         "classe_par_I6": "EXTINCTION",
                         "source": "TOUR65_P1.json (cité, non re-mesuré)"},
        },
        "duree_s": time.perf_counter() - t_debut,
    }
    return payload


def etape_sigma(root: Path) -> dict:
    """P2-sigma : σ̂_d et MDE réalisés par bras, SCELLÉS avant le sign-test
    C2 (règle mécanique gelée §4.5)."""
    exiger(root, "freeze", "duree")
    verifier_gel(root)
    verifier_sceau(root, "duree")
    duree = json.loads((root / ART["duree"]).read_text(encoding="utf-8"))
    lam_full = _lambda_full_t65(root)
    payload = {"tour": 67, "etape": "SIGMA",
               "regle": "gelée §4.5 : magnitude C2 NON-MESURE si sigma_d > "
                        f"{C2_SIGMA_D_MAX} ; ddof=0 (convention T64/T65) ; "
                        "survivants seuls (repli §4.4), effectif publié"}
    for bras in BRAS_VERDICT:
        pg = duree["par_bras"][bras]["par_graine"]
        surv = duree["par_bras"][bras]["survivants"]
        d = np.asarray([lam_full[g] - pg[str(g)]["Lambda"] for g in surv])
        payload[f"d_{bras}"] = {
            "n_survivants": len(surv),
            "sigma_d": float(d.std()),
            "MDE_realise": float(Z_SUM * d.std() / math.sqrt(len(surv))),
            "magnitude_NON_MESURE": bool(d.std() > C2_SIGMA_D_MAX),
        }
    return payload


def etape_contraste(root: Path) -> dict:
    """P2-contraste — C2 : ``d_arm(g) = Λ_FULL^{T65}(g) − Λ_arm(g)``, 50
    graines appariées par init (C6 le prouve), médiane + sign-test exact.
    A1/A2 portent le conjonct ; WD publié en INFO (rival)."""
    exiger(root, "freeze", "duree", "sigma")
    verifier_gel(root)
    verifier_sceau(root, "duree")
    verifier_sceau(root, "sigma")
    duree = json.loads((root / ART["duree"]).read_text(encoding="utf-8"))
    sig = json.loads((root / ART["sigma"]).read_text(encoding="utf-8"))
    p1 = json.loads((root / ART["p1"]).read_text(encoding="utf-8"))
    c6_ok = bool(p1["C6_parite_au_bit"]["PASS"]
                 and p1["P1b_parites_chaine"]["PASS"])
    lam_full = _lambda_full_t65(root)
    resultats = {}
    for bras in BRAS_VERDICT:
        pg = duree["par_bras"][bras]["par_graine"]
        surv = duree["par_bras"][bras]["survivants"]
        d = np.asarray([lam_full[g] - pg[str(g)]["Lambda"] for g in surv])
        k_star = cd.k_star_sign_test(len(surv))
        med = float(np.median(d))
        st = rt.sign_test_exact(d)
        non_mesure = bool(sig[f"d_{bras}"]["magnitude_NON_MESURE"])
        magnitude_ok = bool(med >= C2_D_MIN)
        sign_ok = bool(st["p"] < SEUIL_P and med > 0)
        resultats[bras] = {
            "statut": "PORTEUR C2" if bras in ("A1", "A2") else "INFO (rival)",
            "n_survivants": len(surv), "graines_survivantes": list(surv),
            "d_par_graine": d.tolist(),
            "mediane_d": med, "moyenne_d_INFO": float(d.mean()),
            "sign_test": st, "k_star_effectif": k_star,
            "magnitude_ge_0.25": magnitude_ok,
            "magnitude_NON_MESURE_sigma": non_mesure,
            "sign_test_p_lt_001": sign_ok,
            "C2_verdict": bool(magnitude_ok and sign_ok and not non_mesure),
            "C2_sign_seul_si_sigma_mord": sign_ok,
        }
    return {"tour": 67, "etape": "CONTRASTE",
            "grandeur": "d(g) = Lambda_FULL_T65(g) − Lambda_bras(g), appariée par "
                        "graine ET par init (50), FULL cité jamais re-mesuré",
            "C6_et_parites_chaine_PASS": c6_ok,
            "C2_NON_MESURE_par_C6": bool(not c6_ok),
            "resultats": resultats}


def _selecteur_g_star(duree_bras: dict, deltas_val: Dict[int, float]) -> dict:
    """Sélecteur g* (§4.2) : médiane basse de médiane Δ VAL_V2, parmi les
    graines BORNÉ-VIVANT (≥5), sinon parmi les survivants (conjonction
    perdue, déclarée ; DIVERGED exclus, repli §4.4). JAMAIS le maximum."""
    surv = list(duree_bras["survivants"])
    bv = [g for g in surv
          if duree_bras["par_graine"][str(g)]["classe"] == "BORNE_VIVANT"]
    if len(bv) >= 5:
        pool = bv
        porte_conjonction = True
    else:
        pool = surv
        porte_conjonction = False
    g_star = mediane_basse({g: deltas_val[g] for g in pool})
    return {"g_star": g_star,
            "pool": "BORNE_VIVANT" if porte_conjonction else "SURVIVANTS",
            "effectif_pool": len(pool),
            "porte_conjonction_tenue_et_transport": porte_conjonction,
            "n_borne_vivant": len(bv)}


def etape_transport(root: Path) -> dict:
    """P2-transport — OUVERTURE UNIQUE de TEST_V2 (C3/C4/C5), APRÈS le
    scellement de la durée. D'abord les sélections sur VAL_V2 (g* par bras,
    déjà ouverte depuis P1), PUIS la matérialisation de TEST_V2."""
    exiger(root, "freeze", "duree", "sigma", "contraste", "p1")
    verifier_gel(root)
    verifier_sceau(root, "duree")
    verifier_sceau(root, "contraste")
    t_debut = time.perf_counter()
    p1 = json.loads((root / ART["p1"]).read_text(encoding="utf-8"))
    c6_ok = bool(p1["C6_parite_au_bit"]["PASS"]
                 and p1["P1b_parites_chaine"]["PASS"])
    c7_ok = bool(p1["C7_gates_corpus"]["PASS"])
    if not (c6_ok and c7_ok):
        # Partage de sort ÉCRIT (émission §4.2/§3.3) : C7 FAIL ⇒ C3/C4/C5
        # NON-MESURE et TEST_V2 JAMAIS OUVERT (I7, jamais-vu économisé) ;
        # C6 FAIL ⇒ C3/C4/C5 NON-MESURE. Aucune matérialisation ici.
        return {
            "tour": 67, "etape": "TRANSPORT",
            "NON_MESURE": True,
            "motif": {"C6_et_parites_chaine_PASS": c6_ok,
                      "C7_gates_corpus_PASS": c7_ok},
            "TEST_V2_ouvert": False,
            "VAL_V2_selections_g_star": "non calculées (sans objet sans "
                                        "verdict de transport)",
            "conjoncts": {"C3": "NON-MESURE", "C4": "NON-MESURE",
                          "C5": "NON-MESURE",
                          "P_T64GEN": "NON-MESURE (TEST_V2 non ouvert)"},
            "acquis_C7": p1["C7_gates_corpus"],
            "phrase_publiable": "aba_v2 ne porte pas un Δ apparié au grain "
                                "agrégat / cartouche S (pointe) / MAIN 19d : "
                                "la géométrie de VAL_V2 est dégénérée "
                                "(R_in = 0, support quasi-constant) — acquis "
                                "chiffré pour les tours futurs, 2048 cycles "
                                "de jamais-vu ÉCONOMISÉS",
            "duree_s": time.perf_counter() - t_debut,
        }
    tok, m1 = _instrument()
    tr, _ = _splits(tok, m1, root)
    duree = json.loads((root / ART["duree"]).read_text(encoding="utf-8"))
    train_art = json.loads((root / ART["train"]).read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in train_art["runs"]}

    # --- Sélections sur VAL_V2 (AVANT toute lecture de TEST_V2) -----------
    va2 = charger_tranche_v2(tok, m1, root, "VAL_V2")
    s_raw_va2 = rt.scores_m1(m1, va2.a, va2.ap)
    deltas_val: Dict[str, Dict[int, float]] = {}
    for bras in BRAS_VERDICT:
        dv = {}
        for seed in duree["par_bras"][bras]["survivants"]:
            model = modele_depuis_fichier(
                root, runs[f"{bras}_seed{seed}"]["poids_fichier"])
            x = rt.appliquer_chrono(model, va2.a)
            dv[seed] = float(np.median(rt.deltas(m1, x, va2.ap, s_raw_va2)))
        deltas_val[bras] = dv
    selections = {bras: _selecteur_g_star(duree["par_bras"][bras], deltas_val[bras])
                  for bras in BRAS_VERDICT}
    seed_repere = int(p1["C7_gates_corpus"]["repere_T64"]["graine_repere"])
    # Gardes I-8 du g* de chaque bras sur VAL_V2 (publiées avant TEST_V2).
    pairs_v2 = paires_67(va2.n)
    gardes_g_star = {}
    for bras in BRAS_VERDICT:
        model = modele_depuis_fichier(
            root, runs[f"{bras}_seed{selections[bras]['g_star']}"]["poids_fichier"])
        x = rt.appliquer_chrono(model, va2.a)
        r_disp_k2, dbar = rt.gardes_collapse(x, va2.a, pairs_v2)
        gardes_g_star[bras] = {"R_disp_K2": r_disp_k2, "dbar": dbar,
                               "R_disp_K2_ge_0.20": bool(r_disp_k2 >= R_DISP_K2_MIN),
                               "dbar_ge_0.05": bool(dbar >= DBAR_MIN)}

    # --- OUVERTURE UNIQUE DE TEST_V2 --------------------------------------
    te2 = charger_tranche_v2(tok, m1, root, "TEST_V2")
    attendu = p1["decoupe_aba_v2"]["tranches"]["TEST_V2"]["md5_octets_bruts"]
    if te2.md5_octets != attendu:
        raise RuntimeError("TEST_V2 != gel P1 (md5) — ARRÊT")
    s_raw_te = rt.scores_m1(m1, te2.a, te2.ap)

    def scores_modele(model: ChronoSpiraton) -> np.ndarray:
        return rt.scores_m1(m1, rt.appliquer_chrono(model, te2.a), te2.ap)

    # Rivaux fixes.
    s_t64 = scores_modele(modele_t64(root, seed_repere))
    delta_t64 = s_t64 - s_raw_te
    theta = _ridge_train(tr)
    s_ridge = rt.scores_m1(m1, rt.predire_ridge(theta, te2.a), te2.ap)
    delta_ridge = s_ridge - s_raw_te
    cbar = rt.constante_cbar(tr.ap["S"])
    s_const = rt.scores_m1(m1, np.tile(cbar, (te2.n, 1)), te2.ap)
    delta_const = s_const - s_raw_te

    # P-T64GEN : les poids T64 généralisent-ils sur aba_v2 ?
    st_t64 = rt.sign_test_exact(delta_t64)
    t64gen = {"delta_mediane": float(np.median(delta_t64)),
              "sign_test": st_t64,
              "stride4": contre_epreuve_stride(delta_t64),
              "PASS_p_t64gen": bool(np.median(delta_t64) > 0
                                    and st_t64["p"] < SEUIL_P),
              "graine_repere": seed_repere}

    # RESCALE-γ : oracle TEST_V2 parmi ceux qui TIENNENT (classe BORNÉ-VIVANT).
    rescale_table = []
    meilleurs = []
    for rr in duree["rescale_gamma"]:
        tient = rr["classe"] == "BORNE_VIVANT"
        entry = {"seed": rr["seed"], "gamma": rr["gamma"],
                 "classe": rr["classe"], "Lambda": rr["Lambda"], "tient": tient}
        s_g = scores_modele(modele_rescale(root, rr["seed"], rr["gamma"]))
        entry["delta_mediane_TEST_V2"] = float(np.median(s_g - s_raw_te))
        rescale_table.append(entry)
        if tient:
            meilleurs.append((entry["delta_mediane_TEST_V2"],
                              rr["seed"], rr["gamma"], s_g))
    if meilleurs:
        meilleurs.sort(key=lambda t: (t[0], t[1], t[2]))
        best = meilleurs[-1]
        gamma_star = {"tient_au_moins_un": True, "seed": best[1],
                      "gamma": best[2], "delta_mediane": best[0],
                      "oracle": "max de médiane Δ sur TEST_V2 parmi les "
                                "BORNÉ-VIVANT — donné au rival"}
        s_gamma_star: Optional[np.ndarray] = best[3]
    else:
        gamma_star = {"tient_au_moins_un": False,
                      "note": "aucun rescale-γ ne tient — C5(G) SATISFAIT PAR "
                              "DÉFAUT, table publiée chiffrée"}
        s_gamma_star = None

    # WD rival : tient s'il a ≥1 graine BORNÉ-VIVANT (lecture favorable au
    # rival, déclarée) ; g*_WD = médiane basse parmi ses BORNÉ-VIVANT.
    wd_bv = [g for g in GRAINES_50
             if duree["par_bras"]["WD"]["par_graine"][str(g)]["classe"]
             == "BORNE_VIVANT"]
    if wd_bv:
        g_wd = mediane_basse({g: deltas_val["WD"][g] for g in wd_bv})
        s_wd: Optional[np.ndarray] = scores_modele(modele_depuis_fichier(
            root, runs[f"WD_seed{g_wd}"]["poids_fichier"]))
        wd_rival = {"tient": True, "g_star_WD": g_wd,
                    "n_borne_vivant": len(wd_bv),
                    "delta_mediane": float(np.median(s_wd - s_raw_te))}
    else:
        s_wd = None
        wd_rival = {"tient": False, "n_borne_vivant": 0,
                    "note": "WD ne tient pas — C5(H) SATISFAIT PAR DÉFAUT"}

    # Bras de verdict : g*, Δ/E/F/G/H appariés + les 50 Δ (INFO).
    par_bras = {}
    for bras in BRAS_VERDICT:
        g_star = selections[bras]["g_star"]
        s_arm = scores_modele(modele_depuis_fichier(
            root, runs[f"{bras}_seed{g_star}"]["poids_fichier"]))
        delta_arm = s_arm - s_raw_te
        e = s_arm - s_t64
        f = s_arm - s_ridge
        st_d = rt.sign_test_exact(delta_arm)
        st_f = rt.sign_test_exact(f)
        stride_d = contre_epreuve_stride(delta_arm)
        c3 = {
            "mediane_E": float(np.median(e)),
            "seuil_E": C3_E_MIN,
            "E_ge_seuil": bool(np.median(e) >= C3_E_MIN),
            "sign_test_E_INFO": rt.sign_test_exact(e),
            "delta_mediane": float(np.median(delta_arm)),
            "sign_test_delta": st_d,
            "delta_pos_p_lt_001": bool(np.median(delta_arm) > 0
                                       and st_d["p"] < SEUIL_P),
            "stride4_delta": stride_d,
            "verdict": bool(np.median(e) >= C3_E_MIN
                            and np.median(delta_arm) > 0
                            and st_d["p"] < SEUIL_P
                            and stride_d["concordant"]),
        }
        c4 = {"mediane_F": float(np.median(f)), "sign_test_F": st_f,
              "verdict": bool(np.median(f) > 0 and st_f["p"] < SEUIL_P)}
        c5: Dict[str, object] = {}
        if s_gamma_star is not None:
            g_vec = s_arm - s_gamma_star
            st_g = rt.sign_test_exact(g_vec)
            c5["G_vs_rescale"] = {"mediane_G": float(np.median(g_vec)),
                                  "sign_test": st_g,
                                  "verdict": bool(np.median(g_vec) > 0
                                                  and st_g["p"] < SEUIL_P)}
        else:
            c5["G_vs_rescale"] = {"verdict": True, "par_defaut": True}
        if s_wd is not None and bras != "WD":
            h_vec = s_arm - s_wd
            st_h = rt.sign_test_exact(h_vec)
            c5["H_vs_WD"] = {"mediane_H": float(np.median(h_vec)),
                             "sign_test": st_h,
                             "verdict": bool(np.median(h_vec) > 0
                                             and st_h["p"] < SEUIL_P)}
        elif bras == "WD":
            c5["H_vs_WD"] = {"verdict": None, "note": "N-A (le bras EST le rival)"}
        else:
            c5["H_vs_WD"] = {"verdict": True, "par_defaut": True}
        deltas_50 = {}
        for seed in duree["par_bras"][bras]["survivants"]:
            s_g50 = scores_modele(modele_depuis_fichier(
                root, runs[f"{bras}_seed{seed}"]["poids_fichier"]))
            deltas_50[str(seed)] = float(np.median(s_g50 - s_raw_te))
        par_bras[bras] = {
            "selection": selections[bras],
            "delta_med_VAL_V2_par_graine": {str(k): v for k, v
                                            in deltas_val[bras].items()},
            "gardes_g_star_VAL_V2": gardes_g_star[bras],
            "C3": c3, "C4": c4, "C5": c5,
            "delta_med_TEST_V2_50_graines_INFO": deltas_50,
        }

    payload = {
        "tour": 67, "etape": "TRANSPORT",
        "grandeur": "R-GEN : transport sur aba_v2 TEST_V2 (cycles 2049-4096, F0, "
                    "cartouche S, MAIN 19d, K=2, grain agrégat) — ouverture UNIQUE, "
                    "après scellement de la durée",
        "n_TEST_V2": te2.n,
        "md5_TEST_V2": te2.md5_octets,
        "s_raw_TEST_V2": {"mediane": float(np.median(s_raw_te)),
                          "std": float(s_raw_te.std())},
        "identite": {"delta": 0.0, "note": "0 au bit par construction (P0-b)"},
        "T64_repere": t64gen,
        "ridge": {"delta_mediane": float(np.median(delta_ridge))},
        "constante_cbar": {"delta_mediane": float(np.median(delta_const)),
                           "sign_test": rt.sign_test_exact(delta_const)},
        "rescale_gamma_star": gamma_star,
        "rescale_table_TEST_V2": rescale_table,
        "WD_rival": wd_rival,
        "par_bras": par_bras,
        "duree_s": time.perf_counter() - t_debut,
    }
    return payload


def etape_info(root: Path) -> dict:
    """P2-info — C9 (liste close, hors verdict) : Λ à K ∈ {8,16,32} des g* ;
    transport à K ∈ {8,64} ; ρ₀ des bras ; best_return_step ; corrélation
    Λ↔Δ ; ratio point-fixe (repris) ; variante spectrale NON JOUÉE."""
    exiger(root, "freeze", "duree", "transport")
    verifier_gel(root)
    verifier_sceau(root, "transport")
    t_debut = time.perf_counter()
    tok, m1 = _instrument()
    tr, _ = _splits(tok, m1, root)
    a_s = tr.a["S"]
    duree = json.loads((root / ART["duree"]).read_text(encoding="utf-8"))
    transport = json.loads((root / ART["transport"]).read_text(encoding="utf-8"))
    train_art = json.loads((root / ART["train"]).read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in train_art["runs"]}

    transport_mesure = not bool(transport.get("NON_MESURE", False))
    if transport_mesure:
        g_stars = {bras: transport["par_bras"][bras]["selection"]["g_star"]
                   for bras in BRAS_VERDICT}
        selecteur_info = "g* du transport (VAL_V2)"
    else:
        # Transport NON-MESURE (C7) : sélecteur INFO déclaré, VAL_V2-libre —
        # médiane basse de best_delta_val (VALIDATION dataset_aba, K=2) parmi
        # les graines BORNÉ-VIVANT (≥1), sinon parmi les 50.
        g_stars = {}
        for bras in BRAS_VERDICT:
            surv = duree["par_bras"][bras]["survivants"]
            bv = [g for g in surv
                  if duree["par_bras"][bras]["par_graine"][str(g)]["classe"]
                  == "BORNE_VIVANT"]
            pool = bv if bv else list(surv)
            g_stars[bras] = mediane_basse(
                {g: float(runs[f"{bras}_seed{g}"]["best_delta_val_K2"])
                 for g in pool})
        selecteur_info = ("médiane basse de best_delta_val VALIDATION "
                          "dataset_aba parmi les BORNÉ-VIVANT (transport "
                          "NON-MESURE : VAL_V2 sans objet)")
    modeles_g = {bras: modele_depuis_fichier(
        root, runs[f"{bras}_seed{g}"]["poids_fichier"])
        for bras, g in g_stars.items()}

    # Λ aux K intermédiaires (g* de chaque bras).
    lambda_k = {}
    for bras, model in modeles_g.items():
        lk = {}
        for k in (8, 16, 32):
            trj = cd.derouler(cd.pas_chrono(model), a_s, k_max=k)
            lk[str(k)] = float(np.median(trj.lam))
        lambda_k[bras] = lk

    # Transport à K ∈ {8, 64} sur TEST_V2 (relecture post-verdict, INFO —
    # précédent T64). Si le transport est NON-MESURE (C7), TEST_V2 n'a
    # JAMAIS été ouvert : ces lectures INFO tombent avec lui (I7).
    if transport_mesure:
        te2 = charger_tranche_v2(tok, m1, root, "TEST_V2")
        s_raw_te = rt.scores_m1(m1, te2.a, te2.ap)
        transport_k = {}
        for bras, model in modeles_g.items():
            tk = {}
            for k in (8, 64):
                x = trd.phi_k_np(model, te2.a, k)
                if np.isfinite(x).all():
                    tk[str(k)] = float(np.median(
                        rt.scores_m1(m1, x, te2.ap) - s_raw_te))
                else:
                    fini = np.isfinite(x).all(axis=1)
                    tk[str(k)] = {
                        "n_finis": int(fini.sum()),
                        "delta_mediane_sur_finis": float(np.median(
                            rt.scores_m1(m1, x[fini], te2.ap[fini])
                            - s_raw_te[fini])) if fini.any() else None}
            transport_k[bras] = tk
    else:
        transport_k = {"NON_MESURE": "TEST_V2 jamais ouvert (C7, I7)"}

    # ρ₀ des bras (médiane des survivants + g*).
    rho0_pub = {}
    for bras in BRAS_VERDICT:
        surv = duree["par_bras"][bras]["survivants"]
        rhos = {}
        for seed in surv:
            model = modele_depuis_fichier(
                root, runs[f"{bras}_seed{seed}"]["poids_fichier"])
            rhos[seed] = cd.rho_compagnon(cd.matrices_np(model))
        vals = np.asarray(list(rhos.values()))
        rho0_pub[bras] = {"n_survivants": len(surv),
                          "mediane_survivants": float(np.median(vals)),
                          "min": float(vals.min()), "max": float(vals.max()),
                          "g_star": rhos[g_stars[bras]]}

    # best_return_step des g* (sur TRAIN, M1 inchangé).
    brs = {}
    for bras, model in modeles_g.items():
        trj = cd.derouler(cd.pas_chrono(model), a_s, garder_etats=True)
        argmaxes = []
        for i in range(trj.n):
            fin = int(trj.t_prime[i])
            if fin < 1:
                argmaxes.append(0)
                continue
            scores = [m1.score_retour(trj.etats[i, t - 1, :], tr.ap["S"][i])
                      for t in range(1, fin + 1)]
            argmaxes.append(int(np.argmax(scores)) + 1)
        arr = np.asarray(argmaxes)
        brs[bras] = {"mediane": float(np.median(arr)),
                     "fraction_le_2": float(np.mean(arr <= 2)),
                     "min": int(arr.min()), "max": int(arr.max())}

    # Corrélation Λ (K=64) ↔ médiane Δ TEST_V2 + moyennes (clause T40).
    if transport_mesure:
        correlations = {}
        for bras in BRAS_VERDICT:
            surv = duree["par_bras"][bras]["survivants"]
            lams = [duree["par_bras"][bras]["par_graine"][str(g)]["Lambda"]
                    for g in surv]
            dels = [transport["par_bras"][bras]
                    ["delta_med_TEST_V2_50_graines_INFO"][str(g)]
                    for g in surv]
            correlations[bras] = {"spearman": trd.spearman(lams, dels)}
        moyennes = {bras: {
            "delta_TEST_V2_moyenne_survivants": float(np.mean(list(
                transport["par_bras"][bras]
                ["delta_med_TEST_V2_50_graines_INFO"].values())))}
            for bras in BRAS_VERDICT}
    else:
        correlations = {"NON_MESURE": "TEST_V2 jamais ouvert (C7, I7)"}
        moyennes = {bras: {
            "Lambda_moyenne_survivants_INFO": float(np.mean(
                [duree["par_bras"][bras]["par_graine"][str(g)]["Lambda"]
                 for g in duree["par_bras"][bras]["survivants"]]))}
            for bras in BRAS_VERDICT}

    return {
        "tour": 67, "etape": "INFO",
        "etiquette": "INFO / hors-verdict — aucune de ces mesures n'altère un conjonct",
        "selecteur_g_star_INFO": {"regle": selecteur_info,
                                  "g_star_par_bras": g_stars},
        "Lambda_K_intermediaires": lambda_k,
        "transport_K_8_64": transport_k,
        "rho0_bras": rho0_pub,
        "best_return_step_g_star": brs,
        "correlation_Lambda_Delta": correlations,
        "moyennes_INFO": moyennes,
        "ratio_point_fixe": "publié par modèle dans TOUR67_DUREE.json "
                            "(point_fixe_ratio_INFO)",
        "variante_spectrale": "NON JOUÉE (déclarée §C9) : pénalité spectrale sur "
                              "rho(J0) — branche ouverte, aucun chiffre",
        "duree_s": time.perf_counter() - t_debut,
    }


#: Interprétations conservatrices CONSIGNÉES (soupape INFO — aucune n'altère
#: un seuil de l'émission ; chacune est déclarée dans TOUR67_DEPLOIEMENT.md) :
INTERPRETATIONS_DECLAREES: Tuple[str, ...] = (
    "« RESERVE_V2 jamais lue » = jamais MATÉRIALISÉE en données (aucun agrégat, "
    "aucun score, aucun vecteur) ; le parse structurel du fichier entier est "
    "requis par la découpe PAR INDEX DE CYCLE PARSÉ et le md5/composition "
    "d'opérateurs par tranche est exigé par §2.2/P1-c — gate structurelle : "
    "charger_tranche_v2 refuse toute tranche hors {VAL_V2, TEST_V2}.",
    "σ̂_E (C7) : estimateur gelé AVANT mesure = max des contrastes appariés "
    "disponibles à P1 sur VAL_V2 (3 paires inter-graines T64, 3 contrastes "
    "T64−ridge), ddof=0 — le max est la lecture défavorable.",
    "t_A0 de l'étalonnage := t_WD mesuré (même perte ℓ(2), seul le "
    "weight_decay de l'optimiseur diffère).",
    "R_disp^64 : numérateur ET dénominateur médians sur le MÊME sous-ensemble "
    "de paires exploitables (les deux trajectoires finies à t=64).",
    "Extinction (I-6) : médiane de ‖s_64‖ prise sur les lignes finies à t=64.",
    "Cas limite de partition : R_disp^64 N-A au stade 4 ⇒ classe NA_DISP, "
    "comptée NON BORNÉ-VIVANT (défavorable à l'hypothèse).",
    "Sélecteur « rang 25, convention basse » = élément d'indice (n−1)//2 de la "
    "liste triée (valeur, graine) — départage déterministe par graine.",
    "Rival WD « tient » si ≥ 1 graine BORNÉ-VIVANT (lecture favorable au "
    "rival) ; son g* = médiane basse parmi ses BORNÉ-VIVANT.",
    "γ* : oracle donné au rival = max de médiane Δ sur TEST_V2 parmi les "
    "rescale classés BORNÉ-VIVANT ; la table des 30 (classe, Λ, Δ) est "
    "publiée en entier.",
    "Concordance stride-4 : même signe de médiane que le test plein ET pas de "
    "bascule significative (p<0,01) en direction opposée.",
    "Le mur s'applique au CUMUL des étapes P2 (durées scellées des artefacts + "
    "étape en cours), vérifié avant chaque unité lourde.",
    "Le régime K=64 des poids T64 et du figé T63 est CITÉ (voie close) — "
    "l'étape duree ne les re-mesure pas.",
    "L'étape info relit TEST_V2 (déjà ouvert au verdict) pour les lectures "
    "INFO K∈{8,64} — précédent T64 (étape info post-porteur) ; si le "
    "transport est NON-MESURE (C7), TEST_V2 n'est JAMAIS ouvert et ces "
    "lectures tombent avec lui (I7).",
    "Seule la chute de Q12 ARRÊTE P1 (DISSIPATION I10) ; C6/C7 se consignent "
    "et appliquent leur partage de sort écrit (I6/I7) — le tour continue sur "
    "les conjoncts intacts.",
    "Sous transport NON-MESURE, le g* des lectures INFO est la médiane basse "
    "de best_delta_val (VALIDATION dataset_aba, K=2) parmi les BORNÉ-VIVANT.",
    "Repli instabilité (§4.4) : les descentes DIVERGED sont EXCLUES de toute "
    "mesure (classe EXCLU_DIVERGED_TRAIN publiée) ; C1/C2/σ̂/sélections sur "
    "les SURVIVANTS, effectif publié, k* recalculé Fraction à l'effectif "
    "réel ; le quorum C1 reste 40 en ABSOLU (défavorable).",
    "Incident consigné : le jeton β 1 (98ed38d1…) a été annulé et re-scellé "
    "(jeton 2) APRÈS l'étape train et AVANT toute mesure post-train, pour "
    "implémenter la clause de repli ci-dessus dans les étapes d'agrégation ; "
    "entrainer_bras est inchangé au bit (preuve : re-descente A1 seed 0 == "
    "sha256 du TOUR67_TRAIN.json scellé) — précédent T65 (jeton annulé "
    "déclaré, chronologie aux mtimes).",
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("etape", choices=[
        "etalonnage", "p0b", "p0c", "p1", "gel", "train", "duree",
        "sigma", "contraste", "transport", "info"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    torch.use_deterministic_algorithms(True)
    fn: Dict[str, Callable[[Path], dict]] = {
        "etalonnage": etape_etalonnage, "p0b": etape_p0b, "p0c": etape_p0c,
        "p1": etape_p1, "gel": etape_gel, "train": etape_train,
        "duree": etape_duree, "sigma": etape_sigma,
        "contraste": etape_contraste, "transport": etape_transport,
        "info": etape_info,
    }
    t0 = time.perf_counter()
    payload = fn[args.etape](root)
    dt = time.perf_counter() - t0
    out = Path(args.out) if args.out else root / ART[args.etape]
    sha = trd.ecrire_artefact(payload, out)
    print(f"{args.etape}: {out} sha256={sha} duree={dt:.3f}s")
    if args.etape in ("train", "duree", "sigma", "contraste", "transport"):
        print(f"sceau {args.etape}: {sceller(root, args.etape)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée mesure
    raise SystemExit(main())
