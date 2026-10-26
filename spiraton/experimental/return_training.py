"""return_training.py — T64 : chantier 5, l'apprentissage du retour.

Première boucle d'ENTRAÎNEMENT du projet sur du texte réel : un
``ChronoSpiraton(19)`` (équation du second ordre, ``experimental/chrono.py``
NON MODIFIÉ) est entraîné par descente de gradient sous la perte alpha-oméga
(CLAUDE.md chantier 5 : « A′ prédit doit être proche-et-aligné avec A sans lui
être identique — pénaliser la copie exacte ») sur les cycles ABA réels de
``dataset_aba.txt``, puis confronté à TROIS zéros sur un test scellé jamais vu :

* **identité K=0** — ``Δ_i = 0`` par construction, au bit (le zéro de la mesure) ;
* **Chrono figé** (recette T63, 12 graines, jamais entraîné) — repère d'échelle ;
* **ridge close-form** (le vrai rival linéaire, λ choisi sur la validation) ;
* plus la **garde constante** ``c̄`` (gain atteignable sans information d'entrée).

Grandeur de verdict (TOUR64_EMISSION.md §4.2), appariée par cycle :

    Δ_i(Φ) = score_retour(Φ(a_i), a′_i) − score_retour(a_i, a′_i)

``score_retour`` est le ``cos − l2`` de l'instrument M1 (T42, dépôt Tokenizer),
code INCHANGÉ, float64. La chaîne T63 est reprise terme à terme : segments
isolés → 33D (heuristic OFF) → cartouches S (pointe) / P (moyenne), tranche
MAIN {6..22, 31, 32} (19 dims, dims 0-5 EXCLUES — anti-circularité T48) →
``Φ_θ(a) = forward(s0=a, steps=2, s_prev=0)``. Le segment B n'entre pas.

Splits gelés a priori par plages de lignes 1-indexées de ``dataset_aba.txt``
(TRAIN 101-1000 ; VALIDATION 3001-3500 ; TEST DE VERDICT 3501-5001, une seule
ouverture, en dernier, sous garde matérielle). La route de chargement par plage
reproduit EXACTEMENT la logique d'``agreger_corpus`` (gate G0b : bit-à-bit,
100 % ou FAUX).

CLI à étapes MATÉRIELLES (chaque étape refuse de s'exécuter si l'artefact de
l'étape précédente est absent ; le jeton β est vérifié en tête de chaque étape
post-gel) : ``etalonnage`` / ``p0c`` / ``p1`` / ``gel`` / ``train`` /
``selection`` / ``porteur`` / ``cartouche-p`` / ``info``.

Interprétations conservatrices CONSIGNÉES (soupape INFO, aucune n'altère un
seuil de l'émission) — voir ``INTERPRETATIONS_DECLAREES`` en bas de module.

Déterministe : seeds fixés partout, float64, CPU,
``torch.use_deterministic_algorithms(True)`` dans le runner ; mêmes entrées ⇒
artefacts bit-à-bit identiques. Les verdicts vivent dans TOUR64_DEPLOIEMENT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
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
from .chrono import ChronoSpiraton

# --- Constantes gelées (TOUR64_EMISSION.md §4) --------------------------------

EPS = trd.EPS                    #: 1e-12 — le même objet que M1.
STATE_SIZE = trd.STATE_SIZE      #: 19 = tranche MAIN.
K_PORTEUR = trd.K_PORTEUR        #: K = 2 gelé, aucun balayage.

DATASET = "dataset_aba.txt"
DATASET_MD5 = "a0f9fad13db6db551a23b56b46f5748b"

#: Splits par plages de lignes 1-indexées, bornes incluses (§4.1). Le bloc
#: 1001-3000 (T26) et les lignes 1-100 (dataset-100) sont EXCLUS.
SPLITS: Dict[str, Tuple[int, int]] = {
    "TRAIN": (101, 1000),
    "VALIDATION": (3001, 3500),
    "TEST": (3501, 5001),
}

CARTOUCHES: Tuple[str, ...] = ("S", "P")

#: Grille d'entraînement CLOSE EN EXTENSION (§4.5) : 4 configs × 3 graines
#: × 2 cartouches = 24 runs.
INIT_SCALES: Tuple[float, ...] = (0.1, 0.6)
LRS: Tuple[float, ...] = (1e-3, 1e-2)
GRAINES: Tuple[int, ...] = (0, 1, 2)
BATCH = 64
EPOCHS_MAX = 200
PATIENCE = 20
CLIP_NORM = 1.0
LAMBDA_COPY = 1.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8

#: Baselines (§4.3). Figé = recette T63 exacte via ``trd.make_chrono``.
FIGE_SEEDS: Tuple[int, ...] = tuple(range(12))
RIDGE_LAMBDAS: Tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-2, 1.0)

#: Gardes anti-collapse (Q6/Q7) sur la VALIDATION.
PAIRES_DISP_SEED = 64
N_PAIRES_DISP = 200
R_DISP_MIN = 0.20
DBAR_MIN = 0.05

#: Seuils de recevabilité (§4.8).
SEUIL_C2_DELTA_S = 0.10
SEUIL_C5_DELTA_P = 0.05
SEUIL_P = 0.01
SEUIL_C6_FIGE = -5.0
SIGMA_DELTA_MAX = 1.1331         #: au-delà, C2 = NON-MESURE (publié avant test).
Z_SUM = 3.418                    #: z_{0.995} + z_{0.80} (dimensionnement §4.4).
SEUIL_PERTE_DESCENTE = 0.20      #: ≥ 20 % sur ≥ 20/24 runs sinon échec d'optim.
QUORUM_PERTE = 20
SEUIL_DIVERGENCES_REPLI = 6      #: ≥ 6/24 ⇒ repli bounded=True (grille entière).

#: Gate Q9 (parité de chaîne) : référents T42 re-mesurés au T63.
STD_SRAW_HC_P_T42 = 0.12990046111191347
STD_SRAW_TOL = 1e-9

#: Mur de budget (§4.9).
MUR_PLANCHER_S = 300.0
MUR_PLAFOND_S = 2400.0

#: Artefacts du tour (racine spiraton-enhanced, hors git).
ART = {
    "etalonnage": "TOUR64_ETALONNAGE.json",
    "p0c": "TOUR64_P0C.json",
    "p1": "TOUR64_P1.json",
    "splits": "TOUR64_SPLITS.json",
    "gel": "TOUR64_GEL.json",
    "spec": "TOUR64_SPEC.json",
    "predictions": "TOUR64_PREDICTIONS.json",
    "baselines": "TOUR64_BASELINES.json",
    "freeze": "TOUR64_FREEZE.json",
    "train": "TOUR64_TRAIN.json",
    "courbes": "TOUR64_COURBES.json",
    "selection": "TOUR64_SELECTION.json",
    "porteur": "TOUR64_PORTEUR.json",
    "deltas_test": "TOUR64_DELTAS_TEST.json",
    "cartouche_p": "TOUR64_CARTOUCHE_P.json",
    "info": "TOUR64_INFO.json",
}
POIDS_DIR = "TOUR64_POIDS"

#: Les 16 chemins du digest β (§4.11, liste close), relatifs à la racine.
FREEZE_PATHS: Tuple[str, ...] = (
    "TOUR64_EMISSION.md",
    "TOUR64_SPLITS.json",
    "TOUR64_SPEC.json",
    "TOUR64_PREDICTIONS.json",
    "TOUR64_BASELINES.json",
    "TOUR64_P1.json",
    "dataset_aba.txt",
    "corpus_horscanon_aba.txt",
    "corpus_f0b_aba.txt",
    "corpus_claude_aba.txt",
    "spiraton/spiraton/experimental/chrono.py",
    "spiraton/spiraton/diagnostics/textual_return_dynamics.py",
    "spiraton/spiraton/data/tokenizer_bridge.py",
    "Tokenizer/python/spiraton_tokenizer/alpha_omega_text.py",
    "spiraton/spiraton/experimental/return_training.py",
    "Tokenizer/bin/libspiratontokenizer.so",
)


def config_id(init_scale: float, lr: float) -> str:
    return f"is{init_scale}_lr{lr}"


def configs_grille() -> List[Dict[str, float]]:
    """La grille close : 4 configs (ordre gelé : init_scale puis lr)."""
    return [{"init_scale": s, "lr": lr} for s in INIT_SCALES for lr in LRS]


# --- Chargeur par plage de lignes (route G0b : parité agreger_corpus) ---------

@dataclass(frozen=True)
class SplitAgg:
    """Un split matérialisé : paires (a, a′) float64 par cartouche + provenance.

    ``md5_octets`` = md5 des octets BRUTS de la tranche de lignes ;
    ``md5_numeros`` = md5 de la liste (CSV ascii) des numéros de ligne RETENUS
    après parse et exclusions (F5, agrégat absent). Les deux entrent à P1/β.
    """

    nom: str
    plage: Tuple[int, int]
    n_lignes: int
    md5_octets: str
    n_parses: int
    n_f5: int
    formes: Tuple[str, ...]
    numeros: Tuple[int, ...]
    md5_numeros: str
    a: Dict[str, np.ndarray]
    ap: Dict[str, np.ndarray]
    b_present: Tuple[bool, ...]
    exclusions: Tuple[Dict[str, object], ...]

    @property
    def n(self) -> int:
        return len(self.formes)


def lignes_plage(path: Path, l1: int, l2: int) -> List[bytes]:
    """Octets BRUTS des lignes ``l1..l2`` (1-indexées, incluses, keepends)."""
    if l1 < 1 or l2 < l1:
        raise ValueError(f"plage invalide : ({l1}, {l2})")
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    if l2 > len(lines):
        raise ValueError(f"plage ({l1}, {l2}) hors fichier ({len(lines)} lignes)")
    return lines[l1 - 1:l2]


def cycles_plage(path: Path, l1: int, l2: int):
    """Cycles parsés de la plage — reproduit la logique de ``charger_cycles``.

    Retourne ``(cycles, numeros_ligne, n_lignes)``. Comme ``charger_cycles``
    (M1) : lignes vides / terminateurs / illisibles sautés SANS bruit.
    Décodage utf-8 ; les fins de ligne restent dans la chaîne (le parseur les
    ignore), la parité bit-à-bit est vérifiée par la gate G0b.
    """
    raw = lignes_plage(path, l1, l2)
    cycles, numeros = [], []
    for off, lb in enumerate(raw):
        line = lb.decode("utf-8")
        try:
            c = aba.try_parse_aba_line(line)
        except aba.AbaParseError:
            continue
        if c is not None:
            cycles.append(c)
            numeros.append(l1 + off)
    return cycles, numeros, len(raw)


def agreger_plage(tok, m1, path: Path, l1: int, l2: int):
    """Miroir exact d'``agreger_corpus`` sur une PLAGE : liste de ``CycleAgg``.

    Même boucle, mêmes exclusions F5, même convention d'index (position dans
    la liste des cycles parsés). Gate G0b : sur (1, 100) de ``dataset_aba.txt``
    et sur la plage pleine de hc-44, la sortie doit être BIT-À-BIT identique à
    ``m1.agreger_corpus`` (100 % ou FAUX).
    """
    cycles, numeros, _ = cycles_plage(path, l1, l2)
    out, nums_out = [], []
    for i, c in enumerate(cycles):
        forme = m1.forme_cycle(c, aba_forms)
        if forme == "F5":
            continue
        out.append(m1.CycleAgg(
            index=i,
            forme=forme,
            seg_a=m1.agreger_segment(tok, c.seg_a.text),
            seg_b=m1.agreger_segment(tok, c.seg_b.text),
            seg_ap=m1.agreger_segment(tok, c.seg_a_prime.text),
        ))
        nums_out.append(numeros[i])
    return out, nums_out, len(cycles)


def charger_split(tok, m1, root: Path, nom: str, plage: Tuple[int, int]) -> SplitAgg:
    """Matérialise un split (md5 DUR sur le corpus source) en paires S/P MAIN."""
    path = root / DATASET
    md5 = trd.md5_fichier(path)
    if md5 != DATASET_MD5:
        raise RuntimeError(f"md5 {DATASET} = {md5} != gelé {DATASET_MD5} — ARRÊT")
    l1, l2 = plage
    raw = b"".join(lignes_plage(path, l1, l2))
    cycleaggs, numeros, n_parses = agreger_plage(tok, m1, path, l1, l2)
    formes: List[str] = []
    nums_ret: List[int] = []
    b_present: List[bool] = []
    a: Dict[str, List[np.ndarray]] = {"S": [], "P": []}
    ap: Dict[str, List[np.ndarray]] = {"S": [], "P": []}
    exclusions: List[Dict[str, object]] = []
    for cm, num in zip(cycleaggs, numeros):
        keys = [("S", "MAIN"), ("P", "MAIN")]
        manquants = [k for k in keys
                     if k not in cm.seg_a.aggs or k not in cm.seg_ap.aggs]
        if manquants:
            exclusions.append({
                "ligne": num, "forme": cm.forme,
                "motif": "segment sans token (agrégat absent)",
                "cles_manquantes": [list(k) for k in manquants],
            })
            continue
        formes.append(cm.forme)
        nums_ret.append(num)
        b_present.append(("S", "MAIN") in cm.seg_b.aggs)
        for cart in CARTOUCHES:
            a[cart].append(np.asarray(cm.seg_a.aggs[(cart, "MAIN")], dtype=np.float64))
            ap[cart].append(np.asarray(cm.seg_ap.aggs[(cart, "MAIN")], dtype=np.float64))
    csv = ",".join(str(n) for n in nums_ret).encode("ascii")
    return SplitAgg(
        nom=nom, plage=plage, n_lignes=l2 - l1 + 1,
        md5_octets=hashlib.md5(raw).hexdigest(),
        n_parses=n_parses, n_f5=n_parses - len(cycleaggs),
        formes=tuple(formes), numeros=tuple(nums_ret),
        md5_numeros=hashlib.md5(csv).hexdigest(),
        a={c: np.stack(v) for c, v in a.items()},
        ap={c: np.stack(v) for c, v in ap.items()},
        b_present=tuple(b_present),
        exclusions=tuple(exclusions),
    )


def comparer_cycleaggs(lhs, rhs) -> Dict[str, object]:
    """Comparaison BIT-À-BIT de deux listes de ``CycleAgg`` (gate G0b).

    Substrat EXACT : 100 % ou FAUX (99 % = faux). Compare index, forme, et
    pour chaque segment : n_tokens, pointe_idx, drapeaux dégénérés, ensembles
    de clés d'agrégats et ÉGALITÉ BIT-À-BIT de chaque vecteur.
    """
    if len(lhs) != len(rhs):
        return {"identique": False, "motif": f"n {len(lhs)} != {len(rhs)}"}
    n_vecteurs = 0
    for k, (x, y) in enumerate(zip(lhs, rhs)):
        if x.index != y.index or x.forme != y.forme:
            return {"identique": False, "motif": f"cycle {k}: index/forme"}
        for seg_nom in ("seg_a", "seg_b", "seg_ap"):
            sx, sy = getattr(x, seg_nom), getattr(y, seg_nom)
            if (sx.n_tokens != sy.n_tokens or sx.pointe_idx != sy.pointe_idx
                    or sx.w_degenere != sy.w_degenere
                    or sx.alpha_degenere != sy.alpha_degenere):
                return {"identique": False, "motif": f"cycle {k}/{seg_nom}: meta"}
            if set(sx.aggs) != set(sy.aggs):
                return {"identique": False, "motif": f"cycle {k}/{seg_nom}: clés"}
            for key in sx.aggs:
                vx, vy = sx.aggs[key], sy.aggs[key]
                if vx.dtype != vy.dtype or not np.array_equal(vx, vy):
                    return {"identique": False,
                            "motif": f"cycle {k}/{seg_nom}/{key}: octets"}
                n_vecteurs += 1
    return {"identique": True, "n_cycles": len(lhs), "n_vecteurs_compares": n_vecteurs}


# --- Perte alpha-oméga (§4.5) et évaluation M1 --------------------------------

def phi_theta(model: ChronoSpiraton, a: torch.Tensor) -> torch.Tensor:
    """``Φ_θ(a) = forward(s0=a, steps=2, s_prev=0)`` — chaîne T63, K gelé."""
    return model(a, steps=K_PORTEUR)


def perte_alpha_omega(model: ChronoSpiraton, a: torch.Tensor, ap: torch.Tensor,
                      m: float) -> torch.Tensor:
    """La perte gelée : ``L(θ) = −mean s_i + λ_copy·mean relu(m − d_i)²``.

    ``s_i = cos_i − l2_i`` est la parité M1 EXACTE de
    ``score_retour(Φ_θ(a_i), a′_i)`` (l2 normalisé par ‖x_i‖ = ‖Φ_θ(a_i)‖,
    premier argument de M1). ``d_i = ‖x_i − a_i‖/(‖a_i‖+eps)`` est la distance
    de copie ; le second terme EST la pénalisation de la copie exacte du
    chantier 5 (m = 0,5 × médiane_TRAIN de la distance de retour réelle).
    La perte OPTIMISE, elle ne juge jamais (garde-fou (b)).
    """
    x = phi_theta(model, a)
    nx = torch.linalg.vector_norm(x, dim=-1)
    nap = torch.linalg.vector_norm(ap, dim=-1)
    cos = (x * ap).sum(dim=-1) / (nx * nap + EPS)
    l2 = torch.linalg.vector_norm(ap - x, dim=-1) / (nx + EPS)
    s = cos - l2
    d = torch.linalg.vector_norm(x - a, dim=-1) / (
        torch.linalg.vector_norm(a, dim=-1) + EPS)
    return -s.mean() + LAMBDA_COPY * torch.relu(m - d).pow(2).mean()


def mesurer_m(a: np.ndarray, ap: np.ndarray) -> float:
    """RÈGLE gelée : ``m = 0,5 × médiane(‖a′_i − a_i‖ / (‖a_i‖ + eps))`` (TRAIN)."""
    d = np.linalg.norm(ap - a, axis=1) / (np.linalg.norm(a, axis=1) + EPS)
    return 0.5 * float(np.median(d))


def scores_m1(m1, x: np.ndarray, ap: np.ndarray) -> np.ndarray:
    """``score_retour(x_i, a′_i)`` par cycle — code M1 INCHANGÉ (parité Q9)."""
    return np.asarray([m1.score_retour(x[i], ap[i]) for i in range(x.shape[0])],
                      dtype=np.float64)


def deltas(m1, x: np.ndarray, ap: np.ndarray, s_raw: np.ndarray) -> np.ndarray:
    """``Δ_i = score_retour(x_i, a′_i) − s_raw_i`` (l'identité y fait 0 exact)."""
    return scores_m1(m1, x, ap) - s_raw


def appliquer_chrono(model: ChronoSpiraton, a: np.ndarray) -> np.ndarray:
    """Φ_θ sur matrice numpy float64 (no_grad) — réutilise ``trd.phi_k_np``."""
    return trd.phi_k_np(model, a, K_PORTEUR)


# --- Gardes anti-collapse (Q6/Q7) ---------------------------------------------

def paires_dispersion(n: int) -> List[Tuple[int, int]]:
    """200 paires (i, j), i≠j, gelées sous ``default_rng(64)`` (Q6)."""
    rng = np.random.default_rng(PAIRES_DISP_SEED)
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < N_PAIRES_DISP:
        i, j = rng.integers(0, n, size=2)
        if i != j:
            pairs.append((int(i), int(j)))
    return pairs


def gardes_collapse(x: np.ndarray, a: np.ndarray,
                    pairs: Sequence[Tuple[int, int]]) -> Tuple[float, float]:
    """Retourne ``(R_disp, d̄)`` — Q6 (effondrement ≥ 5×) et Q7 (copie déguisée)."""
    ii = np.asarray([p[0] for p in pairs])
    jj = np.asarray([p[1] for p in pairs])
    num = float(np.median(np.linalg.norm(x[ii] - x[jj], axis=1)))
    den = float(np.median(np.linalg.norm(a[ii] - a[jj], axis=1)))
    r_disp = num / den if den > 0 else float("nan")
    dbar = float(np.median(np.linalg.norm(x - a, axis=1)
                           / (np.linalg.norm(a, axis=1) + EPS)))
    return r_disp, dbar


# --- Entraînement (spec §4.5, close en extension) ------------------------------

@dataclass
class RunResult:
    """Résultat d'UN run : courbes, meilleur état (restauré), diagnostics."""

    cartouche: str
    init_scale: float
    lr: float
    seed: int
    diverged: bool
    epochs_run: int
    best_epoch: int
    best_delta_val: float
    loss_init: float
    loss_min: float
    descente_rel: float
    early_stopped: bool
    courbes: Dict[str, List[float]]
    state: Dict[str, torch.Tensor]
    state_sha256: str


def hash_etat(state: Dict[str, torch.Tensor]) -> str:
    """sha256 canonique des poids : octets '<f8' des tenseurs, clés triées."""
    blob = b"".join(state[k].detach().cpu().numpy().astype("<f8").tobytes()
                    for k in sorted(state))
    return hashlib.sha256(blob).hexdigest()


def sauver_poids(state: Dict[str, torch.Tensor], path: Path) -> str:
    """Sérialisation CANONIQUE (déterministe) : header JSON + octets '<f8'.

    Format : ``u32 LE = longueur header`` puis header JSON (clés triées,
    formes) puis la concaténation des tenseurs en float64 little-endian, clés
    triées. Le sha256 du FICHIER est reproductible bit-à-bit (garde-fou (h)).
    """
    keys = sorted(state)
    header = json.dumps({k: list(state[k].shape) for k in keys},
                        sort_keys=True).encode("utf-8")
    blob = b"".join(state[k].detach().cpu().numpy().astype("<f8").tobytes()
                    for k in keys)
    payload = struct.pack("<I", len(header)) + header + blob
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def modele_depuis_poids(init_scale: float, state: Dict[str, torch.Tensor]) -> ChronoSpiraton:
    """Reconstruit un modèle depuis des poids gelés — init SEEDÉE (écrasée par
    ``load_state_dict`` ; le seed évite toute avance non maîtrisée du RNG)."""
    torch.manual_seed(0)
    model = ChronoSpiraton(state_size=STATE_SIZE, init_scale=init_scale,
                           bounded=False, c_outside=False).double()
    model.load_state_dict(state)
    return model


def charger_poids(path: Path) -> Dict[str, torch.Tensor]:
    """Relit une sérialisation canonique → state_dict float64."""
    payload = path.read_bytes()
    (hlen,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4:4 + hlen].decode("utf-8"))
    out: Dict[str, torch.Tensor] = {}
    off = 4 + hlen
    for k in sorted(header):
        shape = tuple(header[k])
        count = int(np.prod(shape)) if shape else 1
        arr = np.frombuffer(payload[off:off + 8 * count], dtype="<f8").reshape(shape)
        out[k] = torch.from_numpy(arr.copy())
        off += 8 * count
    return out


def train_run(m1, cartouche: str, init_scale: float, lr: float, seed: int,
              a_tr: np.ndarray, ap_tr: np.ndarray,
              a_val: np.ndarray, ap_val: np.ndarray,
              s_raw_val: np.ndarray, m: float,
              *, epochs_max: int = EPOCHS_MAX, bounded: bool = False) -> RunResult:
    """UN run de la grille : Adam, clip 1.0, batch 64, early-stop patience 20.

    * init : ``torch.manual_seed(seed)`` puis ``ChronoSpiraton(19, init_scale,
      bounded, c_outside=False).double()`` (recette ``trd.make_chrono``) ;
    * mélange : ``torch.randperm`` sous ``torch.Generator`` graine 1000+seed
      (générateur créé UNE fois, avance d'époque en époque) ;
    * early-stop : ``médiane Δ`` VALIDATION (évaluée via M1, code inchangé),
      amélioration = strictement supérieure ; patience 20 ; restauration du
      meilleur ``state_dict`` ;
    * divergence : perte non finie ⇒ ``diverged=True``, arrêt immédiat, run
      exclu de la sélection (compté et publié) ;
    * courbes par époque (§4.7, gelées d'avance) : perte train (pleine, après
      l'époque), médiane Δ validation, R_disp, d̄.
    """
    torch.manual_seed(seed)
    model = ChronoSpiraton(state_size=STATE_SIZE, init_scale=init_scale,
                           bounded=bounded, c_outside=False).double()
    gen = torch.Generator()
    gen.manual_seed(1000 + seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=ADAM_BETAS,
                           eps=ADAM_EPS, weight_decay=0.0)
    ta = torch.from_numpy(np.ascontiguousarray(a_tr))
    tap = torch.from_numpy(np.ascontiguousarray(ap_tr))
    n = ta.shape[0]
    pairs = paires_dispersion(a_val.shape[0])

    with torch.no_grad():
        loss_init = float(perte_alpha_omega(model, ta, tap, m).item())

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
        for k0 in range(0, n, BATCH):
            idx = perm[k0:k0 + BATCH]
            opt.zero_grad()
            loss = perte_alpha_omega(model, ta[idx], tap[idx], m)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
        if diverged:
            break
        epochs_run = epoch
        with torch.no_grad():
            lt = float(perte_alpha_omega(model, ta, tap, m).item())
        if not math.isfinite(lt):
            diverged = True
            break
        x_val = appliquer_chrono(model, a_val)
        if not np.isfinite(x_val).all():
            diverged = True
            break
        dv = deltas(m1, x_val, ap_val, s_raw_val)
        dmed = float(np.median(dv))
        r_disp, dbar = gardes_collapse(x_val, a_val, pairs)
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
            if sans_amelioration >= PATIENCE:
                early = True
                break

    if best_state is None:  # divergé avant la première évaluation
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_epoch = 0
        best_val = float("nan")
    loss_min = min(courbes["perte_train"]) if courbes["perte_train"] else float("nan")
    descente = ((loss_init - loss_min) / abs(loss_init)
                if courbes["perte_train"] and abs(loss_init) > 0 else float("nan"))
    return RunResult(
        cartouche=cartouche, init_scale=init_scale, lr=lr, seed=seed,
        diverged=diverged, epochs_run=epochs_run, best_epoch=best_epoch,
        best_delta_val=best_val, loss_init=loss_init, loss_min=loss_min,
        descente_rel=descente, early_stopped=early, courbes=courbes,
        state=best_state, state_sha256=hash_etat(best_state),
    )


# --- Baselines : ridge close-form et constante (§4.3) --------------------------

def ajuster_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """``(W, b) = argmin Σ‖W a_i + b − a′_i‖² + λ‖W‖²_F`` — lstsq augmentée.

    Résolution par ``np.linalg.lstsq`` sur la matrice augmentée ``[X, 1]``
    empilée avec ``√λ·[I, 0]`` (le biais n'est PAS pénalisé), float64,
    déterministe (seed N-A). Retourne θ (d+1, d) : ``pred = X·θ[:d] + θ[d]``.
    """
    n, d = x.shape
    xa = np.hstack([x, np.ones((n, 1), dtype=np.float64)])
    if lam > 0.0:
        reg = np.hstack([math.sqrt(lam) * np.eye(d), np.zeros((d, 1))])
        mat = np.vstack([xa, reg])
        cible = np.vstack([y, np.zeros((d, d), dtype=np.float64)])
    else:
        mat, cible = xa, y
    theta, *_ = np.linalg.lstsq(mat, cible, rcond=None)
    return theta


def predire_ridge(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ theta[:-1] + theta[-1]


def constante_cbar(ap_train: np.ndarray) -> np.ndarray:
    """``c̄`` = moyenne float64 des a′ du TRAIN (zéro information d'entrée)."""
    return ap_train.mean(axis=0)


# --- Statistique de verdict : sign-test exact, Wilcoxon (INFO) -----------------

def sign_test_exact(d: np.ndarray) -> Dict[str, object]:
    """Sign-test EXACT bilatéral sur ``signe(Δ_i)`` (binomiale ½, zéros écartés).

    Arithmétique entière exacte (``math.comb``, ``Fraction``) — aucun
    arrondi avant la conversion finale en float. Capacité dite d'avance :
    à n=1500, p<0,01 exige k ≥ 800 (déséquilibre ≥ 53,33 %).
    """
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    zeros = int((d == 0).sum())
    n = pos + neg
    if n == 0:
        return {"n": 0, "pos": 0, "neg": 0, "zeros": zeros, "p": 1.0}
    den = 1 << n
    cdf_le = sum(math.comb(n, i) for i in range(0, pos + 1))
    cdf_ge = sum(math.comb(n, i) for i in range(pos, n + 1))
    p = float(min(Fraction(1), 2 * Fraction(min(cdf_le, cdf_ge), den)))
    return {"n": n, "pos": pos, "neg": neg, "zeros": zeros, "p": p}


def wilcoxon_approx(d: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signé-rang, approximation normale avec correction d'ex-aequo.

    INFO SEULEMENT (jamais porteur — le porteur est le sign-test exact).
    """
    dnz = d[d != 0]
    n = int(dnz.size)
    if n < 10:
        return {"n": n, "W": float("nan"), "z": float("nan"), "p": float("nan")}
    r = trd.rangs_moyens(np.abs(dnz))
    w_pos = float(r[dnz > 0].sum())
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(dnz), return_counts=True)
    tie = float((counts ** 3 - counts).sum())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    z = (w_pos - mu) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"n": n, "W": w_pos, "z": z, "p": p}


# --- Gardes matérielles d'ordre ------------------------------------------------

def verifier_gel(root: Path) -> None:
    """Vérifie le jeton β T64 — dérive ⇒ ARRÊT (retouche = candidat mort)."""
    token = read_token(root / ART["freeze"])
    report = verify_freeze(token)
    if not report.ok:
        raise RuntimeError(f"gel β dérivé : {report.drifts} — ARRÊT")


def exiger(root: Path, *cles: str) -> None:
    """Garde matérielle : refuse si un artefact d'étape antérieure manque."""
    for cle in cles:
        if not (root / ART[cle]).is_file():
            raise RuntimeError(
                f"artefact {ART[cle]} absent : l'étape refuse de s'exécuter "
                f"(ordre MATÉRIEL du protocole)")


def sceller(root: Path, cle: str) -> str:
    """Écrit le sceau ``.sha256`` d'un artefact et retourne le sha256."""
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


# --- Collecte commune ----------------------------------------------------------

def _instrument():
    tok, m1 = trd.charger_instrument()
    return tok, m1


def _splits_train_val(tok, m1, root: Path) -> Tuple[SplitAgg, SplitAgg]:
    tr = charger_split(tok, m1, root, "TRAIN", SPLITS["TRAIN"])
    va = charger_split(tok, m1, root, "VALIDATION", SPLITS["VALIDATION"])
    return tr, va


def _split_test(tok, m1, root: Path) -> SplitAgg:
    """Le TEST ne se matérialise qu'aux étapes post-scellement (porteur & co)."""
    return charger_split(tok, m1, root, "TEST", SPLITS["TEST"])


# --- Étapes du protocole --------------------------------------------------------

def etape_p0c(root: Path) -> dict:
    """P0-c : UN run d'entraînement complet (S, is=0.1, lr=1e-3, graine 0).

    Preuve de déterminisme (à exécuter DEUX fois, process froids : state_dict
    identique au bit). Les poids de ce run sont JETÉS (proof-only, INFO) —
    le « premier poids » du verdict ne bouge qu'à P2, après le gel β.
    """
    tok, m1 = _instrument()
    tr, va = _splits_train_val(tok, m1, root)
    m = mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = scores_m1(m1, va.a["S"], va.ap["S"])
    rr = train_run(m1, "S", 0.1, 1e-3, 0, tr.a["S"], tr.ap["S"],
                   va.a["S"], va.ap["S"], s_raw_val, m)
    return {
        "tour": 64, "etape": "P0C",
        "config": {"cartouche": "S", "init_scale": 0.1, "lr": 1e-3, "seed": 0},
        "m_train_S": m,
        "state_sha256": rr.state_sha256,
        "best_epoch": rr.best_epoch,
        "best_delta_val": rr.best_delta_val,
        "epochs_run": rr.epochs_run,
        "early_stopped": rr.early_stopped,
        "diverged": rr.diverged,
        "loss_init": rr.loss_init,
        "loss_min": rr.loss_min,
        "courbes_md5": hashlib.md5(json.dumps(rr.courbes, sort_keys=True)
                                   .encode()).hexdigest(),
        "note": "poids JETÉS (preuve de déterminisme) — le verdict n'en dépend pas",
    }


def etape_etalonnage(root: Path) -> dict:
    """P1-e : étalonnage MESURÉ du mur (§4.9), extrapolation gelée AVEC le mur."""
    tok, m1 = _instrument()
    t0 = time.perf_counter()
    cycles = m1.charger_cycles(aba, str(root / DATASET), limit=None)[:20]
    for c in cycles:
        for seg in (c.seg_a, c.seg_b, c.seg_a_prime):
            m1.agreger_segment(tok, seg.text)
    t_tok20 = time.perf_counter() - t0

    tr, va = _splits_train_val(tok, m1, root)
    m = mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = scores_m1(m1, va.a["S"], va.ap["S"])

    t0 = time.perf_counter()
    rr = train_run(m1, "S", 0.1, 1e-3, 0, tr.a["S"], tr.ap["S"],
                   va.a["S"], va.ap["S"], s_raw_val, m, epochs_max=20)
    t_20ep = time.perf_counter() - t0
    t_ep = t_20ep / max(rr.epochs_run, 1)

    model = trd.make_chrono(0)
    t0 = time.perf_counter()
    x = appliquer_chrono(model, va.a["S"])
    _ = deltas(m1, x, va.ap["S"], s_raw_val)
    t_eval = time.perf_counter() - t0

    n_tok_total = (tr.n + va.n + (SPLITS["TEST"][1] - SPLITS["TEST"][0] + 1)
                   + 44 + 24 + 76)
    t_hat = 24 * EPOCHS_MAX * t_ep + 30 * t_eval + (n_tok_total / 20.0) * t_tok20
    mur = max(10 * t_hat, MUR_PLANCHER_S)
    return {
        "tour": 64, "etape": "ETALONNAGE",
        "t_tok20_s": t_tok20, "t_ep_s": t_ep, "t_eval_s": t_eval,
        "epochs_mesurees": rr.epochs_run,
        "T_hat_s": t_hat, "mur_s": mur, "plafond_s": MUR_PLAFOND_S,
        "note": "T_hat majore (early-stop non compté) ; mur = max(10·T_hat, 300), plafond dur 2400 s",
    }


def etape_p1(root: Path) -> dict:
    """P1 : (a) Q9 parité de chaîne ; (b) Q10 route de plage (bit-à-bit) ;
    (c) splits + md5 ×3 + effectifs + exclusions + min/max par dim + m +
    capacités/MDE + Δ_figé (TRAIN, INFO) ; (d) Q11 finitude/non-vacuité.
    AUCUN Δ n'est calculé sur le TEST ici (structure et md5 seulement, §4.1).
    Écrit AUSSI TOUR64_SPLITS.json (digest β).
    """
    tok, m1 = _instrument()

    # (a) Q9 — parité de chaîne : hc-44 MAIN via le code T63 INCHANGÉ.
    hc = trd.collecter_corpus(tok, m1, root, *trd.CORPUS_VERDICT[0])
    s_raw_p = trd.scores_bruts(m1, hc, "P")
    s_raw_s = trd.scores_bruts(m1, hc, "S")
    parite = {
        "P_intra": float(np.median(s_raw_p)),
        "P_inter": trd._mediane_paires_inter(m1, hc, "P"),
        "S_intra": float(np.median(s_raw_s)),
        "S_inter": trd._mediane_paires_inter(m1, hc, "S"),
    }
    deltas_q9 = {k: abs(parite[k] - trd.REFERENTS_T42[k]) for k in trd.REFERENTS_T42}
    std_p = float(s_raw_p.std())
    q9_pass = (all(v <= trd.PARITE_TOL for v in deltas_q9.values())
               and abs(std_p - STD_SRAW_HC_P_T42) <= STD_SRAW_TOL)

    # (b) Q10 — route de plage == agreger_corpus, BIT-À-BIT (100 % ou FAUX).
    q10 = {}
    hc_path = root / trd.CORPUS_VERDICT[0][1]
    n_lignes_hc = len(hc_path.read_bytes().splitlines())
    ref_hc = m1.agreger_corpus(tok, aba, aba_forms, str(hc_path), limit=None)
    mine_hc, _, _ = agreger_plage(tok, m1, hc_path, 1, n_lignes_hc)
    q10["hc-44_plage_pleine"] = comparer_cycleaggs(mine_hc, ref_hc)
    ds_path = root / DATASET
    ref_ds = m1.agreger_corpus(tok, aba, aba_forms, str(ds_path), limit=100)
    mine_ds, _, _ = agreger_plage(tok, m1, ds_path, 1, 100)
    q10["dataset_1-100"] = comparer_cycleaggs(mine_ds, ref_ds)
    q10_pass = all(v["identique"] for v in q10.values())

    # (c) Splits matérialisés (TEST : structure/md5 SEULEMENT, aucun score).
    splits = {nom: charger_split(tok, m1, root, nom, plage)
              for nom, plage in SPLITS.items()}
    splits_pub = {}
    for nom, sp in splits.items():
        splits_pub[nom] = {
            "plage": list(sp.plage), "n_lignes": sp.n_lignes,
            "md5_octets_bruts": sp.md5_octets,
            "md5_numeros_retenus": sp.md5_numeros,
            "n": sp.n, "n_parses": sp.n_parses, "n_exclus_F5": sp.n_f5,
            "exclusions_agregat": list(sp.exclusions),
            "formes": {f: sp.formes.count(f) for f in sorted(set(sp.formes))},
        }
    tr, va = splits["TRAIN"], splits["VALIDATION"]

    m_par_cart = {c: mesurer_m(tr.a[c], tr.ap[c]) for c in CARTOUCHES}
    minmax = {c: {"min": tr.a[c].min(axis=0).tolist(),
                  "max": tr.a[c].max(axis=0).tolist()} for c in CARTOUCHES}

    # Δ_figé sur TRAIN (INFO, lecture anticipée de C6 ; le C6 du verdict est
    # re-mesuré sur le TEST à l'étape info — interprétation consignée).
    s_raw_tr = {c: scores_m1(m1, tr.a[c], tr.ap[c]) for c in CARTOUCHES}
    fige_medians = []
    for seed in FIGE_SEEDS:
        model = trd.make_chrono(seed)
        x = appliquer_chrono(model, tr.a["S"])
        fige_medians.append(float(np.median(deltas(m1, x, tr.ap["S"], s_raw_tr["S"]))))
    delta_fige_train = float(np.median(fige_medians))

    # (d) Q11 — finitude 100 % + non-vacuité (std des Δ_i figé, validation).
    finitude = all(np.isfinite(sp.a[c]).all() and np.isfinite(sp.ap[c]).all()
                   for sp in splits.values() for c in CARTOUCHES)
    s_raw_va = {c: scores_m1(m1, va.a[c], va.ap[c]) for c in CARTOUCHES}
    finitude = finitude and all(np.isfinite(v).all() for v in s_raw_tr.values()) \
        and all(np.isfinite(v).all() for v in s_raw_va.values())
    x0 = appliquer_chrono(trd.make_chrono(0), va.a["S"])
    dv0 = deltas(m1, x0, va.ap["S"], s_raw_va["S"])
    q11_pass = bool(finitude and float(dv0.std()) > 0.0)

    n_test_attendu = splits["TEST"].n
    sigma_ref = {"borne_C2": SIGMA_DELTA_MAX,
                 "mde_a_n": Z_SUM / math.sqrt(n_test_attendu)}
    payload = {
        "tour": 64, "etape": "P1",
        "dataset_md5": DATASET_MD5,
        "Q9_parite": {
            "mesure": parite, "referents_T42": trd.REFERENTS_T42,
            "deltas": deltas_q9, "tolerance": trd.PARITE_TOL,
            "std_s_raw_hc_P": std_p, "referent_std": STD_SRAW_HC_P_T42,
            "tolerance_std": STD_SRAW_TOL, "PASS": bool(q9_pass),
        },
        "Q10_route_plage": {"cas": q10, "PASS": bool(q10_pass)},
        "splits": splits_pub,
        "m_par_cartouche": m_par_cart,
        "minmax_par_dim_TRAIN": minmax,
        "s_raw_stats": {
            "TRAIN": {c: {"mediane": float(np.median(s_raw_tr[c])),
                          "std": float(s_raw_tr[c].std())} for c in CARTOUCHES},
            "VALIDATION": {c: {"mediane": float(np.median(s_raw_va[c])),
                               "std": float(s_raw_va[c].std())} for c in CARTOUCHES},
            "TEST": "NON CALCULÉ à P1 (structure et md5 seulement — garde Q8)",
        },
        "delta_fige_TRAIN_INFO": {
            "par_graine": fige_medians, "mediane_graines": delta_fige_train,
            "prediction_C6": SEUIL_C6_FIGE,
        },
        "capacites": {
            "n_par_split": {nom: splits[nom].n for nom in SPLITS},
            "MDE_formule": "3.418 * sigma_Delta / sqrt(n_test)",
            "MDE_coeff_a_n_test": sigma_ref["mde_a_n"],
            "sigma_max_C2": SIGMA_DELTA_MAX,
            "sign_test": "p<0.01 bilatéral à n=1500 exige k>=800 (déséquilibre >=53.33%)",
        },
        "Q11_finitude_non_vacuite": {
            "finitude_100pct": bool(finitude),
            "std_delta_fige_seed0_validation": float(dv0.std()),
            "PASS": q11_pass,
        },
    }
    # TOUR64_SPLITS.json (plages + 3×3 md5 + n) — entre au digest β.
    trd.ecrire_artefact({
        "tour": 64, "artefact": "SPLITS",
        "dataset": DATASET, "dataset_md5": DATASET_MD5,
        "splits": {nom: {"plage": list(SPLITS[nom]),
                         "md5_octets_bruts": splits[nom].md5_octets,
                         "md5_numeros_retenus": splits[nom].md5_numeros,
                         "n": splits[nom].n} for nom in SPLITS},
    }, root / ART["splits"])
    return payload


def specs_gelees() -> Tuple[dict, dict, dict]:
    """Contenus de TOUR64_SPEC/PREDICTIONS/BASELINES.json (transcription §3-§4)."""
    spec = {
        "tour": 64, "artefact": "SPEC",
        "architecture": "ChronoSpiraton(state_size=19, bounded=False, c_outside=False).double() — chrono.py NON MODIFIÉ",
        "K": K_PORTEUR, "init_scales": list(INIT_SCALES), "lrs": list(LRS),
        "optimiseur": {"type": "Adam", "betas": list(ADAM_BETAS),
                       "eps": ADAM_EPS, "weight_decay": 0.0},
        "clip_grad_norm": CLIP_NORM, "batch": BATCH,
        "epochs_max": EPOCHS_MAX, "patience": PATIENCE,
        "graines": list(GRAINES), "melange": "torch.Generator graine 1000+seed, randperm par époque",
        "cartouches": list(CARTOUCHES), "runs_totaux": 24,
        "perte": "L(θ) = −mean_i s_i + λ_copy·mean_i relu(m − d_i)² ; s_i = cos−l2 parité M1 ; d_i = ‖x_i−a_i‖/(‖a_i‖+eps)",
        "lambda_copy": LAMBDA_COPY, "eps": EPS,
        "m_regle": "0.5 × médiane_TRAIN(‖a′_i − a_i‖/(‖a_i‖+eps)) par cartouche — VALEURS dans TOUR64_P1.json",
        "early_stop": "médiane Δ validation (évaluateur M1 inchangé), amélioration stricte, patience 20, restauration du meilleur state_dict",
        "selection": "par cartouche : config maximisant la moyenne sur 3 graines du meilleur médiane Δ validation ; porteur test = médiane des 3 graines",
        "repli_instabilite": f">= {SEUIL_DIVERGENCES_REPLI}/24 divergences => grille entière rejouée bounded=True (étiquetée repli)",
        "precision": "float64, CPU, torch.manual_seed, use_deterministic_algorithms(True)",
    }
    predictions = {
        "tour": 64, "artefact": "PREDICTIONS",
        "H64": "Δ_S >= +0.10 sur test scellé, sign-test exact p < 0.01, et Chrono > ridge (médiane appariée > 0, p < 0.01)",
        "delta_S_seuil_C2": SEUIL_C2_DELTA_S, "delta_P_seuil_C5": SEUIL_C5_DELTA_P,
        "p_seuil": SEUIL_P, "sigma_delta_max_C2": SIGMA_DELTA_MAX,
        "conjoncts": {
            "C1": "PORTEUR — médiane Δ_S > 0 ET sign-test exact bilatéral p < 0.01",
            "C2": "PORTEUR — Δ_S >= +0.10 (NON-MESURE si σ̂_Δ > 1.1331, publié avant test)",
            "C3": "PORTEUR — médiane_i D_i > 0 ET sign-test p < 0.01 (Chrono − ridge, apparié)",
            "C4": "GATE — médiane_i [s(Φ_θ a_i, a′_i) − s(c̄, a′_i)] > 0, sign-test p < 0.01 ; chute => C1/C2 NON OPPOSABLES",
            "C5": "INFO — Δ_P >= +0.05 ET Δ_P < Δ_S",
            "C6": "INFO — Δ_figé <= −5 (échec = défaut de chaîne, à publier)",
        },
        "prediction_const": "Δ_const > 0 (la garde n'est pas décorative)",
        "estimation_a_priori": "apprenable ~55/45 contre point fixe (dit avant)",
        "perte_descente": f">= {SEUIL_PERTE_DESCENTE:.0%} sur >= {QUORUM_PERTE}/24 runs sinon échec d'optimisation",
        "issues": {
            "I1": "C1✔C2✔C3✔C4✔ — H64 confirmée (PROGRESSION)",
            "I2": "C1✔C2✔C3✘ — la ridge domine (PROGRESSION, info d'architecture)",
            "I3": "C1✔C2✘ — effet réel sous +0.10 (PROGRESSION)",
            "I4": "C1✘ + perte descendue — point fixe au grain agrégat (PROGRESSION si tout publié)",
            "I4b": "C1✘ + perte non descendue — échec d'optimisation, NON-MESURE",
            "I5": "C4✘ — victoires VIDES", "I6": "Q6/Q7 FAIL — collapse, NON-MESURE",
            "I7": ">=6/24 divergences et repli diverge — NON-MESURE",
            "I8": "Q9/Q10 FAIL — NON-MESURE", "I9": "mur dépassé — NON-MESURE",
            "I10": ".so non chargeable — tour non joué", "I11": "P3 FAIL — DISSIPATION",
        },
    }
    baselines = {
        "tour": 64, "artefact": "BASELINES",
        "identite": "Φ = id (K=0), Δ_i = 0 par construction AU BIT (test P0-b) — le zéro de la mesure",
        "fige": {"recette": "trd.make_chrono(seed) : init_scale=0.1, bounded=False, c_outside=False, .double(), K=2",
                 "graines": list(FIGE_SEEDS), "prediction": "Δ_figé <= −5 ; le battre ne compte pour RIEN"},
        "ridge": {"forme": "argmin Σ‖W a_i + b − a′_i‖² + λ‖W‖²_F, lstsq augmentée, float64, TRAIN seul",
                  "lambdas": list(RIDGE_LAMBDAS),
                  "selection": "λ maximisant médiane Δ sur VALIDATION"},
        "constante": "c̄ = moyenne float64 des a′ du TRAIN ; garde C4 (gain sans information d'entrée)",
        "anti_survente": "seules les victoires contre identité (C1/C2) et ridge (C3) comptent",
    }
    return spec, predictions, baselines


def etape_gel(root: Path) -> dict:
    """β : écrit SPEC/PREDICTIONS/BASELINES puis le jeton de gel (16 chemins).

    Le jeton est scellé AVANT que le premier poids du VERDICT ne bouge
    (l'étape ``train`` le vérifie en tête).
    """
    exiger(root, "p1", "splits")
    spec, predictions, baselines = specs_gelees()
    trd.ecrire_artefact(spec, root / ART["spec"])
    trd.ecrire_artefact(predictions, root / ART["predictions"])
    trd.ecrire_artefact(baselines, root / ART["baselines"])
    token = freeze_token([root / p for p in FREEZE_PATHS])
    write_token(token, root / ART["freeze"])
    return {"tour": 64, "etape": "GEL", "digest": token.digest,
            "n_chemins": len(token.artifacts),
            "mtimes_ns": {a.path: a.mtime_ns for a in token.artifacts}}


def etape_train(root: Path) -> dict:
    """P2-train : les 24 runs de la grille close ; courbes + poids + gardes."""
    exiger(root, "p1", "freeze", "etalonnage")
    verifier_gel(root)
    etal = json.loads((root / ART["etalonnage"]).read_text(encoding="utf-8"))
    mur = min(max(float(etal["mur_s"]), MUR_PLANCHER_S), MUR_PLAFOND_S)
    t_debut = time.perf_counter()

    tok, m1 = _instrument()
    tr, va = _splits_train_val(tok, m1, root)
    poids_dir = root / POIDS_DIR
    poids_dir.mkdir(exist_ok=True)

    runs_pub: List[dict] = []
    courbes_pub: Dict[str, dict] = {}
    n_diverges = 0
    mur_depasse = False
    for cart in CARTOUCHES:
        m = mesurer_m(tr.a[cart], tr.ap[cart])
        s_raw_val = scores_m1(m1, va.a[cart], va.ap[cart])
        for cfg in configs_grille():
            for seed in GRAINES:
                if time.perf_counter() - t_debut > mur:
                    mur_depasse = True
                    break
                rr = train_run(m1, cart, cfg["init_scale"], cfg["lr"], seed,
                               tr.a[cart], tr.ap[cart], va.a[cart], va.ap[cart],
                               s_raw_val, m)
                rid = f"{cart}_{config_id(cfg['init_scale'], cfg['lr'])}_seed{seed}"
                fichier = poids_dir / f"{rid}.bin"
                sha_fichier = sauver_poids(rr.state, fichier)
                if rr.diverged:
                    n_diverges += 1
                runs_pub.append({
                    "run_id": rid, "cartouche": cart,
                    "init_scale": cfg["init_scale"], "lr": cfg["lr"], "seed": seed,
                    "m": m, "diverged": rr.diverged,
                    "epochs_run": rr.epochs_run, "best_epoch": rr.best_epoch,
                    "early_stopped": rr.early_stopped,
                    "best_delta_val": rr.best_delta_val,
                    "loss_init": rr.loss_init, "loss_min": rr.loss_min,
                    "descente_rel": rr.descente_rel,
                    "descente_ge_20pct": bool(math.isfinite(rr.descente_rel)
                                              and rr.descente_rel >= SEUIL_PERTE_DESCENTE),
                    "r_disp_final": rr.courbes["r_disp"][-1] if rr.courbes["r_disp"] else float("nan"),
                    "dbar_final": rr.courbes["dbar"][-1] if rr.courbes["dbar"] else float("nan"),
                    "state_sha256": rr.state_sha256,
                    "poids_fichier": fichier.name, "poids_sha256": sha_fichier,
                })
                courbes_pub[rid] = rr.courbes
            if mur_depasse:
                break
        if mur_depasse:
            break
    n_descendus = sum(1 for r in runs_pub if r["descente_ge_20pct"])
    trd.ecrire_artefact({"tour": 64, "artefact": "COURBES",
                         "axes": ["perte_train", "delta_med_val", "r_disp", "dbar"],
                         "par_run": courbes_pub}, root / ART["courbes"])
    return {
        "tour": 64, "etape": "TRAIN",
        "mur_s": mur, "duree_s": time.perf_counter() - t_debut,
        "mur_depasse": mur_depasse,
        "n_runs": len(runs_pub), "n_diverges": n_diverges,
        "seuil_repli": SEUIL_DIVERGENCES_REPLI,
        "repli_bounded_requis": bool(n_diverges >= SEUIL_DIVERGENCES_REPLI),
        "perte_descendue_ge_20pct": {"n": n_descendus, "quorum": QUORUM_PERTE,
                                     "atteint": bool(n_descendus >= QUORUM_PERTE)},
        "runs": runs_pub,
    }


def _meilleure_config(runs: List[dict], cart: str) -> Tuple[dict, List[dict]]:
    """Sélection gelée : config maximisant la moyenne sur graines survivantes
    du meilleur ``médiane Δ`` de validation (DIVERGED exclus, comptés)."""
    meilleurs: List[Tuple[float, dict, List[dict]]] = []
    for cfg in configs_grille():
        rs = [r for r in runs if r["cartouche"] == cart
              and r["init_scale"] == cfg["init_scale"] and r["lr"] == cfg["lr"]]
        vivants = [r for r in rs if not r["diverged"]]
        if not vivants:
            continue
        moy = float(np.mean([r["best_delta_val"] for r in vivants]))
        meilleurs.append((moy, cfg, vivants))
    if not meilleurs:
        raise RuntimeError("toutes les configs ont divergé — issue I7 à déclarer")
    meilleurs.sort(key=lambda t: t[0], reverse=True)
    _, cfg, vivants = meilleurs[0]
    return cfg, vivants


def etape_selection(root: Path) -> dict:
    """P2-selection : configs retenues, σ̂_Δ, MDE réalisé, gardes Q6/Q7 sur les
    runs retenus, λ_ridge (validation), c̄ (train). Scellé AVANT le test."""
    exiger(root, "train", "courbes", "freeze")
    verifier_gel(root)
    train_art = json.loads((root / ART["train"]).read_text(encoding="utf-8"))
    runs = train_art["runs"]
    tok, m1 = _instrument()
    tr, va = _splits_train_val(tok, m1, root)
    poids_dir = root / POIDS_DIR

    selection: Dict[str, dict] = {}
    for cart in CARTOUCHES:
        cfg, vivants = _meilleure_config(runs, cart)
        s_raw_val = scores_m1(m1, va.a[cart], va.ap[cart])
        pairs = paires_dispersion(va.a[cart].shape[0])
        par_graine = []
        for r in sorted(vivants, key=lambda r: r["seed"]):
            state = charger_poids(poids_dir / r["poids_fichier"])
            model = modele_depuis_poids(cfg["init_scale"], state)
            x = appliquer_chrono(model, va.a[cart])
            dv = deltas(m1, x, va.ap[cart], s_raw_val)
            r_disp, dbar = gardes_collapse(x, va.a[cart], pairs)
            par_graine.append({
                "seed": r["seed"], "poids_fichier": r["poids_fichier"],
                "poids_sha256": r["poids_sha256"],
                "delta_med_val": float(np.median(dv)),
                "sigma_delta_val": float(dv.std()),
                "r_disp": r_disp, "dbar": dbar,
                "Q6_pass": bool(r_disp >= R_DISP_MIN),
                "Q7_pass": bool(dbar >= DBAR_MIN),
            })
        sigmas = [g["sigma_delta_val"] for g in par_graine]
        sigma_med = float(np.median(sigmas))
        n_test = SPLITS["TEST"][1] - SPLITS["TEST"][0] + 1  # borne sup ; n réel gelé à P1
        p1 = json.loads((root / ART["p1"]).read_text(encoding="utf-8"))
        n_test = int(p1["splits"]["TEST"]["n"])
        mde = Z_SUM * sigma_med / math.sqrt(n_test)
        selection[cart] = {
            "config": cfg, "config_id": config_id(cfg["init_scale"], cfg["lr"]),
            "n_graines_vivantes": len(par_graine),
            "par_graine": par_graine,
            "sigma_delta_val_par_graine": sigmas,
            "sigma_delta_val_mediane": sigma_med,
            "sigma_delta_val_max": float(max(sigmas)),
            "MDE_realise_mediane": mde,
            "MDE_realise_max": Z_SUM * float(max(sigmas)) / math.sqrt(n_test),
            "C2_sous_dimensionne": bool(sigma_med > SIGMA_DELTA_MAX),
            "gardes_Q6_Q7_PASS": bool(all(g["Q6_pass"] and g["Q7_pass"]
                                          for g in par_graine)),
        }

    # Ridge : fit TRAIN par λ, sélection sur VALIDATION (cartouche S porteuse ;
    # P ajusté aussi, INFO). c̄ par cartouche.
    ridge_pub: Dict[str, dict] = {}
    for cart in CARTOUCHES:
        s_raw_val = scores_m1(m1, va.a[cart], va.ap[cart])
        essais = {}
        for lam in RIDGE_LAMBDAS:
            theta = ajuster_ridge(tr.a[cart], tr.ap[cart], lam)
            dv = deltas(m1, predire_ridge(theta, va.a[cart]), va.ap[cart], s_raw_val)
            essais[str(lam)] = float(np.median(dv))
        lam_star = max(RIDGE_LAMBDAS, key=lambda lam: essais[str(lam)])
        ridge_pub[cart] = {"medianes_val_par_lambda": essais,
                           "lambda_retenu": lam_star}
    cbar = {c: constante_cbar(tr.ap[c]).tolist() for c in CARTOUCHES}
    return {
        "tour": 64, "etape": "SELECTION",
        "selection_par_cartouche": selection,
        "ridge": ridge_pub,
        "constante_cbar": cbar,
        "sigma_delta_max_C2": SIGMA_DELTA_MAX,
        "note": "σ̂_Δ et MDE réalisés publiés AVANT l'ouverture du test (§4.4)",
    }


def _evaluer_test_cartouche(root: Path, cart: str):
    """Charge le TEST (post-scellement seulement) + tout le nécessaire."""
    verifier_sceau(root, "selection")
    sel = json.loads((root / ART["selection"]).read_text(encoding="utf-8"))
    tok, m1 = _instrument()
    tr, _ = _splits_train_val(tok, m1, root)
    te = _split_test(tok, m1, root)
    p1 = json.loads((root / ART["p1"]).read_text(encoding="utf-8"))
    attendu = p1["splits"]["TEST"]
    if (te.md5_octets != attendu["md5_octets_bruts"]
            or te.md5_numeros != attendu["md5_numeros_retenus"]
            or te.n != attendu["n"]):
        raise RuntimeError("split TEST != gel P1 (md5/n) — ARRÊT")
    s_raw_te = scores_m1(m1, te.a[cart], te.ap[cart])
    sc = sel["selection_par_cartouche"][cart]
    cfg = sc["config"]
    par_graine = []
    for g in sc["par_graine"]:
        state = charger_poids(root / POIDS_DIR / g["poids_fichier"])
        model = modele_depuis_poids(cfg["init_scale"], state)
        dv = deltas(m1, appliquer_chrono(model, te.a[cart]), te.ap[cart], s_raw_te)
        par_graine.append((g["seed"], dv))
    return m1, tr, te, s_raw_te, sel, sc, par_graine


def etape_porteur(root: Path) -> dict:
    """P2 — OUVERTURE UNIQUE DU TEST, PORTEUR SEUL (C1-C4, cartouche S)."""
    exiger(root, "selection", "freeze")
    verifier_gel(root)
    m1, tr, te, s_raw_te, sel, sc, par_graine = _evaluer_test_cartouche(root, "S")

    med_par_graine = {str(seed): float(np.median(dv)) for seed, dv in par_graine}
    meds = sorted(med_par_graine.values())
    delta_s = float(np.median(list(med_par_graine.values())))
    # graine médiane = celle dont la médiane est la médiane des 3 (impair).
    seed_median = min((seed for seed, dv in par_graine
                       if float(np.median(dv)) == np.median(list(med_par_graine.values()))),
                      default=par_graine[0][0]) if len(par_graine) % 2 == 1 else par_graine[0][0]
    dv_med = dict(par_graine)[seed_median]

    # C1 — existence : médiane > 0 ET sign-test exact p < 0.01 (graine médiane ;
    # les 3 graines publiées).
    c1_par_graine = {str(seed): sign_test_exact(dv) for seed, dv in par_graine}
    c1_test = c1_par_graine[str(seed_median)]
    c1 = bool(delta_s > 0 and float(np.median(dv_med)) > 0 and c1_test["p"] < SEUIL_P)
    # C2 — magnitude.
    c2 = bool(delta_s >= SEUIL_C2_DELTA_S)
    c2_non_mesure = bool(sel["selection_par_cartouche"]["S"]["C2_sous_dimensionne"])
    # C3 — rival ridge (appariement D_i, graine médiane ; publié par graine).
    lam = sel["ridge"]["S"]["lambda_retenu"]
    theta = ajuster_ridge(tr.a["S"], tr.ap["S"], float(lam))
    s_ridge = scores_m1(m1, predire_ridge(theta, te.a["S"]), te.ap["S"])
    delta_ridge = s_ridge - s_raw_te
    d_par_graine = {str(seed): dv - delta_ridge for seed, dv in par_graine}
    c3_par_graine = {s: sign_test_exact(d) for s, d in d_par_graine.items()}
    d_med = d_par_graine[str(seed_median)]
    c3 = bool(float(np.median(d_med)) > 0 and c3_par_graine[str(seed_median)]["p"] < SEUIL_P)
    # C4 — garde constante (appariée).
    cbar = np.asarray(sel["constante_cbar"]["S"], dtype=np.float64)
    s_const = scores_m1(m1, np.tile(cbar, (te.n, 1)), te.ap["S"])
    delta_const = s_const - s_raw_te
    e_par_graine = {str(seed): dv - delta_const for seed, dv in par_graine}
    c4_par_graine = {s: sign_test_exact(e) for s, e in e_par_graine.items()}
    e_med = e_par_graine[str(seed_median)]
    c4 = bool(float(np.median(e_med)) > 0 and c4_par_graine[str(seed_median)]["p"] < SEUIL_P)

    trd.ecrire_artefact({
        "tour": 64, "artefact": "DELTAS_TEST",
        "lignes": list(te.numeros),
        "s_raw": s_raw_te.tolist(),
        "delta_chrono_graine_mediane": dv_med.tolist(),
        "delta_ridge": delta_ridge.tolist(),
        "delta_const": delta_const.tolist(),
    }, root / ART["deltas_test"])
    return {
        "tour": 64, "etape": "PORTEUR",
        "grandeur": "Δ_S = médiane_i [s(Φ_θ(a_i), a′_i) − s(a_i, a′_i)], test scellé, cartouche S, MAIN, K=2, médiane des graines",
        "n_test": te.n,
        "config_retenue": sc["config"],
        "delta_med_par_graine": med_par_graine,
        "graine_mediane": seed_median,
        "Delta_S": delta_s,
        "delta_med_3graines_triees": meds,
        "s_raw_test": {"mediane": float(np.median(s_raw_te)), "std": float(s_raw_te.std())},
        "C1": {"verdict": c1, "sign_test_par_graine": c1_par_graine,
               "seuil_p": SEUIL_P},
        "C2": {"verdict": c2, "seuil": SEUIL_C2_DELTA_S,
               "NON_MESURE_si_chute": c2_non_mesure},
        "C3": {"verdict": c3, "lambda_ridge": lam,
               "delta_ridge_mediane": float(np.median(delta_ridge)),
               "D_mediane_par_graine": {s: float(np.median(d)) for s, d in d_par_graine.items()},
               "sign_test_par_graine": c3_par_graine},
        "C4": {"verdict": c4,
               "delta_const_mediane": float(np.median(delta_const)),
               "E_mediane_par_graine": {s: float(np.median(e)) for s, e in e_par_graine.items()},
               "sign_test_par_graine": c4_par_graine},
        "gardes_Q6_Q7_selection_PASS": sc["gardes_Q6_Q7_PASS"],
    }


def etape_cartouche_p(root: Path) -> dict:
    """P2 — cartouche P (C5, INFO), APRÈS le porteur scellé."""
    exiger(root, "porteur", "freeze")
    verifier_gel(root)
    verifier_sceau(root, "porteur")
    m1, tr, te, s_raw_te, sel, sc, par_graine = _evaluer_test_cartouche(root, "P")
    med_par_graine = {str(seed): float(np.median(dv)) for seed, dv in par_graine}
    delta_p = float(np.median(list(med_par_graine.values())))
    porteur = json.loads((root / ART["porteur"]).read_text(encoding="utf-8"))
    delta_s = porteur["Delta_S"]
    return {
        "tour": 64, "etape": "CARTOUCHE_P",
        "config_retenue": sc["config"],
        "delta_med_par_graine": med_par_graine,
        "Delta_P": delta_p,
        "C5": {"seuil": SEUIL_C5_DELTA_P,
               "delta_P_ge_seuil": bool(delta_p >= SEUIL_C5_DELTA_P),
               "delta_P_lt_delta_S": bool(delta_p < delta_s),
               "verdict_INFO": bool(delta_p >= SEUIL_C5_DELTA_P and delta_p < delta_s)},
    }


def etape_info(root: Path) -> dict:
    """P2 — INFO (en dernier) : Δ_figé test (C6), moyennes, Wilcoxon,
    hors-distribution hc-44 / f0b-24 / claude-76 (post-hoc déclaré)."""
    exiger(root, "porteur", "cartouche_p", "freeze")
    verifier_gel(root)
    m1, tr, te, s_raw_te, sel, sc, par_graine = _evaluer_test_cartouche(root, "S")

    # C6 — Δ_figé sur le TEST (12 graines, jamais entraîné).
    fige = []
    for seed in FIGE_SEEDS:
        x = appliquer_chrono(trd.make_chrono(seed), te.a["S"])
        fige.append(float(np.median(deltas(m1, x, te.ap["S"], s_raw_te))))
    delta_fige = float(np.median(fige))

    # Moyennes (INFO, clause agrégat-MOYENNE T40) + Wilcoxon (INFO).
    seed_med = json.loads((root / ART["porteur"]).read_text(encoding="utf-8"))["graine_mediane"]
    dv = dict(par_graine)[seed_med]
    stats_info = {
        "delta_moyenne_graine_mediane": float(dv.mean()),
        "wilcoxon_approx": wilcoxon_approx(dv),
    }

    # Hors-distribution : le Chrono S retenu appliqué à hc-44 / f0b-24 /
    # claude-76 (brûlé T63 => post-hoc déclaré). INFO, jamais porteur.
    tok, _ = _instrument()
    cfg = sc["config"]
    hd = {}
    modeles = []
    for g in sc["par_graine"]:
        state = charger_poids(root / POIDS_DIR / g["poids_fichier"])
        modeles.append((g["seed"], modele_depuis_poids(cfg["init_scale"], state)))
    corpus_info = list(trd.CORPUS_VERDICT[:2]) + [trd.CORPUS_VIERGE]
    for nom, fichier, limit, md5 in corpus_info:
        pc = trd.collecter_corpus(tok, m1, root, nom, fichier, limit, md5)
        s_raw = scores_m1(m1, pc.a["S"], pc.ap["S"])
        meds = {str(seed): float(np.median(deltas(m1, appliquer_chrono(model, pc.a["S"]),
                                                  pc.ap["S"], s_raw)))
                for seed, model in modeles}
        hd[nom] = {"n": pc.n, "delta_med_par_graine": meds,
                   "delta_med_mediane": float(np.median(list(meds.values())))}
    return {
        "tour": 64, "etape": "INFO",
        "C6_delta_fige_test": {"par_graine": fige, "mediane": delta_fige,
                               "seuil": SEUIL_C6_FIGE,
                               "verdict_INFO": bool(delta_fige <= SEUIL_C6_FIGE)},
        "stats_INFO": stats_info,
        "hors_distribution_INFO": hd,
        "note": "claude-76 brûlé comme vierge au T63 — post-hoc déclaré ; hc-44/f0b-24 = épreuve de forme non vue, INFO",
    }


#: Interprétations conservatrices CONSIGNÉES (soupape INFO — aucune ne touche
#: un seuil de l'émission ; chacune est déclarée dans TOUR64_DEPLOIEMENT.md) :
INTERPRETATIONS_DECLAREES: Tuple[str, ...] = (
    "P0-c et l'étalonnage entraînent des poids AVANT β : poids JETÉS, preuve de "
    "déterminisme/mur seulement — « le premier poids ne bouge » s'entend des 24 "
    "descentes du verdict (l'émission ordonne elle-même P0-c avant β).",
    "P1 matérialise la STRUCTURE du TEST (md5, n, exclusions — exigés §4.1) mais "
    "n'y calcule AUCUN score ni Δ ; la garde Q8 porte sur les grandeurs.",
    "Δ_figé à P1 est mesuré sur le TRAIN (INFO) ; le C6 opposable est re-mesuré "
    "sur le TEST à l'étape info.",
    "Les sign-tests C1/C3/C4 sont rendus sur la GRAINE MÉDIANE (celle dont la "
    "médiane test est la médiane des 3) ; les 3 graines sont publiées côte à côte.",
    "σ̂_Δ retenu pour la règle C2 = MÉDIANE des 3 graines (max publié aussi).",
    "Amélioration early-stop = STRICTEMENT supérieure ; le générateur de mélange "
    "est créé une fois (graine 1000+seed) et avance d'époque en époque.",
    "descente de perte = (loss_init − loss_min)/|loss_init|, loss_init évaluée "
    "avant le premier pas (perte pleine TRAIN).",
    "m est mesuré PAR CARTOUCHE sur le TRAIN (l'espace d'agrégat diffère).",
    "Q6/Q7 sont mesurées sur les 24 runs (courbes) ; la GATE opposable porte sur "
    "les runs de la config RETENUE (la sélection n'est pas filtrée par les gardes).",
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("etape", choices=[
        "etalonnage", "p0c", "p1", "gel", "train", "selection",
        "porteur", "cartouche-p", "info"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    torch.use_deterministic_algorithms(True)
    fn: Dict[str, Callable[[Path], dict]] = {
        "etalonnage": etape_etalonnage, "p0c": etape_p0c, "p1": etape_p1,
        "gel": etape_gel, "train": etape_train, "selection": etape_selection,
        "porteur": etape_porteur, "cartouche-p": etape_cartouche_p,
        "info": etape_info,
    }
    t0 = time.perf_counter()
    payload = fn[args.etape](root)
    dt = time.perf_counter() - t0
    out = Path(args.out) if args.out else root / ART[
        args.etape.replace("-", "_") if args.etape != "cartouche-p" else "cartouche_p"]
    sha = trd.ecrire_artefact(payload, out)
    print(f"{args.etape}: {out} sha256={sha} duree={dt:.3f}s")
    if args.etape == "selection":
        print(f"sceau selection: {sceller(root, 'selection')}")
    if args.etape == "porteur":
        print(f"sceau porteur: {sceller(root, 'porteur')}")
    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée mesure
    raise SystemExit(main())
