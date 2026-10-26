"""textual_return_dynamics.py — T63 : le pont enfin traversé (chantier 3, dernière marche).

Mesure si une dynamique de cellules du dépôt (``ChronoSpiraton``, équation du
second ordre, POIDS FIGÉS à l'init, jamais entraînés) **transporte** la
géométrie du retour textuel que l'instrument M1 (T42, dépôt Tokenizer) a
gravée sur les agrégats 33D bruts.

Chaîne gelée (TOUR63_EMISSION.md §3.2, TOUR63_CARTOUCHES_GELES.json) :

* segments ABA isolés → 33D (heuristic OFF, oracle) → cartouches **S** (pointe
  énergie) et **P** (moyenne), tranche **MAIN** = dims {6..22, 31, 32}
  (19 dims — les dims 0-5, étiquettes d'opérateur/chiralité, sont EXCLUES) ;
* ``Φ_K(v) = ChronoSpiraton(19, init_scale=0.1).forward(s0=v, steps=K)``,
  poids tirés sous ``torch.manual_seed(seed)`` puis promus float64, ``v``
  BRUT ; **Φ appliqué APRÈS cartouche** ⇒ ``Φ_0 = id`` (parité exacte M1) ;
* ``s_dyn = score_retour(Φ_K(a), Φ_K(a′))`` et ``s_raw = score_retour(a, a′)``
  — le MÊME ``cos − l2`` que M1 (parité portée d'``alpha_omega_spatial``).

Grandeur porteuse (UNE seule) : ``ρ_S`` = Spearman(s_raw, s_dyn) STRATIFIÉ par
corpus (Fisher-z pondéré ``n_c − 3``), cartouche S, K=2, médiane sur 12
graines 0-11. Le PORTEUR ne teste pas l'ordre : Φ s'applique indépendamment à
``a`` et ``a′`` (transport de rang, pas de composition A→B→A′) — la
pré-validation d'order-sensibilité est N-A MOTIVÉE, comme M1 au T42. Le seul
barreau d'ordre (INFO-1) est posé sur ChronoSpiraton, seul organe du dépôt
dont la composition ne commute pas (terme mémoire ``−C(s_prev)``).

LECTEUR PUR : ce module importe ``experimental/chrono.py`` (inchangé) et
l'instrument ``alpha_omega_text`` du dépôt Tokenizer via
``data/tokenizer_bridge.import_alpha_omega_text()`` (miroir du pont existant)
sans modifier ni l'un ni l'autre. Aucune sortie aléatoire non seedée ;
déterministe : mêmes entrées ⇒ mêmes artefacts (JSON canonique, md5 stable).
Les verdicts vivent dans TOUR63_DEPLOIEMENT.md, pas ici.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.cell import SpiratonCell
from ..experimental.chrono import ChronoSpiraton
from ..recursion.recursive import RecursiveSpiraton
from ..data import aba, aba_forms, tokenizer_bridge
from .freeze_token import read_token, verify_freeze

#: eps du score (le même objet que M1 / alpha_omega_metrics).
EPS = 1e-12

#: Dimension d'état = tranche MAIN (19 dims physiques, dims 0-5 exclues).
STATE_SIZE = 19

#: Échelle d'init de Φ (gelée : contractante, norme d'opérateur ≈ 0,12).
INIT_SCALE = 0.1

#: Graines gelées (12, closes en extension).
SEEDS: Tuple[int, ...] = tuple(range(12))

#: Grille K gelée ; le porteur ne lit QUE K=2 (plus petit K où −C(s_prev) agit).
K_GRID: Tuple[int, ...] = (1, 2, 4, 8)
K_PORTEUR = 2

#: Corpus de verdict gelés : (nom, fichier, limit charger_cycles, md5 attendu).
CORPUS_VERDICT: Tuple[Tuple[str, str, Optional[int], str], ...] = (
    ("hc-44", "corpus_horscanon_aba.txt", None, "b1db3c599103148d046869ce4325ade0"),
    ("f0b-24", "corpus_f0b_aba.txt", None, "3c90c13761f16a3eb71b518766c329e1"),
    ("dataset-100", "dataset_aba.txt", 100, "a0f9fad13db6db551a23b56b46f5748b"),
)

#: Split VIERGE (une seule ouverture, APRÈS scellement du porteur).
CORPUS_VIERGE: Tuple[str, str, Optional[int], str] = (
    "claude-76", "corpus_claude_aba.txt", None, "e06ce585a4d7dbe0e313f0f2a5690ce8",
)

#: Référents T42 du gate de parité K=0 (hc-44, tranche MAIN, médianes).
REFERENTS_T42: Dict[str, float] = {
    "P_intra": 0.6826, "P_inter": 0.6649,
    "S_intra": 0.1946, "S_inter": 0.1610,
}
PARITE_TOL = 0.005          # |Δ| ≤ 0.005 sur les 4 médianes
MARGE_S_T42 = 0.293535      # marge de groupe S (replié − ré-émis), hc-44
MARGE_TOL = 0.010           # marge S à ±0.010

#: Sélecteur de groupes FIXE (T42) — contraste à sélecteur fixe, recevable.
GROUPE_REPLIE: Tuple[str, ...] = ("F0", "F0b", "F3", "F4")
GROUPE_RE_EMIS: Tuple[str, ...] = ("F1", "F1b", "F2")

#: Gate de non-vacuité : std(s_dyn) K=2, S, pooled, ≥ 0.02 sur ≥ 10/12 graines.
NON_VACUITE_SEUIL = 0.02
NON_VACUITE_QUORUM = 10
REFERENT_STD_T42 = 0.1299

#: Contrôle H0 (P4) : 200 permutations d'appariement, rng gelé.
N_PERMUTATIONS = 200
PERM_SEED = 63

#: INFO-1 : seuil d'ordre ; INFO-2 : seuil EXACT ; INFO-4/5 : seuils NON-MESURE.
DELTA_MIN_ORDRE = 0.05
RANG1_SEUIL_EXACT = 1e-12
KAPPA_RHO_DETECTABLE = 0.58
MARGE_RATIO_NON_MESURE = 0.375


# --- Φ : la marche neuve (un seul terme ajouté à la chaîne M1) ---------------

def make_chrono(seed: int, *, state_size: int = STATE_SIZE,
                init_scale: float = INIT_SCALE) -> ChronoSpiraton:
    """Construit Φ : ChronoSpiraton à poids FIGÉS, tirés sous ``manual_seed(seed)``.

    Les poids sont tirés en float32 (init torch standard, seedée) puis PROMUS
    float64 (``.double()``) : les agrégats M1 sont float64, la promotion est
    exacte et déterministe. ``bounded=False``, ``c_outside=False`` (équation
    canon §3.2). Les poids ne sont JAMAIS optimisés dans ce tour.
    """
    torch.manual_seed(seed)
    chrono = ChronoSpiraton(
        state_size=state_size, init_scale=init_scale,
        bounded=False, c_outside=False,
    )
    return chrono.double()


def phi_k(chrono: ChronoSpiraton, v: torch.Tensor, steps: int) -> torch.Tensor:
    """``Φ_K(v)`` — déroulé de K pas ; ``K = 0`` est l'IDENTITÉ (v rendu tel quel).

    ``Φ_0 = id`` garantit la parité bit-à-bit avec M1 à K=0 (gate P2).
    ``s_prev`` reste au défaut (zéros) : au premier pas, ``−C(s_{t−1})`` est
    nul ; K=2 est le plus petit K où le terme mémoire agit. Formes ``(d,)``
    ou ``(B, d)`` ; différentiable (aucun ``no_grad`` ici — le tour n'entraîne
    rien, mais le flux de gradient reste testable, quatuor CLAUDE.md).
    """
    if steps < 0:
        raise ValueError("steps doit être >= 0")
    if steps == 0:
        return v
    return chrono(v, steps=steps)


def phi_k_np(chrono: ChronoSpiraton, mat: np.ndarray, steps: int) -> np.ndarray:
    """Applique :func:`phi_k` à une matrice numpy float64 ``(n, d)`` (mesure)."""
    if steps == 0:
        return mat
    with torch.no_grad():
        out = phi_k(chrono, torch.from_numpy(np.ascontiguousarray(mat)), steps)
    return out.numpy()


# --- Statistique : rangs moyens, Spearman, Fisher-z stratifié ----------------

def rangs_moyens(x: Sequence[float]) -> np.ndarray:
    """Rangs 1..n avec moyenne des rangs sur les ex-aequo (tri stable)."""
    arr = np.asarray(x, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(1, arr.size + 1, dtype=np.float64)
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman = Pearson des rangs moyens. Variance nulle ⇒ NaN déclaré."""
    rx, ry = rangs_moyens(x), rangs_moyens(y)
    if rx.size != ry.size:
        raise ValueError(f"tailles incompatibles : {rx.size} vs {ry.size}")
    if rx.size < 2 or float(rx.std()) == 0.0 or float(ry.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def rho_stratifie(rhos_ns: Sequence[Tuple[float, int]]) -> float:
    """Fisher-z stratifié : ``tanh( Σ(n−3)·atanh(ρ) / Σ(n−3) )``.

    ``ρ`` est borné à ±(1 − 1e-12) avant atanh (ρ = ±1 exact serait dégénéré —
    déclaré dans le gel). Un ρ NaN rend NaN (jamais silencieusement 0).
    """
    num, den = 0.0, 0.0
    for rho, n in rhos_ns:
        if math.isnan(rho):
            return float("nan")
        clipped = max(-1.0 + 1e-12, min(1.0 - 1e-12, rho))
        num += (n - 3) * math.atanh(clipped)
        den += (n - 3)
    if den <= 0:
        return float("nan")
    return math.tanh(num / den)


def rho_h0_analytique(n_eff: int) -> float:
    """Plancher H0 analytique : ``tanh(1.96 / sqrt(n_eff))``."""
    return math.tanh(1.96 / math.sqrt(n_eff))


# --- Collecte : corpus gelés → paires (a, a′) par cartouche ------------------

@dataclass(frozen=True)
class PairesCorpus:
    """Paires d'agrégats d'UN corpus : ``a[cart][i]``, ``ap[cart][i]`` (float64).

    ``formes[i]`` = forme étendue du cycle i (sélecteur de groupes INFO-5) ;
    ``exclusions`` = cycles écartés avec MOTIF (réfutabilité §4.6 — jamais de
    cellule vide silencieuse). Les exclusions F5 sont faites par
    ``agreger_corpus`` (M1) en amont et comptées via ``n_charges``.
    """

    nom: str
    n_parses: int
    n_f5: int
    formes: Tuple[str, ...]
    a: Dict[str, np.ndarray]     # cartouche -> (n, 19)
    ap: Dict[str, np.ndarray]    # cartouche -> (n, 19)
    exclusions: Tuple[Dict[str, object], ...]

    @property
    def n(self) -> int:
        return len(self.formes)


def charger_instrument():
    """Charge (tokenizer natif, module M1, modules aba) — le pont, en lecture."""
    tok = tokenizer_bridge.load_native_tokenizer()
    m1 = tokenizer_bridge.import_alpha_omega_text()
    return tok, m1


def md5_fichier(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collecter_corpus(tok, m1, root: Path, nom: str, fichier: str,
                     limit: Optional[int], md5_attendu: str) -> PairesCorpus:
    """Agrège un corpus gelé (garde md5 DURE) en paires (a, a′) S/P × MAIN.

    Un cycle sans agrégat S ou P sur A ou A′ (segment sans token) est exclu
    des DEUX cartouches, motif publié — les effectifs restent appariés.
    """
    path = root / fichier
    md5 = md5_fichier(path)
    if md5 != md5_attendu:
        raise RuntimeError(f"md5 {nom} = {md5} != gelé {md5_attendu} — ARRÊT")
    n_parses = len(m1.charger_cycles(aba, str(path), limit=limit))
    cycles = m1.agreger_corpus(tok, aba, aba_forms, str(path), limit=limit)
    formes: List[str] = []
    a: Dict[str, List[np.ndarray]] = {"S": [], "P": []}
    ap: Dict[str, List[np.ndarray]] = {"S": [], "P": []}
    exclusions: List[Dict[str, object]] = []
    for cm in cycles:
        keys = [("S", "MAIN"), ("P", "MAIN")]
        manquants = [k for k in keys
                     if k not in cm.seg_a.aggs or k not in cm.seg_ap.aggs]
        if manquants:
            exclusions.append({
                "index": cm.index, "forme": cm.forme,
                "motif": "segment sans token (agrégat absent)",
                "cles_manquantes": [list(k) for k in manquants],
            })
            continue
        formes.append(cm.forme)
        for cart in ("S", "P"):
            a[cart].append(np.asarray(cm.seg_a.aggs[(cart, "MAIN")], dtype=np.float64))
            ap[cart].append(np.asarray(cm.seg_ap.aggs[(cart, "MAIN")], dtype=np.float64))
    return PairesCorpus(
        nom=nom,
        n_parses=n_parses,
        n_f5=n_parses - len(cycles),
        formes=tuple(formes),
        a={c: np.stack(v) for c, v in a.items()},
        ap={c: np.stack(v) for c, v in ap.items()},
        exclusions=tuple(exclusions),
    )


def scores_bruts(m1, pc: PairesCorpus, cart: str) -> np.ndarray:
    """``s_raw`` par cycle : ``score_retour(a_i, a′_i)`` (code M1, parité)."""
    return np.asarray(
        [m1.score_retour(pc.a[cart][i], pc.ap[cart][i]) for i in range(pc.n)],
        dtype=np.float64,
    )


def scores_dyn(m1, pc: PairesCorpus, cart: str, chrono: ChronoSpiraton,
               steps: int) -> np.ndarray:
    """``s_dyn`` par cycle : ``score_retour(Φ_K(a_i), Φ_K(a′_i))``.

    Φ s'applique INDÉPENDAMMENT à a et a′ (le porteur ne teste pas l'ordre).
    """
    fa = phi_k_np(chrono, pc.a[cart], steps)
    fap = phi_k_np(chrono, pc.ap[cart], steps)
    return np.asarray(
        [m1.score_retour(fa[i], fap[i]) for i in range(pc.n)], dtype=np.float64,
    )


# --- Contrôles à substrat exact (INFO-2) et mécanisme (INFO-4) ---------------

def sigma_ratio(mat: np.ndarray) -> float:
    """``σ₂/σ₁`` d'une matrice ``(n, d)`` NON centrée (SVD numpy float64)."""
    s = np.linalg.svd(np.asarray(mat, dtype=np.float64), compute_uv=False)
    if s.size < 2 or s[0] == 0.0:
        return float("nan")
    return float(s[1] / s[0])


def deplacements_canon_rang1(vecs: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Déplacements d'1 pas de ``RecursiveSpiraton`` canon sur chaque vecteur.

    Contrôle EXACT (INFO-2) : la mise à jour passe par ``y_to_state ∘ cell``
    (scalaire → Linear(1, d)) ⇒ tout déplacement est colinéaire à un vecteur
    unique ⇒ la matrice des Δ est de rang 1 (``σ₂/σ₁ ≤ 1e-12`` en float64).
    100 % ou FAUX : un échec signalerait un bug d'instrument, pas une
    découverte. Poids figés sous ``manual_seed(seed)``, promus float64.
    """
    torch.manual_seed(seed)
    d = vecs.shape[1]
    cell = SpiratonCell(input_size=2 * d)
    rec = RecursiveSpiraton(
        cell, x_size=d, state_size=d,
        combine="concat", update="residual", alpha=0.1,
    ).double()
    x = torch.from_numpy(np.ascontiguousarray(vecs))
    with torch.no_grad():
        _, state, _ = rec(x, state=x.clone(), steps=1, return_trace=True)
    return (state - x).numpy()


def gap_ordre(pc_list: Sequence[PairesCorpus], chrono: ChronoSpiraton,
              *, steps: int = K_PORTEUR) -> Dict[str, float]:
    """INFO-1 : écart d'ordre de Chrono — ``F(a, a′)`` vs ``F(a′, a)``.

    ``F(x, y) = forward(s0=x, steps=K, s_prev=y)`` : échanger (s0, s_prev)
    est le shuffle d'ordre minimal à dynamique fixe. Un intégrateur additif
    symétrique (``x + y``) donnerait gap = 0 EXACT (témoin par construction) ;
    le terme mémoire ``−C(s_prev)`` de Chrono brise cette symétrie.
    ``gap_i = ‖F(a,a′) − F(a′,a)‖ / (‖F(a,a′)‖ + eps)`` ; médiane + std publiées.
    """
    gaps: List[float] = []
    with torch.no_grad():
        for pc in pc_list:
            a = torch.from_numpy(np.ascontiguousarray(pc.a["S"]))
            ap = torch.from_numpy(np.ascontiguousarray(pc.ap["S"]))
            f_a_ap = chrono(a, steps=steps, s_prev=ap)
            f_ap_a = chrono(ap, steps=steps, s_prev=a)
            num = torch.linalg.vector_norm(f_a_ap - f_ap_a, dim=-1)
            den = torch.linalg.vector_norm(f_a_ap, dim=-1) + EPS
            gaps.extend((num / den).tolist())
    arr = np.asarray(gaps, dtype=np.float64)
    return {
        "mediane": float(np.median(arr)),
        "std": float(arr.std()),
        "n": int(arr.size),
    }


# --- Artefacts JSON canoniques (déterministes, md5 stables) ------------------

def ecrire_artefact(payload: dict, path: Path) -> str:
    """Écrit un JSON canonique (clés triées, LF) et retourne son sha256."""
    txt = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(txt, encoding="utf-8", newline="\n")
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def verifier_gel(root: Path) -> None:
    """Vérifie le jeton β (TOUR63_FREEZE.json) AVANT tout ρ — ordre MATÉRIEL.

    Dérive ⇒ ``RuntimeError`` : le runner CHOISIT de bloquer (le helper
    ``freeze_token`` rapporte, l'appelant statue — contrat T34).
    """
    token = read_token(root / "TOUR63_FREEZE.json")
    report = verify_freeze(token)
    if not report.ok:
        raise RuntimeError(f"gel β dérivé : {report.drifts} — ARRÊT (retouche = candidat mort)")


# --- Étapes du protocole (P1 → porteur → INFO → vierge) ----------------------

def _collecte_verdict(root: Path):
    tok, m1 = charger_instrument()
    pcs = [collecter_corpus(tok, m1, root, nom, fichier, limit, md5)
           for nom, fichier, limit, md5 in CORPUS_VERDICT]
    return tok, m1, pcs


def _mediane_paires_inter(m1, pc: PairesCorpus, cart: str) -> float:
    """Médiane de la baseline inter-cycles s(A_i, A′_j), paires i≠j exhaustives."""
    vals: List[float] = []
    for i in range(pc.n):
        for j in range(pc.n):
            if i != j:
                vals.append(m1.score_retour(pc.a[cart][i], pc.ap[cart][j]))
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _marge_groupes(vals: np.ndarray, formes: Sequence[str]) -> Dict[str, object]:
    """Marge de groupe (replié − ré-émis) au sélecteur FIXE T42, avec effectifs."""
    rep = np.asarray([v for v, f in zip(vals, formes) if f in GROUPE_REPLIE])
    ree = np.asarray([v for v, f in zip(vals, formes) if f in GROUPE_RE_EMIS])
    return {
        "n_replie": int(rep.size),
        "n_re_emis": int(ree.size),
        "mediane_replie": float(np.median(rep)) if rep.size else float("nan"),
        "mediane_re_emis": float(np.median(ree)) if ree.size else float("nan"),
        "marge": (float(np.median(rep)) - float(np.median(ree)))
                 if rep.size and ree.size else float("nan"),
    }


def etape_p1(root: Path) -> dict:
    """P1 : (a) parité K=0 ; (b) effectifs/marginales/capacités/ρ_H0 ;
    (c) non-vacuité. AUCUN ρ n'est calculé ici (le gel β vient après P1).
    """
    _, m1, pcs = _collecte_verdict(root)
    hc = pcs[0]

    # (a) Parité K=0 — Φ_0 = id ⇒ ces médianes SONT celles de M1, re-dérivées.
    s_raw_p = scores_bruts(m1, hc, "P")
    s_raw_s = scores_bruts(m1, hc, "S")
    parite = {
        "P_intra": float(np.median(s_raw_p)),
        "P_inter": _mediane_paires_inter(m1, hc, "P"),
        "S_intra": float(np.median(s_raw_s)),
        "S_inter": _mediane_paires_inter(m1, hc, "S"),
    }
    deltas = {k: abs(parite[k] - REFERENTS_T42[k]) for k in REFERENTS_T42}
    marge = _marge_groupes(s_raw_s, hc.formes)
    parite_pass = (all(d <= PARITE_TOL for d in deltas.values())
                   and abs(float(marge["marge"]) - MARGE_S_T42) <= MARGE_TOL)

    # (b) Effectifs, marginales, capacités, ρ_H0 — publiés AVANT toute lecture.
    n_eff = sum(pc.n - 3 for pc in pcs)
    effectifs = {
        pc.nom: {
            "n": pc.n, "n_parses": pc.n_parses, "n_exclus_F5": pc.n_f5,
            "exclusions_agregat": list(pc.exclusions),
            "formes": {f: pc.formes.count(f) for f in sorted(set(pc.formes))},
        } for pc in pcs
    }

    # (c) Non-vacuité + finitude : s_dyn K=2 S pooled, par graine (PAS de ρ ici).
    stds_par_graine: Dict[str, dict] = {}
    finitude_ok = True
    s_raw_all = {pc.nom: {"S": scores_bruts(m1, pc, "S"),
                          "P": scores_bruts(m1, pc, "P")} for pc in pcs}
    for seed in SEEDS:
        chrono = make_chrono(seed)
        pooled: List[float] = []
        par_corpus: Dict[str, float] = {}
        for pc in pcs:
            sd = scores_dyn(m1, pc, "S", chrono, K_PORTEUR)
            if not np.isfinite(sd).all():
                finitude_ok = False
            par_corpus[pc.nom] = float(sd.std())
            pooled.extend(sd.tolist())
        stds_par_graine[str(seed)] = {
            "std_pooled": float(np.asarray(pooled).std()),
            "std_par_corpus": par_corpus,
        }
    n_ok = sum(1 for v in stds_par_graine.values()
               if v["std_pooled"] >= NON_VACUITE_SEUIL)
    return {
        "tour": 63,
        "etape": "P1",
        "corpus_md5": {nom: md5 for nom, _, _, md5 in CORPUS_VERDICT},
        "parite_K0": {
            "mesure": parite,
            "referents_T42": REFERENTS_T42,
            "deltas": deltas,
            "tolerance": PARITE_TOL,
            "marge_S_groupes": marge,
            "marge_T42": MARGE_S_T42,
            "marge_tolerance": MARGE_TOL,
            "PASS": bool(parite_pass),
        },
        "effectifs": effectifs,
        "n_eff": n_eff,
        "rho_H0_analytique": rho_h0_analytique(n_eff),
        "SE_z": 1.0 / math.sqrt(n_eff),
        "std_s_raw_par_corpus": {
            nom: {c: float(v.std()) for c, v in d.items()}
            for nom, d in s_raw_all.items()
        },
        "non_vacuite": {
            "statistique": "std(s_dyn) K=2 cartouche S pooled, ddof=0, par graine",
            "seuil": NON_VACUITE_SEUIL,
            "quorum": f">= {NON_VACUITE_QUORUM}/12",
            "referent_T42": REFERENT_STD_T42,
            "par_graine": stds_par_graine,
            "n_graines_ok": n_ok,
            "PASS": bool(n_ok >= NON_VACUITE_QUORUM),
        },
        "finitude_100pct": bool(finitude_ok),
    }


def _rhos_par_graine(m1, pcs, cart: str, steps: int
                     ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray],
                                Dict[str, Dict[str, np.ndarray]]]:
    """``ρ_c`` par corpus × graine + s_raw par corpus + s_dyn par corpus × graine."""
    s_raw = {pc.nom: scores_bruts(m1, pc, cart) for pc in pcs}
    rhos: Dict[str, Dict[str, float]] = {pc.nom: {} for pc in pcs}
    s_dyn_all: Dict[str, Dict[str, np.ndarray]] = {pc.nom: {} for pc in pcs}
    for seed in SEEDS:
        chrono = make_chrono(seed)
        for pc in pcs:
            sd = scores_dyn(m1, pc, cart, chrono, steps)
            s_dyn_all[pc.nom][str(seed)] = sd
            rhos[pc.nom][str(seed)] = spearman(s_raw[pc.nom], sd)
    return rhos, s_raw, s_dyn_all


def _stratifie_depuis_rhos(rhos: Dict[str, Dict[str, float]],
                           ns: Dict[str, int]) -> Tuple[Dict[str, float], float]:
    """Médiane sur graines PAR CORPUS puis Fisher-z stratifié (la grandeur §4.2)."""
    rho_c = {nom: float(np.median(list(par_graine.values())))
             for nom, par_graine in rhos.items()}
    strat = rho_stratifie([(rho_c[nom], ns[nom]) for nom in rho_c])
    return rho_c, strat


def etape_porteur(root: Path) -> dict:
    """P2 — PORTEUR SEUL : ρ_S stratifié, K=2, cartouche S, médiane 12 graines.

    Exige le gel β vérifié (ordre matériel). N'écrit QUE le porteur.
    """
    verifier_gel(root)
    _, m1, pcs = _collecte_verdict(root)
    ns = {pc.nom: pc.n for pc in pcs}
    rhos, _, _ = _rhos_par_graine(m1, pcs, "S", K_PORTEUR)
    rho_c, rho_s = _stratifie_depuis_rhos(rhos, ns)
    n_eff = sum(n - 3 for n in ns.values())
    rho_h0 = rho_h0_analytique(n_eff)
    se_z = 1.0 / math.sqrt(n_eff)
    z = math.atanh(max(-1.0 + 1e-12, min(1.0 - 1e-12, rho_s)))
    ic95 = (math.tanh(z - 1.96 * se_z), math.tanh(z + 1.96 * se_z))
    if rho_s >= 0.70:
        bande = "B1"
    elif rho_s >= 0.30:
        bande = "B2"
    elif rho_s >= rho_h0:
        bande = "B3"
    elif abs(rho_s) < rho_h0:
        bande = "B4"
    elif rho_s > -0.30:
        bande = "B5f"
    else:
        bande = "B5"
    return {
        "tour": 63,
        "etape": "PORTEUR",
        "grandeur": "rho_S stratifie (Fisher-z, poids n_c-3), cartouche S, MAIN, K=2, mediane 12 graines",
        "rho_par_corpus_par_graine": rhos,
        "rho_c_mediane_graines": rho_c,
        "n_par_corpus": ns,
        "n_eff": n_eff,
        "rho_S": rho_s,
        "rho_H0_analytique": rho_h0,
        "SE_z": se_z,
        "IC95": list(ic95),
        "bande": bande,
        "indecision_B1_B2": bool(0.61 <= rho_s <= 0.77),
        "f0b_24_solo": "NON-MESURE en solo (n=24) — entre dans le stratifie seulement",
    }


def etape_info(root: Path) -> dict:
    """P2 — INFO après le porteur : ρ_P, poolé, P4, INFO-1/2/4/5, courbe std."""
    verifier_gel(root)
    if not (root / "TOUR63_PORTEUR.json").is_file():
        raise RuntimeError("porteur absent : l'INFO ne se mesure qu'APRÈS le porteur scellé")
    _, m1, pcs = _collecte_verdict(root)
    ns = {pc.nom: pc.n for pc in pcs}
    n_eff = sum(n - 3 for n in ns.values())

    # ρ_S par graine (réutilisé par P4/INFO-4) + ρ_P (C7).
    rhos_s, s_raw_s, s_dyn_s = _rhos_par_graine(m1, pcs, "S", K_PORTEUR)
    rho_c_s, rho_s = _stratifie_depuis_rhos(rhos_s, ns)
    rhos_p, s_raw_p, _ = _rhos_par_graine(m1, pcs, "P", K_PORTEUR)
    rho_c_p, rho_p = _stratifie_depuis_rhos(rhos_p, ns)

    # Poolé (INFO seulement) + confondeur de niveau corpus chiffré.
    rho_poole_par_graine = []
    for seed in SEEDS:
        xs = np.concatenate([s_raw_s[pc.nom] for pc in pcs])
        ys = np.concatenate([s_dyn_s[pc.nom][str(seed)] for pc in pcs])
        rho_poole_par_graine.append(spearman(xs, ys))
    medianes_raw = {pc.nom: float(np.median(s_raw_s[pc.nom])) for pc in pcs}
    ecart_medianes = float(max(medianes_raw.values()) - min(medianes_raw.values()))

    # P4 — contrôle H0 par appariement permuté (200 draws, rng gelé).
    rng = np.random.default_rng(PERM_SEED)
    rho_perm: List[float] = []
    for _ in range(N_PERMUTATIONS):
        perms = {pc.nom: rng.permutation(pc.n) for pc in pcs}
        rhos_perm: Dict[str, Dict[str, float]] = {pc.nom: {} for pc in pcs}
        for seed in SEEDS:
            for pc in pcs:
                sd = s_dyn_s[pc.nom][str(seed)][perms[pc.nom]]
                rhos_perm[pc.nom][str(seed)] = spearman(s_raw_s[pc.nom], sd)
        _, strat = _stratifie_depuis_rhos(rhos_perm, ns)
        rho_perm.append(strat)
    rho_perm_arr = np.asarray(rho_perm, dtype=np.float64)
    q95_abs = float(np.percentile(np.abs(rho_perm_arr), 95))
    rho_h0 = rho_h0_analytique(n_eff)
    seuil_h0_retenu = max(rho_h0, q95_abs)

    # INFO-1 — écart d'ordre (Chrono, par graine ; médiane sur graines).
    gaps = {str(seed): gap_ordre(pcs, make_chrono(seed)) for seed in SEEDS}
    gap_median = float(np.median([g["mediane"] for g in gaps.values()]))

    # INFO-2 — rang-1 du canon (contrôle EXACT, float64).
    vecs = np.concatenate(
        [pc.a["S"] for pc in pcs] + [pc.ap["S"] for pc in pcs], axis=0)
    ratio_rang1 = sigma_ratio(deplacements_canon_rang1(vecs, seed=0))

    # INFO-4 — mécanisme κ_out ↔ ρ_S sur 12 graines.
    kappa: Dict[str, float] = {}
    rho_s_graine: Dict[str, float] = {}
    for seed in SEEDS:
        chrono = make_chrono(seed)
        y = np.concatenate([phi_k_np(chrono, pc.a["S"], K_PORTEUR) for pc in pcs]
                           + [phi_k_np(chrono, pc.ap["S"], K_PORTEUR) for pc in pcs])
        kappa[str(seed)] = sigma_ratio(y)
        rho_s_graine[str(seed)] = rho_stratifie(
            [(rhos_s[pc.nom][str(seed)], pc.n) for pc in pcs])
    rho_kappa = spearman([kappa[str(s)] for s in SEEDS],
                         [rho_s_graine[str(s)] for s in SEEDS])

    # INFO-5 — transport de marge (hc-44, sélecteur FIXE T42).
    hc = pcs[0]
    marges = []
    for seed in SEEDS:
        vals = s_dyn_s[hc.nom][str(seed)]
        marges.append(float(_marge_groupes(vals, hc.formes)["marge"]))
    marge_dyn = float(np.median(marges))
    ratio_marge = marge_dyn / MARGE_S_T42

    # Courbe std(s_dyn) 4 K × 12 graines (48 cellules, INFO même si P3 PASS).
    courbe_std: Dict[str, Dict[str, float]] = {}
    for k in K_GRID:
        par_graine = {}
        for seed in SEEDS:
            chrono = make_chrono(seed)
            pooled = np.concatenate(
                [scores_dyn(m1, pc, "S", chrono, k) for pc in pcs])
            par_graine[str(seed)] = float(pooled.std())
        courbe_std[str(k)] = par_graine

    return {
        "tour": 63,
        "etape": "INFO",
        "rho_P": {
            "rho_par_corpus_par_graine": rhos_p,
            "rho_c_mediane_graines": rho_c_p,
            "rho_P_stratifie": rho_p,
            "prediction_C7": "rho_P >= 0.70 ET rho_P >= rho_S",
            "rho_P_ge_070": bool(rho_p >= 0.70),
            "rho_P_ge_rho_S": bool(rho_p >= rho_s),
            "delta_P_moins_S": rho_p - rho_s,
        },
        "rho_poole_INFO": {
            "par_graine": rho_poole_par_graine,
            "mediane": float(np.median(rho_poole_par_graine)),
            "medianes_s_raw_par_corpus": medianes_raw,
            "ecart_max_medianes": ecart_medianes,
            "note": "confondeur de niveau corpus — jamais porteur",
        },
        "P4_controle_H0": {
            "n_permutations": N_PERMUTATIONS,
            "rng": f"numpy default_rng({PERM_SEED})",
            "q95_abs_rho_perm": q95_abs,
            "q95_rho_perm": float(np.percentile(rho_perm_arr, 95)),
            "min_max": [float(rho_perm_arr.min()), float(rho_perm_arr.max())],
            "rho_H0_analytique": rho_h0,
            "seuil_H0_retenu": seuil_h0_retenu,
            "empirique_gouverne": bool(abs(q95_abs - rho_h0) > 0.03 and q95_abs > rho_h0),
        },
        "INFO1_ordre": {
            "par_graine": gaps,
            "gap_mediane_graines": gap_median,
            "delta_min": DELTA_MIN_ORDRE,
            "order_sensitive": bool(gap_median >= DELTA_MIN_ORDRE),
            "temoin_order_blind": "integrateur additif x+y => gap 0 exact (par construction)",
        },
        "INFO2_rang1_canon": {
            "sigma2_sur_sigma1": ratio_rang1,
            "seuil_exact": RANG1_SEUIL_EXACT,
            "rang1_exact": bool(ratio_rang1 <= RANG1_SEUIL_EXACT),
            "n_vecteurs": int(vecs.shape[0]),
        },
        "INFO4_kappa_out": {
            "kappa_par_graine": kappa,
            "rho_S_par_graine": rho_s_graine,
            "spearman_kappa_rho": rho_kappa,
            "seuil_detection": KAPPA_RHO_DETECTABLE,
            "mesurable": bool(abs(rho_kappa) >= KAPPA_RHO_DETECTABLE)
                         if not math.isnan(rho_kappa) else False,
        },
        "INFO5_marge": {
            "marge_dyn_par_graine": marges,
            "marge_dyn_mediane": marge_dyn,
            "marge_T42": MARGE_S_T42,
            "ratio_transport": ratio_marge,
            "seuil_non_mesure": MARGE_RATIO_NON_MESURE,
            "mesurable": bool(abs(ratio_marge) >= MARGE_RATIO_NON_MESURE),
            "effectifs": _marge_groupes(s_raw_s[hc.nom], hc.formes),
        },
        "courbe_std_4K_x_12_graines": courbe_std,
    }


def etape_vierge(root: Path) -> dict:
    """P2 — SPLIT VIERGE (claude-76), UNE ouverture, EN DERNIER.

    Confirmation de direction et de bande ; jamais substitut du porteur.
    """
    verifier_gel(root)
    if not (root / "TOUR63_PORTEUR.json").is_file():
        raise RuntimeError("porteur absent : le vierge ne s'ouvre qu'APRÈS le porteur scellé")
    if not (root / "TOUR63_INFO.json").is_file():
        raise RuntimeError("INFO absent : le vierge s'ouvre EN DERNIER")
    tok, m1 = charger_instrument()
    nom, fichier, limit, md5 = CORPUS_VIERGE
    pc = collecter_corpus(tok, m1, root, nom, fichier, limit, md5)
    resultats: Dict[str, dict] = {}
    for cart in ("S", "P"):
        s_raw = scores_bruts(m1, pc, cart)
        rhos = {}
        for seed in SEEDS:
            chrono = make_chrono(seed)
            rhos[str(seed)] = spearman(s_raw, scores_dyn(m1, pc, cart, chrono, K_PORTEUR))
        med = float(np.median(list(rhos.values())))
        resultats[cart] = {"rho_par_graine": rhos, "rho_mediane": med}
    n_eff_solo = pc.n - 3
    return {
        "tour": 63,
        "etape": "VIERGE",
        "corpus": nom,
        "n": pc.n,
        "n_parses": pc.n_parses,
        "n_exclus_F5": pc.n_f5,
        "exclusions": list(pc.exclusions),
        "rho": resultats,
        "rho_H0_solo": rho_h0_analytique(n_eff_solo) if n_eff_solo > 0 else float("nan"),
        "role": "confirmation de direction et de bande — jamais substitut du porteur",
    }


def etalonnage(root: Path) -> dict:
    """Étalonnage du mur (~1 % du périmètre) : t_tok, t_dyn, t_perm mesurés."""
    tok, m1 = charger_instrument()
    nom, fichier, limit, md5 = CORPUS_VERDICT[0]
    t0 = time.perf_counter()
    cycles = m1.charger_cycles(aba, str(root / fichier), limit=None)[:2]
    for c in cycles:
        for seg in (c.seg_a, c.seg_b, c.seg_a_prime):
            m1.agreger_segment(tok, seg.text)
    t_tok = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    mat = rng.standard_normal((168, STATE_SIZE))
    chrono = make_chrono(0)
    t0 = time.perf_counter()
    phi_k_np(chrono, mat, K_PORTEUR)
    t_dyn = time.perf_counter() - t0

    xs = rng.standard_normal(168)
    ys = rng.standard_normal(168)
    t0 = time.perf_counter()
    for _ in range(12 * 3):
        spearman(xs, ys)
    t_perm = time.perf_counter() - t0

    t_hat = 84 * t_tok + 48 * t_dyn + N_PERMUTATIONS * t_perm
    return {
        "t_tok_2cycles_s": t_tok,
        "t_dyn_1graine_1K_s": t_dyn,
        "t_perm_1draw_s": t_perm,
        "T_hat_s": t_hat,
        "mur_s": max(10 * t_hat, 120.0),
        "plafond_s": 900.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("etape", choices=["etalonnage", "p1", "porteur", "info", "vierge"])
    parser.add_argument("--root", required=True, help="racine spiraton-enhanced (corpus + artefacts)")
    parser.add_argument("--out", default=None, help="chemin de l'artefact JSON")
    args = parser.parse_args(argv)
    root = Path(args.root)
    fn = {"etalonnage": etalonnage, "p1": etape_p1, "porteur": etape_porteur,
          "info": etape_info, "vierge": etape_vierge}[args.etape]
    t0 = time.perf_counter()
    payload = fn(root)
    dt = time.perf_counter() - t0
    if args.out:
        sha = ecrire_artefact(payload, Path(args.out))
        print(f"{args.etape}: {args.out} sha256={sha} duree={dt:.3f}s")
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        print(f"# duree={dt:.3f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée mesure
    raise SystemExit(main())
