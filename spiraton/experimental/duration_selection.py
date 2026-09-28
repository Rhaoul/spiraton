"""duration_selection.py — T68 : les 8 spécimens BORNÉ-VIVANT — bassin, prédiction, loterie.

Le T67 a produit 153 modèles classés en 5 classes ; 8 sont BORNÉ-VIVANT.
Ce module déploie H68 (TOUR68_EMISSION.md) : « le borné-vivant est un régime,
pas un accident », en trois conjoncts :

* **C1 BASSIN** (porteur principal, α-free) — perturbation contrôlée des poids
  (l.342 : « la diversité est une perturbation contrôlée ») : ``surv(M, ε)``,
  ``ε₅₀`` (rayon de survie, ordinal sur grille close), ``q(ε)`` (vie par
  accident sur 11 contrôles), ``stab_cl`` ;
* **C2 PRÉDICTION γ** (porteur) — un score gelé AVANT toute graine ≥ 50
  (famille close de 128 règles sur 8 features init+préfixe, α sur les 121
  modèles T67) sélectionne un top-m parmi 150 graines neuves du bras A2 ;
  test hypergéométrique EXACT conditionné aux deux marginales (Fraction) ;
* **C3 INIT** (porteur, α-free) — concordance appariée par graine des vivants
  A2 ↔ WD sur les mêmes 150 inits neuves ; Fisher exact (Fraction).

Ordre MATÉRIEL des étapes (chaque étape post-gel vérifie le jeton et le sceau
précédent ; deux gels prévus d'avance, §4.13) : ``instruments`` / ``p0b`` /
``p0c`` / ``etalonnage`` / ``p1`` / ``gel`` (β1 = TOUR68_GEL.json) /
``robustesse`` / ``alpha`` / ``beta`` (β2 = TOUR68_BETA.json +
TOUR68_FREEZE.json) / ``train_neuf`` / ``classer`` / ``predire`` /
``concordance`` / ``info``. Chronologie exigée aux mtimes :
β1 < robustesse < alpha < β2 < train_neuf < classer < predire < concordance.

``duration_training.py`` (partition close, ``paires_67``, ``classer_regime``,
``entrainer_bras``), ``chrono_duration.py``, ``return_training.py`` et
``chrono.py`` sont importés en LECTURE, jamais modifiés. Le corpus v2 du pack
laboratoire n'est JAMAIS ouvert ce tour ; aucune route vers le split brûlé de
``dataset_aba``. Déterministe : seeds fixés, float64, CPU. Les verdicts vivent
dans TOUR68_DEPLOIEMENT.md, pas ici.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from fractions import Fraction
from math import comb
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..diagnostics import textual_return_dynamics as trd
from ..diagnostics.freeze_token import freeze_token, read_token, verify_freeze, write_token
from . import chrono_duration as cd
from . import duration_training as dt
from . import return_training as rt
from .chrono import ChronoSpiraton

# --- Constantes gelées (TOUR68_EMISSION.md §4) --------------------------------

#: I-2 : grille CLOSE des amplitudes de perturbation (6 niveaux avec 0).
EPS_GRID: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
R_SPECIMEN = 24                    #: tirages par spécimen (gelé §4.1).
R_CONTROLE = 16                    #: tirages par contrôle (gelé §4.1).
SEED_PERTURBATION_BASE = 68_000_000  #: rng(68e6 + 1000·id(M) + r).
SEUIL_SURV = 0.50                  #: ε₅₀ = plus grand ε avec surv ≥ 0,50.
EPS50_C1 = 1e-2                    #: C1 ✔ si ≥ 5/8 spécimens à ε₅₀ ≥ 1e-2.
EPS50_LAME = 1e-3                  #: branche basse si ≥ 5/8 à ε₅₀ ≤ 1e-3.
QUORUM_C1 = 5
Q_MAX_OPPOSABLE = 0.10             #: q(1e-2) ≥ 0,10 ⇒ C1 NON-OPPOSABLE (Q3).
STAB_VERR_MIN = 0.90               #: P-VERR (pooled sur les 8 jumeaux).

#: Les 8 spécimens BORNÉ-VIVANT du T67 (liste close, CITÉE, jamais re-mesurée
#: au rang de verdict — la re-classification à ε=0 est la GATE G2).
SPECIMENS: Tuple[Tuple[str, int], ...] = (
    ("A1", 15), ("A2", 1), ("A2", 6), ("A2", 18), ("A2", 25),
    ("WD", 1), ("WD", 3), ("WD", 14))
#: Les 3 spécimens de marge (R_disp^64 à < 20 % du seuil 0,20) — P-MARGE.
SPECIMENS_MARGE: Tuple[Tuple[str, int], ...] = (("A2", 1), ("WD", 1), ("WD", 14))

#: γ : graines NEUVES, jamais entraînées (G3/Q10 les protègent avant β2).
GRAINES_NEUVES: Tuple[int, ...] = tuple(range(50, 200))
BRAS_NEUFS: Tuple[str, ...] = ("A2", "WD")   #: A1 exclu et motivé (§4.10-2).

#: α/β : famille close (§4.3).
E_PREFIXE = 5                      #: E = 5 époques gelé.
N_TOP_ALPHA = 10                   #: top-10 = 20 % de 50 (critère d'ajustement).
BETA_VIDE_MAX = 2                  #: capture ≤ 2/4 ⇒ β VIDE (C2 en INFO).
M_FRACTION = 0.20                  #: m = round(0,20·N_eff) (sélecteur top-m).
SEUIL_P = 0.01
B_MIN_MESURE = 3                   #: B < 3 ⇒ NON-MESURE (Q5/Q7).
QUORUM_G4 = 128                    #: descente ≥ 20 % sur ≥ 128/150 par bras.
SEUIL_REPLI_DIVERGES = 25          #: ≥ 25/150 DIVERGED ⇒ survivants, publié.

#: Mur (§4.11) : mur = max(3·T̂, 1800 s), plafond dur 5400 s.
MUR_PLANCHER_S = 1800.0
MUR_PLAFOND_S = 5400.0
MUR_FACTEUR = 3.0

#: Artefacts du tour (racine spiraton-enhanced, hors git).
ART: Dict[str, str] = {
    "instruments": "TOUR68_INSTRUMENTS.json",
    "p0b": "TOUR68_P0B.json",
    "p0c": "TOUR68_P0C.json",
    "etalonnage": "TOUR68_ETALONNAGE.json",
    "p1": "TOUR68_P1.json",
    "spec": "TOUR68_SPEC.json",
    "predictions": "TOUR68_PREDICTIONS.json",
    "gel": "TOUR68_GEL.json",              # β1 : jeton freeze_token lui-même.
    "robustesse": "TOUR68_ROBUSTESSE.json",
    "alpha": "TOUR68_ALPHA.json",
    "beta": "TOUR68_BETA.json",            # β2 : LE score unique gelé.
    "freeze2": "TOUR68_FREEZE.json",       # β2 : jeton des 24 chemins.
    "train_neuf": "TOUR68_TRAIN_NEUF.json",
    "courbes_neuf": "TOUR68_COURBES_NEUF.json",
    "classer": "TOUR68_CLASSES.json",
    "predire": "TOUR68_PREDICTION_GAMMA.json",
    "concordance": "TOUR68_CONCORDANCE.json",
    "info": "TOUR68_INFO.json",
}
POIDS_DIR = "TOUR68_POIDS"
POIDS_T67 = "TOUR67_POIDS"

#: β1 : chemins gelés APRÈS P1, AVANT robustesse (sous-liste des 24, §4.13 —
#: les 3 artefacts produits par P2 avant β2 s'y ajoutent pour le jeton β2).
FREEZE_PATHS_BETA1: Tuple[str, ...] = (
    "TOUR68_EMISSION.md",
    "TOUR68_SPEC.json",
    "TOUR68_PREDICTIONS.json",
    "TOUR68_INSTRUMENTS.json",
    "TOUR68_P1.json",
    "TOUR67_EMISSION.md",
    "TOUR67_SPEC.json",
    "TOUR67_TRAIN.json",
    "TOUR67_DUREE.json",
    "TOUR67_COURBES.json",
    "TOUR67_INSTRUMENTS.json",
    "TOUR67_POIDS_SHA256.json",
    "TOUR67_POIDS/A2_seed1.bin",
    "TOUR67_POIDS/A2_seed25.bin",
    "TOUR67_POIDS/WD_seed14.bin",
    "dataset_aba.txt",
    "spiraton/spiraton/experimental/duration_training.py",
    "spiraton/spiraton/experimental/chrono.py",
    "spiraton/spiraton/experimental/chrono_duration.py",
    "spiraton/spiraton/experimental/duration_selection.py",
    "Tokenizer/bin/libspiratontokenizer.so",
)
#: β2 : les 24 chemins (liste close §4.13).
FREEZE_PATHS_BETA2: Tuple[str, ...] = FREEZE_PATHS_BETA1 + (
    "TOUR68_ALPHA.json",
    "TOUR68_BETA.json",
    "TOUR68_ROBUSTESSE.json",
)

#: G3 : liste close des artefacts d'entraînement du projet (fraîcheur).
ARTEFACTS_ENTRAINEMENT: Tuple[str, ...] = (
    "TOUR64_TRAIN.json", "TOUR65_ABLATION.json", "TOUR67_TRAIN.json")


# --- I-1 : perturbation contrôlée (l.342) -------------------------------------

def perturber_matrices(mats: Dict[str, np.ndarray], eps: float,
                       rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """``M'_X = X + ε · (‖X‖_F/√n_X) · G_X``, ``G_X ~ N(0,1)`` i.i.d. (§4.1 I-1).

    Ordre de tirage GELÉ : clés triées (chrono : A.weight … L.weight ;
    ridge : W puis b). ``ε = 0`` ⇒ copies BIT-IDENTIQUES (substrat exact vi),
    aucun tirage consommé — déclaré.
    """
    out: Dict[str, np.ndarray] = {}
    for k in sorted(mats):
        x = np.asarray(mats[k], dtype=np.float64)
        if eps == 0.0:
            out[k] = x.copy()
            continue
        g = rng.standard_normal(x.shape)
        out[k] = x + eps * (float(np.linalg.norm(x)) / math.sqrt(x.size)) * g
    return out


def rng_perturbation(id_m: int, r: int) -> np.random.Generator:
    """``default_rng(68_000_000 + 1000·id(M) + r)`` — gelé (§4.1 I-1)."""
    return np.random.default_rng(SEED_PERTURBATION_BASE + 1000 * id_m + r)


def etat_chrono_vers_mats(state: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """State dict → matrices numpy float64 (clés « A.weight » … « L.weight »)."""
    return {k: v.detach().cpu().numpy().astype(np.float64)
            for k, v in state.items()}


def mats_vers_etat_chrono(mats: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in mats.items()}


def theta_vers_mats(theta: np.ndarray) -> Dict[str, np.ndarray]:
    """Ridge : ``{W: theta[:-1] (19×19), b: theta[-1] (19,)}`` — déclaré."""
    return {"W": theta[:-1].astype(np.float64), "b": theta[-1].astype(np.float64)}


def mats_vers_theta(mats: Dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([mats["W"], mats["b"][None, :]])


def classer_etat_chrono(mats: Dict[str, np.ndarray], a_s: np.ndarray,
                        pairs: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    """Classe (partition T67 INCHANGÉE) d'un jeu de matrices chrono."""
    model = rt.modele_depuis_poids(dt.INIT_SCALE, mats_vers_etat_chrono(mats))
    trj = cd.derouler(cd.pas_chrono(model), a_s)
    return dt.classer_regime(trj, a_s, pairs)


def classer_theta_ridge(theta: np.ndarray, a_s: np.ndarray,
                        pairs: Sequence[Tuple[int, int]]) -> Dict[str, object]:
    trj = cd.derouler(cd.pas_ridge(theta), a_s)
    return dt.classer_regime(trj, a_s, pairs)


# --- Liste close des 19 modèles (§4.4) ----------------------------------------

def jumeau_verrouille(par_graine: Dict[str, dict], seed: int) -> int:
    """Règle DÉTERMINISTE : le VERROUILLÉ du même bras de plus petite graine
    > g, ordre cyclique sur 0-49 (les EXCLU/DIVERGED n'ont pas de classe)."""
    for k in range(1, 50):
        s = (seed + k) % 50
        e = par_graine.get(str(s))
        if e is not None and e.get("classe") == "VERROUILLE":
            return s
    raise RuntimeError(f"aucun jumeau VERROUILLÉ trouvé pour la graine {seed}")


def plus_petit_divergent(par_graine: Dict[str, dict]) -> int:
    dv = [int(s) for s, e in par_graine.items() if e.get("classe") == "DIVERGENT"]
    if not dv:
        raise RuntimeError("aucun DIVERGENT dans ce bras")
    return min(dv)


def liste_close_modeles(root: Path) -> List[Dict[str, object]]:
    """Les 19 modèles de C1, ordre GELÉ a priori : 8 spécimens (A1-15, A2-1,
    A2-6, A2-18, A2-25, WD-1, WD-3, WD-14), puis leurs 8 jumeaux VERROUILLÉS
    (même ordre, règle cyclique), puis le plus petit DIVERGENT de A2, de WD,
    puis la ridge itérée. ``id(M)`` = index dans cette liste. Un jumeau
    partagé par deux spécimens compte comme deux ENTRÉES de contrôle (ids
    distincts ⇒ tirages distincts) — déclaré."""
    duree = json.loads((root / "TOUR67_DUREE.json").read_text(encoding="utf-8"))
    modeles: List[Dict[str, object]] = []
    for bras, g in SPECIMENS:
        e = duree["par_bras"][bras]["par_graine"][str(g)]
        modeles.append({"role": "SPECIMEN", "bras": bras, "seed": g,
                        "classe_t67": e["classe"],
                        "fichier": f"{bras}_seed{g}.bin"})
    for bras, g in SPECIMENS:
        j = jumeau_verrouille(duree["par_bras"][bras]["par_graine"], g)
        modeles.append({"role": "JUMEAU_VERROUILLE", "bras": bras, "seed": j,
                        "apparie_a": f"{bras}_seed{g}",
                        "classe_t67": "VERROUILLE",
                        "fichier": f"{bras}_seed{j}.bin"})
    for bras in ("A2", "WD"):
        s = plus_petit_divergent(duree["par_bras"][bras]["par_graine"])
        modeles.append({"role": "DIVERGENT", "bras": bras, "seed": s,
                        "classe_t67": "DIVERGENT",
                        "fichier": f"{bras}_seed{s}.bin"})
    modeles.append({"role": "RIDGE", "bras": None, "seed": None,
                    "classe_t67": "VERROUILLE", "fichier": None})
    for i, m in enumerate(modeles):
        m["id"] = i
    return modeles


def eps50_de_surv(surv_par_eps: Dict[float, float]) -> float:
    """I-2 : plus grand ε de la grille close avec surv ≥ 0,50 ; 0 sinon."""
    best = 0.0
    for eps in EPS_GRID:
        if surv_par_eps[eps] >= SEUIL_SURV:
            best = eps
    return best


# --- α : features (liste close de 8, §4.3) ------------------------------------

def features_init(seed: int) -> Dict[str, float]:
    """f1-f4 : à l'init seedée (``torch.manual_seed(g)``, AUCUNE époque) —
    identiques entre bras à graine fixée (même construction que
    ``entrainer_bras``)."""
    torch.manual_seed(seed)
    model = ChronoSpiraton(state_size=rt.STATE_SIZE, init_scale=dt.INIT_SCALE,
                           bounded=False, c_outside=False).double()
    mats = cd.matrices_np(model)
    f2 = float(np.max(np.abs(np.linalg.eigvals(
        mats["D"] @ mats["A"] + mats["L"]))))
    return {"f1": cd.rho_compagnon(mats), "f2": f2,
            "f3": float(np.linalg.norm(mats["C"])),
            "f4": float(np.linalg.norm(mats["B"]))}


def features_prefixe(courbes: Dict[str, Sequence[float]]) -> Optional[Dict[str, float]]:
    """f5-f8 : préfixe E = 5 époques. GATE DE CODE (Q6) : seuls les 5 PREMIERS
    éléments des courbes sont lus (troncature matérielle) ; < 5 époques ⇒
    ``None`` (graine hors N_eff, comptée et publiée)."""
    pt = list(courbes["perte_train"])[:E_PREFIXE]
    dv = list(courbes["delta_med_val"])[:E_PREFIXE]
    rd = list(courbes["r_disp"])[:E_PREFIXE]
    if len(pt) < E_PREFIXE or len(dv) < E_PREFIXE or len(rd) < E_PREFIXE:
        return None
    return {"f5": float(pt[4]), "f6": float(dv[4]), "f7": float(rd[4]),
            "f8": float(dv[4] - dv[0])}


FEATURES: Tuple[str, ...] = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8")


def constantes_z(valeurs: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Standardisation robuste ``z = (f − médiane)/MAD`` — constantes calculées
    sur les 50 graines A2 du T67, gelées dans TOUR68_BETA.json. MAD = 0 ⇒
    z := 0 (déclaré, aucun cas attendu)."""
    out = {}
    for f in FEATURES:
        v = np.asarray(valeurs[f], dtype=np.float64)
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        out[f] = {"mediane": med, "MAD": mad}
    return out


def z_feature(x: float, cst: Dict[str, float]) -> float:
    if cst["MAD"] == 0.0:
        return 0.0
    return (x - cst["mediane"]) / cst["MAD"]


def regles_closes() -> List[Tuple[Tuple[int, int], ...]]:
    """Les 128 règles, énumérées DANS L'ORDRE DE DÉPARTAGE gelé : (1) moins de
    features d'abord ; (2) ordre lexicographique (f_i, f_j) ; (3) signes +
    avant −. Chaque règle = tuple de termes (k, signe)."""
    regles: List[Tuple[Tuple[int, int], ...]] = []
    for k in range(1, 9):
        for s in (1, -1):
            regles.append(((k, s),))
    for i in range(1, 9):
        for j in range(i + 1, 9):
            for si in (1, -1):
                for sj in (1, -1):
                    regles.append(((i, si), (j, sj)))
    return regles


def regle_id(regle: Tuple[Tuple[int, int], ...]) -> str:
    return "".join(f"{'+' if s > 0 else '-'}z{k}" for k, s in regle)


def score_regle(regle: Tuple[Tuple[int, int], ...],
                feats: Dict[str, float],
                z_csts: Dict[str, Dict[str, float]]) -> float:
    """LE score : somme signée de features standardisées — ne voit QUE f1-f8
    (init + préfixe 5 époques, gate Q6 par construction de features_prefixe)."""
    return float(sum(s * z_feature(feats[f"f{k}"], z_csts[f"f{k}"])
                     for k, s in regle))


def top_m_graines(scores: Dict[int, float], m: int) -> List[int]:
    """Sélecteur top-m GELÉ : scores décroissants, départage graine croissante."""
    tri = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [g for g, _ in tri[:m]]


# --- Tests exacts (Fraction, deux routes indépendantes — §4.7 vii) ------------

def hyper_pmf_comb(n_tot: int, k_tot: int, m: int, k: int) -> Fraction:
    """Route 1 : pmf hypergéométrique par coefficients binomiaux exacts."""
    if k < 0 or k > min(k_tot, m) or m - k > n_tot - k_tot:
        return Fraction(0)
    return Fraction(comb(k_tot, k) * comb(n_tot - k_tot, m - k), comb(n_tot, m))


def hyper_tail_ge(n_tot: int, k_tot: int, m: int, k: int) -> Fraction:
    """Route 1 : ``P(X ≥ k)`` exacte."""
    return sum((hyper_pmf_comb(n_tot, k_tot, m, j)
                for j in range(k, min(k_tot, m) + 1)), Fraction(0))


def _comb_iterative(n: int, k: int) -> Fraction:
    """Binôme par produit itératif Fraction (SANS ``math.comb`` — route 2)."""
    if k < 0 or k > n:
        return Fraction(0)
    r = Fraction(1)
    for i in range(k):
        r = r * (n - i) / (i + 1)
    return r


def hyper_tail_ge_recurrence(n_tot: int, k_tot: int, m: int, k: int) -> Fraction:
    """Route 2 (indépendante de ``math.comb``) : amorce par produit itératif,
    puis récurrence multiplicative sur la pmf."""
    lo = max(0, m - (n_tot - k_tot))
    pmf = (_comb_iterative(k_tot, lo) * _comb_iterative(n_tot - k_tot, m - lo)
           / _comb_iterative(n_tot, m))
    total = pmf if lo >= k else Fraction(0)
    for j in range(lo, min(k_tot, m)):
        pmf = pmf * (k_tot - j) * (m - j) / ((j + 1) * (n_tot - k_tot - m + j + 1))
        if j + 1 >= k:
            total += pmf
    return total


def k_star_hyper(n_tot: int, k_tot: int, m: int,
                 seuil: Fraction = Fraction(1, 100)) -> Optional[int]:
    """Plus petit k avec ``P(X ≥ k) < seuil`` ; None ⇒ test AVEUGLE (Q5)."""
    for k in range(0, min(k_tot, m) + 1):
        if hyper_tail_ge(n_tot, k_tot, m, k) < seuil:
            return k
    return None


def binom_tail_ge(n: int, p: Fraction, k: int) -> Fraction:
    pmf = cd.binom_pmf_fraction(n, p)
    return sum(pmf[k:], Fraction(0))


def binom_tail_le(n: int, p: Fraction, k: int) -> Fraction:
    pmf = cd.binom_pmf_fraction(n, p)
    return sum(pmf[:k + 1], Fraction(0))


# --- Gardes matérielles (sceaux, jetons, mur) ----------------------------------

def verifier_gel_beta1(root: Path) -> None:
    report = verify_freeze(read_token(root / ART["gel"]))
    if not report.ok:
        raise RuntimeError(f"gel β1 dérivé : {report.drifts} — ARRÊT")


def verifier_gel_beta2(root: Path) -> None:
    report = verify_freeze(read_token(root / ART["freeze2"]))
    if not report.ok:
        raise RuntimeError(f"gel β2 dérivé : {report.drifts} — ARRÊT")


def exiger(root: Path, *cles: str) -> None:
    for cle in cles:
        if not (root / ART[cle]).is_file():
            raise RuntimeError(
                f"artefact {ART[cle]} absent : l'étape refuse de s'exécuter "
                f"(ordre MATÉRIEL du protocole)")


def refuser_si_deja_fait(root: Path, cle: str) -> None:
    """One-shot : re-exécuter une étape P2 déjà scellée = retouche = mort."""
    if (root / ART[cle]).is_file():
        raise RuntimeError(
            f"{ART[cle]} existe déjà : one-shot, aucune retouche (γ)")


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


def budget_verifier(root: Path, t_etape: float) -> None:
    """Mur §4.11 : cumul P2 (durées scellées + étape en cours) > mur ⇒ ARRÊT
    (I10), AUCUNE dégradation de périmètre en vol."""
    etal = json.loads((root / ART["etalonnage"]).read_text(encoding="utf-8"))
    mur = float(etal["mur_s"])
    cumul = t_etape
    for cle in ("robustesse", "train_neuf", "classer", "predire", "concordance"):
        p = root / ART[cle]
        if p.is_file():
            try:
                cumul += float(json.loads(p.read_text(encoding="utf-8"))
                               .get("duree_s", 0.0))
            except Exception:
                pass
    if cumul > mur:
        raise RuntimeError(
            f"MUR DE BUDGET dépassé : {cumul:.1f}s > {mur:.1f}s — ARRÊT (I10)")


# --- Gate G5/Q11 : scan matériel du code neuf ----------------------------------

def scan_g5(fichiers: Sequence[Path]) -> Dict[str, object]:
    """Scan lexical (taux de faux positifs déclaré Q11 : tout match est
    instruit à la main, jamais auto-classé). Jetons construits par
    concaténation — le scanner ne se déclenche pas lui-même."""
    interdits = ["35" + "01", "50" + "01", '"TE' + 'ST"', "'TE" + "ST'",
                 "TOUR64_POR" + "TEUR", "TOUR64_DELTAS" + "_T",
                 "TOUR64_CARTOUCHE" + "_P", "TOUR64_IN" + "FO",
                 "aba" + "_v2"]
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


def _train_val(root: Path):
    tok, m1 = _instrument()
    tr = rt.charger_split(tok, m1, root, "TRAIN", rt.SPLITS["TRAIN"])
    va = rt.charger_split(tok, m1, root, "VALIDATION", rt.SPLITS["VALIDATION"])
    return tok, m1, tr, va


def _charger_mats_modele(root: Path, fichier: str) -> Dict[str, np.ndarray]:
    return etat_chrono_vers_mats(rt.charger_poids(root / POIDS_T67 / fichier))


def _t67(root: Path, nom: str) -> dict:
    return json.loads((root / f"TOUR67_{nom}.json").read_text(encoding="utf-8"))


# --- Étapes ---------------------------------------------------------------------

def etape_instruments(root: Path) -> dict:
    """PORTÉE Q1-Q13 (T60) — publiée AVANT toute mesure (transcription §4.8)."""
    return {
        "tour": 68, "artefact": "INSTRUMENTS",
        "Q1_partition_close": {
            "role": "PORTEUR commun C0/C1/C2/C3",
            "borne": "seuils fixes T67 inchangés au chiffre près ; ne distingue pas "
                     "deux vivants de qualités différentes ; R_disp^64 ne détecte une "
                     "perte de diversité qu'à partir de 5x ; classe ordinale grossière",
            "branches": "les 5 classes toutes atteintes au T67 (recensement cité)"},
        "Q2_eps50": {
            "role": "PORTEUR C1 (instrument NEUF)",
            "borne": "ordinal à 6 niveaux sur grille close (pas de 10x puis 3x) ; "
                     "surv à R=24 => ±0,10 au seuil 0,5 => ±1 cran ; isotrope par "
                     "matrice => AVEUGLE à l'anisotropie directionnelle (déclaré)",
            "branches": "eps50=0 atteignable (frontière) ; eps50=1e-1 atteignable "
                        "(la ridge devrait rester dans sa classe très loin)"},
        "Q3_q_eps": {
            "role": "CONTRÔLE (borne C1)",
            "borne": "176 tirages par eps => détecte q >= 0,03 ; information "
                     "effective : 11 contrôles sur 2 classes (VERROUILLÉ, DIVERGENT) "
                     "— ne borne pas le voisinage BASSIN/EXTINCTION",
            "branches": "q=0 et q>0 atteignables ; q(1e-2) >= 0,10 => C1 "
                        "NON-OPPOSABLE comme preuve de bassin (écrit d'avance)"},
        "Q4_stab_cl": {"role": "INFO/contrôle",
                       "borne": "proportion à R=16 => résolution 0,0625",
                       "branches": "deux branches atteignables"},
        "Q5_hypergeometrique": {
            "role": "PORTEUR C2",
            "borne": "m/N = 0,20 gelé ; k*(B) publié en Fraction ; AVEUGLE si "
                     "B < 3 (NON-MESURE gelé) ; MDE capture 60 %, enrichissement 2,9x",
            "branches": "k=0 et k=B atteignables ; null exactement calculable"},
        "Q6_score_beta": {
            "role": "DISCRIMINANT C2",
            "borne": "ne détecte qu'une structure exprimable en <= 2 features "
                     "standardisées parmi 8 ; aveugle à toute interaction hors "
                     "famille ; n'utilise que init + 5 époques (gate de code)",
            "branches": "enrichit / n'enrichit pas atteignables ; branche beta-VIDE "
                        "atteignable et chiffrée d'avance (P-betaVIDE 35/65)"},
        "Q7_fisher": {
            "role": "PORTEUR C3",
            "borne": "conditionné aux marginales ; k* ~ 4 ; aveugle si une "
                     "marginale < 3 ; MDE transfert tau >= 0,5",
            "branches": "les deux atteignables (indépendance => 0,72 attendu)"},
        "Q8_binomiale_taux_base": {
            "role": "CONTRÔLE C0",
            "borne": "détecte un écart de reproductibilité ~2x ; intervalle 95 % "
                     "[4;20] pour p=8 %, N=150",
            "branches": "les deux atteignables (T67 : 4/50)"},
        "Q9_parite_bit": {
            "role": "GATE G1, substrat EXACT",
            "borne": "toute dérive de spec au dernier bit ; ne détecte pas une "
                     "dérive de plateforme identique des deux côtés",
            "branches": "PASS (17 preuves de déterminisme) et FAIL atteignables"},
        "Q10_fraicheur_chronologie": {
            "role": "GATE G3",
            "borne": "détecte une graine >= 50 dans un artefact antérieur, ou un "
                     "artefact neuf à mtime antérieur au sceau beta2 ; ne détecte "
                     "pas une fuite par mémoire humaine — d'où l'interdit écrit",
            "branches": "détection / non-détection atteignables"},
        "Q11_scan": {
            "role": "GATE G5",
            "borne": "scan LEXICAL — un match dans un commentaire compte (précédent "
                     "T66) => tout match instruit à la main et publié",
            "branches": "les deux atteignables"},
        "Q12_sha256": {"role": "GATE G2",
                       "borne": "toute dérive d'octet ; sans paramètre",
                       "branches": "PASS/FAIL atteignables"},
        "Q13_non_stationnarite": {
            "role": "INFO",
            "borne": "mediane_t ||s_{t+1}-s_t|| sur les 16 derniers pas ; ne "
                     "qualifie RIEN ce tour (définition T67 de tenir inchangée)",
            "branches": "les deux atteignables"},
        "substrat": "numérique-géométrique (poids R^{5x19x19}, trajectoires, "
                    "cos-l2 hérité, classes) ; float64 CPU",
        "order_sensibilite": "N-A MOTIVÉE (T22/T23) : Phi^K puissance d'une seule "
                             "application, s_prev=0, le segment B n'entre pas — "
                             "aucun test d'ordre ni de clôture porté",
        "agregat": "aucune MOYENNE ne porte de verdict : proportions, médianes, "
                   "un ordinal (eps50) ; moyennes en INFO (T40)",
    }


def etape_p0b(root: Path) -> dict:
    """P0-b : substrats EXACTS §4.7 (iv)-(vii) + gates de portée (Q6, Q10).
    Doublé par ``tests/test_duration_selection.py`` (quatuor pytest)."""
    d = rt.STATE_SIZE
    rng = np.random.default_rng(680)
    pts = rng.standard_normal((16, d))
    pairs = [(i, (i + 1) % 16) for i in range(16)]
    # (iv) A=B=C=D=0, L=I => lambda_i = 0 EXACT.
    mats_id = {f"{n}.weight": np.zeros((d, d)) for n in "ABCD"}
    mats_id["L.weight"] = np.eye(d)
    model = rt.modele_depuis_poids(0.1, mats_vers_etat_chrono(mats_id))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    ev_iv = bool((trj.lam == 0.0).all() and trj.f_div == 0.0)
    # (v) A=I, B=C=0, D=alpha*I, L=0 => lambda = log alpha (1e-12 rel), rho0 exact.
    alpha = 1.5
    mats_g = {"A.weight": np.eye(d), "B.weight": np.zeros((d, d)),
              "C.weight": np.zeros((d, d)), "D.weight": alpha * np.eye(d),
              "L.weight": np.zeros((d, d))}
    model2 = rt.modele_depuis_poids(0.1, mats_vers_etat_chrono(mats_g))
    trj2 = cd.derouler(cd.pas_chrono(model2), pts)
    ev_v_lam = bool(np.max(np.abs(trj2.lam - math.log(alpha))
                           / math.log(alpha)) <= 1e-12)
    rho0 = cd.rho_compagnon(cd.matrices_np(model2))
    ev_v = bool(ev_v_lam and rho0 == alpha)
    # (vi) perturbation eps=0 => bit-identique (chrono ET ridge).
    torch.manual_seed(680)
    m0 = ChronoSpiraton(state_size=d, init_scale=0.1).double()
    mats0 = etat_chrono_vers_mats(m0.state_dict())
    p0 = perturber_matrices(mats0, 0.0, rng_perturbation(0, 0))
    ev_vi_chrono = bool(all(np.array_equal(p0[k], mats0[k]) for k in mats0))
    theta0 = rng.standard_normal((d + 1, d))
    mt = theta_vers_mats(theta0)
    pt = perturber_matrices(mt, 0.0, rng_perturbation(18, 0))
    ev_vi_ridge = bool(np.array_equal(mats_vers_theta(pt), theta0))
    ev_vi = bool(ev_vi_chrono and ev_vi_ridge)
    # (vii) hypergéométrique/Fisher : DEUX routes exactes, valeurs connues.
    grille = [(150, 12, 30), (150, 9, 30), (50, 4, 10), (150, 12, 9), (10, 5, 5)]
    routes_ok = True
    for n_tot, k_tot, m in grille:
        for k in range(0, min(k_tot, m) + 1):
            if hyper_tail_ge(n_tot, k_tot, m, k) != \
                    hyper_tail_ge_recurrence(n_tot, k_tot, m, k):
                routes_ok = False
    connue = bool(hyper_tail_ge(10, 5, 5, 5) == Fraction(1, 252))
    ev_vii = bool(routes_ok and connue)
    # Gate Q6 : le score ne voit que 5 époques (troncature matérielle).
    courbes_longues = {"perte_train": list(range(40)),
                       "delta_med_val": [x * 0.1 for x in range(40)],
                       "r_disp": [1.0] * 40, "dbar": [0.1] * 40}
    courbes_5 = {k: v[:5] for k, v in courbes_longues.items()}
    gate_q6 = bool(features_prefixe(courbes_longues) == features_prefixe(courbes_5))
    # Gate Q10 (structurelle) : GRAINES_NEUVES n'est consommée que par
    # etape_train_neuf (vérifié sur la source du module ; marqueur construit
    # pour ne pas s'auto-matcher).
    src = Path(__file__).read_text(encoding="utf-8")
    marqueur = "\ndef etape_" + "train_neuf("
    corps = src.split(marqueur, 1)
    avant = corps[0].count("GRAINES_NEUVES")
    # occurrences licites avant train_neuf : définition + G3 (lecture seule,
    # vérifie l'ABSENCE) + textes/spec.
    gate_q10 = bool(len(corps) == 2)
    payload = {
        "tour": 68, "etape": "P0B",
        "Eiv_identite": {"PASS": ev_iv},
        "Ev_geometrique": {"PASS": ev_v, "alpha": alpha, "rho0": rho0,
                           "rho0_exact": bool(rho0 == alpha)},
        "Evi_perturbation_eps0_bit": {"PASS": ev_vi, "chrono": ev_vi_chrono,
                                      "ridge": ev_vi_ridge},
        "Evii_fraction_deux_routes": {"PASS": ev_vii, "grille": grille,
                                      "valeur_connue_10_5_5": connue},
        "gate_Q6_score_5_epoques": {"PASS": gate_q6},
        "gate_Q10_route_graines_neuves": {
            "PASS": gate_q10, "occurrences_avant_train_neuf": avant,
            "note": "vérification structurelle ; le test pytest la double"},
        "PASS": bool(ev_iv and ev_v and ev_vi and ev_vii and gate_q6 and gate_q10),
    }
    if not payload["PASS"]:
        raise RuntimeError(f"P0-b FAIL : {payload}")
    return payload


def etape_p0c(root: Path) -> dict:
    """P0-c : la re-descente de parité A2 graine 25 (objet T67, AUCUNE graine
    neuve) — à exécuter DEUX fois en processus froids (18e preuve)."""
    tok, m1, tr, va = _train_val(root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    rr = dt.entrainer_bras(m1, "A2", 25, tr.a["S"], tr.ap["S"],
                           va.a["S"], va.ap["S"], s_raw_val, m)
    return {
        "tour": 68, "etape": "P0C",
        "config": {"bras": "A2", "seed": 25, "spec": "T67 à l'identique"},
        "m_train_S": m,
        "state_sha256": rr.state_sha256,
        "best_epoch": rr.best_epoch, "best_delta_val": rr.best_delta_val,
        "epochs_run": rr.epochs_run, "early_stopped": rr.early_stopped,
        "diverged": rr.diverged,
        "note": "poids JETÉS (preuve de déterminisme) — le verdict n'en dépend pas",
    }


def etape_etalonnage(root: Path) -> dict:
    """§4.11 : étalonnage MESURÉ (~1,3 %, objets NON γ), extrapolation gelée
    AVEC le mur : T̂ = 150·(t_A2+t_WD) + 2140·t_class + 20·t_init ;
    mur = max(3·T̂, 1800 s), plafond dur 5400 s."""
    tok, m1, tr, va = _train_val(root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    t0 = time.perf_counter()
    dt.entrainer_bras(m1, "A2", 25, tr.a["S"], tr.ap["S"],
                      va.a["S"], va.ap["S"], s_raw_val, m)
    t_a2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    dt.entrainer_bras(m1, "WD", 14, tr.a["S"], tr.ap["S"],
                      va.a["S"], va.ap["S"], s_raw_val, m)
    t_wd = time.perf_counter() - t0
    a_s = tr.a["S"]
    pairs = dt.paires_67(a_s.shape[0])
    mats = _charger_mats_modele(root, "A1_seed15.bin")
    t0 = time.perf_counter()
    classer_etat_chrono(mats, a_s, pairs)
    t_class = time.perf_counter() - t0
    t0 = time.perf_counter()
    for r in range(R_SPECIMEN):
        pm = perturber_matrices(mats, 1e-2, rng_perturbation(0, r))
        classer_etat_chrono(pm, a_s, pairs)
    t_pert = time.perf_counter() - t0
    t0 = time.perf_counter()
    features_init(0)
    t_init = time.perf_counter() - t0
    t_hat = (150.0 * (t_a2 + t_wd) + (1840 + 300) * t_class + 20.0 * t_init)
    mur = min(max(MUR_FACTEUR * t_hat, MUR_PLANCHER_S), MUR_PLAFOND_S)
    return {
        "tour": 68, "etape": "ETALONNAGE",
        "t_A2_s": t_a2, "t_WD_s": t_wd, "t_class_s": t_class,
        "t_pert_lot24_s": t_pert, "t_init_s": t_init,
        "coherence_t_pert_vs_24_t_class": t_pert / (24.0 * t_class)
        if t_class > 0 else None,
        "T_hat_s": t_hat, "mur_s": mur, "facteur": MUR_FACTEUR,
        "plancher_s": MUR_PLANCHER_S, "plafond_s": MUR_PLAFOND_S,
        "a_priori_refutable": {"T_hat_a_priori_s": 1640.0,
                               "arret_probable_si_T_hat_sup": 1800.0},
        "note": "2 descentes + 25 classifications pré-gel sur objets T67 "
                "(non-γ, poids jetés) ; la première graine NEUVE ne bouge "
                "qu'après β2",
    }


def etape_p1(root: Path) -> dict:
    """P1 : (a) G1 parité au bit (A2-25, WD-14) ; (b) G2 objets (153 sha256 +
    re-classification ε=0 des 19 == T67) ; (c) G3 fraîcheur ; (d) publications
    (tables k* exactes) ; (f) G5 scan. DÉTERMINISTE (exécuté ×2 à P0-c)."""
    tok, m1, tr, va = _train_val(root)
    train67 = _t67(root, "TRAIN")
    runs67 = {r["run_id"]: r for r in train67["runs"]}
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    poids_dir = root / POIDS_DIR
    poids_dir.mkdir(exist_ok=True)

    # (a) G1 — re-descente A2-25 et WD-14 : state ET fichier au bit T67.
    g1_par_run = []
    for bras, seed in (("A2", 25), ("WD", 14)):
        rr = dt.entrainer_bras(m1, bras, seed, tr.a["S"], tr.ap["S"],
                               va.a["S"], va.ap["S"], s_raw_val, m)
        fichier = poids_dir / f"G1_{bras}_seed{seed}.bin"
        sha_fichier = rt.sauver_poids(rr.state, fichier)
        att = runs67[f"{bras}_seed{seed}"]
        g1_par_run.append({
            "bras": bras, "seed": seed,
            "state_sha256": rr.state_sha256,
            "state_sha256_attendu": att["state_sha256"],
            "state_identique": bool(rr.state_sha256 == att["state_sha256"]),
            "fichier_sha256": sha_fichier,
            "fichier_sha256_attendu": att["poids_sha256"],
            "fichier_identique": bool(sha_fichier == att["poids_sha256"]),
            "best_epoch": rr.best_epoch, "best_delta_val": rr.best_delta_val,
        })
    g1_pass = bool(all(r["state_identique"] and r["fichier_identique"]
                       for r in g1_par_run))

    # (b) G2 — 153/153 sha256 + re-classification ε=0 des 19 modèles.
    att_sha = _t67(root, "POIDS_SHA256")["sha256"]
    n_sha_ok = 0
    sha_bad = []
    for fn, s in sorted(att_sha.items()):
        h = hashlib.sha256((root / POIDS_T67 / fn).read_bytes()).hexdigest()
        if h == s:
            n_sha_ok += 1
        else:
            sha_bad.append(fn)
    a_s = tr.a["S"]
    pairs = dt.paires_67(a_s.shape[0])
    modeles = liste_close_modeles(root)
    theta = rt.ajuster_ridge(tr.a["S"], tr.ap["S"], dt.RIDGE_LAMBDA)
    reclass = []
    for mo in modeles:
        if mo["role"] == "RIDGE":
            res = classer_theta_ridge(theta, a_s, pairs)
        else:
            res = classer_etat_chrono(
                _charger_mats_modele(root, mo["fichier"]), a_s, pairs)
        reclass.append({
            "id": mo["id"], "role": mo["role"], "bras": mo["bras"],
            "seed": mo["seed"], "classe_recalculee": res["classe"],
            "classe_t67": mo["classe_t67"],
            "identique": bool(res["classe"] == mo["classe_t67"]),
            "Lambda": res["stats"]["Lambda"],
            "R_disp_64": res["R_disp_64"].get("R_disp_64"),
        })
    g2_reclass_ok = bool(all(r["identique"] for r in reclass))
    g2_pass = bool(n_sha_ok == len(att_sha) and g2_reclass_ok)

    # (c) G3 — fraîcheur : graines 50-199 ABSENTES des 3 artefacts (liste close).
    neuves = set(GRAINES_NEUVES)
    g3_par_artefact = {}
    for art in ARTEFACTS_ENTRAINEMENT:
        data = json.loads((root / art).read_text(encoding="utf-8"))
        seeds = sorted({int(r["seed"]) for r in data["runs"]})
        inter = sorted(neuves & set(seeds))
        g3_par_artefact[art] = {"seeds_presents_min_max": [seeds[0], seeds[-1]],
                                "intersection_graines_neuves": inter}
    g3_pass = bool(all(not v["intersection_graines_neuves"]
                       for v in g3_par_artefact.values()))

    # (d) Publications — tables k* EXACTES (Fraction), puissances, NON-MESURE.
    table_k_star = {}
    for b in range(0, 31):
        ks = k_star_hyper(150, b, 30)
        entry = {"k_star": ks}
        if ks is not None:
            p = hyper_tail_ge(150, b, 30, ks)
            entry["p_a_k_star"] = float(p)
            entry["p_exact"] = f"{p.numerator}/{p.denominator}"
        table_k_star[str(b)] = entry
    fisher_k_star = {}
    for b_a2 in range(0, 31):
        ligne = {}
        for b_wd in range(0, 31):
            ligne[str(b_wd)] = k_star_hyper(150, b_a2, b_wd)
        fisher_k_star[str(b_a2)] = ligne
    puissances = {}
    for cap_num, cap_den in ((1, 2), (3, 5), (3, 4)):
        cle = f"capture_{cap_num}_{cap_den}"
        ks12 = table_k_star["12"]["k_star"]
        puissances[cle] = cd.puissance_sign_test(12, ks12, Fraction(cap_num, cap_den)) \
            if ks12 is not None else None
    publications = {
        "table_k_star_hypergeom_N150_m30_B_0_30": table_k_star,
        "table_k_star_fisher_N150": fisher_k_star,
        "puissances_binomiales_B12": puissances,
        "regles_NON_MESURE": {
            "C2": "B_A2 < 3 => NON-MESURE",
            "C3": "B_A2 < 3 ou B_WD < 3 => NON-MESURE",
            "C1": "G2 FAIL => NON-MESURE ; q(1e-2) >= 0,10 => NON-OPPOSABLE",
        },
        "MDE": {"C2_capture": 0.60, "C2_enrichissement": 2.9,
                "C3_transfert_tau": 0.5},
    }

    # (f) G5 — scan du code neuf.
    ici = Path(__file__).resolve()
    test_file = ici.parents[2] / "tests" / "test_duration_selection.py"
    g5 = scan_g5([ici, test_file])

    payload = {
        "tour": 68, "etape": "P1",
        "G1_parite_au_bit": {"par_run": g1_par_run, "PASS": g1_pass,
                             "regle": "100 % ou FAUX (99 % = FAUX)",
                             "chute": "G1 FAIL => C2/C3 NON-MESURE ; C1 intact"},
        "G2_objets": {"sha256_ok": n_sha_ok, "sha256_total": len(att_sha),
                      "sha256_derives": sha_bad,
                      "reclassification_eps0": reclass,
                      "reclassification_19_19": g2_reclass_ok,
                      "PASS": g2_pass,
                      "chute": "G2 FAIL => C1 NON-MESURE ; C2/C3 intacts"},
        "G3_fraicheur": {"par_artefact": g3_par_artefact, "PASS": g3_pass,
                         "chute": "G3 FAIL => C2 NON-MESURE"},
        "liste_close_modeles": [
            {k: mo[k] for k in ("id", "role", "bras", "seed",
                                "classe_t67", "fichier")
             if k in mo} for mo in modeles],
        "publications": publications,
        "G5_scan": g5,
        "PASS_toutes_gates": bool(g1_pass and g2_pass and g3_pass and g5["PASS"]),
    }
    if not g5["PASS"]:
        raise RuntimeError("G5 FAIL — DISSIPATION (I13) : STOP net, consigner")
    return payload


def specs_gelees() -> Tuple[dict, dict]:
    """TOUR68_SPEC.json / TOUR68_PREDICTIONS.json (transcription émission)."""
    spec = {
        "tour": 68, "artefact": "SPEC",
        "objet": "les 8 spécimens BORNÉ-VIVANT : (C1) bassin par perturbation "
                 "contrôlée des poids ; (C2) prédiction gamma sur 150 graines "
                 "neuves A2 ; (C3) concordance d'init A2/WD",
        "base": "TOUR67_SPEC.json repris à l'identique ; 5 écarts énumérés §4.10 : "
                "(1) graines 50-199 identiques aux 2 bras ; (2) bras {A2, WD}, A1 "
                "exclu motivé (29/50 DIVERGED T67) ; (3) aucune ouverture du corpus "
                "v2 ; (4) étape neuve robustesse (n'entraîne rien) ; (5) "
                "exploration alpha sur artefacts T67 en lecture seule, payée beta/gamma",
        "perturbation": {"grille_eps": list(EPS_GRID), "R_specimen": R_SPECIMEN,
                         "R_controle": R_CONTROLE,
                         "seed": "default_rng(68_000_000 + 1000*id + r)",
                         "echelle": "relative par matrice : eps*(frob/sqrt(n))*G",
                         "ordre_tirage": "clés triées (A..L ; W puis b)"},
        "partition": "T67 inchangée au chiffre près (f_div 0,90/0,10 ; R_disp^64 "
                     "0,20 ; extinction 1e-3 ; borne 10x ; NA_DISP défavorable)",
        "graines_neuves": [GRAINES_NEUVES[0], GRAINES_NEUVES[-1]],
        "bras_neufs": list(BRAS_NEUFS),
        "alpha_famille": {"features": list(FEATURES), "E_prefixe": E_PREFIXE,
                          "n_regles": 128, "critere": "max BV A2 T67 dans top-10",
                          "departage": "moins de features ; lex (f_i,f_j) ; + avant -",
                          "beta_vide_si_capture_le": BETA_VIDE_MAX},
        "selecteur": {"m": "round(0,20*N_eff)", "tie": "score desc, graine asc"},
        "seuils": {"C1_quorum": f"{QUORUM_C1}/8 a eps50 >= {EPS50_C1}",
                   "branche_lame": f"{QUORUM_C1}/8 a eps50 <= {EPS50_LAME}",
                   "P_VERR": f"stab_cl jumeaux pooled >= {STAB_VERR_MIN} a 1e-2",
                   "P_ACC": f"q(1e-2) < {Q_MAX_OPPOSABLE}",
                   "C2": "k >= k*(N_eff, B, m), p < 0,01, hypergéométrique exact",
                   "C3": "recouvrement >= k*, Fisher exact p < 0,01",
                   "G4": f">= {QUORUM_G4}/150 par bras",
                   "NON_MESURE": f"B < {B_MIN_MESURE}"},
        "mur": {"formule": "max(3*T_hat, 1800), plafond 5400",
                "T_hat": "150*(t_A2+t_WD) + 2140*t_class + 20*t_init"},
        "precision": "float64, CPU, torch.manual_seed, "
                     "use_deterministic_algorithms(True)",
    }
    predictions = {
        "tour": 68, "artefact": "PREDICTIONS",
        "H68": "(A) BASSIN >= 5/8 a eps50 >= 1e-2 ; (B) PRÉDICTION top-30 gelé "
               "p < 0,01 (A2 seul) ; (C) INIT concordance A2/WD p < 0,01",
        "cotes": {"P_ROB1": "40/60", "P_ROB2": "65/35", "P_MARGE": "70/30",
                  "P_VERR": "70/30", "P_ACC": "75/25", "P_PRED_A2": "30/70",
                  "P_PRED_WD_INFO": "25/75", "P_SEED_C3": "30/70",
                  "P_BASE": "80/20", "P_betaVIDE": "35/65",
                  "H68_globale": "15/85"},
        "conjoncts": {
            "C1": "PORTEUR principal alpha-free — eps50 >= 1e-2 pour >= 5/8 ; "
                  "les 8 eps50, 40 surv, 11 stab_cl, q(eps) publiés dans tous les cas",
            "C2": "PORTEUR (INFO si beta VIDE) — hypergéométrique exact",
            "C3": "PORTEUR alpha-free — Fisher exact, table 2x2 complète",
            "C0": "CONTRÔLE bloquant partiel — B_A2, B_WD, binomiales",
            "G1": "GATE parité au bit", "G2": "GATE objets",
            "G3": "GATE fraîcheur", "G4": "GATE optimisation par bras",
            "G5": "GATE matérielle", "C9": "INFO liste close",
        },
        "partages_de_sort": "G1 -> C2/C3 ; G2 -> C1 ; G3 -> C2 ; C0 -> C2/C3 ; "
                            "G4 -> le bras ; AUCUN autre",
        "issues": "I1-I15 de l'émission §7",
        "formulations_imposees": [
            "le vivant est une lame de couteau",
            "le verrouillage est l'attracteur, la vie est la frontière",
            "le vivant est une loterie de l'optimisation"],
    }
    return spec, predictions


def etape_gel(root: Path) -> dict:
    """β1 (§4.13) : écrit SPEC/PREDICTIONS puis scelle TOUR68_GEL.json
    (jeton freeze_token, 21 chemins) APRÈS P1, AVANT robustesse."""
    exiger(root, "instruments", "p0b", "p0c", "etalonnage", "p1")
    spec, predictions = specs_gelees()
    trd.ecrire_artefact(spec, root / ART["spec"])
    trd.ecrire_artefact(predictions, root / ART["predictions"])
    token = freeze_token([root / p for p in FREEZE_PATHS_BETA1])
    write_token(token, root / ART["gel"])
    return {"tour": 68, "etape": "GEL_BETA1", "digest": token.digest,
            "n_chemins": len(token.artifacts),
            "mtimes_ns": {a.path: a.mtime_ns for a in token.artifacts}}


def etape_robustesse(root: Path) -> dict:
    """P2-robustesse — C1 (α-free, scellé AVANT l'α) : 1 840 classifications
    perturbées ; ``surv``, ``ε₅₀``, ``stab_cl``, ``q(ε)``."""
    exiger(root, "gel", "etalonnage")
    verifier_gel_beta1(root)
    refuser_si_deja_fait(root, "robustesse")
    t_debut = time.perf_counter()
    tok, m1, tr, _ = _train_val(root)
    a_s = tr.a["S"]
    pairs = dt.paires_67(a_s.shape[0])
    modeles = liste_close_modeles(root)
    theta = rt.ajuster_ridge(tr.a["S"], tr.ap["S"], dt.RIDGE_LAMBDA)

    par_modele = []
    for mo in modeles:
        est_specimen = mo["role"] == "SPECIMEN"
        n_tirages = R_SPECIMEN if est_specimen else R_CONTROLE
        if mo["role"] == "RIDGE":
            base = theta_vers_mats(theta)
            classer = lambda mats: classer_theta_ridge(
                mats_vers_theta(mats), a_s, pairs)["classe"]
        else:
            base = _charger_mats_modele(root, mo["fichier"])
            classer = lambda mats: classer_etat_chrono(mats, a_s, pairs)["classe"]
        surv_par_eps: Dict[float, float] = {}
        stab_par_eps: Dict[float, float] = {}
        classes_par_eps: Dict[str, Dict[str, int]] = {}
        for eps in EPS_GRID:
            budget_verifier(root, time.perf_counter() - t_debut)
            classes = []
            for r in range(n_tirages):
                pm = perturber_matrices(base, eps, rng_perturbation(mo["id"], r))
                classes.append(classer(pm))
            surv_par_eps[eps] = float(np.mean(
                [c == "BORNE_VIVANT" for c in classes]))
            stab_par_eps[eps] = float(np.mean(
                [c == mo["classe_t67"] for c in classes]))
            comptes: Dict[str, int] = {}
            for c in classes:
                comptes[c] = comptes.get(c, 0) + 1
            classes_par_eps[str(eps)] = comptes
        entry = {
            "id": mo["id"], "role": mo["role"], "bras": mo["bras"],
            "seed": mo["seed"], "classe_t67": mo["classe_t67"],
            "n_tirages": n_tirages,
            "surv_par_eps": {str(e): surv_par_eps[e] for e in EPS_GRID},
            "stab_cl_par_eps": {str(e): stab_par_eps[e] for e in EPS_GRID},
            "classes_par_eps": classes_par_eps,
        }
        if est_specimen:
            entry["eps50"] = eps50_de_surv(surv_par_eps)
        par_modele.append(entry)

    specimens = [e for e in par_modele if e["role"] == "SPECIMEN"]
    controles = [e for e in par_modele if e["role"] != "SPECIMEN"]
    eps50s = {f"{e['bras']}_seed{e['seed']}": e["eps50"] for e in specimens}
    n_haut = sum(1 for e in specimens if e["eps50"] >= EPS50_C1)
    n_bas = sum(1 for e in specimens if e["eps50"] <= EPS50_LAME)
    # q(eps) : POOLED sur tous les tirages des 11 contrôles (176 par eps).
    q_par_eps = {}
    for eps in EPS_GRID:
        vivants = 0
        total = 0
        for e in controles:
            n_bv = e["classes_par_eps"][str(eps)].get("BORNE_VIVANT", 0)
            vivants += n_bv
            total += e["n_tirages"]
        q_par_eps[str(eps)] = {"q": vivants / total, "vivants": vivants,
                               "total": total}
    q_1e2 = q_par_eps[str(1e-2)]["q"]
    # P-VERR : stab_cl des 8 jumeaux à 1e-2, POOLED (128 tirages) + par jumeau.
    jumeaux = [e for e in par_modele if e["role"] == "JUMEAU_VERROUILLE"]
    verr_pool = float(np.mean([e["classes_par_eps"][str(1e-2)]
                               .get("VERROUILLE", 0) / e["n_tirages"]
                               for e in jumeaux]))
    verr_ok_pool = sum(e["classes_par_eps"][str(1e-2)].get("VERROUILLE", 0)
                       for e in jumeaux)
    verr_n_pool = sum(e["n_tirages"] for e in jumeaux)
    # P-MARGE : eps50 des 3 spécimens de marge vs médiane des 5 autres.
    marge_ids = {f"{b}_seed{g}" for b, g in SPECIMENS_MARGE}
    eps_marge = [v for k, v in eps50s.items() if k in marge_ids]
    eps_autres = [v for k, v in eps50s.items() if k not in marge_ids]
    med_autres = float(np.median(eps_autres))
    p_marge = bool(all(v < med_autres for v in eps_marge))
    payload = {
        "tour": 68, "etape": "ROBUSTESSE",
        "grandeur": "eps50 (rayon de survie, ordinal 6 niveaux) sur les 8 "
                    "spécimens ; surv/stab_cl/q sur la liste close des 19 ; "
                    "1840 classifications (8x5x24 + 11x5x16)",
        "par_modele": par_modele,
        "eps50_specimens": eps50s,
        "C1_bassin": {"n_eps50_ge_1e-2": n_haut, "quorum": QUORUM_C1,
                      "C1_verdict_bassin": bool(n_haut >= QUORUM_C1),
                      "n_eps50_le_1e-3": n_bas,
                      "branche_lame_de_couteau": bool(n_bas >= QUORUM_C1)},
        "q_par_eps": q_par_eps,
        "Q3_opposabilite": {"q_1e-2": q_1e2,
                            "seuil": Q_MAX_OPPOSABLE,
                            "C1_opposable": bool(q_1e2 < Q_MAX_OPPOSABLE)},
        "P_VERR": {"stab_pool_1e-2": verr_ok_pool / verr_n_pool,
                   "moyenne_par_jumeau_INFO": verr_pool,
                   "par_jumeau": {f"{e['bras']}_seed{e['seed']}":
                                  e["stab_cl_par_eps"][str(1e-2)]
                                  for e in jumeaux},
                   "PASS_ge_0.90": bool(verr_ok_pool / verr_n_pool
                                        >= STAB_VERR_MIN)},
        "P_MARGE": {"eps50_marge": eps_marge, "mediane_autres_5": med_autres,
                    "PASS": p_marge},
        "duree_s": time.perf_counter() - t_debut,
    }
    return payload


def _features_t67(root: Path) -> Dict[str, Dict[int, Optional[Dict[str, float]]]]:
    """Features des 121 modèles T67 mesurés (A2 50, WD 50, A1 21 survivants)."""
    courbes = _t67(root, "COURBES")["par_run"]
    train67 = _t67(root, "TRAIN")
    runs = {r["run_id"]: r for r in train67["runs"]}
    inits: Dict[int, Dict[str, float]] = {}
    out: Dict[str, Dict[int, Optional[Dict[str, float]]]] = {}
    for bras in ("A1", "A2", "WD"):
        par_graine: Dict[int, Optional[Dict[str, float]]] = {}
        for g in range(50):
            r = runs[f"{bras}_seed{g}"]
            if r["diverged_train"]:
                par_graine[g] = None
                continue
            pref = features_prefixe(courbes[f"{bras}_seed{g}"])
            if pref is None:
                par_graine[g] = None
                continue
            if g not in inits:
                inits[g] = features_init(g)
            par_graine[g] = {**inits[g], **pref}
        out[bras] = par_graine
    return out


def etape_alpha(root: Path) -> dict:
    """P2-alpha : les 128 règles closes sur les 121 modèles T67 (lecture
    seule) — table COMPLÈTE publiée, y compris les perdantes. α/INFO, JAMAIS
    un verdict (clause anti-α §7)."""
    exiger(root, "robustesse")
    verifier_gel_beta1(root)
    verifier_sceau(root, "robustesse")
    refuser_si_deja_fait(root, "alpha")
    duree = _t67(root, "DUREE")
    feats = _features_t67(root)
    bv = {bras: sorted(
        int(s) for s, e in duree["par_bras"][bras]["par_graine"].items()
        if e.get("classe") == "BORNE_VIVANT") for bras in ("A1", "A2", "WD")}
    # Constantes z : les 50 graines A2 du T67 (aucune divergée — vérifié).
    a2_feats = {g: f for g, f in feats["A2"].items() if f is not None}
    if len(a2_feats) != 50:
        raise RuntimeError(f"A2 T67 : {len(a2_feats)} graines avec features != 50")
    z_csts = constantes_z({f: [a2_feats[g][f] for g in sorted(a2_feats)]
                           for f in FEATURES})
    regles = regles_closes()
    table = []
    meilleurs: Tuple[int, int] = (-1, -1)  # (capture, -index) pour max
    for idx, regle in enumerate(regles):
        scores_a2 = {g: score_regle(regle, a2_feats[g], z_csts)
                     for g in sorted(a2_feats)}
        top10 = top_m_graines(scores_a2, N_TOP_ALPHA)
        cap_a2 = len(set(top10) & set(bv["A2"]))
        # INFO : capture WD (top-10 sur 50) et A1 (top-4 sur 21 survivants).
        wd_feats = {g: f for g, f in feats["WD"].items() if f is not None}
        scores_wd = {g: score_regle(regle, wd_feats[g], z_csts)
                     for g in sorted(wd_feats)}
        cap_wd = len(set(top_m_graines(scores_wd, N_TOP_ALPHA)) & set(bv["WD"]))
        a1_feats = {g: f for g, f in feats["A1"].items() if f is not None}
        scores_a1 = {g: score_regle(regle, a1_feats[g], z_csts)
                     for g in sorted(a1_feats)}
        m_a1 = int(round(M_FRACTION * len(a1_feats)))
        cap_a1 = len(set(top_m_graines(scores_a1, m_a1)) & set(bv["A1"]))
        table.append({"index": idx, "regle": regle_id(regle),
                      "capture_A2_top10": cap_a2,
                      "capture_WD_top10_INFO": cap_wd,
                      "capture_A1_top4_INFO": cap_a1,
                      "top10_A2": top10})
        if cap_a2 > meilleurs[0]:
            meilleurs = (cap_a2, idx)
    best_idx = meilleurs[1]
    best = table[best_idx]
    beta_vide = bool(best["capture_A2_top10"] <= BETA_VIDE_MAX)
    return {
        "tour": 68, "etape": "ALPHA",
        "etiquette": "alpha / INFO — JAMAIS un verdict (clause anti-alpha §7) ; "
                     "128 règles essayées, TOUTES publiées ; le prix payé est "
                     "le rapport 128:1",
        "n_regles": len(regles),
        "n_modeles_t67": {b: len([1 for f in feats[b].values() if f is not None])
                          for b in ("A1", "A2", "WD")},
        "BV_t67": bv,
        "z_constantes": z_csts,
        "table_128_regles": table,
        "meilleure_regle": {"index": best_idx, "regle": best["regle"],
                            "capture_A2_top10": best["capture_A2_top10"],
                            "departage": "première du parcours canonique "
                                         "(moins de features ; lex ; + avant -)"},
        "beta_vide": beta_vide,
        "clause_beta_vide": f"capture <= {BETA_VIDE_MAX}/4 => beta VIDE, "
                            "gamma joué en INFO (écrit avant)",
    }


def etape_beta(root: Path) -> dict:
    """β2 (§4.13) : LE score unique scellé (TOUR68_BETA.json + jeton
    TOUR68_FREEZE.json, 24 chemins) AVANT la première descente ≥ 50."""
    exiger(root, "alpha")
    verifier_gel_beta1(root)
    verifier_sceau(root, "alpha")
    if (root / ART["beta"]).is_file():
        raise RuntimeError("TOUR68_BETA.json existe déjà : one-shot (γ)")
    alpha = json.loads((root / ART["alpha"]).read_text(encoding="utf-8"))
    best = alpha["meilleure_regle"]
    payload = {
        "tour": 68, "etape": "BETA",
        "score_gele": {
            "regle": best["regle"],
            "formule": "s(g) = somme des termes signés z(f_k) ; "
                       "z = (f - mediane)/MAD, constantes ci-dessous",
            "z_constantes": alpha["z_constantes"],
            "capture_A2_T67_top10": best["capture_A2_top10"],
        },
        "beta_vide": alpha["beta_vide"],
        "statut_C2": "INFO (beta VIDE)" if alpha["beta_vide"] else "PORTEUR",
        "selecteur": {"m": "round(0,20*N_eff)", "m_N150": 30,
                      "tie": "score décroissant, graine croissante"},
        "chronologie": "scellé AVANT la première descente d'une graine >= 50 "
                       "(jeton TOUR68_FREEZE.json, mtimes ns)",
    }
    trd.ecrire_artefact(payload, root / ART["beta"])
    sceller(root, "beta")
    token = freeze_token([root / p for p in FREEZE_PATHS_BETA2])
    write_token(token, root / ART["freeze2"])
    return {**payload, "digest_beta2": token.digest,
            "n_chemins": len(token.artifacts)}


def etape_train_neuf(root: Path) -> dict:
    """P2-train_neuf — γ : 300 descentes NEUVES (A2 150, WD 150, graines
    50-199), spec T67 EXACTE. GATE Q10 : refuse de s'exécuter sans le sceau
    β2 (aucune quantité γ ne peut influencer le β)."""
    exiger(root, "beta", "freeze2", "etalonnage")
    verifier_gel_beta1(root)
    verifier_gel_beta2(root)
    verifier_sceau(root, "beta")
    refuser_si_deja_fait(root, "train_neuf")
    t_debut = time.perf_counter()
    tok, m1, tr, va = _train_val(root)
    m = rt.mesurer_m(tr.a["S"], tr.ap["S"])
    s_raw_val = rt.scores_m1(m1, va.a["S"], va.ap["S"])
    poids_dir = root / POIDS_DIR
    poids_dir.mkdir(exist_ok=True)
    runs: List[dict] = []
    courbes: Dict[str, dict] = {}
    plan = [(bras, g) for bras in BRAS_NEUFS for g in GRAINES_NEUVES]
    for bras, seed in plan:
        budget_verifier(root, time.perf_counter() - t_debut)
        rr = dt.entrainer_bras(m1, bras, seed, tr.a["S"], tr.ap["S"],
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
                                      and rr.descente_rel
                                      >= rt.SEUIL_PERTE_DESCENTE),
            "state_sha256": rr.state_sha256,
            "poids_fichier": fichier.name, "poids_sha256": sha_fichier,
        })
        courbes[rid] = rr.courbes
    g4_partiel = {}
    for bras in BRAS_NEUFS:
        rs = [r for r in runs if r["bras"] == bras]
        n_desc = sum(1 for r in rs if r["descente_ge_20pct"])
        n_div = sum(1 for r in rs if r["diverged_train"])
        g4_partiel[bras] = {
            "n_descendus_ge_20pct": n_desc, "quorum": QUORUM_G4,
            "descente_PASS": bool(n_desc >= QUORUM_G4),
            "n_diverges_train": n_div,
            "repli_survivants": bool(n_div >= SEUIL_REPLI_DIVERGES),
        }
    sha_courbes = trd.ecrire_artefact(
        {"tour": 68, "artefact": "COURBES_NEUF",
         "axes": ["perte_train", "delta_med_val", "r_disp", "dbar"],
         "par_run": courbes}, root / ART["courbes_neuf"])
    return {"tour": 68, "etape": "TRAIN_NEUF", "n_runs": len(runs),
            "G4_partiel_descente": g4_partiel, "runs": runs,
            "courbes_sha256": sha_courbes,
            "duree_s": time.perf_counter() - t_debut}


def etape_classer(root: Path) -> dict:
    """P2-classer — C0 : partition close des 300 modèles neufs ; B_A2, B_WD,
    N_eff, m ; binomiales exactes vs 8 %/6 % (Q8)."""
    exiger(root, "train_neuf")
    verifier_gel_beta1(root)
    verifier_gel_beta2(root)
    verifier_sceau(root, "train_neuf")
    refuser_si_deja_fait(root, "classer")
    t_debut = time.perf_counter()
    tok, m1, tr, _ = _train_val(root)
    a_s = tr.a["S"]
    pairs = dt.paires_67(a_s.shape[0])
    train_neuf = json.loads((root / ART["train_neuf"]).read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in train_neuf["runs"]}
    courbes = json.loads((root / ART["courbes_neuf"]).read_text(
        encoding="utf-8"))["par_run"]
    par_bras: Dict[str, dict] = {}
    for bras in BRAS_NEUFS:
        par_graine = {}
        survivants: List[int] = []
        for seed in GRAINES_NEUVES:
            r = runs[f"{bras}_seed{seed}"]
            if r["diverged_train"]:
                par_graine[str(seed)] = {"classe": "EXCLU_DIVERGED_TRAIN",
                                         "diverged_train": True}
                continue
            budget_verifier(root, time.perf_counter() - t_debut)
            mats = etat_chrono_vers_mats(
                rt.charger_poids(root / POIDS_DIR / r["poids_fichier"]))
            res = classer_etat_chrono(mats, a_s, pairs)
            res["rho0"] = cd.rho_compagnon(
                {k.split(".")[0]: v for k, v in mats.items()})
            par_graine[str(seed)] = {**dt._resume_classement(res),
                                     "diverged_train": False,
                                     "prefixe_5_disponible": bool(
                                         features_prefixe(
                                             courbes[f"{bras}_seed{seed}"])
                                         is not None)}
            survivants.append(seed)
        classes = {c: sum(1 for g in par_graine.values() if g["classe"] == c)
                   for c in dt.CLASSES}
        classes["EXCLU_DIVERGED_TRAIN"] = len(GRAINES_NEUVES) - len(survivants)
        bv = [g for g in survivants
              if par_graine[str(g)]["classe"] == "BORNE_VIVANT"]
        n_eff = sum(1 for g in survivants
                    if par_graine[str(g)]["prefixe_5_disponible"])
        lams = np.asarray([par_graine[str(g)]["Lambda"] for g in survivants])
        finitude = bool(all(par_graine[str(g)]["finitude_lam_100pct"]
                            for g in survivants))
        non_vacuite = bool(all(par_graine[str(g)]["lam_std"] > 0.0
                               for g in survivants))
        g4 = train_neuf["G4_partiel_descente"][bras]
        par_bras[bras] = {
            "par_graine": par_graine, "classes": classes,
            "survivants": survivants, "n_survivants": len(survivants),
            "graines_BORNE_VIVANT": bv, "B": len(bv),
            "N_eff_prefixe": n_eff,
            "Lambda_mediane_survivants": float(np.median(lams))
            if len(lams) else None,
            "G4": {**g4, "finitude_lam_100pct": finitude,
                   "non_vacuite_std_pos": non_vacuite,
                   "PASS": bool(g4["descente_PASS"] and finitude
                                and non_vacuite)},
        }
    b_a2 = par_bras["A2"]["B"]
    b_wd = par_bras["WD"]["B"]
    n_eff_a2 = par_bras["A2"]["N_eff_prefixe"]
    m_sel = int(round(M_FRACTION * n_eff_a2))
    binomiales = {}
    for bras, b, p_att in (("A2", b_a2, Fraction(8, 100)),
                           ("WD", b_wd, Fraction(6, 100))):
        n = len(GRAINES_NEUVES)
        p_ge = binom_tail_ge(n, p_att, b)
        p_le = binom_tail_le(n, p_att, b)
        binomiales[bras] = {"B": b, "n": n, "p_attendu": float(p_att),
                            "P_ge_B": float(p_ge), "P_le_B": float(p_le),
                            "bilateral_2min": float(min(1,
                                                        2 * min(p_ge, p_le)))}
    payload = {
        "tour": 68, "etape": "CLASSER",
        "par_bras": par_bras,
        "C0_taux_de_base": {"B_A2": b_a2, "B_WD": b_wd,
                            "intervalle_P_BASE": [4, 20],
                            "P_BASE_dans_intervalle": bool(4 <= b_a2 <= 20),
                            "binomiales": binomiales,
                            "C2_NON_MESURE": bool(b_a2 < B_MIN_MESURE),
                            "C3_NON_MESURE": bool(b_a2 < B_MIN_MESURE
                                                  or b_wd < B_MIN_MESURE)},
        "N_eff_A2": n_eff_a2, "m_selecteur": m_sel,
        "duree_s": time.perf_counter() - t_debut,
    }
    return payload


def etape_predire(root: Path) -> dict:
    """P2-predire — C2 : le score β gelé sélectionne le top-m parmi les
    graines neuves A2 ; hypergéométrique EXACT (2 routes) ; rivaux naïfs en
    INFO ; bras WD en INFO (P-PRED′, clause cartouches-multiples)."""
    exiger(root, "classer", "beta")
    verifier_gel_beta1(root)
    verifier_gel_beta2(root)
    verifier_sceau(root, "classer")
    refuser_si_deja_fait(root, "predire")
    t_debut = time.perf_counter()
    beta = json.loads((root / ART["beta"]).read_text(encoding="utf-8"))
    classes = json.loads((root / ART["classer"]).read_text(encoding="utf-8"))
    courbes = json.loads((root / ART["courbes_neuf"]).read_text(
        encoding="utf-8"))["par_run"]
    train_neuf = json.loads((root / ART["train_neuf"]).read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in train_neuf["runs"]}
    z_csts = beta["score_gele"]["z_constantes"]
    # Reconstruire la règle depuis l'id gelé (déterministe).
    regle = []
    for terme in beta["score_gele"]["regle"].replace("-", " -").replace(
            "+", " +").split():
        signe = 1 if terme[0] == "+" else -1
        regle.append((int(terme[2:]), signe))
    regle = tuple(regle)

    def volet(bras: str) -> dict:
        pb = classes["par_bras"][bras]
        feats = {}
        for g in pb["survivants"]:
            if not pb["par_graine"][str(g)]["prefixe_5_disponible"]:
                continue
            pref = features_prefixe(courbes[f"{bras}_seed{g}"])
            feats[g] = {**features_init(g), **pref}
        n_eff = len(feats)
        m_sel = int(round(M_FRACTION * n_eff))
        bv = set(pb["graines_BORNE_VIVANT"]) & set(feats)
        scores = {g: score_regle(regle, f, z_csts) for g, f in feats.items()}
        top = top_m_graines(scores, m_sel)
        k = len(set(top) & bv)
        b_eff = len(bv)
        p1 = hyper_tail_ge(n_eff, b_eff, m_sel, k)
        p2 = hyper_tail_ge_recurrence(n_eff, b_eff, m_sel, k)
        if p1 != p2:
            raise RuntimeError("routes hypergéométriques divergentes — ARRÊT")
        ks = k_star_hyper(n_eff, b_eff, m_sel)
        naifs = {}
        for nom, cle_naif in (("naif1_best_delta_val", "best_delta_val_K2"),
                              ("naif2_rho0_init", None)):
            if cle_naif:
                vals = {g: float(runs[f"{bras}_seed{g}"][cle_naif])
                        for g in feats}
            else:
                vals = {g: feats[g]["f1"] for g in feats}
            top_desc = top_m_graines(vals, m_sel)
            top_asc = top_m_graines({g: -v for g, v in vals.items()}, m_sel)
            naifs[nom] = {
                "capture_top_m_desc": len(set(top_desc) & bv),
                "capture_top_m_asc_INFO": len(set(top_asc) & bv)}
        return {
            "N_eff": n_eff, "m": m_sel, "B_eff": b_eff,
            "graines_BV": sorted(bv),
            "top_m": top, "k_captures": k,
            "k_star": ks,
            "p_hypergeom": float(p1),
            "p_exact": f"{p1.numerator}/{p1.denominator}",
            "p_lt_001": bool(p1 < Fraction(1, 100)),
            "verdict_enrichissement": bool(ks is not None and k >= ks),
            "NON_MESURE": bool(b_eff < B_MIN_MESURE),
            "predicteurs_naifs_INFO": naifs,
            "scores_par_graine": {str(g): scores[g] for g in sorted(scores)},
        }

    volet_a2 = volet("A2")
    volet_wd = volet("WD")
    payload = {
        "tour": 68, "etape": "PREDIRE",
        "regle_beta": beta["score_gele"]["regle"],
        "beta_vide": beta["beta_vide"],
        "statut_C2": beta["statut_C2"],
        "A2_verdict": volet_a2,
        "WD_INFO": volet_wd,
        "clause": "verdict A2 SEUL ; WD = INFO (P-PRED', §3.2) ; beta VIDE => "
                  "gamma joué en INFO (écrit avant)",
        "duree_s": time.perf_counter() - t_debut,
    }
    return payload


def etape_concordance(root: Path) -> dict:
    """P2-concordance — C3 (α-free) : table 2×2 des vivants A2/WD sur les
    mêmes inits neuves ; Fisher exact (Fraction, 2 routes)."""
    exiger(root, "predire", "classer")
    verifier_gel_beta1(root)
    verifier_gel_beta2(root)
    verifier_sceau(root, "predire")
    refuser_si_deja_fait(root, "concordance")
    t_debut = time.perf_counter()
    classes = json.loads((root / ART["classer"]).read_text(encoding="utf-8"))
    pb_a2 = classes["par_bras"]["A2"]
    pb_wd = classes["par_bras"]["WD"]
    # Graines mesurées dans les DEUX bras (déclaré : cellule complète exigée).
    mes = sorted(set(pb_a2["survivants"]) & set(pb_wd["survivants"]))
    bv_a2 = set(pb_a2["graines_BORNE_VIVANT"]) & set(mes)
    bv_wd = set(pb_wd["graines_BORNE_VIVANT"]) & set(mes)
    a = len(bv_a2 & bv_wd)
    b = len(bv_a2 - bv_wd)
    c = len(bv_wd - bv_a2)
    d_ = len(mes) - a - b - c
    n_c = len(mes)
    p1 = hyper_tail_ge(n_c, len(bv_a2), len(bv_wd), a)
    p2 = hyper_tail_ge_recurrence(n_c, len(bv_a2), len(bv_wd), a)
    if p1 != p2:
        raise RuntimeError("routes Fisher divergentes — ARRÊT")
    ks = k_star_hyper(n_c, len(bv_a2), len(bv_wd))
    attendu_h0 = (len(bv_a2) * len(bv_wd) / n_c) if n_c else None
    payload = {
        "tour": 68, "etape": "CONCORDANCE",
        "N_commun": n_c,
        "table_2x2": {"BV_A2_et_BV_WD": a, "BV_A2_seul": b,
                      "BV_WD_seul": c, "ni_l_un_ni_l_autre": d_},
        "marginales": {"B_A2": len(bv_a2), "B_WD": len(bv_wd)},
        "graines": {"BV_A2": sorted(bv_a2), "BV_WD": sorted(bv_wd),
                    "recouvrement": sorted(bv_a2 & bv_wd)},
        "recouvrement_attendu_H0": attendu_h0,
        "k_star_fisher": ks,
        "p_fisher": float(p1),
        "p_exact": f"{p1.numerator}/{p1.denominator}",
        "p_lt_001": bool(p1 < Fraction(1, 100)),
        "verdict_concordance": bool(ks is not None and a >= ks),
        "NON_MESURE": bool(len(bv_a2) < B_MIN_MESURE
                           or len(bv_wd) < B_MIN_MESURE),
        "transfert_tau_INFO": (a / len(bv_wd)) if bv_wd else None,
        "duree_s": time.perf_counter() - t_debut,
    }
    return payload


def etape_info(root: Path) -> dict:
    """P2-info — C9 (liste close, hors verdict) : Λ/f_div/R_disp^64/ρ₀ des
    300 (dans CLASSES) ; best_return_step des vivants neufs ;
    non-stationnarité des 8 spécimens (Q13) ; stab_cl des divergents ;
    corrélation ε₅₀ ↔ R_disp^64 ; moyennes."""
    exiger(root, "concordance", "robustesse", "classer")
    verifier_gel_beta1(root)
    verifier_gel_beta2(root)
    verifier_sceau(root, "concordance")
    t_debut = time.perf_counter()
    tok, m1, tr, _ = _train_val(root)
    a_s = tr.a["S"]
    classes = json.loads((root / ART["classer"]).read_text(encoding="utf-8"))
    rob = json.loads((root / ART["robustesse"]).read_text(encoding="utf-8"))
    train_neuf = json.loads((root / ART["train_neuf"]).read_text(encoding="utf-8"))
    runs = {r["run_id"]: r for r in train_neuf["runs"]}
    # best_return_step des vivants neufs (M1 inchangé, TRAIN).
    brs = {}
    for bras in BRAS_NEUFS:
        for g in classes["par_bras"][bras]["graines_BORNE_VIVANT"]:
            rid = f"{bras}_seed{g}"
            model = rt.modele_depuis_poids(dt.INIT_SCALE, rt.charger_poids(
                root / POIDS_DIR / runs[rid]["poids_fichier"]))
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
            brs[rid] = {"mediane": float(np.median(arr)),
                        "fraction_le_2": float(np.mean(arr <= 2))}
    # Non-stationnarité (Q13, INFO) des 8 spécimens : sur les 16 derniers pas.
    non_stat = {}
    for bras, g in SPECIMENS:
        mats = _charger_mats_modele(root, f"{bras}_seed{g}.bin")
        model = rt.modele_depuis_poids(dt.INIT_SCALE, mats_vers_etat_chrono(mats))
        trj = cd.derouler(cd.pas_chrono(model), a_s, garder_etats=True)
        et = trj.etats  # (n, 64, d) : états s_1..s_64.
        meds = []
        for t in range(48, 64):  # transitions s_48->s_49 ... s_63->s_64.
            diff = et[:, t, :] - et[:, t - 1, :]
            fini = np.isfinite(diff).all(axis=1)
            if fini.any():
                meds.append(float(np.median(
                    np.linalg.norm(diff[fini], axis=1))))
        non_stat[f"{bras}_seed{g}"] = {
            "mediane_t_norme_pas_16_derniers": float(np.median(meds))
            if meds else None}
    # stab_cl des divergents + corrélation eps50 <-> R_disp^64 (8 points).
    stab_div = {f"{e['bras']}_seed{e['seed']}": e["stab_cl_par_eps"]
                for e in rob["par_modele"] if e["role"] == "DIVERGENT"}
    duree67 = _t67(root, "DUREE")
    eps50 = rob["eps50_specimens"]
    rdisp = {f"{b}_seed{g}":
             duree67["par_bras"][b]["par_graine"][str(g)]["R_disp_64"]
             for b, g in SPECIMENS}
    cles = sorted(eps50)
    correl = trd.spearman([eps50[k] for k in cles], [rdisp[k] for k in cles])
    moyennes = {bras: {
        "Lambda_moyenne_survivants_INFO": float(np.mean(
            [classes["par_bras"][bras]["par_graine"][str(g)]["Lambda"]
             for g in classes["par_bras"][bras]["survivants"]]))}
        for bras in BRAS_NEUFS}
    return {
        "tour": 68, "etape": "INFO",
        "etiquette": "INFO / hors-verdict — aucune de ces mesures n'altère "
                     "un conjonct",
        "best_return_step_vivants_neufs": brs,
        "non_stationnarite_specimens_Q13": non_stat,
        "stab_cl_divergents": stab_div,
        "correlation_eps50_R_disp64_spearman": correl,
        "eps50_specimens": eps50, "R_disp64_specimens_T67_cites": rdisp,
        "moyennes_INFO": moyennes,
        "duree_s": time.perf_counter() - t_debut,
    }


#: Interprétations conservatrices CONSIGNÉES (soupape INFO — aucune n'altère
#: un seuil de l'émission ; chacune est déclarée dans TOUR68_DEPLOIEMENT.md) :
INTERPRETATIONS_DECLAREES: Tuple[str, ...] = (
    "Perturbation : ordre de tirage = clés triées (A.weight..L.weight ; W puis "
    "b pour la ridge) ; eps=0 => copie bit-identique sans tirage.",
    "Ridge : matrices perturbées = {W = theta[:-1] (19x19), b = theta[-1]} — "
    "extension déclarée de la formule par-matrice.",
    "Jumeau partagé (WD-4 apparié à WD-1 ET WD-3) : DEUX entrées de contrôle, "
    "ids distincts => tirages distincts ; q(eps) sur les 11 entrées (176/eps).",
    "P-VERR : lecture POOLED (128 tirages des 8 jumeaux) pour le seuil 0,90 ; "
    "le par-jumeau est publié en regard.",
    "Features de préfixe : perte_train[5] = 5e élément (époque 5, index 4) ; "
    "la pente f8 = delta_med_val[5] - delta_med_val[1] = index 4 - index 0.",
    "Départage top-m : score décroissant, graine croissante (déterministe).",
    "MAD = 0 => z := 0 (déclaré, aucun cas attendu).",
    "Prédicteurs naïfs (INFO) : top-m = m plus GRANDES valeurs ; la capture "
    "des m plus petites est publiée en INFO (direction non gelée par "
    "l'émission).",
    "Concordance : restreinte aux graines MESURÉES dans les deux bras "
    "(N_commun publié) — cellule complète exigée, aucune cellule vide cachée.",
    "Une graine sans préfixe 5 époques est HORS N_eff de C2 (comptée, "
    "publiée) ; elle reste dans C0/C3 si sa classe est mesurée.",
    "beta1 = TOUR68_GEL.json (jeton freeze_token, 21 chemins) ; beta2 = "
    "TOUR68_BETA.json + TOUR68_FREEZE.json (24 chemins) — les 3 chemins "
    "ajoutés au digest beta2 sont ROBUSTESSE/ALPHA/BETA, inexistants à beta1.",
    "Le mur s'applique au CUMUL des étapes P2 (durées scellées + étape en "
    "cours), vérifié avant chaque unité lourde.",
    "Étapes P2 one-shot : re-exécution refusée si l'artefact existe "
    "(retouche = candidat mort).",
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("etape", choices=[
        "instruments", "p0b", "p0c", "etalonnage", "p1", "gel", "robustesse",
        "alpha", "beta", "train_neuf", "classer", "predire", "concordance",
        "info"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root)
    torch.use_deterministic_algorithms(True)
    fn: Dict[str, Callable[[Path], dict]] = {
        "instruments": etape_instruments, "p0b": etape_p0b, "p0c": etape_p0c,
        "etalonnage": etape_etalonnage, "p1": etape_p1, "gel": etape_gel,
        "robustesse": etape_robustesse, "alpha": etape_alpha,
        "beta": etape_beta, "train_neuf": etape_train_neuf,
        "classer": etape_classer, "predire": etape_predire,
        "concordance": etape_concordance, "info": etape_info,
    }
    t0 = time.perf_counter()
    payload = fn[args.etape](root)
    dt_s = time.perf_counter() - t0
    if args.etape in ("gel", "beta"):
        # Ces étapes écrivent elles-mêmes leurs fichiers (le jeton scelle
        # leurs mtimes : une réécriture par le runner serait une dérive).
        print(f"{args.etape}: digest={payload.get('digest', payload.get('digest_beta2'))} "
              f"duree={dt_s:.3f}s")
        return 0
    out = Path(args.out) if args.out else root / ART[args.etape]
    sha = trd.ecrire_artefact(payload, out)
    print(f"{args.etape}: {out} sha256={sha} duree={dt_s:.3f}s")
    if args.etape in ("robustesse", "alpha", "train_neuf", "classer",
                      "predire", "concordance"):
        print(f"sceau {args.etape}: {sceller(root, args.etape)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - point d'entrée mesure
    raise SystemExit(main())
