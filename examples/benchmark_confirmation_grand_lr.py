"""benchmark_confirmation_grand_lr.py — H4 : AlphaOmegaOpt garde-t-il sa qualité à grand lr ?

Protocole figé AVANT toute mesure : ``docs/PROTOCOLE_CONFIRMATION_GRAND_LR.md`` (son
sha256 et son blob git sont gravés dans le rapport).

Le §7.2 de ``docs/CHANTIER_OPTIMISEUR_ABA_RESULTATS.md`` a vu, APRÈS lecture des verdicts
de la grille FENETRE_LR, qu'à lr ≥ 0,1 AlphaOmegaOpt fait mieux que le témoin à montée
courte dans 54 runs appariés sur 54. Une observation post hoc ne vaut rien tant qu'elle
n'est pas confirmée sur des données qui ne l'ont pas suggérée : ce banc rejoue la même
comparaison sur des graines NEUVES (3, 4, 5), avec le même banc T64 (modèle, perte,
splits TRAIN 101-1000 / VALIDATION 3001-3500, TEST 3501-5001 JAMAIS ouvert, boucle
``train_run_candidate``, porte G0 de parité au bit).

Grille : 2 optimiseurs × 2 cartouches × 3 init × 3 lr (0,1 ; 0,3 ; 1) × 3 graines = 108 runs.

Définitions opérationnelles :
  * une **unité** = (cartouche, init, graine), 18 unités ;
  * ``D(unité)`` = moyenne, sur les 3 lr, de ``best_delta_val(AO) − best_delta_val(témoin)``
    (médiane des Δ de VALIDATION à la meilleure époque ; l'identité vaut 0 exact) ;
  * **H4** : médiane de D > 0 et sign test exact bilatéral p < 0,01 (15 unités sur 18) ;
  * **H4m** (mécanisme) : le gain g d'AO à la fin de sa meilleure époque est plus
    petit à lr = 1 qu'à lr = 0,1 : par unité, ``log g(0,1) − log g(1)``, médiane > 0 et
    sign test exact p < 0,01. Le retour baisse g quand le lr nominal monte.

Pour H4m, ``AlphaOmegaOptJournal`` relit g et ω̄ après chaque pas, sans rien calculer :
sa trajectoire est celle d'``AlphaOmegaOpt`` au bit (test).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from spiraton.experimental.alpha_omega_optimizer import AlphaOmegaOpt


def _charger(nom: str) -> ModuleType:
    """Un banc voisin (``examples/`` n'est pas un paquet), chargé par son chemin."""
    if nom in sys.modules:
        return sys.modules[nom]
    chemin = Path(__file__).resolve().parent / f"{nom}.py"
    spec = importlib.util.spec_from_file_location(nom, chemin)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    spec.loader.exec_module(module)
    return module


fen = _charger("benchmark_fenetre_lr")
banc = fen.banc
rt = fen.rt

PROTOCOLE = Path(__file__).resolve().parents[1] / "docs" / "PROTOCOLE_CONFIRMATION_GRAND_LR.md"

LRS_H4: Tuple[float, ...] = (1e-1, 3e-1, 1.0)
INITS_H4: Tuple[float, ...] = fen.INITS_FENETRE
GRAINES_H4: Tuple[int, ...] = (3, 4, 5)
CANDIDATS_H4: Tuple[str, ...] = ("adamw_montee_courte_clipped", "alpha_omega")
SEUIL_P = rt.SEUIL_P  # 0,01, bilatéral

#: Porte de rejeu (INFO) : deux runs de la grille FENETRE_LR (cartouche S, init 0,6,
#: lr 1, graine 0), dont le rapport scellé ``ecb4279a…`` a gravé l'état final. Les
#: rejouer avec ce banc vérifie que le journal ne touche pas la trajectoire, sur la
#: machine même de la mesure. Sans objet en synthétique.
REJEU_ATTENDU: Dict[str, str] = {
    "adamw_montee_courte_clipped": "d750fc6e684c571668988ad82932c50515c7bc8f497abcedcea0520f703892cb",
    "alpha_omega": "9e73622cb9f41482545f4659cb89f61c6e771b6ddb1d77b97810abe32c48a002",
}
REJEU_TACHE = ("S", 0.6, 1.0, 0)


class AlphaOmegaOptJournal(AlphaOmegaOpt):
    """``AlphaOmegaOpt`` qui garde, après chaque pas, le gain g et la moyenne ω̄ du
    premier groupe de paramètres. Il ne fait que relire l'état : la trajectoire des
    poids est identique au bit à celle d'``AlphaOmegaOpt``."""

    def __init__(self, params, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.journal_g: List[float] = []
        self.journal_omega_bar: List[float] = []

    def step(self, closure=None):
        loss = super().step(closure)
        groupe = self.param_groups[0]
        self.journal_g.append(float(groupe["g"]))
        self.journal_omega_bar.append(float(groupe["omega_bar"]))
        return loss


#: Le dernier optimiseur construit par ``candidats_h4`` dans ce processus : un run est
#: séquentiel, on relit donc son journal juste après ``train_run_candidate``.
_DERNIER_OPT: List[torch.optim.Optimizer] = []


def _faire_ao_journal(params, lr: float) -> AlphaOmegaOptJournal:
    opt = AlphaOmegaOptJournal(params, lr=lr, betas=rt.ADAM_BETAS, eps=rt.ADAM_EPS,
                               weight_decay=1e-2)
    _DERNIER_OPT[:] = [opt]
    return opt


def candidats_h4() -> Tuple["banc.CandidateSpec", ...]:
    """Le témoin de FENETRE_LR tel quel, et AlphaOmegaOpt journalisé (mêmes réglages)."""
    fenetre = {s.nom: s for s in fen.candidats()}
    ao = fenetre["alpha_omega"]
    return (
        fenetre["adamw_montee_courte_clipped"],
        banc.CandidateSpec(nom=ao.nom, description=ao.description + " ; g et ω̄ journalisés",
                           clip_norm=ao.clip_norm, make_opt=_faire_ao_journal),
    )


def fins_d_epoque(journal: Sequence[float], pas_par_epoque: int, epoques: int) -> List[float]:
    """Valeur du journal à la fin de chacune des ``epoques`` premières époques."""
    return [float(journal[e * pas_par_epoque - 1]) for e in range(1, epoques + 1)]


# --- Runs ----------------------------------------------------------------------------

def _un_run(tache: Tuple[str, str, float, float, int, int]) -> Dict[str, object]:
    """Un run ; pour AO, ajoute g et ω̄ en fin d'époque et g à la meilleure époque."""
    nom, cart, init, lr, seed, epochs_max = tache
    spec = {s.nom: s for s in candidats_h4()}[nom]
    d = fen._DONNEES
    rr = banc.train_run_candidate(
        d["m1"], spec, cart, init, lr, seed, d["a_tr"][cart], d["ap_tr"][cart],
        d["a_val"][cart], d["ap_val"][cart], d["s_raw"][cart], d["m"][cart],
        epochs_max=epochs_max,
    )
    out: Dict[str, object] = {
        "optimiseur": nom, "cartouche": cart, "init_scale": init, "lr": lr, "seed": seed,
        "diverged": rr.diverged, "best_delta_val": rr.best_delta_val,
        "apprend": bool((not rr.diverged) and math.isfinite(rr.best_delta_val)
                        and rr.best_delta_val > 0.0),
        "epochs_run": rr.epochs_run, "best_epoch": rr.best_epoch,
        "early_stopped": rr.early_stopped, "state_sha256": rr.state_sha256,
    }
    if nom == "alpha_omega":
        opt = _DERNIER_OPT[0]
        pas = math.ceil(d["a_tr"][cart].shape[0] / rt.BATCH)
        g_ep = fins_d_epoque(opt.journal_g, pas, rr.epochs_run)
        w_ep = fins_d_epoque(opt.journal_omega_bar, pas, rr.epochs_run)
        g_best = g_ep[rr.best_epoch - 1] if rr.best_epoch >= 1 else float("nan")
        out.update({"g_fin_epoque": g_ep, "omega_bar_fin_epoque": w_ep,
                    "g_meilleure_epoque": g_best, "lr_effectif_meilleure_epoque": lr * g_best})
    return out


def taches(epochs_max: int, cartouches: Sequence[str], inits: Sequence[float],
           lrs: Sequence[float], graines: Sequence[int]) -> List[tuple]:
    return fen.taches(CANDIDATS_H4, cartouches, inits, lrs, graines, epochs_max)


def rejeu(epochs_max: int) -> Dict[str, object]:
    """Porte de rejeu (INFO) : état final des deux runs de référence contre le sceau."""
    cart, init, lr, seed = REJEU_TACHE
    res = {}
    for nom, attendu in REJEU_ATTENDU.items():
        obtenu = _un_run((nom, cart, init, lr, seed, epochs_max))["state_sha256"]
        res[nom] = {"attendu": attendu, "obtenu": obtenu, "identique": obtenu == attendu}
    return res


# --- Lecture des hypothèses ----------------------------------------------------------

def _diff(a: float, b: float) -> float:
    """``a − b`` où un ``best_delta_val`` non fini vaut −∞ ; −∞ − (−∞) vaut 0."""
    a = a if math.isfinite(a) else -math.inf
    b = b if math.isfinite(b) else -math.inf
    if a == b:
        return 0.0
    return a - b


def ecarts_par_unite(runs: List[Dict[str, object]], a: str, b: str,
                     lrs: Sequence[float]) -> Dict[Tuple[str, float, int], float]:
    """``D(unité)`` : moyenne sur ``lrs`` de ``best_delta_val(a) − best_delta_val(b)``.

    Une moyenne indéterminée (+∞ et −∞ mêlés) vaut 0 : l'unité ne compte pas."""
    idx = {(r["optimiseur"], r["cartouche"], r["init_scale"], r["seed"], r["lr"]): r for r in runs}
    unites = sorted({(r["cartouche"], r["init_scale"], r["seed"]) for r in runs})
    out = {}
    for c, i, s in unites:
        ds = [_diff(idx[(a, c, i, s, lr)]["best_delta_val"], idx[(b, c, i, s, lr)]["best_delta_val"])
              for lr in lrs]
        m = float(np.mean(ds))
        out[(c, i, s)] = 0.0 if math.isnan(m) else m
    return out


def ecarts_log_g(runs: List[Dict[str, object]], lr_bas: float,
                 lr_haut: float) -> Dict[Tuple[str, float, int], float]:
    """Par unité : ``log g(lr_bas) − log g(lr_haut)`` pour AO, g pris à la fin de sa
    meilleure époque ; un g non fini (aucune époque gardée) donne 0 : l'unité ne compte pas."""
    idx = {(r["cartouche"], r["init_scale"], r["seed"], r["lr"]): r
           for r in runs if r["optimiseur"] == "alpha_omega"}
    out = {}
    for u in sorted({k[:3] for k in idx}):
        gb, gh = idx[u + (lr_bas,)]["g_meilleure_epoque"], idx[u + (lr_haut,)]["g_meilleure_epoque"]
        ok = math.isfinite(gb) and math.isfinite(gh) and gb > 0.0 and gh > 0.0
        out[u] = math.log(gb) - math.log(gh) if ok else 0.0
    return out


def lire_hypotheses(runs: List[Dict[str, object]], lrs: Sequence[float]) -> Dict[str, object]:
    d = ecarts_par_unite(runs, "alpha_omega", "adamw_montee_courte_clipped", lrs)
    diff = np.asarray([d[u] for u in sorted(d)], dtype=np.float64)
    st = rt.sign_test_exact(diff)
    med = float(np.median(diff))
    eg = ecarts_log_g(runs, min(lrs), max(lrs))
    diff_g = np.asarray([eg[u] for u in sorted(eg)], dtype=np.float64)
    st_g = rt.sign_test_exact(diff_g)
    med_g = float(np.median(diff_g))

    def _median(vals: List[float]) -> float:
        vals = [v for v in vals if math.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    qualite = {n: {c: {repr(lr): _median([r["best_delta_val"] for r in runs
                                          if r["optimiseur"] == n and r["cartouche"] == c
                                          and r["lr"] == lr])
                       for lr in lrs}
                   for c in sorted({r["cartouche"] for r in runs})}
               for n in CANDIDATS_H4}
    ao = [r for r in runs if r["optimiseur"] == "alpha_omega"]
    return {
        "H4_alpha_omega_vs_montee_courte_grand_lr": {
            "unites": len(diff), "mediane_D": med, "moyenne_D_INFO": float(diff.mean()),
            "sign_test_exact": st, "confirme_p001": bool(med > 0.0 and st["p"] < SEUIL_P),
        },
        "H4m_g_baisse_quand_lr_monte": {
            "lr_bas": min(lrs), "lr_haut": max(lrs), "unites": len(diff_g),
            "mediane_log_g_bas_moins_haut": med_g, "sign_test_exact": st_g,
            "confirme_p001": bool(med_g > 0.0 and st_g["p"] < SEUIL_P),
        },
        "H_NaN_divergences_non_finies": {
            n: int(sum(r["diverged"] for r in runs if r["optimiseur"] == n)) for n in CANDIDATS_H4},
        "INFO_runs_qui_apprennent": {
            n: int(sum(r["apprend"] for r in runs if r["optimiseur"] == n)) for n in CANDIDATS_H4},
        "INFO_mediane_best_delta_val": qualite,
        "INFO_g_meilleure_epoque_mediane": {
            repr(lr): _median([r["g_meilleure_epoque"] for r in ao if r["lr"] == lr]) for lr in lrs},
        "INFO_lr_effectif_meilleure_epoque_mediane": {
            repr(lr): _median([r["lr_effectif_meilleure_epoque"] for r in ao if r["lr"] == lr])
            for lr in lrs},
        "INFO_meilleure_epoque_mediane": {
            n: {repr(lr): _median([float(r["best_epoch"]) for r in runs
                                   if r["optimiseur"] == n and r["lr"] == lr]) for lr in lrs}
            for n in CANDIDATS_H4},
    }


# --- Contexte d'exécution (doctrine T74) ---------------------------------------------

def blob_git_lf(p: Path) -> Optional[str]:
    """sha1 du blob git du fichier, fins de ligne ramenées à LF : identique sous Windows
    (``autocrlf``) et sous Linux, et égal à ``git rev-parse <commit>:<chemin>``."""
    if not p.is_file():
        return None
    data = p.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def contexte(root: Path) -> Dict[str, object]:
    ctx = fen.contexte(root)
    ctx["protocole_sha256"] = fen._sha_fichier(PROTOCOLE)
    ctx["protocole_blob_git_lf"] = blob_git_lf(PROTOCOLE)
    return ctx


def executer(root: Path, *, synthetic: bool, epochs_max: int, cartouches: Sequence[str],
             inits: Sequence[float], lrs: Sequence[float], graines: Sequence[int],
             jobs: int = 1) -> Dict[str, object]:
    t0 = time.perf_counter()
    fen._init_processus(str(root), synthetic)
    d = fen._DONNEES
    parite = banc.verifier_parite_t64(
        d["m1"], banc.candidats_optimiseurs()[0], d["a_tr"]["S"], d["ap_tr"]["S"],
        d["a_val"]["S"], d["ap_val"]["S"], d["s_raw"]["S"], d["m"]["S"],
        epochs_check=min(15, epochs_max),
    )
    porte_rejeu = ({"sans_objet": "données synthétiques"} if synthetic
                   else rejeu(epochs_max))
    liste = taches(epochs_max, cartouches, inits, lrs, graines)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=fen._init_processus,
                                 initargs=(str(root), synthetic)) as ex:
            runs = list(ex.map(_un_run, liste, chunksize=2))
    else:
        runs = [_un_run(t) for t in liste]
    return {
        "chantier": "OPTIMISEUR_ABA_CONFIRMATION_GRAND_LR",
        "mode_donnees": d["mode"],
        "contexte": contexte(root),
        "grille": {"lrs": list(lrs), "inits": list(inits), "graines": list(graines),
                   "cartouches": list(cartouches), "epochs_max": epochs_max},
        "candidats": {s.nom: s.description for s in candidats_h4()},
        "parite_bit_a_bit_T64": parite,
        "porte_rejeu_INFO": porte_rejeu,
        "hypotheses": lire_hypotheses(runs, lrs),
        "runs": runs,
        "duree_totale_s_INFO": time.perf_counter() - t0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="racine contenant dataset_aba.txt et Tokenizer/ (défaut : .)")
    parser.add_argument("--out", type=Path, default=Path("BENCH_CONFIRMATION_GRAND_LR.json"))
    parser.add_argument("--synthetic", action="store_true",
                        help="données 19D synthétiques (calibration, jamais une mesure)")
    parser.add_argument("--quick", action="store_true",
                        help="30 époques, cartouche S, init 0,6, graine 3 (test du script)")
    parser.add_argument("--jobs", type=int, default=1, help="processus de calcul parallèles")
    args = parser.parse_args(argv)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if args.quick:
        grille = dict(epochs_max=30, cartouches=("S",), inits=(0.6,), lrs=LRS_H4,
                      graines=GRAINES_H4[:1])
    else:
        grille = dict(epochs_max=rt.EPOCHS_MAX, cartouches=rt.CARTOUCHES, inits=INITS_H4,
                      lrs=LRS_H4, graines=GRAINES_H4)
    payload = executer(args.root, synthetic=args.synthetic, jobs=args.jobs, **grille)
    hyp = payload["hypotheses"]
    sha = fen.ecrire_rapport_scelle(payload, args.out)
    h4, h4m = hyp["H4_alpha_omega_vs_montee_courte_grand_lr"], hyp["H4m_g_baisse_quand_lr_monte"]
    print(f"Parité T64 : PASS ; mode : {payload['mode_donnees']}")
    print(f"Rejeu (INFO) : {payload['porte_rejeu_INFO']}")
    print(f"H4 : médiane D={h4['mediane_D']:+.4f} p={h4['sign_test_exact']['p']:.3g} "
          f"({h4['sign_test_exact']['pos']}+/{h4['sign_test_exact']['neg']}−) "
          f"→ {'OUI' if h4['confirme_p001'] else 'NON'}")
    print(f"H4m : médiane log g(bas)−log g(haut)={h4m['mediane_log_g_bas_moins_haut']:+.3f} "
          f"p={h4m['sign_test_exact']['p']:.3g} → {'OUI' if h4m['confirme_p001'] else 'NON'}")
    print(f"NaN : {hyp['H_NaN_divergences_non_finies']}")
    print(f"Rapport scellé : {args.out} (sha256={sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
