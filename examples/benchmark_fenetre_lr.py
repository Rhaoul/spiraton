"""benchmark_fenetre_lr.py — Fenêtre de lr : sur quelle plage chaque optimiseur apprend-il ?

Protocole figé AVANT toute mesure sur ``dataset_aba.txt`` : ``docs/PROTOCOLE_FENETRE_LR.md``
(son sha256 est gravé dans le rapport). Le banc T64 (``benchmark_chrono_opt_vs_adamw.py``)
est réutilisé tel quel : même modèle ``ChronoSpiraton(19, bounded=False)``, même perte
``perte_alpha_omega``, mêmes splits (TRAIN 101-1000, VALIDATION 3001-3500, TEST 3501-5001
JAMAIS ouvert), même boucle ``train_run_candidate`` et même porte G0 de parité au bit.

Ce qui change est la grille, élargie là où AdamW échoue :
``LRS_FENETRE`` (9 valeurs, 1e-4 → 1) × ``INITS_FENETRE`` (0,1 ; 0,6 ; 1,5) × graines
(0, 1, 2) × cartouches (S, P) = 162 runs par optimiseur.

Définitions opérationnelles :
  * un run **apprend** ⇔ il ne diverge pas (perte finie) ET ``best_delta_val > 0``
    (sur VALIDATION, Φ_θ fait mieux que la copie, l'identité valant 0 exact) ;
  * une **unité** = (cartouche, init, graine), 18 unités ;
  * ``N_apprend(unité)`` = nombre de lr, sur 9, où le run apprend : la largeur de la
    fenêtre de lr de l'optimiseur pour cette unité.

Les hypothèses H0-H3 et leurs seuils sont ceux du protocole ; ce script les tranche.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import locale
import math
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from spiraton.experimental.alpha_omega_optimizer import AlphaOmegaOpt


def _charger_banc_t64() -> ModuleType:
    """Le banc T64 voisin (``examples/`` n'est pas un paquet) : chargé par son chemin."""
    nom = "benchmark_chrono_opt_vs_adamw"
    if nom in sys.modules:
        return sys.modules[nom]
    chemin = Path(__file__).resolve().parent / f"{nom}.py"
    spec = importlib.util.spec_from_file_location(nom, chemin)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[nom] = module
    spec.loader.exec_module(module)
    return module


banc = _charger_banc_t64()
rt = banc.rt

PROTOCOLE = Path(__file__).resolve().parents[1] / "docs" / "PROTOCOLE_FENETRE_LR.md"

LRS_FENETRE: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
INITS_FENETRE: Tuple[float, ...] = (0.1, 0.6, 1.5)
SEUIL_P = rt.SEUIL_P  # 0,01, bilatéral, comme T64

#: Témoin du premier pas court : même départ que AlphaOmegaOpt (gain 0,01), montée
#: géométrique jusqu'à 1 en ``MONTEE_PAS`` pas (10 époques de 15 lots), SANS retour.
MONTEE_G0 = 1e-2
MONTEE_PAS = 150

CANDIDATS_T64 = ("adam_clipped_t64", "adamw_clipped", "adamw_unclipped",
                 "spiral_closure", "chrono_opt_c_outside")


class AdamWMonteeCourte(torch.optim.AdamW):
    """AdamW dont le lr monte de ``g0·lr`` à ``lr`` en ``pas_montee`` pas (géométrique).

    Témoin de H2 : il partage avec ``AlphaOmegaOpt`` le premier pas court, mais sa
    montée est un calendrier fixe ; aucun retour ne la règle. Au pas t (compté depuis
    0), ``lr_t = lr · g0^(1 − t/pas_montee)`` tant que ``t < pas_montee``, puis ``lr``.
    """

    def __init__(self, params, lr: float, *, g0: float = MONTEE_G0,
                 pas_montee: int = MONTEE_PAS, **kwargs) -> None:
        super().__init__(params, lr=lr, **kwargs)
        self._lr_cible = [grp["lr"] for grp in self.param_groups]
        self._g0 = g0
        self._pas_montee = pas_montee
        self._t = 0

    def facteur(self, t: int) -> float:
        return self._g0 ** (1.0 - t / self._pas_montee) if t < self._pas_montee else 1.0

    @torch.no_grad()
    def step(self, closure=None):
        f = self.facteur(self._t)
        for grp, lr in zip(self.param_groups, self._lr_cible):
            grp["lr"] = lr * f
        self._t += 1
        return super().step(closure)


def candidats() -> Tuple["banc.CandidateSpec", ...]:
    """Les 7 optimiseurs confrontés (ordre gelé par le protocole)."""
    t64 = {s.nom: s for s in banc.candidats_optimiseurs()}
    return tuple(t64[n] for n in CANDIDATS_T64) + (
        banc.CandidateSpec(
            nom="adamw_montee_courte_clipped",
            description=(f"AdamW (wd=1e-2) + clip(1.0), lr monté de {MONTEE_G0}·lr à lr en "
                         f"{MONTEE_PAS} pas, sans retour — témoin H2"),
            clip_norm=rt.CLIP_NORM,
            make_opt=lambda params, lr: AdamWMonteeCourte(
                params, lr, betas=rt.ADAM_BETAS, eps=rt.ADAM_EPS, weight_decay=1e-2),
        ),
        banc.CandidateSpec(
            nom="alpha_omega",
            description="AlphaOmegaOpt (wd=1e-2, réglages par défaut gelés) SANS clip",
            clip_norm=None,
            make_opt=lambda params, lr: AlphaOmegaOpt(
                params, lr=lr, betas=rt.ADAM_BETAS, eps=rt.ADAM_EPS, weight_decay=1e-2),
        ),
    )


# --- Données et runs (un processus de calcul charge ses données une fois) -------------

_DONNEES: Dict[str, object] = {}


def _preparer(root: Path, synthetic: bool) -> Dict[str, object]:
    if synthetic:
        m1, a_tr, ap_tr, a_val, ap_val = banc._donnees_synthetiques()
        mode = "SYNTHETIC_19D_FLOAT64"
    else:
        tok, m1 = rt._instrument()
        tr, va = rt._splits_train_val(tok, m1, root)
        a_tr, ap_tr, a_val, ap_val = tr.a, tr.ap, va.a, va.ap
        mode = f"REAL_DATASET_ABA (md5={rt.DATASET_MD5}, TEST_SCELLE_NON_OUVERT)"
    m = {c: rt.mesurer_m(a_tr[c], ap_tr[c]) for c in rt.CARTOUCHES}
    s_raw = {c: rt.scores_m1(m1, a_val[c], ap_val[c]) for c in rt.CARTOUCHES}
    return dict(m1=m1, a_tr=a_tr, ap_tr=ap_tr, a_val=a_val, ap_val=ap_val, m=m,
                s_raw=s_raw, mode=mode)


def _init_processus(root: str, synthetic: bool) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    _DONNEES.clear()
    _DONNEES.update(_preparer(Path(root), synthetic))


def _un_run(tache: Tuple[str, str, float, float, int, int]) -> Dict[str, object]:
    """Un run (optimiseur, cartouche, init, lr, graine) ; renvoie mesures et Δ par cycle."""
    nom, cart, init, lr, seed, epochs_max = tache
    spec = {s.nom: s for s in candidats()}[nom]
    d = _DONNEES
    rr = banc.train_run_candidate(
        d["m1"], spec, cart, init, lr, seed, d["a_tr"][cart], d["ap_tr"][cart],
        d["a_val"][cart], d["ap_val"][cart], d["s_raw"][cart], d["m"][cart],
        epochs_max=epochs_max,
    )
    model = rt.modele_depuis_poids(init, rr.state)
    x_val = rt.appliquer_chrono(model, d["a_val"][cart])
    dv = (rt.deltas(d["m1"], x_val, d["ap_val"][cart], d["s_raw"][cart])
          if np.isfinite(x_val).all() else np.full(x_val.shape[0], -np.inf))
    apprend = bool((not rr.diverged) and math.isfinite(rr.best_delta_val) and rr.best_delta_val > 0.0)
    return {
        "optimiseur": nom, "cartouche": cart, "init_scale": init, "lr": lr, "seed": seed,
        "diverged": rr.diverged, "apprend": apprend, "best_delta_val": rr.best_delta_val,
        "epochs_run": rr.epochs_run, "best_epoch": rr.best_epoch,
        "early_stopped": rr.early_stopped, "loss_init": rr.loss_init, "loss_min": rr.loss_min,
        "state_sha256": rr.state_sha256, "_dv": dv,
    }


def taches(noms: Sequence[str], cartouches: Sequence[str], inits: Sequence[float],
           lrs: Sequence[float], graines: Sequence[int], epochs_max: int) -> List[tuple]:
    return [(n, c, i, lr, s, epochs_max) for n in noms for c in cartouches for i in inits
            for lr in lrs for s in graines]


# --- Lecture des hypothèses ----------------------------------------------------------

def n_apprend(runs: List[Dict[str, object]], nom: str) -> Dict[Tuple[str, float, int], int]:
    """``N_apprend`` par unité (cartouche, init, graine) pour un optimiseur."""
    out: Dict[Tuple[str, float, int], int] = {}
    for r in runs:
        if r["optimiseur"] != nom:
            continue
        cle = (r["cartouche"], r["init_scale"], r["seed"])
        out[cle] = out.get(cle, 0) + int(r["apprend"])
    return out


def comparer_fenetres(runs: List[Dict[str, object]], a: str, b: str) -> Dict[str, object]:
    """Sign test exact apparié sur les unités : ``N_apprend(a) − N_apprend(b)``."""
    na, nb = n_apprend(runs, a), n_apprend(runs, b)
    unites = sorted(na)
    diff = np.asarray([na[u] - nb[u] for u in unites], dtype=np.float64)
    st = rt.sign_test_exact(diff)
    med = float(np.median(diff))
    return {
        "unites": len(unites), "mediane_diff": med, "moyenne_diff_INFO": float(diff.mean()),
        "somme_a": int(sum(na.values())), "somme_b": int(sum(nb.values())),
        "sign_test_exact": st, "a_plus_large_p001": bool(med > 0.0 and st["p"] < SEUIL_P),
    }


def selection_validation(runs: List[Dict[str, object]], nom: str, cart: str,
                         inits: Sequence[float], lrs: Sequence[float]) -> Dict[str, object]:
    """Règle T64 sur la grille élargie : config maximisant la moyenne (graines vivantes)
    de ``best_delta_val`` ; Δ par cycle de la graine médiane pour l'appariement."""
    meilleurs = []
    for init in inits:
        for lr in lrs:
            rs = [r for r in runs if r["optimiseur"] == nom and r["cartouche"] == cart
                  and r["init_scale"] == init and r["lr"] == lr and not r["diverged"]]
            if rs:
                meilleurs.append((float(np.mean([r["best_delta_val"] for r in rs])), init, lr, rs))
    if not meilleurs:
        return {"erreur": "toutes les configs ont divergé"}
    meilleurs.sort(key=lambda t: t[0], reverse=True)
    moy, init, lr, rs = meilleurs[0]
    meds = sorted((float(np.median(r["_dv"])), r["seed"], r) for r in rs)
    _, seed_med, r_med = meds[(len(meds) - 1) // 2]
    return {"init_scale": init, "lr": lr, "moyenne_best_delta_val": moy,
            "graine_mediane": seed_med, "Delta_val_mediane": float(np.median(r_med["_dv"])),
            "_dv": r_med["_dv"]}


def lire_hypotheses(runs: List[Dict[str, object]], cartouches: Sequence[str],
                    inits: Sequence[float], lrs: Sequence[float]) -> Dict[str, object]:
    noms = [s.nom for s in candidats()]
    profil = {n: {repr(lr): int(sum(r["apprend"] for r in runs
                                    if r["optimiseur"] == n and r["lr"] == lr))
                  for lr in lrs} for n in noms}
    nan = {n: int(sum(r["diverged"] for r in runs if r["optimiseur"] == n)) for n in noms}
    total = {n: int(sum(r["apprend"] for r in runs if r["optimiseur"] == n)) for n in noms}
    echecs_adamw = sum(1 for r in runs if r["optimiseur"] == "adamw_unclipped" and not r["apprend"])
    h0 = echecs_adamw >= 1

    selections = {n: {c: selection_validation(runs, n, c, inits, lrs) for c in cartouches}
                  for n in noms}
    h3: Dict[str, Dict[str, object]] = {}
    for ref in ("adam_clipped_t64", "adamw_clipped"):
        h3[ref] = {}
        for c in cartouches:
            sa, sb = selections["alpha_omega"][c], selections[ref][c]
            if "_dv" not in sa or "_dv" not in sb:
                h3[ref][c] = {"erreur": "sélection impossible"}
                continue
            diff = sa["_dv"] - sb["_dv"]
            st = rt.sign_test_exact(diff)
            med = float(np.median(diff))
            h3[ref][c] = {"mediane_diff_appariee": med, "sign_test_exact": st,
                          "degradation_p001": bool(med < 0.0 and st["p"] < SEUIL_P),
                          "avantage_p001": bool(med > 0.0 and st["p"] < SEUIL_P)}
    sel_pub = {n: {c: {k: v for k, v in s.items() if k != "_dv"} for c, s in par.items()}
               for n, par in selections.items()}
    return {
        "H0_adamw_sans_clip_echoue_quelque_part": {"echecs": echecs_adamw, "PASS": h0},
        "H_NaN_divergences_non_finies": nan,
        "H1_alpha_omega_vs_adamw_clipped": comparer_fenetres(runs, "alpha_omega", "adamw_clipped"),
        "H2_alpha_omega_vs_montee_courte": comparer_fenetres(
            runs, "alpha_omega", "adamw_montee_courte_clipped"),
        "H3_meilleur_reglage_validation": h3,
        "INFO_fenetres_toutes_paires_vs_adamw_clipped": {
            n: comparer_fenetres(runs, n, "adamw_clipped") for n in noms if n != "adamw_clipped"},
        "INFO_runs_qui_apprennent_par_lr": profil,
        "INFO_runs_qui_apprennent_total": total,
        "INFO_selection_validation": sel_pub,
    }


# --- Contexte d'exécution (doctrine T74) ---------------------------------------------

def _sha_fichier(p: Path) -> Optional[str]:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def contexte(root: Path) -> Dict[str, object]:
    return {
        "plateforme": platform.platform(), "python": sys.version.split()[0],
        "torch": torch.__version__, "numpy": np.__version__,
        "encodage_prefere": locale.getpreferredencoding(False),
        "protocole_sha256": _sha_fichier(PROTOCOLE),
        "dataset_aba_sha256": _sha_fichier(root / rt.DATASET),
        "tokenizer_so_sha256": _sha_fichier(root / "Tokenizer" / "bin" / "libspiratontokenizer.so"),
    }


def executer(root: Path, *, synthetic: bool, epochs_max: int, cartouches: Sequence[str],
             inits: Sequence[float], lrs: Sequence[float], graines: Sequence[int],
             jobs: int = 1) -> Dict[str, object]:
    t0 = time.perf_counter()
    _init_processus(str(root), synthetic)
    d = _DONNEES
    parite = banc.verifier_parite_t64(
        d["m1"], banc.candidats_optimiseurs()[0], d["a_tr"]["S"], d["ap_tr"]["S"],
        d["a_val"]["S"], d["ap_val"]["S"], d["s_raw"]["S"], d["m"]["S"],
        epochs_check=min(15, epochs_max),
    )
    liste = taches([s.nom for s in candidats()], cartouches, inits, lrs, graines, epochs_max)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_processus,
                                 initargs=(str(root), synthetic)) as ex:
            runs = list(ex.map(_un_run, liste, chunksize=4))
    else:
        runs = [_un_run(t) for t in liste]
    hyp = lire_hypotheses(runs, cartouches, inits, lrs)
    return {
        "chantier": "OPTIMISEUR_ABA_FENETRE_LR",
        "mode_donnees": d["mode"],
        "contexte": contexte(root),
        "grille": {"lrs": list(lrs), "inits": list(inits), "graines": list(graines),
                   "cartouches": list(cartouches), "epochs_max": epochs_max},
        "candidats": {s.nom: s.description for s in candidats()},
        "parite_bit_a_bit_T64": parite,
        "hypotheses": hyp,
        "runs": [{k: v for k, v in r.items() if k != "_dv"} for r in runs],
        "duree_totale_s_INFO": time.perf_counter() - t0,
    }


def ecrire_rapport_scelle(payload: Dict[str, object], out: Path) -> str:
    """JSON trié + sceau ``.sha256`` ; la durée (seule valeur non reproductible) est exclue
    du sceau en étant écrite dans un ``.log`` séparé."""
    duree = payload.pop("duree_totale_s_INFO", None)
    texte = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out.write_text(texte, encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(sha + "\n", encoding="utf-8")
    out.with_suffix(out.suffix + ".log").write_text(
        f"duree_totale_s={duree}\nsha256={sha}\n", encoding="utf-8")
    return sha


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="racine contenant dataset_aba.txt et Tokenizer/ (défaut : .)")
    parser.add_argument("--out", type=Path, default=Path("BENCH_FENETRE_LR.json"))
    parser.add_argument("--synthetic", action="store_true",
                        help="données 19D synthétiques (calibration, jamais une mesure)")
    parser.add_argument("--quick", action="store_true",
                        help="30 époques, cartouche S, graine 0, 3 lr, 1 init (test du script)")
    parser.add_argument("--jobs", type=int, default=1, help="processus de calcul parallèles")
    args = parser.parse_args(argv)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if args.quick:
        grille = dict(epochs_max=30, cartouches=("S",), inits=(0.6,), lrs=(1e-3, 3e-2, 1.0),
                      graines=(0,))
    else:
        grille = dict(epochs_max=rt.EPOCHS_MAX, cartouches=rt.CARTOUCHES, inits=INITS_FENETRE,
                      lrs=LRS_FENETRE, graines=rt.GRAINES)
    payload = executer(args.root, synthetic=args.synthetic, jobs=args.jobs, **grille)
    hyp = payload["hypotheses"]
    sha = ecrire_rapport_scelle(payload, args.out)
    print(f"Parité T64 : PASS ; mode : {payload['mode_donnees']}")
    print(f"H0 (AdamW sans clip échoue quelque part) : {hyp['H0_adamw_sans_clip_echoue_quelque_part']}")
    for nom, tot in hyp["INFO_runs_qui_apprennent_total"].items():
        print(f"  - {nom:28s} apprend dans {tot:3d} runs | NaN={hyp['H_NaN_divergences_non_finies'][nom]}")
    for cle in ("H1_alpha_omega_vs_adamw_clipped", "H2_alpha_omega_vs_montee_courte"):
        h = hyp[cle]
        print(f"{cle} : médiane={h['mediane_diff']:+.1f} p={h['sign_test_exact']['p']:.3g} "
              f"→ {'OUI' if h['a_plus_large_p001'] else 'NON'}")
    print(f"Rapport scellé : {args.out} (sha256={sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
