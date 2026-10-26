"""benchmark_chrono_opt_tasks.py — ChronoSpiratonOpt vs Adam / AdamW hors protocole T64.

La grille T64 ne juge qu'UN modèle (``ChronoSpiraton(19)``). Pour « remplacer AdamW »,
l'optimiseur doit aussi tenir sur des surfaces de perte classiques. Trois tâches
synthétiques, déterministes (float64, CPU, graines fixées), chacune révélant une
propriété précise :

  * ``spirales`` : classification des deux spirales (MLP 2-64-64-2, tanh, minibatch 64)
    — non-convexe, le cas d'usage ordinaire d'Adam. Mesure : perte et exactitude TEST.
  * ``mal_conditionnee`` : régression linéaire, spectre de covariance sur 3 décades
    (κ = 1e3) — le terrain où la division coordonnée par coordonnée d'Adam aide.
    Mesure : perte TEST (excès sur le bruit).
  * ``pics`` : régression MLP 8-32-1 où 2 % des minibatchs portent des cibles ×1e4
    (pics de gradient), SANS clip_grad_norm_ — la stabilité intrinsèque revendiquée.
    Mesure : perte TEST sur données propres, finitude.

Équité : pour chaque (tâche, optimiseur), le lr est choisi sur une grille commune par
la meilleure perte de VALIDATION moyenne sur les graines ; la perte TEST n'est lue
qu'au lr retenu. Les étiquettes « meilleur » ne sont jamais des verdicts : 3 graines,
tâches jouets — c'est un repérage, pas un test d'hypothèse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from spiraton.experimental.chrono_optimizer import ChronoSpiratonOpt
from spiraton.experimental.spiral_optimizer import SpiralClosureOpt

Tenseurs = Tuple[torch.Tensor, torch.Tensor]

LRS: Tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2)
GRAINES: Tuple[int, ...] = (0, 1, 2)
BATCH = 64


@dataclass(frozen=True)
class Optimiseur:
    nom: str
    make: Callable[[Sequence[nn.Parameter], float], Optimizer]


def optimiseurs() -> Tuple[Optimiseur, ...]:
    """Adam, AdamW (wd=1e-2), SpiralClosureOpt (+ 2 ablations), ChronoSpiratonOpt v1
    (± régulateurs, ± wd) et la spec v0.

    Ablations de ``SpiralClosureOpt`` (wd=1e-2 partout, comme AdamW) :
      * ``spiral_sans_retour`` : α=1, k=1 — seule l'action bornée tanh diffère d'AdamW ;
      * ``spiral_sans_borne``  : borne=False — seul le cycle A→B→A′ diffère d'AdamW.
    """
    return (
        Optimiseur("adam", lambda p, lr: torch.optim.Adam(p, lr=lr)),
        Optimiseur("adamw", lambda p, lr: torch.optim.AdamW(p, lr=lr, weight_decay=1e-2)),
        Optimiseur("spiral", lambda p, lr: SpiralClosureOpt(p, lr=lr)),
        Optimiseur("spiral_sans_retour", lambda p, lr: SpiralClosureOpt(p, lr=lr, alpha=1.0, k=1)),
        Optimiseur("spiral_sans_borne", lambda p, lr: SpiralClosureOpt(p, lr=lr, borne=False)),
        Optimiseur("chrono_v1", lambda p, lr: ChronoSpiratonOpt(p, lr=lr)),
        Optimiseur("chrono_v1_wd", lambda p, lr: ChronoSpiratonOpt(p, lr=lr, weight_decay=1e-2)),
        Optimiseur("chrono_v1_sans_regulation",
                   lambda p, lr: ChronoSpiratonOpt(p, lr=lr, eta_rho=0.0, eta_omega=0.0, omega0=0.0)),
        Optimiseur("chrono_v0_spec", lambda p, lr: ChronoSpiratonOpt(p, lr=lr, legacy_v0=True)),
    )


# --- Tâches -----------------------------------------------------------------------


@dataclass(frozen=True)
class Tache:
    nom: str
    train: Tenseurs
    val: Tenseurs
    test: Tenseurs
    make_model: Callable[[], nn.Module]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    steps: int
    #: perturbation des cibles d'un minibatch d'entraînement (pics), ou None.
    perturber: Optional[Callable[[torch.Tensor, torch.Generator], torch.Tensor]] = None
    #: métrique additionnelle (exactitude) ou None.
    metrique: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None


def _spirales(n: int, gen: torch.Generator) -> Tenseurs:
    k = n // 2
    t = torch.rand(k, generator=gen, dtype=torch.float64) * 3.0 * math.pi + 0.5
    pts = []
    for sgn in (1.0, -1.0):
        x = sgn * torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1) / (3.0 * math.pi)
        pts.append(x + 0.02 * torch.randn(k, 2, generator=gen, dtype=torch.float64))
    x = torch.cat(pts)
    y = torch.cat([torch.zeros(k, dtype=torch.long), torch.ones(k, dtype=torch.long)])
    return x, y


def tache_spirales(steps: int = 3000) -> Tache:
    gen = torch.Generator().manual_seed(2026)
    tr, va, te = _spirales(1024, gen), _spirales(512, gen), _spirales(1024, gen)

    def make() -> nn.Module:
        return nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                             nn.Linear(64, 2)).double()

    def acc(logits: torch.Tensor, y: torch.Tensor) -> float:
        return float((logits.argmax(dim=1) == y).double().mean())

    return Tache("spirales", tr, va, te, make, nn.functional.cross_entropy, steps, metrique=acc)


def tache_mal_conditionnee(steps: int = 2000, d: int = 50) -> Tache:
    gen = torch.Generator().manual_seed(3)
    echelles = torch.logspace(0.0, -1.5, d, dtype=torch.float64)  # variances sur 3 décades
    w_star = torch.randn(d, 1, generator=gen, dtype=torch.float64)

    def donnees(n: int) -> Tenseurs:
        x = torch.randn(n, d, generator=gen, dtype=torch.float64) * echelles
        return x, x @ w_star + 0.01 * torch.randn(n, 1, generator=gen, dtype=torch.float64)

    tr, va, te = donnees(2048), donnees(512), donnees(2048)

    def make() -> nn.Module:
        return nn.Linear(d, 1).double()

    return Tache("mal_conditionnee", tr, va, te, make, nn.functional.mse_loss, steps)


def tache_pics(steps: int = 2000, d: int = 8) -> Tache:
    gen = torch.Generator().manual_seed(4)
    w1 = torch.randn(d, 16, generator=gen, dtype=torch.float64)

    def donnees(n: int) -> Tenseurs:
        x = torch.randn(n, d, generator=gen, dtype=torch.float64)
        y = torch.sin(x @ w1).sum(dim=1, keepdim=True) / 4.0
        return x, y + 0.01 * torch.randn(n, 1, generator=gen, dtype=torch.float64)

    tr, va, te = donnees(2048), donnees(512), donnees(2048)

    def make() -> nn.Module:
        return nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 1)).double()

    def perturber(y: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        if float(torch.rand(1, generator=g, dtype=torch.float64)) < 0.02:
            return y * 1e4
        return y

    return Tache("pics", tr, va, te, make, nn.functional.mse_loss, steps, perturber=perturber)


def taches(echelle: float = 1.0) -> Tuple[Tache, ...]:
    """Les trois tâches ; ``echelle`` réduit le nombre de pas (tests rapides)."""
    s = lambda n: max(1, int(n * echelle))  # noqa: E731
    return (tache_spirales(s(3000)), tache_mal_conditionnee(s(2000)), tache_pics(s(2000)))


# --- Entraînement et sélection --------------------------------------------------


def _evaluer(model: nn.Module, tache: Tache, split: Tenseurs) -> Dict[str, float]:
    with torch.no_grad():
        out = model(split[0])
        res = {"perte": float(tache.loss_fn(out, split[1]))}
        if tache.metrique is not None:
            res["exactitude"] = tache.metrique(out, split[1])
    return res


def entrainer(tache: Tache, opt_spec: Optimiseur, lr: float, graine: int) -> Dict[str, object]:
    """Un run : init sous ``manual_seed(graine)``, minibatchs sous ``Generator(100+graine)``."""
    torch.manual_seed(graine)
    model = tache.make_model()
    opt = opt_spec.make(list(model.parameters()), lr)
    gen = torch.Generator().manual_seed(100 + graine)
    x, y = tache.train
    n = x.shape[0]
    fini = True
    for _ in range(tache.steps):
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        yb = y[idx] if tache.perturber is None else tache.perturber(y[idx], gen)
        opt.zero_grad()
        loss = tache.loss_fn(model(x[idx]), yb)
        if not torch.isfinite(loss):
            fini = False
            break
        loss.backward()
        opt.step()
    if fini:
        fini = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    val = _evaluer(model, tache, tache.val) if fini else {"perte": float("inf")}
    test = _evaluer(model, tache, tache.test) if fini else {"perte": float("inf")}
    if not math.isfinite(val["perte"]):
        fini = False
    return {"fini": fini, "val": val, "test": test}


def executer(echelle: float = 1.0, graines: Sequence[int] = GRAINES,
             lrs: Sequence[float] = LRS) -> Dict[str, object]:
    resultats: Dict[str, object] = {}
    for tache in taches(echelle):
        par_opt: Dict[str, object] = {}
        for spec in optimiseurs():
            grille: Dict[str, object] = {}
            meilleur: Optional[Tuple[float, float]] = None
            for lr in lrs:
                runs = [entrainer(tache, spec, lr, g) for g in graines]
                vals = [r["val"]["perte"] for r in runs]
                moy_val = float(sum(vals) / len(vals)) if all(map(math.isfinite, vals)) else float("inf")
                grille[repr(lr)] = {"perte_val_moyenne": moy_val,
                                    "n_non_finis": sum(1 for r in runs if not r["fini"]),
                                    "runs": runs}
                if math.isfinite(moy_val) and (meilleur is None or moy_val < meilleur[0]):
                    meilleur = (moy_val, lr)
            resume: Dict[str, object] = {"grille": grille}
            if meilleur is not None:
                runs = grille[repr(meilleur[1])]["runs"]
                resume["lr_retenu"] = meilleur[1]
                resume["perte_test_moyenne"] = float(
                    sum(r["test"]["perte"] for r in runs) / len(runs))
                if tache.metrique is not None:
                    resume["exactitude_test_moyenne"] = float(
                        sum(r["test"]["exactitude"] for r in runs) / len(runs))
            resume["n_non_finis_total"] = sum(v["n_non_finis"] for v in grille.values())
            par_opt[spec.nom] = resume
        resultats[tache.nom] = par_opt
    return {"chantier": "OPTIMISEUR_ABA_TACHES", "echelle": echelle,
            "graines": list(graines), "lrs": list(lrs), "resultats": resultats}


def tableau(payload: Dict[str, object]) -> List[str]:
    """Lignes lisibles : tâche | optimiseur | lr retenu | perte test | exactitude | non finis."""
    lignes = []
    for tache, par_opt in payload["resultats"].items():
        for nom, r in par_opt.items():
            acc = r.get("exactitude_test_moyenne")
            lignes.append(
                f"{tache:17s} | {nom:26s} | lr={r.get('lr_retenu', float('nan')):<7g} "
                f"| test={r.get('perte_test_moyenne', float('nan')):.4e} "
                f"| acc={'' if acc is None else f'{acc:.3f}':5s} | non_finis={r['n_non_finis_total']}"
            )
    return lignes


def ecrire_rapport_scelle(payload: Dict[str, object], out_path: Path) -> str:
    texte = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out_path.write_text(texte, encoding="utf-8")
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    out_path.with_suffix(out_path.suffix + ".sha256").write_text(sha + "\n", encoding="utf-8")
    return sha


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("BENCHMARK_CHRONO_OPT_TACHES.json"))
    parser.add_argument("--quick", action="store_true",
                        help="5 %% des pas, graine 0, lr ∈ {1e-2} (test de fumée)")
    args = parser.parse_args(argv)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)  # petits modèles : le parallélisme intra-op coûte ~100× ici
    if args.quick:
        payload = executer(echelle=0.05, graines=(0,), lrs=(1e-2,))
    else:
        payload = executer()
    sha = ecrire_rapport_scelle(payload, args.out)
    for ligne in tableau(payload):
        print(ligne)
    print(f"Rapport scellé : {args.out} (sha256={sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
