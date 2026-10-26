"""benchmark_chrono_opt_vs_adamw.py — Confrontation ChronoSpiratonOpt vs Adam / AdamW.

Évalue sur la grille close de ``return_training.py`` (TRAIN 101-1000, VALIDATION
3001-3500, sans JAMAIS ouvrir le split TEST scellé 3501-5001) cinq régimes
d'optimisation entraînant ``ChronoSpiraton(state_size=19, bounded=False)`` sous
``perte_alpha_omega`` :

  1. ``adam_clipped_t64``    : Adam (wd=0.0) + clip_grad_norm_(1.0) — référence T64.
  2. ``adamw_clipped``       : AdamW (wd=1e-2) + clip_grad_norm_(1.0).
  3. ``adamw_unclipped``     : AdamW (wd=1e-2) SANS clip_grad_norm_ (test de stabilité brute).
  4. ``chrono_opt_c_outside``: ChronoSpiratonOpt (c_outside=True, T3) SANS clip_grad_norm_.
  5. ``chrono_opt_c_inside`` : ChronoSpiratonOpt (c_outside=False, T2) SANS clip_grad_norm_.
  6. ``chrono_opt_v0_spec``  : ChronoSpiratonOpt(legacy_v0=True) — la spec EXACTE avant
     corrections, SANS clip (témoin : ce que mesurent les défauts documentés).
  7. ``spiral_closure``      : SpiralClosureOpt (wd=1e-2, k=5, α=0.5, tanh) SANS clip —
     la conception libre (cycle A→B→A′ au niveau des cycles, ``spiral_optimizer.py``).

GARDE DE PARITÉ BIT-À-BIT (exécutée en tête) :
  Sur (S, init_scale=0.1, lr=1e-3, seed=0), ``adam_clipped_t64`` DOIT produire
  exactement le même ``state_sha256`` et le même ``best_delta_val`` que
  ``return_training.train_run``. Tout écart => RuntimeError immédiate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim import Optimizer

from spiraton.experimental import return_training as rt
from spiraton.experimental.chrono import ChronoSpiraton
from spiraton.experimental.chrono_optimizer import ChronoSpiratonOpt
from spiraton.experimental.spiral_optimizer import SpiralClosureOpt


@dataclass(frozen=True)
class CandidateSpec:
    """Spécification d'un optimiseur candidat."""

    nom: str
    description: str
    clip_norm: Optional[float]
    make_opt: Callable[[Sequence[torch.nn.Parameter], float], Optimizer]


def candidats_optimiseurs(weight_decay_adamw: float = 1e-2) -> Tuple[CandidateSpec, ...]:
    """Les 7 régimes confrontés à graine, batchs et initialisation identiques."""
    return (
        CandidateSpec(
            nom="adam_clipped_t64",
            description="Adam (wd=0.0) + clip_grad_norm_(1.0) — référence exacte T64",
            clip_norm=rt.CLIP_NORM,
            make_opt=lambda params, lr: torch.optim.Adam(
                params, lr=lr, betas=rt.ADAM_BETAS, eps=rt.ADAM_EPS, weight_decay=0.0
            ),
        ),
        CandidateSpec(
            nom="adamw_clipped",
            description=f"AdamW (wd={weight_decay_adamw}) + clip_grad_norm_(1.0)",
            clip_norm=rt.CLIP_NORM,
            make_opt=lambda params, lr: torch.optim.AdamW(
                params,
                lr=lr,
                betas=rt.ADAM_BETAS,
                eps=rt.ADAM_EPS,
                weight_decay=weight_decay_adamw,
            ),
        ),
        CandidateSpec(
            nom="adamw_unclipped",
            description=f"AdamW (wd={weight_decay_adamw}) SANS clip_grad_norm_ (stabilité brute)",
            clip_norm=None,
            make_opt=lambda params, lr: torch.optim.AdamW(
                params,
                lr=lr,
                betas=rt.ADAM_BETAS,
                eps=rt.ADAM_EPS,
                weight_decay=weight_decay_adamw,
            ),
        ),
        CandidateSpec(
            nom="chrono_opt_c_outside",
            description="ChronoSpiratonOpt (c_outside=True, inhibition découplée T3) SANS clip",
            clip_norm=None,
            make_opt=lambda params, lr: ChronoSpiratonOpt(
                params, lr=lr, c_outside=True
            ),
        ),
        CandidateSpec(
            nom="chrono_opt_c_inside",
            description="ChronoSpiratonOpt (c_outside=False, inhibition interne T2) SANS clip",
            clip_norm=None,
            make_opt=lambda params, lr: ChronoSpiratonOpt(
                params, lr=lr, c_outside=False
            ),
        ),
        CandidateSpec(
            nom="chrono_opt_v0_spec",
            description="ChronoSpiratonOpt(legacy_v0=True) — spec exacte avant corrections, SANS clip",
            clip_norm=None,
            make_opt=lambda params, lr: ChronoSpiratonOpt(
                params, lr=lr, c_outside=True, legacy_v0=True
            ),
        ),
        CandidateSpec(
            nom="spiral_closure",
            description="SpiralClosureOpt (wd=1e-2, k=5, alpha=0.5, tanh) SANS clip",
            clip_norm=None,
            make_opt=lambda params, lr: SpiralClosureOpt(
                params, lr=lr, betas=rt.ADAM_BETAS, eps=rt.ADAM_EPS,
                weight_decay=weight_decay_adamw,
            ),
        ),
    )


def train_run_candidate(
    m1,
    spec: CandidateSpec,
    cartouche: str,
    init_scale: float,
    lr: float,
    seed: int,
    a_tr: np.ndarray,
    ap_tr: np.ndarray,
    a_val: np.ndarray,
    ap_val: np.ndarray,
    s_raw_val: np.ndarray,
    m: float,
    *,
    epochs_max: int = rt.EPOCHS_MAX,
    bounded: bool = False,
) -> rt.RunResult:
    """Miroir exact de ``return_training.train_run`` paramétré par ``CandidateSpec``.

    À ``spec == adam_clipped_t64``, la séquence d'opérations flottantes et d'appels
    au RNG est 100 % identique à ``rt.train_run`` (vérifié par ``verifier_parite_t64``).
    """
    torch.manual_seed(seed)
    model = ChronoSpiraton(
        state_size=rt.STATE_SIZE,
        init_scale=init_scale,
        bounded=bounded,
        c_outside=False,
    ).double()
    gen = torch.Generator()
    gen.manual_seed(1000 + seed)
    params = list(model.parameters())
    opt = spec.make_opt(params, lr)

    ta = torch.from_numpy(np.ascontiguousarray(a_tr))
    tap = torch.from_numpy(np.ascontiguousarray(ap_tr))
    n = ta.shape[0]
    pairs = rt.paires_dispersion(a_val.shape[0])

    with torch.no_grad():
        loss_init = float(rt.perte_alpha_omega(model, ta, tap, m).item())

    courbes: Dict[str, List[float]] = {
        "perte_train": [],
        "delta_med_val": [],
        "r_disp": [],
        "dbar": [],
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
            idx = perm[k0 : k0 + rt.BATCH]
            opt.zero_grad()
            loss = rt.perte_alpha_omega(model, ta[idx], tap[idx], m)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            if spec.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), spec.clip_norm)
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
    descente = (
        (loss_init - loss_min) / abs(loss_init)
        if courbes["perte_train"] and abs(loss_init) > 0
        else float("nan")
    )
    return rt.RunResult(
        cartouche=cartouche,
        init_scale=init_scale,
        lr=lr,
        seed=seed,
        diverged=diverged,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        best_delta_val=best_val,
        loss_init=loss_init,
        loss_min=loss_min,
        descente_rel=descente,
        early_stopped=early,
        courbes=courbes,
        state=best_state,
        state_sha256=rt.hash_etat(best_state),
    )


def verifier_parite_t64(
    m1,
    spec_adam_t64: CandidateSpec,
    a_tr: np.ndarray,
    ap_tr: np.ndarray,
    a_val: np.ndarray,
    ap_val: np.ndarray,
    s_raw_val: np.ndarray,
    m: float,
    *,
    epochs_check: int = 15,
) -> Dict[str, object]:
    """Porte G0 : vérifie que train_run_candidate(adam_clipped_t64) == rt.train_run au bit."""
    ref = rt.train_run(
        m1, "S", 0.1, 1e-3, 0, a_tr, ap_tr, a_val, ap_val, s_raw_val, m, epochs_max=epochs_check
    )
    cand = train_run_candidate(
        m1,
        spec_adam_t64,
        "S",
        0.1,
        1e-3,
        0,
        a_tr,
        ap_tr,
        a_val,
        ap_val,
        s_raw_val,
        m,
        epochs_max=epochs_check,
    )
    ok = (
        ref.state_sha256 == cand.state_sha256
        and ref.best_delta_val == cand.best_delta_val
        and ref.courbes == cand.courbes
    )
    if not ok:
        raise RuntimeError(
            f"Parité T64 brisée : sha_ref={ref.state_sha256} vs sha_cand={cand.state_sha256}"
        )
    return {
        "PASS": True,
        "epochs_check": epochs_check,
        "state_sha256": ref.state_sha256,
        "best_delta_val": ref.best_delta_val,
    }


# --- Fallback synthétique (si exécuté hors environnement .so / dataset_aba) ---

class _SyntheticM1:
    """Émulateur exact de m1.score_retour (cos - l2, parité perte_alpha_omega)."""

    @staticmethod
    def score_retour(x: np.ndarray, ap: np.ndarray) -> float:
        nx = float(np.linalg.norm(x))
        nap = float(np.linalg.norm(ap))
        cos = float(np.dot(x, ap)) / (nx * nap + rt.EPS)
        l2 = float(np.linalg.norm(ap - x)) / (nx + rt.EPS)
        return cos - l2


def _donnees_synthetiques(n_tr: int = 256, n_val: int = 128) -> Tuple[object, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Génère des paires (a, a′) synthétiques déterministes en float64 (19D)."""
    rng = np.random.default_rng(20260720)
    a_tr, ap_tr, a_val, ap_val = {}, {}, {}, {}
    for cart in rt.CARTOUCHES:
        base_tr = rng.standard_normal((n_tr, rt.STATE_SIZE)).astype(np.float64)
        rot = np.eye(rt.STATE_SIZE, dtype=np.float64) + 0.15 * rng.standard_normal(
            (rt.STATE_SIZE, rt.STATE_SIZE)
        )
        a_tr[cart] = base_tr
        ap_tr[cart] = base_tr @ rot.T + 0.05 * rng.standard_normal((n_tr, rt.STATE_SIZE))

        base_val = rng.standard_normal((n_val, rt.STATE_SIZE)).astype(np.float64)
        a_val[cart] = base_val
        ap_val[cart] = base_val @ rot.T + 0.05 * rng.standard_normal((n_val, rt.STATE_SIZE))
    return _SyntheticM1(), a_tr, ap_tr, a_val, ap_val


# --- Exécution du benchmark et comparaison appariée ---------------------------

def executer_benchmark(
    root: Path,
    *,
    synthetic: bool = False,
    epochs_max: int = rt.EPOCHS_MAX,
    cartouches: Sequence[str] = rt.CARTOUCHES,
    graines: Sequence[int] = rt.GRAINES,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    if synthetic:
        m1, a_tr, ap_tr, a_val, ap_val = _donnees_synthetiques()
        source_mode = "SYNTHETIC_19D_FLOAT64"
    else:
        tok, m1 = rt._instrument()
        tr, va = rt._splits_train_val(tok, m1, root)
        a_tr, ap_tr = tr.a, tr.ap
        a_val, ap_val = va.a, va.ap
        source_mode = f"REAL_DATASET_ABA (md5={rt.DATASET_MD5}, TEST_SCELLE_NON_OUVERT)"

    specs = candidats_optimiseurs()
    m_s = rt.mesurer_m(a_tr["S"], ap_tr["S"])
    s_raw_val_s = rt.scores_m1(m1, a_val["S"], ap_val["S"])

    # 1. Porte G0 : Parité bit-à-bit avec return_training.train_run
    parite_t64 = verifier_parite_t64(
        m1,
        specs[0],
        a_tr["S"],
        ap_tr["S"],
        a_val["S"],
        ap_val["S"],
        s_raw_val_s,
        m_s,
        epochs_check=min(15, epochs_max),
    )

    resultats_par_opt: Dict[str, Dict[str, object]] = {}
    vecteurs_deltas_val: Dict[Tuple[str, str], np.ndarray] = {}

    for spec in specs:
        t_opt = time.perf_counter()
        runs_pub: List[Dict[str, object]] = []
        states_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        n_diverges = 0

        for cart in cartouches:
            m = rt.mesurer_m(a_tr[cart], ap_tr[cart])
            s_raw_val = rt.scores_m1(m1, a_val[cart], ap_val[cart])
            for cfg in rt.configs_grille():
                for seed in graines:
                    rr = train_run_candidate(
                        m1,
                        spec,
                        cart,
                        cfg["init_scale"],
                        cfg["lr"],
                        seed,
                        a_tr[cart],
                        ap_tr[cart],
                        a_val[cart],
                        ap_val[cart],
                        s_raw_val,
                        m,
                        epochs_max=epochs_max,
                    )
                    rid = f"{cart}_{rt.config_id(cfg['init_scale'], cfg['lr'])}_seed{seed}"
                    states_cache[rid] = rr.state
                    if rr.diverged:
                        n_diverges += 1
                    r_disp_fin = rr.courbes["r_disp"][-1] if rr.courbes["r_disp"] else float("nan")
                    dbar_fin = rr.courbes["dbar"][-1] if rr.courbes["dbar"] else float("nan")
                    runs_pub.append(
                        {
                            "run_id": rid,
                            "cartouche": cart,
                            "init_scale": cfg["init_scale"],
                            "lr": cfg["lr"],
                            "seed": seed,
                            "diverged": rr.diverged,
                            "epochs_run": rr.epochs_run,
                            "best_epoch": rr.best_epoch,
                            "early_stopped": rr.early_stopped,
                            "best_delta_val": rr.best_delta_val,
                            "loss_init": rr.loss_init,
                            "loss_min": rr.loss_min,
                            "descente_rel": rr.descente_rel,
                            "descente_ge_20pct": bool(
                                math.isfinite(rr.descente_rel)
                                and rr.descente_rel >= rt.SEUIL_PERTE_DESCENTE
                            ),
                            "r_disp_final": r_disp_fin,
                            "dbar_final": dbar_fin,
                            "state_sha256": rr.state_sha256,
                        }
                    )

        # Sélection de la meilleure config par cartouche (règle exacte de return_training)
        selection_cart: Dict[str, Dict[str, object]] = {}
        for cart in cartouches:
            s_raw_val = rt.scores_m1(m1, a_val[cart], ap_val[cart])
            pairs = rt.paires_dispersion(a_val[cart].shape[0])
            try:
                cfg_best, vivants = rt._meilleure_config(runs_pub, cart)
                dv_par_graine: List[Tuple[int, np.ndarray]] = []
                q6_all, q7_all = True, True
                for r in sorted(vivants, key=lambda item: item["seed"]):
                    model = rt.modele_depuis_poids(cfg_best["init_scale"], states_cache[r["run_id"]])
                    x_val = rt.appliquer_chrono(model, a_val[cart])
                    dv = rt.deltas(m1, x_val, ap_val[cart], s_raw_val)
                    r_disp, dbar = rt.gardes_collapse(x_val, a_val[cart], pairs)
                    q6_all = q6_all and (r_disp >= rt.R_DISP_MIN)
                    q7_all = q7_all and (dbar >= rt.DBAR_MIN)
                    dv_par_graine.append((int(r["seed"]), dv))

                meds = {str(s): float(np.median(d)) for s, d in dv_par_graine}
                med_val = float(np.median(list(meds.values())))
                # Graine médiane pour l'appariement cycle-à-cycle
                seed_med = min(
                    (s for s, d in dv_par_graine if float(np.median(d)) == med_val),
                    default=dv_par_graine[0][0],
                )
                vecteurs_deltas_val[(spec.nom, cart)] = dict(dv_par_graine)[seed_med]
                selection_cart[cart] = {
                    "config_retenue": cfg_best,
                    "n_graines_vivantes": len(vivants),
                    "delta_med_par_graine": meds,
                    "graine_mediane": seed_med,
                    "Delta_val_mediane": med_val,
                    "gardes_Q6_Q7_PASS": bool(q6_all and q7_all),
                }
            except RuntimeError:
                selection_cart[cart] = {"erreur": "toutes les configs ont divergé (I7)"}

        n_desc = sum(1 for r in runs_pub if r["descente_ge_20pct"])
        resultats_par_opt[spec.nom] = {
            "description": spec.description,
            "clip_norm": spec.clip_norm,
            "duree_s": time.perf_counter() - t_opt,
            "n_runs": len(runs_pub),
            "n_diverges": n_diverges,
            "repli_bounded_requis": bool(n_diverges >= rt.SEUIL_DIVERGENCES_REPLI),
            "n_descente_ge_20pct": n_desc,
            "selection_validation": selection_cart,
            "runs": runs_pub,
        }

    # 2. Confrontations appariées cycle-à-cycle sur VALIDATION (Sign-test exact + Wilcoxon INFO)
    confrontations: Dict[str, Dict[str, object]] = {}
    couples = [
        ("chrono_opt_c_outside", "adam_clipped_t64"),
        ("chrono_opt_c_outside", "adamw_clipped"),
        ("chrono_opt_c_outside", "adamw_unclipped"),
        ("chrono_opt_c_outside", "chrono_opt_c_inside"),
        ("chrono_opt_c_outside", "chrono_opt_v0_spec"),
        ("spiral_closure", "adam_clipped_t64"),
        ("spiral_closure", "adamw_clipped"),
        ("spiral_closure", "adamw_unclipped"),
        ("spiral_closure", "chrono_opt_c_outside"),
    ]
    for cand_a, cand_b in couples:
        cle_couple = f"{cand_a}_VS_{cand_b}"
        confrontations[cle_couple] = {}
        for cart in cartouches:
            if (cand_a, cart) in vecteurs_deltas_val and (cand_b, cart) in vecteurs_deltas_val:
                diff = vecteurs_deltas_val[(cand_a, cart)] - vecteurs_deltas_val[(cand_b, cart)]
                st = rt.sign_test_exact(diff)
                wx = rt.wilcoxon_approx(diff)
                med_diff = float(np.median(diff))
                confrontations[cle_couple][cart] = {
                    "mediane_diff_appariee": med_diff,
                    "moyenne_diff_INFO": float(diff.mean()),
                    "sign_test_exact": st,
                    "wilcoxon_INFO": wx,
                    "avantage_significatif_p001": bool(med_diff > 0.0 and st["p"] < rt.SEUIL_P),
                }

    return {
        "chantier": "OPTIMISEUR_ABA_ETAPE_4",
        "mode_donnees": source_mode,
        "duree_totale_s": time.perf_counter() - t0,
        "epochs_max": epochs_max,
        "parite_bit_a_bit_T64": parite_t64,
        "resultats_par_optimiseur": resultats_par_opt,
        "confrontations_appariees_validation": confrontations,
    }


def ecrire_rapport_scelle(payload: Dict[str, object], out_path: Path) -> Tuple[str, Path]:
    """Écrit le rapport JSON trié et son fichier de sceau ``.sha256``."""
    texte = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out_path.write_text(texte, encoding="utf-8")
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    seal_path = out_path.with_suffix(out_path.suffix + ".sha256")
    seal_path.write_text(sha + "\n", encoding="utf-8")
    return sha, seal_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Racine contenant dataset_aba.txt et Tokenizer/ (défaut: .)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("BENCHMARK_CHRONO_OPT_VS_ADAMW.json"),
        help="Chemin du rapport JSON de sortie",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Utilise des données 19D synthétiques (si .so ou dataset_aba.txt absents)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Mode rapide (epochs_max=30, cartouche S seule, graine 0 seule)",
    )
    args = parser.parse_args(argv)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)  # modèle 19D : le parallélisme intra-op coûte plus qu'il ne rapporte
    epochs = 30 if args.quick else rt.EPOCHS_MAX
    cartouches = ("S",) if args.quick else rt.CARTOUCHES
    graines = (0,) if args.quick else rt.GRAINES

    payload = executer_benchmark(
        args.root,
        synthetic=args.synthetic,
        epochs_max=epochs,
        cartouches=cartouches,
        graines=graines,
    )
    sha, seal_path = ecrire_rapport_scelle(payload, args.out)

    print(f"Parité T64 : PASS (sha256={payload['parite_bit_a_bit_T64']['state_sha256'][:12]}...)")
    for nom, res in payload["resultats_par_optimiseur"].items():
        sel_s = res["selection_validation"].get("S", {})
        d_med = sel_s.get("Delta_val_mediane", float("nan"))
        q67 = sel_s.get("gardes_Q6_Q7_PASS", False)
        print(
            f"  - {nom:22s} | diverges={res['n_diverges']:2d}/{res['n_runs']:2d} "
            f"| desc>=20%={res['n_descente_ge_20pct']:2d} "
            f"| Delta_S_val={d_med:+.4f} | Q6/Q7={q67}"
        )
    print(f"Rapport scellé : {args.out} (sha256={sha}) -> {seal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())