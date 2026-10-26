"""Tests de ``examples/benchmark_fenetre_lr.py`` (aucun symbole sans test).

Chargé via ``importlib`` (``examples/`` n'est pas un paquet). On exerce les
candidats, le calendrier du témoin à montée courte, le comptage des fenêtres, la
règle de sélection T64 sur la grille élargie et le CLI (mode synthétique rapide),
avec déterminisme au bit du rapport, y compris en calcul parallèle.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmark_fenetre_lr.py"
    spec = importlib.util.spec_from_file_location("benchmark_fenetre_lr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()

NOMS = ["adam_clipped_t64", "adamw_clipped", "adamw_unclipped", "spiral_closure",
        "chrono_opt_c_outside", "adamw_montee_courte_clipped", "alpha_omega"]


def test_grille_et_candidats() -> None:
    assert bench.LRS_FENETRE == (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    assert bench.INITS_FENETRE == (0.1, 0.6, 1.5)
    specs = bench.candidats()
    assert [s.nom for s in specs] == NOMS
    assert [s.nom for s in specs if s.clip_norm is not None] == [
        "adam_clipped_t64", "adamw_clipped", "adamw_montee_courte_clipped"]
    p = [torch.nn.Parameter(torch.ones(3, dtype=torch.float64))]
    for s in specs:
        opt = s.make_opt(p, 1e-2)
        p[0].grad = torch.ones(3, dtype=torch.float64)
        opt.step()
        assert torch.isfinite(p[0]).all()


def test_montee_courte_calendrier() -> None:
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    opt = bench.AdamWMonteeCourte([p], 0.5, g0=1e-2, pas_montee=4, weight_decay=0.0)
    assert [opt.facteur(t) for t in (0, 2, 4, 9)] == [1e-2, 1e-1, 1.0, 1.0]
    vus = []
    for _ in range(6):
        p.grad = torch.ones(2, dtype=torch.float64)
        opt.step()
        vus.append(opt.param_groups[0]["lr"])
    assert vus == [0.5 * 1e-2, 0.5 * 1e-2 ** 0.75, 0.5 * 1e-1, 0.5 * 1e-2 ** 0.25, 0.5, 0.5]


def _run(nom, cart, init, lr, seed, apprend, best=0.5, dv=None, diverged=False):
    return {"optimiseur": nom, "cartouche": cart, "init_scale": init, "lr": lr, "seed": seed,
            "apprend": apprend, "diverged": diverged, "best_delta_val": best,
            "_dv": np.zeros(3) if dv is None else np.asarray(dv, dtype=np.float64)}


def test_n_apprend_et_comparaison() -> None:
    runs = []
    for seed in range(10):
        for lr in (1e-3, 1e-2, 1e-1):
            runs.append(_run("a", "S", 0.6, lr, seed, True))
            runs.append(_run("b", "S", 0.6, lr, seed, lr == 1e-2))
    assert bench.n_apprend(runs, "a") == {("S", 0.6, s): 3 for s in range(10)}
    assert bench.n_apprend(runs, "b") == {("S", 0.6, s): 1 for s in range(10)}
    res = bench.comparer_fenetres(runs, "a", "b")
    assert res["mediane_diff"] == 2.0 and res["somme_a"] == 30 and res["somme_b"] == 10
    assert res["sign_test_exact"]["pos"] == 10
    assert math.isclose(res["sign_test_exact"]["p"], 2 * 0.5 ** 10)
    assert res["a_plus_large_p001"] is True
    assert bench.comparer_fenetres(runs, "b", "a")["a_plus_large_p001"] is False


def test_selection_validation_regle_t64() -> None:
    runs = [
        _run("x", "S", 0.6, 1e-2, 0, True, best=0.30, dv=[0.1, 0.2, 0.3]),
        _run("x", "S", 0.6, 1e-2, 1, True, best=0.50, dv=[0.5, 0.6, 0.7]),
        _run("x", "S", 0.6, 1e-2, 2, True, best=0.40, dv=[0.3, 0.4, 0.5]),
        _run("x", "S", 0.6, 1e-1, 0, True, best=0.90, dv=[1.0, 1.0, 1.0], diverged=True),
        _run("x", "S", 0.6, 1e-1, 1, False, best=-0.2),
        _run("x", "S", 1.5, 1e-2, 0, False, best=-1.0),
    ]
    sel = bench.selection_validation(runs, "x", "S", (0.6, 1.5), (1e-2, 1e-1))
    assert (sel["init_scale"], sel["lr"]) == (0.6, 1e-2)  # 1e-1 : seule graine vivante à −0,2
    assert math.isclose(sel["moyenne_best_delta_val"], 0.4)
    assert sel["graine_mediane"] == 2 and math.isclose(sel["Delta_val_mediane"], 0.4)
    assert "erreur" in bench.selection_validation(runs, "absent", "S", (0.6,), (1e-2,))


def test_cli_rapide_synthetique_deterministe(tmp_path: Path, monkeypatch) -> None:
    # Sous Windows (spawn), les processus de calcul réimportent ce module par son nom :
    # examples/ doit être sur leur sys.path, hérité de celui du parent.
    monkeypatch.syspath_prepend(str(Path(bench.__file__).resolve().parent))
    out1, out2 = tmp_path / "f1.json", tmp_path / "f2.json"
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    try:
        assert bench.main(["--synthetic", "--quick", "--out", str(out1)]) == 0
        assert bench.main(["--synthetic", "--quick", "--jobs", "2", "--out", str(out2)]) == 0
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det)
    assert out1.read_bytes() == out2.read_bytes()
    seal = (tmp_path / "f1.json.sha256").read_text(encoding="utf-8").strip()
    assert seal == hashlib.sha256(out1.read_bytes()).hexdigest()
    payload = json.loads(out1.read_text(encoding="utf-8"))
    assert payload["mode_donnees"] == "SYNTHETIC_19D_FLOAT64"
    assert payload["parite_bit_a_bit_T64"]["PASS"] is True
    assert len(payload["runs"]) == len(NOMS) * 3
    assert set(payload["hypotheses"]["INFO_runs_qui_apprennent_total"]) == set(NOMS)
    assert "duree_totale_s_INFO" not in payload
    assert all(r["apprend"] == (not r["diverged"] and r["best_delta_val"] > 0)
               for r in payload["runs"])
