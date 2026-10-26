"""Tests de ``examples/benchmark_chrono_opt_tasks.py`` (aucun symbole sans test).

Chargé via ``importlib`` (``examples/`` n'est pas un paquet). Échelle réduite : on
exerce toutes les branches (tâches, perturbation, métrique, sélection, sceau, CLI)
en quelques secondes, et on vérifie le déterminisme au bit des rapports.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import torch


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmark_chrono_opt_tasks.py"
    spec = importlib.util.spec_from_file_location("benchmark_chrono_opt_tasks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()

NOMS = [
    "adam", "adamw", "adamw_clip", "spiral", "spiral_sans_retour", "spiral_sans_borne",
    "chrono_v1", "chrono_v1_wd", "chrono_v1_sans_regulation", "chrono_v0_spec",
]


def test_optimiseurs_instanciables() -> None:
    specs = bench.optimiseurs()
    assert [s.nom for s in specs] == NOMS
    assert [s.nom for s in specs if s.clip is not None] == ["adamw_clip"]
    p = [torch.nn.Parameter(torch.ones(3, dtype=torch.float64))]
    for s in specs:
        opt = s.make(p, 1e-2)
        p[0].grad = torch.ones(3, dtype=torch.float64)
        opt.step()
        assert torch.isfinite(p[0]).all()


def test_taches_formes_et_determinisme() -> None:
    t1, t2 = bench.taches(echelle=0.01), bench.taches(echelle=0.01)
    assert [t.nom for t in t1] == ["spirales", "mal_conditionnee", "pics"]
    for a, b in zip(t1, t2):
        for sa, sb in ((a.train, b.train), (a.val, b.val), (a.test, b.test)):
            assert torch.equal(sa[0], sb[0]) and torch.equal(sa[1], sb[1])
    spir = t1[0]
    assert spir.train[0].shape == (1024, 2) and set(spir.train[1].tolist()) == {0, 1}
    assert spir.metrique is not None
    assert t1[2].perturber is not None and t1[0].perturber is None


def test_perturbation_pics_multiplie_les_cibles() -> None:
    pics = bench.tache_pics(steps=1)
    y = torch.ones(4, 1, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    sorties = [pics.perturber(y, g) for _ in range(500)]
    n_pics = sum(1 for s in sorties if float(s.max()) == 1e4)
    assert 0 < n_pics < 50  # ≈ 2 %
    assert all(float(s.max()) in (1.0, 1e4) for s in sorties)


def test_entrainer_signale_la_divergence() -> None:
    """Un run qui explose (SGD à lr=1e6) est marqué non fini, perte val = inf."""
    tache = bench.tache_mal_conditionnee(steps=50)
    sgd = bench.Optimiseur("sgd_fou", lambda p, lr: torch.optim.SGD(p, lr=lr))
    res = bench.entrainer(tache, sgd, 1e6, 0)
    assert res["fini"] is False
    assert res["val"]["perte"] == float("inf")


def test_executer_selection_et_cli_deterministes(tmp_path: Path) -> None:
    payload = bench.executer(echelle=0.01, graines=(0,), lrs=(1e-2, 3e-2))
    for tache, par_opt in payload["resultats"].items():
        assert list(par_opt) == NOMS
        for r in par_opt.values():
            assert r["lr_retenu"] in (1e-2, 3e-2)
            vals = {lr: v["perte_val_moyenne"] for lr, v in r["grille"].items()}
            assert vals[repr(r["lr_retenu"])] == min(vals.values())
            assert math.isfinite(r["perte_test_moyenne"])
        if tache == "spirales":
            assert all(0.0 <= r["exactitude_test_moyenne"] <= 1.0 for r in par_opt.values())
    assert len(bench.tableau(payload)) == 3 * len(NOMS)

    out1, out2 = tmp_path / "t1.json", tmp_path / "t2.json"
    threads = torch.get_num_threads()
    try:
        assert bench.main(["--quick", "--out", str(out1)]) == 0
        assert bench.main(["--quick", "--out", str(out2)]) == 0
    finally:
        torch.set_num_threads(threads)
    assert out1.read_bytes() == out2.read_bytes()
    seal = (tmp_path / "t1.json.sha256").read_text(encoding="utf-8").strip()
    assert seal == hashlib.sha256(out1.read_bytes()).hexdigest()
    assert json.loads(out1.read_text(encoding="utf-8"))["echelle"] == 0.05
