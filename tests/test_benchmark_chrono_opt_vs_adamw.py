from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from spiraton.experimental import return_training as rt
from spiraton.experimental.chrono_optimizer import ChronoSpiratonOpt
from spiraton.experimental.spiral_optimizer import SpiralClosureOpt


def _load_benchmark_module() -> ModuleType:
    """Charge examples/benchmark_chrono_opt_vs_adamw.py sans exiger examples/__init__.py."""
    root = Path(__file__).resolve().parents[1]
    script_path = root / "examples" / "benchmark_chrono_opt_vs_adamw.py"
    spec = importlib.util.spec_from_file_location("benchmark_chrono_opt_vs_adamw", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


def test_candidats_optimiseurs_et_specs() -> None:
    """Vérifie les 7 candidats, leurs gardes de clip_norm et l'instanciation des optimiseurs."""
    specs = bench.candidats_optimiseurs(weight_decay_adamw=1e-2)
    noms = [s.nom for s in specs]
    assert noms == [
        "adam_clipped_t64",
        "adamw_clipped",
        "adamw_unclipped",
        "chrono_opt_c_outside",
        "chrono_opt_c_inside",
        "chrono_opt_v0_spec",
        "spiral_closure",
    ]

    p = [torch.nn.Parameter(torch.ones(4, dtype=torch.float64))]
    by_name = {s.nom: s for s in specs}

    # Les 2 baselines clippées utilisent exactement rt.CLIP_NORM (1.0)
    assert by_name["adam_clipped_t64"].clip_norm == rt.CLIP_NORM
    assert by_name["adamw_clipped"].clip_norm == rt.CLIP_NORM

    # Les 3 candidats de test de stabilité brute fonctionnent sans clip_grad_norm_
    assert by_name["adamw_unclipped"].clip_norm is None
    assert by_name["chrono_opt_c_outside"].clip_norm is None
    assert by_name["chrono_opt_c_inside"].clip_norm is None
    assert by_name["chrono_opt_v0_spec"].clip_norm is None
    assert by_name["spiral_closure"].clip_norm is None

    opt_out = by_name["chrono_opt_c_outside"].make_opt(p, 1e-3)
    opt_in = by_name["chrono_opt_c_inside"].make_opt(p, 1e-3)
    assert isinstance(opt_out, ChronoSpiratonOpt) and opt_out.cfg.c_outside is True
    assert isinstance(opt_in, ChronoSpiratonOpt) and opt_in.cfg.c_outside is False
    opt_v0 = by_name["chrono_opt_v0_spec"].make_opt(p, 1e-3)
    assert isinstance(opt_v0, ChronoSpiratonOpt) and opt_v0.cfg.legacy_v0 is True
    opt_sp = by_name["spiral_closure"].make_opt(p, 1e-3)
    assert isinstance(opt_sp, SpiralClosureOpt)
    assert opt_sp.defaults["weight_decay"] == 1e-2


def test_synthetic_m1_et_donnees() -> None:
    """Vérifie la parité de formule cos - l2 de _SyntheticM1 et le déterminisme 19D."""
    m1_a, a_tr_a, ap_tr_a, a_val_a, ap_val_a = bench._donnees_synthetiques(n_tr=16, n_val=8)
    m1_b, a_tr_b, ap_tr_b, a_val_b, ap_val_b = bench._donnees_synthetiques(n_tr=16, n_val=8)

    for cart in rt.CARTOUCHES:
        assert a_tr_a[cart].shape == (16, rt.STATE_SIZE)
        assert a_val_a[cart].shape == (8, rt.STATE_SIZE)
        assert a_tr_a[cart].dtype == np.float64
        assert np.array_equal(a_tr_a[cart], a_tr_b[cart])
        assert np.array_equal(ap_tr_a[cart], ap_tr_b[cart])
        assert np.array_equal(a_val_a[cart], a_val_b[cart])
        assert np.array_equal(ap_val_a[cart], ap_val_b[cart])

    # Vérification analytique de score_retour : sur x == ap non nul, cos=1 et l2=0 => score=1.0
    v = np.ones(rt.STATE_SIZE, dtype=np.float64)
    assert abs(m1_a.score_retour(v, v) - 1.0) < 1e-11


def test_verifier_parite_t64_succes_et_detection_derive() -> None:
    """Porte G0 : parité bit-à-bit avec return_training.train_run et rejet si dérive."""
    torch.use_deterministic_algorithms(True)
    m1, a_tr, ap_tr, a_val, ap_val = bench._donnees_synthetiques(n_tr=64, n_val=32)
    m = rt.mesurer_m(a_tr["S"], ap_tr["S"])
    s_raw_val = rt.scores_m1(m1, a_val["S"], ap_val["S"])

    specs = bench.candidats_optimiseurs()
    rapport_parite = bench.verifier_parite_t64(
        m1,
        specs[0],  # adam_clipped_t64
        a_tr["S"],
        ap_tr["S"],
        a_val["S"],
        ap_val["S"],
        s_raw_val,
        m,
        epochs_check=4,
    )
    assert rapport_parite["PASS"] is True
    assert len(rapport_parite["state_sha256"]) == 64

    # Contre-test : passer un autre optimiseur à la porte de parité T64 DOIT lever RuntimeError
    with pytest.raises(RuntimeError, match="Parité T64 brisée"):
        bench.verifier_parite_t64(
            m1,
            specs[3],  # chrono_opt_c_outside != Adam T64
            a_tr["S"],
            ap_tr["S"],
            a_val["S"],
            ap_val["S"],
            s_raw_val,
            m,
            epochs_check=4,
        )


def test_ecrire_rapport_scelle_sha256(tmp_path: Path) -> None:
    """Vérifie l'écriture du rapport JSON et l'exactitude de son sceau .sha256."""
    out_json = tmp_path / "rapport_test.json"
    payload = {"chantier": "TEST", "valeurs": [1, 2, 3]}
    sha, seal_path = bench.ecrire_rapport_scelle(payload, out_json)

    assert out_json.is_file()
    assert seal_path.is_file()
    assert seal_path.name == "rapport_test.json.sha256"

    sha_reel = hashlib.sha256(out_json.read_bytes()).hexdigest()
    assert sha == sha_reel
    assert seal_path.read_text(encoding="utf-8").strip() == sha_reel


def test_executer_benchmark_et_cli_main_deterministes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teste le pipeline complet et l'entrée CLI main() en mode --synthetic --quick.

    On réduit la taille synthétique à (n_tr=64, n_val=32) pour que le test unitaire
    s'exécute en moins d'une seconde tout en exerçant 100 % des branches du code.
    """
    orig_synth = bench._donnees_synthetiques
    monkeypatch.setattr(
        bench, "_donnees_synthetiques", lambda n_tr=64, n_val=32: orig_synth(n_tr=64, n_val=32)
    )

    out1 = tmp_path / "bench_run1.json"
    out2 = tmp_path / "bench_run2.json"

    rc1 = bench.main(["--synthetic", "--quick", "--out", str(out1)])
    rc2 = bench.main(["--synthetic", "--quick", "--out", str(out2)])
    assert rc1 == 0
    assert rc2 == 0

    # Déterminisme strict : les deux exécutions produisent les mêmes octets de poids et de scores
    data1 = json.loads(out1.read_text(encoding="utf-8"))
    data2 = json.loads(out2.read_text(encoding="utf-8"))

    assert data1["parite_bit_a_bit_T64"] == data2["parite_bit_a_bit_T64"]
    assert data1["confrontations_appariees_validation"] == data2["confrontations_appariees_validation"]

    for opt_name in (
        "adam_clipped_t64",
        "adamw_clipped",
        "adamw_unclipped",
        "chrono_opt_c_outside",
        "chrono_opt_c_inside",
        "chrono_opt_v0_spec",
        "spiral_closure",
    ):
        r1 = data1["resultats_par_optimiseur"][opt_name]
        r2 = data2["resultats_par_optimiseur"][opt_name]
        assert r1["n_runs"] == 4  # 4 configs × 1 graine × 1 cartouche en mode --quick
        assert r1["selection_validation"] == r2["selection_validation"]
        assert r1["runs"] == r2["runs"]

    # Vérifie que toutes les confrontations appariées ont bien été calculées
    confs = data1["confrontations_appariees_validation"]
    assert "chrono_opt_c_outside_VS_adam_clipped_t64" in confs
    assert "chrono_opt_c_outside_VS_adamw_clipped" in confs
    assert "chrono_opt_c_outside_VS_chrono_opt_c_inside" in confs
    assert "chrono_opt_c_outside_VS_chrono_opt_v0_spec" in confs
    assert "spiral_closure_VS_adamw_clipped" in confs
    assert "spiral_closure_VS_adam_clipped_t64" in confs
    assert "sign_test_exact" in confs["chrono_opt_c_outside_VS_adamw_clipped"]["S"]
    assert "sign_test_exact" in confs["spiral_closure_VS_adamw_clipped"]["S"]