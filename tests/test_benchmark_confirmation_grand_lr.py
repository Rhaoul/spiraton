"""Tests de ``examples/benchmark_confirmation_grand_lr.py`` (aucun symbole sans test).

Chargé via ``importlib`` (``examples/`` n'est pas un paquet). On exerce la grille, le
journal d'AlphaOmegaOpt (trajectoire identique au bit), la relecture en fin d'époque,
la lecture de H4 et H4m (y compris les runs non finis), le blob git indépendant des fins
de ligne et le CLI (mode synthétique rapide), déterministe au bit, y compris en
calcul parallèle.
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

from spiraton.experimental.alpha_omega_optimizer import AlphaOmegaOpt


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmark_confirmation_grand_lr.py"
    spec = importlib.util.spec_from_file_location("benchmark_confirmation_grand_lr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()


def test_grille_et_candidats() -> None:
    assert bench.LRS_H4 == (1e-1, 3e-1, 1.0)
    assert bench.INITS_H4 == (0.1, 0.6, 1.5)
    assert bench.GRAINES_H4 == (3, 4, 5)
    assert [s.nom for s in bench.candidats_h4()] == list(bench.CANDIDATS_H4)
    liste = bench.taches(200, ("S", "P"), bench.INITS_H4, bench.LRS_H4, bench.GRAINES_H4)
    assert len(liste) == 108 and len(set(liste)) == 108
    assert not {t[4] for t in liste} & {0, 1, 2}  # graines neuves seulement


def _trajectoire(cls, n_pas: int = 40):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(5, dtype=torch.float64))
    opt = cls([p], lr=0.3, weight_decay=1e-2)
    cible = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    gen = torch.Generator().manual_seed(1)
    for _ in range(n_pas):
        opt.zero_grad()
        bruit = 0.1 * torch.randn(5, dtype=torch.float64, generator=gen)
        loss = (p - cible - bruit).pow(2).sum() * (1.0 + p.pow(2).sum())
        loss.backward()
        opt.step()
    return p.detach().clone(), opt


def test_journal_ne_touche_pas_la_trajectoire() -> None:
    p_ref, opt_ref = _trajectoire(AlphaOmegaOpt)
    p_jou, opt_jou = _trajectoire(bench.AlphaOmegaOptJournal)
    assert torch.equal(p_ref, p_jou)
    assert len(opt_jou.journal_g) == len(opt_jou.journal_omega_bar) == 40
    assert opt_jou.journal_g[-1] == opt_ref.param_groups[0]["g"]
    assert opt_jou.journal_omega_bar[-1] == opt_ref.param_groups[0]["omega_bar"]
    assert opt_jou.journal_g[0] == opt_jou.defaults["g0"]  # pas de retour au premier pas


def test_fins_d_epoque() -> None:
    journal = [float(k) for k in range(12)]
    assert bench.fins_d_epoque(journal, 4, 3) == [3.0, 7.0, 11.0]
    assert bench.fins_d_epoque(journal[:10], 4, 2) == [3.0, 7.0]  # époque interrompue ignorée


def _run(nom, cart, init, seed, lr, best, g=None):
    r = {"optimiseur": nom, "cartouche": cart, "init_scale": init, "seed": seed, "lr": lr,
         "best_delta_val": best, "diverged": not math.isfinite(best), "apprend": best > 0,
         "best_epoch": 3}
    if nom == "alpha_omega":
        r["g_meilleure_epoque"] = g
        r["lr_effectif_meilleure_epoque"] = lr * g
    return r


def test_lecture_h4_et_h4m() -> None:
    lrs = bench.LRS_H4
    runs = []
    for seed in range(16):
        for lr in lrs:
            runs.append(_run("alpha_omega", "S", 0.6, seed, lr, 0.35, g=0.03 / lr))
            runs.append(_run("adamw_montee_courte_clipped", "S", 0.6, seed, lr, 0.30))
    # unité 16 : AO non fini à lr = 1 → D = −∞ ; unité 17 : les deux non finis → D = 0
    for lr in lrs:
        runs.append(_run("alpha_omega", "S", 0.6, 16, lr, float("nan") if lr == 1.0 else 0.4, g=0.5))
        runs.append(_run("adamw_montee_courte_clipped", "S", 0.6, 16, lr, 0.3))
        runs.append(_run("alpha_omega", "S", 0.6, 17, lr, float("-inf"), g=float("nan")))
        runs.append(_run("adamw_montee_courte_clipped", "S", 0.6, 17, lr, float("nan")))
    d = bench.ecarts_par_unite(runs, "alpha_omega", "adamw_montee_courte_clipped", lrs)
    assert math.isclose(d[("S", 0.6, 0)], 0.05)
    assert d[("S", 0.6, 16)] == -math.inf and d[("S", 0.6, 17)] == 0.0
    hyp = bench.lire_hypotheses(runs, lrs)
    h4 = hyp["H4_alpha_omega_vs_montee_courte_grand_lr"]
    assert h4["sign_test_exact"]["pos"] == 16 and h4["sign_test_exact"]["neg"] == 1
    assert h4["sign_test_exact"]["zeros"] == 1 and h4["confirme_p001"] is True
    eg = bench.ecarts_log_g(runs, 0.1, 1.0)
    assert math.isclose(eg[("S", 0.6, 0)], math.log(10.0))
    assert eg[("S", 0.6, 16)] == 0.0 and eg[("S", 0.6, 17)] == 0.0  # g égal ; g non fini
    h4m = hyp["H4m_g_baisse_quand_lr_monte"]
    assert (h4m["lr_bas"], h4m["lr_haut"], h4m["unites"]) == (0.1, 1.0, 18)
    assert h4m["sign_test_exact"]["pos"] == 16 and h4m["sign_test_exact"]["zeros"] == 2
    assert h4m["confirme_p001"] is True
    assert math.isclose(hyp["INFO_lr_effectif_meilleure_epoque_mediane"][repr(1.0)], 0.03)
    assert hyp["H_NaN_divergences_non_finies"] == {"adamw_montee_courte_clipped": 3,
                                                    "alpha_omega": 4}


def test_blob_git_independant_des_fins_de_ligne(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.md", tmp_path / "crlf.md"
    lf.write_bytes(b"abc\n")
    crlf.write_bytes(b"abc\r\n")
    # ``git hash-object`` d'un fichier contenant "abc\n"
    assert bench.blob_git_lf(lf) == bench.blob_git_lf(crlf) == "8baef1b4abc478178b004d62031cf7fe6db6f903"
    assert bench.blob_git_lf(tmp_path / "absent.md") is None


def test_cli_rapide_synthetique_deterministe(tmp_path: Path, monkeypatch) -> None:
    # Sous Windows (spawn), les processus de calcul réimportent les bancs par leur nom.
    monkeypatch.syspath_prepend(str(Path(bench.__file__).resolve().parent))
    out1, out2 = tmp_path / "c1.json", tmp_path / "c2.json"
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    try:
        assert bench.main(["--synthetic", "--quick", "--out", str(out1)]) == 0
        assert bench.main(["--synthetic", "--quick", "--jobs", "2", "--out", str(out2)]) == 0
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(det)
    assert out1.read_bytes() == out2.read_bytes()
    seal = (tmp_path / "c1.json.sha256").read_text(encoding="utf-8").strip()
    assert seal == hashlib.sha256(out1.read_bytes()).hexdigest()
    payload = json.loads(out1.read_text(encoding="utf-8"))
    assert payload["mode_donnees"] == "SYNTHETIC_19D_FLOAT64"
    assert payload["parite_bit_a_bit_T64"]["PASS"] is True
    assert payload["porte_rejeu_INFO"] == {"sans_objet": "données synthétiques"}
    assert len(payload["runs"]) == 2 * 3
    for r in payload["runs"]:
        if r["optimiseur"] == "alpha_omega":
            assert len(r["g_fin_epoque"]) == r["epochs_run"] == len(r["omega_bar_fin_epoque"])
            assert r["g_meilleure_epoque"] == r["g_fin_epoque"][r["best_epoch"] - 1]
        else:
            assert "g_fin_epoque" not in r
    assert "duree_totale_s_INFO" not in payload
