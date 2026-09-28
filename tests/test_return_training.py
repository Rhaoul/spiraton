# -*- coding: utf-8 -*-
"""Quatuor T64 — return_training (chantier 5 : l'apprentissage du retour).

Quatuor CLAUDE.md sur tout code neuf : (1) seeds fixés ; (2) finitude ;
(3) formes simple/batch ; (4) formule exacte sous paramètres forcés —
``A=B=C=D=0, L=I ⇒ Φ_K(v) = v`` AU BIT donc ``Δ_i = 0.0`` exact (le zéro de
la mesure est PROUVÉ, pas supposé), un pas de perte calculé à la main en d=2,
et la gate G0b (route de plage == ``agreger_corpus`` bit-à-bit, 100 % ou
FAUX) ; (5) flux de gradient (les 5 matrices A..L reçoivent un gradient non
nul sous la perte complète). Déterministe : aucun aléa non seedé.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from spiraton.data import aba, aba_forms, tokenizer_bridge
from spiraton.diagnostics import textual_return_dynamics as trd
from spiraton.experimental import return_training as rt
from spiraton.experimental.chrono import ChronoSpiraton

ROOT = Path(__file__).resolve().parents[2]  # racine spiraton-enhanced


def _m1_ou_skip():
    """Le module M1 (sans lib native — l'import ne charge pas le .so)."""
    try:
        return tokenizer_bridge.import_alpha_omega_text()
    except tokenizer_bridge.TokenizerUnavailable:
        pytest.skip("module alpha_omega_text introuvable")


def _instrument_ou_skip():
    try:
        return trd.charger_instrument()
    except tokenizer_bridge.TokenizerUnavailable:
        pytest.skip("tokenizer natif indisponible")


# --- (1) seeds fixés ----------------------------------------------------------

def test_init_seedee_reproductible_et_distincte() -> None:
    m0a = trd.make_chrono(0, init_scale=0.6)
    m0b = trd.make_chrono(0, init_scale=0.6)
    m1_ = trd.make_chrono(1, init_scale=0.6)
    assert rt.hash_etat(m0a.state_dict()) == rt.hash_etat(m0b.state_dict())
    assert rt.hash_etat(m0a.state_dict()) != rt.hash_etat(m1_.state_dict())


def test_paires_dispersion_gelees_sous_graine_64() -> None:
    p1 = rt.paires_dispersion(500)
    p2 = rt.paires_dispersion(500)
    assert p1 == p2 and len(p1) == 200
    assert all(i != j for i, j in p1)
    assert all(0 <= i < 500 and 0 <= j < 500 for i, j in p1)


# --- (2) finitude -------------------------------------------------------------

def test_perte_finie_sur_donnees_seedees() -> None:
    torch.manual_seed(200)
    a = torch.randn(16, rt.STATE_SIZE, dtype=torch.float64)
    ap = a + 0.1 * torch.randn(16, rt.STATE_SIZE, dtype=torch.float64)
    model = trd.make_chrono(0)
    loss = rt.perte_alpha_omega(model, a, ap, m=0.05)
    assert torch.isfinite(loss)


def test_train_run_court_est_fini_et_deterministe() -> None:
    m1 = _m1_ou_skip()
    rng = np.random.default_rng(64)
    a_tr = rng.standard_normal((12, rt.STATE_SIZE))
    ap_tr = a_tr + 0.2 * rng.standard_normal((12, rt.STATE_SIZE))
    a_va = rng.standard_normal((8, rt.STATE_SIZE))
    ap_va = a_va + 0.2 * rng.standard_normal((8, rt.STATE_SIZE))
    s_raw = rt.scores_m1(m1, a_va, ap_va)
    r1 = rt.train_run(m1, "S", 0.1, 1e-3, 0, a_tr, ap_tr, a_va, ap_va,
                      s_raw, m=0.05, epochs_max=3)
    r2 = rt.train_run(m1, "S", 0.1, 1e-3, 0, a_tr, ap_tr, a_va, ap_va,
                      s_raw, m=0.05, epochs_max=3)
    assert not r1.diverged
    assert all(np.isfinite(v).all() for v in
               [np.asarray(r1.courbes[k]) for k in r1.courbes])
    assert r1.state_sha256 == r2.state_sha256  # déterminisme bit-à-bit
    assert r1.courbes == r2.courbes


# --- (3) formes simple/batch --------------------------------------------------

def test_phi_theta_formes_simple_et_batch() -> None:
    model = trd.make_chrono(2)
    torch.manual_seed(201)
    v = torch.randn(rt.STATE_SIZE, dtype=torch.float64)
    seul = rt.phi_theta(model, v)
    lot = rt.phi_theta(model, v.unsqueeze(0)).squeeze(0)
    assert seul.shape == (rt.STATE_SIZE,)
    assert torch.allclose(seul, lot, atol=1e-12, rtol=0.0)


# --- (4) formule exacte sous paramètres forcés ---------------------------------

def _chrono_identite(d: int = rt.STATE_SIZE) -> ChronoSpiraton:
    """``A=B=C=D=0, L=I`` ⇒ chaque pas rend s_t inchangé (Φ_K = id)."""
    model = ChronoSpiraton(state_size=d, init_scale=0.1,
                           bounded=False, c_outside=False).double()
    with torch.no_grad():
        for op in (model.A, model.B, model.C, model.D):
            op.weight.zero_()
        model.L.weight.copy_(torch.eye(d, dtype=torch.float64))
    return model


def test_identite_forcee_phi_egal_v_au_bit_et_delta_zero_exact() -> None:
    m1 = _m1_ou_skip()
    model = _chrono_identite()
    torch.manual_seed(202)
    v = torch.randn(7, rt.STATE_SIZE, dtype=torch.float64)
    assert torch.equal(rt.phi_theta(model, v), v)  # au bit (substrat EXACT)
    a = v.numpy()
    ap = a + 0.3 * np.random.default_rng(3).standard_normal(a.shape)
    s_raw = rt.scores_m1(m1, a, ap)
    d = rt.deltas(m1, rt.appliquer_chrono(model, a), ap, s_raw)
    assert (d == 0.0).all()  # l'identité fait 0 EXACT — 100 % ou faux


def test_un_pas_de_perte_calcule_a_la_main_en_d2() -> None:
    model = ChronoSpiraton(state_size=2, init_scale=0.1,
                           bounded=False, c_outside=False).double()
    with torch.no_grad():
        model.A.weight.copy_(0.5 * torch.eye(2, dtype=torch.float64))
        model.B.weight.zero_()
        model.C.weight.zero_()
        model.D.weight.copy_(torch.eye(2, dtype=torch.float64))
        model.L.weight.zero_()
    a = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    ap = torch.tensor([[0.0, 2.0]], dtype=torch.float64)
    # Φ_2(a) = 0.25·a (dyadique, vérifié au bit).
    x = rt.phi_theta(model, a)
    assert torch.equal(x, torch.tensor([[0.25, 0.0]], dtype=torch.float64))
    # Perte à la main : cos = 0 ; l2 = √4.0625/(0.25+eps) ; d = 0.75/(1+eps).
    eps = rt.EPS
    l2 = math.sqrt(0.0625 + 4.0) / (0.25 + eps)
    d = 0.75 / (1.0 + eps)
    m = 1.0  # choisi > d pour exercer le terme de copie
    attendu = -(0.0 - l2) + rt.LAMBDA_COPY * (m - d) ** 2
    loss = rt.perte_alpha_omega(model, a, ap, m=m)
    assert math.isclose(loss.item(), attendu, rel_tol=0.0, abs_tol=1e-15)


def test_sign_test_exact_a_la_main() -> None:
    # n=5 tous positifs : p = 2·(1/32) = 0.0625 ; 4/5 : p = 2·6/32 = 0.375.
    r = rt.sign_test_exact(np.asarray([1.0, 2.0, 3.0, 0.5, 0.1]))
    assert r["pos"] == 5 and r["p"] == 0.0625
    r = rt.sign_test_exact(np.asarray([1.0, 2.0, 3.0, 0.5, -0.1]))
    assert r["pos"] == 4 and r["p"] == 0.375
    # zéros écartés, jamais comptés.
    r = rt.sign_test_exact(np.asarray([0.0, 0.0, 1.0]))
    assert r["n"] == 1 and r["zeros"] == 2 and r["p"] == 1.0


def test_ridge_close_form_recouvre_une_affine_exacte() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal((200, rt.STATE_SIZE))
    w = rng.standard_normal((rt.STATE_SIZE, rt.STATE_SIZE))
    b = rng.standard_normal(rt.STATE_SIZE)
    y = x @ w + b
    theta = rt.ajuster_ridge(x, y, lam=0.0)
    assert np.allclose(rt.predire_ridge(theta, x), y, atol=1e-9)
    # λ > 0 rétrécit ‖W‖_F (le biais n'est pas pénalisé).
    theta_reg = rt.ajuster_ridge(x, y, lam=1.0)
    assert np.linalg.norm(theta_reg[:-1]) < np.linalg.norm(theta[:-1])


def test_serialisation_poids_canonique_et_reproductible(tmp_path) -> None:
    model = trd.make_chrono(5)
    p1, p2 = tmp_path / "a.bin", tmp_path / "b.bin"
    sha1 = rt.sauver_poids(model.state_dict(), p1)
    sha2 = rt.sauver_poids(model.state_dict(), p2)
    assert sha1 == sha2 and p1.read_bytes() == p2.read_bytes()
    relu = rt.charger_poids(p1)
    for k, v in model.state_dict().items():
        assert torch.equal(relu[k], v)
    assert rt.hash_etat(relu) == rt.hash_etat(model.state_dict())


def test_gardes_collapse_deux_branches_atteignables() -> None:
    rng = np.random.default_rng(11)
    a = rng.standard_normal((50, rt.STATE_SIZE))
    pairs = rt.paires_dispersion(50)
    # Φ = id ⇒ R_disp = 1 (PASS Q6) mais d̄ = 0 (FAIL Q7 : identité déguisée).
    r, d = rt.gardes_collapse(a, a, pairs)
    assert math.isclose(r, 1.0) and d == 0.0
    # Φ constant ⇒ R_disp = 0 (FAIL Q6 : effondrement).
    r, d = rt.gardes_collapse(np.tile(a.mean(0), (50, 1)), a, pairs)
    assert r == 0.0 and d > 0.0


def test_mesurer_m_regle_gelee() -> None:
    a = np.asarray([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    ap = a + np.asarray([[0.0, 5.0], [0.3, 0.4], [0.0, 1.0]])
    # distances relatives : 1.0, 0.5, 0.5 → médiane 0.5 → m = 0.25.
    assert math.isclose(rt.mesurer_m(a, ap), 0.25, rel_tol=0.0, abs_tol=1e-12)


def test_g0b_route_de_plage_bit_a_bit() -> None:
    tok, m1 = _instrument_ou_skip()
    ds = ROOT / rt.DATASET
    if not ds.is_file():
        pytest.skip("dataset_aba.txt absent")
    ref = m1.agreger_corpus(tok, aba, aba_forms, str(ds), limit=30)
    mine, numeros, _ = rt.agreger_plage(tok, m1, ds, 1, 30)
    verdict = rt.comparer_cycleaggs(mine, ref)
    assert verdict["identique"] is True  # 100 % ou FAUX
    assert all(1 <= n <= 30 for n in numeros)
    # Branche FAIL atteignable : une plage décalée n'est PAS identique.
    decale, _, _ = rt.agreger_plage(tok, m1, ds, 2, 31)
    assert rt.comparer_cycleaggs(decale, ref)["identique"] is False


# --- (5) flux de gradient ------------------------------------------------------

def test_flux_de_gradient_sur_les_5_matrices() -> None:
    torch.manual_seed(203)
    model = trd.make_chrono(3)
    a = torch.randn(8, rt.STATE_SIZE, dtype=torch.float64)
    ap = a + 0.2 * torch.randn(8, rt.STATE_SIZE, dtype=torch.float64)
    loss = rt.perte_alpha_omega(model, a, ap, m=10.0)  # m grand : terme copie actif
    loss.backward()
    for nom in ("A", "B", "C", "D", "L"):
        grad = getattr(model, nom).weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0.0, f"gradient nul sur {nom}"


# --- constantes gelées du tour --------------------------------------------------

def test_constantes_gelees_du_tour() -> None:
    assert rt.SPLITS == {"TRAIN": (101, 1000), "VALIDATION": (3001, 3500),
                         "TEST": (3501, 5001)}
    assert rt.K_PORTEUR == 2 and rt.STATE_SIZE == 19
    assert rt.INIT_SCALES == (0.1, 0.6) and rt.LRS == (1e-3, 1e-2)
    assert rt.GRAINES == (0, 1, 2) and rt.BATCH == 64
    assert rt.EPOCHS_MAX == 200 and rt.PATIENCE == 20
    assert rt.LAMBDA_COPY == 1.0 and rt.CLIP_NORM == 1.0
    assert rt.RIDGE_LAMBDAS == (0.0, 1e-6, 1e-4, 1e-2, 1.0)
    assert len(rt.FREEZE_PATHS) == 16
    assert rt.SEUIL_C2_DELTA_S == 0.10 and rt.SEUIL_P == 0.01
    assert rt.SIGMA_DELTA_MAX == 1.1331
    assert len(rt.configs_grille()) == 4
