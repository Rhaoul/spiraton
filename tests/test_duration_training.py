"""Quatuor de tests du T67 (duration_training) — CLAUDE.md :

1. seeds fixés / déterminisme ; 2. finitude ; 3. formes (19,)/(B,19) ;
4. formule exacte (substrats §4.8 v-vii + pertes des bras re-dérivées) ;
plus flux de gradient (5 matrices sous chaque perte, WD analytique 1 pas),
instruments neufs (R_disp^64, extinction, partition close), découpe aba_v2
(gate structurelle Q12) et sélecteur g*.

Aucune sortie aléatoire non seedée. Aucun corpus requis : tout est
synthétique (le pont tokenizer n'est pas chargé ici).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from spiraton.experimental import chrono_duration as cd
from spiraton.experimental import duration_training as dt
from spiraton.experimental import return_training as rt
from spiraton.experimental.chrono import ChronoSpiraton

D = rt.STATE_SIZE


def modele_seed(seed: int) -> ChronoSpiraton:
    torch.manual_seed(seed)
    return ChronoSpiraton(state_size=D, init_scale=0.1).double()


def batch(seed: int, n: int = 8):
    rng = np.random.default_rng(seed)
    a = torch.from_numpy(rng.standard_normal((n, D)))
    ap = torch.from_numpy(rng.standard_normal((n, D)))
    return a, ap


# --- 1. Déterminisme -----------------------------------------------------------

def test_perte_bras_deterministe():
    a, ap = batch(1)
    for bras in dt.BRAS_TOUS:
        v1 = dt.perte_bras(bras, modele_seed(3), a, ap, dt.PARITE_M_S)
        v2 = dt.perte_bras(bras, modele_seed(3), a, ap, dt.PARITE_M_S)
        assert torch.equal(v1, v2), bras


def test_paires_67_deterministes():
    p1 = dt.paires_67(900)
    p2 = dt.paires_67(900)
    assert p1 == p2 and len(p1) == dt.N_PAIRES
    assert all(i != j for i, j in p1)


# --- 2. Finitude ---------------------------------------------------------------

def test_pertes_finies():
    a, ap = batch(2)
    for bras in dt.BRAS_TOUS:
        v = dt.perte_bras(bras, modele_seed(4), a, ap, dt.PARITE_M_S)
        assert torch.isfinite(v), bras


# --- 3. Formes -----------------------------------------------------------------

def test_formes_simple_et_batch():
    a, ap = batch(3, n=5)
    for bras in dt.BRAS_TOUS:
        vb = dt.perte_bras(bras, modele_seed(5), a, ap, dt.PARITE_M_S)
        assert vb.shape == ()
        v1 = dt.perte_bras(bras, modele_seed(5), a[0], ap[0], dt.PARITE_M_S)
        assert v1.shape == () and torch.isfinite(v1)


# --- 4. Formule exacte ---------------------------------------------------------

def test_substrat_v_identite_lambda_zero_delta_zero():
    """§4.8 (v) : A=B=C=D=0, L=I ⇒ s_t = s_0 ⇒ λ_i = 0 EXACT, Δ_i = 0 EXACT."""
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D"):
            getattr(model, nom).weight.zero_()
        model.L.weight.copy_(torch.eye(D, dtype=torch.float64))
    pts = np.random.default_rng(11).standard_normal((12, D))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    assert (trj.lam == 0.0).all() and trj.f_div == 0.0
    x = rt.appliquer_chrono(model, pts)
    assert np.array_equal(x, pts)  # Φ(a) = a au bit ⇒ Δ_i = 0 exact.


def test_substrat_vi_geometrique():
    """§4.8 (vi) : A=I, B=C=0, D=αI, L=0 ⇒ λ = log α (1e-12 rel), ρ₀ = α."""
    alpha = 1.5
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        model.A.weight.copy_(torch.eye(D, dtype=torch.float64))
        model.B.weight.zero_()
        model.C.weight.zero_()
        model.D.weight.copy_(alpha * torch.eye(D, dtype=torch.float64))
        model.L.weight.zero_()
    pts = np.random.default_rng(12).standard_normal((6, D))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    assert np.max(np.abs(trj.lam - math.log(alpha)) / math.log(alpha)) <= 1e-12
    assert cd.rho_compagnon(cd.matrices_np(model)) == alpha


def test_substrat_vii_perte_A0_bit_identique_T64():
    """§4.8 (vii) : la perte A0 est BIT-identique à rt.perte_alpha_omega
    (valeur ET gradients) sur un batch témoin gelé."""
    a, ap = batch(7)
    model = modele_seed(7)
    l_a0 = dt.perte_bras("A0", model, a, ap, dt.PARITE_M_S)
    l_t64 = rt.perte_alpha_omega(model, a, ap, dt.PARITE_M_S)
    assert torch.equal(l_a0, l_t64)
    g1 = torch.autograd.grad(l_a0, list(model.parameters()))
    l_t64b = rt.perte_alpha_omega(model, a, ap, dt.PARITE_M_S)
    g2 = torch.autograd.grad(l_t64b, list(model.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(g1, g2))


def test_formule_A1_moyenne_trois_horizons():
    a, ap = batch(8)
    model = modele_seed(8)
    attendu = (dt.ell_k(model, a, ap, dt.PARITE_M_S, 2)
               + dt.ell_k(model, a, ap, dt.PARITE_M_S, 4)
               + dt.ell_k(model, a, ap, dt.PARITE_M_S, 8)) / 3.0
    assert torch.equal(dt.perte_bras("A1", model, a, ap, dt.PARITE_M_S), attendu)


def test_formule_A2_penalite_croissance():
    """A2 = ℓ(2) + λ_stab·mean relu((1/8)·log(‖s_8‖/‖a‖))² — recomposée par
    une route indépendante (numpy) sur poids forcés."""
    a, ap = batch(9)
    model = modele_seed(9)
    with torch.no_grad():
        base = float(dt.ell_k(model, a, ap, dt.PARITE_M_S, 2))
        x8 = model(a, steps=8)
        total = float(dt.perte_bras("A2", model, a, ap, dt.PARITE_M_S))
    g = (np.log((np.linalg.norm(x8.numpy(), axis=1) + dt.EPS)
                / (np.linalg.norm(a.numpy(), axis=1) + dt.EPS)) / dt.K_PEN)
    pen = float(np.mean(np.maximum(g, 0.0) ** 2))
    assert abs(total - (base + dt.LAMBDA_STAB * pen)) <= 1e-12


# --- Flux de gradient ----------------------------------------------------------

def test_flux_gradient_cinq_matrices_chaque_bras():
    a, ap = batch(10)
    for bras in dt.BRAS_TOUS:
        model = modele_seed(10)
        loss = dt.perte_bras(bras, model, a, ap, dt.PARITE_M_S)
        grads = torch.autograd.grad(loss, list(model.parameters()))
        assert len(grads) == 5
        for g in grads:
            assert g is not None and float(g.abs().sum()) > 0.0, bras


def test_wd_analytique_un_pas():
    """Adam(weight_decay=wd) ≡ Adam(0) sur perte + (wd/2)·‖θ‖² (1 pas)."""
    a, ap = batch(11)
    m_wd = modele_seed(11)
    m_l2 = modele_seed(11)
    o_wd = torch.optim.Adam(m_wd.parameters(), lr=dt.LR, betas=rt.ADAM_BETAS,
                            eps=rt.ADAM_EPS, weight_decay=dt.WEIGHT_DECAY_WD)
    o_l2 = torch.optim.Adam(m_l2.parameters(), lr=dt.LR, betas=rt.ADAM_BETAS,
                            eps=rt.ADAM_EPS, weight_decay=0.0)
    o_wd.zero_grad()
    dt.perte_bras("WD", m_wd, a, ap, dt.PARITE_M_S).backward()
    o_wd.step()
    o_l2.zero_grad()
    (dt.perte_bras("WD", m_l2, a, ap, dt.PARITE_M_S)
     + (dt.WEIGHT_DECAY_WD / 2.0)
     * sum(p.pow(2).sum() for p in m_l2.parameters())).backward()
    o_l2.step()
    for p, q in zip(m_wd.parameters(), m_l2.parameters()):
        assert float((p - q).abs().max()) <= 1e-15


# --- Instruments neufs ---------------------------------------------------------

def _modele_identite() -> ChronoSpiraton:
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D"):
            getattr(model, nom).weight.zero_()
        model.L.weight.copy_(torch.eye(D, dtype=torch.float64))
    return model


def test_r_disp_64_identite_egale_un():
    pts = np.random.default_rng(13).standard_normal((40, D))
    pairs = dt.paires_67(40)
    trj = cd.derouler(cd.pas_chrono(_modele_identite()), pts)
    rd = dt.r_disp_64(trj, pts, pairs)
    assert rd["mesurable"] and rd["R_disp_64"] == 1.0


def test_r_disp_64_collapse_vers_zero():
    """Contraction forte (L = 0.01·I) : diversité perdue ⇒ R ≈ 0 < 0,20."""
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D"):
            getattr(model, nom).weight.zero_()
        model.L.weight.copy_(0.01 * torch.eye(D, dtype=torch.float64))
    pts = np.random.default_rng(14).standard_normal((40, D))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    rd = dt.r_disp_64(trj, pts, dt.paires_67(40))
    assert rd["mesurable"] and rd["R_disp_64"] < dt.R_DISP64_MIN


def test_r_disp_64_na_si_paires_divergees():
    """Explosion à overflow (L = 1e5·I ⇒ ‖s_64‖ → inf) : < 90 % de paires
    finies à t=64 ⇒ N-A publié (le régime est de toute façon DIVERGENT)."""
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D"):
            getattr(model, nom).weight.zero_()
        model.L.weight.copy_(1e5 * torch.eye(D, dtype=torch.float64))
    pts = np.random.default_rng(15).standard_normal((40, D))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    rd = dt.r_disp_64(trj, pts, dt.paires_67(40))
    assert not rd["mesurable"]


def test_partition_close_cinq_classes():
    pts = np.random.default_rng(16).standard_normal((40, D))
    pairs = dt.paires_67(40)

    def classe(l_diag: float) -> str:
        model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
        with torch.no_grad():
            for nom in ("A", "B", "C", "D"):
                getattr(model, nom).weight.zero_()
            model.L.weight.copy_(l_diag * torch.eye(D, dtype=torch.float64))
        trj = cd.derouler(cd.pas_chrono(model), pts)
        return dt.classer_regime(trj, pts, pairs)["classe"]

    assert classe(2.0) == "DIVERGENT"       # ‖s‖ ×2^64 ≫ Θ = 1e6.
    assert classe(1.2) == "BASSIN"          # croît ≫ 10× sans franchir Θ en 64 pas.
    assert classe(0.05) == "EXTINCTION"     # 0.05^64 ⇒ médiane ‖s_64‖ ≈ 0.
    assert classe(1.0) == "BORNE_VIVANT"    # identité : diversité conservée (R=1).
    # VERROUILLÉ : contraction homothétique 0.93^64 ≈ 0.0096 — borné, NON
    # éteint (> 1e-3), diversité ÷ 100 ⇒ R_disp^64 ≈ 0.0096 < 0.20 (l.341).
    res = None
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D"):
            getattr(model, nom).weight.zero_()
        model.L.weight.copy_(0.93 * torch.eye(D, dtype=torch.float64))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    res = dt.classer_regime(trj, pts, pairs)
    assert res["classe"] == "VERROUILLE"
    assert res["R_disp_64"]["mesurable"]


def test_extinction_mesure_identite_ne_mord_pas():
    pts = np.random.default_rng(17).standard_normal((20, D))
    trj = cd.derouler(cd.pas_chrono(_modele_identite()), pts)
    ext = dt.extinction_mesure(trj)
    assert ext["mesurable"] and not ext["mord"]


# --- Sélecteur g* et médiane basse ---------------------------------------------

def test_mediane_basse_rang_25():
    vals = {g: float(g) for g in range(50)}
    assert dt.mediane_basse(vals) == 24            # rang 25, convention basse.
    assert dt.mediane_basse({0: 1.0, 1: 2.0, 2: 3.0}) == 1
    assert dt.mediane_basse({5: 2.0, 9: 1.0}) == 9  # n pair : basse.


def test_selecteur_g_star_regles():
    duree_bras = {"survivants": list(range(50)),
                  "par_graine": {str(g): {
                      "classe": "BORNE_VIVANT" if g < 10 else "DIVERGENT"}
                      for g in range(50)}}
    deltas = {g: float(g) for g in range(50)}
    sel = dt._selecteur_g_star(duree_bras, deltas)
    assert sel["pool"] == "BORNE_VIVANT" and sel["g_star"] == 4  # (10−1)//2.
    duree_bras2 = {"survivants": list(range(50)),
                   "par_graine": {str(g): {"classe": "DIVERGENT"}
                                  for g in range(50)}}
    sel2 = dt._selecteur_g_star(duree_bras2, deltas)
    assert sel2["pool"] == "SURVIVANTS" and sel2["g_star"] == 24
    assert not sel2["porte_conjonction_tenue_et_transport"]
    # Repli §4.4 : les DIVERGED (hors survivants) n'entrent dans aucun pool.
    duree_bras3 = {"survivants": [g for g in range(50) if g % 2 == 0],
                   "par_graine": {str(g): {"classe": "DIVERGENT"}
                                  for g in range(50)}}
    sel3 = dt._selecteur_g_star(duree_bras3, {g: float(g) for g in range(0, 50, 2)})
    assert sel3["effectif_pool"] == 25 and sel3["g_star"] == 24


# --- Gates matérielles ----------------------------------------------------------

def test_tranche_reserve_refusee(tmp_path):
    """Gate Q12 structurelle : RESERVE_V2 (et toute tranche inconnue) n'est
    pas matérialisable — ValueError AVANT toute lecture du fichier."""
    with pytest.raises(ValueError):
        dt.charger_tranche_v2(None, None, tmp_path, "RESERVE_V2")
    with pytest.raises(ValueError):
        dt.charger_tranche_v2(None, None, tmp_path, "TOUT_LE_FICHIER")


def test_scan_q12_detecte_jetons(tmp_path):
    propre = tmp_path / "propre.py"
    propre.write_text("x = 1\n", encoding="utf-8")
    sale = tmp_path / "sale.py"
    sale.write_text("plage = (35" + "01, 50" + "01)\n", encoding="utf-8")
    ok = dt.scan_q12([propre])
    assert ok["PASS"]
    ko = dt.scan_q12([propre, sale])
    assert not ko["PASS"] and len(ko["occurrences"]) == 2


def test_contre_epreuve_stride_concordante():
    rng = np.random.default_rng(18)
    d = rng.standard_normal(2048) + 0.8       # nettement positif.
    ce = dt.contre_epreuve_stride(d)
    assert ce["n_sous"] == 512 and ce["concordant"]
    d2 = np.concatenate([np.full(1024, 1.0), np.full(1024, -1.0)])
    # signe de médiane plein = 0 vs sous-échantillon : la routine reste finie.
    ce2 = dt.contre_epreuve_stride(d2)
    assert "concordant" in ce2


def test_r_in_positif():
    pts = np.random.default_rng(19).standard_normal((100, D))
    val = dt.r_in(pts, dt.paires_67(100))
    assert np.isfinite(val) and val > 0.0
