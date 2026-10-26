"""Quatuor T65 pour ``experimental/chrono_duration.py`` (CLAUDE.md, P0-b).

Seeds fixés, finitude, formes simple/batch, formule exacte sous paramètres
forcés (E1/E2/E3, substrats EXACTS de l'émission §5), flux de gradient sous
ablation, ablation exacte au bit, et PARITÉ de la boucle re-dérivée avec
``return_training.train_run`` (le cœur de la gate C4) sur données synthétiques.

Hermétique : aucun besoin du tokenizer natif — l'évaluateur M1 est doublé par
un shim qui reproduit la formule ``cos − l2`` (parité M1, docstring de
``perte_alpha_omega``) ; la parité C4 réelle (sur corpus) vit dans le runner.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from spiraton.experimental import chrono_duration as cd
from spiraton.experimental import return_training as rt
from spiraton.experimental.chrono import ChronoSpiraton

D = rt.STATE_SIZE
EPS = rt.EPS


class ShimM1:
    """Évaluateur ``score_retour`` = cos − l2 (l2 normalisé par ‖x‖, arg 1)."""

    @staticmethod
    def score_retour(x, ap):
        x = np.asarray(x, dtype=np.float64)
        ap = np.asarray(ap, dtype=np.float64)
        nx = float(np.linalg.norm(x))
        nap = float(np.linalg.norm(ap))
        cos = float(np.dot(x, ap)) / (nx * nap + EPS)
        l2 = float(np.linalg.norm(ap - x)) / (nx + EPS)
        return cos - l2


def _model_zero() -> ChronoSpiraton:
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        for nom in ("A", "B", "C", "D", "L"):
            getattr(model, nom).weight.zero_()
    return model


def _points(n: int = 12, seed: int = 650) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, D))


# --- Formule exacte sous paramètres forcés (E1/E2/E3) -------------------------

def test_e1_identite_lambda_zero_exact():
    model = _model_zero()
    with torch.no_grad():
        model.L.weight.copy_(torch.eye(D, dtype=torch.float64))
    pts = _points()
    tr = cd.derouler(cd.pas_chrono(model), pts, garder_etats=True)
    assert (tr.lam == 0.0).all()          # EXACT, pas une tolérance
    assert tr.f_div == 0.0
    assert (tr.etats[:, -1, :] == pts).all()   # s_64 = s_0 au bit
    assert (tr.t_prime == cd.K_MAX).all()


def test_e2_geometrique_log_alpha_et_rho0():
    alpha = 1.5
    model = _model_zero()
    with torch.no_grad():
        model.A.weight.copy_(torch.eye(D, dtype=torch.float64))
        model.D.weight.copy_(alpha * torch.eye(D, dtype=torch.float64))
    pts = _points()
    tr = cd.derouler(cd.pas_chrono(model), pts)
    # α=1.5 : divergence avant 64 pas (1.5^t > 1e6 vers t≈35) ⇒ censure propre.
    assert tr.diverged.all()
    rel = np.abs(tr.lam - math.log(alpha)) / math.log(alpha)
    assert rel.max() <= 1e-9              # eps du score au dénominateur
    assert cd.rho_compagnon(cd.matrices_np(model)) == alpha   # EXACT (émission E2)


def test_e2_contraction_douce_sans_censure():
    alpha = 0.9
    model = _model_zero()
    with torch.no_grad():
        model.A.weight.copy_(torch.eye(D, dtype=torch.float64))
        model.D.weight.copy_(alpha * torch.eye(D, dtype=torch.float64))
    tr = cd.derouler(cd.pas_chrono(model), _points())
    assert not tr.diverged.any()
    rel = np.abs(tr.lam - math.log(alpha)) / abs(math.log(alpha))
    assert rel.max() <= 1e-9


def test_e3_compagnon_contre_racines_quadratiques():
    rng = np.random.default_rng(651)
    diag_a = rng.uniform(-0.9, 0.9, D)
    diag_c = rng.uniform(-0.5, 0.5, D)
    diag_l = rng.uniform(-0.9, 0.9, D)
    model = _model_zero()
    with torch.no_grad():
        model.A.weight.copy_(torch.diag(torch.from_numpy(diag_a)))
        model.C.weight.copy_(torch.diag(torch.from_numpy(diag_c)))
        model.D.weight.copy_(torch.eye(D, dtype=torch.float64))
        model.L.weight.copy_(torch.diag(torch.from_numpy(diag_l)))
    racines = []
    for j in range(D):
        mm, cc = diag_a[j] + diag_l[j], diag_c[j]
        r = np.sqrt(complex(mm * mm - 4.0 * cc))
        racines.extend([abs((mm + r) / 2.0), abs((mm - r) / 2.0)])
    assert abs(cd.rho_compagnon(cd.matrices_np(model)) - max(racines)) <= 1e-10


# --- Formes simple/batch, finitude, seeds -------------------------------------

def test_formes_simple_et_batch():
    model = cd.trd.make_chrono(0)
    pts = _points(5)
    tr_batch = cd.derouler(cd.pas_chrono(model), pts)
    tr_seul = cd.derouler(cd.pas_chrono(model), pts[0])
    assert tr_batch.n == 5 and tr_seul.n == 1
    assert tr_seul.lam[0] == tr_batch.lam[0]


def test_finitude_et_determinisme():
    model = cd.trd.make_chrono(3)
    pts = _points(20, seed=42)
    tr1 = cd.derouler(cd.pas_chrono(model), pts)
    tr2 = cd.derouler(cd.pas_chrono(model), pts)
    assert np.isfinite(tr1.lam).all()
    assert (tr1.lam == tr2.lam).all() and (tr1.T == tr2.T).all()
    st = cd.stats_trajectoires(tr1)
    assert st["finitude_lam_100pct"] and math.isfinite(st["Lambda"])


def test_divergence_censure_et_tprime():
    # Croissance ×20 par pas : sortie de la borne 1e6 en ~4-5 pas, s_T FINI.
    model = _model_zero()
    with torch.no_grad():
        model.A.weight.copy_(torch.eye(D, dtype=torch.float64))
        model.D.weight.copy_(20.0 * torch.eye(D, dtype=torch.float64))
    pts = _points(6)
    tr = cd.derouler(cd.pas_chrono(model), pts)
    assert tr.diverged.all()
    assert (tr.T < cd.K_MAX).all()
    assert (tr.t_prime == tr.T).all()      # dernier pas fini = pas de sortie
    assert tr.h50 is not None and tr.h50 <= 6.0


# --- Ablation : exactitude au bit et flux de gradient -------------------------

def test_ablation_minus_c_forward_bit_identique():
    torch.manual_seed(7)
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        model.C.weight.zero_()
    rng = np.random.default_rng(650)
    s = torch.from_numpy(rng.standard_normal((8, D)))
    sp = torch.from_numpy(rng.standard_normal((8, D)))
    ref = model.D(model.A(s) + model.B(s * s)) + model.L(s)
    assert torch.equal(model.step(s, sp), ref)


def test_gradient_flux_sous_ablation():
    torch.manual_seed(11)
    model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
    with torch.no_grad():
        model.C.weight.zero_()
    model.C.weight.requires_grad_(False)
    rng = np.random.default_rng(1)
    a = torch.from_numpy(rng.standard_normal((16, D)))
    ap = torch.from_numpy(rng.standard_normal((16, D)))
    loss = rt.perte_alpha_omega(model, a, ap, 0.3)
    loss.backward()
    assert model.C.weight.grad is None            # ablatée : exactement None
    for nom in ("A", "B", "D", "L"):
        g = getattr(model, nom).weight.grad
        assert g is not None and float(g.abs().sum()) > 0.0


def test_init_appariee_entre_bras():
    """A, B, D, L bit-identiques entre FULL et MINUS_C à graine fixée (§4.4)."""
    def build(variante):
        torch.manual_seed(5)
        model = ChronoSpiraton(state_size=D, init_scale=0.1).double()
        for nom in cd.ABLATIONS[variante]:
            with torch.no_grad():
                getattr(model, nom).weight.zero_()
        return model
    full, moins_c = build("FULL"), build("MINUS_C")
    for nom in ("A", "B", "D", "L"):
        assert torch.equal(getattr(full, nom).weight, getattr(moins_c, nom).weight)
    assert (moins_c.C.weight == 0.0).all()


# --- Parité de la boucle re-dérivée (cœur de la gate C4) ----------------------

def test_parite_full_avec_train_run_synthetique():
    m1 = ShimM1()
    rng = np.random.default_rng(64)
    a_tr = rng.standard_normal((40, D))
    ap_tr = a_tr + 0.3 * rng.standard_normal((40, D))
    a_val = rng.standard_normal((20, D))
    ap_val = a_val + 0.3 * rng.standard_normal((20, D))
    s_raw_val = np.asarray([m1.score_retour(a_val[i], ap_val[i]) for i in range(20)])
    m = rt.mesurer_m(a_tr, ap_tr)
    ref = rt.train_run(m1, "S", cd.INIT_SCALE, cd.LR_VERDICT, 0,
                       a_tr, ap_tr, a_val, ap_val, s_raw_val, m, epochs_max=3)
    mien = cd.entrainer_variante(m1, "FULL", 0, cd.LR_VERDICT,
                                 a_tr, ap_tr, a_val, ap_val, s_raw_val, m,
                                 epochs_max=3)
    assert mien.state_sha256 == ref.state_sha256      # au bit
    assert mien.courbes == ref.courbes
    assert mien.loss_init == ref.loss_init


def test_entrainer_variante_determinisme():
    m1 = ShimM1()
    rng = np.random.default_rng(65)
    a_tr = rng.standard_normal((30, D))
    ap_tr = a_tr + 0.2 * rng.standard_normal((30, D))
    a_val = rng.standard_normal((15, D))
    ap_val = a_val + 0.2 * rng.standard_normal((15, D))
    s_raw_val = np.asarray([m1.score_retour(a_val[i], ap_val[i]) for i in range(15)])
    r1 = cd.entrainer_variante(m1, "MINUS_B", 1, cd.LR_VERDICT,
                               a_tr, ap_tr, a_val, ap_val, s_raw_val, 0.3,
                               epochs_max=2)
    r2 = cd.entrainer_variante(m1, "MINUS_B", 1, cd.LR_VERDICT,
                               a_tr, ap_tr, a_val, ap_val, s_raw_val, 0.3,
                               epochs_max=2)
    assert r1.state_sha256 == r2.state_sha256
    assert (r1.state["B.weight"] == 0.0).all()        # l'ablation persiste


# --- Sign-test exact : k*, puissances (récurrence Fraction) -------------------

def test_k_star_et_puissances_n50():
    k = cd.k_star_sign_test(50)
    assert k == 35
    pmf = cd.binom_pmf_fraction(50, Fraction(1, 2))
    assert sum(pmf, Fraction(0)) == 1
    p35 = 2 * sum(pmf[35:], Fraction(0))
    p34 = 2 * sum(pmf[34:], Fraction(0))
    assert p35 < Fraction(1, 100) <= p34
    pw75 = cd.puissance_sign_test(50, k, Fraction(3, 4))
    pw70 = cd.puissance_sign_test(50, k, Fraction(7, 10))
    assert 0.83 < pw75 < 0.85
    assert 0.55 < pw70 < 0.58


def test_regime_partition_et_garde():
    assert cd.regime_partition(0.95, 1e9, 2.0) == "DIVERGENT"
    assert cd.regime_partition(0.0, 10.0, 2.0) == "BORNE"
    assert cd.regime_partition(0.0, 100.0, 2.0) == "BASSIN"
    assert cd.regime_partition(0.5, 10.0, 2.0) == "BASSIN"


def test_scan_q11_sur_soi(tmp_path):
    propre = tmp_path / "propre.py"
    propre.write_text("x = 1\n", encoding="utf-8")
    sale = tmp_path / "sale.py"
    sale.write_text("y = " + "35" + "01" + "\n", encoding="utf-8")
    ok = cd.scan_q11([propre])
    assert ok["PASS"]
    ko = cd.scan_q11([propre, sale])
    assert not ko["PASS"] and len(ko["occurrences"]) == 1
