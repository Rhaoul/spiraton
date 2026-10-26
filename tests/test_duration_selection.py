"""Tests T68 — duration_selection : quatuor + substrats exacts §4.7 + gates.

Quatuor du dépôt : seeds fixés, finitude, formes simples/batch, formule
exacte sous paramètres forcés (+ flux de gradient hérité de
``duration_training``, non re-testé ici : ``entrainer_bras`` est importé sans
modification et couvert par ``test_duration_training.py``). S'y ajoutent les
substrats EXACTS (iv)-(vii) de l'émission et les deux gates de portée :
« le score ne voit que 5 époques » (Q6) et « aucune route vers une graine
≥ 50 hors train_neuf » (Q10). Aucun test ne lit d'artefact hors dépôt
(CI-compatible) : les règles déterministes (jumeau, ε₅₀, top-m) sont testées
sur des structures synthétiques.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from spiraton.experimental import chrono_duration as cd
from spiraton.experimental import duration_selection as ds
from spiraton.experimental import duration_training as dt
from spiraton.experimental import return_training as rt
from spiraton.experimental.chrono import ChronoSpiraton


# --- Perturbation contrôlée (I-1) : formule exacte, seeds, ε = 0 --------------

def test_perturbation_eps0_bit_identique_chrono():
    torch.manual_seed(4242)
    model = ChronoSpiraton(state_size=7, init_scale=0.1).double()
    mats = ds.etat_chrono_vers_mats(model.state_dict())
    out = ds.perturber_matrices(mats, 0.0, ds.rng_perturbation(3, 5))
    assert set(out) == set(mats)
    for k in mats:
        assert np.array_equal(out[k], mats[k])


def test_perturbation_eps0_bit_identique_ridge():
    rng = np.random.default_rng(68)
    theta = rng.standard_normal((8, 7))
    out = ds.perturber_matrices(ds.theta_vers_mats(theta), 0.0,
                                ds.rng_perturbation(18, 0))
    assert np.array_equal(ds.mats_vers_theta(out), theta)


def test_perturbation_formule_exacte():
    """M'_X = X + ε·(‖X‖_F/√n_X)·G_X, tirages dans l'ordre des clés triées."""
    rng = np.random.default_rng(1)
    mats = {"B.weight": rng.standard_normal((4, 4)),
            "A.weight": rng.standard_normal((4, 4))}
    eps = 3e-2
    out = ds.perturber_matrices(mats, eps, ds.rng_perturbation(2, 7))
    ref_rng = np.random.default_rng(
        ds.SEED_PERTURBATION_BASE + 1000 * 2 + 7)
    for k in sorted(mats):  # A.weight puis B.weight
        g = ref_rng.standard_normal(mats[k].shape)
        attendu = mats[k] + eps * (np.linalg.norm(mats[k])
                                   / math.sqrt(mats[k].size)) * g
        assert np.array_equal(out[k], attendu)


def test_perturbation_seedee_reproductible_et_finie():
    torch.manual_seed(11)
    model = ChronoSpiraton(state_size=5, init_scale=0.1).double()
    mats = ds.etat_chrono_vers_mats(model.state_dict())
    a = ds.perturber_matrices(mats, 1e-2, ds.rng_perturbation(0, 0))
    b = ds.perturber_matrices(mats, 1e-2, ds.rng_perturbation(0, 0))
    c = ds.perturber_matrices(mats, 1e-2, ds.rng_perturbation(0, 1))
    for k in mats:
        assert np.array_equal(a[k], b[k])
        assert np.isfinite(a[k]).all()
    assert any(not np.array_equal(a[k], c[k]) for k in mats)


def test_perturbation_formes_simple_et_vecteur():
    """Formes : matrice (n,n) ET vecteur (n,) (le b de la ridge)."""
    mats = {"W": np.eye(3), "b": np.arange(3.0)}
    out = ds.perturber_matrices(mats, 1e-1, ds.rng_perturbation(18, 2))
    assert out["W"].shape == (3, 3) and out["b"].shape == (3,)
    assert np.isfinite(out["W"]).all() and np.isfinite(out["b"]).all()


# --- Substrats exacts (iv)/(v) : classification sur cas forcés ----------------

def test_substrat_iv_identite_lambda_zero():
    d = rt.STATE_SIZE
    mats = {f"{n}.weight": np.zeros((d, d)) for n in "ABCD"}
    mats["L.weight"] = np.eye(d)
    model = rt.modele_depuis_poids(0.1, ds.mats_vers_etat_chrono(mats))
    pts = np.random.default_rng(680).standard_normal((12, d))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    assert (trj.lam == 0.0).all() and trj.f_div == 0.0


def test_substrat_v_geometrique():
    d = rt.STATE_SIZE
    alpha = 1.5
    mats = {"A.weight": np.eye(d), "B.weight": np.zeros((d, d)),
            "C.weight": np.zeros((d, d)), "D.weight": alpha * np.eye(d),
            "L.weight": np.zeros((d, d))}
    model = rt.modele_depuis_poids(0.1, ds.mats_vers_etat_chrono(mats))
    pts = np.random.default_rng(681).standard_normal((12, d))
    trj = cd.derouler(cd.pas_chrono(model), pts)
    assert np.max(np.abs(trj.lam - math.log(alpha)) / math.log(alpha)) <= 1e-12
    assert cd.rho_compagnon(cd.matrices_np(model)) == alpha


# --- Substrat (vii) : Fraction, deux routes, valeurs connues ------------------

def test_hypergeom_deux_routes_identiques():
    for n_tot, k_tot, m in ((150, 12, 30), (150, 9, 30), (50, 4, 10),
                            (150, 12, 9), (10, 5, 5), (7, 3, 4)):
        for k in range(0, min(k_tot, m) + 1):
            assert ds.hyper_tail_ge(n_tot, k_tot, m, k) == \
                ds.hyper_tail_ge_recurrence(n_tot, k_tot, m, k)


def test_hypergeom_valeurs_connues():
    # P(X >= 5 | N=10, K=5, m=5) = 1/C(10,5) = 1/252.
    assert ds.hyper_tail_ge(10, 5, 5, 5) == Fraction(1, 252)
    # P(X >= 0) = 1 toujours.
    assert ds.hyper_tail_ge(150, 12, 30, 0) == Fraction(1)
    # Somme de la pmf = 1.
    tot = sum((ds.hyper_pmf_comb(20, 6, 8, k) for k in range(0, 7)),
              Fraction(0))
    assert tot == Fraction(1)


def test_k_star_hyper_monotone_et_aveugle():
    ks = ds.k_star_hyper(150, 12, 30)
    assert ks is not None and 1 <= ks <= 12
    assert ds.hyper_tail_ge(150, 12, 30, ks) < Fraction(1, 100)
    assert ds.hyper_tail_ge(150, 12, 30, ks - 1) >= Fraction(1, 100)
    # B trop petit => test AVEUGLE (None) : Q5, NON-MESURE gelé.
    assert ds.k_star_hyper(150, 0, 30) is None


def test_fisher_table_connue():
    # Table 2x2 : N=24, marges 8 et 9, recouvrement k ; Fisher unilatéral.
    p = ds.hyper_tail_ge(24, 8, 9, 7)
    q = ds.hyper_tail_ge_recurrence(24, 8, 9, 7)
    assert p == q and 0 < p < Fraction(1, 100)


# --- Famille close de 128 règles + départage ----------------------------------

def test_regles_closes_128_et_ordre_departage():
    regles = ds.regles_closes()
    assert len(regles) == 128
    assert len(set(map(ds.regle_id, regles))) == 128
    # (1) moins de features d'abord : les 16 premières sont simples.
    assert all(len(r) == 1 for r in regles[:16])
    assert all(len(r) == 2 for r in regles[16:])
    # (2)/(3) : ordre lexicographique et + avant -.
    assert ds.regle_id(regles[0]) == "+z1"
    assert ds.regle_id(regles[1]) == "-z1"
    assert ds.regle_id(regles[16]) == "+z1+z2"
    assert ds.regle_id(regles[17]) == "+z1-z2"
    assert ds.regle_id(regles[18]) == "-z1+z2"
    assert ds.regle_id(regles[19]) == "-z1-z2"


def test_z_robuste_et_score_formule_exacte():
    vals = {f: [1.0, 2.0, 3.0, 4.0, 100.0] for f in ds.FEATURES}
    csts = ds.constantes_z(vals)
    assert csts["f1"]["mediane"] == 3.0 and csts["f1"]["MAD"] == 1.0
    feats = {f"f{k}": float(k) for k in range(1, 9)}
    regle = ((1, 1), (3, -1))
    attendu = (feats["f1"] - 3.0) / 1.0 - (feats["f3"] - 3.0) / 1.0
    assert ds.score_regle(regle, feats, csts) == attendu
    # MAD = 0 => z := 0 (déclaré).
    csts0 = ds.constantes_z({f: [2.0, 2.0, 2.0] for f in ds.FEATURES})
    assert ds.z_feature(5.0, csts0["f1"]) == 0.0


def test_top_m_departage_deterministe():
    scores = {10: 1.0, 3: 2.0, 7: 2.0, 5: 0.5}
    assert ds.top_m_graines(scores, 3) == [3, 7, 10]


# --- Gate Q6 : le score ne voit que 5 époques ---------------------------------

def test_gate_q6_score_ne_voit_que_5_epoques():
    longues = {"perte_train": list(np.linspace(5, 0, 60)),
               "delta_med_val": list(np.linspace(0, 3, 60)),
               "r_disp": list(np.linspace(1, 2, 60)),
               "dbar": [0.1] * 60}
    tronquees = {k: v[:5] for k, v in longues.items()}
    assert ds.features_prefixe(longues) == ds.features_prefixe(tronquees)
    courtes = {k: v[:4] for k, v in longues.items()}
    assert ds.features_prefixe(courtes) is None


def test_features_prefixe_indices_exacts():
    c = {"perte_train": [10, 11, 12, 13, 14], "delta_med_val": [0, 1, 2, 3, 4],
         "r_disp": [5, 6, 7, 8, 9], "dbar": [0] * 5}
    f = ds.features_prefixe(c)
    assert f == {"f5": 14.0, "f6": 4.0, "f7": 9.0, "f8": 4.0}


# --- Features d'init : seeds fixés, formes, finitude, parité inter-bras -------

def test_features_init_seedees_et_finies():
    a = ds.features_init(0)
    b = ds.features_init(0)
    c = ds.features_init(1)
    assert a == b and a != c
    assert set(a) == {"f1", "f2", "f3", "f4"}
    assert all(math.isfinite(v) and v >= 0 for v in a.values())


def test_features_init_parite_avec_entrainement():
    """f1 = rho0 de la MÊME init que entrainer_bras (torch.manual_seed(g))."""
    seed = 3
    torch.manual_seed(seed)
    model = ChronoSpiraton(state_size=rt.STATE_SIZE, init_scale=dt.INIT_SCALE,
                           bounded=False, c_outside=False).double()
    rho = cd.rho_compagnon(cd.matrices_np(model))
    assert ds.features_init(seed)["f1"] == rho


# --- Règles déterministes : jumeau, ε₅₀ ---------------------------------------

def test_jumeau_verrouille_cyclique():
    pg = {str(s): {"classe": "DIVERGENT"} for s in range(50)}
    pg["2"] = {"classe": "VERROUILLE"}
    pg["48"] = {"classe": "VERROUILLE"}
    assert ds.jumeau_verrouille(pg, 10) == 48
    assert ds.jumeau_verrouille(pg, 48) == 2   # enroulement cyclique.
    del pg["30"]                                # graine EXCLU : pas de classe.
    assert ds.jumeau_verrouille(pg, 10) == 48


def test_eps50_ordinal():
    grid = ds.EPS_GRID
    assert ds.eps50_de_surv({e: 1.0 for e in grid}) == 1e-1
    assert ds.eps50_de_surv({e: 0.0 for e in grid}) == 0.0
    surv = {1e-4: 1.0, 1e-3: 0.5, 1e-2: 0.49, 3e-2: 0.6, 1e-1: 0.0}
    # « plus grand eps avec surv >= 0,50 » : 3e-2 (non monotone accepté).
    assert ds.eps50_de_surv(surv) == 3e-2


# --- Gate Q10 : aucune route vers une graine >= 50 hors train_neuf ------------

def test_gate_q10_aucune_route_graine_neuve_hors_train_neuf():
    src = Path(ds.__file__).read_text(encoding="utf-8")
    # Les seules consommations de GRAINES_NEUVES en dehors de train_neuf sont
    # la définition, la spec (bornes publiées) et la gate G3 (vérification
    # d'ABSENCE, lecture seule) — aucune n'entraîne ni ne charge un poids.
    occ = [m.start() for m in re.finditer(r"GRAINES_NEUVES", src)]
    assert len(occ) >= 2
    # Ancrage sur la VRAIE définition (début de ligne), pas sur un littéral.
    m_def = re.search(r"^def etape_train_neuf\(", src, flags=re.MULTILINE)
    assert m_def is not None
    idx_train = m_def.start()
    avant = src[:idx_train]
    # Aucun appel d'entraînement sur GRAINES_NEUVES avant train_neuf :
    for bloc in re.findall(r"entrainer_bras\([^)]*\)", avant):
        assert "GRAINES_NEUVES" not in bloc
    # train_neuf vérifie les DEUX jetons AVANT toute descente.
    m_fin = re.search(r"^def etape_classer\(", src, flags=re.MULTILINE)
    corps_train = src[idx_train:m_fin.start()]
    assert corps_train.index("verifier_gel_beta2") \
        < corps_train.index("entrainer_bras")


def test_liste_close_ordre_fige():
    assert ds.SPECIMENS == (("A1", 15), ("A2", 1), ("A2", 6), ("A2", 18),
                            ("A2", 25), ("WD", 1), ("WD", 3), ("WD", 14))
    assert ds.EPS_GRID == (1e-4, 1e-3, 1e-2, 3e-2, 1e-1)
    assert (ds.R_SPECIMEN, ds.R_CONTROLE) == (24, 16)
    assert ds.GRAINES_NEUVES[0] == 50 and ds.GRAINES_NEUVES[-1] == 199
    assert len(ds.GRAINES_NEUVES) == 150
    assert len(ds.FREEZE_PATHS_BETA2) == 24


# --- Binomiales exactes (Q8) ---------------------------------------------------

def test_binomiales_exactes():
    assert ds.binom_tail_ge(4, Fraction(1, 2), 0) == Fraction(1)
    assert ds.binom_tail_ge(4, Fraction(1, 2), 4) == Fraction(1, 16)
    assert ds.binom_tail_le(4, Fraction(1, 2), 4) == Fraction(1)
    s = ds.binom_tail_ge(150, Fraction(8, 100), 4) \
        + ds.binom_tail_le(150, Fraction(8, 100), 3)
    assert s == Fraction(1)
