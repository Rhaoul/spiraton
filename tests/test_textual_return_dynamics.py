# -*- coding: utf-8 -*-
"""Quatuor de tests du lecteur T63 (textual_return_dynamics) — CLAUDE.md :

1. seeds fixés (aucune sortie aléatoire non seedée) ;
2. finitude (pas de NaN/Inf) ;
3. formes simple/batch ;
4. formule exacte sous paramètres forcés — ``Φ`` à ``A=B=C=D=0, L=I`` est
   l'identité AU BIT (⇒ ``s_dyn == s_raw``), plus UN pas de ChronoSpiraton en
   dimension 2 calculé à la main ;
5. flux de gradient (le module reste différentiable, même si le tour
   n'entraîne rien).

Plus : ``Φ_0 = id`` (gate de parité M1), Spearman/rangs moyens exacts à la
main, Fisher-z stratifié, et le miroir ``import_alpha_omega_text`` (skip
propre si le dépôt tokenizer n'est pas à côté).
"""
import math

import numpy as np
import pytest
import torch

from spiraton.data.tokenizer_bridge import TokenizerUnavailable, import_alpha_omega_text
from spiraton.diagnostics.textual_return_dynamics import (
    EPS,
    INIT_SCALE,
    K_PORTEUR,
    SEEDS,
    STATE_SIZE,
    deplacements_canon_rang1,
    make_chrono,
    phi_k,
    phi_k_np,
    rangs_moyens,
    rho_stratifie,
    sigma_ratio,
    spearman,
)


# --- (1) seeds fixés ---------------------------------------------------------

def test_make_chrono_est_seede_et_reproductible() -> None:
    """Même graine ⇒ poids bit-à-bit identiques ; graines ≠ ⇒ poids ≠."""
    c1 = make_chrono(0)
    c2 = make_chrono(0)
    c3 = make_chrono(1)
    for name in ("A", "B", "C", "D", "L"):
        w1 = getattr(c1, name).weight
        w2 = getattr(c2, name).weight
        w3 = getattr(c3, name).weight
        assert w1.dtype == torch.float64  # promotion float64 (agrégats M1)
        assert torch.equal(w1, w2), f"graine 0 non reproductible sur {name}"
        assert not torch.equal(w1, w3), f"graines 0/1 identiques sur {name}"


# --- (2) finitude ------------------------------------------------------------

def test_finitude_sur_les_K_geles() -> None:
    torch.manual_seed(100)
    v = torch.randn(7, STATE_SIZE, dtype=torch.float64)
    for seed in SEEDS[:3]:
        chrono = make_chrono(seed)
        for k in (0, 1, 2, 4, 8):
            out = phi_k(chrono, v, k)
            assert torch.isfinite(out).all(), f"NaN/Inf à seed={seed}, K={k}"


# --- (3) formes simple/batch -------------------------------------------------

def test_formes_simple_et_batch_coincident() -> None:
    torch.manual_seed(101)
    chrono = make_chrono(2)
    batch = torch.randn(4, STATE_SIZE, dtype=torch.float64)
    out_batch = phi_k(chrono, batch, K_PORTEUR)
    assert out_batch.shape == (4, STATE_SIZE)
    for i in range(4):
        out_i = phi_k(chrono, batch[i], K_PORTEUR)
        assert out_i.shape == (STATE_SIZE,)
        # Égalité NUMÉRIQUE serrée : les chemins BLAS (1,d) et (B,d) peuvent
        # sommer dans un ordre différent — le bit-à-bit vit au test de formule
        # forcée, pas ici.
        assert torch.allclose(out_i, out_batch[i], rtol=1e-12, atol=1e-12), (
            f"simple != batch (ligne {i})")


# --- (4a) Φ_0 = identité (parité M1 au bit) ----------------------------------

def test_phi_0_est_l_identite_bit_a_bit() -> None:
    """K=0 rend le vecteur TEL QUEL (même objet) : parité M1 exacte à K=0."""
    chrono = make_chrono(3)
    v = torch.randn(5, STATE_SIZE, dtype=torch.float64)
    assert phi_k(chrono, v, 0) is v
    mat = np.random.default_rng(0).standard_normal((5, STATE_SIZE))
    assert phi_k_np(chrono, mat, 0) is mat


def test_phi_steps_negatif_refuse() -> None:
    chrono = make_chrono(3)
    with pytest.raises(ValueError):
        phi_k(chrono, torch.zeros(STATE_SIZE, dtype=torch.float64), -1)


# --- (4b) formule exacte : A=B=C=D=0, L=I ⇒ Φ_K = id AU BIT ------------------

def test_identite_forcee_donne_s_dyn_egal_s_raw_au_bit() -> None:
    """``s_{t+1} = D(0) + L(s) = s`` exactement quand A=B=C=D=0 et L=I."""
    chrono = make_chrono(4)
    with torch.no_grad():
        for name in ("A", "B", "C", "D"):
            getattr(chrono, name).weight.zero_()
        chrono.L.weight.copy_(torch.eye(STATE_SIZE, dtype=torch.float64))
    v = torch.randn(6, STATE_SIZE, dtype=torch.float64) * 3.0
    for k in (1, 2, 4, 8):
        out = phi_k(chrono, v, k)
        assert torch.equal(out, v), f"Φ_{k} != id au bit sous paramètres forcés"

    # ⇒ s_dyn == s_raw exactement (le même score sur les mêmes bits).
    try:
        m1 = import_alpha_omega_text()
    except TokenizerUnavailable:
        pytest.skip("dépôt tokenizer absent : parité s_dyn==s_raw non vérifiable ici")
    a = v[0].numpy()
    ap = v[1].numpy()
    fa = phi_k(chrono, v[0], K_PORTEUR).detach().numpy()
    fap = phi_k(chrono, v[1], K_PORTEUR).detach().numpy()
    assert m1.score_retour(fa, fap) == m1.score_retour(a, ap)


def test_un_pas_chrono_2d_calcule_a_la_main() -> None:
    """Un pas (et deux pas) de l'équation §3.2 en d=2, valeurs à la main.

    A=I, B=swap, C=0.5·I, D=diag(2,3), L=[[1,1],[0,1]] ;
    s0=[1,2], s_prev=[0.5,−1] :
      s1 = D(A s0 + B s0² − C s_prev) + L s0
         = D([1,2] + [4,1] − [0.25,−0.5]) + [3,2]
         = D([4.75, 3.5]) + [3,2] = [9.5,10.5] + [3,2] = [12.5, 12.5]
      s2 = D([12.5,12.5] + [156.25,156.25] − C[1,2]) + L s1
         = D([168.25, 167.75]) + [25,12.5] = [336.5+25, 503.25+12.5]
         = [361.5, 515.75]        (toutes valeurs dyadiques ⇒ exactes)
    """
    chrono = make_chrono(5, state_size=2)
    with torch.no_grad():
        chrono.A.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
        chrono.B.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64))
        chrono.C.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.5]], dtype=torch.float64))
        chrono.D.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64))
        chrono.L.weight.copy_(torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64))
    s0 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    s_prev = torch.tensor([0.5, -1.0], dtype=torch.float64)

    s1 = chrono.step(s0, s_prev)
    assert torch.equal(s1, torch.tensor([12.5, 12.5], dtype=torch.float64))

    s2 = chrono(s0, steps=2, s_prev=s_prev)
    assert torch.equal(s2, torch.tensor([361.5, 515.75], dtype=torch.float64))


# --- (5) flux de gradient ----------------------------------------------------

def test_flux_de_gradient_a_travers_phi() -> None:
    chrono = make_chrono(6)
    v = torch.randn(3, STATE_SIZE, dtype=torch.float64, requires_grad=True)
    phi_k(chrono, v, K_PORTEUR).sum().backward()
    assert v.grad is not None and float(v.grad.abs().sum()) > 0.0
    for name in ("A", "B", "C", "D", "L"):
        g = getattr(chrono, name).weight.grad
        assert g is not None, f"pas de gradient sur {name}"


# --- Statistique : rangs, Spearman, Fisher-z ---------------------------------

def test_rangs_moyens_avec_ex_aequo() -> None:
    got = rangs_moyens([1.0, 2.0, 2.0, 3.0])
    assert np.allclose(got, [1.0, 2.5, 2.5, 4.0])


def test_spearman_exact_a_la_main() -> None:
    # Monotone parfait (avec ex-aequo appariés) → 1 ; renversé → −1.
    assert spearman([1, 2, 2, 3], [10, 20, 20, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # Cas calculé à la main : rangs = valeurs, Pearson = 1.0/1.25 = 0.8.
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    # Variance nulle ⇒ NaN déclaré (jamais 0 silencieux).
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_rho_stratifie_a_la_main() -> None:
    # Un seul strate : ρ_S = ρ. Deux strates égales : ρ_S = ρ commun.
    assert rho_stratifie([(0.5, 47)]) == pytest.approx(0.5)
    assert rho_stratifie([(0.5, 47), (0.5, 27)]) == pytest.approx(0.5)
    # Pondération (n−3) : z = (44·atanh(.8) + 21·atanh(.2)) / 65.
    z = (44 * math.atanh(0.8) + 21 * math.atanh(0.2)) / 65.0
    assert rho_stratifie([(0.8, 47), (0.2, 24)]) == pytest.approx(math.tanh(z))
    # Un NaN se propage (jamais silencieusement 0).
    assert math.isnan(rho_stratifie([(float("nan"), 47), (0.5, 27)]))


# --- Contrôle exact INFO-2 : rang 1 du canon ---------------------------------

def test_deplacements_canon_sont_de_rang_1_exact() -> None:
    """SpiratonCell→Linear(1,d) ⇒ tous les Δ colinéaires : σ₂/σ₁ ≤ 1e-12.

    Substrat EXACT (100 % ou faux) : un échec ici est un bug d'instrument.
    """
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((50, STATE_SIZE))
    deltas = deplacements_canon_rang1(vecs, seed=0)
    assert deltas.shape == (50, STATE_SIZE)
    assert sigma_ratio(deltas) <= 1e-12


def test_sigma_ratio_sur_matrice_de_rang_2() -> None:
    # Deux directions orthogonales d'amplitudes 3 et 1 ⇒ σ₂/σ₁ = 1/3.
    m = np.zeros((4, STATE_SIZE))
    m[0, 0] = 3.0
    m[1, 0] = -3.0
    m[2, 1] = 1.0
    m[3, 1] = -1.0
    assert sigma_ratio(m) == pytest.approx(1.0 / 3.0)


# --- Le miroir du pont -------------------------------------------------------

def test_import_alpha_omega_text_miroir() -> None:
    """Le miroir rend le module M1 gelé : MAIN_DIMS (19), score_retour, EPS."""
    try:
        m1 = import_alpha_omega_text()
    except TokenizerUnavailable:
        pytest.skip("dépôt tokenizer absent : miroir non testable ici")
    assert tuple(m1.MAIN_DIMS) == tuple(range(6, 23)) + (31, 32)
    assert m1.EPS == EPS
    # s(X,X) ≈ 1 (cos 1, l2 0) — le score du retour, même objet que M1.
    v = np.arange(1.0, 20.0)
    assert m1.score_retour(v, v) == pytest.approx(1.0, abs=1e-9)


def test_constantes_gelees_du_tour() -> None:
    """Les constantes β du T63 sont bien celles de l'émission (liste close)."""
    assert STATE_SIZE == 19
    assert INIT_SCALE == pytest.approx(0.1)
    assert SEEDS == tuple(range(12))
    assert K_PORTEUR == 2
