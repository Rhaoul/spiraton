"""Quatuor canonique pour le diagnostic OPÉRATORIEL de la double dynamique.

Sœur de :mod:`tests.test_double_dynamics` (niveau grille), mais ici on teste le
réordonnancement des deux opérateurs temporels D et L de
:mod:`spiraton.diagnostics.operatorial_double_dynamics`.

Le quatuor exigé par le canon (CLAUDE.md « Le canon ») :
  1. finitude / pas de NaN-Inf sur les conditions non divergentes ;
  2. formes simple + batch ;
  3. formule exacte sous paramètres forcés (D=L=A=I, C=0 ⇒ step = identité,
     et LD ≡ DL exactement ; plus le théorème d'isospectralité AB~BA) ;
  4. déterminisme (seeds fixés ⇒ rapports bit-à-bit identiques).

On NE teste PAS le signe de l'asymétrie : c'est une mesure à constater, pas une
cible (REFUS). On teste que le diagnostic est correct, fini, reproductible, et
que sa baseline CTRL (D=L) annule l'asymétrie par construction.
"""

import torch

from spiraton.diagnostics.operatorial_double_dynamics import (
    OperatorialChrono,
    OperatorialConfig,
    isospectral_check,
    run_operatorial_double_dynamics,
)


# ---------------------------------------------------------------------------
# 1. Finitude (pas de NaN/Inf) sur les conditions non divergentes.
# ---------------------------------------------------------------------------

def test_finiteness_nonlinear() -> None:
    """NONLIN (tanh terminal) est borné : aucune série ne doit diverger."""
    torch.manual_seed(0)
    rep = run_operatorial_double_dynamics(
        d=8, steps=30, seeds=tuple(range(12)),
        cfg=OperatorialConfig(nonlinear=True), condition="NONLIN",
    )
    assert rep.n_diverged_LD == 0
    assert rep.n_diverged_DL == 0
    # Toutes les agrégations sont finies.
    for v in (rep.mean_var_LD, rep.mean_var_DL, rep.mean_variance_gap,
              rep.mean_best_return_LD, rep.mean_best_return_DL,
              rep.mean_final_norm_LD, rep.mean_final_norm_DL,
              rep.commutator_norm_DL):
        assert torch.isfinite(torch.tensor(v)).item()


def test_finiteness_trajectory_nonlinear() -> None:
    """La trajectoire NONLIN reste finie pas à pas (borne tanh)."""
    torch.manual_seed(1)
    cfg = OperatorialConfig(nonlinear=True)
    chrono = OperatorialChrono(8, 1, cfg)
    s0 = torch.randn(1, 8)
    for order in ("LD", "DL"):
        l2, cos, norm = chrono.trajectory(s0, order, steps=30)
        assert torch.isfinite(l2).all()
        assert torch.isfinite(cos).all()
        assert torch.isfinite(norm).all()
        assert l2.shape == cos.shape == norm.shape == (31,)


# ---------------------------------------------------------------------------
# 2. Formes simple + batch.
# ---------------------------------------------------------------------------

def test_shapes_simple_and_batch() -> None:
    """trajectory() accepte s0 de batch 1 (simple) et B>1 (batch)."""
    torch.manual_seed(2)
    cfg = OperatorialConfig(nonlinear=True)
    chrono = OperatorialChrono(6, 2, cfg)

    # Simple (batch=1).
    s_simple = torch.randn(1, 6)
    l2, cos, norm = chrono.trajectory(s_simple, "LD", steps=10)
    assert l2.shape == (11,)

    # Batch (B=4) : les métriques sont moyennées sur le batch, série de même T.
    s_batch = torch.randn(4, 6)
    l2b, cosb, normb = chrono.trajectory(s_batch, "DL", steps=10)
    assert l2b.shape == (11,)
    assert torch.isfinite(l2b).all() and torch.isfinite(cosb).all()


# ---------------------------------------------------------------------------
# 3. Formule exacte sous paramètres forcés.
# ---------------------------------------------------------------------------

def test_exact_identity_step() -> None:
    """D=L=A=I, C=0, linéaire ⇒ step = identité, et LD ≡ DL EXACTEMENT.

    C'est le cœur du diagnostic : quand les deux temps sont l'identité, l'ordre
    de composition n'a aucun effet (LD = L(D(inner)) = inner = D(L(inner)) = DL),
    et inner = A(s) = s. Vérifie que _step implémente bien L(D(inner)) vs
    D(L(inner)) sans terme additif parasite.
    """
    d = 5
    chrono = OperatorialChrono(d, 0, OperatorialConfig(nonlinear=False))
    chrono.A = torch.eye(d)
    chrono.C = torch.zeros(d, d)
    chrono.D = torch.eye(d)
    chrono.L = torch.eye(d)

    s = torch.randn(1, d)
    s_prev = torch.zeros(1, d)

    inner = chrono._inner(s, s_prev)
    assert torch.allclose(inner, s, atol=0.0, rtol=0.0)

    ld = chrono._step(s, s_prev, "LD")
    dl = chrono._step(s, s_prev, "DL")
    assert torch.allclose(ld, s, atol=0.0, rtol=0.0)
    assert torch.equal(ld, dl)  # ordre sans effet quand D=L=I


def test_exact_forced_matrices_composition() -> None:
    """Formule exacte sous matrices forcées arbitraires (linéaire).

    inner = s @ A.t() - s_prev @ C.t()
    LD    = (inner @ D.t()) @ L.t()
    DL    = (inner @ L.t()) @ D.t()
    On recompose à la main et on exige l'égalité bit-à-bit avec _step.
    """
    d = 4
    chrono = OperatorialChrono(d, 0, OperatorialConfig(nonlinear=False))
    A = torch.tensor([[1., 2., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
    C = 0.5 * torch.eye(d)
    D = torch.tensor([[0., 1., 0., 0.],
                      [1., 0., 0., 0.],
                      [0., 0., 2., 0.],
                      [0., 0., 0., 1.]])
    L = torch.tensor([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 1.],
                      [0., 0., 0., 1.]])
    chrono.A, chrono.C, chrono.D, chrono.L = A, C, D, L

    s = torch.randn(2, d)
    s_prev = torch.randn(2, d)

    inner_ref = s @ A.t() - s_prev @ C.t()
    ld_ref = (inner_ref @ D.t()) @ L.t()
    dl_ref = (inner_ref @ L.t()) @ D.t()

    assert torch.allclose(chrono._inner(s, s_prev), inner_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(chrono._step(s, s_prev, "LD"), ld_ref, atol=0.0, rtol=0.0)
    assert torch.allclose(chrono._step(s, s_prev, "DL"), dl_ref, atol=0.0, rtol=0.0)
    # D et L ne commutent pas ici ⇒ LD != DL (l'ordre porte une information).
    assert not torch.allclose(ld_ref, dl_ref, atol=1e-6)


def test_operating_companion_not_isospectral_but_sign_unstable() -> None:
    """Réserve maths Tour 1, rendue testable.

    L'isospectralité AB~BA porte sur D@L / L@D SEULES, pas sur la dynamique
    déroulée. La matrice compagnon opérante M_X=[[XA,-XC],[I,0]] (X=L@D ou D@L)
    n'est PAS isospectrale entre LD et DL (gap de ρ >> bruit float). MAIS le
    signe de l'écart est aléatoire entre graines : il ne confirme donc pas la
    direction de l'axiome 4. On exige (i) un gap non négligeable au moins une
    fois, (ii) un signe non systématique (ni 0/12 ni 12/12).
    """
    d = 8

    def companion(ch: OperatorialChrono, order: str) -> torch.Tensor:
        X = ch.L @ ch.D if order == "LD" else ch.D @ ch.L
        top = torch.cat([X @ ch.A, -(X @ ch.C)], dim=1)
        bot = torch.cat([torch.eye(d), torch.zeros(d, d)], dim=1)
        return torch.cat([top, bot], dim=0)

    ld_wins = 0
    max_gap = 0.0
    for seed in range(12):
        ch = OperatorialChrono(d, seed, OperatorialConfig(nonlinear=False))
        r_ld = torch.linalg.eigvals(companion(ch, "LD")).abs().max().item()
        r_dl = torch.linalg.eigvals(companion(ch, "DL")).abs().max().item()
        max_gap = max(max_gap, abs(r_ld - r_dl))
        if r_ld > r_dl:
            ld_wins += 1

    assert max_gap > 1e-2          # asymétrie spectrale opérante RÉELLE
    assert 0 < ld_wins < 12        # mais signe NON systématique ⇒ bruit


def test_isospectral_theorem() -> None:
    """Théorème : D@L et L@D ont les mêmes valeurs propres (AB ~ BA).

    Le contrôle isospectral du diagnostic. À float64 le gap est l'epsilon
    machine ; on l'exige donc strictement minuscule. C'est la preuve qu'aucune
    asymétrie de STABILITÉ ne peut venir de l'ordre de composition LINÉAIRE des
    matrices D, L prises seules.
    """
    torch.manual_seed(7)
    d = 8
    D = torch.randn(d, d, dtype=torch.float64)
    L = torch.randn(d, d, dtype=torch.float64)
    chk = isospectral_check(D, L)
    assert chk["spectral_radius_gap"] < 1e-9
    assert chk["max_sorted_eigval_gap"] < 1e-9


# ---------------------------------------------------------------------------
# 4. Déterminisme (seeds fixés ⇒ résultats identiques).
# ---------------------------------------------------------------------------

def test_deterministic_report() -> None:
    """Deux exécutions à seeds fixés ⇒ rapport bit-à-bit identique."""
    cfg = OperatorialConfig(nonlinear=True)
    seeds = tuple(range(12))
    r1 = run_operatorial_double_dynamics(d=8, steps=30, seeds=seeds, cfg=cfg, condition="X")
    r2 = run_operatorial_double_dynamics(d=8, steps=30, seeds=seeds, cfg=cfg, condition="X")
    assert r1.mean_variance_gap == r2.mean_variance_gap
    assert r1.seeds_gap_positive == r2.seeds_gap_positive
    assert r1.best_return_gap == r2.best_return_gap
    assert r1.seeds_return_LD_wins == r2.seeds_return_LD_wins
    assert r1.commutator_norm_DL == r2.commutator_norm_DL


def test_deterministic_trajectory() -> None:
    torch.manual_seed(0)
    cfg = OperatorialConfig(nonlinear=True)
    c1 = OperatorialChrono(8, 5, cfg)
    c2 = OperatorialChrono(8, 5, cfg)
    # Mêmes poids (même seed de construction).
    assert torch.equal(c1.D, c2.D) and torch.equal(c1.L, c2.L)
    s0 = torch.randn(1, 8, generator=torch.Generator().manual_seed(99))
    l2a, _, _ = c1.trajectory(s0, "LD", steps=20)
    l2b, _, _ = c2.trajectory(s0, "LD", steps=20)
    assert torch.equal(l2a, l2b)


# ---------------------------------------------------------------------------
# Baseline CTRL obligatoire (REFUS) : D=L ⇒ asymétrie strictement nulle.
# ---------------------------------------------------------------------------

def test_ctrl_baseline_zero_asymmetry() -> None:
    """CTRL (D=L) : LD ≡ DL par construction ⇒ variance_gap = 0 EXACT, 0/12.

    Preuve que le diagnostic ne fabrique pas d'asymétrie artefactuelle :
    tout effet observé en D≠L vient bien de D≠L et non du déroulé.
    """
    rep = run_operatorial_double_dynamics(
        d=8, steps=30, seeds=tuple(range(12)),
        cfg=OperatorialConfig(nonlinear=False, ctrl_same_DL=True), condition="CTRL",
    )
    assert rep.commutator_norm_DL == 0.0
    assert rep.seeds_gap_positive == 0
    assert rep.seeds_return_LD_wins == 0
    assert rep.mean_variance_gap == 0.0
    assert rep.best_return_gap == 0.0
