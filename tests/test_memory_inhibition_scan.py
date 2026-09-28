"""Tests du balayage d'inhibition mémoire ``−C(s_{t−1})`` (Tour 2, axiome §3.2).

Quatuor canon (seeds, finitude, formes simples/batch, gradient/déterminisme) +
test anti-artefact CTRL-0 / CTRL-RAND + cross-vérification de la statistique de
rang (Spearman maison vs scipy).

On NE TOUCHE PAS au canon : ce diagnostic réassigne ``model.C.weight`` sous
no_grad dans le scan ; on vérifie ici que le défaut de ChronoSpiraton reste exact
après usage.
"""

import math

import pytest
import torch

from spiraton.experimental.chrono import ChronoSpiraton
from spiraton.diagnostics.memory_inhibition_scan import (
    _alpha_omega_on_trace,
    _make_C,
    run_memory_inhibition_scan,
    spearman_rho,
    spearman_t_pvalue,
    spectral_radius,
)


def _no_nan_inf(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# --- 1. SEEDS / déterminisme ------------------------------------------------

def test_scan_is_deterministic_across_runs() -> None:
    """Même appel ⇒ chiffres identiques (aucun hasard non seedé)."""
    kw = dict(d=6, steps=20, seeds=tuple(range(6)),
              gamma_ratios=(0.0, 0.5, 2.0), bounded=False)
    r1 = run_memory_inhibition_scan(**kw)
    r2 = run_memory_inhibition_scan(**kw)

    def _eq(a: float, b: float) -> bool:
        return a == b or (math.isnan(a) and math.isnan(b))

    assert _eq(r1.spearman_gamma_diverged, r2.spearman_gamma_diverged)
    assert _eq(r1.diverged_drop_pts, r2.diverged_drop_pts)
    # comparaison NaN-safe sur les médianes (NaN possible si tous invalides).
    def _key(c):
        mn = c.median_max_norm if c.median_max_norm == c.median_max_norm else "nan"
        return (c.condition, c.gamma_ratio, c.diverged_rate, mn)

    c1 = sorted(_key(c) for c in r1.cells)
    c2 = sorted(_key(c) for c in r2.cells)
    assert c1 == c2


def test_make_C_is_seed_reproducible_and_unit_norm() -> None:
    for st in ("dense", "diag+", "diag±", "anti-sym"):
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        C1 = _make_C(8, st, g1)
        C2 = _make_C(8, st, g2)
        assert torch.allclose(C1, C2)
        # norme Frobenius unitaire (anti-artefact CTRL-RAND : même norme).
        assert abs(float(torch.linalg.norm(C1).item()) - 1.0) < 1e-5


# --- 2. FINITUDE ------------------------------------------------------------

def test_report_fields_finite_or_documented_nan() -> None:
    rep = run_memory_inhibition_scan(d=6, steps=20, seeds=tuple(range(6)),
                                     gamma_ratios=(0.0, 0.5, 1.0, 2.0))
    for c in rep.cells:
        assert math.isfinite(c.diverged_rate)
        assert 0.0 <= c.diverged_rate <= 1.0
        # médianes : finies, ou NaN explicite (jamais inf silencieux).
        for v in (c.median_max_norm, c.median_final_norm, c.median_best_return):
            assert math.isfinite(v) or math.isnan(v)
    # rho de Spearman dans [-1, 1] ou NaN.
    rho = rep.spearman_gamma_diverged
    assert math.isnan(rho) or (-1.0 <= rho <= 1.0)


def test_alpha_omega_excludes_trivial_fixed_point() -> None:
    """Un état effondré vers 0 ne doit pas compter comme 'retour' valide."""
    s0 = torch.ones(1, 4)
    # trace qui s'effondre : normes très inférieures au plancher.
    trace = [torch.ones(1, 4) * 1e-6 for _ in range(5)]
    out = _alpha_omega_on_trace(s0, trace, norm_floor_ratio=1e-2)
    assert out["valid"] is False
    assert math.isnan(out["best_return"])

    # trace non triviale : au moins un pas valide.
    trace2 = [torch.ones(1, 4) * 0.9 for _ in range(5)]
    out2 = _alpha_omega_on_trace(s0, trace2, norm_floor_ratio=1e-2)
    assert out2["valid"] is True
    assert math.isfinite(out2["best_return"])


# --- 3. FORMES simple / batch ----------------------------------------------

def test_alpha_omega_accepts_1d_and_2d() -> None:
    s0_1d = torch.randn(5)
    trace_1d = [torch.randn(5) for _ in range(3)]
    out1 = _alpha_omega_on_trace(s0_1d, trace_1d)
    assert set(out1.keys()) == {"best_return", "best_step", "valid", "diverged"}

    s0_2d = torch.randn(1, 5)
    trace_2d = [torch.randn(1, 5) for _ in range(3)]
    out2 = _alpha_omega_on_trace(s0_2d, trace_2d)
    assert set(out2.keys()) == {"best_return", "best_step", "valid", "diverged"}


def test_scan_runs_small_dims() -> None:
    rep = run_memory_inhibition_scan(d=4, steps=10, seeds=(0, 1, 2),
                                     gamma_ratios=(0.0, 1.0),
                                     structures=("dense", "diag+"))
    # CTRL-0, dense, diag+, CTRL-RAND, +C(dense) doivent apparaître.
    conds = {c.condition for c in rep.cells}
    assert "CTRL-0" in conds
    assert "CTRL-RAND" in conds
    assert "dense" in conds


# --- 4. GRADIENT / non-cassage du canon ------------------------------------

def test_canon_default_chrono_unchanged_after_scan() -> None:
    """Le scan ne doit pas altérer le comportement par défaut de ChronoSpiraton.

    On capture la sortie d'une cellule à graine fixée AVANT, on lance le scan
    (qui réassigne des C.weight sur SES PROPRES instances), et on revérifie
    qu'une cellule fraîche à même graine produit la même sortie : preuve que le
    canon n'est pas touché (aucun état global modifié)."""
    torch.manual_seed(42)
    ref = ChronoSpiraton(state_size=5, init_scale=0.1)
    s0 = torch.randn(2, 5) * 0.1
    before = ref(s0, steps=4).clone()

    _ = run_memory_inhibition_scan(d=5, steps=10, seeds=(0, 1),
                                   gamma_ratios=(0.0, 1.0))

    torch.manual_seed(42)
    ref2 = ChronoSpiraton(state_size=5, init_scale=0.1)
    after = ref2(s0, steps=4)
    assert torch.allclose(before, after, atol=0.0)


def test_chrono_gradients_still_flow_with_reassigned_C() -> None:
    """Réassigner C.weight sous no_grad ne casse pas le flux de gradient ensuite."""
    torch.manual_seed(3)
    d = 6
    model = ChronoSpiraton(state_size=d, init_scale=0.1)
    with torch.no_grad():
        model.C.weight.copy_(torch.eye(d) * 0.3)
    s0 = torch.randn(4, d, requires_grad=True) * 0.1
    model(s0, steps=4).sum().backward()
    for name in ("A", "B", "C", "D", "L"):
        g = getattr(model, name).weight.grad
        assert g is not None, f"pas de gradient sur {name}"


# --- ANTI-ARTEFACT : CTRL-0 et CTRL-RAND -----------------------------------

def test_ctrl0_equals_gamma_zero_for_every_structure() -> None:
    """CTRL-0 (C=0) doit coïncider avec le point γ=0 de chaque structure :
    à magnitude nulle, toute structure est la matrice nulle ⇒ même dynamique."""
    rep = run_memory_inhibition_scan(d=6, steps=20, seeds=tuple(range(6)),
                                     gamma_ratios=(0.0, 1.0),
                                     structures=("dense", "diag+", "anti-sym"))
    ctrl0 = next(c for c in rep.cells if c.condition == "CTRL-0" and c.gamma_ratio == 0.0)
    for st in ("dense", "diag+", "anti-sym"):
        cell = next(c for c in rep.cells if c.condition == st and c.gamma_ratio == 0.0)
        assert cell.diverged_rate == ctrl0.diverged_rate
        assert cell.median_max_norm == ctrl0.median_max_norm or (
            math.isnan(cell.median_max_norm) and math.isnan(ctrl0.median_max_norm))


def test_ctrlrand_same_norm_as_structures() -> None:
    """CTRL-RAND doit être de MÊME norme Frobenius que les structures testées
    (anti-artefact : isole l'inhibition structurée de l'atténuation scalaire)."""
    d, gamma = 8, 2.0
    torch.manual_seed(0)
    model = ChronoSpiraton(state_size=d, init_scale=0.5)
    rho_A = spectral_radius(model.A.weight.detach())
    scale = gamma * rho_A

    g_struct = torch.Generator().manual_seed(0 + 200_000)
    g_rand = torch.Generator().manual_seed(0 + 300_000)
    C_dense = scale * _make_C(d, "dense", g_struct)
    C_rand = scale * _make_C(d, "rand", g_rand)
    n_dense = float(torch.linalg.norm(C_dense).item())
    n_rand = float(torch.linalg.norm(C_rand).item())
    assert abs(n_dense - n_rand) < 1e-4 * max(n_dense, 1.0)
    # mêmes normes, matrices DIFFÉRENTES (sinon le contrôle est vide).
    assert not torch.allclose(C_dense, C_rand)


# --- STATISTIQUE DE RANG : cross-vérification contre scipy ------------------

def test_spearman_matches_scipy() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    cases = [
        ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]),   # -1
        ([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),             # +1
        ([0.0, 0.25, 0.5, 1.0, 2.0, 4.0], [0.8, 0.6, 0.4, 0.3, 0.1, 0.0]),
        ([0.0, 0.25, 0.5, 1.0, 2.0, 4.0], [0.5, 0.5, 0.3, 0.3, 0.1, 0.0]),  # ex-aequo
    ]
    for x, y in cases:
        mine = spearman_rho(x, y)
        ref = float(scipy_stats.spearmanr(x, y).statistic)
        assert abs(mine - ref) < 1e-9, (x, y, mine, ref)


def test_spearman_pvalue_matches_scipy() -> None:
    scipy_stats = pytest.importorskip("scipy.stats")
    x = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    y = [0.9, 0.7, 0.55, 0.3, 0.2, 0.05]
    rho = spearman_rho(x, y)
    res = scipy_stats.spearmanr(x, y)
    p_mine = spearman_t_pvalue(rho, len(x))
    # scipy utilise la même approximation t pour n petit ⇒ accord serré.
    assert abs(p_mine - float(res.pvalue)) < 1e-6


def test_spectral_radius_matches_eig() -> None:
    torch.manual_seed(11)
    W = torch.randn(8, 8)
    rho = spectral_radius(W)
    ref = float(torch.linalg.eigvals(W).abs().max().item())
    assert abs(rho - ref) < 1e-5
    assert rho > 0


# --- TOUR 3 : POSITION de l'inhibition (c_outside) -------------------------

def _cell(rep, cond, g):
    return next(c for c in rep.cells if c.condition == cond and c.gamma_ratio == g)


def test_ctrl0_identical_inside_outside() -> None:
    """CONTRÔLE DUR : à C=0 (γ=0), intérieur et extérieur DOIVENT coïncider
    bit-à-bit (final_norm, max_norm, diverged, retour α-ω, validité).

    À C=0, ``D(A+B−0)+L`` == ``D(A+B)+L−0`` : la position de C n'a aucun effet.
    Si ces deux scans diffèrent au point γ=0, le diagnostic a un bug — ce test
    le ferait échouer. C'est le verrou d'attribution : toute différence à γ>0
    est alors imputable à la POSITION, pas à un artefact d'implémentation.
    """
    kw = dict(d=8, steps=30, seeds=tuple(range(8)),
              gamma_ratios=(0.0, 0.5, 2.0),
              structures=("dense", "diag+", "diag±", "anti-sym"),
              bounded=False, init_scale=1.2)
    r_in = run_memory_inhibition_scan(c_outside=False, **kw)
    r_out = run_memory_inhibition_scan(c_outside=True, **kw)
    assert r_in.c_outside is False and r_out.c_outside is True

    ci = _cell(r_in, "CTRL-0", 0.0)
    co = _cell(r_out, "CTRL-0", 0.0)
    assert ci.diverged_rate == co.diverged_rate
    assert ci.valid_return_rate == co.valid_return_rate
    for a, b in ((ci.median_max_norm, co.median_max_norm),
                 (ci.median_final_norm, co.median_final_norm),
                 (ci.median_best_return, co.median_best_return)):
        assert a == b or (math.isnan(a) and math.isnan(b))

    # Et chaque structure à γ=0 partage ce même point dans les DEUX variantes.
    for st in ("dense", "diag+", "diag±", "anti-sym"):
        for rep, ref in ((r_in, ci), (r_out, co)):
            s0 = _cell(rep, st, 0.0)
            assert s0.diverged_rate == ref.diverged_rate


def test_default_scan_is_c_inside() -> None:
    """Le défaut du scan (c_outside non passé) est la position INTÉRIEURE canon."""
    rep = run_memory_inhibition_scan(d=4, steps=10, seeds=(0, 1, 2),
                                     gamma_ratios=(0.0, 1.0),
                                     structures=("dense", "diag+"))
    assert rep.c_outside is False


def test_nonregression_inside_rho_reproduces_tour2() -> None:
    """NON-RÉGRESSION : ``−C`` intérieur (c_outside=False), à la config EXACTE du
    Tour 2 (24 graines, steps=50, init_scale=1.2, dense), reproduit ρ_s≈+0.820.

    Verrou que le refactor de ``step()`` (extraction de ``c_outside``) n'a pas
    dérivé la mesure de référence. Tolérance serrée ±0.01.
    """
    rep = run_memory_inhibition_scan(
        d=8, steps=50, seeds=tuple(range(24)),
        gamma_ratios=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
        structures=("dense", "diag+", "diag±", "anti-sym"),
        bounded=False, c_outside=False, init_scale=1.2,
    )
    assert rep.ref_structure == "dense"
    assert abs(rep.spearman_gamma_diverged - 0.820) < 0.01


def test_outside_scan_is_deterministic() -> None:
    """Déterminisme bit-à-bit de la variante extérieure (2 runs identiques)."""
    kw = dict(d=6, steps=20, seeds=tuple(range(6)),
              gamma_ratios=(0.0, 0.5, 2.0), bounded=False, c_outside=True)
    r1 = run_memory_inhibition_scan(**kw)
    r2 = run_memory_inhibition_scan(**kw)

    def _eq(a, b):
        return a == b or (math.isnan(a) and math.isnan(b))

    assert _eq(r1.spearman_gamma_diverged, r2.spearman_gamma_diverged)
    assert _eq(r1.spearman_pvalue, r2.spearman_pvalue)
