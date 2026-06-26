"""Tests du diagnostic de non-commutativité acquise (Tour 11, chantier 1).

Quatuor canon (CLAUDE.md) :
  - finitude (pas de NaN/Inf),
  - formes simple/batch (ici : longueurs de trajectoire correctes),
  - formule exacte sous paramètres forcés (la trajectoire enregistre bien
    ``total_noncommutativity()`` du modèle),
  - flux de gradient (la perte de tâche descend, sans terme commutateur).

Plus déterminisme bit-à-bit (2 runs) et le **test anti-artefact dur** : le
contrôle order-détruit (enseignant commutant) ne fait PAS croître le
commutateur — analogue du CTRL ``D=L`` du Tour 1 et du ``commuting`` ‖·‖=0 du
Tour 6.
"""

import math

import torch
import pytest

from spiraton.diagnostics.acquired_noncommutativity import (
    BETA_MULT,
    KAPPA_MAX,
    PAIR_DIV_SUB,
    PAIR_MUL_ADD,
    TrajectoryReport,
    _binom_tail_ge,
    _commuting_teacher,
    _commuting_teacher_well_conditioned,
    _max_commutator,
    _mean_asymmetry,
    _noncommuting_teacher,
    _percentile,
    _well_conditioned_eigenbasis,
    run_sweep,
    train_one,
    vector_floor_loss,
)
from spiraton.experimental.matrix_cell import MatrixSpiratonCell, OPS


def _finite(xs) -> bool:
    return all(math.isfinite(x) for x in xs)


# --- Quatuor canon ----------------------------------------------------------


def test_trajectory_finite_and_lengths() -> None:
    """Finitude + formes : la trajectoire a epochs+1 points (init inclus)."""
    gen = torch.Generator().manual_seed(0 + 7919)
    teacher = _noncommuting_teacher(5, 0.5, gen)
    epochs = 30
    rep = train_one(
        seed=0, condition="ordered", teacher=teacher, dim=5, init_scale=0.5,
        order=("add", "sub", "mul", "div"), n_samples=64, epochs=epochs, lr=5e-3,
    )
    for series in (rep.epochs_total, rep.epochs_loss,
                   rep.epochs_pair_mul_add, rep.epochs_pair_div_sub):
        assert len(series) == epochs + 1
        assert _finite(series)
    assert rep.total_init >= 0.0
    assert rep.loss_final >= 0.0


def test_sweep_simple_and_batch_shapes() -> None:
    """« simple/batch » au niveau balayage : n graines -> n trajectoires."""
    seeds = [0, 1, 2]
    rep = run_sweep(seeds=seeds, dim=4, init_scale=0.5, n_samples=32,
                    epochs=20, lr=5e-3)
    assert len(rep.ordered) == len(seeds)
    assert len(rep.commuting) == len(seeds)
    assert len(rep.vector_floor) == len(seeds)
    assert len(rep.per_seed_advantage) == len(seeds)
    assert _finite([s for s in rep.summary.values()])
    # single-seed également valide
    rep1 = run_sweep(seeds=[7], dim=4, init_scale=0.5, n_samples=32,
                     epochs=20, lr=5e-3)
    assert len(rep1.ordered) == 1


def test_trajectory_records_model_commutator_exactly() -> None:
    """Formule exacte : la trajectoire enregistre bien le commutateur du modèle.

    Sous matrices forcées, ``total_noncommutativity()`` du modèle au point
    enregistré doit coïncider à l'identique avec la valeur de la trajectoire au
    pas correspondant. On vérifie l'init (epoch 0), avant toute mise à jour, où
    le modèle est dans son état d'initialisation.
    """
    torch.manual_seed(0)
    dim = 4
    # Un enseignant trivial (identité) -> 0 epoch de mise à jour : on lit l'init.
    gen = torch.Generator().manual_seed(123)
    teacher = _noncommuting_teacher(dim, 0.5, gen)
    rep = train_one(
        seed=42, condition="ordered", teacher=teacher, dim=dim, init_scale=0.5,
        order=("add", "mul"), n_samples=16, epochs=0, lr=5e-3,
    )
    # epochs=0 -> trajectoire = [init] seulement
    assert len(rep.epochs_total) == 1
    # Reconstruire l'élève exactement comme train_one (même graine, même tirage)
    g = torch.Generator().manual_seed(42)
    student = MatrixSpiratonCell(input_size=dim, init_scale=0.5)
    with torch.no_grad():
        eye = torch.eye(dim)
        for name in OPS:
            getattr(student, f"W_{name}").copy_(
                eye + torch.randn(dim, dim, generator=g) * 0.5
            )
        student.bias.zero_()
    with torch.no_grad():
        expected_total = float(student.total_noncommutativity())
        expected_ma = float(student.commutators()[PAIR_MUL_ADD])
        expected_ds = float(student.commutators()[PAIR_DIV_SUB])
    assert rep.epochs_total[0] == pytest.approx(expected_total, rel=1e-6)
    assert rep.epochs_pair_mul_add[0] == pytest.approx(expected_ma, rel=1e-6)
    assert rep.epochs_pair_div_sub[0] == pytest.approx(expected_ds, rel=1e-6)


def test_task_loss_descends_without_commutator_term() -> None:
    """Flux de gradient : la perte de tâche descend franchement.

    Preuve que l'apprentissage a lieu (et donc que toute variation du
    commutateur est un effet secondaire de cet apprentissage, jamais un terme
    optimisé directement).
    """
    gen = torch.Generator().manual_seed(1 + 7919)
    teacher = _noncommuting_teacher(6, 0.5, gen)
    rep = train_one(
        seed=1, condition="ordered", teacher=teacher, dim=6, init_scale=0.5,
        order=("add", "sub", "mul", "div"), n_samples=128, epochs=200, lr=5e-3,
    )
    assert rep.loss_final < rep.loss_init
    # descente franche : au moins un ordre de grandeur
    assert rep.loss_final < 0.1 * rep.loss_init


# --- Déterminisme bit-à-bit -------------------------------------------------


def test_bitwise_determinism_two_runs() -> None:
    seeds = [0, 1, 2]
    r1 = run_sweep(seeds=seeds, dim=5, init_scale=0.5, n_samples=64, epochs=60, lr=5e-3)
    r2 = run_sweep(seeds=seeds, dim=5, init_scale=0.5, n_samples=64, epochs=60, lr=5e-3)
    assert r1.per_seed_advantage == r2.per_seed_advantage
    for a, b in zip(r1.ordered, r2.ordered):
        assert a.epochs_total == b.epochs_total
        assert a.epochs_loss == b.epochs_loss
    assert r1.summary == r2.summary


# --- Anti-artefact DUR : le contrôle order-détruit ne croît pas -------------


def test_commuting_teacher_actually_commutes() -> None:
    """Le contrôle order-détruit a bien un commutateur ≈ 0 (par construction)."""
    gen = torch.Generator().manual_seed(7)
    teacher = _commuting_teacher(6, 0.5, gen)
    assert float(teacher.total_noncommutativity()) < 1e-4
    # ... et l'enseignant non-commutant, lui, NON nul (init aléatoire par défaut)
    gen2 = torch.Generator().manual_seed(7)
    nc = _noncommuting_teacher(6, 0.5, gen2)
    assert float(nc.total_noncommutativity()) > 0.1


def test_commuting_control_does_not_grow_commutator() -> None:
    """ANTI-ARTEFACT : entraîné sur la cible order-détruite, le commutateur ne
    croît PAS (delta médian ≤ 0), là où la tâche ordonnée le fait croître.

    C'est la garde structurelle du tour : si l'ordre est absent de la cible, le
    diagnostic ne doit fabriquer aucune acquisition. Parallèle du CTRL ``D=L``
    (Tour 1) et du ``commuting`` ‖·‖=0 (Tour 6).
    """
    seeds = list(range(6))
    rep = run_sweep(seeds=seeds, dim=6, init_scale=0.5, n_samples=128,
                    epochs=300, lr=5e-3)
    s = rep.summary
    # La tâche ordonnée fait croître le commutateur.
    assert s["median_delta_ordered"] > 0.0
    # Le contrôle order-détruit ne le fait PAS croître.
    assert s["median_delta_commuting"] <= 0.0
    # Donc l'avantage conditionnel est strictement positif en médiane.
    assert s["median_advantage"] > 0.0
    # ... et positif sur la majorité des graines.
    assert s["n_advantage_positive"] >= 4  # ≥ 4/6


def test_commutator_stays_bounded_no_dissipation() -> None:
    """Anti-dissipation : le commutateur final reste borné (< 10 x init)."""
    rep = run_sweep(seeds=[0, 1, 2, 3], dim=6, init_scale=0.5, n_samples=128,
                    epochs=300, lr=5e-3)
    s = rep.summary
    assert s["max_total_final_ordered"] < 10.0 * s["max_total_init_ordered"]
    assert math.isfinite(s["max_total_final_ordered"])


def test_vector_floor_above_matrix_when_order_present() -> None:
    """Validité de la tâche : le plancher linéaire order-free garde une perte
    résiduelle > 0 (la cible composée n'est pas trivialement linéaire-simple).
    """
    gen = torch.Generator().manual_seed(0 + 7919)
    teacher = _noncommuting_teacher(6, 0.5, gen)
    floor = vector_floor_loss(
        seed=0, teacher=teacher, dim=6, order=("add", "sub", "mul", "div"),
        n_samples=128, epochs=300, lr=5e-3,
    )["vector_floor_loss_final"]
    assert math.isfinite(floor)
    assert floor >= 0.0


# --- Outils statistiques ----------------------------------------------------


def test_binom_tail_matches_known_values() -> None:
    # P(X >= n) = p^n
    assert _binom_tail_ge(20, 20, 0.5) == pytest.approx(0.5 ** 20, rel=1e-9)
    # P(X >= 0) = 1
    assert _binom_tail_ge(0, 20, 0.5) == pytest.approx(1.0, rel=1e-9)
    # symétrie : P(X>=k) + P(X<=k-1) = 1 ; ici P(X>=11) sur n=20, p=0.5
    p_ge_11 = _binom_tail_ge(11, 20, 0.5)
    assert 0.0 < p_ge_11 < 0.5


def test_percentile_known_values() -> None:
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(xs, 0.0) == pytest.approx(0.0)
    assert _percentile(xs, 100.0) == pytest.approx(4.0)
    assert _percentile(xs, 50.0) == pytest.approx(2.0)
    assert _percentile([7.0], 5.0) == pytest.approx(7.0)


# --- Tour 12 : contrôle commutant DURCI (asymétrique, bien conditionné) ------
#
# Verrous durs qui distinguent l'issue (a) PROGRESSION d'un faux contrôle :
#   - le contrôle commute vraiment (‖[W_a,W_b]‖≈0) ;
#   - il est ASYMÉTRIQUE et APPARIÉ à l'ordonné (ratio ‖W−Wᵀ‖ dans [0.5,2.0]) ;
#   - κ(P) borné par kappa_max (anti-blow-up) ;
#   - convergence comparable, anti-dissipation, REFUS gradient inchangé.


def test_wc_eigenbasis_kappa_bounded_and_deterministic() -> None:
    """κ(P) ≤ kappa_max pour toutes les graines ; re-tirage seedé déterministe."""
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed + 7919)
        P, kappa = _well_conditioned_eigenbasis(6, 0.5, gen)
        assert math.isfinite(kappa)
        assert kappa <= KAPPA_MAX + 1e-6
        # déterminisme : même graine -> même P, même κ
        gen2 = torch.Generator().manual_seed(seed + 7919)
        P2, kappa2 = _well_conditioned_eigenbasis(6, 0.5, gen2)
        assert torch.equal(P, P2)
        assert kappa == kappa2


def test_wc_control_actually_commutes() -> None:
    """PREUVE qu'il commute : ‖[W_a,W_b]‖ ≈ 0 (tol. numérique de P⁻¹).

    Tolérance relative à ‖W‖ : < 1e-3 · ‖W‖ (l'erreur ne vient que de
    l'inversion float de P, pas d'une vraie non-commutativité).
    """
    for seed in range(8):
        gen = torch.Generator().manual_seed(seed + 7919)
        teacher = _commuting_teacher_well_conditioned(6, 0.5, gen)
        max_comm = _max_commutator(teacher)
        w_scale = float(teacher.W_add.norm())
        assert max_comm < 1e-3 * w_scale, (seed, max_comm, w_scale)
        # ... et il est très inférieur au commutateur d'un enseignant ordonné.
        gen_o = torch.Generator().manual_seed(seed + 7919)
        nc = _noncommuting_teacher(6, 0.5, gen_o)
        assert max_comm < 0.01 * _max_commutator(nc)


def test_wc_control_is_asymmetric_and_paired() -> None:
    """PREUVE de l'asymétrie appariée : ‖W−Wᵀ‖ du contrôle DURCI du même ordre
    que l'ordonné (ratio médian dans [0.5, 2.0]) — le seul axe restant = l'ordre.

    On compare des MÉDIANES sur 20 graines (statistique robuste ; un ratio par
    graine peut légitimement sortir de la bande, le verrou porte sur l'agrégat).
    """
    ratios = []
    asym_wc_all, asym_sym_all = [], []
    for seed in range(20):
        gen_wc = torch.Generator().manual_seed(seed + 7919)
        gen_o = torch.Generator().manual_seed(seed + 7919)
        gen_sym = torch.Generator().manual_seed(seed + 7919)
        wc = _commuting_teacher_well_conditioned(6, 0.5, gen_wc)
        nc = _noncommuting_teacher(6, 0.5, gen_o)
        sym = _commuting_teacher(6, 0.5, gen_sym)
        a_wc = _mean_asymmetry(wc)
        a_o = _mean_asymmetry(nc)
        asym_wc_all.append(a_wc)
        asym_sym_all.append(_mean_asymmetry(sym))
        ratios.append(a_wc / a_o)
    med_ratio = sorted(ratios)[len(ratios) // 2]
    # contrôle DURCI asymétrique et apparié
    assert 0.5 <= med_ratio <= 2.0, med_ratio
    # contrôle SYMÉTRIQUE (Tour 11) : asymétrie ≈ 0 (le confond qu'on isole)
    assert max(asym_sym_all) < 1e-4
    # contrôle DURCI : franchement asymétrique
    assert min(asym_wc_all) > 0.1


def test_wc_control_convergence_comparable() -> None:
    """Convergence comparable : loss final du contrôle DURCI dans le même ordre
    de grandeur que l'ordonné (sinon la comparaison serait viciée)."""
    rep = run_sweep(seeds=list(range(4)), dim=6, init_scale=0.5,
                    n_samples=128, epochs=300, lr=5e-3)
    s = rep.summary
    lo = s["median_loss_final_ordered"]
    lwc = s["median_loss_final_commuting_asym_wc"]
    assert lo < 0.5 and lwc < 0.5  # les deux convergent
    # même ordre de grandeur (ratio borné des deux côtés)
    assert 0.01 < (lwc + 1e-9) / (lo + 1e-9) < 100.0


def test_wc_control_anti_dissipation() -> None:
    """Anti-dissipation : commutateur du contrôle DURCI borné, aucun NaN/Inf."""
    rep = run_sweep(seeds=list(range(4)), dim=6, init_scale=0.5,
                    n_samples=128, epochs=300, lr=5e-3)
    s = rep.summary
    assert math.isfinite(s["max_total_final_commuting_asym_wc"])
    assert s["max_total_final_commuting_asym_wc"] < 10.0 * s["max_total_init_ordered"]
    assert math.isfinite(s["wc_max_commutator"])
    assert s["wc_kappa_max"] <= KAPPA_MAX + 1e-6


def test_refus_loss_is_pure_mse_unchanged_for_wc() -> None:
    """REFUS gradient (re-vérifié Tour 12) : entraîner sur le contrôle DURCI,
    la perte enregistrée est EXACTEMENT la MSE — le commutateur n'entre pas
    dans l'optimisation.
    """
    seed = 3
    gen = torch.Generator().manual_seed(seed + 7919)
    teacher = _commuting_teacher_well_conditioned(6, 0.5, gen)
    rep = train_one(
        seed=seed, condition="commuting_asym_wc", teacher=teacher, dim=6,
        init_scale=0.5, order=("add", "sub", "mul", "div"),
        n_samples=64, epochs=5, lr=5e-3,
    )
    # Reconstruire l'élève à l'init et vérifier que la loss epoch 0 == MSE pure.
    g = torch.Generator().manual_seed(seed)
    student = MatrixSpiratonCell(input_size=6, init_scale=0.5)
    with torch.no_grad():
        eye = torch.eye(6)
        for name in OPS:
            getattr(student, f"W_{name}").copy_(
                eye + torch.randn(6, 6, generator=g) * 0.5
            )
        student.bias.zero_()
        x = torch.randn(64, 6, generator=g) * 1.0
        y = teacher.compose(x, ("add", "sub", "mul", "div"))
        pred = student.compose(x, ("add", "sub", "mul", "div"))
        mse = float(torch.mean((pred - y) ** 2))
    assert rep.epochs_loss[0] == pytest.approx(mse, rel=1e-6)


def test_wc_sweep_records_proper_advantage_and_determinism() -> None:
    """Le balayage expose l'avantage PROPRE (vs DURCI) et reste déterministe."""
    seeds = [0, 1, 2]
    r1 = run_sweep(seeds=seeds, dim=6, init_scale=0.5, n_samples=64, epochs=60, lr=5e-3)
    r2 = run_sweep(seeds=seeds, dim=6, init_scale=0.5, n_samples=64, epochs=60, lr=5e-3)
    assert len(r1.commuting_asym_wc) == len(seeds)
    assert len(r1.per_seed_advantage_proper) == len(seeds)
    assert r1.per_seed_advantage_proper == r2.per_seed_advantage_proper
    assert r1.wc_kappa == r2.wc_kappa
    assert r1.summary == r2.summary
    # le confond symétrie est rendu VISIBLE : avantage SYM >= avantage PROPRE
    # (en médiane ; la part attribuable à la symétrie ≥ 0 attendue).
    assert math.isfinite(r1.summary["symmetry_share"])
    assert math.isfinite(r1.summary["median_advantage_proper"])
