"""Tests Tour 4 — Oscilloscope2D + shape_signature (opérateurs = générateurs de forme).

Quatuor canon (seeds fixés, finitude, formes simple/batch, formule exacte sous
paramètres forcés, flux de gradient au sens applicable) + tests anti-artefact
(la baseline aléatoire ne trace PAS cercles/spirales comme les réglages ; les
gardes anti-trivial rejettent point fixe et divergence droite) + déterminisme.
"""
import math

import torch
import pytest

from spiraton.experimental.oscilloscope import (
    InputSignal,
    Oscilloscope2D,
    rotation_matrix,
)
from spiraton.diagnostics.shape_signature import (
    mann_whitney_u,
    shape_signature,
)


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# --- formes simple / batch + finitude ----------------------------------------

@pytest.mark.parametrize("steps", [10, 30])
def test_trace_shape_and_finite(steps: int) -> None:
    cell = Oscilloscope2D.circle(omega=0.4)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=steps, signal=InputSignal(kind="zero"))
    assert tr.shape == (steps + 1, 2)
    assert _finite(tr)


def test_forward_single_and_batch() -> None:
    cell = Oscilloscope2D.circle(omega=0.4)
    sig = InputSignal(kind="zero")
    # singleton + trace
    s0 = torch.tensor([1.0, 0.0])
    sf, tr = cell(s0, steps=12, signal=sig, return_trace=True)
    assert sf.shape == (2,)
    assert tr.shape == (13, 2)
    assert torch.allclose(sf, tr[-1])
    # batch (état final seulement)
    s0b = torch.randn(5, 2)
    out = cell(s0b, steps=12, signal=sig)
    assert out.shape == (5, 2)
    assert _finite(out)


def test_batch_matches_single_trajectory() -> None:
    """La trajectoire batch d'un point == la trajectoire singleton (cohérence)."""
    cell = Oscilloscope2D.spiral(omega=0.4, gain=1.05)
    sig = InputSignal(kind="zero")
    s0 = torch.tensor([0.3, -0.7])
    out_single = cell(s0, steps=15, signal=sig)
    out_batch = cell(s0.unsqueeze(0), steps=15, signal=sig)
    assert torch.allclose(out_single, out_batch.squeeze(0), atol=1e-6)


# --- formule exacte sous paramètres forcés -----------------------------------

def test_rotation_matrix_is_isometry() -> None:
    """R(ω) est une rotation pure : R Rᵀ = I, det R = 1 (préserve la norme)."""
    for omega in (0.0, 0.4, 1.3, math.pi / 5):
        R = rotation_matrix(omega)
        assert torch.allclose(R @ R.t(), torch.eye(2), atol=1e-6)
        assert abs(float(torch.det(R)) - 1.0) < 1e-6
    v = torch.tensor([0.6, -0.8])
    Rv = rotation_matrix(0.7) @ v
    assert abs(float(Rv.norm()) - float(v.norm())) < 1e-6


def test_step_exact_formula() -> None:
    """s_{t+1} = A·s_t − memory·s_prev + W_in·u_t, à la valeur exacte."""
    cell = Oscilloscope2D(omega=0.4, gain=1.06, memory=0.3, win_scale=2.0)
    s_t = torch.tensor([[0.5, -0.2]])
    s_prev = torch.tensor([[0.1, 0.4]])
    u_t = torch.tensor([[1.0, 0.0]])
    A = 1.06 * rotation_matrix(0.4)
    expected = s_t @ A.t() - 0.3 * s_prev + u_t @ (2.0 * torch.eye(2)).t()
    got = cell.step(s_t, s_prev, u_t)
    assert torch.allclose(got, expected, atol=1e-6)


def test_circle_preserves_radius_exactly() -> None:
    """Réglage cercle (gain=1, memory=0, signal=zero) : ‖s_t‖ constant (isométrie)."""
    cell = Oscilloscope2D.circle(omega=0.5)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=40, signal=InputSignal(kind="zero"))
    radii = torch.linalg.vector_norm(tr, dim=-1)
    assert torch.allclose(radii, torch.ones_like(radii), atol=1e-5)


def test_spiral_radius_grows_geometrically() -> None:
    """Réglage spirale (gain>1) : ‖s_t‖ = gainᵗ·‖s0‖ (croissance géométrique)."""
    gain = 1.06
    cell = Oscilloscope2D.spiral(omega=0.5, gain=gain)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=20, signal=InputSignal(kind="zero"))
    radii = torch.linalg.vector_norm(tr, dim=-1)
    expected = torch.tensor([gain ** t for t in range(21)])
    assert torch.allclose(radii, expected, atol=1e-4)


def test_spiral_requires_gain_above_one() -> None:
    with pytest.raises(ValueError):
        Oscilloscope2D.spiral(gain=1.0)


# --- flux de gradient (l'injection W_in/transition est différentiable) --------

def test_gradient_flows_through_rollout() -> None:
    """Le gradient remonte de l'état final vers s0 (carte linéaire différentiable).

    On passe par le chemin BATCH du forward, qui est différentiable (le chemin
    ``trace()`` est explicitement ``@torch.no_grad`` car c'est un diagnostic).
    """
    cell = Oscilloscope2D.spiral(omega=0.4, gain=1.03)
    s0 = torch.tensor([[0.5, 0.5]], requires_grad=True)
    out = cell(s0, steps=8, signal=InputSignal(kind="zero"))
    out.sum().backward()
    assert s0.grad is not None
    assert float(s0.grad.abs().sum()) > 0.0


# --- signature de forme : cercle vs spirale ----------------------------------

OMEGA = math.pi / 5  # 10 pas/tour : 2nde moitié = nb entier de tours


def test_signature_circle_is_circular() -> None:
    cell = Oscilloscope2D.circle(omega=OMEGA)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=120, signal=InputSignal(kind="zero"))
    sig = shape_signature(tr, s0)
    assert sig.cv_r < 0.05                 # rayon quasi constant
    assert sig.r2_theta_t > 0.98           # phase monotone
    assert sig.passes_circle_guards()      # rayon non nul + phase monotone
    assert abs(sig.slope_logr_theta) < 0.01  # pas de croissance radiale


def test_signature_spiral_is_logspiral() -> None:
    cell = Oscilloscope2D.spiral(omega=OMEGA, gain=1.06)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=120, signal=InputSignal(kind="zero"))
    sig = shape_signature(tr, s0)
    assert sig.r2_logr_theta > 0.95        # log r affine en θ
    assert sig.slope_logr_theta > 0.02     # pente positive (sortante)
    assert sig.passes_spiral_guards(min_turns=2.0)


def test_circle_is_repetition_spiral_is_progression_alpha_omega() -> None:
    """Pont α-ω : cercle revient au même (RÉPÉTITION) ; spirale s'éloigne (PROGRESSION).

    Cercle : best_return_step ≈ une période (revient sur s0), l2_final ≈ 0.
    Spirale : cos reste élevé (aligné) mais l2_final >> 0 (point modifié, l.290).
    """
    s0 = torch.tensor([1.0, 0.0])
    sig = InputSignal(kind="zero")

    circ = shape_signature(
        Oscilloscope2D.circle(omega=OMEGA).trace(s0, steps=120, signal=sig), s0
    )
    spir = shape_signature(
        Oscilloscope2D.spiral(omega=OMEGA, gain=1.06).trace(s0, steps=120, signal=sig), s0
    )
    # RÉPÉTITION : le cercle revient quasi exactement (l2 final négligeable).
    assert circ.ao_l2_final < 0.05
    assert circ.ao_cos_final > 0.99
    # PROGRESSION : la spirale reste alignée mais s'est nettement éloignée.
    assert spir.ao_cos_final > 0.9
    assert spir.ao_l2_final > 1.0
    # best_return EXCLUT t=0 (sinon trivial) ; le cercle revient à une période entière.
    assert circ.ao_best_return_step >= 1


# --- ANTI-ARTEFACT : la baseline aléatoire ne trace PAS les figures -----------

def test_random_baseline_does_not_draw_circle() -> None:
    """Sur 30 graines, aucune transition aléatoire ne passe le critère cercle."""
    n_pass = 0
    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        cell = Oscilloscope2D.random(g, scale=0.6)
        s0 = torch.tensor([1.0, 0.0])
        tr = cell.trace(s0, steps=120, signal=InputSignal(kind="zero"))
        sig = shape_signature(tr, s0)
        v = float(sig.cv_r)
        if math.isfinite(v) and v < 0.05 and sig.passes_circle_guards():
            n_pass += 1
    # 0 cercle attendu : un cercle exige |λ|=1 exactement, mesure nulle pour A aléatoire.
    assert n_pass == 0


def test_random_baseline_rarely_draws_spiral() -> None:
    """La baseline trace RAREMENT une spirale propre (vs réglage : toujours).

    Une matrice 2×2 aléatoire à valeurs propres complexes |λ|>1 EST une spirale
    légitime — donc on n'exige pas 0, mais une fraction faible (≤ ~15%), bien
    en-deçà du réglage ciblé (100%). C'est la nuance honnête : le réglage est
    FIABLE, la baseline OCCASIONNELLE.
    """
    n_pass = 0
    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        cell = Oscilloscope2D.random(g, scale=0.6)
        s0 = torch.tensor([1.0, 0.0])
        tr = cell.trace(s0, steps=120, signal=InputSignal(kind="zero"))
        sig = shape_signature(tr, s0)
        if (
            math.isfinite(sig.r2_logr_theta)
            and sig.r2_logr_theta > 0.95
            and abs(sig.slope_logr_theta) > 0.02
            and sig.passes_spiral_guards(min_turns=2.0)
        ):
            n_pass += 1
    assert n_pass <= 5  # ≤ ~15% sur 30 graines
    # et le réglage spirale, lui, passe systématiquement :
    s0 = torch.tensor([1.0, 0.0])
    sp = shape_signature(
        Oscilloscope2D.spiral(omega=OMEGA, gain=1.06).trace(
            s0, steps=120, signal=InputSignal(kind="zero")
        ),
        s0,
    )
    assert (
        sp.r2_logr_theta > 0.95
        and abs(sp.slope_logr_theta) > 0.02
        and sp.passes_spiral_guards(min_turns=2.0)
    )


# --- ANTI-TRIVIAL : gardes rejettent point fixe et divergence droite ----------

def test_guards_reject_fixed_point() -> None:
    """Un point fixe r→0 a un CV(r) parfait mais doit être REJETÉ (radius_floor)."""
    # transition contractante forte : l'état collapse vers 0.
    A = 0.01 * torch.eye(2)
    cell = Oscilloscope2D(transition=A)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=120, signal=InputSignal(kind="zero"))
    sig = shape_signature(tr, s0)
    # μ_r quasi nul → garde radius_floor échoue → pas un cercle valide.
    assert not sig.passes_radius_floor
    assert not sig.passes_circle_guards()


def test_guards_reject_straight_divergence() -> None:
    """Une divergence en ligne droite (pas de rotation) n'est PAS une spirale."""
    # transition = expansion isotrope SANS rotation (ω=0) : ligne droite sortante.
    cell = Oscilloscope2D(omega=0.0, gain=1.1)
    s0 = torch.tensor([1.0, 0.0])
    tr = cell.trace(s0, steps=120, signal=InputSignal(kind="zero"))
    sig = shape_signature(tr, s0)
    # aucune rotation → n_turns ≈ 0 → garde spirale (min_turns=2) échoue.
    assert sig.n_turns < 2.0
    assert not sig.passes_spiral_guards(min_turns=2.0)


# --- statistique Mann-Whitney : sanité ---------------------------------------

def test_mann_whitney_separates_clearly() -> None:
    """Deux groupes nettement séparés → p très petit ; identiques → p ~ 1."""
    a = [0.0 + 0.001 * i for i in range(20)]       # ~0
    b = [5.0 + 0.001 * i for i in range(20)]       # ~5
    _, p_sep = mann_whitney_u(a, b)
    assert p_sep < 0.01
    _, p_same = mann_whitney_u(a, list(a))
    assert p_same > 0.5


def test_mann_whitney_circle_beats_baseline_cv_r() -> None:
    """CV(r) cercle < CV(r) baseline, séparation significative (p<0.01)."""
    circ_cv, base_cv = [], []
    for seed in range(30):
        g = torch.Generator().manual_seed(seed)
        s0 = torch.tensor([1.0, 0.0])
        zero = InputSignal(kind="zero")
        cs = shape_signature(
            Oscilloscope2D.circle(omega=OMEGA).trace(s0, steps=120, signal=zero), s0
        )
        bs = shape_signature(
            Oscilloscope2D.random(g, scale=0.6).trace(s0, steps=120, signal=zero), s0
        )
        circ_cv.append(float(cs.cv_r))
        v = float(bs.cv_r)
        base_cv.append(v if math.isfinite(v) else 10.0)
    _, p = mann_whitney_u(circ_cv, base_cv)
    assert p < 0.01
    assert sorted(circ_cv)[len(circ_cv) // 2] < sorted(base_cv)[len(base_cv) // 2]


# --- déterminisme bit-à-bit ---------------------------------------------------

def test_determinism_bit_for_bit() -> None:
    """Deux runs identiques produisent des traces bit-à-bit identiques."""
    def run():
        g = torch.Generator().manual_seed(7)
        cell = Oscilloscope2D.random(g, scale=0.6)
        s0 = torch.tensor([0.3, -0.4])
        return cell.trace(s0, steps=50, signal=InputSignal(kind="zero"))

    a = run()
    b = run()
    assert torch.equal(a, b)
