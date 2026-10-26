"""E0 — validation instrumentale de la thermodynamique sémantique (synthétique).

Protocole : docs/SEMANTIC_THERMO_PROTOCOLE.md §3 et révision R1.
"""
import dataclasses
import math

import pytest
import torch

from spiraton.experimental.semantic_thermodynamics import (
    ReferenceField,
    SemanticThermoState,
    ThermoProbeConfig,
    continuity_divergence,
    literal_flux_divergence,
    return_ratio,
    segment_observables,
)
from spiraton.experimental.semantic_thermodynamics.synthetic import (
    closed_loop,
    e0_suite,
    gaussian_field,
    radial,
    stationary,
)


@pytest.fixture(scope="module", params=[8, 27])
def field(request):
    return ReferenceField(gaussian_field(d=request.param, seed=0), k=8)


# --- état -------------------------------------------------------------------

def test_state_is_frozen_and_finite():
    s = SemanticThermoState(1.0, 2.0, 0.1, 0.5, 0.4, 0.0, 0.3, 0.1, 3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.density = 2.0
    with pytest.raises(ValueError):
        SemanticThermoState(float("nan"), 2.0, 0.1, 0.5, 0.4, 0.0, 0.3, 0.1, 3)
    with pytest.raises(ValueError):
        ThermoProbeConfig(neighborhood_k=0)


def test_all_synthetic_observables_finite(field):
    for x in e0_suite(field.dim).values():
        s = segment_observables(x, field)
        for f in dataclasses.fields(s):
            v = getattr(s, f.name)
            if isinstance(v, float):
                assert math.isfinite(v), f.name


# --- signes attendus ----------------------------------------------------------

def test_expansion_lowers_density_positive_divergence(field):
    x = radial(field.dim, 0.2, 3.0)
    rho = field.local(x)["density"]
    assert rho[-1] < rho[0]
    assert continuity_divergence(rho) > 0
    assert segment_observables(x, field).speed > 0


def test_contraction_raises_density_negative_divergence(field):
    x = radial(field.dim, 3.0, 0.2)
    rho = field.local(x)["density"]
    assert rho[-1] > rho[0]
    assert continuity_divergence(rho) < 0


def test_stationary_zero_speed_flux_divergence(field):
    s = segment_observables(stationary(field.dim), field)
    assert s.speed == 0.0 and s.flux_norm == 0.0
    assert abs(s.divergence) < 1e-12


def test_closed_loop_returns():
    x = closed_loop(8)
    assert return_ratio(x) < 1e-12
    assert return_ratio(radial(8, 0.2, 3.0)) == pytest.approx(1.0)


def test_literal_flux_divergence_has_inverted_sign(field):
    """Résultat négatif conservé (R1) : la divergence littérale du doc §9.10
    donne le MAUVAIS signe sur l'expansion radiale — d'où son exclusion."""
    x = radial(field.dim, 0.2, 3.0)
    rho = field.local(x)["density"]
    assert literal_flux_divergence(x, rho) < 0


def test_softmax_entropy_is_saturated(field):
    """Résultat négatif conservé (R1) : S softmax reste collée à ln k."""
    for x in e0_suite(field.dim).values():
        S = field.local(x)["entropy"]
        assert float(S.min()) > 0.99 * math.log(field.k)


# --- invariances -------------------------------------------------------------

def _obs(field, x):
    s = segment_observables(x, field)
    return torch.tensor([s.density, s.divergence, s.speed, s.dispersion], dtype=torch.float64)


def test_translation_and_rotation_invariance():
    R = gaussian_field(d=8, seed=0)
    x = radial(8, 0.2, 3.0)
    base = _obs(ReferenceField(R), x)
    shift = torch.linspace(-2, 3, 8, dtype=torch.float64)
    assert torch.allclose(_obs(ReferenceField(R + shift), x + shift), base, atol=1e-9)
    q, _ = torch.linalg.qr(torch.randn(8, 8, generator=torch.Generator().manual_seed(9),
                                       dtype=torch.float64))
    assert torch.allclose(_obs(ReferenceField(R @ q), x @ q), base, atol=1e-9)


def test_reference_row_permutation_invariance():
    R = gaussian_field(d=8, seed=0)
    perm = torch.randperm(R.size(0), generator=torch.Generator().manual_seed(3))
    x = radial(8, 0.2, 3.0)
    assert torch.allclose(_obs(ReferenceField(R[perm]), x), _obs(ReferenceField(R), x), atol=1e-12)


def test_scale_covariance():
    R = gaussian_field(d=8, seed=0)
    x = radial(8, 0.2, 3.0)
    c = 2.5
    a, b = ReferenceField(R), ReferenceField(c * R)
    sa, sb = segment_observables(x, a), segment_observables(c * x, b)
    assert b.h == pytest.approx(c * a.h)
    assert sb.density == pytest.approx(sa.density / c, rel=1e-6)
    assert sb.temperature == pytest.approx(sa.temperature * c * c, rel=1e-6)
    assert sb.entropy == pytest.approx(sa.entropy, rel=1e-9)
    assert sb.divergence == pytest.approx(sa.divergence, rel=1e-6)


def test_divergence_is_order_sensitive_density_is_not():
    """CTRL-2 sur signal connu : renverser l'ordre inverse div, pas ρ̄."""
    F = ReferenceField(gaussian_field(d=8, seed=0))
    x = radial(8, 0.2, 3.0)
    fwd, bwd = segment_observables(x, F), segment_observables(x.flip(0), F)
    assert fwd.density == pytest.approx(bwd.density)
    assert fwd.divergence == pytest.approx(-bwd.divergence)
