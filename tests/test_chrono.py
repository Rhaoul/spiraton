import torch
import pytest

from spiraton.experimental.chrono import ChronoSpiraton


def _no_nan_inf(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


@pytest.mark.parametrize("d,batch", [(4, 1), (6, 3)])
def test_shapes_and_finiteness(d: int, batch: int) -> None:
    torch.manual_seed(0)
    chrono = ChronoSpiraton(state_size=d, init_scale=0.05)

    s0 = torch.randn(batch, d) * 0.1
    s = chrono(s0, steps=5)
    assert s.shape == (batch, d)
    assert _no_nan_inf(s)

    # singleton + trace
    s0s = torch.randn(d) * 0.1
    sf, trace = chrono(s0s, steps=3, return_trace=True)
    assert sf.shape == (d,)
    assert len(trace) == 3
    assert all(t.shape == (d,) for t in trace)


def test_second_order_memory_enters_update() -> None:
    """s_{t−1} entre réellement dans la mise à jour (vs premier ordre)."""
    torch.manual_seed(1)
    d = 5
    chrono = ChronoSpiraton(state_size=d, init_scale=0.2)

    s_t = torch.randn(2, d)
    prev_a = torch.randn(2, d)
    prev_b = torch.randn(2, d)
    out_a = chrono.step(s_t, prev_a)
    out_b = chrono.step(s_t, prev_b)
    # Même s_t, s_prev différent => sortie différente (terme −C(s_{t−1})).
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_quadratic_term_is_nonlinear() -> None:
    """Le terme B(s²) rend la dynamique non linéaire : f(2s) != 2 f(s)."""
    torch.manual_seed(2)
    d = 4
    chrono = ChronoSpiraton(state_size=d, init_scale=0.3)
    # Forcer B non nul, et un s_prev nul pour isoler la non-linéarité en s_t.
    s_t = torch.randn(1, d)
    zero_prev = torch.zeros(1, d)
    f_s = chrono.step(s_t, zero_prev)
    f_2s = chrono.step(2.0 * s_t, zero_prev)
    assert not torch.allclose(f_2s, 2.0 * f_s, atol=1e-4)

    # En annulant B, le pas redevient linéaire en s_t (avec s_prev=0).
    with torch.no_grad():
        chrono.B.weight.zero_()
    g_s = chrono.step(s_t, zero_prev)
    g_2s = chrono.step(2.0 * s_t, zero_prev)
    assert torch.allclose(g_2s, 2.0 * g_s, atol=1e-5)


def test_matches_manual_formula_when_params_forced() -> None:
    torch.manual_seed(3)
    d = 3
    chrono = ChronoSpiraton(state_size=d)

    with torch.no_grad():
        A = torch.randn(d, d) * 0.1
        B = torch.randn(d, d) * 0.1
        C = torch.randn(d, d) * 0.1
        D = torch.randn(d, d) * 0.1
        L = torch.randn(d, d) * 0.1
        chrono.A.weight.copy_(A)
        chrono.B.weight.copy_(B)
        chrono.C.weight.copy_(C)
        chrono.D.weight.copy_(D)
        chrono.L.weight.copy_(L)

    s_t = torch.randn(2, d)
    s_prev = torch.randn(2, d)

    inner = s_t @ A.t() + (s_t * s_t) @ B.t() - s_prev @ C.t()
    expected = inner @ D.t() + s_t @ L.t()
    got = chrono.step(s_t, s_prev)
    assert torch.allclose(got, expected, atol=1e-6)


def test_gradients_flow() -> None:
    torch.manual_seed(4)
    d = 6
    chrono = ChronoSpiraton(state_size=d, init_scale=0.1)
    s0 = torch.randn(4, d, requires_grad=True) * 0.1
    chrono(s0, steps=4).sum().backward()
    for name in ("A", "B", "C", "D", "L"):
        g = getattr(chrono, name).weight.grad
        assert g is not None, f"pas de gradient sur {name}"
    total = sum(getattr(chrono, n).weight.grad.abs().sum() for n in ("A", "B", "C", "D", "L"))
    assert float(total) > 0.0


def test_bounded_mode_stays_finite_on_long_rollout() -> None:
    """bounded=True garde une trajectoire bornée même sur un long déroulé."""
    torch.manual_seed(5)
    d = 8
    chrono = ChronoSpiraton(state_size=d, init_scale=0.5, bounded=True)
    s0 = torch.randn(3, d)
    report = chrono.stability_scan(s0, steps=200)
    assert not report["diverged"]
    # Tout état PRODUIT par step subit un tanh => norme finale ≤ sqrt(d).
    # (max_norm peut inclure s0, non borné car fourni par l'appelant.)
    assert report["final_norm"] <= (d ** 0.5) + 1e-4
    assert report["max_norm"] < 1e6


def test_stability_scan_keys() -> None:
    torch.manual_seed(6)
    chrono = ChronoSpiraton(state_size=4, init_scale=0.05)
    rep = chrono.stability_scan(torch.randn(2, 4) * 0.1, steps=20)
    assert set(rep.keys()) == {"final_norm", "max_norm", "diverged"}
