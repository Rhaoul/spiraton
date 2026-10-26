import torch

from spiraton.core.cell import SpiratonCell
from spiraton.grid import SpiralGrid
from spiraton.diagnostics import run_double_dynamics


def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def _make_grid(C: int = 5) -> SpiralGrid:
    cell = SpiratonCell(input_size=2 * C)  # grille concatène local+neigh
    return SpiralGrid(cell, channels=C, neighborhood="von_neumann", aggregator="mean")


def test_double_dynamics_shapes_and_finiteness() -> None:
    torch.manual_seed(0)
    B, H, W, C = 2, 5, 5, 5
    grid = _make_grid(C)
    x0 = torch.randn(B, H, W, C)

    k = 4
    rep = run_double_dynamics(grid, x0, k=k)

    # Séries de longueur 2K+1.
    for s in (rep.ld_l2, rep.ld_cos, rep.dl_l2, rep.dl_cos):
        assert s.shape == (2 * k + 1,)
        assert _finite(s)

    # Résumés finis et cohérents.
    assert isinstance(rep.ld_score, float) and isinstance(rep.dl_score, float)
    assert 0 <= rep.ld_best_return_step <= 2 * k
    assert 0 <= rep.dl_best_return_step <= 2 * k
    assert rep.ld_signal_variance >= 0.0 and rep.dl_signal_variance >= 0.0
    # variance_gap = dl - ld, et le verdict est cohérent avec le signe.
    assert abs((rep.dl_signal_variance - rep.ld_signal_variance) - rep.variance_gap) < 1e-9
    assert rep.stabilizes_as_predicted == (rep.ld_signal_variance < rep.dl_signal_variance)


def test_double_dynamics_starts_at_origin() -> None:
    """À t=0, on est exactement à l'origine : l2=0, cos=1, pour les deux chemins."""
    torch.manual_seed(1)
    B, H, W, C = 1, 5, 5, 4
    grid = _make_grid(C)
    x0 = torch.randn(B, H, W, C)

    rep = run_double_dynamics(grid, x0, k=3)
    assert float(rep.ld_l2[0]) == 0.0
    assert float(rep.dl_l2[0]) == 0.0
    assert abs(float(rep.ld_cos[0]) - 1.0) < 1e-5
    assert abs(float(rep.dl_cos[0]) - 1.0) < 1e-5


def test_double_dynamics_paths_differ() -> None:
    """L∘D et D∘L parcourent réellement des trajectoires distinctes.

    (Le diagnostic n'aurait aucun sens si les deux compositions coïncidaient ;
    on ne teste PAS le signe de l'asymétrie, qui est une mesure à constater.)
    """
    torch.manual_seed(2)
    B, H, W, C = 2, 5, 5, 5
    grid = _make_grid(C)
    x0 = torch.randn(B, H, W, C)

    rep = run_double_dynamics(grid, x0, k=5)
    # Les séries doivent différer quelque part (sinon ordre sans effet).
    assert not torch.allclose(rep.ld_l2, rep.dl_l2, atol=1e-6)


def test_double_dynamics_deterministic() -> None:
    torch.manual_seed(3)
    B, H, W, C = 1, 5, 5, 4
    grid = _make_grid(C)
    x0 = torch.randn(B, H, W, C)

    r1 = run_double_dynamics(grid, x0, k=4)
    r2 = run_double_dynamics(grid, x0, k=4)
    assert torch.allclose(r1.ld_l2, r2.ld_l2, atol=0.0, rtol=0.0)
    assert torch.allclose(r1.dl_cos, r2.dl_cos, atol=0.0, rtol=0.0)
    assert r1.variance_gap == r2.variance_gap
