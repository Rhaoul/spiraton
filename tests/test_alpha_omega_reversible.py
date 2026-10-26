"""Tests ``AlphaOmegaReversible`` (experimental/alpha_omega_reversible.py).

Validation, pivot ``AlphaOmegaOpt`` au bit (``seuil_annulation=None``), pas défait
exactement (θ_t rendu au bit, m, v et compteur de pas intacts, gain coupé), pas
suivant non jugé, groupes ajoutés après coup, reprise ``state_dict`` au bit,
déterminisme et finitude float64 sur ``ChronoSpiraton``.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Dict

import pytest
import torch

from spiraton.experimental import AlphaOmegaOpt, AlphaOmegaReversible
from spiraton.experimental.chrono import ChronoSpiraton
from spiraton.training.aba_loss import alpha_omega_loss


def _seq(n: int, shape=(7, 5), seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g, dtype=torch.float64)
    return w, [torch.randn(*shape, generator=g, dtype=torch.float64) for _ in range(n)]


def _hash(state: Dict[str, torch.Tensor]) -> str:
    blob = b"".join(state[k].detach().cpu().numpy().astype("<f8").tobytes() for k in sorted(state))
    return hashlib.sha256(blob).hexdigest()


def test_validation() -> None:
    p = [torch.nn.Parameter(torch.zeros(2))]
    for seuil in (0.0, 0.3, -1.5):
        with pytest.raises(ValueError):
            AlphaOmegaReversible(p, seuil_annulation=seuil)
    for kappa in (0.0, 1.0, -0.5):
        with pytest.raises(ValueError):
            AlphaOmegaReversible(p, kappa=kappa)
    opt = AlphaOmegaReversible(p, seuil_annulation=None, kappa=0.25, eta=0.3)
    grp = opt.param_groups[0]
    assert (grp["seuil_annulation"], grp["kappa"], grp["eta"], grp["n_annul"]) == (None, 0.25, 0.3, 0)


def _chrono(cls, **kw):
    torch.manual_seed(64)
    model = ChronoSpiraton(state_size=19, init_scale=0.6, bounded=False).double()
    a = torch.randn(32, 19, generator=torch.Generator().manual_seed(1064), dtype=torch.float64)
    opt = cls(model.parameters(), lr=1.0, **kw)
    for _ in range(60):
        opt.zero_grad()
        alpha_omega_loss(model(a, steps=2), a, target_dist=0.3).loss.backward()
        opt.step()
    return model, opt


def test_pivot_alpha_omega_bit_for_bit() -> None:
    m_ref, o_ref = _chrono(AlphaOmegaOpt)
    m_rev, o_rev = _chrono(AlphaOmegaReversible, seuil_annulation=None)
    assert _hash(m_ref.state_dict()) == _hash(m_rev.state_dict())
    g_ref, g_rev = o_ref.param_groups[0], o_rev.param_groups[0]
    assert (g_ref["g"], g_ref["omega_bar"], g_ref["n_regul"]) == (g_rev["g"], g_rev["omega_bar"], g_rev["n_regul"])
    assert g_rev["n_annul"] == 0


def test_pas_rate_defait_au_bit() -> None:
    """f = ½x², x0 = 0,3, g0 = 1, lr = 1 : le premier pas (≈ −1) franchit le creux,
    le gradient d'arrivée s'y oppose (ω = −1) : le pas est défait."""
    x = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float64))
    opt = AlphaOmegaReversible([x], lr=1.0, weight_decay=0.0, g0=1.0, kappa=0.5)

    def un_pas():
        opt.zero_grad()
        (0.5 * x * x).sum().backward()
        opt.step()

    x0 = x.detach().clone()
    un_pas()
    x1 = x.detach().clone()
    assert float(x1) < 0.0  # le creux est franchi
    m1, v1 = opt.state[x]["exp_avg"].clone(), opt.state[x]["exp_avg_sq"].clone()
    un_pas()
    grp = opt.param_groups[0]
    assert torch.equal(x.detach(), x0)  # θ_t rendu au bit
    assert grp["omega_last"] == -1.0 and grp["n_annul"] == 1 and grp["n_regul"] == 1
    assert math.isclose(grp["g"], 0.5 * math.exp(0.1 * 0.1 * -1.0), rel_tol=1e-15)
    assert torch.equal(opt.state[x]["exp_avg"], m1) and torch.equal(opt.state[x]["exp_avg_sq"], v1)
    assert opt.state[x]["step"] == 1 and torch.count_nonzero(opt.state[x]["delta"]) == 0
    g_coupe = grp["g"]
    un_pas()  # rien à juger : le pas suit depuis θ_t avec le gain coupé
    assert grp["n_regul"] == 1 and grp["n_annul"] == 1 and grp["g"] == g_coupe
    assert opt.state[x]["step"] == 2
    assert abs(float(x.detach())) < abs(float(x1))  # le pas plus court ne franchit plus aussi loin


def test_groupe_ajoute_apres_coup() -> None:
    a, b = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))
    opt = AlphaOmegaReversible([a], seuil_annulation=-0.8)
    opt.add_param_group({"params": [b], "kappa": 0.2})
    grp = opt.param_groups[1]
    assert (grp["seuil_annulation"], grp["kappa"], grp["n_annul"], grp["g"]) == (-0.8, 0.2, 0, 1e-2)


def test_state_dict_resume_bit_for_bit() -> None:
    """f = ½‖x − ½‖² à grand pas : un pas est défait avant la reprise (pas 2), un autre
    après (pas 6) ; la reprise depuis ``state_dict`` suit la trajectoire au bit."""
    w, _ = _seq(n=0)

    def pas(p, opt, n):
        for _ in range(n):
            p.grad = p.detach() - 0.5
            opt.step()

    p_ref = torch.nn.Parameter(w.clone())
    o_ref = AlphaOmegaReversible([p_ref], lr=2.0, g0=1.0)
    pas(p_ref, o_ref, 12)
    p1 = torch.nn.Parameter(w.clone())
    o1 = AlphaOmegaReversible([p1], lr=2.0, g0=1.0)
    pas(p1, o1, 3)
    n_avant = o1.param_groups[0]["n_annul"]
    saved_p, saved_o = p1.detach().clone(), copy.deepcopy(o1.state_dict())
    p2 = torch.nn.Parameter(saved_p)
    o2 = AlphaOmegaReversible([p2], lr=2.0, g0=1.0)
    o2.load_state_dict(saved_o)
    pas(p2, o2, 9)
    assert 0 < n_avant < o_ref.param_groups[0]["n_annul"]
    assert torch.equal(p2.data, p_ref.data)
    assert o2.param_groups[0]["g"] == o_ref.param_groups[0]["g"]
    assert o2.param_groups[0]["n_annul"] == o_ref.param_groups[0]["n_annul"]


def test_determinisme_et_finitude_float64_sur_chrono_spiraton() -> None:
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        m1, o1 = _chrono(AlphaOmegaReversible)
        m2, _ = _chrono(AlphaOmegaReversible)
        assert _hash(m1.state_dict()) == _hash(m2.state_dict())
        assert all(torch.isfinite(t).all() for t in m1.state_dict().values())
        assert o1.param_groups[0]["n_regul"] > 0
    finally:
        torch.use_deterministic_algorithms(prev)
