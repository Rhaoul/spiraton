"""Statistiques, contrôles et adaptateur de la sonde thermo-sémantique (sans .so)."""
import random

import pytest
import torch

from spiraton.data.semantic_thermo_adapter import (
    SLICES,
    CachedVectors,
    CycleRecord,
    SegmentRecord,
    SliceNormalizer,
    recut_cycle,
    split_cycles,
)
from spiraton.diagnostics.semantic_thermo_probe import (
    auc,
    holm,
    paired_test,
    perm_null_auc,
)


def _brute_auc(s, l):
    pos = [a for a, b in zip(s, l) if b == 1]
    neg = [a for a, b in zip(s, l) if b == 0]
    tot = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return tot / (len(pos) * len(neg))


def test_auc_matches_brute_force_with_ties():
    g = torch.Generator().manual_seed(0)
    s = torch.randint(0, 5, (60,), generator=g).double()
    l = torch.randint(0, 2, (60,), generator=g).double()
    assert float(auc(s, l)) == pytest.approx(_brute_auc(s.tolist(), l.tolist()))


def test_permutation_destroys_known_signal():
    """CTRL-1 : un signal synthétique fort (AUC≈1) retombe à ≈0.5 sous permutation."""
    g = torch.Generator().manual_seed(1)
    lab = torch.tensor([1.0] * 100 + [0.0] * 100)
    sc = lab * 3 + torch.randn(200, generator=g, dtype=torch.float64)
    assert float(auc(sc, lab)) > 0.95
    null = perm_null_auc(sc, lab, torch.zeros(200, dtype=torch.long), n=2000, gen=g)
    assert abs(float(null.mean()) - 0.5) < 0.01
    assert float(null.max()) < 0.7


def test_stratified_permutation_removes_only_within_strata_signal():
    """Un signal porté par la strate (confond de position) ne survit pas au test stratifié :
    l'AUC observée reste dans la distribution nulle stratifiée."""
    g = torch.Generator().manual_seed(2)
    strata = torch.tensor([0] * 100 + [1] * 100)
    lab = torch.cat([torch.bernoulli(torch.full((100,), 0.8), generator=g),
                     torch.bernoulli(torch.full((100,), 0.2), generator=g)]).double()
    sc = (1 - strata).double() * 2 + torch.randn(200, generator=g, dtype=torch.float64)
    a = float(auc(sc, lab))
    assert a > 0.65                                  # confond visible sans stratification
    null = perm_null_auc(sc, lab, strata, n=2000, gen=g)
    assert float((null >= a).double().mean()) > 0.05  # non significatif une fois stratifié


def test_paired_test_and_holm():
    g = torch.Generator().manual_seed(3)
    d = torch.randn(300, generator=g, dtype=torch.float64) + 0.5
    r = paired_test(d, n_boot=200, gen=g)
    assert r["p_one_sided"] < 1e-6 and r["ci_low"] > 0
    r0 = paired_test(-d, n_boot=200, gen=g)
    assert r0["p_one_sided"] > 0.99
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_slices_exclude_logos_dims():
    for name, dims in SLICES.items():
        if name != "full":
            assert not set(dims) & set(range(6)), name
    assert SLICES["no-logos"] == tuple(range(6, 33))


def test_normalizer_zscore_and_constant_dims():
    rows = torch.randn(50, 33, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    rows[:, 10] = 7.0
    n = SliceNormalizer(rows, "phoneme")
    z = n(rows)
    assert z.shape == (50, 15)
    assert torch.all(z[:, 2] == 0)                      # dim 10 constante → 0
    assert torch.allclose(z[:, [0, 1, 3]].mean(0), torch.zeros(3, dtype=torch.float64), atol=1e-12)


def test_split_is_seeded_and_disjoint():
    a = split_cycles(100, seed=0)
    assert a == split_cycles(100, seed=0)
    assert not set(a[0]) & set(a[1]) and len(a[0]) + len(a[1]) == 100
    assert a != split_cycles(100, seed=1)


def test_recut_preserves_words_and_length_multiset():
    vec = CachedVectors(lambda t: torch.ones(len(t.split()), 33))
    segs = tuple(SegmentRecord(n, c, t, vec(t)) for n, c, t in
                 (("SEG_A", "DX", "un deux"), ("SEG_B", "DX", "trois quatre cinq"),
                  ("SEG_A_PRIME", "LV", "six sept huit neuf")))
    c = CycleRecord("x", 0, "ADD", "F0", segs)
    r = recut_cycle(c, vec, random.Random(5))
    assert " ".join(s.text for s in r.segments) == " ".join(s.text for s in c.segments)
    assert sorted(len(s.text.split()) for s in r.segments) == [2, 3, 4]
    assert [s.chirality for s in r.segments] == ["DX", "DX", "LV"]
