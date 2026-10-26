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


def test_sentence_mode_splits_by_segment_token_counts():
    """R2 : la phrase jointe est découpée selon le nombre de tokens de chaque segment ;
    un désalignement lève au lieu d'être imputé."""
    from spiraton.data.semantic_thermo_adapter import sentence_segments
    vec = CachedVectors(lambda t: torch.arange(len(t.split()) * 33, dtype=torch.float64).reshape(-1, 33)
                        + 1000 * len(t.split()))
    rows = sentence_segments(["un deux", "trois", "quatre cinq six"], vec)
    assert [r.size(0) for r in rows] == [2, 1, 3]
    full = vec("un deux trois quatre cinq six")
    assert torch.equal(torch.cat(rows), full)
    bad = CachedVectors(lambda t: torch.zeros(len(t.split()) + (1 if len(t.split()) > 3 else 0), 33))
    with pytest.raises(ValueError):
        sentence_segments(["un deux", "trois quatre"], bad)


def test_sentence_mode_recut_keeps_vectors_moves_boundaries():
    vec = CachedVectors(lambda t: torch.ones(len(t.split()), 33))
    rows = torch.arange(9 * 33, dtype=torch.float64).reshape(9, 33)
    segs = tuple(SegmentRecord(n, c, "", r) for n, c, r in
                 (("SEG_A", "DX", rows[:2]), ("SEG_B", "DX", rows[2:5]), ("SEG_A_PRIME", "LV", rows[5:])))
    c = CycleRecord("x", 0, "ADD", "F0", segs)
    r = recut_cycle(c, vec, random.Random(5), tokenize="sentence")
    assert torch.equal(torch.cat([s.vectors for s in r.segments]), rows)
    assert sorted(s.vectors.size(0) for s in r.segments) == [2, 3, 4]


def test_pca_normalizer_all_columns_deterministic():
    """R3 : tranche 'all' + ACP ajustée sur la calibration, sans blanchiment."""
    g = torch.Generator().manual_seed(6)
    rows = torch.randn(400, 20, generator=g, dtype=torch.float64) * torch.linspace(3, 0.1, 20, dtype=torch.float64)
    a = SliceNormalizer(rows, "all", normalize=False, pca_dim=5)
    b = SliceNormalizer(rows, "all", normalize=False, pca_dim=5)
    z = a(rows)
    assert z.shape == (400, 5) and a.out_dim == 5
    assert torch.equal(z, b(rows))
    assert torch.allclose(z.mean(0), torch.zeros(5, dtype=torch.float64), atol=1e-10)
    v = z.var(0)
    assert torch.all(v[:-1] >= v[1:])                       # variances décroissantes, non blanchies
    assert 0.5 < a.explained <= 1.0


def test_word_level_sentence_split_without_reencoding():
    from spiraton.data.semantic_thermo_adapter import sentence_segments
    calls = []

    def enc(text):
        calls.append(text)
        return torch.arange(len(text.split()) * 4, dtype=torch.float64).reshape(-1, 4)

    vec = CachedVectors(enc, dim=4, word_level=True)
    rows = sentence_segments(["un deux", "trois", "quatre cinq six"], vec)
    assert [r.size(0) for r in rows] == [2, 1, 3]
    assert calls == ["un deux trois quatre cinq six"]         # une seule passe, phrase entière


def test_contextual_vectors_one_per_word_and_context_dependent():
    """Encodeur réel (hors ligne) : skip propre s'il n'est pas en cache."""
    from spiraton.data.contextual_vectors import ContextualUnavailable, ContextualWordVectors
    try:
        enc = ContextualWordVectors("camembert-base")
    except ContextualUnavailable as exc:
        pytest.skip(str(exc))
    a = enc("Le chat dort sur le canapé.")
    assert a.shape == (6, enc.hidden) and torch.isfinite(a).all()
    assert torch.equal(a, enc("Le chat dort sur le canapé."))                 # déterministe
    b = enc("Le chien mange sur le canapé.")
    assert not torch.allclose(a[5], b[5])                                    # même mot, autre contexte
