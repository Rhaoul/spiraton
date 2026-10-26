import torch
import pytest

from spiraton.experimental.operator_embedding import OperatorEmbedding
from spiraton.data import vector33d


def _no_nan_inf(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


def _v33(op_scores, chi_scores, feat_fill=0.3):
    v = torch.full((vector33d.DIM,), feat_fill)
    v[0:4] = torch.tensor(op_scores, dtype=torch.float32)
    v[4:6] = torch.tensor(chi_scores, dtype=torch.float32)
    return v


@pytest.mark.parametrize("batch", [1, 4])
def test_shapes_and_finiteness(batch: int) -> None:
    torch.manual_seed(0)
    emb = OperatorEmbedding(embed_dim=12, source="phoneme")
    x = torch.randn(batch, vector33d.DIM)
    y = emb(x)
    assert y.shape == (batch, 12)
    assert _no_nan_inf(y)

    # chemin singleton
    ys = emb(torch.randn(vector33d.DIM))
    assert ys.shape == (12,)

    # chemin séquence (.., N, 33)
    yseq = emb(torch.randn(2, 5, vector33d.DIM))
    assert yseq.shape == (2, 5, 12)


def test_weights_are_convex_combination() -> None:
    emb = OperatorEmbedding(embed_dim=8)
    x = torch.randn(7, vector33d.DIM)
    w = emb.opchi_weights(x)
    assert w.shape == (7, emb.n_sub)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(dim=-1), torch.ones(7), atol=1e-5)


def test_operator_conditions_representation() -> None:
    """Mêmes traits, opérateur différent ⇒ représentation différente.

    Prouve que l'opérateur *conditionne* (module) la sortie : il ne s'agit pas
    d'un simple ajout de features, car seuls les dims 0-5 changent.
    """
    torch.manual_seed(1)
    emb = OperatorEmbedding(embed_dim=10, source="full", temperature=0.5, init_scale=0.5)

    feats_same = 0.4
    v_add = _v33([5.0, 0.0, 0.0, 0.0], [5.0, 0.0], feat_fill=feats_same)
    v_mul = _v33([0.0, 0.0, 5.0, 0.0], [5.0, 0.0], feat_fill=feats_same)
    # Les dims de traits (8-22 ou tout) sont identiques ; seuls op/chi diffèrent.
    out_add = emb(v_add)
    out_mul = emb(v_mul)
    assert not torch.allclose(out_add, out_mul, atol=1e-3)


def test_onehot_routing_selects_single_subspace() -> None:
    """Routage quasi one-hot ⇒ sortie == projection d'UN seul sous-espace."""
    torch.manual_seed(2)
    emb = OperatorEmbedding(embed_dim=6, source="phoneme", temperature=1.0)

    # Logits très piqués => softmax ≈ one-hot (sélection, pas moyenne).
    v = _v33([50.0, 0.0, 0.0, 0.0], [50.0, 0.0], feat_fill=0.2)  # ADD, DX
    k = emb.subspace_index("ADD", "DX")
    assert k == 0

    feats = v[emb.feat_slice]
    expected = feats @ emb.W[k] + emb.b[k]  # (feature) @ (feature,embed) = (embed)
    got = emb(v)
    assert torch.allclose(got, expected, atol=1e-4)


def test_matches_manual_formula_when_params_forced() -> None:
    """Vérifie la combinaison pondérée exacte sous traits/poids contrôlés."""
    torch.manual_seed(3)
    emb = OperatorEmbedding(embed_dim=5, source="full", temperature=1.0)
    x = torch.randn(3, vector33d.DIM)

    feats = x[..., emb.feat_slice]
    w = emb.opchi_weights(x)
    proj = torch.einsum("...f,kfe->...ke", feats, emb.W) + emb.b
    expected = (w.unsqueeze(-1) * proj).sum(dim=-2)
    assert torch.allclose(emb(x), expected, atol=1e-6)


def test_gradients_flow() -> None:
    torch.manual_seed(4)
    emb = OperatorEmbedding(embed_dim=8, source="phoneme")
    x = torch.randn(6, vector33d.DIM, requires_grad=True)
    emb(x).sum().backward()
    assert emb.W.grad is not None
    assert emb.b.grad is not None
    assert float(emb.W.grad.abs().sum()) > 0.0


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        OperatorEmbedding(embed_dim=4, source="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        OperatorEmbedding(embed_dim=4, temperature=0.0)
    emb = OperatorEmbedding(embed_dim=4)
    with pytest.raises(ValueError):
        emb(torch.randn(10))  # mauvaise dimension d'entrée
