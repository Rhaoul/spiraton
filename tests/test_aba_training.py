import torch
import pytest

from spiraton.data.featurizers import HashingFeaturizer
from spiraton.data.loader import (
    AbaTriples,
    load_aba_triples,
    triples_from_cycles,
    iter_batches,
)
from spiraton.data.aba import parse_aba_line
from spiraton.training import (
    alpha_omega_loss,
    copy_rate,
    train_aba,
    make_aba_predictor,
)


# --- Featurizer -------------------------------------------------------------

def test_hashing_featurizer_deterministic_and_normalized() -> None:
    f = HashingFeaturizer(dim=16)
    v1 = f("spirale")
    v2 = f("spirale")
    assert v1.shape == (16,)
    assert torch.equal(v1, v2)                       # déterministe au bit près
    assert abs(float(v1.norm()) - 1.0) < 1e-5        # L2-normalisé
    assert f("") .norm() == 0.0                       # vide => zéro
    assert f.is_fallback is True                      # étiqueté comme substitut
    # Textes différents => vecteurs différents (en général).
    assert not torch.equal(f("addition"), f("division"))


# --- Loader -----------------------------------------------------------------

_LINES = [
    "<SEG_A> <ADD><DX><OUT><ALPHA> a un </SEG_A> <SEG_B> <ADD><DX><OUT><OMEGA> b deux </SEG_B> <SEG_A_PRIME> <ADD><LV><IN><A_PRIME> c trois </SEG_A_PRIME> <EOL>",
    "<SEG_A> <MUL><DX><OUT><ALPHA> x </SEG_A> <SEG_B> <MUL><DX><OUT><OMEGA> y </SEG_B> <SEG_A_PRIME> <MUL><LV><IN><A_PRIME> z </SEG_A_PRIME> <EOL>",
]


def test_triples_from_cycles_shapes() -> None:
    cycles = [parse_aba_line(l) for l in _LINES]
    feat = HashingFeaturizer(dim=8)
    tr = triples_from_cycles(cycles, feat)
    assert len(tr) == 2
    assert tr.dim == 8
    assert tr.a.shape == tr.b.shape == tr.a_prime.shape == (2, 8)
    assert tr.ops == ["ADD", "MUL"]
    assert tr.featurizer_is_fallback is True


def test_load_aba_triples_from_file(tmp_path) -> None:
    p = tmp_path / "mini_aba.txt"
    p.write_text("\n".join(_LINES) + "\n<EOS>\n", encoding="utf-8")
    tr = load_aba_triples(str(p), HashingFeaturizer(dim=12))
    assert len(tr) == 2            # le terminateur <EOS> est ignoré
    assert tr.dim == 12


def test_iter_batches_covers_all_once() -> None:
    a = torch.randn(5, 4)
    tr = AbaTriples(a=a, b=a.clone(), a_prime=a.clone(), ops=["ADD"] * 5, featurizer_is_fallback=True)
    seen = 0
    for ba, bb, bap in iter_batches(tr, batch_size=2, shuffle_seed=0):
        seen += ba.size(0)
    assert seen == 5


# --- Perte alpha-oméga ------------------------------------------------------

def test_loss_prefers_aligned_at_target_over_copy() -> None:
    torch.manual_seed(0)
    a = torch.randn(32, 10)
    target = 0.3

    copy = a.clone()                          # rel_dist = 0  (répétition)
    aligned_at_target = a * (1.0 + target)    # même direction, distance = target
    far = -a * 3.0                             # anti-aligné et lointain

    l_copy = alpha_omega_loss(copy, a, target_dist=target).loss
    l_ideal = alpha_omega_loss(aligned_at_target, a, target_dist=target).loss
    l_far = alpha_omega_loss(far, a, target_dist=target).loss

    # Le retour aligné-à-la-bonne-distance est le meilleur ; la copie est punie ;
    # se perdre est le pire.
    assert float(l_ideal) < float(l_copy)
    assert float(l_copy) < float(l_far)
    assert float(l_ideal) < 1e-4              # cas idéal ≈ 0


def test_copy_rate_detects_repetition() -> None:
    a = torch.randn(10, 6)
    assert copy_rate(a.clone(), a) == 1.0
    assert copy_rate(a + 1.0, a) == 0.0


def test_loss_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        alpha_omega_loss(torch.randn(3, 5), torch.randn(3, 6))


# --- Boucle d'entraînement (signal apprenable) ------------------------------

def test_training_reduces_loss() -> None:
    """La perte alpha-oméga est un signal réellement apprenable."""
    torch.manual_seed(1)
    n, d = 64, 12
    a = torch.randn(n, d)
    b = torch.randn(n, d)
    a_prime = torch.randn(n, d)  # non utilisé par la perte (cible = A), mais réaliste
    tr = AbaTriples(a=a, b=b, a_prime=a_prime, ops=["ADD"] * n, featurizer_is_fallback=True)

    model = make_aba_predictor(d)
    report = train_aba(model, tr, epochs=120, lr=1e-2, target_dist=0.3)

    assert report.improved
    assert report.loss_history[-1] < 0.5 * report.loss_history[0]
    # Le modèle n'a pas convergé vers la copie (mode dégénéré).
    assert report.final_metrics["copy_rate"] < 0.5


def test_training_deterministic_given_seed() -> None:
    torch.manual_seed(2)
    n, d = 32, 8
    tr = AbaTriples(
        a=torch.randn(n, d), b=torch.randn(n, d), a_prime=torch.randn(n, d),
        ops=["MUL"] * n, featurizer_is_fallback=True,
    )

    torch.manual_seed(7)
    r1 = train_aba(make_aba_predictor(d), tr, epochs=20)
    torch.manual_seed(7)
    r2 = train_aba(make_aba_predictor(d), tr, epochs=20)
    assert r1.loss_history == r2.loss_history
