"""Tests du PhonemeFeaturizer (chantier 7).

Deux familles :
  - **Logique d'agrégation** : testée avec un tokenizer factice injecté, donc
    SANS le .so — elle tourne partout (y compris CI sans compilateur C).
  - **Physique réelle** : testée avec le tokenizer natif, skippée proprement
    s'il est indisponible (même convention que test_vector33d_bridge).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from spiraton.data import vector33d
from spiraton.data.featurizers import Featurizer, HashingFeaturizer, PhonemeFeaturizer
from spiraton.data.tokenizer_bridge import is_available
from spiraton.experimental.operator_embedding import OperatorEmbedding


# --- Tokenizer factice : contrôle total sur les (N, 33) renvoyés -------------


class _StubTokenizer:
    """Imite l'interface ``vectors(text, max_tokens) -> (N, 33)`` du natif.

    Déterministe et lisible : permet de vérifier la formule d'agrégation
    exactement, sans dépendre de la lib C.
    """

    def __init__(self, table):
        self._table = table  # dict[str] -> np.ndarray (N, 33)

    def vectors(self, text, max_tokens=128):
        arr = self._table.get(text, np.zeros((0, vector33d.DIM), dtype=np.float32))
        return np.asarray(arr, dtype=np.float32)[:max_tokens]


def _row(fill, *, op=None, chi=None):
    v = np.full((vector33d.DIM,), float(fill), dtype=np.float32)
    if op is not None:
        v[0:4] = op
    if chi is not None:
        v[4:6] = chi
    return v


def _stub_feat(pool="mean"):
    table = {
        # deux tokens nets pour vérifier mean/sum/first
        "duo": np.stack([_row(1.0), _row(3.0)], axis=0),
        "solo": _row(2.0)[None, :],
        "vide": np.zeros((0, vector33d.DIM), dtype=np.float32),
    }
    return PhonemeFeaturizer(_StubTokenizer(table), pool=pool)


def _no_nan_inf(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


# --- Quatuor canonique (adapté à un featurizer) ------------------------------


def test_satisfies_protocol_and_is_not_fallback() -> None:
    feat = _stub_feat()
    assert isinstance(feat, Featurizer)        # dim / is_fallback / __call__
    assert feat.is_fallback is False           # contraste avec HashingFeaturizer
    assert HashingFeaturizer().is_fallback is True
    assert feat.dim == vector33d.DIM == 33


def test_shapes_single_batch_sequence() -> None:
    feat = _stub_feat()
    assert feat("duo").shape == (33,)
    assert feat.sequence("duo").shape == (2, 33)
    assert feat.signature("duo").shape == (15,)   # dims 8-22
    assert feat.batch(["duo", "solo"]).shape == (2, 33)
    assert feat.batch([]).shape == (0, 33)


def test_finiteness() -> None:
    feat = _stub_feat()
    for txt in ("duo", "solo", "vide"):
        assert _no_nan_inf(feat(txt))


def test_exact_pooling_formula() -> None:
    """Formule exacte sous entrées forcées (mean/sum/first)."""
    assert torch.allclose(_stub_feat("mean")("duo"), torch.full((33,), 2.0))
    assert torch.allclose(_stub_feat("sum")("duo"), torch.full((33,), 4.0))
    assert torch.allclose(_stub_feat("first")("duo"), torch.full((33,), 1.0))


def test_empty_text_returns_zeros() -> None:
    for pool in ("mean", "sum", "first"):
        assert torch.allclose(_stub_feat(pool)("vide"), torch.zeros(33))


def test_signature_is_phoneme_band() -> None:
    """signature(text) == dims 8-22 du vecteur agrégé."""
    feat = _stub_feat("mean")
    full = feat("duo")
    assert torch.allclose(feat.signature("duo"), full[vector33d.PHONEME_SIG])


def test_invalid_pool_raises() -> None:
    with pytest.raises(ValueError):
        PhonemeFeaturizer(_StubTokenizer({}), pool="bogus")


def test_bad_shape_from_tokenizer_raises() -> None:
    class _BadTok:
        def vectors(self, text, max_tokens=128):
            return np.zeros((2, 7), dtype=np.float32)  # mauvaise largeur

    with pytest.raises(ValueError):
        PhonemeFeaturizer(_BadTok())("x")


# --- Intégration : les traits réels routés par OperatorEmbedding -------------


def test_feeds_operator_embedding_and_gradients_flow() -> None:
    """Chaîne chantier 7 : featurizer -> OperatorEmbedding(source='phoneme').

    Le featurizer n'est pas différentiable (extracteur de traits, comme une
    entrée) ; on vérifie que le gradient circule dans l'embedding qu'il nourrit.
    """
    torch.manual_seed(0)
    # token ADD/DX net, traits non nuls dans la bande phonémique
    row = _row(0.0, op=[5.0, 0.0, 0.0, 0.0], chi=[5.0, 0.0])
    row[vector33d.PHONEME_SIG] = 0.4
    feat = PhonemeFeaturizer(_StubTokenizer({"w": row[None, :]}))

    emb = OperatorEmbedding(embed_dim=8, source="phoneme")
    x = feat("w").unsqueeze(0)                 # (1, 33)
    y = emb(x)
    assert y.shape == (1, 8)
    assert _no_nan_inf(y)
    y.sum().backward()
    assert emb.W.grad is not None and float(emb.W.grad.abs().sum()) > 0.0


def test_sequence_feeds_a_real_cell_in_order() -> None:
    """Chaîne complète du chantier 7 : sequence -> OperatorEmbedding -> cellule.

    « Exploiter les traits 8-22 comme canaux d'entrée des cellules » : les
    vecteurs 33D réels (ici simulés par un stub multi-tokens) sont projetés
    par OperatorEmbedding puis consommés **token par token** par une
    SpiratonCell — l'ordre des phonèmes/tokens entre dans la dynamique, pas
    seulement un profil moyen. On vérifie formes, finitude et flux de gradient.
    """
    from spiraton.core.cell import SpiratonCell

    torch.manual_seed(0)
    t0 = _row(0.0, op=[5.0, 0.0, 0.0, 0.0], chi=[5.0, 0.0]); t0[vector33d.PHONEME_SIG] = 0.4
    t1 = _row(0.0, op=[0.0, 0.0, 5.0, 0.0], chi=[0.0, 5.0]); t1[vector33d.PHONEME_SIG] = -0.2
    feat = PhonemeFeaturizer(_StubTokenizer({"phrase": np.stack([t0, t1], axis=0)}))

    embed_dim = 8
    emb = OperatorEmbedding(embed_dim=embed_dim, source="phoneme")
    cell = SpiratonCell(input_size=embed_dim)

    seq = feat.sequence("phrase")              # (N, 33) — ordre préservé
    assert seq.shape == (2, vector33d.DIM)
    tok_emb = emb(seq)                         # (N, embed_dim)
    assert tok_emb.shape == (2, embed_dim)
    y = cell(tok_emb)                          # (N,) — une activation par token
    assert y.shape == (2,)
    assert _no_nan_inf(y)

    y.sum().backward()
    assert float(emb.W.grad.abs().sum()) > 0.0
    assert float(cell.w_add.grad.abs().sum()) > 0.0


# --- Physique réelle : nécessite le tokenizer natif --------------------------

native = is_available()


@pytest.mark.skipif(not native, reason="tokenizer natif (.so/.dll) indisponible")
def test_native_determinism() -> None:
    """Même texte -> même vecteur agrégé, au bit près (contrat déterminisme)."""
    feat = PhonemeFeaturizer()
    assert torch.equal(feat("spirale"), feat("spirale"))
    assert torch.equal(feat.signature("Joie"), feat.signature("Joie"))


@pytest.mark.skipif(not native, reason="tokenizer natif (.so/.dll) indisponible")
def test_native_physics_distinguishes_words() -> None:
    """La physique articulatoire sépare Joie et Mort dans la bande 8-22.

    C'est le sens du chantier 7 : les traits ne sont pas du hachage arbitraire
    mais portent une distinction mesurée. (Contraste documenté Joie/Mort.)
    """
    feat = PhonemeFeaturizer()
    sig_joie = feat.signature("Joie")
    sig_mort = feat.signature("Mort")
    assert sig_joie.shape == (15,)
    assert not torch.allclose(sig_joie, sig_mort, atol=1e-3)


@pytest.mark.skipif(not native, reason="tokenizer natif (.so/.dll) indisponible")
def test_native_training_integration() -> None:
    """Chantier 5 rejoué sur physique réelle : loader -> train_aba tourne.

    Cycles construits en mémoire (pas de fichier externe). On vérifie que la
    boucle alpha-oméga s'exécute sur les traits réels et produit une perte finie
    et apprenable — sans rien viser de chiffré.
    """
    import math

    from spiraton.data.aba import parse_aba_line
    from spiraton.data.loader import triples_from_cycles
    from spiraton.training import train_aba, make_aba_predictor

    lines = [
        "<SEG_A> <ADD><DX><OUT><ALPHA> la joie monte </SEG_A> <SEG_B> <ADD><DX><OUT><OMEGA> vers le jour </SEG_B> <SEG_A_PRIME> <ADD><LV><IN><A_PRIME> et revient apaisée.<EOL> </SEG_A_PRIME> <EOL>",
        "<SEG_A> <SUB><DX><OUT><ALPHA> le bruit cesse </SEG_A> <SEG_B> <SUB><DX><OUT><OMEGA> dans le vide </SEG_B> <SEG_A_PRIME> <SUB><LV><IN><A_PRIME> laissant le silence.<EOL> </SEG_A_PRIME> <EOL>",
        "<SEG_A> <MUL><DX><OUT><ALPHA> la forme passe </SEG_A> <SEG_B> <MUL><DX><OUT><OMEGA> dans une autre </SEG_B> <SEG_A_PRIME> <MUL><LV><IN><A_PRIME> et sort orientée.<EOL> </SEG_A_PRIME> <EOL>",
        "<SEG_A> <DIV><DX><OUT><ALPHA> le sens se divise </SEG_A> <SEG_B> <DIV><DX><OUT><OMEGA> en branches fines </SEG_B> <SEG_A_PRIME> <DIV><LV><IN><A_PRIME> puis se ramifie encore.<EOL> </SEG_A_PRIME> <EOL>",
    ]
    cycles = [parse_aba_line(l) for l in lines]
    triples = triples_from_cycles(cycles, PhonemeFeaturizer())
    assert triples.dim == vector33d.DIM
    assert triples.featurizer_is_fallback is False

    torch.manual_seed(0)
    model = make_aba_predictor(triples.dim)
    report = train_aba(model, triples, epochs=10, lr=1e-2, target_dist=0.3)
    assert all(math.isfinite(x) for x in report.loss_history)
    assert report.loss_history[-1] <= report.loss_history[0]


@pytest.mark.skipif(not native, reason="tokenizer natif (.so/.dll) indisponible")
def test_native_sequence_matches_first_pool() -> None:
    """pool='first' == premier token, ET discrimine vraiment de 'mean'.

    Témoin **multi-tokens** (sinon mean=sum=first et le test ne prouve rien).
    """
    phrase = "le chat dort"
    seq = PhonemeFeaturizer().sequence(phrase)
    assert seq.size(0) >= 2, "témoin doit être multi-tokens pour discriminer les pools"
    first = PhonemeFeaturizer(pool="first")(phrase)
    mean = PhonemeFeaturizer(pool="mean")(phrase)
    assert torch.equal(first, seq[0])          # first == premier token
    assert not torch.equal(first, mean)        # ... et n'est PAS la moyenne
