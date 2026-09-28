import pytest

from spiraton.data import vector33d
from spiraton.data.tokenizer_bridge import is_available, NativeTokenizer33D, TokenizerUnavailable


def _vec(op_scores, chi):
    v = [0.0] * vector33d.DIM
    v[0:4] = op_scores
    v[4:6] = chi
    return v


def test_dim_contract() -> None:
    assert vector33d.DIM == 33
    # Les tranches couvrent exactement 0..33 sans trou ni recouvrement.
    spans = [
        vector33d.OP_SCORES, vector33d.CHIRALITY, vector33d.ENERGY_STRUCTURE,
        vector33d.PHONEME_SIG, vector33d.META_PHRASE, vector33d.LEN_DENSITY,
    ]
    covered = []
    for s in spans:
        covered.extend(range(s.start, s.stop))
    assert covered == list(range(vector33d.DIM))


def test_op_from_vector() -> None:
    assert vector33d.op_from_vector(_vec([0.9, 0.1, 0.0, 0.0], [1.0, 0.0])) == "ADD"
    assert vector33d.op_from_vector(_vec([0.1, 0.1, 0.8, 0.0], [0.0, 1.0])) == "MUL"
    assert vector33d.op_from_vector(_vec([0.0, 0.0, 0.0, 1.0], [0.5, 0.5])) == "DIV"


def test_chirality_from_vector() -> None:
    assert vector33d.chirality_from_vector(_vec([1, 0, 0, 0], [0.9, 0.1])) == "DX"
    assert vector33d.chirality_from_vector(_vec([1, 0, 0, 0], [0.1, 0.9])) == "LV"


def test_check_dim_raises_on_wrong_size() -> None:
    with pytest.raises(ValueError):
        vector33d.check_dim([0.0] * 32)


def test_phoneme_signature_length() -> None:
    sig = vector33d.phoneme_signature([0.0] * vector33d.DIM)
    assert len(sig) == 15  # dims 8-22


# --- Pont natif : skip propre si la lib n'est pas compilée -------------------

native_available = is_available()


@pytest.mark.skipif(not native_available, reason="tokenizer natif (.so/.dll) indisponible")
def test_native_determinism() -> None:
    """Si le natif est dispo : déterminisme au bit près (contrat tokenizer)."""
    import numpy as np

    tok = NativeTokenizer33D()
    v1 = tok.vectors("Joie")
    v2 = tok.vectors("Joie")
    assert v1.shape[1] == vector33d.DIM
    assert np.array_equal(v1, v2)


_FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures" / "parity_33d.json"


@pytest.mark.skipif(
    not (native_available and _FIXTURES.is_file()),
    reason="natif indisponible ou fixtures non enregistrées (scripts/record_33d_fixtures.py)",
)
def test_native_parity_against_fixtures() -> None:
    """Détecte toute dérive d'ABI : sortie courante == vecteurs témoins figés."""
    import json
    import numpy as np

    tok = NativeTokenizer33D()
    fixtures = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    for word, expected in fixtures.items():
        got = tok.vectors(word)[0]
        assert len(expected) == vector33d.DIM
        np.testing.assert_allclose(got, np.asarray(expected, dtype=np.float32), atol=1e-5)


def test_unavailable_is_explicit() -> None:
    """Quand le natif manque, l'erreur est actionnable (pas un ImportError nu)."""
    if native_available:
        pytest.skip("natif disponible : rien à vérifier côté indisponibilité")
    with pytest.raises(TokenizerUnavailable):
        NativeTokenizer33D()
