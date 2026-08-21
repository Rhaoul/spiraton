"""Tests de l'émetteur ABA (côté tokenizer) + invariant aller-retour.

L'émetteur vit dans le dépôt tokenizer (`spiraton_tokenizer.aba_emitter`) ; il
est importé ici via le pont. La logique d'émission/round-trip est testée avec un
tokenizer factice (sans `.so`), puis la physique réelle avec le natif (skip
propre si absent). Le cross-check se fait avec le **parseur de référence** local
(`spiraton.data.aba`) : le `.so` n'est jamais la seule source de vérité.
"""
from __future__ import annotations

import pytest

from spiraton.data.aba import parse_aba_line, OPERATORS
from spiraton.data.tokenizer_bridge import (
    import_aba_emitter,
    is_available,
    load_native_tokenizer,
    TokenizerUnavailable,
)

try:
    _emitter = import_aba_emitter()
except TokenizerUnavailable:
    _emitter = None

pytestmark = pytest.mark.skipif(_emitter is None, reason="package tokenizer introuvable")


# --- Tokenizer factice : contrôle total des tokens, sans .so -----------------


def _tok(text, scores, *, role=-1, orientation=0, operator=0):
    return {"text": text, "oper_scores": list(scores), "role": role,
            "orientation": orientation, "operator": operator}


class _StubTok:
    def __init__(self, toks):
        self._toks = toks

    def tokenize(self, text, max_tokens=128):
        return self._toks


# --- Logique d'émission (toujours exécutée) ----------------------------------


def test_dominant_operator_argmax_of_sums() -> None:
    toks = [
        _tok("a", [0.2, 0.0, 0.7, 0.1]),
        _tok("b", [0.1, 0.0, 0.8, 0.1]),
    ]  # MUL domine la somme
    assert _emitter.dominant_operator(toks) == "MUL"


def test_emit_splits_segments_by_role() -> None:
    toks = [
        _tok("la", [1, 0, 0, 0], role=0),
        _tok("joie", [1, 0, 0, 0], role=1),
        _tok("monte", [1, 0, 0, 0], role=-1),
        _tok("haut", [1, 0, 0, 0], role=2),
    ]
    cyc = _emitter.emit_aba("x", _StubTok(toks))
    assert cyc.op == "ADD"
    assert cyc.seg_a == "la"
    assert cyc.seg_b == "joie monte"
    assert cyc.seg_a_prime == "haut"


def test_emitted_line_roundtrips_through_reference_parser() -> None:
    """Invariant GRAMMAIRE : émettre -> parser de référence -> même triplet."""
    toks = [
        _tok("le", [1, 0, 0, 0], role=0),
        _tok("sens", [1, 0, 0, 0], role=1),
        _tok("revient", [1, 0, 0, 0], role=2),
    ]
    cyc = _emitter.emit_aba("x", _StubTok(toks))
    parsed = parse_aba_line(cyc.line)
    assert parsed.op == cyc.op == "ADD"
    assert parsed.is_closure                      # A,B en DX/OUT ; A' en LV/IN
    assert parsed.seg_a.text == cyc.seg_a
    assert parsed.seg_b.text == cyc.seg_b
    assert parsed.seg_a_prime.text == cyc.seg_a_prime


def test_empty_returns_none() -> None:
    assert _emitter.emit_aba("", _StubTok([])) is None
    assert _emitter.emit_aba_line("", _StubTok([])) is None


def test_single_token_still_parseable() -> None:
    cyc = _emitter.emit_aba("x", _StubTok([_tok("seul", [0, 1, 0, 0], role=0)]))
    parsed = parse_aba_line(cyc.line)            # ne doit pas lever
    assert parsed.op == "SUB"
    assert cyc.seg_a == "seul"
    assert parsed.seg_a.text == "seul"


def test_op_is_always_valid_tag() -> None:
    # même tout-à-zéro produit un opérateur de la grammaire (repli ADD).
    cyc = _emitter.emit_aba("x", _StubTok([_tok("mmm", [0, 0, 0, 0], role=0)]))
    assert cyc.op in OPERATORS


# --- GARDE-FOU TOUR 8 : le défaut canon de l'émetteur reste bit-identique -----
# Les pistes P0/P1 vivent en diagnostic Python (data/phoneme_ref, data/aba_informed,
# examples/measure_*). Le défaut dominant_operator/emit_aba ne doit JAMAIS bouger.

def test_default_dominant_operator_is_argmax_of_summed_scores_exact() -> None:
    """Fige la sémantique exacte du défaut : argmax(Σ scores), aucune pondération.

    Si une future variante 'informée' contaminait le défaut, ce test casse.
    """
    # ADD=0.6, SUB=0.5, MUL=1.1, DIV=0.8  -> MUL domine la SOMME (pas la présence).
    toks = [
        _tok("u", [0.3, 0.2, 0.5, 0.4]),
        _tok("v", [0.3, 0.3, 0.6, 0.4]),
    ]
    # somme = [0.6, 0.5, 1.1, 0.8] -> argmax index 2 = MUL
    assert _emitter.dominant_operator(toks) == "MUL"
    assert _emitter._sum_scores(toks) == pytest.approx([0.6, 0.5, 1.1, 0.8])


def test_default_dominant_operator_is_permutation_invariant() -> None:
    """Le défaut est commutatif : permuter les tokens ne change pas le dominant.

    C'est exactement ce que P0 vérifie comme baseline d'ordre (la somme commute).
    """
    toks = [
        _tok("a", [0.2, 0.0, 0.7, 0.1]),
        _tok("b", [0.1, 0.5, 0.1, 0.3]),
        _tok("c", [0.4, 0.1, 0.2, 0.3]),
    ]
    base = _emitter.dominant_operator(toks)
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        assert _emitter.dominant_operator([toks[i] for i in perm]) == base


def test_informed_variant_does_not_touch_default_module() -> None:
    """La variante informée vit dans data/aba_informed, PAS dans aba_emitter.

    Le défaut ignore toute notion de pondération : il n'expose aucun argument logp.
    """
    import inspect

    sig = inspect.signature(_emitter.dominant_operator)
    assert list(sig.parameters) == ["tokens"]  # pas de logp_table ici


# --- Physique réelle : nécessite le tokenizer natif --------------------------

native = is_available()


@pytest.mark.skipif(not native, reason="tokenizer natif (.so/.dll) indisponible")
def test_native_roundtrip_on_real_sentences() -> None:
    tok = load_native_tokenizer()
    for s in ["Le chat dort sur le canapé.",
              "La joie éclate dans le matin clair.",
              "Soustraire révèle ce qui résiste."]:
        cyc = _emitter.emit_aba(s, tok)
        assert cyc is not None
        parsed = parse_aba_line(cyc.line)
        assert parsed.is_closure
        assert parsed.op == cyc.op
        assert parsed.op in OPERATORS


@pytest.mark.skipif(not native, reason="tokenizer natif (.so/.dll) indisponible")
def test_native_agreement_is_a_measured_fraction() -> None:
    """L'accord opérateur est une MESURE dans [0,1] — on ne vise aucune valeur."""
    from pathlib import Path
    from spiraton.data.aba import iter_aba_cycles

    root = Path(__file__).resolve().parents[2]
    dataset = root / "dataset_aba.txt"
    if not dataset.is_file():
        pytest.skip("dataset_aba.txt absent")

    tok = load_native_tokenizer()
    total = agree = 0
    for cyc in iter_aba_cycles(str(dataset)):
        if total >= 100:
            break
        sentence = " ".join(t for t in (cyc.seg_a.text, cyc.seg_b.text,
                                        cyc.seg_a_prime.text) if t)
        em = _emitter.emit_aba(sentence, tok)
        if em is None:
            continue
        total += 1
        if em.op == cyc.op:
            agree += 1
    assert total > 0
    rate = agree / total
    assert 0.0 <= rate <= 1.0                     # mesure, pas cible
