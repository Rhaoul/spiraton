"""Tests de la référence phonémique Python (G2P + opérateur-par-phonème).

Deux volets :
  * Volet PUR (toujours exécuté) : déterminisme, formes, invariance par
    permutation du vote phonémique (le cœur testable de P0 niveau-1), forme de
    la pondération informée (P1).
  * Volet PARITÉ (skip si .so absent) : la réf. Python reproduit les
    ``oper_scores`` normalisés du socle en mode heuristique (garde-fou de
    non-dérive ABI). Tolérance = boost ±0.3 d'analyser_signature_mot, NON
    reproduit côté Python (documenté).
"""
from __future__ import annotations

import random

import pytest

from spiraton.data.phoneme_ref import (
    OPS,
    PHONEME_OP,
    active_op_counts,
    g2p_heuristic,
    phoneme_ops_for_word,
)
from spiraton.data.aba_informed import dominant_operator_informed
from spiraton.data.tokenizer_bridge import is_available, load_native_tokenizer


# --- Volet pur ---------------------------------------------------------------

def test_g2p_deterministic() -> None:
    for w in ["chat", "silence", "montagne", "joie"]:
        assert g2p_heuristic(w) == g2p_heuristic(w)


def test_g2p_known_words() -> None:
    # 'chat' -> ʃ a t (digramme ch, finale t conservée car len<=3 octets? 'chat'=4)
    assert g2p_heuristic("chat") == ["ʃ", "a", "t"]
    # 'mort' -> m o ʁ t
    assert g2p_heuristic("mort") == ["m", "o", "ʁ", "t"]
    # digramme 'ou' -> u
    assert g2p_heuristic("loup") == ["l", "u", "p"]


def test_phoneme_ops_maps_each_phoneme() -> None:
    ops = phoneme_ops_for_word("mort")  # m=MUL o=ADD ʁ=MUL t=SUB
    assert ops == ["MUL", "ADD", "MUL", "SUB"]


def test_active_counts_excludes_pure() -> None:
    # 'l' et 'j'/'w' sont PURE : ne comptent pas dans le vote actif.
    ops = ["ADD", "PURE", "SUB", "PURE", "MUL"]
    assert active_op_counts(ops) == [1, 1, 1, 0]  # ADD,SUB,MUL,DIV


def test_vote_is_permutation_invariant() -> None:
    """Cœur de P0 niveau-1 : permuter l'ordre des phonèmes ne change pas le vote."""
    rng = random.Random(0)
    for w in ["soustraire", "montagne", "resonance", "ramification"]:
        ops = phoneme_ops_for_word(w)
        base = active_op_counts(ops)
        for _ in range(10):
            shuffled = ops[:]
            rng.shuffle(shuffled)
            assert active_op_counts(shuffled) == base


def test_informed_uniform_equals_presence() -> None:
    """logp_table=None doit reproduire le vote par présence (généralisation)."""
    for sentence in [["mort", "vie"], ["la", "joie", "monte"], ["roc", "bloc"]]:
        # poids uniforme = argmax des comptes actifs
        counts = [0, 0, 0, 0]
        for w in sentence:
            c = active_op_counts(phoneme_ops_for_word(w))
            for i in range(4):
                counts[i] += c[i]
        expected = OPS[max(range(4), key=lambda i: counts[i])]
        assert dominant_operator_informed(sentence, None) == expected


def test_informed_weighting_can_shift_argmax() -> None:
    """Une table p qui sur-pondère un opérateur rare peut déplacer l'argmax.

    Sanity : la fonctionnelle agit bien (sans rien prouver sur l'accord).
    """
    # mot avec ADD majoritaire en présence ; on sur-pondère DIV (s) lourdement.
    words = ["sa"]  # s=DIV, a=ADD  -> présence: ADD et DIV à 1 chacun, argmax=ADD (idx0)
    assert dominant_operator_informed(words, None) == "ADD"
    heavy_div = {ph: 1.0 for ph in PHONEME_OP}
    heavy_div["s"] = 100.0
    assert dominant_operator_informed(words, heavy_div) == "DIV"


# --- Volet parité (skip si .so absent) ---------------------------------------

@pytest.mark.skipif(not is_available(), reason="tokenizer natif indisponible")
def test_parity_with_native_heuristic_mode() -> None:
    """En mode heuristique, la réf. Python reproduit oper_scores du .so.

    Tolérance : 0.31 (le boost ±0.3 d'analyser_signature_mot n'est pas reproduit).
    On exige une parité EXACTE (<1e-4) sur une nette majorité de mots-témoins.

    IMPORTANT : ``set_heuristic_mode`` touche un état GLOBAL du ``.so`` (partagé
    entre tous les tests d'une session). On le restaure à False (mode oracle, le
    défaut) en sortie, sinon on pollue les tests de parité 33D suivants.
    """
    tok = load_native_tokenizer()
    tok.set_heuristic_mode(True)
    try:
        words = ["chat", "silence", "mort", "joie", "vie", "roc", "bloc",
                 "revele", "resiste", "clair", "matin", "eclate", "dort",
                 "montagne", "riviere", "soleil", "ombre", "lumiere"]
        exact = 0
        for w in words:
            toks = tok.tokenize(w)
            if not toks:
                continue
            so = list(toks[0]["oper_scores"])
            c = active_op_counts(phoneme_ops_for_word(w))
            tot = sum(c)
            py = [x / tot if tot else 0.0 for x in c]
            maxdiff = max(abs(a - b) for a, b in zip(so, py))
            assert maxdiff < 0.32, f"{w}: so={so} py={py} diff={maxdiff}"
            if maxdiff < 1e-4:
                exact += 1
        # Au moins 70 % des mots-témoins en parité exacte (autres : oracle/boost).
        assert exact >= int(0.7 * len(words))
    finally:
        tok.set_heuristic_mode(False)  # restaurer le mode oracle (défaut global)
