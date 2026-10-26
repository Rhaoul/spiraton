"""Référence Python LISIBLE du G2P heuristique + opérateur-par-phonème du socle C.

But : exposer la **séquence d'opérateurs phonémiques** d'un mot, granularité que
le ``.so`` n'expose PAS dans ``SpiratonToken`` (seuls les ``oper_scores`` agrégés
et normalisés y sont). Cette granularité est nécessaire pour :

  * P0 niveau-1 (« le plus propre ») : permuter les phonèmes POST-G2P, pré-scoring
    — impossible côté C (aucun point d'entrée IPA), atteignable ici en Python pur.
  * P1 : pondérer chaque vote phonémique par −log p(phonème).

C'est une **référence**, pas une source de vérité concurrente : sa fidélité au
socle est VÉRIFIÉE par parité (``phoneme_ops_for_word`` doit reproduire les
``oper_scores`` normalisés du ``.so`` — cf. tests). Le G2P heuristique transcrit
ici correspond au mode ``set_heuristic_mode(1)`` du C (PAS l'oracle dictionnaire :
le dictionnaire IPA précis n'est pas réimplémenté — voir ``HEURISTIC_ONLY``).

Transcrit fidèlement depuis (lus, NON modifiés) :
  * ``Tokenizer/csrc/tokenizer.c`` : ``map_graphie_to_ipa`` (G2P), règles de
    finales muettes, syllabification.
  * ``Tokenizer/csrc/phonemes_fr.c`` : table 36 phonèmes → opérateur.

Limites assumées :
  * Le G2P heuristique est approximatif (le C le dit lui-même : ``pseudo_g2p``,
    repli). En mode oracle (défaut du ``.so``), un mot du dictionnaire reçoit une
    IPA précise non reproduite ici. La parité Python↔C n'est donc EXACTE que sous
    ``set_heuristic_mode(1)`` ; en mode oracle, on mesure le taux de parité et on
    le rapporte (jamais on ne le force).
  * ``analyser_signature_mot`` (boost ±0.3) modifie ``oper_scores`` au niveau MOT
    selon l'orientation : il n'a PAS d'équivalent phonémique et n'est donc pas
    reproduit dans la séquence d'opérateurs (mais documenté comme source d'écart).

Licence : GPL-3.0-or-later.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# --- Opérateurs (mêmes indices que dims 0-3 du 33D / enum Operateur) ----------
OPS: Tuple[str, ...] = ("ADD", "SUB", "MUL", "DIV")
OP_PURE = "PURE"  # semi-voyelles, [l] : pas de balise ABA, exclu du vote actif

# --- Table phonème IPA → opérateur (transcrite de phonemes_fr.c) --------------
# Voyelles orales + nasales + schwa = ADD sauf nasales = MUL ; occlusives = SUB ;
# nasales consonnes + [ʁ] = MUL ; fricatives = DIV ; semi-voyelles + [l] = PURE.
PHONEME_OP = {
    # Voyelles orales -> ADD
    "a": "ADD", "ɑ": "ADD", "o": "ADD", "ɔ": "ADD", "ɛ": "ADD", "e": "ADD",
    "i": "ADD", "y": "ADD", "u": "ADD", "ø": "ADD", "œ": "ADD", "ə": "ADD",
    # Voyelles nasales -> MUL (résonance nasale, correction Phase 2)
    "ɑ̃": "MUL", "ɔ̃": "MUL", "ɛ̃": "MUL", "œ̃": "MUL",
    # Semi-voyelles -> PURE
    "j": "PURE", "w": "PURE", "ɥ": "PURE",
    # Occlusives -> SUB
    "p": "SUB", "t": "SUB", "k": "SUB", "b": "SUB", "d": "SUB", "g": "SUB",
    # Nasales consonnes -> MUL
    "m": "MUL", "n": "MUL", "ɲ": "MUL",
    # Fricatives -> DIV
    "f": "DIV", "s": "DIV", "ʃ": "DIV", "v": "DIV", "z": "DIV", "ʒ": "DIV",
    # Liquides : [l] PURE, [ʁ] MUL
    "l": "PURE", "ʁ": "MUL",
}

_VOWEL_CHARS = set("aeiouy")


def _is_vowel_char(c: str) -> bool:
    return c.lower() in _VOWEL_CHARS


def g2p_heuristic(word: str) -> List[str]:
    """G2P heuristique : graphèmes → liste de phonèmes IPA.

    Transcrit ``map_graphie_to_ipa`` + la boucle de ``pseudo_g2p`` (mode
    heuristique). Octet-orienté sur l'UTF-8, comme le C (``ptr`` byte-par-byte).
    Les graphèmes non mappés (ex. 'h', inconnus) produisent un phonème nul (skip).
    """
    b = word.encode("utf-8")
    n = len(b)
    out: List[str] = []
    i = 0
    while i < n:
        remaining = n - i
        # Règle 1 : finales muettes (mot > 3 octets graphiques bruts).
        # Le C teste len(word) en octets ; on reste fidèle.
        if len(b) > 3:
            if remaining == 1 and b[i:i + 1] == b"e":
                break
            if remaining == 2 and b[i:i + 2] == b"es":
                break
            if remaining == 3 and b[i:i + 3] == b"ent":
                break
        ipa, consumed = _map_graphie(b, i, n)
        if ipa is not None:
            out.append(ipa)
        i += consumed
    return out


def _starts(b: bytes, i: int, prefix: bytes) -> bool:
    return b[i:i + len(prefix)] == prefix


def _map_graphie(b: bytes, i: int, n: int) -> Tuple[Optional[str], int]:
    """Transcription fidèle de map_graphie_to_ipa (retourne (ipa|None, n_octets))."""
    # 3-char graphs
    if _starts(b, i, b"eau"):
        return "o", 3
    if _starts(b, i, b"ain"):
        return "ɛ̃", 3
    if _starts(b, i, b"ein"):
        return "ɛ̃", 3
    # next_is_vowel : (ptr[1] != '\0') && is_vowel_char(ptr[2])
    next_is_vowel = (i + 1 < n) and (i + 2 < n) and _is_vowel_char(chr(b[i + 2]))
    if _starts(b, i, b"ou"):
        return "u", 2
    if _starts(b, i, b"au"):
        return "o", 2
    if _starts(b, i, b"an") and not next_is_vowel:
        return "ɑ̃", 2
    if _starts(b, i, b"en") and not next_is_vowel:
        return "ɑ̃", 2
    if _starts(b, i, b"on") and not next_is_vowel:
        return "ɔ̃", 2
    if _starts(b, i, b"in") and not next_is_vowel:
        return "ɛ̃", 2
    if _starts(b, i, b"un") and not next_is_vowel:
        return "œ̃", 2
    if _starts(b, i, b"ch"):
        return "ʃ", 2
    if _starts(b, i, b"ph"):
        return "f", 2
    if _starts(b, i, b"gn"):
        return "ɲ", 2
    if _starts(b, i, b"qu"):
        return "k", 2
    if _starts(b, i, b"gu"):
        return "g", 2
    # Multi-byte accents (UTF-8 C3 xx)
    accents = {
        b"\xc3\xa9": "e", b"\xc3\xa8": "ɛ", b"\xc3\xaa": "ɛ", b"\xc3\xa0": "a",
        b"\xc3\xa2": "a", b"\xc3\xb9": "u", b"\xc3\xbb": "y", b"\xc3\xb4": "o",
        b"\xc3\xae": "i", b"\xc3\xa7": "s",
    }
    for pref, ipa in accents.items():
        if _starts(b, i, pref):
            return ipa, 2
    # Single char
    c = chr(b[i]).lower()
    single = {
        "a": "a", "e": "ə", "i": "i", "o": "o", "u": "y", "y": "i",
        "p": "p", "t": "t", "k": "k", "b": "b", "d": "d", "g": "g",
        "f": "f", "s": "s", "v": "v", "z": "z", "m": "m", "n": "n",
        "l": "l", "r": "ʁ", "j": "ʒ", "c": "k",
    }
    ipa = single.get(c)  # 'h' et inconnus -> None
    # Nombre d'octets consommés : copie la logique C (détection start-byte UTF-8).
    byte0 = b[i]
    if (byte0 & 0xC0) == 0xC0:
        if 0xC0 <= byte0 <= 0xDF:
            consumed = 2
        elif 0xE0 <= byte0 <= 0xEF:
            consumed = 3
        elif 0xF0 <= byte0 <= 0xF7:
            consumed = 4
        else:
            consumed = 1
    else:
        consumed = 1
    return ipa, consumed


def phoneme_ops_for_word(word: str) -> List[str]:
    """Liste des opérateurs phonémiques d'un mot (mode heuristique), PURE inclus.

    C'est la séquence sur laquelle P0 niveau-1 permute et P1 pondère.
    """
    return [PHONEME_OP.get(p, OP_PURE) for p in g2p_heuristic(word)]


def active_op_counts(ops: List[str]) -> List[int]:
    """Compte par opérateur ACTIF (ADD/SUB/MUL/DIV), PURE exclu — comme le C."""
    counts = [0, 0, 0, 0]
    for op in ops:
        if op in PHONEME_OP.values() and op != OP_PURE:
            pass
        if op in ("ADD", "SUB", "MUL", "DIV"):
            counts[OPS.index(op)] += 1
    return counts
