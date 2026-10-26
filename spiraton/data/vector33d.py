"""Disposition contractuelle du vecteur 33D (tokenizer v4.1).

Miroir, côté spiraton, du contrat décrit dans CLAUDE_TOKENIZER.md. Ces
constantes sont la *seule* manière autorisée d'indexer un 33D dans ce dépôt :
elles centralisent la connaissance de l'ABI et permettent d'attraper toute
dérive (cf. tests de parité figés). Module sans dépendance lourde (utilise
numpy uniquement pour les helpers, importé paresseusement).

    dims 0-3  : scores opérateurs ADD / SUB / MUL / DIV
    dims 4-5  : orientations dextro / lévo
    dims 6-7  : énergie / structure
    dims 8-22 : signatures phonémiques (impédance, flux) et séquences de spins
    dims 23-30: méta et contexte de phrase
    dims 31-32: longueur et densité phonémiques (v4.1)
"""
from __future__ import annotations

from typing import List

DIM = 33

# Tranches contractuelles (slices).
OP_SCORES = slice(0, 4)      # ADD, SUB, MUL, DIV
CHIRALITY = slice(4, 6)      # dextro, lévo
ENERGY_STRUCTURE = slice(6, 8)
PHONEME_SIG = slice(8, 23)   # signatures + spins (les traits du chantier 7)
META_PHRASE = slice(23, 31)
LEN_DENSITY = slice(31, 33)

OPERATORS = ("ADD", "SUB", "MUL", "DIV")
CHIRALITY_NAMES = ("DX", "LV")  # dextro=DX, lévo=LV


def _as_list(vec) -> List[float]:
    """Convertit tout itérable (np.ndarray, list, tuple…) en liste de floats."""
    return [float(v) for v in vec]


def check_dim(vec) -> None:
    v = _as_list(vec)
    if len(v) != DIM:
        raise ValueError(f"vecteur de dimension {len(v)}, attendu {DIM}")


def op_scores(vec) -> List[float]:
    return _as_list(vec)[OP_SCORES]


def op_from_vector(vec) -> str:
    """Opérateur dominant = argmax des dims 0-3. Égalité tranchée par l'ordre."""
    scores = op_scores(vec)
    best = max(range(len(scores)), key=lambda i: scores[i])
    return OPERATORS[best]


def chirality_from_vector(vec) -> str:
    """Chiralité dominante = argmax des dims 4-5 (DX vs LV)."""
    v = _as_list(vec)[CHIRALITY]
    return CHIRALITY_NAMES[0] if v[0] >= v[1] else CHIRALITY_NAMES[1]


def phoneme_signature(vec) -> List[float]:
    """Les 15 dims phonémiques (8-22) — canaux d'entrée du chantier 7."""
    return _as_list(vec)[PHONEME_SIG]
