"""H8-P1 — variante DIAGNOSTIC : opérateur dominant pondéré par −log p(phonème).

MODULE SÉPARÉ de l'émetteur canon (``spiraton_tokenizer.aba_emitter``) : le défaut
``dominant_operator`` reste bit-identique. Ici, une lentille « informée » qui
pondère chaque vote phonémique par sa surprise −log p (geste DIV/lévogyre :
distinction). Hypothèse : les consonnes rares distinctives (SUB/DIV) cessent
d'être noyées par les voyelles fréquentes (ADD), l'accord monte.

UNE SEULE fonctionnelle déclarée d'avance : f(phonème) = −log p(phonème) (garde
anti-fit (ii) : pas de balayage {1/p, TF-IDF, …}). Le vote pondéré d'un mot est
``Σ_phonème poids(phonème) · 1[op(phonème)=op]`` ; on somme sur les mots du cycle,
puis argmax. p est GELÉE avant toute balise (cf. build_phoneme_logp.py).

Le défaut (présence) correspond à poids=1 pour tout phonème — donc cette fonction
GÉNÉRALISE le vote canon (qu'on retrouve avec ``logp_table=None``). Cela permet de
prouver, dans les tests, qu'avec poids uniforme on reproduit le dominant Python.

NB granularité : le ``.so`` n'expose pas la séquence de phonèmes par token. On
utilise donc la référence Python (``phoneme_ref``, validée par parité). Le calcul
informé est cohérent avec lui-même ; le compte-rendu rapporte le taux de parité
réf./.so sur le dataset réel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .phoneme_ref import OPS, PHONEME_OP, g2p_heuristic


def load_logp_table(path: str) -> Dict[str, float]:
    """Charge une table {phonème: −log p} sérialisée par build_phoneme_logp.py."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return dict(payload["neg_log_p"])


def _weighted_counts(words: List[str], logp: Optional[Dict[str, float]]) -> List[float]:
    """Vote pondéré par opérateur ACTIF (ADD/SUB/MUL/DIV), PURE exclu.

    logp=None  → poids uniforme 1.0 (reproduit le vote canon par présence).
    logp dict  → poids = −log p(phonème) (information ; rareté = poids fort).
    Phonème absent de la table : repli sur la surprise max observée (rare).
    """
    counts = [0.0, 0.0, 0.0, 0.0]
    fallback = max(logp.values()) if logp else 1.0
    for w in words:
        for ph in g2p_heuristic(w):
            op = PHONEME_OP.get(ph, "PURE")
            if op not in OPS:  # PURE
                continue
            weight = 1.0 if logp is None else logp.get(ph, fallback)
            counts[OPS.index(op)] += weight
    return counts


def dominant_operator_informed(
    words: List[str], logp_table: Optional[Dict[str, float]] = None
) -> str:
    """Opérateur dominant pondéré-information du cycle (argmax du vote pondéré).

    ``words`` : liste des mots du cycle (texte nu). ``logp_table`` : table −log p
    gelée. Avec ``logp_table=None`` on retombe sur le vote par présence (canon).
    """
    counts = _weighted_counts(words, logp_table)
    if sum(counts) == 0.0:
        return "ADD"
    return OPS[max(range(4), key=lambda i: counts[i])]


def per_op_weight_mass(words: List[str], logp: Optional[Dict[str, float]]) -> List[float]:
    """Masse de poids par opérateur (pour inspecter le déplacement du biais)."""
    return _weighted_counts(words, logp)
