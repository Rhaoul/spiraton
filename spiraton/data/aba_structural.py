"""H10 — variante DIAGNOSTIC : vote opérateur par SÉLECTION STRUCTURELLE.

MODULE SÉPARÉ de l'émetteur canon (``spiraton_tokenizer.aba_emitter``) et du défaut
``dominant_operator`` : ces défauts restent bit-identiques. Ici, deux lentilles
expérimentales sur la séquence d'opérateurs phonémiques (réf. Python ``phoneme_ref``,
validée par parité contre le ``.so`` au Tour 8) :

  * **H10-a — sélection structurelle (``op_structural``)** : on vote uniquement sur
    les opérateurs CONSONANTIQUES distinctifs {SUB (occlusives), MUL (nasales/[ʁ]),
    DIV (fricatives)}, en EXCLUANT le fond vocalique ADD et les phonèmes PURE
    (semi-voyelles, [l]). Repli ADD si aucune consonne distinctive. C'est un
    **changement de QUOI on compte** (un sous-multiset des opérateurs), donc
    **ORDRE-INVARIANT** : permuter la séquence phonémique ne change rien (le vote
    reste argmax d'un compte). Ce n'est PAS un test de séquentialité.

  * **H10-b — pondération positionnelle (``op_structural_positional``)** : même vote,
    mais chaque consonne distinctive est pondérée par sa **position dans le mot**
    (poids = 1 + index du phonème dans la séquence du mot ; les phonèmes tardifs —
    codas — pèsent plus). C'est le SEUL test d'ordre : permuter la séquence
    phonémique d'un mot DOIT effondrer la part ordre-dépendante du gain. Toute part
    qui survit à la permutation n'est PAS portée par l'ordre (c'est un effet de
    longueur/magnitude) — à reporter comme telle, jamais à vendre comme de la
    séquentialité.

UNE SEULE fonctionnelle figée a priori par hypothèse (garde anti-fit, REFUS) :
  * H10-a : argmax du compte sur {SUB,MUL,DIV} (poids 1, PURE+ADD exclus).
  * H10-b : argmax du compte pondéré par (1 + index intra-mot).
Aucun balayage de fonctionnelles, aucune lecture de balise dans le calcul.

Mapping phonème → opérateur : figé par ``phonemes_fr.c`` (lu, jamais modifié),
transcrit dans ``phoneme_ref.PHONEME_OP``. Les voyelles nasales (ɑ̃ ɔ̃ ɛ̃ œ̃) y
portent l'opérateur MUL (résonance nasale) et comptent donc comme consonne
distinctive au sens opératoriel — la sélection se fait sur l'OPÉRATEUR du phonème,
pas sur sa catégorie articulatoire.

Lien au canon : avec ``distinctive_only=False`` ET ``positional=False``, le vote
reproduit EXACTEMENT l'argmax-sac sur les quatre opérateurs (``dominant_operator``
réf. Python = la baseline 22.8 %). H10-a est donc une généralisation CONTRÔLÉE :
on retrouve la baseline en désactivant le filtre.

Licence : GPL-3.0-or-later.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence

from .phoneme_ref import OPS, phoneme_ops_for_word

# Les opérateurs « consonantiques distinctifs » : tout sauf ADD (fond vocalique)
# et PURE (semi-voyelles, [l], déjà exclus de ``OPS``).
DISTINCTIVE_OPS: tuple = ("SUB", "MUL", "DIV")


def _argmax_counts(counts: Sequence[float], distinctive_only: bool) -> str:
    """argmax du vote ; repli ADD si aucun vote (distinctif) n'a été émis.

    distinctive_only=True  → argmax sur {SUB,MUL,DIV} (ADD jamais prédit), repli ADD
                             si les trois comptes distinctifs sont nuls.
    distinctive_only=False → argmax sur les 4 opérateurs, repli ADD si tout est nul.
    """
    if distinctive_only:
        sub, mul, div = counts[1], counts[2], counts[3]
        if sub == 0 and mul == 0 and div == 0:
            return "ADD"  # aucune consonne distinctive : repli sur le fond vocalique
        return max(DISTINCTIVE_OPS, key=lambda op: counts[OPS.index(op)])
    if sum(counts) == 0:
        return "ADD"
    return OPS[max(range(4), key=lambda i: counts[i])]


def _vote_counts(
    words: Sequence[str],
    *,
    distinctive_only: bool,
    positional: bool,
    shuffle_rng: Optional[random.Random] = None,
) -> List[float]:
    """Compte (pondéré) des opérateurs phonémiques sur les mots d'un cycle.

    distinctive_only : si True, seuls SUB/MUL/DIV votent (ADD et PURE exclus).
                       si False, ADD vote aussi (PURE toujours exclu — hors ``OPS``).
    positional       : si True, poids = (1 + index du phonème dans le mot) ; sinon 1.0.
    shuffle_rng      : si fourni, permute la séquence phonémique de CHAQUE mot avant
                       le vote (ablation d'ordre — n'affecte que le mode positional ;
                       en non-positional le compte est invariant par construction).
    """
    counts = [0.0, 0.0, 0.0, 0.0]
    for w in words:
        seq = phoneme_ops_for_word(w)
        if shuffle_rng is not None:
            seq = list(seq)
            shuffle_rng.shuffle(seq)
        for idx, op in enumerate(seq):
            if op not in OPS:  # PURE
                continue
            if distinctive_only and op == "ADD":
                continue
            weight = (1.0 + idx) if positional else 1.0
            counts[OPS.index(op)] += weight
    return counts


def op_structural(words: Sequence[str]) -> str:
    """H10-a : opérateur dominant par sélection structurelle (ORDRE-INVARIANT).

    Vote sur les consonnes distinctives {SUB,MUL,DIV} ; ADD (voyelles) et PURE
    exclus ; repli ADD si aucune consonne distinctive. argmax d'un sous-multiset
    → insensible à l'ordre des phonèmes.
    """
    counts = _vote_counts(words, distinctive_only=True, positional=False)
    return _argmax_counts(counts, distinctive_only=True)


def op_structural_positional(
    words: Sequence[str], *, shuffle_rng: Optional[random.Random] = None
) -> str:
    """H10-b : sélection structurelle PONDÉRÉE PAR POSITION (ORDRE-DÉPENDANT).

    Même vote que H10-a, mais chaque consonne distinctive pèse (1 + index) dans la
    séquence phonémique de son mot. ``shuffle_rng`` permute la séquence de chaque
    mot avant le vote (ablation : la part ordre-dépendante du gain doit s'effondrer).
    """
    counts = _vote_counts(
        words, distinctive_only=True, positional=True, shuffle_rng=shuffle_rng
    )
    return _argmax_counts(counts, distinctive_only=True)


def op_sac_reference(words: Sequence[str]) -> str:
    """Baseline argmax-sac (réf. Python) : vote présence sur les 4 opérateurs.

    Reproduit ``dominant_operator`` côté réf. Python = la baseline 22.8 % (côté .so,
    avec l'écart oracle/boost documenté au Tour 8). Sert d'ancrage : H10-a avec le
    filtre désactivé doit retomber dessus.
    """
    counts = _vote_counts(words, distinctive_only=False, positional=False)
    return _argmax_counts(counts, distinctive_only=False)
