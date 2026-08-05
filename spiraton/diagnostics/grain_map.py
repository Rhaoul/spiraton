# -*- coding: utf-8 -*-
"""grain_map.py — T70 : la carte des grains (#uniques/N) publiée en GATE.

Diagnostic RÉUTILISABLE de dégénérescence de support par grain (g0-g3 pour un
corpus de texte, g4 pour un corpus balisé ABA), né du fait T67 : sur
``aba_v2`` au grain agrégat, 19 vecteurs ``a`` uniques /1024 (0,0186) ont
produit des distances nulles et un transport NON-MESURE. La carte compte, le
gate borne l'admissibilité au transport — AVANT de brûler un split.

Convention M2 (vue 2 / grain.json) — deux lectures sur la MÊME unité (le
token/mot rendu par ``tokeniser_texte``), donc commensurables entre grains :

* **Lecture A (alphabet du grain)** : un symbole par token — g0
  ``oper_phonemique`` (étiquette OP), g1 ``spin_dominant`` (étiquette SPIN),
  g2 ``operateur`` (OP), g3 ``torsion_phrase`` (TORSION). Publie : counts,
  #uniques, N tokens, ratio, entropie bits (l'entropie est une INFO, jamais
  une grandeur porteuse).
* **Lecture B (signature de phrase/cycle)** : jointure ``"|"`` des symboles
  des tokens de la phrase, dans l'ordre. Publie : #signatures uniques,
  N phrases, ratio.
* **Natif** : g2 = formes de mots (``texte_original`` UTF-8 → ``str.lower()``),
  g3 = phrases exactes ; g0/g1 = types NON exposés au wrapper ctypes
  (**BORNE R16**) ⇒ comptes seuls (Σ ``nb_phonemes``, Σ ``nb_syllabes``),
  jamais un #uniques natif.

Définition d'« unique » (gelée T70, émission §3.3 — identique au mot) :

* **Token (g2 natif)** : chaîne ``texte_original`` décodée UTF-8, passée à
  ``str.lower()`` Python ; accents CONSERVÉS, ponctuation telle que découpée
  par le C, aucun autre nettoyage ; égalité stricte de chaînes. (Identique à
  la route α : vue 2.)
* **Phrase (g3 natif)** : ligne lue, ``strip()`` des blancs tête/queue ;
  casse, ponctuation, accents CONSERVÉS ; lignes vides ignorées et comptées.
* **Signature (lecture B)** : jointure ``"|"`` ordonnée des symboles.
* **Segment (g4)** : texte du segment après retrait des balises et de
  ``<EOL>`` + ``strip()`` ; casse/ponctuation/accents CONSERVÉS. Cycle =
  jointure des 3 segments par un séparateur hors-texte (``"\\x1f"``) ;
  a-texte = SEG_A seul. Route : le parseur de référence
  :func:`spiraton.data.aba.parse_aba_line` (dont ``_clean_text`` normalise
  AUSSI les espaces internes — déclaré : cette normalisation ne touche ni
  casse, ni ponctuation, ni accents).

Bornes déclarées (PORTÉE de l'instrument, contrat T60) :

* **R16** : les types natifs g0 (phonèmes) et g1 (syllabes) ne sont PAS
  exposés au wrapper ctypes — toute cellule (corpus × g0/g1) est
  ``INADMISSIBLE-PAR-BORNE`` au gate natif : ni PASS ni FAIL, motif T72
  (ABI). L'inventaire C du g0 est borné à 36 phonèmes.
* **R43** : au grain g0 l'opérateur phonémique est une constante de table
  (``phonemes_fr.c`` : toutes les voyelles → ``OP_ADD``, etc.) — toute
  cellule g0-lecture-A à #uniques ≤ 1 est ``DEGENERE-PAR-CONSTRUCTION``,
  publiée comme telle, jamais comme une entropie qui mesurerait.
* **Compteur d'uniques** : détecte l'égalité EXACTE de chaînes/symboles ; ne
  détecte PAS la proximité sémantique ni la diversité des flottants 33D. La
  forme de mot est une borne INFÉRIEURE du nombre d'états 33D distincts
  (les dims de contexte 28-30 peuvent séparer deux occurrences du même mot) :
  un PASS du proxy est sûr, un FAIL peut être un faux-FAIL (sens conservateur).

Le gate (:func:`evaluer_gate`) reçoit ses seuils EN PARAMÈTRE : les seuils
gelés d'un tour (T70 : ratio 0,05 ∨ plancher 512) vivent dans l'appel du
tour, pas en dur dans le canon. PASS ⇔ (G = #uniques/N ≥ seuil_ratio) OU
(#uniques ≥ seuil_uniques) ; le plancher absolu protège les grands N contre
l'effet Heaps (le ratio type/token décroît mécaniquement avec N).

Frontière matérielle : TOUTE lecture de fichier passe par
:func:`lire_lignes_bornees` (``readline()`` un à un, arrêt dur AVANT l'appel
excédentaire — jamais de lecture-entière-puis-tranche) ; l'appelant publie
``n_lues`` par fichier. Aucune sortie aléatoire ; comptage déterministe pur.

LECTEUR PUR : ce module ne modifie ni le canon, ni le tokenizer C (le ``.so``
se lit via ``data/tokenizer_bridge``, il ne se modifie pas). Les verdicts
vivent dans les artefacts de tour (TOUR70_*.json), pas ici.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..data import aba
from ..data.aba_forms import classify_form_extended

# ── Vocabulaires (miroir exact des enums C, csrc/spiraton_tokenizer.h) ──────
OP: Tuple[str, ...] = ("ADD", "SUB", "MUL", "DIV", "PURE")
ORIENT: Tuple[str, ...] = ("DEXTRO", "LEVO", "NEUTRE")
SPIN: Tuple[str, ...] = ("OUVERTURE", "FERMETURE", "STABLE", "MIXTE")
TORSION: Tuple[str, ...] = (
    "STABILISATION", "LIBERATION", "COMPRESSION",
    "INVERSION", "RESONANCE", "NEUTRE",
)

#: Clé du symbole de lecture A par grain (convention M2 / vue 2).
GRAIN_CLES: Dict[int, str] = {
    0: "g0_oper",
    1: "g1_spin",
    2: "g2_oper",
    3: "g3_torsion",
}

#: Statuts de cellule publiables par la carte (liste close).
STATUTS = (
    "PASS", "FAIL", "INADMISSIBLE-PAR-BORNE", "DEGENERE-PAR-CONSTRUCTION",
)


# ── Lecture bornée (frontière matérielle) ───────────────────────────────────

def lire_lignes_bornees(
    path: str, n_max: Optional[int]
) -> Tuple[List[str], int]:
    """Lit AU PLUS ``n_max`` lignes de ``path`` — arrêt dur, matériel.

    ``readline()`` un à un : l'appel excédentaire n'est JAMAIS émis (pas de
    lecture-entière-puis-tranche). ``n_max=None`` = sans borne, réservé aux
    corpus α/INFO déjà consommés — tout fichier sous frontière (T70 :
    ``dataset_aba_v2.txt``/``corpus_eve_v2.txt``, lignes 1025+ = réserve)
    DOIT être lu avec la borne du tour (1024). Retourne
    ``(lignes_brutes, n_lues)`` ; l'appelant publie ``n_lues``.
    """
    if n_max is not None and n_max <= 0:
        raise ValueError(f"n_max doit être > 0 ou None, reçu {n_max!r}")
    lignes: List[str] = []
    n_lues = 0
    with open(path, "r", encoding="utf-8") as fh:
        while n_max is None or n_lues < n_max:
            ligne = fh.readline()
            if not ligne:
                break
            lignes.append(ligne)
            n_lues += 1
    assert n_max is None or n_lues <= n_max
    return lignes, n_lues


def phrases_non_vides(lignes: Sequence[str]) -> Tuple[List[str], int]:
    """Phrases = lignes ``strip()`` non vides (route α, vue 2).

    Retourne ``(phrases, n_vides)`` — les vides sont ignorées ET comptées
    (définition §3.3).
    """
    phrases = []
    n_vides = 0
    for ligne in lignes:
        s = ligne.strip()
        if s:
            phrases.append(s)
        else:
            n_vides += 1
    return phrases, n_vides


# ── Tokenisation brute (champs de struct non exposés par tokenize()) ────────

def charger_tokenizer():
    """Charge le tokenizer natif et retourne ``(tok, SpiratonToken)``.

    Route indépendante de la vue 2 : on repasse par le pont du dépôt
    (``data/tokenizer_bridge.load_native_tokenizer``), puis on récupère la
    classe de struct ctypes du wrapper pour lire les champs de grain 0/1/3
    que ``tokenize()`` n'expose pas. Lève ``TokenizerUnavailable`` si le
    dépôt tokenizer n'est pas à côté.
    """
    from ..data import tokenizer_bridge

    tok = tokenizer_bridge.load_native_tokenizer()
    # La struct EXACTE liée au symbole C (celle des argtypes posés par le
    # wrapper) — pas une ré-importation, qui créerait une classe ctypes
    # distincte et incompatible.
    token_struct = tok.lib.tokeniser_texte.argtypes[1]._type_
    return tok, token_struct


def _lbl(table: Sequence[str], idx: int) -> str:
    return table[idx] if 0 <= idx < len(table) else "?"


def tokeniser_grains(
    tok, token_struct, texte: str, max_tokens: int = 250
) -> Tuple[List[Dict[str, object]], bool]:
    """Tokenise ``texte`` et rend, par token, les symboles des 4 grains.

    Lit la struct C brute (``lib.tokeniser_texte``) — mêmes étiquettes que la
    vue 2 (miroir des enums C), code de comptage RÉÉCRIT (route indépendante).
    Retourne ``(tokens, tronque)`` où ``tronque`` signale ``count ==
    max_tokens`` (suspicion de troncature, à publier en INFO).
    """
    arr = (token_struct * max_tokens)()
    count = tok.lib.tokeniser_texte(texte.encode("utf-8"), arr, max_tokens)
    if count < 0 or count > max_tokens:
        raise RuntimeError(f"count C incohérent : {count}")
    tokens: List[Dict[str, object]] = []
    for i in range(count):
        t = arr[i]
        tokens.append({
            "texte": t.texte_original.decode("utf-8", errors="replace"),
            "g0_oper": _lbl(OP, t.oper_phonemique),
            "g0_nb_phonemes": int(t.nb_phonemes),
            "g1_spin": _lbl(SPIN, t.spin_dominant),
            "g1_nb_syllabes": int(t.nb_syllabes),
            "g2_oper": _lbl(OP, t.operateur),
            "g3_torsion": _lbl(TORSION, t.torsion_phrase),
        })
    return tokens, (count == max_tokens)


# ── Comptages purs (testables sans le .so) ──────────────────────────────────

def entropie_bits(counts: Counter) -> float:
    """Entropie de Shannon en bits d'un Counter (INFO, jamais porteuse)."""
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def lecture_A(
    phrases_tokens: Sequence[Sequence[Dict[str, object]]], cle: str
) -> Dict[str, object]:
    """Alphabet du grain : counts / #uniques / N tokens / ratio / entropie."""
    cnt = Counter(t[cle] for ph in phrases_tokens for t in ph)
    n = sum(cnt.values())
    return {
        "counts": dict(sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))),
        "uniques": len(cnt),
        "N": n,
        "ratio": (len(cnt) / n) if n else None,
        "entropie_bits": round(entropie_bits(cnt), 4),
    }


def lecture_B(
    phrases_tokens: Sequence[Sequence[Dict[str, object]]], cle: str
) -> Dict[str, object]:
    """Signatures de phrase : jointure ``"|"`` ordonnée des symboles."""
    sigs = Counter(
        "|".join(str(t[cle]) for t in ph) for ph in phrases_tokens
    )
    n = sum(sigs.values())
    return {
        "uniques": len(sigs),
        "N": n,
        "ratio": (len(sigs) / n) if n else None,
    }


def natif_mots(
    phrases_tokens: Sequence[Sequence[Dict[str, object]]]
) -> Counter:
    """Formes de mots (g2 natif) : ``texte_original`` → ``str.lower()``."""
    return Counter(
        str(t["texte"]).lower() for ph in phrases_tokens for t in ph
    )


def natif_phrases(phrases: Sequence[str]) -> Counter:
    """Phrases exactes (g3 natif) : lignes strip(), casse/accents conservés."""
    return Counter(phrases)


def carte_texte(
    phrases: Sequence[str],
    phrases_tokens: Sequence[Sequence[Dict[str, object]]],
) -> Dict[str, object]:
    """Carte g0-g3 d'un corpus de texte (une phrase par ligne, tokenisée).

    Comptage pur sur des tokens déjà produits (:func:`tokeniser_grains`) —
    testable au bit sans le ``.so``. Les statuts de gate sont posés par
    l'appelant (seuils du tour) ; ici : les deux lectures + le natif/borne.
    """
    if len(phrases) != len(phrases_tokens):
        raise ValueError(
            f"{len(phrases)} phrases vs {len(phrases_tokens)} listes de tokens"
        )
    tokens_plats = [t for ph in phrases_tokens for t in ph]
    n_tok = len(tokens_plats)
    mots = natif_mots(phrases_tokens)
    sents = natif_phrases(phrases)
    grains: List[Dict[str, object]] = []
    for gid in (0, 1, 2, 3):
        cle = GRAIN_CLES[gid]
        entry: Dict[str, object] = {
            "grain": f"g{gid}",
            "lecture_A": lecture_A(phrases_tokens, cle),
            "lecture_B": lecture_B(phrases_tokens, cle),
        }
        if gid == 0:
            entry["natif"] = {
                "expose": False,
                "borne": "R16 — types de phonèmes non exposés au wrapper",
                "N_unites": sum(int(t["g0_nb_phonemes"]) for t in tokens_plats),
                "uniques_borne_sup": 36,
            }
        elif gid == 1:
            entry["natif"] = {
                "expose": False,
                "borne": "R16 — types de syllabes non exposés au wrapper",
                "N_unites": sum(int(t["g1_nb_syllabes"]) for t in tokens_plats),
                "uniques_borne_sup": None,
            }
        elif gid == 2:
            entry["natif"] = {
                "expose": True,
                "N_unites": n_tok,
                "uniques": len(mots),
                "ratio": (len(mots) / n_tok) if n_tok else None,
            }
        else:
            entry["natif"] = {
                "expose": True,
                "N_unites": len(phrases),
                "uniques": len(sents),
                "ratio": (len(sents) / len(phrases)) if phrases else None,
            }
        grains.append(entry)
    return {
        "n_phrases": len(phrases),
        "n_tokens": n_tok,
        "grains": grains,
    }


# ── Grain g4 (corpus balisés ABA seulement) ─────────────────────────────────

def _signe_segment(seg: "aba.AbaSegment") -> int:
    """+1 = <DX><OUT>, −1 = <LV><IN>, 0 = hors grammaire (GRAMMAIRE_ABA.md)."""
    if seg.chirality == "DX" and seg.direction == "OUT":
        return +1
    if seg.chirality == "LV" and seg.direction == "IN":
        return -1
    return 0


def forme_cycle(cycle: "aba.AbaCycle") -> str:
    """Forme d'arrangement du cycle (F0…F5) via ``classify_form_extended``.

    ``"?"`` pour un arrangement hors grammaire (``<DX><IN>``/``<LV><OUT>``).
    """
    segs = (cycle.seg_a, cycle.seg_b, cycle.seg_a_prime)
    signs = tuple(_signe_segment(s) for s in segs)
    if 0 in signs:
        return "?"
    lens = tuple(len(s.text.split()) if s.text else 0 for s in segs)
    return classify_form_extended(signs, lens)


def carte_g4(cycles: Sequence["aba.AbaCycle"]) -> Dict[str, object]:
    """Carte g4 d'un corpus balisé : segments / cycles / a-textes uniques.

    Unités : segments (3 par cycle) et cycles ; grandeurs : #segments-textes
    uniques / 3N, #cycles uniques / N, **#a-textes uniques / N** (l'analogue
    textuel exact du fait T67 « 19 vecteurs ``a`` uniques /1024 »). Formes
    d'arrangement publiées (INFO structurelle).
    """
    n = len(cycles)
    segs = [
        s.text
        for c in cycles
        for s in (c.seg_a, c.seg_b, c.seg_a_prime)
    ]
    a_textes = [c.seg_a.text for c in cycles]
    cycles_txt = [
        "\x1f".join((c.seg_a.text, c.seg_b.text, c.seg_a_prime.text))
        for c in cycles
    ]
    formes = Counter(forme_cycle(c) for c in cycles)
    ops = Counter(c.op for c in cycles)
    seg_cnt, a_cnt, cyc_cnt = Counter(segs), Counter(a_textes), Counter(cycles_txt)
    return {
        "n_cycles": n,
        "segments": {
            "uniques": len(seg_cnt), "N": 3 * n,
            "ratio": (len(seg_cnt) / (3 * n)) if n else None,
        },
        "cycles": {
            "uniques": len(cyc_cnt), "N": n,
            "ratio": (len(cyc_cnt) / n) if n else None,
        },
        "a_textes": {
            "uniques": len(a_cnt), "N": n,
            "ratio": (len(a_cnt) / n) if n else None,
        },
        "formes": dict(sorted(formes.items())),
        "operateurs": dict(sorted(ops.items())),
    }


def parser_cycles(lignes: Sequence[str]) -> Tuple[List["aba.AbaCycle"], int, int]:
    """Parse des lignes ABA via la référence (``try_parse_aba_line``).

    Retourne ``(cycles, n_terminateurs_ou_vides, n_erreurs)`` — les erreurs
    sont comptées, jamais masquées (réfutabilité).
    """
    cycles: List[aba.AbaCycle] = []
    n_sautees = 0
    n_erreurs = 0
    for ligne in lignes:
        try:
            c = aba.try_parse_aba_line(ligne)
        except aba.AbaParseError:
            n_erreurs += 1
            continue
        if c is None:
            n_sautees += 1
        else:
            cycles.append(c)
    return cycles, n_sautees, n_erreurs


# ── Le gate (seuils en paramètre — les seuils gelés vivent dans le tour) ────

def evaluer_gate(
    uniques: int, n: int, seuil_ratio: float, seuil_uniques: int
) -> Dict[str, object]:
    """Gate d'admissibilité au transport (dégénérescence de support).

    ``PASS ⇔ (G = uniques/n ≥ seuil_ratio) OU (uniques ≥ seuil_uniques)`` ;
    ``FAIL`` sinon. PORTÉE : condition NÉCESSAIRE (peu d'états distincts ⇒
    distances nulles ⇒ Δ apparié NON-MESURE), pas suffisante ; le proxy
    textuel est une borne inférieure des états 33D (faux-FAIL possible,
    faux-PASS non — sens conservateur). Les seuils sont passés par l'appel
    du tour (T70 : 0,05 ∨ 512), jamais codés en dur ici.
    """
    if n <= 0:
        raise ValueError(f"n doit être > 0, reçu {n}")
    if uniques < 0 or uniques > n:
        raise ValueError(f"uniques={uniques} hors [0, n={n}]")
    if not (0.0 < seuil_ratio <= 1.0):
        raise ValueError(f"seuil_ratio hors ]0,1] : {seuil_ratio}")
    if seuil_uniques <= 0:
        raise ValueError(f"seuil_uniques doit être > 0 : {seuil_uniques}")
    g = uniques / n
    statut = "PASS" if (g >= seuil_ratio or uniques >= seuil_uniques) else "FAIL"
    return {
        "G": g,
        "uniques": uniques,
        "N": n,
        "seuil_ratio": seuil_ratio,
        "seuil_uniques": seuil_uniques,
        "statut": statut,
    }


def statut_lecture_A(uniques: int) -> Optional[str]:
    """R43 : ``#uniques ≤ 1`` en lecture A ⇒ ``DEGENERE-PAR-CONSTRUCTION``."""
    return "DEGENERE-PAR-CONSTRUCTION" if uniques <= 1 else None
