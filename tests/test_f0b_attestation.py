"""Tests Tour 36 — ATTESTATION de F0b (texte réel → parseur → classificateur).

H36 (émission gelée, ``TOUR36_EMISSION.md``) : des cycles F0b *sincères* écrits
par le linguiste (``corpus_f0b_aba.txt``, 24 cycles, HORS dépôt git) traversent
le pipeline entier SANS code neuf :

    texte → aba.py::parse_aba_line → segment_sign → classify_form_extended = "F0b"

à 100 % (substrat symbolique discret — 23/24 = FAUX). Prédiction dynamique
**P-can** (gelée §3 AVANT mesure) : le profil token de F0b est ``[+1]·a +
[−1]·(b+c)`` = mono-flip contigu = l'arrangement CANONIQUE ⟹
``Δ_profile(o; η) == delta_nom_exact_eta(N, k=a, η)`` EXACT (``Fraction``) aux
η gelés {1/2, 1, 4}, zéro échappement / 24. « F0b est grammaticalement neuve,
dynamiquement canonique — la forme vit dans la SEGMENTATION, pas dans le
PROFIL. »

Quatuor adapté : (1) chemin réel bout-en-bout, valeurs gelées (les 24 (N, a)
gravés + Δ témoins en ``Fraction``) ; (2) déterminisme (aucun aléa, corpus
MD5-gardé : md5 ≠ scellé ⟹ skip déclaré « corpus modifié », jamais un faux
vert) ; (3) cohérence segment→profil (anti-circularité : ``aba.py``
exclusivement, jamais les dims 0-5 du 33D) ; (4) la bascule d'étiquette T36
(la citoyenne de papier devient citoyenne attestée — historique T35 préservé).

Convention skip T24-T35 : les corpus vivent hors dépôt git ; absence = skip
propre, jamais un échec.
"""
import hashlib
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from spiraton.data.aba import parse_aba_line, try_parse_aba_line, validate_cycle
from spiraton.data.aba_forms import classify_form_extended, describe_form
from spiraton.diagnostics.profile_exact import (
    ETAS_T31,
    ETA_LABELS_T31,
    classify_form,
    delta_nom_exact_profile,
    segment_sign,
)
from spiraton.diagnostics.spectral_map import delta_nom_exact_eta
from spiraton.experimental.structural_gap import orientation_profile

# --- corpus SCELLÉ (T36) : chemin, md5, garde ---------------------------------

CORPUS_F0B = "corpus_f0b_aba.txt"
#: md5 scellé par l'orchestrateur T36 AVANT le déploiement — toute divergence
#: signifie « corpus modifié » : skip DÉCLARÉ (le gel est l'ancre, pas le test).
CORPUS_F0B_MD5 = "3c90c13761f16a3eb71b518766c329e1"


def corpus_f0b_path() -> Path:
    """Même résolution que ``corpus_horscanon_path`` (racine du workspace)."""
    p = Path("F:/code/claude/spiraton-enhanced") / CORPUS_F0B
    if not p.is_file():
        p = Path(__file__).resolve().parents[2] / CORPUS_F0B
    return p


_CORPUS = corpus_f0b_path()
_HAVE = _CORPUS.is_file()
_MD5_OK = (
    _HAVE
    and hashlib.md5(_CORPUS.read_bytes()).hexdigest() == CORPUS_F0B_MD5
)

needs_corpus = pytest.mark.skipif(
    not _HAVE, reason="corpus_f0b_aba.txt absent (hors dépôt git)")
needs_sealed = pytest.mark.skipif(
    not _MD5_OK,
    reason="corpus_f0b_aba.txt absent OU corpus modifié (md5 ≠ scellé T36 "
           f"{CORPUS_F0B_MD5}) — le gel est rompu, mesure non significative")

# --- constantes GELÉES (gravées depuis la mesure T36, jamais recalculées à vue) --

#: Les 24 (N, k=a) dans l'ordre du fichier (P2/P3 T36) — N = a+b+c, k = |SEG_A|.
NK_FROZEN: Tuple[Tuple[int, int], ...] = (
    (12, 1), (15, 6), (12, 6), (17, 6), (14, 3), (13, 3), (11, 1), (16, 5),
    (11, 5), (18, 6), (11, 3), (15, 5), (12, 1), (16, 6), (11, 5), (14, 5),
    (12, 3), (14, 4), (16, 5), (18, 5), (15, 5), (12, 3), (17, 7), (13, 1),
)

#: Opérateurs dans l'ordre du fichier (6/6/6/6, répartition volontaire §6).
OPS_FROZEN: Tuple[str, ...] = (
    "MUL", "ADD", "DIV", "SUB", "MUL", "ADD", "SUB", "DIV", "MUL", "ADD",
    "SUB", "DIV", "ADD", "MUL", "SUB", "DIV", "ADD", "MUL", "SUB", "DIV",
    "ADD", "MUL", "SUB", "DIV",
)

#: Δ témoins GRAVÉS (cycle 1-based, η) → Δ exact — extraits de la table P3 T36.
DELTA_WITNESSES: Dict[Tuple[int, str], Fraction] = {
    (1, "1/2"): Fraction(0),            # (12,1) : transparent aux 3 η
    (1, "4"): Fraction(0),
    (3, "1/2"): Fraction(1, 4),         # (12,6) : ligne 3k=2N — le refuge T32
    (3, "1"): Fraction(3, 4),
    (10, "4"): Fraction(5, 9),          # (18,6) : bord-exact ×13, clip-dominé
    (23, "1"): Fraction(7, 17),         # (17,7) : le plus grand k du corpus
    (24, "1/2"): Fraction(-1, 13),      # (13,1) : émission minimale, Δ < 0
    (24, "4"): Fraction(-1, 13),
}


def _load_cycles():
    """Le pipeline RÉEL : lignes du fichier → ``parse_aba_line`` (aucun cache)."""
    cycles = []
    with open(_CORPUS, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            cyc = try_parse_aba_line(stripped)
            if cyc is not None:
                cycles.append(cyc)
    return cycles


# =============================================================================
# Quatuor 1/4 — H36 : le chemin texte → parseur → classificateur, 100 %
# =============================================================================

@needs_sealed
def test_h36_end_to_end_all_f0b() -> None:
    """P2 T36 : les 24 lignes réelles parsent, signent (+,−,−), classent F0b.

    100 % exigé (émission §1 : substrat discret, 23/24 = FAUX). L'ancien
    ``classify_form`` gelé T31 lève sur CHACUN : la lacune historique est
    close (T35) ET peuplée (T36).
    """
    cycles = _load_cycles()
    assert len(cycles) == 24
    for i, cyc in enumerate(cycles):
        segs = (cyc.seg_a, cyc.seg_b, cyc.seg_a_prime)
        signs = tuple(segment_sign(s) for s in segs)
        lens = tuple(len(s.text.split()) for s in segs)
        assert signs == (+1, -1, -1)                      # le triplet F0b
        assert lens[1] >= 1                               # jamais F4 (B non vide)
        assert classify_form_extended(signs, lens) == "F0b"
        with pytest.raises(ValueError):
            classify_form(signs, lens)                    # instrument T31, gelé
        assert cyc.op == OPS_FROZEN[i]
        assert cyc.seg_a.op == cyc.seg_b.op == cyc.seg_a_prime.op
        # F0b n'est PAS la clôture F0 : validate_cycle doit signaler SEG_B
        issues = validate_cycle(cyc)
        assert issues == ["SEG_B attendu DX/OUT/OMEGA, trouvé LV/IN/OMEGA"]


@needs_sealed
def test_h36_census_frozen() -> None:
    """P1 T36 gravée : (N, k=a) par cycle, ops 6/6/6/6, éventail de l'émission."""
    cycles = _load_cycles()
    nks = []
    for cyc in cycles:
        segs = (cyc.seg_a, cyc.seg_b, cyc.seg_a_prime)
        lens = tuple(len(s.text.split()) for s in segs)
        nks.append((sum(lens), lens[0]))
    assert tuple(nks) == NK_FROZEN
    ops = {op: OPS_FROZEN.count(op) for op in set(OPS_FROZEN)}
    assert ops == {"ADD": 6, "SUB": 6, "MUL": 6, "DIV": 6}
    # inclusions de l'éventail gelé (émission §2) — indices 1-based
    a_vals = [k for (_, k) in NK_FROZEN]
    assert [i + 1 for i, a in enumerate(a_vals) if a == 1] == [1, 7, 13, 24]
    ns = [n for (n, _) in NK_FROZEN]
    assert min(ns) == 11 and max(ns) == 18   # comptage AUTOMATIQUE (INFO : manuel disait 19)


# =============================================================================
# Quatuor 2/4 — P-can : F0b est dynamiquement CANONIQUE (Fraction exact)
# =============================================================================

@needs_sealed
def test_pcan_zero_escape() -> None:
    """P3 T36 : ``Δ_profile == delta_nom_exact_eta(N, k=a, η)`` — 72/72 exact.

    Le profil de F0b (``[+1]*a + [−1]*(b+c)``) est mono-flip contigu ⟹
    arrangement canonique ⟹ zéro échappement (contraste : F2/F3 bi-flip
    échappaient à T31). Égalité de ``Fraction``, aucun float, aucun offset
    (bord-exact strict, N-A — émission §3).
    """
    cycles = _load_cycles()
    n_eq = 0
    for cyc in cycles:
        segs = (cyc.seg_a, cyc.seg_b, cyc.seg_a_prime)
        signs = tuple(segment_sign(s) for s in segs)
        lens = tuple(len(s.text.split()) for s in segs)
        orients = [s for s, l in zip(signs, lens) for _ in range(l)]
        n, k = len(orients), lens[0]
        for lbl in ETA_LABELS_T31:
            eta = ETAS_T31[lbl]
            assert delta_nom_exact_profile(orients, eta) == delta_nom_exact_eta(n, k, eta)
            n_eq += 1
    assert n_eq == 72


@needs_sealed
def test_pcan_delta_witnesses_frozen() -> None:
    """Δ témoins GRAVÉS (table P3 T36) : valeurs exactes en ``Fraction``."""
    cycles = _load_cycles()
    for (ci, lbl), expected in DELTA_WITNESSES.items():
        cyc = cycles[ci - 1]
        segs = (cyc.seg_a, cyc.seg_b, cyc.seg_a_prime)
        signs = tuple(segment_sign(s) for s in segs)
        lens = tuple(len(s.text.split()) for s in segs)
        orients = [s for s, l in zip(signs, lens) for _ in range(l)]
        assert delta_nom_exact_profile(orients, ETAS_T31[lbl]) == expected


# =============================================================================
# Quatuor 3/4 — cohérence & anti-circularité (segment → profil, aba.py seul)
# =============================================================================

@needs_sealed
def test_segment_profile_coherence() -> None:
    """``orientation_profile`` (instrument gelé) == expansion des signes de segment.

    Anti-circularité (émission §7.6) : le profil vient d'``aba.py``
    EXCLUSIVEMENT — jamais d'un score sémantique appris ni des dims 0-5 du 33D.
    """
    for cyc in _load_cycles():
        segs = (cyc.seg_a, cyc.seg_b, cyc.seg_a_prime)
        signs = tuple(segment_sign(s) for s in segs)
        lens = tuple(len(s.text.split()) for s in segs)
        expansion = tuple(s for s, l in zip(signs, lens) for _ in range(l))
        orients = tuple(tk.orientation for tk in orientation_profile(cyc))
        assert orients == expansion


@needs_corpus
def test_corpus_sealed_md5_declared() -> None:
    """La garde MD5 elle-même : si le fichier existe, son état de gel est déclaré.

    Ce test ne force RIEN : corpus intact ⟹ md5 == scellé ; corpus modifié ⟹
    les tests ``needs_sealed`` ont skippé « corpus modifié » et celui-ci échoue
    en le DÉCLARANT (le gel rompu doit se voir, pas se taire).
    """
    got = hashlib.md5(_CORPUS.read_bytes()).hexdigest()
    assert got == CORPUS_F0B_MD5, (
        f"corpus_f0b_aba.txt modifié : md5 {got} ≠ scellé {CORPUS_F0B_MD5} "
        "(gel T36 rompu — re-gel explicite requis, jamais un patch furtif)")


# =============================================================================
# Quatuor 4/4 — la bascule d'étiquette (l'arbitrage tranché par la matière)
# =============================================================================

def test_f0b_label_attested() -> None:
    """T36 : la citoyenne de papier est désormais ATTESTÉE (formulation gelée §4).

    Historique préservé : T35 l'avait nommée-définie-testée en synthétique avec
    l'étiquette honnête « non encore attestée en corpus » ; T36 apporte la
    matière (24 cycles réels, 100 % bout-en-bout) et l'étiquette bascule AVEC
    elle. Ce test est INDÉPENDANT du corpus (l'étiquette vit dans le code).
    """
    info = describe_form("F0b")
    assert info.attestation == (
        "attestée en corpus (corpus_f0b_aba.txt, T36) ; "
        "sémantique défendue par symétrie (T35)")
    assert "non encore" not in info.attestation
    # le reste du descripteur T35 est INTACT (seul le champ attestation a bougé)
    assert info.triplet == (+1, -1, -1)
    assert info.family == "mono-flip" and info.flip_position == "précoce"
    assert info.ending == "repliée" and info.measurable
