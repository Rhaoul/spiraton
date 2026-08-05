# -*- coding: utf-8 -*-
"""Quatuor de tests du compteur T70 (grain_map) — adapté au substrat EXACT :

1. **déterminisme ×2 au bit** — mêmes entrées ⇒ mêmes cartes (JSON identique),
   sur comptage pur (sans .so) et, si le tokenizer natif est à côté, sur une
   double tokenisation réelle ;
2. **valeurs exactes sur mini-corpus forcé** — lectures A/B, natif, g4 et gate
   vérifiés à la main (entiers au bit, pas d'à-peu-près) ;
3. **bornes/erreurs** — lecteur borné exercé des DEUX côtés (tronque à n_max ;
   rend tout si le fichier est plus court), gate qui refuse les entrées
   dégénérées ;
4. **parité α** (skip propre sans .so) — les 3 chiffres gelés T70 recomptés
   depuis le corpus eve par la route du module : g2 natif 1605/6724, g1
   signatures 483/877, g3 lecture A uniques=4 et COMPRESSION=2800.

Pas de test de gradient : c'est du comptage déterministe, pas un module
différentiable (adaptation du quatuor déclarée, CLAUDE.md).
"""
import hashlib
import json
import math
import os

import pytest

from spiraton.data.tokenizer_bridge import TokenizerUnavailable
from spiraton.diagnostics import grain_map as gm

# ── Mini-corpus forcé (tokens injectés — aucun .so requis) ──────────────────

def _tok(texte, g0="ADD", spin="OUVERTURE", g2="ADD", tor="COMPRESSION",
         nb_ph=2, nb_sy=1):
    return {
        "texte": texte,
        "g0_oper": g0,
        "g0_nb_phonemes": nb_ph,
        "g1_spin": spin,
        "g1_nb_syllabes": nb_sy,
        "g2_oper": g2,
        "g3_torsion": tor,
    }


PHRASES = ["Le retour transforme.", "Le retour répète.", "Le retour transforme."]
PHRASES_TOKENS = [
    [_tok("Le", g2="ADD"), _tok("retour", g2="MUL", spin="FERMETURE"),
     _tok("transforme.", g2="SUB")],
    [_tok("Le", g2="ADD"), _tok("retour", g2="MUL", spin="FERMETURE"),
     _tok("répète.", g2="SUB", tor="INVERSION")],
    [_tok("Le", g2="ADD"), _tok("retour", g2="MUL", spin="FERMETURE"),
     _tok("transforme.", g2="SUB")],
]

LIGNE_ABA = (
    "<SEG_A> <ADD><DX><OUT><ALPHA> un signal apparaît </SEG_A> "
    "<SEG_B> <ADD><DX><OUT><OMEGA> le signal se déploie </SEG_B> "
    "<SEG_A_PRIME> <ADD><LV><IN><A_PRIME> le signal revient transformé "
    "<EOL> </SEG_A_PRIME> <EOL>"
)
LIGNE_ABA_2 = LIGNE_ABA.replace("un signal apparaît", "une boucle se ferme")


# ── (1) déterminisme ×2 au bit ──────────────────────────────────────────────

def test_carte_texte_deterministe_au_bit():
    c1 = gm.carte_texte(PHRASES, PHRASES_TOKENS)
    c2 = gm.carte_texte(PHRASES, PHRASES_TOKENS)
    j1 = json.dumps(c1, sort_keys=True, ensure_ascii=False)
    j2 = json.dumps(c2, sort_keys=True, ensure_ascii=False)
    assert hashlib.md5(j1.encode()).hexdigest() == \
        hashlib.md5(j2.encode()).hexdigest()


def test_carte_g4_deterministe_au_bit():
    cycles, _, _ = gm.parser_cycles([LIGNE_ABA, LIGNE_ABA_2, LIGNE_ABA])
    g1 = json.dumps(gm.carte_g4(cycles), sort_keys=True)
    g2 = json.dumps(gm.carte_g4(cycles), sort_keys=True)
    assert g1 == g2


# ── (2) valeurs exactes sur mini-corpus forcé ───────────────────────────────

def test_lectures_A_B_exactes():
    carte = gm.carte_texte(PHRASES, PHRASES_TOKENS)
    assert carte["n_phrases"] == 3
    assert carte["n_tokens"] == 9
    g2 = carte["grains"][2]
    # Lecture A g2 : ADD×3, MUL×3, SUB×3 → 3 uniques, entropie log2(3).
    assert g2["lecture_A"]["counts"] == {"ADD": 3, "MUL": 3, "SUB": 3}
    assert g2["lecture_A"]["uniques"] == 3
    assert g2["lecture_A"]["N"] == 9
    assert abs(g2["lecture_A"]["entropie_bits"] - round(math.log2(3), 4)) < 1e-9
    # Lecture B g2 : les 3 phrases ont la même signature ADD|MUL|SUB.
    assert g2["lecture_B"] == {"uniques": 1, "N": 3, "ratio": 1 / 3}
    # Lecture B g3 : COMPRESSION³ ×2 phrases, C|C|INVERSION ×1 → 2 uniques.
    g3 = carte["grains"][3]
    assert g3["lecture_B"]["uniques"] == 2
    # Natif g2 : formes lower — le, retour, transforme., répète. → 4 uniques.
    assert g2["natif"]["uniques"] == 4
    assert g2["natif"]["ratio"] == 4 / 9
    # Natif g3 : 2 phrases distinctes sur 3.
    assert g3["natif"]["uniques"] == 2
    # Bornes R16 : g0/g1 non exposés, comptes seuls.
    assert carte["grains"][0]["natif"]["expose"] is False
    assert carte["grains"][0]["natif"]["N_unites"] == 18  # 9 tokens × 2 phonèmes
    assert carte["grains"][1]["natif"]["N_unites"] == 9   # 9 tokens × 1 syllabe
    # R43 : lecture A g0 uniforme (ADD×9) ⇒ DEGENERE-PAR-CONSTRUCTION.
    assert gm.statut_lecture_A(carte["grains"][0]["lecture_A"]["uniques"]) \
        == "DEGENERE-PAR-CONSTRUCTION"
    assert gm.statut_lecture_A(g2["lecture_A"]["uniques"]) is None


def test_carte_g4_exacte():
    cycles, n_sautees, n_err = gm.parser_cycles(
        [LIGNE_ABA, LIGNE_ABA_2, LIGNE_ABA, "", "<EOS>", "pas une ligne ABA"]
    )
    assert (len(cycles), n_sautees, n_err) == (3, 2, 1)
    g4 = gm.carte_g4(cycles)
    assert g4["n_cycles"] == 3
    # 3 cycles × 3 segments = 9 ; textes distincts : 2 A + 1 B + 1 A′ = 4.
    assert g4["segments"] == {"uniques": 4, "N": 9, "ratio": 4 / 9}
    # a-textes : « un signal apparaît » ×2, « une boucle se ferme » ×1.
    assert g4["a_textes"] == {"uniques": 2, "N": 3, "ratio": 2 / 3}
    assert g4["cycles"]["uniques"] == 2
    assert g4["formes"] == {"F0": 3}
    assert g4["operateurs"] == {"ADD": 3}


def test_gate_exact_faits_T67_et_alpha():
    # Fait FAIL connu (T67) : 19/1024 = 0,0186 — les deux branches mordent.
    fail = gm.evaluer_gate(19, 1024, seuil_ratio=0.05, seuil_uniques=512)
    assert fail["statut"] == "FAIL"
    assert abs(fail["G"] - 0.0185546875) < 1e-12
    # Fait PASS connu (α) : eve g2 natif 1605/6724 = 0,2387.
    ok = gm.evaluer_gate(1605, 6724, seuil_ratio=0.05, seuil_uniques=512)
    assert ok["statut"] == "PASS"
    assert abs(ok["G"] - 0.23869720404521118) < 1e-12
    # Plancher absolu : 512 uniques passent même sous le ratio.
    assert gm.evaluer_gate(512, 100000, 0.05, 512)["statut"] == "PASS"
    assert gm.evaluer_gate(511, 100000, 0.05, 512)["statut"] == "FAIL"
    # Frontière du ratio : exactement 0,05 ⇒ PASS (≥, pas >).
    assert gm.evaluer_gate(5, 100, 0.05, 512)["statut"] == "PASS"
    assert gm.evaluer_gate(4, 100, 0.05, 512)["statut"] == "FAIL"


# ── (3) bornes / erreurs ────────────────────────────────────────────────────

def test_lecteur_borne_deux_cotes(tmp_path):
    p = tmp_path / "cinq_lignes.txt"
    p.write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")
    # Côté tronqué : 5 lignes, borne 3 ⇒ 3 lues.
    lignes, n_lues = gm.lire_lignes_bornees(str(p), 3)
    assert n_lues == 3 and lignes == ["l1\n", "l2\n", "l3\n"]
    # Côté non atteint : borne 10 ⇒ 5 lues (la borne ne fabrique rien).
    lignes, n_lues = gm.lire_lignes_bornees(str(p), 10)
    assert n_lues == 5 and len(lignes) == 5
    # Sans borne (corpus α/INFO seulement).
    lignes, n_lues = gm.lire_lignes_bornees(str(p), None)
    assert n_lues == 5
    # Borne exactement atteinte.
    _, n_lues = gm.lire_lignes_bornees(str(p), 5)
    assert n_lues == 5


def test_erreurs_declarees(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("a\n", encoding="utf-8")
    with pytest.raises(ValueError):
        gm.lire_lignes_bornees(str(p), 0)
    with pytest.raises(ValueError):
        gm.lire_lignes_bornees(str(p), -1)
    with pytest.raises(ValueError):
        gm.evaluer_gate(5, 0, 0.05, 512)
    with pytest.raises(ValueError):
        gm.evaluer_gate(11, 10, 0.05, 512)
    with pytest.raises(ValueError):
        gm.evaluer_gate(5, 10, 0.0, 512)
    with pytest.raises(ValueError):
        gm.evaluer_gate(5, 10, 0.05, 0)
    with pytest.raises(ValueError):
        gm.carte_texte(["une phrase"], [])


def test_phrases_non_vides():
    phrases, n_vides = gm.phrases_non_vides(["  a \n", "\n", "b\n", "   "])
    assert phrases == ["a", "b"]
    assert n_vides == 2


# ── (4) parité α (route indépendante ; skip propre sans .so) ────────────────

_EVE = os.path.join(
    os.path.dirname(__file__), "..", "..", "corpus_eve_clean.txt"
)


@pytest.mark.skipif(not os.path.exists(_EVE), reason="corpus eve absent")
def test_parite_alpha_trois_chiffres_geles():
    """Les 3 chiffres gelés T70 (émission §9-P1), recomptés par CE module."""
    try:
        tok, struct = gm.charger_tokenizer()
    except TokenizerUnavailable:
        pytest.skip("tokenizer natif indisponible (pas de .so à côté)")
    lignes, _ = gm.lire_lignes_bornees(os.path.abspath(_EVE), None)
    phrases, _ = gm.phrases_non_vides(lignes)
    phrases_tokens = [
        gm.tokeniser_grains(tok, struct, ph)[0] for ph in phrases
    ]
    carte = gm.carte_texte(phrases, phrases_tokens)
    assert carte["n_phrases"] == 877
    assert carte["n_tokens"] == 6724
    g2 = carte["grains"][2]
    assert g2["natif"]["uniques"] == 1605
    assert abs(g2["natif"]["ratio"] - 0.238697) < 1e-6
    g1 = carte["grains"][1]
    assert g1["lecture_B"]["uniques"] == 483
    assert g1["lecture_B"]["N"] == 877
    g3 = carte["grains"][3]
    assert g3["lecture_A"]["uniques"] == 4
    assert g3["lecture_A"]["counts"]["COMPRESSION"] == 2800
