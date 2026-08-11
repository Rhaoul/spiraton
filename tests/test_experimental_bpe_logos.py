"""Quatuor de tests du BPE-Logos minimal (T75).

(1) formule exacte sur témoins dérivés à la main (σ_unit, Coherence dont A1/A2,
    Jaccard 0/1/double-vide, profondeur k),
(2) invariants sur mini-corpus forcé (déterminisme de la boucle, frontières
    intra-mot, décroissance stricte, ex æquo lexicographique),
(3) gardes mordantes (MAX_FUSIONS, invariant violé par injection, tranche
    vide, drop < 0 hérité, mur temporel, S = 0),
(4) déterminisme ×2 bit-identique du run complet sur mini-corpus.

Tout est pur Python/numpy : aucun ``.so`` requis (le pilote natif est exercé
par les portes de l'émission, pas ici). AUCUN tirage aléatoire : le tour est
déterministe par construction (émission §4.5).
"""
import json
import math

import numpy as np
import pytest

from spiraton.experimental.bpe_logos import (
    MAX_FUSIONS,
    TABLE_TEMOIN,
    THETA_FUSION,
    CacheStab,
    appliquer_fusion,
    audit_frequence,
    construire_etat,
    coupures_internes,
    enumerer_paires,
    executer_fusions,
    jaccard_distance,
    part_fusion_totale,
    portees_instruments,
    rangs_moyens,
    sigma_unit,
    spearman_rho,
    torsion_corpus,
    vitalite_premier_passage,
    vocabulaire_final,
)

# --- Mini-corpus forcé (tokens fabriqués à la main, interface R16) ----------


def tk(text, g0_ids, g1_nb):
    return {"text": text, "g0_ids": list(g0_ids), "g1_nb": list(g1_nb)}


def mini_corpus():
    """« abab » ×2 et « ab » ×1 : une seule fusion attendue ((0,), (1,))."""
    return [
        [tk("abab", [0, 1, 0, 1], [2, 2]), tk("ab", [0, 1], [2])],
        [tk("abab", [0, 1, 0, 1], [2, 2])],
    ]


# ---------------------------------------------------------------------------
# (1) Formule exacte sur témoins dérivés à la main
# ---------------------------------------------------------------------------


def test_sigma_unit_concatenation_exacte():
    """σ_unit = concat ordonnée des 4 constantes par phonème (§4.1)."""
    sig = sigma_unit((0, 1), TABLE_TEMOIN)
    assert sig.tolist() == [1.0, 1.0, 1.0, 1.0, 8.0, 8.0, 8.0, 8.0]
    assert sig.dtype == np.float64
    with pytest.raises(ValueError, match="identité vide"):
        sigma_unit((), TABLE_TEMOIN)


def test_coherence_temoin_recalcul_independant():
    """Coherence((0,),(1,)) recalculée par une boucle indépendante (math.log)."""
    cache = CacheStab(TABLE_TEMOIN)

    def h_norm_ref(sig):
        s = sum(abs(x) for x in sig)
        h = -sum((abs(x) / s) * math.log(abs(x) / s, 2.0) for x in sig if x != 0.0)
        return h / math.log2(len(sig))

    attendu = -h_norm_ref([1.0] * 4 + [8.0] * 4) + 0.5 * (
        h_norm_ref([1.0] * 4) + h_norm_ref([8.0] * 4)
    )
    obtenu = cache.coherence((0,), (1,))
    assert obtenu == pytest.approx(attendu, abs=1e-12)
    assert obtenu > 0.0  # branche positive atteinte (uniformes distincts)


def test_propriete_A1_identite_exacte_dyadique():
    """A1 : H(a·a) = ½H(a) + ½H(a) + H2(½) — EXACTE au bit (témoin dyadique)."""
    cache = CacheStab(TABLE_TEMOIN)
    assert cache.h_brut((0, 0)) == 0.5 * cache.h_brut((0,)) * 2 + 1.0


def test_propriete_A2_deux_branches():
    """A2 : Coherence(a,a) ≤ 0 ; = 0 EXACT ssi σ(a) uniforme (§4.2)."""
    cache = CacheStab(TABLE_TEMOIN)
    assert cache.coherence((2,), (2,)) < 0.0        # non uniforme ⇒ strictement < 0
    assert cache.coherence((3,), (3,)) == 0.0       # uniforme ⇒ zéro exact
    # avec θ = 0 et test STRICT >, l'auto-paire uniforme ne fusionne pas.
    assert not (cache.coherence((3,), (3,)) > THETA_FUSION)


def test_jaccard_bornes_et_double_vide():
    """Jaccard : 0 (identiques), 1 (disjoints), double-vide → 0 compté (§4.6)."""
    assert jaccard_distance(frozenset({1, 3}), frozenset({1, 3})) == (0.0, False)
    assert jaccard_distance(frozenset({1}), frozenset({2})) == (1.0, False)
    assert jaccard_distance(frozenset(), frozenset()) == (0.0, True)
    d, _ = jaccard_distance(frozenset({1, 2}), frozenset({2, 3}))
    assert d == pytest.approx(1.0 - 1.0 / 3.0)


def test_coupures_internes_bornes_exclues():
    assert coupures_internes([2, 2], 4) == frozenset({2})
    assert coupures_internes([4], 4) == frozenset()
    assert coupures_internes([1, 2, 1], 4) == frozenset({1, 3})


def test_profondeur_k_max_plus_un():
    """k(fusion(a,b)) = max(k(a), k(b)) + 1 (§4.8) sur cascade forcée."""
    # (0,1) fusionne (k=1) ; puis ((0,1),(3,)) si positive — on force la
    # cascade avec un mot (0,1,3) et on lit k dans le journal.
    etat = construire_etat([[tk("abd", [0, 1, 3], [3])]])
    cache = CacheStab(TABLE_TEMOIN)
    r = executer_fusions(etat, cache)
    ks = {tuple(e["a"] + e["b"]): e["k"] for e in r["journal"]}
    assert ks[(0, 1)] == 1
    for ident, k in ks.items():
        if len(ident) == 3:
            assert k == 2  # fusion d'une unité k=1 avec un phonème k=0
    assert r["n_conflits_k"] == 0


def test_rangs_moyens_et_spearman_exacts():
    """Spearman : ±1 exacts sur listes monotones, rangs moyens pour ex æquo."""
    assert spearman_rho([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == 1.0
    assert spearman_rho([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == -1.0
    # ex æquo : [1, 1, 2] → rangs [1.5, 1.5, 3]
    assert rangs_moyens([1.0, 1.0, 2.0]).tolist() == [1.5, 1.5, 3.0]
    # branche NON-MESURE : n < 5 (§4.7)
    assert spearman_rho([1.0, 2.0], [2.0, 1.0]) is None
    # branche DEGENERE : classement constant
    assert spearman_rho([1.0] * 5, [1, 2, 3, 4, 5]) is None


# ---------------------------------------------------------------------------
# (2) Invariants sur mini-corpus forcé
# ---------------------------------------------------------------------------


def test_construction_etat_types_et_multiplicites():
    etat = construire_etat(mini_corpus())
    assert list(etat.types.keys()) == [(0, 1, 0, 1), (0, 1)]
    assert etat.types[(0, 1, 0, 1)]["mult"] == 2
    assert etat.types[(0, 1)]["mult"] == 1
    assert etat.comptes["n_tokens"] == 3
    assert etat.comptes["n_types_g1_variable"] == 0
    assert etat.total_unites_occurrences() == 2 * 4 + 1 * 2


def test_frequences_ponderees_par_multiplicite():
    """Fréquence de paire = occurrences adjacentes × multiplicité (§5a)."""
    paires = enumerer_paires(construire_etat(mini_corpus()))
    # (0,1) : 2 par « abab » (×2) + 1 par « ab » = 5 ; (1,0) : 1 par « abab » (×2) = 2.
    assert paires[((0,), (1,))] == 5
    assert paires[((1,), (0,))] == 2


def test_fusion_intra_mot_seulement():
    """Les frontières de mot sont infranchissables : deux mots adjacents
    « ...a » + « b... » ne produisent AUCUNE paire inter-tokens (§4.3)."""
    etat = construire_etat([[tk("a", [0], [1]), tk("b", [1], [1])]])
    assert enumerer_paires(etat) == {}
    r = executer_fusions(etat, CacheStab(TABLE_TEMOIN))
    assert r["n_fusions"] == 0 and r["arret"] == "ARRET_CRITERE"


def test_ex_aequo_lexicographique():
    """(0,1) et (1,0) ont la MÊME Coherence (entropie invariante par
    permutation) : l'ex æquo est tranché par l'ordre lexicographique — la
    paire ((0,),(1,)) gagne, jamais la fréquence (§4.5)."""
    cache = CacheStab(TABLE_TEMOIN)
    assert cache.coherence((0,), (1,)) == cache.coherence((1,), (0,))
    # corpus où (1,0) est PLUS fréquente : la lexicographie doit primer.
    etat = construire_etat(
        [[tk("ba", [1, 0], [2]), tk("ba", [1, 0], [2]), tk("ab", [0, 1], [2])]]
    )
    r = executer_fusions(etat, cache)
    premiere = (tuple(r["journal"][0]["a"]), tuple(r["journal"][0]["b"]))
    assert premiere == ((0,), (1,))


def test_run_mini_corpus_valeurs_exactes():
    """Run complet dérivé à la main : 1 fusion, arrêt critère, décroissance."""
    etat = construire_etat(mini_corpus())
    cache = CacheStab(TABLE_TEMOIN)
    avant = etat.total_unites_occurrences()  # 10
    r = executer_fusions(etat, cache)
    assert r["arret"] == "ARRET_CRITERE"
    # (0,1) fusionne ; ((0,1),(0,1)) est une auto-paire non uniforme (A2 < 0)
    # et (1,0) vaut coherence((0,),(1,)) > 0… mais après fusion de (0,1)
    # partout, la paire (1,0) n'existe plus : abab → [(01),(01)].
    assert r["n_fusions"] == 1
    j = r["journal"][0]
    assert (tuple(j["a"]), tuple(j["b"])) == ((0,), (1,))
    assert j["frequence"] == 5 and j["k"] == 1
    assert etat.types[(0, 1, 0, 1)]["unites"] == [(0, 1), (0, 1)]
    assert etat.types[(0, 1)]["unites"] == [(0, 1)]
    assert etat.total_unites_occurrences() == avant - 5
    # P-1 : « ab » (g0_len 2) réduit à une unité ; « abab » non.
    p1 = part_fusion_totale(etat)
    assert p1 == {"n_types_g0len2plus": 2, "n_reduits_a_une_unite": 1, "part": 0.5}


def test_vitalite_c2_deux_branches():
    """C2 : n_pos ≥ 1 ET n_neg ≥ 1 sur un état construit pour couper."""
    # (0,1) positive ; (2,2) auto-paire non uniforme négative.
    etat = construire_etat([[tk("ab", [0, 1], [2]), tk("cc", [2, 2], [2])]])
    v = vitalite_premier_passage(etat, CacheStab(TABLE_TEMOIN))
    assert v["n_pos"] >= 1 and v["n_neg"] >= 1 and v["coupe"] is True
    assert v["n_paires_types"] == 2


def test_torsion_mini_corpus_derivee_a_la_main():
    """Torsion : « abab » (B_g1 = {2}) et « ab » (double-vide) après fusion."""
    etat = construire_etat(mini_corpus())
    executer_fusions(etat, CacheStab(TABLE_TEMOIN))
    t = torsion_corpus(etat)
    assert t["n_types_mesurables"] == 2
    # abab : B_g1' = {2} (unités (01)(01)) ⇒ distance 0 ; ab : double-vide ⇒ 0.
    assert t["torsion_mediane"] == 0.0 and t["n_double_vide"] == 1
    assert t["mesurable"] is False and t["classement"] == "NON-MESURE"  # < 30 types
    # unités finales len ≥ 2 : (01)(01) et (01) = 3 ; aucune ne TRAVERSE
    # ({2} est une borne d'unité, jamais strictement interne).
    assert t["n_unites_len2plus"] == 3 and t["n_unites_traversee"] == 0
    cl = t["classification_unites"]
    assert cl["n_unites_classees"] == 3
    assert cl["intra_syllabique"] == 3 and cl["intra_syllabique_exacte_syllabe"] == 3
    assert cl["traversante"] == 0 and cl["multi_syllabique_alignee"] == 0


def test_audit_frequence_statuts():
    journal = [
        {"coherence": 0.5, "frequence": 5},
        {"coherence": 0.4, "frequence": 4},
        {"coherence": 0.3, "frequence": 3},
        {"coherence": 0.2, "frequence": 2},
        {"coherence": 0.1, "frequence": 1},
    ]
    a = audit_frequence(journal)
    assert a["rho_spearman"] == 1.0 and a["statut"] == "I6"
    b = audit_frequence(journal[:3])
    assert b["rho_spearman"] is None and "NON-MESURE" in b["statut"]


def test_vocabulaire_final_et_k():
    etat = construire_etat(mini_corpus())
    r = executer_fusions(etat, CacheStab(TABLE_TEMOIN))
    v = vocabulaire_final(etat, r["profondeurs"], TABLE_TEMOIN)
    # vocabulaire final : la seule unité présente est (0,1), k = 1.
    assert v["taille_vocabulaire"] == 1 and v["n_types_fusionnes"] == 1
    assert v["mediane_k_types_fusionnes"] == 1.0
    [u] = v["unites"]
    assert u["ident"] == [0, 1] and u["rendu_ascii"] == "ab"
    assert u["n_occurrences"] == 5  # 2×2 (abab) + 1 (ab)


def test_portees_instruments_toutes_branches():
    """La routine de portées atteint les deux branches partout (C3)."""
    p = portees_instruments()
    assert p["coherence"]["temoin_positif"]["atteint"]
    assert p["coherence"]["temoin_negatif"]["atteint"]
    assert p["coherence"]["temoin_A2_egalite"]["exactement_zero"]
    assert p["propriete_A1"]["exact"]
    assert p["jaccard"] == {"borne_0": True, "borne_1": True, "double_vide_vaut_0": True}
    assert p["garde_max_fusions"]["mordante"]
    assert p["garde_invariant_decroissance"]["mordante_par_injection"]
    assert p["garde_mur_temporel"]["mordante"]
    assert p["spearman"]["plus_un_exact"] and p["spearman"]["moins_un_exact"]
    assert p["spearman"]["non_mesure_n_inferieur_5"]


# ---------------------------------------------------------------------------
# (3) Gardes mordantes
# ---------------------------------------------------------------------------


def test_garde_max_fusions_mordante():
    """MAX_FUSIONS artificiellement bas ⇒ I5, seulement s'il RESTE du travail."""
    # deux paires positives distinctes ((0,1) et (0,3)) ⇒ 2 fusions requises.
    etat = construire_etat([[tk("ab", [0, 1], [2]), tk("ad", [0, 3], [2])]])
    r = executer_fusions(etat, CacheStab(TABLE_TEMOIN), max_fusions=1)
    assert r["arret"] == "I5_MAX_FUSIONS" and r["n_fusions"] == 1
    # borne par défaut du tour intacte
    assert MAX_FUSIONS == 20_000


def test_garde_pas_de_faux_i5():
    """Si l'arrêt critère et la borne coïncident, l'arrêt critère PRIME."""
    etat = construire_etat(mini_corpus())  # une seule fusion possible
    r = executer_fusions(etat, CacheStab(TABLE_TEMOIN), max_fusions=1)
    assert r["arret"] == "ARRET_CRITERE" and r["n_fusions"] == 1


def test_garde_mur_temporel_mordante():
    import time

    etat = construire_etat(mini_corpus())
    r = executer_fusions(
        etat, CacheStab(TABLE_TEMOIN), mur_s=0.0, t0=time.perf_counter() - 1.0
    )
    assert r["arret"] == "I5_MUR_TEMPOREL" and r["n_fusions"] == 0


def test_garde_invariant_par_injection():
    """Fusionner une paire ABSENTE ⇒ zéro remplacement ⇒ ValueError (§4.4)."""
    etat = construire_etat(mini_corpus())
    with pytest.raises(ValueError, match="invariant de décroissance"):
        appliquer_fusion(etat, (2,), (3,))


def test_garde_drop_negatif_herite():
    """drop < 0 (Σ g1_nb > g0_len) : incohérence ABI, refus hérité T73."""
    with pytest.raises(ValueError, match="incohérence ABI"):
        construire_etat([[tk("x", [0], [2])]])


def test_garde_sigma_vide_et_tranche_vide():
    with pytest.raises(ValueError, match="identité vide"):
        sigma_unit((), TABLE_TEMOIN)
    # entropie_masse (réutilisée T73) garde la tranche vide — vérifié ici
    # à travers le chemin σ_unit → entropie_masse du module.
    from spiraton.experimental.stab_grains import entropie_masse

    with pytest.raises(ValueError, match="tranche vide"):
        entropie_masse([])


def test_garde_s_zero_dementi_du_support():
    """S = 0 (constante de table nulle partout) = démenti du §9 ⇒ garde."""
    table_nulle = [{"flux": 0.0, "structure": 0.0, "energie_totale": 0.0, "impedance": 0.0}]
    cache = CacheStab(table_nulle)
    with pytest.raises(ValueError, match="démenti du support"):
        cache.h_norm((0,))


def test_garde_spearman_longueurs():
    with pytest.raises(ValueError, match="longueurs différentes"):
        spearman_rho([1.0], [1.0, 2.0])


def test_garde_tokens_muets_et_len1_comptes():
    etat = construire_etat([[tk("", [], []), tk("a", [0], [1])]])
    assert etat.comptes["n_tokens_muets"] == 1
    assert etat.comptes["n_tokens_g0_len_1"] == 1
    assert list(etat.types.keys()) == [(0,)]  # le muet est exclu, le len-1 gardé


# ---------------------------------------------------------------------------
# (4) Déterminisme ×2 bit-identique du run complet sur mini-corpus
# ---------------------------------------------------------------------------


def _run_complet_json():
    etat = construire_etat(mini_corpus())
    cache = CacheStab(TABLE_TEMOIN)
    vital = vitalite_premier_passage(construire_etat(mini_corpus()), cache)
    r = executer_fusions(etat, cache)
    return json.dumps(
        {
            "vitalite": vital,
            "journal": r["journal"],
            "arret": r["arret"],
            "vocab": vocabulaire_final(etat, r["profondeurs"], TABLE_TEMOIN),
            "torsion": torsion_corpus(etat),
            "audit": audit_frequence(r["journal"]),
            "p1": part_fusion_totale(etat),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def test_determinisme_run_complet_bit_a_bit():
    """Deux exécutions complètes ⇒ JSON byte-identiques (C1, zéro aléa)."""
    a, b = _run_complet_json(), _run_complet_json()
    assert a == b
    import hashlib

    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()
