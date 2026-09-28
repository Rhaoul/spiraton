"""Quatuor de tests de l'instrument Stab contextuel projeté (T77).

(1) formules exactes sur témoins dérivés à la main (Var ddof=1, Δ_ctx, p_emp
    sur mini-pool, λ IQR),
(2) invariants sur mini-corpus (héritage T1, T2 deux branches ddof=1/ddof=0,
    chevauchement des paires, sélecteurs),
(3) gardes mordantes (n_occ < 2, IQR = 0, sha mismatch, pool vide,
    tranche vide/ABI),
(4) déterminisme ×2 bit-identique du pipeline complet sur mini-corpus
    (graine gelée).

Tout est pur Python/numpy sauf le test de garde sha (sauté si le ``.so`` est
indisponible — le pilote natif est testé par les portes de l'émission).
Aucune source d'aléa hors graines gelées 77201/77003.
"""
import json
from collections import OrderedDict

import numpy as np
import pytest

from spiraton.experimental.bpe_logos import TABLE_TEMOIN, CacheStab
from spiraton.experimental.stab_ctx import (
    GRAINE_SHUFFLE,
    GRAINE_TEMOINS,
    admissibilite_coherence_full,
    delta_ctx_paires,
    extraire_projection,
    info_types_mots_hotes,
    portees_t77,
    recette_lambda,
    stats_grain_projete,
    temoin_theoreme_t1,
    temoin_theoreme_t2,
    var_ctx_proj,
    verdict_tenue,
)

# --- Mini-corpus forcé (valeurs choisies à la main, dyadiques) --------------


def tk(text, ctx, g0_ids, g1_nb):
    v = np.zeros(33, dtype=np.float32)
    v[28], v[29], v[30] = ctx
    return {"text": text, "vector33d": v, "g0_ids": list(g0_ids), "g1_nb": list(g1_nb)}


def mini_corpus():
    """3 contextes : « aba » ×2 (vecteurs distincts), « aa », un token muet.

    g0 : 0 = a, 1 = b. « aba » = [0, 1, 0] (1 syllabe), « aa » = [0, 0].
    """
    return [
        [tk("aba", (1.0, 2.0, 3.0), [0, 1, 0], [3])],
        [tk("aba", (5.0, 2.0, 3.0), [0, 1, 0], [3]), tk("x", (9.0, 9.0, 9.0), [], [])],
        [tk("aa", (1.0, 2.0, 3.0), [0, 0], [2])],
    ]


# ---------------------------------------------------------------------------
# (1) Formules exactes — témoins dérivés à la main
# ---------------------------------------------------------------------------


def test_var_ctx_proj_formule_exacte():
    # dims : (1,3) -> var ddof1 = 2 ; (2,2) -> 0 ; (3,1) -> 2 ; moyenne = 4/3.
    mat = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]
    assert var_ctx_proj(mat) == (2.0 + 0.0 + 2.0) / 3.0
    # ddof=1 à la main sur n = 3 : valeurs 0, 3, 3 -> mean 2, SS = 6, /2 = 3.
    mat3 = [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
    assert var_ctx_proj(mat3) == 3.0
    # Aveugle à l'ordre (déclaré §7).
    assert var_ctx_proj(mat[::-1]) == var_ctx_proj(mat)


def test_delta_ctx_formule_exacte():
    # Hôtes : h0 = 0, h1 = 10, h2 = h3 = 5 (constants sur les 3 dims).
    pool = np.asarray(
        [[0.0] * 3, [10.0] * 3, [5.0] * 3, [5.0] * 3], dtype=np.float64
    )
    g0 = OrderedDict([(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3])])
    # Var(a) = Var(b) : valeurs 0, 10, 5, 5 -> mean 5, SS = 50, ddof1 = 50/3.
    paires = OrderedDict([((0, 1), [2, 3])])  # la paire vit dans les hôtes = 5
    res = delta_ctx_paires(paires, g0, pool)
    (_, delta), = res["deltas"]
    assert delta == 0.5 * (50.0 / 3.0 + 50.0 / 3.0) - 0.0
    assert res["n_pos"] == 1 and res["n_neg"] == 0 and res["n_zero"] == 0
    # Branche négative : la paire vit dans les hôtes extrêmes (var = 50).
    res2 = delta_ctx_paires(OrderedDict([((0, 1), [0, 1])]), g0, pool)
    (_, delta2), = res2["deltas"]
    assert delta2 == 50.0 / 3.0 - 50.0
    assert res2["n_neg"] == 1


def test_p_emp_formule_et_bornes():
    # Association parfaite : V_obs = 0, aucun shuffle ne fait mieux que > 0
    # (types monochromes impossibles à majorité) -> p_emp = 1/(R+1).
    unites = OrderedDict((f"t{t}", [4 * t + j for j in range(4)]) for t in range(10))
    pool = np.asarray(
        [[float(t)] * 3 for t in range(10) for _ in range(4)], dtype=np.float64
    )
    res = verdict_tenue(unites, pool, graine=GRAINE_SHUFFLE, repl=20)
    assert res["v_med_observee"] == 0.0
    assert res["p_emp"] == (res["n_shuffle_inferieurs_ou_egaux_obs"] + 1) / 21
    assert res["p_emp"] == 1 / 21
    # Anti-association : V_obs est le maximum atteignable -> p_emp = 1.
    unites2 = OrderedDict((f"t{t}", [2 * t, 2 * t + 1]) for t in range(10))
    pool2 = np.asarray(
        [[1.0] * 3 if j == 0 else [-1.0] * 3 for _ in range(10) for j in range(2)],
        dtype=np.float64,
    )
    res2 = verdict_tenue(unites2, pool2, graine=GRAINE_SHUFFLE, repl=20)
    assert res2["v_med_observee"] == 2.0
    assert res2["p_emp"] == 1.0


def test_recette_lambda_iqr_exacts():
    # Var par type : 0.5, 4.5, 12.5 -> q25 = 2.5, q75 = 8.5, IQR = 6 exact.
    pool = np.asarray(
        [[0.0] * 3, [1.0] * 3, [0.0] * 3, [3.0] * 3, [0.0] * 3, [5.0] * 3],
        dtype=np.float64,
    )
    paires = OrderedDict([((0, 1), [0, 1]), ((0, 2), [2, 3]), ((1, 2), [4, 5])])
    cache = CacheStab(TABLE_TEMOIN)
    res = recette_lambda(paires, pool, cache)
    assert res["iqr_var_ctx_proj"] == 6.0
    # IQR(H_norm) recalculé indépendamment sur les mêmes identités.
    h = sorted(cache.h_norm(i) for i in [(0, 1), (0, 2), (1, 2)])
    q25, q75 = np.quantile(h, [0.25, 0.75])
    assert res["iqr_h_norm"] == float(q75) - float(q25)
    assert res["statut"] == "CALIBRE"
    assert res["lambda_star"] == res["iqr_h_norm"] / 6.0


def test_admissibilite_coherence_full_exacte():
    cache = CacheStab(TABLE_TEMOIN)
    deltas = [((0, 1), 2.0), ((2, 2), -1.0)]
    res = admissibilite_coherence_full(deltas, cache, 0.5, TABLE_TEMOIN)
    l0 = next(l for l in res["top_paires"] if l["ident"] == [0, 1])
    assert l0["coherence_full"] == cache.coherence((0,), (1,)) + 0.5 * 2.0
    assert res["n_paires"] == 2


# ---------------------------------------------------------------------------
# (2) Invariants sur mini-corpus
# ---------------------------------------------------------------------------


def test_heritage_t1_et_chevauchement():
    proj = extraire_projection(mini_corpus())
    # T1 : toutes les sous-unités d'un hôte partagent le même indice d'hôte.
    # « aa » (hôte 2) : paire (0,0) -> 1 occurrence ; « aba » : (0,1) et (1,0).
    assert proj["paire"][(0, 1)] == [0, 1]
    assert proj["paire"][(1, 0)] == [0, 1]
    assert proj["paire"][(0, 0)] == [2]
    # Chevauchement : « aaa » = 2 occurrences de (0,0) dans le même hôte.
    t1 = temoin_theoreme_t1()
    assert t1["chevauchement_2_par_hote"]
    assert t1["tenu"] and t1["var_ctx_proj"] == 0.0
    # Héritage : le vecteur de (0,1) est bit-identique à celui de son hôte.
    assert proj["pool_bytes"][0] == proj["pool"][0].tobytes()


def test_theoreme_t2_deux_branches():
    t2 = temoin_theoreme_t2(graine=GRAINE_TEMOINS)
    assert t2["n_tirages_total"] >= 1000
    assert t2["branche_ddof1_tenue"]          # E[s²_ddof1] = S² pool, exact
    assert t2["branche_ddof0_biais_visible"]  # contrôle positif (n−1)/n
    assert t2["tenu"]


def test_selecteurs_et_comptes():
    proj = extraire_projection(mini_corpus())
    c = proj["comptes"]
    assert c["n_contextes"] == 3
    assert c["n_tokens"] == 4
    assert c["n_tokens_muets"] == 1          # « x » : g0_len = 0, pas un hôte
    assert c["n_hotes"] == 3
    assert c["n_occ_paires"] == 5            # 2 + 2 + 1
    assert c["n_occ_g0"] == 8                # 3 + 3 + 2
    assert c["n_occ_syllabes"] == 3
    carte = stats_grain_projete(proj["paire"], proj["pool"], proj["pool_bytes"])
    # (0,1) et (1,0) mesurables (n = 2) ; (0,0) creuse (n = 1) : comptes des
    # deux côtés de la sélection.
    assert carte["n_types"] == 3
    assert carte["n_mesurables"] == 2
    assert carte["n_creuses"] == 1
    assert carte["masse_occ_couverte"] == 4 / 5
    assert carte["regime_h62"] == "CREUX"    # n_mesurables < 30
    # Les deux colonnes ddof1/ddof0, jamais mélangées : ddof1 = 2×ddof0 à n=2.
    assert carte["var_ctx_proj_ddof1"]["mediane"] == pytest.approx(
        2.0 * carte["INFO_continuite_t73_ddof0"]["mediane"]
    )


def test_info_types_mots_hotes():
    proj = extraire_projection(mini_corpus())
    info = info_types_mots_hotes(proj["paire"], proj["hote_mot"])
    # (0,1) et (1,0) vivent dans 2 hôtes du même type « aba » -> 1 type chacun.
    assert info["n_paires_mesurables"] == 2
    assert info["mediane_types_mots_hotes"] == 1.0


def test_portees_deux_branches_par_instrument():
    p = portees_t77()
    assert p["var_ctx_proj"]["branche_zero_bit"]["atteinte"]
    assert p["var_ctx_proj"]["branche_positive"]["atteinte"]
    assert p["var_ctx_proj"]["aveugle_a_l_ordre"]
    assert p["classifieur_h62"]["atteintes"]
    assert p["shuffle_p_emp"]["branche_association"]["atteinte"]
    assert p["shuffle_p_emp"]["branche_anti_association"]["atteinte"]
    assert p["shuffle_p_emp"]["determinisme_x2_p_emp"]
    assert p["recette_lambda"]["branche_calculable"]["atteinte"]
    assert p["recette_lambda"]["branche_iqr_zero"]["atteinte"]
    assert p["delta_ctx"]["branche_positive"]["atteinte"]
    assert p["delta_ctx"]["branche_negative"]["atteinte"]
    assert p["theoreme_T1"]["tenu"]
    assert p["theoreme_T2"]["tenu"]


# ---------------------------------------------------------------------------
# (3) Gardes mordantes
# ---------------------------------------------------------------------------


def test_garde_n_occ_inferieur_2():
    with pytest.raises(ValueError, match="creuse"):
        var_ctx_proj([[1.0, 2.0, 3.0]])


def test_garde_matrice_degeneree():
    with pytest.raises(ValueError):
        var_ctx_proj(np.zeros((3, 0)))


def test_garde_iqr_zero_lambda():
    pool = np.zeros((6, 3), dtype=np.float64)
    paires = OrderedDict([((0, 1), [0, 1]), ((0, 2), [2, 3]), ((1, 2), [4, 5])])
    res = recette_lambda(paires, pool, CacheStab(TABLE_TEMOIN))
    assert res["lambda_star"] is None
    assert "NON-MESURE" in res["statut"]


def test_garde_pool_vide_et_aucun_mesurable():
    with pytest.raises(ValueError, match="vide"):
        verdict_tenue(OrderedDict([("u", [0, 1])]), np.zeros((0, 3)))
    with pytest.raises(ValueError, match="mesurable"):
        verdict_tenue(OrderedDict([("u", [0])]), np.ones((2, 3)))
    with pytest.raises(ValueError, match="mesurable"):
        recette_lambda(OrderedDict(), np.ones((2, 3)), CacheStab(TABLE_TEMOIN))


def test_garde_abi_drop_negatif():
    tokens = [[tk("bad", (1.0, 1.0, 1.0), [0], [2])]]  # Σ g1_nb > g0_len
    with pytest.raises(ValueError, match="ABI"):
        extraire_projection(tokens)


def test_garde_tranche_vide_comptee():
    tokens = [[tk("t", (1.0, 1.0, 1.0), [0, 1], [0, 2])]]
    proj = extraire_projection(tokens)
    assert proj["comptes"]["n_syllabes_tranche_vide"] == 1
    assert proj["comptes"]["n_occ_syllabes"] == 1


def test_garde_sha_mismatch_mordante():
    stab_grains = pytest.importorskip("spiraton.experimental.stab_grains")
    try:
        stab_grains.charger_tokenizer_garde(sha256_attendu="0" * 64)
    except RuntimeError:
        return  # garde mordante : c'est la branche attendue
    except Exception:
        pytest.skip("tokenizer natif indisponible dans cet environnement")
    pytest.fail("la garde sha n'a pas mordu sur un sha faux")


# ---------------------------------------------------------------------------
# (4) Déterminisme ×2 au bit du pipeline complet (graine gelée)
# ---------------------------------------------------------------------------


def _pipeline_mini():
    proj = extraire_projection(mini_corpus())
    cache = CacheStab(TABLE_TEMOIN)
    carte = stats_grain_projete(proj["paire"], proj["pool"], proj["pool_bytes"])
    tenue = verdict_tenue(proj["paire"], proj["pool"], graine=GRAINE_SHUFFLE, repl=25)
    lam = recette_lambda(proj["paire"], proj["pool"], cache)
    d = delta_ctx_paires(proj["paire"], proj["g0"], proj["pool"])
    deltas = d.pop("deltas")
    adm = (
        admissibilite_coherence_full(deltas, cache, lam["lambda_star"], TABLE_TEMOIN)
        if lam["lambda_star"] is not None
        else {"statut": "NON-MESURE"}
    )
    return json.dumps(
        {"carte": carte, "tenue": tenue, "lambda": lam, "delta": d, "adm": adm},
        ensure_ascii=False,
        sort_keys=True,
    )


def test_determinisme_x2_pipeline_complet():
    assert _pipeline_mini() == _pipeline_mini()
