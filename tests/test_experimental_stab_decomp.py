"""Quatuor de tests de l'instrument de décomposition intra/inter (T78).

(1) formules exactes sur témoins dérivés à la main (T4 par calcul manuel,
    V_intra^{d1}, Δ_ctx_intra, p_emp sur mini-montage, ρ_S sur listes
    témoins),
(2) invariants (T6 invariance/mobilité, T5 deux branches ddof=1/ddof=0,
    strates singleton = 0 en intra, identité T4 par dim),
(3) gardes mordantes (GC-1, T6-pré, IQR = 0, pool vide, strates non
    partitionnantes, corrélation constante, sha mismatch),
(4) déterminisme ×2 au bit du pipeline complet sur mini-corpus.

Tout est pur Python/numpy sauf le test de garde sha (sauté si le ``.so``
est indisponible). Aucune source d'aléa hors graines gelées
78201/78202/78003.
"""
import json
from collections import OrderedDict

import numpy as np
import pytest

from spiraton.experimental.stab_decomp import (
    GRAINE_CONTROLE_STRATIFIE,
    GRAINE_NULLE_GLOBALE,
    GRAINE_TEMOINS,
    TOL_IDENTITE,
    TOL_INVARIANCE,
    NonMesureError,
    _temoin_decomposition_a,
    _temoin_decomposition_b,
    _mes_temoin,
    case_verdict,
    controle_stratifie,
    decomposition_observee,
    decomposition_pop,
    decomposition_pop_poids_uniformes,
    delta_ctx_intra_paires,
    nulle_globale_decomposition,
    permutation_stratifiee,
    portees_t78,
    preparer_mesurables,
    rangs_moyens_argsort,
    rangs_moyens_unique,
    recalcul_v_med_t77,
    recette_lambda_intra,
    spearman_deux_routes,
    strates_unite,
    t6_pre,
    temoin_theoreme_t5,
    v_intra_d1,
    verifier_gc1,
)
from spiraton.experimental.bpe_logos import TABLE_TEMOIN, CacheStab

# --- Témoins dérivés à la main (valeurs dyadiques, exactes en float) --------


def tk(text, ctx, g0_ids, g1_nb):
    v = np.zeros(33, dtype=np.float32)
    v[28], v[29], v[30] = ctx
    return {"text": text, "vector33d": v, "g0_ids": list(g0_ids), "g1_nb": list(g1_nb)}


def temoin_t4():
    """Pool 1-D {0, 2, 4, 10}, strates A = {0, 2} et B = {4, 10}.

    À la main : μ = 4 ; Var_pop = (16+4+0+36)/4 = 14 ; s²_A = 1, s²_B = 9 ⇒
    V_intra = 5 ; μ_A = 1, μ_B = 7 ⇒ V_inter = 0.5·9 + 0.5·9 = 9 ; 5+9 = 14.
    """
    mat = np.asarray([[0.0], [2.0], [4.0], [10.0]])
    groupes = strates_unite([0, 1, 2, 3], ["A", "A", "B", "B"])
    return mat, groupes


# --- (1) Formules exactes ---------------------------------------------------


def test_t4_calcul_manuel_exact():
    mat, groupes = temoin_t4()
    var, intra, inter, ecart = decomposition_pop(mat, groupes)
    assert var == 14.0
    assert intra == 5.0
    assert inter == 9.0
    assert ecart <= TOL_IDENTITE


def test_t4_branche_cassee_poids_uniformes():
    """Poids uniformes 1/M sur strates déséquilibrées ⇒ identité violée."""
    mat = np.asarray([[0.0], [1.0], [2.0], [3.0], [10.0]])
    groupes = strates_unite([0, 1, 2, 3, 4], ["A", "A", "A", "A", "B"])
    _, _, _, ecart = decomposition_pop_poids_uniformes(mat, groupes)
    assert ecart > 1e-6


def test_v_intra_d1_manuel_et_singleton_exclu():
    """Strates A = {0, 2} (s²_ddof1 = 2) et C = {4} singleton : S₂ = {A},
    N₂ = 2 ⇒ V_intra^{d1} = 2 ; le singleton est exclu du D1 (jamais 0)."""
    mat = np.asarray([[0.0], [2.0], [4.0]])
    groupes = strates_unite([0, 1, 2], ["A", "A", "C"])
    val, n_s2, n2 = v_intra_d1(mat, groupes)
    assert val == 2.0
    assert (n_s2, n2) == (1, 2)


def test_v_intra_d1_sans_strate_s2():
    mat = np.asarray([[0.0], [2.0]])
    groupes = strates_unite([0, 1], ["A", "B"])
    val, n_s2, n2 = v_intra_d1(mat, groupes)
    assert val is None and n_s2 == 0 and n2 == 0


def test_delta_ctx_intra_manuel():
    """Parts dispersées intra-strate, paire homogène ⇒ Δ_intra > 0 exact.

    V^{d1}(a) : strate p = {0, 8} ⇒ 32 ; V^{d1}(b) : strate q = {2, 6} ⇒ 8 ;
    V^{d1}(ab) : strate r = {5, 5} ⇒ 0 ⇒ Δ = ½(32+8) − 0 = 20.
    """
    hotes = ["p", "p", "q", "q", "r", "r"]
    pool = np.asarray(
        [[0.0, 0.0, 0.0], [8.0, 8.0, 8.0], [2.0, 2.0, 2.0], [6.0, 6.0, 6.0],
         [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    )
    res = delta_ctx_intra_paires(
        OrderedDict([((0, 1), [4, 5])]),
        OrderedDict([(0, [0, 1]), (1, [2, 3])]),
        hotes, pool,
    )
    assert res["deltas"] == [((0, 1), 20.0)]
    assert res["n_pos"] == 1 and res["n_neg"] == 0


def test_p_emp_mini_montage_deux_branches():
    """Témoin (a) : intra-tenue injectée ⇒ p_intra = 1/101, p_inter = 1 ;
    témoin (b) : montage inverse ⇒ p_intra = 1 (chaque case dérivable)."""
    unites_a, hotes_a, pool_a = _temoin_decomposition_a()
    nulle_a = nulle_globale_decomposition(
        _mes_temoin(unites_a, hotes_a), pool_a, graine=GRAINE_TEMOINS
    )
    assert nulle_a["v_obs_intra"] == 0.0
    assert nulle_a["p_intra"] == 1.0 / 101.0
    assert nulle_a["p_inter"] == 1.0
    unites_b, hotes_b, pool_b = _temoin_decomposition_b()
    nulle_b = nulle_globale_decomposition(
        _mes_temoin(unites_b, hotes_b), pool_b, graine=GRAINE_TEMOINS
    )
    assert nulle_b["p_intra"] == 1.0


def test_spearman_temoins_exacts():
    assert spearman_deux_routes([1, 2, 3], [10, 20, 90])["rho"] == 1.0
    assert spearman_deux_routes([1, 2, 3], [90, 20, 10])["rho"] == -1.0
    assert spearman_deux_routes([1, 2, 3, 4], [1, 2, 2, 1])["rho"] == 0.0


def test_rangs_moyens_ex_aequo_deux_routes():
    attendu = [1.0, 2.5, 2.5, 4.0]
    assert list(rangs_moyens_argsort([10.0, 20.0, 20.0, 30.0])) == attendu
    assert list(rangs_moyens_unique([10.0, 20.0, 20.0, 30.0])) == attendu


def test_case_verdict_table_complete():
    assert case_verdict(0.01, 0.99) == "DECOMPOSEE"
    assert case_verdict(0.99, 0.5) == "INVERSION"
    assert case_verdict(0.99, 0.99) == "INVERSION"
    assert case_verdict(0.5, 0.99) == "INTER-SEULE"
    assert case_verdict(0.01, 0.5) == "INTRA-SEULE"
    assert case_verdict(0.5, 0.5) == "NON-LOCALISEE"


# --- (2) Invariants ---------------------------------------------------------


def test_t6_invariance_strates_pleines():
    unites, hotes, pool = _temoin_decomposition_a()
    ctrl = controle_stratifie(
        _mes_temoin(unites, hotes), hotes, pool, graine=GRAINE_TEMOINS, repl=10
    )
    assert ctrl["invariance_tenue_1e-12"]
    assert ctrl["max_ecart_med_intra"] <= TOL_INVARIANCE


def test_t6_mobilite_strates_partielles():
    hotes = ["mx", "mx", "mx", "mx"]
    pool = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]]
    )
    ctrl = controle_stratifie(
        _mes_temoin([("u", [0, 1])], hotes), hotes, pool,
        graine=GRAINE_TEMOINS, repl=10,
    )
    assert max(ctrl["max_ecart_med_intra"], ctrl["max_ecart_med_totale"]) > 1e-6


def test_t5_deux_branches():
    t5 = temoin_theoreme_t5()
    assert t5["tenu"]
    assert t5["n_affectations_exhaustives"] == 560
    assert t5["ecart_ddof1"] <= TOL_IDENTITE
    assert t5["branche_ddof0_biais_visible"]


def test_strates_singleton_contribuent_zero_en_intra():
    """Unité aux strates toutes singleton : V_intra_pop = 0, V_inter = Var."""
    mat = np.asarray([[0.0], [2.0], [7.0]])
    groupes = strates_unite([0, 1, 2], ["A", "B", "C"])
    var, intra, inter, ecart = decomposition_pop(mat, groupes)
    assert intra == 0.0
    assert inter == var
    assert ecart <= TOL_IDENTITE


def test_permutation_stratifiee_conserve_les_types():
    hotes = ["a", "b", "a", "b", "a"]
    rng = np.random.default_rng(GRAINE_TEMOINS)
    perm = permutation_stratifiee(hotes, rng)
    assert sorted(perm.tolist()) == [0, 1, 2, 3, 4]
    for h, p in enumerate(perm):
        assert hotes[h] == hotes[p]


def test_t6_pre_zero_violation_sur_textes_constants():
    pre = t6_pre([[tk("aa", (1, 2, 3), [0, 0], [2])],
                  [tk("AA", (4, 5, 6), [0, 0], [2])]])
    assert pre["n_types_mots"] == 1
    assert pre["n_violations"] == 0 and pre["tenu"]


# --- (3) Gardes mordantes ---------------------------------------------------


def test_gc1_mordante_valeur_et_compte():
    paires = OrderedDict([((0, 1), [0, 1]), ((1, 2), [2, 3])])
    pool = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
    )
    v_med, n = recalcul_v_med_t77(paires, pool)
    assert verifier_gc1(paires, pool, v_med, n)["egalite_exacte"]
    with pytest.raises(NonMesureError):
        verifier_gc1(paires, pool, v_med + 1e-9, n)
    with pytest.raises(NonMesureError):
        verifier_gc1(paires, pool, v_med, n + 1)


def test_t6_pre_detecte_la_violation():
    pre = t6_pre([[tk("aa", (1, 2, 3), [0, 0], [2])],
                  [tk("Aa", (4, 5, 6), [0, 1], [2])]])
    assert pre["n_violations"] == 1
    assert pre["violations"][0]["hote_mot"] == "aa"
    assert pre["violations"][0]["sequences_g0"] == [(0, 0), (0, 1)]


def test_recette_lambda_intra_gardes():
    cache = CacheStab(TABLE_TEMOIN)
    assert recette_lambda_intra(
        [((0, 1), 0.5), ((0, 2), 0.5)], cache
    )["lambda_star_intra"] is None
    with pytest.raises(ValueError):
        recette_lambda_intra([], cache)


def test_decomposition_pop_gardes():
    with pytest.raises(ValueError):
        decomposition_pop(np.zeros((1, 3)), [np.asarray([0])])
    with pytest.raises(ValueError):
        decomposition_pop(
            np.zeros((3, 3)), strates_unite([0, 1], ["A", "A"])
        )


def test_nulle_globale_gardes():
    with pytest.raises(ValueError):
        nulle_globale_decomposition([], np.zeros((4, 3)), graine=GRAINE_TEMOINS)
    with pytest.raises(ValueError):
        nulle_globale_decomposition(
            _mes_temoin([("u", [0, 1])], ["a", "a"]),
            np.zeros((0, 3)), graine=GRAINE_TEMOINS,
        )


def test_spearman_constant_refuse():
    with pytest.raises(ValueError):
        spearman_deux_routes([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_garde_sha_so_mordante():
    bridge = pytest.importorskip("spiraton.data.tokenizer_bridge")
    try:
        bridge.load_native_tokenizer()
    except Exception:
        pytest.skip("bibliothèque native indisponible")
    from spiraton.experimental.stab_grains import charger_tokenizer_garde

    with pytest.raises(RuntimeError):
        charger_tokenizer_garde(sha256_attendu="0" * 64)


# --- (4) Déterminisme ×2 au bit du pipeline complet -------------------------


def mini_corpus():
    """3 contextes ; « ab » ×4 (2 vecteurs distincts), « ba » ×2, « cc » ×2.

    Deux types de mots-hôtes par paire mesurable — strates non triviales.
    """
    return [
        [tk("ab", (1.0, 2.0, 3.0), [0, 1], [2]),
         tk("ba", (2.0, 2.0, 2.0), [1, 0], [2])],
        [tk("ab", (5.0, 2.0, 3.0), [0, 1], [2]),
         tk("cc", (0.0, 0.0, 0.0), [2, 2], [2])],
        [tk("ab", (1.0, 2.0, 3.0), [0, 1], [2]),
         tk("ab", (3.0, 0.0, 1.0), [0, 1], [2]),
         tk("ba", (4.0, 4.0, 4.0), [1, 0], [2]),
         tk("cc", (2.0, 0.0, 2.0), [2, 2], [2])],
    ]


def pipeline_mini():
    from spiraton.experimental.stab_ctx import extraire_projection

    proj = extraire_projection(mini_corpus())
    pre = t6_pre(mini_corpus())
    mes = preparer_mesurables(proj["paire"], proj["hote_mot"])
    obs = decomposition_observee(mes, proj["pool"])
    nulle = nulle_globale_decomposition(
        mes, proj["pool"], graine=GRAINE_NULLE_GLOBALE, repl=25
    )
    ctrl = controle_stratifie(
        mes, proj["hote_mot"], proj["pool"],
        graine=GRAINE_CONTROLE_STRATIFIE, repl=25,
    )
    d_intra = delta_ctx_intra_paires(
        proj["paire"], proj["g0"], proj["hote_mot"], proj["pool"]
    )
    return {
        "pre": pre,
        "obs": obs,
        "nulle": nulle,
        "ctrl": ctrl,
        "verdict": case_verdict(nulle["p_intra"], nulle["p_inter"]),
        "deltas": [[list(i), d] for i, d in d_intra["deltas"]],
    }


def test_pipeline_mini_deterministe_x2_au_bit():
    a = json.dumps(pipeline_mini(), sort_keys=True)
    b = json.dumps(pipeline_mini(), sort_keys=True)
    assert a == b


def test_pipeline_mini_t4_et_invariance():
    res = pipeline_mini()
    assert res["obs"]["t4_tenu_reel"]
    assert res["pre"]["n_violations"] == 0
    assert res["ctrl"]["invariance_tenue_1e-12"]


def test_portees_toutes_branches_atteintes():
    p = portees_t78()
    assert p["theoreme_T4"]["branche_identite"]["atteinte"]
    assert p["theoreme_T4"]["branche_cassee_poids_uniformes"]["atteinte"]
    assert p["theoreme_T5"]["tenu"]
    assert p["t6_pre"]["branche_zero_violation"]["atteinte"]
    assert p["t6_pre"]["branche_violation_detectee"]["atteinte"]
    assert p["gc1"]["branche_mismatch"]["mordante"]
    assert p["decomposition_p_emp"]["branche_intra_tenue_injectee"]["atteinte"]
    assert p["decomposition_p_emp"]["branche_montage_inverse"]["atteinte"]
    assert p["decomposition_p_emp"]["determinisme_x2_p_emp"]
    assert p["controle_stratifie"]["branche_invariante_strates_pleines"]["atteinte"]
    assert p["controle_stratifie"]["branche_mobile_strates_partielles"]["atteinte"]
    assert p["spearman"]["accord_deux_routes"]
    assert p["recette_lambda_intra"]["branche_calculable"]["atteinte"]
    assert p["recette_lambda_intra"]["branche_iqr_zero"]["atteinte"]
    assert p["delta_ctx_intra"]["branche_positive"]["atteinte"]
    assert p["delta_ctx_intra"]["branche_negative"]["atteinte"]
