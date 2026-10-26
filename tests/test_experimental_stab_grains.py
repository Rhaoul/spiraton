"""Quatuor de tests de l'instrument Stab par grain (T73).

(1) parité de formule exacte (témoins dérivés à la main, recalcul indépendant),
(2) finitude (aucun NaN/Inf hors cas S = 0 déclaré),
(3) déterminisme ×2 au bit (permutation seedée incluse),
(4) gardes (S = 0, n_occ < 2, drop > 0, drop < 0, tranche vide) mordantes.

Tout est pur Python/numpy : aucun ``.so`` requis (le pilote natif est testé
par les portes de l'émission, pas ici). Aucune source d'aléa non seedée.
"""
import json
import math

import numpy as np
import pytest

from spiraton.experimental.stab_grains import (
    CTX_DIMS,
    GRAINE_PERMUTATION,
    MAIN_DIMS,
    controle_permutation,
    entropie_masse,
    extraire_occurrences,
    stats_h,
    stats_var_ctx,
    var_ctx,
)

# --- Mini-table et mini-corpus forcés (valeurs choisies à la main) ----------

TABLE = [
    {"flux": 0.5, "structure": 0.5, "energie_totale": 1.0, "impedance": 0.25},
    {"flux": 1.0, "structure": 0.0, "energie_totale": 0.5, "impedance": 0.5},
]


def tk(text, v33, g0_ids, g1_nb):
    return {
        "text": text,
        "vector33d": np.asarray(v33, dtype=np.float32),
        "g0_ids": list(g0_ids),
        "g1_nb": list(g1_nb),
    }


def v33_dirac(dims_ctx_vals):
    """33D nul sauf dim 6 = 1.0 (Dirac sur MAIN) et dims 28-30 imposées."""
    v = np.zeros(33, dtype=np.float32)
    v[6] = 1.0
    v[28], v[29], v[30] = dims_ctx_vals
    return v


def mini_corpus():
    """Deux contextes ; « Aa »/« aa » = même identité mot, 2 occurrences."""
    ctx1 = [tk("Aa", v33_dirac((1.0, 2.0, 3.0)), [0, 1], [2])]
    ctx2 = [tk("aa", v33_dirac((1.0, 2.0, 7.0)), [0, 1], [2])]
    return [ctx1, ctx2]


# ---------------------------------------------------------------------------
# (1) Parité de formule exacte
# ---------------------------------------------------------------------------

def test_h_temoin_dirac_exact():
    """Dirac : masse sur une seule dim ⇒ H = 0.0 EXACTEMENT (branche basse)."""
    sigma = np.zeros(19)
    sigma[3] = 2.5
    s, h, h_norm = entropie_masse(sigma)
    assert s == 2.5 and h == 0.0 and h_norm == 0.0


def test_h_temoin_uniforme_exact():
    """Uniforme 19 dims ⇒ H = log2 19 (branche haute atteignable), H_norm = 1."""
    s, h, h_norm = entropie_masse(np.full(19, 0.37))
    assert s == pytest.approx(19 * 0.37)
    assert h == pytest.approx(math.log2(19), abs=1e-12)
    assert h_norm == pytest.approx(1.0, abs=1e-12)


def test_h_temoin_reel_recalcul_independant():
    """σ témoin (constantes de table concaténées) recalculé par une boucle
    indépendante (math.log, aucune vectorisation partagée)."""
    sigma = [0.5, 0.5, 1.0, 0.25, 1.0, 0.0, 0.5, 0.5]  # syllabe (0, 1) de TABLE
    s, h, h_norm = entropie_masse(sigma)
    s_ref = sum(abs(x) for x in sigma)
    h_ref = -sum(
        (abs(x) / s_ref) * math.log(abs(x) / s_ref, 2.0) for x in sigma if x != 0.0
    )
    assert s == pytest.approx(s_ref)
    assert h == pytest.approx(h_ref, abs=1e-12)
    assert h_norm == pytest.approx(h_ref / math.log2(len(sigma)), abs=1e-12)
    # Le signe passe par |·| (déclaré) : σ négatif ⇒ même H.
    assert entropie_masse([-x for x in sigma])[1] == pytest.approx(h_ref, abs=1e-12)


def test_var_ctx_temoin_deux_occurrences():
    """Témoin à 2 occurrences : var pop par dim [0, 0, 4] ⇒ Var_ctx = 4/3."""
    assert var_ctx([[1.0, 2.0, 3.0], [1.0, 2.0, 7.0]]) == pytest.approx(4.0 / 3.0)


def test_var_ctx_zero_exact_sur_float32_identiques():
    """Branche zéro : des float32 identiques donnent EXACTEMENT 0.0 (P-1)."""
    o = np.asarray([0.5, 0.25, 1.0], dtype=np.float32)
    assert var_ctx(np.stack([o, o, o])) == 0.0


def test_var_ctx_aveugle_a_l_ordre():
    """PORTÉE C3 : permuter les occurrences ne change pas la valeur."""
    occs = [[1.0, 0.0, 2.0], [3.0, 1.0, 0.0], [0.5, 2.0, 1.0]]
    assert var_ctx(occs) == var_ctx(occs[::-1])


def test_mini_corpus_valeurs_exactes():
    """Mini-corpus forcé : identités, effectifs et valeurs dérivés à la main."""
    occ = extraire_occurrences(mini_corpus(), TABLE)
    # Identité mot : lower() ⇒ « Aa » et « aa » fusionnent.
    assert list(occ["mot"].keys()) == ["aa"]
    assert occ["comptes"]["n_tokens"] == 2 and occ["comptes"]["n_occ_g0"] == 4
    # Var_ctx(mot) sur dims 28-30 : 4/3 (cf. témoin) ; régime CREUX (n=1 < 30).
    vc_mot = stats_var_ctx(occ["mot"], dims_ctx=CTX_DIMS)
    assert vc_mot["n_mesurables"] == 1 and vc_mot["regime_h62"] == "CREUX"
    assert vc_mot["quartiles_positifs"]["mediane"] == pytest.approx(4.0 / 3.0)
    # Syllabe : identité (0, 1), 2 occurrences, zéro EXACT (P-1b).
    assert list(occ["syllabe"].keys()) == [(0, 1)]
    vc_syl = stats_var_ctx(occ["syllabe"])
    assert vc_syl["n_zero_exact"] == 1 and vc_syl["part_zero_exact"] == 1.0
    # G₀ : 2 types × 2 occurrences, zéro EXACT (P-1).
    vc_g0 = stats_var_ctx(occ["g0"])
    assert vc_g0["n_mesurables"] == 2 and vc_g0["n_zero_exact"] == 2
    # H(mot) : Dirac sur MAIN ⇒ H_norm = 0.0 ; aucune unité à H variable.
    h_mot = stats_h(occ["mot"], tranche_mot=MAIN_DIMS)
    assert h_mot["h_norm"]["mediane"] == 0.0
    assert h_mot["n_unites_h_variable"] == 0 and h_mot["n_degenere_support"] == 0


def test_regime_h62_trois_branches():
    """Frontière H62 : CREUX par part (< 0,50), CREUX par n (< 30), ESTIMABLE."""
    def unites(n_mes, n_creuses):
        d = {}
        for i in range(n_mes):
            d[f"m{i}"] = [[float(i), 0.0, 0.0], [float(i) + 1.0, 0.0, 0.0]]
        for i in range(n_creuses):
            d[f"c{i}"] = [[0.0, 0.0, 0.0]]
        return d

    assert stats_var_ctx(unites(30, 40))["regime_h62"] == "CREUX"      # part 30/70
    assert stats_var_ctx(unites(10, 0))["regime_h62"] == "CREUX"       # n 10 < 30
    est = stats_var_ctx(unites(30, 0))
    assert est["regime_h62"] == "ESTIMABLE" and est["part_mesurable"] == 1.0
    assert est["masse_occ_couverte"] == 1.0


def test_h_non_degenerescence_deux_branches():
    """Verdict H : dégénéré (1 valeur) vs non-dégénéré (≥16 distincts, IQR ≥ 0,01)."""
    uni = {f"u{i}": [[1.0, 1.0, 1.0, 1.0]] for i in range(20)}
    assert stats_h(uni)["non_degeneree"] is False  # 1 seule valeur de H_norm
    varie = {f"v{i}": [[1.0, 0.1 + 0.045 * i, 0.0, 0.0]] for i in range(20)}
    st = stats_h(varie)
    assert st["n_distinct_h_norm"] == 20 and st["non_degeneree"] is True


def test_h_support_degenere_compte():
    """S = 0 : unité comptée DEGENERE-SUPPORT, exclue de la distribution."""
    st = stats_h({"vide": [[0.0, 0.0]], "plein": [[1.0, 1.0]]})
    assert st["n_degenere_support"] == 1 and st["n_support_positif"] == 1
    assert st["part_support_positif"] == 0.5


# ---------------------------------------------------------------------------
# (2) Finitude
# ---------------------------------------------------------------------------

def _scan_finitude(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            _scan_finitude(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scan_finitude(v)
    elif isinstance(obj, float):
        assert math.isfinite(obj), f"valeur non finie : {obj}"


def test_finitude_stats_completes():
    """Aucun NaN/Inf dans les stats (hors cas S = 0, rendu par None déclaré)."""
    occ = extraire_occurrences(mini_corpus(), TABLE)
    for st in (
        stats_h(occ["mot"], tranche_mot=MAIN_DIMS),
        stats_h(occ["syllabe"]),
        stats_var_ctx(occ["mot"], dims_ctx=CTX_DIMS),
        stats_var_ctx(occ["g0"]),
    ):
        _scan_finitude(st)


# ---------------------------------------------------------------------------
# (3) Déterminisme ×2 au bit (permutation seedée incluse)
# ---------------------------------------------------------------------------

def test_determinisme_stats_bit_a_bit():
    """Deux exécutions complètes ⇒ JSON byte-identiques."""
    def run():
        occ = extraire_occurrences(mini_corpus(), TABLE)
        return json.dumps(
            {
                "h_mot": stats_h(occ["mot"], tranche_mot=MAIN_DIMS),
                "vc_mot": stats_var_ctx(occ["mot"], dims_ctx=CTX_DIMS),
                "vc_syl": stats_var_ctx(occ["syllabe"]),
                "vc_g0": stats_var_ctx(occ["g0"]),
            },
            sort_keys=True,
        )

    assert run() == run()


def test_determinisme_permutation_seedee():
    """Le contrôle par permutation (graine gelée 73201) est reproductible au bit."""
    rng = np.random.default_rng(4242)  # données synthétiques, graine fixée
    groupes = [rng.normal(size=(n, 3)) for n in (2, 3, 5, 2)]
    a = controle_permutation(groupes, graine=GRAINE_PERMUTATION, repl=20)
    b = controle_permutation(groupes, graine=GRAINE_PERMUTATION, repl=20)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["n_replicats"] == 20 and a["n_occurrences_pool"] == 12


# ---------------------------------------------------------------------------
# (4) Gardes mordantes
# ---------------------------------------------------------------------------

def test_garde_tranche_vide():
    with pytest.raises(ValueError, match="tranche vide"):
        entropie_masse([])


def test_garde_s_zero_ne_leve_pas_mais_declare():
    assert entropie_masse([0.0, 0.0, 0.0]) == (0.0, None, None)


def test_garde_n_occ_insuffisant():
    with pytest.raises(ValueError, match="n_occ = 1 < 2"):
        var_ctx([[1.0, 2.0, 3.0]])


def test_garde_matrice_mal_formee():
    with pytest.raises(ValueError, match="matrice"):
        var_ctx(np.zeros((3, 0)))


def test_garde_drop_positif_exclut_et_compte():
    """drop = g0_len − Σ g1_nb > 0 ⇒ token hors grain syllabe, COMPTÉ ;
    le grain g0 garde ses occurrences."""
    ctx = [[tk("x", np.zeros(33), [0, 1], [1])]]  # drop = 1
    occ = extraire_occurrences(ctx, TABLE)
    assert occ["comptes"]["n_tokens_drop_sylla"] == 1
    assert len(occ["syllabe"]) == 0 and occ["comptes"]["n_occ_g0"] == 2


def test_garde_drop_negatif_refuse():
    with pytest.raises(ValueError, match="incohérence ABI"):
        extraire_occurrences([[tk("x", np.zeros(33), [0], [2])]], TABLE)


def test_garde_syllabe_nb_zero_comptee():
    ctx = [[tk("x", np.zeros(33), [0, 1], [0, 2])]]
    occ = extraire_occurrences(ctx, TABLE)
    assert occ["comptes"]["n_syllabes_tranche_vide"] == 1
    assert list(occ["syllabe"].keys()) == [(0, 1)]


def test_garde_token_muet_compte():
    """g0_len == 0 (mot muet, légitime T72) : compté, présent au grain mot."""
    ctx = [[tk("", np.zeros(33), [], [])]]
    occ = extraire_occurrences(ctx, TABLE)
    assert occ["comptes"]["n_tokens_muets"] == 1
    assert list(occ["mot"].keys()) == [""] and len(occ["g0"]) == 0


def test_garde_permutation_sans_groupe():
    with pytest.raises(ValueError, match="aucun groupe"):
        controle_permutation([])
