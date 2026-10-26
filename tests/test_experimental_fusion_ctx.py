"""Quatuor de tests T82 pour ``experimental/fusion_ctx.py`` (émission §3.6).

1. Déterminisme : deux exécutions → sorties identiques (zéro tirage).
2. Finitude : aucun NaN/Inf dans journal, candidats, vocabulaire.
3. Formule exacte sur témoins forcés : Coherence_full_intra positif porté
   par le contexte, négatif, égalité exacte (θ = 0 strict), et l'identité
   ``full == coh_H + λ·Δ`` reconstruite indépendamment (numpy nu).
4. Gardes mordantes : GC-82 muté ⇒ échec constaté (NonMesureError) ;
   invariant de décroissance ⇒ ValueError sur injection ; MAX_FUSIONS et
   mur temporel sur témoins.

Témoins = ``TABLE_TEMOIN`` (T75) + tokens forcés dims 28-30 ; AUCUN ``.so``
requis, AUCUN tirage aléatoire (seeds sans objet — déterminisme structurel).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from spiraton.experimental.bpe_logos import CacheStab, TABLE_TEMOIN, appliquer_fusion
from spiraton.experimental.fusion_ctx import (
    LAMBDA_GELE,
    EtatFusionCtx,
    comparer_gc82,
    evaluer_candidats,
    executer_fusions_ctx,
    n_partagees,
    replay_vierge,
    verdict_vierge,
    _temoin_contexte_porte,
    _temoin_egalite,
    _temoin_fusible_delta_zero,
    _temoin_negatif,
    _temoin_partage_un,
    _temoin_partage_zero,
    _tok_temoin,
)
from spiraton.experimental.stab_decomp import NonMesureError
from spiraton.experimental.bpe_logos import vocabulaire_final


def _run(contextes, **kw):
    ef = EtatFusionCtx(contextes)
    cache = CacheStab(TABLE_TEMOIN)
    r = executer_fusions_ctx(ef, cache, LAMBDA_GELE, **kw)
    return ef, r


# ---------------------------------------------------------------------------
# 1. Déterminisme
# ---------------------------------------------------------------------------

def test_determinisme_deux_executions_identiques():
    _, r1 = _run(_temoin_contexte_porte())
    _, r2 = _run(_temoin_contexte_porte())
    assert r1["journal"] == r2["journal"]
    assert r1["arret"] == r2["arret"] == "ARRET_CRITERE"
    assert r1["profondeurs"] == r2["profondeurs"]
    _, ra1 = _run(_temoin_fusible_delta_zero())
    _, ra2 = _run(_temoin_fusible_delta_zero())
    assert ra1["journal"] == ra2["journal"]
    assert ra1["n_fusions"] == 2


def test_determinisme_replay():
    cache = CacheStab(TABLE_TEMOIN)
    j = [{"iteration": 1, "a": [0], "b": [1]}]
    rep1 = replay_vierge(EtatFusionCtx(_temoin_partage_zero()), cache, j)
    rep2 = replay_vierge(EtatFusionCtx(_temoin_partage_zero()), cache, j)
    assert rep1 == rep2


# ---------------------------------------------------------------------------
# 2. Finitude
# ---------------------------------------------------------------------------

def _fini(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def test_finitude_journal_et_candidats():
    ef, r = _run(_temoin_contexte_porte())
    for e in r["journal"]:
        for cle in ("coherence_full", "contribution_H", "contribution_lambda_delta",
                    "delta_ctx"):
            assert _fini(e[cle]), (cle, e[cle])
    fin = r["etat_final_candidats"]
    for c in fin["candidats_refuses"]:
        assert _fini(c["coherence_full"])
    vocab = vocabulaire_final(ef.etat, r["profondeurs"], TABLE_TEMOIN)
    for e in vocab["unites"]:
        assert e["k"] >= 0 and e["n_occurrences"] >= 1


# ---------------------------------------------------------------------------
# 3. Formule exacte sur témoins forcés
# ---------------------------------------------------------------------------

def test_formule_exacte_positif_porte_par_contexte():
    """Témoin B : coh_H((2,),(2,)) < 0 mais Δ_ctx = 50/3 exact ⇒ fusion.

    Reconstruction indépendante (numpy nu, mêmes définitions §3.2) :
    V^{d1}(2,) = (4/6)·0 + (2/6)·Var_ddof1({0,10}) ; V(adj) = 0 ;
    Δ = ½(V+V) − 0 ; full = coh_H + λ·Δ — égalité au bit exigée.
    """
    cache = CacheStab(TABLE_TEMOIN)
    ef, r = _run(_temoin_contexte_porte())
    assert r["n_fusions"] == 1 and r["arret"] == "ARRET_CRITERE"
    e = r["journal"][0]
    assert e["a"] == [2] and e["b"] == [2]
    coh_h = cache.coherence((2,), (2,))
    assert coh_h < 0.0  # A2 : jamais fusible sous H seul
    var_c = float(np.var(np.asarray([[0.0] * 3, [10.0] * 3]), axis=0, ddof=1).mean())
    v_u = (4.0 / 6.0) * 0.0 + (2.0 / 6.0) * var_c
    delta = 0.5 * (v_u + v_u) - 0.0
    assert e["contribution_H"] == coh_h
    assert e["delta_ctx"] == delta
    assert e["coherence_full"] == coh_h + LAMBDA_GELE * delta
    assert e["coherence_full"] > 0.0
    assert e["k"] == 1


def test_formule_exacte_negatif():
    _, r = _run(_temoin_negatif())
    assert r["n_fusions"] == 0 and r["arret"] == "ARRET_CRITERE"
    cand = r["etat_final_candidats"]["candidats_refuses"]
    assert len(cand) == 1
    cache = CacheStab(TABLE_TEMOIN)
    # Δ = 0 exact (c(occ) bit-identiques) ⇒ full == coh_H < 0 au bit.
    assert cand[0]["coherence_full"] == cache.coherence((2,), (2,))
    assert cand[0]["coherence_full"] < 0.0


def test_formule_exacte_egalite_theta_strict():
    _, r = _run(_temoin_egalite())
    assert r["n_fusions"] == 0
    cand = r["etat_final_candidats"]["candidats_refuses"]
    assert cand[0]["coherence_full"] == 0.0  # A2 uniforme + Δ = 0 ⇒ 0 exact
    # θ = 0 STRICT : l'égalité ne fusionne pas.


def test_exclusion_non_mesurable_comptee_sans_repli():
    """Une adjacence à 1 occurrence est EXCLUE-COMPTÉE (jamais de repli H)."""
    ctx = [[_tok_temoin("ab", (1.0, 2.0, 3.0), [0, 1])]]  # 1 seule occurrence
    ef = EtatFusionCtx(ctx)
    cache = CacheStab(TABLE_TEMOIN)
    ev = evaluer_candidats(ef, cache, LAMBDA_GELE)
    assert ev["n_mesurables"] == 0
    assert ev["n_exclues_non_mesurables"] == 1
    assert ev["exclues"][0]["motif"]["adjacence_non_mesurable"] is True


def test_n_partagees_bornes():
    ef0, r0 = _run(_temoin_partage_zero())
    v0 = n_partagees(vocabulaire_final(ef0.etat, r0["profondeurs"], TABLE_TEMOIN))
    assert v0["n_types_fusionnes"] >= 1 and v0["n_partagees"] == 0
    ef1, r1 = _run(_temoin_partage_un())
    v1 = n_partagees(vocabulaire_final(ef1.etat, r1["profondeurs"], TABLE_TEMOIN))
    assert v1["n_partagees"] >= 1
    assert v1["partagees_en_extension"][0]["n_types_mots"] >= 2


def test_verdicts_vierge_liste_close():
    assert verdict_vierge(0.8, 0.3, 40, 5, 5) == "GENERALISE"
    assert verdict_vierge(0.2, 0.3, 40, 5, 5) == "NE GENERALISE PAS"
    assert verdict_vierge(0.5, 0.3, 40, 5, 5) == "INDISTINCT"
    assert verdict_vierge(0.8, 0.3, 29, 5, 5) == "NON-MESURE"
    assert verdict_vierge(0.8, 0.0, 40, 0, 45) == "NON-MESURE"


def test_replay_attestee_et_non_attestee():
    cache = CacheStab(TABLE_TEMOIN)
    j = [
        {"iteration": 1, "a": [0], "b": [1]},  # attestée
        {"iteration": 2, "a": [0], "b": [3]},  # jamais présente
    ]
    rep = replay_vierge(EtatFusionCtx(_temoin_partage_zero()), cache, j)
    assert rep["n_attestees"] == 1
    assert rep["n_non_attestees"] == 1
    assert rep["n_fusions_rejouees"] == 2


# ---------------------------------------------------------------------------
# 4. Gardes mordantes
# ---------------------------------------------------------------------------

def test_garde_gc82_mutee_echec_constate():
    attendus = {"n": 450, "lam": 1.2744331711650387, "ident": [27, 3]}
    obtenus = dict(attendus)
    # Branche égalité : passe.
    detail = comparer_gc82(attendus, obtenus)
    assert all(v["egal_au_bit"] for v in detail.values())
    # Mutation 1 ulp sur le flottant ⇒ NonMesureError (I0).
    mute = dict(attendus)
    mute["lam"] = math.nextafter(attendus["lam"], math.inf)
    with pytest.raises(NonMesureError):
        comparer_gc82(mute, obtenus)
    # Mutation d'un entier ⇒ NonMesureError.
    mute2 = dict(attendus)
    mute2["n"] = 451
    with pytest.raises(NonMesureError):
        comparer_gc82(mute2, obtenus)


def test_garde_invariant_decroissance_injection():
    ef = EtatFusionCtx(_temoin_fusible_delta_zero())
    with pytest.raises(ValueError):
        appliquer_fusion(ef.etat, (2,), (3,))  # paire absente ⇒ garde T75


def test_garde_max_fusions_temoin():
    _, r = _run(_temoin_fusible_delta_zero(), max_fusions=1)
    assert r["arret"] == "I5_MAX_FUSIONS"
    assert r["n_fusions"] == 1


def test_garde_mur_temporel_temoin():
    _, r = _run(
        _temoin_fusible_delta_zero(), mur_s=0.0, t0=time.perf_counter() - 1.0
    )
    assert r["arret"] == "I5_MUR_TEMPOREL"
    assert r["n_fusions"] == 0


def test_garde_controle_etat0_mordante():
    ef = EtatFusionCtx(_temoin_fusible_delta_zero())
    cache = CacheStab(TABLE_TEMOIN)
    with pytest.raises(NonMesureError):
        executer_fusions_ctx(
            ef, cache, LAMBDA_GELE, controle_etat0=(532, 450, 107)
        )  # témoin ≠ état 0 d'eve ⇒ la réduction exacte doit refuser
