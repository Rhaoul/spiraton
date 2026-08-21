"""Stab contextuel par projection 33D — instrument du Tour 77 (voie contextuelle, pont (a)).

Ce module fait DESCENDRE le signal contextuel du grain mot (dims 28-30 du 33D,
gel T73) vers les grains sous-lexicaux par HÉRITAGE : chaque sous-unité d'un
token (phonème, paire ordonnée adjacente, syllabe) hérite le vecteur de
contexte c(occ) de son occurrence-hôte. Toutes les définitions ci-dessous sont
les gels de ``TOUR77_EMISSION.md`` (md5 ``722d60f03e2c112d4272e3b42400ea3f``)
— rien n'est réglable en douce.

Définitions opérationnelles gelées (émission T77) :

- **Occurrence-hôte** (§4.1) : token de ``tokenize_sequences`` (R16) avec
  ``g0_len ≥ 1`` ; c(occ) = composantes ``CTX_DIMS`` = (28, 29, 30) du 33D
  (float32 natif → float64 à l'extraction).
- **Grain PAIRE-projeté** (le grain du verdict) : occurrence de la paire
  ORDONNÉE (a, b) ∈ G₀×G₀ = chaque position i avec ``g0_ids[i] = a`` et
  ``g0_ids[i+1] = b``, toutes positions, CHEVAUCHEMENT COMPRIS ((a,a) dans
  « aaa » = 2 occurrences). Chaque occurrence hérite c(occ) de son hôte.
- **Grain G₀-projeté** (support de Δ_ctx) : chaque position de ``g0_ids``
  hérite c(occ). **Grain syllabe-projeté** (INFO seulement) : idem sur les
  tranches g1 (tuple g0 exact) ; tokens à ``drop > 0`` exclus-comptés (T73).
- **Var_ctx_proj(u)** (§4.2) : ``(1/3) Σ_d s²_d`` avec s²_d la variance
  **ddof=1 (non biaisée)**, float64, des occurrences de u sur la dim d.
  VERSION DÉCLARÉE de la mesure (écart motivé vs ``stab_grains.var_ctx``
  ddof=0, publié en INFO de continuité T73 — les deux colonnes, jamais
  mélangées). Motif : théorème T2 (sous la nulle d'échangeabilité,
  E[s²_ddof1] est indépendante de l'effectif — la nullité mécanique est
  éliminée par CONSTRUCTION, pas par surveillance ; préférence A76-1).
- **Zéro exact** : par IDENTITÉ AU BIT des vecteurs hérités
  (``ndarray.tobytes()`` identiques), jamais par ``Var == 0.0`` flottant
  (instruction T73) ; ``n_zero_bit`` publié, écart résiduel borné-publié.
- **Sélecteurs (CONTRAT T54)** : mesurable ssi ``n_occ ≥ 2`` ; pour Δ_ctx :
  paire mesurable ET ses deux parts mesurables. Fonctions du corpus/interface,
  invariantes par permutation des valeurs mesurées ; comptes publiés des deux
  côtés de chaque sélection.
- **Carte H62** (§4.4) : régime CREUX/ESTIMABLE par les seuils gelés T73/G1
  (``SEUIL_PART_MESURABLE``, ``SEUIL_N_MESURABLES`` réutilisés zéro octet).
- **TENUE** (§4.5) : ``V_med`` = MÉDIANE de Var_ctx_proj sur les types de
  paires mesurables ; nulle (l.342) = R = 100 permutations seedées du multiset
  des c(occ) entre occurrences-hôtes (structure d'héritage conservée — T1
  préservé par construction), ``numpy.random.default_rng(77201)`` unique,
  réplicats dans l'ordre ; ``p_emp = (#{V^r_med ≤ V_obs} + 1)/(R + 1)`` ;
  TENUE ssi p ≤ 0,05, ANTI-TENUE ssi p ≥ 0,95, sinon INDISTINCT — prononcé
  sur EVE SEUL, si paire×eve ESTIMABLE (G1), claude_aba en INFO.
- **λ*** (§4.6) : ``IQR(H_norm) / IQR(Var_ctx_proj)`` sur LES MÊMES types de
  paires mesurables d'eve ; ``H_norm(paire) = H_norm(σ_unit(a+b))`` par la
  fonction gelée T75 (``CacheStab.h_norm``, réutilisée zéro octet).
  IQR(Var) = 0,0 ⇒ λ NON-MESURE (branche déclarée). Zéro hyperparamètre.
- **Δ_ctx(a,b)** (§4.7) : ``½[Var_ctx_proj(a) + Var_ctx_proj(b)] −
  Var_ctx_proj(a·b)`` (positif = la jonction tient mieux que ses parties).
  **Coherence_full(a,b) = Coherence_H(a,b) + λ*·Δ_ctx(a,b)`` avec
  Coherence_H LA fonction gelée T75 (``CacheStab.coherence``). AUCUNE fusion
  n'est exécutée ce tour — admissibilité seulement.
- **R-INSTRUMENT strict (§2)** : tout ce que ce module mesure est une
  propriété du triplet (projection × corpus × interface R16/33D) — jamais
  « le français », jamais « la morphologie ». La « tenue » est l'association
  paire→vecteurs hérités, TOUT MÉCANISME CONFONDU (part lexicale des dims
  28-30 comprise, T73) ; la séparation lexical/contextuel est un tour futur.

AUCUN verdict d'ordre/clôture A→B→A′ ce tour : toutes les grandeurs sont
invariantes par permutation des occurrences (déclaré §4.5/§7 — gap δ_min N-A).

Contexte d'exécution (invariant §9 T76) : chaque artefact écho sa config
complète (graines, seuils, sha COMPLET du .so, md5 corpus, Python/numpy/
plateforme/locale). Les artefacts de verdict sont SANS horodatage :
reproductibles au bit (C1). Les verdicts vivent dans les artefacts
``TOUR77_*.json`` (racine écosystème), pas ici.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import locale as _locale
import math
import platform
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from spiraton.experimental.stab_grains import (
    CTX_DIMS,
    SEUIL_N_MESURABLES,
    SEUIL_PART_MESURABLE,
    charger_tokenizer_garde,
    contextes_claude_aba,
    contextes_eve,
    table_miroir,
)
from spiraton.experimental.stab_grains import var_ctx as var_ctx_ddof0_t73
from spiraton.experimental.bpe_logos import (
    SHA256_SO_REFERENCE_T75,
    CacheStab,
    TABLE_TEMOIN,
)

# ---------------------------------------------------------------------------
# Constantes gelées du tour (émission T77 — jamais cachées, jamais réglables)
# ---------------------------------------------------------------------------

#: Graine du shuffle de verdict (§4.5) — default_rng unique, réplicats en ordre.
GRAINE_SHUFFLE: int = 77201

#: Nombre de réplicats de la nulle (§4.5).
R_SHUFFLE: int = 100

#: Graine des témoins de théorèmes (§4.3) — génère le pool synthétique de T2.
GRAINE_TEMOINS: int = 77003

#: Seuils du verdict de tenue (§4.5) : TENUE ≤ 0,05 ; ANTI-TENUE ≥ 0,95.
SEUIL_P_TENUE: float = 0.05
SEUIL_P_ANTI_TENUE: float = 0.95

#: Tolérance de vérification EXACTE des théorèmes T2 (§4.3) — espérance
#: atteinte par énumération exhaustive, pas par Monte-Carlo.
TOL_THEOREME_T2: float = 1e-9

#: Mur étalonné (§13) : run ≤ min(FACTEUR_MUR × t_cal, PLAFOND_DUR_S).
FACTEUR_MUR: float = 20_000.0
PLAFOND_DUR_S: float = 1800.0
N_PHRASES_CAL: int = 9

Ident = Tuple[int, ...]
Paire = Tuple[int, int]


# ---------------------------------------------------------------------------
# Var_ctx_proj — formule exacte (§4.2), version déclarée ddof=1
# ---------------------------------------------------------------------------

def var_ctx_proj(occurrences: Sequence[Sequence[float]]) -> float:
    """``Var_ctx_proj(u) = (1/D) Σ_d s²_d`` — variance ddof=1, float64.

    ``occurrences`` : matrice (n, D), n ≥ 2 requis (sélecteur T54, garde
    mordante). PORTÉE (C3) : détecte la dispersion inter-occurrences ;
    AVEUGLE à l'ordre des occurrences (permutation des lignes sans effet —
    déclaré : aucun verdict d'ordre ce tour).
    """
    mat = np.asarray(occurrences, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] == 0:
        raise ValueError(f"matrice (n, D) attendue, D >= 1 : shape={mat.shape}")
    if mat.shape[0] < 2:
        raise ValueError(
            f"n_occ = {mat.shape[0]} < 2 : unité creuse, Var_ctx_proj non définie"
        )
    return float(mat.var(axis=0, ddof=1).mean())


# ---------------------------------------------------------------------------
# Extraction : occurrences-hôtes et héritage (§4.1)
# ---------------------------------------------------------------------------

def extraire_projection(
    tokens_par_contexte: Iterable[Sequence[dict]],
) -> Dict[str, object]:
    """Occurrences-hôtes + grains projetés depuis les tokens R16 (§4.1).

    Retour : dict avec

    - ``pool`` : matrice (H, 3) float64 des c(occ) des H occurrences-hôtes
      (``g0_len ≥ 1``), ordre corpus (déterministe) ;
    - ``pool_bytes`` : ``tobytes()`` de chaque c(occ) (zéro exact au bit) ;
    - ``hote_mot`` : identité mot (``text.lower()``, gel T73) de chaque hôte
      — support de l'INFO « types de mots-hôtes par paire » (§2) ;
    - ``paire`` / ``g0`` / ``syllabe`` : OrderedDict identité → liste
      d'indices d'hôtes (une entrée PAR OCCURRENCE, chevauchement compris) ;
    - ``comptes`` : exclusions déclarées des deux côtés de chaque sélection.

    Gardes héritées T73 : ``drop = g0_len − Σ g1_nb < 0`` ⇒ incohérence ABI,
    refus. T1 par construction : toutes les sous-unités d'un hôte reçoivent
    LE MÊME indice d'hôte, donc le même vecteur.
    """
    pool: List[np.ndarray] = []
    pool_bytes: List[bytes] = []
    hote_mot: List[str] = []
    paires: "OrderedDict[Paire, List[int]]" = OrderedDict()
    g0: "OrderedDict[int, List[int]]" = OrderedDict()
    syllabes: "OrderedDict[Ident, List[int]]" = OrderedDict()

    comptes: Dict[str, int] = {
        "n_contextes": 0,
        "n_tokens": 0,
        "n_tokens_muets": 0,          # g0_len == 0 : PAS des hôtes (comptés)
        "n_hotes": 0,                 # g0_len >= 1
        "n_hotes_g0len1_sans_paire": 0,
        "n_occ_g0": 0,
        "n_occ_paires": 0,
        "n_hotes_drop_syllabe": 0,    # drop > 0 : exclus du grain syllabe
        "n_syllabes_tranche_vide": 0,
        "n_occ_syllabes": 0,
    }

    for tokens in tokens_par_contexte:
        comptes["n_contextes"] += 1
        for tok in tokens:
            comptes["n_tokens"] += 1
            g0_ids = [int(p) for p in tok["g0_ids"]]
            if len(g0_ids) == 0:
                comptes["n_tokens_muets"] += 1
                continue
            v33 = np.asarray(tok["vector33d"], dtype=np.float32)
            c_occ = v33[list(CTX_DIMS)].astype(np.float64)
            h = len(pool)
            pool.append(c_occ)
            pool_bytes.append(c_occ.tobytes())
            hote_mot.append(str(tok["text"]).lower())
            comptes["n_hotes"] += 1

            # --- grain G₀-projeté (support de Δ_ctx) -----------------------
            for pid in g0_ids:
                g0.setdefault(pid, []).append(h)
                comptes["n_occ_g0"] += 1

            # --- grain PAIRE-projeté (verdict) : chevauchement compris -----
            if len(g0_ids) == 1:
                comptes["n_hotes_g0len1_sans_paire"] += 1
            for i in range(len(g0_ids) - 1):
                paires.setdefault((g0_ids[i], g0_ids[i + 1]), []).append(h)
                comptes["n_occ_paires"] += 1

            # --- grain syllabe-projeté (INFO) — exclusions gelées T73 ------
            g1_nb = [int(n) for n in tok["g1_nb"]]
            somme_nb = sum(g1_nb)
            drop = len(g0_ids) - somme_nb
            if drop < 0:
                raise ValueError(
                    f"incohérence ABI : Σ g1_nb = {somme_nb} > g0_len = "
                    f"{len(g0_ids)} (token {tok['text']!r})"
                )
            if drop > 0:
                comptes["n_hotes_drop_syllabe"] += 1
                continue
            cur = 0
            for nb in g1_nb:
                if nb == 0:
                    comptes["n_syllabes_tranche_vide"] += 1
                    continue
                tranche = tuple(g0_ids[cur:cur + nb])
                cur += nb
                syllabes.setdefault(tranche, []).append(h)
                comptes["n_occ_syllabes"] += 1

    return {
        "pool": np.asarray(pool, dtype=np.float64).reshape(len(pool), len(CTX_DIMS)),
        "pool_bytes": pool_bytes,
        "hote_mot": hote_mot,
        "paire": paires,
        "g0": g0,
        "syllabe": syllabes,
        "comptes": comptes,
    }


# ---------------------------------------------------------------------------
# Carte H62 par grain projeté (§4.4)
# ---------------------------------------------------------------------------

def _quartiles77(valeurs: Sequence[float]) -> Dict[str, float]:
    """n/min/q25/mediane/q75/max/iqr — np.quantile interpolation linéaire (T73)."""
    arr = np.asarray(valeurs, dtype=np.float64)
    q25, q50, q75 = (float(x) for x in np.quantile(arr, [0.25, 0.5, 0.75]))
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "q25": q25,
        "mediane": q50,
        "q75": q75,
        "max": float(arr.max()),
        "iqr": q75 - q25,
    }


def stats_grain_projete(
    unites: "OrderedDict[object, List[int]]",
    pool: np.ndarray,
    pool_bytes: Sequence[bytes],
) -> Dict[str, object]:
    """Cellule de la carte H62 d'un grain projeté : effectifs, régime, Var.

    Publie n_types, n_mesurables (n_occ ≥ 2), part_mesurable, masse
    d'occurrences couverte, régime CREUX/ESTIMABLE/VIDE (seuils gelés T73/G1),
    ``n_zero_bit`` (zéro exact par identité au bit des vecteurs hérités,
    instruction T73) avec écart résiduel float borné-publié, et les
    distributions ddof=1 (VERSION du tour) et ddof=0 (INFO continuité T73 —
    les deux colonnes, jamais mélangées).
    """
    n_total = len(unites)
    n_occ_total = sum(len(v) for v in unites.values())
    n_creuses = 0
    masse_mesurable = 0
    vals_ddof1: List[float] = []
    vals_ddof0: List[float] = []
    n_zero_bit = 0
    residus_zero_bit: List[float] = []
    for _ident, idx in unites.items():
        if len(idx) < 2:
            n_creuses += 1
            continue
        masse_mesurable += len(idx)
        mat = pool[np.asarray(idx, dtype=np.intp)]
        v1 = var_ctx_proj(mat)
        vals_ddof1.append(v1)
        vals_ddof0.append(var_ctx_ddof0_t73(mat))
        if len({pool_bytes[i] for i in idx}) == 1:
            n_zero_bit += 1
            residus_zero_bit.append(abs(v1))

    n_mes = len(vals_ddof1)
    part_mes = (n_mes / n_total) if n_total else None
    regime = "VIDE"
    if n_total:
        creuse = (part_mes < SEUIL_PART_MESURABLE) or (n_mes < SEUIL_N_MESURABLES)
        regime = "CREUX" if creuse else "ESTIMABLE"
    n_zero_flottant = sum(1 for v in vals_ddof1 if v == 0.0)
    return {
        "n_types": n_total,
        "n_mesurables": n_mes,
        "n_creuses": n_creuses,
        "part_mesurable": part_mes,
        "n_occurrences_total": n_occ_total,
        "masse_occ_couverte": (masse_mesurable / n_occ_total) if n_occ_total else None,
        "regime_h62": regime,
        "n_zero_bit": n_zero_bit,
        "max_var_residuelle_zero_bit": max(residus_zero_bit) if residus_zero_bit else None,
        "INFO_n_var_zero_flottant": n_zero_flottant,
        "var_ctx_proj_ddof1": _quartiles77(vals_ddof1) if vals_ddof1 else None,
        "INFO_continuite_t73_ddof0": _quartiles77(vals_ddof0) if vals_ddof0 else None,
    }


# ---------------------------------------------------------------------------
# Verdict TENUE (§4.5) — nulle d'échangeabilité, l.342
# ---------------------------------------------------------------------------

def verdict_tenue(
    unites: "OrderedDict[object, List[int]]",
    pool: np.ndarray,
    graine: int = GRAINE_SHUFFLE,
    repl: int = R_SHUFFLE,
) -> Dict[str, object]:
    """V_med observée vs R permutations du multiset des c(occ) entre hôtes.

    Le pool = TOUTES les occurrences-hôtes (g0_len ≥ 1) du corpus ; la
    permutation réassigne les vecteurs aux hôtes : la structure d'héritage
    est conservée (toutes les sous-unités d'un hôte partagent encore le même
    vecteur — T1 préservé par construction). Types et effectifs inchangés :
    seuls les vecteurs bougent. ``p_emp = (#{V^r_med ≤ V_obs} + 1)/(R + 1)``.

    Le VERDICT (TENUE/ANTI-TENUE/INDISTINCT) est retourné ici ; sa
    PRONONCIATION (eve seul, chaîne ESTIMABLE) appartient au pilote (§4.5).
    """
    if pool.ndim != 2 or pool.shape[0] == 0:
        raise ValueError(f"pool d'hôtes vide ou non 2-D : shape={pool.shape}")
    mes = [
        (ident, np.asarray(idx, dtype=np.intp))
        for ident, idx in unites.items()
        if len(idx) >= 2
    ]
    if not mes:
        raise ValueError("aucun type mesurable (n_occ >= 2) : V_med non définie")

    v_types = [var_ctx_proj(pool[idx]) for _, idx in mes]
    v_obs = float(np.median(v_types))

    rng = np.random.default_rng(graine)
    v_meds: List[float] = []
    for _ in range(repl):
        perm = rng.permutation(pool.shape[0])
        pp = pool[perm]
        v_meds.append(float(np.median([var_ctx_proj(pp[idx]) for _, idx in mes])))
    arr = np.asarray(v_meds, dtype=np.float64)
    n_inf_eq = int((arr <= v_obs).sum())
    p_emp = (n_inf_eq + 1) / (repl + 1)
    if p_emp <= SEUIL_P_TENUE:
        verdict = "TENUE"
    elif p_emp >= SEUIL_P_ANTI_TENUE:
        verdict = "ANTI-TENUE"
    else:
        verdict = "INDISTINCT"
    return {
        "graine": graine,
        "n_replicats": repl,
        "n_types_mesurables": len(mes),
        "n_hotes_pool": int(pool.shape[0]),
        "v_med_observee": v_obs,
        "INFO_distribution_observee": _quartiles77(v_types),
        "INFO_moyenne_observee": float(np.mean(v_types)),
        "distribution_v_med_shuffle": _quartiles77(arr),
        "n_shuffle_inferieurs_ou_egaux_obs": n_inf_eq,
        "p_emp": p_emp,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# λ — naissance sous G1 (§4.6), recette gelée zéro hyperparamètre
# ---------------------------------------------------------------------------

def recette_lambda(
    paires: "OrderedDict[Paire, List[int]]",
    pool: np.ndarray,
    cache: CacheStab,
) -> Dict[str, object]:
    """``λ* = IQR(H_norm) / IQR(Var_ctx_proj)`` sur LES MÊMES types mesurables.

    ``H_norm(paire) = H_norm(σ_unit(a+b))`` — fonction gelée T75
    (``CacheStab.h_norm``, zéro octet). Garde (branche déclarée) :
    IQR(Var_ctx_proj) = 0,0 ⇒ λ NON-MESURE.
    """
    mes = [(ident, idx) for ident, idx in paires.items() if len(idx) >= 2]
    if not mes:
        raise ValueError("aucun type de paire mesurable : recette λ non définie")
    var_vals = [var_ctx_proj(pool[np.asarray(idx, dtype=np.intp)]) for _, idx in mes]
    h_vals = [cache.h_norm(tuple(ident)) for ident, _ in mes]
    q_var = _quartiles77(var_vals)
    q_h = _quartiles77(h_vals)
    iqr_var = q_var["iqr"]
    iqr_h = q_h["iqr"]
    res: Dict[str, object] = {
        "n_types_paires_mesurables": len(mes),
        "iqr_h_norm": iqr_h,
        "iqr_var_ctx_proj": iqr_var,
        "distribution_h_norm": q_h,
        "distribution_var_ctx_proj": q_var,
    }
    if iqr_var == 0.0:
        res["lambda_star"] = None
        res["statut"] = "NON-MESURE (IQR(Var_ctx_proj) = 0.0, garde §4.6)"
    else:
        res["lambda_star"] = iqr_h / iqr_var
        res["statut"] = "CALIBRE"
    return res


# ---------------------------------------------------------------------------
# Δ_ctx et admissibilité Coherence_full (§4.7) — AUCUNE fusion exécutée
# ---------------------------------------------------------------------------

def delta_ctx_paires(
    paires: "OrderedDict[Paire, List[int]]",
    g0: "OrderedDict[int, List[int]]",
    pool: np.ndarray,
) -> Dict[str, object]:
    """Δ_ctx(a,b) = ½[Var(a) + Var(b)] − Var(a·b) sur paires aux 2 parts mesurables.

    Positif = la jonction tient mieux que ses parties. T3 : supports emboîtés
    — grâce à T2 (ddof=1), un écart à zéro est une ASSOCIATION, pas un effet
    d'effectif. Sélecteur §4.1 : paire mesurable ET parts mesurables ;
    comptes publiés des deux côtés.
    """
    var_g0: Dict[int, float] = {}
    for pid, idx in g0.items():
        if len(idx) >= 2:
            var_g0[pid] = var_ctx_proj(pool[np.asarray(idx, dtype=np.intp)])

    deltas: List[Tuple[Paire, float]] = []
    n_paires_creuses = 0
    n_part_creuse = 0
    for (a, b), idx in paires.items():
        if len(idx) < 2:
            n_paires_creuses += 1
            continue
        if a not in var_g0 or b not in var_g0:
            n_part_creuse += 1
            continue
        v_ab = var_ctx_proj(pool[np.asarray(idx, dtype=np.intp)])
        deltas.append(((a, b), 0.5 * (var_g0[a] + var_g0[b]) - v_ab))

    vals = [d for _, d in deltas]
    n_pos = sum(1 for d in vals if d > 0.0)
    n_neg = sum(1 for d in vals if d < 0.0)
    n_zero = sum(1 for d in vals if d == 0.0)
    return {
        "n_types_paires": len(paires),
        "n_paires_creuses_exclues": n_paires_creuses,
        "n_paires_part_creuse_exclues": n_part_creuse,
        "n_paires_delta_mesurables": len(deltas),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "coupe": bool(n_pos >= 1 and n_neg >= 1),
        "distribution": _quartiles77(vals) if vals else None,
        "deltas": deltas,
    }


def admissibilite_coherence_full(
    deltas: Sequence[Tuple[Paire, float]],
    cache: CacheStab,
    lambda_star: float,
    table: Sequence[dict],
    top_n: int = 10,
) -> Dict[str, object]:
    """Coherence_full(a,b) = Coherence_H(a,b) + λ*·Δ_ctx(a,b) — admissibilité.

    Coherence_H = fonction gelée T75 (théorème T76 : ≤ −0,0009 sur le réel ⇒
    tout positif de Coherence_full vient du terme contextuel, lisible par
    construction). AUCUNE fusion n'est exécutée. Départage du top :
    (−valeur, identité) — déterministe.
    """
    lignes: List[dict] = []
    for (a, b), d in deltas:
        coh_h = cache.coherence((a,), (b,))
        lignes.append(
            {
                "ident": [a, b],
                "rendu_ascii": str(table[a]["ascii"]) + str(table[b]["ascii"]),
                "coherence_h": coh_h,
                "delta_ctx": d,
                "coherence_full": coh_h + lambda_star * d,
            }
        )
    vals = [l["coherence_full"] for l in lignes]
    n_pos = sum(1 for v in vals if v > 0.0)
    n_neg = sum(1 for v in vals if v < 0.0)
    n_zero = sum(1 for v in vals if v == 0.0)
    top = sorted(lignes, key=lambda l: (-l["coherence_full"], tuple(l["ident"])))[:top_n]
    return {
        "lambda_star": lambda_star,
        "n_paires": len(lignes),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "distribution": _quartiles77(vals) if vals else None,
        "top_paires": top,
    }


def info_types_mots_hotes(
    paires: "OrderedDict[Paire, List[int]]",
    hote_mot: Sequence[str],
) -> Dict[str, object]:
    """INFO §2 (hors verdict) : nb de types de mots-hôtes par paire mesurable.

    Médiane publiée SANS verdict — la séparation lexical/contextuel est un
    tour futur (R-INSTRUMENT strict).
    """
    nb = [
        len({hote_mot[i] for i in idx})
        for idx in paires.values()
        if len(idx) >= 2
    ]
    return {
        "n_paires_mesurables": len(nb),
        "mediane_types_mots_hotes": float(np.median(nb)) if nb else None,
        "distribution": _quartiles77([float(x) for x in nb]) if nb else None,
    }


# ---------------------------------------------------------------------------
# Théorèmes T1/T2 (§4.3) — la garde devient propriété (A76-1)
# ---------------------------------------------------------------------------

def temoin_theoreme_t1() -> Dict[str, object]:
    """T1 (héritage constant) sur témoin construit : les sous-unités d'un même
    hôte partagent le MÊME c(occ) ; une unité dont toutes les occurrences
    vivent dans des hôtes bit-identiques a Var_ctx_proj = 0 exact (au bit)."""
    def _tok(text: str, ctx: Tuple[float, float, float], g0_ids: List[int]) -> dict:
        v = np.zeros(33, dtype=np.float32)
        v[28], v[29], v[30] = ctx
        return {"text": text, "vector33d": v, "g0_ids": g0_ids, "g1_nb": [len(g0_ids)]}

    # Deux hôtes bit-identiques portant « aaa » : la paire (0,0) a 4
    # occurrences (chevauchement compris), toutes héritées du même vecteur.
    proj = extraire_projection(
        [[_tok("aaa", (1.0, 2.0, 3.0), [0, 0, 0])],
         [_tok("aaa", (1.0, 2.0, 3.0), [0, 0, 0])]]
    )
    idx = proj["paire"][(0, 0)]
    bytes_uniques = {proj["pool_bytes"][i] for i in idx}
    heritage_constant = all(
        proj["pool_bytes"][i] == proj["pool_bytes"][idx[0]] for i in idx
    )
    v = var_ctx_proj(proj["pool"][np.asarray(idx, dtype=np.intp)])
    return {
        "n_occurrences_paire": len(idx),
        "chevauchement_2_par_hote": len(idx) == 4,
        "heritage_constant": heritage_constant,
        "zero_au_bit": len(bytes_uniques) == 1,
        "var_ctx_proj": v,
        "tenu": heritage_constant and len(bytes_uniques) == 1 and v == 0.0,
    }


def temoin_theoreme_t2(
    graine: int = GRAINE_TEMOINS,
    n_pool: int = 16,
    tailles: Tuple[int, ...] = (2, 3, 4),
) -> Dict[str, object]:
    """T2 : sous la nulle d'échangeabilité, E[s²_ddof1] ne dépend pas de n.

    Pool synthétique seedé (graine 77003) ; l'espérance sous tirage sans
    remise est atteinte EXACTEMENT par énumération EXHAUSTIVE des
    sous-ensembles (C(16,2)+C(16,3)+C(16,4) = 2 500 ≥ 1 000 tirages) :
    ``mean_{|S|=n} s²_ddof1(S) = S²_pool(ddof=1)`` pour tout n — écart ≤ 1e-9
    au sens strict. CONTRÔLE POSITIF (deux branches) : le même montage sous
    ddof=0 DOIT montrer le biais (n−1)/n — sinon T2 est inopposable et Δ_ctx
    NON-MESURE. INFO : moyennes Monte-Carlo sur 1 000 tirages seedés
    (même rng, à la suite) publiées hors-vérification.
    """
    rng = np.random.default_rng(graine)
    pool = rng.normal(size=n_pool)
    s2_pool = float(pool.var(ddof=1))

    par_taille: Dict[str, dict] = {}
    n_tirages_total = 0
    ecart_max_ddof1 = 0.0
    biais_ok = True
    for n in tailles:
        vals1: List[float] = []
        vals0: List[float] = []
        for comb in itertools.combinations(range(n_pool), n):
            sub = pool[list(comb)]
            vals1.append(float(sub.var(ddof=1)))
            vals0.append(float(sub.var(ddof=0)))
        n_tirages_total += len(vals1)
        m1 = float(np.mean(vals1))
        m0 = float(np.mean(vals0))
        ecart1 = abs(m1 - s2_pool)
        ecart0 = abs(m0 - (n - 1) / n * s2_pool)
        ecart_max_ddof1 = max(ecart_max_ddof1, ecart1)
        if not (ecart0 <= TOL_THEOREME_T2 and m0 < m1):
            biais_ok = False
        # INFO Monte-Carlo (1 000 tirages seedés, même rng, hors-vérification)
        mc = [
            float(pool[rng.permutation(n_pool)[:n]].var(ddof=1))
            for _ in range(1000)
        ]
        par_taille[str(n)] = {
            "n_sous_ensembles": len(vals1),
            "moyenne_s2_ddof1": m1,
            "ecart_ddof1": ecart1,
            "moyenne_s2_ddof0": m0,
            "biais_attendu_ddof0": (n - 1) / n * s2_pool,
            "ecart_ddof0_au_biais": ecart0,
            "INFO_moyenne_monte_carlo_ddof1_1000_tirages": float(np.mean(mc)),
        }
    return {
        "graine_pool": graine,
        "n_pool": n_pool,
        "variance_pool_ddof1": s2_pool,
        "note_interpretation": (
            "tirages = énumération EXHAUSTIVE des sous-ensembles (2 500 ≥ 1 000) ; "
            "la graine 77003 seede le POOL — l'espérance est atteinte exactement, "
            "l'écart ≤ 1e-9 est vérifiable au sens strict (déclaré)"
        ),
        "n_tirages_total": n_tirages_total,
        "par_taille": par_taille,
        "ecart_max_ddof1": ecart_max_ddof1,
        "branche_ddof1_tenue": ecart_max_ddof1 <= TOL_THEOREME_T2,
        "branche_ddof0_biais_visible": biais_ok,
        "tenu": (ecart_max_ddof1 <= TOL_THEOREME_T2) and biais_ok,
    }


# ---------------------------------------------------------------------------
# PORTÉES (§7) — deux branches par instrument, témoins construits
# ---------------------------------------------------------------------------

def _pool_temoin(vecteurs: Sequence[Tuple[float, float, float]]):
    pool = np.asarray(vecteurs, dtype=np.float64)
    pool_bytes = [pool[i].tobytes() for i in range(pool.shape[0])]
    return pool, pool_bytes


def portees_t77() -> Dict[str, object]:
    """PORTÉE de chaque instrument du tour (§7) — publiée AVANT toute mesure.

    Chaque branche est ATTEINTE sur témoins construits. Gap δ_min sous
    shuffle : N-A déclaré (§4.5) — Var et médiane sont invariantes par
    permutation des occurrences, aucun verdict d'ordre A→B→A′ ce tour.
    """
    res: Dict[str, object] = {
        "gap_shuffle_ordre": (
            "N-A déclaré (§4.5) : aucun verdict d'ordre/clôture A→B→A′ ce tour "
            "— Var_ctx_proj et V_med sont invariantes par permutation des occurrences"
        )
    }

    # --- Var_ctx_proj : branche 0 exact (bit) ET branche > 0 ; aveugle à l'ordre
    pool0, _ = _pool_temoin([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0)])
    v_zero = var_ctx_proj(pool0)
    pool1, _ = _pool_temoin([(1.0, 2.0, 3.0), (3.0, 2.0, 1.0)])
    v_pos = var_ctx_proj(pool1)
    v_pos_permute = var_ctx_proj(pool1[::-1])
    res["var_ctx_proj"] = {
        "branche_zero_bit": {"valeur": v_zero, "atteinte": v_zero == 0.0},
        "branche_positive": {"valeur": v_pos, "atteinte": v_pos > 0.0},
        "aveugle_a_l_ordre": v_pos == v_pos_permute,
    }

    # --- Classifieur H62 : 3 branches (VIDE / CREUX / ESTIMABLE) ------------
    pool_h, bytes_h = _pool_temoin([(float(i), 0.0, 0.0) for i in range(64)])
    vide = stats_grain_projete(OrderedDict(), pool_h, bytes_h)
    creux = stats_grain_projete(
        OrderedDict([("u0", [0, 1]), ("u1", [2]), ("u2", [3])]), pool_h, bytes_h
    )
    estimable = stats_grain_projete(
        OrderedDict((f"u{i}", [2 * i, 2 * i + 1]) for i in range(30)),
        pool_h, bytes_h,
    )
    res["classifieur_h62"] = {
        "branche_vide": vide["regime_h62"],
        "branche_creux": creux["regime_h62"],
        "branche_estimable": estimable["regime_h62"],
        "atteintes": (
            vide["regime_h62"] == "VIDE"
            and creux["regime_h62"] == "CREUX"
            and estimable["regime_h62"] == "ESTIMABLE"
        ),
    }

    # --- Shuffle / p_emp : DEUX branches + déterminisme ×2 ------------------
    # Association injectée : 10 types × 4 hôtes, vecteurs identiques PAR type
    # ⇒ V_obs = 0 exact, les shuffles dispersent ⇒ p_emp ≤ 0,05.
    unites_assoc: "OrderedDict[object, List[int]]" = OrderedDict(
        (f"t{t}", [4 * t + j for j in range(4)]) for t in range(10)
    )
    pool_assoc, _ = _pool_temoin(
        [(float(t), float(t), float(t)) for t in range(10) for _ in range(4)]
    )
    tenue_assoc = verdict_tenue(unites_assoc, pool_assoc)
    # Anti-association injectée : dispersion intra-type MAXIMALE par
    # construction (chaque type = {+1, −1}, la variance de paire est majorée
    # par la valeur observée) ⇒ tout shuffle a V^r_med ≤ V_obs ⇒ p_emp ≥ 0,95.
    unites_anti: "OrderedDict[object, List[int]]" = OrderedDict(
        (f"t{t}", [2 * t, 2 * t + 1]) for t in range(10)
    )
    pool_anti, _ = _pool_temoin(
        [((1.0, 1.0, 1.0) if j == 0 else (-1.0, -1.0, -1.0)) for _t in range(10) for j in range(2)]
    )
    tenue_anti = verdict_tenue(unites_anti, pool_anti)
    tenue_assoc_bis = verdict_tenue(unites_assoc, pool_assoc)
    res["shuffle_p_emp"] = {
        "branche_association": {
            "v_obs": tenue_assoc["v_med_observee"],
            "p_emp": tenue_assoc["p_emp"],
            "atteinte": tenue_assoc["p_emp"] <= SEUIL_P_TENUE,
        },
        "branche_anti_association": {
            "v_obs": tenue_anti["v_med_observee"],
            "p_emp": tenue_anti["p_emp"],
            "atteinte": tenue_anti["p_emp"] >= SEUIL_P_ANTI_TENUE,
        },
        "determinisme_x2_p_emp": tenue_assoc["p_emp"] == tenue_assoc_bis["p_emp"],
        "estimateur_verdict": "mediane (pointe gelee, §4.5)",
    }

    # --- Recette λ : branche calculable ET branche IQR = 0 (garde mordante) -
    cache_temoin = CacheStab(TABLE_TEMOIN)
    pool_l, _ = _pool_temoin(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (3.0, 3.0, 3.0),
         (0.0, 0.0, 0.0), (5.0, 5.0, 5.0)]
    )
    paires_l: "OrderedDict[Paire, List[int]]" = OrderedDict(
        [((0, 1), [0, 1]), ((0, 2), [2, 3]), ((1, 2), [4, 5])]
    )
    lam_calc = recette_lambda(paires_l, pool_l, cache_temoin)
    pool_l0, _ = _pool_temoin([(0.0, 0.0, 0.0)] * 6)
    lam_zero = recette_lambda(paires_l, pool_l0, cache_temoin)
    res["recette_lambda"] = {
        "branche_calculable": {
            "lambda_star": lam_calc["lambda_star"],
            "atteinte": lam_calc["statut"] == "CALIBRE",
        },
        "branche_iqr_zero": {
            "statut": lam_zero["statut"],
            "atteinte": lam_zero["lambda_star"] is None,
        },
    }

    # --- Δ_ctx : jonction qui tient / qui ne tient pas ----------------------
    # Tient : les parts (g0) dispersées, la paire dans des hôtes identiques.
    pool_d, _ = _pool_temoin(
        [(0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (5.0, 5.0, 5.0), (5.0, 5.0, 5.0)]
    )
    d_tient = delta_ctx_paires(
        OrderedDict([((0, 1), [2, 3])]),
        OrderedDict([(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3])]),
        pool_d,
    )
    # Ne tient pas : la paire vit dans les hôtes EXTRÊMES, les parts partout.
    d_tient_pas = delta_ctx_paires(
        OrderedDict([((0, 1), [0, 1])]),
        OrderedDict([(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3])]),
        pool_d,
    )
    res["delta_ctx"] = {
        "branche_positive": {
            "delta": d_tient["deltas"][0][1],
            "atteinte": d_tient["deltas"][0][1] > 0.0,
        },
        "branche_negative": {
            "delta": d_tient_pas["deltas"][0][1],
            "atteinte": d_tient_pas["deltas"][0][1] < 0.0,
        },
    }

    # --- Théorèmes T1 / T2 (contrôle positif ddof=0 compris) ----------------
    res["theoreme_T1"] = temoin_theoreme_t1()
    res["theoreme_T2"] = temoin_theoreme_t2()
    return res


# ---------------------------------------------------------------------------
# Étalonnage du mur (§13)
# ---------------------------------------------------------------------------

def etalonner_t77(chemin_eve: str) -> Dict[str, object]:
    """t_cal = chargement du .so (garde sha comprise) + tokenisation +
    extraction + carte paire + 1 réplicat de shuffle, sur les 9 premières
    phrases d'eve. Mur du run complet = min(20 000 × t_cal, 1 800 s)."""
    ctx, _ = contextes_eve(chemin_eve)
    t0 = time.perf_counter()
    tok, _prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
    tokens = [tok.tokenize_sequences(c) for c in ctx[:N_PHRASES_CAL]]
    proj = extraire_projection(tokens)
    carte_paire = stats_grain_projete(
        proj["paire"], proj["pool"], proj["pool_bytes"]
    )
    # 1 réplicat de shuffle (structure du §4.5, R = 1, même graine gelée).
    n_mes = carte_paire["n_mesurables"]
    if n_mes:
        _ = verdict_tenue(proj["paire"], proj["pool"], graine=GRAINE_SHUFFLE, repl=1)
    t_cal = time.perf_counter() - t0
    return {
        "n_phrases": N_PHRASES_CAL,
        "n_tokens": proj["comptes"]["n_tokens"],
        "n_paires_mesurables_tranche": int(n_mes),
        "t_cal_s": t_cal,
        "facteur_mur": FACTEUR_MUR,
        "mur_s": min(FACTEUR_MUR * t_cal, PLAFOND_DUR_S),
        "plafond_dur_s": PLAFOND_DUR_S,
    }


# ---------------------------------------------------------------------------
# Pilote (artefacts TOUR77_* — racine écosystème, sans horodatage)
# ---------------------------------------------------------------------------

def _md5(chemin: str) -> str:
    with open(chemin, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _config_echo(md5_eve: Optional[str], md5_claude: Optional[str],
                 so_sha256: Optional[str]) -> Dict[str, object]:
    """Écho de configuration complet (invariant §9 T76 : contexte déclaré).

    Volontairement SANS horodatage : les artefacts de verdict sont
    reproductibles AU BIT (C1)."""
    return {
        "tour": 77,
        "graine_shuffle": GRAINE_SHUFFLE,
        "r_shuffle": R_SHUFFLE,
        "graine_temoins": GRAINE_TEMOINS,
        "seuil_p_tenue": SEUIL_P_TENUE,
        "seuil_p_anti_tenue": SEUIL_P_ANTI_TENUE,
        "seuil_part_mesurable": SEUIL_PART_MESURABLE,
        "seuil_n_mesurables": SEUIL_N_MESURABLES,
        "ctx_dims": list(CTX_DIMS),
        "ddof_verdict": 1,
        "ddof_info_continuite_t73": 0,
        "facteur_mur": FACTEUR_MUR,
        "plafond_dur_s": PLAFOND_DUR_S,
        "so_sha256": so_sha256,
        "so_sha256_reference_t75": SHA256_SO_REFERENCE_T75,
        "md5_corpus_eve": md5_eve,
        "md5_corpus_claude_aba": md5_claude,
        "contexte_execution": {
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "plateforme": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "locale": list(_locale.getlocale()),
            "encodage_prefere": _locale.getpreferredencoding(False),
        },
    }


def _ecrire_json(chemin: str, obj: Dict[str, object]) -> None:
    with open(chemin, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def _cle_paire(ident: Paire) -> str:
    return f"{ident[0]},{ident[1]}"


def run_complet(
    chemin_eve: str,
    chemin_claude: str,
    mur_s: Optional[float],
    dossier_sortie: str,
) -> Dict[str, str]:
    """Le run gelé (§10 P4-P7) : carte H62 → tenue → λ → Δ_ctx/admissibilité.

    Verdicts sur EVE SEUL (chaîne §4.5 : prononcés ssi paire×eve ESTIMABLE) ;
    claude_aba publié aux mêmes grandeurs, RANG INFO (clause T42). Mur §13 :
    dépassement ⇒ I5, artefacts restants marqués INOPPOSABLES.
    """
    import os

    t0 = time.perf_counter()

    def _mur_atteint() -> bool:
        return mur_s is not None and (time.perf_counter() - t0) > mur_s

    tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
    table = table_miroir()
    cache = CacheStab(table)
    md5_eve, md5_claude = _md5(chemin_eve), _md5(chemin_claude)
    config = _config_echo(md5_eve, md5_claude, prov["native_lib_sha256"])

    ctx_e, cpt_e = contextes_eve(chemin_eve)
    ctx_c, cpt_c = contextes_claude_aba(chemin_claude)
    proj_e = extraire_projection(tok.tokenize_sequences(c) for c in ctx_e)
    proj_c = extraire_projection(tok.tokenize_sequences(c) for c in ctx_c)

    chemins: Dict[str, str] = {}

    # --- P4 : carte H62 (C2) — grains × corpus, effectifs complets ----------
    def _carte(proj: Dict[str, object]) -> Dict[str, object]:
        pool, pb = proj["pool"], proj["pool_bytes"]
        return {
            "comptes": proj["comptes"],
            "paire": stats_grain_projete(proj["paire"], pool, pb),
            "g0_projete": stats_grain_projete(proj["g0"], pool, pb),
            "syllabe_projete_INFO": stats_grain_projete(proj["syllabe"], pool, pb),
        }

    carte_obj = {
        "config": config,
        "contextes": {"eve": cpt_e, "claude_aba": cpt_c},
        "eve": _carte(proj_e),
        "claude_aba": _carte(proj_c),
        "INFO_inventaire_T75": {
            "note": (
                "T75 comptait les paires par TYPE de mot, T77 par OCCURRENCE "
                "de token (§4.7) — comparaison d'inventaires, jamais une "
                "grandeur de verdict"
            ),
            "n_types_paires_T75": {"eve": 539, "claude_aba": 369},
            "n_types_paires_T77": {
                "eve": len(proj_e["paire"]),
                "claude_aba": len(proj_c["paire"]),
            },
        },
        "INFO_types_mots_hotes_par_paire": {
            "eve": info_types_mots_hotes(proj_e["paire"], proj_e["hote_mot"]),
            "claude_aba": info_types_mots_hotes(proj_c["paire"], proj_c["hote_mot"]),
        },
    }
    p = os.path.join(dossier_sortie, "TOUR77_CARTE_H62.json")
    _ecrire_json(p, carte_obj)
    chemins["TOUR77_CARTE_H62.json"] = p

    regime_eve = carte_obj["eve"]["paire"]["regime_h62"]
    estimable_eve = regime_eve == "ESTIMABLE"

    # --- P5 : verdict TENUE (§4.5) — eve seul, chaîne d'amont déclarée ------
    tenue_obj: Dict[str, object] = {"config": config}
    if _mur_atteint():
        tenue_obj["issue"] = "I5_MUR (§13) : verdict tenue INOPPOSABLE, état tronqué INFO"
    else:
        tenue_e = verdict_tenue(proj_e["paire"], proj_e["pool"])
        if estimable_eve:
            tenue_obj["eve"] = tenue_e
            tenue_obj["verdict_prononce"] = tenue_e["verdict"]
        else:
            tenue_e_info = dict(tenue_e)
            tenue_e_info["verdict"] = (
                "NON-MESURE (chaîne §4.5 : paire×eve " + regime_eve + " → I3-SOL)"
            )
            tenue_obj["eve"] = tenue_e_info
            tenue_obj["verdict_prononce"] = "NON-MESURE (I3-SOL)"
        tenue_obj["claude_aba_INFO"] = verdict_tenue(proj_c["paire"], proj_c["pool"])
        tenue_obj["claude_aba_INFO"]["rang"] = (
            "INFO (clause T42 : jamais une confirmation partielle)"
        )
    p = os.path.join(dossier_sortie, "TOUR77_TENUE.json")
    _ecrire_json(p, tenue_obj)
    chemins["TOUR77_TENUE.json"] = p

    # --- P6 : λ sous G1 (§4.6) — couple paire×eve seul, aucun repli ---------
    lambda_obj: Dict[str, object] = {"config": config}
    if _mur_atteint():
        lambda_obj["issue"] = "I5_MUR (§13) : λ INOPPOSABLE"
        lam_star: Optional[float] = None
    elif not estimable_eve:
        lambda_obj["lambda"] = {
            "lambda_star": None,
            "statut": "NON-MESURE (G1 : paire×eve " + regime_eve + ", aucun repli)",
        }
        lam_star = None
    else:
        lam = recette_lambda(proj_e["paire"], proj_e["pool"], cache)
        lambda_obj["lambda"] = lam
        lam_star = lam["lambda_star"]
    p = os.path.join(dossier_sortie, "TOUR77_LAMBDA.json")
    _ecrire_json(p, lambda_obj)
    chemins["TOUR77_LAMBDA.json"] = p

    # --- P7 : Δ_ctx (P-3) + admissibilité Coherence_full (P-4) --------------
    adm_obj: Dict[str, object] = {"config": config}
    if _mur_atteint():
        adm_obj["issue"] = "I5_MUR (§13) : Δ_ctx/admissibilité INOPPOSABLES"
    else:
        d_e = delta_ctx_paires(proj_e["paire"], proj_e["g0"], proj_e["pool"])
        d_c = delta_ctx_paires(proj_c["paire"], proj_c["g0"], proj_c["pool"])
        deltas_e = d_e.pop("deltas")
        deltas_c = d_c.pop("deltas")
        adm_obj["delta_ctx_eve"] = d_e
        adm_obj["delta_ctx_claude_aba_INFO"] = d_c
        if lam_star is None:
            adm_obj["coherence_full_eve"] = {
                "statut": "NON-MESURE (chaîne §5 P-4 : λ NON-MESURE)",
            }
        else:
            adm_obj["coherence_full_eve"] = admissibilite_coherence_full(
                deltas_e, cache, lam_star, table
            )
            adm_obj["coherence_full_claude_aba_INFO"] = admissibilite_coherence_full(
                deltas_c, cache, lam_star, table
            )
            adm_obj["coherence_full_claude_aba_INFO"]["rang"] = (
                "INFO (λ* né sur eve, clause T42)"
            )
        adm_obj["note"] = "AUCUNE fusion exécutée ce tour (§4.7) — admissibilité seulement"
    p = os.path.join(dossier_sortie, "TOUR77_ADMISSIBILITE.json")
    _ecrire_json(p, adm_obj)
    chemins["TOUR77_ADMISSIBILITE.json"] = p
    return chemins


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stade", required=True, choices=["portees", "etalonnage", "run"])
    ap.add_argument("--eve", default=None, help="chemin de corpus_eve_clean.txt")
    ap.add_argument("--claude-aba", default=None, help="chemin de corpus_claude_aba.txt")
    ap.add_argument("--sortie", required=True,
                    help="fichier JSON (portees/etalonnage) ou dossier (run)")
    ap.add_argument("--mur-s", type=float, default=None,
                    help="mur du run (depuis TOUR77_ETALONNAGE.json)")
    args = ap.parse_args(argv)

    if args.stade == "portees":
        # Garde sha AVANT toute mesure ; branche mismatch prouvée mordante.
        _tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
        obj: Dict[str, object] = {
            "config": _config_echo(None, None, prov["native_lib_sha256"]),
            "portees": portees_t77(),
        }
        try:
            charger_tokenizer_garde(sha256_attendu="0" * 64)
            obj["garde_sha_so"] = {"mordante": False}
        except RuntimeError:
            obj["garde_sha_so"] = {
                "mordante": True,
                "clause": "1 mismatch => 0 mesure prise, tour NON-MESURE (§7)",
            }
        # Portées T75 de Coherence_H rejouées telles quelles (module gelé).
        from spiraton.experimental.bpe_logos import portees_instruments

        p75 = portees_instruments(table_miroir())
        obj["coherence_h_portees_t75"] = {
            "coherence": p75["coherence"],
            "propriete_A1": p75["propriete_A1"],
            "propriete_A2": p75["propriete_A2"],
            "support_reel": p75["support_reel"],
        }
        _ecrire_json(args.sortie, obj)
    elif args.stade == "etalonnage":
        _ecrire_json(args.sortie, {"tour": 77, "etalonnage": etalonner_t77(args.eve)})
    else:
        chemins = run_complet(args.eve, args.claude_aba, args.mur_s, args.sortie)
        for nom, chemin in chemins.items():
            print(f"{nom}: {chemin}")


if __name__ == "__main__":  # pragma: no cover - pilote
    main()
