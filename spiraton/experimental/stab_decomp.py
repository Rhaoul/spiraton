"""Décomposition intra/inter de l'ANTI-TENUE T77 — instrument du Tour 78.

Ce module DÉCOMPOSE la grandeur gravée au T77 (V_med, ANTI-TENUE p_emp 0,990)
en ses deux composantes par rapport à la partition par TYPE DE MOT-HÔTE
(``hote_mot``, gel T73) : la part INTRA-type (héritée) et la part INTER-type
(lexicale), chacune devant sa nulle. Toutes les définitions sont les gels de
``TOUR78_EMISSION.md`` (md5 ``f7031f78c3d06c04ca0be3dce1c0a43c``) — rien
n'est réglable en douce. Le verdict T77 reste GRAVÉ : rien n'est rejugé ici.

Définitions opérationnelles gelées (émission T78) :

- **Strate** (§5.1) : type de mot-hôte m = ``hote_mot`` de l'occurrence
  (gel T73, ``text.lower()``) ; n_m = nb d'occurrences de l'unité dans m ;
  N = Σ n_m.
- **Colonne POP (verdict de décomposition)** — par dim d puis moyenne sur
  les 3 dims : ``Var_pop`` (ddof=0, toutes occurrences), ``V_intra_pop =
  Σ_m (n_m/N)·s²_pop,d,m`` (strates singleton incluses, contribution 0),
  ``V_inter_pop = Σ_m (n_m/N)·(μ_m,d − μ_d)²``. **Théorème T4 (loi de
  variance totale, exact en POP)** : ``Var_pop = V_intra_pop + V_inter_pop``
  par paire ET par dim (tolérance ``TOL_IDENTITE`` = 1e-9 ; contrôle positif
  = poids uniformes 1/M sur témoin déséquilibré ⇒ identité violée).
- **Colonne D1 (Δ_ctx_intra, λ*_intra)** : ``V_intra^{d1} =
  Σ_{m∈S₂} (n_m/N₂)·s²_ddof1,d,m`` avec S₂ = {m : n_m ≥ 2}, N₂ = Σ n_m.
  Sélecteur : unité **intra-mesurable** ssi S₂ ≠ ∅ (comptes publiés des deux
  côtés). **Théorème T5** : sous la nulle globale, E[s²_ddof1 de tout
  sous-ensemble] = S² du pool ⇒ E[V_intra^{d1}] = S²_global ⇒
  E[Δ_ctx_intra] = 0 exactement.
- **Théorème T6 (invariance)** : si T6-pré tient, le shuffle stratifié est
  un no-op exact sur V_intra_pop/V_inter_pop/Var_pop des unités réelles
  (strates pleines) — écart ≤ ``TOL_INVARIANCE`` = 1e-12 ; il DÉPLACE les
  unités à strates partielles (constructibles en synthétique seulement).
- **T6-pré** (§5.3, mesure) : pour chaque type de mot-hôte, la séquence
  ``g0_ids`` est identique sur toutes ses occurrences (tokenisation
  déterministe par texte, R43/T73/T74). Publié : n_types, n_violations.
- **GC-1** (§5.4, garde de chaîne) : la grandeur T77 exacte (V_med ddof=1,
  486 mesurables sur eve) doit être ÉGALE AU BIT (==) à
  ``TOUR77_TENUE.json``. Mismatch ⇒ tour NON-MESURE (``NonMesureError``).
- **Verdict de décomposition** (§5.5, EVE SEUL) : V_med_intra/V_med_inter =
  MÉDIANES (pointe gelée) de V_intra_pop/V_inter_pop sur les mêmes types
  mesurables ; nulle GLOBALE (forme T77, graine NEUVE 78201) rejouée par
  composantes, recalcul CONJOINT des deux médianes sur chaque permutation ;
  ``p_intra/p_inter = (#{V^r ≤ V_obs} + 1)/(R+1)`` ; table des cases a
  priori (DÉCOMPOSÉE / INVERSION / INTER-SEULE / INTRA-SEULE /
  NON-LOCALISÉE). claude_aba : mêmes grandeurs, RANG INFO (clause T42).
- **Contrôle stratifié** (§5.6, graine 78202) : permutation des c(occ) À
  L'INTÉRIEUR de chaque type de mot-hôte (ordre = première apparition
  corpus) — contrôle du théorème T6, JAMAIS un verdict (un p_emp dessus
  vaudrait 1,0 par construction : observable à support dégénéré).
- **Audit du confondant** (§5.7) : ρ_S = Spearman(Δ_ctx,
  n_types_mots_hôtes) à rangs moyens, deux routes croisées (accord ≤ 1e-12),
  seuil |ρ_S| ≥ 0,80, conséquence gelée dans les deux sens.
- **λ*_intra** (§5.8, G1) : IQR(H_norm)/IQR(V_intra^{d1}) sur les mêmes
  types intra-mesurables d'eve ssi couple intra×eve ESTIMABLE ; IQR = 0 ⇒
  NON-MESURE. **Δ_ctx_intra** (§5.9) : ½[V^{d1}(a)+V^{d1}(b)] − V^{d1}(a·b)
  sur paires aux DEUX parts intra-mesurables ; Coherence_full_intra =
  Coherence_H + λ*_intra·Δ_ctx_intra (AUCUNE fusion exécutée).

AUCUN verdict d'ordre/clôture A→B→A′ ce tour : toutes les grandeurs sont
invariantes par permutation des occurrences à strates fixées (gap δ_min
N-A motivé §5.5). R-INSTRUMENT strict : tout ce que ce module mesure est
une propriété du triplet (projection × corpus × interface R16/33D).

Réutilisation ZÉRO OCTET (imports seuls) : ``stab_ctx`` (extraction,
Var_ctx_proj, Δ_ctx, quartiles), ``stab_grains`` (contextes, garde sha,
miroir), ``bpe_logos`` (CacheStab, sha référence T75). Contexte d'exécution
déclaré dans chaque artefact (invariant §9) ; artefacts de verdict SANS
horodatage, reproductibles au bit (C1).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import locale as _locale
import os
import platform
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

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
from spiraton.experimental.stab_ctx import (
    GRAINE_SHUFFLE as GRAINE_SHUFFLE_T77,
    SEUIL_P_ANTI_TENUE,
    SEUIL_P_TENUE,
    _quartiles77,
    admissibilite_coherence_full,
    delta_ctx_paires,
    extraire_projection,
    var_ctx_proj,
)
from spiraton.experimental.bpe_logos import (
    SHA256_SO_REFERENCE_T75,
    CacheStab,
    TABLE_TEMOIN,
)

# ---------------------------------------------------------------------------
# Constantes gelées du tour (émission T78 — jamais cachées, jamais réglables)
# ---------------------------------------------------------------------------

#: Graine de la nulle GLOBALE (§5.5) — default_rng unique, réplicats en ordre.
GRAINE_NULLE_GLOBALE: int = 78201

#: Graine du contrôle stratifié (§5.6).
GRAINE_CONTROLE_STRATIFIE: int = 78202

#: Graine des témoins de portées/théorèmes (§5.2, §8).
GRAINE_TEMOINS: int = 78003

#: Nombre de réplicats des deux exécutions (§5.5, §5.6).
R_NULLE: int = 100

#: Tolérance de l'identité T4 (§5.2) et de T5 ; violation attendue du
#: contrôle positif > SEUIL_VIOLATION.
TOL_IDENTITE: float = 1e-9
SEUIL_VIOLATION: float = 1e-6

#: Tolérance d'invariance T6 (§5.6) — ordre de sommation flottante seul.
TOL_INVARIANCE: float = 1e-12

#: Accord exigé entre les deux routes de Spearman (§5.7).
TOL_SPEARMAN_ROUTES: float = 1e-12

#: Seuil du confondant (§5.7) — précédent T75 « la variable par la bande ».
SEUIL_RHO_SPEARMAN: float = 0.80

#: Dénominateur gelé de la part intra-mesurable (G1, §5.8) : n_types de
#: paires d'eve au T77.
N_TYPES_PAIRES_EVE_T77: int = 539

#: Grandeurs GC-1 attendues (§5.4) — lues dans TOUR77_TENUE.json ; le compte
#: est aussi gelé en clair dans l'émission.
N_MESURABLES_EVE_T77: int = 486

#: Mur étalonné (§14) : run ≤ min(FACTEUR_MUR × t_cal, PLAFOND_DUR_S).
FACTEUR_MUR: float = 20_000.0
PLAFOND_DUR_S: float = 1800.0
N_PHRASES_CAL: int = 9

Ident = Tuple[int, ...]
Paire = Tuple[int, int]


class NonMesureError(RuntimeError):
    """Garde de chaîne en échec (GC-1, sha, …) ⇒ tour NON-MESURE (§5.4)."""


# ---------------------------------------------------------------------------
# Strates et colonnes (§5.1) — formules exactes
# ---------------------------------------------------------------------------

def strates_unite(idx: Sequence[int], hote_mot: Sequence[str]) -> List[np.ndarray]:
    """Groupes de positions (indices DANS ``idx``) par type de mot-hôte.

    Ordre des strates = première apparition dans ``idx`` (déterministe,
    ordre corpus). Chaque position appartient à exactement une strate.
    """
    groupes: "OrderedDict[str, List[int]]" = OrderedDict()
    for pos, h in enumerate(idx):
        groupes.setdefault(hote_mot[h], []).append(pos)
    return [np.asarray(v, dtype=np.intp) for v in groupes.values()]


def decomposition_pop(
    mat: np.ndarray, groupes: Sequence[np.ndarray]
) -> Tuple[float, float, float, float]:
    """Colonne POP (§5.1) : ``(Var_pop, V_intra_pop, V_inter_pop, ecart_T4)``.

    Par dim puis moyenne sur les dims ; strates singleton incluses
    (contribution 0 à l'intra — l'identité T4 l'exige). ``ecart_T4`` =
    max par dim de ``|Var_pop,d − (V_intra,d + V_inter,d)|`` (théorème T4,
    exact en POP — attendu ~1e-15).
    """
    if mat.ndim != 2 or mat.shape[0] < 2:
        raise ValueError(f"matrice (n >= 2, D) attendue : shape={mat.shape}")
    n_total = mat.shape[0]
    if sum(len(g) for g in groupes) != n_total:
        raise ValueError("les strates ne partitionnent pas les occurrences")
    mu = mat.mean(axis=0)
    var_d = mat.var(axis=0, ddof=0)
    intra_d = np.zeros(mat.shape[1], dtype=np.float64)
    inter_d = np.zeros(mat.shape[1], dtype=np.float64)
    for pos in groupes:
        sub = mat[pos]
        poids = len(pos) / n_total
        intra_d += poids * sub.var(axis=0, ddof=0)
        inter_d += poids * (sub.mean(axis=0) - mu) ** 2
    ecart_t4 = float(np.max(np.abs(var_d - (intra_d + inter_d))))
    return float(var_d.mean()), float(intra_d.mean()), float(inter_d.mean()), ecart_t4


def decomposition_pop_poids_uniformes(
    mat: np.ndarray, groupes: Sequence[np.ndarray]
) -> Tuple[float, float, float, float]:
    """Branche CASSÉE de la portée T4 (§5.2, contrôle positif SEULEMENT).

    Remplace les poids n_m/N par des poids uniformes 1/M : sur un témoin à
    strates déséquilibrées, l'identité T4 DOIT être violée (> 1e-6) —
    sinon l'instrument T4 est inopposable. Jamais utilisée sur le réel.
    """
    n_total = mat.shape[0]
    n_strates = len(groupes)
    mu = mat.mean(axis=0)
    var_d = mat.var(axis=0, ddof=0)
    intra_d = np.zeros(mat.shape[1], dtype=np.float64)
    inter_d = np.zeros(mat.shape[1], dtype=np.float64)
    for pos in groupes:
        sub = mat[pos]
        intra_d += (1.0 / n_strates) * sub.var(axis=0, ddof=0)
        inter_d += (1.0 / n_strates) * (sub.mean(axis=0) - mu) ** 2
    ecart = float(np.max(np.abs(var_d - (intra_d + inter_d))))
    del n_total
    return float(var_d.mean()), float(intra_d.mean()), float(inter_d.mean()), ecart


def v_intra_d1(
    mat: np.ndarray, groupes: Sequence[np.ndarray]
) -> Tuple[Optional[float], int, int]:
    """Colonne D1 (§5.1) : ``(V_intra^{d1} | None, |S₂|, N₂)``.

    S₂ = strates d'effectif ≥ 2 ; poids n_m/N₂, variance ddof=1 par strate
    (théorème T5 : E = S²_global sous la nulle, quel que soit l'effectif).
    S₂ = ∅ ⇒ unité NON intra-mesurable (None — sélecteur §5.1, jamais 0).
    """
    s2 = [g for g in groupes if len(g) >= 2]
    if not s2:
        return None, 0, 0
    n2 = sum(len(g) for g in s2)
    acc = np.zeros(mat.shape[1], dtype=np.float64)
    for pos in s2:
        acc += (len(pos) / n2) * mat[pos].var(axis=0, ddof=1)
    return float(acc.mean()), len(s2), n2


# ---------------------------------------------------------------------------
# Précondition T6-pré (§5.3) — mesure, pas parole
# ---------------------------------------------------------------------------

def t6_pre(tokens_par_contexte: Sequence[Sequence[dict]]) -> Dict[str, object]:
    """Pour chaque type de mot-hôte : la séquence g0_ids est-elle constante ?

    Hôte = token avec ``g0_len ≥ 1`` (même sélection que l'extraction §4.1
    T77). Violation = un type dont ≥ 2 occurrences portent des séquences
    g0 distinctes. Attendu 0 (P-1) : tokenisation déterministe par texte
    (R43/T73, lowering ``TOK_LOWER`` T74).
    """
    seqs: "OrderedDict[str, set]" = OrderedDict()
    n_occ_hotes = 0
    for tokens in tokens_par_contexte:
        for tok in tokens:
            g0_ids = tuple(int(p) for p in tok["g0_ids"])
            if len(g0_ids) == 0:
                continue
            n_occ_hotes += 1
            seqs.setdefault(str(tok["text"]).lower(), set()).add(g0_ids)
    violations = [
        {"hote_mot": m, "sequences_g0": sorted(list(s))}
        for m, s in seqs.items()
        if len(s) > 1
    ]
    return {
        "n_occurrences_hotes": n_occ_hotes,
        "n_types_mots": len(seqs),
        "n_violations": len(violations),
        "violations": violations,
        "tenu": len(violations) == 0,
    }


# ---------------------------------------------------------------------------
# Garde de chaîne GC-1 (§5.4) — égalité AU BIT avec TOUR77_TENUE.json
# ---------------------------------------------------------------------------

def recalcul_v_med_t77(
    paires: "OrderedDict[Paire, List[int]]", pool: np.ndarray
) -> Tuple[float, int]:
    """La grandeur T77 exacte : V_med ddof=1 sur les types mesurables.

    Même route que ``stab_ctx.verdict_tenue`` (ordre d'itération, formule,
    médiane) — c'est une garde d'IDENTITÉ de chaîne, pas une re-mesure.
    """
    mes = [
        (ident, np.asarray(idx, dtype=np.intp))
        for ident, idx in paires.items()
        if len(idx) >= 2
    ]
    v_types = [var_ctx_proj(pool[idx]) for _, idx in mes]
    return float(np.median(v_types)), len(mes)


def verifier_gc1(
    paires: "OrderedDict[Paire, List[int]]",
    pool: np.ndarray,
    v_med_attendu: float,
    n_attendu: int,
) -> Dict[str, object]:
    """GC-1 : mismatch ⇒ ``NonMesureError`` (tour NON-MESURE, §5.4).

    Branche mordante chiffrée : 1 mismatch ⇒ 0 verdict.
    """
    v_med, n_mes = recalcul_v_med_t77(paires, pool)
    egal = (v_med == v_med_attendu) and (n_mes == n_attendu)
    res = {
        "v_med_recalculee": v_med,
        "v_med_attendue_t77": v_med_attendu,
        "n_mesurables_recalcule": n_mes,
        "n_mesurables_attendu": n_attendu,
        "egalite_exacte": egal,
    }
    if not egal:
        raise NonMesureError(
            f"GC-1 : chaîne d'extraction divergente (v_med {v_med!r} vs "
            f"{v_med_attendu!r}, n {n_mes} vs {n_attendu}) — tour NON-MESURE"
        )
    return res


# ---------------------------------------------------------------------------
# Décomposition observée + nulle globale (§5.5) + contrôle stratifié (§5.6)
# ---------------------------------------------------------------------------

def preparer_mesurables(
    paires: "OrderedDict[Paire, List[int]]", hote_mot: Sequence[str]
) -> List[Tuple[object, np.ndarray, List[np.ndarray]]]:
    """(ident, idx, strates) pour chaque type mesurable (n_occ ≥ 2), en ordre."""
    return [
        (ident, np.asarray(idx, dtype=np.intp), strates_unite(idx, hote_mot))
        for ident, idx in paires.items()
        if len(idx) >= 2
    ]


def _composantes(
    mes: Sequence[Tuple[object, np.ndarray, List[np.ndarray]]], pool: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Trois vecteurs (var, intra, inter) sur les unités de ``mes`` + max T4."""
    n = len(mes)
    v_var = np.empty(n, dtype=np.float64)
    v_intra = np.empty(n, dtype=np.float64)
    v_inter = np.empty(n, dtype=np.float64)
    t4_max = 0.0
    for k, (_ident, idx, groupes) in enumerate(mes):
        var, intra, inter, ecart = decomposition_pop(pool[idx], groupes)
        v_var[k], v_intra[k], v_inter[k] = var, intra, inter
        t4_max = max(t4_max, ecart)
    return v_var, v_intra, v_inter, t4_max


def decomposition_observee(
    mes: Sequence[Tuple[object, np.ndarray, List[np.ndarray]]], pool: np.ndarray
) -> Dict[str, object]:
    """Grandeurs observées §5.5 : médianes (pointe gelée) + INFO moyennes."""
    v_var, v_intra, v_inter, t4_max = _composantes(mes, pool)
    return {
        "n_types_mesurables": len(mes),
        "v_med_intra": float(np.median(v_intra)),
        "v_med_inter": float(np.median(v_inter)),
        "INFO_v_med_totale_pop": float(np.median(v_var)),
        "t4_max_ecart_reel": t4_max,
        "t4_tenu_reel": t4_max <= TOL_IDENTITE,
        "INFO_moyennes": {
            "intra": float(np.mean(v_intra)),
            "inter": float(np.mean(v_inter)),
            "totale_pop": float(np.mean(v_var)),
        },
        "INFO_distribution_intra": _quartiles77(v_intra),
        "INFO_distribution_inter": _quartiles77(v_inter),
        "INFO_distribution_totale_pop": _quartiles77(v_var),
    }


def nulle_globale_decomposition(
    mes: Sequence[Tuple[object, np.ndarray, List[np.ndarray]]],
    pool: np.ndarray,
    graine: int = GRAINE_NULLE_GLOBALE,
    repl: int = R_NULLE,
) -> Dict[str, object]:
    """Les deux p_emp sous la nulle GLOBALE (forme T77, graine neuve §5.5).

    Un seul flux rng ; par réplicat, les DEUX médianes sont lues sur la MÊME
    permutation (recalcul conjoint, déclaré). ``p = (#{V^r ≤ V_obs}+1)/(R+1)``.
    """
    if pool.ndim != 2 or pool.shape[0] == 0:
        raise ValueError(f"pool d'hôtes vide ou non 2-D : shape={pool.shape}")
    if not mes:
        raise ValueError("aucun type mesurable : décomposition non définie")
    _, v_intra_obs_arr, v_inter_obs_arr, _ = _composantes(mes, pool)
    v_obs_intra = float(np.median(v_intra_obs_arr))
    v_obs_inter = float(np.median(v_inter_obs_arr))

    rng = np.random.default_rng(graine)
    med_intra: List[float] = []
    med_inter: List[float] = []
    for _ in range(repl):
        perm = rng.permutation(pool.shape[0])
        pp = pool[perm]
        _, vi, ve, _ = _composantes(mes, pp)
        med_intra.append(float(np.median(vi)))
        med_inter.append(float(np.median(ve)))
    arr_i = np.asarray(med_intra, dtype=np.float64)
    arr_e = np.asarray(med_inter, dtype=np.float64)
    n_le_i = int((arr_i <= v_obs_intra).sum())
    n_le_e = int((arr_e <= v_obs_inter).sum())
    return {
        "graine": graine,
        "n_replicats": repl,
        "v_obs_intra": v_obs_intra,
        "v_obs_inter": v_obs_inter,
        "n_shuffle_intra_inf_eq_obs": n_le_i,
        "n_shuffle_inter_inf_eq_obs": n_le_e,
        "p_intra": (n_le_i + 1) / (repl + 1),
        "p_inter": (n_le_e + 1) / (repl + 1),
        "distribution_med_intra_shuffle": _quartiles77(arr_i),
        "distribution_med_inter_shuffle": _quartiles77(arr_e),
    }


def case_verdict(p_intra: float, p_inter: float) -> str:
    """Table complète a priori (§5.5) — chaque case est un résultat."""
    if p_intra <= SEUIL_P_TENUE and p_inter >= SEUIL_P_ANTI_TENUE:
        return "DECOMPOSEE"
    if p_intra >= SEUIL_P_ANTI_TENUE:
        return "INVERSION"
    if p_inter >= SEUIL_P_ANTI_TENUE:
        return "INTER-SEULE"
    if p_intra <= SEUIL_P_TENUE:
        return "INTRA-SEULE"
    return "NON-LOCALISEE"


def permutation_stratifiee(
    hote_mot: Sequence[str], rng: np.random.Generator
) -> np.ndarray:
    """Permutation des hôtes À L'INTÉRIEUR de chaque type (§5.6).

    Ordre des types = première apparition corpus (gelé) ; chaque type,
    y compris singleton, consomme le rng dans cet ordre (déterministe).
    """
    par_type: "OrderedDict[str, List[int]]" = OrderedDict()
    for h, m in enumerate(hote_mot):
        par_type.setdefault(m, []).append(h)
    perm = np.arange(len(hote_mot), dtype=np.intp)
    for hs in par_type.values():
        hs_arr = np.asarray(hs, dtype=np.intp)
        perm[hs_arr] = hs_arr[rng.permutation(len(hs_arr))]
    return perm


def controle_stratifie(
    mes: Sequence[Tuple[object, np.ndarray, List[np.ndarray]]],
    hote_mot: Sequence[str],
    pool: np.ndarray,
    graine: int = GRAINE_CONTROLE_STRATIFIE,
    repl: int = R_NULLE,
) -> Dict[str, object]:
    """Contrôle d'invariance du théorème T6 (§5.6) — JAMAIS un verdict.

    Attendu si T6-pré = 0 : écarts max ≤ 1e-12 sur les trois grandeurs
    (ordre de sommation flottante seul). Un p_emp dessus vaudrait 1,0 par
    construction (support dégénéré) : c'est un théorème contrôlé.
    """
    v_var_obs, v_intra_obs, v_inter_obs, _ = _composantes(mes, pool)
    obs = (
        float(np.median(v_intra_obs)),
        float(np.median(v_inter_obs)),
        float(np.median(v_var_obs)),
    )
    rng = np.random.default_rng(graine)
    ecart_intra = 0.0
    ecart_inter = 0.0
    ecart_total = 0.0
    for _ in range(repl):
        perm = permutation_stratifiee(hote_mot, rng)
        pp = pool[perm]
        v_var_r, v_intra_r, v_inter_r, _ = _composantes(mes, pp)
        ecart_intra = max(ecart_intra, abs(float(np.median(v_intra_r)) - obs[0]))
        ecart_inter = max(ecart_inter, abs(float(np.median(v_inter_r)) - obs[1]))
        ecart_total = max(ecart_total, abs(float(np.median(v_var_r)) - obs[2]))
    return {
        "graine": graine,
        "n_replicats": repl,
        "max_ecart_med_intra": ecart_intra,
        "max_ecart_med_inter": ecart_inter,
        "max_ecart_med_totale": ecart_total,
        "invariance_tenue_1e-12": (
            ecart_intra <= TOL_INVARIANCE
            and ecart_inter <= TOL_INVARIANCE
            and ecart_total <= TOL_INVARIANCE
        ),
    }


# ---------------------------------------------------------------------------
# Spearman à rangs moyens — deux routes croisées (§5.7)
# ---------------------------------------------------------------------------

def rangs_moyens_argsort(x: Sequence[float]) -> np.ndarray:
    """Route 1 : rangs moyens par tri stable + balayage des ex æquo."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    ordre = np.argsort(arr, kind="stable")
    tries = arr[ordre]
    rangs = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and tries[j + 1] == tries[i]:
            j += 1
        rangs[ordre[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return rangs


def rangs_moyens_unique(x: Sequence[float]) -> np.ndarray:
    """Route 2 : rangs moyens par ``np.unique`` (comptes cumulés)."""
    arr = np.asarray(x, dtype=np.float64)
    _uniq, inv, comptes = np.unique(arr, return_inverse=True, return_counts=True)
    debut = np.concatenate(([0], np.cumsum(comptes)[:-1])).astype(np.float64)
    rang_moyen = debut + (comptes + 1) / 2.0
    return rang_moyen[inv]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denom == 0.0:
        raise ValueError("vecteur constant : corrélation non définie (garde)")
    return float((xc * yc).sum() / denom)


def spearman_deux_routes(x: Sequence[float], y: Sequence[float]) -> Dict[str, object]:
    """ρ_S par les deux routes de rangs moyens ; accord exigé ≤ 1e-12."""
    r1 = _pearson(rangs_moyens_argsort(x), rangs_moyens_argsort(y))
    r2 = _pearson(rangs_moyens_unique(x), rangs_moyens_unique(y))
    ecart = abs(r1 - r2)
    return {
        "rho_route_argsort": r1,
        "rho_route_unique": r2,
        "ecart_routes": ecart,
        "accord_routes_1e-12": ecart <= TOL_SPEARMAN_ROUTES,
        "rho": r1,
    }


# ---------------------------------------------------------------------------
# λ*_intra (§5.8) et Δ_ctx_intra (§5.9)
# ---------------------------------------------------------------------------

def intra_mesurables_paires(
    paires: "OrderedDict[Paire, List[int]]",
    hote_mot: Sequence[str],
    pool: np.ndarray,
) -> Tuple[List[Tuple[Paire, float]], Dict[str, int]]:
    """(ident, V_intra^{d1}) des paires intra-mesurables + comptes 2 côtés."""
    vals: List[Tuple[Paire, float]] = []
    comptes = {
        "n_types_paires": len(paires),
        "n_paires_creuses": 0,
        "n_paires_sans_strate_s2": 0,
        "n_intra_mesurables": 0,
    }
    for ident, idx in paires.items():
        if len(idx) < 2:
            comptes["n_paires_creuses"] += 1
            continue
        arr = np.asarray(idx, dtype=np.intp)
        val, _n_s2, _n2 = v_intra_d1(pool[arr], strates_unite(idx, hote_mot))
        if val is None:
            comptes["n_paires_sans_strate_s2"] += 1
            continue
        vals.append((tuple(ident), val))
        comptes["n_intra_mesurables"] += 1
    return vals, comptes


def recette_lambda_intra(
    intra_vals: Sequence[Tuple[Paire, float]],
    cache: CacheStab,
) -> Dict[str, object]:
    """λ*_intra = IQR(H_norm)/IQR(V_intra^{d1}) — recette T77 à l'identique.

    Sur LES MÊMES types intra-mesurables ; IQR(V) = 0,0 ⇒ NON-MESURE
    (garde mordante §5.8). Zéro hyperparamètre. Le régime G1 (ESTIMABLE)
    est vérifié par l'appelant (pilote), jamais contourné ici.
    """
    if not intra_vals:
        raise ValueError("aucune paire intra-mesurable : recette non définie")
    v_vals = [v for _, v in intra_vals]
    h_vals = [cache.h_norm(tuple(ident)) for ident, _ in intra_vals]
    q_v = _quartiles77(v_vals)
    q_h = _quartiles77(h_vals)
    res: Dict[str, object] = {
        "n_types_intra_mesurables": len(intra_vals),
        "iqr_h_norm": q_h["iqr"],
        "iqr_v_intra_d1": q_v["iqr"],
        "distribution_h_norm": q_h,
        "distribution_v_intra_d1": q_v,
    }
    if q_v["iqr"] == 0.0:
        res["lambda_star_intra"] = None
        res["statut"] = "NON-MESURE (IQR(V_intra^d1) = 0.0, garde §5.8)"
    else:
        res["lambda_star_intra"] = q_h["iqr"] / q_v["iqr"]
        res["statut"] = "CALIBRE"
    return res


def delta_ctx_intra_paires(
    paires: "OrderedDict[Paire, List[int]]",
    g0: "OrderedDict[int, List[int]]",
    hote_mot: Sequence[str],
    pool: np.ndarray,
) -> Dict[str, object]:
    """Δ_ctx_intra(a,b) = ½[V^{d1}(a) + V^{d1}(b)] − V^{d1}(a·b) (§5.9).

    Sélecteur : paire intra-mesurable ET ses deux parts G₀ intra-mesurables ;
    comptes publiés des deux côtés. T5 en amont : sous la nulle,
    E[Δ_ctx_intra] = 0 exactement (comparabilité par construction).
    """
    v_g0: Dict[int, float] = {}
    for pid, idx in g0.items():
        if len(idx) < 2:
            continue
        arr = np.asarray(idx, dtype=np.intp)
        val, _, _ = v_intra_d1(pool[arr], strates_unite(idx, hote_mot))
        if val is not None:
            v_g0[pid] = val

    deltas: List[Tuple[Paire, float]] = []
    comptes = {
        "n_types_paires": len(paires),
        "n_paires_creuses_exclues": 0,
        "n_paires_non_intra_mesurables_exclues": 0,
        "n_paires_part_non_intra_exclues": 0,
        "n_paires_delta_intra_mesurables": 0,
    }
    for (a, b), idx in paires.items():
        if len(idx) < 2:
            comptes["n_paires_creuses_exclues"] += 1
            continue
        arr = np.asarray(idx, dtype=np.intp)
        v_ab, _, _ = v_intra_d1(pool[arr], strates_unite(idx, hote_mot))
        if v_ab is None:
            comptes["n_paires_non_intra_mesurables_exclues"] += 1
            continue
        if a not in v_g0 or b not in v_g0:
            comptes["n_paires_part_non_intra_exclues"] += 1
            continue
        deltas.append(((a, b), 0.5 * (v_g0[a] + v_g0[b]) - v_ab))
        comptes["n_paires_delta_intra_mesurables"] += 1

    vals = [d for _, d in deltas]
    n_pos = sum(1 for d in vals if d > 0.0)
    n_neg = sum(1 for d in vals if d < 0.0)
    n_zero = sum(1 for d in vals if d == 0.0)
    return {
        "comptes": comptes,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
        "coupe": bool(n_pos >= 1 and n_neg >= 1),
        "distribution": _quartiles77(vals) if vals else None,
        "deltas": deltas,
    }


def part_inter_paires(
    mes: Sequence[Tuple[object, np.ndarray, List[np.ndarray]]], pool: np.ndarray
) -> Dict[str, object]:
    """part_inter(u) = V_inter_pop/Var_pop sur les paires à Var_pop > 0 (§5.9).

    Support de P-3 ; les paires à variance nulle (zéro-bit) sont
    exclues-comptées. Retourne aussi la liste appariée (ident, part) pour
    les INFO Spearman hors-verdict.
    """
    parts: List[Tuple[object, float]] = []
    n_var_zero = 0
    for ident, idx, groupes in mes:
        var, _intra, inter, _ = decomposition_pop(pool[idx], groupes)
        if var > 0.0:
            parts.append((ident, inter / var))
        else:
            n_var_zero += 1
    vals = [p for _, p in parts]
    return {
        "n_paires_mesurables": len(mes),
        "n_var_pop_zero_exclues": n_var_zero,
        "n_parts_mesurees": len(parts),
        "mediane_part_inter": float(np.median(vals)) if vals else None,
        "distribution": _quartiles77(vals) if vals else None,
        "parts": parts,
    }


# ---------------------------------------------------------------------------
# Théorème T5 (§5.2) — deux branches, énumération exhaustive, graine 78003
# ---------------------------------------------------------------------------

def temoin_theoreme_t5(
    graine: int = GRAINE_TEMOINS,
    n_pool: int = 8,
    tailles_strates: Tuple[int, ...] = (2, 3),
) -> Dict[str, object]:
    """T5 : E[V_intra^{d1}] = S²_pool sous la nulle — exhaustif, exact.

    Unité-témoin à strates (2, 3) tirée d'un pool seedé de 8 valeurs :
    énumération EXHAUSTIVE des C(8,2)·C(6,3) = 560 affectations sans
    remise ; la moyenne des V_intra^{d1} DOIT égaler S²_pool (ddof=1) à
    1e-9. CONTRÔLE POSITIF (branche cassée) : la même moyenne sous ddof=0
    DOIT montrer le biais Σ(n_m−1)/N₂ — visible et exact.
    """
    rng = np.random.default_rng(graine)
    pool = rng.normal(size=n_pool)
    s2_pool = float(pool.var(ddof=1))
    n2 = sum(tailles_strates)

    vals_d1: List[float] = []
    vals_d0: List[float] = []
    indices = list(range(n_pool))
    n_a, n_b = tailles_strates
    for strate_a in itertools.combinations(indices, n_a):
        reste = [i for i in indices if i not in strate_a]
        for strate_b in itertools.combinations(reste, n_b):
            sa = pool[list(strate_a)]
            sb = pool[list(strate_b)]
            vals_d1.append(
                (n_a / n2) * float(sa.var(ddof=1)) + (n_b / n2) * float(sb.var(ddof=1))
            )
            vals_d0.append(
                (n_a / n2) * float(sa.var(ddof=0)) + (n_b / n2) * float(sb.var(ddof=0))
            )
    m_d1 = float(np.mean(vals_d1))
    m_d0 = float(np.mean(vals_d0))
    biais_attendu_d0 = (sum(t - 1 for t in tailles_strates) / n2) * s2_pool
    ecart_d1 = abs(m_d1 - s2_pool)
    ecart_d0 = abs(m_d0 - biais_attendu_d0)
    return {
        "graine_pool": graine,
        "n_pool": n_pool,
        "tailles_strates": list(tailles_strates),
        "n_affectations_exhaustives": len(vals_d1),
        "variance_pool_ddof1": s2_pool,
        "moyenne_v_intra_d1": m_d1,
        "ecart_ddof1": ecart_d1,
        "moyenne_v_intra_d0": m_d0,
        "biais_attendu_ddof0": biais_attendu_d0,
        "ecart_ddof0_au_biais": ecart_d0,
        "branche_ddof1_tenue": ecart_d1 <= TOL_IDENTITE,
        "branche_ddof0_biais_visible": (ecart_d0 <= TOL_IDENTITE) and (m_d0 < m_d1),
        "tenu": (ecart_d1 <= TOL_IDENTITE)
        and (ecart_d0 <= TOL_IDENTITE)
        and (m_d0 < m_d1),
    }


# ---------------------------------------------------------------------------
# PORTÉES (§8) — deux branches par instrument, témoins construits
# ---------------------------------------------------------------------------

def _pool_temoin(vecteurs: Sequence[Tuple[float, float, float]]) -> np.ndarray:
    return np.asarray(vecteurs, dtype=np.float64)


def _mes_temoin(
    unites: Sequence[Tuple[object, Sequence[int]]], hotes: Sequence[str]
) -> List[Tuple[object, np.ndarray, List[np.ndarray]]]:
    return [
        (ident, np.asarray(idx, dtype=np.intp), strates_unite(idx, hotes))
        for ident, idx in unites
    ]


def _temoin_decomposition_a() -> Tuple[list, list, np.ndarray]:
    """Témoin (a) §8 : intra-tenue injectée — 10 unités × 4 strates × 2 hôtes,
    vecteurs IDENTIQUES par strate (intra = 0 exact), moyennes de strates
    étalées {0,1,2,3} pour toute unité (l'inter observée majore celle de
    toute permutation en médiane)."""
    hotes: List[str] = []
    vecteurs: List[Tuple[float, float, float]] = []
    unites: List[Tuple[object, List[int]]] = []
    for t in range(10):
        idx_u: List[int] = []
        for s in range(4):
            for _ in range(2):
                h = len(hotes)
                hotes.append(f"m{t}s{s}")
                vecteurs.append((float(s), float(s), float(s)))
                idx_u.append(h)
        unites.append((f"u{t}", idx_u))
    return unites, hotes, _pool_temoin(vecteurs)


def _temoin_decomposition_b() -> Tuple[list, list, np.ndarray]:
    """Témoin (b) §8 : montage inverse — moyennes de strates identiques (0),
    dispersion intra maximale (chaque strate = {+1, −1})."""
    hotes: List[str] = []
    vecteurs: List[Tuple[float, float, float]] = []
    unites: List[Tuple[object, List[int]]] = []
    for t in range(10):
        idx_u: List[int] = []
        for s in range(4):
            for signe in (1.0, -1.0):
                h = len(hotes)
                hotes.append(f"m{t}s{s}")
                vecteurs.append((signe, signe, signe))
                idx_u.append(h)
        unites.append((f"u{t}", idx_u))
    return unites, hotes, _pool_temoin(vecteurs)


def portees_t78() -> Dict[str, object]:
    """PORTÉE de chaque instrument du tour (§8) — publiée AVANT toute mesure.

    Chaque branche est ATTEINTE sur témoins construits. Gap δ_min sous
    shuffle : N-A motivé (§5.5) — toutes les grandeurs sont invariantes par
    permutation des occurrences à strates fixées, aucun verdict d'ordre
    A→B→A′ ce tour.
    """
    res: Dict[str, object] = {
        "gap_shuffle_ordre": (
            "N-A motivé (§5.5) : aucun verdict d'ordre/clôture A→B→A′ — "
            "V_intra/V_inter/médianes invariantes par permutation des "
            "occurrences à strates fixées"
        )
    }

    # --- T4 : identité tenue (témoin) ET branche cassée (poids uniformes) ---
    hotes_t4 = ["ma", "ma", "ma", "ma", "mb"]  # déséquilibré 4 vs 1
    pool_t4 = _pool_temoin(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0),
         (10.0, 10.0, 10.0)]
    )
    groupes_t4 = strates_unite([0, 1, 2, 3, 4], hotes_t4)
    var, intra, inter, ecart_ok = decomposition_pop(pool_t4, groupes_t4)
    _, _, _, ecart_casse = decomposition_pop_poids_uniformes(pool_t4, groupes_t4)
    res["theoreme_T4"] = {
        "temoin_var_intra_inter": [var, intra, inter],
        "branche_identite": {"ecart": ecart_ok, "atteinte": ecart_ok <= TOL_IDENTITE},
        "branche_cassee_poids_uniformes": {
            "ecart": ecart_casse,
            "atteinte": ecart_casse > SEUIL_VIOLATION,
        },
        "aveugle_a_l_ordre": "déclaré : permutation des occurrences à strates fixées sans effet",
    }

    # --- T5 : deux branches (ddof=1 exact, biais ddof=0 visible) ------------
    res["theoreme_T5"] = temoin_theoreme_t5()

    # --- T6-pré : branche 0-violation ET branche violation détectée ---------
    def _tok(text: str, g0_ids: List[int]) -> dict:
        v = np.zeros(33, dtype=np.float32)
        return {"text": text, "vector33d": v, "g0_ids": g0_ids, "g1_nb": [len(g0_ids)]}

    pre_ok = t6_pre([[_tok("aa", [0, 0])], [_tok("aa", [0, 0])], [_tok("bb", [1])]])
    pre_viol = t6_pre([[_tok("aa", [0, 0])], [_tok("Aa", [0, 1])]])
    res["t6_pre"] = {
        "branche_zero_violation": {
            "n_violations": pre_ok["n_violations"],
            "atteinte": pre_ok["n_violations"] == 0,
        },
        "branche_violation_detectee": {
            "n_violations": pre_viol["n_violations"],
            "violations": pre_viol["violations"],
            "atteinte": pre_viol["n_violations"] == 1,
        },
    }

    # --- GC-1 : branche égalité ET branche mismatch mordante ----------------
    paires_gc: "OrderedDict[Paire, List[int]]" = OrderedDict(
        [((0, 1), [0, 1]), ((1, 2), [2, 3])]
    )
    pool_gc = _pool_temoin(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (4.0, 4.0, 4.0)]
    )
    v_med_gc, n_gc = recalcul_v_med_t77(paires_gc, pool_gc)
    gc_egal = verifier_gc1(paires_gc, pool_gc, v_med_gc, n_gc)
    try:
        verifier_gc1(paires_gc, pool_gc, v_med_gc + 1e-9, n_gc)
        gc_mord = {"mordante": False}
    except NonMesureError:
        gc_mord = {
            "mordante": True,
            "clause": "1 mismatch => 0 verdict, tour NON-MESURE (§5.4)",
        }
    res["gc1"] = {"branche_egalite": gc_egal, "branche_mismatch": gc_mord}

    # --- Décomposition/p_emp : deux branches par composante + déterminisme --
    unites_a, hotes_a, pool_a = _temoin_decomposition_a()
    mes_a = _mes_temoin(unites_a, hotes_a)
    nulle_a = nulle_globale_decomposition(mes_a, pool_a, graine=GRAINE_TEMOINS)
    nulle_a_bis = nulle_globale_decomposition(mes_a, pool_a, graine=GRAINE_TEMOINS)
    unites_b, hotes_b, pool_b = _temoin_decomposition_b()
    mes_b = _mes_temoin(unites_b, hotes_b)
    nulle_b = nulle_globale_decomposition(mes_b, pool_b, graine=GRAINE_TEMOINS)
    res["decomposition_p_emp"] = {
        "estimateur_verdict": "mediane (pointe gelee, §5.5)",
        "branche_intra_tenue_injectee": {
            "p_intra": nulle_a["p_intra"],
            "p_inter": nulle_a["p_inter"],
            "atteinte": (
                nulle_a["p_intra"] <= SEUIL_P_TENUE
                and nulle_a["p_inter"] >= SEUIL_P_ANTI_TENUE
            ),
        },
        "branche_montage_inverse": {
            "p_intra": nulle_b["p_intra"],
            "atteinte": nulle_b["p_intra"] >= SEUIL_P_ANTI_TENUE,
        },
        "determinisme_x2_p_emp": (
            nulle_a["p_intra"] == nulle_a_bis["p_intra"]
            and nulle_a["p_inter"] == nulle_a_bis["p_inter"]
        ),
    }

    # --- Shuffle stratifié : branche INVARIANTE ET branche MOBILE -----------
    ctrl_pleines = controle_stratifie(
        mes_a, hotes_a, pool_a, graine=GRAINE_TEMOINS, repl=10
    )
    # Strates partielles : le type "mx" a 4 hôtes de vecteurs distincts, mais
    # l'unité n'en voit que 2 — le shuffle intra-type la DÉPLACE.
    hotes_part = ["mx", "mx", "mx", "mx"]
    pool_part = _pool_temoin(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (5.0, 5.0, 5.0), (9.0, 9.0, 9.0)]
    )
    mes_part = _mes_temoin([("u_partielle", [0, 1])], hotes_part)
    ctrl_part = controle_stratifie(
        mes_part, hotes_part, pool_part, graine=GRAINE_TEMOINS, repl=10
    )
    ecart_mobile = max(
        ctrl_part["max_ecart_med_intra"], ctrl_part["max_ecart_med_totale"]
    )
    res["controle_stratifie"] = {
        "branche_invariante_strates_pleines": {
            "max_ecarts": [
                ctrl_pleines["max_ecart_med_intra"],
                ctrl_pleines["max_ecart_med_inter"],
                ctrl_pleines["max_ecart_med_totale"],
            ],
            "atteinte": ctrl_pleines["invariance_tenue_1e-12"],
        },
        "branche_mobile_strates_partielles": {
            "max_ecart": ecart_mobile,
            "atteinte": ecart_mobile > SEUIL_VIOLATION,
        },
        "portee_declaree": (
            "sur le réel sous T6-pré = 0, cet instrument ne détecte RIEN par "
            "théorème (no-op exact) — contrôle, jamais verdict (§5.6)"
        ),
    }

    # --- Spearman : ±1 exacts, 0 exact, ex æquo, deux routes ----------------
    sp_plus = spearman_deux_routes([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    sp_moins = spearman_deux_routes([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
    sp_zero = spearman_deux_routes([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 1.0])
    rangs_ex = rangs_moyens_argsort([10.0, 20.0, 20.0, 30.0])
    rangs_ex2 = rangs_moyens_unique([10.0, 20.0, 20.0, 30.0])
    res["spearman"] = {
        "branche_plus_un": {"rho": sp_plus["rho"], "atteinte": sp_plus["rho"] == 1.0},
        "branche_moins_un": {"rho": sp_moins["rho"], "atteinte": sp_moins["rho"] == -1.0},
        "branche_zero_independance": {
            "rho": sp_zero["rho"],
            "atteinte": sp_zero["rho"] == 0.0,
        },
        "rangs_moyens_ex_aequo": {
            "attendu": [1.0, 2.5, 2.5, 4.0],
            "route_argsort": [float(v) for v in rangs_ex],
            "route_unique": [float(v) for v in rangs_ex2],
            "atteinte": list(rangs_ex) == [1.0, 2.5, 2.5, 4.0]
            and list(rangs_ex2) == [1.0, 2.5, 2.5, 4.0],
        },
        "accord_deux_routes": (
            sp_plus["accord_routes_1e-12"]
            and sp_moins["accord_routes_1e-12"]
            and sp_zero["accord_routes_1e-12"]
        ),
    }

    # --- Recette λ*_intra : branche calculable ET branche IQR = 0 -----------
    cache_temoin = CacheStab(TABLE_TEMOIN)
    lam_calc = recette_lambda_intra(
        [((0, 1), 0.1), ((0, 2), 0.5), ((1, 2), 0.9)], cache_temoin
    )
    lam_zero = recette_lambda_intra(
        [((0, 1), 0.5), ((0, 2), 0.5), ((1, 2), 0.5)], cache_temoin
    )
    res["recette_lambda_intra"] = {
        "branche_calculable": {
            "lambda_star_intra": lam_calc["lambda_star_intra"],
            "atteinte": lam_calc["statut"] == "CALIBRE",
        },
        "branche_iqr_zero": {
            "statut": lam_zero["statut"],
            "atteinte": lam_zero["lambda_star_intra"] is None,
        },
    }

    # --- Δ_ctx_intra : branche positive ET négative -------------------------
    # Parts G₀ dispersées INTRA-strate ; paire dans des strates homogènes ⇒ +.
    hotes_d = ["p", "p", "q", "q", "r", "r", "s", "s"]
    pool_d = _pool_temoin(
        [(0.0, 0.0, 0.0), (8.0, 8.0, 8.0),      # strate p : dispersée
         (2.0, 2.0, 2.0), (6.0, 6.0, 6.0),      # strate q : dispersée
         (5.0, 5.0, 5.0), (5.0, 5.0, 5.0),      # strate r : homogène
         (1.0, 1.0, 1.0), (9.0, 9.0, 9.0)]      # strate s : dispersée
    )
    d_pos = delta_ctx_intra_paires(
        OrderedDict([((0, 1), [4, 5])]),
        OrderedDict([(0, [0, 1]), (1, [2, 3])]),
        hotes_d, pool_d,
    )
    d_neg = delta_ctx_intra_paires(
        OrderedDict([((0, 1), [6, 7])]),
        OrderedDict([(0, [4, 5]), (1, [4, 5])]),
        hotes_d, pool_d,
    )
    res["delta_ctx_intra"] = {
        "branche_positive": {
            "delta": d_pos["deltas"][0][1],
            "atteinte": d_pos["deltas"][0][1] > 0.0,
        },
        "branche_negative": {
            "delta": d_neg["deltas"][0][1],
            "atteinte": d_neg["deltas"][0][1] < 0.0,
        },
        "amont": "T5 vérifié deux branches (theoreme_T5) — comparabilité par construction",
    }
    return res


# ---------------------------------------------------------------------------
# Étalonnage du mur (§14)
# ---------------------------------------------------------------------------

def etalonner_t78(chemin_eve: str) -> Dict[str, object]:
    """t_cal = chargement .so (garde sha comprise) + tokenisation + extraction
    + T6-pré + décomposition POP + 1 réplicat de nulle globale + 1 réplicat
    de shuffle stratifié, sur les 9 premières phrases d'eve (§14).
    Mur du run complet = min(20 000 × t_cal, 1 800 s)."""
    ctx, _ = contextes_eve(chemin_eve)
    t0 = time.perf_counter()
    tok, _prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
    tokens = [tok.tokenize_sequences(c) for c in ctx[:N_PHRASES_CAL]]
    proj = extraire_projection(tokens)
    pre = t6_pre(tokens)
    mes = preparer_mesurables(proj["paire"], proj["hote_mot"])
    obs = decomposition_observee(mes, proj["pool"]) if mes else None
    if mes:
        nulle_globale_decomposition(
            mes, proj["pool"], graine=GRAINE_NULLE_GLOBALE, repl=1
        )
        controle_stratifie(
            mes, proj["hote_mot"], proj["pool"],
            graine=GRAINE_CONTROLE_STRATIFIE, repl=1,
        )
    t_cal = time.perf_counter() - t0
    return {
        "n_phrases": N_PHRASES_CAL,
        "n_tokens": proj["comptes"]["n_tokens"],
        "n_paires_mesurables_tranche": len(mes),
        "t6_pre_tranche": {"n_types": pre["n_types_mots"], "n_violations": pre["n_violations"]},
        "v_med_intra_tranche_INFO": (obs or {}).get("v_med_intra"),
        "t_cal_s": t_cal,
        "facteur_mur": FACTEUR_MUR,
        "mur_s": min(FACTEUR_MUR * t_cal, PLAFOND_DUR_S),
        "plafond_dur_s": PLAFOND_DUR_S,
    }


# ---------------------------------------------------------------------------
# Pilote (artefacts TOUR78_* — racine écosystème, sans horodatage, sidecars)
# ---------------------------------------------------------------------------

def _md5(chemin: str) -> str:
    with open(chemin, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _config_echo(md5_eve: Optional[str], md5_claude: Optional[str],
                 so_sha256: Optional[str]) -> Dict[str, object]:
    """Écho de configuration complet (invariant §9 : contexte déclaré).

    SANS horodatage : les artefacts de verdict sont reproductibles AU BIT."""
    return {
        "tour": 78,
        "graine_nulle_globale": GRAINE_NULLE_GLOBALE,
        "graine_controle_stratifie": GRAINE_CONTROLE_STRATIFIE,
        "graine_temoins": GRAINE_TEMOINS,
        "r_nulle": R_NULLE,
        "seuil_p_tenue": SEUIL_P_TENUE,
        "seuil_p_anti_tenue": SEUIL_P_ANTI_TENUE,
        "tol_identite_t4": TOL_IDENTITE,
        "tol_invariance_t6": TOL_INVARIANCE,
        "tol_spearman_routes": TOL_SPEARMAN_ROUTES,
        "seuil_rho_spearman": SEUIL_RHO_SPEARMAN,
        "seuil_part_mesurable": SEUIL_PART_MESURABLE,
        "seuil_n_mesurables": SEUIL_N_MESURABLES,
        "n_types_paires_eve_t77": N_TYPES_PAIRES_EVE_T77,
        "n_mesurables_eve_t77": N_MESURABLES_EVE_T77,
        "graine_shuffle_t77_historique": GRAINE_SHUFFLE_T77,
        "ctx_dims": list(CTX_DIMS),
        "colonne_verdict": "POP (ddof=0, loi de variance totale exacte — T4)",
        "colonne_d1": "V_intra^{d1} (ddof=1 par strate S2 — T5)",
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


def _ecrire_artefact(chemin: str, obj: Dict[str, object]) -> None:
    """Artefact + sidecar sha256 (``<nom>.sha256`` : « <sha>  <base> »)."""
    _ecrire_json(chemin, obj)
    with open(chemin, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(chemin + ".sha256", "w", encoding="ascii", newline="\n") as fh:
        fh.write(f"{sha}  {os.path.basename(chemin)}\n")


def run_complet(
    chemin_eve: str,
    chemin_claude: str,
    chemin_tenue_t77: str,
    mur_s: Optional[float],
    dossier_sortie: str,
) -> Dict[str, str]:
    """Le run gelé (portes P4-P9) : T6-pré/GC-1 → décomposition + verdict →
    contrôle stratifié → audit Spearman → λ*_intra → Δ_ctx_intra.

    Verdicts sur EVE SEUL (GC-1 en amont : mismatch ⇒ NonMesureError) ;
    claude_aba publié aux mêmes grandeurs de décomposition, RANG INFO
    (clause T42). Mur §14 : dépassement ⇒ I5, artefacts restants marqués
    INOPPOSABLES.
    """
    t0 = time.perf_counter()

    def _mur_atteint() -> bool:
        return mur_s is not None and (time.perf_counter() - t0) > mur_s

    tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
    table = table_miroir()
    cache = CacheStab(table)
    md5_eve, md5_claude = _md5(chemin_eve), _md5(chemin_claude)
    config = _config_echo(md5_eve, md5_claude, prov["native_lib_sha256"])

    with open(chemin_tenue_t77, "r", encoding="utf-8") as fh:
        tenue_t77 = json.load(fh)
    v_med_t77 = float(tenue_t77["eve"]["v_med_observee"])
    n_mes_t77 = int(tenue_t77["eve"]["n_types_mesurables"])

    ctx_e, cpt_e = contextes_eve(chemin_eve)
    ctx_c, cpt_c = contextes_claude_aba(chemin_claude)
    tokens_e = [tok.tokenize_sequences(c) for c in ctx_e]
    tokens_c = [tok.tokenize_sequences(c) for c in ctx_c]
    proj_e = extraire_projection(tokens_e)
    proj_c = extraire_projection(tokens_c)
    chemins: Dict[str, str] = {}

    def _sortie(nom: str, obj: Dict[str, object]) -> None:
        p = os.path.join(dossier_sortie, nom)
        _ecrire_artefact(p, obj)
        chemins[nom] = p

    # --- P4 : préconditions — T6-pré (mesure) + GC-1 (garde de chaîne) ------
    pre_e = t6_pre(tokens_e)
    pre_c = t6_pre(tokens_c)
    gc1 = verifier_gc1(proj_e["paire"], proj_e["pool"], v_med_t77, n_mes_t77)
    _sortie("TOUR78_PRECONDITION.json", {
        "config": config,
        "contextes": {"eve": cpt_e, "claude_aba": cpt_c},
        "t6_pre_eve": pre_e,
        "t6_pre_claude_aba_INFO": pre_c,
        "gc1": gc1,
        "note_gc1": (
            "garde d'identité de chaîne (§5.4) — égalité EXACTE (==) avec "
            "TOUR77_TENUE.json ; ce n'est pas une re-mesure"
        ),
    })

    # --- P5 : décomposition + verdict (§5.5) + P6 : contrôle stratifié ------
    mes_e = preparer_mesurables(proj_e["paire"], proj_e["hote_mot"])
    mes_c = preparer_mesurables(proj_c["paire"], proj_c["hote_mot"])
    decomp_obj: Dict[str, object] = {"config": config}
    if _mur_atteint():
        decomp_obj["issue"] = "I5_MUR (§14) : décomposition INOPPOSABLE, état tronqué INFO"
    else:
        obs_e = decomposition_observee(mes_e, proj_e["pool"])
        nulle_e = nulle_globale_decomposition(mes_e, proj_e["pool"])
        ctrl_e = controle_stratifie(mes_e, proj_e["hote_mot"], proj_e["pool"])
        statut_ctrl = (
            "CONTRÔLE D'INVARIANCE T6 (T6-pré = 0)" if pre_e["tenu"]
            else "MESURE-INFO (T6-pré > 0, soupape T34 — jamais un verdict rétroactif)"
        )
        decomp_obj["eve"] = {
            "observe": obs_e,
            "nulle_globale": nulle_e,
            "verdict_decomposition": case_verdict(
                nulle_e["p_intra"], nulle_e["p_inter"]
            ),
            "controle_stratifie": dict(ctrl_e, statut=statut_ctrl),
        }
        decomp_obj["verdict_prononce"] = decomp_obj["eve"]["verdict_decomposition"]
        obs_c = decomposition_observee(mes_c, proj_c["pool"])
        nulle_c = nulle_globale_decomposition(mes_c, proj_c["pool"])
        ctrl_c = controle_stratifie(mes_c, proj_c["hote_mot"], proj_c["pool"])
        decomp_obj["claude_aba_INFO"] = {
            "observe": obs_c,
            "nulle_globale": nulle_c,
            "case_INFO": case_verdict(nulle_c["p_intra"], nulle_c["p_inter"]),
            "controle_stratifie": ctrl_c,
            "rang": "INFO (clause T42 : jamais une confirmation partielle)",
        }
        decomp_obj["note_gap"] = (
            "gap δ_min N-A motivé (§5.5) : aucune grandeur de ce tour ne "
            "teste un ordre A→B→A′"
        )
    _sortie("TOUR78_DECOMP.json", decomp_obj)

    # --- P7 : audit du confondant (§5.7) ------------------------------------
    audit_obj: Dict[str, object] = {"config": config}
    if _mur_atteint():
        audit_obj["issue"] = "I5_MUR (§14) : audit INOPPOSABLE"
        rho_abs: Optional[float] = None
    else:
        d_e = delta_ctx_paires(proj_e["paire"], proj_e["g0"], proj_e["pool"])
        deltas_t77 = d_e["deltas"]
        idents = [ident for ident, _ in deltas_t77]
        x_delta = [d for _, d in deltas_t77]
        y_ntypes = [
            float(len({proj_e["hote_mot"][i] for i in proj_e["paire"][ident]}))
            for ident in idents
        ]
        y_nocc = [float(len(proj_e["paire"][ident])) for ident in idents]
        sp = spearman_deux_routes(x_delta, y_ntypes)
        rho_abs = abs(sp["rho"])
        franchie = rho_abs >= SEUIL_RHO_SPEARMAN
        # INFO hors-verdict
        sp_nocc = spearman_deux_routes(x_delta, y_nocc)
        pi = part_inter_paires(mes_e, proj_e["pool"])
        parts_map = {ident: p for ident, p in pi["parts"]}
        commun = [ident for ident in idents if ident in parts_map]
        sp_part = spearman_deux_routes(
            [parts_map[i] for i in commun],
            [float(len({proj_e["hote_mot"][j] for j in proj_e["paire"][i]})) for i in commun],
        )
        audit_obj.update({
            "n_paires_delta_mesurables": len(deltas_t77),
            "spearman_delta_ctx_vs_n_types_mots_hotes": sp,
            "seuil_bande": SEUIL_RHO_SPEARMAN,
            "bande_franchie": franchie,
            "consequence": (
                "CONFONDANT PROUVÉ au seuil ⇒ proposition de gravure : "
                "Δ_ctx_intra remplace Δ_ctx dans Coherence_full pour toute "
                "admissibilité/fusion future ; liste des 123 et λ* T77 → "
                "rang HISTORIQUE (promotion à Ra)" if franchie else
                "PAS de remplacement par la bande — les deux critères "
                "publiés côte à côte, choix du critère de fusion à Ra sur "
                "pièces (§5.7)"
            ),
            "INFO_spearman_delta_ctx_vs_n_occ": sp_nocc,
            "INFO_spearman_part_inter_vs_n_types": dict(
                sp_part, n_paires_communes=len(commun)
            ),
        })
    _sortie("TOUR78_AUDIT_SPEARMAN.json", audit_obj)

    # --- P8 : λ*_intra sous G1 (§5.8) — couple intra×eve, aucun repli -------
    lam_obj: Dict[str, object] = {"config": config}
    lam_star_intra: Optional[float] = None
    intra_vals_e, comptes_intra_e = intra_mesurables_paires(
        proj_e["paire"], proj_e["hote_mot"], proj_e["pool"]
    )
    n_intra = comptes_intra_e["n_intra_mesurables"]
    part_intra = n_intra / N_TYPES_PAIRES_EVE_T77
    estimable = (part_intra >= SEUIL_PART_MESURABLE) and (n_intra >= SEUIL_N_MESURABLES)
    lam_obj["carte_intra_eve"] = dict(
        comptes_intra_e,
        part_intra_sur_539=part_intra,
        regime_g1="ESTIMABLE" if estimable else "CREUX",
    )
    if _mur_atteint():
        lam_obj["issue"] = "I5_MUR (§14) : λ*_intra INOPPOSABLE"
    elif not estimable:
        lam_obj["lambda_intra"] = {
            "lambda_star_intra": None,
            "statut": "NON-MESURE (G1 : couple intra×eve CREUX, aucun repli §5.8)",
        }
    else:
        lam = recette_lambda_intra(intra_vals_e, cache)
        lam_obj["lambda_intra"] = lam
        lam_star_intra = lam["lambda_star_intra"]
        lam_obj["statut_candidat"] = (
            "CANDIDAT — promotion (et sort du λ* T77 selon §5.7) à Ra"
        )
    _sortie("TOUR78_LAMBDA_INTRA.json", lam_obj)

    # --- P9 : Δ_ctx_intra + admissibilité (§5.9) — AUCUNE fusion ------------
    adm_obj: Dict[str, object] = {"config": config}
    if _mur_atteint():
        adm_obj["issue"] = "I5_MUR (§14) : Δ_ctx_intra/admissibilité INOPPOSABLES"
    else:
        d_intra = delta_ctx_intra_paires(
            proj_e["paire"], proj_e["g0"], proj_e["hote_mot"], proj_e["pool"]
        )
        deltas_intra = d_intra.pop("deltas")
        adm_obj["delta_ctx_intra_eve"] = d_intra
        pi = part_inter_paires(mes_e, proj_e["pool"])
        pi.pop("parts")
        adm_obj["part_inter_eve_P3"] = pi
        if lam_star_intra is None:
            adm_obj["coherence_full_intra_eve"] = {
                "statut": "NON-MESURE (chaîne §5.9 : λ*_intra NON-MESURE)",
            }
        else:
            adm = admissibilite_coherence_full(
                deltas_intra, cache, lam_star_intra, table
            )
            adm["note_champs"] = (
                "fonction gelée T77 réutilisée zéro octet : les champs "
                "delta_ctx/coherence_full portent ici Δ_ctx_intra/"
                "Coherence_full_intra (λ = λ*_intra)"
            )
            adm_obj["coherence_full_intra_eve"] = adm
        adm_obj["note"] = "AUCUNE fusion exécutée ce tour (§5.9) — admissibilité seulement"
    _sortie("TOUR78_ADMISSIBILITE_INTRA.json", adm_obj)

    return chemins


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stade", required=True, choices=["portees", "etalonnage", "run"])
    ap.add_argument("--eve", default=None, help="chemin de corpus_eve_clean.txt")
    ap.add_argument("--claude-aba", default=None, help="chemin de corpus_claude_aba.txt")
    ap.add_argument("--tenue-t77", default=None, help="chemin de TOUR77_TENUE.json (GC-1)")
    ap.add_argument("--sortie", required=True,
                    help="fichier JSON (portees/etalonnage) ou dossier (run)")
    ap.add_argument("--mur-s", type=float, default=None,
                    help="mur du run (depuis TOUR78_ETALONNAGE.json)")
    args = ap.parse_args(argv)

    if args.stade == "portees":
        # Garde sha AVANT toute mesure ; branche mismatch prouvée mordante.
        _tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
        obj: Dict[str, object] = {
            "config": _config_echo(None, None, prov["native_lib_sha256"]),
            "portees": portees_t78(),
        }
        try:
            charger_tokenizer_garde(sha256_attendu="0" * 64)
            obj["garde_sha_so"] = {"mordante": False}
        except RuntimeError:
            obj["garde_sha_so"] = {
                "mordante": True,
                "clause": "1 mismatch => 0 mesure prise, tour NON-MESURE (§8)",
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
        _ecrire_artefact(args.sortie, obj)
    elif args.stade == "etalonnage":
        _ecrire_artefact(args.sortie, {"tour": 78, "etalonnage": etalonner_t78(args.eve)})
    else:
        chemins = run_complet(
            args.eve, args.claude_aba, args.tenue_t77, args.mur_s, args.sortie
        )
        for nom, chemin in chemins.items():
            print(f"{nom}: {chemin}")


if __name__ == "__main__":  # pragma: no cover - pilote
    main()
