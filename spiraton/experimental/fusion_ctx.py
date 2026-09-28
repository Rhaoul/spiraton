"""Fusion contextuelle sous Coherence_full_intra — instrument du Tour 82 (LA FUSION).

Ce module exécute le PREMIER run de fusion du projet sous le critère contextuel
acté par Ra : ``Coherence_full_intra = Coherence_H + λ*_intra · Δ_ctx_intra``
avec λ*_intra = 1,2744331711650387 (3e naissance, chaîne T80, gelé CONSTANT).
Toutes les définitions ci-dessous sont les gels de ``TOUR82_EMISSION.md``
(md5 ``dc077693854d2725dd9907651256729e``) — rien n'est réglable en douce.

Réutilisation ZÉRO OCTET (imports seuls, aucune formule recopiée) :

- ``stab_grains`` : ``charger_tokenizer_garde``, ``contextes_eve``,
  ``table_miroir``, ``CTX_DIMS`` ;
- ``stab_ctx`` : ``extraire_projection`` (chaîne d'extraction T77/T80),
  ``admissibilite_coherence_full``, ``_quartiles77`` ;
- ``stab_decomp`` : ``strates_unite``, ``v_intra_d1`` (colonne D1 ddof=1),
  ``intra_mesurables_paires``, ``recette_lambda_intra``,
  ``delta_ctx_intra_paires``, ``NonMesureError`` ;
- ``bpe_logos`` : ``CacheStab`` (Coherence_H gelée T75), ``EtatCorpus``,
  ``construire_etat``, ``appliquer_fusion`` (balayage gauche→droite non
  chevauchant + garde de décroissance), ``vocabulaire_final``,
  ``torsion_corpus``, ``spearman_rho``, ``MAX_FUSIONS``, ``TABLE_TEMOIN``.

Définitions opérationnelles gelées (émission T82 §3.2) :

- **État** : par TYPE de mot phonémique (identité = tuple g0, T75), la
  segmentation courante (fonction de l'identité — ``EtatCorpus``), PLUS le
  pool des c(occ) (dims 28-30, float32→float64, route T77) de chaque
  occurrence-hôte et l'identité d'hôte ``text.lower()`` (gel T73).
  Occurrence d'une unité = (occurrence de mot-hôte, position) ; les listes
  d'indices d'hôtes sont construites HÔTE-MAJEUR (ordre corpus), la même
  route que ``stab_ctx.extraire_projection`` — à l'état 0 les grandeurs
  REDONNENT exactement celles du T80 (c'est ce que GC-82 vérifie).
- **Critère par paire candidate (u,v) adjacente intra-mot** :
  ``Coherence_full_intra(u,v) = Coherence_H(u,v) + λ_gelé · Δ_ctx_intra(u,v)``
  avec Coherence_H la fonction gelée T75 (``CacheStab.coherence``) et
  ``Δ_ctx_intra(u,v) = ½[V^{d1}(u) + V^{d1}(v)] − V^{d1}(u·v)``,
  V^{d1} = ``stab_decomp.v_intra_d1`` (strates = types de mots-hôtes,
  S₂ = strates à ≥ 2 occurrences), V^{d1}(u·v) sur les occurrences de
  l'ADJACENCE (u,v).
- **λ_gelé = 1,2744331711650387, CONSTANT sur tout le run** — aucune
  recalibration (une recalibration en cours de run ⇒ run NON-MESURE, §6).
- **Fusible** ssi les TROIS V^{d1} sont mesurables ET coherence_full > 0
  strict (θ = 0, float64, aucune tolérance). Paires à V^{d1} non mesurable :
  EXCLUES-COMPTÉES par itération — JAMAIS de repli sur Coherence_H seul
  (la voie bottom-up pure est CLOSE T75 ; le repli serait sa réouverture).
- **Sélection** : Coherence_full_intra maximale ; ex æquo (== exact)
  tranchés par ordre lexicographique croissant sur (a, b). Jamais de
  fréquence, jamais de graine — run intégralement déterministe (zéro tirage).
- **Fusion** : partout, balayage gauche→droite non chevauchant
  (``bpe_logos.appliquer_fusion``) ; invariant de décroissance stricte du
  total d'unités (garde mordante héritée).
- **k** : k(G₀)=0, k(fusion(a,b)) = max(k(a),k(b))+1 ; conflits comptés
  (attendu 0, G81-3).
- **Arrêt** (liste close) : ARRET_CRITERE / I5_MAX_FUSIONS (20 000) /
  I5_MUR_TEMPOREL (mur étalonné §3.4).
- **n_partagées** (grandeur de verdict) : nb de types d'unités du
  vocabulaire FINAL avec k ≥ 1 ET ``n_types_mots ≥ 2`` (champ de
  ``vocabulaire_final``, recette T75 — sélecteur fonction du seul corpus).
- **GC-82 [PORTEUR]** (§3.1) : les 9 grandeurs de l'état 0 (grain paire,
  eve) re-dérivées par la route gelée stab_decomp et comparées AU BIT aux
  artefacts ``TOUR80_LAMBDA_INTRA.json`` / ``TOUR80_ADMISSIBILITE_INTRA.json``
  (clause P2 : l'artefact fait foi, jamais la prose). 1 mismatch ⇒ I0,
  ZÉRO mesure aval (``NonMesureError``). Branche négative par MUTATION.
- **Vierge (§4)** : RESERVE_v2 = cycles 1025-2048 de
  ``aba_v2_lab_pack/dataset_aba_v2.txt`` (md5 du pack vérifiés AVANT toute
  lecture contre le gel de réception post-T65) ; conventions §3.5
  (parseur ``spiraton.data.aba.try_parse_aba_line``, 3 contextes par cycle,
  tokenisés SÉPARÉMENT) ; replay ORDONNÉ du journal eve ;
  J_sel vs J_base ; verdicts gelés (GÉNÉRALISE / NE GÉNÉRALISE PAS /
  INDISTINCT / NON-MESURE à conditions chiffrées). TEST_v2 (cycles
  2049-4096) JAMAIS lu : la lecture s'arrête au cycle 2048.

Contexte d'exécution (clause P1 T81) : chaque artefact écho sa config
complète SANS horodatage — les artefacts de verdict se reproduisent au bit.
Classe de reproductibilité (clause P4) : compteurs entiers et signes
bit-stables ; flottants d'accumulation publiés avec route déclarée (float64,
ordre d'opérations des modules gelés). Les verdicts vivent dans les
artefacts ``TOUR82_*.json`` (racine écosystème), pas ici.
"""
from __future__ import annotations

import argparse
import hashlib
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
    charger_tokenizer_garde,
    contextes_eve,
    table_miroir,
)
from spiraton.experimental.stab_ctx import (
    _quartiles77,
    admissibilite_coherence_full,
    extraire_projection,
)
from spiraton.experimental.stab_decomp import (
    NonMesureError,
    delta_ctx_intra_paires,
    intra_mesurables_paires,
    recette_lambda_intra,
    strates_unite,
    v_intra_d1,
)
from spiraton.experimental.bpe_logos import (
    MAX_FUSIONS,
    MIN_FUSIONS_RHO,
    SEUIL_RHO_SPEARMAN,
    TABLE_TEMOIN,
    CacheStab,
    EtatCorpus,
    appliquer_fusion,
    construire_etat,
    spearman_rho,
    torsion_corpus,
    vocabulaire_final,
)

# ---------------------------------------------------------------------------
# Constantes gelées du tour (émission T82 — jamais cachées, jamais réglables)
# ---------------------------------------------------------------------------

#: sha256 COMPLET du .so de référence aval (gel T80, blob/fichier conservé —
#: clause P6 : JAMAIS de re-lien, seul le fichier conservé lie).
SHA256_SO_REFERENCE_T82: str = (
    "9b2830ef61def5c38a77616b400f384eda4c8ee7e4630b2fc30a6d495b050f94"
)

#: λ*_intra gelé (3e naissance T80), CONSTANT sur tout le run (§3.2).
LAMBDA_GELE: float = 1.2744331711650387

#: Identité de λ par RECETTE (G81-5) — échoée dans chaque artefact.
LAMBDA_RECETTE: Dict[str, str] = {
    "valeur": "1.2744331711650387",
    "recette": "IQR(H_norm) / IQR(V_intra^{d1}) sur les 450 types de paires "
               "intra-mesurables d'eve (stab_decomp.recette_lambda_intra, gelée)",
    "chaine": "T80 (3e naissance ; T78=1re, T79=2e, binaires HISTORIQUES)",
    "blob_so": SHA256_SO_REFERENCE_T82,
    "route_quantiles": "np.quantile interpolation linéaire (module gelé) ; "
                       "concordance fsum/numpy attestée T80",
}

#: θ de fusion : zéro STRICT (float64, aucune tolérance — §3.2).
THETA_FUSION_T82: float = 0.0

#: Seuil de la branche A (« un vocabulaire émerge ») : n_partagées ≥ 30 (§2).
SEUIL_PARTAGE: int = 30

#: Grandeurs d'état 0 gelées par l'émission (§3.1) — contrôle interne du run ;
#: la vérité opposable reste les artefacts T80 (GC-82 les lit).
ETAT0_N_TYPES_PAIRES: int = 532
ETAT0_N_MESURABLES: int = 450
ETAT0_N_POS_FULL: int = 107

#: Murs étalonnés (§3.4) — plafond eve DOUBLÉ vs T75 (critère plein, motivé).
FACTEUR_MUR_EVE: float = 2000.0
PLAFOND_EVE_S: float = 3600.0
N_PHRASES_CAL: int = 9
N_ITERATIONS_CAL: int = 10
FACTEUR_MUR_V2: float = 300.0
PLAFOND_V2_S: float = 1800.0
N_CYCLES_CAL_V2: int = 10

#: Tranche vierge (§4) : cycles 1025-2048 (ordre du fichier), TEST_v2 interdit.
RESERVE_V2_DEBUT: int = 1025
RESERVE_V2_FIN: int = 2048

#: Opposabilité du juge vierge (§4) : effectif et spécificité de la baseline.
MIN_MESURABLES_VIERGE: int = 30

#: Gel de réception du pack aba_v2 (addendum post-T65, JOURNAL_SPIRALE.md
#: l.9019-9027) — vérifié AVANT toute lecture de la tranche.
MD5_PACK_ABA_V2: Dict[str, str] = {
    "dataset_aba_v2.txt": "d7ab47c8a974f06b017ac2d6432f30e0",
    "corpus_eve_v2.txt": "af0bec55e9d772aecd4a4aa6ad403e73",
    "dataset_aba_v2_index.jsonl": "786d100bff652db51ae53936f6f30093",
    "dataset_aba_v2_manifest.json": "a3920b0702e5eddd5ebfcfb2e40af7e7",
    "build_dataset_aba_v2.py": "58afc87976ec67b0c2a7a5246d255b81",
    "validate_dataset_aba_v2.py": "33cf5a56b8cf76be181fc170118c5130",
    "MANIFEST_ABA_V2.md": "6345d657c611292f802176d9cfc583a8",
}

Ident = Tuple[int, ...]
PaireU = Tuple[Ident, Ident]


# ---------------------------------------------------------------------------
# État de fusion contextuel (§3.2) — EtatCorpus T75 + hôtes projetés T77
# ---------------------------------------------------------------------------

class EtatFusionCtx:
    """État T82 : ``EtatCorpus`` (segmentation par type, gelé T75) + le pool
    des c(occ) et l'identité d'hôte de chaque occurrence (route T77).

    ``pool[h]`` = dims (28, 29, 30) du 33D de l'occurrence-hôte h (float32
    natif → float64, MÊME route que ``stab_ctx.extraire_projection``) ;
    ``hote_mot[h]`` = ``text.lower()`` (gel T73) ; ``type_de_hote[h]`` =
    tuple g0 du token (identité de type T75). Tokens muets (g0_len = 0)
    exclus des hôtes (comptés par ``EtatCorpus``).
    """

    def __init__(self, tokens_par_contexte: Sequence[Sequence[dict]]) -> None:
        self.etat: EtatCorpus = construire_etat(tokens_par_contexte)
        pool: List[np.ndarray] = []
        hotes: List[str] = []
        types_h: List[Ident] = []
        for tokens in tokens_par_contexte:
            for tok in tokens:
                g0 = tuple(int(p) for p in tok["g0_ids"])
                if len(g0) == 0:
                    continue
                v33 = np.asarray(tok["vector33d"], dtype=np.float32)
                pool.append(v33[list(CTX_DIMS)].astype(np.float64))
                hotes.append(str(tok["text"]).lower())
                types_h.append(g0)
        self.pool: np.ndarray = np.asarray(pool, dtype=np.float64).reshape(
            len(pool), len(CTX_DIMS)
        )
        self.hote_mot: List[str] = hotes
        self.type_de_hote: List[Ident] = types_h


def occurrences_courantes(
    ef: EtatFusionCtx,
) -> Tuple["OrderedDict[Ident, List[int]]", "OrderedDict[PaireU, List[int]]"]:
    """Indices d'hôtes par unité et par adjacence, dans l'état COURANT.

    Ordre HÔTE-MAJEUR (h croissant, positions gauche→droite dans l'hôte) —
    la même route d'accumulation que ``stab_ctx.extraire_projection`` : à
    l'état 0 (unités = phonèmes) les listes sont IDENTIQUES à celles de la
    chaîne T80, donc les V^{d1} sont identiques au bit (GC-82 le vérifie).
    Une unité présente k fois dans un hôte y contribue k occurrences.
    """
    unites_idx: "OrderedDict[Ident, List[int]]" = OrderedDict()
    adj_idx: "OrderedDict[PaireU, List[int]]" = OrderedDict()
    for h, t in enumerate(ef.type_de_hote):
        seg = ef.etat.types[t]["unites"]
        for u in seg:
            unites_idx.setdefault(u, []).append(h)
        for i in range(len(seg) - 1):
            adj_idx.setdefault((seg[i], seg[i + 1]), []).append(h)
    return unites_idx, adj_idx


def _v_d1_de(idx: Sequence[int], hote_mot: Sequence[str], pool: np.ndarray) -> Optional[float]:
    """V^{d1} d'une liste d'occurrences (None si non mesurable : n < 2 ou S₂ = ∅).

    Route gelée : ``v_intra_d1(pool[idx], strates_unite(idx, hote_mot))`` —
    strates = types de mots-hôtes, ddof=1, poids n_m/N₂ (stab_decomp §5.1).
    """
    if len(idx) < 2:
        return None
    val, _n_s2, _n2 = v_intra_d1(
        pool[np.asarray(idx, dtype=np.intp)], strates_unite(idx, hote_mot)
    )
    return val


class _CacheVd1:
    """Cache exact de V^{d1} par (unité, liste d'occurrences) — la valeur est
    une fonction pure de la liste d'indices (pool et hote_mot fixes) : le
    cache ne change AUCUN chiffre, il évite des recalculs identiques."""

    def __init__(self, ef: EtatFusionCtx) -> None:
        self._ef = ef
        self._vals: Dict[Tuple[Ident, bytes], Optional[float]] = {}

    def v(self, u: Ident, idx: Sequence[int]) -> Optional[float]:
        cle = (u, np.asarray(idx, dtype=np.intp).tobytes())
        if cle not in self._vals:
            self._vals[cle] = _v_d1_de(idx, self._ef.hote_mot, self._ef.pool)
        return self._vals[cle]


# ---------------------------------------------------------------------------
# Évaluation des candidats (§3.2) — critère PLEIN, exclusions comptées
# ---------------------------------------------------------------------------

def evaluer_candidats(
    ef: EtatFusionCtx,
    cache: CacheStab,
    lam: float,
    cache_v: Optional[_CacheVd1] = None,
) -> Dict[str, object]:
    """Toutes les paires adjacentes de l'état courant, sous le critère plein.

    Retour : ``candidats`` = liste ordonnée (première apparition hôte-majeur)
    de dicts {a, b, coherence_h, delta_ctx, coherence_full, frequence} pour
    les paires aux TROIS V^{d1} mesurables ; ``exclues`` = identités des
    paires exclues avec le motif (u/v/adjacence non mesurable) — comptées,
    JAMAIS repliées sur Coherence_H seul (§3.2).
    """
    if cache_v is None:
        cache_v = _CacheVd1(ef)
    unites_idx, adj_idx = occurrences_courantes(ef)
    v_unit: Dict[Ident, Optional[float]] = {
        u: cache_v.v(u, idx) for u, idx in unites_idx.items()
    }
    candidats: List[dict] = []
    exclues: List[dict] = []
    for (u, v), idx in adj_idx.items():
        v_u = v_unit.get(u)
        v_v = v_unit.get(v)
        v_ab = cache_v.v(u + v, idx)  # clé = unité fusionnée candidate (unique)
        if v_u is None or v_v is None or v_ab is None:
            exclues.append(
                {
                    "a": list(u),
                    "b": list(v),
                    "motif": {
                        "u_non_mesurable": v_u is None,
                        "v_non_mesurable": v_v is None,
                        "adjacence_non_mesurable": v_ab is None,
                    },
                }
            )
            continue
        delta = 0.5 * (v_u + v_v) - v_ab
        coh_h = cache.coherence(u, v)
        candidats.append(
            {
                "a": u,
                "b": v,
                "coherence_h": coh_h,
                "delta_ctx": delta,
                "coherence_full": coh_h + lam * delta,
                "frequence": len(idx),
            }
        )
    return {
        "candidats": candidats,
        "exclues": exclues,
        "n_adjacences_types": len(adj_idx),
        "n_mesurables": len(candidats),
        "n_exclues_non_mesurables": len(exclues),
    }


# ---------------------------------------------------------------------------
# Boucle de fusion T82 (§3.2) — déterministe, zéro tirage
# ---------------------------------------------------------------------------

def executer_fusions_ctx(
    ef: EtatFusionCtx,
    cache: CacheStab,
    lam: float = LAMBDA_GELE,
    theta: float = THETA_FUSION_T82,
    max_fusions: int = MAX_FUSIONS,
    mur_s: Optional[float] = None,
    t0: Optional[float] = None,
    controle_etat0: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, object]:
    """Fusion itérative sous Coherence_full_intra jusqu'à arrêt (liste close).

    Sélection (§3.2) : coherence_full maximale ; ex æquo (== exact) tranchés
    par ordre lexicographique CROISSANT sur (a, b). λ CONSTANT (« lam ») sur
    tout le run. ``controle_etat0`` : triplet gelé (n_adjacences_types,
    n_mesurables, n_pos) vérifié à l'itération 1 (réduction exacte à l'état
    0, §3.2) — mismatch ⇒ ``NonMesureError`` (GC-82 par le run lui-même).

    Retour : journal ordonné (contributions H et λ·Δ SÉPARÉES, fréquence à
    la sélection, k, n_remplacements, exclusions par itération), motif
    d'arrêt, profondeurs k, conflits k, état final des candidats (mesurables
    évalués + exclues en extension — la matière de la raison structurelle).
    """
    cache_v = _CacheVd1(ef)
    journal: List[dict] = []
    profondeurs: Dict[Ident, int] = {}
    n_conflits_k = 0
    arret: Optional[str] = None
    etat0_controle: Optional[Dict[str, object]] = None
    candidats_final: Optional[Dict[str, object]] = None

    def k_de(u: Ident) -> int:
        if len(u) == 1:
            return 0
        return profondeurs[u]

    it = 0
    while True:
        if mur_s is not None and t0 is not None and (time.perf_counter() - t0) > mur_s:
            arret = "I5_MUR_TEMPOREL"
            break
        ev = evaluer_candidats(ef, cache, lam, cache_v)
        if it == 0:
            n_pos0 = sum(
                1 for c in ev["candidats"] if c["coherence_full"] > theta
            )
            etat0_controle = {
                "n_adjacences_types": ev["n_adjacences_types"],
                "n_mesurables": ev["n_mesurables"],
                "n_pos_coherence_full": n_pos0,
            }
            if controle_etat0 is not None:
                attendu = {
                    "n_adjacences_types": controle_etat0[0],
                    "n_mesurables": controle_etat0[1],
                    "n_pos_coherence_full": controle_etat0[2],
                }
                if etat0_controle != attendu:
                    raise NonMesureError(
                        f"réduction à l'état 0 divergente : {etat0_controle!r} "
                        f"!= gel {attendu!r} — run NON-MESURE (§3.2)"
                    )
                etat0_controle["egal_gel"] = True
        best: Optional[dict] = None
        for c in ev["candidats"]:
            if not (c["coherence_full"] > theta):  # test STRICT (§3.2)
                continue
            if (
                best is None
                or c["coherence_full"] > best["coherence_full"]
                or (
                    c["coherence_full"] == best["coherence_full"]
                    and (c["a"], c["b"]) < (best["a"], best["b"])
                )
            ):
                best = c
        if best is None:
            arret = "ARRET_CRITERE"
            candidats_final = ev
            break
        if it >= max_fusions:
            # La borne ne mord que s'il RESTE une fusion possible (T75).
            arret = "I5_MAX_FUSIONS"
            candidats_final = ev
            break
        a, b = best["a"], best["b"]
        total_avant = ef.etat.total_unites_occurrences()
        appliquer_fusion(ef.etat, a, b)
        total_apres = ef.etat.total_unites_occurrences()
        if not (total_apres < total_avant):
            raise ValueError(
                f"invariant de décroissance violé à l'itération {it + 1} : "
                f"{total_avant} → {total_apres}"
            )
        c_new = a + b
        k_new = max(k_de(a), k_de(b)) + 1
        if c_new in profondeurs and profondeurs[c_new] != k_new:
            n_conflits_k += 1  # premier k conservé (déclaré, attendu 0)
        else:
            profondeurs.setdefault(c_new, k_new)
        it += 1
        journal.append(
            {
                "iteration": it,
                "a": list(a),
                "b": list(b),
                "coherence_full": best["coherence_full"],
                "contribution_H": best["coherence_h"],
                "contribution_lambda_delta": lam * best["delta_ctx"],
                "delta_ctx": best["delta_ctx"],
                "frequence": best["frequence"],
                "k": profondeurs[c_new],
                "n_remplacements_occ": total_avant - total_apres,
                "n_paires_exclues_non_mesurables": ev["n_exclues_non_mesurables"],
            }
        )

    resume_final: Optional[Dict[str, object]] = None
    if candidats_final is not None:
        cands = candidats_final["candidats"]
        vals = [c["coherence_full"] for c in cands]
        resume_final = {
            "n_adjacences_types": candidats_final["n_adjacences_types"],
            "n_mesurables": candidats_final["n_mesurables"],
            "n_exclues_non_mesurables": candidats_final["n_exclues_non_mesurables"],
            "n_pos": sum(1 for v in vals if v > theta),
            "n_neg": sum(1 for v in vals if v < theta),
            "n_zero": sum(1 for v in vals if v == theta),
            "distribution_coherence_full": _quartiles77(vals) if vals else None,
            "distribution_contribution_H": _quartiles77(
                [c["coherence_h"] for c in cands]
            )
            if cands
            else None,
            "distribution_contribution_lambda_delta": _quartiles77(
                [lam * c["delta_ctx"] for c in cands]
            )
            if cands
            else None,
            "candidats_refuses": [
                {
                    "a": list(c["a"]),
                    "b": list(c["b"]),
                    "coherence_h": c["coherence_h"],
                    "contribution_lambda_delta": lam * c["delta_ctx"],
                    "coherence_full": c["coherence_full"],
                    "frequence": c["frequence"],
                }
                for c in cands
            ],
            "exclues_en_extension": candidats_final["exclues"],
        }

    return {
        "journal": journal,
        "n_fusions": len(journal),
        "arret": arret,
        "profondeurs": profondeurs,
        "n_conflits_k": n_conflits_k,
        "etat0_controle": etat0_controle,
        "etat_final_candidats": resume_final,
    }


def n_partagees(vocab: Dict[str, object]) -> Dict[str, object]:
    """La grandeur de verdict (§2) : types d'unités finales à k ≥ 1 ET
    ``n_types_mots ≥ 2`` (champ de ``vocabulaire_final``, recette T75).

    Sélecteur fonction du seul corpus (CONTRAT T54) ; distribution complète
    de ``n_types_mots`` des unités fusionnées publiée (réfutabilité §6).
    """
    fusionnees = [e for e in vocab["unites"] if e["k"] >= 1]
    partagees = [e for e in fusionnees if e["n_types_mots"] >= 2]
    dist: Dict[str, int] = {}
    for e in fusionnees:
        dist[str(e["n_types_mots"])] = dist.get(str(e["n_types_mots"]), 0) + 1
    return {
        "n_types_fusionnes": len(fusionnees),
        "n_partagees": len(partagees),
        "seuil_branche_a": SEUIL_PARTAGE,
        "distribution_n_types_mots_fusionnes": dict(
            sorted(dist.items(), key=lambda kv: int(kv[0]))
        ),
        "partagees_en_extension": [
            {
                "ident": e["ident"],
                "rendu_ascii": e["rendu_ascii"],
                "k": e["k"],
                "n_types_mots": e["n_types_mots"],
                "n_occurrences": e["n_occurrences"],
            }
            for e in partagees
        ],
    }


# ---------------------------------------------------------------------------
# GC-82 [PORTEUR] (§3.1) — 9 grandeurs au bit contre les artefacts T80
# ---------------------------------------------------------------------------

def derivation_etat0(
    proj: Dict[str, object], cache: CacheStab, table: Sequence[dict]
) -> Dict[str, object]:
    """Re-dérive les grandeurs T80 par la route GELÉE (stab_decomp, zéro octet).

    Chaîne : ``intra_mesurables_paires`` → ``recette_lambda_intra`` →
    ``delta_ctx_intra_paires`` → ``admissibilite_coherence_full`` (λ = le
    λ*_intra re-dérivé, comme au T80).
    """
    vals, comptes = intra_mesurables_paires(
        proj["paire"], proj["hote_mot"], proj["pool"]
    )
    lam = recette_lambda_intra(vals, cache)
    delta = delta_ctx_intra_paires(
        proj["paire"], proj["g0"], proj["hote_mot"], proj["pool"]
    )
    adm = admissibilite_coherence_full(
        delta["deltas"], cache, lam["lambda_star_intra"], table
    )
    return {
        "n_types_paires": comptes["n_types_paires"],
        "n_paires_creuses": comptes["n_paires_creuses"],
        "n_paires_sans_strate_s2": comptes["n_paires_sans_strate_s2"],
        "n_intra_mesurables": comptes["n_intra_mesurables"],
        "iqr_h_norm": lam["iqr_h_norm"],
        "iqr_v_intra_d1": lam["iqr_v_intra_d1"],
        "lambda_star_intra": lam["lambda_star_intra"],
        "delta_n_pos": delta["n_pos"],
        "delta_n_neg": delta["n_neg"],
        "delta_n_zero": delta["n_zero"],
        "n_pos_coherence_full": adm["n_pos"],
        "n_paires_coherence_full": adm["n_paires"],
        "top_ident": adm["top_paires"][0]["ident"],
        "top_coherence_full": adm["top_paires"][0]["coherence_full"],
    }


def attendus_depuis_artefacts(
    art_lambda: Dict[str, object], art_adm: Dict[str, object]
) -> Dict[str, object]:
    """Collationne les grandeurs attendues DEPUIS les artefacts T80 (clause
    P2 : le chiffre de prose ne lie pas, l'artefact fait foi)."""
    carte = art_lambda["carte_intra_eve"]
    lam = art_lambda["lambda_intra"]
    delta = art_adm["delta_ctx_intra_eve"]
    full = art_adm["coherence_full_intra_eve"]
    return {
        "n_types_paires": carte["n_types_paires"],
        "n_paires_creuses": carte["n_paires_creuses"],
        "n_paires_sans_strate_s2": carte["n_paires_sans_strate_s2"],
        "n_intra_mesurables": carte["n_intra_mesurables"],
        "iqr_h_norm": lam["iqr_h_norm"],
        "iqr_v_intra_d1": lam["iqr_v_intra_d1"],
        "lambda_star_intra": lam["lambda_star_intra"],
        "delta_n_pos": delta["n_pos"],
        "delta_n_neg": delta["n_neg"],
        "delta_n_zero": delta["n_zero"],
        "n_pos_coherence_full": full["n_pos"],
        "n_paires_coherence_full": full["n_paires"],
        "top_ident": full["top_paires"][0]["ident"],
        "top_coherence_full": full["top_paires"][0]["coherence_full"],
    }


def comparer_gc82(
    attendus: Dict[str, object], obtenus: Dict[str, object]
) -> Dict[str, object]:
    """Comparaison AU BIT (== exact, entiers ET flottants) grandeur par
    grandeur. 1 mismatch ⇒ ``NonMesureError`` (I0 : ZÉRO mesure aval)."""
    detail: Dict[str, object] = {}
    mismatches: List[str] = []
    for cle in attendus:
        egal = obtenus[cle] == attendus[cle]
        detail[cle] = {
            "attendu": attendus[cle],
            "obtenu": obtenus[cle],
            "egal_au_bit": egal,
        }
        if not egal:
            mismatches.append(cle)
    if mismatches:
        raise NonMesureError(
            f"GC-82 : {len(mismatches)} mismatch au bit ({mismatches}) — "
            "I0, ZÉRO mesure aval (§3.1)"
        )
    return detail


def garde_gc82(
    proj: Dict[str, object],
    cache: CacheStab,
    table: Sequence[dict],
    art_lambda: Dict[str, object],
    art_adm: Dict[str, object],
) -> Dict[str, object]:
    """GC-82 [PORTEUR] : re-dérivation + comparaison au bit + garde λ gelé.

    Vérifie AUSSI que ``LAMBDA_GELE`` == λ*_intra re-dérivé == λ de
    l'artefact (l'identité de recette G81-5 est une égalité de flottants).
    """
    obtenus = derivation_etat0(proj, cache, table)
    attendus = attendus_depuis_artefacts(art_lambda, art_adm)
    detail = comparer_gc82(attendus, obtenus)
    if not (LAMBDA_GELE == obtenus["lambda_star_intra"]):
        raise NonMesureError(
            f"GC-82 : LAMBDA_GELE {LAMBDA_GELE!r} != λ re-dérivé "
            f"{obtenus['lambda_star_intra']!r} — I0"
        )
    return {
        "grandeurs": detail,
        "lambda_gele_egal_rederive": True,
        "statut": "PASS (14 grandeurs au bit, dont les 9 gelées §3.1 + "
                  "comptes full + paire top)",
    }


def gc82_mutation(
    proj: Dict[str, object],
    cache: CacheStab,
    table: Sequence[dict],
    art_lambda: Dict[str, object],
    art_adm: Dict[str, object],
) -> Dict[str, object]:
    """Branche négative de GC-82, attestée par MUTATION (§3.1) : 1 valeur
    altérée (dernier bit de λ via ``math.nextafter``) ⇒ mismatch constaté."""
    attendus = attendus_depuis_artefacts(art_lambda, art_adm)
    mute = dict(attendus)
    mute["lambda_star_intra"] = math.nextafter(
        float(attendus["lambda_star_intra"]), math.inf
    )
    obtenus = derivation_etat0(proj, cache, table)
    try:
        comparer_gc82(mute, obtenus)
        return {"mordante": False}
    except NonMesureError as exc:
        return {
            "mordante": True,
            "mutation": "lambda_star_intra += 1 ulp (math.nextafter)",
            "clause": "1 mismatch au bit => I0, ZERO mesure aval (§3.1)",
            "message": str(exc),
        }


# ---------------------------------------------------------------------------
# Vierge (§4) — RESERVE_v2, conventions §3.5, replay ordonné, J_sel/J_base
# ---------------------------------------------------------------------------

def _md5(chemin: str) -> str:
    with open(chemin, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def verifier_pack_aba_v2(dossier: str) -> Dict[str, object]:
    """md5 des 7 fichiers du pack contre le gel de réception (AVANT lecture).

    Mismatch ⇒ ``NonMesureError`` : la tranche n'est PAS lue.
    """
    import os

    detail: Dict[str, object] = {}
    mismatches: List[str] = []
    for nom, attendu in MD5_PACK_ABA_V2.items():
        obtenu = _md5(os.path.join(dossier, nom))
        detail[nom] = {"attendu": attendu, "obtenu": obtenu, "egal": obtenu == attendu}
        if obtenu != attendu:
            mismatches.append(nom)
    if mismatches:
        raise NonMesureError(
            f"pack aba_v2 : md5 divergents {mismatches} — tranche NON lue (§4)"
        )
    return detail


def contextes_reserve_v2(
    chemin: str,
    debut: int = RESERVE_V2_DEBUT,
    fin: int = RESERVE_V2_FIN,
) -> Tuple[List[str], Dict[str, int]]:
    """Contextes de la tranche RESERVE_v2 (conventions §3.5, G81-6 transcrite).

    Chaque ligne passe par ``spiraton.data.aba.try_parse_aba_line`` ; une
    ligne qui parse fournit EXACTEMENT 3 contextes dans l'ordre (seg_a,
    seg_b, seg_a_prime) ; ligne non vide qui ne parse pas (None) : comptée
    ``n_lignes_non_cycle`` et exclue. Seuls les cycles d'ordinal
    [debut, fin] fournissent des contextes ; les cycles 1..debut−1 sont
    parsés pour l'ordinal SEULEMENT (contenu non utilisé) ; la lecture
    S'ARRÊTE au cycle ``fin`` — TEST_v2 (2049+) jamais lu.
    """
    from spiraton.data.aba import try_parse_aba_line

    contextes: List[str] = []
    n_cycles = 0
    n_non_cycles = 0
    with open(chemin, "r", encoding="utf-8") as fh:
        for ligne in fh:
            cycle = try_parse_aba_line(ligne)
            if cycle is None:
                if ligne.strip():
                    n_non_cycles += 1
                continue
            n_cycles += 1
            if n_cycles >= debut:
                contextes.append(cycle.seg_a.text)
                contextes.append(cycle.seg_b.text)
                contextes.append(cycle.seg_a_prime.text)
            if n_cycles == fin:
                break
    return contextes, {
        "n_cycles_lus": n_cycles,
        "n_cycles_tranche": max(0, n_cycles - debut + 1),
        "n_contextes": len(contextes),
        "n_lignes_non_cycle": n_non_cycles,
        "tranche": [debut, fin],
    }


def verdict_vierge(
    j_sel: Optional[float],
    j_base: Optional[float],
    n_mesurables_sel: int,
    n_pos_base: int,
    n_neg_base: int,
) -> str:
    """Verdicts gelés (§4, gouvernent P-7 SEUL) : GÉNÉRALISE ssi
    J_sel ≥ 2×J_base ; NE GÉNÉRALISE PAS ssi J_sel ≤ J_base ; INDISTINCT
    entre les deux ; NON-MESURE si n_mesurables < 30 OU baseline dégénérée
    (n_pos_base < 1 OU n_neg_base < 1)."""
    if (
        n_mesurables_sel < MIN_MESURABLES_VIERGE
        or n_pos_base < 1
        or n_neg_base < 1
        or j_sel is None
        or j_base is None
    ):
        return "NON-MESURE"
    if j_sel >= 2.0 * j_base:
        return "GENERALISE"
    if j_sel <= j_base:
        return "NE GENERALISE PAS"
    return "INDISTINCT"


def replay_vierge(
    ef_v2: EtatFusionCtx,
    cache: CacheStab,
    journal: Sequence[dict],
    lam: float = LAMBDA_GELE,
    mur_s: Optional[float] = None,
    t0: Optional[float] = None,
) -> Dict[str, object]:
    """Replay ORDONNÉ du journal eve sur l'état RESERVE_v2 (§4).

    Pour chaque fusion i : à l'état V2 juste avant son application,
    ``Coherence_full_intra^V2(a_i, b_i)`` (même λ, V^{d1} sur les
    occurrences V2, strates = mots-hôtes V2) ; puis application (balayage
    standard). Zéro occurrence = fusion NON ATTESTÉE, comptée, on passe.
    Mesurable ssi adjacence attestée ET les trois V^{d1} mesurables.
    ``J_base`` = part positive parmi TOUTES les paires candidates
    mesurables de l'état V2 INITIAL (1er passage, même critère, même λ).
    """
    cache_v = _CacheVd1(ef_v2)

    # --- J_base : 1er passage sur l'état V2 initial -------------------------
    ev0 = evaluer_candidats(ef_v2, cache, lam, cache_v)
    n_pos_base = sum(1 for c in ev0["candidats"] if c["coherence_full"] > 0.0)
    n_neg_base = ev0["n_mesurables"] - n_pos_base  # part non positive (<= 0)
    j_base = (n_pos_base / ev0["n_mesurables"]) if ev0["n_mesurables"] else None

    # --- Replay ordonné -----------------------------------------------------
    par_fusion: List[dict] = []
    n_attestees = 0
    n_non_attestees = 0
    n_mesurables_sel = 0
    n_pos_sel = 0
    mur_atteint = False
    for e in journal:
        if mur_s is not None and t0 is not None and (time.perf_counter() - t0) > mur_s:
            mur_atteint = True
            break
        a, b = tuple(e["a"]), tuple(e["b"])
        unites_idx, adj_idx = occurrences_courantes(ef_v2)
        idx = adj_idx.get((a, b))
        rec: Dict[str, object] = {"iteration_eve": e["iteration"], "a": list(a), "b": list(b)}
        if idx is None:
            n_non_attestees += 1
            rec["attestee"] = False
            par_fusion.append(rec)
            continue
        n_attestees += 1
        rec["attestee"] = True
        rec["frequence_v2"] = len(idx)
        v_a = cache_v.v(a, unites_idx.get(a, []))
        v_b = cache_v.v(b, unites_idx.get(b, []))
        v_ab = cache_v.v(a + b, idx)
        if v_a is None or v_b is None or v_ab is None:
            rec["mesurable"] = False
        else:
            delta = 0.5 * (v_a + v_b) - v_ab
            coh_h = cache.coherence(a, b)
            full = coh_h + lam * delta
            n_mesurables_sel += 1
            if full > 0.0:
                n_pos_sel += 1
            rec["mesurable"] = True
            rec["coherence_h"] = coh_h
            rec["contribution_lambda_delta"] = lam * delta
            rec["coherence_full_v2"] = full
        appliquer_fusion(ef_v2.etat, a, b)  # replay standard (adjacence attestée)
        par_fusion.append(rec)

    j_sel = (n_pos_sel / n_mesurables_sel) if n_mesurables_sel else None
    verdict = (
        "NON STATUABLE (I5_MUR_V2 : consommation partielle consignée)"
        if mur_atteint
        else verdict_vierge(j_sel, j_base, n_mesurables_sel, n_pos_base, n_neg_base)
    )
    return {
        "n_fusions_journal": len(journal),
        "n_fusions_rejouees": len(par_fusion),
        "mur_atteint": mur_atteint,
        "n_attestees": n_attestees,
        "n_non_attestees": n_non_attestees,
        "n_mesurables_sel": n_mesurables_sel,
        "n_pos_sel": n_pos_sel,
        "j_sel": j_sel,
        "baseline": {
            "n_candidats_types": ev0["n_adjacences_types"],
            "n_mesurables_base": ev0["n_mesurables"],
            "n_exclues_non_mesurables_base": ev0["n_exclues_non_mesurables"],
            "n_pos_base": n_pos_base,
            "n_non_pos_base": n_neg_base,
            "j_base": j_base,
        },
        "verdict": verdict,
        "seuils": {
            "generalise": "J_sel >= 2*J_base",
            "ne_generalise_pas": "J_sel <= J_base",
            "min_mesurables": MIN_MESURABLES_VIERGE,
            "specificite_baseline": "n_pos_base >= 1 ET n_non_pos_base >= 1",
        },
        "par_fusion": par_fusion,
    }


# ---------------------------------------------------------------------------
# Étalonnage des murs (§3.4) — T33 : ~1 % mesuré, extrapolation gelée
# ---------------------------------------------------------------------------

def etalonner_t82(chemin_eve: str) -> Dict[str, object]:
    """t_cal = garde sha + tokenisation + état + 10 itérations du critère
    PLEIN (Δ_ctx compris) sur les 9 premières phrases d'eve (~1 %).
    mur_s = min(2000 × t_cal, 3600 s) — plafond doublé vs T75 (motivé §3.4)."""
    ctx, _ = contextes_eve(chemin_eve)
    t0 = time.perf_counter()
    tok, _prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T82)
    table = table_miroir()
    cache = CacheStab(table)
    tokens = [tok.tokenize_sequences(c) for c in ctx[:N_PHRASES_CAL]]
    ef = EtatFusionCtx(tokens)
    r = executer_fusions_ctx(ef, cache, LAMBDA_GELE, max_fusions=N_ITERATIONS_CAL)
    t_cal = time.perf_counter() - t0
    return {
        "n_phrases": N_PHRASES_CAL,
        "n_tokens": ef.etat.comptes["n_tokens"],
        "n_iterations_cal": N_ITERATIONS_CAL,
        "n_fusions_cal": r["n_fusions"],
        "arret_cal": r["arret"],
        "t_cal_s": t_cal,
        "facteur_mur": FACTEUR_MUR_EVE,
        "mur_s": min(FACTEUR_MUR_EVE * t_cal, PLAFOND_EVE_S),
        "plafond_dur_s": PLAFOND_EVE_S,
    }


def etalonner_v2(chemin_dataset: str, tok, journal: Sequence[dict]) -> Dict[str, object]:
    """t_cal_v2 = tokenisation + état + 1 pas de replay (évaluation du
    critère plein sur la 1re fusion du journal) sur les 10 PREMIERS cycles
    de la tranche (~1 %). mur_v2 = min(300 × t_cal_v2, 1800 s)."""
    t0 = time.perf_counter()
    ctx, cpt = contextes_reserve_v2(
        chemin_dataset, debut=RESERVE_V2_DEBUT, fin=RESERVE_V2_DEBUT + N_CYCLES_CAL_V2 - 1
    )
    table = table_miroir()
    cache = CacheStab(table)
    tokens = [tok.tokenize_sequences(c) for c in ctx]
    ef = EtatFusionCtx(tokens)
    _ = replay_vierge(ef, cache, list(journal)[:1])
    t_cal = time.perf_counter() - t0
    return {
        "n_cycles_cal": N_CYCLES_CAL_V2,
        "n_contextes_cal": cpt["n_contextes"],
        "n_tokens_cal": ef.etat.comptes["n_tokens"],
        "t_cal_v2_s": t_cal,
        "facteur_mur_v2": FACTEUR_MUR_V2,
        "mur_v2_s": min(FACTEUR_MUR_V2 * t_cal, PLAFOND_V2_S),
        "plafond_dur_v2_s": PLAFOND_V2_S,
    }


# ---------------------------------------------------------------------------
# PORTÉES (§6) — deux branches par instrument, témoins construits
# ---------------------------------------------------------------------------

def _tok_temoin(
    text: str, ctx: Tuple[float, float, float], g0_ids: Sequence[int]
) -> dict:
    """Token-témoin (interface R16) : dims 28-30 forcées, table TABLE_TEMOIN."""
    v = np.zeros(33, dtype=np.float32)
    v[28], v[29], v[30] = ctx
    return {
        "text": text,
        "vector33d": v,
        "g0_ids": list(g0_ids),
        "g1_nb": [len(g0_ids)],
    }


def _temoin_fusible_delta_zero() -> List[List[dict]]:
    """Témoin A : 2×« ab » + 2×« ad », c(occ) identiques PAR type ⇒ tous les
    V^{d1} = 0 exact ⇒ Δ = 0 ⇒ full = Coherence_H (> 0 sur (0,1) et (0,3)
    par A1/A2 de la table témoin) ⇒ 2 fusions attendues, ARRET_CRITERE."""
    return [
        [_tok_temoin("ab", (1.0, 2.0, 3.0), [0, 1])],
        [_tok_temoin("ab", (1.0, 2.0, 3.0), [0, 1])],
        [_tok_temoin("ad", (4.0, 5.0, 6.0), [0, 3])],
        [_tok_temoin("ad", (4.0, 5.0, 6.0), [0, 3])],
    ]


def _temoin_contexte_porte() -> List[List[dict]]:
    """Témoin B (LE témoin du tour) : l'auto-paire (2,2) a Coherence_H < 0
    (A2, jamais fusible sous H seul) mais ses parts vivent AUSSI dans des
    hôtes « c » à c(occ) dispersés ⇒ V^{d1}(2,) ≫ 0 tandis que V^{d1} de
    l'adjacence (hôtes « cc » bit-identiques) = 0 ⇒ Δ > 0 grand ⇒
    full = coh_H + λ·Δ > 0 : la fusion est portée PAR LE CONTEXTE SEUL."""
    return [
        [_tok_temoin("cc", (5.0, 5.0, 5.0), [2, 2])],
        [_tok_temoin("cc", (5.0, 5.0, 5.0), [2, 2])],
        [_tok_temoin("c", (0.0, 0.0, 0.0), [2])],
        [_tok_temoin("c", (10.0, 10.0, 10.0), [2])],
    ]


def _temoin_negatif() -> List[List[dict]]:
    """Témoin C : auto-paire (2,2) seule, c(occ) bit-identiques ⇒ Δ = 0 ⇒
    full = Coherence_H < 0 ⇒ 0 fusion (branche négative sur témoin)."""
    return [
        [_tok_temoin("cc", (5.0, 5.0, 5.0), [2, 2])],
        [_tok_temoin("cc", (5.0, 5.0, 5.0), [2, 2])],
    ]


def _temoin_egalite() -> List[List[dict]]:
    """Témoin D : auto-paire (3,3) uniforme (A2 : Coherence_H = 0 exact),
    c(occ) bit-identiques ⇒ Δ = 0 ⇒ full == 0.0 exact ⇒ θ strict refuse."""
    return [
        [_tok_temoin("dd", (7.0, 7.0, 7.0), [3, 3])],
        [_tok_temoin("dd", (7.0, 7.0, 7.0), [3, 3])],
    ]


def _temoin_partage_zero() -> List[List[dict]]:
    """Témoin n_partagées = 0 : un seul type de mot ⇒ l'unité fusionnée vit
    dans 1 type ⇒ n_types_mots = 1 ⇒ n_partagées = 0 (borne basse)."""
    return [
        [_tok_temoin("ab", (1.0, 2.0, 3.0), [0, 1])],
        [_tok_temoin("ab", (1.0, 2.0, 3.0), [0, 1])],
    ]


def _temoin_partage_un() -> List[List[dict]]:
    """Témoin n_partagées ≥ 1 : deux types de mots (« ab » ×2 et « dab » ×1)
    partagent l'adjacence (0,1) ⇒ après fusion, l'unité (0,1) vit dans 2
    types (l'adjacence restante de « dab » est non mesurable : 1 occurrence
    ⇒ exclue-comptée, la cascade s'arrête)."""
    return [
        [_tok_temoin("ab", (1.0, 2.0, 3.0), [0, 1])],
        [_tok_temoin("ab", (1.0, 2.0, 3.0), [0, 1])],
        [_tok_temoin("dab", (4.0, 4.0, 4.0), [3, 0, 1])],
    ]


def _run_temoin(
    contextes: List[List[dict]],
    max_fusions: int = MAX_FUSIONS,
    mur_s: Optional[float] = None,
    t0: Optional[float] = None,
) -> Tuple[EtatFusionCtx, Dict[str, object]]:
    ef = EtatFusionCtx(contextes)
    cache = CacheStab(TABLE_TEMOIN)
    r = executer_fusions_ctx(
        ef, cache, LAMBDA_GELE, max_fusions=max_fusions, mur_s=mur_s, t0=t0
    )
    return ef, r


def portees_t82() -> Dict[str, object]:
    """PORTÉE de chaque instrument du tour (§6) — publiée AVANT toute mesure
    sur le réel. Chaque branche est ATTEINTE sur témoins construits ; les
    branches réelles de GC-82 vivent dans TOUR82_GC82.json (PASS + mutation).
    """
    res: Dict[str, object] = {
        "gap_shuffle_ordre": (
            "N-A déclaré (§6.9) : aucun verdict d'ordre/clôture A→B→A′ ce "
            "tour — les paires sont ordonnées dans le critère mais aucun "
            "verdict ne compare des ordres (règle T22/T60 par absence)"
        )
    }

    # --- Coherence_full_intra : positif / négatif / égalité exacte ----------
    cache_t = CacheStab(TABLE_TEMOIN)
    ef_b, r_b = _run_temoin(_temoin_contexte_porte())
    ef_c, r_c = _run_temoin(_temoin_negatif())
    ef_d, r_d = _run_temoin(_temoin_egalite())
    cand_c = r_c["etat_final_candidats"]["candidats_refuses"]
    cand_d = r_d["etat_final_candidats"]["candidats_refuses"]
    res["coherence_full_intra"] = {
        "branche_positive_portee_par_contexte": {
            "n_fusions": r_b["n_fusions"],
            "coherence_h_negative": r_b["journal"][0]["contribution_H"] < 0.0,
            "contribution_lambda_delta": r_b["journal"][0]["contribution_lambda_delta"],
            "coherence_full": r_b["journal"][0]["coherence_full"],
            "atteinte": r_b["n_fusions"] >= 1
            and r_b["journal"][0]["contribution_H"] < 0.0
            and r_b["journal"][0]["coherence_full"] > 0.0,
        },
        "branche_negative": {
            "coherence_full": cand_c[0]["coherence_full"],
            "n_fusions": r_c["n_fusions"],
            "atteinte": cand_c[0]["coherence_full"] < 0.0 and r_c["n_fusions"] == 0,
        },
        "branche_egalite_exacte_theta_strict": {
            "coherence_full": cand_d[0]["coherence_full"],
            "n_fusions": r_d["n_fusions"],
            "atteinte": cand_d[0]["coherence_full"] == 0.0 and r_d["n_fusions"] == 0,
        },
        "borne_declaree": (
            "composante inter-hôtes EXCLUE par construction (G81-7) ; "
            "aucune sémantique (R-INSTRUMENT : projection × corpus × interface)"
        ),
    }

    # --- Boucle de fusion : gardes mordantes (§6.3) -------------------------
    _, r_max = _run_temoin(_temoin_fusible_delta_zero(), max_fusions=1)
    try:
        ef_inj = EtatFusionCtx(_temoin_fusible_delta_zero())
        appliquer_fusion(ef_inj.etat, (2,), (3,))  # paire absente ⇒ garde
        inj_mord = False
    except ValueError:
        inj_mord = True
    _, r_mur = _run_temoin(
        _temoin_fusible_delta_zero(), mur_s=0.0, t0=time.perf_counter() - 1.0
    )
    _, r_full = _run_temoin(_temoin_fusible_delta_zero())
    res["boucle_fusion"] = {
        "garde_max_fusions": {
            "arret": r_max["arret"],
            "mordante": r_max["arret"] == "I5_MAX_FUSIONS",
        },
        "garde_invariant_decroissance": {"mordante_par_injection": inj_mord},
        "garde_mur_temporel": {
            "arret": r_mur["arret"],
            "mordante": r_mur["arret"] == "I5_MUR_TEMPOREL",
        },
        "temoin_arret_critere": {
            "arret": r_full["arret"],
            "n_fusions": r_full["n_fusions"],
            "atteint": r_full["arret"] == "ARRET_CRITERE" and r_full["n_fusions"] == 2,
        },
    }

    # --- n_partagées : bornes 0 et ≥ 1 par construction (§6.5) --------------
    table_t = TABLE_TEMOIN
    ef_p0, r_p0 = _run_temoin(_temoin_partage_zero())
    vocab_p0 = vocabulaire_final(ef_p0.etat, r_p0["profondeurs"], table_t)
    np0 = n_partagees(vocab_p0)
    ef_p1, r_p1 = _run_temoin(_temoin_partage_un())
    vocab_p1 = vocabulaire_final(ef_p1.etat, r_p1["profondeurs"], table_t)
    np1 = n_partagees(vocab_p1)
    res["n_partagees"] = {
        "selecteur": "n_types_mots >= 2 (fonction du seul corpus, CONTRAT T54)",
        "branche_zero": {
            "n_partagees": np0["n_partagees"],
            "atteinte": np0["n_partagees"] == 0 and np0["n_types_fusionnes"] >= 1,
        },
        "branche_au_moins_un": {
            "n_partagees": np1["n_partagees"],
            "atteinte": np1["n_partagees"] >= 1,
        },
        "note_temoin": (
            "borne ≥ 1 atteinte par 2 TYPES de mots-hôtes partageant l'unité "
            "fusionnée (« ab », « dab ») — lecture opérationnelle du témoin "
            "« mot répété » de l'émission : le partage se compte en types"
        ),
    }

    # --- Torsion A : bornes + NON-MESURE < 30 types (§6.6) ------------------
    from spiraton.experimental.bpe_logos import jaccard_distance

    d0, _ = jaccard_distance(frozenset({1, 2}), frozenset({1, 2}))
    d1, _ = jaccard_distance(frozenset({1}), frozenset({2}))
    dv, flag = jaccard_distance(frozenset(), frozenset())
    torsion_temoin = torsion_corpus(ef_p1.etat)
    res["torsion_a"] = {
        "borne_0": d0 == 0.0,
        "borne_1": d1 == 1.0,
        "double_vide_vaut_0": dv == 0.0 and flag,
        "branche_non_mesure_moins_de_30_types": {
            "n_types": torsion_temoin["n_types_mesurables"],
            "classement": torsion_temoin["classement"],
            "atteinte": torsion_temoin["classement"] == "NON-MESURE",
        },
        "g81_2_table_permutee": (
            "N-A motivé (§3.3) : critère sous pont (a) — σ_unit = constantes "
            "de table via le miroir Python, Δ_ctx = dims 28-30 héritées ; "
            "AUCUNE fonction syllabique du C dans le critère"
        ),
    }

    # --- Audit fréquence : ±1 exacts, NON-MESURE, DEGENERE (§6.7) -----------
    rho_p = spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0])
    rho_m = spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [50.0, 40.0, 30.0, 20.0, 10.0])
    rho_deg = spearman_rho([1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    res["audit_frequence"] = {
        "plus_un_exact": rho_p == 1.0,
        "moins_un_exact": rho_m == -1.0,
        "non_mesure_n_inferieur_5": spearman_rho([1.0], [1.0]) is None,
        "degenere_rangs_constants": rho_deg is None,
        "seuil_i6": SEUIL_RHO_SPEARMAN,
        "poids_accru_declare": (
            "Δ_ctx dépend des occurrences : la fréquence peut s'infiltrer "
            "par la variance — c'est pourquoi l'audit est au protocole (§3.3)"
        ),
    }

    # --- Vierge J_sel/J_base : 4 sorties atteignables (§6.8) ----------------
    res["vierge_j_sel_j_base"] = {
        "branche_generalise": verdict_vierge(0.8, 0.3, 40, 5, 5) == "GENERALISE",
        "branche_ne_generalise_pas": verdict_vierge(0.2, 0.3, 40, 5, 5)
        == "NE GENERALISE PAS",
        "branche_indistinct": verdict_vierge(0.5, 0.3, 40, 5, 5) == "INDISTINCT",
        "branche_non_mesure_effectif": verdict_vierge(0.8, 0.3, 29, 5, 5)
        == "NON-MESURE",
        "branche_non_mesure_baseline_degeneree": verdict_vierge(0.8, 0.0, 40, 0, 45)
        == "NON-MESURE",
        "detecte": (
            "le transfert du SIGNE du critère hors corpus d'apprentissage — "
            "pas la qualité linguistique des unités (§6.8)"
        ),
    }

    # --- Replay : branches attestée / non attestée sur témoin ---------------
    ef_r = EtatFusionCtx(_temoin_partage_zero())
    faux_journal = [
        {"iteration": 1, "a": [0], "b": [1]},   # attestée sur « ab »
        {"iteration": 2, "a": [0], "b": [3]},   # jamais présente ⇒ non attestée
    ]
    rep = replay_vierge(ef_r, cache_t, faux_journal)
    res["replay_vierge"] = {
        "n_attestees": rep["n_attestees"],
        "n_non_attestees": rep["n_non_attestees"],
        "branches_atteintes": rep["n_attestees"] == 1 and rep["n_non_attestees"] == 1,
    }

    # --- k : profondeur ≥ 2 constructible (récursion réelle, l.279) ---------
    # « abc » ×2 (c identiques) + « ab » ×2 (c dispersés) : étage 1 = (0,1)
    # par H ; étage 2 = ((0,1),(2,)) porté par le contexte (V((0,1)) grand
    # via les hôtes « ab » dispersés, V(adjacence) = 0) ⇒ k = 2.
    ctx_k = [
        [_tok_temoin("abc", (1.0, 1.0, 1.0), [0, 1, 2])],
        [_tok_temoin("abc", (1.0, 1.0, 1.0), [0, 1, 2])],
        [_tok_temoin("ab", (0.0, 0.0, 0.0), [0, 1])],
        [_tok_temoin("ab", (10.0, 10.0, 10.0), [0, 1])],
    ]
    ef_k, r_k = _run_temoin(ctx_k)
    ks = [r_k["profondeurs"][u] for u in r_k["profondeurs"]]
    res["profondeur_k"] = {
        "k_max_temoin": max(ks) if ks else 0,
        "branche_k2_atteignable": (max(ks) if ks else 0) >= 2,
        "n_conflits_k": r_k["n_conflits_k"],
    }
    return res


# ---------------------------------------------------------------------------
# Pilote (artefacts TOUR82_* — racine écosystème, sans horodatage)
# ---------------------------------------------------------------------------

def _config_echo(
    md5_eve: Optional[str], so_sha256: Optional[str], murs: Optional[dict] = None
) -> Dict[str, object]:
    """Écho de configuration complet (clause P1 T81) — SANS horodatage :
    les artefacts de verdict se reproduisent au bit."""
    cfg: Dict[str, object] = {
        "tour": 82,
        "lambda_gele": LAMBDA_GELE,
        "lambda_recette_g81_5": LAMBDA_RECETTE,
        "theta_fusion": THETA_FUSION_T82,
        "max_fusions": MAX_FUSIONS,
        "seuil_partage": SEUIL_PARTAGE,
        "ctx_dims": list(CTX_DIMS),
        "ddof_v_intra": 1,
        "so_sha256": so_sha256,
        "so_sha256_reference_t82": SHA256_SO_REFERENCE_T82,
        "md5_corpus_eve": md5_eve,
        "contexte_execution": {
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "plateforme": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "locale": list(_locale.getlocale()),
            "encodage_prefere": _locale.getpreferredencoding(False),
        },
        "classe_reproductibilite": (
            "verdicts = compteurs entiers et signes (bit-stables) ; flottants "
            "d'accumulation float64, ordre d'opérations des modules gelés "
            "(clause P4 T81) ; artefacts sans horodatage, reproduction ×2 au bit"
        ),
    }
    if murs is not None:
        cfg["murs_geles"] = murs
    return cfg


def _ecrire_json(chemin: str, obj: Dict[str, object]) -> None:
    with open(chemin, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def _charger_json(chemin: str) -> Dict[str, object]:
    with open(chemin, "r", encoding="utf-8") as fh:
        return json.load(fh)


def stade_gc82(
    chemin_eve: str, chemin_art_lambda: str, chemin_art_adm: str, sortie: str
) -> None:
    """P1 : GC-82 [PORTEUR] — PASS au bit + branche négative par mutation."""
    tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T82)
    table = table_miroir()
    cache = CacheStab(table)
    art_lambda = _charger_json(chemin_art_lambda)
    art_adm = _charger_json(chemin_art_adm)
    ctx, cpt = contextes_eve(chemin_eve)
    proj = extraire_projection(tok.tokenize_sequences(c) for c in ctx)
    obj: Dict[str, object] = {
        "config": _config_echo(_md5(chemin_eve), prov["native_lib_sha256"]),
        "contextes_eve": cpt,
        "artefacts_references": {
            "TOUR80_LAMBDA_INTRA.json": _md5(chemin_art_lambda),
            "TOUR80_ADMISSIBILITE_INTRA.json": _md5(chemin_art_adm),
        },
        "gc82": garde_gc82(proj, cache, table, art_lambda, art_adm),
        "gc82_mutation": gc82_mutation(proj, cache, table, art_lambda, art_adm),
    }
    try:
        charger_tokenizer_garde(sha256_attendu="0" * 64)
        obj["garde_sha_so"] = {"mordante": False}
    except RuntimeError:
        obj["garde_sha_so"] = {
            "mordante": True,
            "clause": "mismatch => 0 mesure, tour NON-MESURE (§6.2) ; "
                      "comparaison par blob/fichier conservé, JAMAIS re-lié (P6)",
        }
    _ecrire_json(sortie, obj)


def stade_run(
    chemin_eve: str,
    chemin_art_lambda: str,
    chemin_art_adm: str,
    mur_s: Optional[float],
    dossier_sortie: str,
) -> None:
    """P4 : le run de fusion eve (GC-82 rejoué en tête — ZÉRO mesure sans lui).

    Écrit TOUR82_VOCAB.json (journal + vocabulaire + k + n_partagées),
    TOUR82_TORSION.json, TOUR82_AUDIT_FREQ.json.
    """
    import os

    tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T82)
    table = table_miroir()
    cache = CacheStab(table)
    md5_eve = _md5(chemin_eve)
    murs = {"mur_eve_s": mur_s}
    config = _config_echo(md5_eve, prov["native_lib_sha256"], murs)

    # --- GC-82 en tête de run (P1 avant P4 — ordre des portes §3.6) ---------
    art_lambda = _charger_json(chemin_art_lambda)
    art_adm = _charger_json(chemin_art_adm)
    ctx, cpt = contextes_eve(chemin_eve)
    tokens = [tok.tokenize_sequences(c) for c in ctx]
    proj = extraire_projection(tokens)
    gc = garde_gc82(proj, cache, table, art_lambda, art_adm)  # lève I0 sinon

    # --- Run de fusion (contrôle état 0 = réduction exacte, §3.2) -----------
    ef = EtatFusionCtx(tokens)
    t0 = time.perf_counter()
    r = executer_fusions_ctx(
        ef,
        cache,
        LAMBDA_GELE,
        max_fusions=MAX_FUSIONS,
        mur_s=mur_s,
        t0=t0,
        controle_etat0=(ETAT0_N_TYPES_PAIRES, ETAT0_N_MESURABLES, ETAT0_N_POS_FULL),
    )
    vocab = vocabulaire_final(ef.etat, r["profondeurs"], table)
    partage = n_partagees(vocab)

    vocab_obj: Dict[str, object] = {
        "config": config,
        "contextes_eve": cpt,
        "comptes_etat": dict(ef.etat.comptes),
        "gc82_statut": gc["statut"],
        "run_eve": {
            "arret": r["arret"],
            "n_fusions": r["n_fusions"],
            "n_conflits_k": r["n_conflits_k"],
            "etat0_controle": r["etat0_controle"],
            "journal": r["journal"],
            "etat_final_candidats": r["etat_final_candidats"],
        },
        "vocabulaire": vocab,
        "n_partagees": partage,
    }
    p = os.path.join(dossier_sortie, "TOUR82_VOCAB.json")
    _ecrire_json(p, vocab_obj)

    torsion_obj = {"config": config, "torsion_a": torsion_corpus(ef.etat)}
    p = os.path.join(dossier_sortie, "TOUR82_TORSION.json")
    _ecrire_json(p, torsion_obj)

    cohs = [e["coherence_full"] for e in r["journal"]]
    freqs = [float(e["frequence"]) for e in r["journal"]]
    rho = spearman_rho(cohs, freqs)
    audit: Dict[str, object] = {
        "n_fusions": len(cohs),
        "rho_spearman": rho,
        "seuil_i6": SEUIL_RHO_SPEARMAN,
        "classements": {
            "coherence_full_a_la_selection": cohs,
            "frequence_a_la_selection": freqs,
        },
    }
    if rho is None:
        audit["statut"] = (
            "NON-MESURE (n < 5)" if len(cohs) < MIN_FUSIONS_RHO
            else "DEGENERE (rangs constants)"
        )
    else:
        audit["statut"] = "I6" if abs(rho) >= SEUIL_RHO_SPEARMAN else "OK"
    p = os.path.join(dossier_sortie, "TOUR82_AUDIT_FREQ.json")
    _ecrire_json(p, {"config": config, "audit_frequence": audit})


def stade_vierge(
    chemin_pack: str,
    chemin_vocab: str,
    mur_v2_s: Optional[float],
    sortie: str,
) -> None:
    """P5 : le vierge (§4) — SEULEMENT si run eve = ARRET_CRITERE et
    n_fusions ≥ 1 (vérifié ici sur l'artefact). md5 du pack AVANT lecture."""
    import os

    vocab_obj = _charger_json(chemin_vocab)
    run = vocab_obj["run_eve"]
    if not (run["arret"] == "ARRET_CRITERE" and run["n_fusions"] >= 1):
        raise NonMesureError(
            f"condition de dépense non remplie (arret={run['arret']!r}, "
            f"n_fusions={run['n_fusions']}) : RESERVE_v2 reste INTACTE (§4)"
        )
    pack_detail = verifier_pack_aba_v2(chemin_pack)  # AVANT toute lecture
    chemin_dataset = os.path.join(chemin_pack, "dataset_aba_v2.txt")

    tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T82)
    table = table_miroir()
    cache = CacheStab(table)
    journal = run["journal"]

    ctx, cpt = contextes_reserve_v2(chemin_dataset)
    tokens = [tok.tokenize_sequences(c) for c in ctx]
    ef_v2 = EtatFusionCtx(tokens)
    t0 = time.perf_counter()
    rep = replay_vierge(ef_v2, cache, journal, mur_s=mur_v2_s, t0=t0)

    obj = {
        "config": _config_echo(None, prov["native_lib_sha256"], {"mur_v2_s": mur_v2_s}),
        "pack_md5": pack_detail,
        "tranche": cpt,
        "comptes_etat_v2": dict(ef_v2.etat.comptes),
        "consommation": (
            "RESERVE_v2 (cycles 1025-2048) CONSOMMÉE par tokenisation pour ce "
            "rôle (§4) ; TEST_v2 (2049-4096) JAMAIS lu (arrêt de lecture au "
            "cycle 2048)"
        ),
        "replay": rep,
    }
    _ecrire_json(sortie, obj)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stade",
        required=True,
        choices=["gc82", "portees", "etalonnage", "etalonnage-v2", "run", "vierge"],
    )
    ap.add_argument("--eve", default=None)
    ap.add_argument("--art-lambda", default=None, help="TOUR80_LAMBDA_INTRA.json")
    ap.add_argument("--art-adm", default=None, help="TOUR80_ADMISSIBILITE_INTRA.json")
    ap.add_argument("--pack", default=None, help="dossier aba_v2_lab_pack")
    ap.add_argument("--vocab", default=None, help="TOUR82_VOCAB.json (pour vierge)")
    ap.add_argument("--mur-s", type=float, default=None)
    ap.add_argument("--sortie", required=True, help="fichier JSON ou dossier (run)")
    args = ap.parse_args(argv)

    if args.stade == "gc82":
        stade_gc82(args.eve, args.art_lambda, args.art_adm, args.sortie)
    elif args.stade == "portees":
        _tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T82)
        _ecrire_json(
            args.sortie,
            {
                "config": _config_echo(None, prov["native_lib_sha256"]),
                "portees": portees_t82(),
            },
        )
    elif args.stade == "etalonnage":
        _ecrire_json(
            args.sortie, {"tour": 82, "etalonnage_eve": etalonner_t82(args.eve)}
        )
    elif args.stade == "etalonnage-v2":
        import os

        verifier_pack_aba_v2(args.pack)  # md5 AVANT lecture (§4)
        vocab_obj = _charger_json(args.vocab)
        run = vocab_obj["run_eve"]
        if not (run["arret"] == "ARRET_CRITERE" and run["n_fusions"] >= 1):
            raise NonMesureError("condition de dépense non remplie : V2 intact (§4)")
        tok, _prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T82)
        cal = etalonner_v2(
            os.path.join(args.pack, "dataset_aba_v2.txt"), tok, run["journal"]
        )
        obj = _charger_json(args.sortie) if os.path.exists(args.sortie) else {"tour": 82}
        obj["etalonnage_v2"] = cal
        _ecrire_json(args.sortie, obj)
    elif args.stade == "run":
        stade_run(args.eve, args.art_lambda, args.art_adm, args.mur_s, args.sortie)
    else:
        stade_vierge(args.pack, args.vocab, args.mur_s, args.sortie)


if __name__ == "__main__":  # pragma: no cover - pilote
    main()
