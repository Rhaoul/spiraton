"""BPE-Logos minimal — fusion G₀ → G₁' par critère ABA-cohérent (Tour 75).

Ce module PRODUIT un grain : à partir des séquences de phonèmes (g0_ids,
interface R16/T72), il fusionne itérativement des paires ordonnées adjacentes
INTRA-MOT selon un critère de gain de stabilité, puis mesure la **Torsion A**
entre le vocabulaire émergent (G₁') et la segmentation syllabique native (g1).
Étape 3 de ``brainstorming/GRANULARITE_RECURSIVE_SPEC_v0.3.1.md`` ; toutes les
définitions ci-dessous sont les gels de ``TOUR75_EMISSION.md`` (md5
``3a77633dd97983c2db095d64c0fd5cc4``) — rien n'est réglable en douce.

Définitions opérationnelles gelées (émission T75) :

- **σ_unit(u)** (§4.1) : u = tuple d'ids g0 (p₁,…,p_L) ; σ_unit ∈ ℝ^{4L} =
  concaténation ordonnée, PAR PHONÈME, des 4 constantes de table
  (``CHAMPS_CONSTANTES`` gelé T73 : flux, structure, energie_totale,
  impedance) lues dans le miroir Python ``phoneme_table_fr``. Fonction de
  l'identité seule — extension exacte du σ(syllabe) T73 à toute séquence.
- **Stab ≡ −H_norm** (§4.2) : H_norm par ``stab_grains.entropie_masse``
  (réutilisée SANS ÉDITION). Le terme λ·Var_ctx est INERTE à ces grains
  (R43 + T73 P-1/P-1b) ; λ reste NON CALIBRÉ. Ce que H mesure, en clair :
  la répartition de la masse articulatoire le long de la séquence phonémique.
  Toute lecture « morphologique » est interdite (R-INSTRUMENT).
- **Coherence(a,b)** (§5) : ``−H_norm(σ(a·b)) + ½[H_norm(σ(a)) + H_norm(σ(b))]``
  — le gain de stabilité de la jonction sur la juxtaposition. Propriétés
  analytiques déclarées a priori (§4.2) : (A1) identité de mélange exacte de
  H brut ; (A2) Coherence(a,a) ≤ 0 sous H_norm (égalité ssi σ(a) uniforme) ⇒
  l'auto-paire ne fusionne jamais avec θ = 0 strict ; (A3) Coherence est une
  fonction du seul contenu phonémique de la paire — la fréquence n'entre
  nulle part dans le critère.
- **θ_fusion = 0 strict** (§4.4) : on fusionne ssi ``coherence > 0.0``
  (float64, aucune tolérance). Zéro recalibration par corpus.
- **Fusion** (§4.3, §5) : paires ORDONNÉES adjacentes, strictement intra-mot
  (le mot = token de ``tokenize_sequences``) ; à chaque itération, UN type de
  paire (Coherence maximale ; ex æquo tranchés par ordre lexicographique
  croissant sur (tuple g0 de a, tuple g0 de b) — jamais fréquence, jamais
  graine) est fusionné PARTOUT, balayage gauche→droite non chevauchant.
- **Terminaison** (§4.4) : arrêt quand plus aucune paire n'a Coherence > 0 ;
  garde dure ``MAX_FUSIONS = 20 000`` ; invariant vérifié à chaque itération :
  le nombre total d'unités du corpus (pondéré occurrences) décroît
  strictement — garde mordante sinon. Mur temporel : §12 de l'émission.
- **Torsion A** (§4.6) : par TYPE de mot phonémique (identité = tuple g0
  complet, dédoublonné ; g0_len ≥ 2, drop = 0, g1_len ≥ 1) : distance de
  Jaccard entre coupures internes syllabiques (B_g1) et émergentes (B_G1') ;
  double-vide → 0 (compté séparément). Statistique de verdict : MÉDIANE D̃
  (pointe gelée, clause T40) ; moyenne/quartiles/pondéré-fréquence en INFO.
  ``taux_traversée`` = part des unités finales de longueur ≥ 2 (occurrences
  dans le corpus dédoublonné par type) dont l'empan contient ≥ 1 frontière
  syllabique interne stricte. Classement (liste close) : RETROUVE (D̃ ≤ 0,30),
  TRAVERSE (D̃ > 0,30 ET taux ≥ 0,30), IGNORE (D̃ > 0,30 ET taux < 0,30) ;
  < 30 types mesurables ⇒ NON-MESURE (I2).
- **Audit fréquence** (§4.7) : ρ de Spearman (rangs moyens, numpy lisible,
  zéro scipy) entre rang de Coherence à la sélection et rang de fréquence
  brute de la paire à la même itération, sur les fusions acceptées ;
  |ρ| ≥ 0,80 ⇒ I6 ; n < 5 ⇒ NON-MESURE.
- **k (profondeur)** (§4.8) : k(G₀) = 0 ; k(fusion(a,b)) = max(k(a),k(b)) + 1
  (assignée à l'événement de fusion ; si une identité déjà créée réapparaît
  avec un k différent, le conflit est COMPTÉ et le premier k est conservé —
  déclaré, attendu 0).

Contexte d'exécution (doctrine T74) : chaque artefact écho sa config complète
(θ, MAX_FUSIONS, sha256 du .so, md5 corpus, Python/numpy/plateforme).
Déterminisme intégral : AUCUN tirage aléatoire dans ce module.

Les verdicts vivent dans les artefacts ``TOUR75_*.json`` (racine écosystème),
pas ici : ce module ne contient que l'instrument et son pilote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import OrderedDict
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from spiraton.experimental.stab_grains import (
    CHAMPS_CONSTANTES,
    charger_tokenizer_garde,
    constantes_phoneme,
    contextes_claude_aba,
    contextes_eve,
    entropie_masse,
    table_miroir,
)

# ---------------------------------------------------------------------------
# Constantes gelées du tour (émission T75 — jamais cachées, jamais réglables)
# ---------------------------------------------------------------------------

#: sha256 COMPLET du .so de référence aval (gel T74, préfixe 4a1df2aa…).
#: Provenance : relevé ``sha256sum Tokenizer/bin/libspiratontokenizer.so`` du
#: 2026-08-11 (porte P0 du T75), préfixe vérifié contre le gel T74
#: (TOUR74_VERDICT). La constante T73 (3105131a…) reste intacte dans
#: ``stab_grains`` : gel d'UN tour, historique assumé — d'où le paramètre
#: ``sha256_attendu`` de ``charger_tokenizer_garde``.
SHA256_SO_REFERENCE_T75: str = (
    "4a1df2aaa2079caf6e5640e84c68c6db05fb83155de24fe9fb4c4e0d432b12fd"
)

#: θ_fusion gelé (§4.4) : zéro SÉMANTIQUE de la spec — fusion ssi coherence > 0.
THETA_FUSION: float = 0.0

#: Garde dure de terminaison (§4.4) : borne atteinte ⇒ issue I5.
MAX_FUSIONS: int = 20_000

#: Seuils de classement Torsion A (§4.6).
SEUIL_TORSION_MEDIANE: float = 0.30
SEUIL_TAUX_TRAVERSEE: float = 0.30

#: Effectif minimal de types mesurables pour prononcer la Torsion A (§4.6).
MIN_TYPES_TORSION: int = 30

#: Audit fréquence (§4.7) : seuil I6 et effectif minimal.
SEUIL_RHO_SPEARMAN: float = 0.80
MIN_FUSIONS_RHO: int = 5

#: Mur étalonné (§12) : run ≤ min(FACTEUR_MUR × t_cal, PLAFOND_DUR_S).
FACTEUR_MUR: float = 2000.0
PLAFOND_DUR_S: float = 1800.0
N_PHRASES_CAL: int = 9
N_ITERATIONS_CAL: int = 10


# ---------------------------------------------------------------------------
# σ_unit, H_norm, Coherence (formules exactes, cache par identité)
# ---------------------------------------------------------------------------

Ident = Tuple[int, ...]


def sigma_unit(ident: Ident, table: Sequence[dict]) -> np.ndarray:
    """σ_unit(u) ∈ ℝ^{4L} (float64) : concat ordonnée des 4 constantes par phonème.

    Fonction de l'identité seule (§4.1) — extension exacte du σ_syll T73.
    """
    if len(ident) == 0:
        raise ValueError("identité vide : σ_unit non définie")
    return np.asarray(
        [val for pid in ident for val in constantes_phoneme(table, int(pid))],
        dtype=np.float64,
    )


class CacheStab:
    """Cache identité → (S, H, H_norm) — exact par construction (§4.5) :
    σ_unit est une fonction de l'identité, rien à invalider."""

    def __init__(self, table: Sequence[dict]) -> None:
        self._table = table
        self._h: Dict[Ident, Tuple[float, float, float]] = {}
        self._coh: Dict[Tuple[Ident, Ident], float] = {}

    def h_norm(self, ident: Ident) -> float:
        """H_norm(σ_unit(ident)) — S = 0 est un DÉMENTI du §9 (garde mordante)."""
        ent = self._h.get(ident)
        if ent is None:
            s, h, h_norm = entropie_masse(sigma_unit(ident, self._table))
            if h_norm is None:
                raise ValueError(
                    f"S = 0 sur {ident!r} : démenti du support déclaré (§9) — "
                    "constantes de table toutes > 0 attendues"
                )
            ent = (s, float(h), float(h_norm))
            self._h[ident] = ent
        return ent[2]

    def h_brut(self, ident: Ident) -> float:
        self.h_norm(ident)
        return self._h[ident][1]

    def coherence(self, a: Ident, b: Ident) -> float:
        """Coherence(a,b) = −H_norm(σ(a·b)) + ½[H_norm(σ(a)) + H_norm(σ(b))].

        A3 (§4.2) : fonction du seul contenu phonémique de la paire — ni
        contexte, ni fréquence, ni corpus dans la signature.
        """
        cle = (a, b)
        val = self._coh.get(cle)
        if val is None:
            val = -self.h_norm(a + b) + 0.5 * (self.h_norm(a) + self.h_norm(b))
            self._coh[cle] = val
        return val


# ---------------------------------------------------------------------------
# État de corpus (types de mots phonémiques, fusion intra-mot — §4.3)
# ---------------------------------------------------------------------------

class EtatCorpus:
    """État de fusion d'un corpus : un enregistrement par TYPE de mot phonémique.

    Identité de type = tuple g0 complet du token (§4.6). Tous les tokens d'un
    même type portent la même séquence g0 ⇒ la fusion (fonction de l'identité)
    les réécrit identiquement : l'état par type + multiplicité est EXACT.

    ``types`` : OrderedDict (ordre de première apparition, déterministe)
    identité → dict {mult, unites (liste de tuples g0), g1_nb (tuple, 1ʳᵉ
    occurrence), drop}. Exclusions comptées-publiées (§4.3) : tokens muets
    (g0_len = 0) exclus ; tokens g0_len = 1 GARDÉS (aucune paire, comptés).
    """

    def __init__(self) -> None:
        self.types: "OrderedDict[Ident, dict]" = OrderedDict()
        self.comptes: Dict[str, int] = {
            "n_contextes": 0,
            "n_tokens": 0,
            "n_tokens_muets": 0,
            "n_tokens_g0_len_1": 0,
            "n_types_g1_variable": 0,  # vérification de chaîne, attendu 0
        }

    def total_unites_occurrences(self) -> int:
        """Nombre total d'unités du corpus, pondéré par les multiplicités
        (grandeur de l'invariant de décroissance stricte, §4.4)."""
        return sum(len(t["unites"]) * t["mult"] for t in self.types.values())


def construire_etat(
    tokens_par_contexte: Iterable[Sequence[dict]],
) -> EtatCorpus:
    """Construit l'état initial (unités = phonèmes, k = 0) depuis les tokens R16.

    Gardes héritées T73 : ``drop = g0_len − Σ g1_nb < 0`` ⇒ incohérence ABI,
    refus. La variabilité de g1_nb entre occurrences d'un même type est
    comptée (attendu 0 : les segmentations sont des fonctions de l'identité).
    """
    etat = EtatCorpus()
    for tokens in tokens_par_contexte:
        etat.comptes["n_contextes"] += 1
        for tok in tokens:
            etat.comptes["n_tokens"] += 1
            g0_ids = tuple(int(p) for p in tok["g0_ids"])
            g1_nb = tuple(int(n) for n in tok["g1_nb"])
            if len(g0_ids) == 0:
                etat.comptes["n_tokens_muets"] += 1
                continue
            if len(g0_ids) == 1:
                etat.comptes["n_tokens_g0_len_1"] += 1
            drop = len(g0_ids) - sum(g1_nb)
            if drop < 0:
                raise ValueError(
                    f"incohérence ABI : Σ g1_nb = {sum(g1_nb)} > g0_len = "
                    f"{len(g0_ids)} (token {tok['text']!r})"
                )
            rec = etat.types.get(g0_ids)
            if rec is None:
                etat.types[g0_ids] = {
                    "mult": 1,
                    "unites": [(p,) for p in g0_ids],
                    "g1_nb": g1_nb,
                    "drop": drop,
                }
            else:
                rec["mult"] += 1
                if rec["g1_nb"] != g1_nb:
                    etat.comptes["n_types_g1_variable"] += 1
    return etat


def enumerer_paires(etat: EtatCorpus) -> "OrderedDict[Tuple[Ident, Ident], int]":
    """Types de paires ordonnées adjacentes intra-mot, fréquences comptées (§5a).

    Fréquence = nombre d'occurrences adjacentes dans l'état courant, pondéré
    par la multiplicité des types de mots (fenêtre glissante ; le
    chevauchement n'existe que pour (a,a), jamais fusionnée par A2).
    Ordre d'énumération = première apparition (déterministe).
    """
    paires: "OrderedDict[Tuple[Ident, Ident], int]" = OrderedDict()
    for rec in etat.types.values():
        unites = rec["unites"]
        mult = rec["mult"]
        for i in range(len(unites) - 1):
            cle = (unites[i], unites[i + 1])
            paires[cle] = paires.get(cle, 0) + mult
    return paires


def appliquer_fusion(etat: EtatCorpus, a: Ident, b: Ident) -> int:
    """Fusionne (a,b) → a·b PARTOUT, balayage gauche→droite non chevauchant (§5d).

    Retourne le nombre de remplacements pondéré occurrences. Garde mordante :
    zéro remplacement ⇒ l'invariant de décroissance stricte serait violé.
    """
    c = a + b
    n_rempl = 0
    for rec in etat.types.values():
        unites = rec["unites"]
        if len(unites) < 2:
            continue
        nouv: List[Ident] = []
        i = 0
        modifie = False
        while i < len(unites):
            if i + 1 < len(unites) and unites[i] == a and unites[i + 1] == b:
                nouv.append(c)
                i += 2
                n_rempl += rec["mult"]
                modifie = True
            else:
                nouv.append(unites[i])
                i += 1
        if modifie:
            rec["unites"] = nouv
    if n_rempl == 0:
        raise ValueError(
            f"invariant de décroissance violé : fusion ({a!r}, {b!r}) sans "
            "occurrence — garde mordante (§4.4)"
        )
    return n_rempl


# ---------------------------------------------------------------------------
# Boucle de fusion (§5) — déterministe, zéro aléa
# ---------------------------------------------------------------------------

def executer_fusions(
    etat: EtatCorpus,
    cache: CacheStab,
    theta: float = THETA_FUSION,
    max_fusions: int = MAX_FUSIONS,
    mur_s: Optional[float] = None,
    t0: Optional[float] = None,
) -> Dict[str, object]:
    """Fusion itérative jusqu'à arrêt par critère, MAX_FUSIONS ou mur temporel.

    Sélection (§4.5) : Coherence maximale (float64) ; ex æquo (égalité ``==``
    exacte) tranchés par ordre lexicographique CROISSANT sur (a, b) — la plus
    petite paire gagne. Jamais de fréquence, jamais de graine.

    Retour : journal ordonné [{iteration, a, b, coherence, frequence, k}],
    motif d'arrêt (ARRET_CRITERE / I5_MAX_FUSIONS / I5_MUR_TEMPOREL),
    profondeurs k par identité créée, n_conflits_k (attendu 0).
    """
    journal: List[dict] = []
    profondeurs: Dict[Ident, int] = {}
    n_conflits_k = 0
    arret = None

    def k_de(u: Ident) -> int:
        if len(u) == 1:
            return 0
        return profondeurs[u]

    it = 0
    while True:
        if mur_s is not None and t0 is not None and (time.perf_counter() - t0) > mur_s:
            arret = "I5_MUR_TEMPOREL"
            break
        paires = enumerer_paires(etat)
        best: Optional[Tuple[Ident, Ident]] = None
        best_coh = 0.0
        for (a, b) in paires:
            coh = cache.coherence(a, b)
            if not (coh > theta):  # test STRICT (§4.4)
                continue
            if (
                best is None
                or coh > best_coh
                or (coh == best_coh and (a, b) < best)
            ):
                best, best_coh = (a, b), coh
        if best is None:
            arret = "ARRET_CRITERE"
            break
        if it >= max_fusions:
            # La borne ne mord que s'il RESTE une fusion possible (sinon
            # l'arrêt par critère ci-dessus a préséance — pas de faux I5).
            arret = "I5_MAX_FUSIONS"
            break
        a, b = best
        total_avant = etat.total_unites_occurrences()
        frequence = paires[best]
        appliquer_fusion(etat, a, b)
        total_apres = etat.total_unites_occurrences()
        if not (total_apres < total_avant):
            raise ValueError(
                f"invariant de décroissance violé à l'itération {it + 1} : "
                f"{total_avant} → {total_apres}"
            )
        c = a + b
        k_new = max(k_de(a), k_de(b)) + 1
        if c in profondeurs and profondeurs[c] != k_new:
            n_conflits_k += 1  # premier k conservé (déclaré, attendu 0)
        else:
            profondeurs.setdefault(c, k_new)
        it += 1
        journal.append(
            {
                "iteration": it,
                "a": list(a),
                "b": list(b),
                "coherence": best_coh,
                "frequence": frequence,
                "k": profondeurs[c],
                "n_remplacements_occ": total_avant - total_apres,
            }
        )

    return {
        "journal": journal,
        "n_fusions": len(journal),
        "arret": arret,
        "profondeurs": profondeurs,
        "n_conflits_k": n_conflits_k,
    }


def vitalite_premier_passage(etat: EtatCorpus, cache: CacheStab) -> Dict[str, object]:
    """C2 (§4.4) : distribution complète des Coherence au 1er passage.

    Verdict (n_pos, n_neg) en TYPES de paires (déclaré) ; pondération
    occurrences publiée en INFO. θ = 0 coupe ssi n_pos ≥ 1 ET n_neg ≥ 1.
    """
    paires = enumerer_paires(etat)
    valeurs: List[float] = []
    n_pos = n_neg = 0
    n_pos_occ = n_neg_occ = 0
    for (a, b), freq in paires.items():
        coh = cache.coherence(a, b)
        valeurs.append(coh)
        if coh > THETA_FUSION:
            n_pos += 1
            n_pos_occ += freq
        else:
            n_neg += 1
            n_neg_occ += freq
    arr = np.asarray(valeurs, dtype=np.float64)
    return {
        "n_paires_types": len(valeurs),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "coupe": bool(n_pos >= 1 and n_neg >= 1),
        "INFO_n_pos_occurrences": n_pos_occ,
        "INFO_n_neg_occurrences": n_neg_occ,
        "distribution": _quartiles_json(arr) if len(valeurs) else None,
    }


# ---------------------------------------------------------------------------
# Torsion A (§4.6) et classification des unités (§4.8)
# ---------------------------------------------------------------------------

def coupures_internes(longueurs: Sequence[int], total: int) -> FrozenSet[int]:
    """Positions de coupure INTERNES (sommes cumulées, bornes 0 et total exclues)."""
    cuts = set()
    cur = 0
    for n in longueurs:
        cur += int(n)
        if 0 < cur < total:
            cuts.add(cur)
    return frozenset(cuts)


def jaccard_distance(s1: FrozenSet[int], s2: FrozenSet[int]) -> Tuple[float, bool]:
    """Distance de Jaccard 1 − |∩|/|∪| ; DEUX ensembles vides → (0.0, True)."""
    if not s1 and not s2:
        return 0.0, True
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return 1.0 - inter / union, False


def torsion_corpus(etat: EtatCorpus) -> Dict[str, object]:
    """Torsion A + taux_traversée + classification des unités finales (§4.6, §4.8).

    Types mesurables : g0_len ≥ 2, drop = 0, g1_len ≥ 1 (≥ 1 tranche non
    vide). Chaque type compte UNE fois (dédoublonné) ; le pondéré-fréquence
    est publié en INFO. Exclusions comptées.
    """
    distances: List[float] = []
    poids: List[int] = []
    par_type: List[dict] = []
    n_double_vide = 0
    n_exclus_len1 = n_exclus_drop = n_exclus_g1_vide = 0
    n_unites_len2 = 0        # unités finales de longueur ≥ 2 (occurrences/type)
    n_unites_traversee = 0   # … dont l'empan contient ≥ 1 frontière g1 stricte
    classes = {
        "intra_syllabique": 0,
        "intra_syllabique_exacte_syllabe": 0,  # sous-cas du précédent
        "multi_syllabique_alignee": 0,
        "traversante": 0,
        "n_unites_classees": 0,
    }

    for ident, rec in etat.types.items():
        g0_len = len(ident)
        if g0_len < 2:
            n_exclus_len1 += 1
            continue
        if rec["drop"] > 0:
            n_exclus_drop += 1
            continue
        g1_nb = [n for n in rec["g1_nb"] if n > 0]
        if len(g1_nb) == 0:
            n_exclus_g1_vide += 1
            continue

        b_g1 = coupures_internes(g1_nb, g0_len)
        long_unites = [len(u) for u in rec["unites"]]
        b_g1p = coupures_internes(long_unites, g0_len)
        dist, double_vide = jaccard_distance(b_g1, b_g1p)
        if double_vide:
            n_double_vide += 1
        distances.append(dist)
        poids.append(rec["mult"])
        par_type.append(
            {
                "ident": list(ident),
                "mult": rec["mult"],
                "b_g1": sorted(b_g1),
                "b_g1_prime": sorted(b_g1p),
                "torsion_a": dist,
                "double_vide": double_vide,
            }
        )

        # Frontières syllabiques complètes (bords inclus) pour la classification.
        bords = {0, g0_len} | set(b_g1)
        bornes_syll = sorted(bords)
        debut = 0
        for lu in long_unites:
            fin = debut + lu
            internes = [x for x in b_g1 if debut < x < fin]
            classes["n_unites_classees"] += 1
            if lu >= 2:
                n_unites_len2 += 1
                if internes:
                    n_unites_traversee += 1
            if not internes:
                classes["intra_syllabique"] += 1
                # exacte-syllabe : l'empan est EXACTEMENT une syllabe.
                if debut in bords and fin in bords and (
                    bornes_syll.index(fin) - bornes_syll.index(debut) == 1
                ):
                    classes["intra_syllabique_exacte_syllabe"] += 1
            elif debut in bords and fin in bords:
                classes["multi_syllabique_alignee"] += 1
            else:
                classes["traversante"] += 1
            debut = fin

    n_types = len(distances)
    arr = np.asarray(distances, dtype=np.float64)
    w = np.asarray(poids, dtype=np.float64)
    mesurable = n_types >= MIN_TYPES_TORSION
    res: Dict[str, object] = {
        "n_types_mesurables": n_types,
        "n_double_vide": n_double_vide,
        "exclusions": {
            "g0_len_1": n_exclus_len1,
            "drop_positif": n_exclus_drop,
            "g1_vide": n_exclus_g1_vide,
        },
        "mesurable": mesurable,
        "torsion_mediane": float(np.median(arr)) if n_types else None,
        "INFO_torsion_moyenne": float(arr.mean()) if n_types else None,
        "INFO_quartiles": _quartiles_json(arr) if n_types else None,
        "INFO_pondere_frequence": {
            "moyenne": float((arr * w).sum() / w.sum()),
            "mediane_ponderee": _mediane_ponderee(arr, w),
        }
        if n_types
        else None,
        "taux_traversee": (n_unites_traversee / n_unites_len2)
        if n_unites_len2
        else None,
        "n_unites_len2plus": n_unites_len2,
        "n_unites_traversee": n_unites_traversee,
        "classification_unites": classes,
        "par_type": par_type,
    }
    # Classement (liste close §4.6) — prononcé seulement si mesurable.
    if not mesurable:
        res["classement"] = "NON-MESURE"
    elif res["torsion_mediane"] <= SEUIL_TORSION_MEDIANE:
        res["classement"] = "RETROUVE"
    elif res["taux_traversee"] is not None and res["taux_traversee"] >= SEUIL_TAUX_TRAVERSEE:
        res["classement"] = "TRAVERSE"
    else:
        res["classement"] = "IGNORE"
    return res


def _mediane_ponderee(valeurs: np.ndarray, poids: np.ndarray) -> float:
    """Médiane pondérée (INFO) : plus petite valeur dont la masse cumulée ≥ ½."""
    ordre = np.argsort(valeurs, kind="stable")
    v, p = valeurs[ordre], poids[ordre]
    cum = np.cumsum(p)
    idx = int(np.searchsorted(cum, 0.5 * float(p.sum())))
    return float(v[min(idx, len(v) - 1)])


# ---------------------------------------------------------------------------
# Audit fréquence (§4.7) — Spearman numpy lisible, zéro scipy
# ---------------------------------------------------------------------------

def rangs_moyens(valeurs: Sequence[float]) -> np.ndarray:
    """Rangs moyens (1-based) avec moyenne des rangs pour ex æquo exacts (==)."""
    arr = np.asarray(valeurs, dtype=np.float64)
    ordre = np.argsort(arr, kind="stable")
    rangs = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[ordre[j + 1]] == arr[ordre[i]]:
            j += 1
        rang_moyen = 0.5 * (i + j) + 1.0  # moyenne des rangs 1-based i+1..j+1
        for m in range(i, j + 1):
            rangs[ordre[m]] = rang_moyen
        i = j + 1
    return rangs


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """ρ de Spearman = corrélation de Pearson des rangs moyens.

    Retourne None si n < MIN_FUSIONS_RHO (NON-MESURE déclarée §4.7) ou si un
    des deux classements est constant (variance nulle — DEGENERE déclaré).
    """
    if len(x) != len(y):
        raise ValueError(f"longueurs différentes : {len(x)} vs {len(y)}")
    if len(x) < MIN_FUSIONS_RHO:
        return None
    rx, ry = rangs_moyens(x), rangs_moyens(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if den == 0.0:
        return None
    return float((dx * dy).sum() / den)


def audit_frequence(journal: Sequence[dict]) -> Dict[str, object]:
    """Audit §4.7 : ρ entre Coherence à la sélection et fréquence à la sélection."""
    cohs = [e["coherence"] for e in journal]
    freqs = [float(e["frequence"]) for e in journal]
    rho = spearman_rho(cohs, freqs)
    res: Dict[str, object] = {
        "n_fusions": len(journal),
        "rho_spearman": rho,
        "seuil_i6": SEUIL_RHO_SPEARMAN,
        "classements": {
            "coherence_a_la_selection": cohs,
            "frequence_a_la_selection": freqs,
        },
    }
    if rho is None:
        res["statut"] = (
            "NON-MESURE (n < 5)" if len(journal) < MIN_FUSIONS_RHO else "DEGENERE (rangs constants)"
        )
    else:
        res["statut"] = "I6" if abs(rho) >= SEUIL_RHO_SPEARMAN else "OK"
    return res


# ---------------------------------------------------------------------------
# Vocabulaire émergent (§4.8)
# ---------------------------------------------------------------------------

def vocabulaire_final(
    etat: EtatCorpus, profondeurs: Dict[Ident, int], table: Sequence[dict]
) -> Dict[str, object]:
    """Types d'unités présents dans la segmentation finale + distribution de k.

    P-4 (verdict) : médiane de k sur les TYPES FUSIONNÉS finaux (k ≥ 1) ;
    occurrences en INFO. ``rendu_ascii`` : lecture humaine dérivée de la table
    (INFO, aucune grandeur).
    """
    unites: "OrderedDict[Ident, Dict[str, int]]" = OrderedDict()
    for rec in etat.types.values():
        for u in rec["unites"]:
            slot = unites.setdefault(u, {"n_types_mots": 0, "n_occurrences": 0})
            slot["n_types_mots"] += 1
            slot["n_occurrences"] += rec["mult"]

    def k_de(u: Ident) -> int:
        return 0 if len(u) == 1 else profondeurs[u]

    entrees = [
        {
            "ident": list(u),
            "rendu_ascii": "".join(str(table[p]["ascii"]) for p in u),
            "longueur": len(u),
            "k": k_de(u),
            "n_types_mots": c["n_types_mots"],
            "n_occurrences": c["n_occurrences"],
        }
        for u, c in unites.items()
    ]
    ks_fusionnes_types = sorted(e["k"] for e in entrees if e["k"] >= 1)
    ks_fusionnes_occ: List[int] = []
    for e in entrees:
        if e["k"] >= 1:
            ks_fusionnes_occ.extend([e["k"]] * e["n_occurrences"])
    dist_k: Dict[str, int] = {}
    for e in entrees:
        dist_k[str(e["k"])] = dist_k.get(str(e["k"]), 0) + 1
    return {
        "taille_vocabulaire": len(entrees),
        "n_types_fusionnes": len(ks_fusionnes_types),
        "distribution_k_types": dict(sorted(dist_k.items(), key=lambda kv: int(kv[0]))),
        "mediane_k_types_fusionnes": (
            float(np.median(ks_fusionnes_types)) if ks_fusionnes_types else None
        ),
        "INFO_mediane_k_occurrences_fusionnees": (
            float(np.median(ks_fusionnes_occ)) if ks_fusionnes_occ else None
        ),
        "unites": entrees,
    }


def part_fusion_totale(etat: EtatCorpus) -> Dict[str, object]:
    """P-1 : part des types de mots à g0_len ≥ 2 réduits à UNE unité finale."""
    n_elig = n_reduits = 0
    for ident, rec in etat.types.items():
        if len(ident) >= 2:
            n_elig += 1
            if len(rec["unites"]) == 1:
                n_reduits += 1
    return {
        "n_types_g0len2plus": n_elig,
        "n_reduits_a_une_unite": n_reduits,
        "part": (n_reduits / n_elig) if n_elig else None,
    }


# ---------------------------------------------------------------------------
# Quartiles JSON (aide locale — np.quantile, comme stab_grains)
# ---------------------------------------------------------------------------

def _quartiles_json(arr: np.ndarray) -> Dict[str, float]:
    q25, q50, q75 = (float(x) for x in np.quantile(arr, [0.25, 0.5, 0.75]))
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "q25": q25,
        "mediane": q50,
        "q75": q75,
        "max": float(arr.max()),
    }


# ---------------------------------------------------------------------------
# PORTÉES (§9) — deux branches par instrument, témoins construits
# ---------------------------------------------------------------------------

#: Mini-table forcée pour les témoins (valeurs choisies à la main, dyadiques
#: quand l'exactitude float64 le permet). Indices : 0 = uniforme bas (1,1,1,1),
#: 1 = uniforme haut (8,8,8,8), 2 = concentré (8,1,1,1), 3 = uniforme (2,2,2,2).
TABLE_TEMOIN: List[dict] = [
    {"flux": 1.0, "structure": 1.0, "energie_totale": 1.0, "impedance": 1.0, "ascii": "a"},
    {"flux": 8.0, "structure": 8.0, "energie_totale": 8.0, "impedance": 8.0, "ascii": "b"},
    {"flux": 8.0, "structure": 1.0, "energie_totale": 1.0, "impedance": 1.0, "ascii": "c"},
    {"flux": 2.0, "structure": 2.0, "energie_totale": 2.0, "impedance": 2.0, "ascii": "d"},
]


def portees_instruments(table_reelle: Optional[Sequence[dict]] = None) -> Dict[str, object]:
    """PORTÉE de chaque instrument du tour (§9) — publiée AVANT toute mesure.

    Chaque branche est ATTEINTE sur témoins construits ; si une branche est
    vide sur le réel, c'est un FAIT publié (les témoins forcés suffisent à
    l'opposabilité). Gap sous shuffle : N-A déclaré — la Torsion A ne teste
    pas un ordre A→B→A′ (règle T22/T60 satisfaite par absence).
    """
    cache = CacheStab(TABLE_TEMOIN)
    res: Dict[str, object] = {"gap_shuffle": "N-A déclaré (§9) : aucun verdict d'ordre A→B→A′ ce tour"}

    # --- Coherence : deux branches sur témoins FORCÉS -----------------------
    coh_pos = cache.coherence((0,), (1,))   # uniformes de niveaux distincts ⇒ > 0 (A1/A2)
    coh_neg = cache.coherence((2,), (2,))   # auto-paire non uniforme ⇒ < 0 (A2)
    coh_zero = cache.coherence((3,), (3,))  # auto-paire UNIFORME ⇒ = 0 exact (A2, égalité)
    res["coherence"] = {
        "temoin_positif": {"paire": "(0,)+(1,)", "valeur": coh_pos, "atteint": coh_pos > 0.0},
        "temoin_negatif": {"paire": "(2,)+(2,)", "valeur": coh_neg, "atteint": coh_neg < 0.0},
        "temoin_A2_egalite": {"paire": "(3,)+(3,)", "valeur": coh_zero, "exactement_zero": coh_zero == 0.0},
    }

    # --- Propriétés analytiques A1/A2/A3 sur témoins ------------------------
    # A1 exact (dyadique) : a=(0,), b=(3,) — masses 4×1 et 4×2… non : pour
    # l'exactitude on prend deux unités de MÊME masse totale : (0,) et (0,)
    # via identités distinctes n'existe pas dans la table témoin ⇒ on prend
    # a = b = (0,) : w = 1/2, H2(1/2) = 1 exact.
    h_a = cache.h_brut((0,))
    h_aa = cache.h_brut((0, 0))
    res["propriete_A1"] = {
        "temoin": "H_brut((0,0)) == 0.5*H(0)+0.5*H(0)+H2(0.5), dyadique exact",
        "gauche": h_aa,
        "droite": 0.5 * h_a + 0.5 * h_a + 1.0,
        "exact": h_aa == 0.5 * h_a + 0.5 * h_a + 1.0,
    }
    res["propriete_A2"] = {
        "non_uniforme_strictement_negatif": coh_neg < 0.0,
        "uniforme_exactement_zero": coh_zero == 0.0,
        "jamais_fusionnee_avec_theta_0_strict": not (coh_zero > THETA_FUSION),
    }
    res["propriete_A3"] = (
        "vérifiée par construction : CacheStab.coherence(a, b) n'a ni corpus, "
        "ni contexte, ni fréquence dans sa signature"
    )

    # --- Jaccard : bornes 0 et 1 + double-vide ------------------------------
    d0, _ = jaccard_distance(frozenset({1, 2}), frozenset({1, 2}))
    d1, _ = jaccard_distance(frozenset({1}), frozenset({2}))
    dv, flag = jaccard_distance(frozenset(), frozenset())
    res["jaccard"] = {
        "borne_0": d0 == 0.0,
        "borne_1": d1 == 1.0,
        "double_vide_vaut_0": dv == 0.0 and flag,
    }

    # --- Garde de terminaison MORDANTE (cas construits) ---------------------
    def _mini_etat() -> EtatCorpus:
        # deux mots-témoins portant DEUX paires fusionnables distinctes :
        # (0,1) et (0,3) — coherence((0,),(1,)) > 0 et coherence((0,),(3,)) > 0
        # (uniformes de niveaux distincts, A1/A2) ⇒ épuisement en 2 fusions.
        tokens = [
            [
                {"text": "ab", "g0_ids": [0, 1], "g1_nb": [2]},
                {"text": "ad", "g0_ids": [0, 3], "g1_nb": [2]},
            ]
        ]
        return construire_etat(tokens)

    r_bas = executer_fusions(_mini_etat(), CacheStab(TABLE_TEMOIN), max_fusions=1)
    res["garde_max_fusions"] = {
        "mordante": r_bas["arret"] == "I5_MAX_FUSIONS",
        "arret": r_bas["arret"],
        "n_fusions": r_bas["n_fusions"],
    }
    try:
        appliquer_fusion(_mini_etat(), (2,), (3,))  # paire absente ⇒ garde
        inv_mord = False
    except ValueError:
        inv_mord = True
    res["garde_invariant_decroissance"] = {"mordante_par_injection": inv_mord}
    r_mur = executer_fusions(
        _mini_etat(), CacheStab(TABLE_TEMOIN), mur_s=0.0, t0=time.perf_counter() - 1.0
    )
    res["garde_mur_temporel"] = {"mordante": r_mur["arret"] == "I5_MUR_TEMPOREL"}

    # --- Spearman : ±1, ≈0, NON-MESURE --------------------------------------
    rho_p = spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0])
    rho_m = spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [50.0, 40.0, 30.0, 20.0, 10.0])
    rho_0 = spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0], [30.0, 10.0, 50.0, 20.0, 40.0])
    res["spearman"] = {
        "plus_un_exact": rho_p == 1.0,
        "moins_un_exact": rho_m == -1.0,
        "cas_median": rho_0,
        "non_mesure_n_inferieur_5": spearman_rho([1.0], [1.0]) is None,
    }

    # --- Support réel (si table fournie) ------------------------------------
    if table_reelle is not None:
        cache_r = CacheStab(table_reelle)
        n_pos = n_neg = 0
        for i in range(len(table_reelle)):
            for j in range(len(table_reelle)):
                if cache_r.coherence((i,), (j,)) > 0.0:
                    n_pos += 1
                else:
                    n_neg += 1
        h_norms = sorted({cache_r.h_norm((i,)) for i in range(len(table_reelle))})
        min_const = min(
            float(e[c]) for e in table_reelle for c in CHAMPS_CONSTANTES
        )
        res["support_reel"] = {
            "n_paires_g0_reelles": n_pos + n_neg,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "deux_branches_sur_le_reel": n_pos >= 1 and n_neg >= 1,
            "min_constante_table": min_const,
            "s_zero_impossible": min_const > 0.0,
            "n_distinct_h_norm_g0": len(h_norms),
        }

    return res


# ---------------------------------------------------------------------------
# Étalonnage du mur (§12)
# ---------------------------------------------------------------------------

def etalonner_t75(chemin_eve: str) -> Dict[str, object]:
    """t_cal = garde sha + tokenisation + paires + 10 itérations de fusion
    sur les 9 premières phrases d'eve. Mur = min(2000 × t_cal, 1800 s)."""
    ctx, _ = contextes_eve(chemin_eve)
    t0 = time.perf_counter()
    tok, _prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
    table = table_miroir()
    tokens = [tok.tokenize_sequences(c) for c in ctx[:N_PHRASES_CAL]]
    etat = construire_etat(tokens)
    cache = CacheStab(table)
    r = executer_fusions(etat, cache, max_fusions=N_ITERATIONS_CAL)
    t_cal = time.perf_counter() - t0
    return {
        "n_phrases": N_PHRASES_CAL,
        "n_tokens": etat.comptes["n_tokens"],
        "n_iterations_cal": N_ITERATIONS_CAL,
        "n_fusions_cal": r["n_fusions"],
        "arret_cal": r["arret"],
        "t_cal_s": t_cal,
        "facteur_mur": FACTEUR_MUR,
        "mur_s": min(FACTEUR_MUR * t_cal, PLAFOND_DUR_S),
        "plafond_dur_s": PLAFOND_DUR_S,
    }


# ---------------------------------------------------------------------------
# Pilote (artefacts TOUR75_* — racine écosystème)
# ---------------------------------------------------------------------------

def _md5(chemin: str) -> str:
    with open(chemin, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _config_echo(md5_eve: Optional[str], md5_claude: Optional[str],
                 so_sha256: Optional[str]) -> Dict[str, object]:
    """Écho de configuration complet (doctrine T74 : contexte déclaré).

    Volontairement SANS horodatage : les artefacts de verdict doivent être
    reproductibles AU BIT (C1)."""
    return {
        "tour": 75,
        "theta_fusion": THETA_FUSION,
        "max_fusions": MAX_FUSIONS,
        "seuil_torsion_mediane": SEUIL_TORSION_MEDIANE,
        "seuil_taux_traversee": SEUIL_TAUX_TRAVERSEE,
        "min_types_torsion": MIN_TYPES_TORSION,
        "seuil_rho_spearman": SEUIL_RHO_SPEARMAN,
        "min_fusions_rho": MIN_FUSIONS_RHO,
        "so_sha256": so_sha256,
        "so_sha256_reference_t75": SHA256_SO_REFERENCE_T75,
        "md5_corpus_eve": md5_eve,
        "md5_corpus_claude_aba": md5_claude,
        "champs_constantes": list(CHAMPS_CONSTANTES),
        "contexte_execution": {
            "python": f"{platform.python_implementation()} {platform.python_version()}",
            "plateforme": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
    }


def _ecrire_json(chemin: str, obj: Dict[str, object]) -> None:
    with open(chemin, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def run_complet(
    chemin_eve: str,
    chemin_claude: str,
    mur_s: Optional[float],
    dossier_sortie: str,
) -> Dict[str, str]:
    """Le run principal (§5) : C2 claude_aba (1er passage) PUIS run eve complet.

    Écrit TOUR75_VOCAB.json, TOUR75_TORSION.json, TOUR75_AUDIT_FREQ.json dans
    ``dossier_sortie``. Retourne {nom → chemin}. Si C2 ne coupe pas : clause
    gelée I3 — distributions publiées, AUCUN run de fusion (pas de repli).
    """
    import os

    tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
    table = table_miroir()
    md5_eve, md5_claude = _md5(chemin_eve), _md5(chemin_claude)
    config = _config_echo(md5_eve, md5_claude, prov["native_lib_sha256"])

    # --- C2 : vitalité sur claude_aba (1er passage, rang instrument) --------
    ctx_c, cpt_c = contextes_claude_aba(chemin_claude)
    etat_c = construire_etat(tok.tokenize_sequences(c) for c in ctx_c)
    cache = CacheStab(table)
    vital_c = vitalite_premier_passage(etat_c, cache)

    # INFO (§6) : comparaison inter-corpus des formes de distributions —
    # 1er passage eve, jamais une confirmation.
    ctx_e, cpt_e = contextes_eve(chemin_eve)
    etat_e = construire_etat(tok.tokenize_sequences(c) for c in ctx_e)
    vital_e_info = vitalite_premier_passage(etat_e, cache)

    vocab_obj: Dict[str, object] = {
        "config": config,
        "contextes": {"claude_aba": cpt_c, "eve": cpt_e},
        "comptes_etat": {"claude_aba": dict(etat_c.comptes), "eve": dict(etat_e.comptes)},
        "vitalite_C2_claude_aba": vital_c,
        "INFO_premier_passage_eve": vital_e_info,
    }

    chemins: Dict[str, str] = {}
    if not vital_c["coupe"]:
        # Clause gelée (§4.4 / I3) : critère dégénéré au grain G₀ — aucun
        # repli sur un quantile, publication des distributions seules.
        # Les artefacts Torsion/audit existent avec NON-MESURE explicite
        # (aucune cellule vide silencieuse) ; C1 reste exigible sur le tout.
        issue = "I3_CRITERE_DEGENERE (C2 ne coupe pas : n_pos = 0 ou n_neg = 0)"
        vocab_obj["issue"] = issue
        p = os.path.join(dossier_sortie, "TOUR75_VOCAB.json")
        _ecrire_json(p, vocab_obj)
        chemins["TOUR75_VOCAB.json"] = p
        p = os.path.join(dossier_sortie, "TOUR75_TORSION.json")
        _ecrire_json(p, {
            "config": config,
            "issue": issue,
            "torsion_a": "NON-MESURE : aucun run de fusion (clause gelée I3, "
                         "émission §4.4/§10) — aucun grain produit à juger",
        })
        chemins["TOUR75_TORSION.json"] = p
        p = os.path.join(dossier_sortie, "TOUR75_AUDIT_FREQ.json")
        _ecrire_json(p, {
            "config": config,
            "issue": issue,
            "audit_frequence": "NON-MESURE : n_fusions = 0 < 5 (émission §4.7)",
        })
        chemins["TOUR75_AUDIT_FREQ.json"] = p
        return chemins

    # --- Run eve : fusion itérative complète --------------------------------
    t0 = time.perf_counter()
    r = executer_fusions(etat_e, cache, mur_s=mur_s, t0=t0)
    vocab_obj["run_eve"] = {
        "arret": r["arret"],
        "n_fusions": r["n_fusions"],
        "n_conflits_k": r["n_conflits_k"],
        "journal": r["journal"],
    }
    vocab_obj["vocabulaire"] = vocabulaire_final(etat_e, r["profondeurs"], table)
    vocab_obj["p1_fusion_totale"] = part_fusion_totale(etat_e)

    p = os.path.join(dossier_sortie, "TOUR75_VOCAB.json")
    _ecrire_json(p, vocab_obj)
    chemins["TOUR75_VOCAB.json"] = p

    torsion_obj = {"config": config, "torsion_a": torsion_corpus(etat_e)}
    p = os.path.join(dossier_sortie, "TOUR75_TORSION.json")
    _ecrire_json(p, torsion_obj)
    chemins["TOUR75_TORSION.json"] = p

    audit_obj = {"config": config, "audit_frequence": audit_frequence(r["journal"])}
    p = os.path.join(dossier_sortie, "TOUR75_AUDIT_FREQ.json")
    _ecrire_json(p, audit_obj)
    chemins["TOUR75_AUDIT_FREQ.json"] = p
    return chemins


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stade", required=True, choices=["portees", "etalonnage", "run"])
    ap.add_argument("--eve", default=None, help="chemin de corpus_eve_clean.txt")
    ap.add_argument("--claude-aba", default=None, help="chemin de corpus_claude_aba.txt")
    ap.add_argument("--sortie", required=True, help="fichier JSON (portees/etalonnage) ou dossier (run)")
    ap.add_argument("--mur-s", type=float, default=None, help="mur du run (depuis TOUR75_ETALONNAGE.json)")
    args = ap.parse_args(argv)

    if args.stade == "portees":
        # La table réelle passe par la garde sha (aucune mesure sans gel .so).
        _tok, prov = charger_tokenizer_garde(sha256_attendu=SHA256_SO_REFERENCE_T75)
        obj = {
            "config": _config_echo(None, None, prov["native_lib_sha256"]),
            "portees": portees_instruments(table_miroir()),
        }
        # Branche mismatch de la garde sha, prouvée mordante (§9).
        try:
            charger_tokenizer_garde(sha256_attendu="0" * 64)
            obj["garde_sha_so"] = {"mordante": False}
        except RuntimeError:
            obj["garde_sha_so"] = {
                "mordante": True,
                "clause": "1 mismatch => 0 mesure, tour NON-MESURE (§9)",
            }
        _ecrire_json(args.sortie, obj)
    elif args.stade == "etalonnage":
        _ecrire_json(args.sortie, {"tour": 75, "etalonnage": etalonner_t75(args.eve)})
    else:
        chemins = run_complet(args.eve, args.claude_aba, args.mur_s, args.sortie)
        for nom, chemin in chemins.items():
            print(f"{nom}: {chemin}")


if __name__ == "__main__":  # pragma: no cover - pilote
    main()
