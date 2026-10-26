"""Stab par grain — instrument descriptif du Tour 73 (« Stab seul, descriptif d'abord »).

Ce module mesure, PAR GRAIN (G₂ mot, g1 syllabe, G₀ phonème) et PAR CORPUS,
les deux composantes de la future mesure Stab de la spec
``brainstorming/GRANULARITE_RECURSIVE_SPEC_v0.3.1.md`` :

- **H(σ)** : entropie de Shannon de la répartition de la masse absolue d'une
  signature sur ses dims (tranche gelée par grain). H≈0 = signature concentrée
  (Dirac) ; H = log2|T| = masse uniformément répartie. Grandeur publiée :
  ``H_norm = H / log2 |T|`` ∈ [0, 1], comparable entre grains.
- **Var_ctx(u)** : variance de population (ddof=0, float64) des occurrences
  d'une unité sur les dims de contexte, moyennée sur les dims. « > 0 » est un
  strict float64 SANS tolérance : le zéro exact est atteignable (float32
  identiques) et significatif.

AUCUNE fusion, AUCUN θ_fusion, AUCUN λ : les deux composantes sont publiées
séparément (émission T73 §2) — la calibration est un tour futur.

Définitions opérationnelles gelées (émission T73, valeurs par défaut du tour) :

- Tranche H au grain mot : dims MAIN = {6..22, 31, 32} (19 dims, gel T42/T45,
  référence ``diagnostics/textual_return_dynamics.py``). Dims 0-5 exclues
  (étiquettes ABA), 23-27 exclues (hors MAIN), 28-30 réservées à Var_ctx.
- Var_ctx au grain mot : dims {28, 29, 30}, D = 3.
- Syllabe (g1) : PAS de 33D propre (fait d'interface R16/T72). Signature =
  concaténation ordonnée, par phonème de sa tranche g0, des 4 constantes de
  table (flux, structure, energie_totale, impedance) lues dans le miroir
  Python ``phoneme_table_fr.PHONEME_TABLE_FR`` (le ``.so`` n'est jamais seule
  source de vérité). Identité = tuple exact d'ids g0. Conséquence déclarée :
  σ_syll est une fonction de l'identité ⇒ Var_ctx(syllabe) ≡ 0 par
  construction — publié comme vérification de chaîne (P-1b), jamais comme
  mesure de contexte.
- G₀ : identité = id de table (0-35) ; σ = les 4 constantes du type. H(G₀) est
  une INFO (R43 : constante de table), jamais « une entropie qui mesurerait ».
- Contexte = la phrase. eve : une ligne ``strip()`` non vide ; corpus ABA :
  les 3 textes de segments de chaque cycle via le parseur de référence
  ``spiraton.data.aba`` (76 cycles → 228 contextes attendus).
- Frontière de régime H62 (spec §8, précurseur ``regime_grain.py`` T62) :
  une distribution de Var_ctx est CREUSE à un grain ssi
  ``part_mesurable < 0,50`` OU ``n_mesurables < 30`` ; sinon ESTIMABLE.
  Toute valeur publiée porte sa déclaration de côté.

Sélecteurs (CONTRAT T54) : ``n_occ(u) >= 2`` pour Var_ctx et ``S > 0`` pour H
— fonctions du seul corpus, invariantes par permutation des valeurs mesurées ;
comptes publiés des deux côtés. Aucune autre sélection.

Les verdicts vivent dans les artefacts ``TOUR73_*.json`` (racine écosystème),
pas dans ce module : ici, uniquement l'instrument et son pilote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constantes gelées du tour (documentées ; paramètres par défaut, jamais cachés)
# ---------------------------------------------------------------------------

#: Tranche MAIN gelée T42/T45 pour H au grain mot : 19 dims physiques.
MAIN_DIMS: Tuple[int, ...] = tuple(range(6, 23)) + (31, 32)

#: Dims de contexte de phrase pour Var_ctx au grain mot (D = 3).
CTX_DIMS: Tuple[int, ...] = (28, 29, 30)

#: Les 4 constantes de table par phonème (ordre gelé, émission §4.3).
CHAMPS_CONSTANTES: Tuple[str, ...] = ("flux", "structure", "energie_totale", "impedance")

#: Frontière de régime H62 (émission §5) : CREUX ssi part < 0,50 OU n < 30.
SEUIL_PART_MESURABLE: float = 0.50
SEUIL_N_MESURABLES: int = 30

#: Seuils de non-dégénérescence de H (émission §7, évalués sur H_norm).
SEUIL_H_N_DISTINCTS: int = 16
SEUIL_H_IQR: float = 0.01

#: Contrôle par permutation (émission §7, INFO hors-verdict) : graine et R gelés.
GRAINE_PERMUTATION: int = 73201
R_PERMUTATION: int = 100

#: sha256 de référence du .so (gel T72) — vérifié avant tout appel ctypes.
SHA256_SO_REFERENCE: str = (
    "3105131a1706c170836fc63abcb11ef731e0f0a073e188e395d1a5b23b71f2a6"
)


# ---------------------------------------------------------------------------
# Les deux composantes — formules exactes (émission §4.2 et §4.5)
# ---------------------------------------------------------------------------

def entropie_masse(sigma: Sequence[float]) -> Tuple[float, Optional[float], Optional[float]]:
    """H de la répartition de masse d'une signature (émission §4.2, float64).

    Retourne ``(S, H, H_norm)`` avec ``S = Σ|σ_i|``. Si ``S == 0.0`` la
    signature est à support dégénéré : H est NON DÉFINIE → ``(0.0, None,
    None)`` ; l'appelant compte l'unité DEGENERE-SUPPORT (jamais de NaN).
    Les valeurs négatives (spins −1) passent par ``|·|`` — déclaré.

    Bornes : ``H ∈ [0, log2 |T|]``, ``H_norm ∈ [0, 1]``. Tranche vide refusée
    (garde mordante).
    """
    arr = np.asarray(sigma, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"tranche vide ou non 1-D refusée : shape={arr.shape}")
    a = np.abs(arr)
    s = float(a.sum())
    if s == 0.0:
        return 0.0, None, None
    p = a / s
    p = p[p > 0.0]
    h = float(-(p * np.log2(p)).sum())
    h_norm = h / math.log2(arr.size)
    return s, h, h_norm


def var_ctx(occurrences: Sequence[Sequence[float]]) -> float:
    """Var_ctx d'une unité (émission §4.5) : ``(1/D) Σ_d Var_pop(σ_d)``.

    ``occurrences`` : matrice (n, D) des occurrences (n ≥ 2 requis — garde
    mordante), en float64, ddof=0. « > 0 » se lit STRICT sans tolérance :
    des float32 identiques donnent exactement 0.0.

    PORTÉE (C3) : détecte la dispersion inter-occurrences ; AVEUGLE à l'ordre
    des occurrences (une permutation des lignes ne change pas la valeur).
    """
    mat = np.asarray(occurrences, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] == 0:
        raise ValueError(f"matrice (n, D) attendue, D >= 1 : shape={mat.shape}")
    if mat.shape[0] < 2:
        raise ValueError(f"n_occ = {mat.shape[0]} < 2 : unité creuse, Var_ctx non définie")
    return float(mat.var(axis=0, ddof=0).mean())


# ---------------------------------------------------------------------------
# Contextes (émission §4.4) — le contexte est la phrase
# ---------------------------------------------------------------------------

def contextes_eve(chemin: str) -> Tuple[List[str], Dict[str, int]]:
    """eve : une phrase par ligne ``strip()`` non vide ; vides ignorées et comptées."""
    contextes: List[str] = []
    n_vides = 0
    with open(chemin, "r", encoding="utf-8") as fh:
        for ligne in fh:
            s = ligne.strip()
            if s:
                contextes.append(s)
            else:
                n_vides += 1
    return contextes, {"n_contextes": len(contextes), "n_lignes_vides": n_vides}


def contextes_claude_aba(chemin: str) -> Tuple[List[str], Dict[str, int]]:
    """Corpus ABA : route gelée = parseur de référence ``spiraton.data.aba``.

    Chaque cycle fournit 3 contextes (les textes de SEG_A, SEG_B,
    SEG_A_PRIME), tokenisés séparément. 76 cycles → 228 contextes attendus
    (compte publié, vérifié par le pilote).
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
            contextes.append(cycle.seg_a.text)
            contextes.append(cycle.seg_b.text)
            contextes.append(cycle.seg_a_prime.text)
    return contextes, {
        "n_cycles": n_cycles,
        "n_contextes": len(contextes),
        "n_lignes_non_cycle": n_non_cycles,
    }


# ---------------------------------------------------------------------------
# Extraction des occurrences par grain (interface R16/T72)
# ---------------------------------------------------------------------------

def constantes_phoneme(table: Sequence[dict], pid: int) -> Tuple[float, float, float, float]:
    """Les 4 constantes de table du phonème ``pid`` (ordre gelé CHAMPS_CONSTANTES)."""
    entree = table[pid]
    return tuple(float(entree[c]) for c in CHAMPS_CONSTANTES)  # type: ignore[return-value]


def extraire_occurrences(
    tokens_par_contexte: Iterable[Sequence[dict]],
    table: Sequence[dict],
) -> Dict[str, object]:
    """Occurrences par grain depuis les tokens ``tokenize_sequences`` (R16).

    Retour : dict avec, par grain, un ``OrderedDict identité → occurrences``
    (ordre = première apparition dans le corpus, déterministe) et les comptes
    d'exclusion déclarés :

    - ``mot``    : identité = ``text.lower()`` (UTF-8 déjà décodé par le
      wrapper, accents conservés, aucun autre nettoyage) ; occurrence =
      le vecteur 33D complet (float32 natif) de ce token dans son contexte.
    - ``syllabe``: identité = tuple exact d'ids g0 de la tranche ; occurrence
      = σ_syll (concat des 4 constantes de table par phonème, float64).
      Tokens à ``drop = g0_len − Σ g1_nb > 0`` : EXCLUS du grain et comptés
      (attendu 0, C4 T72). ``drop < 0`` : incohérence ABI, refus (garde).
      Syllabe à tranche vide (``g1_nb == 0``) : EXCLUE et comptée.
    - ``g0``     : identité = id de table (0-35) ; occurrence = les 4
      constantes du type (recopie par occurrence, support de P-1).
    """
    mots: "OrderedDict[str, List[np.ndarray]]" = OrderedDict()
    syllabes: "OrderedDict[Tuple[int, ...], List[Tuple[float, ...]]]" = OrderedDict()
    g0: "OrderedDict[int, List[Tuple[float, float, float, float]]]" = OrderedDict()

    n_contextes = 0
    n_tokens = 0
    n_tokens_muets = 0        # g0_len == 0 (légitime, wrapper T72)
    n_tokens_drop = 0         # drop > 0 → exclus du grain syllabe, comptés
    n_syllabes_tranche_vide = 0
    n_syllabes = 0
    n_occ_g0 = 0

    for tokens in tokens_par_contexte:
        n_contextes += 1
        for tok in tokens:
            n_tokens += 1
            # --- grain mot -------------------------------------------------
            ident_mot = str(tok["text"]).lower()
            v33 = np.asarray(tok["vector33d"], dtype=np.float32)
            mots.setdefault(ident_mot, []).append(v33)

            # --- grains g0 / syllabe --------------------------------------
            g0_ids = list(tok["g0_ids"])
            g1_nb = list(tok["g1_nb"])
            if len(g0_ids) == 0:
                n_tokens_muets += 1
                continue
            for pid in g0_ids:
                g0.setdefault(int(pid), []).append(constantes_phoneme(table, int(pid)))
                n_occ_g0 += 1

            somme_nb = int(sum(g1_nb))
            drop = len(g0_ids) - somme_nb
            if drop < 0:
                raise ValueError(
                    f"incohérence ABI : Σ g1_nb = {somme_nb} > g0_len = {len(g0_ids)} "
                    f"(token {tok['text']!r})"
                )
            if drop > 0:
                n_tokens_drop += 1
                continue  # exclu du grain syllabe, compté
            cur = 0
            for nb in g1_nb:
                nb = int(nb)
                if nb == 0:
                    n_syllabes_tranche_vide += 1
                    continue
                tranche = tuple(int(p) for p in g0_ids[cur:cur + nb])
                cur += nb
                sig = tuple(
                    val
                    for pid in tranche
                    for val in constantes_phoneme(table, pid)
                )
                syllabes.setdefault(tranche, []).append(sig)
                n_syllabes += 1

    return {
        "mot": mots,
        "syllabe": syllabes,
        "g0": g0,
        "comptes": {
            "n_contextes": n_contextes,
            "n_tokens": n_tokens,
            "n_tokens_muets": n_tokens_muets,
            "n_tokens_drop_sylla": n_tokens_drop,
            "n_syllabes_tranche_vide": n_syllabes_tranche_vide,
            "n_occ_syllabes": n_syllabes,
            "n_occ_g0": n_occ_g0,
        },
    }


# ---------------------------------------------------------------------------
# Statistiques par grain — H et Var_ctx, avec régime H62 déclaré
# ---------------------------------------------------------------------------

def _quartiles(valeurs: Sequence[float]) -> Dict[str, float]:
    """min/q25/mediane/q75/max (np.quantile, interpolation linéaire par défaut)."""
    arr = np.asarray(valeurs, dtype=np.float64)
    q25, q50, q75 = (float(x) for x in np.quantile(arr, [0.25, 0.5, 0.75]))
    return {
        "min": float(arr.min()),
        "q25": q25,
        "mediane": q50,
        "q75": q75,
        "max": float(arr.max()),
        "iqr": q75 - q25,
    }


def stats_h(unites: "OrderedDict[object, List[Sequence[float]]]",
            tranche_mot: Optional[Tuple[int, ...]] = None) -> Dict[str, object]:
    """Distribution de H_norm (+ H brut) sur les UNITÉS d'un grain.

    ``tranche_mot`` : si fourni (grain mot), σ = occurrence[tranche] ; sinon
    (syllabe/g0) l'occurrence EST déjà la signature du grain.

    H par unité = H de la PREMIÈRE occurrence (ordre corpus). Vérification de
    chaîne publiée : ``n_unites_h_variable`` compte les unités dont une
    occurrence donne un H float64 différent (attendu 0 : la signature est une
    fonction de l'identité aux trois grains — au mot via la physique du C).
    Sélecteur (S > 0) : part publiée des deux côtés (frontière H62 de H).
    Verdict non-dégénérescence : ``n_distinct(H_norm) >= 16 ET IQR >= 0,01``.
    """
    idx = list(tranche_mot) if tranche_mot is not None else None
    n_unites = 0
    n_degenere = 0
    n_h_variable = 0
    h_norms: List[float] = []
    h_bruts: List[float] = []
    for _ident, occs in unites.items():
        n_unites += 1
        sigmas = [
            np.asarray(o, dtype=np.float64)[idx] if idx is not None
            else np.asarray(o, dtype=np.float64)
            for o in occs
        ]
        # Déduplication par octets : des σ float64 identiques ont, par
        # définition de la formule, le même (S, H) — on ne calcule l'entropie
        # qu'une fois par motif d'octets (identique au bit au calcul naïf).
        uniques: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
        for sig in sigmas:
            uniques.setdefault(sig.tobytes(), sig)
        entropies = {cle: entropie_masse(sig) for cle, sig in uniques.items()}
        s0, h0, hn0 = entropies[sigmas[0].tobytes()]
        if len({(e[0], e[1]) for e in entropies.values()}) > 1:
            n_h_variable += 1
        if h0 is None:
            n_degenere += 1
            continue
        h_bruts.append(h0)
        h_norms.append(float(hn0))

    n_support = n_unites - n_degenere
    res: Dict[str, object] = {
        "n_unites": n_unites,
        "n_support_positif": n_support,
        "n_degenere_support": n_degenere,
        "part_support_positif": (n_support / n_unites) if n_unites else None,
        "n_unites_h_variable": n_h_variable,
        "n_distinct_h_norm": len(set(h_norms)),
    }
    if h_norms:
        res["h_norm"] = _quartiles(h_norms)
        res["h_brut"] = _quartiles(h_bruts)
        res["non_degeneree"] = bool(
            len(set(h_norms)) >= SEUIL_H_N_DISTINCTS
            and res["h_norm"]["iqr"] >= SEUIL_H_IQR  # type: ignore[index]
        )
    else:
        res["h_norm"] = None
        res["h_brut"] = None
        res["non_degeneree"] = False
    return res


def stats_var_ctx(unites: "OrderedDict[object, List[Sequence[float]]]",
                  dims_ctx: Optional[Tuple[int, ...]] = None) -> Dict[str, object]:
    """Distribution de Var_ctx sur les unités MESURABLES (n_occ ≥ 2) d'un grain.

    ``dims_ctx`` : si fourni (grain mot), la matrice d'occurrences est
    restreinte à ces dims (28-30) ; sinon la formule s'applique aux
    composantes de σ du grain (constantes de table — zéro exact prédit).

    Publie la frontière H62 (émission §5) : effectifs, ``part_mesurable``,
    masse d'occurrences couverte, et le régime CREUX/ESTIMABLE déclaré.
    """
    n_total = len(unites)
    n_occ_total = sum(len(v) for v in unites.values())
    mesurables: List[Tuple[object, float]] = []
    n_creuses = 0
    masse_mesurable = 0
    for ident, occs in unites.items():
        if len(occs) < 2:
            n_creuses += 1
            continue
        masse_mesurable += len(occs)
        if dims_ctx is not None:
            mat = np.stack([np.asarray(o, dtype=np.float64)[list(dims_ctx)] for o in occs])
        else:
            mat = np.asarray(occs, dtype=np.float64)
        mesurables.append((ident, var_ctx(mat)))

    n_mes = len(mesurables)
    part_mes = (n_mes / n_total) if n_total else None
    regime = "VIDE"
    if n_total:
        creuse = (part_mes < SEUIL_PART_MESURABLE) or (n_mes < SEUIL_N_MESURABLES)
        regime = "CREUX" if creuse else "ESTIMABLE"

    valeurs = [v for _, v in mesurables]
    n_zero = sum(1 for v in valeurs if v == 0.0)
    positifs = [v for v in valeurs if v > 0.0]
    res: Dict[str, object] = {
        "n_unites_total": n_total,
        "n_mesurables": n_mes,
        "n_creuses": n_creuses,
        "part_mesurable": part_mes,
        "n_occurrences_total": n_occ_total,
        "masse_occ_couverte": (masse_mesurable / n_occ_total) if n_occ_total else None,
        "regime_h62": regime,
        "n_zero_exact": n_zero,
        "part_zero_exact": (n_zero / n_mes) if n_mes else None,
        "n_strictement_positif": len(positifs),
        "part_strictement_positive": (len(positifs) / n_mes) if n_mes else None,
        "quartiles_positifs": _quartiles(positifs) if positifs else None,
    }
    return res


# ---------------------------------------------------------------------------
# Contrôle par permutation (émission §7 — INFO hors-verdict, graine gelée)
# ---------------------------------------------------------------------------

def controle_permutation(
    groupes: Sequence[np.ndarray],
    graine: int = GRAINE_PERMUTATION,
    repl: int = R_PERMUTATION,
) -> Dict[str, object]:
    """Permutation du multiset des vecteurs (28-30) entre occurrences des mots
    mesurables, tailles de groupes conservées ; R réplicats, une seule graine.

    ``groupes`` : liste ordonnée (ordre corpus) de matrices (n_i, D), n_i ≥ 2.
    Publie la médiane observée de Var_ctx vs la distribution des médianes sous
    shuffle. Déterministe : ``numpy.random.default_rng(graine)`` unique.
    """
    if not groupes:
        raise ValueError("aucun groupe mesurable : contrôle non défini")
    tailles = [g.shape[0] for g in groupes]
    pool = np.concatenate([np.asarray(g, dtype=np.float64) for g in groupes], axis=0)
    mediane_obs = float(np.median([var_ctx(g) for g in groupes]))

    rng = np.random.default_rng(graine)
    medianes: List[float] = []
    for _ in range(repl):
        perm = rng.permutation(pool.shape[0])
        melange = pool[perm]
        vals = []
        cur = 0
        for n_i in tailles:
            vals.append(var_ctx(melange[cur:cur + n_i]))
            cur += n_i
        medianes.append(float(np.median(vals)))

    arr = np.asarray(medianes, dtype=np.float64)
    return {
        "graine": graine,
        "n_replicats": repl,
        "n_groupes": len(groupes),
        "n_occurrences_pool": int(pool.shape[0]),
        "mediane_observee": mediane_obs,
        "medianes_shuffle": _quartiles(arr),
        "n_shuffle_inferieurs_obs": int((arr < mediane_obs).sum()),
        "n_shuffle_superieurs_obs": int((arr > mediane_obs).sum()),
    }


# ---------------------------------------------------------------------------
# Pilote (chargement .so gardé, mesure complète d'un corpus)
# ---------------------------------------------------------------------------

def charger_tokenizer_garde(sha256_attendu: str = SHA256_SO_REFERENCE):
    """Charge le tokenizer natif et VÉRIFIE le sha256 du .so avant tout appel.

    Politique T72 (« le sha identifie un fichier, pas les sources ») : un sha
    différent n'est pas un refus automatique, mais ce pilote de tour exige le
    gel — mismatch ⇒ RuntimeError (la route rebuild+parité appartient aux
    portes de l'émission, pas à ce module).
    """
    from spiraton.data.tokenizer_bridge import load_native_tokenizer

    tok = load_native_tokenizer()
    prov = tok.provenance()
    if prov["native_lib_sha256"] != sha256_attendu:
        raise RuntimeError(
            f"sha256 du .so = {prov['native_lib_sha256']} != gel {sha256_attendu} : "
            "NON-MESURE (porte 1 de l'émission T73)"
        )
    return tok, prov


def table_miroir() -> Sequence[dict]:
    """Le miroir Python de la table phonémique (parité 36×tous-champs, T72)."""
    from spiraton_tokenizer.phoneme_table_fr import PHONEME_TABLE_FR

    return PHONEME_TABLE_FR


def mesurer_corpus(tok, contextes: Sequence[str], table: Sequence[dict],
                   avec_permutation: bool = False) -> Dict[str, object]:
    """Mesure complète d'un corpus : H et Var_ctx aux 3 grains + comptes.

    ``avec_permutation`` : ajoute le contrôle par permutation (grain mot,
    graine gelée) — réservé à eve par l'émission, INFO hors-verdict.
    """
    tokens_ctx = [tok.tokenize_sequences(c) for c in contextes]
    occ = extraire_occurrences(tokens_ctx, table)
    mots = occ["mot"]
    syllabes = occ["syllabe"]
    g0 = occ["g0"]

    resultat: Dict[str, object] = {
        "comptes": occ["comptes"],
        "h": {
            "mot": stats_h(mots, tranche_mot=MAIN_DIMS),
            "syllabe": stats_h(syllabes),
            "g0_INFO": stats_h(g0),
        },
        "var_ctx": {
            "mot": stats_var_ctx(mots, dims_ctx=CTX_DIMS),
            "syllabe": stats_var_ctx(syllabes),
            "g0": stats_var_ctx(g0),
        },
    }
    if avec_permutation:
        groupes = [
            np.stack([np.asarray(o, dtype=np.float64)[list(CTX_DIMS)] for o in occs])
            for occs in mots.values()
            if len(occs) >= 2
        ]
        resultat["controle_permutation_INFO"] = controle_permutation(groupes)
    return resultat


def etalonner(contextes_9: Sequence[str]) -> Dict[str, object]:
    """Étalonnage gelé (émission §11) : ``t_cal`` = chargement du ``.so``
    (garde sha comprise) + ``tokenize_sequences`` sur les 9 premières phrases
    d'eve (≈1 % de 876). Mur du run = ``1000 × t_cal``, plafond dur 30 min."""
    t0 = time.perf_counter()
    tok, _prov = charger_tokenizer_garde()
    t_charge = time.perf_counter() - t0
    n_tok = 0
    for c in contextes_9:
        n_tok += len(tok.tokenize_sequences(c))
    t_cal = time.perf_counter() - t0
    return {
        "n_phrases": len(contextes_9),
        "n_tokens": n_tok,
        "t_chargement_so_s": t_charge,
        "t_cal_s": t_cal,
        "mur_s": min(1000.0 * t_cal, 1800.0),
        "plafond_dur_s": 1800.0,
    }


# ---------------------------------------------------------------------------
# CLI minimal (les artefacts TOUR73_* vivent à la racine écosystème)
# ---------------------------------------------------------------------------

def _md5(chemin: str) -> str:
    with open(chemin, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eve", required=True, help="chemin de corpus_eve_clean.txt")
    ap.add_argument("--claude-aba", required=True, help="chemin de corpus_claude_aba.txt")
    ap.add_argument("--sortie", required=True, help="chemin du JSON de distributions")
    ap.add_argument("--etalonnage", default=None, help="si fourni : n'exécute QUE l'étalonnage vers ce chemin")
    args = ap.parse_args(argv)

    ctx_eve, cpt_eve = contextes_eve(args.eve)
    if args.etalonnage:
        # Le chargement du .so fait partie de t_cal (émission §11) : il est
        # chronométré DANS etalonner, pas ici.
        cal = etalonner(ctx_eve[:9])
        with open(args.etalonnage, "w", encoding="utf-8") as fh:
            json.dump({"tour": 73, "etalonnage": cal}, fh, ensure_ascii=False, indent=2)
        return

    tok, prov = charger_tokenizer_garde()
    table = table_miroir()

    ctx_c, cpt_c = contextes_claude_aba(args.claude_aba)
    sortie = {
        "tour": 73,
        "provenance": {
            "so_sha256": prov["native_lib_sha256"],
            "abi_contract": prov["abi_contract_version"],
            "heuristic_mode": prov["heuristic_mode"],
            "md5_eve": _md5(args.eve),
            "md5_claude_aba": _md5(args.claude_aba),
            "tranche_main": list(MAIN_DIMS),
            "dims_ctx": list(CTX_DIMS),
            "graine_permutation": GRAINE_PERMUTATION,
            "r_permutation": R_PERMUTATION,
        },
        "contextes": {"eve": cpt_eve, "claude_aba": cpt_c},
        "eve": mesurer_corpus(tok, ctx_eve, table, avec_permutation=True),
        "claude_aba": mesurer_corpus(tok, ctx_c, table, avec_permutation=False),
    }
    with open(args.sortie, "w", encoding="utf-8") as fh:
        json.dump(sortie, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover - pilote
    main()
