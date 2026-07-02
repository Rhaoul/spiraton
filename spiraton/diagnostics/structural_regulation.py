from __future__ import annotations

"""Régulation STRUCTURELLE : l'organe ``regulate_step`` sur ``gap_struct`` réel (Tour 24).

H24 (émission linguiste, revue Fable 5 intégrée) : la généricité de l'organe
``regulate_step`` — établie T19/T20 sur DEUX instances numériques disjointes (ρ̂
rayon, cos de phase) — s'étend-elle au substrat STRUCTUREL non-commutatif ? Le
substrat, l'observable gradué ``e_t`` et la boucle fermée vivent dans
``experimental/structural_gap.py`` (choix d'implémentation gelés a priori dans son
en-tête). Ce module est le SQUELETTE LEXICOGRAPHIQUE T21 (``aba_regulation.py`` =
modèle, laissé INTACT comme référence géométrique) branché sur ``gap_struct``.

ORDRE LEXICOGRAPHIQUE GRAVÉ A PRIORI (premier échec = verdict ; seuils GELÉS AVANT
exécution — émission §5) :

  (0) PRÉ-VALIDATION D'INSTRUMENT [PORTE, acquis méta T23]. ``assert_order_sensitive``
      (δ_min = 1e-4, n_shuffle = 8) sur le PRIMAIRE (``obs_struct_frozen`` :
      reconstruction FIGÉE en interne, seul l'ordre bouge ; le shuffle permute les
      tokens, le tag voyage avec son token) ET sur la variante-contrôle MULTISET
      (``obs_struct_multiset``, comptes seuls). EXIGÉ : primaire order-sensible ET
      multiset vacuous (la seconde valide la porte elle-même — prédiction
      falsifiable de Fable 5). Primaire vacuous ⇒ ARRÊT : 7e null, localisé à la
      construction de l'observable. Multiset non-vacuous ⇒ PORTE-SUSPECTE.
  (i) PIVOTS. ``η=0`` ≡ lecteur à g fixe : égalité float EXACTE des traces (deux
      chemins de code — ``reconstruct_profile(eta=0)`` vs ``reconstruct_fixed``).
      Pivot dégénéré : profil SANS flip (tout-DX synthétique) ⇒ Δ = 0 (l'offset
      initial = target rend l'organe exactement inerte : rien à réguler).
      ``regulate_step`` byte-identique (importé, jamais copié).
  (ii) EFFET vs best_fixed. ``Δf_edge = f_edge(organe) − f_edge(best_fixed)`` sur
       ≥ 40 cycles réels DÉTERMINISTES (ordre du fichier ``dataset_aba.txt``),
       médiane appariée + Wilcoxon + compte de signes. Seuils hérités T19/T20 :
       > 0.15 ACTIVE / [0.05, 0.15] INERTE-déclarée / < 0.05 MORTE.
       ``f_edge`` = fraction des tokens avec ``e_t`` dans [target−band, target+band]
       (target = 1 token, band = 0.5 — gelés depuis la grammaire, cf. structural_gap).
  (iii) BASELINE SHUFFLE (leçon T14/T22). Refaire (ii) sur profils SHUFFLÉS (même
        best_g, shuffles seedés). ``Δf_edge(réel) − Δf_edge(shuffle) < 0.05`` ⇒
        NULL-ordre-détruit (l'avantage est la dynamique générique de l'organe, pas
        la structure A→B→A′). Sinon ACTIVE-structurelle.

DIAGNOSTIC CENTRAL α-ω : NON APPLICABLE tel quel — le substrat est une phase
SCALAIRE strictement monotone (``cos`` entre scalaires ∈ {−1, +1}, dégénéré ;
``best_return_step`` suppose un état multidim qui peut revenir près de s_0, ce
qu'une phase monotone ne fait jamais par construction). L'ANALOGUE de la clôture
est rapporté à la place : ``e_N`` = écart de phase à la CLÔTURE du cycle
(``p_ref(N) = N`` exactement) — le « retour transformé » vaut ``e_N ≈ target``
(proche-aligné-non-identique : le lecteur referme le cycle avec son avance d'un
token), ``e_N = 0`` serait la copie, ``|e_N| ≫ target`` la dissipation.

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/`` NI ``regulate_step`` NI
``aba_regulation.py`` (référence T21). Tout déterministe : cycles dans l'ordre du
fichier, shuffles seedés, AUCUNE source aléatoire non seedée. AUCUN ``.so`` requis
(le profil d'orientation est structurel — aucun skip de test).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from ..data.aba import iter_aba_cycles
from ..experimental.structural_gap import (
    ETA_STRUCT,
    G0_STRUCT,
    G_MAX_STRUCT,
    G_MIN_STRUCT,
    BAND_LEAD,
    OrientedToken,
    PHI_STAR,
    STRUCT_GAIN_SWEEP,
    TARGET_LEAD,
    f_edge_struct,
    flip_fraction,
    obs_struct_frozen,
    obs_struct_multiset,
    orientation_profile,
    reconstruct_fixed,
    reconstruct_profile,
    shuffle_orientations,
    shuffle_tokens,
)
from .edge_maintenance import _median, wilcoxon_signed_rank
from .instrument_validation import (
    DELTA_MIN_DEFAULT,
    N_SHUFFLE_DEFAULT,
    OrderSensitivityReport,
    assert_order_sensitive,
)

# Seuils de verdict HÉRITÉS T19/T20/T21 (continuité décisionnelle, gelés — émission §5).
# Importés du module T21 (source unique) plutôt que redéclarés.
from .aba_regulation import THRESHOLD_ACTIVE, THRESHOLD_INERTE, THRESHOLD_STRUCTURE

# Cycles utilisables : au moins MIN_TOKENS tokens (même a priori que T21 : ~un token
# par segment plus un) ET les deux orientations présentes (un flip existe — un profil
# sans flip est le pivot dégénéré, pas un cycle mesurable).
MIN_TOKENS = 4
N_CYCLES_DEFAULT = 40
SHUFFLE_SEED_BASE = 70000       # même base que T21 (continuité, déterminisme)
N_PROFILES_GATE0_INFO = 5       # gaps porte 0 rapportés sur les premiers profils (info)


def collect_profiles(
    path: str,
    *,
    n_cycles: int = N_CYCLES_DEFAULT,
    min_tokens: int = MIN_TOKENS,
) -> List[List[OrientedToken]]:
    """Les ``n_cycles`` premiers profils utilisables, dans l'ORDRE du fichier.

    Déterministe : ordre du fichier, aucun tirage. Utilisable = ``N ≥ min_tokens``
    ET les deux orientations présentes. Source du tag : parseur ``aba.py``
    exclusivement (garde anti-circularité (a) de l'émission).
    """
    profiles: List[List[OrientedToken]] = []
    for cycle in iter_aba_cycles(path):
        toks = orientation_profile(cycle)
        orients = {tk.orientation for tk in toks}
        if len(toks) >= min_tokens and orients == {+1, -1}:
            profiles.append(toks)
        if len(profiles) >= n_cycles:
            break
    return profiles


# --- pivots (porte 1) -------------------------------------------------------------

def pivot_eta0_is_exact(orientations: Sequence[int], *, g0: float = G0_STRUCT) -> bool:
    """``η=0`` ≡ g-fixe : égalité float EXACTE des traces (deux chemins de code)."""
    tr0 = reconstruct_profile(orientations, eta=0.0, g0=g0)
    trf = reconstruct_fixed(orientations, g_fixed=g0)
    return (
        tr0.e == trf.e
        and tr0.p_read == trf.p_read
        and tr0.p_ref == trf.p_ref
        and tr0.g == trf.g
        and tr0.o_hat == trf.o_hat
        and tr0.gap_binary == trf.gap_binary
    )


def pivot_noflip_delta(n_tokens: int = 12) -> float:
    """Pivot dégénéré : profil SANS flip (tout-DX synthétique) ⇒ Δf_edge attendu = 0.

    Incréments uniformes ⇒ à g nominal ``e ≡ target`` ⇒ l'organe ne corrige jamais
    (exactement inerte) ; le meilleur g fixe du sweep est le nominal (f_edge = 1).
    """
    orientations = [+1] * n_tokens
    fe_organ = f_edge_struct(reconstruct_profile(orientations, eta=ETA_STRUCT, g0=G0_STRUCT))
    fe_best = max(
        f_edge_struct(reconstruct_fixed(orientations, g_fixed=g)) for g in STRUCT_GAIN_SWEEP
    )
    return fe_organ - fe_best


# --- baseline dure : meilleur rythme fixe global (porte 2) -------------------------

def best_fixed_gain(
    orientation_lists: Sequence[Sequence[int]],
    *,
    fixed_gains: Sequence[float] = STRUCT_GAIN_SWEEP,
) -> float:
    """Le g fixe qui MAXIMISE le f_edge MÉDIAN sur les cycles (baseline dure, modèle T21)."""
    best_g = fixed_gains[0]
    best_med = -1.0
    for g in fixed_gains:
        fes = [f_edge_struct(reconstruct_fixed(o, g_fixed=g)) for o in orientation_lists]
        med = _median(fes)
        if med > best_med:
            best_med = med
            best_g = g
    return best_g


# --- rapport complet ---------------------------------------------------------------

@dataclass(frozen=True)
class StructuralRegulationReport:
    """Verdict T24 complet : porte 0, pivots, effet (ii), shuffle (iii), clôture e_N."""

    n_cycles: int
    tokens_per_cycle: List[int]
    flip_fracs: List[float]           # k/N par cycle (la variation structurelle, informatif)
    # PORTE 0 : pré-validation d'instrument (T23) sur le 1er profil réel
    order_primary: OrderSensitivityReport
    order_multiset: OrderSensitivityReport
    gate0_primary_sensitive: bool
    gate0_multiset_vacuous: bool
    order_gaps_info: List[float]      # gaps du primaire sur les 1ers profils (informatif)
    # PORTE 1 : pivots
    pivot_eta0_exact: bool
    pivot_noflip: float               # attendu 0.0
    # PORTE 2 : effet vs best_fixed (réel)
    best_fixed: float
    ctrl_f_edge: List[float]
    fixed_f_edge: List[float]
    delta_real: List[float]
    delta_real_median: float
    wilcoxon_p_real: float
    sign_pos_real: int
    sign_neg_real: int
    # PORTE 3 : shuffle (ordre détruit, même best_fixed)
    ctrl_f_edge_shuffle: List[float]
    fixed_f_edge_shuffle: List[float]
    delta_shuffle: List[float]
    delta_shuffle_median: float
    real_minus_shuffle: float
    # analogue de clôture (α-ω non applicable : phase scalaire monotone — en-tête)
    e_final_ctrl_median: float        # e_N organe (attendu ≈ target : retour transformé)
    e_final_fixed_median: float
    gap_binary_ctrl_median: float     # moyenne du gap binaire par cycle, médiane (organe)
    gap_binary_fixed_median: float
    verdict: str


def run_structural_regulation(
    path: str,
    *,
    n_cycles: int = N_CYCLES_DEFAULT,
    fixed_gains: Sequence[float] = STRUCT_GAIN_SWEEP,
    shuffle_seed_base: int = SHUFFLE_SEED_BASE,
    delta_min: float = DELTA_MIN_DEFAULT,
    n_shuffle: int = N_SHUFFLE_DEFAULT,
) -> StructuralRegulationReport:
    """Exécute l'ordre lexicographique (0)→(i)→(ii)→(iii) sur ``n_cycles`` cycles réels.

    DÉTERMINISTE : cycles dans l'ordre du fichier, shuffles seedés
    (``shuffle_seed_base + i``), aucune source aléatoire non seedée. Les portes
    (ii)/(iii) sont calculées même si une porte amont tombe (à titre informatif,
    modèle T21) ; le VERDICT, lui, suit strictement l'ordre lexicographique.
    """
    profiles = collect_profiles(path, n_cycles=n_cycles)
    n = len(profiles)
    if n == 0:
        raise ValueError(f"aucun cycle utilisable dans {path!r}")
    orientation_lists: List[List[int]] = [[tk.orientation for tk in p] for p in profiles]

    # --- PORTE 0 : pré-validation d'instrument (1er profil réel) -----------------
    order_primary = assert_order_sensitive(
        obs_struct_frozen, profiles[0], shuffle_fn=shuffle_tokens,
        delta_min=delta_min, n_shuffle=n_shuffle,
    )
    order_multiset = assert_order_sensitive(
        obs_struct_multiset, profiles[0], shuffle_fn=shuffle_tokens,
        delta_min=delta_min, n_shuffle=n_shuffle,
    )
    gate0_primary = order_primary.is_order_sensitive
    gate0_multiset = order_multiset.is_vacuous
    order_gaps_info = [
        assert_order_sensitive(
            obs_struct_frozen, p, shuffle_fn=shuffle_tokens,
            delta_min=delta_min, n_shuffle=n_shuffle,
        ).gap
        for p in profiles[:N_PROFILES_GATE0_INFO]
    ]

    # --- PORTE 1 : pivots ---------------------------------------------------------
    pivot_exact = pivot_eta0_is_exact(orientation_lists[0])
    pivot_noflip = pivot_noflip_delta()

    # --- PORTE 2 : organe vs best_fixed sur le réel -------------------------------
    best_g = best_fixed_gain(orientation_lists, fixed_gains=fixed_gains)

    ctrl_f, fixed_f, delta_real = [], [], []
    e_fin_c, e_fin_f, gb_c, gb_f = [], [], [], []
    for orients in orientation_lists:
        tr_c = reconstruct_profile(orients, eta=ETA_STRUCT, g0=G0_STRUCT)
        tr_f = reconstruct_fixed(orients, g_fixed=best_g)
        fe_c = f_edge_struct(tr_c)
        fe_f = f_edge_struct(tr_f)
        ctrl_f.append(fe_c)
        fixed_f.append(fe_f)
        delta_real.append(fe_c - fe_f)
        e_fin_c.append(tr_c.e[-1])
        e_fin_f.append(tr_f.e[-1])
        gb_c.append(sum(tr_c.gap_binary) / len(tr_c.gap_binary))
        gb_f.append(sum(tr_f.gap_binary) / len(tr_f.gap_binary))

    delta_real_median = _median(delta_real)
    _, p_real, _ = wilcoxon_signed_rank(delta_real)
    sign_pos = sum(1 for d in delta_real if d > 0)
    sign_neg = sum(1 for d in delta_real if d < 0)

    # --- PORTE 3 : shuffle (ordre détruit ; même best_g, shuffles seedés) ---------
    ctrl_f_sh, fixed_f_sh, delta_sh = [], [], []
    for i, orients in enumerate(orientation_lists):
        sh = shuffle_orientations(orients, shuffle_seed_base + i)
        tr_c = reconstruct_profile(sh, eta=ETA_STRUCT, g0=G0_STRUCT)
        tr_f = reconstruct_fixed(sh, g_fixed=best_g)
        fe_c = f_edge_struct(tr_c)
        fe_f = f_edge_struct(tr_f)
        ctrl_f_sh.append(fe_c)
        fixed_f_sh.append(fe_f)
        delta_sh.append(fe_c - fe_f)
    delta_shuffle_median = _median(delta_sh)
    real_minus_shuffle = delta_real_median - delta_shuffle_median

    # --- verdict lexicographique GELÉ ----------------------------------------------
    if not gate0_primary:
        verdict = "NULL-instrument-vacuous"       # 7e null localisé à l'observable
    elif not gate0_multiset:
        verdict = "PORTE-SUSPECTE"                # la porte elle-même est à réexaminer
    elif (not pivot_exact) or pivot_noflip != 0.0:
        verdict = "PIVOT-CASSE"                   # bug, pas résultat
    elif delta_real_median < THRESHOLD_INERTE:
        verdict = "MORTE"
    elif delta_real_median < THRESHOLD_ACTIVE:
        verdict = "INERTE-déclarée"
    elif real_minus_shuffle < THRESHOLD_STRUCTURE:
        verdict = "NULL-ordre-détruit"
    else:
        verdict = "ACTIVE-structurelle"

    return StructuralRegulationReport(
        n_cycles=n,
        tokens_per_cycle=[len(o) for o in orientation_lists],
        flip_fracs=[flip_fraction(o) for o in orientation_lists],
        order_primary=order_primary,
        order_multiset=order_multiset,
        gate0_primary_sensitive=gate0_primary,
        gate0_multiset_vacuous=gate0_multiset,
        order_gaps_info=order_gaps_info,
        pivot_eta0_exact=pivot_exact,
        pivot_noflip=pivot_noflip,
        best_fixed=best_g,
        ctrl_f_edge=ctrl_f,
        fixed_f_edge=fixed_f,
        delta_real=delta_real,
        delta_real_median=delta_real_median,
        wilcoxon_p_real=p_real,
        sign_pos_real=sign_pos,
        sign_neg_real=sign_neg,
        ctrl_f_edge_shuffle=ctrl_f_sh,
        fixed_f_edge_shuffle=fixed_f_sh,
        delta_shuffle=delta_sh,
        delta_shuffle_median=delta_shuffle_median,
        real_minus_shuffle=real_minus_shuffle,
        e_final_ctrl_median=_median(e_fin_c),
        e_final_fixed_median=_median(e_fin_f),
        gap_binary_ctrl_median=_median(gb_c),
        gap_binary_fixed_median=_median(gb_f),
        verdict=verdict,
    )


def _default_dataset() -> Path:
    p = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
    if not p.is_file():
        p = Path(__file__).resolve().parents[3] / "dataset_aba.txt"
    return p


if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    ds = _default_dataset()
    r = run_structural_regulation(str(ds))
    print(f"corpus            : {ds}")
    print(f"n_cycles          : {r.n_cycles}")
    print(f"tokens/cycle      : min={min(r.tokens_per_cycle)} med={_median([float(x) for x in r.tokens_per_cycle])} max={max(r.tokens_per_cycle)}")
    print(f"k/N (flip_frac)   : min={min(r.flip_fracs):.4f} med={_median(r.flip_fracs):.4f} max={max(r.flip_fracs):.4f}")
    print("--- PORTE 0 (pré-validation d'instrument, δ_min=1e-4, n_shuffle=8) ---")
    print(f"PRIMAIRE  obs_réel={r.order_primary.obs_real:.6f} obs_shuffle={r.order_primary.obs_shuffle:.6f} gap={r.order_primary.gap:.6e} order_sensitive={r.order_primary.is_order_sensitive}")
    print(f"MULTISET  obs_réel={r.order_multiset.obs_real:.6f} obs_shuffle={r.order_multiset.obs_shuffle:.6f} gap={r.order_multiset.gap:.6e} vacuous={r.order_multiset.is_vacuous}")
    print(f"gaps primaire (5 premiers profils, info) : {[f'{g:.4e}' for g in r.order_gaps_info]}")
    print("--- PORTE 1 (pivots) ---")
    print(f"pivot η=0 exact   : {r.pivot_eta0_exact}")
    print(f"pivot sans-flip Δ : {r.pivot_noflip}")
    print("--- PORTE 2 (organe vs best_fixed, réel) ---")
    print(f"best_fixed        : {r.best_fixed}")
    print(f"f_edge organe méd : {_median(r.ctrl_f_edge):.4f} | f_edge fixe méd : {_median(r.fixed_f_edge):.4f}")
    print(f"Δf_edge médian    : {r.delta_real_median:+.4f}  (Wilcoxon p={r.wilcoxon_p_real:.3e}, signes +{r.sign_pos_real}/−{r.sign_neg_real}/{r.n_cycles})")
    print("--- PORTE 3 (shuffle, ordre détruit) ---")
    print(f"Δf_edge shuffle   : {r.delta_shuffle_median:+.4f}")
    print(f"réel − shuffle    : {r.real_minus_shuffle:+.4f}")
    print("--- clôture (analogue α-ω : e_N, cible = retour avec avance d'1 token) ---")
    print(f"e_N organe médian : {r.e_final_ctrl_median:+.4f} | e_N fixe médian : {r.e_final_fixed_median:+.4f}")
    print(f"gap_binaire méd   : organe {r.gap_binary_ctrl_median:.4f} | fixe {r.gap_binary_fixed_median:.4f}")
    print(f"VERDICT (provisoire, ingénieur statue) : {r.verdict}")
