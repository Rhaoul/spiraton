from __future__ import annotations

"""Régulation de l'organe ``regulate_step`` sur une dérive issue de DONNÉES RÉELLES (Tour 21).

VOIE A de Ra — confronter l'organe générique au RÉEL. L'organe ``regulate_step``
(``experimental/edge_controller.py``, INCHANGÉ) est prouvé générique aux Tours 19/20
mais sur GÉOMÉTRIE SYNTHÉTIQUE (perturbations P1/P_ω fabriquées a priori). Question
H21 : régule-t-il une dérive portée par la STRUCTURE A→B→A′ d'un cycle ABA réel, lue
via les vecteurs 33D du tokenizer natif ?

OPÉRATIONNALISATION (émission du linguiste, geste SUB·lévo·in — respectée telle quelle) :

  * DRIVE RÉEL. Pour chaque cycle de ``dataset_aba.txt`` (5000 cycles conformes), le
    pont natif donne ``tokens → (N, 33)``. On forme la séquence des tokens DANS L'ORDRE
    du cycle (SEG_A → SEG_B → SEG_A_PRIME), dims **8-22 SEULEMENT** (phonémique :
    impédance/flux/spins). On EXCLUT les dims 0-5 (op + chiralité) — sinon l'organe
    « réussit » en suivant l'étiquette op constante = CIRCULARITÉ (REFUS, interdit).

  * DYNAMIQUE PILOTÉE par une cellule DÉJÀ AU CANON. Chaque vecteur 33D phonémique est
    projeté en un SCALAIRE de gain natif par une ``SpiratonCell`` (canon, ``core/cell``)
    seedée et NON apprise (poids figés à l'init, jamais entraînés sur l'étiquette : pas
    de fit, pas de circularité). C'est la « dynamique pilotée par l'input 33D » de
    l'émission, sans code de dynamique neuf. Le scalaire centré sur ``base = 1.0`` est
    le facteur de gain natif ``p(t)`` — il joue EXACTEMENT le rôle de ``GainDrift.at``.

  * OBSERVABLE géométrique : ``ρ̂_t = r_t/r_{t-1}`` (T19, commandable par g). CIBLE :
    la bande PROGRESSION de ``edge_maintenance`` (héritée T19, ``_band_mask``).
    ACTIONNEUR : ``g`` via ``regulate_step`` byte-identique. La cellule canon entre
    UNIQUEMENT dans la fabrication du drive ; l'oscilloscope 2D reste le substrat de
    trajectoire (réutilisé en lecture seule, comme aux T15-T20).

``RealAbaDrive`` satisfait le protocole ``.at(t, T) -> float`` des perturbations du
T15+ : il est donc accepté SANS CHANGEMENT par ``EdgeController.run`` / ``run_fixed_gain``
/ ``edge_report``. On remplace simplement ``GainDrift`` (rampe a priori) par la dérive
``p(t)`` issue du 33D réel.

ORDRE LEXICOGRAPHIQUE GRAVÉ A PRIORI (anti-zone-morte ; premier échec = verdict ;
seuils GELÉS AVANT exécution) — implémenté dans ``run_aba_regulation`` :

  (0) RÉALITÉ DE LA DÉRIVE [PORTE]. La séquence 33D réelle, sous g fixe, produit-elle
      une dérive NON-STATIONNAIRE ? ``net_drift = |ρ̂_fin − ρ̂_début|`` agrégé vs
      ``total_var``. Critère GELÉ : ``net_drift_median > DELTA_DRIFT`` (= 0.05·cible
      bande, voir constante). Si ≈ 0 → NULL-STATIONNAIRE honnête (organe correctement
      inerte, cohérent T16/P2). On ne passe pas la porte.
  (i) PIVOT. ``η = 0`` ≡ g-fixe (``torch.equal`` des traces), ``regulate_step``
      byte-identique à ``1ac0269``. Pivot dégénéré : drive constant (un seul vecteur
      33D répété) ⇒ ``Δ = 0``.
  (ii) EFFET vs best_fixed. ``Δf_edge_réel`` médian apparié (Wilcoxon, ≥40 cycles
       déterministes). ``> 0.15`` ACTIVE / ``[0.05, 0.15]`` INERTE-déclarée /
       ``< 0.05`` MORTE.
  (iii) TEST DÉCISIF ANTI-T14 — ORDRE DÉTRUIT. Refaire (ii) sur la séquence 33D
        SHUFFLÉE. Si ``Δf_edge_réel − Δf_edge_shuffle < 0.05`` → NULL par
        ORDRE-DÉTRUIT-AUSSI-BON (l'avantage ne tient pas à la structure A→B→A′,
        leçon T14 littérale). Sinon ACTIVE-structurelle.

DIAGNOSTIC CENTRAL (``alpha_omega_spatial``, via ``edge_report``) : ``cos − l2`` et
``best_return_step`` entre ``s_0`` (entrée du cycle) et l'état final (après SEG_A_PRIME),
organe vs g-fixe. Signature de vraie progression : ``cos`` haut MAIS ``l2 > 0``.

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/`` NI ``regulate_step`` NI le
défaut de l'oscilloscope. Tout déterministe (seeds fixés). Si le ``.so`` est absent,
``RealAbaDrive`` lève ``TokenizerUnavailable`` (skip propre côté tests).
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..core.cell import SpiratonCell
from ..data.aba import AbaCycle, iter_aba_cycles
from ..data import vector33d
from ..data.tokenizer_bridge import NativeTokenizer33D
from ..experimental.edge_controller import EdgeController, run_fixed_gain
from .edge_maintenance import (
    _median,
    edge_report,
    wilcoxon_signed_rank,
    FIXED_GAIN_SWEEP,
    R_FLOOR,  # plancher de bande, sert d'échelle pour DELTA_DRIFT
)

# --- constantes GELÉES A PRIORI (REFUS : jamais réglées sur le résultat) ------

# Le drive projette les dims 8-22 (phonémique) en un facteur de gain natif autour de
# BASE_GAIN. L'échelle GAIN_SPAN borne l'amplitude de modulation : posée a priori dans
# la même plage que les rampes T15-T18 (start 0.95 → end 1.10, amplitude ~0.15) pour
# que la dérive réelle vive dans le même régime que les perturbations synthétiques.
BASE_GAIN = 1.0
GAIN_SPAN = 0.15          # demi-amplitude max de la modulation autour de BASE_GAIN
PROJ_SEED = 20210601      # graine de la cellule canon (poids figés, JAMAIS entraînés)
PROJ_TEMP = 1.0           # température du tanh de projection (1.0 = pleine échelle)

# Porte (0) : seuil de non-stationnarité de la dérive réelle.
#
# Deux quantités, complémentaires :
#   * DELTA_DRIFT (absolu) — GELÉ à 0.05·(1−R_FLOOR) = 0.035. Reporté mais NON
#     décisionnel : le net_drift ABSOLU dépend de l'amplitude de projection GAIN_SPAN
#     (arbitraire) ⇒ son franchissement varie avec proj_seed (mesuré : 3/5 graines
#     passent, 2/5 non, autour de 0.035) ⇒ verdict NON ROBUSTE s'il s'y appuie.
#   * RATIO_DRIFT (sans échelle) — la VRAIE porte. ``net_drift / total_var`` isole la
#     DIRECTIONNALITÉ du marche (composante DC / variation totale), invariante par
#     l'amplitude de projection. ANCRÉE sur les baselines synthétiques (REFUS, jamais
#     réglée sur le résultat) : une rampe pure (P1/T15-T18, régime où l'avantage
#     TENAIT) vaut 1.0 ; l'AR(1) moyenne-nulle (P2/T16, régime où l'avantage S'EST
#     EFFONDRÉ) vaut ~0.004. On exige au moins RATIO_DRIFT = 0.5 (clairement
#     directionnel, mi-chemin vers la rampe pure) pour déclarer la dérive
#     non-stationnaire. En-deçà : NULL-STATIONNAIRE (famille P2, cohérent T16).
DELTA_DRIFT = 0.05 * (1.0 - R_FLOOR)   # = 0.035, reporté (absolu, non décisionnel)
RATIO_DRIFT = 0.5                       # porte sans échelle, gelée avant toute mesure

# Seuils de verdict (hérités T19/T20, gelés)
THRESHOLD_ACTIVE = 0.15        # Δf_edge médian > 0.15 ⇒ ACTIVE
THRESHOLD_INERTE = 0.05        # Δf_edge médian ∈ [0.05, 0.15] ⇒ INERTE-déclarée
THRESHOLD_STRUCTURE = 0.05     # (réel − shuffle) ≥ 0.05 ⇒ ACTIVE-structurelle

# Réglages de run hérités T15-T20 (INCHANGÉS — sinon les bornes ne reproduisent plus
# le pivot ni la bande)
OMEGA = math.pi / 5
ETA = 0.5
G0 = 1.0
G_MIN = 0.80
G_MAX = 1.20


# --- l'organe de projection canon : SpiratonCell figée -----------------------

def _make_projector(seed: int = PROJ_SEED) -> SpiratonCell:
    """Cellule canon ``SpiratonCell(input_size=15)`` à poids FIGÉS (jamais entraînés).

    Projette un vecteur des dims 8-22 (15 canaux phonémiques) en un scalaire. Les poids
    sont tirés une fois par graine et NE SONT JAMAIS ajustés sur l'étiquette : la
    cellule n'apprend rien, elle TRANSFORME l'input 33D en un signal de gain. C'est la
    « dynamique pilotée par l'input 33D via une cellule au canon » de l'émission, sans
    aucun fit (donc sans circularité op/chiralité — les dims 0-5 ne sont jamais lues).
    """
    g = torch.Generator().manual_seed(seed)
    # On reproduit l'init canon de SpiratonCell mais sous générateur seedé explicite,
    # pour un déterminisme bit-à-bit indépendant de l'état global de torch.
    cell = SpiratonCell(input_size=15)
    with torch.no_grad():
        init = cell.cfg.init_scale
        cell.w_add.copy_(torch.randn(15, generator=g) * init)
        cell.w_sub.copy_(torch.randn(15, generator=g) * init)
        cell.w_mul.copy_(torch.randn(15, generator=g) * (init / 2.0))
        cell.w_div.copy_(torch.randn(15, generator=g) * (init / 2.0))
        cell.bias.zero_()
    cell.eval()
    return cell


def _project_to_gain(cell: SpiratonCell, phon: torch.Tensor, *, temp: float = PROJ_TEMP) -> float:
    """Un vecteur phonémique (15,) → facteur de gain natif autour de ``BASE_GAIN``.

    La cellule canon renvoie un scalaire borné (sa branche dextro/levo passe par
    tanh/atan). On le ramène par ``tanh`` dans (−1, 1) puis on le centre :
    ``g_native = BASE_GAIN + GAIN_SPAN · tanh(out / temp)``. Borné, déterministe, et
    centré sur BASE_GAIN (un vecteur nul ⇒ gain BASE_GAIN exact).
    """
    with torch.no_grad():
        out = cell(phon.reshape(1, -1))           # (1,) sortie canon
        scal = float(out.reshape(-1)[0])
    return BASE_GAIN + GAIN_SPAN * math.tanh(scal / temp)


# --- le drive réel : séquence 33D d'un cycle ABA, protocole .at(t, T) ---------

@dataclass(frozen=True)
class RealAbaDrive:
    """Dérive de gain natif issue d'un cycle ABA réel (protocole ``.at(t, T) -> float``).

    ``factors[t]`` est le facteur de gain natif du token ``t`` du cycle (ordre
    SEG_A → SEG_B → SEG_A_PRIME), projeté depuis ses dims 8-22 par la cellule canon.
    ``.at(t, T)`` renvoie ``factors[t]`` (clampé sur le dernier token si ``t`` dépasse).
    Interchangeable avec ``GainDrift`` dans ``EdgeController.run`` / ``run_fixed_gain``.

    ``seg_bounds`` mémorise les frontières (indices de fin) de SEG_A, SEG_B,
    SEG_A_PRIME dans la séquence de tokens — sert au diagnostic α-ω (s_0 = début de A,
    A′ = état après le dernier token de SEG_A_PRIME).
    """

    factors: Tuple[float, ...]
    seg_bounds: Tuple[int, int, int]   # (fin_A, fin_B, fin_A_PRIME) en indices de token
    n_tokens: int

    def at(self, t: int, T: int) -> float:
        """Facteur de gain natif au pas ``t`` (clampé sur le dernier token si t≥n)."""
        if not self.factors:
            return BASE_GAIN
        idx = t if t < len(self.factors) else len(self.factors) - 1
        return self.factors[idx]

    def net_drift(self) -> float:
        """|p(n−1) − p(0)| : dérive nette du facteur de gain (composante non-stationnaire)."""
        if len(self.factors) < 2:
            return 0.0
        return abs(self.factors[-1] - self.factors[0])

    def total_var(self) -> float:
        """Σ_t |p(t+1) − p(t)| : variation totale du facteur de gain natif."""
        if len(self.factors) < 2:
            return 0.0
        return sum(abs(self.factors[i + 1] - self.factors[i]) for i in range(len(self.factors) - 1))

    def shuffled(self, seed: int) -> "RealAbaDrive":
        """Version ORDRE-DÉTRUIT : permute les facteurs (détruit la structure A→B→A′).

        Permutation seedée déterministe (``torch.randperm``). Les seg_bounds ne sont
        plus signifiantes après shuffle ; on les conserve par symétrie de longueur mais
        elles ne servent qu'à la baseline (iii), pas au diagnostic α-ω structurel.
        """
        n = len(self.factors)
        if n < 2:
            return self
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        shuffled = tuple(self.factors[i] for i in perm)
        return RealAbaDrive(factors=shuffled, seg_bounds=self.seg_bounds, n_tokens=self.n_tokens)


def build_real_drive(
    cycle: AbaCycle,
    tok: NativeTokenizer33D,
    cell: SpiratonCell,
    *,
    temp: float = PROJ_TEMP,
) -> RealAbaDrive:
    """Construit le drive d'un cycle : tokenise A,B,A′ (dims 8-22) → facteurs de gain.

    Les trois segments sont tokenisés SÉPARÉMENT dans l'ordre A→B→A′ (le pont natif
    travaille au niveau du texte de segment), concaténés en une séquence de tokens.
    Chaque token → dims 8-22 → cellule canon → facteur de gain. ``seg_bounds`` = indices
    de fin cumulés (pour borner A′ au diagnostic α-ω).
    """
    import numpy as np  # local : numpy déjà requis par le pont

    factors: List[float] = []
    bounds: List[int] = []
    for seg in (cycle.seg_a, cycle.seg_b, cycle.seg_a_prime):
        vecs = tok.vectors(seg.text)          # (N_seg, 33) float32
        for row in vecs:
            phon = torch.from_numpy(np.asarray(row[vector33d.PHONEME_SIG], dtype=np.float32))
            factors.append(_project_to_gain(cell, phon, temp=temp))
        bounds.append(len(factors))
    # bounds a exactement 3 entrées (fin A, fin B, fin A′)
    return RealAbaDrive(
        factors=tuple(factors),
        seg_bounds=(bounds[0], bounds[1], bounds[2]),
        n_tokens=len(factors),
    )


# --- collecte des cycles utilisables (≥ 2 tokens : drive non dégénéré) --------

def collect_real_drives(
    path: str,
    tok: NativeTokenizer33D,
    cell: SpiratonCell,
    *,
    n_cycles: int = 40,
    min_tokens: int = 4,
    temp: float = PROJ_TEMP,
) -> List[RealAbaDrive]:
    """Construit ``n_cycles`` drives réels DÉTERMINISTES depuis le début du corpus.

    On prend les cycles dans l'ORDRE du fichier (déterminisme), en gardant ceux d'au
    moins ``min_tokens`` tokens (sinon le drive est trop court pour une dérive et le
    pivot dégénéré confondrait avec un cycle réel). ``min_tokens = 4`` ≈ un token par
    segment plus un (a priori, pas réglé sur le résultat).
    """
    drives: List[RealAbaDrive] = []
    for cycle in iter_aba_cycles(path):
        d = build_real_drive(cycle, tok, cell, temp=temp)
        if d.n_tokens >= min_tokens:
            drives.append(d)
        if len(drives) >= n_cycles:
            break
    return drives


# --- mesure (ii)/(iii) : f_edge organe vs g-fixe, réel et shuffle -------------

@dataclass(frozen=True)
class AbaRegulationReport:
    """Verdict T21 complet : porte (0), pivots, effet (ii), shuffle (iii), α-ω central."""

    n_cycles: int
    steps_per_cycle: List[int]
    # PORTE (0) : non-stationnarité de la dérive réelle (sous g fixe)
    net_drift_median: float       # absolu (reporté, dépend de GAIN_SPAN : non décisionnel)
    total_var_median: float
    ratio_drift_median: float      # net_drift/total_var (sans échelle) — LA porte
    delta_drift_thresh: float      # seuil absolu (reporté)
    ratio_drift_thresh: float      # seuil sans échelle (décisionnel)
    gate_passed: bool
    # (ii) effet vs best_fixed (réel)
    best_fixed_gain: float
    ctrl_f_edge: List[float]
    fixed_f_edge: List[float]
    delta_real: List[float]
    delta_real_median: float
    wilcoxon_p_real: float
    sign_pos_real: int          # nb de cycles où Δ > 0
    # (iii) shuffle (ordre détruit)
    ctrl_f_edge_shuffle: List[float]
    fixed_f_edge_shuffle: List[float]
    delta_shuffle: List[float]
    delta_shuffle_median: float
    real_minus_shuffle: float   # delta_real_median − delta_shuffle_median
    # diagnostic central α-ω (organe vs g-fixe, médianes)
    ao_cos_ctrl_median: float
    ao_l2_ctrl_median: float
    ao_cos_fixed_median: float
    ao_l2_fixed_median: float
    ao_best_return_ctrl_median: float
    ao_best_return_fixed_median: float
    # verdict
    verdict: str                # ACTIVE-structurelle / NULL-stationnaire / NULL-ordre-détruit / INERTE-déclarée


def _seed_s0_aba(seed: int) -> torch.Tensor:
    """État initial déterministe par cycle (norme ~1, jamais nul) — même loi que T15."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(2, generator=g)
    n = float(torch.linalg.vector_norm(v))
    if n < 1e-6:
        v = torch.tensor([1.0, 0.0])
        n = 1.0
    return (v / n).to(torch.float32)


def _run_pair(
    drive: RealAbaDrive,
    s0: torch.Tensor,
    g_fixed: float,
    *,
    eta: float = ETA,
) -> Tuple[object, object]:
    """Déroule l'organe (η=eta) et la baseline g-fixe (η=0) SOUS la MÊME dérive réelle.

    Horizon = nombre de tokens du cycle (la dérive est lue token par token). Retourne
    (report_organe, report_fixe) via ``edge_report`` (formule α-ω INTACTE).
    """
    steps = max(drive.n_tokens, 1)
    ctrl = EdgeController(omega=OMEGA, g0=G0, eta=eta, g_min=G_MIN, g_max=G_MAX)
    ct_ctrl = ctrl.run(s0, steps=steps, drift=drive)
    ct_fixed = run_fixed_gain(s0, steps=steps, g_fixed=g_fixed, omega=OMEGA, drift=drive)
    return edge_report(ct_ctrl), edge_report(ct_fixed)


def _best_fixed_gain(
    drives: Sequence[RealAbaDrive],
    seeds: Sequence[int],
    *,
    fixed_gains: Sequence[float] = FIXED_GAIN_SWEEP,
) -> float:
    """Meilleur g fixe = celui qui MAXIMISE le f_edge médian sur les cycles (baseline DURE)."""
    best_g = fixed_gains[0]
    best_med = -1.0
    for g in fixed_gains:
        fes: List[float] = []
        for drive, seed in zip(drives, seeds):
            s0 = _seed_s0_aba(seed)
            steps = max(drive.n_tokens, 1)
            r = edge_report(run_fixed_gain(s0, steps=steps, g_fixed=g, omega=OMEGA, drift=drive))
            fes.append(r.f_edge)
        med = _median(fes)
        if med > best_med:
            best_med = med
            best_g = g
    return best_g


def run_aba_regulation(
    path: str,
    tok: NativeTokenizer33D,
    *,
    n_cycles: int = 40,
    proj_seed: int = PROJ_SEED,
    temp: float = PROJ_TEMP,
    fixed_gains: Sequence[float] = FIXED_GAIN_SWEEP,
    shuffle_seed_base: int = 70000,
) -> AbaRegulationReport:
    """Exécute l'ordre lexicographique (0)→(i)→(ii)→(iii) sur ``n_cycles`` cycles réels.

    DÉTERMINISTE : cycles pris dans l'ordre du fichier, s0 seedé par index de cycle,
    cellule de projection seedée (``proj_seed``), shuffle seedé (``shuffle_seed_base+i``).
    L'organe ``regulate_step`` est INCHANGÉ ; le canon n'est pas touché.
    """
    cell = _make_projector(proj_seed)
    drives = collect_real_drives(path, tok, cell, n_cycles=n_cycles, temp=temp)
    n = len(drives)
    seeds = list(range(n))

    # --- PORTE (0) : non-stationnarité de la dérive réelle -------------------
    # La porte décisionnelle est le RATIO sans échelle (net_drift/total_var) : il isole
    # la directionnalité du marche, invariante par l'amplitude de projection (le net_drift
    # ABSOLU, lui, dépend de GAIN_SPAN et donne un verdict instable en proj_seed).
    net_drifts = [d.net_drift() for d in drives]
    total_vars = [d.total_var() for d in drives]
    ratios = [
        (d.net_drift() / d.total_var()) if d.total_var() > 0.0 else 0.0
        for d in drives
    ]
    net_drift_median = _median(net_drifts)
    total_var_median = _median(total_vars)
    ratio_drift_median = _median(ratios)
    gate_passed = ratio_drift_median > RATIO_DRIFT

    # --- (ii) baseline best_fixed sur les cycles RÉELS -----------------------
    best_g = _best_fixed_gain(drives, seeds, fixed_gains=fixed_gains)

    ctrl_f, fixed_f, delta_real = [], [], []
    ao_cos_c, ao_l2_c, ao_cos_f, ao_l2_f = [], [], [], []
    ao_br_c, ao_br_f = [], []
    for drive, seed in zip(drives, seeds):
        s0 = _seed_s0_aba(seed)
        r_ctrl, r_fixed = _run_pair(drive, s0, best_g)
        ctrl_f.append(r_ctrl.f_edge)
        fixed_f.append(r_fixed.f_edge)
        delta_real.append(r_ctrl.f_edge - r_fixed.f_edge)
        ao_cos_c.append(r_ctrl.ao_cos_final)
        ao_l2_c.append(r_ctrl.ao_l2_final)
        ao_cos_f.append(r_fixed.ao_cos_final)
        ao_l2_f.append(r_fixed.ao_l2_final)
        ao_br_c.append(float(r_ctrl.ao_best_return_step))
        ao_br_f.append(float(r_fixed.ao_best_return_step))

    delta_real_median = _median(delta_real)
    _, p_real, _ = wilcoxon_signed_rank(delta_real)
    sign_pos_real = sum(1 for d in delta_real if d > 0)

    # --- (iii) shuffle : ordre détruit (même best_g, mêmes s0) ---------------
    shuffles = [d.shuffled(shuffle_seed_base + i) for i, d in enumerate(drives)]
    ctrl_f_sh, fixed_f_sh, delta_sh = [], [], []
    for drive, seed in zip(shuffles, seeds):
        s0 = _seed_s0_aba(seed)
        r_ctrl, r_fixed = _run_pair(drive, s0, best_g)
        ctrl_f_sh.append(r_ctrl.f_edge)
        fixed_f_sh.append(r_fixed.f_edge)
        delta_sh.append(r_ctrl.f_edge - r_fixed.f_edge)
    delta_shuffle_median = _median(delta_sh)
    real_minus_shuffle = delta_real_median - delta_shuffle_median

    # --- verdict selon l'ordre lexicographique GELÉ --------------------------
    if not gate_passed:
        verdict = "NULL-stationnaire"
    elif delta_real_median < THRESHOLD_INERTE:
        verdict = "MORTE"
    elif delta_real_median < THRESHOLD_ACTIVE:
        verdict = "INERTE-déclarée"
    elif real_minus_shuffle < THRESHOLD_STRUCTURE:
        verdict = "NULL-ordre-détruit"
    else:
        verdict = "ACTIVE-structurelle"

    return AbaRegulationReport(
        n_cycles=n,
        steps_per_cycle=[d.n_tokens for d in drives],
        net_drift_median=net_drift_median,
        total_var_median=total_var_median,
        ratio_drift_median=ratio_drift_median,
        delta_drift_thresh=DELTA_DRIFT,
        ratio_drift_thresh=RATIO_DRIFT,
        gate_passed=gate_passed,
        best_fixed_gain=best_g,
        ctrl_f_edge=ctrl_f,
        fixed_f_edge=fixed_f,
        delta_real=delta_real,
        delta_real_median=delta_real_median,
        wilcoxon_p_real=p_real,
        sign_pos_real=sign_pos_real,
        ctrl_f_edge_shuffle=ctrl_f_sh,
        fixed_f_edge_shuffle=fixed_f_sh,
        delta_shuffle=delta_sh,
        delta_shuffle_median=delta_shuffle_median,
        real_minus_shuffle=real_minus_shuffle,
        ao_cos_ctrl_median=_median(ao_cos_c),
        ao_l2_ctrl_median=_median(ao_l2_c),
        ao_cos_fixed_median=_median(ao_cos_f),
        ao_l2_fixed_median=_median(ao_l2_f),
        ao_best_return_ctrl_median=_median(ao_br_c),
        ao_best_return_fixed_median=_median(ao_br_f),
        verdict=verdict,
    )
