from __future__ import annotations

"""Maintien au bord du chaos — ``f_edge``, ``T_survie``, régulation de ``g_t`` (Tour 15).

À partir d'une ``ControlTrace`` (sortie de ``EdgeController.run`` ou ``run_fixed_gain``),
on mesure si la trajectoire RESTE dans la bande PROGRESSION sous perturbation, et si le
gain ``g_t`` se RÉGULE (bouge dans la bonne direction au bon moment) plutôt que de rester
plat (un g fixe déguisé).

BANDE PROGRESSION — définie A PRIORI (toutes les bornes posées AVANT toute mesure, REFUS).
Un pas ``t`` est DANS la bande si les DEUX conditions tiennent :

  (1) alignement local soutenu : ``cos(s_t, s_{t-W}) ≥ cos_thresh`` (W = fenêtre fixée).
      Le système tourne encore de façon cohérente (la phase n'a pas explosé ni gelé).
  (2) rayon ni effondré ni divergent : ``r_t ∈ [r_floor·r_0, r_ceil·r_0]``.
      r < r_floor·r_0 = effondrement (RÉPÉTITION) ; r > r_ceil·r_0 = divergence (DISSIPATION).

Constantes de bande (FIXÉES, jamais réglées sur le résultat) :
  * ``W = 10``            : une période de rotation à ω=π/5 (cos sur un tour complet).
  * ``cos_thresh = 0.7``  : seuil d'alignement local de l'émission H15.
  * ``r_floor = 0.3``     : multiple de r_0 sous lequel le rayon est « effondré ».
  * ``r_ceil  = 3.0``     : multiple de r_0 au-dessus duquel le rayon a « divergé ».

VARIABLE PRINCIPALE ``f_edge`` = fraction des pas POST-TRANSITOIRE ``t ∈ [T/4, T]`` qui
sont dans la bande. VARIABLE SECONDAIRE ``T_survie`` = premier pas (≥ T/4) où la trace
quitte DÉFINITIVEMENT la bande (effondrement ou divergence sans retour). Si elle n'en
sort jamais, ``T_survie = T`` (survie complète).

RÉGULATION (distinguer « régule vraiment » de « meilleur g moyen », cœur du test) :
  * ``g_std`` = écart-type de ``g_t`` sur le post-transitoire. Plat (< 0.005) ⇒ g fixe
    déguisé (issue (c) RÉPÉTITION du contrôleur), pas une régulation.
  * ``reg_corr`` = corrélation entre Δg_t = g_t − g_{t-1} et −(ρ̂_t − 1). Une vraie
    régulation BAISSE g quand ρ̂ > 1 (trop chaud) et le MONTE quand ρ̂ < 1 : Δg et
    −(ρ̂−1) doivent être positivement corrélés. C'est EXACTEMENT la loi du contrôleur
    (Δg = −η(ρ̂−1)) ⇒ corrélation attendue ≈ +1 quand il n'est pas saturé aux bornes.

Aucune métrique ne lit ``g_drift`` ni le réglage : tout se calcule sur la trace observée.
La formule α-ω (``alpha_omega_metrics``) reste INTACTE (importée, jamais redéfinie).
"""

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..experimental.edge_controller import (
    ControlTrace,
    DriftPlusHFSine,
    EdgeController,
    GainDrift,
    MixedPerturbation,
    Perturbation,
    ProcessNoise,
    SeededDrift,
    run_fixed_gain,
)
from .alpha_omega_spatial import alpha_omega_metrics  # formule cos−l2 INTACTE


# --- constantes de bande, FIXÉES A PRIORI (REFUS : jamais réglées sur le résultat) ---

W_WINDOW = 10           # fenêtre d'alignement local (une période de rotation à ω=π/5)
COS_THRESH = 0.7        # seuil cos(s_t, s_{t-W}) ≥ 0.7 (émission H15)
R_FLOOR = 0.3           # multiple de r_0 : effondrement en-dessous (RÉPÉTITION)
R_CEIL = 3.0            # multiple de r_0 : divergence au-dessus (DISSIPATION)
TRANSIENT_FRAC = 0.25   # post-transitoire = t ∈ [T/4, T]
G_STD_FLOOR = 0.005     # plancher de std(g_t) sous lequel g est « plat » (issue c)

# balayage g fixe (baseline dure) — bornes posées a priori
FIXED_GAIN_SWEEP = (0.94, 0.97, 1.00, 1.03, 1.06, 1.09)


@dataclass(frozen=True)
class EdgeReport:
    f_edge: float            # fraction des pas post-transitoire dans la bande PROGRESSION
    t_survie: int            # premier pas (≥ T/4) de sortie définitive ; T si jamais
    g_std: float             # std(g_t) sur le post-transitoire (régulation active ?)
    reg_corr: float          # corr(Δg_t, −(ρ̂_t−1)) (régule dans le bon sens ?)
    n_in_band: int           # nb de pas dans la bande (post-transitoire)
    n_post: int              # nb total de pas post-transitoire
    # qualification α-ω du retour (sur la trace complète, formule cos−l2 INTACTE)
    ao_score: float
    ao_best_return_step: int
    ao_cos_final: float
    ao_l2_final: float


def _band_mask(ct: ControlTrace, *, window: int, cos_thresh: float,
               r_floor: float, r_ceil: float) -> List[bool]:
    """Masque par pas : ``t`` est-il dans la bande PROGRESSION ? (longueur T+1).

    Pour ``t < window`` l'alignement local ``cos(s_t, s_{t-window})`` n'est pas défini :
    on porte alors UNIQUEMENT la condition de rayon (le test de phase démarre dès qu'on a
    une fenêtre complète). Le post-transitoire (≥ T/4 ≥ window aux horizons utilisés) est
    toujours couvert par les deux conditions.
    """
    trace = ct.trace
    T1 = trace.size(0)
    r0 = float(ct.radius[0])
    lo, hi = r_floor * r0, r_ceil * r0

    mask: List[bool] = []
    for t in range(T1):
        r_t = float(ct.radius[t])
        radius_ok = (lo <= r_t <= hi)
        if t >= window:
            a = trace[t]
            b = trace[t - window]
            na = float(torch.linalg.vector_norm(a))
            nb = float(torch.linalg.vector_norm(b))
            if na > 0 and nb > 0:
                cos_local = float((a @ b) / (na * nb))
            else:
                cos_local = -1.0  # un point nul n'est pas aligné
            cos_ok = (cos_local >= cos_thresh)
        else:
            cos_ok = True  # phase non testable avant une fenêtre complète
        mask.append(bool(radius_ok and cos_ok))
    return mask


def _alpha_omega(ct: ControlTrace) -> Tuple[float, int, float, float]:
    """Qualification α-ω (cos − l2) entre ``s_0`` et chaque ``s_t`` (formule INTACTE).

    Retourne ``(ao_score, best_return_step, cos_final, l2_final)``. ``best_return_step``
    exclut ``t=0`` (trivial : cos=1, l2=0), comme ``shape_signature``.
    """
    trace = ct.trace
    s0 = trace[0].reshape(1, 1, 1, 2)
    T1 = trace.size(0)
    cos_series: List[float] = []
    l2_series: List[float] = []
    for t in range(T1):
        xt = trace[t].reshape(1, 1, 1, 2)
        l2, cos = alpha_omega_metrics(s0, xt)
        l2_series.append(float(l2.mean()))
        cos_series.append(float(cos.mean()))
    signal = [c - l for c, l in zip(cos_series, l2_series)]
    search = range(1, len(signal)) if len(signal) > 1 else range(len(signal))
    best = max(search, key=lambda i: signal[i])
    return signal[best], best, cos_series[-1], l2_series[-1]


def edge_report(
    ct: ControlTrace,
    *,
    window: int = W_WINDOW,
    cos_thresh: float = COS_THRESH,
    r_floor: float = R_FLOOR,
    r_ceil: float = R_CEIL,
    transient_frac: float = TRANSIENT_FRAC,
) -> EdgeReport:
    """Mesure ``f_edge``, ``T_survie``, la régulation de ``g_t`` et le retour α-ω.

    ct : trace sous contrôle (ou sous g fixe). Toutes les bornes de bande sont
         celles, FIXÉES A PRIORI, du module ; les exposer en argument sert UNIQUEMENT
         aux tests de robustesse, jamais à régler sur le résultat.
    """
    T1 = ct.trace.size(0)
    T = T1 - 1
    t_start = int(math.floor(transient_frac * T))

    mask = _band_mask(ct, window=window, cos_thresh=cos_thresh,
                      r_floor=r_floor, r_ceil=r_ceil)
    post = list(range(t_start, T1))
    in_band = [mask[t] for t in post]
    n_in = sum(1 for b in in_band if b)
    n_post = len(post)
    f_edge = n_in / n_post if n_post > 0 else 0.0

    # T_survie : premier pas post-transitoire où l'on QUITTE la bande DÉFINITIVEMENT.
    t_survie = T
    for k, t in enumerate(post):
        if not mask[t]:
            # sortie définitive = aucun retour dans la bande après ce pas
            if not any(mask[u] for u in post[k:]):
                t_survie = t
                break

    # régulation de g_t sur le post-transitoire
    g = ct.g_ctrl
    g_post = g[t_start:]
    g_std = float(g_post.std(unbiased=False)) if g_post.numel() > 1 else 0.0

    # corr(Δg_t, −(ρ̂_t − 1)) : régule-t-il dans le bon sens ?
    dg: List[float] = []
    neg_dev: List[float] = []
    rho = ct.rho_hat
    for t in range(max(t_start, 1), T1):
        dg.append(float(g[t] - g[t - 1]))
        rh = float(rho[t])
        neg_dev.append(-(rh - 1.0) if math.isfinite(rh) else 0.0)
    reg_corr = _pearson(dg, neg_dev)

    ao_score, ao_best, ao_cos, ao_l2 = _alpha_omega(ct)

    return EdgeReport(
        f_edge=f_edge,
        t_survie=t_survie,
        g_std=g_std,
        reg_corr=reg_corr,
        n_in_band=n_in,
        n_post=n_post,
        ao_score=ao_score,
        ao_best_return_step=ao_best,
        ao_cos_final=ao_cos,
        ao_l2_final=ao_l2,
    )


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Corrélation de Pearson (0.0 si variance nulle d'un côté — pas de structure)."""
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0
    return sab / math.sqrt(saa * sbb)


# --- balayage complet : contrôleur vs g fixe vs bounded, sur N graines --------

@dataclass(frozen=True)
class SweepResult:
    """Résultat agrégé d'un balayage sur N graines, sous la MÊME perturbation."""

    # contrôleur
    ctrl_f_edge: List[float]            # f_edge par graine
    ctrl_t_survie: List[int]
    ctrl_g_std: List[float]
    ctrl_reg_corr: List[float]
    # g fixe : dict g -> liste f_edge par graine
    fixed_f_edge: Dict[float, List[float]]
    fixed_t_survie: Dict[float, List[int]]
    best_fixed_gain: float              # le g fixe qui MAXIMISE f_edge médian
    # bounded (non-linéarité bornante : ici g fixe = 1.0 sous tanh radial)
    bounded_f_edge: List[float]
    bounded_t_survie: List[int]
    # méta
    seeds: List[int]


def _seed_s0(seed: int) -> torch.Tensor:
    """État initial déterministe par graine (norme ~1, jamais nul)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(2, generator=g)
    n = float(torch.linalg.vector_norm(v))
    if n < 1e-6:
        v = torch.tensor([1.0, 0.0])
        n = 1.0
    return (v / n).to(torch.float32)


@torch.no_grad()
def _run_bounded(s0: torch.Tensor, *, steps: int, omega: float, drift: GainDrift,
                 g_fixed: float = 1.0, r_sat: float = 1.0) -> ControlTrace:
    """Baseline BOUNDED : g fixe + non-linéarité radiale bornante (tanh sur le rayon).

    À chaque pas, après ``A_t = g_fixed·g_drift·R(ω)``, le rayon est SATURÉ par
    ``r ← r_sat·tanh(r/r_sat)`` (la direction est conservée). C'est la « non-linéarité
    bornante déjà dans l'oscilloscope » du second contrôle (REFUS) : si elle maintient
    déjà ``f_edge`` aussi bien que le contrôleur, c'est NULL (b). Le réglage ``r_sat``
    est posé a priori (= r_0 typique ≈ 1). Aucune lecture de g_drift.
    """
    R = torch.tensor(
        [[math.cos(omega), -math.sin(omega)], [math.sin(omega), math.cos(omega)]],
        dtype=torch.float32,
    )
    cur = s0.to(torch.float32)
    pts = [cur]
    radii = [float(torch.linalg.vector_norm(cur))]
    for t in range(steps):
        A_t = (g_fixed * drift.at(t, steps)) * R
        nxt = cur @ A_t.t()
        r = float(torch.linalg.vector_norm(nxt))
        if r > 0:
            r_clamped = r_sat * math.tanh(r / r_sat)
            nxt = nxt * (r_clamped / r)
        cur = nxt
        pts.append(cur)
        radii.append(float(torch.linalg.vector_norm(cur)))
    g_const = torch.full((steps + 1,), float(g_fixed), dtype=torch.float64)
    rho = torch.full((steps + 1,), float("nan"), dtype=torch.float64)
    drift_t = torch.tensor([drift.at(min(t, steps - 1), steps) for t in range(steps + 1)],
                           dtype=torch.float64)
    return ControlTrace(
        trace=torch.stack(pts, dim=0),
        radius=torch.tensor(radii, dtype=torch.float64),
        g_ctrl=g_const,
        rho_hat=rho,
        g_drift=drift_t,
    )


def run_edge_sweep(
    *,
    n_seeds: int = 40,
    steps: int = 200,
    omega: float = math.pi / 5,
    eta: float = 0.5,
    g0: float = 1.0,
    g_min: float = 0.80,
    g_max: float = 1.20,
    drift: Optional[Perturbation] = None,
    drift_factory: Optional[Callable[[int], Perturbation]] = None,
    fixed_gains: Sequence[float] = FIXED_GAIN_SWEEP,
) -> SweepResult:
    """Balayage : contrôleur vs g fixe vs bounded, perturbation PARTAGÉE ou PAR GRAINE.

    Pour chaque graine (s0 fixé par graine), on déroule : (1) le contrôleur ``g_t`` ;
    (2) chaque ``g`` fixe du balayage ; (3) la baseline bounded (tanh radial, g=1). On
    agrège ``f_edge`` et ``T_survie``. ``best_fixed_gain`` = le g fixe qui MAXIMISE
    ``f_edge`` médian (baseline DURE de comparaison, jamais un g médiocre).

    PERTURBATION (T15 vs T16) :
      * ``drift`` (T15) : MÊME perturbation pour TOUTES les graines (artefact
        d'isotropie connu — la série de rayon est invariante par graine).
      * ``drift_factory`` (T16) : une fonction ``seed -> Perturbation`` qui TIRE une
        perturbation INDÉPENDANTE PAR GRAINE (P1 SeededDrift / P2 ProcessNoise). Le
        contrôleur, chaque g fixe ET le bounded subissent EXACTEMENT la même
        perturbation pour une graine donnée (comparaison appariée valide).

    Si ``drift_factory`` est fourni il PRIME sur ``drift``. TOUS les réglages
    (intervalles de tirage, η, bornes, bande) sont posés A PRIORI (REFUS).
    """
    if drift is None and drift_factory is None:
        drift = GainDrift()
    seeds = list(range(n_seeds))

    ctrl_f, ctrl_ts, ctrl_std, ctrl_corr = [], [], [], []
    fixed_f: Dict[float, List[float]] = {g: [] for g in fixed_gains}
    fixed_ts: Dict[float, List[int]] = {g: [] for g in fixed_gains}
    bnd_f, bnd_ts = [], []

    for seed in seeds:
        s0 = _seed_s0(seed)
        # perturbation PAR GRAINE (T16) ou partagée (T15)
        d = drift_factory(seed) if drift_factory is not None else drift

        ctrl = EdgeController(omega=omega, g0=g0, eta=eta, g_min=g_min, g_max=g_max)
        rc = edge_report(ctrl.run(s0, steps=steps, drift=d))
        ctrl_f.append(rc.f_edge)
        ctrl_ts.append(rc.t_survie)
        ctrl_std.append(rc.g_std)
        ctrl_corr.append(rc.reg_corr)

        for g in fixed_gains:
            rf = edge_report(run_fixed_gain(s0, steps=steps, g_fixed=g,
                                            omega=omega, drift=d))
            fixed_f[g].append(rf.f_edge)
            fixed_ts[g].append(rf.t_survie)

        rb = edge_report(_run_bounded(s0, steps=steps, omega=omega, drift=d))
        bnd_f.append(rb.f_edge)
        bnd_ts.append(rb.t_survie)

    # meilleur g fixe = celui dont le f_edge médian est le plus haut
    best_g = max(fixed_gains, key=lambda g: _median(fixed_f[g]))

    return SweepResult(
        ctrl_f_edge=ctrl_f,
        ctrl_t_survie=ctrl_ts,
        ctrl_g_std=ctrl_std,
        ctrl_reg_corr=ctrl_corr,
        fixed_f_edge=fixed_f,
        fixed_t_survie=fixed_ts,
        best_fixed_gain=best_g,
        bounded_f_edge=bnd_f,
        bounded_t_survie=bnd_ts,
        seeds=seeds,
    )


def _median(xs: Sequence[float]) -> float:
    """Médiane (interpolée pour n pair) — sans numpy."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _iqr(xs: Sequence[float]) -> Tuple[float, float]:
    """(Q1, Q3) par interpolation linéaire — sans numpy."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return s[0], s[0]

    def q(p: float) -> float:
        pos = p * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    return q(0.25), q(0.75)


def wilcoxon_signed_rank(
    deltas: Sequence[float], *, zero_tol: float = 1e-12
) -> Tuple[float, float, int]:
    """Wilcoxon signed-rank apparié (test bilatéral), approximation normale — sans scipy.

    Paires DÉJÀ différenciées : ``deltas[i] = x_i − y_i`` (ici f_edge_ctrl − f_edge_fixed).
    H0 : médiane des différences = 0. Procédure standard :
      1. écarter les différences nulles (|d| ≤ zero_tol) ;
      2. ranger les |d| (rangs moyens en cas d'ex æquo) ;
      3. W = Σ rangs des d positifs ; statistique T = min(W+, W−) ;
      4. approximation normale avec correction de continuité et correction des ex æquo :
         μ = n(n+1)/4 ; σ² = n(n+1)(2n+1)/24 − (Σ(t³−t))/48 ;
         z = (T − μ + 0.5) / σ ; p (bilatéral) = 2·Φ(−|z|).

    Retourne ``(W_plus, p_value, n_effectif)``. Si ``n_effectif < 1`` → ``(0, 1.0, 0)``.
    L'approximation normale est appropriée pour ``n ≥ ~20`` (ici N ≥ 40). Pour les très
    petits N elle reste honnête mais conservatrice ; on RAPPORTE z et n.
    """
    nz = [d for d in deltas if abs(d) > zero_tol]
    n = len(nz)
    if n < 1:
        return 0.0, 1.0, 0

    # rangs des |d| avec rangs moyens pour les ex æquo
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    tie_correction = 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        # rang moyen sur le bloc [i, j]
        avg_rank = (i + 1 + j + 1) / 2.0
        block = j - i + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        if block > 1:
            tie_correction += block ** 3 - block
        i = j + 1

    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nz[i] < 0)
    T = min(w_plus, w_minus)

    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction / 48.0
    if var <= 0.0:
        return w_plus, 1.0, n
    sigma = math.sqrt(var)
    z = (T - mu + 0.5) / sigma  # correction de continuité
    # p bilatéral via Φ(−|z|) = 0.5·erfc(|z|/√2)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return w_plus, p, n


# --- balayage T16 : Δf_edge apparié, Wilcoxon, garde-fou isotropie ------------

@dataclass(frozen=True)
class VarianceSweepResult:
    """Résultat T16 : distribution APPARIÉE de Δf_edge sous perturbation PAR GRAINE.

    Toutes les listes sont indexées PAR GRAINE (même ordre que ``seeds``).
    """

    seeds: List[int]
    # f_edge appariés graine-à-graine
    ctrl_f_edge: List[float]
    best_fixed_gain: float
    best_fixed_f_edge: List[float]      # f_edge du best_fixed, PAR GRAINE
    bounded_f_edge: List[float]
    # contrastes appariés
    delta_vs_fixed: List[float]         # f_edge(ctrl) − f_edge(best_fixed), par graine
    delta_vs_bounded: List[float]       # f_edge(ctrl) − f_edge(bounded), par graine
    # secondaires en distribution
    ctrl_reg_corr: List[float]
    ctrl_g_std: List[float]
    ctrl_t_survie: List[int]
    # statistiques agrégées Δf_edge vs best_fixed
    delta_median: float
    delta_q1: float
    delta_q3: float
    wilcoxon_w_plus: float
    wilcoxon_p: float
    wilcoxon_n: int
    # GARDE-FOU : les séries de rayon r_t DIFFÈRENT-elles entre graines ?
    radius_distinct: bool               # True si variance de population RÉELLE
    radius_spread: float                # écart max entre séries r_t (sur le contrôleur)


def run_variance_sweep(
    drift_factory: Callable[[int], Perturbation],
    *,
    n_seeds: int = 40,
    steps: int = 200,
    omega: float = math.pi / 5,
    eta: float = 0.5,
    g0: float = 1.0,
    g_min: float = 0.80,
    g_max: float = 1.20,
    fixed_gains: Sequence[float] = FIXED_GAIN_SWEEP,
) -> VarianceSweepResult:
    """Balayage T16 : distribution de ``Δf_edge`` sous perturbation INDÉPENDANTE par graine.

    ``drift_factory(seed)`` tire une perturbation par graine (P1 ou P2). Pour CHAQUE
    graine on déroule contrôleur, chaque g fixe, et bounded SOUS LA MÊME perturbation
    (comparaison appariée valide). ``best_fixed_gain`` est sélectionné sur le f_edge
    MÉDIAN du balayage g fixe (baseline préenregistrée), puis ``Δf_edge`` est calculé
    graine-à-graine contre CE g. Wilcoxon signed-rank apparié sur la distribution.

    GARDE-FOU (pré-condition de validité, inverse du constat T15) : on vérifie que les
    séries de rayon ``r_t`` du contrôleur DIFFÈRENT entre graines. Si elles coïncident,
    l'isotropie persiste et le test est VIDE — c'est rapporté (``radius_distinct``).
    """
    seeds = list(range(n_seeds))

    sweep = run_edge_sweep(
        n_seeds=n_seeds, steps=steps, omega=omega, eta=eta, g0=g0,
        g_min=g_min, g_max=g_max, drift_factory=drift_factory, fixed_gains=fixed_gains,
    )
    best_g = sweep.best_fixed_gain
    best_fixed_f = sweep.fixed_f_edge[best_g]

    # contrastes appariés
    delta_vs_fixed = [c - f for c, f in zip(sweep.ctrl_f_edge, best_fixed_f)]
    delta_vs_bounded = [c - b for c, b in zip(sweep.ctrl_f_edge, sweep.bounded_f_edge)]

    # GARDE-FOU : les séries de rayon r_t du contrôleur diffèrent-elles entre graines ?
    radius_series: List[torch.Tensor] = []
    for seed in seeds:
        s0 = _seed_s0(seed)
        d = drift_factory(seed)
        ctrl = EdgeController(omega=omega, g0=g0, eta=eta, g_min=g_min, g_max=g_max)
        radius_series.append(ctrl.run(s0, steps=steps, drift=d).radius)
    radius_spread = _max_pairwise_radius_spread(radius_series)
    radius_distinct = radius_spread > 1e-6

    d_med = _median(delta_vs_fixed)
    d_q1, d_q3 = _iqr(delta_vs_fixed)
    w_plus, p_val, n_eff = wilcoxon_signed_rank(delta_vs_fixed)

    return VarianceSweepResult(
        seeds=seeds,
        ctrl_f_edge=sweep.ctrl_f_edge,
        best_fixed_gain=best_g,
        best_fixed_f_edge=best_fixed_f,
        bounded_f_edge=sweep.bounded_f_edge,
        delta_vs_fixed=delta_vs_fixed,
        delta_vs_bounded=delta_vs_bounded,
        ctrl_reg_corr=sweep.ctrl_reg_corr,
        ctrl_g_std=sweep.ctrl_g_std,
        ctrl_t_survie=sweep.ctrl_t_survie,
        delta_median=d_med,
        delta_q1=d_q1,
        delta_q3=d_q3,
        wilcoxon_w_plus=w_plus,
        wilcoxon_p=p_val,
        wilcoxon_n=n_eff,
        radius_distinct=radius_distinct,
        radius_spread=radius_spread,
    )


def _max_pairwise_radius_spread(series: Sequence[torch.Tensor]) -> float:
    """Écart MAX entre séries de rayon ``r_t`` sur l'ensemble des graines.

    Mesure ``max_{i,j} max_t |r_t^(i) − r_t^(j)|``. Vaut 0 (au float près) SSI toutes
    les graines partagent la même série de rayon = isotropie (constat T15). > 0 ⇒ vraie
    variance de population (pré-condition de validité du test T16). Calcul économe : on
    compare chaque série au min et au max par pas (la borne sup des écarts).
    """
    n = len(series)
    if n < 2:
        return 0.0
    stacked = torch.stack([s.to(torch.float64) for s in series], dim=0)  # (n, T+1)
    spread_per_t = stacked.max(dim=0).values - stacked.min(dim=0).values
    return float(spread_per_t.max())


# --- fabriques de perturbation T16 (intervalles posés A PRIORI) ---------------

def p1_drift_factory(
    *,
    start_lo: float = 0.93,
    start_hi: float = 0.97,
    end_lo: float = 1.07,
    end_hi: float = 1.13,
) -> Callable[[int], SeededDrift]:
    """Fabrique P1 (non-stationnaire) : ``seed -> SeededDrift`` tirée par graine."""
    def factory(seed: int) -> SeededDrift:
        return SeededDrift.from_seed(
            seed, start_lo=start_lo, start_hi=start_hi, end_lo=end_lo, end_hi=end_hi
        )
    return factory


def p2_noise_factory(
    *, steps: int = 200, phi: float = 0.5, sigma: float = 0.04, base: float = 1.0
) -> Callable[[int], ProcessNoise]:
    """Fabrique P2 (proche-stationnaire) : ``seed -> ProcessNoise`` AR(1) par graine."""
    def factory(seed: int) -> ProcessNoise:
        return ProcessNoise.from_seed(seed, steps=steps, phi=phi, sigma=sigma, base=base)
    return factory


# --- Tour 17 : balayage du mélange convexe α·P1 + (1−α)·P2 -------------------
#
# H17 : Δf_edge(α) = médiane appariée (N=40) de f_edge(ctrl) − f_edge(best_fixed)
# sous P_α = α·P1 + (1−α)·P2 (mélange au niveau du gain). best_fixed RESÉLECTIONNÉ
# à chaque α. Variable de contrôle DISCRIMINANTE : Δf_edge doit corréler la DÉRIVE
# NETTE (net_drift, ∝ α) et PAS la variation totale (total_var). Tout réglage
# (grille α, seuils, intervalles P1/P2) est posé A PRIORI.


def mix_factory(
    alpha: float,
    *,
    steps: int = 200,
    start_lo: float = 0.93,
    start_hi: float = 0.97,
    end_lo: float = 1.07,
    end_hi: float = 1.13,
    phi: float = 0.5,
    sigma: float = 0.04,
    base: float = 1.0,
) -> Callable[[int], MixedPerturbation]:
    """Fabrique P_α : ``seed -> MixedPerturbation`` (P1 et P2 depuis la MÊME graine).

    À α=1 le mélange ≡ ``SeededDrift.from_seed(seed)`` ; à α=0 ≡
    ``ProcessNoise.from_seed(seed, steps)`` (bit-à-bit, cf. ``MixedPerturbation``).
    Les défauts reproduisent les perturbations T16 (appariement préservé).
    """
    def factory(seed: int) -> MixedPerturbation:
        return MixedPerturbation.from_seed(
            seed, alpha=alpha, steps=steps,
            start_lo=start_lo, start_hi=start_hi, end_lo=end_lo, end_hi=end_hi,
            phi=phi, sigma=sigma, base=base,
        )
    return factory


def _rankdata(xs: Sequence[float]) -> List[float]:
    """Rangs (1-based) avec rangs MOYENS pour les ex æquo — sans scipy/numpy."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0  # rang moyen 1-based sur le bloc [i, j]
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float:
    """Corrélation de rang de Spearman (Pearson sur les rangs) — sans scipy.

    Gère les ex æquo via rangs moyens (formule générale = Pearson des rangs, pas
    la formule 6Σd²/n(n²−1) qui suppose l'absence d'ex æquo). Retourne 0.0 si
    moins de 2 points ou variance de rang nulle (constante = pas de structure).
    """
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    ra = _rankdata(a)
    rb = _rankdata(b)
    return _pearson(ra, rb)


@dataclass(frozen=True)
class AlphaMixPoint:
    """Résultat T17 à UN α (mélange P_α, baselines resélectionnées à cet α)."""

    alpha: float
    delta_median: float          # médiane appariée de f_edge(ctrl) − f_edge(best_fixed)
    delta_q1: float
    delta_q3: float
    wilcoxon_w_plus: float
    wilcoxon_p: float
    wilcoxon_n: int
    best_fixed_gain: float       # g fixe resélectionné À CET α (jamais figé)
    delta_vs_bounded_median: float
    # variable de contrôle discriminante (moyennes sur graines du FACTEUR de gain p_α)
    net_drift: float             # ⟨|p_α(T−1) − p_α(0)|⟩  (composante DC, ∝ α attendu)
    total_var: float             # ⟨Σ_t |p_α(t+1) − p_α(t)|⟩  (variation totale ∫|dg|)
    radius_distinct: bool        # garde-fou : séries r_t distinctes entre graines
    radius_spread: float
    delta_vs_fixed: List[float]  # distribution appariée (gardée pour Spearman global)


@dataclass(frozen=True)
class AlphaMixSweepResult:
    """Résultat T17 complet : un ``AlphaMixPoint`` par α + statistiques de forme."""

    alphas: List[float]
    points: List[AlphaMixPoint]
    # forme de la loi de réponse (Spearman maison sur les 11 paliers)
    spearman_alpha_delta: float      # ρ_s(α, Δf_edge médian) — monotonie en α
    spearman_netdrift_delta: float   # ρ_s(net_drift, Δf_edge médian) — invariant DC
    spearman_totalvar_delta: float   # ρ_s(total_var, Δf_edge médian) — doit être faible
    # seuil a priori
    alpha_star: Optional[float]      # plus petit α où la médiane franchit threshold_real
    threshold_real: float
    # comptage d'inversions de palier hors bruit ε
    n_inversions: int
    epsilon: float
    # méta
    n_seeds: int
    steps: int


def _net_drift_and_total_var(
    drift_factory: Callable[[int], Perturbation], *, n_seeds: int, steps: int
) -> Tuple[float, float]:
    """Moyennes sur graines de la dérive nette et de la variation totale du FACTEUR p_α.

    Pour chaque graine, on échantillonne ``p_α(0…steps−1)`` (le facteur de gain natif,
    PAS la trajectoire d'état) :
      * ``net_drift_seed`` = |p_α(steps−1) − p_α(0)|       (composante DC / non-stationnarité)
      * ``total_var_seed`` = Σ_t |p_α(t+1) − p_α(t)|       (variation totale ∫|dg/dt|)
    et on renvoie la moyenne sur graines. Aucune lecture d'état : c'est une propriété
    de la PERTURBATION elle-même (le test discriminant de H17).
    """
    nets: List[float] = []
    tvs: List[float] = []
    for seed in range(n_seeds):
        d = drift_factory(seed)
        vals = [d.at(t, steps) for t in range(steps)]
        nets.append(abs(vals[-1] - vals[0]))
        tvs.append(sum(abs(vals[t + 1] - vals[t]) for t in range(steps - 1)))
    return sum(nets) / len(nets), sum(tvs) / len(tvs)


def run_alpha_mix_sweep(
    alphas: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    *,
    n_seeds: int = 40,
    steps: int = 200,
    omega: float = math.pi / 5,
    eta: float = 0.5,
    g0: float = 1.0,
    g_min: float = 0.80,
    g_max: float = 1.20,
    fixed_gains: Sequence[float] = FIXED_GAIN_SWEEP,
    threshold_real: float = 0.15,
    epsilon: float = 0.03,
) -> AlphaMixSweepResult:
    """Balayage T17 : ``Δf_edge(α)`` sous le mélange convexe ``α·P1 + (1−α)·P2``.

    Pour CHAQUE α de la grille (déclarée A PRIORI), on appelle ``run_variance_sweep``
    une fois avec ``drift_factory = mix_factory(α)`` : le ``best_fixed`` est donc
    RESÉLECTIONNÉ indépendamment à chaque α (jamais figé). On mesure en plus, pour
    chaque α, la dérive nette ``net_drift`` et la variation totale ``total_var`` du
    facteur de gain (variable de contrôle discriminante de H17).

    Seuils GELÉS AVANT mesure (REFUS, pas de balayage-puis-sélection) :
      * ``threshold_real = 0.15`` : α* = plus petit α où la médiane franchit ce seuil.
      * ``epsilon = 0.03`` : amplitude de bruit sous laquelle une baisse de palier
        n'est PAS comptée comme inversion.

    Monotonie/forme via ``spearman_rho`` maison sur les 11 paliers (α, net_drift,
    total_var vs Δf_edge médian). N=40, steps=200, η=0.5, g0=1.0, g_min/g_max=
    0.80/1.20, ω=π/5 — INCHANGÉS (sinon les bornes ne reproduisent plus T16).
    """
    alphas = list(alphas)
    points: List[AlphaMixPoint] = []

    for alpha in alphas:
        factory = mix_factory(alpha, steps=steps)
        vs = run_variance_sweep(
            factory, n_seeds=n_seeds, steps=steps, omega=omega, eta=eta,
            g0=g0, g_min=g_min, g_max=g_max, fixed_gains=fixed_gains,
        )
        net_drift, total_var = _net_drift_and_total_var(
            factory, n_seeds=n_seeds, steps=steps
        )
        points.append(AlphaMixPoint(
            alpha=alpha,
            delta_median=vs.delta_median,
            delta_q1=vs.delta_q1,
            delta_q3=vs.delta_q3,
            wilcoxon_w_plus=vs.wilcoxon_w_plus,
            wilcoxon_p=vs.wilcoxon_p,
            wilcoxon_n=vs.wilcoxon_n,
            best_fixed_gain=vs.best_fixed_gain,
            delta_vs_bounded_median=_median(vs.delta_vs_bounded),
            net_drift=net_drift,
            total_var=total_var,
            radius_distinct=vs.radius_distinct,
            radius_spread=vs.radius_spread,
            delta_vs_fixed=list(vs.delta_vs_fixed),
        ))

    medians = [p.delta_median for p in points]
    net_drifts = [p.net_drift for p in points]
    total_vars = [p.total_var for p in points]

    sp_alpha = spearman_rho(alphas, medians)
    sp_net = spearman_rho(net_drifts, medians)
    sp_tv = spearman_rho(total_vars, medians)

    # α* : plus petit α (grille croissante) où la médiane franchit le seuil GELÉ.
    alpha_star: Optional[float] = None
    for p in sorted(points, key=lambda q: q.alpha):
        if p.delta_median >= threshold_real:
            alpha_star = p.alpha
            break

    # inversions de palier hors bruit : médiane qui BAISSE de plus de ε d'un α au suivant
    ordered = sorted(points, key=lambda q: q.alpha)
    n_inv = 0
    for i in range(1, len(ordered)):
        if ordered[i].delta_median < ordered[i - 1].delta_median - epsilon:
            n_inv += 1

    return AlphaMixSweepResult(
        alphas=alphas,
        points=points,
        spearman_alpha_delta=sp_alpha,
        spearman_netdrift_delta=sp_net,
        spearman_totalvar_delta=sp_tv,
        alpha_star=alpha_star,
        threshold_real=threshold_real,
        n_inversions=n_inv,
        epsilon=epsilon,
        n_seeds=n_seeds,
        steps=steps,
    )


# --- Tour 18 : balayage d'amplitude h.f. (DISJONCTION net_drift / total_var) ---
#
# H18 : sous net_drift CONSTANT (P1 fixe par graine) + total_var CROISSANT (sinus
# moyenne-nulle d'amplitude A croissante, annulé aux extrémités), Δf_edge(A) reste
# PLAT ⇒ net_drift gouverne, total_var causalement inerte. Le mélange α du T17
# CONFONDAIT les deux ; ici on les DISJOINT (geste DIV·lévo·in, l.200). Grille A,
# k_periods, seuils : POSÉS A PRIORI (REFUS, pas de balayage-puis-sélection).
#
# CRITÈRE GELÉ A PRIORI :
#   * PLATITUDE (issue i) : |Spearman(A, Δf_edge)| < 0.3 ET variation de Δf_edge
#     sur la grille < ε_flat = 0.05.
#   * RÉFUTATION : |Spearman(A, Δf_edge)| ≥ 0.85 ET variation ≥ 0.05.
# PRÉ-CONDITION DE VALIDITÉ (à reporter EN PREMIER) : net_drift(A) plat à ~1e-6
# (le sinus ne fuit pas) ET total_var(A) croissant. Si net_drift n'est pas plat
# ⇒ BUG, pas résultat (issue iv).


def hf_factory(
    amplitude: float,
    *,
    k_periods: int = 20,
    steps: int = 200,
    start_lo: float = 0.93,
    start_hi: float = 0.97,
    end_lo: float = 1.07,
    end_hi: float = 1.13,
) -> Callable[[int], DriftPlusHFSine]:
    """Fabrique P1+sinus : ``seed -> DriftPlusHFSine`` (P1 = MÊME tirage T16, +sinus A).

    À ``amplitude=0`` le facteur ≡ ``SeededDrift.from_seed(seed)`` bit-à-bit (pivot
    anti-artefact = point P1 du T16). Les défauts P1 reproduisent EXACTEMENT le
    tirage T16 (population de 40 graines), pas la dérive mono-série du T15.
    """
    def factory(seed: int) -> DriftPlusHFSine:
        return DriftPlusHFSine.from_seed(
            seed, amplitude=amplitude, k_periods=k_periods, steps=steps,
            start_lo=start_lo, start_hi=start_hi, end_lo=end_lo, end_hi=end_hi,
        )
    return factory


@dataclass(frozen=True)
class HFAmplitudePoint:
    """Résultat T18 à UNE amplitude A (baselines resélectionnées à cette A)."""

    amplitude: float
    delta_median: float          # médiane appariée de f_edge(ctrl) − f_edge(best_fixed)
    delta_q1: float
    delta_q3: float
    wilcoxon_w_plus: float
    wilcoxon_p: float
    wilcoxon_n: int
    best_fixed_gain: float       # g fixe resélectionné À CETTE A (jamais figé)
    fedge_ctrl_median: float     # f_edge médian du contrôleur
    fedge_fixed_median: float    # f_edge médian du best_fixed (dégradation commune ?)
    delta_vs_bounded_median: float
    # variable de contrôle DISJOINTE (moyennes sur graines du FACTEUR p(t))
    net_drift: float             # ⟨|p(N−1) − p(0)|⟩  (DOIT être constant/plat en A)
    total_var: float             # ⟨Σ_t |p(t+1) − p(t)|⟩  (DOIT croître avec A)
    radius_distinct: bool        # garde-fou : séries r_t distinctes entre graines
    radius_spread: float
    delta_vs_fixed: List[float]  # distribution appariée (gardée pour audit)


@dataclass(frozen=True)
class HFAmplitudeSweepResult:
    """Résultat T18 complet : un ``HFAmplitudePoint`` par A + statistiques de forme."""

    amplitudes: List[float]
    k_periods: int
    points: List[HFAmplitudePoint]
    # forme de la loi de réponse (Spearman maison sur les paliers A)
    spearman_amp_delta: float        # ρ_s(A, Δf_edge médian) — |·| < 0.3 = PLAT (issue i)
    spearman_amp_netdrift: float     # ρ_s(A, net_drift) — DOIT ≈ 0 (net_drift constant)
    spearman_amp_totalvar: float     # ρ_s(A, total_var) — DOIT > 0 (total_var croît)
    # critères GELÉS a priori
    delta_range: float               # max(Δmed) − min(Δmed) sur la grille A
    epsilon_flat: float              # seuil de platitude sur delta_range (ε_flat)
    flat_rho_thresh: float           # seuil |Spearman| < · pour « plat »
    refute_rho_thresh: float         # seuil |Spearman| ≥ · pour « réfuté »
    is_flat: bool                    # |ρ(A,Δ)| < flat_rho_thresh ET delta_range < ε_flat
    is_refuted: bool                 # |ρ(A,Δ)| ≥ refute_rho_thresh ET delta_range ≥ ε_flat
    # pré-condition de validité (reportée EN PREMIER)
    netdrift_range: float            # max(net_drift) − min(net_drift) (DOIT < 1e-6)
    netdrift_is_flat: bool           # netdrift_range < netdrift_flat_tol
    netdrift_flat_tol: float
    totalvar_increasing: bool        # total_var strictement croissant sur la grille A
    totalvar_min: float
    totalvar_max: float
    # méta
    n_seeds: int
    steps: int


def run_hf_amplitude_sweep(
    amplitudes: Sequence[float] = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12),
    *,
    k_periods: int = 20,
    n_seeds: int = 40,
    steps: int = 200,
    omega: float = math.pi / 5,
    eta: float = 0.5,
    g0: float = 1.0,
    g_min: float = 0.80,
    g_max: float = 1.20,
    fixed_gains: Sequence[float] = FIXED_GAIN_SWEEP,
    epsilon_flat: float = 0.05,
    flat_rho_thresh: float = 0.3,
    refute_rho_thresh: float = 0.85,
    netdrift_flat_tol: float = 1e-6,
) -> HFAmplitudeSweepResult:
    """Balayage T18 : ``Δf_edge(A)`` sous P1 (fixe) + sinus h.f. d'amplitude A.

    Pour CHAQUE amplitude A de la grille (déclarée A PRIORI), on appelle
    ``run_variance_sweep`` une fois avec ``drift_factory = hf_factory(A)`` : le
    ``best_fixed`` est RESÉLECTIONNÉ indépendamment à chaque A (jamais figé). On
    mesure en plus, pour chaque A, ``net_drift`` (DOIT rester plat ≈ celui de P1) et
    ``total_var`` (DOIT croître avec A) du facteur de gain — la DISJONCTION de H18.

    Seuils GELÉS AVANT mesure (REFUS, pas de balayage-puis-sélection) :
      * ``epsilon_flat = 0.05`` : variation de Δf_edge sous laquelle on dit « plat ».
      * ``flat_rho_thresh = 0.3`` : |Spearman(A,Δ)| < · = plat (issue i).
      * ``refute_rho_thresh = 0.85`` : |Spearman(A,Δ)| ≥ · = réfuté (issue ii/iii).
      * ``netdrift_flat_tol = 1e-6`` : net_drift(A) doit être plat sous ce seuil ;
        sinon le sinus FUIT dans net_drift ⇒ BUG, pas résultat (issue iv).

    Réglages INCHANGÉS depuis T16/T17 : N=40, steps=200, η=0.5, g0=1.0,
    g_min/g_max=0.80/1.20, ω=π/5 (sinon les bornes ne reproduisent plus le pivot P1).
    """
    amplitudes = list(amplitudes)
    points: List[HFAmplitudePoint] = []

    for A in amplitudes:
        factory = hf_factory(A, k_periods=k_periods, steps=steps)
        vs = run_variance_sweep(
            factory, n_seeds=n_seeds, steps=steps, omega=omega, eta=eta,
            g0=g0, g_min=g_min, g_max=g_max, fixed_gains=fixed_gains,
        )
        net_drift, total_var = _net_drift_and_total_var(
            factory, n_seeds=n_seeds, steps=steps
        )
        points.append(HFAmplitudePoint(
            amplitude=A,
            delta_median=vs.delta_median,
            delta_q1=vs.delta_q1,
            delta_q3=vs.delta_q3,
            wilcoxon_w_plus=vs.wilcoxon_w_plus,
            wilcoxon_p=vs.wilcoxon_p,
            wilcoxon_n=vs.wilcoxon_n,
            best_fixed_gain=vs.best_fixed_gain,
            fedge_ctrl_median=_median(vs.ctrl_f_edge),
            fedge_fixed_median=_median(vs.best_fixed_f_edge),
            delta_vs_bounded_median=_median(vs.delta_vs_bounded),
            net_drift=net_drift,
            total_var=total_var,
            radius_distinct=vs.radius_distinct,
            radius_spread=vs.radius_spread,
            delta_vs_fixed=list(vs.delta_vs_fixed),
        ))

    medians = [p.delta_median for p in points]
    net_drifts = [p.net_drift for p in points]
    total_vars = [p.total_var for p in points]

    sp_amp_delta = spearman_rho(amplitudes, medians)
    sp_amp_net = spearman_rho(amplitudes, net_drifts)
    sp_amp_tv = spearman_rho(amplitudes, total_vars)

    delta_range = (max(medians) - min(medians)) if medians else 0.0
    netdrift_range = (max(net_drifts) - min(net_drifts)) if net_drifts else 0.0
    netdrift_is_flat = netdrift_range < netdrift_flat_tol

    # total_var strictement croissant (ordre de la grille déclarée croissante)
    ordered = sorted(points, key=lambda q: q.amplitude)
    tv_seq = [p.total_var for p in ordered]
    totalvar_increasing = all(tv_seq[i] > tv_seq[i - 1] for i in range(1, len(tv_seq)))

    is_flat = (abs(sp_amp_delta) < flat_rho_thresh) and (delta_range < epsilon_flat)
    is_refuted = (abs(sp_amp_delta) >= refute_rho_thresh) and (delta_range >= epsilon_flat)

    return HFAmplitudeSweepResult(
        amplitudes=amplitudes,
        k_periods=k_periods,
        points=points,
        spearman_amp_delta=sp_amp_delta,
        spearman_amp_netdrift=sp_amp_net,
        spearman_amp_totalvar=sp_amp_tv,
        delta_range=delta_range,
        epsilon_flat=epsilon_flat,
        flat_rho_thresh=flat_rho_thresh,
        refute_rho_thresh=refute_rho_thresh,
        is_flat=is_flat,
        is_refuted=is_refuted,
        netdrift_range=netdrift_range,
        netdrift_is_flat=netdrift_is_flat,
        netdrift_flat_tol=netdrift_flat_tol,
        totalvar_increasing=totalvar_increasing,
        totalvar_min=min(total_vars) if total_vars else 0.0,
        totalvar_max=max(total_vars) if total_vars else 0.0,
        n_seeds=n_seeds,
        steps=steps,
    )


# --- Tour 20 : 2e ACTIONNEUR — réguler ω, observable = cohérence de phase -----
#
# H20 : établir la GÉNÉRICITÉ de l'organe PAR EXTENSION. Au T19 la phase était NON
# commandable par g (obstruction de couplage prouvée). Ici l'actionneur est ω, et la
# phase ``cos(s_t, s_{t−W})`` y EST commandable (= cos(W·ω)). Le MÊME ``regulate_step``
# (INCHANGÉ) doit maintenir la phase dans une bande autour de 0.7 contre une dérive de
# rotation native ``P_ω`` (rampe) et BATTRE le meilleur ω FIXE.
#
# BANDE DE PHASE — définie A PRIORI (anti-circularité, point dur 5 de l'émission) :
# un pas ``t`` (post-transitoire, ``t ≥ T/4``, ``t ≥ W`` pour que la phase soit
# définie) est DANS la bande SSI ``cos(s_t, s_{t−W}) ∈ [PHASE_LO, PHASE_HI]`` =
# ``[0.55, 0.85]`` (bande δ=0.15 AUTOUR de 0.7). La cible 0.7 entre dans la DÉFINITION
# DE LA BANDE, JAMAIS dans le score d'un pas individuel. Le best_fixed_ω vise la MÊME
# bande avec la MÊME métrique ⇒ le Δ apparié annule tout biais circulaire commun.

# constantes de bande de phase, FIXÉES A PRIORI (REFUS : jamais réglées sur le résultat)
PHASE_TARGET = 0.7          # cible de cohérence de phase (= cos(W·ω*), W=10, ω*≈0.0795)
PHASE_DELTA = 0.15          # demi-largeur de bande autour de la cible
PHASE_LO = PHASE_TARGET - PHASE_DELTA   # 0.55
PHASE_HI = PHASE_TARGET + PHASE_DELTA   # 0.85

# zone monotone a priori : W·ω ∈ [0, π] ⇒ ω ∈ [0, π/W]. Grille ω FIXE encadrant ω*.
OMEGA_MIN = 0.0
OMEGA_MAX = math.pi / W_WINDOW          # ≈ 0.3142 (garde anti-repliement)
FIXED_OMEGA_SWEEP = (0.05, 0.065, 0.0795, 0.095, 0.11, 0.125)  # autour de ω*≈0.0795


def f_edge_phase(
    trace: torch.Tensor,
    *,
    window: int = W_WINDOW,
    phase_lo: float = PHASE_LO,
    phase_hi: float = PHASE_HI,
    transient_frac: float = TRANSIENT_FRAC,
) -> Tuple[float, int, int]:
    """Fraction des pas post-transitoire dont la cohérence de phase est DANS la bande.

    Un pas ``t`` (avec ``t ≥ T/4`` ET ``t ≥ window`` pour que la phase soit définie) est
    DANS la bande SSI ``cos(s_t, s_{t−window}) ∈ [phase_lo, phase_hi]``. La cible n'entre
    QUE dans les bornes de bande (anti-circularité) ; aucun pas n'est scoré par sa
    proximité à 0.7. Un point nul (norme 0) n'est pas aligné ⇒ cos = −1 (hors bande).

    Retourne ``(f_edge_phase, n_in_band, n_post)``. ``n_post`` ne compte que les pas
    où la phase est DÉFINIE (``t ≥ window``) dans le post-transitoire.
    """
    T1 = trace.size(0)
    T = T1 - 1
    t_start = int(math.floor(transient_frac * T))
    t_start = max(t_start, window)  # la phase n'est définie qu'à partir de window

    n_in = 0
    n_post = 0
    for t in range(t_start, T1):
        a = trace[t]
        b = trace[t - window]
        na = float(torch.linalg.vector_norm(a))
        nb = float(torch.linalg.vector_norm(b))
        if na > 0 and nb > 0:
            cos_local = float((a @ b) / (na * nb))
        else:
            cos_local = -1.0
        n_post += 1
        if phase_lo <= cos_local <= phase_hi:
            n_in += 1
    f = n_in / n_post if n_post > 0 else 0.0
    return f, n_in, n_post


@dataclass(frozen=True)
class OmegaSweepResult:
    """Résultat T20 : distribution APPARIÉE de Δf_edge_phase sous P_ω (rampe par graine).

    Toutes les listes sont indexées PAR GRAINE (même ordre que ``seeds``). L'actionneur
    régulé est ω ; la baseline DURE est le meilleur ω FIXE (``best_fixed_omega``).
    """

    seeds: List[int]
    # f_edge_phase appariés graine-à-graine
    reg_f_edge: List[float]             # ω-régulateur
    best_fixed_omega: float
    best_fixed_f_edge: List[float]      # f_edge_phase du best_fixed_ω, PAR GRAINE
    fixed_f_edge: Dict[float, List[float]]   # chaque ω fixe -> f_edge_phase par graine
    # contraste apparié
    delta_vs_fixed: List[float]         # f_edge_phase(reg) − f_edge_phase(best_fixed_ω)
    # statistiques agrégées
    delta_median: float
    delta_q1: float
    delta_q3: float
    wilcoxon_w_plus: float
    wilcoxon_p: float
    wilcoxon_n: int
    # diagnostics de garde-fou
    omega_max_abs: float                # max_{graine,t} |ω_ctrl(t)| (doit ≤ ω_max)
    trace_all_finite: bool              # toutes les traces régulées finies ?


def p_omega_ramp_factory(
    *, start: float = 0.05, end: float = 0.12
) -> Callable[[int], "OmegaDrift"]:
    """Fabrique P_ω : ``seed -> OmegaDrift`` (rampe).

    À la différence de P1/P2 (T16) la rampe ω est la MÊME pour toutes les graines
    (comme ``GainDrift`` au T15) : la variance de population vient des ÉTATS INITIAUX
    seedés (``_seed_s0``), pas de la perturbation. C'est suffisant pour un Wilcoxon
    apparié — chaque graine est une trajectoire distincte sous la même cible mobile.
    Les bornes sont posées A PRIORI dans la zone monotone (jamais réglées sur le
    résultat). ``.degenerate()`` (amplitude 0) est ``OmegaDrift.degenerate``.
    """
    from ..experimental.edge_controller import OmegaDrift

    def factory(seed: int) -> "OmegaDrift":
        return OmegaDrift(start=start, end=end)

    return factory


def run_omega_sweep(
    drift_factory: Callable[[int], "OmegaDrift"],
    *,
    n_seeds: int = 40,
    steps: int = 200,
    eta: float = 0.05,
    omega0: float = 0.0795,
    omega_min: float = OMEGA_MIN,
    omega_max: float = OMEGA_MAX,
    g_fixed: float = 1.0,
    fixed_omegas: Sequence[float] = FIXED_OMEGA_SWEEP,
    window: int = W_WINDOW,
) -> OmegaSweepResult:
    """Balayage T20 : ω-régulateur vs ω fixe, sous la dérive de rotation ``P_ω``.

    Pour CHAQUE graine (s0 fixé par ``_seed_s0``), on déroule : (1) l'ω-régulateur ;
    (2) chaque ω fixe du balayage. Tous subissent la MÊME dérive ω pour une graine
    donnée (comparaison appariée). ``best_fixed_omega`` = l'ω fixe qui MAXIMISE le
    ``f_edge_phase`` médian (baseline DURE). ``Δf_edge_phase`` est calculé
    graine-à-graine contre CE ω, puis Wilcoxon signed-rank apparié.

    L'observable régulé est ``−cos(s_t, s_{t−W})`` (CROISSANT en ω, cf. mesure T20 sur
    le signe) vers ``target = −PHASE_TARGET`` ; la BANDE de score (``f_edge_phase``)
    reste sur le ``cos`` brut (anti-circularité). ``η = 0.05 = η_ρ/W`` (gain
    ré-échelonné par la sensibilité ``W·sin(W·ω*)≈W`` du couplage de phase).

    g reste FIXE (un seul actionneur régulé). ω borné par le clip de ``regulate_step``
    (ω_min/ω_max = zone monotone). Garde-fou de stabilité (finitude, |ω| ≤ ω_max)
    rapporté avant toute lecture de Δ.
    """
    from ..experimental.edge_controller import (
        OmegaRegulator,
        make_obs_neg_phase_coherence,
        run_fixed_omega,
    )

    seeds = list(range(n_seeds))
    phase_obs = make_obs_neg_phase_coherence(window)

    reg_f: List[float] = []
    fixed_f: Dict[float, List[float]] = {w: [] for w in fixed_omegas}
    omega_max_abs = 0.0
    all_finite = True

    for seed in seeds:
        s0 = _seed_s0(seed)
        d = drift_factory(seed)

        reg = OmegaRegulator(
            phase_obs, target=-PHASE_TARGET, omega0=omega0, eta=eta,
            omega_min=omega_min, omega_max=omega_max, g_fixed=g_fixed,
        )
        rt = reg.run(s0, steps=steps, drift=d)
        f, _, _ = f_edge_phase(rt.trace, window=window)
        reg_f.append(f)
        all_finite = all_finite and bool(torch.isfinite(rt.trace).all())
        omega_max_abs = max(omega_max_abs, float(rt.omega_ctrl.abs().max()))

        for w in fixed_omegas:
            ft = run_fixed_omega(s0, steps=steps, omega_fixed=w, drift=d, g_fixed=g_fixed)
            ff, _, _ = f_edge_phase(ft.trace, window=window)
            fixed_f[w].append(ff)

    best_w = max(fixed_omegas, key=lambda w: _median(fixed_f[w]))
    best_fixed_f = fixed_f[best_w]
    delta = [r - b for r, b in zip(reg_f, best_fixed_f)]

    d_med = _median(delta)
    d_q1, d_q3 = _iqr(delta)
    w_plus, p_val, n_eff = wilcoxon_signed_rank(delta)

    return OmegaSweepResult(
        seeds=seeds,
        reg_f_edge=reg_f,
        best_fixed_omega=best_w,
        best_fixed_f_edge=best_fixed_f,
        fixed_f_edge=fixed_f,
        delta_vs_fixed=delta,
        delta_median=d_med,
        delta_q1=d_q1,
        delta_q3=d_q3,
        wilcoxon_w_plus=w_plus,
        wilcoxon_p=p_val,
        wilcoxon_n=n_eff,
        omega_max_abs=omega_max_abs,
        trace_all_finite=all_finite,
    )


def omega_phase_spread(
    *,
    omegas: Sequence[float] = FIXED_OMEGA_SWEEP,
    n_seeds: int = 40,
    steps: int = 200,
    g_fixed: float = 1.0,
    window: int = W_WINDOW,
    drift: Optional["OmegaDrift"] = None,
) -> Tuple[float, Dict[float, float]]:
    """Pré-condition (0) de COMMANDABILITÉ : étendue de la phase agrégée sur la grille ω.

    Pour chaque ω de la grille, on déroule à ω FIXE (η=0, g fixe) sous une dérive
    DÉGÉNÉRÉE (rampe nulle ⇒ ω natif constant) et on agrège la cohérence de phase
    moyenne post-transitoire sur la population de graines. On retourne l'ÉTENDUE
    (max − min de cette phase agrégée sur la grille ω) ET le détail par ω.

    Si l'étendue > 0.3 (seuil a priori), la phase est COMMANDABLE par ω (l'exact
    OPPOSÉ du T19 où spread(phase|g)≈0). C'est la PORTE lue EN PREMIER : si elle
    échoue, l'objectif est mal posé, on ne déploie pas. ``cos(W·ω)`` prédit qu'elle
    passe largement (la grille couvre cos de ~0.88 à ~0.36).
    """
    from ..experimental.edge_controller import OmegaDrift, run_fixed_omega

    if drift is None:
        drift = OmegaDrift.degenerate()  # rampe nulle : ω natif constant (test de commande pure)

    seeds = list(range(n_seeds))
    phase_by_omega: Dict[float, float] = {}
    for w in omegas:
        phase_vals: List[float] = []
        for seed in seeds:
            s0 = _seed_s0(seed)
            ft = run_fixed_omega(s0, steps=steps, omega_fixed=w, drift=drift, g_fixed=g_fixed)
            # cohérence de phase moyenne post-transitoire (pas la bande : la valeur brute)
            trace = ft.trace
            T1 = trace.size(0)
            t_start = max(int(math.floor(TRANSIENT_FRAC * (T1 - 1))), window)
            cos_acc: List[float] = []
            for t in range(t_start, T1):
                a = trace[t]
                b = trace[t - window]
                na = float(torch.linalg.vector_norm(a))
                nb = float(torch.linalg.vector_norm(b))
                if na > 0 and nb > 0:
                    cos_acc.append(float((a @ b) / (na * nb)))
            if cos_acc:
                phase_vals.append(sum(cos_acc) / len(cos_acc))
        phase_by_omega[w] = (sum(phase_vals) / len(phase_vals)) if phase_vals else float("nan")

    vals = [v for v in phase_by_omega.values() if math.isfinite(v)]
    spread = (max(vals) - min(vals)) if vals else 0.0
    return spread, phase_by_omega
