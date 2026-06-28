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
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..experimental.edge_controller import (
    ControlTrace,
    EdgeController,
    GainDrift,
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
    drift: Optional[GainDrift] = None,
    fixed_gains: Sequence[float] = FIXED_GAIN_SWEEP,
) -> SweepResult:
    """Balayage complet sous la MÊME perturbation : contrôleur vs g fixe vs bounded.

    Pour chaque graine (s0 fixé par graine), sous la dérive ``drift`` (perturbation
    déclarée a priori), on déroule : (1) le contrôleur ``g_t`` ; (2) chaque ``g`` fixe
    du balayage ; (3) la baseline bounded (tanh radial, g=1). On agrège ``f_edge`` et
    ``T_survie``. ``best_fixed_gain`` = le g fixe qui MAXIMISE ``f_edge`` médian (baseline
    DURE de comparaison, jamais un g médiocre).

    TOUS les réglages (perturbation, η, bornes, bande) sont posés A PRIORI. Aucun n'est
    sélectionné sur le résultat (REFUS).
    """
    if drift is None:
        drift = GainDrift()
    seeds = list(range(n_seeds))

    ctrl_f, ctrl_ts, ctrl_std, ctrl_corr = [], [], [], []
    fixed_f: Dict[float, List[float]] = {g: [] for g in fixed_gains}
    fixed_ts: Dict[float, List[int]] = {g: [] for g in fixed_gains}
    bnd_f, bnd_ts = [], []

    for seed in seeds:
        s0 = _seed_s0(seed)

        ctrl = EdgeController(omega=omega, g0=g0, eta=eta, g_min=g_min, g_max=g_max)
        rc = edge_report(ctrl.run(s0, steps=steps, drift=drift))
        ctrl_f.append(rc.f_edge)
        ctrl_ts.append(rc.t_survie)
        ctrl_std.append(rc.g_std)
        ctrl_corr.append(rc.reg_corr)

        for g in fixed_gains:
            rf = edge_report(run_fixed_gain(s0, steps=steps, g_fixed=g,
                                            omega=omega, drift=drift))
            fixed_f[g].append(rf.f_edge)
            fixed_ts[g].append(rf.t_survie)

        rb = edge_report(_run_bounded(s0, steps=steps, omega=omega, drift=drift))
        bnd_f.append(rb.f_edge)
        bnd_ts.append(rb.t_survie)

    # meilleur g fixe = celui dont le f_edge médian est le plus haut
    def _median(xs: List[float]) -> float:
        s = sorted(xs)
        return s[len(s) // 2]

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
