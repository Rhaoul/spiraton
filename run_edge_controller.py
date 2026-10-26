"""Tour 15 — mesure du maintien au bord du chaos : contrôleur g_t vs g fixe vs bounded.

Perturbation DÉCLARÉE A PRIORI (option 3) : dérive de gain natif g_drift(t) rampant
0.95 → 1.10 sur l'horizon. Le contrôleur ne voit PAS g_drift ; il corrige sur r_t/r_{t-1}.
Tous les réglages (η, bornes, bande, graines) sont fixés AVANT toute mesure.

Sortie : f_edge contrôleur vs CHAQUE g fixe (et le meilleur) vs bounded ; T_survie ;
std(g_t) + corrélation ; per-seed Δf_edge + compte ≥32/40 ; reproduction η=0 bit-à-bit.
"""
import math

import torch

from spiraton.experimental.edge_controller import (
    EdgeController,
    GainDrift,
    run_fixed_gain,
)
from spiraton.diagnostics.edge_maintenance import (
    FIXED_GAIN_SWEEP,
    edge_report,
    run_edge_sweep,
    _seed_s0,
)


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def main() -> None:
    torch.manual_seed(0)
    N = 40
    STEPS = 200
    OMEGA = math.pi / 5
    ETA = 0.5
    G0 = 1.0
    drift = GainDrift(start=0.95, end=1.10)

    print("=== Tour 15 — maintien au bord du chaos (gain auto-régulé) ===")
    print(f"Perturbation A PRIORI : dérive g_drift {drift.start} -> {drift.end} (rampe)")
    print(f"N={N} graines, steps={STEPS}, omega=pi/5, eta={ETA}, g0={G0}, "
          f"g_min=0.80 g_max=1.20")
    print(f"Bande : cos>=0.7 (W=10) ET r in [0.3*r0, 3.0*r0] ; post-transitoire t>=T/4")
    print()

    res = run_edge_sweep(
        n_seeds=N, steps=STEPS, omega=OMEGA, eta=ETA, g0=G0,
        g_min=0.80, g_max=1.20, drift=drift,
    )

    ctrl_med = _median(res.ctrl_f_edge)
    print("--- f_edge (médiane sur 40 graines) ---")
    print(f"  CONTRÔLEUR g_t        : {ctrl_med:.4f}")
    for g in FIXED_GAIN_SWEEP:
        m = _median(res.fixed_f_edge[g])
        tag = "  <-- MEILLEUR g fixe" if g == res.best_fixed_gain else ""
        print(f"  g fixe = {g:.2f}        : {m:.4f}{tag}")
    bnd_med = _median(res.bounded_f_edge)
    print(f"  BOUNDED (tanh, g=1)   : {bnd_med:.4f}")
    print()

    best_g = res.best_fixed_gain
    best_med = _median(res.fixed_f_edge[best_g])
    print(f"--- comparaison au MEILLEUR g fixe (g={best_g}) ---")
    print(f"  Δf_edge médian (ctrl − best_fixe) = {ctrl_med - best_med:+.4f}")
    print(f"  Δf_edge médian (ctrl − bounded)   = {ctrl_med - bnd_med:+.4f}")

    # per-seed : combien de graines où ctrl >= best_fixe (et > de combien)
    wins = sum(1 for cf, ff in zip(res.ctrl_f_edge, res.fixed_f_edge[best_g]) if cf > ff)
    ties = sum(1 for cf, ff in zip(res.ctrl_f_edge, res.fixed_f_edge[best_g]) if cf == ff)
    ge = sum(1 for cf, ff in zip(res.ctrl_f_edge, res.fixed_f_edge[best_g]) if cf >= ff)
    print(f"  per-seed ctrl > best_fixe : {wins}/{N} ; égalités : {ties}/{N} ; "
          f"ctrl >= best_fixe : {ge}/{N}")
    deltas = sorted(cf - ff for cf, ff in zip(res.ctrl_f_edge, res.fixed_f_edge[best_g]))
    print(f"  per-seed Δf_edge : min={deltas[0]:+.3f} méd={deltas[len(deltas)//2]:+.3f} "
          f"max={deltas[-1]:+.3f}")
    print()

    print("--- T_survie (médiane) ---")
    print(f"  CONTRÔLEUR : {_median(res.ctrl_t_survie)}")
    print(f"  best g fixe ({best_g}) : {_median(res.fixed_t_survie[best_g])}")
    print(f"  BOUNDED : {_median(res.bounded_t_survie)}")
    print()

    print("--- régulation de g_t (distingue régule vs g moyen) ---")
    print(f"  std(g_t) médian   : {_median(res.ctrl_g_std):.5f}  (plat si < 0.005)")
    print(f"  corr(Δg, -(ρ̂-1)) médian : {_median(res.ctrl_reg_corr):+.4f}  "
          f"(régule dans le bon sens si > 0)")
    print()

    # reproduction η=0 bit-à-bit
    s0 = _seed_s0(0)
    ct_eta0 = EdgeController(omega=OMEGA, g0=G0, eta=0.0).run(s0, steps=STEPS, drift=drift)
    ct_fix = run_fixed_gain(s0, steps=STEPS, g_fixed=G0, omega=OMEGA, drift=drift)
    bit_id = bool(torch.equal(ct_eta0.trace, ct_fix.trace))
    print(f"--- contrôle η=0 reproduit g fixe bit-à-bit : {bit_id} ---")


if __name__ == "__main__":
    main()
