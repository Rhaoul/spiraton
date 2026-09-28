"""Tour 12 — contrôle DURCI : isoler l'ordre comme SEUL facteur.

Réserve d'équité du Tour 11 (à lever) : le contrôle ``commuting`` SYMÉTRIQUE
(``Q diag Qᵀ``, ‖W−Wᵀ‖=0) différait de l'ordonné sur DEUX axes — l'ordre absent
ET la symétrie. Une probe avec un contrôle commutant-asymétrique faisait tomber
l'avantage de +7.6 à ~+3.8 (8/10). Ce tour DURCIT le contrôle (commutant DURCI
``P diag P⁻¹``, P non orthogonale bien conditionnée) pour que ordonné et contrôle
ne diffèrent QUE par l'ordre, et tranche :

  avantage_propre(seed) = Δ_total(ordered) − Δ_total(commuting_asym_wc)

Seuils de DÉCISION (déclarés d'avance) :
  - ≥ 15/20 graines avec avantage_propre > 0 (binomial P(X≥15|20,0.5)=0.021)
  - médiane avantage_propre ≥ +2.0
  - 5e percentile > 0

Issues : (a) PROGRESSION verrouillée / (b) NULL confond symétrie / (c) BUG
(contrôle pas durci) / (d) DISSIPATION. AUCUN terme de perte ne récompense le
commutateur (garde REFUS). β et κ_max FIXÉS d'avance (jamais fittés sur le
résultat).

Usage :
    python examples/measure_acquired_noncommutativity_tour12.py
"""

from __future__ import annotations

import sys

from spiraton.diagnostics.acquired_noncommutativity import BETA_MULT, KAPPA_MAX, run_sweep


SEEDS = list(range(20))  # 20 graines fixées (seuil binomial honnête)


def main() -> int:
    rep = run_sweep(
        seeds=SEEDS,
        dim=6,
        init_scale=0.5,
        order=("add", "sub", "mul", "div"),
        n_samples=256,
        epochs=800,
        lr=5e-3,
        input_scale=1.0,
    )
    s = rep.summary
    n = int(s["n_seeds"])

    print("=" * 76)
    print("Tour 12 — controle DURCI (asymetrique, bien conditionne) : l'ordre seul")
    print("=" * 76)
    print(f"config: {rep.config}")
    print(f"controle durci : beta = {BETA_MULT}*init_scale, kappa_max = {KAPPA_MAX}")
    print()

    print("--- COMMUTATEUR TOTAL : Delta init->final (mediane sur {0} graines) ---".format(n))
    print(f"  ordered            : init {s['median_total_init_ordered']:.4f}"
          f"  final {s['median_total_final_ordered']:.4f}"
          f"  Delta {s['median_delta_ordered']:+.4f}")
    print(f"  commuting (SYM)    : Delta {s['median_delta_commuting']:+.4f}")
    print(f"  commuting_asym_wc  : Delta {s['median_delta_commuting_asym_wc']:+.4f}")
    print()

    print("--- AVANTAGE Tour 11 (vs SYM) : non-regression + preuve du confond ---")
    print(f"  median avantage (vs SYM)        : {s['median_advantage']:+.4f}"
          f"   (cible ~+7.6)")
    print(f"  avantage>0 (vs SYM)             : {int(s['n_advantage_positive'])}/{n}"
          f"  (binom P = {s['binom_p_ge_advantage']:.6f})")
    print()

    print("--- AVANTAGE PROPRE Tour 12 (vs DURCI) : la mesure de DECISION ---")
    print(f"  median avantage_propre          : {s['median_advantage_proper']:+.4f}"
          f"   (seuil de revendication >= +2.0)")
    print(f"  5e percentile avantage_propre   : {s['p05_advantage_proper']:+.4f}"
          f"   (seuil > 0)")
    print(f"  avantage_propre>0               : {int(s['n_advantage_proper_positive'])}/{n}"
          f"  (binom P = {s['binom_p_ge_advantage_proper']:.6f})"
          f"   (seuil >= 15/20)")
    print(f"  part attribuable a la SYMETRIE  : {s['symmetry_share']:+.4f}"
          f"   (= avantage_SYM - avantage_propre)")
    print()

    print("--- CONSTRUCTION DU CONTROLE DURCI (preuve d'appariement) ---")
    print(f"  kappa(P) range                  : [{s['wc_kappa_min']:.3f}, {s['wc_kappa_max']:.3f}]"
          f"  (kappa_max = {KAPPA_MAX})")
    print(f"  max ||[W_a,W_b]|| (commute?)    : {s['wc_max_commutator']:.3e}  (~0 attendu)")
    print(f"  median ||W-W^T|| ordered        : {s['median_ordered_mean_asym']:.4f}")
    print(f"  median ||W-W^T|| commuting_wc   : {s['median_wc_mean_asym']:.4f}")
    print(f"  median ratio (wc/ordered)       : {s['median_asym_ratio_wc_over_ordered']:.4f}"
          f"   (cible [0.5, 2.0])")
    print()

    print("--- BORNE ANTI-DISSIPATION ---")
    print(f"  max total_final ordered            = {s['max_total_final_ordered']:.4f}"
          f"  (vs 10x max init = {10*s['max_total_init_ordered']:.4f})")
    print(f"  max total_final commuting_asym_wc  = {s['max_total_final_commuting_asym_wc']:.4f}")
    print()

    print("--- CONVERGENCE (preuve de comparabilite) ---")
    print(f"  loss final ordered (med)            : {s['median_loss_final_ordered']:.6f}")
    print(f"  loss final commuting SYM (med)      : {s['median_loss_final_commuting']:.6f}")
    print(f"  loss final commuting_asym_wc (med)  : {s['median_loss_final_commuting_asym_wc']:.6f}")
    print(f"  vector-floor loss final (med)       : {s['median_vector_floor_loss']:.6f}")
    print()

    print("--- PER-SEED (Delta_ord, Delta_SYM, Delta_WC, avantage_propre) ---")
    for ro, rc, rwc, adv_p in zip(
        rep.ordered, rep.commuting, rep.commuting_asym_wc, rep.per_seed_advantage_proper
    ):
        print(f"  seed {ro.seed:2d}: ord {ro.delta_total:+8.3f}"
              f"  sym {rc.delta_total:+8.3f}  wc {rwc.delta_total:+8.3f}"
              f"  adv_propre {adv_p:+8.3f}"
              f"  | loss ord {ro.loss_final:.4f} wc {rwc.loss_final:.4f}")
    print()

    # --- Qualification du retour ---------------------------------------------
    count = int(s["n_advantage_proper_positive"])
    med = s["median_advantage_proper"]
    p05 = s["p05_advantage_proper"]
    print("--- QUALIFICATION DU RETOUR ---")
    decision = (count >= 15) and (med >= 2.0) and (p05 > 0.0)
    if decision:
        print("  (a) PROGRESSION — axiome 1 dynamique VERROUILLE")
    else:
        print("  (b) NULL = confond symetrie (un ou plusieurs seuils non franchis)")
    print(f"  count>=15/20 : {count >= 15} ({count}/20)")
    print(f"  median>=+2.0 : {med >= 2.0} ({med:+.4f})")
    print(f"  5e pct>0     : {p05 > 0.0} ({p05:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
