"""Tour 11 — la non-commutativité **acquise** pendant l'entraînement.

Diagnostic du chantier 1 jamais réalisé : suivre ``‖W_a W_b − W_b W_a‖``
PENDANT l'apprentissage (pas seulement à l'init comme au Tour 6).

Question : un ``MatrixSpiratonCell`` entraîné sur une tâche dont la solution
dépend de l'ORDRE de composition voit-il son commutateur total CROÎTRE, et
cette croissance disparaît-elle quand l'ordre est retiré de la cible ?

Balayage déclaré d'avance, UNE seule tâche :
- *ordered*    : enseignant non-commutant figé → cible ordonnée.
- *commuting*  : enseignant commutant figé (order-détruit) → même difficulté,
                 ordre sans information.
- *vector_floor* : application linéaire unique (order-free) — garde-fou de
                 validité de la tâche.

AUCUN terme de perte ne récompense le commutateur (garde REFUS).

Usage :
    python examples/measure_acquired_noncommutativity.py
"""

from __future__ import annotations

import sys

from spiraton.diagnostics.acquired_noncommutativity import run_sweep


SEEDS = list(range(20))  # ≥20 graines fixées (kickoff)


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

    print("=" * 72)
    print("Tour 11 — non-commutativite ACQUISE pendant l'entrainement")
    print("=" * 72)
    print(f"config: {rep.config}")
    print()

    print("--- COMMUTATEUR TOTAL : init -> final (mediane sur {0} graines) ---".format(n))
    print(f"  ordered   : init {s['median_total_init_ordered']:.4f}"
          f"  final {s['median_total_final_ordered']:.4f}"
          f"  Delta {s['median_delta_ordered']:+.4f}")
    print(f"  commuting : Delta {s['median_delta_commuting']:+.4f}")
    print(f"  avantage (Delta_ordre - Delta_controle), mediane : {s['median_advantage']:+.4f}")
    print()

    print("--- DECOMPTE CONDITIONNEL (acquisition portee par l'ordre) ---")
    print(f"  avantage > 0 sur {int(s['n_advantage_positive'])}/{n} graines"
          f"  (binomial P(X>={int(s['n_advantage_positive'])}) = {s['binom_p_ge_advantage']:.6f})")
    print()

    print("--- BORNE ANTI-DISSIPATION ---")
    print(f"  max total_final ordered = {s['max_total_final_ordered']:.4f}"
          f"  (vs 10 x max total_init = {10*s['max_total_init_ordered']:.4f})")
    print()

    print("--- CONVERGENCE (preuve de comparabilite) ---")
    print(f"  loss init ordered (med)       : {s['median_loss_init_ordered']:.6f}")
    print(f"  loss final ordered (med)      : {s['median_loss_final_ordered']:.6f}")
    print(f"  loss final commuting (med)    : {s['median_loss_final_commuting']:.6f}")
    print(f"  vector-floor loss final (med) : {s['median_vector_floor_loss']:.6f}")
    print()

    print("--- ASYMETRIE DES POLES (Delta median, tache ordonnee) ---")
    print(f"  MUL-ADD (dextro, primaire)  : {s['median_delta_mul_add_ordered']:+.4f}")
    print(f"  DIV-SUB (levo, covariable)  : {s['median_delta_div_sub_ordered']:+.4f}")
    print()

    print("--- PER-SEED (Delta_ordre, Delta_controle, avantage) ---")
    for ro, rc, adv in zip(rep.ordered, rep.commuting, rep.per_seed_advantage):
        print(f"  seed {ro.seed:2d}: ord {ro.delta_total:+8.3f}"
              f"  com {rc.delta_total:+8.3f}  adv {adv:+8.3f}"
              f"  | loss ord {ro.loss_final:.4f} com {rc.loss_final:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
