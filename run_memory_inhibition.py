"""Tour 2 — balayage du terme de mémoire ``−C(s_{t−1})`` (axiome §3.2, second ordre).

Incarne l'émission du linguiste : faire varier la magnitude γ=‖C‖, la structure,
et le signe de l'inhibition de durée, et MESURER le retour (stabilité + α-ω cos−l2).

Lance :  python run_memory_inhibition.py
Tout est seedé ; chiffres bruts reproductibles. bounded=False (anti-artefact).
"""

import sys

import torch

# Console Windows souvent en cp1252 : forcer UTF-8 pour les symboles (ρ, γ, …).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from spiraton.diagnostics.memory_inhibition_scan import run_memory_inhibition_scan

torch.manual_seed(0)

SEEDS = tuple(range(24))
D = 8
STEPS = 50

# init_scale=1.2 : régime à MARGE (≈14/24 divergent déjà à γ=0, bounded=False).
# C'est le seul régime où H1 est testable : à init_scale≤0.5 rien ne diverge
# (rien à stabiliser) ; à init_scale≥1.5 tout diverge (saturé, pas de marge).
report = run_memory_inhibition_scan(
    d=D,
    steps=STEPS,
    seeds=SEEDS,
    gamma_ratios=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    structures=("dense", "diag+", "diag±", "anti-sym"),
    bounded=False,
    init_scale=1.2,
)

print(report.summary())

print("\n" + "=" * 78)
print("LECTURE (verdict préliminaire — l'ingénieur tranche et valide le canon)")
print("=" * 78)
rho = report.spearman_gamma_diverged
p = report.spearman_pvalue
mono = (rho < -0.6) and (p < 0.05)
drop_ok = report.diverged_drop_pts >= 30.0
print(f"  H1 stabilisation monotone : ρ_s={rho:.3f} (cible <−0.6), p≈{p:.3g} (cible <0.05)"
      f"  → {'CONFIRMÉE' if mono else 'NON confirmée'}")
print(f"  H1 chute ≥30 pts diverged : {report.diverged_drop_pts:.1f} pts"
      f"  → {'OUI' if drop_ok else 'NON'}")
print(f"  H2 cloche best_return     : max à γ={report.bell_argmax_gamma:.3g}·ρ(A),"
      f" intérieur={'oui' if report.bell_is_interior else 'non'}"
      f"  → {'cloche' if report.bell_is_interior else 'monotone/plat'}")
