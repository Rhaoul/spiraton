"""Mesure empirique de la double dynamique L∘D vs D∘L (chantier 2).

Applique, sur un même SpiralGrid (même cellule, même projection), les deux
compositions temporelles du Logos :

  - L∘D = outward (expansion D) puis inward (contraction L) → prédite stable
  - D∘L = inward (L) puis outward (D)                       → prédite bifurcante

On rapporte la variance temporelle du signal alpha-oméga (cos − l2) pour
chacune, et l'on agrège sur plusieurs graines pour voir si l'asymétrie prédite
émerge. C'est une MESURE : le résultat est ce qu'il est.

Usage : python examples/double_dynamics.py
"""
from __future__ import annotations

import sys

import torch

# Le projet emploie des symboles Unicode (∘, alpha/oméga) ; sur les consoles
# Windows (cp1252) il faut forcer l'UTF-8 pour éviter UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spiraton.core.cell import SpiratonCell
from spiraton.grid import SpiralGrid
from spiraton.diagnostics import run_double_dynamics


def build_grid(channels: int, seed: int) -> SpiralGrid:
    torch.manual_seed(seed)
    cell = SpiratonCell(input_size=2 * channels)
    return SpiralGrid(cell, channels=channels, neighborhood="von_neumann", aggregator="mean")


def main() -> None:
    B, H, W, C, K = 4, 7, 7, 6, 8
    n_seeds = 12

    agree = 0
    gaps = []
    print(f"Double dynamique — B={B} grille {H}x{W} C={C} K={K}, {n_seeds} graines\n")
    for seed in range(n_seeds):
        grid = build_grid(C, seed)
        torch.manual_seed(1000 + seed)
        x0 = torch.randn(B, H, W, C)
        rep = run_double_dynamics(grid, x0, k=K)
        gaps.append(rep.variance_gap)
        if rep.stabilizes_as_predicted:
            agree += 1
        flag = "✓" if rep.stabilizes_as_predicted else "·"
        print(
            f"  seed {seed:2d} {flag}  var(L∘D)={rep.ld_signal_variance:.4g}  "
            f"var(D∘L)={rep.dl_signal_variance:.4g}  gap={rep.variance_gap:+.4g}  "
            f"final_l2 L∘D={rep.ld_final_l2:.3g} D∘L={rep.dl_final_l2:.3g}"
        )

    mean_gap = sum(gaps) / len(gaps)
    print(
        f"\nRésumé : {agree}/{n_seeds} graines conformes à la théorie "
        f"(var L∘D < var D∘L). gap moyen (dl-ld) = {mean_gap:+.4g}"
    )
    if agree > n_seeds * 0.6:
        print("→ Asymétrie majoritairement conforme : L∘D tend à stabiliser.")
    elif agree < n_seeds * 0.4:
        print("→ Asymétrie inversée ou absente : résultat à consigner tel quel.")
    else:
        print("→ Pas d'asymétrie nette à cette échelle : résultat à consigner.")


if __name__ == "__main__":
    main()
