"""Benchmark de stabilité du ChronoSpiraton (chantier 6).

L'équation du second ordre s_{t+1} = D(A(s)+B(s²)−C(s_{t−1}))+L(s) est non
bornée : selon l'échelle des opérateurs, elle peut converger, osciller ou
diverger (chaos déterministe). Ce script balaie l'échelle d'initialisation et
rapporte, pour chaque régime, la norme finale, la norme max et la divergence —
en mode pur puis en mode borné (tanh terminal).

Usage : python examples/chrono_stability.py
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from spiraton.experimental.chrono import ChronoSpiraton


def scan(bounded: bool, d: int = 16, steps: int = 100, n_seeds: int = 8) -> None:
    label = "borné (tanh)" if bounded else "pur (équation brute)"
    print(f"\n=== Mode {label} — d={d}, {steps} pas, {n_seeds} graines ===")
    # Balayage large : nn.Linear divise déjà par √fan_in, donc l'échelle
    # effective des opérateurs ne devient ~1 que vers init_scale≈√d.
    for scale in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        n_div = 0
        finals = []
        maxes = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            chrono = ChronoSpiraton(state_size=d, init_scale=scale, bounded=bounded)
            s0 = torch.randn(1, d) * 0.5
            rep = chrono.stability_scan(s0, steps=steps)
            n_div += int(rep["diverged"])
            if not rep["diverged"]:
                finals.append(rep["final_norm"])
                maxes.append(rep["max_norm"])
        mean_final = sum(finals) / len(finals) if finals else float("nan")
        mean_max = sum(maxes) / len(maxes) if maxes else float("nan")
        print(
            f"  init_scale={scale:<4}  divergences={n_div}/{n_seeds}  "
            f"norme finale moy.={mean_final:.3g}  norme max moy.={mean_max:.3g}"
        )


def main() -> None:
    print("Benchmark de stabilité ChronoSpiraton (équation du second ordre)")
    scan(bounded=False)
    scan(bounded=True)
    print(
        "\nLecture : en mode pur, la divergence apparaît quand l'échelle des "
        "opérateurs croît (le terme quadratique B(s²) domine) ; le mode borné "
        "la contient au prix d'une fidélité moindre à l'équation."
    )


if __name__ == "__main__":
    main()
