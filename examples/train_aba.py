"""Entraînement minimal sur un corpus ABA réel sous perte alpha-oméga (chantier 5).

Charge un corpus ABA (par défaut ``dataset_aba.txt`` à la racine de
l'écosystème), le vectorise avec le featurizer de substitution, puis entraîne un
prédicteur A,B → A′ sous la perte de clôture spirale (proche-et-aligné avec A,
mais non identique). Affiche la courbe de perte et les métriques.

NB : le featurizer est un SUBSTITUT (hachage), pas le tokenizer 33D. La courbe
montre que le *signal* alpha-oméga est apprenable ; brancher le tokenizer natif
pour des traits porteurs de la physique articulatoire.

Usage : python examples/train_aba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from spiraton.data.featurizers import HashingFeaturizer
from spiraton.data.loader import load_aba_triples
from spiraton.training import train_aba, make_aba_predictor


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def main() -> None:
    torch.manual_seed(0)
    corpus = _resolve("dataset_aba.txt") or _resolve("corpus_claude_aba.txt")
    if corpus is None:
        print("Aucun corpus ABA trouvé à la racine de l'écosystème.")
        return

    dim = 33  # même rang que le 33D, pour faciliter le futur branchement natif
    feat = HashingFeaturizer(dim=dim)
    triples = load_aba_triples(str(corpus), feat, limit=2000, skip_fixed_points=True)
    print(f"Corpus : {corpus.name} — {len(triples)} cycles, dim={triples.dim}, "
          f"featurizer_substitut={triples.featurizer_is_fallback}\n")

    model = make_aba_predictor(dim)
    report = train_aba(
        model, triples, epochs=80, lr=1e-2, batch_size=256, target_dist=0.3
    )

    hist = report.loss_history
    print(f"Perte : {hist[0]:.4f} → {hist[-1]:.4f}  (amélioration={report.improved})")
    print("Métriques finales :")
    for k, v in report.final_metrics.items():
        print(f"  {k:>14s} = {v:.4f}")
    m = report.final_metrics
    print(
        f"\nLecture : alignement≈{m['align']:.3f} (0=aligné), "
        f"distance relative≈{m['mean_rel_dist']:.3f} (cible {m['target_dist']:.2f}), "
        f"taux de copie={m['copy_rate']:.2f}."
    )


if __name__ == "__main__":
    main()
