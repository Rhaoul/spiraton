"""Constat chiffré de conformité des corpus ABA (chantier 3).

Passe chaque corpus connu dans le parseur de référence et rapporte : nombre de
cycles, taux de conformité au canon, lignes illisibles, terminateurs, points
fixes (répétitions), et distribution des opérateurs. C'est une MESURE — jamais
une cible à atteindre par des heuristiques codées en dur.

Les corpus sont attendus à la racine de l'écosystème (un cran au-dessus du
dépôt spiraton). On essaie aussi ``data/`` et le dossier courant.

Usage : python examples/measure_aba_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spiraton.data.aba import measure_corpus

CORPORA = ("corpus_claude_aba.txt", "dataset_aba.txt", "corpus_eve_clean.txt")


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]   # .../spiraton
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def main() -> None:
    print("Conformité des corpus ABA (parseur de référence)\n")
    for name in CORPORA:
        p = _resolve(name)
        if p is None:
            print(f"  {name}: introuvable (placer le corpus à la racine de l'écosystème)")
            continue
        # corpus_eve_clean.txt n'est PAS au format ABA (phrases brutes) :
        # le parseur le rapportera comme 0 cycle / lignes illisibles — c'est
        # attendu, c'est la genèse pédagogique en langue naturelle.
        try:
            print("  " + measure_corpus(str(p)).summary())
        except Exception as exc:  # pragma: no cover
            print(f"  {name}: erreur de lecture ({exc})")


if __name__ == "__main__":
    main()
