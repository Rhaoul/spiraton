"""Tour 7 (E3) — forme géométrique ↔ chiralité ABA : pont 33D↔géométrie↔ABA.

Déploie le diagnostic ``semantic_shape_separability`` sur les corpus ABA connus :
chaque segment → vecteurs 33D (tokenizer) → oscilloscope (gain piloté dims 0-5,
courant = dims 8-22) → ``slope_logr_theta`` → AUC(slope → chiralité ABA) sur held-out
PAR CYCLE, contre CTRL-PERM (≥20 perms) + CTRL-SIG-RAND, verrou 2 (non-fuite).

Aucune étiquette n'entre dans l'entrée (verrous 1-3). Si le tokenizer natif est absent,
le diagnostic SKIPPE proprement (jamais de substitution étiquette→donnée).

Usage :
    python run_semantic_shape.py                  # corpus connus, tous les cycles
    python run_semantic_shape.py --max-cycles 300 # sous-échantillon (coût borné)
"""
import argparse
import sys

# Console Windows en cp1252 : forcer UTF-8 pour les glyphes du rapport (↔, ', …).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - flux non reconfigurable (pipe, etc.)
    pass

from spiraton.diagnostics.semantic_shape_separability import (
    run_semantic_shape_separability,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tour 7 (E3) — forme ↔ chiralité ABA")
    ap.add_argument("--max-cycles", type=int, default=None,
                    help="borne le nombre de cycles (defaut : tous)")
    ap.add_argument("--n-perms", type=int, default=20,
                    help="nombre de permutations CTRL-PERM (defaut 20)")
    ap.add_argument("--holdout-frac", type=float, default=0.3,
                    help="fraction de cycles en held-out (defaut 0.3)")
    args = ap.parse_args()

    rep = run_semantic_shape_separability(
        max_cycles=args.max_cycles,
        n_perms=args.n_perms,
        holdout_frac=args.holdout_frac,
    )
    print(rep.summary())


if __name__ == "__main__":
    main()
