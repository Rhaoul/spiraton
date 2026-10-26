"""Enregistre des vecteurs 33D témoins pour les tests de parité (chantier 3).

À exécuter sur une machine où le tokenizer natif est compilé et chargeable
(``make lib`` dans le dépôt tokenizer, puis éventuellement
``SPIRATON_TOKENIZER_LIB`` / ``SPIRATON_TOKENIZER_PY``). Produit
``tests/fixtures/parity_33d.json`` : pour quelques mots témoins, le premier
vecteur 33D produit. Le test ``test_native_parity_against_fixtures`` compare
ensuite la sortie courante à ces valeurs figées pour attraper toute dérive
d'ABI (l'incident de Phase 16).

NE PAS éditer le JSON à la main : il est, par construction, la photographie
d'un état natif validé. Le régénérer après tout changement *justifié* du socle.

Usage : python scripts/record_33d_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spiraton.data.tokenizer_bridge import NativeTokenizer33D, TokenizerUnavailable

# Mots témoins : couvrent les contrastes physiques connus (Joie/Mort) et
# quelques structures syllabiques variées.
WITNESS_WORDS = ["Joie", "Mort", "Bonjour", "spirale", "amour", "rien"]

OUT = ROOT / "tests" / "fixtures" / "parity_33d.json"


def main() -> int:
    try:
        tok = NativeTokenizer33D()
    except TokenizerUnavailable as exc:
        print(f"Tokenizer natif indisponible : {exc}", file=sys.stderr)
        print("Impossible d'enregistrer les fixtures sans le .so/.dll.", file=sys.stderr)
        return 1

    fixtures = {}
    for w in WITNESS_WORDS:
        vecs = tok.vectors(w)
        if vecs.shape[0] == 0:
            print(f"  (avertissement) aucun token pour {w!r}, ignoré")
            continue
        fixtures[w] = [round(float(x), 6) for x in vecs[0].tolist()]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Écrit {OUT} ({len(fixtures)} mots témoins).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
