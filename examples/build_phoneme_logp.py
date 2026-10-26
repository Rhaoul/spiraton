"""Gèle p(phonème) AVANT toute balise — sérialise une table −log p réutilisable.

GARDE ANTI-FIT (i) du tour : la pondération-information de P1 exige une
distribution p(phonème) estimée SANS regarder les étiquettes ABA. Ce script
compte les phonèmes produits par la référence Python (``data/phoneme_ref``) sur
le CORPUS (texte nu, balises retirées), normalise, et écrit ``phoneme_logp.json``.

DEUX MODES de comptage, pour le test d'ablation anti-fuite (issue d de P1) :
  --source corpus_text  : compte sur le texte des cycles, BALISES RETIRÉES (défaut,
                          légitime : aucune étiquette consultée).
  --source eve          : compte sur corpus_eve_clean.txt (phrases nues, encore plus
                          clairement sans balise).

Si P1 ne gagne QUE quand p est estimée d'une manière qui voit l'étiquette, p a
fuité — mais ici p ne voit JAMAIS l'étiquette par construction (on lit le texte,
pas l'opérateur de la balise). L'ablation (d) est réalisée dans le diagnostic en
comparant aux deux tables.

Usage :
    PYTHONPATH=. python examples/build_phoneme_logp.py --source corpus_text
    PYTHONPATH=. python examples/build_phoneme_logp.py --source eve
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spiraton.data.aba import iter_aba_cycles
from spiraton.data.phoneme_ref import g2p_heuristic


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def _count_from_corpus_text(dataset: Path) -> Counter:
    """Compte les phonèmes sur le TEXTE des cycles (balises retirées via le parseur).

    On consomme uniquement ``seg.text`` (contenu lexical nettoyé) — l'opérateur de
    la balise (``cyc.op``) n'est JAMAIS lu. p ne voit donc pas l'étiquette.
    """
    counts: Counter = Counter()
    for cyc in iter_aba_cycles(str(dataset)):
        for seg in (cyc.seg_a, cyc.seg_b, cyc.seg_a_prime):
            for word in seg.text.split():
                counts.update(g2p_heuristic(word))
    return counts


def _count_from_eve(eve: Path) -> Counter:
    counts: Counter = Counter()
    with open(eve, "r", encoding="utf-8") as fh:
        for line in fh:
            for word in line.split():
                counts.update(g2p_heuristic(word))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("corpus_text", "eve"), default="corpus_text")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.source == "corpus_text":
        dataset = _resolve("dataset_aba.txt") or _resolve("corpus_claude_aba.txt")
        if dataset is None:
            print("(dataset ABA introuvable)")
            return
        counts = _count_from_corpus_text(dataset)
        src_name = dataset.name + " [text, balises retirées]"
    else:
        eve = _resolve("corpus_eve_clean.txt")
        if eve is None:
            print("(corpus_eve_clean.txt introuvable)")
            return
        counts = _count_from_eve(eve)
        src_name = eve.name

    total = sum(counts.values())
    if total == 0:
        print("(aucun phonème compté)")
        return

    # p(phonème) = freq relative ; −log p = surprise (information de Shannon, nats).
    logp = {ph: -math.log(c / total) for ph, c in counts.items()}

    out = _resolve_out(args.out, args.source)
    payload = {
        "source": src_name,
        "total_phonemes": total,
        "counts": dict(counts),
        "neg_log_p": logp,
        "note": "p gelée AVANT toute balise ; aucune étiquette ABA consultée.",
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"== p(phonème) gelée depuis {src_name} ==")
    print(f"  phonèmes comptés (total occurrences) = {total}")
    print(f"  vocabulaire = {len(counts)} symboles")
    print("  top 8 (fréquents → −log p faible) :")
    for ph, c in counts.most_common(8):
        print(f"    {ph!r:6s} n={c:6d}  p={c/total:.4f}  −logp={logp[ph]:.3f}")
    print("  rares (−log p élevé, distinctifs) — 8 derniers :")
    for ph, c in counts.most_common()[-8:]:
        print(f"    {ph!r:6s} n={c:6d}  p={c/total:.4f}  −logp={logp[ph]:.3f}")
    print(f"\n  → écrit : {out}")


def _resolve_out(out_arg, source) -> Path:
    if out_arg:
        return Path(out_arg)
    repo = Path(__file__).resolve().parents[1]
    return repo / f"phoneme_logp_{source}.json"


if __name__ == "__main__":
    main()
