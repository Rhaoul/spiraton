"""Émetteur ABA — invariant aller-retour à 3 niveaux + accord d'opérateur.

Reprend le protocole de la « boucle socle ↔ corpus » (CLAUDE.md du tokenizer) :

  1. OCTETS    — tokeniser + émettre la TOTALITÉ du corpus eve (phrases nues)
                 sans crash ni token manquant. (ASan indisponible sur mingw ;
                 ce niveau vérifie ici la robustesse fonctionnelle, pas la
                 mémoire fine — cf. docs.)
  2. GRAMMAIRE — toute ligne émise, ré-analysée par le parseur de référence
                 (spiraton/data/aba.py), redonne le même triplet
                 {op, chiralité, segments} ET respecte la clôture spirale.
  3. SÉMANTIQUE— sur dataset_aba.txt (~5000 cycles étiquetés), taux d'accord
                 entre l'opérateur ÉMIS (physique phonémique) et l'opérateur de
                 la BALISE. C'est une MESURE à rapporter, jamais une cible : on
                 ne recopie pas l'étiquette dans une heuristique.

Nécessite le tokenizer natif (`make lib`). Usage :
    PYTHONPATH=. python examples/measure_aba_emitter.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spiraton.data.aba import parse_aba_line, iter_aba_cycles, OPERATORS
from spiraton.data.tokenizer_bridge import is_available, load_native_tokenizer


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def _norm(s: str) -> str:
    return " ".join(s.split())


def level1_2_octets_grammaire(emit_aba, tok, eve_path: Path) -> None:
    n = emitted = parse_ok = closure_ok = triplet_ok = no_token = 0
    examples = []
    with open(eve_path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            n += 1
            cyc = emit_aba(s, tok)
            if cyc is None:
                no_token += 1
                continue
            emitted += 1
            try:
                parsed = parse_aba_line(cyc.line)
            except Exception:
                continue
            parse_ok += 1
            if parsed.is_closure:
                closure_ok += 1
            # Triplet : op constant attendu + textes de segments round-trip.
            same_op = parsed.op == cyc.op
            same_text = (
                _norm(parsed.seg_a.text) == _norm(cyc.seg_a)
                and _norm(parsed.seg_b.text) == _norm(cyc.seg_b)
                and _norm(parsed.seg_a_prime.text) == _norm(cyc.seg_a_prime)
            )
            if same_op and same_text:
                triplet_ok += 1
            if len(examples) < 2:
                examples.append((s, cyc.line))

    print("== Niveaux 1-2 : OCTETS + GRAMMAIRE (corpus eve, phrases nues) ==")
    print(f"  phrases={n}  émises={emitted}  sans_token={no_token}")
    print(f"  ré-analysées OK={parse_ok}/{emitted}  clôture spirale OK={closure_ok}/{emitted}"
          f"  triplet round-trip OK={triplet_ok}/{emitted}")
    for src, ln in examples:
        print(f"  ex.: {src!r}\n    → {ln}")
    print()


def level3_semantique(tok, dataset_path: Path) -> None:
    from spiraton_tokenizer.aba_emitter import dominant_operator, OPS

    def verb_operator(toks):
        """Opérateur du VERBE (token rôle=1), mappé en balise ABA si possible.

        La balise du corpus vise « le geste sémantique » = le verbe ; c'est une
        seconde lentille principielle, indépendante du résultat (on rapporte ce
        qui sort). Renvoie None si le verbe est PURE (pas de balise ABA) ou
        absent.
        """
        for t in toks:
            if t.get("role") == 1:
                o = int(t.get("operator", -1))
                return OPS[o] if 0 <= o < 4 else None
        return None

    total = 0
    agree_dom = agree_verb = verb_defined = 0
    tag_counts: Counter = Counter()
    per_op_total: Counter = Counter()
    per_op_agree_dom: Counter = Counter()
    confusion: Counter = Counter()  # (tag, émis dominant)

    for cyc in iter_aba_cycles(str(dataset_path)):
        sentence = " ".join(
            t for t in (cyc.seg_a.text, cyc.seg_b.text, cyc.seg_a_prime.text) if t
        )
        toks = tok.tokenize(sentence)
        if not toks:
            continue
        total += 1
        tag = cyc.op
        dom = dominant_operator(toks)
        verb = verb_operator(toks)

        tag_counts[tag] += 1
        per_op_total[tag] += 1
        confusion[(tag, dom)] += 1
        if dom == tag:
            agree_dom += 1
            per_op_agree_dom[tag] += 1
        if verb is not None:
            verb_defined += 1
            if verb == tag:
                agree_verb += 1

    print("== Niveau 3 : SÉMANTIQUE (dataset_aba, accord opérateur émis vs balise) ==")
    if total == 0:
        print("  (aucun cycle)")
        return
    base_op, base_n = tag_counts.most_common(1)[0]
    baseline = 100.0 * base_n / total
    print(f"  cycles={total}")
    print(f"  [lentille A] dominant (somme des scores) = {agree_dom}/{total} = "
          f"{100.0*agree_dom/total:.1f}%")
    if verb_defined:
        print(f"  [lentille B] opérateur du verbe (rôle B) = {agree_verb}/{verb_defined} = "
              f"{100.0*agree_verb/verb_defined:.1f}%  (verbe défini sur {verb_defined} cycles)")
    print(f"  référence classe majoritaire ({base_op}={base_n}) = {baseline:.1f}%")
    print("  accord par opérateur de balise [lentille A — dominant] :")
    for op in OPERATORS:
        tot = per_op_total.get(op, 0)
        ok = per_op_agree_dom.get(op, 0)
        r = (100.0 * ok / tot) if tot else 0.0
        print(f"    {op}: {ok}/{tot} = {r:.1f}%")
    print("  confusion (balise → dominant émis, top 6) :")
    for (tag, emis), c in confusion.most_common(6):
        print(f"    {tag} → {emis}: {c}")
    print("\n  Rappel : MESURE, pas cible. L'accord modeste (et le biais ADD du")
    print("  'dominant', les voyelles ADD étant partout) est un résultat honnête sur")
    print("  l'état de la physique phonémique ; on ne le 'corrige' pas en recopiant")
    print("  les étiquettes dans le code (leçon d'analyser_signature_mot).")


def main() -> None:
    if not is_available():
        print("Tokenizer natif indisponible — compiler `make lib` dans le dépôt Tokenizer.")
        return
    tok = load_native_tokenizer()  # insère le dossier python du tokenizer dans sys.path
    from spiraton_tokenizer.aba_emitter import emit_aba

    eve = _resolve("corpus_eve_clean.txt")
    dataset = _resolve("dataset_aba.txt") or _resolve("corpus_claude_aba.txt")

    if eve is not None:
        level1_2_octets_grammaire(emit_aba, tok, eve)
    else:
        print("(corpus_eve_clean.txt introuvable — niveaux 1-2 sautés)\n")

    if dataset is not None:
        level3_semantique(tok, dataset)
    else:
        print("(dataset_aba.txt introuvable — niveau 3 sauté)")


if __name__ == "__main__":
    main()
