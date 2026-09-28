"""H8-P1 — PONDÉRATION-INFORMATION : −log p(phonème) fait-il monter l'accord ?

Hypothèse (linguiste) : pondérer chaque vote phonémique par −log p (p gelée AVANT
les balises) plutôt que par sa présence fait monter l'accord au-dessus de 40.3 %,
parce que les consonnes rares distinctives (SUB/DIV) cessent d'être noyées par les
voyelles (ADD). Geste = DIV/lévogyre (distinction).

Issues :
  (a) PROGRESSION = accord > 40.3 % ET held-out > 95e pct CTRL-PERM ET le gain
      vient des cycles SUB/DIV (pas d'un re-déplacement du biais).
  (b) NULL légitime = gain ∈ ]22.8 %, 40.3 %[ ou non survivant en held-out.
  (c) RÉPÉTITION = ≈ 22.8 % inchangé.
  (d) DISSIPATION/FUITE = monte mais s'effondre sous CTRL-PERM, OU le gain
      disparaît si on coupe l'accès aux balises lors du calcul de p.

BASELINES (rappelées à chaque chiffre) :
  * 22.8 %  = accord du DÉFAUT (vote par présence, dominant ; mesuré ici aussi en
              version réf. Python pour comparabilité granulaire) ;
  * 40.3 %  = référence classe MAJORITAIRE (deviner toujours l'op le plus fréquent).

GARDES ANTI-FIT respectées :
  (i)  p gelée AVANT balises (build_phoneme_logp.py ; on charge le .json).
  (ii) UNE SEULE fonctionnelle : −log p (déclarée d'avance, aucun balayage).
  Ablation (d) : on compare DEUX tables p (corpus_text / eve) ET le poids uniforme.
  Held-out : split par cycle (la table p ne dépend d'aucun split ; elle est globale
  et sans balise — le held-out teste la généralisation de la RÈGLE de vote, pas un
  ajustement de paramètres, qu'il n'y en a pas).

CTRL-PERM : ≥20 tables p de CONTRÔLE obtenues en permutant aléatoirement
l'association {phonème → −log p} (on garde le même multiset de poids, mais on les
ré-assigne au hasard aux phonèmes). Si l'accord informé réel ne dépasse pas le 95e
percentile de ces contrôles, le gain n'est pas dû à l'information phonémique
spécifique mais à n'importe quelle re-pondération → FUITE/artefact.

Usage :
    PYTHONPATH=. python examples/build_phoneme_logp.py --source corpus_text
    PYTHONPATH=. python examples/build_phoneme_logp.py --source eve
    PYTHONPATH=. python examples/measure_informed_vote.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spiraton.data.aba import iter_aba_cycles, OPERATORS
from spiraton.data.aba_informed import (
    dominant_operator_informed,
    load_logp_table,
    per_op_weight_mass,
)
from spiraton.data.phoneme_ref import OPS
from spiraton.data.tokenizer_bridge import is_available, load_native_tokenizer

SEED = 4242
N_CTRL = 25            # ≥ 20 contrôles de permutation
HELDOUT_FRAC = 0.30


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def _load_cycles(dataset: Path):
    out = []
    for cyc in iter_aba_cycles(str(dataset)):
        words = " ".join(
            t for t in (cyc.seg_a.text, cyc.seg_b.text, cyc.seg_a_prime.text) if t
        ).split()
        if words:
            out.append((cyc.op, words))
    return out


def _accuracy(cycles, logp) -> float:
    if not cycles:
        return 0.0
    agree = sum(1 for tag, words in cycles
                if dominant_operator_informed(words, logp) == tag)
    return 100.0 * agree / len(cycles)


def _per_op_accuracy(cycles, logp):
    tot = Counter()
    ok = Counter()
    for tag, words in cycles:
        tot[tag] += 1
        if dominant_operator_informed(words, logp) == tag:
            ok[tag] += 1
    return {op: (ok[op], tot[op]) for op in OPERATORS}


def _majority_baseline(cycles):
    c = Counter(tag for tag, _ in cycles)
    op, n = c.most_common(1)[0]
    return op, 100.0 * n / len(cycles), dict(c)


def _control_permuted_tables(logp, n, seed):
    """n tables de contrôle : mêmes valeurs −log p, ré-assignées au hasard."""
    phonemes = list(logp.keys())
    values = list(logp.values())
    tables = []
    rng = random.Random(seed)
    for _ in range(n):
        shuffled = values[:]
        rng.shuffle(shuffled)
        tables.append(dict(zip(phonemes, shuffled)))
    return tables


def _native_parity_rate(cycles, tok, logp) -> float:
    """Taux d'accord entre le dominant réf. Python (informé OFF) et le .so.

    Garde-fou de non-dérive : confirme que la réf. Python (sur laquelle P1 calcule)
    reproduit le vote du socle. p=None => vote par présence, comparable au .so.
    """
    from spiraton_tokenizer.aba_emitter import dominant_operator
    same = 0
    n = 0
    for tag, words in cycles:
        toks = tok.tokenize(" ".join(words))
        if not toks:
            continue
        n += 1
        if dominant_operator_informed(words, None) == dominant_operator(toks):
            same += 1
    return 100.0 * same / n if n else 0.0


def _run_one_table(name, logp, cycles, train, held, maj_rate, seed):
    print(f"\n-- table p : {name} --")
    full = _accuracy(cycles, logp)
    presence = _accuracy(cycles, None)  # défaut réf. Python (poids uniforme)
    print(f"  accord DÉFAUT réf. Python (présence, poids=1) = {presence:.2f}%"
          f"   [baseline défaut ~22.8 % côté .so]")
    print(f"  accord INFORMÉ (−log p) [tout le dataset]    = {full:.2f}%")
    print(f"  gain informé − défaut(réf.) = {full - presence:+.2f} pt   |   "
          f"vs majorité {maj_rate:.1f}% : {full - maj_rate:+.2f} pt")

    # Held-out
    ho_inf = _accuracy(held, logp)
    ho_pres = _accuracy(held, None)
    print(f"  HELD-OUT ({len(held)} cycles) informé = {ho_inf:.2f}%  | "
          f"défaut = {ho_pres:.2f}%  | gain = {ho_inf - ho_pres:+.2f} pt")

    # CTRL-PERM sur held-out
    ctrl_tables = _control_permuted_tables(logp, N_CTRL, seed)
    ctrl_scores = sorted(_accuracy(held, ct) for ct in ctrl_tables)
    pct95 = ctrl_scores[int(0.95 * (len(ctrl_scores) - 1))]
    ctrl_mean = sum(ctrl_scores) / len(ctrl_scores)
    survives = ho_inf > pct95
    print(f"  CTRL-PERM (N={N_CTRL}, held-out) : moy={ctrl_mean:.2f}%  "
          f"95e pct={pct95:.2f}%  min/max={ctrl_scores[0]:.2f}/{ctrl_scores[-1]:.2f}")
    print(f"  → informé held-out {ho_inf:.2f}% {'>' if survives else '≤'} "
          f"95e pct CTRL-PERM {pct95:.2f}%  "
          f"=> {'SURVIT' if survives else 'NE SURVIT PAS'}")

    # Accord par opérateur de balise : le gain vient-il de SUB/DIV ?
    print("  accord par opérateur de balise (informé | défaut) :")
    inf_po = _per_op_accuracy(cycles, logp)
    pre_po = _per_op_accuracy(cycles, None)
    for op in OPERATORS:
        ik, it = inf_po[op]
        pk, pt = pre_po[op]
        ir = 100.0 * ik / it if it else 0.0
        pr = 100.0 * pk / pt if pt else 0.0
        print(f"    {op}: informé {ik}/{it}={ir:.1f}%  | défaut {pk}/{pt}={pr:.1f}%  "
              f"| Δ={ir - pr:+.1f} pt")
    return full, presence, ho_inf, pct95, survives


def main() -> None:
    print("== H8-P1 : PONDÉRATION-INFORMATION (accord opérateur émis vs balise) ==")
    print("   Baselines : 22.8 % (défaut présence) | 40.3 % (classe majoritaire).")
    print("   Garde : p gelée AVANT balises ; une seule fonctionnelle (−log p).\n")

    dataset = _resolve("dataset_aba.txt") or _resolve("corpus_claude_aba.txt")
    if dataset is None:
        print("(dataset ABA introuvable)")
        return
    cycles = _load_cycles(dataset)
    print(f"-- dataset : {dataset.name} ({len(cycles)} cycles) --")

    maj_op, maj_rate, dist = _majority_baseline(cycles)
    print(f"  distribution balises : {dist}")
    print(f"  classe majoritaire = {maj_op} → baseline majorité = {maj_rate:.2f}%")

    # Split held-out par cycle (seedé).
    rng = random.Random(SEED)
    idx = list(range(len(cycles)))
    rng.shuffle(idx)
    n_ho = int(HELDOUT_FRAC * len(cycles))
    held = [cycles[i] for i in idx[:n_ho]]
    train = [cycles[i] for i in idx[n_ho:]]

    # Parité réf. Python ↔ .so (non-dérive)
    if is_available():
        tok = load_native_tokenizer()
        par = _native_parity_rate(cycles[:1500], tok, None)
        print(f"  parité dominant réf.Python ↔ .so (1500 cycles, mode oracle .so) = "
              f"{par:.1f}%")
        print("  (écart = oracle dico + boost analyser_signature_mot non reproduits ;")
        print("   le test P1 reste cohérent — il compare informé vs présence SUR la réf.)")
    else:
        print("  (.so indisponible — parité non mesurée ; P1 reste cohérent sur réf. Python)")

    p_corpus = _resolve("phoneme_logp_corpus_text.json")
    p_eve = _resolve("phoneme_logp_eve.json")
    if p_corpus is None or p_eve is None:
        print("\n  Tables p manquantes — lancer d'abord build_phoneme_logp.py "
              "(--source corpus_text ET --source eve).")
        return

    results = {}
    results["corpus_text"] = _run_one_table(
        "corpus_text (texte des cycles, balises retirées)",
        load_logp_table(str(p_corpus)), cycles, train, held, maj_rate, SEED)
    results["eve"] = _run_one_table(
        "eve (corpus_eve_clean, sans balise)",
        load_logp_table(str(p_eve)), cycles, train, held, maj_rate, SEED + 1)

    # --- Qualification globale ----------------------------------------------
    print("\n== QUALIFICATION (issues a/b/c/d) ==")
    full_c, pres_c, ho_c, pct_c, surv_c = results["corpus_text"]
    full_e, pres_e, ho_e, pct_e, surv_e = results["eve"]
    print(f"  défaut(réf.) ≈ {pres_c:.1f}%  | majorité = {maj_rate:.1f}%")
    print(f"  informé corpus_text = {full_c:.1f}% (held-out {ho_c:.1f}%, "
          f"{'survit' if surv_c else 'ne survit pas'})")
    print(f"  informé eve         = {full_e:.1f}% (held-out {ho_e:.1f}%, "
          f"{'survit' if surv_e else 'ne survit pas'})")
    # Ablation anti-fuite (d) : robustesse au choix de la source de p.
    robust = abs(full_c - full_e) < 2.0
    print(f"  ablation source-de-p (d) : |Δ corpus_text vs eve| = "
          f"{abs(full_c - full_e):.2f} pt → {'ROBUSTE' if robust else 'SENSIBLE (suspect)'}")

    best = max(full_c, full_e)
    best_surv = surv_c if full_c >= full_e else surv_e
    if best > maj_rate and best_surv:
        print("  → (a) PROGRESSION candidate : > 40.3 % ET survit CTRL-PERM. "
              "Vérifier le tableau par-opérateur (gain réel sur SUB/DIV ?).")
    elif pres_c - 1.0 <= best <= maj_rate:
        print("  → (b) NULL LÉGITIME : gain ∈ ]défaut, majorité[ ou non survivant. "
              "Le biais reste structurel-phonémique. Documenté, non forcé.")
    elif abs(best - pres_c) <= 1.0:
        print("  → (c) RÉPÉTITION : ≈ défaut inchangé.")
    else:
        print("  → état intermédiaire : voir held-out + CTRL-PERM ci-dessus.")


if __name__ == "__main__":
    main()
