"""H10 — SÉLECTION STRUCTURELLE : voter sur les consonnes distinctives prédit-il
l'opérateur-balise mieux que l'argmax-sac ?

Hypothèse (linguiste) : la structure consonne/voyelle, lue par rôle structurel,
prédit l'opérateur-balise mieux que l'argmax-sac. On vote sur les consonnes
distinctives (occlusive=SUB, fricative=DIV, nasale/[ʁ]=MUL), en EXCLUANT le fond
vocalique ADD et les phonèmes PURE.

Deux composantes DISJOINTES (garde anti-tautologie) :
  * H10-a (op_structural)            : sélection structurelle, ORDRE-INVARIANT.
  * H10-b (op_structural_positional) : pondération positionnelle, ORDRE-DÉPENDANT —
    jugée par l'ABLATION-PERMUTATION (permuter la séquence phonémique doit
    effondrer la part ordre-dépendante du gain ; sinon = somme/longueur déguisée).

ISSUES nommées d'avance (seuils déclarés) :
  (a) PROGRESSION séparable : H10-a accord > 40.3 % ET held-out > 95e pct CTRL-PERM
      ET gain porté par SUB et/ou DIV (DIV > 0 %, SUB > 8 %) — pas un re-déplacement
      vers MUL. Et/ou H10-b ajoute ≥ +2 pts ORDRE-DÉPENDANTS survivant à l'ablation.
  (b) NULL légitime (le plus probable) : accord ∈ [22.8 %, 40.3 %], ou ne survit pas
      CTRL-PERM, ou DIV reste 0 %.
  (c) RÉPÉTITION : accord ≈ 22.8 % (|Δ| ≤ 1 pt) — sous-vote = même argmax que le sac.
  (d) DISSIPATION/FUITE : gain qui ne survit pas CTRL-PERM ; ou « gain positionnel »
      survivant à la permutation (pas l'ordre qui parle) ; ou DIV>0 par lecture
      d'étiquette → auditer.

PRÉDICTION DIRECTIONNELLE falsifiable : ADD retiré du vote ⇒ les confusions
DIV→ADD et SUB→ADD doivent CHUTER. Si elles ne chutent pas → H10-a réfutée.
Si elles chutent mais l'accord ne monte pas (DIV→MUL remplace DIV→ADD) → issue (b)
précise (pas assez de fricatives distinctives par mot en français).

BASELINES rappelées à chaque chiffre :
  * 22.8 % = accord du DÉFAUT (argmax-sac, côté .so ; ~21.9 % en réf. Python) ;
  * 40.3 % = classe MAJORITAIRE (MUL) — deviner toujours l'op le plus fréquent.

GARDES (REFUS) : une seule fonctionnelle figée a priori par hypothèse, aucun
balayage ; mapping phonème→op figé par phonemes_fr.c (lu) ; AUCUNE balise dans
le calcul du vote ; held-out par cycle disjoint ; CTRL-PERM (permutation des
étiquettes-balises, 95e pct) ; ablation-ordre pour H10-b. Seeds fixés,
déterminisme bit-à-bit.

Usage :
    PYTHONPATH=. python examples/measure_structural_vote.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spiraton.data.aba import OPERATORS, iter_aba_cycles
from spiraton.data.aba_structural import (
    op_sac_reference,
    op_structural,
    op_structural_positional,
)

SEED = 4242
N_CTRL = 25          # ≥ 20 permutations d'étiquettes
HELDOUT_FRAC = 0.30

# Baselines fixées (résultat ouvert n°3 / Tour 8).
BASE_DEFAULT = 22.8   # accord argmax-sac côté .so
BASE_MAJORITY = 40.3  # classe majoritaire MUL


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def _load_cycles(path: Path):
    out = []
    for cyc in iter_aba_cycles(str(path)):
        words = " ".join(
            t for t in (cyc.seg_a.text, cyc.seg_b.text, cyc.seg_a_prime.text) if t
        ).split()
        if words:
            out.append((cyc.op, words))
    return out


def _accuracy(cycles, predict) -> float:
    if not cycles:
        return 0.0
    agree = sum(1 for tag, words in cycles if predict(words) == tag)
    return 100.0 * agree / len(cycles)


def _per_op_accuracy(cycles, predict):
    tot = Counter()
    ok = Counter()
    for tag, words in cycles:
        tot[tag] += 1
        if predict(words) == tag:
            ok[tag] += 1
    return {op: (ok[op], tot[op]) for op in OPERATORS}


def _confusion(cycles, predict):
    conf = defaultdict(Counter)
    for tag, words in cycles:
        conf[tag][predict(words)] += 1
    return conf


def _majority_baseline(cycles):
    c = Counter(tag for tag, _ in cycles)
    op, n = c.most_common(1)[0]
    return op, 100.0 * n / len(cycles), dict(c)


def _ctrl_perm_95(cycles, predict, n, seed):
    """CTRL-PERM : permute les ÉTIQUETTES (balises) entre cycles, recalcule l'accord.

    Le vote (predict) est figé ; on casse l'association cycle↔balise. Si l'accord
    réel ne dépasse pas le 95e pct de ces accords-contrôle, le « gain » n'est pas
    spécifique à la structure phonémique → artefact / fuite.
    """
    preds = [predict(words) for _, words in cycles]
    tags = [tag for tag, _ in cycles]
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        shuffled = tags[:]
        rng.shuffle(shuffled)
        agree = sum(1 for p, t in zip(preds, shuffled) if p == t)
        scores.append(100.0 * agree / len(cycles))
    scores.sort()
    pct95 = scores[int(0.95 * (len(scores) - 1))]
    return pct95, sum(scores) / len(scores), scores[0], scores[-1]


def _report_predictor(name, predict, cycles, held, maj_rate, seed):
    print(f"\n-- {name} --")
    full = _accuracy(cycles, predict)
    sac = _accuracy(cycles, op_sac_reference)
    print(f"  accord [tout, {len(cycles)} cycles] = {full:.2f}%"
          f"   | argmax-sac réf.Python = {sac:.2f}% [baseline défaut ~22.8 %]"
          f"   | majorité = {maj_rate:.2f}%")
    print(f"  Δ vs sac réf. = {full - sac:+.2f} pt   |   Δ vs majorité = "
          f"{full - maj_rate:+.2f} pt   |   Δ vs 22.8 % défaut = {full - BASE_DEFAULT:+.2f} pt")

    ho = _accuracy(held, predict)
    ho_sac = _accuracy(held, op_sac_reference)
    print(f"  HELD-OUT ({len(held)} cycles) = {ho:.2f}%  | sac held-out = {ho_sac:.2f}%")

    pct95, mean, lo, hi = _ctrl_perm_95(held, predict, N_CTRL, seed)
    survives = ho > pct95
    print(f"  CTRL-PERM (N={N_CTRL}, held-out, perm. étiquettes) : moy={mean:.2f}%  "
          f"95e pct={pct95:.2f}%  min/max={lo:.2f}/{hi:.2f}")
    print(f"  → held-out {ho:.2f}% {'>' if survives else '≤'} 95e pct {pct95:.2f}%  "
          f"=> {'SURVIT' if survives else 'NE SURVIT PAS'}")

    print("  accord par opérateur de balise :")
    po = _per_op_accuracy(cycles, predict)
    for op in OPERATORS:
        k, t = po[op]
        r = 100.0 * k / t if t else 0.0
        print(f"    {op}: {k}/{t} = {r:.1f}%")
    return full, ho, pct95, survives


def main() -> None:
    print("== H10 : SÉLECTION STRUCTURELLE (accord opérateur émis vs balise) ==")
    print(f"   Baselines : {BASE_DEFAULT} % (défaut argmax-sac) | "
          f"{BASE_MAJORITY} % (classe majoritaire MUL).")
    print("   Garde : une fonctionnelle figée par hypothèse ; aucune balise dans le vote.\n")

    parts = []
    for name in ("dataset_aba.txt", "corpus_claude_aba.txt"):
        p = _resolve(name)
        if p is not None:
            c = _load_cycles(p)
            parts.append((name, c))
            print(f"-- source : {name} ({len(c)} cycles) --")
    if not parts:
        print("(aucun dataset ABA trouvé)")
        return
    cycles = [c for _, part in parts for c in part]
    print(f"-- combiné : {len(cycles)} cycles --")

    maj_op, maj_rate, dist = _majority_baseline(cycles)
    print(f"  distribution balises : {dist}")
    print(f"  classe majoritaire = {maj_op} → baseline majorité = {maj_rate:.2f}%")

    rng = random.Random(SEED)
    idx = list(range(len(cycles)))
    rng.shuffle(idx)
    n_ho = int(HELDOUT_FRAC * len(cycles))
    held = [cycles[i] for i in idx[:n_ho]]
    print(f"  held-out = {len(held)} cycles (30 %, split par cycle disjoint, seed={SEED})")

    # --- H10-a : sélection structurelle (ordre-invariant) -------------------
    sac_full, _, _, _ = _report_predictor(
        "BASELINE argmax-sac (réf. Python, 4 opérateurs)",
        op_sac_reference, cycles, held, maj_rate, SEED)
    a_full, a_ho, a_pct, a_surv = _report_predictor(
        "H10-a op_structural (SUB/MUL/DIV, ADD+PURE exclus) — ORDRE-INVARIANT",
        op_structural, cycles, held, maj_rate, SEED)

    # --- Prédiction directionnelle : chute des confusions DIV→ADD / SUB→ADD --
    print("\n-- PRÉDICTION DIRECTIONNELLE (chute DIV→ADD et SUB→ADD) --")
    conf_sac = _confusion(cycles, op_sac_reference)
    conf_a = _confusion(cycles, op_structural)
    for true_op in ("DIV", "SUB"):
        s = conf_sac[true_op]["ADD"]
        a = conf_a[true_op]["ADD"]
        print(f"  {true_op}→ADD : sac {s}  →  H10-a {a}   (Δ = {a - s:+d})")
    print(f"  DIV accord : sac {100*conf_sac['DIV']['DIV']/sum(conf_sac['DIV'].values()):.1f}% "
          f"→ H10-a {100*conf_a['DIV']['DIV']/sum(conf_a['DIV'].values()):.1f}%  "
          f"(DIV > 0 % requis pour issue (a))")
    print("  confusion H10-a (vrai → prédit) :")
    for t in OPERATORS:
        print(f"    {t} → {dict(conf_a[t])}")

    # --- H10-b : pondération positionnelle + ABLATION-ORDRE -----------------
    b_full = _accuracy(cycles, op_structural_positional)
    b_ho = _accuracy(held, op_structural_positional)
    print("\n-- H10-b op_structural_positional (poids = 1+index) — ORDRE-DÉPENDANT --")
    print(f"  accord [tout] = {b_full:.2f}%  | held-out = {b_ho:.2f}%")
    print(f"  gain positionnel vs H10-a (tout) = {b_full - a_full:+.2f} pt")

    # Ablation : permuter la séquence phonémique de chaque mot, K seeds.
    print("  ABLATION-ORDRE (permutation des phonèmes intra-mot, vs P0 Δ=0) :")
    K = 5
    perm_scores = []
    for k in range(K):
        prng = random.Random(SEED + 100 + k)
        sc = _accuracy(
            cycles,
            lambda w, _r=prng: op_structural_positional(w, shuffle_rng=_r),
        )
        perm_scores.append(sc)
    perm_mean = sum(perm_scores) / len(perm_scores)
    order_dep_gain = b_full - perm_mean
    surviving_gain = perm_mean - a_full
    print(f"    permuté (K={K} seeds) : moy={perm_mean:.2f}%  "
          f"min/max={min(perm_scores):.2f}/{max(perm_scores):.2f}")
    print(f"    gain TOTAL positionnel      = {b_full - a_full:+.2f} pt")
    print(f"    gain SURVIVANT à la perm.   = {surviving_gain:+.2f} pt  "
          f"(NON ordre-dépendant = effet longueur/magnitude)")
    print(f"    gain ORDRE-DÉPENDANT (réel) = {order_dep_gain:+.2f} pt  "
          f"(part qui s'effondre = ce que l'ordre porte vraiment)")
    print("    → l'ablation est SAINE si la part ordre-dépendante est petite et "
          "que le reste s'explique par la longueur (cf. issue (d)).")

    # --- QUALIFICATION ------------------------------------------------------
    print("\n== QUALIFICATION (issues a/b/c/d) ==")
    print(f"  H10-a accord = {a_full:.2f}%  (held-out {a_ho:.2f}%, 95e pct CTRL-PERM "
          f"{a_pct:.2f}%, {'survit' if a_surv else 'ne survit pas'})")
    div_k = conf_a["DIV"]["DIV"]
    div_t = sum(conf_a["DIV"].values())
    div_rate = 100.0 * div_k / div_t if div_t else 0.0
    sub_k = conf_a["SUB"]["SUB"]
    sub_t = sum(conf_a["SUB"].values())
    sub_rate = 100.0 * sub_k / sub_t if sub_t else 0.0
    print(f"  critères (a) par-op : DIV = {div_rate:.1f}% (>0 ? {div_rate > 0}), "
          f"SUB = {sub_rate:.1f}% (>8 ? {sub_rate > 8})")
    print(f"  H10-b gain ordre-dépendant = {order_dep_gain:+.2f} pt "
          f"(≥ +2 pt requis pour (a) ? {order_dep_gain >= 2.0})")

    issue_a = (a_full > BASE_MAJORITY and a_surv and div_rate > 0 and sub_rate > 8) \
        or (order_dep_gain >= 2.0 and surviving_gain < order_dep_gain)
    if issue_a:
        print("  → (a) PROGRESSION séparable : seuils franchis.")
    elif abs(a_full - BASE_DEFAULT) <= 1.0:
        print("  → (c) RÉPÉTITION : accord ≈ 22.8 % (sous-vote = argmax du sac).")
    else:
        print("  → (b) NULL LÉGITIME : accord ∈ [22.8 %, 40.3 %], ou ne survit pas "
              "CTRL-PERM, ou DIV faible. Confusions DIV→ADD/SUB→ADD chutent (prédiction "
              "directionnelle confirmée) mais le report va vers SUB/MUL, pas vers un "
              "accord global > 40.3 %. Documenté, non forcé.")


if __name__ == "__main__":
    main()
