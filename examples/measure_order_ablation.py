"""H8-P0 — ABLATION D'ORDRE : permuter l'ordre des phonèmes change-t-il l'accord ?

Hypothèse (linguiste) : permuter l'ordre des phonèmes de chaque mot ne change pas
l'accord opérateur-émis-vs-balise (Δ≈0), parce que le vote est context-free et la
somme commutative. Issues :
  (a) |Δ| ≤ 1 pt  → RÉPÉTITION confirmée, baseline d'ordre établie ;
  (b) Δ > 1 pt    → le socle n'était pas si commutatif (G2P/syllabification
                    réintroduit de l'ordre) = résultat ;
  (c) Δ chaotique selon seed → DISSIPATION, isoler la cause.

DEUX NIVEAUX d'injection (cf. compte-rendu pour le verdict de propreté) :

  NIVEAU-1 (le plus propre, mais HORS du .so) : permuter la séquence de phonèmes
    POST-G2P, pré-scoring. Le point d'entrée C n'existe PAS (seule
    ``tokeniser_texte(texte_graphémique)`` est exposée ; le buffer de phonèmes est
    local à ``pseudo_g2p``). On le réalise donc sur la RÉFÉRENCE PYTHON du G2P
    (``data/phoneme_ref``), validée par parité contre le ``.so`` (mode heuristique).
    Ici, Δ teste réellement que le vote par phonème est context-free.

  NIVEAU-2 (repli faisable, le pipeline .so RE-TOURNE) : permuter les GRAPHÈMES du
    texte d'entrée avant ``tok.tokenize``. RÉSERVE : mélanger des graphèmes corrompt
    souvent le G2P (phonèmes différents, pas réordonnés) → artefact. GARDE-FOU
    obligatoire : ne garder que les permutations qui PRÉSERVENT LE MULTISET
    phonémique (mesuré via la réf. Python) ; on rapporte le taux de rejet.

Seeds fixés. Chiffres bruts. La baseline = accord réel (non permuté).

Usage :  PYTHONPATH=. python examples/measure_order_ablation.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from spiraton.data.aba import iter_aba_cycles, OPERATORS
from spiraton.data.phoneme_ref import (
    OPS,
    g2p_heuristic,
    phoneme_ops_for_word,
    active_op_counts,
)
from spiraton.data.tokenizer_bridge import is_available, load_native_tokenizer

K_PERMS = 5          # permutations seedées par configuration
SEED = 12345
MAX_CYCLES = 0       # 0 = tous


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def _sentence(cyc) -> str:
    return " ".join(t for t in (cyc.seg_a.text, cyc.seg_b.text, cyc.seg_a_prime.text) if t)


# ----------------------------------------------------------------------------
# Dominant opérateur À PARTIR DE LA RÉFÉRENCE PYTHON (granularité phonème).
# C'est l'unité où NIVEAU-1 a un sens : on somme les votes phonémiques actifs sur
# tous les mots, puis argmax. (Vote context-free => invariant par permutation
# intra-mot par CONSTRUCTION ; le script le VÉRIFIE empiriquement, pas le suppose.)
# ----------------------------------------------------------------------------

def _dominant_pyref(words, perm_rng=None) -> str:
    counts = [0, 0, 0, 0]
    for w in words:
        ops = phoneme_ops_for_word(w)
        if perm_rng is not None:
            ops = ops[:]
            perm_rng.shuffle(ops)
        c = active_op_counts(ops)
        for i in range(4):
            counts[i] += c[i]
    if sum(counts) == 0:
        return "ADD"
    return OPS[max(range(4), key=lambda i: counts[i])]


def level1_python_postg2p(cycles) -> None:
    """NIVEAU-1 : permuter les phonèmes post-G2P sur la réf. Python."""
    rng = random.Random(SEED)
    total = 0
    agree_real = 0
    # accord permuté : moyenne sur K permutations indépendantes
    agree_perm_runs = [0] * K_PERMS

    for cyc in cycles:
        words = _sentence(cyc).split()
        if not words:
            continue
        total += 1
        tag = cyc.op
        if _dominant_pyref(words) == tag:
            agree_real += 1
        for k in range(K_PERMS):
            sub = random.Random(SEED * 1000 + k * 7 + total)
            if _dominant_pyref(words, perm_rng=sub) == tag:
                agree_perm_runs[k] += 1

    if total == 0:
        print("  (aucun cycle)")
        return
    real = 100.0 * agree_real / total
    perms = [100.0 * a / total for a in agree_perm_runs]
    mean_perm = sum(perms) / len(perms)
    print(f"  cycles={total}")
    print(f"  accord RÉEL (réf. Python, ordre intact) = {real:.2f}%   [BASELINE D'ORDRE]")
    print(f"  accord PERMUTÉ (K={K_PERMS} perms intra-mot) = "
          f"{['%.2f' % p for p in perms]} %  moyenne={mean_perm:.2f}%")
    print(f"  Δ (réel − moy. permuté) = {real - mean_perm:+.2f} pt   "
          f"(min/max permuté : {min(perms):.2f}/{max(perms):.2f})")
    spread = max(perms) - min(perms)
    print(f"  dispersion inter-seed des perms = {spread:.2f} pt")
    _qualify(real, mean_perm, spread)


def level2_native_grapheme(cycles, tok) -> None:
    """NIVEAU-2 : permuter les graphèmes du texte d'entrée, pipeline .so re-tournant.

    Garde-fou : ne compter une permutation que si elle PRÉSERVE le multiset
    phonémique du mot (réf. Python). Sinon rejet (G2P corrompu = artefact).
    On rapporte le taux de rejet.
    """
    from spiraton_tokenizer.aba_emitter import dominant_operator

    total = 0
    agree_real = 0
    agree_perm = 0
    words_total = 0          # mots multi-graphèmes candidats à permutation
    words_kept = 0           # une permutation multiset-préservante trouvée
    words_rejected = 0       # aucune permutation valide en K essais

    for cyc in cycles:
        sentence = _sentence(cyc)
        words = sentence.split()
        if not words:
            continue
        toks = tok.tokenize(sentence)
        if not toks:
            continue
        total += 1
        tag = cyc.op
        if dominant_operator(toks) == tag:
            agree_real += 1

        # Une permutation par mot (K essais pour en trouver une multiset-valide).
        rng = random.Random(SEED * 31 + total)
        permuted_words = []
        for w in words:
            if len(w) < 2:
                permuted_words.append(w)
                continue
            words_total += 1
            ms_orig = Counter(g2p_heuristic(w))
            chosen = w
            found = False
            for _attempt in range(K_PERMS):
                chars = list(w)
                rng.shuffle(chars)
                cand = "".join(chars)
                if cand != w and Counter(g2p_heuristic(cand)) == ms_orig:
                    chosen = cand
                    found = True
                    break
            if found:
                words_kept += 1
            else:
                words_rejected += 1  # garde le mot intact (pas de perm valide)
            permuted_words.append(chosen)

        ptoks = tok.tokenize(" ".join(permuted_words))
        if ptoks and dominant_operator(ptoks) == tag:
            agree_perm += 1

    if total == 0:
        print("  (aucun cycle)")
        return
    real = 100.0 * agree_real / total
    perm = 100.0 * agree_perm / total
    rej_rate = 100.0 * words_rejected / max(1, words_total)
    print(f"  cycles={total}")
    print(f"  accord RÉEL (.so, graphèmes intacts) = {real:.2f}%   [BASELINE D'ORDRE]")
    print(f"  accord PERMUTÉ (.so, graphèmes mélangés, multiset-préservé qd possible) = "
          f"{perm:.2f}%")
    print(f"  Δ (réel − permuté) = {real - perm:+.2f} pt")
    print(f"  mots multi-graphèmes = {words_total} ; permutés (multiset OK) = {words_kept} ; "
          f"laissés intacts (aucune perm valide) = {words_rejected} → TAUX DE REJET = {rej_rate:.1f}%")
    print("  NB : le multiset phonémique est rarement préservable en mélangeant des")
    print("  graphèmes (le G2P dépend des digrammes/positions) → ce niveau mesure")
    print("  un MÉLANGE de réordonnancement et de corruption G2P ; cf. niveau-1 pour")
    print("  le test propre.")


def _qualify(real: float, perm: float, spread: float) -> None:
    delta = abs(real - perm)
    print("  → QUALIFICATION :")
    if spread > 1.0:
        print(f"    (c) DISSIPATION possible : dispersion inter-seed {spread:.2f} pt > 1 pt.")
    if delta <= 1.0:
        print(f"    (a) RÉPÉTITION confirmée : |Δ|={delta:.2f} ≤ 1 pt. Le vote phonémique")
        print("        est context-free ; l'ordre intra-mot n'informe pas l'accord.")
        print("        Baseline d'ordre établie.")
    else:
        print(f"    (b) RÉSULTAT : |Δ|={delta:.2f} > 1 pt. L'ordre réintroduit de")
        print("        l'information (origine à isoler : syllabification / boost orientation).")


def main() -> None:
    eve_note = ("Rappel : MESURE, pas cible. Le réel est la baseline ; on teste si "
                "la permutation s'en écarte.")
    print("== H8-P0 : ABLATION D'ORDRE (accord opérateur émis vs balise) ==")
    print(f"   {eve_note}\n")

    dataset = _resolve("dataset_aba.txt") or _resolve("corpus_claude_aba.txt")
    if dataset is None:
        print("(dataset ABA introuvable)")
        return

    def load_cycles():
        out = []
        for i, cyc in enumerate(iter_aba_cycles(str(dataset))):
            if MAX_CYCLES and i >= MAX_CYCLES:
                break
            out.append(cyc)
        return out

    cycles = load_cycles()
    print(f"-- dataset : {dataset.name} ({len(cycles)} cycles) --\n")

    print("== NIVEAU-1 (propre) : permutation post-G2P sur réf. Python ==")
    print("   (le .so n'expose aucun point d'injection IPA ; réf. Python validée")
    print("    par parité — cf. tests test_phoneme_ref.py)")
    level1_python_postg2p(cycles)
    print()

    if is_available():
        tok = load_native_tokenizer()
        print("== NIVEAU-2 (repli) : permutation de graphèmes via le .so ==")
        level2_native_grapheme(cycles, tok)
    else:
        print("== NIVEAU-2 : tokenizer natif indisponible — sauté ==")


if __name__ == "__main__":
    main()
