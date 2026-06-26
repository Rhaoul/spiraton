"""Tour 14 — non-commutativité acquise sous l'ordre SÉMANTIQUE RÉEL du corpus.

État A (Tour 13) : l'axiome 1 dynamique est verrouillé EN PRATIQUE sur tâche
SYNTHÉTIQUE — la non-commutativité s'ACQUIERT pendant l'entraînement quand la
cible porte de l'ordre (avantage propre médian ≈ +6.9 vs contrôle order-détruit
durci, queue gauche structurelle ~10 %). Garde anti-ornière → passage IMPOSÉ à
l'ABA RÉELLE.

QUESTION du Tour 14 (H14)
-------------------------
Un `MatrixSpiratonCell` entraîné à reproduire la transformation d'un cycle ABA
RÉEL **dans l'ordre des segments DONNÉ par le corpus** (A→B→A′, retournement
DX→LV au 3e segment) acquiert-il PLUS de non-commutativité `Δ‖[W_a,W_b]‖` que le
même élève sur les MÊMES cycles dont l'ordre est DÉTRUIT (permutation de
l'assignation segment→sens, à difficulté appariée, mêmes vecteurs) ?

    avantage_réel(seed) = Δ_total(ordre-corpus) − Δ_total(ordre-détruit)

CONSTRUCTION DE CIBLE — déclarée A PRIORI (cf. docstring du diagnostic)
----------------------------------------------------------------------
* Contenu RÉEL : 3 vecteurs 33D par cycle (PhonemeFeaturizer natif, pool mean),
  L2-normalisés (conditionnement d'échelle), projetés 33→6 par une matrice
  gaussienne FIXE partagée (régime dimensionnel du Tour 12, où l'échelle du
  commutateur laisse lire un Δ d'ordre — à dim=33 l'init ≈404 noie le Δ ~±2).
* ORDRE = lu du corpus, jamais forgé : chiralité du segment → sens de composition
  de la paire primaire ⊗⊕ = (mul,add). DX → (mul,add) ; LV → (add,mul) =
  retournement. Clôture canonique (DX,DX,LV) → (avant, avant, ARRIÈRE).
* CIBLE : trajectoire de cycle chaînée A→B→A′ d'un enseignant non-commutant FIGÉ
  (la même fabrique que Tours 11-13). Elle DONNE DU GRADIENT ORDONNÉ et rien de
  plus. **JAMAIS de cos(pred, A′)** : ce tour ne ré-ouvre PAS CHANTIER5.
* CONTRÔLE ordre-détruit : même enseignant, mêmes vecteurs, MÊME multiset de sens
  {avant,avant,arrière} — assignation segment→sens permutée (le retournement
  lévogyre tombe sur un segment permuté). Détruit la POSITION réelle du pivot.

4 ISSUES nommées d'avance (transposées du Tour 12)
--------------------------------------------------
(a) PROGRESSION : médiane(avantage_réel) ≥ +2.0 ET ≥15/20 positifs ET >0 vs détruit.
(b) NULL LÉGITIME (le plus probable, assumé) : médiane ∈ [−0.5,+2.0[ OU <15/20.
    Lecture : l'ordre sémantique réel est trop faible/bruité — cohérent avec
    « chiralité positionnelle/grammaticale » (Tours 7/9/10).
(c) RÉPÉTITION : avantage_réel ≈ +6.9 (≈ chiffre synthétique) → alarme REFUS#1
    (on aurait refabriqué l'ordre — auditer que la cible vient des vecteurs).
(d) DISSIPATION : blow-ups/NaN, ou contrôle détruit qui ne converge pas
    (loss ≫ corpus → comparaison invalide).

BASELINES OBLIGATOIRES (REFUS)
------------------------------
1. Contrôle ordre-détruit à convergence appariée (loss du même ordre de grandeur).
2. Init (Δ depuis epoch 0, jamais la valeur absolue finale).
3. Covariable = le +6.9 SYNTHÉTIQUE (run en parallèle, MÊMES graines) = référence
   haute — situe le réel sans le confondre.
4. Plancher vectoriel (tâche réelle infaisable optimalement sans ordre ?).
5. ≥20 graines, déterminisme bit-à-bit (md5 ×2).

Usage :
    python examples/measure_real_aba_noncommutativity.py
"""

from __future__ import annotations

import hashlib
import sys

from spiraton.data.featurizers import PhonemeFeaturizer
from spiraton.data.tokenizer_bridge import TokenizerUnavailable
from spiraton.diagnostics.acquired_noncommutativity import (
    _median,
    _percentile,
    run_sweep,
)
from spiraton.diagnostics.real_aba_noncommutativity import (
    default_corpora,
    load_real_cycles,
    run_real_sweep,
)


# --- Configuration déclarée A PRIORI ----------------------------------------
SEEDS = list(range(20))         # ≥20 graines (REFUS#5)
PROJ_DIM = 6                    # régime dimensionnel du Tour 12
INIT_SCALE = 0.5                # init_scale du Tour 12
EPOCHS = 800                    # convergence (parité de loss corpus/détruit)
LR = 5e-3
MAX_CYCLES = 200                # sous-échantillon fixé (déterministe, ordre fichier)

# Covariable synthétique = la tâche EXACTE du Tour 12 (référence haute +6.9).
SYNTH_KW = dict(
    dim=6, init_scale=0.5, order=("add", "sub", "mul", "div"),
    n_samples=256, epochs=800, lr=5e-3, input_scale=1.0,
)


def _fmt(report) -> str:
    s = report.summary
    return (
        f"median_advantage_real = {s['median_advantage_real']:+.4f}\n"
        f"  5e-pct                = {s['p05_advantage_real']:+.4f}\n"
        f"  positifs              = {int(s['n_advantage_positive'])}/{int(s['n_seeds'])}"
        f"   (binom p≥ = {s['binom_p_ge_advantage']:.4g})\n"
        f"  Δ_corpus (médiane)    = {s['median_delta_corpus']:+.4f}\n"
        f"  Δ_détruit (médiane)   = {s['median_delta_destroyed']:+.4f}"
    )


def _digest(report) -> str:
    """md5 des avantages par graine (déterminisme bit-à-bit)."""
    payload = ";".join(f"{a:.10f}" for a in report.per_seed_advantage)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 78)
    print("Tour 14 — ABA RÉELLE : non-commutativité acquise sous l'ordre du corpus")
    print("=" * 78)

    # --- Featurizer NATIF obligatoire (REFUS : jamais le hachage) -------------
    try:
        feat = PhonemeFeaturizer(pool="mean")
    except TokenizerUnavailable as exc:
        print(f"\n[SKIP LÉGITIME] tokenizer natif indisponible : {exc}")
        print("Le fallback de hachage fabriquerait le contenu (faute REFUS) → on skippe.")
        return 0
    assert feat.is_fallback is False

    corpora = default_corpora()
    print(f"\nCorpus réels : {corpora}")
    samples = load_real_cycles(
        corpora, feat, max_cycles=MAX_CYCLES, proj_dim=PROJ_DIM,
    )
    print(f"Cycles réels chargés (clôture canonique) : {len(samples)}  "
          f"(dim de calcul = {PROJ_DIM}, init_scale = {INIT_SCALE}, epochs = {EPOCHS})")
    print(f"Sens corpus du cycle[0] (DX→(mul,add), LV→(add,mul)) : "
          f"{samples[0].senses_corpus}")

    # --- Mesure principale : avantage réel -----------------------------------
    rep = run_real_sweep(
        samples=samples, seeds=SEEDS, dim=PROJ_DIM,
        init_scale=INIT_SCALE, epochs=EPOCHS, lr=LR,
    )
    s = rep.summary

    print("\n" + "-" * 78)
    print("MESURE PRINCIPALE — avantage_réel = Δ_corpus − Δ_détruit (par graine)")
    print("-" * 78)
    print(_fmt(rep))

    # Liste explicite des graines négatives (transparence de la queue).
    negs = sorted(
        ((r.seed, a) for r, a in zip(rep.corpus, rep.per_seed_advantage) if a <= 0.0),
        key=lambda t: t[1],
    )
    print("  graines avantage ≤ 0   : "
          + (", ".join(f"seed {sd}({a:+.2f})" for sd, a in negs) or "(aucune)"))

    # --- Baseline 1 : convergence appariée (REFUS#1 / issue d DISSIPATION) ----
    lc = s["median_loss_final_corpus"]
    ld = s["median_loss_final_destroyed"]
    ratio = ld / lc if lc > 0 else float("inf")
    print("\n" + "-" * 78)
    print("BASELINE 1 — convergence appariée (le contrôle doit converger comparablement)")
    print("-" * 78)
    print(f"  loss finale corpus (méd)  = {lc:.6f}")
    print(f"  loss finale détruit (méd) = {ld:.6f}   ratio détruit/corpus = {ratio:.2f}")
    print(f"  loss init corpus (méd)    = {s['median_loss_init_corpus']:.6f}")
    converge_ok = 0.1 <= ratio <= 10.0
    print(f"  => convergence du même ordre de grandeur ? {converge_ok}  "
          f"(sinon issue (d) DISSIPATION)")
    print(f"  graines où détruit > 1.5×corpus : {int(s['n_destroyed_harder'])}/"
          f"{int(s['n_seeds'])}  (asymétrie de difficulté du chaînage)")

    # --- CONFOND de difficulté (REFUS#2, capital) ----------------------------
    print("\n" + "-" * 78)
    print("CONFOND DE DIFFICULTÉ — l'avantage est-il porté par le contenu d'ordre")
    print("ou par l'écart de fit (dérive générique = 'source 2' du Tour 13) ?")
    print("-" * 78)
    print(f"  corr(avantage_réel, loss_détruit − loss_corpus) = "
          f"{s['corr_advantage_lossgap']:+.3f}")
    print("  (nettement <0 ⇒ une part de l'avantage négatif vient de ce que la")
    print("   cible ordre-détruite est plus DURE à fitter — pas d'un effet d'ordre.")
    print("   Le contrôle ne converge alors PAS à difficulté appariée : la")
    print("   comparaison corpus/détruit est partiellement CONFONDUE.)")

    # --- Baseline 2 : init (Δ depuis epoch 0) --------------------------------
    print("\n" + "-" * 78)
    print("BASELINE 2 — init (on mesure des DELTAS depuis epoch 0, jamais l'absolu)")
    print("-" * 78)
    print(f"  total commutateur init (méd) = {s['median_total_init_corpus']:.4f}")
    print(f"  total commutateur final (méd)= {s['median_total_final_corpus']:.4f}")
    print(f"  (Δ_corpus = final − init     = {s['median_delta_corpus']:+.4f})")

    # --- Baseline 4 : plancher vectoriel -------------------------------------
    floor = s["median_vector_floor_loss"]
    floor_ratio = floor / lc if lc > 0 else float("inf")
    print("\n" + "-" * 78)
    print("BASELINE 4 — plancher vectoriel (tâche infaisable optimalement sans ordre ?)")
    print("-" * 78)
    print(f"  plancher vectoriel (méd)  = {floor:.6f}")
    print(f"  élève matriciel corpus    = {lc:.6f}   plancher/élève = {floor_ratio:.2f}")
    print(f"  => tâche order-dépendante (plancher ≫ élève) ? {floor_ratio > 1.5}")

    # --- Anti-dissipation ----------------------------------------------------
    print("\n" + "-" * 78)
    print("ANTI-DISSIPATION")
    print("-" * 78)
    print(f"  max total_final corpus   = {s['max_total_final_corpus']:.4f}  "
          f"(vs 10×max_init = {10*s['max_total_init_corpus']:.4f})")
    print(f"  max total_final détruit  = {s['max_total_final_destroyed']:.4f}")

    # --- Baseline 3 : covariable SYNTHÉTIQUE (+6.9), MÊMES graines ------------
    print("\n" + "-" * 78)
    print("BASELINE 3 — covariable SYNTHÉTIQUE (Tour 12, +6.9), MÊMES graines")
    print("-" * 78)
    synth = run_sweep(seeds=SEEDS, l2_reg=0.0, **SYNTH_KW)
    syn = synth.summary
    print(f"  avantage propre synthétique (médiane) = {syn['median_advantage_proper']:+.4f}"
          f"   (référence historique +6.89)")
    print(f"  positifs synthétiques                 = "
          f"{int(syn['n_advantage_proper_positive'])}/{len(SEEDS)}")
    print(f"  => le réel ({s['median_advantage_real']:+.3f}) situé RELATIVEMENT au "
          f"synthétique ({syn['median_advantage_proper']:+.3f}) — situé, pas confondu.")

    # --- Déterminisme bit-à-bit (md5 ×2) -------------------------------------
    print("\n" + "-" * 78)
    print("DÉTERMINISME BIT-À-BIT (md5 des avantages, 2 runs)")
    print("-" * 78)
    rep2 = run_real_sweep(
        samples=samples, seeds=SEEDS, dim=PROJ_DIM,
        init_scale=INIT_SCALE, epochs=EPOCHS, lr=LR,
    )
    d1, d2 = _digest(rep), _digest(rep2)
    print(f"  md5 run1 = {d1}")
    print(f"  md5 run2 = {d2}")
    print(f"  identiques ? {d1 == d2}")

    # --- Qualification du retour ---------------------------------------------
    print("\n" + "=" * 78)
    print("QUALIFICATION DU RETOUR (a/b/c/d)")
    print("=" * 78)
    med = s["median_advantage_real"]
    npos = int(s["n_advantage_positive"])
    n = int(s["n_seeds"])
    if not converge_ok or s["max_total_final_corpus"] > 10 * s["max_total_init_corpus"]:
        verdict = "(d) DISSIPATION — convergence non appariée ou blow-up."
    elif med >= 6.0:
        verdict = ("(c) RÉPÉTITION — avantage ≈ synthétique : ALARME REFUS#1, "
                   "auditer que la cible vient des vecteurs du corpus.")
    elif med >= 2.0 and npos >= 15:
        verdict = ("(a) PROGRESSION — l'ordre sémantique réel DRIVE la "
                   "non-commutativité acquise au-dessus du contrôle.")
    else:
        verdict = ("(b) NULL LÉGITIME — l'ordre sémantique réel ne pilote pas le "
                   "commutateur plus que l'ordre détruit (assumé d'avance comme le "
                   "plus probable : chiralité positionnelle/grammaticale, Tours 7/9/10).")
    print(f"  médiane={med:+.3f}, positifs={npos}/{n}, "
          f"convergence_ok={converge_ok} → {verdict}")
    if floor_ratio <= 1.5:
        print("\n  CAVEAT DE VALIDITÉ (rapporté, non masqué) : plancher vectoriel "
              f"≈ élève (ratio {floor_ratio:.2f} ≤ 1.5) → la tâche composée réelle")
        print("  est FAIBLEMENT order-dépendante à ce régime. Le null (b) tient mais")
        print("  signifie surtout : peu d'ordre exploitable, et la position-corpus du")
        print("  retournement n'en concentre pas plus que les positions détruites.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
