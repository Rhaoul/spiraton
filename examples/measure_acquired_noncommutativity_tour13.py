"""Tour 13 — RESSERRER : la queue gauche est-elle STRUCTURELLE ou un artefact ?

État A (Tour 12) = PROGRESSION AVEC RÉSERVE : avantage d'ordre propre (vs contrôle
commutant-asymétrique-bien-conditionné) médian **+6.89, 18/20 graines**, MAIS verrou
strict **5e-pct = −1.99** échoue à cause de **2 graines (1, 5)** où le CONTRÔLE durci
acquiert lui-même un gros commutateur (Δ=+4.90 / +10.99) tout en commutant (≈1e-6) et
bien conditionné (κ≤9). Deux sources d'acquisition superposées :
  (1) signal **porté par l'ordre** de la cible (dominant, l'effet recherché) ;
  (2) **dérive non-commutante générique** du gradient Adam, indépendante de l'ordre
      (le bruit de fond → la queue gauche).

Ce script DÉPLOIE et MESURE, en SÉQUENÇANT (i) PUIS (ii) :

H13a — MESURE PURE (N=48, AUCUN nouveau mécanisme)
--------------------------------------------------
« La queue gauche est STRUCTURELLE (~10 % de graines négatives, stable de N=20 à
N=48), pas un artefact de petit échantillon. »
- Sweep EXACT du Tour 12 mais 48 graines (range(48)). λ=0.
- VÉRIF DURE : les 20 PREMIÈRES graines reproduisent BIT-À-BIT le Tour 12
  (médiane +6.89, 5e-pct −1.99) — preuve que N=48 est une EXTENSION, pas une
  re-mesure cachée.
- Issues : STRUCTURELLE si fraction négative ∈ [5%,15%] stable ET médiane ∈
  [+5.5,+8.0] (→ 5e-pct reste <0, ATTENDU, pas un échec) ; ARTEFACT si fraction
  négative <2.5% ET 5e-pct franchit 0 (→ réserve levée par (i) seul) ; DISSIPATION
  si médiane <+2.0 / blow-ups / NaN.
- **Note dure (honnêteté)** : si la queue est ~10 %, le 5e-pct restera <0 — donc
  (i) seul ne « lève » la réserve QUE si la queue était un artefact. H13a est un
  test de NATURE, pas de réussite.

H13b — REMÈDE (CONDITIONNEL à « queue structurelle » sous H13a)
---------------------------------------------------------------
« Une pénalité L2 douce sur les W, IDENTIQUE aux 3 conditions, réduit la source (2)
→ relève le 5e-pct au-dessus de 0 SANS déplacer la médiane. »
- λ_reg fixé par CRITÈRE ORTHOGONAL, AVEUGLE au 5e-pct de l'avantage :
  le plus petit λ ∈ {1e-4, 3e-4, 1e-3, 3e-3} (déclaré d'avance) qui réduit la
  variance de Δ_commuting_asym_wc (source 2, mesurée sur le CONTRÔLE SEUL) d'au
  moins un facteur 2, SANS dégrader la loss finale médiane de l'ordered de plus de
  10 %. Ce critère ne regarde QUE le contrôle et la convergence — JAMAIS l'avantage.
  INTERDIT de retenir le λ qui fait franchir 0 au 5e-pct.
- Run APPARIÉ : λ=0 vs λ_choisi sur les MÊMES 48 graines.
- Issues : réserve LEVÉE si 5e-pct(λ>0) > 0 ET médiane invariante à ±1.0 vs λ=0 ET
  le commutateur du contrôle sur (1,5) BAISSE vers le régime médian ; L2 INERTE si
  5e-pct reste <0 (→ queue structurelle, bascule ABA réelle au Tour 14) ; L2
  DISSIPATIVE (à rejeter) si médiane bouge >1.0 OU sous-apprentissage déguisé.

Discipline REFUS : λ fixé par critère orthogonal déclaré d'avance, JAMAIS réglé sur
le 5e-pct. L2 strictement identique aux 3 conditions. Seeds fixés, déterminisme
bit-à-bit. Canon byte-identique.

Usage :
    python examples/measure_acquired_noncommutativity_tour13.py
"""

from __future__ import annotations

import sys

from spiraton.diagnostics.acquired_noncommutativity import (
    BETA_MULT,
    KAPPA_MAX,
    _median,
    _percentile,
    run_sweep,
)


# Graines fixées A PRIORI (N=48, range 40-50 du kickoff → on prend 48 = range(48)).
SEEDS_48 = list(range(48))
SEEDS_20 = list(range(20))  # baseline Tour 12 (reproduite par les 20 premières)

# Candidats λ déclarés d'avance (kickoff H13b). On retient le PLUS PETIT qui
# satisfait le critère orthogonal. JAMAIS choisi sur le 5e-pct de l'avantage.
LAMBDA_CANDIDATES = [1e-4, 3e-4, 1e-3, 3e-3]
VAR_REDUCTION_FACTOR = 2.0   # variance contrôle réduite d'au moins ×2
MAX_LOSS_DEGRADATION = 0.10  # loss finale médiane ordered dégradée de ≤ 10 %

# Graines pathologiques du Tour 12 (le contrôle y acquiert un gros commutateur).
PATHO_SEEDS = (1, 5)

# Hyperparamètres EXACTS du Tour 12 (ne PAS changer — extension, pas re-mesure).
SWEEP_KW = dict(
    dim=6,
    init_scale=0.5,
    order=("add", "sub", "mul", "div"),
    n_samples=256,
    epochs=800,
    lr=5e-3,
    input_scale=1.0,
)


def _neg_fraction(advantages):
    n = len(advantages)
    neg = sum(1 for a in advantages if a <= 0.0)
    return neg, n, (neg / n if n else float("nan"))


def _delta_wc_by_seed(rep):
    """{seed: Δ_total du contrôle durci} — pour suivre les graines pathologiques."""
    return {r.seed: r.delta_total for r in rep.commuting_asym_wc}


# ----------------------------------------------------------------------------
# H13a — mesure pure N=48 (λ=0), avec vérif de reproduction bit-à-bit du Tour 12
# ----------------------------------------------------------------------------


def run_h13a():
    print("=" * 78)
    print("H13a — MESURE PURE N=48 (lambda=0, aucun nouveau mecanisme)")
    print("=" * 78)

    # Run N=48, lambda=0 (= Tour 12 etendu).
    rep48 = run_sweep(seeds=SEEDS_48, l2_reg=0.0, **SWEEP_KW)
    # Run N=20 INDEPENDANT (la baseline Tour 12) pour la preuve bit-a-bit.
    rep20 = run_sweep(seeds=SEEDS_20, l2_reg=0.0, **SWEEP_KW)

    adv48 = rep48.per_seed_advantage_proper
    adv20 = rep20.per_seed_advantage_proper

    # --- Preuve d'extension : les 20 premieres graines de N=48 == run N=20 -----
    adv48_first20 = adv48[:20]
    bit_identical = (adv48_first20 == adv20)
    print("\n--- PREUVE D'EXTENSION (pas de re-mesure cachee) ---")
    print(f"  adv_proper[:20] de N=48  ==  run N=20 independant : {bit_identical}")
    print(f"  median(N=20 first-20)  : {_median(adv48_first20):+.4f}  "
          f"(Tour 12 = +6.8911)")
    print(f"  5e-pct(N=20 first-20)  : {_percentile(adv48_first20, 5.0):+.4f}  "
          f"(Tour 12 = -1.9900)")
    if not bit_identical:
        print("  !! ECHEC bit-a-bit : N=48 N'EST PAS une extension propre — STOP.")
        return rep48, False

    # --- Mesures a N=48 --------------------------------------------------------
    med48 = _median(adv48)
    p05_48 = _percentile(adv48, 5.0)
    neg, n, frac = _neg_fraction(adv48)
    npos = n - neg
    print("\n--- AVANTAGE PROPRE a N=48 (la mesure de NATURE) ---")
    print(f"  median avantage_propre   : {med48:+.4f}   (Tour 12 N=20 = +6.8911)")
    print(f"  5e percentile            : {p05_48:+.4f}   (seuil strict > 0)")
    print(f"  positifs                 : {npos}/{n}")
    print(f"  fraction negative        : {frac*100:.1f}%  ({neg}/{n})  "
          f"(structurelle si [5%,15%] ; artefact si <2.5%)")

    # graines negatives explicitement listees (transparence de la queue gauche).
    negs = sorted(
        ((r.seed, a) for r, a in zip(rep48.commuting_asym_wc, adv48) if a <= 0.0),
        key=lambda t: t[1],
    )
    print(f"  graines negatives        : "
          + ", ".join(f"seed {s}({a:+.2f})" for s, a in negs))

    # --- Bornes anti-dissipation ----------------------------------------------
    s = rep48.summary
    print("\n--- BORNE ANTI-DISSIPATION (N=48) ---")
    print(f"  max total_final ordered   = {s['max_total_final_ordered']:.4f}"
          f"  (vs 10x max init = {10*s['max_total_init_ordered']:.4f})")
    print(f"  max total_final wc        = {s['max_total_final_commuting_asym_wc']:.4f}")
    print(f"  loss final ordered (med)  = {s['median_loss_final_ordered']:.6f}")
    print(f"  vector-floor loss (med)   = {s['median_vector_floor_loss']:.6f}")

    # --- Qualification de la NATURE -------------------------------------------
    print("\n--- QUALIFICATION H13a (NATURE de la queue) ---")
    structurelle = (0.05 <= frac <= 0.15) and (5.5 <= med48 <= 8.0)
    artefact = (frac < 0.025) and (p05_48 > 0.0)
    dissipation = (med48 < 2.0)
    if dissipation:
        print("  DISSIPATION — mediane effondree ou blow-up.")
        nature = "dissipation"
    elif artefact:
        print("  ARTEFACT — fraction neg <2.5% ET 5e-pct>0 : reserve LEVEE par (i) seul.")
        nature = "artefact"
    elif structurelle:
        print("  STRUCTURELLE — fraction neg dans [5%,15%], mediane stable.")
        print("  => le 5e-pct reste <0 par CONSTRUCTION ; ce n'est PAS un echec.")
        print("  => H13b (remede L2) est DECLENCHE.")
        nature = "structurelle"
    else:
        print("  INDETERMINEE — hors des bandes pre-declarees (a rapporter brut).")
        nature = "indeterminee"

    return rep48, nature


# ----------------------------------------------------------------------------
# H13b — selection orthogonale de lambda, puis run apparie lambda=0 vs lambda*
# ----------------------------------------------------------------------------


def select_lambda(rep48_l0):
    """Selectionne lambda par le CRITERE ORTHOGONAL (aveugle au 5e-pct avantage).

    Critere (declare d'avance) : le PLUS PETIT lambda parmi LAMBDA_CANDIDATES tel que
      (a) var(Delta_commuting_asym_wc)  <=  var(lambda=0) / VAR_REDUCTION_FACTOR
      (b) median_loss_final_ordered(lambda)  <=  median_loss_final_ordered(0) * (1+MAX_LOSS_DEGRADATION)
    Ne regarde QUE le controle (a) et la convergence (b). JAMAIS l'avantage propre.
    """
    var0 = rep48_l0.summary["var_delta_commuting_asym_wc"]
    loss0 = rep48_l0.summary["median_loss_final_ordered"]
    var_target = var0 / VAR_REDUCTION_FACTOR
    loss_cap = loss0 * (1.0 + MAX_LOSS_DEGRADATION)

    print("=" * 78)
    print("H13b — SELECTION ORTHOGONALE de lambda (aveugle au 5e-pct avantage)")
    print("=" * 78)
    print(f"  reference lambda=0 : var(Delta_wc) = {var0:.4f}"
          f"  | median_loss_ordered = {loss0:.6f}")
    print(f"  cibles : var <= {var_target:.4f} (reduction x{VAR_REDUCTION_FACTOR:.0f})"
          f"  ET loss_ordered <= {loss_cap:.6f} (+{MAX_LOSS_DEGRADATION*100:.0f}%)")
    print()
    print("  --- TABLEAU DE SELECTION (var controle + loss ordered, par lambda) ---")
    print(f"  {'lambda':>8}  {'var(d_wc)':>10}  {'var ok?':>8}  "
          f"{'loss_ord':>10}  {'loss ok?':>9}  {'RETENU?':>8}")
    print(f"  {0.0:>8.0e}  {var0:>10.4f}  {'(ref)':>8}  "
          f"{loss0:>10.6f}  {'(ref)':>9}  {'-':>8}")

    chosen = None
    table = []
    for lam in LAMBDA_CANDIDATES:
        rep = run_sweep(seeds=SEEDS_48, l2_reg=lam, **SWEEP_KW)
        var = rep.summary["var_delta_commuting_asym_wc"]
        loss = rep.summary["median_loss_final_ordered"]
        var_ok = var <= var_target
        loss_ok = loss <= loss_cap
        eligible = var_ok and loss_ok
        # On retient le PREMIER (= plus petit) lambda eligible.
        retained = eligible and (chosen is None)
        if retained:
            chosen = lam
        table.append((lam, var, var_ok, loss, loss_ok, retained, rep))
        print(f"  {lam:>8.0e}  {var:>10.4f}  {str(var_ok):>8}  "
              f"{loss:>10.6f}  {str(loss_ok):>9}  {str(retained):>8}")

    print()
    if chosen is None:
        print("  AUCUN lambda candidat ne satisfait le critere orthogonal.")
        print("  => H13b ne peut pas s'appliquer ; la queue reste structurelle.")
    else:
        print(f"  lambda RETENU (critere orthogonal, AVANT de regarder l'avantage)"
              f" : {chosen:.0e}")
    return chosen, table


def run_h13b(rep48_l0, chosen, table):
    print()
    print("=" * 78)
    print("H13b — RUN APPARIE lambda=0 vs lambda* sur les MEMES 48 graines")
    print("=" * 78)

    # rep a lambda* (deja calcule dans le tableau de selection — reutilise).
    rep48_lp = next(rep for (lam, _, _, _, _, _, rep) in table if lam == chosen)

    adv0 = rep48_l0.per_seed_advantage_proper
    advp = rep48_lp.per_seed_advantage_proper
    med0, medp = _median(adv0), _median(advp)
    p050, p05p = _percentile(adv0, 5.0), _percentile(advp, 5.0)
    neg0 = sum(1 for a in adv0 if a <= 0.0)
    negp = sum(1 for a in advp if a <= 0.0)
    n = len(adv0)

    print(f"\n  lambda choisi : {chosen:.0e}")
    print(f"  {'metrique':>26}  {'lambda=0':>12}  {'lambda*':>12}  {'delta':>10}")
    print(f"  {'median avantage_propre':>26}  {med0:>12.4f}  {medp:>12.4f}"
          f"  {medp-med0:>+10.4f}")
    print(f"  {'5e percentile':>26}  {p050:>12.4f}  {p05p:>12.4f}"
          f"  {p05p-p050:>+10.4f}")
    print(f"  {'positifs':>26}  {n-neg0:>10}/{n}  {n-negp:>10}/{n}")
    print(f"  {'var(d_wc) source 2':>26}  "
          f"{rep48_l0.summary['var_delta_commuting_asym_wc']:>12.4f}  "
          f"{rep48_lp.summary['var_delta_commuting_asym_wc']:>12.4f}")

    # --- Commutateur du controle sur les graines pathologiques (1, 5) ----------
    wc0 = _delta_wc_by_seed(rep48_l0)
    wcp = _delta_wc_by_seed(rep48_lp)
    med_wc0 = _median([r.delta_total for r in rep48_l0.commuting_asym_wc])
    med_wcp = _median([r.delta_total for r in rep48_lp.commuting_asym_wc])
    print(f"\n  --- Delta_commutateur du CONTROLE sur les graines pathologiques ---")
    print(f"  (regime median wc : lambda=0 {med_wc0:+.3f} -> lambda* {med_wcp:+.3f})")
    for sd in PATHO_SEEDS:
        print(f"  seed {sd}: Delta_wc  lambda=0 {wc0.get(sd, float('nan')):+8.3f}"
              f"  ->  lambda* {wcp.get(sd, float('nan')):+8.3f}")

    # --- Convergence (anti sous-apprentissage deguise) -------------------------
    print(f"\n  --- CONVERGENCE (anti sous-apprentissage deguise) ---")
    print(f"  loss final ordered (med) : lambda=0 "
          f"{rep48_l0.summary['median_loss_final_ordered']:.6f}"
          f"  ->  lambda* {rep48_lp.summary['median_loss_final_ordered']:.6f}")
    print(f"  loss final wc (med)      : lambda=0 "
          f"{rep48_l0.summary['median_loss_final_commuting_asym_wc']:.6f}"
          f"  ->  lambda* {rep48_lp.summary['median_loss_final_commuting_asym_wc']:.6f}")

    # --- Qualification du retour H13b -----------------------------------------
    print(f"\n--- QUALIFICATION H13b ---")
    median_stable = abs(medp - med0) <= 1.0
    crossed0 = p05p > 0.0
    # baisse du commutateur du controle sur les graines pathologiques ?
    patho_drop = all(
        abs(wcp.get(sd, float("inf")) - med_wcp) <= abs(wc0.get(sd, 0.0) - med_wc0)
        for sd in PATHO_SEEDS
    )
    loss_blowup = rep48_lp.summary["median_loss_final_ordered"] > (
        rep48_l0.summary["median_loss_final_ordered"] * (1.0 + MAX_LOSS_DEGRADATION)
    )
    if not median_stable or loss_blowup:
        print("  L2 DISSIPATIVE (a rejeter) — la mediane a bouge >1.0 (touche la "
              "source 1) ou sous-apprentissage.")
    elif crossed0:
        print("  RESERVE LEVEE — 5e-pct(lambda*)>0 ET mediane invariante (+-1.0).")
        print(f"     controle sur graines patho baisse vers le regime median : {patho_drop}")
    else:
        print("  L2 INERTE — 5e-pct reste <0 a mediane invariante : la queue est")
        print("     vraiment STRUCTURELLE => bascule ABA reelle au Tour 14.")


def main() -> int:
    # Robustesse console Windows (cp1252) : forcer UTF-8 si possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(f"controle durci : beta = {BETA_MULT}*init_scale, kappa_max = {KAPPA_MAX}")
    print(f"sweep : {SWEEP_KW}\n")

    rep48_l0, nature = run_h13a()
    if nature is False:
        return 1  # echec bit-a-bit : ne pas poursuivre.

    # H13b CONDITIONNEL : seulement si la queue est STRUCTURELLE.
    if nature == "structurelle":
        chosen, table = select_lambda(rep48_l0)
        if chosen is not None:
            run_h13b(rep48_l0, chosen, table)
    else:
        print("\n--- H13b NON DECLENCHE ---")
        if nature == "artefact":
            print("  Reserve levee par (i) seul (artefact) : (ii) inutile.")
        elif nature == "dissipation":
            print("  Dissipation sous (i) : ne pas appliquer le remede.")
        else:
            print("  Nature indeterminee : rapporter brut, ne pas forcer (ii).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
