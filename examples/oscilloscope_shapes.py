"""Tour 4 — déploiement & mesure : OPÉRATEURS COMME GÉNÉRATEURS DE FORME 2D.

Protocole (REFUS, seeds fixés, baseline obligatoire) :

  * Réglage CERCLE  = Oscilloscope2D.circle  (rotation pure, gain=1, mémoire=0).
  * Réglage SPIRALE = Oscilloscope2D.spiral  (rotation + gain radial > 1).
  * Baseline RANDOM = Oscilloscope2D.random  (transition 2×2 aléatoire, MÊME
    échelle spectrale, regénérée par graine).

Les réglages cercle/spirale DÉCOULENT de la sémantique opératoire (rotation =
couplage anti-symétrique D↔L ; gain = MUL/DIV) ; ils ne sont PAS fittés sur la
figure. Seul l'angle ω et le gain g (coordonnées du geste) sont posés.

Mesures sur la SECONDE MOITIÉ de la trajectoire (régime établi). Pour chaque
graine : CV(r), R²(θ,t), R²(log r, θ), pente, N_tours, μ_r, et la qualification
α-ω (cos − l2, best_return_step). Statistique : Mann-Whitney U réglage vs baseline.

Lancer :  python examples/oscilloscope_shapes.py
"""
from __future__ import annotations

import math
import sys
from statistics import median
from typing import Dict, List

import torch

from spiraton.experimental.oscilloscope import InputSignal, Oscilloscope2D, rotation_matrix
from spiraton.diagnostics.shape_signature import mann_whitney_u, shape_signature

# --- protocole fixe ----------------------------------------------------------
N_SEEDS = 40
STEPS = 120
OMEGA = math.pi / 5          # 10 pas/tour : 2nde moitié = nb entier de tours
GAIN_SPIRAL = 1.06
SIGNAL = InputSignal(kind="zero")   # régime libre : s0 porte l'amplitude, figure propre

# Échelle de la baseline = norme spectrale typique des réglages ciblés.
# circle : ‖R‖₂ = 1 (isométrie). spiral : ‖g·R‖₂ = g = 1.06.
# On prend une échelle telle que ‖A_random‖₂ ≈ 1 en moyenne. Pour A = scale·N(0,1)_{2×2},
# E‖A‖₂ ≈ scale · 1.7 → scale ≈ 0.6 donne une transition de magnitude comparable.
BASE_SCALE = 0.6


def percentile(xs: List[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def collect(make_cell) -> List:
    sigs = []
    for seed in range(N_SEEDS):
        g = torch.Generator().manual_seed(seed)
        # s0 unitaire, orientation variable par graine (pour ne pas trivialiser)
        ang = float(torch.rand(1, generator=g) * 2 * math.pi)
        s0 = torch.tensor([math.cos(ang), math.sin(ang)])
        cell = make_cell(g)
        tr = cell.trace(s0, steps=STEPS, signal=SIGNAL)
        sigs.append(shape_signature(tr, s0))
    return sigs


def _finite(x: float, fallback: float) -> float:
    """Remplace un inf/nan par une sentinelle EXPLICITE.

    Les transitions aléatoires de la baseline peuvent diverger (cv_r=inf, mu_r=inf)
    ou collapser (mu_r=0). Ce sont des ÉCHECS DE FORME, pas des données à jeter :
    les remplacer par une sentinelle cohérente (cv_r non fini = très grande
    dispersion ; R² non fini = aucune structure = 0) est plus honnête que de les
    omettre (omission = biais en faveur de la baseline).
    """
    return x if math.isfinite(x) else fallback


def field(sigs, name: str) -> List[float]:
    # sentinelles par métrique : cv_r grand (forme non circulaire), R²→0 (pas de
    # structure), pente→0, n_turns→0, mu_r→0. Sentinelle cv_r = 10 (>> seuil 0.05).
    fallbacks = {
        "cv_r": 10.0, "r2_theta_t": 0.0, "r2_logr_theta": 0.0,
        "slope_logr_theta": 0.0, "n_turns": 0.0, "mu_r": 0.0,
        "ao_score": -10.0, "ao_cos_final": 0.0, "ao_l2_final": 0.0,
        "ao_best_return_step": 0.0,
    }
    fb = fallbacks.get(name, 0.0)
    return [_finite(float(getattr(s, name)), fb) for s in sigs]


def summarize(label: str, sigs) -> Dict[str, List[float]]:
    cols = ["cv_r", "r2_theta_t", "r2_logr_theta", "slope_logr_theta", "n_turns", "mu_r"]
    print(f"\n=== {label} (N={len(sigs)} graines) ===")
    out = {}
    for c in cols:
        vals = field(sigs, c)
        out[c] = vals
        print(
            f"  {c:18s} median={median(vals):+.4f}  "
            f"[5e={percentile(vals,0.05):+.4f}, 95e={percentile(vals,0.95):+.4f}]  "
            f"min={min(vals):+.4f} max={max(vals):+.4f}"
        )
    # α-ω
    ao_score = field(sigs, "ao_score")
    ao_best = field(sigs, "ao_best_return_step")
    ao_cos = field(sigs, "ao_cos_final")
    ao_l2 = field(sigs, "ao_l2_final")
    print(
        f"  {'ao cos_final':18s} median={median(ao_cos):+.4f}   "
        f"{'l2_final':12s} median={median(ao_l2):+.4f}   "
        f"best_return_step median={median(ao_best):.1f}"
    )
    # gardes
    rf = sum(1 for s in sigs if s.passes_radius_floor)
    pm = sum(1 for s in sigs if s.passes_phase_monotone)
    print(f"  gardes: radius_floor {rf}/{len(sigs)}  phase_monotone {pm}/{len(sigs)}")
    out["_ao_score"] = ao_score
    return out


def mw(label: str, metric: str, target: List[float], base: List[float]) -> None:
    u, p = mann_whitney_u(target, base)
    print(f"  {label:28s} {metric:16s} U={u:8.1f}  p={p:.3e}  "
          f"med(régl)={median(target):+.4f}  med(base)={median(base):+.4f}")


def main() -> None:
    # Robustesse Windows : un stdout en cp1252 ne peut pas encoder θ/λ/α/− quand la
    # sortie est redirigée vers un fichier. On force UTF-8 si le flux le permet, sans
    # échouer sinon — la logique de mesure est indépendante de l'encodage d'affichage.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass
    torch.manual_seed(0)

    circ = collect(lambda g: Oscilloscope2D.circle(omega=OMEGA))
    spir = collect(lambda g: Oscilloscope2D.spiral(omega=OMEGA, gain=GAIN_SPIRAL))
    base = collect(lambda g: Oscilloscope2D.random(g, scale=BASE_SCALE))

    c = summarize("RÉGLAGE CERCLE", circ)
    s = summarize("RÉGLAGE SPIRALE", spir)
    b = summarize("BASELINE RANDOM", base)

    print("\n=== TESTS Mann-Whitney (réglage vs baseline) ===")
    print("# CERCLE doit avoir un CV(r) PLUS PETIT que la baseline :")
    mw("cercle vs base", "cv_r", c["cv_r"], b["cv_r"])
    print(f"    -> CV(r) cercle médian = {median(c['cv_r']):.4f} ; "
          f"5e pct baseline = {percentile(b['cv_r'],0.05):.4f} ; "
          f"cercle < 0.05 ? {median(c['cv_r']) < 0.05}")

    print("\n# SPIRALE doit avoir un R²(log r, θ) PLUS GRAND que la baseline :")
    mw("spirale vs base", "r2_logr_theta", s["r2_logr_theta"], b["r2_logr_theta"])
    print(f"    -> R²(spirale) médian = {median(s['r2_logr_theta']):.4f} ; "
          f"95e pct baseline = {percentile(b['r2_logr_theta'],0.95):.4f} ; "
          f"spirale > 0.95 ? {median(s['r2_logr_theta']) > 0.95}")
    mw("spirale vs base", "slope_logr", s["slope_logr_theta"], b["slope_logr_theta"])
    mw("spirale vs base", "n_turns", s["n_turns"], b["n_turns"])

    # FRACTION qui passe les SEUILS DE FORME AVEC GARDES (la mesure la plus juste :
    # un réglage fiable trace TOUJOURS la figure ; la baseline ne l'attrape que par
    # hasard, et les gardes anti-trivial doivent rejeter ses dégénérescences).
    def circle_pass(sig) -> bool:
        v = float(sig.cv_r)
        return math.isfinite(v) and v < 0.05 and sig.passes_circle_guards()

    def spiral_pass(sig) -> bool:
        return (
            math.isfinite(sig.r2_logr_theta)
            and sig.r2_logr_theta > 0.95
            and abs(sig.slope_logr_theta) > 0.02
            and sig.passes_spiral_guards(min_turns=2.0)
        )

    print("\n=== FRACTION qui passe les seuils de FORME + gardes anti-trivial ===")
    print(f"  cercle-pass : réglage CERCLE  {sum(circle_pass(s) for s in circ)}/{len(circ)}   "
          f"baseline {sum(circle_pass(s) for s in base)}/{len(base)}")
    print(f"  spiral-pass : réglage SPIRALE {sum(spiral_pass(s) for s in spir)}/{len(spir)}   "
          f"baseline {sum(spiral_pass(s) for s in base)}/{len(base)}")
    print(f"  (la baseline tracant une spirale ~legitime = matrice a valeurs propres "
          f"complexes |λ|>1 ; rare et non fiable, vs reglage systematique)")

    print("\n=== Qualification α-ω (cos − l2) ===")
    print(f"  CERCLE  : cos_final med={median(field(circ,'ao_cos_final')):+.4f}  "
          f"l2_final med={median(field(circ,'ao_l2_final')):+.4f}  "
          f"-> revient proche (RÉPÉTITION attendue)")
    print(f"  SPIRALE : cos_final med={median(field(spir,'ao_cos_final')):+.4f}  "
          f"l2_final med={median(field(spir,'ao_l2_final')):+.4f}  "
          f"-> aligné mais s'éloigne (PROGRESSION attendue)")


if __name__ == "__main__":
    main()
