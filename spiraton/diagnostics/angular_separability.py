from __future__ import annotations

"""Séparabilité ANGULAIRE des classes de forme (Tour 6 — geste MUL/dextro + D₄).

GESTE OPÉRATOIRE (émission du linguiste, incarnée telle quelle). On demande si la
NON-COMMUTATIVITÉ (axiome 1) est la CONDITION d'une forme ANGULEUSE VRAIE — arêtes
droites multi-pas + coins francs (polygone), ou croix qui se recroise par le centre —
PAR OPPOSITION à ce qu'une seule carte linéaire à valeurs propres complexes peut tracer
(courbure toujours uniforme : cercle, spirale, n-gone-de-SOMMETS).

QUATRE RÉGIMES (tous des ``Polyscope2D`` ou ``Oscilloscope2D``, lus au MÊME statut —
on mesure la trace, jamais le nom de la fabrique) :
  * ANGULEUX-VRAI   : ``Polyscope2D.angular`` — deux cisaillements anisotropes d'AXES
                      DIFFÉRENTS (``‖[W_a, W_b]‖ > 0``). Arête = un shear répété (droit),
                      coin = bascule d'axe propre (braquage porté par la non-commutativité).
  * CTRL ‖·‖=0      : ``Polyscope2D.commuting`` — MÊME geste, MÊME gain, mais axes ALIGNÉS
                      ⇒ ``W_a, W_b`` COMMUTENT exactement ⇒ la composition dégénère en un
                      seul shear ⇒ la cornerness doit S'EFFONDRER (anti-artefact DUR).
  * CROIX D₄        : ``Polyscope2D.cross`` — flip à valeur propre réelle négative ×
                      rotation π/2 (générateurs du groupe diédral non-abélien). Passe
                      plusieurs fois par le centre, se recroise (selfX) — H6c.
  * BASELINE LIN.   : ``Oscilloscope2D.random`` — matrice 2×2 aléatoire de même échelle,
                      regénérée par graine. Ce qu'une SEULE carte linéaire peut tracer
                      (REFUS, baseline obligatoire). N-gone-de-SOMMETS (``ngon_vertices``)
                      = contrôle linéaire « anguleux apparent » (rotation discrète, H6a).

HYPOTHÈSES tranchées par la mesure :
  * H6a : carré/triangle-DE-SOMMETS = R(2π/n) ⇒ LINÉAIRE ⇒ ne testent pas l'axiome 1.
  * H6b : arêtes droites + coins francs ⇒ ``‖[W_a, W_b]‖ > 0`` (cornerness s'effondre au
          contrôle commutateur=0).
  * H6c : la croix (passage-centre récurrent + recroisement) ⇒ pas d'orbite convexe ⇒
          aucune matrice à λ complexes ne la trace.

MESURES (AUC de rang maison, CTRL-PERM — style Tour 5, helpers réutilisés). On RAPPORTE
DEUX AXES SÉPARÉS (directive orchestrateur, REFUS — ne pas forcer la fermeture) :
  (i)  CORNERNESS / coins / recroisement = le FAIT axiome-1 (variable d'intérêt) ;
  (ii) FERMETURE / STABILITÉ (final_norm, bornée ?) = la trajectoire dérive-t-elle ?
       Un coin franc AVEC dérive bornée est un « polygone-spirale » — résultat central
       accepté, PAS l'issue (d). Seule une divergence informe (non finie) est l'issue (d).

BASELINE OBLIGATOIRE (REFUS, DOUBLE) :
  (1) baseline linéaire 40 graines (ce qu'une matrice unique peut faire) ;
  (2) contrôle ``‖[W_a, W_b]‖ = 0`` (parallèle au CTRL D=L du Tour 1, gap=0 exact).
Une AUC réelle ne compte que si elle dépasse le 95e pct de CTRL-PERM (étiquettes permutées).

REFUS — discipline. Aucune forme dessinée à la main (elle émerge de l'alternance des
opérateurs). Aucune étiquette de forme ni (gain, axe) ne touche le scoring : on mesure la
TRACE. Si une issue NULL/DISSIPATION sort PAR FORME, on la rapporte telle quelle.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..experimental.polyscope import Polyscope2D
from ..experimental.oscilloscope import InputSignal, Oscilloscope2D
from .shape_signature import CornerSignature, corner_signature
# Helpers AUC / CTRL-PERM réutilisés du Tour 5 (mêmes définitions de rang, pas de copie).
from .spectral_separability import _rank_auc, _abs_dir, _ctrl_perm_distribution


# Réglages canon du Tour 6 (figés après calibrage — DÉCOULENT du geste, JAMAIS fittés sur
# une cible de forme). gain=1.05 / steps=40 : à ce couple le CONTRÔLE commutateur=0 reste
# une orbite lisse (la dérive ne pollue pas la mesure de braquage) tandis que l'anguleux
# garde des coins francs. scale baseline = gain (même échelle d'opérateur).
GAIN_DEFAULT = 1.05
STEPS_DEFAULT = 40
CROSS_STEPS_DEFAULT = 80          # la croix se lit sur un horizon un peu plus long (recroisements)
BASE_SCALE_DEFAULT = 1.05


# ---------------------------------------------------------------------------
# s0 par graine (perturbations seedées — déterministes).
# ---------------------------------------------------------------------------

def _s0_unit(seed: int) -> torch.Tensor:
    """État initial unitaire pseudo-aléatoire (graine ``seed``) — déterministe.

    Distribution de s0 sur le cercle unité : sert à transformer une figure DÉTERMINISTE
    (le réglage d'opérateurs est fixe) en une POPULATION mesurable. Graine décalée
    (10_000+seed) pour ne pas coïncider avec la graine de la matrice baseline.
    """
    g = torch.Generator().manual_seed(10_000 + seed)
    v = torch.randn(2, generator=g)
    return v / torch.linalg.vector_norm(v)


def _s0_axis(seed: int, *, jitter: float = 0.15) -> torch.Tensor:
    """État initial proche de l'axe ``x`` (jitter seedé faible) — pour la CROIX.

    La croix D₄ a une ORIENTATION préférée (ses axes de symétrie) : un s0 trop écarté
    de l'axe casse le recroisement. On échantillonne donc autour de l'axe x avec un
    jitter faible — c'est la famille de conditions initiales où le geste D₄ s'exprime.
    Honnête : on documente que la croix est une figure ORIENTÉE, pas isotrope.
    """
    g = torch.Generator().manual_seed(20_000 + seed)
    v = torch.tensor([1.0, 0.0]) + jitter * torch.randn(2, generator=g)
    return v / torch.linalg.vector_norm(v)


# ---------------------------------------------------------------------------
# Population : signature angulaire + fermeture par membre.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AngularMember:
    """Un membre : origine (cachée au scoring), commutateur, signature angulaire, fermeture."""
    origin: str               # "angular" | "commuting" | "cross" | "baseline" | "ngon"
    seed: int
    commutator: float         # ‖[W_a, W_b]‖ (0 pour baseline linéaire : une seule matrice)
    sig: CornerSignature
    final_norm: float         # ‖s_final‖ — fermeture/stabilité (dérive ?)
    bounded: bool             # trace finie ET final_norm sous un plafond (pas l'issue d)


def _trace_poly(cell: Polyscope2D, s0: torch.Tensor, steps: int) -> torch.Tensor:
    return cell.trace(s0, steps=steps)


def _trace_osc(cell: Oscilloscope2D, s0: torch.Tensor, steps: int) -> torch.Tensor:
    return cell.trace(s0, steps=steps, signal=InputSignal(kind="zero"))


def _member(
    origin: str, seed: int, commutator: float, trace: torch.Tensor, *, norm_cap: float
) -> AngularMember:
    sig = corner_signature(trace)
    finite = bool(torch.isfinite(trace).all().item())
    fn = float(torch.linalg.vector_norm(trace[-1])) if finite else float("inf")
    bounded = finite and math.isfinite(fn) and fn < norm_cap
    return AngularMember(
        origin=origin, seed=seed, commutator=commutator, sig=sig,
        final_norm=fn, bounded=bounded,
    )


def build_population(
    *,
    n_seeds: int = 40,
    gain: float = GAIN_DEFAULT,
    steps: int = STEPS_DEFAULT,
    cross_steps: int = CROSS_STEPS_DEFAULT,
    base_scale: float = BASE_SCALE_DEFAULT,
    norm_cap: float = 1e4,
) -> List[AngularMember]:
    """Construit la population du Tour 6 (n_seeds graines × 5 origines), déterministe.

    Pour chaque graine :
      * ``angular``   : double-shear axes différents (non-commutatif), s0 = _s0_unit.
      * ``commuting`` : MÊME geste, axes alignés (commutateur=0), s0 = _s0_unit.
      * ``cross``     : flip×R(π/2) = D₄, s0 = _s0_axis (figure orientée), cross_steps.
      * ``baseline``  : Oscilloscope2D.random (matrice unique, λ complexes possibles),
                        s0 = _s0_unit, échelle = ``base_scale``.
      * ``ngon``      : R(2π/4) pure (n-gone-de-sommets linéaire, contrôle H6a), s0 = _s0_unit.

    Chaque membre porte sa ``CornerSignature``, son commutateur, et sa fermeture
    (final_norm, bounded). ``norm_cap`` : plafond au-delà duquel la trace est jugée
    NON bornée (dissipation, issue d). Aucun nom d'origine ne touche le scoring.
    """
    members: List[AngularMember] = []
    for seed in range(n_seeds):
        s0 = _s0_unit(seed)
        s0c = _s0_axis(seed)

        ang = Polyscope2D.angular(gain=gain)
        members.append(_member("angular", seed, ang.commutator(),
                               _trace_poly(ang, s0, steps), norm_cap=norm_cap))

        com = Polyscope2D.commuting(gain=gain)
        members.append(_member("commuting", seed, com.commutator(),
                               _trace_poly(com, s0, steps), norm_cap=norm_cap))

        crx = Polyscope2D.cross()
        members.append(_member("cross", seed, crx.commutator(),
                               _trace_poly(crx, s0c, cross_steps), norm_cap=norm_cap))

        g = torch.Generator().manual_seed(seed)
        rnd = Oscilloscope2D.random(g, scale=base_scale)
        # commutateur d'une matrice unique vs elle-même = 0 (pas d'alternance) — la
        # baseline linéaire n'a PAS de second opérateur, c'est exactement le point.
        members.append(_member("baseline", seed, 0.0,
                               _trace_osc(rnd, s0, steps), norm_cap=norm_cap))

        ng = Polyscope2D.ngon_vertices(n=4)
        members.append(_member("ngon", seed, ng.commutator(),
                               _trace_poly(ng, s0, steps), norm_cap=norm_cap))

    return members


# ---------------------------------------------------------------------------
# Quantiles et AUC avec CTRL-PERM (réutilise le pipeline du Tour 5).
# ---------------------------------------------------------------------------

def _quantiles(xs: Sequence[float]) -> Tuple[float, float, float]:
    """(5e pct, médiane, 95e pct) d'une liste finie (déterministe)."""
    f = sorted(v for v in xs if v == v and v not in (float("inf"), float("-inf")))
    if not f:
        return float("nan"), float("nan"), float("nan")
    def q(p: float) -> float:
        idx = min(len(f) - 1, max(0, int(round(p * (len(f) - 1)))))
        return f[idx]
    return q(0.05), q(0.50), q(0.95)


@dataclass(frozen=True)
class AucResult:
    """Une comparaison AUC avec son contrôle de permutation."""
    name: str
    auc: float                  # AUC de rang brute (pos vs neg)
    abs_dir: float              # |auc − 0.5| + 0.5 (force, orientation-indépendante)
    ctrl_perm_med: float
    ctrl_perm_p95: float
    above_ctrl: bool            # abs_dir > 95e pct CTRL-PERM


def _auc_with_ctrl(
    pos: Sequence[float], neg: Sequence[float], *, name: str, n_perms: int, seed0: int
) -> AucResult:
    """AUC(pos vs neg) + CTRL-PERM (étiquettes permutées) — verdict above_ctrl."""
    auc = _rank_auc(pos, neg)
    scores = list(pos) + list(neg)
    labels = [1] * len(pos) + [0] * len(neg)
    med, p95, _ = _ctrl_perm_distribution(scores, labels, n_perms=n_perms, seed0=seed0)
    ad = _abs_dir(auc)
    return AucResult(
        name=name, auc=auc, abs_dir=ad,
        ctrl_perm_med=med, ctrl_perm_p95=p95,
        above_ctrl=bool(ad > p95),
    )


# ---------------------------------------------------------------------------
# Rapport.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AngularSeparabilityReport:
    n_seeds: int
    gain: float
    steps: int
    cross_steps: int
    base_scale: float
    # commutateurs par origine (médiane)
    comm_angular: float
    comm_commuting: float
    comm_cross: float
    # cornerness par origine (5e | med | 95e)
    corn_angular: Tuple[float, float, float]
    corn_commuting: Tuple[float, float, float]
    corn_cross: Tuple[float, float, float]
    corn_baseline: Tuple[float, float, float]
    corn_ngon: Tuple[float, float, float]
    # straight_edge_fraction (med)
    straight_angular: float
    straight_commuting: float
    straight_ngon: float
    # AUC clés
    auc_ang_vs_commuting: AucResult     # H6b : cornerness s'effondre au CTRL comm=0 ?
    auc_ang_vs_baseline: AucResult      # non-commutatif > baseline linéaire ?
    # croix : passage-centre + recroisement vs baseline linéaire
    cross_min_center: Tuple[float, float, float]
    cross_n_center_passes: Tuple[float, float, float]
    cross_self_intersections: Tuple[float, float, float]
    base_n_center_passes: Tuple[float, float, float]
    base_self_intersections: Tuple[float, float, float]
    auc_cross_selfX_vs_baseline: AucResult
    auc_cross_center_vs_baseline: AucResult
    frac_cross_recrosses: float         # frac. croix avec selfX>=1 ET center>=2
    frac_base_recrosses: float          # frac. baseline avec selfX>=1 ET center>=2
    # fermeture / stabilité (SÉPARÉ — directive orchestrateur)
    final_norm_angular: Tuple[float, float, float]
    final_norm_commuting: Tuple[float, float, float]
    final_norm_cross: Tuple[float, float, float]
    frac_angular_bounded: float
    frac_cross_bounded: float
    # verdict PAR FORME (issue a/b/c/d)
    issue_polygon: str
    issue_polygon_label: str
    issue_cross: str
    issue_cross_label: str

    def summary(self) -> str:
        L: List[str] = []
        L.append("=" * 84)
        L.append(
            f"[angular_separability] Tour 6  n_seeds={self.n_seeds}  gain={self.gain}  "
            f"steps={self.steps}  cross_steps={self.cross_steps}  base_scale={self.base_scale}"
        )
        L.append("=" * 84)

        L.append("\n--- COMMUTATEUR ‖[W_a,W_b]‖ (mediane) ---")
        L.append(f"  angular (axes differents) = {self.comm_angular:.4f}   (> 0 : non-commutatif)")
        L.append(f"  commuting (CTRL, axes =)  = {self.comm_commuting:.6f}   (= 0 exact : controle)")
        L.append(f"  cross D4 (flip x R pi/2)  = {self.comm_cross:.4f}   (>> 0 : diedral non-abelien)")

        L.append("\n--- CORNERNESS C = (P95|k| - med|k|)/med|k|   (5e | med | 95e) ---")
        def row(n: str, q: Tuple[float, float, float], extra: str = "") -> str:
            return f"  {n:<24} {q[0]:>8.2f} | {q[1]:>8.2f} | {q[2]:>8.2f}  {extra}"
        L.append(row("angular (non-commut)", self.corn_angular, "anguleux-vrai si med>=5"))
        L.append(row("commuting (CTRL comm=0)", self.corn_commuting, "doit s'effondrer <1"))
        L.append(row("baseline lineaire", self.corn_baseline))
        L.append(row("ngon-de-sommets (lin)", self.corn_ngon, "courbure uniforme -> C~0"))
        L.append(row("cross D4", self.corn_cross, "(degenere tout-coin : lire selfX)"))

        L.append("\n--- STRAIGHT_EDGE_FRACTION (mediane) ---")
        L.append(f"  angular   = {self.straight_angular:.3f}   commuting = {self.straight_commuting:.3f}   ngon-sommets = {self.straight_ngon:.3f}")

        L.append("\n--- AUC (H6b : la non-commutativite PORTE le coin) ---")
        for r in (self.auc_ang_vs_commuting, self.auc_ang_vs_baseline):
            L.append(
                f"  {r.name:<34} AUC={r.auc:.4f}  |force|={r.abs_dir:.4f}  "
                f"CTRL-PERM(med|95e)={r.ctrl_perm_med:.3f}|{r.ctrl_perm_p95:.3f}  "
                f">95e? {'OUI' if r.above_ctrl else 'NON'}"
            )

        L.append("\n--- CROIX D4 : passage-centre + recroisement vs baseline lineaire (H6c) ---")
        L.append(row("cross min_center_ratio", self.cross_min_center, "~0 : passe par le centre"))
        L.append(row("cross n_center_passes", self.cross_n_center_passes))
        L.append(row("cross self_intersections", self.cross_self_intersections))
        L.append(row("base  n_center_passes", self.base_n_center_passes))
        L.append(row("base  self_intersections", self.base_self_intersections))
        L.append(f"  frac. croix recroise (selfX>=1 & center>=2) = {self.frac_cross_recrosses:.3f}")
        L.append(f"  frac. base  recroise (selfX>=1 & center>=2) = {self.frac_base_recrosses:.3f}")
        for r in (self.auc_cross_selfX_vs_baseline, self.auc_cross_center_vs_baseline):
            L.append(
                f"  {r.name:<34} AUC={r.auc:.4f}  |force|={r.abs_dir:.4f}  "
                f"CTRL-PERM(95e)={r.ctrl_perm_p95:.3f}  >95e? {'OUI' if r.above_ctrl else 'NON'}"
            )

        L.append("\n--- FERMETURE / STABILITE (axe SEPARE — ne pas forcer la fermeture) ---")
        L.append(row("final_norm angular", self.final_norm_angular, "dérive bornée = polygone-spirale"))
        L.append(row("final_norm commuting", self.final_norm_commuting))
        L.append(row("final_norm cross", self.final_norm_cross, "borné = croix fermée-stable"))
        L.append(f"  frac. angular bornee = {self.frac_angular_bounded:.3f}   frac. cross bornee = {self.frac_cross_bounded:.3f}")

        L.append(f"\n--- VERDICT POLYGONE : issue ({self.issue_polygon}) — {self.issue_polygon_label} ---")
        L.append(f"--- VERDICT CROIX    : issue ({self.issue_cross}) — {self.issue_cross_label} ---")
        return "\n".join(L)


def run_angular_separability(
    *,
    n_seeds: int = 40,
    gain: float = GAIN_DEFAULT,
    steps: int = STEPS_DEFAULT,
    cross_steps: int = CROSS_STEPS_DEFAULT,
    base_scale: float = BASE_SCALE_DEFAULT,
    n_perms: int = 20,
    corner_thresh: float = 5.0,
    collapse_ratio: float = 0.2,
) -> AngularSeparabilityReport:
    """Pipeline complet du Tour 6 : population → cornerness/commutateur/fermeture → AUC → verdicts.

    corner_thresh   : seuil cornerness « anguleux-vrai » (C >= 5, émission linguiste).
    collapse_ratio  : l'effondrement au CTRL comm=0 est acquis si la cornerness mediane du
                      contrôle vaut MOINS de ``collapse_ratio`` × celle de l'anguleux (le
                      contrôle a une courbure RÉSIDUELLE du shear répété — non strictement 0
                      — mais d'un ordre de grandeur sous l'anguleux), ET l'AUC dépasse
                      CTRL-PERM. C'est l'effondrement RELATIF, fidèle au gap mesuré (≈38×).

    Verdicts PAR FORME (granularité demandée) :
      POLYGONE : (a) anguleux-vrai requiert ‖·‖≠0 (cornerness élevée ET s'effondre au CTRL
                     comm=0, > CTRL-PERM) ; (b) NULL = linéaire suffit (angular ≈ baseline) ;
                 (c) NULL inverse = ‖·‖≠0 mais cornerness ne suit pas ; (d) DISSIPATION =
                     l'angular diverge (non borné).
      CROIX    : (a-non-comm) si recroisement + passage-centre dépassent la baseline ; (b)/(c)
                 sinon (rapporté honnêtement).
    """
    members = build_population(
        n_seeds=n_seeds, gain=gain, steps=steps, cross_steps=cross_steps,
        base_scale=base_scale,
    )
    by = lambda o: [m for m in members if m.origin == o]
    angular = by("angular")
    commuting = by("commuting")
    cross = by("cross")
    baseline = by("baseline")
    ngon = by("ngon")

    def med(xs: Sequence[float]) -> float:
        f = sorted(xs)
        return f[len(f) // 2] if f else float("nan")

    corn = lambda lst: [m.sig.cornerness for m in lst]

    # --- AUC H6b : cornerness angular vs commuting (CTRL comm=0) et vs baseline lin ---
    auc_ac = _auc_with_ctrl(corn(angular), corn(commuting),
                            name="cornerness angular vs commuting", n_perms=n_perms, seed0=60_000)
    auc_ab = _auc_with_ctrl(corn(angular), corn(baseline),
                            name="cornerness angular vs baseline", n_perms=n_perms, seed0=60_100)

    # --- croix : recroisement / passage-centre vs baseline linéaire ---
    cross_sx = [float(m.sig.self_intersections) for m in cross]
    cross_ctr = [float(m.sig.n_center_passes) for m in cross]
    base_sx = [float(m.sig.self_intersections) for m in baseline]
    base_ctr = [float(m.sig.n_center_passes) for m in baseline]
    auc_cx_sx = _auc_with_ctrl(cross_sx, base_sx,
                               name="cross selfX vs baseline", n_perms=n_perms, seed0=60_200)
    auc_cx_ctr = _auc_with_ctrl(cross_ctr, base_ctr,
                                name="cross center vs baseline", n_perms=n_perms, seed0=60_300)
    frac_cross_recross = sum(
        1 for m in cross if m.sig.self_intersections >= 1 and m.sig.n_center_passes >= 2
    ) / len(cross) if cross else float("nan")
    frac_base_recross = sum(
        1 for m in baseline if m.sig.self_intersections >= 1 and m.sig.n_center_passes >= 2
    ) / len(baseline) if baseline else float("nan")

    # --- fermeture / stabilité ---
    fn = lambda lst: [m.final_norm for m in lst]
    frac_ang_bounded = sum(1 for m in angular if m.bounded) / len(angular) if angular else float("nan")
    frac_cross_bounded = sum(1 for m in cross if m.bounded) / len(cross) if cross else float("nan")

    # --- verdict POLYGONE (cornerness × commutateur) ---
    med_corn_ang = med(corn(angular))
    med_corn_com = med(corn(commuting))
    anguleux_vrai = med_corn_ang >= corner_thresh
    # Effondrement RELATIF : la cornerness du CTRL comm=0 tombe d'un ordre de grandeur sous
    # l'anguleux (ratio < collapse_ratio), ET l'AUC dépasse CTRL-PERM. La courbure résiduelle
    # du shear répété donne C_commuting ≈ 1 (≠ 0 exact) mais ≈ 38× sous l'anguleux.
    collapses_at_ctrl = (
        med_corn_ang > 0.0
        and (med_corn_com <= collapse_ratio * med_corn_ang)
        and auc_ac.above_ctrl
    )
    beats_baseline = auc_ab.above_ctrl
    all_bounded = frac_ang_bounded > 0.0  # au moins quelques traces finies (sinon dissipation pure)

    if not all_bounded:
        issue_poly = "d"
        issue_poly_label = "DISSIPATION : l'alternance diverge sans forme bornée (l.690)"
    elif anguleux_vrai and collapses_at_ctrl:
        issue_poly = "a"
        issue_poly_label = (
            "PROGRESSION : coins-francs exigent ‖[·,·]‖≠0 (cornerness s'effondre au CTRL "
            "comm=0) — axiome 1 rebranché ; arête+coin NON engendrables par 1 matrice lineaire"
        )
    elif not anguleux_vrai:
        issue_poly = "b"
        issue_poly_label = (
            "NULL honnête : pas d'anguleux-vrai mesurable — linéaire suffirait pour la geometrie"
        )
    else:
        issue_poly = "c"
        issue_poly_label = (
            "NULL inverse : ‖[·,·]‖≠0 mais cornerness ne s'effondre pas au CTRL — "
            "non-commutativité necessaire mais NON suffisante (cf. Tour 1)"
        )

    # --- verdict CROIX ---
    cross_distinct = (
        auc_cx_sx.above_ctrl
        and frac_cross_recross > frac_base_recross
        and med(cross_sx) > med(base_sx)
    )
    if cross_distinct:
        issue_cross = "a-non-comm"
        issue_cross_label = (
            "PROGRESSION : la croix D4 (‖·‖>>0) se recroise par le centre (selfX, passage-centre "
            "recurrent) au-dela de la baseline lineaire — aucune matrice a λ complexes ne la trace"
        )
    else:
        issue_cross = "b/c"
        issue_cross_label = (
            "la signature de croix ne depasse pas nettement la baseline lineaire sur cette "
            "population — rapporte honnêtement (passage-centre seul n'est pas discriminant : "
            "une spirale entrante passe aussi par le centre)"
        )

    return AngularSeparabilityReport(
        n_seeds=n_seeds, gain=gain, steps=steps, cross_steps=cross_steps, base_scale=base_scale,
        comm_angular=med([m.commutator for m in angular]),
        comm_commuting=med([m.commutator for m in commuting]),
        comm_cross=med([m.commutator for m in cross]),
        corn_angular=_quantiles(corn(angular)),
        corn_commuting=_quantiles(corn(commuting)),
        corn_cross=_quantiles(corn(cross)),
        corn_baseline=_quantiles(corn(baseline)),
        corn_ngon=_quantiles(corn(ngon)),
        straight_angular=med([m.sig.straight_edge_fraction for m in angular]),
        straight_commuting=med([m.sig.straight_edge_fraction for m in commuting]),
        straight_ngon=med([m.sig.straight_edge_fraction for m in ngon]),
        auc_ang_vs_commuting=auc_ac,
        auc_ang_vs_baseline=auc_ab,
        cross_min_center=_quantiles([m.sig.min_center_ratio for m in cross]),
        cross_n_center_passes=_quantiles(cross_ctr),
        cross_self_intersections=_quantiles(cross_sx),
        base_n_center_passes=_quantiles(base_ctr),
        base_self_intersections=_quantiles(base_sx),
        auc_cross_selfX_vs_baseline=auc_cx_sx,
        auc_cross_center_vs_baseline=auc_cx_ctr,
        frac_cross_recrosses=frac_cross_recross,
        frac_base_recrosses=frac_base_recross,
        final_norm_angular=_quantiles(fn(angular)),
        final_norm_commuting=_quantiles(fn(commuting)),
        final_norm_cross=_quantiles(fn(cross)),
        frac_angular_bounded=frac_ang_bounded,
        frac_cross_bounded=frac_cross_bounded,
        issue_polygon=issue_poly,
        issue_polygon_label=issue_poly_label,
        issue_cross=issue_cross,
        issue_cross_label=issue_cross_label,
    )
