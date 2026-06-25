from __future__ import annotations

"""Séparabilité SPECTRALE des classes de forme (Tour 5, E2 — geste DIV/lévo/in).

GESTE OPÉRATOIRE (émission du linguiste, incarnée telle quelle). DIV / lévogyre /
in = DISTINCTION appliquée à l'ESPACE DES POIDS, pas à l'état. On ne déroule plus
la trajectoire (Tour 4) : on regarde directement la matrice de transition ``A`` et
on demande si ses VALEURS PROPRES — sa signature spectrale ``(ρ, α_eig, is_complex)``
— séparent les classes de forme SANS connaître le réglage (ω, g) ni le nom de la
fabrique. Distinguer dans l'espace des opérateurs, pas dans l'espace des états.

HYPOTHÈSE H5. La signature spectrale de ``A`` sépare les classes :
  * ρ = |λ| (module spectral max) : cercle ρ=1, spirale ρ>1, baseline quelconque.
  * α_eig = |arg λ| (angle de la valeur propre dominante) : 0 ⇒ réel ⇒ pas de
    rotation ; ≠0 ⇒ paire conjuguée ⇒ figure courbe.
  * is_complex : discriminant < 0 (valeurs propres complexes conjuguées).

POPULATION (continuité L avec le Tour 4 : MÊME population). 40 graines, ω = π/5,
scale = 0.6, ``memory = 0`` partout. Trois réglages au MÊME statut spectral —
tous exposent ``model.A`` (buffer), lu IDENTIQUEMENT :
  * ``.circle(omega=π/5)``                 → A = R(ω),         ρ=1.
  * ``.spiral(omega=π/5, gain=1.06)``      → A = 1.06·R(ω),    ρ=1.06.
  * ``.random(g, scale=0.6)``  (par graine)→ A gaussienne,     ρ quelconque.

MESURES (AUC codée maison, style stats-autonomes du dépôt — Mann-Whitney = rang).
  * Test 1 : AUC(ρ : cercle vs spirale), origines mélangées, étiquette CACHÉE au
    discriminateur (l'AUC de rang ne lit que les scores). Progression ≥0.90.
  * Test 2 : AUC(ρ, α_eig : spirale-réglage vs spirale-baseline-qui-passe) — les
    ~2.5% de matrices ``.random`` qui tracent une vraie spirale au sens FORME du
    Tour 4 (critère géométrique shape_signature, JAMAIS le spectre : éviter la
    circularité). ≥0.85 ⇒ séparable (a) ; ≈0.5±0.1 ⇒ NON-séparable (b), résultat
    légitime à NE PAS forcer.
  * Test 3 : fraction des baselines à valeurs propres RÉELLES (is_complex=False) —
    pont de cohérence avec le taux-spirale du Tour 4 (une baseline réelle ne peut
    pas spiraler : pas de rotation).

BASELINE OBLIGATOIRE (REFUS).
  * CTRL-PERM : ≥20 permutations seedées des étiquettes de classe AVANT de mesurer
    l'AUC → distribution sous H0. Une AUC réelle ne compte que si elle dépasse le
    95e percentile de CTRL-PERM.
  * CTRL-SIG-RAND : discriminateur sur signatures ALÉATOIRES de même support
    (ρ tiré uniformément dans l'étendue observée) → doit donner AUC≈0.5.

REFUS — discipline. Aucun seuil ρ lu sur les réglages (le seuil/AUC se trouve sur
le nuage mélangé). Aucune lecture d'étiquette de forme au scoring. AUCUN passage de
(ω, g) ni de nom de fabrique au calcul de signature : on lit ``model.A`` seul. Si
l'issue (b) sort (spirale non-séparable), on la rapporte comme PROGRESSION
épistémique, pas comme échec.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..experimental.oscilloscope import InputSignal, Oscilloscope2D
from .shape_signature import (
    SpectralSignature,
    shape_signature,
    spectral_signature,
)

OMEGA_DEFAULT = math.pi / 5  # même ω que le Tour 4 (10 pas/tour)


# ---------------------------------------------------------------------------
# AUC maison (statistique de rang = Mann-Whitney U normalisé).
# ---------------------------------------------------------------------------

def _rank_auc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """AUC = P(score_pos > score_neg) estimée par les rangs (Mann-Whitney U / n1·n2).

    AUC = U_pos / (n_pos · n_neg) où ``U_pos = R_pos − n_pos(n_pos+1)/2`` et ``R_pos``
    est la somme des rangs (moyens, ex-aequo corrigés) du groupe positif dans le
    pool fusionné. C'est l'aire sous la courbe ROC, SANS choisir de seuil : un
    discriminateur de rang qui ne voit que les scores, jamais l'étiquette pendant le
    tri (l'étiquette ne sert qu'à séparer les deux sommes APRÈS le classement).

    Convention : une AUC > 0.5 signifie que le groupe ``pos`` a des scores PLUS
    GRANDS. On ne « plie » PAS vers max(auc, 1−auc) : la direction est une
    information (REFUS — ne pas forcer). L'appelant choisit l'orientation des
    groupes par convention documentée.
    """
    n1, n2 = len(scores_pos), len(scores_neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    combined = [(float(v), 1) for v in scores_pos] + [(float(v), 0) for v in scores_neg]
    combined.sort(key=lambda t: t[0])

    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # rangs 1-indexés, moyens sur les ex-aequo
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    r_pos = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 1)
    u_pos = r_pos - n1 * (n1 + 1) / 2.0
    return u_pos / (n1 * n2)


def _rank_auc_2d(
    pos: Sequence[Tuple[float, float]],
    neg: Sequence[Tuple[float, float]],
    *,
    w: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """AUC sur une signature 2D ``(ρ, α_eig)`` via un score scalaire de projection.

    Pour rester dans la philosophie « AUC de rang sans seuil et sans fit », on ne
    raffine PAS un classifieur appris : on projette la signature 2D sur un score
    scalaire ``w₀·ρ + w₁·α_eig`` (combinaison FIXÉE par défaut, non optimisée sur
    les étiquettes — pas de fit, donc pas de fuite), puis on prend l'AUC de rang du
    score projeté. ``w`` par défaut (1, 1) donne le même poids aux deux axes
    standardisés en amont par l'appelant. C'est un discriminateur 2D honnête :
    aucun degré de liberté n'est ajusté contre la cible.
    """
    proj_pos = [w[0] * p[0] + w[1] * p[1] for p in pos]
    proj_neg = [w[0] * p[0] + w[1] * p[1] for p in neg]
    return _rank_auc(proj_pos, proj_neg)


def _abs_dir(auc: float) -> float:
    """Force discriminante indépendante de l'orientation : |auc − 0.5| + 0.5 ∈ [0.5, 1]."""
    return abs(auc - 0.5) + 0.5


# ---------------------------------------------------------------------------
# Construction de la population (EXACTE du Tour 4).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PopulationMember:
    """Un membre de la population : son origine (cachée au scoring) et sa signature."""
    origin: str               # "circle" | "spiral" | "random" — JAMAIS passé au scoring
    seed: Optional[int]
    sig: SpectralSignature
    # forme géométrique du Tour 4 (sert UNIQUEMENT à isoler la baseline-qui-spirale,
    # via shape_signature — critère de FORME, jamais le spectre) :
    is_spiral_shape: bool


def _is_spiral_by_shape(cell: Oscilloscope2D, s0: torch.Tensor, *, steps: int) -> bool:
    """Critère SPIRALE du Tour 4, purement GÉOMÉTRIQUE (shape_signature, pas spectre).

    Reprend EXACTEMENT le critère du test anti-artefact du Tour 4
    (``test_random_baseline_rarely_draws_spiral``) : r2_logr_theta > 0.95,
    |slope_logr_theta| > 0.02, gardes spirale (≥2 tours, rayon non nul, phase
    monotone). N'utilise AUCUNE valeur propre : c'est la forme tracée à l'écran qui
    qualifie « spirale », pas le spectre — sinon le Test 2 serait circulaire.
    """
    tr = cell.trace(s0, steps=steps, signal=InputSignal(kind="zero"))
    sig = shape_signature(tr, s0)
    return (
        math.isfinite(sig.r2_logr_theta)
        and sig.r2_logr_theta > 0.95
        and abs(sig.slope_logr_theta) > 0.02
        and sig.passes_spiral_guards(min_turns=2.0)
    )


def build_population(
    *,
    n_seeds: int = 40,
    omega: float = OMEGA_DEFAULT,
    gain: float = 1.06,
    scale: float = 0.6,
    shape_steps: int = 120,
) -> List[PopulationMember]:
    """Construit la population EXACTE du Tour 4 (40 graines, ω=π/5, scale=0.6, mem=0).

    Pour chaque graine ``seed ∈ range(n_seeds)`` :
      * un membre ``circle``  (réglage déterministe, identique à chaque graine —
        on le réplique par graine pour égaliser les effectifs : 40 cercles),
      * un membre ``spiral``  (réglage déterministe, idem : 40 spirales),
      * un membre ``random``  (transition gaussienne regénérée PAR graine).

    Chaque membre porte sa ``SpectralSignature`` calculée sur ``model.A`` (memory=0,
    donc ``A`` seule est la bonne matrice). Le champ ``is_spiral_shape`` est calculé
    via ``shape_signature`` (FORME, pas spectre) et ne sert qu'à isoler la
    baseline-qui-spirale du Test 2.
    """
    s0 = torch.tensor([1.0, 0.0])
    members: List[PopulationMember] = []

    for seed in range(n_seeds):
        # --- cercle (réglage déterministe ; A = R(ω), memory=0) ---
        circ = Oscilloscope2D.circle(omega=omega)
        members.append(
            PopulationMember(
                origin="circle",
                seed=seed,
                sig=spectral_signature(circ.A, memory=circ.cfg.memory),
                is_spiral_shape=False,
            )
        )
        # --- spirale (réglage déterministe ; A = gain·R(ω), memory=0) ---
        spir = Oscilloscope2D.spiral(omega=omega, gain=gain)
        members.append(
            PopulationMember(
                origin="spiral",
                seed=seed,
                sig=spectral_signature(spir.A, memory=spir.cfg.memory),
                is_spiral_shape=True,
            )
        )
        # --- baseline aléatoire (regénérée par graine, MÊME échelle) ---
        g = torch.Generator().manual_seed(seed)
        rnd = Oscilloscope2D.random(g, scale=scale)
        members.append(
            PopulationMember(
                origin="random",
                seed=seed,
                sig=spectral_signature(rnd.A, memory=rnd.cfg.memory),
                is_spiral_shape=_is_spiral_by_shape(rnd, s0, steps=shape_steps),
            )
        )

    return members


@lru_cache(maxsize=None)
def collect_baseline_spirals(
    *,
    scale: float = 0.6,
    target: int = 8,
    reservoir: int = 5000,
    shape_steps: int = 120,
) -> Tuple[PopulationMember, ...]:
    """Extrait du réservoir étendu les baselines qui SPIRALENT (au sens FORME, Tour 4).

    FAIT MESURÉ (rapporté honnêtement) : à scale=0.6, la spirale-forme est TRÈS rare
    (~0.2% sur 2000 graines — encore plus rare que les ~2.5% anticipés). Sur 40
    graines il y en a 0 : le Test 2 ne serait PAS évaluable sur la population stricte.
    Pour l'évaluer quand même — fidèle à « rapporter l'AUC-spirale même si elle vaut
    0.5 » — on PARCOURT un réservoir seedé étendu ``range(reservoir)`` et on collecte
    les premières ``target`` baselines qui passent le critère de FORME (shape_signature,
    JAMAIS le spectre). Ce réservoir NE CONTAMINE PAS la population des tests 1 et 3
    (qui reste à 40 graines) : il ne sert qu'à peupler le groupe « baseline-qui-spirale »
    du Test 2. Les graines collectées sont déterministes (mêmes à chaque run).

    Mémoïsé (``lru_cache``) : le scan du réservoir est coûteux et purement
    déterministe — on le calcule une fois par jeu de paramètres. Retourne un tuple
    (hashable) pour le cache.
    """
    s0 = torch.tensor([1.0, 0.0])
    found: List[PopulationMember] = []
    for seed in range(reservoir):
        g = torch.Generator().manual_seed(seed)
        rnd = Oscilloscope2D.random(g, scale=scale)
        if _is_spiral_by_shape(rnd, s0, steps=shape_steps):
            found.append(
                PopulationMember(
                    origin="random",
                    seed=seed,
                    sig=spectral_signature(rnd.A, memory=rnd.cfg.memory),
                    is_spiral_shape=True,
                )
            )
            if len(found) >= target:
                break
    return tuple(found)


# ---------------------------------------------------------------------------
# CTRL-PERM (distribution sous H0) et CTRL-SIG-RAND (support aléatoire).
# ---------------------------------------------------------------------------

def _ctrl_perm_distribution(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    n_perms: int,
    seed0: int,
) -> Tuple[float, float, List[float]]:
    """Distribution de l'AUC sous permutation des étiquettes (H0 : pas de séparation).

    On garde les scores FIXES et on permute les étiquettes (graine déterministe par
    permutation). Pour chaque permutation, on calcule la force discriminante
    ``|auc − 0.5| + 0.5`` (l'orientation est arbitraire sous H0). Retourne
    ``(médiane, 95e_percentile, distribution)``. Une AUC réelle ne « compte » que
    si sa force dépasse le 95e percentile de cette distribution.
    """
    scores = list(scores)
    labels = list(labels)
    n = len(scores)
    dist: List[float] = []
    for p in range(n_perms):
        gp = torch.Generator().manual_seed(seed0 + p)
        perm = torch.randperm(n, generator=gp).tolist()
        perm_labels = [labels[i] for i in perm]
        pos = [scores[i] for i in range(n) if perm_labels[i] == 1]
        neg = [scores[i] for i in range(n) if perm_labels[i] == 0]
        if not pos or not neg:
            continue
        dist.append(_abs_dir(_rank_auc(pos, neg)))
    dist.sort()
    if not dist:
        return float("nan"), float("nan"), dist
    med = dist[len(dist) // 2]
    idx95 = min(len(dist) - 1, int(math.ceil(0.95 * len(dist))) - 1)
    p95 = dist[idx95]
    return med, p95, dist


def _ctrl_sig_rand_auc(
    n_pos: int,
    n_neg: int,
    support: Tuple[float, float],
    *,
    seed: int,
    n_rep: int = 200,
) -> float:
    """AUC MOYENNE d'un discriminateur sur signatures ALÉATOIRES de même support (≈0.5).

    On tire ``n_pos + n_neg`` scores uniformes dans l'étendue ``support`` observée,
    on les étiquette arbitrairement (les ``n_pos`` premiers = positifs), et on mesure
    l'AUC. Comme les scores sont indépendants de l'étiquette, l'AUC doit osciller
    autour de 0.5 : c'est le contrôle qui prouve que la séparation réelle vient bien
    de la STRUCTURE des signatures, pas du protocole. On MOYENNE sur ``n_rep`` tirages
    seedés : un seul tirage a une variance énorme à petit ``n`` (ex. 6 vs 40) — la
    moyenne converge vers 0.5, ce qui est le contrôle pertinent (le protocole est
    non-biaisé). Déterministe (graines ``seed + i``).
    """
    lo, hi = support
    span = hi - lo if hi > lo else 1.0
    acc = 0.0
    for i in range(n_rep):
        g = torch.Generator().manual_seed(seed + i)
        vals = (torch.rand(n_pos + n_neg, generator=g) * span + lo).tolist()
        acc += _rank_auc(vals[:n_pos], vals[n_pos:])
    return acc / n_rep


# ---------------------------------------------------------------------------
# Rapport.
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
class SpectralSeparabilityReport:
    n_seeds: int
    omega: float
    gain: float
    scale: float
    memory: float
    # distributions de ρ par classe : (5e, médiane, 95e)
    rho_circle: Tuple[float, float, float]
    rho_spiral: Tuple[float, float, float]
    rho_baseline: Tuple[float, float, float]
    rho_baseline_spiral: Tuple[float, float, float]      # baseline qui spirale (forme)
    rho_baseline_nonspiral: Tuple[float, float, float]   # baseline qui ne spirale pas
    # Test 1 : cercle vs spirale (origines mélangées, étiquette cachée)
    auc_test1: float
    ctrl_perm_med_test1: float
    ctrl_perm_p95_test1: float
    ctrl_sig_rand_test1: float
    test1_above_ctrl: bool
    # Test 2 : spirale-réglage vs spirale-baseline-qui-passe (ρ, α_eig)
    n_baseline_spiral: int
    auc_test2: float
    ctrl_perm_med_test2: float
    ctrl_perm_p95_test2: float
    ctrl_sig_rand_test2: float
    test2_above_ctrl: bool
    # Test 3 : fraction de baselines à valeurs propres réelles
    frac_baseline_real: float
    frac_baseline_complex: float
    # verdict
    issue: str          # "a" | "b" | "c"
    issue_label: str

    def summary(self) -> str:
        L: List[str] = []
        L.append("=" * 78)
        L.append(
            f"[spectral_separability] Tour 5 (E2)  n_seeds={self.n_seeds}  "
            f"omega=pi/5  gain={self.gain}  scale={self.scale}  memory={self.memory}"
        )
        L.append("=" * 78)
        L.append("\n--- distributions de rho = |lambda|_max  (5e | mediane | 95e) ---")
        def row(name: str, q: Tuple[float, float, float], n: str = "") -> str:
            return f"  {name:<26} {q[0]:>9.4f} | {q[1]:>9.4f} | {q[2]:>9.4f}  {n}"
        L.append(row("cercle (reglage)", self.rho_circle))
        L.append(row("spirale (reglage)", self.rho_spiral))
        L.append(row("baseline (toutes)", self.rho_baseline))
        L.append(row("baseline-qui-spirale", self.rho_baseline_spiral,
                     f"(n={self.n_baseline_spiral})"))
        L.append(row("baseline-non-spirale", self.rho_baseline_nonspiral))

        L.append("\n--- TEST 1 : AUC(rho : cercle vs spirale) [etiquette cachee] ---")
        L.append(f"  AUC_test1                 = {self.auc_test1:.4f}   (cible progression >=0.90)")
        L.append(f"  CTRL-PERM (med | 95e pct) = {self.ctrl_perm_med_test1:.4f} | {self.ctrl_perm_p95_test1:.4f}")
        L.append(f"  CTRL-SIG-RAND             = {self.ctrl_sig_rand_test1:.4f}   (attendu ~0.5)")
        L.append(f"  AUC > 95e pct CTRL-PERM ? = {'OUI' if self.test1_above_ctrl else 'NON'}")

        L.append("\n--- TEST 2 : AUC(rho,alpha_eig : spirale-reglage vs baseline-qui-spirale) ---")
        L.append(f"  n(baseline-qui-spirale)   = {self.n_baseline_spiral}")
        L.append(f"  AUC_test2                 = {self.auc_test2:.4f}   (>=0.85 ⇒ (a) ; ~0.5 ⇒ (b))")
        L.append(f"  CTRL-PERM (med | 95e pct) = {self.ctrl_perm_med_test2:.4f} | {self.ctrl_perm_p95_test2:.4f}")
        L.append(f"  CTRL-SIG-RAND             = {self.ctrl_sig_rand_test2:.4f}   (attendu ~0.5)")
        L.append(f"  AUC > 95e pct CTRL-PERM ? = {'OUI' if self.test2_above_ctrl else 'NON'}")

        L.append("\n--- TEST 3 : fraction baseline a valeurs propres reelles ---")
        L.append(f"  frac. reelles   (is_complex=False) = {self.frac_baseline_real:.4f}")
        L.append(f"  frac. complexes (is_complex=True)  = {self.frac_baseline_complex:.4f}")

        L.append(f"\n--- VERDICT : issue ({self.issue}) — {self.issue_label} ---")
        return "\n".join(L)


def run_spectral_separability(
    *,
    n_seeds: int = 40,
    omega: float = OMEGA_DEFAULT,
    gain: float = 1.06,
    scale: float = 0.6,
    n_perms: int = 20,
    shape_steps: int = 120,
    auc_sep_thresh: float = 0.85,
    auc_circle_thresh: float = 0.90,
    test2_target: int = 8,
    test2_reservoir: int = 5000,
) -> SpectralSeparabilityReport:
    """Pipeline complet du Tour 5 : population → signatures → AUC → contrôles → verdict.

    memory=0 partout (le réglage du tour). Construit la population du Tour 4, lit
    ``model.A`` (jamais (ω, g)), calcule les signatures spectrales, puis :
      * Test 1 : AUC(ρ : cercle vs spirale), origines mélangées, étiquette cachée.
      * Test 2 : AUC(ρ, α_eig : spirale-réglage vs baseline-qui-spirale [forme]).
      * Test 3 : fraction de baselines à valeurs propres réelles.
    avec CTRL-PERM (≥20 perms seedées, 95e pct) et CTRL-SIG-RAND (≈0.5) partout.
    """
    members = build_population(
        n_seeds=n_seeds, omega=omega, gain=gain, scale=scale, shape_steps=shape_steps
    )

    circle = [m for m in members if m.origin == "circle"]
    spiral = [m for m in members if m.origin == "spiral"]
    baseline = [m for m in members if m.origin == "random"]
    base_nonspiral = [m for m in baseline if not m.is_spiral_shape]

    # Groupe « baseline-qui-spirale » du Test 2. À scale=0.6 il y en a 0 dans les 40
    # graines (fait mesuré) : on PUISE alors dans un réservoir seedé étendu, qui ne
    # contamine PAS la population des tests 1 et 3. Si la population stricte en
    # contient déjà, on les prend ; sinon (ou pour compléter) on étend.
    base_spiral_strict = [m for m in baseline if m.is_spiral_shape]
    if len(base_spiral_strict) >= test2_target:
        base_spiral = base_spiral_strict
    else:
        base_spiral = collect_baseline_spirals(
            scale=scale, target=test2_target, reservoir=test2_reservoir,
            shape_steps=shape_steps,
        )

    rho = lambda lst: [m.sig.rho for m in lst]

    # ------- TEST 1 : ρ cercle vs spirale, origines mélangées, étiquette cachée ----
    # « origines mélangées » : on fusionne les deux groupes en un seul nuage de ρ ;
    # le discriminateur de rang ne voit que les ρ. L'étiquette ne resépare les deux
    # sommes de rangs qu'APRÈS le tri. Convention : pos = spirale (ρ plus grand).
    rho_circle = rho(circle)
    rho_spiral = rho(spiral)
    auc1 = _rank_auc(rho_spiral, rho_circle)

    scores1 = rho_spiral + rho_circle
    labels1 = [1] * len(rho_spiral) + [0] * len(rho_circle)
    cp_med1, cp_p95_1, _ = _ctrl_perm_distribution(scores1, labels1, n_perms=n_perms, seed0=5_000)
    lo1, hi1 = min(scores1), max(scores1)
    csr1 = _ctrl_sig_rand_auc(len(rho_spiral), len(rho_circle), (lo1, hi1), seed=5_100)
    test1_above = _abs_dir(auc1) > cp_p95_1

    # ------- TEST 2 : spirale-réglage vs baseline-qui-spirale (ρ, α_eig) -----------
    # On standardise ρ et α_eig sur le pool des deux groupes (moyenne/écart-type)
    # AVANT projection, pour que la combinaison (1,1) donne un poids comparable aux
    # deux axes. La standardisation utilise les scores, JAMAIS les étiquettes.
    def _standardize_pairs(
        groupA: Sequence[PopulationMember], groupB: Sequence[PopulationMember]
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        allm = list(groupA) + list(groupB)
        rs = [m.sig.rho for m in allm]
        as_ = [m.sig.alpha_eig for m in allm]
        def stats(v: List[float]) -> Tuple[float, float]:
            mu = sum(v) / len(v)
            var = sum((x - mu) ** 2 for x in v) / len(v)
            sd = math.sqrt(var) if var > 0 else 1.0
            return mu, sd
        mr, sr = stats(rs)
        ma, sa = stats(as_)
        def z(m: PopulationMember) -> Tuple[float, float]:
            return ((m.sig.rho - mr) / sr, (m.sig.alpha_eig - ma) / sa)
        return [z(m) for m in groupA], [z(m) for m in groupB]

    if base_spiral:
        zA, zB = _standardize_pairs(spiral, base_spiral)
        auc2 = _rank_auc_2d(zA, zB)  # pos = spirale-réglage
        scores2 = [p[0] + p[1] for p in zA] + [p[0] + p[1] for p in zB]
        labels2 = [1] * len(zA) + [0] * len(zB)
        cp_med2, cp_p95_2, _ = _ctrl_perm_distribution(scores2, labels2, n_perms=n_perms, seed0=6_000)
        lo2, hi2 = min(scores2), max(scores2)
        csr2 = _ctrl_sig_rand_auc(len(zA), len(zB), (lo2, hi2), seed=6_100)
        test2_above = _abs_dir(auc2) > cp_p95_2
    else:
        auc2 = float("nan")
        cp_med2 = cp_p95_2 = csr2 = float("nan")
        test2_above = False

    # ------- TEST 3 : fraction baseline à valeurs propres réelles ------------------
    n_real = sum(1 for m in baseline if not m.sig.is_complex)
    frac_real = n_real / len(baseline) if baseline else float("nan")
    frac_complex = 1.0 - frac_real if baseline else float("nan")

    # ------- VERDICT par issue -----------------------------------------------------
    # (a) tout séparable : cercle séparé (test1≥0.90 & >CTRL) ET spirale séparée
    #     (test2≥0.85 & >CTRL).
    # (b) cercle séparable mais spirale = demi-plan générique, NON privilégiée :
    #     test1 sépare mais test2 ≈ 0.5 (ne dépasse pas CTRL-PERM).
    # (c) rien ne sépare : même le cercle n'est pas séparé.
    circle_separated = (_abs_dir(auc1) >= auc_circle_thresh) and test1_above
    spiral_separated = (
        base_spiral
        and (_abs_dir(auc2) >= auc_sep_thresh)
        and test2_above
    )
    if not circle_separated:
        issue = "c"
        issue_label = (
            "rien ne separe (improbable, rho=1 de mesure nulle protege le cercle)"
        )
    elif spiral_separated:
        issue = "a"
        issue_label = "tout separable (cercle ET spirale)"
    else:
        issue = "b"
        issue_label = (
            "cercle separable mais spirale = demi-plan generique {complexe, rho>1}, "
            "NON privilegiee — PROGRESSION epistemique, pas echec"
        )

    return SpectralSeparabilityReport(
        n_seeds=n_seeds,
        omega=omega,
        gain=gain,
        scale=scale,
        memory=0.0,
        rho_circle=_quantiles(rho_circle),
        rho_spiral=_quantiles(rho_spiral),
        rho_baseline=_quantiles(rho(baseline)),
        rho_baseline_spiral=_quantiles(rho(base_spiral)),
        rho_baseline_nonspiral=_quantiles(rho(base_nonspiral)),
        auc_test1=auc1,
        ctrl_perm_med_test1=cp_med1,
        ctrl_perm_p95_test1=cp_p95_1,
        ctrl_sig_rand_test1=csr1,
        test1_above_ctrl=test1_above,
        n_baseline_spiral=len(base_spiral),
        auc_test2=auc2,
        ctrl_perm_med_test2=cp_med2,
        ctrl_perm_p95_test2=cp_p95_2,
        ctrl_sig_rand_test2=csr2,
        test2_above_ctrl=test2_above,
        frac_baseline_real=frac_real,
        frac_baseline_complex=frac_complex,
        issue=issue,
        issue_label=issue_label,
    )
