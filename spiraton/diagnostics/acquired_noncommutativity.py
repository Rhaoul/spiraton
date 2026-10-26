"""Diagnostic du chantier 1 *jamais réalisé* : la non-commutativité **acquise**.

Le chantier 1 (CLAUDE.md) demande de suivre le commutateur
``‖W_a W_b − W_b W_a‖`` **pendant l'entraînement**, pas seulement à l'init. Au
Tour 6 de la danse, le commutateur n'a été mesuré qu'à l'initialisation (où il
est non nul *par défaut*, parce que deux matrices aléatoires ne commutent pas).
Ce module mesure la grandeur que le chantier appelait vraiment : le **DELTA
signé** ``total_final − total_init`` du commutateur total, et **sa dépendance à
la présence d'ordre dans la tâche**.

Pourquoi le delta et pas la valeur absolue
------------------------------------------
Un ``MatrixSpiratonCell`` initialise ``W_op = I + scale·N(0,1)`` : à l'init, des
matrices aléatoires NE commutent PAS (``total_noncommutativity() > 0``, vérifié
par ``test_matrix_cell.py`` l.168-169). Dire « la non-commutativité est acquise »
ne peut donc JAMAIS vouloir dire « commutateur > 0 en fin d'entraînement ». Cela
doit être un **changement conditionnel** :

    (i)  Δ_ordre = total_final − total_init > 0  sur la tâche-avec-ordre, ET
    (ii) Δ_ordre > Δ_contrôle                    (le delta n'apparaît QUE quand
                                                  l'ordre est dans la cible).

Sans (ii) on ne mesure que la dérive générique des poids sous Adam, pas une
acquisition portée par l'ordre.

Garde REFUS capitale
--------------------
AUCUN terme de perte ne récompense ``total_noncommutativity()``. La perte est
purement la reconstruction de la cible enseignante (MSE sur l'état). Le
commutateur doit croître comme **effet secondaire** de l'apprentissage de la
tâche, ou pas du tout. Récompenser directement le commutateur serait coder en
dur le résultat (faute analogue au « boost recopiant les étiquettes »).

Tâche synthétique enseignant-figé (null interprétable)
------------------------------------------------------
- *Tâche-avec-ordre* : un enseignant ``MatrixSpiratonCell`` aux matrices NON
  commutantes, **gelées**, produit ``y = compose(x, order)`` — la composition
  linéaire ORDONNÉE ``W_{o_k} … W_{o_1} · x``. La cible n'est atteignable
  optimalement QUE si l'élève capture l'ordre.
- *Contrôle order-détruit* : enseignant aux matrices qui **commutent**
  (construites sur une base propre commune), même échelle de norme, même
  difficulté apparente. La fonction ``x → y`` reste déterministe et apprenable,
  mais l'ordre n'y porte plus d'information (``W_a W_b x = W_b W_a x``). C'est
  l'incarnation la plus propre de « order-détruit » : on retire le *contenu
  d'ordre* sans casser la convergence (contrairement à un mélange des cibles,
  qui détruit la fonction elle-même et empêche la convergence → comparaison
  invalide). Parallèle direct du CTRL ``commuting`` (‖·‖=0) du Tour 6 et du
  CTRL ``D=L`` du Tour 1.

On utilise ``compose()`` (composition linéaire brute) et non ``forward()`` : la
sélection de branche dextro/lévo de ``forward`` ajoute un bruit de mode qui
masquerait le signal d'ordre. ``compose`` isole exactement la dépendance à
l'ordre que l'on veut mesurer.

Seeds fixés, déterminisme bit-à-bit. ``matrix_cell.py`` n'est PAS modifié.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..experimental.matrix_cell import MatrixSpiratonCell, OPS


# Paires d'intérêt déclarées d'avance (kickoff Tour 11) :
#   primaire  ⊗⊕ = (mul, add)  — branche dextrogyre
#   covariable ⊘⊖ = (div, sub) — branche lévogyre
# Les clés suivent la convention de ``MatrixSpiratonCell.commutators()`` :
# paire non ordonnée, opérateurs dans l'ordre canonique OPS = (add, sub, mul, div).
PAIR_MUL_ADD = "add,mul"   # ‖[W_add, W_mul]‖  == ‖[W_mul, W_add]‖
PAIR_DIV_SUB = "sub,div"   # ‖[W_sub, W_div]‖


# ----------------------------------------------------------------------------
# Construction des enseignants figés
# ----------------------------------------------------------------------------


def _noncommuting_teacher(
    dim: int, init_scale: float, generator: torch.Generator
) -> MatrixSpiratonCell:
    """Enseignant aux quatre matrices génériques (non commutantes par défaut).

    Identique en distribution à l'init d'un élève (``I + scale·N(0,1)``) mais
    **gelé** (``requires_grad_(False)``). Sa composition ordonnée porte de
    l'ordre : ``compose(x, order)`` dépend de la permutation des opérateurs.
    """
    teacher = MatrixSpiratonCell(input_size=dim, init_scale=init_scale)
    with torch.no_grad():
        eye = torch.eye(dim)
        for name in OPS:
            w = getattr(teacher, f"W_{name}")
            w.copy_(eye + torch.randn(dim, dim, generator=generator) * init_scale)
    teacher.requires_grad_(False)
    return teacher


def _commuting_teacher(
    dim: int, init_scale: float, generator: torch.Generator
) -> MatrixSpiratonCell:
    """Enseignant order-détruit : quatre matrices qui **commutent deux à deux**.

    Construction : on tire une base propre commune ``Q`` (orthogonale, via QR
    d'une gaussienne) ; chaque opérateur est ``Q diag(1 + scale·N(0,1)) Qᵀ``.
    Des matrices simultanément diagonalisables dans la même base commutent
    exactement (``W_a W_b = W_b W_a``), donc ``compose(x, order)`` est invariant
    à la permutation : l'ordre ne porte plus aucune information. Les normes
    individuelles restent du même ordre de grandeur que l'enseignant
    non-commutant (même ``init_scale``, valeurs propres ≈ ``1 + scale·N``), donc
    la difficulté de reconstruction est comparable.

    C'est l'analogue exact du CTRL ``commuting`` (‖·‖=0) du Polyscope (Tour 6).
    """
    teacher = MatrixSpiratonCell(input_size=dim, init_scale=init_scale)
    with torch.no_grad():
        # Base propre commune partagée par les quatre opérateurs.
        a = torch.randn(dim, dim, generator=generator)
        q, _ = torch.linalg.qr(a)
        for name in OPS:
            eigs = 1.0 + torch.randn(dim, generator=generator) * init_scale
            w = q @ torch.diag(eigs) @ q.t()
            getattr(teacher, f"W_{name}").copy_(w)
    teacher.requires_grad_(False)
    return teacher


# ----------------------------------------------------------------------------
# Tour 12 : contrôle commutant DURCI (asymétrique, bien conditionné)
# ----------------------------------------------------------------------------
#
# Réserve d'équité du Tour 11 (à lever) : ``_commuting_teacher`` ci-dessus
# construit ``Q diag Qᵀ`` avec ``Q`` ORTHOGONAL → matrices **symétriques**
# (``‖W−Wᵀ‖ = 0``). L'enseignant ordonné, lui, est asymétrique. Le contrôle
# différait donc sur DEUX axes : (1) l'ordre absent ET (2) la symétrie. Une
# partie de l'avantage +7.6 du Tour 11 peut tenir au confond symétrie, pas à
# l'ordre. Probe rapide (contrôle commutant-asymétrique) → +3.79 sur 8/10.
#
# Ce contrôle DURCI isole l'ordre comme SEUL facteur : on remplace la base
# propre orthogonale ``Q`` par une base propre **non-orthogonale** ``P``, de
# conditionnement borné (``κ(P) ≤ kappa_max``). Les matrices
# ``W_op = P diag(λ_op) P⁻¹`` partagent toujours la même base propre → elles
# **commutent exactement** (``[W_a, W_b] = 0`` modulo l'erreur numérique de
# l'inversion ``P⁻¹``). Mais comme ``P⁻¹ ≠ Pᵀ``, elles sont **asymétriques** :
# le seul axe qui reste distinct de l'ordonné est l'ORDRE.
#
# Constantes de construction FIXÉES D'AVANCE (pas de fit sur le résultat) :
#   β  = BETA_MULT · init_scale   (degré de liberté de l'asymétrie)
#   κ_max = KAPPA_MAX             (plafond de conditionnement)
# Calibrées en UN balayage documenté (β ∈ {0.6,0.9,1.2,1.5,2.0}·scale,
# κ_max=10, dim=6, scale=0.5, 10 graines) : β=0.45·scale (BETA_MULT=0.9)
# place la médiane du ratio ``‖W−Wᵀ‖_wc / ‖W−Wᵀ‖_ordonné`` à ≈0.66 (8/10
# graines dans [0.5, 2.0]), avec κ(P) ∈ [3.0, 9.0] ≤ 10 et commutateur ≈1e-6.
# Ces deux constantes ne sont JAMAIS réajustées sur l'avantage mesuré.

BETA_MULT = 0.9      # β = BETA_MULT · init_scale (asymétrie de la base propre P)
KAPPA_MAX = 10.0     # plafond de conditionnement de P (anti-blow-up)
_KAPPA_MAX_RETRIES = 100  # plafond de re-tirages par rejet (terminaison)


def _well_conditioned_eigenbasis(
    dim: int,
    init_scale: float,
    generator: torch.Generator,
    kappa_max: float = KAPPA_MAX,
    beta_mult: float = BETA_MULT,
    max_retries: int = _KAPPA_MAX_RETRIES,
) -> Tuple[torch.Tensor, float]:
    """Base propre commune NON-orthogonale ``P``, conditionnement ``κ(P)`` borné.

    ``P = I + β·N(0,1)`` avec ``β = beta_mult·init_scale``. On **borne κ(P) par
    REJET / re-tirage SEEDÉ déterministe** : tant que ``κ(P) > kappa_max``, on
    re-tire ``N`` du **même** générateur (donc reproductible). Plafond de
    ``max_retries`` re-tirages pour garantir la terminaison ; si dépassé, on
    **baisse β par une logique fixe** (β ← β/2) et on tire une dernière fois —
    PAS de sélection du meilleur tirage (ce serait du cherry-pick). Retourne
    ``(P, κ(P))``.
    """
    eye = torch.eye(dim)
    beta = beta_mult * init_scale
    for _ in range(max_retries):
        N = torch.randn(dim, dim, generator=generator)
        P = eye + beta * N
        kappa = float(torch.linalg.cond(P))
        if kappa <= kappa_max:
            return P, kappa
    # Garde de terminaison : baisse fixe de β (logique déterministe, pas de
    # sélection du meilleur). En pratique jamais atteint à β=0.45·scale, κ_max=10.
    beta = beta * 0.5
    N = torch.randn(dim, dim, generator=generator)
    P = eye + beta * N
    return P, float(torch.linalg.cond(P))


def _commuting_teacher_well_conditioned(
    dim: int,
    init_scale: float,
    generator: torch.Generator,
    kappa_max: float = KAPPA_MAX,
    beta_mult: float = BETA_MULT,
) -> MatrixSpiratonCell:
    """Contrôle order-détruit DURCI : commute exactement MAIS asymétrique.

    Construction (Tour 12, lève la réserve d'équité du Tour 11) :
      1. Base propre commune **non-orthogonale** ``P = I + β·N(0,1)``,
         ``κ(P) ≤ kappa_max`` par rejet seedé (cf. ``_well_conditioned_eigenbasis``).
      2. Valeurs propres réelles distinctes par opérateur :
         ``λ_op = 1 + init_scale·N(0,1)`` (comme ``_commuting_teacher``).
      3. ``W_op = P · diag(λ_op) · P⁻¹``.

    Les quatre ``W_op`` sont simultanément diagonalisables dans **la même base
    P** → ``[W_a, W_b] = 0`` exactement (modulo l'erreur de ``P⁻¹``). Mais
    ``P⁻¹ ≠ Pᵀ`` (P non orthogonale) → ``W_op ≠ W_opᵀ`` : **asymétrique**,
    comme l'enseignant ordonné. Le SEUL axe qui distingue désormais ce contrôle
    de l'ordonné est l'**ordre de composition**.
    """
    teacher = MatrixSpiratonCell(input_size=dim, init_scale=init_scale)
    with torch.no_grad():
        P, _ = _well_conditioned_eigenbasis(
            dim, init_scale, generator, kappa_max=kappa_max, beta_mult=beta_mult
        )
        Pinv = torch.linalg.inv(P)
        for name in OPS:
            eigs = 1.0 + torch.randn(dim, generator=generator) * init_scale
            w = P @ torch.diag(eigs) @ Pinv
            getattr(teacher, f"W_{name}").copy_(w)
    teacher.requires_grad_(False)
    return teacher


# ----------------------------------------------------------------------------
# Trajectoire d'entraînement instrumentée
# ----------------------------------------------------------------------------


@dataclass
class TrajectoryReport:
    """Trajectoire d'un entraînement instrumenté (une condition, une graine)."""

    seed: int
    condition: str  # "ordered" | "commuting" | ...
    # Par epoch (longueur = epochs + 1 : index 0 = AVANT toute mise à jour) :
    epochs_total: List[float] = field(default_factory=list)        # total_noncommutativity()
    epochs_loss: List[float] = field(default_factory=list)         # MSE
    epochs_pair_mul_add: List[float] = field(default_factory=list)  # ‖[W_mul,W_add]‖
    epochs_pair_div_sub: List[float] = field(default_factory=list)  # ‖[W_div,W_sub]‖

    @property
    def total_init(self) -> float:
        return self.epochs_total[0]

    @property
    def total_final(self) -> float:
        return self.epochs_total[-1]

    @property
    def delta_total(self) -> float:
        return self.total_final - self.total_init

    @property
    def loss_init(self) -> float:
        return self.epochs_loss[0]

    @property
    def loss_final(self) -> float:
        return self.epochs_loss[-1]

    @property
    def delta_mul_add(self) -> float:
        return self.epochs_pair_mul_add[-1] - self.epochs_pair_mul_add[0]

    @property
    def delta_div_sub(self) -> float:
        return self.epochs_pair_div_sub[-1] - self.epochs_pair_div_sub[0]


def _record(cell: MatrixSpiratonCell, report: TrajectoryReport, loss: float) -> None:
    with torch.no_grad():
        comms = cell.commutators()
        report.epochs_total.append(float(cell.total_noncommutativity()))
        report.epochs_pair_mul_add.append(float(comms[PAIR_MUL_ADD]))
        report.epochs_pair_div_sub.append(float(comms[PAIR_DIV_SUB]))
    report.epochs_loss.append(loss)


def train_one(
    *,
    seed: int,
    condition: str,
    teacher: MatrixSpiratonCell,
    dim: int,
    init_scale: float,
    order: Sequence[str],
    n_samples: int,
    epochs: int,
    lr: float,
    input_scale: float = 1.0,
    l2_reg: float = 0.0,
) -> TrajectoryReport:
    """Entraîne un élève à reproduire la composition ordonnée de l'enseignant.

    Perte = MSE(``élève.compose(x, order)``, ``enseignant.compose(x, order)``)
    ``+ l2_reg · Σ_op ‖W_op‖²`` (Tour 13b).

    AUCUN terme ne touche le commutateur. À chaque epoch on enregistre
    ``total_noncommutativity()`` et les deux paires d'intérêt : la trajectoire
    du commutateur est un *observateur passif* de l'entraînement.

    Régularisation L2 (Tour 13b — garde REFUS)
    ------------------------------------------
    ``l2_reg`` ajoute une pénalité Tikhonov **isotrope** sur les quatre matrices
    ``W_op`` : ``Σ_op ‖W_op‖²`` (Frobenius au carré). Ce terme NE touche PAS le
    commutateur — il décourage l'enflure générique des poids (la « source 2 » :
    dérive non-commutante générique du gradient Adam, indépendante de l'ordre),
    pas la non-commutativité elle-même. Il est destiné à être appliqué
    **IDENTIQUEMENT** aux trois conditions (équité ; cf. ``run_sweep``).

    **Défaut ``l2_reg=0.0`` : bit-à-bit identique au Tour 12.** Le terme L2 n'est
    *littéralement pas calculé ni additionné* quand ``l2_reg == 0.0`` (court-
    circuit ``if``), donc l'optimisation reste numériquement la MSE pure et
    reproduit le Tour 12 octet pour octet.
    """
    gen = torch.Generator().manual_seed(seed)

    # Élève : init aléatoire INDÉPENDANT de l'enseignant (graine décalée).
    student = MatrixSpiratonCell(input_size=dim, init_scale=init_scale)
    with torch.no_grad():
        eye = torch.eye(dim)
        for name in OPS:
            getattr(student, f"W_{name}").copy_(
                eye + torch.randn(dim, dim, generator=gen) * init_scale
            )
        student.bias.zero_()

    # Données fixes (mêmes x pour toutes les conditions à graine égale).
    x = torch.randn(n_samples, dim, generator=gen) * input_scale
    with torch.no_grad():
        y = teacher.compose(x, order)

    opt = torch.optim.Adam(student.parameters(), lr=lr)

    report = TrajectoryReport(seed=seed, condition=condition)
    # Mesure initiale (epoch 0 : AVANT toute mise à jour).
    with torch.no_grad():
        pred0 = student.compose(x, order)
        loss0 = float(torch.mean((pred0 - y) ** 2))
    _record(student, report, loss0)

    for _ in range(epochs):
        pred = student.compose(x, order)
        mse = torch.mean((pred - y) ** 2)
        # Court-circuit ``l2_reg == 0.0`` : la perte optimisée RESTE la MSE pure
        # (bit-à-bit Tour 12). Sinon, pénalité L2 isotrope sur les W (jamais sur
        # le commutateur). La trajectoire ``epochs_loss`` enregistre toujours la
        # MSE seule — le diagnostic de reconstruction reste comparable λ=0/λ>0.
        if l2_reg == 0.0:
            loss = mse
        else:
            l2 = sum(
                (getattr(student, f"W_{name}") ** 2).sum() for name in OPS
            )
            loss = mse + l2_reg * l2
        opt.zero_grad()
        loss.backward()
        opt.step()
        _record(student, report, float(mse.detach()))

    return report


# ----------------------------------------------------------------------------
# Baseline plancher : cellule canon vectorielle (‖·‖ ≡ 0, ordre infaisable)
# ----------------------------------------------------------------------------


def vector_floor_loss(
    *,
    seed: int,
    teacher: MatrixSpiratonCell,
    dim: int,
    order: Sequence[str],
    n_samples: int,
    epochs: int,
    lr: float,
    input_scale: float = 1.0,
) -> Dict[str, float]:
    """Plancher : un modèle dont les opérateurs commutent trivialement.

    On approxime la cible ordonnée par une **unique matrice linéaire** ``M`` (le
    modèle le plus expressif qui n'a AUCUNE notion d'ordre de composition : une
    application linéaire est order-free par nature). Si la tâche-avec-ordre est
    bien *infaisable optimalement sans non-commutativité*, ce plancher doit
    garder une perte résiduelle franchement au-dessus du modèle matriciel
    convergé. Si au contraire il atteint une perte aussi basse, la tâche
    n'exigeait pas l'ordre → la comparaison ordonné/commutant serait creuse.

    NB : une seule matrice ``M`` est en fait le MEILLEUR cas pour un modèle
    order-free face à une cible ``compose(x, order)`` qui est elle-même linéaire
    en ``x`` ; le « plancher » mesure donc à quel point la cible composée
    s'éloigne d'être reproductible par une application linéaire — c.-à-d. il ne
    teste pas l'ordre mais la non-linéarité. Pour tester l'ordre proprement on
    compare deux modèles matriciels (ordonné vs commutant). Ce plancher reste
    rapporté comme garde-fou de validité de la tâche.
    """
    gen = torch.Generator().manual_seed(seed + 100003)
    x = torch.randn(n_samples, dim, generator=gen) * input_scale
    with torch.no_grad():
        y = teacher.compose(x, order)

    M = torch.nn.Parameter(torch.eye(dim) + torch.randn(dim, dim, generator=gen) * 0.1)
    opt = torch.optim.Adam([M], lr=lr)
    for _ in range(epochs):
        pred = x @ M.t()
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = float(torch.mean((x @ M.t() - y) ** 2))
    return {"vector_floor_loss_final": final}


# ----------------------------------------------------------------------------
# Balayage multi-graines + agrégation
# ----------------------------------------------------------------------------


@dataclass
class SweepReport:
    config: Dict[str, object]
    ordered: List[TrajectoryReport]
    commuting: List[TrajectoryReport]
    vector_floor: List[float]
    per_seed_advantage: List[float]  # Δ_ordre − Δ_commuting-SYM, par graine (Tour 11)
    summary: Dict[str, float]
    # Tour 12 : contrôle commutant DURCI (asymétrique, bien conditionné).
    # ``commuting_asym_wc`` est la condition de DÉCISION ; ``commuting`` (symétrique)
    # est GARDÉE comme covariable (non-régression Tour 11 + preuve du confond
    # symétrie). ``per_seed_advantage_proper`` = Δ_ordre − Δ_commuting_asym_wc.
    commuting_asym_wc: List[TrajectoryReport] = field(default_factory=list)
    per_seed_advantage_proper: List[float] = field(default_factory=list)
    # Diagnostics de construction du contrôle durci (preuve d'appariement), par graine :
    wc_kappa: List[float] = field(default_factory=list)         # κ(P)
    wc_max_commutator: List[float] = field(default_factory=list)  # max‖[W_a,W_b]‖ (≈0)
    wc_mean_asym: List[float] = field(default_factory=list)     # moyenne ‖W−Wᵀ‖ du contrôle
    ordered_mean_asym: List[float] = field(default_factory=list)  # moyenne ‖W−Wᵀ‖ ordonné


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _variance(xs: Sequence[float]) -> float:
    """Variance de population (sans numpy). Sert à mesurer la dispersion de la
    « source 2 » sur le CONTRÔLE (Tour 13b) — critère orthogonal de sélection L2.
    """
    n = len(xs)
    if n == 0:
        return float("nan")
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / n


def _binom_tail_ge(k: int, n: int, p: float = 0.5) -> float:
    """P(X ≥ k) pour X ~ Binomiale(n, p). Calcul exact entier, sans scipy."""
    from math import comb

    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def _percentile(xs: Sequence[float], q: float) -> float:
    """q-ième percentile (q ∈ [0,100]) par interpolation linéaire, sans numpy."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return s[0]
    pos = (q / 100.0) * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _mean_asymmetry(teacher: MatrixSpiratonCell) -> float:
    """Moyenne sur les 4 opérateurs de ``‖W_op − W_opᵀ‖`` (Frobenius).

    Mesure le degré d'asymétrie des matrices d'un enseignant. Vaut 0 pour le
    contrôle commutant SYMÉTRIQUE (``Q diag Qᵀ``) ; non nul pour l'ordonné et
    pour le contrôle commutant DURCI (``P diag P⁻¹``, P non orthogonale).
    """
    with torch.no_grad():
        vals = []
        for name in OPS:
            w = getattr(teacher, f"W_{name}")
            vals.append(float(torch.linalg.norm(w - w.t())))
    return sum(vals) / len(vals)


def _max_commutator(teacher: MatrixSpiratonCell) -> float:
    """max sur les paires de ``‖[W_a, W_b]‖`` — preuve qu'un contrôle commute."""
    with torch.no_grad():
        return float(max(teacher.commutators().values()))


def run_sweep(
    *,
    seeds: Sequence[int],
    dim: int = 6,
    init_scale: float = 0.5,
    order: Sequence[str] = ("add", "sub", "mul", "div"),
    n_samples: int = 256,
    epochs: int = 400,
    lr: float = 5e-3,
    input_scale: float = 1.0,
    l2_reg: float = 0.0,
) -> SweepReport:
    """Balayage déclaré d'avance : tâche-avec-ordre vs contrôles order-détruits.

    Pour chaque graine on construit, à partir de **la même graine de générateur**
    (mêmes tirages → comparaison appariée) :
      - l'enseignant **ordonné** non-commutant (asymétrique) ;
      - le contrôle **commuting** SYMÉTRIQUE (``Q diag Qᵀ``, Tour 11) — GARDÉ
        comme covariable (non-régression + preuve du confond symétrie) ;
      - le contrôle **commuting_asym_wc** DURCI (``P diag P⁻¹``, P non
        orthogonale bien conditionnée, Tour 12) — la condition de DÉCISION.
    Un élève est entraîné sur chacun (même init, mêmes données). On enregistre
    les trajectoires de commutateur, les deux avantages (vs SYM, vs ASYM-WC), et
    les diagnostics de construction du contrôle durci (κ, commutateur, asymétrie).

    ``l2_reg`` (Tour 13b) est passé **STRICTEMENT IDENTIQUE** aux TROIS conditions
    (ordered, commuting, commuting_asym_wc) — équité dure : aucune condition ne
    reçoit une régularisation différente (sinon on rouvrirait un confond, comme
    la symétrie du Tour 11). Défaut ``0.0`` = balayage Tour 12 bit-à-bit.
    """
    ordered: List[TrajectoryReport] = []
    commuting: List[TrajectoryReport] = []
    commuting_asym_wc: List[TrajectoryReport] = []
    floors: List[float] = []
    advantages: List[float] = []          # Δ_ordre − Δ_commuting-SYM (Tour 11)
    advantages_proper: List[float] = []   # Δ_ordre − Δ_commuting_asym_wc (Tour 12, DÉCISION)
    wc_kappa: List[float] = []
    wc_max_comm: List[float] = []
    wc_mean_asym: List[float] = []
    ord_mean_asym: List[float] = []

    for seed in seeds:
        # Enseignants construits sur des générateurs seedés (déterministe).
        gen_nc = torch.Generator().manual_seed(seed + 7919)
        gen_c = torch.Generator().manual_seed(seed + 7919)
        gen_wc = torch.Generator().manual_seed(seed + 7919)
        teacher_nc = _noncommuting_teacher(dim, init_scale, gen_nc)
        teacher_c = _commuting_teacher(dim, init_scale, gen_c)
        teacher_wc = _commuting_teacher_well_conditioned(dim, init_scale, gen_wc)
        # κ(P) re-calculé séparément (générateur dédié, déterministe) pour le rapport.
        gen_kappa = torch.Generator().manual_seed(seed + 7919)
        _, kappa_p = _well_conditioned_eigenbasis(dim, init_scale, gen_kappa)

        r_ord = train_one(
            seed=seed, condition="ordered", teacher=teacher_nc, dim=dim,
            init_scale=init_scale, order=order, n_samples=n_samples,
            epochs=epochs, lr=lr, input_scale=input_scale, l2_reg=l2_reg,
        )
        r_com = train_one(
            seed=seed, condition="commuting", teacher=teacher_c, dim=dim,
            init_scale=init_scale, order=order, n_samples=n_samples,
            epochs=epochs, lr=lr, input_scale=input_scale, l2_reg=l2_reg,
        )
        r_wc = train_one(
            seed=seed, condition="commuting_asym_wc", teacher=teacher_wc, dim=dim,
            init_scale=init_scale, order=order, n_samples=n_samples,
            epochs=epochs, lr=lr, input_scale=input_scale, l2_reg=l2_reg,
        )
        floor = vector_floor_loss(
            seed=seed, teacher=teacher_nc, dim=dim, order=order,
            n_samples=n_samples, epochs=epochs, lr=lr, input_scale=input_scale,
        )["vector_floor_loss_final"]

        ordered.append(r_ord)
        commuting.append(r_com)
        commuting_asym_wc.append(r_wc)
        floors.append(floor)
        advantages.append(r_ord.delta_total - r_com.delta_total)
        advantages_proper.append(r_ord.delta_total - r_wc.delta_total)
        wc_kappa.append(kappa_p)
        wc_max_comm.append(_max_commutator(teacher_wc))
        wc_mean_asym.append(_mean_asymmetry(teacher_wc))
        ord_mean_asym.append(_mean_asymmetry(teacher_nc))

    n = len(seeds)
    delta_ord = [r.delta_total for r in ordered]
    delta_com = [r.delta_total for r in commuting]
    delta_wc = [r.delta_total for r in commuting_asym_wc]
    n_advantage_pos = sum(1 for a in advantages if a > 0.0)
    n_advantage_proper_pos = sum(1 for a in advantages_proper if a > 0.0)
    med_delta_ord = _median(delta_ord)
    med_delta_com = _median(delta_com)
    med_delta_wc = _median(delta_wc)
    # ratio d'appariement d'asymétrie (médiane des ratios par graine).
    asym_ratios = [
        (w / o) if o > 0.0 else float("nan")
        for w, o in zip(wc_mean_asym, ord_mean_asym)
    ]

    summary = {
        "n_seeds": float(n),
        # médianes des deltas (3 conditions)
        "median_delta_ordered": med_delta_ord,
        "median_delta_commuting": med_delta_com,
        "median_delta_commuting_asym_wc": med_delta_wc,
        # avantage Tour 11 (vs SYM) — non-régression / preuve du confond
        "median_advantage": _median(advantages),
        "n_advantage_positive": float(n_advantage_pos),
        "binom_p_ge_advantage": _binom_tail_ge(n_advantage_pos, n, 0.5),
        # avantage PROPRE Tour 12 (vs ASYM-WC) — la mesure de DÉCISION
        "median_advantage_proper": _median(advantages_proper),
        "p05_advantage_proper": _percentile(advantages_proper, 5.0),
        "n_advantage_proper_positive": float(n_advantage_proper_pos),
        "binom_p_ge_advantage_proper": _binom_tail_ge(n_advantage_proper_pos, n, 0.5),
        # part attribuable à la symétrie (preuve visible du confond)
        "symmetry_share": _median(advantages) - _median(advantages_proper),
        # variance de Δ du CONTRÔLE durci (= la « source 2 », dérive générique
        # non-commutante d'Adam, indépendante de l'ordre). C'est la quantité que
        # le critère ORTHOGONAL de sélection de λ (Tour 13b) cherche à réduire —
        # mesurée sur le CONTRÔLE SEUL, JAMAIS sur l'avantage propre.
        "var_delta_commuting_asym_wc": _variance(delta_wc),
        "std_delta_commuting_asym_wc": _variance(delta_wc) ** 0.5,
        # bornes anti-dissipation
        "median_total_init_ordered": _median([r.total_init for r in ordered]),
        "median_total_final_ordered": _median([r.total_final for r in ordered]),
        "max_total_final_ordered": max(r.total_final for r in ordered),
        "max_total_init_ordered": max(r.total_init for r in ordered),
        "max_total_final_commuting_asym_wc": max(r.total_final for r in commuting_asym_wc),
        # convergence (preuve de comparabilité)
        "median_loss_final_ordered": _median([r.loss_final for r in ordered]),
        "median_loss_final_commuting": _median([r.loss_final for r in commuting]),
        "median_loss_final_commuting_asym_wc": _median([r.loss_final for r in commuting_asym_wc]),
        "median_loss_init_ordered": _median([r.loss_init for r in ordered]),
        "median_vector_floor_loss": _median(floors),
        # asymétrie des pôles (mul-add dextro vs div-sub lévo)
        "median_delta_mul_add_ordered": _median([r.delta_mul_add for r in ordered]),
        "median_delta_div_sub_ordered": _median([r.delta_div_sub for r in ordered]),
        # construction du contrôle DURCI (preuve d'appariement et qu'il commute)
        "wc_kappa_min": min(wc_kappa),
        "wc_kappa_max": max(wc_kappa),
        "wc_max_commutator": max(wc_max_comm),
        "median_wc_mean_asym": _median(wc_mean_asym),
        "median_ordered_mean_asym": _median(ord_mean_asym),
        "median_asym_ratio_wc_over_ordered": _median(asym_ratios),
    }

    return SweepReport(
        config={
            "seeds": list(seeds), "dim": dim, "init_scale": init_scale,
            "order": tuple(order), "n_samples": n_samples, "epochs": epochs,
            "lr": lr, "input_scale": input_scale, "l2_reg": l2_reg,
            "beta_mult": BETA_MULT, "kappa_max": KAPPA_MAX,
        },
        ordered=ordered,
        commuting=commuting,
        vector_floor=floors,
        per_seed_advantage=advantages,
        summary=summary,
        commuting_asym_wc=commuting_asym_wc,
        per_seed_advantage_proper=advantages_proper,
        wc_kappa=wc_kappa,
        wc_max_commutator=wc_max_comm,
        wc_mean_asym=wc_mean_asym,
        ordered_mean_asym=ord_mean_asym,
    )
