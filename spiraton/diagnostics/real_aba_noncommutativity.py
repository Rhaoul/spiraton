"""Tour 14 — non-commutativité acquise sous l'ordre SÉMANTIQUE RÉEL du corpus.

Le fil 11-13 a établi, sur une tâche **synthétique** (enseignant non-commutant
figé, ordre des opérateurs imposé par nous), que la non-commutativité
``Σ‖[W_a,W_b]‖`` s'**ACQUIERT** pendant l'entraînement quand la cible porte de
l'ordre — avantage propre médian ≈ +6.9 contre un contrôle order-détruit durci
(robuste-mais-non-universel, queue gauche structurelle ~10 %).

Ce module pose la question du Tour 14 : la non-commutativité acquise se
manifeste-t-elle quand l'ordre porte un **sens sémantique RÉEL du corpus**
— le cycle A→B→A′ et le retournement ``<DX><OUT> → <LV><IN>`` au troisième
segment — plutôt qu'un ordre synthétique imposé ?

Construction de la cible (déclarée A PRIORI, jamais ajustée sur l'avantage)
--------------------------------------------------------------------------
On **réutilise** intégralement la machinerie synthétique de
``acquired_noncommutativity`` (enseignant non-commutant figé, élève MSE pure,
trajectoire du commutateur par epoch, plancher vectoriel, bornes
anti-dissipation, ``_percentile``/``_binom_tail_ge``). On **remplace** la
source d'ordre :

* On lit chaque **cycle ABA réel** (``data.aba.iter_aba_cycles``) et on
  featurise ses trois segments en trois vecteurs 33D **réels** via le
  ``PhonemeFeaturizer`` natif (``is_fallback=False``, ``pool="mean"``). Les
  vecteurs portent le **contenu** ; aucune étiquette n'entre dans l'entrée.

* L'**ordre** est la **structure donnée du cycle**, lue (jamais fabriquée) :
  la chiralité de chaque segment, ``cycle.seg_*.chirality`` (DX/LV). Le mapping
  chiralité → sens de composition est FIXÉ d'avance et incarne le pivot
  spirale (``matrix_cell.py`` l.91 : la branche lévogyre renverse l'ordre) :

      DX (dextrogyre, déploiement)  → ordre AVANT   : OPS = (add,sub,mul,div)
      LV (lévogyre,   retour)       → ordre ARRIÈRE : reversed(OPS)

  Un cycle de clôture canonique (A=DX, B=DX, A′=LV) donne donc la séquence de
  sens ``(avant, avant, arrière)`` : deux déploiements puis le **retournement**
  au troisième segment. C'est exactement le ``<DX><OUT>→<LV><IN>`` du corpus.

* La **trajectoire de cycle** chaîne les trois segments réels dans l'ordre A→B→A′
  donné par le corpus : on part de ``x_A``, on compose selon le sens du
  segment A, on **ajoute** le vecteur réel ``x_B``, on compose selon le sens du
  segment B, on ajoute ``x_Ap``, on compose selon le sens du segment A′. La
  cible enseignante ``y`` est l'état final ; l'élève la reproduit à partir des
  **mêmes** vecteurs réels et des **mêmes** sens.

Contrôle **ordre-détruit** (à difficulté appariée)
--------------------------------------------------
Mêmes vecteurs réels, même enseignant, **même multiset de sens** ``{avant,
avant, arrière}`` — mais l'assignation segment→sens est **permutée** par une
permutation seedée déterministe par cycle. Le **retournement lévogyre tombe sur
un segment permuté** : la position réelle du pivot A→B→A′ est détruite, sans
changer la difficulté (même enseignant, mêmes vecteurs, même nombre de
retournements). C'est l'incarnation « détruire l'ORDRE RÉEL » du REFUS#2.

La cible ne sert qu'à **DONNER DU GRADIENT ORDONNÉ**. La variable mesurée est
``Δ‖[W_a,W_b]‖`` conditionnelle (ordre-corpus − ordre-détruit). On ne rapporte
**JAMAIS** ``cos(pred, A′)`` : ce tour n'est pas une re-mesure de CHANTIER5.

Garde REFUS héritée
-------------------
Aucun terme de perte ne récompense le commutateur (perte = MSE pure sur la
cible composée). ``matrix_cell.py``, ``data/aba.py``, ``data/featurizers.py``,
``core/`` et ``acquired_noncommutativity.py`` sont **réutilisés en import**,
jamais modifiés. Seeds fixés, déterminisme bit-à-bit. Featurizer **natif**
requis ; s'il est absent, le diagnostic se **skippe** (jamais de bascule sur le
hachage : le fallback fabriquerait le contenu).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..experimental.matrix_cell import MatrixSpiratonCell, OPS
from ..data.aba import AbaCycle, iter_aba_cycles
from .acquired_noncommutativity import (
    PAIR_DIV_SUB,
    PAIR_MUL_ADD,
    TrajectoryReport,
    _binom_tail_ge,
    _median,
    _noncommuting_teacher,
    _percentile,
    _record,
    _variance,
)

def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Corrélation de Pearson (sans numpy). NaN si variance nulle ou n<2."""
    n = len(a)
    if n < 2 or len(b) != n:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sa = sum((x - ma) ** 2 for x in a) ** 0.5
    sb = sum((y - mb) ** 2 for y in b) ** 0.5
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return cov / (sa * sb)


# Mapping chiralité → sens de composition, FIXÉ a priori (pivot spirale l.91).
#
# On compose la **paire primaire dextrogyre** ⊗⊕ = (mul, add) — celle dont le
# Tour 11 suit le commutateur ‖[W_mul, W_add]‖ — et NON les quatre opérateurs.
# Raison de conditionnement déclarée d'avance (pas un fit) : composer les 4 OPS
# fait 4 matmuls/segment × 3 = 12 en profondeur ; à dim=33 cela force soit un
# init_scale si bas que le produit est ≈ I (tâche order-FREE, plancher vectoriel
# ≈ perte élève → null PAR CONSTRUCTION), soit un blow-up. La paire (2 matmuls
# /segment, 6 en profondeur) garde une **fenêtre** où la tâche est order-dépendante
# ET convergente, et le retournement DX→LV agit exactement sur la paire dont on
# mesure le commutateur. Le retournement = inversion de l'ordre de la paire
# (``(mul,add)`` → ``(add,mul)``), seul endroit où ‖[W_mul,W_add]‖>0 importe.
_PRIMARY_PAIR: Tuple[str, ...] = ("mul", "add")
_FORWARD: Tuple[str, ...] = _PRIMARY_PAIR                   # DX : déploiement avant
_BACKWARD: Tuple[str, ...] = tuple(reversed(_PRIMARY_PAIR)) # LV : retour arrière


# ----------------------------------------------------------------------------
# Lecture du corpus → vecteurs réels + sens de composition (lus, jamais forgés)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RealCycleSample:
    """Un cycle ABA réel prêt pour la composition.

    ``x`` : (3, d) — les trois vecteurs 33D réels (A, B, A′), contenu seul.
    ``senses_corpus`` : 3 ordres de composition donnés par la chiralité réelle
        du corpus (DX→avant, LV→arrière). C'est l'ORDRE RÉEL.
    """

    x: torch.Tensor                              # (3, d)
    senses_corpus: Tuple[Tuple[str, ...], ...]   # 3 ordres


def _sense_for_chirality(chirality: str) -> Tuple[str, ...]:
    """DX → avant (OPS) ; LV → arrière (reversed OPS). Le pivot spirale."""
    return _BACKWARD if chirality == "LV" else _FORWARD


def _fixed_projection(in_dim: int, out_dim: int, seed: int = 1789) -> torch.Tensor:
    """Projection gaussienne FIXE et partagée 33→out_dim (jamais ajustée).

    Même geste que le Tour 7 (dims phonétiques 15→2 par matrice gaussienne fixe
    partagée) : ramener le contenu réel 33D dans le **régime dimensionnel du
    Tour 12** (dim=6), où l'échelle du commutateur (≈30 à l'init, 6 paires de
    matrices 6×6) laisse lire un Δ d'ordre franc — à dim=33 l'init ≈404 noie le
    Δ (~±2) dans le bruit. La projection préserve la **direction relative** des
    trois vecteurs de segment (donc l'information d'ordre qu'ils portent), elle
    ne fait que réduire la dimension de calcul. Fixe, seedée, identique aux trois
    segments et aux deux conditions : aucune fuite, aucun fit.
    """
    gen = torch.Generator().manual_seed(seed)
    P = torch.randn(out_dim, in_dim, generator=gen) / (in_dim ** 0.5)
    return P


def load_real_cycles(
    corpus_paths: Sequence[str],
    featurizer,
    *,
    max_cycles: Optional[int] = None,
    require_closure: bool = True,
    normalize: bool = True,
    proj_dim: Optional[int] = None,
    proj_seed: int = 1789,
) -> List[RealCycleSample]:
    """Lit les cycles réels et les featurise (contenu réel, ordre lu du corpus).

    ``featurizer`` : un ``PhonemeFeaturizer`` natif (``is_fallback`` doit être
    False — vérifié). ``require_closure`` : ne garder que les cycles dont la
    structure de clôture canonique est présente (``is_closure``) — la garantie
    que la chiralité lue est bien (DX, DX, LV).

    ``normalize`` (défaut True) : L2-normalise chaque vecteur de segment à norme
    unité. C'est un conditionnement d'ÉCHELLE — il préserve la **direction**
    (le contenu phonético-sémantique réel et donc l'information d'ordre portée
    par les vecteurs distincts), il ne fait que borner la magnitude pour que la
    composition profonde (12 matmuls) reste numériquement saine. Sans lui, les
    pertes valent ~1e11 et le contraste ordre/détruit est noyé par l'échelle.
    """
    if getattr(featurizer, "is_fallback", True):
        raise ValueError(
            "Tour 14 exige le featurizer natif (is_fallback=False) ; le fallback "
            "de hachage fabriquerait le contenu (faute REFUS)."
        )
    P = None
    if proj_dim is not None:
        P = _fixed_projection(int(featurizer.dim), int(proj_dim), seed=proj_seed)

    samples: List[RealCycleSample] = []
    for path in corpus_paths:
        for cyc in iter_aba_cycles(path):
            if require_closure and not cyc.is_closure:
                continue
            segs = (cyc.seg_a, cyc.seg_b, cyc.seg_a_prime)
            texts = [s.text for s in segs]
            if any(not t for t in texts):
                continue
            vecs = [featurizer(t) for t in texts]
            x = torch.stack(vecs, dim=0)  # (3, d33)
            if not torch.isfinite(x).all():
                continue
            if P is not None:
                x = x @ P.t()             # (3, proj_dim) — projection fixe partagée
            if normalize:
                norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                x = x / norms
            senses = tuple(_sense_for_chirality(s.chirality) for s in segs)
            samples.append(RealCycleSample(x=x, senses_corpus=senses))
            if max_cycles is not None and len(samples) >= max_cycles:
                return samples
    return samples


# ----------------------------------------------------------------------------
# Composition de cycle (chaînage A→B→A′ des vecteurs réels selon les sens)
# ----------------------------------------------------------------------------


def compose_cycle(
    cell: MatrixSpiratonCell,
    x: torch.Tensor,
    senses: Sequence[Sequence[str]],
) -> torch.Tensor:
    """Chaîne les 3 segments réels dans l'ordre A→B→A′ selon ``senses``.

    ``x`` : (B, 3, d). On part de x[:,0], on compose selon senses[0], on ajoute
    x[:,1], on compose selon senses[1], on ajoute x[:,2], on compose selon
    senses[2]. Retourne (B, d). La cellule expose ``compose(t, order)`` (brut,
    sans activation) ; aucune copie de matrice n'est faite.
    """
    t = x[:, 0, :]
    t = cell.compose(t, senses[0])
    t = t + x[:, 1, :]
    t = cell.compose(t, senses[1])
    t = t + x[:, 2, :]
    t = cell.compose(t, senses[2])
    return t


def _permuted_senses(
    senses_corpus: Sequence[Sequence[str]], perm: Sequence[int]
) -> Tuple[Tuple[str, ...], ...]:
    """Réassigne le multiset {avant,avant,arrière} aux 3 slots selon ``perm``.

    Détruit la POSITION réelle du retournement lévogyre sans changer la
    difficulté (même multiset de sens).
    """
    return tuple(tuple(senses_corpus[perm[i]]) for i in range(3))


def _build_batch(
    samples: Sequence[RealCycleSample],
) -> torch.Tensor:
    """Empile les vecteurs (N, 3, d). Les sens sont gérés par cycle ailleurs."""
    return torch.stack([s.x for s in samples], dim=0)


# ----------------------------------------------------------------------------
# Entraînement instrumenté (MSE pure ; commutateur = observateur passif)
# ----------------------------------------------------------------------------


def _targets_corpus(
    teacher: MatrixSpiratonCell, samples: Sequence[RealCycleSample]
) -> torch.Tensor:
    """y enseignant pour l'ordre-corpus, calculé cycle par cycle (sens variables)."""
    outs = []
    with torch.no_grad():
        for s in samples:
            xb = s.x.unsqueeze(0)  # (1,3,d)
            outs.append(compose_cycle(teacher, xb, s.senses_corpus)[0])
    return torch.stack(outs, dim=0)  # (N, d)


def _targets_destroyed(
    teacher: MatrixSpiratonCell,
    samples: Sequence[RealCycleSample],
    perms: Sequence[Sequence[int]],
) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for s, perm in zip(samples, perms):
            xb = s.x.unsqueeze(0)
            senses = _permuted_senses(s.senses_corpus, perm)
            outs.append(compose_cycle(teacher, xb, senses)[0])
    return torch.stack(outs, dim=0)


def _student_preds(
    student: MatrixSpiratonCell,
    x: torch.Tensor,
    senses_per_cycle: Sequence[Sequence[Sequence[str]]],
) -> torch.Tensor:
    """Prédictions de l'élève, groupées par sens identiques (efficacité).

    Les cycles partageant le même triplet de sens sont composés en un seul
    batch. Le résultat est ré-ordonné à l'index d'origine. Différentiable.
    """
    n = x.size(0)
    groups: Dict[Tuple[Tuple[str, ...], ...], List[int]] = {}
    for i, senses in enumerate(senses_per_cycle):
        key = tuple(tuple(o) for o in senses)
        groups.setdefault(key, []).append(i)
    out = x.new_zeros(n, x.size(-1))
    for key, idxs in groups.items():
        idx_t = torch.tensor(idxs, dtype=torch.long)
        xb = x.index_select(0, idx_t)              # (k,3,d)
        pred = compose_cycle(student, xb, key)     # (k,d)
        out.index_copy_(0, idx_t, pred)
    return out


def train_one_real(
    *,
    seed: int,
    condition: str,
    teacher: MatrixSpiratonCell,
    x: torch.Tensor,                                   # (N,3,d)
    senses_per_cycle: Sequence[Sequence[Sequence[str]]],
    targets: torch.Tensor,                             # (N,d)
    dim: int,
    init_scale: float,
    epochs: int,
    lr: float,
) -> TrajectoryReport:
    """Entraîne un élève (MSE pure) à reproduire la cible composée réelle.

    Perte = MSE(compose_cycle(student, x, senses), targets). AUCUN terme ne
    touche le commutateur ; ``commutators()`` est enregistré par epoch en
    observateur passif (réutilise ``_record`` de ``acquired_noncommutativity``).
    """
    gen = torch.Generator().manual_seed(seed)
    student = MatrixSpiratonCell(input_size=dim, init_scale=init_scale)
    with torch.no_grad():
        eye = torch.eye(dim)
        for name in OPS:
            getattr(student, f"W_{name}").copy_(
                eye + torch.randn(dim, dim, generator=gen) * init_scale
            )
        student.bias.zero_()

    opt = torch.optim.Adam(student.parameters(), lr=lr)

    report = TrajectoryReport(seed=seed, condition=condition)
    with torch.no_grad():
        pred0 = _student_preds(student, x, senses_per_cycle)
        loss0 = float(torch.mean((pred0 - targets) ** 2))
    _record(student, report, loss0)

    for _ in range(epochs):
        pred = _student_preds(student, x, senses_per_cycle)
        mse = torch.mean((pred - targets) ** 2)
        opt.zero_grad()
        mse.backward()
        opt.step()
        _record(student, report, float(mse.detach()))

    return report


# ----------------------------------------------------------------------------
# Plancher vectoriel (validité : la tâche réelle est-elle order-dépendante ?)
# ----------------------------------------------------------------------------


def real_vector_floor_loss(
    *,
    seed: int,
    x: torch.Tensor,
    targets: torch.Tensor,
    dim: int,
    epochs: int,
    lr: float,
) -> float:
    """Plancher order-free : une UNIQUE matrice M appliquée à chaque segment.

    Modèle sans aucune notion d'ordre de composition : il applique la **même**
    application linéaire à chaque segment puis somme (``M·x_A + M·x_B + M·x_Ap``
    via chaînage additif). S'il atteint la même perte que l'élève matriciel
    convergé, la tâche n'exigeait pas l'ordre → comparaison creuse. S'il reste
    franchement au-dessus, la tâche réelle est bien order-dépendante (valide).
    """
    gen = torch.Generator().manual_seed(seed + 100003)
    M = torch.nn.Parameter(torch.eye(dim) + torch.randn(dim, dim, generator=gen) * 0.1)
    opt = torch.optim.Adam([M], lr=lr)

    def _floor_pred() -> torch.Tensor:
        t = x[:, 0, :] @ M.t()
        t = (t + x[:, 1, :]) @ M.t()
        t = (t + x[:, 2, :]) @ M.t()
        return t

    for _ in range(epochs):
        pred = _floor_pred()
        loss = torch.mean((pred - targets) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(torch.mean((_floor_pred() - targets) ** 2))


# ----------------------------------------------------------------------------
# Balayage multi-graines
# ----------------------------------------------------------------------------


@dataclass
class RealSweepReport:
    config: Dict[str, object]
    corpus: List[TrajectoryReport]
    destroyed: List[TrajectoryReport]
    vector_floor: List[float]
    per_seed_advantage: List[float]  # Δ_corpus − Δ_destroyed, par graine
    summary: Dict[str, float]


def _make_perm(seed: int) -> Tuple[int, ...]:
    """Permutation seedée déterministe de {0,1,2} (l'assignation segment→sens).

    Garde de validité du contrôle : un cycle de clôture canonique a deux sens
    « avant » identiques (slots 0,1) et le retournement « arrière » au slot 2.
    Une permutation qui ne fait qu'échanger les deux slots avant laisse le
    triplet de sens INCHANGÉ → contrôle vide. On exige donc que la **position du
    retournement** change réellement : ``perm[2] != 2`` (le slot final reçoit un
    sens « avant », et l'arrière migre vers le slot 0 ou 1). C'est exactement
    « détruire la POSITION réelle du pivot A→B→A′ » (REFUS#2).
    """
    gen = torch.Generator().manual_seed(seed)
    while True:
        perm = tuple(int(i) for i in torch.randperm(3, generator=gen))
        if perm[2] != 2:
            return perm


def run_real_sweep(
    *,
    samples: Sequence[RealCycleSample],
    seeds: Sequence[int],
    dim: int,
    init_scale: float = 0.5,
    epochs: int = 400,
    lr: float = 5e-3,
) -> RealSweepReport:
    """Balayage Tour 14 : ordre-corpus vs ordre-détruit, vecteurs réels partagés.

    Pour chaque graine : enseignant non-commutant figé (même fabrique que
    Tours 11-13) ; cibles corpus et détruit calculées sur les MÊMES vecteurs
    réels ; deux élèves (même init par graine) entraînés en MSE pure ; deltas du
    commutateur enregistrés. ``avantage_réel = Δ_corpus − Δ_détruit``.
    """
    x = _build_batch(samples)  # (N,3,d)
    senses_corpus = [s.senses_corpus for s in samples]

    corpus_reports: List[TrajectoryReport] = []
    destroyed_reports: List[TrajectoryReport] = []
    floors: List[float] = []
    advantages: List[float] = []

    for seed in seeds:
        gen_t = torch.Generator().manual_seed(seed + 7919)
        teacher = _noncommuting_teacher(dim, init_scale, gen_t)

        # Permutations par cycle, seedées de façon déterministe et indépendante
        # de la graine d'entraînement (l'ordre-détruit est FIXÉ par le cycle).
        perms = [_make_perm(13 + i) for i in range(len(samples))]
        senses_dest = [_permuted_senses(s.senses_corpus, p)
                       for s, p in zip(samples, perms)]

        y_corpus = _targets_corpus(teacher, samples)
        y_dest = _targets_destroyed(teacher, samples, perms)

        r_corpus = train_one_real(
            seed=seed, condition="corpus", teacher=teacher, x=x,
            senses_per_cycle=senses_corpus, targets=y_corpus,
            dim=dim, init_scale=init_scale, epochs=epochs, lr=lr,
        )
        r_dest = train_one_real(
            seed=seed, condition="destroyed", teacher=teacher, x=x,
            senses_per_cycle=senses_dest, targets=y_dest,
            dim=dim, init_scale=init_scale, epochs=epochs, lr=lr,
        )
        floor = real_vector_floor_loss(
            seed=seed, x=x, targets=y_corpus, dim=dim, epochs=epochs, lr=lr,
        )

        corpus_reports.append(r_corpus)
        destroyed_reports.append(r_dest)
        floors.append(floor)
        advantages.append(r_corpus.delta_total - r_dest.delta_total)

    n = len(seeds)
    delta_corpus = [r.delta_total for r in corpus_reports]
    delta_dest = [r.delta_total for r in destroyed_reports]
    n_pos = sum(1 for a in advantages if a > 0.0)

    # Confond de DIFFICULTÉ (REFUS#2) : le contrôle détruit doit converger
    # comparablement. On mesure la corrélation entre l'avantage et l'écart de
    # perte (détruit − corpus) : si elle est nettement négative, une part de
    # l'avantage est portée par la difficulté de fit (dérive générique d'Adam,
    # « source 2 » du Tour 13), PAS par le contenu d'ordre.
    loss_gap = [
        rd.loss_final - rc.loss_final
        for rc, rd in zip(corpus_reports, destroyed_reports)
    ]
    corr_adv_lossgap = _pearson(advantages, loss_gap)
    n_dest_harder = sum(
        1 for rc, rd in zip(corpus_reports, destroyed_reports)
        if rd.loss_final > 1.5 * max(rc.loss_final, 1e-12)
    )

    summary = {
        "n_seeds": float(n),
        "n_cycles": float(len(samples)),
        # médianes des deltas
        "median_delta_corpus": _median(delta_corpus),
        "median_delta_destroyed": _median(delta_dest),
        # avantage réel (la mesure de DÉCISION)
        "median_advantage_real": _median(advantages),
        "p05_advantage_real": _percentile(advantages, 5.0),
        "n_advantage_positive": float(n_pos),
        "binom_p_ge_advantage": _binom_tail_ge(n_pos, n, 0.5),
        "var_advantage_real": _variance(advantages),
        # bornes anti-dissipation
        "median_total_init_corpus": _median([r.total_init for r in corpus_reports]),
        "median_total_final_corpus": _median([r.total_final for r in corpus_reports]),
        "max_total_final_corpus": max(r.total_final for r in corpus_reports),
        "max_total_init_corpus": max(r.total_init for r in corpus_reports),
        "max_total_final_destroyed": max(r.total_final for r in destroyed_reports),
        # convergence (preuve d'appariement de difficulté)
        "median_loss_final_corpus": _median([r.loss_final for r in corpus_reports]),
        "median_loss_final_destroyed": _median([r.loss_final for r in destroyed_reports]),
        "median_loss_init_corpus": _median([r.loss_init for r in corpus_reports]),
        "median_vector_floor_loss": _median(floors),
        # confond de difficulté (REFUS#2) — preuve de validité de la comparaison
        "corr_advantage_lossgap": corr_adv_lossgap,
        "n_destroyed_harder": float(n_dest_harder),
        # pôles
        "median_delta_mul_add_corpus": _median([r.delta_mul_add for r in corpus_reports]),
        "median_delta_div_sub_corpus": _median([r.delta_div_sub for r in corpus_reports]),
    }

    return RealSweepReport(
        config={
            "seeds": list(seeds), "dim": dim, "init_scale": init_scale,
            "epochs": epochs, "lr": lr, "n_cycles": len(samples),
            "forward": _FORWARD, "backward": _BACKWARD,
        },
        corpus=corpus_reports,
        destroyed=destroyed_reports,
        vector_floor=floors,
        per_seed_advantage=advantages,
        summary=summary,
    )


# ----------------------------------------------------------------------------
# Résolution des corpus (mêmes règles que les diagnostics existants)
# ----------------------------------------------------------------------------


def default_corpora() -> List[str]:
    """Corpus ABA réels : dataset_aba.txt (5000) + corpus_claude_aba.txt (76)."""
    here = Path(__file__).resolve().parents[2]   # .../spiraton (dépôt PyTorch)
    root = here.parent                            # .../spiraton-enhanced (corpus racine)
    names = ("dataset_aba.txt", "corpus_claude_aba.txt")
    out: List[str] = []
    for name in names:
        for base in (here, root):
            p = base / name
            if p.is_file():
                out.append(str(p))
                break
    return out
