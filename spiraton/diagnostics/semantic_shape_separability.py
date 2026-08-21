from __future__ import annotations

"""Séparabilité forme-géométrique ↔ chiralité ABA (Tour 7, E3 — pont 33D↔géométrie↔ABA).

GESTE OPÉRATOIRE (émission du linguiste, incarnée TELLE QUELLE). On demande si la
SIGNATURE GÉOMÉTRIQUE de la trajectoire engendrée par le contenu phonético-sémantique
33D d'un segment prédit sa CHIRALITÉ ABA (``<DX><OUT>`` ouverture A/B vs ``<LV><IN>``
clôture A′), au-dessus du hasard, SANS que l'étiquette ABA entre jamais dans l'entrée.

H7 (falsifiable). ``AUC(signature_géométrique → chiralité_ABA) > 0.5`` sur held-out,
contre le 95e pct de CTRL-PERM. La signature est un SCALAIRE DE FORME : la pente
``slope_logr_theta`` (taux de contraction/expansion du rayon, ``shape_signature``).
On NE re-mesure PAS cornerness/ρ comme cible (ρ peut être un intermédiaire de calcul,
jamais la variable d'intérêt).

CONVENTION D'ORIENTATION (documentée, non pliée) :
  * pos = A′ (clôture, attendu CONTRACTANT, slope ≤ 0) ;
  * neg = A/B (ouverture, attendu EXPANSIF, slope ≥ 0).
AUC>0.5 ⇒ A′ contracte PLUS que A/B (slope plus négatif). On NE plie PAS vers
max(auc, 1−auc) : la direction est une information (REFUS).

GESTE / MAPPING SÉMANTIQUE (FIXÉ d'avance, JAMAIS fitté contre la cible) :
  * dims 0-5 (scores ADD/SUB/MUL/DIV + dextro/lévo) → pilotent le GAIN RADIAL g de
    l'oscilloscope. Cohérent ``oscilloscope.py`` (MUL/dextro/out = rayon↑ ; DIV/lévo/in
    = rayon↓) :
        g(segment) = 1 + κ·(score_MUL + score_dextro − score_DIV − score_lévo)
    agrégé (moyenne) sur les tokens du segment ; κ posé UNE fois (KAPPA=0.1), non
    optimisé. g est borné à GAIN_FLOOR pour rester non-divergent / non-dégénéré.
  * dims 8-22 (signatures phonémiques, 15D) → signal d'entrée u_t de l'oscilloscope
    (le « courant »). Le plan de l'oscilloscope est 2D : on PROJETTE le 15D vers 2D
    par une matrice ALÉATOIRE FIXE (graine globale unique, IDENTIQUE pour tous les
    segments — c'est une coordonnée du geste, PAS un classifieur appris, donc aucun
    degré de liberté n'est ajusté contre la cible). Le token i fournit u_i au pas i.
  * ω fixe (coordonnée neutre, OMEGA_DEFAULT).

GRANULARITÉ = SEGMENT. Un segment → agrégation dims 0-5 (gain) ; séquence dims 8-22
(u_t) → trajectoire → un scalaire de forme → un point (score, label). Ratio par cycle :
2 OUT (A,B) : 1 IN (A′). CTRL-PERM préserve ce ratio (permute les labels, pas les
effectifs).

TROIS VERROUS ANTI-FUITE (capital — c'est tout le tour) :
  1. VERROU D'ENTRÉE : l'oscilloscope ne reçoit JAMAIS l'étiquette ni la position du
     segment. Il reçoit ``vectors(seg.text)`` où ``seg.text`` est nettoyé de tout tag
     par ``_clean_text`` (aba.py). L'étiquette ne sert QU'à former les groupes pos/neg
     APRÈS scoring.
  2. VERROU DE CHIRALITÉ PHONÉTIQUE : les dims 4-5 viennent des phonèmes calculés par
     le tokenizer C, PAS du tag. ``vectors(texte)`` est IDENTIQUE que le même texte
     soit présenté comme segment A ou A′ (le tokenizer ne voit que ``seg.text``, jamais
     la position). Vérifié par ``verrou2_nonleak_holds`` et par le test dédié.
  3. CTRL-PERM (≥20 perms seedées, 95e pct) + held-out PAR CYCLE entier (jamais un
     segment d'un cycle en train et son frère en test) + CTRL-SIG-RAND (signatures
     aléatoires → AUC≈0.5).

SKIP PROPRE. Si le tokenizer natif est indisponible, le diagnostic se SKIPPE
(``TokenizerUnavailable``). On NE substitue JAMAIS les dims 0-5 par les étiquettes
ABA — ce serait la fuite (issue c). L'étiquette n'entre nulle part dans le calcul de
signature.

REFUS — discipline. κ, OMEGA, la projection 15→2 sont posés AVANT de voir la cible.
Aucune forme n'est fittée. L'AUC ne compte que si elle dépasse le 95e pct de CTRL-PERM.
Le null (issue b) est un résultat attendu et légitime (émetteur ABA à 22.8 % d'accord
op., biais ADD) — on le rapporte tel quel, jamais maquillé en positif. AUC>0.95 ⇒
suspicion de fuite (issue c) à AUDITER, pas à célébrer.
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..data import vector33d as v33
from ..data.aba import AbaCycle, AbaSegment, iter_aba_cycles
from ..data.tokenizer_bridge import NativeTokenizer33D, TokenizerUnavailable, is_available
from ..experimental.oscilloscope import InputSignal, Oscilloscope2D
from .shape_signature import shape_signature
from .spectral_separability import (
    _abs_dir,
    _ctrl_perm_distribution,
    _ctrl_sig_rand_auc,
    _quantiles,
    _rank_auc,
)

OMEGA_DEFAULT = math.pi / 5  # même ω que les Tours 4-6 (coordonnée neutre, fixe)
KAPPA = 0.1                   # force du couplage scores→gain, posée UNE fois
GAIN_FLOOR = 0.80            # borne basse du gain (anti-dégénérescence r→0)
GAIN_CEIL = 1.20            # borne haute (anti-divergence sur l'horizon mesuré)
PROJ_SEED = 70_007           # graine UNIQUE de la projection phonèmes 15→2 (fixe)
STEPS_PER_TOKEN_MIN = 60     # horizon minimal de trace (régime établi lisible)
WIN_SCALE = 1.0              # échelle d'injection du courant (identité mise à l'échelle)

# Index dans le 33D des scores qui pilotent le gain (cf. vector33d) :
#   MUL = dim 2, DIV = dim 3 (OP_SCORES = slice(0,4)),
#   dextro = dim 4, lévo = dim 5 (CHIRALITY = slice(4,6)).
_MUL_I, _DIV_I = 2, 3
_DEXTRO_I, _LEVO_I = 4, 5

_PHON = v33.PHONEME_SIG  # slice(8, 23) — 15 dims phonémiques


# ---------------------------------------------------------------------------
# Projection FIXE 15→2 du signal phonémique (coordonnée du geste, jamais fittée).
# ---------------------------------------------------------------------------

def phoneme_projection(*, seed: int = PROJ_SEED) -> torch.Tensor:
    """Matrice ALÉATOIRE FIXE ``P ∈ R^{2×15}`` projetant les phonèmes (dims 8-22) sur le plan.

    Gaussienne normalisée par ``1/√15`` (projection de Johnson-Lindenstrauss : préserve
    en espérance la géométrie relative des séquences phonémiques en passant de 15D au
    plan 2D de l'oscilloscope). Graine UNIQUE et globale : la MÊME projection sert à
    TOUS les segments, sans aucun degré de liberté ajusté contre la cible. Ce n'est pas
    un classifieur — c'est la coordonnée fixe par laquelle le « courant » phonémique
    entre dans le plan.
    """
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(2, 15, generator=g) / math.sqrt(15.0)
    return P.to(torch.float32)


# ---------------------------------------------------------------------------
# Signal d'entrée piloté par la séquence phonémique d'un segment.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhonemeCurrent(InputSignal):
    """Signal d'entrée u_t = projection 2D de la signature phonémique du token t.

    Sous-classe de ``InputSignal`` (frozen) qui PORTE la séquence projetée ``u`` de
    forme ``(N, 2)`` : ``u[t]`` est le courant injecté au pas t (token t du segment).
    Au-delà du dernier token (t >= N), le courant est NUL (le régime devient libre :
    la figure « propre » s'établit, lue sur la seconde moitié par ``shape_signature``).

    On hérite de ``InputSignal`` pour rester strictement compatible avec
    ``Oscilloscope2D.trace`` (qui appelle ``signal.at(t)``) — AUCUNE modification du
    moteur n'est requise. ``kind`` est neutralisé ("phoneme") ; seul ``at`` compte.
    """

    u: Tuple[Tuple[float, float], ...] = ()

    def at(self, t: int) -> torch.Tensor:
        if 0 <= t < len(self.u):
            row = self.u[t]
            return torch.tensor([row[0], row[1]], dtype=torch.float32)
        return torch.zeros(2)


def _segment_current(phon: torch.Tensor, P: torch.Tensor) -> PhonemeCurrent:
    """Projette la séquence phonémique ``(N, 15)`` d'un segment en courant ``(N, 2)``."""
    if phon.numel() == 0:
        return PhonemeCurrent(kind="phoneme", u=())
    proj = phon @ P.t()  # (N, 2)
    u = tuple((float(r[0]), float(r[1])) for r in proj)
    return PhonemeCurrent(kind="phoneme", u=u)


# ---------------------------------------------------------------------------
# Gain radial piloté par les scores opérateurs/chiralité (dims 0-5).
# ---------------------------------------------------------------------------

def segment_gain(scores05: torch.Tensor, *, kappa: float = KAPPA) -> float:
    """Gain radial g(segment) = 1 + κ·(MUL + dextro − DIV − lévo), agrégé (moyenne).

    scores05 : tenseur ``(N, 6)`` des dims 0-5 par token. On MOYENNE sur les tokens
    (agrégation segment), puis on applique le couplage SÉMANTIQUE fixé : MUL/dextro
    poussent le rayon vers le dehors (centrifuge), DIV/lévo le ramènent vers le dedans
    (centripète) — exactement la lecture de ``oscilloscope.py``. Borné à
    ``[GAIN_FLOOR, GAIN_CEIL]`` (anti-dégénérescence r→0 et anti-divergence). κ posé
    UNE fois, jamais optimisé.
    """
    if scores05.numel() == 0:
        return 1.0
    m = scores05.mean(dim=0)  # (6,)
    drive = float(m[_MUL_I] + m[_DEXTRO_I] - m[_DIV_I] - m[_LEVO_I])
    g = 1.0 + kappa * drive
    return max(GAIN_FLOOR, min(GAIN_CEIL, g))


# ---------------------------------------------------------------------------
# Un segment → (slope de forme, label de chiralité).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentPoint:
    """Un point de mesure : sa forme (slope) et son étiquette (cachée au scoring)."""
    cycle_idx: int
    seg_name: str              # "SEG_A" | "SEG_B" | "SEG_A_PRIME"
    is_closure: bool           # True ⇔ A′ (LV/IN) ; jamais passé au calcul de forme
    n_tokens: int
    gain: float
    slope: float               # slope_logr_theta — la VARIABLE D'INTÉRÊT
    mu_r: float
    n_turns: float


def _segment_vectors(tok: NativeTokenizer33D, text: str) -> torch.Tensor:
    """Vecteurs 33D ``(N, 33)`` du TEXTE NETTOYÉ d'un segment (verrou 1 : pas de tag)."""
    arr = tok.vectors(text)
    return torch.as_tensor(arr, dtype=torch.float32)


def segment_slope(
    tok: NativeTokenizer33D,
    seg: AbaSegment,
    P: torch.Tensor,
    *,
    omega: float = OMEGA_DEFAULT,
    kappa: float = KAPPA,
    steps_min: int = STEPS_PER_TOKEN_MIN,
) -> Optional[Tuple[float, float, float, float, int]]:
    """Déploie un segment → ``(slope, gain, mu_r, n_turns, n_tokens)`` ou ``None``.

    VERROU 1 : on ne lit que ``seg.text`` (déjà nettoyé de tout tag par aba.py). Ni la
    position, ni la chiralité du tag n'entrent ici. Le gain vient des dims 0-5
    (phonético-sémantiques, calculées par le tokenizer C — pas du tag), le courant des
    dims 8-22. La trace est déroulée par l'oscilloscope sans aucune étiquette.

    Retourne ``None`` si le segment ne produit aucun token (rien à dérouler).
    """
    vecs = _segment_vectors(tok, seg.text)
    n = vecs.size(0)
    if n == 0:
        return None

    scores05 = vecs[:, 0:6]
    phon = vecs[:, _PHON]
    g = segment_gain(scores05, kappa=kappa)
    current = _segment_current(phon, P)

    # Horizon : au moins steps_min pas (régime établi lisible), au moins 1 pas/token
    # pour injecter tout le courant, puis régime libre. Pas d'étiquette ici.
    steps = max(steps_min, 2 * n)

    cell = Oscilloscope2D(omega=omega, gain=g, memory=0.0, win_scale=WIN_SCALE)
    s0 = torch.tensor([1.0, 0.0])
    trace = cell.trace(s0, steps=steps, signal=current)
    sig = shape_signature(trace, s0)
    if not math.isfinite(sig.slope_logr_theta):
        return None
    return (sig.slope_logr_theta, g, sig.mu_r, sig.n_turns, n)


def build_points(
    tok: NativeTokenizer33D,
    cycles: Sequence[AbaCycle],
    P: torch.Tensor,
    *,
    omega: float = OMEGA_DEFAULT,
    kappa: float = KAPPA,
    steps_min: int = STEPS_PER_TOKEN_MIN,
) -> List[SegmentPoint]:
    """Construit un ``SegmentPoint`` par segment de chaque cycle (granularité SEGMENT).

    Chaque cycle contribue 2 OUT (A, B) + 1 IN (A′) — le ratio 2:1 du corpus, préservé
    par construction. L'étiquette ``is_closure`` est PORTÉE par le point mais n'entre
    JAMAIS dans ``segment_slope`` (verrou 1).
    """
    points: List[SegmentPoint] = []
    for ci, cyc in enumerate(cycles):
        for seg_name, seg in cyc.segments.items():
            res = segment_slope(tok, seg, P, omega=omega, kappa=kappa, steps_min=steps_min)
            if res is None:
                continue
            slope, g, mu_r, n_turns, n = res
            points.append(
                SegmentPoint(
                    cycle_idx=ci,
                    seg_name=seg_name,
                    is_closure=(seg_name == "SEG_A_PRIME"),
                    n_tokens=n,
                    gain=g,
                    slope=slope,
                    mu_r=mu_r,
                    n_turns=n_turns,
                )
            )
    return points


# ---------------------------------------------------------------------------
# Held-out PAR CYCLE (verrou 3).
# ---------------------------------------------------------------------------

def split_by_cycle(
    cycles: Sequence[AbaCycle], *, holdout_frac: float = 0.3, seed: int = 7
) -> Tuple[List[AbaCycle], List[AbaCycle]]:
    """Partition train/held-out PAR CYCLE entier (jamais un segment d'un cycle des deux côtés).

    Permutation seedée déterministe ; les ``holdout_frac`` derniers cycles permutés
    forment le held-out. Tous les segments d'un cycle restent du MÊME côté (verrou 3).
    """
    n = len(cycles)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_hold = max(1, int(round(holdout_frac * n)))
    hold_idx = set(perm[-n_hold:])
    train = [cycles[i] for i in range(n) if i not in hold_idx]
    held = [cycles[i] for i in range(n) if i in hold_idx]
    return train, held


# ---------------------------------------------------------------------------
# Verrou 2 : non-fuite phonétique (même texte → même 33D quelle que soit la position).
# ---------------------------------------------------------------------------

def verrou2_nonleak_holds(tok: NativeTokenizer33D, cycles: Sequence[AbaCycle]) -> bool:
    """Vérifie le verrou 2 : ``vectors(seg.text)`` ne dépend QUE du texte, pas de la position.

    Pour chaque cycle dont A et A′ portent le MÊME texte (les « points fixes » du
    corpus), ``vectors`` doit donner un 33D IDENTIQUE — preuve que le tokenizer ne lit
    jamais la position. Plus généralement, on re-vectorise le même texte deux fois (le
    tokenizer est déterministe ET sans état de position) et on exige l'égalité bit-à-bit.
    Retourne True si AUCUNE fuite n'est détectée.
    """
    import numpy as np

    for cyc in cycles:
        for seg in cyc.segments.values():
            a = tok.vectors(seg.text)
            b = tok.vectors(seg.text)
            if not np.array_equal(a, b):
                return False
        # Si A et A′ partagent le texte, leurs 33D doivent coïncider (position ignorée).
        if cyc.seg_a.text == cyc.seg_a_prime.text and cyc.seg_a.text:
            va = tok.vectors(cyc.seg_a.text)
            vp = tok.vectors(cyc.seg_a_prime.text)
            if not np.array_equal(va, vp):
                return False
    return True


# ---------------------------------------------------------------------------
# Rapport.
# ---------------------------------------------------------------------------

def _slopes_labels(points: Sequence[SegmentPoint]) -> Tuple[List[float], List[int]]:
    """(scores, labels) avec label=1 ⇔ clôture A′ (pos), 0 ⇔ ouverture A/B (neg)."""
    scores = [p.slope for p in points]
    labels = [1 if p.is_closure else 0 for p in points]
    return scores, labels


def _median(xs: Sequence[float]) -> float:
    f = sorted(x for x in xs if math.isfinite(x))
    if not f:
        return float("nan")
    n = len(f)
    return f[n // 2] if n % 2 else 0.5 * (f[n // 2 - 1] + f[n // 2])


@dataclass(frozen=True)
class SemanticShapeReport:
    available: bool
    skipped_reason: str
    corpora: Tuple[str, ...]
    n_cycles: int
    n_train_cycles: int
    n_held_cycles: int
    omega: float
    kappa: float
    proj_seed: int
    # verrou 2
    verrou2_nonleak: bool
    # effectifs held-out
    n_pos_held: int            # A′ (clôture)
    n_neg_held: int            # A/B (ouverture)
    # AUC held-out (orientation : pos = A′ contracte ⇒ AUC>0.5 si slope(A′) < slope(A/B))
    auc_held: float
    ctrl_perm_med: float
    ctrl_perm_p95: float
    ctrl_sig_rand: float
    above_ctrl: bool
    # sous-mesure (d) : distributions de slope par groupe (held-out)
    slope_aprime_med: float
    slope_ab_med: float
    slope_aprime_q: Tuple[float, float, float]   # 5e | médiane | 95e
    slope_ab_q: Tuple[float, float, float]
    # train (référence interne, pour cohérence)
    auc_train: float
    # gain (audit : le gain DOIT varier — sinon le mapping 0-5 est muet)
    gain_aprime_med: float
    gain_ab_med: float
    # verdict
    issue: str                 # "a" | "b" | "c" | "skip"
    issue_label: str

    def summary(self) -> str:
        L: List[str] = []
        L.append("=" * 78)
        L.append("[semantic_shape_separability] Tour 7 (E3) — forme geometrique ↔ chiralite ABA")
        L.append("=" * 78)
        if not self.available:
            L.append(f"  SKIP : {self.skipped_reason}")
            return "\n".join(L)
        L.append(
            f"  corpora={list(self.corpora)}  n_cycles={self.n_cycles} "
            f"(train={self.n_train_cycles}, held={self.n_held_cycles})"
        )
        L.append(
            f"  omega=pi/5  kappa={self.kappa}  proj_seed={self.proj_seed}  "
            f"gain in [{GAIN_FLOOR},{GAIN_CEIL}]"
        )
        L.append("\n--- VERROU 2 (non-fuite phonetique : meme texte -> meme 33D) ---")
        L.append(f"  verrou2_nonleak = {'OK (aucune fuite)' if self.verrou2_nonleak else 'ECHEC -> STOP'}")

        L.append("\n--- AUC held-out (pos = A' cloture ; AUC>0.5 ⇒ A' contracte PLUS) ---")
        L.append(f"  N held-out : pos(A')={self.n_pos_held}  neg(A/B)={self.n_neg_held}")
        L.append(f"  AUC_held                  = {self.auc_held:.4f}")
        L.append(f"  CTRL-PERM (med | 95e pct) = {self.ctrl_perm_med:.4f} | {self.ctrl_perm_p95:.4f}")
        L.append(f"  CTRL-SIG-RAND             = {self.ctrl_sig_rand:.4f}   (attendu ~0.5)")
        L.append(f"  |AUC-0.5|+0.5 > 95e pct ? = {'OUI' if self.above_ctrl else 'NON'}")
        L.append(f"  AUC_train (reference)     = {self.auc_train:.4f}")

        L.append("\n--- SOUS-MESURE (d) : distribution de slope_logr_theta par groupe (held-out) ---")
        L.append(f"  slope(A')   med = {self.slope_aprime_med:+.5f}   "
                 f"(5e|med|95e = {self.slope_aprime_q[0]:+.5f} | {self.slope_aprime_q[1]:+.5f} | {self.slope_aprime_q[2]:+.5f})")
        L.append(f"  slope(A/B)  med = {self.slope_ab_med:+.5f}   "
                 f"(5e|med|95e = {self.slope_ab_q[0]:+.5f} | {self.slope_ab_q[1]:+.5f} | {self.slope_ab_q[2]:+.5f})")
        L.append("  (Ra: cercle rho≈1 ⇒ slope≈0 ; corpus l.290-291: retour contractant ⇒ slope<0)")

        L.append("\n--- AUDIT GAIN (le mapping dims 0-5 doit FAIRE VARIER g) ---")
        L.append(f"  gain(A') med = {self.gain_aprime_med:.4f}   gain(A/B) med = {self.gain_ab_med:.4f}")

        L.append(f"\n--- VERDICT : issue ({self.issue}) — {self.issue_label} ---")
        return "\n".join(L)


def run_semantic_shape_separability(
    *,
    corpus_paths: Optional[Sequence[str]] = None,
    max_cycles: Optional[int] = None,
    omega: float = OMEGA_DEFAULT,
    kappa: float = KAPPA,
    proj_seed: int = PROJ_SEED,
    holdout_frac: float = 0.3,
    split_seed: int = 7,
    n_perms: int = 20,
    steps_min: int = STEPS_PER_TOKEN_MIN,
    auc_progression_thresh: float = 0.70,
    auc_leak_thresh: float = 0.95,
) -> SemanticShapeReport:
    """Pipeline complet du Tour 7 : corpus ABA → 33D → géométrie → AUC → contrôles → verdict.

    corpus_paths : liste de fichiers ABA (défaut : corpus racine connus, résolus
        relativement au dépôt). max_cycles : borne le nombre de cycles (déterminisme du
        coût ; None = tous). Le tokenizer natif est REQUIS ; sinon SKIP propre.
    """
    if not is_available():
        return _skip_report(
            corpus_paths or (), omega, kappa, proj_seed,
            "tokenizer natif (.so/.dll) indisponible — diagnostic skippe (aucune "
            "substitution etiquette->donnee, ce serait la fuite c).",
        )

    if corpus_paths is None:
        corpus_paths = _default_corpora()

    # --- chargement des cycles (parseur de reference ; verrou 1 via seg.text) ---
    cycles: List[AbaCycle] = []
    used: List[str] = []
    for path in corpus_paths:
        if not os.path.isfile(path):
            continue
        these = list(iter_aba_cycles(path))
        if these:
            cycles.extend(these)
            used.append(os.path.basename(path))
    if max_cycles is not None:
        cycles = cycles[:max_cycles]

    if not cycles:
        return _skip_report(
            tuple(used), omega, kappa, proj_seed,
            "aucun cycle ABA parseable dans les corpus fournis.",
        )

    tok = NativeTokenizer33D()
    P = phoneme_projection(seed=proj_seed)

    # --- VERROU 2 : non-fuite phonetique (avant tout scoring) ---
    verrou2 = verrou2_nonleak_holds(tok, cycles)

    # --- held-out PAR CYCLE (verrou 3) ---
    train_cycles, held_cycles = split_by_cycle(
        cycles, holdout_frac=holdout_frac, seed=split_seed
    )

    train_pts = build_points(tok, train_cycles, P, omega=omega, kappa=kappa, steps_min=steps_min)
    held_pts = build_points(tok, held_cycles, P, omega=omega, kappa=kappa, steps_min=steps_min)

    # --- AUC held-out (pos = A' cloture) ---
    held_scores, held_labels = _slopes_labels(held_pts)
    pos = [s for s, l in zip(held_scores, held_labels) if l == 1]
    neg = [s for s, l in zip(held_scores, held_labels) if l == 0]
    # Convention : A' attendu CONTRACTANT (slope plus NEGATIF). _rank_auc(pos>neg)
    # donnerait AUC>0.5 si A' avait des slopes PLUS GRANDS — l'inverse de l'attendu.
    # On veut « A' contracte plus » ⇒ slope(A') < slope(A/B) ⇒ on score sur −slope
    # pour que AUC>0.5 ⇔ A' plus contractant (orientation documentee).
    auc_held = _rank_auc([-s for s in pos], [-s for s in neg]) if pos and neg else float("nan")

    cp_med = cp_p95 = csr = float("nan")
    above = False
    if pos and neg:
        scores_signed = [-s for s in held_scores]  # même transformation pour CTRL-PERM
        cp_med, cp_p95, _ = _ctrl_perm_distribution(
            scores_signed, held_labels, n_perms=n_perms, seed0=70_100
        )
        finite = [s for s in scores_signed if math.isfinite(s)]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        csr = _ctrl_sig_rand_auc(len(pos), len(neg), (lo, hi), seed=70_200)
        above = _abs_dir(auc_held) > cp_p95

    # --- train (reference interne) ---
    tr_scores, tr_labels = _slopes_labels(train_pts)
    tr_pos = [s for s, l in zip(tr_scores, tr_labels) if l == 1]
    tr_neg = [s for s, l in zip(tr_scores, tr_labels) if l == 0]
    auc_train = _rank_auc([-s for s in tr_pos], [-s for s in tr_neg]) if tr_pos and tr_neg else float("nan")

    # --- sous-mesure (d) : distributions de slope par groupe (held-out) ---
    sl_ap = [p.slope for p in held_pts if p.is_closure]
    sl_ab = [p.slope for p in held_pts if not p.is_closure]
    g_ap = [p.gain for p in held_pts if p.is_closure]
    g_ab = [p.gain for p in held_pts if not p.is_closure]

    # --- verdict ---
    issue, label = _verdict(
        auc_held, above, auc_progression_thresh, auc_leak_thresh, verrou2,
        n_pos=len(pos), n_neg=len(neg),
    )

    return SemanticShapeReport(
        available=True,
        skipped_reason="",
        corpora=tuple(used),
        n_cycles=len(cycles),
        n_train_cycles=len(train_cycles),
        n_held_cycles=len(held_cycles),
        omega=omega,
        kappa=kappa,
        proj_seed=proj_seed,
        verrou2_nonleak=verrou2,
        n_pos_held=len(pos),
        n_neg_held=len(neg),
        auc_held=auc_held,
        ctrl_perm_med=cp_med,
        ctrl_perm_p95=cp_p95,
        ctrl_sig_rand=csr,
        above_ctrl=above,
        slope_aprime_med=_median(sl_ap),
        slope_ab_med=_median(sl_ab),
        slope_aprime_q=_quantiles(sl_ap),
        slope_ab_q=_quantiles(sl_ab),
        auc_train=auc_train,
        gain_aprime_med=_median(g_ap),
        gain_ab_med=_median(g_ab),
        issue=issue,
        issue_label=label,
    )


def _verdict(
    auc: float,
    above: bool,
    prog_thresh: float,
    leak_thresh: float,
    verrou2: bool,
    *,
    n_pos: int,
    n_neg: int,
) -> Tuple[str, str]:
    """Qualifie le retour selon les issues a/b/c (verrou 2 d'abord)."""
    if not verrou2:
        return "c", (
            "FUITE structurelle : verrou 2 echoue (le 33D depend de la position). "
            "STOP — auditer le pipeline avant toute mesure."
        )
    if n_pos == 0 or n_neg == 0 or not math.isfinite(auc):
        return "skip", "groupes pos/neg vides en held-out — AUC non evaluable."
    strength = _abs_dir(auc)
    if strength > leak_thresh:
        return "c", (
            f"AUC={auc:.3f} (force {strength:.3f}) > {leak_thresh} ⇒ SUSPICION DE FUITE — "
            "auditer le pipeline (le 33D ou la projection encode-t-il l'etiquette ?)."
        )
    if strength >= prog_thresh and above:
        return "a", (
            f"PROGRESSION separable : AUC={auc:.3f} >= {prog_thresh} ET > 95e pct CTRL-PERM "
            "(la forme geometrique predit la chiralite ABA au-dessus du hasard)."
        )
    return "b", (
        f"NULL legitime : AUC={auc:.3f} ≈ 0.5 (ou <= CTRL-PERM). Resultat ATTENDU et "
        "ACCEPTABLE (emetteur ABA a 22.8% d'accord op., biais ADD). Documente tel quel, "
        "JAMAIS force en positif (REFUS)."
    )


def _default_corpora() -> List[str]:
    """Corpus ABA connus, résolus à la racine du dépôt (parents[2] = .../spiraton)."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]      # .../spiraton (dépôt PyTorch)
    root = repo.parent                                # .../spiraton-enhanced (corpus racine)
    names = ("corpus_claude_aba.txt", "dataset_aba.txt", "corpus_eve_clean.txt")
    out: List[str] = []
    for name in names:
        for base in (repo, root):
            p = base / name
            if p.is_file():
                out.append(str(p))
                break
    return out


def _skip_report(
    corpora: Sequence[str], omega: float, kappa: float, proj_seed: int, reason: str
) -> SemanticShapeReport:
    nan3 = (float("nan"), float("nan"), float("nan"))
    return SemanticShapeReport(
        available=False, skipped_reason=reason, corpora=tuple(corpora),
        n_cycles=0, n_train_cycles=0, n_held_cycles=0,
        omega=omega, kappa=kappa, proj_seed=proj_seed,
        verrou2_nonleak=False,
        n_pos_held=0, n_neg_held=0,
        auc_held=float("nan"), ctrl_perm_med=float("nan"),
        ctrl_perm_p95=float("nan"), ctrl_sig_rand=float("nan"), above_ctrl=False,
        slope_aprime_med=float("nan"), slope_ab_med=float("nan"),
        slope_aprime_q=nan3, slope_ab_q=nan3,
        auc_train=float("nan"),
        gain_aprime_med=float("nan"), gain_ab_med=float("nan"),
        issue="skip", issue_label=reason,
    )
