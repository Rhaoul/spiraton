from __future__ import annotations

"""Chiralité séquentielle ↔ pente du flux phonémique ordonné (Tour 9, re-tentative E3).

GESTE OPÉRATOIRE (émission du linguiste H9-P2, incarnée TELLE QUELLE). On demande si la
CHIRALITÉ ABA d'un segment (``<DX><OUT>`` ouverture A/B vs ``<LV><IN>`` clôture A′) se lit
dans le SIGNE/PENTE DOMINANT du gradient signé de sa TRAJECTOIRE DE FLUX ORDONNÉE — la
séquence ``sig_flux`` déjà présente dans le 33D, dans l'ordre des phonèmes. Le mapping est
FIXÉ A PRIORI (jamais fitté contre la cible) et ancré dans le corpus eve (l.486-487) :

    flux qui MONTE le long de la séquence  → centrifuge / émission / OUVERTURE (dextro/D),
    flux qui DESCEND le long de la séquence → centripète / intégration / FERMETURE (lévo/L).

UNE SEULE FONCTIONNELLE, DÉCLARÉE D'AVANCE (interdiction de balayer {signe / pente /
corrélation index-flux} et de retenir la meilleure — piège ``analyser_signature_mot``) :

    g_bar(segment) = moyenne_i ( flux[i+1] − flux[i] )   sur la séquence ordonnée du segment.

C'est la PENTE MOYENNE signée du flux (par télescopage, g_bar = (flux[-1] − flux[0])/(L−1)).
Score de chiralité = signe/amplitude de g_bar. CONVENTION D'ORIENTATION (documentée, non
pliée vers max(auc, 1−auc)) : pos = A′ (clôture), attendu DESCENDANT (g_bar < 0). On score
donc sur ``−g_bar`` pour que AUC>0.5 ⇔ A′ descend PLUS que A/B (cohérent avec le mapping
fixé). La direction est une information (REFUS).

--- LE CŒUR DU TOUR : L'ABLATION DÉCISIVE (permutation de l'ordre) ---------------------------
Le gain de chiralité DOIT s'effondrer quand on PERMUTE l'ordre de la séquence de flux PUIS
qu'on recalcule le gradient et la signature. Le gradient d'une séquence permutée est
génuinement différent :
  * si l'AUC s'effondre sous permutation → la signature lit VRAIMENT l'ordre (issue a, PROGRESSION) ;
  * si l'AUC SURVIT à la permutation → la « signature de gradient » capture en fait une
    statistique d'ENSEMBLE ordre-indépendante (étendue/variance du flux) = somme déguisée =
    DISSIPATION (issue c, à REJETER, pas à célébrer).
On quantifie ``gain_sequentiel = AUC(ordre réel) − AUC(ordre permuté)`` et on teste sa
significativité (> 95e pct de la distribution CTRL-PERM-ORDER : permutations indépendantes
de l'ordre, mêmes séquences, étiquettes fixes).

GRANULARITÉ = SEGMENT. La trajectoire de flux ordonnée d'un segment est la CONCATÉNATION, en
ordre de lecture, des valeurs ``sig_flux`` RÉELLES (zéro-padding du C retiré) de chaque token.
``sig_flux`` (vector33d.SIG_FLUX = dims 13-17, VÉRIFIÉ dans le header C et mot_v4.c) tient le
flux des 5 premiers phonèmes d'un mot, zéro-padé au-delà de ``nb_phonemes`` ; on tronque
chaque token à son préfixe réel (longueur phonémique = dim 31 × 10) avant concaténation.

TROIS VERROUS ANTI-FUITE (capital) :
  1. VERROU D'ENTRÉE : on ne lit que ``vectors(seg.text)`` où ``seg.text`` est nettoyé de tout
     tag par ``_clean_text`` (aba.py). L'étiquette ne sert QU'à former pos/neg APRÈS scoring.
  2. VERROU PHONÉTIQUE : la séquence de flux vient des phonèmes calculés par le tokenizer C,
     PAS du tag. ``vectors(texte)`` est identique que le texte soit présenté en A ou A′.
  3. CTRL-PERM (étiquettes) + held-out PAR CYCLE + CTRL-SIG-RAND (signatures aléatoires →
     AUC≈0.5) + CTRL-PERM-ORDER (permutation de l'ordre intra-séquence — l'ablation).

SKIP PROPRE. Si le tokenizer natif est indisponible, le diagnostic SKIP
(``TokenizerUnavailable``). On NE substitue JAMAIS l'étiquette à la donnée.

REFUS — discipline. UNE fonctionnelle posée AVANT de voir la cible, aucun fit, aucun balayage.
Le mapping (montée→ouverture) est ancré corpus, pas optimisé. L'AUC ne compte que si elle
dépasse le 95e pct de CTRL-PERM ET que le gain s'effondre sous permutation. Le null (issue b)
est attendu et légitime ; un gain qui survit à la permutation est REJETÉ (issue c).
"""

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..data import vector33d as v33
from ..data.aba import AbaCycle, AbaSegment, iter_aba_cycles
from ..data.tokenizer_bridge import NativeTokenizer33D, is_available
from .spectral_separability import (
    _abs_dir,
    _ctrl_perm_distribution,
    _ctrl_sig_rand_auc,
    _quantiles,
    _rank_auc,
)

# Slice EXACT de la signature de flux dans le 33D — VÉRIFIÉ contre la source de vérité C :
#   Tokenizer/csrc/tokenizer.c : `for(i<5) t->vector33d[13+i] = m->sig_flux[i];`  (dims 13-17)
#   Tokenizer/csrc/mot_v4.c    : `mot->sig_flux[p_idx] = p->flux;` (ordre phonémique, zéro-padé)
SIG_FLUX = slice(13, 18)          # 5 valeurs : flux des 5 premiers phonèmes du token
_LEN_PHON_DIM = 31                # dim 31 = nb_phonemes / 10 (longueur phonémique normalisée)

# Significativité : combien de percentile la distribution CTRL-PERM doit-elle franchir.
CTRL_PCTL = 0.95


# ---------------------------------------------------------------------------
# Construction de la trajectoire de flux ordonnée d'un segment.
# ---------------------------------------------------------------------------

def _token_flux_prefix(vec_row: torch.Tensor) -> List[float]:
    """Préfixe RÉEL de flux d'un token : ``sig_flux[:min(5, nb_phonemes)]`` (zéro-padding retiré).

    Le C remplit ``sig_flux[5]`` avec le flux des 5 premiers phonèmes puis zéro-pade. Les zéros
    de padding ne sont PAS des valeurs de flux : les inclure corromprait le gradient (une chute
    artificielle vers 0). On lit la longueur phonémique réelle (dim 31 = nb_phonemes/10) pour
    tronquer au bon préfixe. Borné à [0, 5] (la signature ne porte que les 5 premiers phonèmes).
    """
    flux = vec_row[SIG_FLUX].tolist()
    n_phon = int(round(float(vec_row[_LEN_PHON_DIM]) * 10.0))
    k = max(0, min(5, n_phon))
    return [float(x) for x in flux[:k]]


def segment_flux_sequence(tok: NativeTokenizer33D, text: str) -> List[float]:
    """Trajectoire de flux ORDONNÉE d'un segment : concat. en ordre de lecture des préfixes réels.

    VERROU 1 : on ne lit que ``text`` (déjà nettoyé de tout tag par aba.py). Chaque token
    contribue son préfixe de flux réel (``_token_flux_prefix``) ; on concatène dans l'ordre des
    tokens (ordre de lecture). C'est la séquence sur laquelle on calcule le gradient signé.
    """
    arr = tok.vectors(text)
    vecs = torch.as_tensor(arr, dtype=torch.float32)
    seq: List[float] = []
    for r in range(vecs.size(0)):
        seq.extend(_token_flux_prefix(vecs[r]))
    return seq


# ---------------------------------------------------------------------------
# LA fonctionnelle (UNE seule, fixée a priori) : pente moyenne signée du flux.
# ---------------------------------------------------------------------------

def mean_signed_gradient(seq: Sequence[float]) -> float:
    """g_bar = moyenne_i ( seq[i+1] − seq[i] ) = (seq[-1] − seq[0]) / (L−1).

    La pente moyenne signée de la trajectoire de flux ordonnée. Mapping FIXÉ : g_bar>0 (flux
    monte) → ouverture/dextro ; g_bar<0 (flux descend) → fermeture/lévo. Séquence de longueur
    < 2 : pas de pente définie → ``nan`` (segment écarté du scoring, jamais imputé).
    """
    n = len(seq)
    if n < 2:
        return float("nan")
    return (float(seq[-1]) - float(seq[0])) / (n - 1)


def permuted_gradient(seq: Sequence[float], *, seed: int) -> float:
    """g_bar de la séquence PERMUTÉE (ablation de l'ordre) — recalculé sur le nouvel ordre.

    Permutation seedée déterministe des valeurs de flux du segment, PUIS recalcul de la pente
    moyenne signée. C'est l'ablation décisive : si g_bar permuté porte autant de signal que
    g_bar réel, la « signature de gradient » est en fait une statistique d'ensemble
    ordre-indépendante (somme déguisée). Une permutation ne change PAS l'ensemble {valeurs},
    mais change g_bar = (seq[π(-1)] − seq[π(0)])/(L−1) : un gradient génuinement différent.
    """
    n = len(seq)
    if n < 2:
        return float("nan")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    permuted = [seq[i] for i in perm]
    return (float(permuted[-1]) - float(permuted[0])) / (n - 1)


# ---------------------------------------------------------------------------
# Un segment → (g_bar, longueur de séquence, label de chiralité).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentFluxPoint:
    """Un point de mesure : sa pente de flux et son étiquette (cachée au scoring)."""
    cycle_idx: int
    seg_name: str              # "SEG_A" | "SEG_B" | "SEG_A_PRIME"
    is_closure: bool           # True ⇔ A′ (LV/IN) ; jamais passé au calcul de g_bar
    seq_len: int               # longueur de la trajectoire de flux ordonnée
    seq: Tuple[float, ...]     # la trajectoire elle-même (pour l'ablation de permutation)
    g_bar: float               # pente moyenne signée — la VARIABLE D'INTÉRÊT


def build_flux_points(
    tok: NativeTokenizer33D, cycles: Sequence[AbaCycle]
) -> List[SegmentFluxPoint]:
    """Un ``SegmentFluxPoint`` par segment de chaque cycle (granularité SEGMENT, ratio 2:1).

    L'étiquette ``is_closure`` est PORTÉE par le point mais n'entre JAMAIS dans le calcul de
    ``g_bar`` (verrou 1). Les segments de séquence < 2 (g_bar non défini) sont écartés.
    """
    points: List[SegmentFluxPoint] = []
    for ci, cyc in enumerate(cycles):
        for seg_name, seg in cyc.segments.items():
            seq = segment_flux_sequence(tok, seg.text)
            g_bar = mean_signed_gradient(seq)
            if not math.isfinite(g_bar):
                continue
            points.append(
                SegmentFluxPoint(
                    cycle_idx=ci,
                    seg_name=seg_name,
                    is_closure=(seg_name == "SEG_A_PRIME"),
                    seq_len=len(seq),
                    seq=tuple(seq),
                    g_bar=g_bar,
                )
            )
    return points


# ---------------------------------------------------------------------------
# Held-out PAR CYCLE (verrou 3) — repris à l'identique du Tour 7.
# ---------------------------------------------------------------------------

def split_by_cycle(
    cycles: Sequence[AbaCycle], *, holdout_frac: float = 0.3, seed: int = 7
) -> Tuple[List[AbaCycle], List[AbaCycle]]:
    """Partition train/held-out PAR CYCLE entier (jamais un segment d'un cycle des deux côtés)."""
    n = len(cycles)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_hold = max(1, int(round(holdout_frac * n)))
    hold_idx = set(perm[-n_hold:])
    train = [cycles[i] for i in range(n) if i not in hold_idx]
    held = [cycles[i] for i in range(n) if i in hold_idx]
    return train, held


# ---------------------------------------------------------------------------
# AUC orientée (pos = A′ clôture, attendu DESCENDANT ⇒ on score sur −g_bar).
# ---------------------------------------------------------------------------

def _auc_closure_descends(points: Sequence[SegmentFluxPoint]) -> float:
    """AUC : pos = A′ (clôture), score = −g_bar (A′ attendu descendant ⇒ −g_bar plus grand)."""
    pos = [-p.g_bar for p in points if p.is_closure]
    neg = [-p.g_bar for p in points if not p.is_closure]
    if not pos or not neg:
        return float("nan")
    return _rank_auc(pos, neg)


def _auc_from_gbars(
    gbars: Sequence[float], labels: Sequence[int]
) -> float:
    """AUC sur des g_bar arbitraires (sert pour l'ablation permutée), même convention −g_bar."""
    pos = [-g for g, l in zip(gbars, labels) if l == 1]
    neg = [-g for g, l in zip(gbars, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    return _rank_auc(pos, neg)


# ---------------------------------------------------------------------------
# L'ABLATION : distribution de l'AUC sous permutation de l'ORDRE intra-séquence.
# ---------------------------------------------------------------------------

def _ctrl_perm_order_distribution(
    points: Sequence[SegmentFluxPoint], *, n_perms: int, seed0: int
) -> Tuple[float, float, float, List[float]]:
    """Distribution de l'AUC quand on PERMUTE l'ordre de chaque séquence puis recalcule g_bar.

    Pour chaque permutation globale ``p`` : on permute (seed distinct par segment) l'ordre des
    valeurs de flux de CHAQUE segment, on recalcule g_bar, puis l'AUC orientée. Les étiquettes
    restent FIXES (on n'ablate QUE l'ordre, pas le lien segment↔label). Retourne
    ``(médiane, 5e pct, 95e pct, distribution)`` des AUC permutées. Si le signal lit l'ordre,
    cette distribution doit retomber vers 0.5 (la baseline d'ordre Δ=0 du Tour 8).
    """
    labels = [1 if p.is_closure else 0 for p in points]
    dist: List[float] = []
    for p in range(n_perms):
        gbars: List[float] = []
        for si, pt in enumerate(points):
            gbars.append(permuted_gradient(pt.seq, seed=seed0 + p * 100_003 + si))
        finite = [(g, l) for g, l in zip(gbars, labels) if math.isfinite(g)]
        if not finite:
            continue
        gg = [g for g, _ in finite]
        ll = [l for _, l in finite]
        auc = _auc_from_gbars(gg, ll)
        if math.isfinite(auc):
            dist.append(auc)
    dist.sort()
    if not dist:
        return float("nan"), float("nan"), float("nan"), dist
    med = dist[len(dist) // 2]
    i5 = min(len(dist) - 1, max(0, int(round(0.05 * (len(dist) - 1)))))
    i95 = min(len(dist) - 1, int(math.ceil(0.95 * len(dist))) - 1)
    return med, dist[i5], dist[i95], dist


def _order_perm_force_p95(
    points: Sequence[SegmentFluxPoint], *, n_perms: int, seed0: int
) -> Tuple[float, float, List[float]]:
    """95e pct de la FORCE discriminante ``|auc−0.5|+0.5`` sous permutation de l'ordre (H0-ordre).

    C'est l'analogue de CTRL-PERM mais pour l'ORDRE (et non les étiquettes) : on permute l'ordre
    intra-séquence de chaque segment, on recalcule g_bar puis l'AUC, et on prend la FORCE (la
    direction est arbitraire sous H0). Un signal qui lit VRAIMENT l'ordre doit avoir une force
    réelle dépassant le 95e pct de cette distribution. Retourne ``(médiane, 95e pct, distribution)``
    de la force. Plus robuste que la distribution des différences AUC_real−AUC_p (forte variance).
    """
    labels = [1 if p.is_closure else 0 for p in points]
    forces: List[float] = []
    for p in range(n_perms):
        gbars: List[float] = []
        for si, pt in enumerate(points):
            gbars.append(permuted_gradient(pt.seq, seed=seed0 + p * 100_003 + si))
        finite = [(g, l) for g, l in zip(gbars, labels) if math.isfinite(g)]
        if not finite:
            continue
        gg = [g for g, _ in finite]
        ll = [l for _, l in finite]
        auc_p = _auc_from_gbars(gg, ll)
        if math.isfinite(auc_p):
            forces.append(_abs_dir(auc_p))
    forces.sort()
    if not forces:
        return float("nan"), float("nan"), forces
    med = forces[len(forces) // 2]
    i95 = min(len(forces) - 1, int(math.ceil(0.95 * len(forces))) - 1)
    return med, forces[i95], forces


# ---------------------------------------------------------------------------
# Rapport.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequentialChiralityReport:
    available: bool
    skipped_reason: str
    corpora: Tuple[str, ...]
    n_cycles: int
    n_train_cycles: int
    n_held_cycles: int
    # baseline de majorité (ratio de chiralité, mesuré)
    n_open: int                 # segments A/B (ouverture)
    n_closure: int              # segments A′ (clôture)
    majority_baseline: float    # max(open, closure) / total  — accord d'un classif. constant
    # effectifs held-out
    n_pos_held: int             # A′ (clôture)
    n_neg_held: int             # A/B (ouverture)
    mean_seq_len: float
    # AUC held-out (ordre réel)
    auc_held: float
    auc_train: float
    ctrl_perm_med: float        # CTRL-PERM étiquettes (force discriminante)
    ctrl_perm_p95: float
    ctrl_sig_rand: float
    above_ctrl_perm: bool
    # L'ABLATION : AUC sous permutation de l'ordre + gain séquentiel
    auc_order_perm_med: float
    auc_order_perm_p95: float
    gain_sequentiel: float      # |auc_held−0.5| − |méd(auc ordre permuté)−0.5| (effondrement de force)
    order_perm_force_p95: float # 95e pct de la FORCE discriminante sous H0-ordre
    gain_significant: bool      # force(auc_held) > 95e pct force(ordre permuté)
    gain_collapses: bool        # |auc_order_perm − 0.5| < |auc_held − 0.5| (le gain s'effondre)
    # accord de chiralité (mapping fixé, seuil g_bar = 0)
    chirality_accuracy: float   # held-out : prédit clôture ⇔ g_bar < 0
    # distribution de g_bar par groupe (held-out)
    gbar_aprime_q: Tuple[float, float, float]
    gbar_ab_q: Tuple[float, float, float]
    # verdict
    issue: str                  # "a" | "b" | "c" | "skip"
    issue_label: str

    def summary(self) -> str:
        L: List[str] = []
        L.append("=" * 80)
        L.append("[sequential_chirality] Tour 9 (E3 re-tentative) — pente du flux ordonne ↔ chiralite ABA")
        L.append("=" * 80)
        if not self.available:
            L.append(f"  SKIP : {self.skipped_reason}")
            return "\n".join(L)
        L.append(
            f"  corpora={list(self.corpora)}  n_cycles={self.n_cycles} "
            f"(train={self.n_train_cycles}, held={self.n_held_cycles})"
        )
        L.append(f"  fonctionnelle UNIQUE = pente moyenne signee g_bar ; mapping FIXE : montee→ouverture")
        L.append(f"  sig_flux = dims 13-17 (verifie header C + mot_v4.c) ; granularite = SEGMENT")

        L.append("\n--- BASELINE DE MAJORITE (ratio de chiralite, mesure) ---")
        L.append(f"  segments ouverture(A/B)={self.n_open}  cloture(A')={self.n_closure}")
        L.append(f"  majorite (classif. constant) = {self.majority_baseline:.4f}")

        L.append("\n--- AUC held-out (ordre REEL ; pos = A' cloture ; AUC>0.5 ⇒ A' descend PLUS) ---")
        L.append(f"  N held-out : pos(A')={self.n_pos_held}  neg(A/B)={self.n_neg_held}  mean_seq_len={self.mean_seq_len:.2f}")
        L.append(f"  AUC_held                       = {self.auc_held:.4f}")
        L.append(f"  AUC_train (reference)          = {self.auc_train:.4f}")
        L.append(f"  CTRL-PERM (med | 95e pct)      = {self.ctrl_perm_med:.4f} | {self.ctrl_perm_p95:.4f}")
        L.append(f"  CTRL-SIG-RAND                  = {self.ctrl_sig_rand:.4f}   (attendu ~0.5)")
        L.append(f"  |AUC-0.5|+0.5 > 95e pct CTRL-PERM ? = {'OUI' if self.above_ctrl_perm else 'NON'}")
        L.append(f"  accord de chiralite (g_bar<0 ⇒ cloture) = {self.chirality_accuracy:.4f}")

        L.append("\n--- L'ABLATION DECISIVE : permutation de l'ORDRE intra-sequence ---")
        L.append(f"  AUC(ordre permute) (med | 95e pct) = {self.auc_order_perm_med:.4f} | {self.auc_order_perm_p95:.4f}")
        L.append(f"  gain_sequentiel = force(reel) - force(permute med) = {self.gain_sequentiel:+.4f}")
        L.append(f"  95e pct de FORCE sous H0-ordre              = {self.order_perm_force_p95:.4f}")
        L.append(f"  force reelle > 95e pct (gain significatif) ? = {'OUI' if self.gain_significant else 'NON'}")
        collapse_txt = "OUI (lit l'ordre)" if self.gain_collapses else "NON (somme deguisee ⇒ DISSIPATION)"
        L.append(f"  le gain s'EFFONDRE sous permutation ?       = {collapse_txt}")

        L.append("\n--- distribution de g_bar par groupe (held-out ; 5e | med | 95e) ---")
        L.append(f"  g_bar(A')  = {self.gbar_aprime_q[0]:+.5f} | {self.gbar_aprime_q[1]:+.5f} | {self.gbar_aprime_q[2]:+.5f}")
        L.append(f"  g_bar(A/B) = {self.gbar_ab_q[0]:+.5f} | {self.gbar_ab_q[1]:+.5f} | {self.gbar_ab_q[2]:+.5f}")
        L.append("  (mapping : A' attendu g_bar<0 / descendant ; A/B attendu g_bar>0 / montant)")

        L.append(f"\n--- VERDICT : issue ({self.issue}) — {self.issue_label} ---")
        return "\n".join(L)


def run_sequential_chirality(
    *,
    corpus_paths: Optional[Sequence[str]] = None,
    max_cycles: Optional[int] = None,
    holdout_frac: float = 0.3,
    split_seed: int = 7,
    n_perms: int = 25,
    n_order_perms: int = 25,
    auc_progression_thresh: float = 0.70,
    auc_leak_thresh: float = 0.95,
) -> SequentialChiralityReport:
    """Pipeline Tour 9 : corpus ABA → flux ordonne → g_bar → AUC → ablation-permutation → verdict.

    Le tokenizer natif est REQUIS ; sinon SKIP propre (aucune substitution etiquette→donnee).
    ``n_order_perms`` controle l'ablation (permutation de l'ordre intra-sequence).
    """
    if not is_available():
        return _skip_report(
            corpus_paths or (),
            "tokenizer natif (.so/.dll) indisponible — diagnostic skippe (aucune "
            "substitution etiquette->donnee, ce serait la fuite c).",
        )

    if corpus_paths is None:
        corpus_paths = _default_corpora()

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
        return _skip_report(tuple(used), "aucun cycle ABA parseable dans les corpus fournis.")

    tok = NativeTokenizer33D()

    # --- baseline de majorite (ratio de chiralite, mesure d'abord) ---
    n_open = 2 * len(cycles)        # A + B par cycle (ouverture)
    n_closure = len(cycles)         # A' par cycle (cloture)
    total = n_open + n_closure
    majority = max(n_open, n_closure) / total if total else float("nan")

    # --- held-out PAR CYCLE (verrou 3) ---
    train_cycles, held_cycles = split_by_cycle(cycles, holdout_frac=holdout_frac, seed=split_seed)
    train_pts = build_flux_points(tok, train_cycles)
    held_pts = build_flux_points(tok, held_cycles)

    # --- AUC held-out (ordre reel) ---
    auc_held = _auc_closure_descends(held_pts)
    auc_train = _auc_closure_descends(train_pts)

    n_pos = sum(1 for p in held_pts if p.is_closure)
    n_neg = sum(1 for p in held_pts if not p.is_closure)
    mean_seq_len = (sum(p.seq_len for p in held_pts) / len(held_pts)) if held_pts else float("nan")

    # --- CTRL-PERM (etiquettes) + CTRL-SIG-RAND ---
    cp_med = cp_p95 = csr = float("nan")
    above = False
    if n_pos and n_neg:
        scores_signed = [-p.g_bar for p in held_pts]
        labels = [1 if p.is_closure else 0 for p in held_pts]
        cp_med, cp_p95, _ = _ctrl_perm_distribution(scores_signed, labels, n_perms=n_perms, seed0=90_100)
        finite = [s for s in scores_signed if math.isfinite(s)]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        csr = _ctrl_sig_rand_auc(n_pos, n_neg, (lo, hi), seed=90_200)
        above = _abs_dir(auc_held) > cp_p95

    # --- L'ABLATION : permutation de l'ORDRE intra-sequence (held-out) ---
    op_med = op_p95 = float("nan")
    gain_seq = force_p95 = float("nan")
    gain_sig = gain_collapses = False
    if n_pos and n_neg and math.isfinite(auc_held):
        op_med, _op5, op_p95, _ = _ctrl_perm_order_distribution(
            held_pts, n_perms=n_order_perms, seed0=90_300
        )
        # gain = effondrement de FORCE entre ordre reel et ordre permute (median).
        gain_seq = (_abs_dir(auc_held) - _abs_dir(op_med)) if math.isfinite(op_med) else float("nan")
        _force_med, force_p95, _ = _order_perm_force_p95(
            held_pts, n_perms=n_order_perms, seed0=90_400
        )
        # significatif si la FORCE reelle depasse le 95e pct de la force sous H0-ordre.
        gain_sig = (
            math.isfinite(force_p95) and _abs_dir(auc_held) > force_p95
        )
        if math.isfinite(op_med):
            gain_collapses = _abs_dir(op_med) < _abs_dir(auc_held)

    # --- accord de chiralite (mapping fixe : g_bar<0 ⇒ predit cloture) ---
    chir_acc = float("nan")
    if held_pts:
        correct = 0
        for p in held_pts:
            pred_closure = p.g_bar < 0.0
            if pred_closure == p.is_closure:
                correct += 1
        chir_acc = correct / len(held_pts)

    gbar_ap = [p.g_bar for p in held_pts if p.is_closure]
    gbar_ab = [p.g_bar for p in held_pts if not p.is_closure]

    issue, label = _verdict(
        auc_held=auc_held,
        above_ctrl=above,
        gain_significant=gain_sig,
        gain_collapses=gain_collapses,
        prog_thresh=auc_progression_thresh,
        leak_thresh=auc_leak_thresh,
        n_pos=n_pos,
        n_neg=n_neg,
    )

    return SequentialChiralityReport(
        available=True,
        skipped_reason="",
        corpora=tuple(used),
        n_cycles=len(cycles),
        n_train_cycles=len(train_cycles),
        n_held_cycles=len(held_cycles),
        n_open=n_open,
        n_closure=n_closure,
        majority_baseline=majority,
        n_pos_held=n_pos,
        n_neg_held=n_neg,
        mean_seq_len=mean_seq_len,
        auc_held=auc_held,
        auc_train=auc_train,
        ctrl_perm_med=cp_med,
        ctrl_perm_p95=cp_p95,
        ctrl_sig_rand=csr,
        above_ctrl_perm=above,
        auc_order_perm_med=op_med,
        auc_order_perm_p95=op_p95,
        gain_sequentiel=gain_seq,
        order_perm_force_p95=force_p95,
        gain_significant=gain_sig,
        gain_collapses=gain_collapses,
        chirality_accuracy=chir_acc,
        gbar_aprime_q=_quantiles(gbar_ap),
        gbar_ab_q=_quantiles(gbar_ab),
        issue=issue,
        issue_label=label,
    )


def _verdict(
    *,
    auc_held: float,
    above_ctrl: bool,
    gain_significant: bool,
    gain_collapses: bool,
    prog_thresh: float,
    leak_thresh: float,
    n_pos: int,
    n_neg: int,
) -> Tuple[str, str]:
    """Qualifie le retour selon les issues a/b/c.

    (a) PROGRESSION : AUC ≥ prog_thresh ET > 95e pct CTRL-PERM ET le gain s'effondre sous
        permutation (gain significatif) → le signal lit VRAIMENT l'ordre.
    (b) NULL legitime : AUC ≈ 0.5-0.6 ou ne survit pas a CTRL-PERM → l'ordre intra-mot ne
        porte pas la chiralite ABA. Issue la PLUS PROBABLE (cf. Tour 7).
    (c) DISSIPATION : le gain SURVIT a la permutation (somme deguisee) — a REJETER ; ou AUC
        suspecte de fuite (> leak_thresh).
    """
    if n_pos == 0 or n_neg == 0 or not math.isfinite(auc_held):
        return "skip", "groupes pos/neg vides en held-out — AUC non evaluable."
    strength = _abs_dir(auc_held)
    if strength > leak_thresh:
        return "c", (
            f"AUC={auc_held:.3f} (force {strength:.3f}) > {leak_thresh} ⇒ SUSPICION DE FUITE — "
            "auditer le pipeline avant toute conclusion."
        )
    if strength >= prog_thresh and above_ctrl:
        if gain_significant and gain_collapses:
            return "a", (
                f"PROGRESSION : AUC={auc_held:.3f} >= {prog_thresh}, > 95e pct CTRL-PERM, ET le "
                "gain s'effondre sous permutation de l'ordre (gain significatif). Le signal lit "
                "VRAIMENT l'ordre du flux → materialisation ABI du gradient justifiee."
            )
        return "c", (
            f"DISSIPATION : AUC={auc_held:.3f} forte MAIS le gain SURVIT a la permutation de "
            "l'ordre (somme deguisee — statistique d'ensemble ordre-independante). A REJETER, "
            "pas a celebrer (REFUS)."
        )
    return "b", (
        f"NULL legitime : AUC={auc_held:.3f} ≈ 0.5 (ou <= CTRL-PERM). L'ordre intra-mot du flux "
        "ne porte pas la chiralite ABA au-dessus du hasard. Resultat ATTENDU et ACCEPTABLE "
        "(cf. Tour 7), documente tel quel, JAMAIS force en positif (REFUS)."
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


def _skip_report(corpora: Sequence[str], reason: str) -> SequentialChiralityReport:
    nan3 = (float("nan"), float("nan"), float("nan"))
    return SequentialChiralityReport(
        available=False, skipped_reason=reason, corpora=tuple(corpora),
        n_cycles=0, n_train_cycles=0, n_held_cycles=0,
        n_open=0, n_closure=0, majority_baseline=float("nan"),
        n_pos_held=0, n_neg_held=0, mean_seq_len=float("nan"),
        auc_held=float("nan"), auc_train=float("nan"),
        ctrl_perm_med=float("nan"), ctrl_perm_p95=float("nan"),
        ctrl_sig_rand=float("nan"), above_ctrl_perm=False,
        auc_order_perm_med=float("nan"), auc_order_perm_p95=float("nan"),
        gain_sequentiel=float("nan"), order_perm_force_p95=float("nan"),
        gain_significant=False, gain_collapses=False,
        chirality_accuracy=float("nan"),
        gbar_aprime_q=nan3, gbar_ab_q=nan3,
        issue="skip", issue_label=reason,
    )
