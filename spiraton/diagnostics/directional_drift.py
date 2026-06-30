from __future__ import annotations

"""Dérive DIRECTIONNELLE d'un cycle ABA réel (Tour 22).

VOIE 3 de Ra — quitter l'observable SCALAIRE pour une signature DIRECTIONNELLE.
Le Tour 21 a mesuré l'observable scalaire ``ρ̂ = ‖s_t‖`` sur la dérive 33D réelle
et conclu NULL-STATIONNAIRE (5e null). Hypothèse H22 (linguiste) : la stationnarité
du MODULE peut MASQUER une dérive de la DIRECTION. Argument : le retournement de
clôture ABA ``<DX><OUT> → <LV><IN>`` au troisième segment renverse l'AXE TEMPOREL,
pas le module — la signature de clôture, si elle existe, est ANGULAIRE, et T21
mesurait précisément la grandeur que ce retournement laisse invariante.

OPÉRATIONNALISATION (émission du linguiste, geste DIV·dextro·out — respectée telle
quelle ; on RÉUTILISE le drive 33D et la cellule canon figée du T21, on ne change que
ce qu'on MESURE en sortie) :

  * DRIVE RÉEL. Pour chaque cycle de ``dataset_aba.txt``, le pont natif donne
    ``tokens → (N, 33)``. On forme la séquence des tokens DANS L'ORDRE du cycle
    (SEG_A → SEG_B → SEG_A_PRIME), dims **8-22 SEULEMENT** (phonémique). On EXCLUT
    les dims 0-5 (op + chiralité) et ``op_from_vector`` — sinon on suit l'étiquette
    constante = CIRCULARITÉ (REFUS, interdit).

  * TRAJECTOIRE pilotée par la CELLULE CANON figée du T21 (``_make_projector``,
    poids FIGÉS, jamais entraînés — pas de fit, pas de circularité). État
    ``s_t ∈ R^15`` dans le MÊME espace que le drive phonémique ; le pas est
    ``s_{t+1} = s_t + g_t · v_t`` où ``v_t`` = vecteur phonémique du token ``t`` (la
    DIRECTION de poussée) et ``g_t = cell(s_t)`` = gain scalaire natif borné autour
    de ``BASE_GAIN`` (la cellule canon PILOTE l'intégration sans rien apprendre).
    C'est une intégration cumulative de la dérive phonémique, modulée par la cellule.
    Elle est GÉOMÉTRIQUE de bout en bout : aucune étiquette n'entre.

OBSERVABLES (géométriques, JAMAIS une étiquette) :

  1. STRAIGHTNESS ``R = ‖s_fin − s_début‖ / (Σ_t ‖s_{t+1} − s_t‖ + eps) ∈ [0, 1]``.
     Sans échelle. ``R ≈ 1`` = trajectoire rectiligne (dérive directionnelle nette) ;
     ``R ≈ 0`` = marche qui revient sur elle-même (pas de direction privilégiée).
  2. OUTIL CANON ``cos − l2`` (formule ``alpha_omega_metrics`` INTACTE, importée) :
     série ``cos(s_α, s_t)`` et ``l2(s_α, s_t)`` avec ``s_α = s(fin SEG_A)`` comme
     ORIGINE, et ``best_return_step`` cherché SUR SEG_A_PRIME. Signature de clôture
     spirale : ``cos`` remonte vers ≈1 en fin de A′ (retour ALIGNÉ) MAIS ``l2 > 0``
     (NON-identique) — proche-et-aligné mais non identique.
  3. MATRIOCHKA 2 échelles (gratuit, mêmes ``s_t``) : ``R_token`` (straightness sur
     tous les pas ``s_{t+1} − s_t``) vs ``R_seg`` (straightness sur les 3 déplacements
     NETS de segment : ``s(finA) − s_0``, ``s(finB) − s(finA)``, ``s(finA′) − s(finB)``).
     ``R_seg > R_token`` = dérive visible AU SEGMENT pas au token (signal d'échelle,
     porte vers un tour inter-cycles) ; l'inverse est aussi informatif.

ORDRE LEXICOGRAPHIQUE GRAVÉ A PRIORI (seuils GELÉS AVANT toute mesure ; premier
verdict atteint = résultat). ``τ = 0.15`` (ordre des seuils-décisifs T17/T18) :

  (i)   ``directionnel-structurel`` [POSITIF] : ``R_réel − R_shuffle ≥ τ`` ET
        Wilcoxon apparié sur la population de cycles ``p < 0.01``. La non-stationnarité
        réelle est DIRECTIONNELLE, le réel se rouvre.
  (ii)  ``null-par-shuffle`` : ``|R_réel − R_shuffle| < τ``. La rectitude vient du
        drive/cellule, PAS de la structure A→B→A′ (leçon T14). NULL légitime.
  (iii) ``null-stationnaire-en-direction`` : ``R_réel ≈ R_shuffle ≈ plancher``. La
        direction aussi revient sur elle-même → 6e null.

ANCRAGE du seuil sur 2 repères MESURÉS SUR PLACE (pas choisis à la main, REFUS) :
  * PLANCHER = ``R`` d'un drive GAUSSIEN i.i.d. de même norme moyenne (≈ marche
    aléatoire, ``R → 1/√T``) ;
  * PLAFOND = ``R`` d'une RAMPE directionnelle synthétique (drive ``= t·u``, ``u``
    fixe → sature à 1.0).
On situe ``R_réel`` / ``R_shuffle`` entre plancher et plafond.

BASELINES (non négociables, leçon T14/REFUS) :
  1. ``best_fixed`` : meilleur drive CONSTANT (= aucune structure temporelle).
  2. ``shuffle`` (LE comparateur décisif) : MÊME multiset de vecteurs token, ORDRE
     permuté (seed fixe + N permutations pour la stat Wilcoxon). Le shuffle ne permute
     QUE l'ordre des tokens, JAMAIS les étiquettes.
  3. ``plancher`` gaussien + ``plafond`` rampe (ancrage du seuil ci-dessus).
Population de VRAIS cycles (≥ 40) + Wilcoxon apparié.

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/`` NI ``RealAbaDrive`` NI les
constantes du T21. Tout déterministe (seeds fixés). Si le ``.so`` est absent,
``collect_real_tracks`` lève via le pont (skip propre côté tests).
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

from ..data.aba import AbaCycle, iter_aba_cycles
from ..data import vector33d
from ..data.tokenizer_bridge import NativeTokenizer33D
from .alpha_omega_spatial import alpha_omega_metrics  # formule cos−l2 INTACTE
from .aba_regulation import (  # on RÉUTILISE la cellule canon figée du T21, telle quelle
    BASE_GAIN,
    GAIN_SPAN,
    PROJ_SEED,
    PROJ_TEMP,
    _make_projector,
    _median,
    wilcoxon_signed_rank,
)

# --- constantes GELÉES A PRIORI (REFUS : jamais réglées sur le résultat) ------

# Seuil décisif de l'ordre lexicographique (ordre des seuils T17/T18).
TAU = 0.15

# Wilcoxon : seuil de significativité de la population de cycles.
WILCOXON_ALPHA = 0.01

# Nombre de permutations shuffle par cycle (pour une médiane de shuffle stable et la
# stat de population). N permutations seedées déterministes par cycle.
N_SHUFFLE = 8

# Graine de l'état initial s_0 (déterministe par cycle) et bases de seeds shuffle /
# plancher (gelées, jamais réglées sur le résultat).
S0_SEED_BASE = 30000
SHUFFLE_SEED_BASE = 80000
FLOOR_SEED_BASE = 90000

# Cellule de pilotage : réutilise la projection canon figée du T21 (poids jamais
# entraînés). On garde BASE_GAIN / GAIN_SPAN / PROJ_SEED / PROJ_TEMP du T21 (cohérence).
EPS = 1e-12


# --- la piste réelle : séquence de vecteurs phonémiques d'un cycle ABA --------

@dataclass(frozen=True)
class RealAbaTrack:
    """Séquence des vecteurs phonémiques (dims 8-22) des tokens d'un cycle ABA.

    ``vectors`` est un tenseur ``(N, 15)`` : ligne ``t`` = dims 8-22 du token ``t``,
    dans l'ordre SEG_A → SEG_B → SEG_A_PRIME. ``seg_bounds`` = indices de FIN cumulés
    (fin_A, fin_B, fin_A_PRIME) — bornes des segments pour le diagnostic α-ω et le
    matriochka R_seg. AUCUNE dimension 0-5 n'est stockée (pas d'étiquette).
    """

    vectors: torch.Tensor              # (N, 15) float32
    seg_bounds: Tuple[int, int, int]   # (fin_A, fin_B, fin_A_PRIME) en indices de token
    n_tokens: int

    def shuffled(self, seed: int) -> "RealAbaTrack":
        """ORDRE-DÉTRUIT : permute les LIGNES (tokens), même multiset, structure détruite.

        Permutation seedée déterministe (``torch.randperm``). Les seg_bounds ne sont
        plus signifiantes après shuffle ; conservées par symétrie de longueur (servent
        au matriochka mais le shuffle casse précisément la structure de segment).
        """
        n = self.n_tokens
        if n < 2:
            return self
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        return RealAbaTrack(
            vectors=self.vectors[perm].contiguous(),
            seg_bounds=self.seg_bounds,
            n_tokens=n,
        )


def build_real_track(cycle: AbaCycle, tok: NativeTokenizer33D) -> RealAbaTrack:
    """Construit la piste d'un cycle : tokenise A,B,A′ → dims 8-22 → ``(N, 15)``.

    Les trois segments sont tokenisés SÉPARÉMENT dans l'ordre A→B→A′ (le pont natif
    travaille au niveau du texte de segment), concaténés. ``seg_bounds`` = indices de
    fin cumulés. On ne lit JAMAIS les dims 0-5.
    """
    import numpy as np

    rows: List[torch.Tensor] = []
    bounds: List[int] = []
    for seg in (cycle.seg_a, cycle.seg_b, cycle.seg_a_prime):
        vecs = tok.vectors(seg.text)            # (N_seg, 33) float32
        for row in vecs:
            phon = torch.from_numpy(np.asarray(row[vector33d.PHONEME_SIG], dtype=np.float32))
            rows.append(phon)
        bounds.append(len(rows))
    if rows:
        mat = torch.stack(rows, dim=0).to(torch.float32)
    else:
        mat = torch.zeros((0, 15), dtype=torch.float32)
    return RealAbaTrack(vectors=mat, seg_bounds=(bounds[0], bounds[1], bounds[2]), n_tokens=len(rows))


def collect_real_tracks(
    path: str,
    tok: NativeTokenizer33D,
    *,
    n_cycles: int = 40,
    min_tokens: int = 4,
) -> List[RealAbaTrack]:
    """Construit ``n_cycles`` pistes réelles DÉTERMINISTES depuis le début du corpus.

    Cycles pris dans l'ORDRE du fichier (déterminisme), gardés si ≥ ``min_tokens``
    tokens (sinon trop court pour une dérive — a priori, pas réglé sur le résultat,
    aligné sur ``collect_real_drives`` du T21).
    """
    tracks: List[RealAbaTrack] = []
    for cycle in iter_aba_cycles(path):
        tr = build_real_track(cycle, tok)
        if tr.n_tokens >= min_tokens:
            tracks.append(tr)
        if len(tracks) >= n_cycles:
            break
    return tracks


# --- état initial déterministe (même loi que T21 mais en dim 15) --------------

def _seed_s0(seed: int) -> torch.Tensor:
    """État initial déterministe (norme ~1, jamais nul) dans l'espace phonémique R^15."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(15, generator=g)
    n = float(torch.linalg.vector_norm(v))
    if n < 1e-6:
        v = torch.zeros(15)
        v[0] = 1.0
        n = 1.0
    return (v / n).to(torch.float32)


# --- la trajectoire géométrique pilotée par la cellule canon figée ------------

@torch.no_grad()
def run_track(
    vectors: torch.Tensor,
    cell,
    s0: torch.Tensor,
    *,
    base_gain: float = BASE_GAIN,
    gain_span: float = GAIN_SPAN,
    temp: float = PROJ_TEMP,
) -> torch.Tensor:
    """Déroule ``s_{t+1} = s_t + g_t · v_t``, ``g_t = base + span·tanh(cell(s_t)/temp)``.

    ``vectors`` (N, 15) = directions de poussée phonémiques. ``cell`` = SpiratonCell
    canon FIGÉE (T21). Le gain ``g_t`` est borné autour de ``base_gain`` (la cellule
    pilote l'intégration sans rien apprendre). Retourne la trace ``(N+1, 15)`` des
    états ``s_0 … s_N``. GÉOMÉTRIQUE : ``v_t`` ne porte aucune étiquette (dims 8-22).
    """
    n = vectors.size(0)
    states = [s0]
    s = s0
    for t in range(n):
        out = cell(s.reshape(1, -1)).reshape(-1)[0]
        g = base_gain + gain_span * math.tanh(float(out) / temp)
        s = s + g * vectors[t]
        states.append(s)
    return torch.stack(states, dim=0)   # (N+1, 15)


# --- les observables géométriques --------------------------------------------

def straightness(states: torch.Tensor, eps: float = EPS) -> float:
    """``R = ‖s_fin − s_début‖ / (Σ_t ‖s_{t+1} − s_t‖ + eps) ∈ [0, 1]`` (sans échelle)."""
    if states.size(0) < 2:
        return 0.0
    net = float(torch.linalg.vector_norm(states[-1] - states[0]))
    steps = states[1:] - states[:-1]                       # (N, 15)
    path = float(torch.linalg.vector_norm(steps, dim=-1).sum())
    return net / (path + eps)


def segment_straightness(states: torch.Tensor, seg_bounds: Tuple[int, int, int], eps: float = EPS) -> float:
    """``R_seg`` : straightness sur les 3 déplacements NETS de segment.

    Déplacements : ``s(finA) − s_0``, ``s(finB) − s(finA)``, ``s(finA′) − s(finB)``.
    ``states`` a N+1 lignes (s_0..s_N) ; ``seg_bounds`` = indices de fin de token
    (fin_A, fin_B, fin_A′) ⇒ états aux indices ``seg_bounds`` (s après le dernier
    token du segment).
    """
    fa, fb, fap = seg_bounds
    pts = [states[0], states[fa], states[fb], states[fap]]
    segs = [pts[i + 1] - pts[i] for i in range(3)]          # 3 déplacements nets
    net = float(torch.linalg.vector_norm(pts[-1] - pts[0]))
    path = float(sum(torch.linalg.vector_norm(d) for d in segs))
    return net / (path + eps)


def alpha_omega_on_track(
    states: torch.Tensor, seg_bounds: Tuple[int, int, int]
) -> Tuple[float, float, int]:
    """``cos − l2`` avec ORIGINE ``s_α = s(fin SEG_A)`` ; ``best_return`` cherché SUR A′.

    Réutilise ``alpha_omega_metrics`` (formule INTACTE). Retourne
    ``(cos_final_A′, l2_final_A′, best_return_step)`` où :
      * ``cos_final_A′`` / ``l2_final_A′`` = cos/l2 entre ``s_α`` et ``s_N`` (état final,
        après le dernier token de A′) — le RETOUR à l'origine-de-cycle ``s_α`` ;
      * ``best_return_step`` = pas ``t`` SUR le segment A′ (de fin_B+1 à fin_A′) qui
        MAXIMISE ``cos − l2`` vers ``s_α`` (le meilleur retour aligné-mais-non-identique).
    Origine ``s_α`` = état après le dernier token de SEG_A : c'est l'« A » du cycle dont
    A′ doit être le retour transformé.
    """
    fa, fb, fap = seg_bounds
    s_alpha = states[fa].reshape(1, -1)                     # origine = fin de SEG_A
    s_final = states[fap].reshape(1, -1)                    # fin de SEG_A_PRIME
    l2_f, cos_f = alpha_omega_metrics(s_alpha, s_final)
    cos_final = float(cos_f.reshape(-1)[0])
    l2_final = float(l2_f.reshape(-1)[0])

    # best_return cherché STRICTEMENT sur le segment A′ : indices d'ÉTAT fb+1 .. fap.
    best_step = fap
    best_score = -float("inf")
    for t in range(fb + 1, fap + 1):
        l2_t, cos_t = alpha_omega_metrics(s_alpha, states[t].reshape(1, -1))
        score = float(cos_t.reshape(-1)[0]) - float(l2_t.reshape(-1)[0])
        if score > best_score:
            best_score = score
            best_step = t
    return cos_final, l2_final, best_step


# --- ancrage du seuil : plancher gaussien, plafond rampe ----------------------

def floor_track(n_tokens: int, mean_norm: float, seed: int) -> torch.Tensor:
    """PLANCHER : drive GAUSSIEN i.i.d. de norme moyenne ``mean_norm`` (marche aléatoire).

    Chaque pas est un vecteur gaussien i.i.d. renormalisé à ``mean_norm`` ⇒ la
    trajectoire est une marche aléatoire ⇒ ``R → 1/√T``. Drive géométrique synthétique
    SANS structure : repère bas du seuil (mesuré, pas choisi).
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n_tokens, 15, generator=g)
    norms = torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-9)
    return (v / norms) * mean_norm


def ramp_track(n_tokens: int, mean_norm: float, seed: int) -> torch.Tensor:
    """PLAFOND : RAMPE directionnelle ``v_t = mean_norm · u`` (``u`` fixe) ⇒ ``R = 1.0``.

    Drive purement directionnel : tous les pas pointent dans la même direction ``u``
    (tirée seedée, fixe sur la séquence). ``net = path`` ⇒ ``R`` sature à 1.0. Repère
    haut du seuil (mesuré, pas choisi).
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(15, generator=g)
    u = u / torch.linalg.vector_norm(u).clamp_min(1e-9)
    return (mean_norm * u).reshape(1, 15).expand(n_tokens, 15).contiguous()


# --- rapport complet ----------------------------------------------------------

@dataclass(frozen=True)
class DirectionalDriftReport:
    """Verdict T22 : straightness réel vs shuffle, matriochka, α-ω, ancrage, ordre gelé."""

    n_cycles: int
    tokens_per_cycle: List[int]
    # straightness réel (médianes de population)
    r_token_real_median: float
    r_seg_real_median: float
    # straightness shuffle (médiane sur N permutations par cycle, puis sur cycles)
    r_token_shuffle_median: float
    r_seg_shuffle_median: float
    # écarts réel − shuffle (population appariée sur R_token)
    r_token_delta: List[float]            # R_token_réel − R_token_shuffle par cycle
    r_token_delta_median: float
    r_seg_delta_median: float
    wilcoxon_p: float
    wilcoxon_signed_w: float
    sign_pos: int                          # nb de cycles où R_réel > R_shuffle
    # baselines d'ancrage (mesurées sur place)
    floor_r_median: float                  # plancher gaussien i.i.d.
    ceil_r_median: float                   # plafond rampe directionnelle
    # baseline best_fixed (drive constant = aucune structure temporelle)
    r_best_fixed_median: float
    # diagnostic central α-ω (origine s(finA), cible s(finA′))
    ao_cos_real_median: float
    ao_l2_real_median: float
    ao_best_return_real_median: float
    ao_cos_shuffle_median: float
    ao_l2_shuffle_median: float
    # seuil et verdict
    tau: float
    verdict: str                           # directionnel-structurel / null-par-shuffle / null-stationnaire-en-direction


def _shuffle_medians_for_cycle(
    track: RealAbaTrack, cell, s0: torch.Tensor, *, seed_base: int, n_shuffle: int
) -> Tuple[float, float, float, float]:
    """Médianes (sur ``n_shuffle`` permutations) de R_token, R_seg, cos, l2 d'un cycle shufflé."""
    rt, rs, cos_l, l2_l = [], [], [], []
    for k in range(n_shuffle):
        sh = track.shuffled(seed_base + k)
        st = run_track(sh.vectors, cell, s0)
        rt.append(straightness(st))
        rs.append(segment_straightness(st, sh.seg_bounds))
        cos_f, l2_f, _ = alpha_omega_on_track(st, sh.seg_bounds)
        cos_l.append(cos_f)
        l2_l.append(l2_f)
    return _median(rt), _median(rs), _median(cos_l), _median(l2_l)


def run_directional_drift(
    path: str,
    tok: NativeTokenizer33D,
    *,
    n_cycles: int = 40,
    proj_seed: int = PROJ_SEED,
    n_shuffle: int = N_SHUFFLE,
) -> DirectionalDriftReport:
    """Exécute l'ordre lexicographique (i)→(ii)→(iii) sur ``n_cycles`` cycles réels.

    DÉTERMINISTE : cycles pris dans l'ordre du fichier, ``s_0`` seedé par index de
    cycle, cellule de pilotage seedée (``proj_seed``, poids FIGÉS), shuffle/plancher
    seedés. Le canon et ``RealAbaDrive`` ne sont jamais touchés.
    """
    cell = _make_projector(proj_seed)
    tracks = collect_real_tracks(path, tok, n_cycles=n_cycles)
    n = len(tracks)

    r_token_real, r_seg_real = [], []
    r_token_sh_med, r_seg_sh_med = [], []
    r_token_delta = []
    ao_cos_r, ao_l2_r, ao_br_r = [], [], []
    ao_cos_sh, ao_l2_sh = [], []
    floor_rs, ceil_rs, best_fixed_rs = [], [], []

    for i, track in enumerate(tracks):
        s0 = _seed_s0(S0_SEED_BASE + i)

        # --- réel : trajectoire et observables -------------------------------
        st = run_track(track.vectors, cell, s0)
        rt = straightness(st)
        rs = segment_straightness(st, track.seg_bounds)
        r_token_real.append(rt)
        r_seg_real.append(rs)
        cos_f, l2_f, br = alpha_omega_on_track(st, track.seg_bounds)
        ao_cos_r.append(cos_f)
        ao_l2_r.append(l2_f)
        ao_br_r.append(float(br))

        # --- shuffle : ordre détruit, N permutations -------------------------
        sh_rt, sh_rs, sh_cos, sh_l2 = _shuffle_medians_for_cycle(
            track, cell, s0, seed_base=SHUFFLE_SEED_BASE + i * 100, n_shuffle=n_shuffle
        )
        r_token_sh_med.append(sh_rt)
        r_seg_sh_med.append(sh_rs)
        ao_cos_sh.append(sh_cos)
        ao_l2_sh.append(sh_l2)
        r_token_delta.append(rt - sh_rt)

        # --- ancrage : plancher gaussien, plafond rampe (même norme moyenne) -
        mean_norm = float(torch.linalg.vector_norm(track.vectors, dim=-1).mean())
        nt = track.n_tokens
        st_floor = run_track(floor_track(nt, mean_norm, FLOOR_SEED_BASE + i), cell, s0)
        floor_rs.append(straightness(st_floor))
        st_ceil = run_track(ramp_track(nt, mean_norm, FLOOR_SEED_BASE + 500000 + i), cell, s0)
        ceil_rs.append(straightness(st_ceil))

        # --- best_fixed : drive constant (le vecteur moyen du cycle, répété) --
        # « meilleur » drive sans structure temporelle = le drive constant ; ici on
        # prend le vecteur MOYEN du cycle (sans étiquette) répété nt fois ⇒ R = 1 par
        # construction si le vecteur moyen est non nul (référence : aucune structure
        # temporelle ⇒ rectiligne trivial). Mesuré pour situer le réel.
        mean_vec = track.vectors.mean(dim=0, keepdim=True).expand(nt, 15).contiguous()
        st_bf = run_track(mean_vec, cell, s0)
        best_fixed_rs.append(straightness(st_bf))

    w_plus, p, n_eff = wilcoxon_signed_rank(r_token_delta)
    sign_pos = sum(1 for d in r_token_delta if d > 0)

    r_token_real_median = _median(r_token_real)
    r_token_shuffle_median = _median(r_token_sh_med)
    r_token_delta_median = _median(r_token_delta)
    r_seg_delta_median = _median([a - b for a, b in zip(r_seg_real, r_seg_sh_med)])

    # --- verdict selon l'ordre lexicographique GELÉ (τ, Wilcoxon) ------------
    if r_token_delta_median >= TAU and p < WILCOXON_ALPHA:
        verdict = "directionnel-structurel"
    elif abs(r_token_delta_median) < TAU:
        verdict = "null-par-shuffle"
    else:
        verdict = "null-stationnaire-en-direction"

    return DirectionalDriftReport(
        n_cycles=n,
        tokens_per_cycle=[t.n_tokens for t in tracks],
        r_token_real_median=r_token_real_median,
        r_seg_real_median=_median(r_seg_real),
        r_token_shuffle_median=r_token_shuffle_median,
        r_seg_shuffle_median=_median(r_seg_sh_med),
        r_token_delta=r_token_delta,
        r_token_delta_median=r_token_delta_median,
        r_seg_delta_median=r_seg_delta_median,
        wilcoxon_p=p,
        wilcoxon_signed_w=w_plus,
        sign_pos=sign_pos,
        floor_r_median=_median(floor_rs),
        ceil_r_median=_median(ceil_rs),
        r_best_fixed_median=_median(best_fixed_rs),
        ao_cos_real_median=_median(ao_cos_r),
        ao_l2_real_median=_median(ao_l2_r),
        ao_best_return_real_median=_median(ao_br_r),
        ao_cos_shuffle_median=_median(ao_cos_sh),
        ao_l2_shuffle_median=_median(ao_l2_sh),
        tau=TAU,
        verdict=verdict,
    )
