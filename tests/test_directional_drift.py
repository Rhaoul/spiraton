"""Tests Tour 22 — dérive DIRECTIONNELLE d'un cycle ABA réel (33D, voie 3 de Ra).

Le Tour 21 a mesuré l'observable SCALAIRE ``ρ̂=‖s_t‖`` (NULL-stationnaire). H22 : la
stationnarité du module masque peut-être une dérive de la DIRECTION (le retournement
de clôture ``<DX><OUT>→<LV><IN>`` renverse l'axe temporel, pas le module). On mesure
des observables GÉOMÉTRIQUES — straightness ``R``, ``cos−l2`` — sur la trajectoire
``s_{t+1}=s_t+g_t·v_t`` pilotée par la cellule canon FIGÉE du T21.

Deux familles (modèle ``test_aba_regulation.py`` / ``test_vector33d_bridge.py``) :

  * SANS ``.so`` (toujours exécutés) : pivots et propriétés géométriques qui ne
    dépendent QUE de pistes fabriquées à la main — pivot bit-à-bit (R reproductible),
    pivot dégénéré (drive constant ⇒ R=1), plancher/plafond bornent R, déterminisme,
    shuffle est une vraie permutation seedée, α-ω sur piste synthétique.

  * AVEC ``.so`` (skip propre si absent) : le pipeline complet sur ``dataset_aba.txt``
    — construction des pistes (dims 8-22), exécution de l'ordre lexicographique,
    finitude, déterminisme du verdict.

Le canon ``core/``, ``RealAbaDrive`` et les constantes du T21 ne sont JAMAIS touchés ;
tout est seedé et déterministe.
"""
import math
from pathlib import Path

import pytest
import torch

from spiraton.data.tokenizer_bridge import is_available, NativeTokenizer33D
from spiraton.diagnostics.aba_regulation import _make_projector
from spiraton.diagnostics.directional_drift import (
    DirectionalDriftReport,
    RealAbaTrack,
    TAU,
    WILCOXON_ALPHA,
    alpha_omega_on_track,
    floor_track,
    ramp_track,
    run_directional_drift,
    run_track,
    segment_straightness,
    straightness,
    _seed_s0,
)


_DATASET = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
if not _DATASET.is_file():
    _DATASET = Path(__file__).resolve().parents[2] / "dataset_aba.txt"

native_available = is_available()


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _hand_track(vectors, seg_bounds=None) -> RealAbaTrack:
    """Piste fabriquée à la main (sans ``.so``) — matrice (N,15) explicite."""
    mat = torch.as_tensor(vectors, dtype=torch.float32)
    n = mat.size(0)
    bounds = seg_bounds if seg_bounds is not None else (max(n // 3, 1), max(2 * n // 3, 2), n)
    return RealAbaTrack(vectors=mat, seg_bounds=bounds, n_tokens=n)


# =============================================================================
# Famille SANS .so : pivots et propriétés géométriques (toujours exécutés)
# =============================================================================

# --- PIVOT bit-à-bit : R réel reproductible à la graine ----------------------

def test_pivot_run_track_deterministic_bit_for_bit() -> None:
    """La trajectoire est reproductible BIT-À-BIT à graine fixée (cellule figée + s0 seedé)."""
    cell = _make_projector(20210601)
    g = torch.Generator().manual_seed(11)
    vectors = torch.randn(10, 15, generator=g)
    s0 = _seed_s0(123)
    st1 = run_track(vectors, cell, s0)
    st2 = run_track(vectors, cell, s0)
    assert torch.equal(st1, st2)
    # la straightness qui en découle est elle aussi bit-à-bit identique
    assert straightness(st1) == straightness(st2)


def test_pivot_projector_frozen_weights_identical() -> None:
    """La cellule de pilotage est seedée : poids identiques à graine identique (figés)."""
    c1 = _make_projector(20210601)
    c2 = _make_projector(20210601)
    assert torch.equal(c1.w_add, c2.w_add)
    assert torch.equal(c1.w_div, c2.w_div)


# --- PIVOT dégénéré : drive constant ⇒ trajectoire rectiligne ⇒ R = 1 --------

def test_pivot_constant_drive_gives_straight_line() -> None:
    """Drive constant (même vecteur poussé à chaque pas) ⇒ trajectoire rectiligne ⇒ R=1.

    Avec un drive constant non nul, chaque pas pousse dans la même direction : net = path
    (à des gains positifs près, qui ne changent pas la direction) ⇒ R = 1 exactement.
    Pivot dégénéré : aucune structure temporelle ⇒ rectitude triviale.
    """
    cell = _make_projector(20210601)
    u = torch.ones(15) * 0.2
    vectors = u.reshape(1, 15).expand(12, 15).contiguous()
    s0 = _seed_s0(3)
    st = run_track(vectors, cell, s0)
    R = straightness(st)
    assert abs(R - 1.0) < 1e-5


def test_pivot_zero_drive_gives_no_motion() -> None:
    """Drive nul ⇒ aucun déplacement ⇒ R = 0 (net=0, path=0 ⇒ 0/(0+eps))."""
    cell = _make_projector(20210601)
    vectors = torch.zeros(8, 15)
    s0 = _seed_s0(5)
    st = run_track(vectors, cell, s0)
    # tous les états égaux à s0
    assert torch.allclose(st, s0.reshape(1, 15).expand(9, 15))
    assert straightness(st) == 0.0


# --- BORNES : plancher gaussien < réel-typique < plafond rampe ----------------

def test_straightness_is_bounded_in_unit_interval() -> None:
    """R ∈ [0,1] pour toute trajectoire (net ≤ path par inégalité triangulaire)."""
    cell = _make_projector(20210601)
    g = torch.Generator().manual_seed(7)
    for _ in range(10):
        vectors = torch.randn(12, 15, generator=g)
        s0 = _seed_s0(int(torch.randint(0, 10_000, (1,), generator=g).item()))
        R = straightness(run_track(vectors, cell, s0))
        assert -1e-9 <= R <= 1.0 + 1e-9


def test_floor_below_ceiling_anchors_threshold() -> None:
    """ANCRAGE : plancher (gaussien i.i.d.) < plafond (rampe) ; plafond = 1.0 exact.

    Le plancher est une marche aléatoire (R → 1/√T, ici nettement < 1) ; le plafond est
    une rampe directionnelle pure (R = 1.0). Les deux repères encadrent tout R réel et
    sont MESURÉS, pas choisis à la main.
    """
    cell = _make_projector(20210601)
    s0 = _seed_s0(0)
    T = 16
    mean_norm = 1.0
    R_floor = straightness(run_track(floor_track(T, mean_norm, 90000), cell, s0))
    R_ceil = straightness(run_track(ramp_track(T, mean_norm, 90001), cell, s0))
    assert abs(R_ceil - 1.0) < 1e-5            # rampe directionnelle sature à 1.0
    assert R_floor < R_ceil                     # plancher strictement sous le plafond
    assert R_floor < 0.6                        # marche aléatoire : nettement non-rectiligne
    # ordre de grandeur 1/√T (marche aléatoire) — borne large, pas réglée sur le résultat
    assert R_floor < 3.0 / math.sqrt(T)


# --- SHUFFLE : permutation seedée, multiset conservé -------------------------

def test_shuffle_is_seeded_permutation_same_multiset() -> None:
    """Le shuffle permute les LIGNES (tokens), reproductible et multiset conservé."""
    track = _hand_track([[float(i)] * 15 for i in range(8)])
    sh1 = track.shuffled(123)
    sh2 = track.shuffled(123)
    assert torch.equal(sh1.vectors, sh2.vectors)              # seedé
    # multiset des lignes conservé (somme par ligne identique en ensemble)
    rs_orig = sorted(float(r.sum()) for r in track.vectors)
    rs_sh = sorted(float(r.sum()) for r in sh1.vectors)
    assert rs_orig == rs_sh
    sh3 = track.shuffled(124)
    assert not torch.equal(sh3.vectors, sh1.vectors)          # graine differente ⇒ ordre different


# --- ALPHA-OMEGA sur piste synthetique : formule coherente -------------------

def test_alpha_omega_identity_when_a_prime_returns_to_alpha() -> None:
    """Si l'état final = s_α exactement, cos→1 et l2→0 (retour IDENTIQUE = répétition).

    Piste fabriquée : A pousse, B pousse, A′ pousse l'OPPOSÉ exact du déplacement net de
    B pour ramener l'état à s(finA). On vérifie que l'outil α-ω lit alors un retour
    aligné (cos≈1) ET identique (l2≈0) — le mode dégénéré que le projet oppose à la
    progression (proche-aligné MAIS non identique).
    """
    cell = _make_projector(20210601)
    # 2 tokens par segment ; seg_bounds = (2,4,6)
    s0 = _seed_s0(1)
    # construit des vecteurs tels que le déplacement de A′ annule celui de B ; comme les
    # gains dépendent de l'état on vérifie la cohérence de la formule sur la TRACE réelle
    # via les états mesurés, pas une égalité fabriquée : on teste la propriété de l'outil
    # (cos/l2 entre deux états identiques) directement.
    states = torch.stack([
        s0,
        s0 + torch.ones(15),
        s0 + 2 * torch.ones(15),     # fin A (idx 2)
        s0 + 3 * torch.ones(15),
        s0 + 4 * torch.ones(15),     # fin B (idx 4)
        s0 + 3 * torch.ones(15),
        s0 + 2 * torch.ones(15),     # fin A′ (idx 6) == fin A
    ], dim=0)
    cos_f, l2_f, br = alpha_omega_on_track(states, (2, 4, 6))
    assert abs(cos_f - 1.0) < 1e-5      # final identique a s_alpha ⇒ aligne
    assert l2_f < 1e-5                  # ... et identique (l2≈0) ⇒ REPETITION
    # best_return cherche sur A′ (idx 5..6) ; l'état idx 6 == s_alpha ⇒ best_return=6
    assert br == 6


def test_alpha_omega_progression_aligned_not_identical() -> None:
    """Retour proche-ALIGNÉ mais NON identique : cos haut MAIS l2>0 (progression)."""
    s0 = torch.zeros(15)
    base = torch.ones(15)
    states = torch.stack([
        s0,
        base,
        2 * base,                    # fin A
        3 * base,
        4 * base,                    # fin B
        2.5 * base,
        2.1 * base,                  # fin A′ : proche de fin A (2·base) mais !=
    ], dim=0)
    cos_f, l2_f, br = alpha_omega_on_track(states, (2, 4, 6))
    assert cos_f > 0.99                 # colineaire ⇒ cos≈1 (aligne)
    assert l2_f > 0.0                   # mais distinct ⇒ l2>0 (NON identique) = PROGRESSION


# --- R_seg vs R_token : matriochka coherent ----------------------------------

def test_segment_straightness_well_defined() -> None:
    """R_seg est dans [0,1] et vaut 1 sur une trajectoire rectiligne."""
    cell = _make_projector(20210601)
    u = torch.ones(15) * 0.2
    vectors = u.reshape(1, 15).expand(9, 15).contiguous()
    s0 = _seed_s0(2)
    st = run_track(vectors, cell, s0)
    R_seg = segment_straightness(st, (3, 6, 9))
    assert abs(R_seg - 1.0) < 1e-5
    # cas non rectiligne borne
    g = torch.Generator().manual_seed(4)
    st2 = run_track(torch.randn(9, 15, generator=g), cell, s0)
    R_seg2 = segment_straightness(st2, (3, 6, 9))
    assert 0.0 - 1e-9 <= R_seg2 <= 1.0 + 1e-9


# =============================================================================
# Famille AVEC .so : pipeline complet (skip propre si lib absente)
# =============================================================================

@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_real_tracks_build_and_finite() -> None:
    """Les pistes réelles se construisent (dims 8-22), vecteurs finis, seg_bounds cohérents."""
    from spiraton.diagnostics.directional_drift import collect_real_tracks

    tok = NativeTokenizer33D()
    tracks = collect_real_tracks(str(_DATASET), tok, n_cycles=5)
    assert len(tracks) == 5
    for tr in tracks:
        assert tr.n_tokens >= 4
        assert tr.vectors.shape == (tr.n_tokens, 15)     # dims 8-22 SEULEMENT
        assert _finite(tr.vectors)
        assert 0 < tr.seg_bounds[0] <= tr.seg_bounds[1] <= tr.seg_bounds[2] == tr.n_tokens


@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_run_directional_drift_deterministic_and_null_by_shuffle() -> None:
    """Rapport déterministe ET verdict MESURÉ : null-par-shuffle (R_réel ≈ R_shuffle).

    Résultat MESURÉ (jamais forcé) : la straightness réelle (R_token≈0.65) est
    INDISCERNABLE du shuffle (delta < τ=0.15, Wilcoxon non significatif) — la rectitude
    vient du drive/cellule, PAS de la structure A→B→A′ (leçon T14). On vérifie le
    déterminisme bit-à-bit et la cohérence interne (R entre plancher et plafond). Le
    verdict exact est documenté, pas exigé comme "succès".
    """
    tok = NativeTokenizer33D()
    r1 = run_directional_drift(str(_DATASET), tok, n_cycles=40)
    r2 = run_directional_drift(str(_DATASET), tok, n_cycles=40)
    assert isinstance(r1, DirectionalDriftReport)
    # déterminisme bit-à-bit des quantités décisionnelles
    assert r1.r_token_real_median == r2.r_token_real_median
    assert r1.r_token_shuffle_median == r2.r_token_shuffle_median
    assert r1.r_token_delta_median == r2.r_token_delta_median
    assert r1.wilcoxon_p == r2.wilcoxon_p
    assert r1.verdict == r2.verdict
    # finitude
    assert math.isfinite(r1.r_token_real_median)
    assert math.isfinite(r1.ao_l2_real_median)
    # R réel encadré par les repères mesurés sur place (plancher < réel < plafond)
    assert r1.floor_r_median < r1.r_token_real_median < r1.ceil_r_median + 1e-9
    assert abs(r1.ceil_r_median - 1.0) < 1e-5
    # NULL mesuré : réel indiscernable du shuffle (delta < τ) ⇒ null-par-shuffle
    assert abs(r1.r_token_delta_median) < TAU
    assert r1.verdict == "null-par-shuffle"


@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_null_robust_across_projection_seeds() -> None:
    """Le verdict null-par-shuffle ne dépend PAS de la graine de pilotage (anti-artefact).

    Pour 3 graines de cellule, R_réel reste indiscernable de R_shuffle (|delta| < τ) ⇒
    verdict invariant : la straightness ne porte pas la structure A→B→A′, quelle que
    soit la cellule figée de pilotage.
    """
    tok = NativeTokenizer33D()
    for ps in (20210601, 1, 12345):
        r = run_directional_drift(str(_DATASET), tok, n_cycles=40, proj_seed=ps)
        assert abs(r.r_token_delta_median) < TAU
        assert r.verdict == "null-par-shuffle"


@pytest.mark.skipif(
    not (native_available and _DATASET.is_file()),
    reason="tokenizer natif (.so/.dll) ou dataset_aba.txt indisponible",
)
def test_matriochka_scale_signal_present_but_nonstructural() -> None:
    """R_seg > R_token (signal d'échelle) MAIS présent AUSSI sous shuffle (non structurel).

    La dérive est plus rectiligne au SEGMENT (R_seg≈0.87) qu'au TOKEN (R_token≈0.65) :
    signal d'échelle (matriochka, porte vers un tour inter-cycles). Mais le shuffle
    REPRODUIT le même saut d'échelle ⇒ il vient de la géométrie d'intégration (moins de
    déplacements nets ⇒ plus droit), PAS de la structure A→B→A′.
    """
    tok = NativeTokenizer33D()
    r = run_directional_drift(str(_DATASET), tok, n_cycles=40)
    # saut d'échelle présent dans le réel
    assert r.r_seg_real_median > r.r_token_real_median
    # ... et reproduit par le shuffle (non structurel)
    assert r.r_seg_shuffle_median > r.r_token_shuffle_median
    # le saut d'échelle n'est PAS porté par l'ordre (delta de R_seg sous τ)
    assert abs(r.r_seg_delta_median) < TAU
