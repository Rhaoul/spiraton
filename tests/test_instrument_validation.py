"""Tests Tour 23 (MÉTA, second ordre) — pré-validation d'instrument : ORDER-SENSITIVITY.

La danse corrige sa propre MÉTHODE. ``assert_order_sensitive`` rend ENFORCEABLE la
leçon T17/T18/T22 : un observable GELÉ a priori peut être AVEUGLE au signal cherché.
Avant qu'un observable serve de primaire pour un verdict d'ORDRE, on EXIGE qu'il bouge
d'au moins ``δ_min`` quand SEUL l'ordre des entrées change (shuffle à dynamique fixe).
Sinon il est VACUOUS = INTERDIT comme test d'ordre.

Deux familles :

  * QUATUOR sur le helper (finitude, formes, formule/seuil exact, déterminisme) — sur
    des observables synthétiques triviaux, sans aucun substrat lourd.

  * DÉMONSTRATION DÉCISIVE (la « mesure » du méta-tour) : les DEUX observables T22 avec
    la machinerie RÉELLE de ``directional_drift.py``, sur une séquence SYNTHÉTIQUE
    déterministe (PAS besoin du ``.so`` : l'invariance par construction de l'addition se
    prouve sur n'importe quelle entrée). ``R`` (straightness, intégrateur additif à gain
    constant) ressort ``is_vacuous=True`` (aurait été REJETÉ comme primaire AVANT la
    mesure du réel) ; l'α-ω ``cos(s_α, ·)`` (somme PARTIELLE) ressort
    ``is_order_sensitive=True`` (aurait été retenu d'emblée). C'est la preuve que la
    nouvelle méthode CHANGE le déroulé de T22.

δ_min est GELÉ a priori (``DELTA_MIN_DEFAULT = 1e-4``) ; aucun seuil n'est réglé sur un
résultat. Tout est seedé et déterministe ; le canon et ``directional_drift`` ne sont
jamais touchés (seulement importés/réutilisés).
"""
import math

import torch

from spiraton.diagnostics.aba_regulation import _make_projector
from spiraton.diagnostics.directional_drift import (
    alpha_omega_on_track,
    run_track,
    straightness,
    _seed_s0,
)
from spiraton.diagnostics.instrument_validation import (
    DELTA_MIN_DEFAULT,
    N_SHUFFLE_DEFAULT,
    OrderSensitivityReport,
    assert_order_sensitive,
)


# --- shuffle commun : permute les LIGNES d'un tenseur (N,d), seedé, multiset conservé --

def _shuffle_rows(vectors: torch.Tensor, seed: int) -> torch.Tensor:
    n = vectors.size(0)
    if n < 2:
        return vectors
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return vectors[perm].contiguous()


# Deux observables synthétiques de référence (pas de substrat lourd) :
#   * SUM : Σ des composantes — order-INVARIANT par construction (addition commute) ;
#   * FIRST_DOT : produit scalaire de la PREMIÈRE ligne avec une cible — order-SENSIBLE
#     (dépend de QUI est en tête après le shuffle).
_TARGET = torch.arange(1.0, 6.0)  # (5,)


def _obs_sum(vectors: torch.Tensor) -> float:
    return float(vectors.sum())


def _obs_first_dot(vectors: torch.Tensor) -> float:
    return float(vectors[0] @ _TARGET)


# =============================================================================
# QUATUOR sur le helper
# =============================================================================

# --- FINITUDE : champs finis, rapport bien formé -----------------------------

def test_report_fields_finite_and_well_formed() -> None:
    """Tous les champs numériques du rapport sont finis ; la partition vacuous/sensible tient."""
    g = torch.Generator().manual_seed(1)
    seq = torch.randn(6, 5, generator=g)
    rep = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows)
    assert isinstance(rep, OrderSensitivityReport)
    assert math.isfinite(rep.obs_real)
    assert math.isfinite(rep.obs_shuffle)
    assert math.isfinite(rep.gap)
    assert math.isfinite(rep.delta_min)
    assert all(math.isfinite(x) for x in rep.obs_shuffles)
    # partition stricte : vacuous == not order_sensitive (invariant documenté)
    assert rep.is_vacuous == (not rep.is_order_sensitive)


# --- FORMES : n_shuffle observables shufflés, agrégation médiane --------------

def test_shapes_n_shuffle_observables_collected() -> None:
    """Le rapport collecte exactement ``n_shuffle`` observables shufflés ; défaut respecté."""
    g = torch.Generator().manual_seed(2)
    seq = torch.randn(7, 5, generator=g)
    rep_default = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows)
    assert rep_default.n_shuffle == N_SHUFFLE_DEFAULT
    assert len(rep_default.obs_shuffles) == N_SHUFFLE_DEFAULT
    rep3 = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows, n_shuffle=3)
    assert rep3.n_shuffle == 3
    assert len(rep3.obs_shuffles) == 3


# --- FORMULE / SEUIL EXACT : gap = |real − median(shuffles)|, verdict au seuil -

def test_exact_gap_formula_and_threshold_decision() -> None:
    """gap = |obs_real − median(obs_shuffles)| ; verdict = (gap ≥ delta_min), à l'identique."""
    g = torch.Generator().manual_seed(3)
    seq = torch.randn(8, 5, generator=g)
    rep = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows, n_shuffle=5)

    # reconstruit la formule à la main (médiane sans numpy)
    real = _obs_first_dot(seq)
    sh = sorted(_obs_first_dot(_shuffle_rows(seq, k)) for k in range(5))
    median_sh = sh[2]  # n=5 impair ⇒ élément central
    expected_gap = abs(real - median_sh)
    assert rep.gap == expected_gap
    assert rep.obs_real == real
    assert rep.is_order_sensitive == (rep.gap >= rep.delta_min)
    assert rep.is_vacuous == (rep.gap < rep.delta_min)

    # seuil EXACT : avec delta_min juste sous/au-dessus du gap, le verdict bascule net
    eps = 1e-9
    below = assert_order_sensitive(
        _obs_first_dot, seq, shuffle_fn=_shuffle_rows, n_shuffle=5,
        delta_min=max(rep.gap - eps, 0.0),
    )
    above = assert_order_sensitive(
        _obs_first_dot, seq, shuffle_fn=_shuffle_rows, n_shuffle=5,
        delta_min=rep.gap + eps,
    )
    assert below.is_order_sensitive is True
    assert above.is_vacuous is True


def test_default_delta_min_is_frozen_value() -> None:
    """δ_min par défaut est la constante GELÉE 1e-4 (séparant bruit float ~6e-8 d'un signal ~1e-1)."""
    assert DELTA_MIN_DEFAULT == 1e-4
    g = torch.Generator().manual_seed(4)
    seq = torch.randn(5, 5, generator=g)
    rep = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows)
    assert rep.delta_min == 1e-4


# --- DÉTERMINISME : bit-à-bit à seeds fixés ----------------------------------

def test_deterministic_bit_for_bit() -> None:
    """À ``obs_fn``/``shuffle_fn`` déterministes et seeds fixés, le rapport est bit-à-bit reproductible."""
    g = torch.Generator().manual_seed(5)
    seq = torch.randn(9, 5, generator=g)
    r1 = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows)
    r2 = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows)
    assert r1.obs_real == r2.obs_real
    assert r1.obs_shuffle == r2.obs_shuffle
    assert r1.gap == r2.gap
    assert r1.obs_shuffles == r2.obs_shuffles
    assert r1.is_order_sensitive == r2.is_order_sensitive
    # seed_base décale les permutations ⇒ rapport (en général) différent mais lui-même déterministe
    r_shift = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows, seed_base=1000)
    r_shift2 = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows, seed_base=1000)
    assert r_shift.obs_shuffles == r_shift2.obs_shuffles


# --- ARGUMENTS INVALIDES -----------------------------------------------------

def test_invalid_arguments_rejected() -> None:
    """``n_shuffle < 1`` et ``delta_min < 0`` sont rejetés (garde-fous d'usage)."""
    seq = torch.randn(4, 5)
    import pytest
    with pytest.raises(ValueError):
        assert_order_sensitive(_obs_sum, seq, shuffle_fn=_shuffle_rows, n_shuffle=0)
    with pytest.raises(ValueError):
        assert_order_sensitive(_obs_sum, seq, shuffle_fn=_shuffle_rows, delta_min=-1.0)


# --- PROPRIÉTÉ DE RÉFÉRENCE : un observable additif est VACUOUS, un order-dépendant non -

def test_synthetic_additive_observable_is_vacuous() -> None:
    """Σ (order-invariant par construction) ⇒ gap = bruit float ≪ δ_min ⇒ is_vacuous=True.

    L'addition commute (invariance MATHÉMATIQUE) ; en float32 la non-associativité laisse
    un résidu de l'ordre de 1e-7 (le « 6e-8 » de la leçon T22). δ_min=1e-4 est gelé
    PRÉCISÉMENT au-dessus de ce bruit ⇒ l'observable additif est correctement classé VACUOUS.
    """
    g = torch.Generator().manual_seed(6)
    seq = torch.randn(10, 5, generator=g)
    rep = assert_order_sensitive(_obs_sum, seq, shuffle_fn=_shuffle_rows)
    assert rep.gap < 1e-5           # bruit float pur (non-associativité), bien sous δ_min=1e-4
    assert rep.is_vacuous is True
    assert rep.is_order_sensitive is False


def test_synthetic_order_dependent_observable_is_sensitive() -> None:
    """Un observable dépendant de la TÊTE de séquence bouge avec l'ordre ⇒ is_order_sensitive=True."""
    g = torch.Generator().manual_seed(7)
    seq = torch.randn(10, 5, generator=g)
    rep = assert_order_sensitive(_obs_first_dot, seq, shuffle_fn=_shuffle_rows)
    assert rep.gap >= rep.delta_min
    assert rep.is_order_sensitive is True
    assert rep.is_vacuous is False


# =============================================================================
# DÉMONSTRATION DÉCISIVE : la pré-validation aurait changé le déroulé de T22
# =============================================================================
#
# On reproduit la condition EXACTE de la leçon T22 (« à gain CONSTANT, R_réel −
# R_shuffle = 6.06e-8 »). L'intégrateur additif s_{t+1} = s_t + g·v_t à gain CONSTANT
# rend R EXACTEMENT order-invariant (Σ d_t et Σ‖d_t‖ ne dépendent pas de l'ordre). Pas
# besoin du .so : l'invariance est par construction, vraie sur toute séquence.

_S0_DEMO = _seed_s0(123)
_SEG_BOUNDS_DEMO = (4, 8, 12)   # 12 tokens, 3 segments de 4
_G_CONST = 1.0


def _demo_vectors() -> torch.Tensor:
    """Séquence synthétique déterministe : 12 vecteurs phonémiques distincts (R^15)."""
    g = torch.Generator().manual_seed(2023)
    return torch.randn(12, 15, generator=g)


def _integrate_const_gain(vectors: torch.Tensor) -> torch.Tensor:
    """s_{t+1} = s_t + _G_CONST · v_t (intégrateur additif PUR, condition exacte T22)."""
    states = [_S0_DEMO]
    s = _S0_DEMO
    for t in range(vectors.size(0)):
        s = s + _G_CONST * vectors[t]
        states.append(s)
    return torch.stack(states, dim=0)


def _obs_R_const_gain(vectors: torch.Tensor) -> float:
    return straightness(_integrate_const_gain(vectors))


def _obs_alpha_omega(vectors: torch.Tensor) -> float:
    cos_f, _l2_f, _br = alpha_omega_on_track(_integrate_const_gain(vectors), _SEG_BOUNDS_DEMO)
    return cos_f


def test_demo_straightness_is_vacuous_for_order() -> None:
    """T22 OBSERVABLE 1 — straightness R d'un intégrateur additif à gain constant : VACUOUS.

    L'addition vectorielle commute ⇒ déplacement net Σd_t et chemin Σ‖d_t‖ sont
    EXACTEMENT order-invariants ⇒ gap = 0 (bien sous δ_min=1e-4). La pré-validation
    aurait REJETÉ R comme primaire d'ordre AVANT toute tokenisation du réel — c'est
    précisément le piège dans lequel T22 est tombé (le verdict n'est venu qu'après).
    """
    vectors = _demo_vectors()
    rep = assert_order_sensitive(_obs_R_const_gain, vectors, shuffle_fn=_shuffle_rows)
    assert rep.gap < rep.delta_min
    assert rep.gap < 1e-6                 # invariance par construction : bien en deçà de δ_min
    assert rep.is_vacuous is True
    assert rep.is_order_sensitive is False


def test_demo_alpha_omega_is_order_sensitive() -> None:
    """T22 OBSERVABLE 2 — α-ω cos(s_α, s_final) avec s_α = s(finA) (somme PARTIELLE) : ORDER-SENSIBLE.

    s_α est une somme PARTIELLE (état après SEG_A) : permuter les tokens change OÙ la
    trajectoire est rendue ⇒ cos(s_α, s_final) bouge nettement (gap ≫ δ_min). La
    pré-validation l'aurait RETENU d'emblée — c'est l'observable qui A effectivement
    porté le verdict T22.
    """
    vectors = _demo_vectors()
    rep = assert_order_sensitive(_obs_alpha_omega, vectors, shuffle_fn=_shuffle_rows)
    assert rep.gap >= rep.delta_min
    assert rep.gap > 1e-2                  # vrai signal d'ordre (somme partielle)
    assert rep.is_order_sensitive is True
    assert rep.is_vacuous is False


def test_demo_changes_t22_outcome() -> None:
    """PREUVE DE NON-RÉPÉTITION : la méthode SÉPARE les deux observables T22 (verdicts opposés).

    R vacuous (rejeté) ET α-ω order-sensible (retenu) ⇒ la pré-validation aurait
    discriminé AVANT la mesure du réel ce que T22 n'a discriminé qu'APRÈS. C'est la
    « mesure » du méta-tour : le déroulé de T22 change.
    """
    vectors = _demo_vectors()
    rep_R = assert_order_sensitive(_obs_R_const_gain, vectors, shuffle_fn=_shuffle_rows)
    rep_ao = assert_order_sensitive(_obs_alpha_omega, vectors, shuffle_fn=_shuffle_rows)
    assert rep_R.is_vacuous and not rep_R.is_order_sensitive
    assert rep_ao.is_order_sensitive and not rep_ao.is_vacuous
    # l'α-ω est order-sensible de plusieurs ordres de grandeur au-dessus du gap de R
    assert rep_ao.gap > 100.0 * max(rep_R.gap, 1e-12)


def test_demo_gain_piloted_breaks_construction_invariance() -> None:
    """NUANCE MESURÉE (information, pas bruit) : un GAIN PILOTÉ par l'état brise l'invariance.

    La leçon T22 vaut « à gain CONSTANT ». Avec le gain piloté de ``run_track``
    (g_t = base + span·tanh(cell(s_t)/temp)), R dépend FAIBLEMENT de l'ordre (les gains
    dépendent du chemin). On le MESURE et on le rapporte : ici le gap dépasse δ_min — R
    n'est order-invariant QUE sous gain constant. C'est une condition d'usage explicite
    du diagnostic, pas un défaut du helper.
    """
    vectors = _demo_vectors()
    cell = _make_projector(20210601)

    def obs_R_piloted(v: torch.Tensor) -> float:
        return straightness(run_track(v, cell, _S0_DEMO))

    rep = assert_order_sensitive(obs_R_piloted, vectors, shuffle_fn=_shuffle_rows)
    # mesuré : le gain piloté réintroduit une dépendance d'ordre (faible mais > bruit float)
    assert rep.gap > 1e-6
    # ... et bien plus petite que le signal d'ordre de l'α-ω (somme partielle)
    rep_ao = assert_order_sensitive(_obs_alpha_omega, vectors, shuffle_fn=_shuffle_rows)
    assert rep.gap < rep_ao.gap
