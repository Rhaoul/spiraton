"""Tests Tour 28 — carte spectrale de l'organe : déformation de la carte(η) (``spectral_map``).

H28 : la déformation de la carte signe(Δ_nom)(N, k) en η est une ROTATION de
fréquence (cos θ(η) = (2−η)/2, |λ| = 1 sur (0,4)), pas un amortissement. Les cinq
η gelés {1/4, 1/2, 1, 3/2, 4} sont dérivés en ``Fraction`` EXACTE (clip inclus) et
GELÉS dans ``FROZEN_MAPS``/``FROZEN_WITNESSES`` AVANT le chemin float (forme forte
par η). L'ancre η = 1/2 est un CONTRÔLE DUR : elle doit reproduire byte-pour-signe
la carte T27 gravée et les 6 témoins rationnels T27 inchangés.

Aucun dataset requis (grille synthétique pure — pas de volet réel ce tour, déclaré
à l'émission). L'instrument ``structural_gap.py`` reste GELÉ byte-à-byte ; η passe
par le paramètre d'appel de ``reconstruct_profile`` (``ETA_STRUCT`` jamais édité) ;
``edge_controller.py`` et le canon ``core/`` ne sont JAMAIS touchés ; tout est
déterministe (shuffles seedés de la porte 0a seulement).
"""
import math
from fractions import Fraction

import pytest

from spiraton.diagnostics.horizon_law import (
    WITNESS_CELLS,
    delta_nom_exact,
    delta_nom_float,
    grid_cells,
    is_high_excursion,
    trace_e_exact,
)
from spiraton.diagnostics.spectral_map import (
    ETA_LABELS_FROZEN,
    ETAS_FROZEN,
    FROZEN_MAPS,
    FROZEN_N_PLUS_TARDIF,
    FROZEN_PGATE_VIOLATIONS,
    FROZEN_WITNESSES,
    T27_CONTROL_LABEL,
    clip_dominance_report,
    cos_theta_exact,
    delta_nom_exact_eta,
    delta_nom_float_eta,
    derived_map_exact,
    eta_map_report,
    hamming_between_maps,
    period6_report,
    trace_full_exact,
)
from spiraton.diagnostics.horizon_law import gate0a


# carte T27 gravée (docstring horizon_law) — recopiée EN DUR : contrôle indépendant
_T27_MAP = (
    "-0.",
    "0--.",
    "--+.0",
    "0--+.+",
    "0--++.+",
    "00-0+.++",
    "00-0++.++",
    "----0++.++",
    "000-0++.+++",
    "-00-++++.+++",
    "0----0+++.+++",
    "0000-++++.++++",
    "--00-+++++.++++",
    "00---0+++++.++++",
    "-0000-+++++.+++++",
    "0--00-++++++.+++++",
    "000---0++++++.+++++",
    "--0000-++++++.++++++",
    "00--00-+++++++.++++++",
)


@pytest.fixture(scope="module")
def reports():
    """Les 5 rapports η (exact + float), calculés UNE fois (déterministes)."""
    return {lbl: eta_map_report(lbl) for lbl in ETA_LABELS_FROZEN}


# =============================================================================
# Quatuor 1/4 — protocole et formes : les η gelés et les gravures
# =============================================================================

def test_frozen_protocol_and_shapes() -> None:
    """5 η rationnels gelés, cos θ = (2−η)/2 exact, gravures bien formées."""
    assert ETA_LABELS_FROZEN == ("1/4", "1/2", "1", "3/2", "4")
    assert ETAS_FROZEN["1/2"] == Fraction(1, 2)        # l'ancre T27
    # fréquences de rotation exactes (H28) — dont la racine double défective η=4
    assert [cos_theta_exact(ETAS_FROZEN[l]) for l in ETA_LABELS_FROZEN] == [
        Fraction(7, 8), Fraction(3, 4), Fraction(1, 2), Fraction(1, 4), Fraction(-1)]
    for lbl in ETA_LABELS_FROZEN:
        rows = FROZEN_MAPS[lbl]
        assert len(rows) == 19                          # N = 6..24
        for i, row in enumerate(rows):
            n = 6 + i
            assert len(row) == n - 3                    # k = 2..N−2
            assert set(row) <= {"+", "-", "0", "."}
            # '.' n'apparaît QUE sur des cellules basses (le masque ne cache rien)
            for j, ch in enumerate(row):
                if ch == ".":
                    assert not is_high_excursion(n, 2 + j)


# =============================================================================
# Quatuor 2/4 — formule EXACTE sous entrées forcées
# =============================================================================

def test_trace_full_matches_trace_e_exact_and_recurrence() -> None:
    """(a) ``trace_full_exact`` reproduit les e de ``trace_e_exact`` (deux chemins,
    mêmes Fractions) ; (b) sur les pas non clipés du plateau, la boucle fermée
    satisfait EXACTEMENT u_{t+1} = (2−η)·u_t − u_{t−1} (u = e − 1) — la
    récurrence de rotation dont H28 dérive tout."""
    for (n, k) in [(13, 8), (16, 10), (24, 12)]:
        for lbl in ETA_LABELS_FROZEN:
            eta = ETAS_FROZEN[lbl]
            e_ref = trace_e_exact(n, k, eta)
            e, gs, clips = trace_full_exact(n, k, eta)
            assert e == e_ref
            u = [x - 1 for x in e]
            for s in range(1, k):                       # inc constant sur le plateau
                if not clips[s]:
                    assert u[s + 1] == (2 - eta) * u[s] - u[s - 1]


def test_eta_half_engine_equals_t27_engine() -> None:
    """À η = 1/2 le moteur paramétré est EXACTEMENT le moteur T27 (Fraction et
    float, égalité stricte — mêmes chemins de code, mêmes flottants)."""
    for (n, k) in [(7, 4), (8, 6), (13, 8), (15, 11), (24, 22)]:
        assert delta_nom_exact_eta(n, k, Fraction(1, 2)) == delta_nom_exact(n, k)
        assert delta_nom_float_eta(n, k, 0.5) == delta_nom_float(n, k)


# =============================================================================
# Quatuor 3/4 — exactitude / finitude
# =============================================================================

def test_exactness_and_finiteness_all_etas() -> None:
    """Δ_nom(η) est une Fraction bornée (|Δ| ≤ 1) ; le float est fini — aux 5 η."""
    for lbl in ETA_LABELS_FROZEN:
        eta = ETAS_FROZEN[lbl]
        for (n, k) in [(6, 2), (8, 6), (13, 8), (24, 12), (24, 22)]:
            d = delta_nom_exact_eta(n, k, eta)
            assert isinstance(d, Fraction) and abs(d) <= 1
            assert math.isfinite(delta_nom_float_eta(n, k, float(eta)))


# =============================================================================
# Quatuor 4/4 — déterminisme / pureté
# =============================================================================

def test_determinism_and_purity() -> None:
    """Deux appels identiques bit-à-bit (exact ET float) ; Hamming déterministe."""
    for (n, k) in [(7, 4), (13, 8), (15, 11)]:
        for lbl in ("1/4", "4"):
            eta = ETAS_FROZEN[lbl]
            assert delta_nom_exact_eta(n, k, eta) == delta_nom_exact_eta(n, k, eta)
            assert delta_nom_float_eta(n, k, float(eta)) == delta_nom_float_eta(n, k, float(eta))
    assert hamming_between_maps("1", "3/2") == hamming_between_maps("1", "3/2")


# =============================================================================
# PORTE 0a re-confirmée (héritée T27 — une fois, mandat émission)
# =============================================================================

def test_gate0a_reconfirmed() -> None:
    """Porte 0a re-jouée : primaire sensible (≥ δ_min = 1e-4), multiset vacuous."""
    g = gate0a()
    assert g.primary_high.gap == pytest.approx(0.2185897, abs=1e-6)
    assert g.primary_low.gap == pytest.approx(0.0809524, abs=1e-6)
    assert g.multiset_high.gap == 0.0 and g.multiset_low.gap == 0.0
    assert g.passes


# =============================================================================
# CONTRÔLE DUR η = 1/2 : la carte T27 et ses témoins, byte-pour-signe
# =============================================================================

def test_hard_control_eta_half_reproduces_t27(reports) -> None:
    """L'ancre η = 1/2 reproduit la gravure T27 : carte byte-pour-signe (recopiée
    EN DUR ci-dessus, indépendamment de horizon_law), témoins rationnels T27
    inchangés, divergences de magnitude identiques ((9,7) et (15,11))."""
    assert FROZEN_MAPS[T27_CONTROL_LABEL] == _T27_MAP
    assert derived_map_exact(Fraction(1, 2)) == _T27_MAP
    r = reports[T27_CONTROL_LABEL]
    assert r.map_matches_frozen
    assert r.witness_deltas == dict(WITNESS_CELLS)      # les valeurs T27, inchangées
    assert r.witnesses_match_frozen and r.witnesses_float_ok
    assert [(c.n, c.k) for c in r.magnitude_divergences] == [(9, 7), (15, 11)]


# =============================================================================
# FORME FORTE + CONCORDANCE : gel honoré, signes 5 × 228/228, magnitudes gravées
# =============================================================================

def test_frozen_maps_match_derivation(reports) -> None:
    """Le gel est honoré : à chaque η, la carte dérivée == la gravure du module."""
    for lbl in ETA_LABELS_FROZEN:
        assert derived_map_exact(ETAS_FROZEN[lbl]) == FROZEN_MAPS[lbl]
        assert reports[lbl].map_matches_frozen


def test_sign_concordance_5x228_and_magnitude_divergences(reports) -> None:
    """Concordance dérivation ↔ instrument float GELÉ : SIGNES 5 × 228/228
    (divergence de signe = BUG, arrêt technique). Divergences de MAGNITUDE
    MESURÉES et gravées — chacune avec ≥ 1 point EXACTEMENT au bord |e−1| = 1/2 :
    2 (η=1/4), 2 (1/2, == T27), 6 (1), 2 (3/2), 10 (4)."""
    expected_mag = {
        "1/4": [(9, 7), (15, 11)],
        "1/2": [(9, 7), (15, 11)],
        "1": [(9, 7), (14, 7), (15, 11), (16, 8), (18, 9), (20, 10)],
        "3/2": [(9, 7), (15, 11)],
        "4": [(8, 6), (9, 4), (9, 7), (12, 9), (15, 11), (16, 12),
              (20, 15), (21, 10), (24, 11), (24, 18)],
    }
    for lbl in ETA_LABELS_FROZEN:
        r = reports[lbl]
        assert r.sign_mismatches == []                  # 228/228, par η
        assert [(c.n, c.k) for c in r.magnitude_divergences] == expected_mag[lbl]
        assert all(c.band_touches >= 1 for c in r.magnitude_divergences)


def test_witnesses_frozen_per_eta(reports) -> None:
    """Les 6 témoins prennent à chaque η leurs valeurs Fraction GELÉES avant float ;
    le float les reproduit à 1e-12 partout SAUF η = 4 où (8,6) est une divergence
    de magnitude au bord exact (float 1/8 vs Fraction 1/4, SIGNE préservé)."""
    for lbl in ETA_LABELS_FROZEN:
        r = reports[lbl]
        assert r.witness_deltas == FROZEN_WITNESSES[lbl]
        assert r.witnesses_match_frozen
        assert r.witnesses_float_ok == (lbl != "4")
    assert delta_nom_float_eta(8, 6, 4.0) == pytest.approx(0.125)
    assert delta_nom_exact_eta(8, 6, Fraction(4)) == Fraction(1, 4)


# =============================================================================
# PRÉDICAT 1 — P-gate par η : tient jusqu'à 3/2, VIOLÉE à η = 4 (η_crit = 4)
# =============================================================================

def test_predicate1_pgate_per_eta(reports) -> None:
    """P-gate (cellules basses ⟹ Δ = 0) : TIENT 19/19 à η ∈ {1/4, 1/2, 1, 3/2}
    (le clip protège la bande) ; VIOLÉE à η = 4 par (8,5) → −1/8 et (11,7) →
    −1/11 (basses-précoces) — l'organe sur-corrige et sort de bande en basse
    excursion. η_crit = 4 dans la famille gelée."""
    for lbl in ETA_LABELS_FROZEN:
        viol = [(c.n, c.k, c.delta_exact) for c in reports[lbl].pgate_violations]
        assert viol == list(FROZEN_PGATE_VIOLATIONS[lbl])
    assert FROZEN_PGATE_VIOLATIONS["4"] == (
        (8, 5, Fraction(-1, 8)), (11, 7, Fraction(-1, 11)))
    eta_crit = next(lbl for lbl in ETA_LABELS_FROZEN if FROZEN_PGATE_VIOLATIONS[lbl])
    assert eta_crit == "4"
    # les deux violations sont bien des Δ < 0 (sur-correction, prédiction confirmée)
    assert all(d < 0 for (_n, _k, d) in FROZEN_PGATE_VIOLATIONS["4"])


# =============================================================================
# PRÉDICAT 2 — le zéro (8,6) : plateau de basse fréquence, pas coïncidence à 1/2
# =============================================================================

def test_predicate2_zero_86_leaves_zero_above_half() -> None:
    """Δ(8,6; η) exact : 0 à η ∈ {1/4, 1/2} ; +1/4 à η ∈ {1, 3/2, 4}. La
    prédiction « quitte zéro pour η ≠ 1/2 » est RÉFUTÉE à 1/4 (le zéro tient
    aussi) et confirmée au-dessus : le zéro est un PLATEAU de basse fréquence,
    pas une coïncidence ponctuelle de η = 1/2. Pas « zéro pour tout η » non
    plus : la réfutation dure du prédicat est écartée."""
    expected = {"1/4": Fraction(0), "1/2": Fraction(0),
                "1": Fraction(1, 4), "3/2": Fraction(1, 4), "4": Fraction(1, 4)}
    for lbl, want in expected.items():
        assert delta_nom_exact_eta(8, 6, ETAS_FROZEN[lbl]) == want


# =============================================================================
# PRÉDICAT 3 — bascule des témoins : (7,4) change de signe (η de bascule = 1)
# =============================================================================

def test_predicate3_witness_sign_flip() -> None:
    """(7,4) : −1/7 → −2/7 → 0 → +1/7 → +1/7 — CHANGE de signe, et la bascule
    passe par Δ = 0 EXACT à η = 1 (le η de bascule, dérivé pas ajusté).
    (13,8) : +1/13 puis +3/13 stable — ne bascule JAMAIS. « Au moins un témoin
    change de signe sur (0,4) » : CONFIRMÉ par (7,4)."""
    vals_74 = [delta_nom_exact_eta(7, 4, ETAS_FROZEN[l]) for l in ETA_LABELS_FROZEN]
    assert vals_74 == [Fraction(-1, 7), Fraction(-2, 7), Fraction(0),
                       Fraction(1, 7), Fraction(1, 7)]
    assert vals_74[0] < 0 < vals_74[3] and vals_74[2] == 0     # bascule à η = 1
    vals_138 = [delta_nom_exact_eta(13, 8, ETAS_FROZEN[l]) for l in ETA_LABELS_FROZEN]
    assert vals_138 == [Fraction(1, 13), Fraction(3, 13), Fraction(3, 13),
                        Fraction(3, 13), Fraction(3, 13)]
    assert all(v > 0 for v in vals_138)                        # jamais de bascule


# =============================================================================
# PRÉDICAT 4 — N_plus_tardif(η) CROÎT : la direction attendue est RÉFUTÉE
# =============================================================================

def test_predicate4_n_plus_tardif_increases(reports) -> None:
    """N_plus_tardif (strate haute, 57 cellules tardives) : 55, 56, 57, 57, 57 —
    il CROÎT quand η monte. La direction mécaniste attendue à l'émission
    (décroît dans (0,~2)) est RÉFUTÉE par la dérivation — gravée, pas forcée."""
    seq = [reports[lbl].n_plus_tardif_high for lbl in ETA_LABELS_FROZEN]
    assert seq == [FROZEN_N_PLUS_TARDIF[lbl] for lbl in ETA_LABELS_FROZEN]
    assert seq == [55, 56, 57, 57, 57]
    assert all(a <= b for a, b in zip(seq, seq[1:]))    # croissant, jamais décroissant


# =============================================================================
# PRÉDICAT 5 — η = 1 (θ = π/3) : périodicité-6 exacte sur plateaux non clipés
# =============================================================================

def test_predicate5_period6_at_eta1() -> None:
    """u_{t+6} == u_t EXACT (Fraction) sur TOUTES les fenêtres de plateau non
    clipées à η = 1 : 884/884 fenêtres, 125 cellules éligibles, 0 échec —
    structure de bande mod-6 CONFIRMÉE (cos θ = 1/2, θ = π/3, période 6)."""
    p6 = period6_report()
    assert p6.n_cells_eligible == 125
    assert p6.n_windows == 884
    assert p6.n_windows_ok == 884
    assert p6.failures == []


# =============================================================================
# PRÉDICAT 6 — η = 4 (racine double défective) : clip-dominée, mais Δ>0 majoritaire
# =============================================================================

def test_predicate6_clip_dominance_at_eta4() -> None:
    """η = 4 : clip-dominée — fraction de pas aux bornes {min 0, MÉDIANE 2/3,
    max 23/24}, 61/228 cellules saturées (tous les t ≥ 1 aux bornes). MAIS les
    signes sont {+: 142, 0: 59, −: 27} et Δ ≤ 0 ne fait que 86/228 : la moitié
    « majoritairement Δ≤0 » de la prédiction est RÉFUTÉE (gravé, pas forcé).
    Qualitativement distincte : seule carte à violer la P-gate."""
    cd = clip_dominance_report()
    assert cd.frac_min == 0
    assert cd.frac_median == Fraction(2, 3)
    assert cd.frac_max == Fraction(23, 24)
    assert cd.n_saturated_after_first == 61
    assert cd.sign_counts == {"+": 142, "0": 59, "-": 27}
    assert cd.n_delta_le0 == 86
    assert cd.sign_counts["+"] > 228 // 2               # majoritairement POSITIF


# =============================================================================
# H28 — la carte n'est PAS η-invariante (rotation, pas amortissement)
# =============================================================================

def test_h28_maps_not_eta_invariant() -> None:
    """Hamming inter-cartes gravé : {1/4↔1/2: 50, 1/4↔1: 60, 1/4↔3/2: 65,
    1/4↔4: 76, 1/2↔1: 24, 1/2↔3/2: 29, 1/2↔4: 40, 1↔3/2: 8, 1↔4: 19,
    3/2↔4: 12} — tous > 0 : la réfutation « carte η-invariante » est écartée.
    Et AUCUN η ne rend la carte monotone-simple : le côté précoce haut garde
    les trois signes à chaque η (réfutation « mécanisme faux » écartée)."""
    expected = {("1/4", "1/2"): 50, ("1/4", "1"): 60, ("1/4", "3/2"): 65,
                ("1/4", "4"): 76, ("1/2", "1"): 24, ("1/2", "3/2"): 29,
                ("1/2", "4"): 40, ("1", "3/2"): 8, ("1", "4"): 19, ("3/2", "4"): 12}
    for (a, b), h in expected.items():
        assert hamming_between_maps(a, b) == h
        assert h > 0
    # mélange {+, 0, −} persistant côté précoce haut, à tout η (jamais monotone)
    for lbl in ETA_LABELS_FROZEN:
        eta = ETAS_FROZEN[lbl]
        signs = {(delta_nom_exact_eta(n, k, eta) > 0) - (delta_nom_exact_eta(n, k, eta) < 0)
                 for (n, k) in grid_cells()
                 if is_high_excursion(n, k) and 3 * k - 2 * n < 0}
        assert signs == {-1, 0, 1}
