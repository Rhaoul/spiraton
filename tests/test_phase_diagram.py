"""Tests Tour 29 — diagramme de phase (N, k, η) : découpage exact de l'axe η (``phase_diagram``).

Protocole T29 gelé (émission ``TOUR29_EMISSION.md``) : balayage ``Fraction`` exact
maille 1/12 sur (0, 4], convergence 1/24 (1/48 au besoin), bissection rationnelle
exacte ≤ 2⁻²⁰, certificat de rationalité, prédicats P1-P5, points de contrôle float
a priori. Les valeurs ci-dessous sont GRAVÉES d'après la mesure (2026-07-12) —
y compris les RÉFUTATIONS (P2 violée sous 4, P4-(7,4) réfuté, 2.2 réfutée par 74
certificats e_{t≥3}, arrêt technique formel aux bornes de (9,7)) : elles sont
rapportées telles quelles, jamais forcées.

Aucun dataset requis (grille synthétique pure — pas de volet réel, déclaré à
l'émission). ``structural_gap.py`` reste GELÉ byte-à-byte ; η passe par le
paramètre d'appel ; ``edge_controller.py``, ``horizon_law.py`` (T27) et
``spectral_map.py`` (T28) ne sont JAMAIS modifiés ; tout est déterministe
(shuffles seedés de la porte 0a seulement).
"""
import math
from fractions import Fraction

import pytest

from spiraton.diagnostics.horizon_law import (
    BAND_EXACT,
    TARGET_EXACT,
    gate0a,
    grid_cells,
    is_high_excursion,
)
from spiraton.diagnostics.phase_diagram import (
    CONTROL_ETAS,
    DOMAIN_MAX,
    FROZEN_CONTROL_MAGNITUDE_COUNTS,
    FROZEN_DEG2_RATIONAL,
    FROZEN_E2_CERTIFICATES,
    FROZEN_ENDPOINT_CHECKS,
    FROZEN_ENDPOINT_MISMATCH_CELL,
    FROZEN_ENDPOINT_SIGN_MISMATCHES,
    FROZEN_ETA3_CERTIFIED,
    FROZEN_HISTOGRAM,
    FROZEN_NIVEN_COUNTS,
    FROZEN_P2_AT_4,
    FROZEN_P2_NONZERO_BELOW_4,
    FROZEN_TOTALS,
    MESH_CONVERGENCE,
    MESH_FINE,
    MESH_PRIMARY,
    NIVEN_PERIODS,
    RESOLUTION,
    _clip_pattern,
    _pdeg,
    _peval,
    cell_diagram,
    endpoint_float_report,
    float_control_report,
    p2_report,
    p3_report,
    periodic_windows,
    resonance_report,
    run_phase_diagram,
    sign_at,
    simplest_in_closed,
    structure_at_mesh,
    trace_e_polys,
)
from spiraton.diagnostics.spectral_map import (
    delta_nom_exact_eta,
    delta_nom_float_eta,
    trace_full_exact,
)


@pytest.fixture(scope="module")
def report():
    """Le diagramme complet (228 cellules), calculé UNE fois (déterministe, exact)."""
    return run_phase_diagram()


@pytest.fixture(scope="module")
def by_cell(report):
    return {(c.n, c.k): c for c in report.cells}


# =============================================================================
# Quatuor 1/4 — protocole et formes : mailles, domaine, résolution gelés
# =============================================================================

def test_frozen_protocol_and_shapes() -> None:
    """Mailles 1/12 → 1/24 → 1/48, domaine (0, 4], encadrements ≤ 2⁻²⁰, points de
    contrôle dyadiques (exacts au float), périodes Niven gelées."""
    assert (MESH_PRIMARY, MESH_CONVERGENCE, MESH_FINE) == (12, 24, 48)
    assert DOMAIN_MAX == 4 and RESOLUTION == Fraction(1, 2 ** 20)
    assert list(CONTROL_ETAS.values()) == [Fraction(3, 4), Fraction(2),
                                           Fraction(5, 2), Fraction(3), Fraction(7, 2)]
    for eta in CONTROL_ETAS.values():           # dyadiques : float sans perte
        assert Fraction(float(eta)) == eta
    assert NIVEN_PERIODS == {Fraction(1): 6, Fraction(2): 4,
                             Fraction(3): 3, Fraction(4): 2}
    # la maille primaire contient les 4 Niven, l'ancre 1/2 et les quarts
    nodes = {Fraction(i, MESH_PRIMARY) for i in range(1, 4 * MESH_PRIMARY + 1)}
    assert {Fraction(1), Fraction(2), Fraction(3), Fraction(4),
            Fraction(1, 2), Fraction(1, 4), Fraction(3, 4)} <= nodes
    assert len(nodes) == 48


# =============================================================================
# Quatuor 2/4 — formule EXACTE : polynômes symboliques == trace numérique (2.2)
# =============================================================================

def test_symbolic_polys_match_numeric_trace_and_axiom_22() -> None:
    """(a) les e_t(η) symboliques sur la branche de clip évaluent EXACTEMENT la
    trace numérique au même η ; (b) structure 2.2 : e_0, e_1 η-indépendants
    (degré 0), e_2 AFFINE quand le pas 1 n'est pas clipé, deg(e_t) ≤ t−1."""
    for (n, k) in [(7, 4), (8, 6), (13, 8), (24, 12)]:
        for eta in (Fraction(1, 3), Fraction(1), Fraction(7, 5)):
            pat = _clip_pattern(n, k, eta)
            polys = trace_e_polys(n, k, pat)
            e_num, _gs, _clips = trace_full_exact(n, k, eta)
            assert [_peval(p, eta) for p in polys] == list(e_num)
            assert _pdeg(polys[0]) <= 0 and _pdeg(polys[1]) <= 0
            if pat[1] is None:
                assert _pdeg(polys[2]) <= 1              # e_2 affine (2.2)
            assert all(_pdeg(polys[t]) <= t - 1 for t in range(2, len(polys)))


def test_simplest_in_closed_exact() -> None:
    """Stern–Brocot : le rationnel de plus petit dénominateur de [lo, hi]."""
    assert simplest_in_closed(Fraction(1, 3), Fraction(1, 2)) == Fraction(1, 2)
    assert simplest_in_closed(Fraction(3, 7), Fraction(5, 7)) == Fraction(1, 2)
    assert simplest_in_closed(Fraction(19, 40), Fraction(21, 40)) == Fraction(1, 2)
    assert simplest_in_closed(Fraction(41, 40), Fraction(43, 40)) == Fraction(15, 14)
    assert simplest_in_closed(Fraction(2), Fraction(3)) == Fraction(2)
    # largeur 2⁻²⁰ autour de 11/10 : 11/10 est bien le plus simple de la fenêtre
    assert simplest_in_closed(Fraction(11, 10) - RESOLUTION / 2,
                              Fraction(11, 10) + RESOLUTION / 2) == Fraction(11, 10)


# =============================================================================
# Quatuor 3/4 — exactitude / finitude
# =============================================================================

def test_exactness_and_finiteness() -> None:
    """signe ∈ {−1, 0, +1} exact ; Δ Fraction bornée ; float fini, aux nœuds 1/12."""
    for (n, k) in [(6, 2), (8, 6), (13, 8), (24, 22)]:
        for i in (1, 6, 24, 48):
            eta = Fraction(i, 12)
            d = delta_nom_exact_eta(n, k, eta)
            assert isinstance(d, Fraction) and abs(d) <= 1
            assert sign_at(n, k, eta) in (-1, 0, 1)
            assert math.isfinite(delta_nom_float_eta(n, k, float(eta)))


# =============================================================================
# Quatuor 4/4 — déterminisme / pureté
# =============================================================================

def test_determinism_and_purity() -> None:
    """Deux exécutions identiques : structures, frontières, certificats."""
    a = structure_at_mesh(7, 4, MESH_PRIMARY)
    b = structure_at_mesh(7, 4, MESH_PRIMARY)
    assert a == b
    assert cell_diagram(8, 6) == cell_diagram(8, 6)


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
# P1 — convergence de maille : histogramme, 1/48 au besoin, non-stables signalées
# =============================================================================

def test_p1_histogram_and_convergence(report) -> None:
    """Histogramme GRAVÉ {1: 64, 2: 35, 3: 65, 4: 37, 5: 18, 6: 3, 7: 2, 8: 1,
    9: 3} ; 82 cellules ont requis 1/48 ; 35 non stables 1/24↔1/48 (signalées,
    jamais forcées) ; conjecture annexe « majorité ≤ 2 phases » RÉFUTÉE (99/228) ;
    0 incohérence de signes inter-frontières."""
    assert report.histogram == FROZEN_HISTOGRAM
    assert sum(report.histogram.values()) == 228
    assert len(report.needed48_cells) == FROZEN_TOTALS["needed_48"]
    assert len(report.unstable_cells) == FROZEN_TOTALS["unstable"]
    assert set(report.unstable_cells) <= set(report.needed48_cells)
    assert report.majority_le2 is False          # 99/228 — réfutée, gravée
    assert sum(v for p, v in report.histogram.items() if p <= 2) == 99
    assert report.n_gap_inconsistencies == 0


def test_p1_totals_frozen(report) -> None:
    """Totaux GRAVÉS : 406 frontières = 78 certifiées (4 e₂) + 328 encadrements."""
    assert report.n_frontiers_total == FROZEN_TOTALS["frontiers"] == 406
    assert len(report.certified) == FROZEN_TOTALS["certified"] == 78
    assert report.n_e2 == FROZEN_TOTALS["e2"] == 4
    assert len(report.certified_beyond_e2) == FROZEN_TOTALS["certified_beyond_e2"] == 74
    assert len(report.encadrements) == FROZEN_TOTALS["encadrements"] == 328
    assert 78 + 328 == 406
    # chaque encadrement respecte la largeur gelée (fusions sous-résolution incluses)
    for f in report.encadrements:
        assert f.hi - f.lo <= 3 * RESOLUTION
        assert f.certified is None               # jamais affirmé rationnel


# =============================================================================
# P2 — les 19 basses : VIOLÉE sous 4 (gravé tel que mesuré, pas forcé)
# =============================================================================

def test_p2_violated_below_4() -> None:
    """P2 (« 0 partout < 4 ») est VIOLÉE : (8,5) et (11,7) sortent de zéro vers
    η ≈ 3.95-3.96, et (10,7) a une phase − TRANSITOIRE qui revient à 0 avant 4.
    À η = 4 exactement : (8,5) et (11,7) seules (cohérent T28)."""
    p2 = p2_report()
    assert p2.n_low_cells == 19
    assert p2.n_nodes_checked == 19 * 192
    assert tuple(p2.nonzero_below_4) == FROZEN_P2_NONZERO_BELOW_4
    assert tuple(p2.nonzero_at_4) == FROZEN_P2_AT_4
    assert p2.passes is False                    # réfutation gravée
    # (10,7) : ≠ 0 sous 4 mais revient à 0 à η = 4 (phase transitoire)
    assert sign_at(10, 7, Fraction(4)) == 0
    assert sign_at(10, 7, Fraction(95, 24)) == -1


def test_p2_transient_phase_10_7(by_cell) -> None:
    """La phase − transitoire de (10,7) : deux frontières dans (3.94, 3.96)."""
    c = by_cell[(10, 7)]
    assert c.structure.phase_signs == (0, -1, 0)
    f1, f2 = c.structure.frontiers
    assert (f1.sign_left, f1.sign_right) == (0, -1)
    assert (f2.sign_left, f2.sign_right) == (-1, 0)
    assert Fraction(63, 16) < f1.lo < f2.hi < 4  # 3.9375 < … < 4


# =============================================================================
# P4 — (7,4) RÉFUTÉ (plateau zéro, pas de bascule ponctuelle) ; (8,6) CONFIRMÉ
# =============================================================================

def test_p4_74_refuted_zero_plateau(by_cell) -> None:
    """(7,4) : la prédiction « UNE frontière certifiée η = 1 » est RÉFUTÉE —
    8 phases (0, −, 0, −, 0, +, 0, +), 7 frontières toutes en encadrement, et
    η = 1 est INTÉRIEUR à un plateau zéro ≈ (0.89696, 1.14196). La « bascule à
    η = 1 » du T28 était l'échantillonnage ponctuel d'un plateau."""
    c = by_cell[(7, 4)]
    assert c.structure.phase_signs == (0, -1, 0, -1, 0, 1, 0, 1)
    assert c.structure.n_frontiers == 7
    assert all(f.certified is None for f in c.structure.frontiers)
    assert sign_at(7, 4, Fraction(1)) == 0       # cohérent T28 : Δ(7,4; 1) = 0
    f_in, f_out = c.structure.frontiers[3], c.structure.frontiers[4]
    assert f_in.hi < 1 < f_out.lo                # 1 strictement DANS le plateau zéro
    assert (f_in.sign_left, f_in.sign_right) == (-1, 0)
    assert (f_out.sign_left, f_out.sign_right) == (0, 1)


def test_p4_86_zero_exit_bracketed_in_half_one(by_cell) -> None:
    """(8,6) : la sortie du zéro est ENCADRÉE dans (1/2, 1) OUVERTE — confirmé.
    Encadrement gravé (263245/393216, 1052981/1572864], porteur e_7, degré 6.
    Et le zéro ne part pas de 0⁺ : phase + sur (0, ≈ 0.22807)."""
    c = by_cell[(8, 6)]
    assert c.structure.phase_signs == (1, 0, 1)
    f_exit = c.structure.frontiers[1]
    assert (f_exit.sign_left, f_exit.sign_right) == (0, 1)
    assert Fraction(1, 2) < f_exit.lo and f_exit.hi < 1
    assert f_exit.lo == Fraction(263245, 393216)
    assert f_exit.hi == Fraction(1052981, 1572864)
    assert f_exit.carried_by == (7,) and f_exit.degrees == (6,)
    assert f_exit.certified is None              # encadrement, jamais affirmé rationnel


# =============================================================================
# 2.2 — certificats : famille e₂ gravée, ET réfutation par e_{t≥3} rationnels
# =============================================================================

def test_e2_certificates_frozen(report) -> None:
    """Les 4 certificats e₂ (famille (i)) : (7,3) → 11/10, (8,4) → 1/2,
    (10,4) → 5/4, (17,9) → 1/14 — rationnels EXACTS, portés par e_2 seul,
    branche non clipée au pas 1, degré 1. Aucun aux Niven {1, 2, 3, 4}."""
    e2 = {(f.n, f.k): f for f in report.certified if f.is_e2}
    assert {c: f.certified for c, f in e2.items()} == FROZEN_E2_CERTIFICATES
    for f in e2.values():
        assert f.carried_by == (2,) and f.degrees == (1,) and f.clip_stable
        assert f.certified not in (Fraction(1), Fraction(2), Fraction(3), Fraction(4))
        # vérification directe : e_2 EXACTEMENT au bord de bande à η certifié
        e, _gs, _clips = trace_full_exact(f.n, f.k, f.certified)
        assert abs(e[2] - TARGET_EXACT) == BAND_EXACT


def test_axiom_22_refuted_by_rational_t3plus(report) -> None:
    """RÉFUTATION 2.2 (gravée telle quelle) : 74 frontières certifiées
    RATIONNELLES portées par e_{t≥3}. Mécanismes : branches clipées (degré
    retombe à 1), une racine rationnelle de degré 2 ((15,6) → 1/2), et la
    résonance η = 3 (traversées simultanées multi-hits, 7 cellules N = 3k)."""
    beyond = report.certified_beyond_e2
    assert len(beyond) == 74
    assert all(min(f.carried_by) >= 3 for f in beyond)
    # (b) l'unique porteur de degré ≥ 2 à racine rationnelle : (15,6) → 1/2
    deg2 = [(f.n, f.k, f.certified) for f in report.certified if max(f.degrees) >= 2]
    (cell, eta) = FROZEN_DEG2_RATIONAL
    assert deg2 == [(cell[0], cell[1], eta)]
    # (c) résonance η = 3 : 8 cellules certifiées à 3 exactement, dont les 7 N = 3k
    eta3 = sorted((f.n, f.k) for f in report.certified if f.certified == 3)
    assert tuple(eta3) == FROZEN_ETA3_CERTIFIED
    assert [(n, k) for (n, k) in eta3 if n == 3 * k] == [
        (6, 2), (9, 3), (12, 4), (15, 5), (18, 6), (21, 7), (24, 8)]
    multi = {(f.n, f.k): len(f.carried_by) for f in report.certified
             if f.certified == 3}
    assert multi[(6, 2)] == 3 and multi[(24, 8)] == 15   # traversées simultanées


# =============================================================================
# P3 — aucune loi monotone du compte de phases (structure quasi-périodique)
# =============================================================================

def test_p3_no_monotone_law(report) -> None:
    """AUCUNE monotonie propre : 0/19 lignes monotones en k, non monotone en N
    ni en N−k — la structure quasi-périodique prédite, pas une loi simple."""
    p3 = p3_report(report.cells)
    assert p3.monotone_in_k_all_rows is False
    assert p3.monotone_in_n_all_cols is False
    assert p3.monotone_in_leg_all is False
    assert p3.n_rows_monotone == 0 and p3.n_rows == 19


# =============================================================================
# P5 — les résonances de périodicité exacte PIQUENT aux 4 Niven
# =============================================================================

def test_p5_niven_resonance_peaks() -> None:
    """Comptes de fenêtres exactement p-périodiques GRAVÉS : η=1 (p=6) 884 vs
    36/36 ; η=2 (p=4) 1178 vs 49/49 ; η=3 (p=3) 1292 vs 56/56 ; η=4 (p=2) 185
    vs 63 — pics 4/4. Et η=1/p=6 reproduit le 884/884 du T28 (contrôle)."""
    rr = resonance_report()
    got = {int(e): v for e, v in rr.per_niven.items()}
    assert got == FROZEN_NIVEN_COUNTS
    assert all(rr.peaks_at_niven.values())
    n_win, n_ok = periodic_windows(Fraction(1), 6)
    assert (n_win, n_ok) == (884, 884)           # == period6_report T28


# =============================================================================
# VÉRIFICATION INSTRUMENT — 5 points de contrôle : signes 5 × 228/228
# =============================================================================

def test_float_control_points_sign_concordance() -> None:
    """Concordance de SIGNE exact ↔ float GELÉ : 228/228 aux 5 points de
    contrôle a priori {3/4, 2, 5/2, 3, 7/2}. Divergences de MAGNITUDE gravées
    {2, 6, 2, 8, 3}, toutes avec ≥ 1 point exactement au bord (réserve T28,
    rapportées non corrigées)."""
    fc = float_control_report()
    assert fc.passes and fc.sign_mismatches == []
    assert all(fc.matches[lbl] == 228 for lbl in CONTROL_ETAS)
    assert {lbl: len(v) for lbl, v in fc.magnitude_divergences.items()} == \
        FROZEN_CONTROL_MAGNITUDE_COUNTS
    for lbl, divs in fc.magnitude_divergences.items():
        assert all(bt >= 1 for (_n, _k, _f, _e, bt) in divs)


# =============================================================================
# VÉRIFICATION INSTRUMENT — bornes d'encadrements : arrêt technique FORMEL (9,7)
# =============================================================================

def test_endpoint_reserve_97_documented_not_hidden(report) -> None:
    """812 bornes vérifiées ; 5 divergences de SIGNE, TOUTES sur (9,7) : la
    condition d'arrêt technique gelée est FORMELLEMENT déclenchée. Mécanisme
    gravé (pas un mécanisme neuf) : le lecteur FIXE de (9,7) a e_8 = 3/2
    EXACTEMENT au bord ⟹ Δ_float(9,7; η) = Δ_exact + 1/9 à TOUT η (réserve de
    magnitude T27/T28 déjà gravée) ; là où Δ_exact ∈ {0, −1/9}, elle devient
    une divergence de signe. Rapportée NON corrigée — l'ingénieur statue."""
    ep = endpoint_float_report(report.cells)
    assert ep.n_checked == FROZEN_ENDPOINT_CHECKS == 812
    assert len(ep.sign_mismatches) == FROZEN_ENDPOINT_SIGN_MISMATCHES == 5
    assert ep.passes is False                    # jamais lissé
    assert {(m[0], m[1]) for m in ep.sign_mismatches} == {FROZEN_ENDPOINT_MISMATCH_CELL}
    # le mécanisme : e_8 du lecteur fixe est EXACTEMENT au bord (η-invariant)
    e_fixed, _gs, _clips = trace_full_exact(9, 7, Fraction(0))
    assert e_fixed[8] - TARGET_EXACT == BAND_EXACT
    # et le décalage float est la constante +1/9 aux 5 points fautifs
    for (_n, _k, pt, s_exact, s_float) in ep.sign_mismatches:
        d_exact = delta_nom_exact_eta(9, 7, pt)
        d_float = delta_nom_float_eta(9, 7, float(pt))
        assert d_exact in (Fraction(0), Fraction(-1, 9))
        assert d_float - float(d_exact) == pytest.approx(1 / 9, abs=1e-12)


# =============================================================================
# VERDICT GRAVÉ (après mesure) — la synthèse que l'ingénieur arbitre
# =============================================================================

def test_frozen_verdict_summary(report) -> None:
    """Synthèse gelée : le diagramme est LIVRÉ (406 frontières, suite ordonnée
    par cellule, stabilité signalée), ≥ 1 prédicat tranché non-trivialement
    (P4-(7,4) et P2 et 2.2 réfutés ; P4-(8,6), P3, P5 confirmés), concordance
    signe 5 × 228/228 aux points de contrôle. Les frontières non résolues
    restent des encadrements ; les 35 cellules non stables sont signalées ;
    l'arrêt technique (9,7) est rapporté tel quel."""
    assert report.n_frontiers_total == 406
    assert len(report.unstable_cells) == 35
    # chaque cellule livre une suite ORDONNÉE de frontières
    for c in report.cells:
        fs = c.structure.frontiers
        assert all(a.hi <= b.lo for a, b in zip(fs, fs[1:]))
        assert len(c.structure.phase_signs) >= 1
    # cohérence gravure : les basses instables/certifiées ne se recoupent pas à tort
    low = [(n, k) for (n, k) in grid_cells() if not is_high_excursion(n, k)]
    assert len(low) == 19
