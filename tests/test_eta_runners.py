"""Tests Tour 30 — η sur runners réels : loi de pureté ÉTENDUE P-η (``eta_runners``).

H30 : pour chaque cycle réel MONO-FLIP (N, k) lu par ``aba.py``,
Δ_real(η) == delta_nom_exact_eta(N, k, η) (valeur ≤ 1e-12, identité de signe),
SAUF sur les cellules à bord-exact PRÉ-DÉCLARÉES où l'OFFSET float lui-même est
prédit exactement (table recopiée des gravures T28, LECTURE SEULE). Portes gelées
(ordre lexicographique) : A recensement a priori, 0a order-sensibilité, B baseline
best_fixed == 1.0 ×3 + byte-identité du chemin par défaut, C contrôle dur η = 1/2
(2116/2116 signe ET valeur), D mesure fraîche 4 η × 3 corpus.

Le kwarg ``eta`` de ``run_structural_regulation`` (T30) a pour DÉFAUT la constante
gelée ``ETA_STRUCT`` : le chemin par défaut est BYTE-IDENTIQUE aux runs T24/T25/T26
(testé ici, jamais supposé). ``structural_gap.py`` reste GELÉ byte-à-byte ;
``horizon_law``/``spectral_map``/``phase_diagram`` importés en lecture seule ;
skip propre si les corpus sont absents (AUCUN ``.so`` requis) ; tout déterministe.
"""
import inspect
import math
from fractions import Fraction
from pathlib import Path

import pytest

from spiraton.data.aba import parse_aba_line
from spiraton.experimental.structural_gap import ETA_STRUCT, orientation_profile
from spiraton.diagnostics.eta_runners import (
    ETA_FRESH_LABELS,
    ETA_LABELS_RUNNERS,
    EXPECTED_CONTROL_TOTAL,
    FROZEN_OFFSETS,
    OFFSETS_ETA_INVARIANT,
    T30_CONTROL_LABEL,
    VALUE_TOL,
    census,
    corpus_runs,
    default_path_byte_identical,
    eta_corpus_report,
    frontier_alerts,
    in_grid,
    is_mono_flip,
    transition_counts,
)
from spiraton.diagnostics.spectral_map import (
    ETAS_FROZEN,
    delta_nom_exact_eta,
    delta_nom_float_eta,
)
from spiraton.diagnostics.structural_regulation import run_structural_regulation

_DATASET = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
if not _DATASET.is_file():
    _DATASET = Path(__file__).resolve().parents[2] / "dataset_aba.txt"

_CORPUS_CLAUDE = Path("F:/code/claude/spiraton-enhanced/corpus_claude_aba.txt")
if not _CORPUS_CLAUDE.is_file():
    _CORPUS_CLAUDE = Path(__file__).resolve().parents[2] / "corpus_claude_aba.txt"

_HAVE_CORPORA = _DATASET.is_file() and _CORPUS_CLAUDE.is_file()

_LINE = (
    "<SEG_A> <MUL><DX><OUT><ALPHA> un deux trois </SEG_A> "
    "<SEG_B> <MUL><DX><OUT><OMEGA> quatre cinq </SEG_B> "
    "<SEG_A_PRIME> <MUL><LV><IN><A_PRIME> six sept huit neuf.<EOL> </SEG_A_PRIME> <EOL>"
)


# =============================================================================
# Quatuor 1/4 — protocole et formes : η gelés, table d'offsets pré-déclarée
# =============================================================================

def test_frozen_protocol_and_offset_table() -> None:
    """Les 5 η, l'ancre 1/2, la table d'offsets (émission §4.4) : gelés, bien formés."""
    assert ETA_LABELS_RUNNERS == ("1/2", "1/4", "1", "3/2", "4")
    assert T30_CONTROL_LABEL == "1/2"
    assert ETA_FRESH_LABELS == ("1/4", "1", "3/2", "4")
    assert VALUE_TOL == 1e-12
    assert EXPECTED_CONTROL_TOTAL == 40 + 76 + 2000 == 2116
    # les η-invariantes (bord-exact lecteur FIXE, T27) sont dans CHAQUE liste par-η
    assert OFFSETS_ETA_INVARIANT == {(9, 7): Fraction(1, 9), (15, 11): Fraction(1, 15)}
    for lbl in ETA_LABELS_RUNNERS:
        for cell, off in OFFSETS_ETA_INVARIANT.items():
            assert FROZEN_OFFSETS[lbl][cell] == off
    # cardinaux == divergences de magnitude gravées T28 : 2, 2, 6, 2, 10
    assert {lbl: len(FROZEN_OFFSETS[lbl]) for lbl in ETA_LABELS_RUNNERS} == {
        "1/2": 2, "1/4": 2, "1": 6, "3/2": 2, "4": 10}
    assert FROZEN_OFFSETS["4"][(16, 12)] == Fraction(-1, 16)
    # toutes les cellules à offset sont EN GRILLE (la table vient de la grille T27)
    for lbl in ETA_LABELS_RUNNERS:
        assert all(in_grid(n, k) for (n, k) in FROZEN_OFFSETS[lbl])


def test_offsets_are_float_minus_exact_of_gravures() -> None:
    """Contre-épreuve LECTURE SEULE : offset gravé == Δ_float − Δ_exact du moteur
    T28, cellule par cellule, à chaque η — la transcription des gravures est
    fidèle (et chaque offset est une divergence RÉELLE, > 1e-9)."""
    for lbl in ETA_LABELS_RUNNERS:
        eta = ETAS_FROZEN[lbl]
        for (n, k), off in FROZEN_OFFSETS[lbl].items():
            d_exact = delta_nom_exact_eta(n, k, eta)
            d_float = delta_nom_float_eta(n, k, float(eta))
            assert abs(d_float - float(d_exact + off)) <= VALUE_TOL
            assert abs(d_float - float(d_exact)) > 1e-9


# =============================================================================
# Quatuor 2/4 — formule exacte sous profils forcés : canonicité, grille
# =============================================================================

def test_canonicity_frozen_definitions() -> None:
    """Mono-flip ⟺ exactement une transition +1→−1 et aucune −1→+1 (émission §4.2)."""
    assert transition_counts([+1, +1, -1, -1]) == (1, 0)
    assert is_mono_flip([+1] * 5 + [-1] * 3)
    assert not is_mono_flip([+1, -1, +1, -1])       # multi-flip : 2 transitions ↓
    assert transition_counts([+1, -1, +1, -1]) == (2, 1)
    assert not is_mono_flip([-1, +1, +1, -1])       # retour −1→+1 : hors-carte
    assert not is_mono_flip([+1, +1, +1])           # 0-flip
    assert not is_mono_flip([-1, -1])               # 0-flip
    # un cycle ABA canonique parsé est TOUJOURS mono-flip (fait de grammaire)
    prof = orientation_profile(parse_aba_line(_LINE))
    assert is_mono_flip([tk.orientation for tk in prof])


def test_in_grid_frozen_bounds() -> None:
    """Grille T27 : N ∈ [6, 24], k ∈ [2, N−2] — bornes exactes, pas d'à-peu-près."""
    assert in_grid(6, 2) and in_grid(6, 4) and in_grid(24, 22)
    assert not in_grid(6, 5)        # k > N−2
    assert not in_grid(5, 3)        # N < 6
    assert not in_grid(25, 12)      # N > 24
    assert not in_grid(9, 1)        # k < 2


# =============================================================================
# Quatuor 3/4 — finitude / exactitude des prédictions
# =============================================================================

def test_predictions_finite_and_exact_types() -> None:
    """Prédiction = Fraction exacte (+ offset Fraction) ; le float en est fini."""
    for lbl in ETA_LABELS_RUNNERS:
        eta = ETAS_FROZEN[lbl]
        for (n, k) in [(6, 4), (7, 4), (15, 10), (16, 12), (21, 14)]:
            d = delta_nom_exact_eta(n, k, eta)
            assert isinstance(d, Fraction) and abs(d) <= 1
            off = FROZEN_OFFSETS[lbl].get((n, k), Fraction(0))
            assert math.isfinite(float(d + off))


# =============================================================================
# Quatuor 4/4 — déterminisme / pureté (recensement sur fixture, sans corpus)
# =============================================================================

def test_census_deterministic_on_fixture(tmp_path) -> None:
    """Deux recensements identiques ; cellules, multiplicités et filtre exacts."""
    p = tmp_path / "mini_aba.txt"
    p.write_text(_LINE + "\n" + _LINE + "\n<EOS>\n", encoding="utf-8")
    c1 = census(str(p), label="fixture", n_cycles=10)
    c2 = census(str(p), label="fixture", n_cycles=10)
    assert c1 == c2
    assert c1.n_used == 2 and c1.n_filtered_out == 0
    assert c1.n_mono == 2 and c1.n_non_canonical == 0
    assert c1.mono_cells == ((9, 5, 2),)            # |A|+|B| = 5, N = 9, ×2
    assert c1.off_grid_cells == ()                  # (9, 5) est en grille
    assert all(c1.offset_cells_present[lbl] == () for lbl in ETA_LABELS_RUNNERS)


def test_frontier_alerts_traced_on_witness_cell() -> None:
    """(7,4) : η = 1 est un nœud-zéro (plateau zéro T28/T29 — Δ(7,4;1) = 0 exact) ;
    aucun autre η gelé n'alerte sur cette cellule ; hors-grille jamais croisé."""
    alerts = frontier_alerts([(7, 4), (25, 12)])    # (25,12) hors grille : ignoré
    assert [(a.n, a.k, a.eta_label, a.kind) for a in alerts] == [
        (7, 4, "1", "nœud-zéro")]


# =============================================================================
# BYTE-IDENTITÉ du chemin par défaut (invariant de comparabilité T24→T30)
# =============================================================================

def test_eta_kwarg_default_is_frozen_constant() -> None:
    """Le DÉFAUT du kwarg ``eta`` EST ``ETA_STRUCT`` (jamais une copie divergente)."""
    sig = inspect.signature(run_structural_regulation)
    assert sig.parameters["eta"].default == ETA_STRUCT == 0.5


@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_default_path_byte_identical_t24() -> None:
    """Runner SANS kwarg == runner avec ``eta=ETA_STRUCT`` : rapports STRICTEMENT
    égaux sur le run T24 (mêmes suites d'opérations flottantes, portes comprises)."""
    assert default_path_byte_identical(str(_DATASET), n_cycles=40)


# =============================================================================
# PORTE A gravée — recensement a priori des trois corpus (mesuré 2026-07-12)
# =============================================================================

@pytest.mark.skipif(not _HAVE_CORPORA, reason="corpus réels indisponibles")
def test_gateA_census_frozen() -> None:
    """Recensement GELÉ AVANT tout run η (gravé) : 2116/2116 mono-flip, 0 cycle
    HORS-CARTE, 0 hors-grille, 0 rejeté par le filtre. Cellules distinctes :
    10 (T24) / 19 (T25) / 14 (T26). UNE seule cellule à offset présente dans le
    réel : (16,12) — T25-claude, multiplicité 1, pré-déclarée à η = 4 (−1/16).
    Les η-invariantes (9,7)/(15,11) sont ABSENTES du réel : la réserve
    d'instrument T29 sur (9,7) est un artefact de grille sans présence réelle."""
    runs = {label: (path, n, lr) for label, path, n, lr in corpus_runs()}
    cs = {label: census(path, label=label, n_cycles=n, line_range=lr)
          for label, (path, n, lr) in runs.items()}
    for c in cs.values():
        assert c.n_filtered_out == 0
        assert c.n_non_canonical == 0 and c.non_canonical == ()
        assert c.off_grid_cells == ()
        assert c.n_mono == c.n_used
    assert sum(c.n_mono for c in cs.values()) == EXPECTED_CONTROL_TOTAL
    assert cs["T24-defaut"].mono_cells == (
        (6, 4, 7), (7, 4, 10), (8, 5, 5), (9, 6, 5), (10, 6, 1), (11, 7, 2),
        (13, 8, 3), (14, 9, 3), (15, 10, 3), (17, 11, 1))
    assert cs["T25-claude"].mono_cells == (
        (11, 8, 1), (12, 7, 4), (13, 8, 5), (13, 9, 3), (14, 8, 4), (14, 9, 3),
        (14, 10, 4), (15, 8, 1), (15, 9, 6), (15, 10, 16), (16, 10, 16),
        (16, 11, 5), (16, 12, 1), (17, 10, 1), (17, 11, 2), (18, 12, 1),
        (19, 12, 1), (20, 12, 1), (21, 14, 1))
    assert cs["T26-bloc26"].mono_cells == (
        (6, 4, 257), (7, 4, 503), (8, 5, 376), (9, 6, 310), (10, 6, 207),
        (11, 7, 110), (12, 8, 79), (13, 8, 41), (14, 9, 16), (15, 10, 93),
        (16, 10, 5), (17, 11, 1), (18, 12, 1), (19, 12, 1))
    # cellules à offset PRÉSENTES dans le réel : (16,12) à η=4 chez claude, RIEN d'autre
    for label, c in cs.items():
        for lbl in ETA_LABELS_RUNNERS:
            expected = ((16, 12),) if (label, lbl) == ("T25-claude", "4") else ()
            assert c.offset_cells_present[lbl] == expected


# =============================================================================
# PORTE C gravée — contrôle dur η = 1/2 : 2116/2116 signe ET valeur
# =============================================================================

@pytest.fixture(scope="module")
def control_reports():
    """Les 3 rapports η = 1/2 (contrôle dur), calculés UNE fois (déterministes)."""
    if not _HAVE_CORPORA:
        pytest.skip("corpus réels indisponibles")
    return {label: eta_corpus_report(path, label=label,
                                     eta_label=T30_CONTROL_LABEL,
                                     n_cycles=n, line_range=lr)
            for label, path, n, lr in corpus_runs()}


@pytest.mark.skipif(not _HAVE_CORPORA, reason="corpus réels indisponibles")
def test_gateC_hard_control_eta_half_2116(control_reports) -> None:
    """CONTRÔLE DUR (gravé) : les 2116 cycles T27 reproduits en SIGNE et en
    VALEUR (≤ 1e-12) à η = 1/2 ; best_fixed = 1.0 sur les trois runs ; aucune
    cellule à offset rencontrée à cet η (ni (9,7) ni (15,11) au réel)."""
    totals = {"n": 0, "sign": 0, "value": 0}
    for label, r in control_reports.items():
        assert r.best_fixed == 1.0
        assert r.n_hors_carte == 0
        assert r.offset_hits == () and r.mismatches == ()
        assert r.all_pass
        totals["n"] += r.n_mono
        totals["sign"] += r.n_sign_ok
        totals["value"] += r.n_value_ok
    assert totals == {"n": 2116, "sign": 2116, "value": 2116}
    assert control_reports["T24-defaut"].n_mono == 40
    assert control_reports["T25-claude"].n_mono == 76
    assert control_reports["T26-bloc26"].n_mono == 2000


# =============================================================================
# PORTE D gravée — mesure fraîche 4 η × 3 corpus (mesuré 2026-07-12)
# =============================================================================

@pytest.fixture(scope="module")
def fresh_reports():
    """Les 12 rapports frais (4 η × 3 corpus), calculés UNE fois (déterministes)."""
    if not _HAVE_CORPORA:
        pytest.skip("corpus réels indisponibles")
    return {(lbl, label): eta_corpus_report(path, label=label, eta_label=lbl,
                                            n_cycles=n, line_range=lr)
            for lbl in ETA_FRESH_LABELS
            for label, path, n, lr in corpus_runs()}


@pytest.mark.skipif(not _HAVE_CORPORA, reason="corpus réels indisponibles")
def test_gateD_fresh_etas_all_pass(fresh_reports) -> None:
    """P-η MESURÉE (gravé) : 4 η frais × 3 corpus, signes 100 % hors
    pré-déclarées et valeurs ≤ 1e-12 sur TOUS les mono-flip — 8464 comparaisons
    de valeur (2116 × 4), zéro écart. La loi de pureté étendue TIENT aux 4
    régimes neufs (dont η = 4, clip-dominé, P-gate violée sur grille : les
    cellules (8,5)/(11,7) réelles y suivent la carte, pas la P-gate)."""
    for (lbl, label), r in fresh_reports.items():
        assert r.best_fixed == 1.0, (lbl, label)      # η-invariant (précondition B)
        assert r.n_hors_carte == 0
        assert r.n_value_ok == r.n_value_total == r.n_mono
        assert r.n_sign_ok == r.n_sign_total
        assert r.mismatches == ()
        assert r.all_pass
    n_value_checks = sum(r.n_value_total for r in fresh_reports.values())
    assert n_value_checks == 4 * EXPECTED_CONTROL_TOTAL == 8464


@pytest.mark.skipif(not _HAVE_CORPORA, reason="corpus réels indisponibles")
def test_gateD_offset_cell_16_12_measured_at_eta4(fresh_reports) -> None:
    """LE point diagnostique du tour (gravé) : la SEULE cellule à offset présente
    au réel — (16,12), cycle 72 de corpus_claude — est rencontrée à η = 4 et
    l'OFFSET FLOAT MESURÉ == l'offset PRÉDIT par les gravures T28 : −1/16
    exactement (Δ_exact = 9/16, Δ_real = 1/2). Le défaut d'instrument prédit
    sur grille synthétique est observé AU RÉEL, à la valeur près — la
    compréhension T29 du bord-exact float est complète sur ce point."""
    hits = {(lbl, label): r.offset_hits for (lbl, label), r in fresh_reports.items()}
    # un seul hit, à (η=4, claude) ; aucun ailleurs
    for key, h in hits.items():
        if key == ("4", "T25-claude"):
            continue
        assert h == (), key
    (hit,) = hits[("4", "T25-claude")]
    assert (hit.index, hit.n, hit.k) == (72, 16, 12)
    assert hit.offset_predicted == float(Fraction(-1, 16)) == -0.0625
    assert hit.offset_measured == pytest.approx(-0.0625, abs=1e-12)
    assert hit.ok
    # la valeur réelle est bien float_map == exact + offset (identité de chemin)
    assert delta_nom_exact_eta(16, 12, Fraction(4)) == Fraction(9, 16)
    assert delta_nom_float_eta(16, 12, 4.0) == pytest.approx(0.5, abs=1e-12)
