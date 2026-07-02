"""Tests Tour 24 — ``gap_struct`` structurel + régulation par l'organe ``regulate_step``.

H24 : l'organe (INCHANGÉ, importé) régule-t-il un observable STRUCTUREL
order-preserving (écart de phase du flip DX/OUT→LV/IN) mieux qu'un rythme fixe et
mieux que le shuffle ? Deux familles :

  * SANS dataset (toujours exécutées) : quatuor adapté (déterminisme, finitude,
    formes, formule EXACTE sur profil connu k=2/N=4), pivots porte 1 (η=0 exact
    bit-à-bit sur deux chemins de code ; profil sans flip ⇒ Δ=0), pré-validation
    porte 0 sur synthétique (primaire order-sensible, multiset vacuous — prédiction
    falsifiable de Fable 5), shuffle déterministe avec tag voyageant avec le token,
    organe importé (jamais copié).

  * AVEC ``dataset_aba.txt`` (skip propre si absent — AUCUN ``.so`` requis, le
    profil est structurel) : collecte déterministe, rapport complet déterministe et
    fini, et le VERDICT MESURÉ documenté (jamais forcé, modèle T21).

Le canon ``core/`` et ``regulate_step`` ne sont JAMAIS touchés ; tout est seedé.
"""
import math
from pathlib import Path

import pytest

from spiraton.data.aba import parse_aba_line
from spiraton.experimental.edge_controller import regulate_step
from spiraton.experimental import structural_gap
from spiraton.experimental.structural_gap import (
    BAND_LEAD,
    ETA_STRUCT,
    G0_STRUCT,
    G_MAX_STRUCT,
    G_MIN_STRUCT,
    OrientedToken,
    PHI_STAR,
    STRUCT_GAIN_SWEEP,
    TARGET_LEAD,
    f_edge_struct,
    flip_fraction,
    obs_struct_frozen,
    obs_struct_multiset,
    orientation_profile,
    phi_ref_increments,
    reconstruct_fixed,
    reconstruct_profile,
    shuffle_orientations,
    shuffle_tokens,
)
from spiraton.diagnostics.instrument_validation import assert_order_sensitive
from spiraton.diagnostics.structural_regulation import (
    StructuralRegulationReport,
    best_fixed_gain,
    collect_profiles,
    pivot_eta0_is_exact,
    pivot_noflip_delta,
    run_structural_regulation,
)


_DATASET = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
if not _DATASET.is_file():
    _DATASET = Path(__file__).resolve().parents[2] / "dataset_aba.txt"


_LINE = (
    "<SEG_A> <MUL><DX><OUT><ALPHA> un deux trois </SEG_A> "
    "<SEG_B> <MUL><DX><OUT><OMEGA> quatre cinq </SEG_B> "
    "<SEG_A_PRIME> <MUL><LV><IN><A_PRIME> six sept huit neuf.<EOL> </SEG_A_PRIME> <EOL>"
)


def _toks(orientations):
    """Profil synthétique : textes distincts (t0, t1, …) pour tracer les permutations."""
    return [OrientedToken(text=f"t{i}", orientation=o) for i, o in enumerate(orientations)]


# =============================================================================
# Profil d'orientation (source : parseur aba.py, jamais les dims 0-5)
# =============================================================================

def test_orientation_profile_from_cycle() -> None:
    """Marche à un seul flip : k = |A|+|B| tokens à +1 puis |A′| à −1, ordre préservé."""
    cycle = parse_aba_line(_LINE)
    prof = orientation_profile(cycle)
    orients = [tk.orientation for tk in prof]
    assert orients == [+1] * 5 + [-1] * 4      # |A|=3, |B|=2, |A′|=4
    assert [tk.text for tk in prof[:3]] == ["un", "deux", "trois"]
    assert flip_fraction(orients) == pytest.approx(5.0 / 9.0)


def test_orientation_profile_rejects_non_canonical_header() -> None:
    """DX/IN (hors canon) ⇒ ValueError : le profil est un objet de la grammaire canonique."""
    line = _LINE.replace("<MUL><LV><IN><A_PRIME>", "<MUL><DX><IN><A_PRIME>")
    cycle = parse_aba_line(line)
    with pytest.raises(ValueError):
        orientation_profile(cycle)


def test_phi_ref_increments_sum_to_n_and_degenerate_uniform() -> None:
    """Les incréments somment à N ; profil sans flip ⇒ incrément uniforme 1.0."""
    orients = [+1] * 6 + [-1] * 3
    incs = phi_ref_increments(orients)
    assert sum(incs) == pytest.approx(len(orients))
    assert incs[0] == pytest.approx(PHI_STAR * 9 / 6)
    assert incs[-1] == pytest.approx((1 - PHI_STAR) * 9 / 3)
    assert phi_ref_increments([+1] * 5) == [1.0] * 5
    assert phi_ref_increments([-1] * 5) == [1.0] * 5


# =============================================================================
# Quatuor adapté : déterminisme, finitude, formes, formule exacte
# =============================================================================

def test_reconstruction_deterministic() -> None:
    """Deux déroulés identiques bit-à-bit (aucune source aléatoire)."""
    orients = [+1] * 7 + [-1] * 5
    t1 = reconstruct_profile(orients)
    t2 = reconstruct_profile(orients)
    assert t1.e == t2.e and t1.g == t2.g and t1.o_hat == t2.o_hat


def test_reconstruction_finite() -> None:
    """Aucun NaN/Inf sur profils synthétiques variés (organe et fixe)."""
    for orients in ([+1] * 9 + [-1] * 3, [+1, -1] * 6, [+1] * 2 + [-1] * 10):
        for tr in (
            reconstruct_profile(orients),
            reconstruct_fixed(orients, g_fixed=1.25),
        ):
            for series in (tr.e, tr.p_read, tr.p_ref, tr.g, tr.gap_binary):
                assert all(math.isfinite(v) for v in series)


def test_reconstruction_shapes() -> None:
    """Formes : p_read/p_ref/e de longueur N+1 ; g/o_hat/gap_binary de longueur N."""
    orients = [+1] * 4 + [-1] * 4
    tr = reconstruct_profile(orients)
    n = len(orients)
    assert len(tr.p_read) == len(tr.p_ref) == len(tr.e) == n + 1
    assert len(tr.g) == len(tr.o_hat) == len(tr.gap_binary) == n
    with pytest.raises(ValueError):
        reconstruct_profile([])
    with pytest.raises(ValueError):
        reconstruct_profile([+1, -1], g0=G_MAX_STRUCT + 1.0)


def test_exact_formula_k2_n4_frozen() -> None:
    """Formule EXACTE (profil connu k=2, N=4, lecteur figé g=1) — analytique à la main.

    inc(+1) = (2/3)·4/2 = 4/3 ; inc(−1) = (1/3)·4/2 = 2/3.
    p_ref  = [0, 4/3, 8/3, 10/3, 4] ; p_read = [1, 2, 3, 4, 5]
    e      = [1, 2/3, 1/3, 2/3, 1]  (offset initial = target = 1 token)
    ô (p_read < φ*·N = 8/3) = [+1, +1, −1, −1] = o ⇒ gap_binary ≡ 0
    f_edge (|e_t − 1| ≤ 0.5, t=1..4) = 3/4 (seul e_2 = 1/3 sort de la bande).
    """
    orients = [+1, +1, -1, -1]
    tr = reconstruct_fixed(orients, g_fixed=1.0)
    assert tr.p_read == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert tr.p_ref == pytest.approx([0.0, 4.0 / 3.0, 8.0 / 3.0, 10.0 / 3.0, 4.0])
    assert tr.e == pytest.approx([1.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert tr.o_hat == [+1, +1, -1, -1]
    assert tr.gap_binary == [0.0, 0.0, 0.0, 0.0]
    assert f_edge_struct(tr) == pytest.approx(0.75)


# =============================================================================
# PORTE 1 — pivots (η=0 exact ; profil sans flip ⇒ Δ=0)
# =============================================================================

def test_pivot_eta_zero_exact_two_code_paths() -> None:
    """``η=0`` ≡ g-fixe : égalité float EXACTE (==) entre les DEUX chemins de code."""
    orients = [+1] * 8 + [-1] * 5
    for g0 in (0.8, 1.0, 1.25):
        tr0 = reconstruct_profile(orients, eta=0.0, g0=g0)
        trf = reconstruct_fixed(orients, g_fixed=g0)
        assert tr0.e == trf.e                    # égalité float stricte, pas approx
        assert tr0.p_read == trf.p_read
        assert tr0.g == trf.g
        assert tr0.o_hat == trf.o_hat
    assert pivot_eta0_is_exact(orients)


def test_pivot_noflip_delta_zero_exact() -> None:
    """Profil SANS flip ⇒ Δ = 0 EXACT : l'organe est inerte (e ≡ target) et le
    nominal g=1 est déjà parfait (f_edge = 1) — rien à réguler."""
    orients = [+1] * 12
    tr_organ = reconstruct_profile(orients, eta=ETA_STRUCT, g0=G0_STRUCT)
    tr_fixed = reconstruct_fixed(orients, g_fixed=G0_STRUCT)
    # arithmétique exacte en unités token (incréments 1.0) ⇒ traces bit-à-bit
    assert tr_organ.e == tr_fixed.e
    assert tr_organ.g == [G0_STRUCT] * 12       # l'organe n'a jamais corrigé
    assert f_edge_struct(tr_organ) == 1.0
    assert pivot_noflip_delta() == 0.0


def test_regulate_step_is_the_imported_organ() -> None:
    """L'organe est IMPORTÉ (jamais copié) et sa loi T15 est intacte."""
    assert structural_gap.regulate_step is regulate_step
    assert regulate_step(1.0, 1.2, 1.0, 0.5, 0.5, 2.0) == 1.0 - 0.5 * (1.2 - 1.0)
    assert regulate_step(0.93, 5.0, 1.0, 0.0, 0.5, 2.0) == 0.93   # η=0 ⇒ inerte


def test_feedback_sign_e_increasing_in_g() -> None:
    """Condition de monotonie T20 (a priori) : e est CROISSANT en g (de/dg = +1/token)."""
    orients = [+1] * 6 + [-1] * 3
    e_lo = reconstruct_fixed(orients, g_fixed=0.9).e[-1]
    e_hi = reconstruct_fixed(orients, g_fixed=1.1).e[-1]
    assert e_hi > e_lo
    # pente exacte : Δe_N = N·Δg
    assert (e_hi - e_lo) == pytest.approx(len(orients) * 0.2)


# =============================================================================
# PORTE 0 — pré-validation sur synthétique (primaire sensible, multiset vacuous)
# =============================================================================

def test_primary_obs_is_order_sensitive_on_synthetic() -> None:
    """Profil k/N = 0.25 (loin de φ*=2/3) : le primaire BOUGE sous shuffle (gap ≫ δ_min)."""
    toks = _toks([+1] * 4 + [-1] * 12)
    rep = assert_order_sensitive(obs_struct_frozen, toks, shuffle_fn=shuffle_tokens)
    assert rep.is_order_sensitive
    assert rep.gap >= 1e-2      # vrai signal d'ordre, ordres de grandeur au-dessus de δ_min


def test_multiset_obs_is_vacuous_by_construction() -> None:
    """La variante-contrôle multiset renvoie gap = 0 EXACT sous shuffle (prédiction Fable 5)."""
    toks = _toks([+1] * 4 + [-1] * 12)
    rep = assert_order_sensitive(obs_struct_multiset, toks, shuffle_fn=shuffle_tokens)
    assert rep.is_vacuous
    assert rep.gap == 0.0


def test_primary_obs_order_invariant_at_exact_phi_star() -> None:
    """Cas limite DOCUMENTÉ : à k/N = φ* exactement, les incréments s'égalisent et le
    primaire devient order-invariant POUR CE PROFIL (gap ≈ 0). C'est une propriété de
    l'instrument à connaître (rapportée, pas cachée) — la porte 0 du run réel doit
    donc être lue avec le k/N du profil testé."""
    toks = _toks([+1] * 8 + [-1] * 4)           # k/N = 8/12 = 2/3 = φ*
    rep = assert_order_sensitive(obs_struct_frozen, toks, shuffle_fn=shuffle_tokens)
    assert rep.is_vacuous


def test_shuffle_tokens_deterministic_tag_travels() -> None:
    """Shuffle seedé reproductible ; permutation vraie ; le tag VOYAGE avec son token."""
    toks = _toks([+1] * 5 + [-1] * 3)
    s1 = shuffle_tokens(toks, 42)
    s2 = shuffle_tokens(toks, 42)
    assert s1 == s2
    assert sorted(tk.text for tk in s1) == sorted(tk.text for tk in toks)
    # appariement texte↔tag intact après permutation
    orig = {tk.text: tk.orientation for tk in toks}
    assert all(orig[tk.text] == tk.orientation for tk in s1)
    # orientations nues : même mécanique
    o1 = shuffle_orientations([+1] * 5 + [-1] * 3, 42)
    assert o1 == [tk.orientation for tk in s1]


# =============================================================================
# best_fixed et collecte (sans dataset : fixture tmp_path)
# =============================================================================

def test_best_fixed_gain_is_argmax_of_median_f_edge() -> None:
    """``best_fixed_gain`` est bien l'argmax du f_edge MÉDIAN sur la grille a priori."""
    def med(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    lists = [[+1] * 6 + [-1] * 6, [+1] * 7 + [-1] * 7, [+1] * 9 + [-1] * 3]
    g = best_fixed_gain(lists)
    assert g in STRUCT_GAIN_SWEEP
    med_g = med([f_edge_struct(reconstruct_fixed(o, g_fixed=g)) for o in lists])
    for other in STRUCT_GAIN_SWEEP:
        med_other = med([f_edge_struct(reconstruct_fixed(o, g_fixed=other)) for o in lists])
        assert med_other <= med_g + 1e-12


def test_collect_profiles_from_fixture(tmp_path) -> None:
    """Collecte déterministe : ordre du fichier, filtre N ≥ 4 et deux orientations."""
    good = _LINE
    tiny = (
        "<SEG_A> <ADD><DX><OUT><ALPHA> a </SEG_A> "
        "<SEG_B> <ADD><DX><OUT><OMEGA> b </SEG_B> "
        "<SEG_A_PRIME> <ADD><LV><IN><A_PRIME> c<EOL> </SEG_A_PRIME> <EOL>"
    )  # N = 3 < MIN_TOKENS : filtré
    p = tmp_path / "mini_aba.txt"
    p.write_text(tiny + "\n" + good + "\n" + good + "\n<EOS>\n", encoding="utf-8")
    profs = collect_profiles(str(p), n_cycles=10)
    assert len(profs) == 2
    assert all(len(pr) == 9 for pr in profs)
    assert [tk.orientation for tk in profs[0]] == [+1] * 5 + [-1] * 4


# =============================================================================
# Rapport complet sur le RÉEL (skip propre si dataset absent — aucun .so requis)
# =============================================================================

@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_run_structural_regulation_deterministic_and_finite() -> None:
    """Le rapport complet est déterministe bit-à-bit et toutes les médianes sont finies."""
    r1 = run_structural_regulation(str(_DATASET), n_cycles=40)
    r2 = run_structural_regulation(str(_DATASET), n_cycles=40)
    assert isinstance(r1, StructuralRegulationReport)
    assert r1.n_cycles == 40
    # déterminisme bit-à-bit des quantités décisionnelles
    assert r1.order_primary.gap == r2.order_primary.gap
    assert r1.delta_real_median == r2.delta_real_median
    assert r1.delta_shuffle_median == r2.delta_shuffle_median
    assert r1.verdict == r2.verdict
    # finitude
    for v in (
        r1.order_primary.gap, r1.order_multiset.gap, r1.delta_real_median,
        r1.wilcoxon_p_real, r1.delta_shuffle_median, r1.real_minus_shuffle,
        r1.e_final_ctrl_median, r1.e_final_fixed_median,
    ):
        assert math.isfinite(v)
    # pivots (porte 1) tiennent sur le réel
    assert r1.pivot_eta0_exact
    assert r1.pivot_noflip == 0.0
    # la variante-contrôle multiset est vacuous (prédiction Fable 5 — valide la porte)
    assert r1.order_multiset.is_vacuous
    assert r1.order_multiset.gap == 0.0


@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_measured_verdict_gates_documented() -> None:
    """Verdict MESURÉ (jamais forcé, modèle T21) : porte 0 PASSE, porte 2 = MORTE.

    Mesuré le 2026-07-02 (robuste n_cycles ∈ {40, 80, 120}) : l'instrument est
    VALIDE (gap primaire 2.19e-1 ≫ δ_min=1e-4 ; multiset vacuous, gap=0.0 — la
    prédiction Fable 5 tient), les pivots tiennent, MAIS ``Δf_edge`` médian réel
    = +0.0000 < 0.05 ⇒ MORTE : la position du flip ``k/N`` du corpus est
    quasi-constante (σ ≈ 0.038, médiane 0.636 ≈ φ* = 2/3) et les cycles courts
    (méd. 8 tokens) ⇒ le rythme fixe nominal sature déjà la bande (f_edge = 1.0
    médian) — RIEN à réguler. C'est le mode d'échec pré-déclaré §6 de l'émission
    (le plus informatif : il BORNE la généricité T19/T20 sur CE corpus). On grave
    ici les quantités décisionnelles pour détecter toute dérive future.

    ANOMALIE GRAVÉE (rapportée, jamais cachée — REFUS) : le Wilcoxon apparié est
    significatif EN DÉFAVEUR de l'organe (n=40 : p = 5.89e-3, signes +4/−10 sur
    les 14 cycles où organe et fixe diffèrent ; persiste et s'amplifie à n=80
    (+4/−23, p=3.46e-6) et n=120 (+13/−25, p=2.69e-5)). Cause mesurée :
    sur-correction proportionnelle sur horizons très courts — quand la baseline
    ne sature pas, l'organe dégrade plus souvent qu'il ne répare. Le Δ médian
    reste +0.0000 (26/40 ex æquo), donc le verdict MORTE tient, mais la
    DIRECTION du différentiel est adverse : toute reprise de H24 sur un corpus
    à k/N varié devra re-mesurer ce point, pas seulement la médiane.
    """
    r = run_structural_regulation(str(_DATASET), n_cycles=40)
    assert r.gate0_primary_sensitive           # porte 0 : instrument VALIDE
    assert r.order_primary.gap > 1e-1          # signal d'ordre net (~2.2e-1)
    assert r.gate0_multiset_vacuous            # la porte est saine (Fable 5)
    assert r.delta_real_median == 0.0          # porte 2 : aucun avantage médian
    assert r.verdict == "MORTE"
    # anomalie Wilcoxon-négatif gravée (déterministe : ordre du fichier, shuffles seedés)
    assert r.sign_pos_real == 4
    assert r.sign_neg_real == 10
    assert r.sign_neg_real > r.sign_pos_real   # direction ADVERSE : à re-mesurer si reprise
    assert r.wilcoxon_p_real == pytest.approx(5.892e-3, rel=1e-3)
