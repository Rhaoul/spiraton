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
from spiraton.diagnostics.aba_regulation import THRESHOLD_ACTIVE, THRESHOLD_STRUCTURE
from spiraton.diagnostics.instrument_validation import assert_order_sensitive
from spiraton.diagnostics.memory_inhibition_scan import spearman_rho, spearman_t_pvalue
from spiraton.diagnostics.structural_regulation import (
    BLOCK26_LINES,
    EXCURSION_THRESHOLD,
    LONG_MIN_TOKENS,
    MIN_NONZERO_STRATUM,
    N_CYCLES_BLOCK26,
    N_CYCLES_CLAUDE,
    SHORT_MAX_TOKENS,
    SIGMA_KN_MATERIAL,
    SIGMA_KN_REF_T24,
    ExcursionStrata,
    LengthStrata,
    PopulationDescriptor,
    StructuralRegulationReport,
    best_fixed_gain,
    collect_profiles,
    excursion,
    excursion_strata,
    length_strata,
    pivot_eta0_is_exact,
    pivot_noflip_delta,
    population_descriptor,
    run_structural_regulation,
)


_DATASET = Path("F:/code/claude/spiraton-enhanced/dataset_aba.txt")
if not _DATASET.is_file():
    _DATASET = Path(__file__).resolve().parents[2] / "dataset_aba.txt"

_CORPUS_CLAUDE = Path("F:/code/claude/spiraton-enhanced/corpus_claude_aba.txt")
if not _CORPUS_CLAUDE.is_file():
    _CORPUS_CLAUDE = Path(__file__).resolve().parents[2] / "corpus_claude_aba.txt"


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


# =============================================================================
# TOUR 25 — descripteur de population + strate longueur (lecture partitionnée)
# =============================================================================

def test_length_strata_partitioned_reading_synthetic() -> None:
    """Partition des MÊMES Δ par longueur : bornes pré-déclarées, strate vide → NaN."""
    deltas = [0.1, -0.2, 0.3, 0.0, 0.5]
    tokens = [8, 9, 12, 15, 10]              # 2 courts (<10), 2 longs (≥12), 1 hors strates
    st = length_strata(deltas, tokens)
    assert isinstance(st, LengthStrata)
    assert st.n_short == 2
    assert st.delta_short_median == pytest.approx(-0.05)   # médiane de {0.1, −0.2}
    assert st.n_long == 2
    assert st.delta_long_median == pytest.approx(0.15)     # médiane de {0.3, 0.0}
    # strate vide rapportée telle quelle (effectif 0, médiane NaN — jamais masquée)
    st_empty = length_strata([0.2, 0.4], [12, 15])
    assert st_empty.n_short == 0
    assert math.isnan(st_empty.delta_short_median)
    # séquences non alignées ⇒ erreur, pas silence
    with pytest.raises(ValueError):
        length_strata([0.1], [8, 9])
    # bornes gelées a priori (émission T25 §2a)
    assert SHORT_MAX_TOKENS == 10 and LONG_MIN_TOKENS == 12


def test_population_descriptor_from_fixture(tmp_path) -> None:
    """Descripteur sur fixture : mêmes cycles que collect_profiles, σ échantillon,
    comptage des profils à k/N = φ* EXACT (cas limite d'instrument T24)."""
    # _LINE : N = 9, k/N = 5/9 ; cycle équilibré 2+2+2 : N = 6, k/N = 4/6 = 2/3 EXACT
    balanced = (
        "<SEG_A> <ADD><DX><OUT><ALPHA> a b </SEG_A> "
        "<SEG_B> <ADD><DX><OUT><OMEGA> c d </SEG_B> "
        "<SEG_A_PRIME> <ADD><LV><IN><A_PRIME> e f<EOL> </SEG_A_PRIME> <EOL>"
    )
    p = tmp_path / "mini_aba.txt"
    p.write_text(_LINE + "\n" + balanced + "\n<EOS>\n", encoding="utf-8")
    d = population_descriptor(str(p), n_cycles=10)
    assert isinstance(d, PopulationDescriptor)
    assert d.n_cycles == 2
    assert (d.tokens_min, d.tokens_max) == (6, 9)
    assert d.kn_min == pytest.approx(5.0 / 9.0)
    assert d.kn_max == pytest.approx(2.0 / 3.0)
    assert d.n_at_phi_star_exact == 1
    # σ échantillon (ddof=1) de {5/9, 2/3} = |2/3 − 5/9|/√2
    assert d.kn_sigma == pytest.approx(abs(2.0 / 3.0 - 5.0 / 9.0) / math.sqrt(2.0))
    assert d.variance_material is (d.kn_sigma >= SIGMA_KN_MATERIAL)
    # corpus sans cycle utilisable ⇒ erreur explicite, pas descripteur vide
    empty = tmp_path / "vide.txt"
    empty.write_text("<EOS>\n", encoding="utf-8")
    with pytest.raises(ValueError):
        population_descriptor(str(empty))


@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_sigma_definition_anchored_on_t24_reference() -> None:
    """La définition gelée de σ (échantillon, ddof=1) REPRODUIT la référence T24 :
    σ(k/N) = 0.038 sur les 40 cycles de dataset_aba.txt — ancre de comparabilité."""
    d = population_descriptor(str(_DATASET), n_cycles=40)
    assert d.kn_sigma == pytest.approx(SIGMA_KN_REF_T24, abs=5e-4)   # 0.0382 ≈ 0.038
    assert not d.variance_material
    assert SIGMA_KN_MATERIAL == pytest.approx(2 * SIGMA_KN_REF_T24)  # = 0.076, gelé


@pytest.mark.skipif(not _CORPUS_CLAUDE.is_file(), reason="corpus_claude_aba.txt indisponible")
def test_population_descriptor_claude_measured() -> None:
    """Descripteur MESURÉ de corpus_claude_aba.txt (gravé, jamais forcé — T25 §2a).

    Mesuré le 2026-07-02 : 76 cycles utilisables (tous), cycles nettement plus
    LONGS que dataset_aba (min 11 / méd 15 / max 21 vs 6/8/17) mais k/N à peine
    plus dispersé (σ = 0.0429 vs 0.038). Le critère gelé σ ≥ 0.076 ÉCHOUE ⇒
    P-a (variance) NON testable sur ce corpus, seule P-b (horizon) est en jeu —
    exactement le pronostic du linguiste (§1 : « c'est la LONGUEUR qui diffère,
    pas la position relative du flip »). 18 des 76 profils sont à k/N = φ* EXACT
    (cas limite d'instrument T24 : Δ = 0 par construction pour ces cycles).
    """
    d = population_descriptor(str(_CORPUS_CLAUDE), n_cycles=N_CYCLES_CLAUDE)
    assert d.n_cycles == 76
    assert (d.tokens_min, d.tokens_median, d.tokens_max) == (11, 15.0, 21)
    assert d.kn_min == pytest.approx(0.5333, abs=1e-4)
    assert d.kn_median == pytest.approx(0.6283, abs=1e-4)
    assert d.kn_max == pytest.approx(0.7500, abs=1e-4)
    assert d.kn_sigma == pytest.approx(0.0429, abs=1e-4)
    assert not d.variance_material            # critère σ ≥ 0.076 : FAIL ⇒ seule P-b en jeu
    assert d.n_at_phi_star_exact == 18


@pytest.mark.skipif(not _CORPUS_CLAUDE.is_file(), reason="corpus_claude_aba.txt indisponible")
def test_measured_verdict_corpus_claude_documented() -> None:
    """Verdict T25 MESURÉ sur corpus_claude_aba.txt (jamais forcé) : ACTIVE-structurelle.

    Mesuré le 2026-07-02, instrument ``structural_gap.py`` GELÉ byte-à-byte (git
    diff vide), seuils/portes STRICTEMENT identiques T24. Cellule du tableau des
    issues (émission §2c) : **σ < 0.076 × Δ ACTIVE** (« Surprise : l'avantage ne
    vient PAS de la variance k/N — à disséquer avant toute célébration »).

    PORTE 0 (re-jouée sur CE corpus) : primaire gap = 3.357e-1 ≫ δ_min ; multiset
    vacuous gap = 0.0. PORTE 1 : pivots exacts. PORTE 2 : Δf_edge médian = +0.2308
    > 0.15 (ACTIVE), signes +45/−0/76 (l'organe n'est JAMAIS pire — l'anomalie
    adverse T24 (+4/−10) ne se reproduit pas sur ces horizons), Wilcoxon
    p = 4.29e-9. PORTE 3 : Δ shuffle = +0.0000, réel − shuffle = +0.2308 ≥ 0.05
    ⇒ l'avantage est porté par la STRUCTURE A→B→A′, pas par la dynamique générique.

    DISSECTION (prudence artefact exigée par la cellule, non-décisionnelle,
    rapportée) : verdict INVARIANT sous 3 bases de seed de shuffle (70000/80000/
    91000) et sous-échantillon n = 40 (+25/−0). Les 18 cycles à k/N = φ* exact ont
    Δ = 0 exact (rien à réguler par construction). Hors eux (n = 58) :
    Spearman(Δ, |k/N−φ*|) = +0.854 et Spearman(Δ, N·|k/N−φ*|) = +0.875 mais
    Spearman(Δ, N) = −0.026 ⇒ la variable opérante est l'EXCURSION INTRA-CYCLE
    N·|k/N−φ*| (dérive de phase non-uniforme, excursion méd 0.667 > bande 0.5 —
    la baseline fixe ne sature plus : 31/76 vs 26/40 au T24), PAS la variance
    inter-cycle σ(k/N) ni la longueur seule. Cohérent T16 : la dérive nette
    intra-cycle gouverne, pas la variance de population.

    STRATE LONGUEUR (pré-déclarée §2a) : strate courte VIDE (0 cycle < 10 tokens,
    min = 11) ⇒ P-b (sens négatif sur cycles courts) NON testable frontalement ici ;
    son corollaire (remontée du signe sur cycles longs) est confirmé : n = 75
    cycles ≥ 12, Δ méd = +0.2308, aucun signe négatif.

    ANOMALIES RAPPORTÉES : (1) sur l'observable binaire NON régulé (gap_binaire),
    l'organe est légèrement PIRE en médiane (0.0625 vs 0.0488) — il optimise la
    bande de phase e_t, pas le flip binaire ; (2) e_N organe méd = +0.847 ≠ target
    exactement (le fixe g=1 donne e_N = +1.0 par identité) — retour transformé,
    proche-aligné-non-identique.
    """
    r = run_structural_regulation(str(_CORPUS_CLAUDE), n_cycles=N_CYCLES_CLAUDE)
    assert r.n_cycles == 76
    # PORTE 0 re-jouée sur CE corpus : instrument VALIDE, porte saine
    assert r.gate0_primary_sensitive
    assert r.order_primary.gap == pytest.approx(0.335714, abs=1e-5)
    assert r.gate0_multiset_vacuous
    assert r.order_multiset.gap == 0.0
    # PORTE 1 : pivots tiennent sur le 1er profil réel (N=12, k/N=0.583 ≠ φ*)
    assert r.pivot_eta0_exact
    assert r.pivot_noflip == 0.0
    # PORTE 2 : ACTIVE, jamais adverse
    assert r.best_fixed == 1.0
    assert r.delta_real_median == pytest.approx(0.23077, abs=1e-4)
    assert r.delta_real_median > THRESHOLD_ACTIVE
    assert r.sign_pos_real == 45
    assert r.sign_neg_real == 0
    assert min(r.delta_real) >= 0.0            # aucun cycle où l'organe est pire
    assert r.wilcoxon_p_real == pytest.approx(4.287e-9, rel=1e-3)
    # PORTE 3 : l'avantage survit à la destruction de l'ordre
    assert r.delta_shuffle_median == pytest.approx(0.0, abs=1e-12)
    assert r.real_minus_shuffle == pytest.approx(0.23077, abs=1e-4)
    assert r.real_minus_shuffle >= THRESHOLD_STRUCTURE
    assert r.verdict == "ACTIVE-structurelle"
    # strate longueur : courte VIDE (fait de population), longue = tout l'effet
    st = length_strata(r.delta_real, r.tokens_per_cycle)
    assert st.n_short == 0
    assert math.isnan(st.delta_short_median)
    assert st.n_long == 75
    assert st.delta_long_median == pytest.approx(0.23077, abs=1e-4)
    # anomalie gap_binaire (observable NON régulé) rapportée, jamais cachée
    assert r.gap_binary_ctrl_median == pytest.approx(0.0625, abs=1e-4)
    assert r.gap_binary_fixed_median == pytest.approx(0.0488, abs=1e-4)
    assert r.gap_binary_ctrl_median > r.gap_binary_fixed_median


@pytest.mark.skipif(not _CORPUS_CLAUDE.is_file(), reason="corpus_claude_aba.txt indisponible")
def test_fairness_control_fine_grid_and_oracle_documented() -> None:
    """Contrôle d'ÉQUITÉ du sweep (réparation REFUS de l'ingénieur, intégration T25).

    Le Δf_edge médian = +0.2308 de la porte 2 est mesuré contre ``STRUCT_GAIN_SWEEP``
    (7 rythmes, GELÉ a priori au T24 — traitement symétrique T24↔T25 : la même
    grille a produit MORTE au T24). Ce test grave la BORNE d'équité mesurée à
    l'intégration, pour que la taille médiane ne soit JAMAIS sur-revendiquée :

      * contre une grille FINE (0.500..2.000, pas 0.005, 301 points, hors
        protocole gelé), le meilleur rythme fixe global est g = 1.020 et la
        MÉDIANE de Δ tombe à 0.0000 (les deux médianes saturent à f_edge = 1.0) ;
        l'avantage vit alors dans la QUEUE : organe strictement meilleur sur
        36/76, JAMAIS pire, Wilcoxon p = 1.50e-7, moyenne +0.152 ;
      * contre l'ORACLE par cycle (meilleur g fixe choisi PAR CYCLE sur la grille
        fine — borne supérieure inatteignable d'une baseline fixe), l'organe reste
        strictement meilleur sur 17/76 et JAMAIS pire (p = 2.88e-4) : un lecteur à
        rythme CONSTANT ne peut pas suivre deux régimes de phase intra-cycle
        (inc_plus ≠ inc_minus dès que k/N ≠ φ*) — l'organe si. C'est la dominance
        SANS artefact de grille ; la taille médiane +0.2308, elle, est
        grille-relative (l'organe sature f_edge = 1.0 sur 74/76).
    """
    r = run_structural_regulation(str(_CORPUS_CLAUDE), n_cycles=N_CYCLES_CLAUDE)
    profiles = collect_profiles(str(_CORPUS_CLAUDE), n_cycles=N_CYCLES_CLAUDE)
    ol = [[tk.orientation for tk in p] for p in profiles]
    from spiraton.diagnostics.edge_maintenance import wilcoxon_signed_rank

    # l'organe sature son propre score : borne mécanique sur toute médiane appariée
    assert sum(1 for f in r.ctrl_f_edge if f == 1.0) == 74

    fine = [0.5 + 0.005 * i for i in range(301)]           # hors protocole gelé (contrôle)
    best_fine = best_fixed_gain(ol, fixed_gains=fine)
    assert best_fine == pytest.approx(1.020, abs=1e-9)
    fe_fine = [f_edge_struct(reconstruct_fixed(o, g_fixed=best_fine)) for o in ol]
    d_fine = [c - f for c, f in zip(r.ctrl_f_edge, fe_fine)]
    med_fine = sorted(d_fine)[37]                          # ~médiane basse, ex æquo à 0
    assert med_fine == 0.0                                 # la médiane s'effondre à 0
    assert min(d_fine) >= 0.0                              # ... mais JAMAIS pire
    assert sum(1 for d in d_fine if d > 0) == 36           # queue positive stricte
    _, p_fine, n_eff = wilcoxon_signed_rank(d_fine)
    assert n_eff == 36 and p_fine == pytest.approx(1.501e-7, rel=1e-2)

    fe_oracle = [max(f_edge_struct(reconstruct_fixed(o, g_fixed=g)) for g in fine) for o in ol]
    d_oracle = [c - f for c, f in zip(r.ctrl_f_edge, fe_oracle)]
    assert min(d_oracle) >= 0.0                            # jamais pire, même vs l'oracle
    assert sum(1 for d in d_oracle if d > 0) == 17         # dominance stricte 17/76
    _, p_or, n_or = wilcoxon_signed_rank(d_oracle)
    assert n_or == 17 and p_or == pytest.approx(2.881e-4, rel=1e-2)


# =============================================================================
# TOUR 26 — descripteur d'excursion GELÉ + bloc frais 1001-3000 (stratification)
# =============================================================================

def test_excursion_frozen_descriptor_algebra() -> None:
    """Descripteur GELÉ a priori : ``excursion = |k − φ*·N|`` — algèbre d'instrument.

    (a) valeur exacte sur profil connu ; (b) identité VOULUE avec la variante
    multiset (une étiquette de partition doit être shuffle-invariante et
    non-circulaire avec l'observable order-sensible) ; (c) shuffle-invariance
    (un cycle reste dans sa strate sous la porte 3) ; (d) ancrage algébrique :
    l'excursion est le PIC EXACT de |e_t − target| du lecteur fixe nominal ;
    (e) constantes gelées (seuil = band, plancher 20, bloc 1001-3000).
    """
    orients = [+1] * 4 + [-1] * 3                     # N=7, k=4, |4 − 14/3| = 2/3
    assert excursion(orients) == pytest.approx(2.0 / 3.0)
    # (b) identité algébrique avec obs_struct_multiset (propriété voulue, documentée)
    toks = _toks(orients)
    assert excursion(orients) == pytest.approx(obs_struct_multiset(toks))
    # (c) étiquette shuffle-INVARIANTE : la strate d'un cycle survit à la porte 3
    assert excursion(shuffle_orientations(orients, 123)) == pytest.approx(excursion(orients))
    # (d) pic exact de |e_t − target| du lecteur fixe nominal (g=1) — profil k=2, N=4
    o2 = [+1, +1, -1, -1]
    tr = reconstruct_fixed(o2, g_fixed=1.0)
    peak = max(abs(e_t - TARGET_LEAD) for e_t in tr.e[1:])
    assert peak == pytest.approx(excursion(o2)) == pytest.approx(2.0 / 3.0)
    # (e) constantes gelées a priori (émission T26 §1) — seuil DÉRIVÉ de la bande
    assert EXCURSION_THRESHOLD == BAND_LEAD == 0.5
    assert MIN_NONZERO_STRATUM == 20
    assert BLOCK26_LINES == (1001, 3000)


def test_excursion_strata_partitioned_reading_synthetic() -> None:
    """Partition des MÊMES Δ par excursion : seuil, plancher de puissance, NaN, erreurs."""
    deltas = [0.1, 0.0, -0.2, 0.3]
    excs = [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    st = excursion_strata(deltas, excs)
    assert isinstance(st, ExcursionStrata)
    assert st.threshold == EXCURSION_THRESHOLD
    assert st.n_high == 3 and st.n_low == 1
    assert st.delta_high_median == pytest.approx(0.1)   # médiane de {0.1, −0.2, 0.3}
    assert st.delta_low_median == 0.0
    assert st.n_high_nonzero == 3 and st.n_low_nonzero == 0
    # plancher de puissance : 3 < 20 ⇒ non interprétable (critère, pas verdict)
    assert not st.high_interpretable and not st.low_interpretable
    st2 = excursion_strata(deltas, excs, min_nonzero=2)
    assert st2.high_interpretable and not st2.low_interpretable
    assert st.contrast == pytest.approx(0.1)
    # strate vide rapportée telle quelle (médiane NaN, jamais masquée)
    st_empty = excursion_strata([0.2, 0.4], [0.1, 0.2])
    assert st_empty.n_high == 0 and math.isnan(st_empty.delta_high_median)
    assert math.isnan(st_empty.contrast)
    # séquences non alignées ⇒ erreur, pas silence
    with pytest.raises(ValueError):
        excursion_strata([0.1], [0.5, 0.6])


def test_collect_profiles_line_range_fixture(tmp_path) -> None:
    """Chargement par offset gelé : 1-based inclusif, ``None`` ≡ comportement T24/T25."""
    line_b = _LINE.replace("un deux trois", "aaa bbb ccc")
    p = tmp_path / "mini_aba.txt"
    # lignes 1..4 : A, B, A, B (+ terminateur ligne 5)
    p.write_text(_LINE + "\n" + line_b + "\n" + _LINE + "\n" + line_b + "\n<EOS>\n",
                 encoding="utf-8")
    full = collect_profiles(str(p), n_cycles=10)
    assert len(full) == 4
    assert collect_profiles(str(p), n_cycles=10, line_range=None) == full
    # tranche 2-3 (1-based inclusif) : B puis A, dans l'ordre du fichier
    sl = collect_profiles(str(p), n_cycles=10, line_range=(2, 3))
    assert len(sl) == 2
    assert sl[0][0].text == "aaa" and sl[1][0].text == "un"
    # tranche 1-1 : la première ligne seulement
    one = collect_profiles(str(p), n_cycles=10, line_range=(1, 1))
    assert len(one) == 1 and one[0][0].text == "un"
    # tranche hors fichier : vide, sans erreur (le filtre décide, jamais le chargement)
    assert collect_profiles(str(p), n_cycles=10, line_range=(6, 9)) == []


@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_population_descriptor_block26_measured() -> None:
    """Descripteur MESURÉ du bloc frais gelé lignes 1001-3000 (gravé, jamais forcé).

    Mesuré le 2026-07-02, indices de bloc gelés AVANT toute lecture de contenu
    (émission T26 §0). Population : 2000 cycles utilisables (100 % du bloc),
    cycles COURTS comme au T24 (min 6 / méd 8 / max 19) et k/N quasi-constant
    (σ = 0.0380 ≈ référence T24 0.038 ; critère σ ≥ 0.076 FAIL). Distribution
    d'excursion : quantifiée sur {0, 1/3, 2/3} (grammaire à segments courts),
    max = 2/3 ; strate HAUTE (> band = 0.5) n = 757, strate BASSE n = 1243.
    740 profils à k/N = φ* exact (Δ = 0 par construction, cas limite T24).
    """
    d = population_descriptor(
        str(_DATASET), n_cycles=N_CYCLES_BLOCK26, line_range=BLOCK26_LINES
    )
    assert d.n_cycles == 2000
    assert (d.tokens_min, d.tokens_median, d.tokens_max) == (6, 8.0, 19)
    assert d.kn_min == pytest.approx(0.5714, abs=1e-4)
    assert d.kn_median == pytest.approx(0.6250, abs=1e-4)
    assert d.kn_max == pytest.approx(2.0 / 3.0, abs=1e-4)
    assert d.kn_sigma == pytest.approx(0.0380, abs=1e-4)
    assert not d.variance_material
    assert d.n_at_phi_star_exact == 740
    assert d.exc_min == 0.0
    assert d.exc_median == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert d.exc_max == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert d.n_exc_high == 757


@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_measured_verdict_block26_documented() -> None:
    """Verdict T26 MESURÉ sur le bloc frais (jamais forcé) : INVERSION dans la strate haute.

    Mesuré le 2026-07-02, instrument ``structural_gap.py`` GELÉ byte-à-byte,
    seuils/portes inchangés, descripteur d'excursion GELÉ AVANT la mesure.

    PORTE 0 (re-jouée sur ce bloc) : le 1er profil du bloc EST à excursion haute
    (N = 13, exc = 2/3) — primaire gap = 2.186e-1 ≫ δ_min, multiset vacuous.
    PORTE 1 : pivots exacts. PORTE 2 globale : Δ médian = +0.0000 ⇒ verdict
    global MORTE (les 62 % de cycles à excursion ≤ band écrasent la médiane).

    LECTURE STRATIFIÉE (le cœur du tour) :
      * strate BASSE (n = 1243) : Δ = 0 EXACT sur 1243/1243 — la prédiction gelée
        « excursion ≤ 0.5 ⟹ rien à réguler » tient PARFAITEMENT (0 non-nul) ;
      * strate HAUTE (n = 757, 757/757 non-nuls, interprétable) : Δ méd = −0.2857
        (−2/7), signes +254/−503, Wilcoxon p = 2.4e-82 EN DÉFAVEUR de l'organe ⇒
        CONTRE-PRÉDICTION « INVERSION » réalisée : P26-strat (Δ_haute > 0) est
        RÉFUTÉE telle qu'énoncée ; Spearman(Δ, excursion) bloc = −0.287 < 0.

    DISSECTION (exigée par la contre-prédiction : « défaut d'instrument à
    investiguer d'abord ») — le signe de Δ dans la strate haute est une fonction
    en ESCALIER DÉTERMINISTE de l'horizon N, pas un défaut d'instrument :
      * N = 7 (flip précoce k = 4, exc = 2/3) : 503/503 cycles à Δ = −2/7 EXACT
        (sur-correction sur horizon court — l'anomalie adverse T24 +4/−10,
        maintenant isolée et quantifiée à grande échelle) ;
      * N ≥ 10 : 254/254 cycles à Δ > 0 (N=10 : +0.2000 ; N=13 : +0.2308 = la
        valeur médiane T25 ; N=16 : +0.25 ; N=19 : +0.2105) ;
      * Spearman(Δ, N | strate haute) = +0.99996 — le SIGNE est un escalier
        parfait ; la seule inversion de RANG vient de l'unique cycle N=19
        (+0.2105 < +0.2308 des N=13 : la magnitude n'est pas monotone au
        sommet, quantum de f_edge oblige) ; stable sur les deux demi-blocs
        (1001-2000 : +138/−250 ; 2001-3000 : +116/−253) ;
      * côté du flip : la strate haute du bloc est 100 % flip-PRÉCOCE
        (k < φ*·N) alors que corpus_claude (39 précoces + 6 tardifs, TOUS
        positifs) couvrait les deux côtés ⇒ le côté n'est PAS le discriminateur,
        l'HORIZON l'est. Raffinement rétro-unifiant T24/T25/T26 — proposition
        POST-HOC (au journal, jamais critère de verdict de CE tour ; à geler
        a priori s'il y a suite, exactement comme l'excursion l'a été entre
        T25 et T26) : l'excursion gouverne QU'IL Y A quelque chose à réguler
        (|Δ| ≠ 0, séparation binaire parfaite 757/757 vs 0/1243 SUR CE BLOC —
        fait de bloc, pas théorème : la strate haute y est 100 % flip-précoce
        et un contre-exemple synthétique flip-tardif N=8/k=6, exc=2/3, donne
        Δ=0 — vérification ingénieur) ; l'horizon N gouverne le SIGNE
        (frontière mesurée entre N = 7 et N = 10 sur ce corpus).

    PORTE 3 PARTITIONNÉE : Δ shuffle méd (haute) = +0.0000 (3 bases de seed :
    70000/80000/91000) ⇒ réel − shuffle (haute) = −0.2857 : l'effet ADVERSE est
    LUI AUSSI porté par la structure A→B→A′ et détruit par le shuffle.
    """
    r = run_structural_regulation(
        str(_DATASET), n_cycles=N_CYCLES_BLOCK26, line_range=BLOCK26_LINES
    )
    assert r.n_cycles == 2000
    # PORTE 0 : re-jouée sur ce bloc ; le 1er profil est déjà à excursion HAUTE
    orients0 = [tk.orientation for tk in collect_profiles(
        str(_DATASET), n_cycles=1, line_range=BLOCK26_LINES)[0]]
    assert len(orients0) == 13
    assert excursion(orients0) == pytest.approx(2.0 / 3.0)   # > seuil : porte 0 jouée LÀ où l'effet est revendiqué
    assert r.gate0_primary_sensitive
    assert r.order_primary.gap == pytest.approx(0.218590, abs=1e-5)
    assert r.gate0_multiset_vacuous and r.order_multiset.gap == 0.0
    # PORTE 1 : pivots
    assert r.pivot_eta0_exact and r.pivot_noflip == 0.0
    # PORTE 2 globale : médiane écrasée par la strate basse ⇒ verdict global MORTE
    assert r.best_fixed == 1.0
    assert r.delta_real_median == 0.0
    assert r.verdict == "MORTE"
    assert (r.sign_pos_real, r.sign_neg_real) == (254, 503)
    assert r.wilcoxon_p_real == pytest.approx(2.395e-82, rel=1e-2)
    # LECTURE STRATIFIÉE (mêmes Δ, aucun recalcul)
    profiles = collect_profiles(
        str(_DATASET), n_cycles=N_CYCLES_BLOCK26, line_range=BLOCK26_LINES
    )
    ol = [[tk.orientation for tk in p] for p in profiles]
    excs = [excursion(o) for o in ol]
    st = excursion_strata(r.delta_real, excs)
    assert (st.n_high, st.n_low) == (757, 1243)
    # strate BASSE : Δ = 0 EXACT partout — la moitié « rien à réguler » TIENT
    assert st.n_low_nonzero == 0 and not st.low_interpretable
    assert st.delta_low_median == 0.0
    assert all(d == 0.0 for d, e in zip(r.delta_real, excs) if e <= st.threshold)
    # strate HAUTE : interprétable ET INVERSÉE (contre-prédiction réalisée)
    assert st.n_high_nonzero == 757 and st.high_interpretable
    assert st.delta_high_median == pytest.approx(-2.0 / 7.0, abs=1e-9)
    assert st.contrast == pytest.approx(-2.0 / 7.0, abs=1e-9)
    rho = spearman_rho(r.delta_real, excs)
    assert rho == pytest.approx(-0.2870, abs=1e-3)
    assert spearman_t_pvalue(rho, len(excs)) < 1e-30
    # DISSECTION : signe = escalier déterministe de l'horizon N (strate haute)
    d_high = [(len(o), d) for o, d, e in zip(ol, r.delta_real, excs) if e > st.threshold]
    d_n7 = [d for n, d in d_high if n == 7]
    d_n10p = [d for n, d in d_high if n >= 10]
    assert len(d_n7) == 503 and len(d_n10p) == 254
    assert all(d == pytest.approx(-2.0 / 7.0, abs=1e-12) for d in d_n7)
    assert all(d > 0 for d in d_n10p)
    assert sorted(set(n for n, _ in d_high)) == [7, 10, 13, 16, 19]
    rho_n = spearman_rho([d for _, d in d_high], [float(n) for n, _ in d_high])
    assert rho_n == pytest.approx(0.999956, abs=1e-5)   # quasi-parfait, PAS 1.0 (cycle N=19)
    # côté du flip : strate haute 100 % PRÉCOCE (k < φ*·N) sur ce bloc
    from spiraton.experimental.structural_gap import PHI_STAR as _PHI
    assert all(
        sum(1 for x in o if x == +1) - _PHI * len(o) < 0
        for o, e in zip(ol, excs) if e > st.threshold
    )
    # PORTE 3 PARTITIONNÉE : l'effet adverse est porté par l'ordre, détruit au shuffle
    dsh_high = [d for d, e in zip(r.delta_shuffle, excs) if e > st.threshold]
    s = sorted(dsh_high)
    med_sh_high = 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    assert med_sh_high == 0.0
    assert st.delta_high_median - med_sh_high == pytest.approx(-2.0 / 7.0, abs=1e-9)


@pytest.mark.skipif(not _DATASET.is_file(), reason="dataset_aba.txt indisponible")
def test_fairness_oracle_high_stratum_block26() -> None:
    """Lentille d'équité héritée T25 (REFUS) : strate haute vs ORACLE fixe par cycle.

    Grille fine 0.500..2.000 pas 0.005 (301 points, hors protocole gelé —
    contrôle). Mesuré le 2026-07-02 : dans la strate haute, l'organe fait
    JEU ÉGAL avec l'oracle sur les 254 cycles N ≥ 10 (ses Δ > 0 y sont donc
    grille-INDÉPENDANTS : aucun lecteur fixe ne fait mieux) mais il est
    STRICTEMENT PIRE que l'oracle sur les 503 cycles N = 7 (Δ_oracle = −3/7
    exact, jamais meilleur, Wilcoxon p = 2.1e-111) — l'inverse exact de la
    dominance T25 (+17/−0 vs oracle). La sur-correction sur horizon N = 7 est
    donc un fait de l'ORGANE (η = 0.5 gelé sur horizon court), pas un artefact
    de la grille gelée.
    """
    r = run_structural_regulation(
        str(_DATASET), n_cycles=N_CYCLES_BLOCK26, line_range=BLOCK26_LINES
    )
    profiles = collect_profiles(
        str(_DATASET), n_cycles=N_CYCLES_BLOCK26, line_range=BLOCK26_LINES
    )
    ol = [[tk.orientation for tk in p] for p in profiles]
    excs = [excursion(o) for o in ol]
    from spiraton.diagnostics.edge_maintenance import wilcoxon_signed_rank

    fine = [0.5 + 0.005 * i for i in range(301)]
    hi = [i for i, e in enumerate(excs) if e > EXCURSION_THRESHOLD]
    fe_oracle = [
        max(f_edge_struct(reconstruct_fixed(ol[i], g_fixed=g)) for g in fine)
        for i in hi
    ]
    d_or = [r.ctrl_f_edge[i] - f for i, f in zip(hi, fe_oracle)]
    assert len(d_or) == 757
    assert sum(1 for d in d_or if d > 0) == 0            # jamais meilleur que l'oracle ici
    assert sum(1 for d in d_or if d < 0) == 503          # strictement pire sur tous les N=7
    assert min(d_or) == pytest.approx(-3.0 / 7.0, abs=1e-9)
    # les 254 ex æquo avec l'oracle sont EXACTEMENT les 254 cycles à Δ > 0 (N ≥ 10)
    ties = [i for i, d in zip(hi, d_or) if d == 0.0]
    assert len(ties) == 254
    assert all(r.delta_real[i] > 0 for i in ties)
    _, p_or, n_or = wilcoxon_signed_rank(d_or)
    assert n_or == 503 and p_or == pytest.approx(2.123e-111, rel=1e-2)


@pytest.mark.skipif(not _CORPUS_CLAUDE.is_file(), reason="corpus_claude_aba.txt indisponible")
def test_claude_high_stratum_side_reading_documented() -> None:
    """Relecture T25 sous la grille de dissection T26 (côté du flip) — gravée.

    Sur corpus_claude, la strate haute (n = 45) couvre LES DEUX côtés du flip
    (39 précoces k < φ*·N, 6 tardifs) et TOUS ses Δ sont positifs (N ≥ 11
    partout) ⇒ le côté du flip n'est pas le discriminateur du signe ; l'horizon
    N l'est (cohérent avec la frontière 7 < N* ≤ 10 mesurée sur le bloc T26).
    """
    r = run_structural_regulation(str(_CORPUS_CLAUDE), n_cycles=N_CYCLES_CLAUDE)
    profiles = collect_profiles(str(_CORPUS_CLAUDE), n_cycles=N_CYCLES_CLAUDE)
    ol = [[tk.orientation for tk in p] for p in profiles]
    excs = [excursion(o) for o in ol]
    from spiraton.experimental.structural_gap import PHI_STAR as _PHI
    hi = [i for i, e in enumerate(excs) if e > EXCURSION_THRESHOLD]
    assert len(hi) == 45
    sides = [sum(1 for x in ol[i] if x == +1) - _PHI * len(ol[i]) for i in hi]
    assert sum(1 for s in sides if s < 0) == 39          # flips précoces
    assert sum(1 for s in sides if s > 0) == 6           # flips tardifs
    assert all(r.delta_real[i] > 0 for i in hi)          # tous positifs, deux côtés
    assert min(len(ol[i]) for i in hi) == 11             # aucun horizon court ici
