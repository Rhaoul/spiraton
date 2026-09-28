from __future__ import annotations

"""η sur runners réels — loi de pureté ÉTENDUE P-η (Tour 30, clôture lévo·in).

H30 (émission linguiste, gelée AVANT toute mesure — ``TOUR30_EMISSION.md``) : pour
chaque cycle réel de profil canonique MONO-FLIP (N, k) lu par ``aba.py``,

    Δ_real(η) == delta_nom_exact_eta(N, k, η)   (valeur ≤ 1e-12, identité de signe)

SAUF sur les cellules à bord-exact PRÉ-DÉCLARÉES (table d'offsets ci-dessous),
où l'OFFSET float lui-même est prédit exactement : Δ_real(η) == exact + offset.
Précondition gravée : ``reconstruct_fixed`` IGNORE η ⟹ ``best_fixed_gain`` est
η-INVARIANT ⟹ le best_fixed = 1.0 gravé au T27 sur les trois runs tient à tout η.
Sous ces préconditions la loi est largement TAUTOLOGIQUE — VOULU (émission §1) :
la valeur du tour est DIAGNOSTIQUE ; ce qui la casserait est exactement
l'information cherchée (§4bis : non-canonicité, dérive de baseline, bord-exact,
ou une variable du réel HORS (N, k) — résultat MAJEUR à graver tel quel).

LES η GELÉS (émission §2) : {1/2 (CONTRÔLE DUR : doit reproduire les 2116/2116
T27 en signe ET valeur), 1/4, 1, 3/2, 4}. Prédictions = ``delta_nom_exact_eta``
(spectral_map, LECTURE SEULE) + la table d'offsets recopiée des gravures T28 —
ZÉRO dérivation neuve, zéro degré de liberté.

PORTES (ordre lexicographique, premier échec = verdict — émission §8) :

  A.  RECENSEMENT A PRIORI gelé AVANT tout run η (:func:`census`, §4) : liste des
      (N, k) réels + multiplicités par corpus depuis ``aba.py`` exclusivement ;
      canonicité = EXACTEMENT une transition +1→−1 et aucune −1→+1 (mono-flip ⟹
      soumis au critère ; 0/multi-flip ⟹ HORS-CARTE, recensés à part, EXCLUS du
      critère — candidat (3) T27 nommé, pas traité ce tour) ; appartenance grille
      [6, 24] × [2, N−2] (hors-grille : le moteur exact calcule quand même,
      signalés) ; TABLE D'OFFSETS pré-déclarée (``FROZEN_OFFSETS``) ; bandes
      d'alerte de frontière (croisement diagramme T29, :func:`frontier_alerts`,
      lecture seule — valeur prédite = valeur du nœud, tracée).
  0a. Order-sensibilité re-confirmée + multiset vacuous (``gate0a`` T27, importée).
  B.  Précondition baseline : best_fixed == 1.0 sur les TROIS corpus ; chemin par
      défaut du runner BYTE-IDENTIQUE (:func:`default_path_byte_identical`).
  C.  CONTRÔLE DUR η = 1/2 : 2116/2116 reproduits (signe ET valeur ≤ 1e-12),
      sinon ARRÊT (aucun η frais n'est mesuré).
  D.  Mesure fraîche 4 η × 3 corpus : signe + valeur vs prédiction gelée hors
      pré-déclarées ; sur pré-déclarées : offset mesuré == offset prédit.

TABLE D'OFFSETS PRÉ-DÉCLARÉE (émission §4.4). Offset = float_gravé − exact_gravé,
recopié des divergences de magnitude GRAVÉES au VERDICT T28 (``spectral_map``,
en-tête, mesure du 2026-07-11) — lecture seule, aucune dérivation neuve. Les
cellules (9,7) → +1/9 et (15,11) → +1/15 sont η-INVARIANTES (bord-exact du
lecteur FIXE, T27) ; les autres sont des bords-exacts de l'ORGANE, par η.
(9,7) est la SEULE à pouvoir basculer un signe (quand Δ_exact ∈ {0, −1/9}).

RÉFUTATIONS ÉTIQUETÉES AVANT MESURE (émission §4bis) :
  * Δ_real ≠ delta_nom_float_eta (valeur) sur mono-flip ⟹ best_fixed ≠ 1.0 OU
    non-canonicité non détectée ;
  * Δ_real ≠ delta_nom_exact_eta hors pré-déclarées ⟹ variable HORS (N, k) au
    réel (borne du modèle, MAJEUR) OU défaut du chemin runner-η (ordre de
    dissection gelé : best_fixed → canonicité → identité chemin float → offset
    bord — champ ``float_map`` des mismatches) ;
  * offset ≠ offset prédit ⟹ compréhension T29 du défaut float incomplète ;
  * best_fixed ≠ 1.0 à η = 1/2 ⟹ régression du montage : ARRÊT.

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/``, NI ``regulate_step``,
NI ``structural_gap.py`` (GELÉ byte-à-byte), NI ``horizon_law.py`` /
``spectral_map.py`` / ``phase_diagram.py`` (T27-T29 gravés — importés en lecture
seule). Le chemin η passe par le kwarg ``eta`` de ``run_structural_regulation``
(T30, défaut ``ETA_STRUCT`` byte-identique — jamais par édition de constante).
Anti-circularité : profils depuis ``aba.py`` exclusivement (jamais les dims 0-5
du 33D). Tout est déterministe (shuffles seedés hérités du runner).
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..experimental.structural_gap import ETA_STRUCT, orientation_profile
from .horizon_law import GRID_K_MIN, GRID_N_MAX, GRID_N_MIN, gate0a
from .phase_diagram import cell_diagram
from .spectral_map import ETAS_FROZEN, delta_nom_exact_eta, delta_nom_float_eta
from .structural_regulation import (
    BLOCK26_LINES,
    MIN_TOKENS,
    N_CYCLES_BLOCK26,
    N_CYCLES_CLAUDE,
    N_CYCLES_DEFAULT,
    _iter_cycles,
    collect_profiles,
    run_structural_regulation,
)

# --- constantes GELÉES A PRIORI (émission T30 — jamais ajustées après mesure) -------

VALUE_TOL = 1e-12                 # tolérance de VALEUR héritée T27 (porte 3)
T30_CONTROL_LABEL = "1/2"         # l'ancre : contrôle dur byte-identique T27
ETA_FRESH_LABELS: Tuple[str, ...] = ("1/4", "1", "3/2", "4")   # les 4 régimes frais
ETA_LABELS_RUNNERS: Tuple[str, ...] = (T30_CONTROL_LABEL,) + ETA_FRESH_LABELS
EXPECTED_CONTROL_TOTAL = 2116     # 40 + 76 + 2000 (T27 porte 3, gravé)

# Table d'offsets PRÉ-DÉCLARÉE : offset = float_gravé − exact_gravé, recopié des
# divergences de magnitude du VERDICT T28 (spectral_map, lecture seule).
FROZEN_OFFSETS: Dict[str, Dict[Tuple[int, int], Fraction]] = {
    "1/4": {(9, 7): Fraction(1, 9), (15, 11): Fraction(1, 15)},
    "1/2": {(9, 7): Fraction(1, 9), (15, 11): Fraction(1, 15)},
    "1": {(9, 7): Fraction(1, 9), (14, 7): Fraction(-1, 14),
          (15, 11): Fraction(1, 15), (16, 8): Fraction(-1, 16),
          (18, 9): Fraction(-1, 18), (20, 10): Fraction(-1, 20)},
    "3/2": {(9, 7): Fraction(1, 9), (15, 11): Fraction(1, 15)},
    "4": {(8, 6): Fraction(-1, 8), (9, 4): Fraction(-1, 9), (9, 7): Fraction(1, 9),
          (12, 9): Fraction(-1, 12), (15, 11): Fraction(1, 15),
          (16, 12): Fraction(-1, 16), (20, 15): Fraction(-1, 20),
          (21, 10): Fraction(-1, 21), (24, 11): Fraction(-1, 24),
          (24, 18): Fraction(-1, 24)},
}

# η-invariantes (bord-exact du lecteur FIXE, T27) — présentes à TOUS les η gelés.
OFFSETS_ETA_INVARIANT: Dict[Tuple[int, int], Fraction] = {
    (9, 7): Fraction(1, 9),
    (15, 11): Fraction(1, 15),
}


def corpus_runs() -> List[Tuple[str, str, int, Optional[Tuple[int, int]]]]:
    """Les TROIS runs réels gelés (label, chemin, n_cycles, line_range) — T24/T25/T26."""
    root = Path("F:/code/claude/spiraton-enhanced")
    if not (root / "dataset_aba.txt").is_file():
        root = Path(__file__).resolve().parents[3]
    return [
        ("T24-defaut", str(root / "dataset_aba.txt"), N_CYCLES_DEFAULT, None),
        ("T25-claude", str(root / "corpus_claude_aba.txt"), N_CYCLES_CLAUDE, None),
        ("T26-bloc26", str(root / "dataset_aba.txt"), N_CYCLES_BLOCK26, BLOCK26_LINES),
    ]


# --- canonicité et grille (porte A, définitions gelées) -----------------------------

def transition_counts(orientations: Sequence[int]) -> Tuple[int, int]:
    """(# transitions +1→−1, # transitions −1→+1) — la canonicité en deux comptes."""
    ud = sum(1 for a, b in zip(orientations, orientations[1:]) if a == +1 and b == -1)
    du = sum(1 for a, b in zip(orientations, orientations[1:]) if a == -1 and b == +1)
    return ud, du


def is_mono_flip(orientations: Sequence[int]) -> bool:
    """Canonique MONO-FLIP : EXACTEMENT une transition +1→−1 et aucune −1→+1.

    Équivalent (les deux orientations étant présentes) à ``[+1]*k + [−1]*(N−k)`` :
    le profil est entièrement décrit par (N, k) — condition d'application de la
    carte. Tout autre profil est HORS-CARTE (recensé, exclu du critère).
    """
    ud, du = transition_counts(orientations)
    return ud == 1 and du == 0


def in_grid(n: int, k: int) -> bool:
    """Appartenance à la grille T27 gelée : N ∈ [6, 24], k ∈ [2, N−2]."""
    return GRID_N_MIN <= n <= GRID_N_MAX and GRID_K_MIN <= k <= n - 2


# --- PORTE A : recensement a priori (gelé AVANT tout run η) --------------------------

@dataclass(frozen=True)
class CorpusCensus:
    """Recensement d'un corpus AVANT toute mesure η (émission §4).

    ``mono_cells`` = (N, k, multiplicité) triés ; ``non_canonical`` = cycles
    HORS-CARTE (index runner, N, # +1→−1, # −1→+1) ; ``filtered_out`` = cycles
    du fichier REJETÉS par le filtre du runner dans la même fenêtre de lecture
    (< 4 tokens ou une seule orientation — les 0-flip vivent ici) ;
    ``off_grid_cells`` = mono-flip hors [6,24]×[2,N−2] (signalés, moteur exact
    calcule quand même) ; ``offset_cells_present`` = cellules de la table
    d'offsets PRÉSENTES dans le réel, par η (le cœur diagnostique du tour :
    vide partout ⟹ livrable négatif légitime, émission §10).
    """

    label: str
    n_used: int                                   # cycles retenus par le runner
    n_filtered_out: int                           # rejetés par le filtre (même fenêtre)
    n_mono: int
    n_non_canonical: int
    mono_cells: Tuple[Tuple[int, int, int], ...]  # (N, k, multiplicité)
    n_distinct_cells: int
    non_canonical: Tuple[Tuple[int, int, int, int], ...]
    off_grid_cells: Tuple[Tuple[int, int, int], ...]
    offset_cells_present: Dict[str, Tuple[Tuple[int, int], ...]]


def census(path: str, *, label: str, n_cycles: int,
           line_range: Optional[Tuple[int, int]] = None) -> CorpusCensus:
    """Porte A : recense les (N, k) réels d'un corpus — ``aba.py`` exclusivement.

    MÊME fenêtre de lecture et MÊME filtre que :func:`collect_profiles` (le
    recensement décrit exactement la population que les portes C/D jugeront) ;
    aucune mesure η, aucun float d'organe : gelable AVANT tout run.
    """
    used: List[List[int]] = []
    n_filtered = 0
    for cycle in _iter_cycles(path, line_range):
        orients = [tk.orientation for tk in orientation_profile(cycle)]
        if len(orients) >= MIN_TOKENS and set(orients) == {+1, -1}:
            used.append(orients)
        else:
            n_filtered += 1
        if len(used) >= n_cycles:
            break
    mono: Counter = Counter()
    non_canon: List[Tuple[int, int, int, int]] = []
    for i, o in enumerate(used):
        if is_mono_flip(o):
            n = len(o)
            k = sum(1 for x in o if x == +1)
            mono[(n, k)] += 1
        else:
            ud, du = transition_counts(o)
            non_canon.append((i, len(o), ud, du))
    cells = tuple(sorted((n, k, m) for (n, k), m in mono.items()))
    off_grid = tuple((n, k, m) for (n, k, m) in cells if not in_grid(n, k))
    offset_present = {
        lbl: tuple(sorted(c for c in FROZEN_OFFSETS[lbl] if c in mono))
        for lbl in ETA_LABELS_RUNNERS
    }
    return CorpusCensus(
        label=label, n_used=len(used), n_filtered_out=n_filtered,
        n_mono=sum(mono.values()), n_non_canonical=len(non_canon),
        mono_cells=cells, n_distinct_cells=len(cells),
        non_canonical=tuple(non_canon), off_grid_cells=off_grid,
        offset_cells_present=offset_present,
    )


@dataclass(frozen=True)
class FrontierAlert:
    """Bande d'alerte : un η gelé coïncide avec une structure du diagramme T29.

    ``kind`` : « frontière-certifiée » (η EXACTEMENT sur une frontière rationnelle
    certifiée du diagramme) ou « nœud-zéro » (Δ_exact(η) = 0 : le nœud vit dans un
    plateau zéro). INFORMATIF, tracé — la prédiction reste la valeur du nœud.
    """

    n: int
    k: int
    eta_label: str
    kind: str


def frontier_alerts(cells: Sequence[Tuple[int, int]]) -> List[FrontierAlert]:
    """Croise les cellules réelles EN GRILLE avec le diagramme T29 (lecture seule)."""
    alerts: List[FrontierAlert] = []
    for (n, k) in sorted(set(cells)):
        if not in_grid(n, k):
            continue
        cd = cell_diagram(n, k)
        certified = {f.certified for f in cd.structure.frontiers
                     if f.certified is not None}
        for lbl in ETA_LABELS_RUNNERS:
            eta = ETAS_FROZEN[lbl]
            if eta in certified:
                alerts.append(FrontierAlert(n=n, k=k, eta_label=lbl,
                                            kind="frontière-certifiée"))
            elif delta_nom_exact_eta(n, k, eta) == 0:
                alerts.append(FrontierAlert(n=n, k=k, eta_label=lbl,
                                            kind="nœud-zéro"))
    return alerts


# --- PORTE B : byte-identité du chemin par défaut (invariant de comparabilité) ------

def default_path_byte_identical(path: str, *, n_cycles: int,
                                line_range: Optional[Tuple[int, int]] = None) -> bool:
    """Le runner SANS kwarg == le runner avec ``eta=ETA_STRUCT`` (rapports égaux).

    Le défaut du kwarg EST la constante gelée : mêmes suites d'opérations
    flottantes ⟹ égalité stricte des rapports (portes, Δ, shuffles compris).
    """
    r_default = run_structural_regulation(path, n_cycles=n_cycles, line_range=line_range)
    r_eta = run_structural_regulation(path, n_cycles=n_cycles, line_range=line_range,
                                      eta=ETA_STRUCT)
    return r_default == r_eta


# --- PORTES C/D : croisement prédiction gelée ↔ mesure du runner paramétré ----------

@dataclass(frozen=True)
class OffsetHit:
    """Un cycle réel tombé sur une cellule à offset PRÉ-DÉCLARÉE (émission §4.4)."""

    index: int
    n: int
    k: int
    offset_measured: float            # Δ_real − float(Δ_exact)
    offset_predicted: float           # float(offset gravé)
    ok: bool                          # |Δ_real − float(exact + offset)| ≤ 1e-12


@dataclass(frozen=True)
class CycleMismatch:
    """Un écart prédiction ↔ mesure, avec le matériel de dissection (§4bis).

    ``float_map`` = Δ_nom_float(N, k, η) recalculé sur le profil synthétique
    (identité de chemin float : si Δ_real == float_map, l'écart vit dans la
    dérivation exacte/offset ; sinon dans le runner — baseline ou profil).
    """

    index: int
    n: int
    k: int
    delta_real: float
    predicted: float                  # float(exact [+ offset gravé])
    exact: str                        # Fraction, str (lisible)
    float_map: float
    kind: str                         # 'signe' | 'valeur'


@dataclass(frozen=True)
class EtaCorpusReport:
    """P-η sur UN corpus à UN η : concordances signe/valeur + cellules à offset.

    ``n_sign_*`` : cycles mono-flip HORS pré-déclarées (identité de signe vs
    Δ_exact) ; ``n_value_*`` : TOUS les mono-flip (valeur vs exact [+ offset],
    ≤ 1e-12) ; les pré-déclarées sont jugées via ``offset_hits``. ``all_pass``
    = critère PROGRESSION de l'émission §9 pour ce (corpus, η).
    """

    label: str
    eta_label: str
    n_cycles: int
    best_fixed: float
    n_mono: int
    n_hors_carte: int
    n_sign_ok: int
    n_sign_total: int
    n_value_ok: int
    n_value_total: int
    offset_hits: Tuple[OffsetHit, ...]
    mismatches: Tuple[CycleMismatch, ...]
    all_pass: bool


def eta_corpus_report(path: str, *, label: str, eta_label: str, n_cycles: int,
                      line_range: Optional[Tuple[int, int]] = None) -> EtaCorpusReport:
    """Mesure Δ_real(η) par le runner paramétré et le croise avec la prédiction gelée.

    Le runner est re-joué tel quel (``eta`` par kwarg, best_fixed de SON sweep —
    η-invariant) ; les profils viennent de ``aba.py`` exclusivement. Prédiction
    par cycle mono-flip : ``delta_nom_exact_eta(N, k, η)`` + offset gravé si la
    cellule est pré-déclarée. Aucun paramètre ajusté après lecture.
    """
    eta_frac = ETAS_FROZEN[eta_label]
    eta_f = float(eta_frac)           # dyadique exact pour les 5 η gelés (sans perte)
    r = run_structural_regulation(path, n_cycles=n_cycles, line_range=line_range,
                                  eta=eta_f)
    profiles = collect_profiles(path, n_cycles=n_cycles, line_range=line_range)
    offsets = FROZEN_OFFSETS[eta_label]

    n_mono = n_hc = 0
    n_sign_ok = n_sign_total = n_value_ok = n_value_total = 0
    hits: List[OffsetHit] = []
    mismatches: List[CycleMismatch] = []
    for i, (prof, d_real) in enumerate(zip(profiles, r.delta_real)):
        orients = [tk.orientation for tk in prof]
        if not is_mono_flip(orients):
            n_hc += 1                 # HORS-CARTE : exclu du critère (recensé porte A)
            continue
        n_mono += 1
        n = len(orients)
        k = sum(1 for o in orients if o == +1)
        d_exact = delta_nom_exact_eta(n, k, eta_frac)
        off = offsets.get((n, k))
        predicted = float(d_exact + off) if off is not None else float(d_exact)
        value_ok = abs(d_real - predicted) <= VALUE_TOL
        n_value_total += 1
        if value_ok:
            n_value_ok += 1
        if off is None:
            n_sign_total += 1
            sign_ok = ((d_real > 0) - (d_real < 0)) == ((d_exact > 0) - (d_exact < 0))
            if sign_ok:
                n_sign_ok += 1
        else:
            sign_ok = True            # cellule pré-déclarée : jugée via offset_hits
            hits.append(OffsetHit(
                index=i, n=n, k=k,
                offset_measured=d_real - float(d_exact),
                offset_predicted=float(off), ok=value_ok,
            ))
        if not (value_ok and sign_ok):
            mismatches.append(CycleMismatch(
                index=i, n=n, k=k, delta_real=d_real, predicted=predicted,
                exact=str(d_exact), float_map=delta_nom_float_eta(n, k, eta_f),
                kind=("valeur" if sign_ok else "signe"),
            ))
    all_pass = (
        n_sign_ok == n_sign_total
        and n_value_ok == n_value_total
        and all(h.ok for h in hits)
    )
    return EtaCorpusReport(
        label=label, eta_label=eta_label, n_cycles=r.n_cycles,
        best_fixed=r.best_fixed, n_mono=n_mono, n_hors_carte=n_hc,
        n_sign_ok=n_sign_ok, n_sign_total=n_sign_total,
        n_value_ok=n_value_ok, n_value_total=n_value_total,
        offset_hits=tuple(hits), mismatches=tuple(mismatches),
        all_pass=all_pass,
    )


# --- runner de mesure (ordre lexicographique A → 0a → B → C → D) ---------------------

if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    runs = corpus_runs()

    print("=== PORTE A : RECENSEMENT A PRIORI (gelé AVANT tout run η) ===")
    censuses = {}
    all_cells: List[Tuple[int, int]] = []
    for label, path, n_run, lr in runs:
        c = census(path, label=label, n_cycles=n_run, line_range=lr)
        censuses[label] = c
        all_cells.extend((n, k) for (n, k, _m) in c.mono_cells)
        print(f"\n--- {label} ---")
        print(f"cycles retenus     : {c.n_used} (rejetés par le filtre : {c.n_filtered_out})")
        print(f"mono-flip          : {c.n_mono} | HORS-CARTE (0/multi-flip) : {c.n_non_canonical}")
        print(f"cellules distinctes: {c.n_distinct_cells}")
        print(f"(N,k)×mult         : {list(c.mono_cells)}")
        print(f"hors-grille        : {list(c.off_grid_cells)}")
        for tr in c.non_canonical:
            print(f"  hors-carte : cycle {tr[0]} N={tr[1]} (+1→−1: {tr[2]}, −1→+1: {tr[3]})")
        for lbl in ETA_LABELS_RUNNERS:
            pres = c.offset_cells_present[lbl]
            if pres:
                print(f"  cellules à OFFSET présentes à η={lbl} : {list(pres)}")
        if all(not c.offset_cells_present[lbl] for lbl in ETA_LABELS_RUNNERS):
            print("  cellules à OFFSET présentes : AUCUNE (à tout η gelé)")
    print("\n--- bandes d'alerte de frontière (diagramme T29, lecture seule) ---")
    alerts = frontier_alerts(all_cells)
    if not alerts:
        print("aucune cellule réelle sur une frontière certifiée à un η gelé")
    for a in alerts:
        print(f"  ({a.n},{a.k}) η={a.eta_label} : {a.kind}")

    print("\n=== PORTE 0a : order-sensibilité re-confirmée (héritée T27) ===")
    g0a = gate0a()
    print(f"(13,8) primaire gap={g0a.primary_high.gap:.6e} sensible={g0a.primary_high.is_order_sensitive} | "
          f"multiset vacuous={g0a.multiset_high.is_vacuous}")
    print(f"(7,5)  primaire gap={g0a.primary_low.gap:.6e} sensible={g0a.primary_low.is_order_sensitive} | "
          f"multiset vacuous={g0a.multiset_low.is_vacuous}")
    print(f"porte 0a : {'PASSE' if g0a.passes else 'ECHEC — ARRÊT'}")

    print("\n=== PORTE B : byte-identité du défaut + best_fixed η-invariant ===")
    label0, path0, n0, lr0 = runs[0]
    bi = default_path_byte_identical(path0, n_cycles=n0, line_range=lr0)
    print(f"chemin par défaut byte-identique (T24-defaut) : {bi}")

    print("\n=== PORTE C : CONTRÔLE DUR η = 1/2 (2116/2116 signe ET valeur) ===")
    total_sign = total_value = total_n = 0
    ctrl_ok = True
    for label, path, n_run, lr in runs:
        rep = eta_corpus_report(path, label=label, eta_label=T30_CONTROL_LABEL,
                                n_cycles=n_run, line_range=lr)
        total_n += rep.n_mono
        total_sign += rep.n_sign_ok + sum(1 for h in rep.offset_hits if h.ok)
        total_value += rep.n_value_ok
        ctrl_ok = ctrl_ok and rep.all_pass and rep.best_fixed == 1.0
        print(f"{label} : n={rep.n_cycles} best_fixed={rep.best_fixed} "
              f"mono={rep.n_mono} hors-carte={rep.n_hors_carte} | "
              f"signe {rep.n_sign_ok}/{rep.n_sign_total} (hors pré-déclarées) | "
              f"valeur {rep.n_value_ok}/{rep.n_value_total} | "
              f"offset_hits={len(rep.offset_hits)} | pass={rep.all_pass}")
        for mm in rep.mismatches[:10]:
            print(f"  écart : cycle {mm.index} N={mm.n} k={mm.k} Δ_real={mm.delta_real:+.6f} "
                  f"prédit={mm.predicted:+.6f} exact={mm.exact} float_map={mm.float_map:+.6f} [{mm.kind}]")
    print(f"TOTAL contrôle : signe+valeur {min(total_sign, total_value)}/{total_n} "
          f"(attendu {EXPECTED_CONTROL_TOTAL}/{EXPECTED_CONTROL_TOTAL} sur mono-flip)")
    if not (ctrl_ok and total_n == total_value == total_sign):
        print("CONTRÔLE η=1/2 EN ÉCHEC — ARRÊT (aucun η frais mesuré, dissection requise)")
        raise SystemExit(1)

    print("\n=== PORTE D : mesure fraîche 4 η × 3 corpus ===")
    for eta_lbl in ETA_FRESH_LABELS:
        print(f"\n--- η = {eta_lbl} ---")
        for label, path, n_run, lr in runs:
            rep = eta_corpus_report(path, label=label, eta_label=eta_lbl,
                                    n_cycles=n_run, line_range=lr)
            print(f"{label} : n={rep.n_cycles} best_fixed={rep.best_fixed} "
                  f"mono={rep.n_mono} hors-carte={rep.n_hors_carte} | "
                  f"signe {rep.n_sign_ok}/{rep.n_sign_total} (hors pré-déclarées) | "
                  f"valeur {rep.n_value_ok}/{rep.n_value_total} | pass={rep.all_pass}")
            for h in rep.offset_hits:
                print(f"  OFFSET : cycle {h.index} ({h.n},{h.k}) mesuré={h.offset_measured:+.6f} "
                      f"prédit={h.offset_predicted:+.6f} ok={h.ok}")
            for mm in rep.mismatches[:10]:
                print(f"  écart : cycle {mm.index} N={mm.n} k={mm.k} Δ_real={mm.delta_real:+.6f} "
                      f"prédit={mm.predicted:+.6f} exact={mm.exact} float_map={mm.float_map:+.6f} [{mm.kind}]")
            if len(rep.mismatches) > 10:
                print(f"  ... (+{len(rep.mismatches) - 10})")
