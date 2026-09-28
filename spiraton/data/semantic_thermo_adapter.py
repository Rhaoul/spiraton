"""Adaptateur 33D → trajectoires de segments pour la thermodynamique sémantique.

Protocole : ``docs/SEMANTIC_THERMO_PROTOCOLE.md`` §1.

VERROU ANTI-FUITE : chaque segment est tokenisé SEUL, sur son texte nettoyé
(``AbaSegment.text``, tags retirés) ; le vecteur d'un mot ne dépend donc pas de
la position du segment dans le cycle. Les étiquettes (chiralité, opérateur,
forme) sont portées à côté des vecteurs et ne servent qu'à l'évaluation.

Tranches explicites (``SLICES``) — les dims 0-5 (scores opérateurs et
chiralité, construites pour parler la langue du Logos) sont exclues de toutes
les tranches sauf ``full``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import vector33d as v33
from .aba import AbaCycle, iter_aba_cycles
from .aba_forms import classify_form_extended, pole_of

SLICES: Dict[str, Tuple[int, ...]] = {
    "full": tuple(range(0, 33)),
    "no-logos": tuple(range(6, 33)),     # primaire (doc §30)
    "no-energy": tuple(range(8, 33)),    # CTRL-4
    "phoneme": tuple(range(8, 23)),      # CTRL-5
    "context": tuple(range(23, 31)),     # CTRL-6
}

SEG_NAMES = ("SEG_A", "SEG_B", "SEG_A_PRIME")

VectorFn = Callable[[str], "object"]   # texte → array (n, 33)


@dataclass(frozen=True)
class SegmentRecord:
    """Un segment tokenisé. ``vectors`` : (n, 33) float64, n ≥ 0."""

    seg_name: str
    chirality: str      # DX | LV   (évaluation seulement)
    text: str
    vectors: torch.Tensor


@dataclass(frozen=True)
class CycleRecord:
    corpus: str
    index: int
    op: str             # évaluation seulement
    form: str           # forme étendue (F0, F0b, …, F5+, F5-)
    segments: Tuple[SegmentRecord, SegmentRecord, SegmentRecord]


def cycle_form(c: AbaCycle) -> str:
    signs = [+1 if s.chirality == "DX" else -1 for s in c.segments.values()]
    lens = [len(s.text.split()) for s in c.segments.values()]
    form = classify_form_extended(signs, lens)
    return pole_of(signs) if form == "F5" else form


class CachedVectors:
    """Mémoïse ``fn(text)`` (le tokenizer est déterministe, cf. test de parité)."""

    def __init__(self, fn: VectorFn) -> None:
        self._fn = fn
        self._cache: Dict[str, torch.Tensor] = {}

    def __call__(self, text: str) -> torch.Tensor:
        if text not in self._cache:
            if text.strip():
                arr = torch.as_tensor(self._fn(text), dtype=torch.float64).reshape(-1, v33.DIM)
            else:
                arr = torch.zeros(0, v33.DIM, dtype=torch.float64)
            self._cache[text] = arr
        return self._cache[text]


def load_cycles(
    path: str,
    vectors: CachedVectors,
    *,
    corpus: Optional[str] = None,
    drop_fixed_points: bool = True,
    dedupe: bool = True,
) -> List[CycleRecord]:
    """Cycles d'un corpus ABA, dédoublonnés sur la ligne brute, points fixes exclus."""
    name = corpus or path.replace("\\", "/").rsplit("/", 1)[-1]
    seen = set()
    out: List[CycleRecord] = []
    for c in iter_aba_cycles(path):
        if drop_fixed_points and c.is_fixed_point:
            continue
        if dedupe:
            if c.raw in seen:
                continue
            seen.add(c.raw)
        segs = tuple(
            SegmentRecord(k, s.chirality, s.text, vectors(s.text))
            for k, s in c.segments.items()
        )
        out.append(CycleRecord(name, len(out), c.op, cycle_form(c), segs))  # type: ignore[arg-type]
    return out


def split_cycles(n: int, *, seed: int, frac: float = 0.5) -> Tuple[List[int], List[int]]:
    """Partage seedé des indices de cycles en (calibration, test)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(round(frac * n))
    return sorted(idx[:cut]), sorted(idx[cut:])


def pool_rows(cycles: Sequence[CycleRecord]) -> torch.Tensor:
    rows = [s.vectors for c in cycles for s in c.segments if s.vectors.size(0)]
    return torch.cat(rows, dim=0) if rows else torch.zeros(0, v33.DIM, dtype=torch.float64)


class SliceNormalizer:
    """Sélection d'une tranche + z-score gelé sur un pool de calibration."""

    def __init__(self, calib_rows: torch.Tensor, slice_name: str, *, normalize: bool = True) -> None:
        if slice_name not in SLICES:
            raise KeyError(f"tranche inconnue : {slice_name!r} ({sorted(SLICES)})")
        self.slice_name = slice_name
        self.dims = list(SLICES[slice_name])
        x = calib_rows[:, self.dims]
        self.mean = x.mean(dim=0) if normalize else torch.zeros(len(self.dims), dtype=torch.float64)
        std = x.std(dim=0, unbiased=False) if normalize else torch.ones(len(self.dims), dtype=torch.float64)
        self.scale = torch.where(std > 1e-8, 1.0 / std.clamp_min(1e-8), torch.zeros_like(std))

    def __call__(self, rows: torch.Tensor) -> torch.Tensor:
        return (rows[:, self.dims] - self.mean) * self.scale


# --- contrôles de position (CTRL-CUT, CTRL-EVE) -----------------------------------

def recut_cycle(c: CycleRecord, vectors: CachedVectors, rng: random.Random) -> CycleRecord:
    """CTRL-CUT : même phrase (mots de A+B+A′), frontières redistribuées.

    Les longueurs en mots des trois segments sont permutées aléatoirement
    (multiensemble de longueurs conservé), puis chaque tronçon est retokenisé
    seul. Détruit l'alignement des frontières ABA, garde texte et longueurs.
    """
    words = [w for s in c.segments for w in s.text.split()]
    lens = [len(s.text.split()) for s in c.segments]
    rng.shuffle(lens)
    chunks, i = [], 0
    for L in lens:
        chunks.append(" ".join(words[i:i + L]))
        i += L
    segs = tuple(
        SegmentRecord(name, orig.chirality, t, vectors(t))
        for name, orig, t in zip(SEG_NAMES, c.segments, chunks)
    )
    return CycleRecord(c.corpus + "+cut", c.index, c.op, c.form, segs)  # type: ignore[arg-type]


def sentence_thirds(path: str, vectors: CachedVectors) -> List[CycleRecord]:
    """CTRL-EVE : phrases non-ABA coupées en trois tronçons de mots (≥ 1 mot chacun),
    tailles ⌊n/3⌋ réparties du début vers la fin, chaque tronçon tokenisé seul.
    Étiquettes factices (LV sur le 3ᵉ tronçon, pour la seule symétrie des tables)."""
    out: List[CycleRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            words = line.split()
            if len(words) < 3:
                continue
            n = len(words)
            base, extra = divmod(n, 3)
            lens = [base + (1 if i >= 3 - extra else 0) for i in range(3)]
            chunks, i = [], 0
            for L in lens:
                chunks.append(" ".join(words[i:i + L]))
                i += L
            segs = tuple(
                SegmentRecord(name, "LV" if name == "SEG_A_PRIME" else "DX", t, vectors(t))
                for name, t in zip(SEG_NAMES, chunks)
            )
            out.append(CycleRecord("eve_thirds", len(out), "NONE", "NONE", segs))  # type: ignore[arg-type]
    return out
