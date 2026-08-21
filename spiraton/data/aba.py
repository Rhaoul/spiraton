"""Parseur de référence du format ABA (cycle A → B → A′).

Implémentation Python **lisible et sans dépendance** (ni torch ni numpy) du
vocabulaire ABA décrit dans CLAUDE.md. C'est la *référence partagée* entre le
dépôt spiraton (chantier 3) et le tokenizer C (« parseur de référence ») :
le `.so` n'est jamais la seule source de vérité sur ce qu'est une ligne ABA.

Grammaire (CLAUDE.md) :

    <SEG_A> <OP><DX><OUT><ALPHA> ... </SEG_A>
    <SEG_B> <OP><DX><OUT><OMEGA> ... </SEG_B>
    <SEG_A_PRIME> <OP><LV><IN><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>

Le retournement `<DX><OUT>` (A, B) → `<LV><IN>` (A′) encode la clôture spirale :
on revient au point d'origine, mais transformé. L'opérateur reste constant sur
tout le cycle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

# --- Vocabulaire canonique --------------------------------------------------

OPERATORS: Tuple[str, ...] = ("ADD", "SUB", "MUL", "DIV")
CHIRALITIES: Tuple[str, ...] = ("DX", "LV")
DIRECTIONS: Tuple[str, ...] = ("OUT", "IN")
POSITIONS: Tuple[str, ...] = ("ALPHA", "OMEGA", "A_PRIME")
SEGMENTS: Tuple[str, ...] = ("SEG_A", "SEG_B", "SEG_A_PRIME")

# Index des scores d'opérateur dans le vecteur 33D (dims 0-3) — cf. vector33d.py.
OP_INDEX: Dict[str, int] = {op: i for i, op in enumerate(OPERATORS)}


class AbaParseError(ValueError):
    """Ligne ABA malformée (segment manquant, en-tête de tags absent…)."""


@dataclass(frozen=True)
class AbaSegment:
    """Un segment d'un cycle : son en-tête de tags + son contenu lexical."""

    segment: str     # "SEG_A" | "SEG_B" | "SEG_A_PRIME"
    op: str          # ADD/SUB/MUL/DIV
    chirality: str   # DX/LV
    direction: str   # OUT/IN
    position: str    # ALPHA/OMEGA/A_PRIME
    text: str        # contenu nettoyé (tags retirés, espaces normalisés)


@dataclass(frozen=True)
class AbaCycle:
    """Un cycle complet A → B → A′, déjà validé structurellement à minima."""

    op: str                  # opérateur dominant du cycle (constant attendu)
    seg_a: AbaSegment
    seg_b: AbaSegment
    seg_a_prime: AbaSegment
    raw: str

    @property
    def segments(self) -> Dict[str, AbaSegment]:
        return {
            "SEG_A": self.seg_a,
            "SEG_B": self.seg_b,
            "SEG_A_PRIME": self.seg_a_prime,
        }

    @property
    def triplet(self) -> Dict[str, object]:
        """Le triplet `{op, chiralité, segments}` attendu par le tokenizer."""
        return {
            "op": self.op,
            "chirality": {
                "SEG_A": self.seg_a.chirality,
                "SEG_B": self.seg_b.chirality,
                "SEG_A_PRIME": self.seg_a_prime.chirality,
            },
            "direction": {
                "SEG_A": self.seg_a.direction,
                "SEG_B": self.seg_b.direction,
                "SEG_A_PRIME": self.seg_a_prime.direction,
            },
            "segments": {k: v.text for k, v in self.segments.items()},
        }

    @property
    def is_closure(self) -> bool:
        """Vrai si le retournement spirale canonique est respecté.

        A et B émis en DX/OUT (déploiement dextrogyre vers le dehors),
        A′ rentrant en LV/IN (réception lévogyre vers le dedans).
        """
        return (
            self.seg_a.chirality == "DX" and self.seg_a.direction == "OUT"
            and self.seg_b.chirality == "DX" and self.seg_b.direction == "OUT"
            and self.seg_a_prime.chirality == "LV" and self.seg_a_prime.direction == "IN"
        )

    @property
    def is_fixed_point(self) -> bool:
        """Cycle dégénéré : A, B, A′ portent le même texte (répétition).

        C'est le « point fixe de contrôle » du corpus (l'écho du vide) — la
        répétition que le projet oppose à la progression de la spirale.
        """
        a, b, c = self.seg_a.text, self.seg_b.text, self.seg_a_prime.text
        return a == b == c and a != ""


# --- Expressions régulières -------------------------------------------------

# Bloc de segment : <SEG_X> ... </SEG_X>. A_PRIME avant A pour matcher le long.
_SEG_RE = re.compile(r"<SEG_(A_PRIME|A|B)>(.*?)</SEG_(?:A_PRIME|A|B)>", re.S)
# En-tête de tags + texte : <OP><CHI><DIR><POS> texte…
_HEAD_RE = re.compile(
    r"<(ADD|SUB|MUL|DIV)><(DX|LV)><(OUT|IN)><(ALPHA|OMEGA|A_PRIME)>(.*)",
    re.S,
)
_ANY_TAG_RE = re.compile(r"<[A-Z_]+>")


def _clean_text(raw: str) -> str:
    """Retire les tags résiduels (<EOL>…) et normalise les espaces."""
    no_tags = _ANY_TAG_RE.sub(" ", raw)
    return re.sub(r"\s+", " ", no_tags).strip()


def is_terminator_line(line: str) -> bool:
    """Vrai pour les lignes de fin de corpus (`<EOS>`), seules ou en segment.

    Un terminateur contient ``<EOS>`` ET ne porte aucun en-tête d'opérateur
    valide ``<OP><CHI><DIR><POS>``. Cette condition évite de jeter par erreur un
    *vrai* cycle dont le texte d'un segment contiendrait la sous-chaîne
    littérale ``<EOS>`` (un tel cycle a, lui, des en-têtes valides).
    """
    return "<EOS>" in line and _HEAD_RE.search(line) is None


def try_parse_aba_line(line: str) -> Optional[AbaCycle]:
    """Parse une ligne. Retourne ``None`` pour vide/terminateur, sinon le cycle.

    Lève ``AbaParseError`` si la ligne ressemble à un cycle mais est malformée.
    """
    stripped = line.strip()
    if not stripped or is_terminator_line(stripped):
        return None
    return parse_aba_line(stripped)


def parse_aba_line(line: str) -> AbaCycle:
    """Parse strictement une ligne ABA en :class:`AbaCycle`.

    Lève :class:`AbaParseError` si la structure attendue est absente.
    """
    blocks = _SEG_RE.findall(line)
    # findall renvoie [(seg_name, inner), ...]
    found: Dict[str, str] = {}
    for seg_name, inner, *_ in blocks:
        found[f"SEG_{seg_name}"] = inner

    missing = [s for s in SEGMENTS if s not in found]
    if missing:
        raise AbaParseError(f"segments manquants {missing} dans : {line!r}")

    seg_a = _parse_segment("SEG_A", found["SEG_A"])
    seg_b = _parse_segment("SEG_B", found["SEG_B"])
    seg_ap = _parse_segment("SEG_A_PRIME", found["SEG_A_PRIME"])

    # Opérateur dominant : celui de A (constant attendu ; vérifié par validate).
    return AbaCycle(op=seg_a.op, seg_a=seg_a, seg_b=seg_b, seg_a_prime=seg_ap, raw=line)


def _parse_segment(seg_name: str, inner: str) -> AbaSegment:
    m = _HEAD_RE.search(inner)
    if not m:
        raise AbaParseError(f"en-tête de tags absent dans {seg_name} : {inner!r}")
    op, chi, direction, position, text = m.groups()
    return AbaSegment(
        segment=seg_name,
        op=op,
        chirality=chi,
        direction=direction,
        position=position,
        text=_clean_text(text),
    )


# --- Validation structurelle -----------------------------------------------

# Forme attendue de chaque segment : (chiralité, direction, position).
_EXPECTED: Dict[str, Tuple[str, str, str]] = {
    "SEG_A": ("DX", "OUT", "ALPHA"),
    "SEG_B": ("DX", "OUT", "OMEGA"),
    "SEG_A_PRIME": ("LV", "IN", "A_PRIME"),
}


def validate_cycle(cycle: AbaCycle) -> List[str]:
    """Retourne la liste des écarts au canon (vide = cycle conforme).

    Sert à MESURER la conformité d'un corpus, jamais à la forcer.
    """
    issues: List[str] = []

    # Opérateur constant sur les trois segments.
    ops = {cycle.seg_a.op, cycle.seg_b.op, cycle.seg_a_prime.op}
    if len(ops) != 1:
        issues.append(f"opérateur non constant : {sorted(ops)}")

    for seg_name, seg in cycle.segments.items():
        exp_chi, exp_dir, exp_pos = _EXPECTED[seg_name]
        if (seg.chirality, seg.direction, seg.position) != (exp_chi, exp_dir, exp_pos):
            issues.append(
                f"{seg_name} attendu {exp_chi}/{exp_dir}/{exp_pos}, "
                f"trouvé {seg.chirality}/{seg.direction}/{seg.position}"
            )
        if not seg.text:
            issues.append(f"{seg_name} sans texte")

    return issues


# --- Itération sur un corpus -----------------------------------------------

def iter_aba_cycles(path: str, *, strict: bool = False) -> Iterator[AbaCycle]:
    """Itère les cycles d'un fichier ABA, en sautant vides et terminateurs.

    strict=True : propage ``AbaParseError`` ; sinon les lignes illisibles sont
    ignorées silencieusement (à réserver à l'exploration ; préférer
    :func:`measure_corpus` pour un constat chiffré).
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or is_terminator_line(stripped):
                continue
            try:
                yield parse_aba_line(stripped)
            except AbaParseError:
                if strict:
                    raise
                continue


@dataclass(frozen=True)
class CorpusReport:
    path: str
    n_lines: int
    n_terminators: int
    n_cycles: int
    n_parse_errors: int
    n_conform: int           # cycles sans aucun écart au canon
    n_fixed_points: int
    op_counts: Dict[str, int]

    def summary(self) -> str:
        rate = (100.0 * self.n_conform / self.n_cycles) if self.n_cycles else 0.0
        dist = ", ".join(f"{op}={self.op_counts.get(op, 0)}" for op in OPERATORS)
        return (
            f"{self.path}: {self.n_cycles} cycles "
            f"({self.n_conform} conformes = {rate:.1f}%), "
            f"{self.n_parse_errors} illisibles, {self.n_terminators} terminateurs, "
            f"{self.n_fixed_points} points fixes | {dist}"
        )


def measure_corpus(path: str) -> CorpusReport:
    """Constat chiffré de conformité d'un corpus ABA (mesure, pas cible)."""
    n_lines = n_term = n_cycles = n_err = n_conform = n_fixed = 0
    op_counts: Dict[str, int] = {op: 0 for op in OPERATORS}

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            n_lines += 1
            if is_terminator_line(stripped):
                n_term += 1
                continue
            try:
                cycle = parse_aba_line(stripped)
            except AbaParseError:
                n_err += 1
                continue
            n_cycles += 1
            if cycle.op in op_counts:
                op_counts[cycle.op] += 1
            if not validate_cycle(cycle):
                n_conform += 1
            if cycle.is_fixed_point:
                n_fixed += 1

    return CorpusReport(
        path=path,
        n_lines=n_lines,
        n_terminators=n_term,
        n_cycles=n_cycles,
        n_parse_errors=n_err,
        n_conform=n_conform,
        n_fixed_points=n_fixed,
        op_counts=op_counts,
    )
