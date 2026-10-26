"""Jeton de gel MATÉRIEL pour l'ordre α → β → γ (Tour 34, méta).

MÉTA-TOUR de second ordre : la danse corrige sa propre MÉTHODE. Le T33 a validé la
discipline α (exploration) → β (gel écrit) → γ (one-shot) mais a nommé DEUX défauts
d'infrastructure : (1) les horodatages du relais étaient FAUX (ancrés sur l'horloge
perçue en fin de job, pas sur les artefacts) — l'audit d'ordre n'a tenu que parce que
l'ingénieur est allé lire les mtimes réels ; (2) le runner n'IMPOSAIT pas l'ordre
α-β-γ — rien n'empêchait matériellement de retoucher un cartouche après le premier
calcul de validation.

CE MODULE FOURNIT LA PRIMITIVE, PAS LE GATE. Il rend le gel MATÉRIEL :

  * ``freeze_token(paths)`` construit un jeton = manifeste des artefacts gelés
    (sha256 du CONTENU + mtime_ns RÉEL du filesystem, par chemin) scellé par un
    digest global. Aucune horloge perçue : seuls les mtimes des artefacts comptent.
  * ``verify_freeze(token)`` re-lit les artefacts et RAPPORTE la dérive éventuelle :
    contenu altéré (hash différent), retouche postérieure (mtime_ns > gelé), fichier
    disparu. Il ne lève JAMAIS sur une dérive : il rapporte, l'appelant statue.
  * ``write_token`` / ``read_token`` matérialisent le jeton sur disque (JSON trié) ;
    le mtime DU FICHIER-JETON devient l'horodatage matériel du gel β. ``read_token``
    re-scelle le digest et refuse un jeton falsifié (intégrité du jeton lui-même).

NON-FORÇANT (REFUS : transparence, jamais forcer). Le helper n'exécute rien, ne
bloque rien, n'altère aucun verdict tout seul : un futur runner α-β-γ CHOISIT
d'appeler ``verify_freeze`` avant son one-shot γ (recommandation T33 : deux points
d'entrée, le second exigeant le jeton). Parallèle exact du contrat
``instrument_validation.assert_order_sensitive`` (T23) : mesurer et rapporter.

DÉTERMINISTE. Aucune horloge lue, aucun aléa : à contenus et mtimes identiques, le
jeton est bit-à-bit identique (les chemins sont canonicalisés en absolu POSIX).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, Union

_PathLike = Union[str, Path]

#: Version du format de jeton (incluse dans le digest : un vieux jeton ne se
#: vérifie pas silencieusement contre un format futur).
TOKEN_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FrozenArtifact:
    """Un artefact gelé : chemin absolu POSIX, sha256 du contenu, mtime_ns réel."""

    path: str
    sha256: str
    mtime_ns: int


@dataclass(frozen=True)
class ArtifactDrift:
    """Dérive d'UN artefact depuis le gel (un seul rapporté par chemin dérivé).

    * ``missing``         : le fichier a disparu.
    * ``content_changed`` : sha256 du contenu ≠ sha256 gelé (même si le mtime a été
      remis en arrière — le contenu fait foi, pas l'horodatage).
    * ``touched_later``   : mtime_ns actuel > mtime_ns gelé (retouche postérieure,
      même si le contenu est revenu à l'identique).
    """

    path: str
    missing: bool
    content_changed: bool
    touched_later: bool


@dataclass(frozen=True)
class FreezeToken:
    """Jeton de gel : manifeste d'artefacts + digest global scellant le manifeste."""

    format_version: int
    artifacts: Tuple[FrozenArtifact, ...]
    digest: str


@dataclass(frozen=True)
class FreezeReport:
    """Rapport de ``verify_freeze`` : ``ok`` ⇔ aucun chemin dérivé. NON-FORÇANT."""

    ok: bool
    drifts: Tuple[ArtifactDrift, ...]


def _canonical_path(path: _PathLike) -> str:
    return Path(path).resolve().as_posix()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_digest(format_version: int, artifacts: Sequence[FrozenArtifact]) -> str:
    """Digest global = sha256 des lignes canoniques ``path\\tsha256\\tmtime_ns``."""
    lines = [f"freeze_token/v{format_version}"]
    lines += [f"{a.path}\t{a.sha256}\t{a.mtime_ns}" for a in artifacts]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def freeze_token(paths: Sequence[_PathLike]) -> FreezeToken:
    """Gèle MATÉRIELLEMENT une liste d'artefacts : hash de contenu + mtime_ns réels.

    Les chemins sont canonicalisés (absolus POSIX) et TRIÉS : le jeton ne dépend pas
    de l'ordre d'appel (déterminisme). Fichier manquant ⇒ ``FileNotFoundError``
    (on ne gèle pas du vide) ; liste vide ⇒ ``ValueError``.
    """
    if len(paths) == 0:
        raise ValueError("freeze_token exige au moins un artefact (gel de rien = symbole sans test)")
    canon = sorted({_canonical_path(p) for p in paths})
    artifacts = tuple(
        FrozenArtifact(path=p, sha256=_sha256_file(p), mtime_ns=os.stat(p).st_mtime_ns)
        for p in canon
    )
    return FreezeToken(
        format_version=TOKEN_FORMAT_VERSION,
        artifacts=artifacts,
        digest=_manifest_digest(TOKEN_FORMAT_VERSION, artifacts),
    )


def verify_freeze(token: FreezeToken) -> FreezeReport:
    """Vérifie le gel : re-lit chaque artefact et RAPPORTE toute dérive. Ne lève pas.

    ``ok=True`` ⇔ chaque artefact existe, contenu bit-identique (sha256) ET
    mtime_ns ≤ mtime_ns gelé. ``drifts`` ne contient QUE les chemins dérivés.
    L'appelant (ex. le point d'entrée γ d'un runner) statue ; le helper rapporte.
    """
    drifts = []
    for a in token.artifacts:
        if not os.path.exists(a.path):
            drifts.append(ArtifactDrift(a.path, missing=True, content_changed=False, touched_later=False))
            continue
        content_changed = _sha256_file(a.path) != a.sha256
        touched_later = os.stat(a.path).st_mtime_ns > a.mtime_ns
        if content_changed or touched_later:
            drifts.append(ArtifactDrift(a.path, missing=False, content_changed=content_changed, touched_later=touched_later))
    return FreezeReport(ok=(len(drifts) == 0), drifts=tuple(drifts))


def write_token(token: FreezeToken, path: _PathLike) -> None:
    """Écrit le jeton en JSON canonique (clés triées, déterministe).

    Le mtime du fichier écrit est l'HORODATAGE MATÉRIEL du gel (leçon T33 : ancrer
    sur les mtimes des artefacts, jamais sur l'horloge perçue d'un relais).
    """
    payload = {
        "format_version": token.format_version,
        "digest": token.digest,
        "artifacts": [
            {"path": a.path, "sha256": a.sha256, "mtime_ns": a.mtime_ns}
            for a in token.artifacts
        ],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")


def read_token(path: _PathLike) -> FreezeToken:
    """Relit un jeton et RE-SCELLE son digest : un jeton falsifié ⇒ ``ValueError``.

    (Seule levée du module côté vérification : elle protège l'intégrité du JETON
    lui-même — pas un gate sur les artefacts, que seul ``verify_freeze`` rapporte.)
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    artifacts = tuple(
        FrozenArtifact(path=a["path"], sha256=a["sha256"], mtime_ns=int(a["mtime_ns"]))
        for a in payload["artifacts"]
    )
    token = FreezeToken(
        format_version=int(payload["format_version"]),
        artifacts=artifacts,
        digest=str(payload["digest"]),
    )
    expected = _manifest_digest(token.format_version, token.artifacts)
    if token.digest != expected:
        raise ValueError(
            f"jeton de gel falsifié : digest {token.digest} != manifeste {expected}"
        )
    return token
