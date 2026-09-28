"""Tests Tour 34 (MÉTA, second ordre) — jeton de gel MATÉRIEL pour l'ordre α-β-γ.

La danse corrige sa propre MÉTHODE. Le T33 a nommé deux défauts d'infra : horodatages
de relais faux (à ancrer sur les mtimes RÉELS des artefacts) et runner qui n'impose
pas matériellement l'ordre α-β-γ. ``freeze_token`` fournit la primitive : geler
(hash de contenu + mtime_ns), matérialiser (fichier-jeton), vérifier (rapport de
dérive, NON-FORÇANT — il rapporte, l'appelant statue).

Quatuor adapté :
  * DÉTERMINISME : à contenus/mtimes fixes, jeton bit-à-bit identique, indépendant de
    l'ordre des chemins ; round-trip write/read identique.
  * FORMES : N chemins → N artefacts triés, champs bien formés (sha256 hex 64).
  * VALEURS EXACTES : sha256/digest recalculés à la main ; jeton écrit puis vérifié
    ⇒ ok ; CONTENU ALTÉRÉ détecté (même à mtime remis en arrière — anti-spoof) ;
    MTIME POSTÉRIEUR détecté (contenu identique) ; jeton falsifié refusé.
  * NON-FORÇANT : l'import n'a aucun effet de bord ; ``verify_freeze`` ne lève JAMAIS
    sur une dérive (il rapporte).

AUCUNE horloge lue : les mtimes viennent de fichiers réels créés dans ``tmp_path``
et sont déplacés EXPLICITEMENT via ``os.utime(ns=...)`` (déterminisme total).
"""
import hashlib
import os

import pytest

from spiraton.diagnostics.freeze_token import (
    TOKEN_FORMAT_VERSION,
    ArtifactDrift,
    FreezeReport,
    FreezeToken,
    FrozenArtifact,
    freeze_token,
    read_token,
    verify_freeze,
    write_token,
)

ONE_SECOND_NS = 1_000_000_000


def _make_files(tmp_path, contents=(b"alpha\n", b"beta\n")):
    """Crée des artefacts réels et fige leurs mtimes explicitement (pas d'horloge)."""
    paths = []
    for i, blob in enumerate(contents):
        p = tmp_path / f"artefact_{i}.txt"
        p.write_bytes(blob)
        # mtime figé explicitement : base arbitraire + i s (croissant, reproductible)
        ns = 1_600_000_000 * ONE_SECOND_NS + i * ONE_SECOND_NS
        os.utime(p, ns=(ns, ns))
        paths.append(p)
    return paths


# --- DÉTERMINISME -------------------------------------------------------------

def test_deterministic_token_and_order_independence(tmp_path) -> None:
    """Mêmes contenus + mêmes mtimes ⇒ jeton identique ; l'ordre des chemins est neutre."""
    p0, p1 = _make_files(tmp_path)
    t_ab = freeze_token([p0, p1])
    t_ab2 = freeze_token([p0, p1])
    t_ba = freeze_token([p1, p0])          # ordre inversé
    t_dup = freeze_token([p0, p1, p0])     # doublon canonicalisé
    assert t_ab == t_ab2
    assert t_ab == t_ba
    assert t_ab == t_dup
    assert t_ab.digest == t_ba.digest


def test_write_read_round_trip_identical(tmp_path) -> None:
    """write_token → read_token restitue le jeton à l'identique (matérialisation fidèle)."""
    paths = _make_files(tmp_path)
    token = freeze_token(paths)
    token_file = tmp_path / "gel_beta.json"
    write_token(token, token_file)
    assert read_token(token_file) == token
    # double écriture ⇒ contenu du fichier-jeton bit-à-bit identique (JSON canonique)
    blob1 = token_file.read_bytes()
    write_token(token, token_file)
    assert token_file.read_bytes() == blob1


# --- FORMES ---------------------------------------------------------------------

def test_shapes_and_well_formed_fields(tmp_path) -> None:
    """N chemins ⇒ N artefacts, triés par chemin canonique, champs bien formés."""
    paths = _make_files(tmp_path, contents=(b"a", b"bb", b"ccc"))
    token = freeze_token(paths)
    assert isinstance(token, FreezeToken)
    assert token.format_version == TOKEN_FORMAT_VERSION
    assert len(token.artifacts) == 3
    canon = [a.path for a in token.artifacts]
    assert canon == sorted(canon)                      # tri canonique
    for a in token.artifacts:
        assert isinstance(a, FrozenArtifact)
        assert len(a.sha256) == 64 and int(a.sha256, 16) >= 0   # hex sha256
        assert isinstance(a.mtime_ns, int) and a.mtime_ns > 0
    assert len(token.digest) == 64
    rep = verify_freeze(token)
    assert isinstance(rep, FreezeReport)
    assert isinstance(rep.drifts, tuple)


def test_invalid_arguments_rejected(tmp_path) -> None:
    """Liste vide ⇒ ValueError ; fichier manquant au gel ⇒ FileNotFoundError."""
    with pytest.raises(ValueError):
        freeze_token([])
    with pytest.raises(FileNotFoundError):
        freeze_token([tmp_path / "inexistant.txt"])


# --- VALEURS EXACTES --------------------------------------------------------------

def test_exact_hash_mtime_and_digest(tmp_path) -> None:
    """sha256, mtime_ns et digest global reproduits À LA MAIN (formule exacte)."""
    (p,) = _make_files(tmp_path, contents=(b"contenu gel\n",))
    token = freeze_token([p])
    a = token.artifacts[0]
    assert a.sha256 == hashlib.sha256(b"contenu gel\n").hexdigest()
    assert a.mtime_ns == os.stat(p).st_mtime_ns
    manifest = f"freeze_token/v{TOKEN_FORMAT_VERSION}\n{a.path}\t{a.sha256}\t{a.mtime_ns}"
    assert token.digest == hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def test_written_then_verified_is_ok(tmp_path) -> None:
    """Le cas nominal β→γ : jeton écrit, artefacts intouchés ⇒ verify ok, zéro dérive."""
    paths = _make_files(tmp_path)
    token = freeze_token(paths)
    token_file = tmp_path / "gel.json"
    write_token(token, token_file)
    rep = verify_freeze(read_token(token_file))
    assert rep.ok is True
    assert rep.drifts == ()


def test_content_alteration_detected_even_with_spoofed_mtime(tmp_path) -> None:
    """CONTENU ALTÉRÉ détecté par le hash — même si le mtime est remis à la valeur gelée."""
    p0, p1 = _make_files(tmp_path)
    token = freeze_token([p0, p1])
    frozen_ns = next(a.mtime_ns for a in token.artifacts if a.path == p1.resolve().as_posix())
    p1.write_bytes(b"beta ALTERE\n")
    os.utime(p1, ns=(frozen_ns, frozen_ns))   # spoof : mtime remis en arrière
    rep = verify_freeze(token)
    assert rep.ok is False
    assert len(rep.drifts) == 1               # seul p1 dérive ; p0 reste sain
    drift = rep.drifts[0]
    assert isinstance(drift, ArtifactDrift)
    assert drift.path == p1.resolve().as_posix()
    assert drift.content_changed is True
    assert drift.touched_later is False       # le spoof de mtime ne masque PAS le hash
    assert drift.missing is False


def test_later_mtime_detected_with_identical_content(tmp_path) -> None:
    """MTIME POSTÉRIEUR détecté — retouche après gel, même à contenu bit-identique."""
    p0, p1 = _make_files(tmp_path)
    token = freeze_token([p0, p1])
    frozen_ns = next(a.mtime_ns for a in token.artifacts if a.path == p0.resolve().as_posix())
    later = frozen_ns + ONE_SECOND_NS         # +1 s EXPLICITE (aucune horloge lue)
    os.utime(p0, ns=(later, later))
    rep = verify_freeze(token)
    assert rep.ok is False
    assert len(rep.drifts) == 1
    drift = rep.drifts[0]
    assert drift.path == p0.resolve().as_posix()
    assert drift.touched_later is True
    assert drift.content_changed is False
    assert drift.missing is False


def test_missing_artifact_detected(tmp_path) -> None:
    """Artefact disparu ⇒ dérive ``missing`` (rapportée, jamais levée)."""
    p0, p1 = _make_files(tmp_path)
    token = freeze_token([p0, p1])
    p1.unlink()
    rep = verify_freeze(token)                # NE LÈVE PAS
    assert rep.ok is False
    assert [d.missing for d in rep.drifts] == [True]


def test_tampered_token_file_rejected(tmp_path) -> None:
    """Jeton falsifié (mtime_ns retouché dans le JSON) ⇒ read_token lève ValueError."""
    paths = _make_files(tmp_path)
    token = freeze_token(paths)
    token_file = tmp_path / "gel.json"
    write_token(token, token_file)
    text = token_file.read_text(encoding="utf-8")
    tampered = text.replace(str(token.artifacts[0].mtime_ns), str(token.artifacts[0].mtime_ns + 1), 1)
    assert tampered != text
    token_file.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError):
        read_token(token_file)


# --- NON-FORÇANT ------------------------------------------------------------------

def test_import_has_no_side_effects(tmp_path) -> None:
    """L'import du module ne crée rien, ne lit aucune horloge, ne gate rien.

    On (ré)importe dans un répertoire témoin vide : aucun fichier créé, et le module
    n'expose AUCUN état mutable global (uniquement constantes + types + fonctions).
    """
    import importlib

    # NB : le package exporte la FONCTION freeze_token (masque l'attribut sous-module) ;
    # import_module retourne le vrai module depuis sys.modules.
    ft = importlib.import_module("spiraton.diagnostics.freeze_token")

    before = sorted(os.listdir(tmp_path))
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        importlib.reload(ft)
    finally:
        os.chdir(cwd)
    assert sorted(os.listdir(tmp_path)) == before     # zéro effet de bord fichier
    public = [n for n in vars(ft) if not n.startswith("_")]
    assert set(public) >= {
        "freeze_token", "verify_freeze", "write_token", "read_token",
        "FreezeToken", "FreezeReport", "FrozenArtifact", "ArtifactDrift",
        "TOKEN_FORMAT_VERSION",
    }


def test_verify_never_raises_on_drift_it_reports(tmp_path) -> None:
    """NON-FORÇANT : toute dérive (contenu + mtime + disparition CUMULÉS) est
    RAPPORTÉE, jamais levée — c'est l'appelant (le point d'entrée γ) qui statue."""
    p0, p1 = _make_files(tmp_path)
    token = freeze_token([p0, p1])
    ns0 = next(a.mtime_ns for a in token.artifacts if a.path == p0.resolve().as_posix())
    p0.write_bytes(b"tout change\n")
    os.utime(p0, ns=(ns0 + ONE_SECOND_NS, ns0 + ONE_SECOND_NS))
    p1.unlink()
    rep = verify_freeze(token)                # aucune exception
    assert rep.ok is False
    assert len(rep.drifts) == 2
    by_path = {d.path: d for d in rep.drifts}
    d0 = by_path[p0.resolve().as_posix()]
    assert d0.content_changed is True and d0.touched_later is True
    assert by_path[p1.resolve().as_posix()].missing is True
