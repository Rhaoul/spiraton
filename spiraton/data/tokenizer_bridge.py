"""Pont vers le tokenizer 33D (dépôt spiraton-tokenizer).

Le tokenizer C produit, par token, un vecteur 33D. Ce module charge le wrapper
ctypes ``SpiratonTokenizerV4`` du dépôt voisin et expose une interface stable
``vectors(text) -> (N, 33) float32``, avec **résolution de chemin robuste** :

  1. variable d'environnement ``SPIRATON_TOKENIZER_LIB`` (chemin du .so/.dll) ;
  2. variable ``SPIRATON_TOKENIZER_PY`` (dossier du package python du tokenizer) ;
  3. emplacements relatifs usuels au layout des deux dépôts côte à côte.

Si rien n'est trouvable (cas fréquent : pas de compilateur C sur la machine),
``load_native_tokenizer`` lève ``TokenizerUnavailable`` avec un message
actionnable. Les tests de parité se *skippent* alors proprement — le pipeline
aval (embeddings, entraînement) peut toujours être développé sur les dims 0-5
issues des étiquettes ABA (qui, elles, ne dépendent pas du .so).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import vector33d


class TokenizerUnavailable(RuntimeError):
    """Le tokenizer natif 33D n'a pu être ni localisé ni chargé."""


_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../spiraton (dépôt PyTorch)


def _candidate_py_dirs() -> List[Path]:
    """Dossiers où chercher le package python du tokenizer (spiraton_tokenizer)."""
    cands: List[Path] = []
    env = os.environ.get("SPIRATON_TOKENIZER_PY")
    if env:
        cands.append(Path(env))
    # Layouts usuels : dépôts côte à côte.
    parent = _REPO_ROOT.parent
    for name in ("Tokenizer", "spiraton-tokenizer", "spiraton_tokenizer"):
        cands.append(parent / name / "python")
    return [c for c in cands if c.is_dir()]


def load_native_tokenizer(lib_path: Optional[str] = None):
    """Charge et retourne une instance de ``SpiratonTokenizerV4``.

    lib_path : chemin explicite du .so/.dll (prioritaire). Sinon, repli sur
        ``SPIRATON_TOKENIZER_LIB`` puis sur le chemin par défaut du wrapper.

    Lève :class:`TokenizerUnavailable` si le package python ou la lib native
    sont introuvables/illisibles.
    """
    py_dirs = _candidate_py_dirs()
    module = None
    last_err: Optional[Exception] = None

    for py_dir in py_dirs:
        mod_path = py_dir / "spiraton_tokenizer" / "spiraton_v4.py"
        if not mod_path.is_file():
            continue
        try:
            if str(py_dir) not in sys.path:
                sys.path.insert(0, str(py_dir))
            spec = importlib.util.spec_from_file_location(
                "spiraton_tokenizer.spiraton_v4", mod_path
            )
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            break
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            last_err = exc
            continue

    if module is None:
        raise TokenizerUnavailable(
            "Package python du tokenizer introuvable. Cherché dans : "
            f"{[str(p) for p in py_dirs] or '(aucun dossier candidat)'}. "
            "Définir SPIRATON_TOKENIZER_PY (dossier .../python) "
            "ou placer le dépôt tokenizer à côté de celui-ci."
        )

    resolved_lib = lib_path or os.environ.get("SPIRATON_TOKENIZER_LIB")
    try:
        cls = getattr(module, "SpiratonTokenizerV4")
        return cls(resolved_lib) if resolved_lib else cls()
    except Exception as exc:
        raise TokenizerUnavailable(
            f"Lib native non chargeable ({type(exc).__name__}: {exc}). "
            "Compiler le tokenizer (make lib) puis définir SPIRATON_TOKENIZER_LIB "
            "vers le .so/.dll si le chemin par défaut ne convient pas."
        ) from exc


class NativeTokenizer33D:
    """Adaptateur stable autour de ``SpiratonTokenizerV4``.

    Expose uniquement ce dont les cellules ont besoin : la matrice des vecteurs
    33D. Centralise la vérification de dimension (l'ABI 33D est un contrat).
    """

    def __init__(self, lib_path: Optional[str] = None) -> None:
        self._tok = load_native_tokenizer(lib_path)

    def set_heuristic_mode(self, enable: bool = True) -> None:
        self._tok.set_heuristic_mode(enable)

    def vectors(self, text: str, max_tokens: int = 128):
        """Retourne un ``np.ndarray`` (N, 33) float32 des vecteurs des tokens."""
        import numpy as np

        toks = self._tok.tokenize(text, max_tokens=max_tokens)
        if not toks:
            return np.zeros((0, vector33d.DIM), dtype=np.float32)
        rows = []
        for t in toks:
            v = np.asarray(t["vector33d"], dtype=np.float32)
            vector33d.check_dim(v)
            rows.append(v)
        return np.stack(rows, axis=0)


def import_aba_emitter():
    """Importe le module ``aba_emitter`` du tokenizer (émetteur ABA, chantier boucle).

    Insère le dossier python du tokenizer dans ``sys.path`` puis importe le
    module. **Ne nécessite pas la lib native** : l'émetteur travaille sur un
    objet tokenizer qu'on lui fournit (et l'import du package ne charge pas le
    ``.so`` — le ``CDLL`` n'arrive qu'à la construction de ``SpiratonTokenizerV4``).
    Cela permet de tester la logique d'émission/round-trip sans compilateur C.

    Lève :class:`TokenizerUnavailable` si le module est introuvable.
    """
    for d in _candidate_py_dirs():
        if (d / "spiraton_tokenizer" / "aba_emitter.py").is_file():
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
            return importlib.import_module("spiraton_tokenizer.aba_emitter")
    raise TokenizerUnavailable(
        "Module aba_emitter introuvable. Cherché le package python du tokenizer "
        "(définir SPIRATON_TOKENIZER_PY ou placer le dépôt tokenizer à côté)."
    )


def is_available() -> bool:
    """True si le tokenizer natif peut être chargé (sinon False, sans lever)."""
    try:
        load_native_tokenizer()
        return True
    except TokenizerUnavailable:
        return False
