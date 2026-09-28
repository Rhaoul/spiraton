"""Vecteurs contextuels par mot (CamemBERT) — représentation externe au Logos.

Protocole : ``docs/SEMANTIC_THERMO_PROTOCOLE.md`` révision R3.

Vecteur d'un mot = moyenne, sur les sous-mots de ce mot, de la dernière couche
cachée d'un encodeur pré-entraîné (défaut ``camembert-base``), la phrase étant
encodée ENTIÈRE (contexte complet). Un « mot » est un élément de ``texte.split()``.

Dépendance OPTIONNELLE (``transformers``), importée paresseusement. Chargement
**hors ligne** par défaut (``local_files_only=True``) : aucun téléchargement
implicite. Si le modèle ou la bibliothèque manque, :class:`ContextualUnavailable`
est levée. Aucun poids n'est copié dans le dépôt.
"""
from __future__ import annotations

from typing import Optional

import torch


class ContextualUnavailable(RuntimeError):
    """Bibliothèque ``transformers`` ou poids du modèle indisponibles hors ligne."""


class ContextualWordVectors:
    """``texte → (n_mots, hidden)`` float64, un vecteur par mot de ``texte.split()``."""

    word_level = True

    def __init__(self, model_name: str = "camembert-base", *, revision: Optional[str] = None,
                 local_files_only: bool = True) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            raise ContextualUnavailable(f"transformers indisponible : {exc}") from exc
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only)
            self.model = AutoModel.from_pretrained(
                model_name, revision=revision, local_files_only=local_files_only).eval()
        except Exception as exc:  # pragma: no cover - dépend du cache local
            raise ContextualUnavailable(f"modèle {model_name!r} introuvable hors ligne : {exc}") from exc
        self.model_name = model_name
        self.hidden = int(self.model.config.hidden_size)
        self.commit = getattr(self.model.config, "_commit_hash", None)

    @torch.no_grad()
    def __call__(self, text: str) -> torch.Tensor:
        words = text.split()
        if not words:
            return torch.zeros(0, self.hidden, dtype=torch.float64)
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt",
                             truncation=True, max_length=512)
        h = self.model(**enc).last_hidden_state[0].to(torch.float64)
        word_ids = enc.word_ids(0)
        out = torch.zeros(len(words), self.hidden, dtype=torch.float64)
        cnt = torch.zeros(len(words), dtype=torch.float64)
        for pos, w in enumerate(word_ids):
            if w is not None:
                out[w] += h[pos]
                cnt[w] += 1
        if bool((cnt == 0).any()):
            raise ValueError(f"mot(s) sans sous-mot (troncature ?) dans : {text[:80]!r}")
        return out / cnt[:, None]
