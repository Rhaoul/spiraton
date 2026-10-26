"""Chantier 7 — les traits phonétiques RÉELS comme canaux d'entrée des cellules.

Démonstration de bout en bout : un texte traverse le tokenizer 33D natif, ses
signatures articulatoires (dims 8-22 : impédance, flux, spins) sont extraites
par :class:`PhonemeFeaturizer`, puis routées par
:class:`OperatorEmbedding(source="phoneme")` selon l'opérateur/chiralité
(dims 0-5). On contraste avec le substitut de hachage pour rendre tangible ce
que le chantier 7 apporte : une distinction *mesurée* plutôt qu'arbitraire.

Nécessite le tokenizer natif compilé (``make lib`` dans le dépôt Tokenizer).

Usage : PYTHONPATH=. python examples/phoneme_channels.py
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from spiraton.data import vector33d
from spiraton.data.featurizers import HashingFeaturizer, PhonemeFeaturizer
from spiraton.data.tokenizer_bridge import is_available
from spiraton.experimental.operator_embedding import OperatorEmbedding


def _fmt(vec, n=8):
    if isinstance(vec, torch.Tensor):
        vec = vec.detach()
    return "[" + " ".join(f"{float(x):+.2f}" for x in vec[:n]) + (" …]" if len(vec) > n else "]")


def main() -> None:
    if not is_available():
        print("Tokenizer natif indisponible — compiler `make lib` dans le dépôt Tokenizer.")
        print("(Le pipeline reste développable sur HashingFeaturizer, mais sans physique.)")
        return

    feat = PhonemeFeaturizer()
    mots = ["Joie", "Mort", "spirale", "amour"]

    print("== Signatures phonémiques réelles (dims 8-22 = impédance | flux | spins) ==\n")
    for m in mots:
        sig = feat.signature(m)
        imp, flux, spins = sig[0:5], sig[5:10], sig[10:15]
        print(f"  {m:>8s}  impédance={_fmt(imp,5)}  flux={_fmt(flux,5)}  spins={_fmt(spins,5)}")

    # Distinction mesurée Joie vs Mort dans la bande phonémique.
    d = torch.norm(feat.signature("Joie") - feat.signature("Mort")).item()
    print(f"\n  ‖sig(Joie) − sig(Mort)‖ = {d:.3f}  (physique articulatoire distincte)\n")

    print("== Routage par OperatorEmbedding(source='phoneme') ==\n")
    torch.manual_seed(0)
    emb = OperatorEmbedding(embed_dim=6, source="phoneme")
    for m in mots:
        v33 = feat(m)                         # 33D réel agrégé
        op = vector33d.op_from_vector(v33)
        chi = vector33d.chirality_from_vector(v33)
        e = emb(v33)
        print(f"  {m:>8s}  op={op:>3s} chi={chi}  embedding={_fmt(e,6)}")

    print("\n== Chaîne complète : sequence → OperatorEmbedding → SpiratonCell ==\n")
    from spiraton.core.cell import SpiratonCell
    embed_dim = 8
    emb_seq = OperatorEmbedding(embed_dim=embed_dim, source="phoneme")
    cell = SpiratonCell(input_size=embed_dim)
    phrase = "la spirale revient"
    seq = feat.sequence(phrase)               # (N, 33) — ORDRE des tokens préservé
    tok_emb = emb_seq(seq)                     # (N, embed_dim) — un embedding par token
    y = cell(tok_emb)                          # (N,) — une activation de cellule par token
    print(f"  « {phrase} » → {seq.shape[0]} tokens")
    print(f"    sequence {tuple(seq.shape)} → embeddings {tuple(tok_emb.shape)} "
          f"→ cellule {tuple(y.shape)}")
    print(f"    activations cellule (token par token) = {_fmt(y, seq.shape[0])}")
    print("    Les traits articulatoires réels sont bien les canaux d'entrée de la")
    print("    cellule, dans l'ordre des phonèmes/tokens (pas un profil moyen).")

    print("\n== Réel vs substitut (le sens du chantier 7) ==\n")
    hashed = HashingFeaturizer(dim=33)
    vh_joie, vh_mort = hashed("Joie"), hashed("Mort")
    vp_joie, vp_mort = feat("Joie"), feat("Mort")
    print(f"  substitut  : is_fallback={hashed.is_fallback}  "
          f"sig(8-22) Joie={_fmt(vh_joie[vector33d.PHONEME_SIG],5)}")
    print(f"  réel (C)   : is_fallback={feat.is_fallback}  "
          f"sig(8-22) Joie={_fmt(vp_joie[vector33d.PHONEME_SIG],5)}")
    print("\n  Le substitut met dans la bande 8-22 des valeurs de hachage ARBITRAIRES")
    print("  (sans rapport articulatoire) ; le réel y met la physique MESURÉE par le")
    print("  socle C (impédance, flux, spins). Même rang (33), sens opposé.")


if __name__ == "__main__":
    main()
