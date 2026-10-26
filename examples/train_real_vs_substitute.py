"""Chantier 5 rejoué — perte alpha-oméga sur PHYSIQUE RÉELLE vs SUBSTITUT.

But : MESURER si le signal de clôture spirale (A′ proche-et-aligné avec A, non
identique) s'apprend différemment quand les traits d'entrée portent la physique
articulatoire réelle (`PhonemeFeaturizer`, tokenizer 33D natif) plutôt que le
substitut de hachage (`HashingFeaturizer`).

C'est une **mesure à rapporter, jamais une cible à atteindre**. On n'ajuste rien
pour « faire gagner » le réel : on entraîne deux fois le MÊME modèle, aux MÊMES
graines, et on consigne ce qui sort.

Mise en garde de lecture : les deux featurizers produisent des géométries
d'entrée différentes (le substitut est L2-normalisé ; le réel garde l'échelle
sémantique des dims 0-7). Les pertes ABSOLUES ne sont donc pas directement
comparables. Ce qui l'est : la *forme* de la convergence, le taux de copie
(mode dégénéré), et si la bande de distance / l'alignement visés sont atteints.

Nécessite le tokenizer natif (`make lib`). Usage :
    PYTHONPATH=. python examples/train_real_vs_substitute.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from spiraton.data.featurizers import HashingFeaturizer, PhonemeFeaturizer
from spiraton.data.loader import load_aba_triples
from spiraton.data.tokenizer_bridge import is_available
from spiraton.training import train_aba, make_aba_predictor


def _resolve(name: str) -> Path | None:
    repo = Path(__file__).resolve().parents[1]
    for base in (repo.parent, repo, repo / "data", Path.cwd()):
        p = base / name
        if p.is_file():
            return p
    return None


def _holdout_vs_true_aprime(model, triples):
    """A′ prédit (appris par clôture sur A SEUL) vs le A′ HUMAIN réel du corpus.

    La perte d'entraînement ignore le A′ du corpus (auto-supervision sur A).
    Ce A′ réel est donc un signal *tenu à l'écart* : mesurer cos/dist(pred, A′)
    teste si les traits portent assez de sens pour que la clôture apprise
    atterrisse près de ce qu'un humain a écrit comme retour transformé.
    """
    with torch.no_grad():
        pred = model(triples.a, triples.b)
        true_ap = triples.a_prime
        cos = torch.nn.functional.cosine_similarity(pred, true_ap, dim=-1, eps=1e-8)
        rel = (pred - true_ap).norm(dim=-1) / (true_ap.norm(dim=-1) + 1e-8)
    return float(cos.mean()), float(rel.mean())


def _identity_baseline(triples) -> float:
    """cos(A, A′_réel) BRUT — le contrôle « renvoyer A inchangé » (identité).

    C'est la référence décisive : si le held-out cos(pred, A′) ne dépasse pas ce
    nombre, l'apprentissage de clôture n'apporte RIEN vers A′ au-delà de la
    corrélation intra-cycle déjà présente dans les traits d'entrée.
    """
    with torch.no_grad():
        return float(torch.nn.functional.cosine_similarity(
            triples.a, triples.a_prime, dim=-1, eps=1e-8).mean())


def _run(label: str, featurizer, corpus: str, *, dim: int, limit: int, seed: int):
    triples = load_aba_triples(str(corpus), featurizer, limit=limit, skip_fixed_points=True)
    id_cos = _identity_baseline(triples)  # AVANT tout apprentissage
    torch.manual_seed(seed)  # même init de modèle pour les deux featurizers
    model = make_aba_predictor(dim)
    report = train_aba(model, triples, epochs=80, lr=1e-2, batch_size=256, target_dist=0.3)
    m = report.final_metrics
    cos_true, rel_true = _holdout_vs_true_aprime(model, triples)
    print(f"\n[{label}]  cycles={len(triples)}  substitut={triples.featurizer_is_fallback}")
    print(f"  perte {report.loss_history[0]:.4f} → {report.loss_history[-1]:.4f}")
    print(f"  align(pred,A)={m['align']:.4f}  rel_dist(pred,A)={m['mean_rel_dist']:.3f} "
          f"(cible {m['target_dist']:.2f})  copy_rate={m['copy_rate']:.3f}")
    print(f"  → cos(pred, A′_réel)={cos_true:+.3f}   vs   "
          f"BASELINE identité cos(A, A′_réel)={id_cos:+.3f}")
    print(f"    (held-out − identité = {cos_true - id_cos:+.3f} : "
          f"ce que l'apprentissage ajoute VRAIMENT vers A′)")
    return report, cos_true, id_cos


def main() -> None:
    if not is_available():
        print("Tokenizer natif indisponible — compiler `make lib` dans le dépôt Tokenizer.")
        return

    corpus = _resolve("dataset_aba.txt") or _resolve("corpus_claude_aba.txt")
    if corpus is None:
        print("Aucun corpus ABA trouvé.")
        return

    dim, limit, seed = 33, 2000, 0
    print(f"Corpus : {corpus.name}  (limit={limit}, dim={dim}, seed={seed})")
    print("Deux entraînements identiques ; seule l'origine des traits change.")

    sub, sub_cos, sub_id = _run("SUBSTITUT (hachage)", HashingFeaturizer(dim=dim), corpus,
                                dim=dim, limit=limit, seed=seed)
    real, real_cos, real_id = _run("RÉEL (physique 33D)", PhonemeFeaturizer(), corpus,
                                   dim=dim, limit=limit, seed=seed)

    print("\n== Lecture (mesure, pas verdict) ==")
    print(f"  Signaux alpha-oméga apprenables ? substitut={sub.improved}, réel={real.improved}.")
    print("  Ce que MESURE vraiment le cos au A′ humain :")
    print(f"    réel      : held-out={real_cos:+.3f} ≈ identité={real_id:+.3f} "
          f"(écart {real_cos-real_id:+.3f})")
    print(f"    substitut : held-out={sub_cos:+.3f} ≈ identité={sub_id:+.3f} "
          f"(écart {sub_cos-sub_id:+.3f})")
    print("  → Le contraste réel/substitut (≈0.86 vs ≈0.15) mesure la QUALITÉ DES")
    print("    TRAITS (corrélation intra-cycle), PAS une clôture apprise : renvoyer A")
    print("    inchangé fait aussi bien contre A′. L'apprentissage colle pred à A ;")
    print("    il n'ajoute rien vers A′ au-delà de ce que les traits portaient déjà.")
    print("  On consigne ces nombres ; on n'ajuste rien pour les améliorer.")


if __name__ == "__main__":
    main()
