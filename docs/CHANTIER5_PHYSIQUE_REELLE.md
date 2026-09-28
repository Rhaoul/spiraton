# Chantier 5 (rejoué) — perte alpha-oméga sur physique réelle vs substitut

Script reproductible : `examples/train_real_vs_substitute.py`.
Featurizers : `PhonemeFeaturizer` (33D natif réel) vs `HashingFeaturizer` (substitut).

## Protocole

Le **même** modèle (`make_aba_predictor(33)`), aux **mêmes graines**, est entraîné
deux fois sous la perte de clôture spirale `alpha_omega_loss` (chantier 5) ; seule
**l'origine des traits** change. Corpus `dataset_aba.txt`, 2000 cycles (points
fixes exclus), 80 époques.

Rappel : la perte ne compare A′ prédit qu'à **A** (auto-supervision : A′ doit être
proche-et-aligné avec A, sans le copier). Le A′ **réel** du corpus n'entre PAS
dans l'entraînement — il sert de signal **tenu à l'écart**.

## Mesures (graine 0 ; stable sur graines 0/1/2)

| | perte init→fin | rel_dist(pred,A) | copy_rate | cos(pred, A′ réel) | **baseline identité** cos(A, A′ réel) | apport appris |
|---|---|---|---|---|---|---|
| Substitut (hachage) | 1.078 → 0.0001 | 0.300 | 0.000 | +0.153 | **+0.153** | **−0.000** |
| Réel (physique 33D) | 0.369 → 0.0021 | 0.300 | 0.000 | +0.862 | **+0.861** | **+0.001** |

La colonne **baseline identité** = `cos(A, A′ réel)` *avant tout apprentissage*
(le contrôle « renvoyer A inchangé »). La dernière colonne (held-out − identité)
est ce que l'apprentissage ajoute **réellement** vers A′ : ≈ 0 dans les deux cas.
Stabilité sur graines 0/1/2 : substitut +0.153 (×3) ; réel +0.862/+0.861/+0.862.

## Lectures (mesure, jamais cible)

1. **Les deux signaux alpha-oméga sont apprenables.** Chacun atteint la bande de
   distance visée (rel_dist ≈ 0.30) avec un taux de copie nul — le mode dégénéré
   (répétition) est évité dans les deux cas. Sans surprise : la perte est
   auto-supervisée sur A et le modèle (MLP) sait produire « A transformé ».

2. **La perte absolue n'est pas comparable** : le réel démarre plus bas (0.37 vs
   1.08) parce que ses vecteurs gardent l'échelle sémantique des dims 0-7 (A et B
   d'un même cycle partagent l'opérateur et une phonétique proche → déjà plus
   alignés à l'init), tandis que le substitut L2-normalise un hachage quasi
   aléatoire. C'est une différence de **géométrie d'entrée**, pas de qualité
   d'apprentissage.

3. **Résultat saillant — mais ce n'est PAS une clôture apprise.** Le A′ prédit
   est à cos ≈ **+0.86** du A′ humain en réel, contre ≈ **+0.15** en substitut.
   Tentant d'y voir un apprentissage… sauf que la **baseline identité**
   `cos(A, A′ réel)` vaut elle aussi **+0.861** en réel : *renvoyer A inchangé*
   atteint le même score contre A′. L'apprentissage de clôture n'ajoute donc
   **rien** vers A′ (écart held-out − identité = +0.001). Ce que mesure
   réellement le contraste 0.86 vs 0.15, c'est la **qualité des traits** : en
   réel, A, B et A′ d'un même cycle vivent dans le même cône (corrélés par la
   physique) ; en haché, ce sont des vecteurs décorrélés. C'est une propriété de
   l'**entrée**, pas de la dynamique apprise.

## Honnêteté sur le mécanisme

Ce n'est **pas** une « compréhension sémantique ». L'effet vient de ce que, avec
les traits réels, les trois segments d'un même cycle sont **corrélés par
construction** : même opérateur dominant (dims 0-3 issues de la même physique),
mots de registre proche, donc impédance/flux voisins. Le modèle, en apprenant
« A′ ≈ A décalé de 0.3 », tombe mécaniquement près d'un A′ qui partage cette
structure. Le substitut détruit cette corrélation (hachage de bigrammes), d'où
l'écart. C'est exactement la propriété qu'on attend de *bons* traits — mais on la
nomme pour ce qu'elle est : une corrélation intra-cycle portée par la physique,
pas une inférence de sens.

Aucun réglage n'a été fait pour produire cet écart : il sort tel quel du signal
tenu à l'écart. C'est un argument empirique en faveur du chantier 7 (les traits
mesurés portent du sens exploitable), à reconduire — jamais à forcer — sur
d'autres corpus et architectures.
