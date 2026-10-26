# Chantier OPTIMISEUR_ABA — résultats (remplacer AdamW par la logique A → B → A′)

Date : 2026-10-26 · Claude · spécification : `docs/CHANTIER_OPTIMISEUR_ABA.md`
(conservée telle que livrée) · code : `spiraton/experimental/chrono_optimizer.py`,
`spiraton/experimental/spiral_optimizer.py` · bancs : `examples/benchmark_chrono_opt_vs_adamw.py`,
`examples/benchmark_chrono_opt_tasks.py`.

## 0. Verdict en une ligne

**Aucune des deux versions ne bat AdamW sur une tâche ordinaire. Leur seul avantage
mesuré, c'est la robustesse aux pics de gradient sans `clip_grad_norm_`, et AdamW la
retrouve avec le clip.** Le banc T64 réel (où l'hypothèse de stabilité H_stab
devrait vraiment se jouer) reste à lancer sur la machine de Ra (§5).

## 1. La spec v0 telle que livrée : quatre défauts mesurés

`ChronoSpiratonOpt(legacy_v0=True)` reproduit le code exact du §3 de la spec. Chaque
défaut est démontré par un test (`tests/test_chrono_optimizer.py`, `test_correction*`).

| # | Défaut v0 | Mesure | Correction v1 |
|---|---|---|---|
| 1 | Terme B `b·sign(x)·tanh(log x²)` ≈ `−b·sign(x)` dès \|x\| < 1 : il retourne le signe du pas | premier pas : cos(Δθ, −g) = **−0,76** ; quadratique ½‖x−x*‖² : 56,8 → 58,1 en 2000 pas (Adam : → 0) | `b·x·tanh(log x²)` : gain par coordonnée dans (1−b, 1+b), signe préservé |
| 2 | ρ̂ compare ‖s_cur‖ à sa propre norme stockée au pas précédent | ρ̂ ≡ 1 au bit, g ≡ 1,0 sur tout l'horizon | ρ̂ = ‖s_t‖/‖s_{t−1}‖ |
| 3 | Boucle ω de signe positif : ω est un *rappel* vers θ_{t−W}, il augmente cos, donc −cos décroît en ω | ω sature à 0,314 | observable +cos (croissant en ω, la règle de stabilité même du T20) |
| 4 | Gradient normalisé par ‖g‖ : coordonnées ~1/√n | pas par coordonnée 4096× plus petit qu'Adam sur un tenseur 4096×4096 | normalisation RMS g·√n/‖g‖ |

Deux des sept tests fournis échouaient aussi sur la v0 : le décalage d'un pas de
`c_outside` (depuis le repos, C(s_{t−1}) n'agit qu'à partir du pas 3, pas du pas 2)
et g figé (défaut 2). Ils sont corrigés et passent en v0 comme en v1 quand ils ne
dépendent pas des défauts.

**Constat de fond sur ω.** Corrigée, la boucle ω se ramène d'elle-même à ω_min = 0 :
dans l'espace des poids, cos(θ_t, θ_{t−W}) ≈ 1 > 0,7, il n'y a rien à rappeler. La
cible 0,7 vient de l'oscilloscope (T20), où ω est une vitesse de rotation. Elle n'a
pas d'équivalent ici. L'anneau de W copies de θ qu'elle exige coûte 10× la taille
du modèle.

## 2. `SpiralClosureOpt` : une conception libre

La spirale vit au niveau des **cycles** et non des pas. Chaque cycle est une forme F0 :

* **SEG_A (+)** : ancrage φ (poids lents) ;
* **SEG_B (+)** : k = 5 pas d'excursion des poids rapides, avec l'action dextrogyre
  bornée `tanh(m̂/√v̂)`, soit le rapport « mul − div » en log-espace de `cell.py` ;
* **SEG_A′ (−)** : retour transformé `φ ← φ + α(θ − φ)`, `θ ← φ`, avec 0 < α ≤ 1.
  α = 0 serait la copie (F5−), α = 1 l'excursion sans retour (F5+, c'est-à-dire AdamW).

Le gain de bord est régulé par `regulate_step` (importé, Porte 0) sur le rapport des
rayons de deux excursions successives ‖Δ_k‖/‖Δ_{k−1}‖, avec la cible 1,0. **Pivot :**
α = 1, k = 1, sans borne, η = 0 donne une trajectoire égale **au bit** à
`torch.optim.AdamW` (test). Mémoire : 3 tampons (m, v, φ), contre 2 pour AdamW.

## 3. Banc T64 en mode synthétique (24 runs × 7 optimiseurs)

`python examples/benchmark_chrono_opt_vs_adamw.py --synthetic`. Les données sont des
paires 19D a′ = R·a + bruit, et M1 est émulé (cos − l2). Parité G0 : PASS, la boucle
du banc reproduit `return_training.train_run` au bit.

| Optimiseur | Divergences | Descente ≥ 20 % | Δ_S val. (médiane) |
|---|---|---|---|
| adam_clipped_t64 (référence) | 0/24 | 24/24 | +0,7181 |
| adamw_clipped | 0/24 | 24/24 | +0,7181 |
| adamw_unclipped | 0/24 | 24/24 | +0,7152 |
| chrono_opt_c_outside (v1) | 0/24 | 24/24 | +0,7172 |
| chrono_opt_c_inside (v1) | 0/24 | 24/24 | +0,7174 |
| chrono_opt_v0_spec | 0/24 | 24/24 | **−1,0673** |
| spiral_closure | 0/24 | 24/24 | +0,7182 |

Comparaison appariée cycle par cycle (sign-test exact, cartouche S, n = 128) :
`spiral_closure − adamw_clipped` donne une médiane de −0,005, 25+/103−, p = 2·10⁻¹².
`chrono v1 − adamw_clipped` donne −0,004, 31+/97−, p = 4·10⁻⁹. L'écart est minuscule
mais **systématique en faveur d'Adam clippé**. Face à AdamW non clippé, les deux sont
à égalité ou meilleurs (S : p ≈ 0,3 à 0,7 ; P : avantage, p < 0,01).

Limite : ces données synthétiques ne font diverger personne, AdamW sans clip compris.
L'hypothèse H_stab n'y est donc pas testée.

## 4. Trois tâches classiques (`examples/benchmark_chrono_opt_tasks.py`)

Le lr est choisi par optimiseur sur {1e-3, 3e-3, 1e-2, 3e-2} d'après la perte de
validation moyenne. La perte TEST est lue au lr retenu, moyenne sur 3 graines.
C'est un repérage, pas un verdict.

| Optimiseur | Spirales (exactitude) | Mal conditionnée, κ = 1e3 (MSE test, plancher ≈ 1e-4) | Pics ×1e4, sans clip (MSE test) |
|---|---|---|---|
| adam | 1,000 | **1,02e-4** | 9,40 |
| adamw | 1,000 | 1,15e-4 | 9,13 |
| adamw_clip (clip 1,0) | 1,000 | 1,09e-4 | 0,298 |
| spiral | 1,000 | 1,11e-4 | 0,921 |
| spiral_sans_retour (tanh seul) | 0,999 | 1,16e-4 | 4,34 |
| spiral_sans_borne (retour seul) | 1,000 | 1,10e-4 | 1,15 |
| chrono_v1 | 1,000 | 1,51e-4 | **0,272** |
| chrono_v1_wd | 1,000 | 1,99e-4 | 0,275 |
| chrono_v1_sans_regulation (η_ρ = η_ω = ω0 = 0) | 1,000 | 1,45e-4 | 0,272 |
| chrono_v0_spec | 0,632 | 4,76 | 0,496 |

Lecture :

* **Tâches ordinaires.** Les spirales saturent (≥ 99,9 % pour tous sauf v0). Sur le
  problème mal conditionné, Adam reste le meilleur et chrono v1 est 50 % au-dessus du
  plancher : la normalisation par tenseur entier perd l'adaptation coordonnée par
  coordonnée. SpiralClosureOpt est à égalité avec AdamW.
* **Pics de gradient.** Sans clip, Adam et AdamW sont 30× moins bons que chrono v1,
  qui est insensible à l'amplitude du gradient. Mais **AdamW avec clip (0,298) fait
  presque aussi bien que chrono v1 (0,272)**, à graines chevauchantes (0,28 à 0,32
  contre 0,26 à 0,29). L'avantage est celui d'un clip intégré, pas d'une meilleure
  descente. Pour SpiralClosureOpt, c'est le retour A′ qui apporte la robustesse
  (0,92 contre 4,34 sans retour), puis la borne tanh (0,92 contre 1,15 sans borne).
* **Les régulateurs g et ω n'apportent aucun gain mesurable.** chrono v1 avec et
  sans régulation sont identiques sur les pics (0,2723 contre 0,2724) et à 4 % près
  sur le cas mal conditionné, en faveur de la version *sans*. Sur les spirales, ils
  ont la même exactitude, mais la perte d'entropie croisée est plus basse sans
  régulation (6e-7 contre 5e-4). Sur une tâche saturée, c'est une mesure de marge,
  pas de généralisation.

## 5. Ce qui reste à mesurer, et la suite proposée

1. **Le banc T64 réel**, sur la machine de Ra (dans `spiraton-enhanced`, où se
   trouvent `dataset_aba.txt` et le tokenizer natif, même environnement que l'étape
   `train` du T64) :
   `python examples/benchmark_chrono_opt_vs_adamw.py --root <spiraton-enhanced> --out BENCHMARK_CHRONO_OPT_VS_ADAMW.json`
   (`--quick` pour une fumée : S, graine 0, 30 époques). Le TEST scellé n'est jamais
   ouvert. C'est le seul endroit où `init_scale = 0.6`, `bounded = False` sans clip
   peut faire diverger Adam, donc le seul où H_stab se tranche.
2. Si l'on poursuit ChronoSpiratonOpt : **retirer ω et son anneau** (inertes,
   10× la mémoire). La mémoire tombe alors à 2 tampons comme AdamW. Remplacer
   ensuite la normalisation par tenseur par une normalisation par coordonnée bornée,
   pour récupérer le cas mal conditionné.
3. Avant toute revendication « remplace AdamW » : un vrai modèle (petit
   Transformer ou LM caractère), plusieurs graines, et un budget de calcul égal.
