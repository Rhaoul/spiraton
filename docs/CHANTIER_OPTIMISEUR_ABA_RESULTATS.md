# Chantier OPTIMISEUR_ABA — résultats (remplacer AdamW par la logique A → B → A′)

Date : 2026-10-26 · Claude · spécification : `docs/CHANTIER_OPTIMISEUR_ABA.md`
(conservée telle que livrée) · code : `spiraton/experimental/chrono_optimizer.py`,
`spiraton/experimental/spiral_optimizer.py`, `spiraton/experimental/alpha_omega_optimizer.py`
· bancs : `examples/benchmark_chrono_opt_vs_adamw.py`, `examples/benchmark_chrono_opt_tasks.py`,
`examples/benchmark_fenetre_lr.py` (protocole gelé `docs/PROTOCOLE_FENETRE_LR.md`).

## 0. Verdict en une ligne

**Aucune des deux versions ne bat AdamW sur une tâche ordinaire. Leur seul avantage
mesuré, c'est la robustesse aux pics de gradient sans `clip_grad_norm_`, et AdamW la
retrouve avec le clip.** Sur le banc T64 réel (§5), personne ne diverge, pas même
AdamW sans clip à `init_scale = 0.6`. H_stab n'y discrimine donc aucun optimiseur, et
aucun nouvel optimiseur n'y bat Adam T64 de façon significative.

**Mise à jour, fenêtre de lr (§7).** Sur une grille gelée de 1e-4 à 1, `AlphaOmegaOpt`
apprend aux 9 lr dans les 18 unités (162 runs sur 162), contre 107 pour AdamW + clip
(H1 OUI, p = 7,6e-6). Un AdamW qui part du même pas court, sans aucune régulation, fait
presque autant (160 ; H2 NON) : la largeur de la fenêtre vient du départ court, pas du
retour. Hors protocole, la qualité d'AO reste plate jusqu'à lr = 1 quand celle de ce
témoin baisse ; c'est une piste à confirmer par un test gelé, pas un verdict.

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

## 5. Banc T64 réel (`dataset_aba.txt`, TRAIN 101-1000 / VALIDATION 3001-3500)

Exécuté le 2026-10-26 sur la machine de Ra, depuis la racine `spiraton-enhanced`,
avec le clone au commit `7d4f5d3` :
`python spiraton\examples\benchmark_chrono_opt_vs_adamw.py --root . --out BENCH_CHRONO_OPT_REEL.json`.
Durée : 5 min 18 s. Le TEST scellé n'a pas été ouvert (le rapport porte
`TEST_SCELLE_NON_OUVERT`). Gate G0 de parité au bit avec `return_training.train_run` :
**PASS** (state sha256 `17debf25783d…`, best_delta_val 0,349356).

**Contexte d'exécution (doctrine T74).**
* Système : Windows 11 Pro 10.0.26200, CPython 3.13.3, torch 2.8.0+cpu, numpy 2.3.2.
* Tokenizer natif chargé : `Tokenizer\bin\libspiratontokenizer.so`, 234 455 o, sha256
  `9b2830ef61def5c38a77616b400f384eda4c8ee7e4630b2fc30a6d495b050f94`. **Ce n'est pas
  le fichier du T64** (`c4d5c53e…`, 231 868 o). Les sept optimiseurs ont tourné sur le
  même binaire, et G0 compare deux boucles sur ce même binaire : la comparaison entre
  optimiseurs est valide, mais les valeurs absolues ne se comparent pas au gel T64.
* Locale : LC_CTYPE=French_France.1252, autres catégories C ; E/S en utf-8
  (PYTHONUTF8=1, PYTHONIOENCODING=utf-8).
* Le paquet `spiraton` importé est le clone, via `PYTHONPATH` : un autre `spiraton`
  installé dans site-packages le masquait.
* Rapport scellé : `BENCH_CHRONO_OPT_REEL.json`, sha256
  `8867673cf0cfe2568e93b7a1c1872fb2553ca0c60f38efa07f3eb4bcd0032f22` (racine
  `spiraton-enhanced`, hors git, avec son `.sha256` et son `.log`).

| Optimiseur | Divergences | Descente ≥ 20 % | Config S retenue (init/lr) | Δ_S val. | Δ_P val. | Q6/Q7 (S) |
|---|---|---|---|---|---|---|
| adam_clipped_t64 (référence) | 0/24 | 24/24 | 0,1 / 1e-2 | +0,3499 | +0,1921 | non |
| adamw_clipped | 0/24 | 24/24 | 0,1 / 1e-2 | +0,3477 | +0,1927 | oui |
| adamw_unclipped | 0/24 | 24/24 | 0,6 / 1e-2 | +0,3501 | +0,1907 | oui |
| chrono_opt_c_outside (v1) | 0/24 | 24/24 | 0,1 / 1e-2 | +0,3550 | +0,1942 | non |
| chrono_opt_c_inside (v1) | 0/24 | 24/24 | 0,1 / 1e-2 | +0,3514 | +0,1930 | non |
| chrono_opt_v0_spec | 0/24 | 24/24 | 0,6 / 1e-2 | +0,3417 | +0,0230 | oui |
| spiral_closure | 0/24 | 24/24 | 0,6 / 1e-2 | +0,3498 | +0,1915 | oui |

Aucun repli `bounded=True` n'est requis, et les 3 graines restent vivantes partout.
La v0 est instable selon la graine (S graine 0 : −0,190 ; P graine 2 : −0,200). À
lr = 1e-3, elle tombe entre −0,93 et −1,24.

**H_stab.** Sur la tranche `init_scale = 0.6`, `bounded = False`, sans clip, il y a
0 divergence sur 6 runs par cartouche **pour tous les optimiseurs, AdamW sans clip
compris**. L'hypothèse n'est donc pas réfutée, mais ce banc ne la teste pas : la
stabilité n'y discrimine personne.

**Qualité (sign-test exact apparié, VALIDATION, n = 500, aucun zéro).**

| Comparaison | S : médiane, +/−, p | P : médiane, +/−, p |
|---|---|---|
| chrono v1 − adam_t64 | −0,0036, 225/275, 0,028 | **−0,0078, 202/298, 2e-5** |
| chrono v1 − adamw_clipped | −0,0028, 233/267, 0,14 | −0,0012, 242/258, 0,50 |
| chrono v1 − adamw_unclipped | +0,0001, 253/247, 0,82 | −0,0030, 238/262, 0,30 |
| chrono v1 − chrono c_inside | −0,0004, 240/260, 0,40 | −0,0035, 225/275, 0,028 |
| chrono v1 − v0_spec | **+0,0056, 292/208, 2e-4** | **+0,1448, 459/41, 2e-90** |
| spiral − adam_t64 | −0,0009, 240/260, 0,40 | **−0,0080, 191/309, 1,5e-7** |
| spiral − adamw_clipped | +0,0053, 277/223, 0,018 | −0,0050, 225/275, 0,028 |
| spiral − adamw_unclipped | **+0,0210, 306/194, 6e-7** | **−0,0049, 215/285, 0,002** |
| spiral − chrono v1 | +0,0050, 273/227, 0,044 | −0,0037, 227/273, 0,044 |

En gras : p < 0,01, le seuil du protocole. Aucun nouvel optimiseur ne bat Adam T64.
En P, chrono v1 et spiral font significativement **moins bien** que lui. Le seul
avantage net est spiral contre AdamW sans clip en S (+0,021), renversé en P. La
correction de la v0 est, elle, nette sur les deux cartouches.

## 6. La suite proposée

1. **Pour vraiment tester H_stab**, il faut un régime où Adam sans clip diverge
   réellement : lr plus grand, init plus grande, ou K plus long. C'est une nouvelle
   grille, à geler d'abord comme un tour propre. **Fait au §7** (lr jusqu'à 1).
2. Si l'on poursuit ChronoSpiratonOpt : **retirer ω et son anneau** (inertes,
   10× la mémoire). La mémoire tombe alors à 2 tampons comme AdamW. Remplacer
   ensuite la normalisation par tenseur par une normalisation par coordonnée bornée,
   pour récupérer le cas mal conditionné.
3. Avant toute revendication « remplace AdamW » : un vrai modèle (petit
   Transformer ou LM caractère), plusieurs graines, et un budget de calcul égal.

## 7. Fenêtre de lr sur le banc réel (protocole gelé `docs/PROTOCOLE_FENETRE_LR.md`)

Le §5 montrait qu'à lr = 1e-3 la stabilité ne discrimine personne. La grille élargie
(9 lr de 1e-4 à 1, 3 init, 3 graines, 2 cartouches, 7 optimiseurs : 1134 runs) a été
gelée au commit `b04b13d`, avant toute exécution sur les données réelles. Elle ajoute
deux candidats :

* `AlphaOmegaOpt` (AO) : AdamW dont le gain multiplicatif g est régulé par le cosinus
  entre le pas qui vient d'être fait et le gradient du batch **suivant**, à l'arrivée.
  C'est le miroir externe : A′ est jugé par ce qu'il rencontre, pas par sa propre
  intention. Départ g0 = 0,01.
* `adamw_montee_courte_clipped` : le témoin. Il part du même gain 0,01 et remonte à 1
  en 150 pas, sans aucune régulation. Il isole la part du « premier pas court ».

**Contexte déclaré (doctrine T74).** Machine de Ra, Windows-11-10.0.26200, Python
3.13.3, torch 2.8.0+cpu, numpy 2.3.2, encodage utf-8, 8 processus, 262 s.
`dataset_aba.txt` sha256 `6de843a2…`, `.so` du tokenizer conservé `9b2830ef…`. Le
protocole lu par le banc a pour sha256 `a79973f5…` : c'est la forme CRLF du fichier
gelé (extraction Windows avec `autocrlf`), vérifiée en reconvertissant le fichier du
dépôt ; son blob git `19aef861` est celui du commit de gel. Rapport scellé
`BENCH_FENETRE_LR.json`, sha256 `ecb4279ab75a7e30…3276a243`, revérifié après
transfert. Parité T64 : PASS. TEST scellé, non ouvert.

### 7.1 Verdicts pré-enregistrés

| Id | Mesure | Verdict | Prédit (§7 du protocole) |
|---|---|---|---|
| H0 | AdamW sans clip n'apprend pas dans 54 runs sur 162 | **PASS** : la grille discrimine | PASS |
| H_NaN | 0 perte non finie, pour les 7 optimiseurs | publié | 0 |
| H1 | AO : 162 runs qui apprennent, AdamW + clip : 107 ; médiane +3 lr par unité ; 18 unités positives sur 18 ; p = 7,6e-6 | **OUI** | à la limite, NON probable |
| H2 | AO : 162, montée courte : 160 ; médiane 0 ; 16 unités à égalité, 2 positives ; p = 0,5 | **NON** | NON probable |
| H3 | S : +0,011 contre Adam T64 (302/198, p = 3,8e-6), +0,017 contre AdamW + clip (315/185, p = 6,6e-9) ; P : −0,0013 contre les deux (235/265, p = 0,19) | **pas de dégradation** | pas de dégradation |

Runs qui apprennent, par lr (18 par case) :

| Optimiseur | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 | 3e-2 | 0,1 | 0,3 | 1 | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| `alpha_omega` | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | **162** |
| `adamw_montee_courte_clipped` | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 16 | 160 |
| `chrono_opt_c_outside` | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 8 | 0 | 134 |
| `spiral_closure` | 12 | 14 | 18 | 18 | 18 | 18 | 7 | 4 | 2 | 111 |
| `adamw_unclipped` | 12 | 18 | 18 | 18 | 18 | 16 | 4 | 3 | 1 | 108 |
| `adamw_clipped`, `adam_clipped_t64` | 18 | 18 | 18 | 18 | 18 | 17 | 0 | 0 | 0 | 107 |

**Lecture pré-enregistrée (H1 oui, H2 non) : l'élargissement vient du départ court ;
le retour n'ajoute rien de mesurable à la largeur de la fenêtre.** H1 sort plus fort
que prévu : en synthétique, les unités à init 0,1 freinaient AO ; sur les données
réelles, il apprend partout. Chrono v1 élargit aussi la fenêtre (134, 18 unités sur
18 contre AdamW + clip, INFO), sans atteindre lr = 1.

**Deux réserves.**

* Le NON de H2 tient à ce que les deux fenêtres sont presque pleines. Avec 16 unités
  à égalité, il reste n = 2 : sous le plancher de capacité dit d'avance (8), aucune
  issue positive n'était possible. Le banc réel n'a pas de bord bas (Adam apprend dès
  1e-4), alors que le synthétique montrait l'apport du retour justement là (1e-4 et
  3e-4, §6.2 du protocole). H2 ne pouvait donc tester que le bord haut.
* L'avantage apparié de H3 en S ne se lit pas comme un gain de qualité. AO y retient
  lr = 1 à init 0,1, le bord de la grille. Il gagne sur 302 cycles sur 500, mais la
  médiane de ses Δ par cycle (0,3509) reste sous celle d'Adam T64 (0,3578) : ses Δ
  ne sont pas décalés vers le haut, ils sont réordonnés. On s'en tient au verdict
  pré-enregistré : pas de dégradation.

### 7.2 Ce que la largeur ne voit pas (INFO, post hoc, non porteur)

Ce tableau a été construit **après** lecture des verdicts, à partir des runs du
rapport scellé. Il ne tranche rien. Médiane de `best_delta_val` sur les 9 runs
(init × graine) de chaque case :

| Cartouche | Optimiseur | 1e-3 | 1e-2 | 3e-2 | 0,1 | 0,3 | 1 |
|---|---|---|---|---|---|---|---|
| S | `adamw_clipped` | 0,3468 | 0,3477 | 0,3545 | −0,1451 | −0,1451 | −0,1451 |
| S | montée courte | 0,3457 | 0,3530 | 0,3558 | 0,3326 | 0,3321 | 0,2915 |
| S | `alpha_omega` | 0,3445 | 0,3470 | 0,3487 | 0,3484 | 0,3488 | 0,3502 |
| P | `adamw_clipped` | 0,1904 | 0,1907 | 0,1880 | −0,4937 | −0,4941 | −0,4940 |
| P | montée courte | 0,1905 | 0,1914 | 0,1899 | 0,1765 | 0,1722 | 0,1434 |
| P | `alpha_omega` | 0,1882 | 0,1887 | 0,1881 | 0,1889 | 0,1889 | 0,1900 |

Appariés run à run (même cartouche, init, graine et lr), AO fait mieux que la montée
courte dans **54 runs sur 54 à lr ≥ 0,1**, soit 18 unités sur 18 à chacun des trois
lr. À lr ≤ 3e-2, c'est l'inverse : la montée courte fait mieux dans 73 runs sur 108.

Le témoin ne fait que retarder le lr nominal. Après 150 pas il est revenu à sa
valeur, et à lr = 1 le pas borné d'Adam (de l'ordre de lr par coordonnée) est trop
grand : le run bat encore la copie, il « apprend » au sens du protocole, mais moins
bien. AO continue de se réguler. Mécanisme probable, **non mesuré ici** (g n'est pas
journalisé dans le rapport) : quand le pas dépasse, le gradient suivant pointe en
arrière (ω̄ < 0) et g baisse jusqu'à ce que le pas s'aligne sur ce qu'il rencontre.
Sa qualité devient alors indépendante du lr nominal, de 1e-3 à 1. Ce serait l'apport
propre du retour : il ne porte pas sur la largeur de la fenêtre mais sur ce qui se
passe dedans. Le prix se voit aux lr usuels, où AO reste un peu en dessous.

### 7.3 Suite

1. **Confirmer 7.2 par un test gelé d'abord**, sur des graines neuves (3, 4, 5) pour
   ne pas réutiliser les runs qui ont suggéré l'hypothèse. Proposition : par unité,
   moyenne de `best_delta_val` sur lr ∈ {0,1 ; 0,3 ; 1}, AO contre la montée courte,
   médiane > 0 et p < 0,01 (n = 18, il faut 15 unités du même signe). Journaliser g
   et ω̄ pour vérifier le mécanisme. Coût : 108 runs, environ une minute.
2. AO n'est **pas** un remplaçant d'AdamW aux lr usuels : il y fait un peu moins
   bien. Son intérêt mesuré est de dispenser du choix du lr, et cela reste à montrer
   sur un vrai modèle (§6, point 3, ou `run_c2` du dépôt `cybernetique`).
