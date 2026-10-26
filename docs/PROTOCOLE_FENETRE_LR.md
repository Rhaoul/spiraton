# Protocole FENETRE_LR — sur quelle plage de lr chaque optimiseur apprend-il ?

Date de gel : 2026-10-26 · Claude, à la demande de Ra · statut : **GELÉ avant toute
mesure sur `dataset_aba.txt`**. Ce fichier n'est plus modifié après son commit ; son
sha256 est gravé dans le rapport (`contexte.protocole_sha256`). Le verdict s'écrit
ailleurs (`docs/CHANTIER_OPTIMISEUR_ABA_RESULTATS.md`, §7).

Code : `spiraton/experimental/alpha_omega_optimizer.py` (`AlphaOmegaOpt`),
`examples/benchmark_fenetre_lr.py` (le banc). Tests : `tests/test_alpha_omega_optimizer.py`,
`tests/test_benchmark_fenetre_lr.py`.

## 1. Pourquoi une nouvelle grille

Sur le banc T64 réel, aucun des 7 optimiseurs n'a divergé, pas même AdamW sans clip à
`init_scale = 0,6` : l'hypothèse de stabilité n'y discriminait rien.

La calibration synthétique (§6) dit pourquoi. **Adam ne diverge pas au sens « perte
non finie », même à lr = 1 et init = 3** : son pas est borné par construction
(|m̂/√v̂| ≲ 1 par coordonnée). À trop grand lr il ne s'emballe pas, il s'égare : la
perte reste finie, mais Φ_θ fait moins bien que la copie. Le clip n'y change presque
rien, car le pas d'Adam est invariant à l'échelle du gradient ; le clip ne compte que
quand cette échelle varie de plusieurs ordres dans le temps (init 0,1, où la perte
initiale vaut environ 250 et où v garde la mémoire de ce premier pic).

La question de stabilité qui discrimine n'est donc pas « qui explose ? » mais **« sur
quelle plage de lr apprend-on encore ? »**. C'est ce que mesure cette grille.

## 2. Banc (inchangé)

Le banc T64 tel quel (`benchmark_chrono_opt_vs_adamw.train_run_candidate`) :
`ChronoSpiraton(19, bounded=False)`, `perte_alpha_omega`, lots de 64, au plus 200
époques, patience 20, meilleure époque choisie sur VALIDATION. Splits : TRAIN
101-1000, VALIDATION 3001-3500. **Le TEST 3501-5001 n'est jamais ouvert.** La porte G0
(parité au bit avec `return_training.train_run`) est exécutée en tête ; tout écart
arrête le banc.

## 3. Grille

| Axe | Valeurs |
|---|---|
| lr | 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1 |
| init_scale | 0,1 ; 0,6 ; 1,5 |
| graine | 0, 1, 2 |
| cartouche | S, P |

162 runs par optimiseur, 1134 au total.

## 4. Candidats (gelés)

| Nom | Optimiseur | Clip |
|---|---|---|
| `adam_clipped_t64` | Adam, wd = 0 (référence T64) | 1,0 |
| `adamw_clipped` | AdamW, wd = 1e-2 | 1,0 |
| `adamw_unclipped` | AdamW, wd = 1e-2 | non |
| `spiral_closure` | `SpiralClosureOpt` (k = 5, α = 0,5, tanh) | non |
| `chrono_opt_c_outside` | `ChronoSpiratonOpt` v1 | non |
| `adamw_montee_courte_clipped` | AdamW, wd = 1e-2, lr monté géométriquement de 0,01·lr à lr en 150 pas, **sans retour** (témoin de H2) | 1,0 |
| `alpha_omega` | `AlphaOmegaOpt`, wd = 1e-2, réglages par défaut | non |

Réglages gelés de `AlphaOmegaOpt` : betas (0,9 ; 0,999), eps 1e-8, `cible = 0`,
`eta = 0,1`, `beta_q = 0,9`, `g0 = 0,01`, `g_min = 1e-4`, `g_max = 1e2`. Le lr de la
grille est son lr nominal ; son lr effectif est `lr·g`.

## 5. Définitions et hypothèses

* Un run **apprend** ⇔ il ne diverge pas (perte finie) **et** `best_delta_val > 0` :
  sur VALIDATION, Φ_θ fait mieux que la copie (l'identité vaut 0 exact).
* Une **unité** = (cartouche, init, graine) : 18 unités.
* `N_apprend(unité)` = nombre de lr, sur 9, où le run apprend. C'est la largeur de la
  fenêtre de lr de l'optimiseur sur cette unité.

Les tests sont des sign tests exacts bilatéraux (`return_training.sign_test_exact`,
zéros écartés), au seuil p < 0,01 comme au T64.

**Capacité dite d'avance.** Avec n unités non nulles, p < 0,01 exige au moins k
différences de même signe : n = 18 → 15 ; n = 16 → 14 ; n = 12 → 11 ; n = 10 → 10 ;
n = 8 → 8. **Sous 8 unités non nulles, aucune issue positive n'est possible.**

| Id | Énoncé | Critère |
|---|---|---|
| **H0** (préalable) | AdamW sans clip échoue quelque part | ≥ 1 run de `adamw_unclipped` qui n'apprend pas. Sinon la grille ne discrimine rien et H1-H2 sont des NON-MESURES. |
| **H_NaN** (prédiction) | Les quatre variantes d'Adam ne produisent aucune perte non finie | compte publié par optimiseur ; aucune décision n'en dépend |
| **H1** (principale) | Le retour α-ω élargit la fenêtre de lr par rapport à AdamW + clip | `N_apprend(alpha_omega) − N_apprend(adamw_clipped)` sur les 18 unités : médiane > 0 et p < 0,01 |
| **H2** (témoin) | L'élargissement vient du retour, pas seulement du premier pas court | même test contre `adamw_montee_courte_clipped` |
| **H3** (non-dégradation) | Au meilleur réglage, `alpha_omega` ne fait pas moins bien | par cartouche : config choisie par la règle T64 sur toute la grille (moyenne des `best_delta_val` des graines vivantes), Δ par cycle de VALIDATION de la graine médiane, apparié contre `adam_clipped_t64` et contre `adamw_clipped`. Dégradation ⇔ médiane < 0 et p < 0,01. |

Lecture croisée de H1 et H2 :

| H1 | H2 | Lecture |
|---|---|---|
| oui | oui | le retour, et pas seulement le départ court, élargit la fenêtre |
| oui | non | l'élargissement vient du départ court ; le retour n'ajoute rien de mesurable |
| non | — | pas d'élargissement mesurable sur ce banc |

Non porteurs (INFO) : nombre de runs qui apprennent par lr et par optimiseur,
comparaison de chaque optimiseur à `adamw_clipped`, configs retenues, durées.

## 6. Calibration (synthétique seulement, déclarée)

Tout réglage de `AlphaOmegaOpt` a été choisi sur les données 19D **synthétiques** du
banc (`--synthetic` : a′ = R·a + bruit, M1 émulé), jamais sur `dataset_aba.txt`. La
grille réelle est donc une mesure hors échantillon de ces choix.

### 6.1 Trois versions essayées (cartouche S, graine 0, 200 époques, 9 lr × 3 init)

| Variante | Cases où l'on apprend (sur 27) |
|---|---|
| AdamW + clip | 15 |
| AdamW sans clip | 12 |
| Adam T64 | 14 |
| `SpiralClosureOpt` | 12 |
| `ChronoSpiratonOpt` v1 | 18 |
| α-ω v1 : rapport q = ω/α avec α = −m̂·Δθ | **verrouillage** : g descend jusqu'à g_min, plus rien n'apprend (diagnostiqué sur traces) |
| α-ω v2 : cosinus d'arrivée, g0 = 1, η = 0,1 | 18 |
| α-ω v2, g0 = 1, η = 0,3 | 16 |
| α-ω v2, g0 = 1, η = 1 | 11 |
| α-ω v2, g0 = 1, η = 0,1, β_q = 0 | 18 |
| α-ω v2, g0 = 0,1, g_min = 1e-4 | 22 |
| **α-ω v2, g0 = 0,01, g_min = 1e-4 (gelé)** | **23** |

Ce que la calibration a appris :

* **v1 → v2 : il faut un miroir externe.** Juger le pas par ω/α avec une intention α
  lue sur le moment m̂ revient à laisser l'intention se confirmer elle-même : Δθ est
  fait de m̂, donc α contient la même variance de bruit que l'action, et q tend vers un
  rapport signal/bruit. Le gain s'éteint. Le gradient du lot suivant est indépendant
  de Δθ : le signe de ω = cos(−∇L(θ_{t+1}), Δθ_t) est sans biais.
* **g0 = 0,01 : le premier pas est court.** Avec g0 = 1, les échecs à grand lr se
  décident dans les premiers pas, avant que la boucle se ferme. Partir court et
  laisser le retour allonger le pas fait passer de 18 à 23 cases. C'est aussi ce qui
  impose le témoin H2 : un AdamW qui part aussi court mais dont la montée est un
  calendrier fixe.

### 6.2 Grille complète en synthétique (attendu, non porteur)

`python examples/benchmark_fenetre_lr.py --synthetic --jobs 4`, mêmes 162 runs par
optimiseur sur les données synthétiques. Rapport (hors dépôt) sha256
`9aaee1eb268f13fb2d6d9f2369b4461059e556365896fa9c1a44260ca56044e3`. Parité G0 : PASS.

Runs qui apprennent, par lr (sur 18 runs par case) :

| lr | Adam T64 | AdamW + clip | AdamW sans clip | montée courte | spiral | chrono v1 | α-ω |
|---|---|---|---|---|---|---|---|
| 1e-4 | 0 | 0 | 0 | 0 | 0 | 0 | **10** |
| 3e-4 | 12 | 12 | 6 | 6 | 0 | 0 | 12 |
| 1e-3 | 16 | 16 | 12 | 14 | 6 | 12 | 12 |
| 3e-3 | 17 | 17 | 12 | 16 | 12 | 18 | 12 |
| 1e-2 | 18 | 18 | 12 | 16 | 12 | 18 | 13 |
| 3e-2 | 18 | 18 | 17 | 17 | 18 | 18 | 15 |
| 1e-1 | 0 | 1 | 2 | **18** | 18 | 18 | 16 |
| 3e-1 | 0 | 0 | 5 | 16 | 2 | 18 | 16 |
| 1 | 0 | 0 | 3 | 12 | 5 | 2 | **18** |
| **total /162** | 81 | 82 | 69 | 115 | 73 | 104 | **124** |

Divergences non finies : 0 partout. H1 en synthétique : médiane +3,5, 13 unités
pour, 3 contre, 2 nulles, **p = 0,021 (NON au seuil 0,01)**. H2 : médiane +1,
12 pour, 5 contre, p = 0,14 (NON). H3 : aucune dégradation (p ≥ 0,18).

Lecture : à init 0,6 et 1,5, α-ω apprend dans 8 à 9 lr sur 9 (AdamW + clip : 5 ;
montée courte : 7 à 8). À init 0,1, il fait à peu près comme Adam (18 cases sur les
6 unités, contre 21 pour AdamW + clip) : ce sont ces unités qui font perdre H1. Le
témoin à montée courte récupère presque tout le bord haut : **l'échec d'Adam à grand
lr se joue dans les premiers pas**. L'apport propre du retour est au bord bas
(lr 1e-4 et 3e-4), là où un calendrier fixe ne peut pas allonger le pas.

## 7. Prédictions écrites avant la mesure

Écrites après la calibration synthétique (§6.2) et avant toute exécution sur
`dataset_aba.txt` :

* H0 : PASS. AdamW cesse d'apprendre à grand lr.
* H_NaN : 0 pour les quatre variantes d'Adam.
* H1 : **à la limite**. Le synthétique donne p = 0,021, au-dessus du seuil ; un NON
  est l'issue la plus probable si les unités à init 0,1 se comportent comme en
  synthétique.
* H2 : NON probable. Le départ court explique l'essentiel du bord haut ; le retour
  n'ajoute que le bord bas.
* H3 : pas de dégradation, avec des différences proches de zéro.

## 8. Exécution

Sur la machine de Ra, depuis `F:\code\claude\spiraton-enhanced` (racine qui contient
`dataset_aba.txt` et `Tokenizer\`), avec le clone au commit de ce gel :

```
set PYTHONPATH=F:\code\claude\spiraton-enhanced\spiraton
python spiraton\examples\benchmark_fenetre_lr.py --root . --out BENCH_FENETRE_LR.json --jobs 4
```

Sorties : `BENCH_FENETRE_LR.json`, son `.sha256` et un `.log` (durée). Le rapport
contient le contexte d'exécution (doctrine T74) : plateforme, versions, encodage,
sha256 de ce protocole, de `dataset_aba.txt` et du `.so` du tokenizer conservé.
