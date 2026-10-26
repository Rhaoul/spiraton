# Protocole CONFIRMATION_GRAND_LR (H4) — gelé avant toute mesure

Date : 2026-10-26 · Claude, à la demande de Ra (option A, « geler d'abord ce test de
confirmation ») · banc : `examples/benchmark_confirmation_grand_lr.py` · rapport attendu :
`BENCH_CONFIRMATION_GRAND_LR.json` (+ `.sha256`, `.log`).

Ce fichier est figé par le commit qui l'introduit. Il ne sera plus modifié : le verdict
s'écrit dans `docs/CHANTIER_OPTIMISEUR_ABA_RESULTATS.md`. Le rapport grave son sha256 et
son blob git ramené à LF (`protocole_blob_git_lf`), qui se compare à
`git rev-parse <commit>:docs/PROTOCOLE_CONFIRMATION_GRAND_LR.md` quelle que soit la
plateforme (leçon du rapport FENETRE_LR, dont le sha256 était celui de la forme CRLF).

## 1. Pourquoi

La grille FENETRE_LR (protocole `docs/PROTOCOLE_FENETRE_LR.md`, rapport scellé
`ecb4279a…`) a tranché : la fenêtre de lr d'`AlphaOmegaOpt` (AO) s'élargit grâce au
premier pas court, et le retour n'ajoute rien de mesurable à sa **largeur** (H2 NON).

Après lecture de ces verdicts, le §7.2 des résultats a relevé autre chose : à lr ≥ 0,1,
AO fait mieux que le témoin à montée courte dans 54 runs appariés sur 54 (graines 0 à 2),
et sa qualité reste plate jusqu'à lr = 1 quand celle du témoin baisse. Une observation
faite après coup, sur les données mêmes qui l'ont suggérée, ne prouve rien. Ce protocole
la soumet à des graines **neuves**, jamais utilisées par aucun banc du chantier, avec un
critère écrit d'avance.

## 2. Banc (inchangé)

Celui du T64 et de FENETRE_LR, sans aucune modification : `ChronoSpiraton(19, bounded=False)`
en float64, perte `perte_alpha_omega`, batch 64, au plus 200 époques, patience 20,
meilleure époque choisie sur VALIDATION (médiane des Δ par cycle). Splits : TRAIN
101-1000, VALIDATION 3001-3500. **TEST 3501-5001 n'est jamais ouvert.** Porte G0 :
parité au bit avec `return_training.train_run`.

## 3. Grille

| Axe | Valeurs |
|---|---|
| Optimiseurs | `adamw_montee_courte_clipped`, `alpha_omega` |
| Cartouches | S, P |
| `init_scale` | 0,1 ; 0,6 ; 1,5 |
| lr nominal | 0,1 ; 0,3 ; 1 |
| Graines | **3, 4, 5** (neuves) |

2 × 2 × 3 × 3 × 3 = **108 runs**.

## 4. Candidats (gelés)

* `adamw_montee_courte_clipped` : le témoin de FENETRE_LR tel quel. AdamW (wd 1e-2) +
  clip 1,0, lr monté géométriquement de 0,01·lr à lr en 150 pas, sans retour.
* `alpha_omega` : `AlphaOmegaOpt` aux réglages par défaut gelés (η = 0,1, β_q = 0,9,
  g0 = 0,01, g ∈ [1e-4, 1e2], cible 0), wd 1e-2, sans clip. Il est instancié par
  `AlphaOmegaOptJournal`, qui relit g et ω̄ après chaque pas sans rien calculer : la
  trajectoire des poids est celle d'`AlphaOmegaOpt` au bit
  (`tests/test_benchmark_confirmation_grand_lr.py::test_journal_ne_touche_pas_la_trajectoire`).

## 5. Définitions et hypothèses

* Une **unité** = (cartouche, init, graine) : 18 unités.
* `best_delta_val` = médiane des Δ de VALIDATION à la meilleure époque (l'identité vaut
  0 exact). Un `best_delta_val` non fini vaut −∞ ; −∞ − (−∞) vaut 0.
* `D(unité)` = moyenne, sur les 3 lr, de `best_delta_val(AO) − best_delta_val(témoin)`.
  Une moyenne indéterminée (+∞ et −∞ mêlés) vaut 0.
* `g*` = gain g d'AO à la fin de sa meilleure époque. Un `g*` non fini (aucune époque
  gardée) fait compter l'unité comme nulle dans H4m.

Tests : sign tests exacts bilatéraux (`return_training.sign_test_exact`, zéros
écartés), seuil p < 0,01. **Capacité dite d'avance** : n = 18 → 15 unités de même signe ;
n = 16 → 14 ; n = 12 → 11 ; sous 8 unités non nulles, aucune issue positive.

| Id | Énoncé | Critère |
|---|---|---|
| **H4** (principale) | À grand lr, AO fait mieux que le témoin | médiane de D > 0 et p < 0,01 |
| **H4m** (mécanisme) | Le retour baisse g quand le lr nominal monte | par unité `log g*(0,1) − log g*(1)` : médiane > 0 et p < 0,01 |
| **H_NaN** (publié) | Pertes non finies | compte par optimiseur ; aucune décision n'en dépend |

Lecture croisée :

| H4 | H4m | Lecture |
|---|---|---|
| oui | oui | confirmé : à grand lr, le retour garde la qualité en raccourcissant le pas ; c'est l'apport propre du retour, distinct du départ court |
| oui | non | AO fait mieux, mais pas par le mécanisme supposé ; il reste à l'identifier |
| non | — | l'observation du §7.2 ne se confirme pas sur graines neuves : elle tenait aux graines 0 à 2 ou à la lecture après coup |

**Non porteurs (INFO).** Runs qui apprennent ; médiane de `best_delta_val` par optimiseur,
cartouche et lr ; médianes de `g*`, du lr effectif `lr·g*` et de la meilleure époque ;
séries de g et ω̄ en fin d'époque pour chaque run d'AO ; durée.

**Porte de rejeu (INFO).** Avant la grille, le banc rejoue deux runs de FENETRE_LR
(cartouche S, init 0,6, lr 1, graine 0) et compare leur état final au rapport scellé
`ecb4279a…` : `d750fc6e…` pour le témoin, `9e73622c…` pour AO. Identiques, ils montrent
sur la machine même que le journal ne touche pas la trajectoire. Un écart n'invalide pas
H4 (doctrine T74 : un écart de sha sans divergence de suites est un horodatage), mais il
est publié avec le contexte. Sans objet en synthétique.

## 6. Calibration (synthétique seulement, déclarée)

Données 19D synthétiques du banc T64, même grille complète, jamais une mesure.
Rapport `H4_SYNTH.json`, sha256 `76593315…`, Linux, 4 processus.

| Quantité | Synthétique |
|---|---|
| H4 : unités D > 0 / D < 0 | 15 / 3, médiane +0,104, p = 0,0075 → OUI, à la limite |
| H4m : unités `g*(0,1) > g*(1)` | 18 / 0, p = 7,6e-6 → OUI |
| Médiane de `g*` à lr 0,1 / 0,3 / 1 | 0,044 / 0,0046 / 0,0014 |
| Médiane de `lr·g*` à lr 0,1 / 0,3 / 1 | 0,0044 / 0,0014 / 0,0014 |
| Pertes non finies | 0 / 0 |

Deux choix ont été arrêtés sur cette calibration, avant toute mesure réelle :

* **Le critère de H4m.** La première version exigeait g*(0,1) > g*(0,3) > g*(1)
  strictement dans 15 unités ; le synthétique en donne 14, parce qu'entre 0,3 et 1 le lr
  effectif est déjà presque le même et le bruit de g décide de l'ordre. Le mécanisme
  supposé dit seulement que g baisse quand le lr monte : la comparaison des extrêmes
  (0,1 contre 1) en sign test est l'énoncé exact, et elle a la même capacité que H4.
* **Aucun autre réglage.** AO, le témoin et la grille sont ceux de FENETRE_LR.

Le synthétique montre aussi une faiblesse que le réel n'a pas montrée : à init 0,1 et
lr 0,1, AO fait bien moins bien que le témoin dans 4 unités sur 6 (écart ≈ −1,8 à ce
lr), car g y remonte jusqu'à 0,6-1,5 après le pic de perte initial. Trois de ces quatre
unités sont les 3 unités négatives de H4 en synthétique.

## 7. Prédictions écrites avant la mesure

* H4 : **OUI probable**. Sur le réel, les graines 0 à 2 donnaient 18 unités sur 18 à
  chacun des trois lr. Un NON viendrait des unités à init 0,1 si elles se comportent
  comme en synthétique.
* H4m : OUI probable.
* H_NaN : 0 pour les deux.
* Rejeu : identique, si la machine et les versions sont celles de FENETRE_LR.

## 8. Exécution

Sur la machine de Ra, depuis `F:\code\claude\spiraton-enhanced` (racine qui contient
`dataset_aba.txt` et `Tokenizer\`), avec une copie de travail au commit de ce gel :

```
set PYTHONPATH=<copie de travail au commit de gel>
python <copie de travail>\examples\benchmark_confirmation_grand_lr.py --root . --out BENCH_CONFIRMATION_GRAND_LR.json --jobs 8
```

Sorties : `BENCH_CONFIRMATION_GRAND_LR.json`, son `.sha256` et un `.log` (durée). Le
rapport contient le contexte d'exécution (doctrine T74) : plateforme, versions, encodage,
sha256 et blob git de ce protocole, sha256 de `dataset_aba.txt` et du `.so` du tokenizer.
