# R4 — Réplication pré-enregistrée du signal E1 sous CamemBERT

Rédigé le 2026-09-28 (Claude Opus 5.5), **avant que le corpus existe**. Aucun
paragraphe de ce fichier ne se modifie après la création du corpus ; toute
précision s'ajoute en bas, datée.

**À l'auteur du corpus : ne lisez pas ce fichier.** Votre consigne est
`docs/SEMANTIC_THERMO_R4_CONSIGNE_AUTEUR.md`, et elle seule.

## 1. Ce qu'on réplique

Addendum R3 de `docs/SEMANTIC_THERMO_RESULTATS_E0_E3.md` : sous vecteurs contextuels
CamemBERT, à position fixée, les segments DX/OUT ont une divergence de continuité plus
positive que les segments LV/IN. AUC ≈ 0,60 sur horscanon + f0b ; 0,60-0,61 sur
horscanon seul ; nul en position B. Trois réserves motivent la réplication :
- critère p < 0,01 tenu sur 3 graines sur 5 seulement ;
- 3e représentation essayée (chemins multiples) ;
- corpus écrit par un auteur qui voulait que la forme soit le sens.

## 2. Pipeline gelé (identique à R3, aucun paramètre libre)

`scripts/semantic_thermo_replicate.py`, mode `run` :
- **Vecteurs** : `camembert-base` rév. `a75967561c78f2aa81cc41045378d3b4ee25af9e`, poids
  sha256 `486643fdcac936afc551aa4b0fedcd9f61c5f71f42b8333e07c709f38043475d`, hors ligne,
  dernière couche, moyenne des sous-mots, phrase encodée entière, découpage par nombre
  de mots.
- **Champ** : ACP 64 sans blanchiment + champ k-NN (k = 8), ajustés sur la calibration
  de `dataset_aba.txt` (partage seedé `split_cycles(·, seed)`), graines {0, 1, 2, 3, 4}.
  Le corpus de réplication n'entre **jamais** dans l'ajustement.
- **Score** : `div` = − pente OLS de ln ρ_t contre t (continuité), segments de ≥ 2 mots.
- **Statistiques** : AUC(DX > LV) ; permutation des étiquettes à position fixée (10 000) ;
  IC bootstrap 95 % par cycle (2000). Mêmes fonctions que la sonde
  (`semantic_thermo_probe._e1_block`).

## 3. Corpus attendu

Fichier `corpus_r4_replication_aba.txt` (racine de l'atelier, à côté des autres).
Cahier des charges transmis à l'auteur (consigne) :
- **144 cycles** : 6 formes non uniformes × 24 (F0 +,+,− ; F0b +,−,− ; F1 −,+,+ ;
  F1b −,−,+ ; F2 +,−,+ ; F3 −,+,−) ; 6 cycles par opérateur et par forme.
- À chaque position, 72 segments DX et 72 LV : le plan est **équilibré par construction**.
- Chaque segment : 3 à 7 mots. Une phrase française par cycle. Aucun doublon. Aucune
  phrase reprise des corpus existants.
- F4 et F5± sont exclus (pas de contraste de chiralité à position fixée).

Puissance (approximation sans effet de grappe) : 432 segments, AUC 0,60 → z ≈ 3,6,
soit ≈ 0,85 de chances d'atteindre p < 0,01.

## 4. Mise à l'aveugle

- L'auteur ne reçoit que la consigne. Elle décrit l'orientation (DX/OUT = émission,
  vers le dehors ; LV/IN = réception, vers le dedans) comme le fait `GRAMMAIRE_ABA.md`.
  Elle ne dit **rien** de la mesure : pas de densité, de divergence, de modèle, de
  résultat antérieur, ni de position où un effet est attendu.
- L'auteur n'a lu ni ce fichier, ni les résultats R1-R3, ni `corpus_horscanon_aba.txt`.
- **Gel avant mesure** : `validate` doit passer, puis le sha256 du fichier est consigné
  en §7 de ce document. Seulement après, `run --expect-sha <sha>` est lancé ; le script
  refuse de tourner si le sha diffère. Aucune correction du corpus n'est permise après
  la première mesure.

## 5. Prédictions et critères (figés)

- **P1, primaire** : AUC stratifiée par position > 0,5. **RÉPLIQUÉ** si, sur **5/5**
  graines, l'IC bootstrap exclut 0,5 par le haut **et** p perm. bilatéral < 0,01 ; et si
  la longueur seule (même test) n'atteint pas p < 0,01.
- **NON RÉPLIQUÉ** si l'IC inclut 0,5 sur au moins 3 graines, ou si l'AUC est < 0,5.
- **INTERMÉDIAIRE** dans tous les autres cas. Il est rapporté tel quel, jamais arrondi
  vers l'un ou l'autre.
- **P2, secondaires** (descriptifs, sans effet sur le verdict) :
  - AUC en position B : IC incluant 0,5 ;
  - AUC en A et en A′ > 0,5 ;
  - effet de position à chiralité fixée (A/B vs A′) : IC incluant 0,5.
- **Prior de l'auteur de ce protocole** : RÉPLIQUÉ ≈ 35 %, INTERMÉDIAIRE ≈ 25 %, NON
  RÉPLIQUÉ ≈ 40 %.

Si NON RÉPLIQUÉ, le signal R3 est attribué à l'écriture de horscanon, et Gate 2 reste
fermée pour toutes les représentations testées. Si RÉPLIQUÉ, Gate 2 passe pour la
représentation contextuelle ; E3 et E2 restent nuls et ne sont pas rouverts par ce
seul résultat.

## 6. Intégrité de l'outil (avant le corpus)

`run` appliqué à `corpus_horscanon_aba.txt` doit reproduire le contrôle post hoc R3
(horscanon seul : AUC 0,599-0,612). Résultat consigné en §7.

## 7. Consignation (ajout seulement)

- **2026-09-28, intégrité de l'outil (§6), avant tout corpus R4.**
  `run corpus_horscanon_aba.txt --any-plan` (sha `1d26bfcc…`) reproduit **au bit** les
  AUC du contrôle post hoc R3 : 0,612 · 0,609 · 0,599 · 0,607 · 0,604 ; par position
  A 0,671-0,731 · B 0,447-0,475 · A′ 0,610-0,633. Les p et IC diffèrent légèrement
  (autre ordre de tirage du générateur ; l'AUC est déterministe). Pour mémoire, le
  script classe horscanon « INTERMÉDIAIRE » (IC 5/5, p < 0,01 sur 0/5), d'où la
  nécessité de la réplication. `validate` : 88 écarts sur horscanon (hors plan,
  attendu) ; `VALIDE` sur un faux corpus synthétique conforme ; `run` refuse un sha
  incorrect.

- **2026-09-28, GEL DU CORPUS (avant toute mesure).** `corpus_r4_replication_aba.txt`
  (racine de l'atelier), 160 lignes dont 144 cycles, écrit par un sous-agent Claude
  (general-purpose) à qui seule la consigne a été transmise, avec interdiction de lire
  tout autre fichier et d'exécuter autre chose que `validate`. L'agent rapporte deux
  passes du validateur : 4 segments de 8 mots raccourcis, puis `VALIDE`. Revalidé par
  l'orchestrateur : `VALIDE`. Relecture d'un échantillon de 18 cycles pour la qualité
  seulement, **aucune modification** : orientations lisibles, 1055 mots distincts sur
  2444, marqueurs lexicaux attendus (« reçoit » 20, « recueille » 15).
  **sha256 = `0b721fe0aad3dac72d4e826f552f665cc93c62370f1245900b2585737b7a9a1d`.**
  Commande de mesure (à lancer après feu vert de Ra) :
  `HF_HUB_OFFLINE=1 python scripts/semantic_thermo_replicate.py run ../corpus_r4_replication_aba.txt --expect-sha 0b721fe0aad3dac72d4e826f552f665cc93c62370f1245900b2585737b7a9a1d --out runs/semantic_thermo/r4_replication.json`
  Limite d'aveuglement déclarée : l'auteur est de la même famille de modèle que
  l'auteur de horscanon. C'est une autre écriture, pas une autre espèce d'auteur.

- **2026-09-28, MESURE (après le commit `193cd22` qui gèle protocole et sha).**
  `run` sur `corpus_r4_replication_aba.txt` (sha `0b721fe0…` vérifié, CamemBERT rév.
  `a7596756…`), sortie `runs/semantic_thermo/r4_replication.json` (hors dépôt).

  | graine | AUC strat. (P1) | IC 95 % | p perm. | d | A | B | A′ | longueur p |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 0,534 | [0,470 ; 0,599] | 0,121 | 0,07 | 0,442 | 0,578 | 0,571 | 0,020 |
  | 1 | 0,538 | [0,478 ; 0,603] | 0,072 | 0,08 | 0,439 | 0,598 | 0,564 | 0,021 |
  | 2 | 0,550 | [0,484 ; 0,612] | 0,021 | 0,12 | 0,438 | 0,610 | 0,610 | 0,022 |
  | 3 | 0,548 | [0,486 ; 0,612] | 0,029 | 0,13 | 0,449 | 0,605 | 0,606 | 0,024 |
  | 4 | 0,530 | [0,468 ; 0,592] | 0,151 | 0,07 | 0,446 | 0,577 | 0,564 | 0,022 |

  **VERDICT (règle §5, appliquée par le script) : NON RÉPLIQUÉ.** L'IC inclut 0,5 sur
  5/5 graines ; p < 0,01 sur 0/5.
  - P1 : AUC 0,53-0,55, contre 0,60-0,61 sur horscanon. Même sens, effet environ quatre
    fois plus faible (d 0,07-0,13 contre ≈ 0,30), non significatif.
  - P2, profil par position : **non reproduit, et inversé**. Horscanon donnait
    A 0,67-0,73 · B ≈ 0,46 · A′ ≈ 0,62. R4 donne A 0,44-0,45 · B 0,58-0,61 ·
    A′ 0,56-0,61. La position où l'effet était fort (A) est ici en sens contraire ; celle
    où il était nul (B) est ici la plus forte.
  - P2, effet de position à chiralité fixée : **0,81-0,83** (p 1e-4), contre 0,51 sur
    horscanon.
  - Longueur seule : p 0,020-0,024, non significatif au seuil de 0,01 fixé.

  **Observation descriptive, post hoc, sans effet sur le verdict.** Les 144 A′ de R4
  finissent par une ponctuation (« . »), contre 0/46 dans horscanon. La consigne le
  permettait (§3). Collée au dernier mot, cette ponctuation modifie son vecteur
  CamemBERT. C'est une explication plausible de l'effet de position massif (0,82), mais
  elle n'a pas été testée. Elle est commune à tous les A′ quelle que soit la chiralité,
  et P1 est stratifié par position ; elle ne suffit donc pas à expliquer l'absence de
  réplication. Toute mesure sans ponctuation serait un **nouveau** test, à
  pré-enregistrer comme tel, et non un sauvetage de R4.

  **Conclusion** : le signal R3 (AUC ≈ 0,60, fort en A, nul en B) ne se transporte pas
  à une autre écriture de la même consigne. Conformément au §5, il est attribué à
  l'écriture de horscanon. **Gate 2 reste fermée** pour toutes les représentations
  testées (33D segment, 33D phrase, CamemBERT). Prior de l'auteur du protocole :
  NON RÉPLIQUÉ ≈ 40 %.
