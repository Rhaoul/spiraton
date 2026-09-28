# Thermodynamique sémantique — résultats E0, E1, E3, E2 (chantier hors tour)

Date : 2026-09-28 · Claude Opus 5.5 · protocole : `docs/SEMANTIC_THERMO_PROTOCOLE.md`
(écrit avant mesure, révision R1 issue de E0 seulement) · run :
`runs/semantic_thermo/20260928_060936_all/` (manifest avec sha des corpus, du `.so` et
des sources ; `summary.md` = tables complètes) · graines 0-4 · commit de base `13b09ac`
(arbre modifié par ce chantier seul) · Python 3.13.3, torch 2.8.0+cpu.

**Verdict court : la lecture thermodynamique n'est pas supportée par cet instrument sur
ces corpus.** Les prédictions pré-enregistrées ne se vérifient pas ; les seules
régularités solides existent aussi dans des phrases non-ABA : elles tiennent à la
**position dans la phrase**, pas au cycle. Critères de réfutation du document directeur
rencontrés : §27-1, §27-2, §27-3, §27-7. Gates : **G0 passe (partiellement), G1 passe,
G2, G3 et G4 échouent** ; G5 et G6 n'ont pas été lancés.

## E0 — instrument (Gate 0)

Détail en R1 du protocole ; 18 tests `tests/test_semantic_thermo_observables.py`.

- **Passent** : densité k-NN ρ (↓ en expansion, ↑ en contraction), divergence de
  continuité `div = −d ln ρ/dt` (signe correct à d = 8 et d = 27), vitesse et flux nuls
  au repos, retour nul sur la boucle fermée, invariances exactes (translation,
  rotation, permutation du champ) et covariance d'échelle.
- **Échouent, à la lettre du document directeur** :
  - la divergence `J = ρv`, `(J_t − J_{t−1})·û` (§9.10), donne le **signe inversé** sur
    une expansion radiale (J diminue quand on gagne des régions rares, même à vitesse
    constante) ;
  - l'entropie softmax sur k voisins (§9.5) est **saturée** à ln k (plage < 0,5 %) ;
  - la température `Var(d_j)` (§9.7) **change de sens** selon la dimension (↓ à d = 8,
    ↑ à d = 27).
- **Fait de structure** : dans un champ k-NN, la seule entropie qui croît avec la
  dispersion (log-volume de Kozachenko-Leonenko) vaut `−d·ln ρ + cste`. Entropie et
  densité sont donc **une seule variable**. L'espace d'état à 9 variables du doc
  (§1) se réduit ici à {ρ, vitesse, dispersion, div}.

## E1 — DX/LV ↔ signe de divergence (Gate 2)

AUC = P(div d'un segment DX > div d'un segment LV) ; prédiction AUC > 0,5.

| test | no-logos, g0…g4 | IC 95 % bootstrap par cycle | lecture |
|---|---|---|---|
| **stratifié horscanon + f0b** (chiralité à position fixée, 199 segments) | 0,441 0,416 0,438 0,433 0,429 | [0,354 ; 0,528] … g1 [0,336 ; 0,495] | **nul, tendance CONTRAIRE** (p perm. médian 0,09) |
| canon `dataset_aba` (≡ A/B vs A′, 7068 segments) | 0,478 0,478 0,472 0,474 0,469 | exclut 0,5 **du côté contraire** | faible, contraire, porté par la position |
| canon `corpus_claude_aba` | 0,453 … 0,446 | inclut 0,5 | nul |
| longueur seule, canon (CTRL-LEN) | 0,307 | — | A′ est plus long (3,16 vs 2,50 tokens) : la longueur sépare bien mieux que div |

Contrôles : CTRL-2 (ordre des tokens mélangé) ramène la canon à 0,500 ; l'effet
contraire dépend donc de l'ordre, ce n'est pas un artefact d'ensemble. CTRL-7
(rotation orthogonale) donne une invariance exacte au bruit flottant près
(|Δdiv| ≤ 3e-14 après passage aux distances exactes) ; la projection JL d → d/2 conserve
l'AUC. Le résultat est identique sur toutes les tranches (full, no-energy, phoneme,
context). La tranche `full`, qui inclut les dims 4-5 d'orientation du tokenizer, ne
fait pas mieux.

**Conclusion E1** : la correspondance « DX = expansion (div > 0) / LV = contraction » **n'est
pas validée**. À position fixée, la faible tendance observée va même dans l'autre sens.
C'est le 3e null convergent sur la chiralité dans le 33D (T7 forme 0,538, T9 pente de
flux 0,466, ici divergence 0,43).

## E3 — A → B → A′ comme changement de phase (Gate 3)

Tests appariés par cycle, unilatéraux, Holm sur 5 ; 2356 cycles test par graine.
Fraction de cycles où l'inégalité prédite tient (g0…g4), tranche no-logos :

| hypothèse | dataset test | CTRL-CUT (frontières redistribuées) | CTRL-EVE (phrases non-ABA en tiers) | claude |
|---|---|---|---|---|
| H3.1a ρ_A > ρ_B (expansion) | **0,61-0,64**, p ≤ 1e-39 | 0,58-0,61, p ≤ 1e-16 | **0,67-0,70**, p ≤ 1e-17 | 0,41-0,48 |
| H3.2a ρ_A′ > ρ_B (recondensation) | 0,50 (nul) | 0,42 (contraire) | 0,55 | 0,41-0,45 |
| H3.1d D_B > D_A | 0,48-0,49 (nul) | 0,45 | 0,50 | 0,47 |
| H3.2d D_A′ < D_B | 0,36-0,38 (**contraire** : A′ plus dispersé, plus long) | 0,44 | 0,39 | 0,48 |
| H3.3 d(A,B) > d(A,A′) | 0,54-0,56, p ≤ 1e-6 | 0,47-0,50 (nul) | **0,57** | 0,39-0,41 (contraire) |

A′ ≠ A dans 100 % des cycles.

- H3.1a (expansion A→B) est significatif. Mais il est **plus fort dans les phrases
  d'eve**, qui ne sont pas des cycles ABA, et survit au recoupage aléatoire. Il vient de
  la tranche context (dims 23-30) et disparaît en tranche phoneme. C'est une propriété
  des **débuts de phrase** (mots fréquents/courts, rôle de phrase), pas de A.
- H3.2a (recondensation, le cœur de l'hypothèse de phase) est **nul**.
- H3.3 est significatif en dataset mais aussi dans les tiers d'eve, et contraire sur
  claude. C'est de la position dans la phrase, pas une clôture ABA.
- Sur le corpus claude (76 cycles écrits pour que le sens suive la forme), aucune
  hypothèse ne tient.

**Conclusion E3** : aucune signature A→B→A′ ne se distingue des contrôles de
position. La lecture « A condensé → B expansé → A′ recondensé » **n'est pas supportée**.

H3.4 (formes, horscanon + f0b) : n = 1 à 24 par forme, **descriptif seulement**
(tables dans `summary.md`). Aucune conclusion ; F5± ont chacune 1 cycle.

## E2 — opérateurs ↔ observables (Gate 4 partiel)

Régression logistique sur les agrégats thermo des 3 segments, entraînée sur la
calibration et testée hors calibration :

| modèle | dataset test | claude |
|---|---|---|
| classe majoritaire (MUL) | 0,390-0,405 | 0,360 |
| thermo | 0,390-0,403 (prédit MUL à 99,9 %) | 0,360 |
| thermo, étiquettes permutées | 0,389-0,404 | — |
| moyennes brutes de la tranche (non-thermo) | 0,376-0,394 | 0,28-0,36 |

Log-loss thermo ≈ 1,34 = entropie des fréquences de classes. **Aucune information sur
l'opérateur**, ni dans les observables thermo ni dans les traits bruts sans dims 0-5.
Ce résultat est cohérent avec T8/T10 : l'opérateur ABA ne se lit pas dans le socle
phonémique.

## Ce que ce passage établit, et ses limites

1. **Instrument.** La divergence de continuité est une bonne observable (E0). La
   divergence `ρv` du doc ne l'est pas, et entropie/température k-NN n'apportent rien
   de plus que la densité. C'est un résultat réutilisable, indépendant du langage.
2. **Langage.** Sur le 33D tokenisé par segment, ni la chiralité, ni le cycle, ni
   l'opérateur ne prennent de signature thermodynamique au-delà de la position dans la
   phrase.
3. **Limites** :
   - le test de chiralité non confondu repose sur 70 cycles : il ne détecte qu'un effet
     d'AUC ≳ 0,58 ;
   - `dataset_aba` est dominé par des cycles où A, B, A′ sont trois tiers d'une même
     phrase courte (2-3 mots par segment) : la « dynamique » d'un segment tient en 1 à
     2 pas ;
   - seul le 33D a été testé (pas d'embedding contextuel) ;
   - la densité est prise dans un champ de vocabulaire (lignes uniques), pas de
     fréquences.
4. **Non lancés** (hors périmètre de ce passage) : énergie libre (P5), prédiction de
   transition (E5), distribution de Boltzmann (E6), intégration générative (P7). Le
   doc directeur les conditionne à G2-G4, qui échouent : **les lancer serait contraire
   au protocole**, sauf nouvelle représentation.

## Si Ra veut poursuivre : une seule piste ne répète pas ce null

Les régularités solides (H3.1a, H3.3) sont des effets de **position dans la phrase**
visibles dans eve comme dans l'ABA. Les observables thermo mesurent donc quelque chose
de réel, mais c'est la structure de la phrase, pas le Logos. Une suite honnête
changerait de **représentation** (vecteurs contextuels, ou 33D tokenisé sur la phrase
entière avec le rôle de phrase des dims 28-30), et non de définition d'observable. Sans
ce changement, E1/E3 se reproduiront à l'identique.

## Fichiers

- `spiraton/experimental/semantic_thermodynamics/` : `state.py`, `observables.py`,
  `synthetic.py` (instrument, sans étiquette).
- `spiraton/data/semantic_thermo_adapter.py` : tranches, tokenisation par segment,
  partage seedé, CTRL-CUT, CTRL-EVE.
- `spiraton/diagnostics/semantic_thermo_probe.py` : E1/E3/E2, statistiques, manifeste.
  CLI : `python -m spiraton.diagnostics.semantic_thermo_probe --experiment all --seeds 0,1,2,3,4`.
- `scripts/summarize_semantic_thermo.py` : `summary.md` d'un run.
- Tests : `tests/test_semantic_thermo_observables.py` (18),
  `tests/test_semantic_thermo_controls.py` (8). Suite complète : 810 passed / 3 skipped.
  MODEL_CARD inchangé. `runs/` ajouté au `.gitignore`.

---

# Addendum R2 — 33D tokenisé sur la phrase entière (2026-09-28, demande de Ra)

Run : `runs/semantic_thermo/20260928_070219_all_sentence/` (5 graines, 7 tranches,
protocole R2 écrit avant mesure). Alignement phrase/segments exact sur 5146/5146
cycles ; dims de forme identiques au bit. **Seules les dims 28-30 changent** : 28 porte
le rôle A/B/A′ que l'heuristique `phrase.c` du tokenizer devine sur le texte, et 29-30
deviennent constantes sur la phrase.

**Contrôle d'intégrité (prédit) :** tranche `phoneme` identique au bit au run segment,
sur toutes les AUC et toutes les graines. ✔

**Verdict : inchangé.** Le mode phrase fait apparaître de nouveaux effets. Tous
viennent des dims de contexte de phrase du tokenizer (28-30), tous existent aussi dans
les phrases non-ABA d'eve, et la tranche `form-only` (sans 28-30) retrouve les nulls
du mode segment.

## E1

| test (AUC, g0…g4) | no-logos | no-role | form-only | context |
|---|---|---|---|---|
| **chiralité à position fixée** (horscanon + f0b) | 0,511 0,509 0,516 0,498 0,521 (p 0,94) | 0,46-0,48 | 0,45-0,47 | 0,54-0,56 (p 0,32) |
| position A/B vs A′ à chiralité fixée | **0,62** [0,543 ; 0,704], p 0,003 | 0,61, p 0,006 | **0,50** | 0,51-0,52 |
| canon dataset (≡ position) | 0,444-0,456 (contraire) | 0,44-0,46 | 0,435-0,453 | 0,43-0,45 |

- La chiralité reste illisible (0,51).
- Un effet de **position** apparaît dans les corpus mixtes (A/B diverge plus que A′,
  dans le sens de la prédiction). Mais il vient des dims 29-30 : il survit à `no-role`
  et disparaît en `form-only`. Et il **change de signe** sur le corpus canonique
  (0,45, contraire). Un même contraste positionnel qui s'inverse d'un corpus à l'autre
  n'est pas une loi.

## E3 (fraction de cycles où l'inégalité prédite tient, g0…g4)

| hypothèse, tranche | dataset | CTRL-CUT | CTRL-EVE (non-ABA) | claude |
|---|---|---|---|---|
| H3.2a recondensation ρ_A′ > ρ_B, no-logos | 0,54 (p Holm 0,005) | 0,46 | **0,60** | 0,45-0,51 |
| même, no-role | 0,43-0,46 (contraire) | 0,37 | 0,55 | 0,45-0,52 |
| même, form-only | 0,51-0,54 (n.s.) | 0,50-0,53 | 0,55 | 0,49-0,52 |
| H3.1a ρ_A > ρ_B, no-logos | 0,63-0,65 | 0,59-0,62 | 0,62-0,64 | 0,39-0,45 |
| H3.3 d(A,B) > d(A,A′), form-only | 0,57-0,59 | 0,49-0,50 | 0,58 | 0,40 |
| context : H3.1d / H3.2a / H3.2d | 0,59 / 0,59 / 0,61 | 0,53 / 0,50 / 0,63 | **0,66 / 0,65 / 0,64** | 0,31 / 0,48 / 0,60 |

- La **recondensation** (H3.2a), nulle en mode segment, devient significative en
  `no-logos`. Elle vient de la dim 28 (rôle deviné par le tokenizer) : sans cette dim,
  elle s'inverse, et elle est plus forte dans les phrases d'eve. Le tokenizer marque
  « A′ » sur n'importe quelle fin de phrase ; la « recondensation » est cette marque,
  pas une dynamique du cycle.
- En tranche `context`, le motif complet expansion/recondensation apparaît. Mais il
  est **plus fort dans eve que dans l'ABA** et contraire sur claude : c'est la
  signature des dims de phrase du tokenizer.
- H3.3 vit dans les dims de forme (`form-only`), existe dans eve et disparaît sous
  CTRL-CUT, qui ne change que les frontières en gardant les vecteurs. Il dépend de la
  répartition des longueurs (A′ plus long, centroïde plus moyenné), pas du contenu ABA.

## E2

Inchangé : thermo = classe majoritaire (0,386-0,405), égale aux étiquettes permutées,
sur `no-logos`, `no-role` et `phoneme`. Aucune information sur l'opérateur.

## Lecture

Le 33D sur la phrase entière n'ajoute que le contexte de phrase calculé par le
tokenizer. Ce contexte encode une position (début/fin de phrase, rôle A/B/A′
heuristique) qui produit, dans des phrases quelconques, un motif ressemblant à une
recondensation. Cela montre deux choses :

1. L'instrument voit ce que le 33D contient. Le résultat nul du mode segment ne venait
   donc pas d'un instrument aveugle.
2. Ce que le 33D contient, c'est la position dans la phrase, pas le cycle ABA. Le
   risque de **circularité** (§28.3 du doc directeur) se matérialise exactement ici :
   la dim 28 a été construite pour parler ABA, et c'est elle qui « découvre » la
   recondensation.

Critères de réfutation §27-1, 2, 3 toujours rencontrés. Une suite exigerait une
représentation qui ne soit ni phonémique ni construite par le tokenizer pour le Logos
(vecteurs contextuels appris sur un corpus externe), et un corpus où la position dans
la phrase est décorrélée de A/B/A′.

---

# Addendum R3 — vecteurs contextuels CamemBERT (2026-09-28, demande de Ra)

Run : `runs/semantic_thermo/20260928_080624_all_camembert_sentence/` (5 graines ;
`camembert-base` révision `a7596756…`, poids sha256 `486643fd…` au manifeste ; dernière
couche, moyenne des sous-mots, phrase encodée entière, ACP 64 sur la calibration).
Contrôle de confond post hoc : `confound_posthoc.json` du même run, script
`scripts/semantic_thermo_confound_r3.py`.

**Verdict : premier signal non nul du chantier, limite, sur E1 seulement. E3 et E2
restent nuls.**

## E1 — le résultat

| test (AUC DX > LV, g0…g4) | valeur | IC 95 % bootstrap | p perm. bilatéral |
|---|---|---|---|
| **chiralité à position fixée** (horscanon + f0b, 199 segments) | 0,598 0,610 0,599 0,602 0,603 | exclut 0,5 sur **5/5** (bornes basses 0,526-0,542) | 0,011 0,009 0,015 0,008 0,009 |
| position A/B vs A′ à chiralité fixée | 0,501-0,527 | inclut 0,5 | 0,84 |
| longueur seule (strat.) | 0,503 | — | 0,95 |
| CTRL-2, ordre des mots mélangé (strat.) | 0,489-0,504 | — | — |
| CTRL-7, rotation orthogonale / projection JL d/2 | identique / 0,55-0,62 | — | — |
| canon dataset / claude (≡ position) | 0,74-0,75 / 0,68-0,69 | — | 1e-4 |

Contrôle post hoc (déclaré après lecture) du confond de **corpus**. À position B, les
DX viennent surtout de horscanon (T31) et les LV surtout de f0b (T36).

| variante (g0…g4) | AUC | IC 95 % | p |
|---|---|---|---|
| strates position × corpus | 0,598-0,610 | exclut 0,5 sur 5/5 | 0,011-0,023 |
| **horscanon seul** (un lot, DX et LV à chaque position) | 0,599-0,612 | [0,509-0,519 ; 0,689-0,704], exclut 0,5 sur 5/5 | 0,026-0,051 |
| AUC par position, horscanon seul | A : 0,67-0,73 · **B : 0,45-0,48** · A′ : 0,61-0,63 | — | — |

**Lecture.**
- Dans l'espace contextuel, et à position fixée, un segment écrit DX/OUT gagne
  davantage les régions rares du champ (divergence de continuité plus positive) qu'un
  segment écrit LV/IN. C'est le sens prédit par le document directeur (§5.1).
- Le signal n'est pas la longueur (0,50), pas la position (0,51), pas un effet
  d'ensemble (CTRL-2 le détruit : il dépend de l'ordre des mots), et pas le corpus
  d'origine (il tient sur horscanon seul).
- Effet modeste (d de Cohen ≈ 0,30). Il est absent en position B.

**Pourquoi ce n'est pas encore une validation.**
1. Critère pré-enregistré (§4) : l'IC exclut 0,5 sur 5/5 ✔ ; p < 0,01 sur **3/5
   graines seulement** ✘ (0,008-0,015) ; pas reproduit par la longueur ✔. Au sens
   strict, **Gate 2 n'est pas franchie**.
2. **Chemins multiples** : c'est la 3e représentation essayée (33D segment, 33D phrase,
   CamemBERT). Une correction ×3 porte p à ≈ 0,03.
3. **Corpus conçu** : horscanon a été écrit par le linguiste pour que « la forme SOIT le
   sens ». Un segment DX y est rédigé pour « sortir », un segment LV pour « revenir ».
   Le signal peut refléter cette intention d'écriture (le choix des mots par un auteur
   unique) plutôt qu'une loi du langage. C'est déjà une information : la sémantique
   contextuelle **voit** l'intention d'orientation que le 33D ne voyait pas (T7, T9,
   R1, R2).
4. Petit effectif : 134 segments sur horscanon seul, puissance faible, IC larges.

## E3 — A → B → A′

Aucune signature propre à l'ABA :
- H3.1a ρ_A > ρ_B tient à 72-75 % en dataset mais **autant sous CTRL-CUT** (74-75 %), et
  aussi dans eve (64-66 %). C'est la position dans la phrase.
- H3.2a (recondensation) est nulle en dataset (0,50-0,51), présente dans eve (0,72) :
  c'est une fin de phrase.
- H3.1d, H3.2d, H3.3 sont **contraires** partout, y compris dans les contrôles.

## E2 — opérateur

Classe majoritaire (0,385-0,403 contre 0,390-0,405), égale aux étiquettes permutées.
Même la sémantique contextuelle ne porte pas l'opérateur du dataset (prior de l'auteur
à 55 % : faux).

## Ce qui départagerait (proposition, non lancée)

Une **réplication pré-enregistrée sur un corpus frais** : des cycles de formes mixtes
(DX et LV à chaque position), écrits par une autre main que celle de horscanon (Ra, ou
une génération dont l'auteur n'a pas lu ce résultat), mesurés avec le pipeline R3
**gelé** (même modèle, même ACP ajustée sur `dataset_aba`, même critère). Prédiction
figée d'avance : AUC stratifiée > 0,5 avec IC excluant 0,5, et effet nul en position B.
Si elle réplique, Gate 2 passe pour la représentation contextuelle. Sinon, le signal
était celui d'un auteur.
