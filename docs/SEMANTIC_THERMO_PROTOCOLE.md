# Thermodynamique sémantique — protocole pré-enregistré (chantier hors tour)

Document directeur : `brainstorming/SEMANTIC_THERMODYNAMICS.md` (racine de
l'atelier). Statut : **expérimental, non canonique**. Ce fichier est écrit
**avant** toute mesure sur données réelles ; les définitions ne sont modifiables
qu'à l'étape E0 (instrument synthétique, sans étiquette Logos). Toute
modification ultérieure doit être ajoutée en bas, datée, jamais réécrite.

Rédigé le 2026-09-28 (Claude Opus 5.5), avant exécution de E0.

## 0. Faits structurels établis avant mesure (lecture des corpus, pas des scores)

| Corpus | Cycles | Motif de chiralité (A, B, A′) |
|---|---|---|
| `dataset_aba.txt` | 5000 (269 points fixes A=B=A′, 22 doublons) | 100 % (DX, DX, LV) |
| `corpus_claude_aba.txt` | 76 | 100 % (DX, DX, LV) |
| `corpus_horscanon_aba.txt` | 46 | 7 motifs (15 F0, 14 DX-LV-DX, 8 LV-DX-LV, …) |
| `corpus_f0b_aba.txt` | 24 | 100 % (DX, LV, LV) |

Conséquence : **sur les corpus canoniques, « DX vs LV » est identique à
« A/B vs A′ »** (position dans la phrase). Le test E1 sur ces corpus ne peut pas
séparer chiralité et position ; il est rapporté comme **E1-canon (confondu)**.
Le seul test de chiralité *au-delà de la position* est **E1-strat** sur
horscanon + f0b (70 cycles, 210 segments), avec permutation des étiquettes
**à l'intérieur de chaque position**. Petits effectifs : on s'attend à une
puissance faible, et on le dira.

Antécédents : T7 (forme 33D, AUC 0,538) et T9 (pente de flux, AUC 0,466) ont
déjà trouvé la chiralité illisible dans le 33D. Prior de l'auteur : E1-strat
nul (≈ 75 %).

## 1. Unité, entrée, anti-fuite

- Unité : le **segment** (A, B ou A′) d'un cycle ; un segment est une
  trajectoire de tokens `x_1 … x_n`, vecteurs 33D produits par le tokenizer
  natif sur le **texte nettoyé du segment seul** (`seg.text`, tags retirés).
  Le vecteur d'un mot ne dépend donc pas de la position du segment dans le
  cycle (verrou hérité de T7/T9).
- Tranches (`data/semantic_thermo_adapter.py`) :
  `full` 0-32 · **`no-logos` 6-32 (primaire)** · `no-energy` 8-32 (CTRL-4) ·
  `phoneme` 8-22 (CTRL-5) · `context` 23-30 (CTRL-6).
  Les dims 0-5 ne servent jamais aux observables des expériences E1-E3.
- Normalisation : z-score par dimension avec moyenne/écart-type du **pool de
  calibration** ; dimension d'écart-type < 1e-8 mise à 0.
- Champ de référence R : lignes **uniques** des vecteurs (normalisés) des
  tokens des cycles de calibration de `dataset_aba.txt` (moitié des cycles,
  tirage seedé par cycle, points fixes et doublons exclus). Le champ est la
  géométrie du vocabulaire, pas sa fréquence. Tous les autres corpus sont
  entièrement hors calibration.

## 2. Observables (définitions de départ, révisables en E0 seulement)

Par token `t`, avec `d_j` les distances aux `k = 8` plus proches voisins dans R :

- densité `ρ_t = 1 / (mean_j d_j + ε)` (doc §9.4) ;
- entropie `S_t = −Σ p_j log p_j`, `p = softmax(−d / h)`, `h` = médiane de
  `mean_j d_j` sur le pool de calibration, gelée (doc §9.5) ;
- température `T_t = Var_j(d_j)` (doc §9.7) ;
- vitesse `v_t = x_t − x_{t−1}`, `speed_t = ‖v_t‖` (doc §9.2) ;
- énergie cinétique `E_kin = ½‖v_t‖²` ; énergie 33D `E_33D = dim 6 brute`.

Par segment (trajectoire) :

- **divergence primaire (équation de continuité lagrangienne)** :
  le long d'une trajectoire, `d ln ρ / dt = −∇·v`, donc
  `div_seg = − pente OLS de ln ρ_t contre t` (n ≥ 2). Positive si la
  trajectoire va vers des régions plus rares du champ (expansion).
- divergence littérale du doc (§9.9-9.10), rapportée en secondaire :
  `J_t = ρ_t v_t`, `div^J = moyenne_t (J_t − J_{t−1})·û`,
  `û = (x_n − x_1)/‖x_n − x_1‖` (n ≥ 3).
- dispersion `D_seg = moyenne_t ‖x_t − centroïde‖` ;
- agrégats `ρ̄, S̄, T̄, speed̄` = moyennes sur les tokens.

## 3. E0 — instrument (synthétique, avant toute étiquette)

Champ gaussien isotrope seedé, trajectoires : expansion radiale, contraction,
stationnaire, marche aléatoire, boucle fermée, diffusion (rayon ∝ √t).
Attendus du doc : densité ↓ en expansion, ↑ en contraction ; divergence de
signe correct ; vitesse et flux nuls au repos ; entropie ↑ avec la dispersion ;
retour mesurable sur la boucle. Invariances : translation, rotation
orthogonale, permutation des lignes de R (exactes), échelle (ρ ∝ 1/c,
T ∝ c², S et signe de div invariants).

**Règle** : une observable qui échoue E0 est soit remplacée *pendant E0*
(la révision est consignée en §8), soit exclue ; elle n'est jamais conservée
en espérant qu'elle marche sur le langage.

## 4. E1 — DX/LV ↔ divergence

- Score : `div_seg` (primaire), `div^J` (secondaire). Prédiction :
  `E[div | DX] > E[div | LV]`, AUC(DX > LV) > 0,5.
- **E1-canon** : segments test de `dataset_aba` + tout `corpus_claude_aba`.
  Confondu avec la position (rapporté comme tel).
- **E1-strat** : horscanon + f0b ; p par permutation des étiquettes DX/LV
  **à position fixée** (10 000 permutations) ; plus E1-pos (A′ vs A/B) pour
  mesurer ce que la position seule porte.
- Mesures : moyenne, médiane, Cohen's d, AUC, IC bootstrap 95 % (rééchantillonnage
  par cycle, 2000), p de permutation, pour 5 graines {0,1,2,3,4} (la graine
  fixe le partage calibration/test, les permutations et le bootstrap).
- Contrôles : CTRL-1 (étiquettes permutées), CTRL-2 (ordre des tokens
  mélangé dans chaque segment, 20 tirages), CTRL-3/4/5/6 (tranches),
  CTRL-7 (projection orthogonale carrée : invariance exacte attendue ;
  projection aléatoire vers d/2 : survie partielle attendue),
  CTRL-LEN (AUC de la seule longueur du segment).
- Critère : effet retenu seulement si l'IC bootstrap de l'AUC exclut 0,5 sur
  5/5 graines **et** p_perm < 0,01 **et** l'effet n'est pas reproduit par la
  longueur seule. Sinon : « correspondance DX=expansion non validée par cet
  instrument ».

## 5. E3 — cycle A → B → A′

Sur les cycles test de `dataset_aba` (points fixes exclus), apparié par cycle,
unilatéral, signe de la différence + Wilcoxon (approximation normale) +
IC bootstrap de la médiane des différences :

- H3.1a `ρ̄_B < ρ̄_A` · H3.1b `S̄_B > S̄_A` · H3.1c `T̄_B > T̄_A` · H3.1d `D_B > D_A`
- H3.2a `ρ̄_A′ > ρ̄_B` · H3.2b `S̄_A′ < S̄_B` · H3.2d `D_A′ < D_B`
- H3.3 `d(A, A′) < d(A, B)` (distances entre centroïdes) et `d(A, A′) > 0`.

Correction de Holm sur les 8 tests. Contrôles décisifs :
**CTRL-CUT** (même phrase A+B+A′ recoupée à des frontières aléatoires, mêmes
longueurs de segments permutées) et **CTRL-EVE** (phrases non-ABA de
`corpus_eve_clean.txt` coupées en trois) : si la signature y survit, elle est
une propriété de *position dans la phrase*, pas du cycle ABA.
H3.4 (formes) : descriptif par forme sur horscanon + f0b, IC bootstrap ;
aucune conclusion inférentielle sous 10 cycles par forme.

## 6. E2 — opérateurs ↔ observables

Classifieur logistique multinomial (torch, L2 = 1e-2 fixé, 500 pas LBFGS),
entraîné sur la calibration de `dataset_aba`, testé sur son test, puis sur
`corpus_claude_aba` (hors distribution). Entrées : agrégats thermo (ρ̄, S̄,
T̄, speed̄, D, div) des trois segments, tranche `no-logos`. Comparaisons :
classe majoritaire, longueurs seules, moyennes brutes de la tranche (baseline
non thermo), étiquettes permutées (20 réentraînements). Métriques : accuracy,
log-loss, accuracy par classe.

## 7. Hors périmètre de ce premier passage

Énergie libre (P5), prédiction de transition (E5), distribution physique (E6),
intégration générative (P7) : non lancés tant que E0-E3 n'ont pas été lus par Ra.

## 8. Révisions (ajout seulement)

### R1 — issue de E0 (2026-09-28, synthétique seul, aucune étiquette vue)

Champ N(0, I_d), k = 8, d ∈ {8, 27} (27 = dimension de `no-logos`).

| Observable | Expansion | Contraction | Repos | Verdict E0 |
|---|---|---|---|---|
| ρ | ↓ (d=8 : 1,04→0,57 ; d=27 : 0,305→0,270) | ↑ | constant | **passe** |
| div (continuité) | + (0,056 ; 0,010) | − | 0 | **passe** |
| div^J littérale (doc §9.10) | **−** (−0,010 ; −0,0009) | **+** | 0 | **échoue** (signe inversé : J = ρv décroît quand on s'éloigne, même à vitesse constante) |
| S softmax (doc §9.5) | 2,074→2,078 ; 2,0791→2,0782 | — | — | **échoue** : saturée à ln 8 = 2,079 (plage < 0,5 % de ln k) ; ne mesure rien |
| T = Var(d_j) (doc §9.7) | ↓ à d=8, **↑ à d=27** | — | — | **échoue** : sens dépendant de la dimension, aucune lecture stable |
| vitesse / flux | > 0 | > 0 | 0 / 0 | passe |
| retour | 1,0 | 1,0 | 0 | passe (boucle fermée : 0 ; marche aléatoire : 0,46) |

Décisions (prises sur E0 seul) :
- **div^J, S softmax, T** : calculés et rapportés à titre descriptif,
  **exclus de toute inférence**. div^J reste le témoin d'échec de la
  définition littérale.
- Pas de remplacement de S : l'entropie « log-volume » de Kozachenko-Leonenko,
  seule variante qui croît avec la dispersion, vaut `−d·ln ρ + cste` dans ce
  formalisme — **entropie et densité sont une seule variable** ici (critère de
  réfutation §27-7 du doc directeur, rencontré dès l'instrument).
- E3 : H3.1b, H3.1c, H3.2b retirés ; Holm sur les 5 tests restants
  (H3.1a, H3.1d, H3.2a, H3.2d, H3.3).
