# CLAUDE.md — Spiraton

Ce fichier guide Claude Code dans ce dépôt. Lis-le avant toute modification.

## Ce qu'est ce projet

Spiraton est une cellule de calcul fondée sur le **Logos Opératoire** : quatre
opérateurs primitifs (addition, soustraction, multiplication, division) plus
une **orientation** (dextrogyre = expression/expansion vers le dehors,
lévogyre = réception/contraction vers le dedans). La dynamique visée est le
cycle **A → B → A′** : émission, déploiement, retour transformé — la spirale.
Le but n'est pas de simuler la cognition par probabilités brutes, mais de
construire des cellules qui **réfléchissent et composent** plutôt que répéter.

L'intention fait partie du modèle : `MANIFESTE.md`, `REFUS.md` et
`CODE_OF_CONDUCT.md` sont intégrés au package et au MODEL_CARD. Toute
contribution qui rend le cœur opaque "by design", ou qui produit "du symbole
sans test", est hors périmètre (voir `REFUS.md`).

## Fondement théorique (THEORIE_LOGOS.md)

Le document fondateur formalise le Logos Opératoire : espace d'états S,
quatre opérations primitives (⊕ agrégation, ⊖ distinction, ⊗ amplification,
⊘ ramification), et deux opérateurs temporels D (dextrogyre, action) et
L (lévogyre, mémoire) qui **ne sont pas inverses l'un de l'autre**. Trois
axiomes y sont directement testables dans ce dépôt, et toute implémentation
doit pouvoir être confrontée à eux :

1. **Contextualité** : les opérations ne sont pas nécessairement
   commutatives. → C'est la justification théorique du chantier 1
   (poids matriciels).
2. **Double dynamique** : la composition L∘D tend à stabiliser des
   attracteurs ; D∘L tend à engendrer des bifurcations. → Prédiction
   falsifiable : composer `SpiralGrid(outward)` puis `inward` ne doit pas
   donner la même dynamique alpha-oméga que `inward` puis `outward`
   (voir chantier 2).
3. **Dynamique du second ordre** : s_{t+1} = D(A(s_t) + B(s_t²) − C(s_{t−1}))
   + L(s_t). L'état précédent s_{t−1} entre dans la mise à jour ; le terme
   B(s²) est quadratique. → `RecursiveSpiraton` est aujourd'hui du premier
   ordre (seul `update="momentum"` garde une trace) ; le ChronoSpiraton
   (chantier 5) doit incarner cette équation, pas seulement une EDO
   générique.

## Écosystème à deux dépôts

Ce dépôt (`spiraton`, PyTorch) contient les cellules. Le dépôt
`spiraton-tokenizer` (C + wrapper ctypes) contient le tokenizer
phonético-sémantique v4.1 qui produit des vecteurs **33D** par token, dont
les dims 0-5 (scores ADD/SUB/MUL/DIV, dextro/lévo) parlent exactement la
langue des balises ABA ci-dessous. Il possède son propre CLAUDE.md ; le lire
avant tout travail trans-dépôt. La phase spiratons a la priorité, mais le
tokenizer se travaille en spirale selon le principe de la "boucle
socle ↔ corpus" décrit dans son CLAUDE.md : la justesse du socle C et la
fermeture de la boucle ABA se valident mutuellement, elles ne se séquencent
pas.

## Commandes

```bash
pip install -e ".[dev]"          # installation dev (torch>=2.0 requis)
pytest                            # tests (déterministes, seeds fixés)
python scripts/gen_model_card.py  # régénérer MODEL_CARD.md
git diff --exit-code MODEL_CARD.md  # la CI vérifie qu'il est à jour
```

- `MODEL_CARD.md` est **généré** : ne jamais l'éditer à la main.
- CI : CPU-only, Python 3.9/3.10/3.11. Tout doit passer sans GPU.
- Licence : **GPL-3.0-or-later**. Pas de code copié de sources incompatibles,
  pas de données/poids sans provenance documentée (voir `CONTRIBUTING.md`).

## Carte conceptuelle → code

| Concept | Code |
|---|---|
| Les 4 opérateurs | `spiraton/core/operators.py` |
| Chiralité dextro/lévo (mode) | `spiraton/core/modes.py`, `spiraton/core/mode_policy.py` |
| Cellule canon (stable) | `spiraton/core/cell.py` → `SpiratonCell` |
| Cellule expérimentale (gates + coeffs) | `spiraton/experimental/gated_cell.py` |
| Adaptation second ordre | `spiraton/experimental/adaptation.py` |
| Propagation spirale spatiale | `spiraton/grid/spiral_grid.py` |
| Cycle A → B → A′ (récursion) | `spiraton/recursion/recursive.py` |
| Mesure du retour (clôture du cycle) | `spiraton/diagnostics/alpha_omega_spatial.py` |

## Le canon (invariants à ne pas casser)

API publique v0.1, petite et stable :
`from spiraton import SpiratonCell, GatedSpiratonCell`.
Le reste est semi-public. Tout changement de comportement du canon exige :
discussion (issue), tests de formule exacte mis à jour, et mention explicite.

Formules canon (vérifiées à l'identique par `tests/test_core_cell.py`) :

```
add = Σ(x · w_add)
sub = Σ(x − w_sub)
mul = tanh( Σ(w_mul · log(|x|+eps)) )
div = tanh( −Σ(w_div · log(|x|+eps)) )
raw_dextro   = add + mul − div    → activation tanh
raw_levogyre = sub + div − mul    → activation atan
```

Sémantique : le dextrogyre additionne et amplifie (sortie), le lévogyre
soustrait et divise (retour). Les deux branches sont des miroirs — ce
retournement est structurel, pas décoratif.

Autres invariants :
- `spiral_indices` : départ au centre, anneaux horaires ; `inward` est
  l'inverse exact de `outward`. La propagation de `SpiralGrid` est
  **séquentielle à dessein** (chaque cellule voit les mises à jour
  précédentes le long de la spirale). Ne pas "vectoriser" cette boucle d'une
  manière qui rendrait les mises à jour simultanées : ce serait changer la
  physique du système, pas l'optimiser. Des accélérations qui préservent
  l'ordre (moins d'allocations, indexation précalculée) sont bienvenues.
- `run_alpha_omega_spatial` : le score `cos − l2` mesure le **retour à
  l'origine transformé** (best_return_step). C'est le diagnostic central du
  projet : un système qui ne revient pas se perd ; un retour identique est
  une répétition ; on cherche le retour *proche et aligné mais non identique*.
- Tests : seeds fixés, finitude (pas de NaN/Inf), formes simples/batch,
  formule exacte sous paramètres forcés, flux de gradient. Toute nouvelle
  cellule ou opérateur doit reproduire ce quatuor de tests.

## Le vocabulaire ABA (interface données)

Les corpus d'entraînement (hors dépôt, attendus sous `data/`, provenance à
documenter) suivent une grammaire fixe. Vocabulaire de balises **canonique** :

```
Structure : <SEG_A> </SEG_A> <SEG_B> </SEG_B> <SEG_A_PRIME> </SEG_A_PRIME>
Opérateurs : <ADD> <SUB> <MUL> <DIV>      (constant sur tout le cycle)
Chiralité  : <DX> <LV>    Direction : <OUT> <IN>
Position   : <ALPHA> <OMEGA> <A_PRIME>
Fins       : <EOL> <EOS>
```

Schéma d'une ligne :

```
<SEG_A> <OP><DX><OUT><ALPHA> ... </SEG_A> <SEG_B> <OP><DX><OUT><OMEGA> ... </SEG_B> <SEG_A_PRIME> <OP><LV><IN><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

Le retournement `<DX><OUT>` → `<LV><IN>` au troisième segment encode la
clôture spirale. Corpus connus : `corpus_eve_clean.txt` (876 phrases,
progression pédagogique qui culmine en énonçant le cycle A→B→A′),
`dataset_aba.txt` (~5000 cycles), `corpus_claude_aba.txt` (76 cycles, lecture
de Claude ; opérateur aligné sur le geste sémantique de la phrase),
`corpus_horscanon_aba.txt` (46 cycles gelés T31, formes non canoniques),
`corpus_f0b_aba.txt` (24 cycles gelés T36, forme F0b attestée).

**Changement de canon-données (Tour 35, décision explicite de Ra)** : la
grammaire ABA est **étendue** aux formes d'arrangement per-segment. Le schéma
ci-dessus (`<DX><OUT>`×2 puis `<LV><IN>`) est la forme **F0**, qui reste la
clôture canonique ; s'y ajoutent officiellement les formes mesurables
**F0b (+,−,−)** (attestée en corpus T36, `corpus_f0b_aba.txt`), **F1 (−,+,+)**, **F1b (−,−,+)**, **F2 (+,−,+)**,
**F3 (−,+,−)**, le modificateur **F4** (SEG_B vide) et les deux pôles
dégénérés **F5± (uniformes, nommés-non-mesurés)**. Définitions, schémas de
ligne, sémantique ancrée corpus et exclusions motivées (`<DX><IN>`/`<LV><OUT>`
restent hors grammaire) : voir **`GRAMMAIRE_ABA.md`** (racine du dépôt) ;
référence Python : `spiraton/data/aba_forms.py`
(`classify_form_extended`, total — cohérence rétrograde 100 % avec le
`classify_form` gelé de `diagnostics/profile_exact.py`).

## Chantiers prioritaires (issus des discussions avec Ra)

Travailler dans `spiraton/experimental/` d'abord ; promotion vers `core/`
seulement après stabilisation et tests.

1. **Poids matriciels (non-commutativité).** Les `w_op` actuels sont des
   vecteurs : les opérateurs commutent trivialement. Créer
   `MatrixSpiratonCell` où chaque opérateur est une application linéaire
   `W_op ∈ R^{d×d}` et où la sortie dépend de l'**ordre de composition**.
   Ajouter un diagnostic de non-commutativité acquise : norme du commutateur
   `‖W_a W_b − W_b W_a‖` suivie pendant l'entraînement. Reproduire le quatuor
   de tests + un test prouvant que permuter l'ordre change la sortie.
   (Incarne l'axiome de contextualité.)
2. **Test de la double dynamique L∘D vs D∘L.** Nouveau diagnostic dans
   `spiraton/diagnostics/` : appliquer K pas de grille `outward` puis K pas
   `inward`, et l'inverse, sur le même x0 ; comparer les séries alpha-oméga
   (variance, score, best_return_step). La théorie prédit une asymétrie
   mesurable (L∘D stabilise, D∘L bifurque). Si elle n'apparaît pas, c'est un
   résultat — le documenter, ne pas forcer le code à la produire.
3. **Pont tokenizer 33D ↔ cellules.** Le pont C↔Python existe déjà
   (`SpiratonTokenizerV4`, ctypes, dépôt tokenizer). Ici : un adaptateur
   `data/` qui consomme les vecteurs 33D (`input_size=33`), des tests de
   parité figés (mots témoins → vecteurs attendus, pour détecter toute
   dérive d'ABI), et une référence Python lisible du parseur ABA
   (ligne → triplet `{op, segments, chiralité, direction}`). Le `.so` n'est
   jamais la seule source de vérité.
4. **Embeddings par opérateur.** Une table d'embedding distincte par
   opérateur et par chiralité, pas un vocabulaire plat. Les dims 0-5 du 33D
   fournissent les scores qui pondèrent ces tables : l'opérateur conditionne
   la représentation, il ne s'y ajoute pas.
5. **Entraînement sur cycles ABA.** Un loader `data/` → batches, et une boucle
   d'entraînement minimale où la perte inclut le signal alpha-oméga : Φ(A)
   prédit doit être proche-et-aligné avec le **A′ réel du cycle** — la vérité
   de terrain du corpus — sans être la copie de A (pénaliser la copie exacte :
   la répétition est le mode dégénéré ; l'identité vaut zéro sur la métrique
   appariée par cycle). Lecture confirmée par Ra (post-T64, question §11-2 de
   l'émission). Le régime **génératif** — la cellule émet son propre A′ sans
   vérité de terrain, jugé par un critère externe gelé (p. ex. tomber dans la
   distribution géométrique des A′ réels) — est une étape future distincte,
   à geler comme un tour propre le jour venu.
6. **ChronoSpiraton (durée).** Prolonger `RecursiveSpiraton` vers l'équation
   du second ordre de la théorie : mémoire explicite de s_{t−1} (terme
   −C(s_{t−1})) et terme quadratique B(s²), puis passage au temps continu.
   Strictement expérimental, benchmarks de stabilité obligatoires.
7. **Phonétique → sémantique.** Les dims 8-22 du 33D (signatures
   d'impédance, de flux, séquences de spins) sont déjà les traits phonétiques
   cherchés : les exploiter comme canaux d'entrée des cellules plutôt que
   d'en recalculer. Ne lancer qu'après 3 et 4.

## Points de vigilance connus (état honnête du dépôt)

- `subtractive(x, w) = Σ(x − w) = Σx − Σw` : dégénéré — le gradient sur `w`
  est constant et l'appariement élément-à-élément ne joue aucun rôle. C'est
  un candidat naturel de la migration matricielle ; ne pas le "corriger" en
  douce dans le canon (test de formule exacte en dépend).
- `mul/div` en log-domaine perdent le **signe** (`|x|`) et sont sensibles à
  `eps` près de zéro. Documenté ; toute alternative (log signé, soft-abs)
  passe par experimental.
- `dextro_mask` (moyenne ≥ 0) est volontairement simpliste ; pour de vraies
  tâches, préférer `LearnedGateMode` (différentiable) via `mode_policy`.
- Versions incohérentes : `pyproject.toml` dit 0.1.0, `__init__.py` dit
  0.2.0-dev. À réconcilier (source unique de vérité dans `__init__`).
- Le workflow CI vit sous `github/workflows/` : vérifier qu'il est bien dans
  `.github/workflows/`, sinon il ne s'exécute pas.
- `README.md` : placeholders `YOUR_USERNAME`, liste de fichiers mal formée.

## Style de travail

- **La nuance est une information.** Préférer un petit changement juste et
  testé à une refonte brillante. Pas de dépendances lourdes sans nécessité.
- Le vocabulaire de Ra est phénoménologique (durée, syntonie, lévogyre,
  retour, souffle) mais il désigne toujours quelque chose de précis et de
  testable. Traduire chaque métaphore en quantité mesurable avant de coder ;
  si la traduction est ambiguë, poser la question plutôt que d'interpréter.
- Garder les noms symboliques du projet (dextro/levo, alpha/omega, spirale) —
  ils sont l'API conceptuelle — mais chaque nom doit avoir une définition
  opérationnelle dans une docstring.
- Docstrings et commentaires : français ou anglais, au choix du fichier
  existant ; cohérence locale avant tout.
- Jamais de sortie aléatoire non seedée dans les tests.

## Définition de "fait"

Un changement est terminé quand : les tests passent (y compris les nouveaux),
`MODEL_CARD.md` est régénéré et committé, l'API publique est intacte (ou le
changement est explicitement annoncé), la provenance de toute donnée ajoutée
est documentée, et le changement renforce — ou au minimum préserve — la
lisibilité du cœur. En cas de doute entre performance et transparence :
transparence. C'est écrit dans `REFUS.md`, et c'est la raison d'être du
projet.
