# Chantier 7 — Phonétique → sémantique (traits réels en canaux d'entrée)

Producteur : `spiraton/data/featurizers.py` → `PhonemeFeaturizer`.
Consommateur : `spiraton/experimental/operator_embedding.py` → `OperatorEmbedding(source="phoneme")`.
Démonstration reproductible : `examples/phoneme_channels.py`.

## Ce que demande le plan

CLAUDE.md, chantier 7 :

> Les dims 8-22 du 33D (signatures d'impédance, de flux, séquences de spins)
> sont déjà les traits phonétiques cherchés : les exploiter comme canaux
> d'entrée des cellules plutôt que d'en recalculer. Ne lancer qu'après 3 et 4.

Lien théorique (CLAUDE_TOKENIZER.md / THEORIE_LOGOS.md) : le tokenizer C ne
traite pas le texte comme des symboles arbitraires mais comme de la **physique
articulatoire**. Chaque phonème porte un opérateur (voyelles = ADD, occlusives =
SUB, nasales/[ʁ] = MUL, fricatives = DIV) et une orientation. Ces grandeurs sont
**mesurées**, pas apprises ; le chantier 7 consiste à les faire *circuler* dans
les cellules au lieu de réinventer un encodage.

## Disposition consommée (contrat 33D)

`vector33d.PHONEME_SIG` = dims **8-22** (15 canaux), remplis par
`compute_33d_vector` côté C :

| dims | contenu | sens |
|---|---|---|
| 8-12 | signature d'impédance (5 premiers phonèmes) | résistance articulatoire |
| 13-17 | signature de flux (5 premiers phonèmes) | énergie de passage de l'air |
| 18-22 | séquence de spins syllabiques (∈ {−1,0,+1,½}) | ouverture/fermeture/stable/mixte |

> **Limite mesurée (honnêteté).** Côté C, `compute_33d_vector` ne remplit ces
> bandes qu'avec les **5 premiers phonèmes/spins** du mot (boucles `i<5`,
> `tokenizer.c`). Les dims 8-22 ne sont donc pas « tous les traits » d'un mot
> long mais ceux de sa tête articulatoire (≤ 5 phonèmes). Suffisant pour la
> plupart des mots français courts ; à élargir côté socle C si un mot long
> doit être pleinement décrit.

`PhonemeFeaturizer` tokenise un segment ABA en `N` tokens (chacun un 33D réel)
puis agrège (`pool="mean"` par défaut). Il **ne renormalise pas** le 33D
globalement : les dims 0-5 (opérateur/chiralité) gardent l'échelle que
`OperatorEmbedding` lit pour **router** la représentation (l'opérateur
conditionne, il ne s'ajoute pas — chantier 4). Méthodes :
`__call__` (33D agrégé), `sequence` (`(N,33)`, canaux token-par-token),
`signature` (`(15,)`, la bande phonémique seule).

## Déblocage : le socle C

La pleine réalisation était *gated* sur le tokenizer natif (pas de compilateur C
auparavant). Avec gcc MSYS2 (UCRT64) disponible, l'amorçage du dépôt Tokenizer a
été effectué et la lib compilée :

- **Fix mémoire** `map_graphie_to_ipa` : `ptr[2]` n'est plus lu sans garde
  `ptr[1] != '\0'` (lecture hors borne sur un digraphe en fin de buffer).
- **`DICO_EVE_SIZE`** : constante manuelle (`1343`) remplacée par une macro
  dérivée de `sizeof(DICO_EVE)` — ne peut plus désynchroniser du tableau réel.
- **`make test_asan`** : cible ajoutée (`-fsanitize=address,undefined`), avec
  repli propre quand la toolchain ne fournit pas libasan (cas mingw/MSYS2 ;
  valide sur Linux/CI).
- `make rebuild && make test_all` : **OK**. `make lib` produit
  `bin/libspiratontokenizer.so`. Self-test Python : déterminisme OK (rendu
  robuste à la console cp1252).

## Résultat empirique (physique réelle)

`examples/phoneme_channels.py`, tokenizer natif :

```
   Joie  impédance=[+0.60 +0.15 +0.05 …]  flux=[+0.45 +0.80 +0.90 …]  spins=[+1 …]
   Mort  impédance=[+0.50 +0.12 +0.40 …]  flux=[+0.50 +0.75 +0.60 …]  spins=[−1 …]
spirale  impédance=[+0.65 +0.95 +0.15 +0.40 +0.05]  spins=[+1 −1 …]
  ‖sig(Joie) − sig(Mort)‖ = 2.056
```

Lectures :

- **Distinction mesurée** : Joie et Mort sont séparés dans la bande 8-22
  (impédance et spin diffèrent) — la physique articulatoire, pas un hachage
  arbitraire, porte le contraste. C'est le sens même du chantier.
- **« spirale » fait la spirale** : sa séquence de spins est `[+1, −1]` —
  ouverture *puis* fermeture, exactement le geste A→B→A′ au niveau syllabique.
- **Réel vs substitut** : `HashingFeaturizer` (is_fallback=True) met dans la
  bande 8-22 des **valeurs de hachage arbitraires** (bigrammes hachés, sans
  rapport articulatoire — p. ex. 3/15 dims non nulles pour « Joie », 0/15 pour
  « Mort ») ; `PhonemeFeaturizer` (is_fallback=False) y met des grandeurs
  **mesurées** (impédance, flux, spins). Même rang (33), sens opposé : arbitraire
  vs physique.

## État honnête et frontières

- **Le natif est requis** pour `PhonemeFeaturizer` : sans `.so`, la construction
  lève `TokenizerUnavailable`. C'est volontaire — la frontière réel/substitut
  reste explicite, l'appelant choisit son repli en connaissance de cause.
- **CI** : la lib C n'y est pas compilée (dépôt séparé). Les tests de *physique*
  skippent proprement ; la **logique d'agrégation** est testée partout via un
  tokenizer factice injecté (`tests/test_phoneme_featurizer.py`). Le pont ABI
  est figé par fixtures (`tests/fixtures/parity_33d.json`,
  `scripts/record_33d_fixtures.py`).
- **Agrégation** : `mean` par défaut mélange les tokens d'un segment. Pour des
  canaux séquentiels fidèles au temps (nourrir une cellule récurrente token
  après token), `sequence()` expose les `(N, 33)` bruts — branchement naturel
  avec `OperatorEmbedding` puis une cellule, à explorer.

## Chaîne complète vers la cellule (fait)

La pièce qui matérialise « canaux d'entrée **des cellules** » est livrée :
`sequence()` → `OperatorEmbedding(source="phoneme")` (par token) →
`SpiratonCell` (une activation par token). L'**ordre** des phonèmes/tokens entre
ainsi dans la dynamique (et non un simple profil moyen). Démontré dans
`examples/phoneme_channels.py` (section « Chaîne complète ») et figé par
`tests/test_phoneme_featurizer.py::test_sequence_feeds_a_real_cell_in_order`
(formes, finitude, flux de gradient à travers embedding ET cellule).

## Ce qui reste (mesures, jamais à forcer)

1. **Entraînement sur physique réelle** : rejouer le chantier 5 (perte
   alpha-oméga) avec `PhonemeFeaturizer` au lieu du substitut, et comparer
   l'apprenabilité du signal de clôture. Le rang (33) est déjà aligné.
2. **Élargir la tête articulatoire** (socle C) : lever la troncature à 5
   phonèmes/spins de `compute_33d_vector` pour les mots longs, si une mesure
   montre qu'elle plafonne la distinction.
3. **Émetteur ABA du tokenizer** (boucle socle↔corpus, dépôt C) : mesurer le
   taux d'accord opérateur-émis / opérateur-balise sur `dataset_aba.txt` — une
   mesure à rapporter, jamais une cible à atteindre par des boosts codés en dur.
