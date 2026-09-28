# Consigne d'écriture — corpus ABA de formes mixtes (R4)

Vous écrivez un corpus de **144 phrases françaises**. Chacune est découpée en trois
segments étiquetés. Ce corpus servira à une mesure dont l'objet ne vous est pas
communiqué, volontairement. **Ne lisez rien d'autre dans le dépôt** (ni les corpus
`*_aba.txt` existants, ni les documents `docs/SEMANTIC_THERMO_*` autres que celui-ci,
ni le journal). Écrivez à partir de cette seule consigne.

## 1. Ce qu'est un cycle

Une phrase en trois segments consécutifs : A, puis B, puis A′. Chaque segment porte une
**orientation** :

- **DX/OUT** (dextrogyre, vers le dehors) : le segment **émet** — expression, élan,
  ouverture, mouvement vers l'extérieur, ce qui part, s'étend ou se donne.
- **LV/IN** (lévogyre, vers le dedans) : le segment **reçoit** — réception, retour,
  repli, intégration, ce qui revient, se rassemble ou se garde.

Chaque cycle porte aussi un **opérateur**, qui qualifie le geste de toute la phrase :

- **ADD** : agréger, ajouter, rassembler, réunir ;
- **SUB** : ôter, distinguer, séparer, retrancher ;
- **MUL** : amplifier, multiplier, faire croître, engendrer ;
- **DIV** : partager, répartir, ramifier, diviser.

Écrivez chaque segment pour que son orientation soit **vraie de ce qu'il dit**, et
chaque phrase pour que son opérateur soit vrai du geste d'ensemble. La phrase doit se
lire d'un seul souffle, naturellement, de A à A′.

## 2. Les six formes à écrire

Le triplet donne l'orientation de (A, B, A′) : + = DX/OUT, − = LV/IN.

| forme | triplet | nombre de cycles |
|---|---|---|
| F0 | (+, +, −) | 24 |
| F0b | (+, −, −) | 24 |
| F1 | (−, +, +) | 24 |
| F1b | (−, −, +) | 24 |
| F2 | (+, −, +) | 24 |
| F3 | (−, +, −) | 24 |

Dans chaque forme : **6 cycles par opérateur** (ADD, SUB, MUL, DIV).

## 3. Contraintes

- Chaque segment compte **3 à 7 mots** (séparés par des espaces ; « l'arbre » compte
  pour un mot).
- 144 phrases **toutes différentes**. Variez les sujets (nature, corps, travail,
  pensée, musique, cuisine, ville, mer, relations, saisons…), les tournures et le
  vocabulaire. Pas de gabarit répété, pas de refrain.
- Pas de balises dans le texte, pas de ponctuation interne entre segments (un point
  final est permis dans A′).
- Personne grammaticale libre.

## 4. Format exact d'une ligne

Une ligne par cycle. `OP` est l'opérateur du cycle, identique sur les trois segments.
Chaque segment porte `<DX><OUT>` ou `<LV><IN>` selon la forme :

```
<SEG_A> <OP><CHI><DIR><ALPHA> texte A </SEG_A> <SEG_B> <OP><CHI><DIR><OMEGA> texte B </SEG_B> <SEG_A_PRIME> <OP><CHI><DIR><A_PRIME> texte A′<EOL> </SEG_A_PRIME> <EOL>
```

Exemple de forme F2 (+, −, +), opérateur ADD (exemple de format seulement, à ne pas
imiter pour le fond) :

```
<SEG_A> <ADD><DX><OUT><ALPHA> Nous invitons tout le quartier </SEG_A> <SEG_B> <ADD><LV><IN><OMEGA> chacun rapporte un plat chez nous </SEG_B> <SEG_A_PRIME> <ADD><DX><OUT><A_PRIME> puis la fête déborde dans la rue<EOL> </SEG_A_PRIME> <EOL>
```

## 5. En-tête du fichier

Commencez le fichier `corpus_r4_replication_aba.txt` par des lignes de commentaire `#`
indiquant l'auteur, la date, et la mention : « écrit à partir de la seule consigne
SEMANTIC_THERMO_R4_CONSIGNE_AUTEUR.md, sans lecture des corpus ni des résultats ;
GPL-3.0-or-later ». Les formes peuvent être regroupées par sections commentées.

Quand le fichier est écrit, lancez le validateur :

```
python scripts/semantic_thermo_replicate.py validate ../corpus_r4_replication_aba.txt
```

Corrigez jusqu'à ce qu'il réponde `VALIDE`. Ne lancez aucune autre commande de mesure.
