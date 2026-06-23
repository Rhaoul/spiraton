# Chantier 2 — Double dynamique L∘D vs D∘L

Diagnostic : `spiraton/diagnostics/double_dynamics.py` → `run_double_dynamics`.
Exemple reproductible : `examples/double_dynamics.py`.

## Ce que prédit la théorie

THEORIE_LOGOS.md §2.4 (axiome 4, « Double dynamique ») :

> La composition L∘D tend à stabiliser des attracteurs ; la composition D∘L
> tend à engendrer des divergences ou bifurcations.

Traduction opératoire dans ce dépôt (CLAUDE.md, chantier 2) : sur un **même**
`SpiralGrid` (même cellule, même projection `y_to_vec`), appliquer K pas
`outward` (expansion dextrogyre D) puis K pas `inward` (contraction lévogyre L)
— c'est **L∘D** — et comparer à l'ordre inverse **D∘L** (inward puis outward).
On mesure, le long de chaque trajectoire, le signal alpha-oméga `cos − l2`
relativement à x0, puis sa **variance temporelle**. Prédiction falsifiable :
`var(L∘D) < var(D∘L)` (L∘D plus stable).

Convention de nommage : `∘` se lit de droite à gauche (l'opérateur de droite
agit en premier). **L∘D = D puis L = outward puis inward.**

## Résultat empirique (cellule canon non entraînée)

Configuration : grille 7×7, C=6, K=8, batch 4, 12 graines, `SpiratonCell`
à l'initialisation aléatoire (`examples/double_dynamics.py`).

```
Résumé : 6/12 graines conformes à la théorie (var L∘D < var D∘L).
gap moyen (dl − ld) = +0.014   (à comparer à des variances de 0.1 à 25)
→ Pas d'asymétrie nette à cette échelle.
```

**Conclusion honnête : à l'initialisation aléatoire, l'asymétrie prédite
n'apparaît pas.** Le signe du `gap` est à peu près équiprobable et son ampleur
est négligeable devant la variance du signal. Conformément à CLAUDE.md, c'est
un *résultat* que l'on consigne — on ne trafique pas le code pour produire
l'asymétrie.

## Pourquoi (lecture mécaniste)

Au niveau `SpiralGrid`, `outward` et `inward` ne changent **que l'ordre de
balayage** de la mise à jour séquentielle : la règle locale appliquée à chaque
cellule (`local + y_to_vec(cell(concat(local, neigh)))`) est rigoureusement la
même dans les deux sens. La seule source d'asymétrie est donc « quelles
cellules voient les voisins déjà mis à jour en premier ». Avec des poids
aléatoires non entraînés, cet effet d'ordre est réel mais minuscule — ce que la
mesure confirme.

Autrement dit : la double dynamique de la théorie demande que **D et L diffèrent
par leur nature**, pas seulement par l'ordre de visite. Trois pistes pour que
l'asymétrie ait une chance d'émerger, à mesurer (jamais à forcer) :

1. **Cellule entraînée** : rebrancher ce diagnostic après le chantier 5
   (entraînement ABA) sur une cellule dont les branches dextro/lévo ont acquis
   des effets distincts.
2. **`MatrixSpiratonCell`** (chantier 1) : ses opérateurs non commutatifs
   rendent l'ordre de composition intrinsèquement signifiant. À brancher quand
   un pont grille↔cellule-vectorielle existera (la grille attend aujourd'hui
   une cellule scalaire).
3. **Mise à jour directionnelle** : faire dépendre la règle de mise à jour du
   sens (D amplifie, L contracte) plutôt que de la seule orientation de
   balayage. Changement de *physique* — à discuter en issue avant tout code,
   car il touche un invariant du canon.

Le diagnostic, lui, est prêt et testé : il suffit de lui passer un système plus
expressif pour reconduire la mesure.
