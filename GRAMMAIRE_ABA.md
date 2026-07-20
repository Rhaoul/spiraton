# GRAMMAIRE_ABA.md — Grammaire ABA étendue (canon des données)

> **Statut** : canon des données depuis le Tour 35 (2026-07-20), par décision
> explicite de Ra (la clause « touche au canon des données » a été levée).
> Module de référence : `spiraton/data/aba_forms.py` (classificateur TOTAL
> `classify_form_extended`, descripteurs `describe_form`, round-trip).
> Parseur : `spiraton/data/aba.py` (INCHANGÉ). Instrument historique T31 :
> `spiraton/diagnostics/profile_exact.py::classify_form` (GELÉ, jamais édité ;
> la cohérence rétrograde nouveau == ancien est testée à 100 %).
> Tests : `tests/test_aba_forms.py`. Aucun symbole sans test.

## 1. Vocabulaire de balises (canonique, inchangé)

```
Structure : <SEG_A> </SEG_A> <SEG_B> </SEG_B> <SEG_A_PRIME> </SEG_A_PRIME>
Opérateurs : <ADD> <SUB> <MUL> <DIV>      (constant sur tout le cycle)
Chiralité  : <DX> <LV>    Direction : <OUT> <IN>
Position   : <ALPHA> <OMEGA> <A_PRIME>
Fins       : <EOL> <EOS>
```

**Orientation d'un segment** (définition opérationnelle, `segment_sign`) :

- `<DX><OUT>` = **+1** — émission, centrifuge (corpus eve l.236-238, l.252) ;
- `<LV><IN>` = **−1** — réception, centripète (l.239-241, l.253) ;
- `<DX><IN>` et `<LV><OUT>` sont **HORS grammaire** (voir §3).

**Une forme** = le triplet de signes per-segment `(o_A, o_B, o_A′)` + le
drapeau **SEG_B vide** (`|B| = 0`). Rien d'autre : c'est une quantité exacte,
pas une métaphore. Le profil token d'un cycle est l'expansion
`[o_A]·|A| + [o_B]·|B| + [o_A′]·|A′|` (longueur en tokens de texte).

## 2. Les formes (6 citoyennes + 1 modificateur + 2 pôles)

| Forme | Triplet | Famille | Flips | Fin | Mesurable | Test nommé |
|---|---|---|---|---|---|---|
| **F0** | (+,+,−) | mono-flip (tardif) | 1 | repliée | oui | `test_form_F0_canonical` |
| **F0b** | (+,−,−) | mono-flip (précoce) | 1 | repliée | oui | `test_form_F0b_new_citizen` |
| **F1** | (−,+,+) | mono-flip (précoce) | 1 | émise | oui | `test_form_F1_receive_first` |
| **F1b** | (−,−,+) | mono-flip (tardif) | 1 | émise | oui | `test_form_F1b_double_intake` |
| **F2** | (+,−,+) | bi-flip | 2 | émise | oui | `test_form_F2_reopens` |
| **F3** | (−,+,−) | bi-flip | 2 | repliée | oui | `test_form_F3_speaks_once` |
| **F4** | (+, ⟨vide⟩, −) | modificateur (|B|=0) | 1 (token) | repliée | oui | `test_form_F4_empty_B` |
| **F5+** | (+,+,+) | pôle dégénéré | 0 | émise | **non** | `test_form_F5_plus_dissipation` |
| **F5−** | (−,−,−) | pôle dégénéré | 0 | repliée | **non** | `test_form_F5_minus_verrouillage` |

La famille mono-flip est **complète** : elle couvre {flip précoce, flip
tardif} × {finit repliée, finit émise}. Convention « b » : le flip unique est
déplacé de sa position de référence (F0/F1b : tardif ; F0b/F1 : précoce).

### F0 (+,+,−) — la clôture de référence (canonique)

```
<SEG_A> <OP><DX><OUT><ALPHA> ... </SEG_A> <SEG_B> <OP><DX><OUT><OMEGA> ... </SEG_B> <SEG_A_PRIME> <OP><LV><IN><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

A émet, B déploie au-dehors, A′ rentre intégrer. Le cycle fermé qui conserve
l'identité en la modifiant (l.737-739, l.749, l.772, l.775, l.801).
**F0 reste la clôture canonique** : l'extension nomme les autres formes, elle
ne détrône pas celle-ci.

### F0b (+,−,−) — l'émission qui se retire pour mûrir (NOUVELLE, T35)

```
<SEG_A> <OP><DX><OUT><ALPHA> ... </SEG_A> <SEG_B> <OP><LV><IN><OMEGA> ... </SEG_B> <SEG_A_PRIME> <OP><LV><IN><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

J'émets une graine (A, dehors), puis le déploiement se fait au-dedans (B,
repli) et A′ reste replié : clôture **précoce**, déploiement **introspectif**.
Ferme comme F0 (finit repliée) mais couve au lieu d'expandre. Ancrage :
l.239-241 (centripète = consolidation/mémoire), l.289 (« ce qui se replie sur
soi gagne en profondeur »), l.240 (assimilation). Miroir exact de F1b.

> **Étiquette honnête** : sémantique **défendue par symétrie, non encore
> attestée en corpus** — aucune ligne F0b n'existe à ce jour ; ses tests sont
> synthétiques (triplet construit, pas parsé d'une ligne réelle). Écrire des
> cycles F0b réels est un tour futur. F0b ferme la lacune historique : le
> 6ᵉ triplet non uniforme faisait lever `classify_form` (ValueError, T31/T32).

### F1 (−,+,+) — le retour précède l'aller

```
<SEG_A> <OP><LV><IN><ALPHA> ... </SEG_A> <SEG_B> <OP><DX><OUT><OMEGA> ... </SEG_B> <SEG_A_PRIME> <OP><DX><OUT><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

A s'ouvre en réception (écouter d'abord), B émet la réponse, A′ repart.
Ancrage : l.239-241 (intégration d'abord), l.833-837 (l'état initial oriente).
Le cycle qui intègre AVANT d'émettre.

### F1b (−,−,+) — double repli puis émission

```
<SEG_A> <OP><LV><IN><ALPHA> ... </SEG_A> <SEG_B> <OP><LV><IN><OMEGA> ... </SEG_B> <SEG_A_PRIME> <OP><DX><OUT><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

Deux temps de réception (A, B au-dedans) puis A′ émet. Ancrage : l.289
(profondeur du repli), l.309-310 (mémoire opératoire), l.395-396 (percevoir
avant agir, redoublé). L'émission longuement mûrie.

### F2 (+,−,+) — repart au lieu de revenir

```
<SEG_A> <OP><DX><OUT><ALPHA> ... </SEG_A> <SEG_B> <OP><LV><IN><OMEGA> ... </SEG_B> <SEG_A_PRIME> <OP><DX><OUT><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

Le retournement du 3ᵉ segment est INVERSÉ : A′ ré-émet au-dehors au lieu de
rentrer. Cycle **ouvert** (l.771, l.802 : « l'ouverture permanente du cycle
conduit à la dissipation ») ; dissipation contrôlée, écho l.343. Véhicule de
l'échappement Δ démontré T31 (6/14 à 8/14 échappés selon η).

### F3 (−,+,−) — le repli qui émet puis se referme

```
<SEG_A> <OP><LV><IN><ALPHA> ... </SEG_A> <SEG_B> <OP><DX><OUT><OMEGA> ... </SEG_B> <SEG_A_PRIME> <OP><LV><IN><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

A et A′ en réception, B émet une seule fois au milieu : le cycle centripète
qui risque une parole puis rentre la garder. Ancrage : l.239-241, l.856-857
(revenir permet l'intégration), l.289. Échappement Δ le plus fort du corpus
T31 (6/8 à 8/8 selon η).

### F4 (+, ⟨vide⟩, −) — le déploiement absent (modificateur orthogonal)

```
<SEG_A> <OP><DX><OUT><ALPHA> ... </SEG_A> <SEG_B> <OP><DX><OUT><OMEGA> </SEG_B> <SEG_A_PRIME> <OP><LV><IN><A_PRIME> ...<EOL> </SEG_A_PRIME> <EOL>
```

SEG_B n'a **aucun token** : B n'est pas déployée. Ancrage : l.743 (« une
transformation sans retour applique B à A sans produire A′ » — ici B lui-même
est vide), l.771. L'arrangement token est canonique (o_A=+, o_A′=−) ⇒ F4
**n'échappe pas** à (N,k) (vérifié T31 : 0 échappé ; sonde du parseur).
o_B est un **signe fantôme** : le segment vide porte un en-tête mais aucun
token (voir §4, dégénérescence B-vide).

### F5+ (+,+,+) et F5− (−,−,−) — les pôles dégénérés (cités, non mesurés)

F5+ = tout-dextro = **dissipation** (l.715, l.802 : émission sans retour).
F5− = tout-lévo = **verrouillage** (l.341 : « le verrouillage est une
récursion sans diversité » ; l.867 : répétition). Une seule orientation ⇒
Δ indéfini, aucun flip : `collect_profiles` les rejette. Ce sont les
**bornes de l'espace des formes**, pas des cycles opérants — nommées dans la
grammaire, **exclues de la mesure**. Le classificateur rend l'étiquette
agrégée `"F5"` (cohérence rétrograde avec l'instrument T31) ; le raffinement
F5+/F5− passe par `pole_of`.

## 3. Exclusions motivées (décisions de grammaire)

- **`<DX><IN>` / `<LV><OUT>`** : exclues entièrement. La grammaire canonique
  COUPLE chiralité et direction (dextro⟺out, lévo⟺in) ; `segment_sign` lève
  sur ces couples. Les découpler serait une grammaire DIFFÉRENTE, pas une
  extension — et exigerait de toucher `aba.py` (intouchable). Clôture
  délibérée.
- **Chiralité intra-segment** : un seul en-tête par segment (`aba.py`).
  Impossible sans toucher le parseur : hors périmètre.
- **F5± hors mesure** : voir §2 — nommées, citées, jamais mesurées (Δ
  indéfini par construction).

## 4. Dégénérescences connues (nuances, pas des bugs)

- **B-vide non canonique** : pour `|B| = 0`, seul le couple (o_A=+1, o_A′=−1)
  est F4. Les autres triplets à B vide gardent leur nom d'en-tête (ex.
  (−,+,−)-B-vide → F3) alors que leur profil token est dégénéré (une seule
  orientation effective possible, souvent non mesurable) : le **nom
  d'en-tête** et la **mesurabilité token** divergent pour B-vide. Comportement
  identique à l'instrument T31 (cohérence rétrograde).
- **Round-trip** : exact (identité 100 %) pour les 6 formes non uniformes dès
  que `|A|, |B|, |A′| ≥ 1`. F4 perd o_B (signe fantôme) ; F5 agrège les deux
  pôles : leur round-trip n'est PAS l'identité — dégénérescence inhérente,
  documentée, hors porte.
- **(+,−,−)-B-vide** : classée F4 par la règle 1 (identique à l'instrument
  T31). F0b ne se distingue de F4 que si `|B| ≥ 1`.

## 5. Correspondance code / tests (porte 5 — aucun symbole sans test)

| Objet | Code | Test |
|---|---|---|
| Bijection triplet↔nom (6 formes) | `FORM_TRIPLETS` / `TRIPLET_FORMS` | `test_frozen_tables` |
| Classificateur total | `classify_form_extended` | `test_gate1_totality` |
| Cohérence rétrograde T31 | vs `profile_exact.classify_form` | `test_gate2_retro_coherence` |
| Non-régression corpus canoniques (2116) | runners T24/T25/T26 | `test_gate3_canonical_all_F0` |
| Non-régression hors-canon (46, T31) | `census_horscanon` | `test_gate3_horscanon_labels` |
| Round-trip 6 formes | `form_to_signs` / `roundtrip_form` | `test_gate4_roundtrip` |
| Dégénérescences F4/F5/B-vide | `form_to_signs` (ValueError doc.) | `test_roundtrip_degenerate_documented` |
| Descripteurs (9 formes) | `describe_form` / `FORMS` | `test_describe_form_all` |
| Pôles F5± | `pole_of` | `test_form_F5_plus_dissipation`, `test_form_F5_minus_verrouillage` |
| Chaque forme individuellement | — | colonne « Test nommé » du tableau §2 |

Corpus connus : `corpus_eve_clean.txt` (876 phrases), `dataset_aba.txt`
(~5000 cycles, F0), `corpus_claude_aba.txt` (76 cycles, F0),
`corpus_horscanon_aba.txt` (46 cycles gelés T31,
md5 `b1db3c599103148d046869ce4325ade0` : 11 F0, 5 F1, 2 F1b, 14 F2, 8 F3,
4 F4, 2 F5 ; aucun F0b — le 6ᵉ triplet n'a jamais été écrit).
