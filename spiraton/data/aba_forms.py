"""Grammaire ABA ÉTENDUE — les formes F0/F0b/F1/F1b/F2/F3 (+ F4, pôles F5±).

Référence Python **lisible et sans dépendance** (ni torch ni numpy) de la
taxonomie des formes de cycle A → B → A′, promue au canon des données au
Tour 35 (décision explicite de Ra ; document de grammaire :
``GRAMMAIRE_ABA.md`` à la racine du dépôt). Historique : la taxonomie est née
comme instrument de mesure au Tour 31 (``diagnostics/profile_exact.py``,
``classify_form`` GELÉ — jamais modifié) ; ce module la rend OFFICIELLE et
**TOTALE** : le 6ᵉ triplet non uniforme (+,−,−), qui faisait lever l'ancien
classificateur, reçoit ici son nom (**F0b**).

Définition opérationnelle d'une *forme* (aucun symbole sans mesure) :

* orientation d'un segment : ``(DX, OUT) = +1`` (émission, centrifuge),
  ``(LV, IN) = −1`` (réception, centripète) — ``segment_sign`` de
  ``profile_exact`` ; toute autre combinaison est HORS grammaire (``aba.py``
  couple chiralité et direction).
* une forme = le **triplet de signes per-segment** ``(o_A, o_B, o_A′)``
  + le **drapeau SEG_B vide** (``|B| = 0``). Rien d'autre.

Les 6 triplets non uniformes sont bijectivement nommés (table
``FORM_TRIPLETS``) ; les 2 triplets uniformes sont les **pôles dégénérés**
F5+ (dissipation) / F5− (verrouillage), nommés mais NON mesurables (une seule
orientation ⇒ Δ indéfini, exclus par le filtre des runners) ; F4 est le
**modificateur orthogonal** « déploiement absent » (SEG_B vide, arrangement
token canonique).

COHÉRENCE RÉTROGRADE (contrainte dure, testée) : sur tout profil où le
``classify_form`` gelé T31 ne lève pas, :func:`classify_form_extended` rend
la MÊME étiquette — y compris « F5 » agrégé pour les pôles (la distinction
F5+/F5− vit dans :func:`pole_of` et :func:`describe_form`, PAS dans le
classificateur). La SEULE différence de domaine : (+,−,−) avec SEG_B non
vide → « F0b » là où l'ancien levait ``ValueError``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

Triplet = Tuple[int, int, int]

# --- tables canoniques (bijections gelées, GRAMMAIRE_ABA.md §2) ---------------

#: Les 6 formes NON uniformes ⟷ leur triplet de signes (bijection exacte).
#: Mono-flip : F0 (canonique), F0b (nouvelle T35), F1, F1b ; bi-flip : F2, F3.
FORM_TRIPLETS: Dict[str, Triplet] = {
    "F0": (+1, +1, -1),    # mono-flip tardif, finit repliée — la clôture canon
    "F0b": (+1, -1, -1),   # mono-flip précoce, finit repliée — NOUVELLE (T35)
    "F1": (-1, +1, +1),    # mono-flip précoce, finit émise
    "F1b": (-1, -1, +1),   # mono-flip tardif, finit émise
    "F2": (+1, -1, +1),    # bi-flip dehors–dedans–dehors
    "F3": (-1, +1, -1),    # bi-flip dedans–dehors–dedans
}

#: Bijection inverse triplet → nom (round-trip, porte 4 T35).
TRIPLET_FORMS: Dict[Triplet, str] = {t: f for f, t in FORM_TRIPLETS.items()}

#: Les 2 pôles dégénérés (triplets uniformes) — nommés, NON mesurables.
POLE_TRIPLETS: Dict[str, Triplet] = {
    "F5+": (+1, +1, +1),   # dissipation : tout-émission, jamais de retour
    "F5-": (-1, -1, -1),   # verrouillage : tout-réception, récursion sans diversité
}

#: Les 9 noms de la grammaire étendue (ordre du document).
ALL_FORMS: Tuple[str, ...] = ("F0", "F0b", "F1", "F1b", "F2", "F3", "F4", "F5+", "F5-")


def _check_signs(seg_signs: Sequence[int]) -> Triplet:
    """Valide un triplet de signes per-segment (chaque signe ∈ {+1, −1}).

    Un signe hors {±1} n'est PAS parseable (``segment_sign`` ne produit que
    ±1) : lever ici est hors du domaine « profil valide » de la totalité.
    """
    signs = tuple(seg_signs)
    if len(signs) != 3 or any(s not in (+1, -1) for s in signs):
        raise ValueError(f"triplet de signes invalide (attendu 3 valeurs ±1) : {signs!r}")
    return signs  # type: ignore[return-value]


def _check_lens(seg_lens: Sequence[int]) -> Tuple[int, int, int]:
    """Valide les longueurs de segments (3 entiers ≥ 0 ; |B| = 0 permis)."""
    lens = tuple(seg_lens)
    if len(lens) != 3 or any((not isinstance(l, int)) or l < 0 for l in lens):
        raise ValueError(f"longueurs de segments invalides (3 entiers ≥ 0) : {lens!r}")
    return lens  # type: ignore[return-value]


def classify_form_extended(seg_signs: Sequence[int], seg_lens: Sequence[int]) -> str:
    """Classification TOTALE d'un cycle parseable en forme nommée.

    Définition opérationnelle (identique au ``classify_form`` gelé T31, plus
    le 6ᵉ triplet) — ordre des règles STRUCTUREL, jamais réordonné :

    1. ``|B| = 0`` ET ``o_A = +1`` ET ``o_A′ = −1``  ⇒ ``"F4"`` (déploiement
       absent, arrangement token canonique — o_B est un signe fantôme).
    2. triplet uniforme                              ⇒ ``"F5"`` (pôle agrégé ;
       raffiner avec :func:`pole_of` — cohérence rétrograde pure).
    3. sinon : bijection ``TRIPLET_FORMS``           ⇒ F0/F0b/F1/F1b/F2/F3.

    TOTALITÉ (porte 1 T35) : zéro ``ValueError`` sur tout (signes ±1³,
    longueurs ≥ 0) — les 8 triplets × {B vide, B non vide} sont couverts.
    Fonction PURE, déterministe, sans état.
    """
    signs = _check_signs(seg_signs)
    lens = _check_lens(seg_lens)
    if lens[1] == 0 and signs[0] == +1 and signs[2] == -1:
        return "F4"
    if len(set(signs)) == 1:
        return "F5"
    return TRIPLET_FORMS[signs]


def pole_of(seg_signs: Sequence[int]) -> str:
    """Raffine un triplet UNIFORME en son pôle nommé : « F5+ » ou « F5− » (ASCII ``F5-``).

    F5+ = (+,+,+) tout-dextro (dissipation) ; F5− = (−,−,−) tout-lévo
    (verrouillage). Lève ``ValueError`` sur un triplet non uniforme — le pôle
    n'est défini que là où ``classify_form_extended`` rend « F5 ».
    """
    signs = _check_signs(seg_signs)
    for name, trip in POLE_TRIPLETS.items():
        if signs == trip:
            return name
    raise ValueError(f"triplet non uniforme, pas un pôle F5 : {signs!r}")


# --- descripteurs (les définitions opérationnelles de GRAMMAIRE_ABA.md) -------

@dataclass(frozen=True)
class FormInfo:
    """Descripteur structuré d'une forme — sa docstring vit dans GRAMMAIRE_ABA.md.

    * ``triplet``   : signes per-segment ; ``None`` pour F4 (o_B fantôme).
    * ``family``    : « mono-flip » | « bi-flip » | « modificateur » | « pôle ».
    * ``n_flips``   : changements de signe du triplet (F4 : 1 au niveau token).
    * ``flip_position`` : « précoce » (A→B) | « tardif » (B→A′) | ``None``.
    * ``ending``    : « repliée » (o_A′ = −1) | « émise » (o_A′ = +1).
    * ``b_vide``    : le modificateur orthogonal (SEG_B sans token).
    * ``measurable``: retenue par le filtre des runners (deux orientations).
    * ``pole``      : « dissipation » | « verrouillage » | ``None``.
    * ``attestation``: statut de provenance corpus (honnêteté T35 : F0b n'a
      AUCUNE ligne de corpus — sémantique défendue par symétrie).
    * ``corpus_refs``: ancrages ``corpus_eve_clean.txt`` (lignes citées).
    """

    name: str
    triplet: Optional[Triplet]
    family: str
    n_flips: int
    flip_position: Optional[str]
    ending: str
    b_vide: bool
    measurable: bool
    pole: Optional[str]
    attestation: str
    corpus_refs: Tuple[str, ...]
    semantics: str


_ATTESTED_T31 = "attestée (corpus_horscanon_aba.txt, T31)"

FORMS: Dict[str, FormInfo] = {
    "F0": FormInfo(
        name="F0", triplet=(+1, +1, -1), family="mono-flip", n_flips=1,
        flip_position="tardif", ending="repliée", b_vide=False, measurable=True,
        pole=None, attestation="canonique (tous corpus ABA)",
        corpus_refs=("l.737-739", "l.749", "l.772", "l.775", "l.801"),
        semantics="la clôture de référence : A émet, B déploie, A′ rentre intégrer",
    ),
    "F0b": FormInfo(
        name="F0b", triplet=(+1, -1, -1), family="mono-flip", n_flips=1,
        flip_position="précoce", ending="repliée", b_vide=False, measurable=True,
        pole=None,
        attestation="sémantique défendue par symétrie (miroir de F1b), "
                    "non encore attestée en corpus — aucune ligne F0b n'existe ; "
                    "tests synthétiques (T35)",
        corpus_refs=("l.239-241", "l.289", "l.240"),
        semantics="l'émission qui se retire pour mûrir : clôture précoce, "
                  "déploiement introspectif — ferme comme F0 mais couve au lieu d'expandre",
    ),
    "F1": FormInfo(
        name="F1", triplet=(-1, +1, +1), family="mono-flip", n_flips=1,
        flip_position="précoce", ending="émise", b_vide=False, measurable=True,
        pole=None, attestation=_ATTESTED_T31,
        corpus_refs=("l.239-241", "l.833-837"),
        semantics="le retour précède l'aller : écouter d'abord, répondre, repartir",
    ),
    "F1b": FormInfo(
        name="F1b", triplet=(-1, -1, +1), family="mono-flip", n_flips=1,
        flip_position="tardif", ending="émise", b_vide=False, measurable=True,
        pole=None, attestation=_ATTESTED_T31,
        corpus_refs=("l.289", "l.309-310", "l.395-396"),
        semantics="double repli puis émission : la parole longuement mûrie",
    ),
    "F2": FormInfo(
        name="F2", triplet=(+1, -1, +1), family="bi-flip", n_flips=2,
        flip_position=None, ending="émise", b_vide=False, measurable=True,
        pole=None, attestation=_ATTESTED_T31 + " ; échappe à (N,k) — T31",
        corpus_refs=("l.771", "l.802", "l.343"),
        semantics="repart au lieu de revenir : le retournement du 3e segment inversé, "
                  "cycle ouvert (dissipation contrôlée)",
    ),
    "F3": FormInfo(
        name="F3", triplet=(-1, +1, -1), family="bi-flip", n_flips=2,
        flip_position=None, ending="repliée", b_vide=False, measurable=True,
        pole=None, attestation=_ATTESTED_T31 + " ; échappement le plus fort — T31",
        corpus_refs=("l.239-241", "l.856-857", "l.289"),
        semantics="le repli qui émet une fois puis se referme : risquer une parole, "
                  "rentrer la garder",
    ),
    "F4": FormInfo(
        name="F4", triplet=None, family="modificateur", n_flips=1,
        flip_position=None, ending="repliée", b_vide=True, measurable=True,
        pole=None, attestation=_ATTESTED_T31 + " ; 0 échappé (sonde du parseur)",
        corpus_refs=("l.743", "l.771"),
        semantics="le déploiement absent : SEG_B sans token, arrangement token "
                  "canonique (+, ⟨vide⟩, −) — o_B est un signe fantôme",
    ),
    "F5+": FormInfo(
        name="F5+", triplet=(+1, +1, +1), family="pôle", n_flips=0,
        flip_position=None, ending="émise", b_vide=False, measurable=False,
        pole="dissipation", attestation=_ATTESTED_T31 + " (exclue de la mesure)",
        corpus_refs=("l.715", "l.802"),
        semantics="tout-émission, jamais de retour : borne dégénérée de l'espace "
                  "des formes, citée non mesurée",
    ),
    "F5-": FormInfo(
        name="F5-", triplet=(-1, -1, -1), family="pôle", n_flips=0,
        flip_position=None, ending="repliée", b_vide=False, measurable=False,
        pole="verrouillage", attestation=_ATTESTED_T31 + " (exclue de la mesure)",
        corpus_refs=("l.341", "l.867"),
        semantics="tout-réception, récursion sans diversité : l'autre borne "
                  "dégénérée, citée non mesurée",
    ),
}


def describe_form(name: str) -> FormInfo:
    """Le descripteur d'une forme nommée — les définitions SONT ici, pas ailleurs.

    Noms acceptés : les 9 de ``ALL_FORMS`` (« F5 » agrégé n'est PAS un
    descripteur : le classificateur le rend pour cohérence rétrograde, le
    raffinement passe par :func:`pole_of`).
    """
    info = FORMS.get(name)
    if info is None:
        raise ValueError(
            f"forme inconnue {name!r} (attendu : {ALL_FORMS} ; "
            f"« F5 » agrégé se raffine via pole_of)")
    return info


# --- round-trip forme + longueurs ⟺ profil (porte 4 T35) ----------------------

def form_to_signs(form: str) -> Triplet:
    """Triplet de signes d'une forme NON uniforme (bijection ``FORM_TRIPLETS``).

    F4 et F5± sont DÉGÉNÉRÉES pour le round-trip (F4 perd o_B — signe fantôme
    d'un segment vide ; F5 agrège deux pôles) : ``ValueError`` documentée,
    hors porte 4 (dégénérescence inhérente, pas un échec).
    """
    trip = FORM_TRIPLETS.get(form)
    if trip is None:
        raise ValueError(
            f"round-trip exact réservé aux 6 formes non uniformes {tuple(FORM_TRIPLETS)} ; "
            f"{form!r} est dégénérée (F4 : o_B fantôme ; F5± : agrégées)")
    return trip


def form_profile(form: str, seg_lens: Sequence[int]) -> Tuple[int, ...]:
    """Profil plat ±1 d'une forme non uniforme aux longueurs données.

    Expansion ``[o_A]*|A| + [o_B]*|B| + [o_A′]*|A′|`` — la même que
    ``census_horscanon`` (cohérence segment→profil vérifiée T31). C'est le
    profil que mesurent les moteurs Δ (``delta_nom_exact_profile``).
    """
    signs = form_to_signs(form)
    lens = _check_lens(seg_lens)
    return tuple(s for s, l in zip(signs, lens) for _ in range(l))


def roundtrip_form(form: str, seg_lens: Sequence[int]) -> str:
    """Round-trip porte 4 : forme → triplet → :func:`classify_form_extended`.

    Identité EXACTE pour les 6 formes non uniformes dès que ``|B| ≥ 1`` et
    ``|A|, |A′| ≥ 1`` (avec ``|B| = 0``, F0 et F0b retombent sur F4 : c'est
    la dégénérescence B-vide, documentée dans GRAMMAIRE_ABA.md §4).
    """
    return classify_form_extended(form_to_signs(form), seg_lens)
