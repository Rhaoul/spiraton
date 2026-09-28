from __future__ import annotations

"""Diagramme de phase (N, k, η) — découpage exact de l'axe η en phases et frontières (Tour 29).

ÉMISSION T29 (linguiste, gelée AVANT toute mesure — ``TOUR29_EMISSION.md``) : passer
des 5 échantillons η du T28 au CONTINUUM (0, 4] et livrer, PAR CELLULE, la SUITE
ordonnée des frontières de signe(Δ_nom(N, k; η)) — jamais « LE η de bascule »
(avertissement gelé : Δ(η) est en escalier, le signe peut osciller ; l'unicité de la
bascule n'est PAS présupposée). Phase = intervalle maximal de signe constant ;
frontière = point de changement.

STRUCTURE GELÉE A PRIORI (émission §2, l'algèbre avant la mesure) :

  * 2.1 — f_edge fixe η-invariant ; f_edge organe(η) en ESCALIER (change aux
    traversées de bord |e_t − 1| = 1/2 et aux (dés)activations de clip) ; Δ(η) à
    valeurs dans {j/N}, signe constant par morceaux.
  * 2.2 (LE point falsifiable) — e_0 = 1 et e_1 sont η-INDÉPENDANTS ; la dépendance
    η n'entre qu'à e_2, AFFINE ; sur branche non clipée, e_t est POLYNOMIALE de
    degré ≤ t−1 en η. Frontière portée par e_2 = RATIONNELLE exacte (certificat) ;
    portée par e_{t≥3} = algébrique, génériquement irrationnelle → ENCADREMENT.
    Si la mesure certifie des frontières RATIONNELLES portées par e_{t≥3},
    l'émission est RÉFUTÉE sur 2.2 — à rapporter tel quel.
  * 2.3 — méthode exacte : (a) balayage ``Fraction`` sur maille gelée ; (b)
    bissection rationnelle exacte entre nœuds adjacents de signes différents,
    largeur ≤ 2⁻²⁰ → encadrement ; (c) certificat de rationalité : traversée
    portée par e_2 sur branche affine non clipée ⟹ η rationnel exact reporté
    comme valeur ; sinon encadrement, JAMAIS affirmé rationnel.

PROTOCOLE GELÉ (émission §3-§8, zéro degré de liberté) :

  * Domaine η ∈ (0, 4] ; bord η → 0 singulier EXCLU (déclaré) ; η > 4 hors
    périmètre (|λ| > 1). MAILLE PRIMAIRE 1/12 (48 nœuds — contient les Niven
    {1, 2, 3, 4}, l'ancre 1/2 et les quarts) ; maille de CONVERGENCE 1/24
    (1/48 au besoin). Compte de phases STABLE exigé (P1) ; cellule non stable =
    encadrements + cellule SIGNALÉE, jamais un compte forcé.
  * Grille : les 228 cellules T27/T28 (N ∈ [6, 24], k ∈ [2, N−2]).
  * Baseline : Δ_nom(η) = f_edge(organe η, g0=1) − f_edge(fixe g=1), fixe
    η-INVARIANTE mise en cache (UN calcul par cellule — ``delta_nom_exact_eta``
    T28 réutilisé tel quel). Oracle par cellule INTERDIT. AUCUN volet réel.
  * Seuils T24 hérités : band = 1/2, φ* = 2/3, g0 = 1, target = 1, bornes
    [1/2, 2], δ_min = 1e-4. ``structural_gap.py`` GELÉ byte-à-byte ;
    ``edge_controller.py`` INTACT (η par paramètre d'appel, jamais ETA_STRUCT) ;
    ``horizon_law.py`` et ``spectral_map.py`` importés en LECTURE SEULE.
  * Vérification instrument : porte 0a re-confirmée UNE fois (patron T28) ;
    points de contrôle float a priori η ∈ {3/4, 2, 5/2, 3, 7/2} (dyadiques,
    exacts au float) — concordance de SIGNE 228/228 par point ; signes float aux
    bornes de chaque encadrement == signes exacts ; divergences de MAGNITUDE au
    bord |e−1| = 1/2 rapportées NON corrigées (réserve T28) ; divergence de
    SIGNE = BUG, arrêt technique.

PRÉDICATS FALSIFIABLES GELÉS (émission §5) :

  P1 — compte de phases fini, stable 1/12 ↔ 1/24 (1/48 au besoin) ; conjecture
       annexe : majorité des cellules ≤ 2 phases.
  P2 — les 19 cellules basses (P-gate) : signe(Δ) = 0 à TOUS les nœuds < 4 ;
       ≠ 0 possible seulement à η = 4.
  P3 — interdiction de réintroduire une loi monotone (N, k, N−k) du compte de
       phases ; prédiction = structure arithmétique quasi-périodique ; une
       monotonie propre serait une SURPRISE à rapporter.
  P4 — (7,4) : UNE frontière certifiée rationnelle η = 1 (− sur (0,1), 0 en 1,
       + sur (1,4]) ; (8,6) : frontière de sortie du zéro ENCADRÉE dans (1/2, 1)
       ouverte.
  P5 — DEUX familles de rationalité distinctes : (i) frontières d'ordre-e₂
       rationnelles (raison algébrique) ; (ii) résonances de périodicité exacte
       concentrées aux Niven {1, 2, 3, 4} (le compte de fenêtres exactement
       périodiques doit PIQUER à ces 4 η). Ne pas conflater.

MÉCANIQUE DE LOCALISATION (implémentation directe de 2.3, déclarée) :

  * Bissection : entre deux nœuds adjacents de signes différents, dichotomie
    rationnelle exacte ; si le point médian porte un TROISIÈME signe, l'intervalle
    est scindé (structure plus riche que la maille — découverte, pas anomalie) ;
    arrêt à largeur ≤ 2⁻²⁰. Deux transitions séparées par un écart ≤ 2⁻²⁰ sont
    FUSIONNÉES en une frontière unique à signes intermédiaires (sous-résolution,
    rapporté tel quel).
  * Certificat : entre les bornes (lo, hi) d'un encadrement, si le motif de clip
    est IDENTIQUE (branche stable), les e_t(η) sont dérivés SYMBOLIQUEMENT en
    polynômes à coefficients ``Fraction`` sur cette branche ; chaque hit qui
    change entre lo et hi désigne son polynôme porteur ; degré 1 ⟹ racine
    rationnelle exacte ; degré ≥ 2 ⟹ seul le rationnel de PLUS PETIT dénominateur
    de [lo, hi] (Stern–Brocot) est testé comme racine exacte. Certification ⟺
    toutes les traversées coïncident en UN rationnel r, la trace exacte à r pose
    chaque e_t porteur EXACTEMENT sur le bord, le motif de clip à r est celui de
    la branche, et les signes aux mi-points (lo,r)/(r,hi) confirment. Sinon :
    encadrement, jamais affirmé rationnel. ``is_e2`` ⟺ porté par {e_2} seul,
    branche non clipée au pas 1, degré 1 — la famille (i) du protocole 2.3(c).
  * Segment terminal : une frontière certifiée EXACTEMENT à η = 4 fait du bord
    droit un POINT (pas une phase) ; un encadrement collé à 4 non certifié est un
    micro-segment ≤ 2⁻²⁰ signalé (jamais compté comme phase).

STRICTEMENT DIAGNOSTIC. Ne touche NI le canon ``core/``, NI ``regulate_step``,
NI ``structural_gap.py`` (gelé), NI ``horizon_law.py``/``spectral_map.py`` (T27/T28
gravés — importés en lecture seule). Anti-circularité : profils construits depuis
(N, k) directement ; jamais les dims 0-5 du 33D. Tout est déterministe (aucune
source aléatoire hors shuffles seedés de la porte 0a, infra T23).

VERDICT MESURÉ (2026-07-12, gravé APRÈS exécution du protocole — jamais forcé ;
l'ingénieur statue) :

  * Porte 0a re-confirmée : PASSE ((13,8) gap primaire 2.185897e-1, (7,5)
    8.095238e-2, ≥ δ_min = 1e-4 ; multiset vacuous 0.0 sur les deux).
  * DIAGRAMME : histogramme des comptes de phases {1: 64, 2: 35, 3: 65, 4: 37,
    5: 18, 6: 3, 7: 2, 8: 1, 9: 3} — la conjecture annexe P1 (« majorité ≤ 2
    phases ») est RÉFUTÉE (99/228). 406 frontières : 78 certifiées rationnelles
    (dont 4 e₂), 328 encadrements ≤ 2⁻²⁰ ; 0 incohérence de signes ; aucune
    fusion sous-résolution ; aucun point/micro-segment terminal à η = 4.
  * P1 : 82 cellules ont requis la maille 1/48 ; 35 restent NON stables
    1/24 ↔ 1/48 (signalées, jamais forcées). La richesse non convergée se
    concentre à PETIT η (< 1/4), où Δ oscille près de zéro.
  * 2.2 : les 4 certificats e₂ (famille (i)) — (7,3) → 11/10, (8,4) → 1/2,
    (10,4) → 5/4, (17,9) → 1/14. MAIS 74 frontières certifiées RATIONNELLES
    portées par e_{t≥3} : l'émission est RÉFUTÉE sur 2.2 tel qu'énoncé.
    Mécanismes mesurés : (a) branches CLIPÉES en amont ⟹ le degré du porteur
    RETOMBE (souvent à 1) ⟹ racine rationnelle exacte — le cas générique des
    74 ; (b) UNE racine rationnelle non générique d'un porteur de degré 2 :
    (15,6) → η = 1/2 ; (c) résonance η = 3 (θ = 2π/3) : 8 cellules certifiées
    à η = 3 EXACTEMENT, dont les 7 cellules N = 3k de la grille, à traversées
    SIMULTANÉES multi-hits ((6,2) : 3 hits ; (24,8) : 15 hits).
  * P2 : VIOLÉE SOUS 4 — (8,5) ≠ 0 dès le nœud 191/48 (frontière ≈ 3.96472),
    (11,7) dès 95/24 (≈ 3.95367), et (10,7) a une phase − TRANSITOIRE
    (≈ (3.94927, 3.95992)) qui REVIENT à 0 avant η = 4. À η = 4 exactement :
    (8,5) et (11,7) seules (== T28). La sortie de bande basse commence vers
    η ≈ 3.95, pas à 4 : le « η_crit = 4 » du T28 était un artefact de maille.
  * P3 : AUCUNE monotonie (0/19 lignes monotones en k ; non monotone en N ni en
    N−k) — structure arithmétique quasi-périodique, dont des frontières
    PARTAGÉES entre cellules (ex. ≈ 0.4188611 pour (6,3), (8,4), (10,5),
    (12,5), (18,7)).
  * P4 : (7,4) RÉFUTÉ — pas « UNE frontière certifiée η = 1 » : HUIT phases
    (0, −, 0, −, 0, +, 0, +), 7 frontières TOUTES en encadrement, et η = 1 est
    INTÉRIEUR à un plateau zéro ≈ (0.89696, 1.14196) — la « bascule à η = 1 »
    du T28 était l'échantillonnage ponctuel d'un plateau. (8,6) CONFIRMÉ —
    sortie du zéro encadrée (263245/393216, 1052981/1572864] ⊂ (1/2, 1)
    ouverte (porteur e_7, degré 6) ; et son zéro ne part pas de 0⁺ : phase +
    sur (0, ≈ 0.22807).
  * P5 : les comptes de fenêtres exactement p-périodiques PIQUENT aux 4 Niven —
    η=1 (p=6) : 884 vs 36/36 (voisins ∓1/12) ; η=2 (p=4) : 1178 vs 49/49 ;
    η=3 (p=3) : 1292 vs 56/56 ; η=4 (p=2) : 185 vs 63 — pics 4/4. Familles
    (i) et (ii) bien DISTINCTES : les e₂ certifiés sont à {11/10, 1/2, 5/4,
    1/14}, aucun aux Niven.
  * Résonances irrationnelles : l'encadrement (9,4) (460681/786432,
    307121/524288] ≈ (0.58578618, 0.58578682] CONTIENT 2−√2 (θ = π/4, période
    8 ; vérifié exactement : (2−hi)² < 2 < (2−lo)²) — capturée par bissection
    comme prévu §3. Les 7 encadrements à motif de clip INSTABLE
    (non certifiables par CE protocole) serrent des rationnels simples
    ((10,2) ∋ 3/7, (9,4) ∋ 1⁻, (11,4) ∋ 6/5, (23,8) ∋ 12/11) : candidate
    TROISIÈME famille — frontières d'ACTIVATION DE CLIP (affines en η donc
    rationnelles) — hors périmètre du certificat gelé, rapportée pour une
    émission future, jamais affirmée.
  * VÉRIFICATION INSTRUMENT : 5 points de contrôle float → SIGNES 5 × 228/228,
    PASSE ; divergences de MAGNITUDE {3/4: 2, 2: 6, 5/2: 2, 3: 8, 7/2: 3},
    toutes avec ≥ 1 point exactement au bord |e−1| = 1/2 (réserve T28, non
    corrigées). BORNES d'encadrements : 812 vérifiées, 5 divergences de
    SIGNE — la condition d'ARRÊT TECHNIQUE gelée est FORMELLEMENT déclenchée.
    Diagnostic gravé : les 5 sont sur (9,7) SEULE, dont le lecteur FIXE
    (η-invariant) a e_8 = 3/2 EXACTEMENT au bord ⟹ f_edge fixe float 4/9 vs
    exact 5/9 ⟹ Δ_float(9,7; η) = Δ_exact + 1/9 à TOUT η — c'est la réserve
    de magnitude T27/T28 DÉJÀ gravée ((9,7) diverge en magnitude à chaque η
    depuis T27) ; aux bornes où Δ_exact ∈ {0, −1/9}, ce décalage CONSTANT
    devient une divergence de signe. Aucun mécanisme neuf ; le signe
    décisionnel reste partout celui de la Fraction. Rapporté NON corrigé —
    l'ingénieur statue sur la portée de l'arrêt.

NOTES D'INTÉGRATION (voix L, T29 — gravées après reproduction indépendante,
moteur Fraction réimplémenté depuis la spec, aucun import de ce module) :

  * ARBITRAGE DE L'ARRÊT TECHNIQUE : l'arrêt porte sur la CONTRE-ÉPREUVE FLOAT
    en ces 5 bornes, pas sur le tour. La clause gelée vise un BUG d'un des deux
    moteurs ; ici les deux moteurs exacts concordent partout (dérivation ↔
    reproduction indépendante : mêmes 5 bornes, mêmes Δ, même offset +1/9), et
    la divergence est la conséquence arithmétique d'une réserve d'instrument
    GRAVÉE depuis T27 ((9,7) : e_8 fixe == 3/2 au bord exact ⟹ float 4/9 vs
    exact 5/9). La contre-épreuve float est déclarée NON PROBANTE sur (9,7) ;
    le verdict au fond est maintenu. Mesure de cadrage : 36 cellules de la
    grille ont un lecteur FIXE avec ≥ 1 hit au bord exact ; l'instrument float
    n'en lit mal que 2 — (9,7) (offset +1/9, 5 flips de signe aux bornes) et
    (15,11) (offset +1/15, 1 seule phase ⟹ aucune borne à vérifier, aucun
    flip possible ce tour). Toute contre-épreuve float future doit soit
    pré-déclarer l'exclusion motivée de ces cellules, soit passer par un
    instrument traitant le bord exact — jamais par un lissage de
    ``structural_gap`` (gelé).
  * BORD η → 0 SOUS-RÉSOLU (protocole-relatif, même famille d'artefact que le
    « η_crit = 4 » T28 diagnostiqué ce tour) : les comptes de phases sont
    relatifs au protocole gelé — une structure entièrement contenue SOUS la
    maille primaire échappe au critère P1 quand 1/12 et 1/24 concordent.
    Mesuré au scan fin 1/2016 (reproduction indépendante) : (8,6) a une phase
    ZÉRO sur (0, ≈ 0.02569) avant sa phase + (le « phase + sur (0, ≈ 0.22807) »
    ci-dessus est donc protocole-relatif, pas absolu) ; (6,2) porte 0, −, 0, −
    sous 1/12 (5 phases au lieu des 2 du protocole). Les frontières localisées
    et P4-(8,6) (sortie du zéro dans (1/2, 1)) ne sont pas affectés. À trancher
    par une émission future (maille dédiée près de 0), jamais en douce.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .horizon_law import (
    BAND_EXACT,
    GRID_K_MIN,
    GRID_N_MAX,
    GRID_N_MIN,
    TARGET_EXACT,
    flip_side,
    gate0a,
    grid_cells,
    is_high_excursion,
)
from .spectral_map import (
    band_edge_touches_eta,
    delta_nom_exact_eta,
    delta_nom_float_eta,
    trace_full_exact,
)


# --- constantes GELÉES A PRIORI (émission T29 — jamais ajustées après mesure) ------

MESH_PRIMARY = 12                 # maille primaire : 48 nœuds sur (0, 4]
MESH_CONVERGENCE = 24             # maille de convergence
MESH_FINE = 48                    # maille fine « au besoin » (P1)
DOMAIN_MAX = Fraction(4)          # domaine (0, 4] ; η → 0 exclu, η > 4 hors périmètre
RESOLUTION = Fraction(1, 2 ** 20)  # largeur maximale d'un encadrement de frontière

# points de contrôle float A PRIORI (dyadiques : float(Fraction) sans perte)
CONTROL_ETAS: Dict[str, Fraction] = {
    "3/4": Fraction(3, 4),
    "2": Fraction(2),
    "5/2": Fraction(5, 2),
    "3": Fraction(3),
    "7/2": Fraction(7, 2),
}

# résonances Niven : les seuls η ∈ (0, 4] rationnels avec θ(η)/π ∈ ℚ (périodes gelées)
NIVEN_PERIODS: Dict[Fraction, int] = {
    Fraction(1): 6, Fraction(2): 4, Fraction(3): 3, Fraction(4): 2,
}
PERIODS_SCANNED: Tuple[int, ...] = (2, 3, 4, 6)


# --- moteur exact mémoïsé (trace par (cellule, η) unique ; baseline cachée T28) ----

@lru_cache(maxsize=None)
def _trace_full(n: int, k: int, eta: Fraction
                ) -> Tuple[Tuple[Fraction, ...], Tuple[Fraction, ...], Tuple[bool, ...]]:
    """Trace exacte (e, g, clipped) mémoïsée — UN calcul par (cellule, η)."""
    e, gs, clips = trace_full_exact(n, k, eta)
    return tuple(e), tuple(gs), tuple(clips)


@lru_cache(maxsize=None)
def sign_at(n: int, k: int, eta: Fraction) -> int:
    """signe(Δ_nom(N, k; η)) EXACT ∈ {−1, 0, +1} (baseline fixe cachée, T28)."""
    d = delta_nom_exact_eta(n, k, eta)
    return (d > 0) - (d < 0)


@lru_cache(maxsize=None)
def _hits(n: int, k: int, eta: Fraction) -> FrozenSet[int]:
    """Indices t ∈ [1, N] avec |e_t − 1| ≤ 1/2 (les hits de f_edge, organe η)."""
    e, _gs, _clips = _trace_full(n, k, eta)
    return frozenset(t for t in range(1, len(e))
                     if abs(e[t] - TARGET_EXACT) <= BAND_EXACT)


@lru_cache(maxsize=None)
def _clip_pattern(n: int, k: int, eta: Fraction) -> Tuple[Optional[Fraction], ...]:
    """Motif de clip du pas t : None (libre) ou la borne atteinte (1/2 ou 2)."""
    _e, gs, clips = _trace_full(n, k, eta)
    return tuple((gs[t] if clips[t] else None) for t in range(len(gs)))


# --- polynômes exacts en η (coefficients Fraction, ordre croissant) -----------------

Poly = Tuple[Fraction, ...]


def _pnorm(p: Sequence[Fraction]) -> Poly:
    q = list(p)
    while len(q) > 1 and q[-1] == 0:
        q.pop()
    return tuple(q)


def _padd(p: Poly, q: Poly) -> Poly:
    m = max(len(p), len(q))
    return _pnorm(tuple(
        (p[i] if i < len(p) else Fraction(0)) + (q[i] if i < len(q) else Fraction(0))
        for i in range(m)))


def _psub(p: Poly, q: Poly) -> Poly:
    return _padd(p, tuple(-x for x in q))


def _pmul_eta(p: Poly) -> Poly:
    """η · p(η) — décale les coefficients d'un degré."""
    return _pnorm((Fraction(0),) + tuple(p))


def _pshift(p: Poly, c: Fraction) -> Poly:
    """p(η) − c."""
    q = list(p)
    q[0] = q[0] - c
    return _pnorm(q)


def _peval(p: Poly, x: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * x + c
    return acc


def _pdeg(p: Poly) -> int:
    p = _pnorm(p)
    return -1 if p == (Fraction(0),) else len(p) - 1


def trace_e_polys(n: int, k: int,
                  pattern: Tuple[Optional[Fraction], ...]) -> List[Poly]:
    """e_t(η) SYMBOLIQUES sur une branche de motif de clip FIXÉ (émission 2.2).

    Même récurrence que ``trace_full_exact``, g et e portés en polynômes de η :
    g_raw = g − η·(e − 1) ; si le motif dit « clipé au pas t », g devient la
    CONSTANTE de la borne (le degré retombe à 0). Vérité algébrique attendue :
    e_0 = 1 et e_1 constants, e_2 affine (branche non clipée), deg(e_t) ≤ t−1.
    """
    inc_plus = Fraction(2, 3) * n / k
    inc_minus = Fraction(1, 3) * n / (n - k)
    g: Poly = (Fraction(1),)
    e: Poly = (Fraction(1),)
    polys: List[Poly] = [e]
    for t in range(n):
        g_raw = _psub(g, _pmul_eta(_pshift(e, TARGET_EXACT)))
        g = (pattern[t],) if pattern[t] is not None else g_raw
        inc = inc_plus if t < k else inc_minus
        e = _pshift(_padd(e, g), inc)
        polys.append(e)
    return polys


def simplest_in_closed(lo: Fraction, hi: Fraction) -> Fraction:
    """Le rationnel de PLUS PETIT dénominateur dans [lo, hi] (Stern–Brocot exact).

    Seul candidat testé comme racine exacte d'un polynôme porteur de degré ≥ 2
    (émission 2.3(c) : jamais affirmé rationnel sans racine exacte vérifiée).
    """
    if lo > hi:
        raise ValueError("intervalle vide")
    c = Fraction(math.ceil(lo))
    if c <= hi:
        return c
    fl = math.floor(lo)
    return fl + 1 / simplest_in_closed(1 / (hi - fl), 1 / (lo - fl))


# --- frontières : bissection exacte, fusion sous-résolution, certificat -------------

@dataclass(frozen=True)
class Frontier:
    """Une frontière de signe : encadrement (lo, hi], certifiée rationnelle ou non.

    ``carried_by`` = indices t des hits organe qui changent entre lo et hi ;
    ``degrees`` = degrés des polynômes porteurs sur la branche (motif de clip
    stable) ; ``certified`` = η rationnel EXACT si le certificat passe, sinon None
    (encadrement, jamais affirmé rationnel) ; ``is_e2`` = famille (i) du protocole
    (porté par e_2 seul, branche non clipée au pas 1, degré 1).
    """

    n: int
    k: int
    lo: Fraction
    hi: Fraction
    sign_left: int
    sign_right: int
    intermediate_signs: Tuple[int, ...]   # signes fusionnés sous-résolution (rapportés)
    carried_by: Tuple[int, ...]
    degrees: Tuple[int, ...]
    clip_stable: bool
    certified: Optional[Fraction]
    is_e2: bool


def _localize(n: int, k: int, a: Fraction, sa: int, b: Fraction, sb: int
              ) -> List[Tuple[Fraction, int, Fraction, int]]:
    """Bissection rationnelle exacte de (a, b] (sa ≠ sb) → transitions ≤ 2⁻²⁰.

    Un point médian portant un TROISIÈME signe scinde l'intervalle : la structure
    plus riche que la maille est découverte, pas écrasée (avertissement gelé §1).
    """
    out: List[Tuple[Fraction, int, Fraction, int]] = []
    stack = [(a, sa, b, sb)]
    while stack:
        a, sa, b, sb = stack.pop()
        while b - a > RESOLUTION:
            m = (a + b) / 2
            sm = sign_at(n, k, m)
            if sm == sa:
                a = m
            elif sm == sb:
                b = m
            else:                          # troisième signe : scinder
                stack.append((m, sm, b, sb))
                b, sb = m, sm
        out.append((a, sa, b, sb))
    out.sort(key=lambda tr: (tr[0], tr[2]))
    return out


def _certify(n: int, k: int, lo: Fraction, sl: int, hi: Fraction, sr: int,
             intermediate: Tuple[int, ...]) -> Frontier:
    """Certificat de rationalité (émission 2.3(c)) — jamais affirmé sans preuve."""
    h_lo, h_hi = _hits(n, k, lo), _hits(n, k, hi)
    diff = tuple(sorted(h_lo ^ h_hi))
    pat = _clip_pattern(n, k, lo)
    clip_stable = pat == _clip_pattern(n, k, hi)
    base = dict(n=n, k=k, lo=lo, hi=hi, sign_left=sl, sign_right=sr,
                intermediate_signs=intermediate, carried_by=diff,
                clip_stable=clip_stable)

    def _uncert(degrees: Tuple[int, ...] = ()) -> Frontier:
        return Frontier(degrees=degrees, certified=None, is_e2=False, **base)

    if not clip_stable or not diff:
        return _uncert()

    e_lo, _g1, _c1 = _trace_full(n, k, lo)
    e_hi, _g2, _c2 = _trace_full(n, k, hi)
    polys = trace_e_polys(n, k, pat)
    degrees: List[int] = []
    roots: set = set()
    ok = True
    for t in diff:
        outside = e_hi[t] if t in h_lo else e_lo[t]   # valeur hors bande
        edge = (TARGET_EXACT + BAND_EXACT if outside > TARGET_EXACT
                else TARGET_EXACT - BAND_EXACT)
        q = _pshift(polys[t], edge)
        d = _pdeg(q)
        degrees.append(d)
        if d == 1:
            r = -q[0] / q[1]
            if lo <= r <= hi:
                roots.add(r)
            else:
                ok = False
        elif d >= 2:
            r = simplest_in_closed(lo, hi)
            if _peval(q, r) == 0:
                roots.add(r)
            else:
                ok = False
        else:
            ok = False
    degs = tuple(degrees)
    if not ok or len(roots) != 1:
        return _uncert(degs)
    r = roots.pop()
    # vérifications DIRECTES au point exact (indépendantes des polynômes)
    if _clip_pattern(n, k, r) != pat:
        return _uncert(degs)
    e_r, _gr, _cr = _trace_full(n, k, r)
    if any(abs(e_r[t] - TARGET_EXACT) != BAND_EXACT for t in diff):
        return _uncert(degs)
    if r > lo and sign_at(n, k, (lo + r) / 2) != sl:
        return _uncert(degs)
    if r < hi and sign_at(n, k, (r + hi) / 2) != sr:
        return _uncert(degs)
    if len(intermediate) == 1 and sign_at(n, k, r) != intermediate[0]:
        return _uncert(degs)
    is_e2 = (diff == (2,) and pat[1] is None and degs == (1,))
    return Frontier(degrees=degs, certified=r, is_e2=is_e2, **base)


# --- structure d'une cellule à une maille donnée ------------------------------------

@dataclass(frozen=True)
class MeshStructure:
    """Phases + frontières d'une cellule, dérivées d'UNE maille (comparables P1)."""

    mesh: int
    phase_signs: Tuple[int, ...]          # phases de mesure > 0, ordonnées
    frontiers: Tuple[Frontier, ...]
    terminal_point_at_4: bool             # frontière certifiée EXACTEMENT à η = 4
    terminal_micro_at_4: bool             # segment terminal ≤ 2⁻²⁰ non résolu en 4
    n_gap_inconsistencies: int            # signes discordants entre frontières (attendu 0)

    @property
    def n_phases(self) -> int:
        return len(self.phase_signs)

    @property
    def n_frontiers(self) -> int:
        return len(self.frontiers)

    @property
    def counts(self) -> Tuple[int, int, bool, bool]:
        """Signature comparée entre mailles (P1)."""
        return (self.n_phases, self.n_frontiers,
                self.terminal_point_at_4, self.terminal_micro_at_4)


def _mesh_nodes(mesh: int) -> List[Fraction]:
    return [Fraction(i, mesh) for i in range(1, 4 * mesh + 1)]


def structure_at_mesh(n: int, k: int, mesh: int) -> MeshStructure:
    """Balayage exact aux nœuds i/mesh, localisation et certification des frontières."""
    nodes = _mesh_nodes(mesh)
    signs = [sign_at(n, k, x) for x in nodes]
    raw: List[Tuple[Fraction, int, Fraction, int]] = []
    for i in range(len(nodes) - 1):
        if signs[i] != signs[i + 1]:
            raw.extend(_localize(n, k, nodes[i], signs[i], nodes[i + 1], signs[i + 1]))
    raw.sort(key=lambda tr: (tr[0], tr[2]))

    # fusion des transitions séparées par un écart ≤ 2⁻²⁰ (frontière à signes
    # intermédiaires sous-résolution — rapportée, jamais écrasée)
    groups: List[List[Tuple[Fraction, int, Fraction, int]]] = []
    for tr in raw:
        if groups and tr[0] - groups[-1][-1][2] <= RESOLUTION:
            groups[-1].append(tr)
        else:
            groups.append([tr])
    frontiers = tuple(
        _certify(n, k, g[0][0], g[0][1], g[-1][2], g[-1][3],
                 tuple(tr[3] for tr in g[:-1]))
        for g in groups)

    phase_signs: List[int] = []
    terminal_point = terminal_micro = False
    n_gap_inc = 0
    if not frontiers:
        phase_signs = [signs[0]]
    else:
        phase_signs.append(frontiers[0].sign_left)
        for fa, fb in zip(frontiers, frontiers[1:]):
            if fa.sign_right != fb.sign_left:
                n_gap_inc += 1
            phase_signs.append(fb.sign_left)
        last = frontiers[-1]
        if last.certified == DOMAIN_MAX:
            terminal_point = True                 # le bord droit est un POINT, pas une phase
        elif last.hi == DOMAIN_MAX and last.certified is None:
            terminal_micro = True                 # micro-segment ≤ 2⁻²⁰ signalé
        else:
            phase_signs.append(last.sign_right)
    return MeshStructure(
        mesh=mesh, phase_signs=tuple(phase_signs), frontiers=frontiers,
        terminal_point_at_4=terminal_point, terminal_micro_at_4=terminal_micro,
        n_gap_inconsistencies=n_gap_inc,
    )


# --- diagramme d'une cellule : convergence de maille (P1) ---------------------------

@dataclass(frozen=True)
class CellDiagram:
    """Le diagramme de phase d'une cellule : structure la plus fine + stabilité P1."""

    n: int
    k: int
    high: bool
    side: str
    counts12: Tuple[int, int, bool, bool]
    counts24: Tuple[int, int, bool, bool]
    counts48: Optional[Tuple[int, int, bool, bool]]
    stable: bool                          # compte stable entre les 2 dernières mailles
    needed_48: bool
    structure: MeshStructure              # la maille la plus fine utilisée


def cell_diagram(n: int, k: int) -> CellDiagram:
    """P1 : stable ⟺ counts(1/12) == counts(1/24), sinon 1/48 exigé et re-comparé."""
    s12 = structure_at_mesh(n, k, MESH_PRIMARY)
    s24 = structure_at_mesh(n, k, MESH_CONVERGENCE)
    if s12.counts == s24.counts:
        return CellDiagram(n=n, k=k, high=is_high_excursion(n, k),
                           side=flip_side(n, k), counts12=s12.counts,
                           counts24=s24.counts, counts48=None,
                           stable=True, needed_48=False, structure=s24)
    s48 = structure_at_mesh(n, k, MESH_FINE)
    return CellDiagram(n=n, k=k, high=is_high_excursion(n, k),
                       side=flip_side(n, k), counts12=s12.counts,
                       counts24=s24.counts, counts48=s48.counts,
                       stable=s24.counts == s48.counts, needed_48=True,
                       structure=s48)


# --- rapport grille entière ----------------------------------------------------------

@dataclass(frozen=True)
class PhaseDiagramReport:
    """Le diagramme (N, k, η) complet : 228 cellules, agrégats et prédicats."""

    cells: List[CellDiagram]
    histogram: Dict[int, int]             # compte de phases → nombre de cellules
    n_frontiers_total: int
    certified: List[Frontier]             # frontières certifiées rationnelles
    n_e2: int                             # famille (i) : portées par e_2 (protocole)
    certified_beyond_e2: List[Frontier]   # certifiées rationnelles portées par e_{t≥3}
    encadrements: List[Frontier]          # jamais affirmées rationnelles
    unstable_cells: List[Tuple[int, int]]
    needed48_cells: List[Tuple[int, int]]
    majority_le2: bool                    # conjecture annexe P1
    n_gap_inconsistencies: int


def run_phase_diagram() -> PhaseDiagramReport:
    """Exécute le protocole gelé sur les 228 cellules (déterministe, exact)."""
    cells = [cell_diagram(n, k) for (n, k) in grid_cells()]
    histogram: Dict[int, int] = {}
    certified: List[Frontier] = []
    encadrements: List[Frontier] = []
    n_front = 0
    n_gap = 0
    for c in cells:
        histogram[c.structure.n_phases] = histogram.get(c.structure.n_phases, 0) + 1
        n_gap += c.structure.n_gap_inconsistencies
        for f in c.structure.frontiers:
            n_front += 1
            (certified if f.certified is not None else encadrements).append(f)
    n_e2 = sum(1 for f in certified if f.is_e2)
    beyond = [f for f in certified if not f.is_e2 and f.carried_by
              and min(f.carried_by) >= 3]
    le2 = sum(v for p, v in histogram.items() if p <= 2)
    return PhaseDiagramReport(
        cells=cells, histogram=dict(sorted(histogram.items())),
        n_frontiers_total=n_front, certified=certified, n_e2=n_e2,
        certified_beyond_e2=beyond, encadrements=encadrements,
        unstable_cells=[(c.n, c.k) for c in cells if not c.stable],
        needed48_cells=[(c.n, c.k) for c in cells if c.needed_48],
        majority_le2=le2 > len(cells) // 2,
        n_gap_inconsistencies=n_gap,
    )


# --- P2 : les 19 cellules basses -----------------------------------------------------

@dataclass(frozen=True)
class P2Report:
    """P-gate continue : signe = 0 à tous les nœuds 1/48 < 4 ; ≠ 0 seulement à η = 4."""

    n_low_cells: int
    n_nodes_checked: int
    nonzero_below_4: List[Tuple[int, int, Fraction, int]]   # attendu : []
    nonzero_at_4: List[Tuple[int, int, int]]                # (8,5) et (11,7) attendues (T28)
    passes: bool


def p2_report() -> P2Report:
    """Vérifie P2 aux nœuds de la maille fine 1/48 (les mailles gelées incluses)."""
    nodes = _mesh_nodes(MESH_FINE)
    low = [(n, k) for (n, k) in grid_cells() if not is_high_excursion(n, k)]
    bad: List[Tuple[int, int, Fraction, int]] = []
    at4: List[Tuple[int, int, int]] = []
    checked = 0
    for (n, k) in low:
        for x in nodes:
            s = sign_at(n, k, x)
            checked += 1
            if x < DOMAIN_MAX and s != 0:
                bad.append((n, k, x, s))
        s4 = sign_at(n, k, DOMAIN_MAX)
        if s4 != 0:
            at4.append((n, k, s4))
    return P2Report(n_low_cells=len(low), n_nodes_checked=checked,
                    nonzero_below_4=bad, nonzero_at_4=at4, passes=not bad)


# --- P5 : résonances de périodicité exacte aux Niven ---------------------------------

@dataclass(frozen=True)
class ResonanceReport:
    """Comptes de fenêtres exactement p-périodiques : pics attendus aux Niven (P5-ii)."""

    per_niven: Dict[Fraction, Tuple[Optional[int], int, Optional[int]]]
    # η* → (compte à η*−1/12, compte à η*, compte à η*+1/12) pour SA période p
    curve: Dict[Fraction, int]            # nœud 1/12 → Σ_p fenêtres exactes (p ∈ 2,3,4,6)
    peaks_at_niven: Dict[Fraction, bool]  # curve(η*) > voisins immédiats


def periodic_windows(eta: Fraction, p: int) -> Tuple[int, int]:
    """(fenêtres éligibles, fenêtres avec u_{t+p} == u_t EXACT) sur la grille.

    Fenêtre éligible en (N, k), t : t ∈ [0, k−p] (plateau, incrément constant) et
    pas t+1..t+p−1 non clipés — généralisation directe de ``period6_report`` T28
    (qui est le cas η = 1, p = 6 : 884/884).
    """
    n_win = n_ok = 0
    for (n, k) in grid_cells():
        if k < p:
            continue
        e, _gs, clips = _trace_full(n, k, eta)
        u = [x - TARGET_EXACT for x in e]
        for t in range(0, k - p + 1):
            if any(clips[s] for s in range(t + 1, t + p)):
                continue
            n_win += 1
            if u[t + p] == u[t]:
                n_ok += 1
    return n_win, n_ok


def resonance_report() -> ResonanceReport:
    """Mesure P5-ii : le compte de fenêtres exactement périodiques pique-t-il aux Niven ?"""
    step = Fraction(1, MESH_PRIMARY)
    per_niven: Dict[Fraction, Tuple[Optional[int], int, Optional[int]]] = {}
    for eta_star, p in NIVEN_PERIODS.items():
        left = periodic_windows(eta_star - step, p)[1] if eta_star - step > 0 else None
        mid = periodic_windows(eta_star, p)[1]
        right = (periodic_windows(eta_star + step, p)[1]
                 if eta_star + step <= DOMAIN_MAX else None)
        per_niven[eta_star] = (left, mid, right)
    curve: Dict[Fraction, int] = {}
    for x in _mesh_nodes(MESH_PRIMARY):
        curve[x] = sum(periodic_windows(x, p)[1] for p in PERIODS_SCANNED)
    peaks: Dict[Fraction, bool] = {}
    for eta_star in NIVEN_PERIODS:
        neigh = [curve[eta_star - step]] if eta_star - step > 0 else []
        if eta_star + step <= DOMAIN_MAX:
            neigh.append(curve[eta_star + step])
        peaks[eta_star] = all(curve[eta_star] > v for v in neigh)
    return ResonanceReport(per_niven=per_niven, curve=curve, peaks_at_niven=peaks)


# --- vérification instrument : points de contrôle float + bornes d'encadrements -----

@dataclass(frozen=True)
class FloatControlReport:
    """Concordance de SIGNE exact ↔ float aux 5 points de contrôle a priori."""

    matches: Dict[str, int]                                  # label → /228
    sign_mismatches: List[Tuple[str, int, int]]              # attendu : [] (BUG sinon)
    magnitude_divergences: Dict[str, List[Tuple[int, int, float, str, int]]]
    passes: bool


def float_control_report() -> FloatControlReport:
    """Les 5 η dyadiques : signe float (instrument gelé) == signe Fraction, 228/228."""
    matches: Dict[str, int] = {}
    mism: List[Tuple[str, int, int]] = []
    mags: Dict[str, List[Tuple[int, int, float, str, int]]] = {}
    for label, eta in CONTROL_ETAS.items():
        ok = 0
        mags[label] = []
        for (n, k) in grid_cells():
            d_exact = delta_nom_exact_eta(n, k, eta)
            d_float = delta_nom_float_eta(n, k, float(eta))
            s_e = (d_exact > 0) - (d_exact < 0)
            s_f = (d_float > 0) - (d_float < 0)
            if s_e == s_f:
                ok += 1
            else:
                mism.append((label, n, k))
            if abs(d_float - float(d_exact)) > 1e-9:
                mags[label].append((n, k, d_float, str(d_exact),
                                    band_edge_touches_eta(n, k, eta)))
        matches[label] = ok
    return FloatControlReport(matches=matches, sign_mismatches=mism,
                              magnitude_divergences=mags, passes=not mism)


@dataclass(frozen=True)
class EndpointFloatReport:
    """Signes float aux bornes (lo, hi) de chaque frontière == signes exacts."""

    n_checked: int
    sign_mismatches: List[Tuple[int, int, Fraction, int, int]]   # attendu : []
    passes: bool


def endpoint_float_report(cells: Sequence[CellDiagram]) -> EndpointFloatReport:
    """Vérification ciblée gelée : l'instrument float aux bornes des encadrements.

    Les bornes ne sont pas dyadiques (mailles 1/12·2⁻ʲ) : float(Fraction) arrondit
    à ~1 ulp, très en dessous de la largeur 2⁻²⁰ des encadrements — le signe doit
    concorder. Toute divergence de SIGNE est rapportée (BUG, arrêt technique).
    """
    checked = 0
    mism: List[Tuple[int, int, Fraction, int, int]] = []
    for c in cells:
        for f in c.structure.frontiers:
            for pt, s_exact in ((f.lo, f.sign_left), (f.hi, f.sign_right)):
                d = delta_nom_float_eta(c.n, c.k, float(pt))
                s_f = (d > 0) - (d < 0)
                checked += 1
                if s_f != s_exact:
                    mism.append((c.n, c.k, pt, s_exact, s_f))
    return EndpointFloatReport(n_checked=checked, sign_mismatches=mism,
                               passes=not mism)


# --- P3 : pas de loi monotone du compte de phases (descriptif, jamais forcé) --------

def phase_count_map(cells: Sequence[CellDiagram]) -> List[str]:
    """Carte des cardinaux de phases (lignes N = 6..24, colonnes k = 2..N−2)."""
    by_cell = {(c.n, c.k): c.structure.n_phases for c in cells}
    rows: List[str] = []
    for n in range(GRID_N_MIN, GRID_N_MAX + 1):
        rows.append("".join(
            (str(p) if p < 10 else "*")
            for k in range(GRID_K_MIN, n - 1)
            for p in [by_cell[(n, k)]]))
    return rows


def _is_monotone(seq: Sequence[int]) -> bool:
    return (all(a <= b for a, b in zip(seq, seq[1:]))
            or all(a >= b for a, b in zip(seq, seq[1:])))


@dataclass(frozen=True)
class P3Report:
    """Le compte de phases suit-il une loi monotone en k, N ou N−k ? (surprise si oui)."""

    monotone_in_k_all_rows: bool          # à N fixé, monotone en k pour TOUTES les lignes
    monotone_in_n_all_cols: bool          # à k fixé, monotone en N pour TOUTES les colonnes
    monotone_in_leg_all: bool             # à N−k fixé, monotone en N
    n_rows_monotone: int
    n_rows: int


def p3_report(cells: Sequence[CellDiagram]) -> P3Report:
    by_cell = {(c.n, c.k): c.structure.n_phases for c in cells}
    rows = [[by_cell[(n, k)] for k in range(GRID_K_MIN, n - 1)]
            for n in range(GRID_N_MIN, GRID_N_MAX + 1)]
    cols: Dict[int, List[int]] = {}
    legs: Dict[int, List[int]] = {}
    for n in range(GRID_N_MIN, GRID_N_MAX + 1):
        for k in range(GRID_K_MIN, n - 1):
            cols.setdefault(k, []).append(by_cell[(n, k)])
            legs.setdefault(n - k, []).append(by_cell[(n, k)])
    return P3Report(
        monotone_in_k_all_rows=all(_is_monotone(r) for r in rows),
        monotone_in_n_all_cols=all(_is_monotone(v) for v in cols.values()),
        monotone_in_leg_all=all(_is_monotone(v) for v in legs.values()),
        n_rows_monotone=sum(1 for r in rows if _is_monotone(r)),
        n_rows=len(rows),
    )


# --- GRAVURE POST-MESURE (2026-07-12) : synthèse gelée du diagramme mesuré ----------
#
# Ces constantes sont GRAVÉES d'après l'exécution du protocole gelé (voir VERDICT
# MESURÉ en tête de module) ; les tests vérifient que le recalcul les reproduit.

FROZEN_HISTOGRAM: Dict[int, int] = {1: 64, 2: 35, 3: 65, 4: 37, 5: 18,
                                    6: 3, 7: 2, 8: 1, 9: 3}
FROZEN_TOTALS: Dict[str, int] = {
    "frontiers": 406, "certified": 78, "e2": 4, "certified_beyond_e2": 74,
    "encadrements": 328, "unstable": 35, "needed_48": 82,
}
# famille (i) : les 4 certificats e₂ (rationnels exacts, aucun aux Niven)
FROZEN_E2_CERTIFICATES: Dict[Tuple[int, int], Fraction] = {
    (7, 3): Fraction(11, 10), (8, 4): Fraction(1, 2),
    (10, 4): Fraction(5, 4), (17, 9): Fraction(1, 14),
}
# résonance η = 3 : cellules à frontière certifiée EXACTEMENT à 3 (7 × N = 3k + (7,3))
FROZEN_ETA3_CERTIFIED: Tuple[Tuple[int, int], ...] = (
    (6, 2), (7, 3), (9, 3), (12, 4), (15, 5), (18, 6), (21, 7), (24, 8))
# la racine rationnelle NON générique d'un porteur de degré 2 (réfutation fine 2.2)
FROZEN_DEG2_RATIONAL: Tuple[Tuple[int, int], Fraction] = ((15, 6), Fraction(1, 2))
# P2 : nœuds 1/48 < 4 à signe ≠ 0 sur cellules basses (violations SOUS 4, gravées)
FROZEN_P2_NONZERO_BELOW_4: Tuple[Tuple[int, int, Fraction, int], ...] = (
    (8, 5, Fraction(191, 48), -1), (10, 7, Fraction(95, 24), -1),
    (11, 7, Fraction(95, 24), -1), (11, 7, Fraction(191, 48), -1))
FROZEN_P2_AT_4: Tuple[Tuple[int, int, int], ...] = ((8, 5, -1), (11, 7, -1))
# P5 : comptes de fenêtres exactement p-périodiques (voisin−, η*, voisin+)
FROZEN_NIVEN_COUNTS: Dict[int, Tuple[Optional[int], int, Optional[int]]] = {
    1: (36, 884, 36), 2: (49, 1178, 49), 3: (56, 1292, 56), 4: (63, 185, None),
}
# vérification instrument : divergences de magnitude aux 5 points de contrôle
FROZEN_CONTROL_MAGNITUDE_COUNTS: Dict[str, int] = {
    "3/4": 2, "2": 6, "5/2": 2, "3": 8, "7/2": 3,
}
# bornes d'encadrements : 5 divergences de SIGNE, TOUTES (9,7) — réserve T28
# (e_8 du lecteur fixe == 3/2 exact ⟹ Δ_float(9,7) = Δ_exact + 1/9 à tout η)
FROZEN_ENDPOINT_CHECKS = 812
FROZEN_ENDPOINT_SIGN_MISMATCHES = 5
FROZEN_ENDPOINT_MISMATCH_CELL: Tuple[int, int] = (9, 7)
FROZEN_97_FIXED_OFFSET = Fraction(1, 9)   # f_edge fixe exact 5/9 vs float 4/9


# --- runner de mesure ----------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover — runner déterministe de mesure
    print("=== PORTE 0a re-confirmée (héritée T27, δ_min=1e-4) ===")
    g0a = gate0a()
    print(f"(13,8) primaire gap={g0a.primary_high.gap:.6e} sensible={g0a.primary_high.is_order_sensitive} | "
          f"multiset gap={g0a.multiset_high.gap:.6e} vacuous={g0a.multiset_high.is_vacuous}")
    print(f"(7,5)  primaire gap={g0a.primary_low.gap:.6e} sensible={g0a.primary_low.is_order_sensitive} | "
          f"multiset gap={g0a.multiset_low.gap:.6e} vacuous={g0a.multiset_low.is_vacuous}")
    print(f"porte 0a : {'PASSE' if g0a.passes else 'ECHEC'}")

    print("\n=== DIAGRAMME DE PHASE (mailles 1/12 -> 1/24 -> 1/48 au besoin) ===")
    rep = run_phase_diagram()
    print(f"histogramme des comptes de phases : {rep.histogram}")
    print(f"majorité <= 2 phases (conjecture annexe P1) : {rep.majority_le2}")
    print(f"frontières totales : {rep.n_frontiers_total} ; certifiées rationnelles : "
          f"{len(rep.certified)} (dont e2 : {rep.n_e2}) ; encadrements : {len(rep.encadrements)}")
    print(f"certifiées portées par e_(t>=3) (test 2.2) : {len(rep.certified_beyond_e2)}")
    print(f"cellules ayant requis 1/48 : {len(rep.needed48_cells)} {rep.needed48_cells}")
    print(f"cellules NON stables (P1) : {len(rep.unstable_cells)} {rep.unstable_cells}")
    print(f"incohérences de signes inter-frontières : {rep.n_gap_inconsistencies}")

    print("\n--- carte des cardinaux de phases (N=6..24, k=2..N-2) ---")
    for i, row in enumerate(phase_count_map(rep.cells)):
        print(f"N={GRID_N_MIN + i:2d} : {row}")

    p3 = p3_report(rep.cells)
    print(f"\nP3 : monotone en k (toutes lignes) : {p3.monotone_in_k_all_rows} ; "
          f"monotone en N (toutes colonnes) : {p3.monotone_in_n_all_cols} ; "
          f"monotone en N-k : {p3.monotone_in_leg_all} ; "
          f"lignes monotones : {p3.n_rows_monotone}/{p3.n_rows}")

    print("\n--- frontières CERTIFIÉES rationnelles ---")
    for f in rep.certified:
        print(f"({f.n},{f.k}) eta={f.certified} carried_by={f.carried_by} "
              f"deg={f.degrees} e2={f.is_e2} signes {f.sign_left}->{f.sign_right} "
              f"inter={f.intermediate_signs}")

    print(f"\n--- encadrements (non certifiés) : {len(rep.encadrements)} ---")
    for f in rep.encadrements:
        print(f"({f.n},{f.k}) ({f.lo} , {f.hi}] ~({float(f.lo):.7f},{float(f.hi):.7f}] "
              f"carried_by={f.carried_by} deg={f.degrees} clip_stable={f.clip_stable} "
              f"signes {f.sign_left}->{f.sign_right} inter={f.intermediate_signs}")

    print("\n=== P2 : les 19 cellules basses (nœuds 1/48) ===")
    p2 = p2_report()
    print(f"cellules basses : {p2.n_low_cells} ; nœuds vérifiés : {p2.n_nodes_checked}")
    print(f"non-zéro sous 4 : {p2.nonzero_below_4} (attendu []) ; "
          f"non-zéro à 4 : {p2.nonzero_at_4}")
    print(f"P2 : {'TIENT' if p2.passes else 'VIOLÉE'}")

    print("\n=== P4 : (7,4) et (8,6) ===")
    by_cell = {(c.n, c.k): c for c in rep.cells}
    for cell in [(7, 4), (8, 6)]:
        c = by_cell[cell]
        print(f"{cell} : phases {c.structure.phase_signs} stable={c.stable}")
        for f in c.structure.frontiers:
            print(f"  frontière ({f.lo}, {f.hi}] certifiée={f.certified} "
                  f"carried={f.carried_by} deg={f.degrees} e2={f.is_e2} "
                  f"{f.sign_left}->{f.sign_right} inter={f.intermediate_signs}")

    print("\n=== P5 : résonances Niven (fenêtres exactement périodiques) ===")
    rr = resonance_report()
    for eta_star, (l, m, r) in rr.per_niven.items():
        print(f"eta*={eta_star} (p={NIVEN_PERIODS[eta_star]}) : "
              f"voisin- {l} | eta* {m} | voisin+ {r} ; pic={rr.peaks_at_niven[eta_star]}")

    print("\n=== VÉRIFICATION INSTRUMENT : 5 points de contrôle float ===")
    fc = float_control_report()
    for label in CONTROL_ETAS:
        print(f"eta={label} : signes {fc.matches[label]}/228 ; "
              f"divergences magnitude : {len(fc.magnitude_divergences[label])} "
              f"{fc.magnitude_divergences[label]}")
    print(f"concordance de signe : {'PASSE' if fc.passes else 'BUG (arrêt technique)'} "
          f"{fc.sign_mismatches}")

    print("\n=== VÉRIFICATION INSTRUMENT : bornes des encadrements ===")
    ep = endpoint_float_report(rep.cells)
    print(f"bornes vérifiées : {ep.n_checked} ; divergences de signe : "
          f"{ep.sign_mismatches} -> {'PASSE' if ep.passes else 'BUG (arrêt technique)'}")
