from __future__ import annotations

"""Pré-validation d'instrument : un observable est-il SENSIBLE À L'ORDRE ? (Tour 23, méta).

MÉTA-TOUR de second ordre : la danse corrige sa propre MÉTHODE. Trois fois (T17,
T18, T22) un observable GELÉ a priori s'est révélé AVEUGLE au signal cherché. Le cas
T22 est mathématiquement net : la straightness ``R = ‖Σ d_t‖ / Σ‖d_t‖`` d'un
intégrateur additif ``s_{t+1} = s_t + g_t·v_t`` est EXACTEMENT invariante à l'ordre
des entrées PAR CONSTRUCTION — l'addition vectorielle commute, donc le déplacement
net ``Σ d_t`` et la longueur de chemin ``Σ‖d_t‖`` ne dépendent pas de l'ordre des
``v_t`` (à gain CONSTANT). Mesuré sur le réel : ``R_réel − R_shuffle = 6.06e-8`` =
bruit float pur. ``R`` était donc un INSTRUMENT VACUOUS pour un test d'ordre, et on
ne l'a su qu'APRÈS avoir tokenisé le réel. Seul l'α-ω ``cos(s_α, ·)`` a porté le
verdict, parce que ``s_α = s(fin A)`` est une somme PARTIELLE (donc order-sensible).

LA LEÇON, RENDUE ENFORCEABLE. AVANT de mesurer un observable sur le réel pour un
verdict d'ORDRE (ou de dérive temporelle), on exige DEUX tests A PRIORI :

  1. SENSIBILITÉ (pivot dégénéré — déjà en place dans les modules T15-T22) : quand on
     annule la variable censée piloter l'observable (gain constant / dérive constante /
     amplitude=0), l'observable ne doit PAS produire d'avantage (≈0). Couvert par les
     pivots dégénérés existants ; ce module ne le ré-implémente pas.

  2. NON-VACUITÉ D'ORDRE (NOUVEAU, ce module) : quand on SHUFFLE l'ordre de la séquence
     d'entrée À DYNAMIQUE FIXE (mêmes poids, même cellule, même graine d'état initial —
     SEUL l'ordre des entrées change), l'observable DOIT bouger d'au moins ``δ_min``.
     Sinon il est order-invariant PAR CONSTRUCTION = VACUOUS, donc INTERDIT comme test
     d'ordre. C'est précisément ce que ``R`` aurait échoué et ce que l'α-ω aurait passé.

CE MODULE NE FORCE RIEN. Il MESURE l'écart ``gap = |obs(réel) − obs(shuffle)|`` et le
compare à un seuil GELÉ ``δ_min``, puis RAPPORTE ``is_vacuous`` / ``is_order_sensitive``.
Il ne « répare » aucun observable et n'altère aucun verdict : il dit seulement si
l'instrument a le DROIT de servir de primaire pour l'ordre. L'ingénieur statue.

AGNOSTIQUE AU SUBSTRAT. ``obs_fn`` est une fonction ``séquence -> float`` ; ``shuffle_fn``
est une fonction ``(séquence, seed) -> séquence`` qui permute l'ORDRE sans toucher au
contenu (même multiset). La séquence peut être un tenseur ``(N, d)``, une liste de
tokens, n'importe quoi — le helper ne lit jamais l'intérieur. Pour réduire le bruit
d'un shuffle particulier on agrège ``n_shuffle`` permutations seedées (médiane).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, TypeVar


# --- seuil GELÉ A PRIORI (REFUS : jamais réglé sur un résultat) ----------------
#
# δ_min sépare « bruit float d'un observable order-invariant » de « vrai signal
# d'ordre ». Repères MESURÉS, pas choisis pour faire passer un cas :
#   * borne BASSE (bruit) : l'intégrateur additif T22 donne |R_réel − R_shuffle|
#     ≈ 6e-8 (invariance par construction ; il ne reste que l'arithmétique float).
#   * borne HAUTE (signal) : un observable order-sensible comme l'α-ω sur somme
#     PARTIELLE bouge de l'ordre de 1e-1 à 1e0 quand l'ordre change.
# On gèle δ_min = 1e-4 : ~3-4 ordres de grandeur AU-DESSUS du bruit float (6e-8) et
# ~3 ordres EN DESSOUS d'un vrai signal d'ordre. La zone [6e-8, 1e-4] est une marge
# de sécurité : tout ce qui y tombe est traité comme VACUOUS (prudence — un
# instrument à peine sensible ne doit pas porter un verdict d'ordre).
DELTA_MIN_DEFAULT = 1e-4

# Nombre de permutations agrégées par défaut (médiane des gaps). Robustifie contre un
# shuffle « chanceux » sans dépendre d'une permutation unique.
N_SHUFFLE_DEFAULT = 8


S = TypeVar("S")  # type de la séquence (tenseur (N,d), liste de tokens, ...)


def _median(xs: Sequence[float]) -> float:
    """Médiane (interpolée pour n pair) — sans numpy (cohérent avec edge_maintenance)."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


@dataclass(frozen=True)
class OrderSensitivityReport:
    """Verdict de pré-validation d'un observable comme test d'ORDRE.

    Champs :
      * ``obs_real``   : ``obs_fn(séquence_réelle)`` (dynamique/poids/graine FIXES).
      * ``obs_shuffle``: médiane de ``obs_fn(séquence_shufflée)`` sur ``n_shuffle``
        permutations seedées (même dynamique, SEUL l'ordre change).
      * ``gap``        : ``|obs_real − obs_shuffle|``. Écart imputable à l'ORDRE seul.
      * ``delta_min``  : seuil GELÉ a priori sous lequel ``gap`` = bruit (order-invariant).
      * ``is_order_sensitive`` : ``gap ≥ delta_min`` → l'observable BOUGE avec l'ordre,
        autorisé comme primaire d'ordre.
      * ``is_vacuous``         : ``gap < delta_min`` → order-invariant (par construction
        ou en pratique) → INTERDIT comme test d'ordre.
      * ``obs_shuffles`` : la liste brute des observables shufflés (audit/déterminisme).
      * ``n_shuffle``    : nombre de permutations agrégées.

    Invariant : ``is_vacuous == (not is_order_sensitive)`` (partition stricte sur ``gap``).
    """

    obs_real: float
    obs_shuffle: float
    gap: float
    delta_min: float
    is_order_sensitive: bool
    is_vacuous: bool
    n_shuffle: int
    obs_shuffles: List[float] = field(default_factory=list)


def assert_order_sensitive(
    obs_fn: Callable[[S], float],
    sequence: S,
    *,
    shuffle_fn: Callable[[S, int], S],
    delta_min: float = DELTA_MIN_DEFAULT,
    n_shuffle: int = N_SHUFFLE_DEFAULT,
    seed_base: int = 0,
) -> OrderSensitivityReport:
    """Pré-valide ``obs_fn`` comme test d'ORDRE : bouge-t-il quand SEUL l'ordre change ?

    On évalue ``obs_fn`` sur la séquence RÉELLE puis sur ``n_shuffle`` permutations
    seedées de la séquence (``shuffle_fn(sequence, seed_base + k)``). La DYNAMIQUE est
    censée être figée À L'INTÉRIEUR de ``obs_fn`` (mêmes poids, même cellule, même
    graine d'état initial) : ``obs_fn`` reçoit UNIQUEMENT la séquence, donc tout ce qui
    diffère entre réel et shuffle est l'ORDRE des entrées. ``gap = |obs_real −
    median(obs_shuffle)|`` est alors entièrement imputable à l'ordre.

    Verdict (seuil ``delta_min`` GELÉ a priori) :
      * ``gap ≥ delta_min``  ⇒ ``is_order_sensitive = True``  (autorisé comme primaire) ;
      * ``gap < delta_min``  ⇒ ``is_vacuous = True``          (INTERDIT comme test d'ordre).

    Le helper NE FORCE RIEN : il rapporte le verdict, il ne corrige pas l'observable.
    Malgré son nom ``assert_*`` (convention de pré-validation), il NE LÈVE PAS : il
    retourne un rapport structuré que l'appelant inspecte. DÉTERMINISTE : les seeds de
    shuffle sont ``seed_base + k`` ; à ``obs_fn``/``shuffle_fn`` déterministes le rapport
    est bit-à-bit reproductible.

    Args:
      obs_fn:     séquence -> float. DOIT figer toute dynamique en interne (sinon le gap
                  mélange ordre et bruit de dynamique — défaut d'usage, pas du helper).
      sequence:   la séquence ordonnée à pré-valider (substrat quelconque).
      shuffle_fn: (séquence, seed) -> séquence permutée (même multiset, ordre détruit),
                  déterministe par seed.
      delta_min:  seuil GELÉ ; défaut ``DELTA_MIN_DEFAULT = 1e-4`` (justifié dans l'en-tête).
      n_shuffle:  nombre de permutations agrégées (médiane). ``≥ 1``.
      seed_base:  base des seeds de shuffle (déterminisme).

    Returns:
      ``OrderSensitivityReport``.
    """
    if n_shuffle < 1:
        raise ValueError("n_shuffle doit être >= 1")
    if delta_min < 0.0:
        raise ValueError("delta_min doit être >= 0")

    obs_real = float(obs_fn(sequence))

    obs_shuffles: List[float] = []
    for k in range(n_shuffle):
        shuffled = shuffle_fn(sequence, seed_base + k)
        obs_shuffles.append(float(obs_fn(shuffled)))
    obs_shuffle = _median(obs_shuffles)

    gap = abs(obs_real - obs_shuffle)
    is_order_sensitive = gap >= delta_min
    is_vacuous = not is_order_sensitive

    return OrderSensitivityReport(
        obs_real=obs_real,
        obs_shuffle=obs_shuffle,
        gap=gap,
        delta_min=delta_min,
        is_order_sensitive=is_order_sensitive,
        is_vacuous=is_vacuous,
        n_shuffle=n_shuffle,
        obs_shuffles=obs_shuffles,
    )
