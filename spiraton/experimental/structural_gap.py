from __future__ import annotations

"""``gap_struct`` — gap STRUCTUREL order-preserving sur cycles ABA (Tour 24, H24).

H24 (émission linguiste, geste SUB·lévo·in, revue Fable 5 intégrée) : l'organe
``regulate_step`` (T19/T20, INCHANGÉ, importé d'``edge_controller``) s'étend-il du
substrat numérique au substrat STRUCTUREL ? Le substrat est ici le **profil
d'orientation** d'un cycle ABA : la séquence ordonnée des tags DX/OUT (+1) et LV/IN
(−1) portés par les tokens, lue EXCLUSIVEMENT depuis le parseur ``data/aba.py``
(garde anti-circularité (a) : JAMAIS les dims 0-5 du 33D, jamais ``op_from_vector``).

POURQUOI CE SUBSTRAT PEUT PORTER L'ORDRE (leçon T22/T23). Les 6 nulls réels vivent
sur un intégrateur additif qui COMMUTE. Ici, la seule variation structurelle
inter-cycle d'un corpus 100 % conforme (T14 : motif (DX,DX,LV) constant) est la
POSITION du retournement ``k = |SEG_A|+|SEG_B|`` sur ``N`` tokens : ``k/N`` varie
avec les longueurs de segments. La reconstruction ci-dessous est un lecteur
STRICTEMENT MONOTONE : l'ordre des tokens détermine l'alignement de son flip avec le
flip observé — composition non-commutative par construction (readback structurel).

CHOIX D'IMPLÉMENTATION — TOUS GELÉS A PRIORI (déclarés AVANT toute mesure sur le
réel ; dérivés de la GRAMMAIRE ABA, jamais d'une statistique des données) :

  1. TOKENISATION = découpage en mots (whitespace) du texte des segments. Le profil
     est structurel : seuls comptent ``|A|``, ``|B|``, ``|A′|`` en tokens et la
     position du flip ``k = |A|+|B|``. Aucun besoin du tokenizer natif ``.so``
     (avantage : aucun skip de test ; le ``.so`` n'est jamais la seule source de
     vérité — CLAUDE.md chantier 3).

  2. UNITÉS TOKEN. Les deux phases sont comptées en TOKENS (pas en fraction de
     cycle) : le quantum grammatical « un token » vaut exactement 1.0, donc
     ``TARGET_LEAD = 1.0`` et ``BAND_LEAD = 0.5`` (la moitié du quantum) sortent de
     la grammaire sans conversion. Bénéfice de pureté : sur le pivot dégénéré
     (profil sans flip, incrément 1.0), l'arithmétique float est EXACTE (petits
     entiers) ⇒ le pivot Δ=0 est bit-à-bit, pas approché.

  3. PHASE VRAIE ``p_ref`` (graduée, order-preserving). Soit ``n_plus = #(+1)``,
     ``n_minus = #(−1)`` (comptes GLOBAUX du profil = son multiset). Chaque token
     lu avance la phase vraie de :
         inc(+1) = PHI_STAR·N / n_plus        (plateau avant flip)
         inc(−1) = (1−PHI_STAR)·N / n_minus   (descente après flip)
     avec ``PHI_STAR = 2/3`` : le flip canonique tombe après 2 segments sur 3
     (prior grammatical d'équilongueur — la grammaire a 3 segments, le retournement
     arrive au 3e). ``p_ref(N) = N`` exactement (les incréments somment à N). Pour
     un profil canonique à un seul flip, ``p_ref(t) = PHI_STAR·N·t/k`` sur le
     plateau — c'est le « φ_ref ∝ t/k avant flip » de l'émission. L'ORDRE entre par
     le CUMUL : shuffler les tokens réordonne les incréments ⇒ autre trajectoire
     (les comptes, eux, sont shuffle-invariants — c'est la variante-contrôle).
     DÉGÉNÉRÉ : si ``n_plus == 0`` ou ``n_minus == 0`` (profil sans flip), aucune
     information de flip ⇒ incrément UNIFORME 1.0 (horloge parfaite).

  4. LECTEUR MONOTONE PILOTÉ PAR L'ACTIONNEUR. ``p_read(0) = TARGET_LEAD`` (offset
     initial = l'avance-cible d'un token, voir 5) puis ``p_read(t+1) = p_read(t) +
     g_t`` : le rythme d'avance EST le gain ``g_t``. Le flip reconstruit :
     ``ô_t = +1`` si ``p_read(t) < PHI_STAR·N`` sinon ``−1`` (un seul flip, lecteur
     strictement monotone car ``g_t ≥ G_MIN_STRUCT > 0``).
     ``gap_binary(t) = |ô_t − o_t|/2 ∈ {0,1}`` est le gap de l'émission (rapporté) ;
     l'observable RÉGULÉ est sa version graduée : l'écart de phase SIGNÉ
         e_t = p_read(t) − p_ref(t)      (en tokens).
     BOUCLE FERMÉE : ``g_t = regulate_step(g_{t−1}, e_t, TARGET_LEAD, η, …)`` —
     l'actionneur ``g`` influence ``p_read`` donc ``e`` (condition sine qua non).

  5. OFFSET INITIAL = TARGET (choix a priori). Le lecteur démarre avec exactement
     l'avance-cible : ``e_0 = TARGET_LEAD``. Conséquences voulues : (a) sur un
     profil SANS structure (incréments uniformes) à gain nominal, ``e ≡ target``
     ∀t ⇒ ``regulate_step`` ne corrige jamais ⇒ l'organe est EXACTEMENT le g-fixe
     (pivot dégénéré Δ=0 bit-à-bit, exigé porte 1) ; (b) ``target > 0`` (corpus
     l.751-753 : on veut A′≠A MINIMALEMENT) : la reconstruction vise une avance
     d'UN token — « proche-aligné-non-identique », jamais la copie exacte.

  6. SIGNE ET GAIN DE LA RÉTROACTION (leçon T20, vérifiée A PRIORI). ``e`` est
     CROISSANT en l'actionneur : ``de/dg = +1`` token/token (g plus grand ⇒ p_read
     avance plus vite ⇒ e monte). ``regulate_step`` applique ``g ← g − η(e−target)``
     = rétroaction négative stable sur observable croissant ✓. Échelle :
     ``|de/dg| = 1 ≈ |dρ̂/dg|`` du T15 ⇒ MÊME η = 0.5 (règle T20
     ``η ∝ 1/|d obs/d actionneur|``), hérité tel quel.

  7. BORNES ET GRILLE DU RYTHME (a priori, grammaticales). Le lecteur doit couvrir
     N tokens de phase en N tokens lus ; les rythmes requis par régime valent
     ``PHI_STAR·N/k`` et ``(1−PHI_STAR)·N/(N−k)``. Sous le prior grammatical « les
     longueurs de segments varient dans un facteur ~2 autour de l'équilongueur »,
     ces rythmes vivent dans ~[0.67, 1.5]. On gèle ``G_MIN_STRUCT = 0.5``,
     ``G_MAX_STRUCT = 2.0`` (facteur 2 symétrique autour du nominal) et la grille
     ``STRUCT_GAIN_SWEEP`` ci-dessous (l'émission autorise « une grille adaptée au
     rythme d'avance — a priori ») : la grille T15 (0.94…1.09) est calibrée sur le
     bord du chaos d'un oscillateur, pas sur un rythme de lecture.

  8. EN-TÊTES NON CANONIQUES : l'orientation n'est définie que pour DX/OUT (+1) et
     LV/IN (−1) ; toute autre combinaison lève ``ValueError`` (le profil
     d'orientation est un objet de la grammaire canonique ; dataset_aba.txt est
     100 % conforme, mesure T14).

L'ORGANE N'EST PAS TOUCHÉ : ``regulate_step`` est IMPORTÉ d'``edge_controller``
(byte-identique ``1ac0269``). Tout est déterministe (aucune source aléatoire hors
shuffles seedés). Strictement expérimental : le canon ``core/`` est intouché.
"""

import torch
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..data.aba import AbaCycle
from .edge_controller import regulate_step  # l'ORGANE, INCHANGÉ (T19/T20)


# --- constantes GELÉES A PRIORI (grammaire, jamais les données) ----------------

PHI_STAR = 2.0 / 3.0     # flip canonique après 2 segments sur 3 (prior d'équilongueur)
TARGET_LEAD = 1.0        # cible = avance d'UN token (quantum grammatical ; > 0 : A′≠A minimal)
BAND_LEAD = 0.5          # bande = ± la moitié du quantum (émission §5)
ETA_STRUCT = 0.5         # hérité T15 ; justifié |de/dg| = 1 ≈ |dρ̂/dg| (règle T20)
G0_STRUCT = 1.0          # rythme nominal (1 token de phase par token lu)
G_MIN_STRUCT = 0.5       # rythme borné : facteur 2 autour du nominal (a priori)
G_MAX_STRUCT = 2.0
# grille de rythmes fixes (baseline dure), a priori — cf. choix 7 de l'en-tête
STRUCT_GAIN_SWEEP: Tuple[float, ...] = (0.67, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50)


# --- profil d'orientation (source EXCLUSIVE : parseur aba.py) -------------------

@dataclass(frozen=True)
class OrientedToken:
    """Un token (mot) et son orientation structurelle (+1 = DX/OUT, −1 = LV/IN).

    L'orientation vient de l'EN-TÊTE DU SEGMENT du token (``AbaSegment.chirality``/
    ``direction``, parseur ``data/aba.py``) — jamais des dims 0-5 du 33D. Au
    shuffle, le tag VOYAGE avec son token (on permute des ``OrientedToken``).
    """

    text: str
    orientation: int  # +1 | −1


def _orientation_of(chirality: str, direction: str) -> int:
    if (chirality, direction) == ("DX", "OUT"):
        return +1
    if (chirality, direction) == ("LV", "IN"):
        return -1
    raise ValueError(
        f"orientation non canonique {chirality}/{direction} : le profil "
        "d'orientation n'est défini que pour DX/OUT (+1) et LV/IN (−1)"
    )


def orientation_profile(cycle: AbaCycle) -> List[OrientedToken]:
    """Profil d'orientation d'un cycle : tokens (mots) dans l'ordre, tag ±1 attaché.

    Tokenisation = ``str.split()`` du texte de chaque segment (choix 1 de
    l'en-tête : structurel, sans ``.so``). Ordre : SEG_A → SEG_B → SEG_A_PRIME.
    Pour un cycle conforme : marche à un seul flip, ``k = |A|+|B|`` tokens à +1
    puis ``|A′|`` tokens à −1 — ``k/N`` est LA variation structurelle inter-cycle.
    """
    out: List[OrientedToken] = []
    for seg in (cycle.seg_a, cycle.seg_b, cycle.seg_a_prime):
        o = _orientation_of(seg.chirality, seg.direction)
        for word in seg.text.split():
            out.append(OrientedToken(text=word, orientation=o))
    return out


def flip_fraction(orientations: Sequence[int]) -> float:
    """``n_plus / N`` : position relative du flip pour un profil canonique (informatif)."""
    n = len(orientations)
    if n == 0:
        return 0.0
    return sum(1 for o in orientations if o == +1) / n


# --- phase vraie graduée p_ref (order-preserving) -------------------------------

def phi_ref_increments(orientations: Sequence[int], *, phi_star: float = PHI_STAR) -> List[float]:
    """Incréments de la phase vraie (en TOKENS) — choix 3 de l'en-tête.

    ``inc(+1) = phi_star·N/n_plus``, ``inc(−1) = (1−phi_star)·N/n_minus`` ; les
    incréments somment à N (``p_ref(N) = N``). DÉGÉNÉRÉ (profil sans flip :
    ``n_plus == 0`` ou ``n_minus == 0``) : incrément uniforme 1.0 — aucune
    information de flip, horloge parfaite. Les COMPTES sont shuffle-invariants ;
    l'ORDRE entre par le cumul des incréments.
    """
    n = len(orientations)
    n_plus = sum(1 for o in orientations if o == +1)
    n_minus = n - n_plus
    if n_plus == 0 or n_minus == 0:
        return [1.0] * n
    inc_plus = phi_star * n / n_plus
    inc_minus = (1.0 - phi_star) * n / n_minus
    return [inc_plus if o == +1 else inc_minus for o in orientations]


# --- reconstruction monotone en boucle fermée -----------------------------------

@dataclass(frozen=True)
class ReconstructionTrace:
    """Trace complète d'une reconstruction (toutes les séries alignées par token).

    * ``p_read``     : (N+1) phase du lecteur en tokens (``p_read[0] = TARGET_LEAD``).
    * ``p_ref``      : (N+1) phase vraie graduée du profil observé (``p_ref[0] = 0``).
    * ``e``          : (N+1) écart de phase SIGNÉ ``p_read − p_ref`` (tokens) —
                       l'observable RÉGULÉ ; ``e[0] = TARGET_LEAD`` par construction.
    * ``g``          : (N,) gain (rythme d'avance) appliqué au token t.
    * ``o_hat``      : (N,) orientation reconstruite ``ô_t`` (+1 avant flip, −1 après).
    * ``gap_binary`` : (N,) ``|ô_t − o_t|/2 ∈ {0,1}`` — le gap_struct de l'émission.
    """

    p_read: List[float]
    p_ref: List[float]
    e: List[float]
    g: List[float]
    o_hat: List[int]
    gap_binary: List[float]


def reconstruct_profile(
    orientations: Sequence[int],
    *,
    eta: float = ETA_STRUCT,
    g0: float = G0_STRUCT,
    target: float = TARGET_LEAD,
    g_min: float = G_MIN_STRUCT,
    g_max: float = G_MAX_STRUCT,
    phi_star: float = PHI_STAR,
) -> ReconstructionTrace:
    """Déroule le lecteur monotone en BOUCLE FERMÉE sur un profil d'orientation.

    Par token t = 0..N−1 (ordre des opérations gelé a priori) :
      1. observe ``e_t = p_read(t) − p_ref(t)`` ;
      2. corrige ``g ← regulate_step(g, e_t, target, eta, g_min, g_max)`` (l'ORGANE,
         importé INCHANGÉ ; à ``eta=0`` : ``g`` inerte, exactement le g-fixe) ;
      3. émet ``ô_t`` (+1 si ``p_read(t) < phi_star·N``, −1 sinon) et le compare à
         ``o_t`` (``gap_binary``) ;
      4. avance ``p_read ← p_read + g`` et ``p_ref ← p_ref + inc_t``.

    À ``eta = 0`` : ``regulate_step`` retourne ``g_prev`` EXACTEMENT (IEEE754 :
    ``g − 0.0·x = g`` pour x fini ; le clip est neutre si ``g_min ≤ g0 ≤ g_max``)
    ⇒ mêmes suites d'opérations flottantes que :func:`reconstruct_fixed` ⇒ pivot
    porte 1 bit-à-bit. DÉTERMINISTE : aucune source aléatoire.
    """
    n = len(orientations)
    if n < 1:
        raise ValueError("profil vide : au moins 1 token requis")
    if not (g_min <= g0 <= g_max):
        raise ValueError("g0 doit être dans [g_min, g_max] (sinon le pivot η=0 n'est pas exact)")
    incs = phi_ref_increments(orientations, phi_star=phi_star)
    flip_at = phi_star * n   # seuil de flip du lecteur, en tokens

    p_read = target          # offset initial = l'avance-cible (choix 5)
    p_ref = 0.0
    g_cur = g0

    p_read_l: List[float] = [p_read]
    p_ref_l: List[float] = [p_ref]
    e_l: List[float] = [p_read - p_ref]
    g_l: List[float] = []
    o_hat_l: List[int] = []
    gap_bin_l: List[float] = []

    for t in range(n):
        e_t = p_read - p_ref
        g_cur = regulate_step(g_cur, e_t, target, eta, g_min, g_max)
        o_hat = +1 if p_read < flip_at else -1
        o_hat_l.append(o_hat)
        gap_bin_l.append(abs(o_hat - orientations[t]) / 2.0)
        g_l.append(g_cur)
        p_read = p_read + g_cur
        p_ref = p_ref + incs[t]
        p_read_l.append(p_read)
        p_ref_l.append(p_ref)
        e_l.append(p_read - p_ref)

    return ReconstructionTrace(
        p_read=p_read_l, p_ref=p_ref_l, e=e_l, g=g_l,
        o_hat=o_hat_l, gap_binary=gap_bin_l,
    )


def reconstruct_fixed(
    orientations: Sequence[int],
    *,
    g_fixed: float,
    target: float = TARGET_LEAD,
    phi_star: float = PHI_STAR,
) -> ReconstructionTrace:
    """Baseline : lecteur à rythme CONSTANT ``g_fixed`` (chemin de code SÉPARÉ).

    Même boucle que :func:`reconstruct_profile` mais SANS ``regulate_step`` — sert
    de baseline indépendante ET de cible du test d'équivalence exacte ``eta=0``
    (deux chemins de code, mêmes flottants — modèle ``run_fixed_gain`` T15).
    """
    n = len(orientations)
    if n < 1:
        raise ValueError("profil vide : au moins 1 token requis")
    incs = phi_ref_increments(orientations, phi_star=phi_star)
    flip_at = phi_star * n

    p_read = target
    p_ref = 0.0

    p_read_l: List[float] = [p_read]
    p_ref_l: List[float] = [p_ref]
    e_l: List[float] = [p_read - p_ref]
    g_l: List[float] = []
    o_hat_l: List[int] = []
    gap_bin_l: List[float] = []

    for t in range(n):
        o_hat = +1 if p_read < flip_at else -1
        o_hat_l.append(o_hat)
        gap_bin_l.append(abs(o_hat - orientations[t]) / 2.0)
        g_l.append(g_fixed)
        p_read = p_read + g_fixed
        p_ref = p_ref + incs[t]
        p_read_l.append(p_read)
        p_ref_l.append(p_ref)
        e_l.append(p_read - p_ref)

    return ReconstructionTrace(
        p_read=p_read_l, p_ref=p_ref_l, e=e_l, g=g_l,
        o_hat=o_hat_l, gap_binary=gap_bin_l,
    )


# --- score : fraction des tokens dans la bande [target−band, target+band] --------

def f_edge_struct(
    trace: ReconstructionTrace,
    *,
    target: float = TARGET_LEAD,
    band: float = BAND_LEAD,
) -> float:
    """``f_edge`` structurel : fraction des tokens t = 1..N avec ``|e_t − target| ≤ band``.

    ``e_0 = target`` par construction (offset initial) : trivialement en bande, donc
    EXCLU du score (a priori). Pas de coupe transitoire : les cycles réels sont
    courts (~10-15 tokens) et l'offset initial place déjà le lecteur sur la cible.
    """
    n = len(trace.e) - 1
    if n < 1:
        return 0.0
    hits = sum(1 for e_t in trace.e[1:] if abs(e_t - target) <= band)
    return hits / n


# --- observables de PRÉ-VALIDATION (porte 0) -------------------------------------

def obs_struct_frozen(tokens: Sequence[OrientedToken]) -> float:
    """PRIMAIRE, reconstruction FIGÉE : ``mean_{t=1..N} |e_t − target|`` à g fixe nominal.

    Toute dynamique est figée EN INTERNE (``eta=0`` de fait : lecteur à rythme
    constant ``G0_STRUCT``) : seul l'ORDRE des tokens fait bouger ``p_ref`` donc
    l'observable — condition d'usage d'``assert_order_sensitive`` (T23).
    """
    orientations = [tk.orientation for tk in tokens]
    tr = reconstruct_fixed(orientations, g_fixed=G0_STRUCT)
    n = len(tr.e) - 1
    return sum(abs(e_t - TARGET_LEAD) for e_t in tr.e[1:]) / n


def obs_struct_multiset(tokens: Sequence[OrientedToken]) -> float:
    """VARIANTE-CONTRÔLE multiset : ``|n_plus − PHI_STAR·N|`` (comptes SEULS, sans ordre).

    Transposition de la classe d'instrument du ``reflective_gap`` numérique de
    logos-engine (Jaccard sur multisets — revue Fable 5 §A.1) : PRÉDITE vacuous à
    la porte 0 (le shuffle préserve le multiset ⇒ gap = 0). Si elle passait la
    porte, c'est la PORTE qu'il faudrait réexaminer (prédiction falsifiable §6).
    """
    n = len(tokens)
    n_plus = sum(1 for tk in tokens if tk.orientation == +1)
    return abs(n_plus - PHI_STAR * n)


def shuffle_tokens(tokens: Sequence[OrientedToken], seed: int) -> List[OrientedToken]:
    """Permutation seedée des tokens — le tag d'orientation VOYAGE avec son token.

    ``torch.randperm`` seedé (déterministe, même mécanique que ``RealAbaDrive.shuffled``
    T21). Multiset conservé, ordre détruit.
    """
    n = len(tokens)
    if n < 2:
        return list(tokens)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return [tokens[i] for i in perm]


def shuffle_orientations(orientations: Sequence[int], seed: int) -> List[int]:
    """Permutation seedée d'un profil d'orientation nu (portes 2-3, même mécanique)."""
    n = len(orientations)
    if n < 2:
        return list(orientations)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return [orientations[i] for i in perm]
