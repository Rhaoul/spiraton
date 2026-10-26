from __future__ import annotations

"""Signature de FORME d'une trace 2D (Tour 4, oscilloscope géométrique).

À partir d'une trajectoire ``s_t ∈ R²`` (l'écran de l'oscilloscope), on mesure la
FIGURE tracée : est-ce un cercle, une spirale, ou rien d'identifiable ? Toutes les
mesures portent sur la SECONDE MOITIÉ de la trajectoire (régime établi : on laisse
passer le transitoire dû à l'impulsion / à s_0).

Procédure (déterministe, sans paramètre caché) :

1. RE-CENTRAGE sur le barycentre de la seconde moitié (un cercle/spirale n'est pas
   forcément centré sur l'origine ; on mesure la forme RELATIVE au centre du motif).
2. Passage en POLAIRE : ``r = ‖s − centre‖``, ``θ`` = angle, puis DÉROULEMENT
   (unwrap) de θ pour obtenir une phase cumulée monotone (sinon le saut ±π fausse
   les régressions). ``N_tours = Δθ_total / 2π``.
3. MÉTRIQUES DE FORME :
   * ``cv_r = σ_r / μ_r``        — coefficient de variation du rayon. CERCLE ⇒ ~0.
   * ``r2_theta_t``             — R² de ``θ`` (déroulé) vs ``t``. Rotation régulière
                                  (phase monotone) ⇒ ~1.
   * ``r2_logr_theta``          — R² de ``log r`` vs ``θ``. SPIRALE LOG ⇒ ~1.
   * ``slope_logr_theta``       — pente de ``log r`` vs ``θ``. >0 sortante, ~0 cercle.
   * ``n_turns``, ``mu_r``      — nombre de tours, rayon moyen.

4. GARDES ANTI-TRIVIAL (un « bon » score obtenu par dégénérescence est rejeté) :
   * ``mu_r > floor_frac · ‖s0‖`` : exclut le point fixe r→0 (un cercle de rayon
     nul a un CV(r) parfait mais ne trace rien). Défaut ``floor_frac = 0.1``.
   * phase monotone (``r2_theta_t > 0.98``) : exclut une « rotation » qui n'avance
     pas (oscillation sur place) ou recule.
   * pour la spirale : ``n_turns >= 2`` (Δθ ≥ 4π) : exclut une divergence en ligne
     droite déguisée en « spirale » (anti-divergence informe).

5. QUALIFICATION α-ω : on réutilise ``alpha_omega_metrics`` (formule ``cos − l2``
   INTACTE) entre s_0 et chaque point de la trace ; le ``best_return_step`` et le
   score ``cos − l2`` qualifient le RETOUR (l.290 : revient-il à un point modifié ?).
   Pont attendu : CERCLE = RÉPÉTITION (revient au même point, cos→1 & l2→0) ;
   SPIRALE = PROGRESSION (proche-aligné mais NON identique : s'éloigne, cos reste
   élevé, l2 croît).

Statistiques (Mann-Whitney U, sans dépendance externe — cohérent avec
``memory_inhibition_scan.py`` qui code Spearman à la main) : ``mann_whitney_u``
renvoie ``(U, p)`` bilatéral par approximation normale avec correction de continuité
et correction des ex-aequo. Seeds fixés en amont par l'appelant.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .alpha_omega_spatial import alpha_omega_metrics


# --- signature SPECTRALE de la matrice de transition (Tour 5, E2) -------------
#
# Le Tour 4 lit la FORME tracée à l'écran (cv_r, log r vs θ). Le Tour 5 demande
# si la même information est LISIBLE DANS LES VALEURS PROPRES de la matrice de
# transition effective ``A`` — sans dérouler la trajectoire, sans connaître le
# réglage (ω, g) ni le nom de la fabrique. Pont théorique : pour la récurrence
# ``s_{t+1} = A·s_t``, le comportement géométrique EST gouverné par le spectre.
#   * |λ| = ρ (module spectral) : ρ=1 isométrie (cercle), ρ>1 sortant (spirale),
#     ρ<1 contractant (point fixe). C'est ``gain`` dans le réglage spirale/cercle.
#   * arg λ ≠ 0 ⇔ valeurs propres COMPLEXES conjuguées ⇔ rotation (figure courbe).
#     arg λ = 0 ⇔ valeurs propres réelles ⇔ pas de rotation (ligne, nœud).
#   * is_complex : discriminant du polynôme caractéristique < 0 (paire conjuguée).
#
# La SIGNATURE est calculée à partir de la matrice DE TRANSITION seule, au même
# statut pour ``.circle`` / ``.spiral`` / ``.random`` (toutes exposent ``model.A``).
# Pour le second ordre (memory ≠ 0), ``A`` seule N'EST PAS la bonne matrice : on
# construit la matrice COMPAGNON de la récurrence à deux pas (leçon du Tour 1).


@dataclass(frozen=True)
class SpectralSignature:
    """Signature spectrale d'une matrice de transition (valeurs propres, float64).

    rho        : module spectral max ``max_i |λ_i|`` (ρ). Gain géométrique par pas.
    alpha_eig  : ``|arg λ|`` de la valeur propre dominante (module max), dans [0, π].
                 Angle de rotation par pas porté par le spectre (0 ⇒ réel ⇒ pas de
                 rotation). Pris sur la valeur propre de plus grand module (celle
                 qui domine la dynamique asymptotique).
    is_complex : True si au moins une valeur propre a une partie imaginaire non
                 négligeable (paire conjuguée ⇔ rotation). Pour une 2×2 réelle :
                 discriminant ``(tr)² − 4·det < 0``.
    """

    rho: float
    alpha_eig: float
    is_complex: bool


def companion_matrix(A: torch.Tensor, memory: float) -> torch.Tensor:
    """Matrice compagnon de la récurrence du second ordre ``s_{t+1}=A·s_t−c·s_{t−1}``.

    L'état augmenté ``z_t = [s_t ; s_{t−1}]`` (dimension ``2d``) évolue linéairement
    par ``z_{t+1} = M · z_t`` avec

        M = [[ A,   −c·I ],
             [ I,    0   ]]

    dont les valeurs propres gouvernent EXACTEMENT la dynamique à deux pas. C'est
    cette matrice — PAS ``A`` seule — qu'il faut diagonaliser quand ``memory ≠ 0``
    (leçon du Tour 1 : lire ``A`` seule sous-estime/ignore le mode de mémoire).
    Quand ``memory = 0``, ``M`` se réduit au bloc ``A`` (plus un bloc nilpotent
    découplé à valeurs propres nulles) : on renvoie alors ``A`` directement, dont
    le spectre est exactement celui qui compte.

    A      : matrice de transition ``(d, d)``.
    memory : coefficient ``c`` du terme ``−c·s_{t−1}``.
    """
    if A.dim() != 2 or A.size(0) != A.size(1):
        raise ValueError("A doit être carrée (d, d)")
    if memory == 0.0:
        return A
    d = A.size(0)
    Ad = A.to(torch.float64)
    I = torch.eye(d, dtype=torch.float64)
    Z = torch.zeros(d, d, dtype=torch.float64)
    top = torch.cat([Ad, -float(memory) * I], dim=1)
    bot = torch.cat([I, Z], dim=1)
    return torch.cat([top, bot], dim=0)


def spectral_signature(transition: torch.Tensor, *, memory: float = 0.0) -> SpectralSignature:
    """Calcule la signature spectrale d'une matrice de transition (float64).

    transition : la matrice ``A`` de la récurrence ``s_{t+1} = A·s_t (− c·s_{t−1})``.
                 Pour ``Oscilloscope2D`` c'est le buffer ``model.A`` — lu au MÊME
                 statut pour ``.circle`` / ``.spiral`` / ``.random`` (jamais (ω, g)).
    memory     : coefficient ``c`` du second ordre. Si ≠ 0, on diagonalise la
                 matrice COMPAGNON ``M`` (pas ``A``) — voir :func:`companion_matrix`.

    Retourne ``SpectralSignature(rho, alpha_eig, is_complex)`` :
      * rho       = ``max_i |λ_i|`` (module spectral max),
      * alpha_eig = ``|arg λ*|`` où ``λ*`` est la valeur propre de module max,
      * is_complex= au moins une λ a une partie imaginaire non négligeable.

    Calcul en float64 via ``torch.linalg.eigvals`` (déterministe, exact en petite
    dimension). Aucun seuil de réglage, aucune étiquette de forme n'intervient.
    """
    M = companion_matrix(transition, memory).to(torch.float64)
    ev = torch.linalg.eigvals(M)  # valeurs propres complexes (float64 → complex128)

    mods = ev.abs()
    rho = float(mods.max().item())

    # valeur propre dominante = celle de plus grand module (dynamique asymptotique).
    dom_idx = int(torch.argmax(mods).item())
    dom = ev[dom_idx]
    alpha_eig = abs(float(torch.angle(dom).item()))

    # is_complex : une λ a une partie imaginaire non négligeable. Tolérance
    # relative à l'échelle de la matrice pour rester robuste au bruit numérique.
    scale = float(M.abs().max().item())
    imag_tol = 1e-9 * max(scale, 1.0)
    is_complex = bool((ev.imag.abs() > imag_tol).any().item())

    return SpectralSignature(rho=rho, alpha_eig=alpha_eig, is_complex=is_complex)


# --- statistique : Mann-Whitney U (implémentation autonome) ------------------

def _normal_cdf(z: float) -> float:
    """CDF de la loi normale standard via la fonction d'erreur."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Test U de Mann-Whitney bilatéral. Retourne ``(U, p)``.

    ``U`` est la statistique du groupe ``a``. ``p`` est calculé par approximation
    normale (n suffisamment grand, ≥ ~20 par groupe ici) avec correction de
    continuité et correction de variance pour les rangs ex-aequo. Sans dépendance
    scipy (le dépôt code ses propres stats, cf. memory_inhibition_scan).
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        raise ValueError("groupes non vides requis")

    combined = [(v, 0) for v in a] + [(v, 1) for v in b]
    combined.sort(key=lambda t: t[0])

    # rangs moyens (gestion des ex-aequo)
    ranks = [0.0] * len(combined)
    i = 0
    tie_correction = 0.0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # rangs 1-indexés
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        tcount = j - i + 1
        if tcount > 1:
            tie_correction += tcount ** 3 - tcount
        i = j + 1

    r1 = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    n = n1 + n2
    mu = n1 * n2 / 2.0
    sigma_sq = (n1 * n2 / 12.0) * ((n + 1) - tie_correction / (n * (n - 1)))
    if sigma_sq <= 0:
        return u1, 1.0
    sigma = math.sqrt(sigma_sq)
    z = (u - mu + 0.5) / sigma  # correction de continuité (u <= mu)
    p = 2.0 * _normal_cdf(z)
    p = min(1.0, max(0.0, p))
    return u1, p


# --- régressions linéaires simples (R², pente) -------------------------------

def _linregress(x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
    """Régression OLS ``y ~ a·x + b``. Retourne ``(pente, R²)``.

    R² = 1 − SS_res / SS_tot. Si la variance de y est nulle, R² = 0 (aucune
    structure linéaire à expliquer) et pente = 0.
    """
    x = x.to(torch.float64)
    y = y.to(torch.float64)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    sxy = ((x - xm) * (y - ym)).sum()
    syy = ((y - ym) ** 2).sum()
    if float(sxx) == 0.0 or float(syy) == 0.0:
        return 0.0, 0.0
    slope = float(sxy / sxx)
    ss_res = float(syy - sxy * sxy / sxx)
    r2 = 1.0 - ss_res / float(syy)
    return slope, r2


def _unwrap(theta: torch.Tensor) -> torch.Tensor:
    """Déroulement de phase : supprime les sauts de ±2π pour une phase cumulée."""
    th = theta.to(torch.float64).clone()
    out = th.clone()
    cum = 0.0
    for i in range(1, len(th)):
        d = float(th[i] - th[i - 1])
        while d > math.pi:
            d -= 2 * math.pi
        while d < -math.pi:
            d += 2 * math.pi
        cum += d
        out[i] = float(th[0]) + cum
    return out


# --- signature de forme ------------------------------------------------------

@dataclass(frozen=True)
class ShapeSignature:
    cv_r: float                 # σ_r / μ_r (cercle ⇒ ~0)
    r2_theta_t: float           # R²(θ déroulé vs t) (rotation régulière ⇒ ~1)
    r2_logr_theta: float        # R²(log r vs θ) (spirale log ⇒ ~1)
    slope_logr_theta: float     # pente log r vs θ (>0 sortante)
    n_turns: float              # |Δθ_total| / 2π
    mu_r: float                 # rayon moyen (seconde moitié)
    # qualification α-ω (cos − l2)
    ao_score: float             # max_t (cos − l2) entre s0 et s_t
    ao_best_return_step: int
    ao_cos_final: float         # cos(s0, s_final)
    ao_l2_final: float          # ‖s_final − s0‖ / ‖s0‖
    # gardes anti-trivial
    passes_radius_floor: bool   # μ_r > floor_frac · ‖s0‖
    passes_phase_monotone: bool # r2_theta_t > monotone_thresh

    def passes_circle_guards(self) -> bool:
        """Gardes communes au verdict cercle (rayon non nul + phase monotone)."""
        return self.passes_radius_floor and self.passes_phase_monotone

    def passes_spiral_guards(self, *, min_turns: float = 2.0) -> bool:
        """Gardes du verdict spirale (rayon non nul + phase monotone + ≥ N tours)."""
        return (
            self.passes_radius_floor
            and self.passes_phase_monotone
            and self.n_turns >= min_turns
        )


def shape_signature(
    trace: torch.Tensor,
    s0: torch.Tensor,
    *,
    floor_frac: float = 0.1,
    monotone_thresh: float = 0.98,
) -> ShapeSignature:
    """Calcule la signature de forme d'une trace 2D ``(T+1, 2)``.

    trace : positions ``s_0 … s_T`` (sortie de ``Oscilloscope2D.trace``).
    s0    : état initial (forme ``(2,)``) — sert au plancher de rayon et à α-ω.
    floor_frac : fraction de ‖s0‖ sous laquelle μ_r est jugé trivial (r→0).
    monotone_thresh : seuil de R²(θ,t) au-delà duquel la phase est dite monotone.

    Mesures de forme SUR LA SECONDE MOITIÉ ; α-ω sur la trace complète.
    """
    if trace.dim() != 2 or trace.size(-1) != 2:
        raise ValueError("trace doit être (T+1, 2)")
    T = trace.size(0)
    if T < 4:
        raise ValueError("trace trop courte (>= 4 points requis)")

    half = trace[T // 2:]  # régime établi

    centre = half.mean(dim=0)
    rel = half - centre
    r = torch.linalg.vector_norm(rel, dim=-1)
    theta = torch.atan2(rel[:, 1], rel[:, 0])
    theta_u = _unwrap(theta)

    mu_r = float(r.mean())
    sigma_r = float(r.std(unbiased=False))
    cv_r = sigma_r / mu_r if mu_r > 0 else float("inf")

    t_idx = torch.arange(half.size(0), dtype=torch.float64)
    _, r2_theta_t = _linregress(t_idx, theta_u)

    # log r vs θ (déroulé) : structure de spirale logarithmique
    eps = 1e-12
    logr = torch.log(r.to(torch.float64).clamp_min(eps))
    slope_lt, r2_lt = _linregress(theta_u, logr)

    n_turns = float(abs(theta_u[-1] - theta_u[0]) / (2 * math.pi))

    # qualification α-ω sur la trace complète (cos − l2 entre s0 et s_t)
    s0b = s0.reshape(1, 1, 1, 2)
    cos_series: List[float] = []
    l2_series: List[float] = []
    for t in range(T):
        xt = trace[t].reshape(1, 1, 1, 2)
        l2, cos = alpha_omega_metrics(s0b, xt)
        l2_series.append(float(l2.mean()))
        cos_series.append(float(cos.mean()))
    signal = [c - l for c, l in zip(cos_series, l2_series)]
    # best_return_step EXCLUT t=0 : à t=0, cos=1 & l2=0 trivialement (le point de
    # départ est toujours le « meilleur » au sens cos−l2, mais ce n'est pas un
    # RETOUR). Le diagnostic central cherche un retour APRÈS le transitoire — le
    # point où la trajectoire est revenue le plus proche-et-aligné (l.290), donc
    # parmi t >= 1. (Si T==1, on retombe sur t=0 faute d'autre choix.)
    search = range(1, len(signal)) if len(signal) > 1 else range(len(signal))
    best_step = max(search, key=lambda i: signal[i])
    ao_score = signal[best_step]

    s0_norm = float(torch.linalg.vector_norm(s0))
    passes_radius_floor = mu_r > floor_frac * s0_norm
    passes_phase_monotone = r2_theta_t > monotone_thresh

    return ShapeSignature(
        cv_r=cv_r,
        r2_theta_t=r2_theta_t,
        r2_logr_theta=r2_lt,
        slope_logr_theta=slope_lt,
        n_turns=n_turns,
        mu_r=mu_r,
        ao_score=ao_score,
        ao_best_return_step=best_step,
        ao_cos_final=cos_series[-1],
        ao_l2_final=l2_series[-1],
        passes_radius_floor=passes_radius_floor,
        passes_phase_monotone=passes_phase_monotone,
    )


# --- signature de CORNERNESS (Tour 6 — ADDITIF, ne touche rien ci-dessus) -----
#
# Le Tour 4 lit la COURBURE GLOBALE (cv_r, log r vs θ : cercle vs spirale). Le Tour 6
# demande si la trajectoire est ANGULEUSE par MORCEAUX : arêtes DROITES (courbure ≈ 0
# sur plusieurs pas) séparées par des COINS FRANCS (un seul pas de forte courbure). Une
# orbite lisse (cercle, spirale, ellipse) a une courbure UNIFORME : elle ne peut pas
# avoir ce contraste arête/coin. La cornerness mesure exactement ce contraste.
#
# κ_t = ANGLE DE BRAQUAGE au pas t = angle(s_{t+1} − s_t, s_t − s_{t−1}) ∈ [0, π] :
# l'angle entre deux pas successifs. |κ| ≈ 0 ⇒ on continue tout droit (arête) ;
# |κ| grand ⇒ on braque (coin). Défini pour t = 1 … T−2 (besoin de s_{t−1} et s_{t+1}).
#
# Toutes les mesures de Tour 6 portent sur ce profil de braquage et sur le rapport au
# centre du motif — AUCUNE n'utilise (ω, g), un nom de fabrique, ou une étiquette de forme.


@dataclass(frozen=True)
class CornerSignature:
    """Signature ANGULEUSE d'une trace 2D : arêtes droites + coins francs vs orbite lisse.

    cornerness          : ``C = (P95|κ| − médiane|κ|) / médiane|κ|`` (contraste coin/arête).
                          Orbite LISSE (courbure uniforme) ⇒ P95 ≈ médiane ⇒ C ≈ 0.
                          ANGULEUX VRAI (arêtes droites + coins) ⇒ P95 >> médiane ⇒ C grand.
                          Médiane plancher à ``kappa_floor`` quand la médiane est ~0
                          (cas DÉGÉNÉRÉ : trace soit parfaitement droite — P95 aussi ~0,
                          C→0 — soit tout-coin sans arête courbe — P95 grand). Dans ce cas
                          C est SATURÉ à ``cornerness_max`` (borne finie, pas un nombre sale)
                          et ``degenerate_median=True`` le signale : la cornerness n'est alors
                          PAS la métrique d'intérêt (lire straight_edge_fraction / coins).
    degenerate_median   : True si médiane|κ| < kappa_floor (cornerness saturée — voir ci-dessus).
    straight_edge_fraction : fraction des pas où ``|κ| < straight_thresh`` (arête droite).
                          ≠ n-gone-DE-SOMMETS, où CHAQUE pas est un coin (fraction ≈ 0).
    n_corners           : nombre de pas où ``|κ| > corner_thresh`` (coins francs).
    min_center_ratio    : ``min_t ‖s_t − centre‖ / μ_r`` — la trace passe-t-elle PRÈS du
                          centre du motif ? (croix ⇒ proche de 0 ; orbite convexe ⇒ ~1).
    n_center_passes     : nombre de passages ``‖s_t − centre‖ < center_frac · μ_r`` (croix
                          ⇒ ≥ 2 : la trace traverse le centre plusieurs fois).
    self_intersections  : nombre de paires de segments NON adjacents qui se croisent
                          (croix/figure recroisée ⇒ ≥ 1 ; orbite convexe simple ⇒ 0).
    n_corners_per_turn  : n_corners normalisé par le nb de tours (densité de coins).
    median_kappa, p95_kappa : médiane et 95e pct de |κ| (pour lecture brute).
    """

    cornerness: float
    straight_edge_fraction: float
    n_corners: int
    min_center_ratio: float
    n_center_passes: int
    self_intersections: int
    n_corners_per_turn: float
    median_kappa: float
    p95_kappa: float
    degenerate_median: bool


def _segments_cross(p1, p2, p3, p4) -> bool:
    """Les segments [p1,p2] et [p3,p4] se croisent-ils (intersection propre) ?

    Test d'orientation standard (produit en croix 2D). Croisement PROPRE : les
    extrémités d'un segment sont de part et d'autre de l'autre, strictement (on exclut
    les contacts par extrémité partagée — gérés en amont en sautant les segments adjacents).
    """
    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _count_self_intersections(pts: torch.Tensor) -> int:
    """Nombre de croisements entre segments NON adjacents d'une polyligne ``(N, 2)``.

    On saute les paires de segments adjacents (partageant un sommet) : leur « croisement »
    par extrémité n'est pas une self-intersection. O(N²) — acceptable aux longueurs de
    diagnostic. Déterministe, aucun seuil de réglage.
    """
    p = pts.tolist()
    n = len(p) - 1  # nombre de segments
    count = 0
    for i in range(n):
        a1, a2 = p[i], p[i + 1]
        for j in range(i + 2, n):
            # segments i et j ; on saute aussi le cas où j est le dernier et i le premier
            # s'ils partagent un sommet (boucle fermée) — ici trace ouverte, pas de wrap.
            if j == i + 1:
                continue
            b1, b2 = p[j], p[j + 1]
            if _segments_cross(a1, a2, b1, b2):
                count += 1
    return count


def corner_signature(
    trace: torch.Tensor,
    *,
    straight_thresh: float = 0.05,
    corner_thresh: float = 1.0,
    center_frac: float = 0.2,
    kappa_floor: float = 1e-3,
    cornerness_max: float = 100.0,
) -> CornerSignature:
    """Calcule la signature ANGULEUSE d'une trace 2D ``(T+1, 2)`` — ADDITIF (Tour 6).

    trace : positions ``s_0 … s_T`` (sortie de ``Polyscope2D.trace`` ou ``Oscilloscope2D.trace``).
    straight_thresh : seuil ``|κ| <`` pour qualifier un pas d'ARÊTE droite (défaut 0.05 rad).
    corner_thresh   : seuil ``|κ| >`` pour qualifier un pas de COIN franc (défaut 1.0 rad).
    center_frac     : fraction de ``μ_r`` sous laquelle un point est « près du centre ».
    kappa_floor     : seuil sous lequel la médiane de |κ| est jugée DÉGÉNÉRÉE (~0). En-deçà,
                      la cornerness est saturée à ``cornerness_max`` (si P95 > floor) ou 0
                      (si P95 aussi ~0 : trace droite), et ``degenerate_median=True``.
    cornerness_max  : borne finie de la cornerness en régime dégénéré (évite un nombre
                      arbitraire dépendant du floor ; la cornerness reste interprétable).

    Mesures sur la SECONDE MOITIÉ de la trace (régime établi, comme ``shape_signature`` :
    on laisse passer le transitoire). Les croisements et le rapport-centre se calculent
    sur cette même seconde moitié. Aucun (ω, g) ni nom de fabrique n'intervient.
    """
    if trace.dim() != 2 or trace.size(-1) != 2:
        raise ValueError("trace doit être (T+1, 2)")
    T = trace.size(0)
    if T < 6:
        raise ValueError("trace trop courte (>= 6 points requis pour le braquage)")

    half = trace[T // 2:].to(torch.float64)  # régime établi
    N = half.size(0)

    # --- angles de braquage κ_t = angle(s_{t+1}-s_t, s_t-s_{t-1}) ---
    diffs = half[1:] - half[:-1]                  # (N-1, 2) : pas successifs
    norms = torch.linalg.vector_norm(diffs, dim=-1)
    kappas: List[float] = []
    for i in range(diffs.size(0) - 1):
        v0 = diffs[i]
        v1 = diffs[i + 1]
        n0 = float(norms[i])
        n1 = float(norms[i + 1])
        if n0 < 1e-12 or n1 < 1e-12:
            # pas immobile : pas de direction définie ⇒ braquage nul (continue tout droit).
            kappas.append(0.0)
            continue
        cos = float((v0 @ v1) / (n0 * n1))
        cos = max(-1.0, min(1.0, cos))
        kappas.append(math.acos(cos))  # ∈ [0, π]

    if not kappas:
        # trace dégénérée : pas assez de pas pour un braquage.
        return CornerSignature(
            cornerness=0.0, straight_edge_fraction=0.0, n_corners=0,
            min_center_ratio=float("nan"), n_center_passes=0, self_intersections=0,
            n_corners_per_turn=0.0, median_kappa=0.0, p95_kappa=0.0,
            degenerate_median=True,
        )

    abs_k = sorted(kappas)
    median_kappa = abs_k[len(abs_k) // 2]
    idx95 = min(len(abs_k) - 1, int(math.ceil(0.95 * len(abs_k))) - 1)
    p95_kappa = abs_k[max(0, idx95)]

    # Cornerness = (P95 − médiane) / médiane. Quand la médiane est ~0 (régime DÉGÉNÉRÉ :
    # trace soit parfaitement droite, soit tout-coin sans arête courbe), on ne divise PAS
    # par le floor (cela donnerait un nombre arbitraire ∝ 1/floor). On SATURE :
    #   * P95 ~0 aussi (trace droite, commuting)           ⇒ C = 0 ;
    #   * P95 grand (tout-coin sans arête : cross, ngon)    ⇒ C = cornerness_max (borné).
    # Le flag degenerate_median=True signale ce cas : la cornerness n'y est PAS la métrique
    # d'intérêt (on lit alors straight_edge_fraction et les coins/passages-centre).
    degenerate_median = median_kappa < kappa_floor
    if degenerate_median:
        cornerness = 0.0 if p95_kappa < kappa_floor else cornerness_max
    else:
        cornerness = (p95_kappa - median_kappa) / median_kappa
        if cornerness > cornerness_max:
            cornerness = cornerness_max

    n_straight = sum(1 for k in kappas if k < straight_thresh)
    straight_edge_fraction = n_straight / len(kappas)
    n_corners = sum(1 for k in kappas if k > corner_thresh)

    # --- rapport au centre du motif (sur la seconde moitié) ---
    centre = half.mean(dim=0)
    rel = half - centre
    r = torch.linalg.vector_norm(rel, dim=-1)
    mu_r = float(r.mean())
    if mu_r > 0:
        min_center_ratio = float(r.min()) / mu_r
        n_center_passes = int((r < center_frac * mu_r).sum().item())
    else:
        min_center_ratio = float("nan")
        n_center_passes = 0

    # --- croisements (self-intersections) sur la seconde moitié ---
    self_intersections = _count_self_intersections(half)

    # --- densité de coins par tour (réutilise l'unwrap de θ pour n_turns) ---
    theta = torch.atan2(rel[:, 1], rel[:, 0])
    theta_u = _unwrap(theta)
    n_turns = float(abs(theta_u[-1] - theta_u[0]) / (2 * math.pi))
    n_corners_per_turn = n_corners / n_turns if n_turns > 1e-6 else float(n_corners)

    return CornerSignature(
        cornerness=cornerness,
        straight_edge_fraction=straight_edge_fraction,
        n_corners=n_corners,
        min_center_ratio=min_center_ratio,
        n_center_passes=n_center_passes,
        self_intersections=self_intersections,
        n_corners_per_turn=n_corners_per_turn,
        median_kappa=median_kappa,
        p95_kappa=p95_kappa,
        degenerate_median=degenerate_median,
    )
