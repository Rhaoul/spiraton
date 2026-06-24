from __future__ import annotations

"""Balayage du terme de mémoire ``−C(s_{t−1})`` du second ordre (axiome §3.2, Tour 2).

PORTEUR THÉORIQUE. Dans l'équation fondatrice du Logos

    s_{t+1} = D( A(s_t) + B(s_t²) − C(s_{t−1}) ) + L(s_t)        (THEORIE §3.2 l.136)

le terme ``−C(s_{t−1})`` est l'INHIBITION DE LA DURÉE : l'état précédent retranché
de la mise à jour courante. Geste SUB / lévogyre / in (le retour qui retient — corpus
l.290-291 « la boucle ne revient jamais au même point, elle revient à un point
modifié » ; l.605, 630-632). Ce module fait varier ce seul terme — magnitude,
structure, signe — toutes choses égales par ailleurs (A, B, D, L, s0 partagés à
graine fixée), et MESURE l'effet sur la stabilité du déroulé et sur le retour α-ω.

HYPOTHÈSE FALSIFIABLE (linguiste, Tour 2). Faire croître ``γ = ‖C‖`` STABILISE la
trajectoire : taux de ``diverged`` et ``max_norm`` décroissants de façon monotone
(Spearman ρ_s < −0.6). Secondaire : le retour ``cos−l2`` est NON-MONOTONE EN CLOCHE
en γ (maximum à γ intermédiaire = retour proche-aligné-non-identique, l.290).

RÉSERVE D'HONNÊTETÉ (centrale). « ‖C‖ grand → norme bornée » peut être en partie
TRIVIAL : retrancher davantage = amplitude plus petite. C'est pourquoi CTRL-RAND
(C aléatoire de MÊME norme, regénéré par graine) est obligatoire : il sépare
l'inhibition *structurée* de la simple atténuation scalaire. Si seul le scalaire
survit (CTRL-RAND ≈ C-structuré), ce n'est pas une réfutation de l'axiome 3 mais
une DÉLIMITATION de ce que « inhibition » veut dire — à consigner tel quel.

MÉTHODE (REFUS). Mesure, pas cible. Baselines : CTRL-0 (C=0, markovien) et
CTRL-RAND (même norme). Contrôle de signe ``+C``. Anti-artefact : balayage en
``bounded=False`` (sinon le tanh terminal confond l'effet de C avec le clamp).
Garde-fou point fixe trivial : un best_return atteint parce que ‖s_t‖→0 est exclu
(plancher de norme). Seeds fixés partout, aucun boost en dur, aucune lecture
d'étiquette. On NE TOUCHE PAS ``chrono.py`` : C est réassigné sous ``no_grad``
DANS ce diagnostic, le défaut de la cellule reste exact.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..experimental.chrono import ChronoSpiraton
from .alpha_omega_spatial import alpha_omega_metrics


# ---------------------------------------------------------------------------
# Statistique de rang déterministe (Spearman) — pas de dépendance externe.
# ---------------------------------------------------------------------------

def _rankdata(x: Sequence[float]) -> List[float]:
    """Rangs moyens (gestion des ex-aequo par rang moyen), déterministe."""
    n = len(x)
    order = sorted(range(n), key=lambda i: x[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        # rangs 1-based moyens sur le bloc d'ex-aequo [i, j]
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Corrélation de rang de Spearman (Pearson sur les rangs), déterministe.

    Implémentée à la main pour ne dépendre d'aucun paquet et rester reproductible.
    Cross-vérifiée contre scipy.stats.spearmanr dans les tests.
    """
    if len(x) != len(y):
        raise ValueError("x et y de longueurs différentes")
    n = len(x)
    if n < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


def _betainc_reg(a: float, b: float, x: float) -> float:
    """Beta incomplète régularisée I_x(a,b), série/fraction continue de Lentz.

    Implémentation déterministe et autonome (pas de dépendance externe), pour la
    p-value de Student quand scipy n'est pas disponible. Précision largement
    suffisante pour un seuil à 0.05 sur n petit.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # log B(a,b) via lgamma.
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta)

    def _cf(a: float, b: float, x: float) -> float:
        # fraction continue de Lentz pour I_x(a,b)/front-factor.
        tiny = 1e-30
        c = 1.0
        d = 1.0 - (a + b) * x / (a + 1.0)
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            # pas pair
            aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            # pas impair
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
            d = 1.0 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1.0 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-12:
                break
        return h

    # symétrie pour convergence.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _cf(a, b, x) / a
    return 1.0 - front * _cf(b, a, 1.0 - x) / b


def spearman_t_pvalue(rho: float, n: int) -> float:
    """p-value bilatérale approchée via la statistique t = ρ·√((n−2)/(1−ρ²)).

    Approximation standard (loi de Student à n−2 ddl), valable pour n modéré ;
    on la documente comme APPROXIMATION. p calculée via la Beta incomplète
    régularisée (autonome, sans dépendance), avec la relation
    ``P(|T|>|t|) = I_{df/(df+t²)}(df/2, 1/2)``. Cross-vérifiée contre scipy.
    """
    if not (n > 2) or rho != rho:  # rho!=rho ⇒ NaN
        return float("nan")
    if abs(rho) >= 1.0:
        return 0.0
    df = n - 2
    t = rho * ((df / (1.0 - rho * rho)) ** 0.5)
    x = df / (df + t * t)
    return _betainc_reg(df / 2.0, 0.5, x)


# ---------------------------------------------------------------------------
# Échelle de γ : rayon spectral ρ(A).
# ---------------------------------------------------------------------------

@torch.no_grad()
def spectral_radius(W: torch.Tensor) -> float:
    """ρ(W) = max |valeur propre| d'une matrice carrée.

    Calcul direct par décomposition spectrale (``torch.linalg.eigvals``),
    déterministe et exact en petite dimension — préféré à l'itération de
    puissance (qui requiert un vecteur initial seedé et peut manquer un mode).
    Sert d'unité naturelle pour γ : ``γ = c · ρ(A)`` cale la magnitude de C
    sur celle de l'opérateur markovien A qu'il vient inhiber.
    """
    ev = torch.linalg.eigvals(W.to(torch.double))
    return float(ev.abs().max().item())


# ---------------------------------------------------------------------------
# Constructeurs de la matrice C à norme contrôlée et structure choisie.
# ---------------------------------------------------------------------------

_STRUCTURES = ("diag+", "diag±", "dense", "anti-sym")


@torch.no_grad()
def _make_C(
    d: int,
    structure: str,
    gen: torch.Generator,
    *,
    sign: float = 1.0,
) -> torch.Tensor:
    """Construit une matrice C ``d×d`` de structure donnée, de norme UNITAIRE
    (Frobenius), à multiplier ensuite par la magnitude voulue.

    - ``diag+``   : diagonale positive (inhibition coordonnée par coordonnée, ≥0).
    - ``diag±``   : diagonale de signes mixtes.
    - ``dense``   : matrice pleine gaussienne (mélange les coordonnées).
    - ``anti-sym``: M − Mᵀ (rotation pure, valeurs propres imaginaires).
    - ``rand``    : alias de ``dense`` (utilisé par CTRL-RAND).

    ``sign`` permet le contrôle ``+C`` (sign=-1 inverse la convention, voir note
    dans le scan). La normalisation Frobenius garantit que CTRL-RAND a EXACTEMENT
    la même norme que la structure testée — c'est l'anti-artefact central.
    """
    if structure in ("dense", "rand"):
        M = torch.randn(d, d, generator=gen)
    elif structure == "diag+":
        M = torch.diag(torch.rand(d, generator=gen) + 0.1)  # >0 strict
    elif structure == "diag±":
        v = torch.rand(d, generator=gen) + 0.1
        s = torch.where(torch.rand(d, generator=gen) > 0.5, 1.0, -1.0)
        M = torch.diag(v * s)
    elif structure == "anti-sym":
        R = torch.randn(d, d, generator=gen)
        M = R - R.t()
    else:
        raise ValueError(f"structure inconnue: {structure}")

    fro = float(torch.linalg.norm(M).item())
    if fro > 0:
        M = M / fro
    return sign * M


# ---------------------------------------------------------------------------
# Mesure du retour α-ω « cos − l2 » sur une trace ChronoSpiraton.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _alpha_omega_on_trace(
    s0: torch.Tensor,
    trace: List[torch.Tensor],
    *,
    norm_floor_ratio: float = 1e-2,
) -> Dict[str, float]:
    """Calcule la série ``cos−l2`` (retour vers s0) sur une trace, à la main.

    Reprend EXACTEMENT la formule de :func:`alpha_omega_metrics`
    (l2 normalisée, cosine), puis ``signal = cos − l2``. NE PASSE PAS par
    ``run_alpha_omega_spatial`` (qui exige une grille B,H,W,C).

    best_return = max du signal sur t≥1 (on exclut t=0 : s0 vs s0 ⇒ retour trivial),
    EN EXCLUANT les pas où l'état s'est effondré vers 0 : si ‖s_t‖ < ratio·‖s0‖,
    le cosine est mal défini / le « retour » est un point fixe trivial, pas un
    retour proche-aligné. Ces pas sont marqués invalides.

    Retourne best_return (score), best_step, et flags de validité.
    """
    s0b = s0 if s0.dim() == 2 else s0.unsqueeze(0)
    base_norm = float(s0b.norm(dim=-1).mean().item())
    floor = norm_floor_ratio * base_norm

    best_score = float("-inf")
    best_step = -1
    any_valid = False
    diverged = False
    for t, st in enumerate(trace, start=1):
        stb = st if st.dim() == 2 else st.unsqueeze(0)
        if not torch.isfinite(stb).all():
            diverged = True
            break
        cur_norm = float(stb.norm(dim=-1).mean().item())
        l2, cos = alpha_omega_metrics(s0b, stb)
        signal = float((cos.mean() - l2.mean()).item())
        if cur_norm < floor:
            # point fixe trivial : retour non comptabilisé.
            continue
        any_valid = True
        if signal > best_score:
            best_score = signal
            best_step = t

    return {
        "best_return": best_score if any_valid else float("nan"),
        "best_step": best_step,
        "valid": any_valid,
        "diverged": diverged,
    }


# ---------------------------------------------------------------------------
# Rapport agrégé.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GammaCell:
    """Résultat agrégé pour une (condition, γ) sur toutes les graines."""
    condition: str          # nom de structure ou de baseline
    gamma_ratio: float      # multiplicateur c tel que ‖C‖ = c·ρ(A) (médiane inter-graines)
    n_seeds: int
    diverged_rate: float    # fraction de graines divergentes (bounded=False)
    median_max_norm: float
    median_final_norm: float
    median_best_return: float   # médiane du score cos−l2 (sur graines valides)
    valid_return_rate: float    # fraction de graines avec un best_return valide


@dataclass(frozen=True)
class MemoryInhibitionReport:
    d: int
    steps: int
    seeds: Tuple[int, ...]
    bounded: bool
    c_outside: bool       # position de l'inhibition (Tour 3) : True = −C hors de D.
    init_scale: float
    gamma_ratios: Tuple[float, ...]
    structures: Tuple[str, ...]
    cells: List[GammaCell]
    # Statistiques de monotonie (sur la structure de référence = première structure).
    ref_structure: str
    spearman_gamma_diverged: float
    spearman_pvalue: float
    diverged_drop_pts: float        # (taux à γ=0) − (taux à γ max), en points de %
    bell_argmax_gamma: float        # γ au max de best_return (cloche)
    bell_is_interior: bool          # le max est-il à un γ intérieur (ni 0 ni max) ?

    def _cells_for(self, cond: str) -> List[GammaCell]:
        return [c for c in self.cells if c.condition == cond]

    def summary(self) -> str:
        lines: List[str] = []
        lines.append("=" * 78)
        pos = "EXTÉRIEUR (−C hors de D)" if self.c_outside else "INTÉRIEUR (−C dans D, canon)"
        lines.append(
            f"[memory_inhibition] d={self.d} steps={self.steps} "
            f"bounded={self.bounded} init_scale={self.init_scale} n_seeds={len(self.seeds)}"
        )
        lines.append(f"  position C = {pos}")
        lines.append("=" * 78)
        conds = []
        for c in self.cells:
            if c.condition not in conds:
                conds.append(c.condition)
        for cond in conds:
            lines.append(f"\n--- condition: {cond} ---")
            lines.append(
                f"{'c·ρ(A)':>8} | {'div_rate':>8} | {'med_maxN':>12} | "
                f"{'med_finN':>12} | {'med_return':>10} | {'valid':>6}"
            )
            for cell in self._cells_for(cond):
                mn = ("%.4g" % cell.median_max_norm) if cell.median_max_norm == cell.median_max_norm else "nan"
                fn = ("%.4g" % cell.median_final_norm) if cell.median_final_norm == cell.median_final_norm else "nan"
                br = ("%.4f" % cell.median_best_return) if cell.median_best_return == cell.median_best_return else "nan"
                lines.append(
                    f"{cell.gamma_ratio:8.3g} | {cell.diverged_rate:8.3f} | "
                    f"{mn:>12} | {fn:>12} | {br:>10} | {cell.valid_return_rate:6.2f}"
                )
        lines.append("\n--- monotonie / cloche (structure de réf = %s) ---" % self.ref_structure)
        lines.append(
            f"  Spearman ρ_s(γ → taux diverged) = {self.spearman_gamma_diverged:.4f}  "
            f"(p≈{self.spearman_pvalue:.4g})"
        )
        lines.append(
            f"  chute taux diverged (γ=0 → γ max) = {self.diverged_drop_pts:.1f} points"
        )
        lines.append(
            f"  best_return max à γ={self.bell_argmax_gamma:.3g}·ρ(A)  "
            f"(intérieur={'oui' if self.bell_is_interior else 'non'})"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Diagnostic principal.
# ---------------------------------------------------------------------------

def _median(xs: Sequence[float]) -> float:
    f = [x for x in xs if x == x and x not in (float("inf"), float("-inf"))]
    if not f:
        return float("nan")
    f = sorted(f)
    n = len(f)
    mid = n // 2
    if n % 2 == 1:
        return f[mid]
    return 0.5 * (f[mid - 1] + f[mid])


@torch.no_grad()
def run_memory_inhibition_scan(
    *,
    d: int = 8,
    steps: int = 50,
    seeds: Sequence[int] = tuple(range(24)),
    gamma_ratios: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    structures: Sequence[str] = ("dense", "diag+", "diag±", "anti-sym"),
    bounded: bool = False,
    c_outside: bool = False,
    init_scale: float = 0.5,
    s0_scale: float = 1.0,
    norm_floor_ratio: float = 1e-2,
) -> MemoryInhibitionReport:
    """Balaye la magnitude et la structure de C, agrège sur ``seeds``.

    Pour chaque graine :
      1. construit un ChronoSpiraton(d, init_scale, bounded) à init seedée
         (A, B, D, L fixés, C original ignoré : on le réassigne) ;
      2. mesure ρ(A) (échelle de γ) ;
      3. tire s0 et s_prev seedés (s_prev = 0 par défaut de l'équation) ;
      4. pour chaque (structure, γ), réassigne ``model.C.weight`` = γ·ρ(A)·C_unit
         sous no_grad, déroule (stability_scan + forward return_trace), mesure.

    Conditions agrégées dans le rapport :
      - chaque ``structure`` (dense, diag+, diag±, anti-sym) à chaque γ>0 ;
      - ``CTRL-0`` : C=0 (équivaut à γ=0, markovien) — point commun à toutes ;
      - ``CTRL-RAND`` : C dense de MÊME norme γ·ρ(A), regénéré PAR GRAINE
        (anti-artefact : isole l'atténuation scalaire de la structure) ;
      - ``+C(dense)`` : contrôle de signe (C → −C dans l'inner, soit +C(s_prev)).

    Toutes les matrices C sont de norme Frobenius γ·ρ(A) EXACTEMENT, ce qui rend
    CTRL-RAND comparable structure-à-structure à norme égale.
    """
    seeds = tuple(int(s) for s in seeds)
    gamma_ratios = tuple(float(g) for g in gamma_ratios)
    structures = tuple(structures)

    # Accumulateurs : clé = (condition, gamma_ratio) -> listes inter-graines.
    div_flags: Dict[Tuple[str, float], List[int]] = {}
    max_norms: Dict[Tuple[str, float], List[float]] = {}
    fin_norms: Dict[Tuple[str, float], List[float]] = {}
    returns: Dict[Tuple[str, float], List[float]] = {}
    valids: Dict[Tuple[str, float], List[int]] = {}

    def _push(cond: str, g: float, stab: Dict[str, float], ao: Dict[str, float]) -> None:
        key = (cond, g)
        div_flags.setdefault(key, []).append(int(stab["diverged"]))
        max_norms.setdefault(key, []).append(stab["max_norm"])
        fin_norms.setdefault(key, []).append(stab["final_norm"])
        valids.setdefault(key, []).append(int(ao["valid"] and not ao["diverged"]))
        if ao["valid"] and not ao["diverged"]:
            returns.setdefault(key, []).append(ao["best_return"])
        else:
            returns.setdefault(key, [])

    for seed in seeds:
        # --- init seedée du modèle (A,B,D,L fixés une fois par graine) ---
        # c_outside ne touche AUCUN tirage de poids (il n'agit que dans step) :
        # à graine fixée, intérieur et extérieur partagent A,B,C,D,L identiques —
        # condition sine qua non pour attribuer toute différence à la POSITION.
        torch.manual_seed(seed)
        model = ChronoSpiraton(
            state_size=d, init_scale=init_scale, bounded=bounded, c_outside=c_outside
        )
        rho_A = spectral_radius(model.A.weight.detach())

        # s0 / s_prev seedés et reproductibles (générateur dédié, indépendant
        # du tirage des poids).
        gen = torch.Generator().manual_seed(seed + 100_000)
        s0 = s0_scale * torch.randn(1, d, generator=gen)
        s_prev = torch.zeros(1, d)  # s_{-1} = repos (défaut de l'équation)

        # Générateurs déterministes pour la construction de C (structure / rand).
        gen_struct = torch.Generator().manual_seed(seed + 200_000)
        gen_rand = torch.Generator().manual_seed(seed + 300_000)

        # Matrices unitaires (norme 1) par structure, FIXÉES une fois par graine
        # (la magnitude γ varie ensuite multiplicativement → même direction).
        C_unit: Dict[str, torch.Tensor] = {
            st: _make_C(d, st, gen_struct) for st in structures
        }

        def run_with_C(Cmat: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, float]]:
            with torch.no_grad():
                model.C.weight.copy_(Cmat)
            stab = model.stability_scan(s0, steps=steps, s_prev=s_prev)
            _, trace = model.forward(s0, steps=steps, s_prev=s_prev, return_trace=True)
            ao = _alpha_omega_on_trace(s0, trace, norm_floor_ratio=norm_floor_ratio)
            return stab, ao

        for g in gamma_ratios:
            scale = g * rho_A
            if g == 0.0:
                # C = 0 : markovien. Condition partagée CTRL-0 ; on l'enregistre
                # aussi sous chaque structure pour tracer la courbe complète.
                zeroC = torch.zeros(d, d)
                stab0, ao0 = run_with_C(zeroC)
                _push("CTRL-0", 0.0, stab0, ao0)
                for st in structures:
                    _push(st, 0.0, stab0, ao0)
                # CTRL-RAND à γ=0 = aussi nul (norme 0) → même point.
                _push("CTRL-RAND", 0.0, stab0, ao0)
                _push("+C(dense)", 0.0, stab0, ao0)
                continue

            # structures à norme γ·ρ(A)
            for st in structures:
                stab, ao = run_with_C(scale * C_unit[st])
                _push(st, g, stab, ao)

            # CTRL-RAND : dense gaussien REGÉNÉRÉ par (graine, γ), même norme.
            Crand = _make_C(d, "rand", gen_rand)
            stab_r, ao_r = run_with_C(scale * Crand)
            _push("CTRL-RAND", g, stab_r, ao_r)

            # Contrôle de signe : +C(s_prev) au lieu de −C. La cellule applique
            # toujours ``− C(...)`` (dans D si intérieur, sur la sortie si
            # extérieur) ; pour obtenir l'effet ``+C(s_prev)`` on copie −C_unit.
            # En position EXTÉRIEURE (feedback), le signe devrait enfin compter :
            # −C ext stabilise / +C ext déstabilise = signature forte (Tour 3).
            stab_s, ao_s = run_with_C(-(scale * C_unit["dense"]))
            _push("+C(dense)", g, stab_s, ao_s)

    # --- agrégation en cellules ---
    cells: List[GammaCell] = []
    all_conditions = ["CTRL-0"] + list(structures) + ["CTRL-RAND", "+C(dense)"]

    for cond in all_conditions:
        gset = sorted({g for (c, g) in div_flags if c == cond})
        for g in gset:
            key = (cond, g)
            df = div_flags.get(key, [])
            if not df:
                continue
            cells.append(
                GammaCell(
                    condition=cond,
                    gamma_ratio=g,
                    n_seeds=len(df),
                    diverged_rate=sum(df) / len(df),
                    median_max_norm=_median(max_norms[key]),
                    median_final_norm=_median(fin_norms[key]),
                    median_best_return=_median(returns[key]),
                    valid_return_rate=(sum(valids[key]) / len(valids[key])) if valids[key] else 0.0,
                )
            )

    # --- monotonie sur la structure de référence (première structure listée) ---
    ref = structures[0]
    ref_cells = sorted((c for c in cells if c.condition == ref), key=lambda c: c.gamma_ratio)
    gx = [c.gamma_ratio for c in ref_cells]
    dy = [c.diverged_rate for c in ref_cells]
    rho_s = spearman_rho(gx, dy)
    p = spearman_t_pvalue(rho_s, len(gx))

    drop = float("nan")
    if ref_cells:
        d0 = next((c.diverged_rate for c in ref_cells if c.gamma_ratio == 0.0), None)
        dmax = ref_cells[-1].diverged_rate
        if d0 is not None:
            drop = (d0 - dmax) * 100.0

    # cloche du best_return
    valid_ref = [c for c in ref_cells if c.median_best_return == c.median_best_return]
    bell_argmax = float("nan")
    bell_interior = False
    if valid_ref:
        best = max(valid_ref, key=lambda c: c.median_best_return)
        bell_argmax = best.gamma_ratio
        gammas_sorted = [c.gamma_ratio for c in valid_ref]
        bell_interior = (best.gamma_ratio != gammas_sorted[0]) and (best.gamma_ratio != gammas_sorted[-1])

    return MemoryInhibitionReport(
        d=d,
        steps=steps,
        seeds=seeds,
        bounded=bounded,
        c_outside=c_outside,
        init_scale=init_scale,
        gamma_ratios=gamma_ratios,
        structures=structures,
        cells=cells,
        ref_structure=ref,
        spearman_gamma_diverged=rho_s,
        spearman_pvalue=p,
        diverged_drop_pts=drop,
        bell_argmax_gamma=bell_argmax,
        bell_is_interior=bell_interior,
    )
