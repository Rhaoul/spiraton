from __future__ import annotations

"""Test OPÉRATORIEL de la double dynamique (axiome §2.4), Tour 1.

Sœur opératorielle de :mod:`double_dynamics` — mais là où ce dernier ne fait
varier que l'ORDRE DE BALAYAGE d'une grille (dualité *directionnelle*, même
cellule parcourue à l'endroit/à l'envers), ce module oppose deux *compositions
de deux opérateurs temporels distincts* D et L (dualité *opératorielle*, le seul
niveau où l'axiome 4 peut être vrai ou faux de façon non triviale).

Substrat. On reprend exactement l'inner de :class:`ChronoSpiraton`
(``inner = A(s) + B(s²) − C(s_prev)``) mais on réordonne l'application des deux
temps, là où ``step()`` les FUSIONNE additivement (``D(inner) + L(s)``, donc
insensible à l'ordre). Deux variantes :

    LD : s_next = L( D(inner) )     (D=action puis L=mémoire/intégration)
    DL : s_next = D( L(inner) )     (L puis D)

Convention de nommage. ``LD`` = « L∘D » au sens fonctionnel : D agit d'abord,
L referme — c'est la fermeture spirale du corpus (émission centrifuge l.205-207
suivie de l'intégration centripète l.208-210, retournement <DX><OUT>→<LV><IN>).
``DL`` = « D∘L » : on projette ce qui n'a pas été déployé, contre-sens spiral,
prédit dissipant (l.802, l.825-826).

FAIT MATHÉMATIQUE (vérifié par :func:`isospectral_check`). Les MATRICES PRODUIT
``D L`` et ``L D`` sont ISOSPECTRALES (théorème AB~BA : mêmes valeurs propres
non nulles, donc même rayon spectral). C'est ce que mesure ``isospectral_check``.

NUANCE IMPORTANTE (réserve du contrôle maths, Tour 1). Cette isospectralité
AB~BA porte sur les matrices ``D L`` / ``L D`` PRISES SEULES — elle ne décrit
PAS la dynamique réellement déroulée. La récurrence opérante est du second ordre
``s_next = (L D)(A s − C s_prev)`` (resp. ``(D L)(...)``) ; sur l'état empilé
``z = [s; s_prev]`` la matrice compagnon est ``M_LD = [[LDA, −LDC], [I, 0]]`` vs
``M_DL = [[DLA, −DLC], [I, 0]]``, et ``LDA`` / ``DLA`` ne sont PAS isospectrales
(gaps de ρ mesurés ≈ 0.09–0.39, bien au-dessus du bruit float). L'argument
« l'ordre linéaire pur ne peut produire AUCUNE asymétrie » est donc trop fort
pour les trajectoires : il existe bien une asymétrie spectrale opérante. Ce qui
sauve le verdict empirique : le SIGNE de cette asymétrie est ALÉATOIRE entre
graines (≈8/12 pour ρ(M_LD)>ρ(M_DL)), donc elle moyenne vers du bruit et ne
confirme PAS la direction prédite par l'axiome 4 (L∘D stabilise). La vraie raison
pour laquelle LIN ne montre pas d'asymétrie directionnelle n'est donc pas le
théorème AB~BA mais « asymétrie spectrale réelle mais de signe non reproductible ».

Si une asymétrie ROBUSTE de l'axiome 4 existait, elle viendrait de :
  (a) la NON-LINÉARITÉ (terme B(s²) ou activation), ou
  (b) une ASYMÉTRIE INTRINSÈQUE D/L (D expansif ρ>1, L contractif ρ<1) —
      l'opérateur appliqué EN DERNIER dominerait le régime asymptotique.
Aucune des deux ne survit à l'augmentation des graines (voir lecture Tour 1).

MÉTHODE (REFUS). C'est une MESURE, pas une cible. Baseline CTRL obligatoire
(D=L ⇒ LD≡DL par construction ⇒ asymétrie nulle attendue). Aucun boost codé en
dur, aucune lecture d'étiquette. Seeds fixés. Tout résultat négatif est un
verdict, pas un échec à corriger.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch

from .alpha_omega_spatial import alpha_omega_metrics


# ---------------------------------------------------------------------------
# Contrôle 1 : isospectralité de DL vs LD (fait linéaire pur).
# ---------------------------------------------------------------------------

@torch.no_grad()
def isospectral_check(D: torch.Tensor, L: torch.Tensor) -> Dict[str, float]:
    """Compare le spectre de ``D@L`` et ``L@D`` (matrices carrées d×d).

    Théorème : pour toutes matrices carrées, AB et BA ont les mêmes valeurs
    propres. Donc ``spectral_radius(DL) == spectral_radius(LD)`` à l'arrondi
    machine. Cette fonction le confirme numériquement — c'est la BASELINE qui
    prouve qu'aucune asymétrie linéaire pure n'est possible par l'ordre seul.
    """
    DL = D @ L
    LD = L @ D
    ev_dl = torch.linalg.eigvals(DL)
    ev_ld = torch.linalg.eigvals(LD)
    rho_dl = float(ev_dl.abs().max().item())
    rho_ld = float(ev_ld.abs().max().item())
    # Comparaison spectre trié par module (les ev sont identiques en théorie).
    s_dl, _ = torch.sort(ev_dl.abs(), descending=True)
    s_ld, _ = torch.sort(ev_ld.abs(), descending=True)
    max_ev_gap = float((s_dl - s_ld).abs().max().item())
    return {
        "spectral_radius_DL": rho_dl,
        "spectral_radius_LD": rho_ld,
        "spectral_radius_gap": abs(rho_dl - rho_ld),
        "max_sorted_eigval_gap": max_ev_gap,
    }


# ---------------------------------------------------------------------------
# Opérateur temporel réordonnable minimal.
# ---------------------------------------------------------------------------

def _make_op(d: int, gen: torch.Generator, *, scale: float, spectral_radius: Optional[float]) -> torch.Tensor:
    """Construit une matrice d×d.

    Si ``spectral_radius`` est None : ``I + scale·N(0,1)`` (proche identité,
    comme MatrixSpiratonCell). Sinon : matrice normalisée pour avoir le rayon
    spectral demandé (pour la condition d'asymétrie intrinsèque D/L).
    """
    M = torch.eye(d) + scale * torch.randn(d, d, generator=gen)
    if spectral_radius is not None:
        rho = float(torch.linalg.eigvals(M).abs().max().item())
        if rho > 0:
            M = M * (spectral_radius / rho)
    return M


@dataclass
class OperatorialConfig:
    """Définit une condition expérimentale.

    nonlinear : si True, l'inner inclut le terme quadratique B(s²) et une
        activation tanh terminale (non-linéarité, piste (a)).
    rho_D, rho_L : rayons spectraux imposés à D et L (piste (b)). None ⇒
        opérateurs proche-identité (init_scale).
    ctrl_same_DL : si True, L est une COPIE de D (baseline : LD≡DL).
    """
    nonlinear: bool = False
    rho_D: Optional[float] = None
    rho_L: Optional[float] = None
    ctrl_same_DL: bool = False
    init_scale: float = 0.3
    quad_scale: float = 0.2  # poids du terme B(s²) quand nonlinear


class OperatorialChrono:
    """Porte D, L (et A, B, C) et déroule les deux variantes LD / DL.

    Reprend l'inner de ChronoSpiraton. La SEULE différence entre les deux
    variantes est l'ordre d'application des deux temps D et L — tout le reste
    (A, B, C, s_prev, s0) est partagé, à init identique. C'est ce qui isole
    l'effet de l'ordre opératoriel.
    """

    def __init__(self, d: int, seed: int, cfg: OperatorialConfig) -> None:
        self.d = d
        self.cfg = cfg
        gen = torch.Generator().manual_seed(seed)

        self.A = _make_op(d, gen, scale=cfg.init_scale, spectral_radius=None)
        self.B = _make_op(d, gen, scale=cfg.init_scale, spectral_radius=None)
        self.C = _make_op(d, gen, scale=cfg.init_scale, spectral_radius=None)
        self.D = _make_op(d, gen, scale=cfg.init_scale, spectral_radius=cfg.rho_D)
        if cfg.ctrl_same_DL:
            self.L = self.D.clone()
        else:
            self.L = _make_op(d, gen, scale=cfg.init_scale, spectral_radius=cfg.rho_L)

    def _inner(self, s: torch.Tensor, s_prev: torch.Tensor) -> torch.Tensor:
        inner = s @ self.A.t() - s_prev @ self.C.t()
        if self.cfg.nonlinear:
            inner = inner + self.cfg.quad_scale * (s * s) @ self.B.t()
        return inner

    def _step(self, s: torch.Tensor, s_prev: torch.Tensor, order: str) -> torch.Tensor:
        inner = self._inner(s, s_prev)
        if order == "LD":          # D puis L : s_next = L(D(inner))
            out = (inner @ self.D.t()) @ self.L.t()
        elif order == "DL":        # L puis D : s_next = D(L(inner))
            out = (inner @ self.L.t()) @ self.D.t()
        else:
            raise ValueError(order)
        if self.cfg.nonlinear:
            out = torch.tanh(out)
        return out

    @torch.no_grad()
    def trajectory(self, s0: torch.Tensor, order: str, steps: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Déroule ``steps`` pas. Retourne (l2_series, cos_series, norm_series).

        l2/cos mesurés à chaque pas relativement à s0 (signal de retour α-ω) ;
        norm = norme de l'état (croissance/divergence).
        """
        s_prev = torch.zeros_like(s0)
        s = s0
        l2s: List[torch.Tensor] = []
        coss: List[torch.Tensor] = []
        norms: List[torch.Tensor] = []
        l2, cos = alpha_omega_metrics(s0, s)
        l2s.append(l2.mean()); coss.append(cos.mean()); norms.append(s.norm(dim=-1).mean())
        for _ in range(steps):
            nxt = self._step(s, s_prev, order)
            s_prev, s = s, nxt
            if not torch.isfinite(s).all():
                # divergence dure : on fige des marqueurs et on arrête.
                big = torch.tensor(float("inf"))
                l2s.append(big); coss.append(torch.tensor(0.0)); norms.append(big)
                break
            l2, cos = alpha_omega_metrics(s0, s)
            l2s.append(l2.mean()); coss.append(cos.mean()); norms.append(s.norm(dim=-1).mean())
        return torch.stack(l2s), torch.stack(coss), torch.stack(norms)


# ---------------------------------------------------------------------------
# Diagnostic principal : agrège sur graines.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorialReport:
    condition: str
    n_seeds: int
    steps: int
    # Moyennes inter-graines.
    mean_var_LD: float      # variance temporelle moyenne du signal cos-l2, LD
    mean_var_DL: float
    mean_variance_gap: float        # mean(var_DL - var_LD) ; >0 ⇒ conforme
    seeds_gap_positive: int         # nb de graines avec var_DL > var_LD
    mean_best_return_LD: float      # max(cos-l2) moyen, LD
    mean_best_return_DL: float
    best_return_gap: float          # LD - DL ; >0 ⇒ conforme (l.290)
    seeds_return_LD_wins: int
    mean_final_norm_LD: float
    mean_final_norm_DL: float
    n_diverged_LD: int
    n_diverged_DL: int
    commutator_norm_DL: float       # ‖[D,L]‖ moyen (covariable)

    def summary(self) -> str:
        return (
            f"[{self.condition}] seeds={self.n_seeds} steps={self.steps}\n"
            f"  var(cos-l2): LD={self.mean_var_LD:.4g}  DL={self.mean_var_DL:.4g}  "
            f"gap(DL-LD)={self.mean_variance_gap:.4g}  ({self.seeds_gap_positive}/{self.n_seeds} graines gap>0)\n"
            f"  best_return: LD={self.mean_best_return_LD:.4g}  DL={self.mean_best_return_DL:.4g}  "
            f"gap(LD-DL)={self.best_return_gap:.4g}  ({self.seeds_return_LD_wins}/{self.n_seeds} LD gagne)\n"
            f"  final_norm: LD={self.mean_final_norm_LD:.4g}  DL={self.mean_final_norm_DL:.4g}  "
            f"diverged LD={self.n_diverged_LD} DL={self.n_diverged_DL}\n"
            f"  ‖[D,L]‖={self.commutator_norm_DL:.4g}"
        )


@torch.no_grad()
def run_operatorial_double_dynamics(
    *,
    d: int = 8,
    steps: int = 30,
    seeds: Tuple[int, ...] = tuple(range(12)),
    cfg: Optional[OperatorialConfig] = None,
    condition: str = "?",
    s0_scale: float = 1.0,
) -> OperatorialReport:
    """Oppose LD vs DL sur le même s0, sur ``seeds`` graines fixées.

    Pour chaque graine : on construit un OperatorialChrono (D,L,A,B,C tirés de
    la graine), un s0 (tiré de la même graine, donc reproductible), on déroule
    LD et DL, on extrait variance temporelle du signal cos-l2 et best_return.
    """
    cfg = cfg or OperatorialConfig()

    var_ld_list: List[float] = []
    var_dl_list: List[float] = []
    br_ld_list: List[float] = []
    br_dl_list: List[float] = []
    fn_ld_list: List[float] = []
    fn_dl_list: List[float] = []
    comm_list: List[float] = []
    gap_pos = 0
    ret_ld_wins = 0
    div_ld = 0
    div_dl = 0

    for seed in seeds:
        chrono = OperatorialChrono(d, seed, cfg)
        gen = torch.Generator().manual_seed(seed + 10_000)
        s0 = s0_scale * torch.randn(1, d, generator=gen)

        l2_ld, cos_ld, n_ld = chrono.trajectory(s0, "LD", steps)
        l2_dl, cos_dl, n_dl = chrono.trajectory(s0, "DL", steps)

        sig_ld = cos_ld - l2_ld
        sig_dl = cos_dl - l2_dl

        diverged_ld = not torch.isfinite(sig_ld).all()
        diverged_dl = not torch.isfinite(sig_dl).all()
        div_ld += int(diverged_ld)
        div_dl += int(diverged_dl)

        # Variance temporelle (sur les pas finis seulement).
        fin_ld = sig_ld[torch.isfinite(sig_ld)]
        fin_dl = sig_dl[torch.isfinite(sig_dl)]
        var_ld = float(fin_ld.var(unbiased=False).item()) if fin_ld.numel() > 1 else float("inf")
        var_dl = float(fin_dl.var(unbiased=False).item()) if fin_dl.numel() > 1 else float("inf")
        # Divergence ⇒ variance traitée comme +inf (bifurcation extrême).
        if diverged_ld:
            var_ld = float("inf")
        if diverged_dl:
            var_dl = float("inf")

        # best_return EXCLUANT t=0 (s0 vs s0 ⇒ signal=1 trivial, non un retour).
        # On cherche le retour PROCHE-ET-ALIGNÉ après avoir quitté l'origine (l.290).
        br_ld = float(fin_ld[1:].max().item()) if fin_ld.numel() > 1 else float("-inf")
        br_dl = float(fin_dl[1:].max().item()) if fin_dl.numel() > 1 else float("-inf")

        var_ld_list.append(var_ld); var_dl_list.append(var_dl)
        br_ld_list.append(br_ld); br_dl_list.append(br_dl)
        fn_ld_list.append(float(n_ld[torch.isfinite(n_ld)][-1].item()) if torch.isfinite(n_ld).any() else float("inf"))
        fn_dl_list.append(float(n_dl[torch.isfinite(n_dl)][-1].item()) if torch.isfinite(n_dl).any() else float("inf"))

        comm = float(torch.linalg.norm(chrono.D @ chrono.L - chrono.L @ chrono.D).item())
        comm_list.append(comm)

        # gap>0 si DL plus variable (bifurque davantage) que LD.
        if var_dl > var_ld:
            gap_pos += 1
        if br_ld > br_dl:
            ret_ld_wins += 1

    def _finite_mean(xs: List[float]) -> float:
        f = [x for x in xs if x != float("inf") and x != float("-inf")]
        return float(sum(f) / len(f)) if f else float("inf")

    mean_var_ld = _finite_mean(var_ld_list)
    mean_var_dl = _finite_mean(var_dl_list)
    mean_br_ld = _finite_mean(br_ld_list)
    mean_br_dl = _finite_mean(br_dl_list)

    return OperatorialReport(
        condition=condition,
        n_seeds=len(seeds),
        steps=steps,
        mean_var_LD=mean_var_ld,
        mean_var_DL=mean_var_dl,
        mean_variance_gap=(mean_var_dl - mean_var_ld),
        seeds_gap_positive=gap_pos,
        mean_best_return_LD=mean_br_ld,
        mean_best_return_DL=mean_br_dl,
        best_return_gap=(mean_br_ld - mean_br_dl),
        seeds_return_LD_wins=ret_ld_wins,
        mean_final_norm_LD=_finite_mean(fn_ld_list),
        mean_final_norm_DL=_finite_mean(fn_dl_list),
        n_diverged_LD=div_ld,
        n_diverged_DL=div_dl,
        commutator_norm_DL=float(sum(comm_list) / len(comm_list)),
    )
