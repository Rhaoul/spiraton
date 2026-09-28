from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .alpha_omega_spatial import alpha_omega_metrics


@dataclass(frozen=True)
class DoubleDynamicsReport:
    """Comparaison des deux compositions temporelles du Logos (axiome §2.4).

    Convention de composition (notation fonctionnelle ``∘`` : on lit de droite
    à gauche, l'opérateur le plus à droite agit en premier) :

      - **L∘D** = D *puis* L = ``outward`` (expansion dextrogyre) suivi de
        ``inward`` (contraction lévogyre). La théorie prédit une
        *stabilisation* (attracteur : faible variance, retour proche).
      - **D∘L** = L *puis* D = ``inward`` suivi de ``outward``. La théorie
        prédit une *bifurcation/divergence* (variance plus forte, dérive).

    PORTÉE (à ne pas sur-interpréter) : ici ``outward``/``inward`` ne sont PAS
    deux opérateurs D et L distincts au sens de §2.3 (« non inverses »). C'est
    la *même* cellule et la *même* projection parcourues dans un sens inversé :
    la dualité testée est **directionnelle** (ordre de balayage séquentiel), pas
    **opératorielle**. Un éventuel ``variance_gap`` mesure donc une asymétrie de
    parcours, pas la dualité D≠L de la théorie. Voir
    ``docs/CHANTIER2_DOUBLE_DYNAMIQUE.md`` pour la lecture mécaniste et les
    pistes vers une dualité pleinement opératorielle.

    Toutes les séries vont de t=0 (x0) à t=2K et sont moyennées sur le batch.
    Le « signal » à chaque pas est ``cosine − l2_norm`` (même heuristique que
    ``run_alpha_omega_spatial`` : haut = aligné ET proche de l'origine).
    """

    k: int

    # Séries (2K+1,) moyennées sur le batch.
    ld_l2: torch.Tensor
    ld_cos: torch.Tensor
    dl_l2: torch.Tensor
    dl_cos: torch.Tensor

    # Résumés scalaires.
    ld_score: float           # max du signal (cos - l2) sur la trajectoire L∘D
    dl_score: float           # idem pour D∘L
    ld_best_return_step: int
    dl_best_return_step: int
    ld_signal_variance: float  # variance temporelle du signal L∘D
    dl_signal_variance: float
    ld_final_l2: float         # distance normalisée finale à l'origine
    dl_final_l2: float

    # Mesure d'asymétrie (NON forcée — simple constat).
    variance_gap: float        # dl_var - ld_var  (>0 ⇒ conforme à la théorie)
    stabilizes_as_predicted: bool  # ld_var < dl_var

    def summary(self) -> str:
        verdict = (
            "L∘D stabilise davantage que D∘L (conforme à la théorie)"
            if self.stabilizes_as_predicted
            else "asymétrie absente ou inversée (résultat à documenter tel quel)"
        )
        return (
            f"[double dynamique, K={self.k}]\n"
            f"  L∘D (outward→inward): var={self.ld_signal_variance:.4g}, "
            f"score={self.ld_score:.4g}, return@{self.ld_best_return_step}, "
            f"final_l2={self.ld_final_l2:.4g}\n"
            f"  D∘L (inward→outward): var={self.dl_signal_variance:.4g}, "
            f"score={self.dl_score:.4g}, return@{self.dl_best_return_step}, "
            f"final_l2={self.dl_final_l2:.4g}\n"
            f"  variance_gap (dl-ld) = {self.variance_gap:.4g} → {verdict}"
        )


def _trajectory(grid_module, x0: torch.Tensor, first: str, second: str, k: int):
    """K pas de grille dans le sens ``first`` puis K pas dans le sens ``second``.

    Retourne (l2_series, cos_series), chacune de longueur 2K+1, moyennées sur
    le batch, mesurées à chaque pas relativement à x0.
    """
    l2_series: List[torch.Tensor] = []
    cos_series: List[torch.Tensor] = []

    def record(xt: torch.Tensor) -> None:
        l2, cos = alpha_omega_metrics(x0, xt)
        l2_series.append(l2.mean())
        cos_series.append(cos.mean())

    xt = x0
    record(xt)  # t=0
    for _ in range(k):
        xt = grid_module(xt, steps=1, order=first)
        record(xt)
    for _ in range(k):
        xt = grid_module(xt, steps=1, order=second)
        record(xt)

    return torch.stack(l2_series), torch.stack(cos_series)


@torch.no_grad()
def run_double_dynamics(grid_module, x0: torch.Tensor, *, k: int = 6) -> DoubleDynamicsReport:
    """Mesure l'asymétrie L∘D vs D∘L sur un même système (chantier 2).

    grid_module : un ``SpiralGrid`` (sa ``forward`` doit accepter ``order=``).
        On le pilote dans les deux sens : même cellule, même projection
        ``y_to_vec``, seules changent les compositions temporelles.
    x0 : (B,H,W,C).
    k  : nombre de pas par demi-trajectoire.

    AVERTISSEMENT MÉTHODOLOGIQUE. C'est une MESURE, pas une cible. Si
    l'asymétrie prédite (L∘D stabilise, D∘L bifurque) n'apparaît pas, c'est un
    résultat en soi : il faut le documenter, pas trafiquer le code pour la
    produire (cf. CLAUDE.md, chantier 2).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if x0.dim() != 4:
        raise ValueError("x0 must be (B,H,W,C)")

    # L∘D : D (outward) puis L (inward).
    ld_l2, ld_cos = _trajectory(grid_module, x0, first="outward", second="inward", k=k)
    # D∘L : L (inward) puis D (outward).
    dl_l2, dl_cos = _trajectory(grid_module, x0, first="inward", second="outward", k=k)

    ld_signal = ld_cos - ld_l2
    dl_signal = dl_cos - dl_l2

    ld_best = int(torch.argmax(ld_signal).item())
    dl_best = int(torch.argmax(dl_signal).item())

    ld_var = float(ld_signal.var(unbiased=False).item())
    dl_var = float(dl_signal.var(unbiased=False).item())

    return DoubleDynamicsReport(
        k=k,
        ld_l2=ld_l2,
        ld_cos=ld_cos,
        dl_l2=dl_l2,
        dl_cos=dl_cos,
        ld_score=float(ld_signal[ld_best].item()),
        dl_score=float(dl_signal[dl_best].item()),
        ld_best_return_step=ld_best,
        dl_best_return_step=dl_best,
        ld_signal_variance=ld_var,
        dl_signal_variance=dl_var,
        ld_final_l2=float(ld_l2[-1].item()),
        dl_final_l2=float(dl_l2[-1].item()),
        variance_gap=dl_var - ld_var,
        stabilizes_as_predicted=ld_var < dl_var,
    )
