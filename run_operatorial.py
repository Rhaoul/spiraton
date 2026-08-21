import torch

from spiraton.diagnostics.operatorial_double_dynamics import (
    OperatorialChrono,
    OperatorialConfig,
    isospectral_check,
    run_operatorial_double_dynamics,
)

torch.manual_seed(0)
SEEDS = tuple(range(12))
D = 8
STEPS = 30

print("=" * 72)
print("CONTROLE 1 — ISOSPECTRALITE de D@L vs L@D (fait lineaire pur)")
print("=" * 72)
for seed in range(3):
    chrono = OperatorialChrono(D, seed, OperatorialConfig(nonlinear=False))
    chk = isospectral_check(chrono.D, chrono.L)
    print(f"  seed={seed}: rho(DL)={chk['spectral_radius_DL']:.10f}  "
          f"rho(LD)={chk['spectral_radius_LD']:.10f}  "
          f"gap_rho={chk['spectral_radius_gap']:.2e}  "
          f"max_eigval_gap={chk['max_sorted_eigval_gap']:.2e}")

def block(title, cfg):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    rep = run_operatorial_double_dynamics(
        d=D, steps=STEPS, seeds=SEEDS, cfg=cfg, condition=title.split("—")[0].strip()
    )
    print(rep.summary())
    return rep

# Condition 0 : CTRL baseline — D=L (LD doit etre identique a DL).
block("CTRL — D=L (baseline, asymetrie attendue NULLE), lineaire",
      OperatorialConfig(nonlinear=False, ctrl_same_DL=True))

# Condition 1 : lineaire pur, D!=L (isospectral => pas d'asymetrie attendue).
block("LIN — lineaire pur, D!=L (isospectral, asymetrie attendue ~0)",
      OperatorialConfig(nonlinear=False))

# Condition 2 : non-lineaire (B(s2) + tanh), D!=L proche identite.
block("NONLIN — non-lineaire B(s2)+tanh, D!=L (piste a)",
      OperatorialConfig(nonlinear=True))

# Condition 3 : asymetrie intrinseque D expansif / L contractif, lineaire.
block("INTRINSIC-LIN — D expansif(rho=1.3) / L contractif(rho=0.5), lineaire (piste b)",
      OperatorialConfig(nonlinear=False, rho_D=1.3, rho_L=0.5))

# Condition 3b : asymetrie intrinseque + non-lineaire (cumul (a)+(b)).
block("INTRINSIC-NONLIN — D rho=1.3 / L rho=0.5 + non-lineaire (a+b)",
      OperatorialConfig(nonlinear=True, rho_D=1.3, rho_L=0.5))

# Condition CTRL intrinseque : D=L mais rho fixe (les deux egaux) -> pas d'asymetrie.
block("CTRL-INTRINSIC — D=L (rho=1.3 commun), lineaire (controle piste b)",
      OperatorialConfig(nonlinear=False, ctrl_same_DL=True, rho_D=1.3))
