"""Sonde de thermodynamique sémantique — E1 (DX/LV ↔ divergence), E3 (A→B→A′),
E2 (opérateurs ↔ observables), avec contrôles. Expérimental, non canonique.

Protocole pré-enregistré : ``docs/SEMANTIC_THERMO_PROTOCOLE.md`` (définitions,
hypothèses, critères, révision R1 issue de E0). Ce module n'ajuste aucune
définition sur les étiquettes : le seul paramètre « appris » est le champ de
référence (vecteurs de la calibration de ``dataset_aba``) et, pour E2, le
classifieur entraîné sur cette même calibration.

Usage :
    python -m spiraton.diagnostics.semantic_thermo_probe --experiment all --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..data.semantic_thermo_adapter import (
    SLICES,
    CachedVectors,
    CycleRecord,
    SliceNormalizer,
    load_cycles,
    pool_rows,
    recut_cycle,
    sentence_thirds,
    split_cycles,
)
from ..experimental.semantic_thermodynamics import (
    ReferenceField,
    ThermoProbeConfig,
    literal_flux_divergence,
    return_ratio,
    segment_observables,
)

REPO = Path(__file__).resolve().parents[2]
ATELIER = REPO.parent
CORPORA = {
    "dataset": "dataset_aba.txt",
    "claude": "corpus_claude_aba.txt",
    "horscanon": "corpus_horscanon_aba.txt",
    "f0b": "corpus_f0b_aba.txt",
    "eve": "corpus_eve_clean.txt",
}
N_PERM = 10_000
N_BOOT = 2_000
N_SEQ_SHUFFLE = 20


# =============================================================================
# Statistiques (torch, vectorisées, seedées)
# =============================================================================

def _ranks(x: torch.Tensor) -> torch.Tensor:
    """Rangs moyens (ex æquo moyennés), 1-indexés, le long de la dernière dim."""
    order = x.argsort(dim=-1)
    sx = x.gather(-1, order)
    n = x.size(-1)
    r = torch.arange(1, n + 1, dtype=torch.float64).expand_as(sx).clone()
    # moyenne des rangs sur les blocs d'égalité
    if n > 1:
        diff = torch.ones_like(sx, dtype=torch.bool)
        diff[..., 1:] = sx[..., 1:] != sx[..., :-1]
        block = diff.long().cumsum(-1)
        nb = int(block.max()) + 1
        sums = torch.zeros(*sx.shape[:-1], nb, dtype=torch.float64).scatter_add_(-1, block, r)
        cnts = torch.zeros(*sx.shape[:-1], nb, dtype=torch.float64).scatter_add_(-1, block, torch.ones_like(r))
        r = (sums / cnts).gather(-1, block)
    out = torch.empty_like(r)
    out.scatter_(-1, order, r)
    return out


def auc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AUC de Mann-Whitney, P(score_pos > score_neg), ex æquo = ½. Batch sur dims initiales."""
    labels = labels.to(torch.float64)
    r = _ranks(scores.to(torch.float64))
    n_pos = labels.sum(-1)
    n_neg = labels.size(-1) - n_pos
    return ((r * labels).sum(-1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cohen_d(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.numel(), b.numel()
    va, vb = a.var(unbiased=True), b.var(unbiased=True)
    sp = math.sqrt(((na - 1) * float(va) + (nb - 1) * float(vb)) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def perm_null_auc(scores, labels, strata, *, n: int, gen: torch.Generator,
                  batch: int = 500) -> torch.Tensor:
    """AUC sous permutation des étiquettes À L'INTÉRIEUR de chaque strate.
    Les rangs des scores sont fixes : seules les étiquettes bougent (par lots)."""
    lab = labels.to(torch.float64)
    r = _ranks(scores.to(torch.float64))
    n_pos = float(lab.sum())
    n_neg = lab.numel() - n_pos
    strata_idx = [(strata == s).nonzero().flatten() for s in strata.unique()]
    res = []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        out = lab.expand(b, -1).clone()
        for idx in strata_idx:
            perm = torch.rand(b, idx.numel(), generator=gen, dtype=torch.float64).argsort(dim=1)
            out[:, idx] = lab[idx][perm]
        res.append(((out * r).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return torch.cat(res)


def boot_auc_by_cycle(scores, labels, cycle_ids, *, n: int, gen: torch.Generator) -> Tuple[float, float]:
    """IC 95 % de l'AUC par rééchantillonnage des CYCLES (segments d'un cycle ensemble)."""
    uniq, inv = cycle_ids.unique(return_inverse=True)
    m = uniq.numel()
    vals = []
    for _ in range(n):
        pick = torch.randint(0, m, (m,), generator=gen)
        w = torch.bincount(pick, minlength=m)[inv]          # multiplicité de chaque segment
        keep = w > 0
        s = scores[keep].repeat_interleave(w[keep])
        l = labels[keep].repeat_interleave(w[keep])
        if 0 < int(l.sum()) < l.numel():
            vals.append(float(auc(s, l)))
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.quantile(0.025)), float(v.quantile(0.975))


def paired_test(diff: torch.Tensor, *, n_boot: int, gen: torch.Generator) -> Dict[str, float]:
    """Test apparié UNILATÉRAL (H : diff > 0). Wilcoxon signé (approx. normale,
    correction d'ex æquo), taux de signe, médiane et IC bootstrap 95 %."""
    d = diff[diff != 0]
    n = d.numel()
    out = {"n": int(diff.numel()), "n_nonzero": int(n),
           "median": float(diff.median()), "mean": float(diff.mean()),
           "frac_positive": float((diff > 0).double().mean())}
    if n == 0:
        out.update(z=0.0, p_one_sided=1.0, ci_low=0.0, ci_high=0.0)
        return out
    r = _ranks(d.abs())
    w_plus = float(r[d > 0].sum())
    mu = n * (n + 1) / 4
    _, counts = d.abs().unique(return_counts=True)
    tie = float((counts.double() ** 3 - counts.double()).sum()) / 48
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie)
    z = (w_plus - mu) / sigma if sigma > 0 else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))
    idx = torch.randint(0, diff.numel(), (n_boot, diff.numel()), generator=gen)
    meds = diff[idx].median(dim=1).values
    out.update(z=z, p_one_sided=p, ci_low=float(meds.quantile(0.025)), ci_high=float(meds.quantile(0.975)))
    return out


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


# =============================================================================
# Mesure des segments
# =============================================================================

@dataclass
class SegMeasure:
    cycle: int
    pos: int              # 0 = A, 1 = B, 2 = A′
    chir: int             # 1 = DX, 0 = LV
    n: int
    density: float
    dispersion: float
    speed: float
    div: float
    div_j: float
    entropy: float
    temperature: float


def _traj(seg, norm: SliceNormalizer, proj: Optional[torch.Tensor]) -> torch.Tensor:
    x = norm(seg.vectors)
    return x @ proj if proj is not None else x


def measure_cycles(
    cycles: Sequence[CycleRecord], norm: SliceNormalizer, field: ReferenceField,
    *, proj: Optional[torch.Tensor] = None, shuffle_gen: Optional[random.Random] = None,
) -> Tuple[List[SegMeasure], List[Optional[torch.Tensor]]]:
    """Mesure chaque segment non vide. Renvoie aussi les centroïdes (3 par cycle, None si vide).
    ``shuffle_gen`` : CTRL-2, mélange l'ordre des tokens de chaque segment."""
    ms: List[SegMeasure] = []
    cents: List[Optional[torch.Tensor]] = []
    for ci, c in enumerate(cycles):
        for pos, seg in enumerate(c.segments):
            if seg.vectors.size(0) == 0:
                cents.append(None)
                continue
            x = _traj(seg, norm, proj)
            if shuffle_gen is not None and x.size(0) > 1:
                order = list(range(x.size(0)))
                shuffle_gen.shuffle(order)
                x = x[order]
            s = segment_observables(x, field)
            rho = field.local(x)["density"]
            ms.append(SegMeasure(ci, pos, 1 if seg.chirality == "DX" else 0, s.n_tokens,
                                 s.density, s.dispersion, s.speed, s.divergence,
                                 literal_flux_divergence(x, rho), s.entropy, s.temperature))
            cents.append(x.mean(dim=0))
    return ms, cents


def build_field(calib: Sequence[CycleRecord], slice_name: str, cfg: ThermoProbeConfig,
                proj: Optional[torch.Tensor] = None) -> Tuple[SliceNormalizer, ReferenceField]:
    rows = pool_rows(calib)
    norm = SliceNormalizer(rows, slice_name, normalize=cfg.normalize)
    pts = torch.unique(norm(rows), dim=0)   # dédoublonner AVANT projection : le bruit
    if proj is not None:                    # flottant du produit ne doit pas créer de quasi-doublons
        pts = pts @ proj
    return norm, ReferenceField(pts, k=cfg.neighborhood_k, eps=cfg.eps)


def orthogonal(d: int, out: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q[:, :out]


def jl_projection(d: int, out: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, out, generator=g, dtype=torch.float64) / math.sqrt(out)


# =============================================================================
# E1 — DX/LV ↔ divergence
# =============================================================================

def _e1_block(ms: List[SegMeasure], field_name: str, *, strata: str, gen: torch.Generator,
              label: str = "chir", n_perm: int = N_PERM, n_boot: int = N_BOOT,
              with_ci: bool = True) -> Dict[str, float]:
    """AUC(DX > LV) sur ``field_name`` ; p par permutation stratifiée (``strata`` ∈ {none,pos,chir})."""
    ms = [m for m in ms if m.n >= 2]
    sc = torch.tensor([getattr(m, field_name) for m in ms], dtype=torch.float64)
    if label == "chir":
        lab = torch.tensor([m.chir for m in ms], dtype=torch.float64)
    else:  # "notprime" : A/B = 1, A′ = 0 (même sens que DX=1 sur F0)
        lab = torch.tensor([1 if m.pos < 2 else 0 for m in ms], dtype=torch.float64)
    st = {"none": torch.zeros(len(ms), dtype=torch.long),
          "pos": torch.tensor([m.pos for m in ms]),
          "chir": torch.tensor([m.chir for m in ms])}[strata]
    cyc = torch.tensor([m.cycle for m in ms])
    a = float(auc(sc, lab))
    null = perm_null_auc(sc, lab, st, n=n_perm, gen=gen)
    p = float(((null - 0.5).abs() >= abs(a - 0.5)).double().sum() + 1) / (n_perm + 1)
    p_one = float((null >= a).double().sum() + 1) / (n_perm + 1)
    pos, neg = sc[lab == 1], sc[lab == 0]
    out = {"n_pos": int(pos.numel()), "n_neg": int(neg.numel()), "auc": a,
           "mean_pos": float(pos.mean()), "mean_neg": float(neg.mean()),
           "median_pos": float(pos.median()), "median_neg": float(neg.median()),
           "cohen_d": cohen_d(pos, neg), "p_perm_two_sided": p, "p_perm_one_sided": p_one,
           "null_mean": float(null.mean()), "null_q95": float(null.quantile(0.95))}
    if with_ci:
        lo, hi = boot_auc_by_cycle(sc, lab, cyc, n=n_boot, gen=gen)
        out.update(ci_low=lo, ci_high=hi)
    return out


def run_e1(data: Dict[str, List[CycleRecord]], seed: int, cfg: ThermoProbeConfig) -> Dict:
    calib_i, test_i = split_cycles(len(data["dataset"]), seed=seed)
    calib = [data["dataset"][i] for i in calib_i]
    test = [data["dataset"][i] for i in test_i]
    mixed = data["horscanon"] + data["f0b"]
    # indices de cycle uniques pour le corpus mixte
    res: Dict = {"seed": seed, "n_calib_cycles": len(calib), "n_test_cycles": len(test)}
    for sl in SLICES:
        g = torch.Generator().manual_seed(1000 * seed + 17)
        norm, field = build_field(calib, sl, cfg)
        m_test, _ = measure_cycles(test, norm, field)
        m_cl, _ = measure_cycles(data["claude"], norm, field)
        m_mx, _ = measure_cycles(mixed, norm, field)
        block = {
            "canon_dataset_div": _e1_block(m_test, "div", strata="none", gen=g),
            "canon_claude_div": _e1_block(m_cl, "div", strata="none", gen=g),
            "strat_mixed_div": _e1_block(m_mx, "div", strata="pos", gen=g),
            "pos_mixed_div": _e1_block(m_mx, "div", strata="chir", gen=g, label="notprime"),
            "canon_dataset_divJ": _e1_block(m_test, "div_j", strata="none", gen=g, with_ci=False),
            "strat_mixed_divJ": _e1_block(m_mx, "div_j", strata="pos", gen=g, with_ci=False),
            "canon_dataset_len": _e1_block(m_test, "n", strata="none", gen=g, with_ci=False),
            "strat_mixed_len": _e1_block(m_mx, "n", strata="pos", gen=g, with_ci=False),
            "canon_dataset_density": _e1_block(m_test, "density", strata="none", gen=g, with_ci=False),
            "strat_mixed_density": _e1_block(m_mx, "density", strata="pos", gen=g, with_ci=False),
        }
        if sl == "no-logos":
            # CTRL-2 : ordre des tokens mélangé (20 tirages), AUC moyenne
            shuf = []
            for k in range(N_SEQ_SHUFFLE):
                rg = random.Random(10_000 * seed + k)
                ms_k, _ = measure_cycles(test, norm, field, shuffle_gen=rg)
                shuf.append(_e1_block(ms_k, "div", strata="none", gen=g, n_perm=200, with_ci=False)["auc"])
            block["ctrl2_seq_shuffle_dataset_auc_mean"] = sum(shuf) / len(shuf)
            block["ctrl2_seq_shuffle_dataset_auc_all"] = shuf
            shuf = []
            for k in range(N_SEQ_SHUFFLE):
                rg = random.Random(20_000 * seed + k)
                ms_k, _ = measure_cycles(mixed, norm, field, shuffle_gen=rg)
                shuf.append(_e1_block(ms_k, "div", strata="pos", gen=g, n_perm=200, with_ci=False)["auc"])
            block["ctrl2_seq_shuffle_mixed_auc_mean"] = sum(shuf) / len(shuf)
            # CTRL-7 : projection orthogonale carrée (invariance exacte) et JL d → d/2
            d = len(SLICES[sl])
            for tag, P in (("ctrl7_orth_square", orthogonal(d, d, seed + 101)),
                           ("ctrl7_jl_half", jl_projection(d, d // 2, seed + 202))):
                norm_p, field_p = build_field(calib, sl, cfg, proj=P)
                mt, _ = measure_cycles(test, norm_p, field_p, proj=P)
                mm, _ = measure_cycles(mixed, norm_p, field_p, proj=P)
                block[tag] = {
                    "canon_dataset_div_auc": _e1_block(mt, "div", strata="none", gen=g, n_perm=500, with_ci=False)["auc"],
                    "strat_mixed_div_auc": _e1_block(mm, "div", strata="pos", gen=g, n_perm=500, with_ci=False)["auc"],
                }
        res[sl] = block
    return res


# =============================================================================
# E3 — A → B → A′
# =============================================================================

E3_HYP = ("H3.1a_rhoA_gt_rhoB", "H3.1d_DB_gt_DA", "H3.2a_rhoAp_gt_rhoB",
          "H3.2d_DB_gt_DAp", "H3.3_dAB_gt_dAAp")


def _cycle_table(cycles, norm, field) -> List[Dict[str, float]]:
    ms, cents = measure_cycles(cycles, norm, field)
    by = {}
    for m in ms:
        by.setdefault(m.cycle, {})[m.pos] = m
    rows = []
    for ci in range(len(cycles)):
        segs = by.get(ci, {})
        cs = cents[3 * ci: 3 * ci + 3]
        if len(segs) < 3 or any(c is None for c in cs):
            continue
        A, B, Ap = segs[0], segs[1], segs[2]
        dAB = float((cs[0] - cs[1]).norm()); dBAp = float((cs[1] - cs[2]).norm())
        dAAp = float((cs[0] - cs[2]).norm())
        full = torch.cat([norm(s.vectors) for s in cycles[ci].segments], dim=0)
        rows.append({
            "cycle": ci, "form": cycles[ci].form, "op": cycles[ci].op,
            "rho_A": A.density, "rho_B": B.density, "rho_Ap": Ap.density,
            "D_A": A.dispersion, "D_B": B.dispersion, "D_Ap": Ap.dispersion,
            "div_A": A.div, "div_B": B.div, "div_Ap": Ap.div,
            "S_A": A.entropy, "S_B": B.entropy, "S_Ap": Ap.entropy,
            "T_A": A.temperature, "T_B": B.temperature, "T_Ap": Ap.temperature,
            "n_A": A.n, "n_B": B.n, "n_Ap": Ap.n,
            "distance_A_B": dAB, "distance_B_Ap": dBAp, "distance_A_Ap": dAAp,
            "return_ratio": return_ratio(torch.stack(cs)),
            "cycle_return_ratio_tokens": return_ratio(full),
        })
    return rows


def _e3_tests(rows, gen) -> Dict:
    t = lambda k: torch.tensor([r[k] for r in rows], dtype=torch.float64)
    diffs = {
        "H3.1a_rhoA_gt_rhoB": t("rho_A") - t("rho_B"),
        "H3.1d_DB_gt_DA": t("D_B") - t("D_A"),
        "H3.2a_rhoAp_gt_rhoB": t("rho_Ap") - t("rho_B"),
        "H3.2d_DB_gt_DAp": t("D_B") - t("D_Ap"),
        "H3.3_dAB_gt_dAAp": t("distance_A_B") - t("distance_A_Ap"),
    }
    out = {k: paired_test(v, n_boot=1000, gen=gen) for k, v in diffs.items()}
    adj = holm({k: v["p_one_sided"] for k, v in out.items()})
    for k in out:
        out[k]["p_holm"] = adj[k]
    out["n_cycles"] = len(rows)
    out["frac_dAAp_positive"] = float((t("distance_A_Ap") > 0).double().mean())
    out["mean_n"] = {k: float(t(k).mean()) for k in ("n_A", "n_B", "n_Ap")}
    return out


def _form_table(rows) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    keys = ("rho_A", "rho_B", "rho_Ap", "D_A", "D_B", "D_Ap", "div_A", "div_B", "div_Ap",
            "distance_A_B", "distance_B_Ap", "distance_A_Ap", "return_ratio")
    for f in sorted({r["form"] for r in rows}):
        rs = [r for r in rows if r["form"] == f]
        out[f] = {"n": len(rs), **{k: sum(r[k] for r in rs) / len(rs) for k in keys}}
    return out


def run_e3(data, seed: int, cfg: ThermoProbeConfig, vectors: CachedVectors) -> Dict:
    calib_i, test_i = split_cycles(len(data["dataset"]), seed=seed)
    calib = [data["dataset"][i] for i in calib_i]
    test = [data["dataset"][i] for i in test_i]
    res: Dict = {"seed": seed}
    for sl in ("no-logos", "phoneme", "context", "no-energy"):
        g = torch.Generator().manual_seed(2000 * seed + 29)
        norm, field = build_field(calib, sl, cfg)
        rng = random.Random(3000 * seed + 7)
        cut = [recut_cycle(c, vectors, rng) for c in test]
        blk = {
            "dataset_test": _e3_tests(_cycle_table(test, norm, field), g),
            "ctrl_cut": _e3_tests(_cycle_table(cut, norm, field), g),
            "ctrl_eve_thirds": _e3_tests(_cycle_table(data["eve_thirds"], norm, field), g),
            "claude": _e3_tests(_cycle_table(data["claude"], norm, field), g),
        }
        if sl == "no-logos":
            blk["forms_mixed"] = _form_table(_cycle_table(data["horscanon"] + data["f0b"], norm, field))
            blk["forms_dataset_test"] = _form_table(_cycle_table(test, norm, field))
        res[sl] = blk
    return res


# =============================================================================
# E2 — opérateurs
# =============================================================================

OPS = ("ADD", "SUB", "MUL", "DIV")


def _fit_logreg(X: torch.Tensor, y: torch.Tensor, *, l2: float = 1e-2, steps: int = 500) -> torch.Tensor:
    torch.manual_seed(0)
    W = torch.zeros(X.size(1) + 1, len(OPS), dtype=torch.float64, requires_grad=True)
    Xb = torch.cat([X, torch.ones(X.size(0), 1, dtype=torch.float64)], dim=1)
    opt = torch.optim.LBFGS([W], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xb @ W, y) + l2 * (W[:-1] ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach()


def _eval(W, X, y) -> Dict[str, float]:
    Xb = torch.cat([X, torch.ones(X.size(0), 1, dtype=torch.float64)], dim=1)
    logits = Xb @ W
    pred = logits.argmax(1)
    return {"accuracy": float((pred == y).double().mean()),
            "log_loss": float(torch.nn.functional.cross_entropy(logits, y)),
            "per_class_acc": {OPS[k]: float((pred[y == k] == k).double().mean()) if (y == k).any() else float("nan")
                              for k in range(len(OPS))}}


def _features(cycles, norm, field, kind: str) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.tensor([OPS.index(c.op) for c in cycles])
    if kind == "length":
        X = torch.tensor([[s.vectors.size(0) for s in c.segments] for c in cycles], dtype=torch.float64)
        return X, y
    if kind == "raw_mean":
        X = torch.stack([torch.cat([norm(s.vectors).mean(0) if s.vectors.size(0) else torch.zeros(len(norm.dims), dtype=torch.float64)
                                    for s in c.segments]) for c in cycles])
        return X, y
    ms, cents = measure_cycles(cycles, norm, field)
    by: Dict[int, Dict[int, SegMeasure]] = {}
    for m in ms:
        by.setdefault(m.cycle, {})[m.pos] = m
    feats = []
    for ci in range(len(cycles)):
        f = []
        for pos in range(3):
            m = by.get(ci, {}).get(pos)
            f += [m.density, m.speed, m.dispersion, m.div] if m else [0.0] * 4
        cs = cents[3 * ci: 3 * ci + 3]
        for i, j in ((0, 1), (1, 2), (0, 2)):
            f.append(float((cs[i] - cs[j]).norm()) if cs[i] is not None and cs[j] is not None else 0.0)
        feats.append(f)
    return torch.tensor(feats, dtype=torch.float64), y


def run_e2(data, seed: int, cfg: ThermoProbeConfig, n_perm_models: int = 20) -> Dict:
    calib_i, test_i = split_cycles(len(data["dataset"]), seed=seed)
    calib = [data["dataset"][i] for i in calib_i]
    test = [data["dataset"][i] for i in test_i]
    res: Dict = {"seed": seed}
    for sl in ("no-logos", "phoneme"):
        norm, field = build_field(calib, sl, cfg)
        blk = {}
        for kind in ("thermo", "length", "raw_mean"):
            Xtr, ytr = _features(calib, norm, field, kind)
            mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-8)
            W = _fit_logreg((Xtr - mu) / sd, ytr)
            ent = {}
            for name, cyc in (("dataset_test", test), ("claude", data["claude"])):
                X, y = _features(cyc, norm, field, kind)
                ent[name] = _eval(W, (X - mu) / sd, y)
            if kind == "thermo":
                g = torch.Generator().manual_seed(4000 * seed + 3)
                accs = []
                Xte, yte = _features(test, norm, field, kind)
                for _ in range(n_perm_models):
                    yp = ytr[torch.randperm(ytr.numel(), generator=g)]
                    Wp = _fit_logreg((Xtr - mu) / sd, yp)
                    accs.append(_eval(Wp, (Xte - mu) / sd, yte)["accuracy"])
                a = torch.tensor(accs, dtype=torch.float64)
                ent["ctrl1_perm_dataset_test"] = {"mean": float(a.mean()), "max": float(a.max())}
            blk[kind] = ent
        for name, cyc in (("dataset_test", test), ("claude", data["claude"])):
            ytr = torch.tensor([OPS.index(c.op) for c in calib])
            maj = int(torch.bincount(ytr, minlength=4).argmax())
            y = torch.tensor([OPS.index(c.op) for c in cyc])
            blk.setdefault("majority", {})[name] = {"class": OPS[maj], "accuracy": float((y == maj).double().mean())}
        res[sl] = blk
    return res


# =============================================================================
# Chargement, manifeste, CLI
# =============================================================================

def load_all(vectors: CachedVectors) -> Dict[str, List[CycleRecord]]:
    data = {k: load_cycles(str(ATELIER / f), vectors, corpus=k)
            for k, f in CORPORA.items() if k != "eve"}
    data["eve_thirds"] = sentence_thirds(str(ATELIER / CORPORA["eve"]), vectors)
    return data


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(args, cfg, data) -> Dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
        except Exception:
            return None
    files = {f: _sha(ATELIER / f) for f in CORPORA.values()}
    for rel in ("spiraton/experimental/semantic_thermodynamics/observables.py",
                "spiraton/data/semantic_thermo_adapter.py",
                "spiraton/diagnostics/semantic_thermo_probe.py",
                "docs/SEMANTIC_THERMO_PROTOCOLE.md"):
        files[rel] = _sha(REPO / rel)
    try:
        from ..data.tokenizer_bridge import load_native_tokenizer  # noqa
        libs = [p for p in (ATELIER / "Tokenizer" / "bin").glob("libspiratontokenizer.*")]
        for p in libs:
            files["Tokenizer/bin/" + p.name] = _sha(p)
    except Exception:
        pass
    return {
        "commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
        "seeds": args.seeds, "config": asdict(cfg), "files_sha256": files,
        "n_cycles": {k: len(v) for k, v in data.items()},
        "n_perm": N_PERM, "n_boot": N_BOOT, "n_seq_shuffle": N_SEQ_SHUFFLE,
    }


def main(argv: Optional[Sequence[str]] = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", default="all", choices=("e1", "e2", "e3", "all"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=str(REPO / "runs" / "semantic_thermo"))
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",")]
    cfg = ThermoProbeConfig()

    from ..data.tokenizer_bridge import NativeTokenizer33D
    tok = NativeTokenizer33D()
    vectors = CachedVectors(tok.vectors)
    data = load_all(vectors)

    run_dir = Path(args.out) / (time.strftime("%Y%m%d_%H%M%S") + f"_{args.experiment}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest(args, cfg, data), indent=2), encoding="utf-8")
    results: Dict = {}
    exps = ("e1", "e3", "e2") if args.experiment == "all" else (args.experiment,)
    for e in exps:
        results[e] = []
        for s in args.seeds:
            t0 = time.time()
            if e == "e1":
                r = run_e1(data, s, cfg)
            elif e == "e3":
                r = run_e3(data, s, cfg, vectors)
            else:
                r = run_e2(data, s, cfg)
            r["seconds"] = round(time.time() - t0, 1)
            results[e].append(r)
            print(f"[{e}] seed {s} ok ({r['seconds']} s)", flush=True)
            (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
