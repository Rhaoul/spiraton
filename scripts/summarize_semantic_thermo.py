"""Résume un run de ``semantic_thermo_probe`` en tables Markdown (summary.md du run).

Usage : python scripts/summarize_semantic_thermo.py runs/semantic_thermo/<run>
"""
import json
import statistics as st
import sys
from pathlib import Path


def f(x, n=3):
    return "—" if x is None else f"{x:+.{n}f}" if isinstance(x, float) and x < 0 else f"{x:.{n}f}"


def e1_tables(runs):
    out = ["## E1 — DX/LV ↔ divergence (AUC = P(score DX > score LV))", ""]
    keys = [("canon_dataset_div", "canon dataset (≡ position)"),
            ("canon_claude_div", "canon claude (≡ position)"),
            ("strat_mixed_div", "**strat. horscanon+f0b (chiralité à position fixée)**"),
            ("pos_mixed_div", "position A/B vs A′ à chiralité fixée"),
            ("canon_dataset_density", "ρ̄ canon dataset"),
            ("strat_mixed_density", "ρ̄ strat."),
            ("canon_dataset_len", "longueur seule, canon (CTRL-LEN)"),
            ("strat_mixed_len", "longueur seule, strat. (CTRL-LEN)"),
            ("canon_dataset_divJ", "div^J littérale (exclue, E0)")]
    for sl in ("no-logos", "full", "no-energy", "phoneme", "context"):
        out += [f"### tranche `{sl}`", "",
                "| mesure | " + " | ".join(f"g{r['seed']}" for r in runs) + " | IC 95 % (g0) | p perm (médiane) |",
                "|---|" + "---|" * (len(runs) + 2)]
        for k, name in keys:
            vals = [r[sl][k] for r in runs]
            ci = vals[0].get("ci_low")
            ci_s = f"[{vals[0]['ci_low']:.3f}, {vals[0]['ci_high']:.3f}]" if ci is not None else "—"
            p = st.median(v["p_perm_two_sided"] for v in vals)
            out.append(f"| {name} | " + " | ".join(f"{v['auc']:.3f}" for v in vals) + f" | {ci_s} | {p:.4f} |")
        if sl == "no-logos":
            out.append("")
            out.append("Contrôles (no-logos) : " + "; ".join(
                f"g{r['seed']} CTRL-2 dataset {r[sl]['ctrl2_seq_shuffle_dataset_auc_mean']:.3f} / mixte "
                f"{r[sl]['ctrl2_seq_shuffle_mixed_auc_mean']:.3f} ; CTRL-7 orth {r[sl]['ctrl7_orth_square']['canon_dataset_div_auc']:.3f}"
                f"/{r[sl]['ctrl7_orth_square']['strat_mixed_div_auc']:.3f}, JL {r[sl]['ctrl7_jl_half']['canon_dataset_div_auc']:.3f}"
                f"/{r[sl]['ctrl7_jl_half']['strat_mixed_div_auc']:.3f}" for r in runs))
        out.append("")
    ci_ok = [(r["no-logos"]["strat_mixed_div"]["ci_low"], r["no-logos"]["strat_mixed_div"]["ci_high"]) for r in runs]
    out.append("IC bootstrap de l'AUC strat. (no-logos) par graine : " + ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in ci_ok))
    ci_c = [(r["no-logos"]["canon_dataset_div"]["ci_low"], r["no-logos"]["canon_dataset_div"]["ci_high"]) for r in runs]
    out.append("")
    out.append("IC bootstrap de l'AUC canon dataset (no-logos) par graine : " + ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in ci_c))
    return out + [""]


def e3_tables(runs):
    out = ["## E3 — cycle A → B → A′ (tests appariés unilatéraux, Holm sur 5)", ""]
    hyps = ("H3.1a_rhoA_gt_rhoB", "H3.1d_DB_gt_DA", "H3.2a_rhoAp_gt_rhoB", "H3.2d_DB_gt_DAp", "H3.3_dAB_gt_dAAp")
    for sl in ("no-logos", "phoneme", "context", "no-energy"):
        out += [f"### tranche `{sl}`", "",
                "| hypothèse | condition | frac>0 (g0…g4) | médiane Δ (g0) [IC] | p Holm (max sur graines) |",
                "|---|---|---|---|---|"]
        for h in hyps:
            for cond in ("dataset_test", "ctrl_cut", "ctrl_eve_thirds", "claude"):
                vals = [r[sl][cond][h] for r in runs]
                fr = " ".join(f"{v['frac_positive']:.2f}" for v in vals)
                v0 = vals[0]
                out.append(f"| {h} | {cond} | {fr} | {f(v0['median'])} [{f(v0['ci_low'])}, {f(v0['ci_high'])}] | "
                           f"{max(v['p_holm'] for v in vals):.2g} |")
        r0 = runs[0][sl]["dataset_test"]
        out += ["", f"n cycles test = {r0['n_cycles']} ; longueurs moyennes (tokens) {r0['mean_n']} ; "
                f"part d(A,A′) > 0 = {r0['frac_dAAp_positive']:.3f}", ""]
    fm = runs[0]["no-logos"]["forms_mixed"]
    cols = ("n", "rho_A", "rho_B", "rho_Ap", "D_A", "D_B", "D_Ap", "distance_A_B", "distance_A_Ap", "return_ratio")
    out += ["### H3.4 — formes (horscanon + f0b, graine 0, descriptif)", "",
            "| forme | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, v in fm.items():
        out.append(f"| {k} | " + " | ".join(str(v[c]) if c == "n" else f"{v[c]:.3f}" for c in cols) + " |")
    fd = runs[0]["no-logos"]["forms_dataset_test"]
    for k, v in fd.items():
        out.append(f"| {k} (dataset test) | " + " | ".join(str(v[c]) if c == "n" else f"{v[c]:.3f}" for c in cols) + " |")
    return out + [""]


def e2_tables(runs):
    out = ["## E2 — opérateur depuis les observables (accuracy test)", ""]
    for sl in ("no-logos", "phoneme"):
        out += [f"### tranche `{sl}`", "", "| modèle | dataset test (g0…g4) | claude (g0…g4) |", "|---|---|---|"]
        for kind in ("majority", "length", "thermo", "raw_mean"):
            ds = " ".join(f"{r[sl][kind]['dataset_test']['accuracy']:.3f}" for r in runs)
            cl = " ".join(f"{r[sl][kind]['claude']['accuracy']:.3f}" for r in runs)
            out.append(f"| {kind} | {ds} | {cl} |")
        pm = " ".join(f"{r[sl]['thermo']['ctrl1_perm_dataset_test']['mean']:.3f}/{r[sl]['thermo']['ctrl1_perm_dataset_test']['max']:.3f}" for r in runs)
        out.append(f"| thermo, étiquettes permutées (moy/max de 20) | {pm} | — |")
        ll = " ".join(f"{r[sl]['thermo']['dataset_test']['log_loss']:.3f}" for r in runs)
        out += ["", f"log-loss thermo dataset test : {ll} (uniforme = {__import__('math').log(4):.3f})",
                f"accuracy par classe thermo g0 : {runs[0][sl]['thermo']['dataset_test']['per_class_acc']}", ""]
    return out


def main(run_dir):
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    lines = [f"# Résumé — {run_dir.name}", ""]
    if "e1" in m:
        lines += e1_tables(m["e1"])
    if "e3" in m:
        lines += e3_tables(m["e3"])
    if "e2" in m:
        lines += e2_tables(m["e2"])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
