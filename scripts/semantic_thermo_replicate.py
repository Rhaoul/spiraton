"""R4 — réplication pré-enregistrée du signal E1 sous CamemBERT.

Protocole : docs/SEMANTIC_THERMO_R4_REPLICATION.md (figé avant le corpus).

    validate <corpus>                       cahier des charges, AUCUNE mesure
    sha <corpus>                            sha256 à consigner avant mesure
    run <corpus> --expect-sha <sha> [--out f.json] [--any-plan]
                                            pipeline R3 gelé, 5 graines

``--any-plan`` n'est permis que pour le contrôle d'intégrité (§6 : horscanon).
Lancer depuis la racine du dépôt spiraton : HF_HUB_OFFLINE=1 python scripts/...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from spiraton.data.aba import AbaParseError, try_parse_aba_line  # noqa: E402
from spiraton.data.aba_forms import classify_form_extended  # noqa: E402

FORMS = ("F0", "F0b", "F1", "F1b", "F2", "F3")
OPS = ("ADD", "SUB", "MUL", "DIV")
N_PER_FORM, N_PER_FORM_OP = 24, 6
MIN_W, MAX_W = 3, 7
OTHER_CORPORA = ("dataset_aba.txt", "corpus_claude_aba.txt", "corpus_horscanon_aba.txt",
                 "corpus_f0b_aba.txt", "corpus_eve_clean.txt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def validate(path: Path) -> list:
    problems = []
    cycles = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            c = try_parse_aba_line(line)
        except AbaParseError as exc:
            problems.append(f"l.{i} : ligne non parseable ({str(exc)[:60]})")
            continue
        if c is None:
            continue
        segs = list(c.segments.values())
        pairs = {(s.chirality, s.direction) for s in segs}
        if not pairs <= {("DX", "OUT"), ("LV", "IN")}:
            problems.append(f"l.{i} : orientation hors grammaire {sorted(pairs)}")
            continue
        if len({s.op for s in segs}) != 1:
            problems.append(f"l.{i} : opérateur non constant")
        lens = [len(s.text.split()) for s in segs]
        for name, L in zip(("A", "B", "A′"), lens):
            if not MIN_W <= L <= MAX_W:
                problems.append(f"l.{i} : segment {name} de {L} mots (attendu {MIN_W}-{MAX_W})")
        form = classify_form_extended([+1 if s.chirality == "DX" else -1 for s in segs], lens)
        if form not in FORMS:
            problems.append(f"l.{i} : forme {form} hors plan")
        cycles.append((i, c.op, form, " ".join(s.text for s in segs)))
    fc = Counter(f for _, _, f, _ in cycles)
    for f in FORMS:
        if fc[f] != N_PER_FORM:
            problems.append(f"forme {f} : {fc[f]} cycles (attendu {N_PER_FORM})")
        oc = Counter(op for _, op, ff, _ in cycles if ff == f)
        for op in OPS:
            if oc[op] != N_PER_FORM_OP:
                problems.append(f"forme {f}, opérateur {op} : {oc[op]} (attendu {N_PER_FORM_OP})")
    seen = {}
    for i, _, _, t in cycles:
        k = _norm(t)
        if k in seen:
            problems.append(f"l.{i} : doublon de la l.{seen[k]}")
        seen.setdefault(k, i)
    existing = set()
    for name in OTHER_CORPORA:
        p = REPO.parent / name
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                try:
                    c = try_parse_aba_line(line)
                except AbaParseError:
                    c = None                      # phrase non-ABA (eve) : texte brut
                txt = " ".join(s.text for s in c.segments.values()) if c else line
                existing.add(_norm(txt))
    for i, _, _, t in cycles:
        if _norm(t) in existing:
            problems.append(f"l.{i} : phrase déjà présente dans un corpus existant")
    return problems


def run(path: Path, expect_sha: str, any_plan: bool) -> dict:
    import torch
    from spiraton.data.contextual_vectors import ContextualWordVectors
    from spiraton.data.semantic_thermo_adapter import CachedVectors, load_cycles, split_cycles
    from spiraton.diagnostics.semantic_thermo_probe import (
        ATELIER, REP_CAMEMBERT, _e1_block, auc, build_field, measure_cycles)
    from spiraton.experimental.semantic_thermodynamics import ThermoProbeConfig

    got = sha256(path)
    if got != expect_sha:
        raise SystemExit(f"REFUS : sha256 {got} ≠ attendu {expect_sha} (corpus non gelé)")
    if not any_plan:
        pb = validate(path)
        if pb:
            raise SystemExit("REFUS : corpus non valide :\n  " + "\n  ".join(pb))
    enc = ContextualWordVectors("camembert-base",
                                revision="a75967561c78f2aa81cc41045378d3b4ee25af9e")
    vec = CachedVectors(enc, dim=enc.hidden, word_level=True)
    ds = load_cycles(str(ATELIER / "dataset_aba.txt"), vec, corpus="dataset", tokenize="sentence")
    rc = load_cycles(str(path), vec, corpus="r4", tokenize="sentence")
    cfg = ThermoProbeConfig()
    res = {"corpus": str(path.name), "sha256": got, "n_cycles": len(rc),
           "model_revision": enc.commit, "seeds": []}
    for seed in range(5):
        ci, _ = split_cycles(len(ds), seed=seed)
        norm, field = build_field([ds[i] for i in ci], "all", cfg, rep=REP_CAMEMBERT)
        g = torch.Generator().manual_seed(1000 * seed + 17)
        ms, _ = measure_cycles(rc, norm, field)
        r = {"seed": seed,
             "P1_strat_div": _e1_block(ms, "div", strata="pos", gen=g),
             "len_strat": _e1_block(ms, "n", strata="pos", gen=g, with_ci=False),
             "P2_position": _e1_block(ms, "div", strata="chir", gen=g, label="notprime")}
        ms2 = [m for m in ms if m.n >= 2]
        by_pos = {}
        for q, name in enumerate(("A", "B", "A_prime")):
            sub = [m for m in ms2 if m.pos == q]
            lab = torch.tensor([m.chir for m in sub], dtype=torch.float64)
            if 0 < int(lab.sum()) < len(sub):
                by_pos[name] = _e1_block(sub, "div", strata="none", gen=g, n_perm=2000)
        r["P2_by_position"] = by_pos
        res["seeds"].append(r)
        p1 = r["P1_strat_div"]
        print(f"seed {seed} : AUC {p1['auc']:.3f} IC [{p1['ci_low']:.3f}, {p1['ci_high']:.3f}] "
              f"p {p1['p_perm_two_sided']:.4f} | longueur p {r['len_strat']['p_perm_two_sided']:.4f} | "
              + " ".join(f"{k} {v['auc']:.3f}" for k, v in by_pos.items()), flush=True)
    s = res["seeds"]
    ci_up = sum(x["P1_strat_div"]["ci_low"] > 0.5 for x in s)
    p_ok = sum(x["P1_strat_div"]["p_perm_two_sided"] < 0.01 for x in s)
    len_ok = all(x["len_strat"]["p_perm_two_sided"] >= 0.01 for x in s)
    ci_incl = sum(x["P1_strat_div"]["ci_low"] <= 0.5 <= x["P1_strat_div"]["ci_high"] for x in s)
    below = any(x["P1_strat_div"]["auc"] < 0.5 for x in s)
    if ci_up == 5 and p_ok == 5 and len_ok:
        verdict = "RÉPLIQUÉ"
    elif ci_incl >= 3 or below:
        verdict = "NON RÉPLIQUÉ"
    else:
        verdict = "INTERMÉDIAIRE"
    res["verdict"] = {"verdict": verdict, "ci_excludes_05_above": ci_up, "p_lt_001": p_ok,
                      "length_not_significant": len_ok, "ci_includes_05": ci_incl}
    print(json.dumps(res["verdict"], ensure_ascii=False))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("validate", "sha", "run"))
    ap.add_argument("corpus")
    ap.add_argument("--expect-sha")
    ap.add_argument("--out")
    ap.add_argument("--any-plan", action="store_true")
    a = ap.parse_args(argv)
    path = Path(a.corpus).resolve()
    if a.mode == "sha":
        print(sha256(path))
    elif a.mode == "validate":
        pb = validate(path)
        print("VALIDE" if not pb else f"{len(pb)} problème(s) :\n  " + "\n  ".join(pb))
        return 0 if not pb else 1
    else:
        if not a.expect_sha:
            raise SystemExit("REFUS : --expect-sha requis (gel avant mesure)")
        res = run(path, a.expect_sha, a.any_plan)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
