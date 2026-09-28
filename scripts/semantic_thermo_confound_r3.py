"""Contrôle de confond POST HOC (déclaré après lecture du run R3) de E1-strat sous CamemBERT.

Question : l'effet chiralité → divergence à position fixée vient-il du corpus d'origine
(horscanon T31 vs f0b T36, dont les chiralités diffèrent à position B) ? Trois tests :
strates = position ; strates = position × corpus ; horscanon seul (DX et LV à chaque
position). Usage : HF_HUB_OFFLINE=1 python scripts/semantic_thermo_confound_r3.py <sortie.json>
"""
import json
import sys

import torch

from spiraton.data.contextual_vectors import ContextualWordVectors
from spiraton.data.semantic_thermo_adapter import CachedVectors, load_cycles, split_cycles
from spiraton.diagnostics.semantic_thermo_probe import (
    ATELIER, REP_CAMEMBERT, auc, boot_auc_by_cycle, build_field, measure_cycles, perm_null_auc,
)
from spiraton.experimental.semantic_thermodynamics import ThermoProbeConfig

enc = ContextualWordVectors("camembert-base")
vec = CachedVectors(enc, dim=enc.hidden, word_level=True)
load = lambda f, k: load_cycles(str(ATELIER / f), vec, corpus=k, tokenize="sentence")
ds, hc, fb = load("dataset_aba.txt", "dataset"), load("corpus_horscanon_aba.txt", "horscanon"), load("corpus_f0b_aba.txt", "f0b")
cfg = ThermoProbeConfig()
out = []
for seed in range(5):
    ci, _ = split_cycles(len(ds), seed=seed)
    norm, field = build_field([ds[i] for i in ci], "all", cfg, rep=REP_CAMEMBERT)
    g = torch.Generator().manual_seed(1000 * seed + 17)
    res = {"seed": seed}

    def block(cycles, corpus_ids, strata_kind, name):
        ms, _ = measure_cycles(cycles, norm, field)
        keep = [i for i, m in enumerate(ms) if m.n >= 2]
        ms = [ms[i] for i in keep]
        sc = torch.tensor([m.div for m in ms], dtype=torch.float64)
        lab = torch.tensor([m.chir for m in ms], dtype=torch.float64)
        corp = torch.tensor([corpus_ids[m.cycle] for m in ms])
        pos = torch.tensor([m.pos for m in ms])
        st = pos if strata_kind == "pos" else pos * 10 + corp
        a = float(auc(sc, lab))
        null = perm_null_auc(sc, lab, st, n=10_000, gen=g)
        p = float(((null - 0.5).abs() >= abs(a - 0.5)).double().sum() + 1) / 10_001
        lo, hi = boot_auc_by_cycle(sc, lab, torch.tensor([m.cycle for m in ms]), n=2000, gen=g)
        # AUC par position (descriptif)
        per_pos = {}
        for q in range(3):
            sel = pos == q
            if 0 < int(lab[sel].sum()) < int(sel.sum()):
                per_pos[q] = round(float(auc(sc[sel], lab[sel])), 3)
        res[name] = {"auc": round(a, 3), "p_two": round(p, 4), "ci": [round(lo, 3), round(hi, 3)],
                     "n_dx": int(lab.sum()), "n_lv": int((1 - lab).sum()), "auc_by_pos": per_pos}

    block(hc + fb, [0] * len(hc) + [1] * len(fb), "pos", "mixed_strata_pos")
    block(hc + fb, [0] * len(hc) + [1] * len(fb), "pos_corpus", "mixed_strata_pos_x_corpus")
    block(hc, [0] * len(hc), "pos", "horscanon_only_strata_pos")
    # f0b seul : chiralité ≡ position (A=DX, B/A′=LV) — donné pour mémoire, non informatif
    out.append(res)
    print(json.dumps(res, ensure_ascii=False), flush=True)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=2, ensure_ascii=False)
