# Spiraton Model Card (Library)

> This is a **library model card** for the Spiraton reference cells (PyTorch).
> Generated automatically from the codebase by `scripts/gen_model_card.py`.

## Overview
**Project:** Spiraton
**Version:** 0.2.0-dev _(single source: `spiraton/__init__.py`)_
**Torch requirement:** >=2.0 (see `pyproject.toml`)

> This card is a **pure function of the tracked source** (docstrings, version,
> module layout): no wall-clock timestamp, git hash, or environment-specific
> versions are embedded, so the CI check `git diff --exit-code MODEL_CARD.md`
> only flags a *real* drift between code and card.

Spiraton provides operator-based computation cells with:
- **Mode selection**: dextrogyre vs levogyre (per-sample)
- **Operators**: additive, subtractive, multiplicative (log-domain), divisive (inverse log-domain)
- A stable **core** cell and an **experimental** gated/weighted variant.

## Intent & Governance (Integral to the Model)
Spiraton’s behavior is not only code-level: its **intended direction, refusals, and contribution norms** are treated as part of the model definition.

- **Manifesto:** `MANIFESTE.md`
  - Excerpt: Manifeste
- **Refusals / Non-goals:** `REFUS.md`
  - Excerpt: Ce projet refuse de devenir :
- **Code of Conduct:** `CODE_OF_CONDUCT.md`
  - Excerpt: Ce projet repose sur une idée simple : la nuance est fragile, et le bruit est facile.

## Intended Use
- Research / prototyping of operator-composed cells
- System dynamics style compositions (add/sub/mul/div in log-domain)
- Testing “mode-conditioned” nonlinearities

## Non-Intended Use
- Not a drop-in replacement for standard RNN/LSTM/Transformer layers without validation
- Not intended as a safety-critical decision system
- Uses conflicting with the project’s refusal constraints are out of scope by design

## Public API (v0.1)
The v0.1 public surface is intentionally small and stable:
- `from spiraton import SpiratonCell, GatedSpiratonCell`

Everything else (`spiraton.core.*`, `spiraton.experimental.*`) is accessible but considered **semi-public** unless explicitly stated.

## Core Cell: `SpiratonCell`
SpiratonCell (CORE)
- Canon operators: add/sub/mul/div
- Mode selection:
    - default: dextro_mask(inputs) (mean>=0)
    - optional: mode_policy(inputs) returning hard mask (bool) or soft gate (float in [0,1])

### Core behavior
- Composition: `raw_dextro = add + mul - div`, `raw_levogyre = sub + div - mul`
- Activation: `tanh` in dextro mode, `atan` in levogyre mode
- `mul/div` are stabilized with `tanh(log_*)`

## Experimental Cell: `GatedSpiratonCell`
Experimental gated cell:
- log-domain mul/div + tanh stabilization
- learned gates per operator
- learned global coeffs (sigmoid)
- optional mode_policy: hard mask or soft gate

### Experimental behavior
- Learnable **gates** per operator (sigmoid projection)
- Learnable global **coeffs** for add/sub/mul/div (sigmoid-bounded)
- Optional adaptation heuristic available via `second_order_adjust()`

## Experimental & Ecosystem Modules (CLAUDE.md chantiers)
These extend the codebase toward the three testable axioms of the Logos theory
(contextuality, double dynamics, second-order memory) and the ABA data path.
They are **experimental / semi-public** — the v0.1 public API above is unchanged.

- **Chantier 1 — contextualité (poids matriciels)** — `spiraton.experimental.matrix_cell.MatrixSpiratonCell`
  - Cellule Spiraton à poids matriciels — incarnation de la contextualité.
- **Chantier 4 — embeddings par opérateur×chiralité** — `spiraton.experimental.operator_embedding.OperatorEmbedding`
  - Représentation conditionnée par l'opérateur × chiralité (chantier 4).
- **Chantier 6 — dynamique du second ordre** — `spiraton.experimental.chrono.ChronoSpiraton`
  - Dynamique du second ordre du Logos (THEORIE_LOGOS §3.2, chantier 6).
- **Chantier 2 — diagnostic L∘D vs D∘L** — `spiraton.diagnostics.double_dynamics.run_double_dynamics`
  - Mesure l'asymétrie L∘D vs D∘L sur un même système (chantier 2).
- **Chantier 3 — parseur ABA de référence** — `spiraton.data.aba.parse_aba_line`
  - Parse strictement une ligne ABA en :class:`AbaCycle`.
- **Chantier 3 — pont tokenizer 33D (avec repli)** — `spiraton.data.tokenizer_bridge.NativeTokenizer33D`
  - Adaptateur stable autour de ``SpiratonTokenizerV4``.
- **Chantier 5 — perte de clôture spirale** — `spiraton.training.aba_loss.alpha_omega_loss`
  - Perte de clôture spirale : A′ *proche-et-aligné* avec A, mais non identique.
- **Chantier 7 — traits phonémiques réels (dims 8-22) en canaux d'entrée** — `spiraton.data.featurizers.PhonemeFeaturizer`
  - Featurizer adossé au tokenizer 33D natif — le livrable du chantier 7.

## Operators
- `additive()`
- `subtractive()`
- `multiplicative()`
- `divisive()`
Notes:
- `multiplicative/divisive` operate in **log(|x|+eps)** space for numerical stability.

## Mode Logic
- `dextro_mask()`
Current rule:
- **dextro** iff `inputs.mean(dim=-1) >= 0`

## Evaluation
- Unit tests under `tests/` validate:
  - shapes (single vs batch)
  - finiteness (no NaN/Inf under typical inputs)
  - exact formula match under forced parameters
  - gradient flow

## Limitations
- The dextro/levogyre mode rule is currently a simple mean-threshold; you may want to replace it for real tasks.
- Log-domain operators depend on `eps` and absolute values; this changes behavior near zero.
- No training recipe is shipped (cells are building blocks).

## Reproducibility
- The repository includes deterministic unit tests using fixed seeds.
- CI runs CPU-only tests across supported Python versions.

## License
See `LICENSE`.

---
*This file is generated by `scripts/gen_model_card.py`. Do not edit by hand.*
