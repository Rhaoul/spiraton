from __future__ import annotations

import datetime as _dt
import importlib
import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "MODEL_CARD.md"


# --- helpers -------------------------------------------------------------

def _run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, cwd=ROOT, stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_info() -> dict[str, Optional[str]]:
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": _run(["git", "describe", "--tags", "--always", "--dirty"]),
    }


def _get_version(pkg: Any) -> str:
    v = getattr(pkg, "__version__", None)
    if isinstance(v, str) and v:
        return v
    try:
        from importlib.metadata import version
        return version("spiraton")
    except Exception:
        return "0.0.0"


def _doc(obj: Any) -> str:
    return (inspect.getdoc(obj) or "").strip()


def _safe_import(name: str) -> Any:
    return importlib.import_module(name)


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _first_nonempty_paragraph(md: str, max_chars: int = 280) -> str:
    """
    Grab a short excerpt from the first meaningful paragraph (non-empty, not a heading).
    Keeps it short so MODEL_CARD stays readable.
    """
    lines = [ln.rstrip() for ln in md.splitlines()]
    buf: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            if buf:
                break
            continue
        # skip headings / horizontal rules
        if s.startswith("#") or s in ("---", "___", "***"):
            continue
        buf.append(s)
        # stop early if already long
        if sum(len(x) for x in buf) > max_chars:
            break
    excerpt = " ".join(buf).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 1].rstrip() + "…"
    return excerpt if excerpt else "_(No excerpt found)_"


def _resolve_policy_file(candidates: list[str]) -> tuple[Optional[Path], Optional[str]]:
    """
    Try multiple filename candidates (e.g., MANIFESTE.md and MANIFESTE.md).
    Returns (path, content) if found.
    """
    for name in candidates:
        p = ROOT / name
        txt = _read_text(p)
        if txt is not None:
            return p, txt
    return None, None


# --- main ---------------------------------------------------------------

def main() -> int:
    sys.path.insert(0, str(ROOT))  # allow local import without install

    pkg = _safe_import("spiraton")
    core_cell_mod = _safe_import("spiraton.core.cell")
    gated_mod = _safe_import("spiraton.experimental.gated_cell")
    ops_mod = _safe_import("spiraton.core.operators")
    modes_mod = _safe_import("spiraton.core.modes")

    version = _get_version(pkg)
    git = _git_info()

    # Torch info (optional)
    try:
        import torch  # type: ignore
        torch_version = torch.__version__
    except Exception:
        torch_version = "not installed"

    SpiratonCell = getattr(core_cell_mod, "SpiratonCell")
    GatedSpiratonCell = getattr(gated_mod, "GatedSpiratonCell")

    core_doc = _doc(SpiratonCell) or "_(No docstring found)_"
    gated_doc = _doc(GatedSpiratonCell) or "_(No docstring found)_"

    operators_doc = "\n".join(
        f"- `{name}()`"
        for name in ["additive", "subtractive", "multiplicative", "divisive"]
        if hasattr(ops_mod, name)
    ) or "- _(operators not found)_"

    modes_doc = "\n".join(
        f"- `{name}()`"
        for name in ["dextro_mask"]
        if hasattr(modes_mod, name)
    ) or "- _(modes not found)_"

    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Intent & governance files (integral part of the model) ----------
    manifeste_path, manifeste_txt = _resolve_policy_file(["MANIFESTE.md", "MANIFESTE.md"])
    refus_path, refus_txt = _resolve_policy_file(["REFUS.md", "REFUS.md"])
    coc_path, coc_txt = _resolve_policy_file(["CODE_OF_CONDUCT.md", "CODE_OF_CONDUCT.md"])

    missing: list[str] = []
    if manifeste_txt is None:
        missing.append("MANIFESTE.md")
    if refus_txt is None:
        missing.append("REFUS.md")
    if coc_txt is None:
        missing.append("CODE_OF_CONDUCT.md")

    # We don't hard-fail generation, but we *surface* missing files loudly
    missing_block = ""
    if missing:
        missing_block = (
            "\n\n⚠️ **Missing governance files** (expected at repo root):\n"
            + "\n".join([f"- `{m}`" for m in missing])
            + "\n\nThese files are considered part of the Spiraton model definition."
        )

    manifeste_excerpt = _first_nonempty_paragraph(manifeste_txt or "")
    refus_excerpt = _first_nonempty_paragraph(refus_txt or "")
    coc_excerpt = _first_nonempty_paragraph(coc_txt or "")

    def _rel(p: Optional[Path]) -> str:
        return str(p.relative_to(ROOT)) if p else "_(missing)_"

    content = f"""# Spiraton Model Card (Library)

> This is a **library model card** for the Spiraton reference cells (PyTorch).  
> Generated automatically from the codebase.

## Overview
**Project:** Spiraton  
**Version:** {version}  
**Generated:** {now}  
**Torch:** {torch_version}  
**Git:** {git.get('describe') or ''}  
**Commit:** {git.get('commit') or ''}  
**Branch:** {git.get('branch') or ''}

Spiraton provides operator-based computation cells with:
- **Mode selection**: dextrogyre vs levogyre (per-sample)
- **Operators**: additive, subtractive, multiplicative (log-domain), divisive (inverse log-domain)
- A stable **core** cell and an **experimental** gated/weighted variant.

## Intent & Governance (Integral to the Model)
Spiraton’s behavior is not only code-level: its **intended direction, refusals, and contribution norms** are treated as part of the model definition.

- **Manifesto:** `{_rel(manifeste_path)}`
  - Excerpt: {manifeste_excerpt}
- **Refusals / Non-goals:** `{_rel(refus_path)}`
  - Excerpt: {refus_excerpt}
- **Code of Conduct:** `{_rel(coc_path)}`
  - Excerpt: {coc_excerpt}{missing_block}

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
{core_doc}

### Core behavior
- Composition: `raw_dextro = add + mul - div`, `raw_levogyre = sub + div - mul`
- Activation: `tanh` in dextro mode, `atan` in levogyre mode
- `mul/div` are stabilized with `tanh(log_*)`

## Experimental Cell: `GatedSpiratonCell`
{gated_doc}

### Experimental behavior
- Learnable **gates** per operator (sigmoid projection)
- Learnable global **coeffs** for add/sub/mul/div (sigmoid-bounded)
- Optional adaptation heuristic available via `second_order_adjust()`

## Operators
{operators_doc}
Notes:
- `multiplicative/divisive` operate in **log(|x|+eps)** space for numerical stability.

## Mode Logic
{modes_doc}
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
"""

    OUT.write_text(content, encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
