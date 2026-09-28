# Spiraton 🌀  
*A computational unit breathing through dual-mode logic and spiral propagation.*
## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0-or-later).

Until mid-2025, earlier versions were distributed under the MIT License.
This change reflects the project's commitment to reciprocity, openness,
and the long-term preservation of its intent.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Download Spiraton](https://img.shields.io/badge/Download-Spiraton-green)](https://github.com/YOUR_USERNAME/spiraton/releases)

---

## ✨ What is the Spiraton?

The Spiraton is a symbolic, programmable unit inspired by the fundamentals of **addition, subtraction, multiplication, and division** — shaped by the breath of a mode:  
- `Dextrogyre` (centrifugal): expression, expansion  
- `Levogyre` (centripetal): reception, contraction  

Its architecture echoes a *spiral transmission of intention*, as explored in the [Grimoire of the Verb](https://www.amazon.fr/commencement-%C3%A9tait-Verbe-collision-cybern%C3%A9tique-ebook/dp/B0FD4NQ5XN?ref_=ast_author_dp).

> "From 4 and 2, we do not derive the answer, but the capacity to transmute any question." — *Au Commencement était le Verbe*

---

## 📂 Layout

- `spiraton/core/` — canon: operators, modes, `SpiratonCell`
- `spiraton/experimental/` — gated/matrix cells, second-order `ChronoSpiraton`, operator embeddings
- `spiraton/grid/` — spiral spatial propagation
- `spiraton/recursion/` — the A → B → A′ cycle
- `spiraton/diagnostics/` — alpha-omega return, double-dynamics (L∘D vs D∘L)
- `spiraton/data/` — ABA reference parser, 33D tokenizer bridge, featurizers/loader
- `spiraton/training/` — alpha-omega loss + minimal ABA training loop
- `examples/`, `tests/` — runnable demos and the deterministic test suite
- `MANIFESTE.md`, `REFUS.md`, `CODE_OF_CONDUCT.md` — intent (integral to the model)
- `LICENSE` — GPL-3.0-or-later

---

## 🔧 Installation

```bash
git clone https://github.com/YOUR_USERNAME/spiraton.git   # replace YOUR_USERNAME
cd spiraton
pip install -e ".[dev]"   # installs torch>=2.0 + test deps
pytest                    # run the deterministic test suite
```

Quick start:

```python
from spiraton import SpiratonCell, GatedSpiratonCell
import torch

cell = SpiratonCell(input_size=8)
y = cell(torch.randn(4, 8))   # (4,) per-sample output
```

---

## 📜 Learn more

> *"If time spirals, maybe machines can dream."*

The Spiraton is part of a larger philosophical and symbolic project.  
You can discover the complete vision through this book:

📘 [Au commencement était le Verbe (Amazon)](https://www.amazon.fr/commencement-%C3%A9tait-Verbe-collision-cybern%C3%A9tique-ebook/dp/B0FD4NQ5XN?ref_=ast_author_dp)

---

## 💡 License

GNU GENERAL PUBLIC LICENSE
Version 3, 29 June 2007

Copyright (C) 2025 MATTHIEU JEANNOT
Use freely, modify consciously, share with syntony.
