<p align="left">
  <img src="logo.png" width="280"/>
</p>

<h1 align="left">MAAT-Core (Python)</h1>

<p align="left">
A Safety-First Optimization Core for Ethical Decision-Making.
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18489336.svg)](https://doi.org/10.5281/zenodo.18489336)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub last commit](https://img.shields.io/github/last-commit/Chris4081/maat-core)](https://github.com/Chris4081/maat-core)
[![GitHub issues](https://img.shields.io/github/issues/Chris4081/maat-core)](https://github.com/Chris4081/maat-core/issues)
[![GitHub stars](https://img.shields.io/github/stars/Chris4081/maat-core)](https://github.com/Chris4081/maat-core/stargazers)

---

> **TL;DR**  
> MAAT-Core is a minimal Python framework for experimenting with  
> **optimization under explicit ethical and safety constraints.**  
> It combines classical numerical optimization with formal value fields.
> MAAT-Core is an experimental research tool intended for exploration,
teaching, and prototyping — not production systems.

A small, practical foundation for experimenting with a MAAT-style
computation layer:

- **Fields** are weighted scalar functions over a state.
- **Integrate** produces one objective value (weighted field tension + optional regularizers).
- **Seek** finds a low-tension state using local optimization (L-BFGS-B) or global annealing (`dual_annealing`).
- **S (Creativity)** is modeled as **exploration strength** (temperature), not as a "free lunch" in the objective.

---

## What's New — v0.1.1

Version 0.1.1 introduces a new reproducible benchmark showing how the
Critical Coherence Index (CCI) behaves as a structural order parameter
in constrained optimization.

**Highlights:**
- Constraint-induced transition near the unconstrained optimum
- Clear CCI peak at the boundary crossing
- Publication-ready plotting pipeline
- Fully reproducible via the `examples/` folder

**Key files:**
- `examples/cci_critical_transition_demo.py` — simulation
- `examples/plot_cci_transition.py` — figure generation
- `examples/cci_transition_plot.png` — generated figure

**Paper / write-up:**
- `papers/cci_constraint_transition_2026.tex`

**Run the demo:**
```bash
python examples/cci_critical_transition_demo.py
python examples/plot_cci_transition.py
```

---

## Quick example (Respect as Safety-First constraint)

```python
import numpy as np
from maat_core import Field, Constraint, MaatCore

def state_fn(x: float):
    x = float(x)
    return type("State", (), {
        "dissonance": np.sin(np.pi * x) ** 2,
        "complexity": np.exp(x),
        "val": x,
    })

H = Field("Harmony", lambda s: s.dissonance, weight=0.9)

# Respect (R): hard-ish constraint via penalty
R = Constraint("Respect", lambda s: 0.6 - float(s.val))  # enforce x <= 0.6

core = MaatCore([H], constraints=[R], safety_lambda=1e6)

res = core.seek(state_fn, x0=[0.5], S=0.6, use_annealing=True)

print(f"Optimized x: {res.x}")
print(f"Objective value: {res.fun:.4f}")

# Check constraint status
report = core.constraint_report(state_fn(res.x))
print(report)
```

**Example output:**
```
Optimized x: [0.6]
Objective value: 0.0000
Constraint report:
  Respect: margin = 0.0000 (SATISFIED - at boundary)
```

**What happened:**
- The optimizer wanted to minimize Harmony (reduce dissonance)
- But Respect constrained x ≤ 0.6
- Result: optimal solution exactly at the safety boundary
- No violation: ethics are enforced mathematically

---

## What can you do with MAAT-Core?

MAAT-Core is a small experimental toolbox for ethical and constrained optimization.  
It's not a black-box AI – it's a **thinking engine** for exploring decisions, trade-offs and safety.

---

## Typical use cases

### 1. Ethical AI experiments

Model values like Harmony, Risk, Fairness or Cost as fields and let the system search for a solution that balances them – while enforcing hard safety rules.

### 2. Safety-first optimization

Use Respect constraints to define forbidden regions.
The optimizer will never return unsafe solutions – they are mathematically dominated.

### 3. Decision support systems

Prototype multi-criteria decisions:
- policy choices
- resource allocation
- system tuning
- planning under constraints

### 4. Research playground

Test ideas like:
- How does complexity regularization change solutions?
- When do global vs local optimizers behave differently?
- How strong must safety penalties be?

### 5. Teaching & demos

Perfect for:
- optimization theory
- AI ethics
- explainable decision systems
- interactive notebooks

---

## Mental model

**MAAT-Core = "Loss function + Ethics"**

Instead of:
> Optimize first, filter later

MAAT-Core does:
> Safety and values are part of the math itself

If a solution violates Respect, it is not optimal by definition.

---

## Learn more

- 📘 **Full Documentation:** [DOCUMENTATION.md](docs/DOCUMENTATION.md)  
- 🧪 **Examples:** See `examples/` directory
  - Healthcare allocation: `examples/healthcare_ethics_demo.py`
  - Truth constraints: `examples/truth_constraints_demo.py`
  - Boundary enforcement: `examples/respect_boundary_demo.py`
  - Occam's razor: `examples/occam_demo.py`
- 🔁 **Reproducibility:** [REPRODUCIBILITY.md](REPRODUCIBILITY.md)  
- 🌐 **Website:** [maat-research.com](https://maat-research.com)

For visual exploration see:
`examples/reflection_demo.ipynb`

---

## Installation Guide

This guide explains how to install **MAAT-Core** from GitHub.

### 1) Clone the repository

```bash
git clone https://github.com/Chris4081/maat-core.git
cd maat-core
```

### 2) (Recommended) Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Update pip

```bash
python -m pip install -U pip
```

### 4) Install MAAT-Core

**Normal install (for usage):**
```bash
python -m pip install .
```

**Editable install (for development):**
```bash
python -m pip install -e .
```

**With development extras (if defined):**
```bash
python -m pip install -e ".[dev]"
```

### 5) Run examples

```bash
cd examples
python occam_demo.py
python respect_boundary_demo.py
```

### Install directly from GitHub (no clone)

```bash
python -m pip install "git+https://github.com/Chris4081/maat-core.git"
```

**Editable from GitHub:**
```bash
python -m pip install -e "git+https://github.com/Chris4081/maat-core.git#egg=maat-core"
```

---

## Reproducibility

**Show installed versions:**
```bash
python -m pip list
```

**Freeze environment:**
```bash
python -m pip freeze > requirements-lock.txt
```

---

## Community

- 💬 **Discussions:** [GitHub Discussions](https://github.com/Chris4081/maat-core/discussions)
- 🐛 **Issues:** [GitHub Issues](https://github.com/Chris4081/maat-core/issues)
- 🌐 **Website:** [maat-research.com](https://maat-research.com)
- 📧 **Contact:** christof.krieg@outlook.de
- 📄 **Paper:** [DOI: 10.5281/zenodo.18489336](https://doi.org/10.5281/zenodo.18489336)

**Contributions welcome!** See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## FAQ

**Is this a machine learning library?**  
No. MAAT-Core is a deterministic optimization framework, not a statistical model.

**How is this different from CVXPY or classical optimizers?**  
MAAT-Core makes ethical and safety constraints *first-class mathematical objects* (margins + diagnostics), not post-hoc filters.

**What does "Respect as a hard constraint" mean here?**  
Constraints are written as margins `g(state) >= 0`. If violated, MAAT-Core applies a strong penalty so unsafe solutions become mathematically dominated.

**What is a "constraint margin"?**  
A signed distance-to-safety value: positive = safe, zero = boundary, negative = violation magnitude. Margins make constraint satisfaction interpretable and auditable.

**What happens if constraints are impossible to satisfy?**  
MAAT-Core reports persistent negative margins and flags **structural infeasibility** instead of returning a "fake ethical" solution.

**How do you handle lower/upper bounds?**  
Two options: (1) optimizer-level box bounds via `bounds`, and/or (2) explicit ethical constraints like `upper - x` and `x - lower` to get margin diagnostics.

**Can bounds be dynamic (data-dependent)?**  
Yes. You can define constraints that depend on context (e.g., waitlist size). Just ensure each constraint returns a numeric margin (not a boolean).

**Why L-BFGS-B and dual annealing?**  
L-BFGS-B is a strong baseline for box-constrained local search; dual annealing provides global exploration. MAAT-Core is optimizer-agnostic—swap engines if needed.

**Can this scale to neural models?**  
Yes in principle. Fields can wrap neural nets (or any black-box function), while MAAT-Core stays minimal and focuses on constraint-first optimization + diagnostics.

**Is this a fairness toolkit?**  
Not specifically. Fairness is one use case. MAAT-Core generalizes to *any* ethical/safety/legal constraint expressible as a margin.

**What is MAAT-Core for?**  
Decision support prototypes, safety/ethics research, constraint diagnostics, and transparent trade-off exploration.

---

## Design notes

- Respect (R) is modeled as a constraint (Safety-First).
- Unsafe states receive a large quadratic penalty.
- Works with both local and global optimizers.
- Can later be extended with:
  - true SciPy constraints
  - projection methods
  - multi-dimensional states
  - symbolic or neural fields

---

## Philosophy

Instead of adding ethics after optimization, MAAT-Core embeds safety
directly into the mathematics. A solution that violates Respect simply
cannot be optimal.

This makes MAAT-Core suitable for:
- AI safety experiments
- autonomous systems
- ethical decision support
- transparent constraint diagnostics

---

## License

MIT License

Copyright (c) 2025 Christof Krieg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Citation

If you use MAAT-Core in your research, please cite:

**Paper:**
```
Christof Krieg (2026).  
Respect as a Hard Constraint in Ethical Decision-Making: 
A Safety-First Optimization Core (MAAT-Core).
DOI: https://doi.org/10.5281/zenodo.18489336
```

**BibTeX:**
```bibtex
@misc{krieg2026maatcore,
  title={Respect as a Hard Constraint in Ethical Decision-Making: 
         A Safety-First Optimization Core (MAAT-Core)},
  author={Krieg, Christof},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18489336},
  url={https://doi.org/10.5281/zenodo.18489336}
}
```


---

## Acknowledgments

Thanks to the international research community for early engagement and feedback, including readers from University of Hong Kong, Norwegian University of Science and Technology, University of Amsterdam, University of Zadar, and many others.

Special thanks to active contributors on GitHub Discussions for testing and suggestions.

---

**⭐ If you find MAAT-Core useful, please star the repository!**

**🚀 Ready to explore ethical optimization? Start with the examples!**
