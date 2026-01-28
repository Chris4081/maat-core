# MAAT-Core (Python)
> **TL;DR**  
> MAAT-Core is a minimal Python framework for experimenting with  
> **optimization under explicit ethical and safety constraints.**  
> It combines classical numerical optimization with formal value fields.

A small, practical foundation for experimenting with a MAAT-style
computation layer:

- **Fields** are weighted scalar functions over a state.
- **Integrate** produces one objective value (weighted field tension + optional regularizers).
- **Seek** finds a low-tension state using local optimization (L-BFGS-B) or global annealing (dual_annealing).
- **S (Creativity)** is modeled as **exploration strength** (temperature), not as a "free lunch" in the objective.

# What can you do with MAAT-Core?

MAAT-Core is a small experimental toolbox for ethical and constrained optimization.
It’s not a black-box AI – it’s a thinking engine for exploring decisions, trade-offs and safety.

## Learn more

- 📘 **Full Documentation:** [DOCUMENTATION.md](DOCUMENTATION.md)  
- 🧪 Examples: `examples/`  
- 🔁 Reproducibility: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)  
- 🧠 Applications: [APPLICATIONS.md](APPLICATIONS.md)

For visual exploration see:
examples/reflection_demo.ipynb

## Typical use cases

## 1. Ethical AI experiments

Model values like Harmony, Risk, Fairness or Cost as fields and let the system search for a solution that balances them – while enforcing hard safety rules.

## 2. Safety-first optimization

Use Respect constraints to define forbidden regions.
The optimizer will never return unsafe solutions – they are mathematically dominated.

## 3. Decision support systems

Prototype multi-criteria decisions:
	•	policy choices
	•	resource allocation
	•	system tuning
	•	planning under constraints

## 4. Research playground

Test ideas like:
	•	How does complexity regularization change solutions?
	•	When do global vs local optimizers behave differently?
	•	How strong must safety penalties be?

## 5. Teaching & demos

Perfect for:
	•	optimization theory
	•	AI ethics
	•	explainable decision systems
	•	interactive notebooks

Mental model

MAAT-Core = “Loss function + Ethics”

Instead of:

Optimize first, filter later

MAAT-Core does:

Safety and values are part of the math itself

If a solution violates Respect, it is not optimal by definition.

## FAQ

**Is this a machine learning library?**  
No. MAAT-Core is a deterministic optimization framework, not a statistical model.

**How is this different from CVXPY or classical optimizers?**  
MAAT-Core makes ethical and safety constraints *first-class mathematical objects*,  
not post-hoc filters.

**Can this scale to neural models?**  
Yes in principle. Fields can wrap neural networks, but MAAT-Core itself stays minimal.


- decision support systems
- ethical optimization research

# Installation Guide — MAAT-Core

This guide explains how to install **MAAT-Core** from GitHub.

## 1) Clone the repository

```bash
git clone https://github.com/Chris4081/maat-core.git
cd maat-core
```

## 2) (Recommended) Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3) Update pip

```bash
python -m pip install -U pip
```

## 4) Install MAAT-Core

### Normal install (for usage)

```bash
python -m pip install .
```

### Editable install (for development)

```bash
python -m pip install -e .
```

### With development extras (if defined)

```bash
python -m pip install -e ".[dev]"
```

## 5) Run examples

```bash
cd examples
python occam_demo.py
python respect_boundary_demo.py
```

## Install directly from GitHub (no clone)

```bash
python -m pip install "git+https://github.com/Chris4081/maat-core.git"
```

Editable from GitHub:

```bash
python -m pip install -e "git+https://github.com/Chris4081/maat-core.git#egg=maat-core"
```

---

## Reproducibility

Show installed versions:

```bash
python -m pip list
```

Freeze environment:

```bash
python -m pip freeze > requirements-lock.txt
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
print(res.x, res.fun)
```

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

## Philosophy

Instead of adding ethics after optimization, MAAT-Core embeds safety
directly into the mathematics. A solution that violates Respect simply
cannot be optimal.

This makes MAAT-Core suitable for:
- AI safety experiments
- autonomous systems

## License

MIT License

Copyright (c) 2026 Christof Krieg


## Citation

If you use MAAT-Core in academic work, research, or publications,  
please consider citing:

Christof Krieg, *MAAT-Core: Safety-First Optimization Framework*, 2026.  
https://github.com/Chris4081/maat-core

