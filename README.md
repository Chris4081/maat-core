# MAAT-Core (Python)

A small, practical foundation for experimenting with a MAAT-style
computation layer:

-   **Fields** are weighted scalar functions over a state.\
-   **Integrate** produces one objective value (weighted field tension +
    optional regularizers).\
-   **Seek** finds a low-tension state using local optimization
    (L-BFGS-B) or global annealing (dual_annealing).\
-   **S (Creativity)** is modeled as **exploration strength**
    (temperature), not as a "free lunch" in the objective.

## Install (editable)

``` bash
cd maat-core
python -m pip install -e .
```

## Quick example (Respect as Safety-First constraint)

``` python
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
# Convention: Constraint(...) returns a **margin**; it's satisfied when margin >= 0.
R = Constraint("Respect", lambda s: 0.6 - float(s.val))  # enforce x <= 0.6

core = MaatCore([H], constraints=[R], safety_lambda=1e6)

res = core.seek(state_fn, x0=[0.5], S=0.6, use_annealing=True)
print(res.x, res.fun)
```

## Design notes

-   **Respect (R)** is modeled as a constraint (Safety-First).\
    Unsafe states receive a large quadratic penalty, so they become
    numerically unstable minima.
-   Works with both local and global optimizers.
-   Can later be extended with:
    -   true SciPy constraints
    -   projection methods
    -   multi-dimensional states
    -   symbolic or neural fields

## Philosophy

Instead of adding ethics *after* optimization, MAAT-Core embeds safety
directly into the mathematics. A solution that violates Respect simply
cannot be optimal.

This makes MAAT-Core suitable for: - AI safety experiments - autonomous
systems - decision support systems - ethical optimization research

## License

MIT
