# Healthcare Benchmark: MAAT-Core vs SciPy

Comparison of MAAT-Core against native SciPy constrained optimizers
on a bed-allocation problem with fairness constraints.

Christof Krieg — MAAT Research

---

## Problem

Allocate hospital beds across three departments to **maximize lives saved**
while respecting fairness and capacity constraints.

| Department | Lives saved per bed | Minimum beds |
|------------|-------------------|--------------|
| COVID      | 5.0               | 50           |
| Heart      | 3.0               | 50           |
| Cancer     | 4.0               | 50           |

**Total capacity:** 200 beds

---

## Methods Compared

| Method | Constraint handling | Margin diagnostics |
|--------|--------------------|--------------------|
| MAAT-Core | Penalty-based (`safety_lambda=1e6`) | ✓ |
| SciPy SLSQP | Native inequality constraints | ✗ |
| SciPy trust-constr | Native nonlinear constraints | ✗ |

MAAT-Core's key advantage is not raw numerical performance but
**interpretable safety semantics**: constraint margins are explicitly
tracked and reported, making violations visible before they occur.

---

## Usage

```bash
python3 benchmark_healthcare_vs_scipy.py
```

Output:

- Console summary per method
- `maat-core/runs/benchmark_healthcare_vs_scipy.csv`

---

## Output Fields

| Field | Description |
|-------|-------------|
| `method` | Optimizer name |
| `success` | Convergence flag |
| `runtime_sec` | Wall-clock time |
| `objective_value` | Raw optimizer output (negative lives saved) |
| `lives_saved` | Total lives saved (positive) |
| `min_margin` | Smallest constraint margin (negative = violation) |
| `violations` | Number of violated constraints |
| `x_covid/heart/cancer` | Final bed allocation |
| `diagnostics` | Method-specific notes |

---

## Requirements

```bash
pip install numpy scipy pandas
```

MAAT-Core must be installed in the active Python environment.

---

## Interpretation

All three methods should find the same optimal allocation. The
comparison is not about which optimizer wins numerically — it is
about what information each method surfaces:

- SciPy returns a solution and a success flag.
- MAAT-Core additionally returns **constraint margins**, making it
  possible to reason about *how safe* the solution is, not just
  *whether* it is feasible.

This distinction matters in safety-critical domains where boundary
proximity is as important as constraint satisfaction.

**Analytical optimum** for this problem:

```
x = [100, 50, 50]  →  850 lives saved
```

All methods should converge to this solution, making deviations
immediately visible in the CSV output.

---

## License

MIT License
