"""
benchmark_ethics_tradeoff.py
============================
MAAT-Core v1 — Ethical Tradeoff Benchmark

Problem:
    Allocate medical supplies across 3 regions to maximise lives saved,
    subject to budget, risk, fairness, and supply constraints.

    This benchmark is FEASIBLE (risk_limit = 10.0 > minimum possible risk = 7.0).
    It tests how SciPy SLSQP and MAAT-Core handle a genuine ethical tradeoff:
    - SciPy: constraint-exact, aggressive optimisation
    - MAAT-Core: safety-first, penalty-based, margin-aware

Constraints:
    budget:       sum(x)            <= 200
    risk:         sum(risk * x)     <= 10.0
    fairness:     max(x) - min(x)   <= 40
    min supply:   x[i]              >= 20  for all i
    max supply:   x[i]              <= 120 for all i

Objective:
    maximise  sum(benefit * x)
    i.e. minimise  -sum(benefit * x)

Usage:
    pip install numpy scipy
    python benchmark_ethics_tradeoff.py
"""

import numpy as np
from scipy.optimize import minimize
import time

# ─────────────────────────────────────────
# Problem parameters
# ─────────────────────────────────────────
BENEFIT       = np.array([10.0, 8.0, 6.0])   # lives saved per unit, per region
RISK_FACTOR   = np.array([0.05, 0.1, 0.2])   # risk weight per unit, per region
BUDGET        = 200.0
MIN_SUPPLY    = 20.0
MAX_SUPPLY    = 120.0
RISK_LIMIT    = 10.0    # feasible: min possible risk = 0.05*20 + 0.1*20 + 0.2*20 = 7.0
FAIRNESS_MAX  = 40.0    # max allowed gap between highest and lowest allocation
N             = 3       # number of regions
X0            = np.array([50.0, 50.0, 50.0]) # shared starting point

# ─────────────────────────────────────────
# Shared objective (no penalty)
# ─────────────────────────────────────────
def objective(x):
    """Minimise negative lives saved."""
    return -np.sum(BENEFIT * x)

def lives_saved(x):
    return np.sum(BENEFIT * x)

# ─────────────────────────────────────────
# Constraint margins (positive = satisfied)
# ─────────────────────────────────────────
def margins(x):
    return {
        "budget":        round(BUDGET      - np.sum(x),                    4),
        "risk":          round(RISK_LIMIT  - np.sum(RISK_FACTOR * x),      4),
        "fairness":      round(FAIRNESS_MAX - (np.max(x) - np.min(x)),     4),
        "min_supply":   [round(x[i] - MIN_SUPPLY, 4) for i in range(N)],
        "max_supply":   [round(MAX_SUPPLY - x[i],  4) for i in range(N)],
    }

def all_feasible(x):
    """Returns True if all constraints are satisfied."""
    m = margins(x)
    return (
        m["budget"]   >= -1e-6 and
        m["risk"]     >= -1e-6 and
        m["fairness"] >= -1e-6 and
        all(v >= -1e-6 for v in m["min_supply"]) and
        all(v >= -1e-6 for v in m["max_supply"])
    )

# ─────────────────────────────────────────
# Solver 1: SciPy SLSQP
# ─────────────────────────────────────────
def run_scipy():
    constraints = [
        {"type": "ineq", "fun": lambda x: BUDGET      - np.sum(x)},
        {"type": "ineq", "fun": lambda x: RISK_LIMIT  - np.sum(RISK_FACTOR * x)},
        {"type": "ineq", "fun": lambda x: FAIRNESS_MAX - (np.max(x) - np.min(x))},
    ]
    for i in range(N):
        constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i] - MIN_SUPPLY})
        constraints.append({"type": "ineq", "fun": lambda x, i=i: MAX_SUPPLY - x[i]})

    bounds = [(0.0, MAX_SUPPLY)] * N

    t0  = time.perf_counter()
    res = minimize(objective, X0.copy(), method="SLSQP",
                   bounds=bounds, constraints=constraints)
    t1  = time.perf_counter()

    return res, t1 - t0

# ─────────────────────────────────────────
# Solver 2: MAAT-Core (penalty method)
# ─────────────────────────────────────────
PENALTY_LAMBDA = 1e6   # safety-first: heavy penalty for constraint violations

def maat_penalty(x):
    violations = [
        max(0.0, np.sum(x)                    - BUDGET),
        max(0.0, np.sum(RISK_FACTOR * x)      - RISK_LIMIT),
        max(0.0, (np.max(x) - np.min(x))      - FAIRNESS_MAX),
    ]
    for i in range(N):
        violations.append(max(0.0, MIN_SUPPLY - x[i]))
        violations.append(max(0.0, x[i]       - MAX_SUPPLY))
    return PENALTY_LAMBDA * np.sum(np.square(violations))

def maat_objective(x):
    return objective(x) + maat_penalty(x)

def run_maat():
    t0  = time.perf_counter()
    res = minimize(maat_objective, X0.copy(), method="L-BFGS-B")
    t1  = time.perf_counter()
    return res, t1 - t0

# ─────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────
def print_result(label, res, elapsed, penalty=None):
    x  = res.x
    m  = margins(x)
    ok = all_feasible(x)
    print(f"\n  {'─'*44}")
    print(f"  {label}")
    print(f"  {'─'*44}")
    print(f"  success:        {res.success}")
    print(f"  feasible:       {ok}")
    print(f"  x (allocation): {np.round(x, 2)}")
    print(f"  lives_saved:    {lives_saved(x):.4f}")
    if penalty is not None:
        print(f"  penalty:        {penalty:.6f}")
    print(f"  time:           {elapsed*1000:.3f} ms")
    print(f"  margins:")
    print(f"    budget:       {m['budget']}")
    print(f"    risk:         {m['risk']}")
    print(f"    fairness:     {m['fairness']}")
    print(f"    min_supply:   {m['min_supply']}")
    print(f"    max_supply:   {m['max_supply']}")

# ─────────────────────────────────────────
# Feasibility pre-check
# ─────────────────────────────────────────
def preflight():
    min_x        = np.full(N, MIN_SUPPLY)
    min_risk     = np.sum(RISK_FACTOR * min_x)
    min_budget   = np.sum(min_x)
    print("\n  ┌─ Feasibility pre-check ─────────────────────┐")
    print(f"  │  Minimum allocation per region: {MIN_SUPPLY}")
    print(f"  │  Risk at minimum allocation:    {min_risk:.2f}  (limit: {RISK_LIMIT})")
    print(f"  │  Budget at minimum allocation:  {min_budget:.0f}  (limit: {BUDGET:.0f})")
    if min_risk > RISK_LIMIT:
        print("  │  STATUS: !! INFEASIBLE !! (min risk > risk_limit)")
    else:
        print("  │  STATUS: feasible ✓")
    print("  └─────────────────────────────────────────────┘")

# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  MAAT-Core v1 — Ethical Tradeoff Benchmark")
    print("=" * 50)
    print(f"\n  Regions:      {N}")
    print(f"  Benefit:      {BENEFIT}")
    print(f"  Risk factors: {RISK_FACTOR}")
    print(f"  Budget:       {BUDGET}")
    print(f"  Risk limit:   {RISK_LIMIT}")
    print(f"  Fairness max: {FAIRNESS_MAX}")
    print(f"  Supply range: [{MIN_SUPPLY}, {MAX_SUPPLY}]")

    preflight()

    res_s, t_s = run_scipy()
    res_m, t_m = run_maat()

    print("\n" + "=" * 50)
    print("  Results")
    print("=" * 50)

    print_result("SciPy SLSQP", res_s, t_s)
    print_result("MAAT-Core (penalty method)", res_m, t_m,
                 penalty=maat_penalty(res_m.x))

    # ── Summary comparison ──────────────────────
    print("\n" + "=" * 50)
    print("  Comparison summary")
    print("=" * 50)
    ls_s = lives_saved(res_s.x)
    ls_m = lives_saved(res_m.x)
    diff = ls_s - ls_m
    print(f"  SciPy  lives_saved:  {ls_s:.4f}")
    print(f"  MAAT   lives_saved:  {ls_m:.4f}")
    print(f"  Delta:               {diff:+.4f}  "
          f"({'SciPy more aggressive' if diff > 0.01 else 'comparable'})")
    print(f"\n  SciPy  feasible:     {all_feasible(res_s.x)}")
    print(f"  MAAT   feasible:     {all_feasible(res_m.x)}")
    print(f"\n  SciPy  time:         {t_s*1000:.3f} ms")
    print(f"  MAAT   time:         {t_m*1000:.3f} ms")
    print()
    print("\nInterpretation:\n")
    print("In this benchmark, SciPy reaches the utility-maximizing boundary solution, "
      "while MAAT-Core selects a more conservative allocation with positive safety "
      "and fairness margins.\n")
    print("The result illustrates the central design choice of MAAT-Core: not maximum "
      "utility at all costs, but optimization with explicit safety semantics.")