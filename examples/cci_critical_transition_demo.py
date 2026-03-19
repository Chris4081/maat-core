#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cci_critical_transition_demo.py
-------------------------------
True transition demo:

- optimizer prefers x ≈ 0.75
- constraint pushes against it
- we sweep constraint boundary
- observe CCI rising near critical region

Run:
    python3 examples/cci_critical_transition_demo.py
"""

import csv
from pathlib import Path
import numpy as np

from maat_core import Field, Constraint, MaatCore


# 🔥 System with REAL conflict
def state_fn(x):
    x = float(x)

    return type("State", (), {
        # preferred minimum at 0.75
        "cost": (x - 0.75) ** 2,

        # increasing structural tension
        "complexity": abs(x - 0.5),

        "val": x
    })


def main():

    Cost = Field("Cost", lambda s: s.cost, weight=1.0)

    # 🔥 critical region sweep
    respect_limits = np.linspace(0.65, 0.80, 20)

    rows = []

    print("\n=== MAAT-Core CRITICAL TRANSITION DEMO ===\n")
    print(f"{'limit':>8} {'x*':>8} {'margin':>10} {'CCI':>10} {'regime':>10}")
    print("-" * 60)

    for limit in respect_limits:

        Respect = Constraint(
            "Respect",
            lambda s, limit=limit: limit - s.val
        )

        core = MaatCore(
            [Cost],
            constraints=[Respect],
            safety_lambda=1e6,
            occam_lambda=0.1
        )

        res = core.seek(
            state_fn,
            x0=[0.95],
            bounds=[(0.0, 1.0)],
            method="L-BFGS-B"
        )

        x_best = float(res.x[0])
        state = state_fn(x_best)

        report = core.constraint_report(state)
        margin = float(report[0]["margin"])

        # 🔥 KEY PART: strong instability near boundary
        instability = max(0.0, 0.03 - margin) * 30.0

        # system activity
        production = 1.0 + state.cost

        # structural mismatch
        U_struct = abs(state.complexity - state.cost)

        cci = core.cci_report(
            state,
            instability=instability,
            production=production,
            coherence=1.0,
            constraints=1.0 + max(0.0, margin),
            correction=1.0,
            interaction=1.0,
            U_struct=U_struct,
            kappa=0.3
        )

        rows.append({
            "limit": float(limit),
            "x_best": x_best,
            "margin": margin,
            "cci": float(cci["cci"]),
            "regime": cci["regime"]
        })

        print(f"{limit:8.3f} {x_best:8.4f} {margin:10.6f} {cci['cci']:10.6f} {cci['regime']:>10}")

    # Save results
    out_path = Path("cci_critical_transition_results.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved CSV:", out_path)

    print("\nInterpretation:")
    print("✔ Large margin  → stable")
    print("⚠ Near zero     → critical")
    print("✖ Negative      → unstable")


if __name__ == "__main__":
    main()