#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cci_demo.py
-----------
Minimal demo for the Critical Coherence Index (CCI) in MAAT-Core.

Shows:
- standard optimization
- constraint report
- CCI report
- rough regime classification

Run:
    python3 examples/cci_demo.py
"""

from maat_core import Field, Constraint, MaatCore


def state_fn(x):
    x = float(x)

    # A simple toy system:
    # low cost near x=0.7
    # increasing complexity away from center
    # one safety boundary at x <= 0.8
    return type("State", (), {
        "cost": (x - 0.7) ** 2,
        "complexity": abs(x - 0.5),
        "val": x
    })


def main():
    # Objective field
    Cost = Field("Cost", lambda s: s.cost, weight=1.0)

    # Safety constraint: x <= 0.8
    Respect = Constraint("Respect", lambda s: 0.8 - s.val)

    core = MaatCore(
        [Cost],
        constraints=[Respect],
        safety_lambda=1e6,
        occam_lambda=0.1
    )

    # Optimize
    res = core.seek(
        state_fn,
        x0=[0.95],
        bounds=[(0.0, 1.0)],
        method="L-BFGS-B"
    )

    x_best = float(res.x[0])
    state = state_fn(x_best)

    print("\n=== MAAT-Core CCI Demo ===\n")
    print(f"Optimized x        : {x_best:.6f}")
    print(f"Objective value    : {res.fun:.6f}")
    print(f"State cost         : {state.cost:.6f}")
    print(f"State complexity   : {state.complexity:.6f}")

    # Constraint diagnostics
    print("\n--- Constraint Report ---")
    report = core.constraint_report(state)
    for r in report:
        print(r)

    # CCI diagnostics
    #
    # These are currently toy values / user-defined diagnostics.
    # Later you can replace them with more meaningful signals.
    #
    # production: interpreted here as "system activity"
    # instability: larger if we are near the boundary
    margin = report[0]["margin"]

    instability = max(0.0, 0.2 - margin) * 5.0
    production = 1.0 + state.cost
    coherence = 1.0
    constraints = 1.0 + max(0.0, margin)
    correction = 1.0
    interaction = 1.0
    U_struct = abs(state.complexity - state.cost)

    cci = core.cci_report(
        state,
        instability=instability,
        production=production,
        coherence=coherence,
        constraints=constraints,
        correction=correction,
        interaction=interaction,
        U_struct=U_struct,
        kappa=0.2
    )

    print("\n--- CCI Report ---")
    print(f"instability        : {instability:.6f}")
    print(f"production         : {production:.6f}")
    print(f"U_struct           : {U_struct:.6f}")
    print(f"CCI                : {cci['cci']:.6f}")
    print(f"Regime             : {cci['regime']}")

    print("\nInterpretation:")
    print("- stable   : system is comfortably inside constraints")
    print("- critical : system approaches a transition zone")
    print("- unstable : system is highly stressed / near breakdown")


if __name__ == "__main__":
    main()