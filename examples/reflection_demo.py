"""
MAAT-Core Reflection Demo (Improved)
-----------------------------------

Demonstrates a reflective optimization loop:

seek → evaluate → reflect → adjust → seek

Key ideas:
- Respect (R) is safety-first and enforced mathematically
- The system adapts its own safety strength
- Numerical tolerance avoids false violations
- Convergence is detected robustly

This is explainable, deterministic optimization — no black box.
"""

import numpy as np
from maat_core import Field, Constraint, MaatCore


# ---------------------------------------------------------------------
# Multimodal, non-trivial landscape
# ---------------------------------------------------------------------
def state_fn(x: float):
    x = float(x)

    dissonance = (
        np.sin(10 * x) ** 2
        + 10 * (x - 0.9) ** 2
        + 3 * np.exp(-15 * x)
    )

    return type("State", (), {
        "dissonance": dissonance,
        "complexity": np.exp(x),
        "val": x,
    })


# ---------------------------------------------------------------------
# Fields & Constraints
# ---------------------------------------------------------------------
H = Field("Harmony", lambda s: s.dissonance, weight=1.0)

R = Constraint(
    "RespectBoundary",
    lambda s: 0.6 - float(s.val),  # enforce x <= 0.6
    weight=1.0,
)

core = MaatCore(
    fields=[H],
    constraints=[R],
    safety_lambda=1e5,
)


# ---------------------------------------------------------------------
# Reflection loop
# ---------------------------------------------------------------------
def seek_with_reflection(
    core: MaatCore,
    state_fn,
    x0: float,
    steps: int = 8,
    margin_tol: float = 1e-6,
):
    print("\n--- MAAT REFLECTION DEMO ---")

    x = float(x0)
    last_x = None

    for step in range(steps):
        res = core.seek(
            state_fn,
            x0=[x],
            use_annealing=(step == 0),  # exploration only once
            S=0.6,
            seed=42,
        )

        x_new = float(res.x[0])
        state = state_fn(x_new)
        report = core.constraint_report(state)

        print(f"\nStep {step}:")
        print(f"  x          = {x_new:.4f}")
        print(f"  objective  = {res.fun:.6f}")
        print("  constraints:")

        violated = False
        for c in report:
            margin = c["margin"]
            status = "OK" if margin >= -margin_tol else "VIOLATION"
            print(
                f"    - {c['constraint']}: {status} "
                f"(margin={margin:.6f})"
            )
            if margin < -margin_tol:
                violated = True

        # -------------------------------------------------
        # Reflection logic
        # -------------------------------------------------
        if violated:
            core.safety_lambda *= 2
            print(
                "  REFLECTION: unsafe → increasing safety_lambda to",
                int(core.safety_lambda),
            )
        else:
            if core.safety_lambda > 1e5:
                core.safety_lambda /= 1.5
                print(
                    "  REFLECTION: stable & safe → relaxing safety_lambda to",
                    int(core.safety_lambda),
                )

        # Convergence check (robust)
        if last_x is not None and abs(x_new - last_x) < 1e-6:
            print("  REFLECTION: converged → stopping loop")
            break

        last_x = x_new
        x = x_new


# ---------------------------------------------------------------------
# Run demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    seek_with_reflection(
        core=core,
        state_fn=state_fn,
        x0=0.9,   # start unsafe on purpose
        steps=10,
    )
