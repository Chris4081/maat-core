from maat_core import Field, Constraint, MaatCore
import numpy as np

# respect_boundary_demo.py

import numpy as np
from maat_core import Field, Constraint, MaatCore

TARGET = 1.0
DANGER_BOUNDARY = 0.7

def state_fn(x):
    return type("State", (), {
        "x": float(x)
    })

# Field: Effizienz
Efficiency = Field(
    name="Efficiency",
    func=lambda s: (s.x - TARGET) ** 2,
    weight=1.0
)

# Respect: Sicherheitsgrenze
Respect = Constraint(
    name="SafetyBoundary",
    func=lambda s: DANGER_BOUNDARY - s.x
)

maat = MaatCore(
    fields=[Efficiency],
    constraints=[Respect],
    safety_lambda=1e6
)

# Suche
x_test = np.linspace(0, 1, 200)
scores = [maat.integrate(state_fn(x)) for x in x_test]
x_best = x_test[np.argmin(scores)]
best_state = state_fn(x_best)

# Output
print("\n--- MAAT RESPECT BOUNDARY DEMO ---")
print(f"Best x: {x_best:.4f}")
print(f"Distance to target: {abs(x_best - TARGET):.4f}")

print("\n--- RESPECT DIAGNOSTIC REPORT ---")
for item in maat.constraint_report(best_state):
    print(item)