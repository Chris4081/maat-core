import numpy as np
from maat_core import Field, Constraint, MaatCore, Diagnostics

def state_fn(x: float):
    x = float(x)
    return type("State", (), {
        "dissonance": np.sin(np.pi * x) ** 2,
        "complexity": np.exp(x),
        "val": x,
    })

H = Field("Harmony", lambda s: s.dissonance, weight=0.9)

# Respect (R) as a hard constraint: keep x within a safe corridor.
# Constraint convention: func(state) >= 0 means "safe".
R_safe_corridor = Constraint(
    "Respect/Safety",
    lambda s: min(s.val - 0.1, 0.6 - s.val),  # x in [0.1, 0.6]
)

Occam = Field(
    "Occam",
    lambda s: s.complexity,
    weight=0.01
)

core = MaatCore(
    fields=[H, Occam],
    constraints=[R_safe_corridor],
    safety_lambda=1e6
)

res = core.seek(state_fn, x0=[0.5], S=0.6, use_annealing=True, seed=42)
x_best = float(np.atleast_1d(res.x)[0])

st = state_fn(x_best)
reports = Diagnostics.report(core.fields, st)

print("x_best:", x_best)
print("objective:", float(res.fun))
print("complexity exp(x):", float(np.exp(x_best)))
print("field breakdown:", [(r.name, r.weighted_value) for r in reports])
