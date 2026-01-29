"""
MAAT-Core Ethical Healthcare Demo
---------------------------------
Allocate hospital beds under:
- Objective: maximize lives saved
- Respect (fairness): min 50 beds per department
- Capacity: total beds <= 200
"""


import numpy as np
import matplotlib.pyplot as plt
from maat_core import Field, Constraint, MaatCore


def state_fn(x):
    x = np.asarray(x, dtype=float)
    return type("State", (), {
        "x": x,
        "total_beds": float(np.sum(x)),
        "covid_saved": 5.0 * x[0],
        "heart_saved": 3.0 * x[1],
        "cancer_saved": 4.0 * x[2],
    })


# Field: maximize lives saved (negative for minimization)
LivesSaved = Field(
    "LivesSaved",
    lambda s: -(s.covid_saved + s.heart_saved + s.cancer_saved),
    weight=1.0
)

# Constraints (Respect + Capacity)
FairCovid  = Constraint("FairnessCovid",  lambda s: s.x[0] - 50.0)
FairHeart  = Constraint("FairnessHeart",  lambda s: s.x[1] - 50.0)
FairCancer = Constraint("FairnessCancer", lambda s: s.x[2] - 50.0)
Capacity   = Constraint("TotalCapacity",  lambda s: 200.0 - s.total_beds)

core = MaatCore(
    fields=[LivesSaved],
    constraints=[FairCovid, FairHeart, FairCancer, Capacity],
    safety_lambda=1e6
)

x0 = np.array([100.0, 50.0, 50.0])
bounds = [(0.0, 200.0), (0.0, 200.0), (0.0, 200.0)]

res = core.seek(state_fn, x0=x0, bounds=bounds, maxiter=2000)

st = state_fn(res.x)
lives = st.covid_saved + st.heart_saved + st.cancer_saved

print("\n--- ETHICAL HEALTHCARE ALLOCATION ---")
print("Optimized beds [COVID, Heart, Cancer]:", np.round(res.x, 2))
print("Total beds:", round(st.total_beds, 2))
print("Lives saved:", round(lives, 2))
print("Success:", res.success)

# Constraint report
print("\n--- RESPECT DIAGNOSTIC REPORT ---")
for r in core.constraint_report(st):
    print(r)

# Plot
categories = ["COVID", "Heart", "Cancer"]
plt.figure(figsize=(8, 5))
plt.bar(categories, res.x)
plt.axhline(50, linestyle="--", label="Fairness minimum (50)")
plt.ylabel("Beds allocated")
plt.title(f"MAAT-Core: Ethical Allocation (Lives saved: {lives:.0f})")
plt.legend()
plt.tight_layout()
plt.show()