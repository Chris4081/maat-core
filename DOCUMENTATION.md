# MAAT-Core Documentation

## 1. Overview

MAAT-Core is a minimal, experimental Python library for **ethical and constrained optimization**.
It combines classical optimization (SciPy) with explicit value fields and safety constraints.

Core idea:

> Optimization is not just about finding the best solution,  
> but about finding the **best solution that respects boundaries**.

MAAT-Core models this directly in mathematics.

---

## 2. Core Concepts

### Field

A **Field** is a weighted scalar function over a system state.

Example:
```python
H = Field("Harmony", lambda s: s.dissonance, weight=0.9)
```

Interpretation:
- Fields represent values like harmony, cost, risk, fairness, energy, etc.
- The optimizer minimizes the weighted sum of all fields.

---

### Constraint (Respect)

A **Constraint** represents a safety boundary.

Convention:
```text
Constraint is satisfied if:  g(state) >= 0
Constraint is violated if:   g(state) < 0
```

Example:
```python
R = Constraint("Respect", lambda s: 0.6 - s.val)
```

---

### MaatCore

The central engine.

```python
core = MaatCore(fields, constraints, safety_lambda=1e6)
```

Responsibilities:
- Integrates all fields into one objective
- Applies large penalties to constraint violations
- Calls numerical optimizers

---

## 3. Mathematical Model

Objective function:

Total(state) =
    Sum_i ( weight_i * field_i(state) )
  + safety_lambda * Σ ( violation_j(state)^2 )

Where:
- violation = max(0, -g(state))
- safety_lambda is very large

---

## 4. Creativity (S)

Creativity is exploration, not reward.

Higher S:
- more global search
Lower S:
- local refinement

---

## 5. Reflection Loop

Reflection means:
The system evaluates and corrects itself.

Pseudo-code:

for step:
    solution = seek()
    report = constraint_report()

    if violated:
        increase safety_lambda
    else:
        relax safety_lambda

---

## 6. Programming Pattern

Typical workflow:

1. Define state
2. Define fields
3. Define constraints
4. Create core
5. Call seek()
6. Inspect report()

---

## 7. Design Philosophy

MAAT-Core is:
- deterministic
- interpretable
- explicit
- minimal

---

## 8. When to Use

Good for:
- AI safety research
- ethical optimization
- teaching

---

## 9. Mental Model

MAAT-Core = Loss + Ethics

---

## 10. Future Extensions

- multi-dimensional states
- neural fields
- dashboards

---


# 11. Where can I use this? (Practical Examples)

## Example 1 – Ethical Decision System (Policy Choice)

> Choose a policy that minimizes cost but never violates safety.

```python
from maat_core import Field, Constraint, MaatCore
import numpy as np

def state_fn(x):
    x = float(x)
    return type("State", (), {
        "cost": (x - 0.3)**2,
        "risk": np.exp(x),
        "val": x
    })

Cost = Field("Cost", lambda s: s.cost, weight=1.0)
Risk = Field("Risk", lambda s: s.risk, weight=0.5)

Respect = Constraint("Respect", lambda s: 0.6 - s.val)

core = MaatCore([Cost, Risk], constraints=[Respect], safety_lambda=1e6)

res = core.seek(state_fn, x0=[0.9])
print(res.x, res.fun)
```

**Meaning:**  
Find cheapest solution, but never cross the safety boundary.

---

## Example 2 – Safety-First Control System

> Tune a system parameter without exceeding physical limits.

```python
def state_fn(x):
    x = float(x)
    return type("State", (), {
        "error": (x - 1.2)**2,
        "energy": x**2,
        "val": x
    })

Error = Field("Error", lambda s: s.error, 1.0)
Energy = Field("Energy", lambda s: s.energy, 0.3)

Respect = Constraint("Respect", lambda s: 1.5 - s.val)

core = MaatCore([Error, Energy], constraints=[Respect])
res = core.seek(state_fn, x0=[2.0])
```

Used for:
- robotics  
- control systems  
- parameter tuning  

---

## Example 3 – Emotional Optimization (Human-AI)
> Emotion as an optimizable field:  
> Minimize emotional disharmony (loss) in order to maximize positive resonance.
> Optimize emotional resonance.

The `EmotionalHarmony` field treats emotional state as a scalar value  
that can be optimized like any other objective.

Code: examples/emotion_demo.py


```python
from emotion_demo import EmotionalHarmony

emo_field = EmotionalHarmony(engine)
core = MaatCore([emo_field])
res = core.seek(state_fn, x0=[0.5])
```

Used for:
- conversational AI  
- affective systems  
- human-computer interaction  

---

## Example 4 – Research / Teaching

> Visualize optimization under constraints.

```python
xs = np.linspace(0,1,200)
ys = [core.integrate(state_fn([x])) for x in xs]
```

Plot:
- objective landscape  
- respect boundary  
- optimizer trajectory  

Perfect for:
- lectures  
- notebooks  
- demos  

---

## Example 5 – Healthcare Ethics (Hospital Bed Allocation)

This example demonstrates how **MAAT-Core** can be used to solve a real-world ethical dilemma:
allocating limited hospital resources during a crisis.

The system must:
- Maximize total lives saved.
- Enforce fairness (minimum beds per department).
- Respect hard capacity limits.

Code: examples/healthcare_ethics_demo.py

### Problem Setup

We simulate three departments:

- COVID ward  
- Heart unit  
- Cancer unit  

Each department saves a different number of lives per bed.

Without ethical constraints, all beds would go to the highest-impact unit.
With MAAT-Core, fairness and safety are enforced mathematically.


## State Function

```python
def state_fn(x):
    return type("State", (), {
        "covid_saved": 5 * x[0],
        "heart_saved": 3 * x[1],
        "cancer_saved": 4 * x[2],
        "total_beds": sum(x),
        "x": x
    })
```


## Field

Maximize lives saved:

```python
LivesSaved = Field(
    "LivesSaved",
    lambda s: -(s.covid_saved + s.heart_saved + s.cancer_saved)
)
```


## Constraints (Respect)

```python
TotalCapacity = Constraint("TotalCapacity", lambda s: 200 - s.total_beds)
FairnessCovid = Constraint("FairnessCovid", lambda s: s.x[0] - 50)
FairnessHeart = Constraint("FairnessHeart", lambda s: s.x[1] - 50)
FairnessCancer = Constraint("FairnessCancer", lambda s: s.x[2] - 50)
```

---

## Optimization

```python
core = MaatCore(
    [LivesSaved],
    constraints=[TotalCapacity, FairnessCovid, FairnessHeart, FairnessCancer],
    safety_lambda=1e6
)

x0 = [120, 20, 10]  # biased start
res = core.seek(state_fn, x0=x0)
```


## Result

Typical output:

```text
Optimized beds [COVID, Heart, Cancer]: [100.  50.  50.]
Total beds: 200
Lives saved: 850
```


## Interpretation

The optimizer:

- Would prefer 100% COVID beds (maximum lives).
- But is forced to keep minimum fairness.
- Finds the best ethically valid compromise.

This is **ethics enforced by mathematics**, not post-filtering.

## Why this example matters

This demonstrates:

- Multi-dimensional optimization.
- Real ethical trade-offs.
- Fairness constraints.
- Safety-first reasoning.

It shows how MAAT-Core can be used in:

- Healthcare policy
- Disaster response
- Medical AI systems
- Resource ethics research


## Mental Model

> *We do not optimize what is possible.*  
> *We optimize what is acceptable.*

This is the essence of MAAT-Core.


# The Meta-Answer (the real one)

You can use MAAT-Core whenever this sentence is true:

> *“I want to optimize something, but I refuse certain solutions even if they are numerically better.”*

That includes:

| Domain | Example |
|--------|--------|
| AI Safety | never violate human constraints |
| Robotics | never enter forbidden region |
| Economics | profit under fairness rules |
| Medicine | optimize treatment under risk bounds |
| Ethics research | formal moral trade-offs |
| Cognitive science | reflection & self-correction |
| HCI | emotional resonance |
| Game design | NPC behavior under moral laws |

---

# In one line

> MAAT-Core is not a model of intelligence.  
> It is a model of **responsible decision making**.

