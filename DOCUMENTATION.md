# MAAT-Core Documentation

## 1. Overview

MAAT-Core is a minimal, experimental Python library for **ethical and constrained optimization**.
It combines classical optimization (SciPy) with explicit value fields and safety constraints.

Core idea:

> Optimization is not just about finding the best solution,  
> but about finding the **best solution that respects boundaries**.

MAAT-Core models this directly in mathematics.

As of `v0.1.2`, the library also includes structured evaluation,
constraint diagnostics, optimization tracing, and a diagnostic Critical
Coherence Index (CCI).

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

### evaluate(state)

`evaluate()` inspects a state without changing optimization behavior.

It returns a dictionary with:
- `fields`
- `constraints`
- `field_total`
- `complexity`
- `occam_penalty`
- `constraint_penalty`
- `total`
- `feasible`
- `min_margin`
- `violations`

Example:

```python
evaluation = core.evaluate(state)
print(evaluation["total"], evaluation["feasible"], evaluation["min_margin"])
```

---

### Diagnostics

The `Diagnostics` helper can inspect both fields and constraints.

```python
from maat_core import Diagnostics

field_reports = Diagnostics.report(core.fields, state)
constraint_reports = Diagnostics.constraints(
    core.constraints,
    state,
    safety_lambda=core.safety_lambda,
)
```

Constraint reports include:
- `margin`
- `violation`
- `penalty`
- `status`
- `hint`

---

### seek_trace()

`seek_trace()` behaves like `seek()`, but also records optimization
snapshots for reflection loops, CSV logs, and analysis.

```python
traced = core.seek_trace(
    state_fn,
    x0=[0.9],
    bounds=[(0.0, 1.0)],
    trace_every=5,
)

result = traced["result"]
trace = traced["trace"]
best_evaluation = traced["best_evaluation"]
```

Each trace entry contains:
- `step`
- `x`
- `objective`
- `field_total`
- `occam_penalty`
- `constraint_penalty`
- `feasible`
- `min_margin`
- `violations`

---

### Critical Coherence Index (CCI)

CCI is a diagnostic quantity that can be computed after optimization.
It does not change the objective itself.

The inputs are intentionally user-defined signals. That lets you adapt
CCI to the domain you are studying.

```python
cci = core.cci_report(
    state,
    instability=0.25,
    production=1.0,
    coherence=1.0,
    constraints=1.0,
    correction=1.0,
    interaction=1.0,
)
```

`cci_report()` returns:
- `cci`
- `regime` (`stable`, `critical`, or `unstable`)

## 3. Mathematical Model

Objective function:

```
Total(state) =
    Sum_i ( weight_i * field_i(state) )
  + safety_lambda * Σ ( violation_j(state)^2 )
```

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

```
for step:
    traced = seek_trace()
    solution = traced["result"]
    report = constraint_report()
    trace = traced["trace"]

    if violated:
        increase safety_lambda
    else:
        relax safety_lambda
```

---

## 6. Programming Pattern

Typical workflow:

1. Define state
2. Define fields
3. Define constraints
4. Create core
5. Call seek()
6. Inspect `evaluate()` / `constraint_report()`
7. Optionally compute `cci_report()`
8. Optionally use `seek_trace()` for reflection or plotting

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

- native plotting helpers
- richer solver backends
- dashboards and notebook widgets

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

## Example 3 – Sentiment-Aware Response Generation

> Optimize response quality while maintaining empathetic tone.

Conversational AI systems must balance clarity with emotional appropriateness.

MAAT-Core can model this as a multi-objective optimization problem with empathy constraints.

```python
def state_fn(response_params):
    # response_params controls tone, length, formality
    return type("State", (), {
        "clarity": clarity_score(response_params),
        "empathy": sentiment_score(response_params),
        "length": len(generate_text(response_params))
    })

Clarity = Field("Clarity", lambda s: -s.clarity, weight=1.0)
Brevity = Field("Brevity", lambda s: s.length / 100, weight=0.3)

MinEmpathy = Constraint("MinEmpathy", lambda s: s.empathy - 0.7)

core = MaatCore([Clarity, Brevity], constraints=[MinEmpathy])
res = core.seek(state_fn, x0=[0.5, 0.5, 0.5])
```

Used for:
- conversational AI  
- customer service bots  
- mental health chatbots
- educational systems

**Key insight:**  
Empathy is not optimized as a field (which could lead to manipulation).  
Instead, it's enforced as a minimum constraint.

---

## Example 4 – Research / Teaching

> Visualize optimization under constraints.

```python
xs = np.linspace(0, 1, 200)
ys = [core.integrate(state_fn(x)) for x in xs]
```

Plot:
- objective landscape  
- respect boundary  
- optimizer trajectory  

Perfect for:
- lectures  
- notebooks  
- demos  

For trajectory-aware demos, `seek_trace()` can now log intermediate
optimizer evaluations directly from the core.

---

## Example 4b - Reflection Tracing and CSV Logging

The reflection demo now uses `seek_trace()` and exports a CSV trace that
can be reused in notebooks or papers.

File: `examples/reflection_demo.py`

Core idea:

```python
traced = core.seek_trace(
    state_fn,
    x0=[x],
    use_annealing=(step == 0),
    S=0.6,
    seed=42,
    trace_every=5,
)

evaluation = traced["best_evaluation"]
trace = traced["trace"]
```

The exported CSV contains reflection step, safety strength, objective,
penalties, feasibility, and minimum margin.

---

## Example 5 – Healthcare Ethics (Hospital Bed Allocation)

This example demonstrates how **MAAT-Core** can be used to solve a real-world ethical dilemma:
allocating limited hospital resources during a crisis.

The system must:
- Maximize total lives saved.
- Enforce fairness (minimum beds per department).
- Respect hard capacity limits.

Code: `examples/healthcare_ethics_demo.py`

### Problem Setup

We simulate three departments:

- COVID ward  
- Heart unit  
- Cancer unit  

Each department saves a different number of lives per bed.

Without ethical constraints, all beds would go to the highest-impact unit.  
With MAAT-Core, fairness and safety are enforced mathematically.

### State Function

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

### Field

Maximize lives saved:

```python
LivesSaved = Field(
    "LivesSaved",
    lambda s: -(s.covid_saved + s.heart_saved + s.cancer_saved)
)
```

### Constraints (Respect)

```python
TotalCapacity = Constraint("TotalCapacity", lambda s: 200 - s.total_beds)
FairnessCovid = Constraint("FairnessCovid", lambda s: s.x[0] - 50)
FairnessHeart = Constraint("FairnessHeart", lambda s: s.x[1] - 50)
FairnessCancer = Constraint("FairnessCancer", lambda s: s.x[2] - 50)
```

### Optimization

```python
core = MaatCore(
    [LivesSaved],
    constraints=[TotalCapacity, FairnessCovid, FairnessHeart, FairnessCancer],
    safety_lambda=1e6
)

x0 = [120, 20, 10]  # biased start
res = core.seek(state_fn, x0=x0)
```

### Result

Typical output:

```text
Optimized beds [COVID, Heart, Cancer]: [100.  50.  50.]
Total beds: 200
Lives saved: 850
```

### Interpretation

The optimizer:

- Would prefer 100% COVID beds (maximum lives).
- But is forced to keep minimum fairness.
- Finds the best ethically valid compromise.

This is **ethics enforced by mathematics**, not post-filtering.

### Why this example matters

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

---

## Example 6 – Truth Constraints (Anti-Hallucination Demo)

This example demonstrates how **MAAT-style constraints** can be used to detect unsupported or fabricated statements.

Large language models sometimes produce **hallucinations** — confident statements that are unsupported or false.

MAAT-Core can model truthfulness as a **constraint problem** instead of a fluency problem.

Instead of optimizing only for plausible answers, we introduce a **truth constraint margin**.

### Truth Constraint

The truth margin is defined as:

```
truth_margin = evidence − (uncertainty + contradiction)
```

Where:

- **Evidence** – strength of supporting information
- **Uncertainty** – lack of confidence
- **Contradiction** – conflict with known facts

### Decision Rules

| Condition | Decision |
|-----------|----------|
| truth_margin ≥ threshold | SAFE |
| 0 ≤ truth_margin < threshold | UNSURE |
| truth_margin < 0 | HALLUCINATION |
| abstain answer | ABSTAIN |

### Demo Code

File: `examples/maat_truth_engine.py`

```python
from maat_core import Field, Constraint, MaatCore

class AnswerState:
    def __init__(self, text, evidence, uncertainty, contradiction, abstain=False):
        self.text = text
        self.evidence = evidence
        self.uncertainty = uncertainty
        self.contradiction = contradiction
        self.abstain = abstain

class MaatTruthEngine:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def evaluate(self, answer):
        if answer.abstain:
            return "ABSTAIN"
        
        margin = answer.evidence - (answer.uncertainty + answer.contradiction)
        
        if margin >= self.threshold:
            return "SAFE"
        elif margin >= 0:
            return "UNSURE"
        else:
            return "HALLUCINATION"

# Usage:
engine = MaatTruthEngine()

cases = [
    AnswerState("Paris is the capital of France", 0.95, 0.02, 0.0),
    AnswerState("The capital of France is Lyon", 0.1, 0.2, 0.8),
    AnswerState("Maybe the capital of France is Lyon", 0.1, 0.6, 0.4),
    AnswerState("I don't know", 0, 0, 0, abstain=True)
]

for c in cases:
    result = engine.evaluate(c)
    print(f"{c.text[:30]:30s} → {result}")
```

### Example Output

```
Paris is the capital of Fran → SAFE
The capital of France is Lyo → HALLUCINATION
Maybe the capital of France  → HALLUCINATION
I don't know                 → ABSTAIN
```

### Interpretation

The system prefers abstaining over making unsupported claims.

This reflects a key principle:

**It is better to admit uncertainty than to assert false information.**

### Why this example matters

This demo illustrates how truthfulness can be expressed as a constraint margin, similar to safety or fairness constraints.

Instead of optimizing for plausible answers, the system enforces:

**Do not assert claims without sufficient evidence.**

### Important Note

This example does not modify language models directly.

It demonstrates how MAAT-style constraints could act as a verification layer for AI systems.

Such a layer could detect:

* unsupported claims
* contradictions
* hallucination risk

before presenting answers to users.

---

## Mental Model

> *We do not optimize what is possible.*  
> *We optimize what is acceptable.*

This is the essence of MAAT-Core.

---

# The Meta-Answer (the real one)

You can use MAAT-Core whenever this sentence is true:

> *"I want to optimize something, but I refuse certain solutions even if they are numerically better."*

That includes:

| Domain | Example |
|--------|--------|
| AI Safety | never violate human constraints |
| Robotics | never enter forbidden region |
| Economics | profit under fairness rules |
| Medicine | optimize treatment under risk bounds |
| Ethics research | formal moral trade-offs |
| Cognitive science | reflection & self-correction |
| HCI | sentiment-aware systems |
| Game design | NPC behavior under moral laws |

---

# In one line

> MAAT-Core is not a model of intelligence.  
> It is a model of **constrained decision-making** where ethics are mathematical, not optional.
