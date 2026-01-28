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

## 11. Final Thought

MAAT-Core is not an AI.
It is a mathematical conscience for optimization.
