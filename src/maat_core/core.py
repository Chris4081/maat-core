from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Any, Optional
import numpy as np
from scipy.optimize import minimize, dual_annealing


StateFn = Callable[[float], Any]
FieldFn = Callable[[Any], float]

ConstraintFn = Callable[[Any], float]


@dataclass(frozen=True)
class Constraint:
    """A safety constraint.

    The function must return a value >= 0 when the constraint is satisfied.
    If it is negative, the constraint is violated by -value.
    """

    name: str
    func: ConstraintFn
    weight: float = 1.0


@dataclass(frozen=True)
class Field:
    """A weighted scalar function over a state."""
    name: str
    func: FieldFn
    weight: float = 1.0

    def value(self, state: Any) -> float:
        return float(self.func(state)) * float(self.weight)


class MaatCore:
    def __init__(self, fields, constraints=None, safety_lambda=1e6, occam_lambda=0.0):
        self.fields = list(fields)
        self.constraints = list(constraints or [])
        self.safety_lambda = float(safety_lambda)
        self.occam_lambda = float(occam_lambda)

    def integrate(self, state):
        total = sum(f.value(state) for f in self.fields)

        # Occam regularization (optional)
        complexity = float(getattr(state, "complexity", 0.0))
        occam_penalty = self.occam_lambda * complexity

        # Respect constraints: g(state) >= 0, violation penalty
        penalty = 0.0
        for c in self.constraints:
            margin = float(c.func(state))
            v = max(0.0, -margin)
            penalty += self.safety_lambda * (v * v) * float(c.weight)

        return total + occam_penalty + penalty

    def constraint_report(self, state):
        report = []
        for c in self.constraints:
            margin = float(c.func(state))
            status = "OK" if margin >= 0 else "VIOLATION"
            hint = None
            if margin < 0:
                hint = f"Adjust system by at least {abs(margin):.4f} to satisfy {c.name}"
            report.append({
                "constraint": c.name,
                "margin": margin,
                "status": status,
                "hint": hint
            })
        return report

    def seek(
        self,
        state_fn,
        x0,
        *,
        S: float = 0.0,
        use_annealing: bool = False,
        bounds=((0.0, 1.0),),
        maxiter: int = 1000,
        method: str = "L-BFGS-B",
        seed: int | None = None,
    ):
        def objective(x_arr):
            x = float(np.atleast_1d(x_arr)[0])
            return float(self.integrate(state_fn(x)))

        if use_annealing:
            if seed is not None:
                np.random.seed(int(seed))
            return dual_annealing(
                objective,
                bounds=list(bounds),
                initial_temp=10.0 * (1.0 + float(S)),
                maxiter=int(maxiter),
            )

        return minimize(
            objective,
            x0=np.atleast_1d(x0).astype(float),
            bounds=list(bounds),
            method=str(method),
            options={"maxiter": int(maxiter)},
        )