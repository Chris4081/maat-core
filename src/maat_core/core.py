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
        scalar_compat: bool = True,
    ):
        """
        Multi-dim safe seek().

        - If x0 is scalar-like, we run in 1D mode and call state_fn(float(x)).
        - If x0 is vector-like, we run in ND mode and call state_fn(np.ndarray).

        bounds:
            - For 1D: ((lo, hi),)
            - For ND: list/tuple of (lo, hi) for each dimension
            - If bounds has length 1 and x0 is ND, bounds is broadcast to all dims.
        """
        S = float(S)

        # ---------- detect scalar vs vector ----------
        def _is_scalar_like(v) -> bool:
            if isinstance(v, (int, float, np.floating, np.integer)):
                return True
            arr = np.asarray(v)
            return arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1)

        x0_is_scalar = _is_scalar_like(x0)

        # Normalize x0 to ndarray for optimizers
        x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
        n_dim = int(x0_arr.size)

        # ---------- bounds handling ----------
        b = list(bounds)

        if len(b) == 0:
            raise ValueError("bounds must contain at least one (lo, hi) tuple")

        # Broadcast single bound to all dims if needed
        if len(b) == 1 and n_dim > 1:
            b = [tuple(b[0]) for _ in range(n_dim)]

        if len(b) != n_dim:
            raise ValueError(f"bounds length ({len(b)}) must match x0 dimension ({n_dim})")

        # ---------- objective ----------
        def objective(x_arr):
            x_vec = np.atleast_1d(np.asarray(x_arr, dtype=float))

            # keep old 1D examples working: state_fn gets float
            if scalar_compat and x0_is_scalar:
                x_scalar = float(x_vec[0])
                return float(self.integrate(state_fn(x_scalar)))

            # ND mode: state_fn gets full vector
            return float(self.integrate(state_fn(x_vec)))

        # ---------- solve ----------
        if use_annealing:
            if seed is not None:
                np.random.seed(int(seed))
            return dual_annealing(
                objective,
                bounds=b,
                initial_temp=10.0 * (1.0 + S),
                maxiter=int(maxiter),
            )

        return minimize(
            objective,
            x0=x0_arr,
            bounds=b,
            method=str(method),
            options={"maxiter": int(maxiter)},
        )