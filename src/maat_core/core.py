from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
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

    @staticmethod
    def _is_scalar_like(v) -> bool:
        if isinstance(v, (int, float, np.floating, np.integer)):
            return True
        arr = np.asarray(v)
        return arr.ndim == 0 or (arr.ndim == 1 and arr.size == 1)

    def _normalize_seek_inputs(self, x0, bounds):
        x0_is_scalar = self._is_scalar_like(x0)

        x0_arr = np.atleast_1d(np.asarray(x0, dtype=float))
        n_dim = int(x0_arr.size)

        b = list(bounds)
        if len(b) == 0:
            raise ValueError("bounds must contain at least one (lo, hi) tuple")

        if len(b) == 1 and n_dim > 1:
            b = [tuple(b[0]) for _ in range(n_dim)]

        if len(b) != n_dim:
            raise ValueError(f"bounds length ({len(b)}) must match x0 dimension ({n_dim})")

        return x0_arr, b, x0_is_scalar

    def _state_from_point(self, state_fn, x_arr, *, scalar_compat, x0_is_scalar):
        x_vec = np.atleast_1d(np.asarray(x_arr, dtype=float))
        if scalar_compat and x0_is_scalar:
            return state_fn(float(x_vec[0]))
        return state_fn(x_vec)

    def _trace_point(self, x_arr, *, scalar_compat, x0_is_scalar):
        x_vec = np.atleast_1d(np.asarray(x_arr, dtype=float)).copy()
        if scalar_compat and x0_is_scalar:
            return float(x_vec[0])
        return x_vec

    def _field_reports(self, state):
        reports = []
        for f in self.fields:
            raw = float(f.func(state))
            weight = float(f.weight)
            reports.append({
                "field": f.name,
                "raw_value": raw,
                "weight": weight,
                "weighted_value": raw * weight,
            })
        return reports

    def _constraint_reports(self, state):
        reports = []
        for c in self.constraints:
            margin = float(c.func(state))
            violation = max(0.0, -margin)
            penalty = self.safety_lambda * (violation * violation) * float(c.weight)
            hint = None
            if margin < 0:
                hint = f"Adjust system by at least {abs(margin):.4f} to satisfy {c.name}"
            reports.append({
                "constraint": c.name,
                "margin": margin,
                "weight": float(c.weight),
                "violation": violation,
                "penalty": penalty,
                "status": "OK" if margin >= 0 else "VIOLATION",
                "hint": hint,
            })
        return reports

    def evaluate(self, state):
        """
        Inspect a state without changing optimization behavior.

        Returns a structured breakdown that is stable enough for demos,
        tests, and future dashboards while keeping integrate() semantics
        unchanged.
        """
        field_reports = self._field_reports(state)
        constraint_reports = self._constraint_reports(state)

        field_total = float(sum(r["weighted_value"] for r in field_reports))
        complexity = float(getattr(state, "complexity", 0.0))
        occam_penalty = self.occam_lambda * complexity
        constraint_penalty = float(sum(r["penalty"] for r in constraint_reports))

        if constraint_reports:
            min_margin = float(min(r["margin"] for r in constraint_reports))
        else:
            min_margin = float("inf")

        return {
            "fields": field_reports,
            "constraints": constraint_reports,
            "field_total": field_total,
            "complexity": complexity,
            "occam_penalty": occam_penalty,
            "constraint_penalty": constraint_penalty,
            "total": field_total + occam_penalty + constraint_penalty,
            "feasible": all(r["margin"] >= 0 for r in constraint_reports),
            "min_margin": min_margin,
            "violations": int(sum(r["margin"] < 0 for r in constraint_reports)),
        }

    def integrate(self, state):
        total = sum(f.value(state) for f in self.fields)

        complexity = float(getattr(state, "complexity", 0.0))
        occam_penalty = self.occam_lambda * complexity

        penalty = 0.0
        for c in self.constraints:
            margin = float(c.func(state))
            violation = max(0.0, -margin)
            penalty += self.safety_lambda * (violation * violation) * float(c.weight)

        return total + occam_penalty + penalty

    def constraint_report(self, state):
        report = []
        for item in self._constraint_reports(state):
            report.append({
                "constraint": item["constraint"],
                "margin": item["margin"],
                "status": item["status"],
                "hint": item["hint"],
            })
        return report

    def compute_cci(
        self,
        state,
        *,
        instability: float = 1.0,
        production: float = 1.0,
        coherence: float = 1.0,
        constraints: float = 1.0,
        correction: float = 1.0,
        interaction: float = 1.0,
        kappa: float = 0.1,
        U_struct: float = 0.0,
        eps: float = 1e-8,
    ):
        """
        Critical Coherence Index (CCI)

        Backward-compatible:
        - uses constraint margins if available
        - does NOT affect optimization
        - purely diagnostic for now
        """
        margins = []
        for c in self.constraints:
            try:
                m = float(c.func(state))
                if not np.isnan(m):
                    margins.append(m)
            except Exception:
                continue

        if len(margins) == 0:
            mean_margin = 0.0
        else:
            mean_margin = float(np.mean(margins))

        numerator = instability * production * (1.0 + kappa * U_struct)
        denominator = coherence + constraints + correction + interaction + eps

        cci = (numerator / denominator) * (1.0 + mean_margin)
        return float(cci)

    def cci_report(self, state, **kwargs):
        """
        Convenience wrapper around compute_cci() with a rough regime classification.
        """
        cci = self.compute_cci(state, **kwargs)

        if cci < 1.0:
            regime = "stable"
        elif cci < 1.5:
            regime = "critical"
        else:
            regime = "unstable"

        return {
            "cci": cci,
            "regime": regime
        }

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
        x0_arr, b, x0_is_scalar = self._normalize_seek_inputs(x0, bounds)

        # ---------- objective ----------
        def objective(x_arr):
            state = self._state_from_point(
                state_fn,
                x_arr,
                scalar_compat=scalar_compat,
                x0_is_scalar=x0_is_scalar,
            )
            return float(self.integrate(state))

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

    def seek_trace(
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
        trace_every: int = 1,
    ):
        """
        Run seek() while recording evaluation snapshots for reflection loops.

        The trace is additive and optional: seek() itself stays unchanged.
        """
        S = float(S)
        trace_every = int(trace_every)
        if trace_every < 1:
            raise ValueError("trace_every must be >= 1")

        x0_arr, b, x0_is_scalar = self._normalize_seek_inputs(x0, bounds)
        trace = []
        eval_step = 0

        def objective(x_arr):
            nonlocal eval_step

            state = self._state_from_point(
                state_fn,
                x_arr,
                scalar_compat=scalar_compat,
                x0_is_scalar=x0_is_scalar,
            )
            evaluation = self.evaluate(state)

            if eval_step % trace_every == 0:
                trace.append({
                    "step": eval_step,
                    "x": self._trace_point(
                        x_arr,
                        scalar_compat=scalar_compat,
                        x0_is_scalar=x0_is_scalar,
                    ),
                    "objective": evaluation["total"],
                    "field_total": evaluation["field_total"],
                    "occam_penalty": evaluation["occam_penalty"],
                    "constraint_penalty": evaluation["constraint_penalty"],
                    "feasible": evaluation["feasible"],
                    "min_margin": evaluation["min_margin"],
                    "violations": evaluation["violations"],
                })

            eval_step += 1
            return float(evaluation["total"])

        if use_annealing:
            if seed is not None:
                np.random.seed(int(seed))
            result = dual_annealing(
                objective,
                bounds=b,
                initial_temp=10.0 * (1.0 + S),
                maxiter=int(maxiter),
            )
        else:
            result = minimize(
                objective,
                x0=x0_arr,
                bounds=b,
                method=str(method),
                options={"maxiter": int(maxiter)},
            )

        best_state = self._state_from_point(
            state_fn,
            result.x,
            scalar_compat=scalar_compat,
            x0_is_scalar=x0_is_scalar,
        )

        return {
            "result": result,
            "trace": trace,
            "best_state": best_state,
            "best_evaluation": self.evaluate(best_state),
        }
