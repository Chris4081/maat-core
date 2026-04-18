from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from .core import Constraint, Field


@dataclass(frozen=True)
class FieldReport:
    name: str
    weight: float
    raw_value: float
    weighted_value: float


@dataclass(frozen=True)
class ConstraintReport:
    name: str
    weight: float
    margin: float
    violation: float
    penalty: float
    status: str
    hint: str | None


class Diagnostics:
    """Small helper for inspecting field contributions."""

    @staticmethod
    def report(fields: List[Field], state: Any) -> List[FieldReport]:
        out: List[FieldReport] = []
        for f in fields:
            raw = float(f.func(state))
            weighted = raw * float(f.weight)
            out.append(FieldReport(f.name, float(f.weight), raw, weighted))
        return out

    @staticmethod
    def as_dict(reports: List[FieldReport]) -> Dict[str, float]:
        return {r.name: r.weighted_value for r in reports}

    @staticmethod
    def constraints(
        constraints: List[Constraint],
        state: Any,
        *,
        safety_lambda: float = 1e6,
    ) -> List[ConstraintReport]:
        out: List[ConstraintReport] = []
        for c in constraints:
            margin = float(c.func(state))
            violation = max(0.0, -margin)
            penalty = float(safety_lambda) * (violation * violation) * float(c.weight)
            hint = None
            if margin < 0:
                hint = f"Adjust system by at least {abs(margin):.4f} to satisfy {c.name}"
            out.append(
                ConstraintReport(
                    name=c.name,
                    weight=float(c.weight),
                    margin=margin,
                    violation=violation,
                    penalty=penalty,
                    status="OK" if margin >= 0 else "VIOLATION",
                    hint=hint,
                )
            )
        return out

    @staticmethod
    def constraints_as_dict(reports: List[ConstraintReport]) -> Dict[str, float]:
        return {r.name: r.margin for r in reports}
