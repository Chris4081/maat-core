from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from .core import Field


@dataclass(frozen=True)
class FieldReport:
    name: str
    weight: float
    raw_value: float
    weighted_value: float


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
