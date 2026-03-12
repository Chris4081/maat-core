#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark healthcare_vs_scipy.py
--------------------------------
Compare MAAT-Core against native SciPy constrained optimizers
on a simple healthcare bed-allocation problem.

Goal:
- Maximize total lives saved
- Enforce fairness (minimum beds per department)
- Respect total capacity

Compared methods:
1. MAAT-Core (penalty + margin diagnostics)
2. SciPy SLSQP (native constraints)
3. SciPy trust-constr (native constraints)

Outputs:
- console summary
- CSV file with benchmark results

"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
import csv

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

from maat_core import Field, Constraint, MaatCore


# ============================================================
# Problem definition
# ============================================================

CAPACITY = 200.0
MIN_COVID = 50.0
MIN_HEART = 50.0
MIN_CANCER = 50.0

# Lives saved per bed
COVID_FACTOR = 5.0
HEART_FACTOR = 3.0
CANCER_FACTOR = 4.0

# Start point (intentionally biased)
X0 = np.array([120.0, 20.0, 10.0], dtype=float)

# Simple box bounds
LOWER = np.array([0.0, 0.0, 0.0], dtype=float)
UPPER = np.array([200.0, 200.0, 200.0], dtype=float)


@dataclass
class ResultRow:
    method: str
    success: bool
    runtime_sec: float
    objective_value: float
    lives_saved: float
    min_margin: float
    violations: int
    x_covid: float
    x_heart: float
    x_cancer: float
    diagnostics: str


def state_fn(x):
    x = np.asarray(x, dtype=float)
    return type("State", (), {
        "covid_saved": COVID_FACTOR * x[0],
        "heart_saved": HEART_FACTOR * x[1],
        "cancer_saved": CANCER_FACTOR * x[2],
        "total_beds": float(np.sum(x)),
        "x": x
    })


def lives_saved(x: np.ndarray) -> float:
    return COVID_FACTOR * x[0] + HEART_FACTOR * x[1] + CANCER_FACTOR * x[2]


def objective_for_scipy(x: np.ndarray) -> float:
    # minimize negative lives saved
    return -lives_saved(np.asarray(x, dtype=float))


def margins(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.array([
        CAPACITY - np.sum(x),     # total capacity
        x[0] - MIN_COVID,         # fairness covid
        x[1] - MIN_HEART,         # fairness heart
        x[2] - MIN_CANCER,        # fairness cancer
    ], dtype=float)


def min_margin(x: np.ndarray) -> float:
    return float(np.min(margins(x)))


def violation_count(x: np.ndarray) -> int:
    return int(np.sum(margins(x) < 0.0))


# ============================================================
# MAAT-Core benchmark
# ============================================================

def run_maat_core() -> ResultRow:
    LivesSaved = Field(
        "LivesSaved",
        lambda s: -(s.covid_saved + s.heart_saved + s.cancer_saved),
        weight=1.0,
    )

    TotalCapacity = Constraint("TotalCapacity", lambda s: CAPACITY - s.total_beds)
    FairnessCovid = Constraint("FairnessCovid", lambda s: s.x[0] - MIN_COVID)
    FairnessHeart = Constraint("FairnessHeart", lambda s: s.x[1] - MIN_HEART)
    FairnessCancer = Constraint("FairnessCancer", lambda s: s.x[2] - MIN_CANCER)

    core = MaatCore(
        [LivesSaved],
        constraints=[TotalCapacity, FairnessCovid, FairnessHeart, FairnessCancer],
        safety_lambda=1e6,
        occam_lambda=0.0
    )

    t0 = time.perf_counter()
    res = core.seek(
        state_fn,
        x0=X0,
        bounds=list(zip(LOWER, UPPER))
    )
    dt = time.perf_counter() - t0

    # robust extraction
    x = np.asarray(getattr(res, "x", None) if hasattr(res, "x") else res["x"], dtype=float)
    obj = float(getattr(res, "fun", None) if hasattr(res, "fun") else res["fun"])

    report_text = "margin report available"

    return ResultRow(
        method="MAAT-Core",
        success=True,
        runtime_sec=dt,
        objective_value=obj,
        lives_saved=lives_saved(x),
        min_margin=min_margin(x),
        violations=violation_count(x),
        x_covid=float(x[0]),
        x_heart=float(x[1]),
        x_cancer=float(x[2]),
        diagnostics=report_text,
    )


# ============================================================
# SciPy SLSQP benchmark
# ============================================================

def run_slsqp() -> ResultRow:
    cons = [
        {"type": "ineq", "fun": lambda x: CAPACITY - np.sum(x)},
        {"type": "ineq", "fun": lambda x: x[0] - MIN_COVID},
        {"type": "ineq", "fun": lambda x: x[1] - MIN_HEART},
        {"type": "ineq", "fun": lambda x: x[2] - MIN_CANCER},
    ]

    t0 = time.perf_counter()
    res = minimize(
        objective_for_scipy,
        X0,
        method="SLSQP",
        bounds=list(zip(LOWER, UPPER)),
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9, "disp": False},
    )
    dt = time.perf_counter() - t0

    x = np.asarray(res.x, dtype=float)

    return ResultRow(
        method="SciPy-SLSQP",
        success=bool(res.success),
        runtime_sec=dt,
        objective_value=float(res.fun),
        lives_saved=lives_saved(x),
        min_margin=min_margin(x),
        violations=violation_count(x),
        x_covid=float(x[0]),
        x_heart=float(x[1]),
        x_cancer=float(x[2]),
        diagnostics="native constraints, no margin report",
    )


# ============================================================
# SciPy trust-constr benchmark
# ============================================================

def run_trust_constr() -> ResultRow:
    bounds = Bounds(LOWER, UPPER)

    nlc = NonlinearConstraint(
        fun=lambda x: margins(x),
        lb=np.zeros(4),
        ub=np.full(4, np.inf),
    )

    t0 = time.perf_counter()
    res = minimize(
        objective_for_scipy,
        X0,
        method="trust-constr",
        bounds=bounds,
        constraints=[nlc],
        options={"maxiter": 500, "verbose": 0},
    )
    dt = time.perf_counter() - t0

    x = np.asarray(res.x, dtype=float)

    return ResultRow(
        method="SciPy-trust-constr",
        success=bool(res.success),
        runtime_sec=dt,
        objective_value=float(res.fun),
        lives_saved=lives_saved(x),
        min_margin=min_margin(x),
        violations=violation_count(x),
        x_covid=float(x[0]),
        x_heart=float(x[1]),
        x_cancer=float(x[2]),
        diagnostics="native constraints, no margin report",
    )


# ============================================================
# Output helpers
# ============================================================

def print_summary(rows: list[ResultRow]) -> None:
    print("\n=== Healthcare Benchmark: MAAT-Core vs SciPy ===\n")
    for r in rows:
        print(f"Method       : {r.method}")
        print(f"Success      : {r.success}")
        print(f"Runtime [s]  : {r.runtime_sec:.6f}")
        print(f"Objective    : {r.objective_value:.6f}")
        print(f"Lives saved  : {r.lives_saved:.2f}")
        print(f"Min margin   : {r.min_margin:.6f}")
        print(f"Violations   : {r.violations}")
        print(f"x            : [{r.x_covid:.2f}, {r.x_heart:.2f}, {r.x_cancer:.2f}]")
        print(f"Diagnostics  : {r.diagnostics}")
        print("-" * 60)


def save_csv(rows: list[ResultRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    print(f"\nSaved CSV: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    rows = [
        run_maat_core(),
        run_slsqp(),
        run_trust_constr(),
    ]

    print_summary(rows)
    save_csv(rows, Path("runs/benchmark_healthcare_vs_scipy.csv"))

    print("\nInterpretation:")
    print("- MAAT-Core uses penalty-style safety-first optimization.")
    print("- SLSQP and trust-constr use native numerical constraints.")
    print("- MAAT-Core's main advantage is interpretable margin diagnostics and safety semantics,")
    print("  not necessarily raw numerical superiority.")
    print()


if __name__ == "__main__":
    main()
