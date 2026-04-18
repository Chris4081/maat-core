import numpy as np
from maat_core import Constraint, ConstraintReport, Diagnostics, Field, MaatCore


def scalar_state_fn(x):
    x = float(x)
    return type("State", (), {
        "cost": (x - 0.25) ** 2,
        "complexity": x + 1.0,
        "val": x,
    })


def test_constraint_diagnostics_include_margin_violation_and_penalty():
    state = scalar_state_fn(0.8)
    respect = Constraint("Respect", lambda s: 0.6 - s.val, weight=2.0)

    reports = Diagnostics.constraints([respect], state, safety_lambda=100.0)

    assert len(reports) == 1
    assert isinstance(reports[0], ConstraintReport)
    assert reports[0].name == "Respect"
    assert reports[0].status == "VIOLATION"
    assert np.isclose(reports[0].margin, -0.2)
    assert np.isclose(reports[0].violation, 0.2)
    assert np.isclose(reports[0].penalty, 8.0)
    assert Diagnostics.constraints_as_dict(reports) == {"Respect": reports[0].margin}


def test_seek_trace_returns_trace_and_best_evaluation():
    cost = Field("Cost", lambda s: s.cost, weight=1.0)
    respect = Constraint("Respect", lambda s: 0.6 - s.val)
    core = MaatCore([cost], constraints=[respect], safety_lambda=100.0, occam_lambda=0.25)
    baseline = core.seek(
        scalar_state_fn,
        x0=[0.9],
        bounds=[(0.0, 1.0)],
        maxiter=200,
    )

    traced = core.seek_trace(
        scalar_state_fn,
        x0=[0.9],
        bounds=[(0.0, 1.0)],
        maxiter=200,
        trace_every=2,
    )

    result = traced["result"]
    trace = traced["trace"]
    best_evaluation = traced["best_evaluation"]

    assert np.allclose(result.x, baseline.x, atol=1e-6)
    assert np.isclose(float(result.fun), float(baseline.fun))
    assert len(trace) > 0
    assert {
        "step",
        "x",
        "objective",
        "field_total",
        "occam_penalty",
        "constraint_penalty",
        "feasible",
        "min_margin",
        "violations",
    }.issubset(trace[0].keys())
    assert np.isclose(best_evaluation["total"], core.integrate(traced["best_state"]))
    assert best_evaluation["feasible"] is True
