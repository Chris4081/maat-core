import numpy as np
from maat_core import Field, Constraint, MaatCore


def scalar_state_fn(x):
    x = float(x)
    return type("State", (), {
        "cost": (x - 0.25) ** 2,
        "complexity": x + 1.0,
        "val": x,
    })


def vector_state_fn(x):
    x = np.asarray(x, dtype=float)
    return type("State", (), {
        "x": x,
    })


def test_evaluate_matches_integrate_and_exposes_breakdown():
    cost = Field("Cost", lambda s: s.cost, weight=1.5)
    respect = Constraint("Respect", lambda s: 0.6 - s.val, weight=2.0)
    core = MaatCore([cost], constraints=[respect], safety_lambda=100.0, occam_lambda=0.25)

    state = scalar_state_fn(0.8)
    evaluation = core.evaluate(state)

    assert np.isclose(evaluation["total"], core.integrate(state))
    assert np.isclose(
        evaluation["field_total"],
        sum(item["weighted_value"] for item in evaluation["fields"]),
    )
    assert evaluation["constraint_penalty"] > 0.0
    assert evaluation["violations"] == 1
    assert evaluation["feasible"] is False
    assert evaluation["min_margin"] < 0.0


def test_constraint_report_schema_remains_backward_compatible():
    cost = Field("Cost", lambda s: s.cost, weight=1.0)
    respect = Constraint("Respect", lambda s: 0.6 - s.val)
    core = MaatCore([cost], constraints=[respect])

    report = core.constraint_report(scalar_state_fn(0.4))

    assert len(report) == 1
    assert set(report[0].keys()) == {"constraint", "margin", "status", "hint"}
    assert report[0]["constraint"] == "Respect"
    assert report[0]["status"] == "OK"
    assert report[0]["hint"] is None


def test_seek_keeps_scalar_and_vector_compatibility():
    scalar_core = MaatCore([Field("Cost", lambda s: s.cost, weight=1.0)])
    scalar_res = scalar_core.seek(
        scalar_state_fn,
        x0=0.9,
        bounds=[(0.0, 1.0)],
        maxiter=200,
    )
    assert abs(float(np.atleast_1d(scalar_res.x)[0]) - 0.25) < 1e-3

    vector_core = MaatCore([
        Field("Quadratic", lambda s: (s.x[0] - 0.25) ** 2 + (s.x[1] - 0.75) ** 2, weight=1.0)
    ])
    vector_res = vector_core.seek(
        vector_state_fn,
        x0=np.array([0.9, 0.1]),
        bounds=[(0.0, 1.0)],
        maxiter=200,
    )
    assert np.allclose(vector_res.x, [0.25, 0.75], atol=1e-3)
