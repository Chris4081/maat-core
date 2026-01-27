import numpy as np
from maat_core import Field, Constraint, MaatCore, Diagnostics


def state_fn(x: float):
    x = float(x)
    return type("State", (), {
        "dissonance": np.sin(np.pi * x) ** 2,
        "complexity": np.exp(x),
        "val": x,
    })


def test_integrate_occam_prefers_simpler_when_equal_harmony():
    # Minima at x=0 and x=1 have equal harmony=0, but complexity differs.
    H = Field("Harmony", lambda s: s.dissonance, weight=1.0)
    core = MaatCore([H], occam_lambda=0.01)

    s0 = state_fn(0.0)
    s1 = state_fn(1.0)

    assert core.integrate(s0) < core.integrate(s1)


def test_diagnostics_outputs_field_values():
    H = Field("Harmony", lambda s: s.dissonance, weight=0.9)
    core = MaatCore([H], occam_lambda=0.01)

    st = state_fn(0.25)
    rep = Diagnostics.report(core.fields, st)
    assert rep[0].name == "Harmony"
    assert rep[0].weighted_value == rep[0].raw_value * rep[0].weight


def test_respect_constraint_penalizes_violation():
    # Respect: keep x <= 0.6
    H = Field("Harmony", lambda s: s.dissonance, weight=1.0)
    R = Constraint("Respect", lambda s: 0.6 - s.val, weight=1.0)
    core = MaatCore([H], constraints=[R], occam_lambda=0.0, safety_lambda=1e6)

    ok = state_fn(0.5)
    bad = state_fn(0.9)

    assert core.integrate(bad) > core.integrate(ok)
