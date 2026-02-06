import time
import numpy as np
from maat_core import Field, Constraint, MaatCore


class VecState:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)


def state_fn(x_vec):
    return VecState(np.asarray(x_vec, dtype=float))


def build_feasible_linear_system(rng, n_dim, n_constraints, x_feas, slack=0.25):
    """
    Build constraints a_i^T x <= b_i such that x_feas is guaranteed feasible:
    b_i = a_i^T x_feas + slack_i, where slack_i >= slack > 0.
    """
    A = rng.normal(size=(n_constraints, n_dim))
    slack_vec = rng.uniform(slack, 2.0 * slack, size=n_constraints)
    b = A @ x_feas + slack_vec
    return A, b


def make_problem(n_dim=8, n_constraints=25, seed=7, infeasible=False, break_k=0):
    rng = np.random.default_rng(seed)

    # Choose a known feasible anchor point (inside bounds)
    x_feas = rng.uniform(-0.5, 0.5, size=n_dim)

    # Natural optimum target (can be outside feasible set)
    target = rng.uniform(-1.0, 1.0, size=n_dim)

    # Build constraints guaranteed feasible at x_feas
    A, b = build_feasible_linear_system(rng, n_dim, n_constraints, x_feas, slack=0.25)

    # Optionally "break" k constraints so the system becomes infeasible around x_feas
    # This is useful to demonstrate structural infeasibility detection.
    if infeasible and break_k > 0:
        break_k = min(break_k, n_constraints)
        # Make some constraints impossible by lowering b dramatically
        b[:break_k] = b[:break_k] - 10.0  # strong shift -> violates at most points

    def utility_field(state: VecState):
        return float(np.sum((state.x - target) ** 2))

    field = Field("UtilityDistanceToTarget", utility_field, weight=1.0)

    constraints = []
    for i in range(n_constraints):
        ai = A[i].copy()
        bi = float(b[i])
        constraints.append(
            Constraint(
                f"lin_{i:04d}",
                lambda s, ai=ai, bi=bi: bi - float(np.dot(ai, s.x))
            )
        )

    bounds = [(-2.0, 2.0)] * n_dim

    return field, constraints, target, bounds, x_feas


def summarize_margins(core, x_best):
    s = state_fn(x_best)
    margins = np.array([float(c.func(s)) for c in core.constraints], dtype=float)
    return {
        "min_margin": float(np.min(margins)),
        "mean_margin": float(np.mean(margins)),
        "violations": int(np.sum(margins < 0.0)),
        "n_constraints": int(margins.size),
    }


def pick_x(result):
    if hasattr(result, "x_best") and getattr(result, "x_best") is not None:
        return np.asarray(getattr(result, "x_best"), dtype=float)
    if hasattr(result, "x") and getattr(result, "x") is not None:
        return np.asarray(getattr(result, "x"), dtype=float)
    if isinstance(result, dict) and result.get("x") is not None:
        return np.asarray(result["x"], dtype=float)
    raise ValueError("Could not extract solution vector from result.")


def pick_fun(result):
    if hasattr(result, "objective") and getattr(result, "objective") is not None:
        return float(getattr(result, "objective"))
    if hasattr(result, "fun") and getattr(result, "fun") is not None:
        return float(getattr(result, "fun"))
    if isinstance(result, dict) and result.get("fun") is not None:
        return float(result["fun"])
    return float("nan")


def run_case(n_dim, n_constraints, safety_lambda=1e6, seed=7, infeasible=False, break_k=0):
    field, constraints, target, bounds, x_feas = make_problem(
        n_dim=n_dim,
        n_constraints=n_constraints,
        seed=seed,
        infeasible=infeasible,
        break_k=break_k
    )

    core = MaatCore(
        fields=[field],
        constraints=constraints,
        safety_lambda=safety_lambda,
        occam_lambda=0.0
    )

    # Start point: the known feasible anchor (good to show it stays feasible)
    x0 = np.asarray(x_feas, dtype=float)

    t0 = time.perf_counter()
    result = core.seek(
        state_fn=state_fn,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
    )
    dt = time.perf_counter() - t0

    x_best = pick_x(result)
    fun = pick_fun(result)
    stats = summarize_margins(core, x_best)

    return dt, fun, stats, x_best, target


def main():
    n_dim = 8
    sizes = [3, 10, 25, 100, 300]  # extend to 1000 if you like
    print("=== MAAT-Core Many-Constraints Proof (Feasible-by-Construction) ===")
    print(f"dim = {n_dim}")
    print()

    print("--- FEASIBLE SCALING RUNS ---")
    for m in sizes:
        dt, fun, stats, x_best, target = run_case(n_dim, m, infeasible=False)
        print(f"m={m:4d} | time={dt*1000:7.1f} ms | fun={fun:10.6f} | "
              f"min_margin={stats['min_margin']: .6f} | violations={stats['violations']}")

    print()
    print("--- INFEASIBILITY DEMO (BREAK SOME CONSTRAINTS) ---")
    dt, fun, stats, x_best, target = run_case(n_dim, 50, infeasible=True, break_k=5)
    print(f"m=  50 (with 5 broken) | time={dt*1000:7.1f} ms | fun={fun:10.6f} | "
          f"min_margin={stats['min_margin']: .6f} | violations={stats['violations']}")
    print("Expected: persistent negative margins -> structural infeasibility signal (not a solver limit).")


if __name__ == "__main__":
    main()