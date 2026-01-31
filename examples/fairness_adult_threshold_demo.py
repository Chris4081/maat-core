"""
MAAT-Core Fairness Benchmark (Adult Income)
-------------------------------------------
State = classification threshold t in [0, 1]
Field = minimize error (1 - accuracy)
Constraint (Respect) = demographic parity gap <= eps

Outputs:
- Baseline metrics at t=0.5
- MAAT-optimized metrics
- Reflection loop log (CSV) for plotting in paper

NOTE:
This demo requires internet access (OpenML) for data download.
It is intended as a benchmark example, not a core dependency of MAAT-Core.
On macOS you may need:

export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

If OpenML is unavailable, consider replacing load_adult()
with a local CSV version for full reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from maat_core import Field, Constraint, MaatCore


# -----------------------------
# Data + Model
# -----------------------------
def load_adult():
    adult = fetch_openml("adult", version=2, as_frame=True)
    X = adult.data
    y = (adult.target == '>50K').astype(int)   # <<< DAS FEHLT
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series):
    # Identify categorical/numerical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = LogisticRegression(max_iter=200, n_jobs=None)  # n_jobs may be ignored depending on solver
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe


# -----------------------------
# Metrics
# -----------------------------
def demographic_parity_gap(y_pred: np.ndarray, sensitive: pd.Series, pos_label: int = 1) -> float:
    # rates P(ŷ=1 | group)
    s = sensitive.astype(str).str.strip()
    groups = sorted(s.unique())

    rates = []
    for g in groups:
        idx = (s == g).to_numpy()
        if idx.sum() == 0:
            continue
        rates.append(float((y_pred[idx] == pos_label).mean()))

    if len(rates) < 2:
        return 0.0
    return float(np.max(rates) - np.min(rates))


# -----------------------------
# MAAT State function
# -----------------------------
def make_state_fn(pipe, X_test: pd.DataFrame, y_test: pd.Series, sensitive_series: pd.Series):
    # Precompute probabilities once per evaluation call (threshold changes often -> probs stay constant)
    # But sklearn pipeline predicts proba depending on X -> we can compute on demand; for simplicity, compute once:
    proba = pipe.predict_proba(X_test)[:, 1]

    def state_fn(t):
        # t comes as scalar or array -> normalize
        t = float(np.atleast_1d(t)[0])
        y_pred = (proba >= t).astype(int)

        acc = float(accuracy_score(y_test, y_pred))
        gap = float(demographic_parity_gap(y_pred, sensitive_series))
        # optional: complexity proxy
        complexity = float(abs(t - 0.5))  # tiny proxy (not important, but exists)

        return type("State", (), {
            "t": t,
            "accuracy": acc,
            "error": 1.0 - acc,
            "dp_gap": gap,
            "complexity": complexity,
        })

    return state_fn


# -----------------------------
# Reflection loop helper
# -----------------------------
def reflection_loop(core: MaatCore, state_fn, x0, bounds, max_steps=10):
    log = []
    for i in range(max_steps):
        res = core.seek(state_fn, x0=x0, bounds=bounds, maxiter=2000)
        st = state_fn(res.x)

        # minimum margin across constraints (>=0 is safe)
        margins = [float(c.func(st)) for c in core.constraints]
        min_margin = float(np.min(margins))
        log.append({
            "iter": i,
            "lambda_safety": core.safety_lambda,
            "min_margin": min_margin,
            "t": float(np.atleast_1d(res.x)[0]),
            "accuracy": st.accuracy,
            "dp_gap": st.dp_gap,
        })

        # adapt safety
        if min_margin < 0:
            core.safety_lambda *= 2.0
        else:
            # optional relax a bit once stable (keep conservative)
            core.safety_lambda *= 0.9

        # stop when comfortably safe
        if min_margin >= 1e-6:
            break

        x0 = res.x  # warm start

    return pd.DataFrame(log)


def main():
    X, y = load_adult()

    # Sensitive attribute: sex (Adult has 'sex' column)
    if "sex" not in X.columns:
        raise RuntimeError("Adult dataset: expected column 'sex' not found.")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = train_model(X_train, y_train)
    pipe.fit(X_train, y_train)

    sens_test = X_test["sex"].copy()

    state_fn = make_state_fn(pipe, X_test, y_test, sens_test)

    # Baseline at t=0.5
    base = state_fn(0.5)
    print("\n--- BASELINE (t=0.5) ---")
    print(f"accuracy     : {base.accuracy:.4f}")
    print(f"dp_gap       : {base.dp_gap:.4f}")

    # MAAT setup
    eps = 0.05  # fairness tolerance
    F_error = Field("Error", lambda s: s.error, weight=1.0)
    R_fair  = Constraint("DemographicParity", lambda s: eps - s.dp_gap)
    core = MaatCore([F_error], constraints=[R_fair], safety_lambda=1e6)

    bounds = [(0.0, 1.0)]
    x0 = np.array([0.5], dtype=float)

    # Optimize
    res = core.seek(state_fn, x0=x0, bounds=bounds, maxiter=2000)
    st = state_fn(res.x)

    print("\n--- MAAT-OPTIMIZED THRESHOLD ---")
    print(f"t            : {float(np.atleast_1d(res.x)[0]):.4f}")
    print(f"accuracy     : {st.accuracy:.4f}")
    print(f"dp_gap       : {st.dp_gap:.4f}")
    print(f"constraint OK: {st.dp_gap <= eps}")

    # Optional reflection-loop log for paper figure
    core2 = MaatCore([F_error], constraints=[R_fair], safety_lambda=1e4)  # start smaller to show convergence
    df_log = reflection_loop(core2, state_fn, x0=np.array([0.5]), bounds=bounds, max_steps=12)
    df_log.to_csv("adult_fairness_reflection_log.csv", index=False)
    print("\nSaved reflection log: adult_fairness_reflection_log.csv")
    print(df_log)

    # Save summary for LaTeX table
    summary = pd.DataFrame([
        {"method": "baseline(t=0.5)", "accuracy": base.accuracy, "dp_gap": base.dp_gap, "constraint_ok": base.dp_gap <= eps},
        {"method": "maat-core",       "accuracy": st.accuracy,   "dp_gap": st.dp_gap,   "constraint_ok": st.dp_gap <= eps},
    ])
    summary.to_csv("adult_fairness_summary.csv", index=False)
    print("\nSaved summary: adult_fairness_summary.csv")


if __name__ == "__main__":
    main()