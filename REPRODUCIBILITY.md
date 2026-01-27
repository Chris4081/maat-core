# Reproducibility (MAAT-Core Examples)

This repository includes minimal, reproducible toy demos for:
1) **Occam tie-breaker** (complexity regularization)  
2) **Respect as a hard constraint** (safety-first boundary corridor)

## Requirements
- Python 3.10+ (tested with 3.11)
- NumPy
- SciPy

## Install (editable)
From the repo root:

```bash
python3 -m pip install -U pip
python3 -m pip install -e .
```

If you run into permission issues on macOS:

```bash
python3 -m pip install --user -e .
```

## Run demos

```bash
cd examples

# Occam demo
python3 occam_demo.py

# Respect demo
python3 respect_boundary_demo.py
```

## Expected output (roughly)

### occam_demo.py
You should see:
- `x_best` close to the simpler basin (e.g. ~0.1)
- a breakdown like:
  - Harmony contribution  
  - Occam contribution (`occam_lambda * complexity`)

### respect_boundary_demo.py
You should see:
- best x inside the allowed region
- a constraint diagnostic report such as:
  - status: OK  
  - margin: small positive value (near boundary is normal)

## What “Respect” means here

Respect (R) is implemented as a hard inequality constraint:

- Each constraint is a margin function `g(state) >= 0`
- Violations are penalized with a large quadratic term
- With sufficiently large `safety_lambda`, unsafe solutions become unstable minima

This provides a safety-first default without post-hoc filtering.

## Repro tips

Use a fixed random seed for annealing:
```python
seed=42
```

Optional: lock exact versions
```bash
python3 -m pip freeze > requirements-lock.txt
```
