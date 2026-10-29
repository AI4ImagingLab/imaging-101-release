# Approach: MRI T2 Mapping

## Algorithm Overview

We estimate T2 relaxation times from multi-echo spin-echo MRI data using per-pixel curve fitting of the mono-exponential decay model.

## Two-Stage Fitting Strategy

### Stage 1: Log-Linear Initialization

The mono-exponential model S = M0 * exp(-TE / T2) is linearized by taking the logarithm:

    log(S) = log(M0) - TE / T2

This is solved as a linear regression problem:
- Design matrix: A = [1, TE] (N_echoes x 2)
- Observation vector: y = log(S) (N_echoes x 1)
- Solution: [log(M0), -1/T2] = (A^T A)^{-1} A^T y

This is computed for all masked pixels simultaneously using vectorized least squares.

**Limitations:**
- Log transform amplifies noise at low signal values
- Rician noise bias causes systematic T2 overestimation
- Negative or zero signal values require clamping before log

### Stage 2: Nonlinear Least-Squares Refinement

Starting from the log-linear estimate, refine T2 and M0 per pixel by minimizing:

    min_{M0, T2} sum_n (S_n - M0 * exp(-TE_n / T2))^2

using a hand-coded Levenberg-Marquardt solver (ported from the classical formulation that scipy.optimize.curve_fit wraps internally). Bounds: M0 >= 0, T2 in [0.1, 5000] ms.

**Solver parameters:**
- Maximum function evaluations per pixel: 500
- Bounds: M0 in [0, inf), T2 in [0.1, 5000] ms
- Fallback: if fitting fails, use log-linear estimate

## Evaluation

Metrics are computed only within the tissue mask (T2 > 0):
- **NCC** (cosine similarity) between estimated and ground truth T2 maps
- **NRMSE** normalized by dynamic range of the reference T2 map
