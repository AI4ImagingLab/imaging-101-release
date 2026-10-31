# Approach: Diffusion MRI DTI

## Algorithm Overview

We estimate diffusion tensors from multi-direction DWI data using per-voxel linearized fitting of the Stejskal-Tanner equation, followed by eigendecomposition to derive FA and MD maps.

## Linearization of the Stejskal-Tanner Equation

Taking the logarithm of S_i = S0 * exp(-b_i * g_i^T D g_i):

    ln(S_i) = ln(S0) - b_i * g_i^T D g_i

Expanding the quadratic form g^T D g:

    g^T D g = gx^2*Dxx + 2*gx*gy*Dxy + 2*gx*gz*Dxz + gy^2*Dyy + 2*gy*gz*Dyz + gz^2*Dzz

This gives a linear system: y = B @ p, where:
- y = [ln(S_1), ..., ln(S_N)]^T                         (N_volumes x 1)
- p = [ln(S0), Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]^T        (7 x 1)
- B = design matrix with rows [1, -b*gx^2, -2b*gx*gy, -2b*gx*gz, -b*gy^2, -2b*gy*gz, -b*gz^2]

## Two-Stage Fitting Strategy

### Stage 1: Ordinary Least Squares (OLS)

Solve the linearized system for all masked voxels simultaneously:

    p_ols = (B^T B)^{-1} B^T y

This is computed vectorized across all masked voxels using numpy.linalg.lstsq.

**Limitations:**
- Log transform introduces heteroscedastic noise (variance depends on signal level)
- Rician noise bias at low SNR causes systematic errors
- Negative or zero signal values require clamping before log

### Stage 2: Weighted Least Squares (WLS)

Starting from OLS estimates, re-solve with diagonal weight matrix:

    W = diag(w_1, ..., w_N),  w_i = exp(2 * B_i @ p_ols)

The WLS solution:

    p_wls = (B^T W B)^{-1} B^T W y

Weights w_i = S_i^2 (predicted squared signal) correct for the heteroscedasticity introduced by the log transform, giving a more efficient estimator.

**Solver parameters:**
- Weights clamped to [1e-10, 1e10] to avoid numerical overflow
- Fallback to OLS if WLS system is singular

## Eigendecomposition

For each fitted tensor D (from 6 elements), compute eigenvalues and eigenvectors:

    D = V @ diag(lambda_1, lambda_2, lambda_3) @ V^T

with lambda_1 >= lambda_2 >= lambda_3. Negative eigenvalues are clamped to 0.

Derived scalar maps:
- FA = sqrt(3/2) * sqrt(sum((lambda_i - MD)^2) / sum(lambda_i^2))
- MD = (lambda_1 + lambda_2 + lambda_3) / 3

## Evaluation

Metrics are computed within the tissue mask:
- **NCC** (cosine similarity) between estimated and ground truth FA maps (primary metric)
- **NRMSE** normalized by dynamic range of the reference FA map
- MD map NCC/NRMSE reported as secondary metrics
- Boundary thresholds: 90% of WLS baseline NCC, 110% of WLS baseline NRMSE
