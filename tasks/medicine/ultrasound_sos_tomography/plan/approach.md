# Approach: Ultrasound Speed-of-Sound Tomography

## Problem Summary

Reconstruct a 128x128 speed-of-sound map from 60 noisy travel-time projections through a breast-mimicking phantom. The reconstruction operates on the slowness perturbation (delta_s = 1/c - 1/c_water) using the Radon transform as the forward model. FBP provides a reasonable baseline but iterative TV-regularized reconstruction better resolves tissue boundaries.

## Algorithm 1: Filtered Back Projection (Baseline)

FBP applies the ramp filter in the frequency domain and backprojects. It is the standard non-iterative reconstruction for Radon-type problems. With 60 projection angles, FBP produces mild streak artifacts and blurs tissue boundaries.

## Algorithm 2: SART

SART (Simultaneous Algebraic Reconstruction Technique) iteratively updates the image by:

1. Forward-projecting the current estimate: $\mathbf{t}_\text{est} = \mathbf{A}\Delta\hat{\mathbf{s}}$
2. Computing the residual: $\mathbf{r} = \mathbf{t}_\text{est} - \mathbf{t}_\text{meas}$
3. Backprojecting the normalised residual: $\Delta\hat{\mathbf{s}} \leftarrow \Delta\hat{\mathbf{s}} - \alpha \cdot \mathbf{A}^T \mathbf{r}$

Initialized with FBP. No explicit regularization, but the relaxation parameter and limited iterations provide implicit regularization.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_iter | 30 | Sufficient for convergence |
| relaxation | 0.15 | Conservative step size for stability |

## Algorithm 3: TV-PDHG (Main Method)

The Chambolle-Pock primal-dual hybrid gradient algorithm solves:

$$\min_{\Delta\mathbf{s}} \frac{1}{2}\|\mathbf{A}\Delta\mathbf{s} - \mathbf{t}\|_2^2 + \lambda \cdot \text{TV}(\Delta\mathbf{s})$$

where $\text{TV}(\Delta\mathbf{s}) = \sum_{i,j} \sqrt{(\nabla_x \Delta s)_{i,j}^2 + (\nabla_y \Delta s)_{i,j}^2}$ is the isotropic total variation.

### PDHG Iterations

Initialize $\Delta\hat{\mathbf{s}}^0$ with FBP, dual variable $\mathbf{p}^0 = 0$.

For $k = 0, 1, \ldots, N_\text{iter}-1$:

1. **Dual update:** $\mathbf{p}^{k+1} = \text{proj}_{\|\cdot\|_\infty \leq \lambda}(\mathbf{p}^k + \sigma \nabla \bar{\Delta\hat{\mathbf{s}}}^k)$
2. **Primal update:** $\Delta\hat{\mathbf{s}}^{k+1} = \Delta\hat{\mathbf{s}}^k - \tau[\mathbf{A}^T(\mathbf{A}\Delta\hat{\mathbf{s}}^k - \mathbf{t})] + \tau \text{div}(\mathbf{p}^{k+1})$
3. **Extrapolation:** $\bar{\Delta\hat{\mathbf{s}}}^{k+1} = \Delta\hat{\mathbf{s}}^{k+1} + \theta(\Delta\hat{\mathbf{s}}^{k+1} - \Delta\hat{\mathbf{s}}^k)$

No positivity constraint is applied because the slowness perturbation can be negative (fat has lower SoS than water).

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $\lambda$ | 1e-6 | Balances data fidelity and TV; tuned for slowness perturbation scale (~1e-4 s/m) |
| $\tau$ (primal step) | 0.01 | Conservative for convergence |
| $\sigma$ (dual step) | 0.5 | Standard dual step for PDHG |
| $\theta$ | 1.0 | Standard extrapolation |
| $N_\text{iter}$ | 300 | Loss plateaus by ~200 iterations |

## Expected Results

| Method | NCC | NRMSE | SSIM |
|--------|-----|-------|------|
| FBP (60 angles) | 0.970 | 0.019 | 0.875 |
| SART (30 iter) | 0.971 | 0.019 | 0.882 |
| TV-PDHG (300 iter) | 0.983 | 0.015 | 0.970 |

TV-PDHG provides the best reconstruction, recovering sharp tissue boundaries and suppressing streak artifacts. The piecewise-constant phantom structure is well matched to TV regularization.
