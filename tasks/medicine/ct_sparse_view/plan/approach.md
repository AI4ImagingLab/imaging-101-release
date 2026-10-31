# Approach: Sparse-View CT Reconstruction

## Problem Summary

Reconstruct a 256x256 Shepp-Logan phantom from 30 noisy Radon projections (out of 180). The sparse angular sampling makes FBP insufficient; TV regularization exploits the piecewise-constant phantom structure.

## Algorithm: Chambolle-Pock Primal-Dual (PDHG)

The reconstruction solves:

$$\min_{\mathbf{f} \geq 0} \frac{1}{2}\|\mathbf{A}\mathbf{f} - \mathbf{g}\|_2^2 + \lambda \cdot \text{TV}(\mathbf{f})$$

where:
- $\mathbf{A}$: discrete Radon transform (scikit-image `radon` with `circle=True`)
- $\mathbf{g}$: noisy sparse sinogram (256 detectors x 30 angles)
- $\text{TV}(\mathbf{f}) = \sum_{i,j} \sqrt{(\nabla_x f)_{i,j}^2 + (\nabla_y f)_{i,j}^2}$: isotropic total variation
- $\lambda$: regularization weight

### PDHG Iterations

Initialize $\mathbf{f}^0$ with FBP reconstruction, dual variable $\mathbf{p}^0 = 0$.

For $k = 0, 1, \ldots, N_\text{iter}-1$:

1. **Dual update:** $\mathbf{p}^{k+1} = \text{proj}_{\|\cdot\|_\infty \leq \lambda}(\mathbf{p}^k + \sigma \nabla \bar{\mathbf{f}}^k)$
   - Projection clips per-pixel gradient magnitude to $\lambda$
2. **Primal update:** $\mathbf{f}^{k+1} = \max(0, \mathbf{f}^k - \tau[\mathbf{A}^T(\mathbf{A}\mathbf{f}^k - \mathbf{g})] + \tau \text{div}(\mathbf{p}^{k+1}))$
   - Data fidelity gradient via forward projection + backprojection
   - TV gradient via divergence of dual variable
   - Non-negativity projection
3. **Extrapolation:** $\bar{\mathbf{f}}^{k+1} = \mathbf{f}^{k+1} + \theta(\mathbf{f}^{k+1} - \mathbf{f}^k)$

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| $\lambda$ | 0.01 | Balances data fidelity and TV smoothness; tuned for Shepp-Logan at SNR~50 |
| $\tau$ (primal step) | 0.01 | Conservative for convergence given operator norm of A |
| $\sigma$ (dual step) | 0.5 | Dual step for TV proximal |
| $\theta$ | 1.0 | Standard extrapolation for PDHG |
| $N_\text{iter}$ | 300 | Sufficient for convergence (loss plateaus by ~200) |
| Positivity | True | Physical constraint: attenuation >= 0 |

### Initialization

FBP with ramp filter on the sparse sinogram. This provides a reasonable starting point that accelerates convergence compared to zero initialization.

## Baseline: Filtered Back Projection

FBP applies frequency-domain filtering (ramp filter) to the sinogram then backprojects. With 30 views it produces streak artifacts (NCC ~ 0.81, NRMSE ~ 0.18).

## Expected Results

| Method | NCC | NRMSE | SSIM |
|--------|-----|-------|------|
| FBP (30 views) | 0.81 | 0.18 | 0.16 |
| TV-PDHG (30 views) | 0.97 | 0.07 | 0.43 |

TV-PDHG recovers smooth regions well and preserves ellipse boundaries. Remaining error is mostly at sharp edges and low-contrast features.
