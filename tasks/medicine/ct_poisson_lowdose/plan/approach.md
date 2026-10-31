# Approach: Low-Dose CT Reconstruction with Poisson Noise

## Problem Summary

Reconstruct a 256x256 attenuation image from 256-view parallel-beam sinogram data corrupted by Poisson photon-counting noise. The key challenge is that noise variance varies across measurements: rays through thick/dense material receive fewer photons and have higher noise.

## Algorithm

### Penalized Weighted Least Squares (PWLS) via SVMBIR

The reconstruction minimizes:

$$\hat{x} = \arg\min_x \; \frac{1}{2\sigma_y^2}(\mathbf{y} - \mathbf{A}x)^T \mathbf{W} (\mathbf{y} - \mathbf{A}x) + R(x)$$

where:
- $\mathbf{y}$: post-log sinogram measurements
- $\mathbf{A}$: parallel-beam Radon system matrix
- $\mathbf{W} = \text{diag}(I_1, \ldots, I_M)$: weights from photon counts
- $R(x)$: q-GGMRF prior for edge-preserving regularization
- $\sigma_y$: noise parameter (set automatically from SNR)

### q-GGMRF Prior

The prior encourages piecewise smoothness:

$$R(x) = \sum_{(i,j) \in \mathcal{N}} b_{ij} \frac{|x_i - x_j|^p / (\sigma_x^p \cdot p)}{1 + |x_i - x_j|^{q-p} / (T\sigma_x)^{q-p}}$$

Parameters:
- $p = 1.2$: controls behavior for large differences (edge-preserving when $p < 2$)
- $q = 2.0$: controls behavior for small differences (quadratic smoothing)
- $T = 0.1$: transition threshold between the two regimes
- $\sigma_x$: scale parameter (set automatically from `sharpness=0.0`)

### Solver: Multi-Resolution Super-Voxel ICD

SVMBIR uses:
1. **Multi-resolution hierarchy** (2 additional levels): coarse-to-fine initialization to avoid local minima and speed convergence.
2. **Iterative Coordinate Descent (ICD)**: updates one pixel at a time with an exact 1D minimization, using precomputed sufficient statistics from the system matrix.
3. **Super-voxel grouping**: processes voxels in groups that share similar system matrix rows, enabling cache-efficient computation.

### Convergence

- `stop_threshold = 0.02` (2% average change between iterations)
- `max_iterations = 100` (rarely reached; typically converges in 10-25 iterations)

## Pipeline

1. **Generate data**: Shepp-Logan phantom -> SVMBIR forward projection -> Poisson noise simulation at two dose levels.
2. **Unweighted baseline**: SVMBIR reconstruction with uniform weights (ignoring Poisson statistics).
3. **PWLS reconstruction**: SVMBIR reconstruction with $w_i = I_i$ weights at both low and high dose.
4. **Evaluation**: NCC and NRMSE on 80% centre crop against ground truth.

## Expected Results

| Method | NCC | NRMSE |
|--------|-----|-------|
| Unweighted SVMBIR (I0=1000) | ~0.990 | ~0.066 |
| PWLS SVMBIR (I0=1000) | ~0.997 | ~0.037 |
| PWLS SVMBIR (I0=50000) | ~0.999 | ~0.018 |
