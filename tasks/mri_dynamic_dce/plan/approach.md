# Approach: Dynamic Contrast-Enhanced MRI Reconstruction

## Problem Summary

Reconstruct a time-series of 128x128 MRI images (20 frames) from per-frame
undersampled Cartesian k-space (15% sampling rate) using temporal Total
Variation regularization. The key challenge is exploiting temporal redundancy:
most of the image is static, with only small regions exhibiting dynamic
contrast enhancement.

## Algorithm: Proximal Gradient Descent with Temporal TV

### Objective Function

$$\hat{x} = \arg\min_x \sum_{t=1}^{T} \frac{1}{2}\|M_t \mathcal{F} x_t - y_t\|_2^2 + \lambda \sum_{t=1}^{T-1} \|x_{t+1} - x_t\|_1$$

where:
- $x_t \in \mathbb{C}^{N \times N}$: complex image at frame $t$
- $y_t$: undersampled k-space measurements at frame $t$
- $M_t$: binary 2D undersampling mask for frame $t$
- $\mathcal{F}$: centered 2D FFT with ortho normalization
- $\lambda$: temporal TV regularization weight
- The L1 norm of temporal differences promotes piecewise-constant temporal behavior

### Solver: ISTA / Proximal Gradient Descent

Each iteration consists of:

1. **Gradient step** (data fidelity):
   $$x^{k+1/2} = x^k - \eta \cdot \mathcal{F}^H M_t (M_t \mathcal{F} x^k_t - y_t)$$
   where $\eta = 1.0$ (step size equals inverse Lipschitz constant of gradient).

2. **Proximal step** (temporal TV):
   $$x^{k+1} = \mathrm{prox}_{\eta\lambda \|\cdot\|_{TV_t}}(x^{k+1/2})$$
   The proximal operator is computed via Chambolle's dual algorithm
   (20 inner iterations) operating on temporal finite differences.

### Hyperparameters

- Regularization weight: $\lambda = 0.001$
- Maximum PGD iterations: 200
- Convergence tolerance: $10^{-5}$ (relative change in $x$)
- Chambolle inner iterations: 20 per proximal step
- Step size: $\eta = 1.0$ (operator norm of $\mathcal{F}^H M_t \mathcal{F}$ is 1)

### Chambolle's Dual Algorithm for 1D TV Proximal

To solve $\mathrm{prox}_{\lambda TV}(x)$, we use the dual formulation:

$$z = x - \lambda D^H p^*$$

where $p^*$ is obtained by iterating:
$$p^{n+1} = \frac{p^n + \tau D(x - \lambda D^H p^n)}{\max(1, |p^n + \tau D(x - \lambda D^H p^n)|)}$$

with $\tau = 1/4$ (related to the operator norm of the temporal difference operator).

## Baseline: Zero-Filled Reconstruction

Frame-by-frame inverse FFT of the undersampled k-space, taking the magnitude.
This serves as the lower-quality baseline that exhibits temporal flickering
artifacts due to the different random masks per frame.

## Expected Results

| Method        | Avg NRMSE | Avg NCC | Avg PSNR |
|---------------|-----------|---------|----------|
| Zero-filled   | 0.087     | 0.971   | 21.3 dB  |
| Temporal TV   | 0.081     | 0.975   | 21.9 dB  |
