# Approach: Non-Cartesian MRI with L1-Wavelet CS

## Algorithm Overview

We reconstruct a complex image from radially sampled multi-coil k-space using two methods:

1. **Gridding** (baseline): Density-compensated adjoint NUFFT with coil combination.
2. **L1-Wavelet CS**: Iterative proximal gradient minimization of the L1-wavelet regularized objective with NUFFT forward/adjoint.

## Gridding Reconstruction

Density compensation weights $w_m$ are computed via the iterative pipe method (30 iterations):

$$w^{(k+1)}_m = \frac{w^{(k)}_m}{|\mathcal{F}_\text{NU} \mathcal{F}^H_\text{NU} w^{(k)}|_m}$$

The gridding reconstruction then applies:

$$\hat{x}_\text{grid} = \frac{\sum_c \bar{S}_c \cdot \mathcal{F}^H_\text{NU}(w \odot y_c)}{\sqrt{\sum_c |S_c|^2}}$$

## L1-Wavelet CS Solver

SigPy's `L1WaveletRecon` uses ADMM with proximal gradient:

1. **Power iteration** (30 iterations): Estimate maximum eigenvalue of $A^H A$ where $A$ is the multi-coil NUFFT operator, for automatic step-size selection.
2. **Proximal gradient iterations** (100 iterations):
   - Gradient step on data fidelity: $z = x - \alpha A^H(Ax - y)$
   - Proximal step on L1-wavelet: $x = \text{SoftThresh}(\Psi z, \alpha\lambda)$ where $\text{SoftThresh}(u, t) = \text{sign}(u) \max(|u| - t, 0)$

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda` | 5e-5 | L1-wavelet regularization weight |
| `max_iter` (power iteration) | 30 | For step-size estimation |
| `max_iter` (proximal gradient) | 100 | Main reconstruction iterations |
| DCF iterations | 30 | Pipe method for density compensation |

## Preprocessing

1. Load multi-coil k-space, trajectory coordinates, and coil sensitivity maps from `raw_data.npz`.
2. Cast to float64/complex128 for numerical precision during reconstruction.
3. Data is synthetic (Shepp-Logan phantom), no normalization needed.

## Expected Results

| Method | PSNR (dB) | NCC | NRMSE |
|--------|-----------|-----|-------|
| Gridding | ~2.0 | ~0.60 | ~0.80 |
| L1-Wavelet CS | ~28.6 | ~0.989 | ~0.037 |

The large gap between gridding and L1-wavelet demonstrates the power of sparsity-promoting regularization for non-Cartesian MRI reconstruction.
