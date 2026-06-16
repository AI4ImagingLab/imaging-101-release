# Approach: Multi-Coil MRI L1-Wavelet Reconstruction

## Algorithm Overview

We use L1-Wavelet regularized reconstruction for compressed sensing MRI. The algorithm solves:

$$\hat{x} = \arg\min_x \; \frac{1}{2} \|A x - y\|_2^2 + \lambda \|\Psi x\|_1$$

where $A$ is the multi-coil forward operator (sensitivity encoding + FFT + undersampling mask) and $\Psi$ is the Daubechies-4 (db4) discrete wavelet transform.

## Solver: FISTA (Fast Iterative Shrinkage-Thresholding)

The L1-Wavelet problem is solved using FISTA, a proximal gradient method with Nesterov momentum. The implementation is ported from SigPy's `L1WaveletRecon`.

1. **Power iteration** (30 iterations): Estimate the maximum eigenvalue of $A^H A$ for step-size selection: $\alpha = 1/\lambda_{\max}$.
2. **FISTA iterations** (100 iterations by default): Each iteration performs:
   - Gradient step on data fidelity: $z \leftarrow z - \alpha A^H(Az - y)$
   - Proximal step (soft-thresholding in wavelet domain): $x \leftarrow \Psi^{-1} \text{soft}(\Psi z, \alpha\lambda)$
   - Momentum update: $t_{\text{new}} = (1 + \sqrt{1 + 4t^2})/2$, $z \leftarrow x + (t-1)/t_{\text{new}} \cdot (x - x_{\text{old}})$

The soft-thresholding operator $\text{soft}(w, t) = \text{sign}(w) \max(|w| - t, 0)$ applied to wavelet coefficients $w$ promotes sparsity.

The wavelet transform (Daubechies-4, multi-level DWT/IDWT) is implemented from scratch using convolution-based 1D and 2D decomposition/reconstruction, without external wavelet libraries.

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda` (L1-Wavelet) | 1e-3 | Selected via grid search over [1e-7, 1e-2] to maximize NCC |
| `wave_name` | db4 | Daubechies-4 wavelet; good balance of smoothness and compactness |
| `lambda` (TV, comparison) | 5e-3 | Best TV regularization weight for this dataset |
| `max_iter` (power iteration) | 30 | For step-size estimation |
| `max_iter` (proximal gradient) | 100 | Main reconstruction iterations |

## Data Generation

1. Generate a 128x128 Shepp-Logan phantom (normalized to [0, 1]).
2. Generate 8 Gaussian coil sensitivity maps placed evenly around the image, RSS-normalized.
3. Compute fully-sampled multi-coil k-space via centered 2D FFT.
4. Apply 8x random undersampling mask (16/128 lines, including 8% ACS center).

## Why L1-Wavelet Over TV

- **Multi-scale sparsity**: wavelets capture edges at all scales, while TV only captures first-order gradient sparsity.
- **Texture preservation**: wavelet coefficients at fine scales encode texture information that TV smooths away.
- **No staircase artifacts**: TV's piecewise-constant bias creates blocky artifacts in smooth gradient regions.

## Expected Results

| Method | NCC | NRMSE | PSNR (dB) |
|--------|-----|-------|-----------|
| Zero-Filled | 0.831 | 0.130 | 17.7 |
| TV | 0.844 | 0.125 | 18.1 |
| L1-Wavelet | 0.872 | 0.114 | 18.8 |

L1-Wavelet consistently outperforms TV on this synthetic phantom dataset.
