# Approach: Multi-Coil MRI TV Reconstruction

## Algorithm Overview

We use Total Variation (TV) regularized reconstruction for compressed sensing MRI. The algorithm solves:

$$\hat{x} = \arg\min_x \; \frac{1}{2} \|A x - y\|_2^2 + \lambda \cdot \text{TV}(x)$$

where $A$ is the multi-coil forward operator (sensitivity encoding + FFT + undersampling mask) and $\text{TV}(x)$ is the isotropic total variation.

## Solver: PDHG (Chambolle-Pock)

The TV-regularized problem is solved using the Primal-Dual Hybrid Gradient (PDHG) algorithm, also known as the Chambolle-Pock algorithm. The implementation is ported from SigPy's `TotalVariationRecon`.

The key idea is to form a stacked operator $\tilde{A} = [A_{\text{sense}}; G]$ where $G$ is the finite difference (gradient) operator, and solve the saddle-point problem with alternating proximal updates:

1. **Power iteration** (30 iterations): Estimate the maximum eigenvalue of $\tilde{A}^H \tilde{A}$ for step-size selection: $\sigma = 1$, $\tau = 1/\lambda_{\max}$.
2. **PDHG iterations** (100 iterations by default):
   - **Dual update (data fidelity)**: $u_{\text{sense}} \leftarrow \text{prox}_{f_1^*}(\sigma,\; u_{\text{sense}} + \sigma \cdot A_{\text{sense}}(x_{\text{ext}}))$ where $\text{prox}_{f_1^*}$ is $(u + \sigma y)/(1+\sigma)$
   - **Dual update (TV)**: $u_{\text{grad}} \leftarrow \text{prox}_{(\lambda\|\cdot\|_1)^*}(\sigma,\; u_{\text{grad}} + \sigma \cdot G(x_{\text{ext}}))$ via Moreau decomposition with soft thresholding
   - **Primal update**: $x \leftarrow x - \tau \cdot \tilde{A}^H [u_{\text{sense}}; u_{\text{grad}}]$
   - **Extrapolation**: $x_{\text{ext}} \leftarrow 2x - x_{\text{old}}$

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `lambda` | 1e-4 | TV regularization weight, selected via grid search over [1e-6, 1e-2] |
| `max_iter` (power iteration) | 30 | For step-size estimation |
| `max_iter` (primal-dual) | 100 | Main reconstruction iterations |

## Preprocessing

1. Load multi-coil k-space, sensitivity maps, and undersampling mask from `raw_data.npz`.
2. The k-space and ground truth are pre-normalized by the 99th percentile of the per-sample MVUE magnitude.
3. The undersampling mask is a 1-D binary vector (320 entries), applied along the phase-encode (last) dimension.

## Expected Results

| Metric | Average over 6 samples |
|--------|----------------------|
| PSNR | ~23.8 dB |
| NCC | ~0.958 |
| NRMSE | ~0.066 |

These metrics compare the magnitude of the TV reconstruction against the magnitude of the fully-sampled MVUE ground truth.
