# Approach: Dual-Energy CT Material Decomposition

## Overview

The reconstruction pipeline has two stages: (1) sinogram-domain material decomposition using Gauss-Newton optimisation with Poisson likelihood, and (2) filtered back-projection (FBP) to reconstruct spatial density maps from the decomposed material sinograms.

## Stage 1: Gauss-Newton Material Decomposition

### Objective function

At each sinogram bin $(j)$, we minimise the Poisson negative log-likelihood:

$$\mathcal{L}(a) = \sum_{m=1}^{2} \left[\nu_m(a) - g_m \ln \nu_m(a)\right]$$

where the predicted count for measurement $m$ is:

$$\nu_m(a) = \sum_E I_{0,m}(E) \cdot \exp\!\left(-\sum_{k=1}^{2} a_k \cdot \mu_k(E)\right) \cdot \Delta E$$

### Gradient

$$\frac{\partial \mathcal{L}}{\partial a_k} = -\sum_m \left(\frac{g_m}{\nu_m} - 1\right) \frac{\partial \nu_m}{\partial a_k}$$

where $\frac{\partial \nu_m}{\partial a_k} = -\sum_E I_{0,m}(E) \cdot \mu_k(E) \cdot \exp(-\sum_l a_l \mu_l(E)) \cdot \Delta E$.

### Hessian

$$H_{kl} = -\sum_m \left[\left(\frac{g_m}{\nu_m} - 1\right) \frac{\partial^2 \nu_m}{\partial a_k \partial a_l} - \frac{g_m}{\nu_m^2} \frac{\partial \nu_m}{\partial a_k} \frac{\partial \nu_m}{\partial a_l}\right]$$

where $\frac{\partial^2 \nu_m}{\partial a_k \partial a_l} = \sum_E I_{0,m}(E) \cdot \mu_k(E) \cdot \mu_l(E) \cdot \exp(-\sum_l a_l \mu_l(E)) \cdot \Delta E$.

### Newton update

$$a \leftarrow a - H^{-1} \nabla \mathcal{L}$$

The 2x2 Hessian is inverted analytically for speed.

### Hyperparameters

- Number of iterations: 20 (convergence is rapid for this smooth problem)
- Initial value: eps = 1e-6 (small positive to avoid log(0))
- Non-negativity clamp after each update

### Implementation strategy

The outer loop iterates over projection views (angles). Within each view, all detector bins are processed simultaneously in a vectorised manner. The inner loop performs Newton iterations. This follows the reference implementation's `optimize_sino_cpu()` structure.

Precomputed arrays:
- `ssff[m, k, :, E] = I_0,m(E) * mu_k(E) * dE` — for gradient computation
- `ssff2[m, k, l, :, E] = I_0,m(E) * mu_k(E) * mu_l(E) * dE` — for Hessian computation

These avoid redundant multiplications in the inner loop.

## Stage 2: FBP Reconstruction

After obtaining material sinograms $a_k(j)$ in g/cm^2 units, convert to pixel-unit sinograms by dividing by pixel_size (0.1 cm), then apply `iradon()` with a ramp filter to reconstruct density maps (g/cm^3).

Final post-processing: clip negative values to zero (physical density constraint).

## Quality metrics

- NCC and NRMSE computed within a body mask (where ground-truth tissue + bone density > 0.01 g/cm^3)
- Evaluated separately for tissue and bone maps, then averaged
