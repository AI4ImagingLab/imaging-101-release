# Approach

## Problem Statement

Recover a 64x64 radio image of a black hole from 421 gain-corrupted complex visibility measurements collected by 7 EHT stations, using closure phases and log closure amplitudes as gain-invariant observables.

## Mathematical Formulation

### Forward Model

The VLBI forward model follows the van Cittert-Zernike theorem. Each telescope pair measures one complex visibility -- a Fourier coefficient of the sky brightness:

$$\mathbf{y} = \mathbf{A}\,\mathbf{x}$$

- **x** in R^{N^2}: vectorized sky brightness (N=64, so 4096 unknowns)
- **A** in C^{M x N^2}: DFT measurement matrix with triangle pulse, evaluated at (u,v) baseline positions using ehtim sign convention (+2*pi*i)
- **y** in C^M: 421 measured complex visibilities (M << N^2)

### Gain Corruption

Station-based gains corrupt the visibilities:

$$y_{ij}^{\text{corr}} = g_i \, g_j^* \, y_{ij}^{\text{true}}$$

with |g_i| ~ 1 +/- 20% and arg(g_i) ~ U(-30, 30) degrees.

### Closure Quantities

Closure quantities cancel station-based gains:

- **Closure phase** for triangle (i, j, k): phi_C = arg(V_ij * V_jk * V_ki) -- invariant because each station gain cancels.
- **Log closure amplitude** for quadrangle (i, j, k, l): log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl| -- each station appears once in numerator and denominator.

For 7 stations and 421 baselines: ~35 independent closure phases (triangles) and ~35 independent log closure amplitudes (quadrangles).

## Solution Strategy

### Step 1: Data Preprocessing

Load the observation data (`raw_data.npz` containing calibrated and corrupted visibilities, uv-coordinates, noise levels, station IDs) and imaging metadata (`meta_data` JSON with image size, pixel scale, station info).

Enumerate all valid triangles and quadrangles from the station connectivity. Compute closure phases, log closure amplitudes, and their noise uncertainties from error propagation:
- sigma_CP = sqrt(sum sigma_i^2 / |V_i|^2)
- sigma_logCA = sqrt(sum sigma_i^2 / |V_i|^2)

### Step 2: Forward Model Construction

Build the DFT measurement matrix A using ehtim's conventions:
- Sign: +2*pi*i (ehtim convention since Jan 2017)
- Pixel grid: xlist = arange(0, -N, -1) * psize + (psize*N)/2 - psize/2
- Pulse: triangle pulse in Fourier domain

For closure quantity chi-squared evaluation, build separate DFT matrices for each triangle leg (uv1, uv2, uv3) and quadrangle leg (uv1, uv2, uv3, uv4). This avoids re-indexing during optimization.

### Step 3: Image Reconstruction

Implement three RML solvers of increasing gain-robustness:

1. **Visibility RML** (comparison baseline):
   - Objective: (1/M) sum |A*x - y_corrupt|^2 / sigma^2 + regularization
   - Fails on gain-corrupted data because the objective fits gain-corrupted amplitudes and phases.

2. **Amplitude + Closure Phase RML**:
   - Data terms: visibility amplitudes (gain-sensitive) + closure phases (gain-invariant)
   - Partially robust: closure phases immune to gains, but amplitude ratios distorted.

3. **Closure-only RML** (main method, following Chael 2018):
   - Data terms: closure phase chi-squared (Eq. 11) + log closure amplitude chi-squared (Eq. 12)
   - Objective: alpha_CP * (chi2_CP - 1) + alpha_CA * (chi2_CA - 1) + alpha_gs * S_gs + alpha_simple * S_simple
   - Fully gain-invariant: both data terms are immune to station-based gains.

Regularizers (matching ehtim):
- **Gull-Skilling entropy**: S(I) = sum(I - P - I*log(I/P))
- **Simple entropy**: S(I) = -sum(I*log(I/P))
- **Total Variation**: TV(I) = sum sqrt(dx^2 + dy^2 + eps^2)

Prior image: Gaussian blob at image center (total flux matching source).

Optimization: L-BFGS-B with positivity constraints (x >= 1e-30). Multi-round strategy (3 rounds of 300 iterations each), matching ehtim's imaging workflow.

### Step 4: Evaluation

Compare all three methods on both calibrated and gain-corrupted data using:
- **NRMSE**: ||x_hat - x||_2 / ||x||_2 (lower is better)
- **NCC**: normalized cross-correlation (higher is better, max 1)
- **Dynamic Range**: peak / RMS(background) (higher means better sensitivity)

All metrics computed after flux normalization (matching total flux to ground truth).

### Step 5: Visualization

Generate comparison panels showing:
- Ground truth image
- Reconstructions from all three methods on calibrated vs. corrupted data
- UV coverage plot
- Metrics table demonstrating gain robustness of closure-only imaging

## Expected Results

| Method | Calibrated NRMSE/NCC | Corrupted NRMSE/NCC |
|------------------|---------------------|---------------------|
| Vis RML          | ~0.26 / ~0.97       | ~5.0 / ~0.01        |
| Amp+CP           | ~0.70 / ~0.76       | ~1.7 / ~0.41        |
| Closure-only     | ~0.82 / ~0.75       | ~0.85 / ~0.72       |

The key scientific result: closure-only imaging sacrifices some fidelity on well-calibrated data but provides robust reconstructions under gain corruption, which is the typical regime for real EHT observations. Visibility RML is best when calibration is perfect, but catastrophically fails when gains are present.
