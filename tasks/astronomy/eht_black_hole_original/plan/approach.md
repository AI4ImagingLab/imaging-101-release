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

The bundled seven-station observation contains 421 visibility samples, 269 closure-phase measurements, and 233 log closure-amplitude measurements aggregated across scans. Use the supplied per-leg UV coordinates rather than treating every sample as one simultaneous station graph.

## Solution Strategy

### Step 1: Data Preprocessing

Load the public observation data (`data/raw_data.npz` containing gain-corrupted visibilities, uv-coordinates, noise levels, station IDs, and per-scan closure arrays). Stored arrays have a leading batch axis `(1, ...)`; `load_observation` removes this axis for computation. Acquisition metadata is documented in README and returned as matching constants by `load_metadata`; no standalone metadata JSON is required. Preserve the original pixel pitch `9.696273622188272e-12` radians exactly.

Calibrated observations for the original controlled comparison are hidden maintainer assets in `notebooks/reference_outputs/reference_results.npz`; they are loaded separately by the notebook and are not public solver inputs.

The preprocessing helpers enumerate valid triangles and quadrangles from station connectivity and compute closure quantities and their noise uncertainties. For the bundled multi-scan data, the reference pipeline uses the supplied closure arrays and UV legs. The error-propagation formulas are:
- sigma_CP = sqrt(sum sigma_i^2 / |V_i|^2)
- sigma_logCA = sqrt(sum sigma_i^2 / |V_i|^2)

### Step 2: Forward Model Construction

Build the DFT measurement matrix A using ehtim's conventions:
- Sign: +2*pi*i (ehtim convention since Jan 2017)
- Pixel grid: xlist = arange(0, -N, -1) * psize + (psize*N)/2 - psize/2
- Pulse: triangle pulse in Fourier domain

For closure quantity chi-squared evaluation, build separate DFT matrices for each triangle leg (uv1, uv2, uv3) and quadrangle leg (uv1, uv2, uv3, uv4). This avoids re-indexing during optimization.

### Step 3: Image Reconstruction

The original study compares three RML formulations of increasing gain-robustness. The maintained notebook actively runs visibility and closure-only imaging on calibrated and corrupted observations; amplitude + closure phase remains an archived comparison result:

1. **Visibility RML** (comparison baseline):
   - The notebook uses `ClosureRMLSolver(data_terms={'vis': 100}, ...)`, with chi2_vis = (1/(2*M)) * sum(|A*x - y|^2 / sigma^2) and objective alpha_vis * (chi2_vis - 1) - alpha_gs * S_gs - alpha_simple * S_simple.
   - The separate `VisibilityRMLSolver` helper instead uses 0.5 * sum(|A*x - y|^2 / sigma^2) + sum(weight * regularizer_value), without division by M. It is not the class used for the notebook's visibility bars.
   - Fails on gain-corrupted data because the objective fits gain-corrupted amplitudes and phases.

2. **Amplitude + Closure Phase RML**:
   - Data terms: visibility amplitudes (gain-sensitive) + closure phases (gain-invariant)
   - Partially robust: closure phases immune to gains, but amplitude ratios distorted.

3. **Closure-only RML** (main method, following Chael 2018):
   - Data terms: closure phase chi-squared (Eq. 11) + log closure amplitude chi-squared (Eq. 12)
   - Objective: alpha_CP * (chi2_CP - 1) + alpha_CA * (chi2_CA - 1) - alpha_gs * S_gs - alpha_simple * S_simple
   - Fully gain-invariant: both data terms are immune to station-based gains.

For measured closure phase phi and modeled phase phi_hat in radians,
chi2_CP = (2 / n_CP) * sum((1 - cos(phi - phi_hat)) / sigma_phi^2).
For measured log closure amplitude a and modeled value a_hat,
chi2_CA = (1 / n_CA) * sum(((a - a_hat) / sigma_a)^2).
Here n_CP and n_CA are the respective measurement counts and sigma denotes the supplied uncertainty. These explicit formulas define the cited closure terms independently of the paper equation numbers.

Regularizer helpers (I is the current image and P is the Gaussian prior):
- **Gull-Skilling entropy**: S(I) = sum(I - P - I*log(I/P))
- **Simple entropy**: S(I) = -sum(I*log(I/P))
- **Total Variation helper**: TV(I) = sum(sqrt(dx^2 + dy^2 + eps^2) - eps), where dx and dy are forward differences with periodic wrapping and eps defaults to 1e-6.

The preserved `ClosureRMLSolver` subtracts every selected regularizer's returned value, including positive TV. Thus its optional `tv` branch contributes -weight * TV, not the usual positive TV penalty. The standard reconstruction and notebook use only `gs` and `simple`; they do not enable TV. This describes the existing implementation rather than asserting validation of that optional branch.

Prior image: Gaussian blob at image center, with 20 microarcsecond FWHM and total flux 0.6 Jy. The closure-only configuration uses data weights 50 for each closure term and entropy weights 1 (Gull-Skilling) and 10 (simple entropy). The visibility comparison uses data weight 100 and the same entropy weights.

Preserve the prior's coordinate convention `(arange(N) - N/2) * pixel_size_rad`, as used in `src/main.py` and the notebook. It is distinct from the half-pixel Fourier grid above. The Gaussian width is FWHM / (2*sqrt(2*log(2))) after converting FWHM to radians.

Optimization: L-BFGS-B in log brightness, I = exp(z), for positivity. Initialize z = log(max(x0, 1e-30)) and apply the chain rule grad_z = I * grad_I. Each of three rounds warm-starts from the preceding `result.x`, keeping the same prior and weights. Each round allows at most 300 iterations with `ftol=1e-14` and `gtol=1e-10`; it may stop earlier. The preserved code consumes `result.x` without checking optimizer success. The entry point checks the final image for finite values and positive total flux before scaling and saving it.

### Step 4: Evaluation

The notebook computes fresh visibility and closure-only results in both data states, and displays the archived Amp+CP comparisons alongside the archived reference results. The metrics are:
- **NRMSE (historical notebook output)**: ||x_hat - x||_2 / ||x||_2 (lower is better).
- **NRMSE (current source and Judge)**: sqrt(mean((x_hat - x)^2)) / (max(x) - min(x)). The two normalizations must not be mixed; their numerical relationship is documented in `evaluation/validation.md`.
- **NCC**: dot(x_hat, x) / (norm(x_hat) * norm(x)), with flattened arrays and no mean subtraction (higher is better, max 1).
- **Dynamic Range**: peak(reconstruction) / RMS(reconstruction - truth), as implemented by the visualization module.

Notebook metrics are computed after flux normalization (matching total flux to ground truth). The public-data entry point `python -m src.main` scales its result to the known 0.6 Jy total flux and saves `output/reconstruction.npz`, key `reconstruction`, shape `(1, 64, 64)`, dtype `float64`, in Jy/pixel. The standalone `evaluation/judge.py` scores that declared array directly without evaluation-time rescaling.

### Step 5: Visualization

Generate comparison panels showing:
- Ground truth image
- Reconstructions from all three methods on calibrated vs. corrupted data
- UV coverage plot
- Metrics table demonstrating gain robustness of closure-only imaging

These plots and metric tables are produced by the maintainer notebook. `src/main.py` does not call the visualization module: it reconstructs the public corrupted observation and saves the submission only. The independent Judge writes `output/judge_report.json` when explicitly invoked.

## Expected Results (Historical Upstream Estimates)

The following table is retained from the upstream plan as historical planning estimates using its L2-ratio NRMSE convention. It is not a table of validated archived or current measurements and does not define pass thresholds. Measured reference metrics are stored in `notebooks/reference_outputs/reference_metrics.json`; metric provenance and fresh-run differences are documented in `evaluation/validation.md`.

| Method | Calibrated NRMSE/NCC | Corrupted NRMSE/NCC |
|------------------|---------------------|---------------------|
| Vis RML          | ~0.26 / ~0.97       | ~5.0 / ~0.01        |
| Amp+CP           | ~0.70 / ~0.76       | ~1.7 / ~0.41        |
| Closure-only     | ~0.82 / ~0.75       | ~0.85 / ~0.72       |

The key scientific result: closure-only imaging sacrifices some fidelity on well-calibrated data but provides robust reconstructions under gain corruption, which is the typical regime for real EHT observations. Visibility RML is best when calibration is perfect, but catastrophically fails when gains are present.

## Format Migration Provenance

This plan restores the full upstream structure from local `origin/main` commit `dc2f668939b21e8312e22529615def610f8611df`. Adaptations document batch-first NPZ storage, the public/hidden data split, README acquisition constants, `src/main.py`, and the current metric contract. The remote endpoint was unavailable during verification; no newer remote revision is claimed.
