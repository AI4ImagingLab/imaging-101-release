# Approach: Shack-Hartmann Wavefront Reconstruction

## Problem

Given raw Shack-Hartmann WFS detector images (128×128 pixel spot mosaic) and a
calibrated response matrix R (N_slopes × N_modes), reconstruct the wavefront phase
map φ(x, y) at four WFE levels (50, 100, 200, 400 nm RMS) and evaluate accuracy
using per-level NCC and NRMSE against ground-truth phase maps.

## Algorithm

### Step 1 — Response matrix calibration (done in `generate_data.py`)

The DM is calibrated using push-pull probing: each mode j is poked by ±amplitude,
WFS slope differences are measured, and the column R[:, j] is computed:

    R[:, j] = (s(+amp_j) − s(−amp_j)) / (2 · amp_j)

Slopes are extracted from the raw detector images using the weighted-centroid
estimator (same algorithm as Step 2 below).

### Step 2 — Centroid estimation (new in `src/physics_model.py`)

For each valid subaperture j (identified by `subap_map`), compute the
intensity-weighted centroid of the measured and reference spot images:

    slope_x[j] = Σ(x_k · I_k) / Σ(I_k)  −  Σ(x_k · I_ref_k) / Σ(I_ref_k)

where k iterates over pixels in subaperture j, and (x_k, y_k) are focal-plane
coordinates from `detector_coords_x/y`.

Output ordering: [slope_x_0, …, slope_x_{N-1}, slope_y_0, …, slope_y_{N-1}]
(all x first, then y), matching HCIPy's ShackHartmannWavefrontSensorEstimator.

### Step 3 — Tikhonov reconstruction matrix

The reconstruction matrix M = R⁺ is the regularised pseudo-inverse of R:

    R = U Σ Vᵀ  (truncated SVD)
    M = V Σ⁺ Uᵀ    where  Σ⁺ = diag(1/σⱼ for σⱼ > rcond · σ_max, else 0)

**Key hyperparameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| rcond     | 1e-3  | Tikhonov regularisation threshold |

### Step 4 — Wavefront reconstruction

    a_hat = M · s           (mode amplitude estimation)
    φ     = 4π/λ × Ψᵀ a_hat  (phase assembly from DM modes)

### Step 5 — Evaluation

Per-level quality metrics computed over aperture-valid pixels only:

**NCC**: cosine similarity, φ̂ · φ_GT / (‖φ̂‖ · ‖φ_GT‖)
**NRMSE**: RMS(φ̂ − φ_GT) / (max(φ_GT) − min(φ_GT))
**Timing**: wall-clock time for centroid extraction + reconstruction, all 4 levels
           (not including one-time SVD setup)

## Expected Results

| WFE level | NCC    | NRMSE  |
|-----------|--------|--------|
| 50 nm     | 0.9960 | 0.0836 |
| 100 nm    | 0.9843 | 0.0706 |
| 200 nm    | 0.9745 | 0.0714 |
| 400 nm    | 0.8642 | 0.0752 |

Reconstruction time (all 4 levels): ~72 ms (numpy, no GPU).
Timing boundary: ≤ 0.1 s (100 ms).

## Alternative Methods

- **Correlation-based centroid**: cross-correlate spot image with reference instead
  of computing absolute centroids; more robust to spot asymmetry
- **Zonal reconstruction**: direct wavefront integration from gradients
- **Bayesian / MAP estimation**: adds Kolmogorov turbulence prior
- **Neural-network reconstructors**: trained end-to-end; faster and more accurate
  at large WFE but require training data
