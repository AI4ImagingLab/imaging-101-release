# Approach: Dynamic Black Hole Imaging with StarWarps

## Problem

Reconstruct a time-varying video of a black hole from sparse interferometric
(VLBI) measurements at each time step. Each frame has its own set of
(u,v) baseline measurements with thermal noise. The key challenge is that
individual frames are severely under-determined, so temporal coupling
between frames is essential for quality reconstruction.

## Algorithm: StarWarps (Bouman et al. 2017)

StarWarps models the video as a Gaussian Markov process:

    x_t = A(θ) x_{t-1} + w_t,    w_t ~ N(0, Q)
    y_t = F_t x_t + n_t,          n_t ~ N(0, Σ_t)

where:
- x_t is the image at time t (vectorized, N² elements)
- A(θ) is a parametric warp matrix (Fourier-domain phase shifts)
- Q (Upsilon) is the process noise covariance
- y_t are the interferometric measurements
- F_t is the DFT measurement matrix for frame t

### EM Algorithm

**E-step**: Forward-backward message passing (Kalman filter/RTS smoother)
computes sufficient statistics E[x_t], E[x_t x_t^T], E[x_{t-1} x_t^T]
given current θ.

**M-step**: Optimize θ (warp parameters) via L-BFGS-B to minimize the
expected negative log-likelihood.

### Key Components

1. **Forward pass** (Kalman filter): Processes frames left-to-right,
   combining prior with data likelihood using Gaussian product lemmas.

2. **Backward pass / RTS smoother**: Propagates information backwards
   to refine estimates using future observations.

3. **Gaussian prior**: Power-law Fourier spectrum covariance, scaled
   by image intensity (from gaussImgCovariance_2).

4. **Phase warping**: Inter-frame motion modeled as phase shifts in
   the Fourier domain, parameterized by affine motion (6 DOF).

## Baseline: Static Per-Frame

Each frame is reconstructed independently using Gaussian MAP:
x_hat = argmax N(x; μ, Λ) * N(y; Fx, Σ)

This ignores temporal structure and serves as the baseline comparison.

## Metrics

- **NRMSE** (Normalized Root Mean Square Error): pixel-wise error
  normalized by ground truth range.
- **NCC** (Normalized Cross-Correlation): structural similarity
  measure, 1.0 = perfect match.

## Expected Result

StarWarps should achieve lower average NRMSE and higher average NCC
than the static per-frame baseline, demonstrating the benefit of
temporal coupling.

## References

- Bouman et al. 2017, "Reconstructing Video from Interferometric
  Measurements of Time-Varying Sources", arXiv:1711.01357
- EHT Collaboration Paper IV (2019), ApJL 875, L4
