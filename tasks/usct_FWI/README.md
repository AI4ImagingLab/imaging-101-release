# Ultrasound Computed Tomography (USCT) via Full-Waveform Inversion

> Reconstruct the spatial distribution of sound speed in breast tissue from multi-frequency ultrasound transmission data using frequency-domain Full-Waveform Inversion with a Convergent Born Series (CBS) Helmholtz solver.

> Domain: Medicine | Keywords: wave imaging, full-waveform inversion | Difficulty: Hard

## Background

Ultrasound Computed Tomography (USCT) is a non-invasive medical imaging technique that reconstructs acoustic property maps (primarily sound speed) of biological tissue. A ring array of ultrasonic transducers surrounds the tissue; each transducer sequentially transmits a pulse while all others record the resulting wavefield. By solving the inverse problem that maps these measurements back to the tissue's sound speed distribution, USCT can produce quantitative images useful for breast cancer detection.

The forward physics is governed by the Helmholtz equation in the frequency domain:

$$(\nabla^2 + \omega^2 s^2(x)) u_k(x) = -\rho_k(x)$$

where $u_k$ is the complex acoustic pressure field for source $k$, $s(x) = 1/c(x)$ is the slowness (inverse sound speed), $\omega = 2\pi f$ is the angular frequency, and $\rho_k$ is the source term.

## Problem Description

Given frequency-domain observation data $y_k \in \mathbb{C}^K$ recorded at $K = 256$ transducer positions for each of $K$ sources, the goal is to reconstruct the slowness field $s(x)$ on a 480x480 grid by minimizing the data misfit:

$$\min_s \sum_{k=1}^{K} \| \alpha_k \hat{u}_k(x_r) - y_k \|_2^2$$

where $\hat{u}_k$ is the forward-modeled wavefield from a unit source and $\alpha_k$ is the estimated source intensity obtained by least-squares fitting:

$$\alpha_k = \frac{\sum_i \overline{\hat{u}_k(x_r^{(i)})} y_k^{(i)}}{\sum_i |\hat{u}_k(x_r^{(i)})|^2}$$

The gradient is computed via the adjoint-state method and the optimization uses Nonlinear Conjugate Gradient (NCG) with More-Thuente line search. A multi-frequency strategy (low-to-high frequency bootstrapping) avoids local minima.

The Helmholtz equation is solved using the Convergent Born Series (CBS) method, which guarantees convergence for arbitrarily large scattering contrasts by introducing a preconditioned iterative scheme with "wiggle" phase ramps and FFT-based propagation.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `receiver_ix` | (1, 256) | float32 | Receiver x-positions (1-indexed grid coordinates) |
| `receiver_iy` | (1, 256) | float32 | Receiver y-positions (1-indexed grid coordinates) |
| `dobs_{freq}` | (1, 256, 256) | complex64 | Observed wavefield data at frequency `freq` MHz; element (0,i,j) is the field at receiver i when source j transmits |

There are 20 frequency keys: `dobs_0.3`, `dobs_0.35`, ..., `dobs_1.25`.

### data/baseline_reference.npz

No `ground_truth.npz` is provided for this task; the observation data was generated from a numerical breast phantom whose sound speed map is not distributed separately. Instead, a `baseline_reference.npz` stores the output of a complete 20-frequency CBS-FWI run as the reference reconstruction. The `ncc_vs_ref` / `nrmse_vs_ref` in `evaluation/metrics.json` are computed against this file.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `vp_reconstructed` | (1, 480, 480) | float32 | Reference sound speed reconstruction from a complete 20-frequency CBS-FWI run (m/s) |

### data/meta_data.json

Physical and discretization parameters: grid size (480x480), grid spacing (50 um), 256 active transducers in circular geometry, 20 frequencies from 0.3 to 1.25 MHz, background velocity 1480 m/s, CBS solver parameters (PPW=8, boundary widths=20).

## Method Hints

- Use the Convergent Born Series (CBS) to solve the Helmholtz equation for each source. CBS uses FFT-based propagation with "wiggle" phase ramps and guaranteed convergence via a preconditioner $\gamma = iV/\epsilon$.
- Multi-frequency bootstrapping: start from low frequencies (large gradient smoothing $\sigma=5$) and progressively increase frequency (reducing $\sigma$ to 1).
- NCG (Polak-Ribiere) optimizer with More-Thuente line search, 3 iterations per frequency, bounds-constrained to [1300, 1700] m/s.
- Near-source muting: mask receiver-source pairs within 7500 um to suppress near-field artifacts.
- Source intensity estimation via least-squares fitting of simulated to observed data.

## References

1. Osnabrugge, G., Leedumrongwatthanakun, S., & Vellekoop, I. M. (2016). A convergent Born series for solving the inhomogeneous Helmholtz equation in arbitrarily large media. *Journal of Computational Physics*, 322, 113-124.
2. Wiskin, J. W., et al. (2013). Full wave, non-linear, inverse scattering. *Acoustical Imaging*, 31, 183-193.
