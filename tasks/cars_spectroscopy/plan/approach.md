# Approach

## Overview

The task is to recover gas temperature from a synthetic N2 CARS spectrum using model-based inversion. The pipeline has four stages: data loading, preprocessing, forward-model-based optimization, and evaluation.

## Algorithm

### Forward Model (G-Matrix CARS)

1. For each vibrational level $v \in \{0, 1\}$ and branch $b \in \{Q, O, S\}$:
   - Compute Raman line positions $\nu_j$ from Dunham spectroscopic constants.
   - Build the relaxation matrix $\Gamma$ using modified exponential gap (MEG) law parameters for N2–N2 collisions. Off-diagonal elements model rotational energy transfer; diagonal elements enforce detailed balance. At the operating pressure (~10 bar), individual rotational lines overlap and these off-diagonal couplings produce collisional narrowing of the Q-branch envelope — an effect the isolated-line approximation cannot reproduce.
   - Form the G-matrix: $K = \text{diag}(\nu_j) + i\Gamma$.
   - Eigendecompose $K$ to compute the resonant susceptibility $\chi_\text{RS}(\nu)$ as a sum of complex Lorentzian-like terms weighted by population differences and Raman cross-sections.
2. Add non-resonant susceptibility $\chi_\text{NR}$ scaled by mole fraction, pressure, and temperature.
3. Compute intensity $I \propto |\chi_\text{total}|^2$.
4. Convolve with pump laser linewidth (Gaussian) using the Kataoka approximation.
5. Convolve with instrument slit function (asymmetric Gaussian).
6. Downsample to experimental wavenumber grid.

### Inversion

- Use `scipy.optimize.least_squares` with trust-region reflective (TRF) method to minimize $\sum_i (I_\text{meas}(\nu_i) - I_\text{model}(\nu_i; T, x, \ldots))^2$.
- Free parameters: temperature, mole fraction, spectral shift, slit width.
- Spectral stretch is fixed at 1.0.
- Maximum 100 function evaluations (sufficient for convergence from a reasonable initial guess).

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pressure | 10 bar | Elevated pressure where Q-branch lines overlap and collisional narrowing is significant; G-matrix is required |
| Initial temperature | 2000 K | Deliberately offset from true 2400 K to test convergence |
| Temperature bounds | [300, 3500] K | Physical range for combustion |
| x_mol bounds | [0.1, 1.0] | Physical mole fraction range |
| nu_shift bounds | [-5, 5] cm$^{-1}$ | Typical calibration uncertainty |
| max_nfev | 100 | Sufficient for LM convergence |

## Evaluation

- **NCC** (cosine similarity) and **NRMSE** between predicted and measured spectra.
- **Temperature error** in Kelvin vs ground truth.
- Visual comparison plot of measured, ground truth, and fitted spectra.
