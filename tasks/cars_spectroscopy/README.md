# CARS Spectroscopy Temperature Inversion

> Recover gas temperature from a synthetic N2 coherent anti-Stokes Raman scattering (CARS) spectrum 

> Domain: chemistry | Keywords: Raman spectroscopy, nonlinear least-squares inversion | Difficulty: Medium

## Background

Coherent anti-Stokes Raman scattering (CARS) is a nonlinear optical technique used to measure gas-phase temperature and species concentration in combustion environments. The CARS signal depends on the complex third-order susceptibility $\chi^{(3)}$, which encodes molecular rotational-vibrational population distributions that are functions of temperature.

The forward model computes a theoretical CARS intensity spectrum $I(\nu)$ for a given temperature $T$, pressure $P$, and mole fraction $x$. It accounts for:

- **Raman line positions and strengths** from molecular spectroscopic constants (Dunham expansion for N2).
- **Collisional narrowing** at elevated pressure (~10 bar), where rotational energy transfer between overlapping lines reshapes the Q-branch and the isolated-line approximation breaks down.
- **Non-resonant background** $\chi_\text{NR}$ that interferes coherently with the resonant susceptibility.
- **Pump laser linewidth** convolved with the susceptibility.
- **Instrument (slit) function** modeled as an asymmetric Gaussian.

## Problem Description

The measurement equation is:

$$I(\nu) = \left| x \cdot N \cdot \chi_\text{RS}(\nu; T, P) + \chi_\text{NR}(x, P, T) \right|^2 \ast S_\text{pump} \ast S_\text{slit}$$

where $\chi_\text{RS}$ is the resonant susceptibility computed via eigendecomposition of the G-matrix $K = \text{diag}(\nu_\text{Raman}) + i\Gamma$, $N$ is the number density, $S_\text{pump}$ is the pump laser profile, and $S_\text{slit}$ is the instrument function.

The inverse problem is: given a measured spectrum $I_\text{meas}(\nu)$, find the temperature $T$ (and optionally mole fraction, spectral shift, and slit width) that minimizes $\|I_\text{meas} - I_\text{model}(T, \ldots)\|^2$.

The problem is nonlinear because $\chi_\text{RS}$ depends on $T$ through both the Boltzmann population factors and the relaxation matrix elements.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `measurements` | `(1, 200)` | float64 | Noisy normalized CARS intensity spectrum (dimensionless) |
| `nu_axis` | `(1, 200)` | float64 | Wavenumber axis ($cm^{-1}$) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `spectrum` | `(1, 200)` | float64 | Clean normalized CARS intensity spectrum (dimensionless) |
| `temperature` | `(1,)` | float64 | True gas temperature (K) |
| `x_mol` | `(1,)` | float64 | True N2 mole fraction (dimensionless) |

### data/meta_data.json

Contains imaging/physical parameters: species, wavenumber range, pressure, pump linewidth, slit parameters, noise level, and random seed.

## Method Hints

- Use a **nonlinear least-squares** optimizer (e.g., Levenberg-Marquardt via `scipy.optimize.least_squares`) to minimize the residual between measured and modeled spectra.
- The forward model should use the **G-matrix (relaxation matrix) formulation** for the resonant susceptibility, solved via eigendecomposition. This captures collisional narrowing effects important at atmospheric and higher pressures.
- Initial temperature guess should be within a few hundred K of the true value for reliable convergence. The Q-branch, O-branch, and S-branch transitions should all be included.
- The non-resonant background creates an interference pattern that shifts the apparent peak positions — the forward model must include it.

## References

- Kataoka, H., Maeda, S., & Hirose, C. (1982). Effects of laser linewidth on the coherent anti-Stokes Raman spectroscopy spectral profile. Applied Spectroscopy, 36(5), 565-569.
- Palmer, R. E. (1989). The CARSFT computer code for calculating coherent anti-Stokes Raman spectra. Sandia National Laboratories Report SAND89-8206.
- Martinsson, L., Bengtsson, P. E., Aldén, M., Kröll, S., & Bonamy, J. (1993). A test of the G-matrix approach to N2 CARS modelling. Journal of Chemical Physics, 99(4), 2466-2477.
