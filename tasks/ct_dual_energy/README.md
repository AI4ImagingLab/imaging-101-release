# Dual-Energy CT Material Decomposition

> Decompose dual-energy polychromatic CT sinograms into basis material density maps (soft tissue and cortical bone) by solving a Poisson-likelihood inverse problem, which is ill-posed due to spectral overlap and photon counting noise.

> Domain: medicine | Keywords: dual-energy CT, material decomposition, spectral imaging | Difficulty: Medium

## Background

Dual-energy computed tomography (DECT) acquires two sets of X-ray projections at different tube voltages (e.g., 80 kVp and 140 kVp). Because materials attenuate X-rays differently at different energies — governed by energy-dependent mass attenuation coefficients (MACs) — two measurements at distinct spectra provide enough information to decompose the attenuation into contributions from two basis materials.

Material decomposition enables quantitative imaging: separating bone from soft tissue, identifying contrast agents, or mapping electron density — all impossible with single-energy CT. The clinical value lies in improved tissue characterisation and artifact reduction.

## Problem Description

### Forward model

The measured photon count at detector bin $(j)$ for spectrum $m \in \{1, 2\}$ follows the polychromatic Beer-Lambert law:

$$g_m(j) = \sum_E I_{0,m}(E) \cdot \exp\!\Bigl(-\sum_{k=1}^{K} a_k(j) \cdot \mu_k(E)\Bigr) \cdot \Delta E$$

where:
- $I_{0,m}(E)$ is the incident photon fluence of spectrum $m$ at energy $E$
- $a_k(j)$ is the density line integral (g/cm$^2$) of material $k$ along ray $j$
- $\mu_k(E)$ is the mass attenuation coefficient (cm$^2$/g) of material $k$ at energy $E$
- $K = 2$ basis materials (soft tissue and cortical bone)

The photon counts are Poisson-distributed: $g_m(j) \sim \text{Poisson}(\nu_m(j))$, where $\nu_m(j)$ is the expected count from the forward model.

### Inverse problem

Given noisy sinograms $\{g_1, g_2\}$, estimate the material density line integrals $\{a_1, a_2\}$ at every sinogram bin. The problem is ill-posed because:

1. **Spectral overlap**: the two X-ray spectra are broad and overlapping, making the system nearly rank-deficient at some detector bins.
2. **Nonlinearity**: the exponential attenuation model is nonlinear in the material densities.
3. **Poisson noise**: low-count measurements (through dense anatomy) have high relative noise.

After estimating material sinograms, filtered back-projection (FBP) reconstructs spatial density maps.

### Input / Output

- **Input**: Two noisy sinograms (low-energy and high-energy photon counts), X-ray spectra, and mass attenuation coefficient tables.
- **Output**: Two material density maps — tissue density (g/cm$^3$) and bone density (g/cm$^3$) — on a 128x128 pixel grid.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sinogram_low` | `(1, 128, 180)` | float64 | Low-energy sinogram, photon counts |
| `sinogram_high` | `(1, 128, 180)` | float64 | High-energy sinogram, photon counts |
| `spectra` | `(1, 2, 131)` | float64 | Incident photon spectra [low, high], photons per keV bin |
| `mus` | `(1, 2, 131)` | float64 | Mass attenuation coefficients [tissue, bone], cm^2/g |
| `energies` | `(1, 131)` | float64 | Energy grid, keV (20-150 keV at 1 keV spacing) |
| `theta` | `(1, 180)` | float64 | Projection angles, degrees (0 to 180) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `tissue_map` | `(1, 128, 128)` | float64 | Ground truth tissue density map, g/cm^3 |
| `bone_map` | `(1, 128, 128)` | float64 | Ground truth bone density map, g/cm^3 |
| `tissue_sinogram` | `(1, 128, 180)` | float64 | True tissue density line integrals, g/cm^2 |
| `bone_sinogram` | `(1, 128, 180)` | float64 | True bone density line integrals, g/cm^2 |

### data/meta_data.json

Imaging parameters: image size (128), number of angles (180), energy range (20-150 keV), pixel size (0.1 cm), material names, and spectrum peak energies.

## Method Hints

The standard approach is sinogram-domain material decomposition: at each detector bin independently, estimate the two material line integrals that best explain the measured low- and high-energy photon counts under the polychromatic Beer-Lambert model. This is a small nonlinear inverse problem (2 unknowns per bin) that can be solved via Newton-type optimization of the Poisson negative log-likelihood. After decomposition, filtered back-projection reconstructs spatial density maps from the material sinograms. The mass attenuation coefficients and X-ray spectra needed for the forward model are provided as calibration data in `raw_data.npz`.

## References

- Alvarez, R. E., & Macovski, A. (1976). Energy-selective reconstructions in X-ray computerized tomography. *Physics in Medicine and Biology*, 21(5), 733.
- Jadick, G. (2023). dex-ct-sim: Dual-energy X-ray CT simulation. https://github.com/gjadick/dex-ct-sim
- Hubbell, J. H., & Seltzer, S. M. (1995). Tables of X-ray mass attenuation coefficients. NIST Standard Reference Database 126.
