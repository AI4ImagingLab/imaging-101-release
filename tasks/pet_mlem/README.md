# PET MLEM Reconstruction

> Reconstruct a 2D positron emission tomography (PET) activity distribution from Poisson-noisy sinogram data using the MLEM algorithm, a fundamental maximum-likelihood approach for photon-counting emission tomography distinct from Gaussian-noise reconstruction methods.

> Domain: medicine | Keywords: emission tomography, Poisson reconstruction, expectation maximization | Difficulty: Medium

## Background

Positron Emission Tomography (PET) images the distribution of a radioactive tracer injected into the body. When a positron annihilates with an electron, two 511 keV gamma photons are emitted in opposite directions and detected in coincidence by a ring of detectors. Each coincidence event defines a line of response (LOR) through the body, and the collected LOR counts form a sinogram.

Unlike CT where transmitted X-rays follow Gaussian noise statistics, PET data follows **Poisson statistics** because each detector bin counts individual photon pairs. This fundamentally changes the reconstruction algorithm: instead of minimizing a least-squares objective (L2 norm), PET reconstruction maximizes the Poisson log-likelihood, leading to the MLEM (Maximum Likelihood Expectation Maximization) algorithm with multiplicative — not additive — updates.

## Problem Description

The PET measurement model is:

$$y_i \sim \text{Poisson}\!\left(\sum_j A_{ij} x_j + r_i\right)$$

where:
- $y_i$ is the number of detected coincidences in the $i$-th LOR (sinogram bin)
- $x_j \geq 0$ is the radiotracer activity concentration in the $j$-th image pixel
- $A_{ij}$ is the system matrix element (probability that an emission from pixel $j$ is detected in LOR $i$)
- $r_i$ is the expected background count (random coincidences + scatter)

**Inverse problem:** Given the Poisson-noisy sinogram $y$ and background estimate $r$, reconstruct the activity image $x \geq 0$.

**Ill-conditioning:** PET reconstruction is ill-posed due to:
- Limited angular sampling and detector resolution
- High noise (low photon counts, especially in whole-body PET)
- The maximum-likelihood estimate becomes increasingly noisy with more iterations (semiconvergence), requiring early stopping or regularization

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sinogram` | `(1, 128, 120)` | float32 | Poisson-noisy PET sinogram (photon counts, scaled by count_level) |
| `background` | `(1, 128, 120)` | float32 | Estimated background (randoms) sinogram (photon counts) |
| `theta` | `(1, 120)` | float32 | Projection angles (degrees, 0 to 180) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `activity_map` | `(1, 128, 128)` | float32 | True radiotracer activity distribution (a.u.) |

### data/meta_data.json

| Key | Type | Description |
|-----|------|-------------|
| `image_size` | int | Image dimension (128) |
| `n_angles` | int | Number of projection angles (120) |
| `n_radial_bins` | int | Number of radial sinogram bins (128) |
| `count_level` | float | Poisson count scaling factor (1000) |
| `randoms_fraction` | float | Background-to-signal ratio (0.1) |
| `noise_model` | str | "poisson" |
| `modality` | str | "2D PET" |

The synthetic phantom simulates a brain-like activity distribution with two hot lesions (simulating tumors with 4-6x background uptake). Poisson noise and uniform random coincidence background are added.

## Method Hints

**MLEM** (Maximum Likelihood Expectation Maximization) is the standard iterative algorithm for Poisson emission tomography. It maximizes the Poisson log-likelihood via multiplicative updates that naturally enforce non-negativity. A precomputed sensitivity image (back-projection of uniform data) normalizes each update.

**OSEM** (Ordered Subsets EM) accelerates MLEM by processing subsets of projection angles separately, achieving faster convergence in early iterations at the cost of potential non-convergence.

The 2D PET system matrix can be implemented as the Radon transform (line integrals), with unfiltered back-projection as the adjoint.

## References

- Shepp, L.A. and Vardi, Y. (1982). Maximum Likelihood Reconstruction for Emission Tomography. IEEE Transactions on Medical Imaging, 1(2), 113-122.
- Hudson, H.M. and Larkin, R.S. (1994). Accelerated image reconstruction using ordered subsets of projection data. IEEE Transactions on Medical Imaging, 13(4), 601-609.
- Lange, K. and Carson, R. (1984). EM reconstruction algorithms for emission and transmission tomography. Journal of Computer Assisted Tomography, 8(2), 306-316.
