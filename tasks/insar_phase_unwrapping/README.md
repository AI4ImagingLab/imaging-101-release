# InSAR Phase Unwrapping via Sparse Optimization

> Recover the absolute surface deformation phase from a 2π-wrapped InSAR interferogram, where the true phase can span many multiples of 2π but only its remainder is measured.

> Domain: Earth Science | Keywords: phase unwrapping, interferometry | Difficulty: Medium

## Background

**Phase unwrapping** is the problem of recovering the absolute phase from measurements that are wrapped modulo 2π. In interferometric synthetic aperture radar (InSAR), the measured interferogram phase lies in (−π, π], but the true deformation signal can span many multiples of 2π. Unwrapping is essential for extracting quantitative surface displacement information from satellite radar data.

## Problem Description

An InSAR interferogram is formed by computing the complex conjugate product of two SAR images acquired at different times. The phase of this complex product encodes the path-length difference between the two acquisitions, which relates to surface displacement, topography, and atmospheric delay.

The **wrapped phase** θ is related to the **absolute phase** Φ by:

$$\theta = \Phi \bmod 2\pi \quad (\text{wrapped to } (-\pi, \pi])$$

The challenge is that the wrapping is ambiguous: θ and θ + 2πk are indistinguishable for any integer k. The goal is to recover Φ from θ across the 2D image domain, which is ill-posed in general and requires additional regularity assumptions (such as the phase gradient being piecewise sparse).

## Data Description

### `data/raw_data.npz`

Real Sentinel-1A interferometric SAR data (12-day temporal baseline, range-azimuth coordinates).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `interferogram` | (1, 778, 947) | complex64 | Complex interferogram (real + imaginary parts) |
| `magnitude` | (1, 778, 947) | float32 | Interferogram magnitude |
| `wrapped_phase` | (1, 778, 947) | float32 | Measured wrapped phase (radians, range [−π, π]) |

### `data/baseline_reference.npz`

No `ground_truth.npz` is provided for this task; `data/baseline_reference.npz` contains the SNAPHU reference solution — SNAPHU is the standard operational algorithm for InSAR phase unwrapping.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `unwrapped_phase` | (1, 778, 947) | float32 | SNAPHU unwrapped phase (reference solution) |
| `magnitude` | (1, 778, 947) | float32 | SNAPHU magnitude output (reference) |

### `data/meta_data.json`

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sensor` | "Sentinel-1A" | SAR sensor |
| `dates` | ["2015-03-28", "2015-04-09"] | Acquisition dates |
| `temporal_baseline_days` | 12 | Days between acquisitions |
| `image_dimensions.rows` | 778 | Image height (range direction) |
| `image_dimensions.columns` | 947 | Image width (azimuth direction) |
| `geolocation.x_first` | −103.452 | Longitude of first pixel (°) |
| `geolocation.y_first` | 31.859 | Latitude of first pixel (°) |

## Method Hints

**Algorithm family:** ADMM (Alternating Direction Method of Multipliers) with a nonconvex sparsity-promoting penalty.

**Key idea:** The wrapped gradient of θ approximates the true gradient of Φ at all pixels where no phase discontinuity (2π jump) occurs. The unwrapping problem then reduces to finding Φ whose gradient matches this estimated gradient field at as many pixels as possible — a sparse error model:

$$\min_\Phi \|D\Phi - \phi\|_0$$

where D is the discrete gradient operator and φ is the wrapped-gradient estimate. ADMM splits this into a linear Poisson solve (efficient via the Discrete Cosine Transform, which diagonalises the Laplacian under Neumann boundary conditions) and a pointwise shrinkage step. The G0 penalty (p = 0) promotes sparser residuals than L1, improving accuracy at phase discontinuities.

**Why it suits the problem:** Phase discontinuities (2π jumps) are sparse relative to the full image — they occur only at fringes and low-coherence regions. The sparse prior directly models this physical property. The DCT-based solver makes each ADMM iteration O(N log N), converging in ∼15 iterations.

## References

- Chartrand, Calef, Warren. "Exploiting Sparsity for Phase Unwrapping." IGARSS 2019.
- Chen and Zebker. "Two-dimensional phase unwrapping with use of statistical models for cost functions." JOSA A, 2001 (SNAPHU).
- Chartrand. "Shrinkage mappings and their induced penalty functions." ICASSP 2014.
