# Confocal NLOS Imaging — f-k Migration

> Domain: computational photography / optics | Keywords: non-line-of-sight, wave imaging | Difficulty: Medium

## Background

Non-line-of-sight (NLOS) imaging recovers 3-D scenes hidden around a corner
using indirect light paths.  A pulsed laser illuminates a visible relay wall;
photons scatter to the hidden scene and back, and a single-photon detector
(SPAD) records time-resolved histograms at each wall scan point.

The f-k migration method of Lindell et al. (ACM TOG 2019) models light as a
wave and applies Stolt frequency-wavenumber interpolation — a technique from
seismic imaging — to back-propagate the wavefield to the hidden scene in
O(N³ log N) time without free parameters.

## Problem Description

Given confocal transient measurements τ(x′, y′, t), recover the 3-D albedo
volume ρ(x, y, z).

The forward model (spherical Radon transform / wave equation) is:

    τ(x′, y′, t) = ∭ (1/r⁴) ρ(x, y, z) δ(2r/c − t) dx dy dz

where r = √((x′−x)² + (y′−y)² + z²).

The migration maps the 3-D Fourier transform of the pre-processed transient data directly to the 3-D Fourier transform of the hidden scene via the dispersion relation of the wave equation (Stolt mapping), then recovers the volume by inverse FFT. It is non-iterative, parameter-free, and runs in O(N³ log N) time.

## Data Description

`data/raw_data.npz` contains:

| Key              | Shape          | Dtype   | Description                               |
|------------------|----------------|---------|-------------------------------------------|
| `meas`           | (128, 128, 2048) | float32 | Confocal transient measurements (photon counts). Storage order (Ny, Nx, Nt). |
| `tofgrid`        | (128, 128)     | float32 | TOF calibration delays in picoseconds     |
| `wall_size`      | scalar         | float64 | Side length of scanned wall (metres)      |
| `bin_resolution` | scalar         | float64 | Temporal bin width (seconds)              |

`data/meta_data` (JSON): imaging parameters including `n_time_crop=512`.

No `ground_truth.npz` is provided for this task; `data/baseline_reference.npz` contains the reference f-k reconstruction from the original authors on the same dataset, used to verify correctness (expected NCC ≈ 1.0 against our own implementation).

### `data/baseline_reference.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `reconstruction` | (1, 512, 128, 128) | float32 | Reference f-k volume reconstruction (batch, depth, height, width) |

The scene is the real outdoor test case from Lindell et al. 2019: a building
facade at ~2.45 m depth scanned from a wall 2 m × 2 m.

## Method Hints

Implement `fk_reconstruction` using Stolt f-k migration: pre-process the transient data with amplitude scaling, pad and 3-D FFT, apply the hyperbolic Stolt frequency mapping (temporal frequencies → depth frequencies via the wave dispersion relation with a Jacobian correction), then inverse 3-D FFT. No regularisation parameters are needed.

## References

- Lindell, Wetzstein, O'Toole, "Wave-Based Non-Line-of-Sight Imaging using
  Fast f-k Migration", ACM Trans. Graph. 38(4), 2019.  [nlos_fk.pdf]
- O'Toole et al., "Confocal Non-Line-of-Sight Imaging Based on the Light
  Cone Transform", Nature 555, 2018.  [nature25489.pdf]
- Reference code: https://github.com/computational-imaging/nlos-fk
