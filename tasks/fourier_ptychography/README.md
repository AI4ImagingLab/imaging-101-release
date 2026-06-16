# Fourier Ptychography Reconstruction

> Computationally synthesize a wide-field high-resolution complex image from low-resolution images captured under varying LED illuminations using Fourier ptychographic microscopy.

> Domain: Biology | Keywords: phase retrieval, ptychography, super-resolution | Difficulty: Medium

---

## Background

Fourier ptychography (FPM) is a computational microscopy technique that achieves
simultaneous high resolution and wide field of view by computationally stitching
together low-resolution images captured under different LED illumination angles.

This task implements the full FPM reconstruction pipeline via a self-contained qNewton
solver, reproducing a synthetic data example inspired by the PtyLab benchmark
(Loetgering et al., Opt. Express 2023).

## Problem Description

```
   LED array (11×11 = 121 LEDs, z=60mm)
         ↓ angle-dependent illumination
   [sample O(r)]  ──→  lens (NA=0.1, M=4×)  ──→  camera (256×256 px)
                       pupil P̃(q)                   I_j(r)
```

| Parameter | Value |
|---|---|
| Wavelength | 625 nm (red LED) |
| Numerical aperture | NA = 0.1 |
| Magnification | 4× |
| Camera pixel size | 6.5 μm |
| Object pixel size | 1.625 μm |
| Low-res image size | 256 × 256 px |
| High-res object size | 730 × 730 px (No computed from LED array geometry) |
| LED array | 11×11, pitch 2 mm, z = 60 mm |
| Super-resolution factor | ~4.3× |

## Data Description

### `data/raw_data.npz`

Low-resolution images from a synthetic 11×11 LED-array Fourier ptychography experiment.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `ptychogram` | (121, 256, 256) | float32 | Low-resolution images (one per LED position) |
| `encoder` | (121, 2) | float32 | LED positions in meters [x, y] at distance z_led |

### `data/ground_truth.npz`

Ground-truth high-resolution complex object (real space).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `object` | (730, 730) | complex64 | Ground-truth complex object in real space O(r); pure-phase (amplitude=1, binary phase ∈ {0, π/2}) |

### `data/meta_data.json`

JSON file with optical system parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `wavelength_m` | 6.25e-7 | LED wavelength (m) |
| `NA` | 0.1 | Objective numerical aperture |
| `magnification` | 4.0 | Objective magnification |
| `dxd_m` | 6.5e-6 | Camera pixel size (m) |
| `dxp_m` | 1.625e-6 | Object pixel size (m) (= dxd / magnification) |
| `z_led_m` | 0.06 | LED array to sample distance (m) |
| `led_pitch_m` | 0.002 | LED pitch (m) |
| `n_leds_side` | 11 | LEDs per side (11×11 array) |
| `num_leds` | 121 | Total valid LEDs after k-space filtering |
| `Nd` | 256 | Low-resolution image size (pixels) |
| `No` | 730 | High-resolution object size (pixels) |

## Method Hints

The key insight in FPM is that each LED illuminates the sample from a different
angle, shifting the object's k-space spectrum under the pupil bandpass:

    I_j(r) = |IFT{P̃(q) · Õ(q − q_j)}|²

By choosing LED angles such that adjacent k-space regions **overlap**, we can
perform phase retrieval via iterative projection algorithms (similar to CP).

The super-resolution is achieved because the maximum recoverable spatial
frequency is: k_max = 2π(NA + sin θ_max)/λ, where sin θ_max is determined by
the outermost LED angle.

## References

L. Loetgering et al., "PtyLab.m/py/jl: a cross-platform, open-source inverse
modeling toolbox for conventional and Fourier ptychography," Opt. Express 31,
13763–13797 (2023).
