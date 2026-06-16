# Single-Molecule Light Field Microscopy (SMLFM): 3D Super-Resolution Reconstruction

> Localize single fluorescent molecules in 3D from multi-view 2D detections by exploiting parallax between micro-lens views of a hexagonal Fourier light field microscope.

> Domain: Biology | Keywords: super-resolution, localization microscopy, light field imaging | Difficulty: Medium

---

## Background

Fourier Light Field Microscopy (FLFM) is a technique that achieves 3D single-molecule super-resolution by replacing the standard camera with a micro-lens array (MLA) placed at the back focal plane of the imaging system.

Each micro-lens samples a different **view** of the sample — a different angle of incidence — so the same fluorescent molecule simultaneously appears at slightly different lateral positions in several sub-images. The lateral shift between views (parallax) carries depth information: molecules above or below focus shift in opposite directions, and the shift magnitude encodes how far they are from focus.

The computational task is the **inverse problem**: given 2D localisations (x, y) detected by standard SMLM fitting software (e.g. ThunderStorm or PeakFit) across thousands of camera frames, recover the 3D positions (x, y, z) of individual fluorescent molecules.

## Problem Description

The Fourier relay maps the objective back focal plane (BFP) onto the MLA plane. Each micro-lens therefore samples a unique solid angle of the emission cone, parameterised by normalised pupil coordinates (u, v):

```
  u = (x_bfp / BFP_radius),  v = (y_bfp / BFP_radius)
  BFP_radius = NA * f_obj * (f_fourier / f_tube)  [microns]
```

A point source at 3D position (x₀, y₀, z) forms an image at position (xᵢ, yᵢ) through microlens i:

```
  xᵢ = x₀  +  uᵢ / ρ  +  z · αᵤᵢ
  yᵢ = y₀  +  vᵢ / ρ  +  z · αᵥᵢ
```

where:
- **ρ** (rho_scaling) = magnification / BFP_radius  converts image-plane microns → normalised pupil coords
- **α(u, v)** is the angular sensitivity — how much a unit depth change shifts the apparent lateral position in a given view. For a spherical wavefront:

```
  α(u, v)  =  -(NA/n) / √(1 − ρ² (NA/n)²)  ×  (u, v)
```

The most accurate model (INTEGRATE_SPHERE) phase-averages this expression over the finite aperture of each micro-lens.

## Data Description

`data/raw_data.npz` — 2D localisations from a fixed BCR (B-cell receptor) B cell, imaged on a hexagonal FLFM. Localised using ImageJ GDSC SMLM2 PeakFit plugin.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `localizations_2d` | (150996, 35) | float32 | Raw PeakFit localisation array bundled with the task |

The bundled NPZ stores the original PeakFit-style columns. During loading,
`src.preprocessing.load_localizations` converts the required subset into the
standardised internal `(N, 8)` array used by the pipeline:
- raw **0** → frame index
- raw **9** → X position (pixels, converted to μm via `pixel_size_sample`)
- raw **10** → Y position (pixels, converted to μm via `pixel_size_sample`)
- raw **12** → sigma_X / sigma_Y (pixels, converted to μm)
- raw **8** → intensity (photons)
- raw **7** → background (photons)
- raw **13** → precision (nm, converted to μm)

Internal standardised layout after loading:
- **0** frame index
- **1** X position (μm)
- **2** Y position (μm)
- **3** sigma_X (μm)
- **4** sigma_Y (μm)
- **5** intensity (photons)
- **6** background (photons)
- **7** precision (μm)

> **Note**: No `ground_truth.npz` is provided for this task — the data comes from
> real experimental recordings with no simulation ground truth.

- **150,996 localisations** in **5,000 frames**
- Hexagonal MLA: 7 views in the back focal plane
- 60× water-immersion objective, NA 1.27

Source: `hexSMLFM` repository (TheLeeLab, University of Cambridge).
Reference paper: [R. R. Sims et al., Optica 7, 1065 (2020)](https://doi.org/10.1364/OPTICA.397172)

## Method Hints

1. **Build optics model**: Construct a hexagonal MLA lattice rotated by `mla_rotation = 30.8°`, assign each localisation to its nearest micro-lens by k-NN in normalised pupil (u, v) coordinates, and compute the angular sensitivity α using the phase-averaged sphere model (INTEGRATE_SPHERE).

2. **3D fitting**: For each frame, group localisations from different views by geometric consistency (parallax direction and disparity), then solve an overdetermined linear system (OLS) for (x₀, y₀, z) per molecule. Perform a first-pass fit on early frames to estimate per-view aberration offsets before full reconstruction.

3. **Z calibration**: Scale the raw optical z by `z_calib` (from bead scan measurements) to convert to physical depth in microns.

## References

- R. R. Sims, S. Abdul Rehman, M. D. Lew, S. I. Golding, E. Galgani, J. G. Bhatt, C. Croop, S. F. Lee, K. Bhatt, G. Bhatt. *Single-molecule light field microscopy.* Optica 7, 1065–1072 (2020). DOI: [10.1364/OPTICA.397172](https://doi.org/10.1364/OPTICA.397172)
- `hexSMLFM` repository, TheLeeLab, University of Cambridge: https://github.com/TheLeeLab/hexSMLFM
