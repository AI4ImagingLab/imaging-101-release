# Approach: Single-Molecule Light Field Microscopy 3D Reconstruction

## Problem Summary

Given 2D single-molecule localisations (x, y, intensity, frame) detected in a Fourier Light Field Microscope (FLFM), reconstruct the 3D positions (x, y, z) of individual molecules using multi-view parallax.

## Core Idea: Multi-View Parallax

The FLFM's micro-lens array (MLA) sits at the back focal plane, creating multiple simultaneous views of the sample. A molecule at axial depth z appears shifted laterally by different amounts in each view — the shift is proportional to depth and the view's normalised pupil position (u, v).

The forward model for molecule i observed in view k:
```
  x_k = x₀ + u_k / ρ + z · α_u(u_k, v_k)
  y_k = y₀ + v_k / ρ + z · α_v(u_k, v_k)
```

Inverting this over-determined system (multiple views → OLS) gives (x₀, y₀, z).

## Step-by-Step Algorithm

### 1. Preprocessing
- Load raw PeakFit localisations from `raw_data.npz`, extract the required columns,
  and scale pixel-based quantities to microns using `pixel_size_sample`
- Centre X, Y by subtracting their means (aligns data with MLA centre)

### 2. Optics Model
- Compute `BFP_radius = NA × f_obj × (f_fourier / f_tube)` [µm]
- Compute `magnification = (f_tube / f_obj) × (f_mla / f_fourier)`
- Compute `rho_scaling = magnification / BFP_radius`
- Generate hexagonal MLA lattice, rotate by `mla_rotation` degrees

### 3. Lens Assignment
- Convert each 2D localisation (x, y) → normalised pupil coordinate (u, v) = (x, y) × rho_scaling
- Assign each localisation to its nearest MLA lens centre by k-NN search in (u, v) space

### 4. Filtering
- Remove localisations assigned to lenses outside the BFP circle
- Optional: filter by rho range, spot size, photon count

### 5. Alpha Model (Phase-Averaged Sphere)
For each localisation at view (u, v), compute angular sensitivity α by averaging over the microlens aperture:
```
α(u, v) = ∬_{lens} [-(NA/n) / √(1 - (u'² + v'²)(NA/n)²)] × (u', v') du'dv'
```
This is the INTEGRATE_SPHERE model — more accurate than the linear or point-sphere approximations.

### 6. Aberration Correction (Two-Pass Strategy)
- **Pass 1**: fit first 1000 frames with loose thresholds (min 3 views, threshold 1 µm)
- Compute per-view mean residuals → correction table (one (dx, dy) per view)
- Apply correction by subtracting per-view offsets from X, Y before full fitting

### 7. Full 3D Fitting
For each frame, iteratively group localisations into putative single-molecule events:
1. **Seed selection**: start with brightest central-view localisation
2. **Grouping**: find companion localisations in other views that are consistent with a single Z hypothesis (angle test + disparity test)
3. **OLS fit**: solve A·[x₀, y₀, z]ᵀ = b by `np.linalg.lstsq`
4. **Quality filter**: accept only if total fit error < threshold (0.3 µm)
5. **Remove** accepted localisations from candidate pool; repeat

### 8. Z Calibration
Multiply raw optical Z by `z_calib = 1.534` (calibrated from bead scans at known axial positions).

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| NA | 1.27 | Numerical aperture |
| f_obj | 200/60 mm | Objective focal length |
| f_tube | 200 mm | Tube lens focal length |
| f_fourier | 175 mm | Fourier relay focal length |
| f_mla | 175 mm | MLA focal length |
| mla_pitch | 2390 µm | Lens-to-lens spacing |
| pixel_size | 16 µm | Camera pixel |
| mla_rotation | 30.8° | MLA tilt relative to camera |
| alpha_model | INTEGRATE_SPHERE | Depth-angle mapping model |
| z_calib | 1.534 | Optical-to-physical Z factor |
| disparity_max | 8 µm | Max Z search range (full pass) |
| threshold | 0.3 µm | Max accepted OLS residual |
| min_views | 2 | Min views for valid 3D fit |

## Expected Results

For the BCR B cell dataset:
- ~24,800 3D localisations from 5,000 frames
- Median lateral precision: ~52 nm
- Median axial precision: ~55 nm
- 7 unique microlens views used; median 6 views per localisation
- Z range: −3 to +6 µm (thin membrane structure)

These numbers match the precision values reported in Sims et al. (2020, Optica) for this type of FLFM configuration.

## Default Parameters

These values are hard-coded in `main.py` and were removed from `meta_data.json` (which is reserved for imaging parameters only):

### Aberration-correction pass (`fit_params_aberration`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `frame_min` / `frame_max` | −1 | Use all frames (−1 = no limit) |
| `disparity_max` | 5.0 | Maximum parallax disparity search range (μm) |
| `disparity_step` | 0.1 | Disparity search step size (μm) |
| `dist_search` | 0.5 | Nearest-neighbour search radius (lens spacings) |
| `angle_tolerance` | 2.0 | Angle tolerance for multi-view matching (degrees) |
| `threshold` | 1.0 | Minimum fit quality threshold |
| `min_views` | 3 | Minimum number of views required per molecule |
| `z_calib` | None | Z calibration factor (None = not applied at this stage) |

### Aberration map criteria (`aberration_params`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `axial_window` | 1.0 | Axial depth window for aberration estimation (μm) |
| `photon_threshold` | 1 | Minimum photon count to include in aberration map |
| `min_views` | 3 | Minimum views per molecule for aberration estimation |

### Full-dataset pass (`fit_params_full`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `frame_min` / `frame_max` | −1 | Use all frames |
| `disparity_max` | 8.0 | Maximum parallax disparity search range (μm) |
| `disparity_step` | 0.1 | Disparity search step size (μm) |
| `dist_search` | 0.5 | Nearest-neighbour search radius (lens spacings) |
| `angle_tolerance` | 1.0 | Angle tolerance for multi-view matching (degrees) |
| `threshold` | 0.3 | Minimum fit quality threshold (looser than aberration pass) |
| `min_views` | 2 | Minimum number of views required per molecule |
| `z_calib` | 1.534 | Z calibration factor (μm/μm, determined from bead measurements) |
