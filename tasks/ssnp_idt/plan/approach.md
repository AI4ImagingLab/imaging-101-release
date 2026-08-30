# Approach

## Problem Statement

Recover a 308x256x256 voxel 3D refractive index (RI) distribution from 8 intensity-only images collected under annular LED illumination at NA = 0.65, using the Split-Step Non-Paraxial (SSNP) multiple-scattering model.

## Mathematical Formulation

The SSNP model describes light propagation through an inhomogeneous medium by tracking both the field and its z-derivative as a state vector. The 3D sample is discretized into axial slices, and for each slice the model applies:

1. **P operator** (diffraction): free-space propagation by Δz in Fourier space — a 2×2 rotation matrix involving cos(kz·Δz) and sin(kz·Δz)/kz
2. **Q operator** (scattering): local RI modulation in real space — modifies the z-derivative by k₀²(n₀² − n²)Δz · field

The forward model for each illumination angle:
- Initialize tilted plane wave with direction (kx, ky) at NA = 0.65
- Propagate through all 308 slices: alternating P·Q, i.e. scattering by Q followed by propagation by P.
- propagate to focal plane, apply pupil, extract forward component
- Compute intensity: I = |φ_out|²

The inverse problem minimizes:

$$\hat{\Delta n} = \arg\min_{\Delta n \geq 0} \sum_{l=1}^{8} \left\| \sqrt{I_{pred}^l} - \sqrt{I_{meas}^l} \right\|^2 + \tau \cdot TV_{3D}(\Delta n)$$

## Solution Strategy

### Step 1: Data Preprocessing

Load the phantom TIFF (308×256×256 uint16), scale to RI contrast Δn via `tiff_scale * ri_contrast_scale / 65535`. Load metadata JSON for physical parameters.

### Step 2: Forward Model Construction

Build the SSNP forward model in PyTorch:
- Precompute kz grid, evanescent damping mask, and binary pupil
- Implement P operator as element-wise 2×2 matrix multiply in Fourier space
- Implement Q operator as element-wise update in real space
- Handle incident field construction (tilted plane wave + z-derivative)
- Forward/backward decomposition and pupil filtering at the exit plane

### Step 3: Measurement Simulation

For each of the 8 illumination angles (uniformly spaced on a ring at NA = 0.65):
- Construct incident field
- Propagate through the phantom volume
- Extract intensity at the camera plane

This replaces experimental data collection and serves as the "observation."

### Step 4: 3D RI Reconstruction

Gradient-descent optimization:
- Initialize Δn = 0 (uniform background)
- For each iteration:
  - Compute predicted intensities via SSNP forward model (all 8 angles)
  - Amplitude MSE data-fidelity loss
  - 3D Total Variation regularization (Huber-smoothed for differentiability)
  - Backpropagate gradient through entire model via PyTorch autograd
  - Update Δn with fixed learning rate
  - Project to non-negative: Δn = max(Δn, 0)
- Track loss history for convergence monitoring

### Step 5: Evaluation

Compare reconstructed Δn against ground truth phantom:
- **NRMSE**: ‖Δn̂ − Δn_GT‖₂ / ‖Δn_GT‖₂
- **NCC**: normalized cross-correlation
- **SSIM**: structural similarity index

### Step 6: Visualization

Generate XY slices, XZ cross-sections, comparison panels, and loss convergence curves.

## Expected Results

| Metric | Value |
|--------|-------|
| NRMSE  | < 0.5 |
| NCC    | > 0.8 |

The reconstruction should recover the main structural features of the phantom with correct RI contrast values. Resolution is limited by the finite Fourier coverage (NA = 0.65) and the missing-cone artifact inherent to limited-angle tomography.
