# Approach: Electron Ptychography of Gold Nanoparticles

## Problem Statement

Recover the complex transmission function O(r) of gold nanoparticles on amorphous carbon from a 4D-STEM dataset of 48x48 scan positions with 192x192 diffraction patterns each, recorded at 300 keV with a defocused probe (~355 Angstrom defocus).

## Data Preprocessing

1. **Load data**: Read the 4D-STEM datacube (48, 48, 192, 192) and vacuum probe (192, 192) from HDF5 files.
2. **Mean diffraction pattern**: Average all 2304 diffraction patterns to estimate the average probe position and bright-field disk.
3. **Probe size estimation**: Threshold the vacuum probe intensity (thresh_upper=0.7) to find the bright-field disk radius (~49.2 pixels) and center (~96.9, ~88.3).
4. **Calibration**: Set real-space pixel size (2.39 A) and reciprocal-space pixel size (convergence_semiangle / probe_radius = 22.6 / 49.2 ~ 0.46 mrad/pixel).
5. **Bright-field mask**: Threshold the mean diffraction pattern at 0.8 of its maximum to create a binary mask of the BF disk for DPC.

## Reconstruction Pipeline

### Stage 1: DPC (Baseline)

- Compute the center-of-mass (CoM) of each diffraction pattern within the BF mask.
- Estimate the rotation angle between scan and detector coordinates from the CoM field (best fit ~ -15 degrees).
- The 180-degree ambiguity must be resolved: gold has positive electrostatic potential, so the phase must be positive. Add 180 degrees to get the correct rotation (165 degrees).
- Integrate the rotated CoM field to obtain the phase image using iterative Fourier integration (up to 64 iterations, step size halving with stopping criterion 1e-6).

### Stage 2: Parallax (Intermediate)

- Extract virtual bright-field images for each detector pixel within the BF disk (threshold_intensity=0.6, edge_blend=2).
- Align images via hierarchical cross-correlation: start at bin 64, proceed to bin 8, with 6 iterations at the minimum bin.
- Compute subpixel shifts using kernel density estimation (KDE) with upsample factor 4.
- Fit the aberration surface from cross-correlation shifts (up to 3rd order radial and angular):
  - Affine fit gives rotation and defocus.
  - BF shift fit refines higher-order aberrations.
  - CTF FFT fit matches Thon ring zero-crossings.
- Apply 180-degree rotation correction (force_rotation_deg = -11 + 180 = 169 degrees).
- Apply CTF correction to the upsampled BF image using the fitted aberration coefficients.

### Stage 3: Ptychography (Primary)

- Initialize the probe from the vacuum measurement with the defocus estimate from parallax (C1 ~ -355 A, so defocus = 355 A).
- Use the rotation (169 degrees) and transpose (False) parameters from the preprocessing.
- Run single-slice gradient-descent ptychography:
  - 10 iterations
  - Step size: 0.5
  - Mini-batch size: J/4 = 576 patterns per batch
  - Complex object type (amplitude and phase)
- The reconstruction simultaneously solves for the complex object and refines the probe.

## Evaluation

- Compare DPC, parallax, and ptychographic phase images qualitatively.
- Compute NCC and NRMSE between agent reconstruction and reference ptychographic phase.
- The ptychographic reconstruction provides the highest resolution and serves as the reference.

## Key Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Energy | 300 keV | Experiment |
| Scan step | 2.39 A | Experiment |
| Convergence semi-angle | 22.6 mrad | Experiment |
| BF mask threshold | 0.8 | Empirical |
| CoM rotation | 169 deg | DPC fit + 180 flip |
| Defocus | 355 A | Parallax CTF fit |
| Transpose | False | Parallax fit |
| Ptycho iterations | 10 | Convergence |
| Ptycho step size | 0.5 | Default |
| Ptycho batch size | 576 | J/4 |
