# SSNP Optical Diffraction Tomography

> Reconstruct the 3D refractive index (RI) distribution of a biological sample from multi-angle complex field measurements using split-step non-paraxial (SSNP) wave propagation, where access to both amplitude and phase makes the inverse problem better constrained than intensity-only IDT.

> Domain: biology | Keywords: optical diffraction tomography, wave imaging, inverse scattering | Difficulty: Hard

## Background

Optical Diffraction Tomography (ODT) recovers the 3D refractive index (RI) distribution
of a semi-transparent sample by illuminating it from multiple angles and recording the
transmitted complex field at a detector. Because both amplitude and phase are measured
(e.g., via interferometry or holography), the inverse problem is better constrained than
Intensity Diffraction Tomography (IDT), which measures only |E|².

The Split-Step Non-Paraxial (SSNP) model propagates the scalar field-and-derivative wave state
Φ = (φ, ∂φ/∂z) through the sample, accounting for non-paraxial effects and multiple
scattering that are ignored by the first Born or paraxial approximations. This is
important for thick samples (>10 μm) or high-NA systems.

## Problem Description

The forward measurement model maps a 3D RI contrast volume Δn(x,y,z) = n(x,y,z) − n₀
to a set of L complex field images {E^l}_{l=1}^L, one per illumination angle:

    E^l = F · P_NA · P_{-Nz/2} · [ P · Q ]^{Nz} · Φ₀^l

where:
- Φ₀^l = (φ₀^l, ∂φ₀^l/∂z) is the incident tilted plane wave for angle l
- P (propagation operator): free-space diffraction step in Fourier space
- Q (scattering operator): RI-induced phase modulation in real space
- P_{-Nz/2}: back-propagation from exit plane to focal plane
- P_NA: binary pupil filter limiting spatial frequencies to NA/λ
- F: projection onto the forward-propagating component

The P operator acts on the 2D Fourier coefficients (â, â_d) per axial slice:

    â_new   =  cos(kz·Δz) · â  +  sin(kz·Δz)/kz · â_d
    â_d_new = −kz·sin(kz·Δz) · â + cos(kz·Δz) · â_d

where kz = (2π·res_z) · √(1 − fx² − fy²) is the axial wave vector.

The Q operator updates the z-derivative in real space:

    u_d ← u_d − k₀²(2·n₀·Δn + Δn²)·Δz · u

The inverse problem is to recover Δn from the measured complex fields {E^l}.
Because multiple scattering couples all z-slices, the problem is strongly
nonlinear and large-scale (308 × 256 × 256 unknowns).

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `measurements` | (1, 8, 256, 256) | complex128 | Complex exit fields for 8 illumination angles (batch × angles × Ny × Nx); real and imaginary parts are quadratures of the same scalar field after pupil filtering. Units: normalised field amplitude. |

Illumination geometry: 8 LED positions on an annular ring at NA = 0.65,
uniformly spaced in azimuth: angle_l = 2π·l/8 for l = 0, …, 7.

### data/meta_data.json

| Key | Type | Value | Description |
|-----|------|-------|-------------|
| `volume_shape` | list[int] | [308, 256, 256] | Volume dimensions [Nz, Ny, Nx] |
| `res_um` | list[float] | [0.1, 0.1, 0.1] | Isotropic voxel size in μm |
| `wavelength_um` | float | 0.5 | Illumination wavelength in μm |
| `n0` | float | 1.0 | Background refractive index |
| `NA` | float | 0.65 | Objective numerical aperture |
| `n_angles` | int | 8 | Number of illumination angles |
| `ri_contrast_scale` | float | 0.01 | Scale factor applied to phantom TIFF |
| `tiff_scale` | float | 1.0 | TIFF normalisation factor |

## Method Hints

The standard approach is gradient-descent on the complex field MSE loss:

    L = Σ_l ‖E_pred^l − E_meas^l‖² / N_pixels

where E_pred^l is the SSNP forward model output for angle l and E_meas^l is
the measured complex field. Because the loss is real-valued (squared modulus
of a complex residual), gradients w.r.t. the real Δn volume are well-defined.
PyTorch autograd propagates through the complex SSNP operators automatically.

A non-negativity constraint (Δn ≥ 0, appropriate for biological cells in
aqueous medium) can be enforced by projection after each gradient step.
3D total variation regularization can suppress noise at the cost of smoothing
fine structure.

## References

1. Chen, M., Ren, D., Liu, H.-Y., Chowdhury, S., & Waller, L. (2020).
   Multi-layer Born multiple-scattering model for 3D phase microscopy.
   *Optica*, 7(5), 394–403.

2. Tian, L., & Waller, L. (2015). 3D intensity and phase imaging from
   light field measurements in an LED array microscope.
   *Optica*, 2(2), 104–111.

3. Liu, H.-Y., Liu, D., Mansour, H., Boufounos, P. T., Waller, L., &
   Kamilov, U. S. (2022). SEAGLE: Sparsity-driven image reconstruction
   under multiple scattering. *IEEE Transactions on Computational Imaging*,
   4(1), 73–86.
