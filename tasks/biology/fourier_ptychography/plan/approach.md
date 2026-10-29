# Approach: Fourier Ptychography Reconstruction via qNewton

## Problem Statement

Fourier ptychography (FPM) reconstructs a high-resolution complex object spectrum
Õ(q) = FT{O(r)} and lens pupil function P̃(q) from a sequence of low-resolution
(LR) images recorded under varying LED illumination angles.

For each LED at angle θ_j, the pupil bandpass-filters a shifted region of the
object spectrum:

    I_j(r) = |IFT{P̃(q) · Õ(q − q_j)}|²

where q_j = 2π sin(θ_j) / λ is the k-space shift. The LR images from different
LEDs overlap in k-space, enabling joint phase retrieval and super-resolution.

## Dataset

Synthetic simulation parameters (matching visible-light FPM setup):
- **Wavelength**: λ = 625 nm (red LED)
- **Magnification**: 4×
- **Numerical aperture**: NA = 0.1
- **Camera pixel size**: dxd = 6.5 μm
- **Object pixel size**: dxp = dxd / M = 1.625 μm
- **Low-res image size**: Nd = 256 px
- **High-res object size**: No = 730 px (computed from LED array extent)
- **LED array**: 11×11 grid (121 LEDs, pitch 2 mm, z_led = 60 mm; all 121 within k-space boundary)
- **NAmax**: NA + 5 × 2mm/60mm ≈ 0.27 (bright + dark field coverage)
- **Noise**: Poisson noise (10-bit dynamic range)

## Algorithm: Quasi-Newton (qNewton)

qNewton [Loetgering et al., Opt. Express 2023] is the FPM solver used here.
It uses a Hessian-scaled gradient step that improves convergence speed over
plain PIE without requiring a momentum buffer.

### PIE-type update (per LED position j)

In FPM, the PIE update (not ePIE) is used because:
- The bright-field images have much higher SNR than dark-field images
- ePIE (which updates only near the pupil maximum) would concentrate updates
  in bright-field regions and neglect dark-field, limiting resolution
- PIE distributes updates more uniformly, improving super-resolution

Object spectrum update:
    Õ_{n+1}(q − q_j) = Õ_n(q − q_j) + β_O · (|P̃_n|/Pmax) · P̃_n* · (ψ̂_j − ψ_j) / (|P̃_n|² + reg_O)

Pupil update:
    P̃_{n+1}(q) = P̃_n(q) + β_P · (|Õ_j|/Omax) · Õ_j* · (ψ̂_j − ψ_j) / (|Õ_j|² + reg_P)

where ψ̂_j is the constrained detector wave: ψ̂_j = sqrt(I_j) · ψ_j / |ψ_j|.

The (|P̃|/Pmax) prefactor is the quasi-Newton Hessian approximation — it
scales each update step by the local intensity, giving faster convergence
than plain PIE (which uses a constant step) without the complexity of
full momentum methods.

Default parameters: β_O = β_P = 1.0, reg_O = reg_P = 1.0.

### LED ordering

Positions are sorted by NA (bright-field first, dark-field last). This ordering
ensures stable convergence: the bright-field images provide a good initial
estimate, which is then refined by the dark-field images.

## Convergence Metric

Same as CP: normalized amplitude discrepancy:

    E = Σ_j |I_meas,j − I_est,j| / Σ_j I_meas,j

Target: E ≈ 30 after 200 iterations on the simulated dataset (Poisson noise, 10-bit).

## Key References

- Zheng, Horstmeyer & Yang, Nat. Photonics 7, 739 (2013) — FPM concept
- Ou, Zheng & Yang, Opt. Express 22, 4960 (2014) — pupil recovery
- Loetgering et al., Opt. Express 31, 13763 (2023) — PtyLab (qNewton/mqNewton)
