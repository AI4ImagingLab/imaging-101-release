# Approach: Stolt f-k Migration for Plane Wave Ultrasound

## Overview

We reconstruct focused ultrasound B-mode images from multi-angle plane-wave RF data using Stolt's f-k migration with coherent angle compounding. The algorithm operates entirely in the frequency–wavenumber domain and is O(N log N) in both image dimensions.

## Physical Model

### Exploding Reflector Model (ERM)

For a plane wave steered at angle θ (relative to the array normal), the round-trip travel time for a scatterer at position (x_s, z_s) is:

$$t = \frac{1}{c}\bigl(z_s \cos\theta + (x_s - x_e)\sin\theta\bigr) \cdot 2 \cdot \frac{c}{c + c\cos\theta} + \text{lateral term}$$

Under the ERM, this two-way propagation is equivalent to one-way propagation at an effective velocity:

$$v = \frac{c}{\sqrt{1 + \cos\theta + \sin^2\theta}}$$

This allows us to apply a classical single-way migration (Stolt's method) to ultrasound data.

### Stolt Mapping

In the (f, k_x) domain, the dispersion relation for upward-propagating waves at velocity v is:

$$k_z = \frac{1}{v}\sqrt{\left(\frac{2\pi f}{v}\right)^2 - k_x^2}$$

Stolt migration maps the recorded temporal frequency f to the migrated axial wavenumber k_z:

$$f_{kz} = v\sqrt{k_x^2 + \frac{4 f^2}{\beta^2 c^2}}$$

where β = (1 + cosθ)^{1.5} / (1 + cosθ + sin²θ) is a geometric factor accounting for the plane-wave steering. The mapping is applied by linear interpolation in the frequency axis for each k_x slice.

An obliquity factor f / (f_kz + ε) is applied after remapping to account for the Jacobian of the coordinate change.

### Steering Compensation

Before the spatial FFT, the per-element arrival-time offsets due to the steering angle are removed:

$$\text{SIG}(f, x_e) \;\leftarrow\; \text{SIG}(f, x_e) \cdot \exp\!\bigl(-2\pi i\, f\, t_\text{shift}(x_e)\bigr)$$

where t_shift(x_e) = sin(θ) × (index of element) × pitch / c + t₀.

After the inverse axial FFT (IFFT along f), a lateral phase ramp corrects for the steering-induced spatial shift that varies with depth:

$$\text{SIG}(z, k_x) \;\leftarrow\; \text{SIG}(z, k_x) \cdot \exp\!\bigl(-2\pi i\, k_x\, \Delta x(z)\bigr)$$

where Δx(z) = −γ z, γ = sinθ / (2 − cosθ).

## Full Pipeline

1. **DC removal** — subtract the spatial mean from the raw 8-bit RF data.
2. **Zero-padding** — pad the time axis to 4×N_t + N_shift to reduce circular aliasing from the linear interpolation. Pad the spatial axis to 1.5×N_x to avoid lateral wrap-around artifacts.
3. **Axial FFT** — apply FFT along the time axis; retain only the positive-frequency half (signal is real).
4. **Steering delay removal** — apply per-element phase rotation in (f, x) domain.
5. **Spatial FFT** — apply FFT along the element axis.
6. **Evanescent suppression** — zero out components where |f|/(|k_x| + ε) < c (sub-sonic phase velocity; these cannot propagate in tissue).
7. **Stolt f-k mapping** — for each k_x column, interpolate the f-axis from uniform frequency samples to f_kz(k_x, f).
8. **Obliquity correction** — multiply by f / (f_kz + ε).
9. **Axial IFFT** — reconstruct from the one-sided spectrum, apply IFFT along the frequency axis.
10. **Lateral steering compensation** — multiply by exp(−2πi k_x Δx(z)) for each depth.
11. **Lateral IFFT** — apply IFFT along the k_x axis.
12. **Crop** — extract the valid (N_t, N_x) region.
13. **Coherent compounding** — average the complex migrated RF images over all angles.
14. **Envelope detection** — apply Hilbert transform along the time axis to obtain the analytic signal, then take |·|. Optionally apply power-law compression (e.g., ^0.5 or ^0.7) for display.

## Implementation Notes

- All FFT lengths are chosen as integers; even lengths avoid off-by-one issues in RFFT symmetry.
- The linear interpolation in step 7 is vectorized: the floor index and fractional part are computed for the entire 2-D (f, k_x) grid simultaneously.
- Evanescent zeroing in step 6 must be applied before interpolation, otherwise evanescent energy is smeared into the physical cone.
- The final cropping in step 12 uses `ntshift = 2*ceil(t0*fs/2)` to account for the acquisition start time offset (nonzero for dataset 2).
