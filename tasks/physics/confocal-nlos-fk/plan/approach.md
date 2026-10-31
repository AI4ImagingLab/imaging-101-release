# f-k Migration Reconstruction — Approach

## Algorithm Overview

The f-k (frequency-wavenumber) migration method treats NLOS imaging as a
seismic-style wave backpropagation problem, applying the Stolt migration
algorithm originally developed for subsurface imaging.

### Step 1: Pre-processing

Apply amplitude correction and depth compensation:

    Ψ(t, y, x) = √( |τ(t,y,x)| · t² )

where t = linspace(0,1,M). The t² factor compensates geometric spreading
and the sqrt provides the correct amplitude weighting for the wave equation.

### Step 2: Zero-pad and 3-D FFT

Pad Ψ to a (2M, 2N, 2N) array (zero-padding in all dimensions).
Take the 3-D FFT followed by fftshift:

    T(kf, ky, kx) = fftshift( FFT3D( Ψ_padded ) )

### Step 3: Stolt frequency mapping

Map the temporal frequency kf to a depth frequency kz_new:

    kz_new = √( scale² · (kx² + ky²) + kf² )

where the scale factor is:

    scale = (N · range_m) / (M · width · 4)
    range_m = M · c · bin_resolution
    width   = wall_size / 2

This mapping implements the dispersion relation of a diffracted wavefield.
Interpolate T at the new coordinates using bilinear (`order=1`) interpolation
of real and imaginary parts separately.

Apply the Jacobian weight |kf| / max(kz_new, ε) and zero out negative-
depth components (kf ≤ 0).

### Step 4: Inverse 3-D FFT and intensity

    vol_padded = |iFFT3D( ifftshift(T_migrated) )|²

Unpad: return the first (M, N, N) block.

## Complexity

O(N³ log N) time, O(N³) memory. The Stolt interpolation is the only
non-FFT step; it runs in O(N³) with `map_coordinates`.

## Key Considerations

- **No free parameters**: unlike LCT, f-k requires no SNR tuning.
- **Handles specular objects**: the wave-based formulation does not assume
  Lambertian reflectance.
- **Scale factor**: the factor of 4 in the denominator accounts for the
  round-trip light path (2×) and the confocal geometry normalisation (2×).
- **Stolt artefacts**: mild interpolation artefacts at large off-axis angles;
  manifest as faint ghost images from |·|² cross-terms at low intensity.
