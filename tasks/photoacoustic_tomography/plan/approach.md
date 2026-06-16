# Approach: Universal Back-Projection for PAT

## Algorithm Overview

We use the Xu & Wang (2005) universal back-projection (UBP) algorithm to reconstruct the initial pressure distribution from planar detector measurements.

## Pipeline Stages

### 1. Data Generation
- Define 4 spherical absorbers (3 small, 1 large) in a plane 15 mm above the detector array
- Simulate PA signals using the analytical forward model for spherical absorbers
- Each 2 mm detector is subdivided into 5x5 sub-elements for spatial integration
- Signal superposition from all targets (linear acoustics)

### 2. Preprocessing
- Load signals from npz format
- Load detector geometry and time vector
- Load imaging parameters from metadata

### 3. Reconstruction (Universal Back-Projection)
- **Reconstruction grid**: pixels spaced at 500 um resolution over the detector aperture
- **Ramp filter**: for each detector trace, compute `pf = IFFT(-j*k * FFT(p, NFFT))` where k is the wavenumber vector and NFFT = 2048
- **Back-projection quantity**: `b(t) = 2*p(t) - 2*t*c*pf(t)`
- **Delay mapping**: for each pixel, compute distance to detector and convert to time index via `index = round(distance * fs / c)`
- **Solid angle weighting**: `omega = det_area * cos(theta) / r^2` where `cos(theta) = z/r`
- **Accumulation**: `p0(r) = sum(omega * b[index]) / sum(omega)`
- **Normalisation**: divide by the maximum absolute value

### 4. Evaluation
- Centre-crop both reconstruction and ground truth (80% central region)
- Compute NCC (cosine similarity) and NRMSE (RMS error / dynamic range)

## Key Parameters
- Sound speed: 1484 m/s (water at ~20 C)
- Sampling: 20 MHz, 65 us duration (1301 samples)
- Detector array: 31x31, 20 mm aperture, 2/3 mm pitch
- Detector size: 2 mm square with 5x5 sub-element integration
- Reconstruction resolution: 500 um
- FFT length: 2048 (zero-padded)

## Expected Performance
- NCC ~ 0.6 (limited by finite aperture and binary vs continuous comparison)
- NRMSE ~ 0.36
- Runtime: < 5 seconds on CPU
