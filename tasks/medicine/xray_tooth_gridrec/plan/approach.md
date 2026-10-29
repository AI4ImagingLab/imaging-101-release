# Solution Approach

## Problem Formulation

Recover a 640x640 attenuation image from 181 parallel-beam X-ray projections spanning [0, pi) radians, with 640 detector pixels per projection and 2 axial slices.

## Pipeline

### 1. Preprocessing

- **Flat-field correction:** Average the 10 flat-field and 10 dark-field frames. Compute normalised transmission: `T = (proj - dark_avg) / (flat_avg - dark_avg)`.
- **Log-linearization:** Apply Beer-Lambert law: `sinogram = -ln(T)`. Clip T to [1e-12, inf) before log to avoid numerical issues.

### 2. Rotation Center Estimation

Use cross-correlation of opposing projections:
- The first projection (theta ~ 0) and last projection (theta ~ pi) are approximate mirror pairs.
- Cross-correlate the first projection with the flipped last projection.
- The peak offset gives twice the center displacement from the detector midpoint.
- Refine with a sub-pixel grid search using reconstruction variance as the quality metric.

Expected center: ~296 pixels (offset ~-24 from detector midpoint 319.5).

### 3. Reconstruction (Filtered Back-Projection)

**Ramp filter:**
- Zero-pad each projection to the next power-of-two length (>= 2 * n_det) for efficient FFT.
- Multiply FFT of each projection by the ramp filter |omega| (frequencies from `np.fft.fftfreq`).
- Inverse FFT and truncate back to n_det.

**Back-projection:**
- For each pixel (x, y) in the reconstruction grid, compute the detector coordinate `t = x*cos(theta) + y*sin(theta)` at each angle.
- Linearly interpolate the filtered projection at coordinate t.
- Sum over all angles, scaled by `pi / n_angles`.

**Circular mask:** Zero out pixels outside 95% of the image half-width to remove edge artifacts.

### 4. Evaluation

Compare reconstruction against the gridrec baseline using:
- **NCC** (cosine similarity, no mean subtraction): target >= 0.88
- **NRMSE** (RMS error / dynamic range): target <= 0.032

## Solver Parameters

| Parameter | Value |
|-----------|-------|
| Rotation center initial guess | 290 pixels |
| Center search radius | 30 pixels (coarse), 5 pixels (fine) |
| Center search tolerance | 0.5 pixels |
| Ramp filter | Ram-Lak (|omega|) |
| FFT padding | next power of 2 >= 2 * n_det |
| Circular mask ratio | 0.95 |
