---
name: oracle-ct-fan-beam
description: "Reconstruct a 2D attenuation image from fan-beam CT sinograms (full-scan and short-scan divergent geometry) and write it as outputs/{case_id}.npz with key phantom of shape (128,128). Covers fan-beam geometry conventions, distance-weighted filtered back-projection with Parker weighting for short-scan, and TV-regularized Chambolle-Pock iterative reconstruction."
---

# oracle-ct-fan-beam Oracle Skill

This skill reconstructs the attenuation map f(x,y) from divergent fan-beam CT sinograms. It provides the geometry conventions (source-to-isocenter D_sd, isocenter-to-detector D_dd, centered detector positions, centered pixel grid, and the rotated coordinates t and s), the fan-beam forward and adjoint operators with magnification mapping and 1/U^2 distance weighting, and two reconstruction routes: analytical filtered back-projection (pre-weighting, ramp filtering, distance-weighted back-projection, plus Parker weighting for short-scan) and TV-regularized Chambolle-Pock (PDHG) iterative reconstruction seeded by FBP. It also covers final packaging to the required NPZ schema, submission validation, and a geometry smoke probe. Each capability is an independently removable section so variants can be tested. Use the run-local python interpreter and only run-local public/, workspace/, and outputs/ paths.

<!-- ORACLE_OP_START op_010_output_contract -->
## Output contract and submission schema

## Output contract and submission schema

Reconstruct one 2D attenuation image per case and write it to a run-local NPZ file.

Hard invariants (must all hold or the submission is rejected):
- Write to `outputs/{case_id}.npz` relative to the run root. For the single public case the id is `case_000`, so the file is `outputs/case_000.npz`.
- The NPZ must contain array key `phantom` with shape exactly `(128, 128)`.
- dtype must be `float32` or `float64`. No implicit casting is allowed, so create the array in one of those dtypes explicitly.
- All values must be finite (no NaN/Inf). Apply a non-negativity clip and replace any non-finite values with 0 before writing.
- The image is the reconstructed attenuation map `f(x, y)` (the phantom), not a sinogram.

Recommended save call:
```python
import numpy as np, os
os.makedirs('outputs', exist_ok=True)
np.savez('outputs/case_000.npz', phantom=recon.astype(np.float32))
```

There is exactly one visible case; iterate over the case list if more are present, always keying the output filename by the case id. Do not add extra required arrays; only `phantom` is scored.
<!-- ORACLE_OP_END op_010_output_contract -->

<!-- ORACLE_OP_START op_020_load_inputs -->
## Load sinograms, angles, detector positions, and params

## Load sinograms, angles, detector positions, and params

For each case, read two files from the case input directory under `public/` (or the provided run-local case path):
- `input_data/raw_data.npz` with keys and shapes:
  - `sino_full` float32 `(1, 180, 192)` — full-scan (360 deg) noisy fan-beam sinogram
  - `sino_short` float32 `(1, 116, 192)` — short-scan noisy sinogram
  - `angles_full` float32 `(180,)` — full-scan projection angles in radians
  - `angles_short` float32 `(116,)` — short-scan projection angles in radians
  - `det_pos` float32 `(192,)` — detector element positions (pixels, centered)
- `params_data.json` with keys: `image_size` (128), `n_det` (192), `n_angles_full` (180), `n_angles_short` (116), `source_to_isocenter_pixels` (256.0), `isocenter_to_detector_pixels` (256.0), `fan_half_angle_deg`, `short_scan_range_deg`, `noise_sigma_relative` (0.02), `detector_type` ("flat").

Procedure:
1. `np.load` the npz and cast every array to `float64` for numerical work.
2. Remove the leading batch dimension of each sinogram: `sino = sino[0]`, giving `(n_angles, n_det)` = `(180,192)` and `(116,192)`.
3. Read params JSON into a dict; extract `N`, `D_sd`, `D_dd`, `n_det`.

Sinogram axis order is `(angle, detector)`. Preserve this ordering throughout; do not transpose.
<!-- ORACLE_OP_END op_020_load_inputs -->

<!-- ORACLE_OP_START op_030_fan_beam_geometry -->
## Fan-beam acquisition geometry parameters

## Fan-beam acquisition geometry parameters

Build a geometry description reused by the forward, back-projection, and FBP steps. Flat linear detector; point source rotates on a circle of radius `D_sd` (source-to-isocenter); detector is opposite the source at distance `D_sd + D_dd` from the source.

Constants for the public case: `N=128`, `n_det=192`, `D_sd=256.0`, `D_dd=256.0`.

Compute (verified to reproduce the provided arrays):
```python
import numpy as np
det_width   = 2.0 * (D_sd + D_dd) * np.tan(np.arctan(N / 2 / D_sd))   # = 256.0
det_spacing = det_width / n_det                                       # = 1.3333...
det_pos     = (np.arange(n_det) - (n_det - 1) / 2.0) * det_spacing    # centered
angles      = np.linspace(0, angle_range, n_angles, endpoint=False)
```
- Full-scan: `angle_range = 2*np.pi`, `n_angles = 180`.
- Short-scan: `angle_range = short_scan_range_deg * np.pi/180`, `n_angles = 116`.

Store in a dict `geo` with keys `N, n_det, n_angles, D_sd, D_dd, det_spacing, det_pos, angles, angle_range`. You may instead reuse the loaded `det_pos`/`angles` arrays directly — they match these formulas to <1e-3.

Pixel grid convention (used everywhere): centered coordinates `x = np.arange(N) - N/2 + 0.5`, same for `y`, then `xx, yy = np.meshgrid(x, y)`. Rotated coordinates for source angle `theta`: `t = x*cos + y*sin` (along source-detector line), `s = -x*sin + y*cos` (perpendicular).
<!-- ORACLE_OP_END op_030_fan_beam_geometry -->

<!-- ORACLE_OP_START op_040_forward_projection -->
## Fan-beam forward projection operator (pixel-driven)

## Fan-beam forward projection operator (pixel-driven)

Maps an image `(N,N)` to a sinogram `(n_angles, n_det)`. Fully vectorized over pixels, loops over angles only. Required by the iterative solver as operator `A`.

Algorithm per angle `theta`:
1. Rotate the centered pixel grid: `t = xx*cos_t + yy*sin_t`, `s = -xx*sin_t + yy*cos_t` (flattened).
2. Source-to-pixel axial distance: `U = D_sd - s`.
3. Magnified detector coordinate: `det_coord = t * (D_sd + D_dd) / U`.
4. Fractional detector index: `det_idx = (det_coord - det_min) / det_spacing`, where `det_min = det_pos[0]`, `det_spacing = det_pos[1]-det_pos[0]`.
5. Validity mask: `(det_idx >= 0) & (det_idx < n_det-1) & (U > 0)`.
6. Ray-divergence weight: `weight = (D_sd + D_dd) / U`.
7. Linear splat each valid pixel value into its two neighboring detector bins:
   - `idx0 = floor(det_idx)`, `frac = det_idx - idx0`.
   - `np.add.at(sino[i], idx0, (1-frac)*img*weight)`
   - `np.add.at(sino[i], idx0+1, frac*img*weight)`

Accumulate into `sinogram` of dtype float64, shape `(n_angles, n_det)`; pixel size = 1 (no extra scale).
<!-- ORACLE_OP_END op_040_forward_projection -->

<!-- ORACLE_OP_START op_050_backprojection -->
## Fan-beam back-projection (adjoint) with 1/U^2 weighting

## Fan-beam back-projection (adjoint) with 1/U^2 weighting

Maps a sinogram `(n_angles, n_det)` to an image `(N,N)`. Used both as the FBP back-projection stage and as the adjoint operator `A^T` in the iterative solver.

Per angle `theta` over the centered pixel grid `(xx, yy)`:
1. `t = xx*cos_t + yy*sin_t`, `s = -xx*sin_t + yy*cos_t`.
2. `U = D_sd - s` (source-to-pixel axial distance).
3. Magnified detector coordinate: `det_coord = t * (D_sd + D_dd) / U`.
4. Distance weight: `weight = D_sd**2 / U**2` (normalized by `D_sd^2`).
5. Interpolate the sinogram row onto `det_coord` with linear interpolation, out-of-range filled with 0: `scipy.interpolate.interp1d(det_pos, sino[i], kind='linear', bounds_error=False, fill_value=0.0)`.
6. Accumulate: `image += weight * f(det_coord) * d_angle`, where `d_angle = angles[1]-angles[0]` (1.0 if a single angle).

The `1/U^2` distance weighting and the same magnification mapping as the forward operator are essential — they encode the divergent fan geometry.
<!-- ORACLE_OP_END op_050_backprojection -->

<!-- ORACLE_OP_START op_060_ramp_filter -->
## Ramp filter construction with windowing

## Ramp filter construction with windowing

Build the frequency-domain ramp filter used to filter each pre-weighted sinogram row in FBP. Default is a Hann window with `cutoff=0.3`.

Steps:
```python
pad_len = int(2 ** np.ceil(np.log2(2 * (n_det - 1))))   # = 512 for n_det=192
N_filt = pad_len + 1
x = np.arange(N_filt) - (N_filt - 1) / 2.0
h = np.zeros(N_filt)
h[x == 0] = 1 / (8 * d**2)                     # d = effective detector spacing
odds = (x % 2 == 1)
h[odds] = -0.5 / (np.pi * d * x[odds])**2
h = h[:-1]
filt = np.abs(np.fft.fftshift(np.fft.fft(h))) * 2
w = 2 * np.pi * (np.arange(pad_len) - pad_len/2) / pad_len   # centered
```
Apply the window (multiply `filt` by `np.fft.fftshift(window)`):
- ram-lak: no window
- shepp-logan: `sinc(w/(2*pi*cutoff))`
- cosine: `cos(w/(2*cutoff))`
- hamming: `0.54 + 0.46*cos(w/cutoff)`
- hann: `0.5 + 0.5*cos(w/cutoff)`

Then zero out frequencies past the cutoff: `filt[fftshift(|w| > pi*cutoff)] = 0`.

Return `(filt, pad_len)`. The spacing `d` passed here is the effective spacing at the isocenter plane: `det_spacing * D_sd / (D_sd + D_dd)` (= 0.6667). The filter models the ramp impulse response `h[0]=1/(8d^2)`, `h[n]=-1/(2*pi*d*n)^2` for odd n, 0 for even n.
<!-- ORACLE_OP_END op_060_ramp_filter -->

<!-- ORACLE_OP_START op_070_parker_weights -->
## Parker weighting for short-scan redundancy

## Parker weighting for short-scan redundancy

Short-scan fan-beam covers `delta = pi + 2*gamma_max` and has partially redundant rays at the two angular edges. Parker weights smoothly ramp these contributions to 1 in the interior and to a sin^2 S-curve at the edges. Apply only to the short-scan sinogram; full-scan uses uniform weights.

Compute per detector element and angle:
```python
gamma = np.arctan(det_pos / D_sd)      # fan angle per detector element
gamma_max = np.max(np.abs(gamma))
epsilon = gamma_max                    # transition width
delta = angles[-1] - angles[0] + (angles[1]-angles[0])   # total sweep
```
For each angle offset `beta = angles[i] - angles[0]` and element fan angle `g`:
- If `beta < 2*(epsilon + g)`: `w = sin(pi/2 * clip(beta / (2*(epsilon+g)), 0, 1))**2` (or 0 if `epsilon+g <= 0`).
- Elif `beta > delta - 2*(epsilon - g)`: with `denom = 2*(epsilon - g)`, `w = sin(pi/2 * clip((delta-beta)/denom, 0,1))**2` (or 0 if `denom <= 0`).
- Else: `w = 1`.

Result is `weights` of shape `(n_angles, n_det)`; multiply the short-scan sinogram elementwise by it before pre-weighting. Removing this step reintroduces short-scan streak/redundancy artifacts.
<!-- ORACLE_OP_END op_070_parker_weights -->

<!-- ORACLE_OP_START op_080_fan_beam_fbp -->
## Fan-beam FBP reconstruction pipeline

## Fan-beam FBP reconstruction pipeline

Analytical reconstruction combining pre-weighting, ramp filtering, and distance-weighted back-projection. Call with `filter_type='hann', cutoff=0.3`; set `short_scan=True` for the short-scan geometry, `False` for full-scan.

Steps:
1. Copy the sinogram. If `short_scan`, multiply by the Parker weights.
2. Effective detector position projected to the isocenter plane: `gamma_iso = det_pos * D_sd / (D_sd + D_dd)`.
   Pre-weight each row: `sino *= (D_sd / np.sqrt(D_sd**2 + gamma_iso**2))[None, :]`.
3. Build the ramp filter with effective spacing `eff_spacing = det_spacing * D_sd / (D_sd + D_dd)` (=0.6667). For each angle row:
   ```python
   padded = np.fft.fft(sino[i], pad_len)
   row = np.real(np.fft.ifft(np.fft.ifftshift(filt * np.fft.fftshift(padded))))
   filtered[i] = row[:n_det]
   ```
4. Back-project the filtered sinogram using the `1/U^2` distance-weighted back-projection with the magnification mapping (same geometry as the adjoint operator).

After reconstruction apply `recon = np.maximum(recon, 0)`.

The key fan-beam-specific ingredients (vs parallel-beam) are: pre-weighting `D_sd/sqrt(D_sd^2+gamma_iso^2)`, the `1/U^2` back-projection weight, and the magnification mapping `t*(D_sd+D_dd)/U`.
<!-- ORACLE_OP_END op_080_fan_beam_fbp -->

<!-- ORACLE_OP_START op_090_tv_pdhg -->
## TV-regularized reconstruction via Chambolle-Pock (PDHG)

## TV-regularized reconstruction via Chambolle-Pock (PDHG)

Solves `min_x (1/2)||A x - b||^2 + lam * TV(x)` on the short-scan sinogram, where `A` is the fan-beam forward operator and `A^T` its adjoint. This is the highest-accuracy method and is preferred as the submitted result.

Parameters: `lam=0.005`, `n_iter=150`, `tau=0.005`, `sigma_dual=0.005`, `theta=1.0`, `positivity=True`.

Initialization: `x = fan_beam_fbp(sino, geo, 'hann', 0.3, short_scan=(angle_range < 1.9*pi))`, then `x = max(x, 0)`. Set `x_bar = x.copy()`; dual variables `p = zeros((2,N,N))`, `q = zeros_like(sino)`.

Helpers (forward differences):
- Gradient `grad(x) -> (2,N,N)`: `gx[:, :-1] = x[:,1:]-x[:,:-1]`; `gy[:-1,:] = x[1:,:]-x[:-1,:]`.
- Divergence (negative adjoint): `dx[:,1:-1]=px[:,1:-1]-px[:,:-2]; dx[:,0]=px[:,0]; dx[:,-1]=-px[:,-2]` (and analogously for `dy`); return `dx+dy`.
- Isotropic TV prox on gradient dual: `norm=sqrt(p0^2+p1^2)`, `norm=max(norm,1e-10)`, `factor=max(norm, sigma)/norm`, `p/=factor`.

Each iteration:
```
Ax_bar = A(x_bar);  q = (q + sigma_dual*(Ax_bar - b)) / (1 + sigma_dual)
p = prox_l1( p + sigma_dual*grad(x_bar), lam )
x_old = x
x = x - tau*(A^T(q) - div(p))
if positivity: x = max(x, 0)
x_bar = x + theta*(x - x_old)
```
Optionally track `loss = 0.5*sum((A(x)-b)^2)`. Return the final `x`.
<!-- ORACLE_OP_END op_090_tv_pdhg -->

<!-- ORACLE_OP_START op_100_write_output -->
## Select best reconstruction and write submission

## Select best reconstruction and write submission

Produce the final `(128,128)` image and write it to `outputs/{case_id}.npz` under key `phantom`.

Selection policy (matches the reference which reports TV short-scan as best):
- Prefer the TV-PDHG short-scan reconstruction as the submitted `phantom`.
- If TV is unavailable, fall back to the short-scan FBP (with Parker weighting).

Final cleanup before writing:
```python
recon = np.maximum(recon, 0.0)          # non-negativity
recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
recon = recon.astype(np.float32)         # or float64; do not rely on implicit cast
np.savez('outputs/case_000.npz', phantom=recon)
```
Do not center-crop or resize the submitted array — it must stay `(128,128)` on the full grid. (Cropping/normalization is only used internally for scoring comparisons, not for the saved output.)
<!-- ORACLE_OP_END op_100_write_output -->

<!-- ORACLE_OP_START op_110_validate_submission -->
## Validate submission shape, dtype, and finiteness

## Validate submission shape, dtype, and finiteness

Before finishing, re-open each written file and assert the hard invariants so a malformed submission is caught early.

Checks:
- File exists at `outputs/{case_id}.npz`.
- Contains key `phantom`.
- `phantom.shape == (128, 128)`.
- `phantom.dtype` is `float32` or `float64`.
- `np.all(np.isfinite(phantom))` is True.

A reusable checker script is bundled; run it from the run root after writing outputs. If any assertion fails, fix the reconstruction/writing step rather than editing the check.
<!-- ORACLE_OP_END op_110_validate_submission -->

<!-- ORACLE_OP_START op_120_geometry_probe -->
## Geometry consistency smoke probe

## Geometry consistency smoke probe

Cheap non-mutating diagnostic to confirm the geometry conventions before running the full reconstruction. It recomputes `det_pos` and the angle grids from params and compares them against the arrays in `raw_data.npz`.

Expected results for the public case: recomputed `det_pos` matches the provided `det_pos` (atol 1e-3), `angles_full` matches a `linspace(0, 2*pi, 180, endpoint=False)`, and `angles_short` matches `linspace(0, short_scan_range_deg*pi/180, 116, endpoint=False)`. `det_width=256.0`, `det_spacing=1.3333`, `eff_spacing=0.6667`, `pad_len=512`.

Run the bundled probe from the run root with the run-local `python`. If a mismatch is reported, revisit the geometry conventions before proceeding.
<!-- ORACLE_OP_END op_120_geometry_probe -->
