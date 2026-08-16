---
name: oracle-electron-ptychography
description: "Reconstruct the projected phase of gold nanoparticles from a defocused 4D-STEM dataset by iterative single-slice ptychographic phase retrieval, writing a fixed-shape phase array per case."
---

# oracle-electron-ptychography Oracle Skill

This skill guides a pure-numpy/scipy reconstruction of the complex object transmission function from a 4D-STEM (scan x scan x detector x detector) intensity dataset plus a vacuum probe measurement and imaging parameters. The primary deliverable is a real-valued phase image saved as an npz array under the run-local outputs directory. The reconstruction chain covers reading the case inputs, calibrating detector sampling from the vacuum probe, fixing the scan-to-detector coordinate convention (rotation with a 180-degree ambiguity resolved so gold phase is positive, plus transpose), building a defocused complex probe from the vacuum intensity, running mini-batch gradient-descent ptychography on a fine detector-sampled object grid, extracting the object phase, and forcing it to the required output shape. Optional differential-phase-contrast and parallax baselines are included as diagnostics and coordinate-convention cross-checks but do not produce the graded array. Follow the operation sections in order; each is independently removable for ablation.

## Output contract and submission protocol

Produce one submission file per case in NPZ format at the run-local path `outputs/{case_id}.npz` (relative to the run root). Case ids come from the run-local cases list `visible_data/cases.json` (the single public case is `case_000`).

The file MUST contain exactly one required array:

| key | shape | dtype | meaning |
|-----|-------|-------|---------|
| `ptycho_phase` | `(505, 505)` | `float32` or `float64` | reconstructed projected phase image (specimen phase), used by the metrics |

Hard invariants enforced by the grader:
- Shape must be exactly `(505, 505)`. Do not transpose to another size; if your reconstruction grid differs, resample/crop/pad to `(505, 505)` (see the shape-forcing operation).
- All values must be finite (no NaN/Inf). Casting is NOT allowed to fix dtype mismatches, so write a genuine float32/float64 array.
- Save with `numpy.savez` using the exact key name `ptycho_phase`.

Phase sign convention: gold nanoparticles have positive projected electrostatic potential, so the recovered specimen phase over particles must be POSITIVE relative to background. If the map comes out inverted, the coordinate-rotation convention is off by 180 degrees (see the coordinate-convention operation).

Write outputs with run-local `python` from the environment on PATH. Read inputs only from `public/` (or the provided visible case directory) and write only under `outputs/`. Create `outputs/` if missing.

## Case metadata and 4D-STEM array loading

For each case, the input directory contains a raw NPZ data file and a JSON parameters file (per the run-local cases manifest, e.g. `input_data/raw_data.npz` and `params_data.json`).

Load the raw NPZ (`numpy.load`) and read two arrays:
- `datacube`: shape `(1, Rx, Ry, Qx, Qy)` = `(1, 48, 48, 192, 192)`, float32. Drop the leading batch axis to get `(Rx, Ry, Qx, Qy)`.
- `vacuum_probe`: shape `(1, Qx, Qy)` = `(1, 192, 192)`, float32. Drop the leading batch axis to get `(Qx, Qy)`.

Axis meaning: `Rx, Ry` are real-space scan positions; `Qx, Qy` are detector (reciprocal-space) pixels. Total scan positions `J = Rx * Ry = 2304`.

Read the JSON parameters into a dict with keys and this case's values:
- `energy_eV` = 300000 (beam energy, eV)
- `R_pixel_size_A` = 2.39 (real-space scan step, Angstrom)
- `convergence_semiangle_mrad` = 22.6 (probe convergence semi-angle, mrad)
- `scan_shape` = [48, 48]
- `detector_shape` = [192, 192]
- `defocus_A` = 355 (defocus estimate, Angstrom)
- `com_rotation_deg` = 169 (scan-to-detector rotation, degrees, already includes a 180-degree flip)

Keep the datacube as float; only cast per-pattern to float64 where needed for accumulation. Do not assume any other files exist.

## Probe radius/center and detector reciprocal sampling

Estimate the bright-field disk radius and center from the vacuum probe intensity, then derive reciprocal-space sampling. This calibrates the ptychography object grid.

Probe-size algorithm (threshold sweep, from the vacuum probe `(Qx,Qy)`):
1. Let `dp_max = max(probe)`. Sweep `N=100` thresholds linearly in `[thresh_lower=0.01, thresh_upper=0.7]` (note: use upper=0.7 for the vacuum probe).
2. For each threshold `t`, form mask `probe > dp_max * t`, and compute equivalent radius `r = sqrt(count(mask)/pi)`.
3. Take `dr = gradient(r_vals)`; select the stable plateau where `dr <= 0` AND `dr >= 2*median(dr)`; set `probe_radius = mean(r_vals[stable])` (approximately 49.2 px).
4. Center: use mean of `thresh_vals[stable]` as `thresh_mean`, mask `probe > dp_max*thresh_mean`, and compute intensity-weighted center of mass `(qx0, qy0)` over row/col index grids (approximately (96.9, 88.3)).

Reciprocal sampling:
- `dk` [mrad/px] = `convergence_semiangle_mrad / probe_radius` (approximately 22.6/49.2 = 0.459 mrad/px).
- Convert to inverse-Angstrom via the electron wavelength (see the probe-construction operation for the wavelength formula): `dk_invA = dk_mrad * 1e-3 / wavelength_A` (approximately 0.02333 1/A).
- Detector-defined real-space pixel size of the reconstruction grid: `dx_obj_A = 1 / (Q * dk_invA)` with `Q=192` (approximately 0.223 A/px). This fine sampling (not the 2.39 A scan step) sets the ptychographic object resolution.

These values drive scan-position placement on the fine object grid and the final output size.

## Scan-to-detector rotation, transpose, and 180-degree sign resolution

The detector (reciprocal) axes are rotated (and possibly transposed) relative to the scan axes. This convention affects both center-of-mass integration and how measured diffraction is oriented for ptychography.

Use these values for this dataset:
- `com_rotation_deg = 169` degrees. This angle ALREADY includes a 180-degree flip. Physically the raw fit is near -11 degrees; adding 180 gives 169.
- `transpose = False` (do not transpose the detector axes).

180-degree ambiguity rule: the rotation is only determined up to 180 degrees. Resolve it physically: gold has positive projected electrostatic potential, so specimen phase over particles must be POSITIVE. If the reconstructed phase comes out inverted (particles darker/negative relative to background), add 180 degrees to the rotation and redo. With the provided `com_rotation_deg=169`, the correct positive-phase solution is obtained directly.

Rotation application for a center-of-mass vector field `(com_x, com_y)` (used by the DPC baseline):
```
a = deg2rad(com_rotation_deg)
com_rx = cos(a)*com_x - sin(a)*com_y
com_ry = sin(a)*com_x + cos(a)*com_y
```
For ptychography, the same rotation/transpose describe how the measured diffraction relates to scan geometry; the primary solver here consumes rotation/transpose as passthrough metadata and relies on the correct sign convention above.

## Electron wavelength and defocused complex probe construction

Build the initial complex probe (real-space) from the vacuum probe intensity plus the defocus estimate. This is the ptychography probe initialization.

Relativistic electron wavelength (Angstrom), with `E = energy_eV * e`:
```
m0=9.10938e-31; e=1.60218e-19; h=6.62607e-34; c=2.99792e8
lambda_m = h / sqrt(2*m0*E*(1 + E/(2*m0*c**2)))
wavelength_A = lambda_m * 1e10        # ~0.01969 A at 300 keV
```

Probe construction from vacuum intensity `vp` shape `(Qx,Qy)` and `defocus` (Angstrom, positive):
1. Center `vp`: compute intensity-weighted center `(cx,cy)` over integer index grids; Fourier-shift `vp` so the centroid moves to `(Qx/2, Qy/2)` using phase ramp `exp(-2j*pi*(KX*sx + KY*sy))` with `KX,KY = fftfreq(Qx),fftfreq(Qy)` and `sx=Qx/2-cx, sy=Qy/2-cy`; take real part; clip to `>=0`.
2. Diffraction-space amplitude: `probe_amp_k = sqrt(centered)`.
3. Defocus aberration phase over physical spatial frequencies: `kx=fftfreq(Qx, d=R_pixel_size_A)`, `ky=fftfreq(Qy, d=R_pixel_size_A)`, `k2 = KX^2 + KY^2`, `chi = -pi * wavelength_A * defocus * k2`.
4. Fourier-space probe: `probe_k = fftshift(probe_amp_k) * exp(1j*chi)`.
5. Real-space complex probe: `probe = ifft2(probe_k)`; cast to complex128.

Use `defocus = 355` Angstrom for this case (the parallax CTF estimate). Note the sign discipline: the pipeline passes a positive defocus (approximately +355) into this builder.

## Ptychographic forward model (exit wave and far-field intensity)

The measurement model links a complex object `O` and complex probe `P` to diffraction intensities. For scan position `j` mapped to object patch offset `(r,c)`:
```
obj_patch = O[r:r+Qx, c:c+Qy]        # same size as probe/detector
exit_wave = P * obj_patch            # elementwise product
far_field = fft2(exit_wave)          # NO fftshift; use raw fft2
I_j       = |far_field|^2            # predicted detector intensity
```
This is `I_j(k) = |F{P(r-r_j) O(r)}|^2`. Conventions to preserve:
- Probe and object patch are the same `(Qx, Qy) = (192,192)` size.
- Use unnormalized `numpy.fft.fft2` (its default normalization) consistently for forward and inverse transforms.
- Position `(r,c)` is the top-left corner of the patch (not the center); the probe multiplies the patch directly.

Measured amplitude used by the solver is `sqrt(max(intensity, 0))` per pattern. This forward operator is the core of the amplitude-projection update in the solver.

## Object grid sizing and scan-position layout

The complex object array must be large enough that every scan position plus the probe footprint fits inside it. Two grid conventions exist; pick the one consistent with your target output size.

Baseline (unit-step) layout used by the reference solver:
- `Nx = Rx + Qx - 1`, `Ny = Ry + Qy - 1` (i.e. `48+192-1 = 239`).
- Initialize object `O = ones((Nx,Ny), complex128)`.
- Positions: iterate `rx in 0..Rx-1`, `ry in 0..Ry-1` in row-major order; position `j` extracts `O[rx:rx+Qx, ry:ry+Qy]`. Here scan positions are spaced by 1 object pixel.

Physically-scaled layout (matches the required 505x505 output resolution):
- Object pixels are the fine detector-defined `dx_obj_A` (~0.223 A). Scan positions are spaced by `step_px = R_pixel_size_A / dx_obj_A` (~10.7 object px), so the scan spans `~ (Rx-1)*step_px` object px.
- Size the object so the reconstructed field-of-view, after cropping the probe-width border, can be resampled to `(505,505)`. A grid of about `Rx*step_px + Qx` (~706) covering the full scan plus probe margin is appropriate; positions become `round(rx*step_px)` and `round(ry*step_px)`.

Either layout is valid internally as long as the final phase is forced to `(505,505)` (see the shape-forcing operation). Keep positions as integer top-left offsets and ensure `max offset + Q <= grid size`.

## Mini-batch gradient-descent single-slice ptychography

Primary reconstruction. Jointly refine complex object `O` and complex probe `P` by enforcing consistency with measured amplitudes.

Setup:
- `J = Rx*Ry`; `batch_size = max(1, J // batch_fraction)` with `batch_fraction = 4` (batch approximately 576).
- Measured amplitudes: reshape datacube to `(J,Qx,Qy)`, optionally transpose detector axes if `transpose=True` (here False), then `measured_amp = sqrt(max(I,0))`. `mean_intensity = mean(I)`.
- Normalize the initial probe to the data: `probe *= sqrt(mean_intensity / mean(|fft2(probe)|^2))`.
- Initialize object to ones (complex128).
- `eps = 1e-16`.

Iterate `max_iter = 10` epochs, `step_size = 0.5`:
```
for it in range(max_iter):
    order = rng.permutation(J)
    for each contiguous batch of `order`:
        zero obj_grad, probe_grad, obj_weight, probe_weight
        for idx in batch:
            r,c = positions[idx]; patch = O[r:r+Qx, c:c+Qy]
            exit = P*patch; ff = fft2(exit)
            err += sum((measured_amp[idx] - |ff|)^2)
            ff_mod = measured_amp[idx] * exp(1j*angle(ff))   # amplitude projection
            delta = ifft2(ff_mod) - exit
            obj_grad[r:r+Qx,c:c+Qy]   += conj(P)*delta
            obj_weight[r:r+Qx,c:c+Qy] += |P|^2
            probe_grad   += conj(patch)*delta
            probe_weight += |patch|^2
        O += step_size * obj_grad   / sqrt(eps + obj_weight^2)
        P += step_size * probe_grad / sqrt(eps + probe_weight^2)
    nmse = err / (mean_intensity * J * Qx * Qy)
```
Note the denominators use `sqrt(eps + weight^2)` (weight squared under the root), exactly as specified. Use a seeded `numpy.random.default_rng` for reproducibility if desired. Track `nmse` per epoch as the error history. After iterations, extract the object phase (next operations).

## Object phase extraction and border crop

After the solver, convert the complex object to a phase image and remove the unreliable probe-width border.

1. Crop a border of `border = Qx // 4` (=48) pixels from each side of the object array: `obj_cropped = O[border : Nx-border, border : Ny-border]` (guard against negative indices).
2. Phase image: `phase = angle(obj_cropped)` (radians, wrapped to (-pi, pi]).

The cropped-and-phased array is the reconstruction estimate. Do not subtract a global offset or invert sign here; sign correctness is governed by the coordinate-convention operation (gold phase must be positive). The result is then resampled/cropped/padded to the required `(505,505)` output shape.

## Force to 505x505 and write submission NPZ

The grader requires exactly `(505, 505)`. Reconciling your reconstruction grid to this shape is mandatory.

Rule: center-crop or center-pad each axis to 505.
- If an axis is larger than 505, take the centered 505-wide slice.
- If smaller, place it centered inside a zero-filled 505 axis.
- If your object grid was the fine physically-scaled layout (~706), a centered crop to 505 keeps the well-sampled scan field of view.

Sanitize and write:
1. Replace any non-finite values with 0 (`nan_to_num`) so the finite-only check passes.
2. Cast to `float32` (allowed dtype; do NOT rely on grader casting).
3. Ensure C-contiguous, shape `(505,505)`.
4. `numpy.savez('outputs/{case_id}.npz', ptycho_phase=arr)` (create `outputs/` first).

Use the bundled helper to force the shape deterministically.

## DPC baseline via CoM Fourier integration

## DPC baseline via CoM Fourier integration (diagnostic)

Differential phase contrast is a fast baseline and a cross-check that the rotation/sign convention yields positive gold phase. It is NOT the graded output.

Steps:
1. Bright-field mask: `dp_mean = mean(datacube, axis=(0,1))`; `bf_mask = dp_mean > 0.8 * ` ... here the reference thresholds `dp_mean > 0.8` directly (data pre-scaled), producing the BF disk mask `(Qx,Qy)`.
2. Center of mass per scan position within the mask: over index grids `QX,QY`, `com_x = sum(dp*mask*QX)/sum(dp*mask)`, `com_y = sum(dp*mask*QY)/sum(dp*mask)`. Subtract the field mean from each.
3. Rotate CoM into scan frame using `com_rotation_deg` (see coordinate-convention operation).
4. Integrate the rotated gradient field to phase using iterative Fourier integration: `k`-space operators `-1j*0.25*KX/k2` and `-1j*0.25*KY/k2` (with `k2[0,0]=inf`), step-size start 0.5 with halving when residual increases, `max_iter=64`, `stopping_criterion=1e-6`, spacing `dx=dy=R_pixel_size_A`.

Result is a `(Rx,Ry)=(48,48)` phase map. Use only for diagnostics/plots.

## Parallax alignment and defocus/rotation estimate

## Parallax alignment and defocus/rotation estimate (diagnostic)

Parallax gives an intermediate phase image and, more usefully, a data-driven defocus/rotation/transpose estimate that can seed ptychography. It is NOT the graded output.

Steps:
1. BF pixels: `dp_mean = mean(datacube,axis=(0,1))`; `bf_mask = dp_mean > 0.6*dp_mean.max()`; `bf_pixels = argwhere(bf_mask)`.
2. Reference virtual image = scan image at the BF pixel closest to the BF centroid `(qx0,qy0)`.
3. For each BF detector pixel, form the virtual image `datacube[:,:,qxi,qyi]` and estimate integer shift vs reference by phase cross-correlation (`ifft2(F1*conj(F2))`, peak with wrap-around to signed shift).
4. Aligned sum: roll each virtual image by its negative shift and average -> approximate parallax phase.
5. Fit affine map `M` from BF-pixel offsets `dq = bf_pixels - (qx0,qy0)` to shifts via least squares (`Mx,My`). Polar-decompose `M = R*S`: `rotation_rad = -arctan2(R[1,0],R[0,0])`; `C1 = mean(diag(S)) * R_pixel_size_A`.
6. Defocus for ptychography: `defocus = -C1 if abs(C1) > 1 else defocus_A` (falls back to metadata `defocus_A=355`). Pass `rotation_Q_to_R_rads = deg2rad(com_rotation_deg)` and `transpose=False` through.

The defocus (~355 A), rotation (169 deg), and transpose (False) feed the probe builder and solver.

## Fast pipeline smoke probe

## Fast pipeline smoke probe (diagnostic)

Before the full 10-iteration reconstruction (which is O(J*Qx*Qy) per epoch and can be slow), run a cheap sanity probe to catch shape/convention errors early:
- Confirm datacube loads as `(48,48,192,192)` and probe as `(192,192)` after dropping batch dims.
- Build the complex probe and verify `|probe|` is finite and non-zero.
- Run a single mini-batch of the forward+amplitude-projection update on a small random subset (e.g. 32 positions, 1 iteration) and confirm the NMSE is finite and decreases relative to the ones-initialized object.
- Verify the extracted phase can be forced to `(505,505)` and is finite.

Use run-local `python`. This probe is non-mutating (do not overwrite the real submission). Keep runtime to a few seconds; it exists to fail fast, not to reconstruct.
