---
name: oracle-light-field-microscope
description: "Reconstruct target images from single-shot light-field microscope sensor captures by building a wave-optics forward model (Debye objective PSF, microlens phase mask, Rayleigh-Sommerfeld propagation) and solving with 3D Richardson-Lucy deconvolution."
---

# oracle-light-field-microscope Oracle Skill

This skill reconstructs fluorescence target images from single 2D light-field sensor captures. Each case provides four noisy sensor images (each 450x450) taken at distinct axial defocus depths, plus the reconstruction depth grid. For each image, build a wave-optics point-spread-function bank over the case's depth grid, assemble a shift-variant forward projector and its adjoint, then run a fixed number of multiplicative Richardson-Lucy updates to estimate a fluorescence volume; the reconstructed target is the single reconstructed depth slice. Stack the four reconstructed images and write them to the required output archive. Each operation below is independently removable so its contribution can be ablated; enable all operations for full-fidelity reconstruction.

## Submission contract and case protocol

Produce exactly one output archive per case at the run-local path `outputs/{case_id}.npz` (base = run root). For the single public case this is `outputs/case_000.npz`.

The archive MUST contain one required array:

- key `rl_reconstructions`, shape `(4, 450, 450)`, dtype `float32` or `float64`.

This array holds the four estimated reconstructed target images, one per input sensor capture, in the same order as the input observations. All values MUST be finite (no NaN/Inf); casting is not allowed, so write the array already in an allowed dtype.

Discover cases from `visible_data/cases.json` (a `cases` list with `id`, `input_dir`, `params` path). For each case, read its inputs (see the data-loading step), reconstruct each of the four observations, stack the four reconstructed images along axis 0, and save. Create the `outputs/` directory if missing. Do not write to any directory other than `outputs/`, `workspace/`, or `logs/agent/`.

Interpretation: each of the four images corresponds to one occupied target depth; the benchmark treats each reconstruction as a thickness-1 volume, so the reconstructed target image is a single axial slice of the reconstructed volume.

## Load sensor captures, depth grid, and microscope parameters

For each case, read two files under the case `input_dir` and `params` path.

1. `input_data/raw_data.npz` contains:
   - `light_field_observations`: shape `(1, 4, 450, 450)`, `float32`. Sensor irradiance images. Index `[0]` to get the four `(450, 450)` images; process each of the 4 independently.
   - `reconstruction_depths_um`: shape `(1, 4, 1)`, `float32`. The reconstruction depth grid for each image. Index `[0, i, :]` to get image `i`'s depth grid; for the benchmark each is a singleton axial slice (e.g. `[0]`, `[-30]`, `[-50]`, `[-70]` um). The depth grid is what defines the axial planes to reconstruct for that image.

2. `params_data.json` provides configuration. Read as JSON. Key blocks and values:
   - `microscope`: `M=20` (magnification), `NA=0.5` (numerical aperture), `ftl=165000` um (tube-lens focal length), `WaveLength=0.525` um (emission wavelength). Refractive index `n=1.0` (use `1.0` if absent).
   - `mla`: `lensPitch=150` um, `fm=3000` um (microlens focal length), `pixelPitch=6.5` um, `gridType='reg'`, `focus='single'`, `plenoptic=1`, `uLensMask=1`, `n_lenslets=15`.
   - `reconstruction`: `newSpacingPx=15` (sensor pixels per lenslet), `depthStep=10` um.
   - `usaf_data`: `fieldOfViewScale=2.0` (field-of-view expansion of the base lenslet count).

Use the run-local Python interpreter (`python`, resolved from `<run_context>/PATH` or `public/envs/runtime/.venv-posix/bin/python`) with `numpy`, `scipy`, and `pyyaml` available. Load numeric params as `float`; do not assume solver constants live in params (they are fixed separately).

Each image's reconstruction uses that image's own depth grid; per-image the forward model geometry is identical (same microscope/MLA), only the depth grid (and thus PSF stack) differs.

## Derive camera geometry constants from raw parameters

From the raw microscope/MLA parameters, derive the following constants (all lengths in um, angles in radians). These feed every downstream optics computation.

- `spacing_px = lensPitch / pixelPitch`
- `fobj = ftl / M`  (objective focal length; e.g. 8250)
- `delta_ot = ftl + fobj`  (object-to-tube distance)
- `wave_number k = 2*pi*n / WaveLength`  (e.g. 11.96797)
- `objective_radius = fobj * NA`  (e.g. 4125)
- `tube_radius = objective_radius`
- `dof = fobj`
- `offset_fobj = dof - fobj`  (equals 0 here)
- `tube_to_mla = ftl` (use `ftl` when the configured `tube2mla` is 0)
- `mla_to_sensor = fm` (use `fm` when the configured `mla2sensor` is 0)
- `ulens_radius = tube_radius * mla_to_sensor / tube_to_mla`  (e.g. 75)
- `new_pixel_pitch = lensPitch / newSpacingPx`  (e.g. 10)
- `mla_magnification = M`

This compact model is fixed to a regular-grid, single-focus, plenoptic-1 microlens array (`gridType='reg'`, `focus='single'`, `plenoptic=1`). See the bundled resource for the exact formula list and units.

## Sensor/object sampling grid and PSF field-of-view sizing

Build the sampling geometry used to lay out PSF kernels and lenslets.

Let `ns = newSpacingPx` (15).
- `sensor_res = (lensPitch/ns, lensPitch/ns)` = (10, 10) um per pixel.
- `nnum = ns + (1 - ns % 2)` (force odd; 15 stays 15). `tex_nnum = nnum`.
- Object-space resolution `tex_res = (sensor_res[0]/(1*M), sensor_res[1]/(1*M), depthStep)` = (0.5, 0.5, 10) um.
- `depths = arange(depth_range[0], depth_range[1] + 0.5*depthStep, depthStep)` for the image's depth grid (singleton for the benchmark).

PSF field-of-view (lenslets), from the most-defocused depth: with `max_depth = depths[argmax(|depths+offset_fobj|)] + offset_fobj`,
- `zobj = fobj - (max_depth - offset_fobj)`; if `zobj` is close to `fobj` or `dof`, nudge `zobj += 1e-5*fobj`.
- `z1 = zobj*fobj/(zobj - fobj)`
- `tube_r = objective_radius*delta_ot*|1/z1 - 1/delta_ot|`
- `z2 = ftl*(delta_ot - z1)/(delta_ot - z1 - ftl)`
- `blur_r = tube_r*tube_to_mla*|1/z2 - 1/tube_to_mla|`
- `psf_size_lenslets = ceil(blur_r / lensPitch) + 2`

Sampling axes (integer index grids scaled by `sensor_res`):
- `img_half = max(nnum*psf_size_lenslets, 2*nnum)`
- `yspace = xspace = sensor_res * arange(-img_half, img_half+1)` (length `2*img_half+1`).
- `y_ml_space = x_ml_space = sensor_res * arange(-(nnum//2), nnum//2 + 1)` (per-lenslet local grid, length `nnum`).

Also build a lenslet local mask `tex_mask` over the `nnum x nnum` local grid: a disk `sqrt(yy^2+xx^2) < tex_nnum*tex_res/2` after the regularization pass that removes/fills isolated cells so exactly the expected on-grid pattern remains. A `sens_mask` disk of radius `ulens_radius` is used for the microlens aperture. See the resource for the FOV sizing table and mask rules.

## Debye diffraction objective PSF at defocus

Compute the complex pre-MLA field for a point source at axial position `depth_um` (already offset by `offset_fobj` when called from kernel construction). This is the Debye/Wolf integral for an aplanatic objective.

Setup:
- `alpha = arcsin(NA/n)`; `sin_alpha_half_sq = sin(alpha/2)^2`.
- `demag = 1/M`; `d1 = dof - depth_um`.
- `u = 4*k*depth_um*sin_alpha_half_sq` (Lommel defocus parameter).
- Complex prefactor `koi = demag/((d1*WaveLength)^2) * exp(-1j*u/(4*sin_alpha_half_sq))`.

Transverse grid: `yy, xx = meshgrid(yspace, xspace, indexing='ij')`; object-plane radius `radial = sqrt(yy^2+xx^2)/M`; `v = k*radial*sin(alpha)`.

Integrate over `theta in linspace(0, alpha, theta_samples)` (theta_samples = 128):
- `phase = exp(1j*(u/2)*sin(theta/2)^2/sin_alpha_half_sq)`
- `prefactor = sqrt(cos(theta)) * (1+cos(theta)) * sin(theta)`  (aplanatic apodization x sin weight)
- `bessel_arg = (sin(theta)/sin(alpha)) * v`
- `integrand = prefactor * phase * J0(bessel_arg)` (J0 = zeroth-order Bessel, `scipy.special.j0`)
- Integrate along theta with the trapezoid rule: `integral = trapz(integrand, x=theta, axis=0)`.
- Field `U_mla = koi * integral` (complex, shape of the transverse grid).

At `depth_um = 0` the field is compact (Airy-like); larger `|depth_um|` broadens it into a defocus ring spanning more lenslets. This same routine is reused for the conventional baseline. See the resource for the exact vectorized form.

## Microlens array phase mask and tiling

Build the tiled microlens transmittance that multiplies the pre-MLA field.

Single-lenslet phase pattern over the local grid `(y_ml_space, x_ml_space)`:
- `yy, xx = meshgrid(y_ml_space, x_ml_space, indexing='ij')`
- `phase = exp(-1j * k/(2*fm) * (yy^2 + xx^2))`  (thin converging lens, focal length `fm`).
- Apply the aperture: if `uLensMask == 1`, keep `phase` only where `sens_mask` is true (else 0); otherwise keep where `sqrt(yy^2+xx^2) < lensPitch/2 - 3` (else 0).

Tile the lenslet pattern onto a full canvas at each used lenslet center. Lenslet centers are on a regular grid: `coords = arange(n_lenslets)*newSpacingPx`, recentered by subtracting `coords[n_lenslets//2]`; take the `meshgrid` (rows, cols). Restrict to centers within `psf_size_lenslets + 3` of the array center (`select_used_centers`). Stamp the single-lenslet complex phase (added) at each center on a canvas padded by `2*len(y_ml_space)` in each axis, then crop back to the `yspace x xspace` size. This produces `mla_array` (complex), the full MLA transmittance.

The post-MLA field for a source is `U_plus = U_mla_field * mla_array` (elementwise complex product). See the resource for the tiling/crop bookkeeping.

## Rayleigh-Sommerfeld propagation MLA to sensor

Propagate the post-MLA complex field a distance `z = mla_to_sensor` (= `fm`) to the sensor using the angular-spectrum (Rayleigh-Sommerfeld) transfer function.

For a field `field_at_mla` of shape `(ny, nx)` with pixel spacing `sensor_res`:
- `k = 2*pi/WaveLength`.
- Frequency grids `fy = fftfreq(ny, d=sensor_res[0])`, `fx = fftfreq(nx, d=sensor_res[1])`; `fy_grid, fx_grid = meshgrid(fy, fx, indexing='ij')`.
- `argument = (1 - WaveLength^2 * (fy_grid^2 + fx_grid^2))` cast to complex.
- Transfer function `transfer = exp(1j * sqrt(argument) * z * k)`. Casting `argument` to complex makes `sqrt` produce imaginary values for `|k_perp| > k`, which decays (zeros out) evanescent components.
- Output `U_sens = exp(1j*k*z) * ifft2( fft2(field_at_mla) * transfer )`.

Use `scipy.fft` for `fft2`/`ifft2`/`fftfreq`. `U_sens` is the complex sensor field; its squared magnitude becomes the PSF kernel. See the resource for the exact expression.

## Assemble the shift-variant PSF kernel bank

Build the intensity PSF kernels that make the forward operator shift-variant within a lenslet. There is one kernel per (local sub-pixel coordinate, depth).

1. Build `mla_array` (MLA transmittance) once. Build the native PSF stack: for each depth in `depths + offset_fobj`, compute the Debye field over `(yspace, xspace)`.
2. For each local coordinate `(local_row, local_col)` in the `tex_nnum x tex_nnum` grid, with `shift_row = local_row - tex_nnum//2`, `shift_col = local_col - tex_nnum//2`, and for each depth's native PSF field:
   - integer-shift the PSF field by `(shift_row, shift_col)` (zero-fill, no wrap),
   - multiply by `mla_array` -> post-MLA field,
   - propagate to the sensor (RS propagation),
   - integer-shift the sensor field back by `(-shift_row, -shift_col)`,
   - `kernel = |sensor_field|^2`.
3. Clamp and normalize each kernel: if `kernel.max() > 0`, zero all entries `< kernel.max() * kernel_tol` (kernel_tol = 0.005), then divide by the (post-clamp) sum so each kernel sums to 1. Store as `float64`.

The `theta_samples = 128` and `kernel_tol = 0.005` constants are fixed (not in params). Integer shift uses rounded integer pixel offsets with edge clipping. The adjoint kernel for each kernel is its 180-degree flip `kernel[::-1, ::-1]`. See the resource for the full loop and shift helper.

## Shift-variant forward and adjoint (backward) projectors

Map between the fluorescence volume and the sensor image using per-local-coordinate scatter/gather indices plus FFT convolution with the kernel bank.

Geometry: `img_shape = tex_shape = (n_lenslets*newSpacingPx, n_lenslets*newSpacingPx)` (450x450 for this config, `n_lenslets = round(base_n_lenslets * fieldOfViewScale) = round(15*2.0) = 30`).

Build pattern operations: for each local coordinate `(local_row, local_col)` where `tex_mask[local_row, local_col]` is true, precompute flattened index arrays mapping lenslet centers (regular grid) offset by `(local - center)` into both the object (`tex`) and image grids, keeping only in-bounds entries. Each pattern op owns its per-depth `kernels` and 180-degree-flipped `adjoint_kernels`.

Forward project (volume -> image), output shape `img_shape`, init zeros float64:
- For each depth plane and each pattern op: gather `sampled = plane.ravel()[tex_indices]`; skip if all zero; scatter into a sparse image at `img_indices`; `projection += fftconvolve(sparse_image, kernel_for_this_depth, mode='same')`.
- Finally clamp negatives to 0.

Backward project (image -> volume), output shape `tex_shape + (n_depths,)`, init zeros:
- For each depth: for each pattern op, `conv = fftconvolve(projection, adjoint_kernel_for_depth, mode='same')`; accumulate `back_flat[tex_indices] += conv.ravel()[img_indices]`; reshape into the depth slice.
- Finally clamp negatives to 0.

Use `scipy.signal.fftconvolve`. The backward projector is the exact adjoint of the forward projector. See the resource for index construction.

## Richardson-Lucy 3D deconvolution loop

Recover the non-negative volume from one 2D observation with the multiplicative Poisson-likelihood (Richardson-Lucy) update:

`v^(q+1) = v^(q) * H^T( y / (H v^(q) + eps) ) / H^T(1)`

where `H` is the forward projector, `H^T` the backward projector, `y` the observed sensor image.

Procedure (fixed `iterations = 10`, `eps = 1e-8`):
1. Cast `observation` to float64.
2. Initialize `recon = ones(tex_shape + (n_depths,))` (all-ones volume).
3. Precompute the normalizer `norm_bp = backward_project(ones_like(observation))`, then `norm_bp = maximum(norm_bp, eps)`.
4. For each of 10 iterations:
   - `estimate = forward_project(recon)`
   - `ratio = observation / maximum(estimate, eps)`
   - `correction = backward_project(ratio) / norm_bp`
   - set non-finite entries of `correction` to 0 and clamp negatives to 0
   - `recon *= correction`; set non-finite entries of `recon` to 0 and clamp negatives to 0.
5. Return `recon` (shape `(450, 450, n_depths)`).

The reconstructed target image for the case is the axial slice `recon[:, :, 0]` (for the singleton-depth benchmark the volume has one slice, so slice 0 is the reconstruction). Do NOT use geometric shift-and-sum refocusing; the wave-optics forward operator is required. See the resource for exact update code.

## Stack reconstructions and write output archive

After reconstructing all four observations of a case:
1. For each observation `i` (0..3), take the reconstructed target image = the first axial slice `recon_i[:, :, 0]`, shape `(450, 450)`.
2. Stack the four images along axis 0 into `rl_reconstructions` of shape `(4, 450, 450)`.
3. Cast to `float32`, verify all values are finite (replace any non-finite with 0 before writing, since casting/invalid values are rejected).
4. Save with `numpy.savez_compressed` to `outputs/{case_id}.npz` under the key `rl_reconstructions`.

Do not add extra required arrays. Keep the observation ordering identical to the input `light_field_observations` order (matching the input depth grid order). Ensure `outputs/` exists first.

## Conventional-microscope defocus baseline (diagnostic)

A conventional (non-light-field) microscope image at the same defocus provides a comparison baseline. It is NOT the submitted output; use it only as a sanity reference to confirm the wave-model reconstruction improves optical sectioning away from the native plane.

Procedure:
- Compute the Debye field at `depth_um = target_depth_um + offset_fobj` over `(yspace, xspace)`.
- Intensity PSF `psf = |field|^2`; normalize by its sum (if positive).
- Baseline image = `fftconvolve(object_2d, psf, mode='same')` (float64).

The scientific expectation: at `z=0` the conventional image can be competitive; at large defocus (e.g. `-50`, `-70` um) it is strongly blurred while the wave-model RL reconstruction stays sharper. This operation is a diagnostic and can be omitted without affecting the submitted arrays.

## Cheap end-to-end shape and finiteness probe

Before committing to the full kernel build (which is the expensive step), run a fast environment and contract probe to catch mistakes early:
- Confirm the run-local Python can import `numpy`, `scipy.fft`, `scipy.signal`, `scipy.special`, and `yaml`.
- Load `raw_data.npz`; assert `light_field_observations[0]` has shape `(4, 450, 450)` and `reconstruction_depths_um[0]` has shape `(4, 1)`.
- Verify derived `n_lenslets*newSpacingPx == 450` so `img_shape` matches the required output.
- After producing `rl_reconstructions`, assert shape `(4, 450, 450)` and `np.isfinite(...).all()` before writing.

Run the bundled probe with the run-local interpreter. This is a guardrail only; it can be removed without changing outputs.
