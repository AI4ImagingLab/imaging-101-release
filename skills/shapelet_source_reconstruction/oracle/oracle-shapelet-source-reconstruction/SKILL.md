---
name: oracle-shapelet-source-reconstruction
description: "Reconstruct the intrinsic (unlensed) source-galaxy surface brightness from a noisy, PSF-convolved, gravitationally lensed observation using a Cartesian Hermite shapelet basis and weighted-least-squares inversion of a linear forward model."
---

# oracle-shapelet-source-reconstruction Oracle Skill

This skill reconstructs the intrinsic (unlensed) surface brightness of a galaxy seen through a known gravitational lens. The domain is astronomy: a source galaxy is represented in a Cartesian Hermite shapelet basis, imaged through a linear forward model (backward ray-tracing through the lens, evaluation of the source, Gaussian PSF convolution, and block-average downsampling), and recovered by weighted-least-squares inversion. The workflow covers the public submission contract, per-case loading of parameters and shapelet coefficients, shared grid and array conventions, the shapelet basis, the lens deflection field, PSF convolution, the forward image simulation, the heteroscedastic noise model, response-matrix construction, the weighted-least-squares solve, high-resolution source evaluation and output writing, plus validation and a cheap smoke test. All numerics use only numpy and scipy. Constants, axis and ordering conventions, normalization factors, scale parameters, and the output schema are preserved so the standard behavior can be reconstructed. Run everything from the eval run root with the run-local Python (for example `python` on PATH, or `public/envs/runtime/.venv-posix/bin/python`).

<!-- ORACLE_OP_START op_010_public_contract -->
## Public task contract and submission schema

Produce, for each listed case, the reconstructed intrinsic source image of the galaxy behind the (known) lens.

Contract:
- Case list lives in `public/`; each case has an id (e.g. `case_000`), an `input_data` directory, and a params JSON.
- Write exactly one output file per case to `outputs/{case_id}.npz` relative to the run root.
- Required array key: `source_image`, shape exactly `(320, 320)`, dtype `float32` or `float64`.
- All values must be finite (no NaN/Inf). Casting is not applied for you: write a float array directly.
- `source_image` is the sole array scored by evaluation; it is the high-resolution reconstructed source surface brightness (flux per high-res pixel).

Hard invariants to respect end to end: detector grid is 64x64 at 0.05 arcsec/pixel; the high-resolution (supersampled) grid is 320x320 at 0.01 arcsec/pixel with supersampling factor 5; the output lives on the 320x320 grid.

Do not read outside `public/`, `workspace/`, `outputs/`, `logs/agent/`, or `${CLAUDE_SKILL_DIR}`.
<!-- ORACLE_OP_END op_010_public_contract -->

<!-- ORACLE_OP_START op_020_case_loading -->
## Case metadata and raw array loading

Load per-case inputs from the run-local `public/` tree.

Steps:
1. Read the cases list JSON to enumerate case ids and their `input_dir` / params paths.
2. For each case, read `params_data.json`. Expected keys and values for the public case: `background_rms=10.0`, `exp_time=100.0`, `numPix=64`, `deltaPix=0.05`, `high_res_factor=5`, `fwhm_lensing=0.1`, `fwhm_deconv=0.25`, `n_max_decomp=150`, `beta_decomp=10`, `n_max_recon=20`, `beta_recon_lensing=0.15`, `beta_recon_deconv=0.3`, `source_x=0.2`, `source_y=0.0`, `beta_model_lensing=0.06`, `beta_model_deconv=0.12`, `coeff_scale_deconv=5.0`, `theta_E=0.5`, `gamma=2.0`, `e1=0.0`, `e2=0.0`, `gamma1=0.0`, `gamma2=0.0`.
3. Load `input_data/raw_data.npz`. Keys: `coeff_ngc` shape `(1, 11476)` float32 (shapelet coefficients of the source galaxy at n_max=150, beta=10 px), `ngc_resized` `(1, 264, 264)`, `image_reconstructed` `(1, 264, 264)`.
4. Flatten the coefficient vector: `coeff_ngc = raw['coeff_ngc'].reshape(-1)` giving length 11476. Only `coeff_ngc` is needed downstream; the two image arrays are informational.

These loaded params/coefficients feed the forward simulation and reconstruction.
<!-- ORACLE_OP_END op_020_case_loading -->

<!-- ORACLE_OP_START op_030_grid_array_conventions -->
## Pixel grid, array flattening, and block-average downsampling conventions

These coordinate and array conventions are shared by every later step; use them exactly.

- Centered grid: for a square grid of `N` pixels at pixel size `d`, build flattened coordinate vectors with `np.meshgrid(np.arange(N), np.arange(N))` (default 'xy' indexing), stack column-major into pairs, then `x = (col - (N-1)/2)*d`, `y = (row - (N-1)/2)*d`. Length is N*N. The grid is centered on 0.
- `image2array`: flatten 2D to 1D via row-major `reshape(nx*ny)`.
- `array2image`: reshape a length-M vector to `(n, n)` with `n = int(sqrt(M))`.
- `re_size(image, factor)`: block-average downsampling. If factor==1 copy; else `image.reshape(nx//f, f, ny//f, f).mean(3).mean(1)`. `factor` must divide both axes.

Full reference code: `${CLAUDE_SKILL_DIR}/resources/op_030_grid_array_conventions.md`.
<!-- ORACLE_OP_END op_030_grid_array_conventions -->

<!-- ORACLE_OP_START op_040_shapelet_basis -->
## Cartesian Hermite shapelet basis and ordering

Evaluate 2D Cartesian shapelets built from physicist Hermite polynomials.

1D profile: phi_n(u) = [2^n * sqrt(pi) * n!]^(-1/2) * H_n(u) * exp(-u^2/2), with H_n the physicist Hermite (use `numpy.polynomial.hermite.hermval`).

Precompute per axis with `u = (coord - center)/beta`: for each order n up to n_max, prefactor = 1/sqrt(2^n * sqrt(pi) * n!); apply the Hermite value only where `|u| < sqrt(n+1)*5.0` (zero outside for stability), then multiply by prefactor and exp(-u^2/2). This yields arrays `H_x`, `H_y` of shape `(n_max+1, len(coord))`.

Ordering (critical): iterate index/(n1,n2) with constraint n1+n2 <= n_max as (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),... The count is `(n_max+1)(n_max+2)/2` (231 for n_max=20; 11476 for n_max=150).

Field from coefficients: `f = sum_i amp[i] * H_x[n1] * H_y[n2]`, then `nan_to_num`. Single-basis responses (amp=1) are `nan_to_num(H_x[n1]*H_y[n2])` per index.

Important: these basis arrays carry NO explicit 1/beta factor; overall scaling is handled by the amplitude normalization applied when configuring the source and when writing output. Full code: `${CLAUDE_SKILL_DIR}/resources/op_040_shapelet_basis.md`.
<!-- ORACLE_OP_END op_040_shapelet_basis -->

<!-- ORACLE_OP_START op_050_lens_deflection -->
## SPEP + external shear deflection and backward ray-shooting

Map image-plane coordinates to source-plane via the lens equation beta = theta - alpha(theta).

Deflection has two parts summed together.

SPEP (Softened Power-law Elliptical Potential): convert ellipticity to angle/axis-ratio with phi = arctan2(e2,e1)/2, c = min(sqrt(e1^2+e2^2), 0.9999), q = (1-c)/(1+c). Clip gamma to [1.4, 2.9] and q to >= 0.01. Then phi_E = theta_E*q; E = phi_E / ( ((3-gamma)/2)^(1/(1-gamma)) * sqrt(q) ); eta = 3-gamma. Rotate offsets by phi_G: xt1 = cos*dx + sin*dy, xt2 = -sin*dx + cos*dy, xt2dq2 = xt2/q^2, P2 = xt1^2 + xt2*xt2dq2, a = max(P2, 1e-6). fac = (1/eta) * (a/E^2)^(eta/2 - 1) * 2; fx_p = fac*xt1, fy_p = fac*xt2dq2; rotate back: alpha_x = cos*fx_p - sin*fy_p, alpha_y = sin*fx_p + cos*fy_p.

For the public case (gamma=2, e1=e2=0) this reduces to the Singular Isothermal Sphere alpha = theta_E * (x,y)/r.

External shear: alpha_x = g1*x + g2*y, alpha_y = g2*x - g1*y (both g1=g2=0 here).

Ray-shoot: x_src = x - alpha_x^SPEP - alpha_x^shear, y_src = y - alpha_y^SPEP - alpha_y^shear.

Lens kwargs: {theta_E:0.5, gamma:2.0, center_x:0, center_y:0, e1:0, e2:0}; shear {gamma1:0, gamma2:0}. Full code: `${CLAUDE_SKILL_DIR}/resources/op_050_lens_deflection.md`.
<!-- ORACLE_OP_END op_050_lens_deflection -->

<!-- ORACLE_OP_START op_060_psf_convolution -->
## Gaussian PSF convolution convention

Convolve the high-resolution image with a Gaussian PSF.

- sigma from FWHM: sigma = fwhm / (2*sqrt(2*ln 2)).
- Convert to pixels using the SUPERSAMPLED pixel size: `sigma_pix = sigma / pixel_size` where `pixel_size = deltaPix / supersampling_factor` (0.01 arcsec on the 320-grid).
- Apply `scipy.ndimage.gaussian_filter(image, sigma_pix, mode='constant', truncate=5)`. This preserves total flux (sum).
- If fwhm <= 0, return the image unchanged (copy).

Use `fwhm_lensing = 0.1` arcsec for the lensed reconstruction (both when simulating the observation and when building the response matrix). Full code: `${CLAUDE_SKILL_DIR}/resources/op_060_psf_convolution.md`.
<!-- ORACLE_OP_END op_060_psf_convolution -->

<!-- ORACLE_OP_START op_070_forward_simulation -->
## Forward image simulation and lensed source configuration

Assemble the forward model to produce the noise-free lensed observation.

Pipeline `simulate_image(numPix, deltaPix, ss, fwhm, kwargs_source, kwargs_spemd, kwargs_shear, apply_lens, apply_psf)`:
1. `n_super = numPix*ss`, `dpix_super = deltaPix/ss`; build the centered grid at `dpix_super`.
2. If lensing, ray-shoot the grid to source plane; else keep image-plane coords.
3. Evaluate the shapelet source field at those coords (amp, n_max, beta, center from kwargs_source).
4. Reshape to 2D; if PSF, convolve with `fwhm` at `dpix_super`; if `ss>1`, block-average down by `ss`.
5. Multiply the image by `deltaPix**2` (convert surface brightness to flux per detector pixel).

Source configuration for the lensed observation (critical constants):
- `coeff_lens = coeff_ngc / deltaPix**2` (deltaPix=0.05).
- `kwargs_source_lens = {n_max:150, beta:beta_model_lensing=0.06, amp:coeff_lens, center_x:source_x=0.2, center_y:0.0}`.
- `kwargs_spemd = {theta_E:0.5, gamma:2.0, center_x:0, center_y:0, e1:0, e2:0}`, `kwargs_shear = {gamma1:0, gamma2:0}`.

Generate the detector-resolution noise-free image (64x64):
`image_no_noise = simulate_image(numPix=64, deltaPix=0.05, ss=high_res_factor=5, fwhm=fwhm_lensing=0.1, kwargs_source_lens, kwargs_spemd, kwargs_shear, apply_lens=True, apply_psf=True)`.

Full code: `${CLAUDE_SKILL_DIR}/resources/op_070_forward_simulation.md`.
<!-- ORACLE_OP_END op_070_forward_simulation -->

<!-- ORACLE_OP_START op_080_noise_observation -->
## Poisson+background noise model and reproducible observation

Form the observed image by adding heteroscedastic noise to the noise-free simulation.

Noise components:
- Poisson (Gaussian approximation): per-pixel draw `randn(shape) * sqrt(|image| / exp_time)`.
- Background: `randn(shape) * background_rms`.
- Diagonal noise variance model used by the weights and residuals: `C_D_i = background_rms**2 + |value_i| / exp_time`.

Reproduce the reference observation deterministically (order matters):
```
np.random.seed(42)
image_real = image_no_noise \
    + randn(shape)*sqrt(|image_no_noise|/exp_time) \
    + randn(shape)*background_rms
```
with `exp_time=100`, `background_rms=10`. Seed first, draw Poisson term, then draw background term. `image_real` is the 64x64 data vector `d` for the WLS solve.

Full code: `${CLAUDE_SKILL_DIR}/resources/op_080_noise_observation.md`.
<!-- ORACLE_OP_END op_080_noise_observation -->

<!-- ORACLE_OP_START op_090_response_matrix -->
## Response (design) matrix construction

Build the linear forward operator A (num_basis x numPix^2). Each row is one reconstruction basis function pushed through the full forward pipeline.

Procedure `build_response_matrix(numPix, deltaPix, ss, fwhm, n_max_recon, beta_recon, cx, cy, kwargs_spemd, kwargs_shear, apply_lens)`:
1. `n_super=numPix*ss`, `dpix_super=deltaPix/ss`; build grid.
2. If lensing, ray-shoot grid to source plane; else keep image-plane coords.
3. Get the list of single-basis responses (amp=1) at n_max_recon, beta_recon, center (cx,cy).
4. For each basis: reshape to 2D; convolve with PSF at `dpix_super` (truncate=5) if fwhm>0; block-average by `ss` if ss>1; multiply by `deltaPix**2`; flatten into row i of A.

Reconstruction configuration (lensed case): `n_max_recon=20` -> `num_basis = 21*22/2 = 231`; `beta_recon = beta_recon_lensing = 0.15`; center `(source_x=0.2, source_y=0.0)`; `apply_lens=True`; `fwhm=fwhm_lensing=0.1`; detector `numPix=64`, `deltaPix=0.05`, `ss=5`. Result `A_lens` shape `(231, 4096)`.

Note the reconstruction uses far fewer basis functions (n_max=20) than the source decomposition (n_max=150); this finite order is the implicit regularization. Full code: `${CLAUDE_SKILL_DIR}/resources/op_090_response_matrix.md`.
<!-- ORACLE_OP_END op_090_response_matrix -->

<!-- ORACLE_OP_START op_100_wls_solve -->
## Weighted least squares coefficient inversion

Solve for shapelet coefficients by weighted least squares on the linear model d = A^T c.

Given A (num_basis x numPix^2) and the observed 2D data:
1. Flatten data: `d = image2array(data_2d)` (length numPix^2).
2. Noise weights: `C_D = background_rms**2 + |d| / exp_time`, `w = 1/C_D` (background_rms=10, exp_time=100).
3. Form `Aw = A * w[None, :]`, `M = Aw @ A.T` (num_basis x num_basis), `R = Aw @ d`.
4. Condition guard: `cond = np.linalg.cond(M)`; if `cond < 1e14` use `np.linalg.solve(M, R)`, else fall back to `np.linalg.lstsq(M, R, rcond=None)[0]`.
5. Model image (optional diagnostic): `model = array2image(A.T @ params)`.

Apply: `param_lens, model_lens = linear_solve(A_lens, image_real, 10.0, 100.0)`. `param_lens` has length 231 and feeds the source-image output. Full code: `${CLAUDE_SKILL_DIR}/resources/op_100_wls_solve.md`.
<!-- ORACLE_OP_END op_100_wls_solve -->

<!-- ORACLE_OP_START op_110_source_image_output -->
## High-resolution source reconstruction and submission writing

Evaluate the recovered coefficients on the high-resolution grid and write the submission.

1. `numPix_hr = numPix * high_res_factor = 320`; `deltaPix_hr = deltaPix / high_res_factor = 0.01`.
2. Build the centered grid `x_hr, y_hr = make_grid(320, 0.01)`.
3. Evaluate the source field with the recovered coefficients: `source = shapelet_function(x_hr, y_hr, param_lens, n_max_recon=20, beta=beta_recon_lensing=0.15, cx=source_x=0.2, cy=source_y=0.0)`. Note beta and center MUST match those used to build the response matrix (0.15, (0.2,0.0)).
4. `source_image = array2image(source) * deltaPix_hr**2` -> shape `(320, 320)`, flux per high-res pixel.
5. Ensure finite float array; write `outputs/{case_id}.npz` with key `source_image` (float32 or float64).

Create the `outputs/` directory if missing. Full code: `${CLAUDE_SKILL_DIR}/resources/op_110_source_image_output.md`.
<!-- ORACLE_OP_END op_110_source_image_output -->

<!-- ORACLE_OP_START op_120_submission_validation -->
## Submission schema and finiteness validation

Validate each written submission before finishing.

Checks:
- File exists at `outputs/{case_id}.npz`.
- Contains key `source_image`.
- Shape is exactly `(320, 320)`.
- dtype is floating (float32 or float64).
- All entries finite (no NaN/Inf).

Run with the run-local Python from the run root:
`python ${CLAUDE_SKILL_DIR}/scripts/op_120_submission_validation.py outputs/case_000.npz`.
A nonzero exit or printed FAIL means the output must be regenerated before submission.
<!-- ORACLE_OP_END op_120_submission_validation -->

<!-- ORACLE_OP_START op_130_runtime_probe -->
## Cheap forward-model smoke test

Fast, non-mutating sanity check to catch convention/shape bugs before the full run. Uses a tiny grid and low shapelet order so it finishes in well under a second.

It verifies: the centered grid and array reshape round-trip, the shapelet ordering count `(n_max+1)(n_max+2)/2`, a small response matrix has the expected shape, and the normal-equations matrix `M = Aw @ A.T` is finite and square. It does not read any case data and writes nothing.

Run from the run root: `python ${CLAUDE_SKILL_DIR}/scripts/op_130_runtime_probe.py`. Exit 0 = pass.
<!-- ORACLE_OP_END op_130_runtime_probe -->
