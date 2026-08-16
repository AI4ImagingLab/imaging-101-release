---
name: oracle-shapelet-source-reconstruction
description: "Reconstruct the intrinsic (unlensed) source-galaxy surface brightness from a noisy, PSF-convolved, gravitationally lensed observation using a Cartesian Hermite shapelet basis and weighted-least-squares inversion of a linear forward model."
---

# oracle-shapelet-source-reconstruction Oracle Skill

This skill reconstructs the intrinsic (unlensed) surface brightness of a galaxy seen through a known gravitational lens. The domain is astronomy: a source galaxy is represented in a Cartesian Hermite shapelet basis, imaged through a linear forward model (backward ray-tracing through the lens, evaluation of the source, Gaussian PSF convolution, and block-average downsampling), and recovered by weighted-least-squares inversion. The workflow covers the public submission contract, per-case loading of parameters and shapelet coefficients, shared grid and array conventions, the shapelet basis, the lens deflection field, PSF convolution, the forward image simulation, the heteroscedastic noise model, response-matrix construction, the weighted-least-squares solve, high-resolution source evaluation and output writing, plus validation and a cheap smoke test. All numerics use only numpy and scipy. Constants, axis and ordering conventions, normalization factors, scale parameters, and the output schema are preserved so the standard behavior can be reconstructed. Run everything from the eval run root with the run-local Python (for example `python` on PATH, or `public/envs/runtime/.venv-posix/bin/python`).

## Cartesian Hermite shapelet basis and ordering

Evaluate 2D Cartesian shapelets built from physicist Hermite polynomials.

1D profile: phi_n(u) = [2^n * sqrt(pi) * n!]^(-1/2) * H_n(u) * exp(-u^2/2), with H_n the physicist Hermite (use `numpy.polynomial.hermite.hermval`).

Precompute per axis with `u = (coord - center)/beta`: for each order n up to n_max, prefactor = 1/sqrt(2^n * sqrt(pi) * n!); apply the Hermite value only where `|u| < sqrt(n+1)*5.0` (zero outside for stability), then multiply by prefactor and exp(-u^2/2). This yields arrays `H_x`, `H_y` of shape `(n_max+1, len(coord))`.

Ordering (critical): iterate index/(n1,n2) with constraint n1+n2 <= n_max as (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),... The count is `(n_max+1)(n_max+2)/2` (231 for n_max=20; 11476 for n_max=150).

Field from coefficients: `f = sum_i amp[i] * H_x[n1] * H_y[n2]`, then `nan_to_num`. Single-basis responses (amp=1) are `nan_to_num(H_x[n1]*H_y[n2])` per index.

Important: these basis arrays carry NO explicit 1/beta factor; overall scaling is handled by the amplitude normalization applied when configuring the source and when writing output. Full code: `${CLAUDE_SKILL_DIR}/resources/resource_001.md`.

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

Full code: `${CLAUDE_SKILL_DIR}/resources/resource_002.md`.

## High-resolution source reconstruction and submission writing

Evaluate the recovered coefficients on the high-resolution grid and write the submission.

1. `numPix_hr = numPix * high_res_factor = 320`; `deltaPix_hr = deltaPix / high_res_factor = 0.01`.
2. Build the centered grid `x_hr, y_hr = make_grid(320, 0.01)`.
3. Evaluate the source field with the recovered coefficients: `source = shapelet_function(x_hr, y_hr, param_lens, n_max_recon=20, beta=beta_recon_lensing=0.15, cx=source_x=0.2, cy=source_y=0.0)`. Note beta and center MUST match those used to build the response matrix (0.15, (0.2,0.0)).
4. `source_image = array2image(source) * deltaPix_hr**2` -> shape `(320, 320)`, flux per high-res pixel.
5. Ensure finite float array; write `outputs/{case_id}.npz` with key `source_image` (float32 or float64).

Create the `outputs/` directory if missing. Full code: `${CLAUDE_SKILL_DIR}/resources/resource_003.md`.
