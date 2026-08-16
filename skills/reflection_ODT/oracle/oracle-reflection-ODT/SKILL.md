---
name: oracle-reflection-ODT
description: "Task-specific reconstruction guidance for reflection-mode optical diffraction tomography from multi-angle reflected-light intensity data."
---

# oracle-reflection-ODT Oracle Skill

Use this skill when preparing a submission for the reflection-mode ODT case. It records the public I/O contract, numerical conventions, reflection BPM equations, solver settings, and submission checks needed for a faithful reconstruction.

<!-- ORACLE_OP_START op_001_public_contract -->
## Public case and submission contract

The run has one public case, `case_000`. Read inputs from `public/visible_data/cases/case_000/` and write exactly `outputs/case_000.npz` from the run root. The NPZ must contain a required array named `delta_n` with shape `(4, 128, 128)`, finite values only, and dtype exactly `float32` or `float64` without relying on validator casting. Treat `delta_n` as RI contrast volume ordered `(Nz, Ny, Nx)`. Create `outputs/` if needed, and do not add a batch dimension in the submitted array. Additional files may be produced only as diagnostics, not as the primary submission.
<!-- ORACLE_OP_END op_001_public_contract -->

<!-- ORACLE_OP_START op_002_load_public_inputs -->
## Load metadata and measured intensities

Load `public/visible_data/cases/case_000/params_data.json` as the metadata and `public/visible_data/cases/case_000/input_data/raw_data.npz` as the measured data. The NPZ key is `measurements` with shape `(1, 56, 128, 128)` and float intensities. Remove the leading singleton batch axis before reconstruction, yielding `(56, 128, 128)`. Use float64 internally for the inverse solver and compute measured amplitudes as `sqrt(measurements)`; the loss convention compares amplitudes, not raw intensities.
<!-- ORACLE_OP_END op_002_load_public_inputs -->

<!-- ORACLE_OP_START op_003_config_units_and_dtypes -->
## Normalize physical units and tensor dtypes

Convert metadata to the normalized BPM convention before building optical operators. Use `res = tuple(res_um_i * n0 / wavelength_um for i in x,y,z)`. With the public metadata this is approximately `(0.6677848101265823, 0.6677848101265823, 2.4367088607594938)`. Keep array order `(Nz, Ny, Nx)` and lateral axes `(y, x)` for images. Use complex128 for fields and Fourier-domain operators and float64 for real volumes, frequencies, amplitudes, and losses when running the iterative solver. Select `cuda` only if available; CPU is valid but slower.
<!-- ORACLE_OP_END op_003_config_units_and_dtypes -->

<!-- ORACLE_OP_START op_004_illumination_order_and_tilts -->
## Illumination rings, ordering, and tilted plane waves

Simulate and compare angles in the metadata ring order: BF NA 0.130 with 8 angles, BF NA 0.251 with 16 angles, DF NA 0.420 with 16 angles, then DF NA 0.560 with 16 angles, for a total of 56 images. Within each ring, angle `m` has `theta = 2*pi*m/n_angles_in_ring`, direction cosines `c_a = (NA/n0)*cos(theta)` and `c_b = (NA/n0)*sin(theta)`. Truncate each to a periodic Fourier-compatible integer grid by `c_a = int(c_a * nx * res[0]) / (nx * res[0])` and `c_b = int(c_b * ny * res[1]) / (ny * res[1])`. The incident field is `exp(2j*pi*c_b*res[1]*y)[:,None] * exp(2j*pi*c_a*res[0]*x)[None,:]` on pixel coordinates starting at 0.
<!-- ORACLE_OP_END op_004_illumination_order_and_tilts -->

<!-- ORACLE_OP_START op_005_frequency_pupil_masks -->
## Frequency grid, propagation phase, evanescent damping, and pupil

Precompute Fourier-grid operators on `(Ny,Nx)`. Use `fx = fftfreq(nx)/res[0]`, `fy = fftfreq(ny)/res[1]`, `FY, FX = meshgrid(fy, fx, indexing='ij')`. Define `c_gamma = sqrt(clamp(1 - FX**2 - FY**2, min=1e-8))` and `kz = c_gamma * (2*pi*res[2])`. Multiply each propagation transfer function by the evanescent damping mask `exp(clamp((c_gamma - 0.2)*5.0, max=0.0))`. The objective pupil uses `na_norm = NA_obj/n0`, `cutoff = sqrt(1 - na_norm**2)`, `mask = exp(clamp(c_gamma - cutoff, max=0.01)*10000.0)`, then `pupil = mask/(1+mask)` cast to complex. Apply the pupil in Fourier space just before intensity detection.
<!-- ORACLE_OP_END op_005_frequency_pupil_masks -->

<!-- ORACLE_OP_START op_006_bpm_operators -->
## Free-space propagation and slice scattering operators

Use two elementary BPM operators. Free-space propagation for distance `dz` is `ifft2(fft2(u) * exp(1j*kz*dz) * eva_mask)`. Slice scattering is a real-space multiplicative phase `u * exp(1j * dn_slice * (2*pi*res[2]/n0*dz))`. Use the layer thickness `dz_layer=0.5` for the scatter phase, not the large gap. The sign of `delta_n` is physically meaningful; do not impose positivity by default because the public contrast structures are negative relative to background.
<!-- ORACLE_OP_END op_006_bpm_operators -->

<!-- ORACLE_OP_START op_007_reflection_round_trip -->
## Reflection-mode round-trip layer traversal

For one angle, start with its tilted incident field and traverse all four slices downward. For each slice index `iz=0..Nz-1`: propagate by `dz_layer`, scatter with `dn[iz]` using `dz_layer`, then propagate by `dz_gap`. Reflect at the mirror by multiplying the whole field by `-1`. Traverse upward in reverse order `iz=Nz-1..0`: propagate by `dz_gap`, propagate by `dz_layer`, then scatter with `dn[iz]` using `dz_layer`. Back-propagate to the focal plane using `total_depth = Nz*(dz_gap + dz_layer)` and propagation distance `-total_depth`. Then apply the pupil in Fourier space and return intensity `abs(u)**2`. This exact round-trip ordering, including two gap traversals per layer and the mirror phase, is central.
<!-- ORACLE_OP_END op_007_reflection_round_trip -->

<!-- ORACLE_OP_START op_008_amplitude_loss -->
## Amplitude-domain data fidelity

Fit measured amplitudes, not intensities. Convert each measured intensity image to `meas_amp = sqrt(measurements)`. For a predicted intensity image use `pred_amp = sqrt(pred_intensity + 1e-12)` and per-angle loss `sum((pred_amp - meas_amp[global_idx])**2)/(Ny*Nx)`. Accumulate the scalar loss over all 56 angles in ring order before applying the parameter update. The small `1e-12` stabilizer belongs only under the predicted square root.
<!-- ORACLE_OP_END op_008_amplitude_loss -->

<!-- ORACLE_OP_START op_009_fista_update -->
## FISTA reconstruction update and default hyperparameters

Initialize `delta_n` as zeros with shape `(4,128,128)`, float64, and gradients enabled. Initialize `x_prev` as a zero volume. Use default outer iterations `n_iter=50`, gradient step size `lr=5.0`, TV weight `8e-7`, and `positivity=False`. At each iteration compute the amplitude loss for all angles and call backpropagation so the gradient accumulates in `delta_n.grad`. Take `x_k = delta_n - lr*grad`, clear the gradient, optionally apply the TV proximal operation with parameter `lr*tv_weight`, do not clamp nonnegative by default, then use FISTA momentum `q(0)=1`, `q(k)=(1+sqrt(1+4*q(k-1)^2))/2`, `beta=(q(step)-1)/q(step+1)`, and `delta_n = x_k + beta*(x_k - x_prev)`. Copy `x_k` to `x_prev` after the momentum step.
<!-- ORACLE_OP_END op_009_fista_update -->

<!-- ORACLE_OP_START op_010_tv_proximal -->
## Per-slice 2D isotropic TV proximal denoising

Apply total variation denoising independently to each z-slice after the gradient step. Use Chambolle dual projection for `n_iter=20` inner iterations and parameter `tau = lr*tv_weight = 4e-6` under the default settings. Forward differences are `dx[:, :-1] = img[:, 1:] - img[:, :-1]` and `dy[:-1, :] = img[1:, :] - img[:-1, :]` with zero at the right and bottom boundaries. The divergence adjoint sets `div[:,0]=px[:,0]`, `div[:,1:-1]=px[:,1:-1]-px[:,:-2]`, `div[:,-1]=-px[:,-2]`, adds `py[0,:]` to the top row, `py[1:-1,:]-py[:-2,:]` to interior rows, and `-py[-2,:]` to the bottom row. Each dual step uses `x_hat = img - tau*div_p`, `px_new = px + gx/(8*tau)`, `py_new = py + gy/(8*tau)`, then projects by dividing by `sqrt(px_new**2+py_new**2).clamp(min=1)`. Return `img - tau*div_p`.
<!-- ORACLE_OP_END op_010_tv_proximal -->

<!-- ORACLE_OP_START op_011_output_packaging -->
## Package reconstructed contrast as submission NPZ

After reconstruction, detach any tensor, move it to CPU, and convert to a NumPy array with shape `(4,128,128)`. Preserve the signed contrast values; do not convert to absolute RI by adding `n0`. If using float64 internally, either save float64 directly or explicitly cast to float32. Save with `np.savez('outputs/case_000.npz', delta_n=dn_recon)` from the run root. Ensure the output directory exists before saving.
<!-- ORACLE_OP_END op_011_output_packaging -->

<!-- ORACLE_OP_START op_012_submission_validation -->
## Validate submission schema and numerical sanity

Before finishing, run a local schema check on `outputs/case_000.npz`: key `delta_n` exists, shape is `(4,128,128)`, dtype is float32 or float64, and all entries are finite. Also print min, max, mean, and nonzero count. A plausible signed-contrast reconstruction for this task usually has values near zero background and negative structure-like excursions; the validator should not enforce exact ranges, but it should catch NaNs, axis mistakes, and accidental intensity outputs.
<!-- ORACLE_OP_END op_012_submission_validation -->

<!-- ORACLE_OP_START op_013_fast_smoke_probe -->
## Cheap runtime and convention smoke probe

Use a cheap non-mutating probe to confirm that the runtime can import NumPy and Torch, that public files are reachable, and that the illumination and mask conventions produce reasonable arrays before launching a long iterative solve. Run probes with the run-local `python`, for example `python ${CLAUDE_SKILL_DIR}/scripts/op_013_smoke_probe.py`. The probe should not read hidden data, should not write the submission, and should finish quickly on CPU.
<!-- ORACLE_OP_END op_013_fast_smoke_probe -->
