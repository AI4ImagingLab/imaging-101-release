---
name: oracle-usct-FWI
description: "Reconstruct a 480x480 breast-tissue sound-speed map from multi-frequency ultrasound transmission data by frequency-domain Full-Waveform Inversion built on a Convergent Born Series Helmholtz forward solver, nonlinear conjugate gradient optimization, and low-to-high multi-frequency bootstrapping."
---

# oracle-usct-FWI Oracle Skill

This skill reconstructs a quantitative sound-speed (velocity) map on a 480x480 grid from frequency-domain ultrasound computed tomography (USCT) transmission data using Full-Waveform Inversion. The pipeline: read the case parameters and complex observed wavefields (256 sources x 256 receivers per frequency); build a sparse receiver-restriction operator and a homogeneous initial slowness; solve the Helmholtz forward problem per source with a Convergent Born Series solver (FFT propagation, wiggle phase ramps, i*V/epsilon preconditioner); estimate per-source intensity by least squares and form a near-source-muted data misfit; compute the adjoint-state gradient with frequency-dependent Gaussian smoothing; minimize with Polak-Ribiere nonlinear conjugate gradient and a bound-constrained More-Thuente line search; and bootstrap low-to-high across 20 frequencies. Finally recover velocity as 1/slowness and write outputs/{case_id}.npz with key vp_reconstructed. Operations are ordered as contract/data/conventions -> forward physics -> objective/gradient/solver -> output/validation, each independently removable for ablation.

<!-- ORACLE_OP_START op_010_contract -->
## Task contract, case discovery, and submission schema

## Task contract and submission

Goal: for each case, reconstruct the acoustic sound-speed (velocity) map `vp` on a 480x480 grid from frequency-domain USCT transmission data, and write it as the required output array.

- Cases live under `public/` (run-local). Read the case list from `public/visible_data/cases.json`; each case has an id (e.g. `case_000`), an `input_dir`, and a `params` JSON. There is one public case, but iterate generically over all listed cases.
- Per-case inputs: `<input_dir>/raw_data.npz` (observations) and a `params_data.json` (physical + discretization parameters).
- Output: write `outputs/{case_id}.npz` relative to the run root, containing a single required array `vp_reconstructed` of shape `(480, 480)`, dtype float32 or float64. All values must be finite (no NaN/Inf). Casting is not allowed, so produce the array already in float32/float64.
- The reconstructed map is the primary evaluation target; nothing else is required in the npz.

Create the `outputs/` directory if missing. Keep the array orientation consistent with the grid conventions defined by later operations (velocity = 1/slowness, shape (nx, ny) = (480, 480)).
<!-- ORACLE_OP_END op_010_contract -->

<!-- ORACLE_OP_START op_020_metadata_params -->
## Physical and discretization parameters

## Read case parameters

Load the case `params_data.json` (path from the case entry) into a dict. Use these fields verbatim; do not hardcode if present:

- `nx=480`, `ny=480`, `dh_um=50` (grid spacing in micrometers), `domain_size_cm=[24,24]`.
- `n_transducers_active=256` (this is the number of sources = number of receivers = K), `transducer_geometry="circular"`.
- `frequencies_MHz`: 20 values `[0.3, 0.35, ..., 1.25]` in ascending order.
- `background_velocity_mps=1480` (initial homogeneous velocity).
- `velocity_bounds_mps=[1300, 1700]` (bound constraints).
- `mute_distance_um=7500` (near-source muting radius).
- CBS solver params: `cbs_ppw=8`, `cbs_lamb=1`, `cbs_energy_threshold=1e-5`, `cbs_born_max=500`, `cbs_boundary_widths=20`.

These values feed the initial model, geometry, muting, CBS domain setup, and optimization bounds. Keep them in a single config object passed through the pipeline.
<!-- ORACLE_OP_END op_020_metadata_params -->

<!-- ORACLE_OP_START op_030_load_observations -->
## Load observed wavefields and receiver positions

## Load raw_data.npz

Open `<input_dir>/raw_data.npz` with numpy. Keys and handling:

- `receiver_ix` shape `(1, 256)` float32, `receiver_iy` shape `(1, 256)` float32: transducer x/y positions as **1-indexed** grid coordinates (MATLAB-style). Squeeze the leading batch axis to get `(256,)` arrays; keep them as float32. The receiver positions and source positions are the same 256 transducer locations.
- Frequency data keys are named `dobs_{freq}` where `{freq}` is the frequency in MHz formatted with `%g` (e.g. `dobs_0.3`, `dobs_0.35`, ..., `dobs_1.25`). Each is shape `(1, 256, 256)` complex64; element `(0, i, j)` is the recorded field at receiver `i` when source `j` transmits.
- Squeeze the leading axis to `(256, 256)` complex64. Build a dict `{freq_str: dobs}` by iterating all keys starting with `dobs_` and stripping the `dobs_` prefix (5 chars) to recover the frequency string.
- Load arrays onto the compute device (GPU tensors, complex64) for the solve; keep receiver position arrays as CPU numpy/float32 for indexing and mask construction.

Do not transpose the (receiver, source) axis order: axis 0 = receiver index, axis 1 = source index. This ordering is assumed by source-intensity estimation and misfit.
<!-- ORACLE_OP_END op_030_load_observations -->

<!-- ORACLE_OP_START op_040_restriction_operator -->
## Sparse restriction operator to receiver locations

## Restriction operator R

Build a sparse operator `R` of shape `(n_rec, nx*ny)` that samples a flattened full wavefield at the receiver grid nodes.

Given 1-indexed receiver positions `ix, iy` (length `n_rec=256`) and grid `nx, ny`:

- Linear (1-indexed) node index: `src_linear_index = (iy - 1) * ny + ix`.
- Zero-based column index: `icol = src_linear_index - 1` (int), flattened.
- Row index: `irow = arange(n_rec)`; values all ones (float32).
- Assemble as a COO sparse matrix of shape `(n_rec, ny*nx)`, then convert to a sparse tensor on the compute device.

Usage: for the per-source wavefield stack `all_u` of shape `(nx, ny, n_rec)`, reshape to `(nx*ny, n_rec)` and apply `R` separately to real and imaginary parts via sparse matmul, recombining as complex. This yields `dsrc` of shape `(n_rec, n_rec)` = simulated field at each receiver for each source. The `(iy-1)*ny + ix` convention (note the mixed 1-indexing on `ix`) matches the way positions are stored and must be preserved exactly to hit the correct grid nodes.
<!-- ORACLE_OP_END op_040_restriction_operator -->

<!-- ORACLE_OP_START op_050_initial_slowness -->
## Homogeneous initial slowness model

## Initial model

Start from a homogeneous slowness field `s = 1 / v0` where `v0 = background_velocity_mps = 1480`. Create a float32 tensor of shape `(nx, ny) = (480, 480)` filled with `1/1480` on the compute device.

The optimization variable throughout is **slowness** `s` (units 1/(m/s)); velocity is recovered as `vp = 1/s`. Keep this convention consistent everywhere: the forward solver, bounds, and output writing all convert between `s` and `vp` via reciprocal. This single homogeneous starting model is reused as the initialization for the first (lowest) frequency; each subsequent frequency continues from the previous result.
<!-- ORACLE_OP_END op_050_initial_slowness -->

<!-- ORACLE_OP_START op_060_near_source_muting -->
## Near-source muting and observation masks

## Near-source muting masks

Suppress near-field artifacts by muting receiver-source pairs closer than `mute_distance_um = 7500` um.

For each source `isrc` and receiver `irec` (both over 256 transducers), compute the grid-node distance `sqrt((ix[isrc]-ix[irec])^2 + (iy[isrc]-iy[irec])^2) * dh` (with `dh=50` um). Build a boolean `mask` of shape `(n_rec, n_rec)` indexed `[irec, isrc]` that is `False` (muted) when this distance `< mute_dist`, else `True`.

Produce:
- `mask_esi = mask` (used for source-intensity estimation).
- `mask_misfit = mask.clone()` then additionally set entries to `False` where `abs(dobs) == 0` (missing/zero observations excluded from misfit).
- `dobs_masked = dobs * mask_esi` (apply muting to the observed data used for fitting).

All masks live on the compute device. This muting removes strong near-source pairs that would otherwise dominate the misfit and gradient.
<!-- ORACLE_OP_END op_060_near_source_muting -->

<!-- ORACLE_OP_START op_070_cbs_domain_setup -->
## CBS Helmholtz domain setup (potential, preconditioner, wiggles)

## CBS domain preparation

The forward model solves `(nabla^2 + omega^2 s^2) u = -rho` per source with the Convergent Born Series (CBS), which converges for arbitrarily large contrast via a preconditioner `gamma = i*V/epsilon`. Prepare a reusable domain for a given velocity field and frequency (frequency is passed in **100 kHz units**, i.e. `freq_100k = freq_MHz * 10`).

Key steps and constants (see resource for the exact sequence):
- `pixel_size = lamb/ppw` (with `lamb=1`, `ppw=8` -> `0.125`); boundary in cells `bw = boundary_widths*ppw`; `scale = dh/pixel_size`.
- `speed_mean = mean(velocity)`; `lamb_val = speed_mean/(freq_100k*scale)`; `k0 = 2*pi/lamb_val`.
- refractive index `n = speed_mean/velocity`, permittivity `e_r = n^2`; `e_r_center = (max+min)/2`; `k0c = sqrt(e_r_center)*k0`.
- Pad the 2D field to a 3D FFT-friendly size (dimensions grown until prime factors <= 11 with the second-largest factor <= 5) with replicate padding, tracking the ROI crop indices `[Bl, Br]`.
- Scattering potential `V = e_r*k0^2 - k0c^2 - i*epsilon`, where `epsilon = max(max|V_tot|, 3)` (with a 1.05 bump if the max-magnitude entry is weakly real); apply absorbing edge filters to `V`.
- Preconditioner `gamma = (i/epsilon)*V`; mixing weights `mix_weight1 = 1 - gamma`, `mix_weight2 = i*gamma^2`.
- Propagation kernel scalar `k02e = k0c^2/epsilon + i`; build the 4 wiggle descriptors (2D -> `2^2=4` permutations); each shifts Fourier coordinates by a quarter pixel and provides `filter_step0`, `scaler_step1 = px^2+py^2+pz^2 - k02e`, `filter_step2`.
- Source injection scale `A = i/epsilon/n_wiggle`; `max_iterations = ceil(born_max * lamb_val/(2*k0c/epsilon))`.

Return a dict with grid, wiggles, gamma, mix weights, ROI, epsilon, k02e, Bl, A, dh, energy_threshold, born_max. Removing this operation removes the entire forward physics.
<!-- ORACLE_OP_END op_070_cbs_domain_setup -->

<!-- ORACLE_OP_START op_080_cbs_solve_iteration -->
## CBS iterative wavefield solve per source

## CBS solve for one source and stacking all sources

Using the prepared domain, solve the Helmholtz wavefield for each source (transducer) and stack results.

For a single source at 0-indexed grid position `(ix, iy)` (convert stored 1-indexed positions with `-1`):
- Shift into padded coordinates: `src_x = ix + Bl[0]`, `src_y = iy + Bl[1]`, `src_z = Bl[2]`; source value `= 1.0 * A`.
- Initialize `dE = 0`, `E = 0` (complex, padded size). Iterate `iteration = 0,1,...`:
  - For the first `n_wiggle` iterations, inject the source into a copy of `dE` at the shifted node; afterward do not re-inject.
  - Select wiggle `w = iteration % n_wiggle`. CBS propagation step:
    1. `E_filtered = Etmp * filter_step0`
    2. `fftE = fftn(E_filtered)`
    3. `fftE_scaled = fftE / scaler_step1`
    4. `E_ifft = ifftn(fftE_scaled)`
    5. `E_return = E_ifft * filter_step2`
    6. `dE = mix_weight1 * dE - mix_weight2 * E_return`
  - Accumulate `E += dE`.
  - Every `n_wiggle` iterations (when `iteration % n_wiggle == 0`), check energy of the cropped `dE`: on iteration 0 store `init_energy`; terminate when `last_energy/init_energy < energy_threshold` (1e-5) or `iteration >= born_max` (500).
- After convergence: `E = E / gamma`, crop to ROI, then scale: `scaled_E = -conj(E).T / dh / dh * 2.5e10`. Return complex64 `(nx, ny)`.

`solve_all_sources`: build the domain once per (velocity, frequency), then loop over all 256 sources filling `all_u[..., i]` of shape `(nx, ny, n_src)`. The `2.5e10` scale, the `1/dh/dh`, the conjugate-transpose, and the `E/gamma` recovery are all required to match measurement units used by source estimation. Preserve them exactly.
<!-- ORACLE_OP_END op_080_cbs_solve_iteration -->

<!-- ORACLE_OP_START op_090_source_intensity_objective -->
## Source-intensity least squares and data misfit

## Source-intensity estimation and misfit

After the forward solve produces `all_u` (nx,ny,n_rec), restrict to receivers via the sparse operator to get `dsrc` (n_rec, n_rec) complex, then apply the estimation mask: `dsrc = dsrc * mask_esi`.

Estimate per-source complex intensity `alpha_k` by least squares of simulated to observed data (per source column `k`):

`alpha_k = sum_i conj(dsrc_ik) * dobs_masked_ik / sum_i |dsrc_ik|^2`

Concretely: `den = dsrc * conj(dsrc)`, `num = dobs_masked * conj(dsrc)`, and `alpha[k] = sum(num[:,k]) / sum(den[:,k])` for each source column.

Simulated observations scaled by intensity: `dobs_sim = dsrc * alpha[None,:] * mask_misfit`; also form intensity-scaled fields `u_source_intense = all_u * alpha[None,None,:]` for the gradient.

Misfit: `dobs_misfit = dobs_sim - dobs_masked`; column L2 norms `J_column = norm(dobs_misfit, p=2, dim=0)`; total `J = sum(J_column^2)`. This is `sum_k || alpha_k*u_k(x_r) - y_k ||_2^2`. The estimation uses `mask_esi` while the misfit sum uses `mask_misfit` (which also drops zero observations).
<!-- ORACLE_OP_END op_090_source_intensity_objective -->

<!-- ORACLE_OP_START op_100_adjoint_gradient -->
## Adjoint-state gradient with Gaussian smoothing

## Adjoint gradient and smoothing

Compute the gradient of the misfit w.r.t. slowness via the adjoint-state method, then smooth it. With `omg = 2*pi*freq_100k` and `vp = 1/slowness`:

- For each source `isrc`, form `q_isrc = sum_j all_u[:,:,j] * conj(dobs_misfit[j, isrc])` summed over receiver axis, i.e. `q[:,:,isrc] = sum(all_u * conj(dobs_misfit[:,isrc]), dim=2)`. Stack over sources.
- Gradient accumulation: `G = - real( sum_over_sources( u_source_intense * q ) ) * (omg^2 / vp * 2)`.
- Transpose and rescale: `G = G.T / 1e10`.
- Gaussian blur: build a separable 2D Gaussian kernel of `kernel_size=9` and per-frequency `sigma`; convolve `G` (as a 1x1xHxW float tensor) with `padding=kernel_size//2`.
- Apply objective/gradient normalization scalars passed by the optimizer: `G = G / gscale`, and the objective returned as `J = J / fscale / 1e9`.

Gaussian kernel construction: `x = arange(-k//2+1, k//2+1)`, `w = exp(-x^2/(2*sigma^2))`, normalize `w /= w.sum()`, outer product for 2D. The `sigma` value is set by the multi-frequency schedule. The factors `2`, `omg^2/vp`, `/1e10`, and `/1e9` are calibrated for numerical conditioning and must be preserved.

The closure `get_grad(slowness, fscale=1, gscale=1)` returns `(J, G)` and is the objective handed to the optimizer.
<!-- ORACLE_OP_END op_100_adjoint_gradient -->

<!-- ORACLE_OP_START op_110_ncg_linesearch -->
## NCG optimizer with More-Thuente line search and bounds

## NCG optimization per frequency

Minimize the misfit over slowness with Polak-Ribiere Nonlinear Conjugate Gradient and a bound-constrained More-Thuente cubic line search.

Per-frequency driver (`invert_single_frequency`):
- Convert frequency to 100 kHz units: `freq_100k = freq_MHz * 10`.
- Build masks (muting) and the objective closure for this frequency/sigma.
- Initial scaling: evaluate `f0, g0 = get_grad(slowness)`; set `fscale = f0/1e3` and `gscale = norm(g0,2)/norm(slowness,2)*1e3`. Then optimize `fun = lambda x: get_grad(x, fscale, gscale)`.
- Run NCG for `max_iters=3` with line-search `maxfev=5`, velocity bounds `[1300,1700]`.

NCG details:
- Bounds on slowness: `ub = 1/v_min = 1/1300`, `lb = 1/v_max = 1/1700` (upper slowness = lower velocity).
- Iter 0: `pk = -gk`, `ak=0.01`, initial line-search step `a0=1.0`.
- Iter>0: Polak-Ribiere `bk = <gk, gk-gk_old> / <gk_old,gk_old>` (if denominator>0 else 0), clamp `bk=max(0,bk)`; direction `pk = -gk + bk*pk_old`; step seed `a0 = ak`.
- Line search updates `x = clip(wa + stp*s, lb, ub)` each trial and re-evaluates the objective.
- Stopping: `||g||/N < 1e-13`, or relative function change `<= 1e-10` after the first iteration.

More-Thuente line search constants: `xtol=1e-15`, `ftol=1e-4`, `gtol=1e-2`, `stpmin=1e-15`, `stpmax=1e15`, `xtrapf=4.0`, cubic-step safeguard factor `p66=0.66`; directional derivative uses `dg = <conj(g_flat), conj(s_flat)>`. See resource for the cstep/cvsrch structure. Preserve the bound clipping inside the line search so velocities stay in range.
<!-- ORACLE_OP_END op_110_ncg_linesearch -->

<!-- ORACLE_OP_START op_120_multifreq_bootstrap -->
## Multi-frequency bootstrapping and sigma schedule

## Multi-frequency bootstrapping

Process the 20 frequencies in ascending order (0.3 -> 1.25 MHz). Each frequency's optimized slowness initializes the next (low-to-high bootstrapping avoids local minima and progressively resolves finer structure).

For each `freq_mhz` in `frequencies_MHz`:
- Look up its observation `dobs = dobs_dict[f"{freq_mhz:g}"]`.
- Set the gradient-smoothing sigma from the schedule:
  - `sigma = 5` if `freq_mhz <= 0.3`
  - `sigma = 2` if `0.3 < freq_mhz < 0.8`
  - `sigma = 1` if `freq_mhz >= 0.8`
- Run the per-frequency NCG inversion (3 iterations) starting from the current slowness, updating it in place for the next frequency.
- Optionally record `(freq_mhz, (1/slowness))` history for diagnostics.

Large sigma at low frequency captures large-scale structure; shrinking sigma at higher frequency sharpens fine detail. Run the full 20-frequency sweep to obtain the final slowness; a single-frequency variant is possible but produces a lower-quality map.
<!-- ORACLE_OP_END op_120_multifreq_bootstrap -->

<!-- ORACLE_OP_START op_130_output_writing -->
## Recover velocity and write submission npz

## Write the reconstruction

After the multi-frequency sweep, convert the final slowness to velocity: `vp_reconstructed = (1 / slowness)`, move to CPU, shape `(480, 480)`.

- Cast to float32 (float32 or float64 accepted; casting by the harness is disallowed, so make the array the correct floating dtype yourself).
- Ensure all values are finite. Since the optimizer is bound-constrained to slowness in `[1/1700, 1/1300]`, velocities should lie in `[1300, 1700]`; still verify no NaN/Inf before writing.
- Save to `outputs/{case_id}.npz` (relative to run root) with the single key `vp_reconstructed`. Create `outputs/` if needed.
- Do not transpose or flip at the end unless the internal grid convention already requires it; keep orientation consistent with how velocity was assembled (the internal wavefield scaling already applied the transpose in the solver).

Write one npz per case using its `case_id`.
<!-- ORACLE_OP_END op_130_output_writing -->

<!-- ORACLE_OP_START op_140_output_validation_probe -->
## Output schema and finiteness validator

## Validate the submission

Before finishing, verify each written `outputs/{case_id}.npz` meets the contract. This is a cheap guardrail, not part of the reconstruction.

Checks:
- File exists and loads with numpy.
- Contains key `vp_reconstructed` with shape exactly `(480, 480)`.
- Dtype is float32 or float64.
- All entries finite (no NaN/Inf).
- Optional sanity: values roughly within `[1300, 1700]` m/s (warn only).

Run the bundled validator with run-local python, e.g. `python ${CLAUDE_SKILL_DIR}/scripts/op_140_output_validation_probe.py outputs/case_000.npz`. A non-zero exit indicates a schema violation to fix before submission.
<!-- ORACLE_OP_END op_140_output_validation_probe -->
