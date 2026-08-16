---
name: oracle-eht-black-hole-feature-extraction-dynamic
description: "Per-frame geometric crescent parameter inference from time-varying EHT closure quantities using an alpha-DPI (alpha-divergence variational inference plus importance sampling) Real-NVP normalizing flow. Produces one estimated bright-side position angle per time frame."
---

# oracle-eht-black-hole-feature-extraction-dynamic Oracle Skill

This skill reconstructs the bright-side position angle of a rotating black-hole crescent independently at each time snapshot of a dynamic EHT observation. For each of the 10 frames it extracts gain-invariant closure phases and log closure amplitudes from complex visibilities, builds a differentiable geometric-crescent-to-image renderer and a NUFFT interferometric forward model, trains a Real-NVP flow to minimize an alpha-divergence data-fidelity objective with a log-linear data warmup, then performs importance-sampling reweighting to obtain a calibrated posterior. The importance-weighted mean of the position-angle column is written per frame. The individual conventions (grid orientation, angle mapping, closure construction, NUFFT scaling, loss normalization, flow architecture, training schedule, importance weights, and output packaging) are separated into removable operations so their contribution can be ablated.

<!-- ORACLE_OP_START op_010_contract -->
## Public task contract and per-frame output schema

## Task contract and output schema

Goal: for a single case, infer the bright-side position angle of a rotating crescent at each of `T = 10` time frames and write one value per frame.

Inputs live under run-local `public/` (discover the case directory from `public/cases.json`; the single visible case is `case_000`). Each case provides `input_data/raw_data.npz` (per-frame interferometric data) and `params_data.json` (imaging + inference settings).

Output: write `outputs/{case_id}.npz` relative to the run root, containing exactly one required array:
- key `position_angle_deg`, shape `(10,)`, dtype `float32` or `float64`.
- All values must be finite (no NaN/Inf). No implicit dtype casting is applied by the grader, so produce a float array directly.
- Angle semantics: degrees, periodic modulo 360, measured as `atan2(grid_y, grid_x)` in the generator image plane, where `0 deg` points along `+x` (increasing column index) and `+90 deg` points along `+y` (decreasing row index). Do not apply any extra astronomical axis rotation.

Key invariants:
- Frames are processed independently; there is no temporal coupling between frames.
- Frame ordering in the output vector follows `t = 0..9`, matching `frame_times_hr` in the params file.
- Only the position angle is graded, but recovering it reliably requires solving the full 4-parameter crescent posterior per frame.

Write outputs with the run-local `python` interpreter available on PATH (or the runtime venv python under `public/envs/`). Create the `outputs/` directory if missing.
<!-- ORACLE_OP_END op_010_contract -->

<!-- ORACLE_OP_START op_020_load_raw_frames -->
## Load per-frame visibilities and case metadata

## Loading raw frames and metadata

Read the case's `params_data.json` and `input_data/raw_data.npz` from the run-local `public/` case directory.

From `params_data.json` read at least: `npix` (64), `fov_uas` (120.0), `n_frames` (10), `frame_times_hr` (length 10), `geometric_model` ("simple_crescent"), `r_range` ([10.0, 40.0] uas), `width_range` ([1.0, 40.0] uas), and the inference settings `n_flow` (16), `n_epoch` (5000), `batch_size` (2048), `lr` (1e-4), `logdet_weight` (1.0), `grad_clip` (1e-4), `alpha_divergence` (1.0), `beta` (0.0), `start_order` (4), `decay_rate` (1000), `seqfrac_inv` (4).

`raw_data.npz` stores per-frame arrays with a numeric suffix. For frame index `t`:
- `vis_{t}`: shape `(28,)` complex128 — complex visibilities (Jy).
- `sigma_{t}`: shape `(28,)` float64 — thermal noise sigma per baseline (Jy).
- `uv_{t}`: shape `(28, 2)` float64 — baseline `(u, v)` coordinates in wavelengths.
- `station_ids_{t}`: shape `(28, 2)` int64 — station index pair per baseline.
Scalar `n_frames` and vector `frame_times` are also present. Load with `np.load(path, allow_pickle=False)` and build a per-frame dict with keys `vis`, `vis_sigma` (from `sigma_{t}`), `uv_coords` (from `uv_{t}`), `station_ids`.

The 8-station EHT array yields 28 baselines per frame; UV coverage varies across frames due to Earth rotation, so closure indices must be recomputed per frame (see the closure-extraction operation).
<!-- ORACLE_OP_END op_020_load_raw_frames -->

<!-- ORACLE_OP_START op_030_closure_extraction -->
## Closure phase and log closure amplitude extraction with sigmas

## Extracting closure quantities

From each frame's `vis`, `vis_sigma`, `station_ids`, build gain-invariant closure observables and the visibility index maps the forward model needs. Use `${CLAUDE_SKILL_DIR}/scripts/op_030_closure_extraction.py` as the reference implementation.

Closure phase triangles: build a baseline lookup `(s1,s2) -> (index, +1)` and `(s2,s1) -> (index, -1)`. For every 3-station combination `(i,j,k)` (sorted stations) that has baselines `(i,j)`, `(j,k)`, `(k,i)`, record three baseline indices `cphase_ind_list[0..2]` and their conjugation signs `cphase_sign_list[0..2]`. Observed closure phase (degrees) is `degrees(sign1*angle(V_idx1) + sign2*angle(V_idx2) + sign3*angle(V_idx3))`. Per-triangle sigma (degrees) is `degrees(sqrt(sum_i (sigma_i/|V_i|)^2))` with `|V_i|` floored by `1e-20`.

Closure amplitude quadrangles: build a lookup keyed by sorted pair `(min,max) -> index`. For every 4-station combination `(i,j,k,l)` with all 6 baselines present, record `camp_ind_list = [idx_ij, idx_kl, idx_ik, idx_jl]`. Closure amplitude `CA = (|V_ij||V_kl|)/(|V_ik||V_jl|)`, `logCA = log(CA)`, and `sigma_logCA = sqrt(sum (sigma/|V|)^2)` over the 4 baselines, `|V|` floored by `1e-20`.

Return a dict with `cphase_ind_list` (3 int64 arrays), `cphase_sign_list` (3 float64 arrays), `camp_ind_list` (4 int64 arrays), `cphase_data={cphase, sigmacp}`, `camp_data={camp}`, `logcamp_data={camp (=logCA), sigmaca}`. If a frame has fewer than 3 closure phases or fewer than 3 closure amplitudes, treat it as degenerate (fall back rather than train).
<!-- ORACLE_OP_END op_030_closure_extraction -->

<!-- ORACLE_OP_START op_040_nufft_params -->
## NUFFT trajectory and pulse-function correction from UV coordinates

## NUFFT parameters (trajectory + pulse correction)

Convert each frame's `uv_coords` (wavelengths) into the k-space trajectory and pulse-function correction consumed by the torchkbnufft forward model. Reference: `${CLAUDE_SKILL_DIR}/scripts/op_040_nufft_params.py`.

Constants and steps:
- `uas_to_rad = pi / (180 * 3600 * 1e6)`; `fov = fov_uas * uas_to_rad`; `psize = fov / npix`.
- Reorder columns to `(v, u)`: `vu = hstack([uv[:,1:2], uv[:,0:1]])`. This `(v, u)` ordering is required by the KbNufft convention.
- Scaled trajectory: `vu_scaled = vu * psize * 2*pi`; `ktraj_vis = tensor(vu_scaled.T, float32).unsqueeze(0)` giving shape `(1, 2, M)`.
- Pulse amplitude: `pulsefac_amp = sinc(u*psize)^2 * sinc(v*psize)^2` (numpy `np.sinc` includes the pi factor).
- Pulse phase: `phase = -pi * (u + v) * psize`; real `= amp*cos(phase)`, imag `= amp*sin(phase)`.
- `pulsefac_vis = tensor(stack([real, imag], axis=0), float32)` giving shape `(2, M)`.

These factors correct for the finite pixel (pulse) shape and the half-pixel image-center offset. Compute them per frame because the UV coordinates change across frames.
<!-- ORACLE_OP_END op_040_nufft_params -->

<!-- ORACLE_OP_START op_050_crescent_model -->
## SimpleCrescent image renderer, grid orientation, and angle convention

## SimpleCrescent geometric model

A differentiable map from 4 unit-interval parameters `[p0,p1,p2,p3]` in [0,1] to a normalized `(npix, npix)` image. This defines the grid orientation and position-angle convention that the graded output must match, so preserve it exactly. Detailed formulas in `${CLAUDE_SKILL_DIR}/resources/op_050_crescent_model.md`.

Grid setup (image coordinates):
- `gap = 1/npix`; `xs = arange(-1+gap, 1, 2*gap)`.
- `grid_y, grid_x = meshgrid(-xs, xs, indexing='ij')` (note the negation of `xs` for the y axis).
- `grid_r = sqrt(grid_x^2 + grid_y^2)`; `grid_theta = atan2(grid_y, grid_x)`.

Physical feature mapping (with `half_fov = 0.5*fov_uas`):
- radius `r = r_range[0]/half_fov + p0*(r_range[1]-r_range[0])/half_fov`.
- width `sigma = width_range[0]/half_fov + p1*(width_range[1]-width_range[0])/half_fov`.
- asymmetry `s = p2`.
- position angle `eta = (181/180)*pi*(2*p3 - 1)` radians.

Image: `ring = exp(-0.5*(grid_r - r)^2 / sigma^2)`, brightness modulation `S = 1 + s*cos(grid_theta - eta)`, `crescent = S*ring`, normalized to unit sum via `crescent / (sum(crescent) + 1e-4)`. Use float32 tensors on the compute device. `nparams = 4`.

The angle `eta` here uses the same `atan2(grid_y, grid_x)` frame as the required output, so the physical position angle in degrees is `181/180 * 180 * (2*p3 - 1)` = `181*(2*p3 - 1)`, periodic modulo 360.
<!-- ORACLE_OP_END op_050_crescent_model -->

<!-- ORACLE_OP_START op_060_nufft_forward -->
## NUFFT interferometric forward model: image to closure quantities

## NUFFT forward model

Maps a batch of images `(B, npix, npix)` to model visibilities and closure quantities using `torchkbnufft.KbNufft(im_size=(npix,npix), numpoints=3)`. Reference formulas in `${CLAUDE_SKILL_DIR}/resources/op_060_nufft_forward.md`.

Steps per batch:
1. Cast images to float32, add a trailing imaginary channel of zeros, shape to the KbNufft real/imag layout, and evaluate `kdata = nufft_ob(x, ktraj_vis)`.
2. Transpose the last two axes and multiply by the pulse correction using real-valued complex multiplication: `vis_real = k_r*p_r - k_i*p_i`, `vis_imag = k_r*p_i + k_i*p_r`, producing `vis` shape `(B, 2, M)` (`[real; imag]`), with `eps = 1e-16`.
3. `visamp = sqrt(vis_r^2 + vis_i^2 + eps)`.
4. Model closure phase (degrees): gather the 3 triangle baselines by index, compute per-baseline `atan2(imag, real)`, apply the stored signs, sum, and multiply by `180/pi`.
5. Model log closure amplitude: gather the 4 quadrangle baselines, take log-amplitude of each (`log(sqrt(r^2+i^2+eps))`), and compute `log|V_ij| + log|V_kl| - log|V_ik| - log|V_jl|`.

The forward model returns `(vis, visamp, cphase, logcamp)`; only `cphase` and `logcamp` feed the loss. Keep `numpoints=3` and the `[real; imag]` channel ordering consistent with the pulse correction.
<!-- ORACLE_OP_END op_060_nufft_forward -->

<!-- ORACLE_OP_START op_070_fidelity_losses -->
## Closure-phase and log-closure-amplitude chi-squared losses

## Data-fidelity losses

Two per-sample chi-squared data terms comparing model closure quantities to observed ones. Both average over the closure dimension and return a per-batch vector.

Closure phase loss (uses `sigmacp` in degrees):
- `L_cp = 2 * mean_over_closures[ (1 - cos(angle_true - angle_pred)) / (sigma * pi/180)^2 ]`, where `angle_true`, `angle_pred` are converted from degrees to radians (`* pi/180`). This cosine form is robust to the 360-degree wrap of phases.

Log closure amplitude loss (uses `sigmaca`):
- `L_ca = mean_over_closures[ (logcamp_true - logcamp_pred)^2 / sigma^2 ]`.

Sigmas are converted to float32 tensors on the device. These two functions are the only data terms used for the `cphase_logcamp` data product; a visibility-amplitude squared-difference loss exists in the source but is not used here. Keep the factor-of-2 and the degrees-to-radians conversion in the closure-phase term.
<!-- ORACLE_OP_END op_070_fidelity_losses -->

<!-- ORACLE_OP_START op_080_realnvp_flow -->
## Real-NVP normalizing flow architecture

## Real-NVP flow

The variational posterior `q_theta` is a Real-NVP flow over the 4-dim parameter space. It is sampled in the reverse direction: `z ~ N(0, I)` -> `params_samp = flow.reverse(z)` -> `params = sigmoid(params_samp)` in [0,1]. Architecture detail in `${CLAUDE_SKILL_DIR}/resources/op_080_realnvp_flow.md`.

Components:
- `ActNorm`: per-tensor location and log-scale with data-dependent init from the first batch; contributes `logdet = in_dim * sum(-log_scale_inv)`.
- `AffineCoupling`: splits input in half; a small MLP (`Linear -> LeakyReLU(0.01) -> BatchNorm1d(eps=1e-2) -> Linear -> LeakyReLU -> BatchNorm1d -> ZeroFC`) on the first half yields `log_s0, t`; `log_s = tanh(log_s0)`, `s = exp(log_s)`, `out_b = (in_b + t)*s`, `logdet = sum(log_s)`. Hidden width `= max(2, ndim/(2*seqfrac))`. `ZeroFC` is a zero-initialized linear layer scaled by `exp(scale*3)`.
- `Flow` block: `ActNorm -> AffineCoupling -> reverse-index -> ActNorm -> AffineCoupling -> reverse-index`.
- `RealNVP`: stacks `n_flow=16` blocks; between blocks apply a fixed random permutation (`np.random.RandomState(seed=i).permutation(ndim)`), and its inverse on the reverse pass.

Use `affine=True`, `batch_norm=True`, `permute='random'`, `seqfrac = 1/seqfrac_inv = 0.25`. The flow's accumulated `logdet` (reverse pass) is combined with a sigmoid log-det correction downstream.
<!-- ORACLE_OP_END op_080_realnvp_flow -->

<!-- ORACLE_OP_START op_090_alphadpi_train -->
## Alpha-DPI training loop, data warmup, and best-state selection

## Alpha-DPI training

Train `q_theta` per frame to minimize the alpha-divergence data-fidelity objective. With `alpha_divergence = 1` (default) this reduces to the KL/ELBO objective. Full loop in `${CLAUDE_SKILL_DIR}/resources/op_090_alphadpi_train.md`.

Setup per frame:
- Build the geometric model, NUFFT forward model, and loss functions. Optimizer: `Adam(lr=1e-4)`. `n_epoch=5000`, `batch_size=2048`, `grad_clip=1e-4`.
- Loss weights: `camp_weight = 1.0`; `cphase_weight = n_cphase / n_camp`; `scale_factor = 1/n_camp`; `logdet_weight = 2*logdet_weight_param/n_camp` (kept for compatibility).

Each epoch:
1. `data_weight = min(10^(-start_order + k/decay_rate), 1.0)` with `start_order=4`, `decay_rate=1000` (log-linear warmup that ramps the data term in).
2. `z ~ N(0,I)` shape `(batch, 4)`; `params_samp, logdet = flow.reverse(z)`; `params = sigmoid(params_samp)`.
3. Render images, run forward model to get `cphase`, `logcamp`.
4. Sigmoid log-det correction: `det_sigmoid = sum(-params_samp - 2*softplus(-params_samp), dim=-1)`; `logdet += det_sigmoid`.
5. `loss_data = 0.5 * (camp_weight*L_ca + cphase_weight*L_cp) / scale_factor`.
6. `logprob = -logdet - 0.5*sum(z^2, dim=1)`.
7. `loss = data_weight*loss_data + logprob`.
8. With `alpha_divergence == 1`: `loss_train = mean(scale_factor * loss)`. (For `alpha != 1` with `beta==0`, reweight by `softmax(-(1-alpha)*loss).detach()` and use `sum(rej_weights*scale_factor*loss)`.)
9. `optimizer.zero_grad(); loss_train.backward(); clip_grad_norm_(params, grad_clip); optimizer.step()`.

Track a smoothed ELBO-like scalar (`elbo_loss = mean(scale_factor*(loss_data+logprob))` for alpha=1) and keep the best model state by the mean of the last `n_smooth=10` logged totals, restoring it after training. Frames with too few closures are skipped with placeholder posteriors.
<!-- ORACLE_OP_END op_090_alphadpi_train -->

<!-- ORACLE_OP_START op_100_importance_resample -->
## Importance-sampling reweighting and physical parameter extraction

## Importance sampling (Step 2 of alpha-DPI)

After training, draw `n_samples = 10000` samples from the trained flow (in eval mode, batched at 2048) and reweight them to correct the approximate posterior. Reference in `${CLAUDE_SKILL_DIR}/resources/op_100_importance_resample.md`.

For each batch (no grad):
1. `z ~ N(0,I)`; `params_samp, logdet = flow.reverse(z)`; `params = sigmoid(params_samp)`.
2. Render images, run forward model.
3. `det_sigmoid = sum(-params_samp - 2*softplus(-params_samp), dim=-1)`; `logdet += det_sigmoid`.
4. `loss_data = camp_weight*L_ca + cphase_weight*L_cp` (note: no `1/scale_factor` division here, unlike training).
5. `logprob = -logdet - 0.5*sum(z^2, dim=1)`; `loss_orig = 0.5*loss_data + logprob`.
6. Collect `neg_log_weights = loss_orig`.

After all batches: `importance_weights = softmax(-neg_log_weights)` (scipy `softmax`). Convert unit params to physical units per the crescent convention: `diameter = 2*(r0 + p0*(r1-r0))`, `width = w0 + p1*(w1-w0)`, `asymmetry = p2`, `pa_deg = 181*(2*p3 - 1)`. The physical parameter matrix columns are `[diameter, width, asymmetry, pa_deg]`. Also compute the importance-weighted mean image if needed. Return `params_physical`, `importance_weights`.
<!-- ORACLE_OP_END op_100_importance_resample -->

<!-- ORACLE_OP_START op_110_pa_aggregation -->
## Per-frame position-angle aggregation and NPZ output writing

## Position-angle aggregation and output

For each frame, reduce the importance-weighted posterior of the position-angle column to a single scalar and assemble the length-10 output vector.

Aggregation (matches the reference posterior-mean reduction):
1. Normalize weights: `w = importance_weights / sum(importance_weights)`.
2. `pa_frame = sum(w * params_physical[:, 3])` — the importance-weighted arithmetic mean of the `pa_deg` column (column index 3). This is the same weighted-mean estimator used for reporting posterior means.
3. Store `pa_frame` at the frame's slot `i` in a length-`n_frames` array.

For degenerate/skipped frames (too few closures), use the placeholder posterior mean (e.g. 0) so the vector stays length 10 and finite.

Write `outputs/{case_id}.npz` with `np.savez` (or `savez_compressed`) containing key `position_angle_deg` as a `float64` array of shape `(10,)`. Ensure all entries are finite before writing. Create `outputs/` if it does not exist. Angles may be left as produced (the estimator is periodic modulo 360); do not rescale or apply extra rotation.
<!-- ORACLE_OP_END op_110_pa_aggregation -->

<!-- ORACLE_OP_START op_120_runtime_probe -->
## Environment and dependency smoke check

## Runtime probe

A cheap non-mutating check to confirm the required stack is importable before launching per-frame training. Run `${CLAUDE_SKILL_DIR}/scripts/op_120_runtime_probe.py` with the run-local python. It reports numpy, scipy, torch (and CUDA availability), and torchkbnufft. If CUDA is available the per-frame training runs on GPU; otherwise it falls back to CPU (much slower but functionally identical). This probe does not change any outputs and can be skipped in production runs.
<!-- ORACLE_OP_END op_120_runtime_probe -->

<!-- ORACLE_OP_START op_130_output_validation -->
## Output NPZ schema and finiteness validation

## Output validation gate

After writing `outputs/{case_id}.npz`, validate it against the contract before finishing. Use `${CLAUDE_SKILL_DIR}/scripts/op_130_output_validation.py <path-to-npz>` with the run-local python. Checks:
- The file loads and contains key `position_angle_deg`.
- Shape is exactly `(10,)`.
- Dtype is `float32` or `float64`.
- All values are finite (no NaN/Inf).

This is a correctness gate; it does not modify the output. If any check fails, fix the aggregation/writing step rather than the validator.
<!-- ORACLE_OP_END op_130_output_validation -->
