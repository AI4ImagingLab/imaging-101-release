---
name: oracle-eht-black-hole-tomography
description: "Reconstruct the static 3D volumetric emission field of a black hole accretion flow from sparse time-series interferometric images by fitting a coordinate-based neural radiance field (BH-NeRF) that is warped by Keplerian orbital dynamics and rendered through pre-computed gravitationally lensed ray paths. The single required output is a 64x64x64 emission cube written to outputs/{case_id}.npz under key emission_3d."
---

# oracle-eht-black-hole-tomography Oracle Skill

This skill encodes a faithful BH-NeRF tomography pipeline. Each operation is an independently removable unit covering one convention or step: the public output contract, run-local case loading, the geometry/units conventions, positional encoding, the MLP emission network with its sigmoid offset, the Keplerian inverse-rotation velocity warp, the radial/z support mask, the radiative-transfer volume render, the L2 image loss, the Adam training loop with a polynomial learning-rate schedule, the final regular-grid emission sampling, output serialization, and a finiteness/shape validator. A solver enabling all operations can reproduce the standard reconstruction; ablating individual operations yields meaningful behavior variants.

<!-- ORACLE_OP_START op_010_contract -->
## Public task contract and submission protocol

## Task contract

Goal: recover the static 3D volumetric emission field of an orbiting black-hole hotspot from a set of noisy time-series images, and write it as a single cube per case.

### Cases
- Cases are listed in `public/.../cases.json`-style metadata available under `public/`. Each case has an `id` (e.g. `case_000`) and an `input_dir` containing `params_data.json` and `input_data/raw_data.npz`.
- Process every listed case independently.

### Required output (one file per case)
- Path: `outputs/{case_id}.npz` relative to the run root.
- Array key `emission_3d`, required.
- Shape exactly `(64, 64, 64)`.
- Dtype `float32` or `float64`.
- All values must be finite (no NaN/Inf). Casting is not applied by the grader, so emit the correct dtype yourself.
- Only `emission_3d` is scored; any auxiliary arrays are ignored but should not break loading.

### Hard invariants
- The cube is the static emission field sampled on a regular Cartesian grid spanning the image field of view (see geometry conventions). It is NOT a per-frame image stack.
- Do not read or depend on any files outside `public/`, `workspace/`, `outputs/`, `logs/agent/`, and `${CLAUDE_SKILL_DIR}`.

Keep the writer aligned with this contract; the grid sampling and output-writing operations implement it.
<!-- ORACLE_OP_END op_010_contract -->

<!-- ORACLE_OP_START op_020_case_loading -->
## Load case parameters and raw ray/observation arrays

## Loading a case

For each case read two files from its run-local `input_dir`.

### `params_data.json` (training/geometry hyperparameters)
Use these keys (values shown are the public case):
- `fov_M` = 24.0, `num_alpha` = 48, `num_beta` = 48, `ngeo` = 80
- `emission_resolution` = 64
- `net_depth` = 4, `net_width` = 128, `posenc_deg` = 3
- `n_iters` = 2000, `lr_init` = 1e-4, `lr_final` = 1e-6, `batch_size` = 8
- `rmin_M` = 4.0, `rmax_M` = 12.0, `z_width_M` = 4.0
- `spin` = 0.0, `inclination_deg` = 60.0, `orbit_radius_M` = 8.0, `n_frames` = 40, `noise_std` = 0.01
Read values from the JSON rather than hard-coding, but fall back to these defaults if a key is absent.

### `input_data/raw_data.npz`
Load with numpy and build float32 torch tensors:
- Stack `ray_x, ray_y, ray_z` -> `ray_coords` of shape `(3, 48, 48, 80)` (axis 0 is [x,y,z]).
- `Omega` -> `(48,48,80)` Keplerian angular velocity per ray point.
- `g_doppler` -> `(48,48,80)` Doppler factor.
- `ray_dtau` -> `dtau` `(48,48,80)` Mino-time differential.
- `ray_Sigma` -> `Sigma` `(48,48,80)` metric factor (= r^2).
- `ray_t_geo` -> `t_geo` `(48,48,80)` coordinate time along the geodesic (slow-light).
- `t_frames` -> `(40,)` float32 frame times (units of M).
- `images_noisy` -> `images` `(40,48,48)` float32 observed frames (training targets).
- Scalars `fov_M`, `t_start_obs`, `t_injection` -> python floats.

These arrays feed the forward model; keep names/axes consistent with later operations.
<!-- ORACLE_OP_END op_020_case_loading -->

<!-- ORACLE_OP_START op_030_geometry_units -->
## Coordinate units, domain scale, and array-axis conventions

## Geometry and unit conventions

All spatial quantities are in geometric units of `M` (GM/c^2); all times are in units of M with `GM_c3 = 1.0` (no extra unit conversion).

### Domain scale for the network
- `scale = fov_M / 2.0` (= 12.0 for the public case).
- Network inputs are normalized by dividing world coordinates by `scale` so the field-of-view maps to roughly [-1, 1].

### Ray-tensor layout
- `ray_coords` has axis order `(3, num_alpha, num_beta, ngeo)`; the leading 3 are [x, y, z].
- The per-ray sample axis (`ngeo`, last axis after moving channels) is the integration axis for volume rendering.

### Emission grid
- The final cube is sampled on `linspace(-fov_M/2, fov_M/2, emission_resolution)` along each of x, y, z with `numpy.meshgrid(..., indexing='ij')`, giving coordinate stack `(3, R, R, R)` with R=64.
- The same world bounds `[-fov_M/2, +fov_M/2]` define interpolation extents and the support mask.

Keep these conventions identical across forward rendering and final sampling; mismatched axis order or scale corrupts the reconstruction.
<!-- ORACLE_OP_END op_030_geometry_units -->

<!-- ORACLE_OP_START op_040_positional_encoding -->
## Sinusoidal positional encoding of coordinates

## Positional encoding

Before the MLP, expand the normalized 3D coordinate with a Fourier positional encoding of degree `posenc_deg` (L = 3 frequency bands).

Algorithm for input `x` of shape `(*, 3)`:
1. If `deg == 0`, return `x` unchanged.
2. `scales = 2.0 ** arange(deg)` -> `[1, 2, 4]` for deg=3.
3. `xb = (x[..., None, :] * scales[:, None]).reshape(*x.shape[:-1], -1)` giving shape `(*, 3*deg)`.
4. `four_feat = sin(concat([xb, xb + 0.5*pi], dim=-1))` (using cos(x)=sin(x+pi/2)).
5. Return `concat([x, four_feat], dim=-1)`.

Output width is `3 + 2*3*deg` = 21 channels for deg=3. This width must match the MLP `in_features` and, for the skip connection, the re-concatenated input width.
<!-- ORACLE_OP_END op_040_positional_encoding -->

<!-- ORACLE_OP_START op_050_nerf_mlp -->
## BH-NeRF MLP with skip connection and sigmoid-offset activation

## Emission network

The static emission is a coordinate MLP `e0(x) = sigmoid(MLP(posenc(x/scale)) - 10)`.

### MLP architecture
- `net_depth = 4` hidden layers, `net_width = 128`, single output channel.
- Linear layers with ReLU after each hidden layer.
- Skip connection: at layer index `net_depth // 2` (= 2), concatenate the original encoded input back in, so that layer's input dim is `net_width + in_features`.
- Weight init: He/Kaiming uniform with `nonlinearity='relu'`; biases initialized to zero (including the output layer).
- Output layer is a plain Linear (no activation) producing logits.

### Activation / offset (critical)
- Emission = `sigmoid(net_output[..., 0] - 10.0)`. The `-10.0` bias keeps initial emission near zero so the field starts dark and grows where data demands it. Do not omit the offset.

### Forward contract
Given warped coordinates of shape `(*, 3)`:
1. Replace non-finite (pre-injection) rows with zeros before the network (track a validity mask).
2. Divide by `scale`, apply positional encoding, run the MLP, apply `sigmoid(x-10)`.
3. Apply the support mask, then multiply by the validity mask (float) to zero pre-injection samples.

See resource for compact pseudocode.
<!-- ORACLE_OP_END op_050_nerf_mlp -->

<!-- ORACLE_OP_START op_060_keplerian_warp -->
## Keplerian inverse-rotation velocity warp with pre-injection masking

## Keplerian velocity warp (temporal dynamics)

The static field is shared across time by inverse-rotating query coordinates so that orbital shearing is undone before the NeRF lookup.

### Time in M units
`t_M = (t_frame - t_start_obs) / GM_c3 + t_geo - t_injection`, with `GM_c3 = 1.0`. `t_geo` is the per-sample slow-light coordinate time.

### Rotation angle and masking
- `theta_rot = t_M * Omega` per ray sample (radius-dependent because `Omega = sqrt(M)/r^{3/2}`).
- Where `t_M < 0` (before hotspot injection), set `theta_rot = NaN`; these samples are later masked to zero emission.

### Rotation axis (critical)
- Use a FIXED rotation axis `[0, 0, 1]` (z-axis). The axis is NOT optimized in the standard solver, despite multi-init hints in the background text. Keeping it fixed at z is what reproduces the reference behavior.

### Applying the rotation
- Build the inverse rotation `R(axis, -theta_rot)` via the Euler-Rodrigues formula (per-sample, batched over `theta_rot`).
- Apply `warped = einsum('ij...,j...->...i', R, coords)` where `coords` is `(3, *spatial)`; result is `(*spatial, 3)`.

At the final grid-sampling step, `Omega` and `t_geo` are set to zero and `t_frame = t_start_obs = t_injection = 0`, so `t_M = 0`, `theta_rot = 0`, and the warp is identity (no masking). See resource for the rotation-matrix formula.
<!-- ORACLE_OP_END op_060_keplerian_warp -->

<!-- ORACLE_OP_START op_070_support_mask -->
## Radial and z-width emission support mask

## Support / unsupervised-region mask

After computing emission, zero it outside the physically recoverable shell so the network does not place flux where rays give no constraint.

Given coordinates with components x, y, z:
- `r_sq = x^2 + y^2 + z^2`
- `mask = (r_sq >= rmin^2) & (r_sq <= rmax^2) & (abs(z) <= z_width)`
- `emission = emission * mask.float()`

Use `rmin = rmin_M` (= 4.0), `rmax = rmax_M` (= 12.0), `z_width = z_width_M` (= 4.0) from the case params. Apply this mask in BOTH per-frame rendering and the final 64^3 grid sampling, using the (unwarped) query coordinates for the radius test as in the forward pass. This mask largely defines the geometric extent of the final cube (an equatorial annular shell).
<!-- ORACLE_OP_END op_070_support_mask -->

<!-- ORACLE_OP_START op_080_volume_render -->
## Radiative-transfer volume rendering along lensed rays

## Volume rendering

Integrate emission along each pre-computed gravitationally lensed ray to form a pixel image:

`I = sum_i ( g_i^2 * emission_i * dtau_i * Sigma_i )` summed over the last (ngeo) axis.

Implementation: `image = (g**2 * emission * dtau * Sigma).sum(dim=-1)`.
- `emission`, `g` (`g_doppler`), `dtau`, `Sigma` all have ray shape `(num_alpha, num_beta, ngeo)`.
- Result image has shape `(num_alpha, num_beta)` = `(48, 48)`.
- `g^2` applies Doppler boosting, `dtau` is the Mino-time differential step, and `Sigma = r^2` is the Schwarzschild metric factor.

This renderer is used every training iteration (per sampled frame) and when producing predicted movies. Do not add extinction/attenuation terms — the model is pure emission accumulation.
<!-- ORACLE_OP_END op_080_volume_render -->

<!-- ORACLE_OP_START op_090_image_loss -->
## L2 image-plane data loss

## Data loss

The standard configuration (`measurement_type = 'image'`) fits rendered images directly to the noisy observed frames with a summed squared-error loss:

`loss = sum( ((pred_image - target_image) / sigma)^2 )` with `sigma = 1.0`.

- Computed per sampled frame against the corresponding `images[idx]` target.
- Summed over all pixels (not averaged), then averaged over the frames in the minibatch (see training loop).
- No regularization term is added in the standard run.

(Alternative losses exist in the reference — a lightcurve total-flux L2 and a complex-visibility chi-squared — but the standard image-domain run uses the per-pixel image L2 above.)
<!-- ORACLE_OP_END op_090_image_loss -->

<!-- ORACLE_OP_START op_100_training_loop -->
## Adam training loop with polynomial LR schedule and gradient clipping

## Training loop

Fit the MLP weights only (rotation axis stays fixed at z).

### Setup
- Seed both `torch.manual_seed(42)` and `numpy.random.seed(42)` for reproducibility.
- Build the model with `scale = fov_M/2`, `rmin=rmin_M`, `rmax=rmax_M`, `z_width=z_width_M`, `posenc_deg`, `net_depth`, `net_width`.
- Optimizer: `Adam(model.parameters(), lr=lr_init)` with `lr_init = 1e-4`.
- Move all ray tensors and target images to the compute device once.

### Polynomial learning-rate schedule
Use a `LambdaLR` with
```
t = min(step / max(n_iters, 1), 1.0)
factor = (1 - t) * 1.0 + t * (lr_final / lr_init)
```
so the effective LR decays linearly from `lr_init` (1e-4) to `lr_final` (1e-6) over `n_iters` (2000) steps.

### Per-iteration step
1. `optimizer.zero_grad()`.
2. Sample `batch_size` (=8) frame indices without replacement via `numpy.random.choice(n_frames, min(batch_size, n_frames), replace=False)`.
3. For each sampled frame index: forward the model at `t_frames[idx]` to get ray emission, volume-render to a `(48,48)` image, and accumulate the image L2 loss against `images[idx]`.
4. `total_loss = accumulated_loss / len(batch_idx)`; `total_loss.backward()`.
5. Clip gradients with `clip_grad_norm_(params, max_grad_norm=1.0)`.
6. `optimizer.step()`, then `scheduler.step()`.
7. Record `total_loss.item()` in a loss history.

Run for `n_iters` iterations. After training, the trained model is sampled to produce the final cube.
<!-- ORACLE_OP_END op_100_training_loop -->

<!-- ORACLE_OP_START op_110_emission_grid_sampling -->
## Sample trained NeRF on the regular 64^3 emission grid

## Final emission sampling

Produce the scored cube by evaluating the trained network on a static regular grid (no temporal warp).

Steps (under no-grad / eval mode):
1. `grid_1d = linspace(-fov_M/2, fov_M/2, resolution)` with `resolution = emission_resolution` (= 64).
2. `xx, yy, zz = meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')`; stack into `coords` of shape `(3, 64, 64, 64)` as float32.
3. Provide zero `Omega` and zero `t_geo` grids of shape `(64,64,64)`, and call the model with `t_frame = 0.0`, `t_start_obs = 0.0`, `t_injection = 0.0`, and the fixed rotation axis `[0,0,1]`.
   - With these zeros, `t_M = 0`, so the warp is identity and nothing is NaN-masked; only the radial/z support mask shapes the output.
4. The model output has shape `(64, 64, 64)`; move to CPU numpy.

This array is the `emission_3d` payload. Do not re-normalize or threshold it beyond what the model+mask already apply.
<!-- ORACLE_OP_END op_110_emission_grid_sampling -->

<!-- ORACLE_OP_START op_120_output_writing -->
## Serialize emission cube to the required npz path

## Writing the submission

For each case write exactly one file:
- Path: `outputs/{case_id}.npz` relative to the run root (create `outputs/` if missing).
- Save with `numpy.savez` (or `savez_compressed`) using key `emission_3d`.
- Ensure dtype is `float32` (or `float64`); convert before saving since the grader does not cast.
- Ensure the array shape is exactly `(64, 64, 64)` and contains only finite values; replace any stray NaN/Inf with 0.0 before writing.

Example: `numpy.savez('outputs/'+case_id+'.npz', emission_3d=emission.astype('float32'))`.

Do not write to any path outside `outputs/`.
<!-- ORACLE_OP_END op_120_output_writing -->

<!-- ORACLE_OP_START op_130_output_validation -->
## Validate output shape, dtype, and finiteness

## Output validation gate

After writing each `outputs/{case_id}.npz`, re-open it and confirm the contract before finishing:
- The file loads and contains key `emission_3d`.
- `emission_3d.shape == (64, 64, 64)`.
- `emission_3d.dtype` is float32 or float64.
- `numpy.isfinite(emission_3d).all()` is True.

If any check fails, fix the producing step and rewrite the file. Use the bundled checker on the run root. This is a guardrail; it does not alter the reconstruction.
<!-- ORACLE_OP_END op_130_output_validation -->
