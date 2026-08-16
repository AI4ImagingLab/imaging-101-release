---
name: oracle-weather-radar-data-assimilation
description: "Reconstruct three full-resolution 128x128 future VIL radar frames from six past frames and sparse (~10%) noisy future observations, using a pretrained UNet drift model as a flow-matching generative prior with guided Euler-Maruyama sampling and autoregressive rollout."
---

# oracle-weather-radar-data-assimilation Oracle Skill

This skill reconstructs future weather-radar (VIL) frames as an ill-posed inverse problem. Each case provides six past full-resolution conditioning frames, sparse masked+noisy observations of three future frames, a shared binary observation mask (~10% coverage), and a pretrained drift-model checkpoint. The reconstruction fuses the learned generative prior with the sparse measurements through gradient-based guidance during stochastic sampling, then rolls forward autoregressively for three steps. Each numbered section below is an independently removable unit; enable them together for the full standard behavior. Read every section before writing code, because the model-space scaling, forward operator, interpolant coefficients, guidance formula, and output schema are tightly coupled. Bundled scripts live under ${CLAUDE_SKILL_DIR} and must be run with the run-local python from your PATH (or public/envs/runtime). Note the evaluation runtime has torch/numpy/scipy but NOT einops, so the bundled model is written without external reshape libraries.

<!-- ORACLE_OP_START op_010_public_contract -->
## Task contract, case discovery, and submission schema

## Task contract and submission schema

Goal: for each case, reconstruct the three future full-resolution VIL radar frames from six past frames and sparse noisy observations of the three future frames.

**Case discovery.** Read the manifest at `public/visible_data/cases.json`. It lists cases; each case has an `id` (e.g. `case_000`) and an `input_dir` under `public/visible_data/`. Case files:
- `public/visible_data/cases/<case_id>/input_data/raw_data.npz` — the radar arrays.
- `public/visible_data/cases/<case_id>/params_data.json` — imaging parameters.
- `public/visible_data/pretrained_models/latest.pt` — the shared pretrained drift-model checkpoint (one file for all cases).

**Submission.** For every case write `outputs/<case_id>.npz` (relative to the run root). It must contain exactly one array:
- key `reconstructed_frames`, shape **(3, 128, 128)** (the three future frames, no leading batch axis), dtype `float32` (float64 also accepted).

**Hard invariants.**
- Output values must be finite (no NaN/Inf). Casting is not applied for you (`allow_cast: false`), so write the exact declared shape and a floating dtype.
- Frame order is chronological: index 0 = t+1, index 1 = t+2, index 2 = t+3.
- Do not write any other array key; keep the shape exactly (3,128,128).

Process every case listed in the manifest and write one output file per case.
<!-- ORACLE_OP_END op_010_public_contract -->

<!-- ORACLE_OP_START op_020_case_data_loading -->
## Load radar arrays and case parameters

## Loading case arrays and parameters

Load `raw_data.npz` for the case with numpy. It contains three keys (all float32):
- `condition_frames` — shape `(1, 6, 128, 128)`: six past full-resolution frames, pixel values in [0, 1].
- `observations` — shape `(1, 3, 128, 128)`: sparse noisy observations of the three future frames (already masked + noised).
- `observation_mask` — shape `(1, 1, 128, 128)`: binary mask (0/1), the SAME spatial mask applied to all three future frames, ~10% coverage.

Drop the leading batch axis when using them: `condition = condition_frames[0]` → (6,128,128); `obs = observations[0]` → (3,128,128); `mask = observation_mask[0]` → (1,128,128).

Read `params_data.json` for scalars: `image_height`/`image_width` (128), `n_condition_frames` (6), `n_prediction_frames` (3), `observation_mask_ratio` (0.1), `noise_sigma` (0.001), `pixel_value_range` ([0,1]). Use `n_prediction_frames` as the number of autoregressive steps and `noise_sigma` for the noise model. Do NOT expect or require any target/answer array in the case inputs — only the three keys above are present.

Example:
```python
import numpy as np, json
d = np.load(f"public/visible_data/cases/{cid}/input_data/raw_data.npz")
condition = d["condition_frames"][0]
obs = d["observations"][0]
mask = d["observation_mask"][0]
with open(f"public/visible_data/cases/{cid}/params_data.json") as f:
    params = json.load(f)
```
<!-- ORACLE_OP_END op_020_case_data_loading -->

<!-- ORACLE_OP_START op_030_model_space_scaling -->
## Pixel <-> model-space scaling convention

## Model-space scaling (critical convention)

The pretrained drift model was trained in a scaled space, NOT raw [0,1] pixels. Apply this affine map to every input before feeding the model and invert it on the model output.

- To model space: `x_model = (x - 0.5) * 10.0`  → maps [0,1] to [-5, 5].
- Back to pixel space: `x_pixel = x_model / 10.0 + 0.5`.

Scale BOTH the conditioning frames and the sparse observations with `scale_to_model` before sampling. The forward measurement operator and guidance are all computed in model space (the observations are scaled the same way, so mask*x consistency holds). After autoregressive reconstruction, convert the model-space result back to pixel space with `scale_from_model` before writing outputs.

```python
def scale_to_model(x):   return (x - 0.5) * 10.0
def scale_from_model(x): return x / 10.0 + 0.5
```

Skipping or altering these constants (0.5 offset, factor 10) will put the model far outside its trained input distribution and corrupt the reconstruction.
<!-- ORACLE_OP_END op_030_model_space_scaling -->

<!-- ORACLE_OP_START op_040_forward_operator -->
## Sparse masking forward operator and noise model

## Forward measurement model

The measurement model is a sparse binary masking with additive Gaussian noise:

  y = M ⊙ x + η,   η ~ N(0, σ²I),  σ = noise_sigma = 0.001

Build two callables (they operate in model space):
- **operator(x) = x * mask** — element-wise multiply by the binary mask. The mask has shape (1,128,128) and broadcasts against a (B,1,128,128) frame estimate.
- **noiser(x) = x + randn_like(x) * σ** — additive Gaussian noise with σ from params (`noise_sigma`, 0.001).

Use a torch tensor mask (`torch.from_numpy(mask).float().to(device)`) so the operator is differentiable inside guidance. These are used only to compare a clean-frame estimate against the observation during guidance (the observations you load are already masked+noised, so you do not re-apply the forward model to them).

```python
def make_operator(mask):  # mask: torch tensor
    return lambda x: x * mask
def make_noiser(sigma):
    return lambda x: x + torch.randn_like(x) * sigma
```
<!-- ORACLE_OP_END op_040_forward_operator -->

<!-- ORACLE_OP_START op_050_drift_model_unet -->
## UNet drift-model architecture and checkpoint loading

## Learned drift model (generative prior)

The pretrained checkpoint `public/visible_data/pretrained_models/latest.pt` holds a UNet drift model that parameterizes the velocity field of a stochastic interpolant. Instantiate the exact architecture and load `checkpoint['model_state_dict']`.

**Do not depend on `einops`** — it is absent from the evaluation runtime. Use the bundled, einops-free implementation at `${CLAUDE_SKILL_DIR}/scripts/op_050_drift_model_unet.py`, which was verified to load this checkpoint and run a forward pass.

**Architecture (must match the checkpoint):**
- UNet base `dim=128`, `dim_mults=(1,2,2,2)`, `resnet_block_groups=8`.
- Learned sinusoidal time conditioning, `learned_sinusoidal_dim=32` (fourier_dim = 33), time embedding dim = 512.
- Linear attention in down/up stages; full attention at the mid block, `attn_heads=4`, `attn_dim_head=64`.
- No class conditioning (`use_classes=False`, so the `classes`/`label` argument is `None`).
- `init_conv`: Conv2d(**in_channels=7**, 128, kernel 7, padding 3). Out conv: Conv2d(128, **out_channels=1**, 1).

**Input assembly.** The drift model forward is `model(zt, t, label=None, cond=cond)`. Internally it concatenates `zt` (B,1,128,128) with `cond` (B,6,128,128) along channels → 7 channels → UNet. Output is (B,1,128,128). `t` is a 1-D tensor of length B (per-sample time in [0,1]).

**Loading:**
```python
import sys; sys.path.insert(0, "${CLAUDE_SKILL_DIR}/scripts")
from op_050_drift_model_unet import load_drift_model
model = load_drift_model("public/visible_data/pretrained_models/latest.pt", device=device)
```
Run on CUDA if available else CPU. Put the model in eval mode (the loader does this). Wrap sampling drift/model calls appropriately — note guidance needs gradients w.r.t. the state even though model weights are frozen.
<!-- ORACLE_OP_END op_050_drift_model_unet -->

<!-- ORACLE_OP_START op_060_stochastic_interpolant -->
## Stochastic interpolant coefficients (alpha, beta, sigma)

## Stochastic interpolant coefficients

The generative prior is a stochastic interpolant connecting a base sample z0 to the target z1:

  z_t = alpha(t)·z0 + beta(t)·z1 + sigma(t)·epsilon

with these coefficient functions (t a scalar/tensor in [0,1]; broadcast to (B,1,1,1)):
- `alpha(t) = 1 - t`
- `beta(t)  = t^2`   (the beta schedule is quadratic — this is the FlowDAS convention, NOT linear t)
- `sigma(t) = sigma_coef · (1 - t)` with `sigma_coef = 1.0`

Implement as a small holder that widens t to 4-D before scaling:
```python
class Interpolant:
    def __init__(self, sigma_coef=1.0): self.sigma_coef = sigma_coef
    def _wide(self, t): return t[:, None, None, None]
    def alpha(self, t): return self._wide(1 - t)
    def beta(self, t):  return self._wide(t.pow(2))
    def sigma(self, t): return self.sigma_coef * self._wide(1 - t)
```
Only `sigma(t)` is used directly in the sampler (as the diffusion coefficient); `beta=t^2` defines the trained interpolant the model expects. Keep `sigma_coef=1.0`.
<!-- ORACLE_OP_END op_060_stochastic_interpolant -->

<!-- ORACLE_OP_START op_070_taylor_x1_estimate -->
## Second-order SRK clean-frame estimate with MC averaging

## Estimating the clean frame x1 (second-order stochastic RK)

At each sampling step, estimate the clean prediction x1 from the current state `xt` using a second-order stochastic Runge-Kutta (Taylor) extrapolation, then Monte-Carlo average it to reduce guidance-gradient variance.

Given the current drift `bF = model(xt, t, None, cond)` and time `t` widened to (B,1,1,1):

1. Variance term (a scalar function of t): `v(t) = 2/3 - sqrt(t) + (1/3)·sqrt(t)^3`.
2. First estimate: `hat_x1 = xt + bF·(1 - t) + randn_like(xt)·v(t)`.
3. Re-evaluate drift at that estimate at time t=1: `bF2 = model(hat_x1, ones(B), None, cond)` (mark it requires_grad).
4. Corrected estimate (averaged drift): `hat_x1 = xt + (bF + bF2)/2·(1 - t) + randn_like(xt)·v(t)`.
5. Monte-Carlo: repeat step 4 `mc_times = 25` times with fresh noise, returning a LIST of 25 estimates (each with `requires_grad_(True)`).

The drift `bF2` is computed once; only the final noise draw is repeated across the MC samples. Return the list of estimates; the guidance step averages the measurement residual over them. (If mc_times were 1, return a single tensor instead of a list.)

```python
def taylor_est(model, xt, t_wide, bF, cond, mc_times=25):
    v = 2.0/3.0 - t_wide.sqrt() + (1.0/3.0)*(t_wide.sqrt())**3
    hat = xt + bF*(1 - t_wide) + torch.randn_like(xt)*v
    t1 = torch.ones(xt.shape[0], device=xt.device)
    bF2 = model(hat, t1, None, cond=cond).requires_grad_(True)
    ests = []
    for _ in range(mc_times):
        e = xt + (bF + bF2)/2*(1 - t_wide) + torch.randn_like(xt)*v
        ests.append(e.requires_grad_(True))
    return ests
```
<!-- ORACLE_OP_END op_070_taylor_x1_estimate -->

<!-- ORACLE_OP_START op_080_guidance_gradient -->
## Observation-likelihood guidance gradient

## Data-fidelity guidance gradient

Compute the gradient of the observation-likelihood norm w.r.t. the current state `xt`, used to steer sampling toward measurement consistency.

Given the list of MC clean-frame estimates `x1_list` (from the SRK estimate), the sparse observation `y` for this frame (model-space, shape (B,1,128,128)), and the masking `operator`:

1. Average the residual over the MC estimates (operator only; do NOT add measurement noise here — the noiser is only used in the single-tensor branch):
   `difference = mean_i( y - operator(x1_list[i]) )`.
2. `norm = ||difference||` (a scalar; `torch.linalg.norm` over the whole tensor).
3. `norm_grad = autograd.grad(norm, xt, allow_unused=True)[0]`. If it comes back `None`, treat it as 0.

```python
def grad_and_value(xt, x1_list, y, operator):
    diff = sum(y - operator(x1_list[i]) for i in range(len(x1_list))) / len(x1_list)
    norm = torch.linalg.norm(diff)
    g = torch.autograd.grad(norm, xt, allow_unused=True)[0]
    return g, norm
```
The state `xt` must have `requires_grad_(True)` and the drift/estimates must form a differentiable graph back to it for this gradient to be non-None. This gradient is subtracted (scaled) from the Euler-Maruyama update.
<!-- ORACLE_OP_END op_080_guidance_gradient -->

<!-- ORACLE_OP_START op_090_guided_em_sample -->
## Guided Euler-Maruyama sampling loop for one frame

## Guided Euler-Maruyama sampler (single frame)

Reconstruct one future frame by integrating the guided SDE from t=0 to t≈1.

**Time grid.** `ts = linspace(0.0, 0.999, n_steps)` with `n_steps = 500`; `dt = ts[1]-ts[0]`.

**Initialization.** Start from the last conditioning frame as the base state: `xt = base.clone().requires_grad_(True)` where `base` is `cond[:, -1:, :, :]` (shape (B,1,128,128)). `label = None`.

**Per step i (t = ts[i]):**
1. `t = tscalar * ones(B)` on device.
2. Drift `bF = model(xt, t, None, cond=cond)` (requires_grad).
3. `sigma = interpolant.sigma(t)`.
4. `es_x1 = taylor_est(model, xt, interpolant._wide(t), bF, cond, mc_times)` (MC list).
5. `norm_grad, norm = grad_and_value(xt, es_x1, observation, operator)`.
6. Drift-mean update: `mu = xt + bF * dt`.
7. State update (Euler-Maruyama + guidance):
   `xt = mu + sigma * randn_like(mu) * sqrt(dt) - guidance_scale * norm_grad`
   with `guidance_scale = 0.1` (treat `norm_grad = 0` if None).
8. Keep the graph so the next step's gradient works (do not detach xt).

**Return** the drift-mean `mu` from the FINAL step (the denoised estimate), NOT the last noisy `xt`. Shape (B,1,128,128).

Run the whole rollout under `torch.no_grad()` at the outer level is NOT possible because guidance needs autograd; instead let autograd track the state while keeping model weights frozen (eval mode + no optimizer). Constants to preserve exactly: `n_steps=500`, `mc_times=25`, `guidance_scale=0.1`, `tmax=0.999`.
<!-- ORACLE_OP_END op_090_guided_em_sample -->

<!-- ORACLE_OP_START op_100_autoregressive_rollout -->
## Autoregressive multi-frame rollout with sliding window

## Autoregressive rollout (3 frames)

Reconstruct the three future frames one at a time, feeding each prediction back into the conditioning window.

Setup (all in model space, batch axis added → shape (1,·,128,128) on device):
- `cond = condition_scaled` (1,6,128,128).
- `obs = observations_scaled` (1,3,128,128).
- `operator` from the torch mask, `noiser` from `noise_sigma`.

For `step` in 0..(n_prediction_frames-1) (= 0,1,2):
1. `base = cond[:, -1:, :, :]` — the most recent frame in the window is the sampler's starting state.
2. `obs_step = obs[:, step:step+1, :, :]` — this step's sparse observation.
3. `sample = guided_em_sample(model, interpolant, base, cond, obs_step, operator, noiser, n_steps=500, mc_times=25, guidance_scale=0.1)` → (1,1,128,128).
4. Append `sample` to results.
5. If not the last step, slide the window: `cond = torch.cat([cond[:, 1:], sample], dim=1)` — drop the oldest frame, append the new prediction (keeps 6 channels).

Concatenate the three samples along the frame axis → `recon_model` shape (1,3,128,128). Errors propagate through this autoregressive chain, so order and window management matter. Keep the conditioning window length fixed at 6.
<!-- ORACLE_OP_END op_100_autoregressive_rollout -->

<!-- ORACLE_OP_START op_110_output_writing -->
## Inverse-scale and write submission npz

## Writing the submission

1. Move the reconstruction to CPU/numpy and drop the batch axis: `recon = recon_model[0].detach().cpu().numpy()` → shape (3,128,128) in model space.
2. Invert the scaling back to pixel space: `recon = recon / 10.0 + 0.5` (see the scaling convention). Do NOT clip unless needed for finiteness; the reference leaves values unclipped.
3. Cast to `float32` and ensure all values are finite (replace any non-finite with 0.0 as a safety net, though a correct run has none).
4. Save to `outputs/<case_id>.npz` with the single key `reconstructed_frames` and shape exactly **(3, 128, 128)** — no leading batch dimension:
```python
import os, numpy as np
os.makedirs("outputs", exist_ok=True)
recon = (recon / 10.0 + 0.5).astype(np.float32)
np.savez(f"outputs/{cid}.npz", reconstructed_frames=recon)
```
Write one file per case id from the manifest. The array shape and key are validated strictly (no auto-cast), so match them exactly.
<!-- ORACLE_OP_END op_110_output_writing -->

<!-- ORACLE_OP_START op_120_smoke_probe -->
## Cheap end-to-end smoke test

## Smoke test before the full run

The full sampler is expensive (500 EM steps × 25 MC drift evaluations × 3 frames). Before committing to the full settings, validate the pipeline end-to-end with a tiny configuration to catch shape/dtype/device/gradient errors quickly.

Use the bundled probe: `python ${CLAUDE_SKILL_DIR}/scripts/op_120_smoke_probe.py public/visible_data/pretrained_models/latest.pt`. It loads the checkpoint, runs a forward pass, and runs a handful of guided EM steps on a single frame with `n_steps=4, mc_times=2` to confirm the guidance gradient is non-None and the output shape is (1,1,128,128). This does not produce a valid submission — only a health check. Once it passes, run the real reconstruction with `n_steps=500, mc_times=25, guidance_scale=0.1`.
<!-- ORACLE_OP_END op_120_smoke_probe -->

<!-- ORACLE_OP_START op_130_submission_validation -->
## Validate output shape, dtype, and finiteness

## Final submission validation

After writing every `outputs/<case_id>.npz`, verify each against the contract before finishing. For each case id in `public/visible_data/cases.json`:
- The file `outputs/<case_id>.npz` exists.
- It contains the key `reconstructed_frames`.
- The array shape is exactly `(3, 128, 128)`.
- The dtype is floating (`float32` or `float64`).
- All values are finite (no NaN/Inf).

Use the bundled validator: `python ${CLAUDE_SKILL_DIR}/scripts/op_130_submission_validation.py`. It reads the manifest and checks each expected output; it exits non-zero and prints the first failure if any check fails. Fix and re-run before considering the task complete.
<!-- ORACLE_OP_END op_130_submission_validation -->
