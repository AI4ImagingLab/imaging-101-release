---
name: oracle-era5-tensorvar
description: "Oracle guidance for reconstructing the bundled ERA5 analysis tensor using the task's public observation arrays, bundled neural checkpoints, and the specified normalized output contract."
---

# oracle-era5-tensorvar Oracle Skill

Use the operation sections independently: public I/O rules, batch/axis conventions, checkpoint-compatible Tensor-Var modules, latent quadratic objective details, assimilation execution, and submission validation are separated so each can be ablated without depending on hidden files or network access.

## Public case and submission contract

### Public case and submission contract
There is one public case, `case_000`. Read inputs from run-local `public/visible_data/cases/case_000/input_data/` and case metadata from `public/visible_data/cases/case_000/params_data.json` (or from the parent of `input_data` if resolving generically). Write exactly `outputs/case_000.npz` from the run root.

The `.npz` submission must contain one required array named `state`. Its shape is `(5, 5, 64, 32)` with axes `(time, physical_channel, grid_height, grid_width)`. Acceptable dtypes are `float32` or `float64`; keep all values finite. The values are in the same normalized space as the input observation history, not denormalized physical units.

The five physical channels, in order, are `geopotential`, `temperature`, `humidity`, `wind_u`, and `wind_v`. The task window length is five timesteps and the history length per timestep is ten observations stacked over the five physical channels, so each inverse-observation input frame has 50 channels.

## Load batch-first raw arrays and metadata

### Load batch-first raw arrays and metadata
Load `raw_data.npz` from the case input directory and preserve all keys. The file is batch-first with a single sample on axis 0. For the assimilation calculation, select sample index 0 from each array: `obs_history` becomes `(5, 50, 64, 32)`, `lat_weight_matrix` becomes `(5, 64, 32)`, and `min_val`/`max_val` become `(5,)`.

`obs_history` is already normalized, noisy observation history. Do not reorder or denormalize it before feeding the inverse-observation model. Its axes after sample selection are `(time, history_len*channel, grid_height, grid_width)`. Metadata is informational for geometry and channel names; solver hyperparameters are fixed by the method, not read from metadata.

## Validate and load bundled checkpoints

### Validate and load bundled checkpoints
Use only the four checkpoint files bundled under the case input `checkpoints/` directory: `forward_model.pt`, `C_forward.pt`, `inverse_model.pt`, and `z_b.pt`. Verify all four files exist before building the model so missing-input failures are explicit.

Instantiate the checkpoint-compatible forward model and inverse model before loading state dictionaries. Load `forward_model.pt` strictly into the forward model. Load `C_forward.pt` separately as the forward model's latent dynamics matrix attribute. Load `inverse_model.pt` strictly into the inverse model. Load `z_b.pt` as the latent background mean vector. Use `torch.load(..., map_location=...)`; if the runtime PyTorch supports `weights_only=True`, use it for checkpoint loading.

Select `cuda` only if available; otherwise CPU is valid. Set NumPy and Torch seeds to 0 for deterministic surrounding behavior. Put neural inputs and modules on the same device.

## Inverse-observation transformer architecture

### Inverse-observation transformer architecture
The inverse-observation network maps each `(50, 64, 32)` observation-history tensor to a `(5, 64, 32)` pseudo-state. Its wrapper contains a `K_O` module and frozen `K_S`/`K_S_preimage` copies for checkpoint compatibility; the forward pass returns only `K_O(obs)`.

`K_O` uses a Restormer-style stack with input channels `50` and output channels `5`. The channel widths are `[100, 200, 100, 5]`, the attention heads are `[5, 10, 5, 1]`, each of four stages has two transformer blocks, layer normalization is with bias, the feed-forward expansion factor is `2.66`, and convolution bias is `False`. The patch embed is a `3x3` convolution with stride 1 and padding 1. Between transformer stages use simple convolutional `Upsample_Flex` layers: `100→200`, `200→100`, then `100→5`, each a single `3x3` convolution with stride 1, padding 1, and no bias; these layers change channel count but do not change spatial resolution.

Each transformer block applies residual attention after layer normalization, then residual feed-forward after layer normalization. Attention forms `q,k,v` with a `1x1` convolution followed by a depthwise `3x3` convolution, normalizes q and k along the spatial-token dimension, applies per-head learned temperature, softmaxes along the key dimension, and projects back with a `1x1` convolution. The feed-forward block uses `1x1` projection to `2*hidden_features`, depthwise `3x3`, chunks in half, applies `gelu(x1) * x2`, and projects out by `1x1`.

## State encoder and latent feature convention

### State encoder and latent feature convention
The state encoder maps a normalized `(B, 5, 64, 32)` state-like tensor to a latent vector of dimension `512`. Use the exact convolution, pooling, and skip-capture order expected by the checkpoint.

The filter dimensions are `[32, 64, 128, 256]`. The encoder layers are: `Conv2d(5,32,kernel=9,stride=1,padding=4)`, average-pool by 2 and ReLU; `Conv2d(32,32,kernel=5,stride=1,padding=2)`, ReLU then average-pool by 2; `Conv2d(32,64,kernel=3,stride=1,padding=1)`, average-pool by 2 and ReLU; `Conv2d(64,128,kernel=3,stride=1,padding=1)`, average-pool by 2 and ReLU; `Conv2d(128,256,kernel=3,stride=1,padding=1)`, ReLU; flatten; `Linear(2048,512)`. The flattened dimension is `256*(64/16)*(32/16)=2048`.

When `return_encode_list=True`, capture cloned skip tensors in this order: the input state, then each convolution output before the subsequent pooling/ReLU transform. The returned feature and skip list must come from the same pseudo-state produced by the inverse-observation network; those skips are later used for decoding the optimized latent.

## Latent decoder with encoder skip connections

### Latent decoder with encoder skip connections
Decode optimized latent vectors through the state decoder using the encoder skip list captured from the inverse-observation pseudo-state. This is not an optional cosmetic step: the skip tensors align the decoder with the spatial feature hierarchy and materially affect the reconstructed `(5,64,32)` normalized state.

The decoder starts with `Linear(512,2048)`, ReLU, and reshape to `(B,256,4,2)`. With skip tensors, add the deepest skip before the first transposed convolution, then upsample by factor 2 between levels and add the next shallower skip before each transposed convolution. The transposed convolutions are `ConvTranspose2d(256,128,3,1,1)`, `ConvTranspose2d(128,64,3,1,1)`, `ConvTranspose2d(64,32,3,1,1)`, `ConvTranspose2d(32,32,5,1,2)`, and `ConvTranspose2d(32,5,9,1,4)`, with ReLU after the first four. Finish with `1x1 Conv2d(5,128)`, ReLU, then `1x1 Conv2d(128,5)`.

## Latent dynamics and fallback precision matrices

### Latent dynamics and fallback precision matrices
The latent dimension is `D=512`. The feature-space dynamics matrix is loaded from `C_forward.pt` and has shape `(512,512)`. A single latent forward step in the wrapper is `z @ C_forward`, while the quadratic dynamics term uses `F.T @ z_{t-1}` when `F` is the NumPy copy of `C_forward`.

Use identity fallback information matrices because no calibrated matrices are bundled: `B = 0.1 * I_512`, `R = I_512`, and `Q = 0.1 * I_512`. These matrices are precision/information matrices used directly inside quadratic forms, not covariance matrices to invert at runtime. Store them as NumPy arrays for the QP.

## Create the observation feature sequence

### Create the observation feature sequence
Run the inverse-observation model on the selected `obs_history` tensor for the full five-step window. The tensor supplied to the model has shape `(5,50,64,32)` and dtype `torch.float32` on the selected device; the five time steps are treated as the batch dimension for the neural network call. The result is `inv_seq` with shape `(5,5,64,32)`.

Immediately encode `inv_seq` with the state encoder using `return_encode_list=True`. This gives `seq_z` with shape `(5,512)` and one shared list of skip tensors whose leading dimension is the five time steps. Convert `seq_z` to CPU NumPy for the QP, but keep the skip tensors on the model device for decoding. This sequence `seq_z` is the observation feature target \(\tilde z_t\) in the objective.

## Solve the Tensor-Var latent quadratic program

### Solve the Tensor-Var latent quadratic program
Solve one convex unconstrained QP over a variable `pred_seq_z` of shape `(T,D)` with `T=5` and `D=512`. The objective is the sum of: a background term only at `t=0`, an observation-feature term at every timestep, and a latent-dynamics term for `t=1..T-1`.

Use the exact dynamics penalty schedule `penalty = np.linspace(0.1, 1, T - 1)[::-1]`, which for `T=5` gives `[1.0, 0.7, 0.4, 0.1]`. For `t=0`, add `(z_0 - z_b)^T B (z_0 - z_b)` and `(z_0 - seq_z[0])^T R (z_0 - seq_z[0])`. For `t>0`, add `(z_t - seq_z[t])^T R (z_t - seq_z[t])` and `(z_t - F.T @ z_{t-1})^T (penalty[t-1] * Q) (z_t - F.T @ z_{t-1])`. There are no constraints.

A direct `cvxpy` implementation with `cp.Variable((T,D))`, `quad_form`, `cp.Problem(cp.Minimize(cost), []).solve()`, and then `pred_seq_z.value` reproduces the intended solver behavior. Use the runtime environment's installed QP backend; no network installation is needed.

## Run the single five-step assimilation window

### Run the single five-step assimilation window
Use one assimilation window with `assimilation_window=5` and `total_steps=5`. The outer loop therefore runs exactly once with `start=0` and `end=5`; do not roll shorter overlapping windows or average multiple windows. The input `obs_history` already contains the five histories needed for the five analysis timesteps.

Within `torch.no_grad()`: compute the inverse-observation pseudo-states, encode them and keep the skip list, solve the latent QP, convert the QP result back to a `torch.float32` tensor on the model device, and decode with the same skip list. Detach the decoded tensor to CPU NumPy. The concatenated trajectory shape after the single window is `(5,5,64,32)`.

Recommended runtime discipline: set `np.random.seed(0)` and `torch.manual_seed(0)`, select sample index 0, use `matplotlib` only if generating optional diagnostics, and avoid training or updating checkpoint weights.

## Package the normalized trajectory submission

### Package the normalized trajectory submission
Create `outputs/` if needed and save the final normalized trajectory to `outputs/case_000.npz` with key `state`. Do not save optional diagnostics instead of the required file, and do not change the key name. A `float32` trajectory is suitable; `float64` is also schema-valid, but do not rely on automatic casting by validation.

Before writing, check the array shape is exactly `(5,5,64,32)` and all entries are finite. If the decode output has an accidental leading sample axis or channel/time swap, fix the upstream shape handling rather than reshaping blindly. Do not denormalize using `min_val` or `max_val` for the submission.

## Optional normalized-space metric sanity checks

### Optional normalized-space metric sanity checks
If a reference comparison array is available in a run-local workspace during development, compute diagnostics in normalized space with the same axes `(T,C,H,W)`. Per-channel normalized cross-correlation uses flattened channel/time/grid values after subtracting each channel's mean. Per-channel NRMSE is RMSE divided by that channel's value range in the comparison array. Weighted NRMSE uses the loaded `(C,64,32)` latitude weights: for each time and channel compute `|| (estimate-reference) * weight ||_2 / || reference * weight ||_2`, then average over time.

These diagnostics are only a sanity gate; the submission contract remains the `.npz` file with finite normalized `state`. Do not read hidden files or non-public task directories to obtain comparison data.
