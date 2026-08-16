---
name: oracle-spectral-snapshot-compressive-imaging
description: "Reconstruct a 256x256x31 hyperspectral cube from a single 2D coded-aperture snapshot spectral (SD-CASSI) measurement using a Plug-and-Play GAP solver with total-variation and optional learned denoiser priors, and write the cube to the required npz output."
---

# oracle-spectral-snapshot-compressive-imaging Oracle Skill

This skill reconstructs a hyperspectral data cube from a single compressed CASSI measurement. It covers the public output contract, run-local input loading, the single-disperser dispersion geometry and sensing-matrix construction, the forward/transpose measurement operators, the accelerated Generalized Alternating Projection (GAP) data-consistency update, the total-variation prior, an optional learned band-window CNN denoiser prior with its hybrid iteration schedule, and final output assembly and validation. Each operation is independently removable; the total-variation GAP path alone yields a valid, finite cube, while the learned denoiser is an optional quality enhancement that requires provided weights.

<!-- ORACLE_OP_START op_010_output_contract -->
## Output contract and submission protocol

Produce exactly one output file per case at run-local path `outputs/{case_id}.npz` (base = run root). The visible case is `case_000`; iterate over the cases listed in `public/.../cases.json` (each case has an `input_dir` and an `expected_output` of `outputs/{case_id}.npz`).

Each output npz MUST contain array key `hyperspectral_cube` with:
- shape exactly `(256, 256, 31)` = `(H, W, nC)` = (height, width, spectral channels),
- dtype `float32` or `float64`,
- all-finite values (no NaN/Inf); casting is not allowed by the checker, so build the array already in an accepted dtype.

Spectral channel index `k` (0..30) corresponds to wavelength `400 + 10*k` nm (400-700 nm, 10 nm spacing). The spatial axes are the un-dispersed scene coordinates (see the dispersion-geometry section). Save with `numpy.savez` (uncompressed is fine) using exactly the key name `hyperspectral_cube`. Do not add a leading batch dimension to the saved cube.

Run everything with the run-local runtime interpreter `public/envs/runtime/.venv-posix/bin/python` (available packages include numpy, scipy, scikit-image, opencv, torch).
<!-- ORACLE_OP_END op_010_output_contract -->

<!-- ORACLE_OP_START op_020_load_inputs -->
## Load measurement, mask, and imaging parameters

For each case, read the raw inputs from the case `input_dir`:
- `raw_data.npz` holds `measurement` with shape `(1, 256, 286)` float32 and `mask` with shape `(1, 256, 256)` float32 (binary 0/1 coded aperture).
- `params_data.json` holds imaging parameters: `r=256` (height H), `c=256` (width W), `nC=31` (spectral channels), `step=1` (dispersion shift per channel, integer pixels), `wavelength_start_nm=400`, `wavelength_end_nm=700`, `wavelength_step_nm=10`.

Loading rules that the standard pipeline relies on:
- Squeeze the leading batch dimension: `meas = raw['measurement'][0]`, `mask2d = raw['mask'][0]`.
- Cast both to `float64` for the numerics of the solver (`.astype(np.float64)`), then produce the final saved cube in float32/float64 per the output contract.
- The measurement width `286 = W + step*(nC-1) = 256 + 1*30` is the dispersed detector width; keep `meas` at full `(256, 286)`.
- Ignore any other `.mat` files present in the case folder; the solver only needs `measurement`, `mask`, and the JSON params.

Do not assume a ground-truth cube is available at solve time; the reconstruction is driven only by `meas`, `mask2d`, and the params.
<!-- ORACLE_OP_END op_020_load_inputs -->

<!-- ORACLE_OP_START op_030_dispersion_geometry -->
## Single-disperser dispersion shift geometry

The SD-CASSI prism shifts channel `k` (0-indexed) horizontally by `d_k = k*step` pixels, with `step=1`. Two operators encode this and are used throughout the solver:

`shift(inputs, step)`: maps an un-dispersed cube `(H, W, nC)` to a dispersed cube `(H, W+(nC-1)*step, nC)` by placing band `i` into columns `[i*step : i*step+W]`:
```
output = zeros((H, W+(nC-1)*step, nC))
for i in range(nC):
    output[:, i*step : i*step+W, i] = inputs[:, :, i]
```
(Note: the code writes `i*step : i*step+row`; for square scenes H==W==256 this equals `i*step+W`.)

`shift_back(inputs, step)`: reverses dispersion on a dispersed cube `(H, W+(nC-1)*step, nC)` back to `(H, W, nC)` by rolling each band left by `step*i` then cropping:
```
row, col, nC = inputs.shape
for i in range(nC):
    inputs[:, :, i] = np.roll(inputs[:, :, i], -step*i, axis=1)
output = inputs[:, 0 : col - step*(nC-1), :]
```
Important conventions: `shift_back` mutates its input array in place via the `np.roll` assignment and crops the trailing `step*(nC-1)` columns. Use `step=1` everywhere for this task. Getting these two operators and the shift direction/crop right is essential; they define the mapping between detector coordinates and scene coordinates. A reusable helper is provided.
<!-- ORACLE_OP_END op_030_dispersion_geometry -->

<!-- ORACLE_OP_START op_040_sensing_matrix -->
## Build the 3D block-diagonal sensing matrix Phi

The sensing matrix `Phi` is the coded aperture mask replicated and shifted per channel. From the squeezed 2D mask `mask2d` of shape `(r, c)=(256,256)`:
```
Phi = np.zeros((r, c + step*(nC-1), nC))   # (256, 286, 31)
for i in range(nC):
    Phi[:, i:i+c, i] = mask2d
```
Every channel slab `Phi[:, :, i]` is the same binary mask placed at horizontal offset `i` (with `step=1`). `Phi` shares the dispersed width `286` with the measurement. This block-diagonal-by-shift structure is what makes the GAP data-consistency projection analytic and cheap.

Do not re-seed randomness or regenerate the mask; use the provided `mask2d` exactly. A helper builder is provided.
<!-- ORACLE_OP_END op_040_sensing_matrix -->

<!-- ORACLE_OP_START op_050_forward_transpose -->
## Forward and transpose measurement operators

The SD-CASSI forward model integrates masked, dispersed channels onto the 2D detector. With a dispersed cube `x` and `Phi` both shaped `(H, W+step*(nC-1), nC)`:

Forward `A(x, Phi)`: `y = np.sum(x * Phi, axis=2)`  -> shape `(H, W+step*(nC-1))` = `(256, 286)`.

Transpose `At(y, Phi)`: `x = np.repeat(y[:, :, None], nC, axis=2) * Phi` -> shape `(H, W+step*(nC-1), nC)`.

These match the measurement equation `y(m,n) = sum_k Phi(m, n-d_k) x(m, n-d_k, k)` once `x` is expressed in dispersed coordinates. `A` and `At` operate in the dispersed frame; convert to/from scene coordinates with the dispersion operators only around the prior (denoising) step. A reusable helper is provided.
<!-- ORACLE_OP_END op_050_forward_transpose -->

<!-- ORACLE_OP_START op_060_gap_initialization -->
## GAP initialization and normalization

Before iterating, initialize the accelerated GAP state in the dispersed frame:
- Initial estimate `x = At(y, Phi)` (back-projection of the measurement), shape `(256, 286, 31)`.
- Accelerated residual accumulator `y1 = np.zeros_like(y)`, shape `(256, 286)`.
- Per-pixel normalization `Phi_sum = np.sum(Phi, axis=2)`; then set zero entries to 1: `Phi_sum[Phi_sum == 0] = 1`. This avoids division by zero at detector pixels never illuminated by any channel and is essential for a finite reconstruction.

Set `lambda = 1` (data-step weight) and `accelerate = True`. These initializations feed the GAP update step.
<!-- ORACLE_OP_END op_060_gap_initialization -->

<!-- ORACLE_OP_START op_070_gap_data_step -->
## Accelerated GAP data-consistency update

Each iteration first enforces the measurement constraint by an analytic projection, in the dispersed frame. Let `yb = A(x, Phi)`.

Accelerated GAP (used here, `accelerate=True`):
```
y1 = y1 + (y - yb)
x  = x + lambda * At((y1 - yb) / Phi_sum, Phi)
```
Non-accelerated variant (for reference): `x = x + lambda * At((y - yb)/Phi_sum, Phi)`.

Here `lambda = 1`, `Phi_sum` is the zero-guarded per-pixel channel sum, and division is elementwise over the 2D detector before back-projection. After this data step, convert `x` to the scene frame with `shift_back(x, step=1)`, apply the prior/denoiser (TV or learned), then map back with `shift(x, step=1)` for the next iteration. The accelerated residual `y1` persists across all iterations and drives faster convergence.
<!-- ORACLE_OP_END op_070_gap_data_step -->

<!-- ORACLE_OP_START op_080_tv_prior -->
## Total-variation prior (skimage chambolle)

The default GAP prior is 3D total-variation denoising applied to the scene-frame cube `x` of shape `(256, 256, 31)`:
```
from skimage.restoration import denoise_tv_chambolle
x = denoise_tv_chambolle(x, nsig/255, max_num_iter=tv_iter_max, channel_axis=-1)
```
where `nsig` is the current stage noise level and `tv_iter_max = 5`. The weight passed to skimage is `nsig/255` (noise level divided by 255). `channel_axis=-1` treats the 31 spectral bands as channels, so TV smooths spatially per band while jointly weighting the channel gradient. The TV-only GAP path is fully self-contained (no external weights) and already yields a valid, finite reconstruction; it is the reliable fallback if a learned denoiser is unavailable.
<!-- ORACLE_OP_END op_080_tv_prior -->

<!-- ORACLE_OP_START op_090_learned_denoiser -->
## Optional learned band-window denoiser prior

As a quality enhancer, later GAP iterations can replace TV with a learned hyperspectral CNN denoiser (SDeCNN-style) applied band-by-band over a sliding 7-band spectral window. This prior is optional and only usable if pretrained weights (`.pth`) are provided under `workspace/` or `${CLAUDE_SKILL_DIR}`; if no weights are present, keep using the TV prior for all iterations (the pipeline still produces a valid cube).

Architecture (input 7 bands + 1 noise-map channel -> 1 denoised band):
- PixelUnshuffle downsample by factor `sf=2` (with replication padding so H,W are even), giving `7*sf*sf` feature planes; concatenate a constant noise-level map plane -> `7*4+1` input channels.
- Head conv+ReLU to 128 channels; 13 body conv+ReLU layers at 128 channels (total `nb=15` conv layers); tail conv (no ReLU) to `1*sf*sf`; PixelShuffle upsample by 2; crop back to original H,W.
- Noise-level input `Nsigma = noise_level/255` broadcast as the extra channel.

Band-window construction (for each target band `i` of `nC=31`), the 7-channel input is built with mirrored/duplicated boundary bands:
- interior (`3 <= i <= nC-4`): `x[:, :, i-3:i+4]` (7 bands centered at i).
- low boundary (`i<3`): duplicate the lowest available band to fill the leading slots, then `x[:, :, i:i+4]`.
- high boundary (`i>nC-4`): `x[:, :, i-3:i+1]` then duplicate the highest available band to fill trailing slots.
Each 7-band window is arranged as a tensor `(1, 7, H, W)` (permute channels-first, float), run through the model with its `Nsigma`, and the single-band output is stacked along axis 2 to rebuild `(H, W, nC)`. The CNN noise level used at these iterations is `10/255` for all bands. See the resource for exact window slicing.

Because this prior requires external weights not part of the input case, treat it as an ablatable enhancement layered on top of the TV path.
<!-- ORACLE_OP_END op_090_learned_denoiser -->

<!-- ORACLE_OP_START op_100_iteration_schedule -->
## GAP noise schedule and hybrid prior switching

The standard GAP run configuration is:
- `lambda = 1`, `accelerate = True`, `tv_iter_max = 5`.
- `sigma = [130, 130, 130, 130, 130, 130, 130, 130]` (8 stages, each noise level 130).
- `iter_max = 20` per stage (broadcast to a per-stage list of length 8).
- A global iteration counter `k` starts at 0 and increments each inner iteration; the run terminates as soon as `k == 123` (i.e. 124 total inner iterations across the stages), even if stages remain.

Hybrid prior selection by iteration index `k`:
- Early iterations use the TV prior with weight `sigma_stage/255` (stable coarse structure).
- A specific band of late iterations uses the learned CNN denoiser (finer detail). The CNN is used exactly for `k` in the set: 83-85, 87-89, 91-93, 95-97, 99-101, 103-105, 107-109, 111-113, 115-117, 119-121, 122-123 (a repeating 3-CNN-then-1-TV cadence in the late phase). All other `k` use TV. This yields 32 CNN iterations and 92 TV iterations over the 124 total.
- The CNN, when active, uses noise level `10/255` on every band.

If the learned denoiser weights are unavailable, run TV for all `k` (still 124 iterations, still terminating at `k==123`); this is a valid degraded mode. The exact CNN `k`-set is provided as a resource.
<!-- ORACLE_OP_END op_100_iteration_schedule -->

<!-- ORACLE_OP_START op_110_assemble_output -->
## Final de-dispersion and output writing

The GAP loop returns `x` in the dispersed frame. Convert to the deliverable scene cube with one final `shift_back(x, step=1)` -> shape `(256, 256, 31)`. Do not re-apply `shift` afterward.

Write the result to `outputs/{case_id}.npz` with array key `hyperspectral_cube`:
```
import os, numpy as np
os.makedirs('outputs', exist_ok=True)
cube = shift_back(x, step=1).astype(np.float32)   # (256,256,31)
np.savez('outputs/{}.npz'.format(case_id), hyperspectral_cube=cube)
```
Ensure the saved cube is finite and in float32/float64 (no batch dimension). Preserve the raw reconstructed intensities (arbitrary units); do not rescale, clip to [0,1], or normalize the cube, as the evaluation compares against the reference in these native units.
<!-- ORACLE_OP_END op_110_assemble_output -->

<!-- ORACLE_OP_START op_120_validate_output -->
## Output schema and finiteness validator

After writing each `outputs/{case_id}.npz`, verify it before finishing: the file loads, contains key `hyperspectral_cube`, has shape exactly `(256, 256, 31)`, dtype float32 or float64, and is entirely finite. This is a cheap guardrail that catches wrong axis order, leftover dispersed width (286 instead of 256), NaNs from a missing `Phi_sum` zero-guard, or an accidental batch dimension. Run the provided validator with the run-local runtime interpreter.
<!-- ORACLE_OP_END op_120_validate_output -->

<!-- ORACLE_OP_START op_130_smoke_probe -->
## TV-only GAP smoke probe

Before committing to the full 124-iteration run (which is slow, especially without a GPU), run a short TV-only GAP smoke test (e.g. ~30 iterations) to confirm the geometry, sensing matrix, forward/transpose operators, and de-dispersion are wired correctly and produce a finite `(256,256,31)` cube. This is a diagnostic only; it does not need to reach final quality. The provided script loads a case, builds `Phi`, runs accelerated GAP with the TV prior, and reports the output shape and value range.
<!-- ORACLE_OP_END op_130_smoke_probe -->
