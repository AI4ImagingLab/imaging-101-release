---
name: oracle-spectral-snapshot-compressive-imaging
description: "Reconstruct a 256x256x31 hyperspectral cube from a single 2D coded-aperture snapshot spectral (SD-CASSI) measurement using a Plug-and-Play GAP solver with total-variation and optional learned denoiser priors, and write the cube to the required npz output."
---

# oracle-spectral-snapshot-compressive-imaging Oracle Skill

This skill reconstructs a hyperspectral data cube from a single compressed CASSI measurement. It covers the public output contract, run-local input loading, the single-disperser dispersion geometry and sensing-matrix construction, the forward/transpose measurement operators, the accelerated Generalized Alternating Projection (GAP) data-consistency update, the total-variation prior, an optional learned band-window CNN denoiser prior with its hybrid iteration schedule, and final output assembly and validation. Each operation is independently removable; the total-variation GAP path alone yields a valid, finite cube, while the learned denoiser is an optional quality enhancement that requires provided weights.

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

## GAP initialization and normalization

Before iterating, initialize the accelerated GAP state in the dispersed frame:
- Initial estimate `x = At(y, Phi)` (back-projection of the measurement), shape `(256, 286, 31)`.
- Accelerated residual accumulator `y1 = np.zeros_like(y)`, shape `(256, 286)`.
- Per-pixel normalization `Phi_sum = np.sum(Phi, axis=2)`; then set zero entries to 1: `Phi_sum[Phi_sum == 0] = 1`. This avoids division by zero at detector pixels never illuminated by any channel and is essential for a finite reconstruction.

Set `lambda = 1` (data-step weight) and `accelerate = True`. These initializations feed the GAP update step.

## Accelerated GAP data-consistency update

Each iteration first enforces the measurement constraint by an analytic projection, in the dispersed frame. Let `yb = A(x, Phi)`.

Accelerated GAP (used here, `accelerate=True`):
```
y1 = y1 + (y - yb)
x  = x + lambda * At((y1 - yb) / Phi_sum, Phi)
```
Non-accelerated variant (for reference): `x = x + lambda * At((y - yb)/Phi_sum, Phi)`.

Here `lambda = 1`, `Phi_sum` is the zero-guarded per-pixel channel sum, and division is elementwise over the 2D detector before back-projection. After this data step, convert `x` to the scene frame with `shift_back(x, step=1)`, apply the prior/denoiser (TV or learned), then map back with `shift(x, step=1)` for the next iteration. The accelerated residual `y1` persists across all iterations and drives faster convergence.

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
