---
name: oracle-xray-ptychography-tike
description: "Reconstruct the complex-valued transmission function of a Siemens star X-ray ptychography dataset from far-field diffraction intensities, and write the raw object phase into the benchmark object frame as outputs/{case_id}.npz. The skill covers the public input/output contract, the shifted scan-position and object-canvas conventions (offset=20, buffer=2 -> (482, 479) frame), the FFT-based ptychographic forward model and Gaussian amplitude objective, the ePIE-style least-squares gradient solver (mini-batching over scan positions, probe refinement, intensity-matching probe rescale, preconditioned conjugate-gradient updates), and the final phase extraction/serialization rules plus runtime diagnostics. The solver must run with plain numpy: the runtime environment has no tike package and GPU kernels can fail to compile, so every algorithmic step is specified numpy-compatible."
---

# oracle-xray-ptychography-tike Oracle Skill

This skill is organized in four removable layers. (1) Contract and data conventions: case layout and submission contract, raw-array loading with batch-axis removal, the scan-position shift convention (minimum coordinate becomes 20.0), and the object-canvas convention (buffer=2, fill 0.5+0j, yielding the (482, 479) benchmark frame for the public case). (2) Physics/statistical model: patch extraction/insertion on an integer-truncated scan grid, the far-field forward model (probe x patch -> ortho-normalized 2D FFT -> summed per-mode intensity), and the Gaussian sqrt-amplitude objective with its detector-plane gradient weight. (3) Solver: shape/dtype validation guards, the probe intensity-rescale step, the preconditioned ePIE batch update, and the epoch/mini-batch orchestration loop (64 epochs, 7 batches, probe updates from epoch 1). (4) Output and validation: raw phase extraction and npz packaging with finite-value and frame checks, plus metric/diagnostic helpers for self-checks. Workflow: read the contract operation first, apply the coordinate conventions before reconstructing, run the solver loop, then package the output exactly per the output operation. All operations default to enabled and are independently removable through their anchors; dependencies point only to earlier operations.

## Task Layout, Inputs, and Submission Contract

<!-- the related step -->
## Task Layout, Inputs, and Submission Contract

This task is X-ray ptychographic phase retrieval: from (N, W, H) far-field diffraction intensities, (N, 2) scan positions, and an initial probe guess, recover the complex object transmission function and output its phase in a fixed object frame.

Run-root layout for the solver:
- Input data: `public/cases/case_000/input_data/raw_data.npz` and `public/cases/case_000/params_data.json` (case manifest `public/cases.json`).
- Submission: write `outputs/case_000.npz` with exactly one array named `object_phase`.

`raw_data.npz` keys (all with a leading batch dimension to drop):
- `diffraction_patterns` (1, 516, 128, 128) float32, photon-count intensities, values in [0, 963].
- `scan_positions` (1, 516, 2) float32, patch upper-left corner coordinates in (row, col) pixel order.
- `probe_guess` (1, 1, 1, 1, 128, 128) complex64, initial probe estimate.

`params_data.json` keys: `n_positions` (516), `probe_shape` [1,1,1,128,128], `diffraction_shape` [128,128], `scan_range_pixels`, `data_min`/`data_max`, `description`.

Output contract:
- Array `object_phase` with shape (482, 479), dtype float32 or float64, finite values only, raw reconstructed phase in radians: `np.angle(reconstructed_object[0])`.
- The (482, 479) frame is top-left origin aligned. Do NOT center-crop a larger canvas, and do NOT apply phase unwrapping, global sign flips, global offset alignment, shift registration, or renormalization before writing.
- The judge compares pixel-by-pixel: `ncc` is the uncentered cosine similarity `dot(est, ref)/(||est|| * ||ref||)` and `nrmse` is `sqrt(mean((est-ref)^2)) / (max(ref) - min(ref))`. A visually similar image can fail if phase signs or the frame origin are misaligned.

## Raw Array Loading and Batch-Axis Removal

<!-- the related step -->
## Raw Array Loading and Batch-Axis Removal

Load `raw_data.npz` and remove the leading batch dimension (index 0) from every array so the solver works on single-case arrays:
- `diffraction_patterns[0]` -> (516, 128, 128) float32; these are measured far-field intensities in photon counts, not amplitudes.
- `scan_positions[0]` -> (516, 2) float32; coordinates are (row, col) pixel positions of the patch upper-left corner.
- `probe_guess[0]` -> (1, 1, 1, 128, 128) complex64; leading axes (batch, row-batch?) are both 1.

Optionally read `params_data.json` for `n_positions`, `probe_shape`, `diffraction_shape`, `data_min`, `data_max` to sanity-check loaded arrays. Do not normalize or otherwise transform the intensities here; the solver consumes raw counts (see the objective operation for how square roots are applied).

Run the loader from the eval run root with the runtime python:
`python ${CLAUDE_SKILL_DIR}/scripts/script_001.py <input_dir>` prints each key, shape, and dtype.

## Scan Position Shift Convention (offset=20)

<!-- the related step -->
## Scan Position Shift Convention (offset=20)

Shift the (N, 2) scan positions so the minimum coordinate becomes exactly 20.0:

```python
scan_shifted = scan - np.amin(scan, axis=0) + 20.0
```

This is a benchmark-fixed preprocessing convention. Critical semantics:
- The shifted coordinates are still patch **upper-left corners** in (row, col) order, relative to the object frame. Do NOT subtract half the probe size to reinterpret them as probe centers.
- Do NOT retune the offset (e.g., to 0 or to a different value) in an attempt to make the output shape match; the offset is 20.0 by convention.
- Do NOT reuse `offset=20` as an additional trailing object padding (the canvas rule lives in the object-frame operation).
- For the public case: raw scan min (row, col) = (-164.338, -164.391), max = (167.580, 164.502); after shifting, min = (20.0, 20.0) and max = (351.918, 348.893), i.e. ceil to (352, 349).

These shifted positions are consumed by patch extraction (integer-truncated) and by the object-canvas sizing convention.

## Object Canvas Sizing and Initialization (buffer=2, 0.5+0j)

<!-- the related step -->
## Object Canvas Sizing and Initialization (buffer=2, 0.5+0j)

Initialize the complex object (psi) as a single-slice array with exactly `buffer=2` extra pixels beyond the shifted scan extent plus the probe footprint:

```python
scan_max = np.amax(scan_shifted, axis=0)          # (row, col) maxima
probe_h, probe_w = probe_shape[-2], probe_shape[-1]  # 128, 128
H = int(np.ceil(scan_max[0])) + probe_h + buffer    # 352 + 128 + 2 = 482
W = int(np.ceil(scan_max[1])) + probe_w + buffer    # 349 + 128 + 2 = 479
psi = np.full((1, H, W), 0.5 + 0j, dtype=np.complex64)
```

Rules:
- `buffer = 2` exactly; the public case gives the benchmark grid (482, 479). Do not add extra trailing padding and do not construct a larger canvas (e.g., (500, 497)) and center-crop it. If some other code path produces a larger canvas, align to the top-left origin and take `phase[:482, :479]`; never center-crop or shift-search.
- Fill value `0.5 + 0j` (complex64): a uniform complex starting point that avoids scaling instabilities.
- D = 1 single slice; the solver operates on `psi[-1]`.
- The exact buffer=2 is what keeps every extracted patch in bounds: with int-truncated positions the maximum slice end is row 479 and col 476, both strictly inside (482, 479).

A self-check helper for the canvas computation is bundled with the epoch-loop script; this operation's convention is the authoritative sizing rule.

## Probe Mode Initialization (single mode, random-phase extension)

<!-- the related step -->
## Probe Mode Initialization (single mode, random-phase extension)

Keep the probe as a (1, 1, S, 128, 128) complex64 array with S incoherent modes. The dataset convention is a single coherent mode, `n_modes = 1`, which leaves the loaded `probe_guess` unchanged (identity operation for this task).

If more modes are requested, each new mode is built from the amplitude of mode 0 with a random phase:

```python
amplitude = np.abs(probe[..., 0:1, :, :])
random_phase = np.exp(2j * np.pi * np.random.rand(*amplitude.shape).astype(np.float32))
new_mode = (amplitude * random_phase).astype(np.complex64)
probe = np.concatenate([probe] + new_modes, axis=-3)
```

The forward model sums per-mode intensities (incoherent summation), so the mode axis is folded into the intensity calculation. For the public case, do not create additional modes: S = 1 reproduces the benchmark behavior; the forward-model and solver code must handle the mode loop generically (it runs once).

## Object Patch Extraction and Gradient Insertion

<!-- the related step -->
## Object Patch Extraction and Gradient Insertion

Patch extraction maps scan positions to object slices. Exact convention:

```python
obj = psi[-1]                              # single slice (H, W)
positions = scan.astype(np.int32)          # FLOOR truncation of coordinates
patches[i] = obj[r:r + ph, c:c + pw]       # (N, ph, pw), same dtype as psi
```

- `(r, c) = (int(positions[i, 0]), int(positions[i, 1]))` in (row, col) order.
- Coordinates are patch upper-left corners; slicing starts directly at `(r, c)` with no offset adjustments.
- With the object-frame convention (buffer=2) every patch is in bounds; no boundary clamping is used in the reference path.
- Insertion is the adjoint accumulation: `obj_update[r:r+ph, c:c+pw] += grad_patches[i]` into a zeros-initialized (H, W) array.

These two primitives are the only places scan coordinates touch the object grid, so the integer-truncation and (row, col) order conventions live here. Run from the eval root:
`python ${CLAUDE_SKILL_DIR}/scripts/script_002.py` exercises both directions on a toy grid (self-contained smoke check).

## Far-Field Forward Model (exit wave, ortho FFT, per-mode intensity)

<!-- the related step -->
## Far-Field Forward Model (exit wave, ortho FFT, per-mode intensity)

For each scan position the measurement model is `I_j = |FFT[P * O_j]|^2`. The concrete forward operator:

```python
patches = extract_patches(psi, scan, (ph, pw))     # (N, ph, pw) complex64
probe_modes = probe[0, 0]                          # (S, ph, pw)
intensity = np.zeros((N, ph, pw), dtype=np.float32)
for s in range(n_modes):
    exit_wave = probe_modes[s] * patches           # elementwise (N, ph, pw)
    farplane = np.fft.fft2(exit_wave, norm="ortho")  # 2D FFT, ortho normalization
    intensity += np.abs(farplane) ** 2             # incoherent sum over modes
```

Conventions that must be preserved:
- `np.fft.fft2` (not numpy-compatible ifft ordering mistakes, no custom fftshift) with `norm="ortho"`; the probe and patches share the same (ph, pw) = (128, 128) grid.
- No extra zero-padding of the exit wave: the far-field grid is the 128x128 patch itself.
- Intensity is float32 and summed over modes BEFORE the square root is applied downstream (sum of |F|^2, not |sum F|^2).
- The full forward pass is differentiable with respect to both psi and probe through this same chain; the objective operation uses the per-mode farplanes, not just the summed intensity.

## Gaussian Sqrt-Amplitude Objective and Detector Gradient

<!-- the related step -->
## Gaussian Sqrt-Amplitude Objective and Detector Gradient

The objective is the Gaussian (least-squares in amplitude) cost, computed per batch:

```python
sqrt_sim  = np.sqrt(total_intensity + 1e-12)   # simulated amplitude
sqrt_data = np.sqrt(data + 1e-12)              # measured amplitude
cost = float(np.mean((sqrt_sim - sqrt_data) ** 2))
```

- `data` are raw photon counts; the model is compared in sqrt-amplitude space (Poisson-like / Gaussian noise model).
- The 1e-12 inside the square roots is the numerical guard, kept consistent in both terms.
- Detector-plane gradient weight (chain rule of the squared-amplitude loss w.r.t. each farplane):

```python
weight = 1.0 - sqrt_data / sqrt_sim      # (N, ph, pw)
grad_farplane_s = farplanes[s] * weight  # per mode
```

- The per-batch cost is a scalar mean; the epoch cost is the mean of the batch costs (see the epoch-loop operation).
- Convergence expectation: cost starts near ~2.0 and falls to ~0.33 after 64 epochs (observed ~5.2 -> ~0.38 in the first 32 epochs of a numpy run); a non-decreasing or divergent cost indicates a convention or scaling error.

## Input Shape and Dtype Validation Guards

<!-- the related step -->
## Input Shape and Dtype Validation Guards

Before running the solver, validate shapes/dtypes and raise `ValueError` on mismatch (cheap, prevents silent corruption):
- `data` must be 3D (N, W, H); `scan` must be 2D with shape (N, 2); `data.shape[0] == scan.shape[0]`.
- `probe` must be 5D with leading axes (1, 1, S, W, H); `psi` must be 3D (D, H, W).
- `probe` and `psi` must be complex-valued (complex64/complex128 accepted).
- `data` should be real float32; values are photon counts >= 0.

These are guardrails only: they do not transform data. Removing them does not change the math, only the failure mode when inputs are malformed.

## Probe Intensity-Matching Rescale

<!-- the related step -->
## Probe Intensity-Matching Rescale

Before the iteration loop, rescale the probe so the simulated intensity matches the measured data magnitude. This removes the global scale ambiguity between object and probe amplitude:

```python
n_sample = min(50, scan.shape[0])
sample_scan = scan[:n_sample]
patches = extract_patches(psi, sample_scan, (ph, pw))
sim_intensity = <forward model intensities for the sampled positions>
data_mag = np.sqrt(np.mean(data[:n_sample]))
sim_mag  = np.sqrt(np.mean(sim_intensity)) + 1e-12
scale = float(data_mag / sim_mag)
probe *= scale
```

- Uses the first 50 scan positions (subset), the forward model with the SAME probe normalization (ortho FFT, summed mode intensities), and the same sqrt-amplitude view as the objective.
- `probe` is scaled in place, preserving shape and dtype (complex64).
- Removing this step is an experimental variant: the joint object/probe updates then start from a different relative scaling and typically converge to a differently scaled (still phase-valid) solution.

## Preconditioned ePIE Gradient Update (single mini-batch)

<!-- the related step -->
## Preconditioned ePIE Gradient Update (single mini-batch)

This is the core solver step: an ePIE-style update with a least-squares (preconditioned) step size, processing all positions of one mini-batch simultaneously. Steps per batch:

1. Patches `(N, ph, pw)` at int-truncated positions; probe modes `P_s = probe[0, 0]`.
2. Forward: `fp_s = fft2(P_s * patches, norm="ortho")`; `I = sum_s |fp_s|^2`.
3. Cost and weight per the objective operation: `weight = 1 - sqrt(data)/sqrt(I+1e-12)`.
4. Backpropagation per mode: `chi_s = ifft2(fp_s * weight, norm="ortho")` (detector-plane gradient back to near plane).
5. Gradient accumulation:
   - Object: `obj_grad_patches = conj(P_s) * chi_s`, scatter-added into an (H, W) numerator array at each scan position; denominator accumulates `|P_s|^2` in the same scatter pattern.
   - Probe: `probe_grad = conj(patches) * chi_s`, summed over positions into (S, ph, pw) numerator/denominator (`|patches|^2` summed).
6. Preconditioned updates (ePIE denominator with alpha-blended max, alpha = 0.05):

```python
max_den = np.max(obj_update_den) + 1e-12
denom = (1 - alpha) * obj_update_den + alpha * max_den
psi[-1] = obj - obj_update_num / (denom + 1e-12)
# probe update only when update_probe=True:
p_max_den = np.max(probe_update_den[s]) + 1e-12
p_denom = (1 - alpha) * probe_update_den[s] + alpha * p_max_den
probe[0, 0, s] = probe_s - probe_update_num[s] / (p_denom + 1e-12)
```

Constants that must be preserved: alpha = 0.05, guard eps = 1e-12 (both in the square roots and in the denominators). The 1e-12 denominator guards prevent division blow-up where no update energy exists.

See the resource for the full annotated reference implementation of this step (pure numpy).

## Epoch / Mini-Batch Orchestration Loop (64 epochs, 7 batches)

<!-- the related step -->
## Epoch / Mini-Batch Orchestration Loop (64 epochs, 7 batches)

Run the batch update over 64 epochs with 7 mini-batches per epoch:

```python
num_iter, num_batch, alpha = 64, 7, 0.05
costs = []
for epoch in range(num_iter):
    indices = np.random.permutation(N)          # reshuffle scan positions each epoch
    batches = np.array_split(indices, num_batch) # ~74 positions per batch
    epoch_cost = 0.0
    for batch in batches:
        cost, psi, probe = update_batch(data[batch], scan[batch], psi, probe,
                                        alpha=alpha, update_probe=(epoch >= 1))
        epoch_cost += cost
    epoch_cost /= num_batch
    costs.append([epoch_cost])                  # costs is a list of 1-element lists
```

- Probe refinement is enabled from epoch 1 onward (`update_probe = epoch >= 1`); epoch 0 updates only the object. Jointly updating the probe is essential because the initial probe guess may not exactly match the experimental illumination.
- No random seed is set in the reference path (stochasticity from the per-epoch permutation is part of the algorithm).
- Execution environment: run with the task runtime `python` (numpy-only). The environment may import a GPU package but its kernels can fail to compile on this machine, so the loop must not depend on GPU arrays; the algorithm above is written with plain numpy (`np.fft`, `np.random`) and completes the full 64-epoch run in well under a minute.
- Returns: reconstructed `psi` (1, 482, 479) complex64, refined `probe` (1, 1, 1, 128, 128) complex64, and `costs` (64, 1) floats.

A complete end-to-end script implementing this loop plus all earlier conventions is bundled:
`python ${CLAUDE_SKILL_DIR}/scripts/script_003.py <input_dir> <output_dir>`
It loads, preprocesses, reconstructs (64 epochs, 7 batches), verifies the phase frame, and writes `outputs/case_000.npz`.

## Phase Extraction and Submission Packaging

<!-- the related step -->
## Phase Extraction and Submission Packaging

Extract the raw object phase and write the submission:

```python
phase = np.angle(reconstructed_psi[0])        # (482, 479), float32 for complex64 input
assert phase.shape == (482, 479)
assert np.isfinite(phase).all()
np.savez("outputs/case_000.npz", object_phase=phase.astype(np.float32))
```

- `object_phase` is the RAW phase in radians: no phase unwrapping, no global sign flip, no offset subtraction, no shift registration, no renormalization, no centered normalization. The judge compares pixel-by-pixel in this exact frame.
- The frame must be exactly (482, 479) top-left aligned (the object-canvas convention). If a differently-sized canvas was produced, align to the top-left origin and slice `[:482, :479]`; never center-crop.
- Dtype: float32 or float64 both accepted; finite values required (schema validation rejects NaN/Inf).
- File name template `outputs/{case_id}.npz` (case_000 for the public case), npz archive containing exactly the `object_phase` array.

A helper that performs the extraction plus checks and writes the archive:
`python ${CLAUDE_SKILL_DIR}/scripts/script_004.py <psi.npy> <output_dir>`

## Environment Probe and Metric Diagnostics (NCC / NRMSE)

<!-- the related step -->
## Environment Probe and Metric Diagnostics (NCC / NRMSE)

Environment probe (run before the solver):
- Confirm `numpy` and `scipy` import with the task runtime python.
- A ptychography toolkit package is NOT required and is typically absent from the environment; implement the solver directly as specified in the solver operations. Do not waste time installing it.
- GPU array support may import successfully but its kernels can fail to compile without the CUDA runtime libraries present; the reference algorithm runs with plain numpy, so use numpy arrays throughout.

Self-check metrics (match the judge's formulas):

```python
def compute_ncc(e, r):
    e = e.ravel().astype(np.float64); r = r.ravel().astype(np.float64)
    return float(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r) + 1e-12))

def compute_nrmse(e, r):
    e = e.ravel().astype(np.float64); r = r.ravel().astype(np.float64)
    return float(np.sqrt(np.mean((e - r) ** 2)) / (r.max() - r.min()))
```

- `ncc` is the UNCENTERED cosine similarity; `nrmse` is RMSE divided by the reference dynamic range. Both are computed pixel-by-pixel without alignment or phase post-processing.
- Sanity tests: identical arrays give ncc ~ 1.0 and nrmse ~ 0.0; a global sign flip of the estimate gives ncc = -1.0 (catastrophic), which is why the raw-phase semantics are mandatory.
- Cheap runtime sanity checks: the cost curve should strictly decrease (e.g., ~5.2 at epoch 0 -> ~0.33 by epoch 64 in a numpy run); the output phase must be finite and exactly (482, 479).
