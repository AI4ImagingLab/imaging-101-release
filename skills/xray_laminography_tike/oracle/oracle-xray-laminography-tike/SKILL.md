---
name: oracle-xray-laminography-tike
description: "Reconstruct a 3D complex-valued volume from parallel-beam/laminographic projections by minimizing a least-squares data-fidelity objective with a conjugate-gradient solver built on a Fourier-slice NUFFT forward/adjoint pair."
---

# oracle-xray-laminography-tike Oracle Skill

This skill reconstructs a 128x128x128 complex volume from a single case of simulated laminographic projection data. The reconstruction poses a linear inverse problem min_V ||P_tilt[V] - d||^2 and solves it with conjugate gradient descent. The forward operator P_tilt maps a volume to per-angle 2D complex projections using the generalized Fourier Slice Theorem: a 3D FFT of the volume is sampled along tilted/rotated planes, then transformed back to detector space. The adjoint scatters projection Fourier data back onto the 3D grid. Each operation below is an independently removable piece: the public contract, run-local input loading, the tilt/rotation frequency-grid geometry, the NUFFT primitives and FFT phase conventions, the forward and adjoint projection assembly, the cost and gradient, the CG step-length estimate, the Dai-Yuan conjugate-gradient update with backtracking line search, the multi-round reconstruction driver, GPU acceleration notes, output writing, and submission validation. Preserve every constant, axis order, FFT normalization, sign convention, and normalization factor exactly as stated; these determine whether the reconstruction converges and matches the expected volume semantics.

<!-- ORACLE_OP_START op_010_contract -->
## Public task contract and submission schema

## Public task contract

Reconstruct a single case (`case_000`). For each case listed in the run-local `public/cases.json`, read its `input_data/` and write one reconstruction file.

**Inputs (per case, run-local under `public/`):**
- `raw_data.npz` with keys `projections` (shape `(1,128,128,128)`, complex64) and `theta` (shape `(1,128)`, float32, radians, uniformly spaced over [0, pi)).
- `params_data.json` with `volume_shape=[128,128,128]`, `n_angles=128`, `tilt_rad≈1.5708` (pi/2), `tilt_deg=90.0`, `theta_range_rad=[0.0, pi]`.

**Output (submission):**
- Write to `outputs/{case_id}.npz` relative to the run root (e.g. `outputs/case_000.npz`).
- The file must contain array key `volume` of shape exactly `(128, 128, 128)` and dtype `complex64` or `complex128`.
- Values must be finite. Casting is not allowed, so produce the required dtype directly (complex64 is expected).

**Objective semantics:** the target volume V is the complex refractive index distribution whose laminographic projections match the given data d, i.e. it minimizes `||P_tilt[V] - d||^2`. tilt = pi/2 means standard parallel-beam tomography. Report/produce V only; no other arrays are required.

Use the run-local `python` from `public/envs/runtime/.venv-posix/bin/python` (or the environment on PATH) for all execution.
<!-- ORACLE_OP_END op_010_contract -->

<!-- ORACLE_OP_START op_020_load_inputs -->
## Load projection data, angles, and imaging parameters

## Load inputs and strip the batch dimension

For each case:
1. Load `raw_data.npz`. Cast `projections` to `complex64` and `theta` to `float32`.
2. Read `params_data.json`; take `tilt = float(params['tilt_rad'])` and `volume_shape = tuple(params['volume_shape'])` = `(128,128,128)`.
3. Remove the leading batch dimension before use:
   - `data = projections[0]` -> shape `(128, 128, 128)` = `(R, n, n)` with `R = n_angles = 128`.
   - `theta_1d = theta[0]` -> shape `(128,)`.
4. The volume grid size is `n = volume_shape[0] = 128`; the projection stack has one 2D detector image of shape `(n, n)` per rotation angle, so `data` is indexed `[angle, v, u]`.

Do not reorder axes. The solver consumes `data (R,n,n) complex64`, `theta_1d (R,) float32`, and scalar `tilt`.
<!-- ORACLE_OP_END op_020_load_inputs -->

<!-- ORACLE_OP_START op_030_geometry_grid -->
## Tilt/rotation frequency-grid geometry (Fourier slice coordinates)

## Laminographic frequency-grid geometry

The forward model samples the 3D Fourier transform of the volume along tilted, rotated planes (generalized Fourier Slice Theorem). For each rotation angle theta_p and each 2D detector frequency `(ku, kv)`, compute a 3D frequency point `f = (f0, f1, f2)`:

```
f0 =  kv * sin(tilt)
f1 = -ku * sin(theta) + kv * cos(theta) * cos(tilt)
f2 =  ku * cos(theta) + kv * sin(theta) * cos(tilt)
```

Conventions (must match exactly):
- Frequency index axis: `k = (arange(n) - n//2) / n`, giving values in `[-0.5, 0.5)`. Use the SAME `k` for both `ku` and `kv`.
- Build `(ku, kv)` with `meshgrid(k, k, indexing='xy')`, then `ravel()` each to length `n*n`. So the fast layout iterates `kv` over rows (`y`) and `ku` over columns (`x`); index `idx = p*n*n + y*n + x`.
- Output `xi` has shape `(R*n*n, 3)`, dtype float32, ordered `[angle, y(row/v), x(col/u)]` flattened.
- `ctilt = cos(tilt)`, `stilt = sin(tilt)`. At tilt = pi/2: `stilt=1`, `ctilt≈0`, so `f0=kv`, `f1=-ku*sin(theta)`, `f2=ku*cos(theta)` (standard tomography slice normal to detector, kv maps straight to the z-frequency).

This grid is reused unchanged by both the forward and adjoint operators (the adjoint negates it). See the resource for a vectorized implementation.
<!-- ORACLE_OP_END op_030_geometry_grid -->

<!-- ORACLE_OP_START op_040_nufft_primitives -->
## NUFFT primitives and centered-FFT / checkerboard conventions

## NUFFT primitives and FFT phase conventions

Two non-uniform-FFT primitives connect the regular volume grid to the tilted frequency samples `xi`.

**Type-2 (forward), volume -> values at `xi`:**
1. Centered 3D FFT: `U = fftshift(fftn(ifftshift(u)))`.
2. Map frequencies to grid indices: `grid_coords = (xi + 0.5) * n` (maps [-0.5,0.5) -> [0,n)).
3. Nearest-neighbor gather: `idx = clip(round(grid_coords).astype(int32), 0, n-1)`; `F = U[idx[:,0], idx[:,1], idx[:,2]]`.

**Type-1 (adjoint), values at `xi` -> volume:**
1. Same index mapping `idx` from `(xi+0.5)*n`, clipped to `[0,n-1]`.
2. Scatter-add F onto a zero 3D grid. Complex scatter-add is done by splitting real/imag into float32 accumulators (`np.add.at` on CPU, `cupyx.scatter_add` on GPU), then `U = (U_real + 1j*U_imag).astype(complex64)`.
3. Inverse centered FFT: `u = fftshift(ifftn(ifftshift(U)))`.

**Checkerboard phase** (used on the 2D detector planes before/after the 2D FFT to emulate fftshift cheaply): multiply along each listed axis by `(-1)^index`, i.e. a length-`s` vector of ones with `[1::2] = -1`, reshaped to broadcast on that axis.

Keep NN interpolation, the clip bounds, the real/imag split for scatter, and the exact fftshift/ifftshift ordering; changing any of these breaks the forward/adjoint consistency. See resource for code.
<!-- ORACLE_OP_END op_040_nufft_primitives -->

<!-- ORACLE_OP_START op_050_forward_adjoint -->
## Forward and adjoint laminographic projection assembly

## Forward and adjoint projection operators

**Forward `P_tilt[obj] -> data (R,n,n)`:**
1. `n = obj.shape[0]`, `R = theta.shape[0]`; build `xi` (op geometry) for `(theta, tilt, n)`.
2. `F = nufft_forward(obj, xi, n)`, then `F = F.reshape(R, n, n)`.
3. Apply checkerboard on axes `(1,2)`, then inverse 2D FFT: `data = ifft2(F, axes=(1,2), norm='ortho')`, then checkerboard again on axes `(1,2)`.
4. Return `data.astype(complex64)`.

**Adjoint `P_tilt^H[data] -> obj (n,n,n)`:**
1. Checkerboard `data` on axes `(1,2)`, forward 2D FFT `fft2(., axes=(1,2), norm='ortho')`, checkerboard again on axes `(1,2)`.
2. Build `xi` for `(theta, tilt, n)`. Use the NEGATED grid `-xi` for the adjoint scatter (conjugate transpose of the gather).
3. `obj = nufft_adjoint(F.ravel(), -xi, n)`.
4. Normalize: `obj /= n**2`. Return `obj.astype(complex64)`.

Critical conventions to preserve: `norm='ortho'` on both 2D FFTs, the checkerboard sandwich on both sides of the 2D FFT, `axes=(1,2)` (per-angle detector plane), the `-xi` sign flip in the adjoint, and the `1/n**2` adjoint normalization. See resource for full code.
<!-- ORACLE_OP_END op_050_forward_adjoint -->

<!-- ORACLE_OP_START op_060_cost_gradient -->
## Least-squares cost and gradient

## Data-fidelity cost and gradient

Objective: `f(V) = ||P_tilt[V] - d||^2`.

**Cost:**
```
residual = forward_project(obj, theta, tilt) - data
cost = float(sum(abs(residual).astype(float32)**2))   # real scalar
```

**Gradient:**
```
residual = forward_project(obj, theta, tilt) - data
grad = adjoint_project(residual, theta, tilt, n)
grad /= (R * n**3)      # R = n_angles, n = grid size
```

Keep the normalization factor `1/(R * n**3)` on the gradient exactly; it sets the natural scale that makes the CG step-length estimate and line search behave. The cost is an unnormalized sum of squared residual magnitudes (this is what converges from ~1.24 to ~0.013 over the rounds).
<!-- ORACLE_OP_END op_060_cost_gradient -->

<!-- ORACLE_OP_START op_070_step_length -->
## Initial CG step-length estimate from operator scaling

## Initial step-length estimate

Before the first CG round, estimate an initial line-search step length from the forward-adjoint operator scaling. Because the zero initial volume would give a zero scaler, first perturb it: `test_obj = obj + ones_like(obj) * 1e-6`.

```
fwd = forward_project(test_obj, theta, tilt)
adj = adjoint_project(fwd, theta, tilt, n)
norm_adj = sqrt(sum(abs(adj)**2).real)
norm_obj = sqrt(sum(abs(test_obj)**2).real) + 1e-32
scaler = norm_adj / norm_obj
step_length = 2.0 * scaler if isfinite(scaler) else 1.0   # 2x over-estimate for safety
```

The step length is estimated once (round 0) and then carried/updated by the line search across all rounds. This is a tuning aid: removing it and starting from `step_length = 1.0` still lets the backtracking line search recover, but convergence per round may differ.
<!-- ORACLE_OP_END op_070_step_length -->

<!-- ORACLE_OP_START op_080_cg_step -->
## Dai-Yuan conjugate-gradient update with backtracking line search

## Conjugate-gradient inner step

Each CG iteration on the current `obj`:
1. Compute `grad1 = gradient(obj, data, theta, tilt)`.
2. Search direction via **Dai-Yuan**:
   - First iteration (no previous direction): `dir = -grad1`.
   - Otherwise: `diff = grad1 - grad0`; `denom = sum(conj(dir_prev) * diff).real + 1e-32`; `beta = sum(abs(grad1)**2).real / denom`; `dir = -grad1 + beta * dir_prev`.
3. Set `grad0 = grad1` (store for next iteration).
4. **Backtracking line search** along `dir`: starting from the current `step_length`, repeatedly test `x_new = x + step_length * dir`; accept when `cost(x_new) <= cost(x)`; otherwise `step_length *= 0.5` (step_shrink). If `step_length < 1e-32`, give up (return step 0, keep x, warn). The accepted `step_length` is carried into the next iteration.
5. Update `obj = x_new`, keep the accepted `cost`.

Use the real part for all inner products (`.real`) and the `+1e-32` guards exactly as shown. `cost(x)` is the op_060 least-squares cost. See resource for the full inner loop.
<!-- ORACLE_OP_END op_080_cg_step -->

<!-- ORACLE_OP_START op_090_reconstruct_driver -->
## Multi-round reconstruction driver and zero initialization

## Reconstruction driver

Pipeline per case:
1. Cast: `data -> complex64`, `theta -> float32`, `tilt -> float32`. Assert `data.ndim == 3` and `theta.ndim == 1`.
2. Move arrays to GPU if available (see GPU notes). Initialize `obj = zeros(volume_shape, complex64)` (all-zeros initial guess).
3. Run **5 rounds of 4 CG iterations** each (`N_ROUNDS=5`, `N_ITER_PER_ROUND=4`, 20 total iterations).
   - On round 0, first estimate the initial `step_length` (op step-length) using a `1e-6`-perturbed copy of the current object; otherwise carry the running `step_length`.
   - Each round runs the CG inner loop (op CG step) for `n_iter_per_round` iterations, updating `obj` in place and recording the round's final `cost`.
4. Collect `costs` per round (expected to fall roughly 1.24 -> ~0.15 -> ~0.05 -> ~0.025 -> ~0.017 -> ~0.013; final ~0.013).
5. Return `obj` moved back to CPU as complex64 for output.

Parameters are fixed constants; do not shuffle angles or change the round/iteration counts. The zero initialization is important: the step-length estimate perturbs it to `1e-6` so the first direction is well defined.
<!-- ORACLE_OP_END op_090_reconstruct_driver -->

<!-- ORACLE_OP_START op_100_gpu_runtime -->
## GPU/CuPy acceleration with NumPy fallback

## GPU acceleration and array-module dispatch

The forward/adjoint/FFT operators run identically on NumPy or CuPy. Use an array-module dispatch:
- `import cupy as cp` guarded by `try/except ImportError` (set `cp = None` if unavailable).
- `get_xp(arr)` returns `cp` if `arr` is a `cp.ndarray`, else `numpy`.
- `to_gpu(arr)` = `cp.asarray(arr)` when cupy is present, else identity.
- `to_cpu(arr)` = `cp.asnumpy(arr)` for cupy arrays, else `np.asarray(arr)`.

The scatter-add in the adjoint differs by backend: `np.add.at` on CPU, `cupyx.scatter_add` on GPU (complex split into real/imag float32 accumulators). Everything else (fftn/ifftn/fft2/ifft2, meshgrid, clip, round) has the same API on both. This op is an acceleration wrapper: removing it and running pure NumPy yields the same numerical result but slower. Always move the final volume back to host (`to_cpu`) before writing output.

Use the run-local `python` from `public/envs/runtime/.venv-posix/bin/python`.
<!-- ORACLE_OP_END op_100_gpu_runtime -->

<!-- ORACLE_OP_START op_110_write_output -->
## Write reconstruction to the submission npz

## Write the submission file

For each case, save the reconstructed 3D volume (host/CPU array, batch dimension already removed) to `outputs/{case_id}.npz`:
- Ensure `outputs/` exists.
- Store under array key `volume` with shape exactly `(128, 128, 128)` and dtype `complex64` (allowed: complex64 or complex128; do not rely on casting at validation time, so produce complex64 directly).
- Ensure all values are finite (no NaN/Inf).

```python
import os, numpy as np
os.makedirs('outputs', exist_ok=True)
vol = np.ascontiguousarray(recon).astype(np.complex64)
assert vol.shape == (128, 128, 128)
np.savez(os.path.join('outputs', f'{case_id}.npz'), volume=vol)
```

Do not add extra required arrays; only `volume` is consumed by evaluation.
<!-- ORACLE_OP_END op_110_write_output -->

<!-- ORACLE_OP_START op_120_validate_submission -->
## Validate submission shape, dtype, and finiteness

## Submission self-check

After writing each `outputs/{case_id}.npz`, verify it before finishing:
- The file exists and loads.
- It contains key `volume`.
- `volume.shape == (128, 128, 128)`.
- `volume.dtype` is `complex64` or `complex128`.
- `np.isfinite(volume).all()` is True (both real and imaginary parts finite).

This is a guardrail (non-mutating). Run the bundled validator with the run-local `python`. If any check fails, fix the reconstruction/writing step rather than editing the file in place.
<!-- ORACLE_OP_END op_120_validate_submission -->
