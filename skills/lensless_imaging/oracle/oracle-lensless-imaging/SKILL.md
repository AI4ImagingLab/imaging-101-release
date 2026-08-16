---
name: oracle-lensless-imaging
description: "Reconstruct a scene image from a single lensless (DiffuserCam) sensor measurement by inverting a diffuser-encoded cropped convolution, using an FFT-based ADMM solver with total-variation and non-negativity priors."
---

# oracle-lensless-imaging Oracle Skill

This skill reconstructs an RGB scene image v from a lensless sensor measurement b and a calibration point-spread function (PSF) h. The forward model is a shift-invariant linear system b = crop(h * v), where * is 2-D linear convolution and crop trims the (2H-1)x(2W-1) result to the central sensor extent HxW. The reconstruction solves the regularised non-negative least-squares problem min_{v>=0} (1/2)||b - Av||^2 + tau*||grad v||_1 with an anisotropic total-variation penalty, via the Alternating Direction Method of Multipliers (ADMM) with variable splitting. Each operation below is an independently removable piece of the pipeline: reading the public contract, loading the preprocessed arrays, establishing padding/crop geometry, building the FFT forward operator and its adjoint, defining the finite-difference gradient and its frequency-domain Gram spectrum, precomputing the diagonal solve factors, running the four proximal sub-updates plus dual ascent, and finally cropping/clipping and writing the output npz. Follow the operations in order; later operations assume the conventions established earlier.

<!-- ORACLE_OP_START op_001_public_contract -->
## Public task contract: inputs, output schema, and case protocol

## Public task contract

For each case you are given a lensless sensor measurement `b` and a calibration point-spread function (PSF) `h`, and must reconstruct the scene image `v`.

**Inputs** (run-local, under `public/`): for each case there is an input directory containing `raw_data.npz`. Discover cases from the run-local cases manifest; each case has an `id` (e.g. `case_000`) and an input directory.

**Output**: write one file per case to `outputs/{case_id}.npz` relative to the run root. The npz must contain a single required array under key `reconstruction`.

**Output array constraints**:
- key: `reconstruction`
- shape: exactly `(760, 1014, 3)` (H, W, C = rows, cols, RGB channels)
- dtype: `float32` or `float64`
- all values must be finite (no NaN/Inf); casting is NOT allowed, so produce the exact declared shape/dtype yourself.

**Physical model to invert**: `b = crop(h * v)` where `*` is 2-D linear convolution and `crop` extracts the central `H x W` region of the `(2H-1) x (2W-1)` full convolution. The system is ill-conditioned (crop discards information; the diffuse PSF contracts dynamic range), so regularisation is essential.

**Reconstruction objective**: `min_{v>=0} (1/2)||b - Av||_2^2 + tau*||grad v||_1`, with `A = crop o (h *)` the lensless forward operator and `||grad v||_1` the anisotropic total-variation (TV) penalty. Non-negativity reflects that intensities cannot be negative.

Write outputs with run-local `python` from PATH; keep everything under `public/`, `workspace/`, `outputs/`, and `logs/agent/`.
<!-- ORACLE_OP_END op_001_public_contract -->

<!-- ORACLE_OP_START op_002_load_npz -->
## Load preprocessed PSF and measurement from raw_data.npz

## Load PSF and measurement

The per-case `raw_data.npz` already contains the fully preprocessed arrays; you do NOT need to redo downsampling, background subtraction, or normalisation. Load it and strip the leading batch dimension.

- The npz has exactly two keys: `psf` and `measurement`.
- Each has batch-first shape `(1, H, W, C)` = `(1, 760, 1014, 3)`, dtype `float32`, values already normalised to `[0, 1]`.
- Take index `[0]` to drop the batch axis, giving `(H, W, C)` arrays. Cast both to `float32`.

```python
import numpy as np
z = np.load(npz_path)
psf = z["psf"][0].astype(np.float32)          # (760, 1014, 3)
measurement = z["measurement"][0].astype(np.float32)  # (760, 1014, 3)
```

Conventions (for reference; already applied to the arrays you load): RGB channel order R,G,B; downsampling factor 4 from a 3040x4056 sensor to 760x1014; PSF normalised by subtract-min then divide-by-max into `[0,1]`; measurement had the PSF's dark level subtracted, negatives clamped, then divided by the PSF max. Both `psf` and `measurement` share identical spatial shape `(H, W, C)`; the ADMM solver assumes `measurement.shape == psf.shape`.
<!-- ORACLE_OP_END op_002_load_npz -->

<!-- ORACLE_OP_START op_003_pad_crop_geometry -->
## Center padding / crop geometry for linear convolution

## Padding and crop geometry

Linear (non-circular) 2-D convolution of two `H x W` signals produces a `(2H-1) x (2W-1)` result. To compute it with FFTs you zero-pad to at least that size, rounded up to an FFT-friendly length, and you must center-pad so a later `ifftshift` restores spatial alignment.

Let `psf_shape = (H, W)` (spatial only). Then:

```python
from scipy.fft import next_fast_len
import numpy as np
psf_shape = np.array(psf.shape[:2])              # (H, W)
conv_shape = 2 * psf_shape - 1                    # (2H-1, 2W-1)
padded_shape = np.array([next_fast_len(int(s)) for s in conv_shape])  # (pH, pW)
start = (padded_shape - psf_shape) // 2           # center offset
end = start + psf_shape
```

**Center-pad** places an `(H, W, C)` signal at `[start:end]` inside a zeros array of shape `(pH, pW, C)`; the rest is zero. **Crop** extracts exactly that `[start:end]` window back to `(H, W, C)`. This guarantees `crop(pad(v)) == v`.

```python
def pad(v):
    ph, pw = padded_shape
    out = np.zeros((ph, pw, v.shape[2]), dtype=v.dtype)
    out[start[0]:end[0], start[1]:end[1], :] = v
    return out

def crop(x):
    return x[start[0]:end[0], start[1]:end[1], :]
```

Critical: use the SAME `[start:end]` window for pad and crop, and use `next_fast_len` on `(2H-1, 2W-1)` (not on `H, W`). The ADMM optimisation variable lives entirely in this padded `(pH, pW, C)` space; cropping happens only at the very end.
<!-- ORACLE_OP_END op_003_pad_crop_geometry -->

<!-- ORACLE_OP_START op_004_fft_forward_operator -->
## FFT forward convolution operator and adjoint

## FFT forward operator M and adjoint M^H

Build the shift-invariant convolution operator in the Fourier domain from the center-padded PSF. Use real FFTs (`rfft2`/`irfft2`) over the two spatial axes with `norm="backward"` (no scaling on forward transform, 1/N on inverse), which matches the standard convolution theorem and the LenslessPiCam default.

**Precompute once** from the PSF:
```python
from scipy.fft import rfft2, irfft2, ifftshift
norm = "backward"
psf_pad = pad(psf)                                   # (pH, pW, C)
H_fft   = rfft2(psf_pad, axes=(0, 1), norm=norm)     # (pH, pW//2+1, C) complex
Hadj    = np.conj(H_fft)
```

**Forward** `M x` (operand is already padded, output stays padded):
```python
def convolve(x_padded):
    Fx = rfft2(x_padded, axes=(0, 1), norm=norm)
    out = irfft2(H_fft * Fx, axes=(0, 1), s=tuple(padded_shape), norm=norm)
    return ifftshift(out, axes=(0, 1))
```

**Adjoint** `M^H y` (uses the conjugate spectrum):
```python
def deconvolve(y_padded):
    Fy = rfft2(y_padded, axes=(0, 1), norm=norm)
    out = irfft2(Hadj * Fy, axes=(0, 1), s=tuple(padded_shape), norm=norm)
    return ifftshift(out, axes=(0, 1))
```

Critical conventions:
- The `ifftshift` on the spatial axes after `irfft2` is REQUIRED; it corrects the phase offset introduced by center-padding the PSF. Omitting it misaligns the reconstruction.
- FFTs are taken over axes `(0, 1)` only; the channel axis (2) is untouched (per-channel independent convolution).
- `convolve`/`deconvolve` expect padded `(pH, pW, C)` inputs and return padded outputs; they do NOT add padding or crop.
- A standalone `forward(v)` on an unpadded image is `crop(convolve(pad(v)))`; use it for reprojection/diagnostics only, not inside the ADMM loop.
<!-- ORACLE_OP_END op_004_fft_forward_operator -->

<!-- ORACLE_OP_START op_005_finite_diff -->
## Finite-difference gradient operator and its adjoint

## TV gradient operator Psi and adjoint Psi^T

The TV penalty uses the 2-D first-difference gradient with periodic (roll-based) boundary conditions, stacked along a leading direction axis.

```python
import numpy as np

def finite_diff(v):
    # v: (H, W, C) or padded (pH, pW, C) -> u: (2, ., ., C)
    return np.stack([
        np.roll(v, 1, axis=0) - v,   # row (vertical) difference
        np.roll(v, 1, axis=1) - v,   # col (horizontal) difference
    ], axis=0)

def finite_diff_adj(u):
    # u: (2, ., ., C) -> v: (., ., C); adjoint of finite_diff
    return (np.roll(u[0], -1, axis=0) - u[0]) + (np.roll(u[1], -1, axis=1) - u[1])
```

Critical conventions:
- `finite_diff` output has axis 0 of length 2 indexing (vertical, horizontal) differences; keep this ordering consistent everywhere.
- The forward difference is `roll(v, +1) - v`; the adjoint is `roll(u, -1) - u`. These two must be an exact adjoint pair (`<Psi v, u> = <v, Psi^T u>`) for ADMM to converge.
- Anisotropic TV = sum of absolute values of both difference components (handled by the soft-threshold sub-update).
<!-- ORACLE_OP_END op_005_finite_diff -->

<!-- ORACLE_OP_START op_006_finite_diff_gram -->
## Frequency-domain Gram spectrum of the TV operator

## Gram spectrum |Psi~|^2 of the gradient operator

`Psi^T Psi` is diagonalised by the DFT; its eigenvalues equal the DFT of the discrete-Laplacian kernel implied by the roll-based finite differences. Build it once on the PADDED spatial shape `(pH, pW)`.

```python
from scipy.fft import rfft2
import numpy as np

def finite_diff_gram(shape, dtype=np.float64):
    Hh, Ww = shape
    gram = np.zeros((Hh, Ww), dtype=dtype)
    gram[0, 0]   =  4
    gram[0, 1]   = -1
    gram[0, -1]  = -1
    gram[1, 0]   = -1
    gram[-1, 0]  = -1
    return rfft2(gram)[:, :, np.newaxis]   # (pH, pW//2+1, 1), broadcastable over C
```

Equivalently `|Psi~(k1,k2)|^2 = 4 - 2cos(2*pi*k1/pH) - 2cos(2*pi*k2/pW)`. Take `abs()` of this when assembling the v-update denominator. The trailing singleton channel axis lets it broadcast against per-channel spectra. Compute in `float64` for numerical stability of the denominator.
<!-- ORACLE_OP_END op_006_finite_diff_gram -->

<!-- ORACLE_OP_START op_007_precompute_divmats -->
## Precompute diagonal solve factors R_divmat and X_divmat

## Precompute the diagonal solve factors

Both the x-update and v-update reduce to element-wise divisions because the relevant operators are diagonalised. Precompute these once after the measurement is set.

**X_divmat** (padded spatial domain) for the data-consistency solve `(C^T C + mu1 I)^{-1}`. `C^T C` is a mask: 1 inside the crop (sensor) window `[start:end]`, 0 outside. So:
```python
ones_padded = pad(np.ones((H, W, C), dtype=np.float32))   # 1 in window, 0 outside
X_divmat = (1.0 / (ones_padded + mu1)).astype(np.float32)  # = 1/(1+mu1) inside, 1/mu1 outside
```

**pad(b)**: the zero-padded measurement, `data_padded = pad(measurement)`, used in the x-update.

**R_divmat** (frequency domain) for the v-update denominator:
```python
HtH = np.abs(Hadj.astype(np.complex128) * H_fft.astype(np.complex128))      # |H~|^2, (pH, pW//2+1, C)
PsiTPsi = np.abs(finite_diff_gram((pH, pW), dtype=np.float64))               # |Psi~|^2, (pH, pW//2+1, 1)
R_divmat = 1.0 / (mu1 * HtH + mu2 * PsiTPsi + mu3)                           # float64
```

CRITICAL numerical detail: compute and keep `R_divmat` in `float64`. Its max is ~`1/mu3` (~2.5e4); in `float32` the ~1e-7 relative error becomes ~2.5e-3 absolute error per iteration that compounds and causes divergence. Because `R_divmat` is `float64`, the v-update automatically promotes to `complex128`.
<!-- ORACLE_OP_END op_007_precompute_divmats -->

<!-- ORACLE_OP_START op_008_admm_init -->
## ADMM state initialisation in padded space

## ADMM initialisation

All primal and dual variables live in the padded `(pH, pW, C)` space (gradient variables carry a leading direction axis of length 2). Initialise everything to zero.

```python
image_est = np.zeros((pH, pW, C), dtype=np.float32)   # v_pad (the image estimate)
u   = np.zeros((2, pH, pW, C), dtype=np.float32)       # = Psi v
x   = np.zeros((pH, pW, C), dtype=np.float32)          # = M v
w   = np.zeros((pH, pW, C), dtype=np.float32)          # = v (nonneg var)
xi  = np.zeros((pH, pW, C), dtype=np.float32)          # dual for x = Mv
eta = np.zeros((2, pH, pW, C), dtype=np.float32)       # dual for u = Psi v
rho = np.zeros((pH, pW, C), dtype=np.float32)          # dual for w = v
```

Also cache the operators evaluated at the current estimate so the sub-updates and dual updates reuse them:
```python
forward_out = convolve(image_est)   # M v_pad, (pH, pW, C)
Psi_out     = finite_diff(image_est)  # Psi v_pad, (2, pH, pW, C)
```

The initial image `v0 = 0` and all-zero duals are the standard cold start.
<!-- ORACLE_OP_END op_008_admm_init -->

<!-- ORACLE_OP_START op_009_admm_updates -->
## ADMM proximal sub-updates and dual ascent (one iteration)

## One ADMM iteration

Each iteration performs the four proximal primal updates (u, x, w, v), refreshes the cached operators, then the three dual ascents. Use the soft-threshold operator `S_lambda(z) = sign(z) * max(|z| - lambda, 0)`.

```python
def soft_thresh(z, thresh):
    return np.sign(z) * np.maximum(np.abs(z) - thresh, 0)

# 1. u-update (TV proximal / soft threshold)
u = soft_thresh(Psi_out + eta / mu2, tau / mu2)

# 2. x-update (data consistency, element-wise)
x = X_divmat * (xi + mu1 * forward_out + data_padded)

# 3. w-update (non-negativity projection)
w = np.maximum(rho / mu3 + image_est, 0.0)

# 4. v-update (frequency-domain solve)
rk = ((mu3 * w - rho)
      + finite_diff_adj(mu2 * u - eta)
      + deconvolve(mu1 * x - xi))          # deconvolve = M^H
freq = R_divmat * rfft2(rk, axes=(0, 1), norm=norm)
image_est = irfft2(freq, axes=(0, 1), s=tuple(padded_shape), norm=norm).real.astype(np.float32)

# refresh cached operators at the new estimate (needed by dual updates & next iter)
forward_out = convolve(image_est)
Psi_out     = finite_diff(image_est)

# 5. dual updates
xi  += mu1 * (forward_out - x)
eta += mu2 * (Psi_out - u)
rho += mu3 * (image_est - w)
```

Critical ordering / conventions:
- Order per iteration is exactly: u, x, w, v, then refresh `forward_out`/`Psi_out`, then xi, eta, rho.
- The v-update residual `rk` combines the w-term `(mu3 w - rho)`, the TV-adjoint term `Psi^T(mu2 u - eta)`, and the data-adjoint term `M^H(mu1 x - xi)`.
- `R_divmat` (float64) multiplies the rfft2 of `rk`, so the product is complex128; take `.real` and cast back to float32.
- The dual updates use the REFRESHED `forward_out`/`Psi_out` (evaluated at the new v), not the pre-update caches.
<!-- ORACLE_OP_END op_009_admm_updates -->

<!-- ORACLE_OP_START op_010_hyperparameters -->
## ADMM penalty/regularisation hyperparameters and iteration count

## Hyperparameters and iteration count

Use these fixed values for the augmented-Lagrangian penalties, TV weight, and iteration budget:

| Symbol | Value  | Role |
|--------|--------|------|
| mu1    | 1e-6   | penalty for x = M v (data-consistency split) |
| mu2    | 1e-5   | penalty for u = Psi v (TV split) |
| mu3    | 4e-5   | penalty for w = v (non-negativity split) |
| tau    | 1e-4   | TV (L1) regularisation weight |
| n_iter | 300    | number of ADMM iterations |

Notes:
- These penalties enter `R_divmat`, `X_divmat`, and the soft-threshold level `tau/mu2`; changing them requires recomputing the precomputed factors.
- The reference pipeline runs 300 iterations. Roughly 100 iterations already give a visually interpretable result; 300 sharpens it. Each iteration is O(HW log HW), dominated by FFTs.
- Do not add early stopping or adaptive penalty schedules; run a fixed `n_iter` sweep.
<!-- ORACLE_OP_END op_010_hyperparameters -->

<!-- ORACLE_OP_START op_011_postprocess_output -->
## Crop, clip, and write reconstruction npz

## Finalise and write output

After the ADMM loop, convert the padded estimate into the required output image:

```python
img = crop(image_est)          # (pH, pW, C) -> (760, 1014, 3)
img = np.clip(img, 0, None)    # enforce non-negativity
reconstruction = img.astype(np.float32)
```

Then save one npz per case to the run-local outputs directory:
```python
import os, numpy as np
os.makedirs("outputs", exist_ok=True)
np.savez(f"outputs/{case_id}.npz", reconstruction=reconstruction)
```

Requirements to satisfy the schema:
- key must be exactly `reconstruction`;
- shape exactly `(760, 1014, 3)` (this is what `crop` produces from the padded estimate);
- dtype `float32` (also acceptable `float64`); all values finite and non-negative;
- do NOT apply display-only transforms (gamma correction, min-max display normalisation) to the saved array — those are for visualisation only. Save the raw cropped/clipped reconstruction.

Process every case in the manifest, writing one `outputs/{case_id}.npz` each.
<!-- ORACLE_OP_END op_011_postprocess_output -->

<!-- ORACLE_OP_START op_012_reprojection_probe -->
## Reprojection NRMSE diagnostic (optional sanity check)

## Reprojection sanity check (diagnostic only)

After reconstructing, optionally verify that re-applying the forward model to your reconstruction approximates the measurement. This is a non-mutating diagnostic; it does not change the saved output.

```python
reprojected = crop(convolve(pad(reconstruction)))   # A v_hat, shape (H, W, C)
reprojection_nrmse = float(
    np.sqrt(np.mean((reprojected - measurement) ** 2))
    / (measurement.max() - measurement.min() + 1e-12)
)
print("reprojection NRMSE:", reprojection_nrmse)
```

Interpretation: a well-converged reconstruction gives a small residual (the reference reports `||A v_hat - b|| / ||b||` typically well below ~0.5 after enough iterations). A large or non-finite value signals a bug in the operator conventions (padding, ifftshift, adjoint sign, or float64 denominator). Log this under `logs/agent/`; do not gate the saved output on it.
<!-- ORACLE_OP_END op_012_reprojection_probe -->

<!-- ORACLE_OP_START op_013_output_validation -->
## Validate output npz against the submission schema

## Output validation gate

Before finishing, confirm every `outputs/{case_id}.npz` meets the schema. Check:
- the file exists and loads;
- it contains the key `reconstruction`;
- shape is exactly `(760, 1014, 3)`;
- dtype is `float32` or `float64`;
- all values are finite (no NaN/Inf).

The schema forbids casting, so the array must already be the correct dtype/shape — do not rely on the harness to fix it. If any check fails, fix the pipeline and re-run rather than shipping an invalid file. Run the bundled validator with run-local `python`:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/op_013_output_validation.py outputs/case_000.npz
```
<!-- ORACLE_OP_END op_013_output_validation -->
