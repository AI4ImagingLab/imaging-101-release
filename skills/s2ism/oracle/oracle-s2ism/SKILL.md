---
name: oracle-s2ism
description: "Oracle guidance for reconstructing a two-plane fluorescent volume from a 25-channel ISM measurement using the provided PSF stack and a multi-plane Poisson ML-EM/Richardson-Lucy update."
---

# oracle-s2ism Oracle Skill

Use the public case metadata and arrays exactly as packaged. The bundled guidance emphasizes the array shapes, PSF normalization, FFT convolution conventions, ML-EM initialization/update/stopping choices, and final NPZ schema needed for this task.

<!-- ORACLE_OP_START op_001_public_contract -->
## Public submission contract

### Public submission contract

Produce one file per case at `outputs/{case_id}.npz`; for the public case this is `outputs/case_000.npz`. The NPZ must contain a single required array named `reconstruction` with shape `(2, 201, 201)`. Either `float32` or `float64` is accepted, but `float32` is the preferred compact output. All entries must be finite. The array order is axial plane first, then image rows, then image columns.

Do not add a leading batch dimension in the submitted array. If your reconstruction routine temporarily stores `(1, 2, 201, 201)`, remove the leading singleton before saving.

A minimal save pattern is:

```python
import numpy as np
from pathlib import Path
Path('outputs').mkdir(exist_ok=True)
np.savez('outputs/case_000.npz', reconstruction=recon.astype(np.float32))
```
<!-- ORACLE_OP_END op_001_public_contract -->

<!-- ORACLE_OP_START op_002_case_loading -->
## Case file discovery and raw array loading

### Case file discovery and raw array loading

Load the public case from `public/visible_data/cases/case_000/input_data/raw_data.npz`. It contains:

- `measurements`: stored as `(1, 201, 201, 25)`; take index `0` to obtain `(201, 201, 25)`.
- `psf`: stored as `(1, 2, 145, 145, 25)`; take index `0` to obtain `(2, 145, 145, 25)`.

Cast both arrays to a floating type for computation. The reference-style solver converts NumPy arrays to Torch tensors in `float32`; loading as `float64` then casting inside the solver is also fine. The measurement axes are rows, columns, detector-channel. The PSF axes are axial plane, rows, columns, detector-channel.
<!-- ORACLE_OP_END op_002_case_loading -->

<!-- ORACLE_OP_START op_003_geometry_and_forward_model -->
## Multi-plane, multi-channel forward model convention

### Multi-plane, multi-channel forward model convention

The measurement at detector channel `c` is modeled as a sum over axial planes of spatial convolutions:

\[
\hat y_c(y,x) = \sum_z (h_{z,c} * X_z)(y,x).
\]

Here `X` has shape `(Nz, Ny, Nx)`, the provided PSF has shape `(Nz, psf_y, psf_x, Nch)`, and the measurement has shape `(Ny, Nx, Nch)`. Convolutions are 2-D over the image row/column axes only. The detector channel axis is not spatial and must be summed only after applying the appropriate per-channel PSF in the forward/back-projection equations.

The public PSF planes are already generated at the task's distinctive axial spacing, with the out-of-focus plane convention recorded in metadata as 720 nm. Do not replace the provided PSF with a generic Gaussian or a PSF regenerated at an arbitrary small z spacing.
<!-- ORACLE_OP_END op_003_geometry_and_forward_model -->

<!-- ORACLE_OP_START op_004_psf_padding_normalization -->
## PSF centering, padding, and normalization before FFT

### PSF centering, padding, and normalization before FFT

Before iterative deconvolution, adapt the PSF support to the measurement image size. For the public case the measurement is `201 x 201` and the PSF is `145 x 145`, so pad the PSF symmetrically by `(201 - 145) / 2 = 28` pixels on each side of each spatial axis. After padding, the PSF has spatial size `201 x 201`.

Normalize the padded PSF per axial plane over all non-z axes: divide each `h[z]` by `h[z].sum()` where the sum includes y, x, and all detector channels. This preserves photon-count scaling across planes. Build the adjoint PSF by flipping the normalized PSF over axes corresponding to rows, columns, and detector channels in the reference tensor layout.

If an even-size measurement is encountered, the reference-style solver first drops the first row and/or first column to make odd dimensions, then pads zeros back at the end. The public case is already odd (`201 x 201`), so no crop/repad is needed.
<!-- ORACLE_OP_END op_004_psf_padding_normalization -->

<!-- ORACLE_OP_START op_005_flat_initialization -->
## Flat positive initialization scaled to total counts

### Flat positive initialization scaled to total counts

Initialize the object estimate as a strictly positive constant image for each axial plane. For `Nz=2` and public image size `Ny=Nx=201`, use:

\[
X^{(0)}_{z,y,x} = \frac{\sum_{y,x,c} Y_{y,x,c}}{N_z\,N_y\,N_x}.
\]

This matches the fixed-iteration ML-EM behavior and keeps the multiplicative update non-negative. The public data sum is about 5.36 million counts, so the starting value is roughly tens of photons per voxel. A channel-summed initialization can also run, but the flat initialization is the reference-style choice.
<!-- ORACLE_OP_END op_005_flat_initialization -->

<!-- ORACLE_OP_START op_006_fft_partial_convolution -->
## FFT partial convolution with channel and plane summations

### FFT partial convolution with channel and plane summations

For efficiency, perform 2-D convolutions with real FFTs over the row/column axes and combine plane/channel tensors with Einstein summation. Use an FFT shape equal to the padded PSF spatial size (`201 x 201` for the public case). After the inverse real FFT, apply `ifftshift` over the spatial axes to align the convolution output with the image grid.

The forward projection is: convolve every `(z, channel)` PSF with `X[z]`, then sum over `z`, producing `(Ny, Nx, Nch)`. The backprojection is: convolve the ratio image with the adjoint/flipped PSF and sum over detector channels, producing `(Nz, Ny, Nx)`.

A direct `scipy.signal.convolve(..., mode='same')` implementation can be useful for debugging, but the FFT version is faster and matches the task hints.
<!-- ORACLE_OP_END op_006_fft_partial_convolution -->

<!-- ORACLE_OP_START op_007_poisson_mlem_update -->
## Poisson ML-EM multiplicative update

### Poisson ML-EM multiplicative update

Use the multi-plane Richardson-Lucy/ML-EM update appropriate for Poisson photon counts:

1. Forward project the current object: `Yhat = A(X)`.
2. Compute the ratio `R = Y / Yhat`, but set the ratio to zero wherever `Yhat` is below machine epsilon to avoid division instability.
3. Backproject the ratio: `U = A_adjoint(R)`.
4. Update multiplicatively: `X = X * U`.

With a positive initialization and non-negative data/PSF, this preserves non-negativity. Do not subtract gradients or add arbitrary regularizers unless you have confirmed they improve the public task; the core reference behavior is pure fixed-iteration Poisson ML-EM.
<!-- ORACLE_OP_END op_007_poisson_mlem_update -->

<!-- ORACLE_OP_START op_008_fixed_iteration_schedule -->
## Use fixed 50-iteration reconstruction schedule

### Use fixed 50-iteration reconstruction schedule

Run exactly 50 ML-EM iterations with `stop='fixed'` and save the last iterate. The reference-style public reconstruction uses:

- `max_iter = 50`
- `stop = 'fixed'`
- `rep_to_save = 'last'`
- `initialization = 'flat'`

The solver may track photon counts and relative changes for diagnostics, but the fixed schedule ignores convergence thresholds and stops only when the iteration index reaches 50. More iterations can sharpen but may over-amplify noise; fewer iterations usually under-deconvolve the two planes.
<!-- ORACLE_OP_END op_008_fixed_iteration_schedule -->

<!-- ORACLE_OP_START op_009_device_and_dtype_runtime -->
## Runtime dtype and device discipline

### Runtime dtype and device discipline

Use the run-local Python environment provided by the task. Generated commands should use `python` from the current run context, not an absolute authoring path. Torch is available in the runtime environment.

A practical approach is to cast inputs to `float32` tensors for FFT/update work. The reference-style code briefly normalizes PSFs in higher precision, but then FFT/update operations are compatible with float tensors. Attempt CUDA if available, but fall back to CPU if CUDA is absent or runs out of memory. The public case is small enough for CPU if needed.
<!-- ORACLE_OP_END op_009_device_and_dtype_runtime -->

<!-- ORACLE_OP_START op_010_plane_intensity_interpretation -->
## Axial-plane interpretation and intensity balance

### Axial-plane interpretation and intensity balance

The public object has two axial planes: an in-focus plane and an out-of-focus/background plane. The synthetic convention scales the second plane more strongly to mimic a thicker labelled background, and the PSF stack was generated at the metadata-reported 720 nm separation to make the two planes distinguishable.

For reconstruction, do not collapse channels or planes into a single 2-D image. The output must preserve both planes. A simple channel-sum image is useful as a diagnostic baseline, but it cannot separate axial contributions. The ML-EM update should let the two PSF planes distribute photons between `reconstruction[0]` and `reconstruction[1]`.
<!-- ORACLE_OP_END op_010_plane_intensity_interpretation -->

<!-- ORACLE_OP_START op_011_output_postprocess -->
## Final array cleanup and NPZ writing

### Final array cleanup and NPZ writing

After the final iteration, move the reconstruction to CPU NumPy if using Torch. Ensure shape is exactly `(2, 201, 201)`, with no singleton case dimension and no detector-channel dimension. Cast to `float32` for saving. Replace accidental non-finite values before saving only as a last-resort safety guard; a stable ML-EM implementation should already be finite.

Create `outputs/` if it does not exist and save:

```python
np.savez('outputs/case_000.npz', reconstruction=recon.astype(np.float32))
```
<!-- ORACLE_OP_END op_011_output_postprocess -->

<!-- ORACLE_OP_START op_012_smoke_test_and_diagnostics -->
## Cheap reconstruction smoke checks

### Cheap reconstruction smoke checks

After writing the output, run the bundled schema validator and inspect basic statistics. A plausible ML-EM reconstruction should be finite, non-negative or nearly so, non-constant, and have nonzero signal in both planes. The raw channel-summed image `measurements.sum(axis=-1)` is a useful diagnostic for spatial support, but it is not the requested output.

Use:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/op_001_validate_submission.py
python ${CLAUDE_SKILL_DIR}/scripts/op_012_recon_diagnostics.py
```
<!-- ORACLE_OP_END op_012_smoke_test_and_diagnostics -->
