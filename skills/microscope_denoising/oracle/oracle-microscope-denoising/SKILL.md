---
name: oracle-microscope-denoising
description: "Zero-shot self-supervised restoration of low-SNR fluorescence microscopy images: jointly train a denoiser U-Net and a deconvolver U-Net directly on noisy frames via recorruption (no clean reference), then run two-stage sliding-window inference to produce denoised and PSF-deconvolved outputs."
---

# oracle-microscope-denoising Oracle Skill

This skill restores a stack of noisy TIRF fluorescence microscopy frames using the ZS-DeconvNet recorruption scheme. It covers reading the run-local case inputs, estimating the Poisson-Gaussian noise parameters, generating self-supervised recorrupted training patch pairs, building two U-Nets, training them jointly with a denoising loss plus a PSF-consistency deconvolution loss with a Hessian smoothness penalty, running two-stage patch inference with tapered blending, and writing the required output arrays. Each operation is independently removable so ablation can test which conventions and model components matter. Use the run-local Python interpreter on PATH (or the runtime venv Python under public/envs) for all execution; never a system python3.

<!-- ORACLE_OP_START op_010_public_contract -->
## Case discovery and submission contract

## Case discovery and submission contract

Process every case listed in `public/visible_data/cases.json`. For each case id (e.g. `case_000`), inputs live under `public/visible_data/cases/<case_id>/` and the result MUST be written to `outputs/<case_id>.npz` relative to the run root.

Each output `.npz` MUST contain exactly two arrays:

| key | shape | dtype | meaning |
|-----|-------|-------|---------|
| `denoised` | (502, 502) | float32 or float64 | Stage-1 denoised image |
| `deconvolved` | (502, 502) | float32 or float64 | Stage-2 denoised + PSF-deconvolved image |

Hard invariants enforced by validation:
- Both arrays are required and must have shape exactly (502, 502).
- All values must be finite (no NaN/Inf). Sanitize before writing.
- No dtype casting is allowed downstream: write arrays already as float32 or float64.
- Do not add extra keys with different shapes that could break the reader.

Create the `outputs/` directory if missing. Write with `numpy.savez` using the exact keys `denoised` and `deconvolved`.

See `${CLAUDE_SKILL_DIR}/resources/op_010_public_contract.md` for the exact schema summary.
<!-- ORACLE_OP_END op_010_public_contract -->

<!-- ORACLE_OP_START op_020_data_loading -->
## Load raw measurements, PSF, and case parameters

## Load raw measurements, PSF, and case parameters

For each case, read from `public/visible_data/cases/<case_id>/`:

- `input_data/raw_data.npz`:
  - `measurements`: shape (55, 502, 502), float32 — 55 independent low-SNR TIRF frames (camera background ~100 ADU, mean signal above background ~10 ADU). Cast to float64 for all numeric processing.
  - `psf`: shape (1, 32, 32), float32 — unit-sum PSF kernel. Take `psf[0]` to get the (32, 32) 2-D kernel; keep it as-is (already normalized to sum 1).
- `params_data.json` (the `params` file named in the case entry). Read at least:
  - `background_adu` (b, the camera bias offset; 100 for the public case).
  - `noise_model.beta1_estimated`, `noise_model.beta2_estimated` (fallback noise parameters).
  - `noise_model.filter_size` (local-averaging window size, 5).

The designated test frame for producing outputs is frame index 0 (`measurements[0]`). All 55 frames are usable as a source of training patches.

Conventions: images are raw ADU values (not normalized); keep them in ADU until an explicit normalization step. Load once per case and reuse.
<!-- ORACLE_OP_END op_020_data_loading -->

<!-- ORACLE_OP_START op_030_noise_model -->
## Poisson-Gaussian pixel noise variance model

## Poisson-Gaussian pixel noise variance model

Fluorescence images under the sCMOS Gaussian approximation follow `y = x + n`, `n ~ N(0, sigma^2(x))` with a signal-dependent, spatially varying variance:

```
sigma2(x) = beta1 * max(H(x - b), 0) + beta2
```

where:
- `b` is the camera background offset (`background_adu`, e.g. 100).
- `H` is a local averaging (uniform/box) filter of size `filter_size` (=5) applied to `x - b`, estimating the local mean signal. Use `scipy.ndimage.uniform_filter(x - b, size=5)`.
- `max(..., 0)` clamps the local signal estimate to be non-negative before scaling.
- `beta1` is the Poissonian (shot-noise) scaling factor (theoretical optimum 1.0; estimated ~0.75 for this data).
- `beta2` is the Gaussian read-noise variance (estimated ~3.7 for this data).

This variance map is the shared foundation for both noise-parameter estimation and the recorruption noise draw. Compute in float64.

Reference implementation of the variance map is in `${CLAUDE_SKILL_DIR}/resources/op_030_noise_model.md`.
<!-- ORACLE_OP_END op_030_noise_model -->

<!-- ORACLE_OP_START op_040_noise_param_estimation -->
## Estimate beta1 and beta2 by block variance-vs-mean regression

## Estimate beta1 and beta2 by block variance-vs-mean regression

Estimate the noise parameters directly from a single noisy frame (frame 0) so the recorruption matches the data. Procedure:

1. Partition the (H, W) image into non-overlapping 32x32 blocks (loop `i` and `j` in steps of 32, stopping before the last partial block: `range(0, H-32, 32)`).
2. For each block:
   - `smooth = uniform_filter(block, size=filter_size)` (filter_size=5).
   - `signal_est = mean(max(smooth - b, 0))` — the block's local mean signal above background.
   - `residual = block - smooth`; `var_est = residual.var()` — the block's noise variance.
3. Collect `(signal_est, var_est)` over all blocks and fit a line `var = beta1 * signal + beta2` via least-squares (`scipy.stats.linregress`): slope = beta1, intercept = beta2.
4. Floor both: `beta1 = max(slope, 0.1)`, `beta2 = max(intercept, 0.1)`.

If estimation is unstable, fall back to the `noise_model.beta1_estimated` / `beta2_estimated` values from the case parameters. Compute in float64.

Full helper in `${CLAUDE_SKILL_DIR}/resources/op_040_noise_param_estimation.md`.
<!-- ORACLE_OP_END op_040_noise_param_estimation -->

<!-- ORACLE_OP_START op_050_recorruption -->
## Recorruption pair generation

## Recorruption pair generation

The self-supervised training signal comes from recorrupting each noisy patch `y`. Draw one noise map with the same variance structure as the original noise and split into a more-noisy input and a less-noisy target:

```
sigma2 = beta1 * max(H(y - b), 0) + beta2        # per-pixel, from the noise model
g      = standard_normal(y.shape) * sqrt(sigma2)  # fresh Gaussian noise map
y_hat  = y + alpha * g                            # more-noisy -> network input
y_bar  = y - g / alpha                            # less-noisy -> training target
```

- Use `alpha = 1.5` (corruption strength; results are robust for alpha in [1, 2]; larger alpha => noisier input and cleaner target but a harder learning problem).
- The corruption is purely additive Gaussian noise sharing the original variance map — no blurring, no spatial distortion.
- Both `y_hat` and `y_bar` are unbiased estimates of the underlying clean signal, so training a net to map `y_hat -> y_bar` is equivalent to supervised denoising.
- Use a seeded `numpy.random.default_rng` for reproducibility and compute in float64.

See `${CLAUDE_SKILL_DIR}/resources/op_050_recorruption.md`.
<!-- ORACLE_OP_END op_050_recorruption -->

<!-- ORACLE_OP_START op_060_patch_extraction -->
## Training patch sampling, augmentation, and joint normalization

## Training patch sampling, augmentation, and joint normalization

Build the training set of recorrupted pairs from the frame stack:

1. Target total ~10000 pairs; distribute as `patches_per_img = max(1, n_patches // n_frames)` across all 55 frames.
2. For each patch:
   - Random square crop of size 128x128: `i = rng.integers(0, H-128)`, `j = rng.integers(0, W-128)`.
   - Augment: random `numpy.rot90(patch, k)` with `k` in {0,1,2,3}, then horizontal flip (`numpy.fliplr`) with probability 0.5.
   - Recorrupt the augmented patch into `(y_hat, y_bar)` (previous operation).
   - Joint normalization: compute `lo, hi = percentile(y_hat, 0), percentile(y_hat, 100)` (i.e. min/max of the INPUT). Normalize BOTH with the same `lo/hi`: `y_hat_n = clip((y_hat - lo)/(hi - lo + eps), 0, 1)` and `y_bar_n = clip((y_bar - lo)/(hi - lo + eps), 0, 1)` with `eps = 1e-6`. Never renormalize the target independently.
3. Stack into arrays of shape (N, 1, 128, 128), dtype float32. Add the channel axis at position 1.

Use a fixed seed (e.g. 42). See `${CLAUDE_SKILL_DIR}/resources/op_060_patch_extraction.md`.
<!-- ORACLE_OP_END op_060_patch_extraction -->

<!-- ORACLE_OP_START op_070_unet_architecture -->
## U-Net backbone for denoiser and deconvolver

## U-Net backbone for denoiser and deconvolver

Both stages use the same U-Net backbone (two independent instances). Structure:

- Building block `DoubleConv(in, out)`: `Conv2d(in,out,3,padding=1) -> ReLU -> Conv2d(out,out,3,padding=1) -> ReLU` (same-padding, ReLU inplace).
- Base channel count `base = 32`.
- Encoder: `enc1: 1->32`, `enc2: 32->64`, `enc3: 64->128`, `enc4: 128->256`, each followed by `MaxPool2d(2,2)` before the next.
- Bottleneck: `DoubleConv(256, 256)` on the pooled `enc4`.
- Decoder with skip connections: upsample with `F.interpolate(..., size=skip.shape[2:], mode='bilinear', align_corners=True)`, concatenate the matching encoder feature on the channel dim, then `DoubleConv`:
  - `dec4: 256+256 -> 128`, `dec3: 128+128 -> 64`, `dec2: 64+64 -> 32`, `dec1: 32+32 -> 32`.
- Output head: `Conv2d(32, 1, kernel_size=1)` with NO activation (linear output).

Input/output tensors are (B, 1, H, W). Instantiate one U-Net for denoising (Stage 1) and one for deconvolution (Stage 2).

Full module in `${CLAUDE_SKILL_DIR}/resources/op_070_unet_architecture.md`.
<!-- ORACLE_OP_END op_070_unet_architecture -->

<!-- ORACLE_OP_START op_080_hessian_regularizer -->
## Hessian smoothness regularizer for deconvolution

## Hessian smoothness regularizer for deconvolution

Deconvolution amplifies high-frequency oscillations (ringing). Penalize curvature with the sum of squared second-order finite differences of the Stage-2 output `z` (shape (B,1,H,W)):

```
R_Hessian(z) = mean(dxx^2) + mean(dyy^2) + mean(dxy^2) + mean(dyx^2)
```

Compute via first then second differences:
- `dx = z[..., :, 1:] - z[..., :, :-1]`   (difference along width)
- `dy = z[..., 1:, :] - z[..., :-1, :]`   (difference along height)
- `dxx = dx[..., :, 1:] - dx[..., :, :-1]`
- `dyy = dy[..., 1:, :] - dy[..., :-1, :]`
- `dxy = dy[..., :, 1:] - dy[..., :, :-1]`
- `dyx = dx[..., 1:, :] - dx[..., :-1, :]`

Return the sum of the four `.pow(2).mean()` terms. This penalizes curvature rather than edges, preserving sharp filaments while suppressing ringing. It is added to the Stage-2 loss with weight `lambda = 0.02`.

See `${CLAUDE_SKILL_DIR}/resources/op_080_hessian_regularizer.md`.
<!-- ORACLE_OP_END op_080_hessian_regularizer -->

<!-- ORACLE_OP_START op_090_psf_conv_layer -->
## PSF convolution layer for deconvolution consistency

## PSF convolution layer for deconvolution consistency

Stage 2 is physically grounded by re-blurring its output with the known PSF and comparing to the target. Implement the PSF as a fixed convolution kernel:

- Build the kernel `psf_t` of shape (1, 1, kH, kW) from the (32, 32) unit-sum PSF (float).
- Because the kernel size is even (32), use asymmetric reflect padding so the convolution output matches the input size:
  - `ph_lo = (kH-1)//2`, `ph_hi = kH-1-ph_lo`; `pw_lo = (kW-1)//2`, `pw_hi = kW-1-pw_lo`.
  - `z_pad = F.pad(z, (pw_lo, pw_hi, ph_lo, ph_hi), mode='reflect')`.
  - `z_conv = F.conv2d(z_pad, psf_t, padding=0)`.
- The kernel is not a learnable parameter; keep it fixed (do not register as trainable).

The resulting `z_conv` (the deconvolved output re-blurred by the PSF) feeds the Stage-2 data term. See `${CLAUDE_SKILL_DIR}/resources/op_090_psf_conv_layer.md`.
<!-- ORACLE_OP_END op_090_psf_conv_layer -->

<!-- ORACLE_OP_START op_100_joint_training -->
## Joint two-stage training loop and optimizer schedule

## Joint two-stage training loop and optimizer schedule

Train the denoiser (Stage 1) and deconvolver (Stage 2) end-to-end simultaneously on the recorrupted patch pairs `(y_hat -> y_bar)`.

Forward per mini-batch (input `xb = y_hat`, target `yb = y_bar`):
```
out_den = f_den(xb)
loss_den = MSE(out_den, yb)                         # Stage 1 data term
out_dec = f_dec(out_den)                            # Stage 2 takes Stage-1 output
out_dec_conv = psf_blur(out_dec)                    # re-blur with PSF (reflect asym pad)
loss_dec = MSE(out_dec_conv, yb) + lambda * hessian_loss(out_dec)
loss = mu * loss_den + (1 - mu) * loss_dec
```

Constants and schedule:
- `mu = 0.5`, `lambda (hess_weight) = 0.02`.
- Optimizer: `Adam` over the concatenation of both models' parameters, initial `lr = 5e-4`.
- LR schedule: `StepLR(step_size=10000, gamma=0.5)` (halve lr every 10000 steps), step every iteration.
- `batch_size = 4`, `n_iters = 30000`, DataLoader with `shuffle=True, drop_last=True`, re-iterating when exhausted.
- Put both models in `.train()`; use GPU if available else CPU. Record `(loss, loss_den, loss_dec)` every 100 steps for diagnostics.
- After training, move both models to CPU and set `.eval()` for inference.

If compute/time is constrained, `n_iters` may be reduced, but keep the loss composition, mu, lambda, and the PSF-consistency term intact. Full loop in `${CLAUDE_SKILL_DIR}/resources/op_100_joint_training.md`.
<!-- ORACLE_OP_END op_100_joint_training -->

<!-- ORACLE_OP_START op_110_sliding_window_inference -->
## Two-stage sliding-window inference with tapered blending

## Two-stage sliding-window inference with tapered blending

Restore the full (502, 502) test frame (`measurements[0]`) with patch inference.

Normalization: `lo = percentile(y, 0)`, `hi = percentile(y, 100)` (min/max); `y_norm = clip((y-lo)/(hi-lo+1e-6), 0, 1)` as float32. Keep `lo, hi` to rescale outputs back to ADU.

Sliding window over `y_norm`:
- `patch_size = 128`, `overlap = 32`, `stride = patch_size - overlap = 96`.
- Enumerate top-left positions with `range(0, dim - patch_size + 1, stride)` in both axes; additionally append edge rows/cols at `dim - patch_size` and the bottom-right corner so the whole image is covered.
- Run the model on each patch (batched); collect predictions.
- Blend with a floored Hann window: `h = numpy.hanning(patch_size)`, then `h = 0.1 + 0.9*h` (floor 0.1 so corner pixels keep weight); 2-D weight = `outer(h, h)`. Accumulate `output += patch * blend` and `weight += blend`, then `output / max(weight, 1e-6)`.

Two stages:
1. Stage 1 (denoiser) on `y_norm` -> `den_norm`; `denoised = den_norm*(hi-lo) + lo` (float32).
2. Stage 2 (deconvolver) on `clip(den_norm, 0, 1)` (the NORMALIZED Stage-1 output, not the ADU-rescaled one) -> `dec_norm`; `deconvolved = dec_norm*(hi-lo) + lo` (float32).

Return `denoised` and `deconvolved`, both (502, 502). Full routine in `${CLAUDE_SKILL_DIR}/resources/op_110_sliding_window_inference.md`.
<!-- ORACLE_OP_END op_110_sliding_window_inference -->

<!-- ORACLE_OP_START op_120_output_writing -->
## Assemble and write per-case output npz

## Assemble and write per-case output npz

For each case, package the two restored arrays and write to `outputs/<case_id>.npz`:

- Keys exactly `denoised` and `deconvolved`.
- Each shape (502, 502), dtype float32 (float64 also accepted).
- Sanitize to finite values before writing: `numpy.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)`.
- Do not rely on downstream casting (allow_cast is false) — ensure dtype is float32/float64 at write time.
- Create `outputs/` if it does not exist.

```python
import os, numpy as np
os.makedirs('outputs', exist_ok=True)
den = np.nan_to_num(denoised).astype(np.float32)
dec = np.nan_to_num(deconvolved).astype(np.float32)
np.savez(f'outputs/{case_id}.npz', denoised=den, deconvolved=dec)
```
<!-- ORACLE_OP_END op_120_output_writing -->

<!-- ORACLE_OP_START op_130_psf_residual_probe -->
## Deconvolution PSF-consistency and sharpness self-check

## Deconvolution PSF-consistency and sharpness self-check

A cheap, non-mutating diagnostic to sanity-check outputs before finalizing (not required for the submission, and does not read any reference data):

- PSF residual (Stage-2 consistency): re-blur the deconvolved image with the PSF and compare to the noisy frame:
  `residual = mean((conv(deconvolved, psf) - noisy)^2) / (mean(noisy^2) + 1e-10)`. Expect a small value (order < ~0.01) if deconvolution is physically consistent.
- Background-noise reduction: on the darkest 32x32 block, `bg_std_noisy / bg_std_denoised` should be well above 1 (target ~5-10x).
- Sharpness: `scipy.ndimage.laplace(img).var()` should increase from denoised to deconvolved (target several-fold).

Use `${CLAUDE_SKILL_DIR}/scripts/op_130_psf_residual_probe.py` to compute these on saved arrays. Treat failures as a signal to revisit training length or the PSF-consistency term, not as a hard gate.
<!-- ORACLE_OP_END op_130_psf_residual_probe -->
