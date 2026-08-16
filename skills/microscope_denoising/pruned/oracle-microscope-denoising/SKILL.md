---
name: oracle-microscope-denoising
description: "Zero-shot self-supervised restoration of low-SNR fluorescence microscopy images: jointly train a denoiser U-Net and a deconvolver U-Net directly on noisy frames via recorruption (no clean reference), then run two-stage sliding-window inference to produce denoised and PSF-deconvolved outputs."
---

# oracle-microscope-denoising Oracle Skill

This skill restores a stack of noisy TIRF fluorescence microscopy frames using the ZS-DeconvNet recorruption scheme. It covers reading the run-local case inputs, estimating the Poisson-Gaussian noise parameters, generating self-supervised recorrupted training patch pairs, building two U-Nets, training them jointly with a denoising loss plus a PSF-consistency deconvolution loss with a Hessian smoothness penalty, running two-stage patch inference with tapered blending, and writing the required output arrays. Each operation is independently removable so ablation can test which conventions and model components matter. Use the run-local Python interpreter on PATH (or the runtime venv Python under public/envs) for all execution; never a system python3.

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

See `${CLAUDE_SKILL_DIR}/resources/resource_001.md`.

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

Return `denoised` and `deconvolved`, both (502, 502). Full routine in `${CLAUDE_SKILL_DIR}/resources/resource_002.md`.
