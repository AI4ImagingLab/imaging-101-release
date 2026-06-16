# Zero-Shot Fluorescence Microscopy Restoration (ZS-DeconvNet)

> Restore low-SNR fluorescence microscopy images — removing noise and reversing PSF blurring — using only the noisy observations themselves, without any ground truth data.

> Domain: Biology | Keywords: denoising, deconvolution, self-supervised learning | Difficulty: Medium

## Background

Live-cell fluorescence microscopy is fundamentally limited by the trade-off between
signal-to-noise ratio (SNR) and phototoxicity: acquiring high-SNR images requires
high laser power or long exposure, which damages living cells and bleaches fluorophores.
At low-photon-count conditions (mean signal ~10 ADU above a ~100 ADU camera background),
fine biological structures such as microtubule filaments (~200 nm diameter) are buried
in noise and blurred by the optical point-spread function (PSF).

Classical denoising (Gaussian filtering) and deconvolution (Richardson-Lucy) make
explicit assumptions that fail at very low SNR. Supervised deep learning achieves
superior performance but requires large paired training datasets (noisy + clean images),
which are impractical to collect for live specimens.

**ZS-DeconvNet** (Qiao et al., *Nature Communications* 2024) solves both problems
simultaneously using a zero-shot self-supervised training scheme: two U-Nets are trained
directly on the noisy images themselves — no ground truth required — to jointly remove
noise (Stage 1) and reverse PSF blurring (Stage 2).

## Problem Description

A fluorescence microscopy image $y$ is a noisy, PSF-blurred observation of the true fluorescence signal $x$:

$$y = (x * \text{PSF}) + n, \quad n \sim \mathcal{N}(0, \sigma^2(x))$$

$$\sigma^2(x) = \beta_1 \cdot \max\!\bigl(H(x - b),\, 0\bigr) + \beta_2$$

where $*$ denotes convolution, $H$ is a local averaging filter, $b$ is the camera background offset (ADU), $\beta_1$ is a Poissonian scaling factor (signal-dependent shot noise), and $\beta_2$ is the Gaussian read noise variance.

The inverse problem is to recover $x$ (and optionally the PSF-deconvolved image) from a set of noisy observations $\{y_i\}$ — without access to any clean reference images.

## Data Description

### `data/raw_data.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `measurements` | (55, 502, 502) | float32 | 55 noisy TIRF Microtubule images. Camera background ~100 ADU; mean signal above background ~10 ADU (very low SNR). Each frame is an independent low-SNR acquisition. |
| `psf` | (1, 32, 32) | float32 | Center-cropped, unit-sum PSF kernel for the Microtubule channel (525 nm emission, NA 1.3, pixel size 0.0313 µm). Used in Stage 2 deconvolution training and PSF-residual evaluation. |

### `data/baseline_reference.npz`

No `ground_truth.npz` is provided for this task; since all 55 frames are from different cells (no pixel-level reference exists), `data/baseline_reference.npz` contains the ZS-DeconvNet outputs from the reference run (30 000 iterations), used for NCC/NRMSE evaluation.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `denoised` | (1, 502, 502) | float32 | Stage 1 denoised output for frame 01 |
| `deconvolved` | (1, 502, 502) | float32 | Stage 2 deconvolved output for frame 01 |

### `data/meta_data.json`

| Field | Value | Description |
|-------|-------|-------------|
| `modality` | `"TIRF fluorescence microscopy"` | Total internal reflection fluorescence |
| `structure` | `"Microtubule (filamentous)"` | Cytoskeletal filaments ~25 nm diameter |
| `pixel_size_um` | 0.0313 | Lateral pixel size in micrometres |
| `emission_wavelength_nm` | 525 | Fluorescence emission wavelength |
| `na` | 1.3 | Numerical aperture |
| `background_adu` | 100 | Camera bias offset $b$ |
| `noise_model.beta1_estimated` | ~0.75 | Poissonian factor (estimated from data) |
| `noise_model.beta2_estimated` | ~3.7 | Read noise variance (estimated from data) |
| `noise_model.alpha_recorruption` | 1.5 | Recorruption noise strength $\alpha$ |
| `psf.kernel_size` | 32 | PSF kernel size in pixels |

## Method Hints

**Algorithm family:** Self-supervised deep learning via recorruption-based training (zero-shot — no paired clean/noisy data required).

**Key idea:** A noisy image can serve as its own training target. By synthetically adding extra noise to create a "more corrupted" input and using the original noisy image as the target, one can prove that minimising this recorruption loss is equivalent to the supervised denoising objective. This allows training directly on the test data itself.

**Architecture:** Two-stage pipeline — a denoiser U-Net (Stage 1) followed by a deconvolver U-Net (Stage 2) trained jointly end-to-end. The PSF is applied as an explicit convolution layer in Stage 2's loss, making deconvolution physically grounded. A Hessian regulariser suppresses noise amplification during deconvolution.

**Why it suits the problem:** Live-cell imaging cannot produce paired clean/noisy training sets (acquiring a clean image would require destructive high-dose illumination). The recorruption scheme bypasses this entirely, needing only the noisy frames already collected.

## References

- Qiao C. et al., "Zero-shot learning enables instant denoising and super-resolution
  in optical fluorescence microscopy," *Nature Communications* 15, 4180 (2024).
  https://doi.org/10.1038/s41467-024-48575-9

- Zenodo dataset: https://doi.org/10.5281/zenodo.7261163

- GitHub: https://github.com/TristaZeng/ZS-DeconvNet
