# Lensless Imaging
> Reconstruct a scene image from a single lensless (DiffuserCam) sensor measurement by inverting a diffuser-encoded convolution.

> Domain: Computational Photography | Keywords: lensless imaging, deconvolution | Difficulty: Medium

## Background

A **lensless camera** replaces the conventional lens with a thin, flat diffuser
(e.g. Scotch tape).  Each point source in the scene produces a complex
"caustic" pattern — the **point spread function (PSF)** — that covers the entire
sensor.  The sensor measurement is therefore a superposition of shifted, scaled
copies of the PSF, one per scene point.

Although the raw sensor image looks like a blurred, structureless smear, all
scene information is preserved (under the shift-invariance approximation) and
can be recovered computationally.  The key advantage over a conventional camera
is that the imaging element is drastically thinner and cheaper, enabling
lightweight, miniaturised, or otherwise unconventional camera designs.

The forward model for this system is a **shift-invariant linear system**:

    b = crop( h * v )

where
- `v` ∈ ℝ^{H×W×C}  is the unknown scene image (H rows, W cols, C colour channels),
- `h` ∈ ℝ^{H×W×C}  is the measured PSF (same spatial dimensions as the sensor),
- `*` denotes 2-D linear convolution,
- `crop` extracts the central H×W region (the sensor extent trims the full
  (2H-1)×(2W-1) convolution result).

The goal is to **invert this model** — recover `v` from the measurement `b`
given the calibration PSF `h`.

## Problem Description

Given a lensless measurement `b` and a known PSF `h`, reconstruct the scene
image `v` by solving the regularised least-squares problem:

    min_{v ≥ 0}  (1/2) ||b - Av||²₂  +  τ ||∇v||₁

where `A = crop ∘ (h *)` is the lensless forward operator, and `||∇v||₁` is
the **total variation (TV)** regulariser (sum of absolute finite differences),
which promotes piecewise-smooth reconstructions.  The non-negativity constraint
`v ≥ 0` reflects that image intensities cannot be negative.

The system is ill-conditioned: the crop operation discards information, and
convolution with a diffuse PSF contracts the dynamic range of the measurement.
Regularisation is therefore essential.

## Data Description

All arrays are stored in `data/raw_data.npz` with batch-first shape `(1, ...)`.

### raw_data.npz

| Key           | Shape          | Dtype   | Description                                              |
|---------------|----------------|---------|----------------------------------------------------------|
| `measurement` | (1, 760, 1014, 3) | float32 | Normalised lensless sensor measurement (b), RGB, [0, 1] |
| `psf`         | (1, 760, 1014, 3) | float32 | Normalised PSF (h), RGB, [0, 1]                         |

Both arrays are derived from full-resolution (3040×4056) PNG images captured
with a Raspberry Pi HQ Camera behind a piece of Scotch tape (the diffuser).
Downsampled by factor 4 for computational tractability.  Background
(dark-current) is subtracted before normalisation.

### baseline_reference.npz

No `ground_truth.npz` is provided for this task; the task uses a real capture of a "thumbs-up" shown on a smartphone screen where no aligned ground-truth image was collected. `data/baseline_reference.npz` contains the ADMM reconstruction (300 iterations) used as the reference for NCC/NRMSE evaluation.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `reconstruction` | (1, 760, 1014, 3) | float32 | Reference ADMM reconstructed scene image, RGB, [0, 1] |

### meta_data.json

Imaging parameters (sensor model, pixel size, downsampling factor, etc.).

## Method Hints

Use the **Alternating Direction Method of Multipliers (ADMM)** with variable
splitting.  Introduce auxiliary variables `x = Mv`, `u = Ψv`, `w = v` to
decouple the forward model, the TV penalty, and the non-negativity constraint
into three cheap proximal sub-problems:

1. **u-update** (TV proximal): soft thresholding on the finite-difference
   image gradient `Ψv`.
2. **x-update** (data consistency): element-wise division in the padded spatial
   domain (the operator `CᵀC + μ₁I` is diagonal).
3. **w-update** (non-negativity): component-wise ReLU.
4. **v-update** (image synthesis): solved in the frequency domain because
   `MᵀM` and `ΨᵀΨ` are simultaneously diagonalised by the 2-D DFT.  The
   denominator `μ₁|H̃|² + μ₂|Ψ̃|² + μ₃` is precomputed once.

Each iteration runs in O(HW log HW) time thanks to the FFT.  Typical
convergence requires 100 iterations; see `plan/approach.md` for full details.

## References

- Biscarrat, C. & Parthasarathy, S. (2018). *Build your own DiffuserCam:
  Tutorial.* waller-lab.github.io/DiffuserCam/tutorial.html
- Bezzam, E. et al. (2023). *LenslessPiCam: A hardware and software platform
  for lensless computational imaging with a Raspberry Pi.* JOSS, 8(86).
  https://doi.org/10.21105/joss.04747
- Boyd, S. et al. (2011). *Distributed optimisation and statistical learning
  via the alternating direction method of multipliers.* Found. Trends Mach.
  Learn., 3(1), 1–122.
