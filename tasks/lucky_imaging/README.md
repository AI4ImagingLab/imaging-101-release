# Lucky Imaging: Planetary Surface Reconstruction

> Reconstruct a high-resolution lunar surface image from a short-exposure video sequence by exploiting rare moments of calm atmosphere, where the unknown space-variant atmospheric PSF makes direct deconvolution infeasible.

> Domain: Astronomy | Keywords: imaging through turbulence, lucky imaging, image sharpening | Difficulty: Medium

## Background

Ground-based astronomical imaging is fundamentally limited by atmospheric turbulence ("seeing"), which randomly distorts the incoming wavefront on timescales of tens of milliseconds. The Fried parameter $r_0$ characterises the turbulence strength: the effective angular resolution is $\lambda / r_0$ rather than $\lambda / D$, where $D$ is the telescope aperture. For typical amateur telescopes, $D / r_0 \gg 1$, so the telescope operates far below its diffraction limit.

**Lucky imaging** (Fried, 1978) exploits the statistical nature of turbulence. By recording thousands of short-exposure frames at high frame rates, a small fraction will have been captured during moments of unusually calm atmosphere. These "lucky" frames contain substantially more high-frequency detail than the long-exposure average. The technique selects the sharpest frames, aligns them to compensate for tip-tilt wander, and stacks them to build signal-to-noise — recovering angular resolution much closer to the diffraction limit.

Modern implementations extend this idea with **local adaptive stacking**: different parts of the field of view may be sharpest in different frames. The image is divided into a grid of alignment points (APs), each ranked and stacked independently, then blended with smooth weights. This local approach recovers significantly more detail than global frame selection alone.

## Problem Description

Each observed frame $y_k$ is a degraded snapshot of the underlying sharp scene $x$:

$$y_k(\mathbf{r}) = h_k(\mathbf{r}) * x(\mathbf{r}) + n_k(\mathbf{r}), \quad k = 1, \ldots, K$$

where $h_k(\mathbf{r})$ is the instantaneous atmospheric point spread function, $*$ denotes spatial convolution, and $n_k$ is additive noise. The PSF varies both across frames (temporal turbulence) and across the field of view (anisoplanatism).

The inverse problem — recovering $x$ from $\{y_k\}$ — is ill-posed because $h_k$ is unknown, time-varying, and spatially non-uniform. Direct deconvolution is infeasible without estimating the PSF. The key observation is that the temporal statistics of atmospheric turbulence guarantee that a small fraction of frames will have been captured during unusually calm moments, where $h_k$ is nearly diffraction-limited. By identifying and combining these "lucky" frames rather than estimating the PSF, high-resolution reconstruction becomes tractable.

**Input:** $K = 101$ short-exposure RGB frames, 960 × 1280 pixels, 8-bit per channel.
**Output:** a single reconstructed image at 16-bit dynamic range.

## Data Description

### `raw_data.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `frames` | `(1, 101, 960, 1280, 3)` | uint8 | Short-exposure video frames, RGB, 8-bit per channel |

### `baseline_reference.npz`

No `ground_truth.npz` is provided for this task; `data/baseline_reference.npz` contains the reference stacked output produced by the PlanetarySystemStacker pipeline, used for NCC/NRMSE evaluation.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `stacked` | (1, 954, 1275, 3) | float32 | Reference stacked image (PlanetarySystemStacker), RGB, 16-bit range (uint16 source) |

### `meta_data.json`

| Key | Type | Description |
|-----|------|-------------|
| `n_frames` | int | Number of video frames (101) |
| `height` | int | Frame height in pixels (960) |
| `width` | int | Frame width in pixels (1280) |
| `n_channels` | int | Number of colour channels (3) |
| `fps` | float | Capture frame rate in Hz (59.0) |
| `bit_depth` | int | Bits per channel per pixel (8) |

## Method Hints

The reconstruction follows a **select-align-stack** pipeline that exploits the temporal statistics of atmospheric turbulence rather than estimating the PSF.

**Frame quality ranking.** A sharpness metric (e.g. Laplacian variance, computed on a monochrome blurred version of each frame) quantifies instantaneous seeing quality. Frames with higher sharpness contain more near-diffraction-limited detail. Normalising by mean brightness prevents bright frames from dominating the ranking.

**Multi-level cross-correlation alignment.** Frames must be registered to a common reference before stacking, or averaging would blur features. A coarse-to-fine normalised cross-correlation approach aligns each frame globally (correcting tip-tilt drift), with sub-pixel refinement via paraboloid fitting at the correlation peak.

**Local adaptive stacking.** Anisoplanatism means different regions of the field may be sharpest in different frames. Dividing the image into a grid of overlapping alignment points (APs) and selecting frames independently at each AP captures more detail than global frame selection. Patches are blended with smooth (triangular) weights to avoid seams.

**Post-processing: unsharp masking.** Even a perfect stack softens the image because it averages over residual sub-pixel misalignments. Unsharp masking — amplifying the difference between the stack and its blurred version — recovers the suppressed high-frequency detail. This is safe after stacking (where SNR is high) but would amplify noise if applied to a single frame.

## References

1. Fried, D. L. (1978). "Probability of getting a lucky short-exposure image through turbulence." *Journal of the Optical Society of America*, 68(12), 1651–1658.
2. Hempel, R. "PlanetarySystemStacker" — open-source Python implementation. https://github.com/Rolf-Hempel/PlanetarySystemStacker
3. Law, N. M., Mackay, C. D., & Baldwin, J. E. (2006). "Lucky imaging: high angular resolution imaging in the visible from the ground." *Astronomy & Astrophysics*, 446(2), 739–745.
