# Fourier Ptychographic Microscopy Image Stack Reconstruction

> Reconstruct a high-resolution 3D complex field (amplitude and phase) from multiplexed low-resolution intensity measurements captured under varying LED illumination angles, where the inverse problem is ill-posed due to phase loss and limited per-image bandwidth.

> Domain: Medicine / Biophotonics | Keywords: computational microscopy, phase retrieval, implicit neural representation | Difficulty: Medium

## Background

Fourier Ptychographic Microscopy (FPM) is a computational imaging technique that achieves both high resolution and wide field of view by capturing multiple low-resolution images under varying LED illuminations. Each LED illuminates the sample from a different angle, causing a different sub-band of the sample's spatial frequency spectrum to fall within the objective's limited numerical aperture (NA). By combining information from all LEDs, FPM can synthesize an image with resolution far exceeding the objective's native resolving power.

The sample in this task is a human blood smear slide tilted approximately 4 degrees relative to the optical axis, creating a depth-varying specimen that requires 3D reconstruction across 161 z-planes spanning -20 to +20 um.

## Problem Description

Given $L = 68$ intensity-only measurements $\{I_\ell\}_{\ell=1}^{L}$ from LEDs at known illumination angles, recover the 3D complex field $O(x, y, z) = A(x, y, z) \exp(i\phi(x, y, z))$ — both amplitude $A$ and phase $\phi$ — across a volume.

The forward model for LED $\ell$ at depth $z$ is:

$$I_\ell(x, y; z) = \left| \mathcal{F}^{-1}\left\{ \tilde{O}_{\text{sub},\ell}(k_x, k_y) \cdot H(k_x, k_y; z) \right\} \right|^2$$

where:

- $\tilde{O}(k_x, k_y) = \mathcal{F}\{O(x, y, z)\}$ is the 2D Fourier transform of the complex field at depth $z$
- $\tilde{O}_{\text{sub},\ell}$ is the sub-band of $\tilde{O}$ extracted at the spatial frequency offset $(k_0 u_\ell, k_0 v_\ell)$ corresponding to LED $\ell$'s illumination angle $(u_\ell, v_\ell)$, windowed by the pupil function $P(k_x, k_y)$
- $H(k_x, k_y; z) = P(k_x, k_y) \cdot \exp\!\bigl(i k_z z\bigr)$ is the angular spectrum propagation kernel with $k_z = \sqrt{k_0^2 - k_x^2 - k_y^2}$ and $k_0 = 2\pi / \lambda$
- $\lambda = 0.5126\;\mu\text{m}$ (green channel wavelength)
- The pupil $P(k_x, k_y)$ is a binary disk defined by the objective's numerical aperture: $P = 1$ where $\sqrt{k_x^2 + k_y^2} \le \text{NA} \cdot k_0$

The problem is ill-posed because (1) each measurement captures only intensity (phase is lost), (2) each LED probes only a narrow sub-band of the spectrum, and (3) the 3D field must be recovered from 2D measurements.

**Input**: 68 low-resolution intensity images (1024 x 1024 pixels each) plus calibrated LED positions in NA space.

**Output**: High-resolution 3D complex field at 2048 x 2048 pixels (2x upsampled) across 161 z-slices, from which an all-in-focus image is derived.

## Data Description

### `data/raw_data.npz`

Low-resolution FPM intensity measurements under 68 LED illuminations (green channel).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `I_low` | `(1, 1024, 1024, 68)` | float32 | Raw intensity images. Axes: (batch, H, W, LED index). Units: arbitrary intensity counts. |
| `na_calib` | `(1, 68, 2)` | float32 | Calibrated illumination NA for each LED. Axes: (batch, LED index, [NAx, NAy]). Dimensionless. |
| `mag` | `()` | float32 | Microscope magnification factor (10x). Dimensionless. |
| `dpix_c` | `()` | float32 | Camera pixel pitch in micrometers (3.45 um). |
| `na_cal` | `()` | float32 | Objective numerical aperture (0.256). Dimensionless. |

### `data/ground_truth.npz`

High-resolution amplitude z-stack from conventional FPM reconstruction, used as ground truth for evaluation.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `I_stack` | `(1, 1024, 1024, 161)` | float32 | Ground truth amplitude z-stack. Axes: (batch, H, W, z-slice). Units: arbitrary amplitude. |
| `zvec` | `(1, 161)` | float32 | Z-positions corresponding to each slice, linearly spaced from -20 to +20 um. |

### `data/meta_data.json`

Imaging parameters and training hyperparameters. Key fields include: wavelength (0.5126 um), objective NA (0.256), magnification (10x), pixel pitch (3.45 um), 68 LEDs, 512 spatial frequency modes, 2x upsampling, z-range [-20, +20] um with 161 evaluation slices.

**Data source**: https://doi.org/10.22002/7aer7-qhf77

## Method Hints

Represent the 3D complex field as two continuous functions (one for amplitude, one for phase) parameterized by implicit neural representations. Each representation uses a factored spatial-depth architecture: a learnable 2D feature grid sampled via bilinear interpolation is combined with a 1D depth feature array sampled via linear interpolation, fused through element-wise multiplication, and decoded by a small MLP to produce a scalar value at each (x, y, z) location.

Optimize by minimizing the discrepancy between predicted and measured amplitudes across all LED illuminations and sampled depth planes. An alternating z-sampling strategy (uniform grid on even iterations, random subset on odd) improves convergence.

An all-in-focus composite image can be derived from the reconstructed z-stack using a Normal Variance focus measure that selects the sharpest focal plane per overlapping patch.

## References

- Zhou, H., Feng, B.Y., Guo, H., Lin, S., Liang, M., Metzler, C.A., & Yang, C. (2023). "Fourier ptychographic microscopy image stack reconstruction using implicit neural representations." *Optica*, 10(12), 1679-1687. https://doi.org/10.1364/OPTICA.505283
- Project page: https://hwzhou2020.github.io/FPM-INR-Web/
- arXiv: https://arxiv.org/abs/2310.18529
