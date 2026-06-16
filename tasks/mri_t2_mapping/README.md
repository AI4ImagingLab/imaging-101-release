# Quantitative MRI T2 Mapping

> Estimate per-pixel T2 relaxation time maps from multi-echo spin-echo MRI magnitude data by fitting a mono-exponential signal decay model, a canonical parameter estimation inverse problem in quantitative MRI.

> Domain: medicine | Keywords: quantitative MRI, parameter estimation | Difficulty: Easy

## Background

In multi-echo spin-echo (MESE) MRI, a series of images is acquired at different echo times (TE). The signal at each voxel decays exponentially with a time constant T2, which is characteristic of the tissue type. Quantitative T2 mapping converts this qualitative signal decay into a physical tissue property map (in milliseconds), enabling standardized, reproducible tissue characterization for clinical diagnosis of edema, fibrosis, iron overload, and demyelination.

## Problem Description

The forward model for mono-exponential T2 decay at a single voxel is:

$$S(TE_n) = M_0 \cdot \exp\!\left(-\frac{TE_n}{T_2}\right)$$

where:
- $S(TE_n)$ is the measured signal intensity at echo time $TE_n$ (arbitrary units)
- $M_0$ is the equilibrium magnetization (proton density weighting, a.u.)
- $T_2$ is the transverse relaxation time (ms)
- $TE_n$ is the $n$-th echo time (ms), for $n = 1, \ldots, N_\text{echoes}$

In magnitude MRI, the measured signal is corrupted by Rician noise:

$$S_\text{noisy}(TE_n) = \left| S(TE_n) + \eta_n \right|, \quad \eta_n \sim \mathcal{CN}(0, \sigma^2)$$

where $\eta_n$ is complex circular Gaussian noise with variance $\sigma^2$ per channel (real and imaginary).

**Inverse problem:** Given noisy multi-echo measurements $\{S_\text{noisy}(TE_n)\}_{n=1}^{N}$ at each pixel, estimate the T2 relaxation time map and M0 proton density map.

**Ill-conditioning:** The problem becomes ill-conditioned when:
- SNR is low (Rician noise bias distorts the exponential decay, especially at long TE)
- Few echo times are available (underdetermined fitting)
- T2 is much shorter or longer than the TE range (decay is fully decayed or barely started)

The input is a 3D array of multi-echo signal intensities (Ny x Nx x N_echoes) and the output is a 2D T2 map (Ny x Nx) in milliseconds.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `multi_echo_signal` | `(1, 256, 256, 10)` | float32 | Multi-echo MRI magnitude signal (a.u.) at 10 echo times |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `T2_map` | `(1, 256, 256)` | float32 | Ground truth T2 relaxation time map (ms) |
| `M0_map` | `(1, 256, 256)` | float32 | Ground truth proton density map (a.u.) |
| `tissue_mask` | `(1, 256, 256)` | bool | Binary tissue mask (True where T2 > 0) |

### data/meta_data.json

| Key | Type | Description |
|-----|------|-------------|
| `image_size` | int | Image dimension (256) |
| `n_echoes` | int | Number of echo times (10) |
| `echo_times_ms` | list[float] | Echo times in ms: [10, 20, ..., 100] |
| `noise_sigma` | float | Rician noise standard deviation (0.02) |
| `signal_model` | str | Signal model type ("mono_exponential") |
| `modality` | str | Imaging modality |
| `field_of_view_mm` | float | Field of view in mm (220.0) |
| `pixel_size_mm` | float | Pixel size in mm |

The synthetic data is generated from a modified Shepp-Logan phantom with T2 values assigned per tissue region: bone/scalp (40 ms), white matter (70 ms), gray matter (80 ms), deep gray matter (120 ms), and CSF/bright tissue (150 ms).

## Method Hints

Two approaches for T2 estimation from the mono-exponential model:

1. **Log-linear regression** (fast baseline): Taking the logarithm of the signal linearizes the model: $\log S = \log M_0 - TE/T_2$. This becomes a simple linear regression problem solvable by ordinary least squares. It is fast but biased because Rician noise in the magnitude domain is not Gaussian in the log domain, especially at low SNR.

2. **Nonlinear least-squares fitting**: Directly fit $S = M_0 \exp(-TE/T_2)$ per pixel using iterative optimization (e.g., Levenberg-Marquardt). More accurate than log-linear fitting because it correctly accounts for the signal model, but computationally more expensive. Initialize with the log-linear estimate for robust convergence.

Both methods operate independently per pixel. A tissue mask should be applied to exclude background pixels where no meaningful signal exists.

## References

- Hennig, J. (1988). Multiecho imaging sequences with low refocusing flip angles. Journal of Magnetic Resonance, 78(3), 397-407.
- Prasloski, T., et al. (2012). Applications of stimulated echo correction to multicomponent T2 analysis. Magnetic Resonance in Medicine, 67(6), 1803-1814.
- BAMMri/MyoQMRI: https://github.com/BAMMri/MyoQMRI — Extended Phase Graph T2 mapping reference implementation.
