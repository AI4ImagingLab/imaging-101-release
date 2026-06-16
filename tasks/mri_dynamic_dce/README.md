# Dynamic Contrast-Enhanced (DCE) MRI Reconstruction

> Reconstruct a time-series of MRI images from per-frame undersampled 2D Cartesian k-space by exploiting temporal sparsity via temporal Total Variation regularization.

> Domain: Medicine | Keywords: dynamic MRI, temporal regularization | Difficulty: Medium

## Background

Dynamic Contrast-Enhanced (DCE) MRI monitors the passage of a contrast agent (e.g., gadolinium chelate) through tissue over time. A bolus of contrast agent is injected intravenously and a rapid time-series of MRI images is acquired. The signal intensity in each voxel follows a characteristic uptake curve determined by local tissue perfusion, vascular permeability, and extracellular volume.

Because each time frame must be acquired quickly to capture the fast dynamics of contrast agent wash-in and wash-out, the k-space data for each frame is heavily undersampled. The key insight for reconstruction is that consecutive frames share most of their content -- only the enhancing regions change between frames. This temporal redundancy can be exploited through regularization that penalizes temporal differences, allowing accurate reconstruction from far fewer measurements per frame than would be needed for independent frame-by-frame reconstruction.

## Problem Description

The forward model for each time frame $t$ is:

$$y_t = M_t \cdot \mathcal{F}(x_t) + \eta_t, \quad t = 1, \ldots, T$$

where:
- $x_t \in \mathbb{C}^{N \times N}$ is the complex image at frame $t$
- $\mathcal{F}$ is the centered 2D discrete Fourier transform (ortho-normalized)
- $M_t \in \{0, 1\}^{N \times N}$ is a binary 2D undersampling mask for frame $t$, selecting a variable-density random subset of k-space points
- $y_t \in \mathbb{C}^{N \times N}$ is the measured (zero-filled) k-space
- $\eta_t$ is complex Gaussian measurement noise

Each frame uses a different random mask $M_t$, so the aliasing artifacts are incoherent across time. The undersampling rate is 15% per frame (approximately 6.7x acceleration).

The inverse problem is ill-posed because each frame's k-space is severely undersampled. However, temporal redundancy can be exploited — the image sequence changes slowly and sparsely over time, enabling joint reconstruction of all frames.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `undersampled_kspace` | `(1, 20, 128, 128)` | complex64 | Per-frame undersampled k-space measurements (zero-filled at unsampled locations) |
| `undersampling_masks` | `(1, 20, 128, 128)` | float32 | Binary 2D variable-density random undersampling masks, different per frame |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `dynamic_images` | `(1, 20, 128, 128)` | float32 | Ground truth dynamic phantom images, real-valued non-negative, intensity in arbitrary units |
| `time_points` | `(20,)` | float32 | Time axis in seconds (0 to 60 s) |

### data/meta_data.json

Imaging parameters including image size (128), number of frames (20), sampling rate (0.15), noise level (0.005), and time range.

## Method Hints

- **Baseline**: Frame-by-frame inverse FFT of undersampled k-space (zero-filled reconstruction). Fast but produces temporal flickering from incoherent aliasing.
- **Temporal TV regularization**: Jointly reconstruct all frames with an L1 penalty on temporal finite differences, exploiting the fact that frame-to-frame changes are spatially sparse. This promotes piecewise-constant temporal behaviour and suppresses temporal flickering artifacts. Proximal gradient methods can handle the non-smooth temporal TV term.

## References

- Lustig, M., Donoho, D., & Pauly, J. M. (2007). Sparse MRI: The application of compressed sensing for rapid MR imaging. Magnetic Resonance in Medicine, 58(6), 1182-1195.
- Chambolle, A. (2004). An algorithm for total variation minimization and applications. Journal of Mathematical Imaging and Vision, 20(1-2), 89-97.
- Feng, L., et al. (2014). Golden-angle radial sparse parallel MRI: Combination of compressed sensing, parallel imaging, and golden-angle radial sampling for fast and flexible dynamic volumetric MRI. Magnetic Resonance in Medicine, 72(3), 707-717.
- Demo DCE Recon: https://github.com/ZhengguoTan/demo_dce_recon
