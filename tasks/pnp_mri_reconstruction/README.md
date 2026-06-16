# PnP-MSSN: Plug-and-Play MRI Reconstruction with Multiple Self-Similarity Network

| Property | Value |
|---|---|
| **Domain** | Medicine |
| **Modality** | Magnetic Resonance Imaging (MRI) |
| **Difficulty** | Medium |
| **Forward model** | Subsampled 2D Fourier transform |
| **Inverse method** | Plug-and-Play ADMM/PGM with learned denoiser |
| **Reference** | Song et al., "A New Recurrent Plug-and-Play Prior Based on the Multiple Self-Similarity Network," IEEE SPL, 2020 |

## Background

MRI acquires data in **k-space** (spatial frequency domain). Full k-space sampling is slow, so accelerated MRI uses **radial undersampling** — acquiring only a subset of radial lines through the k-space center. Reconstruction from these incomplete measurements is an ill-posed inverse problem.

### Physical model

The forward model maps an image $x \in \mathbb{R}^{N \times N}$ to subsampled k-space measurements:

$$y = M \cdot \mathcal{F}(x) + \eta$$

where:
- $\mathcal{F}$: 2D discrete Fourier transform
- $M$: binary sampling mask (radial lines in k-space)
- $\eta$: measurement noise

The naive reconstruction via inverse FFT of the zero-filled k-space produces severe **streaking artifacts** from radial undersampling.

### Plug-and-Play framework

The Plug-and-Play Priors (PnP) framework solves the regularized inverse problem:

$$\hat{x} = \arg\min_x \frac{1}{2} \|y - M \mathcal{F}(x)\|_2^2 + \lambda R(x)$$

by replacing the proximal operator of the regularizer $R(x)$ with a learned image denoiser. The PnP proximal gradient method (PGM) alternates:

1. **Gradient step:** $s = x^{(k)} - \gamma \nabla_x \frac{1}{2}\|y - M\mathcal{F}(x^{(k)})\|^2$
2. **Denoising step:** $x^{(k+1)} = D_\sigma(s)$

where $D_\sigma$ is an image denoiser trained at noise level $\sigma$.

### MSSN denoiser

The Multiple Self-Similarity Network (MSSN) is a recurrent neural network denoiser that uses **multi-head attention** to exploit non-local self-similarity in images. It processes image patches through:
- A shared recurrent residual block with multiple states
- Multi-head attention that captures both dimension-wise and sequence-wise correlations
- Patch-based inference with overlapping patches and averaging

The MSSN is pre-trained on the BSD500 dataset at noise level $\sigma = 5$.

## Data

- **Source:** [fastMRI dataset](https://fastmri.org/) — single knee MRI slice
- **Image size:** 320 x 320 pixels, normalized to [0, 1]
- **Sampling:** 36-line radial k-space mask

### `data/raw_data.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `img` | `(1, 320, 320)` | `float32` | Raw MRI knee image (unnormalized). Normalized to [0, 1] during preprocessing. |

### `data/ground_truth.npz`

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `img` | `(1, 320, 320)` | `float32` | Ground truth MRI image (same as raw_data, used for evaluation). |

### `data/meta_data.json`

Imaging and reconstruction parameters: image size, number of radial lines (36), PGM iterations (200), step size, MSSN denoiser config (patch size, stride, sigma, state count), and model checkpoint path.

## Method hints

- Use **Proximal Gradient Method (PGM)** with the MSSN denoiser as the proximal operator
- The denoiser operates on **patches** (42 x 42) with stride 7, averaging overlapping regions
- Input images are scaled to [0, 255] before denoising, output scaled back to [0, 1]
- Clip gradient step output to $[0, \infty)$ for positivity
- Pre-trained MSSN checkpoint is provided (trained at $\sigma = 5$)
- 200 PGM iterations with step size 1.0, no acceleration
