# MRI PnP-ADMM Reconstruction with Spectrally-Normalized Denoiser

> Reconstruct a 256x256 brain MRI image from 30% randomly sampled k-space using Plug-and-Play ADMM, where a pretrained DnCNN denoiser with real spectral normalization replaces the proximal operator for the regularization term, providing an implicit learned image prior with provable convergence.

> Domain: Medicine | Keywords: compressed sensing MRI, plug-and-play, learned prior | Difficulty: Medium

## Background

Compressed Sensing MRI (CS-MRI) accelerates acquisition by undersampling k-space (the Fourier domain of the image) and exploiting image structure during reconstruction. Classical CS-MRI uses explicit regularizers like Total Variation or wavelet sparsity. **Plug-and-Play (PnP)** methods replace these hand-crafted priors with a pretrained image denoiser, which implicitly defines a more expressive image prior.

The key challenge is convergence: naively plugging a denoiser into an iterative algorithm (ADMM, forward-backward splitting) gives no convergence guarantee. This task uses **Real Spectral Normalization (RealSN)** on the denoiser's convolutional layers to bound its Lipschitz constant, ensuring provable convergence of the PnP iterations.

## Problem Description

The forward model for single-coil Cartesian CS-MRI is:

$$y = M \odot \mathcal{F}(x) + \eta$$

where:
- $x \in \mathbb{R}^{256 \times 256}$ is the unknown brain image
- $\mathcal{F}$ is the 2D discrete Fourier transform
- $M \in \{0, 1\}^{256 \times 256}$ is a binary k-space undersampling mask (~30% sampling)
- $\eta \in \mathbb{C}^{256 \times 256}$ is complex Gaussian measurement noise
- $y \in \mathbb{C}^{256 \times 256}$ is the observed undersampled noisy k-space

**What makes this hard**: The problem is underdetermined (only 30% of k-space observed). The reconstruction uses a pretrained denoiser as an implicit image prior within an ADMM framework. The denoiser must be properly integrated — input normalization must match the training distribution, and the denoiser must have bounded Lipschitz constant to guarantee convergence.

**Input**: k-space undersampling mask $(256, 256)$, complex noise $(256, 256)$, ground truth image $(256, 256)$.

**Output**: Reconstructed image $\hat{x} \in \mathbb{R}^{256 \times 256}$.

## Data Description

**Source**: Brain.jpg (256x256 grayscale MRI brain image) from the Provable Plug-and-Play demo. Three k-space undersampling masks are provided (~30% sampling each). Complex Gaussian noise is pre-generated for reproducibility.

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `mask_random` | (1, 256, 256) | float32 | Random k-space undersampling mask (30.0% sampled) |
| `mask_radial` | (1, 256, 256) | float32 | Radial k-space undersampling mask (29.4% sampled) |
| `mask_cartesian` | (1, 256, 256) | float32 | Cartesian variable-density mask (29.7% sampled) |
| `noises_real` | (1, 256, 256) | float32 | Real part of complex noise (unscaled) |
| `noises_imag` | (1, 256, 256) | float32 | Imaginary part of complex noise (unscaled) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `image` | (1, 256, 256) | float32 | Ground truth brain MRI image, normalized to [0, 1] |

### data/RealSN_DnCNN_noise15.pth

Pretrained RealSN-DnCNN weights (17-layer residual denoiser, 64 features, trained at noise level sigma=15 with real spectral normalization). ~11 MB.

### data/meta_data.json

Contains imaging parameters: image shape, mask names, sampling rate, noise scaling factor.

## Method Hints

Plug-and-Play ADMM replaces the proximal operator of a traditional regularizer with a pretrained image denoiser, using it as an implicit prior within the ADMM splitting framework. The algorithm alternates between a data fidelity step (which has a closed-form solution in the Fourier domain for Cartesian MRI), a denoising step using the pretrained CNN, and a dual variable update. The denoiser is a residual DnCNN with real spectral normalization to ensure convergence of the PnP iterations.

## References

1. Ryu, E.K., Liu, J., Wang, S., Chen, X., Wang, Z. & Yin, W. (2019). Plug-and-Play Methods Provably Converge with Properly Trained Denoisers. ICML 2019.
2. Zhang, K., Zuo, W., Chen, Y., Meng, D. & Zhang, L. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising. IEEE TIP, 26(7).
3. Miyato, T., Kataoka, T., Koyama, M. & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. ICLR 2018.
4. GitHub: https://github.com/uclaopt/Provable_Plug_and_Play
