# Approach

## Problem Statement

Recover a high-resolution 3D complex field (amplitude and phase) at 2048x2048 pixels across 161 z-slices from 68 low-resolution intensity-only FPM measurements captured under varying LED illumination angles, using an implicit neural representation as the reconstruction parameterization.

## Mathematical Formulation

### Forward Model

The FPM forward model maps a complex field at depth $z$ to a measured intensity for LED $\ell$. The measurement chain is:

$$I_\ell(x, y; z) = \left| \mathcal{F}^{-1}\left\{ \tilde{O}_{\text{sub},\ell} \cdot H(k_x, k_y; z) \right\} \right|^2$$

where:

- $\tilde{O}(k_x, k_y) = \mathcal{F}\{O(x,y,z)\}$ is the 2D Fourier transform of the complex field at depth $z$
- $\tilde{O}_{\text{sub},\ell}$ is the sub-band of $\tilde{O}$ centered at the LED's spatial frequency offset $(k_0 u_\ell, k_0 v_\ell)$, windowed by the pupil $P(k_x, k_y)$
- $H(k_x, k_y; z) = P(k_x, k_y) \cdot \exp(i k_z z)$ is the angular spectrum propagation kernel
- $k_z = \sqrt{k_0^2 - k_x^2 - k_y^2}$, $k_0 = 2\pi / \lambda$, $\lambda = 0.5126\;\mu\text{m}$
- $P(k_x, k_y) = \mathbb{1}[\sqrt{k_x^2 + k_y^2} \le \text{NA} \cdot k_0]$ is the binary pupil with NA = 0.256

The training objective compares predicted and measured amplitudes (square root of intensity) rather than intensities directly.

### Implicit Neural Representation

The 3D complex field is decomposed into amplitude and phase:

$$O(x, y, z) = A(x, y, z) \cdot \exp\!\bigl(i\,\phi(x, y, z)\bigr)$$

Each component ($A$ and $\phi$) is parameterized by an independent factored 3D implicit representation:

- **Spatial features**: a learnable 2D grid $\mathbf{F}_{xy} \in \mathbb{R}^{512 \times 512 \times 32}$, sampled at arbitrary $(x, y)$ via bilinear interpolation
- **Depth features**: a learnable 1D array $\mathbf{F}_z \in \mathbb{R}^{5 \times 32}$ with 5 depth modes, sampled at arbitrary $z$ via linear interpolation (z normalized to [0, 4] from the physical range [-20, +20] um)
- **Feature fusion**: element-wise product $\mathbf{f}(x, y, z) = \mathbf{F}_{xy}(x, y) \odot \mathbf{F}_z(z)$
- **Rendering MLP**: a 2-layer ReLU network (32 -> 32 -> 1) maps fused features to a scalar output

This factored design decouples spatial and depth variation, making 3D reconstruction tractable with far fewer parameters than a full voxel grid.

### Loss Function

Smooth L1 loss between predicted and measured amplitudes:

$$\mathcal{L} = \text{SmoothL1}\!\left(\sqrt{I_\ell^{\text{meas}}},\; \left|\mathcal{F}^{-1}\left\{\tilde{O}_{\text{sub},\ell} \cdot H_z\right\}\right|\right)$$

summed over all LEDs and sampled z-planes per iteration.

## Solution Strategy

### Step 1: Data Preprocessing

Load the 68 raw intensity images from `raw_data.npz` and imaging parameters from `meta_data.json`. Extract a 1024x1024 ROI (2 x num_modes). Compute LED positions in frequency space by finding the nearest grid point to each LED's calibrated NA offset. Sort LEDs by distance from the optical axis and normalize measurements by the global maximum intensity.

Compute the pupil support mask $P(k_x, k_y)$ and the angular spectrum propagation kernel $k_z(k_x, k_y)$ from the spatial frequency grid. Determine the z-sampling parameters: depth of field DOF = $0.5 / \text{NA}^2 \approx 7.6\;\mu\text{m}$, z-step = $0.8 \cdot \text{DOF}$, yielding ~14 training z-planes.

### Step 2: Forward Model Construction

Build the FPM forward model that, given a complex field and LED index:
1. Computes the defocus transfer function $H_z = P \cdot \exp(ik_z z)$ for each z-depth
2. Extracts the sub-spectrum at each LED's frequency offset by cropping a 1024x1024 window from the zero-padded 2048x2048 FFT of the complex field
3. Applies the transfer function and inverse FFTs to produce predicted amplitudes

The model operates on batches of z-planes and LEDs, with one LED per gradient step (led_batch_size = 1).

### Step 3: INR Training

Initialize a FullModel containing two G_Tensor3D networks (amplitude and phase), each with 512x512 spatial modes, 5 depth modes, and 32 feature channels. Train for 15 epochs with:

- **Optimizer**: Adam, learning rate $10^{-3}$
- **Scheduler**: StepLR, decay by factor $0.1$ every 6 epochs
- **Z-sampling strategy**: even epochs use a uniform grid of 14 z-planes; odd epochs randomly sample 7 z-planes from the same range. This alternation prevents overfitting to a fixed depth discretization.
- **Mixed precision**: bfloat16 via `torch.cuda.amp.autocast` for memory efficiency
- **Compilation**: `torch.compile` with inductor backend for throughput

Each training iteration: for each sampled z-plane, iterate over all 68 LEDs one at a time, compute the forward model prediction, evaluate the smooth L1 loss against the measured amplitude, and backpropagate.

### Step 4: Per-Slice Evaluation

After training, run inference at the 161 ground truth z-positions (linearly spaced from -20 to +20 um). The model outputs 2048x2048 amplitude images; bilinearly downsample these to 1024x1024 to match the ground truth resolution. Normalize both predicted and GT stacks to [0, 1] globally, then compute per-slice MSE, PSNR, and SSIM.

### Step 5: All-in-Focus Reconstruction

Compute all-in-focus images from both the predicted and ground truth z-stacks using the Normal Variance focus measure:

1. Divide the image into overlapping 64x64 patches with stride 16
2. For each patch, compute the normalized variance $\text{NV}_k = \sum(I - \mu)^2 / (N \cdot \mu)$ at each z-plane $k$
3. Select the z-plane with maximum NV (sharpest focus) for each patch
4. Fuse overlapping patches using an inverse-overlap balance map

Evaluate using mean-subtracted L2 error: subtract each image's mean before computing MSE, so the metric is insensitive to global brightness offset. This matches the paper's primary quantitative metric (Fig. 2).

### Step 6: Save Results

Save the trained model weights, per-slice metrics (L2, PSNR, SSIM arrays), all-in-focus images (predicted and GT), and a `metrics.json` summary containing: AIF L2 error, AIF PSNR, per-slice PSNR mean/std, per-slice SSIM mean/std, overall MSE, overall PSNR.

## Expected Results

| Metric | Expected Value | Paper Reference |
|--------|---------------|-----------------|
| All-in-focus L2 error (MSE) | ~1.33 x 10^-3 | 1.41 x 10^-3 |
| All-in-focus PSNR | ~28.8 dB | -- |
| Per-slice PSNR (mean +/- std) | 22.8 +/- 0.5 dB | -- |
| Per-slice SSIM (mean +/- std) | 0.775 +/- 0.034 | -- |
| NCC vs ground truth | 0.997 | -- |
| NRMSE vs ground truth | 0.071 | -- |
| Training time (A40 GPU) | ~7 minutes | -- |

The reconstruction slightly outperforms the paper's reported L2 error (1.33e-3 vs 1.41e-3) likely due to differences in random initialization and z-sampling. The key scientific result is that an implicit neural representation can reconstruct a full 3D complex field from 2D intensity measurements without explicit phase retrieval, achieving continuous depth representation with only 5 learnable depth modes.
