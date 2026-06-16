# Diffusion MRI: Diffusion Tensor Imaging (DTI)

> Estimate per-voxel 3x3 symmetric diffusion tensors from multi-direction diffusion-weighted MRI signals using the Stejskal-Tanner equation, a tensor parameter estimation inverse problem fundamentally different from scalar or image-domain reconstruction.

> Domain: medicine | Keywords: quantitative MRI, parameter estimation, diffusion imaging | Difficulty: Medium

## Background

Diffusion MRI exploits the random Brownian motion of water molecules to probe tissue microstructure. By applying magnetic field gradients in different spatial directions, the MRI signal becomes sensitized to water diffusion along those directions. In Diffusion Tensor Imaging (DTI), the diffusion process at each voxel is modeled as a 3x3 symmetric positive-definite tensor, capturing both the magnitude and directional preference of water diffusion.

In biological tissue, water diffusion is anisotropic — it diffuses more freely along nerve fiber bundles than perpendicular to them. This directional information, encoded in the diffusion tensor, enables non-invasive mapping of white matter architecture, detection of stroke and demyelination, and presurgical planning for neurosurgery.

## Problem Description

The forward model for diffusion-weighted MRI at a single voxel is the Stejskal-Tanner equation:

$$S(\mathbf{g}_i, b_i) = S_0 \cdot \exp\!\left(-b_i \cdot \mathbf{g}_i^\top \mathbf{D} \, \mathbf{g}_i\right)$$

where:
- $S(\mathbf{g}_i, b_i)$ is the measured signal with gradient direction $\mathbf{g}_i$ and b-value $b_i$
- $S_0$ is the non-diffusion-weighted signal (b = 0)
- $b_i$ is the diffusion weighting factor (s/mm²)
- $\mathbf{g}_i = (g_x, g_y, g_z)^\top$ is the unit gradient direction vector
- $\mathbf{D}$ is the 3x3 symmetric positive-definite diffusion tensor (mm²/s)

The diffusion tensor has 6 independent elements:

$$\mathbf{D} = \begin{pmatrix} D_{xx} & D_{xy} & D_{xz} \\ D_{xy} & D_{yy} & D_{yz} \\ D_{xz} & D_{yz} & D_{zz} \end{pmatrix}$$

**Inverse problem:** Given noisy DWI measurements $\{S(\mathbf{g}_i, b_i)\}_{i=1}^{N}$ at each voxel, estimate the 6 tensor elements and the $S_0$ signal (7 unknowns per voxel).

**Ill-conditioning:** The problem becomes ill-conditioned when:
- SNR is low (Rician noise bias distorts log-signal, especially at high b-values)
- Few gradient directions are available (minimum 6 required, more is better)
- The b-value is too low (weak diffusion weighting) or too high (signal in noise floor)
- Tensor eigenvalues span a wide range (high anisotropy voxels)

The input is a 3D array of DWI signals (Ny x Nx x N_volumes) with a gradient table (b-values and directions), and the output is a 2D FA map (Ny x Nx) and MD map (Ny x Nx) derived from the fitted tensor via eigendecomposition.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `dwi_signal` | `(1, 128, 128, 31)` | float32 | Diffusion-weighted MRI signal (a.u.) at 1 b=0 + 30 gradient directions |
| `bvals` | `(31,)` | float32 | b-values in s/mm² (0 for b=0 volumes, 1000 for DWI) |
| `bvecs` | `(31, 3)` | float32 | Gradient direction unit vectors (zero vector for b=0) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `fa_map` | `(1, 128, 128)` | float32 | Ground truth fractional anisotropy map (dimensionless, 0–1) |
| `md_map` | `(1, 128, 128)` | float32 | Ground truth mean diffusivity map (mm²/s) |
| `tensor_elements` | `(1, 128, 128, 6)` | float32 | Ground truth tensor elements [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] (mm²/s) |
| `tissue_mask` | `(1, 128, 128)` | bool | Binary tissue mask (True where tissue is present) |

### data/meta_data.json

| Key | Type | Description |
|-----|------|-------------|
| `image_size` | int | Image dimension (128) |
| `n_directions` | int | Number of DWI gradient directions (30) |
| `n_b0` | int | Number of b=0 volumes (1) |
| `b_value_s_per_mm2` | float | b-value for DWI volumes (1000.0 s/mm²) |
| `noise_sigma` | float | Rician noise standard deviation (0.02) |
| `signal_model` | str | Signal model type ("stejskal_tanner") |
| `modality` | str | Imaging modality |
| `field_of_view_mm` | float | Field of view in mm (220.0) |
| `pixel_size_mm` | float | Pixel size in mm |

The synthetic data is generated from a modified Shepp-Logan phantom with diffusion tensor properties assigned per tissue region: bone/scalp (low MD, low FA), white matter (moderate MD, high FA with directional fibers), gray matter (moderate MD, low FA), deep gray matter (intermediate FA), and CSF (high MD, isotropic).

## Method Hints

From the fitted tensor, clinically relevant scalar maps are derived via eigendecomposition $\mathbf{D} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top$:
- **Fractional Anisotropy (FA):** $\text{FA} = \sqrt{\frac{3}{2}} \cdot \frac{\sqrt{\sum_i (\lambda_i - \bar{\lambda})^2}}{\sqrt{\sum_i \lambda_i^2}} \in [0, 1]$
- **Mean Diffusivity (MD):** $\text{MD} = (\lambda_1 + \lambda_2 + \lambda_3)/3$

Two linearization-based approaches for tensor fitting:

1. **Ordinary Least Squares (OLS)** (fast baseline): Taking the logarithm of the Stejskal-Tanner equation linearizes the model: $\ln S_i = \ln S_0 - b_i \mathbf{g}_i^\top \mathbf{D} \mathbf{g}_i$. Expanding the quadratic form gives a linear system with 7 unknowns ($\ln S_0$ plus 6 tensor elements) per voxel. Solve via standard least squares. Fast but biased because Rician noise is not Gaussian in the log domain.

2. **Weighted Least Squares (WLS)** (improved): Two-step procedure — first obtain OLS estimates, then re-solve with weights $w_i = S_i^2$ (the predicted squared signal), which accounts for the heteroscedasticity introduced by the log transform. More accurate than OLS, especially at moderate-to-low SNR.

Both methods require at least 6 non-collinear gradient directions plus one b=0 measurement. More directions improve conditioning and SNR. A tissue mask should exclude background voxels. After fitting, eigendecompose each tensor to obtain FA and MD maps.

## References

- Basser, P.J., Mattiello, J., LeBihan, D. (1994). MR diffusion tensor spectroscopy and imaging. Biophysical Journal, 66(1), 259-267.
- Salvador, R., et al. (2005). Formal characterization and extension of the linearized diffusion tensor model. Human Brain Mapping, 24(2), 144-155.
- DIPY: Diffusion Imaging in Python — https://dipy.org/ (reference implementation for DTI fitting).
