# Shapelet Source Reconstruction

> Reconstruct the intrinsic (unlensed) source galaxy from a noisy, PSF-convolved, gravitationally lensed observation using a linear shapelet basis decomposition solved via weighted least squares.

> Domain: astronomy | Keywords: gravitational lensing, shapelet decomposition, image deconvolution | Difficulty: Medium

## Background

Gravitational lensing distorts the images of distant galaxies as their light passes near massive foreground objects. Recovering the intrinsic source morphology from a lensed, blurred, noisy observation is a fundamental inverse problem in observational astronomy. Shapelets — orthonormal 2D basis functions built from Hermite polynomials — provide a compact, analytic representation of galaxy light profiles that linearizes the reconstruction problem when the lens model is known.

## Problem Description

The measurement equation is:

$$d = A \, c + n$$

where $d \in \mathbb{R}^{N_\text{pix}}$ is the observed (lensed, convolved, pixelized) image flattened to a vector, $c \in \mathbb{R}^{N_\text{coeff}}$ are the shapelet coefficients of the source, $A \in \mathbb{R}^{N_\text{pix} \times N_\text{coeff}}$ is the response matrix encoding the full forward model (ray-tracing + PSF convolution + downsampling), and $n$ is noise (Poisson + Gaussian background).

The forward model per pixel proceeds as:

1. **Backward ray-tracing**: Map image-plane coordinate $\vec{\theta}$ to source-plane coordinate $\vec{\beta}$ via the lens equation $\vec{\beta} = \vec{\theta} - \vec{\alpha}(\vec{\theta})$, where $\vec{\alpha}$ is the deflection angle from the lens model.
2. **Source evaluation**: Evaluate the shapelet model $I(\vec{\beta}) = \sum_k c_k B_k(\vec{\beta}; \beta_s)$ at the ray-traced source position.
3. **PSF convolution**: Convolve the high-resolution image with a Gaussian PSF kernel.
4. **Pixelization**: Downsample the supersampled grid to detector resolution by block-averaging.

The lens model is a Softened Power-law Elliptical Potential (SPEP) with deflection angles:

$$\alpha_x = \frac{\theta_E}{1 + q} \left(\frac{\theta_E}{\sqrt{q^2 x'^2 + y'^2}}\right)^{\gamma - 2} x', \quad \alpha_y = \frac{\theta_E}{1 + q} \left(\frac{\theta_E}{\sqrt{q^2 x'^2 + y'^2}}\right)^{\gamma - 2} \frac{y'}{q}$$

where $\theta_E$ is the Einstein radius, $\gamma$ is the power-law slope, $q$ is the axis ratio, and $(x', y')$ are coordinates rotated by the position angle $\phi$. For $\gamma = 2$ and $q = 1$, this reduces to the Singular Isothermal Sphere (SIS): $\vec{\alpha} = \theta_E \, \hat{r}$.

External shear adds: $\alpha_x^{\text{shear}} = \gamma_1 x + \gamma_2 y$, $\alpha_y^{\text{shear}} = \gamma_2 x - \gamma_1 y$.

The shapelet basis functions are:

$$B_{n_1, n_2}(\vec{x}; \beta_s) = \frac{1}{\beta_s} \phi_{n_1}\!\left(\frac{x}{\beta_s}\right) \phi_{n_2}\!\left(\frac{y}{\beta_s}\right)$$

where $\phi_n(u) = [2^n \sqrt{\pi}\, n!]^{-1/2} H_n(u)\, e^{-u^2/2}$ and $H_n$ is the physicist's Hermite polynomial. Coefficients are indexed with constraint $n_1 + n_2 \leq n_{\max}$, giving $N_\text{coeff} = (n_{\max}+1)(n_{\max}+2)/2$ terms.

The problem is ill-posed because the PSF convolution suppresses high-frequency information and noise contaminates the observation. The finite shapelet order $n_{\max}$ acts as implicit regularization.

## Data Description

### data/raw_data.npz

Preprocessed NGC 1300 galaxy data used as source input for the simulation pipeline.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `coeff_ngc` | `(1, 11476)` | `float32` | Shapelet coefficients from decomposing NGC 1300 at $n_{\max}=150$, $\beta_s=10$ pixels |
| `ngc_resized` | `(1, 264, 264)` | `float32` | Downsampled (25x) and Gaussian-smoothed galaxy image in counts |
| `image_reconstructed` | `(1, 264, 264)` | `float32` | Galaxy image reconstructed from shapelet coefficients in counts |

### data/ground_truth.npz

Ground truth source image and lens model parameters (known from simulation).

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `source_image` | `(1, 320, 320)` | `float32` | Intrinsic (unlensed) high-resolution source image at 0.01 arcsec/pixel in counts |
| `theta_E` | `(1,)` | `float64` | Einstein radius of the SIS lens model in arcsec |
| `gamma` | `(1,)` | `float64` | Power-law slope of the SPEP lens model (2.0 = SIS) |
| `source_x` | `(1,)` | `float64` | Source center x-offset in arcsec |
| `source_y` | `(1,)` | `float64` | Source center y-offset in arcsec |

### data/meta_data.json

Imaging and instrument parameters for the forward model. Contains: detector geometry (`numPix=64`, `deltaPix=0.05` arcsec/pixel), PSF specification (`fwhm_lensing=0.1` arcsec, `fwhm_deconv=0.25` arcsec), noise parameters (`background_rms=10` counts, `exp_time=100` s), supersampling factor (`high_res_factor=5`), and lens parameters ($\theta_E=0.5$ arcsec, $\gamma=2.0$, zero ellipticity and shear).

## Method Hints

The reconstruction uses **weighted least squares (WLS)** inversion of the linear forward model. The key algorithmic steps are:

- **Supersampling**: Evaluate the forward model on a 5x finer grid (320x320) to capture sub-pixel structure, then block-average to detector resolution (64x64).
- **Response matrix construction**: Each column of $A$ is obtained by evaluating a single shapelet basis function through the complete forward pipeline (ray-trace, PSF convolve, downsample). With $n_{\max}=20$, the matrix has 231 columns.
- **Noise-weighted inversion**: Solve $\hat{c} = (A^T W A)^{-1} A^T W d$ where the diagonal weight matrix $W_{ii} = 1/\sigma_i^2$ accounts for heteroscedastic noise: $\sigma_i^2 = \sigma_{\text{bkg}}^2 + |d_i| / t_{\text{exp}}$ (background + Poisson).
- **Reconstruction uses fewer basis functions** ($n_{\max}=20$) than the original source decomposition ($n_{\max}=150$), which regularizes against noise amplification.

The task includes two sub-problems: (1) source reconstruction through a known gravitational lens, and (2) image deconvolution without lensing (identity lens mapping). Both use the same WLS framework but with different PSF widths and shapelet scale parameters.

## References

- Refregier, A. (2003). "Shapelets: I. A Method for Image Analysis." *Monthly Notices of the Royal Astronomical Society*, 338(1), 35–47.
- Birrer, S. et al. (2021). "lenstronomy II: A gravitational lensing software ecosystem." *Journal of Open Source Software*, 6(62), 3283.
- Barkana, R. (1998). "Fast calculation of a family of elliptical mass gravitational lens models." *The Astrophysical Journal*, 502(2), 531.
