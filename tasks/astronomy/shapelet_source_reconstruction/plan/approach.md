# Approach

## Problem Statement

Recover the intrinsic (unlensed) morphology of a galaxy from a 64x64 noisy, PSF-convolved, gravitationally lensed observation, using a Cartesian Hermite shapelet basis decomposition with weighted least squares inversion. The lens model is assumed known.

## Mathematical Formulation

### Forward Model

The imaging forward model maps source-plane shapelet coefficients to a detector-plane observation:

$$d = A \, c + n$$

- **c** in R^{K}: shapelet coefficients of the source, K = (n_max+1)(n_max+2)/2
- **A** in R^{N^2 x K}: response matrix encoding ray-tracing + PSF convolution + downsampling
- **d** in R^{N^2}: vectorized observed image (N=64, so 4096 pixels)
- **n** in R^{N^2}: noise vector (Poisson + Gaussian background)

Each column of A is constructed by evaluating one shapelet basis function through the full forward pipeline:

1. Create a supersampled grid (N_hr = N * s, s=5) at pixel scale deltaPix/s
2. Backward ray-trace each grid point through the lens: beta = theta - alpha(theta)
3. Evaluate basis function B_k(beta; beta_s) at the ray-traced source coordinates
4. Convolve the N_hr x N_hr image with a Gaussian PSF kernel (FWHM = 0.1")
5. Block-average s x s subpixels to produce the detector-resolution N x N column

### Lens Model

The deflection field alpha(theta) combines two components:

**SPEP (Softened Power-law Elliptical Potential):**

For the spherical case (gamma=2, e1=e2=0), reduces to the Singular Isothermal Sphere:

$$\vec{\alpha}_{\text{SIS}} = \theta_E \, \frac{\vec{\theta}}{|\vec{\theta}|}$$

with Einstein radius theta_E = 0.5". The general SPEP deflection for elliptical cases uses the axis ratio q derived from (e1, e2) and applies a rotation to align with the ellipse axes before computing the power-law deflection.

**External shear:** alpha_x = gamma1*x + gamma2*y, alpha_y = gamma2*x - gamma1*y (set to zero in this task).

### Shapelet Basis

The 2D Cartesian Hermite shapelets are separable products:

$$B_{n_1,n_2}(x, y; \beta_s) = \frac{1}{\beta_s} \phi_{n_1}\!\left(\frac{x}{\beta_s}\right) \phi_{n_2}\!\left(\frac{y}{\beta_s}\right)$$

where phi_n(u) = [2^n sqrt(pi) n!]^{-1/2} H_n(u) exp(-u^2/2) and H_n is the physicist's Hermite polynomial. The ordering convention is (n1, n2) with constraint n1 + n2 <= n_max, iterated as: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...

### Noise Model

The noise covariance is diagonal with pixel-dependent variance:

$$\sigma_i^2 = \sigma_{\text{bkg}}^2 + \frac{|d_i|}{t_{\text{exp}}}$$

combining Gaussian background (sigma_bkg = 10 counts) and Poisson photon noise (exposure time t_exp = 100 s).

## Solution Strategy

### Step 1: Data Preprocessing

Load the NGC 1300 galaxy JPEG image (6600x4400 pixels). Convert to grayscale float, subtract the median background level, and pad to a square (6637x6637). Apply Gaussian smoothing (sigma=5 pixels) to suppress JPEG artifacts. Downsample by a factor of 25 via block-averaging to obtain a 264x264 image suitable for shapelet decomposition.

Decompose the preprocessed image into K=11,476 shapelet coefficients at n_max=150 and beta_s=10 pixels using orthogonal projection:

$$c_k = \frac{(\Delta x)^2}{\beta_s^2} \sum_i I(x_i, y_i) \, B_k(x_i, y_i; \beta_s)$$

The normalization factor (deltaPix/beta_s)^2 accounts for the discrete sum approximating a continuous integral. This high-order decomposition captures the full morphological detail of the barred spiral galaxy.

### Step 2: Forward Simulation (Lensed Observation)

Generate a synthetic lensed observation through the full forward pipeline:

1. **Source evaluation**: Evaluate the n_max=150 shapelet model on a 320x320 supersampled grid (5x oversampling of 64x64 detector, pixel scale 0.01"/pixel) to produce the intrinsic source image.
2. **Ray-tracing**: Backward ray-trace every supersampled pixel through the SIS lens (theta_E=0.5", source offset 0.2" in x). The ray-traced coordinates define where each pixel samples the source.
3. **Lensed image**: Evaluate the shapelet source at the ray-traced positions to produce the lensed high-resolution image.
4. **PSF convolution**: Convolve with a Gaussian PSF (FWHM=0.1", sigma=2.12 supersampled pixels) using scipy.ndimage.gaussian_filter, with truncation at 4 sigma.
5. **Downsampling**: Block-average 5x5 subpixel blocks to produce the 64x64 detector image.
6. **Noise injection**: Add Poisson noise (draw from Poisson(|I_i| * t_exp) / t_exp) then additive Gaussian noise (sigma_bkg=10 counts).

### Step 3: Source Reconstruction (Lensed Case)

Recover the source from the noisy lensed observation:

1. **Build response matrix A** (4096 x 231): For each of the K=231 reconstruction basis functions (n_max=20, beta_s=0.15"), evaluate the basis through the full forward pipeline (ray-trace, convolve, downsample). Each basis function produces one column of A.

2. **Compute noise weights**: Diagonal weight matrix W with W_ii = 1/sigma_i^2 where sigma_i^2 = sigma_bkg^2 + |d_i|/t_exp.

3. **Solve WLS normal equations**: c_hat = (A^T W A)^{-1} A^T W d. Check the condition number of A^T W A; if > 10^14, fall back to np.linalg.lstsq for numerical stability.

4. **Reconstruct**: Compute the model image d_hat = A c_hat and evaluate the source on the 320x320 high-resolution grid using the recovered coefficients with the reconstruction shapelet parameters (n_max=20, beta_s=0.06").

5. **Assess fit quality**: Compute reduced chi-squared from normalised residuals r_i = (d_hat_i - d_i) / sigma_i. A value chi2_red ~ 1.0 indicates the model fits the data to within the expected noise level.

### Step 4: Deconvolution (No Lensing)

Repeat the same pipeline as Steps 2-3 but with identity lens mapping (no ray-tracing deflection):

- Blurrier PSF: FWHM = 0.25" (simulating worse seeing conditions)
- Larger shapelet scale: beta_s = 0.3" (reconstruction), beta_model = 0.12" (source evaluation)
- Scaled source amplitudes: coefficients multiplied by 5.0 to maintain signal-to-noise

This sub-problem isolates PSF deconvolution from the lensing inversion, demonstrating the shapelet framework's versatility. The reconstruction quality is lower (NCC ~ 0.74 vs 0.96 for lensing) because the wider PSF destroys more spatial information.

### Step 5: Visualization

Generate five diagnostic figures:

1. **Shapelet decomposition** (4-panel): Original galaxy, smoothed, resized, and shapelet-reconstructed images — validates that the decomposition preserves source morphology.
2. **Lensing stages** (5-panel): Intrinsic source → lensed → PSF-convolved → pixelized → noisy — shows progressive degradation through the forward model.
3. **Lensed reconstruction** (2x3): Input data, model, normalised residuals (top row); true source, reconstructed source, source residuals (bottom row).
4. **Deconvolution stages** (4-panel): Intrinsic → convolved → pixelized → noisy — forward model without lensing.
5. **Deconvolution reconstruction** (2x3): Same layout as figure 3 but for the unlensed case.

## Expected Results

| Sub-problem | NCC vs ground truth | NRMSE vs ground truth | Reduced chi-squared |
|-------------|--------------------:|----------------------:|--------------------:|
| Lensing reconstruction | 0.955 | 0.054 | 0.991 |
| Deconvolution (no lens) | 0.743 | 0.209 | 0.996 |

The lensing reconstruction achieves high fidelity (NCC > 0.95) because the SIS lens magnifies the source, effectively increasing spatial sampling. The deconvolution case performs worse because the wider PSF (0.25" vs 0.1") erases more high-frequency information, and without lensing magnification the effective resolution is lower. Both cases achieve chi-squared ~ 1.0, confirming that the WLS inversion correctly fits the data to within the noise level.
