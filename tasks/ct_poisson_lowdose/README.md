# Low-Dose CT Reconstruction with Poisson Noise

> Reconstruct a 2D attenuation image from Poisson-noise sinogram measurements acquired at low photon counts, where signal-dependent noise variance requires weighted least-squares reconstruction to avoid noise amplification in low-count regions.

> Domain: medicine | Keywords: computed tomography, statistical reconstruction, Poisson noise | Difficulty: Medium

## Background

In X-ray computed tomography (CT), photons passing through the object are attenuated according to Beer's law and counted by detectors. At standard clinical dose, photon counts are high enough that noise is approximately Gaussian and uniform across detector bins. However, at reduced dose (low photon count), the discrete photon-counting process follows Poisson statistics, making the noise variance signal-dependent: rays passing through highly attenuating regions receive fewer photons and therefore have higher relative noise. Ignoring this non-uniform noise structure leads to streak artifacts and noise amplification in reconstruction.

Model-based iterative reconstruction (MBIR) with penalized weighted least squares (PWLS) addresses this by incorporating the Poisson noise model directly into the reconstruction objective. Each sinogram measurement is weighted inversely proportional to its noise variance, so that unreliable low-count measurements contribute less to the reconstruction while high-count measurements are trusted more.

## Problem Description

The CT forward model is the Radon transform with Poisson noise on the photon counts:

$$I_i \sim \text{Poisson}\!\left(I_0 \exp\!\left(-[\mathbf{A}x]_i\right)\right)$$

where $x \in \mathbb{R}^N$ is the 2D attenuation image, $\mathbf{A} \in \mathbb{R}^{M \times N}$ is the discrete Radon system matrix (parallel-beam geometry), $I_0$ is the incident photon count per ray, and $I_i$ is the detected photon count for ray $i$.

After taking the negative logarithm (post-log transform), the measurements become:

$$y_i = -\log(I_i / I_0)$$

In the post-log domain, the noise is approximately Gaussian with signal-dependent variance $\text{Var}(y_i) \approx 1/I_i$. The penalized weighted least-squares (PWLS) reconstruction solves:

$$\hat{x} = \arg\min_x \;\frac{1}{2}(\mathbf{y} - \mathbf{A}x)^T \mathbf{W} (\mathbf{y} - \mathbf{A}x) \;+\; \beta\, R(x)$$

where $\mathbf{W} = \text{diag}(I_1, \ldots, I_M)$ is the diagonal weight matrix derived from the Poisson statistics, and $R(x)$ is a regularization prior (here, a q-Generalized Gaussian Markov Random Field).

**Ill-posedness:** At low dose ($I_0 = 1000$ photons/ray), rays through highly attenuating paths may receive only a handful of photons, making those measurements extremely noisy. Without proper weighting, these noisy measurements corrupt the entire reconstruction.

**Input:** Noisy post-log sinogram $\mathbf{y} \in \mathbb{R}^{256 \times 367}$, Poisson-derived weights $\mathbf{W} \in \mathbb{R}^{256 \times 367}$, and projection angles.

**Output:** Reconstructed attenuation image $\hat{x} \in \mathbb{R}^{256 \times 256}$.

## Data Description

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `phantom` | `(1, 256, 256)` | float64 | Shepp-Logan phantom scaled to realistic linear attenuation coefficients (cm$^{-1}$), values in [0, 0.04] |

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sinogram_clean` | `(1, 256, 367)` | float64 | Noiseless sinogram (line integrals), units: dimensionless (integrated attenuation) |
| `sinogram_low_dose` | `(1, 256, 367)` | float64 | Post-log sinogram with Poisson noise at $I_0 = 1000$ photons/ray |
| `sinogram_high_dose` | `(1, 256, 367)` | float64 | Post-log sinogram with Poisson noise at $I_0 = 50000$ photons/ray |
| `weights_low_dose` | `(1, 256, 367)` | float64 | PWLS weights for low-dose data ($w_i = I_i$, photon counts) |
| `weights_high_dose` | `(1, 256, 367)` | float64 | PWLS weights for high-dose data |
| `angles` | `(1, 256)` | float64 | Projection angles in radians, uniformly spaced in $[0, \pi)$ |

### data/meta_data.json

| Key | Value | Description |
|-----|-------|-------------|
| `image_size` | 256 | Phantom image dimensions (pixels) |
| `num_views` | 256 | Number of projection angles |
| `num_channels` | 367 | Number of detector channels per view |
| `angle_range_rad` | 3.14159 | Angular range in radians ($[0, \pi)$) |
| `geometry` | "parallel" | Beam geometry |
| `delta_channel` | 1.0 | Detector channel spacing in ALU |
| `delta_pixel` | 1.0 | Image pixel spacing in ALU |
| `attenuation_scale` | 0.02 | Scale factor applied to Shepp-Logan phantom |
| `I0_low_dose` | 1000 | Incident photon count for low-dose acquisition |
| `I0_high_dose` | 50000 | Incident photon count for high-dose acquisition |
| `random_seed` | 42 | Random seed for noise generation |

## Method Hints

- **Unweighted reconstruction** (baseline): treats all sinogram measurements as equally reliable by setting $\mathbf{W} = \mathbf{I}$. This ignores the Poisson noise structure and produces noisier reconstructions, especially in regions behind highly attenuating paths.
- **PWLS reconstruction** (main method): uses $w_i = I_i$ (the measured photon count) as the weight for each sinogram element. This is the statistically optimal weighting for the post-log Poisson noise model and down-weights unreliable low-count measurements.
- The SVMBIR library implements the multi-resolution Super-Voxel MBIR algorithm with a q-GGMRF prior, which encourages piecewise smoothness while preserving edges. The key API call is `svmbir.recon()` with the `weights` parameter set to the Poisson-derived weight array.
- The q-GGMRF prior has the form $R(x) = \sum_{(i,j)} b_{ij} \rho(x_i - x_j)$ where $\rho(t) = \frac{|t/\sigma_x|^p}{p(1 + |t/(T\sigma_x)|^{q-p})}$ with shape parameters $p = 1.2$ and $q = 2.0$, producing behavior that is $\ell^{1.2}$-like for large differences (edge-preserving) and $\ell^2$-like for small differences (smooth in flat regions).

## References

- Bouman, C.A. and Sauer, K. (1993). "A Generalized Gaussian Image Model for Edge-Preserving MAP Estimation." *IEEE Transactions on Image Processing*, 2(3), 296-310.
- Thibault, J.-B., Sauer, K.D., Bouman, C.A., and Hsieh, J. (2007). "A three-dimensional statistical approach to improved image quality for multislice helical CT." *Medical Physics*, 34(11), 4526-4544.
- Venkatakrishnan, S.V., Bouman, C.A., and Wohlberg, B. (2013). "Model-Based Iterative Reconstruction for Bright-Field Electron Tomography." *IEEE Transactions on Computational Imaging*, 1(1), 1-15.
- SVMBIR software package: https://github.com/cabouman/svmbir
