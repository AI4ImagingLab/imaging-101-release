# Sparse-View CT Reconstruction

> Reconstruct a 2D phantom image from a severely undersampled set of 30 Radon projections (out of 180), where filtered back projection produces strong streak artifacts and total variation regularization is needed to exploit the piecewise-constant structure of the object.

> Domain: medicine | Keywords: computed tomography, sparse-view imaging, regularized reconstruction | Difficulty: Medium

## Background

X-ray computed tomography (CT) acquires projections of an object at many angles and reconstructs a cross-sectional image. In clinical practice, reducing the number of projection angles (sparse-view CT) lowers radiation dose but makes the reconstruction problem severely ill-posed. Standard filtered back projection (FBP) fails under sparse angular sampling, producing streak artifacts. Iterative methods with appropriate priors — particularly total variation (TV) — can recover high-quality images from far fewer measurements.

## Problem Description

The CT forward model is the **Radon transform**, which computes line integrals of the image attenuation along rays at angle $\theta$ and detector offset $s$:

$$g(\theta, s) = \int_{-\infty}^{\infty} f(s\cos\theta - t\sin\theta,\; s\sin\theta + t\cos\theta)\, dt$$

where $f(x, y)$ is the 2D attenuation image and $g(\theta, s)$ is the sinogram.

In matrix form, $\mathbf{g} = \mathbf{A}\mathbf{f}$, where $\mathbf{A}$ is the discrete Radon transform matrix.

**Inverse problem:** Given noisy sparse measurements $\mathbf{g}_\text{sparse} = \mathbf{A}_\text{sparse}\mathbf{f} + \mathbf{n}$ with only 30 angles (out of 180), recover $\mathbf{f}$.

**Ill-posedness:** With 30 angles, the system is heavily underdetermined. The null space of $\mathbf{A}_\text{sparse}$ contains high-frequency components aligned between projection angles, manifesting as streak artifacts in FBP.

**Input:** Noisy sparse sinogram $\mathbf{g}_\text{sparse} \in \mathbb{R}^{256 \times 30}$ and projection angles.

**Output:** Reconstructed image $\hat{\mathbf{f}} \in \mathbb{R}^{256 \times 256}$.

## Data Description

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `phantom` | `(1, 256, 256)` | float64 | Shepp-Logan phantom, pixel values in [0, 1] (dimensionless attenuation) |

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sinogram_sparse` | `(1, 256, 30)` | float64 | Noisy sparse sinogram (30 angles), units: integrated attenuation |
| `sinogram_full` | `(1, 256, 180)` | float64 | Clean full sinogram (180 angles), units: integrated attenuation |
| `angles_sparse` | `(1, 30)` | float64 | Sparse projection angles in degrees |
| `angles_full` | `(1, 180)` | float64 | Full projection angles in degrees |

### data/meta_data.json

| Key | Value | Description |
|-----|-------|-------------|
| `image_size` | 256 | Phantom image dimensions (pixels) |
| `n_angles_full` | 180 | Number of full-sampling angles |
| `n_angles_sparse` | 30 | Number of sparse-sampling angles |
| `angle_step_sparse` | 6 | Sparse subsampling factor |
| `noise_std` | 0.02 | Gaussian noise std (relative to sinogram max) |
| `n_detectors` | 256 | Number of detector elements per projection |
| `angle_range_deg` | 180.0 | Angular range in degrees |
| `pixel_size_mm` | 1.0 | Pixel size in mm |

## Method Hints

- **Filtered back projection** serves as the baseline but produces severe streak artifacts at 30 views.
- **TV-regularized reconstruction** exploits the piecewise-constant structure of the Shepp-Logan phantom. The optimization minimizes $\frac{1}{2}\|\mathbf{A}\mathbf{f} - \mathbf{g}\|^2 + \lambda \cdot \text{TV}(\mathbf{f})$ with optional non-negativity constraint. Primal-dual splitting methods handle the non-smooth TV term efficiently.

## References

- Sidky, E.Y. and Pan, X. (2008). "Image reconstruction in circular cone-beam computed tomography by constrained, total-variation minimization." *Physics in Medicine & Biology*, 53(17), 4777.
- Chambolle, A. and Pock, T. (2011). "A first-order primal-dual algorithm for convex problems with applications to imaging." *Journal of Mathematical Imaging and Vision*, 40(1), 120-145.
- Shepp, L.A. and Logan, B.F. (1974). "The Fourier reconstruction of a head section." *IEEE Transactions on Nuclear Science*, 21(3), 21-43.
