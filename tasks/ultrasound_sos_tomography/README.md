# Ultrasound Speed-of-Sound Tomography

> Reconstruct a 2D speed-of-sound map of a breast-mimicking phantom from noisy travel-time measurements acquired with a ring transducer array, where limited angular sampling and measurement noise require regularized iterative reconstruction to resolve tissue structures.

> Domain: medicine | Keywords: ultrasound tomography, transmission imaging, iterative reconstruction | Difficulty: Medium

## Background

Ultrasound computed tomography (USCT) images the acoustic properties of tissue by transmitting pulses through the body from a ring array of transducers. Unlike reflection-mode ultrasound (B-mode), transmission tomography measures the travel time of acoustic waves through the medium, enabling quantitative mapping of the speed of sound. Speed-of-sound maps provide tissue characterization complementary to conventional ultrasound: fat (~1450 m/s) and glandular tissue (~1540 m/s) are distinguishable, and tumors often exhibit elevated speed of sound (~1580 m/s). USCT is actively investigated for breast cancer screening as a radiation-free alternative to X-ray mammography.

In the straight-ray approximation, the travel-time forward model is mathematically identical to the Radon transform used in X-ray CT, but operates on the slowness field (reciprocal of speed of sound) rather than attenuation.

## Problem Description

The forward model for straight-ray ultrasound transmission tomography computes travel times as line integrals of the slowness perturbation field. Defining the slowness perturbation $\Delta s(\mathbf{r}) = 1/c(\mathbf{r}) - 1/c_0$, where $c(\mathbf{r})$ is the local speed of sound and $c_0 = 1500$ m/s is the water background, the travel-time perturbation along a ray from transmitter $i$ to receiver $j$ is:

$$\Delta t_{ij} = \int_{\text{ray}_{ij}} \Delta s(\mathbf{r}) \, dl$$

In discrete form with $N$ image pixels and $M$ rays:

$$\mathbf{t} = \mathbf{A} \, \Delta\mathbf{s} + \mathbf{n}$$

where $\mathbf{A} \in \mathbb{R}^{M \times N}$ is the system matrix (ray-pixel intersection lengths, equivalent to the discrete Radon transform), $\Delta\mathbf{s} \in \mathbb{R}^N$ is the vectorized slowness perturbation, and $\mathbf{n}$ is measurement noise.

**Ill-posedness:** With 60 projection angles and 128 detector elements, the system has 7,680 measurements for 16,384 unknowns (128x128 pixels). The problem is underdetermined, and the null space of $\mathbf{A}$ produces streak artifacts in unregularized reconstructions. The high dynamic range of the phantom (water at 1500 m/s to calcification at 2500 m/s) further challenges reconstruction.

**Input:** Noisy travel-time sinogram $\mathbf{t} \in \mathbb{R}^{128 \times 60}$ and projection angles.

**Output:** Reconstructed speed-of-sound map $\hat{c} \in \mathbb{R}^{128 \times 128}$ in m/s.

## Data Description

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sos_phantom` | `(1, 128, 128)` | float64 | Speed-of-sound phantom in m/s, range [1450, 2500] |
| `slowness_perturbation` | `(1, 128, 128)` | float64 | Slowness perturbation field in s/m, defined as 1/c - 1/c_water |

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sinogram` | `(1, 128, 60)` | float64 | Noisy travel-time sinogram (60 angles), units: integrated slowness perturbation (s) |
| `sinogram_clean` | `(1, 128, 60)` | float64 | Clean travel-time sinogram (no noise), units: integrated slowness perturbation (s) |
| `sinogram_full` | `(1, 128, 180)` | float64 | Full travel-time sinogram (180 angles), units: integrated slowness perturbation (s) |
| `angles` | `(1, 60)` | float64 | Sparse projection angles in degrees |
| `angles_full` | `(1, 180)` | float64 | Full projection angles in degrees |

### data/meta_data.json

| Key | Value | Description |
|-----|-------|-------------|
| `image_size` | 128 | Phantom image dimensions (pixels) |
| `n_angles` | 60 | Number of sparse projection angles |
| `n_angles_full` | 180 | Number of full-sampling angles |
| `noise_std` | 0.01 | Gaussian noise std (relative to sinogram max) |
| `n_detectors` | 128 | Number of detector elements per projection |
| `angle_range_deg` | 180.0 | Angular range in degrees |
| `pixel_size_mm` | 0.5 | Pixel size in mm |
| `background_sos_m_per_s` | 1500.0 | Background speed of sound (water) in m/s |
| `ring_radius_mm` | 50.0 | Transducer ring radius in mm |

## Method Hints

- **Filtered back projection (FBP)** serves as the baseline. It applies frequency-domain ramp filtering to the sinogram and backprojects, treating the problem as a standard Radon inversion. With 60 angles it produces mild streak artifacts.
- **SART (Simultaneous Algebraic Reconstruction Technique)** iteratively updates the image by back-projecting normalised residuals. It reduces artifacts compared to FBP but does not enforce structural priors.
- **TV-regularized reconstruction** with the Chambolle-Pock primal-dual algorithm (PDHG) is the recommended approach. Total variation regularization promotes piecewise-constant structure, which matches the geometric phantom well. The optimization problem is: $\min_{\Delta\mathbf{s}} \frac{1}{2}\|\mathbf{A}\Delta\mathbf{s} - \mathbf{t}\|^2 + \lambda \cdot \text{TV}(\Delta\mathbf{s})$.
- scikit-image's `radon`/`iradon` provide CPU-based forward and adjoint operators. Using `circle=True` restricts reconstruction to the inscribed circle.
- The reconstruction operates on slowness perturbation; speed of sound is recovered as $c = 1/(\Delta s + 1/c_0)$.

## References

- Duric, N. et al. (2007). "Detection of breast cancer with ultrasound tomography: First results with the Computed Ultrasound Risk Evaluation (CURE) prototype." *Medical Physics*, 34(2), 773-785.
- Li, C. et al. (2009). "In vivo breast sound-speed imaging with ultrasound tomography." *Ultrasound in Medicine & Biology*, 35(10), 1615-1628.
- Chambolle, A. and Pock, T. (2011). "A first-order primal-dual algorithm for convex problems with applications to imaging." *Journal of Mathematical Imaging and Vision*, 40(1), 120-145.
