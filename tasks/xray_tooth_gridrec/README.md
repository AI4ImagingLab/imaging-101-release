# X-ray CT Tooth Reconstruction (Gridrec)

> Reconstruct a 2-D cross-section of a tooth from parallel-beam X-ray transmission projections (sinograms), where the challenge is accurate rotation-center estimation and ramp-filter design for artifact-free filtered back-projection.

> Domain: Medicine | Keywords: computed tomography, filtered back-projection | Difficulty: Easy

## Background

X-ray computed tomography (CT) recovers the internal structure of an object from a series of X-ray transmission measurements taken at different angles. In parallel-beam geometry, each projection records the line integral of the linear attenuation coefficient along parallel rays. The collection of all projections for a fixed axial slice forms a sinogram. Recovering the attenuation map from the sinogram is the inverse Radon transform.

## Problem Description

The forward model is the Radon transform. For a 2-D attenuation image $\mu(x, y)$, the projection at angle $\theta$ and detector position $t$ is:

$$p(\theta, t) = \int_{-\infty}^{\infty} \mu(t \cos\theta - s \sin\theta,\; t \sin\theta + s \cos\theta)\, ds$$

The inverse problem is to recover $\mu(x, y)$ from the discrete sinogram $p(\theta_i, t_j)$ measured at $N_\theta$ angles and $N_t$ detector pixels.

The problem is mildly ill-posed: the Radon transform has a smoothing effect (high-frequency information is attenuated), and the finite number of projections causes angular undersampling. Noise in the measured projections amplifies high-frequency errors in the reconstruction. Additionally, if the rotation center is not correctly identified, the reconstruction exhibits characteristic doubling artifacts.

**Input:** Raw detector counts (projections), flat-field images, dark-field images, and projection angles.

**Output:** Reconstructed 2-D attenuation cross-section of the tooth.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `projections` | `(1, 181, 2, 640)` | float32 | Raw detector counts at 181 projection angles, 2 sinogram rows, 640 detector pixels |
| `flat_field` | `(1, 10, 2, 640)` | float32 | Flat-field (open beam) reference images, 10 frames |
| `dark_field` | `(1, 10, 2, 640)` | float32 | Dark-field (beam off) reference images, 10 frames |
| `theta` | `(1, 181)` | float64 | Projection angles in radians, uniformly spaced from 0 to ~pi |

### data/meta_data.json

Imaging parameters: number of projections (181), sinogram rows (2), detector pixels (640), flat/dark field counts (10 each), angular range (0 to 3.12 rad), and modality (X-ray transmission CT).

### data/baseline_reference.npz

No `ground_truth.npz` is provided for this task; the data are from a real experiment (APS beamline 2-BM/32-ID tooth specimen). The baseline reference is a gridrec reconstruction produced by TomoPy.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `reconstruction` | `(1, 2, 640, 640)` | float32 | Gridrec baseline reconstruction, 2 slices of 640x640 pixels, attenuation coefficient (dimensionless) |

## Method Hints

The standard analytical solution is **Filtered Back-Projection (FBP)**:

1. **Preprocessing:** Apply flat-field correction to convert raw counts to transmission: $(I - I_\text{dark}) / (I_\text{flat} - I_\text{dark})$. Then apply $-\ln(\cdot)$ to obtain sinogram (line-integral) data via the Beer-Lambert law.

2. **Rotation center:** Find the center of rotation using cross-correlation of opposing projections (projections at $\theta$ and $\theta + \pi$ are mirror images). An incorrect center produces characteristic doubling artifacts.

3. **Ramp filtering:** Apply the ramp (Ram-Lak) filter $|\omega|$ in the 1-D Fourier domain to each projection. This compensates for the $1/|\omega|$ blurring inherent in back-projection.

4. **Back-projection:** For each image pixel $(x, y)$, sum the filtered projection values at the corresponding detector position $t = x\cos\theta + y\sin\theta$ across all angles. This produces the final attenuation image.

The gridrec algorithm is an FFT-based variant that performs the filtering and back-projection jointly via gridding in 2-D Fourier space, yielding equivalent results with better computational efficiency.

## References

- Dowd, B.A., Campbell, G.H., Marr, R.B., Nagarkar, V.V., Tipnis, S.V., Axe, L. and Siddons, D.P. (1999). Developments in synchrotron x-ray computed microtomography at the National Synchrotron Light Source. *Proc. SPIE*, 3772.
- Gursoy, D., De Carlo, F., Xiao, X., and Jacobsen, C. (2014). TomoPy: a framework for the analysis of synchrotron tomographic data. *Journal of Synchrotron Radiation*, 21(5), 1188-1193.
- Kak, A.C. and Slaney, M. (2001). *Principles of Computerized Tomographic Imaging*. SIAM.
