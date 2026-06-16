# Fan-Beam CT Reconstruction

> Reconstruct a 2D image from fan-beam CT sinograms acquired with divergent X-ray geometry, requiring distance-dependent weighting and magnification correction absent in parallel-beam CT, with additional complexity from short-scan acquisition and Parker weighting.

> Domain: medicine | Keywords: computed tomography, fan-beam geometry, filtered back-projection | Difficulty: Medium-Hard

## Background

In clinical CT scanners, X-rays emanate from a point source that rotates around the patient, creating a divergent fan-beam geometry. This differs fundamentally from the parallel-beam Radon transform: each projection ray has a different angle and passes through the object at a magnification that depends on the source-to-object distance. Fan-beam CT is the standard clinical acquisition geometry for all modern diagnostic CT scanners.

A **short-scan** acquisition covers only $\pi + 2\gamma_{\max}$ radians (where $\gamma_{\max}$ is the half-fan angle), which is sufficient for exact reconstruction but requires **Parker weighting** — a smooth angular weighting function that compensates for the reduced and partially redundant angular coverage.

## Problem Description

The fan-beam forward model computes the line integral of the attenuation map $f(x, y)$ along each divergent ray from the X-ray source to a detector element:

$$p(\beta, t) = \int_0^\infty f\!\left(s_x(\beta) + \ell \cdot d_x(\beta, t),\; s_y(\beta) + \ell \cdot d_y(\beta, t)\right) d\ell$$

where:
- $\beta$ is the source rotation angle
- $t$ is the detector element position
- $(s_x, s_y) = (D_{sd} \cos\beta,\; D_{sd} \sin\beta)$ is the source position on a circle of radius $D_{sd}$
- $(d_x, d_y)$ is the unit direction from source to detector element
- $D_{sd}$ is the source-to-isocenter distance, $D_{dd}$ is the isocenter-to-detector distance

**Inverse problem:** Given the fan-beam sinogram $p(\beta, t)$, reconstruct the attenuation image $f(x, y)$.

**Challenges specific to fan-beam:**
- Divergent ray geometry requires magnification-dependent weighting: the apparent detector spacing varies with object position
- Back-projection must account for distance weighting $1/U^2$ where $U$ depends on each pixel's position relative to the source
- Short-scan acquisition requires Parker weighting to avoid artifacts from incomplete angular coverage
- The fan-beam FBP formula involves pre-weighting by $D_{sd}/\sqrt{D_{sd}^2 + \gamma^2}$ before filtering

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `sino_full` | `(1, 180, 192)` | float32 | Full-scan (360 deg) noisy fan-beam sinogram |
| `sino_short` | `(1, ~120, 192)` | float32 | Short-scan (pi + 2*gamma_max) noisy sinogram |
| `angles_full` | `(180,)` | float32 | Full-scan projection angles (radians) |
| `angles_short` | `(~120,)` | float32 | Short-scan projection angles (radians) |
| `det_pos` | `(192,)` | float32 | Detector element positions (pixels, centered) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `phantom` | `(1, 128, 128)` | float32 | Ground truth Shepp-Logan phantom (dimensionless attenuation) |

### data/meta_data.json

| Key | Type | Description |
|-----|------|-------------|
| `image_size` | int | Image dimension (128) |
| `n_det` | int | Number of detector elements (192) |
| `n_angles_full` | int | Full-scan projection count (180) |
| `n_angles_short` | int | Short-scan projection count |
| `source_to_isocenter_pixels` | float | Source-to-rotation-center distance (256) |
| `isocenter_to_detector_pixels` | float | Center-to-detector distance (256) |
| `fan_half_angle_deg` | float | Half fan-beam opening angle (degrees) |
| `short_scan_range_deg` | float | Short-scan angular range (degrees) |
| `noise_sigma_relative` | float | Gaussian noise level relative to max sinogram value |
| `detector_type` | str | Detector geometry ("flat") |

## Method Hints

**Fan-beam FBP** (analytical reconstruction): The standard fan-beam FBP consists of distance-dependent pre-weighting of the sinogram, ramp filtering, and distance-weighted back-projection with magnification correction. For short-scan acquisitions ($\pi + 2\gamma_{\max}$), Parker weighting must be applied to smoothly handle the redundant angular overlap at the scan edges. The key difference from parallel-beam FBP is the $1/U^2$ distance weighting and the magnification mapping from pixel coordinates to detector coordinates during back-projection.

**Iterative reconstruction** (e.g., TV-regularized): Formulate as a convex optimization minimizing data fidelity plus total variation regularization. This requires implementing both the fan-beam forward projection operator and its adjoint (back-projection). The Chambolle-Pock (PDHG) algorithm is well-suited, with separate dual variables for the data fidelity and TV terms. The forward model maps each image pixel to its projected detector coordinate via the fan-beam magnification geometry.

## References

- Kak, A.C. and Slaney, M. (1988). Principles of Computerized Tomographic Imaging. IEEE Press. Chapter 3: Fan-beam algorithms.
- Parker, D.L. (1982). Optimal short scan convolution reconstruction for fan beam CT. Medical Physics, 9(2), 254-257.
- leehoy/CTReconstruction: https://github.com/leehoy/CTReconstruction — Fan-beam FBP reference (Python).
- xtie97/CT_fanbeam_recon_numba: https://github.com/xtie97/CT_fanbeam_recon_numba — Fan-beam forward projection reference.
