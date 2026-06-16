# X-ray Laminography/Tomography Reconstruction

> Reconstruct a 3D complex-valued volume from simulated parallel-beam projections of a nanoparticle assembly phantom using iterative conjugate gradient.

**Domain:** Physics
**Keywords:** tomographic imaging, iterative reconstruction
**Difficulty:** Easy

---

## Background

Laminography is a 3D imaging technique that generalizes computed tomography (CT) by
tilting the rotation axis relative to the beam direction. When the tilt angle equals
pi/2, the geometry reduces to standard parallel-beam tomography; at other tilt angles
the technique is called laminography, and it is particularly useful for imaging flat,
laterally extended specimens that cannot be rotated through a full 180 degrees in
conventional CT.

The object under study is a nanoparticle assembly (NALM) phantom -- a synthetic
3D complex-valued volume that models the complex refractive index distribution of
a nanomaterial sample. An asymmetric marker is embedded in the phantom to break
rotational symmetry, making it possible to visually verify reconstruction orientation.

---

## Problem Description

The forward model is the laminographic projection operator, which for a given tilt
angle `tilt` and rotation angle `theta`, maps a 3D complex-valued volume to a 2D
complex-valued projection:

$$d(\theta) = \mathcal{P}_{\text{tilt}}[\, V \,](\theta)$$

where V is the 3D volume (complex refractive index), theta is the rotation angle, and
the projection operator integrates through the volume along rays determined by the tilt
geometry.

For `tilt = pi/2` (standard tomography), the projection reduces to the Radon
transform -- integration along lines perpendicular to the detector plane for each
rotation angle:

$$d(\theta, u, v) = \int V(R_\theta \cdot \mathbf{r})\, dl$$

where R_theta is the rotation matrix for angle theta and the integral is along the
beam direction.

In Fourier space, the laminographic forward model corresponds to extracting tilted
planes from the 3D Fourier transform of the volume (a generalization of the Fourier
Slice Theorem).

**Inverse problem:** Given N_theta projections {d(theta_i)} at known angles, recover
the 3D volume V. This is a linear inverse problem that is typically underdetermined
or ill-conditioned, especially for limited angular coverage or laminographic (non-pi/2)
tilt angles.

| Symbol | Description | Size |
|--------|-------------|------|
| **V** | 3D complex refractive index volume | 128 x 128 x 128 |
| **d** | Complex projection data (sinogram) | 128 x 128 x 128 (n_theta x n x n) |
| **theta** | Rotation angles | 128 |
| **tilt** | Tilt angle between rotation axis and beam | scalar (pi/2 radians) |

---

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `projections` | (1, 128, 128, 128) | complex64 | Simulated laminographic projections (batch, n_angles, n, n), unitless complex values |
| `theta` | (1, 128) | float32 | Projection angles in radians, uniformly spaced over [0, pi) |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `volume` | (1, 128, 128, 128) | complex64 | Ground truth 3D complex refractive index of the NALM phantom with asymmetric marker, unitless |

### data/meta_data.json

Contains imaging parameters:
- `volume_shape`: [128, 128, 128] -- dimensions of the 3D volume
- `n_angles`: 128 -- number of projection angles
- `tilt_rad`: 1.5708 (pi/2) -- tilt angle in radians
- `tilt_deg`: 90.0 -- tilt angle in degrees
- `theta_range_rad`: [0.0, pi] -- range of rotation angles

---

## Method Hints

Use an iterative conjugate gradient solver to minimize the data-fidelity objective:

$$\min_V \| \mathcal{P}_{\text{tilt}}[V] - d \|^2$$

The `tike` library provides GPU-accelerated implementations of the laminographic forward
and adjoint operators, as well as a conjugate gradient solver (`algorithm='cgrad'`).
The reconstruction can be organized as multiple rounds of a fixed number of CG
iterations, allowing the cost function to be monitored between rounds. With 5 rounds
of 4 iterations each (20 total iterations), the cost converges from approximately 1.24
to approximately 0.013 for this dataset.

---

## References

- Ching, D. J., & Gursoy, D. (2020). "XDesign: an open-source software package for
  designing X-ray imaging phantoms and experiments." Journal of Synchrotron Radiation.
- Tike: https://github.com/AdvancedPhotonSource/tike
