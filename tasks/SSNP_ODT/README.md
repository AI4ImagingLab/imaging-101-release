# SSNP Intensity Diffraction Tomography

> Recover the 3D refractive index distribution of a biological sample from intensity-only measurements under multi-angle LED illumination, using a non-paraxial multiple-scattering forward model.

> Domain: Biology | Keywords: tomography, wave imaging, inverse scattering | Difficulty: Hard

---

## Background

Optical Diffraction Tomography (ODT) recovers the 3D refractive index (RI)
distribution of transparent biological samples. Traditional ODT requires
coherent illumination and an interferometric setup to measure the complex
optical field (amplitude + phase). **Intensity Diffraction Tomography (IDT)**
is a simpler alternative that recovers 3D phase information from intensity-only
measurements on a standard microscope with a programmable LED array.

The key challenge: intensity measurements lose phase information, and the
relationship between RI and intensity is highly nonlinear — especially for
strongly scattering samples like *C. elegans* worms and embryos. Simple
linear models (first-Born, first-Rytov) assume weak scattering and break
down for such samples. The **Beam Propagation Method (BPM)** handles
multiple scattering but relies on a paraxial approximation that degrades
accuracy under high-NA illumination.

This task uses the **Split-Step Non-Paraxial (SSNP)** model, which
simultaneously propagates the field and its axial derivative through the
sample — accurately modeling both diffraction and scattering without
paraxial assumptions.

---

## Problem Description

### SSNP Forward Model

The scalar wave in an inhomogeneous medium with RI distribution n(**r**) satisfies
the Helmholtz equation. The SSNP model reformulates this as a first-order system
by defining a state vector containing both the field and its z-derivative:

$$
\boldsymbol{\Phi}(\mathbf{r}) = \begin{pmatrix} \varphi(\mathbf{r}) \\\\ \partial\varphi / \partial z \end{pmatrix}
$$

The evolution along z is governed by:

$$
\frac{\partial \boldsymbol{\Phi}}{\partial z} = \mathbf{H}(\mathbf{r})\boldsymbol{\Phi}(\mathbf{r})
$$

The operator **H** is split into a diffraction part **H1** (spatially invariant,
computed in Fourier space) and a scattering part **H2** (spatially variant,
computed in real space):

**P operator (diffraction/propagation)** — applied in Fourier space:

$$
\hat{\mathbf{P}} = \begin{pmatrix} \cos(k_z \Delta z) & \sin(k_z \Delta z)/k_z \\\\ -k_z \sin(k_z \Delta z) & \cos(k_z \Delta z) \end{pmatrix}
$$

where $k_z = \sqrt{k_0^2 n_0^2 - k_x^2 - k_y^2}$ and evanescent components are damped.

**Q operator (scattering)** — applied in real space:

$$
\mathbf{Q}(z)\boldsymbol{\Phi} = \begin{pmatrix} 1 & 0 \\\\ k_0^2(n_0^2 - n_{xy}^2)\Delta z & 1 \end{pmatrix}\boldsymbol{\Phi}
$$

The forward model propagates a tilted plane wave through the sample by
alternating P and Q slice-by-slice, then propagates to the camera plane, applies the microscope pupil function, extracts the forward-propagating
component, and takes the intensity: $I_{out} = |\varphi_{out}|^2$.

### Inverse Problem

The 3D RI is reconstructed by minimizing:

$$
\hat{n} = \arg\min_{n \geq n_0} \sum_{l=1}^{L} \left\| \sqrt{I_{out}^l} - \sqrt{I_{meas}^l} \right\|^2 + \tau R_{TV}(n)
$$

The amplitude-domain comparison is more robust to signal-dependent noise than
intensity-domain. 3D Total Variation regularization promotes piecewise-smooth
RI distributions. The gradient is computed by automatic differentiation
(backpropagation through the SSNP forward model).

---

## Data Description

### `data/raw_data.npz`

Simulated intensity measurements for all illumination angles.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `measurements` | `(1, 8, 256, 256)` | float64 | Intensity images — batch × n_angles × Ny × Nx |

### `data/ground_truth.npz`

Ground truth refractive index contrast volume.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `delta_n` | `(1, 308, 256, 256)` | float32 | RI contrast Δn = n(x,y,z) − n₀ — batch × Nz × Ny × Nx |

`delta_n` is stored as float32 to stay within the GitHub 100 MB file size limit (~78 MB).
Load with `.astype(np.float64)` before computation.

### `data/meta_data.json`

JSON file with imaging parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `volume_shape` | [308, 256, 256] | Nz x Ny x Nx voxels |
| `res_um` | [0.1, 0.1, 0.1] | Voxel size in micrometers (dz, dy, dx) |
| `wavelength_um` | 0.5 | Illumination wavelength in μm |
| `n0` | 1.0 | Background refractive index |
| `NA` | 0.65 | Objective numerical aperture |
| `n_angles` | 8 | Number of illumination angles |
| `ri_contrast_scale` | 0.01 | RI contrast scaling factor |

### Illumination Geometry

8 LED positions uniformly distributed on a ring at NA = 0.65 (annular IDT).
Each LED approximates a plane wave with direction:

$$
k_x^{in} = \text{NA} \cdot \cos(2\pi m / 8), \quad k_y^{in} = \text{NA} \cdot \sin(2\pi m / 8), \quad m = 0, \ldots, 7
$$

---

## Method Hints

1. **Forward model**: Implement the SSNP propagation (alternating P and Q operators)
   in PyTorch for GPU acceleration and automatic differentiation.

2. **Simulate measurements**: Propagate tilted plane waves through the phantom for
   each illumination angle to generate synthetic intensity images.

3. **Reconstruction**: Use gradient descent on the amplitude MSE loss with 3D TV
   regularization. PyTorch autograd computes the gradient of the loss with respect
   to the RI volume automatically through the forward model.

4. **Constraints**: Enforce non-negativity of RI contrast (Dn >= 0) by projection
   after each gradient step.

---

## References

- Zhu, J., Wang, H., & Tian, L. (2022). *High-fidelity intensity diffraction tomography with a non-paraxial multiple-scattering model*. Optics Express, 30(18), 32808-32821.
- Lim, J., Ayoub, A.B., Antoine, E.E., & Psaltis, D. (2019). *High-fidelity optical diffraction tomography of multiple scattering samples*. Light: Science & Applications, 8(1), 82.
- Sharma, A. & Agrawal, A. (2004). *New method for nonparaxial beam propagation*. J. Opt. Soc. Am. A, 21(6), 1082-1087.
