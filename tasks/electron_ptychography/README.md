# Electron Ptychography of Gold Nanoparticles

> Reconstruct the complex transmission function of gold nanoparticles on amorphous carbon from a defocused 4D-STEM dataset using iterative ptychographic phase retrieval.

> Domain: Physics | Keywords: electron microscopy, phase retrieval, ptychography | Difficulty: Medium

## Background

In scanning transmission electron microscopy (STEM), a focused or defocused electron probe is raster-scanned across a thin specimen. At each probe position, a convergent-beam electron diffraction (CBED) pattern is recorded on a pixelated detector, forming a four-dimensional dataset (two real-space scan dimensions and two reciprocal-space detector dimensions). This "4D-STEM" dataset encodes rich information about the specimen's atomic structure, including phase shifts imparted on the electron wave by the specimen's electrostatic potential.

Conventional imaging modes (bright-field, dark-field) discard phase information and produce images whose resolution is limited by the probe size. Phase retrieval methods recover the complex specimen transmission function, enabling quantitative measurement of the projected electrostatic potential with resolution beyond the probe size.

Three phase-contrast methods of increasing sophistication are commonly applied:

1. **Differential Phase Contrast (DPC)**: Measures the center-of-mass shift of each diffraction pattern, which is proportional to the gradient of the specimen phase. Numerical integration yields a phase image. Fast but resolution-limited by the probe.

2. **Parallax Reconstruction**: Each pixel on the bright-field disk corresponds to a plane wave at a different tilt angle. By aligning these virtual images via cross-correlation ("parallax correction") and applying contrast transfer function (CTF) correction, a focused phase image is obtained. Achieves sub-probe resolution.

3. **Ptychography**: Iteratively solves for both the complex object transmission function and the probe function by enforcing consistency between the measured diffraction intensities and the forward model. Achieves the highest resolution and simultaneously deconvolves the probe.

## Problem Description

**Forward model**: At scan position $\mathbf{r}_j$, the exit wave is the product of the probe $P(\mathbf{r} - \mathbf{r}_j)$ and the object transmission function $O(\mathbf{r})$:

$$\psi_j(\mathbf{r}) = P(\mathbf{r} - \mathbf{r}_j) \cdot O(\mathbf{r})$$

The measured diffraction intensity is:

$$I_j(\mathbf{k}) = \left| \mathcal{F}\{\psi_j(\mathbf{r})\} \right|^2$$

where $\mathcal{F}$ denotes the Fourier transform and $\mathbf{k}$ is the reciprocal-space coordinate.

**Inverse problem**: Given the 4D-STEM dataset $\{I_j(\mathbf{k})\}_{j=1}^{J}$ and optionally a vacuum probe intensity measurement, recover the complex object $O(\mathbf{r}) = A(\mathbf{r}) e^{i\phi(\mathbf{r})}$, where $\phi(\mathbf{r})$ is the projected electrostatic potential (phase) of the specimen.

The problem is non-convex due to the squared magnitude in the measurement, and ill-posed because the diffraction phases are lost. Ptychographic algorithms exploit the redundancy from overlapping probe positions to iteratively recover both the object and probe.

**Complication**: The probe is intentionally defocused (~400 Angstrom defocus), which improves probe overlap for ptychography but degrades DPC and virtual imaging.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| datacube | (1, 48, 48, 192, 192) | float32 | 4D-STEM dataset: (batch, Rx, Ry, Qx, Qy) scan and detector dimensions |
| vacuum_probe | (1, 192, 192) | float32 | Vacuum probe intensity measured on the detector (arbitrary units) |

### data/meta_data.json

| Key | Value | Description |
|-----|-------|-------------|
| energy_eV | 300000 | Electron beam energy in eV |
| R_pixel_size_A | 2.39 | Real-space scan step size in Angstroms |
| convergence_semiangle_mrad | 22.6 | Probe convergence semi-angle in milliradians |
| scan_shape | [48, 48] | Number of scan positions (Rx, Ry) |
| detector_shape | [192, 192] | Detector dimensions (Qx, Qy) |
| defocus_A | 355 | Estimated defocus in Angstroms (from parallax CTF fit) |
| com_rotation_deg | 169 | Rotation angle from scan to detector coordinates (degrees), includes 180-degree flip |

### data/ground_truth.npz

No `ground_truth.npz` is provided for this task; the specimen is a real experimental sample with no known ground truth.

### data/baseline_reference.npz

Reference reconstructions from the cleaned pipeline, used for NCC/NRMSE evaluation.

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| ptycho_phase | (1, 505, 505) | float64 | Phase of ptychographic object reconstruction |
| ptycho_complex | (1, 505, 505) | complex128 | Complex ptychographic object reconstruction |
| probe_recon | (1, 192, 192) | complex128 | Reconstructed complex probe |
| dpc_phase | (1, 48, 48) | float64 | DPC phase reconstruction |
| parallax_phase | (1, 192, 192) | float64 | Parallax phase reconstruction |

## Method Hints

- **DPC** provides a fast baseline: compute center-of-mass of each diffraction pattern, correct for the scan-to-detector rotation, and integrate the vector field to obtain phase. Apply a bright-field mask to exclude scattered beams.

- **Parallax** provides an intermediate result: align virtual bright-field images via cross-correlation shifts, upsample with kernel density estimation, fit the aberration surface (defocus and astigmatism), and apply CTF correction.

- **Ptychography** (primary method): use gradient-descent single-slice ptychography. Initialize the probe from the vacuum measurement with the defocus estimate from parallax. Use the rotation and transpose parameters from the parallax/DPC preprocessing. Run ~10 iterations with mini-batch gradient descent (batch size = J/4).

- The scan-to-detector coordinate rotation includes a 180-degree ambiguity. Physically, the reconstructed phase of gold nanoparticles must be positive (positive electrostatic potential). If the phase appears inverted, add 180 degrees to the rotation estimate.

## References

- Rodenburg, J. M. & Faulkner, H. M. L. (2004). A phase retrieval algorithm for shifting illumination. Applied Physics Letters, 85(20), 4795-4797.
- Ophus, C. (2019). Four-dimensional scanning transmission electron microscopy (4D-STEM): from scanning nanodiffraction to ptychography and beyond. Microscopy and Microanalysis, 25(3), 563-582.
- Savitzky, B. H., et al. (2021). py4DSTEM: A Software Package for Four-Dimensional Scanning Transmission Electron Microscopy (4D-STEM) Data Analysis. Microscopy and Microanalysis, 27(4), 712-743.
