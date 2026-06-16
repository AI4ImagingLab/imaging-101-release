# X-ray Ptychography Reconstruction

> Reconstruct a complex-valued transmission function from far-field diffraction patterns collected by raster-scanning a focused X-ray probe across a Siemens star test sample.

> Domain: Physics | Keywords: wave imaging, phase retrieval | Difficulty: Medium

## Background

Ptychography is a coherent diffractive imaging technique that recovers both the amplitude and phase of a specimen's transmission function. A focused X-ray beam (the "probe") is raster-scanned across the sample with overlapping illumination areas. At each scan position, a far-field diffraction pattern is recorded on a photon-counting detector. Because adjacent positions share object information, the redundancy in the dataset enables simultaneous recovery of the complex object and the illumination probe, even though only intensity (not phase) is measured at the detector.

This task uses real experimental data collected at the Velociprobe instrument at the Advanced Photon Source (APS), Argonne National Laboratory. The sample is a Siemens star test pattern, a standard resolution target in X-ray microscopy.

## Problem Description

The measurement process for a single scan position $j$ is modeled as:

$$I_j = \left| \mathcal{F} \left[ P \cdot O_j \right] \right|^2$$

where:
- $O_j$ is the patch of the complex object $\psi$ extracted at scan position $\mathbf{r}_j$
- $P$ is the complex-valued illumination probe
- $\mathcal{F}$ denotes the 2D discrete Fourier transform (far-field propagation)
- $I_j$ is the measured diffraction intensity at the detector

The inverse problem is: given the set of measured intensities $\{I_j\}_{j=1}^{N}$, the scan positions $\{\mathbf{r}_j\}$, and an initial estimate of the probe $P$, recover the complex-valued object transmission function $\psi$ (and optionally refine $P$).

This problem is ill-posed because the detector measures only intensity, discarding all phase information. The overlap between adjacent scan positions provides the constraint that makes recovery possible, but the problem remains non-convex with no closed-form solution.

**Inputs:**
- Diffraction patterns: $(N, W, H)$ real-valued intensities
- Scan positions: $(N, 2)$ pixel coordinates
- Probe guess: $(1, 1, S, W, H)$ complex-valued illumination function

**Outputs:**
- Reconstructed object: $(D, W', H')$ complex-valued transmission function (D=1 for single-slice)
- Reconstructed probe: $(1, 1, S, W, H)$ refined illumination function
- Cost history: per-iteration objective function values

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `diffraction_patterns` | (1, 516, 128, 128) | float32 | Measured far-field diffraction intensities in photon counts. Batch dimension first. |
| `scan_positions` | (1, 516, 2) | float32 | Probe center coordinates in pixel units, relative to the object frame. |
| `probe_guess` | (1, 1, 1, 1, 128, 128) | complex64 | Initial estimate of the focused X-ray probe illumination function. |

### data/meta_data.json

Contains imaging parameters: `n_positions` (516), `probe_shape` [1,1,1,128,128], `diffraction_shape` [128,128], `scan_range_pixels`, `data_min`, `data_max`, and `description`.

### data/ground_truth.npz

No `ground_truth.npz` is provided for this task; this is real experimental data with no simulation ground truth available. Instead, a `baseline_reference.npz` is used for evaluation, containing the output of the reference reconstruction pipeline.

### data/baseline_reference.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `object_phase` | (1, H, W) | float32 | Phase of the reconstructed object from the baseline pipeline. |
| `object_amplitude` | (1, H, W) | float32 | Amplitude of the reconstructed object from the baseline pipeline. |

## Method Hints

Use an iterative ptychographic engine that alternates between updating the object and the probe. The least-squares gradient (lstsq_grad) algorithm family is well-suited: it uses gradient-based updates with automatic step-size selection and supports mini-batching over scan positions.

Key considerations:
- **Mini-batching**: Dividing the scan positions into batches (e.g., 7 batches for 516 positions) provides stochastic acceleration and reduces per-iteration memory.
- **Probe refinement**: Jointly updating the probe alongside the object is essential because the initial probe guess may not exactly match the experimental illumination.
- **Object initialization**: A uniform complex array (e.g., $0.5 + 0i$) is a reasonable starting point for the object.
- **Scan position preprocessing**: Shift scan coordinates so the minimum position has a small positive offset (e.g., 20 pixels) to ensure all positions map to valid regions of the object array.

The domain-standard library `tike` provides a complete implementation of this algorithm family.

## References

- Thibault, P., et al. "High-resolution scanning x-ray diffraction microscopy." Science 321.5887 (2008): 379-382.
- Ching, D. J. and Hidayetoglu, M. and Leyffer, S. and Wild, S. M. and Gursoy, D. "Tike: a toolbox for tomographic reconstruction of ptychographic data." (2024).
