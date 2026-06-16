# Photoacoustic Tomography (PAT) Reconstruction

> Recover the initial pressure distribution of spherical absorbers from acoustic time-series measured on a planar detector array, using universal back-projection.

> Domain: medicine | Keywords: photoacoustic imaging, wave imaging, back-projection | Difficulty: Medium

## Background

Photoacoustic tomography (PAT) exploits the photoacoustic effect: a short laser pulse illuminates tissue, optical absorbers (e.g., blood vessels) absorb the light and undergo rapid thermoelastic expansion, launching broadband ultrasonic pressure waves. These waves propagate outward and are recorded by an array of ultrasonic detectors on the tissue boundary. The inverse problem is to recover the initial pressure distribution (proportional to absorbed optical energy) from the measured acoustic time-series.

PAT combines the high contrast of optical imaging with the high spatial resolution of ultrasound, making it valuable for vascular imaging, tumour detection, and functional imaging.

## Problem Description

### Forward model

For a spherical absorber of radius $R_a$ centred at position $\mathbf{r}_s$, the photoacoustic pressure detected at position $\mathbf{r}_d$ is:

$$p(\mathbf{r}_d, t) = \frac{R - ct}{2R} \cdot \Theta\bigl(R_a - |R - ct|\bigr)$$

where $R = |\mathbf{r}_d - \mathbf{r}_s|$ is the source-detector distance, $c$ is the speed of sound, $t$ is time, and $\Theta$ is the Heaviside step function. The signal is nonzero only when $|R - ct| \leq R_a$, i.e., when the expanding wavefront overlaps the absorber.

Each physical detector has a finite aperture (2 mm square). Spatial integration across the detector surface is modelled by subdividing each detector into a 5x5 grid of sub-elements and averaging their signals.

For multiple targets, the total measured signal is the superposition of individual contributions (linear acoustics).

### Inverse problem

Given boundary measurements $p(\mathbf{r}_d, t)$ on the detector array, recover the initial pressure distribution $p_0(\mathbf{r})$ in the target plane. This is ill-posed because:

1. **Limited aperture**: the planar detector array covers a finite solid angle, so spatial frequencies outside the detection cone are unobserved.
2. **Finite detector size**: spatial averaging over each detector acts as a low-pass filter.
3. **Bandwidth limitations**: the discrete time sampling and finite recording duration limit the recoverable frequency band.

**Input**: pressure time-series $p(\mathbf{r}_d, t)$ of shape `(n_time, n_det_x, n_det_y)`, detector positions, time vector, and imaging parameters.

**Output**: 2D image $p_0(x, y)$ at the target plane depth.

## Data Description

### data/raw_data.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `signals` | `(1, 1301, 31, 31)` | float64 | PA time-series at each detector (Pa, arbitrary units) |
| `detector_x` | `(1, 31)` | float64 | X-coordinates of detector centres (m) |
| `detector_y` | `(1, 31)` | float64 | Y-coordinates of detector centres (m) |
| `time_vector` | `(1, 1301)` | float64 | Time samples (s), 20 MHz sampling over 65 us |

### data/ground_truth.npz

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `ground_truth_image` | `(1, 41, 41)` | float64 | Binary target map: 1 inside absorbers, 0 outside (dimensionless) |
| `image_x` | `(1, 41)` | float64 | X-coordinates of image pixels (m) |
| `image_y` | `(1, 41)` | float64 | Y-coordinates of image pixels (m) |

### data/meta_data.json

Contains imaging parameters: sound speed (1484 m/s), sampling frequency (20 MHz), aperture length (20 mm), detector pitch (2/3 mm), detector size (2 mm), number of sub-detectors (25), target plane depth (15 mm), and array dimensions (31x31).

## Method Hints

**Universal back-projection (Xu & Wang 2005)** is the standard analytical reconstruction for PAT. It recovers the initial pressure distribution by ramp-filtering each detector's time-series, mapping recorded signals back to image pixels via time-of-flight delays, and accumulating contributions with solid-angle weighting. This approach is non-iterative, handles arbitrary detection geometries, and the ramp filter compensates for the derivative relationship between initial pressure and the measured acoustic field.

## References

- M. Xu and L. V. Wang, "Universal back-projection algorithm for photoacoustic computed tomography," Physical Review E, vol. 71, no. 1, 016706, 2005.
- Reference implementation: csheaff/pa-tomography-xu-wang (GitHub).
