# Ultrasonic Notch Total Focusing

> Reconstruct an ultrasonic reflectivity image of an aluminium notch from measured reciprocal array echoes despite limited aperture and coherent clutter.
> Domain: mechanics | Keywords: ultrasonic imaging, nondestructive testing, coherent beamforming | Difficulty: Medium

## Background

Elastic waves propagate through solids and scatter at cracks, voids, boundaries and inclusions. Recording echoes with an array permits spatial localization of internal discontinuities without cutting the component. In pulse-echo inspection, both the outward and returning travel paths determine the arrival time.

Array inspection supports manufacturing quality control, structural integrity assessment and maintenance of safety critical metal components. Finite aperture, acoustic bandwidth, multipath reflection and coherent interference constrain the attainable resolution. Echo amplitude is an instrument-dependent scattering signal rather than an absolute defect concentration.

## Problem Description

The observations are a measured contact inspection of an aluminium specimen containing a manufactured notch. Assume a homogeneous longitudinal-wave speed and straight propagation from each surface element. For transmitter i, receiver j, point r and scattering amplitude a(r), the single-scattering acquisition approximation is

```math
s_{ij}(t)=\int a(\mathbf r)h\!\left(t-\tau_i(\mathbf r)-\tau_j(\mathbf r)\right)d\mathbf r+n_{ij}(t),\qquad
\tau_i(\mathbf r)=\|\mathbf r-\mathbf p_i\|/c.
```

Here s is the measured voltage normalized by the digitizer full scale, h is the transmitted and received impulse response, n includes noise and model mismatch, p is an element position in metres and c is wave speed. Only one reciprocal half of the transmit-receive matrix is recorded. Its off-diagonal entries represent both reciprocal paths. The finite array and band-limited pulse make the spatial inverse problem ill-posed.

Save `output/reconstruction.npz` containing `reconstruction`, a real finite float64 array of shape `(1, 251, 201)`, axes `(batch, depth z, lateral x)`, in dimensionless normalized echo-amplitude units. The physical grid is x=-20 to20mm and z=0 to50mm, both inclusive, spacing0.2mm; positive z points into the sample, y=0. Output is the nonnegative analytic echo envelope on the native amplitude scale, without per-image normalization. Reference agreement uses cosine NCC and RMS error divided by reference dynamic range, with no alignment, masking, rescaling or crop. The comparison evaluates method agreement, not independently measured defect truth.

## Data Description

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| traces | (1, 2080, 750) | float64 | Measured HMC traces, dimensionless digitizer full-scale amplitude; axes batch, pair, time |
| tx | (1, 2080) | uint32 | Zero-based transmitting element index, dimensionless |
| rx | (1, 2080) | uint32 | Zero-based receiving element index, dimensionless |
| time | (1, 750) | float64 | Acquisition sample time in seconds |
| positions | (1, 64, 3) | float64 | Element centres in metres, coordinate order x,y,z |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Longitudinal speed | 6300 | m/s | Homogeneous aluminium propagation assumption |
| Sampling interval | 4e-8 | s | 25MHz digitizer sampling |
| Time origin | 0 | s | Stored acquisition time origin, no additional fitted delay |
| Elements | 64 | 1 | Linear surface array |
| Pitch | 0.0003 | m | Uniform element centre spacing |
| Element centre range x | -0.00945 to0.00945 | m | Array centred at x=0 |
| Element coordinates y,z | 0,0 | m | Contact surface at z=0 |
| Recording mode | reciprocal half matrix | 1 | Includes diagonal and upper triangular pairs |
| Amplitude quantization | 1/128 | 1 | Observed stored digitizer increment |
| Noise variance | unspecified | 1 | Real measured electronic noise and clutter, no calibrated Gaussian variance |
| Boundary convention | finite homogeneous half-space model | 1 | Direct arrivals only; out-of-record times contribute zero |

Data originate from the measured example accompanying the cited ultrasonic array study. No new synthetic observations are generated. Data/code redistribution is under the accompanying MIT notice.

## Method Hints

Coherent total focusing evaluates each analytic trace at the sum of its outward and returning travel times. Reciprocal off-diagonal records have weight2, diagonal records weight1. Summation is normalized by the number of recorded pairs. A zero-phase fifth-order Butterworth bandpass spanning0.5–6MHz suppresses out-of-band noise; an analytic-signal transform then permits envelope imaging. Preserve the acquisition amplitude scale and sample-time convention.

## References

Budyn et al., *A Model for Multiview Ultrasonic Array Inspection of Small Two-Dimensional Defects*, IEEE TUFFC (2019), [doi:10.1109/TUFFC.2019.2909988](https://doi.org/10.1109/TUFFC.2019.2909988). This task binds the author's real contact-notch example and the direct-wave total-focusing component; it does not reproduce all immersion views in the paper.
