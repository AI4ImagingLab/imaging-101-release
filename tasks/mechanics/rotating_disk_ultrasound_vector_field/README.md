# Rotating Disk Ultrasound Vector Field

> Recover both components of a disk's in-plane velocity field from measured radio-frequency ultrasound pulse ensembles, despite carrier undersampling, wrapped phase and noisy lateral motion sensitivity.
> Domain: mechanics | Keywords: wave imaging, vector velocimetry, phase inference | Difficulty: Hard

## Background

Ultrasound scattered by a moving material carries a phase shift that depends on its displacement along the propagation directions. Rapid pulse ensembles can measure this shift much more finely than an image can resolve the corresponding displacement. Steering the receive aperture provides different directional measurements of the same motion.

A single Doppler view primarily measures motion toward or away from the transducer. Several directions can recover a vector field, but small angular separation amplifies lateral errors. Phase wrapping, speckle decorrelation, weak echoes and finite spatial resolution complicate the inference. These effects matter when ultrasound is used to study solid-body motion, fluid flow and deformation in opaque media.

## Problem Description

The observations are a real author-distributed ensemble of 32 unsteered plane-wave acquisitions of a rotating disk in water. They are signed RF samples from all 128 array elements, already saved at a retrospectively reduced sampling rate. Recover the two-dimensional velocity field on the complete supplied disk support. The supplied support describes the known object geometry; no rotation rate or true velocity field is supplied.

Let $q$ index transmitted pulses, $j$ index receive elements, $f_c$ be the carrier frequency and $c$ the sound speed. An echo's travel time combines its transmit and receive paths. After coherent focusing at a point, motion with lateral component $v_x$ and depth component $v_z$ changes the complex phase. For receive directions $+\theta$ and $-\theta$ and normal transmission, the short-ensemble bistatic relation is

```math
\Delta\phi_\pm=
-\frac{2\pi f_c}{c f_p}
\bigl[\pm\sin\theta\,v_x+(1+\cos\theta)\,v_z\bigr]
\pmod{2\pi}.
```

Here $f_p$ is pulse repetition frequency, $\Delta\phi_\pm$ is the pulse-to-pulse phase change of the corresponding focused complex echo, $x$ increases laterally and $z$ increases away from the array. With the signed Doppler convention $d_\pm=c f_p\Delta\phi_\pm/(4\pi f_c)$, the vector relations are

```math
v_x=\frac{d_- - d_+}{\sin\theta},
\qquad
v_z=-\frac{d_-+d_+}{1+\cos\theta}.
```

Wrapped phases and noisy directional estimates make this inverse uncertain. The requested result is the selected conditional velocity estimate described below; agreement with that estimate does not prove an independently calibrated mechanical speed.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing exactly:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `reconstruction` | `(1, 2, 251, 251)` | float32 or float64 | Finite real velocity in m/s. Axes are specimen, component, depth z, lateral x; component 0 is $v_x$, component 1 is $v_z$. Coordinates follow the supplied axes. Both components must be zero outside `observation_roi`. |

Every value of both components on the complete public disk support is evaluated using cosine similarity and RMS error divided by the corresponding reference dynamic range. There is no fitted gain, rotation, sign, alignment or crop. Evaluation also checks the exterior-zero convention and the weighted complex-phase residual against both receive views derived independently from the public RF observations.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`. All arrays have a leading specimen axis.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `rf` | `(1, 32, 128, 334)` | int16 | Real measured ADC counts, in arbitrary uncalibrated amplitude units. Axes are specimen, pulse ensemble, receive element, fast-time sample. |
| `observation_roi` | `(1, 251, 251)` | bool | Dimensionless known disk support on depth/lateral grid; exactly 31,404 pixels. This is the author's geometric support, not a mask selected from inferred velocities. |
| `x` | `(1, 251)` | float64 | Lateral output coordinates in metres, increasing along columns. |
| `z` | `(1, 251)` | float64 | Depth output coordinates in metres, increasing along rows and away from the array. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Medium sound speed | 1480 | m/s | Known homogeneous propagation speed used for travel times and phase conversion. |
| Carrier frequency | 5,000,000 | Hz | Physical ultrasound carrier, retained when RF is undersampled. |
| Saved RF sampling rate | 6,666,666.666666667 | samples/s | Four thirds of carrier frequency; the stored measurements are already retrospectively downsampled. |
| Pulse repetition frequency | 10,000 | Hz | Separation between the 32 transmitted pulses is 0.0001 s. |
| First fast-time sample | 0.00000995 | s | Time offset relative to transmission; subsequent samples are separated by the inverse saved RF rate. |
| Array elements | 128 | — | Element order matches the RF element axis. |
| Element pitch | 0.000298 | m | Element centre $x_j=(j-63.5)$ times pitch, for zero-based element index $j$. |
| Element width | 0.000262 | m | Known width for receive directivity. |
| Array depth and curvature | z = 0; infinite radius | m | Rectilinear array. |
| Transmit delays | All zero | s | Normal unsteered plane-wave transmission across all elements. |
| Fractional bandwidth | 15 | percent | Nominal full band width relative to carrier in the distributed author demodulation/directivity model. |
| Receive directions | +15 and −15 | degree | Two coherent receive apertures applied to the same complete RF observations. |
| Lateral grid | −0.0125 through +0.0125, step 0.0001 | m | Exact floating-point coordinates are supplied in `x`. |
| Depth grid | 0.0100 through 0.0350, step 0.0001 | m | Exact floating-point coordinates are supplied in `z`. |
| Disk geometry | Centre x = 0, z = 0.023; radius = 0.010 | m | Supplied boolean support preserves the author's strict-circle floating-point boundary convention. |
| Phase periodicity | 2π | rad | Each signed directional velocity wraps with period 1.48 m/s; its principal interval is ±0.74 m/s. |
| Numerical observation uncertainty | Not supplied | — | Speckle, decorrelation and finite-packet effects remain in the measured ensemble; no calibrated noise variance is asserted. |

The example's nominal disk diameter and pulse ensemble differ from the historical paper experiment. The acquisition above and the actual RF file define this task. A sampling rate below twice the carrier need not destroy a narrow bandpass signal, but its alias bands must remain disjoint. The physical carrier in phase conversion must not be replaced by its apparent sampled alias. The distributed bandwidth setting describes the selected model/processing band, rather than asserting that every measured RF sample belongs to an ideal strictly bandlimited signal.

## Method Hints

Use complex demodulation, coherent delay-and-sum reception and ensemble autocorrelation to obtain the two signed directional estimates. The selected author recipe uses all 31 adjacent pulse pairs and an 11 by 11 symmetric Hamming spatial average with replicated boundaries. Its receive f-number, selected from known element directivity, is 3.365971459145058. For receive angle $\beta$, include element $j$ when $|x-x_j-z\tan\beta|\leq z/(2F\cos\beta)$, where $F$ is that f-number.

The author virtual-array convention reads the transmit distance by a parabolic interpolation. Let $j$ be the nearest element to $x$, let $a_i=|x-x_i|$ and $b_i=\sqrt{(x-x_i)^2+z^2}$ for neighbouring elements $i$. The selected travel time and transmit distance are

```math
T=\frac{b_{j-1}+b_{j+1}}{2}
-\frac{(b_{j+1}-b_{j-1})(a_{j+1}-a_{j-1})}
{2a_{j-1}-4a_j+2a_{j+1}},
\qquad \tau_i=\frac{T+b_i}{c}.
```

For this supplied grid the nearest element is always interior to the array. Use the earlier element on a distance tie. Linear interpolation at $\tau_i$ uses the saved fast-time origin, and coherent complex reception rotates by $\exp(2\pi\mathrm{i} f_c\tau_i)$, where $\mathrm{i}$ is the imaginary unit. This is the selected discretization of normal transmission, rather than replacing $T$ by an ideal exact $z$. The author demodulation uses the physical carrier, a fifth-order zero-phase Butterworth low-pass with normalized cutoff $0.15f_c/f_s$ and fifteen samples of odd-reflected padding, where $f_s$ is the saved RF rate.

The selected field uses robust weighted penalized least squares with a DCT diagonalization of the repeating-boundary second-difference operator. For either velocity component $y$, an inner update has the form

```math
u_{k+1}=r\,\mathcal C^{-1}
\left[\frac{\mathcal C\{W(y-u_k)+u_k\}}
{1+s\Lambda^2}\right]+(1-r)u_k.
```

Here $u_k$ is the current field, $\mathcal C$ and $\mathcal C^{-1}$ are orthonormal two-dimensional DCT and inverse DCT, $W$ is the pointwise weight, and $\Lambda$ contains the eigenvalues of the negative repeating-boundary grid Laplacian. The selected smoothing scale is $s=10^4$ on the given pixel grid and the missing-data relaxation is $r=1.75$. Exterior observations have zero weight. Initialize missing values by each component's measured-support mean and retain the lowest ten percent of DCT frequencies per axis, as in the toolbox-free author path. Three vector-bisquare reweighting passes share the residual scale across components; each inner solve stops below relative field change $10^{-3}$ or at 100 updates. These conventions define a reproducible conditional method result, rather than a measured true rotation rate. Comparing unsmoothed and regularized fields helps reveal the effect of phase noise and the spatial prior.

## References

Madiena et al., “Color and Vector Flow Imaging in Parallel Ultrasound With Sub-Nyquist Sampling,” *IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control* (2018), [published paper](https://doi.org/10.1109/TUFFC.2018.2817885).

Garcia, “Robust smoothing of gridded data in one and higher dimensions with missing values,” *Computational Statistics & Data Analysis* (2010), [paper](https://doi.org/10.1016/j.csda.2009.09.020), with its [convergence-proof corrigendum](https://doi.org/10.1016/j.csda.2011.12.001); and “A fast all-in-one method for automated post-processing of PIV data,” *Experiments in Fluids* (2011), [paper](https://doi.org/10.1007/s00348-010-0985-y).

The real `PWI_disk.mat` acquisition is credited to the authors of the cited vector-Doppler study. Software and data rights are distinguished in the accompanying notice; no separate redistribution license is stated for the real observations.
