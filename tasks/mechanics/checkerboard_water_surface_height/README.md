# Checkerboard Water Surface Height

> Reconstruct a spatial liquid-surface height field from a refracted checkerboard image whose apparent displacements encode surface slopes.
> Domain: mechanics | Keywords: free-surface imaging, optical refraction, phase demodulation | Difficulty: Hard

## Background

A transparent liquid surface bends light when its local normal changes. A camera viewing a patterned backdrop through that surface therefore observes a displaced pattern. This optical effect allows noncontact imaging of small water waves without adding particles or attaching probes to the interface.

Surface profilometry helps investigate wave propagation, fluid disturbances and interactions between waves and immersed structures. Turning apparent optical displacements into height requires accounting for the optical layers and integrating a noisy two-component slope field. Occlusion, strong curvature and finite camera coverage limit which parts of the surface can be recovered reliably.

## Problem Description

The supplied images are real experimental observations of water waves and a separate flat-surface checkerboard reference. The fixed field of view contains transparent liquid and excludes an opaque probe visible elsewhere in the original recording. Let h(x,y) be surface elevation, I0 the flat-surface reference image, I the distorted image, u the apparent two-dimensional pattern displacement, and H the effective optical distance. The small-slope acquisition model is

```math
I(\boldsymbol{x})\simeq I_0(\boldsymbol{x}-\boldsymbol{u}(\boldsymbol{x})),
\qquad \boldsymbol{u}=-H\nabla h+\boldsymbol{u}_0.
```

The constant vector u0 represents mean deflection, which is excluded from the periodic height component requested here. The point x=(x,y) is a physical position in metres. The supplied reference also contains spatial illumination and nonideal pattern contrast; raw intensity agreement is approximate. For a carrier wave vector q=(qx,qy), its relative pattern phase is phi=q dot u, modulo 2*pi. Two nonparallel carriers constrain both slope components. Finite aperture, phase wrapping, nonintegrable noise and boundary mismatch make recovery difficult.

For layer thicknesses dl and refractive indices nl, numbered from the backdrop toward the camera, the supplied reference convention is

```math
H=\left(1-\frac{n_c}{n_w}\right)n_w\sum_{l=1}^{3}\frac{d_l}{n_l}.
```

Here nw is water index, nc is camera-side air index, and the three summed layers are backdrop-side air, glass and water. Their values are listed below.

Save `output/reconstruction.npz` containing `reconstruction`, real float32 or float64, shape `(1,641,641)`, axis order `(batch,y,x)`, in metres. The field is the zero-spatial-mean periodic elevation component relative to its mean surface and mean tilt; its sample pitch is 0.00031747269890795626 m on both axes. Array x increases with column and y increases with row. No amplitude rescaling, translation, crop, or output mean subtraction is performed during reference scoring. Evaluation uses cosine NCC, RMS error divided by reference dynamic range, and wrapped carrier-phase RMS mismatch in radians. The observation check allows one spatially constant circular phase offset per carrier, corresponding only to the discarded mean slope.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reference | (1,641,641) | float32 | Flat-surface grayscale calibration image, digital counts [DN] |
| measurements | (1,641,641) | float32 | Refracted grayscale wave image, digital counts [DN] |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Checkerboard square side | 0.0022 | m | Pattern calibration length |
| Backdrop-side air thickness | 0.057 | m | First optical layer |
| Backdrop-side air index | 1.0003 | dimensionless | First optical refractive index |
| Glass thickness | 0.012 | m | Second optical layer |
| Glass index | 1.48899 | dimensionless | Second optical refractive index |
| Water depth | 0.043 | m | Third optical layer |
| Water index | 1.34 | dimensionless | Water refractive index |
| Camera-side air distance | 0.8 | m | Camera-side layer, excluded from the backdrop-to-surface sum |
| Camera-side air index | 1.0003 | dimensionless | Camera-side refractive index |
| Effective optical distance H | 0.0329956084466042 | m | Small-slope displacement-to-height-gradient scale |
| Calibrated sample spacing | 0.00031747269890795626 | m/pixel | Reference carrier calibration in the selected field of view |
| First carrier row | 276 | FFT pixel | Row coordinate in the shifted reference spectrum |
| First carrier column | 369 | FFT pixel | Column coordinate in the shifted reference spectrum |
| Second carrier row | 271 | FFT pixel | Row coordinate in the shifted reference spectrum |
| Second carrier column | 277 | FFT pixel | Column coordinate in the shifted reference spectrum |
| Original observation crop | rows 256:897, columns 80:721 | original pixels | Zero-based, stop-exclusive coordinates in the 1024-square recording |
| Boundary | periodic | dimensionless | Finite-domain height and Fourier derivatives |
| Height gauge | zero mean elevation and zero mean slope component | m and dimensionless | Constant elevation and affine tilt are excluded |
| Camera variance | unspecified | DN squared | No calibrated stochastic noise law is supplied |

The nonparallel carrier calibration is obtained only from the separate flat-surface reference. The observations are reused experimental images; no new synthetic observations are generated.

## Method Hints

Fourier carrier demodulation isolates two checkerboard harmonics and compares their complex phases with the reference. Unwrapping phase before solving the two-component displacement relation avoids interpreting full phase cycles as physical slope discontinuities. A reliability-guided path can handle noisy two-dimensional phase fields.

Integrate the inferred slopes by the periodic least-squares solution in Fourier space. If gx and gy are the slopes and kx,ky are angular spatial frequencies, the nonzero-frequency height coefficient is

```math
\widehat h=\frac{-i k_x\widehat g_x-i k_y\widehat g_y}{k_x^2+k_y^2}.
```

Set the zero-frequency coefficient to zero to impose the elevation gauge. The hat denotes the two-dimensional Fourier transform and i is the imaginary unit. This integration projects a noisy slope field onto its integrable component; it cannot reproduce arbitrary curl or nonperiodic boundary trends.

## References

- Wildeman, Real-time quantitative Schlieren imaging by fast Fourier demodulation of a checkered backdrop, Experiments in Fluids59,97 (2018). [Method paper](https://doi.org/10.1007/s00348-018-2553-9).
- Moisy, Rabaud and Salsac, A synthetic Schlieren method for the measurement of the topography of a liquid interface, Experiments in Fluids46,1021-1036 (2009). [Optical model](https://doi.org/10.1007/s00348-008-0608-z).
- Herraez et al., Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path, Applied Optics41,7437-7444 (2002). [Unwrapping paper](https://doi.org/10.1364/AO.41.007437).

The bound experimental observations and selected modern reference implementation are separately supplied by Juan Giordano's MIT-licensed 2025 research archive; they are not presented as the original method authors' code or observations.
