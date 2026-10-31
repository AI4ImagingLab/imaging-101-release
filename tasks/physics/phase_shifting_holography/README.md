# Fresnel Lens Phase-Shifting Holography

> Recover a complex interferometric image of a Fresnel lens from four measured phase-shifted holograms despite background intensity, conjugate interference and reference-wave tilt.
> Domain: Physics | Keywords: wave imaging, optical interferometry, phase retrieval | Difficulty: Medium

## Background

Coherent optical imaging measures interference between light transmitted through
an object and a reference beam. Unlike an ordinary intensity image, interference
contains information about optical phase as well as amplitude. Phase can reveal
transparent structures whose absorption contrast is weak, and changes in optical
path length caused by surface relief or refractive index.

Digital holography replaces photographic recording and optical readout with
camera measurements and numerical wavefield recovery. It supports optical
component inspection, surface metrology, quantitative microscopy and monitoring
of changing transparent specimens. Accurate recovery requires separating the
object information from illumination background and the conjugate interference
term; a slight angle between the two beams adds a spatial phase carrier.

## Problem Description

Four experimentally acquired frames image the same Fresnel lens while the
reference phase advances by a quarter cycle. At each camera pixel the ideal
interference model is

```math
I_j(y,x)=|O(y,x)+R(y,x)\exp(i j\pi/2)|^2+\epsilon_j(y,x),\qquad j=0,1,2,3.
```

Here I denotes recorded intensity in digital numbers (DN), O is the object
wave, R is the tilted reference wave, i is the imaginary unit, and epsilon
includes noise, encoding error and experimental model mismatch. Indices x and y
increase along columns and rows. O and R have amplitude units sqrt(DN) under
this idealized intensity convention; their individual physical irradiance is
not calibrated. The reference is approximately a plane wave. The inverse
problem is to separate a complex object-reference cross-term from real
intensities up to the unidentifiable reference tilt. Absolute object height and an
absolute optical phase origin are not supplied by this acquisition.

Save `output/reconstruction.npz` relative to the task workspace with the key
`reconstruction`, a real `float32` or `float64` array of shape
`(1, 2, 1024, 1024)` and axis order `(batch, component, row, column)`.
Component 0 is the real part and component 1 the imaginary part of C, the
compensated object-reference cross-term in DN. Under the ideal model its
physical scale and camera-plane DC convention are

```math
C(y,x)=4R_d(y,x)\left[O(y,x)\overline{R(y,x)}
                       -\left\langle O\overline R\right\rangle\right],
\qquad
R_d=\exp\!\left[2\pi i\left(\frac{N/2-f_x}{N}(x-N/2)
                      +\frac{M/2-f_y}{M}(y-M/2)\right)\right].
```

The overbar is complex conjugation and the brackets denote a full spatial mean.
N and M are the column and row counts. The frequencies fx,fy parameterize an
arbitrary reference-tilt gauge. No independently calibrated reference-beam tilt
is available, so these frequencies cannot be uniquely separated from a linear
object phase. Any unit plane-wave compensation is permitted, including Rd=1.
Frequency slopes are interpreted modulo 2pi radians per pixel, in [-pi,pi).
Since both array half-sizes are integers, these aliases give identical sampled
phasors. The half-array origin fixes Rd(512,512)=1; no independent global phase
rotation is allowed. Preserve the factor four, the frame order and full complex
amplitude; do not crop, normalize or submit only magnitude or unwrapped phase.

Evaluation independently forms the mean-removed quadrature Q from the four
public frames, then checks that C equals Q times a unit linear phase on the
complete grid. The relative complex L2 error must be at most 1e-6. The
best adjacent cross-product phase slopes are used to verify the plane without
dividing by zero-amplitude pixels. A nonlinear spatial phase change or an
amplitude change fails even if its magnitude image looks similar. This tests
interferometric recovery up to the acquisition's actual ambiguity; it does not
assert independent lens-height accuracy or a uniquely calibrated reference tilt.

## Data Description

The observations are the released experimental Fresnel lens example associated
with the reference below. They are repository-distributed image exports, already
processed and lossy encoded upstream, rather than original raw sensor files.
The supplied arrays preserve all released grayscale pixels without new spatial
cropping, resizing, noise injection or intensity normalization. No independent
physical irradiance or object-plane magnification calibration is available.

| Key | Shape | Dtype | Description |
| --- | --- | --- | --- |
| holograms | (1, 4, 1024, 1024) | uint8 | Four quarter-cycle phase-shifted measured intensities in DN, ordered phase 0, pi/2, pi, 3pi/2; axes batch, phase, row, column. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
| --- | --- | --- | --- |
| Illumination wavelength | 632.8e-9 | m | Illumination value supplied with the experimental example. |
| Camera pitch x | 6.9e-6 | m/pixel | Sensor column spacing. |
| Camera pitch y | 6.9e-6 | m/pixel | Sensor row spacing. |
| Frame phase increments | pi/2 | rad | Nominal reference phase increase between adjacent frames, introduced using a liquid lens. |
| Image rows and columns | 1024 by 1024 | pixel | Released spatial extent; no additional padding. |
| Frame count | 4 | dimensionless | Same sample and nominal geometry. |
| Pixel coordinate origin | x = column - N/2; y = row - M/2 | pixel | Half-array-centered coordinates, not (N-1)/2. Rows increase downward. |
| Reference tilt | not independently calibrated | rad/pixel | Linear phase gauge is free modulo sampled frequency aliases. |
| Discrete transform | negative exponential, no forward normalization | dimensionless | Forward FFT uses a pre-shift and a post-shift; full finite camera window, no taper or padding. |
| Boundary convention | finite camera window; periodic DFT extension | dimensionless | Applies only to spectral carrier estimation; no spatial propagation. |
| Noise calibration | not supplied | DN | No calibrated likelihood or known Gaussian noise variance is assumed. |
| Physical scale | released 8-bit grayscale intensity | DN | Upstream image exports; no conversion to optical power or lens thickness. |

## Method Hints

Use quadrature phase-shifting interferometry to cancel the background and
conjugate image. A synthetic plane reference compensates the residual spatial
carrier. With Hj denoting each intensity frame after subtracting its spatial
mean, the quadrature combination Q is

```math
H_j=I_j-\langle I_j\rangle,\qquad
Q=H_0+iH_1-H_2-iH_3,\qquad C=R_d Q.
```

Here Q is the complex uncompensated cross-term in DN and Rd is the unit
linear-phase gauge defined above. The quadrature cancels intensity background
and conjugate interference. Carrier estimation may choose a convenient visual
reference plane, but the observations do not fix a unique reference tilt.

## References

Castaneda R, Trujillo C, Doblas A. *pyDHM: A Python library for applications in
digital holographic microscopy*. PLoS ONE 17(10), e0275818 (2022), published
version. https://doi.org/10.1371/journal.pone.0275818 . The slightly off-axis
experimental example uses a Fresnel lens and four phase-shifted measurements.

De Nicola S, Ferraro P, Finizio A, Pierattini G. *Wave front reconstruction of
Fresnel off-axis holograms with compensation of aberrations by means of
phase-shifting digital holography*. Optics and Lasers in Engineering 37(4),
331-340 (2002).
