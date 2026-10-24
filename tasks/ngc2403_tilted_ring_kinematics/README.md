# NGC 2403 Tilted-Ring Kinematics

> Recover radial rotation, intrinsic velocity dispersion and disk orientation from a neutral-hydrogen spectral image cube whose telescope beam mixes spatial and velocity structure.
> Domain: astronomy | Keywords: spectral imaging, galaxy kinematics, inverse modeling | Difficulty: Hard

## Background

Neutral hydrogen emits a radio spectral line whose Doppler shift measures motion along the line of sight. A rotating gas disk therefore appears as a sequence of spatial images at different velocities. Its velocity structure helps determine the gravitational potential and the distribution of matter in a galaxy.

A finite telescope beam blends nearby regions. Unresolved velocity gradients broaden spectral lines, which makes circular rotation and random gas motion difficult to separate. Recovering the disk kinematics requires accounting for the spatial resolution and the velocity distribution together.

## Problem Description

The observation is the complete supplied VLA neutral-hydrogen cube of NGC 2403. Recover circular rotation speed, intrinsic one-dimensional velocity dispersion and receding-side position angle on all 41 radial knots. The inclination, center, systemic velocity and vertical scale height are specified conditional model assumptions; they are held fixed in this inverse problem.

For a cloud at disk radius $R$, azimuth $\theta$ and height $z$, the projected coordinates and line-of-sight velocity obey

```math
x_d=R\cos\theta,\qquad y_d=R\sin\theta\cos i-z\sin i,
```

```math
x_p=x_0+\frac{x_d\sin\phi-y_d\cos\phi}{\delta_x},\qquad
y_p=y_0+\frac{x_d\cos\phi+y_d\sin\phi}{\delta_y},\qquad
v=v_s+V(R)\cos\theta\sin i+\epsilon.
```

Here $i$ is inclination, $\phi(R)$ is position angle, $(x_0,y_0)$ is the zero-based pixel center, $(\delta_x,\delta_y)$ are signed angular pixel steps, and $v_s$ is systemic velocity. Radius, height and angular pixel steps must use the same angular unit. The Gaussian deviate $\epsilon$ has variance $s(R)^2+s_I^2$, where $s$ is intrinsic dispersion and $s_I$ is the specified instrumental standard deviation. Cloud azimuths are uniform, radius is uniform in annular area, and the vertical density is $\operatorname{sech}^2(z/z_0)/(2z_0)$ with scale height $z_0$. Velocities are assigned to nearest spectral channels; spatial emission is convolved with the normalized elliptical Gaussian beam. The nonnegative annular flux is a nuisance quantity inferred from the observed intensities, rather than an independently known gas-density map.

Save **`output/reconstruction.npz`** relative to the task workspace. It must contain all three keys:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `rotation` | `(1, 41)` | float32 or float64 | Circular rotation speed, km/s; increasing radial knot order. |
| `dispersion` | `(1, 41)` | float32 or float64 | Intrinsic one-dimensional velocity standard deviation, km/s; same knots. |
| `position_angle` | `(1, 41)` | float32 or float64 | Receding-side position angle, degrees anticlockwise from sky north; same knots. |

Use physical values and include every knot. No rescaling, spatial cropping or fitted registration is applied during evaluation. Rotation and dispersion are assessed separately using RMS error divided by the corresponding profile's dynamic range. Angle differences are wrapped into $[-180,180)$ degrees and assessed by their RMS, permitting equivalent whole turns. A full-cube forward residual in Jy/beam additionally checks observation consistency. This real observation has no independently measured kinematic truth: agreement assesses the stated conditional standard-method estimate, with moderate uncertainty from the stochastic integral and model assumptions, rather than establishing the galaxy's exact physical fields.

## Data Description

The NPZ contains the author's existing, reduced real observation; it is not raw telescope voltage data. Its axes retain all native channels and spatial pixels. All stored intensities are finite, including negative noise samples.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `spectral_cube` | `(1, 62, 150, 150)` | float32 | Intensity, Jy/beam; axes batch, velocity channel, image y, image x. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Spectral grid | $289.610969-5.15286k$, $k=0,\ldots,61$ | km/s | Native heliocentric velocity-channel centers; decreasing velocity with channel index. |
| Image x step | −16.00000038756 | arcsec/pixel | Negative right-ascension direction; signed step retained in the projected model. |
| Image y step | +16.00000038756 | arcsec/pixel | Positive declination direction. |
| Spatial WCS reference | pixel (76,76), RA 114.22708333, Dec 65.588888889 | FITS one-based pixel, degrees | Native SIN image WCS; the model center below is a separate supplied assumption. |
| Beam FWHM major/minor | 59.3287870284 / 58.6053617292 | arcsec | Elliptical Gaussian point-spread function. |
| Beam position angle | −79.446372986 | degree | Beam major-axis sky position angle. |
| Spectral instrumental FWHM | 5.15286 | km/s | Selected one-channel Gaussian response approximation; $s_I=5.15286/(2\sqrt{2\ln2})$ km/s. Channel binning is an additional discretization. |
| Rest frequency | 1420405750 | Hz | Neutral-hydrogen line header value. |
| Observation date | 1999-01-02 | date | VLA observation date in the supplied header. |
| Disk center | (77,77) | zero-based pixel | Fixed author-supplied model center, x then y; do not replace with the WCS reference pixel. |
| Systemic velocity | 132.8 | km/s | Fixed conditional disk parameter. |
| Inclination | 60 | degree | Fixed conditional disk parameter. |
| Vertical scale height | 10 | arcsec | Fixed conditional disk parameter; normalized sech² vertical profile. |
| Radial output knots | $15+30j$, $j=0,\ldots,40$ | arcsec | All 41 knots, including the last knot at 1215 arcsec. |
| Distance | 3.2 | Mpc | Supplied distance for physical-radius interpretation. |
| Spatial boundary | zero outside the cube for beam convolution | Jy/beam | Finite native field of view; retain all input and output samples. |

## Method Hints

Fit beam-convolved three-dimensional tilted-ring models rather than interpreting spectral width entirely as intrinsic gas motion. A two-dimensional intensity-weighted velocity-field fit can initialize the geometry, while a derivative-free simplex search accommodates the discretely sampled cloud integral. The selected method fits each annulus jointly in rotation, dispersion and position angle, regularizes the low-scatter angle profile by its median, then refits the two kinematic profiles with that angle fixed.

The conditional fit uses azimuthal flux normalization and a cosine-squared weighted absolute residual with a blank-pixel penalty:

```math
\mu=\frac{\sum_{\mathcal A} b d}{\sum_{\mathcal A} m},\qquad
J=\left(1+\frac{N_b}{N_c}\right)
\frac{\sum_{\mathcal C}\cos^2\theta\,|\mu m-d_+|}{N_c-N_b}.
```

Here $d$ and $m$ are observed and unnormalized model voxel intensities, $b$ is an observation-only emission mask, $\mathcal A$ is the projected annulus over all channels, and $\theta$ is its disk azimuth. The counted set $\mathcal C$ contains voxels where $b$ is true or the normalized model is nonzero, $N_c$ is its size, and $N_b$ counts its mask-false voxels. The quantity $d_+$ is the observed value when positive and the observation-derived noise standard deviation otherwise. An empty model has normalization gain zero. The emission mask can be obtained by robust intensity thresholding and connected-source growth; it is not an input kinematic calibration. Normalizing the intensity scale during optimization is compatible with returning the physical kinematic units specified above.

The selected observation-only support convention uses the median and median absolute deviation of finite nonzero intensities, with noise standard deviation equal to that deviation divided by 0.6744888. Its fixed source-selection assumptions are four-noise-standard-deviation detection, 2.5-standard-deviation growth, adjacency within one sky pixel in each axis and two channels, and retained components with at least 16 distinct sky pixels, 32 voxels and two consecutive occupied channels; these are conditional quality-control assumptions, not independently measured instrument parameters.

## References

- Di Teodoro, E. M. and Fraternali, F. (2015), *3D BAROLO: a new 3D algorithm to derive rotation curves of galaxies*, MNRAS 451, 3021–3033. [Paper](https://doi.org/10.1093/mnras/stv1213), [arXiv version 1](https://arxiv.org/abs/1505.07834v1).
- Author-supplied NGC 2403 example and current azimuthal-normalization configuration. This task retains the complete example acquisition and explicitly qualifies a bounds correction in the initialization copy; it does not claim an unchanged executable reproduction of the 2015 paper's numerical results.
