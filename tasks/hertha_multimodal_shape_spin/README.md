# Hertha Multimodal Shape and Spin

> Recover a three-dimensional asteroid surface, spin, and image-registration quantities from photometry, adaptive-optics images, and occultation chords whose complementary constraints remain nonlinear and incomplete.
> Domain: astronomy | Keywords: multimodal imaging, shape reconstruction, inverse scattering | Difficulty: Hard

## Background

An unresolved rotating body changes its reflected brightness as illuminated surface facets move into and out of view. The variation contains information about shape and orientation, but different surfaces can produce similar disk-integrated signals. Long time spans help constrain the spin while leaving parts of the surface weakly determined.

Resolved optical images provide projected shape and brightness information. Atmospheric and instrumental blurring limit their detail, and registration and relative radiometry introduce additional unknowns. A stellar occultation adds a different geometric measurement: each observing station records the endpoints of a chord across the projected silhouette. Joint use of these measurements supports asteroid size, shape, rotational-state, and dynamical studies.

## Problem Description

Reconstruct Hertha from the complete supplied30 relative lightcurves923samples, two author-reduced adaptive-optics(AO) windows and their measured point-spread functions(PSFs), and18 detected occultation chords. The images are reduced astronomical observations, rather than raw detector exposures. The fixed `vertex_angles` and `facets` specify the ordering of a closed402vertex,800facet body-frame surface; they contain no measured surface radii.

For facet index $t$, let $A_t$ be its area, $\boldsymbol n_t$ its outward unit normal, and $\mu_t$ and $\mu_{0t}$ its normal projections onto the body-frame observer and Sun unit vectors. Only simultaneously visible and illuminated facets, including self-occlusion, contribute. The unnormalized disk-integrated brightness is

```math
L=\sum_{t\in\mathcal V} A_t\mu_t\mu_{0t}\left(\frac{1}{\mu_t+\mu_{0t}}+0.1\right).
```

Here $\mathcal V$ denotes this visible, illuminated set. Each relative lightcurve compares the measured brightness with $L$ divided by its mean over that curve. A fixed zero spin phase at the stated epoch determines the relation between the body axes and inertial ecliptic coordinates; latitude, longitude, and rotation period are unknown.

For colatitude $\beta$ (90 degrees minus latitude), longitude $\lambda$, and angular speed $\omega=48\pi/P$ radians/day for period $P$ in hours, the inertial-to-body rotation at supplied relative time $t$ in days is $M=F(\omega t)B$, with

```math
F(q)=\begin{bmatrix}\cos q&\sin q&0\\-\sin q&\cos q&0\\0&0&1\end{bmatrix},\qquad
B=\begin{bmatrix}
\cos\beta\cos\lambda&\cos\beta\sin\lambda&-\sin\beta\\
-\sin\lambda&\cos\lambda&0\\
\sin\beta\cos\lambda&\sin\beta\sin\lambda&\cos\beta
\end{bmatrix}.
```

Angles in these matrices are radians. Camera rows are the normalized observer vector $\boldsymbol z$, $\boldsymbol x=(\boldsymbol U\times\boldsymbol z)/\|\boldsymbol U\times\boldsymbol z\|$ for supplied up vector $\boldsymbol U$, and $\boldsymbol y=\boldsymbol z\times\boldsymbol x$. Body-to-camera projection uses these rows multiplied by $M^{\mathsf T}$.

For exposure index $k$, the normalized image Fourier observation $D_k(\boldsymbol f)$ obeys

```math
D_k(\boldsymbol f)=g_k P_k(\boldsymbol f)e^{2\pi\mathrm{i}\boldsymbol f\cdot\boldsymbol s_k}
\frac{\sum_{t\in\mathcal V_k} B_t\int_{\Pi_k(t)}e^{-2\pi\mathrm{i}\boldsymbol f\cdot\boldsymbol u}\,d\boldsymbol u}
{d_k^2\sum_{t\in\mathcal V_k}B_t\mu_tA_t}+\epsilon_k,
\qquad B_t=\mu_{0t}\left(\frac{1}{\mu_t+\mu_{0t}}+0.1\right).
```

The imaginary unit is $\mathrm{i}$, $\boldsymbol f$ is spatial frequency on the sky in cycles/arcsec, $\boldsymbol u$ is a sky-plane position in arcsec, and $\Pi_k(t)$ is the projected facet. The known conversion $d_k$ maps km to arcsec using the observer distance. The known PSF transform $P_k$ and image transform use a center-phase convention and normalization by their own zero-frequency values; the zero-frequency sample is discarded. The unknown positive gain is $g_k$, the Fourier offset is $\boldsymbol s_k$, and $\epsilon_k$ represents measurement and model discrepancies. Positive Fourier offsets use the displayed positive exponential and correspond to negative translation of the image-plane model.

The occultation forward model projects the rotating mesh into its supplied sky plane, adds a two-component translation, and intersects every measured chord line with the projected mesh. The outer entry and exit intersections predict the four measured endpoint coordinates. The rotation is evaluated at the mean of all supplied occultation endpoint times. Both endpoint coordinates of every detected chord remain part of the inverse problem. All individual chord-time offsets are fixed to zero.

Save **`output/reconstruction.npz`**, relative to the task workspace, containing every array below as real float32 or float64:

| Required key | Shape | Units and convention |
|---|---|---|
| `reconstruction` | `(1, 402, 3)` | km; ordered body-frame vertex XYZ, with the supplied closed triangulation. |
| `spin` | `(1, 3)` | Latitude(degrees), longitude(degrees modulo360), and positive rotation period(hours); right-handed inertial ecliptic coordinates, zero phase at the stated epoch. |
| `ao_offsets` | `(1, 2, 2)` | arcsec; exposure then Fourier offset x,y, with the positive-exponential convention above. |
| `ao_log_gains` | `(1, 2)` | Dimensionless natural logarithms of the positive gains. |
| `occultation_offset` | `(1, 1, 2)` | km; event then additive sky-plane x,y translation of the projected mesh. |

Evaluation checks all402vertices in their supplied order, all three spin quantities, both AO registrations and gains, and both occultation translation components. Each Cartesian coordinate has its own dynamic-range-normalized RMS error; spin differences use physical units and longitude wraps modulo360, while each nuisance block uses RMS error in its declared units. There is no fitted rotation, translation, scale, crop, or vertex permutation during evaluation. A complete forward sum of squares is also required. The comparison is to a validated conditional method estimate; it does not establish independent shape truth or the physical precision of the inferred spin. Independent numerical implementations show finite surface and nuisance-parameter variability.

## Data Description

All arrays are stored in `data/raw_data.npz`, with a leading batch axis of length1.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `ao_images` | `(1, 2, 150, 150)` | `float64` | Two author-reduced images; axes exposure,y,x; arbitrary instrumental intensity units. |
| `ao_psf` | `(1, 2, 150, 150)` | `float64` | Matched measured PSF windows; axes exposure,y,x; arbitrary instrumental intensity units. |
| `vertex_angles` | `(1, 402, 2)` | `float64` | Fixed representation grid, ordered colatitude and longitude; radians; independent of the unknown shape. |
| `facets` | `(1, 800, 3)` | `int32` | Closed outward triangulation; face,corner; zero-based vertex indices, dimensionless. |
| `ao_observer` | `(1, 2, 3)` | `float64` | Asteroid-to-observer unit vectors in inertial ecliptic XYZ; dimensionless. |
| `ao_sun` | `(1, 2, 3)` | `float64` | Asteroid-to-Sun unit vectors in inertial ecliptic XYZ; dimensionless. |
| `ao_up` | `(1, 2, 3)` | `float64` | Supplied equatorial camera-up directions expressed in inertial ecliptic XYZ; dimensionless. |
| `ao_time` | `(1, 2, 1)` | `float64` | Light-travel-corrected times relative to the given epoch; days. |
| `ao_distance` | `(1, 2, 1)` | `float64` | Observer distance for each exposure; astronomical units (AU). |
| `lc_brightness` | `(1, 923, 1)` | `float64` | All923 measured mean-normalized photometric brightness values; dimensionless. |
| `lc_time` | `(1, 923, 1)` | `float64` | Source-provided photometry JD minus the epoch; no extra light-time correction in the selected reader; days. |
| `lc_observer` | `(1, 923, 3)` | `float64` | Per-sample asteroid-to-observer unit vectors in inertial ecliptic XYZ; dimensionless. |
| `lc_sun` | `(1, 923, 3)` | `float64` | Per-sample asteroid-to-Sun unit vectors in inertial ecliptic XYZ; dimensionless. |
| `lc_curve_index` | `(1, 923)` | `int32` | Photometric curve labels0 through29 in native sample order; dimensionless. |
| `lc_relative` | `(1, 30)` | `int32` | Per-curve relative-photometry flag;1 denotes mean normalization; dimensionless. |
| `oc_observer` | `(1, 1, 3)` | `float64` | Occultation asteroid-to-observer unit vector in inertial ecliptic XYZ; dimensionless. |
| `oc_velocity` | `(1, 1, 3)` | `float64` | Author asteroid velocity in inertial ecliptic XYZ; km/s; chord offsets are fixed to zero in this conditional model. |
| `oc_up` | `(1, 1, 3)` | `float64` | Occultation sky-plane up direction expressed in inertial ecliptic XYZ; dimensionless. |
| `oc_time` | `(1, 18, 2)` | `float64` | Disappearance/appearance times relative to epoch, already light-travel corrected; days. |
| `oc_chords` | `(1, 18, 4)` | `float64` | All18 centered measured endpoints, columns entry-x,entry-y,exit-x,exit-y; km in the supplied equatorial sky plane. |
| `oc_distance` | `(1, 1, 1)` | `float64` | Occultation observer distance used for the supplied light-travel correction; km. |
| `oc_chord_type` | `(1, 18, 1)` | `int32` | Original positive detection flags; every supplied chord is a detection in this model; dimensionless. |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Target | Hertha, asteroid135 | — | The number135 identifies the asteroid, not the number of lightcurves. |
| Photometry |30curves,923samples | count | Every supplied photometric sample is retained. |
| AO acquisition |2windows of150by150 | pixel | Native author windows starting at FITS one-based x=y=51; array axes y,x. |
| Pixel pitch x,y |0.009942,0.009942 | arcsec/pixel | Same pitch for AO and matched PSF windows. |
| Image origin |Zero-based pixel(75,75) | pixel | Center phase of the even150by150 transforms. |
| Native AO reception dates |2454729.060938,2456285.770312 | JD | Supplied author exposure dates; `ao_time` already includes light-travel correction. |
| Common epoch |2443846.0 | JD | Zero phase and origin of all supplied relative times. |
| Astronomical unit |149597871.0 | km | Native angular-scale conversion constant. |
| Speed of light |299792.458 | km/s | Native AO/occultation time-correction constant; those corrected times are supplied. |
| Surface convention |Closed outward piecewise-flat triangulation | — | Fixed vertex order and zero-based facet corners are supplied as arrays. |
| Inertial axes |Right-handed ecliptic XYZ | — | Sun/observer/up vectors use this frame; image/occultation sky-plane orientation follows the supplied up vector. |
| Reflectance |Constant albedo, LS/Lambert coefficient0.1 | — | Selected normalized scattering model; no fitted albedo map. |
| Photometric calibration |Per-curve relative means | — | Unknown absolute radiometry is removed by mean normalization. |
| AO radiometry |Relative instrumental intensity | — | No absolute flux or physical albedo calibration is asserted. |
| Occultation support |18detected chords,36endpoints | count | Native coordinates centered by their common endpoint mean; every supplied endpoint is used. |
| Boundary |Closed mesh, silhouette outer intersections | — | No artificial image-plane boundary condition on the surface. |
| Noise model |No calibrated common variance supplied | — | Residual weighting is a conditional modeling choice, not measured sensor noise. |

The original descriptive occultation timing-error column is not used by the selected estimator and is not needed to define this input. The known geometry and per-modality time conventions preserve the original author preparation. Photometric times are the source-provided JD minus the epoch; no additional photometric light-time correction is applied or claimed. The task does not claim that the instrument blur, constant-reflectance assumption, or finite shape parameterization explains every observed detail.

## Method Hints

A joint regularized Gauss--Newton or Levenberg--Marquardt method can combine the three observation types while retaining analytic facet Fourier transforms, occultation intersection derivatives, and visibility. A real spherical-harmonic octantoid parameterization describes positive coordinate radii in units of1km using real orthonormal harmonics with the Condon--Shortley convention: with fixed colatitude $\theta$ and longitude $\phi$, harmonic fields $a,b,c$ give $x=e^a\sin\theta\cos\phi$, $y=e^{a+b}\sin\theta\sin\phi$, and $z=e^{a+c}\cos\theta$; harmonics through degree9 define the selected representation.

The selected conditional target minimizes $\|\boldsymbol r\|_2^2$, where residual multipliers are8 for relative photometry,2 for complex AO data,0.1 for occultation endpoints, and30,5,0.5,1 for convexity, octantoid complexity, dihedral, and center-of-mass residuals respectively; these are author-fixed modeling choices, rather than independently measured uncertainties. Convexity penalizes directed adjacent concavities relative to total area, octantoid complexity penalizes squared degree-weighted differences of the harmonic coordinate radii, dihedral roughness sums squared neighboring normal deviations, and the center-of-mass residual is its distance from the body-frame origin.

For trial silhouettes missing a detected chord, the selected conditional objective retains the author support-line sigmoid penalty rather than treating nonexistent intersections as measured endpoints. In the native kilometre coordinates, the signed support distance is negative when projected vertices lie on both sides of the chord line, using the smaller-magnitude extreme, and otherwise is the nearest nonzero absolute vertex distance. If that distance is $d$, set $s=1/(1+e^{-8d})$ (the fixed coefficient8 has units inverse km); for measured endpoints $(a_x,a_y)$ and $(b_x,b_y)$ the four pre-weight residuals are $[a_x(1+s),a_y,b_x,b_y]$. This is an optimization heuristic, not a physical endpoint prediction.

## References

- M. Viikinkoski, M. Kaasalainen, and J. Durech, “ADAM: a general method for using various data types in asteroid reconstruction,” Astronomy & Astrophysics576,A8(2015), [doi:10.1051/0004-6361/201425259](https://doi.org/10.1051/0004-6361/201425259).
- [Author manuscript, arXiv1501.05958v1](https://arxiv.org/abs/1501.05958v1). This task uses the accompanying author's Hertha observations and later implementation of the same multimodal method; selected discretization and numerical conventions are defined above.
