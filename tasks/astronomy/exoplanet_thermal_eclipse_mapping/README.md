# Exoplanet Thermal Eclipse Mapping

> Recover a two-dimensional dayside thermal brightness map from an unresolved secondary-eclipse light curve whose occulting stellar limb only weakly resolves spatial structure.
> Domain: astronomy | Keywords: occultation imaging, thermal mapping, Bayesian inversion | Difficulty: Hard

## Background

An exoplanet passing behind its host star is progressively hidden by the stellar limb. The changing visible planetary flux encodes where the planet emits strongly, even though no telescope pixel resolves its surface. Thermal brightness maps help investigate atmospheric circulation and the redistribution of absorbed stellar energy.

The observations mix broad surface regions into one brightness value at each time. Latitude and longitude patterns can produce similar curves, while finite sampling and photon or instrument uncertainty obscure weak spatial modes. Low-order spatial representations and explicit uncertainty treatment are therefore central to interpreting an eclipse map.

## Problem Description

The supplied data are an existing author simulation at3.31micrometres for an HD189733b-like system. They are noiseless normalized flux samples accompanied by simulated instrument uncertainties; they are not measured JWST observations. Recover the conditional dayside brightness field supported by the three dominant eclipse eigencurves described below. No independent spatial truth is supplied or assumed.

```math
f(t)=b+\sum_{j=1}^{400}a_j(t)\,I(\phi_j(t),\lambda_j(t))+e(t).
```

Here f is observed system flux divided by the adopted stellar flux, b is a nonnegative constant stellar offset, I is real surface brightness in stellar-flux units per unit planet-normalized projected area, a is the visible projected area of tile j in units of squared planet radius, phi and lambda are latitude and longitude in radians, and e is residual flux error with supplied standard deviation sigma. The planet has unit projected radius. Its disk is divided into20annuli, annulus i spanning i/20 to(i+1)/20 with2i+1equal-angle sectors for i=0..19. Each sector has area pi/400; its brightness is evaluated at its radial/angular midpoint, except that the central tile uses the disk center. The star is an opaque circular occultor. There is no planetary limb darkening.

For the circular orbit, with phase angle q=2*pi*(t-t0)/P, the stellar-center position in planet-radius units is (R*sin(q),-R*cos(q)*cos(i),R*cos(q)*sin(i)), where R=(a/Rstar)/(Rp/Rstar), i is inclination, P period and t0 primary-transit time. Eclipse occurs for the negative line-of-sight coordinate. Correct occultor time by dt=aAbs*sin(i)*(1-cos(q))/c; aAbs is in metres and c is light speed. Rotate the surface at the uncorrected time, as prescribed by the selected acquisition approximation. Exposure times are represented by their supplied centers, without additional temporal averaging. Surface coordinates use the substellar point as latitude0,longitude0, east-positive longitude and north-positive latitude.

For an uncorrected orbit position (X,Y,Z), define observer angles L=mod(pi+q,2*pi) and F=atan(Y/Z), using the one-argument arctangent convention. At a tile's projected-disk midpoint (mx,my), let h=sqrt(1−mx²−my²). Its surface latitude is asin(h*sin(F)+my*cos(F)); its longitude is −L−atan2(mx,h*cos(F)−my*sin(F)). The annular sector areas are occulted by the stellar circle at the light-time-corrected (X,Y). These conventions define the finite-inclination orientation completely.

Save `output/reconstruction.npz` with key `reconstruction`, real float64 shape `(1,32,64)`, axes `(batch,latitude,longitude)`, sampled at the supplied latitude/longitude grids spanning the dayside from−pi/2throughpi/2. Values are stellar-flux-normalized surface brightness per unit projected area measured in squared planet radii; they are not kelvin or an SI radiance. Preserve absolute scale and coordinate orientation. The requested map is the posterior mean under the declared three-mode model and Gaussian likelihood. Do not impose extra pixelwise positivity clipping: only uniform-brightness amplitude and stellar offset have nonnegative priors.

Evaluation uses uncentered cosine similarity and RMS error divided by the reference-map dynamic range, with no map alignment, rescaling, crop or offset. It also checks uncertainty-normalized observation RMS after projecting the submitted map onto the declared degree-two harmonic space and fitting only the physical constant stellar-offset nuisance parameter. This nuisance fit does not shift or rescale the submitted map. All metrics must pass. Reference agreement measures reproducibility of the conditional inversion, not recovery of a known true atmosphere.

## Data Description

All visible arrays are float64 and batch-first. Load `data/raw_data.npz` with `allow_pickle=False`.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| time | (1,126) | float64 | Exposure-center time relative to the observation origin, day |
| measurements | (1,126) | float64 | Normalized star-plus-planet flux, dimensionless flux ratio |
| uncertainty | (1,126) | float64 | Simulated one-sigma instrument flux uncertainty, dimensionless ratio |
| wavelength | (1,1) | float64 | Representative thermal wavelength, micrometre |
| latitude | (1,32,64) | float64 | Requested map latitude, radians, increasing northward |
| longitude | (1,32,64) | float64 | Requested map longitude, radians, increasing eastward |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Orbital period P |2.21857567 | day | Circular synchronous orbit |
| Primary transit t0 |−2.21857/2 | day | Time convention used by the supplied model |
| aAbs |0.0313 | AU | Absolute semimajor axis for light-time correction |
| Astronomical unit |1.4960e11 | m | Selected source conversion |
| Light speed c |299792458 | m/s | Light-time correction |
| a/Rstar |8.863 | dimensionless | Orbital radius in stellar radii |
| Rp/Rstar |0.155313 | dimensionless | Planet-to-star radius ratio |
| Inclination i |85.71 | degree | Fixed orbital inclination |
| Eccentricity |0 | dimensionless | Circular model |
| Argument of periastron |90 | degree | Circular-orbit source convention |
| Surface rotation |Synchronous; substellar latitude/longitude0 | radian | Fixed planet coordinate origin |
| Disk discretization |20rings,400equal-area sectors | count | Constant brightness per sector |
| Limb-darkening coefficients |0,0 | dimensionless | Uniform directional emission per surface element |
| Time sampling |136.77324 | second | Exposure-center spacing; no further averaging |
| Flux uncertainty |3.6395766103691654e-5 | ratio | Constant simulated error, not an added noise realization |
| Wavelength |3.31 | micrometre | Existing single-band author simulation |
| Map extent |latitude and longitude−pi/2..pi/2 | radian | Inclusive rectangular coordinate sampling |
| Map boundary |Degree-two spherical extension | dimensionless | Global harmonic model evaluated on requested dayside |

## Method Hints

Use real spherical harmonics through degree2 and perform PCA on the eclipse light curves of paired positive/negative nonuniform harmonics, centered in time. The normalized harmonic vector H, with s=sin(phi),h=cos(phi), is

```math
H=(1,\sqrt3h\sin\lambda,\sqrt3s,\sqrt3h\cos\lambda,
\tfrac{\sqrt{15}}2h^2\sin2\lambda,
\sqrt{15}sh\sin\lambda,
\tfrac{\sqrt5}2(3s^2-1),
\sqrt{15}sh\cos\lambda,
\tfrac{\sqrt{15}}2h^2\cos2\lambda).
```

For each of the eight nonuniform harmonics generate curves for both signs, including the known unit stellar offset, then subtract that offset and each curve's temporal mean. Diagonalize their covariance using the sample normalization1/(126−1), rank eigenvalues descending and retain the first three components. Keep the uncentered uniform harmonic curve separately. Fit its amplitude c0, a constant stellar offset b and the three eigencurve amplitudes under independent Gaussian flux errors. The target uses flat priors with c0>=0,b>=0 and unrestricted eigencurve amplitudes. Transform posterior coefficients back to the spherical brightness field. Eigenvector signs are arbitrary if transformed consistently. Ensemble stretch sampling is suitable; an independently justified equivalent posterior calculation may also recover the requested mean.

## References

- Mansfield et al. (2020), *Eigenspectra: A Framework for Identifying Spectra from3D Eclipse Mapping*. [Paper](https://doi.org/10.1093/mnras/staa3179), [author preprint](https://arxiv.org/abs/2010.02197).
- Rauscher, Suri and Cowan (2018), *A More Informative Map*. [Author preprint](https://arxiv.org/abs/1806.05700).
- Louden and Kreidberg (2018), *SPIDERMAN: an open-source code to model phase curves and secondary eclipses*. [Paper](https://doi.org/10.1093/mnras/sty558).
