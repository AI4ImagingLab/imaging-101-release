# Metallic Paint Microfacet Slope Distribution

> Recover a finite-grid normalized effective surface-slope distribution and two roughness summaries from measured angular reflectance of metallic paint despite masking and a nonparametric scattering profile.
> Domain: chemistry & material science | Keywords: reflectance spectroscopy, inverse scattering, surface microstructure | Difficulty: Hard

## Background

Microscopic surface slopes determine how a material distributes reflected light across viewing directions. A polished surface concentrates reflection near the specular direction, while a broad distribution of facet orientations produces a wider highlight. Quantifying this distribution supports material appearance measurement, coating characterization, and predictive optical rendering.

Facets can hide and shadow one another. Consequently, the measured angular intensity does not directly equal the distribution of surface orientations. An effective microfacet model describes both the distribution and its associated visibility. For complex coatings, that model is an optical description; it need not equal a distribution measured by microscopy.

## Problem Description

The observation is the complete isotropic RGB reflectance table for a measured gold-metallic-paint sample. Recover its finite-grid radial two-dimensional slope-density samples and its Beckmann and GGX moment-based roughness parameters. This task uses the coincident illumination/viewing slice, with both directions pointing away from the surface. It does not ask for an angle-dependent Fresnel function or a fit to every entry of the full table.

Let $\theta$ be the polar direction angle, $s=\tan\theta$ the slope magnitude, and $P(s)$ the isotropic density per unit area in the dimensionless two-dimensional slope plane. An ideal continuous model would obey the following normalization and orientation relation; the released numerical representation instead uses the explicitly specified finite quadrature below:

```math
2\pi\int_0^\infty P(s)s\,ds=1,\qquad D(\theta)=P(\tan\theta)/\cos^4\theta.
```

For coincident directions, the difference angle is zero. A constant normal-incidence Fresnel factor $F_0$ multiplies the scattering profile:

```math
f(\theta,\theta)=\frac{F_0 D(\theta)G_1(\theta)}{4\cos^2\theta},\qquad
G_1(\theta)=\frac{1}{\int_0^\infty P(u)u A(\theta,\arctan u)\,du},
```

```math
A(\theta,\eta)=\int_0^{2\pi}\max(1,\tan\theta\tan\eta\cos\phi)\,d\phi.
```

Here $f$ is RGB luminance reflectance in inverse steradians, $G_1$ is the Smith monostatic visibility, $u$ is an integration slope, $\eta$ its angle, and $\phi$ the integration azimuth. The equations assume an isotropic effective single-scattering microfacet material with correlated visibility and no separately fitted diffuse component. $F_0$ is an unknown scale factored out of the distribution estimation. A distribution inferred under these assumptions is not independent microscopic truth.

Save `output/reconstruction.npz` with both arrays below. Use real float32 or float64 values, finite everywhere; no rescaling or alignment is permitted.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 90) | float32 or float64 | Nonnegative radial slope density, dimensionless inverse slope-area, batch then increasing square-root polar angle; samples at theta=(j/89)^2 pi/2 for j=0,...,89; last horizon sample is zero. |
| roughness | (1, 2) | float32 or float64 | Positive dimensionless slope scales, batch then [Beckmann alpha, GGX alpha]. |

Between tabulated density values, use linear interpolation in $x=\sqrt{2\arctan(s)/\pi}$. Define the numerical moment convention using 128 left endpoints: $t_k=(k/128)\sqrt{\pi/2}$, $\eta_k=t_k^2$, $s_k=\tan\eta_k$, and $\Delta t=\sqrt{\pi/2}/128$, where $k=0,\ldots,127$:

```math
w_k=2\Delta t\,t_k P(s_k)s_k/\cos^2\eta_k,\quad
M=2\pi\sum_k w_k,\quad
\alpha_{\rm B}=\sqrt{2\pi\sum_k s_k^2w_k},\quad
\alpha_{\rm G}=4\sum_k s_kw_k.
```

The required finite-grid density has $M=1$ under this 128-point convention. Its largest integration angle is $(127/128)^2\pi/2$; no infinite-slope extrapolation is part of the output contract. Linear interpolation to the horizon endpoint is not integrable as a continuous slope density at infinity: a positive penultimate sample yields a tail proportional to $1/s$. Thus the two roughness numbers are finite-quadrature summaries, not convergent full-domain moments. Do not infer continuous normalization or asymptotic roughness from them. Evaluation checks density cosine similarity (no mean subtraction), RMS density error divided by reference dynamic range, absolute roughness errors, normalization and moment consistency. It also checks the relative residual of the positive backscatter integral eigenproblem. Reference agreement measures reproduction of an established method, not microscopic accuracy.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| brdf | (1, 3, 90, 90, 180) | float64 | Full measured MERL encoded reflectance; batch, RGB, half-angle, difference-angle, difference-azimuth. Encoded units; multiply channels by (1,1.15,1.66)/1500 to obtain sr^-1. Negative entries denote invalid measurements, not negative physical reflectance. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | gold-metallic-paint, MERL material library | — | Measured coated spherical sample; individual preparation date unavailable. |
| Acquisition class | Author-processed real measurements | — | Camera measurements calibrated, trimmed and averaged into angular bins; not simulated reflectance. |
| Camera system | QImaging Retiga 1300, 1300 by 1030, 10-bit | pixels, bits | Published library acquisition apparatus; no raw frames bundled. |
| Illumination | Hamamatsu SQ xenon source; approximately 0.5-degree angular steps | degree | Moving light, stationary camera and spherical sample. |
| Exposure sequence | 330 HDR photographs, 18 exposures per photograph, 40 microseconds to 20 seconds | s | Published acquisition protocol for the library. |
| Angular axes | half theta in [0,pi/2], difference theta in [0,pi/2], difference phi in [0,pi) | rad | Isotropy removes half-azimuth; reciprocity halves difference azimuth. |
| Bin lookup | floor(90 sqrt(2 theta-half/pi)); floor(180 theta-difference/pi); floor(180 phi-difference/pi), clamped to [0,89], [0,89], [0,179] | index | Half-angle axis is nonlinear; negative difference azimuth receives pi before binning. |
| Backscatter sampling | theta=(j/89)^2 pi/2, j=0,...,88; difference theta=phi=0 | rad | Excludes horizon; direction pole snaps to theta=0 when cos(theta)>0.99999. |
| RGB luminance weights | (0.2126, 0.7152, 0.0722) | dimensionless | Apply after channel scaling; any negative RGB component makes that sampled triplet invalid and zero. All bound backscatter samples are valid. |
| Surface symmetry and visibility | Isotropic; Smith correlated monostatic visibility | — | Both illumination and observation outward; surface normal is polar axis. |
| Geometric boundary | Density at horizon fixed to zero | inverse slope-area | Infinite-slope boundary of the finite tabulation. |
| Noise | No per-bin uncertainty supplied | — | Do not assume an independently measured noise variance. |

The full table is supplied to preserve measurement context, including directions outside the selected model. Off-slice agreement is not asserted.

## Method Hints

The backscatter equation becomes a positive linear eigenproblem in the slope density: the unknown Fresnel scale is its eigenvalue. Power iteration followed by radial normalization yields a nonparametric profile. Use the angular quadrature implied by the square-root-angle grid and account for visibility in the operator. Moments then give two analytic-distribution roughness summaries; these summaries need not describe all features of the recovered profile.

## References

- Dupuy et al. (2015), *Extracting Microfacet-based BRDF Parameters from Arbitrary Materials with Power Iterations*. [Paper](https://doi.org/10.1111/cgf.12675).
- Matusik et al. (2003), *A Data-Driven Reflectance Model*. [Paper](https://doi.org/10.1145/882262.882343).
- MERL BRDF Database, measured material data, CC BY-SA 4.0. [Dataset](https://doi.org/10.5281/zenodo.8101681).
