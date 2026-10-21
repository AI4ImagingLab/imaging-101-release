# N2H+ Hyperfine Cloud Mapping

> Reconstruct excitation temperature, optical depth, radial velocity and velocity dispersion maps from real molecular-line spectra whose blended hyperfine components create nonlinear parameter coupling.
> Domain: astronomy | Keywords: spectral imaging, radiative transfer, parameter mapping | Difficulty: Hard

## Background

Cold molecular clouds contain the material from which stars form. Molecular rotational emission carries information about gas excitation and motion that is unavailable from continuum intensity alone. Hyperfine splitting produces several nearby spectral components with known relative strengths; their saturation encodes optical depth as well as brightness.

Mapping line centroids and widths reveals systematic gas motion and turbulent broadening. Excitation and optical-depth maps help distinguish a genuinely bright dense structure from a warmer or less opaque sightline. Single-dish radio spectral imaging is used to investigate prestellar condensations and the internal dynamics of star-forming clouds.

## Problem Description

The observations are a contiguous 3 by 3 angular patch of the OphA region in the Ophiuchus molecular cloud, measured in the N2H+ rotational transition J=1-0 with the IRAM 30m telescope. Each spatial pixel contains 501 measured brightness channels. Reconstruct four real physical maps, with a single uniform excitation component per sightline and unity beam filling. This task estimates properties on the telescope beam scale; no angular deconvolution is specified.

For frequency $\nu$, background-subtracted brightness $T$ in kelvin follows

```math
T(\nu)=[J(\nu,T_e)-J(\nu,T_b)]\,[1-\exp(-t(\nu))]+\epsilon(\nu),
\qquad J(\nu,T)=\frac{h\nu/k}{\exp(h\nu/(kT))-1}.
```

Here $T_e$ is excitation temperature, $T_b$ is background temperature, $h$ is Planck's constant, $k$ is Boltzmann's constant, and $\epsilon$ is measurement noise. The total-opacity parameter $t_0$ multiplies normalized component strengths:

```math
t(\nu)=t_0\sum_{j=1}^{15}\frac{w_j}{\sum_l w_l}
\exp\left[-\frac{(\nu+v L_j/c-f_j)^2}{2(\sigma L_j/c)^2}\right],
\qquad f_j=f_c(1-d_j/c),\quad L_j=f_j(1-d_j/c).
```

The index $j$ denotes hyperfine components; $d$ and $w$ are supplied offsets and relative strengths; $f_c$ is the component reference frequency; $c$ is light speed in km/s; $v$ is fitted velocity in km/s, and $\sigma>0$ is Gaussian velocity dispersion in km/s. The component-specific factor $L$ fixes the numerical Doppler convention. $t_0$ is the sum of component peak optical depths, not the maximum blended opacity and not integrated opacity. Blending, saturation and the temperature/opacity tradeoff make the inverse nonlinear. At very small opacity the latter two parameters become poorly distinguishable; the selected patch has resolved hyperfine emission with finite-opacity fits.

Save `output/reconstruction.npz` relative to your task workspace, containing every following array:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| excitation_temperature | (1, 3, 3) | float32 or float64 | Excitation temperature in K; axes batch, y, x; strictly positive. |
| optical_depth | (1, 3, 3) | float32 or float64 | Total-opacity parameter, dimensionless; axes batch, y, x; nonnegative. |
| velocity | (1, 3, 3) | float32 or float64 | Signed line velocity in km/s in the stated frequency convention; axes batch, y, x. |
| dispersion | (1, 3, 3) | float32 or float64 | Positive Gaussian sigma in km/s, not FWHM; axes batch, y, x. |

The arrays are real and retain physical scale. No phase, translation, cropping or amplitude alignment is performed. Each map is scored by the maximum absolute fractional difference from an independently reproduced standard-method map; all four must be at most 0.0002. The forward spectra must also have maximum reduced chi-square at most 1.3 over the nine pixels. Reduced chi-square is the sum of squared noise-normalized spectral residuals divided by 501-4. This comparison assesses agreement with the specified standard fit and observation consistency, rather than establishing uniquely known physical truth.

## Data Description

`data/raw_data.npz` contains existing experimental observations and molecular calibration. No simulated observations are generated.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| brightness | (1, 501, 3, 3) | float64 | Measured brightness in K; batch, frequency, y, x; original float32 values promoted without resampling. |
| frequency | (1, 501) | float64 | Channel frequencies in Hz, increasing along channel index. |
| hyperfine_velocity_offsets | (1, 15) | float64 | Laboratory component offsets d in km/s, with the ordering used by the strength vector. |
| hyperfine_relative_strengths | (1, 15) | float64 | Relative strengths w, dimensionless; normalize by their actual sum (0.9999990000000001). |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Telescope | IRAM 30m | none | Single-dish spectral mapping of OphA, on-the-fly acquisition. |
| Angular resolution | approximately 26 | arcsec | Telescope beam, not pixel spacing. |
| Spatial sampling | 7.50000019203 | arcsec | Pixel pitch in x and y. |
| Axis orientation | increasing x decreases projected right ascension; increasing y increases declination | none | Original ARC sky-projection array orientation; no periodic spatial boundary. |
| Bound spatial indices | y=25,26,27; x=12,13,14 | zero-based pixel | Contiguous crop of the released CDS cube. |
| Original reference sky coordinate | RA=246.6102288723; Dec=-24.40855924977 | deg | J2000 ARC projection reference. |
| Original spatial reference pixel | x=14.93333352010; y=9.153333173918 | one-based pixel | Reference pixels in the original 21 by 30 grid. |
| Original velocity reference | 2.5 at channel pixel 403.0960083008 | km/s, one-based pixel | FITS velocity-axis calibration. |
| Velocity increment | -0.06284131109715 | km/s | Original consecutive-channel spacing. |
| Radio reference frequency | 93176265000 | Hz | Supplied channel frequencies equal this value times (1-u/c), with u the original radio velocity. |
| Component reference frequency f_c | 93173763700 | Hz | Reference for the 15 molecular component offsets; distinct from the radio-axis reference. |
| Light speed c | 299792.458 | km/s | Doppler conversion convention. |
| Planck constant h | 6.62607015e-34 | J s | Exact SI value. |
| Boltzmann constant k | 1.380649e-23 | J/K | Exact SI value. |
| Background temperature T_b | 2.73 | K | Uniform cosmic microwave background model. |
| Beam filling | 1 | dimensionless | Uniform slab approximation. |
| Spectrometer resolution | 20 | kHz | Published OphA nominal instrumental resolution; use channel samples directly without additional convolution. |
| Noise estimate | population standard deviation of brightness at radio velocities 20 through 28 inclusive, independently at each pixel | K | Data-derived white-noise scale used in chi-square; u=c(1-frequency/93176265000). |
| Baseline convention | supplied background-subtracted brightness; no additional fitted continuum | K | Use every supplied channel in the forward comparison. |
| Spectral boundary | evaluate analytic component profiles only at supplied frequencies | none | No convolution padding or periodic continuation. |

## Method Hints

Use nonlinear weighted least squares with a hyperfine radiative-transfer model. Shared excitation, velocity and dispersion couple all components along a sightline, while the relative strengths provide optical-depth information beyond a single Gaussian. Positive temperature, opacity and width constraints avoid unphysical fits. Comparing the modeled spectra with all measured channels helps detect incorrect frequency signs and confusion between sigma and FWHM. Numerical optimization alone cannot remove thin-limit excitation/opacity degeneracy.

## References

- Ginsburg et al. (2022), *pyspeckit: A spectroscopic analysis and plotting package*, Astronomical Journal 163, 291. DOI: https://doi.org/10.3847/1538-3881/ac695a ; paper version: https://arxiv.org/abs/2205.04987v1 . Appendix C describes N2H+ hyperfine fitting.
- Andre, Belloche, Motte and Peretto (2007), *The initial conditions of star formation in the Ophiuchus main cloud: Kinematics of the protocluster condensations*, Astronomy & Astrophysics 472, 519. Data and observational description: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/472/519 .
- Pagani, Daniel and Dubernet (2009), *On the frequency of N2H+ and N2D+*, Astronomy & Astrophysics 494, 719. DOI: https://doi.org/10.1051/0004-6361:200810570 . The supplied component calibration preserves the selected implementation's updated values.

The observational data are provided through CDS/VizieR for scientific use with attribution; commercial usage depends on the originating publisher's terms. VizieR is acknowledged as the data service. See https://cds.unistra.fr/vizier-org/licences_vizier.html .
