# DIRBE Dust Colour Temperature and Surface Density

> Recover apparent interstellar-dust temperature and surface-density maps from three real far-infrared brightness images whose spectral shapes couple temperature and column.
> Domain: astronomy | Keywords: thermal imaging, spectral inversion, dust emission | Difficulty: Medium

## Background

Interstellar dust grains absorb starlight and emit thermal radiation in the far infrared. Their emission encodes both how much material lies along a viewing direction and how strongly that material is heated. Mapping these quantities helps characterize molecular clouds, the interstellar medium and the environments where stars form.

The temperature changes the shape of the far-infrared spectrum, while the amount of emitting material sets its amplitude. Limited wavelength coverage, broad instrument bands, unresolved temperature mixtures and uncertain dust optical properties create ambiguities. A fitted single temperature is an effective colour temperature, rather than a measurement of every grain's physical temperature.

## Problem Description

Recover two apparent dust fields on the supplied Taurus-region sky patch from COBE DIRBE zodiacal-subtracted mission-average brightness at100,140 and240micrometres. The explicitly selected acquisition approximation treats each quoted broad-band radiance as monochromatic at its nominal wavelength. Adopt optically thin emission with fixed dust opacity:

```math
I_\nu(p)=\Sigma(p)\kappa_{350}\left(\frac{\nu}{\nu_{350}}\right)^2 B_\nu(T(p))+e_\nu(p),\qquad
B_\nu(T)=\frac{2h\nu^3}{c^2[\exp(h\nu/(kT))-1]}.
```

Here p is a sky pixel, I is specific radiance in W m^-2 Hz^-1 sr^-1, T the apparent colour temperature in K, Sigma the apparent dust surface density in kg/m², kappa350=0.192m²/kg, nu=c/lambda and nu350=c/(350micrometres). The residual e includes instrument noise and model discrepancy. One MJy/sr is1e-20W m^-2 Hz^-1 sr^-1. Constants h,k,c appear in the acquisition table. The unknown fields share the same32by48 spatial grid; there is no distance parameter or hidden spatial registration.

DIRBE radiances are quoted under a constant nu*I spectrum convention. Broad-band colour corrections can therefore bias the apparent quantities fitted by this monochromatic approximation. The public response curves allow a diagnostic of that bias; they do not redefine the specified monochromatic inverse problem. The benchmark evaluates faithful recovery of the conditional apparent fields, not true cloud temperature or mass. Multiple temperatures along a sight line and uncertain opacity remain physical limitations.

Save `output/reconstruction.npz` with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 32, 48) | float64 | Real apparent colour-temperature map, K |
| surface_density | (1, 32, 48) | float64 | Real apparent dust surface-density map, kg/m² |

Axes are `(batch, latitude sampling row, longitude sampling column)`. Positions are the supplied actual native-pixel Galactic coordinates, rather than an exact uniform spherical grid. Preserve absolute physical scale. Each field is evaluated separately with uncentered cosine similarity and RMS error divided by that field's reference dynamic range. Every metric must pass. No crop, rescaling, offset removal, translation or coordinate alignment is applied.

## Data Description

Load `data/raw_data.npz` with `allow_pickle=False`. All arrays are real float64.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intensity | (1, 3, 32, 48) | float64 | Measured zodiacal-subtracted brightness, MJy/sr; band,row,column |
| uncertainty | (1, 3, 32, 48) | float64 | Product standard deviation of weekly means, MJy/sr; relative fitting weights |
| wavelengths | (1, 3) | float64 | Nominal wavelengths100,140,240, micrometres |
| galactic_longitude | (1, 32, 48) | float64 | Actual selected native-pixel centre longitude, degrees |
| galactic_latitude | (1, 32, 48) | float64 | Actual selected native-pixel centre latitude, degrees |
| response_wavelength | (1, 249) | float64 | Instrument-response sampling wavelengths, micrometres |
| spectral_response | (1, 3, 249) | float64 | Pass3B relative spectral responses, dimensionless, band then wavelength |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Mission/product | COBE DIRBE ZSMA, bands08/09/10 | identifier | Zodiacal-subtracted mission-average observations |
| Nominal wavelengths |100,140,240 | micrometre | Selected monochromatic approximation |
| Nominal beam |0.7 by0.7 | degree | Finite instrument angular footprint; samples are correlated |
| Native pixelization | Quadrilateralized spherical cube,393216pixels | dimensionless | Product sky grid |
| Sampling request | Longitude160..184,48points;latitude−26..−10,32points | degree | Nearest native sky centre; no radiance interpolation |
| Maximum centre deviation |0.225291 | degree | Largest request-to-native-centre angular distance |
| Precision | Native float32 values promoted to float64 | dimensionless | Original measurement precision preserved |
| Quoted spectrum | nu*I_nu constant | dimensionless | DIRBE broad-band calibration convention |
| Spectral response | Pass3B November1996 | identifier | Actual dimensionless full-band calibration, supplied for bias diagnostics |
| Dust opacity at350micrometres |0.192 | m²/kg | Fixed adopted opacity scale |
| Emissivity exponent |2 | dimensionless | Fixed spectral assumption, not target-derived calibration |
| Speed of light c |2.99792458e8 | m/s | Frequency-wavelength conversion |
| Planck constant h |6.62606957e-34 | J s | Selected reference-method constant |
| Boltzmann constant k |1.3806488e-23 | J/K | Selected reference-method constant |
| Noise information | Provided weekly-mean standard deviation | MJy/sr | Relative channel weights; not a complete covariance model |
| Boundary conditions | Independent lines of sight | dimensionless | No spatial padding, periodicity or spatial prior |

## Method Hints

A single-temperature modified blackbody separates spectral shape from amplitude. Weighted nonlinear least squares can estimate temperature and surface density at each direction. Keep the emissivity exponent fixed because only three bands are supplied. An analytic or numerical Jacobian can expose temperature-column coupling. Map residuals and inspect the predicted broad-band colour factors to assess limitations of the monochromatic approximation.

## References

- Camps et al. (2016), *Far-infrared and dust properties of present-day galaxies in the EAGLE simulations*. [Article](https://doi.org/10.1093/mnras/stw1735).
- [NASA DIRBE zodiacal-subtracted mission-average products](https://lambda.gsfc.nasa.gov/product/cobe/dirbe_zsma_data_get.html).
- NASA DIRBE colour-correction convention.
