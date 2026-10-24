# A3158 X-ray Gas Deprojection

> Reconstruct the spherical radial hydrogen-density and emission-density fields of a galaxy cluster from a noisy, exposure-modulated X-ray count image with foreground and outer-shell contamination.
> Domain: astronomy | Keywords: X-ray imaging, spherical deprojection | Difficulty: Medium

## Background

Hot intracluster gas emits X-rays through thermal plasma processes. Its brightness measures an integral of the product of electron and hydrogen densities along the line of sight. Spatially resolved observations can therefore reveal the distribution of the diffuse baryonic matter in a galaxy cluster.

An image superposes emission from many depths. Recovering a three-dimensional gas distribution requires geometric assumptions, careful treatment of instrumental exposure and backgrounds, and a count-to-emission calibration. Radial gas profiles support studies of cluster structure and mass, while their interpretation remains conditional on symmetry and plasma assumptions.

## Problem Description

Use the complete supplied ROSAT image of A3158 to recover 35 concentric spherical shells. Let C(p), E(p), and B(p) denote the count image, exposure in seconds, and instrumental-background counts at pixel p. An annulus of observed area A has background-corrected surface rate s = sum((C-B)/E)/A - b, where b is the constant sky background measured between 30 and 40 arcmin. Its count uncertainty is sqrt(sum(C/E²))/A; propagate the fitted background uncertainty in quadrature.

Let q(j) be the hydrogen emission density nH² in shell j, V(i,j) its intersection volume with cylindrical annulus i, and a(i) the full geometric annulus area. The model is

```math
s_i = \frac{\sum_j V_{ij}q_j}{K a_i}+s_i^{\mathrm{ext}},\qquad
K=\frac{4\pi 10^{14}D_L^2}{(1+z)^2\eta\,c}.
```

Here z is redshift, DL the luminosity distance in cm, eta = ne/nH the electron-to-hydrogen ratio, c the count rate per XSPEC normalization, and the exterior term accounts for emitting gas beyond the outer shell. For spherical shells with inner and outer radii u(j), v(j), and annular edges u(i), v(i), the intersection is the corresponding difference of four positive-part terms 4pi/3 times (radius²-impact²)^(3/2). The upper triangular projection mixes outer gas into every inner annulus. Noise amplification, uncertain background, finite field extent, and spherical symmetry limit the recovered field.

Save `output/reconstruction.npz` with both required real float64 arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 35) | float64 | Hydrogen-nucleus density nH, cm^-3, increasing radius |
| emissivity | (1, 35) | float64 | Hydrogen emission density q, cm^-6, increasing radius |

The two quantities are evaluated separately. The standard method smooths q and then smooths its signed square root, so the returned density need not square exactly to q. Evaluation uses uncentered cosine similarity and RMS error divided by the reference dynamic range, with no alignment or rescaling. Independent forward checks compare all 35 projected annuli with their count uncertainties, using both q and density squared. No physical ground truth is implied by agreement with the standard deprojection.

## Data Description

All arrays in `data/raw_data.npz` are batch first.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| counts | (1, 512, 512) | float64 | Native observed counts, count; axes batch,y,x |
| exposure | (1, 512, 512) | float64 | Vignetting-corrected on-axis exposure, s |
| instrument_background | (1, 512, 512) | float64 | Instrumental-background estimate, count |
| radial_edges | (1, 36) | float64 | Contiguous spherical/annular edges, arcmin |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Instrument | ROSAT PSPC | - | Real count image and supplied exposure/background maps |
| Pixel size | 0.2491218 | arcmin | Square sky pixels |
| Pixel coordinates | x right, y down; zero indexed | pixel | Center is the maximum of count image Gaussian-smoothed with sigma 5 pixels after setting nonpositive-exposure pixels to zero; bound image gives (271,280) |
| Valid exposure | greater than 0.05 times maximum | s relative fraction | Fixed observed-area mask |
| Sky region | 30 to 40 | arcmin | Constant-background measurement; outer profile uses 20 arcsec bins to 45 arcmin |
| Annulus assignment | rounded edges to 5 decimals, plus 0.000005 | arcmin | Lower inclusive and upper exclusive; first lower edge zero |
| Redshift | 0.059 | dimensionless | Distance and spectral calibration |
| Angular conversion | 70.73126033926604 | kpc/arcmin | Planck15 cosmology at this redshift |
| Luminosity distance | 272.69501551950657 | Mpc | DA = DL/(1+z)^2 |
| Length conversions | 3.0856776e21; 3.0856776e24 | cm/kpc; cm/Mpc | Physical shell units |
| Electron/hydrogen ratio | 1/0.8527 | dimensionless | Fixed aspl abundance convention |
| Count conversion c | 43.53 | count/s per XSPEC normalization | Author-supplied spectral calibration, independent of the unknown radial field |
| Plasma assumptions | temperature 5; foreground NH 0.0118 | keV; 10^22 cm^-2 | Absorbed thermal APEC calibration, aspl abundances |
| Energy band / response | 0.42-2.01; pspcb_gain2_256.rsp | keV; identifier | On-axis PSPC gain-2 calibration |
| XSPEC normalization | 1e-14 integral(ne nH dV)/(4pi DA²(1+z)²) | conventional | Physical normalization; count rate equals c times normalization |
| Geometry | spherical symmetry; contiguous shells; no PSF deconvolution | - | Conditional radial model, not arbitrary 3D structure |
| Exterior model | McLaughlin finite-edge correction | - | Exterior amplitude tied to observed outer annulus; not zero emission at field edge |

These count, exposure, and instrumental-background maps are real author-provided observations. The spectral conversion is supplied calibration, rather than a new spectral fit. Unknown departures from the plasma, symmetry, or exterior assumptions are systematic limitations.

## Method Hints

Spherical onion peeling solves the upper triangular shell-projection system from outside inward. An exterior-emission correction reduces finite-field bias. Count-error realizations propagate measurement uncertainties through the inverse, and a short median filter controls oscillatory noise amplification. A constant sky component can be estimated independently from the outer annuli by weighted least squares.

## References

- Kriss, Cioffi and Canizares (1983), *The X-ray emitting gas in poor clusters with central dominant galaxies*, Astrophysical Journal 272, 439. [Article](https://adsabs.harvard.edu/pdf/1983ApJ...272..439K).
- Eckert et al. (2020), *Low-scatter galaxy cluster mass proxies for the eROSITA all-sky survey*. [Article](https://arxiv.org/abs/2009.03944). The selected classical onion-peeling comparison differs from the multiscale method introduced in this article.
- [Author deprojection and calibration documentation](https://pyproffit.readthedocs.io/en/latest/Deprojection.html).
