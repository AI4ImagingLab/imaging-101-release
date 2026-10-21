# Spitzer Stellar PSF Photometry

> Reconstruct subpixel stellar positions, point-source amplitudes and local sky levels from four real infrared image patches using an independently measured instrument point-spread function despite sampling and model mismatch.
> Domain: astronomy | Keywords: point-source imaging, inverse photometry, calibrated optics | Difficulty: Medium

## Background

An unresolved star is imaged as the telescope's point-spread function rather than a geometric point. Diffraction, detector response and optical aberrations spread its light across neighboring pixels. Infrared imaging reveals stars behind interstellar dust and supports studies of Galactic structure and stellar populations.

Estimating a source position to a fraction of a pixel and separating its amplitude from diffuse sky emission requires a sampled optical model. Instrument calibration can provide an independent point-spread function, although differences between calibration and survey conditions introduce systematic uncertainty. Mosaicking also produces correlated noise and can alter the effective response.

## Problem Description

Each of the four patches contains one dominant unresolved source. Adopt the spatially invariant calibrated point-source model

```math
D_{kij}=F_k P(i-x_k,j-y_k)+b_k+e_{kij}.
```

Here (k=0,1,2,3) indexes patches, (i) and (j) are zero-based column and row coordinates, (D) is observed surface brightness in MJy/sr, (F) is a positive amplitude in MJy/sr times image pixel, ((x,y)) is the source position in image pixels, (b) is constant local sky brightness in MJy/sr, and (e) includes noise, unresolved contaminants and optical-model discrepancy. The calibration is a separate flight focus-star observation. It is adopted as the effective PSF; it is not an empirical PSF measured from these target stars. No independent target truth or detector variance map is supplied.

To specify the model fully, resample the provided 81-by-81 flight PSF by an exact cubic tensor-product spline, centered at calibration coordinate (40,40), to spacing 0.3 arcsec. The new coordinate (u) maps to old coordinate (40+(u-40)0.3/0.30325). Normalize the resampled array to sum 16. Let (S) be its exact cubic tensor-product spline in column,row order. Then (P(a,c)=S(40+4a,40+4c)), with value zero outside the closed calibration grid [0,80] on either axis. This preserves the instrument's small signed calibration sidelobes. The PSF is sampled at output pixel centers: do not integrate it over a second pixel response. No rotation, additional blur, phase shift, deconvolution of the calibration, or data-dependent PSF fitting is part of this acquisition model.

Save one `output/reconstruction.npz` relative to the task workspace, with all three float64 real arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| positions | (1, 4, 2) | float64 | Local source coordinates in image pixels; final axis is (x column, y row), increasing right/up in an origin-lower display; origin is the center of pixel (0,0) |
| flux | (1, 4) | float64 | Positive PSF amplitude in MJy/sr times image pixel; each patch retains its input order |
| background | (1, 4) | float64 | Constant additive local sky level in MJy/sr |

The leading axis is batch. All values must be finite and source positions must lie in [0,16]. The amplitude is a fitted sampled-image flux parameter, not a value in Jy. The pixel solid angle is approximately ((1.2/206264.806247)^2) sr; multiplication by this solid angle and (10^6) converts the amplitude to the corresponding approximate Jy scale. The optical and pixel-sampling assumptions limit the interpretation as an accurate total astrophysical flux.

Evaluation measures the maximum Euclidean position discrepancy in pixels, maximum relative amplitude discrepancy, maximum absolute sky discrepancy in MJy/sr, and the relative L2 residual between the modeled and observed images. Every metric must pass. Comparisons preserve absolute calibration, patch order and pixel coordinates; no matching, alignment, amplitude rescaling, background subtraction or masking is applied at evaluation time. The benchmark evaluates inference under the specified approximate instrument model; residual structure need not vanish on real survey data.

## Data Description

The observations are four unaltered 17-by-17 cutouts from a public Spitzer GLIMPSE channel-2 mosaic. They preserve the mosaic surface-brightness scale. Conversion from FITS storage to float64 NPZ changes neither image sampling nor brightness values. The separate flight calibration is supplied in its original sampling and values.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| images | (1, 4, 17, 17) | float64 | Observed survey surface brightness in MJy/sr; axes batch, patch, row, column |
| flight_psf | (1, 81, 81) | float64 | Independent focus-star PSF samples, dimensionless normalized calibration weights; axes batch, row, column |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Telescope / instrument | Spitzer / IRAC channel 2 | dimensionless | Infrared survey instrument |
| Nominal wavelength | 4.5 | micrometre | Channel band designation |
| Survey product | GLM_01800+0000_mosaic_I2 | dimensionless | Processed GLIMPSE mosaic; not raw detector exposures |
| Mosaic creation | 2006-12-02 | UTC date | FITS product creation date |
| Mosaic pixel scale | 1.2 | arcsec/pixel | Both output image axes |
| Native frame pixel scale | 1.213 | arcsec/pixel | Single-exposure sampling before mosaicking |
| FITS effective integration time | 1.2 | s | Exposure metadata; not an independent noise variance for the mosaic |
| Flight PSF pixel spacing | 0.30325 | arcsec/sample | Fourfold oversampling of native detector pixels |
| Flight PSF provenance | IOC/SV campaign Q focus star; product 2004-01-02 | dimensionless | Independent calibration observation |
| Adopted PSF resampling | 0.3 | arcsec/sample | Fourfold oversampling relative to the mosaic |
| PSF center and support | (40,40); 81 by 81 | calibration samples | Zero-based center, exact spline inside grid and zero outside |
| Resampled PSF sum | 16 | dimensionless | Oversampling-squared normalization |
| Image coordinate convention | x column, y row; zero-based pixel centers | pixel | Patch-local coordinates; array row increases with y |
| Noise covariance | Not supplied | dimensionless | Correlated mosaic noise is possible; adopt unweighted least squares without assuming a known variance |
| Boundary and background | Zero PSF outside support; constant additive sky per patch | MJy/sr for sky | Finite calibration support; no periodic boundaries |

## Method Hints

Calibrated PSF fitting is a nonlinear least-squares inverse problem in subpixel position and linear in amplitude and constant sky when position is fixed. A bright local maximum supplies a position initialization; a robust border statistic supplies an initial sky level. A bounded trust-region fit can constrain the amplitude to be positive and avoid jumping to nearby contaminants. Inspect the residual image as well as parameter convergence because a historical instrument PSF is an approximate model for a resampled survey mosaic.

## References

- Stetson, P. B. (1987). DAOPHOT: A Computer Program for Crowded-Field Stellar Photometry. PASP 99, 191. DOI: 10.1086/131977. General scientific basis for PSF photometry; this task uses the explicit spline model and bounded least-squares implementation described here rather than claiming a complete DAOPHOT reproduction.
- Photutils 2.3.0, `ImagePSF`: public documentation at https://photutils.readthedocs.io/en/stable/api/photutils.psf.ImagePSF.html (the online stable page may describe a newer software release). The selected reference method is the 2.3.0 sampled PSF model plus a constant sky fitted by Astropy's trust-region reflective least-squares fitter.
- Churchwell et al. (2009). The Spitzer/GLIMPSE Surveys: A New View of the Milky Way. PASP 121, 213. DOI: 10.1086/597811.
- NASA/IPAC Infrared Science Archive, Spitzer GLIMPSE: https://irsa.ipac.caltech.edu/data/SPITZER/GLIMPSE/ . Image and independent flight calibration are distributed as Photutils example data. NASA mission/IRSA acknowledgments and educational factual-use conditions apply; see https://www.nasa.gov/nasa-brand-center/images-and-media/ .
