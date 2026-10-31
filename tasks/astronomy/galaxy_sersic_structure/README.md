# Galaxy Sérsic Structure

> Reconstruct the intrinsic extended brightness and structural parameters of a real galaxy in three optical bands from atmospheric-blurred, noisy survey images.
> Domain: astronomy | Keywords: galaxy structure, optical deconvolution, Bayesian inference | Difficulty: Hard

## Background

Extended galaxy images encode the spatial distribution of stellar populations. Atmospheric seeing and telescope response spread light across neighboring detector pixels, while sky background and spatially varying noise hide faint outskirts. Recovering the underlying light profile helps quantify sizes, concentration and shape across wavelengths.

Such measurements support studies of galaxy evolution and comparisons between stellar populations. Intrinsic ellipticity, radial concentration, total flux and background can trade off under blur. Real galaxies also contain structure that a smooth radial model cannot completely capture, so observation residuals remain scientifically useful.

## Problem Description

Infer an independent elliptical Sérsic light distribution and constant residual sky level in each of theg,r,z images of one DECaLS DR9-south galaxy. The image model is

```math
y_b(i,j)=[P_b*I_b](i,j)+s_b+\epsilon_b(i,j),
\qquad I_b(R)=\frac{F_b\,b_n^{2n}}{2\pi n\Gamma(2n)r_e^2(1-e)}
\exp[-b_n(R/r_e)^{1/n}].
```

Here b indexes band, y is measured flux per pixel, P is the known calibrated point-spread function, I is intrinsic surface brightness, s is constant sky flux per pixel and ε is measurement/model error. F is total profile flux integrated over the infinite plane, re the major-axis half-light radius inpixels, n the dimensionless Sérsic concentration ande=1−axisratio the ellipticity. Γ is the gammafunction and bn is the positive constant satisfying the half-light relation γ(2n,bn)=Γ(2n)/2, where γ is the lower incomplete gammafunction. For offsets x−xc,y−yc from centroid, rotate by angleθ+π/2 to obtain major/minor coordinates X,Y and let R=sqrt[X²+(Y/(1−e))²]. Pixelx increases with imagecolumn (westward),pixely withrow (northward); θ is measured counterclockwise from positivepixely in this pixel-coordinate definition. Define model centroids in zero-based pixelcoordinates, centers0..119.

Integrate the central pixels when evaluating the sharply varying profile, and model the calibrated PSF rather than interpreting its blurred core as intrinsic concentration. For this task use periodic convolution on the stated image grid; the galaxy is well inside the boundary. PixelizedPSFs are centered at pixel31in each63×63kernel. The selected numerical convention applies the original approximateπconstant3.1415 for the PSF Fourier shift, an independently quantified0.000914pixeloffset peraxis, which is below the declared structural tolerance. No fitted gain or offset alignment is allowed after reconstruction.

Save `output/reconstruction.npz` relative to the task workspace with both real float32 orfloat64 arrays:

| Output key | Batch-first shape | Contract |
|---|---|---|
| reconstruction | (1,3,120,120) | Intrinsic pixel-integrated galaxy brightness,withoutPSF orsky,innanomaggies/pixel;axesbatch,band(g,r,z),row,column;fullphysicalgain |
| parameters | (1,3,8) | Structural/sky parameters in the exactcolumnorder below; no undeterminedscale |

Parametercolumns: xc(pixel),yc(pixel),F(nanomaggies),re(pixel),n(dimensionless),e(dimensionless),θ(radian),s(nanomaggies/pixel). Angleθ uses the stated[0,π)gauge. Flux is positive,re>0,n>0,0≤e<1. The finite delivered image is the profile over the full stated grid; do not renormalize it to make its finite sum equal the infinite-profileF. Each band's intrinsic image is evaluated by uncentered cosine and dynamic-range-normalized RMS error. Every structural/sky parameter is assessed independently in its statedphysicalunits, with no centroid,rotation,flux orbackgroundalignment.

## Data Description

These are real measured three-band coadds, pixelized observational PSFs and inverse-variance maps. The120×120source-centered region is the original author example's fixedscientificROI(68:188fromthe256×256cutout), preserving native0.262arcsec pixels. The public mask identifies other observed sources from the r-band segmentation,not the inferred target profile. MaskTrueexcludes thosepixels from fitting in allbands; their original values remain available. No target-derived calibration is supplied.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| images | (1,3,120,120) | float32 | Measuredg,r,zflux,nanomaggies/pixel;row/column imageaxes |
| inverse_variance | (1,3,120,120) | float32 | Supplied inverse noise variance,(nanomaggies/pixel)⁻² |
| psfs | (1,3,63,63) | float32 | Calibratedg,r,zpoint-spreadkernels,dimensionlessfractionalfluxperkernelpixel |
| mask | (1,1,120,120) | bool | Observedneighbor-exclusionmask,Trueexcluded,dimensionless;sharedacrossbands |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Survey | DECaLS DR9-south | — | Real opticalcoaddedobservations |
| Target center | RA59.1067,Dec−30.4848 | degree | Original example cutout skyposition |
| Bands | g,r,z | — | Fixed bandaxisorder |
| Pixel size | 0.262 | arcsec/pixel | Nativeangularsampling |
| Field size | 31.44×31.44 | arcsec | Complete selected120×120region |
| Imageaxisorientation | +columnwest,+rownorth | — | NegativeRA/positiveDec FITSCDdiagonal |
| Localorigin | (59.5,59.5) | pixel | Geometriccenter;centroidremainsunknown |
| Flux calibration | ABmagnitude22.5correspondsto1nanomaggy | — | Surveylinearfluxunit;noindependentbandgainfit |
| PSFcenter | (31,31) | kernelpixel | Zero-based63×63kernelcenter |
| Noise model | Independent Gaussianusing inverse_variance | — | Approximation;coaddcorrelationsandmodelresidualremain |
| Background | Independent unknownconstantperband | nanomaggies/pixel | Residualskyisfitted,notprivilegedinput |
| Boundary | Periodicon120×120grid | — | SelectedFFTconvolutionconvention |
| Source mask | r-band3RMSsegmentation,keepcentralobject | — | Publicobserved-neighborcalibration,notanswer-derived |

Credit: **Legacy Surveys / D. Lang (Perimeter Institute)**. Survey images are licensedCC BY4.0; original values were only losslessly packaged into NPZ and the author-scopedROI/mask was retained.

## Method Hints

The conditional target is the physical-coordinate Gaussian MAP estimate, independently per band. On all unmasked pixels minimize one half of sum inverse_variance*(PSF-convolved intrinsic image+sky-observation)² plus one half of [((xc-60)/5)²+((yc-60)/5)²+(sky/0.01)²]. There is no observation-count normalization or change-of-coordinate Jacobian penalty. Position priors have pixel units and sky's standard deviation is 0.01 nanomaggies/pixel. Uniform supports are total flux F in [0,300] nanomaggies, effective radius in [0.5,30] pixels, concentration n in [0.65,8], ellipticity in [0,0.9] and theta in [0,pi] radians. These are fixed analysis priors, not measured answers. Accurate pixel integration is required; fitting an unintegrated surface-brightness sample defines a different observation model.

Use a PSF-convolved elliptical Sérsic model with accurate centralpixelintegration. A Gaussian likelihood weights each unmaskedpixel by its suppliedvariance; weakposition/backgroundpriors and boundedphysicalprofileparameters stabilize the morphology/backgroundtradeoff. Gradient-based maximuma-posterior fitting can work in unconstrainedvariables with transformedpositive/boundedparameters. Fit allthreebandsandinspectactualresiduals, rather than equating numericalagreement with a true detailedgalaxymap.

## References

1. Pasha andMiller(2023),pysersic:A Python package for determining galaxy structural properties via Bayesian inference,accelerated withjax. https://doi.org/10.21105/joss.05703
2. LegacySurveys DR9data description. https://www.legacysurvey.org/dr9/description/
3. Image use,licenseandacknowledgment. https://www.legacysurvey.org/acknowledgment/
