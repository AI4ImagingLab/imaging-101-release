# MCF10A Differential Phase and Absorption Imaging

> Reconstruct spatial optical phase delay and absorption potential of real MCF10A cells from four asymmetric-illumination intensity images despite directional spectral gaps and illumination shading.
> Domain: biology | Keywords: quantitative phase microscopy, asymmetric illumination, weak scattering | Difficulty: Hard

## Background

Transparent cells delay transmitted light through their integrated refractive-index contrast, while also weakly attenuating it. Conventional intensity microscopy often suppresses this phase information. Oblique illumination converts phase gradients into intensity contrast, revealing otherwise difficult-to-see cell structure without fluorescent labels.

Recording several complementary illumination directions helps distinguish phase from absorption. This provides label-free maps useful for studying cell morphology, growth and optical mass distributions. Finite numerical aperture, weak-scattering approximations, illumination shading and loss of low-frequency phase information make quantitative inversion challenging.

## Problem Description

The observations are four real MCF10A cell exposures acquired with top, bottom, left and right half-circular sources. Recover both signed absorption potential u and optical phase delay phi at every observed pixel. Define complex sample transmission T=exp(u−i phi), where u=ln|T|is dimensionless and phi is phase delay in rad. Under the weak-object approximation the normalized observation for sourcej is

```math
d_j= h_{u,j}*u+h_{\phi,j}*\phi+\epsilon_j,\qquad
\widehat d_j=H_{u,j}\widehat u+H_{\phi,j}\widehat\phi+\widehat\epsilon_j.
```

d is intensity contrast after removing smoothly varying illumination/DC, h are spatial impulse responses, H their Fourier transfer functions, hats denote the discrete Fourier transform, *denotes2Dconvolution, andepsilon includes noise and model error. The transfer functions follow the supplied illumination geometry and circular objective pupil. A single asymmetric source has phase-frequency gaps; complementary sources improve coverage, but constantphase remains unobservable. This is a2D projected optical inverse, not a3D refractive-index or cell-thickness measurement.

Save `output/reconstruction.npz` with both required arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| reconstruction | (1, 512, 512) | float64 | Optical phase delay phi in rad; axes batch, row, column; native projected phase convention |
| absorption | (1, 512, 512) | float64 | Signed log-amplitude absorption potential u=ln|T|,dimensionless; same axes and scale |

Use x rightward, y downward, the supplied illumination order, and native pixel pitch. The phase DC piston is fixed by zero-frequency regularized inversion, yielding zero spatial mean up to floating point leakage; do not fit an additional piston/plane. Do not change absorption sign, exponentiate it, rescale, unwrap, rotate or crop the outputs. Both full arrays are evaluated separately by relative Euclidean error,norm(estimate−reference)/norm(reference),without alignment or intensity fitting. Every required quantity must pass.

The preserved original phase variable is a **phase delay**: positive phi corresponds to the negative complex argument in T=exp(u−i phi). Under the alternative convention T=exp(u+i theta), theta=−phi. This sign is fixed for both observations and outputs; do not interchange these two conventions.

## Data Description

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intensities | (1, 4, 512, 512) | uint16 | Original measured detector ADU, axes batch, illumination(top,bottom,left,right),row,column |

The fixed central512×512native detector region retains all four illumination exposures and has83.2µmfield width. No image resizing or precision conversion has been applied.

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Wavelength | 0.514 | µm | Green illumination central wavelength |
| Objective numerical aperture | 0.4 | dimensionless | Circularobjective pupil |
| Illumination numerical aperture | 0.4 | dimensionless | Half-circular brightfield source |
| Inner illumination numerical aperture | 0.0 | dimensionless | No source annular hole |
| Magnification | 40.0 | dimensionless | 20× objective plus2× front-port magnification |
| Camera pixel pitch | 6.5 | µm | Original detector pitch |
| Object pixel pitch | 0.1625 | µm | Native camera/magnification ratio |
| Source rotations | 0,180,90,270 | degree | Source patterns in top,bottom,left,right order under declared FFT coordinates |
| LED spacing | 4 | mm | Original array spacing |
| Array to specimen distance | 69.5 | mm | Original illumination geometry |
| Native full acquisition | 2160 rows ×2560 columns | pixel | Bound region chosen before inversion |
| Bound native ROI | rows824:1336,columns1024:1536 | zero-based half-open pixel | Fixed central region, no reconstruction-dependent selection |
| Bound field width | 83.2 | µm | 512 native pixels per axis |
| Spatial boundary | periodic discrete weak-object operator | dimensionless | No implicit padding, crop or alignment during evaluation |
| Detector precision | unsigned 16-bit | ADU | Preserved observations |
| Noise variance | not calibrated | unavailable | No independently measured noise variance provided |

## Method Hints

Use the weak-object transfer-function decomposition into signed log-amplitude and phase. Quadratic regularization permits a coupled 2×2 Fourier inverse per frequency. Complementary illumination directions fill directional phase gaps. A periodic finite-difference regularizer and explicit proximal updates provide another useful comparison when assessing noise and structure. Smoothly varying illumination can be estimated from each observed intensity image; it is not supplied as an independent target-derived calibration.

For the fixed comparison convention, form each contrast by dividing its intensity by a 256-pixel uniform local mean with reflected boundaries, dividing by the resulting image mean, and subtracting one. A reproducible quadratic objective is the sum over four images of ||F(d)-Hu F(u)-Hphi F(phi)||² plus 0.1||F(u)||²+0.005||F(phi)||², where F is the unnormalized discrete Fourier transform. These constants fix the selected regularized spatial output and its phase DC convention.

## References

Tian and Waller,*Quantitative differential phase contrast imaging in an LED array microscope*,Optics Express23,11394–11403(2015). [Paper](https://doi.org/10.1364/OE.23.011394).

Chen, Phillips and Waller,*Quantitative differential phase contrast microscopy with computational aberration correction*,Optics Express26,32888–32899(2018). [Paper](https://doi.org/10.1364/OE.26.032888). The selected task uses the author-provided MCF10A observations and unaberrated Tikhonov method, with the original TV comparison; it does not claim joint pupil-aberration recovery.
