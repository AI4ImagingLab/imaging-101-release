# Approach: effective-PSF fluorescence Wiener reconstruction

## 1. Problem, data and selected method

Recover a 310×310 latent effective fluorescence image from SDeconv 1.0.4's real processed confocal Airyscan C. elegans intestine export. The provider's loader crops three edge pixels from the original TIFF and casts float32; packaging preserves that result without new degradation or intensity scaling. The supplied 13×13 sigma 1.5-pixel Gaussian is the provider example's assumed effective PSF, not an experimentally measured calibration or an estimate from hidden truth. Physical sampling, detector gain, optical wavelength and original processing are unknown. Work exclusively in pixel coordinates and arbitrary intensity units.

The extracted method is the framework's 2D SWiener/Tikhonov demonstration, not its SPITFIR(e) method, its 3D inverse, or Richardson-Lucy. There is no independent sharp biological truth. The independent original run provides a method reference; reproduction is conditional on the effective blur model.

## 2. Forward model and boundaries

Let y (i,j) be the observed intensity, x the effective latent image and h the dimensionless unit-sum PSF. Reflect y by 13 pixels on every side, excluding endpoint duplication, to obtain p on a 336×336 extended grid. Use circular convolution only on this extended grid:

```math
p=Hx+\epsilon,\qquad (Hx)(r)=\sum_s h(s)x(r-s),\qquad\sum_s h(s)=1.
```

Indices r,s are integer (row,column); rows increase downward and columns rightward. Epsilon represents unknown residual noise and model mismatch, not calibrated Poisson counting noise. x,p are in a.u. No pixel-area or physical-pitch factor enters the discrete sum. Reflection is an estimator boundary prescription, not a claim that tissue literally repeats outside the field.

Normalize the float64-promoted tabulated PSF by its sum (the original float32 sum differs from one by 2.09e-8). On the extended grid embed its 13×13 support starting at (int (336/2-13/2)+1,int (336/2-13/2)+1)=(162,162), so its center is (168,168). Roll by (-168,-168) before fft2. This exactly fixes the blur origin and avoids an unintended translation. For arbitrary helper shapes use the implemented int (n/2-k/2)+1 embedding and int (-n/2) roll conventions; do not replace them with a different centering formula.

## 3. Objective and regularizer derivation

The positive five-point periodic Laplacian is

```math
(Lx)_{ij}=4x_{ij}-x_{i-1,j}-x_{i+1,j}-x_{i,j-1}-x_{i,j+1}.
```

All neighbors wrap on the extended grid. The selected objective is

```math
J(x)=\tfrac12\|Hx-p\|_2^2+\tfrac\beta2\|Lx\|_2^2,\qquad\beta=0.005.
```

The norms sum over every extended pixel with no division by 336², physical area, noise variance or observation count. The regularizer penalizes squared curvature rather than TV or first differences. The unconstrained problem permits negative ringing; no positivity constraint, maximum clipping, row normalization or fitted gain is applied.

Taking the gradient yields H*(Hx-p)+beta L*Lx=0 and normal equation (H*H+beta L*L) x=H*p. Here H* uses the complex-conjugated blur transfer function, not array flipping with an extra origin shift. L is self-adjoint under periodic boundaries.

## 4. Fourier inverse and origin conventions

For unnormalized forward DFT F and inverse F^-1 divided by N=336², let q=Fh_origin, P=Fp and d be the Fourier transform of the centered five-point impulse. Convolution diagonalization gives

```math
X(k)=\frac{\overline{q(k)}P(k)}{|q(k)|^2+\beta|d(k)|^2},\qquad x=\operatorname{Re}F^{-1}X.
```

The source's Laplacian impulse is centered rather than rolled. Its DFT differs from an origin stencil by a unit-modulus phase; the squared modulus in this quotient is unchanged. The independent normal-equation check uses the origin-equivalent spatial periodic stencil, so it tests stationarity without simply repeating the same spectral expression.

The blur DC response is 1 because h sums to 1, while L annihilates constants and d (0)=0. Thus the DC denominator is 1 and the full extended solution conserves the sum of p. The Gaussian blur and quadratic term jointly condition nonzero frequencies. No denominator epsilon or undocumented regularization is inserted.

## 5. Complete execution algorithm

1. Safely load observation (1,310,310) and psf (1,13,13), reject nonfinite/negative arrays or a zero-sum PSF, then return unbatched float64 copies.
2. Reflect the observation by 13 pixels per edge; normalize and embed the PSF on the common 336×336 grid.
3. Roll the embedded blur kernel to origin and calculate its complex128 OTF. Build the centered float64 Laplacian impulse and fft2 it separately.
4. Calculate the explicit spectral quotient above using beta 0.005, inverse FFT and real part. The negligible imaginary cancellation residual is not a magnitude image.
5. Crop[13:-13,13:-13], preserve signed float64 intensities and save a leading batch axis in output/reconstruction.npz.

There is no initialization, iterative inverse update, line search, learning rate or stopping tolerance: the selected minimizer is evaluated directly. Generic FFTs supply numerical primitives while task-specific padding, origin, objective and inverse remain local. The return_full helper omits the final crop for full-grid mathematical tests; it is not a different estimator.

| Parameter | Selected value | Meaning |
|---|---|---|
| beta |0.005| Squared-Laplacian penalty weight |
| pad |13 pixels per edge| Endpoint-excluding reflect extension |
| Extended size |336×336| FFT/normal-equation domain |
| PSF sigma/support |1.5 pixels;13×13| Supplied assumed effective blur |
| Output scale |Input a.u.| No sum, norm, percentile or peak rescaling |

## 6. Comparisons, metrics and numerical evidence

The notebook actively computes fresh local reconstruction and retains all original scientific groups: acquired observation, independently rerun original Wiener, fresh local Wiener, signed difference, central-row profiles and observation/fresh NRMSE comparison. All three image panels share display percentiles only; stored arrays and metrics remain unclipped. The original result is not biological truth. No historical TIFF or paper score is substituted for an actual original run on the bound observations.

NCC=dot (a,b)/(||a||||b||) without mean subtraction; NRMSE=sqrt (mean ((a-b)²))/(max (b)-min (b)). Evaluate all 310² pixels on the original scale, no mask/translation/gain fit. Both NCC>=0.999999 and NRMSE<=2e-6 are mandatory. Boundaries were fixed before extraction from original repeats and independent float32/mixed-double sensitivity 4.9049516436171566e-8. They are nonzero method tolerances, not a physical accuracy criterion. A zero estimate has NCC 0; zero-norm or constant references are evaluator errors.

Original Torch float32 and local NumPy float64 may differ through kernel normalization/FFT arithmetic. Stage tests compare extension and Laplacian exactly, embedded kernel at 1e-8, OTF at 3e-7, and complete reconstruction with rtol 2e-5/atol 0.02 a.u. The independent spatial normal-equation residual must be below 1e-13 relative to p. An asymmetric PSF tests H/H* adjoints, DC response and flux independently of a symmetric Gaussian. The fresh residual is approximately 2.73e-16; the nonzero forward residual about 0.03535 is expected under regularization. Cropped-pixel flux conservation is not asserted: the identity concerns the full extended solution only.

## 7. Limitations and validated scope

Numerical reproduction establishes correct extraction of the specified standard linear estimator. Unknown Airyscan processing and an assumed Gaussian PSF limit optical interpretation. Edge reflection can create model mismatch, and ringing/negative values remain possible. No restoration-quality claim against an independently measured sharp specimen, true photon counts, calibrated resolution or actual molecular concentration is made. Generalizing to measured PSFs, alternate beta,3D stacks or other tissue is outside the demonstrated benchmark.
