# Calibrated point-source astrometry and photometry: implemented approach

## Bound measurements, units and unknown parameters

Four independent 17-by-17 real GLIMPSE 4.5 micrometre mosaic patches are
fit with an independently recorded IRAC channel-2 flight focus-star PSF.
For patch k the unknown p_k=(F_k,x_k,y_k,b_k) contains a positive amplitude
in MJy/sr times image pixel, local column/row pixel-center positions and
constant sky in MJy/sr. D_k[j,i] is measured brightness at row j,column i,
both 0..16. No externally known stellar catalogue, shape, source amplitude,
background, target-derived PSF or calibrated detector variance is input.

The mosaic product is dated 2006; the flight calibration product is 2004.
Assume its shape is a spatially invariant effective mosaic PSF. This is an
explicit imperfect physical model, not an empirical target-star PSF. Signed
small calibration sidelobes remain. Source patches were selected from moderate
interior local maxima with good one-source model consistency before extraction;
this is an isolated-source benchmark, not a representative crowded-field test
or proof that the selected stars have independently verified saturation flags.

## Instrument sampling, normalization and continuous response

Mosaic pixels have d=1.2 arcsec/pixel, while flight calibration C[u,v] has
h=0.30325 arcsec/sample and shape 81-by-81 centered at (40,40). A literal
fourfold oversampling without adjusting the angular scale is inconsistent,
since d/h is not exactly four. The independent angular metadata fixes the
following calibration transformation without target inference.

Let B(s,t) be the exact cubic tensor-product spline through C in column,row
order, with no smoothing and SciPy/FITPACK's default not-a-knot boundaries.
For calibration output indices u,v=0..80 define a(u)=40+(u-40)*(d/4)/h and

```math
Q[v,u]=\frac{16\,B(a(u),a(v))}
 {\sum_{m=0}^{80}\sum_{n=0}^{80}B(a(n),a(m))}.
```

Q is sampled at d/4=0.3 arcsec; sum(Q)=16=4^2 preserves Photutils ImagePSF
oversampling-squared normalization. Original calibration sum is approximately
0.999999996322208. The resampling stays within original calibration support.
Let S be the exact cubic spline through Q. Evaluate output response
P(a,c)=S(40+4a,40+4c) when both coordinates are in [0,80], zero otherwise.
The calibration already represents detector response, so no second pixel-area
integration is applied. There is no extra shift, rotation, smoothing, absolute
value of sidelobes, periodic extension or re-estimation from target stars.

The discrete normalization sum(Q)/16=1 approximates the continuum integral
over detector-pixel coordinates. Pixel-center sampled sum(P) can change slightly
with subpixel phase and finite support; F is therefore the model amplitude
under this convention rather than an exactly measured aperture sum. The pixel
solid angle is approximately Omega=(1.2/206264.806247)^2 sr, so F*1e6*Omega
gives an approximate Jy scale. No Jy conversion or physical-accuracy assertion
enters optimization or evaluation.

## Acquisition model, identifiable structure and objective

The modeled image is M_k[j,i]=F_k*P(i-x_k,j-y_k)+b_k. The residual
e_k=M_k-D_k includes correlated mosaic noise, PSF mismatch, faint contaminants
and unmodeled sky structure. No detector variance/covariance map is supplied.
Minimize unweighted J_k(p)=sum_ij e_k[j,i]^2 over all 289 pixels in each
patch, with units (MJy/sr)^2. There is no mask, robust loss, regularizer,
channel normalization or covariance/uncertainty estimator. Generic SciPy uses
half this sum internally; the minimizing p is unchanged.

At fixed x,y the model is linear in F,b: its two design columns are sampled
PSF and all ones. A nonconstant localized PSF distinguishes sky from amplitude;
position derivatives distinguish shifts from amplitude when adequately sampled.
The shipped original/local fit jointly optimizes all four parameters rather
than analytically eliminating the linear pair. It does not assume knowledge
of the result or recover missing faint companions.

## Initialization, bounded inverse updates and stopping

For each patch find the first row-major maximum of D as initial x0,y0.
The border is top row, bottom row, then each side excluding its corner pixels;
every boundary pixel appears once. Set b0 to its median and
F0=sum_ij max(D[j,i]-b0,0). Start p=(F0,x0,y0,b0).
Bounds are F>=0, x0-2<=x<=x0+2, y0-2<=y<=y0+2 and unbounded sky.
These are fixed inverse numerical settings, not acquisition metadata or
independent uncertainty limits.

The original selected ImagePSF+Const2D model is fitted using Astropy's
TRFLSQFitter, which delegates to scipy.optimize.least_squares. Local code
implements the full PSF, residual and initialization and passes the same
generic optimizer arguments: method='trf',jac='2-point', relative
diff_step=sqrt(float64 epsilon), max_nfev=200,xtol=1e-10,ftol=gtol=1e-8.
It does not call a domain photometry library or replace a custom inverse update.
No explicit x_scale is passed; generic SciPy's default for this method applies.

For each current p, numerical A=dr/dp linearizes r(p+delta) as r+A delta.
The generic trust-region reflective algorithm selects a bounds-aware step
within a scaled trust region, evaluates the actual versus predicted cost
decrease and adjusts its radius. It repeats until the cost/step/scaled-gradient
criterion or evaluation limit. A non-success result raises RuntimeError; a
successful result supplies its fitted parameters without post-fit correction.
On the validated original bound run all four ended by ftol in 9,9,9,10 function
evaluations. There is no outer alternating loop, training, background refit,
candidate matching or data-dependent calibration update.

Away from support boundaries, independent derivatives are dM/dF=S,
dM/db=1, dM/dx=-4F*(partial S/partial first coordinate),
dM/dy=-4F*(partial S/partial second coordinate). At samples outside support,
the PSF contributions vanish and the sky derivative remains one. The test suite
compares the resulting 2*A^T*r objective gradient against central objective
differences and verifies exact known-source recovery, translation and intensity
linearity. These are independent mathematical tests, not fixtures generated by
the cleaned implementation and passed off as original parity.

## Submission, comparisons and metric definitions

Preserve four-patch order. Save positions (1,4,2),flux (1,4),background (1,4),
all float64, with final position axis x-column,y-row. Position must be in
[0,16] and amplitude nonnegative under Judge validation; the observed fitted
amplitudes are strictly positive. No global coordinate, phase or scaling freedom
is introduced. The standalone independent Judge imports no solver and uses
the absolute amplitude/sky units above.

For standard parameters p_ref, metrics are maximum_k Euclidean position error
in pixels, maximum_k |F-F_ref|/F_ref for strictly positive references, maximum_k
|b-b_ref| in MJy/sr, and global ||M-D||2/||D||2 over four full image patches.
Boundaries are respectively 1e-5,1e-5,1e-4,0.12. Actual true positions/fluxes
are unavailable; this is independently executed original-method agreement plus
public-data consistency, not physical truth or a confidence interval.

Original repeated double fits were identical. Rounding effective PSF to float32
in a separate original run perturbed positions at most 1.4987535863424255e-8
pixel, relative amplitudes 1.1003613781237246e-8 and sky 4.0027259196051546e-7
MJy/sr. Nonzero tolerances were frozen before extraction from that experiment.
Original/fresh observation residual is 0.10814110360050014; the declared 0.12
consistency bound allows that measured model discrepancy, without pretending
to fit real images to numerical zero. Structured residuals remain in review.

Frozen original stage fixtures establish calibration, initialization, model
evaluation and converged parameter parity. Independent tests establish
sum/oversampling normalization, one-pixel translation, flux/sky linearity,
known-source identifiability, objective gradient and analytic metric cases.
The active notebook preserves independent/adopted calibration plots, all four
observed/original/fresh/residual rows, complete parameter table and paired
fresh-initialization/original/fresh residual bars. Neither raw initialization
nor independent references are relabeled as fresh fitted results.
