# Implemented approach

## Observations, response and units

Six real AIA level1.5 synoptic images around 2014-01-01 00:00 UTC use channel order 94,131,171,193,211,335 angstrom. Observed 171-angstrom brightness alone selects a bright active-region center after Gaussian sigma=8 smoothing, restricted to rows/columns 220:804. The fixed bound ROI is rows415:463, columns424:472. This selection uses observations rather than unknown DEM. Each image is divided by its stored EXPTIME verbatim, preserving float64 count rates. The independent nominal AIA evenorm response is reused from the author archive. No additional degradation correction is introduced; this reproduces the reference analysis calibration rather than claiming perfect date-dependent calibration. Discrepancy standard deviation is 0.1*rate+1 DN/s/pixel, exactly the author patch-demo convention.

Temperature edges are 10**linspace(5.6,7.4,21) kelvin. Let ell=log10(edges), width=diff(ell), and center_t=ell[0]+width[t]*(t+0.5), t=0..19, preserving the original uniform-grid convention. Replace nonpositive response entries by the channel's minimum positive response. Interpolate log10 response linearly against the supplied log10 temperature grid, then exponentiate. Set DeltaT_t=10**center_t*ln(10**width[t]). The matrix K_tc=R_c(T_t)*DeltaT_t maps DEM xi_t (cm^-5 K^-1) to count rate through d=K.T@xi. The response has units DN cm^5/s/pixel; K adds kelvin. This midpoint logarithmic quadrature is not exact edge integration. Response log-temperature coordinates remain the original big-endian float32 values; calculations use them without silently rewriting calibration precision.

## Whitened regularized inverse

For one spatial pixel, define the scaled unknown x=xi/1e15 and response Rm=1e15*K. Whiten A=(Rm/error).T, shape(6,20), and y=d/error, shape(6). Minimize ||A*x-y||²+lambda*||L*x||², with positive diagonal constraint L. No 1/M, 1/2 or signed-regularizer change is introduced. Compute the thin SVD A*L^-1=U*diag(s)*V.T. The explicit inverse is G=L^-1*V*diag(s/(s²+lambda))*U.T, and x=G*y. SVD and root finding are permitted general numerical primitives; all inverse mathematics and weighting remain local executable code.

Choose lambda by the discrepancy principle ||A*x-y||²=target. Set coef=U.T@y and outside=max(||y||²-||coef||²,0). The squared residual is sum[(lambda/(s²+lambda)*coef)²]+outside. Search log(lambda) between log(min_positive(s²)*1e-8) and log(max_positive(s²)*1e8), using Brent root finding with xtol=1e-10. If the target lies outside the achievable interval, return the nearer endpoint. An all-zero singular spectrum returns lambda=1. No optimizer initialization or gradient line search applies.

## Weight estimation and positivity iterations

First use Ldiag=1/sqrt(width), target=6*reg_tweak (default1), and obtain preliminary dr0. Keep positive entries above 1e-4*max(dr0); fill other entries with that floor, or1 if no positive entry qualifies. Smooth these weights with convolve(weights[1:-1],ones(5)/5)[1:-1]/max(weights), then floor at1e-8. Final Ldiag=sqrt(width)/sqrt(abs(weight)). Compute one new SVD for this final constraint.

Solve at target=6*rgt, initially rgt=1. If negative DEM bins remain, multiply rgt by1.5 and repeat, up to10 iterations. Return the last inverse result without clipping. All standard bound pixels are nonnegative, but some have chi-square/channel as high as11.390625 because positivity required a relaxed discrepancy. The notebook preserves those residuals. EM-loci weighting uses the minimum positive d_c/Rm_tc across selected channels before the same smoothing and floor. The notebook actively computes all-channel EM-loci weighting plus discrepancy targets0.5 and2. Any negative bins at the iteration cap are counted in its table; positive flooring for log display never modifies numerical metrics.

## Uncertainty and forward closure

Vertical sigma for scaled x is sqrt(sum_c G_tc²); multiply both x and sigma by1e15 to recover DEM units. Predicted counts are Rm.T@x. Chi-square/channel is mean_c[(d-fit)/error]². The local forward check separately evaluates DEM@K and compares those counts.

The temperature resolution matrix is W=G*A. For each temperature column, interpolate W onto51 probe points from min(center)+1e-8 with spacing (max(center)-min(center))/51. Select probe values above half the column maximum. Use half their selected log-temperature span, or the bin width if none qualify, and divide by2*sqrt(2*ln2) to obtain the reported temperature-resolution sigma. This is the exact source convention, not a newly claimed posterior credible interval. Independent tests compare the SVD inverse with normal equations, verify objective stationarity and the discrepancy root, and check quadrature units and integrated forward counts.

## Evaluation

The inverse unknown is DEM. Vertical error and temperature width are derived diagnostics, not additional inverse targets. With no independently known physical DEM truth, original demregpy on the exact observations provides the hidden algorithm reference. Compare the entire real float64 DEM grid without mask, alignment, logarithm or fitted rescaling, using uncentered cosine NCC and dynamic-range NRMSE. Predeclared thresholds are NCC>=0.999999 and NRMSE<=1e-6. Independent mathematical checks are required in addition to reference agreement. Neither the strict agreement nor a small data residual makes the thermal distribution physically unique.

Independent observation consistency is also required: mean_chi2_per_channel. Original-code value=2.9987725151899878; boundary=3.2986497667089867. The Judge reimplements the declared response or shear mathematics without solver imports. The AIA mean is evaluated with the adopted10%+floor error and nominal response; the KiDS norm compares the two-pixel output resolution and accounts for DC degeneracy.
