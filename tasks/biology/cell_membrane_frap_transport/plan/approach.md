# Implemented FRAP inverse mathematics

## Bound acquisition and unknowns

The complete author Sample_1D movie contains42 frames of519×408 uint16 camera intensities. The authoritative file-specific spacing is float32(0.0431)µm/pixel and frame interval float32(30)s. One prebleach frame precedes41 recovery frames. D is a scalar effective lateral diffusivity. The unknown initial fluorescence field is eliminated as a linear nuisance function; the evaluated field is the reconstructed optical observation in nine spatial modes, not an independently calibrated molecular concentration or an unregularized inverse initial profile.

## Sampling and normalization

Both supplied centerlines were exported from the original ImageJ1.54f straightening operation on the author-defined polygons. They are observation-defined ROI geometry, not inferred transport parameters. For each current center point (x,y), use the difference from its predecessor, extrapolating the first predecessor. Let dx be the horizontal difference, dy the negative vertical difference, and L the Euclidean length rounded to float32. Nine samples have coordinates (x+k dy/L,y+k dx/L), k=−4,…,4. Evaluate bilinear interpolation from the original uint16 image, rounding each result to float32. Reduce the transverse strip by the original triangular downsampling kernel with replicated edges; accumulate float32 weighted samples in source-index order. This exactly reproduces the independently captured ImageJ profiles.

Let r be the full42×89 FRAP profile. The implementation computes s=1.2 min(r), a=0.8(max(r)−s), and normalized arrays (array−s)/a, preserving original NumPy scalar promotion. B is the mean normalized camera intensity in background rectangle x[63:204],y[50:452] across all frames. c_s is the prebleach normalized FRAP mean minus B. Fit control-frame means to A exp(−epsilon n)+B using scipy.optimize.curve_fit: initial(A,epsilon)=(1,.001), bounds(0,0) to (1000,(1/42)*1000), other defaults unchanged. The order of arithmetic in that upper bound is preserved because tiny changes can affect the nearly singular subsequent projection. epsilon is per frame, not per second.

## Explicit forward Gram operator

Internal diffusion d=D Delta_t/Delta_x² has units pixels²/frame. Recovery index t=0,…,40 and spatial index x=0,…,88. C is the9×89 orthonormal real Fourier matrix with rows 1/sqrt(89), sqrt(2/89)cos(2πkx/89) for k=1,…,4 and sine rows k=4,…,1. Compression is Y=y C^T and its adjoint is Y C. The heat model is not periodic even though its finite observation compression uses trigonometric rows.

For two recovery times i,j and sampled positions x,z, the implemented uncompressed Gram kernel is

```math
K_0(i,x,j,z)=\frac{\exp[-(x-z)^2/(4(d(i+j)+1))]}{\sqrt{4\pi(d(i+j)+1)}}\exp[-(\beta+\epsilon)(i+j)].
```

All variables in this formula are in the internal pixel/frame convention. The constant1 is the original DX² with DX=1. This is the convolution of two Gaussian forward responses: its variance is2d(i+j)+2. The corresponding individual optical Gaussian has standard deviation1 pixel. beta=0 for the standard pure-diffusion method; beta is the exchange/loss per frame in notebook comparisons. Compress each spatial block as C K0 C^T, then assemble a369×369 block Hankel matrix K using the dependence on i+j. No unobserved boundary values are synthesized.

## Initial-field elimination and objective

Use scipy.sparse.linalg.svds(K,k=1) for the largest singular value sigma. The explicit execution convention supplies default_rng(1729) for its starting-vector generator. Original default-RNG ensembles are separately retained; fixed initialization is not asserted to establish physical truth or cross-platform bit identity. lambda=sigma*(41*9*float64_epsilon) is unchanged from upstream, including multiplication grouping. Cholesky-factorize K+lambda I using dpotrf, invert the triangular factor with dtrtri, and form H=L^-T L^-1. The residual projector is P=lambda H. Compute tau=trace(P^T P)=lambda² sum(H²).

Subtract B sqrt(89) from the DC component of each observation. For reaction-containing models also subtract c_s exp(−epsilon t)sqrt(89). The normalized residual vector is R=P(Y−offset)/sqrt(tau). Its sum of squares J is the objective. No independent regularizer on D, artificial iterations, added smoothing penalty or altered ridge is introduced.

The original near-machine-precision ridge makes the objective sensitive to small linear-algebra differences. Unmodified original runs with unconstrained ARPACK initialization gave D5.91724e−5 to6.14589e−5µm²/s while J remained approximately0.2033. A controlled initialization convention gave repeated D5.993295324677043e−5. Stage fixtures and a pre-local numerical ensemble envelope are therefore both tested; no claim of bitwise full-pipeline parity is made.

## Nonlinear inversion

Initialization uses d0=(89/2)²/((41−1)/2)/8. Bounds are ((89/9)²/41/40,89²/5). scipy.optimize.least_squares uses trust-region reflective, two-point finite differences with relative diff_step=.01, linear loss, x_scale=abs(initial), xtol=1e−4, ftol=gtol=1e−12 and max_nfev=500. Default trust-region linear solver is retained. Stop according to those original criteria; expose success and evaluation count in diagnostics.

Recover the fitted compressed optical observations as Y_fit=Y−sqrt(tau)R. Transform to pixels as a(Y_fit C)+s and restore the batch axis. Return D=d Delta_x²/Delta_t in µm²/s. No magnitude conversion, clipping, baseline removal or rescaling is applied to the evaluated output. The covariance diagnostic is pinv(J_R^T J_R)*J/tau, where J_R is the optimizer Jacobian; it is an approximate local uncertainty, not an independent truth error.

Adjusted R² follows the author convention: remove the mean of the DC channel from Y to obtain Y0, then 1−[J*tau/(tau−p)]/[sum(Y0²)/(N−1)], where p is the author's parameter-count convention:1 for pure diffusion,2 for both reaction-containing models, and N=369. The reaction-only branch retains that author count despite its single optimized loss parameter and stationarity-constrained source; it is a display diagnostic, not a Judge metric. The notebook retains pure diffusion, reaction-only and reaction-diffusion comparisons. Reaction-only fixes d=0; reaction-containing models initialize beta=log(2)/20 with bounds(−3/41,100/41), preserving the upstream diagnostic allowance of negative loss. A negative loss is explicitly labeled physically inadmissible, not presented as a measured uptake rate. A separately labeled no-imaging-bleaching ablation sets epsilon=0; it is not the benchmark method.

## Evaluation

Independent truth is unavailable. The oracle's fitted field and D form a hidden method reference. Standard cosine NCC and reference-range NRMSE evaluate every field pixel; relative D error evaluates the scalar. Pre-extraction boundaries are NCC≥.9999, NRMSE≤.003, relative D error≤.05, based on original-run stability and limiting weak initial-field modes by the declared nine-mode output representation. These are numerical reproducibility criteria, not physical confidence intervals. Independent checks cover Gaussian convolution, compression adjoints, Gram symmetry/positivity, calibration units and sampling parity.
