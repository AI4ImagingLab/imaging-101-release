# Implemented inverse mathematics

The complete measured native magnitude acquisition supplies 2523 supported voxel spectra, each at11 b-values. No normalization, image resizing, segmentation modification or intensity denoising is performed. Public affine maps native voxel indices to millimetres; the voxelwise inverse has no spatial boundary operator. Signals are positive on support, allowing a high-b logarithm. S0 has native intensity units, D and Dstar mm²/s, f is dimensionless. The model is S0 exp(-bD)[f exp(-bDstar)+1-f]; Dstar is extra pseudodiffusion, not the total fast coefficient or blood speed.

## High-b initialization

For b>=400 s/mm² in the primary method, fit log(S)=intercept+slope*b by NumPy polyfit degree1. The comparison uses b>=500, which selects the same three native weights here. Initialize lnS0=intercept,D=-slope. Bounded conjugate gradients minimize sum(log(S)-lnS0+bD)². Bounds for lnS0 are[0.5,1.5]times intercept and D[0.00015,0.0045]. Initial values are clipped to bounds. Define Dinit as the resulting diffusion and Sbase=exp(fitted lnS0). No regularizer, noise weighting or residual normalization is introduced.

## Parameter transform and generic optimizers

Every bounded varying parameter p with lower l and upper u is represented by p=l+(sin(q)+1)(u-l)/2. Its inverse initialization is q=arcsin(2(p-l)/(u-l)-1). Values with absolute q<1e-15 are set to zero, preserving the original generic-library convention. Scalar evaluation order matters: transforms use individual scalar sin and native model multiplies S0*exp(-bD) before the mixture. Tests caught and resolved divergent optimizer paths caused by algebraically equivalent floating-point order; no tolerances were widened.

CG uses SciPy minimize methodCG, default finite-difference Jacobian/line search and maxiter twice the evaluation cap. For n varying parameters the original cap is2000(n+1) objective evaluations. Exceeding it retains the last evaluated internal parameters and records capped=True. Differential evolution operates on q in[-pi/2,pi/2] with best1bin, popsize15,maxiter equal evaluation cap,tol0.01,atol0,mutation(0.5,1),recombination0.7,Latin-hypercube initialization, immediate updates, one worker and polishing. A separate NumPy RandomState(20261027+native support index) preserves original reproducibility without changing shared RNG state. This seed is a maintainer implementation choice, not acquisition information. Objective is the unweighted sum of native signal squared residuals.

## Joint primary fit and refinement

Parameter order is S0,D,f,Dstar. Initialize[1.2Sbase,Dinit,0.1,10Dinit]. Lower bounds[0.7Sbase,0.00015,0,2Dinit], upper[1.7Sbase,1.2Dinit,0.2,0.05]. Optimize all four with differential evolution. Initialize a second differential-evolution fit at this result, with all bounds±5% of first-fit values. If first f<=1e-12, its refinement bounds are[0,0.05] instead. The final fit is returned, not the best result chosen against hidden reference. Outside-support zeros are storage sentinels, never fits.

The fixed-D comparison instead holds D=Dinit in both population fits, initializes Dstar=(Dinit+0.0355)/2, and uses first f bounds[0,0.35], Dstar[Dinit,0.0355]. Its other parameters/refinement rules match primary. The notebook also actively fits a monoexponential ablation from the high-b log-linear regression with nonnegative D and f=Dstar=0; this is a labeled simpler alternative, not a surrogate for the benchmark method.

## Independent physics and evaluation

The analytic signal Jacobian in parameter order is[exp(-bD)M, -bS0 exp(-bD)M, S0 exp(-bD)(exp(-bDstar)-1), -bS0 f exp(-b(D+Dstar))], with M=f exp(-bDstar)+1-f. Finite differences validate all four. S(0)=S0; nonnegative coefficients give nonincreasing signal; Dstar has no effect at f=0 and f has no effect at Dstar=0. These checks are independent of original parity.

Every output field is required. S0,D use absolute RMSE. Dstar uses norm(reference_f*error)/norm(reference_f). Fraction uses the same weighted RMS with weight1-exp(-max(b)*reference_Dstar). Reference weights are fixed evaluation definitions only, not public calibration; zero weights contribute zero, globally zero weights are errors. Product fDstar uses ordinary RMSE. Observation metric is norm(prediction-observation)/norm(observation), across every supported voxel and all11 weights, no scaling/alignment. This is an L2 ratio, not dynamic-range NRMSE.

An independent full original alternate-seed run fixed thresholds before extraction: twice observed coefficient/product repeat differences, plus small declared numerical floors where needed; forward threshold1.1 times the larger original residual. Exact numbers and rationale are in metrics.json and validation.md. No actual ground truth exists. Zero-f voxels and high residuals limit interpretation; neither reference agreement nor weighted metrics prove absolute perfusion accuracy.
