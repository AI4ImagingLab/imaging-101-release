# Implemented renal inverse

The task preserves all 228 kidneys from the released 114 scans/100 participants. Signals and aortic inputs are measured author-processed ROI averages, not raw images. No generator or image-segmentation claim. Parse the original decimal CSV values as float64, preserve row order, and pad to 125 with zeros. Lengths exclude padding from every numerical fit and metric. Calibration uses per-case TR, FA, independent precontrast tissue T1 where supplied, author library T1 otherwise, and per-field blood relaxivity. No isotope target enters public data or solver.

## Signal calibration

Define n0 as the integer count of leading precontrast samples. For blood baseline R10=1/T1b, E0=exp(-TR R10), c=cos(alpha), and mean precontrast aortic magnitude Sb, the dimensionless normalized signal is Sn=(S/Sb)(1-E0)/(1-c E0). If Sn>=1 the original interpolates those samples in frame-index space from remaining in-range samples. Then E=(1-Sn)/(1-c Sn), R1=-log(E)/TR, and blood concentration ca=(R1-R10)/r1. Original E=0 convention sets R1 to zero. Any original interpolation is kept and reported rather than silently replaced by clipping. Plasma concentration is ca/(1-H), H=.45. The reference noise model is unweighted residuals in magnitude units, not known Gaussian variance.

Renal nuisance scale A equals mean leading kidney magnitude divided by unit-scale steady-state baseline signal. For R1(t)=1/T1t+r1(Cp+Ct), predict A sin(alpha)(1-exp(-TR R1))/(1-cos(alpha)exp(-TR R1)). Source arithmetic computes n*R1/R1 before the magnitude; preserve that order rather than simplifying it to ensure full nonlinear-optimizer parity. No signal normalization is applied to the fitting objective.

## Two-compartment transport and discretization

The unknown vector is p=(Fp,vp,f,Tt), where Fp and Ft=f Fp have units mL/s/mL tissue, vp and f are dimensionless, and Tt seconds. Tp=vp/(Fp+Ft). The model is dCp/dt=Fp ca-Cp/Tp and dCt/dt=Ft Cp/vp-Ct/Tt with both tissue-normalized concentrations initially zero. Its f=Ft/Fp is not the extraction probability E=f/(1+f). Total tracer balance is input Fp ca minus venous outflow Fp Cp/vp minus tubular outflow Ct/Tt. An independently integrated ODE and integrated mass balance check this convention.

For input J linearly interpolated between times t(i), let x=delta_t/T, E=exp(-x), E0=1-E, E1=x-E0. Normalized exponential convolution g obeys

$$g_{i+1}=E_i g_i+J_i(1-E_i)+\frac{J_{i+1}-J_i}{x_i}(x_i-1+E_i),\quad g_0=0.$$

The compartment concentration is T*g. Apply this once to Fp ca with transit Tp to obtain Cp, then to Ft Cp/vp with transit Tt to obtain Ct. This second application treats the sampled Cp values as piecewise linear, matching the original; it is not claimed to be the exact coupled ODE between arbitrary samples. Independent ODE convergence is tested on a fine grid. Preserve original 1-exp arithmetic rather than silently replacing cancellation conventions.

## Inverse and parameters

Use scipy.optimize.curve_fit with original p0=[.02,.15,.1,120], lower=[0,0,0,30], upper=[.05,.3,.3,infinity], no sigma, default bounded trust-region reflective solver, default finite differences and x_scale=1. In this pinned SciPy environment convergence tolerances are default 1e-8; curve_fit forwards its original defaults. No regularizer, custom penalty, iterative reweighting or extra normalization. Optimizer's built-in interior bound handling applies. Return covariance from curve_fit as a local linear diagnostic, not proven posterior uncertainty. Original library catches failures and returns p0; all bound runs had no failure. Cleaned fit raises failures rather than disguising failed output as reconstruction; this does not alter any bound result.

The first local extraction used vp initial .10 by transcription error, giving similarly fitted signals but different weakly identifiable parameters. Independent initial fixtures exposed it. Correcting to original .15 restored exact full-array equality for all primary and comparison outputs without threshold changes. This demonstrates why signal agreement alone is insufficient.

## Scientific comparisons and clinical reference

The primary selected author example uses steady-state signal nonlinearity and measured tissue T1 where available. Preserve additional full-cohort original/fresh comparisons: linear signal conversion, linear with free arterial delay [0,3] s, and steady-state with literature tissue T1. Comparison defaults match the modern author implementation, not a claim of exact historical paper numerical results. Delay shifts arterial blood concentration with linear interpolation and left zero; all transport and signal stages remain local. The literature T1 comparison sets 1.017 s at 1 T and 1.542 s at 3 T. The linear model uses R1=R10*S/Sb and signal=A*R1. Baseline scale is re-estimated consistently per variant.

Derived MRI GFR=60*Fp*f*volume in mL/min is compared to independent radio-isotope measurements in the notebook. The released data dictionary labels isotope values mL/min, but the author executable example uses those exact values as mL/s and multiplies by 60 before reporting clinical comparisons. We retain original values privately and explicitly follow that author computational convention; the metadata inconsistency is disclosed, not silently repaired. Isotope measurements are an independent clinical reference with their own uncertainty/time separation, not ground truth for Fp, vp, f or Tt. Therefore primary parameter evaluation uses method references, and no isotope-derived calibration is public.

Large transit estimates and active bounds are retained for all cases. A sensitivity profile actively reoptimizes the other three parameters while fixing Tt at a sequence of values for measured weakly/strongly constrained examples. The notebook reports cohort signal errors, all model comparisons, field-strength groups, pre/post scans, clinical bias/dispersion/P30, total GFR and left/total split-function comparisons across all114paired-kidney scans, and limitations. It does not claim a clinically accurate GFR assay or that a finite fitted Tt is identifiable.

## Evaluation

Frozen before extraction: stages absolute/relative1e-10; each Fp/vp/f column scaled RMSE <=1e-4 with scales .05/.3/.3; RMS log(Tt/Tt_ref)<=1e-3; maximum per-curve signal L2 relative error against original <=1e-6; maximum observed-signal relative error <=.2; padding maximum1e-12. The original maximum observed error is .1626, and all cases remain included. Original independent repeat is identical. No threshold was relaxed after local results. These nonzero numerical tolerances cover independent implementation rounding; they are not error bounds against physiological truth.


## Public-contract revision 2026-10-29

2.0-observation-identifiable-renal-class: V1 demanded identity with a weakly identifiable finite optimizer parameter estimate; v2 independently validates all GRE signal/parameter identities and per-case squared-error budgets. An independently fitted 200000-second transit alternative passes; physically valid zero-flow controls fail. No actual truth exists. Full source algorithm and arrays unchanged.
