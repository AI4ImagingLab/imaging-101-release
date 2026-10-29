# Implemented mathematical approach

## Acquisition and units

The80 observed initial rates y(g,i), groups g=1..8 and concentrations i=1..10, are the original ONPG dataset converted before analysis to micromolar/min. Concentrations s(i) are mM. Native groups3–6 are multiplied1000; no other preprocessing, smoothing, pooling or outlier removal occurs in the primary path. Temperature, enzyme loading and original optical traces are unavailable; therefore only effective assay-specific Km and Vmax are inferred. There is no spatial discretization, boundary operator, noise variance, or regularization.

## Forward map and loss

The model f(i)=V*s(i)/(K+s(i)) has K in mM and V in micromolar/min. Residual r(i)=y(i)-f(i); objective E=sum r(i)^2 is unnormalized, unweighted, in squared rate units. The Jacobian J with respect to (K,V) has columns -V*s/(K+s)^2 and s/(K+s). Thus the loss gradient is -2*J.T*r. The original vignette's displayed Km loss derivative has an inconsistent sign; local code uses the mathematically correct derivative and independent finite differences verify it. Physical limits f(0)=0, f(K)=V/2 and f(infinity)=V are tested.

## Initialization and upstream discrepancy

For every pair i<j, let m(i)=y(i)/s(i). The original ECB initializer computes Kij=(y(j)-y(i))/(m(i)-m(j)), then Vij=m(i)*Kij+y(j). It takes separate medians, ignoring NaN as original median(na.rm=TRUE), and rounds each median to three decimals. Infinite intersections are retained as in the original. On these bound data all resulting seeds are finite. **The second formula is a known upstream error.** The correct geometric intersection is Vij=m(i)*Kij+y(i). This error is independently demonstrated by exact two-point data and by loss of permutation invariance. The primary method preserves the erroneous original initializer for upstream parity; it never uses that formula in the physical forward model or final NLS solution. Notebook labels the original ECB comparison as buggy and separately calculates corrected geometric ECB. That corrected version is an explicitly different diagnostic, never the primary method or a replacement for an upstream result.

## Direct inverse updates

At current p=(K,V), perform reduced QR decomposition J=Q*R. Let q=Q.T*r and rperp=r-Q*q. R nls's default relative offset stopping criterion is c=norm(q)/norm(rperp), with scaleOffset0. Stop when c<=1e-5. Solve R*delta=q; use pnew=p+alpha*delta. Starting alpha=1, accept only finite predictions with nonincreasing E; otherwise halve alpha down to1/1024. At the next iteration restore alpha by at most a factor2, capped at1. A failure to find a step or convergence by50 updates raises an error. Analytic derivatives replace R's numericalDeriv; objective, initialization, update family, stopping and reporting conventions are preserved. There are no bounds in original nls, no parameter priors, and no regularizers. The selected native seeds yield positive physical final parameters for all8 groups. Remote starts can converge to unphysical negative-K rational branches, disclosed in independent diagnostics, so no global optimization claim is made.

Once converged, round K and V to three decimal places (author convention) and recompute the predicted observations from those rounded parameters. A convergence history stores unrounded K,V,E,c for review. There are no stochastic operations or seeds. Full96 original comparisons matched all rounded parameters exactly on first local execution; the pre-frozen tolerance remains0.002 for direct parameters and curves, rather than being changed to match the observed result.

## Required transformed comparisons

All methods run independently for all8 groups and for two scenarios: full observations (primary) and the author vignette deletion of group2's eighth point (comparison only). The six method axes are direct, LB, weighted LB, original buggy ECB, Hanes-Woolf(HW), Eadie-Hofstee(EH). Negative or poor transformed estimates are retained and plotted with failure diagnostics.

For LB, x=round(1/s,4), z=round(1/y,4). Fit z=a+b*x by least squares with intercept. Weighted LB uses w=1/z^4 after this original reciprocal rounding. Multiplying all weights by a common positive constant does not affect the fit; local code rescales by max(w) for numerical stability. Set V=round(1/a,2), K=round(V*b,2), preserving the sequential rounding. HW uses x=1/round(1/s,4), z=x*round(1/y,4), fits z=a+b*x, then V=round(1/b,2), K=round(V*a,2). EH uses unrounded x=y/s,z=y, fits z=a+b*x, then K=round(-b,2),V=round(a,2). Linear least squares uses an explicit design matrix and NumPy's general lstsq, with weights applied as square roots. There are no upstream domain imports.

## Diagnostic inference and evaluation

For a grid of positive K, set a(i)=s(i)/(K+s(i)). The optimal conditional V=sum(a*y)/sum(a^2); profile E(K)=sum(y-V*a)^2. This eliminates the linear parameter exactly and exposes finite positive-K minima without imposing new priors. Local linearized covariance is (E/(n-2))*inverse(J.T*J), a homoscedastic approximation, not calibrated biological uncertainty. The Notebook also recomputes all80 leave-one-observation-out direct fits, explicitly records any solver failures, compares author deletion and corrected ECB, and checks the original bug on exact independent data.

The standard output is K(1,8),V(1,8),f(1,8,10), all float64. Judge compares each parameter with maximum relative error<=0.002, all predicted rates with maximum absolute error<=0.1micromolar/min, and output forward consistency<=1e-8micromolar/min. No masks, alignment or rescaling. Hidden reference is the independent original standard method, not ground truth. These practical Judge thresholds do not replace tighter mandatory original parity tests or physical checks. No biological accuracy is claimed from reference agreement. Undefined/invalid input metrics are errors, not passes.
