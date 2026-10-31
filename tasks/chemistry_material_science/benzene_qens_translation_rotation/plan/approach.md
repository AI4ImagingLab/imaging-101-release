# Implemented mathematical approach

The runtime estimates one 31-dimensional physical-coordinate point under the two-energy isotropic l<=2 model in README. It does not rerun nested sampling or estimate a fresh posterior. The complete original 500-live run provides an independent best sampled point and posterior comparisons. General-purpose constrained point optimization substitutes for sampling when maximizing the same uniform-physical-prior point objective; it does not reproduce the stochastic trajectory or posterior integration.

## Preprocessing and coordinate correction

`load_raw(path)` validates fourteen keys, native precision, shapes and acquisition masks. NaNs are permitted only in the four native signal/error arrays at explicit mask-false locations; all others are finite. A mask is acquisition support, not an analysis cut. `overlap_rebin` implements positive overlap divided by old width, inclusive native Q masks before rebinning, divisors 8/9 and squared overlap weights for independent variances. Only positive weights participate; zero times NaN is avoided. `prepare` then chooses fixed energy indices and inclusive new Q centers, subtracts empty, propagates variances, halves both sample and standard error at 3.60 meV, and normalizes reference-minus-empty by algebraic row sum. No fit-based masking is used.

Native Q centers are exact adjacent-edge means. True energy centers are adjacent-edge means, retaining the exact original 92/169 columns. The original erroneous coordinate xi=1.5 E-deltaE/4 is not exposed as physical E. Since all kernel energy use is through differences, the offset cancels exactly. If c=1.5, gamma_native=c gamma_physical, each Lorentzian and sampled delta scales as 1/c, and their discrete convolution scales as 1/c^2. Thus g,h divide by c and N multiplies by c^2, preserving every predicted observation and the Gaussian likelihood. The 31-coordinate linear pushforward has constant Jacobian c^2=2.25. Original uniform boxes therefore remain uniform physical boxes. We do not claim a second physical-coordinate source sampler execution.

## Local forward operator

`kernel_basis(global_parameters,measurement)` uses g in meV Angstrom^2, h in meV and v in Angstrom^2. For Q in inverse Angstrom, the translational Lorentzian has width gQ^2 and amplitude exp(-Q^2 v/3). The rotational basis has weights j0(QR)^2, 3j1(QR)^2, 5j2(QR)^2 at widths zero, h, 3h, R=2.48 Angstrom. The delta at the exact center is divided by true energy spacing. The inner ordinary convolution is central-cropped to 2n-1; no spacing multiplier is used. Valid convolution with algebraically normalized empirical reference yields n points. Higher-order missing weight is not renormalized. N has units meV^-2/U. The final curve is H/N+b. Numerical signed empirical reference values remain in the operator.

For all 3470 observations, the negative log likelihood up to constants is half J:

```math
J(p)=\sum_{a,Q,e}\left[\frac{\widehat y_{aQe}(p)-y_{aQe}}{\sigma_{aQe}}\right]^2.
```

The group index a denotes the two energies; p is the full physical parameter vector. No division by sample count, regularizer or log-prior term is added inside the uniform box. Parameter bounds are exactly the README physical support.

## Exact conditional nuisance minimization

`profile_energy(H,y,sigma)` sets a=1/N in [1/67.5,1/0.0225] solely as an optimization coordinate. Define weights w=1/sigma^2 and row weighted means Hbar and ybar. Conditional background is clip(ybar-a Hbar,0.001,10). Background regime changes occur at (ybar-b)/Hbar for either background bound b, when Hbar is nonzero. These breakpoints and both a endpoints partition its support. In an interval, free rows use centered H,y; bound rows use uncentered H and y-b. The stationary reciprocal normalizer is

```math
a_* = \frac{\sum_{Q,e} w_{Qe}\widetilde H_{Qe}\widetilde y_{Qe}}
{\sum_{Q,e} w_{Qe}\widetilde H_{Qe}^{\,2}}.
```

The tildes denote those centered/free or uncentered/bound values. Clip this scalar to its interval and evaluate the stable direct squared residual at each candidate and endpoint. If its denominator is zero, use the interval midpoint; all points there have the same quadratic value. Choose the minimum and convert back N=1/a. This finite active-piece enumeration is exact conditional quadratic optimization, not posterior marginalization. The uniform prior is on N; no uniform-a prior or integration Jacobian is substituted.

## Global point inference and stopping

`profile` solves the two independent nuisance subproblems at fixed (g,h,v). `reconstruct` scales these three variables affinely to the unit cube, evaluates the 27 Cartesian points {0.1,0.5,0.9}^3 and takes four with smallest J as starts. Each uses SciPy general L-BFGS-B with a three-point numerical Jacobian, finite_diff_rel_step=2e-5, ftol=1e-13, gtol=1e-5, maxiter=500, maxls=40, closed cube bounds. The lowest attained objective is returned, without claiming proven global optimality. Initialization depends only on public data and support; no source point, fixture or posterior is read. The method reports actual optimizer termination and runtime.

## Evaluation and scientific interpretation

Judge independently implements the operator, including closed sin/cos spherical Bessel formulas rather than the solver special-function call. All three submitted arrays are scored. J<=5661.294967552445 is a pre-local quality budget 1.0001 times original complete-run best sampled J5660.728894662979, not a confidence interval. Each submitted curve must have relative L2 discrepancy <=1e-8 from the public forward of its own submitted parameters. A better fit with different parameters passes. All parameters must lie in declared finite physical boxes. There is no hidden-vector identity test. Source and local results are numerical model estimates, never physical truth.

The 260 K sample retains mobility; the conditional response is not an independently static instrument delta. The rotational truncation omits substantial high-Q angular weight. The classical symmetric approximation lacks detailed balance, and independent cell errors neglect shared-empty/rebin correlations. Boundary v and residual structure must be shown, not converted into a physical accuracy claim.
