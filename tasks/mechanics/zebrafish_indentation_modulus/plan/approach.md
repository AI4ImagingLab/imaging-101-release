# Implemented contact-mechanics inverse

The benchmark binds all 454 original acquisitions with pre-existing manual
quality >=5, the paper's usable threshold. Full approach and retract samples
remain in the public NPZ, but only approach samples enter the elastic fit.
No samples are interpolated or downcast. Acquisition index is not a registered
spatial-image pixel. Units are N, m, N/m and Pa throughout computation.

For each curve, raw force f and measured cantilever height h yield physical tip
position z=h+f/k, with independently calibrated cantilever spring constant k.
This uses raw f before force-baseline subtraction, preserving upstream order.
The detector first clips the force trace strictly before its first maximum.
From the first floor(0.1*n) clipped samples it computes average a and threshold
t=2*max(abs(f-a)). The first sample exceeding a+t is the initial contact index.
If no crossing exists, use half the clipped length. The average force preceding
that contact is the baseline b (or f[0] for index zero). The corrected force is
y=f-b. Reapply the detector to y and subtract the corresponding physical tip
position o to get x=z-o. The returned fitted contact is always physical c=o+c0,
where c0 is the relative fit parameter, not an arbitrary plotted origin.

The published truncated spherical Sneddon law is:

```math
d=\max(c_0-x,0),\quad q=d/R,\quad
P(q)=1-q/10-q^2/840+11q^3/15120+1357q^4/6652800,
\qquad F(x;E,c_0)=\frac{4E\sqrt R}{3(1-\nu^2)}d^{3/2}P(q).
```

E is apparent Young's modulus in Pa; R=18.64e-6 m is the rigid sphere radius;
nu=0.5 is Poisson ratio. Noncontact force is zero after baseline correction.
The model assumes a homogeneous isotropic linearly elastic half-space and no
adhesion. A soft biological tissue violates these idealizations to varying
degrees. Therefore E is an apparent fit parameter and there is no physical
ground-truth modulus file.

The exact spherical relation used independently in physics tests is
d=(a/2)*log((R+a)/(R-a)) and
F=E/(1-nu²)*((R²+a²)/2*log((R+a)/(R-a))-aR), with contact radius a in (0,R).
Tests solve the first relation numerically and compare force at d/R <=0.5;
this is independent of the truncated polynomial being tested.

The residual is r=(y-F)*w, with w=min(abs(x-c0)/s,1), s=2e-6 m. The minimized
objective is sum(r²), no factor 1/N or 1/2 and no regularizer. The weights depend
on the fitted contact. They suppress near-contact artifacts on both sides,
not merely remove negative indentation. The force baseline stays fixed at zero
after preprocessing, matching the selected published configuration (the current
library's default varying baseline is explicitly overridden in the oracle).

Optimization uses SciPy MINPACK Levenberg–Marquardt (`leastsq`). For E>=0 the
unconstrained variable u obeys E=sqrt(u²+1)-1, the original lmfit lower-bound
transform. Initialize E=50 Pa, u=sqrt(51²-1), c0=0. Finite differences use
epsfcn=1e-10, ftol=xtol=1.5e-8, gtol=0, factor=100, maxfev=12000. MINPACK
controls damped steps and stops on its reduction/step criteria; no additional
line search or per-curve tuning is performed. Status outside 1–4 raises an
error. The two-parameter fit runs independently for every acquisition.

Notebook comparisons actively refit all 454 curves with P=1 (parabolic Hertz)
and with w=1 (unweighted spherical). Each method retains the same initialization
and solver; these are labeled ablations rather than replacements for the
reference method. Their unweighted force RMSE is computed from their own
forward predictions, so the lower residual of the unweighted method does not
establish better elasticity inference. Five preselected rejected curves (first
source-key-sorted example at each quality 0–4) illustrate failure of physical
interpretation despite numerical convergence. No trained quality classifier is
part of the inverse task; historical human scores are private context.

Evaluation uses cosine NCC and dynamic-range NRMSE for E, maximum absolute E
error and maximum absolute physical contact error. No alignment, offset removal,
masking or post-hoc scale adjustment is applied. All 454 entries of both outputs
must meet the declared global maximum bounds. Thresholds enforce method
reproducibility, not independent tissue accuracy. Detailed numerical provenance
and tolerance rationale are in test/validation.md.


## Current benchmark contract v2 (2026-10-29)

The v2 evaluator independently profiles one additive force baseline for each approach curve under the public corrected spherical contact law. Its force-range-normalized weighted residual MSE may exceed the attainable fit by at most 1e-6 for each of all 454 curves. The baseline is an explicit measurement nuisance, not an output rescaling; modulus must be nonnegative and contact position must lie in the approach-tip interval. An analytic baseline-shifted contact control and independently improved linear modulus/baseline fit pass. No independent modulus truth is claimed. The cleaned reconstruction algorithm is unchanged.

Earlier method-identity metrics are historical v1 reports, superseded for current benchmark acceptance. Full command stdout, native exits, controls and immutable old assets are indexed in the external benchmark_readiness_20261029 maintainer archive.
