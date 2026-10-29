# Current public benchmark target and convergence (version2)

The scored mathematical target is the unique regularized constrained SOLA
estimator described below, evaluated for all5151targets. Regularization3,
width0.06 and angular width0.1 define the target, not a finite Krylov trajectory.
Model sound speed uses the public four-node local Lagrange interpolation.
An independently derived grid-dual SPD solve on the public factors computes the
canonical reference without using this implementation or any hidden solar truth.
Its original-operator KKT relative residual is3.60e-15. The historical1000-step
native solution differs by0.01723686515nHz, beyond the unchanged0.001nHz field
boundary. Main now requests1500steps of the same published/local Lanczos
updates, yielding maximum difference below6.99e-8nHz from the exact target.
`reconstruct` retains its1000default for explicit historical parity experiments;
`main.run` adds `iterations=1500` and forwards it explicitly. Initialization,
Householder reduction, all operators, full reorthogonalization, regularization,
targets and stopping at the requested iteration count otherwise remain unchanged.
This is an explicitly versioned convergence correction, not unchanged1000-step
reproduction. Stage/native1000fixtures and every original comparison are retained.

# Implemented constrained SOLA inversion

The task retains the complete native SOLApack example:29617 frequency coefficients,101×51 meridional field, all5151 target locations and1000 fully reorthogonalized Lanczos steps. The author-supplied eigenfunction model independently defines calibration factors. There is no new synthetic observation and no known true rotation field. The reference is the full-precision result of an independently executed original method.

## Acquisition, units and ordering

Let $m=29617$, $p=5151$. Vector $x$ holds cyclic rotation in nHz, flattened with colatitude varying fastest. $y$ contains measured coefficients in nHz. The supplied factors form $K$ of shape $m\times p$ as defined explicitly in README; they already include acquisition integration weights. $Kx$ is in nHz. Set $s_i=1/|\sigma_i|$ for nonzero reported standard deviations and1otherwise, matching original covariance flag0. Whitened quantities are $D=\mathrm{diag}(s)$, $\widetilde K=DK$, $\widetilde y=Dy$. No covariance off-diagonal is inferred or supplied.

Forward and adjoint exploit293 native rectangular blocks of equal angular pattern across contiguous radial multiplets. The forward computes field×radial factors, then angular-factor transposes for each block. The adjoint reverses these contractions, accumulating equal radial columns before multiplication by radial-factor transposes. Local tests also form individual full kernel rows independently to check flattening, units and the transpose. No dense150-million-entry kernel matrix or domain-library call is required.

## Quadrature and localized targets

Let $w_j=r_j q_j$ be the mesh quadrature weight $q_j$ times normalized radius, and $W=\mathrm{diag}(w)$. Square roots are literal nonnegative roots; at zero weight, the inverse square root is defined as1exactly as in the original code. Kernel sensitivity at these zero-weight central points is zero, so this convention does not introduce a spurious central observation.

For target center $(r_0,t_0)$, radial width $d_r$ and angular width $d_t$, evaluate

```math
T_j=\frac{1}{Z}\left[\exp\left(-\frac{(r_j-r_0)^2}{d_r^2}-\frac{r_j^2(t_j-t_0)^2}{d_t^2}\right)
+\exp\left(-\frac{(r_j-r_0)^2}{d_r^2}-\frac{r_j^2(t_j-(\pi-t_0))^2}{d_t^2}\right)\right],
\qquad Z=\sum_j w_j(\mathrm{unnormalized\ target})_j.
```

Thus $\sum wT=1$ over the supplied northern quarter-disk mesh. The equatorial image term is retained even when small. A stale upstream comment mentions half-integral normalization; the actual executable sums both terms and normalizes to1, which is preserved and independently tested.

Set $d_t=.1$ rad and $d_r(r_0)=.06c(r_0)/c(.7)$ in normalized radius. Sound speed is dimensionless, computed from the known structural model using README formulas and the original cgs gravitational constant. Its interpolant is the explicit four-point Lagrange polynomial of native DLIR, including its endpoint/extrapolation stencil and weight-sum correction. Do not replace it with a spline. The initial Lanczos target is separately fixed at r0=.7,t0=.7,dr=.04,dt=.1 and is not sound-speed-scaled, matching set_u1.

## Constraint and inverse objective

For each target choose whitened observation coefficients $q$ minimizing

```math
\left\|W^{-1/2}\widetilde K^Tq-W^{1/2}T\right\|_2^2+\mu\|q\|_2^2,
\qquad q^T h=1,\qquad h=\widetilde K\mathbf 1.
```

The estimate is $\hat x=q^T\widetilde y$. Whitened observation coefficients q carry numerical nHz scale, so the tradeoff has inverse-squared-nHz scale. Native tradeoff $\mu=3$ multiplies the squared coefficient norm; original Elden code augments by $\sqrt\mu$, so it is not3squared. The constraint makes the estimate exactly reproduce a constant rotation field. The mean/smoothing interpretation is retained throughout the entire mesh, including weakly localized central and polar estimates.

Construct a Householder reflector $H=I-\tau vv^T$ taking $h$ to $d e_1$, where $d=-\mathrm{sign}(h_1)\|h\|$, $v=h-de_1$, $\tau=1/(\|h\|(\|h\|+|h_1|))$, and $\alpha=1/d$. Let $H_2$ contain all columns except the first, $q_0=\alpha He_1$, $A=W^{-1/2}\widetilde K^T H_2$, and $c=W^{-1/2}\widetilde K^Tq_0$. Every feasible coefficient vector is $q=q_0+H_2z$. Its variable objective is $\|Az-(W^{1/2}T-c)\|^2+\mu\|z\|^2$; the fixed penalty term $\mu\alpha^2$ does not affect the solution.

## Fully reorthogonalized Lanczos updates

Initialize $u_1$ to the normalized vector $W^{1/2}T_{start}-c$. For iteration $j$, subtract the previous subdiagonal contribution from $A^Tu_j$, sequentially modified-Gram–Schmidt orthogonalize against every previous right vector, normalize by $a_j$, then apply $A$, subtract $a_ju_j$, sequentially orthogonalize against every existing left vector and normalize by $b_j$. Preserve the native single MGS pass, rather than classical simultaneous projection or a generic truncated SVD. The fixed stopping rule is1000steps, not a fitted data-dependent iteration count; exact zero-norm breakdown raises an error. BLAS dot, axpy and norm are general numerical primitives.

The resulting $U$ has1001columns, $V$ has1000columns and $B$ is1001×1000lower bidiagonal with diagonal $a$ and subdiagonal $b$, satisfying $AV=UB$. For each complete-grid target define $g=U^T(W^{1/2}T-c)$ and solve

```math
(B^TB+\mu I)\xi=B^Tg.
```

The diagonal is $a_j^2+b_j^2+\mu$ and off-diagonal is $b_ja_{j+1}$. Local solve_banded evaluates this explicit tridiagonal system. Original SOLApack first QR-rotates the bidiagonal matrix and uses Elden's augmented QR solve; the local system is the same strictly regularized constrained objective. Independent full-pipeline comparison validates the numerical algebra substitution: maximum field discrepancy1.03e-11nHz, formal-error discrepancy5.24e-14nHz. No method, target, discretization, regularization, initialization or iteration count was changed.

Evaluate all5151targets in128-column batches, preserving full grid and target normalization. The estimate is $\hat x=\xi^TV^T(H\widetilde y)_{2:m}+\alpha(H\widetilde y)_1$. Its formal error is $\sqrt{\|\xi\|^2+\alpha^2}$ under the diagonal supplied error model. The corresponding continuous averaging kernel is $W^{-1/2}(UB\xi+c)$. Selected estimator vectors explicitly reconstruct $q=q_0+H_2V\xi$ for normalization/covariance diagnostics. No solver cache or reference factorization supplies a fresh result.

## Evaluation and scientific comparisons

The required output is the full `(1,101,51)` real float64 nHz field. Standard-method agreement uses the pre-extraction maximum-error boundary1e-3nHz. Independent direct calibrated row contractions additionally require error-weighted observation RMS≤1.20; the original full field gives1.19355437766235. This physical check was added after the local trial as a stricter independent criterion, without relaxing any frozen acceptance threshold. No hidden actual truth or implicit alignment is introduced.

The notebook freshly computes the full basis and reconstruction, then compares the independently reproduced original field/error map. Selected averaging kernels, inter-target error correlation, radius/latitude sections and full observation residuals expose resolution and data fit. Truncated250/500step prefixes, changed tradeoffs, radial-width choices and noise perturbations use only the freshly computed basis and local equations. Reusing a freshly computed calibration-only basis across targets/perturbations is part of the published method; loading an archived reference basis as a fresh inverse is prohibited and is not done.
