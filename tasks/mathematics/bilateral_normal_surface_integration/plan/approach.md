# Actual orthographic bilateral integration

## Domain, observations and coordinates

The task uses complete author-provided vase168² and tent128² normal images, decoded from16-bit RGB as q/65535*2-1, float64, with full bool support. No new synthetic observations are generated, no exact analytic depth is provided, and this is not claimed to be real optical data. The raw RGB channels are n_r right, n_g up, n_b toward viewer. The output coordinates are column right, row down, z into scene. Heights and pixel spacing use the same arbitrary pixel-pitch unit, default h=1. The input is not renormalized: quantization near-zero channels remains.

## Explicit discrete acquisition operator

Serialize valid pixels in C row-major order; n is their number. Build CSR(n,n) one-sided matrices Dright,Dleft,Dup,Ddown, each already multiplied row-wise by nz=-n_b and divided by h. Right encodes nz*(z_right-z_here)/h; left nz*(z_here-z_left)/h; up nz*(z_up-z_here)/h; down nz*(z_here-z_down)/h. A missing neighbor yields an all-zero row. Define nx=n_g and ny=n_r, A=stack(Dup,Ddown,Dright,Dleft), b=concat(-nx,-nx,-ny,-ny). This axis naming preserves the validated reference arithmetic. Equivalently the physical tangent equations are n_b ∂column z=n_r and n_b ∂row z=-n_g.

The source keeps the corresponding RHS even for a zero boundary row. Such terms add an unavoidable normal-constraint energy, not an extra height boundary condition. They are deliberately retained and separately identified; no zero-padding observation, reflection, Dirichlet value or periodic neighbor is invented. A constant height is in the operator nullspace. The mathematical tests independently verify signs, masked boundaries, transpose identity and objective gradient rather than only source agreement.

## Objective, inverse updates and stopping

For height vector z, let gu=Dup*z, gd=Ddown*z, gr=Dright*z, gl=Dleft*z. With sigmoid σ(x)=1/(1+exp(-x)), compute vertical weight u=σ(k*(gd²-gu²)), horizontal weight v=σ(k*(gl²-gr²)), and w=concat(u,1-u,v,1-v). Squares and σ are elementwise. The nonconvex energy is E(z)=(Az-b)ᵀdiag(w(z))(Az-b), with no division by pixel count. These are contrast-based bilateral weights, not standard residual IRLS weights; the frozen-weight update omits derivatives of w by design and is not substituted with generic minimization.

Initialize z=0,w=.5 for all4n constraints,Eold=.5*bᵀb. At each iteration solve Aᵀ W A z=Aᵀ W b by SciPy conjugate gradients, initialized at the previous z, with Jacobi inverse preconditioner diag(1/clip(diag(AᵀWA),1e-5,∞)). CG relative tolerance1e-3, absolute default0, maximum5000; nonzero CG status raises rather than falsely claiming convergence. Update weights from the new z, then recompute energy with new weights. Stop once abs(Enew-Eold)/Eold<1e-4 or100 iterations. Canonical k=4 for vase and1 for tent. No height prior or extra regularizer is used. Saturated logistic exponent overflow has the finite limiting weight0/1; local code suppresses that expected warning only, not failures or nonfinite values.

For an exactly flat field b=0 the original relative stop is0/0 and keeps iterating. The local implementation explicitly returns the stationary zero residual after its first solve. This independently tested edge-case correction is documented; the two bound fields have strictly positive energy and take an unchanged path. No true surface is used to resolve gauge: subtract the valid-pixel arithmetic mean after convergence and set outside support to0. Original pre-gauge values are retained only in independent fixtures/provenance.

## Evaluation and actual comparisons

Every output is float64, batch/row/column, zero-mean height in pixel-pitch units. Judge evaluates each map's cosine NCC without centering and RMS error normalized by its own reference dynamic range. Numerical boundaries NCC>=.999999 and NRMSE<=1e-5 were fixed before local extraction; original repeat differences are recorded separately. These are nonzero numerical tolerances for method-reference agreement, not physical accuracy. No evaluator gauge fit, sign, scale or crop is allowed.

The active notebook reconstructs both bound fields and actually runs k={0,1,2,4,8} and inner CG tolerance={1e-2,1e-3,1e-5}. It preserves two-axis weight maps, equal-weight/bilateral maps and surfaces, original/fresh traces, profiles, all directional residual histograms and sensitivity tables. k=0 is the equal-weight smooth member of this same discretization. The selected benchmark isolates this normal-integration method; the paper's six external methods, perspective scenes and full comparative leaderboard are not implemented here, and their historical metrics are not shown as new results. This explicit scope reduction makes no claim of complete paper-benchmark reproduction.


## Current v2 conditional evaluation

The v2 target is the public zero-mean nonlinear bilateral fixed point, not one run's height identity. The independent Judge constructs the four unit-pitch normal-scaled one-sided difference operators. It freezes the candidate's sigmoid weights when testing stationarity. Both vase and tent require energy no worse than attained standard*(1+1e-8), stationarity ratio no greater than max(1e-6,1.1 times attained standard), zero mean within1e-8pixel-pitch units and exact outside-support zero. Standard stationarity ratios are approximately.001901929(vase) and.000998665(tent); these are achieved quality measurements rather than an assertion that the finite standard exactly solves the nonlinear system. Differences and sigmoid sharpness are dimensionless after division by unit pitch. All8unit tests passed. Height-offset and reversed-normal controls fail. No arrays changed.
