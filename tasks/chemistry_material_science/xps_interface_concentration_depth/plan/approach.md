# Implemented mathematics

The unknown c is a real dimensionless 201-vector on the public -2..20 nm depth grid, spacing0.11nm. All entries are free except nonnegativity. Supplied H is10×201 in µm³, and forward(c,H)=Hc; adjoint(v,H)=Hᵀv. The author-calibrated matrix already includes cylindrical volume quadrature and attenuated escape paths; no new acquisition is generated. Reconstructing H from an incomplete upstream smooth-interface generator is not claimed. It is legitimately fixed across different chemical target profiles.

For K noisy repeats, y=mean(observations,axis0), Γy=(0.01²/K)I10 in µm⁶. D is199×201 with interior stencil [-1,2,-1], no mesh-spacing factor and no imposed endpoints. Γd=(201/10)0.01²p I199 is dimensionless, with prior_factor p=1 in the selected path. This is the general diagonal-Gaussian prior supported by XPSinv, rather than the paper example's endpoint-conditioned correlated prior. The objective is (Hc-y)ᵀΓy⁻¹(Hc-y)+(Dc)ᵀΓd⁻¹Dc+indicator(c≥0), with no factor1/2. The input normalization makes the two whitened terms dimensionless.

Set A=[H;D], b=[y;0], C=diag(Γy,Γd). With τ0=10, L=||A||2, σ0=1/(τ0L²), γ=2sqrt(201)/τ0, initialize c=bar(c)=0.5 and s=0. Every iteration evaluates

    s ← (s+σ A bar(c)−σb)/(1+0.5σdiag(C))
    old ← c; c ← max(c−τAᵀs,0)
    θ ← 1/sqrt(1+2γτ); τ ← θτ; σ ← σ/θ
    bar(c) ← c+θ(c−old).

The source prox follows from the convex conjugate F*(s)=sᵀb+0.25sᵀCs. It specializes exactly to componentwise division because both covariances are diagonal scalar identities. No generic inverse solver replaces the published primal–dual update. Stop only if median(abs(y−Hc)/y)≤1e-4 AND ||c−old||2/||c||2≤1e-9, or after100000 updates. All bound measurements are strictly positive. This acquisition reaches the full budget in both original and cleaned runs. Initial values and priors do not access truth.

Retain initial plus first50 primal/dual/step states for upstream stage parity, and objective/data-fit/step history every1000 updates and at update1. Bounded diagnostic storage changes no mathematics. Diagnostic costs call the same explicit objective. Separate notebook experiments actively recompute prior factors0.25,1,4; repeat counts1,10,199; budgets100,1000,10000,100000; and an independent constrained quadratic optimum using SciPy general-purpose bound-constrained linear least squares (BVLS, tolerance1e-12, maximum2000iterations) with the same locally constructed whitened objective. The latter checks convergence and is never substituted for the selected solver.

Evaluate all201 points with cosine NCC=dot(c,t)/(||c||||t||) and range NRMSE=sqrt(mean((c−t)²))/(max(t)−min(t)), no registration or rescaling. The independent reference is the author's simulated chemical profile. Boundaries are0.9×originalNCC and1.1×originalNRMSE, with poor constant/zero/shifted-profile controls verified. They are unrelated to stage-parity tolerances fixed before local extraction.

The independently solved constrained optimum has objective9.9056790583 versus selected CP10.3938009239, yet worse truth NRMSE0.317495 versus0.0565895. Thus the finite100000-update method has an important implicit iterative regularization effect. Do not claim convergence to the quadratic optimum or treat the lower objective as greater physical accuracy.
