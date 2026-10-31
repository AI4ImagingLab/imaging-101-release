# Benchmark acceptance revision 2 — 2026-10-29

The reference algorithm remains historical local nonconvex SLPRU described below. Acceptance now assesses the separately declared global convex target: normalize interior Y/255 by its RMS to B, minimize generalized KL(B||MU)+0.1 nuclear(U)+0.1 sum row-L2(U), U>=0. The submitted max-one field has one missing common scale profiled analytically. This is an explicit semantic change, not equivalence to inverse-singular-value reweighting. Digital intensities are not calibrated Poisson counts. The independent Judge imports no reconstruction functions or hidden maps and retains the existing observation residual and physical/nonnegative/max-one/border constraints.

Objective-per-datum limit0.07310834324578482 equals the feasible independently reproduced standard field value0.07310833324578482 plus1e-8 numerical allowance. Independent per-pixel NNLS initialization followed by general L-BFGS-B convex fitting yields0.07229811631277298 and observation error0.3318427636271077, passing without matching SLPRU. This control reaches its500-iteration budget; no global convergence certificate is claimed or needed to verify its feasible quality. Constant and reversed-spectrum controls yield0.5196790091932176 and0.8628110268585083 and fail; zero also fails. This is an explicit attainable regularized quality bound, not a broadened image-agreement tolerance.

Private evidence under benchmark_readiness_20261029: ecoli_metrics_v1.json, ecoli_v2_controls.json/log, ecoli_pytest_v2.log, ecoli_v2_metriccell_report.json. Full tests:22 passed in25.52s. A scale-profiling unit control verifies invariance to common gain. Algorithm and original arrays are unchanged. Notebook diagnostics are labeled historical; its current evaluator call retains report/exit API and was exercised on the existing reconstruction. Fresh blind agent reconstruction was not run.

# Implemented mathematics

The benchmark fixes measured digital fluorescence Y with 24 channels and a separately measured calibration M of size24×13. M is not estimated from the mixed target. Output A is a nonnegative relative brightness field with one global max normalization. Neither M nor A is a concentration fraction. This implementation reproduces the author's SLPRU code at the recorded revision, not a different generic optimizer.

## Discretization and preprocessing

Read uint8 without downcasting, then divide by255 in float64, matching MATLAB im2double. For each interior pixel, take its3×3 neighborhood and flatten the two spatial axes in MATLAB column-major order to a24×9 window Y. The center is column4 in zero-based indexing. The original checks whether the center has any nonzero observed channel; a zero center is skipped even when neighbors are nonzero. Exterior pixels are zero. Pixel pitch is9.9521491860627815e-8m, but the local spectral operator uses stored control columns directly; no Fourier, interpolation, wavelength resampling or pixel-area factor is introduced.

Each window is divided by s=sqrt(mean(Y²)). The original noiseless zero-window case would divide by zero; the wrapper skips such centers and the callable explicitly returns zeros for an entirely zero window. For nonzero windows, write B=Y/s. Initialize U=max((MᵀM)⁻¹MᵀB,0), where U is13×9. Set V1=MU, V2=V3=V4=U and all scaled dual variables D1...D4 to zero. μ=0.01, maximum1000 iterations, ε=1e-4 and tolerance sqrt((3R+C)N)ε with R=13,C=24,N=9.

## Objective and actual reweighting

The count fidelity is sum((MU)-B log(MU)), without factorial constants or1/N normalization. A conceptual split objective adds λ1 times a weighted nuclear penalty, λ2 times a weighted row-group penalty, and a nonnegativity indicator. Selected λ1=λ2=0.1 were fixed before local extraction, within the author's published tuning range. They are method hyperparameters, not acquisition metadata.

The paper motivates convex nuclear and mixed-row norms. The shipped reference code actually recomputes inverse-singular-value and inverse-row-norm weights at each step. This is a reweighted, nonconvex implementation; we preserve it and do not claim convex convergence or monotonic objective values. In addition, the function named PNMF in the original repository has Euclidean NMF multiplicative updates; it is not called here. Supplied control spectra are reused as calibration and not regenerated or described as newly Poisson-estimated.

At each iteration:

```math
U=(M^TM+3I)^{-1}[M^T(V_1+D_1)+(V_2+D_2)+(V_3+D_3)+(V_4+D_4)].
```

The first auxiliary update is the positive root

```math
Z=MU-D_1-1/\mu,\qquad V_1=(Z+\sqrt{Z^2+4B/\mu})/2.
```

For an SVD U-D2=P diag(σ) Qᵀ, set V2=P diag(max(σ-(λ1/μ)/(σ+η),0)) Qᵀ, with η the float64 machine epsilon. For each row x of U-D3, n=||x||2, w=(λ2/μ)/(n+η), a=max(0,n-w), and the corresponding row of V3 is a/(a+w)x. At λ2=0 this is the identity map; the local explicit branch avoids the removable0/0 at an all-zero row. Set V4=max(U-D4,0). Scaled duals update as D1←D1-MU+V1 and Dk←Dk-U+Vk for k2...4. This negative dual sign convention must be paired with the plus D in the U update.

At iterations1,11,21,... store old auxiliary values before the step. After the step, compute primal residual ||[MU;U;U;U]-[V1;V2;V3;V4]||F and dual residual μ||Mᵀ(V1-oldV1)+Σ(Vk-oldVk)||F. If primal exceeds10×dual, double μ and halve all D; if dual exceeds10×primal, halve μ and double all D. Otherwise keep μ. Residuals remain unchanged between these checks, as in the reference. Stop if both residuals are below the tolerance or after1000 steps. Finally multiply U by s, clip negative entries to zero, retain the central column. Normalize the complete collected image by its global maximum. No artificial convergence is claimed for windows reaching the iteration cap.

## Scientific comparisons

NLS, SNLS and SLNLS use the same splitting with quadratic fidelity0.5||MU-B||F², replacing V1 with (B+μ(MU-D1))/(1+μ). NLS uses both penalties zero; SNLS uses only λ2=0.1; SLNLS uses both0.1. These comparisons actively reconstruct every benchmark pixel in the notebook. Additional λ2=0 count-aware ablation tests the sparsity contribution. Each method has its own global max normalization, preserving the upstream convention.

## Evaluation conventions

No independent true abundance is available. The independent original SLPRU result is a method baseline. Cosine NCC and dynamic-range NRMSE compare exact arrays without alignment; numerical tolerances were frozen before extraction. Observation residuals use the fixed interior and one explicitly declared nonnegative global gain because normalization removes scale. This gain does not enter baseline metrics. The processed8-bit intensities are not literal photon counts; report residuals as digital-intensity diagnostics, not χ² or physical accuracy. All thirteen channels are evaluated, not only visually attractive labels.
