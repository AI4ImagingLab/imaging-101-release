# Implemented conditional mixture estimator

The unknowns are two dimensionless fields on the complete 428-position acquisition: nonnegative RNA mixing weights V and conditional fractions P=V/rowSum(V), each with20 cell-type columns. The full counts and the separate matched-patient single-cell assay are inputs. Neither fitted field is physical cell-count truth. Gene identifiers, type order and native positions are preserved. No spatial interpolation or post-fit support selection is used.

## Input-only preprocessing and units

`reference_basis` removes only reference genes with zero total count. For each type k and reference sample j, form the mean gene counts, multiply by the number of reference cells to obtain group sums, and normalize these sums by the type/sample library total. Average these gene frequencies across available samples and multiply by the mean type/sample per-cell library total. This is the author's frequency-times-cell-size basis. The binding contains one sample group; the result equals the appropriate mean raw counts up to preserved floating arithmetic. The20 types are ordered by their first appearance in reference metadata.

Spatial QC retains genes expressed in more than5 positions, followed by locations whose retained-library total lies between100 and1000000 inclusive. All428 supplied positions pass. Common reference/spatial genes exclude the case-sensitive `mt-` substring. An informative gene has positive reference expression in some type and natural-log fold change greater than1.25 versus the arithmetic mean over the other19 types, using the source's1e-6 logarithm offset. For the union of informative common genes, compute unbiased within-type variance divided by within-type mean, average over finite type values, and retain genes strictly below the linear-interpolated99th percentile. All7050 selected genes are sorted with case-insensitive ASCII collation, matching the original R C.UTF-8 ordering on these names. This convention was validated against native gene IDs; naive case-sensitive sorting exchanges30 C1orf/C1Q rows.

For selected count matrix C, set X(g,i)=C(g,i)/sum_g C(g,i). Match the basis rows to identical genes. Independently multiply X and B by0.1/mean(X) and0.1/mean(B). Both resulting matrices are dimensionless, with means0.1. The forward model is X=B V^T+E. This Gaussian normalized-expression approximation is an inverse model, not a claim of constant Gaussian raw-count noise.

Subtract minima from the native x,y coordinates and divide both axes by their largest combined extent. The resulting positions r preserve aspect ratio. Define W(i,j)=exp(-||r(i)-r(j)||²/(2*0.1²)) off diagonal and W(i,i)=0. Let d=W1, D=diag(d), and L=D-phi W. All degrees are positive, and each phi in the declared grid is less than1; therefore L is positive definite. Its row sums are (1-phi)d. No periodic or image-edge boundary convention is introduced.

## Objective and qualified correction

Let G=7050, N=428, K=20, s be expression residual variance, b be the K-vector of intercepts, lambda the positive K-vector of CAR variances, alpha=1, beta=N/2, and z(k)=V(:,k)-b(k)1. The complete published supplementary Eq9 score is

```math
\ell=-\frac{GN}{2}\log s-\frac{\|X-BV^{\mathsf T}\|_F^2}{2s}
-\frac N2\sum_k\log\lambda_k-\frac12\sum_k\frac{z_k^{\mathsf T}Lz_k}{\lambda_k}
-\sum_k\left((\alpha+1)\log\lambda_k+\frac\beta{\lambda_k}\right)
+\frac K2\log\det L.
```

The first terms are Gaussian expression fit, the next terms are the CAR field density, and the last sum is the inverse-gamma prior. The determinant term follows independently from det(lambda(k)L^-1)=lambda(k)^N/det(L). It is constant with respect to the fitted variables at one fixed phi, but varies across phi. `CARDref.cpp` omits it in `Obj`; the author R wrapper compares `Obj` across phi. On the actual data this selects0.99, whereas the complete score selects0.7. This package explicitly corrects **cross-phi selection only**. Both complete qualified native reruns preceded corrected local selection; all seven fixed-phi fits are bitwise identical to the earlier frozen unmodified outputs. This is a qualified source reproduction, not an unchanged reproduction of the wrapper's selection.

For fidelity, fixed-phi fitting retains the original conditional score without its constant determinant term, including that score's relative stopping normalization. No update or convergence threshold was changed when correcting selection. `log_objective` independently evaluates direct residuals for physical checks; the solver retains the source's trace expansion for floating arithmetic.

## Explicit inverse updates

Fit all phi values (0.01,0.1,0.3,0.5,0.7,0.9,0.99) independently. Initialize V(i,k)=1/K, b=0, s=0.1 and lambda(k)=10. Uniform initialization is an explicit public API configuration; it is not observation calibration. A separate original R fixed-seed Dirichlet initialization is retained as a labeled comparison.

Precompute B^T X, X^T B, B^T B, d, D and L. Initially set T=(V-1b^T)^T L(V-1b^T), calculate the conditional objective, and save V_old. At each iteration:

```math
b=\frac{V^{\mathsf T}L\mathbf1}{\mathbf1^{\mathsf T}L\mathbf1},\qquad
\lambda_k=\frac{T_{kk}/2+\beta}{N/2+\alpha+1}.
```

The lambda update deliberately uses the **previously stored** T, centered with the preceding intercept, exactly as the source orders its arithmetic. It does not silently substitute a newly recomputed quadratic after updating b. Form A1=s(DV+phi*d*b^T) and A2=s(phi*WV+d*b^T) once, then update columns k sequentially:

```math
V_k\leftarrow V_k\odot
\frac{\lambda_k X^{\mathsf T}B_k+(A2)_k}
{\lambda_k\left[V_k(B^{\mathsf T}B)_{kk}+
\left(V(B^{\mathsf T}B)_k-V_k(B^{\mathsf T}B)_{kk}\right)\right]+(A1)_k}.
```

Here the operations and division are componentwise, Bk is column k of B, and the matrix product in the denominator uses already updated preceding columns. The subtraction/addition ordering of its diagonal term is preserved. Then compute

```math
R=\mathrm{tr}(XX^{\mathsf T})-2\mathrm{tr}(B^{\mathsf T}XV)
+\mathrm{tr}\!\left((B^{\mathsf T}B)(V^{\mathsf T}V)\right),\qquad s=R/(GN).
```

Recompute T with the new V and b, evaluate the conditional objective, and record iteration, objective, RMS weight change and s. Stop after more than5 iterations if the RMS change from V_old is below1e-4, or the objective increases and twice its absolute change divided by the absolute sum of old/new objectives is below1e-4. The source also tests NaN; local code raises rather than returning a nonfinite trajectory. Maximum iteration count is1000. If a stopping condition is false, update saved objective and V_old; if true during the first5 iterations, preserve the source's saved-state convention. The bound uniform fits stop after48 updates.

Add K/2 logdet(L) to each converged conditional score. Choose the **last** maximizer, yielding phi0.7 on this binding. Return its V and P=V/rowSum(V), both with a leading batch dimension. Positive weights are not clipped or thresholded.

## Evaluation and active scientific comparisons

The hidden comparison is the qualified original standard-method result, not cell composition truth. All20 types in both outputs have nonzero norm and range. For each type separately require NCC>=0.999999, dynamic-range NRMSE<=1e-6, and maximum absolute error<=1e-6. These unchanged prelocal thresholds exceed independently measured float32 storage errors (maximum absolute2.97e-8) while remaining sensitive to rare-type fields. Original independent reruns are bitwise identical; these thresholds are not a noise-confidence interval.

Also require fractions to agree with normalized weights to2e-6 and the relative excess direct forward SSE to be<=1e-6. The standalone Judge independently loads input-derived model calibration, never solver code. Actual truth takes precedence if supplied; an invalid truth asset errors without fallback. No actual truth is shipped.

The active notebook computes all seven fits, qualified and original ranking, all20 field maps, native parity, same-basis uniform/global/independent NNLS controls, unconstrained gene-mean diagnostic, residuals, independent-reference marker maps and all20 marker correlations, initialization sensitivity, coordinate-correlation ablations, count-retention sensitivities and basis identifiability. Generated binomial sensitivities use `generate_data.thin_counts` with explicit retention, seed and count units. Original data remain unchanged. Source mismatch, sparse counts, negative/weak rare-type marker correlations and the difference between composition estimates and truth are displayed rather than masked. The selected CARD task does not claim a new reproduction of the paper's six separate domain packages or its unbound high-resolution imputation experiments.


## Current benchmark contract v2 (2026-10-29)

All seven CAR candidates use the complete determinant-normalized likelihood and inverse-gamma prior, with intercept, residual variance and type variances profiled exactly. Current acceptance evaluates negative profiled score per gene-spot (excess <=1e-6), nonnegative weights and proportions relation <=2e-6; historical per-type identity is superseded. Independent direct posterior and nuisance-stationarity controls pass; an improved conditional scale passes and tenfold weights fail. No mixture truth is supplied.

Historical pixel/parameter identity thresholds and prior notebook reports refer to v1. The external benchmark_readiness_20261029 archive preserves old README/Judge/metrics/tests and actual current CPU command logs, including failed diagnostics. Source algorithms and arrays were not changed.
