# Implemented mathematics and numerical decisions

## Conditional acquisition model

The task reconstructs unit-density T2-bin masses, in AU, on the complete author ex vivo plane. Echo observations are the arithmetic mean of all four original float64 magnitude acquisitions. Every one of the 2629 frozen estimation-mask pixels is fitted; the 1752-pixel display mask has no role in the inverse objective. Original float32 B1 scale and float64 independent T1 remain visible. The fixed 100 cluster labels and (T1_ms,kappa) centers depend only on independent calibration. Unstandardized clustering has maximum absolute kappa discrepancy 1.6641494558781993, including 1.095176070703367 on the display region; this is not B1 measurement precision. No pixel is filtered after fitting.

There are B=100 bin coordinates t(j)=10*(3000/10)^(j/99) ms. Each coefficient x(j)>=0 is an AU bin mass, not a density per ms or per log-ms. The dictionary has shape (100 clusters,32 echoes,100 bins), with no column normalization. Its columns are the source-defined magnitudes of unit-density EPG echoes. The mixture is y=A*x, not magnitude of an arbitrary complex sum. No bin widths occur.

`rf_matrix(angle,phase)` independently constructs the Cartesian right-handed Rodrigues rotation and changes basis with C whose rows are (1,i,0),(1,-i,0),(0,0,1). Thus its input/output components are F+=Mx+iMy, F-=Mx-iMy and Z. The axis n=(cos(phi),sin(phi),0) is set by phase phi; angle is kappa times nominal flip. Kappa never scales phi. V1 source did scale phi; that isolated error is preserved only in the fully run historical comparison. V2 changes only that phase factor and has independent 1024-isochromat Bloch/Rodrigues validation across 21 parameter cases and all 32 echoes.

For Fourier order k=-128,...,128, free evolution over h=4.6ms multiplies F+ and F- by exp(-h/t) and Z by exp(-h/T1), then adds 1-exp(-h/T1) at k=0. Initial Z0 is 1-exp(-1000/T1) and all other components vanish. Excitation has nominal flip pi/2 and phase zero. Each of 32 echoes applies free evolution, order shifts (+2,-2,0), refocusing nominal flip pi at phase pi/2, the same shifts, then free evolution. `np.roll` implements the original cyclic array shifts, but the +/-128 range prevents occupied-state wrapping for this train. The complex F+0 signal is optionally returned; the inverse dictionary uses its magnitude. The model omits exchange, diffusion, intrapulse dynamics and pre-excitation transverse components, and approximates longitudinal initialization rather than simulating a full periodic steady state.

## Objective and initialization

The selected standard is the paper's classical RNNLS comparator, with explicit RF-phase correction and documented MATLAB-to-Octave runtime adaptation. It is not the DESS/PERK learning branch. The task objective is

```math
J_v(x)=\|A_vx-y_v\|_2^2+\lambda\|x\|_2^2,\qquad x\geq0.
```

The subscript v indexes the 2629 fitted pixels; A(v) uses that pixel's fixed cluster. All echo weights are one. There is no 1/2, per-echo/per-voxel mean, or bin-width factor. Lambda=0.003719667767540722 is dimensionless in the fixed AU coordinate. The source takes the nominal 2^-13 weight times the square of the first-echo median 5.5201012990427625, using pixels strictly above 0.05 times the global estimation-mask maximum. That extra source scaling is disclosed as a paper/code difference. It is computed globally, not separately by cluster or parallel chunk. Source input/output AU coordinates are preserved.

`active_set_nnls` starts from x=0 and an empty passive set P. It forms G=A-transpose*A+lambda*I and b=A-transpose*y. The dual improvement vector is w=b-G*x. Add the inactive variable with greatest positive w if it exceeds 1e-11*max(1,max(abs(b))). Solve the passive normal equations for a candidate z, setting inactive entries to zero. For ridge>0 use a dense linear-system solve. If every passive candidate is positive, accept z. Otherwise take the largest feasible segment step

```math
\alpha=\min_{j\in P,\ z_j\leq0}\frac{x_j}{x_j-z_j},\qquad x\gets x+\alpha(z-x).
```

Entries <=1e-13 are set to zero and removed from P, followed by another passive solve. The line step preserves nonnegativity. The algorithm terminates when no inactive dual improvement exceeds the stated scale-dependent tolerance; a 1000-update safety limit raises an error rather than silently returning an unfinished estimate. The unregularized notebook comparison sets ridge=0 and uses a general least-squares solve of A restricted to P instead of normal equations. Its nonunique coefficients are not judged by strict reference identity.

`solve_distribution` independently applies these updates to every fitting pixel. The only numerical primitives are NumPy matrix operations, small Rodrigues rotations, dense solves and least squares. There is no domain solver, external operator, trained checkpoint or cross-task import.

## Output and scientific evaluation

The standard entry point returns the four declared batch-first float64 arrays. Total density is the sum over all bins, including T2<=15ms. Myelin fraction is the sum for 15<T2<=40ms divided by total, with the zero-total convention zero. Full predicted curves are A*x for all 32 echoes. Every output is exactly zero outside the estimation mask; display clipping is reserved for plotting. Source upstream original outputs clipped map displays, so the private capture stores the absolute coefficient field before that display operation.

There is no actual distribution truth. The hidden baseline is the complete independently executed V2 author RNNLS result, not fresh local output or a cached publication plot. Global source J=118.33721425358966 AU squared, data-only SSE=96.26406644624548. The pre-local frozen total bound is 118.3373325908039; a one-part-per-million objective allowance is paired with a per-pixel primal-dual certificate. For u=A*x-y and g=A-transpose*u,

```math
D(u)=-\|u\|_2^2-2y^{\mathsf T}u-\|\min(g,0)\|_2^2/\lambda,
```

and the stable gap, computed without subtracting nearly equal objective values, is

```math
J-D=\|\sqrt{\lambda}x+\min(g,0)/\sqrt{\lambda}\|_2^2+2x^{\mathsf T}\max(g,0).
```

No clipping of this gap is used. The maximum gap/max(J(v),1e-6) must be <=1e-6. The nonnegative gap certifies optimization suboptimality; strong convexity also gives the bound below, where x star denotes the unique nonnegative minimizer for this pixel under the stated calibrated operator and positive lambda. It does not certify tissue physics or myelin truth. Independent analytic diagonal cases cover nonoptimal, boundary-active and zero observations. Source maximum scaled gap is 1.0842120065467846e-20.

The standalone Judge uses independent closed-form RF matrices for excitation and phase-pi/2 refocusing, without importing `src`. It checks every mass, total, fraction and full curve, using the same stated objective convention and strict finite JSON-number config validation. Each derived output has maximum discrepancy/max(1,absolute expected) <=1e-8; outside-mask maximum <=1e-8. Actual truth takes precedence if present, otherwise the validated method reference is used. No truth is supplied for this task. Reference quality is verified before submission scoring. There is no parameter rescaling or alignment.


```math
\lambda\|x-x_\star\|_2^2\leq J(x)-J(x_\star)\leq G.
```

## Notebook comparisons

The active notebook reruns complete V2 ridge and unregularized local inversions. It separately loads full author V1 NNLS/RNNLS and corrected V2 NNLS/RNNLS measurements. V1 is a documented incorrect-phase historical control, not an unchanged accepted physical model. Source means, pixel curves and mass spectra are identified as original-run measurements; fresh local values are recomputed. No histogram, map, method or curve is fabricated or filled from a cache under a fresh label. The five figures retain all six groups and show acquisition/calibration, full-plane fractions, fixed spatially selected curves/spectra, calibration-error and residual diagnostics, and display/halo spectra. All fitting pixels remain in metrics and distributions.
