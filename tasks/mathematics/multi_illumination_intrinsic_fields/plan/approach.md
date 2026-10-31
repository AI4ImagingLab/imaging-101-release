# Implemented relative intrinsic-image reconstruction

The selected method is the parameter-free multi-image Weiss baseline distributed with the ICCV2009 intrinsic-image benchmark, not a newly trained model. The native 600×534 dinosaur acquisition is fixed before any inverse run. Ten grayscale illumination images and one grayscale diffuse reference image are the primary observations. All grayscale values are arithmetic RGB means of uint16 PNG/255 values. They must not be divided by65535 or rescaled to[0,1]. The support is the independent author mask, containing70430 pixels in one connected component.

## Physical unknowns and ambiguity

For pixel p and light k, intensity is approximately Iₖ(p)=R(p)Sₖ(p)+eₖ(p), with relative reflectance R, light-dependent shading Sₖ, and mismatch eₖ. Reference-image shading is S₀. Logarithms convert multiplication into addition on positive intensities. The method estimates spatial log-reflectance gradients by assuming illumination gradients have approximately zero median across lights. This assumption can fail for persistent shadows or changing specularities. Independent measured reflectance and shading remain relative measurements, not absolute spectral albedo or calibrated irradiance.

The multiplicative gauge is fixed by mean(log R)=0 over the foreground. Hence geometric mean R=1. The exact reference-image product R S₀=I₀ is maintained. This does not imply perfect independent-reference agreement; the reference experiment itself has modeling and measurement mismatch.

## Discrete acquisition and target gradients

The source floors each illumination intensity at3 before logarithm, in original PNG/255 units. This is a method constant, not measured detector noise. With j indexing columns and i rows,

```math
G^y_k(i,j)=\log\max(I_k(i,j),3)-\log\max(I_k(i-1,j),3),
```

with the first row zero. The horizontal gradient Gˣ has the analogous previous-column difference and zero first column. The target gradients tʸ,tˣ are the medians of the ten corresponding gradients; even-count medians average their two central order statistics. No RGB chromaticity or hidden measurement is used in this primary path.

## Masked integration and normalization

Let u=log R, and let E contain each horizontal or vertical neighboring pair wholly inside the support. For an oriented edge a→b, D applies u(b)−u(a). The unknown-vector order is foreground row-major order. We minimize

```math
\frac12\sum_{e\in E}w_e\big((Du)_e-t_e\big)^2.
```

Primary weights are all one. There is no division by edge count, extra smoothness regularizer, periodic wrap, target-derived mask, or outside Dirichlet penalty. Explicit assembly places w on both endpoint diagonals and −w on the two off-diagonals; the right-hand side receives −w t at a and +w t at b. Thus A=DᵀWD and b=DᵀWt. Source row-major numbering is preserved, including zero placeholder numbering outside support; those placeholders are never solver unknowns.

The graph is connected. A is symmetric positive semidefinite with the constant nullspace. The implementation solves the gauge-fixed SPD system (A+11ᵀ/n)u=b−mean(b), where n is the support size. The constant projector changes only the gauge, not any gradient equation. The generic solver is SciPy conjugate gradients, zero initialization, Jacobi inverse diagonal1/(diag(A)+1/n), relative tolerance1e-11, absolute tolerance0, and at most20000 iterations. A nonzero solver status raises an error. The result's foreground mean is explicitly subtracted after convergence. The actual primary solve takes2086 iterations on the validated backend; elapsed time is recorded from execution, not promised as a fixed runtime.

The original uses generic Ruge–Stuben AMG at default relative tolerance1e-5. Before any local implementation, a separate original-code1e-10 tolerance diagnostic established the numerical difference due to integration precision, and analytic gradient cases verified the nullspace and signs. Local high-accuracy integration must have range-NRMSE≤1e-7 against this separately labeled source diagnostic and≤0.001 against the original default solve, with default-source NCC≥0.99999. This is the same least-squares method with a documented general numerical backend change, not bitwise AMG reproduction.

Finally set R=exp(u) and S₀=I₀/R within support, with exact zeros outside. Output uses real float64 and batch-first(1,600,534) arrays. No post-hoc intensity clipping or geometric alignment is applied.

## Computed review comparisons

The notebook retains the original family comparisons: grayscale Retinex, color Retinex, Weiss–Retinex, and Weiss with L1 integration. Fixed threshold0.1 is declared before consulting measured targets; it is not selected by hidden-reference tuning. Gray Retinex first floors the reference intensity at3, uses its masked log gradient, and retains components with absolute value strictly greater than threshold. Weiss–Retinex similarly thresholds median gradients. Color Retinex uses full RGB reference observations only for an explicitly different-input contextual comparison, separates mean-channel and chromatic gradients, and keeps a gradient when the chromatic norm or absolute gray gradient exceeds0.1. Its grayscale intensity is the RGB mean after the source RGB floor. Color results are not evidence that a primary grayscale-only solver received RGB input.

L1 integration uses the source initial unweighted solve followed by exactly20 reweighted solves. At each step the row and column weights are1/(2 max(mask·|Du−t|,1e-4)). Only valid interior edges enter the actual objective. The same local integration solver is used; residual L1/L2 and numerical convergence are measured after every update. The source's broad exception handler that silently returns its previous iterate is not required by a successful reference run: local failure raises rather than claiming an unexecuted update succeeded. Higher inner-solve accuracy can alter finite-step IRLS outputs; source-default, source-high-accuracy, and fresh controls are separately identified.

Additional computed experiments use predeclared first2/4/6/10 illuminations, a permutation-invariance check, Weiss–Retinex threshold sweeps, and Gaussian observation stress. Stress noise has known illustrative sigma in PNG/255 units and a fixed maintainer seed; it is not claimed to model measured sensor noise. Negative perturbed intensities alone are clipped tozero, and the standard log floor remains3. All10 original illumination images remain in the bound benchmark.

## Evaluation and physical checks

Independent measured shading and arithmetic-RGB-mean reflectance take priority over method results. Compare the entire fixed support. NCC is uncentered cosine. NRMSE permits one positive least-squares global gain per field before division by that reference's range; this measurement-gain convention is explicit in README and metrics.json. Source LMSE uses20×20 windows, stride10, separately optimized per-window scalar, source1e-5 low-estimate-energy switch, and sums squared residual/reference energy across all windows. It is never relabeled global NRMSE. No spatial shift or local warp is allowed.

Each NCC boundary is0.9 times the reproduced original metric; each NRMSE and source-LMSE boundary is1.1 times the corresponding original error. Both independent measured fields are nonconstant and nonzero. The nontrivial source errors make those rules meaningful; simple all-shading and all-reflectance decompositions are rejected. Submitted product relative L2≤1e-10, foreground absolute mean log R≤1e-10, exact background zeros, and foreground positivity further prevent independent scale cheating. Stage parity, an independent analytic illumination example, normal-equation/adjoint-energy/nullspace tests, and strict standalone Judge tests complement reference agreement. Every threshold was frozen before local implementation; Judge success alone is not physical proof.
