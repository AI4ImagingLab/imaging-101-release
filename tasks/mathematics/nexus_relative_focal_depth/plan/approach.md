# Implemented mathematics

## Conditional observation and unknown

All19 author-registered, native1080×1920 BGR images are used. Pixel intensity I is an8-bit digital number, not a calibrated radiance. The physical defocus relation is a local scene/blur convolution; neither its PSF nor metric lens settings are supplied. The reconstructed u is therefore the dimensionless relative best-focus coordinate, not an absolute physical distance. Frame k, zero-based, has coordinate t=20(k/18−1/2). This uniformly parameterizes frame order, without assuming equally spaced metric distances. There is no true metric depth array, optical gain fit, additional registration or latent synthetic object.

## Focus measurement and normalization

Convert each BGR channel to binary32 by multiplying by the binary32 reciprocal1/255. For each channel c and pixel(row,column), define horizontal and vertical central second differences, with image values replicated at an edge. The modified-Laplacian focus score S is the sum over all three channels of the absolute horizontal second difference plus absolute vertical second difference. Its units are normalized digital intensity per squared unit pixel. Pixel pitch is1in image coordinates. Do not turn this into a signed Laplacian or take an absolute value only after summing channels.

Normalize each pixel's19 scores by multiplying by1/(max(S)+0.3), computed in binary32. The0.3 is a source numerical denominator regularizer, not sensor calibration. It prevents very low-contrast responses from receiving the same weight as textured regions. Preserve the source reciprocal/multiply order.

## Degree8 focus polynomial

The descending monomial design matrix X has rows(t^8,t^7,...,t,1). Build powers by repeated binary32 multiplication, beginning with t; form X-transpose X by binary64 accumulation and serialize it to binary32, as in the original small OpenCV GEMM. The source left inverse is

```math
P=(X^\mathsf{T}X)^{-1}X^\mathsf{T},\qquad
c(\mathbf r)=P\widetilde S(\mathbf r),\qquad
C_{\mathbf r}(u)=\sum_{j=0}^{8}c_j(\mathbf r)u^{8-j}.
```

P is computed locally, never supplied as a captured operator. Its small inverse uses explicit partial-pivoted binary32 Gaussian elimination in the source operation order. The final inverse-times-X-transpose uses binary64 accumulation then binary32 serialization. The original monomial design is ill-conditioned (condition approximately8.93e7; its normal matrix approximately7.97e15 in binary64 assessment). This is a numerical limitation of this selected published port. Replacing it with an SVD or a differently scaled polynomial would change the selected numerical method; the independent source fixtures check the actual computation. Polynomial constant/quadratic cases independently assess meaningful response evaluation, rather than requiring an unrealistic exact coefficient-space inverse identity.

Each coefficient is accumulated over frames in source order. The original CUDA implementation fuses product/add. The local helper computes an exact binary64 product of binary32 factors, adds in binary64, then casts to binary32. This approximates FMA but is not universally identical: the binary64 sum can round, and a rare midpoint tie can produce a one-ULP double-rounding difference. An independently constructed exact-rational counterexample is tested, and source stages/full runs quantify the actual acquisition's difference under the predeclared source-compiler tolerance. The returned derivative coefficients represent **negative** contrast, namely−(8c0,7c1,...,c7). Polynomial evaluation uses the original add-before-multiply Horner order. There is no learned model, checkpoint or domain-library fitting call.

## Objective, discrete operators and initialization

The conditional focus objective is an unnormalized sum over the native image grid:

```math
E(u)=-a\sum_{\mathbf r}C_{\mathbf r}(u(\mathbf r))
+\sum_{\mathbf r}\sqrt{(K_xu)^2+(K_yu)^2},\qquad -10\leq u\leq10.
```

a=6is the source contrast weight; K consists of forward unit-pixel differences with zero terminal component at the last column/row. K-transpose is the exact Euclidean adjoint. For valid split/dual fields whose terminal components vanish, it equals the source negative backward-zero difference. The isotropic TV is dimensionless focal-coordinate change per unit pixel, summed without a pixel-count normalization. Its weight is1; this sets the relative contrast/TV tradeoff. Symbols gx,gy denote auxiliary gradients; bx,by are scaled dual variables, all in focal-coordinate-per-pixel units.

For initialization, smooth each **unnormalized** S frame with a15×15 box, select its first strict maximum focus index k, then smooth that index map with a21×21 box. Both filters use replicate boundaries and source column-first tap order, with the approximate FMA helper described above. This tap order matters when nearly tied responses change a discrete argmax. Exact-source halo cases and full-field numerical parity validate the bound acquisition; universal bit identity is not asserted. The main initializer is20((mean(k)−1)/18−1/2).

**Source discrepancy:** the CUDA port returns zero-based maxima, while its final scaling subtracts1as if indices were one-based. The selected reference explicitly retains that initialization convention, shifted by−20/18relative to correctly zero-based nodes. This does not shift the forward curve or output coordinate system. An independent constant-index case proves the offset. An original-source variant changing only this subtraction was separately executed before extraction and is shown as a qualified diagnostic, never as an unchanged-source run. It reaches a different nonconvex solution, with a slightly higher original objective on this acquisition; neither solution has independent physical depth truth. Initialization is not an acquisition parameter.

## Explicit linearized ADMM

Let tau=8/a, lambda=1initially, and initialize g=b=0. D denotes the negative fitted contrast, D=−C. Each iteration performs

```math
v=(I+\lambda K^\mathsf{T}K)^{-1}
\left[u-\tau a D'(u)+\lambda K^\mathsf{T}(g-b)\right],
\qquad u\leftarrow\operatorname{clip}(v,-10,10),
```

```math
q=Ku+b,\qquad
g\leftarrow\max\left(1-\frac{\tau}{\lambda\lVert q\rVert_2},0\right)q,
\qquad b\leftarrow b+Ku-g.
```

At q=0define g=0. Clipping after the spatial linear solve preserves the source safeguard; it is not claimed to be the exact constrained quadratic proximal solution. The paper's printed split/dual equations contain inconsistent repeated d and sign terms. The source update above independently follows the scaled augmented term(lambda/2)||Ku−g+b||², verified through its adjoint and vector proximal identities. No incorrect printed expression is copied into the executable method.

The Neumann spatial matrix is diagonalized with an orthonormal two-dimensional DCT-II and its inverse. Its eigenvalue at row mode p,column mode q is4sin²(pi p/(2H))+4sin²(pi q/(2W)), for H1080,W1920. Divide transformed right-hand-side entries by1+lambda times this eigenvalue; the identity removes the constant-mode singularity. SciPy provides only the general cosine-transform primitive. Dense independent small-grid normal equations and nullspace tests check signs, factors and boundaries.

After zero-based iteration index greater than0, multiply lambda by binary32(1.02), then multiply b by binary32(1)/binary32(1.02). This preserves the selected scaled dual under penalty growth. There is no line search, gradient optimizer, automatic stopping or claimed global minimizer. The selected benchmark executes exactly400iterations. Source scalar multiplication and addition order and binary32 state storage are retained; independent original FMA-enabled/disabled whole runs define a nonzero numerical tolerance before local scoring.

## Comparison and evaluation conventions

The primary output contains all1080×1920 finite u values in batch-first form. Baseline agreement uses the independently computed unmodified source result, never relabeled as physical truth. Cosine NCC has no mean subtraction; NRMSE is RMS error divided by the full reference range. There is no spatial/phase/gain alignment or mask. The original FMA-on/off400-step spread was NRMSE4.668425928513327e−6; pre-extraction full-pipeline boundary is1.5times that measurement,7.00263889276999e−6, with NCC≥0.9999999997503717. Intermediate source trajectory tolerance and distinct stage tolerances are frozen separately. The zero-based-initializer variant is a methodological sensitivity comparison and does not inflate numerical tolerances.

The active notebook must compute the full primary result using these cleaned modules. Scientifically relevant comparisons include direct MLAP selection with small/large focus windows, spatial denoising alone, multiple variational contrast weights and the explicit initializer diagnostic, all labeled by their actual computed status. Original/corrected/native precision reports remain separately identified. Fitted contrast and spatial regularity are conditional consistency diagnostics; without calibrated PSFs, they do not prove correct optical distance.


## Current benchmark v2 acceptance

The v2 Judge uses the explicitly observation-defined degree-eight float64 least-squares contrast polynomial plus TV, accepting any equal or lower attainable mean energy within1e-5. No initialized400-step field identity remains. Existing stored output passes; float32 and an independently computed sparse-gradient lower-energy field pass, while constant/reversed fields fail. No actual depth truth exists. See nexus_pytest.log and nexus_descent.log in the private benchmark_readiness_20261029 archive.

Source reconstruction and historical comparison computations remain unchanged; the standalone Judge is independent and uses the current README/metrics contract.
