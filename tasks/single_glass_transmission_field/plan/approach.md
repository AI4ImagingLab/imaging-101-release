# Implemented convex transmission-image inverse

The selected method is the complete channel-independent inverse in Yang et al., CVPR2019, official reflectSuppress.m commit5a693a2d1ccf7d2b2f991dc4c81c41ece9077aaa. The bound observation is the author's sample-run train2.jpg, complete1080×1440RGB, selected before any inverse output. The source sample parameters are h=.033 and epsilon=1e-8. The task estimates the published conditional transmission-like image, not a uniquely known calibrated scene radiance or a full two-layer separation.

## Acquisition approximation and unavoidable ambiguity

For RGB channel c, a single image is approximately Yc=a Tc+(1−a)(Gs*Rc)+nc, where T is the desired scene, R the reflected environment, a unknown mixing fraction, Gs a blur kernel and n noise/model mismatch. The bound JPEG is not supplied with photometric linearization, a, s, camera response or independent layers. We therefore preserve normalized JPEG/255 detector units and label the image-domain model conditional. A constant reflected contribution cannot be uniquely separated; the method's fidelity term retains each observed channel mean. This convention fixes an inverse ambiguity, not an independently measured reflection offset.

## Source discretization and objective

Image axes are row i downward, column j rightward; all native pixels and channels are retained. The two gradient components are forward column and row differences, with the outgoing component zero at the last column/row. Divergence is their negative adjoint on this zero-outgoing-flux subspace: current minus previous column flux plus current minus previous row flux, with outside incoming flux zero. L=div∇ is a negative semidefinite Neumann Laplacian, equivalent to mirrored boundary extension. No periodic wrap or crop is used.

For each channel independently, retain BOTH gradient components at a pixel only when their Euclidean magnitude is strictly greater than h=.033. This strict inequality is the actual MATLAB wthresh hard-threshold API convention. Paper Eq5 prints greater-or-equal; exact tie cases follow executable source/API and the discrepancy is documented. Let q=δh∇Y and f=div q. The published objective is

```math
\Phi(X)=\frac12\lVert LX-f\rVert_F^2+
\frac{\varepsilon}{2}\lVert X-Y\rVert_F^2.
```

X is the desired output field, Y the observed field, F denotes the Frobenius norm over all pixels, ε=1e-8 and no division by image area is applied. There is no additional TV penalty, unknown-layer optimizer, positivity projection, learned checkpoint, mask or post-hoc clipping. Its gradient is L(LX−f)+ε(X−Y), since L is self-adjoint. The unique stationary solution satisfies (L²+εId)X=L f+εY. The positive ε removes the constant nullspace and makes the Hessian strictly positive definite. No iterative stopping rule or artificial iteration is needed.

## Explicit closed-form inverse

For grid height H and width W, use orthonormal type-II2D cosine transforms on the two spatial axes. The Laplacian eigenvalue at row-frequency r and column-frequency s is

```math
\kappa_{r,s}=2\big(\cos(\pi r/H)+\cos(\pi s/W)-2\big),
\quad 0\le r<H,\quad 0\le s<W.
```

Let b=L f+εY. Compute X=IDCT2(DCT2(b)/(κ²+ε)), independently per RGB channel. SciPy dctn/idctn provide only the generic transform; local code explicitly constructs q,f,L,b and the denominator. Their orthonormal scaling matches the source Octave dct2/idct2, as established by full-array parity and independent dense/cosine cases. At zero frequency κ=0 and the mean is inherited from Y. Values below0 or above1 are retained in saved float64 arrays. RGB display alone uses[0,1] mapping with exact out-of-range sample counts.

The source PoissonDCT_variant also accepts general μ and λ, but this published selected call uses μ=1 and λ=0. Those unused generalized terms are not part of current runtime. Every function acts on a complete2D scalar or channel-last image; only standard raw loading enforces the bound batch-first shape.

## Independent original verification and frozen tolerances

The official source is unchanged in the private Octave8.4 oracle. An initial native run fails because Octave lacks general wthresh. The private API shim implements only the documented strict hard-threshold primitive; no domain inverse or operator is replaced. Known threshold ties establish its semantics. Source stages captured by read-only instrumentation reproduce the entire original output bitwise.

Before local inverse implementation, independent dense incidence matrices, cosine eigenmodes, constant fields and source h=0 identity checks establish signs, boundary conventions and numerical scale. Dense-vs-source tiny-system difference8.8408616e-9 reflects ill-conditioned ε=1e-8 DC recovery; full h=0 identity error6.9050912e-10 and channel mean drift3.1500752e-10 are measured. Whole-source repeated outputs are bitwise. Frozen full primary boundaries are NCC≥.9999999, reference-range NRMSE≤1e-7, maximum absolute≤5e-7, stationarity relative residual≤1e-8 and channel mean difference≤1e-6. These nonzero cross-transform tolerances are fixed before local results and are much smaller than actual image/reflection variations. They are not0.9/1.1 applied to a degenerate self-reference error.

No independent transmission measurement exists for train2; evaluation/baseline_reference.npz is explicitly the actual original method result, not fabricated truth. The Judge evaluates all RGB values with no rescaling or alignment. Its physical residual uses an independent endpoint-flux Laplacian and the observed public image; agreement with a reference is not proof of true layer recovery.

## Computed scientific comparisons

The review actively reconstructs the complete benchmark, compares full original stages and outputs, explores predeclared h=[0,.015,.033,.06,.1] and ε=[1e-10,1e-8,1e-6,1e-4], and preserves edge-loss/reflection tradeoffs. The smallest epsilon is explicitly a more ill-conditioned contextual diagnostic, not a changed primary criterion. Supplementary original/fresh full scenes building, conference_room and Dog-In-Window are fixed before their original inverses; no pixel crop or target-dependent selection is used. The same h/ε illustrate transfer and failures rather than per-scene hidden-reference tuning.

Independent generated cases use known focused transmission plus finite normalized Gaussian-blurred reflection with reflected boundary convolution and an explicitly known mixture fraction. Strong and weak transmission examples expose model success/failure. Comparison target includes the known transmitted component plus the reflected mean only to illustrate the selected mean convention; actual unshifted true transmission is shown separately. This synthetic evaluation convention never rescales the bound output. Optional Gaussian image stress is labeled illustrative detector-unit noise, not instrument calibration. External paper methods without supplied runnable implementations are described as historical comparisons, never filled with zeros or passed off as fresh results.
