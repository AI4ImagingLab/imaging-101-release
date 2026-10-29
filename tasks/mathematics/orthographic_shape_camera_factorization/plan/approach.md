# Implemented factorization and metric inference

## Bound problem and units

All51frames and491points from the author-provided point_trajectory.mat are retained. No acquisition-origin claim is made. Unknowns are shape X[491,3] in pixel-equivalent units, cameras R[51,2,3] dimensionless, and translations t[51,2] pixels. Projection w=RX+t is explicit in physics_model.forward. The shared rigid shape has zero centroid. Global orientation/reflection is unobservable; camera normalization fixes scale and observed centroids fix translation. No physical millimetre calibration is available. There is no independent3DGT, so the reproduced original helper is a baseline only.

## Centre, normalize and factor

Interleave x/y measurement rows into W[102,491]. t=W*ones(491,1)/491; subtract each mean from its entire row. Define scale a=max(abs(centred odd rows))+max(abs(centred even rows)), with both maxima across all frames and points. Divide by a. All computations are float64, C-order arrays locally. Reject zero scale.

Compute thin SVD Wn=U diag(s)V^T, sorted descending, using general LAPACK. Retain exactly the first3singular values. Mhat=U[:,:3]diag(sqrt(s[:3])); Shat=diag(sqrt(s[:3]))V[:,:3]^T. This minimizes the unweighted squared image residual over rank3factors. No artificial iteration, learned weights, regularizer, initialization seed or line search occurs.

## Metric correction

For each frame, let x and y be its two rows of Mhat. For a symmetric3x3 C, define the six-term product vector

```math
g(x,y)=[x_1y_1,\ x_1y_2+x_2y_1,\ x_1y_3+x_3y_1,\ x_2y_2,\ x_2y_3+x_3y_2,\ x_3y_3].
```

The six unknown entries are ordered C11,C12,C13,C22,C23,C33. Stack g(x,x) for all frames, then g(y,y), then g(x,y), giving A[153,6]. Right-hand side b=[ones102,zeros51]. Solve min||Ac-b||² by np.linalg.lstsq(rcond=None), then assemble symmetric C. On the full bound case eigenvalues are [.26310892,1.49262276,1.55018907], strictly positive. The original uses SVD C=Uc diag(sc) Vc^T and G=Uc diag(sqrt(sc)), rather than an upper Cholesky factor. This algorithm is preserved. Indefinite noisy C would be transformed using singular magnitudes by the source; it is not needed on this case and no alternate nearest-SPD algorithm is silently inserted.

Native additional correction: N=Mhat G. For each2x3frame block Nf, stack pinv(Nf.T) into D[102,3]. Solve H=lstsq(N,D). Take the principal matrix square root J=sqrtm(H), then Q=GJ. This uses all frames jointly; it is not separate frame reorthogonalization. Reject a material imaginary root (>1e-12) rather than silently converting a genuinely complex solution. Roundoff imaginary parts below this bound may be discarded; the bound case is real. Return M=Mhat Q and S=inv(Q)Shat, preserving source explicit inverse conventions. Restore physical image scale to shape: X=(aS)^T. Translations are the original row means, cameras reshape M in interleaved order. The output still has finite metric residuals because tracks are not exactly orthographic/rigid.

## Gauge, metrics and validation

For predicted row-shape X and reference Y, SVD X^TY=Uq diag(sq)Vq^T gives one Qg=UqVq^T. Reflection is permitted. Apply X->XQg and R->RQg jointly. Translations stay in image coordinates. No centring, scale fitting, per-frame rotation or point permutation is performed in evaluation. This transformation leaves RX+t invariant. Evaluate NCC and dynamic-range NRMSE separately on full shape, cameras and translations; all six bounds must pass, together with reprojection RMS<=2.54pixel. This is1.1times original2.30846pixel rounded, not a changed noise estimate. Shape/camera/translation NRMSE<=1e-6 and NCC>=.999999 allow float32 while rejecting1%gain, zeros, sign changes in one factor and inconsistent gauges. The original/local full discrepancies are below2e-12pixel and4e-15camera entries after joint alignment.

Known mathematical checks include exact symmetric products, the wrong-upper-Cholesky algebraic diagnostic, joint reflection invariance, projection adjoint/directional gradient, rank3tail-error identity and PSD of C. Camera row norms span .9878..1.0096 and maximum row dot is .01519; these are reported imperfections, not hidden normalization. Notebook comparisons actively compute the standard method, affine-only ablation, full-data measured residuals and two controlled image-noise perturbations. Historical paper results are not represented as fresh measurements.
