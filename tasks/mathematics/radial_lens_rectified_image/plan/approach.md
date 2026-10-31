# Implemented algebraic inversion

The bound observation is the complete author photograph and both29 observed edge coordinates in the official source archive. The selected method is the original AMI two-parameter algebraic function distributed in the recorded GitHub revision. It is not the later automatic edge/Hough workflow, and not the official older executable's additional gradient refinement. Paper and source differ in normalization and zoom conventions; this task explicitly preserves the selected source.

## Coordinates and normalization

For measured point x and fixed c=(344,253), compute q=(x−c)/A, where A=sqrt(sum(||x−c||²)/N), N=29. The paper prints a denominator2N; the selected source uses N. Source polynomial magnification is1+b2*r²+b4*r⁴ in normalized coordinates. Physical pixel coefficients are k2=b2/A²,k4=b4/A⁴. No center search or tangential model is performed.

## Quartic objective and algebraic solve

For each group, construct coordinate coefficient vectors from q,q*r²,q*r⁴ and subtract their within-group means. Convolve the affine coordinate polynomials to accumulate unnormalized scatter polynomials X,Y,Z for x²,y²,xy. Sum P=X*Y−Z² across groups. The source does not divide group scatters by group size, unlike the paper's normalized covariance description: unequal15/14 sample counts consequently retain original group weighting. Validate this distinction with frozen source polynomial coefficients and an independent direct-scatter determinant calculation.

Normalize coefficient variables by sx=P[4,0]^(1/4),sy=P[0,4]^(1/4). Compute both derivative polynomials explicitly. Construct a6×6 Sylvester matrix whose rows contain shifted cubic coefficients in the first variable and polynomial coefficients in the second. Expand its determinant using permutation signs and ordinary polynomial convolution. Find resolvent roots using NumPy's general polynomial eigensolver. Preserve the source real-resolvent cutoff1e−15; compute real cubic candidate roots (imaginary cutoff1e−12) for both derivative slices. Evaluate every candidate on the normalized quartic and choose the least finite energy. Undo variable and coordinate normalization. There is no generic nonlinear optimizer or artificial inverse iteration. Polynomial-root numerical primitives differ from the author's generic root routine, so stage/geometry tolerances are independently checked.

## Radial inverse and image resampling

Find the maximum source corner radius R and transformed radius R*L(R). Build an inverse-radius table up to int(max(R,R*L(R))+1.5). At each integer target radius d, solve r*L(r)=d by explicit Newton steps, using derivative sum((i+1)*ki*r^i). Start r=1 at the first radius, reuse the preceding solution for successive radii, stop at relative change1e−12 or10000 steps. Set the zero entry equal to the first ratio, matching the source convention. Bilinearly interpolate this inverse-radius ratio at every output pixel's transformed radius. Source frame scaling uses width,height corners rather than width−1,height−1. Apply the original all-visible scale and translation order, truncating source cell indices toward zero. Accumulate four bilinear weights in source order, retaining only available positive extra-neighbor weights; normalize their sum and truncate intensity to uint8.

Independent uniform-scene testing found native divide-then-truncate roundoff converts36,423 valid channel samples from213 to212. This violates exact reproduction of a constant intensity field. The local implementation adds one explicit correction: when all four available corner intensities for a channel are identical, return that observed integer exactly. This is a documented roundoff correction, not a claim of unchanged bitwise reproduction; it reads no reference result. Original fixtures remain uncorrected and the pre-local image tolerance0.2ADU is unchanged. Coefficient inference is unaffected. The correction passes both independent constant-field physics and the original complete-image parity threshold.

## Evaluation and experiments

All pixel values and both nonzero geometric coefficients are scored. Independent geometric GT is unavailable. Judge image NCC≥.9999, dynamic-range NRMSE≤1/255, relative nonzero coefficient error≤1e−4, central gauge error≤1e−12 and odd coefficients≤1e−16 were fixed before extraction. Source-stage tests are tighter: coefficient relative1e−5, transformed point absolute.003pixel, full image RMS.2ADU. Reference numerical agreement does not establish unobserved scene accuracy.

Notebook comparisons actively recompute full reconstruction, unchanged-geometry diagnostic, individual-line fits, objective landscape and explicitly synthetic point-localization perturbations. Unidentifiable or invalid optional candidates are reported as failures rather than substituted images. All fresh reconstruction calls use local modules; original arrays only support labeled comparisons.
