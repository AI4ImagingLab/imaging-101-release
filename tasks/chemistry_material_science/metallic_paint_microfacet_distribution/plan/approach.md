# Implemented approach

The task estimates a finite-grid effective isotropic radial slope density from measured RGB backscatter under the monostatic Smith model. It retains all RGB angular bins as public context but only the coincident direction slice determines the selected inverse. No angle-dependent Fresnel recovery, off-slice BRDF fitting, diffuse separation, or microscopy-truth claim is made.

## Coordinates and observations

The encoded array is (3,90,90,180), ordered RGB, half polar, difference polar, difference azimuth. Multipliers (1,1.15,1.66)/1500 convert encoded entries to sr^-1. Half-polar bins use floor(90 sqrt(2 theta/pi)); the original direction constructor snaps cos(theta)>0.99999 to the pole. At backscatter the difference angles vanish. Negative triplets are zeroed; none of the selected samples is negative. RGB luminance weights are (.2126,.7152,.0722).

For i=0,...,88, let t_i=i sqrt(pi/2)/89, theta_i=t_i², c_i=cos(theta_i), and s_i=tan(theta_i). The last tabulated PDF endpoint (i=89) is fixed to zero. The function P(s) is a two-dimensional density per slope area, not a one-dimensional radial density; radial probability is 2 pi s P(s) ds.

## Positive operator and inverse updates

For each row i and column j, evaluate

$$K_{ij}=8\Delta t\,f_i c_i^6\,t_j s_j c_j^{-2} A_{ij},\qquad
A_{ij}=\Delta\phi\sum_m\max(1,s_i s_j\cos\phi_m).$$

Here f is measured luminance, Delta t=sqrt(pi/2)/89, Delta phi=pi/180. The original floating accumulation starts phi=0 and continues strictly while phi<2pi; binary rounding gives 361 terms. Preserve this convention, including its slight duplicate-endpoint quadrature bias. It is not silently replaced by 360 evenly indexed samples.

Initialize p=ones(89) and apply exactly four unnormalized updates p=K p. Append a zero horizon endpoint and divide all values by the radial mass below. There is no optimization loss, regularizer, line search, gradient or convergence-driven stopping criterion in the published selected example; four powers are the actual fixed stopping rule. The positive dominant eigenvector has nonnegative entries. The eigenvalue absorbs normal-incidence Fresnel scale and is not an independently identified RGB Fresnel function.

## Interpolation, normalization and roughness

The original routine named spline_eval performs linear interpolation. At slope s, interpolate the 90 samples in x=sqrt(2 atan(s)/pi). For k=0,...,127 use t=k sqrt(pi/2)/128, theta=t², s=tan(theta), and w=2 Delta t t P(s)s/cos²(theta). The mass is M=2pi sum(w). Normalize P by M. Beckmann alpha=sqrt(2pi sum(s²w)); GGX alpha=4 sum(sw). Both are dimensionless. The latter uses a first absolute-component moment because the GGX second moment diverges. These moment estimates are not least-squares fits of analytic distributions.

Analytic comparison densities are exp(-s²/alpha²)/(pi alpha²) for Beckmann and alpha²/[pi(alpha²+s²)²] for GGX. They are evaluated freshly from the estimated parameters, not inserted as alternative reconstructions.

## Evaluation and independent checks

The finite quadrature ends at theta=(127/128)^2 pi/2 and slope about 40.9. The continuous interpolation has a 1/s tail near the zero horizon endpoint and is not integrable over infinite slopes. Consequently the roughness values are finite-quadrature summaries. This is an explicit scope clarification of the original frozen contract wording, not a changed output, tail correction, or relaxed threshold. No actual surface truth exists. The reference is the original measured-data C++ example with an allocation-only repair, independently verified bit-identical to its unchecked execution. All arrays retain their native scale and grid. Density NRMSE uses reference dynamic range, and NCC is uncentered cosine similarity. Physical metrics are |M-1|, roughness moment consistency, and ||Kp-lambda p||/||Kp|| with lambda=(p dot Kp)/(p dot p). All thresholds were frozen before local implementation. Large finite overflow produces a descriptive evaluator error, never invalid JSON.

The notebook repeats full reconstruction and compares 1,2,4,8 powers, a direct dominant eigenvector, native outputs, and both analytic PDFs. It also reports the continuous normalization separately, since the selected method's fixed 128-point convention is the contract. Uniform intensity scaling leaves the normalized distribution unchanged, providing an independent invariance check. No coefficients or tolerances are adjusted to improve the clean result.
