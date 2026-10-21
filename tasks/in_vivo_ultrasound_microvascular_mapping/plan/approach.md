# Approach: Real In-Vivo Microvascular Density and Flow

## 1. Bound inverse problem and evidence

The benchmark reconstructs two maps from the first 30 complete IQ blocks of the PALA real mouse-tumor acquisition: trajectory-crossing density and mean scalar path speed. Each block contains 83 axial by 128 lateral pixels and 200 complex frames. The full bound experiment therefore contains 6000 frames and 63,744,000 complex samples. No frame subsampling, spatial crop, artificial noise, or simulated microbubble sequence is used. At 500 Hz inside each block, the archived block-repetition setting is 2 Hz, whereas the manuscript reports 150 blocks over 150 seconds. Absolute block timestamps are therefore not inferred from either account. Blocks remain distinct and trajectories are never linked across their boundaries.

The selected method is the PALA in-vivo pipeline with radial-symmetry localization. The notebook also actively executes weighted-average and integer-peak localization with the same observations, filtering, assignment, and rendering. These comparisons isolate localization differences while retaining their effects on downstream trajectories. They are not substituted cached images. Original native-single and independently verified double-computation oracle results are separately identified in validation.

Actual anatomical or velocity truth is unavailable. The hidden reference is a validated standard-method result on these exact observations, not a true vascular measurement. Agreement tests establish faithful computation, while independent analytic checks establish applicable symmetry, association, temporal, and rendering properties. Neither establishes complete anatomical sampling or calibrated absolute flow.

## 2. Measurement model, coordinates, and units

Let r=(z,x) be a native acquisition-grid coordinate. A useful complex beamformed observation model is

```math
Y(r,t)=T(r,t)+\sum_{j=1}^{J(t)}a_j(t)h(r-r_j(t))+\epsilon(r,t).
```

Y is measured IQ amplitude in arbitrary acquisition units. T is tissue clutter; J is the instantaneous number of contributing bubbles; a is the complex scattering amplitude; h is the local spatial response; r_j is bubble position; epsilon includes electronic noise and model mismatch. Beamforming and five-angle coherent compounding have already occurred in the supplied observations. The task does not reconstruct raw RF channel propagation or estimate a target-derived calibration PSF.

The local response-magnitude symmetry is used for position inference. No exact measured PSF, independent noise variance, or likelihood for the final density image is supplied. Consequently there is no invented global image-fitting objective or a claim that the final density can directly predict complex IQ frames.

Native z increases down and x right. The local position table uses zero-based z,x in acquisition pixels and a one-based frame identifier. Within a block, dt=1/500 s. Original metadata gives 0.11 mm transducer pitch, while examples use wavelength 1540/(15e6) m. These are distinct physical quantities; neither independently verifies the actual beamformed axial/lateral grid for these files. All reported speeds remain acquisition pixel/s. Multiplying by transducer pitch or wavelength would introduce an unsupported physical calibration.

## 3. Precision, initialization, and boundary indexing

Public IQ samples remain complex64 exactly. Computation explicitly promotes them to complex128 before the Gram decomposition. The original unmodified source inherits input precision; an oracle driver passes double-valued copies of the identical samples. This is an explicit precision variant, not a claim of bitwise reproduction of the original native-single run.

The change is necessary because the strong tissue components make lower temporal Gram modes sensitive to single-precision BLAS/LAPACK. On the first block, native-single Windows/Octave stage disagreement was about 0.00274 relative L2, exceeding the predeclared 1e-4 gate. Even feeding the captured Gram into a different single LAPACK backend did not fix it. Double precision has a nondegenerate ninth/tenth Gram singular-value gap of about 14%, orthogonal retained/removed subspaces, a Hermitian idempotent projector, and cross-driver agreement near machine precision. Exact diagnostic measurements and all failed checks remain in validation; tolerances were not widened.

The original in-vivo script contains two MATLAB statements:

```text
IQ(:,[1 end])=0;
IQ([1 end],:)=0;
```

In three dimensions, the first statement treats the remaining dimensions as a collapsed column dimension. It zeros the first and last collapsed columns, rather than both lateral edges in every frame. The second zeros the first and last axial row across the collapsed remainder. `original_boundary_zero` implements that actual behavior through a Fortran-order (83,128*200) view. It copies the input and never changes the released samples. Tests explicitly distinguish the preserved operation from an ordinary all-frame rectangular border mask.

There is no stochastic initialization, learned checkpoint, warm start, or cross-block state. Each block begins with its own filtered sequence and empty association graph. All smoothing/filter states are restarted at block boundaries.

## 4. Temporal tissue rejection and causal filtering

For a block, stack the spatial pixels in MATLAB column-major order into X in C^(10624 by 200). Form the Hermitian Gram matrix G=X* X, where * denotes conjugate transpose. Let G=U S V* be its singular-value decomposition, sorted in descending singular value order. Keep temporal columns 10 through 200 inclusive:

```math
Q=U[:,9:200],\qquad X_f=(XQ)Q^*.
```

Indices in the formula's slice are zero-based. The first nine temporal singular vectors are rejected as dominant clutter. No energy normalization, amplitude rescaling, or division by number of frames is introduced. The result returns to (83,128,200) in the same column-major convention. The implementation uses a general dense SVD, not a domain reconstruction routine. A complement projector X-X U9 U9* is an independent mathematical check, rather than a silently substituted runtime operation.

Apply a causal second-order Butterworth bandpass with passband endpoints 25 and 249 Hz at sampling rate 500 Hz. Bandpass transformation produces five numerator and denominator coefficients. For each complex pixel series y[n], the zero-state recurrence is

```math
v[n]=\sum_{k=0}^{4}b_k y[n-k]-\sum_{k=1}^{4}a_k v[n-k],\qquad a_0=1.
```

Negative-index samples are zero. General-purpose SciPy coefficient design and filtering are permitted numerical primitives. This is a single forward causal filter, not zero-phase `filtfilt`. Coefficients are independently captured from the oracle; no per-block filter tuning occurs. Nonfinite filtered values are set to zero, matching upstream safeguards, though finite bound inputs and validated arithmetic should not require this fallback.

## 5. Detection and candidate rejection

Take A=abs(v). Exclude a three-pixel image border by forming a zero-filled reduced magnitude image with only zero-based rows 3:80 and columns 3:125 populated. The extra margin beyond the five-by-five ROI avoids erroneous boundary localization.

For regional-maximum detection, PALA concatenates axial/time dimensions into an (83*200,128) image: first permute to (z,time,x), then reshape in column-major order. Use eight-connected regional maxima on that image and restore (z,x,time). The border margin prevents temporal concatenation edges from creating usable spurious maxima. A connected flat plateau is accepted only if its entire boundary has no brighter neighbor; a local maximum filter alone is not equivalent on arbitrary plateaus.

Sort masked magnitudes independently for each frame. Subtract the 201st largest value and retain only strictly positive candidates. Thus 200 is an upper bound, with ties at the cutoff removed. Enumerate detections in MATLAB column-major order, preserving within-frame axial-first candidate order for subsequent tie-sensitive assignment.

Extract a five-by-five ROI at each retained candidate. Although the upstream fwhm parameter is 3, MATLAB round(3/2)=2 creates offsets {-2,-1,0,1,2}. Reject ROIs containing more than four eight-connected maxima. After localization, reject offsets with absolute z or x shift greater than 1.5 pixels, or undefined positions. The original sigma-size estimate has its removal statements commented out and does not change accepted positions; the cleaned implementation omits that unused calculation rather than adding a new rejection criterion.

## 6. Radial-symmetry position estimator

For an ROI I on a centered five-by-five grid, define diagonal finite differences on its four-by-four midpoint grid:

```math
d_u(i,j)=I(i,j+1)-I(i+1,j),\qquad
 d_v(i,j)=I(i,j)-I(i+1,j+1).
```

Convolve each with a three-by-three averaging kernel of value 1/9, using zero extension outside that difference grid, to obtain f_u,f_v. Let g=f_u^2+f_v^2 be squared gradient magnitude. Coordinates z and x at these midpoints are {-1.5,-0.5,0.5,1.5}. The gradient line has slope and intercept

```math
m=-\frac{f_v+f_u}{f_u-f_v},\qquad b=z-mx.
```

Compute a gradient-energy centroid (z0,x0) by weighting midpoint coordinates with g. Then assign weights w=g/sqrt((z-z0)^2+(x-x0)^2). The position estimate minimizes weighted squared perpendicular distance to these lines:

```math
\min_{(z_c,x_c)}\sum_{i,j}\frac{w(i,j)}{1+m(i,j)^2}
 \left(z_c-m(i,j)x_c-b(i,j)\right)^2.
```

No additional factor, noise normalization, regularizer, or iterative optimizer is used. Let q=w/(1+m^2), and define S0=sum(q), S1=sum(qm), S2=sum(qm^2), B0=sum(qb), B1=sum(qmb), and D=S1^2-S2*S0. The explicit solution is

```math
x_c=(B_1S_0-S_1B_0)/D,\qquad
z_c=(B_1S_1-S_2B_0)/D.
```

This is the normal-equation solution of the stated least-squares problem; artificial iteration would be unnecessary. Add the ROI offset to the candidate integer position. The localization table retains double precision for the validated double-computation variant. Shift from one-based upstream coordinates to zero-based native positions before tracking.

The supplement method 1-7 prints a point-to-line distance denominator of 1+m^2 without its square root, while the subsequent squared-distance differentiation and source use 1+m^2 in the squared objective. The implemented perpendicular-distance objective and independent weighted-lstsq check resolve that typographical inconsistency. Its prose describes a weighted-average initial center, while the executable source specifically uses a gradient-energy centroid; that executable convention is preserved.

The upstream fallback for NaN slopes uses the unsmoothed ratio with the opposite sign from the primary formula. Infinite slopes are capped using ten times the largest finite slope. This is a documented source discrepancy, not a newly derived physical rule. Bound-data fallback incidence and independent analytic checks are recorded in validation. Degenerate profiles for which the normal equations are singular do not supply meaningful subpixel estimates. The benchmark does not certify arbitrary degenerate responses as physically valid.

For the weighted-centroid comparison, use sum(I*z)/sum(I), sum(I*x)/sum(I) on the same centered grid. The integer comparison uses zero offset. Detection, ROI rejection, and all subsequent stages remain the same.

## 7. Global temporal association

At each pair of adjacent frames within a block, let p_i and q_j be source and target positions. Define the squared-distance assignment cost

```math
C_{ij}=\lVert p_i-q_j\rVert_2^2\quad\mathrm{if}\quad
\lVert p_i-q_j\rVert_2\leq1,\qquad C_{ij}=+\infty\ \mathrm{otherwise}.
```

Obtain a partial one-to-one set of links through explicit Munkres/Hungarian updates on the source's padded complete assignment problem. Remove rows and columns with no finite edge. Let s be the sum and cmax the maximum of all finite original costs. Replace forbidden entries by B=10^(ceil(log10(s))+1) when s>0, or B=0 when s=0. Pad the reduced rectangle to a square using dummy cost 10*cmax. Minimize the total complete padded cost, then remove dummy and originally forbidden assignments when mapping back to the original indices. An entirely invalid graph returns no links. This objective is not the unconstrained sum over an arbitrary partial matching (whose empty matching would have zero cost), and no universal maximum-cardinality guarantee is claimed: the source's all-zero valid-cost case also assigns zero cost to forbidden edges before filtering. Preserve this degeneracy and tie convention rather than silently substituting a different matching objective. Distances exactly one pixel are allowed. A point can have at most one predecessor and one successor. There is no gap closing and no inter-block association.

The implementation retains the source's column-major zero selection and augmenting-path order, important for ties in the integer-peak comparison. Invalid rows/columns are excluded, rectangular costs are padded, row/column minima identify zeros, starred and primed zero paths extend a matching, and uncovered minima adjust the potentials until a complete padded matching exists. Forbidden assignments are removed when mapping back to the original rectangular graph. The exact finite-cost padding is described in design and tested against independent upstream cases.

Build maximal directed paths from nodes without predecessors, in original node order. Retain only trajectories whose number n of observed points is strictly greater than 8. An eight-point path is rejected; a nine-point path is accepted. There is no optimization convergence tolerance or early stopping beyond termination of the finite assignment procedure and exhaustion of all 199 adjacent frame pairs.

## 8. Trajectory regularity, interpolation, and speed

For each accepted track, independently smooth axial and lateral coordinates with the MATLAB default moving-average behavior requested at span 20. An even span is reduced to the nearest lower odd span: a sufficiently long track therefore uses 19. If n is shorter, reduce the span to n and again make it odd. At an endpoint i, shrink the centered window symmetrically to the largest available odd length. Endpoint observations are not extrapolated or zero-padded. Constant signals and linear trajectories are preserved, including their endpoints.

Interpolate the smoothed coordinates linearly at original-frame abscissae

```math
s=1,1+\delta,1+2\delta,\ldots\leq n,\qquad
\delta=0.8/(d_{\max}R)=0.08,
```

where dmax=1 native pixel and R=10. The final query need not equal n. For each track compute

```math
v=\frac{\sum_l\lVert r(s_{l+1})-r(s_l)\rVert_2}{n\,dt}.
```

This is upstream's scalar path-speed convention. The denominator is n rather than n-1, and the last interpolation point may precede the last observation. These cause a finite-track downward bias compared with an endpoint-complete estimate. Preserve and disclose the convention; do not call it an unbiased absolute hemodynamic measurement. The independent straight-line case with n=10, displacement 0.1 pixel/frame, and final query 9.96 gives 44.8 pixel/s, not 50 or 45.

Store each interpolated row [z,x,v] as float32 after smoothing/interpolation/speed calculation. Raw tracks used for noninterpolated comparisons are also cast to float32 at the same point. These quantizations reproduce the source's deliberate rendering inputs. Do not quantize localization coordinates before association or smoothing in the double-computation variant.

## 9. Rendering and accumulation

Map float32 native coordinates to the source rendering grid by u=10*(r+1). Round to the nearest integer with halfway values away from zero, convert to a zero-based index by subtracting one, and reject points outside (831,1281). The native coordinate at output index (i,j) is ((i+1)/10-1,(j+1)/10-1). This mapping preserves the original padding/origin convention; r*10+1 is a different transformation.

For each track, form the set of distinct visited superpixels. Increment density by one per set element. Add the track's constant v once at each visited pixel. After all tracks and blocks, divide accumulated speed sums by accumulated density where density is positive; empty pixels have speed zero. This is equivalent to the source's per-block density-weighted mean aggregation. Both final arrays are float64. The noninterpolated comparison uses the same distinct-track counting applied to the original observed positions.

Density is not normalized by observation duration or total detected bubbles. Speed is not weighted by bubble intensity, per-frame samples, or number of interpolation points. No Gaussian display smoothing is applied to submitted maps. Notebook square-root density display is explicitly labeled and does not change evaluated arrays.

## 10. Validation, termination, and limitations

Run all 30 blocks for all three notebook methods. The standard CLI runs all 30 blocks with radial localization, writes the two declared submission arrays, and reports actual runtime and localization/track totals. A complete notebook independently recomputes all three methods, evaluates fresh arrays, plots upstream/fresh differences and scientific method comparisons, and invokes the standalone Judge on its fresh radial submission.

Stage fixtures are captured from the separate unmodified upstream-function run, including complex SVD output, causal filter output/coefficient arrays, detections, raw tracks, and rendered maps. Applicable checks include subspace identities, radial translation/reflection/amplitude symmetry, partial assignment and tie cases, strict length and no-gap rules, analytic smoothing endpoints, unique crossing counts, path-speed units, and known metric cases. Stage relative-L2 and submillipixel localization tolerances were fixed before the local implementation; map thresholds were also fixed before results were judged.

All four Judge conditions are required: density NCC>=0.995 and dynamic-range NRMSE<=0.005; speed NCC>=0.99 and NRMSE<=0.01. These are finite tolerances for a standard-method reference, not mechanically zero-error self-comparison thresholds. They do not certify anatomy. The source-single failure remains visible; no threshold is weakened to absorb it.

The method has no training, stochastic seed, gradient-based imaging optimizer, global reconstructed-IQ residual, or acoustic adjoint. Its central numerical inverses are temporal subspace rejection, weighted radial line intersection, and discrete multiobject assignment. Limitations include real-data model mismatch, finite-duration vascular coverage, tissue/bubble spectral overlap, unresolved overlapping bubbles, association errors, finite-track speed bias, and the absence of independently verified mm/pixel calibration.
