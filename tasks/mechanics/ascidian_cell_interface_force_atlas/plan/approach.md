# Implemented force-inference mathematics

## Binding, unknowns and conventions

The bound observation is the complete existing author-provided real microscopy-derived Phallusia mesh. Raw vertices have shape (70433,3), triangles (144312,3), and region labels (144312,2). Normalize each label pair in ascending order and reverse its triangle winding whenever swapping labels. Remove only vertices referenced by no triangle: 67303 vertices remain, with every observed triangle unchanged. Coordinates are the author's uncalibrated voxel/mesh units; there is no micrometre or pascal calibration.

There are 357 unordered interfaces and 65 region labels. Labels are nonconsecutive integers from 0 to 90. Interface order is lexicographic order of sorted unique label pairs; pressure order is ascending unique region label, exterior 0 first. The unknown tension vector gamma has 357 entries; the complete pressure vector has 65 entries with exterior pressure fixed to zero. Arithmetic mean tension is one. No force truth is available; the evaluator compares the selected conditional author-method estimate and independent mechanical consistency.

The selected default is scalar Young–Dupré tension inference, followed by variational Laplace pressure inference. The current source's dense exact-gauge path is used. The method is closed-form linear algebra; no artificial iteration, optimizer stopping tolerance, random initialization, neural checkpoint or synthetic target is introduced.

## Explicit area and volume operators

For a triangle with ordered vertices a,b,c, define its right-hand unit normal n and area A by

```math
\boldsymbol n=\frac{(\boldsymbol b-\boldsymbol a)\times(\boldsymbol c-\boldsymbol a)}{\|(\boldsymbol b-\boldsymbol a)\times(\boldsymbol c-\boldsymbol a)\|},
\qquad A=\tfrac12\|(\boldsymbol b-\boldsymbol a)\times(\boldsymbol c-\boldsymbol a)\|.
```

The analytic area derivatives are one half of n cross (c−b), n cross (a−c), and n cross (b−a), respectively. Each face contributes to the one interface identified by its label pair. Scatter-add these values into sparse D, of shape (357,3N), with N=67303 and vertex-major x,y,z columns. Its units are mesh length. Cross-product areas and the author's Heron areas agree to the independently frozen geometric tolerance; the differential operator itself retains the source cross-product formula.

The signed volume contribution is det(a,b,c)/6. Its three vertex derivatives are cross(b,c)/6, cross(c,a)/6 and cross(a,b)/6. Add these to the first listed region and subtract them from the second. Sparse Vfull has shape (65,3N) and units length squared. The first region's normal points outward toward the second, so the exterior volume is negative and every bounded cell's volume is positive. Let V be Vfull without the exterior row.

For interface tensions gamma and 64 cell pressures p, the vertex force residual and pressure normal equations are

```math
\boldsymbol f=D^{\mathsf T}\boldsymbol\gamma-V^{\mathsf T}\boldsymbol p,
\qquad G\boldsymbol p=H\boldsymbol\gamma,
\qquad G=VV^{\mathsf T},\quad H=VD^{\mathsf T}.
```

G has full rank64 on this binding, with measured condition number91.78183984946723. The selected dense inverse evaluates p=inv(G) H gamma, then prepends exterior pressure0. The author's implementation first row-normalizes a larger Gram system; those cell-row factors cancel in this pressure product. Local unnormalized blocks give the same formula and are independently compared with source blocks and source pressures. This algebraic cancellation is documented rather than presented as a different regularizer.

## Junction observation and angle convention

Group the three directed sides of every triangle by their unordered endpoint indices. A valid triple edge has exactly three incident triangles, three distinct material pairs and exactly three regions. Edges of other valences provide no triple-junction angle observation. Specifically, 128 malformed three-region edges are retained and identified for diagnostics, but not given guessed contact angles. Their triangles still contribute to all area and volume operators. No target-derived quality mask exists.

Compute each face's normal with the source cross-product and normalization order. At each incident half-edge visit, take that face first and the other two incident faces in original half-edge order. Reverse a neighboring face's ordered material pair and its normal if it shares the same first label or same second label with the first face. Consecutive normals then define the interior region between the two directed material transitions. Read its chord wedge as arccos of the clipped normal dot product. Keys are (min(e,g),f,max(e,g)), with f the interior region.

The default reader is unoriented arccos in [0,pi]. A reflex angle can therefore be reflected; neither the implementation nor the reported uncertainty assumes all measured wedge sums equal2pi. The notebook plots that departure. This is a documented source/paper measurement convention on imperfect geometry, not an exact convex-foam assertion.

Preserve historical source accumulation: visit each geometric triple edge through each of its three incident half-edges, appending each measured wedge twice. The mean angle is the arithmetic mean of those visits, not an arclength-weighted mean. Each returned line length is six times its true arclength. Both numerator and denominator in line-length weights carry the same factor, which cancels. The binding has6874 valid triple edges,569 triple lines and1707 wedge keys.

## Scalar Young–Dupré system and gauge

For a line joining ordered regions a<b<c, theta_a,theta_b,theta_c denote the three mean wedges inside those regions. In local column order (gamma_ab,gamma_bc,gamma_ac), use

```math
C=\begin{pmatrix}
1&\cos\theta_b&\cos\theta_a\\
\cos\theta_b&1&\cos\theta_c\\
\cos\theta_a&\cos\theta_c&1
\end{pmatrix}.
```

Multiply all three rows by line length divided by total valid line length. Triple lines follow the author's scalar index order, equivalent to increasing (c,b,a); interface columns remain lexicographic. The1707 physics rows form A. The last stored row is a row of ones with right-hand side357, giving stored M shape(1708,357). The scientific target is

```math
\min_{\boldsymbol\gamma}\|A\boldsymbol\gamma\|_2^2,
\qquad \boldsymbol1^{\mathsf T}\boldsymbol\gamma=357.
```

Let n=357 and r be the nonzero row norms of A. The numeric gauge-row weight is w=2 raised to round(log2(median(r)/sqrt(n))); on this binding w=0.0001220703125. Solve dense least squares with rows [A;w1-transpose] and right-hand side [0;wn] using scipy.linalg.lstsq, then rescale the homogeneous solution by n/sum(gamma). The latter makes the mean exact without changing tension ratios. For homogeneous rows, the soft gauge solve is proportional to the exactly constrained optimum. No positivity constraint is silently added; nonpositive diagnostic results would be retained. The actual bound default gives positive tensions.

The paper's historical formulation used an appended normalization equation; the selected current author revision corrects its scale to an exact mean. This current-source normalization and the arccos reader are explicitly identified in provenance. There is no claim that every source version is byte-identical to the2023 release.

## Actual scientific comparisons

All comparisons use the same complete geometry and explicit local mathematics. Projection Young–Dupré uses two rows per line: (1,cos(theta_b),cos(theta_a)) and (0,sin(theta_b),−sin(theta_a)). Lami relates gamma_ab/sin(theta_c), gamma_bc/sin(theta_a), gamma_ac/sin(theta_b) by three pairwise differences. Inverse Lami cross-multiplies the same relations. Log Lami relates pairwise log-tension differences to corresponding log-sine differences. Its difference rows sum to zero; after its gauge-row least squares, shift log-tensions to mean log-tension0 and exponentiate. Its native gauge fixes geometric mean1, not arithmetic mean1. The plotting comparison rescales the entire Log Lami tension/pressure atlas to arithmetic mean1 and preserves native outputs in numeric sources.

Lami and Log Lami exclude a triple line if any wedge is within sqrt(8 epsilon64) of0 or pi, where their coefficients are singular. This follows the source arithmetic-resolution rule; no bound-data line is discarded by a newly tuned threshold. Inverse Lami's sine coefficients stay bounded and do not use this exclusion. The no-length-weight comparison sets every line-row multiplier to1 while retaining all other equations.

The historical variational Gram comparison explicitly constructs the interface/cell Gram block matrix [DD-transpose,−DV-transpose;VD-transpose,−VV-transpose]. Normalize each nonzero-sum row by its signed row sum, as the source does. Partition its blocks as Fg,−Fp;Gg,−Gp and solve the homogeneous tension system Fg−Fp inv(Gp) Gg with the exact mean-tension gauge. Its weighted noisy-data objective is distinct from the following later source route.

The additional current-source Operator comparison forms B=[D-transpose,−V-transpose], with shape(201909,421). Define anchor a=(1/n on tension entries,0 on pressure entries), unit anchor u=a/||a|| and Householder vector h=u−e1. The columns1 onward of I−2hh-transpose/||h||² form orthonormal Z spanning the nullspace of a-transpose; use the corresponding identity basis if ||h||<1e−14. The particular solution is beta0=a/(a-transpose a). Solve min_y ||BZy+B beta0||² by dense least squares and return tension entries of beta0+Zy. Both Gram and Operator are actually run and independently source-validated. They are not conflated on noisy observations. The Operator route is comparison-only; the standard inverse uses Young–Dupré.

For curvature pressure comparisons, side lengths a,b,c opposite a triangle's vertices give Heron area and cotangents (b²+c²−a²)/(4 area), cyclically. Scatter opposite-edge weights into symmetric sparse L. If Iv is incident triangle area at vertex v, the mean-curvature vector is .75/Iv times (Lx−x L1). Use the sign of its scalar product with the area-weighted mean oriented face normal, times its magnitude. Vertices incident to any edge of valence>2 are excluded from reading interface curvature. Average remaining vertex curvatures equally (Laplace) or with weights Iv (WeightedLaplace). An interface with no such vertex has NaN curvature and supplies no equation.

The pressure relation for interface(a,b) is p_a−p_b=−2 H_ab gamma_ab. Both variants weight each equation by interface area divided by total interface area; their difference is curvature averaging. Pin exterior pressure exactly by eliminating its column. Cells not connected to the exterior through finite-curvature interfaces remain NaN, rather than receiving a fabricated least-norm pressure. All64 cells are connected on this binding, and both full comparison outputs are finite. Negative comparison pressures are shown; no biological correctness claim is inferred.

Arithmetic order matters in the curvature sign: one bound vertex is almost tangential to its normal. Replacing reciprocal multiplication by equivalent division, or changing the three-term dot reduction, changed that sign and caused a comparison parity failure. The final code preserves the author's reciprocal/multiply and sum order; original Laplacian vectors, vertex normals and inverse areas are bit-identical in the captured diagnostic. No comparison tolerance was enlarged.

## Experiments, reporting and evaluation

The notebook actively computes the full selected inverse, all stated formulations, independent Gaussian coordinate-jitter cases sigma=.1,.25,.5 with seeds210,211,212, full coordinate scales.5and2, a mean-tension2 solve, and three LAPACK drivers in float64/float32. Jitter modifies only observed coordinates, is not a calibrated noise model, and never replaces benchmark raw data. Scaling preserves tensions and changes pressures inversely with length; changing the gauge rescales tensions and pressures together. Diagnostic thresholds are not retuned from fresh results.

The standalone evaluator scores every357 tension and65 pressure values. NCC is uncentered cosine similarity. NRMSE is RMS error divided by the reference dynamic range, separately for each quantity. There is no crop, mask, gain fit, permutation or alignment. Independent original reruns were bitwise equal; alternative LAPACK drivers/precisions and serialization yielded nonzero pre-extraction limits: tension NRMSE<=2.384619215064322e−6 and NCC>=.9999999999794431; pressure NRMSE<=2.2095445828954433e−6 and NCC>=.9999999999979622. These are numerical reproduction limits, not force-accuracy bounds.

Additional checks require |mean(gamma)−1|<=1e−6, |p_exterior|<=1e−10 and relative pressure normal-equation residual<=1e−5. The latter is ||V(D-transpose gamma−V-transpose p)||/||V D-transpose gamma||, computed independently from public triangles. A zero denominator uses the absolute numerator norm; reference metrics still reject degenerate fields. No zero vertex-force requirement is imposed on noisy measured geometry. Zero, uniform, reversed and separately corrupted quantities fail. The inverse never reads any reference, fixture or upstream package.
