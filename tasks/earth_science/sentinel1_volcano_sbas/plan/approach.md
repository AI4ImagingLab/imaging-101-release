# Implemented Approach

## 1. Bound scientific problem and scope

This task reconstructs the complete **450 x 600 spatial scene and 98 acquisition dates** of the real Fernandina Sentinel-1 descending-track-128 tutorial dataset. Its 288 original interferograms are not replaced by synthetic observations, a single-pixel regression or a small favourable crop. The acquisition spans 2014-12-13 through 2018-06-19. The selected method comprises common-region phase-closure correction, coherence-weighted minimum-velocity-norm SBAS, the independently supplied ERA5 LOS correction, affine spatial deramping, and joint elevation-residual/polynomial/eruption-step estimation. A uniform-weight SBAS reconstruction is an actively computed comparison, not the shipped main solver.

This is a modern numerical port of selected stages of the published MintPy workflow. It is not a claim to reproduce every figure of the 2019 paper: that paper also discusses larger geographic scenes, different networks, bridging, simulated method comparisons, GNSS and alternative atmospheric products. Those unavailable acquisition conditions and independent measurements are not invented. The exact source revision, postprint and data hashes are recorded in `test/validation.md`.

The five output quantities are the corrected LOS movie, apparent network LOS movie, elevation-residual map, five deformation-parameter maps and temporal-coherence map. The corrected movie retains its temporal residual, which contains real unmodelled deformation and remaining atmospheric/noise contributions. It is not simply the fitted quadratic-plus-step function.

## 2. Data, coordinate and precision conventions

Let M=288 be the number of edges, N=98 the number of dates, H=450 and W=600. Pixel p=(y,x) uses the source azimuth/range raster order. No spatial interpolation is applied. Source phase/coherence/incidence/baseline/weather arrays retain float32; source latitude/longitude retain float64; connected-component labels retain uint8. Public storage adds one leading batch axis. `load_data` checks the declared shapes, exact dtypes and finiteness and removes that axis.

The reference pixel p0=(76,156) is fixed by the source reference latitude/longitude, not chosen for numerical convenience. The wavelength is exactly lambda=0.05546576 m. LOS conversion is

```math
z=-\frac{\lambda}{4\pi}\psi,
```

where psi is acquisition-referenced phase in radians and z is LOS displacement in metres. Positive z means decreasing radar range in this convention. The multiplicative factor is explicitly cast to float32 before multiplying the float32 phase history, as in the oracle capture. The temporal gauge is the first acquisition, 20141213. Each spatially referenced phase observation is the original phase minus its original value at p0.

An original zero connected-component label means unavailable phase. A zero phase value by itself is not treated as missing. In particular, the reference pixel must not be rejected because its spatially referenced phase is zero. All 288 edges remain in the supplied graph; each pixel may omit unavailable rows during inversion. Neither the observation values nor the coherence are inferred from a hidden target.

The public `analysis_mask` is the conjunction of positive original component labels across every edge. It includes 159762 pixels, including low-quality pixels. It is used for fixed evaluation and full-date uniform-weight comparison. It is distinct from the dynamically calculated spatial-plane fit mask: all 288 inversion observations available and temporal coherence >=0.7. The latter selects 159447 pixels in the independently reproduced upstream run. Poor-coherence pixels are not silently removed from reference-agreement scoring.

## 3. Temporal acquisition operators and their null spaces

For an edge e=(i,j), the incidence matrix A0 has A0[e,i]=-1 and A0[e,j]=+1. Deleting its first column gives A with shape (M,N-1). The first date is not estimated independently. With acquisition phase vector psi and psi[0]=0, the ideal observation vector is y=A psi[1:].

The integrated-velocity matrix B uses times

```math
t_i=\operatorname{float32}\left(\frac{\text{days}(i)-\text{days}(0)}{365.25}\right),
\qquad \Delta_i=t_{i+1}-t_i.
```

For i<j, B[e,i:j] contains the corresponding float32 intervals Delta; the other entries are zero. Reversed edges, supported by the helper, reverse the sign. B has shape (288,97), and an inter-acquisition phase velocity v in rad/year produces y=Bv. Integrating v gives psi[k]=sum_{l<k} Delta[l] v[l]. A and B therefore describe the same observations under an invertible cumulative transformation for a fully connected temporal network. The distinction matters when rows are missing or a network is rank-deficient: the selected SVD returns the minimum norm in velocity coordinates.

Every available chronological triple i<j<k contributes a row of C with +1 on edges (i,j),(j,k) and -1 on (i,k). Enumeration is lexicographic in acquisition indices. For this graph C has shape (286,288). In exact arithmetic C A=0; this identity is tested independently. The graph is connected before pixelwise masking. Component omissions are handled during inversion rather than by deleting whole acquisition dates.

## 4. Common spatial regions and integer ambiguity correction

### 4.1 Equal-label connectivity, filtering and sample selection

The common mask is true where every original component label is positive. Four-neighbour connected regions are identified. For a labelled interferogram, two adjacent pixels of different positive labels remain in different regions; converting the label image to a binary foreground would incorrectly merge them. `component_labels` therefore labels each distinct nonzero integer separately and numbers all surviving regions by their first raster index, matching upstream ordering.

The source cleanup removes regions of area strictly less than the threshold. If there is more than one region, the threshold is capped at 0.3% of image size: min(2500,450*600*0.003)=810 pixels. A 5 x 5 grey erosion of the integer label image, with reflected finite-frame extension and the source uint8 cast, identifies labels that would disappear; those original labels are removed before relabelling. This is the actual labelled-image operation, not an undocumented binary erosion. Source local-component boundary cleanup performs the erosion check a second time, which the port preserves. The observed label counts do not approach uint8 wraparound.

Two common regions survive, of areas 155840 and 2779 pixels. Each receives 100 distinct raster coordinates selected without replacement. The maintainer capture and deterministic local default use a fixed sampling seed, recorded as an implementation parameter rather than an acquisition parameter. The source RNG factory is controlled only at the private capture boundary; source mathematical functions remain unmodified. Captured sample coordinates and integer solutions are frozen independently. Region ordering for subsequent matching is descending area, as in the source.

The region containing p0 is assigned zero integer correction: all spatially referenced observations at the reference pixel are identically zero. No sparse inverse problem is solved for that reference region.

### 4.2 Closure equations and LASSO

At each sampled pixel, let y be the spatially referenced unwrapped phase vector. Compute the double-precision closure c=C y and wrap it into [-pi,pi):

```math
\operatorname{wrap}(c)=(c+\pi)\bmod(2\pi)-\pi,\qquad
q=\operatorname{round}\left(\frac{c-\operatorname{wrap}(c)}{2\pi}\right).
```

q is an integer vector of closure cycles. Floating round uses NumPy's nearest-even convention. To find sparse edge corrections u, solve the exact selected objective

```math
\min_u \|{-C}u-q\|_2^2+\alpha\|u\|_1,\qquad \alpha=0.01.
```

There is no division by the number of triangles and no factor 1/2 in the quadratic objective. The minus sign is required because correcting phase by +2 pi u must cancel the existing closure q. The corresponding KKT conditions are 2(-C)^T((-C)u-q)+alpha sign(u)=0 on nonzero coefficients, with absolute gradient <=alpha on zero coefficients.

The source uses the general CVXOPT interior-point LASSO backend. The local port keeps this objective and implements an explicit ADMM numerical backend using only SciPy linear algebra. This backend substitution is disclosed; intermediate ADMM iterates are not claimed to reproduce the interior-point iterates. Introduce x=z and scaled dual eta. For rho=1 the iterations are

```math
\begin{aligned}
(2D^TD+\rho I)x^{k+1}&=2D^Tq+\rho(z^k-\eta^k),\quad D=-C,\\
z^{k+1}&=\operatorname{soft}(x^{k+1}+\eta^k,\alpha/\rho),\\
\eta^{k+1}&=\eta^k+x^{k+1}-z^{k+1},
\end{aligned}
```

where soft(w,s)=sign(w) max(|w|-s,0). All sample columns share one Cholesky factor. Both z and eta start at zero. Every 20 iterations the implementation checks the maximum absolute KKT violation, primal residual x-z and dual residual rho(z-new minus z-old). All three must be below 1e-7; failure to converge in 20000 iterations raises RuntimeError. This is a real stopping test, not a fixed iteration count. The bound-data run converges in 380 iterations with maximum KKT violation about 6.98e-8.

Each sample's continuous solution is rounded to integer cycles. The component-wise median of these 100 rounded vectors defines the common region's correction. The 2779-pixel region has 28 nonzero edge corrections in the source run. All median corrections match the independent source capture. The complete 288 x 450 x 600 corrected phase stack is bitwise identical in a separate full-stage parity check.

### 4.3 Spatial propagation

For each original interferogram, subtract its reference-pixel phase. Clean and label its original local connected components as described above. Process common regions from largest to smallest. If all 100 sample coordinates of a common region lie inside a local region, that region receives the common region's edge correction, and matching stops. Otherwise its correction remains zero. Add +2 pi times the selected correction in place in a float32 phase image. Invalid original labels are not rewritten to manufacture observations; the inversion still uses the original availability array.

## 5. Coherence-weighted minimum-velocity-norm inversion

For one pixel, discard rows with original component label zero by marking those phase values NaN and then selecting valid rows. The full date/interval basis is retained. The source redundancy gate requires every retained unknown acquisition column of A to occur in at least one row. If not, return an all-zero phase history, zero quality and zero inversion count. The source does not impose an additional full-rank gate here; its truncated SVD supplies the selected minimum-norm result. A LAPACK linear-algebra exception likewise returns the initialized history and quality, with the retained observation count.

Coherence gamma is floored at 0.05. The selected weighting is the published **linear coherence** strategy, not inverse phase variance or squared coherence:

```math
\min_v \sum_{e\in E(p)}\max(\gamma(e,p),0.05)\,[B_ev-y_e]^2.
```

Thus sqrt(gamma) multiplies both B and y. The floor and float32 square root reproduce upstream `coherence2weight(...,weight_func='coh')`. `scipy.linalg.lstsq` solves the weighted system using its default divide-and-conquer SVD (`gelsd`), with relative singular-value cutoff 1e-5. The design and weighted observations are float32. No normal-equation shortcut is substituted. Retained velocity is integrated with the original float32 intervals; the result is explicitly stored in float32 before the LOS conversion.

Temporal coherence uses **unweighted** phase residuals even though the estimate is weighted:

```math
\gamma_{\mathrm{temp}}(p)=\frac{1}{|E(p)|}
\left|\sum_{e\in E(p)}\exp\{\mathrm{i}(y_e-B_ev)\}\right|.
```

A high value is a network-consistency measure, not proof of accurate ground motion. Four worker threads partition the 450 spatial rows; each pixel's original weighted SVD remains independent and unchanged. Threading does not alter the update order within a pixel. Numerical backend thread counts are controlled in the maintainer execution environment to avoid nested BLAS oversubscription.

For the uniform-weight scientific comparison, the notebook actively solves the same corrected B system on every pixel of the fixed all-edge-valid mask with no coherence weight. A single multiple-right-hand-side SVD is valid there because all these pixels have the same complete row set. This comparison retains the entire temporal history and source spatial structure; its result is not substituted for the weighted main output.

## 6. Atmospheric correction and affine spatial nuisance

The supplied ERA5 HDF5 product was independently produced by the dataset provider and is distributed here as the original float32 LOS array. It is not a reconstruction or a hidden-truth-derived screen. Its original temporal reference is the last date. The local procedure first subtracts its value at p0 for every date and then subtracts the resulting first-date image. The source capture applies the identical two gauge changes. Subtract this product from the apparent network LOS movie to obtain the atmosphere-corrected history.

The spatial fit mask is `(number_of_used_edges == 288) & (temporal_coherence >= 0.7)`. Following the source deramp operation, exclude pixels with a nonfinite or exactly zero mean history from the plane fit. Let F[p]=[y,x,1] in float32 source-pixel coordinates and Z be the atmosphere-corrected history in pixel-by-date orientation. Estimate all per-date plane coefficients by

```math
L=F_{\mathcal M}^{+} Z_{\mathcal M},\qquad
R=FL,\qquad R\leftarrow R-R[p_0,:].
```

The pseudoinverse is NumPy `pinv` with rcond=1e-15, preserving the selected source convention. R is cast to the input float32 dtype before reference anchoring. At individual entries where the original input Z equals zero, set the fitted ramp to zero, as the source does. Subtract R from Z. The operation does not use periodic spatial boundaries, a polynomial higher than degree one, geodetic coordinates instead of raster coordinates, or a hidden target. The estimated plane may absorb broad real deformation; that limitation is retained and discussed, not concealed by calling the correction ground truth.

## 7. Joint elevation residual and eruption model

The selected source mode uses scene-mean geometry, as in the distributed historical configuration, rather than silently upgrading to pixelwise geometry. The column ranges are R[x]=799547.2189154713+x*34.94343172072985 m. Their mean is 810012.77671583 m. The complete-scene mean sine of original incidence is 0.5953686237335205. The elevation sensitivity is g[t]=baseline[t]/(mean(R)*mean(sin(theta))). The zero first-date baseline fixes its temporal gauge.

The deformation basis uses the source's separate calendar-year convention Y=year+(day_of_year-1)/365.25. Its first-date difference tau is calculated in float64 and the basis columns are stored in float32. This is deliberately not replaced by a different leap-year denominator or by the float32 day-difference coordinate used by B. Its columns are [1,tau,tau^2/2,I(date>20170910),I(date>20180613)]. The coefficient of tau^2/2 is an acceleration in m/year^2. The step inequalities are strict.

Append this basis to g to obtain G with shape (98,6). Stacking the float64 geometry column and float32 basis promotes G to float64. For the deramped LOS history Z, solve

```math
\widehat\beta=\arg\min_\beta\|G\beta-Z\|_F^2,
\qquad \beta=[h,c,v,q,s_1,s_2]^T.
```

`scipy.linalg.lstsq(G,Z,cond=1e-8)` uses all 98 dates without extra coherence reweighting. The selected phase-velocity fitting option is off; no acquisition is excluded. The outputs are

```math
d_{\mathrm{final}}=Z-g\widehat h,\qquad
e_{\mathrm{model}}=Z-G\widehat\beta.
```

The final history therefore contains `deformation_model + model_residual`. Only the baseline-dependent elevation term is removed at this stage. The solver returns float32 final, height, five deformation parameters and residual. A fully analytic known-coefficient case tests the sign, acceleration factor, units and retention of the deformation signal. The actual G is checked elementwise against the independent original capture.

## 8. Evaluation conventions and numerical tolerances

There is no actual displacement ground truth for these observations. The hidden reference is the independently reproduced selected standard method, explicitly named `baseline_reference.npz`. Standard flattened cosine NCC and dynamic-range NRMSE evaluate each of the five required arrays at all fixed public-mask pixels. Every date and every coefficient participates. No fitted amplitude, phase, translation, sign reversal, rescaling or implicit crop is allowed. Per-array nonzero boundaries, fixed before the first cleaned full result, are NCC>=0.99999 and NRMSE<=1e-5. They assess numerical fidelity to this ported standard method, not physical centimetre accuracy.

Independent reruns of the original weighted SVD on nine bound pixels, including both common regions and a missing-observation corner, give zero float32 LOS differences. Region-median rounded LASSO coefficients and full corrected phase are separately checked; individual rounded sample coefficients differ as quantified below. Nonzero evaluation tolerances still allow ordinary backend variation rather than using a meaningless zero-error self-comparison threshold. No tolerance is weakened after inspecting the fresh full result.

The independent Judge additionally computes mean circular observation residual coherence from the submitted apparent network movie and the original phase after reference-pixel subtraction. Its forward phase is -4*pi/lambda times edge displacement differences. Complex exponentiation makes this invariant to integer 2*pi spatial ambiguities, as physically required; it is therefore an additional consistency check, not a substitute for spatial ambiguity/reference tests. The required boundary is 0.99 times the upstream public-mask mean temporal coherence. The public mask and original observations are legitimate calibration inputs for this metric.

Degenerate reference norms/ranges are rejected. Invalid keys, shapes, numeric dtypes, complex values, NaN or infinity are errors, not low scores. Valid poor reconstructions fail with exit code 1; malformed inputs or missing evaluator assets return 2. The notebook and plotting metrics are checked against the independently coded Judge on known vectors.

## 9. Validation and practical limitations

The upstream run, explicit local stages, analytic physics checks, fresh CLI, standalone Judge, strict input-contract tests, uniform-weight comparison and actively executed notebook have separate provenance records. The notebook presents the original dataset, physical equations, actual method, all major nuisance-correction stages, source/fresh comparison, parameter maps and residuals. Historical paper reports are never presented as measurements of this task.

No computational-imaging or domain reconstruction library is imported by the shipped code, tests or notebook. NumPy/SciPy supply only array operations, morphology primitives, SVD, Cholesky and matrix algebra. Matplotlib is used only for visualization. HDF5, MintPy, CVXOPT and the other original import dependencies exist solely in the private oracle environment; the cleaned runtime consumes NPZ directly.

Important limitations remain: LOS does not identify three-dimensional motion; common-region sparsity cannot repair arbitrary random unwrapping noise; closure consistency can coexist with wrong phase histories; ERA5 may leave water-vapour residuals; spatial deramping can remove large-scale geophysical signal; the polynomial/event model may mix acceleration and event steps; and the end-of-record 2018 step has limited post-event sampling. These are scientifically meaningful challenges in the bound real-data task rather than reasons to replace it by synthetic easy data.


### Verified LASSO backend discrepancy

The CVXOPT oracle and local ADMM solve the same convex LASSO objective, but their individual coefficient vectors are not identical. Among 28,800 sampled coefficients, nine rounded values differ, across four sample pixels. The actual difference has Frobenius norm 0.7920480220 and closure-image norm 0.0050718090 (maximum absolute closure difference 0.0003704676); it is not exactly a null-space vector. The local proximal KKT residual is 6.9795e-8 versus 6.0454930e-4 for the original backend. Local objective values are no worse, with maximum local-minus-original objective -7.9200372e-7. Independent active-support null-space perturbations preserve both closure predictions and L1 norm to 1e-12 on each of the four affected samples, proving genuine nonuniqueness; the observed differences also include original stopping precision.

Region-wise median rounded corrections agree for every one of the 288 edges. Full spatially repaired phase agrees bitwise at all 77,760,000 values, and all five complete cleaned output quantities have zero NRMSE against the original method. Validation therefore checks the convex objective/KKT condition and independent nonuniqueness proof at this stage, plus exact scientific region corrections; it does not claim bitwise agreement of the CVXOPT trajectory or individual sample coefficients. The original stricter coefficient test failure and all nine half-integer distances are retained in the private audit indexed by test/validation.md.
