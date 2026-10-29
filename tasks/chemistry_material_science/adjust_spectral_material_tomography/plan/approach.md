# Implemented Approach: Joint Spectral Material Tomography

## 1. Scope, selected variant, and scientific status

The bound benchmark uses every acquired frame of the real Sittner W/Au/Pb spectral micro-CT slice: 600 angles, 128 spectral channels and 257 detector positions. There are 65,025 image pixels and four shared material components. The standard path first reconstructs all 128 spectral channels for 100 SIRT iterations, estimates a quartz-matrix spectrum from those fresh channel images, and performs up to 1,000 ADJUST outer iterations on 65 channels. Neither the matrix spectrum nor the reconstructed spectral images are available as solver input.

The implemented inverse algorithm is specifically the **reference-release equality-constrained variant**, with the documented invalid-projection skip guard removed. It is not advertised as an exact implementation of every mathematical statement in the paper. Three distinctions are essential:

1. Paper equation (11) constrains both dictionary row and column sums to at most one. The released `ADJUST.m` instead uses row equality and column inequality. This task preserves the released equality variant.
2. The released `projectTwoSimplexConstr` performs alternating row/column feasibility operations, not the paper's stated orthogonal projection. Independent constrained quadratic programming gives different nearest points on small examples. The implementation preserves that release algorithm and makes no orthogonal-proximal or global-convergence claim.
3. The released routine skips every operation when the sum of the proposed coefficients is below one. Analytic counterexamples and the earlier full43-atom investigation demonstrate that this can return negative mixtures. The current four-basis unmodified1000-step run instead returned a feasible best iterate; its behavior must not be described as the historical43-atom failure. The task removes only this invalid guard, leaving the feasibility operations, iteration limit, and stopping tolerance intact. Original and corrected executions remain separately identified in validation evidence.

The numerical forward discretization is an exact matched CPU line-fanflat model, rather than the demo's unmatched CUDA Joseph-forward/pixel-driven-backprojector pair. The paper allows a generic known linear W. All geometry and data are retained; this is a documented discretization selection, not a reduction of the inverse problem. The full local 46,901,466-weight matrix is compared independently to the frozen ASTRA matrix.

## 2. Measured counts and ordering

I(a,c,d) is the uint16 acquired photon count, with acquired angle a=0..599, energy channel c=0..127 and detector index d=0..256. I0(c,d) and D(c,d) are measured flat and dark counts. Convert to float64 **before** subtracting unsigned counts. Define

```math
L(a,c,d)=\log\frac{I_0(c,d)-D(c,d)}{I(a,c,d)-D(c,d)}.
```

All dark-corrected intensities and flat values are strictly positive in the actual dataset, so there is no synthetic noise model, logarithm clipping, or missing-data mask. Reverse a, then flatten rays in MATLAB-compatible order j=a+600d. The data matrix is Y128 in R^(154200 by 128). Image vector index is i=row+255*column; increasing image row points downward, while physical y points upward. The complete physical image is preserved.

The energy array gives detector-bin midpoints20.90to160.60keV at width1.10. Independent official NIST-table interpolation verifies that the three supplied rows match this actual midpoint grid to relative4.1e-16. Filename limits20.35and161.15are bin edges, not a spectral-grid mismatch.

## 3. Fan-beam geometry and units

Let SOD=57.999001 mm, SDD=883.081116 mm, detector pitch h=0.80 mm, M=SDD/SOD and image-pixel pitch delta=h/M mm. Distances passed to the projection model are measured in image pixels: s=SOD/delta and o=(SDD-SOD)/delta. The detector's displacement per channel is u=h/delta. At angle theta,

```math
\mathbf{s}(\theta)=(s\sin\theta,-s\cos\theta),\quad
\mathbf{u}(\theta)=u(\cos\theta,\sin\theta),
```

```math
\mathbf{d}(\theta)=(-o\sin\theta,o\cos\theta)+q\mathbf{u}(\theta),
\quad q=257/2-127.379570.
```

Here the bold source s and scalar source distance s are distinguished by context; d is the **central** detector location, and q is a detector-pixel offset. Detector center t has coordinate d+(t-128)u. Source angles are the stored float64 inclusive linspace(0,2*pi,600), including the duplicated endpoint convention of the released demonstration.

Pixel centers have x=column-127 and y=127-row, with square bounds at +/-127.5. Outside this support the attenuation is zero. W(j,i) is the intersection length inside square pixel i in **image-pixel lengths**. Its product with a per-pixel optical-depth coefficient is dimensionless. W is target independent; it depends only on public geometry.

The acquisition settings contain SDD=883.081305 mm and rotation-axis estimates slightly different from the released reconstruction constants. The selected values above remain explicit and tested. They are not silently replaced by the settings-file values.

## 4. Explicit line discretization and exact transpose

The matrix builder traverses the major direction of each ray. With vector from detector center to source (Rx,Ry), a ray is vertical when |Rx|<|Ry|. Its horizontal displacement per row is -Rx/Ry, and its length through a complete row is ell=sqrt(Rx^2+Ry^2)/|Ry|. The horizontal case exchanges coordinates and uses length sqrt(Rx^2+Ry^2)/|Rx|.

For a vertical ray, let c be the column-coordinate intersection at a row center, n=floor(c+1/2), e=c-n, v=|Rx/Ry|, S=(1-v)/2 and T=(1+v)/2. If -S<=e<=S, assign the whole length ell to pixel n. If e<-S, assign ell*(e+T)/(T-S) to n and its complement to n-1. If e>S, assign ell*(e-S)/(T-S) to n+1 and its complement to n. Omit contributions outside the square support. Horizontal rays use the same construction with rows and columns exchanged. Axis-aligned rays use the central branch rather than dividing by zero in a crossing branch.

Float32 conventions are intentional. Detector basis components are rounded to float32 before computing the detector-start coordinate, following the source converter's mixed-precision order. Ray-center coordinates then advance by repeated float32 additions, not a mathematically equivalent multiplication by the row index. These details matter for near-axis rays: substituting double arithmetic can change individual small crossing weights significantly. The assembled matrix stores float32 weights; ADJUST promotes these exact values to float64. The original ASTRA sparse export supplies an independent full-weight fixture, rather than a locally synthesized expected matrix.

The inverse-gradient operator is exactly W transpose, not an independently interpolated backprojection. Sparse storage, multiplication, and transposition are general numerical primitives; every acquisition weight is explicitly formed locally. Within each CSR row the source kernel's dominant-axis traversal is preserved: vertical rays visit image rows then columns, horizontal rays columns then rows. This is deliberately different from globally sorting MATLAB column labels, which changes floating-point summation in the iterative method. An independent dot-product identity, analytic rays through square support, positivity and geometric coordinate tests supplement upstream parity.

## 5. Full spectral reconstruction and matrix-response estimation

For each of all 128 energy channels, use unconstrained simultaneous iterative reconstruction (SIRT) from x0=0. Let rj=1/sum_i Wji and ci=1/sum_j Wji, with zero inverse for a zero row or column. At iteration t,

```math
X^{t+1}=X^t+\operatorname{diag}(c)W^{\mathsf T}
\operatorname{diag}(r)(Y_{128}-WX^t).
```

X is the 65025 by 128 spectral image matrix. Use exactly 100 iterations, relaxation one, no positivity projection and no early stopping. Storage/computation for this stage are float32 to match compiled ASTRA CPU SIRT. Simultaneous right-hand-side operations retain independent per-channel updates; they do not introduce a block-coupling algorithm. The actual floating-point evaluation begins each residual at its measurement and subtracts source-ordered pixel products. It is implemented with the explicit augmented CSR [I,-W] multiplying stacked [Y;X], rather than first forming W-times-X and subtracting that rounded result. Row/column reciprocal is zero for absolute sum<=1e-6. During this stage, rays use detector-fast/angle-major native acquisition order and pixels use row-major labels; output returns to MATLAB order. These conventions reproduce independently captured original normalizers, first two updates and a full100-step real channel bitwise. The complete128-channel calibration is a separate mandatory validation gate.

CSR precision changes explicitly copy indices/pointers and cast only weights. SciPy's ordinary astype can otherwise sort an unsorted matrix, silently destroying the original ray traversal. The independent actual-solver-boundary test checks this property after conversion, in addition to the matrix-builder fixture. ADJUST uses its original MATLAB ray labels; SIRT's CPU backprojection uses native ray order. They are separately preserved conventions. ROI reduction materializes the channel/row/column images as C-contiguous before the source float32 mean, preserving the actual observation-derived spectrum rather than only its mathematical average.

Reshape each column of X into a 255 by 255 image in column-major order. The matrix spectrum is the mean over rows and columns 130..140 inclusive in one-based notation, equivalently slices [129:140,129:140]. This 11 by 11 region is an explicitly selected observation-derived matrix patch, not a hidden target or true material map. Its interpretation as quartz follows the published experiment; contamination or reconstruction artifacts can bias it. It is recomputed in the entry point and active notebook.

Let D3 be the public 3 by 128 mass-attenuation dictionary in cm^2/g. Form T128 by vertically concatenating 0.015*D3 and the fresh matrix spectrum, where the nominal 0.015 g/cm^2 basis scale gives optical depth per image-pixel length. Select columns 29..93 inclusive to obtain T in R^(4 by 65) and Y in R^(154200 by 65). This is 65 channels, despite the original demonstration's comment saying 64. No channel is discarded before the self-calibration stage.

## 6. Joint unknowns, objective and gradients

Let A in R^(65025 by 4) contain abundance maps, R in R^(4 by 4) contain dictionary coefficients and F=RT in R^(4 by 65) contain component spectra. Their predicted measurement is WART. The selected equality-release feasible sets are A>=0, A*1<=1, R>=0, R*1=1 and transpose(R)*1<=1. The map sums apply separately at each pixel; the mixture column inequality prevents unrestricted reuse of one atom across all components.

The misfit at lifted observation B is

```math
f(A,R;B)=\frac{\lVert WART-B\rVert_F^2}{2\lVert Y\rVert_F}.
```

The normalization is **norm(Y)**, not norm(Y)^2, the number of pixels, the number of observations, or the norm of the changing lifted data. Define E=WART-B. The exact task gradients are

```math
\nabla_R f=(WA)^{\mathsf T}(ET^{\mathsf T})/\lVert Y\rVert_F,
\quad
\nabla_A f=W^{\mathsf T}(EF^{\mathsf T})/\lVert Y\rVert_F.
```

No total-variation penalty, spatial smoothing, learned prior or Tikhonov term is added to ADJUST. The constraints are its regularization. The generic sparse matrix multiplication may cache WA inside the R subproblem because A is fixed throughout that subproblem; this is an algebraic reuse, not a different objective.

## 7. Initialization and outer update order

The benchmark fixes seed10 with the independently verified Octave twister convention. Python `random.Random(10)` generates the same uniform sequence; values fill arrays in column-major order. Draw A first, normalize every row to sum one, then draw R. This is algorithmic initialization, not a simulator seed or specimen description. No archived reconstruction initializes either stage.

Initialize accumulated residual Q=0, rho=0.01 and best residual=+infinity. For outer iteration k:

1. Set B=Y+rho*Q, using Q before its current update.
2. Hold A fixed and update R with the local SPG subsolver and repaired alternating feasibility routine.
3. Set F=RT, then hold F fixed and update A using SPG and row sub-simplex projection.
4. Compute residual Edata=Y-WAF and accumulate Q=Q+Edata.
5. Record relative residual ||Edata||F/||Y||F, iterate progress ||R-Rold||F+||A-Aold||F, absolute residual change, and the rho actually used.
6. Retain A,R,F if their relative residual is no greater than the best previously seen residual. Return this coherent best triple, not a mix of independently best arrays and not necessarily the final iterate.

Stop before the next update if relative residual, iterate progress, or absolute residual change is below 1e-9; the first residual change is defined as one. Otherwise restore rho to0.01 when rho>0.05 and residual change>1e-3, then multiply rho by10 when residual change<1e-5. Preserve this order, and use a maximum of1,000 outer iterations. The observed trajectories and actual stopping counts must be recorded; the notebook cannot stand in for an unexecuted optimization run.

## 8. SPG subproblem mathematics and counters

Each call projects its input, evaluates objective and gradient, and initializes its evaluation counter to one. The maximum is two evaluations, with the original `while evaluations<=maximum` convention: an ordinary call may perform two accepted updates and end with three evaluations. Backtracking evaluations also count. This is not equivalent to an unconditional two-gradient-step loop.

At the first step use BB scale alpha=1; later use alpha=(s dot s)/(s dot y), with s the difference between accepted iterates and y the gradient difference. Replace alpha with one when alpha<=1e-10 or alpha>1e10. Direction d=-alpha*g is unprojected. Stop when g dot d>-1e-12. Start step length min(1,1/sum(abs(g))) on the first step and one thereafter.

The trial follows the feasibility arc xnew=P(x+t*d). Accept when fnew<=max(recent objectives)+1e-12*g dot(xnew-x). The nonmonotone memory is two. A rejected step uses the original two-point cubic interpolant based on both function values and directional derivatives, falling back to halving for a non-real discriminant or illegal objective. Safeguard the new step between0.001 and0.6 times the preceding step. Stop on maximal component displacement below1e-12, objective change below1e-12 or exhausted evaluations. The cubic derivative uses gnew dot d, preserving the released projected-arc convention.

Map projection sorts each pixel's four candidate abundances and subtracts a nonnegative threshold so its sum does not exceed one. Dictionary feasibility alternates row equality-simplex and column sub-simplex operations until their Frobenius discrepancy is below1e-6, or10,000 passes. Removing the invalid total-sum guard is the sole selected correction to the released inverse control flow. Independent QP examples deliberately verify that this map is feasible but generally **not** the closest point to its input.

For descending entries s and allowed mass q=1, form candidate thresholds as `(1/j)*(cumsum(s)[j]-q)` for j=1,...,4, with reciprocals computed before multiplication. Select the last j whose sorted entry exceeds its candidate threshold, truncate a negative threshold to zero for the inequality case, then subtract and clip at zero. The reciprocal-first association preserves the original diagonal-matrix multiplication, whereas direct division by j can differ one ULP at active count three. An independent original initial-weight trajectory checks all 39 alternating row/column half-steps bitwise.

## 9. Scientific comparison paths

The maintainer notebook retains RU, UR, classical joint and ADJUST groups. cJoint locally extracts the release's 500-iteration residual-lift algorithm with three inner evaluations, nonnegative spectra, box-constrained maps0..1 and1e-9 SPG progress/sufficient-decrease tolerances. Its initialization is unnormalized and independent, and its stopping residual/progress thresholds are1e-6.

RU solves each channel's (W'W+0.001I)x=W'y by ordinary CG with zero initialization,20 iterations and relative tolerance1e-6, clips negative channel images, then factorizes them. UR factorizes the measured spectral sinograms first, then solves four material tomograms with the same regularization and100 CG iterations. These independent right-hand sides may be evaluated together, with separate CG coefficients and stopping status; this is not block CG.

Both sequential comparison ports use ten multiplicative NMF initializations,100 updates each, followed by30 alternating exact four-variable nonnegative least-squares fits on the best start. The small NNLS solver enumerates all15 nonempty active supports plus the zero solution. It is checked against independent SciPy NNLS, not against itself. The unavailable MATLAB Statistics Toolbox `nnmf` is not imported or imitated by a domain shortcut. These are explicitly labeled mathematical comparison ports; no claim of closed-source MATLAB numerical parity is made. Their unconstrained factor scales differ from the dictionary-constrained abundance scale and are labeled accordingly in figures.

The source example's additional "ADJUST-true" call uses T_gt exactly equal to T; it is another initialization of the same method, not independently measured truth. The notebook preserves that comparison as an actively computed same-dictionary stability repeat with seed11, alongside the seed10 benchmark. It retains full data, dictionary, iterations, and corrected feasibility rule. The explicit seed convention replaces an unavailable global stream after MATLAB nnmf and is disclosed rather than represented as its numerical reproduction.

## 10. Evaluation, physical limits and environment

### Scalar arithmetic and source execution roles

Frobenius norms preserve the selected original Octave8.4 serial scaled-sum convention in column-major order. Starting with scale s=0 and accumulator h=1, visit t=abs(value): if t=s increment h; if t>s replace h by h*(s/t)^2+1 and set s=t; otherwise, for nonzero t, increment h by (t/s)^2. Return s*sqrt(h). Local code groups intervals between record maxima, using sequential cumsum within each fixed-scale interval; it does not replace these additions by a parallel sum. Both the objective numerator norm-then-square and the observation normalization use this routine. This convention reproduces the original bound-observation norm exactly, whereas mathematically equivalent parallel norms caused amplified iteration drift.

The objective subsequently evaluates the scalar power of this norm using the selected Linux libm `pow` primitive through the explicit generic numerical context. The order remains `0.5 * pow(norm, 2) / norm(Y)`. This is a platform arithmetic requirement: on an actual update164 residual, both norm implementations returned exactly780.4069459910742, but the original Linux power returned609035.0013511154 whereas Windows Python power and direct multiplication returned609035.0013511153. That one-ULP objective difference entered backtracking before any cubic input differed. The local helper implements general scalar power without a value-dependent branch; it does not substitute multiplication for the selected original power operation. Outside the source-compatible context it uses the host `math.pow`, whose last-bit equality across platforms is not assumed. The denominator, gradients, residual arrays, cubic formula and stopping conditions are unchanged.

The independent original scalar fixture covers412 powers, including three real residual norms, all384 cubic d1 operands from original updates101..164, numerical boundaries and non-square powers. Every generic power matched bitwise. On these384 actual cubic operands the original d1^2 and multiplication agree, and every returned cubic step matches; this bounded observation does not assert that power and multiplication are interchangeable on arbitrary floats. The completed source-role port therefore evaluates the cubic discriminant as pow(d1,2)-gtd*derivative_new through the same helper, preserving the original power operation for future inputs. Original explicit multiplications in scaled-norm accumulation remain multiplications. The repaired actual update164 spectral subproblem matches all16 original SPG stages and its final R bitwise. Complete fresh1000 parity remains a separate mandatory check.

SPG products use column-major vectorization. In the explicit SourceBlas context, distinct-operand vector products use Linux OpenBLAS DDOT and self products use DSYRK; the latter matches the actual source s-transpose-s optimization and differs from DDOT by one ULP on an independently captured original BB case. The first-step gradient L1 sum instead uses sequential column-major addition. For dense matrix products, F-output NumPy suffices for short contractions; inner dimension>=4096 uses the same generic Linux DGEMM as the original numerical environment. The dimension-only split is validated against original real subproblem products, not selected by any hidden target or scalar value. The cubic backtracking formula explicitly forms its numerator/denominator ratio before multiplying by the old step, preserving the source parentheses. These general primitives and local operation ordering reproduce the independent three-evaluation deep-state SPG bitwise. The worker has no acquisition or inverse logic; all constraints, objectives, derivatives, updates and stopping rules remain executable local source. The WSL/Linux OpenBLAS0.3.26 environment exception and library hashes are documented separately.

The selected `cuda_ordered` mode keeps W and W-transpose multiplication on CUDA, forces dense inputs C-contiguous, and returns F-contiguous host arrays. One explicit general CUDA thread accumulates each output in original CSR order using rounded multiply and add without fusion or atomics. Float32 SIRT evaluates exactly the same augmented residual and native transpose; CPU float32 normalizers, pointwise updates and ROI reduction are preserved. Independent original CPU100 and cleaned orderedGPU100 trajectories are bitwise equal at all captured checkpoints. The historical `cuda_sparse` PyTorch operation is nondeterministic on identical inputs and is retained only as a rejected diagnostic. Dense factor arithmetic stays on the host, with the explicit SourceBlas context supplying the validated general contractions. This matches the independently executed source relay's actual CPU/GPU roles. Complete benchmark validation remains pending. Earlier all-GPU and entrance-order-only trajectories remain failed diagnostics; no full parity is inferred from their similar residuals.

An independent Judge forward check uses fixed reconstruction views 0,50,...,550, every detector, and every retained bin. For projected abundance maps P=Wv*A and submitted spectra F, its statistic is ||P*F-Yv||F/||Yv||F, with public Beer-Lambert observations Yv and no fitted scale. Ray weights independently implement the selected float32 coordinate/intersection convention; an exact-double Siddon helper is used only for analytic checks because it is a different finite-precision discretization. The upper bound is fixed before measuring the cleaned full output as 1.1 times the independently validated source residual on these rays. It complements reference agreement and does not establish unique chemical truth.

Write one NPZ containing reconstruction(1,4,255,255), spectra(1,4,65) and dictionary_weights(1,4,4), all finite real float64. A single Hungarian assignment based on normalized spectral distance resolves the unordered-component symmetry and is shared across all arrays. No scaling, translation or cropping is allowed. Every component of all three unknowns is scored, so a visually similar matrix map alone cannot hide missing small metal components. NCC is uncentered cosine similarity; NRMSE uses the reference's dynamic range.

No ground-truth array is manufactured. The final hidden baseline must be an independently validated corrected reference-release result; at this in-progress stage full parity and the baseline asset remain unverified as recorded in test/validation.md. Nonzero numerical thresholds must be justified using independent original/local reruns and are not evidence of absolute chemical accuracy. Observation residuals, positivity, row/column bounds, material morphology and preserved comparison groups provide complementary scientific evidence. The physical spectrum approximation, ROI-derived matrix estimate, generic mixture model and absence of true material density measurements remain explicit limitations.

The complete maintained benchmark uses the pinned research environment and CUDA float64 for ADJUST, float32 for SIRT. CPU analytic/stage checks and the complete original compiled CPU SIRT are separately recorded. A complete cleaned CPU benchmark is not claimed unless actually executed. CuPy supplies compilation and device buffers for explicit general CSR arithmetic; PyTorch supplies only the retained diagnostic sparse/tensor modes; no ASTRA, SPOT, MinConf or domain library is imported by the cleaned runtime, tests, Judge or notebook. Upstream source, isolated oracle environments, relay scripts, logs and paper PDF remain in the maintainer archive.

Numerical-reference boundaries were frozen before the selected fresh full reconstruction: each component of maps, spectra and weights requires NCC>=0.999999 and dynamic-range NRMSE<=1e-6. Nonnegativity, map-sum excess and R-column-sum excess have1e-10 tolerances. R-row-sum error alone has2e-6 tolerance, the four-entry Cauchy bound for the source alternating feasibility discrepancy below1e-6. Fixed12-view/all-detector/all-selected-energy observation relative L2 must be<=0.17007280335223202, or1.1times the prior independently measured corrected source0.15461163941112. These tolerances test reproduction, constraints and public-data fit; no image/composition truth exists. Full1000 parity, fresh Judge and executed notebook remain required.


## Historical feasibility implementation qualification

This benchmark uses the reference-release row-equality variant of ADJUST with one explicitly documented correction: its dictionary feasibility routine must run even when a trial array's total sum is below one. The original skip condition can return negative, infeasible mixtures. Its alternating row/column feasibility map is retained and should not be confused with an exact Euclidean projection onto the intersection, or used to claim the paper's orthogonal-proximal convergence guarantees. The paper itself formulates weaker row inequalities. These distinctions matter for both mathematical interpretation and reproducibility.



## 2026-10-29 current v2 public joint-factorization contract

The output remains abundance maps A(4,255,255),65-bin spectra F(4,65), and dictionary weights R(4,4), stored batch first as float64. Component permutation is a gauge: all three arrays must be permuted together. The independent standalone Judge does not import src, use initialization identity, or compare method arrays. It tests A,R,F nonnegative, per-pixel sum(A)<=1, each R row sum1, each column sum<=1, F=R T, and all600views/all65selected energies through the public fan-beam optical-depth model. The full squared residual may not exceed independently attained standard by more than factor1.00000001; the original twelve-view relative residual boundary.1700728033 is additionally retained. Dictionary discrepancy limit is1e-6, R row sum2e-6, other simplex/nonnegative limits1e-10.

T is not hidden pure-material truth or target-derived calibration: its first three atoms are the visible calibration and its fourth is the explicitly defined observation-only100step,all128energy SIRT ROI statistic. README defines X0=0, the exact100step recurrence, ray/voxel normalization cutoff1e-6, float32 arithmetic, and11by11ROI indices129..139. The independent original observation-derived T is cached in the hidden baseline only to avoid repeating that deterministic functional inside every Judge; new fresh T was bitwise identical. This cache is not solver input and no pure quartz truth is asserted.

The real full cuda_ordered pipeline completed100SIRT+1000ADJUST, exit0,2002.839s CLI runtime (2004.266s controller). Three required arrays, T and every entry of all four1000step trajectories were bitwise identical to the independently completed original: selected best iterate795, residual.153078688318426. The independent corrected-original ADJUST epoch wall time6567.527808189392s, independent original cJoint3490.8370950222015s, historical unmodified4385.1106532s, and this fresh run have distinct provenance. Independent Judge full fan-beam projection gave.15307868831848234, a5.6e-14difference from the source residual. The baseline was added only after these actual gates; it is the independent original method output, not physical truth and not the new fresh output.

Validated backend boundary: Windows10.0.26200,Python3.11.11,PyTorch2.9.1+cu128,CuPy13.6,RTX4080SUPER16GiB,driver610.88. Generic Linux OpenBLAS0.3.26 Haswell4threads and Linux libm are reached through the existing WSL worker, explicitly configured via IMAGING_BLAS_LIBRARY; native Windows NumPy/OpenMP use1thread. Default system libblas probe failed (no openblas_get_config); the configured existing library probe and real complete pipeline succeeded without installation. This is an explicit Windows/WSL numerical exception, not a claim of a wholly native Windows CPU run. Live notebook guard requires9.5GiB initial physical/commit headroom and stops its owned processes below6GiB; resource samples/logs are private.

Stage/mathematical/Judge tests62passed in32.87s on this exact configured backend. The active full notebook completed exit0 in5315.453s with a research-runtime kernel and10800s per-cell timeout. It genuinely computed RU/UR/cJoint500 and the second ADJUST1000 initialization, preserving every original comparison. All9rendered figures and every metric panel were actually reviewed; zero cell errors. Source/fresh cJoint maps and spectra were exact; the second-initialization difference is retained as model/optimizer sensitivity. Final full pytest test/ run passed64tests in61.37s, including complete ADJUST history/quartz parity and fresh cJoint parity; exact commands/stdout/exit are in semantic_v2/adjust_complete_pytest.json. Canonical notebook outputs were refreshed only from this actual executed artifact after strict per-cell source identity checks, retaining portable original kernel metadata. No further reconstruction was required. Initial ADJUST missing-output/missing-baseline evaluator failure and earlier divergence evidence remain historical; no cached result was substituted for the fresh reconstruction.

Private evidence: benchmark_readiness_20261029/semantic_v2/adjust_spectral_material_tomography/{fresh_full_pipeline_execution.json,fresh_original_full_parity.json,independent_original_physical_validation.json,backend_probe.json,backend_configured_probe.json,notebook_execution.json}; stage tests in semantic_v2/adjust_stage_pytest.json. Raw observations and independent actual-truth assets were unchanged. No fresh blind-agent solve is claimed.
