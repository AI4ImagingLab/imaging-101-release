# Code design: HARDI CSA reconstruction

## Package boundaries and dependency graph

```text
src.main -> preprocessing.load_data -> solvers.reconstruct
                                      |-> preprocessing.normalize
                                      `-> physics_model.fit_operator -> basis -> indices
notebook -> same reconstruct + visualization -> physics_model.sample_odf
evaluation/judge.py -> NumPy + stdlib only (no src import)
```

NumPy handles local array arithmetic, SVD and safe NPZ I/O; SciPy supplies general complex SH and Legendre special functions. Neither package implements the task-specific CSA inverse as an opaque call. Matplotlib is imported only by visualization; no library module chooses a backend. main imports neither visualization nor hidden references. src/__init__.py declares the package without execution side effects.

## Files and array contracts

| Path | Array/shape/dtype/units | Caller/visibility |
|---|---|---|
| data/raw_data.npz | signal (1,24,24,1,160) float64 a.u.; bvalues (1,160) float64 s/mm²; directions (1,160,3) float64 dimensionless | Public loader and notebook; all measured volumes retained |
| output/reconstruction.npz | reconstruction (1,24,24,1,45) float64 sr^-1 | Written fresh by main/notebook; read by independent Judge |
| output/judge_report.json | status,passed,metrics,thresholds,errors; finite UTF-8 JSON | Written only on explicit Judge invocation |
| evaluation/baseline_reference.npz | reconstruction with identical output contract | Hidden original-method result, never anatomical truth |
| evaluation/metrics.json | relative_l 2_boundary 1e-6 and definition/provenance | Hidden Judge configuration |
| test/fixtures/upstream_stages.npz | basis (150,45),fit_matrix (45,150),normalized (24,24,1,160) float32, reconstruction (1,24,24,1,45) | Frozen independent original stages for parity |
| notebooks/reference_outputs/upstream.npz | reconstruction (1,24,24,1,45) | Original-versus-fresh review, not a solver input |
| notebooks/reference_outputs/cleaned.npz | Earlier measured cleaned reconstruction, same contract | Repeatability test only; remains immutable during new review |

Spatial axes are (image_x,image_y,image_z); no reorientation occurs. The coefficient axis follows degree then signed order. All arrays are NPZ with allow_pickle=False; scalar acquisition metadata resides in README rather than another JSON. No generator is shipped because observations are reused. Complete original legal attribution is consolidated in test/validation.md, with old files archived privately.

## preprocessing.py interfaces

### `load_data(path)`

path is str/Path to NPZ. Returns a dictionary of all arrays with stored dtypes/shapes unchanged; reads then closes the archive. It does not remove batch axes, floor magnitudes, verify the entire acquisition contract, or load metadata. Missing/corrupt archives, object arrays and unsupported files raise underlying NumPy/I/O errors. Missing expected dictionary keys are detected by callers' indexing, not a custom loader check. The fixed bound arrays are validated by acquisition tests. No file write or input mutation occurs.

### `normalize(signal, bvalues)`

signal has arbitrary leading dimensions followed by acquisition dimension 160, floating-convertible magnitudes in a.u.; bvalues is a one-dimensional NumPy vector (160,) in s/mm². Returns a new float32 array of the same shape with dimensionless attenuation. Internally copies, floors at 1e-5 and divides by float32 mean over bvalues<=50. No explicit input validation is added: an incompatible mask raises a NumPy indexing error; absent b0 entries or nonfinite source can yield warnings/nonfinite arithmetic. The selected acquisition contains 10 b0 samples and finite positive values. This helper is not the strict submission validator. It neither mutates caller arrays nor changes stored observations.

## physics_model.py interfaces

### `indices(order=8)`

order is a nonnegative integer maximum degree. Returns two NumPy integer vectors (l,m) of length K=sum (2 l+1) for even l<=order,45 for 8. Degree ascends, then m ascends. Odd maxima include even degrees below them; helper does not enforce a task-level maximum. Noninteger input naturally fails range; negative input produces empty vectors and is not a validated reconstruction mode. No I/O or mutation occurs.

### `basis(directions, order=8)`

directions shape (...,3), real floating-convertible, dimensionless nonzero Cartesian vectors. Returns float64(...,K) dimensionless orthonormal real SH samples. It normalizes only while computing polar angles, clips the acos argument and uses the legacy negative-m convention. Zero vectors cause invalid division; NaN/malformed dimensions propagate standard numerical/indexing failures rather than a tailored exception. The reconstruction excludes b0 vectors before calling. Uses generic scipy.special.sph_harm_y, not a reconstruction model. No caller mutation.

### `fit_operator(directions, order=8, smooth=0.006)`

directions (N,3) are weighted-shell vectors, N=150. Returns (B,A):float64(N,K) and (K,N), respectively. B samples the real basis; A maps dimensionless double-log signal to the CSA coefficients before constant-mode replacement. smooth is a nonnegative dimensionless quadratic penalty. Implementation forms (N+K,K) augmented design, calls NumPy pinv with verified NumPy 2.4.6 default rcond = 1e-15 (singular values at or below rcond times sigma_max are discarded) and applies the local Legendre/Laplacian factors. Invalid negative smooth can trigger sqrt warnings and SVD errors; this helper does not promise custom validation or arbitrary ill-conditioned data recovery. No caching, files or reference access occurs.

### `sample_odf(coefficients, directions)`

coefficients (...,45) in sr^-1 and directions (N,3) dimensionless map to float64(...,N) in sr^-1 by matrix multiplication with the fixed order-eight basis. No order parameter exists on this helper; mismatched coefficient count raises matmul dimensional errors. It does not clip signs or normalize the result. The notebook supplies 361 xy-plane directions and selected voxel coefficients. This is a forward representation sampler rather than the physical MRI Fourier acquisition operator.

## solvers.py and entry point

### `reconstruct(signal, bvalues, directions, order=8, smooth=0.006)`

Inputs signal (...,160) a.u., bvalues (160,) s/mm² and directions (160,3) dimensionless follow the acquisition ordering. Returns (...,K) float64 ODF coefficients in sr^-1, K=45 for the selected contract. Calls normalization, uses b>50 for shell/matrix construction, transforms clipped attenuation, multiplies by A.T and overwrites the first coefficient analytically. No initialization, checkpoint, loop termination or optimizer state exists. No array masking beyond measured-shell classification is applied. It writes no files, loads no truth/reference and does not mutate inputs. Preconditions/failure behavior inherit the numerical helpers; strict output shape/finiteness are independently enforced by the Judge.

### `main()` in src/main.py

No arguments. Resolves root=Path (__file__).resolve ().parents[1], so data/output paths are task-root-relative even from a foreign current directory. Loads data, removes the leading batch axis only from bvalues/directions, computes and writes output/reconstruction.npz with key reconstruction and returns the same fresh float64(1,24,24,1,45) array. Creates output/if absent and overwrites only the generated submission. I/O or numerical failures propagate to process failure; there is no internal fallback. The executable guard calls main; importing the module performs no reconstruction or backend changes.

## visualization.py interfaces

| Exact callable | Inputs and returned figure | Side effects and limitations |
|---|---|---|
| `comparison(signal, fresh, upstream)` | signal (1,24,24,1,160) a.u.; two coefficient arrays (1,24,24,1,45) sr^-1; returns Figure with 3 maps | Mean first 10 volumes, nonconstant coefficient norm, and max coefficient discrepancy; transposes for image x/y display, origin lower; creates figure only, no save/clip of data |
| `angular_comparison(fresh, upstream)` | Two coefficient arrays as above; returns Figure with 3 polar axes | Samples 361 directions in xy plane at voxels (5,5,0),(12,12,0),(18,18,0); displays radius>=0, explicitly documented as display-only; no files |

Both assume compatible shapes and valid Matplotlib data; shape/index errors are ordinary NumPy/plotting exceptions. They do not compute Judge metrics or retrieve missing results. Plot calls leave figures available to notebook plt.show ().

## Independent Judge callable contract

### `score(estimate, reference)`

Receives matching real numeric arrays already validated by evaluate. Returns Python float relative Euclidean error across all entries. A nonpositive reference norm raises ValueError. It applies no alignment, crop, scale or magnitude conversion. Direct helper calls do not perform the shape/dtype/finiteness checks belonging to evaluate.

### `evaluate(prediction, asset_dir=None)`

prediction is str/Path, caller-relative unless absolute. asset_dir is optional evaluator-test injection; default is the Judge's own directory, independent of current working directory. Loads metrics, requires a finite positive boundary, prefers ground_truth.npz if it exists and otherwise requires baseline_reference.npz. Both reference and prediction must be exact (1,24,24,1,45) float64 finite arrays with reconstruction key. Score measures every coefficient. Returns (report,code), where report is a dict with status/pass flag/measured metrics/threshold comparison/errors;0=pass,1=valid quality failure,2=malformed/evaluator error. Exceptions are caught and returned as error reports with empty metrics. This callable does not write the report or execute solver code.

### `main()` in evaluation/judge.py

Parses required --prediction and --report. Calls evaluate, creates report parents and writes finite UTF-8 JSON. Report-write failure prints a diagnostic to stderr and returns 2. The executable guard exits with that return value. No src import, submitted-code execution or reference substitution is allowed. Default reference is a method result, not actual truth.

## CLI versus notebook execution

CLI: three public arrays -> local normalization/CSA -> one output NPZ. Judge runs separately. Notebook: full data table/source -> actual public array load -> self-contained physics -> same active reconstruct -> generated submission -> independent Judge -> separately loaded original references -> all metric/map/angular comparisons -> limitations. Historical reference_outputs remain byte-identical; fresh review arrays/timing may be archived privately. Importing reference data for comparisons never supplies the solver answer.

## Test-to-interface coverage

| Case | Interfaces/contract checked |
|---|---|
| test_original_stages_and_pipeline | normalize, basis, fit_operator, reconstruct against independent original fixtures |
| test_standard_entry_point | main output key/shape/bytes versus returned fresh array |
| test_archived_notebook_measurement_consistency | Previously measured cleaned snapshot repeatability, not independent parity |
| test_basis_orthogonality_and_antipodes | basis normalization/signed convention and inversion symmetry |
| test_adjoint_identity | Angular matrix forward/transpose dot-product identity |
| test_isotropic_odf_and_exact_normalization | reconstruct/sample_odf constant density and sphere integral |
| test_normalization_is_scale_invariant_and_float32 | normalize's scale and precision |
| test_indices_and_acquisition_documentation |45 coefficients,10/150 split,b2000,gradient rounding,README cutoff |
| test_metric_known_case | Independent Judge Euclidean metric |
| test_contract and evaluator configuration tests | Missing/key/shape/dtype/complex/NaN/Inf/poor/good; absent/zero/wrong reference and invalid boundary |
| test_foreign_cwd_and_report_write_failure | Asset path resolution, report parent creation,exit code 2 on write failure |

Interface documentation intentionally distinguishes implemented numerical preconditions from explicit error checks. No hypothetical validation API or alternate reconstruction family is described as shipped functionality.
