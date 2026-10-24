# Shipped interfaces and execution flow

The cleaned runtime subset is NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1 plus Python3.11 standard library. The reference runtime imports no SMT domain library, source checkout, NIfTI package, upstream environment or other task. Public arrays are already native NPZ observations. Jupyter and pytest are shared development tools, not task runtime requirements.

## Standard path

From the task root, `python -m src.main` calls `main(argv=None)`. Options `--data` and `--output` default to `data/raw_data.npz` and `output/reconstruction.npz`. The CLI loads observations, fits every fixed-mask voxel freshly, creates the output parent and saves exactly four required fields in one compressed NPZ. It prints actual runtime, voxel count, convergence count and maximum update count. Imports do not start a fit or change a Matplotlib backend.

`preprocessing.load_data(path)` uses allow_pickle=False and returns a dictionary with diffusion_signal int16(1,96,96,19,496), bvalues float64(1,496), gradient_directions float64(1,496,3) and observation_mask bool(1,96,96,19). It checks exact keys, shapes/dtypes, finiteness and nominal shell values. It preserves intensities, native coordinate order and every acquisition. The mask was fixed before original/local fitting; no segmentation library runs during reconstruction.

## Forward primitives

`physics_model.spherical_mean(bvalues,parallel,perpendicular)` returns broadcast float64 orientation responses, evaluating the explicit error-function formula with continuous isotropic/b0 limits. Inputs use s/mm² and mm²/s. The max/min swap convention follows source. It does not load data or mutate inputs.

`transform_parameters(logits)` maps (...,2) real logits to (...,2) physical(v,D), with maxima1 and3.05e-3mm²/s and source exponential arithmetic/overflow guard. `predicted_signal(parameters,bvalues,b0)` takes(n,2) physical parameters, one-dimensional b values and(n,) baseline intensities and returns(n,n_b) predicted scanner-unit shell means.

`signal_objective(logits,observed,bvalues)` takes(n,2) logits, observed(n,n_acquisitions) and bvalues(n_acquisitions,) and returns(n,) unnormalized positive-b intensity SSE. It averages observed b0 entries, evaluates each distinct positive b once, then gathers model means back to every native directional residual. No reference arrays, orientation estimate or target-derived calibration enters this computation.

## Explicit inverse

`solvers._sort_three(values,indices)` and `_sort_last(values,indices)` are deterministic three-vertex rank operations. Both take(n,3) objective/identity arrays and return rank identities. Initial/shrink insertion reproduces the fixed original small-array equal-key order safely; updated-worst insertion is strict-less. They do not delegate the non-strict original comparator to a generic sort.

`fit_signals(observed,bvalues,max_iter=10000,return_history=False)` accepts a finite(n,n_acquisition) intensity matrix with at least one b0 and matching one-dimensional b values. It returns(n,2) float64 physical parameters plus diagnostics: `iterations` int32(n,), `converged` bool(n,), `objective` float64(n,). With history requested, also returns `parameter_history`(n_executed_updates,n,2) best logits and `objective_history`(n_executed_updates,n). Converged members remain fixed in subsequent batch history. The source reflection, expansion, contractions, shrink, storage identities and stopping rules are explicit local code.

`reconstruct(data,batch_size=1024)` extracts all fixed support signal rows, fits independent voxel batches and places results in native axis order. It returns a dictionary of four float64(1,96,96,19) fields and concatenated objective/iteration/convergence diagnostics in mask Boolean-index order. The fields are intra_fraction v, intrinsic_diffusivity D, extra_transverse_diffusivity(1-v)D and extra_mean_diffusivity(1-2v/3)D. All are exactly zero outside observation_mask. It never guesses a smaller fitting mask or replaces failed results by source references. Defaults preserve the full10000-update limit.

## Independent evaluator

`evaluation/judge.py` imports only NumPy and the standard library. Its hidden truth/baseline/metrics paths and legitimate raw-mask path are resolved relative to its own file. `load_prediction(path)` validates exact keys, shapes, real float32/float64 and finiteness. `metrics(estimate,reference)` independently computes uncentered cosine NCC and dynamic-range NRMSE, rejects degenerate reference configurations and defines NCC0 for zero estimate norm.

`evaluate(path)` returns(report,exitcode). It selects ground_truth.npz when present, otherwise baseline_reference.npz; invalid existing truth is an evaluator error and never falls back. It loads every required reference field, validates fixed support and zero padding, evaluates all eight field metrics and requires every threshold. It returns0/pass,1/scientific fail or2/malformed prediction/evaluator error. Invalid inputs have empty measured metrics and descriptive errors. No solver import, submission execution, implicit alignment, dtype coercion of complex arrays, axis reshape or source fallback occurs. `main(argv=None)` implements required `--prediction` and `--report`, creates report parents and returns2 on report-write failure with stderr diagnostics.

The method reference is independently original, not simulated/histological truth. Source float32 storage is preserved in hidden references; local float64 fits are fresh measurements. Numerical thresholds were fixed before local extraction and are documented in validation. No generated observations replace the bound real acquisition. The maintainer-only sensitivity helper is shipped as src/generate_data.py and called actively by the notebook.

## Tests and review flow

Frozen original/independent cases under test/fixtures are consumed by test/comparing_test/test_original_stages.py. Tests check theoretical orientation integration, source initial objective, all first-ten updates and independent source64 fits. Strict Judge tests cover analytic known metrics, every quantity, malformed NPZ variants, missing evaluator assets, configuration failures, foreign cwd and report-write errors. The full native pipeline is tested on all44427 supported voxels; final execution and audit records are indexed in validation.

The active six-chapter notebook loads the complete raw data and invokes reconstruct, computes original-versus-fresh comparisons for all four fields, displays all19 native slices, computes full powder/directional residuals separately, reviews optimization and boundary diagnostics, and fits the isotropic null, shell subsets and added-noise cases freshly. It never imports the original domain package.

`fit_isotropic_signals(observed,bvalues,absolute_precision=1e-12)` is a review-only one-dimensional isotropic-null fit returning physical diffusivity and objective; it is not the standard entry point or scored output. The loop limit100 is an internal constant, not a callable argument.

`visualization.draw_slice(ax,field,slice_index,mask=None,title='',vmin=None,vmax=None,cmap='viridis',norm=None)` displays a native x/y slice without resampling. It returns the image artist; optional norm supplies an explicit Matplotlib normalization. Relative powder residuals use LogNorm over their complete observed positive range; signed residuals use symmetric zero-neutral scales. `export_figure(fig,name,sources,metadata,figure_dir=None)` optionally writes PNG/SVG/PDF, numeric-source NPZ and hash/provenance JSON to an explicit private maintainer directory. It does not change the Matplotlib backend.

Source/reference arrays are used only as labeled comparison inputs. The notebook writes fresh standard reconstruction and evaluates it with the independent Judge. Its final native nbconvert exit code, canonical notebook hash and all-figure inspection are recorded in test/validation.md.

`generate_data.add_gaussian_noise(observed,bvalues,relative_sigma,seed=20261022)` returns a fresh float64(n,acquisitions) matrix, independently perturbs all positive-b intensities with Gaussian noise of SD relative_sigma times the observed voxel b0, keeps b0 fixed and does not clip or mutate inputs. Standard CLI never calls it; notebook1%/3% sensitivity cases do.
