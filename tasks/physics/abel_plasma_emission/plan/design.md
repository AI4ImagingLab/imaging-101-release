# Shipped interfaces and execution flow

All paths below are relative to this task. Nothing imports another task or the
private oracle environment. The standard entry point is `python -m src.main`.
Its input and output path arguments are relative to the caller's current
directory. The notebook determines the task root explicitly. Solver libraries
never change the Matplotlib backend.

## Runtime modules

| Callable / constant | Input / default | Return and behavior |
|---|---|---|
| `preprocessing.DR`, `DZ`, `NOISE_STD`, `SHAPE` | 0.025 mm, 0.1 mm, 0.002 radiance, (1,65,257) | Explicit acquisition constants matching README |
| `preprocessing.load_projection(path)` | str or Path to NPZ | float64 (1,65,257) projection in radiance; validates key, shape, dtype and finiteness; no preprocessing changes |
| `physics_model.validate_half_image(image, dr, hold_order)` | real numeric vector (C,) or array (Z,C), C>=3; dr>0 in mm; order 0 or 1 | (float64 2-D array, was_vector boolean); rejects nonfinite and complex inputs |
| `physics_model.state_integrals(n, a)` | descending integer vector, a=0,1,2 as needed internally | float64 (len (n),9) signed interval integrals; no units |
| `physics_model.state_coefficients(cols, direction='inverse', hold_order=1)` | integer column count, 'forward'/'inverse', order 0/1 | tuple (Phi,B0,B1), each float64 (cols-2,9); no files |
| `physics_model.propagate(drive, coefficients)` | float64 (Z,C), coefficient tuple | float64 (Z,C), state sweep and endpoint copies; same units as drive; no input mutation |
| `physics_model.forward_transform(emission, dr=0.025, hold_order=1)` | (Z,C) or (C,) real emission in radiance/mm | same-shape float64 projection in radiance |
| `solvers.inverse_drive(image, dr=DR, hold_order=1)` | validated float64 (Z,C) radiance | same-shape derivative in radiance/mm; internal helper assumes validated inputs |
| `solvers.inverse_transform(projection, dr=DR, hold_order=1)` | (Z,C) or (C,) real radiance | same-shape float64 emission in radiance/mm; validates and does not clip |
| `solvers.reconstruct(projection, dr=DR, hold_order=1)` | float64 batch-first (1,65,257) radiance | float64 (1,65,257) emission; no files |
| `main.run(input_path='data/raw_data.npz', output_path='output/reconstruction.npz')` | filesystem paths | actively loads and reconstructs, creates output parent, writes NPZ key reconstruction; returns runtime_seconds and shape |
| `main.main()` | CLI `--input` and `--output` options | invokes run and prints runtime/shape; executable guard only |

Numerical transforms use core array arithmetic, finite differences and an
explicit nine-state recursion. There is no optimization, checkpoint, learned
weight, domain-library call, hidden reference access or cached solver result
in the runtime. `main` imports preprocessing/solvers only; it does not import
plotting modules. Source has no upstream path or package lookup.

## Generation and visualization

`generate_data.synthetic_fields(seed=1407)` returns three float64 arrays of
shape (1,65,257): physical emission truth (radiance/mm), noisy observation
(radiance), and analytic noiseless projection (radiance). Coordinates match
preprocessing constants exactly. `generate(destination, seed=1407)` writes
`data/raw_data.npz` and `evaluation/ground_truth.npz` below the supplied
destination, creating parents. It is a maintainer-only helper, never called by
reconstruction. CLI: `python -m src.generate_data <separate_destination>`.

`visualization.comparison_metrics(estimate, reference)` accepts matching real
arrays of any nonempty shape with nonconstant reference, and returns floats
`ncc` and `nrmse`. Its numerical formulas are tested against independent known
cases and Judge. Plot helpers return Matplotlib figures without saving files
or selecting a backend:

- `plot_comparison(projection, truth, upstream, fresh)`: six physical-axis
  images, shared density scale for truth/reconstructions, separate residual
  scales and colorbars. All arguments have shape (1,65,257).
- `plot_profiles(truth, upstream, fresh, zero_order)`: radial profiles at z=0,
  including both actively computed hold-order results.
- `plot_metric_bars(measurements)`: insertion-ordered mapping from method
  label to metric dict, producing NCC/NRMSE comparison panels.

## Standalone evaluation

`evaluation/judge.py` has no solver imports. `load_array(path)` enforces key
reconstruction, exact shape (1,65,257), float64, real finite values and safe NPZ
loading. `compute_metrics(estimate, reference)` implements the independent
metric calculation. `evaluate(prediction, assets=None)` returns (report, code),
where optional assets is an evaluator-test dependency injection; default assets
is the Judge's own directory. Actual truth takes precedence over optional
baseline_reference; unavailable or invalid reference/configuration is an error.
The shipped asset is actual simulation truth, not a standard-method result.

`main()` requires `--prediction` and `--report`; these paths are relative to
the caller. It creates the report parent, writes UTF-8 finite JSON, and exits
0/1/2 for pass/quality fail/error. Write failure prints a stderr diagnostic and
returns 2. The report has status, passed, metrics, thresholds and errors. It
never runs a submission. A valid all-zero output fails with measured metrics.

## Execution and review flow

1. `python -m src.main`: NPZ observations -> loader -> batch wrapper -> inverse
   derivative -> coefficient construction -> explicit state propagation -> NPZ.
2. `python evaluation/judge.py --prediction output/reconstruction.npz --report
   output/judge_report.json`: independent configuration/reference loading ->
   input validation -> both metrics -> boundaries -> JSON and exit status.
3. `python -m pytest test/ -v`: frozen upstream stage/pipeline parity, analytic
   physics, units, malformed output/evaluator failures, and metadata contracts.
4. `notebooks/abel_plasma_emission.ipynb`: active first- and zero-order inversions,
   actual truth and separately labeled upstream reference loading, Gaussian
   comparison, fresh Judge execution, metric table, paired metric panels,
   profiles, physical-image and residual plots. It records actual runtime and
   executes from any task/notebook working directory. It imports cleaned src,
   never the upstream library. No cache preview is used.

Fixture and reference consumers and exact acquisition hashes are indexed in
`test/validation.md`. Upstream checkout, paper, capture script, logs, temporary
plots and executed-notebook transfer file live in an external maintainer archive.


## Function-level failure and mutation semantics

The signatures and shapes in the module table describe supported caller inputs, not a universal argument-validation layer. The following distinctions matter when reproducing the actual execution rather than a proposed API.

| Callable | Explicit failures, assumptions and side effects |
|---|---|
| `load_projection(path: str | Path) -> np.ndarray` | Missing key,wrong exact shape/dtype or nonfinite values raise ValueError; missing/corrupt files propagate NumPy/I/O exceptions. Returns loaded array without normalization; no writes. |
| `validate_half_image(image, dr, hold_order)` | Explicitly rejects empty,complex,nonnumeric,nonfinite,wrong-dimensional/<3-column data; dr must be finite positive and order 0/1. Returns float64 2D copy and original-vector flag; does not mutate source. |
| `state_integrals(n, a)` | Internal helper assumes n>1 and supported a; it has no independent invalid-n/exponent validation. Produces signed dimensionless integrals, including analytic log limit for lambda0+a=0. No files. |
| `state_coefficients(cols, direction='inverse', hold_order=1)` | Explicitly rejects cols<3,unknown direction or unsupported order; natural type errors remain possible. Returns three newly allocated float64(cols-2,9) arrays. No cache or input mutation. |
| `propagate(drive, coefficients)` | Internal helper assumes2D float64(Z,C) drive and aligned coefficients; native unpack/broadcast/index errors expose mismatch. Allocates state/output; input and coefficients unchanged. Both endpoint copies are part of its result. |
| `forward_transform(emission, dr=0.025, hold_order=1)` | Calls common validation; returns vector iff input was vector,otherwise2D. Multiplies by the forward unit factor; no normalization,file access or hidden arrays. |
| `inverse_drive(image, dr=DR, hold_order=1)` | Internal validated2D input assumed; order 1 uses NumPy gradient,other values take forward-difference branch. Public inverse validates order before reaching it. No separate helper guard is claimed. |
| `inverse_transform(projection, dr=DR, hold_order=1)` | Common validation then local drive/coefficient/recurrence. Returns float64 same original dimensionality,with no clipping. |
| `reconstruct(projection, dr=DR, hold_order=1)` | Checks exact SHAPE/dtype explicitly; inverse_transform checks numeric finiteness,spacing and order. Assumes an ndarray exposing shape/dtype. Returns fresh batch-first array; no files. |
| `run(input_path='data/raw_data.npz', output_path='output/reconstruction.npz')` | Caller-relative paths; actively loads/computes,creates parents,overwrites the generated output NPZ; returns runtime_seconds (float) and shape (tuple). I/O/numerical failures propagate. |
| `main()` | Parses optional --input/--output,prints run's dict; argparse rejects invalid options. Executable guard only; import neither computes nor changes backend. |
| `synthetic_fields(seed=1407)` | Maintainer-only default_rng draw; returns float64truth/noisy/clean arrays. Invalid seed propagates NumPy errors; no output writes or modification of bound files. |
| `generate(destination, seed=1407)` | Maintainer-only destination creation and two NPZ writes; returns None. No automatic protection against overwriting a supplied destination,so CLI instructions require a separate destination. Never invoked in reconstruction. |
| `comparison_metrics(estimate, reference)` | Explicit shape/realness/finiteness/nonempty/nonconstant-reference checks; zero estimate gives NCC 0. Returns Python float dict without alignment; no files. |
| `plot_comparison(projection, truth, upstream, fresh)` | Assumes four (1,65,257) real arrays in their stated units; creates Figure 2×3,physical extents,colorbars,shared density scale and separate residual scales. No array mutation/save/backend change. |
| `plot_profiles(truth, upstream, fresh, zero_order)` | Assumes same fixed arrays; creates z=0 profile Figure,keeps upstream and both fresh hold orders. No I/O. |
| `plot_metric_bars(measurements)` | Assumes insertion-ordered mapping labels->dict with ncc,nrmse; creates two panels and numeric annotations. Missing metrics naturally raise KeyError; never fills missing methods with zeros. |

## Independent evaluator internals and errors

`load_array(path)` requires reconstruction key,exact (1,65,257),realfloat64 and finite entries. It raises ValueError for declared violations; malformed ZIP/I/O errors are caught by evaluate. `compute_metrics(estimate, reference)` checks matching nonempty real finite arrays. It rescales intermediate vectors and squared-error accumulation solely for numerical overflow stability,not to fit a gain or change the scientific metric. Zero estimate gets NCC 0; a zero/constant reference raises ValueError. Outputs are finite Python floats or an explicit error.

`evaluate(prediction, assets=None)` defaults assets to the source-relative evaluator directory. It rejects boolean/nonfinite/out-of-range NCC thresholds and nonpositive NRMSE thresholds; actual truth takes precedence over baseline. Returns (report,code) without writing; code 0 pass,1 valid quality failure,2 asset/prediction/metric/configuration error. Caught types include OSError,ValueError,KeyError,TypeError,FloatingPointError,OverflowError,BadZipFile andEOFError. `main()` parses required paths,creates report parents,writes finite UTF-8 JSON and returns 2 with stderr on write failure. Neither callable imports the solver or executes a prediction.

## Test-to-callable and notebook review map

| Test group | Specific mathematical/interface coverage |
|---|---|
| test_preprocessing_preserves_observations | Public loader versus frozen exact upstream input |
| test_intermediate_coefficients_and_derivative | Phi/B0/B1 and inverse_drive stage parity |
| test_bound_forward_inverse | Both forward/inverse directions and both hold orders on bound fields |
| test_signed_probe_and_one_dimensional_input | Arbitrary signed boundaries,vector/2D wrapper conventions,spacing 0.13 |
| test_gaussian_pair_absolute_scale_and_grid | Independent Gaussian projection/inverse sign,origin and unit normalization |
| test_generator_line_of_sight_quadrature_and_noise | Analytic generator versus SciPy integral and noise statistics |
| test_generator_forward_roundtrip_and_flux | Independent analytic source and cylindrical conservation |
| test_linearity_and_units | Direct linearity,dr scaling and endpoint copies |
| test_fresh_pipeline_matches_upstream | run path/write contract versus full independent original output |
| test_metrics_known_values and metric-error tests | Judge and plotting formulas on known vectors/degenerate cases |
| test_judge_cli/test_evaluator_error | Valid/poor/malformed arrays,asset/shape/dtype/threshold failures and report status/codes |
| test_truth_precedes_baseline/test_fallback_reference | Exact evaluator reference selection |
| test_report_write_failure and metadata/input tests | Error JSON/write stderr,source constants,data/generator contracts,backend-preserving imports |

The notebook's Data section actively loads the public projection and separately identifies hidden simulation truth/original references. Physics states the continuous acquisition before Method explains the recurrence. Method computes both hold orders using cleaned modules. Results keeps Gaussian/residual diagnostics,all four method-provenance rows,six images,radial profiles,paired bars and fresh standalone Judge; Conclusion limits the interpretation to the validated synthetic known-axis problem. Old cleaned_measurements/runtime/source hashes remain historical snapshots,not evidence of the latest document-only rerun. New review timing and source-docstring hashes are archived separately with preserved snapshot bytes.
