# Shipped design and execution

The cleaned runtime requires NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1. Python3.11.11 CPU execution was validated. pytest and Jupyter are development tools, not task runtime dependencies. Original R/Bioconductor/Rcpp packages exist only in the private maintainer oracle; no cleaned module imports them. The software is GPL-3.0 with preserved LICENSE and NOTICE.

## Data and standard path

`data/raw_data.npz` contains seven batch-first arrays described completely in README: spatial_counts int32(1,25753,428), single_cell_counts int32(1,19736,1926), cell_type_index and cell_sample_index int16(1,1926), spatial_coordinates float64(1,428,2), and two Unicode gene-ID arrays. Count units are reads/transcripts as supplied; coordinate units remain native grid units. Input data are real acquisitions. No target weights, fraction fields, reference-derived seed or fitted calibration is public.

`python -m src.main` resolves paths from its own file location, calls `prepare_data`, evaluates all seven original fixed-phi fits, applies the qualified complete-score selector, and writes exactly mixing_weights and cell_type_proportions to output/reconstruction.npz. Output arrays are float64(1,428,20). It prints runtime and creates output/ if needed. It neither imports visualization nor changes a notebook backend. No root entry-point wrapper exists.

## Module signatures and returns

- `preprocessing.reference_basis(counts, types, samples, n_types=20)`: counts is genes-by-reference-cells float-compatible counts; type/sample indices are one-dimensional integer labels. Returns `(basis, active_gene_mask)`. Basis is nonzero-reference-genes-by-types float64 and retains the author's cell-size normalization.
- `preprocessing.informative_genes(basis, sc_counts, types, common)`: matched nonzero-gene matrices, type labels and common-gene boolean mask; returns selected reference-gene row indices in native order, using exact fold-change and dispersion criteria.
- `preprocessing.prepare_data(path)`: reads NPZ with allow_pickle=False, removes only input-defined QC exclusions, derives the independent basis, selects and matches genes, normalizes observations/basis/coordinates and builds the Gaussian kernel. Returns a dictionary: X(7050,428), basis(7050,20), kernel(428,428), coordinates(428,2), genes(7050), selected_counts(7050,428), mean_X/mean_B scalar normalization factors, and reference_basis(7050,20) before mean scaling. Numeric computation arrays are float64; identifiers retain Unicode. The bound acquisition retains all428 spots. No output file is written.
- `physics_model.forward_model(basis, weights)`: matrix multiplication B V^T, returning genes-by-spots expression in source-scaled dimensionless units.
- `physics_model.car_precision(kernel, phi)`: returns D-phi W and rejects phi outside[0,1). This includes the boundary and degree conventions explicitly.
- `physics_model.log_objective(X, basis, V, b, variance, lambdas, phi, kernel)`: returns the direct-residual conditional scalar score, without the fixed-phi determinant constant. This independent path checks the solver's trace-expanded objective.
- `physics_model.car_log_normalizer(kernel, phi, n_types)`: returns K/2 logdet(L), rejecting a nonpositive determinant. This is the explicit supplementary-Eq9 correction needed for cross-phi comparison.
- `physics_model.normalize_weights(V)`: returns V divided by its row sums and rejects nonpositive row totals.
- `solvers.fit_card(X, basis, kernel, phi=.99, max_iter=1000, epsilon=1e-4, initial=None)`: performs explicit sequential multiplicative updates. `None` initializes V uniformly1/K. A supplied initial array is copied. Returns V, b, lambda, sigma_e2, Obj (the original conditional score), history(iteration,conditional score,RMS weight change,variance), and phi. Fixed-phi arithmetic, stored-quadratic order and stopping match original numeric fixtures.
- `solvers.reconstruct(prepared, phis=PHIS)`: calls `fit_card` for each declared PHIS value, adds the determinant correction only to comparison scores, and selects the last maximizer. Returns `(fields, results)`: fields has both batch-first output arrays, and results contains every fixed-phi result plus normalized_score. No reference array is loaded.
- `generate_data.thin_counts(counts, retention, seed)`: maintainer-only sensitivity helper, not the bound-data source. It validates nonnegative integer counts and retention in(0,1], draws independent RandomState-seeded binomial counts, and preserves shape/dtype/count units. It never clips intensities or consults fitted fields.
- `visualization.spatial_panel(ax, coordinates, values, title, vmin=0, vmax=None, cmap='viridis', size=28)`: plots native square-position markers and returns the scatter artist. No interpolation occurs; dense markers are rasterized within vector exports.
- `visualization.field_atlas(coordinates, fields, names, title)`: returns a4x5 figure containing every type, with individual nonnegative scales and visible units. Library modules never choose a Matplotlib backend.
- `main.main()`: the standard entry point described above; writes the declared submission and prints elapsed seconds.

## Independent evaluator

`evaluation/judge.py` is standalone and imports only standard libraries and NumPy. It never imports src or executes a prediction. CLI arguments are `--prediction` and `--report`. Caller-relative CLI paths remain caller-relative; hidden asset paths resolve relative to the Judge file.

`load_fields(path)` enforces exactly the two declared keys, shape(1,428,20), real float32/float64 dtype, finite values and nonnegativity. `field_metrics` computes NCC without mean subtraction, RMS error divided by each type's reference range, and maximum absolute error, independently for all20 columns. Zero prediction norm scores NCC0; a degenerate evaluator reference errors. `evaluate(prediction)` validates boundaries, loads actual ground_truth.npz if it exists and otherwise baseline_reference.npz, validates model calibration, evaluates all120 field metrics plus two consistency metrics, and returns `(report, code)`. An invalid present truth file cannot fall back to a baseline.

`notebooks/reference_outputs/public_preprocessing.npz` contains X and B captured from the original preprocessing. These are independently derived from public observations/reference cells, not estimated from unknown composition. The standalone Judge consumes these arrays for direct forward SSE; tests compare them with both native fixtures and fresh preprocessing. They provide no hidden physical information. The baseline_reference.npz contains only the two qualified original field arrays. No actual truth is available or invented.

Reports contain status, passed, metrics, thresholds and errors. Codes are0 pass,1 well-formed scientific fail,2 malformed input/evaluator/report-write error. Invalid inputs produce empty metrics. Reports use UTF-8 strict JSON and create their parent directory. Failed writes print a diagnostic to stderr.

## Tests and notebook flow

`test/fixtures/upstream_stages.npz` freezes native preprocessing, all seven original fixed-phi results, random-initialized native results and1/2/5/6/20-iteration source cases. It was captured before local implementation. `frozen_criteria.json` records original limits and the separately timed qualified selector correction. No test imports an upstream library.

Comparing tests verify each preprocessing stage, update trajectories, full fields, corrected selection and direct physics. Unit tests cover the independent analytic mixture/adjoint/CAR/objective cases, count-sensitivity helper, every Judge malformed input, bad controls, every type's scoring, truth precedence, evaluator errors and foreign-working-directory/report-write behavior. The pipeline actively runs the CLI and Judge. Private explicit pytest temporary directories avoid touching other tasks.

The canonical notebook has six active chapters: problem, data, physics, method, results and conclusion. It reads raw observations, reconstructs with local modules and writes/evaluates a fresh submission. All scientific comparisons are fresh unless specifically identified as reproduced original runs. The original and qualified selectors, same-basis null and NNLS fits, marker correlations including weak/negative groups, spatial-correlation and initialization sensitivities, count retention and identifiability are retained. Complete PNG/SVG/PDF figures and paired NPZ/JSON sources are exported to the maintainer archive; no private oracle is imported. Full native nbconvert execution, actual figure review, numerical identities, asset hashes and final file inventory are recorded in validation after execution.
