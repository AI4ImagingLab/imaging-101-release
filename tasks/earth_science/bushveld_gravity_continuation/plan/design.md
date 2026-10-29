# Shipped interfaces and execution

The task is standalone and uses only NumPy, SciPy and Matplotlib. No runtime, test, notebook or evaluator imports Harmonica, Verde, scikit-learn, projection libraries or another task. Projection, data extraction and upstream capture live in a maintainer archive outside this task. The shared environment supplies test and notebook development tooling, which is not listed as runtime requirements.

## Callable contracts

- `src.preprocessing.load_observations(path)` loads NPZ with pickle disabled, validates exact bound dimensions and finite values, and returns independent arrays (the bound archive is float64) `(stations, data, grid)`: stations `(306,3)` in m with columns easting/northing/upward; data `(306,)` in mGal; grid `(48,64,3)` in m. The only file read is the requested observation archive. No normalization or metadata parsing occurs.
- `src.physics_model.source_coordinates(stations, depth=6000.0)` returns a float64 copy with upward coordinate reduced by depth in m, preserving shape `(N,3)`. It has no file side effects.
- `src.physics_model.greens_matrix(coordinates, sources)` flattens the leading dimensions of float64 coordinate arrays internally and returns `(number_of_points, number_of_sources)` inverse-distance values in 1/m. Source/evaluation coincidences raise ValueError. This internal flattening is part of operator construction; Judge never reshapes submitted arrays.
- `src.physics_model.forward(coordinates, sources, coefficients)` evaluates the Green matrix times `(N,)` coefficients in mGal m, returning the original coordinate leading dimensions in mGal. It does not save arrays.
- `src.solvers.solve_coefficients(kernel, data, damping=0.1)` accepts kernel `(M,N)` in 1/m and data `(M,)` in mGal. It computes population column scales, forms explicit scaled normal equations, and returns `(coefficients, scale)`, shapes `(N,)` in mGal m and `(N,)` in 1/m. Nonpositive damping or zero-variance columns raise ValueError.
- `src.solvers.reconstruct(stations, data, grid, depth=6000.0, damping=0.1)` returns `(image, coefficients, station_prediction)`, respectively `(48,64)` mGal, `(306,)` mGal m, `(306,)` mGal. Inputs are not modified and no files are read or written.
- `src.visualization.plot_comparison(stations, data, grid, fresh, reference, station_prediction)` returns a four-panel Matplotlib Figure containing actual station disturbance, the fresh continued field, identified upstream field and observed-versus-fitted scatter. Plot coordinates are converted to km. It neither selects the Matplotlib backend nor loads files.
- `src.main.main()` resolves task paths relative to its own module, loads `data/raw_data.npz`, calls reconstruct with defaults, creates `output/` and writes `output/reconstruction.npz` with only batch-first float64 reconstruction. It does not import plotting modules or select a backend. Standard invocation is `python -m src.main` from the task directory.

## Standard flow

CLI -> public NPZ loader -> station-aligned source positions -> explicit Green matrix -> column-scaled damped linear solve -> target-grid forward evaluation -> declared reconstruction NPZ. The CLI does not read hidden arrays. Intermediate coefficients exist only in memory. Scalar geometry constants match the README acquisition table; model depth and damping remain in source and approach plan, outside acquisition metadata.

## Notebook flow

`notebooks/bushveld_gravity_continuation.ipynb` determines task root, loads public observations, actively calls local reconstruct, saves a fresh standard submission, invokes standalone Judge, computes station residuals and compares fresh and archived upstream values. It also actively reconstructs a separately labeled weak-damping comparison. It creates physical and error figures from computed values, prints measured runtime and metrics, and stores an identified fresh result plus numeric measurements in `notebooks/reference_outputs/cleaned.npz` and `cleaned_measurements.json`. Archived `upstream.npz` is used only as comparison. There is no cache-preview default and no substituted fresh result.

## Standalone evaluator

`evaluation/judge.py` imports only standard library and NumPy. `validate(path)` enforces required key, exact `(1,48,64)` shape, float32/64 and finite values. `score(estimate,reference)` implements independent NCC and range-normalized NRMSE. `evaluate(prediction,asset_dir=None)` returns `(report,exit_code)`; the optional asset_dir is used in evaluator failure tests. Default hidden paths are relative to the Judge file. Actual ground truth takes precedence if supplied; otherwise the explicit baseline path is used. Neither file guessing nor submitted solver imports occurs. CLI paths resolve relative to the caller. Reports contain status, passed, metrics, thresholds and errors, use strict finite JSON and create parent directories. Codes are 0 pass, 1 valid below threshold, 2 malformed/evaluator/report-write error.

## Assets and consumers

`test/fixtures/upstream_stages.npz` is consumed by stage parity tests. `notebooks/reference_outputs/upstream.npz` is consumed by notebook comparisons and Judge subprocess tests; `cleaned.npz` and `cleaned_measurements.json` are identified fresh notebook review evidence. `evaluation/baseline_reference.npz` is consumed only by Judge and its contract tests. There is no `ground_truth.npz`, because independent truth does not exist. `evaluation/metrics.json` is the evaluator boundary contract. All tests are under `test/`; archive source, PDF, captures and complete logs remain outside the package.

## Exact helper and evaluator boundary details

`load_observations(path)` reads the first batch entry and copies arrays; it checks
unbatched shape/finiteness but does not cast arbitrary archive dtypes to float64.
The documented float64 contract is satisfied by the bound archive, not enforced
by a generic dtype validator in this loader. Missing file/key errors propagate.
All three outputs are independent of the closed NPZ and use x/easting,
y/northing,z/upward order. No coordinate transform runs during task execution.

For `source_coordinates(stations,depth=6000.0)`, stations may have general `(N,3)`
shape and are copied/cast float64 before subtracting depth from column2. Depth
is metres; a positive-down convention applies only to this scalar. No positivity
check is implemented. `greens_matrix(coordinates,sources)` casts both point
sets to float64 and flattens leading coordinate axes, requiring last axis3;
its dense `(M,N)` return has no batch dimension. `forward(coordinates,sources,
coefficients)` takes an `(N,)` coefficient vector and reshapes the matrix product
to `coordinates.shape[:-1]`; it never silently reshapes a submitted output.
These helpers allocate arrays and have no file side effects or hidden data.

`solve_coefficients(kernel,data,damping=DAMPING)` uses the module constant
DAMPING=0.1. It returns coefficients and scale, not the scaled intermediate m,
Hessian, covariance or residual history. `reconstruct(stations,data,grid,
depth=DEPTH,damping=DAMPING)` uses DEPTH=6000.0 and DAMPING=0.1; all positional
and optional arguments retain their physical units above. For general valid
shapes its returns are grid-leading-shape image, `(N,)` coefficients and `(N,)`
station fit; the bound shapes are48x64/306/306. Inputs are not mutated. Dense
memory is O(N^2+M*N); no undocumented blocks or iterative approximation is used.

`plot_comparison(stations,data,grid,fresh,reference,station_prediction)` accepts
unbatched arrays `(306,3)`, `(306,)`, `(48,64,3)`, two `(48,64)` fields and
`(306,)` fitted data. It returns a Figure and no numeric metric or filename.
The notebook itself creates the additional error and weak-damping figures;
these are separate executed comparison cells rather than hidden CLI branches.
`main()` has no task flags or Python parameters, writes a batch dimension around
only the standard image, and leaves coefficients in memory.

Evaluator `validate(path)` returns the exact reconstruction array `(1,48,64)`,
float32/64, after independent shape/dtype/finite validation; errors are raised
for missing files/keys and malformed arrays. `score(estimate,reference)` returns
a two-float dictionary, explicitly casting flattened inputs float64 for metrics.
It gives zero prediction NCC0 and rejects zero/constant references. It assumes
validated arrays and has no side effects. `evaluate(prediction,asset_dir=None)`
accepts path-like prediction and optional hidden-assets directory (default its
own directory), loads metrics and explicit truth-before-baseline reference,
and returns `(report,exit_code)`. Reference/config/input exceptions produce
status error; the function does not write a report. `main()` parses required
flags, writes strict JSON with allow_nan=False and returns2 on report-write
failure. No blanket overflow-rejection claim beyond these actual checks is made.

## Parameter/result ownership and verification map

No source or solver function changes its input arrays. The only persistent
standard side effect is CLI creation of output/reconstruction.npz; notebook also
writes labeled review measurements and report. Tests use frozen upstream source
coordinates/kernel/coefficients/station/target predictions, independent inverse
distance, harmonicity, coordinate scaling, transpose and augmented-system cases.
Unit contracts additionally exercise exact output shape/dtype, zero/constant
reference handling, malformed submissions, numerical boundaries, asset failure,
foreign cwd and report failure. Physical uncertainty, unique rock density and
actual image truth remain outside these guarantees. The notebook's weak solve
is freshly computed and uses the same public observations, never hidden truth.
