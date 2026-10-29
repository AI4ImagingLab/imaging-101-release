# Design: Shipped Interfaces and Execution Flow

## 1. Package boundary and actual runtime

The standalone task is `tasks/in_vivo_ultrasound_microvascular_mapping`. Runtime imports are Python standard library, NumPy, SciPy, and Matplotlib only. No upstream checkout, PALA MATLAB code, Octave process, imaging library, downloaded model, cross-task import, training checkpoint, or private capture script is required by the cleaned solver, tests, Judge, or notebook. The private original-source verification process is outside this task and supplies frozen numerical fixtures only.

Solver-visible input is the README, `data/raw_data.npz`, and environment information. `src/`, plans, notebook, tests, and evaluation assets are maintainer-only. There is no metadata JSON or public dependency on upstream source URLs. Instrument/acquisition scalars are written directly in the README; matching runtime constants appear in code and are checked by tests. Existing real observations are reused, so no synthetic generator is shipped.

The standard entry point is `python -m src.main` from the task root. It always reconstructs the entire 30-block acquisition with radial localization and writes one submission containing both required maps. The notebook separately reconstructs all three scientific comparison methods from the public IQ observations. It imports cleaned modules, loads archived upstream results only for comparison, and evaluates a freshly computed radial submission. It does not call the private upstream implementation or substitute a stored reconstruction for the active computation.

## 2. Files and ownership of state

| Path | Role and consumer |
| --- | --- |
| README.md | Public observation, acquisition, units, forward model, output contract, conceptual hints |
| requirements.txt | Pinned runtime subset of the project research environment |
| data/raw_data.npz | Public `iq` only; consumed by `load_observations` |
| src/preprocessing.py | Input validation, exact source boundary indexing, temporal Gram projection and causal filter |
| src/physics_model.py | Regional maxima, explicit radial weighted-line least squares, centroid/integer comparisons |
| src/solvers.py | Explicit Munkres partial assignment, track graph, endpoint-aware smoothing, interpolation, rendering, block accumulation |
| src/main.py | Complete radial acquisition reconstruction and submission write |
| src/visualization.py | Shared metric formulas and review figures; no solver or file I/O |
| output/reconstruction.npz | Generated submission, consumed by independent Judge |
| output/judge_report.json | Generated standalone evaluation report |
| evaluation/baseline_reference.npz | Hidden validated double-computation radial density and speed; not anatomical truth |
| evaluation/metrics.json | Four fixed metric boundaries, definitions, units, and provenance |
| evaluation/judge.py | Independent evaluator; imports no solver code |
| test/fixtures/upstream_block001.npz | Independent original-source double-computation first-block stage and rendering outputs |
| test/fixtures/upstream_math_cases.npz | Independent upstream assignment, localization, and smoothing numerical cases |
| test/unit_test/test_contracts_and_physics.py | Mathematical properties, metric cases, acquisition constants, malformed and poor submissions |
| test/comparing_test/test_upstream_and_pipeline.py | Original-stage parity and generated submission/Judge integration |
| test/validation.md | Source/data/license evidence, exact commands, gates, precision exception, hashes and archive index |
| notebooks/reference_outputs/upstream.npz | Original-source double-computation maps for all three methods, counts and timings |
| notebooks/reference_outputs/upstream_single_precision.npz | Identified native-single upstream run for numerical-precision comparisons |
| notebooks/in_vivo_ultrasound_microvascular_mapping.ipynb | Executed active six-section scientific review of full acquisition |
| plan/approach.md | Actual mathematical operations, conventions, parameters, and limitations |
| plan/design.md | This interface and execution description |

Only the CLI and notebook intentionally write submission/report paths. Solver mathematical routines have no file I/O. There is no persistent module-level mutable state. Calibration is not estimated from hidden maps. Each block has independent filter state and an independent tracking graph.

## 3. `preprocessing.py`

Constants: `SHAPE=(30,83,128,200)`, `FRAME_RATE=500.0` Hz, `BLOCK_RATE=2.0` Hz. Shapes use block, axial, lateral, frame ordering after removing the public batch dimension.

### `load_observations(path='data/raw_data.npz')`

Accepts a string or `pathlib.Path` interpreted relative to the caller's working directory. Opens NPZ with `allow_pickle=False`. Requires exactly the key `iq`, shape `(1,30,83,128,200)`, dtype complex64, and finite real and imaginary components. Missing or invalid ZIP/files/keys raise the corresponding standard I/O/NumPy exception or explicit `ValueError`; they are not replaced with test data. The result is an ndarray view of shape `(30,83,128,200)` and dtype complex64, in arbitrary acquisition amplitude units. The enclosing archive is closed before return; the array is materialized in memory. It is not a memory-mapped stream and the load therefore needs approximately 510 MB plus validation temporaries. No input array is changed.

### `original_boundary_zero(iq)`

Accepts one complex block `(H,W,T)`; the bound caller supplies `(83,128,200)` complex64. Promotes a copy to complex128 in Fortran order, reshapes its view to `(H,W*T)`, zeros collapsed columns 0 and -1 and axial rows 0 and -1, and returns the complex128 three-dimensional copy. It exactly preserves the two original MATLAB indexing statements rather than interpreting them as a conventional lateral border across all frames. The caller's array is never mutated. Non-three-dimensional inputs are outside the helper's internal precondition; public validation is performed by `load_observations`.

### `svd_filter(iq, cutoff=10)`

Accepts finite complex `(H,W,T)` input. Requires three dimensions and integer-like cutoff in `[1,T]`, otherwise raises `ValueError`. Reshapes to `(H*W,T)` in column-major order, forms `X.conj().T @ X`, calls general-purpose `scipy.linalg.svd(..., full_matrices=False, lapack_driver='gesvd')`, retains columns `cutoff-1:` and directly evaluates `(X @ Q) @ Q.conj().T`. Returns `(H,W,T)` with input computation precision and units preserved. It introduces no division by T, no mode-energy normalization, no cross-block basis, and no randomness. The standard path supplies complex128. Generic linear-algebra failures propagate rather than yielding a cached answer. The function has no I/O and does not mutate input.

The selected method is a full temporal Gram decomposition, not a randomized approximate SVD. A direct complement projection is used only in independent physical verification. LAPACK singular-vector phases do not affect the resulting projector; cross-backend subspace differences and degeneracy are explicitly checked.

### `temporal_filter(iq)`

Accepts one complex IQ block with time in its last axis. Designs a second-order Butterworth bandpass through `scipy.signal.butter(2,[25,249],btype='bandpass',fs=FRAME_RATE)`. Calls causal `signal.lfilter` along the last axis with zero initial state. Coefficients follow the input's real precision: float64 for the shipped complex128 path and float32 for an explicitly supplied complex64 diagnostic. Returns the same shape with the corresponding complex precision; units are unchanged. Replaces nonfinite result entries with zero as in the original safeguard. Coefficient design/filter exceptions propagate. No input mutation or I/O occurs.

### `preprocess_block(iq)`

Returns `temporal_filter(svd_filter(original_boundary_zero(iq)))`. This is the sole normal preprocessing composition. It accepts the complete raw block; it does not slice a few demonstration frames or accept a hidden clean counterpart. Each call restarts temporal state. No global state carries information across acquisition gaps.

## 4. `physics_model.py`

### `regional_maximum(image)`

Accepts a nonnegative two-dimensional magnitude array `(H,W)`. Returns a boolean array of the same shape identifying eight-connected regional maxima. Uses a general maximum filter for candidate discovery and explicit connected equal-value plateau traversal to reject plateau interiors connected to a brighter boundary. Equal neighboring positive values trigger plateau processing; zeros are excluded when any positive magnitude exists, which is equivalent for all downstream positive-threshold detection. An entirely constant nonnegative image is a regional-maximum plateau. This function does not perform peak interpolation or choose the strongest 200 values. It has no side effects. Input validity/nonnegativity is an internal mathematical precondition, not a separately advertised robust public parser.

### `radial_centers(patches)`

Accepts magnitude patches `(N,H,W)`, normally `(N,5,5)`, float64 in the selected runtime. Returns `(N,2)` floating offsets `[z,x]` in acquisition pixels relative to each patch center. Evaluates diagonal derivatives, zero-extended 3-by-3 averaging of those derivatives, gradient slopes, gradient-energy centroid and inverse-distance weights, and the explicit two-variable weighted line-intersection solution. The leading N dimension is independent: no convolution or smoothing crosses patches.

It retains the original NaN-slope unsmoothed fallback and infinite-slope handling, including the documented sign discrepancy in the rare fallback. If all smoothed slopes are infinite, it applies the original all-patch unsmoothed fallback. Floating divide/invalid warnings are locally suppressed because nonfinite position results are intentionally rejected by `localize`. The function does not replace an undefined center with zero. Degenerate normal equations may return nonfinite offsets. No claim of universal localization validity is made for those patches. It does not change its inputs.

### `localize(magnitude, method='radial', particles=200, max_local_peaks=4)`

Accepts a complete filtered block `(H,W,T)`, either complex or real; internally takes magnitude. `method` must be `radial`, `wa`, or `integer`; otherwise raises `ValueError`. The normal data path has `(83,128,200)` complex128 and the resulting magnitude is float64. The helper assumes H and W admit the three-pixel margin and `particles+1 <= H*W`; these are fixed benchmark preconditions rather than generalized shape support.

Returns `(N,4)` with rows `[candidate magnitude, z, x, frame]`. Dtype follows the real magnitude precision, so it is float64 for the selected computation. Intensity is in acquisition amplitude units; z,x are zero-based acquisition pixels; frame is a one-based integer value stored in the same floating table. Empty detection results have shape `(0,4)`.

Actual stages: form the three-pixel-margin reduced magnitude; concatenate z/time with MATLAB order; detect eight-connected maxima; retain values strictly above the 201st masked magnitude independently per frame; enumerate coordinates in Fortran order; gather five-by-five ROIs; reject more than four regional maxima; compute the selected offset; reject absolute offsets above 1.5 or undefined positions; subtract one from the original one-based coordinates. ROI plateau cases use the explicit regional-maximum helper rather than counting every local-filter plateau pixel as a separate peak.

Weighted centroid uses centered ROI coordinate weights and magnitude sum. Integer mode uses zero subpixel offset. The original optional sigma estimate has no active rejection and is intentionally not computed. The function never uses reference maps, hidden truth, or tracks to decide candidate acceptance. It allocates patch arrays but does not mutate input.

## 5. `solvers.py`: assignment and trajectories

Constants: `METHODS=('radial','wa','integer')`; `OUTPUT_SHAPE=(831,1281)`.

### `munkres(cost)`

Input is a two-dimensional rectangular cost array `(Ns,Nt)` with finite valid squared distances and infinity for forbidden edges. Return is integer `(Ns,)`: zero-based target column or -1 for unmatched source. Empty source or entirely invalid graphs return the corresponding empty/unmatched vector.

This is a local extraction of Yi Cao's source algorithm rather than a dependency on a ready-made tracking solver. It removes entirely invalid rows/columns; replaces forbidden entries with the original big-M value `10**(ceil(log10(sum(valid_cost)))+1)`; pads to square with `10*max(valid_cost)`; obtains row and column potentials; stars zeros in column-major order; primes uncovered zeros and augments alternating paths; adjusts potentials by the minimum uncovered reduced cost; maps matches back and removes forbidden links. For all-zero valid costs, the source's big-M expression is zero and is preserved. Column-major tie order is explicit, important for integer-position tracks.

The procedure terminates on the finite padded matching problem. It is not an iterative imaging objective with a user-selected tolerance or approximate stopping rule. Standard array/numeric errors propagate for inputs outside the stated internal cost contract. The supplied array is converted and not modified in place.

### `link_frames(source, target, max_distance=1.0)`

Inputs `(Ns,2)` and `(Nt,2)` are acquisition-pixel positions. Converts coordinates to float64, explicitly constructs squared Euclidean costs, marks values strictly greater than `max_distance**2` as infinity, and calls `munkres`. Returns target indices as above. A distance exactly one pixel is permitted at the default. Empty source/target cases are supported and tested. No gap inference, appearance descriptor, intensity weighting, learned association, or motion predictor is added.

### `track_localizations(localizations, n_frames=200, min_length=8, max_distance=1.0)`

Accepts an `(N,4)` localization table. Copies to float64, renormalizes the minimum recorded frame to 1 as upstream does, and groups detections in each frame in their original order. Builds pairwise adjacent-frame assignments and explicit predecessor/successor arrays. Traverses paths from nodes without a predecessor in original node order. Retains only lengths strictly greater than `min_length`.

Returns a list of float64 arrays `(L,3)` with `[z,x,frame]`, units pixel,pixel,frame identifier. Empty input returns an empty list. No gap closing is attempted, including when an interior frame has no detection. The function only handles a single acquisition block; `reconstruct` invokes it independently for each block. It has no persistent state, no mutation of caller tables, and no I/O. float64 is retained through trajectory smoothing; quantizing to float32 before interpolation would differ from the selected double-computation upstream execution.

### `smooth_track(x, span=20)`

Accepts a one-dimensional finite coordinate vector of length n>=1 and a positive span. Returns a new float64 vector of length n. Reduces span to min(span,n), then to the nearest odd integer no larger than that value. At zero-based position i, half-window is the minimum of `(span-1)//2`, i, and n-1-i. Averages the available centered odd window without extrapolation, zero padding, or periodic boundaries. The helper assumes a valid nonempty trajectory and positive span; the normal caller only supplies accepted paths with n>8. It is tested against independent upstream output and analytic constant, ramp, quadratic, and short-window cases.

### `interpolate_tracks(tracks, frame_rate=500.0, max_distance=1.0, resolution=10)`

Input is a list of float64 `(L,3)` raw tracks. For each, smooth z and x separately; form query positions `1+k*.8/(max_distance*resolution)` not exceeding n; interpolate linearly; compute the smoothed interpolated path length divided by n/frame_rate; repeat that scalar speed alongside each resampled position. Return a list of float32 `(M,3)` arrays `[z,x,speed]`, units acquisition pixel,pixel,pixel/s. Casting occurs after all continuous calculations, reproducing PALA's final track quantization. Query generation follows MATLAB colon endpoint conventions; the final query need not be n. Input frames are not interpolated through acquisition gaps because tracks come from one block. No input is mutated and no I/O occurs.

### `render_tracks(tracks, shape=OUTPUT_SHAPE, resolution=10, with_speed=True)`

Input list contains float32 tracks, either `[z,x,speed]` or raw `[z,x,frame]` with `with_speed=False`. Maps coordinates using `(position+1)*resolution` in float32, rounds half away from zero, converts to zero-based indices, removes outside-grid points, and forms the unique column-major pixel set per track. Density is incremented once per distinct track/pixel pair. If `with_speed`, adds the track's constant scalar speed once per pixel. Returns two float64 arrays `(831,1281)` at defaults: density and crossing-weighted mean speed, with empty-pixel speed zero. The no-speed branch returns a zero speed map. It does not count multiple interpolation samples as multiple vessel crossings. No normalization, alignment, blur, smoothing, or hidden mask is applied to the output arrays.

## 6. Block and acquisition orchestration

### `reconstruct_block(iq, methods=METHODS)`

Accepts one full complex64 observed block `(83,128,200)` and requested method names. Preprocesses once, then independently localizes and tracks each method. Smooths/interpolates from float64 raw tracks, renders float32 interpolated tracks, and separately casts raw tracks to float32 for noninterpolated counting. Returns a dictionary keyed by method; each value has:

| Field | Shape/dtype | Meaning |
| --- | --- | --- |
| density | (831,1281), float64 | Unique retained trajectory crossings |
| speed | (831,1281), float64 | Mean path speed, acquisition pixel/s |
| nointerp | (831,1281), float64 | Raw-observation-position unique trajectory crossings |
| counts | (2,), integer | Accepted localizations, retained tracks |
| track_length_histogram | (201,), int64 | Retained trajectory counts indexed by number of observed positions |
| method_runtime_seconds | scalar, float | Measured localization-through-rendering seconds; shared preprocessing excluded |

No intermediate detection or trajectory archive is written by this helper. Errors in an unsupported method propagate from `localize`. All requested methods use the same complete observations rather than different favorable subsets.

### `reconstruct(iq, methods=METHODS, progress=None)`

Input is iterable/block-first observed IQ, normally `(30,83,128,200)` complex64 from the public loader. Allocates running float64 density, speed-sum, and nointerpolated maps per method. Calls `reconstruct_block` once per block. Adds density and noninterpolated counts and adds `block_density*block_speed` to speed sums. At the end divides accumulated speed sums by total density where positive and returns method dictionaries with `density`, `speed`, `nointerp`, and `counts` shaped `(number_of_blocks,2)`, `track_length_histogram` shaped `(201,)`, and `saturation` shaped `(number_of_blocks,)` containing cumulative nonzero-density superpixel counts, plus `method_runtime_seconds` shaped `(number_of_blocks,)` with actual localization-through-rendering elapsed seconds. Wall-clock timings are diagnostic measurements, not inputs to reconstruction.

Optional `progress(done,total)` is invoked after each complete block; its return value is ignored and exceptions propagate. It is used only for CLI/notebook status. The helper permits smaller inputs for independent numerical tests, while the standard loader/CLI and review notebook bind exactly the complete 30-block experiment. It never bridges block trajectories or reuses temporal filter states. It does not depend on a GPU or automatic differentiation.

## 7. Entry point, plots, metrics, and independent Judge

### `src.main.main()`

No arguments. Calls the public loader on its default relative path, runs all 30 blocks with radial localization, creates `output/`, writes compressed `reconstruction.npz` with batch-first float64 keys `reconstruction` and `speed`, and prints block progress plus JSON containing actual elapsed seconds and localization/track totals. Returns `None`. Failures propagate to a nonzero process exit instead of silently loading an archived result. The module has no import-time reconstruction, filesystem writes, or Matplotlib backend change. Importing it in a notebook is safe.

### `visualization.metrics(estimate, reference)`

Accepts same-shaped real finite arrays, converts to float64 for calculation, and returns Python floats `ncc` and `nrmse`. NCC is dot product of flattened arrays divided by the product of their L2 norms, with no mean subtraction. NRMSE is RMS error divided by reference max-minus-min. Shape mismatch, nonfinite values, zero norm, or constant reference raises `ValueError`. No crop, alignment, fitted gain, or rescaling is performed. No file I/O or plotting occurs in this metric function.

### `plot_acquisition(iq)`

Accepts full block-first complex IQ. Returns a Matplotlib Figure with actual block 1/15/30, frame100 magnitude images, relative dB color scales, and native-pixel axes. It does not save files, display hidden truth, or synthesize a missing frame. The caller displays the returned figure.

### `plot_comparisons(fresh, upstream)`

`fresh` is the `reconstruct` result; `upstream` supplies arrays `density` and `speed` shaped `(831,1281,3)` in the declared method order. Returns two Figures, one per output quantity. Each has all three methods in rows and upstream/fresh/difference in columns. Density display uses square root and labels that transform; differences and evaluated arrays remain untransformed. Speed plots remain acquisition pixel/s. Returns figures rather than writing images.

### `plot_sampling(fresh, upstream)`

Returns a Figure comparing upstream and fresh full-acquisition localization counts and track totals for all three methods, plus their lateral profiles of axially summed density and the original/fresh per-block sampling-saturation curves. `upstream['counts']` is `(30,3,2)` and `upstream['saturation']` is `(30,3)`. This preserves the scientific consequence of localization choice, beyond merely plotting three visually similar maps. Counts and profiles are computed from actual arrays, never zero placeholders.

### `evaluation.judge.evaluate(prediction, asset_dir=None)`

The independent Judge imports only standard library and NumPy. Default asset directory is relative to `Path(__file__).resolve().parent`; optional `asset_dir` exists for evaluator-failure unit tests. It does not import submitted/reference solver modules or execute arbitrary submission code. CLI prediction/report paths remain relative to the caller's working directory.

Reads JSON boundaries and hidden truth if genuinely available; otherwise reads the declared baseline reference. Validates both required reference arrays against `(1,831,1281)`, float64, finite values, nonzero norm, and nonconstant range. Loads prediction with `allow_pickle=False`; requires both exact shapes and dtype float64, finite nonnegative values. No reshape, coercion, crop, magnitude, normalization, alignment, or substitute answer is allowed. Only after all arrays validate does it compute the four metrics independently of the plotting module. An all-zero estimate has NCC0, so it is a valid low-quality prediction rather than NaN JSON.

Returns `(report,exit_code)`. Report fields: `status` pass/fail/error, `passed` boolean, measured `metrics`, each `thresholds` comparison/boundary, and descriptive `errors`. Invalid inputs or missing/corrupt evaluation assets return empty metrics and code2. Valid below-threshold predictions return code1. Passing returns code0. Both outputs must pass NCC and NRMSE boundaries; perfect density cannot compensate for absent or poor speed.

### `evaluation.judge.main(argv=None)`

Parses required `--prediction` and `--report`. Calls `evaluate`; creates report parent directories; writes UTF-8 JSON with `allow_nan=False`. A report-write failure prints a stderr diagnostic and returns2. No current-working-directory-dependent guessing of hidden reference files is used.

## 8. Review notebook and test flow

The notebook contains exactly six second-level sections: Problem, Data, Physics, Method, Results, Conclusion. Its setup locates the task root, imports cleaned modules, and loads the public observation. Data identifies real versus simulated provenance and exposes the actual array/table shape. Physics derives the measurement/subspace/response assumptions. Method explains the weighted-line inverse, constrained assignment, smoothing/endpoints, and actual speed/grid conventions. Results actively runs all three methods on all30blocks, computes the fresh metric table, writes/evaluates its radial submission, and displays fresh/upstream/difference, method counts/profiles, saturation curves, trajectory-length histograms and spatial spectra, interpolation effects, and native-single versus double precision comparisons. Conclusion distinguishes numerical reproduction from anatomical truth and states physical/calibration limitations.

No private MATLAB environment or original source is imported. Original results are read solely as identified references after fresh reconstruction. Runtime is measured with `time.perf_counter`; execution timeout accommodates the complete acquisition. Stored reference measurements are not overwritten by notebook execution. The temporary nbconvert output remains private or is removed after its executed cells/outputs are promoted to the primary review notebook.

Unit tests check source boundary indexing, acquisition/README constants, independent projector identities, regional-maximum plateaus, analytic radial/reflection/scaling behavior, moving-average endpoints, partial assignment and ties, no-gap/strict-track-length rules, finite-track speed, unique crossing weights, known metric cases, malformed predictions, poor speed with otherwise perfect density, and evaluator errors. Parity tests consume independent original output fixtures, compare temporal stages, positions, raw tracks, and rendered maps for allthree methods. The final full-pipeline test evaluates the fresh CLI output and must not skip in the completed handoff. Test tooling and Jupyter belong to the shared development environment, not task requirements.

## 9. Material limitations and intentional exclusions

Source numerical precision is explicitly stabilized, with the original single run preserved and compared. Rare fallback-sign behavior is investigated rather than silently corrected or advertised as universally valid. The package does not provide absolute mm/s calibration, a measured PSF, complete vascular coverage, biological truth, training, acoustic wave simulation, domain-library wrappers, or a second task's helpers. These exclusions describe the actual released acquisition and method; they do not reduce the complete bound frame count or substitute an easier inverse problem.
