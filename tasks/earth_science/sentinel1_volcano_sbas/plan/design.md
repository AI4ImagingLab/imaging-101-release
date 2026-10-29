# Shipped Code Design and Interfaces

## 1. Runtime boundary, files and actual call flow

The solver-facing allowlist is the task README, `data/raw_data.npz` and environment information. Everything described in this plan is maintainer-only. The cleaned runtime imports Python standard-library modules, NumPy, SciPy and Matplotlib. It does not import MintPy, CVXOPT, HDF5 readers, scikit-image or a package from any neighbouring task. The original source checkout, raw source archive, private container/capture scripts, paper and original logs stay outside this task directory.

```text
python -m src.main
  main.main(argv=None)
    preprocessing.load_data(data_path)
    solvers.reconstruct(data, workers=4)
      physics_model.network_matrices(dates, pairs)
      solvers.repair_closure(phase, components, C)
        clean_components / component_labels
        physics_model.closure_integers
        solvers.lasso_ambiguities
        local-region matching and whole-cycle propagation
      solvers.invert_network(corrected, coherence, components, A, B, delta)
        per-row worker -> solvers.estimate_pixel at each original pixel
      preprocessing.reference_atmosphere
      solvers.deramp_timeseries
      physics_model.dem_design -> physics_model.temporal_model
      solvers.correct_dem
    main.save_submission(output, output_path)

python evaluation/judge.py --prediction ... --report ...
  independent file/schema/reference validation
  independent masked NCC / NRMSE for all five quantities
  independent original-phase observation-consistency check
  JSON report and exit code
```

The standard entry point reconstructs the entire weighted method. It does not compute the optional uniform-weight comparison or make figures. The notebook actively runs the same `reconstruct` call, saves and independently judges its new result, then actively calls `unweighted_common_baseline` for the scientific comparison. Plotting helpers receive computed arrays; none of them reconstructs an image or substitutes an archived output for a fresh one.

`src/generate_data.py` is deliberately absent: existing real observations are reused without adding synthetic observations. All runtime arrays are NPZ. Private HDF5 conversion only transfers original binary observations and attributes into an oracle-readable container; it does not enter the cleaned runtime.

## 2. Data structures, axes and acquisition constants

`preprocessing.EXPECTED` is the authoritative public-array schema before computation, excluding the leading batch axis:

| Dictionary key | Computation shape | Dtype | Units and semantic role |
|---|---|---|---|
| `phase` | `(288,450,600)` | float32 | Original unwrapped interferogram radians; edge, azimuth, range |
| `coherence` | `(288,450,600)` | float32 | Dimensionless original edge reliability |
| `components` | `(288,450,600)` | uint8 | Original region labels; zero means unavailable |
| `dates` | `(98,)` | int32 | Ordered YYYYMMDD calendar dates |
| `pairs` | `(288,2)` | int16 | Zero-based first/second date indices |
| `baseline` | `(98,)` | float32 | Per-date perpendicular baseline in m |
| `atmospheric_delay` | `(98,450,600)` | float32 | Independent original ERA5 LOS contribution in m |
| `incidence` | `(450,600)` | float32 | Original incidence in degrees |
| `latitude`, `longitude` | `(450,600)` | float64 | Original geolocation in degrees north/east |
| `analysis_mask` | `(450,600)` | bool | Original all-edge availability; fixed evaluation/comparison support |

The arrays in the NPZ have exactly one leading batch axis, so `phase` is stored as `(1,288,450,600)`. The returned data dictionary removes only this axis; it does not squeeze arbitrary singleton axes or transpose data. The source binary phase is the second band of a two-band BIL raster; this original-data interpretation is performed once privately and independently validated by exact bound sample comparisons.

Named constants are `SHAPE=(450,600)`, `REFERENCE=(76,156)`, `WAVELENGTH=0.05546576` m, `STARTING_RANGE=799547.2189154713` m, `RANGE_SPACING=34.94343172072985` m, `AZIMUTH_SPACING=78.08212603857966` m and `STEPS=(20170910,20180613)`. The matching README metadata values are tested. The two geometry defaults in `dem_design` deliberately match these exact original constants. Function defaults are not parsed from Markdown.

The five reconstruction quantities, without batch internally, are:

| Output dictionary key | Shape | Dtype | Units |
|---|---|---|---|
| `reconstruction` | `(98,450,600)` | float32 | Corrected LOS m |
| `network_timeseries` | `(98,450,600)` | float32 | Apparent network LOS m |
| `dem_error` | `(450,600)` | float32 | Elevation-residual coefficient m |
| `deformation_parameters` | `(5,450,600)` | float32 | Offset m, velocity m/year, acceleration m/year^2, two steps m |
| `temporal_coherence` | `(450,600)` | float32 | Dimensionless unweighted residual coherence |

The private diagnostics dictionary contains `corrected_phase`, `regions`, `num_observations`, `atmosphere`, `troposphere_corrected`, `deramped`, `spatial_ramp`, `residual`, `fit_mask` and `design`. These are actual returned intermediates used by the review notebook. They are not required solver submissions or public calibration. In particular, no full corrected phase or reconstruction is smuggled into the public NPZ.

## 3. preprocessing.py

### `load_data(path='data/raw_data.npz')`

`path` is a string or Path, resolved relative to the caller's current working directory. Open NPZ with `allow_pickle=False`; load each key listed in `EXPECTED`; require exact shape `(1,*expected_shape)`, exact declared dtype and finite values. Return a new dictionary of copied arrays with only the leading batch axis removed. The returned arrays own their computation state; caller mutation does not modify an open archive or an upstream source file. Unknown extra NPZ keys are ignored rather than loaded.

Missing files propagate the normal filesystem exception; missing expected keys raise KeyError; wrong shapes/dtypes or nonfinite arrays raise ValueError with the key and expectation. This function validates the structural contract. It does not claim to re-derive provenance or validate every physical value: dates must be valid ordered YYYYMMDD dates, pair indices must be in range, and original labels/coherence must have their documented physical meaning. These preconditions are established by the bound-data validation and downstream model operations.

### `reference_atmosphere(delay, reference=REFERENCE)`

Input: real LOS delay array `(N,H,W)` in m; default reference `(76,156)`. Copy as float32, subtract each date's reference-pixel value, then subtract the first resulting image. Return float32 `(N,H,W)` in m, zero at the reference pixel for every date and zero across the first date. The input is not mutated; an out-of-range reference raises the underlying IndexError. No temporal interpolation or atmospheric inversion is performed. The independent external product is used directly after gauge changes.

## 4. physics_model.py

### `network_matrices(dates, pairs)`

Input: ordered calendar dates `(N,)` and edge indices `(M,2)`, normally int32/int16. Date parsing uses Python datetime; malformed dates raise ValueError. Index errors propagate for invalid pairs. The helper assumes distinct, meaningful acquisition pairs; it does not silently sort, drop or repair the input graph.

Return a four-tuple:

| Return | Shape / dtype | Units and convention |
|---|---|---|
| `a` | `(M,N-1)`, float32 | Dimensionless incidence, first-date column removed |
| `b` | `(M,N-1)`, float32 | Years, integrated velocity interval per edge |
| `c` | `(T,M)`, float32 | Dimensionless chronological triangle incidence; `T=286` here |
| `intervals` | `(N-1,1)`, float32 | Consecutive day differences/365.25 in years |

Matrices are newly allocated. No files are read or written. Reversed temporal edges are supported by the B sign rule. Triangle enumeration requires the three chronological oriented edges; it does not invent reverse-edge observations. If no triangle exists, return a zero-row `(0,M)` matrix rather than a fabricated closure. The bound graph has 98 dates, rank97 and 286 triangles; tests check both source elementwise parity and independent `C@A=0`/velocity integration identities.

### `closure_integers(c, phase)`

`c` is `(T,M)` and `phase` is `(M,P)` radians. Return a floating numeric array `(T,P)` whose values are nearest integer closure cycles. The function computes the matrix product, wraps into `[-pi,pi)` and rounds. It preserves normal NumPy input arithmetic and has no side effects. The main repair path explicitly passes float64 C, preserving the source CVXOPT-matrix arithmetic for closure formation. Shape errors are normal matrix-multiplication errors. This helper does not label spatial regions or solve the inverse problem.

### `temporal_model(dates, steps=(20170910,20180613))`

Input dates `(N,)` use valid YYYYMMDD integers. Output float32 `(N,3+len(steps))`, normally `(98,5)`. Columns are `[1,tau,tau^2/2,strict_post_step_1,...]`. Tau is the float64 first-date difference of `year+(day_of_year-1)/365.25`, followed by float32 storage. This differs intentionally from B's float32 elapsed-day convention. Coefficients of these columns are respectively m, m/year, m/year^2 and m. All returned indicators use `date > event`, not `>=`. No input mutation/I/O; invalid dates propagate ValueError. Analytic tests verify the acceleration factor and the exact date-boundary behaviour.

### `dem_design(baseline, incidence, dates, starting_range=799547.2189154713, range_spacing=34.94343172072985)`

Input baseline `(N,)` m, incidence `(H,W)` degrees, ordered dates and exact range constants in m. Calculate every column's slant range, mean that range, mean the sine of degree-converted incidence over the complete scene, and form the dimensionless geometry column. Append `temporal_model(dates)`. Return `(N,6)` float64 for this task: stacking the float64 geometry column with the float32 temporal model promotes the combined array. The helper does not use latitude/longitude to change the acquisition geometry and does not choose pixelwise geometry. Positive finite range and nonzero mean incidence sine are physical preconditions, not separately checked custom exceptions. Tests compare all elements against independently captured G and check a known physical coefficient case.

## 5. solvers.py: spatial operations and LASSO

### `component_labels(values)`

Input `(H,W)` boolean/integer region image. Zero is background. For every distinct positive label, use four-neighbour `scipy.ndimage.label` on the equal-label mask, collect each connected region's flattened indices and order regions by first raster index. Return int32 labels `(H,W)`, zero background and positive consecutive region numbers. This preserves boundaries between touching unequal source labels. No input mutation/I/O. The caller supplies finite nonnegative integer/bool values; there is no additional dtype conversion/physical validation gate in this primitive.

### `clean_components(values, min_area=2500., erosion_size=5)`

Call `component_labels`. If more than one region exists, remove regions strictly smaller than `min(min_area,values.size*0.003)` and relabel. If `erosion_size>0`, apply the minimum filter of that square size with `mode='reflect'`, cast its result to uint8 as in the source, remove labels absent after erosion, and relabel. Return int32 `(H,W)` without changing the input. The source's label-count precondition is checked against the actual data, where cleaned region counts remain far below256. `erosion_size=0` disables erosion. Negative/meaningless filter sizes are outside the intended API and propagate SciPy errors; this function does not promise a generic input-sanitization layer.

### `lasso_ambiguities(a, y, alpha=.01, rho=1., tolerance=1e-7, max_iterations=20000)`

Input `a` `(T,M)` and `y` `(T,P)` in dimensionless cycle units, usually a=-C and P=100. A one-dimensional y is made a single column. Convert both to float64. The objective is exactly `sum((a@x-y)^2)+alpha*sum(abs(x))`; no sample/triangle normalization is added. Rho must be positive, alpha nonnegative and iteration/tolerance parameters meaningful; these are caller preconditions rather than explicit ValueError checks.

Compute and factor `2*a.T@a+rho*I` once with SciPy Cholesky. Initialize z and the scaled dual to zero. Perform the explicit x solve, soft threshold and dual update documented in `approach.md`. Every20 iterations evaluate KKT, primal and dual maximum residuals. Return `(z, diagnostics)` on convergence: z is float64 `(M,P)`, diagnostics contains integer `iterations` and float `kkt_residual`. Raise RuntimeError when the iteration ceiling is reached, rather than returning an unchecked solution. Factorization/matrix-shape errors propagate. No input mutation, RNG or file I/O. Independent source fixtures include original continuous solutions, every rounded sample solution and the region median; tests verify objective quality, rounding and KKT convergence.

### `repair_closure(phase, components, c, reference=REFERENCE, num_samples=100, seed=20261018, min_area=2500.)`

Inputs: phase `(M,H,W)` float32 rad; original components of the same shape uint8; triangle C `(T,M)`; reference pixel; algorithm sample count/seed/area. Create a common availability mask, clean regions, sample100 coordinates without replacement within each region, and determine median rounded LASSO ambiguities for non-reference common regions. The fixed seed controls the estimator's numerical sampling, not synthetic data or a hidden target. Regions must contain at least `num_samples` pixels; otherwise NumPy raises rather than silently sampling with replacement.

Return `(corrected, regions)`. Corrected is a new float32 phase stack `(M,H,W)` in radians after spatial reference subtraction and region correction; original phase/components are not mutated. Each region dictionary contains `coords` `(100,2)` integers, `ambiguity` `(M,)` float64 rounded-solution median, integer `area`, integer `iterations` and float `kkt_residual`. Regions are sorted by descending area, not current labels. Region corrections are propagated to an interferogram component only when it contains all captured coordinates of a matching common region; the first such region wins. Unmatched components receive zero correction. Original missing labels remain missing in subsequent inversion.

No file is read or written. LASSO convergence failures propagate. The intended acquisition has a valid reference point in the common region; the upstream data validation establishes this instead of inventing a new reference. Two-stage erosion cleanup preserves the actual source call flow. The private full-stage audit compares all77760000 corrected phase values with original outputs; normal tests consume frozen original sample/region fixtures and also run complete spatial repair.

## 6. solvers.py: temporal inversion and nuisance separation

### `estimate_pixel(a, b, phase, delta, weight_sqrt, rcond=1e-5, min_redundancy=1.)`

The main caller supplies A/B `(M,N-1)`, a single-pixel phase `(M,1)` in rad, intervals `(N-1,1)` years and square-root weights `(M,1)` dimensionless. NaN phase rows are unavailable; choose rows using the first observation column, matching the original routine. Allocate float32 `(N,1)` phase history. If any unknown acquisition has fewer than `min_redundancy` incident retained observations, return initialized zeros, quality0 and count0. Otherwise solve weighted B/y by SciPy `lstsq(...,cond=rcond)`, calculate unweighted circular residual quality and integrate phase velocity.

Return `(history, coherence, number)`: history float32 radians with zero first date; coherence a Python float; number an integer retained-row count. On a LAPACK LinAlgError, preserve the source fallback of a zero history/quality and retained-row count. Invalid shapes are not silently reshaped to a different acquisition graph; ordinary NumPy/SciPy errors propagate. Inputs are not mutated. The helper reshapes its array inputs internally as in the original source, but the shipped caller always uses one pixel; multi-column quality-map support is not claimed.

### `invert_network(corrected, coherence, components, a, b, delta, workers=4)`

Inputs: full `(M,H,W)` corrected phase rad, original coherence/labels, and temporal matrices. Return float32 apparent LOS movie `(N,H,W)` m, float32 temporal coherence `(H,W)` and int16 inversion count `(H,W)`. The nested `row_solver(row)` copies that row's phase/coherence, marks original zero-label phase entries NaN, floors coherence at0.05 and uses its float32 square root. It calls `estimate_pixel` independently for every range column, then applies the exact float32 LOS factor.

`ThreadPoolExecutor(workers)` distributes rows; results are assigned back by their original row index. The standard four-worker default does not reduce the frame, share phase histories between pixels or replace SVD by a normal-equation approximation. It performs no file I/O and does not mutate public arrays. Invalid worker counts raise the executor's ValueError. Numerical threading settings are controlled externally in recorded runs, not changed process-globally inside a library module.

### `unweighted_common_baseline(corrected, mask, b, delta)`

This notebook-only comparison takes corrected `(M,H,W)` phase, the fixed bool all-edge-valid mask `(H,W)`, B and intervals. Stack all valid-pixel right-hand sides and use a single unweighted B SVD with cond1e-5, integrate, convert to LOS. Return float32 `(N,H,W)` apparent LOS m and float32 `(H,W)` temporal coherence. Entries outside the supplied comparison mask are zero and must not be interpreted as estimates there. All comparison scoring/plotting explicitly uses the mask. It neither loads nor substitutes the original uniform baseline. Independent frozen upstream maps and quality validate this additional method.

### `deramp_timeseries(data, mask, reference=REFERENCE)`

Input LOS data `(N,H,W)` in m and bool/int spatial fit mask `(H,W)`. Internally view data as `(H*W,N)`, exclude nonfinite/exact-zero mean histories from the fit, create float32 row/column/one design, and estimate all date coefficients using NumPy `pinv(...,rcond=1e-15)`. Cast the plane to input dtype, anchor it to zero at the reference pixel and suppress its entries at original zero-valued inputs. Return `(corrected,ramp)`, both `(N,H,W)` with the input floating dtype and m units. Inputs are not overwritten despite local reshape views. There is no added ramp/curve normalization, hidden target or geolocation regridding. Empty/degenerate masks are not silently replaced by all pixels; matrix arithmetic determines the result. Tests exercise an independently known reference-anchored plane.

### `correct_dem(data, design, rcond=1e-8)`

Input deramped `(N,H,W)` LOS m and G `(N,6)` with dimensionless geometry column and five temporal basis columns. Solve unweighted multiple-right-hand-side `scipy.linalg.lstsq`. Return a four-tuple, all explicitly float32: corrected movie `(N,H,W)` m, elevation residual `(H,W)` m, five deformation coefficients `(5,H,W)` with the units above, and model residual `(N,H,W)` m. The corrected movie removes only `G[:,0]*height`; residual removes all six fitted terms. No observation-date exclusion or phase-velocity option is part of this shipped path. Shape/factorization errors propagate; no file I/O or input mutation.

### `reconstruct(data, workers=4)`

This is the complete in-memory orchestrator. The input dictionary is the validated unbatched `load_data` result. Call temporal construction, spatial closure repair, weighted network inversion, atmospheric gauge correction/subtraction, spatial-plane fit with actual coherence-derived mask, scene-mean geometry and joint DEM solve **in that order**. Return `(output, diagnostics)` with the dictionaries defined in section2. The code never reads hidden references or source fixtures. Worker and numerical exceptions propagate to the caller. It does not write output files or select a Matplotlib backend. Peak working memory includes the full observation stack, corrected phase and multiple temporal histories; this is a research-sized real-data calculation, not a per-pixel demo.

## 7. main.py and files it writes

### `save_submission(output, path='output/reconstruction.npz')`

Take the five unbatched output arrays, add a single leading batch axis and explicitly store float32 in one compressed NPZ. Create the parent directory if needed. Return None. The function assumes the supplied output dictionary follows the internal reconstruction schema; it does not silently infer shapes or fill missing arrays. Filesystem/serialization errors propagate. This is the only standard solver submission writer; no root-level wrapper is added.

### `main(argv=None)`

Parse `--data` (default `data/raw_data.npz`), `--output` (default `output/reconstruction.npz`) and integer `--workers` (default4). Load data, actively reconstruct, save the submission, print actual wall time and region count, and return0. Relative paths are relative to the caller's working directory. The module guard calls `SystemExit(main())`. Failures propagate as unsuccessful process execution; there is no hidden fallback reconstruction. No Matplotlib import/backend selection is performed by this entry module, so importing it in a notebook does not change that notebook's backend. CLI and import-backend behaviour are tested.

## 8. visualization.py and notebook-only interfaces

All plotting functions create and return a Matplotlib Figure; callers explicitly display/close it. They do not write screenshots or modify Matplotlib backends. Matplotlib remains an actual task dependency because the required active review notebook uses these helpers.

### `array_metrics(estimate, reference)`

Flatten real estimate/reference values as float64; require identical shapes and finite values. Reject zero reference norm or zero reference dynamic range with ValueError. Return `{'ncc':float,'nrmse':float}` with uncentred cosine and RMS/dynamic-range definitions. A zero estimate receives NCC0 rather than NaN. Units cancel in both normalized metrics. This routine is checked on independent known vectors against a separately implemented Judge routine.

### `compare_outputs(output, reference, mask)`

Inputs are unbatched quantity dictionaries and bool `(H,W)` fixed public mask. For each reference key, select all leading axes at masked pixels and call `array_metrics`. Return nested dictionary `quantity -> {'ncc','nrmse'}`. Missing keys/errors propagate. No crop, alignment, fit, rescaling, magnitude conversion or ground-truth substitution occurs.

### `plot_stages(data, output, diagnostics, upstream)`

Build a2-by3 complete-frame comparison at the final date: fresh apparent network LOS, fresh atmosphere-corrected, fresh deramped, fresh final, reproduced upstream final and fresh-minus-upstream. All fields come from computed output/diagnostics or the explicitly supplied reference. Mask display outside `data['analysis_mask']`; colour bars are metres. Reference/fresh error has its own nonzero numerical display limit even when exactly zero. No numerical result is invented to fill a panel.

### `plot_parameters(data, output, diagnostics)`

Build a2-by3 figure of temporal coherence, elevation residual, velocity, acceleration and both eruption steps. Axes remain original azimuth/range pixels, with complete scene and public mask. Colour bars carry the correct per-panel units. `diagnostics` is accepted by this plotting signature but not read; the plotted quantities are the actual `output` maps. The function does not infer a missing parameter map from an archived result.

### `plot_traces(output, upstream, diagnostics, dates, coordinates)`

Input `dates` is an N-element calendar/datetime array; `coordinates` lists valid raster `(y,x)` pairs. Build a2-by2 figure with paired fresh/upstream final traces, fresh temporal-model residual traces, full-scene residual RMS in mm, and per-date maximum absolute fresh/upstream final difference in mm. The residual is a model-fit diagnostic, explicitly not ground-truth error. The full-scene RMS display includes regions outside the fixed analysis mask and labels this scope; numeric evaluation still follows the fixed public mask.

The notebook additionally makes source-observation, full temporal graph, closure-count/integer-bar and actively computed uniform-versus-weighted comparison figures. These diagnostic cells call the shipped physical operators; none imports private capture scripts or a domain library. Diagnostic arrays and comparison settings are identified with their measured acquisition/method condition.

## 9. Independent Judge contract and thresholds

`evaluation/judge.py` resolves its hidden references/configuration relative to `Path(__file__).resolve().parent`; it resolves public calibration relative to that task. Prediction/report CLI paths remain relative to the caller. Its independent EXPECTED table requires the exact five batch-first float32 shapes. Hidden `ground_truth.npz`, if genuinely present, takes precedence; this real-data task supplies only `baseline_reference.npz`. No function imports solver code or a checkpoint.

| Judge function | Actual interface and behaviour |
|---|---|
| `validate_arrays(path)` | Load NPZ with pickle disabled, require every array/file/shape/real-float32 dtype/finite value; return new unbatched dict; invalid input raises a descriptive exception |
| `score_pair(estimate,reference)` | Independent float64 cosine NCC/RMS-range NRMSE; reject mismatched/nonfinite/degenerate reference; return `(ncc,nrmse)` |
| `observation_consistency(network,data_path)` | Read original phase/pairs/public mask, calculate residual phasors from LOS edge differences on fixed mask; return mean magnitude of edge-averaged phasors |
| `evaluate(prediction,directory=None)` | Validate hidden assets/config/public mask, then prediction, then all ten array metrics and forward consistency; return `(report,exit_code)` without solver execution |
| `main(argv=None)` | Require `--prediction`/`--report`; create report parent and write strict finite UTF-8 JSON; return0 pass,1 valid low-quality,2 input/evaluator/report-write error |

Reports have `status`, `passed`, `metrics`, `thresholds` and `errors`. Every threshold includes its comparison direction. Errors return empty measured metrics. Report-write failure additionally prints stderr and returns2. The config uses nonzero numerical boundaries despite perfect source/self agreement; their independent-rerun rationale and physical limits are in validation. No reference path guessing beyond the two specified reserved filenames is used.

Observation consistency deliberately uses circular phase residuals: integer region corrections disappear under the complex exponential. This tests the supplied physical network movie against original observations, while spatial ambiguity correctness is separately enforced by independent fixtures and full-array reference agreement. It would be scientifically wrong to call this alone proof of correct displacement.

## 10. Reference, fixtures, tests and notebook execution

`evaluation/baseline_reference.npz` stores all five quantities from the original selected method on exact bound observations. It is not generated by cleaned functions and is not called truth. `test/fixtures/upstream_stages.npz` contains source A/B/C/time intervals, combined temporal/geometry design, raw/corrected bound pixel data, phase reliability/availability and source intermediate/final pixel outputs. Nine bound pixels include both common regions and an invalid corner. `upstream_regions.npz` contains independently captured common-region sample coordinates and median integer offsets. `upstream_lasso.npz` contains original continuous and rounded100-sample solutions with their exact matrices/closure cycles.

`notebooks/reference_outputs/upstream_measurements.npz` contains four identified source stage snapshots and selected source traces, consumed by `test_upstream_correction_stage_snapshots_on_full_bound_history` for the same scientific stage review shown in the notebook. `upstream_uniform_sbas.npz` contains original uniform-weight comparison maps, quality and traces, consumed by parity tests and notebook comparisons. `cleaned_measurements.npz`, recorded after actual final execution and indexed in validation, contains fresh numerical summaries rather than a duplicate full movie. Historical source measurements are not overwritten by subsequent notebook execution.

Tests are grouped under `test/unit_test` and `test/comparing_test`. They cover exact acquisition constants/data samples, temporal/triangle/time-model parity, known velocity integration and closure cycles, LASSO KKT, nonuniqueness, original region-median integer rounding and objective, equal-label spatial connectivity and erosion cleanup, atmosphere gauge/no-mutation, analytic joint DEM coefficients, weighted SVD on original real pixels, complete spatial repair and uniform comparison, independently known metrics, strict Judge key/shape/dtype/complex/NaN/Inf cases, missing/evaluator/report-write errors, poor valid zero reconstructions, full fresh CLI output and standalone execution from another working directory. The full-output test intentionally consumes the real submission produced by the documented CLI before pytest; it does not regenerate an oracle fixture using the cleaned solver.

The notebook has exactly six ordered H2 sections: Problem, Data, Physics, Method, Results, Conclusion. It loads raw data, actively runs the complete selected reconstruction, actively runs uniform SBAS, saves/judges the fresh output and displays original/fresh comparisons. Runtime is recorded from the actual kernel executable. Execution uses `jupyter nbconvert --execute`, the shared research kernel and a1800-second per-cell timeout, justified against the approximately7-minute complete CLI run with margin for concurrent CPU use. The executed notebook is retained in the canonical notebook file, while temporary executed copies, PNG inspection files and logs are archived privately outside the final task directory.


### Verified LASSO backend discrepancy

The CVXOPT oracle and local ADMM solve the same convex LASSO objective, but their individual coefficient vectors are not identical. Among 28,800 sampled coefficients, nine rounded values differ, across four sample pixels. The actual difference has Frobenius norm 0.7920480220 and closure-image norm 0.0050718090 (maximum absolute closure difference 0.0003704676); it is not exactly a null-space vector. The local proximal KKT residual is 6.9795e-8 versus 6.0454930e-4 for the original backend. Local objective values are no worse, with maximum local-minus-original objective -7.9200372e-7. Independent active-support null-space perturbations preserve both closure predictions and L1 norm to 1e-12 on each of the four affected samples, proving genuine nonuniqueness; the observed differences also include original stopping precision.

Region-wise median rounded corrections agree for every one of the 288 edges. Full spatially repaired phase agrees bitwise at all 77,760,000 values, and all five complete cleaned output quantities have zero NRMSE against the original method. Validation therefore checks the convex objective/KKT condition and independent nonuniqueness proof at this stage, plus exact scientific region corrections; it does not claim bitwise agreement of the CVXOPT trajectory or individual sample coefficients. The original stricter coefficient test failure and all nine half-integer distances are retained in the private audit indexed by test/validation.md.
