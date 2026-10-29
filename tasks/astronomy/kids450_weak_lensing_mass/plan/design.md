# Shipped design

The standard entry point is `python -m src.main` from the task directory. It reads only `data/raw_data.npz` and writes `output/reconstruction.npz`. The core unbatched result is E/B(64,64)dimensionless. All public files are solver-independent. No upstream package, archive, hidden reference or cross-task module is imported at runtime.

## Module roles and actual flow

`src.preprocessing.load_data` reads NPZ with allow_pickle=False, removes exactly the one batch dimension and copies arrays without dtype conversion. `src.physics_model` implements acquisition/resolution operators. `src.solvers` applies the fixed selected inverse and exposes explicit comparison arguments. `src.main.run` calls these in sequence, creates the output parent, writes batch-first arrays and returns those same arrays. It never imports plotting utilities or changes a backend.

`src.visualization.review(data,upstream)` actively reruns the local inverse and every plotted alternative; it returns a list of (name, Matplotlib Figure), numerical experiment-table rows, and an exact numeric-source dictionary. It does not save files itself. The notebook writes only fresh scalar measurements to notebooks/reference_outputs/fresh_summary.npz in addition to the standard output. It calls `src.main.run`, loads the independently captured algorithm reference, calls review, displays all four figures and numerical metrics, then launches the independent Judge. An optional environment-supplied private gallery destination saves PNG/PDF/SVG and source NPZ outside the task directory. No cache-preview replaces computation.

`evaluation/judge.py` is standalone and imports no solver. Its evaluate(prediction,asset_dir) validates all declared arrays before scoring, prefers ground_truth.npz when present, otherwise uses baseline_reference.npz, and returns (report,exit_code). CLI report creation is relative to caller paths. Output status is pass/fail/error; codes0/1/2. Missing assets and undefined reference metrics are evaluator errors. Zero estimates are valid below-quality predictions with NCC0.

## Callable signatures actually shipped

| Module | Callable | Inputs / return and side effects |
| --- | --- | --- |
| main.py | `run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')` | Actively reconstruct the bound observations and save batch-first physical arrays. |
| physics_model.py | `kernels(shape)` | Return dimensionless P1,P2 on NumPy's FFT grid; axes are (north,east). |
| physics_model.py | `forward(e, b)` | Return real shear maps gamma1,gamma2 from dimensionless E/B convergence. |
| physics_model.py | `inverse(g1, g2)` | KS93 inversion with zero mean, periodic boundary and unnormalised forward FFT. |
| preprocessing.py | `load_data(path='data/raw_data.npz')` | Return a dict of unbatched copies, retaining stored dtypes and units. |
| solvers.py | `grid_catalogue(data, calibrate=True)` | Weighted ensemble-corrected shear maps; empty pixels are zero. |
| solvers.py | `solve(data, sigma=SMOOTHING_PIXELS, calibrate=True)` | Return unbatched E/B maps after periodic Gaussian sigma (pixels). |
| solvers.py | `reconstruct(data)` | Return required dimensionless float64 arrays (1,64,64). |
| visualization.py | `metrics(a, b)` | Local helper. |
| visualization.py | `style()` | Local helper. |
| visualization.py | `plate(ax, image, title, cmap='viridis', **kwargs)` | Local helper. |
| visualization.py | `review(data, upstream)` | Compute the standard method, calibration ablation, resolution sweep and32 shape-noise nulls. |

## Shapes, precision, defaults and files

The complete stored shapes/dtypes/units and coordinate conventions are in README. Internal arrays preserve stored observation precision; FFT, interpolation, SVD and filtering compute in float64. Default GRID_SIZE=64, RA_CENTRE129.0degrees, DEC_CENTRE-0.5degrees, SMOOTHING_PIXELS2.0. solve returns E,B; grid_catalogue returns gamma1,gamma2,count,weight maps.

Input path defaults to data/raw_data.npz, output path defaults to output/reconstruction.npz. main.run is the only runtime file writer. Judge alone writes output/judge_report.json. Frozen upstream arrays in test/fixtures/upstream.npz are consumed by parity tests; separately identified original/fresh scalar measurement summaries are indexed in test/validation.md. Both tests and notebooks may access hidden assets as maintainer tools; released solvers receive only the README, raw NPZ and environment information.

## Scope and reviewed consistency

The implementation is a current executable numerical extraction, not a proposed architecture. Upstream full frameworks and readers remain in the private archive. No generate_data.py ships because native real observations are reused. Internal source signatures above were generated from the shipped AST, then the equations, factors, axes, scalar constants and caller roles were semantically checked against the source and upstream fixtures. The final validation ledger records actual commands and outcomes.
