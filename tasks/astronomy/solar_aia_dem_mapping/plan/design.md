# Shipped design

The standard entry point is `python -m src.main` from the task directory. It reads only `data/raw_data.npz` and writes `output/reconstruction.npz`. The core unbatched result is DEM(48,48,20)cm^-5K^-1. All public files are solver-independent. No upstream package, archive, hidden reference or cross-task module is imported at runtime.

## Module roles and actual flow

`src.preprocessing.load_data` reads NPZ with allow_pickle=False, removes exactly the one batch dimension and copies arrays without dtype conversion. `src.physics_model` implements acquisition/resolution operators. `src.solvers` applies the fixed selected inverse and exposes explicit comparison arguments. `src.main.run` calls these in sequence, creates the output parent, writes batch-first arrays and returns those same arrays. It never imports plotting utilities or changes a backend.

`src.visualization.review(data,upstream)` actively reruns the local inverse and every plotted alternative; it returns a list of (name, Matplotlib Figure), numerical experiment-table rows, and an exact numeric-source dictionary. It does not save files itself. The notebook writes only fresh scalar measurements to notebooks/reference_outputs/fresh_summary.npz in addition to the standard output. It calls `src.main.run`, loads the independently captured algorithm reference, calls review, displays all four figures and numerical metrics, then launches the independent Judge. An optional environment-supplied private gallery destination saves PNG/PDF/SVG and source NPZ outside the task directory. No cache-preview replaces computation.

`evaluation/judge.py` is standalone and imports no solver. Its evaluate(prediction,asset_dir) validates all declared arrays before scoring, prefers ground_truth.npz when present, otherwise uses baseline_reference.npz, and returns (report,exit_code). CLI report creation is relative to caller paths. Output status is pass/fail/error; codes0/1/2. Missing assets and undefined reference metrics are evaluator errors. Zero estimates are valid below-quality predictions with NCC0.

## Callable signatures actually shipped

| Module | Callable | Inputs / return and side effects |
| --- | --- | --- |
| main.py | `run(data_path='data/raw_data.npz', output_path='output/reconstruction.npz')` | Actively reconstruct the bound observations and save batch-first physical arrays. |
| physics_model.py | `response_matrix(response, response_log_temperature, temperature_edges)` | Return R(nt,6) in DN cm^5 K s^-1 pixel^-1, mid-log-bin quadrature. |
| physics_model.py | `forward(dem, matrix)` | Temperature-integrated model count rates for any leading spatial axes. |
| preprocessing.py | `load_data(path='data/raw_data.npz')` | Return a dict of unbatched copies, retaining stored dtypes and units. |
| solvers.py | `_warn_nmu_deprecated(nmu)` | Warn if the deprecated ``nmu`` argument was passed. |
| solvers.py | `dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, reg_tweak=1.0, max_iter=10, rgt_fact=1.5, dem_norm0=None, nmu=None, warn=True, l_emd=False)` | Recover a DEM for one observation vector. |
| solvers.py | `_standard_form_svd(A, ldiag)` | SVD of the response with the diagonal constraint divided out, ``A L^-1 = U diag(s) V^T``. |
| solvers.py | `_misfit(basis, coef, outside, lam)` | Misfit ||A x - d||^2 of the regularised solution at strength ``lam``. |
| solvers.py | `_discrepancy_lambda(basis, data, target)` | Find the regularisation strength at which the misfit just reaches ``target`` (the |
| solvers.py | `_regularised_inverse(basis, lam)` | The (nt, nf) matrix ``L^-1 V diag(s / (s^2 + lam)) U^T`` taking weighted data to the DEM. |
| solvers.py | `solve(data, reg_tweak=1.0, gloci=False, error_scale=1.0)` | Return DEM, vertical sigma, temperature sigma, chi2/channel, fitted count rates. |
| solvers.py | `reconstruct(data)` | Return float64 DEM (1,48,48,20), in cm^-5 K^-1. |
| visualization.py | `metrics(a, b)` | Local helper. |
| visualization.py | `style()` | Local helper. |
| visualization.py | `plate(ax, image, title, cmap='viridis', **kwargs)` | Local helper. |
| visualization.py | `review(data, upstream)` | Compute the full48x48x20 inverse, propagated errors and three discrepancy/weighting alternatives. |

## Shapes, precision, defaults and files

The complete stored shapes/dtypes/units and coordinate conventions are in README. Internal arrays preserve stored observation precision; FFT, interpolation, SVD and filtering compute in float64. solve returns DEM,vertical_sigma,temperature_sigma,chi2_per_channel,fitted_counts. Defaults reg_tweak1.0, glociFalse, error_scale1.0. Temperature scaling1e15 is algebraic conditioning and is undone before output. dem_pix max_iter10, rgt_fact1.5, selfnormalization and all uncertainty conventions are explained in approach.

Input path defaults to data/raw_data.npz, output path defaults to output/reconstruction.npz. main.run is the only runtime file writer. Judge alone writes output/judge_report.json. Frozen upstream arrays in test/fixtures/upstream.npz are consumed by parity tests; separately identified original/fresh scalar measurement summaries are indexed in test/validation.md. Both tests and notebooks may access hidden assets as maintainer tools; released solvers receive only the README, raw NPZ and environment information.

## Scope and reviewed consistency

The implementation is a current executable numerical extraction, not a proposed architecture. Upstream full frameworks and readers remain in the private archive. No generate_data.py ships because native real observations are reused. Internal source signatures above were generated from the shipped AST, then the equations, factors, axes, scalar constants and caller roles were semantically checked against the source and upstream fixtures. The final validation ledger records actual commands and outcomes.
