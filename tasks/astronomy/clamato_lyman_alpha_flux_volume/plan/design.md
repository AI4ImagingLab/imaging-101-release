# Shipped interfaces and current execution flow

The standard entry point is `python -m src.main` from the task workspace. Importing `src.main` only defines functions and never changes a plotting backend. `run(task_dir=None)` resolves the task root from its source location unless explicitly supplied, loads `data/raw_data.npz`, reconstructs every native voxel, creates `output/`, writes `output/reconstruction.npz` and prints scalar diagnostics. It returns `(field, alpha, info)` to a caller. CLI output contains only the declared reconstruction array; no private cache is required.

## Preprocessing

`Observations(positions, noise, flux)` is a dataclass with float64 arrays `(n,3)`, `(n,)`, `(n,)`. Positions have units Mpc/h and noise/flux are dimensionless.

`load_observations(path)` reads NPZ with `allow_pickle=False`, removes the explicitly declared batch axis from `positions`, `noise_std`, `flux_contrast`, checks consistent shapes/finite values/positive noise and returns `Observations`. It does not clip, refit, normalize, reorder or transform observations.

`grid_coordinates(shape=(60,48,876), extent=(30.,24.,438.))` returns float64 `(product(shape),3)` cell centers with z fastest, in Mpc/h. Optional dimensions support independent small mathematical tests and explicitly labeled diagnostics; the standard runtime uses the native defaults.

`whiten(noise, flux)` returns two float64 `(n,)` arrays `(weights,rhs)` with weights=1/noise and rhs=weights*flux.

## Local covariance and acquisition operators

`kernel_from_q(q, variance=0.05)` accepts nonnegative finite scalar or array squared separations, applies the 200-point source table and cutoff, and returns covariance values with matching shape. Constants are table size200 and maximum separation parameter8.

`pairwise_covariance(a,b,lengths=(2.5,2.5,2.),variance=0.05)` returns dense float64 `(len(a),len(b))` covariance. It is used for small parity/physics cases, rather than the full data-data matrix.

`sparse_covariance(a,b=None,lengths=(2.5,2.5,2.),variance=0.05,tree_b=None)` returns sorted CSR covariance of shape `(len(a),len(b))`. `b=None` means b=a. An optional general spatial tree for b can be reused for grid chunks; the tree must index b divided by lengths. Sparse indexing locates candidates, then the local kernel evaluates their actual covariance. All array coordinates have shape `(n,3)` and units Mpc/h. The two transverse lengths are fixed equal in the selected axisymmetric physical model.

`whitened_product(covariance,weights,vector)` returns float64 `(n,)` A*vector for A=I+WKW. Covariance may be dense for independent tests or sparse in the runtime.

`map_from_coefficients(positions,coefficients,coordinates,chunk_size=1024,lengths=(2.5,2.5,2.),variance=0.05)` returns float64 `(len(coordinates),)` mapped contrasts. It reuses the observation tree, builds covariance chunks and multiplies each by the coefficient vector. It does not retain a full grid-data matrix.

## Explicit solver

`pcg_step(product,rhs,diagonal,x,r,d,delta,full_residual=False)` executes one original mathematical update. `product` maps `(n,)` to `(n,)`; rhs, diagonal and x/r/d are float64 `(n,)`; delta is scalar. It returns new `(x,r,d,delta,q)` arrays/scalar, where q is the matrix product on the old direction. This helper permits independent frozen upstream update tests.

`conjugate_gradients(product,rhs,diagonal,max_iter=500,tol=1e-3,step_r=50)` returns `(whitened_solution,info)`. Info contains boolean success, actual update count `iterations` and float64 residual-norm history including the initialization. Initialization is always zero; a zero rhs returns zero with an explicitly defined zero update count. Exhaustion returns success=False after max_iter−1 updates.

`reconstruct(observations,tol=1e-3,max_iter=500,step_r=50,chunk_size=1024,lengths=(2.5,2.5,2.),variance=0.05,coordinates=None)` builds the observation covariance, runs the custom PCG, converts coefficients, releases the large covariance and maps requested points. With coordinates=None it returns `(field,alpha,info)` with native float64 field `(60,48,876)`, alpha `(n,)` and diagnostics. With explicit `(m,3)` coordinates the field is `(m,)`. Diagnostics include residual norms, update count/success, build/runtime seconds, covariance nonzero count, exact recomputed relative linear-system residual and predicted data contrasts `(n,)`. No hidden assets or upstream packages are loaded.

## Review assets and current checks

`test/fixtures/upstream_stages.npz` is independently captured by a private C++ harness calling unchanged source. The consumer is `test/comparing_test/test_upstream_stages.py`, which covers whitening, the kernel/covariance, preconditioner, matrix product, first/fifth PCG steps, the explicit third-step refresh diagnostic, stopping solve, native grid and map product. Fixture generation never imports local code. `test/unit_test/test_physics.py` independently checks the Gaussian-approximation error bound, symmetry/adjoint identity, noisy-system positive eigenvalues, an independent direct linear solve and the zero-rhs case.

## Standalone evaluation

`evaluation/judge.py:load_field(path)` reads `reconstruction` from NPZ without pickle and requires exact shape `(1,60,48,876)`, float32/float64 and finite real values. `measure(prediction,reference)` independently computes full-array cosine NCC and dynamic-range NRMSE, defining a zero prediction's NCC as0 and rejecting constant/zero evaluator references. `evaluate(prediction_path,asset_dir=None)` loads evaluator-relative metric/reference assets, prefers actual truth if present, and returns `(report,exit_code)`; this real binding has only a method reference. `main(argv=None)` implements required CLI flags, report-directory creation/write handling and native0/1/2. It never imports any solver. `metrics.json` holds the independently frozen nonzero bounds. The default reference is the complete original native map.

## Active review notebook and visualization

The notebook imports `src.main.run` and actively reconstructs the complete primary binding. Its original comparison loads `evaluation/baseline_reference.npz`; `notebooks/reference_outputs/original_diagnostics.npz` supplies separately captured original coefficient/pixel/residual vectors. Fresh `fresh_measurements.npz` stores only active coefficients/predicted pixels/residual history; the full fresh field remains in the declared output NPZ, avoiding an unnecessary additional field copy. `fresh_measurements.json` records actual kernel environment, timings, metrics and sensitivity values. These assets are maintainer-only and are not loaded by the solver.

Fresh equal-noise ablation uses `Observations` with all original positions/contrasts and a constant median released noise. Four additional `reconstruct` calls vary the transverse lengths to2.0/3.0 or the longitudinal length to1.5/2.5 Mpc/h. Every call solves all64332 observations. For these five diagnostics only, explicit coordinates request the entire native middle-x plane, `(48*876,3)`; primary reconstruction and grading remain full-volume. Alternative noise weights and prior lengths are comparison assumptions, never changes to the benchmark data or selected target definition.

`visualization.export(fig,directory,name,arrays,metadata)` writes actual PNG/SVG/PDF and numeric NPZ/JSON into the supplied private figure directory. `contrast(ax,field,title,extent=(0,438,0,24),limit=0.6,difference=False)` renders signed real slices with a zero-centered diverging normalization. Each actual plot has a documented numeric consumer:

- `figure01(observations,directory)`: measured spatial geometry, noise, contrast distribution and sightline coverage.
- `figure02(field,directory)`: six full native transverse sections.
- `figure03(field,original,metrics,directory)`: original/fresh/signed-difference sections and display scatter; metrics use every voxel.
- `figure04(observations,field,info,uniform_slice,uniform_info,original_history,directory)`: fresh noise ablation, complete-observation weighted residuals, disjoint noise-rank quartiles and convergence.
- `figure05(field,directory)`: additionally smoothed display maps/profiles, explicit one-dimensional Fourier power and full-field differences from the native fresh field. Reflect boundaries are explicitly specific to this display experiment.
- `figure06(field,sensitivity_slices,sensitivity_metrics,directory)`: signed prior-length perturbation maps, plane NRMSE and complete-observation weighted RMS. Dictionary keys identify actual independent fresh runs.

All visualization functions consume explicit numeric arguments, do not load an upstream package or cached method result, and return a Matplotlib Figure. No library module selects a plotting backend. Numeric figure arrays and metadata are exported for the private atlas; canonical notebook outputs retain actual plots. The exact kernel executable/backend and native nbconvert process are recorded in validation evidence, independently of notebook artifact existence.
