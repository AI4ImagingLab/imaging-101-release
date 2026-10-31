# Shipped execution flow

`python -m src.main` loads only `data/raw_data.npz`, reconstructs from the identity map and writes `output/reconstruction.npz`. No checkpoint, numerical fixture, truth or upstream package is read by this flow. Existing author observations are reused, so no synthetic-data generator is shipped. The optional CLI `--diagnostics PATH` writes fresh internal arrays to a maintainer-selected path; it never loads or resumes from that path. The entire default reconstruction uses CPU general NumPy/SciPy primitives.

## Preprocessing

- `preprocessing.load_observations(path=None) -> dict`: default path is task-local `data/raw_data.npz`. It loads with `allow_pickle=False`, removes the single batch axis and returns independent float64 arrays `x`, `y`, `fluence`, each(200,200). Coordinates are cm and fluence counts/pixel. It requires finite arrays and strictly positive counts. No mean subtraction, filtering, transpose or downcast is applied.
- `preprocessing.uniform_incident_fluence(fluence) -> ndarray`: returns a same-shaped float64 constant array with value mean(fluence), preserving the original count normalization.

## Local physics

- `physics_model.mapping_derivatives(potential,x,y) -> (gx,gy,hxx,hyy,hxy)`: accepts equal-shaped two-dimensional float64 arrays; potential cm², coordinates cm. Returns gradients cm and Hessian entries dimensionless. Normal Neumann gradients and zero mixed edge Hessian are assigned explicitly. Spacings come from coordinate subtraction. The rectangular upstream fixture checks these conventions.
- `fluence_interpolator(x,y,fluence) -> RegularGridInterpolator`: constructs a bilinear general numerical interpolator. Its two axes are x[:,0], y[0,:]; exterior fill is mean(fluence). It is built once per reconstruction and reused.
- `transport_state(potential,x,y,incident,interpolate) -> dict`: evaluates the adjusted mapping, Hessian, determinant, interpolated observed counts and log-mass residual. Return keys are `phix`, `phiy` (cm), `phixx`, `phiyy`, `phixy`, `det` (dimensionless), `fluxn` (counts/pixel), `Fn` (dimensionless). It performs no potential update.
- `velocity_from_mapping(x,y,map_x,map_y) -> ndarray`: returns(2,nx,ny) cm/s in laboratory X/Y components using the fixed author distances and speed.
- `magnetic_from_velocity(velocity) -> ndarray`: accepts(2,nx,ny) cm/s and returns the two laboratory field integrals in G cm. Its signs assume the specified beam into the detector, along negative laboratory Z.
- `magnetic_from_mapping(x,y,map_x,map_y) -> ndarray`: composes those two physical conversions. The output is(2,nx,ny), ordered Bx then By.
- `mapping_from_magnetic(x,y,magnetic) -> (map_x,map_y)`: reverses the same physical conversion for forward checks. It is an analysis function, not the initialization or a truth-derived calibration.
- `ray_forward_fluence(x,y,magnetic,incident_mean,subpixels=8,beam_direction=-1) -> ndarray`: independent analysis forward model with counts/pixel output(nx,ny). Uses equal-weight uniform rays, bilinear physical field samples and detector binning. `beam_direction=+1` is an explicitly wrong experimental-sign control used only in the notebook. Processes128 sampled x rows per block. It writes no files and reads no hidden field.

The constants in this module exactly match the README acquisition table. There are no imports from the upstream source or another task. SciPy interpolation is a general numerical primitive; no domain forward/inverse package hides any field conversion or transport update.

## Local inverse and baseline

`solvers.reconstruct(x,y,fluence,dt=1e-8,tol=1e-3,max_steps=400000,snapshot_steps=(0,1000,10000,50000,100000,200000,399999)) -> dict` actively reconstructs the supplied arrays. Snapshot indices are zero-based iteration indices; index1000 denotes1001 Euler updates. The initialization is always the identity potential. The return dictionary contains:

- `reconstruction`: float64(2,nx,ny), G cm; physical answer from the last pre-update map.
- `phin`: float64(nx,ny), final post-update potential in cm².
- All eight named physical state arrays from `transport_state`.
- `RMSfluxerror`: float64(max_steps), RMS potential increment in cm²; unused tail is zero if stopped early, following source storage.
- `residual_norm`: float64(iterations), Euclidean log-residual norm before each update.
- `snapshots`: dictionary mapping selected zero-based iteration indices to freshly computed field arrays.
- `iterations`: completed-update integer; `stopping`: `tolerance` or `maximum_steps`; `negative_determinant_count`: total number of nonpositive determinant pixels encountered; `runtime_seconds`: actual elapsed wall time.

`solvers.linear_poisson(x,y,fluence) -> ndarray` computes the explicitly stated separate DCT weak-contrast comparison. Its output has the same two-component field shape and units. It does not call the nonlinear reconstruction, load references or write files.

## Main and diagnostic capture

`main.main(diagnostics=None) -> dict` calls the default loader and complete default nonlinear inverse, creates task-local `output/`, writes exactly the batch-first `reconstruction` array to the submission NPZ and prints iterations, stopping reason and actual runtime. It returns the fresh result dictionary for maintainers. If diagnostics is provided, `save_diagnostics(result,path)` serializes fresh ndarray values, snapshot arrays, iteration count, runtime and determinant count into a numeric NPZ at that path. This optional side effect is recorded by the private validation runner.

Importing `src.main` does not select a Matplotlib backend. No library module calls `matplotlib.use`. The notebook imports the cleaned functions directly and performs a separate fresh full reconstruction rather than loading the CLI result as its reconstruction. Upstream results loaded there are explicitly identified comparisons.

## Figure and metric interfaces

`visualization.image_metrics(estimate,reference) -> dict` returns cosine `ncc`, dynamic-range `nrmse` and explicitly named `relative_l2` for finite real two-dimensional image arrays in matching units. It uses no alignment or fitted gain and rejects a constant/zero reference. `field_metrics(estimate,reference) -> dict` applies it independently to the two components, returning Bx/By subdictionaries. Independent known numerical cases check these formulas separately from the Judge.

`visualization.make_figures(data,truth,upstream,fresh,linear,forwards,directory=None) -> (figures,metrics)` accepts the public data dictionary, two-component actual truth and original-field arrays, a freshly computed full inverse dictionary, a freshly computed linear field, and named actually computed ray-forward count arrays. It creates six scientific comparisons. Passing a directory exports each as PNG/SVG/PDF plus full numeric NPZ, full-precision metrics JSON and a figure-to-array provenance index. The comparison count panels share the observed-image range, with colorbar triangles marking display saturation. Signed fields/residuals use neutral-zero diverging scales. No missing method is filled or recomputed secretly inside plotting. The notebook is the caller and actively produces every local result before plotting.

## Evaluation and tests

`evaluation/judge.py` is a standalone script with required `--prediction` and `--report` paths relative to its caller. Hidden assets resolve relative to the Judge file. It imports no solver. Its independently implemented float64 metrics check actual truth on both components. Shapes, real floating dtype and finite values are validated before scoring. The public contract permits float32 or float64 submissions. A valid low-quality result exits1; malformed input or evaluator/report failures exit2; a pass exits0. JSON uses finite values and includes status, passed, metrics, thresholds and errors.

Upstream fixtures are frozen numeric outputs of unchanged source execution. Stage tests compare native1/2/7-step full-grid states and a rectangular actual-data seven-step state. The large fixture contains the full native potential, map, Hessian, determinant and history, with its capture hash recorded in validation. Unit tests independently check analytic Hessians, physical boundary constraints, bilinear interpolation, fluence-scale invariance, Lorentz conversion and a known-positive-field particle ODE. The notebook and figure functions have explicit numeric-source provenance rather than image-pixel fixtures.

Complete default400000-update CLI and active-Notebook runs are independently required and timed in validation. `test_full_pipeline_artifacts.py` then audits their produced full state/submission against frozen original arrays and the independent Judge. It intentionally does not redundantly perform another25-minute inverse on each pytest call. Missing full executed assets cause failure; their presence never replaces the mandatory recorded active executions.
