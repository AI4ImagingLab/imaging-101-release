# Shipped interfaces and data flow

The task is self-contained. Runtime imports are standard library, NumPy and
SciPy general computation; Matplotlib is used only for review figures. No
MosGIM2, Lemke package, receiver-processing library or other task is imported.
All source-oracle environments and capture scripts are outside the task.

## Preprocessing

`src.preprocessing.load_data(path)` opens public NPZ with allow_pickle=False,
validates each of the nine observation arrays as finite float64 shape(1,504358),
and returns `(data,grids)`. Data values are unbatched copies shape(504358,);
grids contains grid_time(25,), grid_colatitude(71,), grid_longitude(72,).
Units are seconds, radians and TECu as declared in README. It does not remove,
smooth, interpolate or re-reference observations. The source preprocessing is
already present in the bound measured arrays. `OBS_KEYS` lists exactly the nine
ray keys. The four unused second-shell source coordinate arrays are not bundled;
the selected model uses only the450km first shell and every ray is retained.

## Physical operators

`harmonic_orders(degree=15)` returns degree/order integer arrays length(degree+1)^2,
degree-major then signed-order-major. `real_harmonics(longitude,colatitude,
degree=15)` takes matching1D radians arrays and returns real float64 shape
(points,(degree+1)^2), using the explicit normalized Legendre recurrence.
`slant_factor(elevation,height=450000)` returns a matching dimensionless array;
EARTH_RADIUS=6371000m. These functions have no I/O side effects.

`design_matrix(data,degree=15,intervals=24,sigma0=.02)` constructs the CSR
current-minus-reference operator, shape(points,(intervals+1)*(degree+1)^2), and
diagonal inverse-variance weights(points,). Inputs are the unbatched ray dict;
rhs is not needed for this operator. It validates observation time bounds and
preserves repeated-column summation. Sigma0 is in TECu.

`chunk_bounds(count,gigs=.1)` returns source-compatible half-open row intervals.
It is a deterministic memory partition, not a quality sample.
`temporal_precision(degree=15,intervals=24,sigma_rel=.005)` returns dense float64
coefficient precision, shape6400² by default, first differences only. SigmaRel
has TECu units. `grid_basis(colatitude=None,longitude=None,degree=15)` uses the
declared native grid by default and returns(5112,256). Caller-supplied1D angles
are radians. `render_maps(coefficients,colatitude=None,longitude=None,degree=15)`
accepts a flattened or hour-by-mode float array and returns(hour,nlat,nlon).
`predict(coefficients,data,chunk_size=4096)` evaluates every ray using bounded
working-memory CSR blocks and returns slant-TEC differences(points,).
`residual_metrics(prediction,data)` returns rms_TECu and weighted_rms_TECu as
finite Python floats using all samples, with no mask or alignment.

## Inverse routines

`LemkeTableau(M,q)` receives dense real square M and vector q. Its mutable
`table` stores n×(2n+2) numerical entries; `labels` stores2×(2n+2) variable-kind
and variable-index integers. `wpos` and `zpos` track complementary columns.
`swap(a,b)`, `pivot(row)` and `eliminate(row)` perform the explicit numerical
updates and mutate the tableau. `initialize()` and `step()` return booleans;
`solution()` extracts a length-n float64 z. `lemke(M,q,max_iterations=1000000)`
returns z or raises RuntimeError on a secondary ray or exhausted budget. The
frozen upstream fixture contains initialization and both pivots for an
independently solved3variable known case.

`assemble_normal_system(data,sigma_rel=.005,progress=False)` consumes all bound
rays, sums the source's168chunks and adds the temporal precision. It returns
`(N,b,rPr,dof)`: dense float64 N(6400,6400), b(6400,), scalar weighted y-square,
and integer504102. With progress=True it prints chunk counts. No cache or
hidden output is read. Peak working storage includes the dense normal matrix
and a chunk's sparse matrices.

`negative_node_correction(coefficients,inverse_normal)` returns corrected
coefficients(6400,), multiplier vector(initial_negative_count,), and flattened
initial-negative grid indices of matching length. It evaluates the full native
grid, builds only the selected-row sparse G, computes the LCP matrix and applies
one correction. The new-negative limitation is preserved. The dense inverse
normal is supplied by the caller, not estimated using a hidden target.

`solve_system(N,b,rhs_weighted_square,dof,nonnegative=True)` evaluates the direct
inverse and optional LCP. It returns a dict with coefficients, unconstrained_
coefficients, dual, initial_negative_indices and scalar dispersion. Setting
nonnegative=False is an explicitly labeled notebook ablation; it is not the
standard output. `reconstruct(data,progress=False)` performs fresh assembly and
the selected solve, adds reconstruction(1,25,71,72) and runtime_seconds, and
returns that dict. It does not write any file or load a reference. The inverse
and temporary matrices are released when their calls return.

## CLI and review

`src.main.run(raw_path='data/raw_data.npz',output_path='output/reconstruction.npz')`
loads public observations, actively reconstructs the complete day, creates the
output parent directory and writes only reconstruction in the declared NPZ.
It returns the fresh result dict. `main()` exposes `--raw` and `--output` and
prints runtime. Standard invocation is `python -m src.main`. No backend is
selected on import, and the reconstruction path itself needs no plotting.

`src.visualization.map_metrics(estimate,reference)` returns two length25arrays:
cosine NCC and each hour's dynamic-range NRMSE. It raises for a degenerate
reference range. `save_figure(fig,directory,name)` creates the requested private
directory and writes actual PNG/SVG/PDF files, returning their paths. It does
not select Matplotlib's backend. The active notebook chooses its own display
backend before plotting and executes reconstruction using these local modules.
Its comparison and sensitivity calls are separate from the CLI default.

## Independent evaluation

`evaluation/judge.py` has no solver imports. `evaluate(prediction,asset_dir=ROOT)`
returns `(report,exit_code)`. ROOT is the evaluator's own directory. Actual
ground_truth.npz, when present, takes precedence; malformed actual truth is an
error, never a fallback. Otherwise baseline_reference.npz is used. Both fields
must contain finite real float64 reconstruction(1,25,71,72). All25 hour-specific
metrics must pass. Full-field failures return1 without a costly ray follow-up.
Candidates passing the field gates undergo independent harmonic fitting and
every-ray forward evaluation. The Judge uses general associated-Legendre
functions rather than importing the solver's recurrence. Boundary values must
be finite numeric scalars, excluding bool/string; NCC is within[-1,1] and error
bounds are nonnegative.

The CLI takes required `--prediction` and `--report`. Prediction and report
paths resolve from caller cwd; hidden assets resolve from ROOT. It creates
the report parent, writes UTF-8 finite JSON and returns0/1/2 for pass/fail/error.
Report-write failures return2 with stderr evidence. Invalid NPZ inputs never
execute submitted code. Unit tests independently cover malformed inputs,
reference precedence, configuration types/ranges, known metrics, poor fields,
foreign cwd and report-write failure.

## Evidence consumers

`test/fixtures/qualified_stages.npz` is consumed by upstream stage comparisons;
its24 predetermined rows are diagnostic inputs, not a replacement acquisition.
`lemke_stages.npz` supplies independently captured source tableaux.
`notebooks/reference_outputs/upstream_outputs.npz` identifies original qualified
maps, unconstrained and wrong-time coefficients, and all original predicted rays;
the notebook uses these only for explicitly labeled comparison. Hidden
baseline_reference.npz contains the independent original method maps.
No reference file is consumed by src. Validation records all hashes, archive
locations, precision criteria, commands, environments, limitations and final
native execution evidence.
