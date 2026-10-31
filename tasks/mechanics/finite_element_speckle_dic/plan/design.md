# Finite-element displacement imaging: callable and execution design

## 1. Bound interface and numerical responsibilities

The task has two images420-by260 and a known Q4 mesh15-by2 with8-by4
connectivity. Images and coordinates are float64; connectivity is int64. Units
are digital counts, pixels and dimensionless node indices, respectively. Internal
arrays omit the leading batch axis. The only submitted inverse quantity is the
15-node2-component displacement, not absolute current positions or strain.
No module reads an upstream checkout or uses another task. NumPy provides dense
assembly/inverse products; SciPy provides explicit spline interpolation; the
objective, Jacobian blocks, normal assembly and inverse iterations are local.
Matplotlib is imported only by visualization and the review notebook.

## 2. Preprocessing API

```python
load_data(path=Path('data/raw_data.npz'))
```

`path` is str/Path to a visible NPZ, default caller-relative path above. Returns a
4-tuple `(reference,deformed,nodes,elements)`: float64 `(420,260)` counts for each
image, float64 `(15,2)` pixel coordinates in x/y order, and int64 `(8,4)` node
indices. The implementation copies arrays after selecting batch0; it does not
normalize intensities, decode images or apply a second vertical reversal. NPZ is
opened with `allow_pickle=False`. Missing file/key and invalid NPZ errors
propagate. This loader assumes the fixed validated observation contract; it does
not claim to validate every malformed shape/dtype. No output file is written.

## 3. Geometry, sampling and differentiation APIs

```python
shape_functions(xi, eta)
```

Both arguments are scalar or broadcast-compatible real arrays of parent
coordinates, dimensionless in[0,1] for the bound element. Return float values with
broadcast leading shape plus final axis4, in lower-left/lower-right/upper-right/
upper-left node order. No mutation or files. Incompatible broadcast shapes raise
NumPy errors; it does not clip parent coordinates.

```python
element_pixels(corners)
```

`corners` is float `(4,2)`, reference x/y pixels for an axis-aligned rectangular
Q4 element. Return `(x,y,N)`: integer `(P,)` image columns/rows and float64 `(P,4)`
shape values; P=6561 on each bound80-square element. Floor/ceil determine bounds,
then both endpoints are included. Pixel order is flattened meshgrid row-major.
Only this validated regular reference geometry is supported; there is no general
nonrectangular inverse-map solver hidden behind this helper.

```python
prepare_reference(image, nodes, elements)
```

`image` float64 `(420,260)` counts, `nodes` float64 `(15,2)` pixels and `elements`
int64 `(8,4)` node indices. Returns `(blocks,inverse)`; blocks is length8 list of
`(N,B,I0)`, respectively float64 `(6561,4)` dimensionless, `(8,6561)` count/pixel,
and `(6561,)` count. `inverse` is float64 `(30,30)`, inverse normal curvature in
pixel-squared/count-squared. Local B rows contain all x then all y DOFs; global
matrix interleaves components. Shared node scatter and shared-edge residual
multiplicity are explicit. Image/node inputs are not changed. Invalid indices
raise; singular texture propagates `numpy.linalg.LinAlgError` from the actual
matrix inverse. No unimplemented regularization or pseudoinverse is promised.

```python
filtered_frames(image, nodes, elements, pad=10, order=4)
```

`image`, `nodes`, `elements` have the same contracts, using the deformed image.
`pad` is integer pixels; `order` integer spline degree. Return length8 list of
`(frame,lower)`: float64 `(100,100)` spline coefficients in counts and integer
`(2,)` initial lower x/y bounds in pixels. The full image is prefiltered first
using SciPy mirror boundary, then cropped with exclusive upper endpoint.
Defaults match the original method; changing arguments is a helper capability,
not part of the frozen standard configuration. SciPy validates unsupported
orders. Bound reference nodes lie far enough from image edges; arbitrary mesh
boundary safety is not silently added. Inputs remain unchanged.

```python
sample_element(frame, lower, N, current_nodes, pad=10, order=4)
```

`frame` float64 `(100,100)` coefficients, `lower` integer `(2,)` pixel lower
bounds, `N` float64 `(6561,4)`, `current_nodes` float64 `(4,2)` pixels; scalar
pad/order match frame construction. Returns float64 `(6561,)` counts. It computes
`N@(current_nodes-lower+pad)`, reverses x/y for SciPy array-coordinate order and
calls `map_coordinates(prefilter=False,order=order)` with default constant-zero
exterior. It does not re-prefilter or recrop at moving positions. No mutation.

```python
green_strain(nodes, displaced_nodes, elements)
```

`nodes` and `displaced_nodes` float64 `(15,2)` reference/current pixels;
`elements` int64 `(8,4)`. Return float64 `(8,2,2)` dimensionless Green-Lagrange
strain at element centers. Uses exact Q4 parent derivatives and right
Cauchy-Green tensor, without small-strain linearization. Singular reference
Jacobian raises NumPy LinAlgError. This is a geometric review helper, not a
second reconstruction or an additional required submitted array.

## 4. Solver and CLI APIs

```python
reconstruct(reference, deformed, nodes, elements,
            max_iterations=80, tolerance=1e-8, order=4, pad=10)
```

Images/nodes/connectivity follow Section2. `max_iterations` is a count;
`tolerance` is pixel maximum component increment; `order` and `pad` are as above.
Return `(displacement,history)`: float64 `(15,2)` pixels and float64 `(K,2)`,
K<=80, columns pre-update SSD in count-squared and maximum increment in pixels.
Input arrays are not mutated. The current-node copy starts at reference nodes,
reference blocks/H inverse and deformed-frame coefficients are prepared once,
then every update gathers local nodes, samples, forms signed residual, scatters
B*r, multiplies by H inverse and adds its interleaved reshape. No files or hidden
assets are read by this function. It raises RuntimeError if no update meets the
stop tolerance. Linear algebra/interpolation input errors propagate unchanged.
There is no line-search or false optimizer-success flag.

```python
main()
```

No Python parameters; CLI has no task-specific flags. Standard invocation is
`python -m src.main` from this task workspace. This main uses caller-relative loader/output
paths: read `data/raw_data.npz`, reconstruct with defaults, create `output/` and
save compressed NPZ key `reconstruction=displacement[None]`. It prints update
count/final increment and returns None. History remains in memory. An import
only defines functions and never changes the Matplotlib backend. Execution or
write failures propagate rather than replacing the result with an archived one.

## 5. Review visualization API

```python
plot_comparison(reference, deformed, nodes, elements, displacement, upstream)
```

Images/nodes/elements as above; `displacement` and `upstream` float64 `(15,2)`
pixels. Return one Matplotlib Figure with six axes panels and a strain colorbar.
The arguments identify fresh/independent measurements; the function itself loads
no references and computes no fit. It shows both images at0..255,25-times arrow
lengths, paired x/y displacement curves and derived Eyy. Its pcolormesh deliberately
uses the bound node grid `(5,3,2)`; arbitrary mesh visualization is not claimed.
Inputs are unchanged and no figure file is saved. Backend selection belongs to
the caller; plotting has no effect on the standard CLI.

## 6. Standalone Judge contracts

```python
score(prediction, reference)
evaluate(prediction, asset_directory=None)
main()
```

`score` receives matching real numeric arrays, normally `(1,15,2)` pixels. Return
a dict of Python floats: displacement_rmse_pixels in pixels, cosine ncc and
range-normalized nrmse dimensionless. It assumes validated shapes; a constant
reference raises ValueError. Zero estimate NCC is0. `evaluate` takes prediction
str/Path (caller-relative if not absolute) and optional evaluator directory
str/Path, default file-relative `evaluation/`. It reads metrics.json and actual
truth if present, otherwise baseline_reference.npz, never solver code. Both
reference and prediction must be exact `(1,15,2)`, float32/64 and finite. Required
key is reconstruction; no reshape, crop, alignment or dtype conversion is applied.
Overflow/nonfinite derived metrics are errors with empty metrics. Return is
`(report,code)`, report fields status,passed,metrics,thresholds,errors, and code
0 pass/1 numerical-quality failure/2 invalid input or evaluator error. Exceptions
are captured into descriptive errors. It writes nothing.

Judge `main()` parses required --prediction/--report, calls evaluate, creates
report parents and writes strict finite UTF-8 JSON. Paths follow caller cwd.
Failure to write prints stderr and returns2. Otherwise it returns evaluate's
code, used by SystemExit. The Judge imports stdlib/NumPy only; numerical
comparison is independent of the source solver.

## 7. Distinct execution and data flows

```text
CLI: public NPZ -> copied unbatched arrays -> reference blocks/H inverse
  + deformed spline crops -> fixed-reference Newton -> displacement[None]
  -> output/reconstruction.npz
Judge CLI: submission + hidden declared reference + metrics -> strict report
Notebook Data: public NPZ inspection + local loader
Notebook Results: active reconstruct -> fresh submission -> independent Judge
  + archived upstream measurement -> component/strain/convergence comparisons
```

The notebook follows exactly Problem,Data,Physics,Method,Results,Conclusion.
Results also recomputes the final photometric residual (history is pre-update),
checks metric definitions independently and preserves both scientific figures.
The review runner snapshots every reference_outputs file before execution,
archives newly generated measurements externally, then restores the exact
snapshot bytes. The shipped executed notebook/output are fresh; restored
reference snapshots remain identified historical measurements. This preservation
is an external review procedure, not an extra callable runtime API.

## 8. Tests and hidden-asset consumers

| Component | Independent check / consumer |
|---|---|
| shape_functions/green_strain | unit_test/test_physics: nodal identity, partition, affine F, rigid rotation |
| Reference B and normal assembly | analytic polynomial finite differences; comparing_test/test_upstream N/B/I/H fixtures |
| Sampling/update | original one-update fixture; full10-update original final displacement |
| Nonlinear pipeline | comparing_test/test_physics_pipeline: residual reduction and valid interpolation interior |
| CLI and public geometry | test_pipeline: actual CLI, foreign-cwd Judge, README dimensions/bounds/connectivity |
| Judge metrics/contracts | test_judge: known arrays, missing/key/shape/dtype/object/corrupt/overflow, poor/valid, missing assets and report-write stderr |

`test/fixtures/upstream_stages.npz` is frozen from unchanged original code.
`notebooks/reference_outputs/upstream_measurements.npz` is loaded only for review;
`cleaned_measurements.npz` identifies prior fresh review arrays/history. Judge
uses `evaluation/baseline_reference.npz`, never these notebook caches. No actual
truth is invented. `evaluation/metrics.json` remains unchanged. The complete
original MIT notice is consolidated in `test/validation.md`; private original
files and all revision execution evidence are outside the released task.
