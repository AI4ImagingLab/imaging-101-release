# Shipped interfaces and execution flow

The runtime is self-contained within this task. Its only numerical dependencies
are NumPy and SciPy; Matplotlib is used for maintainer visualization. There are no
cross-task imports, checkpoint files or runtime OpenPIV dependencies. Image
decoding occurs only in private capture; the packaged observations are NPZ.

## Standard entry point

`python -m src.main` invokes `main()`. The task root is resolved from its module
location. It loads `data/raw_data.npz`, calls the local solver and writes
`output/reconstruction.npz`, creating `output/` when needed. The keys are
`reconstruction` (float64, 1,7,10,2, pixel/frame interval) and `confidence`
(float64, 1,7,10, dimensionless). It prints elapsed wall time. Importing this
module has no execution or Matplotlib-backend side effects.

## Preprocessing

- `load_frames(path)` opens NPZ with `allow_pickle=False`, reads `frame_a` and
  `frame_b`, removes their batch dimension, casts to float64, and requires finite
  (369,511) images. Returns `(a,b)` in digital counts. Does not mutate the file.
- `windows(frame, size=64, overlap=16)` returns row-major stacked square image
  windows, (70,64,64) for bound frames, using step `size-overlap`. It neither
  pads nor recenters. Returns a copied stack; original input is unchanged.
- `normalize(w)` copies numeric windows to float64, removes their last-two-axis
  mean and divides by population standard deviation. Constants return zero.
  Input and shape are preserved.
- `prepare(a,b,window=32,search=64,overlap=16)` extracts and normalizes both
  search arrays and masks the central support in A. Returns `(aa,bb)`, both
  (70,64,64) float64, dimensionless. No mutation of a or b.

## Physics / correlation

- `correlate(a,b)` population-normalizes each input again and returns the
  dimensionless circular cross-correlation, same shape as the window stack.
  Uses SciPy real FFTs and NumPy fftshift with explicit final division by area.
  Inputs are not mutated.
- `field_coordinates(shape=(369,511),search=64,overlap=16)` returns `(x,y)`
  arrays (7,10) of actual search-window centers, in image pixel coordinates.
  Used by visualization only. It does not use upstream's optional recentering.

## Solver

- `subpixel_peak(c)` takes one real 2D map and returns a float64 two-element
  array `(row,column)`. An edge maximum returns NaN pair. A valid peak causes
  an intentional in-place addition of 1e-7 to the entire map before Gaussian
  or parabolic fitting; this is a scientifically documented upstream convention.
- `peak_confidence(c,width=2)` takes the post-peak map and returns a Python/NumPy
  real scalar peak ratio, or zero on an upstream rejection condition. It copies
  its map only for exclusion and does not change its input.
- `solve(a,b,window=32,search=64,overlap=16,dt=1.)` prepares windows, computes
  correlations, fits every map in row-major order, converts indices to signed
  displacements, and computes confidence on the mutated maps. It returns
  `(velocity,confidence)` with shapes `(1,7,10,2)` and `(1,7,10)` for bound inputs.
  `dt` must be positive. Invalid nonfinite velocity causes a ValueError rather
  than silently interpolating or replacing it. Alternative sizes used only by
  independent physics tests return their corresponding grid shape. No files
  are written. This is a direct one-pass method.

## Notebook and visualization

The active notebook calls `load_frames` and `solve` on the bound observations,
times the fresh execution, writes the declared NPZ, and invokes the standalone
Judge in a subprocess. It loads `notebooks/reference_outputs/upstream.npz`
solely as a labeled historical comparison from this task's independent upstream
run. It calls `compare(a,b,reconstruction,reference,confidence)` which returns a
Matplotlib Figure with two frames, vector field, paired upstream/fresh vertical
velocity maps and fresh confidence. A separate paired metric bar chart and
absolute component residual map explain numerical parity. The function creates
no files and never changes the plotting backend. The notebook has no cache-only
branch and no upstream imports.

## Evaluation and tests

`evaluation/judge.py --prediction PATH --report PATH` resolves its own hidden
assets by file location and caller-supplied paths against caller cwd. It validates
every required shape, real floating dtype and finite value before computing
metrics. It prefers actual truth if present, otherwise the baseline. No truth is
included here. It creates report parents and exits 0/1/2 for pass/fail/error.
`load_arrays`, `score`, `evaluate`, `main` are evaluator-only functions, independent
of solver imports. Tests directly load this Judge with importlib and also run the
CLI from a foreign working directory. Tests never import the original library.

All unit and comparison tests live under `test/`. Frozen `upstream_stages.npz`
is consumed only by comparison tests; two-key `upstream.npz` is consumed only by
the notebook; `baseline_reference.npz` is consumed by Judge and its contract tests.
`cleaned_measurements.json` records fresh notebook timing and exact Judge metrics.
These independent consumers explain the small separate reference archives.

## Complete argument, error and ownership contracts

All indices/sizes below are Python integers in pixels. Unless explicitly stated,
functions return newly allocated arrays and perform no disk I/O. Numerical
helpers assume compatible real numeric arrays; they are not a second public
NPZ validator. Invalid unsupported window configurations may propagate NumPy or
Python errors; no broad geometry validation is claimed.

| Callable | Each input and default | Return, dtype and ownership |
|---|---|---|
| `load_frames(path)` | String/Path to batch-first NPZ; required keys frame_a/frame_b | Tuple of copied float64 `(369,511)` count arrays; file/key errors propagate, wrong unbatched shape/nonfinite raises ValueError |
| `windows(frame,size=64,overlap=16)` | Real `(H,W)` counts; positive size/stride expected | Copied stack `(ny*nx,size,size)`, input dtype, same units; ny=floor((H-size)/(size-overlap))+1, analogous nx |
| `normalize(w)` | Numeric `(...,H,W)` with last two window axes | Same shape float64 dimensionless; zero where per-window std is zero |
| `prepare(a,b,window=32,search=64,overlap=16)` | Two equal-shaped count images; integer support/search/overlap | Tuple aa,bb `(70,64,64)` float64 dimensionless for bound data |
| `correlate(a,b)` | Compatible prepared `(K,S,S)` arrays | `(K,S,S)` float64 normalized mean-product; no in-place modification |
| `field_coordinates(shape=(369,511),search=64,overlap=16)` | Image `(H,W)` shape and pixel sizes | Tuple x,y float arrays `(ny,nx)` in image pixels |
| `subpixel_peak(c)` | Writable real2D correlation map | Two float64 row/column coordinates in map pixels; NaN pair if edge; adds1e-7 to c only for interior peak |
| `peak_confidence(c,width=2)` | Post-fit2D map; exclusion halfwidth in pixels | Dimensionless scalar ratio or0; copies map before exclusion |
| `solve(a,b,window=32,search=64,overlap=16,dt=1.)` | Count images; dt in frame intervals, must be positive | Tuple batch-first float64 velocity `(1,ny,nx,2)` pixel/interval and confidence `(1,ny,nx)`; raises ValueError for invalid dt or nonfinite velocity |
| `compare(a,b,reconstruction,reference,confidence)` | Two count images; batch-first velocity pair and fresh confidence | Matplotlib Figure, no array/file modification; public x/y field coordinates used for quiver |
| `main()` | No Python arguments; no task flags | None; loads module-root data, writes compressed module-root submission and prints time |

The nested `offset(a,b)` inside subpixel_peak is a closure over the central
intensity, takes two neighbor scalar correlations and returns a scalar pixel
offset; it is not a separate module API. Its zero-denominator guard applies to
the logarithmic fit only. The solver computes confidence on the same maps after
all subpixel_peak calls, preserving their intentional mutation. It checks
velocity finiteness; confidence finiteness is enforced at the Judge boundary.

Evaluator contracts are independent of these numerical helpers:
`load_arrays(path)` returns a two-key dictionary with exact float64 batch shapes
or raises a descriptive file/key/shape/dtype/finiteness error. `score(pred,ref)`
takes those dictionaries and returns ncc, nrmse and confidence_relative_l2 Python
floats. It rejects zero-norm/constant reference or zero confidence norm.
`evaluate(prediction)` has no optional asset parameter: it loads configuration
and reference relative to its own file, returns `(report,code)`, catches numeric
and asset errors, and records empty metrics on errors. `main()` parses required
prediction/report flags and writes strict UTF8 JSON; only this wrapper writes a
report. No evaluator function reads, executes or imports submitted solver code.

## Tests and review execution

Unit tests cover constant normalization, windows, analytic Gaussian/parabolic
fits, edge/weak/secondary-peak confidence, signed translation and metric cases.
Comparison tests consume independently captured upstream stages, including the
post-fit mutated maps; physical tests compare FFT correlation to an explicit
periodic sum. Pipeline tests execute the declared CLI and standalone evaluator,
including malformed and poor submissions and foreign cwd. The notebook executes
all local numerical stages before loading the labeled upstream comparison. The
three figure groups and full metric table remain part of the review, not an
optional cache branch. No helper imports or downloads an upstream implementation.
