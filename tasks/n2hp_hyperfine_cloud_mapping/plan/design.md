# Shipped code design and scientific execution contract

## Scope, dependency boundaries and actual caller flow

This document describes shipped callable functions, not a proposed architecture.
All paths are task-workspace-relative unless explicitly called script-relative.
Real arrays are row-major and batch-first on disk. Computation removes only the
documented batch axis. Units are specified per interface; a shape match is not
permission to normalize, transpose or change physical scale. Library imports
do not reconstruct, write files or select a Matplotlib backend.

CLI: raw NPZ -> load_observations -> reconstruct -> estimate_noise
from visible line-free channels -> nine independent fit_spectrum calls, each
with local brightness/optical_depth and generic bounded TRF -> four map arrays
-> compressed standard submission. main imports preprocessing/solvers; both
preprocessing and solvers import physical-model constants/functions; solvers
also imports generic scipy.optimize.least_squares. Visualization uses Matplotlib.
Judge separately imports NumPy and standard library and evaluates its own model.

Notebook Data loads brightness/frequencies and displays all visible calibration
array contracts. Method explains the original MPFIT/local TRF substitution and
the bound patch selection; Results actively reconstructs all nine spectra,
predicts fresh model channels, writes/scorers four maps, and compares independently
captured original parameters/models/losses. It preserves the four-map rows,
three measured/fresh/original spectral groups with normalized residuals,
four-map error bars and paired nine-pixel reduced chi-square bars. It records
new runtime privately and does not overwrite packaged reference_outputs files.
No extra neighboring-guess algorithm or spatial deconvolution is claimed.

## Function-by-function runtime and evaluator interfaces

The signatures are transcribed from actual source. The following contracts
distinguish explicit validation from caller preconditions; they do not invent
checks that an individual numerical helper does not implement.

### src/main.py

```python
def main():
```

No parameters. Loads default public observations, calls reconstruct, creates output/, writes the four required compressed NPZ map arrays at output/reconstruction.npz and returns the same dictionary. Arrays are float64 (1,3,3), units in OUTPUT_KEYS order K,1,km/s,km/s. Uses caller task cwd and prints nothing. Loader/optimizer/filesystem exceptions propagate. No plotting or backend change on import. Consumer: subprocess CLI/Judge pipeline test and backend-import test.

### src/physics_model.py

```python
def optical_depth(frequency, parameters):
```

Parameters `frequency`: scalar/array float Hz, normally (501,); `parameters`: numeric (4,) ordered Te K,tau dimensionless,v km/s,sigma km/s. Returns float64 opacity with frequency shape, dimensionless. Te is unused by this stage. Sigma<=0 raises ValueError; otherwise each of 15 components uses explicit offsets/frequencies, normalized actual strengths and component-specific shift/width. No finite/domain/parameter-count validator beyond sigma here; the inverse bounds and loading provide normal inputs. No file/state mutation. Tests: frozen original opacity, linear tau scaling, Gaussian integral and redshift sign.

```python
def brightness(frequency, parameters):
```

Parameters `frequency`: float Hz array shape (F,); `parameters`: (Te,tau,v,sigma) as above. Returns same-shaped floating brightness K from positive-temperature Planck-equivalent source minus fixed 2.73 K background times 1-exp(-opacity). Te<=0 raises ValueError; sigma errors propagate from optical_depth. No baseline fit, telescope convolution, group aggregation, normalization or I/O. Tests: frozen original forward probe, zero opacity/equilibrium, positive/saturating transfer and numerical Jacobian parity.

```python
def predict_cube(frequency, parameters):
```

Parameters `frequency`: (F,) Hz; `parameters`: float64 (4,Y,X) physical maps in Te,tau,v,sigma order. Returns float64 (F,Y,X) modeled K brightness by row-major calls to brightness. No batch axis is accepted in this helper's intended map shape. Shape/domain errors propagate from indexing and forward calls; no hidden-reference read, interpolation or fit. Consumer: full-spectrum parity/weighted-loss test and notebook fresh spectra/residual plots.

### src/preprocessing.py

```python
def load_observations(path=Path('data/raw_data.npz')):
```

Parameter `path`: str/Path, default Path('data/raw_data.npz'). Reads with allow_pickle=False and requires brightness (1,501,3,3), frequency (1,501), both finite. Returns tuple (cube,frequency) newly cast float64 (501,3,3) K and (501,) Hz. It removes only the leading batch axis. Missing keys/files propagate; wrong shape/nonfinite values raise ValueError. Public calibration arrays remain in NPZ; the runtime uses matching explicit constants checked by tests rather than pretending this loader reads those arrays. No write, noise injection or interpolation.

```python
def estimate_noise(cube, frequency):
```

Parameters `cube`: float64 (F,Y,X) observed K brightness; `frequency`: float64 (F,) Hz centers. Returns float64 (Y,X) K, population standard deviation over radio velocity 20..28 km/s inclusive using REST_FREQUENCY=93176265000 Hz and C_KMS. At least two selected channels and all positive noise values are required, else ValueError. Finiteness of normal inputs is enforced upstream; helper does not introduce a target-dependent calibration or fit a continuum. No I/O. Tests: exact original noise, independently computed ddof=0 expression and missing-calibration-channel case.

### src/solvers.py

```python
def fit_spectrum(frequency, spectrum, noise):
```

Parameters `frequency`: (F,) float Hz; `spectrum`: (F,) real K observations; `noise`: positive scalar K. Returns `(parameters,chi_square)` with float64 (4,) Te/tau/v/sigma and Python float dimensionless sum of squared noise-normalized residuals. Nested residual(parameters) closes over the three inputs and has no independent public API. Uses fixed INITIAL=[5,.5,3,1], lower=[1e-5,0,-inf,1e-12], upper=inf; TRF/3-point/Jacobian scaling and tolerances are specified in approach. A failed optimizer result raises RuntimeError; generic SciPy handles incompatible/nonfinite inputs, with positivity ensured by estimate_noise in standard calls. No I/O. Tests: independent noiseless recovery, original converged maps/losses and forward Jacobian checks.

```python
def reconstruct(cube, frequency):
```

Parameters `cube`: (F,Y,X) floating K observations; `frequency`: (F,) Hz. Returns `(maps,chi_square)` where maps has four OUTPUT_KEYS each float64 (1,Y,X) and chi_square float64 (Y,X), dimensionless. Estimates per-pixel noise once, allocates parameters (4,Y,X), calls fit_spectrum independently in np.ndindex row-major order with fixed initialization. No neighbor warm start or spatial regularization. Noise/fit errors propagate; no file read/write or plot. Standard loader fixes F=501,Y=X=3. Tests: complete nine-pixel original fits and weighted objectives, CLI output and standalone Judge.

### src/visualization.py

```python
def compare_maps(fresh, upstream):
```

Parameters `fresh`,`upstream`: dictionaries with all four (1,3,3) arrays in declared units. Returns 4-by-3 Matplotlib Figure with rows Te,tau,v,sigma and columns original,fresh,signed difference. Each original/fresh pair shares min/max; signed differences use their own color range and retain sign. No alignment, normalization, save/show or backend changes. Missing keys/invalid array shapes follow dictionary/plot errors. Consumer: all-map visual review; it never replaces fresh arrays with reference data.

### evaluation/judge.py

```python
def load_maps(path):
```

Parameter `path`: str/Path NPZ. Requires existing file, four KEYS, exact (1,3,3) shape, real float32/float64 and finite values; allow_pickle=False. Returns dict of original dtype arrays, with same physical units. It validates storage contract, not positivity (evaluate checks physical domains). Missing/invalid data raise ValueError or archive/key errors; no coercion, crop or write. Tests: shape/key/integer/complex/object/NaN/Inf/missing/broken files and valid both precisions.

```python
def relative_max(estimate, reference):
```

Parameters `estimate`,`reference`: matching numeric arrays of the same physical quantity. Returns dimensionless float max(abs((estimate-reference)/reference)); rejects any zero reference with ValueError. No hidden alignment, means or dynamic-range normalization. Normal inputs are storage/finiteness-validated before this helper. No I/O. Test: independent [1.1,3.6] versus [1,4] case gives .1 and zero reference rejects.

```python
def reduced_chi_square(maps, data_path):
```

Parameters `maps`: four declared physical maps (1,3,3); `data_path`: str/Path public NPZ. Independently reads frequency (1,501),brightness (1,501,3,3), validates finite exact shapes, recomputes line-free noise and 15-component transfer with its own constants/equations. Returns float maximum over nine sum(((model-data)/noise)^2)/497. Rejects invalid dimensions, nonpositive/nonfinite noise, Te<=0,tau<0,width<=0 or nonfinite resulting score. It does not import solver, use submitted checkpoints or fit any parameter. Missing raw/config conditions surface as evaluator errors. Consumer: forward observation-consistency threshold and paired notebook chi-square comparisons.

```python
def evaluate(prediction, asset_dir=HERE):
```

Parameters `prediction`: caller-relative str/Path submission; `asset_dir`: evaluator Path default HERE=Path(__file__).resolve().parent. Returns `(report,code)`. Reads positive finite config bounds, truth if present else baseline, requires nonzero reference maps, loads estimate and checks physical domains before scoring. Requires all four maximum relative errors and maximum reduced chi-square pass. Errors produce empty metrics and code 2; valid poor/pass predictions give 1/0. No writes, src imports or submitted code execution. Tests include ground-truth precedence, missing evaluator assets, invalid bounds and all prediction cases.

```python
def main(argv=None):
```

Parameter `argv`: optional CLI string sequence, None reads sys.argv. Requires --prediction and --report. Calls evaluate, creates destination parent, writes strict UTF-8 JSON plus newline. Returns evaluate code or 2 on report failure with Could not write report diagnostic to stderr. All CLI arguments remain caller-relative; default hidden assets resolve beside Judge. Tests: foreign cwd, nested report path and unwritable-directory destination. No I/O occurs merely by importing this module.

## On-disk assets, consumers and scientific test coverage

`data/raw_data.npz`: visible brightness(1,501,3,3)K,
frequency(1,501)Hz, offsets(1,15)km/s,strengths(1,15)dimensionless, allfloat64.
`test/fixtures/upstream_run.npz`: original noise, probe/model/opacity, parameters,
modeled spectra and chi-square; `upstream_jacobian.npz`: independent original
numerical derivative probe. These are consumed only by parity tests.
`evaluation/baseline_reference.npz`: four original physical maps, not truth.
`notebooks/reference_outputs/upstream_measurements.npz`: original parameters
(1,4,3,3),model(1,501,3,3),chi_square(3,3). `cleaned_measurements.npz` has the
same scientific arrays and an earlier runtime, retained byte-frozen; new review
measurements are private. Generated output contains all four maps/report.
Tests cover original forward/noise/loss/Jacobian/maps, physical limits and
noiseless fit, constants, backend behavior and standalone contracts. No generator
is shipped because real observed channels are reused.

All tests/fixtures and the authoritative validation report reside in test/.
The original source checkout, full acquisition files, capture scripts, oracle
environment, journal material and logs are maintainer-only external archives
indexed by test/validation.md. No normal test imports that oracle. Code/data
licenses and attributions are preserved in full in the validation report's
license section; removed task-root LICENSE/NOTICE originals are privately
backed up. Runtime source copyright/SPDX notices still identify the adaptations.

## Standard execution and standalone report semantics

Run `python -m src.main` from this task directory. It writes the exact declared
single NPZ submission described above. Run
`python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`
as a separate process. The Judge never imports solver code or executes a
submitted module. Config/reference assets resolve beside judge.py; explicit
prediction/report paths resolve from caller cwd. A legitimately supplied actual
truth takes priority, but this task ships only a separately identified standard-
method baseline. No output-name guessing, hidden crop, alignment or magnitude
coercion is performed.

The report contains status(pass/fail/error), passed(bool), metrics(dict),
thresholds(each comparison/boundary) and errors(list). Valid below-bound output
has exit1; valid passing output exit0. Malformed prediction, unavailable evaluator
assets, undefined metrics or report-write failure returns2. In error reports
metrics is empty; report-write failure diagnoses stderr. Every required metric
must pass and full-precision finite numeric values are serialized. CLI path
semantics, invalid array/storage/domain cases, bad assets, poor output and
report failures are tested independently of successful reconstruction.

## Notebook lifecycle, reproducibility and semantic review

The active notebook has exactly six second-level scientific sections: Problem,
Data, Physics, Method, Results, Conclusion. Data includes every visible array
shape/dtype/unit and actual loading; Physics defines the acquisition/discrete
conventions; Method explains the actual inverse with all initialization/parameter/
stopping choices; Results actively reconstructs the bound observations and
preserves all original scientific comparison methods, panels and metrics;
Conclusion states measured outcomes and task-specific physical limitations.
Reference arrays remain labeled separately from fresh measurements and truth.
There is no cache-preview reconstruction branch or zero-filled missing method.

Notebook execution is via shared-environment nbconvert with a documented timeout,
private writable TEMP/TMP and shared kernel PATH. New runtime and comparison
evidence are captured under the external revision archive, and the official
notebook retains executed outputs. Packaged reference_outputs bytes are frozen
before/after execution; this prevents wall-time metadata rewriting older hashes.
CLI and notebook have different review side effects but share exactly the local
selected reconstruction. Figures are visually inspected, not merely counted.
Tests, source signatures and actual call roles are checked against this design;
the approach document separately verifies equations, signs, factors, units,
boundary/normalization conventions, inverse updates and metric definitions.
