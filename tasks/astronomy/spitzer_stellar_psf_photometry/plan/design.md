# Shipped code design and scientific execution contract

## Scope, dependency boundaries and actual caller flow

This document describes shipped callable functions, not a proposed architecture.
All paths are task-workspace-relative unless explicitly called script-relative.
Real arrays are row-major and batch-first on disk. Computation removes only the
documented batch axis. Units are specified per interface; a shape match is not
permission to normalize, transpose or change physical scale. Library imports
do not reconstruct, write files or select a Matplotlib backend.

CLI: raw NPZ -> load_data -> prepare_psf -> make_interpolator ->
for each patch initialize -> bounded least_squares calling residual/model ->
three physical arrays -> compressed standard submission. main imports local
preprocessing/solvers; solvers imports local model/preprocessing and generic
SciPy optimizer; model/preprocessing import generic SciPy interpolation and
NumPy. visualization adds Matplotlib. Judge imports its own generic interpolation
and NumPy, never src or a submitted module. Photutils/Astropy are absent from
all cleaned runtime paths and needed only by the independent archived oracle.

Notebook Data loads real patches and independent flight calibration; Physics
explains angular resampling, sum16 and model amplitude units; Method explains
initial peak/border fit, exact TRF settings and support convention. Results
actively reconstructs and writes the submission, evaluates independent Judge,
compares original/fresh parameter tables, shows both PSF samplings, all four
observed/original/fresh/residual rows and all three initial/original/fresh bar
groups. New runtime is printed/captured privately; no reference snapshot write
occurs. The forward residual remains about10.8%, with model mismatch disclosed.

## Function-by-function runtime and evaluator interfaces

The signatures are transcribed from actual source. The following contracts
distinguish explicit validation from caller preconditions; they do not invent
checks that an individual numerical helper does not implement.

### src/main.py

```python
def main():
```

No parameters; return None. Reads caller-relative data/raw_data.npz, calls reconstruct, creates output/ and writes compressed output/reconstruction.npz with positions,flux,background. All arrays float64, shapes (1,4,2),(1,4),(1,4); units pixel,MJy/sr pixel,MJy/sr. No plotting or filename print; read/solver/write errors propagate. Import does not run this function or change backend. Consumer: standard CLI/foreign-cwd Judge pipeline test.

### src/physics_model.py

```python
def make_interpolator(psf):
```

Parameter `psf`: floating (81,81) dimensionless effective calibration weights, normalized by preprocessing. Returns scipy.interpolate.RectBivariateSpline with x,y coordinate arrays 0..80, psf.T, kx=ky=3,s=0. It does not normalize, infer the center or read calibration data. Invalid shape/grid errors propagate from SciPy. No I/O/state mutation. Consumers: model/residual, original model parity, derivative/translation/identifiability tests.

```python
def model(parameters, interpolator, shape=(17, 17)):
```

Parameters `parameters`: (4,) [F,x0,y0,sky] in MJy/sr pixel,pixel,pixel,MJy/sr; `interpolator`: calibrated x,y cubic spline; `shape`: (rows,columns) integer tuple default (17,17). Returns float64 image of that shape in MJy/sr. Evaluates xi=4*(x-x0)+40,yi=4*(y-y0)+40, overwrites out-of-grid samples with zero before amplitude scaling and adds sky everywhere. No extra pixel integration, background fit, rescaling or validation of physical domains; shape/broadcast/spline errors propagate. No I/O. Tests: original images, linearity/translation/sky, exact-source recovery and objective derivative.

```python
def residual(parameters, interpolator, patch):
```

Parameters `parameters`,`interpolator` as model; `patch`: floating (H,W) observed MJy/sr image. Returns float64 flattened (H*W,) model(parameters,shape=patch.shape)-patch in row-major order, MJy/sr, without weights or normalization. No file I/O or separate validator. Called by generic least_squares and finite-difference gradient test. The model-data sign is retained; SciPy minimizes half the squared norm.

### src/preprocessing.py

```python
def load_data(path):
```

Parameter `path`: required str/Path NPZ. With allow_pickle=False reads images and flight_psf, requiring shapes (1,4,17,17),(1,81,81) and all finite values. Returns newly cast float64 images (4,17,17) in MJy/sr and flight calibration (81,81) dimensionless weights. Shape/nonfinite invalid input raises ValueError; missing keys/file and conversion errors propagate. No sky subtraction or PSF resampling occurs until later functions. Only side effect is input read. Consumers: all physical/original parity tests and CLI/notebook.

```python
def prepare_psf(flight):
```

Parameter `flight`: real float (81,81) independent dimensionless calibration. Returns float64 (81,81) effective calibration at 0.3 arcsec/sample, sum 16. Uses PIXEL_ARCSEC=1.2,FLIGHT_PIXEL_ARCSEC=.30325,OVERSAMPLING=4,center40 and exact cubic interpolation through flight.T in x,y order. No input mutation or I/O. This helper assumes valid finite calibration with nonzero resampled sum; it does not add a separate zero-sum check or silently substitute a Gaussian. Interpolation/normalization errors follow SciPy/NumPy. Tests: frozen original calibration and sum/physical flux checks.

```python
def initialize(patch):
```

Parameter `patch`: finite floating (17,17) observed MJy/sr. Returns float64 (4,) in [F,x,y,sky] order, units MJy/sr pixel,pixel,pixel,MJy/sr. Uses row-major argmax, unique-border median and positive excess sum. No parameters/defaults, mutation, file reads or target data. Empty/incompatible inputs are not explicitly validated here; normal inputs come from load_data. Tests: analytic constant-plus-impulse patch and exact original initial arrays.

### src/solvers.py

```python
def reconstruct(images, flight_psf):
```

Parameters `images`: float64 (K,17,17) observed stack, standard K=4; `flight_psf`: (81,81) independent dimensionless calibration. Returns `(submission,fitted)` with positions (1,K,2) pixel,flux (1,K) MJy/sr pixel,background (1,K) MJy/sr, all float64, and fitted (K,17,17) MJy/sr. Calls prepare_psf/make_interpolator once; each patch initialize then bounded TRF with exact parameters in approach. Failure success=False raises RuntimeError; lower level errors propagate. No files, plots, catalog inputs or calibration updates. Tests: frozen stages/full fit, analytic exact-source recovery and CLI output.

### src/visualization.py

```python
def compare(images, upstream, fresh):
```

Parameters `images`,`upstream`,`fresh`: real (4,17,17) MJy/sr stacks. Returns Figure with four source rows and observed/original/fresh/data-fresh columns. Original/fresh/observation share each source's observed min/max; signed residual uses a symmetric range. Origin lower means increasing row is up in this display; arrays keep row,column order. There is no fit, alignment, masking, save/show or backend selection. Plot/index errors propagate for incompatible inputs. Consumer: complete active original/fresh scientific review.

### evaluation/judge.py

```python
def load_arrays(path):
```

Parameter `path`: str/Path NPZ. Requires positions (1,4,2),flux/background (1,4), all exactly float64, finite; rejects negative amplitudes or positions outside [0,16]. Sky has no positivity bound. Returns dict with the three arrays, no dtype cast/reshape/crop. Missing keys/file/corruption/object payloads and shape/domain errors propagate to evaluate. No report write. Tests: all eight malformed types, corrupt/object archive, valid reference and poor positive-amplitude outputs.

```python
def observation_error(arrays, raw_path):
```

Parameters `arrays`: validated three physical output arrays; `raw_path`: public NPZ Path. Reads images/calibration allow_pickle=False, independently resamples flight at .3/.30325, normalizes to16, constructs cubic response and models all four patches with zero external support. Returns float global ||models-images||2/||images||2, dimensionless. Does not call/import src, fit sky/amplitude/position or normalize to a reference result. Raw errors propagate; evaluate requires finite final metric. Consumer: notebook's independently computed fresh residual norm and forward-consistency gate.

```python
def evaluate(prediction, assets=ASSETS):
```

Parameters `prediction`: caller-relative str/Path NPZ; `assets`: evaluator Path default ASSETS=Path(__file__).resolve().parent. Returns `(report,code)`; requires finite positive boundaries for position_error,flux_relative_error,background_absolute_error,observation_relative_l2. Chooses truth if actually present else baseline; checks reference arrays and strictly positive reference flux, validates estimate, computes all four metrics. No alignment, assignment, file write, solver import or submitted code execution. Catches errors to empty metrics/status error/code2; valid poor/pass give1/0. Tests: analytic known metrics, malformed/poor data, missing config/assets, invalid bounds and CLI paths.

```python
def main():
```

No Python arguments; argparse reads required --prediction/--report. Calls evaluate(Path(prediction)), creates report parent, writes UTF-8 strict JSON (allow_nan=False), returns its0/1/2 code. Write failure prints Report write failed to stderr and returns2. Caller-relative CLI paths are independent of script-relative hidden assets. Tests: foreign-cwd CLI, nested report path, destination-directory write failure. No side effects during module import.

## On-disk assets, consumers and scientific test coverage

`data/raw_data.npz`: visible images(1,4,17,17)MJy/sr and
flight_psf(1,81,81)dimensionless,float64. `test/fixtures/upstream.npz`:
original resampled calibration, initial/final parameters and fitted images,
consumed by stage/full fit tests. `evaluation/baseline_reference.npz`: original
three parameter arrays for Judge, not truth. `notebooks/reference_outputs/
upstream.npz`: identified original parameters/images for review; `cleaned.npz`:
earlier measured local parameter snapshot. Both remain byte-frozen; notebook
actively recomputes and does not load cleaned.npz as a fresh result. Generated
output is the three-array NPZ/report. Tests map to calibration, initialization,
linear response/translation, known-source identifiability, objective derivatives,
known metrics and malformed/error contracts. No generator is needed.

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
