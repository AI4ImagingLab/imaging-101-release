# Shipped code design and scientific execution contract

## Scope, dependency boundaries and actual caller flow

This document describes shipped callable functions, not a proposed architecture.
All paths are task-workspace-relative unless explicitly called script-relative.
Real arrays are row-major and batch-first on disk. Computation removes only the
documented batch axis. Units are specified per interface; a shape match is not
permission to normalize, transpose or change physical scale. Library imports
do not reconstruct, write files or select a Matplotlib backend.

CLI: raw uint8 NPZ -> load_observations -> center_frames ->
centered_spectrum/find_carrier -> 16 reference_wave/quadrature/phase_score
evaluations -> selected recomputation -> stack real/imag -> compressed output.
Only NumPy is required for this path. main imports preprocessing and solvers;
solvers imports preprocessing/physics_model. visualization imports Matplotlib
only for review. Judge imports only standard library/NumPy and never src.

Notebook Data loads all four observations and identified original reference;
Physics defines the independent intensity expansion; Method documents centering,
finite search and exact settings; Results actively calls reconstruct, writes the
submission, evaluates independent Judge, computes spectrum/search scores and
all original phase/amplitude/error/metric groups. A new fresh runtime is printed
and captured privately, not written over existing reference_outputs JSON.
The notebook does not replace new C with an archived field or call the upstream
oracle. Constant-phase guard and original ROI discrepancy remain documented.

## Function-by-function runtime and evaluator interfaces

The signatures are transcribed from actual source. The following contracts
distinguish explicit validation from caller preconditions; they do not invent
checks that an individual numerical helper does not implement.

### src/main.py

```python
def run(data_path: str | Path='data/raw_data.npz', output_path: str | Path='output/reconstruction.npz') -> dict:
```

Parameters: `data_path` is a str/Path public NPZ input, default data/raw_data.npz; `output_path` is a str/Path submission destination, default output/reconstruction.npz. Both relative paths resolve from caller cwd. Return: a dictionary with carrier tuple (float column,row bins), score int, peak tuple (int column,row bins), candidates int. Reads observations once; calls local reconstruct; creates output parent and writes only `reconstruction`, real float64 (1,2,1024,1024) DN, component real then imaginary. Loader/search/path/write exceptions propagate. The module CLI calls run() then prints JSON; importing main has no reconstruction or backend side effect. Test consumer: full pipeline parity plus standalone Judge/CLI contract tests.

### src/physics_model.py

```python
def centered_spectrum(frame: np.ndarray) -> np.ndarray:
```

Parameter `frame`: 2D (M,N) real intensity or centered intensity in DN. Returns complex128 (M,N) centered unnormalized FFT, same DN-amplitude convention (its finite sums scale with sample count). Computes fftshift(fft2(fftshift(frame))). It does not remove a mean, choose an order or pad. Input dimensional compatibility follows NumPy FFT behavior; this helper adds no separate shape/finiteness checks. No I/O or mutation. Tests: captured spectrum and unnormalized Parseval identity.

```python
def reference_wave(shape: tuple[int, int], fx: float, fy: float, wavelength: float=WAVELENGTH, dx: float=DX, dy: float=DY) -> np.ndarray:
```

Parameters `shape`: (M,N) positive image dimensions; `fx`,`fy`: scalar centered column,row carrier bins; `wavelength`: scalar metres default WAVELENGTH=632.8e-9; `dx`,`dy`: scalar column/row pitches metres/pixel, each default 6.9e-6. Returns complex128 (M,N), dimensionless unit phasor, using x-N/2,y-M/2 coordinates and actual asin/sin arithmetic. Out-of-domain asin raises a math domain error; zero pitch/wavelength or invalid shape propagate arithmetic/grid errors. No intensity calibration, mean subtraction or I/O. Tests: analytic phase slope, SI cancellation and unit norm, including odd toy image dimensions.

```python
def quadrature(centered: np.ndarray, reference: np.ndarray) -> np.ndarray:
```

Parameters `centered`: real (4,M,N) DN frames expected already centered; `reference`: complex (M,N) dimensionless phasor. Returns complex128 (M,N) DN weighted sum R*H0+iR*H1-R*H2-iR*H3 with the exact written multiplication/addition order. This helper does not center, normalize or divide by four. No explicit shape/domain checking; incompatible arrays raise NumPy indexing/broadcast errors. No I/O or input mutation. Tests: analytic DC/twin cancellation, amplitude identity and captured full field.

```python
def interference(object_field: np.ndarray, reference: np.ndarray) -> np.ndarray:
```

Parameters `object_field`,`reference`: same-shaped complex (M,N) optical amplitudes in sqrt(DN). Returns float64 (4,M,N) ideal intensities in DN for reference factors [1,i,-1,-i]. Uses direct squared magnitude. NumPy handles shape incompatibility; no noise, calibration fitting, random state or file write. Only physics tests call this helper; neither CLI nor review substitutes generated images for real observations. Test: independent known complex O*conj(R) quadrature identity.

```python
def phase_score(field: np.ndarray) -> int:
```

Parameter `field`: complex image (M,N) in DN (its nonzero scale does not affect phase). Returns Python int count of min-max-normalized principal phase below 0.1; zero if phase span is exactly zero. It applies np.angle, global min/max and count_nonzero without a mask. Empty/nonfinite arrays are not separately validated here; normal solver input is finite and nonempty. No I/O. Tests: known five-phase array, constant-phase case, exact original bound score.

### src/preprocessing.py

```python
def load_observations(path: str | Path) -> np.ndarray:
```

Parameter `path`: required str/Path NPZ acquisition file. Uses allow_pickle=False; requires key holograms, exact (1,4,1024,1024) uint8. Returns newly cast float64 (4,1024,1024) DN, preserving frame order and values without normalization. Shape/dtype mismatch raises ValueError; missing file/key and malformed archive exceptions propagate. The only side effect is reading that file. Test consumers: test_bad_load, test_preprocessing_and_spectrum, public-contract/constants test.

```python
def center_frames(holograms: np.ndarray) -> np.ndarray:
```

Parameter `holograms`: array-like (4,M,N) finite real DN frames. Converts to float64, explicitly checks ndim=3, first dimension=4 and finiteness. Return float64 array of identical unbatched shape in DN after subtracting each frame's mean over both spatial axes. Invalid dimension/count/nonfinite inputs raise ValueError. No file I/O and no mutation of the original acquisition. Tests: original centered fixture, zero-mean/amplitude identity and bad-frame cases.

### src/solvers.py

```python
def find_carrier(centered_first: np.ndarray, upper: bool=True) -> tuple[int, int]:
```

Parameters `centered_first`: finite float64 (M,N) first centered frame in DN; `upper`: bool default True. Returns `(column,row)` integer Fourier-bin coordinates, adding one to ROI-local indices. Calls centered_spectrum then power argmax in the literal upstream slice. upper=False selects the alternative column-limited region, and is not used by standard reconstruct. No new array/ROI domain validator is added; empty ROI would raise through argmax. No files, plotting or state mutation. Tests: original exact peak, notebook independent strict-upper maximum.

```python
def reconstruct(holograms: np.ndarray, s: float=SEARCH_HALF_WIDTH, steps: int=SEARCH_STEPS) -> tuple[np.ndarray, dict]:
```

Parameters `holograms`: (4,M,N) real DN observations; `s`: nonnegative scalar search half-width in bins, default SEARCH_HALF_WIDTH=1; `steps`: integer samples per axis default SEARCH_STEPS=4. Returns `(field,info)`, field complex128 (M,N) DN, info carrier=(float column,row), score=int, peak=(int column,row), candidates=steps**2. Explicitly rejects steps<s, steps<1 or s<0, then center_frames handles observation validation. np.linspace handles nonintegral step-count errors. Runs x-outer/y-inner exhaustive search, strict-greater tie update and selected recomputation. No I/O, optimizer, cache or plotting. Tests: exact upstream carrier/score/full field, invalid-search case, independent physics identities.

### src/visualization.py

```python
def metrics(estimate: np.ndarray, reference: np.ndarray) -> dict:
```

Parameters `estimate`,`reference`: matching real output arrays, normally (1,2,M,N) DN. Casts float64 and flattens, returning dict ncc_vs_ref,nrmse_vs_ref as full-precision dimensionless floats; no alignment or rescaling. Rejects zero product of norms or constant reference with ValueError. Shape compatibility otherwise follows NumPy dot/subtraction. No I/O or plot. Unlike the Judge's poor-zero convention, this presentation helper rejects a zero estimate; the selected notebook arrays are nonzero. Test: independent two-element metric case agrees with Judge.

```python
def comparison_plot(holograms: np.ndarray, fresh: np.ndarray, upstream: np.ndarray, uncompensated: np.ndarray):
```

Parameters `holograms`: (4,M,N) DN; `fresh`,`upstream`,`uncompensated`: complex (M,N) DN fields. Returns a Matplotlib Figure, 2-by-3 axes plus colorbars: measured first frame; fresh raw phase; fresh compensated phase; original-method compensated phase; fresh amplitude; absolute complex difference. Principal phase uses fixed [-pi,pi] radians, amplitude/error DN. Plot input compatibility follows NumPy/Matplotlib. No save/show/close or backend selection; the notebook owns display and private evidence exports. Tests/consumers: active notebook visual review; metrics remain computed from actual arrays.

### evaluation/judge.py

```python
def score_arrays(estimate: np.ndarray, reference: np.ndarray) -> dict:
```

Parameters `estimate`,`reference`: matching real arrays, contract shape (1,2,1024,1024), DN. Returns full-precision dict ncc_vs_ref,nrmse_vs_ref. Casting/flattering happens here; contract validation happens in evaluate. A zero estimate yields NCC=0 (valid poor result), while constant or zero-norm reference raises ValueError. Dynamic range is over both real channels together, without centering, crop or registration. No I/O. Tests: known analytic metric case and poor/constant-reference contract cases.

```python
def evaluate(prediction: str | Path, assets: str | Path=ASSETS) -> tuple[dict, int]:
```

Parameters `prediction`: str/Path submission archive relative to caller cwd; `assets`: str/Path evaluator directory, default ASSETS=Path(__file__).resolve().parent. Returns `(report,exit_code)` with status/pass/metrics/thresholds/errors. Reads metrics.json and validates finite numeric boundaries, validates file/key/shape (1,2,1024,1024)/real float32-or-float64/finiteness, then chooses ground_truth.npz if present else baseline_reference.npz and validates reference. All metrics must pass. No src import, submitted-code execution or output write. Caught exceptions produce status error, empty metrics, descriptive errors and code 2; valid poor/pass results give 1/0. Tests cover absent/bad config, prediction and reference states.

```python
def main(argv=None) -> int:
```

Parameter `argv`: optional CLI argument sequence, default None reads sys.argv. Requires --prediction and --report. Returns integer 0 pass,1 below-threshold,2 malformed/evaluator/report-write failure. Reads via evaluate, creates report parent, writes UTF-8 strict JSON (allow_nan=False); report failure prints a diagnostic to stderr. Prediction/report paths remain caller-relative and assets remain evaluator-relative. argparse exits conventionally for missing required arguments. Tests: foreign-cwd absolute prediction/report, nested report creation and unwritable-directory destination.

## On-disk assets, consumers and scientific test coverage

`data/raw_data.npz`: sole visible holograms (1,4,1024,1024),uint8,DN.
`test/fixtures/upstream_stages.npz`: original centered frame, centered spectrum,
integer peak, selected carrier/score; consumed by test_parity.py.
`evaluation/baseline_reference.npz`: original two-real-channel cross-term for
Judge/full-pipeline parity, not truth. `notebooks/reference_outputs/` contains
separately identified original/fresh measurement JSON for provenance. The active
notebook reads the original JSON and original array, computes a new field, and
does not refresh packaged JSON. Generated output is reconstruction NPZ plus
explicit Judge report. Tests map to carrier/quadrature/FFT/mean/metric/contracts
as detailed beside each function. No generator is shipped for reused real data.

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


## Current benchmark scoring (2026-10-29)

The reference implementation still runs its historical synthetic-reference
search. Benchmark acceptance instead independently verifies the four-frame
quadrature up to a unit linear-phase gauge with fixed half-array origin.
Unknown reference tilt is not independently calibrated. Finite search grid
frequency, lower-tail score, and optimizer trajectory no longer determine
acceptance. See README and evaluation/metrics.json for exact tolerance.
