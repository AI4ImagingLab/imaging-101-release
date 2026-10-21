# Shipped code design and scientific execution contract

## Scope, dependency boundaries and actual caller flow

This document describes shipped callable functions, not a proposed architecture.
All paths are task-workspace-relative unless explicitly called script-relative.
Real arrays are row-major and batch-first on disk. Computation removes only the
documented batch axis. Units are specified per interface; a shape match is not
permission to normalize, transpose or change physical scale. Library imports
do not reconstruct, write files or select a Matplotlib backend.

CLI: raw NPZ -> batch-preserving load_observations -> vacuum[0]
estimate_carrier -> shared carrier/radius -> two sideband_wave calls -> numerical
vacuum-zero guard -> complex ratio[None] -> compressed standard submission.
main imports preprocessing and solvers; solvers imports physics_model and
SHAPE; physics_model imports PIXEL_SIZE_NM. The CLI uses NumPy only.
visualization adds Matplotlib for maintainer review. Judge depends on NumPy
and standard library, with a separate fixed acquisition calculation.

Notebook Data actively loads both measured exposures; Results actively computes
the selected wave via reconstruct, writes/scorers it and separately calls
sideband_wave for uncorrected and smaller-aperture variants. It loads only the
labeled original reference for comparison, retains all intensity/spectrum and
six wave-comparison panels and the three-method metric table. The smaller-
aperture result is not submitted. New runtime/output evidence stays private;
historical reference_outputs snapshots are not rewritten. The review explains
why cross-term counts differ from electron-wave sqrt(count) amplitudes and why
neither ratio is charge-density inference.

## Function-by-function runtime and evaluator interfaces

The signatures are transcribed from actual source. The following contracts
distinguish explicit validation from caller preconditions; they do not invent
checks that an individual numerical helper does not implement.

### src/main.py

```python
def main() -> None:
```

No parameters; return None. Reads default data/raw_data.npz using local loader, calls reconstruct, creates output/ if needed, writes compressed output/reconstruction.npz key reconstruction, complex128 (1,512,512), dimensionless. Prints the written filename. All paths are caller-relative; run from task root. Read/solver/write exceptions propagate; import does not execute this function or change backend. Consumer: pipeline CLI test verifies exact equality with a separate local reconstruction.

### src/physics_model.py

```python
def radial_frequency(shape: tuple[int, int], pixel_size: float=PIXEL_SIZE_NM) -> np.ndarray:
```

Parameters `shape`: integer row,column lengths (M,N); `pixel_size`: scalar nm/pixel default PIXEL_SIZE_NM=0.9197516441345215. Returns float64 (M,N) nonnegative radial norm of unshifted fftfreq coordinates, in inverse nm. No fftshift, sideband selection or explicit positivity/shape validator; NumPy handles invalid dimensions/zero spacing. No I/O or mutation. Tests: original frozen radial grid at 1e-15 tolerance and DC frequency exactly zero.

```python
def aperture(r: np.ndarray, radius: float, smoothness: float) -> np.ndarray:
```

Parameters `r`: numeric frequency/radius array of arbitrary shape; `radius`,`smoothness`: positive scalar quantities with the same units as r (inverse nm or Fourier pixels as chosen by caller). Returns elementwise float window, shape equal to r, dimensionless, 0.5*(1-tanh((abs(r)-radius)/(.5*smoothness))). Nonpositive scalar radius/smoothness explicitly raises ValueError. No I/O. Tests: center/half-value/outer limits and invalid parameters, plus complete frozen-wave parity.

```python
def sideband_wave(image: np.ndarray, position: tuple[int, int], radius: float, smoothness: float | None=None) -> np.ndarray:
```

Parameters `image`: finite real 2D (M,N) count-valued intensity; `position`: integer (row,column) positive roll in unshifted Fourier bins; `radius`: scalar aperture radius in Fourier pixels; `smoothness`: same units, default None means .05*radius. Returns complex128 (M,N) cross-term in counts, no crop/batch axis. Converts image to float64, explicitly rejects ndim!=2 or nonfinite values, converts radius/smoothness by mean reciprocal image physical extent using PIXEL_SIZE_NM, then normalized FFT, positive roll, tanh aperture and size-restored inverse. Invalid aperture values raise through aperture. No I/O. Tests: both original waves, analytic known-wave phase/sign and constant-intensity rejection.

```python
def interference(object_wave: np.ndarray, reference_wave: np.ndarray, position: tuple[int, int]) -> np.ndarray:
```

Parameters `object_wave`,`reference_wave`: same-shaped complex (M,N) amplitudes in sqrt(count); `position`: integer row,column carrier bins. Returns real (M,N) intensity in counts, |object+reference*exp(2pi*i*(ky*y/M+kx*x/N))|^2. Shapes follow NumPy broadcasting/index behavior; no separate data validator, file I/O, noise or random generation. This is a physics-test helper, never the source of benchmark observations. Test: constant known O,R proves extracted O*conj(R), factor and phase sign.

### src/preprocessing.py

```python
def load_observations(path: str | Path='data/raw_data.npz') -> dict[str, np.ndarray]:
```

Parameter `path`: str/Path default 'data/raw_data.npz'. Reads allow_pickle=False. Requires hologram and vacuum, each real numeric integer/unsigned/float, finite nonnegative exact (1,512,512). Returns dict with both arrays as new float64 count-valued batches, retaining batch axis. Missing keys or invalid data raise ValueError; filesystem/archive errors propagate. No crop, background subtraction, normalization or writer. Tests: all preprocessing rejection modes, full original parity and vacuum self-calibration.

### src/solvers.py

```python
def estimate_carrier(vacuum: np.ndarray) -> tuple[tuple[int, int], float]:
```

Parameter `vacuum`: unbatched real finite (M,N) calibration intensity in counts; standard input (512,512). Returns `(position,radius)`, position=(int row,column) unshifted bins, radius float Fourier pixels. Uses radial_frequency/aperture to exclude center, normalized FFT, first-half-row magnitude argmax, and half nearest-corner distance. The helper assumes valid nonempty finite input; reconstruct/loader enforce those requirements rather than adding a separate validator here. No hidden target input or I/O. Tests: original (122,454) exact position and radius within 1e-12.

```python
def reconstruct(hologram: np.ndarray, vacuum: np.ndarray) -> np.ndarray:
```

Parameters `hologram`,`vacuum`: finite real observation batches exact SHAPE=(1,512,512), in detector counts. Returns complex128 (1,512,512) dimensionless transmission. Explicit shape/finiteness checks raise ValueError; calls estimate_carrier only on vacuum[0], sideband_wave twice with common position/radius/default smoothness; requires abs(reference)>eps*max(abs(reference)) at every pixel before ratio. Undefined vacuum division raises ValueError instead of adding epsilon. Nonnegative real input enforcement belongs to loader; direct helper is not an additional complete dtype validator. No files/plots/iteration. Tests: original ratio, vacuum/vacuum=1, zero-vacuum rejection and CLI parity.

### src/visualization.py

```python
def observation_figure(hologram: np.ndarray, vacuum: np.ndarray):
```

Parameters `hologram`,`vacuum`: count-valued (1,512,512) arrays; displays [0] of each. Returns Figure with two rows, columns measured intensity and log10(1+abs(centered DFT)), respective count and log-amplitude labels. It does not reconstruct, save, display, close or change backend. Array compatibility follows plotting calls. Consumer: active Data/Results review and visual checks; original real intensity contrast is retained in arrays.

```python
def comparison_figure(fresh: np.ndarray, upstream: np.ndarray, uncorrected: np.ndarray, weak_aperture: np.ndarray):
```

Parameters `fresh`,`upstream`: complex (1,512,512) dimensionless waves; `uncorrected`: complex (512,512) count-valued object cross term; `weak_aperture`: complex (512,512) dimensionless smaller-aperture ratio. Returns 2-by-3 Figure: selected amplitude/principal phase, absolute complex original-method error, uncorrected principal phase, smaller-aperture phase, angle(fresh*conj(weak)). Phase radians and ratio amplitudes remain labeled. No file I/O/save/show/backend selection. Consumer: active review preserving selected/uncorrected/bandwidth comparisons, not claiming truth.

### evaluation/judge.py

```python
def relative_l2(estimate: np.ndarray, reference: np.ndarray) -> float:
```

Parameters `estimate`,`reference`: matching numeric arrays, normally complex (1,512,512), identical units. Returns dimensionless float ||estimate-reference||2/||reference||2. Rejects nonfinite or nonpositive reference norm; contract validity of estimate is enforced in evaluate. Uses complex Euclidean norm, so conjugation/global phase error is not discarded. No I/O. Tests: doubling yields one, multiplication by i yields sqrt(2), zero reference raises.

```python
def observation_consistency(prediction: np.ndarray, raw_path: Path) -> float:
```

Parameters `prediction`: dimensionless complex (1,512,512); `raw_path`: Path public NPZ. Reads hologram/vacuum with allow_pickle=False, independently constructs the fixed normalized-frequency tanh aperture, positive roll (122,454), and count-valued inverse waves, then returns relative_L2(prediction[0]*vacuum_wave,object_wave). It does not call/import the reference solver or fit a carrier from prediction. Raw-path/key/shape/arithmetic errors propagate to evaluate; there is no report write here. Consumer: Judge threshold and notebook original/fresh/smaller-aperture metric table.

```python
def evaluate(prediction_path: Path, assets: Path=HERE) -> dict:
```

Parameters `prediction_path`: Path submission archive; `assets`: Path evaluator directory default HERE=Path(__file__).resolve().parent. Returns report dict only (not a tuple). Validates strictly positive finite numeric boundaries, required reconstruction shape (1,512,512), complex64 or complex128, finite values; prioritizes actual truth if present else the shipped baseline and validates its complex shape/finiteness. Computes relative_complex_l2 and sideband_relative_l2, both must pass. Exceptions produce status error, empty metrics and messages. Does not write files, import src or execute submission code. Tests: prediction domain/shape/dtype cases, bad references/config/raw availability and poor/pass waves.

```python
def main() -> int:
```

No Python arguments; argparse reads required --prediction and --report as Paths. Calls evaluate, creates report parent and writes UTF-8 finite JSON plus newline; returns {'pass':0,'fail':1,'error':2}[status]. Write failures print Cannot write report to stderr and return 2. CLI path arguments are caller-relative; hidden assets are script-relative. Tests: foreign-cwd good, poor, absent and report-destination-directory cases. Importing judge does not run CLI.

## On-disk assets, consumers and scientific test coverage

`data/raw_data.npz`: visible hologram and vacuum, each(1,512,512),
int64,counts. `test/fixtures/upstream.npz`: original frequency grid, carrier,
radius and both count-valued complex waves for stage/full-ratio tests.
`evaluation/baseline_reference.npz`: standard complex ratio for Judge, not truth.
`notebooks/reference_outputs/upstream_measurements.npz`: identified original
ratio for notebook; `cleaned_measurements.npz`: byte-frozen earlier fresh ratio
and runtime, retained as historical run evidence. The new review computes a new
ratio and records its runtime privately rather than overwriting this snapshot.
Generated output is reconstruction NPZ/report. Tests map to physical sign,
aperture, DC rejection, vacuum calibration, division guard and contracts as
listed above. No HDF5/domain dependency or generator is in the cleaned task.

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
