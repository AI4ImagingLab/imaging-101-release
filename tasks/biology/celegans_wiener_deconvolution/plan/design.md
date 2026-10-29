# Code design: C. elegans effective-PSF Wiener inverse

## Module ownership and dependencies

```text
main.main -> preprocessing.load_data -> solvers.reconstruct
                                       |-> preprocessing.prepare
                                       `-> physics_model.transfer_function/laplacian_kernel
notebook -> same solver + visualization.compare/agreement
         `-> physics_model.apply_blur/laplacian for independent stationarity
judge.py -> stdlib + NumPy only; never src
```

Runtime NumPy FFTs and array arithmetic implement all convolution, centering, objective and inverse steps locally. Matplotlib is an optional visualization consumer imported inside compare; no library function selects a backend. No SDeconv, Torch, skimage, cross-task import, checkpoint or upstream module enters runtime/tests/notebook. src/__init__.py only declares the package. No generator exists because the observation is reused.

## Array and file boundaries

| File | Contents, shape, dtype and units | Producer/consumer |
|---|---|---|
| data/raw_data.npz | observation (1,310,310) float32 a.u.; psf (1,13,13) float32 dimensionless | Immutable processed acquired image and assumed Gaussian, loaded safely by preprocessing |
| output/reconstruction.npz | reconstruction (1,310,310) float64 a.u. | Fresh CLI/notebook output; Judge accepts float32 orfloat64 |
| output/judge_report.json | finite UTF-8 dict,status/passed/metrics/thresholds/errors | Explicit standalone evaluator command |
| evaluation/baseline_reference.npz | Independently reproduced original float32 standard-method result | Hidden Judge reference, not biological truth |
| evaluation/metrics.json | NCC>=0.999999,NRMSE<=2e-6 with formula/provenance | Immutable evaluator configuration |
| notebooks/reference_outputs/upstream.npz | Original method reconstruction on exact observations | Notebook original/fresh comparison |
| test/fixtures/upstream_stages.npz | padded image,embedded kernel,OTF,centered Laplacian,original and mixed-double results | Frozen independent stage/full parity cases |

The public observation has (row,column) axes, zero-based pixel centers, rows down and columns right. There is no inferred micrometer scale. Internal extended arrays are 336×336 and are not submission arrays. Signed output is permitted. Original source/data attributions and complete license terms reside in test/validation.md rather than separate task-level LICENSE/NOTICE files.

## preprocessing.py

### `load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]`

Safely reads observation and psf with allow_pickle=False. Requires fixed shapes SHAPE=(1,310,310) and (1,13,13), all finite, nonnegative values and positive PSF sum; explicit violations raise ValueError. Missing keys, corrupt archives or inaccessible paths raise underlying NPZ/I/O errors. Returns unbatched float64 copies: observation (310,310) a.u.and psf (13,13) dimensionless. It does not normalize PSF yet or rescale image. No mutations or files written.

### `prepare(image: np.ndarray, psf: np.ndarray, pad: int = PAD) -> tuple[np.ndarray, np.ndarray]`

PAD=13. image (Ny,Nx) is floating-convertible a.u.; psf (Ky,Kx) dimensionless. Returns float64 padded image and embedded normalized kernel, both (Ny+2 pad,Nx+2 pad). With pad 0 it copies image instead of reflection. Otherwise np.pad uses reflect excluding endpoint. Kernel starts at int (n/2-k/2)+1 per axis; for default sizes this is 162 and the center 168. PSF is normalized by its sum before assignment. The helper assumes validated nonzero PSF and compatible support; invalid pad/support/shape produces native NumPy errors rather than a bespoke loader check. No I/O or input mutation.

## physics_model.py

### `transfer_function(kernel: np.ndarray) -> np.ndarray`

kernel is a real2D embedded dimensionless kernel, default 336×336. Computes shifts tuple (int (-n/2)), rolls along axes (0,1), and returns complex128 fft2 OTF with the same shape. The generic rule preserves upstream truncation conventions on arbitrary helper sizes. Does not renormalize; prepare owns that step. Invalid dimensionality follows NumPy roll/FFT errors. No files or mutation.

### `laplacian_kernel(shape: tuple[int, int]) -> np.ndarray`

shape is the extended2D grid, default (336,336). Returns a float64 zero array with 4 at (ny//2,nx//2) and-1 at four immediate neighbors. Values are dimensionless under pixel-coordinate derivatives. Requires enough support for neighbors; there is no custom shape validation and tiny shapes can produce indexing/overlap behavior outside the selected case. This centered impulse supplies a DFT whose squared modulus is used in the inverse.

### `apply_blur(image: np.ndarray, otf: np.ndarray, adjoint: bool = False) -> np.ndarray`

image real2D a.u.and matching complex OTF dimensionless produce float64 real2D a.u.by ifft2(fft2(image)*multiplier). multiplier=otf for forward and conj (otf) for adjoint. No padding/cropping occurs here: boundary is circular on the supplied grid. Returns the real part, not magnitude. Shape incompatibilities raise NumPy broadcasting errors; no data validation beyond numerical operations. Used for independent tests/notebook diagnostics, not internally by the direct solver.

### `laplacian(image: np.ndarray) -> np.ndarray`

Receives a real2D array, returns same-shape periodic five-point stencil 4 image minus four np.roll neighbors. Units match image in the pixel-coordinate convention. This separately coded spatial form verifies the inverse normal equation rather than reusing its FFT multiplier. No mutation or I/O; shape and realness are caller preconditions.

## solvers.py

### `reconstruct(image: np.ndarray, psf: np.ndarray, beta: float = BETA, pad: int = PAD, return_full: bool = False) -> np.ndarray`

BETA=0.005,PAD=13. Inputs are unbatched images/kernel from load_data; beta is dimensionless relative squared-Laplacian weight, pad a nonnegative integer. Explicitly raises ValueError for beta<=0 or pad<0. Other invalid types/nonfinite parameters are not comprehensively validated here; native numerical failures may propagate. prepare creates the reflected observation and embedded PSF, then the function explicitly constructs OTF/Laplacian FFT, spectral denominator and inverse. It writes no files and reads no references.

Returns float64(310,310) a.u.for the selected default case. If return_full=True, returns 336×336; if pad==0, returns the full original-size solution regardless of return_full. Cropping is exactly[pad:-pad,pad:-pad]only when pad>0 and return_full=False. No positivity, scale fitting, learned parameter or iteration occurs. The output can be negative. Helper calls support other grid sizes for mathematical tests, but the public benchmark contract remains fixed.

## Entry point and plotting callables

### `main() -> None` in src/main.py

No arguments. Resolves its task root from source location, loads fixed data path, calls reconstruct with defaults, creates output/and writes reconstruction=result[None]in the standard NPZ. Thus paths do not depend on caller cwd. Returns None; does not print, plot or invoke the Judge. I/O/numerical exceptions propagate. Importing main performs no reconstruction/backend configuration.

### `agreement(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]`

Accepts matching nondegenerate real arrays of arbitrary shape, flattens and casts to Python-compatible float arrays, then returns uncentered NCC and dynamic-range NRMSE. It does not provide strict shape/finiteness/reference validation or a zero-norm fallback; caller must use valid review arrays. Degenerate input can issue NumPy warnings/nonfinite results. The independent Judge provides stricter handling; known-vector tests ensure the formulas agree on valid arrays. No files, scaling or alignment.

### `compare(observation: np.ndarray, upstream: np.ndarray, fresh: np.ndarray)`

Three 310×310 arrays in a.u.return a Matplotlib Figure with 4 axes: observation, reproduced original, fresh, fresh-minus-original. The first three share observation 1 st/99.5 th percentile display limits; signed difference has its own labeled colorbar. Axes are hidden for imagery but data orientation is unchanged. pyplot is imported inside this function. Creates figure only, neither saves nor clips arrays. Shape/plotting errors propagate; missing comparison results are never fabricated.

## Standalone evaluation/judge.py

### `read_array(path: Path) -> np.ndarray`

Requires an existing file, reconstruction key, exact (1,310,310) shape, realfloat32/float64dtype and all finite values, using safe NPZ loading. Explicit violations raise ValueError; malformed archives propagate normal loader exceptions to evaluate. Returns a float64 copy, preserving physical scale. Same validator is used for reference and submission. No src imports or solver execution.

### `score(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]`

Input arrays already validated by read_array. Returns NCC/NRMSE over every flattened pixel. Constant dynamic range or zero reference norm raises ValueError. Zero estimate receives NCC 0; no estimate/reference normalization beyond the metric formula occurs. Shapes are assumed matching here. This function does not choose boundaries or write reports.

### `evaluate(prediction: Path) -> tuple[dict, int]`

HERE=Path (__file__).resolve ().parent fixes evaluator assets independently of cwd. Reads metrics, requires finite boundaries with 0<NCC<=1 andNRMSE>0, prefers ground_truth.npz if supplied otherwise baseline_reference.npz, validates both arrays and computes metrics. Returns report and 0 pass/1 valid numerical failure/2 malformed-or-evaluator-error. Report always includes status,passed,metrics,thresholds,errors; exceptions caught before scoring leave empty metrics. It checks measured numbers are finite. Does not write JSON or invoke submitted code. HERE is monkeypatched only by asset-error tests.

### `main() -> int` in Judge

Requires --prediction and --report as Path options; these are caller-relative unless absolute. Calls evaluate, creates report parent and writes UTF-8 JSON with allow_nan=False. Report-write failure prints Cannot write report to stderr and returns 2. The executable guard exits with returned code. No filename guessing, alignment, hidden cropping or replacement prediction exists.

## Actual CLI and review flow

CLI executes only load_data -> reconstruct -> output NPZ. Independent Judge is a second command. Notebook first displays full data/source/units and actually loads inputs, explains physical convolution separately from the inverse, actively runs the same solver with timing, saves submission, loads original comparison arrays, evaluates fresh data and preserves every image/profile/metric group. Full-grid reconstruction is separately requested for normal-equation and flux checks; stationarity is not incorrectly imposed on the cropped image. Reexecution preserves all reference_outputs bytes; fresh outputs/measurements are archived privately.

## Test and fixture mapping

| Test case/group | Actual covered functions or behavior |
|---|---|
| test_frozen_upstream_stages | load_data,prepare,transfer_function,laplacian_kernel,reconstruct versus independently captured original stages |
| test_independent_adjoint_and_flux | prepare,apply_blur and asymmetric-PSF forward/adjoint/DC/flux |
| test_regularized_normal_equation | return_full,apply_blur,independent spatial laplacian; stationary residual and full-grid flux |
| test_pipeline | Standard main and standalone Judge from foreign cwd |
| test_bad_submission | Missing/key/shape/integer/complex/NaN/Inf/object/corrupt predictions,exit 2/empty metrics |
| test_zero_and_negative_predictions_fail | Valid poor arrays measured and return 1 |
| test_exact_reference_passes | Numerical reference comparison returns 0 |
| test_metric_known_cases | Independent NCC/NRMSE case and degenerate reference error |
| test_evaluator_asset_error/test_report_write_failure | Evaluator assets,JSON write failure and stderr/exit 2 |
| test_model_inputs_and_limits | Explicit loader/solver rejections and pad 0 constant reconstruction |
| test_readme_acquisition_constants | Public shape,PSF/padding/precision/modality literals |

The contract intentionally reports actual validation and caller assumptions; it does not promise new checks, parameter selection, hyperparameter search or biological ground truth absent from the shipped code.
