# Shipped execution and interfaces

The cleaned runtime uses NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1. Octave and image/signal packages belong only to the separate private original-code oracle. No original checkout, capture driver or domain library is imported by src, tests, Judge or notebook. No library module selects a Matplotlib backend, including on import of src.main.

## Standard input to output

`preprocessing.load_observation(path: str|Path) -> ndarray` opens allow_pickle=False, requires exactly observation with real float64 shape(1,1080,1440,3), finite values in[0,1], and returns the unmodified batch's(H,W,3) array in RGB normalized JPEG units. It does not resize, linearize gamma or infer calibration.

`main.run(root=None) -> ndarray` resolves its default root from the source location, loads data/raw_data.npz, calls the full reconstruct path, creates output/ and writes output/reconstruction.npz with exact key reconstruction and leading batch dimension. It returns the fresh unclipped field. `python -m src.main` invokes this standard path and prints full shape/range; no plots or hidden array reads occur. The output file is generated and overwritten by this run.

## Explicit mathematics

`physics_model.gradient(image) -> ndarray` takes2D or channel-last numeric arrays, converts to float64 and returns shape image.shape+(2,), ordered column then row gradients, with final outgoing fluxes zero. `divergence(flux)` returns its current-minus-previous finite difference. Its caller must supply the documented zero-outgoing fluxes; gradient and threshold_gradients always do. `laplacian(image)` composes these local operators.

`threshold_gradients(image,h=.033)` validates a finite nonnegative h and returns the same gradient shape, retaining a pixel's two components only when its channelwise magnitude is strictly greater than h. `right_hand_side(image,h=.033,epsilon=1e-8)` returns L div δh∇Y+εY. `stationarity(image,reconstruction,h=.033,epsilon=1e-8)` returns the full Frobenius relative normal-equation residual (one if needed fallback denominator for zero RHS is not used on bound observations); it reads no files.

`solvers.reconstruct(image,h=.033,epsilon=1e-8) -> ndarray` rejects nonfinite arrays, invalid dimensionality and nonpositive/nonfinite/Boolean epsilon. It accepts scalar(H,W) or RGB/general channel-last(H,W,C) images, constructs spatial Neumann eigenvalues, evaluates the explicit cosine closed inverse and returns float64 with unchanged shape. It never clips, loads a cache or trains a model. Only bound loading enforces3channels and native size; generic helpers support independent small physical cases and supplementary native images. There are no iteration counts or stopping rules for this closed-form solver.

## Maintainer review helpers

`generate_data.gaussian_blur(image,sigma)` explicitly constructs/normalizes a finite separable Gaussian kernel with radiusceil(4sigma), uses generic reflected convolution, and returns a new scalar image. `synthetic_mixture(shape=(96,128),alpha=.75,sigma=5.,noise=0.,weak=False,seed=2019)` returns an identified dictionary of observed image, true layers, blurred reflected layer, known transmitted component, target under the mean convention, alpha and sigma. It generates only review cases, never the bound real observation. Scientific arrays are not clipped.

`visualization.image_grid(arrays,titles,columns=3,signed=False) -> Figure` plots supplied arrays with channel samples clipped only for RGB display, printing the exact out-of-range count when present; scalar signed plots use symmetric zero-centered limits. Saved solver arrays are unchanged. `scientific_style()` sets plotting parameters when invoked without choosing a backend.

The active notebook reconstructs the complete public image with local modules, writes the standard prediction, runs the independent Judge, compares original numeric stages/results, computes all sensitivity and synthetic experiments and reconstructs all three supplementary native measured scenes. It loads original numeric results only for separately labeled comparisons and has no cache-preview replacement path. Actual scientific PNG/SVG/PDF/numeric exports are maintainer-only outside the final task. External paper results are not misrepresented as locally run baselines.

## Standalone evaluation

`evaluation/judge.py --prediction ... --report ...` uses only NumPy/standard library, never imports src, and evaluates the exact declared array. Configuration/reference files resolve from its own directory; prediction/report paths resolve from caller cwd. Existing actual ground_truth.npz takes priority if present; the current task has only the explicitly labeled baseline_reference.npz. A malformed existing truth is an error, never a fallback opportunity. Reference/observation arrays, metric boundaries, predictions, keys/dtype/shape/finiteness and nondegenerate reference norm/range are validated before scoring.

Reference metrics are flattened cosine, global dynamic-range RMS and max absolute error. Physical metrics independently assemble endpoint-flux Neumann Laplacians, fixed source threshold and stationarity, with every channel and pixel retained. No crop, fitted gain, magnitude conversion or arbitrary alignment is applied. JSON status/pass flag/metrics/thresholds/errors are UTF8 finite-only. Exit0 is pass,1valid below boundary and2malformed input or evaluator/report-write error; parent report directories are created.

## Tests and private evidence

test/comparing_test consumes frozen original full targets/RHS, complete method output, native h/epsilon comparisons, and standard CLI/Judge pipeline. test/unit_test has independent adjoint/energy/constant/cosine/strict-tie cases, known metrics, strict bad-input tests, valid zero failure, priority/reference/configuration/report-write/foreign-cwd tests. Temporary files use standard-library TemporaryDirectory fixtures, avoiding the machine's inaccessible stale global pytest temp root. No upstream import or self-generated expected fixture is permitted. test/validation.md records archive/source/data hashes, exact command/backend/runtime/status and equation/function/case mapping. Unused original code, papers, MAT capture files, full upstream archives and temporary executed notebooks are outside this task.
