# Shipped runtime and review flow

Every task executes independently; no cross-task imports, upstream domain package, checkpoints or remote runtime retrieval. Python3.11/NumPy2.4.6/PyTorch2.9.1/Matplotlib3.11.1 are the validated shared subset. SciPy is used only in private alternate-FFT oracle verification, not the cleaned runtime subset. All functions execute on CPU. Main sets one Torch computation thread; library modules do not change plotting backends or thread settings.

## Standard entry point

`python -m src.main` calls `main() -> None`. It resolves the task root from its own source path, loads `data/raw_data.npz`, calls the complete fresh `reconstruct`, creates `output/` if absent and writes exactly float32 `reconstruction` with shape(1,128,32,32) to `output/reconstruction.npz`. Its only printed diagnostic is actual reconstruction time. Importing main changes neither the Matplotlib backend nor the working directory. No figures or hidden references are needed for this path.

## Callable interfaces

| Module/function | Input/defaults | Return and effects |
|---|---|---|
| preprocessing.load_observations(path) | str or Path NPZ | copied float32(512,32,32), counts/bin; validates exact key, batch shape, finite nonnegative values; no output writes |
| preprocessing.preprocess(counts) | finite nonnegative numeric(512,32,32) | float32(128,32,32) peak-normalized interpolated observations; rejects empty signal; no mutation |
| physics_model.diffusion_kernel(mu_s=262.,thickness=.0254) | positive finite SI coefficients | float64(256,64,64) normalized double-pass circular impulse; complete local series and FFT |
| physics_model.transfer_function(mu_s=262.,thickness=.0254) | same | CPU float32 Tensor(256,64,64,2), last axis real/imag FFT transfer |
| physics_model.multiply_pair(a,b) | broadcast-compatible real/imag tensors | explicit complex-product tensor; no mutation |
| physics_model.convolve(x,adjoint=False) | real CPU Tensor(256,64,64) | same-shape circular forward or exact-transpose diffusion, float32 for standard input |
| physics_model.paired_wiener(x,snr=2500.,mu_s=262.,thickness=.0254) | real padded float32 tensor; positive finite scalar regularization | real float32 deconvolution(256,64,64), original paired arithmetic; no hidden arrays |
| physics_model.frequency_mapping() | none | CPU float32(256,64,64) normalized positive Stolt coordinate |
| solvers.migrate(measurement) | real padded(256,64,64) tensor | nonnegative CPU tensor same shape, squared wave envelope; caller responsible for declared crop |
| solvers.reconstruct(counts,snr=2500.,mu_s=262.,thickness=.0254) | native numeric counts(512,32,32) and explicit optional sensitivity settings | dict of float32(128,32,32): observations, migration_only, signed deconvolution_only, reconstruction; standard suppression only on reconstruction; no files, hidden assets or plotting |

All local arrays use(time-or-depth,x,y), with x/y array order retained from original MAT transpose. Public batch dimension is added only at the NPZ boundary. Kernel constants and grid equations are explicit in physics_model. The reference quantity is a fixed relative response, not probability or absolute reflectance.

## Independent evaluation

`python evaluation/judge.py --prediction <path> --report <path>` resolves the truth file, otherwise method-reference file, and metrics relative to the evaluator's own directory. CLI paths are caller-relative. `load_field(path)` validates exact reconstruction key, shape and real float32/64 data before returning float64; no reshaping or coercion. `measurements(estimate,reference)` computes stable cosine, dynamic-range NRMSE, maximum/range and physical boundary counts. `evaluate(prediction,assets=None)` returns(report,exit-code), with explicit optional assets directory used by isolated contract tests. `main()` writes strict UTF-8 JSON after creating the report parent. Return codes0/1/2 mean pass, below boundaries, input/evaluator error. Failed report writing diagnoses stderr and returns2. The Judge does not import src or execute solver code.

## Maintainer notebook and fixtures

The notebook sets an inline backend explicitly, locates this task, calls fresh cleaned modules and evaluates the fresh full result. Archived upstream measurements are loaded only for clearly labeled comparisons. Active sensitivity and known-geometry experiments are separate calls; no source framework or private archive is imported. No cache-preview shortcut is shipped. The complete actual notebook reconstructs three independent physical point controls, the full bound baseline, three SNR settings, three scattering-coefficient settings and a separately labeled conventional-Wiener diagnostic; all call cleaned modules. A diffusion reblur diagnostic explicitly tests only that stage, not a full albedo likelihood. Standard baseline is saved before these comparisons; sensitivity helpers do not overwrite output/reconstruction.npz.

visualization.metrics(estimate,reference) returns a dict of cosine ncc and dynamic-range nrmse after shape/finite/nondegenerate checks, with float64 arithmetic. It is separately checked against the same known four-value case as the Judge. visualization.export(fig,name,arrays,caption,directory=None) returns the figure unchanged when directory is None; otherwise it writes PNG/SVG/PDF, numerical NPZ source arrays and an incrementally updated figure_manifest.json to the explicitly supplied directory. It validates finite numeric arrays and hashes every export; it never chooses a backend. The notebook optionally reads IMAGING_FIGURE_DIR for a maintainer-only external export path, and otherwise displays figures inline without writing figure files.

All plot functions return Matplotlib Figure objects without writing files: observation_figure(raw) accepts native counts(512,32,32); kernel_figure(kernel) accepts normalized(256,64,64); comparison_figure(result) accepts the four reconstruct dict arrays; slice_figure(field) accepts(128,32,32); parity_figure(fresh,upstream) accepts two such fields; sensitivity_figure(fields,labels,title) accepts the three full fields and their textual labels; controls_figure(fields) accepts three independently controlled fields at .3/.5/.75m. Functions show declared individual/common color scales and nominal coordinates; they do not normalize submitted arrays or invent methods. The notebook's two additional inline diagnostics plot fresh diffusion-stage reblur and the actual paired/conventional Wiener cancellation experiment.

`test/fixtures/upstream_stages.npz` contains frozen independent source preprocessing, full kernel/transfer, full padded deconvolution, migration-only and final results plus sampled frequency coordinates. Source/physics tests consume every key. `physical_point_controls.npz` holds independent analytic forward observations and original source reconstructions for three nominal-depth points. The tests consume observations; the notebook consumes observations and original reconstructions for its numerical-source comparison export. These are not measured object truth. `notebooks/reference_outputs/upstream_measurements.npz` contains the four source comparisons, loaded in the notebook's source comparison section. `fresh_measurements.json` is written by the active notebook and consumed in maintainer validation/reporting; it contains data hash, method/reference identifiers, measured runtime and metrics, rather than a redundant reconstruction array. No original checkout, capture script, paper PDF or original log belongs to the final task directory.
