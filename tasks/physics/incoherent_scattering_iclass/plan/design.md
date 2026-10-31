# Shipped implementation and review flow

## Runtime boundary

Run `python -m src.main` from the task directory. The runtime reads only
`data/raw_data.npz` and writes `output/reconstruction.npz`. All three output
arrays are required: `reconstruction` float32, `phase` complex64 and `mtf`
float32, each `(1,400,400)`. Images are dimensionless peak-normalized intensity;
MTF is dimensionless relative amplitude and phase is a dimensionless correction
phasor. Image axes are batch, camera y, camera x; transfer-map axes are batch,
centered vertical frequency, centered horizontal frequency. No checkpoint, training corpus,
hidden reference, upstream package or neighboring task is loaded.

## Call graph and interfaces

| Callable | Inputs and defaults | Return and effects |
|---|---|---|
| `preprocessing.load_observations(path='data/raw_data.npz')` | Path to public NPZ; `camera_frames` float32 `(1,150,400,400)` in resampled camera DN | Contiguous `(150,400,400)` NumPy array retaining its loaded backing allocation; rejects absent key, wrong shape/dtype or nonfinite entries |
| `preprocessing.prepare_matrix(frames, device='cpu')` | Unbatched float32 frame array; device string | Complex64 Torch matrix `(160000,150)` and float32 RMS `(400,400)`; copies input, sequential y/x symmetric Hann, temporal centering and spatial FFT |
| `physics_model.circular_convolution(a,b)` | Complex Torch arrays with FFT-compatible broadcast dimensions | Circular 2D FFT convolution with extra division by product of transformed dimensions |
| `physics_model.circular_correlation(a,b)` | Same; second operand may have one column | Circular 2D correlation, conjugating spectrum of second operand; same extra scale |
| `physics_model.phase_direction(matrix)` | Complex `(N,M)` Fourier/frame matrix | Complex `(N,)` direction before unit-modulus projection; no dense covariance allocation |
| `physics_model.spatial_frames(matrix,shape)` | Complex `(H*W,M)`; `(H,W)` | Complex `(M,H,W)` physical spatial frames after restoring row packing and inverse FFT; rejects inconsistent row count |
| `physics_model.estimate_mtf(matrix)` | Complex Fourier/frame matrix | Real `(N,)` relative MTF; rejects zero/nonfinite row power |
| `physics_model.rms_object(matrix,shape,mtf=None,sigma=0.01)` | Complex matrix, physical shape; optional relative MTF vector; positive dimensionless sigma | Real peak-normalized `(H,W)` image; rejects nonpositive sigma or degenerate image; applies MTF inverse to individual spectra before RMS |
| `physics_model.covariance_coherence(matrix)` | Complex `(N,M)` | Python float diagonal covariance energy divided by squared trace; normalizes before fourth-order accumulation to avoid overflow |
| `physics_model.incoherent_forward(object_image,illumination,psf)` | Broadcast-compatible NumPy arrays `(...,H,W)` | Real periodic convolution `PSF*(object*illumination)`; independent physical-test helper, absent from standard inverse path |
| `solvers.solve(frames,iterations=250,sigma=0.01,device=None,record_steps=())` | All raw observations; positive iteration count and sigma; auto-select CUDA when available | `(result,details)` dictionaries; modifies Torch process thread count to six, does not mutate frames or write files |
| `main.run(data_path='data/raw_data.npz',output_path='output/reconstruction.npz',device=None)` | Paths resolved from caller working directory | Loads observations, calls `solve`, creates output parent, writes compressed NPZ; returns both dictionaries |
| `main.main()` | CLI flags `--data`, `--output`, `--device {cpu,cuda}` | Calls `run`, prints scalar diagnostics; does not select a Matplotlib backend |

`solve` prepares the matrix on CPU, matching the oracle FFT backend placement,
then moves it to the requested device. It estimates MTF once: phase multiplication
preserves row power. Each of 250 updates computes the local convolution/correlation
direction, multiplies the cumulative phase and matrix by its unit phasor, and
optionally saves requested phase snapshots. Final image formation uses physical
2D inverse transforms, rather than the upstream flattened-axis output bug.
The complete derivation, scales and deviations are in `approach.md`.

`result` contains only the three submission arrays. `details` contains
`runtime_seconds`, `iterations`, `sigma`, `device`, `initial_coherence`,
`final_coherence`, `phase_only` `(1,H,W)`, `raw_rms` `(1,H,W)` and `snapshots`
(integer step to unbatched flattened complex phase). CUDA is synchronized before
timing stops. Snapshot arrays are CPU-owned; standard execution requests none.
No per-iteration history or square 160000-by-160000 reflection matrix is stored.

## Packing, precision and memory

Camera frames are float32. CPU FFT produces complex64 and the iterative kernel
retains complex64 on CPU or GPU. Matrix row `ix*H+iy` corresponds to spatial
Fourier pixel `(iy,ix)`; public maps use `reshape(W,H).T`. `fftshift` follows the
original preprocessing on all three dimensions, including illumination order;
inverse image formation shifts only the two spatial axes. An illumination-axis
permutation does not change the RMS or row powers. The even 400-pixel grid has
centered zero frequency at index 200. No pixel pitch was supplied, so no invented
metric spatial calibration is attached to these arrays.

The independent Judge uses float64 SciPy transforms. It does not share solver
implementations. Full GPU reconstruction used about 1.55 GB peak allocation;
CPU execution is supported and validated separately. CUDA is an acceleration
choice, not an output or physics requirement.

## Independent evaluator

`evaluation/judge.py` imports NumPy and SciPy only. `evaluate(prediction_path,
evaluation_dir=HERE,raw_path=None)` returns a JSON-compatible dictionary.
References/configuration resolve relative to the evaluator directory; default
observations resolve to its parent `data/raw_data.npz`. Prediction/report CLI
paths resolve from the caller. `observation_metrics(prediction,raw_path)`
independently recomputes Hann/centering, empirical MTF, S18 image consistency and
covariance coherence. `aligned_metrics(image,reference)` fixes the integer
circular translation maximizing correlation, with no amplitude fitting.

Malformed submissions, missing assets, nonfinite numbers, negative intensity or
MTF, wrong keys/shapes/dtypes and nonunit phase give status `error`, empty metrics
and exit 2. Valid poor submissions give status `fail` and exit 1; all five
thresholds must pass for status `pass` and exit 0. Report directories are created;
write failure emits stderr and exit 2. No submitted code is imported. Hidden
`baseline_reference.npz` is explicitly a standard-method reconstruction, not
ground truth. The fixed sigma and physical consistency checks connect image,
phase and MTF, preventing unrelated image/transfer-map submissions.

## Notebook-only comparisons and plotting

The notebook loads raw observations and actively calls `main.run`, then calls
the independent evaluator on that fresh file. It separately loads
`reference_outputs/upstream_legacy.npz` (unaltered upstream output plus independent
widefield) and `upstream_paper_corrected.npz` (original phase with corrected output
formation). These files have no role in the solver. The widefield reference is
an independent measurement with different exposure/illumination and is never
named truth. Circular alignment and individual peak normalization are explicitly
limited to descriptive cross-acquisition comparisons.

`visualization.align_circular(image,reference)` returns shifted image and signed
integer shift. `comparison_metrics(image,reference)` reports NCC, range-NRMSE,
shift and percentile contrast after explicit peak normalization;
`review_figure(images,titles)` returns a 2-by-3 peak-normalized comparison figure;
`transfer_figure(phase,mtf)` returns correction angle, relative MTF and inferred
absolute PSF. The inferred PSF is a model visualization, not a separately measured
calibration. Plotting helpers do not write files or change the backend.

## Verification ownership

`test/unit_test/` covers mathematical finite cases, dense covariance equivalence,
physical intensity conservation, input immutability, malformed Judge contracts,
and metric conventions. `test/comparing_test/` consumes frozen independent
upstream operators, real-data preprocessing, first-three phase updates, and final
250-update reference output. Frozen files have explicit upstream consumers and
hashes in `test/validation.md`. No test imports the archived upstream checkout.
The notebook is the active scientific review, while the CLI and Judge establish
the released input/output boundary. Runtime logs, original HDF5 files, publication
assets, capture scripts and environment records remain in the private archive.


## Current benchmark v2 acceptance

The v2 Judge retains independently evaluated image/MTF consistency and the already frozen covariance coherence threshold6.906686629445275, replacing reference-image similarity. No ground truth exists. A distinct smooth-phase candidate with its freshly computed consistent image passes; a random-phase self-consistent image fails covariance quality.31 unit tests passed. Phase-update250 iterations no longer define acceptance. See iclass_pytest.log.

Source reconstruction and historical comparison computations remain unchanged; the standalone Judge is independent and uses the current README/metrics contract.
