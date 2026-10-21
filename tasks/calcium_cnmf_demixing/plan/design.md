# Shipped interfaces and execution flow

## Module responsibilities and dependencies

`src/main.py` is the standard command-line entry point. It accepts `--data` (default `data/raw_data.npz`) and `--output` (default `output/reconstruction.npz`), relative to the caller's workspace. It loads observations, invokes the full reconstruction once, creates the output parent directory, writes exactly the eight declared NPZ keys and prints source count and measured runtime. It has no plotting backend side effect on import. There is no root-level wrapper and no generator, because the task reuses complete existing real observations.

`src/preprocessing.py` validates observations and implements all selected 2D NoRMCorre mathematics: localized DFT registration, constrained displacement search, image shifting, window grid construction, field interpolation, template construction and both full-movie motion passes. OpenCV supplies DFT, correlation, interpolation and resampling primitives. It does not supply NoRMCorre or any inverse imaging method.

`src/physics_model.py` implements pixel spectral noise, AR autocovariance estimation, greedy independent-set ordering and original streaming dot-product arithmetic. It also exposes an independently tested fluorescence forward/adjoint pair; these helpers describe the acquisition model and do not bypass fitting. NumPy/SciPy supply transforms and general linear algebra.

`src/solvers.py` contains the extracted explicit selected inverse mathematics. Its sections implement NNDSVDar/background coordinate descent, greedy ROI/HALS, positive LARS, support morphology and spatial update, pure-Python OASIS pools, temporal Gram/residual updates, patch embedding/background ALS, source merging, the actual global fit, classifier-free quality and diffusion-KDE relative fluorescence. Concatenating these related stages into one module avoids upstream/private imports while preserving readable named stage functions. All callable definitions are listed in the checked signature appendix.

`src/visualization.py` provides source comparisons and common metric calculation for maintainer figures. It imports Matplotlib for plotting only and never calls `matplotlib.use()`. The Judge independently implements scoring and imports no `src` module.

The task runtime imports only the standard library, NumPy, SciPy, OpenCV and Matplotlib. Requirements pin their shared-environment versions. No CaImAn, compiled OASIS, scikit-learn, peakutils, skimage, TIFF reader, HDF5 reader, pretrained model, upstream checkout or cross-task module is imported anywhere in shipped source, tests, Judge or notebook. Original TIFF decoding and HDF5 stage capture happened only in the private oracle archive.

## Standard call and data flow

1. `main(argv=None)` parses public paths and calls `load_movie(path)`. The loader accepts exactly `movie`, native float32 `(1,3000,170,170)`, finite values. It returns `(3000,170,170)` without altering intensities.
2. `reconstruct(movie, progress=None)` sets the selected deterministic NumPy initialization seed 0, starts its runtime clock and calls `motion_correct_movie(movie)`. The input is never replaced by a hidden template or cached movie.
3. Motion returns float32 `corrected`, two float32 image templates, rigid correction shifts, piecewise correction shifts and scalar bias. Temporary complete corrected movie occupies about 347 MB. Full `Y` matrix is a column-major spatial view where possible; NumPy may make explicit copies for algorithms requiring contiguous storage.
4. `fit(corrected, progress, checkpoint=None)` calls `initialize_patches`. Each of 49 patches estimates noise, initializes four source factors and one background, computes source-projected residual, normalizes and embeds into the full image. Rank-two global background compression and 100 ALS cycles follow.
5. `merge` runs until no candidate groups remain, then `update_temporal(...,p=1,streamed=True)` and `normalize` produce the initial global result. Optional `checkpoint('initial',r)` is called only if a maintainer supplies it; the standard CLI does not write checkpoint files.
6. The first global pass calls `update_spatial(...,streamed=True)`, `update_temporal(...,p=0,streamed=True)`, then `merge(...,p=0,mx=50)`. The second calls spatial update and AR1 temporal update. `normalize` fixes spatial scale. Optional `checkpoint('refit',r)` identifies the completed global method before source selection.
7. `quality(corrected,r,corrected_mode=False)` evaluates the released non-CNN branch. It returns accepted indices, SNR and spatial correlations for all fitted sources. `dff` computes fresh relative fluorescence on accepted sources only. The alternate `corrected_mode=True` is a diagnostic quality comparison on the same factors, not an alternate reconstruction algorithm or source initializer.
8. `reconstruct` returns `(output,review)`. `main` saves `output`; the notebook consumes `review` for scientific diagnostics and saves an identified fresh review array file. No source writes reports or reference arrays implicitly.

## Array contracts, mutations and return values

| Interface/group | Input shapes/dtypes and units | Return values and effects |
| --- | --- | --- |
| `load_movie` | path-like public NPZ | float32 `(T,H,W)` counts; raises on extra/missing keys, dtype, shape or nonfinite values |
| `register_translation` | two real 2D images; integer/subpixel support in pixels | measured row/column displacement, complex source transform, global phase; allocates transforms, no image mutation |
| `_upsampled_dft` | complex DFT array, local region and offsets | explicit matrix-DFT subpixel samples; no registration library |
| `apply_shift_iteration` | real image, correction shift in row/column pixels | shifted/clipped image; cubic OpenCV affine warp and chosen boundary rule |
| `get_patch_edges/get_patch_centers/sliding_window*` | image dimensions, overlap and stride tuples | grid edges/centers and deterministic local image views; row-major window enumeration |
| `tile_and_correct` | real frame/template plus support, bias and warp controls | `(corrected, shifts, coordinates, grid)`; supports the selected 2D OpenCV remap branch; returns pre-final-bias image |
| `apply_pw_shifts_remap_2d` | image, two local shift fields, patch centers | full corrected image via explicit displacement-map construction and OpenCV cubic remap |
| `_TemplateMovie` | private ndarray subclass holding sampled frames | only template-initialization math; its `apply_shifts` mutates that private sample array, never the public movie |
| `motion_correct_movie` | native float32 `(T,H,W)` counts | dict: corrected `(T,H,W)`, template images `(H,W)`, rigid `(T,2)`, shifts `(T,9,2)`, bias; optional progress callback `(completed,T)` |
| `get_noise_fft/mean_psd` | pixel/trace-by-time arrays in counts; periodograms count-squared | noise RMS per leading trace and selected one-sided PSD; helper default log-mean, selected pipeline explicitly mean |
| `estimate_parameters/estimate_time_constant` | 1D trace counts, selected AR order and noise options | AR roots/coefficient and noise RMS; unstable-root fallback changes NumPy RNG seed to original 45 |
| `axcov/nextpow2` | 1D trace or scalar FFT length | autocovariances or next power-of-two exponent |
| `blocked_projection` | `Y(d,T)` float32; dense/sparse `B(d,n)` if transposed or `(T,n)` otherwise | float32 `(T,n)` transposed or `(d,n)` direct product; 19 spatial blocks, casts B and each accumulation |
| `fluorescence_forward/fluorescence_adjoint` | `A(d,K), C(K,T), b(d,2), f(2,T)` or residual `(d,T)` | full counts movie or spatial gradients; general acquisition helper, no solver side effects |
| `NMF.fit_transform/fit` | nonnegative dense 2D `X` | W or self; sets `components_`, `n_iter_`; integer random state is restored externally after its two internal seed resets |
| `randomized_svd/initialize_background/coordinate_update` | dense matrix and rank; W/H factors | thin SVD, NNDSVDar factors, or projected-gradient violation; coordinate update mutates W |
| `initialize_components` | time-last patch `(h,w,T)` float32 counts; selected K4, nb1 | sparse A `(h*w,4)`, C `(4,T)`, dense b `(h*w,1)`, f `(1,T)`, centers `(4,2)`; default nb2 remains for independent original API fixture |
| `greedyROI/finetune/imblur/hals` | normalized time-last patch and factor estimates | explicit greedy ROI factors, rank-one factors, blurred movie or HALS factors; helpers may mutate their local working factor arrays |
| `downscale/resize` | normalized movie or factor, explicit spatial/time sizes | original OpenCV area or linear interpolation; initializer-only resizing, not a change to benchmark data |
| `positive_lars` | float64 design `(T,n)`, response `(T)`, alpha | float64 nonnegative coefficient vector `(n)`; local standardized centered working arrays; up to 500 finite path steps |
| `threshold_components_parallel` | packed tuple with sparse column, dimensions/morphology settings | thresholded sparse row and index; despite historical name, executed serially here |
| `HALS4shape_bckgrnd` | residual-times-background projection `(d,nb)`, B `(d,nb)`, F `(nb,T)`, supported pixel lists | updated B, mutates supported background pixels for five sweeps by default |
| `update_spatial` | Y `(d,T)`, A sparse `(d,K)`, C `(K,T)`, b `(d,nb)`, f `(nb,T)`, sn `(d)`, dims `(H,W)` | updated sparse A, dense b, C and f; copies C, background b/f working arrays can be changed; empty components removed |
| `_Pool/constrained_oasisAR1` | 1D float32 y counts, g and sn; selected optimized baseline | calcium, innovations `(T)`, baseline, g and dual multiplier; pool value/weight explicitly float32; selected decimate1, optimize_g0, penalty1 only |
| `deconvolve_trace` | 1D trace, p0 or p1 | baseline-inclusive C, S, baseline, initial calcium, noise, g, dual multiplier; preserves subtract/add initial-condition rounding |
| `update_temporal` | full Y, A/b, C/f, AR order | dict with C, sparse A, b/f, S, baseline, initial calcium, neuron noise, object-array g, projected residual and dual multiplier; at most two sweeps |
| `normalize` | mutable stage-result dict | same dict with unit spatial columns and compensated temporal/S/residual/baseline scale; float32 epsilon prevents division by zero |
| `patches/initialize_patches` | dimensions or corrected movie | patch index/shape iterator or stitched stage dict with A/C/YrA/b/f/sn; no source pruning shortcut |
| `merge` | mutable stage dict, p and max group count | `(dict, merged_group_count)`; removes grouped rows and appends fitted merged factors; explicitly rejects unverified fixed-g merge branch if reached |
| `fit` | corrected full movie and optional callbacks | full refit stage dict before source quality selection; callback stage names initial/refit; no output file writes |
| `quality` | corrected movie and full stage dict | accepted indices int64, all-source SNR and spatial Pearson arrays; optional corrected mode changes only quality diagnostic |
| `release_compatibility_mode` | 1D or axis-selected real trace | original released recursive heuristic including identified three-point asymmetry; no claim of ideal symmetric HSM |
| `symmetric_half_sample_mode` | 1D real trace | independent corrected shortest-half interval mode; diagnostic only in standard reconstruction |
| `peak_indexes/find_activity_intervals` | 1D trace or C `(K,T)` | detected peak indices or per-source frame sets; explicit plateau handling and clipping |
| `compute_event_exceptionality/evaluate_components/classify_components_ep` | full movie or pixel matrix plus selected factor groups | log-tail event scores, projected-noise statistics and spatial Pearson values; baseline subtraction mutates the passed copied trace array only |
| `df_percentile/kde/fixed_point` | trace samples, grid settings or bandwidth root arguments | adaptive baseline percentile/value, density/CDF/bandwidth or scalar root function; no sklearn density model |
| `dff` | accepted A/C/YrA and shared b/f | float64 `(K,T)` dimensionless relative fluorescence; no input file access |
| `reconstruct` | `(T,H,W)` native movie, optional callback | `(output,review)`; changes NumPy RNG state, retains all benchmark frames; no filesystem side effects |

Public outputs are always explicitly converted to float64. Source arrays share accepted index order. A is reshaped from column-major pixel vectors into `(1,K,H,W)`. Background similarly becomes `(1,2,H,W)`. Trace arrays become `(1,K,T)`; shifts become `(1,T,2)` and `(1,T,9,2)`. The reference solver's CPU arithmetic has a mix of float32 recurrence/streaming and float64 regression/linear algebra; replacing everything with one precision would change parity.

`review` contains all preselection footprints `(d,149)`, all C and projected residual `(149,T)`, background factors, the fresh corrected movie, accepted indices, SNR, spatial correlation, mean corrected image and measured wall time. Review arrays are maintainer-only and are not public solver inputs. The notebook builds a stage dictionary from these in-memory arrays for its corrected-mode comparison. It never saves the full corrected movie or redundant copies of the submission as extra reference files.

## Judge and tests

`evaluation/judge.py` resolves its baseline and metric files relative to its own file, validates all eight output keys and shape relationships, real floating dtype, finite values and spatial gauge before scoring. Output paths and report paths remain relative to caller cwd. It uses independent SciPy Hungarian assignment, independent scalar metric routines, and blockwise full-background products. There is no import of `src`, no fallback to running a submitted solver and no hidden crop/scale/shape coercion. Error/pass/fail statuses use exit codes 2/0/1. Report-write errors are diagnosed on stderr and return 2.

All tests reside in `test/`. `unit_test` covers acquisition adjoints, noise units, AR recurrence, feasible noise budget, scale normalization, motion direction and metric/Judge contracts. `comparing_test` consumes original frozen numerical fixtures, including actual pixel regressions, both temporal orders and global rank-two NMF; the full pipeline test runs `python -m src.main` on all 3000 frames and invokes the Judge from a foreign cwd. No normal test imports upstream libraries. Selected single-thread BLAS conventions are specified by the shared environment and applied explicitly by subprocess pipeline tests.

## Notebook and reference consumers

The notebook contains exactly six scientific H2 sections: Problem, Data, Physics, Method, Results, Conclusion. It calls cleaned source to reconstruct full observations; historical or cached arrays never substitute for fresh reconstruction. Its active path performs motion, patch/global reconstruction, quality and relative fluorescence, evaluates fresh output and plots motion, source maps, traces, innovation events, relative activity, accepted/rejected quality, upstream/fresh numeric comparisons and corrected-mode sensitivity.

`notebooks/reference_outputs/upstream_first_*` are first independent oracle measurements. `upstream_repeat_*` are an independent repeated original execution, used to quantify reproducibility and validate thresholds. Fresh notebook measurements are separately named. The numerical references carry no physiological ground-truth label. Fixture consumers and hashes are individually indexed in validation; private capture scripts and complete source stay outside the task.

## Resource behavior and support limits

The native movie alone is about 347 MB in memory. Corrected images, Fourier workspaces, full regression projections and notebook figures add working memory; this is deliberately a full research pipeline, not a seconds-only demonstration. CPU single-thread runs are supported and tested. No GPU is required. Notebook timeout is 1800 seconds to include complete fresh reconstruction, comparisons and rendering on the validated host.

Only the selected 2D motion, greedy-ROI patch, finite positive-LARS, rank-two ALS and AR1 OASIS path is claimed. The exposed extracted helper defaults exist for documented independent fixtures; they are not a promise that arbitrary upstream modalities, neural classifiers, 3D movies, fixed-g merge defects, L0 OASIS, decimation or AR optimization are implemented. Unsupported OASIS modes are rejected explicitly. No optional helper replaces a required stage in the shipped call graph.

## Checked signature appendix

The appendix is generated from the shipped AST, while the tables and equations above supply the semantic contracts; signature agreement alone is not treated as algorithmic validation.


<!-- GENERATED SIGNATURES -->

### main.py

```python
def main(argv=None)
```

### physics_model.py

```python
def fluorescence_forward(A, C, b, f)
def fluorescence_adjoint(residual, C, f)
def get_noise_fft(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=3072, opencv=True)
def mean_psd(y, method='logmexp')
def estimate_parameters(fluor, p=2, sn=None, g=None, range_ff=[0.25, 0.5], method='logmexp', lags=5, fudge_factor=1.0)
def estimate_time_constant(fluor, p=2, sn=None, lags=5, fudge_factor=1.0)
def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp')
def update_order_greedy(A, flag_AA=True)
def axcov(data, maxlag=5)
def nextpow2(value)
def blocked_projection(Y, B, transpose=False)
```

### preprocessing.py

```python
def load_movie(path='data/raw_data.npz')
def apply_shift_iteration(img, shift, border_nan=False, border_type=cv2.BORDER_REFLECT)
def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None)
def _compute_phasediff(cross_correlation_max)
def register_translation(src_image, target_image, upsample_factor=1, space='real', shifts_lb=None, shifts_ub=None, max_shifts=(10, 10), use_cuda=False)
def get_patch_edges(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...])
def get_patch_centers(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...], shifts_opencv=False, upsample_factor_grid=1)
def sliding_window_dims(dims: tuple[int, ...], overlaps: tuple[int, ...], strides: tuple[int, ...])
def sliding_window(image: np.ndarray, overlaps: tuple[int, int], strides: tuple[int, int])
def tile_and_correct(img, template, strides, overlaps, max_shifts, newoverlaps=None, newstrides=None, upsample_factor_grid=4, upsample_factor_fft=10, show_movie=False, max_deviation_rigid=2, add_to_movie=0, shifts_opencv=False, gSig_filt=None, use_cuda=False, border_nan=True, shifts_interpolate=False)
def apply_pw_shifts_remap_2d(img: np.ndarray, shifts_y: np.ndarray, shifts_x: np.ndarray, patch_centers: tuple[list[float], ...], border_nan: Union[bool, Literal['copy', 'min']], shifts_interpolate=False)
class _TemplateMovie
    def __new__(cls, array)
    def __array_finalize__(self, obj)
    def motion_correct(self, max_shift_w=5, max_shift_h=5, num_frames_template=None, template=None, method: str='opencv', remove_blanks: bool=False, interpolation: str='cubic')
    def bin_median(self, window: int=10)
    def extract_shifts(self, max_shift_w: int=5, max_shift_h: int=5, template=None, method: str='opencv')
    def apply_shifts(self, shifts, interpolation: str='linear', method: str='opencv', remove_blanks: bool=False)
def motion_correct_movie(movie, progress=None)
```

### solvers.py

```python
def reconstruct(movie, progress=None)
def randomized_svd(X, k)
def initialize_background(X, k, fill_seed=None)
def coordinate_update(X, W, H)
class NMF
    def __init__(self, n_components=2, max_iter=200, init='nndsvdar', tol=0.0001, random_state=None)
    def fit_transform(self, X)
    def fit(self, X)
def downscale(Y, ds, opencv=True)
def resize(Y, size, interpolation=cv2.INTER_LINEAR)
def initialize_components(Y, K=4, gSig=(4, 4), gSiz=(9, 9), ssub=2, tsub=2, nIter=5, maxIter=5, nb=2, kernel=None, normalize_init=True, img=None, rolling_sum=True, rolling_length=100, greedyroi_nmf_init_method='nndsvdar', greedyroi_nmf_max_iter=200)
def greedyROI(Y, nr=30, gSig=[5, 5], gSiz=[11, 11], nIter=5, kernel=None, nb=1, rolling_sum=False, rolling_length: int=100, seed_method: str='auto', nmf_overrides: dict=None)
def finetune(Y, cin, nIter=5)
def imblur(Y, sig=5, siz=11, nDimBlur=None, kernel=None, opencv=True)
def hals(Y, A, C, b, f, bSiz=3, maxIter=5)
def positive_lars(X, y, alpha)
def HALS4shape_bckgrnd(Y_resf, B, F, ind_B, iters=5)
def threshold_components_parallel(pars)
def update_spatial(Y, A, C, b, f, sn, dims, streamed=False)
class _Pool
    def __init__(self)
    def __setattr__(self, key, value)
def constrained_oasisAR1(y, g, sn, optimize_b=False, b_nonneg=True, optimize_g=0, decimate=1, max_iter=5, penalty=1, s_min=-3)
def deconvolve_trace(y, p=1)
def update_temporal(Y, A, b, C, f, p=1, streamed=False)
def normalize(r)
def patches(dims)
def initialize_patches(movie, progress=print)
def merge(r, p=1, mx=1000)
def fit(movie, progress=print, checkpoint=None)
def quality(movie, r, corrected_mode=False)
def dff(A, b, C, f, YrA)
def mode_robust_fast(inputData, axis=None)
def release_compatibility_mode(inputData, axis=None, dtype=None)
def _hsm(data)
def df_percentile(inputData, axis=None)
def kde(data, N=None, MIN=None, MAX=None)
def fixed_point(t, M, I, a2)
def peak_indexes(y, thres=0.3, min_dist=1, thres_abs=False)
def symmetric_half_sample_mode(values)
def compute_event_exceptionality(traces, robust_std=False, N=5, use_mode_fast=False, sigma_factor=3.0, corrected_mode=False)
def find_activity_intervals(C, Npeaks=5, tB=-3, tA=10, thres=0.3)
def classify_components_ep(Y, A, C, b, f, Athresh=0.1, Npeaks=5, tB=-3, tA=10, thres=0.3)
def evaluate_components(Y, traces, A, C, b, f, final_frate, remove_baseline=True, N=5, robust_std=False, Athresh=0.1, Npeaks=5, thresh_C=0.3, sigma_factor=3.0, corrected_mode=False)
```

### visualization.py

```python
def metrics(estimate, reference)
def source_comparison(fresh, reference, source=0)
```
