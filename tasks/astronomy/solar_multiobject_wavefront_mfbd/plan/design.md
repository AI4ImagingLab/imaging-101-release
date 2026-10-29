# Shipped implementation and interfaces

## Runtime boundary and complete data flow

The standard command is `python -m src.main` from the task workspace. It loads
only `data/raw_data.npz`, constructs known optics from the public theoretical KL
mixing table, estimates noise from the measured burst, reconstructs the complete
two-passband scene, and writes `output/reconstruction.npz`. No evaluation file,
notebook reference, archived implementation, checkpoint, or cross-task import is
read by this path. There is no synthetic generator or training stage.

The execution chain is:

```text
main(argv) -> load_data -> reconstruct
  normalize_frames -> extract_patches
  estimate_noise (CPU, original noncontiguous layout)
  cosine_window -> apodize (selected device) -> CPU contiguous FFT
  build_optics (CPU geometric grids; selected-device diffraction cosine)
  optimize_wavefronts
    canonical_modes -> pupil_to_psf x2 -> reduced_loss
    autograd -> Adam update -> update active-mode count
  pupil_to_psf x2 -> project_objects -> scharmer_filter
  mosaic -> CPU NumPy result
main -> save the three contract arrays
```

Both channels use the same aperture/basis and unknown phase, but separate image
spectra and measured noise. Keeping two OTF evaluations preserves the selected
original arithmetic and gradient graph. All 121 regions and twelve frames are
optimized simultaneously. Splitting the original loss into independently
normalized patch batches would change its noise variance and is not implemented.

## Array and coordinate conventions

Visible observations have shape `(1,2,12,512,512)`, float32, with axes burst,
passband, exposure, detector row, detector column. `load_data` removes only the
leading singleton. Passband zero is broadband and one is Ca II narrowband. Raw
values retain the upstream calibrated digital-intensity scale. Normalized
intensities are dimensionless, relative to each entire frame's spatial mean.

The public mixing table is `(1,436,3)`, float32. Each record contains a one-based
KL index, one-based Noll Zernike index, and dimensionless real mixing weight.
It is theoretical calibration, independent of the observed Sun. Optical phase
coefficients are radians. The 54-slot basis comprises KL indices 2 through 54
and a final zero slot, preserving the upstream allocation discrepancy. The last
coefficient remains zero; first-exposure tip/tilt is subtracted per region.

Patches are row-major on an 11 by 11 grid, with top-left detector coordinate
`(40*row,40*column)`. Each contains 12 frames of 88 by 88 pixels. The complete
512-square frame sets normalization even though the patch lattice ends at 488.
The final 476-square mosaic covers detector `[6:482,6:482]`; it does not silently
pad to 512. Its outermost boundary is zero by the shipped Hann overlap weights.
Images use row then column; PSFs have zero displacement at FFT index `(0,0)`.
Only visualization shifts PSFs to their center.

## Preprocessing module

`src.preprocessing.load_data(path='data/raw_data.npz')` returns two NumPy arrays:
`observations (2,12,512,512)` and `mixing (436,3)`. It uses `allow_pickle=False`
and rejects incorrect declared shapes, non-float32 dtypes, and nonfinite input.
Missing files/keys propagate explicit loader exceptions. It does not rescale or
alter the input arrays.

`normalize_frames(observations)` copies to float32, divides each frame by its
float32 spatial mean, rejects nonpositive means, and returns a CPU Torch tensor.
No reference image is used to estimate an intensity scale.

`extract_patches(frames, patch_size=88, stride=40)` returns a list of two Torch
tensors, each `(121,12,88,88)`. It uses general `functional.unfold`, reshape and
permute. The returned strides are intentionally `(1,937024,10648,121)`. In the
validated numerical backend, making these arrays contiguous before the noise
SVD changes rounding; the source therefore preserves their original layout.
The function permits other spatial grids as a numerical helper, while the
benchmark and CLI use exactly the stated complete grid.

`cosine_window(size=88, border=6)` returns a float32 CPU tensor. Its separable
profile uses NumPy's `hanning(2*border)` at the two edges and ones inside.
`apodize(patches, window)` returns `(patches-mean)*window+mean`, where the mean is
over the final two dimensions. The caller controls device placement. The solver
moves both arguments to the selected device, applies the taper there, copies to
a contiguous CPU buffer, computes `fft2(norm='ortho')` on CPU, and moves complex64
spectra to the reconstruction device. This apparently unusual sequence matches
the original implementation; device-dependent mean accumulation materially
changes a continued nonconvex trajectory.

`estimate_noise(patches)` returns float32 CPU `(121,12)` estimates in normalized
intensity units. It obtains a NumPy view of CPU patches, forms each 88 by 88
spatial covariance, computes its general NumPy SVD, and evaluates the explicit
Gavish-Donoho median formula in `approach.md`. No upstream noise package is used.

`mosaic(patch_objects, apodization=6, weight_border=30, stride=40)` takes
`(objects,patches,88,88)` real Torch tensors, checks a square patch grid, crops
six-pixel borders, and weighted overlap-adds 76-square interiors. It returns
`(objects,476,476)` on the input device/dtype. Accumulated weights receive an
explicit `1e-10` denominator. It mutates only newly allocated accumulators.

## Optical model module

`src.physics_model.noll_indices(index)` converts a one-based integer Noll index
to `(radial_order,signed_azimuthal_order)`. `build_optics(mixing,size=88,device='cpu')` uses
that mapping and SciPy's general Jacobi polynomial to evaluate local Zernikes,
then explicitly accumulates KL mixing. It returns a dictionary of selected-device float32
Torch tensors: `pupil (88,88)`, `basis (54,88,88)`,
`frequency_radius (88,88)`, and `diffraction_mask (88,88)`. The clear aperture
uses integer coordinates -44 through 43; the basis instead uses inclusive
linspace coordinates -1 through 1. Constants are 100 cm diameter, 8542 Angstrom
effective wavelength, 0.059 arcsec pixels, and prior spectral power 100. The cosine transition is evaluated on the selected device, preserving original GPU rounding. No domain package constructs the basis, pupil, phase, or inverse operator.

`canonical_modes(modes)` clones `(patches,frames,54)` and subtracts each region's
first-frame coefficients from the first two slots at every frame. It preserves
autograd and does not mutate its argument.

`pupil_to_psf(modes,pupil,basis,active=None)` contracts active real coefficients
with basis functions, evaluates the complex pupil, computes its orthonormal
FFT, normalizes squared modulus to unit flux, and returns `(psf,otf)`, each
`(patches,frames,88,88)` with float32/complex64 in standard execution. `otf` is
the orthonormal FFT of `psf`, with DC `1/88`. `active=None` means all 54 slots.
No image-space centering is applied in the scientific runtime.

`reduced_loss(frame_fts,otfs,sigmas,diffraction_mask)` accepts lists of two
complex64 `(121,12,88,88)` data/OTF arrays and two float32 `(121,12)` noise arrays.
It returns a real scalar, retaining the input real precision. The variance is
one global mean squared noise per channel. It explicitly forms the eliminated
object quadratic, multiplies by the fixed diffraction mask, averages indices
`[1:,1:]` over regions/frequencies, and sums the channels. Autograd supplies the
gradient through these locally implemented optical and objective operations.

`scharmer_filter(normal,cross,variance,diffraction_mask)` takes complex spectra
`(121,88,88)`, scalar global channel variance, and the real mask. It transfers
the shifted real spectral ratio to CPU, applies SciPy's general periodic
3-square median filter and explicit threshold logic, then uses connected
component labeling with eight-neighbor adjacency to implement the original
central flood criterion. It returns a real support mask on `normal.device`.
CPU transfers here are part of the selected algorithm, not cached inference.

`project_objects(frame_fts,otfs,sigmas,diffraction_mask)` explicitly forms
per-frame inverse-variance weights normalized by each patch's mean variance,
normal/cross spectra, their division with `1e-10` protection, and the above
support mask. The support filter still uses global channel mean variance. It
uses an ordinary inverse FFT and returns float32 `(2,121,88,88)` on the spectra
device. Object values are not clipped or mean-renormalized after inversion.

## Solver and CLI

`src.solvers.annealing_schedule(iterations=250)` returns an int64 NumPy vector
of length `iterations`. Values progress through 2,5,9,14,20,27,35,44,54 according
to the original 10%-60% schedule. Activation occurs after the current update;
the first update always uses two modes.

`optimize_wavefronts(frame_fts,sigmas,optics,iterations=250,progress=None)`
allocates zero float32 leaf coefficients `(121,12,54)` on the spectra device,
uses Adam at constant 0.08 with Torch defaults, and returns detached canonical
coefficients and a device loss-history tensor of length `iterations`. It does
not instantiate the upstream scheduler that was never stepped. The optional
callback is called as `progress(zero_based_iteration,float_loss,active_count)`
after each update and before the activation count changes. It has no effect on
the updates. There is no early stop, line search, warm start, pretrained model,
or hidden-image initialization.

`reconstruct(observations,mixing,device=None,iterations=250,progress=None)` is
the complete callable pipeline. `device=None` chooses CUDA when available and
CPU otherwise. It returns a dictionary with float32 NumPy arrays
`reconstruction (1,2,476,476)`, `objects (1,2,121,88,88)`,
`wavefront_coefficients (1,121,12,54)`, float32 `loss_history (250,)`, and a
Python float `runtime_seconds`. The configurable iteration count supports
stage diagnostics; the benchmark path always runs 250. Runtime covers
preprocessing, optics, optimization, projection and CPU conversion; it excludes
file compression and plotting. The function performs no file writes.

`src.main.main(argv=None)` parses `--data`, `--output`, and optional
`--device {cpu,cuda}`. Defaults are the standard public input and output paths.
It sets Torch CPU threads to one, prints objective progress every 25 iterations
and at the last update, creates the output parent, writes exactly the three
contract arrays with compressed NPZ, prints runtime, and returns the in-memory
result dictionary. It always runs the complete 250-update method. Paths are
relative to the caller's workspace unless absolute. Importing main or any
library module does not change Matplotlib's backend.

## Standalone evaluation path

`evaluation/judge.py` imports only NumPy and the standard library. It never
imports `src` or calls submitted code. `load_arrays(path)` enforces required
keys, exact shapes, float32 and finiteness. `intensity_metrics(estimate,reference)`
computes flattened uncentered cosine NCC and dynamic-range NRMSE in float64;
zero estimates have NCC zero and degenerate hidden references raise errors.
`relative_error` computes explicitly named L2-relative error. `merge_objects`
independently evaluates the overlap-add convention. If that merge has zero norm, mosaic consistency uses absolute RMS error in normalized intensity units, avoiding an evaluator error for a valid but poor zero submission.

`psf_and_observation_metrics(coefficients,objects,reference_psfs,pupil,basis,
data_path)` independently evaluates complex pupils with NumPy, all 121 by 12
PSFs and both channels' normalized/tapered measured frames. It returns two
Python floats: relative PSF error and relative observation residual. It does
not compare nonunique coefficients directly or fit translation/brightness.
The known basis and pupil are recorded reference calibration, not inferred
hidden phases; public inputs suffice to derive them.

`evaluate(prediction,directory=HERE)` returns `(report,exit_code)`. Hidden paths
resolve against `HERE`, the evaluator file's directory. It reads the fixed
`metrics.json`, reproduced `baseline_reference.npz`, and public observations.
There is no actual ground truth in this task. All nine configured metrics must
pass, including intensity, PSF, observation, mosaic and gauge criteria.
Malformed input, absent evaluator assets or invalid metric configurations
produce status `error`, empty metrics, errors, and code 2. Valid below-boundary
predictions produce `fail` and code 1; pass returns code 0. `main(argv=None)`
accepts required `--prediction` and `--report`, creates the report parent, writes
finite UTF-8 JSON, and prints it. Report-write errors return 2 on stderr.

## Notebook and visualization path

The notebook has six numbered sections: Problem, Data, Physics, Method, Results,
Conclusion. It uses `reconstruct` for a fresh complete run, saves every contract
array, and launches the standalone Judge. It loads reproduced original arrays
only for comparisons; no archived data replaces a fresh result. Its kernel and
device are displayed before reconstruction. The optional upstream-only oracle
environment is never imported.

`src.visualization` exposes `contrast(image)` (population RMS percent),
`radial_power(image)` (annular fluctuation power and cycles/arcsec), and
`plot_data(observations)`, `plot_comparison(best,upstream,fresh,traditional)`,
`plot_wavefront(coefficients,pupil,basis)`,
`plot_diagnostics(losses,upstream_losses,best,upstream,fresh,traditional)`, and
`plot_differences(upstream,fresh)`. Plot functions return Matplotlib figures;
they do not save arrays, select backends, or change the solver. Figure methods
assume the declared full benchmark shapes. The notebook explicitly calls
`plt.show()` after each figure group.

The traditional MOMFBD archive is a separately processed comparison, registered
by a documented fixed crop and normalized by its own crop mean solely for
display. It is not truth, a Judge reference, or an input to reconstruction.
Best-exposure selection is similarly a comparison only. Contrast/power figures
exclude the known one-pixel zero mosaic border from every method, explicitly
labeled; Judge intensity comparisons retain every declared output pixel.

## Assets and their consumers

`test/fixtures/upstream_stages.npz` is frozen original CPU numeric evidence,
consumed by stage parity tests. `evaluation/baseline_reference.npz` stores the
complete original method result and its pupil-generated PSFs/calibration for
the independent Judge and notebook comparison. `upstream_measurements.npz`
records the original objective/runtime; `fresh_measurements.npz` records an
identified local complete run; `traditional_comparison.npz` stores display-only
independent processing. These are inside `notebooks/reference_outputs/`, with
source, archive index and consumers documented in `test/validation.md`.
Capture scripts, complete upstream source, HDF5/FITS downloads, logs and failed
investigation outputs remain in the maintainer archive outside this task.
