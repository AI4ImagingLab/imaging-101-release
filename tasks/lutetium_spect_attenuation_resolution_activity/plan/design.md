# Shipped interfaces and execution

Requirements are NumPy2.4.6, SciPy1.17.1, PyTorch2.9.1 and Matplotlib3.11.1.
SciPy supplies only independent interpolation checks in development tests.
PyTorch supplies tensor operations, grouped convolution, bilinear sampling,
small matrix inverse and CPU/CUDA arithmetic; no acquisition or inverse domain
library is imported. Pytest and Jupyter are shared development tools and do not
belong in task requirements. No data generator exists because the task reuses
real observations and creates no new synthetic observation dataset.

## Preprocessing

`load_data(path)` opens NPZ with allow_pickle=False, validates exact batch-first
shapes/dtypes, finite numeric observations/calibrations, nonnegative attenuation
and positive radii, and returns a dictionary with the batch axis removed.
Shapes are energy_windows (3,96,128,128) uint16, attenuation (128,128,128)
float32 inverse cm, angles (96,) float32 degrees, radii (96,) float64 cm.
It performs no changes of scale, spatial interpolation or inferred support.
`WINDOW_BOUNDS` holds the three exact archived endpoints; `SHAPES` and `DTYPES`
define the storage contract. `scatter_estimate(energy_windows,device='cpu')`
returns float32 Torch (96,128,128), in counts, on the selected device using
explicit TEW arithmetic. It neither mutates inputs nor writes files.

## Forward and backward mathematics

`rotation_grid(angle,height,width,device,dtype)` returns Torch
(1,height,width,2), using explicit homogeneous transforms, inverse sampling
and corner alignment. Angle is signed degrees. Float32 pi and normalisation
constants preserve original precision even in float64 independent tests.
`rotate_volume(volume,grid)` maps a Torch (x,y,z) tensor by permuting z into
the channel dimension, bilinear sampling with zero extension, and reversing
the permutation; dtype/device are retained.

`detection_probability(mu,dx=DX)` maps detector-frame inverse-cm attenuation
to dimensionless centre-to-exit survival probability with an explicit reverse
cumulative sum. `gaussian_kernel(radius,size,dx=DX,device='cpu')` returns a
Torch float32 (size,1,kernel_width) tensor. Radius/dx are cm and size is padded
x depth. The independently computed NumPy float64 distance/width is converted
at the same point as source to float32 Gaussian arithmetic. `blur_volume(volume,
kernel)` returns a tensor of the same shape/device, applying y then z grouped
convolution with zero boundaries. Constants `DX`, `DELTA`, `PSF_PARAMETERS`
are copied exactly from independently bound source metadata.

`Acquisition(attenuation,angles,radii,device='cpu',attenuation_enabled=True,
psf_enabled=True,n_subsets=8)` holds calibration tensors, padded attenuation,
native angle order, one kernel per unique radius, boundary box, and a cache of
sampling grids. The two booleans exist solely for explicitly labelled notebook
physics ablations; selected benchmark defaults retain both physics effects.
It does not cache rotated attenuation probability arrays, fit calibration,
read files or infer a support mask. The unpadded shape is the attenuation shape.

`indices(subset=None)` returns all Python angle indices, or the interleaved
subset with stride n_subsets. `grid(angle)` returns/builds its cached grid.
`probability(index)` independently rotates the known padded mu and evaluates
Beer-Lambert integration. `forward(volume,subset=None)` accepts Torch activity
(128,128,128) and returns (number_of_subset_angles,128,128) predicted primary
counts. `backward(projections,subset=None)` returns the full activity-shaped
inverse-angle approximate continuous backprojection. Both are no-grad paths;
no claim of exact discrete adjoint is made. Shape constraints of the public
benchmark are enforced by preprocessing/Judge, while small analytic operator
helpers are exposed to tests.

## Inverse and standard entry point

`reconstruct(acquisition,photopeak,scatter,n_iterations=4,capture=False)` converts
counts/scatter to float32 on acquisition.device, caches all subset sensitivities,
and executes the explicit fixed-count OSEM updates from ones. It returns
`(image,states)`, where image is the full Torch tensor on its device and states
is an empty list unless capture=True. In capture mode it contains an independent
NumPy float32 copy after each subset update, in iteration-major/subset-major
order (32 for the standard run). No files, runtime global device setting or
Matplotlib backend changes occur. The source's nonpositive-value projection is
part of each update and does not reduce evaluation support.

`run(data_path,output_path,device=None)` in src.main sets Torch CPU thread count
to one, selects CUDA if available unless explicitly overridden, loads raw data,
forms TEW scatter and Acquisition, actively reconstructs, and writes compressed
NPZ with only reconstruction (1,128,128,128) float32. It creates the output
parent directory and returns the unbatched Torch image. The standard command
`python -m src.main` uses task-relative data/raw_data.npz and
output/reconstruction.npz. Optional --data, --output and --device cpu|cuda
override those paths/device. No backend is changed when importing src.main.

## Standalone evaluation

`evaluation/judge.py` imports no src module or submitted code. `read_volume(path)`
validates presence, exact shape, real float32 dtype and finite values.
`agreement(estimate,reference)` computes independent NumPy float64 cosine,
dynamic-range RMS and dynamic-range maximum errors; zero estimates receive
NCC zero, while a degenerate evaluator reference raises an evaluator error.
`observation_error(estimate,raw_path)` independently evaluates the declared
acquisition using public arrays and explicit geometry, Beer-Lambert factors,
Gaussian kernels and full forward projection. It uses CUDA if available and
CPU otherwise, one CPU thread, float64 reductions, and never imports the
reference forward model. It performs no reconstruction or parameter fitting.

`evaluate(prediction,reference_dir=HERE)` loads metrics relative to the evaluator
directory. If ground_truth.npz exists it is mandatory and takes precedence;
invalid truth does not trigger baseline fallback. Otherwise baseline_reference
is used. It returns (JSON-compatible report,exit_code). All metrics must pass.
Malformed submissions or evaluator failures return error/2 and empty measured
metrics; valid poor submissions fail/1; accepted submissions pass/0. The CLI
requires --prediction and --report, treats their relative paths relative to the
caller's cwd, creates report parent directories and reports write errors to
stderr with exit2. JSON contains status/passed/metrics/thresholds/errors and
disallows nonfinite JSON numbers.

## Maintainer comparisons and evidence

Tests under test/unit_test exercise independent analytic geometry, attenuation,
PSF, metrics and malformed NPZ/priority contracts. test/comparing_test consumes
only frozen upstream numeric fixtures, not the source library. Upstream stages
are test/fixtures/upstream_stages.npz. Independently reproduced full activity,
prediction and 32 states are notebooks/reference_outputs/upstream_measurements.npz.
Historical Siemens vendor reconstruction is a separate, explicitly labelled
reference_outputs/historical_vendor.npz. No historical result is used as a
freshly reconstructed figure. Each fixture's source/capture/hash/consumer is
listed in test/validation.md. The notebook imports only cleaned src modules,
reconstructs the complete real bound observation set and saves actual figures
with paired numeric sources through src.visualization. Plans, notebook metric
definitions and Judge are checked on common independent cases. Private source
checkouts, DICOMs, paper, raw run logs, temporary notebooks and final atlas
exports remain outside this task package.

Final evaluator contract: a finite, shape-correct prediction containing any negative activity is a valid quality failure through negative_count<=0, including a single arbitrarily small negative voxel. Negative hidden reference is an evaluator error. NCC boundaries must lie in (0,1] and error boundaries must be nonnegative finite numbers, with booleans rejected. Every measured metric is checked for finiteness before report publication: a finite float32 input whose forward arithmetic overflows returns error2 with empty metrics and a descriptive error, producing valid finite JSON. No reference-agreement threshold is changed.
