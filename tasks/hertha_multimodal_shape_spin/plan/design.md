# Shipped interfaces and data flow

This task runs independently on the validated Windows x64 shared Python runtime.
NumPy/SciPy provide general numerical primitives; Matplotlib is used only by
review figures. The existing MSVC14.29 C++17compiler and Windows SDK are an
explicit environment exception for readable local facet/intersection derivative
kernels embedded as `CPP_SOURCE` in `src/physics_model.py`. No upstream package,
program, checkout, FITS/WCS library, parser, or network download is used at
runtime. A standard-library-only /MT DLL is compiled into a source-hashed
temporary cache; this is a math-code cache, never a reconstruction/result cache.

## Standard caller and files

`python -m src.main` calls `main(argv=None)`, parses optional `--data`/`--output`
paths, and calls `run(data_path=None, output_path=None, progress=None)`. Defaults
resolve relative to the module's task root, not an unrelated current directory:
`data/raw_data.npz` and `output/reconstruction.npz`. Explicit command-line paths
retain normal caller-relative semantics. `run` loads observations, actively
reconstructs every selected measurement, creates the output parent, writes the
one declared NPZ and returns `(output, record)`. `main` prints actual iteration
records and elapsed time. No numerical reference/evaluation asset is read by
the reconstruction. Importing `src.main` does not select a Matplotlib backend.

`output` contains float64 real arrays: `reconstruction`(1,402,3)km;
`spin`(1,3)latitude-deg,longitude-deg,period-hours; `ao_offsets`(1,2,2)arcsec;
`ao_log_gains`(1,2)dimensionless; `occultation_offset`(1,1,2)km. `record` contains
the311final internal parameters and the ordered list of trial records. A trial
record contains iteration, current objective, trial objective, damping, accepted
boolean and step_norm. These are measurements, not alternate hidden targets.

## Observation preparation

`image_fourier(image, pixel_scale=0.009942)` accepts a finite real even2Dimage,
clips negative values to zero, applies the local center phase and DC normalization,
and returns a complex1Dvector without DC and a float64(N,2)frequency array in
cycles/arcsec. It rejects nonpositive total flux, odd shape and nonfinite input.
`load_observations(path='data/raw_data.npz')` uses `allow_pickle=False`, strips
the leading batch axis into a dictionary, checks the selected AO/sample sizes,
and computes `ao_fourier`(2,11399),`ao_frequency`(11399,2), and
`ao_psf_fourier`(2,11399). It flattens LCtime/brightness from(923,1)to(923,).
No observations or calibration are synthesized. Public fixed representation
angles/faces are checked against the local generated native order in tests.

## Local geometry and mathematical kernels

`sphere_grid(nrows=10)` returns readonly cached(theta,phi,faces), with402angle
entries and800zero-based triangle rows for the selected nrows. It validates a
positive integer nrows. `harmonic_basis(lmax=9,nrows=10)` returns a readonly
cached(402,100)float64 real harmonic matrix. `octantoid(coefficients,lmax=9,
nrows=10,padding=0)` validates the3*(lmax+1)^2real finite coefficients and returns
mesh(402,3)km plus coordinate-major derivatives(1206+padding,300+padding).
`rotation_matrix(angles,time,derivatives=False)` accepts four internal radian
spin values and a scalar day offset; it returns a3by3rotation or that matrix
with three3by3analytic derivatives. The fourth phase is fixed by callers.

`_math_library()` compiles `CPP_SOURCE` only when the source-SHA256-named cache
lacks its DLL. The default cache is the system temporary directory under
`imaging101-hertha-math/<source-sha256>`. `HERTHA_MATH_CACHE` changes only the
cache root. The existing compiler/SDK can be located with `HERTHA_MSVC_ROOT`,
`HERTHA_SDK_ROOT` and `HERTHA_SDK_VERSION`; their documented defaults match the
validated installed MSVC14.29.30133 and SDK10.0.19041.0. The subprocess uses
shell=False and explicit command arguments: /LD /O2 /fp:precise /EHsc
/std:c++17 /MT /W3. Child INCLUDE/LIB values are set only for this compilation;
the shared environment is not changed. Compile stdout/stderr and `build.json`
are stored with the source in that cache. A missing compiler or failed native
build raises a clear exception. Static runtime linking avoids private runtime
DLLs; the loaded binary contains only the embedded local standard-library math.

The C++ source declares only selected facet Fourier/reflectance, visibility,
projection, intersection and surface-regularizer kernels, explicit analytic
derivatives, general small matrix helpers and typed allocation. It contains no
original data parser, fitting entry point, optimizer, checkpoint loader or domain
API linkage. Unselected constant-albedo/Hapke branches are specialized to the
selected model. The C99-to-C++ complex representation, allocation lifetime and
memory-only COMreturn relocation are documented adaptations. Complete numeric
fixtures and independently formulated physical checks validate these kernels.

Python wrappers expose:

| Function | Arguments/units | Returned values |
|---|---|---|
| `triangle_fourier(frequencies,vertices)` | (N,2)cycles/arcsec; three2Dvertices in arcsec | Complex(N,)oriented integral and(N,6)vertex derivative. |
| `ao_operator(mesh,angles,observer,sun,up,time,distance,frequencies,offset,derivative=True)` | Mesh(402,3)km; internal4spin; three unit vectors; days; AU; (N,2)cycles/arcsec; (2,)arcsec | Complex(N,)normalized unblurred transform; optionally ten real arrays: paired x/y/z vertex derivatives(N,402),paired3spin derivatives(N,3),paired2offset derivatives(N,2). |
| `lightcurve_operator(mesh,angles,observer,sun,time,relative=True)` | Mesh/spin; (N,3)observer/Sun vectors; (N,)days | Mean-normalized brightness(N,)and tuple(dx,dy,dz,(N,3)spin derivative); XYZblocks each(N,402). |
| `occultation_operator(mesh,angles,up,observer,velocity,time,offset,chords,chord_type)` | Mesh/spin; 3vectors, velocity km/s; (18,2)days; (2,)km; (18,4)km; native integer flags | (72,)predicted-minus-measured residual and tuple of three(72,402)mesh derivative blocks,(72,3)spin derivative,(72,2)offset derivative. |
| `mesh_penalties(mesh)` | Native(402,3)km | (3,)convexity/dihedral/COMvalues and(3,1209)coordinate-major derivatives with spin padding. |

`_native_mesh(mesh)` validates the finite native mesh shape, supplies its fixed
outward1-based CABIindices, and makes C-contiguous buffers. CABIwrappers supply
all unused-but-source-required phase/chord derivative work arrays, avoiding null
reads. The C++kernels never load observation/reference files or perform inversion.

## Joint model and solver

`initial_parameters()` returns the independent311component native seed.
`octantoid_penalty(coefficients)` returns the selected scalar degree-weighted
quadratic coordinate-radius residual and its300derivatives.
`residual_and_jacobian(parameters,data,derivative=True,
weights=(8.,2.,.1,30.,5.,.5,1.))` returns the complete(46595,)weighted residual and
(46595,311)negative residual derivative, or `None` for the latter in residual-only
mode. Residual-only mode avoids AOcoordinate derivative arrays; the LC/OC/mesh
mathematical helpers retain their explicit source formulas. It creates no files.
The order is923LCrows;22798AOreal rows then22798AOimag rows;72OCcoordinates;
convexity,octantoid,dihedral,COMscalars. AOradiometric multipliers and PSFcomplex
multiplication are included in the derivative chain, and every nuisance column
has its actual physical unit/sign. The array ordering and shapes are verified
against original captured residual, sampled Jacobian, complete normal matrix,
right-hand side and first native damped update.

`reconstruct(data,max_iterations=100,weights=(8.,2.,.1,30.,5.,.5,1.),progress=None)`
performs the complete custom accepted/rejected LMtrial loop in Python. It calls
general `scipy.linalg.solve(...,assume_a='sym',lower=True)` only for the explicit
311by311damped normal system. It returns the batch-first output dictionary and
the internal parameter/history record. The optional `progress` callback receives
one actual trial dictionary. No optimizer from a domain library or generic
minimize wrapper replaces the native custom update. Full defaults and stopping
conditions are in `approach.md`; main and the active notebook use those defaults.

## Standalone evaluation

`evaluation/judge.py` is executable independently from any caller directory with
`--prediction` and `--report`. `evaluate(prediction_path,evaluation_directory=HERE)`
returns the full report. Hidden paths use its own resolved directory. Actual
`ground_truth.npz`, if present, takes precedence; an invalid truth file produces
an evaluator error with empty metrics instead of a baseline fallback.
`baseline_reference.npz` is a validated standard-method accepted state, not
claimed truth. All required keys, exact shapes, real floating types, finite
values and typed nonnegative finite metric boundaries are checked before grading.
Derived metrics are also checked for overflow/nonfinite results. No submitted
solver code or reconstruction kernel is imported.

`field_metrics(prediction,reference)` computes all9field/nuisance metrics, with
coordinate-specific ranges and only longitude wrapping. If any fails, the valid
prediction is already a fail; potentially undefined degenerate forward geometry
is not evaluated. If all pass, `forward_objective(output,raw,return_details=False)`
computes the tenth complete weighted objective using independent NumPygeometry,
visibility, Green-boundary Fourier integrals, endpoint intersections and
regularizers. Its real harmonics infer the three coefficient fields from the
submitted mesh using the fixed public representation; no hidden coefficients are
used. `return_details=True` also returns components, actual LC/AOmodels,
occultation endpoints, missed_chord_count and penalties for independent scientific review. A missed detected chord uses the native support-line sigmoid residual; its unavailable endpoint detail is NaN, never a substituted observation or a metric value. The helper `_missing_chord_residual(xy,chord)` takes offset-inclusive projected vertices (N,2)km and measured endpoints (4,)km and returns four unweighted optimization residuals plus its scalar sigmoid. All complete objective contributions remain finite.

Reports contain status, passed, finite metrics, each comparison/boundary and
errors. `main(argv=None)` writes UTF-8 strict JSON, creates report parents, and
returns0forpass,1forvalidbelow-gate predictions,2formalformed/evaluator/report
errors. Report-write errors also emit a stderr diagnostic. Errors always have
empty metrics. Public paths supplied on the command line stay caller-relative.

## Maintainer notebook and scientific figures

The canonical notebook actively calls `run` on the complete raw acquisition,
checks the independent Judge, and computes actual comparison forward models.
Its six chapters are problem,data,physics,method,results,conclusion. Original
accepted O1/O0fields and the original author subdivision export are identified
in `notebooks/reference_outputs/`; they are never relabeled as fresh or truth.
The native subdivision support/configuration comparison is an archived original
method run, not an unimplemented alternate local solver. The selected local
octantoid reconstruction is recomputed by default, without a result-cache path.

`src/visualization.py` receives already-computed measurements, produces the six
scientific figure groups, and writes matching PNG/SVG/PDF and numeric source
arrays outside the task into the caller's figure directory. It does not select
a Matplotlib backend or perform inversion. The notebook records its actual
interpreter,NumPy/SciPyversions/backend and elapsed reconstruction time. A private
subprocess wrapper executes nbconvert with that shared interpreter, captures real
stdout/stderr/native exit, and retains canonical outputs. Temporary executed
notebooks and compiler products remain outside the final package.
