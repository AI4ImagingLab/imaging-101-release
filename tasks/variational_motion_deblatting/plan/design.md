# Shipped interfaces and execution flow

## Runtime dependencies and public interface

The task uses Python 3.11 with NumPy 2.4.6, SciPy 1.17.1 and Matplotlib 3.11.1.
NumPy/SciPy supply FFTs, sparse matrices and general conjugate gradients. No
upstream package, scikit-image or OpenCV is imported in the cleaned task.
Matplotlib is used for review figures only. The original upstream environment
is archived outside this task and never imported at runtime.

`python -m src.main` reads `data/raw_data.npz`, reconstructs all three unknowns,
and writes `output/reconstruction.npz`. CLI flags `--data` and `--output` replace
these paths, resolved relative to the caller. `--projection exact` is the default;
`--projection upstream_cyclic` explicitly requests the original defective
projection for a diagnostic reproduction and must not be confused with the
standard corrected method. No root-level wrapper exists.

## Modules and callable contracts

`src/preprocessing.py`:

- `load_data(path='data/raw_data.npz')` checks both public arrays are finite
  float64 with shape (1,169,137,3), then returns unbatched observation/background
  (169,137,3), BGR, normalized encoded camera intensity. It does not modify inputs.
- `OBJECT_SHAPE=(36,36)` fixes the unknown object support from the upstream
  measurement-only detection result.
- `line_aa(r0,c0,r1,c1)` returns integer row indices, integer column indices and
  float64 antialiasing weights in upstream order. `renderTraj(pars,H)` writes
  trajectory weights into H in place and returns it; (2,2) pars encodes start and
  displacement, while (2,3) additionally supports the upstream quadratic term.
- `boundingBox(img,pads=None)` returns rmin,rmax,cmin,cmax with the same upstream
  exclusive slicing conventions. Standard runs use the full crop support.

`src/physics_model.py` contains extracted executable operators, constraints and
the explicit forward model:

- `forward(background,kernel,appearance,mask)` returns the (169,137,3) real
  normalized predicted observation using the exact shipped circular embedding.
- `psfshift(H,usize)` zero-pads compact 2D/3D fields and moves their center to
  the Fourier origin; `psfshift_idx(small,sz_large)` provides matching index
  vectors in Fortran order. `ipsfshift(H,hsize)` supports inverse 3D embedding.
- `vec3(I)` and `ivec3(I,ishape)` implement Fortran-order spatial vectorization
  and inverse reshaping. They do not alter physical units.
- `createDerivatives0(sz)` returns CSC D_x,D_y with shape
  ((rows+1)*(columns+1),rows*columns), implementing unit-pixel zero-extended
  differences. `createRnMatrix(img_sz,angles=None)` returns the sparse rotational
  averaging map, using radians and the upstream four default angles.
- `proj2simplex(Y)` returns a nonnegative unit-sum array of the same shape.
- `project2pyramid_exact(m,f,eps)` returns exact projected opacity/color arrays;
  m has shape (n,) or (n,1), f has shape (n,3), and validated eps=1 is required.
- `project2pyramid(m,f,eps)` preserves the source revision's finite cyclic
  approximation exclusively for the explicit original-parity mode. It is not
  represented as exact projection.
- `lp_proximal_mapping(val_norm,amount,p)` returns vector shrink factors. The
  shipped method calls only p=1, with zero norm handled by zero shrink factor.
- `psffit(H,returnpars=False)` returns the normalized fitted straight trajectory;
  with returnpars=True it additionally returns its (2,2) start/displacement.

`src/solvers.py`:

- `Params()` stores all settings listed in approach.md. The added
  `projection_mode='exact'` selects the mathematically corrected projector.
- `estimateFMH(I,B,M=None,F=None,Hmask=None,params=None)` returns H,F,M.
  Standard callers supply I,B (169,137,3), initial M=ones((36,36)), omit F and
  Hmask, and disable verbose/visualization. It initializes F=ones, reuses
  `StateH`/`StateFM`, changes params.maxiter to 1 for blind alternations and 50
  for the final fixed-trajectory refinement. It preserves the upstream final
  trajectory fit and stopping conditions. Input arrays I,B are not overwritten.
- `estimateH(I,B,M,F,Hmask=None,state=None,params=None)` implements the explicit
  ADMM/CG kernel subproblem. With state=None it returns H; with state supplied,
  returns H,state and mutates that warm splitting state.
- `estimateFM(I,B,H,M=None,F=None,F_T=None,M_T=None,state=None,params=None)` returns
  F,M or F,M,state. It implements coupled appearance/opacity ADMM/CG; optional
  templates are not supplied in the standard task. It mutates warm state when
  supplied, caching derivative matrices, rotation normal matrix, auxiliaries and
  scaled multipliers. Finite primal values are not clipped after the solve.
- `StateH()` and `StateFM()` hold the upstream warm-start variables, initially
  zero/None. No checkpoints or reference results are accessed.

`src/main.py`:

- `reconstruct(observation,background,projection_mode='exact')` sets up Params,
  disables interactive visualization, initializes the compact support and returns
  a dict with batch-first float64 keys reconstruction (1,36,36,3), mask (1,36,36),
  kernel (1,169,137). It does not read hidden assets or write files.
- `main()` parses CLI arguments, loads observations, calls reconstruct, creates
  the output parent directory and writes compressed NPZ. It reports elapsed wall
  time. Importing src.main does not change a Matplotlib backend.

`src/visualization.py` draws the review comparisons from supplied arrays and
computes comparison metrics using their declared formulas. It selects no backend,
loads no hidden assets itself, and does not reconstruct or alter data. The
notebook identifies every historical/upstream/fresh array before calling it.

## Evaluation, tests and notebook

`evaluation/judge.py` is standalone and imports no solver modules. Its CLI accepts
`--prediction` and `--report`; hidden asset lookup is relative to its own file,
while CLI paths follow the caller. `load_arrays` enforces real float64, exact
batch shapes, required keys and finite values. `compare` implements cosine NCC
and dynamic-range NRMSE. `model` independently embeds compact fields and predicts
the exposure from the public background. `evaluate(prediction,assets=None)`
returns status, passed, metrics, thresholds and errors; malformed inputs or
missing/invalid evaluator assets produce empty metrics and status error. `main`
writes strict UTF-8 JSON, creates report parents, and returns 0/1/2 for pass,
valid threshold failure, or malformed/evaluator/report-writing error.

`test/fixtures/upstream_operators.npz` freezes independently captured source
operator values. `upstream_pipeline.npz` freezes the unmodified source result.
The corrected-projection fixture is separately identified. Unit tests check
independent differences/adjoints, simplex KKT, exact constraint projection against
SLSQP and normal-cone inequalities, the preserved source defect, metrics,
malformed predictions, CLI status codes and public contracts. Full pipeline tests
reconstruct the observations in each explicitly identified mode and compare to
the appropriate independent fixture; they do not import the oracle checkout.

`notebooks/variational_motion_deblatting.ipynb` actively runs the standard exact
reconstruction on the bound public observations, evaluates the fresh result and
compares it with original upstream, local original-parity and corrected oracle
measurements. It shows observation/background/model residual, foreground/opacity/
kernel comparisons, numerical metric tables, and paired metric bars. Archived
arrays are labeled comparisons, never substituted for a fresh reconstruction.
Execution timeout is 900 seconds per cell, reflecting measured CPU runtime.
The single shipped notebook retains executed outputs; temporary executed copies
and standalone preview PNGs are removed from the task directory after review.

## Shapes, units, errors and callable edge semantics

The common bound notation is R=169,C=137,h=w=36,n=1296. Camera intensities,
premultiplied appearance and opacity are dimensionless; kernel samples represent
dimensionless exposure occupancy, summing to1 in the fitted final result. Pixels
are the coordinate unit. Except explicit in-place trajectory writing and warm
state/Params mutation, numerical helpers allocate results and perform no disk I/O.
They assume compatible numeric input arrays; validation in load_data/Judge must
not be confused with broad defensive checks inside extracted upstream helpers.

| Callable | Detailed input -> output contract |
|---|---|
| `forward(background,kernel,appearance,mask)` | float64 `(R,C,3)`, `(R,C)`, `(h,w,3)`, `(h,w)` -> real float64 `(R,C,3)` predicted camera scale |
| `boundingBox(img,pads=None)` | 2D occupancy and optional2-vector integer pads -> four integer bounds; empty occupancy errors via indexing; upper extrema are not incremented |
| `line_aa(r0,c0,r1,c1)` | Integer endpoint pixel coordinates -> equally long integer rr/cc and float64 weights; endpoints included, no image clipping here |
| `renderTraj(pars,H)` | Real start/displacement `(2,2)` or quadratic `(2,3)` pixels and writable2D target -> same target after rasterization; rounds endpoints and excludes row0/column0; repeated hits overwrite |
| `psffit(H,returnpars=False)` | Real2D occupancy -> normalized real same-shape fitted line; if True returns `(Hfit,pars)` with `(2,2)` pixel parameters; insufficient/empty support may propagate polyfit/index errors |
| `createDerivatives0(sz)` | First two integer image dimensions -> tuple CSC float64 matrices `(1369,1296)` for bound support; unit pixel difference operator |
| `createRnMatrix(img_sz,angles=None)` | First two dimensions and optional radian angle vector -> CSC float64 `(n,n)` row-normalized valid nearest rotation average; default four angles specified in approach |
| `project2pyramid_exact(m,f,eps)` | m `(n,)` or `(n,1)`, f `(n,3)`, eps exactly1 -> tuple same-shaped real opacity and color; no input mutation; eps!=1 raises ValueError |
| `project2pyramid(m,f,eps)` | Same array shapes and scalar coupling -> tuple finite cyclic approximate opacity/color; source alias defect deliberately retained; not exact projection |
| `lp_proximal_mapping(val_norm,amount,p)` | Real norm array, scalar shrink threshold, exponent -> same-shaped shrink factors; p=1 validated, p=1/2 retained unvalidated branch, other p raises Exception |
| `proj2simplex(Y)` | Nonempty real numeric array -> same-shaped nonnegative unit-sum array using Fortran flatten; no physical units changed |
| `psfshift(H,usize)` | Compact2D/3D/4D numeric array and full first-two-axis dimensions -> float64 padded/rolled array preserving trailing axes |
| `ipsfshift(H,hsize)` | Full3D array and compact dimensions -> inverse-rolled compact3D array; its3D slicing is not a general2D inverse API |
| `psfshift_idx(small,sz_large)` | Dimension tuple or boolean support and full dimensions -> tuple of integer index vectors in the compact Fortran ordering |
| `vec3(I)` | Real `(h,w,k)` field -> `(h*w,k)` Fortran spatial matrix, same dtype/units |
| `ivec3(I,ishape)` | `(h*w,k)` matrix and target shape -> reshaped `(h,w,k)` array, same dtype/units |

`load_data(path='data/raw_data.npz')` returns two unbatched float64 arrays
in observation/background order. Missing file or keys propagate; wrong exact
shape/dtype or nonfinite values raise ValueError. It reads only the named public
archive with allow_pickle=False. `OBJECT_SHAPE` is a constant, not a function or
estimated hidden-template lookup.

## Inverse state and parameter ownership

`Params.__init__(self)` accepts no arguments and installs every default listed
in approach. A standard caller may supply the same mutable Params to the block
solvers. `estimateFMH(I,B,M=None,F=None,Hmask=None,params=None)` requires equal
image shapes or raises Exception. If M is omitted its general default is full
image ones, not the standard compact36x36 support; main explicitly supplies that
compact mask. F defaults to corresponding all-ones color. Hmask=None means full
image support; optional bool Hmask triggers bounding-box cropping and later
embedding. Return `(H,F,M)` has full kernel `(R,C)`, compact appearance `(h,w,3)`
and opacity `(h,w)` float64. The function mutates params.maxiter to1 and later50
if final fitting is enabled; it creates and mutates internal states, not I/B.

`estimateH(I,B,M,F,Hmask=None,state=None,params=None)` takes fixed real intensity,
opacity and appearance arrays with shapes above, optional boolean kernel support,
StateH and Params. It returns H alone if state is None and `(H,state)` otherwise.
StateH.__init__(self) creates zero/None-valued splitting storage: passed state
is an owned warm object and changes in place. `estimateFM(I,B,H,M=None,F=None,
F_T=None,M_T=None,state=None,params=None)` analogously returns `(F,M)` or
`(F,M,state)`. With absent M and supplied F, its generic M default is zeros with
F's support; absent both uses full image zero support, unlike the main caller's
ones initialization. Optional templates have respective compact shapes and camera/
opacity units; they are not used in this task. StateFM.__init__(self) initializes
cached derivative/rotation normals, gradient/constraint auxiliaries and scaled
multipliers to None/zero. Neither state holds upstream arrays or hidden assets.

The nested estimateFM_cg_Ax(fmfun0) receives one float64 vector `(4*n,)`, reshapes
in Fortran field-block order, applies the explicit coupled normal operator and
returns a vector of identical shape. estimateH_cg_Ax(hfun) receives a kernel
support vector and returns its normal-operator action. They close over current
spectra, background and split matrices, have no independent saved state/file I/O,
and are passed only to SciPy's general LinearOperator/CG. They are not callable
upstream reconstruction wrappers. CG info is ignored as in the selected source.

`reconstruct(observation,background,projection_mode='exact')` takes unbatched
float64 bound observations, prepares the compact initialization, and returns
three-key batch-first float64 dict with shapes already declared. main() has no
Python parameters and parses --data/--output/--projection; unknown CLI projection
choices are rejected by argparse. Paths remain caller-relative; writes happen
only after reconstruction and parent-directory creation. No backend is selected
when importing this module.

## Plotting and Judge function contracts

`metrics(estimate,reference)` accepts equal-shaped real numeric arrays, flattens
without clipping and returns `(ncc,nrmse)` Python floats. It assumes nonzero norm
and dynamic range; unlike Judge it is not a public malformed-input validator.
`acquisition_figure(observation,background,result)` accepts unbatched images and
the three-key batch dict, calls only the local forward model and returns a1x4
Figure. `comparison_figure(results)` accepts an insertion-ordered label->dict
mapping, returns a len(results)-by3 Figure and uses common kernel range.
`metric_figure(results,reference)` takes that mapping and reference dict, returns
a1x2 Figure with paired per-field bars, full NCC range and an explicitly labeled
1e-16 log-display floor. All scientific metric arrays remain unclipped; plotting
changes no arrays/backend and saves no files.

Judge `load_arrays(path)` returns a validated three-key dict, requiring exact
float64 batch shapes and finite values. `compare(estimate,reference)` accepts two equal-shaped real arrays and returns
a dictionary with NCC/NRMSE Python floats; `model(arrays,background)` independently computes
camera prediction from batched output dict and unbatched background.
`evaluate(prediction,assets=None)` returns a report dictionary alone, not an
exit-code tuple; assets defaults to Judge's directory and may be overridden in
tests. It validates config/reference/public-background/submission and adds
observation_rmse. `main()` derives0/1/2 from that report, creates report parents
and handles write errors. No Judge call executes or imports solver code.
The independent model and metric implementations are checked using known cases,
not by treating the cleaned reconstruction as truth.
