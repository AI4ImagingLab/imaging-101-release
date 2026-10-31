# Shipped architecture and interfaces

## 1. Runtime boundary and actual data flow

The standard invocation is `python -m src.main` from this task directory. It requires the validated CUDA runtime. It reads only `data/raw_data.npz`, constructs the complete single-step TGV inverse problem locally, runs its explicit double-precision primal-dual updates, and writes `output/reconstruction.npz`. There are no trained weights, domain-library calls, cross-task imports, downloads, or hidden-reference reads in this path. `--device cpu` selects the NumPy implementation; full-volume CPU completion has not been validated. The CUDA full-volume path and CPU numerical stages have been validated separately.

The caller chain is `main.main -> preprocessing.load_data -> solvers.reconstruct -> preprocessing.crop_slices -> physics_model.wrapped_laplacian -> solvers.primal_dual_torch`. `reconstruct` converts the cropped phase-unit susceptibility to ppm, restores the original full volume, and returns both physical quantities. `main` adds the batch dimension and saves them. Magnitude is loaded and validated as a public observation but does not enter this particular inverse objective.

The evaluator is a separate process: `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`. It has no import dependency on `src`. The notebook is another maintainer caller of the same reconstruction interface and actively executes both TGV and TV; it loads frozen upstream and independent measurement arrays only after reconstruction for scientific comparisons. Frozen outputs cannot enter `reconstruct`.

## 2. Array, coordinate, and unit conventions

Disk observations and submissions are batch first, `(1,160,160,160)`, with axes `(batch,x,y,z)`. Inside `src` arrays are unbatched `(x,y,z)`. Field direction is positive z. No array transpose, resampling, orientation registration, magnitude conversion, or reference-derived scaling occurs in reconstruction. Plot transposes are display operations only. Phase and discrepancy potential are radians, susceptibility on disk is ppm, spacing is mm. Solver variable `chi` is the phase-scaled susceptibility `s*chi_ppm`, where `s=19.35383488668318 rad/ppm`. This internal name does not imply ppm.

Primal scalar states `chi,eta` have shape `(nx,ny,nz)`. Auxiliary vector `w` and vector dual `p` have shape `(3,nx,ny,nz)`. Tensor dual `q` has shape `(6,nx,ny,nz)` in order `(xx,xy,xz,yy,yz,zz)`. Constraint dual `nu` is scalar. States and all arithmetic are float64; masks are boolean. The tensor norm uses weights `(1,2,2,1,2,1)` to implement the Frobenius inner product of a symmetric matrix.

The bound mask has 933481 voxels. Its crop is `(slice(18,143),slice(7,157),slice(11,137))`, shape `(125,150,126)`. The crop retains every mask voxel and two exterior cells around its bounding box. This is a computational restriction of the full support, not a replacement acquisition or reduced task. The PDE mask `Mi` contains 887715 sites. Its predicate requires all six axial neighbors to be in the acquisition mask; it does not require the center itself. All outermost faces are excluded. The actual acquired mask has no `Mi` sites outside the mask; the center-hole convention is nevertheless preserved and tested on an artificial unit case.

## 3. Preprocessing module

`load_data(path='data/raw_data.npz') -> dict[str,numpy.ndarray]` opens NPZ with `allow_pickle=False` and requires exactly the keys `phase`, `magnitude`, and `mask`. It checks the declared full batch shape, exact dtypes float64/float32/bool respectively, finite entries, a nonempty mask, and nonnegative magnitude. It returns independent unbatched copies. Contract violations raise `ValueError`; file/archive errors propagate. It does not estimate a mask or calibration from a hidden image.

`crop_slices(mask) -> tuple[slice,slice,slice]` finds the nonzero bounding box and extends it by two cells on each side, clipped to array bounds. An empty mask raises `ValueError`. A three-dimensional boolean mask is a caller precondition. No file is written and no input is mutated.

Constants `SPACING=(1.0625,1.0625,1.0714285714285714)`, `B0_T=2.89362`, `TE_S=.025`, `GYRO_MHZ_T=42.58`, `PHASE_PER_PPM=2*pi*GYRO_MHZ_T*B0_T*TE_S`, and `SHAPE=(1,160,160,160)` match README acquisition metadata. They are explicit rather than parsed from Markdown. The protocol identity evidence is recorded in validation.

## 4. Physics module

All functions allocate outputs and have no filesystem or plotting side effects. They accept three positive spacings and finite, consistently shaped inputs as numerical preconditions; the low-level helpers are not standalone public-file validators.

| Callable | Input / output | Actual operation |
|---|---|---|
| `wrapped_laplacian(phase, spacing)` | scalar 3-D radians -> float64 same shape, rad/mm^2 | Interior first differences wrapped to the principal interval, then backward divergence; outer faces explicitly zero |
| `interior_mask(mask)` | bool 3-D -> bool same shape | Six-neighbor predicate above; outer faces false |
| `forward_difference(u, axis, spacing)` | scalar 3-D -> same shape | `(roll(u,-1,axis)-u)/h[axis]` |
| `backward_difference(u, axis, spacing)` | scalar 3-D -> same shape | `(u-roll(u,1,axis))/h[axis]`; negative transpose of forward difference |
| `laplacian(u, spacing)` | scalar 3-D -> same shape | Seven-point second differences with separate axis pitches |
| `wave_operator(u, spacing)` | scalar 3-D -> same shape | Coefficients `(-1/3,-1/3,2/3)` on the three second differences |
| `symmetric_gradient(w, spacing)` | vector 4-D -> six-component tensor | Forward diagonal derivatives and half sums for off-diagonals |
| `symmetric_divergence(q, spacing)` | six-component tensor -> three-component vector | Backward derivatives with the pitch of each derivative axis; negative Frobenius adjoint |
| `project_balls(vector, radius, tensor=False)` | three/six components -> fresh same shape | Divide by `max(1,norm/radius)` pointwise; positive radius required |

The roll-based low-level derivative primitives are periodic scratch operators. They do not themselves impose zero extension. The actual solver clears outer faces of primal mask `M`, restricts PDE and regularizer dual increments to `Mi`, and multiplies primal updates by `M`. The cropped observation has a zero-support margin. Consequently wraparound cannot couple nonzero primal or dual states through opposite faces. Tests separately verify the primitive periodic adjoint identity and the source-matched masked pipeline. Documentation must not describe `np.roll` itself as a zero-extension implementation.

## 5. Solver module and defaults

`step_sizes(spacing) -> float` returns the shared step `1/sqrt(g*g+a*a)` with `g=4*sum(h^-2)` and `a=2*(hx^-2+hy^-2+2*hz^-2)`. It does not search for a step size. See approach for the complete update derivation.

`primal_dual(field_rhs, mask, spacing=SPACING, lam=DEFAULT_LAMBDA, ratio=DEFAULT_RATIO, tolerance=3e-4, max_iterations=5000, return_state=False) -> dict` is the NumPy backend. `field_rhs` means **minus** the wrapped Laplacian, in rad/mm^2. It checks matching three-dimensional shapes with at least three cells per axis, finite RHS, positive lambda, nonnegative ratio/tolerance, and positive iteration cap. Valid finite positive spacings and ordinary numeric finite hyperparameters are caller preconditions; the routine is not a comprehensive validator for arbitrary Python objects. It creates private states and masks and never mutates observation arrays.

`primal_dual_torch(field_rhs, mask, spacing=SPACING, lam=DEFAULT_LAMBDA, ratio=DEFAULT_RATIO, tolerance=3e-4, max_iterations=5000, return_state=False, device='cuda') -> dict` implements the same explicit mathematical stages with torch general tensor operations. It converts contiguous input arrays to float64 tensors on the selected device, creates a private bool mask, and uses `torch.no_grad()`. No neural network, autodifferentiated opaque inverse, or general optimizer replaces the published custom updates. Nested `difference`, `second_order`, and `project` are the torch equivalents of the local NumPy primitives. `device='cpu'` supports small independent CPU stage comparisons. An unavailable CUDA device raises the torch exception rather than silently falling back to a different path.

Both return `chi`, `eta`, `iterations`, and `history`; scalar states are NumPy arrays even for CUDA. History has shape `(number_of_checks,3)` and columns iteration, relative single-update chi norm, relative single-update eta norm. With `return_state=True` they additionally expose `w,nu,p,q,interior` for captured-state tests. Returned `eta` remains the same named scalar state. `ratio=0` selects TV: `w` and `q` remain zero and their updates are skipped; it does not call another package.

`DEFAULT_LAMBDA=PHASE_PER_PPM*.0015/2=0.014515376165012384` and `DEFAULT_RATIO=1/3` are fixed from the paper's physical pair after the documented half-fidelity normalization conversion. They are not upstream wrapper defaults or parameters selected to maximize COSMOS scores. All states start at zero, dual stages precede primal stages, and extrapolation is `2*new-old`. Every tenth update compares the **previous single update**, with thresholds `tolerance` and `2*tolerance`; maximum 5000 updates. The bound TGV run stops at 2480 and the TV comparison at 2040. There is no line search, stochastic sampling, or warm start.

`reconstruct(phase, mask, spacing=SPACING, lam=DEFAULT_LAMBDA, ratio=DEFAULT_RATIO, tolerance=3e-4, max_iterations=5000, device='cpu') -> (full_chi_ppm, diagnostics)` performs crop, wrapped-Laplacian RHS construction, solver dispatch, unit conversion and zero embedding. Inputs are unbatched full arrays. `device=='cpu'` selects NumPy; other supported device strings are passed to torch. The function does not subtract a target-derived mean or fit an offset. The source updates produce zero brain-mean states to numerical precision. Diagnostics contain `iterations`, `history`, measured `runtime_seconds`, crop slices, and full float64 `residual_phase`. The internal cropped chi/eta entries are removed from this dictionary after embedding. It writes no files and does not evaluate references.

## 6. CLI and notebook callers

`src.main.main(argv=None) -> diagnostics` uses argparse choices `cuda` and `cpu`, defaults to CUDA, and reads paths relative to the task working directory. It creates `output/`, overwrites only `output/reconstruction.npz`, and prints JSON containing actual iteration count and runtime. Required saved arrays are real float64 and batch-first. Importing `src.main` does not invoke the solver, import pyplot, or change a notebook backend. No library calls `matplotlib.use()`.

The notebook has exactly six major sections: Problem, Data, Physics, Method, Results, Conclusion. It locates this task, loads raw observations with `load_data`, actively reconstructs TGV and TV on CUDA, evaluates fresh joint quantities with the independent Judge, and computes independently named COSMOS/STI comparisons. Five rendered figures show raw observations; orthogonal COSMOS/fresh/difference; original-default/original-paper-pair/fresh-TGV/fresh-TV and paired metrics; orthogonal TGV/TV/COSMOS/STI; and discrepancy/PDE/convergence. The notebook's full execution, not a preview, is archived and reviewed. Frozen outputs are read only for labeled comparisons. Fresh measurements are not written back over upstream reference arrays.

`visualization.metrics(estimate, reference, mask)` returns cosine `ncc`, dynamic-range `nrmse`, and separately named `relative_l2`. It rejects empty/zero/constant references. `observation_metrics(reconstruction,residual_phase,phase,mask,spacing)` computes PDE-relative residual and eta RMS. These are maintainer convenience routines; the Judge independently implements its own physics. `observation_figure(observations)`, `reconstruction_figure(reconstruction,reference,mask,reference_name='COSMOS')`, and `method_comparison_figure(arrays,names,reference,mask)` return Matplotlib figures; the last additionally returns score dictionaries. Fixed displayed slices are voxel 80, and susceptibility panels use the same -0.15..0.15 ppm scale. No plotting routine modifies solver arrays or performs metric alignment.

## 7. Independent evaluation and failures

`evaluation.judge.evaluate(prediction, asset_dir=HERE) -> (report,exit_code)` loads metrics and references relative to `asset_dir`; `HERE` is the evaluator file's directory. Actual truth takes precedence if supplied; this task intentionally has only `baseline_reference.npz`, containing the independently corrected C method result. COSMOS and STI are imperfect independent measurements used for scientific review, not relabeled exact targets.

The evaluator requires both output keys and exact full shape, float64 real dtype, finite entries, exactly zero exterior, and absolute brain-mean no greater than 1e-8 in each physical unit. It does not crop, reshape, real-cast, scale, or align a submission. Array metrics use the public brain support. Its independently sliced seven-point PDE uses all six neighbors, physical axis pitches, a fixed rad/ppm scale, and `atan2(sin,cos)` wrapping rather than importing the solver modulo code. Eta RMS discourages making the auxiliary potential arbitrarily large. Zero chi has NCC zero and fails. Invalid constant or zero-norm evaluator references raise evaluator errors rather than producing undefined JSON values.

All six boundaries must pass. Successful evaluation returns 0, valid but inadequate arrays return 1, and malformed submissions or missing/invalid assets return 2. JSON contains status, passed, measured metrics, directional thresholds and errors. Error cases leave metrics empty. `main(argv=None)` requires prediction/report paths, creates the report parent, emits UTF-8 finite JSON, and handles report-write failures with stderr plus exit 2. Prediction/report paths follow the caller's working directory. Hidden references do not.

Finite inputs can still overflow intermediate norms. Before attaching measurements to a report, the evaluator explicitly checks that every computed metric is finite; overflow therefore returns a descriptive error with empty metrics and exit 2. A subprocess test exercises this using a valid-shaped, mask-contained, exactly zero-mean pair of finite +1e200/-1e200 values and checks that the resulting report remains valid UTF-8 JSON.

## 8. Validation consumers and dependency roles

`test/comparing_test/test_upstream_stages.py` consumes the frozen real-patch oracle stages for wrapped Laplacian, all TGV state variables at 1/2/10/50 updates on NumPy and torch CPU, and original TV susceptibility at those stages. These fixtures predate the local port. `test/unit_test/test_physics.py` independently checks phase wrap invariance, harmonic background annihilation, quadratic Laplacian, Fourier dipole sign, first and symmetric adjoints, directional derivatives, projections, mask/crop convention, stopping and invalid numeric parameters. `test/unit_test/test_judge.py` checks known metric numbers, independent physics parity, malformed arrays, missing assets, rejected poor/degenerate joint solutions and subprocess interface behavior. `test/comparing_test/test_contract.py` checks public constants/contracts and evaluates the already independently run fresh full CLI artifact; it does not mislabel an import or a cached upstream result as a pipeline execution.

NumPy supplies arrays and elementary linear algebra; torch supplies the CUDA tensor backend; Matplotlib supplies maintainer visualization. No SciPy dependency is required by the shipped runtime. SciPy MAT reading, Octave MEX compilation, Jupyter, and pytest belong only to the private upstream/development environment and are absent from task runtime requirements. The source archive, download ZIPs, PDFs, capture scripts, logs, stage comparison manifests, and original runs are outside this task; validation indexes their consumers and identities. There is no generate_data module because no new synthetic observations are generated.


## Public-contract revision 2026-10-29

Version2.0-joint-tgv-primal-witness adds actual source-computed auxiliary_w, physical ppm/mm, shape(1,3,160,160,160). Source primal updates unchanged; reconstruct requests existing return_state and zero-pads/scales its real w. Historical output/baseline/source/interface archived under benchmark_readiness_20261029/qsm_v1_archive. Fresh CUDA main2480updates31.2764s and full actual TGV+TV nbconvert exit0/no cell errors completed with .24process GPU memory cap coordinated with ADJUST. The new independently implemented physical objective is14.571308579816776; bound14.571308725529862 fixed with relative1e-8 numerical allowance, original PDE/eta thresholds unchanged. No actual tissue truth exists. Independent conditional witness descent lowers objective to14.570909040355454 and passes, without changing chi/eta; zero auxiliary and all-zero outputs fail. This control is not a fresh blind full susceptibility solve.
