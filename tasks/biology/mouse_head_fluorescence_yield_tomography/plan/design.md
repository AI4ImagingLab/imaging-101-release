# Shipped architecture and execution

The runtime subset is Python standard library, NumPy 2.4.6, SciPy 1.17.1 and Matplotlib 3.11.1. No original domain library, MATLAB/Octave, mesh reader or upstream checkout is imported by shipped code. Existing public observations are reused; no new synthetic acquisition is generated and no generate_data module is required. General analytic test cases are unit tests, not substituted benchmark observations.

## Input and geometry

`preprocessing.load_raw(path)` strictly loads `data/raw_data.npz` with pickle disabled. It checks exact keys, batch-first shapes, float64 numeric observations/coordinates, int64 indices, boolean boundary flags, finiteness, positive measured intensities and index bounds. It returns an unbatched dictionary, with all ten shapes specified in README.

`barycentric_locations(nodes,elements,points,tolerance=1e-10)` returns tetrahedron indices, barycentric weights and outside markers through the returned index -1. It constructs local inverse affine maps and uses a centroid KD-tree only to shortlist candidates; explicit barycentric containment performs the geometric calculation. `interpolation_matrix(nodes,elements,points,nearest_inside_fallback=False)` returns the sparse interpolation matrix, indices, weights and the original outside indicator. The optional fallback copies the nearest valid fine-point weights; it does not interpolate unknown values during geometry construction. `geometry_operators(raw)` returns source-load matrix (5707,8), detector matrix (8,5707), unfiltered adjoint-load matrix (5707,8), fine-to-coarse map C (3323,5707), and coarse-to-fine map B (5707,3323).

## Physics

`physics_model.tetra_geometry(nodes,elements)` returns element volume in mm^3 and basis gradients in mm^-1, rejecting zero-volume tetrahedra. `assemble_local(elements,values,n)` performs explicit local-to-global sparse summation. `exterior_faces(elements)` returns connectivity-defined exterior triangles and rejects nonmanifold multiplicities. `refractive_robin(ri=1.33)` evaluates the original Fresnel coefficient. `assemble_fem(nodes,elements,mua,kappa,ksi=None)` returns a CSC CW diffusion matrix, CSC P1 mass matrix and volumes. Optional `ksi` supports independent analytic checks; standard runtime derives it from refractive index. Constants are exact supplied mesh values, not recomputed from rounded scattering values.

`continuum_prejacobian(elements,vol,phix,adjoint,links)` explicitly evaluates triple-barycentric moments and returns complex128 (56,5707) sensitivity. `forward_fluorescence(gamma,phix,mass,emission_lu,detector,links)` returns all56 positive predicted amplitudes and the (5707,8) emission field. Sparse LU is a general numerical primitive; no acquisition or inverse algorithm is hidden in a library.

## Inverse and caller flow

`solvers.initialize_uniform(observed,phix,mass,emission_lu,detector,links,max_iterations=40,tolerance=1e-4)` runs the native derivative-bisection initialization, returning scalar absorption and the midpoint/objective/derivative history. `reconstruct(raw,regularization=10.,max_iterations=40,calibrate=True)` builds geometry and operators, solves excitation and adjoints, calibrates observations, initializes uniform yield, computes fixed continuum sensitivity and runs source inexact relative-coordinate updates. Positive finite regularization and positive integer iteration count are required.

Its result dictionary contains reconstruction (5707,), prediction (56,), calibrated_observation (56,), excitation_prediction (56,), initial_muaf scalar, initial_history (number_steps,3), objective_history (number_evaluations,), regularization_history (number_updates,), iterate_fields (number_updates,5707), phix and phim (5707,8), prejacobian (56,5707), gamma (5707,), and forward_relative_L2 scalar. These are fresh algorithm measurements, not hidden inputs. Only reconstruction is submitted. Scientific comparisons may inspect the other fields without exposing them to the benchmark solver.

`python -m src.main` calls `main(argv=None)`, using task-relative raw data by default and caller-relative `output/reconstruction.npz`. Optional `--input` and `--output` accept paths. It creates the output parent, writes exactly float64 reconstruction of shape (1,5707), and prints runtime/forward residual/evaluation count. Importing main does not select a Matplotlib backend. No output is silently read as a cache.

## Independent evaluation

`evaluation/judge.py --prediction PATH --report PATH` resolves hidden assets relative to its own file. Prediction/report paths are caller-relative. It checks exactly one float64 field of the declared shape, finite and nonnegative, validates thresholds and references, and prioritizes ground_truth.npz whenever present; invalid truth never falls back. The real binding uses baseline_reference.npz. Independent P1 assembly and exhaustive optode barycentric lookup compute full observation consistency from public inputs; no source module or submitted code is imported.

The UTF-8 JSON report contains status, passed, metrics, thresholds and errors. Invalid submissions/evaluator errors have empty metrics. CLI codes are 0/pass, 1/valid below limits, 2/invalid/evaluator/report-write failure, with report-directory creation and stderr write diagnostics. Numerical metric functions reject zero-range/zero-norm references rather than serializing undefined values.

## Maintainer assets

Frozen native stages are consumed by comparing tests for matrices, geometry interpolation, fields, calibration, continuum sensitivity and complete inverse history. baseline_reference.npz is a qualified author-method result, not truth. The original incomplete-Robin source result is retained only as an explicitly failed-source notebook comparison. Tests cover native physics, analytic P1 conservation, complete pipeline, metric definitions, truth priority, invalid submissions and threshold cases. The active notebook reconstructs the complete bound data with local modules; separate original outputs are comparison assets. All downloads, source trees, compiled oracle code, run logs and complete papers remain in the private maintainer archive outside this task.


## Current benchmark contract v2 (2026-10-29)

The task now asks for a nonnegative measurement-consistent conditional field, with no unique spatial truth certification. The independent P1 FEM evaluates all 56 calibrated observations; mean squared log-residual excess <=1e-6 and the original relative L2 <=0.191 remain. GN damping 10 is a numerical update setting, not a static Tikhonov prior. An independent complete 56x5707 adjoint sensitivity matrix and NNLS-derived improved feasible field pass; zero/scaled/spatially permuted fields fail. The source reconstruction is unchanged.

Historical pixel/parameter identity thresholds and prior notebook reports refer to v1. The external benchmark_readiness_20261029 archive preserves old README/Judge/metrics/tests and actual current CPU command logs, including failed diagnostics. Source algorithms and arrays were not changed.
