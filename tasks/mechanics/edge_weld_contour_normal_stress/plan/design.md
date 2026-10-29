# Shipped modules, shapes and execution flow

The cleaned runtime is standard library plus NumPy2.4.6, SciPy1.17.1and Matplotlib3.11.1. It imports no VTK, Qt, native FE solver, source checkout or cross-task implementation. Existing measured observations are reused. There is no synthetic-data generator because no new observations are generated.

## Public input and preprocessing

`preprocessing.load_raw(path)` reads NPZ with `allow_pickle=False` and checks exact keys, shapes, dtypes, finite values and mesh index bounds. It returns an unbatched dictionary. Input shapes are exactly those in README:2010×3samples,2010bool selection,128×3outline,34826×3mesh nodes,30370×8connectivity,3166surface indices and2anchors. Coordinates/heights are float64mm and indices are zero-based int64. Its only side effect is file reading; no hidden field or reference is loaded.

`fit_surface(raw,knot_spacing=(3.,7.))` returns the five-entry SciPy spline representation (xknots,yknots,coefficients,degreeX,degreeY). The cubic degrees are3/3. It performs the source's explicit knot construction and fits only the public true-mask samples. Invalid/nonpositive controls or too few knots raise ValueError. Coefficients have height units mm. It does not save a file.

`boundary_displacements(raw,tck)` evaluates the profile at the pre-deck cut-node x/y positions minus the fixed1mmframe offset. It returns float64(3166)mm after source float32point storage and six-decimal formatting. `serialized_geometry(raw)` returns float64(34826,3)mm after source six-decimal geometry formatting. Those two roles deliberately use the supplied original geometry at their appropriate precisions.

Module constants are E_MPA195600, POISSON.294and FRAME_OFFSET_MM1. README metadata matches them. Geometry and mask are public acquisition/context, not target-derived initialization.

## Explicit physics routines

`physics_model.elastic_tensor(modulus=195600.,poisson=.294)` returns6×6float64D in MPa for engineering-strain order xx,yy,zz,xy,xz,yz; it rejects nonpositive modulus or Poisson outside(-1,.5). `shape_gradient(natural)` returns8×3natural-coordinate derivatives of the explicit trilinear basis.

`strain_matrix(element_coordinates,natural)` accepts(nelements,8,3)mm and a3-vector natural location, computes explicit mapping Jacobians/physical derivatives and returns B(nelements,6,24)mm^-1and determinant(nelements)mm³. It rejects nonpositive or nonfinite determinants. `element_dofs(elements)` returns int32(nelements,24)node-major Cartesian global indices.

`assemble_stiffness(nodes,elements,modulus=195600.,poisson=.294)` explicitly sums eight fully integrated brick matrices and assembles their sparse global contributions. It returns CSC(3N,3N)in N/mm; default geometry N34826has104478degrees of freedom. Local element/dof arrays are temporary. No matrix is loaded from hidden assets.

`integration_stress(nodes,elements,displacement,modulus=195600.,poisson=.294)` accepts N×3mm displacement and returns float64(nelements,8,6)MPa, using the declared native Gauss and stress ordering. `nodal_normal_stress(elements,gauss_stress,n_nodes)` returns float64(N)MPa after inverse shape interpolation of zz and equal-count node averaging. It rejects unused nodes. It does not relabel the interior response as original residual stress.

## Inverse and standard entry point

`solvers.reconstruct(raw,knot_spacing=(3.,7.),modulus=E_MPA,poisson=POISSON)` performs fresh profile fitting, original boundary/geometry representation, explicit stiffness assembly, Dirichlet elimination, rigid gauge fixation, direct sparse LU, reaction evaluation and stress recovery. Optional arguments are declared scientific comparison settings, not alternate datasets or hidden initializations.

Its returned dictionary contains:

| Key | Shape/type | Role |
|---|---|---|
| reconstruction | float64(1,3166)MPa | The sole required submitted normal-stress field |
| displacement | float64(34826,3)mm | Fresh full restoration solve, with explicitly fixed gauge |
| reaction | float64(34826,3)N | K u, for free-force/equilibrium checks |
| gauss_stress | float64(30370,8,6)MPa | Complete fresh integration stress diagnostic |
| nodal_normal_stress | float64(34826)MPa | Averaged normal restoration response, before selecting cut nodes |
| boundary_displacements | float64(3166)mm | Actual serialized prescribed heights |
| free_dof_force_relative | float scalar | Free-force norm divided by the full reaction norm |
| strain_energy_n_mm | float scalar | One half displacement dot reaction |
| spline | five-entry tuple | Fresh fitted profile for contour review |

No returned diagnostic is used as a cached answer. `reconstruct` itself has no file-writing side effects. It reads only the supplied unbatched public dictionary.

`python -m src.main` calls `main(argv=None)`: default input is caller-relative data/raw_data.npz; default output is caller-relative output/reconstruction.npz. Optional --input/--output accept explicit paths. It loads observations, reconstructs once, creates the output parent, writes exactly reconstruction and prints actual elapsed time, force residual and energy. Importing src.main does not select or alter a plotting backend.

## Independent evaluator

`evaluation/judge.py --prediction PATH --report PATH` is standalone and never imports src or submitted code. `load_field(path)` requires exactly reconstruction, float64(1,3166)and finite signed values. `field_metrics(x,y)` computes scaled cosine, reference-range NRMSE and maximum absolute error in MPa, rejecting undefined reference configurations and overflow. `evaluate(prediction,evaluation_dir=None)` validates finite numeric threshold types/ranges and prioritizes actual ground_truth.npz if it exists. Invalid actual truth never falls back. This binding uses baseline_reference.npz, explicitly not truth.

Prediction/report paths follow caller cwd; hidden paths follow the Judge file. UTF-8strict JSON contains status, passed, metrics, thresholds and errors. Invalid prediction or unavailable/invalid hidden assets give empty metrics and descriptive errors. Exit codes are0pass,1valid-below-limits,2malformed/evaluator/report-write failure. Report parent creation and stderr write diagnostics are implemented. No implicit reshape, crop, gain or magnitude conversion occurs.

## Tests, notebook and scientific graphics

`test/fixtures/upstream_stages.npz` contains native original knots/coefficients and serialized boundaries; `native_full_stress.npz` contains actual complete native six-component integration stresses; `native_brick.npz` contains independently prescribed/native known-element cases. Comparing tests verify source-stage parity and a fresh complete pipeline followed by a foreign-cwd Judge. Unit tests cover analytic elasticity, rigid null modes, symmetry/energy, Gauss inversion, strict inputs/metrics/evaluator failures and public metadata. They do not import the oracle, native solver or domain packages.

`visualization.comparison_metrics(estimate,reference)` returns ncc,nrmse,max_abs_mpa for finite nondegenerate arrays. Independently known numeric cases check agreement with the standalone metric conventions. `stress_panel(ax,xy,values,title,limit=None)` returns a scatter artist in author-frame mm, with equal spatial aspect and a symmetric signed MPa scale. `export_figure(fig,directory,name,numeric,caption)` writes PNG/SVG/PDF, paired numeric NPZ and JSONhash/source metadata to the explicitly supplied private directory. It returns the figure record. No library calls matplotlib.use.

The active notebook imports these cleaned modules, reconstructs all bound observations and writes/evaluates the fresh submission. It then executes three further complete reconstructions with declared fitting changes. Original source arrays are loaded only for identified comparisons; no historical array is substituted for a fresh field. It exports eight scientific groups and a private figure index, plus phase progress JSON. Its figure directory comes from IMAGING_FIGURE_DIR or a newly created system temporary directory, not an extra final task report folder.

The standard output directory contains only reconstruction.npz and judge_report.json. Native logs, compiler/executable files, source trees, papers, capture scripts, intermediate notebook revisions and graphics exports remain in the maintainer archive. No temporary executed notebook is left in output. The canonical notebook is copied from a successful actual nbconvert execution only after native exit0.
