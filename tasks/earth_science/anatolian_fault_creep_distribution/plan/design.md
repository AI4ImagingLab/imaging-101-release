# Shipped interfaces and execution

## Public and hidden boundary

Solvers receive README,`data/raw_data.npz`,and environment information only.
All source,plans,tests,notebook and evaluation assets are maintainer-only.
The input contains existing real velocities,known geometry/coordinates,LOS
vectors and GNSS uncertainties. It contains no inverse result,elastic Green
matrix or target-derived calibration. Small constants are in README and
matching source constants. No runtime reads a paper,upstream library,network
resource,another task or private archive.

## preprocessing.py

`load_observations(path: str | Path) -> dict[str,np.ndarray]` validates every
required batch-one key,shape,numeric kind and finite values, then removes
only the declared leading batch dimension. It checks positive GNSS standard
deviations and exact contiguous radar track order. It returns copied arrays;
coordinates remain km and velocities/uncertainties remain mm/year.

`observation_vector(data) -> ndarray(3554,float64)` concatenates radar3476,
GNSS east39,GNSS north39. Constants `SAR_COUNTS`,`NOISE_SIGMA`,
`NOISE_LENGTH_KM`,`NOISE_STD` match README and the source observations.

## physics_model.py

`RotateXyVec(x,y,alpha) -> (xp,yp)` performs the explicit two-dimensional
rotation with the source one-spacing floating-point offset. Inputs/outputs
have the same shape and physical units;alpha is radians.

`adv(y1,y2,y3,a,beta,nu,B1,B2,B3) -> (v1,v2,v3)` evaluates every expanded
angular-dislocation algebraic term locally using NumPy. Coordinate inputs
are same-shaped arrays,endpoint depth is scalar,beta is radians,nu is
dimensionless,and slip components set the returned displacement scale.

`displacement(sx,sy,sz,vertices,ss,ds,ts,nu=0.25) -> (ux,uy,uz)` forms the
oriented triangle,global/edge slip bases,two endpoint angular responses per
edge,and sum. The selected callable domain is surface receivers `sz=0`
with nonnegative-depth vertices. Inputs positions/vertices share km units;
the response is linear in slip units. Returned axes are east,north,up.
No interior-prism branch or polygon-library dependency is shipped.

`mogi_deflation_basis(xy_km) -> ndarray(n,3,float64)` returns fixed east,
north,up basis values in mm/kPa. Constants define the known empirical shape.
It has no dependency on hidden slip or fitted amplitude.

`build_green_matrix(data) -> ndarray(3554,1305,float64)` constructs every
triangle column,represents seven rectangles by exact pairs of triangles,
projects LOS/selects GNSS components,and appends basin/datum columns.
It does not load a precomputed matrix or cache results between invocations.

`build_data_covariance(data,diagonal=False) -> ndarray(3554,3554,float64)`
builds source noise blocks in squared mm/year units. The optional diagonal
mode is an active notebook ablation;standard reconstruction uses full blocks.

`build_model_covariance(triangles,sigma=20.,lengths_km=(10.,1.)) ->
ndarray(1305,1305,float64)` constructs the executable source conditional
covariance,including center-dependent normalization and nuisance variances.
No fitted target is used to determine this covariance.

## solvers.py

`assemble_problem(data,diagonal_noise=False,model_scale=1.) -> (A,d,Cd,Cm)`
returns the full3554x1305operator,3554observations,and the two covariance
matrices. `model_scale` multiplies the full prior covariance for sensitivity
only. The selected path uses its default1and full observation covariance.

`tie_deep_parameters(green,covariance) -> (Ared,Cmred,E)` implements the
source column-sum elimination. Shapes are3554x1301,1301x1301,1305x1301.
It explicitly assigns the shared deep variance100and preserves source
reduced ordering. It does not change acquisition geometry.

`objective_gradient(model,green,observations,inverse_data_covariance,
inverse_model_covariance,prior) -> (float,ndarray)` evaluates the stated
half-weighted Gaussian objective and analytic gradient in model units.

`solve_system(green,observations,data_covariance,model_covariance,
method='tied_nnls') -> (model,diagnostics)` accepts `gls`,`nnls`,or
`tied_nnls`. It forms source inverse covariance matrices and Cholesky
whitening,uses SciPy general NNLS or the source GLS formula,and restores
all1305parameters. Diagnostics include actual runtime,objective,weighted
observation RMS,mm/year RMS,gradient,reduced model,prior,residual norm,
and the NNLS projected-gradient norm. They are fresh numeric measurements.

`output_arrays(model) -> dict[str,ndarray]` partitions a finite1305-vector
into `slip_rate(1,1299)`,`basin_coefficient(1,1)`,`datum_offset(1,5)`,all
float64. Slip order and units exactly match README; the basin unit is formal
kPa/year. No component or low-sensitivity region is omitted.

`reconstruct(data,method='tied_nnls',diagonal_noise=False,model_scale=1.)`
actively assembles and solves the complete model and adds total runtime.

## CLI and notebook flow

`main(argv=None) -> int` in `src/main.py` parses `--data` (default
`data/raw_data.npz`) and `--output` (default`output/reconstruction.npz`).
It loads observations,reconstructs the standard tied NNLS target,creates
the output parent directory,and writes the three arrays to one NPZ.
The standard command is `python -m src.main` from the task workspace.
Importing main does not select or alter a Matplotlib backend.

The active notebook imports cleaned functions,loads only public observations
for fresh reconstructions,assembles one system,and solves the three groups
from that fresh system. Identified original comparisons are loaded from
`notebooks/reference_outputs/original_comparisons.npz`. Further fresh solves
use deliberately changed noise/prior assumptions and diagnostic observations;
none is substituted by a cached method result.

`generate_data.correlated_velocity_perturbation(data,covariance,
seed=20261022,fraction=.15) -> (copied_data,noise)` is notebook-only.
It samples a3554-vector in mm/year with covariance fraction²times the
supplied covariance and updates copied velocity arrays in declared order.
It neither mutates raw data nor constructs hidden targets. Its docstring
specifies seed,units and finite-vector boundary convention.

`visualization.py` supplies plotting/metric/export helpers for the notebook.
It has no reconstruction side effect or upstream dependency. Scientific
figures save numeric inputs and paired PNG/SVG/PDF in the private report
directory provided by the notebook environment. Canonical notebook outputs
remain embedded. A cache preview cannot satisfy active execution.

## Independent evaluation

`evaluation/judge.py` imports only general NumPy/SciPy and standard library.
`load_output(path)` validates the three exact shapes,float64 real dtypes,
and finite values with `allow_pickle=False`. It never reshapes,crops,
rescales,coerces complex arrays or imports a submission.

`load_reference(directory=ASSET_DIR)` selects existing actual truth first;
invalid truth is an evaluator error and does not fall back. No actual truth
is supplied for this task. The original method's three output arrays are
stored in `baseline_reference.npz`. Its separately identified `_green`,
`_observations`,`_data_covariance` are original geometry/noise calibration
for standalone forward scoring and full-stage tests,not privileged targets
or solver input. The Judge reads no source/test modules.

`field_metrics(prediction,reference)` computes the1292main-patch cosine and
range-NRMSE and separate deep/east/west/basin/five-datum maximum errors.
It rejects degenerate reference norms/ranges. `evaluate(prediction_path,
directory=ASSET_DIR)` adds independently calculated Mahalanobis RMS and
requires all12metrics to meet `metrics.json` boundaries. Missing/corrupt
references or calibration are errors with empty metrics.

`main(argv=None)` implements the required `--prediction`/`--report` CLI.
Hidden paths resolve from the Judge file; CLI paths resolve from caller cwd.
JSON is UTF-8 and finite-only. Exit codes0/1/2mean pass/valid-low-score/error.
Report-write failure emits stderr and returns2. Tests cover known metric
values,truth precedence,each required output group,invalid files/keys/shapes/
dtypes/nonfinite values,unavailable assets,and report-write errors.

## Fixture/reference consumers

- `test/fixtures/original_system.npz`: independent original model covariance,
  consumed by complete prior-stage parity. Other full original system arrays
  have one shared identified copy in the evaluator baseline archive.
- `test/fixtures/prefrozen_criteria.json`: source-only numerical/poor-control
  gate provenance,selected before local field scoring.
- `notebooks/reference_outputs/original_comparisons.npz`: the three actual
  upstream inverse results,consumed by comparison tests and notebook figures.
- `evaluation/baseline_reference.npz`: canonical method reference plus separate
  original forward/noise calibration,consumed by independent Judge and stage
  tests. Reconstruction is never labeled truth.

Complete upstream checkouts,papers,capture scripts,large original logs and
failed compatibility experiments stay in the private maintainer archive.

`vector_extent(ax, xy, vectors, scale, margin=.06)` sets plotting limits to include every native vector tail and its displayed tip; it changes display limits only.
