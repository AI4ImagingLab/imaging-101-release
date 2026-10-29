# Provenance, mathematical review and completed gates

## Fixed inverse and source classification

Unknown is a17×17 spatial current-release fraction map. Observations are **existing original author simulation**, not measured laboratory data, from examples/published/iCSD_fig7/ObsData.txt. They were not generated or altered by the cleaned runtime. Target-independent known-medium calibration is VRTeSim.txt, the prescribed medium's unit-current response at289candidate nodes; VRTeCoord.txt supplies their coordinates. Matrix G incorporates the electrode sequence, return and boundary treatment. No target-derived conductivity estimate, hidden source shape or reconstructed array enters the public inputs. This is source imaging with known medium, not ERT conductivity inversion. Author simulation is used after investigating available real source materials; these exact self-contained files bind the verified published-method path.

Original author repository https://github.com/Peruz/icsd, remotely verified exact commit def56a90b3aee95587203f541f003676adf0a39d. Live response Peruz__icsd_remote.json and full unmodified source zip/checkout are private. Copyright2018 ICSD Developers and BSD3source notices retained in local mathematics, with complete LICENSE.md preserved as LICENSE. Source data are bundled in that author repository under its license; original figures and paper PDF are not redistributed in the task.

Paper Peruzzo et al.2020, *Imaging of plant current pathways for non-invasive root Phenotyping using a newly developed electrical current source density approach*, PlantSoil450567–584, DOI10.1007/s11104-020-04529-w. Exact19-page author-institution archived PDF, including eScholarship cover, downloaded from https://escholarship.org/content/qt4901v42x/qt4901v42x.pdf and stored private peruzzo2020.pdf. SHA256 362f62811d75493e861a3e1744448323a00be0b206e90d42eb1f6e1abe039046. Full text peruzzo2020.txt read, especially Electrical measurements, iCSD inversion Eq1/2, Synthetic tests, Fig3 and Fig7. Original publisher PDF request returned an HTML challenge; it is not the retained PDF. The institutional copy is the actual paper version read and hashed.

Paper explicitly describes204transfer resistances, normalized unit current, nonnegative bounded linear least squares, a **conservation weight1000**, and spatial-neighbor differences. Thus selected wc1000matches the paper; current source defaults wc100000differ and are separately compared. Source driver run_synth_nail.py describes a single off-grid point source like paperFig3, while folder name iCSD_fig7 refers to a different paper figure. The source naming is ambiguous and is not resolved by inventing a physical source map or calling it an exact eight-source-paper reproduction. We use unchanged author observation/calibration files, identify the ambiguity, and assess independently reproduced method output. No claim of a known true number/position of sources is made. Source truemodel.txt has204values identical to ObsData, so it is **not** spatial truth; it is not used as evaluation truth or calibration.

Archive root `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth`; icsd_archive_index.json contains byte counts and SHA256 for paper/text, complete checkout archive/live revision, capture, oracle arrays, commands, logs and actual kernel. Source-file SHA256:

```json
{
  "ObsData.txt": "79577229ace4b156e56b2ef6703970064530577c9762fa5fa2961cf3a461510f",
  "VRTeSim.txt": "667748936adeb7f4cd2cf3900aa648ff9ccbcd4606cf9e4312d9dea1465a5e0d",
  "VRTeCoord.txt": "4e7a7d4e2e83266c7b33024947721a0d376a44debe82a2e857f33200171651e5"
}
```

Public NPZ SHA256 997288ee0ee8b839079f71b17d7d7d74cf6f8a499b64e81c87107b2310120af8. Its arrays are unchanged numerical text values converted to float64, with exact source Fortran-order reshape G=flat.reshape((-1,289),order='F'). Candidate source order xslow,yfast,spacing.03m. Output Creshape17×17 axesx,y. Negative transfer resistances preserved because potentials are electrode differences; source fractions constrained nonnegative. No observation discard or noise addition.

## Independent original run before extraction

Private capture_icsd.py loads unmodified icsd/inversion/solve.py and regularize.py by importlib and executes their original check_nVRTe,reshape_A,obs_w_f,regularize_A/b/w,stack_A/b/w,weight_A/b and iCSD bounded-linear branch. This bypasses unrelated full-package plotting and mesh imports but does not modify the selected math/functions. No cleaned tests/runtime import the oracle source. Runs use exact bundled values, wc1000,wr1,obs_errsqrt,error floor1,no prior/no x0. Original branch calls scipy.optimize.lsq_linear bounds0..1,verbose0,standardTRFdefaults. The comparison original x0uniform branch is separately frozen and labeled least_squares, not misidentified as lsq_linear.

Original command/exit0 in icsd_oracle_command.json; all stdout/stderr in icsd_oracle.log. Runtime 1.836672s includes uniform-initial alternative. Python3.11.11,NumPy2.4.6,SciPy1.17.1 CPU1thread shared standard numerical primitives, no installed ICSD/pygimli/PyVista domain runtime. All frozen intermediate arrays precede extraction. Tolerances fixed before local scoring: exact equality for weights,graph,Aw,bw; reconstruction rtol1e-7/atol1e-9, predictions rtol1e-9/atol1e-7Ω. Measured maximum original/local source-fraction difference 0. Fresh sum 1.0006324393820427, KKT optimality 1.1887452711309775e-08, residual RMS 20.53663018021722Ω, original weighted RMS 1.8456366521831535. No post-fit current normalization.

## Traceability and discrepancy investigation

| Required stage | Paper and unmodified original | Local function | Validation |
|---|---|---|---|
| Known-medium superposition | Eq1, solve.reshape_A | physics_model.forward/adjoint | exact source calibration; linearity/adjoint identity |
| Relative data weights | solve.obs_w_f sqrt | preprocessing.observation_weights | frozen original weights; cap/zero/sign analytic cases |
| Unit-current conservation | Eq1/2 text uses1000, solve.con_A/b/w | physics_model.weighted_system | exact frozen stackedrows; current sum and KKT |
| Spatial graph | Eq2 roughness, regularize.regularize_A | physics_model.regularizer | exact source R; constantnullspace,rank/connectivity,physicalneighbor distances |
| Half-square objective and derivative | bounded linear objective | physics_model.objective_gradient | independent finite differences |
| Bound-constrained inverse | solve.iCSD nox0 branch | solvers.reconstruct | frozen complete original q and prediction |
| Standard metrics/contracts | declared output | standalone judge | known metricvectors, poor/invalid/evaluator/foreign-cwd tests |

Original regularize_A nests first-index edges inside second-index-neighbor condition, giving528rather than544nearest-neighbor edges. In actual physical coordinate order the omitted16edges lie along final-yboundary. This discrepancy from a full isotropic stencil is preserved and explicitly named rather than silently fixed. Independent graph eigenvalues show one constant nullspace and positive remaining eigenvalues, so the connected graph's quadratic remains physically admissible smoothing, albeit anisotropic at that boundary. All included edges have actual distance.03m. Notebook displays missing edges and computes a separately labeled complete-graph ablation; it is not substituted into the standard method. Original defaultwc1e5 yields poor optimizer optimality dueconditioning; selectedwc1000is explicitly stated by paper and improves numericalKKT. Both are actively compared on unchanged data.

## Reference, metric and notebook provenance

evaluation/baseline_reference.npz contains original method's source fractions, **not** spatial truth. No ground_truth.npz created. notebooks/reference_outputs/upstream_measurements.npz identifies original selected fractions,prediction,and separately initialized original nonlinear comparison; consumers are notebook comparisons and source provenance. test/fixtures/upstream_stages.npz is consumed by independent stage and full-pipeline tests. No unused standalone measurements or hidden target generation seed is public.

Cosine NCC uses no mean subtraction; NRMSE is RMS error/reference dynamic range. No alignment,renormalization or mask. Nonzero method-agreement boundaries NCC.99999 and NRMSE1e-4 selected before local evaluation are far above reproducibility error and reject zero/poor maps. Reference agreement is distinct from source localization accuracy. All quality metrics for standard method compare exact same calibrated data and absolute scales.

Six-chapter notebook actively solves exact observations, compares independently run original fields, computes residualmaps/curves, source-graph eigenmodes and responseSVD, recomputes original defaultwc and altered graph comparisons, and performs nine full spatial regularization refits. Historical paper figures are contextual only. Every plotted fresh reconstruction is computed. Actual kernel printed and saved the executable/backend/library environment; source libraries do not select Matplotlib backend. Canonical executed NB retains outputs. Private temporary NB and six PNG/SVG/PDF groups plus numeric_sources.npz remain outside task.

Actual commands:

```json
[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\soil_electrical_current_source_imaging",
    "runtime_seconds": 0.7568193999904906,
    "exit_code": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\soil_electrical_current_source_imaging",
    "runtime_seconds": 0.17151230000308715,
    "exit_code": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\icsd_pytest"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\soil_electrical_current_source_imaging",
    "runtime_seconds": 1.8900858999986667,
    "exit_code": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "jupyter",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/soil_electrical_current_source_imaging.ipynb",
      "--output",
      "icsd_executed.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth",
      "--ExecutePreprocessor.timeout=180"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\soil_electrical_current_source_imaging",
    "runtime_seconds": 9.123551500000758,
    "exit_code": 0
  }
]
```

Actual kernel:

```json
{
  "executable": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
  "os": "Windows-10-10.0.26200-SP0",
  "numpy": "2.4.6",
  "numpy_file": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Lib\\site-packages\\numpy\\__init__.py",
  "scipy": "1.17.1",
  "matplotlib": "3.11.1",
  "backend": "module://matplotlib_inline.backend_inline",
  "threads": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  }
}
```

All four commands exited0;12tests pass. Notebook timeout180s, actual 9.124s. CPU1thread noGPU. Only NumPy,SciPy,Matplotlib pinned shared dependencies; no domain libraries, installation or shared upgrades. pytest --basetemp private due Windows default-temp permission behavior, not test weakening. Independent Judge checks required float64shape,keys,complex/object/nonfinite/missingreference/reportwriteerrors, valid poor mapfailure,foreigncwdpath resolution and known metric arithmetic. Standard entrypoint and fresh Judge pass.

## Final review gates

Passed: exact paper/live source/data hashes and equation mapping; original bound-observation execution; independent stages/pipeline/physics; source constants/order/coordinates; CLI/Judge; adversarial and evaluator contracts; visible input sufficiency; no hidden truth/calibration/GitHub URL leak; explicit local forward/objective/graph/inverse;12tests; complete active NB/comparisons; metrics consistent acrossJudge/notebook; semantic review of both plans against actual signatures, graph nesting, units, signs and execution; license/inventory/privatearchive cleanup. Visual review recorded in private icsd_visual_review.json before final handoff. Source graph/sample-name ambiguities are documented limitations rather than falsely claimed resolved historical truth.

## Independent rerun and numerical tolerance audit

Original unmodified capture and clean CLI were independently re-executed in new Python processes; both exit0. Full private commands/logs/arrays in icsd_independent_rerun.log/.npz, icsd_clean_independent_rerun.log/.npz and icsd_tolerance_reruns.json. First and second local outputs each compared against the independently rerun original arrays, rather than self-comparing one saved method reference. Deterministic CPU reruns agree to the measured errors below. Since exact reproducibility does not imply a meaningful positive numerical budget, float32 serialization then promotion was separately measured as a precision stress (not an accepted dtype or a new standard result). The unchanged thresholds allow modest differences beyond this quantization floor while remaining orders of magnitude below spatially wrong/uniform/zero maps; all three adversarial submissions actually return exit1. No tolerance was relaxed. The absolute per-quantity scale is retained.

```json
{
  "reconstruction": {
    "first_clean": {
      "ncc": 1.0,
      "nrmse": 0.0,
      "max_absolute_error": 0.0
    },
    "second_clean": {
      "ncc": 1.0,
      "nrmse": 0.0,
      "max_absolute_error": 0.0
    },
    "float32_roundtrip": {
      "ncc": 1.0,
      "nrmse": 2.205633428442166e-09,
      "max_absolute_error": 3.3560421386358286e-09
    },
    "zero": {
      "ncc": 0.0,
      "nrmse": 0.10854045408750194,
      "max_absolute_error": 0.13934381527532297
    },
    "uniform": {
      "ncc": 0.2289272038300303,
      "nrmse": 0.10565800440532279,
      "max_absolute_error": 0.13588141929130207
    },
    "spatially_reversed": {
      "ncc": 0.1683180057970895,
      "nrmse": 0.13998621838223008,
      "max_absolute_error": 0.13934381527532297
    },
    "nrmse_threshold_over_float32_roundtrip": 45338.449585718234
  }
}
```
