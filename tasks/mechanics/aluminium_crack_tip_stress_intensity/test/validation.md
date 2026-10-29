# Provenance and validation

## Scope, source and data

The selected inverse is the direct Williams displacement fit on the complete author nodemap `Dummy2_WPXXX_DummyVersuch_2_dic_results_1_52.txt` (44581 valid points). This is an author-provided measured ARAMIS nodemap, not newly simulated data, and is not claimed to be a particular published figure. Its header identifies Dummy2, June 2021, ARAMIS 12M, stage52/reference1, camera geometry and facet settings copied to README. No exact alloy designation is invented; the author's aluminium elastic calibration is E72000MPa, nu.33.

Paper: Strohmann et al., Next generation fatigue crack growth experiments of aerospace materials, Scientific Reports14,14075,18June2024, DOI10.1038/s41598-024-63915-x. Primary full article verified via Nature/PMC (https://pmc.ncbi.nlm.nih.gov/articles/PMC11189514/). Modern code is CrackPy1.4.0, not claimed identical to the historical1.1.1 publication package. The paper explains Williams displacement fitting and separate integral/CJP analyses. Only the displacement-only Williams direct case is selected; integral methods need extra strain/stress fields not supplied here. Foundational Williams1957 DOI10.1115/1.4011454.

Repository https://github.com/dlr-wf/crackpy, remotely verified exact revision `f5178fd4248adca457238ab55551444c1c7ae5ee`. Source archive SHA256 `26b6a0bb7a0ff8658691a853e21825ce181e31f8d39a043b188afed2a4a89125`,35628456bytes. Author source and data are distributed under MIT; LICENSE copied unchanged. Nodemap SHA256 `0c91e5613a260b9b5317ccbe525609b6b933bc5589264caee88c96f668cee352`. The known right-tip geometry comes from author's `test_data/crack_info_by_nodemap.txt`; the fitting calibration/material is the native case, not estimated from hidden stress.

Private archive base is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch9_20261026/math_mechanics`. `crack/archive_index.json` indexes source checkout, original archive, capture script, logs, evidence and review figures with bytes and SHA256. `crack/artifact_hashes.json` records every packaged numeric fixture, observation and evaluation asset hash. A Nature PDF request returned an HTML anti-bot page (`paper_download.bin`); it is explicitly not a paper PDF. Paper text was reviewed through primary article web access; no claim that this failed download is a PDF. No source checkout, PDF, archive or original log is shipped in the task.

## Independent original run and frozen fixtures

Before local extraction, `crack/capture_original.py` executed the unchanged original numerical modules on the exact full nodemap: InputData parsing and transform, Optimization interpolation/system assembly, direct GELSS displacement fit and original Williams stress-field evaluation. Runtime1.670737200009171s, exit0; source command and evidence in `original_evidence.json`, successful log `original_run_2.log`. The earlier failed import log `original_run.log` remains identified as failed, not evidence of completion. Package initializers were bypassed solely to avoid unrelated YAML logging and VTK export imports. A dummy pyvista.CellType is present but no VTK numerical/export function is called; numerical modules themselves remain unchanged. This isolated oracle setup is neither installed in nor imported by the cleaned runtime.

Original complete stages were saved in `crack/original_stages.npz`; frozen task fixture `test/fixtures/upstream_stages.npz` contains those numerical outputs, not local results. Input copies are omitted from that fixture because exact observations are already public. Upstream intensity is the source coefficient-to-quantity conversion of original coefficients, not a new stress measurement. `crack/preimplementation_criteria.json` fixed tolerances before extraction. Full stress and coefficients remain independent original outputs.

Fixture consumer is exclusively `test/comparing_test/test_upstream.py`: transformed coordinates/vectors, evaluation grid, interpolation, full258258x12 matrix, target, coefficients, residual, stress and intensity. `notebooks/reference_outputs/upstream_measurements.npz` contains original stress/intensity/runtime for the active notebook. `evaluation/baseline_reference.npz` contains the same identified method result for independent Judge. These three copies have different documented consumers; no ground_truth.npz exists. Method agreement is not physical truth.

## Equation, implementation and validation traceability

| Selected mathematical stage | Original file/function | Local implementation | Independent validation |
|---|---|---|---|
| Translation and vector rotation | input/input_data.py, InputData.transform_data | preprocessing.transform | Full transformed fixture; norm/zero-tip check |
| Tip-rebased scattered interpolation | fracture_analysis/optimization.py, Optimization initialization | preprocessing.sample_displacements | Full original interpolation fixture,1e-12mm |
| Williams displacement eigenfields and plane stress | fracture_analysis/crack_tip_fields/williams/solutions.py and optimization system assembly | physics_model.displacement_basis, assemble_system | Full system1e-14 tolerance; independent Cartesian finite-difference Hooke check |
| Direct least-squares inversion | Optimization.optimize_williams_displacements_xy(solver='direct') | solvers.reconstruct, scipy.linalg.lstsq GELSS | Original coefficients1e-7absolute/1e-8relative, residual1e-12mm; adjoint and directional derivative |
| Analytical elastic stresses | williams_in_plane_stress_field | physics_model.stress_field | Full original stress1e-6MPa/1e-8relative; traction-free all12basis fields |
| Singular coefficients, sign and mm/m units | williams/quantities.py | physics_model.stress_intensity | Original quantities1e-8; independent phi0 singular check |
| NCC and dynamic-range NRMSE | Independent task protocol, not relabelled historical paper metrics | evaluation/judge.py, visualization.metrics | Known vector case; strict NPZ errors and poor predictions |

The formulas, derivative signs, conversion sqrt1000 and full-grid ordering were semantically checked against original code and independently against Hooke's law. No suspected original numerical error was found in this selected case. The direct method, discretization, default cutoff and GELSS driver are preserved. Rank12 is required; no regularization or hidden gain correction is introduced.

## Environment, fresh results and thresholds

Cleaned and oracle numerical runtime: shared research-runtime Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, Windows CPU, threads1. `crack/environment.json` records OS, executable and exact NumPy BLAS/LAPACK backend. No GPU is needed; no upstream domain package is installed/imported by cleaned source, Judge, tests or notebook. Minimal requirements list only actual runtime packages. Import/call-role audit in `crack/import_role_audit.json`: SciPy only generic Delaunay interpolation and GELSS, NumPy explicit eigenfields/operators, Matplotlib plotting; no domain inverse shortcut. Tests consume fixtures without importing original modules. No cross-task imports, no hidden reads in src.

Original maximum absolute coefficient deviation after extraction3.98e-13; stress1.5347723092418164e-12MPa; intensity3.2e-14MPa sqrt(m). Fresh full notebook metrics and comparisons follow; all are newly computed, not paper historical reports:

```json
{
  "fresh_runtime_seconds": 0.37209670001175255,
  "fresh_intensity": [
    7.179205815846947,
    3.477528644646307
  ],
  "original_intensity": [
    7.179205815846979,
    3.4775286446463105
  ],
  "maximum_stress_parity_error_mpa": 1.5347723092418164e-12,
  "full_displacement_rmse_mm": 0.0008683199250469595,
  "singular_only_displacement_rmse_mm": 0.00210350236722638,
  "singular_only_intensity": [
    9.525903173572987,
    0.3009414126779167
  ],
  "noise_sigma_mm": 0.0001,
  "noise_stress_relative_l2": 0.032260489029335294,
  "noisy_intensity": [
    7.31013018495215,
    3.6464997448019627
  ],
  "judge_metrics": {
    "reconstruction_ncc": 1.0000000000000002,
    "reconstruction_nrmse": 9.551964330154104e-16,
    "stress_intensity_ncc": 1.0,
    "stress_intensity_nrmse": 6.120596838322759e-15
  }
}
```

The reference supplies no independent GT, so mechanical .9/1.1 self-comparison thresholds would be zero/vacuous. Instead NCC>=.999999 and dynamic-range NRMSE<=1e-6 separately on each complete required array allow float32 serialization but reject1%gain, zeros, sign inversion and huge finite outputs in either stress or intensity. Original and fresh K=[7.179205815846979,3.4775286446463105] MPa sqrt(m) agree to floating precision. Observation residual is reported as a model-fit diagnostic, not an additional output-derived truth. NRMSE denominator is each entire reference array's max-min; NCC has no mean subtraction or alignment. No cropping/rescaling is applied.

## Executed checks and review

`crack/final_execution.json` records exact commands, runtimes and native exits; stdout/stderr retained as `{main,judge,pytest}_final.*`. All standard commands executed from the task:

* `python -m src.main`: exit0, fresh two-key reconstruction saved.
* `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`: exit0, pass.
* `python -m pytest test/ -v --basetemp <private crack/pytest_tmp_final>`: exit0,38tests passed. Default C:pytest temp parent has an existing access denial; the explicit maintainer-only D:basetemp resolves this environmental problem. The initial18test-setup failures are not claimed as successful tests.
* `python -m jupyter nbconvert --to notebook --execute notebooks/aluminium_crack_tip_stress_intensity.ipynb --ExecutePreprocessor.kernel_name=batch9math --ExecutePreprocessor.timeout=180 --output crack_executed.ipynb --output-dir <private crack>`: native exit0, no cell errors. The privately provisioned kernel points to the same shared Python; notebook metadata remains portable Python3. Windows ZMQ selector-thread/TCP warnings are recorded and do not affect numeric execution.

Executed notebook retained privately at `crack/crack_executed.ipynb`, six PNG outputs and contact sheet at `crack/review_figures/`; all six figures visually inspected. Observation maps have physical labels and calibrated tip; stress components and original-difference panels are complete; parity errors are explicitly ~1e-12MPa; measured residual distribution is nonzero; ablation and robustness bars have fresh nonzero computed values. No zero-filled method, hidden GT claim or cache replacement is present. Temporary executed notebook is outside task output. The active source notebook writes standard submission and invokes standalone Judge.

README and both plans reviewed against actual source signatures, constants, array shapes, operator signs, units and side effects. Acquisition constants checked by executable tests, known metric case shared with independent numerical expectation. Public materials scanned for forbidden source URLs/hidden asset references/control characters. Source code retains MIT notice. Generated Python/pytest caches are excluded from handoff.

## Completion gates

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/revision/data/license/equation mapping | passed | Primary article, source_revision, archive/hash indexes and table above |
| Original exact full-data run, independent fixtures frozen first | passed | original_evidence, successful original log, preimplementation_criteria |
| Full local stage parity and independent elasticity | passed |38tests, analytical/finite-difference/traction/unit checks |
| Standard CLI and strict independent two-quantity Judge | passed | main/judge final logs; malformed and poor-output cases |
| Visible observations/calibration only, self-contained README | passed | Input/hash/allowlist audit; no hidden stress/calibration or source URL leak |
| Minimal local CPU runtime without domain package | passed | environment/import-role audit and full commands |
| Active full notebook and all comparisons inspected | passed | Native nbconvert0, six figures and fresh measurements |
| Plans/README and package structure consistency | passed | Semantic source review, constants tests, final inventory |

Parent independent acceptance remains separate from this worker verification.


## Parent review update: signed maps and replicated sensitivity

All signed displacement/stress/difference maps now use symmetric zero-centred diverging limits; original-versus-fresh and representative noise-difference stress maps use the same physical radius(mm)/azimuth(rad) axes as the fresh stress map. This corrects the misleading zero-colour in a one-sided yy-difference map without changing arrays, reconstruction or thresholds. The leading-order ablation and all six previous scientific comparisons are retained.

The active notebook additionally performs eight complete fresh reconstructions: four independent Gaussian perturbations at each of0.0001mm and0.0005mm, seed20261027, over every measured displacement component. It reports KI/KII spread, observation-fit RMS and stress relativeL2 change. These are hypothetical sensitivity experiments, not calibrated uncertainty. The earlier single representative perturbation remains separately identified.

Updated native nbconvert exit0 in12.289394900028128s, record `crack/notebook_review_execution.json`, logs notebook_review.stdout/stderr. Executed notebook `crack/crack_review_fixed.ipynb`, seven images plus `review_figures/contact_review.png`; all seven inspected after the update. Figure07contains all eight independently computed draws and unperturbed reference;01/02/03/06mapcolour/coordinates inspected. Latest fullprecision fresh measurements:

```json
{
  "fresh_runtime_seconds": 0.37680720002390444,
  "fresh_intensity": [
    7.179205815846947,
    3.477528644646307
  ],
  "original_intensity": [
    7.179205815846979,
    3.4775286446463105
  ],
  "maximum_stress_parity_error_mpa": 1.5347723092418164e-12,
  "full_displacement_rmse_mm": 0.0008683199250469595,
  "singular_only_displacement_rmse_mm": 0.00210350236722638,
  "singular_only_intensity": [
    9.525903173572987,
    0.3009414126779167
  ],
  "noise_sigma_mm": 0.0001,
  "noise_stress_relative_l2": 0.032260489029335294,
  "noisy_intensity": [
    7.31013018495215,
    3.6464997448019627
  ],
  "noise_ensemble": [
    {
      "sigma_mm": 0.0001,
      "replicate": 0,
      "KI": 7.266018863655649,
      "KII": 3.547725573267938,
      "displacement_rmse_mm": 0.0008665193643096413,
      "stress_relative_l2_vs_unperturbed": 0.018469778445294377
    },
    {
      "sigma_mm": 0.0001,
      "replicate": 1,
      "KI": 7.49026989704171,
      "KII": 3.64373188927031,
      "displacement_rmse_mm": 0.0008629372189453335,
      "stress_relative_l2_vs_unperturbed": 0.04799381340113485
    },
    {
      "sigma_mm": 0.0001,
      "replicate": 2,
      "KI": 7.329461808047367,
      "KII": 3.7872468361084035,
      "displacement_rmse_mm": 0.0008741841621957574,
      "stress_relative_l2_vs_unperturbed": 0.04666182592002213
    },
    {
      "sigma_mm": 0.0001,
      "replicate": 3,
      "KI": 7.265923105685445,
      "KII": 3.7581083112106186,
      "displacement_rmse_mm": 0.0008722083701276012,
      "stress_relative_l2_vs_unperturbed": 0.037010524505873596
    },
    {
      "sigma_mm": 0.0005,
      "replicate": 0,
      "KI": 8.363529020078484,
      "KII": 2.5382194369030846,
      "displacement_rmse_mm": 0.000814016093601558,
      "stress_relative_l2_vs_unperturbed": 0.23282224585816605
    },
    {
      "sigma_mm": 0.0005,
      "replicate": 1,
      "KI": 7.068726281617483,
      "KII": 3.408738471637596,
      "displacement_rmse_mm": 0.0009883182670106373,
      "stress_relative_l2_vs_unperturbed": 0.06466351899105242
    },
    {
      "sigma_mm": 0.0005,
      "replicate": 2,
      "KI": 6.284490607562663,
      "KII": 3.50664000261376,
      "displacement_rmse_mm": 0.0009655071722626787,
      "stress_relative_l2_vs_unperturbed": 0.1165024283194531
    },
    {
      "sigma_mm": 0.0005,
      "replicate": 3,
      "KI": 8.033857874958965,
      "KII": 3.666373839604297,
      "displacement_rmse_mm": 0.0008844017495775121,
      "stress_relative_l2_vs_unperturbed": 0.16693240015668925
    }
  ],
  "judge_metrics": {
    "reconstruction_ncc": 1.0000000000000002,
    "reconstruction_nrmse": 9.551964330154104e-16,
    "stress_intensity_ncc": 1.0,
    "stress_intensity_nrmse": 6.120596838322759e-15
  }
}
```

Model/physics/Judge/testsource unchanged; the previously validated38tests remain applicable. Source notebook uses local source on full public data; no cached reconstruction or zero-filled comparison. Plans updated for additional calls and display conventions. `crack/post_parent_review_hashes.json` records updated package hashes.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 7 computed PNG outputs, 36 formulas. Private evidence: research_batch9_20261026/acceptance/aluminium_crack_tip_stress_intensity/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
