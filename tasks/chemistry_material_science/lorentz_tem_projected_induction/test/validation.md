# Provenance and validation: Lorentz TEM projected induction

Status: ready for independent parent review. All mandatory gates below passed; no independent actual truth is available. This task's references are explicitly standard-method estimates. No prior task was edited.

## Bound scope and sources

Paper: McCray, Cote, Li, Petford-Long and Phatak, Physical Review Applied15,044025(2021), DOI10.1103/PhysRevApplied.15.044025. The full accepted manuscript (10 pages plus references) was read, particularly magnetic/electrostatic phase separation around Eq.13, integrated induction discussion, Tikhonov Eq.14 and AppendixA propagation. Archived PDF: https://www.osti.gov/servlets/purl/1813111. The original paper's Pt/Co/W experimental specimen is not relabeled as this task's specimen. The authors' official2024 TIE demo identifies the bound data as an artificial-spin-ice island structure, demonstrating the same published TIE method. The separate CGT single-image SITIE acquisition is excluded and explained in the notebook.

Original code: https://github.com/PyLorentz/PyLorentz, remotely API-verified exact commit9a7884af7c5685463ca6953bbf9d5d6110562c69, BSD3Clause; original LICENSE preserved. Data: official author-linked Zenodo13147848, version0.1.0(2024-08-01), DOI10.5281/zenodo.13147848; API explicitly states CC BY4.0. Download222454225-byte example_data.zip (published md5 097dcd560b136bea904903949cf82bf7). Full source checkout, archive, TIFFs, paper, captures and logs remain private.

Original acquisition arrays are two7×2048×2048 float32 aligned stacks. Bound arrays are native center rows768:1280,columns768:1280, no resizing. All seven focus planes remain public for meaningful sensitivity evaluation. Pixel pitch5.1066 nm, 200000 V, signed lens defocus[-928972,-398131,-199065,0,199065,398131,928972] nm, directly from author tfs_mdata.json. No target-derived calibration/mask/truth is public. The inversion uniform mask and periodic boundary are explicit choices, not independently measured exterior fields. ∫Bdz is T·nm; no thickness is supplied and no local B or magnetization is claimed.

## Upstream-first runs and backend investigation

Unchanged original ThroughFocalSeries(...use_mask=False,simulated=False).preprocess(hotpix=False), TIE(...sym=False).reconstruct(index=0,sym=False) ran first on the exact bound native ROI; runtime approximately1.7 s. Captures froze original stages, all three focus pairs, qc0.0002/0.0005/0.001 and even extension. Neither cached notebooks nor cleaned solver outputs established the fixtures. The source checkout was never patched. Private capture scripts only handle IO, call original functions and save numeric arrays.

The first isolated oracle used NumPy1.26.4/SciPy1.10.1, as required by another upstream verification environment. Compared with shared NumPy2.4.6 float32 sums over~4e8 intensities changed by32 or64 (1–2 ULP), producing phase differences up to3.62e-6 rad and induction up to5.99e-6 T·nm. A private diagnostic froze raw stacks, image totals and gain factors and proved the discrepancy originated in reduction backend, not acquisition equations. Those measurements remain separately archived as lorentz_numpy126_original*. The unchanged original was then independently re-executed with the exact cleaned NumPy2.4.6/SciPy1.17.1 backend and compatible isolated scikit-image0.26 imports. This produced exact phase equality and induction differences<6e-13 T·nm. It was not a change to the published method or manufactured fixture. Strict stage tolerances fixed before local scoring remain relative2e-10/absolute2e-9 for double outputs; float32 preprocessing is checked bit for bit.

Commands, logs, environments and file hashes are in the private archive index below. All actual successful source captures exited0; original warnings were dependency/deprecation messages, not numerical failures. Source CPU execution uses one thread; no GPU, domain package installation into shared runtime, or shared package upgrade occurred. Captures used no stochastic source algorithm, training, or simulation seed. All benchmark observations are existing experimental data.

## Equation / original / local / independent-test traceability

| Mathematical stage | Exact original source/function | Local executable stage | Validation |
|---|---|---|---|
| Focal-series selection, common I0 and gain normalization | dataset/through_focal_series.py:infocus; phase/tie.py:_select_images,_scale_stack | preprocessing.prepare | test_original_preprocessing_and_filter; exact frozen selected stack |
| Flip half-difference/half-sum and zero-mean source | phase/tie.py:_get_derivatives; paper Eq.13 separation discussion | preprocessing.prepare; solvers.reconstruct mirrored branch | same stage test; test_flip_reversal_swaps_magnetic_sign_only |
| Relativistic wavelength and double inverse Laplacian prefactor | phase/base_tie.py:_pre_Lap | physics_model.wavelength_nm; solvers.reconstruct_phase | frozen scalar; independent relativistic energy-momentum test |
| Reciprocal inverse, Tikhonov Eq.14, circular cutoff | phase/base_tie.py:_make_qi,_reconstruct_phase | physics_model.frequency_filters | exact frozen qi; all qc pipeline fixtures |
| Periodic centered gradient/divergence and Teague inverse | phase/base_tie.py:_reconstruct_phase | physics_model.periodic_gradient; solvers.reconstruct_phase | source pipeline parity; discrete adjoint and constant-mode tests |
| Lens defocus sign and physical scale | utils/microscopes.py:_get_chiQ,get_transfer_function; paper AppendixA | physics_model.propagate,defocus_derivative | independent coherent weak-phase sinusoid: correct sign and sinc² centered-gradient response |
| Even boundary extension and crop | phase/base_tie.py:_symmetrize; phase/tie.py:reconstruct | physics_model.even_extension; solvers.reconstruct | symmetric original fixture, including float64 promotion before differences |
| Separate minimum-zero phase gauges | phase/tie.py:reconstruct | solvers.reconstruct_phase,reconstruct | all phase fixtures; uniform phase gauge test |
| Phase to ∫Bdz | phase/base_phase.py:induction_from_phase | physics_model.phase_to_induction | linear phase exact units/sign; divergence-free rotated gradient; independent Judge consistency |
| Direct standard execution | phase/tie.py:TIE.reconstruct | src.main.main | pipeline entry point and foreign-CWD Judge tests |

The source returns (By,Bx), despite the physical naming convention; local phase_to_induction explicitly returns(Bx,By), and frozen arrays prove correct output association. Source gradients before gauge fixing versus local gradients after fixing differ only at roundoff. The source's circular cutoff and centered stencils remain unchanged, not replaced with a more convenient spectral inverse. Its nonuniform-I0 auxiliary-potential approximation is documented rather than claimed exact.

## Reference, metrics and limitations

evaluation/baseline_reference.npz is the independent exact-backend original method result, four batch-first float64 arrays. No ground_truth.npz exists. notebooks/reference_outputs/upstream_methods.npz contains named original method/parameter comparisons. test/fixtures/upstream_stages.npz contains captured input, preprocessing, filter and original final/comparison quantities; duplicate original dIdZ copies were omitted from the shipped fixture because difference_B/E are the documented consumers. The remaining original input arrays are consumed by raw-data equality tests. Notebook, fixture and evaluator arrays have distinct consumers and identified provenance; none is a newly synthesized observation.

NCC is cosine without centering; NRMSE is RMS error/reference dynamic range. Known numerical cases independently verify metric definitions. Main thresholds per required quantity are NCC≥0.999999, NRMSE≤1e-6, a nonzero tolerance supported by the separate original backend run. There is no crop/gain/translation/alignment. Judge prioritizes actual truth if present, never falls back from malformed truth, and currently uses method reference only.

The independent spectral TIE forward residual relative L2 is0.40443195160816003 for magnetic and0.2413322995439095 for electrostatic contrast; boundaries are fixed1.05 times these original-standard measurements. No arbitrary intensity/contrast fit is applied. These sizeable residuals are prominently plotted and explained by finite defocus, physical bandwidth loss, centered-gradient attenuation, periodic cropping and Teague approximation. They are not presented as physical truth error. Reference parity alone is not proof of quantitative specimen accuracy. Additional relative induction/phase consistency boundary1e-10 prevents independently inconsistent output arrays. Every output and every declared metric must pass. Malformed submissions produce empty metrics/status error, with no NaN/Infinity JSON.

## Active notebook, figures and semantic review

The canonical notebook actively runs the full bound standard method and six comparison settings using cleaned code. Five scientific figure groups cover actual raw focal pairs, separated phases and integrated induction/vector direction, original/local comparisons, independently calculated observation residuals, and focus/regularization/boundary sensitivity. Every group is exported as actual PNG/SVG/PDF with exact numeric_sources.npz, full-precision metrics.json and SHA manifest in the private figure directory. All five PNGs were opened and visually reviewed for units, coordinate direction, residual scales, colorbars, readability and honest limitations. Quiver uses increasing row y downward, consistent with image axes. Zero actual parity residuals are computed zeros, explicitly labeled max0, not placeholders for missing methods. Historical paper metrics are not substituted for this run.

Source callable signatures, array shapes, default parameters, constants, units, sign, gradient roles and actual caller flow were compared semantically against README and both plans. No optimizer/training/stopping procedure is claimed for the direct inverse. Optional helpers are identified. Import/call audit covers source, Judge, tests and notebook: only general array/FFT/constants/plot/file primitives, local math, no imaging reconstruction library or cross-task import; no library backend mutation. README/public requirements have no GitHub URLs; data keys are legitimate raw observations. Repeated execution used a private kernel specification bound to the shared tested Python; an initial exploratory global Python3.12 kernel run was superseded and is not the final compatibility evidence.

## Gate evidence

All mandatory gates PASSED: paper/revision/data hashes/mapping; exact-bound original and cleaned CPU execution; independent upstream stages and physical sign/scale/adjoint checks; constants/units/precision; fresh CLI and independent Judge; poor/malformed/overflow/config/reference/report-write handling; public sufficiency/no leakage; complete README and batch-first NPZ contract; minimal shared dependencies; local mathematics/call-role audit; complete unit/parity/physics/pipeline tests; active executed notebook with five actual figures; result/metric provenance; both plans semantically matched to implementation; exact template, licenses and private archive inventory. No failed/unverified mandatory gate remains. Independent absolute accuracy is explicitly unavailable, not a failed execution gate.

Private archive root: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch5_20261021\material_physics
Archive index: lorentz_archive_index.json. Every entry records SHA256 and available byte size; complete exact source checkout is pylorentz_source/PyLorentz-9a7884af7c5685463ca6953bbf9d5d6110562c69.

Bundled raw_data.npz SHA256 1f17e012ed9063f0161bae1211c603f69f689d6b060aabdc627b3784981870fe; baseline_reference.npz SHA256 ec88cd6e3558fcfbc41561f42114ba18321fea466f6230184c8415a9934d66db; upstream fixture SHA256 93514ee6d6567f755ed7ef67b4202f10292905a93b1aab772d175e159b2b7386.

Final CLI/Judge/notebook command records:
```json
[
  {
    "command": [
      "D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics\\lorentz-pytest-final"
    ],
    "exit_status": 0,
    "seconds": 13.55189120000432,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics\\lorentz_tests_final.log"
  },
  {
    "command": [
      "D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe",
      "-m",
      "src.main"
    ],
    "exit_status": 0,
    "seconds": 0.7026482000001124,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics\\lorentz_cli_final.log"
  },
  {
    "command": [
      "D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "exit_status": 0,
    "seconds": 0.40371500000037486,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics\\lorentz_judge_final.log"
  },
  {
    "command": [
      "D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe",
      "-m",
      "jupyter",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/lorentz_tem_projected_induction.ipynb",
      "--ExecutePreprocessor.timeout=240",
      "--output",
      "lorentz_executed_final.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics"
    ],
    "exit_status": 0,
    "seconds": 44.37389350000012,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\material_physics\\lorentz_notebook_final.log"
  }
]
```
Canonical executed notebook audit:
```json
{
  "execution_counts": [
    1,
    2,
    3,
    4
  ],
  "inline_png_count": 5,
  "errors": 0,
  "source_cells_identical": true,
  "sha256": "33de63de0519e96751437178937e3a4455d594c7a97243b0d8f519ad30aa69c9"
}
```
Final fresh Judge measurements:
```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "phase_B_ncc": 1.0,
    "phase_B_nrmse": 0.0,
    "phase_E_ncc": 1.0,
    "phase_E_nrmse": 0.0,
    "Bx_ncc": 0.9999999999999998,
    "Bx_nrmse": 3.6782169715487465e-16,
    "By_ncc": 1.0000000000000002,
    "By_nrmse": 3.1757243048844493e-16,
    "observation_B": 0.4044319516081601,
    "observation_E": 0.24133229954390936,
    "induction_consistency": 0.0
  },
  "thresholds": {
    "phase_B_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "phase_B_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "phase_E_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "phase_E_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "Bx_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "Bx_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "By_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "By_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "observation_B": {
      "comparison": "<=",
      "boundary": 0.42465354918856807
    },
    "observation_E": {
      "comparison": "<=",
      "boundary": 0.25339891452110497
    },
    "induction_consistency": {
      "comparison": "<=",
      "boundary": 1e-10
    }
  },
  "errors": []
}
```
Latest full suite after final independent report-write/raw-input/prefactor tests: **38 passed** (private final38 log). No scientific source or notebook source changed after that final test hardening.


## Parent acceptance review

Parent independently ran38tests (38passed12.72s), strict inventory/public-array/call-role audit errors[] and foreign-cwd standaloneJudge exit0. Read complete README/both plans, explicit gain normalization, flip half-sum/half-difference, relativistic wavelength, aperture/regularized double inverse Laplacian with periodic centered gradients, minimum-zero gauge and physical induction mapping, plus independent evaluator and detailed source provenance. Opened all5 actual scientific groups: real aligned native focal pairs; separated magnetic/electrostatic phases with both integrated induction components and directions; exact original/local comparisons; complete independent spectral transport residuals; actual focus separation, Tikhonov and even-boundary sensitivity on shared physical scales. Four output arrays are independently scored; projected integral B dz in Tnm is never mislabeled local B without thickness or magnetization. Full original real-data run is independent and exact-bound; originalNumPy1.26 reduction differences were traced to1-2ULP float32 accumulation, preserved, and original code reexecuted under same modern backend without loosening predeclared tolerances. Main phases agree bitwise and induction below6e-13Tnm. Independent coherent weak-phase propagation verifies lens-defocus sign and normalization, including source centered-gradient sinc-squared attenuation; adjoints, flip symmetry, projected units and divergence-free gradient are checked. All11metrics include observation closure and phase/induction consistency with nonzero reference tolerances supported by independent backend variation. Sizeable .4044/.2413 transport residuals are shown and limited by finite-focus/periodic-boundary/bandwidth/nonuniform-intensity approximation, not claimed truth accuracy.38tests cover original stages and7parameter settings, native pipeline, poor/malformed/extreme outputs, strict configuration, hidden-truth priority and report failures. Complete fresh CLI/Judge and active notebook44.374s/5PNG/noerrors recorded. The actual specimen differs from original paper PtCoW and is explicitly identified as the official author experimental spin-ice demo. BSD3 code and independently CC BY4 data licensing preserved. Semantic factors, signs, units, numerical promotion, endpoint versus periodic gradient roles, defaults/signatures/caller flow agree with plans.
