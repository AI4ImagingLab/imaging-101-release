# Authoritative provenance and validation

## Binding and semantic scope

Status: all mandatory cleaning gates passed locally; parent independent acceptance remains separate. Original exact-data runs, independent frozen physics/parity, local CLI/Judge, full tests, active native Notebook and actual figure/export review are identified below.

The task estimates complete native32x32 pyruvate→lactate conversion-rate kPL and initial longitudinal lactate L0 fields from25 paired real metabolite-image frames. It is different from undersampled MRI k-space reconstruction, ordinary MRS spectral amplitudes, DCE vascular permeability, ASL perfusion and static PET activity. The acquisition physics is hyperpolarized two-pool exchange with RF consumption and measured dynamic substrate forcing. Complete1024 pixels and both independently active unknowns are retained. No arbitrary crop, output-conditioned mask or hidden target calibration is used. There is no independently measured parameter truth; baseline_reference is explicitly a conditional method result.

## Primary paper and version

Method: Larson et al., Investigation of analysis methods for hyperpolarized13C-pyruvate metabolic MRI in prostate cancer patients, NMRBiomed31:e3997(2018), doi10.1002/nbm.3997. The complete31-page author manuscript was actually parsed from the institutional public URL https://escholarship.org/content/qt98b2x2fj/qt98b2x2fj.pdf?t=rv3wa0 (945 parsed lines). Full Methods/Eqs2–14, results/discussion and figure/table captions were read, with parent independent full-response/4segment capture. Private `parent_method_pdf_web_capture.json` and `method_primary_web_capture.json` retain actual responses. The direct command-line PDF download returned HTML, so no local PDF-byte hash or PDF-download success is claimed. A library thesis explicitly disallowing automated retrieval was not fetched. No restricted route or author contact was used.

Acquisition: Gordon et al., Development of a symmetric echo planar imaging framework for clinical translation of rapid dynamic hyperpolarized13C imaging, MagnResonMed77:826–832(2017; online2016), doi10.1002/mrm.26123, PMC4992668. Parent actually opened complete308-line body; private `parent_primary_web_capture.json` retains it. The source sample README identifies InVivo rat-kidney constant-excitation exp2 and this acquisition reference. The method paper's17 human prostate studies and their masked clinical analyses are not this sample. The acquisition paper's variable-excitation figure is not mislabelled as this exact constant-excitation array. There is no claim to repeat all published datasets, simulate their clinical truth or reproduce a missing bolus-model comparison.

The method equation uses fixed relaxation assumptions and inputless measured substrate. The code evaluates xstar=-inverse(A)b; its sign is independently required by Axstar+b=0 and verified by analytic ODE controls. Extracted PDF text appears to omit a minus in the paper's printed equilibrium expression, but without a verified rendered-formula inspection this is recorded as a text ambiguity, not a proven paper typo. The code's source sign is preserved.

## Author source, exact revision, license and observations

Author repository: https://github.com/LarsonLab/hyperpolarized-mri-toolbox
Remote revision actually verified live: `729f0d74d3f8d707a5f992a83aada9e45f53dc75`. The captured commit/tree metadata, source files, author README/notebook and LICENSE are private. Git blob SHA1 verification against that exact remote tree independently proves the bytes of every selected source/data member, including source downloaded initially through its branch. Software is BSD-3-Clause, Copyright2014–2024 Regents of the University of California; Python translation Copyright2024UCSF, SuleSahin. LICENSE and NOTICE are preserved. This is licensed extraction, not an access-only repository.

Bound data member: `sample_data/Rat Kidneys EPI/exp2_constant/exp2_constant.mat`,189769 bytes. Author sample_data README explicitly labels it InVivo and separates the constant and variable excitation experiments. All pyr/lac32x32x25 float32 arrays, TR2s and both25-point nominal15degree uint8 flip arrays are retained. A separate author DICOM first frame independently confirms32x32,3mm pixel pitch,20mm slice and3T; its export flip90/TR200ms headers are inconsistent with the MAT/author-analysis dynamics and are NOT adopted as the bound excitation/timing. No absolute anatomical registration is inferred from a single exported frame. Raw image pitch/FOV and amplitude convention remain explicit.

Data-specific redistribution terms are not stated. The public offered sample supports the authorized local research/benchmark work. We do not infer a separate data licence from BSD software terms, do not claim CC rights and have not uploaded or published external data. Public redistribution remains a material release limitation. No controlled-access original clinical patient data were accessed.

Raw is authored real signed data, not simulation, magnitude-converted data or synthetic truth. Negative measured samples are retained. Native pyruvate/lactate channels remain in one common signal convention; no fitted target scale, B1 estimate, hidden rate, anatomical segmentation or independently measured relaxation map is introduced as public calibration. T1P30s/T1L25s are declared assumptions. The full native grid is used; the128² zero-filled paper display is not used.

## Original exact-data execution and configuration chronology

Private archive is `research_batch8_20261022/groups/bio_med/hyperpolarized/` outside the task. Source original file `analysis__python__py_fit_pyr_kinetics.py` is never imported by shipped solver/Judge/tests/Notebook. Isolated `oracle_env` uses Python3.11.11/NumPy1.26.4 with SciPy1.17.1 and the shared installed general plotting helpers exposed read-only. It is not a shared-environment upgrade. NumPy1.26 is required for the original legacy shape-(1,) scalar assignment. All processes use OMP/OPENBLAS/MKL threads1.

1. Original API with nonnegative k but free unused S0P and unbounded L0: full1024 native0/207.291s. It returned negative L0 and a completely unused drifting S0P. Preserved `native_full_nonnegative*`; not the selected reference.
2. Original supported API with S0P fixed0, k/L0>=0 and initial L0=0: full1024 native0/156.052s. Full-data fit was useful but independent analytic L0/profile checks caught nine boundary-start stalls. Preserved `native_identifiable_physical*`, `independent_profile*` and four original API diagnostic probes; not claimed accepted.
3. Source-only four fixed cases with uniform interior L0=1 remove the stall, e.g.(15,17) SSE1591262.7→1473244.0, initial fitted L0.0056→701.839. No observations, bounds, model, solver tolerances or target pixels changed. Uniform1 longitudinal unit gives only≈.259 transverse signal unit, well below observed hundreds-unit noise scale. This is an API initialization condition, not observation calibration or target-derived field.
4. Final selected original API on all1024: fixed R1P1/30,R1L1/25,S0P0; kPL/L0 lower bounds0; initial(.002,1); original TRF ftol/xtol/gtol1e-6, other defaults. `run_native_interior.py` full native0/71.6043212s, all seven returned arrays finite. An independent full original repeat is bitwise for every output. Original exact final logs, command/process JSON and outputs are retained separately.
5. Original NumPy2.4.6 compatibility comparison changes ONLY Mz_init[0]→Mz_init[0,0] in scalar forcing assignment (private `oracle_numpy2_compat.py`), all1024 pixels, native0. This does not replace the mandatory unmodified original1.26 run. Fixed-input field discrepancies establish the backend numerical envelope; objective relative difference2.94e-13. Source CHIsq contains signed observed+fitted denominators and is unstable near poles, is retained as a labelled original diagnostic, and is not a scientific fit target or a valid statistical claim.

Only after complete final original runs, independent quality/physics and fixed-input envelope did `freeze_source.py` capture `frozen_stages_final.npz` and `PRELOCAL_FROZEN_ACCEPTANCE.json`, before local implementation. Git-verified source/data and all original seven final outputs remain private. Frozen fixtures include RF factors and five actual-data trajectory/forcing/residual cases. No cleaned output generated its own claimed upstream fixture.

## Independent scientific gate

Constant-substrate closed-form ODE, zero-exchange relaxation/RF product, independent solve_ivp interval integration, multi-excitation RF factor grouping and strict zero-input non-identifiability checks passed. Source S0P mutation0→123456 has bitwise zero forward effect, justifying its fixed nuisance status without discarding an active scientific quantity. All two active fields remain nonnegative and finite.

An independent two-pool closed-form recurrence (separate from the4x4 source implementation) profiles nonnegative L0 exactly and evaluates a broad rate grid through50s^-1 on all1024 real time courses. At the final selected original solution, exact recurrence replay max1.21e-11 signal unit, conditional-L0 total SSE improvement2.15e-11, grid maximum per-pixel relative improvement2.38e-10, no pixel improvement>0.1%. No global profile output is substituted for source optimization. Source full-data relative residual≈.490 improves over per-pixel constant≈.566 and k=0 with optimal nonnegativeL0≈.833, zero1. Full tail-inclusive Frobenius norms are used, never median-only quality.

All1024 conditional numerical Jacobians have rank2; column-normalized condition spans1.007–2.418. Full rank does NOT establish useful biological precision: measured-pyruvate peak quartile1 vs4 diagnostic rate sensitivity medians≈.02369 vs.0009975s^-1. Temporal-difference MAD is an observed-data diagnostic scale, not independent scanner noise and not a calibrated confidence interval. Noise-dominated background is retained and explicitly described as conditional coefficients, not claimed tissue-rate recovery. Negative measured forcing/fitted samples are not interpreted as real negative metabolism and are not clipped. The paper's clinical tSNR>80 support is not silently imported into this different all-field author demonstration.

## Equation/function/independent-test traceability

| Paper/stage | Exact upstream source | Local callable | Independent check / consumer |
| --- | --- | --- | --- |
| Eqs4–7 RF transverse/longitudinal depletion | flips_scaling_factors | physics_model.rf_factors | frozen RF factors; independent grouped pulses and zero-exchange control |
| Eqs2–3 exchange/relaxation matrix | trajectories_inputless A | physics_model.trajectory | constant-substrate analytic and solve_ivp; source state fixture |
| Eq9 piecewise input elimination | trajectories_inputless u | physics_model.trajectory | exact measured endpoint, source forcing fixture, independent closed form |
| Eq8 exponential interval state | trajectories_inputless xstar/expm | physics_model.trajectory | source5actual-state cases; physical equilibrium sign and continuous solution |
| Measured-minus-model product residual | difference_inputless | physics_model.residual | frozen source residuals and full pipeline |
| Constrained TRF minimization | py_fit_pyr_kinetics least_squares | solvers.reconstruct | all1024 two-field parity, source objectives, independent profile-quality check |
| Native calibration precision/zero padding | original author notebook + reshape/padding | preprocessing.load_data/prepare | exact raw dtype/shape, float16 angle, source factor parity |
| AUC/timing Eq10/13 | paper definitions | active Notebook expressions | actual measured full-field summaries, not substituted rate calibration |
| Fixed relaxation/flip sensitivity | paper Fig3–5 scientific comparison logic | actual full-field local refits in Notebook | four actual1024 alternative-condition fits, complete fields and residuals |
| Strict field metrics | declared frozen criteria | independent evaluation/judge | analytic NCC/DR-NRMSE; truth priority, physical negatives, malformed inputs, overflow, cwd |

Paper Eq6 sum versus Python mean for multiple pulses is explicitly documented: selected single excitation makes them identical. Source4x4 dummy states are retained; the backwards Istart path is irrelevant because selected Istart=0. Local caches constant matrix inverse/exponential within a trajectory only, not solver solutions; every fresh fit is computed. General SciPy TRF and matrix exponential are permitted numerical primitives, not a domain solver. No domain package or upstream import appears anywhere in shipped execution/tests/Notebook.

## References, criteria and contracts

`evaluation/baseline_reference.npz` contains two fields from final selected unmodified original1.26 run, not truth. `notebooks/reference_outputs/upstream_full.npz` preserves all seven original source outputs; the Notebook consumes parameters, fitted lactate and objective, with other diagnostic output definitions recorded here. `source_boundary_initialization.npz` retains original failing initial0 configuration for the explicitly labelled comparison. `independent_profile.npz` is the prelocal independent source-quality grid/Jacobian diagnostic, not fitted truth. `test/fixtures/upstream_stages.npz` is consumed by exact5case stage tests and physical Notebook review. Fresh local arrays and summary JSON are separately named. No reference/calibration derived from a hidden target is public.

Prelocal criteria: every field cosine NCC>=.999999, standard dynamic-range NRMSE<=1e-6, maximum absolute error/reference range<=1e-5 and nonnegative_count<=0. Limits are NOT from input ULP perturbations or tuned to cleaned discrepancies. Original fixed-input repeat is bitwise; NumPy2 fixed-input source comparison supplies the measured numerical envelope below. No crop, rescale, phase conversion or alignment is permitted. Every one of the1024 values of both fields is evaluated independently. Constant, zero, doubled and physically negative fields fail. A real ground_truth file would take priority; malformed truth produces evaluator error without fallback.

The standalone evaluator uses module-relative hidden assets and cwd-relative CLI paths. Exact two keys, shape(1,32,32), real float32/64 and finite values are required. Error metrics must have finite nonnegative bounds; NCC bound must be in(0,1]; negative-count bound is exactly0. A tiny negative parameter is a valid physical failure. Huge finite inputs that overflow arithmetic return error with empty metrics and writable finite JSON. No solver is imported, no solver code is executed. Missing hidden assets, malformed input, report failure, foreign cwd and truth priority are independently covered.

## Local environment and actual checks

Python3.11.11 on Windows, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. CPU execution with one numerical thread. No GPU or domain library is needed. Source and local NumPy configuration/backend captures are archived; private1.26 source environment does not alter the shared runtime. Runtime requirements contain only actual source dependencies. Jupyter/pytest are development tools, not task dependencies.

First pytest attempt:7 checks passed;22 tmp_path setups failed because the Windows shared user TEMP pytest directory denied enumeration. This is an environment failure, not22 algorithm failures. No test was removed; the complete rerun used an explicit private --basetemp and all29 passed. The same full source/local parity and physical tests passed. Logs retained below. Public data/README/environment audit and full source signatures/signs/factors/units/call-role semantic review are recorded at finalization.

## Final execution and figure review

Final complete active Notebook native exit0, runtime 169.7351481000078 seconds; canonical SHA256 `b4bb043f1975cb4fb3e2b5ce224d3e2df699d2b93e094dfff537768ef5bbf3f6`. Canonical bytes exactly equal the actual private executed notebook. All20 cells remain in six self-contained chapters; all code cells executed without errors. Standard full reconstruction plus four complete1024-pixel alternative-assumption refits run actively.

First complete notebook native0/151.0902581s is retained separately. A subsequent attempted display revision through Windows PowerShell default text encoding corrupted non-ASCII punctuation and caused a code-string SyntaxError after38.2883s; the failed log/record are retained. The final revision restores the original UTF8 notebook through Python explicit UTF8/nbformat, makes only the two intended display improvements and executes completely again. No failed result is called completed.

Every one of the13 final PNG groups was opened at actual resolution; references/fresh/difference colour scales, signed zero-neutral colours, physical units, no hidden exclusions and legends/captions were checked. Fig08 now zooms its declared0..0.03s^-1 parameter interval with matching y autoscale; all broad-grid numeric values remain. Fig12 preserves extreme signed-sum ratios/time moments with full-range symmetric-log display, explicitly explaining these are unstable near-zero-denominator diagnostics rather than valid arrival times. PNG/SVG/PDF exports, paired numerical NPZ, individual JSON and figure_manifest.json are private. All manifest hashes are independently checked. No figure shows an uncomputed method. Exact science summaries before and after display revision agree; visual_revision_numeric_identity.json records that check.

Semantic source/plan review checked state order, active vs dummy parameters, RF sum/mean single-pulse identity, transverse residual scale/sign, inverse-equilibrium sign, fixed relaxation units, source initialization/bounds/stopping and every callable signature/default. README contains all raw keys with correct batch-first shapes and calibration precision, all physical targets and common amplitude convention. No public GitHub URL, hidden fit calibration, truth key or privileged observation leaks. Actual import/call-role and standalone foreign-cwd audit is archived; no domain solvers, cross-task imports or upstream runtime appear. Mathematical formulas were parsed with both KaTeX and MathJax; final counts and checks are in the B8 qa record.

Final complete test suite includes29 original checks plus5 additional independent per-field/CLI-exit/README-default regressions:34 total. All passed with the private basetemp. The initial29-pass evidence remains unchanged.

Completion checklist: source/paper/hash/equation mapping PASS; unmodified original exact observations PASS; independent fixtures/physics PASS; units/precision/native full-grid conventions PASS; local standard CLI/Judge PASS; all34 unit/parity/physics/contract tests PASS; public self-contained input/no leaks PASS; permitted local mathematical runtime PASS; complete active Notebook native0 PASS; identified metrics/references/tolerances PASS; source/plans/README semantic consistency PASS; final inventory/licenses/export numeric sources PASS. Actual biological truth is unavailable, public data redistribution rights unspecified, and weak-signal conditional rates are limitations rather than unverified success claims.

## Exact hashes and machine-readable evidence

Private archive: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch8_20261022\groups\bio_med\hyperpolarized`.

| Asset | SHA256 |
| --- | --- |
| `sample_data__Rat Kidneys EPI__exp2_constant__exp2_constant.mat` | `baa488a2d00d1d9f5f8e423905387e402f8824606058900c503cf194166c1c82` |
| `analysis__python__py_fit_pyr_kinetics.py` | `8ee08434093805e6610bf8072d58bfc1016d6231dbac4d243ca1cf3b0e7e60a2` |
| `LICENSE` | `d981e150dd70e8aa6f39da59b0a5b9a7abcecb719e0e4bb64c6aa92f8d622a8e` |
| `native_interior_initialization.npz` | `b4064ed5f198402c1e814220803dc12014317794c21a738fe21942b2a483e677` |
| `native_interior_repeat.npz` | `b4064ed5f198402c1e814220803dc12014317794c21a738fe21942b2a483e677` |
| `native_numpy2_compat.npz` | `d72ad51df4a0bde72067ab89095a51c03866e67e50a4053e3f662af76bd3a640` |
| `PRELOCAL_FROZEN_ACCEPTANCE.json` | `9edb6a9172bb0051d53821bcb1508440e04c00a5953277c0496261b90214dbca` |
| `frozen_stages_final.npz` | `855a9a17c6906f03393ad13f169225fa935a4483d8d438a9b2286920559e35f9` |
| `parent_method_pdf_web_capture.json` | `aafa54aa6d93af4babb75d3cc8fb0272746e5585d07b7e7350e3cf3a73e484bc` |
| `parent_primary_web_capture.json` | `5bf1a9ce50f041ae68f1545f75e1c49ef0fd24de0c28fbbd9e30eac0414ee74e` |
| `interior_profile_quality.json` | `a136ed06054b8643a816e8bdded678370fe6916f9f336166a571ac00f4720045` |
| task `data/raw_data.npz` | `155d525978226c502aa9abdd2319d9e931e4ddb5f8a9c11d19ad6b741bf5d880` |
| task `evaluation/baseline_reference.npz` | `d539d2f67cf08a31d65d53b6be0d8f2820dc8f2d4c5b45d6a1eff0963120c138` |
| task `test/fixtures/upstream_stages.npz` | `855a9a17c6906f03393ad13f169225fa935a4483d8d438a9b2286920559e35f9` |

Fixed-input numerical envelope, before local extraction (field scales and all-pixel errors):

```json
{
  "exchange_rate": {
    "source_min": 5.286621094965384e-36,
    "source_max": 0.19395795720443706,
    "fixed_input_numpy2_dr_nrmse": 4.573957748950697e-08,
    "fixed_input_numpy2_max_abs_over_range": 8.201226081877829e-07,
    "fixed_input_numpy2_ncc": 0.9999999999999817
  },
  "initial_lactate": {
    "source_min": 1.1206853045390985e-12,
    "source_max": 6410.0953890800565,
    "fixed_input_numpy2_dr_nrmse": 7.139788926671538e-08,
    "fixed_input_numpy2_max_abs_over_range": 1.6910595211765337e-06,
    "fixed_input_numpy2_ncc": 0.9999999999999082
  }
}
```

Recorded commands/results (all exact bound observations):

```json
[
  {
    "tag": "main",
    "command": [
      "D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe",
      "-m",
      "src.main"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staging\\tasks\\rat_hyperpolarized_pyruvate_lactate_exchange",
    "exit": 0,
    "seconds": 21.349140799982706
  },
  {
    "tag": "judge",
    "command": [
      "D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staging\\tasks\\rat_hyperpolarized_pyruvate_lactate_exchange",
    "exit": 0,
    "seconds": 0.1515955000068061
  },
  {
    "tag": "tests",
    "command": [
      "D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe",
      "-m",
      "pytest",
      "test",
      "-q",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\bio_med\\hyperpolarized\\pytest-temp-final"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staging\\tasks\\rat_hyperpolarized_pyruvate_lactate_exchange",
    "exit": 0,
    "seconds": 23.98283530000481
  }
]
```

Fresh standalone Judge result:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "exchange_rate/ncc": 0.9999999999999819,
    "exchange_rate/dr_nrmse": 4.573957748950697e-08,
    "exchange_rate/max_abs_over_range": 8.201226081877829e-07,
    "exchange_rate/negative_count": 0,
    "initial_lactate/ncc": 0.9999999999999076,
    "initial_lactate/dr_nrmse": 7.139788926671538e-08,
    "initial_lactate/max_abs_over_range": 1.6910595211765337e-06,
    "initial_lactate/negative_count": 0
  },
  "thresholds": {
    "exchange_rate/ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "exchange_rate/dr_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "exchange_rate/max_abs_over_range": {
      "comparison": "<=",
      "boundary": 1e-05
    },
    "exchange_rate/negative_count": {
      "comparison": "<=",
      "boundary": 0
    },
    "initial_lactate/ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "initial_lactate/dr_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "initial_lactate/max_abs_over_range": {
      "comparison": "<=",
      "boundary": 1e-05
    },
    "initial_lactate/negative_count": {
      "comparison": "<=",
      "boundary": 0
    }
  },
  "errors": []
}
```

Final fresh Notebook summary:

```json
{
  "reconstruction_seconds": 28.823038399976213,
  "sensitivity_seconds": {
    "T1L20": 28.562922999990406,
    "T1L30": 28.53327280000667,
    "flip095": 28.698565200000303,
    "flip105": 26.943119400006253
  },
  "agreement": {
    "exchange_rate": {
      "ncc": 0.9999999999999819,
      "dr_nrmse": 4.573957748950697e-08,
      "max_abs_over_range": 8.201226081877829e-07
    },
    "initial_lactate": {
      "ncc": 0.9999999999999076,
      "dr_nrmse": 7.139788926671538e-08,
      "max_abs_over_range": 1.6910595211765337e-06
    }
  },
  "forward_null_labels": [
    "Zero signal",
    "Per-pixel constant",
    "k=0, fitted L0",
    "Fresh inputless"
  ],
  "forward_relative_l2": [
    1.0,
    0.5660148859024048,
    0.8332749565668754,
    0.489943916066924
  ],
  "negative_fitted_lactate_samples": 21,
  "negative_forcing_samples": 11161,
  "condition_range": [
    1.0069201886673118,
    2.4179172868738585
  ],
  "quartile_sensitivity_medians": [
    0.02368831972603345,
    0.018031594283638097,
    0.008167042707288097,
    0.0009974704978246577
  ],
  "judge": {
    "status": "pass",
    "passed": true,
    "metrics": {
      "exchange_rate/ncc": 0.9999999999999819,
      "exchange_rate/dr_nrmse": 4.573957748950697e-08,
      "exchange_rate/max_abs_over_range": 8.201226081877829e-07,
      "exchange_rate/negative_count": 0,
      "initial_lactate/ncc": 0.9999999999999076,
      "initial_lactate/dr_nrmse": 7.139788926671538e-08,
      "initial_lactate/max_abs_over_range": 1.6910595211765337e-06,
      "initial_lactate/negative_count": 0
    },
    "thresholds": {
      "exchange_rate/ncc": {
        "comparison": ">=",
        "boundary": 0.999999
      },
      "exchange_rate/dr_nrmse": {
        "comparison": "<=",
        "boundary": 1e-06
      },
      "exchange_rate/max_abs_over_range": {
        "comparison": "<=",
        "boundary": 1e-05
      },
      "exchange_rate/negative_count": {
        "comparison": "<=",
        "boundary": 0
      },
      "initial_lactate/ncc": {
        "comparison": ">=",
        "boundary": 0.999999
      },
      "initial_lactate/dr_nrmse": {
        "comparison": "<=",
        "boundary": 1e-06
      },
      "initial_lactate/max_abs_over_range": {
        "comparison": "<=",
        "boundary": 1e-05
      },
      "initial_lactate/negative_count": {
        "comparison": "<=",
        "boundary": 0
      }
    },
    "errors": []
  }
}
```


## Parent independent private acceptance

Parent independently ran all34 tests: native0/26.620751900s; all38 formulas passed both renderers. Parent read original method and acquisition papers, complete selected local physics and solver, independent Judge, both plans and provenance; all13 final scientific figure groups were actually opened individually. All READY task hashes and figure export hashes matched. Canonical notebook is byte-identical to native0/169.7351481s archived full execution, SHA256 b4bb043f1975cb4fb3e2b5ce224d3e2df699d2b93e094dfff537768ef5bbf3f6; full1024 pixels, signed observations and all25 time points remain, including four active full-field T1/flip sensitivity fits. Independent source profile and initialization diagnostics justify the prelocal source configuration, rather than post-hoc threshold changes. Fixed T1/flip and signed measured-input forcing are conditional assumptions; weak-input background rank does not establish biological identifiability. Hidden arrays are reproduced parameter estimates, not measured truth or absolute mM. Source float16 angle conversion is preserved explicitly. Data redistribution rights remain unspecified. This records private acceptance pending batch completion.
