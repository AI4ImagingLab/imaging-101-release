# Validation and provenance

## Physical binding and semantic scope

All28 native U Gem spectra are bound without a new crop or synthetic replacement. The recovered quantity is the full144×144 signed velocity-plane H-alpha emission density, not stellar-surface brightness, vector magnetic field, eclipse brightness or a sky image. The original angular/wavelength calibration is approximate and this standard-method output is not truth. All16124 known-support pixels and all20736 output samples are evaluated; no postfit target mask, shift, rescale or magnitude conversion is used. The five predetermined eclipse phases remain in independent full-forward evaluation. The common phase-file width.03930 was fixed before inverse quality, with tutorial.003 and half-width.01965 retained as separate diagnostics. Actual spectra are real; perturbations are diagnostic simulations only. The complete actual README corpus was checked against existing LO Peg brightness/Zeeman, thermal eclipse, spectral-component and astronomy mappings; acquisition and velocity-plane unknown differ. Previously rejected ZDI and pressure candidates remain excluded.

## Primary sources, rights and archive

Original GitHub: https://github.com/Alymantara/pydoppler, remote-verified exact commit dc2d395680862159d74f164f44c5465041c5e8fd. Source archive ZIP SHA256 fd4391c9afa2792049f3c73db3aeb5f2bd6ec90b527cbbf372d410432eca4054. Full exact checkout and downloaded archive remain private under Alymantara__pydoppler/. Actual wrapper used pydoppler/pydoppler.py, not its build/lib duplicate. Historical Fortran dop.f identifies Version2.0, H.C.Spruit3-3-1999. This is the supplied source variant of the1998 method, not a claim of identical paper-era code.

Full method paper: H.C.Spruit, Fast maximum entropy Doppler mapping, arXiv astro-ph/9806141v1,10Jun1998,8pages. Full observational paper: Echevarria et al., arXiv0704.1641v1, AJ134,262,doi10.1086/518562,36pages. Both full PDFs and figures were read. The observed/computed/difference/map scientific comparisons are preserved for this binding; different objects in the paper are not additional bound acquisitions. Roche/stream overlay requires independent binary calibration and is not claimed. The source display averages phase bins; this notebook displays each actual observation row to retain all data and avoid empty-bin NaNs.

Python wrapper MIT-template notice does not license historical Fortran or spectra. No explicit grant for those materials was established. Source and original notices are private; NOTICE accurately limits this to authorized local research and flags public redistribution unresolved. No invented CC/MIT data license is asserted.

Private archive root: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch8_20261022\groups\astro_earth_mech\doppler. archive_index.json indexes exact scripts, hashes, source logs, process records and papers. paper_downloads.json records download provenance. Paper files:
```json
[
  {
    "path": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\astro_earth_mech\\doppler\\spruit_astro-ph_9806141v1.pdf",
    "sha256": "8a3d4ffeca21112a18f75656bbe6a177de65e4cffad7521e10e1fc2f47c3998d",
    "bytes": 219882
  },
  {
    "path": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\astro_earth_mech\\doppler\\ugem_0704.1641v1.pdf",
    "sha256": "cf0f81af9442f1e2cd7c6c80f9209f9113eb9adb1700a50077418f6c31817d82",
    "bytes": 558787
  }
]
```

## Original exact-observation execution

Unchanged domain source run native exit0, runtime4.037547900s; independent run exit0, runtime3.867976600s. Commands, complete transported shell invocations and per-command runtimes are in metadata_bound_original/original_native_process.json and metadata_bound_repeat/original_native_process.json; supervisor_process.json separately records outer runtime. Full dop.in/dop.log are in their original_work directories. Original27updates, alpha.002, ns7, accuracy8e-4, maximum150, onealpha only. Native data files are txhugem4004..4031,2048samples each. Their original radiometric units are unavailable.

Generic environment adapters only: removed NumPy scalar aliases; unchanged POSIX source commands transported verbatim through WSL Ubuntu24.04; all original filename/phase tokens retained in a two-column phase alias because the native parser cannot parse the supplied third metadata column. No physics or inverse formula was changed. Exact adapters and phase token checks are recorded by phase_input_format_adapter.json. Original full three-column parser failure, a wrong build/lib file attempt, and a successful core followed by a display-only NaN assertion are retained separately. Empty displayed phase bins are not invalid inverse pixels.

Passive source capture adds only stream writes in a separate private copy; passive_capture/passive_capture_diff.json preserves the exact instrumentation diff. Its final dop.out is byte-identical to uninstrumented output, as recorded in passive_parity.json. Independent original subroutine derivatives and source real32 constants are tested before local extraction. The original fixed-form line `if (wim.lt.wlm) then wl=wlm-wim` assigns THENWL rather than changing WL; normalization_statement_check proves WL=0 and THENWL=.3000000119. Local normalization preserves the actually executed exact per-spectrum flux normalization, without silently implementing a different offset algorithm.

Original derivative harness initially declared MINUS implicitly integer and failed; that own-harness mistake and its correction to real*8 are preserved in original_derivative_checks/initial_harness_failure.json. No upstream formula was altered. Corrected independent source entropy FD relative error6.01615e-9, likelihood FD4.19674e-7, projection inner-product error1.55854e-19 and smearing symmetry error2.03288e-19; criteria1e-5 and1e-12 were fixed independently.

## Traceability

| Paper/source stage | Upstream implementation | Local implementation | Independent test |
|---|---|---|---|
| Phase-resolved line calibration | pydoppler.py spectrum interpolation/continuum/native file writer | preprocessing.prepare | test_spectral_preprocessing |
| Finite-exposure triangular projection | dop.f prom/dist/fproj/bproj | DopplerModel constructor/forward/backward | original64columns; independent adjoint/column-sum |
| Positive data and eclipse weights | dop.f basel/norms/blanks/weighd | initialize_data | original preprocessing; isolated fixed-form statement |
| Floating Gaussian default | dop.f chimg | DopplerModel.smear | source q, self-adjoint inner product |
| Likelihood/entropy derivatives | dop.f like/entr | objective_derivatives | source first2gradients; independent FD |
| Accelerated two-direction Newton | dop.f lamun/rlfldn | newton_coefficients/solve | first2updates; full27updates |
| Signed emission density | source dop.out writer | reconstruct/scatter | original fullprecision full144² result |

## Independent freeze and metric evidence

source_only_quality_freeze.json was written2026-10-22T07:06:08Z before local inverse implementation. Complete source-format/repeat error5.12778738208111e-6, float32 serialization7.024492154932004e-9 establish nonzero field tolerance1.5×max=7.691681073121665e-6; cosine boundary.999999. Full original forward error.08115434873292197, boundary.08521206616956807. No bound was relaxed after local results. All zero/flat/shuffled/transposed/sign-reversed controls fail both meaningful field or forward requirements; exact fullprecision measurements are retained in the freeze.

Metrics use cosine without centering and RMS error divided by full reference range. Whole forward RMS is divided by the complete positive-observation range, including eclipse rows. The baseline is a method result, never actual truth. Fresh exact CLI report:
```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "ncc": 1.0,
    "nrmse": 9.896093712720143e-10,
    "forward_range_rms": 0.08115434872885191
  },
  "thresholds": {
    "ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "nrmse": {
      "comparison": "<=",
      "boundary": 7.691681073121665e-06
    },
    "forward_range_rms": {
      "comparison": "<=",
      "boundary": 0.08521206616956807
    }
  },
  "errors": []
}
```

## Runtime, fixtures and notebook

Clean runtime actual environment:
```json
{
  "python": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "python_version": "3.11.11",
  "platform": "Windows-10-10.0.26200-SP0",
  "numpy": "2.4.6",
  "scipy": "1.17.1",
  "matplotlib": "3.11.1",
  "threads": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  }
}
```

NumPy/SciPy BLAS/compiler configuration is in numerical_backend.txt; upstream removed-alias and WSL/gfortran environment is independently captured in original process logs. Shared packages were not installed or upgraded. No domain library is imported by runtime, tests, generator, Judge or notebook. Source calculations and tests run CPU single-thread.

Frozen numeric fixture consumers:
```json
{
  "original_geometry.npz": "test_projection_and_preprocessing_stages: index_x/index_y; original physical geometry metadata retained for provenance",
  "original_projection_columns.npz": "test_projection_and_preprocessing_stages: all64 frozen columns, rows, indptr, weights and shape",
  "original_spectral_preprocessing.npz": "test_spectral_preprocessing: velocity and normalised_flux; source-selected calibration diagnostics",
  "original_preprocessing.npz": "test_projection_and_preprocessing_stages: positive_data,baseline_data,baseline_field,weights; full captured initialization metadata",
  "original_iteration1.npz": "test_first_two_source_updates: seven explicitly pre/post-labeled arrays",
  "original_iteration2.npz": "test_first_two_source_updates: seven explicitly pre/post-labeled arrays",
  "original_fullprecision.npz": "active Notebook: reconstruction; source full-precision positive/signed field, reconstructed spectra and final parameter metadata retained as identified original measurements",
  "original_convergence.npz": "active Notebook figure5: source iteration, pre-update objective and convergence"
}
```

Final tests command: ['D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe', '-X', 'utf8', '-m', 'pytest', 'test/', '-v', '--basetemp', 'D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\astro_earth_mech\\doppler\\pytest_release_temp']; exit0. The first expanded suite had30pass/1failed because its known cosine test required exact1 rather than observed.9999999999999998; changed only that independent assertion to1e-15. No method/metric/threshold change. Final full tests are recorded verbatim in final_pytest_stdout.log. Main and Judge commands/logs are separately recorded in final_main_process.json and final_judge_process.json. Native foreign-directory CLI success/failure/error/report-write branches are tested.

Notebook command: ['D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe', '-X', 'utf8', '-m', 'nbconvert', '--to', 'notebook', '--execute', 'D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staging\\tasks\\ugem_velocity_emissivity\\notebooks\\ugem_velocity_emissivity.ipynb', '--output', 'ugem_executed.ipynb', '--output-dir', 'D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\astro_earth_mech\\doppler\\executed_notebook', '--ExecutePreprocessor.timeout=900']; actual native exit0, runtime48.282890100s, canonicalSHA256 f8d6301e90718f6090d6d52b3b42f44868b35de976e55299a44925c6c3561021. Six explicit Problem/Data/Physics/Method/Results/Conclusion headings; every code cell executed; six actual embedded PNG groups. First successful execution51.1729552s was visually inspected; raw negative-value heatmap range and phase-curve sorting were corrected and a complete new execution recorded. General Windows Proactor/ZMQ warnings are retained; no warning was misclassified as inverse failure. The actual kernel executable, package versions and backend are printed in the notebook.

All six actual final figure groups were opened individually: full raw observations, physics/preprocessing, original/fresh/signed difference, whole observed/model/residual spectra, acceleration ablation/convergence, and alpha/raw-noise/exposure sensitivity. Signed scales are symmetric and no data are silently saturated. Figures and NPZ/JSON numeric sources live privately in figures/; plot_provenance.json binds hashes to this canonical notebook and parent atlas consumer.

Plans were checked against actual source signatures, derivatives/Hessian, normalization/units, update/stop order and output axes. Source objective is pre-update and saved field post-update. Local signed output includes all weak values and outside-support zeros. Field agreement does not mean high-accuracy physical emissivity or exact exposure calibration.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data/equation mapping | passed | Primary full papers, exact commit, archive index and traceability below. |
| Original and local exact observations | passed | Two original complete runs, passive capture and final_main_process.json. |
| Independent parity and physics | passed | Frozen source fixtures, directional derivatives, adjoints and full pipeline tests. |
| Constants/units/precision/coordinates | passed | Native source real32 constants, quantization, axes and documented legacy calibration retained. |
| Standard main and fresh Judge | passed | final_main_process.json and final_judge_process.json native0. |
| Judge adversarial and poor outputs | passed | Known metrics, truth precedence, malformed input/configuration, overflow, report failure and five controls. |
| Public information sufficient | passed | README gives full acquisition, conditional model and output scale. |
| No target/GitHub leakage | passed | Raw has only native wavelengths, spectra and phases; public URLs are paper DOI/arXiv only. |
| Acquisition/output metadata | passed | README key table, acquisition table and full signed output contract; no metadata JSON. |
| Minimal shared runtime | passed | NumPy/SciPy/Matplotlib only; exact research executable and backends recorded. |
| Explicit local mathematics | passed | Local calibration, triangular projection, floating entropy, derivatives and Newton updates; no domain package imports. |
| Full tests | passed | final_pytest_process.json native0, runtime 17.044038s. |
| Active notebook and comparisons | passed | notebook_process.json native0, runtime 48.282890s; six rendered groups opened individually. |
| Metric and result provenance | passed | Source-only freeze preceded extraction; original, fresh and diagnostic outputs labeled. |
| Plans and README consistency | passed | Signatures, analytic Hessian, stopping conventions, units and source fixed-form behavior reviewed against shipped code. |
| Inventory and licenses | passed | Template-only files, private archives and truthful unresolved original/data redistribution NOTICE; caches moved privately. |

Parent private acceptance: pending independent parent review; not claimed by this worker.

## Final independent contract and rendering review

Raw NPZ SHA256: 4d53738fb5145d1d6960bcd42851e59c8bfbcb5d144a35cb410d55d249742513. All28original file SHA256 values and sizes are indexed in private data_provenance.json; no native spectra were dropped.

Final full suite:32passed/16.74s, supervisor native0/17.04403759998968s. Final notebook native0/48.28289010000299s; all six actual groups inspected, including corrected full raw-value range and phase-sorted physical tracks. Audit qa/audit.json reports errors[] and includes actual foreign-cwd Judge exit0. Markdown/notebook scanner found18formulas, no delimiter hazards; both MathJax3.2.2 and KaTeX0.16.22 report0failures. README now explicitly defines selected alpha=.002 and Gaussian default weights so the conditional estimate is self-contained. No numerical code, output or executed-notebook payload was changed by this clarification.

The first evidence-finalization check incorrectly counted PNG-containing cells rather than PNG outputs; one results cell renders two figures. That maintainer counting assertion failed, then was corrected to count individual outputs (six); the already successful tests/main/Judge and nbconvert were not rerun or relabeled. Process evidence remains actual captured subprocess return codes.


## Parent independent private acceptance

Parent independently reran all32tests: native0/17.764457500s; both math renderers passed all18formulas, visibility/import/call-role audit errors[] and foreign-cwd Judge0. Parent actually opened all6scientific PNG groups individually, and semantically reviewed the README, both plans, acquisition/inverse source, standalone evaluator, six-chapter notebook and provenance. Parent independently compared every canonical code source/output/execution count with the actual successful native executed notebook; they are identical. Only documented Markdown clarification differs for Polaris, with no new execution claimed. All limitations and source qualification distinctions remain explicit. Canonical SHA256 f8d6301e90718f6090d6d52b3b42f44868b35de976e55299a44925c6c3561021. This is private staging acceptance only; no batch8release or aggregate60completion is claimed.
