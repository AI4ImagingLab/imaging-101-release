# Authoritative provenance and validation — mri_b1_actual_flip_angle

Validation date: 2026-10-20. Scope is a complete independently executed selected standard method on the exact bound real observations. All results below distinguish upstream measurements from fresh local measurements; there is no physical ground truth and no reconstruction is relabeled as truth.

## Frozen selection and source identity

Selected implementation: qmrpy source at frozen commit; repository https://github.com/SugimotoKohei/qmrpy; exact executed revision `6dbe5ce7b4b6b7057c8138aa2d7cfed7a70e815f`. Remote identity was verified with a successful `git ls-remote` check (a later network retry reset and is not used to assert newer freshness) (PhasorPy v0.7 tag or qmrpy HEAD) and matched the archived checkout. This identifies an actually checked remote revision on the validation date, rather than merely a local tracking pointer. The private checkout is retained; no freshness beyond that check is claimed.

Paper and version: Yarnykh, Magnetic Resonance in Medicine 57(1), 192–200 (2007), first published 2006-12-26, DOI 10.1002/mrm.21120, PMID 17191242, version of record. The publisher primary abstract explicitly provides the complete selected rational AFI relation r=(1+n cos(a))/(n+cos(a)) and its spoiling/short-TR assumptions: https://onlinelibrary.wiley.com/doi/10.1002/mrm.21120. No inaccessible full journal PDF is falsely claimed as archived. qMRLab software article JOSS 5(53), 2343, DOI 10.21105/joss.02343, archived as qmrlab_joss_2020.pdf, identifies the method ecosystem. Exact finite-TR Bloch recurrence is independently derived and tested; it is a diagnostic, not substituted as the selected inverse.

Data and redistribution: qMRLab/NeuroPoly independent real AFI acquisition, https://osf.io/csjgx/download?version=9, OSF node https://osf.io/tmdfu/. b1afi.zip is archived with API afi_metadata.json. Node CC BY 4.0 is confirmed in osf_node.json and vfa_license.json (the node license, applicable independently of the VFA file). Only acquired AFIData1.nii.gz and AFIData2.nii.gz are bound, retaining full 88×128×32 native volume as original float32 values. The acquisition description data_details.png is archived and inspected. Existing example B1 maps and FitResults are private historical assets, never released as calibration, truth or oracle outputs. No assets are imported from the T1 task; it is an independent acquisition problem.

Code license is MIT; original source copyright/license notices are preserved verbatim in task NOTICE, including qMRLab notices where applicable. Data attribution and conversion changes are also recorded there. Paper PDFs, complete checkouts, original observations, capture scripts and original logs are maintainer-only and are not distributed with this task.

## Maintainer-only archive index

The archive root is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/bio_medical`. Relevant original checkout is `qmrpy/`. Independent capture command is `oracle/Scripts/python.exe oracle_afi.py`; command was run from the archive with the same process interpreter resolved absolutely. The complete SHA256/size archive index is `archive_inventory.json`. Capture source, full stdout/stderr log `afi_oracle.log`, machine record `afi_oracle_record.json`, and raw downloads/data files are retained there. Source capture is independent of all cleaned modules; failing preimplementation import/collection evidence is `afi_red.log`.

Original independent source file hashes:

| Archived source | SHA256 |
| --- | --- |
| qmrpy/src/qmrpy/models/b1/afi.py | 7943ffca389c9ec863423a7501b1b8fffa93310c8d7997e3f6d756e547e9ca56 |
| qmrpy/THIRD_PARTY_NOTICES.md | a371b001ecd3585d15bc0cd5e2f837d7b7e197fd0b5b574487b980360b40e0be |
| qmrpy/LICENSE | 8be06e159a442778357f5f03acc42b96f76808e89b8ca2da4e61f8d2ac5d3666 |

`runtime_environment.json` records the cleaned runtime, platform and complete NumPy backend; `oracle_environment.txt` records the complete isolated upstream pip environment; `oracle_environment.txt` records the complete isolated upstream pip environment; `oracle/` contains isolated upstream-only packages. `upstream_reruns.json` and `upstream_reruns.log` record a later independent original-source run on the exact NPZ arrays. `mri_b1_actual_flip_angle_check_0.log` through `_check_3.log` retain all final entrypoint/Judge/pytest/notebook stdout and stderr. `mri_b1_actual_flip_angle_verification.json` records their subprocess exit codes and wall runtimes. Extracted notebook PNGs are private visual-review evidence, listed in that JSON. No private script is imported by source, tests, Judge or notebook.

## Original upstream run, inputs and numeric capture

This is an actual source execution on the bound observations, not an import check, cached notebook or paper metric. No random seed affects this deterministic method. Observation source hashes and capture environment/runtime are:

```json
{
  "runtime_seconds": 0.17299339990131557,
  "code_commit": "6dbe5ce7b4b6b7057c8138aa2d7cfed7a70e815f",
  "python": "3.11.11",
  "numpy": "2.4.6",
  "shape": [
    88,
    128,
    32,
    2
  ],
  "zooms": [
    2.0,
    2.0,
    5.0
  ],
  "input_hashes": {
    "AFIData1.nii.gz": "9c49ee433bd08f4063e08f8f6ce0b0e4ef4685287199eea554873f386f695d77",
    "AFIData2.nii.gz": "817986f49f11198acf2c0b590e3dafda242c0e5883986439c99ac7ec75077000"
  },
  "raw_sha256": "9e35ca54e8d970b8dfd8c189709d38cfaa70003238a0e52caf6725abf8c4ccf2",
  "valid_voxels": 80238,
  "b1_percentiles": [
    0.5000452485227355,
    0.5296606478374075,
    0.9105089473554756,
    1.1217932546581981,
    1.4305886444403602
  ]
}
```

Capture uses Python 3.11.11 and NumPy 2.4.6 on Windows AMD64 with single-thread OMP/OpenBLAS/MKL settings. PhasorPy runs its published v0.7 Windows compiled wheel whose matching source tag is archived; qmrpy runs directly from the exact original archived source. Upstream-only IO packages and dependencies are isolated in oracle and never become cleaned dependencies. Capture exit status is 0; no numerical warnings occurred on supported voxels. Published/historical example outputs in original download archives are retained only for provenance and were not treated as this capture's result.

## Equation, stage and independent test traceability

| Required published/algorithm stage | Original source or independent derivation | Local callable | Validation case |
| --- | --- | --- | --- |
| Published short-TR AFI ratio and arccos inverse | qmrpy/models/b1/afi.py:B1AFI.fit_raw and fit_image | physics_model.afi_ratio; invert_afi | test_upstream_scalar_inverse; test_analytic_inverse |
| Nominal/actual angle convention and shared intensity cancellation | B1AFI.nom_fa_deg, tr1_ms, tr2_ms | physics_model.invert_afi | test_receive_intensity_cancels; public constants test |
| Observation validity and upstream spurious B1<0.5 flag | B1AFI.fit_raw:spurious; oracle_afi.py fixed signal criteria | preprocessing.load_data; solvers.reconstruct | test_metadata_and_observation_support |
| Fully spoiled two-delay steady-state recurrence, independent diagnostic | Yarnykh pulsed steady state / direct Bloch derivation; qMRLab b1_afi analytical context | physics_model.exact_steady_state | test_exact_bloch_fixed_point; test_short_tr_limit_and_bias |
| Full native-volume mapping | B1AFI.fit_image in independent oracle | solvers.reconstruct | test_full_native_volume; Judge observation ratio consistency |

No equation number is invented where the original source does not provide one. The task-specific full equations, signs, units, normalization and branches are self-contained in README and plan/approach.md.

## Ambiguities, adaptations and scientific limits

The acquisition screenshot lists readout/phase/slice as 128×88×32, while NIfTI stores 88×128×32; public contract uses stored phase/readout/slice and FOV 176/256/160 mm. Protocol 60 degrees, 20/100 ms, TE3.53 ms is copied from the screenshot; the qMRLab source protocol comment says seconds but its default numeric values are 20,100 and its Bloch function documents ms. Only their ratio enters raw inversion, but the exact diagnostic uses correct ms. The selected qmrpy raw method clips cosine and marks B1<0.5 as spurious; no optional qMRLab spatial-filtered result is claimed reproduced. Fixed mask excludes that spurious branch via an explicit observed-ratio bound and requires each measured magnitude ≥10% of its own global maximum; it does not load an existing B1 target map. All original observations, including excluded voxels, remain in raw NPZ. The short-TR inverse is approximate for finite T1. A first physics test incorrectly assumed exact finite-TR equality; it failed, and was replaced by the mathematically appropriate halving-TR convergence test without relaxing any upstream parity or Judge threshold. This validates the approximation limit rather than hiding finite-TR bias.

## Frozen fixtures and reference consumers

upstream_stages.npz contains every 521st valid original 3D coordinate, its untouched paired magnitude signal, and independently executed scalar B1AFI.fit output. Full native-volume oracle is separately captured using the original vectorized fit_image. Consumer: test/comparing_test/test_parity.py.

`evaluation/baseline_reference.npz` is the independently captured standard-method reconstruction used only by standalone Judge and full-pipeline tests; actual ground truth is unavailable, so `ground_truth.npz` is deliberately absent. `notebooks/reference_outputs/upstream.npz` identifies the corresponding independent upstream comparison used in the active review notebook. Its purpose is separately labeled notebook comparison, not a fresh reconstruction. `notebooks/reference_outputs/fresh_metrics.json` identifies fresh cleaned-code source/data/reference hashes, the generated submission and exact full-precision measured metrics; the notebook actively checks this ledger against a new reconstruction. Fresh arrays live in `output/reconstruction.npz`, avoiding a redundant fresh-array copy. Every frozen fixture/reference file has an explicit consumer.

## Tolerances, repeatability and metric provenance

Stage acceptance tolerances were fixed before local extraction: rtol=1e-12, atol=1e-13 for AFI relative field. Exact copied inputs/constants are not validated by decimal rounding. Analytic tests independently exercise physical equations and units; agreement between source and local implementations alone is not treated as physical proof.

Original-source repeatability evidence on exact NPZ data:

```json
{
  "seconds": 0.033390300115570426,
  "upstream_repeat_max_abs": {
    "reconstruction": 0.0
  },
  "bitwise_repeat_equal": {
    "reconstruction": true
  },
  "relative_B1_sensitivity_0p1pct_signal_percentiles_50_90_99": [
    0.00076305604637261,
    0.0016238172928230083,
    0.0034308280787204615
  ]
}
```

Repeated upstream execution is bitwise deterministic on this backend. Independent original and cleaned arithmetic pathways give the fresh errors below; the nonzero one-part-per-million reference-range tolerance is a conservative float64 numerical-equivalence allowance, fixed before Judge scoring. It is not inferred mechanically from the zero self-comparison error and is not a physical-accuracy claim. NCC is uncentered cosine similarity (no mean subtraction); NRMSE is RMS/reference dynamic range. Numerical scoring uses scaling internally to avoid norm overflow without changing these definitions. All-zero estimates have NCC=0 and fail, while degenerate zero/constant references are evaluator errors. Every required quantity must pass, and the independent forward/harmonic consistency metric must also pass.

Fresh cleaned Judge result, retaining full numeric precision:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "reconstruction_ncc": 1.0,
    "reconstruction_nrmse": 0.0,
    "observation_nrmse": 1.0306173095685474e-16
  },
  "thresholds": {
    "reconstruction_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "reconstruction_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "observation_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    }
  },
  "errors": []
}
```

There are no paper-reported scalar performance values silently compared against these numbers. Upstream repeat records, upstream on-bound arrays, and fresh cleaned metrics have distinct identities and paths. Judge and visualization metric functions are checked with the independent numerical example estimate [0,1,2], reference [0,2,4]: NCC=1, dynamic-range NRMSE=sqrt(5/3)/4.

## Runtime, tests and active notebook execution

The cleaned runtime is Python 3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]; NumPy 2.4.6; Matplotlib 3.11.1; platform Windows-10-10.0.26200-SP0. NumPy backend details are in the private environment JSON. Only NumPy and Matplotlib are required by shipped source/review plotting. SciPy is available in shared environment but not declared because these selected closed-form tasks do not import it. Domain libraries are absent from the cleaned interpreter. CPU execution is validated; GPU is not required. Maximum working arrays are well below 1.5 GiB; all notebook/kernel operations run serially with single-thread numeric settings.

Actual final commands and subprocess evidence:

- Exit 0; 0.174313 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m src.main`
- Exit 0; 0.173824 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`
- Exit 0; 1.668677 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m pytest test/ -v --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch_20261020\bio_medical\tmp_final_mri_b1_actual_flip_angle`
- Exit 0; 4.217964 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m jupyter nbconvert --to notebook --execute notebooks/mri_b1_actual_flip_angle.ipynb --output mri_b1_actual_flip_angle_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging101-research`

Pytest: **25 passed**, including per-function, original-source parity, independent physics, full pipeline, numeric metric and standalone Judge contracts. Missing files/keys, wrong shape/dtype, complex, NaN/Inf, zero and poor predictions, missing evaluator assets, caller-independent hidden paths, nested report creation and report-write failure are covered. The Windows default pytest temporary root denied access in this session; the recorded `--basetemp` points into the private archive and does not change tests or mathematical acceptance. This is a host temporary-directory exception, not a task runtime dependency.

Notebook timeout is 180 seconds, comfortably exceeding measured complete reconstruction and rendering on this CPU. Default system `python3` initially selected an unrelated Anaconda kernel; those exploratory runs are not environment validation. Final nbconvert explicitly used an isolated `imaging101-research` kernelspec via private JUPYTER_PATH with argv pointing to the validated research runtime. The executed notebook prints actual sys.executable and NumPy version. Benign Windows ZMQ selector-thread and debugger/frozen-module warnings are preserved in logs; there are no notebook error outputs. The final ledger-aware notebook was separately rerun with subprocess exit 0 in 4.225199 s, and actual research-runtime executable/NumPy2.4.6 plus active ledger agreement were programmatically verified. Canonical notebook contains the executed review outputs; the temporary `output/*_executed.ipynb` was archived and removed.

Visual review: Inspected native middle-slice upstream/fresh/difference panels, three native z=8,16,24 paired observation/field displays, finite-TR relative bias curves at independently specified diagnostic T1 values 500/1000/2000 ms, and metric bars. Full-volume inference is active; slice selection is explicitly display-only. Field equality to upstream is exact on this backend, but the diagnostic shows up to about 1% short-TR bias at T1=500 ms; it is not presented as measured physical truth.

## Semantic plan/source and visibility audit

Reviewed every shipped callable against plan/design.md, and equations against actual numerical call roles. Checked conversion units, angle/frequency sign, normalization, branches, fixed support, parameter defaults, caller flow, file paths, side effects and output order. The exact Bloch diagnostic, where present, is explicitly distinguished from runtime inverse. The current runtime contains no unimplemented historical method. AST import audit is archived as `mri_b1_actual_flip_angle_symbol_audit.json`; all local acquisition/inverse steps are readable NumPy mathematics, and there is no domain-solver API or vendored reconstruction framework. Tests only execute local entrypoint/Judge and consume frozen upstream numeric fixtures; they do not import oracle source. No cross-task import or hidden-answer-derived calibration exists.

The solver-visible allowlist consists solely of README, raw NPZ and requirements/shared environment information. README includes every NPZ key, acquisition metadata, units, axes and full output contract, with no hidden reference paths or GitHub domain URL. No metadata JSON exists. Original source repositories appear only in this maintainer document/NOTICE, never in the solver allowlist. Data are observations and legitimately available independent calibration/support, not a hidden target or privileged extra measurement.

## Mandatory completion gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper/version, exact source revision, data hashes and stage mapping | passed | Frozen selection, hashes, table above; inaccessible optional paper PDFs explicitly distinguished |
| Original and cleaned code executed on exact bound observations | passed | Original capture log/record; repeated original source run; final main exit0 |
| Independent original-source parity and physical checks | passed | Comparing tests and independently analytic/Bloch/harmonic checks |
| Constants, units, precision, coordinates and conventions | passed | Source constants, README metadata and per-function/metadata tests |
| Standard entrypoint and fresh standalone Judge | passed | Final main/Judge subprocess exit0 and full report above |
| Invalid/poor submission and evaluator/report failure handling | passed | Contract tests, all failures measured with required exit codes |
| Public observations/environment suffice | passed | Complete forward model and declared acquisition/support in README |
| No truth, privileged calibration, seed/object or URL leakage | passed | Allowlist audit, real independent observation provenance |
| Complete acquisition metadata/output contract, no metadata JSON | passed | README key/output/metadata tables and inventory |
| Minimal shared dependencies and CPU execution without domain libs | passed | Two pinned required packages; separate oracle only |
| Local executable forward/inverse math and call-role audit | passed | Local NumPy code and archived AST/symbol review |
| Unit, metric, Judge, parity, physics and pipeline tests | passed | 25 passed in recorded final pytest subprocess |
| Active complete notebook reconstruction/execution and visual inspection | passed | Final research-kernel nbconvert exit0; reviewed private PNGs |
| Distinct metric/result provenance and justified boundaries | passed | Upstream arrays, fresh measurement ledger, reruns, config/report |
| README/plans match source signatures, units and execution flow | passed | Semantic source review and metadata/forward tests |
| Template inventory, cleanup and original license notices | passed | Inventory/hashes below; NOTICE; private-only original artifacts |

No mandatory gate is marked failed or unverified. Scientific limitations above remain limitations of the selected method, not unexecuted validation claims.

## Numeric artifact hashes

| Task artifact | SHA256 |
| --- | --- |
| data/raw_data.npz | 9e35ca54e8d970b8dfd8c189709d38cfaa70003238a0e52caf6725abf8c4ccf2 |
| evaluation/baseline_reference.npz | d2d869781fd5f3996143f1d046b95d5c99b62d078fd9faabe1d62db72e8f7847 |
| evaluation/metrics.json | f5d307d103ea0d38fb6ba82252c0284483ac41c7b37d09f8a5f24ca7d2492d78 |
| notebooks/reference_outputs/fresh_metrics.json | 9f6d5bfede0c47048128cc93f49bdcbe31f6817c81ecdf2235049b9f1f4d8842 |
| notebooks/reference_outputs/upstream.npz | d2d869781fd5f3996143f1d046b95d5c99b62d078fd9faabe1d62db72e8f7847 |
| output/judge_report.json | 51216dad0428b9d9fb511f380572e1953a193759ec5eb52b96903030944b6fbe |
| output/reconstruction.npz | d2d869781fd5f3996143f1d046b95d5c99b62d078fd9faabe1d62db72e8f7847 |
| test/fixtures/upstream_stages.npz | 763679f4a5b049eb5cb691da05b06af25a45aec23c624b71df3ec60b94e41088 |

## Final review inventory

- `data/raw_data.npz`
- `evaluation/baseline_reference.npz`
- `evaluation/judge.py`
- `evaluation/metrics.json`
- `notebooks/mri_b1_actual_flip_angle.ipynb`
- `notebooks/reference_outputs/fresh_metrics.json`
- `notebooks/reference_outputs/upstream.npz`
- `NOTICE`
- `output/judge_report.json`
- `output/reconstruction.npz`
- `plan/approach.md`
- `plan/design.md`
- `README.md`
- `requirements.txt`
- `src/__init__.py`
- `src/main.py`
- `src/physics_model.py`
- `src/preprocessing.py`
- `src/solvers.py`
- `src/visualization.py`
- `test/comparing_test/test_parity.py`
- `test/fixtures/upstream_stages.npz`
- `test/unit_test/test_contract.py`
- `test/unit_test/test_physics.py`
- `test/validation.md`


## Independent parent acceptance (2026-10-20)

Parent independently reran all 25 tests with exit 0, ran the standalone Judge from a foreign working directory (exit 0), reviewed source mathematics, plans, output and public-input contracts, and inspected all 4 actual notebook figures. After display-only label/unit fixes, the complete active notebook was re-executed successfully and updated comparisons were inspected. No numerical source changed after the parent tests. Structural/import/public-input audit found no errors. See private batch parent-audit JSON and test logs. Accepted for this ten-task batch; method-reference agreement is not independent physical truth.


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- OSF project: https://osf.io/tmdfu/; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
