# Authoritative provenance and validation — mri_t1_variable_flip_angle

Validation date: 2026-10-20. Scope is a complete independently executed selected standard method on the exact bound real observations. All results below distinguish upstream measurements from fresh local measurements; there is no physical ground truth and no reconstruction is relabeled as truth.

## Frozen selection and source identity

Selected implementation: qmrpy source at frozen commit; repository https://github.com/SugimotoKohei/qmrpy; exact executed revision `6dbe5ce7b4b6b7057c8138aa2d7cfed7a70e815f`. Remote identity was verified with a successful `git ls-remote` check (a later network retry reset and is not used to assert newer freshness) (PhasorPy v0.7 tag or qmrpy HEAD) and matched the archived checkout. This identifies an actually checked remote revision on the validation date, rather than merely a local tracking pointer. The private checkout is retained; no freshness beyond that check is claimed.

Paper and version: Fram et al., Magnetic Resonance Imaging 5(3), 201–208 (1987), DOI 10.1016/0730-725X(87)90021-X, version of record. Bibliographic identity was checked against publisher and PubMed PMID 3626789. Selected exact equations are additionally presented in the primary qMRLab authors’ complete method document https://qmrlab.org/mooc/t1-mapping/, archived as qmrlab_t1_method.html; the original journal PDF was not accessible and is not falsely listed as downloaded. qMRLab software paper, Karakuzu et al., JOSS 5(53), 2343 (2020), DOI 10.21105/joss.02343, version of record, is archived as qmrlab_joss_2020.pdf. This is a validated modern Python port of the published VFA linearization, not a claim of original 1987 scanner reproduction.

Data and redistribution: qMRLab/NeuroPoly real human-brain VFA example: https://osf.io/7wcvh/download?version=3, OSF node https://osf.io/tmdfu/. The archive is vfa.zip; only VFAData.nii.gz, B1map.nii.gz and the supplied Mask.nii.gz define released observations/calibration/support. OSF metadata is archived as osf_vfa_metadata.json, osf_node.json and vfa_license.json. The node license is CC BY 4.0. All 128×128 pixels are retained, with the single slice axis removed. The original supplied mask contains 4668 voxels. get_fdata float64 preserves the exact decoded stored values; no intensity conversion or resampling occurs. Bundled example FitResults are private historical files and are not used as the fresh oracle or truth.

Code license is MIT; original source copyright/license notices are preserved verbatim in task NOTICE, including qMRLab notices where applicable. Data attribution and conversion changes are also recorded there. Paper PDFs, complete checkouts, original observations, capture scripts and original logs are maintainer-only and are not distributed with this task.

## Maintainer-only archive index

The archive root is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/bio_medical`. Relevant original checkout is `qmrpy/`. Independent capture command is `oracle/Scripts/python.exe oracle_vfa.py`; command was run from the archive with the same process interpreter resolved absolutely. The complete SHA256/size archive index is `archive_inventory.json`. Capture source, full stdout/stderr log `vfa_oracle.log`, machine record `vfa_oracle_record.json`, and raw downloads/data files are retained there. Source capture is independent of all cleaned modules; failing preimplementation import/collection evidence is `vfa_red.log`.

Original independent source file hashes:

| Archived source | SHA256 |
| --- | --- |
| qmrpy/src/qmrpy/models/t1/vfa_t1.py | 0940d4697309eacd88409510ae74f2551db782cb45b5d5c4e20f356fdf43a85f |
| qmrpy/THIRD_PARTY_NOTICES.md | a371b001ecd3585d15bc0cd5e2f837d7b7e197fd0b5b574487b980360b40e0be |
| qmrpy/LICENSE | 8be06e159a442778357f5f03acc42b96f76808e89b8ca2da4e61f8d2ac5d3666 |

`runtime_environment.json` records the cleaned runtime, platform and complete NumPy backend; `oracle_environment.txt` records the complete isolated upstream pip environment; `oracle_environment.txt` records the complete isolated upstream pip environment; `oracle/` contains isolated upstream-only packages. `upstream_reruns.json` and `upstream_reruns.log` record a later independent original-source run on the exact NPZ arrays. `mri_t1_variable_flip_angle_check_0.log` through `_check_3.log` retain all final entrypoint/Judge/pytest/notebook stdout and stderr. `mri_t1_variable_flip_angle_verification.json` records their subprocess exit codes and wall runtimes. Extracted notebook PNGs are private visual-review evidence, listed in that JSON. No private script is imported by source, tests, Judge or notebook.

## Original upstream run, inputs and numeric capture

This is an actual source execution on the bound observations, not an import check, cached notebook or paper metric. No random seed affects this deterministic method. Observation source hashes and capture environment/runtime are:

```json
{
  "runtime_seconds": 0.31626410013996065,
  "python": "3.11.11",
  "numpy": "2.4.6",
  "code_commit": "6dbe5ce7b4b6b7057c8138aa2d7cfed7a70e815f",
  "inputs": {
    "B1map.nii.gz": "680ce52a40f8a943094c9aa27477b70a9250a25d00fe95efd27bb1a8187e1073",
    "Mask.nii.gz": "40b22e5bad4080f8e0b89017886daf5af32195f6eedf2b28cb8290c8b8df5812",
    "VFAData.nii.gz": "c0fa52d43524294858e3ba520216414b4326ab8e901f9de1bf651fa04cb1f688"
  },
  "raw_sha256": "f6c5e946f21f164553be3fb52d94c8ed20b869e3434de3188e41c128f6153675",
  "nifti_zooms": [
    2.0,
    2.0,
    1.9999983310699463,
    0.0
  ],
  "mask_pixels": 4668
}
```

Capture uses Python 3.11.11 and NumPy 2.4.6 on Windows AMD64 with single-thread OMP/OpenBLAS/MKL settings. PhasorPy runs its published v0.7 Windows compiled wheel whose matching source tag is archived; qmrpy runs directly from the exact original archived source. Upstream-only IO packages and dependencies are isolated in oracle and never become cleaned dependencies. Capture exit status is 0; no numerical warnings occurred on supported voxels. Published/historical example outputs in original download archives are retained only for provenance and were not treated as this capture's result.

## Equation, stage and independent test traceability

| Required published/algorithm stage | Original source or independent derivation | Local callable | Validation case |
| --- | --- | --- | --- |
| Spoiled steady-state GRE / Fram VFA acquisition model | qmrpy/models/t1/vfa_t1.py:T1VFA.forward | physics_model.spgr_signal | test_upstream_stages; test_bloch_recurrence_independent |
| Transmit correction a=B1·nominal angle | T1VFA.__post_init__, forward, fit_linear | physics_model.spgr_signal; linearized_fit | test_calibration_changes_lifetime; unit and inverse tests |
| Linearized x=S/tan(a), y=S/sin(a), OLS and physical E inversion | T1VFA.fit_linear; _fit_line_with_outlier_rejection with robust=False/outlier_reject=False | physics_model.linearized_fit | test_upstream_stages; analytic inverse and half-recovery cases |
| Supported full-slice corrected mapping | scalar T1VFA.fit calls in oracle_vfa.py | solvers.reconstruct | test_pipeline; test_metadata_mask_and_forward_consistency |
| Scientifically relevant transmit-uncorrected comparison | T1VFA with b1=1 in independent oracle | solvers.reconstruct_uncorrected | test_pipeline and active notebook comparison |

No equation number is invented where the original source does not provide one. The task-specific full equations, signs, units, normalization and branches are self-contained in README and plan/approach.md.

## Ambiguities, adaptations and scientific limits

Protocol is TR=0.015 seconds in qMRLab and 15 milliseconds in selected qmrpy; both local computation and output explicitly use ms. Angles are degrees until converted once to radians and multiplied by B1. Original NIfTI array x,y orientation is preserved; the upstream teaching document rotates images for display, while this contract expressly uses stored indexing. Original slice spacing 1.9999983310699463 mm is preserved in metadata. Upstream fit defaults to ordinary least squares without robust regression or outlier rejection; local centered OLS is mathematically equivalent and has verified float64 parity. E clipping to [10⁻¹²,1−10⁻¹²] is retained but inactive on all 4668 masked voxels. M0 is an effective scanner amplitude and not absolutely calibrated proton density. The two-angle fit interpolates observations, so its near-zero residual alone does not prove in-vivo model accuracy. No independent true T1 is provided or invented.

## Frozen fixtures and reference consumers

upstream_stages.npz contains every 53rd original masked coordinate, original signal/B1, scalar upstream fitted parameters [T1(ms),M0], forward inputs [M0,T1(ms)], and original scalar upstream predicted signals. Consumer: test/comparing_test/test_parity.py. upstream_uncorrected.npz in reference_outputs is separately identified original-source b1=1 comparison, consumed by parity tests and active notebook.

`evaluation/baseline_reference.npz` is the independently captured standard-method reconstruction used only by standalone Judge and full-pipeline tests; actual ground truth is unavailable, so `ground_truth.npz` is deliberately absent. `notebooks/reference_outputs/upstream.npz` identifies the corresponding independent upstream comparison used in the active review notebook. Its purpose is separately labeled notebook comparison, not a fresh reconstruction. `notebooks/reference_outputs/fresh_metrics.json` identifies fresh cleaned-code source/data/reference hashes, the generated submission and exact full-precision measured metrics; the notebook actively checks this ledger against a new reconstruction. Fresh arrays live in `output/reconstruction.npz`, avoiding a redundant fresh-array copy. Every frozen fixture/reference file has an explicit consumer.

## Tolerances, repeatability and metric provenance

Stage acceptance tolerances were fixed before local extraction: rtol=1e-10 with scale-aware absolute tolerances 1e-8 parameters, 1e-9 signals. Exact copied inputs/constants are not validated by decimal rounding. Analytic tests independently exercise physical equations and units; agreement between source and local implementations alone is not treated as physical proof.

Original-source repeatability evidence on exact NPZ data:

```json
{
  "seconds": 0.1304544999729842,
  "upstream_repeat_max_abs": {
    "reconstruction": 0.0,
    "equilibrium_magnetization": 0.0
  },
  "bitwise_repeat_equal": {
    "reconstruction": true,
    "equilibrium_magnetization": true
  }
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
    "reconstruction_nrmse": 5.7139154018493586e-15,
    "equilibrium_magnetization_ncc": 0.9999999999999999,
    "equilibrium_magnetization_nrmse": 7.906602433684241e-15,
    "observation_nrmse": 3.8508133740517735e-15
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
    "equilibrium_magnetization_ncc": {
      "comparison": ">=",
      "boundary": 0.999999
    },
    "equilibrium_magnetization_nrmse": {
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

- Exit 0; 0.136929 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m src.main`
- Exit 0; 0.149854 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`
- Exit 0; 1.245499 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m pytest test/ -v --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch_20261020\bio_medical\tmp_final_mri_t1_variable_flip_angle`
- Exit 0; 4.529544 s: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe -m jupyter nbconvert --to notebook --execute notebooks/mri_t1_variable_flip_angle.ipynb --output mri_t1_variable_flip_angle_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=imaging101-research`

Pytest: **24 passed**, including per-function, original-source parity, independent physics, full pipeline, numeric metric and standalone Judge contracts. Missing files/keys, wrong shape/dtype, complex, NaN/Inf, zero and poor predictions, missing evaluator assets, caller-independent hidden paths, nested report creation and report-write failure are covered. The Windows default pytest temporary root denied access in this session; the recorded `--basetemp` points into the private archive and does not change tests or mathematical acceptance. This is a host temporary-directory exception, not a task runtime dependency.

Notebook timeout is 180 seconds, comfortably exceeding measured complete reconstruction and rendering on this CPU. Default system `python3` initially selected an unrelated Anaconda kernel; those exploratory runs are not environment validation. Final nbconvert explicitly used an isolated `imaging101-research` kernelspec via private JUPYTER_PATH with argv pointing to the validated research runtime. The executed notebook prints actual sys.executable and NumPy version. Benign Windows ZMQ selector-thread and debugger/frozen-module warnings are preserved in logs; there are no notebook error outputs. The final ledger-aware notebook was separately rerun with subprocess exit 0 in 4.466715 s, and actual research-runtime executable/NumPy2.4.6 plus active ledger agreement were programmatically verified. Canonical notebook contains the executed review outputs; the temporary `output/*_executed.ipynb` was archived and removed.

Visual review: Inspected corrected upstream/fresh/difference T1 and M0 maps, original dual-angle and independent B1 images, measured/fitted voxel curve, and fresh corrected versus fresh uncorrected comparison. Anatomic extent and units are retained; T1 differences against source are around 10⁻⁹ ms, while transmit correction visibly changes the map by hundreds of ms. The notebook explicitly warns that a two-angle exact fit is not independent accuracy evidence.

## Semantic plan/source and visibility audit

Reviewed every shipped callable against plan/design.md, and equations against actual numerical call roles. Checked conversion units, angle/frequency sign, normalization, branches, fixed support, parameter defaults, caller flow, file paths, side effects and output order. The exact Bloch diagnostic, where present, is explicitly distinguished from runtime inverse. The current runtime contains no unimplemented historical method. AST import audit is archived as `mri_t1_variable_flip_angle_symbol_audit.json`; all local acquisition/inverse steps are readable NumPy mathematics, and there is no domain-solver API or vendored reconstruction framework. Tests only execute local entrypoint/Judge and consume frozen upstream numeric fixtures; they do not import oracle source. No cross-task import or hidden-answer-derived calibration exists.

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
| Unit, metric, Judge, parity, physics and pipeline tests | passed | 24 passed in recorded final pytest subprocess |
| Active complete notebook reconstruction/execution and visual inspection | passed | Final research-kernel nbconvert exit0; reviewed private PNGs |
| Distinct metric/result provenance and justified boundaries | passed | Upstream arrays, fresh measurement ledger, reruns, config/report |
| README/plans match source signatures, units and execution flow | passed | Semantic source review and metadata/forward tests |
| Template inventory, cleanup and original license notices | passed | Inventory/hashes below; NOTICE; private-only original artifacts |

No mandatory gate is marked failed or unverified. Scientific limitations above remain limitations of the selected method, not unexecuted validation claims.

## Numeric artifact hashes

| Task artifact | SHA256 |
| --- | --- |
| data/raw_data.npz | f6c5e946f21f164553be3fb52d94c8ed20b869e3434de3188e41c128f6153675 |
| evaluation/baseline_reference.npz | 72a9d18d83bfba5bcf90386ce24bbbeb007cd9f3542b0f14d9785782345f7a4a |
| evaluation/metrics.json | 0fd686135d2d791d6125ea88667753703c9b1d60d5791bce96a7c9817050dd93 |
| notebooks/reference_outputs/fresh_metrics.json | 58c2168d6a77922c888dc8e7d4b7134e19c8ea0c8080e75e2b46bf3a80545143 |
| notebooks/reference_outputs/upstream.npz | 72a9d18d83bfba5bcf90386ce24bbbeb007cd9f3542b0f14d9785782345f7a4a |
| notebooks/reference_outputs/upstream_uncorrected.npz | 39de7e6f9ba6b9371e84f71d222ae245274ef892d3ab0012416c58ce7ae51ce4 |
| output/judge_report.json | 2e46b0c96a45fa36cbd323dd2ee87ed1ebe283ea2c918c2bdcacb93964e4af72 |
| output/reconstruction.npz | d90ecc7a2a4a19e60abcee914872e35fbcfe689daca93a7f2494d20ce80a9d56 |
| test/fixtures/upstream_stages.npz | c5e5112eacef8ff109b0d6b0525f8d1b9c0e37b1642da5c717f79464289e7761 |

## Final review inventory

- `data/raw_data.npz`
- `evaluation/baseline_reference.npz`
- `evaluation/judge.py`
- `evaluation/metrics.json`
- `notebooks/mri_t1_variable_flip_angle.ipynb`
- `notebooks/reference_outputs/fresh_metrics.json`
- `notebooks/reference_outputs/upstream.npz`
- `notebooks/reference_outputs/upstream_uncorrected.npz`
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

Parent independently reran all 24 tests with exit 0, ran the standalone Judge from a foreign working directory (exit 0), reviewed source mathematics, plans, output and public-input contracts, and inspected all 5 actual notebook figures. After display-only label/unit fixes, the complete active notebook was re-executed successfully and updated comparisons were inspected. No numerical source changed after the parent tests. Structural/import/public-input audit found no errors. See private batch parent-audit JSON and test logs. Accepted for this ten-task batch; method-reference agreement is not independent physical truth.


## Final public-link role correction (2026-10-29)

OSF qMRLab collection is mixed observations plus historical FitResults. Existing validation explicitly identifies bundled example fits; only VFAData, B1map and Mask are legitimate public inputs.

The following direct links were removed only from the solver-visible README, while full bibliography, data attribution and license information remain. This does not alter the inverse, raw observations, calibration, outputs, Judge or numeric thresholds. The local provenance already identifies the archived sources; no new remote retrieval is claimed.

- https://osf.io/tmdfu/

No formula was changed and no unrelated reconstruction or notebook rerun is claimed. Private link-role and before-file evidence: benchmark_readiness_20261029/final_public_leak_link_fixes.
