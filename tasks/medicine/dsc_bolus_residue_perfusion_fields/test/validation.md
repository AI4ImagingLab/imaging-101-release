# Validation and provenance

## Bound inverse and semantic deduplication

Complete author-supplied real128×128×22×80GRE DSC acquisition, external80-point arterial curve and input-only89588-voxel support. Unknown is all160signed flow-scaled residue lags plus relative flow, leakage-corrected relative volume and transit time. All four quantities are scored. No patient perfusion truth, absolute relaxivity calibration, diagnosis/dose/scanner field is invented. The source external AIF's spatial selection/fitting provenance is unavailable. It is a supplied public input, not hidden target calibration.

Existing ASL tasks use tagged arterial-spin kinetics and recover flow/transit; existing DCE tasks infer T1-based compartment exchange. This benchmark instead recovers the full DSC susceptibility bolus convolution response. The different acquisition/unknown/model was checked before local implementation in candidate_binding_before_local.json; it is not another sample or solver of an existing task.

## Source, papers, rights and raw provenance

Original supplied author toolbox: https://github.com/FAIR-Unipd/dsc-mri-toolbox ; exact revision9e1e8cc7349d00ed269072d1fc85cbdb9f754339. Root actually verified that exact commit through the remote API2026-10-22, archived in remote_exact_commit.json. This verifies a concrete remote revision, not a claim that it is current branch HEAD. Original full45-file source/data checkout remains private; original_source_manifest.json indexes every byte hash. Its MIT copyright/license is preserved in task LICENSE. No separate data redistribution grant was identified; NOTICE does not invent CC terms and limits preparation to authorized local research. Public redistribution rights remain unresolved, not a numerical validation claim.

The toolbox is associated with Peruzzo2017 DOI10.1002/mrm.26582, titled Stable spline deconvolution for dynamic susceptibility contrast MRI. Selected task is its published SVD-family baseline, not stable spline and not a claim that this is Wu's original author repository. Full Peruzzo/Wu journal versions were unavailable; only abstracts were obtained and are not called full-text reading. Complete primary Wu2003ISMRM2201 and full Fieselmann2011article support selected deconvolution mathematics. Complete Boxerman2006AJNR859–867publisher-version mirror supports leakage Eqs1–2/AppendixA1–A12; original last10mean/1SD preprocessing differs from supplied source last11median/2SD/CV gate and is explicitly retained as source variant. Full PDFs, text, original checkouts, adapters, captures and logs are outside the task. Exact actual paper files/versions/URLs/hashes:

```json
[
  {
    "file": "wu_ismrm2003.pdf",
    "url": "https://cds.ismrm.org/ismrm-2003/2201.pdf",
    "bytes": 361169,
    "sha256": "3579de4ab13d4d135d90a53453c80e19712adfe0e326658fc509f030fff1b273",
    "pdf_pages": 1,
    "article_pages": 1,
    "extra_pages": "None",
    "note": "Primary complete one-page author conference contribution: Wu et al., Tracer arrival timing insensitive technique for estimating CBF in PWI, Proc ISMRM11(2003)2201. Related journal10.1002/mrm.10522 full text remains inaccessible; abstract is not claimed as full reading."
  },
  {
    "file": "fieselmann2011_publisher_version_mirror.pdf",
    "url": "https://mriquestions.com/uploads/3/4/5/7/34572113/review_perfusion_467563.pdf",
    "bytes": 2538464,
    "sha256": "2c581a1ddc08b0163d3e6dd46c765997e831171275b040ee4be3f3bc30dc913d",
    "pdf_pages": 21,
    "article_pages": 20,
    "extra_pages": "Final PDF page21 is a publisher journal advertisement, not article content",
    "note": "Complete20-page publisher-version review DOI10.1155/2011/467563, hosted on a third-party mirror; actual paper is primary authored text, not a claim that mirror is publisher. Selected section4.2 equations36-45 and AppendixA.1-A.3 inspected for forward discretization, regularization and time units. This review does not make the FAIR toolbox its original code."
  },
  {
    "file": "boxerman2006_publisher_mirror.pdf",
    "url": "https://mriquestions.com/uploads/3/4/5/7/34572113/boxerman_ajnr_uncorrected_maps.pdf",
    "bytes": 406403,
    "sha256": "7e34005aeaab4d86d4c9c16fc30f927ca00baf1caafb4bad95460931bacac038",
    "note": "Complete primary publisher-version AJNR27(2006)859-867,9pages, third-party mirror; methods Eqs1-2 and AppendixA1-A12 actually read. Primary PMC access was challenged; mirror is not misattributed as publisher server. Original paper last10mean/1SD differs from source last11median/2SD/CV gate; source variant retained."
  }
]
```

Original GRE_DSC.nii.gz SHA256 f9f678f4beacb9eba8f29f2217f08bbeab76940420680f3c84f75bd7a4e396e0; aif_conc.mat SHA2566da33041189effac20a106916e5295d56ed1705af156f0cc43cb31db13ddd495. Lossless NIfTI1int16 magnitude samples are read via standard gzip/explicit Fortran ordering, without resampling; original MATLAB niftiread is unavailable under Octave. Original numerical API receives those exact samples through original_input.mat, not a different demo. Source API explicitly permits supplied support/AIF. It receives the same supplied curve in aif.conc and aif.fit.gv because its original external demo lacks the latter field/path. No target estimation is inserted into calibration.

Support was fixed before inverse/local code using only observed time-sum histogram:100bins, two Gaussian SSE fits with independent positive-width multistarts, equal-component crossing6833.82454ADU, per-slice8-connected area>=300and hole filling. Anatomical support89611then excludes23nonpositive-log-input voxels, leaving89588. Every raw intensity remains in public input. This is a maintainer observation-derived mask, not original MATLAB gauss2default-start parity. All fit starts/failures/threshold and the pre-inverse support hash are in external_support_proposal.json/proposed_support.npz; no post-fit quality mask is used.

## Actual original executions and two-line qualification

Initial original demo fails because Octave lacks niftiinfo. External API originally fails missing nanstd; private finite-only nanstd/nanmean/nanmedian compatibility adapters assert finite inputs and use ordinary statistics. They do not claim general missing-value behavior or change inverse math. Two complete attempts terminate with unexplained native139 despite ample headroom; exact logs retained. Invalid-UTF8warnings are retained and not asserted as the crash cause. Debugger then runs the complete unchanged original successfully: original_fault_gdb wrapper0/700.4407529s, actual inverse687.8965859s, explicit COMPLETE marker and Inferior exited normally. This is a real native complete run, not successful import/cached notebook. Original component stage capture36.051s is separately labelled partial concentration/volume/leakage-only, never full inverse evidence.

Independent known-slice example proves source baseline uses incorrect nSlice stride instead of nRow*nCol; actual slice19baseline changes11to4samples. Independent TR1versus1.55covariance test proves source oSVD omits inverse-time factor, unlike sSVD/cSVD. Qualification changes exactly these two mathematical lines, with full before/after hashes/diffs in qualification_changes.json and independent proof JSONs. No cutoff, OI, objective, data, support, Simpson endpoints, leakage rule or source stopping changes. Historical qualification/binding snapshots saying pending are preserved as pre-run records, not final status.

First qualified-v7compressed capture computes all three inverses but breaches the6GiBguard during save (5.511GiB) and is terminated; this failed run/partial file is not accepted. Real-array-v6/v7native and independent SciPy exact round-trip checks pass. Only capture serialization changes to streaming uncompressed-v6; source numerical math stays unchanged. Complete qualified capture and independent repeat both exit normally with explicit COMPLETE/Inferior-normal markers:

```json
{
  "qualified_streaming_gdb": {
    "command": [
      "wsl",
      "env",
      "OMP_NUM_THREADS=1",
      "OPENBLAS_NUM_THREADS=1",
      "MKL_NUM_THREADS=1",
      "gdb",
      "--batch",
      "-ex",
      "set pagination off",
      "-ex",
      "run",
      "-ex",
      "thread apply all bt 12",
      "--args",
      "/usr/bin/octave-cli",
      "--no-gui",
      "--quiet",
      "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/root/dsc/qualified_streaming_gdb.m"
    ],
    "exit_code": 0,
    "seconds": 713.3672678000003,
    "headroom_initial_GiB": {
      "physical": 11.207633972167969,
      "commit": 25.788528442382812,
      "disk_D": 105.45082473754883
    },
    "headroom_minimum_GiB": {
      "physical": 6.873691558837891,
      "commit": 20.77291488647461,
      "disk_D": 103.15967559814453
    },
    "resource_abort": null,
    "scope": "Complete selected original default3-method inverse on full native volume with fixed input-only supplied mask and supplied author AIF; not MATLAB-auto-mask parity or a claim of stable-spline reproduction",
    "capture_format": "MAT-v6 uncompressed capture, after independent real-array v6/v7 exact round-trip proof. No source math or computed output modified; avoids compressed-variable buffer peak."
  },
  "qualified_streaming_repeat_gdb": {
    "command": [
      "wsl",
      "env",
      "OMP_NUM_THREADS=1",
      "OPENBLAS_NUM_THREADS=1",
      "MKL_NUM_THREADS=1",
      "gdb",
      "--batch",
      "-ex",
      "set pagination off",
      "-ex",
      "run",
      "-ex",
      "thread apply all bt 12",
      "--args",
      "/usr/bin/octave-cli",
      "--no-gui",
      "--quiet",
      "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/root/dsc/qualified_streaming_repeat_gdb.m"
    ],
    "exit_code": 0,
    "seconds": 703.4393819999968,
    "headroom_initial_GiB": {
      "physical": 8.406070709228516,
      "commit": 22.122783660888672,
      "disk_D": 102.98737335205078
    },
    "headroom_minimum_GiB": {
      "physical": 7.009136199951172,
      "commit": 20.620315551757812,
      "disk_D": 101.13778686523438
    },
    "resource_abort": null,
    "scope": "Complete selected original default3-method inverse on full native volume with fixed input-only supplied mask and supplied author AIF; not MATLAB-auto-mask parity or a claim of stable-spline reproduction",
    "capture_format": "MAT-v6 uncompressed capture, after independent real-array v6/v7 exact round-trip proof. No source math or computed output modified; avoids compressed-variable buffer peak."
  }
}
```

All47numeric fields/options/input arrays compare bytewise in bounded chunks; only measured elapsed time is excluded. Source MTT has270860outside-support0/0NaNs per method; all89588supported outputs are finite. Declared zero outside input support replaces that sentinel only, without dropping any measured supported voxel. Component source capture qualified_streaming_stage_probe exits0/36.038846s and identifies full concentration/S0/volume/leakage stages separately from the complete inverse run.

## Pre-local criteria, fixtures and physical checks

source_only_acceptance_freeze.json precedes every local acquisition/inverse implementation. Two source reruns are bitwise equal. All four full outputs require cosine>=.999999and dynamic-range NRMSE<=1e-7, plus whole supported-observation relative L2<=.3119447203693109. Nonzero tolerance permits independent linear-algebra rounding; no zero self-error tolerance or after-local relaxation is used. Source oSVD whole observation residual.2970902098755342, per-voxel best constant.794398454618137, zero1. Source sSVD/cSVD residuals.28411153963729063/.24828614382379033. Independent scalar convolution/adjoint, known-slice signal/logarithm, time-unit/linear scaling and manual second-difference checks pass. Agreement with a method does not prove physiological accuracy.

upstream_stages.npz freezes32deterministically input-index-selected supported coordinates, actual source concentration/S0/counts, source inverse/map/OI/leakage/CV/MTT measurements before extraction. Its two matrix entries were independently calculated pre-local mathematical expectations, NOT native capture; this distinction was corrected on review. Supplementary passive original source G captures are in native_matrices.npz, obtained after extraction with the already-fixed1e-14tolerance unchanged. Passive source edits add only save(G), exact diffs/hashes in passive_matrix_capture.json. All three captured native matrices are bitwise equal to the independent pre-local expectations. This supplemental source check is not falsely dated pre-local or generated by a cleaned operator.

Fresh whole-domain parity (not just32probes) compares all89588voxels and every80/160time/lag entry for preprocessing, all six leakage stages and all three method residual fields. Full results in whole_source_parity.json: concentration maxabs2.842e-14; S0exact; oSVD maxabs1.276e-15/NRMSE9.612e-17; cSVD maxabs1.665e-15; sSVD maxabs6.106e-16. High percentage CV entries have maxabs1.466e-5but relativeL2only2.067e-11; their original relative/absolute stage criterion passes, without changing the threshold or hiding large-CV values.

| Paper/source stage | Upstream function | Local function | Independent evidence |
|---|---|---|---|
| Signal relaxation conversion; slice baseline | DSC_mri_conc/S0 | preprocessing.signal_to_concentration | Original stages/full parity; independent two-slice/positive-log case |
| Simpson acquisition and temporal boundary | DSC_mri_SVD/cSVD/oSVD matrix construction | physics_model.convolution_matrix/forward/adjoint | Independent scalar sum/adjoint and supplementary native G capture |
| Truncated singular inverse and physical units | DSC_mri_SVD/cSVD/oSVD | solvers.deconvolve | Source32curves/all3whole fields; independent TR/amplitude control |
| Adaptive OI and signed peak | DSC_mri_oSVD | oscillation_index/deconvolve | Original OI stages, full fields and manual second-difference case |
| Integrated volume; Boxerman-family source variant | DSC_mri_cbv/cbv_lc | leakage_volume | All6full source stages; independent relative volume/MTT identity |
| Volume/flow transit | DSC_mri_mtt | main.reconstruct | All3source MTT fixture comparisons and full primary Judge |
| Cosine/range metrics, exact4quantity contract | Independently specified benchmark | standalone judge.score_arrays/evaluate | Known analytic values, norm-product overflow/underflow cases and native CLI branches |

## Local runtime, tests, notebook and visual review

Clean runtime actual environment:
```json
{
  "python": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "python_version": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
  "platform": "Windows-10-10.0.26200-SP0",
  "numpy": "2.4.6",
  "scipy": "1.17.1",
  "matplotlib": "3.11.1",
  "threads": 1,
  "gpu": "Not used; validated CPU execution",
  "shared_environment_changed": false
}
```
Clean NumPy/SciPy numerical backend/compiler is captured in cleaned_numerical_backend.txt. NumPy/SciPy show_config emits an optional-pyyaml advisory; no unnecessary package is installed. Original WSL Ubuntu24.04/Octave8.4.0,image2.14.0 and linked BLAS/LAPACK records are in upstream_environment.stdout.bin. Original/cleaned use CPU1; GPU not required. Shared packages are unchanged. Original domain environment is never imported by runtime/tests/Judge/notebook.

Standard full main native0/8.0850216s, original fresh standalone Judge0/6.618652s; commands/logs in cleaned_native_runs.json and cleaned_main/judge stdout/stderr. Subsequent active notebook independently reruns the entire selected path. Full47-test command native0, pytest-reported15.78s in final_all47_native.log (UTF16PowerShellcapture). Latest36-case stage/contract rerun also passes0/1.52s after adding source-MTT assertions; no current case is unexecuted and duplicates are not double-counted. Two deliberately triggered overflow warnings are retained; huge numeric inputs correctly return error with empty metrics/finite JSON. Reported native full-pipeline child executes a fresh reconstruction and foreign-cwd independent Judge, not a cached result.

Contract checks cover missing files/keys, exact shapes/dtypes, object/complex/NaN/Inf, all four separately failing outputs, unavailable/invalid reference, actual-truth priority without fallback, bad thresholds, constant-reference degeneracy, norm-product1e150/1e-150known cases, outside-support zeros, foreign cwd and report-write failure. A peer read-only review identified the finite norm-product corner case; a fallback product-free cosine fixes it while leaving normal-case arithmetic/results unchanged. No field threshold is relaxed.

Final complete active notebook:
```json
{
  "command": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "notebooks/dsc_bolus_residue_perfusion_fields.ipynb",
    "--output",
    "dsc_executed.ipynb",
    "--output-dir",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\root\\dsc",
    "--ExecutePreprocessor.timeout=1200",
    "--ExecutePreprocessor.kernel_name=imaging101_pinned"
  ],
  "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staged_tasks\\dsc_bolus_residue_perfusion_fields",
  "exit_code": 0,
  "seconds": 67.03803110000445,
  "minimum_headroom_GiB": {
    "physical": 8.939617156982422,
    "commit": 17.190696716308594,
    "disk_D": 97.56008911132812
  },
  "initial_headroom_GiB": {
    "physical": 10.57790756225586,
    "commit": 19.1495361328125,
    "disk_D": 97.56024932861328
  },
  "resource_abort": null,
  "timeout_rationale": "Full native whole-volume three-method reconstruction, four whole-field regularization variants and four noise variants plus12scientific groups;1200s conservatively allows backend variation, not a seconds-only requirement",
  "canonical_sha256": "ae865aacfdb274a7a111a7948d39aabced7bc65a45860e02df5b1b910889324b"
}
```
The initial no-such-kernel failure is retained; execution then uses the already existing private imaging101_pinned kernelspec via JUPYTER_PATH, without system installation. Native ZMQ/selector warnings are kept and do not imply reconstruction failure. Earlier successful62.034s/10figure and68.038s/12figure notebooks remain historical; final67.038snotebook is actual native0, not a Markdown-only substituted execution. All code cells execute; every fresh figure derives from local modules. The complete three-method comparison plus four whole-domain OI settings/four noise levels executes. All8source-map/volume/transit reference quantities are consumed by the full-volume printed comparison table; no historical metric is labelled fresh.

All12final scientific PNGs were individually opened: real signal/support/AIF; physics/curves/counts; paired original/fresh3methods; four required fields; full-lag residue/OI/cutoffs; paired metrics/nulls; leakage;6fixed whole-volume slices; OI sensitivity; synthetic proxy-noise sensitivity; actual observed/model/residual map; additional explicitly symmetric-log full-range view. Linear signed maps remain alongside nonlinear display. Full extrema and negative values remain in every scored array. The graphics use existing numerical results, not AI-generated scientific imagery. PNG/SVG/PDF/NPZ/caption/hash pairs are in private figures/figure_index.json. No exported figures, executed temporary notebook or duplicate report directory is retained inside output/.

README and both plans were checked against source signatures/units/defaults, signed versus absolute peaks, objective/filter decisions, stop inequality, leakage normalization and caller paths. A peer confirmed physical unit closure dt G k, K2and transit. Stale10figure documentation was corrected to12. The160lag endpoint246.45s/period248s was corrected before the final actual notebook execution. Formula dual-rendering:14formulas, no MathJax/KaTeX failures.

## Reference and fixture consumers

- data/raw_data.npz: only real intensities/external arterial input/input-derived support; public key/dtype/README metadata test.
- evaluation/baseline_reference.npz: independently qualified original four fields, consumed by standalone Judge/full pipeline; never called patient truth.
- test/fixtures/upstream_stages.npz: all original preprocessing/inverse/leakage/MTT/probe arrays consumed by comparing tests; two independently derived matrices explicitly identified above.
- test/fixtures/native_matrices.npz: supplemental actual native acquisition matrices consumed by both matrix tests.
- notebooks/reference_outputs/upstream_maps.npz: complete original3flow/3transit/2volume maps, consumed by the active full-volume comparison table and paired maps.

Final numerical/notebook file hashes:
```json
{
  "data/raw_data.npz": "9b20b5cd34edfbb91b522d6be48d436e7aad7aa33fc1841bb61e5399f66e7a49",
  "evaluation/baseline_reference.npz": "d2f27f40e27cee2a1ae7ded290902d5fea8a9d8029b58ba9223533f5f25cca7b",
  "test/fixtures/upstream_stages.npz": "42f0597e1480aa10aebf13927dbf25ace3e3732fa1c3eb824c2ff0295af44931",
  "test/fixtures/native_matrices.npz": "d1bce5fdfbe1291f17f99bc929d7301a7bbd65a28eb50375b92d2e5beb01ad71",
  "notebooks/reference_outputs/upstream_maps.npz": "d28d4d279a6c1a4009b21f5c691e65442d64cc81729c0fa74e11df2fd64f46b2",
  "notebooks/dsc_bolus_residue_perfusion_fields.ipynb": "ae865aacfdb274a7a111a7948d39aabced7bc65a45860e02df5b1b910889324b"
}
```

## Mandatory gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data/equation mapping | passed | Exact remote/source manifest, actual primary PDFs and stage table above |
| Original and cleaned bound observations | passed with explicit source qualification | Complete original/native qualified/repeat, localmain and full47pipeline; failed attempts retained |
| Independent parity/physics | passed | Pre-local source stages, independent physical cases, supplemental native matrices/full-domain parity |
| Constants/units/coordinates/precision | passed | README metadata test, time-factor proof, slice baseline proof, explicit source conventions |
| Standard NPZ and fresh Judge | passed | Actual main/native independentJudge and complete active notebook |
| Judge poor/malformed/evaluator failures | passed | All4quantities and adversarial/known-case/native CLI tests |
| Public information sufficient | passed | Raw arrays, complete README/output/relative units, environment subset |
| No hidden truth/GitHub leaks | passed | Observation-only allowlist, acquisition-only support/AIF; final parent audit required below |
| Metadata/output tables | passed | Exact3public keys, complete4output arrays and acquisition table, no standalone metadataJSON |
| Minimal shared CPU runtime | passed | Actual pinned environment/backend, no domain dependencies |
| Explicit local math | passed | Local log/Simpson/SVD/OI/leakage/units/stop; peer read-only review |
| Complete tests | passed |47unique cases plus latest36changed-stage/contract checks, native0 |
| Active notebook/visual comparisons | passed | Final67.038s native0,12individual plots opened, all methods/settings actively computed |
| Result/metric provenance | passed | Paper versus native versus fresh identities and pre-local limits explicitly separated |
| Plans/README consistency | passed | Peer and parent semantic signature/unit/path review; latest final12group notebook |
| Inventory/licensing | passed | Exact template/NOTICE; generated caches archived privately with resolved path/hash checks; final independent audit below |

Parent final private acceptance is recorded separately after final inventory/audit. No release/batch8delivery is claimed by this document.


## Final private archive and acceptance

Private archive root: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/root/dsc. Its archive_index.json indexes exact source evidence, capture scripts, complete native arrays, original/qualified logs, paper files, backend/proof records and final plotted numeric arrays with SHA256/size. original_source_manifest.json separately indexes all45original source/data files outside the package. Archived generated caches have an explicit move/hash record and no scientific files were removed. The task output directory contains only the declared reconstruction/report.

Parent independently reran/covered47current unique cases (full native0/15.78s pytest report and latest36changed-stage/contract checks0/1.52s), actually opened all12final scientific PNGs, verified all14formulas in both renderers and checked native complete Notebook0/67.03803110000445s. A peer independently reviewed source/units/qualifications/output contract and identified the resolved finite cosine norm-product corner case. Final source-field parity uses every bound voxel/lag, and fresh full-pipeline child tests execute a foreign-cwd standalone evaluator. This accepts only private staging; aggregate released count remains39/60 and no batch8package is released until batch7delivery is frozen.
