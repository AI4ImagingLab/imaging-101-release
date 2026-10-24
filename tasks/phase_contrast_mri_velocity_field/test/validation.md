# Provenance and validation: dynamic phase-contrast MRI

Status: all implemented execution and scientific gates passed; independent parent acceptance is recorded separately below when complete. No independently measured velocity truth is available. The selected inverse is a full3Dtime-resolved relative tissue velocity field from complex MRI observations, not a claim to implement the entire paper's acquisition, displacement or strain workflow.

## Paper, exact code, data and selected boundary

Selected publication: Maggioni et al., *Vendor-agnostic4DPhaseContrastMRI: a complete open-source pipeline for velocities, displacement, and strain analysis*, arXiv2606.09444v1, submitted2026-06-08,15pages. Complete PDF archived as Maggioni2026_2606.09444v1.pdf, source https://arxiv.org/pdf/2606.09444v1 . Full Methods and task-relevant comparisons/Results/Figures were read, including phase-derived velocities, gradient probing, ESPIRiT/PICS acquisition reconstruction and the separate displacement/strain stages. The latter stages are explicitly not selected, rather than falsely plotted from unprovided calibration or time intervals.

Original author code: https://github.com/BAMMri/Open4DFlow , exact remotely API-verified commit9e752d86644f52c789c3c9392886653feed33052. Original source archive SHA256 b209201e600adcacb201e6dfc086e6bf1a8ae57e80a8bb3842836a6acf30285f. Receipts are provenance.json/remote_commits.json and source.zip. No claim that the revision remains remotely latest after retrieval. Selected analysis/reconstruct_and_process.py SHA25629b0925ff61fe0fbcc1994b48f859248617607c0fda7615e3a2f2979919c7f22; the source on disk is unchanged. GPL3-or-later author code license and Copyright2025MartaMaggioni are preserved in LICENSE/NOTICE/local headers.

Real author data: https://zenodo.org/records/20085845 , *Test data for open source MRI4Dflow/strain pipeline*. Its actual API metadata identifies demonstration volunteer data and CC0-1.0; saved zenodo_record.json. It does not independently identify which anatomy/site/subject is this demonstration. No diagnosis or consent particulars are invented. Original raw MRD gzip208623160bytes SHA256de17ed749735c8f257412ecba7760f42c8e1d59aff1a62bc932060b0c613cdfa; publishedMD5also checked. Complete400349968byteMRD and711byte derived_params.json are private. The latter supplies VENC.16m/s, TE4.7ms, TR7ms,100x67x54matrix,150x100x80mmFOV,27phase labels and known13x10acquisition calibration center. TR is not taken as the time between triggered phases.

Private raw inspection actually ran successfully in27.733s:88776acquisitions,100readout samples,4coils,27phases and4sets, ky0..66, kz1..53with absent samples zero-filled by the source sorting stage. There is no spatial crop/downsampling of the benchmark. Original convert_mrd_to_bart completed on this exact MRD and wrote the complete native complex frequency array. Original acquisition reconstruction then produces four complex encoding images for every voxel/phase. Those images, not velocity estimates, are the bound solver observations. This derived-observation boundary is public and scientifically explicit. No author velocity answer, target-derived coil calibration or hidden motion field enters public raw_data.npz.

The known source sequence uses reference(0,0,0)plus individual first moments0.5e9/(gamma*VENC)alongx/y/z, rather than silently substituting tetrahedral phase combinations. With gamma in cycles/s/T this yields pi*u/VENC phase. Gradient-probing physical polarity metadata are not deposited, so the task preserves acquisition encoding axes and does not invent anatomical direction signs. This limits anatomical interpretation but does not prevent recovery in the declared encoding basis.

## Original exact-observation run and independent captures

All original code, PDFs, source archives, MRD, native CFL, logs and capture scripts are outside the task under D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch5_20261021/root/Open4DFlow. Original CPU environment: WSLUbuntu24.04, Python3.12.3, NumPy2.4.6, SciPy1.17.1, h5py3.16.0, ismrmrd1.15.0, nibabel5.4.2, tqdm4.70.1. Packages are installed only to private oracle-deps; they are not cleaned dependencies or shared upgrades. Host Windows11build26200, Inteli9-14900KF; actual original maximum RSS2174928KiB. No GPU or CUDA computation is used.

BART is confined to this private acquisition verification. The paper cites published1.0.00; annotated tagv1.0.00 was remotely resolved to commit37f3c5f0feaf0a19a7fca85c926ad74101f4a034 and exact source archive SHA256bda588cb8af5f70590628dce282c3aa4f6ed29a8d69e68156fdece436fd70493. A pristine CPU build uses make-j2CUDA=0OMP=1PNG=0NOLAPACKE=0, systemFFTW/BLAS/LAPACK and privately extracted Ubuntu liblapacke/liblapacke-dev/libtmglib3version3.12.0-3build1.1. Package/binary/source hashes, build command and actual exit0 are in root/bart/published_release_build.*. Earlier failed header/shim/link builds and the HEAD build are separately archived and never claimed as successful selected acquisition runs. No computational-imaging library is imported in the cleaned runtime, tests, Judge or notebook.

The initial HEAD exact-source run failed at volume101(venc4,phase20), BART subprocessSIGSEGV followed by None.squeeze. Its original_full/complex_encoding_images.npz is an exception-captured PARTIAL ARRAY, not a valid completed result and never bundled or used as reference. Identical failing-volume commands were independently rerun three times, all successful with identical image hash3480dca76dd28392cc4083797028f58a7b81a3a74565542edee43fe67fa789a5. This establishes intermittent executable failure rather than an unverified mathematical correction; its native cause is not claimed resolved.

Selected successful command: wsl-dUbuntu-24.04--python3 capture_published_original.py. It invokes the unchanged original reconstruct_and_process_all(venc_cms=16,use_gpu=False,ecalib_r='13:10',compressed_coils=0,joint_recon=False), with published-release BART. For all108volumes, original commands are ecalib-c0-m1-r13:10 followed by pics-RW:7:0:0.005-e-i20-S-d5. Every image is actually computed. The private subprocess observer records command/shape/dtype/finite/hash/runtime; on a binary crash only the identical command and identical arguments are retried, never replaced by an archived image/zero/simpler method. Three calls needed such retries; the complete216successful calls and actual exit0 are recorded. Numerical source equations/parameters were not modified. The previous successful original MRD sorting stage is reused unchanged through private CFL links; it is the same exact observations, not another demo.

The successful full invocation took515.0213s(process517.9134s), wrote original_release/complex_encoding_images.npz SHA256a294756d9e10e6c8de7bf45a2c9c338b83e3f18d657cb54ee4ac6b0ad22fc0f5 and complete original velocities/magnitude/mask. Trace capture copies local recon only on a successful function return. It is a read-only observer; no cleaned implementation was available at this stage. Complete acquisition results are not conflated with the failed HEAD partial result. Original backend intermittence is a private verification-environment limitation; cleaned NumPy CPU execution does not invoke that binary.

Before local implementation, private capture_velocity_stages.py compiles exactly original source ASTlines128-149, including principal angles, float32 scale, directionwise time mean and strict reference support. The block runs independently on the exact complete complex observations in WindowsPython3.11.11/NumPy2.4.6. Relative to the completed Linux original it gives max2.9802322387695312e-8m/s, RMS7.910816474037584e-10m/s and DRNRMSE2.0920382016755813e-9; mask agrees exactly. A separate complex128execution of the same original block measures the expected precision variation. These are independently executed original-code measurements, not values generated by cleaned functions.

scope_before_extraction.json and original_velocity_stage_capture.json fix phase tolerance2e-6rad, velocity5e-8m/s, exact mask and nonzero final evaluation tolerances before cleaned implementation. Frozen6x6x6native stage cases retain all27phases; the complete original result separately validates the full grid. First expected test-driven red run fails because local physics_model is absent, preserved in tdd_red.log; after extraction all stage and full-pipeline cases pass without widening tolerances.

## Equation / original / local / independent-test traceability

| Required stage | Original source/function | Local executable function | Independent evidence |
|---|---|---|---|
| Reference and3individual VENC encodings | sequences/4Dflow_undersampled.py:venc_values; paper Methods | physics_model.encode_phasors | analytic signed velocity/VENC/alias tests |
| Complex encoding observations | convert_mrd_to_bart,reconstruct_and_process_all | preprocessing.load_observations,native_order | full original raw acquisition,exact data hash/axes,no hidden target |
| Common phase cancellation and principal argument | analysis/reconstruct_and_process.py128-131 | physics_model.phase_differences | frozen original phases; analytic common-phase invariance |
| Physical phase-to-velocity scale | original line134 | solvers.recover_velocity | known sinusoidal velocity/sign; modulo2Vcase |
| Directionwise temporal gauge before support | original136-140 | recover_velocity | original full27phase fixture; independent changing-mask case |
| Strict observed support and zeros | original142-144 | recover_velocity | exact original mask/magnitude fixture; outside-support physical test |
| Complete output and axis order | original full velocities | reconstruct,main | entire100x67x54x27x3original parity,CLI/freshJudge |
| Independent reference metrics | standard cosine/DRNRMSE | evaluation.compare | analytically known vectors,zero/constant cases |
| Independent phase closure | separate real/imaginary complex128forward | evaluation.observation_metrics | full original/fresh data; no solver imports or fitted reference alignment |
| Full temporal scientific curves | actual original temporal summaries | visualization.plot_component_means | independently known direction/time fields and actual artist27sample checks |

There is no spatial derivative, adjoint, custom iterative optimizer or gradient to omit from this closed-form inverse. The independent complex forward tests check its applicable mathematical identities and sign/units. No artificial solver iteration is added.

## References, metrics, data separation and nonuniqueness

Actual velocity truth is unavailable; no ground_truth.npz is manufactured. evaluation/baseline_reference.npz is the complete independently reproduced original velocity result,float32(1,27,3,54,67,100),m/s. Its array values are never read by src. If a legitimate actual truth file exists later, Judge prioritizes it and does not fall back from malformed truth. Standard metrics are cosine NCC with no mean subtraction and RMS/reference dynamic range. Global and each of3directions must pass NCC.9999995andNRMSE1e-6. These nonzero allowances come from the independently measured original OS/backend/precision variation, rather than multiplying a zero self-error. Fixed public temporal gauge/axes/velocity scale are not aligned to a hidden target.

Two independent observation criteria also apply: complex-phasor RMS<=2e-6in the fixed public mean-phase gauge over all3directions and observed support; max velocity outside support<=1e-8m/s. Judge uses complex128products/atan2and exp(i[pi*v/V+mean_ttheta]), with background derived only from public observations, not the submission. Fresh closure2.2264118676856452e-8, outside0. The baseline and these phase criteria validate this selected relative inverse; they do not establish true muscle velocity, correct missing gradient-polarity calibration, fully sampled image accuracy or unwrapped steady motion.

Public raw_data.npz contains only full complex encoding observations. Their312595200byte uncompressed complex payload and compressed hash are listed precisely below; no source velocity, mask inferred from hidden motion, target-derived calibration or seed/object is public. Required scalar acquisition geometry/phase/VENC metadata are in README, not another metadata JSON. The support is a fixed function of public reference magnitude and is explicitly part of the output contract. At each voxel mean removal precedes the time-varying mask; final mean need not vanish and final centered velocity can exceedVENC. Low-signal extrema around±.2m/sremain in the full scored output and histogram, not silently clipped.

test/fixtures/upstream_stages.npz has five original arrays with documented consumers: native complex encoding_images,phase_differences,velocity,mask,reference_magnitude. It is a small independently captured stage case. notebooks/reference_outputs/upstream_measurements.npz holds original3orthogonal velocity sections at alltimes/components plus mean_components,mask_fraction,mask_count; all are consumed/compared in figures. Hidden full baseline serves independent Judge/full-grid parity, not a redundant notebook copy. Private figure numeric_sources.npz stores exactly plotted sections/statistics/histograms and true comparison results, rather than a duplicate full observation archive.

## Execution, scientific figure review and semantic consistency

Fresh standard CLI executed exit0in7.5131s, standaloneJudge exit0in6.7073s. Final complete test suite31passed26.54s in independent parent invocation; first28then30test runs are superseded by added temporal-curve/metadata/asset-priority tests. Full current tests cover independently captured source stages/native result, analytic phase/VENC/gauge/support/alias cases, all numerical contracts, malformed file/key/shape/dtype/complex/object/NaN/Inf, bad/bool/null/nonfinite/negative/out-of-range boundaries, actual true-reference priority including malformed truth over valid baseline, missing references, every threshold direction, poor zero output, foreign-CWD CLI, report-parent creation and report-write errors. Fixtures are not generated by cleaned functions.

Complete active canonical notebook was executed successfully by nbconvert with the shared imaging101-researchkernel. Final execution54.6326s,4code cells in order,6actual inlinePNGgroups,noerrors. Timeout1800spercell is conservative for full native IO and review figures; actual complete runtime is recorded, not assumed seconds. A first58.473snotebook run exposed a temporal-plot indexing error: the global fresh mean array was(time,3)but the plot took a row, showing3samples. It was corrected to all27samples per component, paired colors fixed and two independent known-field/artist tests added; complete active notebook was rerun. No inverse data/result/threshold changed in that figure correction.

Every actual final figure was opened and inspected:1all4magnitude/complex-phase observations;2all3relative component maps at4temporal phases;3native orthogonal magnitude/speed views;4complete27phase means/support/counts/high-signal distributions/nonzero masked-mean behavior;5original/fresh3component maps and actual nm/srounding differences;6fresh5/10/20percent support and gauge comparisons/full untruncated histogram/physical phase closure. Display saturation±3cm/sand speed4cm/sare explicit; complete scored arrays and full-range histograms retain larger weak-signal extremes. No vector arrows imply unprovided anatomical-axis calibration. No paper cohort results/strain curves/other methods are invented or substituted as fresh. FigurePNG/SVG/PDF/numeric/metrics/runtime/manifest are private and hashed.

Semantic review checked all source signatures/defaults, data/return shapes, axes, VENC/units/weak-scalarprecision, complex-product sign, principal-angle branch, exact temporal averaging axis, strict magnitude threshold, mean-before-mask order, actual IO paths and notebook callers against both plans/README. Known public constants are tested. Importing main/library does not change Matplotlib backend; visualization selects none. General NumPyarray/angle/linear primitives only, no opaque inverse or domain library in source/Judge/tests/notebook and no cross-task imports. Runtime dependency subset is NumPy2.4.6andMatplotlib3.11.1; shareddevtoolsremainoutside requirements. CPU execution is validated on the shared WindowsPython3.11.11environment.

Warnings in notebook logs concern local JupyterWindowsselector/TCPtransport configuration, not numerical failure; they are retained without claiming a failed computation passed. Private original binary retries and prior failed build/run histories are explicitly separate. New mandatory scientific gates are not bypassed by dropping a comparison, replacing an output with zero, importing upstream code, relaxing thresholds or inventing truth.

## Mandatory gate ledger

| Gate | Status | Actual evidence |
|---|---|---|
| Paper/exact remote revision/datahashes/equation mapping | passed | full paper,API/archive,table and index |
| Original exact-bound complete observations and inverse | passed with documented private backend retries |216successful calls/108computed images,full logs/capture; failedHEADseparate |
| Independent prelocal fixtures and applicable physics | passed | original sourceAST/fullrun,known signed phase/gauge/alias tests |
| Constants/units/coordinates/precision | passed | complete31tests plus semantic review |
| Fresh standard CLI and standaloneJudge | passed | actual exit0,10metrics |
| Invalid/poor/evaluator/report contracts | passed |31test suite,fullforward and independent CLI |
| Public sufficiency/no target/GitHub leakage | passed | self-contained README/table/output,observation-onlyNPZ,call-role audit |
| Minimal shared runtime and CPU compatibility | passed | only pinnedNumPy/Matplotlib; no original dependencies imported |
| Local explicit forward/inverse and all applicable steps | passed | complex model/direct inverse/temporal support; no artificial iterations |
| Complete test suite | passed |31passed26.54sindependent parent run |
| Complete active notebook and scientific groups | passed | final54.6326s,4codecells/6PNG,noerror,all6opened |
| Historical/original/fresh metrics and thresholds | passed | independent source reports,reference provenance,fullprecisionJSON |
| README/plans/signatures/math/caller consistency | passed | explicit semantic review and known-field tests |
| Template/private archive/license/no temporary task files | passed | inventory and originalGPL3/dataCC0notices |
| Independent parent acceptance | pending | recorded below after strict final audit |

## Actual commands and hash index

Full selected command records, runtimes and exitcodes:

```json
[
  {
    "stage": "cli",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\phase_contrast_mri_velocity_field",
    "exit_code": 0,
    "runtime_seconds": 7.513057300006039,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\cli_final.log"
  },
  {
    "stage": "judge",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\phase_contrast_mri_velocity_field",
    "exit_code": 0,
    "runtime_seconds": 6.707324100003461,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\judge_final.log"
  },
  {
    "stage": "pytest",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\pytest-final-temp"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\phase_contrast_mri_velocity_field",
    "exit_code": 0,
    "runtime_seconds": 29.25450709999859,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\pytest_final.log"
  },
  {
    "stage": "notebook",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/phase_contrast_mri_velocity_field.ipynb",
      "--ExecutePreprocessor.timeout=1800",
      "--ExecutePreprocessor.kernel_name=imaging101-research",
      "--output",
      "phase_contrast_mri_velocity_field_executed.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\phase_contrast_mri_velocity_field",
    "exit_code": 0,
    "runtime_seconds": 58.47297719999915,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\notebook_final.log"
  },
  {
    "stage": "pytest",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\pytest-final-temp"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\phase_contrast_mri_velocity_field",
    "exit_code": 0,
    "runtime_seconds": 27.827448000003642,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\pytest_final.log"
  },
  {
    "stage": "notebook",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/phase_contrast_mri_velocity_field.ipynb",
      "--ExecutePreprocessor.timeout=1800",
      "--ExecutePreprocessor.kernel_name=imaging101-research",
      "--output",
      "phase_contrast_mri_velocity_field_executed.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\phase_contrast_mri_velocity_field",
    "exit_code": 0,
    "runtime_seconds": 54.63256430000183,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch5_20261021\\root\\Open4DFlow\\notebook_final.log"
  }
]
```

Bound independently computed observation/reference/fixture hashes:

```json
{
  "data\\raw_data.npz": {
    "sha256": "1b4ababc17451a9f43df416573de872c3edce32dde664283f85f186583342d21",
    "bytes": 291515650
  },
  "evaluation\\baseline_reference.npz": {
    "sha256": "9e14548a65b7d33c7fe2b6f0b3b73ea6a3c47bf8782ee3521b4ffdee14511ef3",
    "bytes": 59176199
  },
  "notebooks\\reference_outputs\\upstream_measurements.npz": {
    "sha256": "7a8e53ecd5680448ce48a95bca587347961efe83c8f89139fc5eeee26f999c78",
    "bytes": 3441804
  },
  "test\\fixtures\\upstream_stages.npz": {
    "sha256": "019c5e4f02931b32c6979157466bd28f8b6423b2cbe5a38fa460de968ce69116",
    "bytes": 314134
  }
}
```

Fresh10metricreport:

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "ncc": 0.9999999999998646,
    "nrmse": 2.0920382841155924e-09,
    "x_ncc": 0.9999999999999399,
    "x_nrmse": 1.7025534097473565e-09,
    "y_ncc": 0.9999999999999128,
    "y_nrmse": 2.3442871104635158e-09,
    "z_ncc": 0.9999999999999151,
    "z_nrmse": 2.2956571282445864e-09,
    "phase_phasor_rms": 2.2264118676856452e-08,
    "outside_mask_max": 0.0
  },
  "thresholds": {
    "ncc": {
      "comparison": ">=",
      "boundary": 0.9999995
    },
    "nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "x_ncc": {
      "comparison": ">=",
      "boundary": 0.9999995
    },
    "x_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "y_ncc": {
      "comparison": ">=",
      "boundary": 0.9999995
    },
    "y_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "z_ncc": {
      "comparison": ">=",
      "boundary": 0.9999995
    },
    "z_nrmse": {
      "comparison": "<=",
      "boundary": 1e-06
    },
    "phase_phasor_rms": {
      "comparison": "<=",
      "boundary": 2e-06
    },
    "outside_mask_max": {
      "comparison": "<=",
      "boundary": 1e-08
    }
  },
  "errors": []
}
```

Archive index is D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch5_20261021/root/Open4DFlow/archive_index.json. It records exact source/archive/paper/data/metadata/script/log/output/figure hashes and bytecounts, with actual original source and native failed/selected environments retained separately. The complete source checkouts and compiled binary/dependencies can be reconstructed from the indexed exact archives/build records. They are not shipped in this task. Final task inventory is independently audited by the parent and batch delivery manifest.


## Parent acceptance review

Parent independently ran31tests (31passed26.54s), strict inventory/public-array/call-role audit errors[] and foreign-cwd standaloneJudge exit0. Read complete publication and original selected velocity block, all local forward/inverse code, README and both plans; opened all6 actual scientific figure groups, including corrected all27sample time curves and independent artist tests. Full real volunteer100x67x54volume,27phases,3velocity directions are retained. The legitimate observation boundary is four complex MRI images obtained through the actually completed original raw-acquisition reconstruction; it is publicly explicit, not falsely a locally reimplemented CS acquisition solver or supplied hidden velocity target. All108original image volumes and216successful published-release BART calls executed; original source unchanged, private identical-command retries and failedHEADvolume101partial are transparently separated. Complete original numerical results are independent of the cleaned direct inverse. Windows original source-block repetition and full Linux original differ max2.98e-8m/s/DRNRMSE2.092e-9, supporting prelocal nonzero tolerances without relaxing gates. Frozen stages, complete native-field parity and independent signed phase/VENC/common-phase/alias/mean-before-dynamic-mask checks pass. All10Judge metrics score global and3component cosine/DRNRMSE, independent complex128phasor closure2.2264e-8 and outside-support zero.31tests include strict malformed/configuration/reference/evaluator/report/foreign-CWD paths, true-reference precedence, public constants and actual27sample visualization. Actual fresh CLI7.513s/Judge6.707s and final active shared-kernel notebook54.633s/6PNG/noerrors recorded. Scientific display saturation is explicit and full weak-signal extrema retained; all4encoding amplitude/phase,3direction-time maps,3orthogonal views, original/freshnm/sdifferences, support/gauge sensitivities and fulltemporal statistics inspected. No actual motion truth, anatomical signs, calibrated interphase seconds, unwrapped/steady velocity, displacement or strain is invented. GPL3code/CC0realdata rights and exact source/paper/data/build provenance preserved privately. NumPy-only direct mathematics is local, with Matplotlib review-only and no domain/cross-task imports. Semantic axes, precision, signs, VENC, principal branch, averaging/masking order, callable signatures and caller/IO flow agree with plans; no fake solver iteration.
