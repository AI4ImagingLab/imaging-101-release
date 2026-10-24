# Provenance and validation: diffusive slab hidden scattering volume

## Binding and source identity

Status: all mandatory gates and independent semantic/parent acceptance passed; private staging pending complete20-task batch release.

Physics domain. Unknown is complete128x32x32 relative scattering-response geometry beyond a homogeneous scattering slab. Observations are all512x32x32 native float32 photon histograms from the author's real letter_t experiment. The original inversion uses first417bins resampled to128 and the full32x32 scan. No region, depth selection or calibration was derived from a hidden target. This double-pass diffusive-slab acquisition differs from earlier ballistic confocal NLOS tasks; semantic search found no prior confocal-diffuse/slab binding.

Original repository: https://github.com/computational-imaging/confocal-diffuse-tomography ; exact remotely verified revision0992bf5a58f972533800e944e4a606f64d7283e0. source_acquisition.json records exact URLs, hashes and file sizes; remote_tree.json records the verified source tree. No claim is made about future remote freshness. Code copyright2020Stanford University, MIT; full LICENSE preserved. Data are source-bundled author-offered experimental observations; no separate explicit redistribution grant was found. NOTICE states the uncertainty; local research staging does not declare a new data licence.

Paper: Lindell and Wetzstein, Nature Communications11,4517(2020), published version DOI10.1038/s41467-020-18346-3, PMCIDPMC7481188. Published JATS, extracted text and equation extraction are archived privately, not shipped. Supplementary PDF SHA256d8108465b2fb7af85d392b73cbfe199f0a5c5daf527e02124d7d3428b37689f4; author_supplement_acquisition.json records the actual successful author/publisher supplement retrieval, distinct from unsuccessful URLs in supplement_acquisition.json. The supplement provides independent slab calibration, finite-image expansion, approximate confocal model, migration and closed-form inverse.

Original MAT SHA256d351015fd4048d4a0e9f4877038df9e696d3aef2f146d48ebce49a6a4502c195. Converted public raw NPZ SHA25662e0322e95338b6aac2f6f90acba38a0fa882c96d36d7d0621f9887aab3ea949; exact observation transpose(2,1,0), batch prefix and lossless NPZ compression only. No storage downcast: original and public observations are float32. Counts are nonnegative integral-valued floats, range0...789, total36,742,324. Public key photon_counts has shape(1,512,32,32). The full stored observation is public, including the95late bins outside the selected original inversion window. Hidden baseline key reconstruction has shape(1,128,32,32), float32, source relative-response units; it is NOT truth. No ground_truth.npz is supplied.

## Private archive and environments

Maintainer-only archive root: D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/physics_materials/confocal_diffuse. Archive index archive_index.json, SHA2568280377f73c180d4c91594238e1c59de91e2d3fca67f2eb4a5050f743498bdd1, contains222 files with exact sizes/hashes. Original checkouts source/ and upstream/, paper/supplement, original logs, capture scripts, source controls and all plotted numerical exports remain outside the task package. Test/kernel transient caches are explicitly excluded from the scientific index.

Original oracle uses Python3.11.11/NumPy2.4.6/Torch2.9.1+cu128 on Windows CPU, one numerical thread, with h5py only to read original MAT and Matplotlib only for upstream imports. An isolated alternate general FFT bridge uses SciPy1.17.1. No upstream domain reconstruction package is installed in the cleaned runtime. The shared environment was inspected and not upgraded. validated_backend.stdout records actual Python/OS/NumPy BLAS/LAPACK and Torch configuration; source script fixes CPU device and all relevant thread counts. Deterministic closed-form execution has no stochastic seed. CPU is supported; no GPU/driver compatibility claim is required. Temporary RAM starts above6GiB available physical/commit; selected array volumes are small. Cleaned requirements contain only NumPy, Torch and Matplotlib; h5py/SciPy are private oracle tools.

## Original executions and test-first extraction

The byte-identical source CLI was run on exactly the bundled observations before local implementation. Removed PyTorch generic rfft/fft/ifft APIs were adapted only with equivalent general FFT primitives; a known60-value round trip max error4.76837158203125e-7 validates the compatibility bridge. No acquisition/inverse formula was changed. Original warning about future meshgrid indexing is retained in stderr. Full original and independent complete repeat are bitwise over all23 captured arrays. Actual process commands, native exits0 and wall times are in source_execution_records.json:

[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\confocal_diffuse\\capture_formal_original.py",
      ""
    ],
    "exit_code": 0,
    "wall_seconds": 5.329998900007922,
    "stdout": "source.stdout",
    "stderr": "source.stderr"
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\confocal_diffuse\\capture_formal_original.py",
      "_repeat"
    ],
    "exit_code": 0,
    "wall_seconds": 5.301717199996347,
    "stdout": "source_repeat.stdout",
    "stderr": "source_repeat.stderr"
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\confocal_diffuse\\capture_alternate_fft.py",
      "_alternate_fft"
    ],
    "exit_code": 0,
    "wall_seconds": 5.32752230000915,
    "stdout": "source_alternate_fft.stdout",
    "stderr": "source_alternate_fft.stderr"
  }
]

An earlier PowerShell wrapper reported1 on a benign stderr warning despite the source completing and writing results; that wrapper is not used as native-success evidence. The subsequent subprocess-based records above directly capture each actual process return code. Initial source preflight remains separately identified and is not substituted for formal captures.

capture_formal_original.py passively observes original function returns; it copies original kernel, transfer, preprocessing, both complete migration stages, deconvolution and final response without altering numerical calculations. formal_original.npz SHA256e89fe48ce73e8a10b682fb96cd6c9b86ff9fb1ebebdda2d4fbc828ba463b8b71. Complete alternate-backend original run uses scipy.fft only in the general legacy API bridge, keeping source operators untouched. It gives final DR-NRMSE6.654339179536791e-5, relativeL2.0009541063811923277 and maxabs.0005067884922027588. Same-backend independent repeat is bitwise. source_repeat_backend_checks.json contains all intermediate envelopes.

Before src existed, source/physics tests were written against independent fixtures and collected with expected ModuleNotFoundError(src.preprocessing), native1. First extracted-stage run passed10cases and failed three analytic-control inputs because inverse-FFT roundoff produced tiny negative values. Independent controls quantify negative L1fractions around1e-16; tests assert fraction<1e-12 before zeroing only these theoretical-control roundoff values. Real measured counts and their validation are unchanged. The stage suite then passed13cases. First contract expansion passed41cases and exposed Windows producer/consumer encoding mismatch; tests now set both child-Xutf8 and parent encoding=utf8. Final full suite is48passed, native0, recorded below. No original fixture was generated by cleaned code.

## Equation/function/test traceability

| Paper or algorithm stage | Upstream file/function | Local function | Independent validation |
|---|---|---|---|
| SI Eq2/10 finite-slab dipole series, Eq12 double passage | cdt_reconstruction.py:setDiffusionKernel | physics_model.diffusion_kernel | full original kernel fixture, normalization, independently generated physical-point observations |
| Discrete data windows/crop/normalization | cdt_reconstruction.py:run | preprocessing.preprocess/load_observations | full128x32x32 original preprocessing exact; public raw-shape and metadata tests |
| SI Eq21/25 diffusion operator | diffusion_fpsf, M/MT | physics_model.convolve | constant flux and independent inner-product transpose identity |
| SI Eq24 Wiener inverse; source pair-channel variant | run:tmp/invpsf, utils.compl_mul/conj | physics_model.paired_wiener/multiply_pair | full padded original deconvolution, independent constant response, alternate original FFT backend and null-term proof |
| SI Eq19 dispersion/Jacobian | utils.fk/interpolate | physics_model.frequency_mapping, solvers.migrate | original sampled full-frequency mapping; complete migration-only source output; independent physical points |
| Combined closed-form and final15slice boundary rule | run:x_npy | solvers.reconstruct | full original final field and actual CLI; no artificial iteration |
| NCC and dynamic-range NRMSE | task-specific independent evaluator | evaluation.judge.measurements and visualization.metrics | separately known [0,1,2,3] vs[1,2,3,4], no mean subtraction |
| Strict NPZ and standalone evaluation | new independent boundary | judge.load_field/evaluate/main | missing/key/shape/dtype/object/NaN/Inf, malformed references, degenerate reference, actual-truth priority, zero/negative/boundary errors, invalid thresholds, overflow and report-write errors |

## Independent physics and discrepancies

physical_point_controls.json uses a separately implemented analytical finite-slab kernel on an independent nominal16ps clock with explicit t0=0,41images, geometric round-trip time2r/c0 and inverse-fourth-power radiometry. Three known targets at.3,.5,.75m are not the real object. Limits9cm axial and15cm lateral were fixed from the published measured prototype resolution before the control run. All unchanged original and fresh controls pass. Source recovered errors are about.0244/.0276/.0256m axial and.0479/.0479/.01597m lateral. Flux is nonnegative up to quantified inverse-FFT roundoff. These checks validate approximate geometry at the stated resolution, not exact real-data absolute calibration.

The following differences remain explicit; none is hidden under a generic reproduction claim:

- Native16ps is only an author comment; MAT contains no timing metadata. Original417-to128 interpolation implies52.4094ps, but diffusion-grid endpoints imply58.5621ps, ratio1.1173956762. No guessed calibration is treated as measured. Output i/127nominal metres follows author plotting, not independently surveyed geometry.
- Inclusive64-point kernel lateral spacing differs from inclusive32-point scan spacing. Both source conventions are preserved and documented.
- The source flux omits an overall1/2 compared with the analytical paper expression; normalization cancels it exactly.
- SI pseudocode multiplies the square-root wave envelope by time; shipped utils.fk does not. Selected current author implementation is identified and preserved, independently checked on geometric points. No claim is made of bit-for-bit equivalence to every paper pseudocode stage.
- Source componentwise paired reciprocal equals conventional Wiener plus i*alpha*conj(H); Hermitian symmetry makes the added inverse-transform component theoretically imaginary. audit_source_filter.py independently derives/tests this identity, with actual original real-space relative disagreement8.875e-5 from cancellation. It is not incorrectly labeled a different real inverse. The notebook actively compares paired/conventional variants and shows large discarded imaginary response; its diagnostic is not substituted for the bound method.
- Source crop and final15zero slices suppress amplified end-boundary noise. The fixed public boundary convention is not a hidden-target mask. Deconvolution-only outputs are signed and remain signed in comparisons; no missing method is replaced by zero.
- The notebook's reblur experiment checks the diffusion/deconvolution stage only; it does not establish a full albedo observation likelihood, calibrated Poisson degrees of freedom or absolute physical accuracy.

source_scientific_precheck.png shows all original comparisons on the full selected real data and was actually opened before criteria freeze. The clear localized T-shaped response is qualitative method evidence, not an invented truth array. PRELOCAL_FROZEN_ACCEPTANCE.json SHA256ba8db5cae5b297a5e250866b710b9743f6276f90345c6aa74b0ec79e5feefc41 predates all local mathematics. Gates are NCC>=.99999, DR-NRMSE<=.0002, max/range<=.0025 and exact nonnegative/zero-tail conventions. Nonzero tolerances come from approximately3times independent whole-source alternate-FFT envelopes, not self-comparison zero error. Reference agreement is not physical truth. No tolerance was relaxed after local results.

## Fresh local run, Judge and notebook

Commands and actual process records:

{
  "main": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-X",
      "utf8",
      "-m",
      "src.main"
    ],
    "exit_code": 0,
    "seconds": 1.9009796000027563,
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staged_tasks\\diffusive_slab_hidden_scattering_volume"
  },
  "judge": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-X",
      "utf8",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "exit_code": 0,
    "seconds": 0.15810520001105033,
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staged_tasks\\diffusive_slab_hidden_scattering_volume"
  },
  "tests": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\physics_materials\\confocal_diffuse\\final_pytest_temp"
    ],
    "exit_code": 0,
    "seconds": 9.96227910000016,
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staged_tasks\\diffusive_slab_hidden_scattering_volume"
  },
  "math": {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\review_math.py",
      "diffusive_slab_hidden_scattering_volume"
    ],
    "exit_code": 0,
    "seconds": 0.30916020000586286,
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staged_tasks\\diffusive_slab_hidden_scattering_volume"
  }
}

Full CLI wall1.900979600s, independent Judge0. Measured metrics:

{
  "ncc": 0.9999999999999956,
  "nrmse": 6.519459343235176e-09,
  "max_error_range": 2.3608480737014048e-07,
  "negative_count": 0,
  "tail_nonzero_count": 0
}

Fresh notebook baseline runtime0.632583600s. Canonical notebook SHA256db2822e6a60436ec7cca417291fa7ff4e67b626f7acc301662ba98b850d162fe is byte-identical to full actual nbconvert executed artifact. Whole notebook native0/40.571362800s, timeout600s/cell justified by measured sub-second inverse calls plus rendering/export overhead. Private cdt-research kernelspec explicitly targets the validated interpreter; notebook prints actual versions and sets inline rendering. All13cells execute, all code counts are present, no error outputs,11embedded PNG groups. Windows ZMQ selector-thread/provisioning warnings are retained in notebook_execution.stderr; they are not inverse failures. No temporary executed notebook remains under output/.

The notebook tells problem/data/physics/method/results/conclusion, computes fresh baseline,3known controls, actual three-SNR and three-scattering-coefficient sweeps, public-energy-rule time gating, diffusion reblur and paired/conventional arithmetic comparison. All11final PNG groups were actually opened individually; common/individual color scales, signed deconvolution, original/fresh labels and nominal-coordinate limitations are visible. No large methods/comparison groups were silently dropped. Final math scan:23 formulas passed KaTeX and MathJax with zero failures; complex README formulas use math fences. Public materials contain no upstream code URLs or hidden reference paths/keys.

## Inventory consumers and semantic review

All keys in upstream_stages.npz are consumed by test/comparing_test/test_source_and_physics.py; final hidden method result is separately consumed by Judge. upstream_measurements.npz is the notebook's labeled original comparison. Independent point-control observation keys are consumed by the test and active notebook; original control reconstructions are consumed by notebook numerical comparison export. fresh_measurements.json is produced by active execution and consumed by this validation/report workflow. External figure_manifest.json maps all11groups to exact PNG/SVG/PDF/NPZ source hashes. No final task file is an unconsumed original-source archive, PDF, capture script, duplicate root wrapper or generated screenshot.

README output shape/dtype/units/axis order and zero-tail convention agree with code/Judge. Acquisition constants have direct consistency tests. Both plans were reviewed against actual signatures, numerical factors, signs, endpoint conventions, exact FFT normalization, pair-channel arithmetic and complete caller flow; method is closed form, so no fabricated iterative update is described. Core code, tests, notebook and Judge use no computational-imaging library or cross-task import; only private oracle scripts import original modules. Judge is independent of solver and checkpoints. Public input allowlist suffices to formulate the inverse using provided observations and explicit scalar acquisition information; single spectral regularization convention is stated conceptually without a hyperparameter table or pipeline pseudocode.

## Mandatory gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data hashes/traceability | passed | Original records, published JATS+supplement, table above |
| Original and fresh complete selected bound runs | passed | source_execution_records.json; final_checks.json |
| Independent source parity and physics | passed | Frozen23source arrays,13stage/physics cases, physical point controls |
| Constants/units/coordinates/precision | passed | metadata test and explicit discrepancies |
| Main NPZ and independent Judge | passed | Native0 commands, fresh measured metrics |
| Negative/malformed/evaluator contract cases | passed | Complete48case suite |
| Public self-contained input and no leaks | passed | README/raw-only description and call-role audit to be independently confirmed |
| Acquisition metadata/no root metadataJSON | passed | README tables |
| Minimal dependency/CPU isolation | passed | Only NumPy/Torch/Matplotlib runtime; backend record |
| Explicit local acquisition/inverse mathematics | passed | Three readable local numerical modules; no domain solver |
| Full unit/parity/physics/pipeline tests | passed | 48tests native0 |
| Active full notebook and scientific visuals | passed | Native0/40.571362800s;11groups actually inspected |
| Historical/original/fresh metric provenance | passed | Separate originals, fresh_measurements.json, independent metric cases |
| README/plans semantic agreement | passed | Actual code/signature/factor/units flow review above |
| Final template/license/privatearchive cleanup | passed | LICENSE/NOTICE, private index, no temporary task notebook or caches |
| Independent semantic peer and parent batch acceptance | passed | Read-only source/units/method review, narrow Judge regression fixed and full current48case/Notebook confirmation; parent audit below |


Additional independent coordinate audit: after local extraction, a specific unresolved axis-order concern justified two new unchanged-original physical controls, (.12,-.16,.5)m and(-.15,.10,.65)m. The independent analytic generator explicitly uses X,Y meshgrid(indexing=ij), matching the public first-spatial-axis x convention. Both original and cleaned outputs meet the same earlier fixed9cm axial/15cm lateral limits and original-stage tolerances. Source lateral errors are.02986/.03537m; axial errors.02756/.02795m. No baseline, criterion, real observations or method changed. offaxis_controls.npz contains independently generated observations and original responses, every key consumed by test_independent_offaxis_coordinate_order. This is additional post-extraction validation, not mislabeled as a prelocal capture. Final complete suite now48cases passes; earlier45case success remains separately archived.


Independent read-only semantic review found no scientific blocker; independent_semantic_review.json records source/plan/operator/units/nominal-clock/paired-null qualifications with13file hashes. Reviewer correctly identified an evaluator corner: a directory named ground_truth.npz must be an evaluator error, not baseline fallback. Parent first reproduced the new regression failing(native1), then changed only the reference-existence test and obtained targeted native0. Final full48case suite and complete actual notebook were rerun after that narrow Judge fix. after_peer_review.json verifies all11final PNG bytes equal the already individually opened figures. Earlier complete47case notebook and logs remain in superseded_47case; this final canonical notebook is the current actual execution, not a presentation-only patch. No inverse formula, observation or frozen boundary changed.


## Parent independent private acceptance

All48tests native0/9.962279100s,23formulas dual-rendered without error. Actual current full notebook native0/40.571362800s, canonical/executed SHA256db2822e6a60436ec7cca417291fa7ff4e67b626f7acc301662ba98b850d162fe byte-identical. Parent opened all11final PNG groups; subsequent full current execution retained exactly those PNG bytes. Every source method comparison and fresh sweep is active. Independent read-only peer source/units/math/interface review SHA2563a4cdc746c06f4d83774360d882ba18bee4222341de982c8a2408f70b8f33019 found no scientific blocker. Its malformed-truth-directory observation was reproduced by failing regression, minimally corrected, and verified in the current full suite/Notebook; peer report itself was not rewritten. Parent semantically reviewed code, both plans, original equations, nominal-coordinate qualifications, frozen numerical criteria and nontruth distinction. Accepted as a conditional relative-geometry inverse, with unavailable actual clock and albedo truth clearly disclosed. Final independent audit confirms standalone foreign-cwd Judge and no public code/truth leakage. This is private acceptance, not an incomplete-batch release.
