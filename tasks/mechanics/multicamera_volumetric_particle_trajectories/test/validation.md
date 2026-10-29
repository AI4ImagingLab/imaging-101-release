# Provenance, scientific validation and execution evidence

Worker status: source, qualified-source, local full numerical and current46 delta/stage/physics/contract tests passed. Final active Notebook status is recorded below from its actual execution record; pending is not treated as success. This is private batch8 staging, not release insertion or parent acceptance. Existing accepted tasks remain unchanged.

## Binding, primary source and rights

Unknowns are complete3D particle position records and temporal identity segments from four synchronized native1024²camera views over the full author100frame protocol. All400TIFFs are the author's existing SIMULATED test_STB data, not real measurements or newly generated maintainer observations. No spatial resampling/crop/frame removal is used. The independent existing manifest contains1250000true positions and physical particle identities over100frames. All truth is retained; there is no postfit visibility/quality mask or target-derived camera/OTF calibration.

Paper: Tan, Salibindla, Masuk and Ni, Introducing OpenLPT: new method of removing ghost particles and high-concentration particle shadow tracking, Experiments in Fluids61:47(2020), doi10.1007/s00348-019-2875-2. Primary author-institution PDF https://me.jhu.edu/fluidtransportlab/wp-content/uploads/2020/02/Tan2020_Article_IntroducingOpenLPTNewMethodOfR.pdf was actually opened by the web reader:16pages,1247parsed lines. The complete relevant method, validation and comparison descriptions and full body were reviewed across actual response captures. `paper_actual_web_capture.json` retains the initial response and four subsequent full chunks. Local urllib/curl returned403; no successful local PDF download or PDF-byte hash is claimed. The paper's2020 integer-preference pruning, Gaussian constants, IPR/shake counts and performance tables differ from the selected modern code and existing example; no equivalence to those historical results is asserted.

Author project: https://github.com/Shiyong-Tan/pyOpenLPT ; exact remotely verified revision1c669d54fd674d955952c24d31de2014778bdb4f. `native_download_manifest.json` identifies every downloaded source/input file with exact Git blob and SHA256; all selected files and400original image bytes are pinned. Remote metadata was actually obtained by the source-screening worker, not inferred solely from a tracking branch. The repository contains MIT License, Copyright(c)[2024][Shijie Zhong], retained verbatim in LICENSE and attributed in NOTICE. Modern source is author-project code, not claimed identical to the2020 paper implementation.

The actual independent truth download is https://media.githubusercontent.com/media/Shiyong-Tan/pyOpenLPT/1c669d54fd674d955952c24d31de2014778bdb4f/test/inputs/test_STB/manifest.csv,296881711bytes, SHA25686b913b70a0d61c12d29be2261551ae13521a43a389d1a9679b02bdfd3655e35. The134byte pointer in the checkout is not loaded as truth. GitLFS expected hash and actual download agree. Data-specific redistribution permission is not separately stated; we preserve source MIT notices without inventing a separate CC data license. This work is authorized local research/benchmark work, with external redistribution rights explicitly unresolved. No external publication/upload or author contact was performed.

Public raw contains only full native observations and supplied K,R,algebraicRinverse,t,cameraorigins,distortion. Native integer values are losslessly stored uint32; calibration float64. Negative focal signs and rounded nonorthogonal rotations are preserved. The0.04mm author voxel conversion is a source configuration unit, not a resampled image grid. Nominal1Hz is the author's runtime time convention, not independently measured acquisition metadata. OTF is estimated afresh from first5images/camera by the original observation-only procedure; no hidden fitted OTF is shipped as public calibration.

## Original source runs before extraction

Private archive P is `research_batch8_20261022/groups/astro_earth_mech/openlpt`, outside the task. Full source is P/native_source, native source build is private WSL `/tmp/imaging_b8_openlpt_build`, private TIFF/CMake tools `/tmp/imaging_b8_openlpt_tools/prefix`; no shared package installation/upgrade. WSLUbuntu24.04/GCC13/C++20/O1, CPU1. Original command/config/binary/data provenance and native logs are in original_native_process.json and original_native_stdout/stderr. Only the source-supported thread setting0→1 was selected; exact full0..99frame/four-camera input protocol remained.

Unchanged original native exit0,292.942296744s; independent complete repeat exit0,315.4559s. Original binary SHA2561bf957d792d77b6c5cd2dafdd8ef072057afe26d5bed21cc8d2f1430f6b0f37d; config f9e28ac07552f8b2f2cdb30e5b29999852c7b6fb576a0f9fed57f90b0b1c75d8. Full four final category CSV hashes are identical across original/repeat/successful output-only capture, independently checked in full_rerun_and_passive_parity.json.

The first passive capture failed with invalid pointer at frame35/native-6,140.19s; its stderr/stdout/processes are retained and it is not a successful run. The same capture binary under GDB then completed all100frames; the inferior exited normally after287.50s, full original CSV bytes identical. GDB supervisor1/300.6319s results from post-exit `bt` without a stack, not native failure. An earlier wrong-config-directory debug attempt failed camera loading and is not claimed scientific success. Original success, capture failure and successful inferior are distinguished.

Source saves track IDs starting0 separately in LongTrackActive,LongTrackInactive,ExitTrack,ShortTrackActive files. Their IDs are namespaced before combining. Reusing equal IDs from different files would invent false connections. The local method uses separate segment namespaces and preserves every returned row, all100frames. True labels are independent author particle IDs; their numeric values need not equal submitted labels.

## Independent physics and minimal qualified correction

Geometry harness calls actual native source models on fixed known points. Independent explicit camera projection max discrepancy5.7e-14pixel; backprojected known-point/ray error≤7.7e-12mm. Stored inverse rotations are algebraic inverses, not silently replaced by transpose. All four calibrations and cam3 radial distortion1e-6 are retained.

Exact bound-image source-only state probe proves the reusable trial centre is stale across shake axes: axis1 sees trial X13.4708mm versus accepted13.4608mm. ObjectInfo value copying excludes a shared-alias explanation. Config.cpp creates a clone once; Shake.cpp updates only trial active coordinate and accepted object, leaving previous trial coordinates unsynchronized. This is an actual objective-evaluation inconsistency, not a parity discrepancy to copy silently.

Qualified source change is exactly `obj_tmp->_pt_center = obj._pt_center;` at each coordinate-axis entry. Private qualified_source contains selected source/diffs/hashes; every other mathematical stage, input, support, iteration count and stopping rule is unchanged. We explicitly do not claim unchanged reproduction after this correction. Original unchanged outputs remain archived and plotted separately.

Independent known-particle controls use four actual native camera models and explicit Gaussian observations at four specified3Dpositions, starting(.017,-.012,.009)mm away, four decreasing shake widths. All48axis updates and16complete coordinate updates have nonincreasing independently evaluated objective; native0. Final known-position errors are approximately5–7e-5mm. Source-only singleton files/commands/results are in single_particle_control. The shipped generate_data.py generates only these deterministic known-particle physics-control images, not new bound benchmark observations; no noise, clipping or hidden seed is used. Local independent controls repeat final position/image-error improvement.

Qualified full native run and independent output-only full capture both complete all100frames with native0; exact commands/native statuses/runtimes are in qualified_source_processes.json. All four final CSVs are byte identical, qualified_full_capture_byte_parity.json. Ninety-eight actual stage arrays were frozen into qualified_full_success_stages.npz and QUALIFIED_PRELOCAL_FROZEN_ACCEPTANCE.json **before writing local shake**. Original unchanged stages/freeze remain separate. The qualified capture adds only numeric output writes, including16actual LMS histories/results.

## Frozen criteria and actual quality

PRELOCAL_FROZEN_ACCEPTANCE.json established five independent truth boundaries before local work from the unchanged original. Qualification retains them unchanged: pointrecall≥0.8612913600000001, pointprecision≥0.9881026586624504, matched3DpositionRMS≤0.0021250889586245936mm, consecutive-edge recall≥0.8595396363636364, edgeprecision≥0.9880195586014949. Matching radius0.1mm. Recall retains90percent original coverage; precision retains99percent original precision; RMS permits10percent original error. Every spatial/temporal quantity must pass, no zero/singleton shortcut. These are meaningful true-reference metrics rather than NCC on variable point sets.

All1250000true points and1237500true consecutive identity edges are evaluated. Stable per-frame nearest-distance greedy matching gives unique truth ownership; no crop, fitted scale, rigid alignment or subset target. No-match RMS uses finite0.1mm failure penalty and other metrics0. Singletons preserve spatial accuracy but get no correct edges; shuffled IDs fail temporal metrics. Actual source point/edge diagnostics and full failed controls were measured before local freeze.

Qualified source and local each return1198288rows and14290segments with identical segment/frame indices. Pointrecall0.9569872, precision0.9982858878666898, edge recall0.95504 and edgeprecision0.9981959428985522 are identical. Qualified source positionRMS0.0019266109845151344mm; fresh local0.0019266107426291282mm. Full local-source maximum coordinate difference4.999998142185547e-7mm and RMS2.8870205737711176e-7mm reflect six-decimal source CSV export, while exact binary shake fixtures match to7.1e-15mm. No tolerance is relaxed for this expected export quantization; frozen full-stage position tolerance1e-6mm remains.

Both native qualified source and local retain34postshake estimates slightly outside the input/true domain: minima[-20.001442,-20.035311,-20.016226]mm, maxima[20.013908,20.075361,20.015483]mm (native CSVprecision). The original checks volume before prediction/triangulation and does not constrain later coordinate refinement. All34remain in output and full precision/edge scoring; no clamp/delete/mask is added. An erroneous extra output-cube assertion introduced during packaging was removed after this source-consistent diagnosis, without changing any frozen metric boundary. Independent actual truth remains within the cube.

## Local mathematics, environment and stage checks

Owned local source implements plain numeric structs and free functions for PSF detection/estimation, projection/distortion/ray inversion, projector triangulation, epipolar candidate enumeration, percentile competition, Gaussian residual/augmented cost, coordinate descent/ghost/repeat scoring, complete full/reduced IPR, initial displacement histogram/smoothing/interpolation, adaptive LMS, nearest/weighted association/conflict resolution, pruning and category-independent identity. It does not import or call OpenLPT, domain headers, nanoflann, Eigen, upstream binaries, OpenMP or another task. No opaque framework is vendored. Python owns NPZ/contract/full input flow; the embedded readable C++kernel owns explicit numerical loops. General STL/ctypes/compiler calls do not hide acquisition or inverse mathematics.

Windows x64 CPU exception was explicitly approved because full high-density image/trajectory loops benefit from a compiled numeric core. Actual MSVC19.29.30133, _MSC_VER1929, pointer8/double8, C++17 `/O2 /MD /fp:strict /LD`; no fast-math/FMA substitution. Exact compiler path and whitelisted private build environment/probe/source hashes are in compiler_probe_result.json. The runtime initializes vcvars64 then explicitly supplies missing MSVC include/lib paths; no shared environment is mutated. First-use cold compilation writes standard source/log/hash into a temporary content-addressed cache, no runtime download. Standard Python main works independently of the oracle checkout. Only NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1 and standard library are cleaned dependencies; pytest/Jupyter remain development tools.

Initial local kernel stages: all20calibration detector lists bitwise equal, projections max0pixel, rays1.11e-16, OTF coefficients relative7.03e-14. Both frozen stereo candidate/selected ID sets exactly equal. Qualified first shake all9299positions max7.1e-15mm, scores4.4e-15, flags exact. Complete first-frame IPR11830points max8.46e-12mm; all four initial-frame clouds are tested. Sixteen actual adaptive LMS outputs are bitwise equal;51³node displacement field differs≤4.997e-10mm from8-digit source CSV export, within predeclared5.1e-10 export envelope. Independent least-square OTF normal-equation check validates captured design/log/fit. All98fixture arrays have meaningful numeric consumers in test_native_stages.py.

Selected source conventions disclosed in plans: executable bottom/right detector tie differs its comment; reduced-camera tracer selection still has true flags for inactive empty ROIs, vetoing optional vertex/expansion while cost ignores inactive cameras; min-fusion residual differs summed forward subtraction; current mean-percentile competition differs paper integer preferences; VSC is disabled. None is silently replaced by an easier model.

## Traceability: paper stage, upstream, local mathematics, independent case

| Paper equation/stage | Exact upstream file/function | Local function | Validation |
|---|---|---|---|
| Perspective camera calibration and viewing rays | Camera.cpp/project, lineOfSight | project,ray | native geometry fixture, known-point ray test |
| Gaussian Eqs1–3 | Shake.cpp/gaussIntensity, OTF.cpp/estimate | intensity,estimate_otf,local_otf | all four camera OTF/design fixtures, independent known Gaussian |
| Subpixel centre localization, section2.1 | ObjectFinder.cpp/tracer detection | detect | all24actual detector lists, explicit tie case |
| Stereo triangulation | StereoMatch.cpp, myMATH.cpp/triangulation | triangulate,inverse3_original | two complete candidate/selected fixtures, known ray physics |
| Exclusive pruning, section2.2; modern percentile adaptation | StereoMatch.cpp preference/pruning | prune_candidates,stereo_match | actual full candidate IDs and ownership |
| Shaking/local image objective, section2.1 | Shake.cpp/shakeOneObject,calShakeResidue | shake_coordinate_descent,shake_cost | qualified exact stage,48axis independent descent |
| Intensity ghost removal | Shake.cpp/calObjectScore,markRepeatedObj | particle_score,shake_particles | all source scores/flags, known singleton recovery |
| Iterative particle reconstruction | IPR.cpp/runIPR,runSingleIPRIteration | iterative_particle_reconstruction | all four complete initial cloud fixtures |
| Initial particle-space correlation | PredField.cpp/calPredField,findPDFPeakLoc,smoothDispField | DisplacementField | full51³export fixture, independent grid definition |
| Adaptive trajectory predictor | STB.cpp/predictNext | lms_predict |16frozen histories/next results, known constant history |
| Track association/conflicts/linear pruning | STB.cpp/runConvPhase,linkShortTrack,checkLinearFit | numeric_tracking,short_link,linear_consistent | complete100frame ID/frame parity and independent temporal edges |
| Coverage, fragmentation, correctness, raw/smoothed error | Papersection2.3.1/Fig3; author manifest | standalone Judge + visualization diagnostics | analytic toy labels/edges, full truth, singleton/shuffle nulls |

## Contract tests and failure history

First pytest attempt default temporary directory hit Windows permission denial,21passed/21setup errors; it is not a successful mandatory suite. A private explicit --basetemp avoids unrelated system-temp permissions. The next complete numerical suite produced correct full100frame output but failed its Judge due solely to the extra output-cube contract;41other tests passed. `tests_stdout.log`, task_checks.json and that actual fresh NPZ remain private. After the contract repair, independent standalone Judge passes that exact already-computed output, contract_corrected_prior_fresh_judge.json. No repeated numerical solve was needed for that interface-only correction.

Current46unit/Judge/physics/stage tests native0,36.26reported seconds, delta_tests_result.json. They cover each98stage consumer, all initial clouds, independent Gaussian/ray/LMS/OTF physics, analytic metric row/ID invariance, singleton/wrong-ID controls, missing keys/files, wrong shape/dtype/complex, NaN/Inf, malformed frame/ID/duplicate pairs, invalid metric ranges/bools/NaN, actual-truth priority/no fallback, extreme finite poor estimates yielding finite writable failJSON, foreign-cwd paths and report-write failures. Full fresh pipeline numerics and source row parity were separately measured before packaging; final active Notebook reruns the complete standard CLI. The final single-process current47test suite subsequently passed; see final closeout evidence below.

## Notebook, review and artifact inventory

The active six-chapter Notebook tells problem/data/physics/method/results/conclusion, uses explicit shared-Python kernelspec and `%matplotlib inline`, and runs standard `python -m src.main` in a native child on all100frames. It releases its parent observation buffer while the child runs to preserve memory. The child writes actual fresh output and optionally a private diagnosticsNPZ; no cached reconstruction is substituted. Source original/qualified results remain explicitly identified comparisons. No alternate density/DoF/real-camera experiment or paper-era table is fabricated.

Fifteen actual figure groups cover raw native views/details; known camera/ray/OTF; subpixel detections; full first-pass candidate competition/refinement; independent known-particle controls; unchanged/qualified/fresh full truth metrics; spatial errors; complete error distributions; coverage/fragmentation; velocity-colored trajectories/mean-flow subtraction; full raw/smoothed sensitivity; identity-null controls; actual full-image IPR residuals; actual calibration-length sensitivity; complete100frame state evolution. Some comparison methods are explicitly first-pass studies, not unexecuted full sequence alternatives. Each figure exports PNG/SVG/PDF plus paired numericNPZ and JSON consumers/hashes in P/figures. All plots must be opened and inspected before READY.

Public README, both plans, Python/C++symbols/call roles, exact constants/signs/units/array flow and references were semantically reviewed. README has all seven batch-first keys and no public source URLs/hidden targets. Math source escaping is checked semantically in addition to independent KaTeX/MathJax rendering. Temporary notebooks, PDF/archive/sourcecheckouts/logs/figureexports stay outside T. Only the actual canonical executed Notebook remains in notebooks. Reference-output files have distinct unchanged-source,qualified-source,fresh-measurement roles; hidden actual truth has priority. No cross-task import or backend mutation occurs.

## Measured final records

Qualified source full runs:

```json
[
  {
    "label": "full_native_qualified_original",
    "command": [
      "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/astro_earth_mech/openlpt/qualified_source/qualified_original/qualified_native",
      "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/astro_earth_mech/openlpt/native_source/test/inputs/test_STB/config_qualified_original.txt"
    ],
    "cwd": "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/astro_earth_mech/openlpt/native_source/test/inputs/test_STB",
    "native_exit": 0,
    "seconds": 282.28982643700147
  },
  {
    "label": "full_native_qualified_capture",
    "command": [
      "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/astro_earth_mech/openlpt/qualified_source/qualified_capture/qualified_native",
      "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/astro_earth_mech/openlpt/native_source/test/inputs/test_STB/config_qualified_capture.txt"
    ],
    "cwd": "/mnt/d/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/astro_earth_mech/openlpt/native_source/test/inputs/test_STB",
    "native_exit": 0,
    "seconds": 279.51321293500223
  }
]
```

Final active Notebook execution (pending if null):

```json
{
  "command": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staging\\tasks\\multicamera_volumetric_particle_trajectories\\notebooks\\multicamera_volumetric_particle_trajectories.ipynb",
    "--output",
    "openlpt_executed.ipynb",
    "--output-dir",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\astro_earth_mech\\openlpt",
    "--ExecutePreprocessor.timeout=1800",
    "--ExecutePreprocessor.kernel_name=openlpt-shared"
  ],
  "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staging\\tasks\\multicamera_volumetric_particle_trajectories",
  "native_exit": 0,
  "seconds": 301.8302091000078,
  "kernel_argv": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "threads": 1,
  "canonical_sha256": "0073074628069982f71ca561bc5faa8e0f971e62ae6724480168bbb894b07ade"
}
```

Environment: {"python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]", "platform": "Windows-10-10.0.26200-SP0", "numpy": "2.4.6", "scipy": "1.17.1", "matplotlib": "3.11.1"}

Current raw SHA256 `4722603dcedc9810981e74e2bb19397aa810a53fe3b5f56606fbb591e58319ce`; ground_truth SHA256 `8c9447c02c2d05d0384ddaf6b8d476e290ce3fe8eb38ac6bc22e938731aab6ef`. Exact private archive index is P/acceptance_evidence_index.json; full current task file hashes are P/worker_task_hashes.json.

Mandatory gates: source revision/paper/data PASS; original exact full run/repeat PASS; qualified source correction independently proved/full repeat PASS; prelocal stages/physics PASS; local full100frame/truth thresholds PASS; strict Judge/contracts PASS; explicit local math/runtime exception PASS; current46stage/unit/physics/contract checks PASS; single full47suite PASS (final_tests.log); active canonical Notebook PASS; all15actual figure visual inspection PASS (final_visual_review.json); final inventory/semantic review recorded above, parent acceptance separate.


## Final closeout evidence, 2026-10-24

The final unchanged numerical implementation passed the complete single-process command `python -m pytest test/ -v --basetemp <private P>/pytest_final03`: 47 passed in 214.84 seconds, native exit0 (P/final_tests.log). This includes a fresh complete100frame reconstruction, exact qualified-source segment/frame parity and coordinate tolerance, all98frozen stage arrays, physics and standalone Judge contracts. All five original truth boundaries remain unchanged. The extra full pipeline is final test evidence, not a claim that presentation edits required a new scientific run.

The full active nbconvert execution native0/301.8302091000078s produced SHA256 `0073074628069982f71ca561bc5faa8e0f971e62ae6724480168bbb894b07ade`. After that run, only display cells for figures5,7,12 were separately executed by nbconvert native0/24.735381s: integer case ticks; colorbar padding and color domain covering all matches; legend placement. P/visual_revision_equivalence.json proves all paired numerical arrays and all other cells/figures unchanged. The current canonical notebook SHA256 is `def8e01e9e3410eaa3790671b6ccfd82b91b1abea34e2f726c2f04af2382c044`. It combines that complete run with the three actual executed display outputs. It is not claimed that this later byte-identical-to-canonical notebook file was itself rerun end to end. No errors remain in canonical outputs.

All15actual PNGs were opened individually during closeout, including revised5/7/12; labels, units, reference identities, displayed subsets versus complete scored scope, log scales, comparisons and nonfabricated values were checked. No scientific panel/group was removed. Private P/final_visual_review.json identifies the reviewed hashes.

The embedded numerical kernel was changed from a giant repr to a raw multiline literal for reviewability. P/readable_literal_equivalence.json proves complete Python AST identity excluding locations and exact kernel byte identity, SHA256 `43aa46013a0b3d45f21c34108e5e32bacbd5716c8d5d074cb2781cdf37f7e313`; compilation cache identity is unchanged. The final full test suite also exercised this current formatting. All9README formulas rendered in both KaTeX and MathJax with zero failures (P/math_render_check.json); this tests syntax in addition to semantic coordinate/unit review.

Final semantic review checked physical K/R/inverse/distortion signs, ADU/mm/pixel conventions, OTF objective signs and normalization, trial-centre qualification, source residual fusion, inactive-camera flags, LMS history shift/step/stopping, track-category namespaces, full acquisition scope and strict independent five-field Judge scoring. Public inputs contain only observations and supplied calibration; no truth/seed/fitted-target calibration or source URL is exposed. src/generate_data.py is retained solely for independently known control-image generation and is not imported by the benchmark reconstruction. No domain reconstruction library, source checkout or binary, cross-task import, solver-backed Judge or backend mutation occurs. Numerical/source exceptions and historical failures remain disclosed above. Source MIT notice is preserved; separate data redistribution permission remains unresolved and external release is not authorized by this local closeout.

Final file inventory is P/worker_task_hashes.json; fixture-to-test consumers are P/fixture_consumers.json; source/archive evidence remains P/acceptance_evidence_index.json. All task cache directories were removed after verification; no temporary executed notebook, standalone PDF, archive, source checkout, screenshot, private log or compiled binary is shipped. All mandatory local cleaning gates are PASS. Parent scientific acceptance and external redistribution decisions remain separate.


## Parent independent private acceptance,2026-10-24

Parent independently ran all47tests native0/134.083291900s, including complete100frame fresh pipeline, all98frozen-stage consumers, physical and independent Judge contracts. All9formulas passed both mathematical renderers. Parent read README, both plans, local explicit mathematical flow and standalone metrics, and opened all15final PNG groups individually. All worker-frozen task hashes and scientific exports matched before this parent evidence addition. Canonical SHA256def8e01e9e3410eaa3790671b6ccfd82b91b1abea34e2f726c2f04af2382c044 is the full301.8302091s execution SHA2560073074628069982f71ca561bc5faa8e0f971e62ae6724480168bbb894b07ade plus actually executed display revisions for figures5/7/12; equal numerical arrays and unchanged other cells are independently verified, without claiming another full canonical run. Independent read-only semantic review SHA2561e84879d69809db2a9f14436a8d8795ea9fa291cbc32249e72b792effdfb68e2 checked29files, source qualification, coordinate-descent controls,1250000truth points/1237500truth edges and complete unmasked scoring. The actual-source34slightly out-of-domain position estimates remain scored; no coordinate clipping or postfit deletion is introduced. Current modern-author percentile competition and the minimally qualified stale-axis correction are explicit, not falsely paper2020table reproduction. Full numeric kernels are embedded readable standard C++/STL with a validated Windows/MSVC exception; no domain framework or upstream program is called. Parent acceptance is private, pending complete20-task release.
