# Validation and provenance

## Bound problem, selection and limitations

The full author sample train2.jpg (1080 by1440 RGB), shown in the paper opening train-window example, was fixed before original execution. This is the published single-image conditional transmission-like intensity inverse, not absolute radiance, a uniquely recovered two-layer decomposition or measured transmission truth. All native pixels and channels remain. Observed JPEG values are uint8/255; unknown camera response, mixing fraction, reflection blur and constant reflection offset are not fabricated calibration. The method preserves observed channel means. The independent synthetic review explicitly shows strong-edge and weak-transmission failures. Semantic repository searches distinguish this task from dehazing, intrinsic reflectance/shading, lensless reconstruction and ordinary denoising.

## Primary paper, source revision, data and rights

Yang Yang, Wenye Ma, Yin Zheng, Jian-Feng Cai and Weiyu Xu, Fast Single Image Reflection Suppression via Convex Optimization, CVPR2019. All nine paper pages and three supplementary pages were read. Paper PDF SHA256 0d63f9a99348fcb5a3a7c3ed4c6b62edd098a316ef3379d4daf6c73cffac8e57,7243090bytes; complete paper, supplementary text and original supplementary PDF are private. Paper URL https://openaccess.thecvf.com/content_CVPR_2019/html/Yang_Fast_Single_Image_Reflection_Suppression_via_Convex_Optimization_CVPR_2019_paper.html.

Original author repository https://github.com/yyhz76/reflectSuppress exact remotely verified commit 5a693a2d1ccf7d2b2f991dc4c81c41ece9077aaa. GitHub API remote HEAD evidence was actually obtained at2026-10-21T23:04:46UTC, retained in capture.json/head_response.json. Exact codeload ZIP36441010bytes SHA c8e5f387e734728c02833270dc5a5802f03d7b31ad29a4c20b9abf0c0479dc57; original code/reflectSuppress.m SHA867eeb378d2e622829ca6800b29fabb59668612407c783a8fe4d6b6ed59a0a75. Exact primary JPEG SHA35fb04e7ee6ba05400b1169d7cbd30476d09d83a47bc231909f5768bc48ea258. No upstream redistribution license was identified; do not invent MIT/CC terms. Supplementary material explicitly includes Internet photographs; authorship of every repository image is not verified. NOTICE preserves attribution and this local-research limitation; public redistribution permission is not claimed.

## Original execution before local mathematics

The unchanged original file first failed under Octave because wthresh was unavailable; unmodified.json/stdout/stderr record native failure. The only qualification is the general hard-threshold API compatibility primitive x*(abs(x)>threshold), in compatibility/wthresh.m. Strict greater-than matches the actual MATLAB API; the paper Equation5 prints greater-or-equal. Exact tie cases [-2,-1,0,1,2] at threshold1 independently demonstrate the discrepancy. No source inverse/operator/objective was changed. The unchanged original file's hash remains the same. Qualified command/environment/native status and shim hash are qualified.json: WSL Ubuntu24.04, Octave8.4.0, image2.14.0/signal1.4.5, CPU OMP/BLAS1, native0/5.009455699997488seconds including loading/repeat/save. Complete first/repeat outputs were bitwise equal. Actual primary-only timing is stored in original_capture/primary.mat and listed below.

Full source runs use h=.033 and epsilon=1e-8, the author's exact sample parameters. The read-only instrumentation reflectSuppressCapture.m exposes gradient, Laplacian target and RHS; its complete output is bitwise equal to the unchanged original. Original h=[0,.015,.033,.06,.1] and epsilon=[1e-10,1e-8,1e-6,1e-4] controls were captured independently, native0/22.3060313seconds. Original arrays were frozen before local implementation. Additional three real scenes building, conference_room and Dog-In-Window were preselected before their inverses, full respective dimensions1440x1080,1080x1440,667x1000, same parameters; original native0/6.0417297seconds. Their exact individual JPEG/NPZ hashes, original timing and commands are additional_original_selection.json/additional_originals.json. These are author-repository observations with unknown independent transmission, not extra truth.

oracle_environment.stdout records actual Octave build BLAS_LIBS=-lblas, FFTW3_LIBS=-lfftw3_threads -lfftw3, FFTW thread count1 and planner estimate. An initial --eval environment probe failed from WSL argument quoting, native2 before any inverse; its failed logs remain. Running the environment.m driver succeeded. This argument failure is not represented as a numerical failure or hidden passing run. Shared cleaned runtime uses Python3.11.11/NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1; runtime_environment.txt records numerical backend. No shared package was installed or upgraded. Octave/image and original compatibility helpers are private oracle-only and never imported by cleaned runtime, tests, Judge or notebook.

## Independent physics, equations and frozen criteria

Before local code, independent dense incidence matrices, Neumann cosine eigenmodes, constant preservation, threshold ties and full-image h=0 identity established signs and conventions. Full source stationarity relativeL2=7.940378869677037e-15; maximum channel-mean error3.150075200686331e-10; independent dense/source tiny-case max difference8.840861598358174e-9; complete h=0 identity max difference6.905091221032222e-10; independent cosine-Laplacian discrepancy1.6653345369377348e-15. independent_physics.json and independent_checks.py preserve this evidence. No expected fixture was generated by cleaned functions.

frozen_preimplementation_criteria.json was written before local operators existed. Primary boundaries: cosine NCC>=.9999999, dynamic-range NRMSE<=1e-7, maximum absolute difference<=5e-7, normalized stationarity<=1e-8 and maximum channel-mean difference<=1e-6. Full original stage tolerance1e-12. Independent repeated source results are bitwise; small dense/transform conditioning differences justify nonzero numerical tolerance, not mechanical multiplication of zero self-error. No threshold was relaxed after local results. The epsilon1e-10 contextual control is explicitly more ill-conditioned and does not inherit the primary5e-7 parity claim; all results are still computed and shown. No positivity clipping, output scaling or spatial alignment is used.

| Paper equation/stage | Original function | Local function | Independent evidence |
|---|---|---|---|
| Image-domain focused transmission plus blurred reflection | paper image model/sample acquisition | README; generate_data known-layer experiment | Real image ambiguity explicitly retained; strong/weak independently known synthetic layers |
| Forward differences and zero outgoing flux | grad,div | physics_model.gradient/divergence/laplacian | Adjoint-energy, constants, known cosine eigenmodes, dense incidence |
| Equation5 magnitude hard threshold | reflectSuppress,wthresh | threshold_gradients | Original last-channel gradient fixture; independently known strict threshold ties |
| Convex biharmonic objective, normal equation | reflectSuppress,PoissonDCT_variant | solvers.reconstruct | Full original Laplacian target/RHS; independent stationarity |
| Neumann DCT closed inverse | PoissonDCT_variant,dct2/idct2 | explicit denominator with general scipy.fft.dctn/idctn | Full source output; known dense system and cosine modes |
| Fidelity zero mode | epsilon observation term | reconstruct; independent Judge | Full channel-mean check, constant image, h=0 identity |
| Metrics | separately documented cosine/DRNRMSE | independent evaluation/judge.py | Known metric cases, poor outputs and strict shape/dtype/error contracts |

Local plans were reviewed against actual signatures/defaults, normalization factors, signs, units, shapes, boundary fluxes and caller flow. The source generalized Poisson interface has unused parameters; the shipped selected call uses mu=1,lambda=0. Core acquisition approximation, gradient threshold, objective, Laplacian and spectral denominator are explicit local mathematics. Generic cosine transforms are not a domain reconstruction shortcut. No upstream or cross-task imports, reference loading in solver, or library backend mutation are present.

## Fixture and reference consumers

test/fixtures/stages.npz contains original full Laplacian target/RHS and last-channel thresholded gradient; test_original_full_stages compares all arrays. controls.npz contains original complete five h/four epsilon results, consumed by full control parity and notebook comparisons. known.npz contains original known-mode, constant and exact-tie inputs/outputs, consumed by original known-case tests. evaluation/baseline_reference.npz contains the complete actual source method result, not ground truth. No ground_truth.npz is invented. notebook reference files for the three extra scenes contain their real observations and original results, consumed by the active transfer comparison; fresh_review_metrics.json/fresh_cleaned.npz identify fresh local values separately. The original raw unbatched captures remain private.

src/generate_data.py produces independent review-only focused/transmitted and finite normalized Gaussian-blurred reflected layers with reflected convolution boundaries and known mixture fraction. It also provides Gaussian detector-unit noise sensitivity cases. These are not newly generated bound observations or claimed real calibrated sensor noise. Synthetic mean-adjusted comparison explains the method's reflection-offset ambiguity; unshifted actual synthetic transmission is shown too. No synthetic truth or hidden target enters released observations.

## Active review, tests and figure inspection

The six-part notebook actively runs the full bound benchmark through cleaned modules and independent Judge, then full h/epsilon controls, three further complete real scenes, independent known-layer cases and noise stress. All11actual PNG/SVG/PDF groups were individually opened: full measured RGB/channels; full gradient selection; full original/fresh maps; two explicitly display-only ROIs; complete signed source discrepancy; h tradeoff; epsilon mean-free differences; known-layer success/failure; three full native transfer scenes; measured-unit synthetic noise sensitivity; modal response and full unclipped histogram. Every plotted fresh field was actually computed. The primary control reuses the active reconstruction and its additional runtime is None with explicit role, not a fabricated zero-second run. Display RGB clips only for plotting, annotated with exact out-of-range counts; stored results and all metrics retain negatives/>1. Source primary range[-.021371120655870414,1.0773757632708767].

External Li/Brown, Arvanitopoulos and Wan baselines shown in the paper are not supplied runnable implementations here. The notebook labels their absence and does not claim to reproduce the whole paper leaderboard, substitute archived results for fresh methods, or fabricate missing values. Supplementary video processing is outside the bound single-frame inverse. Independently produced full controls and transfer experiments are preserved.

Final tests include full CLI/pipeline, all source stages/results/known cases, independently known synthetic/operator physics, metrics, missing input/key/shape/dtype/NaN/Inf/poor output, truth precedence, asset failure and report-write failure. Judge is independent of src, NPZ allow_pickle=False, strict declared float64 batch-first output, deterministic finite JSON and native0/1/2. Initial TDD-red failed missing modules before implementation; inaccessible global pytest temporary directory was avoided with scoped TemporaryDirectory fixtures, not filesystem permission changes. Final standard command succeeds. Historical failures and early runs are retained privately.

## Archive and completion gates

Authoritative private archive: ../imaging-101-maintainer/research_batch8_20261022/groups/root/reflection. Associated original complete checkout/ZIP/remote verification remain at ../imaging-101-maintainer/research_batch7_20261022/groups/root/future_math_readonly/yyhz76__reflectSuppress. archive_index.json indexes exact hashes for both roots. Full paper/source/downloads, original MAT/logs/capture scripts, executed temporary notebook, image exports and environment probes remain outside the task. All packaged arrays use NPZ. No standalone metadata JSON, public GitHub URL, hidden truth or target-derived calibration leaks into visible materials.

Local gates: source/paper/data and equation traceability PASS; original exact bound execution/repeat PASS; independent frozen parity/physics PASS; local full extraction/constants/units PASS; CLI/Judge/poor-input contracts PASS; all tests PASS; active complete notebook/comparisons PASS; all11actual figure groups visually inspected PASS; README/plans/signatures/call-role semantic review PASS; seven dual-render mathematical expressions PASS. Final structural audit is appended below. This is PRIVATE batch8staging: release handoff remains pending until batch7delivery is frozen; it does not increase the aggregate accepted count.

Final tests: {"command": ["D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe", "-m", "pytest", "test/", "-v", "-p", "no:cacheprovider"], "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staged_tasks\\single_glass_transmission_field", "exit_code": 0, "seconds": 34.7532768999954}, 30passed.

Final complete notebook: {"command": ["D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe", "-m", "nbconvert", "--to", "notebook", "--execute", "notebooks/single_glass_transmission_field.ipynb", "--ExecutePreprocessor.timeout=1200", "--output", "executed_review.ipynb", "--output-dir", "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\groups\\root\\reflection"], "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch8_20261022\\staged_tasks\\single_glass_transmission_field", "exit_code": 0, "seconds": 94.73169510001026, "timeout_rationale": "Complete1080x1440RGB reconstruction, full h/epsilon/noise sweeps, three additional full real images, independent known-layer cases and11vector/raster science exports;1200seconds allows a genuine full review.", "canonical_sha256": "dd8c35acf2a0b0801ac6be9efe74a0ae992fec8119f04a8b677f1348901ce482", "actual_figure_groups": 11}.

Fresh unrounded Judge: {
  "status": "pass",
  "passed": true,
  "metrics": {
    "ncc": 1.000000000000006,
    "nrmse": 1.3890866529870287e-10,
    "maximum_absolute": 4.598692537882698e-10,
    "stationarity_relative": 5.067320257622532e-15,
    "channel_mean_absolute": 1.8149759473118365e-10
  },
  "thresholds": {
    "ncc": {
      "comparison": ">=",
      "boundary": 0.9999999
    },
    "nrmse": {
      "comparison": "<=",
      "boundary": 1e-07
    },
    "maximum_absolute": {
      "comparison": "<=",
      "boundary": 5e-07
    },
    "stationarity_relative": {
      "comparison": "<=",
      "boundary": 1e-08
    },
    "channel_mean_absolute": {
      "comparison": "<=",
      "boundary": 1e-06
    }
  },
  "errors": []
}


Original MAT scalar timing fields: {"elapsed": 1.189539909362793}.

Final independent structural/visibility/import-call-role audit PASS, errors[], foreign-working-directory Judge0. Evidence: private batch8qa/single_glass_transmission_field-parent-audit.json. No unverified mandatory local numerical gate remains; batch ordering and redistribution authorization are not falsely represented as completed public release.
