# Provenance and validation

**Status: all mandatory cleaning gates passed. Final native execution, numerical tests, actual visual inspection and independent parent acceptance evidence follow below. Earlier interim UNVERIFIED entries describe history and are superseded by the final observed gate ledger.**

## Binding, exact sources and private archive

Unknown: full1080×1920 dimensionless relative focal-coordinate field. Observations: all19 real author-registered Nexus5 tabletop frames, archive label `balcony`. This is a defocus/focus-stack inverse, distinct from binocular parallax depth and generic image deconvolution. No arbitrary crop, lower-resolution proxy or metric physical-depth truth is substituted.

- Paper: M.Möller, M.Benning, C.-B.Schönlieb and D.Cremers, *Variational Depth from Focus Reconstruction*, IEEE TIP24,5369–5378(2015), https://doi.org/10.1109/TIP.2015.2479469. Actual full15-page archived manuscript: https://arxiv.org/pdf/1408.0173v2, SHA256`732d39ea6b7cdc83d2d069c32dcba23642a446eb7faf9477e03b8758ffdb23c0`. Task-relevant mathematical method, all real/synthetic comparison groups and initialization/convergence discussion were read. Real comparison Figure5/page9 was also rendered and actually opened. The selected later author smartphone dataset is not the paper's ARRI dataset.
- The paper explicitly links its accompanying CUDA port. Exact author repository https://github.com/adrelino/variational-depth-from-focus, revision`0a6342bbc0b896e850c256d48d99b76b5d1ff7cb`, actually obtained from https://codeload.github.com/adrelino/variational-depth-from-focus/zip/0a6342bbc0b896e850c256d48d99b76b5d1ff7cb. Archive2435879bytes, SHA256`56fa28bba68a1060e75625e8cb870cc59b33f54c426dba22a9b55e3b76b9c006`. This verifies exact remote source availability, not current default-branch freshness. All original checkout file hashes match the immutable downloaded ZIP after execution; `unmodified_source_verification.json` records this.
- Data: Adrian Haarbach, *Depth from Focus datasets*, https://doi.org/10.5281/zenodo.438196, author CC BY4.0 release. Exact balcony ZIP8672357bytes, SHA256`b236fd04903e60467c3f74ea42985a83bf1154c00958b8037f5b25d33b8248a2`. Original19JPEG filenames/hashes/order are in `native_data_binding.json`. The ZIP's Apple `__MACOSX/._` resource forks are metadata, not additional acquired images. An initial38-versus19 preflight assertion failed before the original run and was resolved by excluding only those metadata entries.
- Data category: **real_author_data**, reused, no synthetic generator. Native decoder values are captured from the original OpenCV read, before its intensity conversion. `decoded_0` through`decoded_18` are exactly the initial size-check pass through the19 source-ordered images; their BGR uint8 arrays are stacked losslessly into public NPZ. This avoids an alternative JPEG decoder changing numerical observations. There is no target-derived PSF or metric focus calibration.
- Source license: original GPLversion2 text retained unchanged in LICENSE, original attribution retained in code/NOTICE. No additional GPL later-version grant is inferred. The generic OpenCV3.4.16 small-matrix LU arithmetic is adapted locally with the complete3-clause BSD copyright/license notice in NOTICE. Exact OpenCV source/LICENSE URLs and hashes are archived in `generic_opencv_source/source_records.json`; OpenCV is never imported by cleaned runtime/tests/notebook/Judge.

All complete checkouts, paper, ZIP/EXE archives, original EXR/PNG outputs, capture/build/inspection scripts and native logs are in maintainer-only archive `research_batch7_20261022/groups/root/vdff/`, abbreviated P. Private scripts are one level above P, named `*_vdff_*.py`. Complete SDK/extracted portable libraries stay under P/sdk. None of these archives/scripts/domain dependencies are shipped in the task. The archive index records documented consumers and hashes; frozen fixtures are original-run values, not cleaned-function outputs.

## Exact unchanged original execution before local extraction

Actual native command: `P/build/Release/vdff.exe -dir P/bound_native -polyDegree 8 -nrIterations 400 -export P/original_balcony_run1`. The explicit degree8 follows the paper, rather than the port's default6. Other numerical defaults are source dataFidelity6, descentStep8/6, denomRegu0.3, min/max−10/10, initial lambda1, convIterations0, smoothGPUtrue and all frames. `binding_before_original.json` explicitly records `local_method_written:false`.

The unchanged original source builds and runs on Windows11 with existing CUDA12.2, VS2019compiler19.29 and CMake3.20. The GPU is RTX4080SUPER16GiB, driver610.88. Official portable OpenCV3.4.16vc15 and a private POSIX-compatible dirent header supply only native build/image I/O compatibility, not a modified numerical method. Their exact versions/hashes/download records are archived. Initial compile failure for missing dirent and a capture configure backslash-escape failure remain archived. Final original build returns native0/69.0604s; the original source itself is unchanged.

Native original **exit0/4.2030829s**, all400iterations completed; no RNG or seed. The source opens GUI windows and waits only after the completed solve. A private controller hides windows belonging only to that child PID and sends a final targeted key event after the completed-source prompt. No global input or numerical patch is applied. Exact handles/timestamps, command, backend/GPU, runtime, stdout/stderr and native exit are `original_run1.json` and paired logs. The output EXR SHA256 is`b1654952868879045cfea2417b021f8369a15ef5cd8deb62cb8d3f2382651805`.

Private read-only hooks capture decoded observations, raw MLAP, design/normal/inverse matrices, polynomial/negative-derivative coefficients, initializer and the seven state fields at iterations1,2,10,100,400. Their exact modifications are `capture_hooks.json`. This separately built capture run exits0/3.9769373s and its complete final EXR is **byte-identical** to the independent unmodified CLI output. Domain arithmetic is not changed by the capture hooks. These fixtures precede local extraction.

A second build changes only the general compiler flag`--fmad=false`, preserving all domain source. Its full400-step native run exits0 and measures independent compiler-rounding spread. A third run changes only the documented initializer's`meanIndex−1` to`meanIndex`; this is explicitly **qualified corrected-initializer diagnostic**, not an unchanged reproduction, physical truth or a tolerance source. All three captures/logs remain separately identified.

## Predeclared parity/quality and independent mathematics

`frozen_acceptance_criteria.json` precedes the local solver and binds full native observations/degree8/400steps. Full original FMA-on/off NRMSE is4.668425928513327e−6 and cosine NCC0.9999999998335811. Reference-only limits are fixed at1.5times this measured spread: NRMSE≤7.00263889276999e−6, NCC≥0.9999999997503717. No tolerance is derived from local error or from the much larger initialization change. The fixed intermediate trajectory range-NRMSE allowance8e−4 follows the largest original FMA-variant stage spread0.00053194158, not the final local result. Stage polynomial/MLAP/design tolerances are separate and remain fixed.

Before code existed, tests were created against the independent fixtures and failed collection because local modules did not exist; `initial_red_collection.log`. First implementation passed14tests but failed six real halo-initializer cases. Investigation found the source CUDA box traverses column taps before row taps; the first local version reversed this. Correcting traversal order restored all20stage/analytic tests without changing a boundary. Reciprocal-times-score normalization was also preserved instead of replacing it by a different division order. Final numerical primitives use no domain libraries.

Independent checks include a known RGB impulse and image-edge second differences, exact gradient-adjoint dot identity for arbitrary terminal inputs, dense Neumann normal-matrix inverse and constant nullspace, polynomial finite-difference derivative, vector shrinkage exact cases, constant-index proof of the source initializer offset, classical unshifted focal nodes, analytic two-pixel ROF optimum and independent figure/Judge numeric cases. The small source monomial normal matrix is very ill-conditioned; its meaningful constant/quadratic curve evaluation and exact source operators are checked separately rather than pretending a coefficient-space inverse is accurate to machine precision.

An independent worker cross-review confirmed data-ascent sign, split-divergence/adjoint subspace, Neumann eigenvalues, dual rescaling and coordinate/initializer separation. It identified an overly broad early claim of exact FMA simulation. The local helper's binary64 product is exact for binary32 operands, but its binary64 sum can round, allowing a one-ULP double-rounding tie. An exact-rational counterexample now tests this limitation. Arithmetic is unchanged; documentation no longer claims universal FMA bit identity. Native stages and full source comparisons establish actual bound-data numerical accuracy under the already frozen compiler-variation limit.

## Paper/code/data discrepancies and their treatment

1. **Zero-based initializer offset.** Source zero-based argmax is mapped with an extra subtract1. This shifts initialization by−20/18only; the fitted nodes, objective and output coordinates stay unshifted. Main runtime retains and identifies the source initialization convention. Independent exact constant-index case proves the offset. The separately qualified source correction changes the final field (NCC0.99740458, NRMSE0.01848926); its original objective−1859459.48 is slightly higher than the unchanged source−1861312.20. Nonconvexity prevents calling this necessarily more physically accurate. No depth truth exists and no hidden offset fit is applied.
2. **Printed split/dual notation.** The paper's repeated d and dual/sign terms are not mutually consistent. The local/source update independently derives from(lambda/2)||Ku−g+b||²: u uses+lambda K-transpose(g−b), g shrinks Ku+b, and b adds Ku−g. Both plans define this fully. The zero terminal split/dual components make source negative backward-zero differences equal the exact adjoint on the actual subspace.
3. **Degree and dataset.** Paper uses degree8; source CLI defaults6. The primary command explicitly selects8before any original run. The real Nexus5 acquisition is later author data, not a rebranded paper ARRI sample. Relevant MLAP/window/median, joint-weight and post-estimation TV comparisons remain required and explicitly identified on this actual data.
4. **Numerical conventions.** Source coefficient LU uses binary32 normal equations, while small GEMM accumulates in binary64. Source design condition approximately8.93e7 and normal condition7.97e15 imply sensitivity. Local explicit LU/source order preserves the selected method. Source CUDA FCT is replaced only by a general cosine-transform primitive with the same discrete Neumann spectrum; complete-stage/final numerical agreement is separately checked. Approximate FMA limitations are disclosed above.
5. **Clipping and stopping.** The source clips after the unconstrained spatial solve. This is not an exact box-constrained quadratic proximal solution and not asserted as such. Exactly400steps are executed; no global optimum or inferred stopping-tolerance success is claimed.
6. **No metric calibration.** Ordered relative focal nodes are a coordinate convention, not equal metric focus-distance increments. No near/far metric sign is invented. Conditional contrast, TV and reference agreement do not establish optical depth accuracy. The focus margin diagnostic is not posterior variance.

## Equation / source / local / validation traceability

| Paper equation or stage | Original file/function | Local executable stage | Independent validation |
|---|---|---|---|
| Eq1channel MLAP focus response | common_kernels.cu::MLAP | preprocessing.modified_laplacian | Actual source points/full CLI, RGB impulse and edge case |
| Source per-pixel contrast normalization | scaleSharpnessValuesGPU | fit_focus_polynomials reciprocal-times-score | Original raw/normalized coefficient fixtures; actual all-point comparison |
| Degree8 least-squares focus polynomial | DataPreparator::precomputePseudoInverse; matrixPolyfit | polynomial_operator,lu_inverse_float32,fit_focus_polynomials | Frozen X/normal/P and coefficient/derivative points; constant/quadratic and derivative cases |
| Paper focus initializer | smoothDepthEstimate; convolutionShared;findMaxIndices | explicit boxes,strict first maximum,relative mapping | Six native64² windows with18pixel halo guard, four-corner/full-data source measurements, analytic shifted constant case |
| Eq2joint contrast/TV energy | LinearizedADMM update functions | objective plus explicit local inverse | Independent polynomial derivative,gradient-adjoint and proximal tests |
| Eq3/4linearized u update | updateU;LaplaceInversion;FCT | local scalar update,solve_neumann | Original1/2/10/100/400state points; independent dense Neumann solve |
| Split gradient and isotropic proximal update | calcUxUy;updateG;isoShrinkage | gradient,shrink | Analytic vector proximal cases and source split/gradient points |
| Scaled dual and penalty growth | updateB;scaleBWithFactor;run | local explicit b/lambda updates | Original full actual states; source-before-rescale capture convention |
| Paper classical comparison groups | Paper real/simulated experiments | classical_focus windows9/41/21 and median11/5 | Actually executed full-data notebook comparisons; analytic unshifted-node check |
| Paper joint regularity tradeoff | Paper Figure5conditions | full400-step a4/12/28 plus selecteda6 | Actual full-data fresh notebook runs, common contrast/TV metrics |
| Paper two-stage TV comparison | Paper Figure5g/h | explicit tv_denoise_only weights6/26 | Analytic two-pixel ROF optimum and actual complete250-step comparisons |

## Hidden numeric assets and consumers

- `evaluation/baseline_reference.npz`: actual unmodified source full field, keyreconstruction; standalone Judge and notebook comparison. Not actual physical truth. No ground_truth.npz is created.
- `test/fixtures/upstream_stages.npz`: source matrices, nine fixed pixel locations including all four corners and source complete-state measurements; source-stage/complete-execution tests. Frozen criteria bytes document pre-local provenance, never public data.
- `test/fixtures/native_window_0.npz` through`native_window_5.npz`: exact original decoded observation windows plus independently captured original initializer, origin and18pixel halo guard. Used only for independent preprocessing/initializer parity; benchmark still uses every native pixel.
- `notebooks/reference_outputs/source_diagnostics.npz`: separately identified original compiler-noFMA and qualified initializer-correction full results; notebook scientific sensitivity comparison only. Never used as fresh results or solver input.
- `notebooks/reference_outputs/fresh_cli_stage_measurements.npz`: fresh actual standard CLI stage measurements, complete-execution test consumer. No copied upstream result is substituted. Standard solver does not read this file.

| Packaged numeric file | Bytes | SHA256 |
|---|---|---|
| `data/raw_data.npz` | 61988205 | `c4adf98cab47fb3d28d145ff5eb7d2e20f3f7986d95e0968c4bc49e06b62b89d` |
| `evaluation/baseline_reference.npz` | 3953441 | `a49d5605fb0649210a1123bfc14d845e6e2edcb27108dc35f1ad1c2bfeb012f2` |
| `notebooks/reference_outputs/fresh_cli_stage_measurements.npz` | 9631 | `42390803c3cbafe38fd5b19bc168a4f08032b5961d364399cc03651353dd52c7` |
| `notebooks/reference_outputs/source_diagnostics.npz` | 7983487 | `da80f97d6157e719a8e5f8681128e722459ab39774c06d97ea5457fec60e3994` |
| `output/reconstruction.npz` | 3945270 | `36d8ba36fc75f206bf3425cc15a8ed6a9523ef7a1c28dbcb0eff134635536662` |
| `test/fixtures/native_window_0.npz` | 125607 | `daf0392f03722de5ff6d8d362a626ac78fe2310f54ccacd19aeb2d5cefab9fc7` |
| `test/fixtures/native_window_1.npz` | 49416 | `54187f619e24f3d3436077b56e2be5025d84c052557d598f2e3625913c5d56a6` |
| `test/fixtures/native_window_2.npz` | 165421 | `37f7594e2e1dfe8aa724d5cb14f08f37cd00670e8cf423c99e4537c22bfa6b04` |
| `test/fixtures/native_window_3.npz` | 45435 | `c8bf39200ba8f4446dd1596a5b2bd609b8587014cdc502862aa1ab86bef39ee5` |
| `test/fixtures/native_window_4.npz` | 48503 | `5d95f0fdf30b248abe70af915c6a09cbfd773eaaf273cfe80efa17484336889a` |
| `test/fixtures/native_window_5.npz` | 120693 | `5763f7f110037e69bc509f1d90d744e1daf81cd282cde35cc2f70d88930601c8` |
| `test/fixtures/upstream_stages.npz` | 13717 | `0511af1523e44e9758ea05466cad902cbcb10e36583866b2feccf4bc5946a6a5` |

## Actual clean runs and interim gate ledger

The runtime is the existing shared research Python3.11.11, NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1; CPU1thread per process, no new shared packages or task upgrade. Private original build dependencies stay outside cleaned requirements. `clean_environment.json` and `numpy_backend.txt` retain actual platform/executable/backend. Public requirements contain only the cleaned subset; no private path, upstream dependency or source URL is present.

- **PASS** independent unmodified original exact-data/native400step execution and read-only capture invariance, separate compiler/qualified-initializer diagnostics.
- **PASS** first complete local inverse, actual native child0,208.1551125s; NCC0.9999999999434083/NRMSE2.736546037599222e−6, max absolute coordinate difference0.01308441. Conditional objective−1861312.2108028138. These are fresh measured results, not historical paper values.
- **PASS** standard `python -m src.main --diagnostics notebooks/reference_outputs/fresh_cli_stage_measurements.npz`, actual native exit0; reported complete pipeline192.6133944s. Same full output/conditional objective. Command/stdout/actual-session exit are in`clean_main_execution.json` and`clean_main.stdout.log`. It reads no reference/checkpoint.
- **PASS** stage/independent mathematics/Judge interim suite43tests, excluding only still-running canonical notebook execution check; temp fixtures stored in P because the default Windows pytest temporary root returnedPermissionError. That17-case setup error was environmental, not scoring success, and no numerical checks were skipped to accommodate it. The final full suite remains mandatory after the notebook finishes.
- **PASS** current11math expressions parse in both MathJax3.2.2 and KaTeX0.16.22. Source/plan semantic cross-review is separately described above; rendering is not mathematical correctness.
- **UNVERIFIED** native completed canonical nbconvert, all fresh comparison runs and figure inspection until actual completion.
- **UNVERIFIED** final full pytest, final independent standalone Judge, final public/template/import/call-role/hash/cleanup checks and independent parent acceptance.

No mandatory pending gate is called complete. Final execution, visual evidence and all failed/unverified-to-passed transitions must be appended with actual evidence before acceptance.


## Final observed completion and independent parent review

The final canonical six-stage notebook actively reconstructed every one of the
19 native1080x1920 frames, executed all400 primary inverse iterations, every
full400-iteration a4/12/28 comparison and both250-iteration ROF comparisons.
Native `jupyter nbconvert --execute` exited0 in **539.3967288999993s**, with
1800s justified per-cell timeout. Final notebook SHA256
`cac8331eac73d05a6e4b328669b8d4fe5840c3d75b88f8cd68bb8a17a8e9b89c`.
`P/nbconvert.json`, separate stdout/stderr and executed_review.ipynb are actual
process evidence. The canonical notebook retains those actual outputs.

Two earlier complete native-zero notebook runs (524.35168s and494.6959687s)
were archived privately as revision1/revision2. Actual inspection found a
scatter/fit colour mismatch and a3D label overlap; only presentation was
corrected. All8 scientific figure groups were opened; final figure7 was opened
after the final run and has a clear horizontal coordinate annotation. Final
groups1-6/8 are byte-identical to previously opened revision2 figures.
`P/final_completion.json` records these comparisons. No archived reconstruction
was substituted for fresh computation; no mathematical or threshold change was
made to obtain these visual revisions. Numeric figure data and matching
PNG/SVG/PDF files are private under P/figures, with identified provenance.

Final complete parent `python -m pytest test/ -v -p no:cacheprovider` passed
**47 tests**, native0,3.427367800002685s including setup (pytest2.98s).
The private explicit basetemp avoids the earlier Windows temporary-directory
permission error without skipping cases. Full log is the batch-level
`nexus_relative_focal_depth-parent-tests.log`. Independent foreign-working-
directory Judge exited0. Public NPZ/README/requirements, exact template,
imports and actual symbol roles passed the parent audit with errors=[].
All11 extracted formulas passed MathJax and KaTeX without rendering failures.

Parent semantic review checked actual function signatures and caller flow,
degree8 polynomial conventions and explicit LU arithmetic, MLAP column-first
float32 ordering, source initialization, gradient/adjoint boundaries, Neumann
DCT inversion, objective signs/factors, shrinkage, continuation and clipping,
relative focal units, all output pixels and standalone Judge conventions.
The read-only independent worker review identified the rare float64-emulated
FMA double-rounding caveat; the limitation is now explicit and has an exact
rational counterexample test. Dataset parity remains within unchanged limits.

Every mandatory gate is passed: exact paper/source/data/traceability; original
and fresh cleaned native runs; independent source fixtures and mathematical
physics checks; preserved constants/units/numerics; standard CLI/Judge; bad
and malformed submission controls; sufficient public allowlist and no privileged
truth/GitHub leakage; minimal shared runtime and local mathematics; full tests;
actively executed self-contained notebook with required comparisons; correctly
labeled source/fresh/qualified/historical results and justified thresholds;
source/plan consistency; final inventory, preserved copyright and notices.

There is no new synthetic acquisition and no generator. No absolute-depth truth
is invented. All full sources, SDKs, PDFs, ZIPs, original logs, capture scripts,
temporary executed notebooks and reversibly archived Python caches remain
outside this task. P/archive_index.json records private file identities and
roles; the batch parent review records final task file hashes. Preserved
limitations remain scientific qualifications, not hidden failed gates.


## Current benchmark v2 evaluation, 2026-10-29

The v2 Judge uses the explicitly observation-defined degree-eight float64 least-squares contrast polynomial plus TV, accepting any equal or lower attainable mean energy within1e-5. No initialized400-step field identity remains. Existing stored output passes; float32 and an independently computed sparse-gradient lower-energy field pass, while constant/reversed fields fail. No actual depth truth exists. See nexus_pytest.log and nexus_descent.log in the private benchmark_readiness_20261029 archive.

This intentionally changes benchmark acceptance semantics while preserving the validated published reference algorithm and historical parity evidence. No observations, hidden reference NPZ files or stored reconstruction bytes changed. No full fresh blind Agent solve is claimed. Old public/evaluation files are archived privately under benchmark_readiness_20261029 before changes.
