# Provenance and validation: sediment lead-210 chronology

All mandatory gates below passed on Windows x86_64 CPU. This is a new inverse problem: real sediment radioactivity → deposition chronology and its coupled accumulation/activity history; it does not duplicate an existing279 task. Parent bound the problem before extraction. No release or other worker directory was edited.

## Sources and exact identity

- Published selected method: J.M. Abril-Hernández, J.Environ.Radioact.296(2026)107989, DOI10.1016/j.jenvrad.2026.107989. Full article archived privately from https://idus.us.es/server/api/core/bitstreams/d58eec91-03ec-4325-97f7-b9f833faf823/content . Article PDF says CC BY-NC-ND4.0; it is not redistributed in the task or relabeled MIT.
- Author repository https://github.com/jmabril1964/TERESApy , actual remotely verified main revision4a6900049d088139553889af7de6f33cf22afa9b, checked from the first line of https://github.com/jmabril1964/TERESApy/commit/main.patch . The archived patch records that40SHA. Selected source files were downloaded from immutable revision URLs; source license is MIT, preserved as LICENSE/NOTICE.
- Real source observations: author Core_C1.txt, all14 measured layers. Author guide Figure1 and article§2.4 identify Lake Łazduny,2007,20m water depth, core LAZ07/02, original Tylmann etal.2013 DOI10.1016/j.quageo.2012.10.001. These are real processed radiometric concentrations, not simulated counts or raw detector spectra.
- Independent actual age measurements: formal article TableS1, https://ars.els-cdn.com/content/image/1-s2.0-S0265931X26001049-mmc1.xlsx ,36066bytes. C1 rows6..19 contain midpoint mass, bottom mass, activity/error, varve age/error. The last five observed/age columns match the bound source and independently measured varve chronology. Hoja1 separately pairs the same bottom mass depths and ages. Data are factual numerical measurements; no full paper/Excel is shipped. Attribution is retained.
- Public raw NPZ SHA256 `5b05c89d7b587a8dbbd3057852fc1d27ec873ebaf344fe9a40ed53ad552cea11`. It has only mass_depth/activity/activity_error, each float64(1,14). No normal proposals, fitted parameters, varve ages, source seed, output activity or target-derived calibration is in raw. README alone supplies all units, geometry and output conventions.

## Original execution and frozen independent evidence

Original CPU environment: D:/Anaconda3/python.exe Python3.12.4, NumPy1.26.4, SciPy1.13.1, MKL2023.1; detailed numerical backend/compiler/CPU capabilities are retained in both original JSON records. OMP/OPENBLAS/MKL threads were each1. Run commands were `D:/Anaconda3/python.exe archive/oracle_teresa.py` and the same command with argument `20261030`; the capture script executes the exact unchanged author Random_generator.py then TERESA_map.py. Seeds20261029/20261030 fix algorithm randomness only. The library generator processes sizes5..99, writes four-decimal paired quantiles, then the exact native map exhausts30^4=810000 cases. Original source sys.exit is caught solely to save full-precision namespace measurements. No paper result/cache was substituted.
First original full map native exit0, 490.587202400s; second independent library native exit0, 439.265409700s. Their minimum χ values are 0.584293879254715 and 0.5196088365866703. Source, commands, seeds and logs are archived. No numerical warnings occurred. Then exact TERESA_clouds.py and TERESA_plots.py were run separately in each original run directory. All three confidence groups and full profiles remain in those private directories. Primary groups are481/7864/25064 (33409 cases), not just a favorable subset.
`capture_teresa_evidence.py` executes original sorting/profile function definitions before any local implementation and saves101 distributed parameter cases, scores and full5×14 profiles. It also reads the independently downloaded TableS1. `frozen_teresa_case.json` was saved before local extraction, after both original full-grid runs and independent varve assessment. It fixes parity rtol1e−9/atol1e−9 for profiles and1e−10 absolute for χ, and all Judge thresholds. No threshold was changed to conceal local error.
The original two normal libraries give age RMSE2.2348190386yr and3.5207094663yr against real varves. Their dynamic-range NRMSE values are0.03724698398 and0.05867849110. The maximum independent original NRMSE×1.1 fixes0.06454634021492335. NCC minimum is0.9×the smaller original NCC. Original observation χ values after native rounded profile exchange are0.58464761861 and0.52015397787; the larger×1.1 fixes0.6431123804714854. Physical consistency boundaries1e−8 are nonzero numerical tolerances independent of target agreement. Independent reruns are scientific sensitivity, not exact-seed tests.

## Equation/function/test traceability

| Published stage | Original file/function | Local function | Independent validation |
|---|---|---|---|
| Canonical paired normal entries, paperEq2 | Random_generator.py/generar_normal_canonica, shuffle | preprocessing.canonical_proposals | Support/distribution test; immutable original library fixtures |
| Parameter hypercube and exhaustive mapping, §2.3.3 | TERESA_map.py native four nested loops | solvers.parameter_grid/reconstruct | Complete810000 optimum and all three group counts |
| Radioactive layer averaging, Eq3 | TERESA_map.py/sorting and TERESA_plots.py/profile | physics_model.layer_activity | Independent numerical quadrature plus101 original profiles |
| Greedy paired sorting | TERESA_map.py/sorting | solvers.profile_batch |101 original χ and five-quantity profiles, fixed strict tolerances |
| χν and no time-marker branch | TERESA_map.py/sorting | solvers.profile_batch |101 original losses; measured residual check in standalone Judge |
| Four-parameter68.3% confidence mapping | TERESA_clouds.py | solvers.reconstruct | All481/7864/25064 cases retained and envelopes actively computed |
| Initial activity, SAR, bottom age and deposition flux | TERESA_plots.py/profile | solvers.profile_batch | All5×14 profile outputs, flux unit conversion×10 |
| Independent age validation, paperFig3 | TableS1 measured varves | independent Judge.score and notebook | Source-table/GT matching plus both independent original runs |

## Investigated conventions and limitations

The paper describes best sorting in broad terms; actual code is greedy top-down, not a global permutation optimum. This is stated explicitly rather than silently strengthened. The map clips SAR at0.2×global nominal mean; plotting omits that floor. All selected proposals are above the floor, so both are equivalent here; the distinction is retained. No time-marker penalty is used, and no varve data enter reconstruction. Original search upper endpoints are excluded. Full-precision grid search chooses the optimum, then original exchange formatting rounds A to2decimals, rate/relative deviations to4 and χ to3 before confidence selection/profiling. The local implementation preserves this meaningful quantization rather than comparing a differently rounded pipeline.
Age refers to the bottom of each layer, with surface age zero at2007 collection. Activity is a layer average displayed at midpoint mass depth. No arbitrary shift/scale/alignment is used. GT stores actual varve ages and uncertainties only. SAR and initial activity are model unknowns; no reconstructed values are mislabeled truth. Judge evaluates these outputs using positive deposition and both mass/age and full-decay consistency, then compares the modeled observations with all14 measurements. Coupled physical checks are necessary because the inverse is nonunique.
The primary relative SAR dispersion optimum is on the lower bound of this author-selected native scan. Mean A and mean SAR are interior. This bound and the demonstrated pairing sensitivity limit interpretation of individual deposition histories. The bound experiment is the complete native30^4 map rather than a claim to reproduce every iterative refinement/run in the paper. All task-relevant three-confidence comparisons and independent varve validation are preserved. A better concentration fit in the independent pairing has worse varve RMSE, shown explicitly. Normal distribution assumptions and lack of redistribution are model limitations, not validated geology.

## Fresh cleaned execution, tests and notebook

- PASS `cli` native exit0, 3.227560s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main`; private log `teresa_cli_final.log` SHA256`060775b2ccb9ac20599cf91a2ca85bbfba603e8803bf3b94f3ef5551d7d1c6be`.
- PASS `judge` native exit0, 0.133524s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; private log `teresa_judge_final.log` SHA256`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- PASS `tests` native exit0, 4.612618s: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v`; private log `teresa_tests_final.log` SHA256`ff737b99d9ec4f555c68344f65c605fc3b9eb31aef4382906b2c4d8a813470f3`.

Fresh Judge metrics: `{"age_ncc": 0.9988448437488565, "age_nrmse": 0.03724698397612326, "age_rate_relative_consistency": 4.500904153885774e-16, "decay_relative_consistency": 4.521039090724916e-16, "observation_chi": 0.5846476186104412}`.
PASS full active notebook: `python -m jupyter nbconvert --to notebook --execute notebooks/sediment_210pb_chronology.ipynb --output teresa_executed.ipynb --output-dir <private archive> --ExecutePreprocessor.timeout=180`, native exit0; executed-cell elapsed10.307103s. Explicit kernel argv is the shared research-runtime Python3.11.11, verified in the handoff manifest. The nbconvert frontend uses its available host dependencies; solver/kernel computations use shared NumPy2.4.6/SciPy1.17.1. All4 code cells executed, no error outputs. Both full810000-grid inverses and all33409 confidence-case profiles were actively computed, not loaded as fresh results.
Notebook infrastructure emitted standard Windows Proactor/ZeroMQ and missing-cell-ID warnings, with no numerical failure. Canonical notebook now retains executed outputs/normalized IDs. Private executed copy remains indexed; no temporary executed notebook remains in output. PNG exports were actually opened with view_image: activity/varve confidence comparison, independent-pairing actual experiment, full-grid deposition/history map. Visual QA corrected mixed-unit bars into separate panels and reduced crowded map ticks; the complete notebook was rerun afterward. All final legends, axes, units and data/reference/fresh labels were inspected. Final review PNGs are private, canonical notebook embeds the figures.
Tests cover18 cases: all101 original stage measurements, native full map and every confidence group, independent TableS1/raw/GT identity, quadrature physics, positive decay, normal proposal statistics/support, README constants/output contract, valid original scoring, poor ages, each additional quantity perturbation, missing file/assets, missing key, shape/dtype, NaN/Inf, zero-range hidden truth, cwd independence and report-write failure. Separate command logs preserve exact test counts and status.
Manual semantic review matched every implemented equation, factor10 flux conversion, λ sign/units, mass/age axis, grid upper-end convention, greedy tie ordering, exchange precision, confidence rule, acquisition constant and callable/default against both plans and source. AST symbol/call-role audit is archived as teresa_symbol_audit.json. Cleaned source/physics/Judge/tests/notebook use only permitted general NumPy/SciPy/Matplotlib/standard-library tools. There is no TERESA/oracle import, cross-task import, domain solver shortcut, generator, training checkpoint or target-aware calibration. CPU execution was verified; no GPU was used.

## Shipped numeric asset consumers

- data/raw_data.npz: observations consumed by preprocessing, independent Judge and source-table tests. README lists every key with units and batch-first shapes.
- test/fixtures/upstream_stages.npz: parameters/chi/profiles/z1/z2 consumed by101-case original parity test; independent_varve_table consumed by separate source-table/GT identity test. All fixture keys have documented consumers.
- notebooks/reference_outputs/upstream_profiles.npz: profile and profile_20261030 are the separately identified original runs, consumed by notebook comparisons and full-pipeline parity test. No cached output substitutes for active reconstruction.
- evaluation/ground_truth.npz: reconstruction is the actual independently measured varve age; age_uncertainty is its reported error, consumed by notebook/identity tests. No inferred SAR/A0 truth was fabricated. metrics.json records definitions, full-precision original measurements and all preselected boundaries.
- output/reconstruction.npz and judge_report.json: generated fresh standard CLI/Judge outputs. No duplicate task-root entry points or report/test trees exist.

## Mandatory gate ledger

| Gate | Status and evidence |
|---|---|
| Paper/revision/data hashes and mapping | PASS; source identity and full archive index below |
| Exact bound observations original/clean runs | PASS; two original native NR30 runs and fresh CLI logs |
| Independent parity and physics | PASS;101 stages, native pipeline, quadrature and mass/decay checks |
| Constants/units/conventions | PASS; README constants tests, manual equation/sign/unit review |
| Standard CLI and fresh Judge | PASS; exact native exit0 logs and report |
| Malformed/poor/evaluator Judge contracts | PASS; standalone tests including zero truth and write failure |
| Solver-visible sufficiency and no leak | PASS; only3 measured columns, full README, public URL scan |
| Minimal environment/no shortcuts | PASS; shared runtime, AST calls and independent native execution |
| Full required tests | PASS; exact final native log |
| Complete active notebook and visual review | PASS; both full reconstructions,3groups, true varves, actual pairing experiment |
| Metric provenance/threshold consistency | PASS; independent original reruns before local extraction |
| README/plans mathematical/interface consistency | PASS; manual semantic and signature review |
| Inventory/license/private archive cleanup | PASS; final manifest/index, MIT code notices, no supplied PDFs/archives in task |

## Private archive index

All paths below are relative to the worker archive outside the final task. Original run directories additionally retain canonical libraries, native complete Mapa4D, Absolute_min, all3 Clouds, all3 Plots and Solution text. These are original evidence, not shipped redundant arrays. The exact archive index is teresa_archive_index.json.

| Archive file | Bytes | SHA256 |
|---|---|---|
| Core_C1.txt | 194 | `0cf468d7889405204024353b29e41d7abe04df43595583e93228591ce792ce84` |
| TERESA_map.py | 9481 | `6df53dfb46c803b5d8ae982ec728bfbc0f703c2661693ff60d33d374913e0d60` |
| TERESA_clouds.py | 3656 | `21d9f704f1ac9f674d333f0943259084200e1fc888508c5c5279158179d830ef` |
| TERESA_plots.py | 7704 | `1442590bf0e82fd7e20902acec4997a52609765c2a4299ad8c3ee571c8fc90c6` |
| Random_generator.py | 1741 | `f23be0ca8917eba38884cd24444dc53a0950b4bcc116583fe0508bd5a2d4e760` |
| TERESA_LICENSE | 1068 | `34bb5f50e02d4257daadd3eb8063ad0f008cbf26b0eee10f64c82cefe5429e92` |
| TERESA_head.patch | 620 | `974756f5d43f22d9002624d10cc771d5bc6bd9b6d4b4cb4b6934e516c1d2004b` |
| TERESA_paper.pdf | 3688542 | `e4d0804dc7a473752969078ae6a37adfbf885d2cda87ecf2da664b57cec7dff4` |
| TERESA_guide.pdf | 1433451 | `8bf43b3e8a2c0200002f8a496774930dad6ffb624abea2e3bc07ec07e9e1da24` |
| TERESA_supplement.xlsx | 36066 | `1a1cc23cf5fd264195711e7fa18637d3a53cdc6e8e65f460bd3871951e7de66d` |
| oracle_teresa.py | 1449 | `581072e2bced5dc51c23eca554bd79d6f07c49d67e4a30462f3abf1b322f7d6b` |
| capture_teresa_evidence.py | 2078 | `2eeb874f0561052c625b14d16163866d3b0b5368d7038b2185479cf248b2a531` |
| teresa_original.npz | 1840 | `1a0167021ceaf84840613463c0591d876afe543558fecd9b0019bb01a025a66a` |
| teresa_original.json | 2438 | `1d0ea2d744972b2bd5acdced8dee1177b72b6dfbc2362734d739dda63526072a` |
| teresa_original.log | 5510 | `8d87b4a1705330cac924bea86e068afc9c26f92afddf5b9bf7251483a6a808af` |
| teresa_original_20261030.npz | 1840 | `89c1a6213a27e551935450b54d5098bb100927fb60eddd24b0ccfe66a53ba49d` |
| teresa_original_20261030.json | 2439 | `5fb2e8f8ac512d1122267a5a822206691e389e377a37c690376416387c8ea5bc` |
| teresa_original_20261030.log | 5512 | `20488100b0b95459028128d40c8b31ac312ce9225a6a5001f58a642043d21862` |
| teresa_preextract_fixtures.npz | 63006 | `ec5c3260f7c0c83b055f411684bbdc47f65ad6ba5196e305c3d7a7566dd22476` |
| teresa_reproduced_profiles.npz | 1652 | `fcfd44b454a15ec37faa1d1fa6ada823e682752c39f946ed130c76a5a00a9c25` |
| teresa_independent_metrics.json | 675 | `7394784b8c15f3bc6d1af2017d1e0f5f8c7485628688bf5dac567730d1063b9f` |
| teresa_clouds.log | 366 | `8d0320c55433766f488639d79fde09b276a27a40a16d7fead9d9143b2159b1bf` |
| teresa_plots.log | 46 | `56d4d7e6a8b8905f35f8a09160345960c7c9730bde387465072a0af242f33e28` |
| teresa_clouds_20261030.log | 370 | `200306dcc8d5bc34c24cf09105c9d305f2be6d03974881825166b43dad26a9c0` |
| teresa_plots_20261030.log | 46 | `56d4d7e6a8b8905f35f8a09160345960c7c9730bde387465072a0af242f33e28` |
| teresa_executed.ipynb | 502051 | `0966ec36cc743cb72d46089f19628f996644f162f2da1919c1fcce3afb6106cc` |
| teresa_nbconvert.log | 3296 | `bf03f162d980cbd313eae0500a906abff9c4d6c47f96c385e4e35ac53ad345e2` |
| teresa_symbol_audit.json | 5495 | `64a2d53102c2d499f3f5b3742d471bf7aa42604e2ce1e3f74f38630bfe0eca07` |


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 4 executed code cells, 3 computed PNG outputs, 3 formulas. Private evidence: research_batch12_20261029/acceptance/sediment_210pb_chronology/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Public benchmark contract review (2026-10-29)

The Judge already uses independent varve ages and complete physical consistency, not a finite stochastic scanning trajectory or hidden method result. No Judge, metric boundary, scientific array or reference implementation changed in this review. Necessary public parameters, including decay constant0.03118 yr^-1, layer averaging, mass coordinates, all observations/errors and output units, are self-contained. Method hints were generalized to conceptual regularization without source implementation recipes.

An independent hand-written control loaded only public raw arrays and implemented the README decay/mass equations. A constant concentration/rate submodel failed (retained diagnostic). A robust stationary-rate estimate from Theil-Sen log-concentration slope, with layer-specific positive initial activities, passed the unchanged Judge: age NCC0.9988518438, dynamic-range NRMSE0.03002566891, observation chi0 and both physical consistency errors below1e-15. These are maintainer controls, not a new blind Agent experiment, and no hidden arrays were read by either candidate script. Exact commands, native exits0/1, runtimes, scripts and hashes are privately indexed in benchmark_readiness_20261029/independent_controls/sediment_review.json. This corrects an initial audit inference that source scanning implied a source-identity Judge.


Final packaging: three notebook-generated teresa_*.png output exports were SHA256-verified into the maintainer benchmark_readiness_20261029/root_task_output_archives/sediment_210pb_chronology archive and removed from output. Canonical notebook embedded figures and reconstruction/report remain; no solver input or numeric result changed. Archive index.json identifies every original file and consumer.
