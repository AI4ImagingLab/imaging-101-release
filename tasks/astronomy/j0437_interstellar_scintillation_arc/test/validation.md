# Provenance and validation: interstellar scintillation geometry

All mandatory selected-task gates passed on Windows x86_64 CPU. Primary unknown is κ=De/Vparallel²; neither separate screen distance nor velocity is claimed. This differs from all existing279 physical problems, including Faraday-depth, brightness imaging, stellar atmosphere and orbital Doppler tasks. Parent formally bound the real eight-segment thin-screen inverse before extraction.

## Source, paper and real observations

- Author source: https://github.com/danielreardon/scintools , actual remotely verified master revision26f5d06feeea93de516c075ec273f34211fb5cd1. The first line of https://github.com/danielreardon/scintools/commit/master.patch was saved as scintools_head.patch; this is remote evidence, not a local tracking assumption. Selective source files and MIT license are archived, with original Copyright(c)2018DanielReardon retained in shipped LICENSE/NOTICE.
- Published algorithm: Reardon etal.,ApJ904(2020)104, DOI10.3847/1538-4357/abbd40. Exact reviewed paper version is arXiv2009.12757v2, revised2024-10-01, downloaded from https://arxiv.org/pdf/2009.12757v2 . Full PDF/text stay private. The task selects the published arc-measurement stage on the current author example; it does not claim to reproduce the paper full16year orbital-model fit or all its numerical settings.
- Exact bound observations: all eight files under source scintools/examples/data/J0437-4715/p111220_*.rf.pcm.dynspec. Source header says psrflux/StandardFlux/ReferenceFluxDensity, CASPSR acquisition path and MJD0. Full original ASCII remains private; the NPZ retains all512channels and all original121/122integrations before active leading-short-outlier/edge processing. Source channels were descending and are reversed together with their coordinates during numeric conversion. No intensity, frequency or time sample is downsampled. The source unused flux_err column is archived but not required by its selected method; no absolute mJy calibration is fabricated.
- Classification: real processed instrument observations, not a simulation or synthetic tutorial. The current author arc_modelling notebook loads exactly these files and performs native processing plus full temporal concatenation. That complete source notebook is archived. Dataset identification and observable clock/frequency calibration are public; private acquisition-system path strings from ASCII headers are not released.
- Public raw NPZ SHA256 `9a354ab141f7b8f1e16d46b1b048455d3398fc93862951b47679dd769be99d64`. All32keys are measured intensity/frequency/time/headerMJD arrays, numeric batch first; every key/shape/dtype/unit appears in README. No target κ, reconstructed screen, estimated gain, hidden source model, seed or GitHub URL is solver-visible.

## Original execution and pre-extraction fixtures

Original isolated source-path environment: private elecsus-oracle Python3.12.4 (Anaconda), NumPy1.26.4, SciPy1.13.1, scikit-image0.23.2, Astropy6.1.0, lmfit1.3.4, MKL2023.1; CPU with OMP/OPENBLAS/MKL threads1. Source files are unmodified. The original inpainting/domain dependency is used only in this private oracle run, never imported by the cleaned task. The shared cleaned Python3.11.11 uses NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1 and no original scintools installation.
Capture commands: private oracle Python archive/oracle_scintools.py (full first-file stage capture), then archive/oracle_scintools_full.py (all eight individual complete measurements plus actual concatenation). Each file is actually read through the original loader. It actively rejects leading short subintegrations, trims flagged edges, refills missing samples using the source default biharmonic method, estimates/removes rank-one gain, computes frequency and wavelength secondary spectra, and fits each arc. The stitched case uses the exact source gap-clock convention followed by mean refill and another active SVD gain estimate. No archived notebook result, successful import or different demo substitutes for this original run.
Independent complete rerun: archive/repeat_scintools.py invokes the original full path again, captures native exit0 and4.0716216s, and hashes every numeric array before/after. All array hashes matched exactly. See scintools_repeat_evidence.json and complete repeat stdout/stderr. First-file core time was0.1583804s; per-segment/stitched core timings are identified in original JSON rather than mixed with frontend/import wall time. There is no stochastic seed. Source warnings about rounded/nonuniform channels and differing inter-segment frequency properties are preserved, not claimed absent.
Pre-extraction freeze: frozen_scintools_case.json records all eight observations, nine outputs, physical κ units, exact source discrete protocol, stage tolerances and Judge criteria. The later wording correction distinguishes native121/122integrations from source-post-short-exposure120/121; it does not change the original input files or measurements. The original full run had already read all native samples. Stage fixtures were captured by the original implementation before local mathematics.

## Traceability of selected mathematics

| Published stage | Original file/function | Local function | Independent evidence |
|---|---|---|---|
| Clock/calibration and edge processing | dynspec.py/load_file,remove_short_subs,trim_edges | preprocessing.make_spectrum;physics_model.Spectrum.trim_edges | Exact native-after-clock and all first-file preprocessing fixtures |
| Smooth gap equations | dynspec.py/refill→private original inpaint_biharmonic | preprocessing.reflecting_laplacian/fill_gaps | Full source refill parity plus constant-nullspace/symmetry/plane recovery |
| Broad-gain nuisance estimate | scint_utils.py/svd_reconstruct,svd_model | physics_model.svd_reconstruct/svd_model and Spectrum.correct_dyn | Original corrected dynamic-spectrum fixture |
| Cubic wavelength resampling, paper§2.1 | dynspec.py/scale_dyn | physics_model.Spectrum.scale_dyn | Entire original resampled first-file array |
| Secondary Fourier power,paperEq1 | dynspec.py/calc_sspec;scint_utils.py/get_window | physics_model.Spectrum.calc_sspec/get_window | Full physical power arrays; independent cyclic Fourier-bin test |
| Geometric inverse,paperEq2–4 | paper geometric relation and source betaeta units | physics_model.arc_beta/physical_kappa | Independent Hz/mHz and2π test, all nine upstream κ |
| Normalized arc power,paper§2.2 | dynspec.py/norm_sspec | physics_model.Spectrum.norm_sspec | Complete source axes/normalized power fixtures |
| Asymmetric quadratic peak,paper§3 | dynspec.py/fit_arc;scint_models.py/fit_parabola | Spectrum.fit_arc;physics_model.fit_parabola | All9exact peaks; independent finite-difference covariance Jacobian |
| Longer observing span,source notebook | dynspec.py/__add__ and all-eight example | preprocessing.stitch;solvers.reconstruct | Complete eight+stitched source output, active notebook span comparison |

## Investigated discrepancies, physical scope and precision

The selected source current defaults differ from parts of the paper:10%Hann/no-prewhitening versus paper Hamming/first differences; auto dλ=(maxλ−minλ)/N versus paper lowest-channel spacing; unweighted profile average; source asymmetric fit region1/0.5dB below peak versus paper3/1.5dB. These original example settings are preserved, not silently replaced with a purported exact paper-figure reproduction. The full paper orbital/distance model is outside the deliberately bound κ unknown. The source notebook scientific comparisons—processing, frequency/wavelength spectra and short/long observing spans—are retained and actively computed.
Mean removal makes the Fourier DC physically zero. The first log-field comparison failed at exactly1of131072frequency-spectrum bins: original DC−256.7066776dB versus local−283.2439616dB, both linear powers≤2.2e−26, while all other log bins met the fixed tolerance. This is the undefined log of a numerical zero, not a measurable field discrepancy. Tests now compare every bin in physical linear power using unchanged rtol1e−8/atol1e−9 and independently require DC<1e−20; no bin is discarded and no signal tolerance or Judge boundary is relaxed. The wavelength-power field is checked the same way. All required axes, dynamic arrays and power profiles retain their original fixed numerical tolerances. Source global seterr is replaced by a scoped numerical context without changing values.
The source auxiliary etaerr2 formula is incorrect for a quadratic peak: it lacks coefficient powers and cross covariance. The primary peak−b/(2a) is correct and unchanged. The source empirical noise-width error (with its original√2 convention) is retained. The separate new auxiliary parabola diagnostic uses the full covariance Jacobian and omits the legacy extra√2 because that covariance already describes the fitted symmetric profile. It is renamed parabola_standard_error, not passed off as unchanged original output or actual physical truth. Original legacy values are preserved and plotted alongside the corrected diagnostic. A numerical finite-difference Jacobian independently validates this correction. No main κ or primary empirical-error parity criterion is dropped.
Rounded inter-segment frequency centres differ at most0.000854MHz: below0.002channel and below2e−6relative observing frequency. Original first-frequency and time-gap conventions are preserved and documented. The combined clock uses source linspace labels while its FFT retains the first dt, so this is the author example approximation, not a claim of exact irregular-time Fourier inference. The precision discrepancy is much smaller than the source empirical curvature uncertainty. All native calibrations remain visible for alternate solvers.
Geometric units are independently checked: β[m⁻¹]=κ[s²/m]fHz²/2, while ηβ usesmHz so κ=2×10⁶ηβ. Fourier frequencies are cyclic, not angular. For anisotropic scattering Vparallel is projected along the relevant direction. No unique distance or speed is inferred. A conditional standard-method result is not actual physical truth.

## Fresh implementation, tests, notebook and visual review

- PASS `cli` native exit0, 1.824508s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main`; log `scintools_cli_final.log` SHA256`40e7fa0c79ae785ab3ad103ec7b4a22180cd0ca5e634ad4ce8aa789e13bb8032`.
- PASS `judge` native exit0, 0.133012s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; log `scintools_judge_final.log` SHA256`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- PASS `tests` native exit0, 2.452267s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v`; log `scintools_tests_final.log` SHA256`e79562e54c7b73a37244eec41dfdf9b0d7b69989aaa1ed85ecfa3f14e4f9d800`.
- PASS `notebook` native exit0, 9.156143s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/j0437_interstellar_scintillation_arc.ipynb --output scintools_executed.ipynb --output-dir D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive --ExecutePreprocessor.timeout=180`; log `scintools_notebook_final.log` SHA256`9170d17051d6c402ce7cc8bdb7a635ff3332ab95b15e5d8732a90996d72d4e5c`.

Fresh independent Judge report metrics: `{"maximum_kappa_relative_error": 1.9090020282790393e-15, "maximum_power_deficit_db": 0.29515963503895115}`.
The primary all-nine κ relative discrepancy was1.91e−15; exact measurement is in the generated report. Judge keeps preselected maximum relative error0.01 and measured-power deficit0.5dB. Every original source empirical error exceeds1%of its κ; this tolerance is numerical standard-method agreement, not a claimed actual screen accuracy. All nine outputs are checked; corrupting any individual column is rejected. No NCC/NRMSE is imposed on this scalar-parameter problem.
Canonical active notebook has4executed code cells, no errors; executed-cell elapsed6.088492s. It reconstructs the entire eight+stitched path from raw, evaluates fresh results and writes separately identified fresh measurements. Original arrays are loaded only for labeled comparison. Explicit kernel argv points to the shared research-runtime Python, and the notebook prints sys.executable. Notebook frontend warnings are Windows Proactor/ZeroMQ infrastructure only. Canonical notebook contains executed outputs; the temporary executed notebook stays in the private archive, not output.
All3actual figure exports were opened with view_image: native/gap/SVD/frequency/λ processing; all9original/fresh values with empirical errors and the fresh stitched spectrum; all8plusstitched observed power profiles and original-invalid/corrected/empirical error comparison. Visual QA made fresh red points render above coincident original points. Axes, signs, physical units, line/marker labels and fit truncation were inspected. No missing result was replaced by zero or reference-as-fresh. Final figures are archived privately and embedded in the canonical notebook.
The16tests cover native clock and source stage parity, full eight+stitched pipeline, reflection boundary/nullspace/adjoint symmetry/plane recovery, exact cyclic Fourier axes/Hz-to-mHz/2π, corrected covariance via unit scaling and independent coefficient finite differences, mean refill, README every key/unit/shape/physical constants, independent known reference and all9bad columns, malformed key/shape/dtype/NaN/Inf/negative values, missing prediction/assets, zero hidden reference, cwd independence and report-write failure.
The source/core mathematical and call-role review verified all required explicit operators and inverse steps. There is no source/oracle import, cross-task dependency, domain image-inpainting shortcut, opaque reconstruction library or generator. All ordinary tests consume frozen original numeric fixtures, never original imports. Notebook and Judge metrics share independently tested physical conventions; Judge imports no solver. README/plans/signatures/defaults were checked against actual source, including the complete four-dimensional raw intensity/time/coordinate contract and output order.

## Shipped asset consumers and provenance

- data/raw_data.npz: all32original intensity/time/frequency/MJD keys, consumed by preprocessing and README/key tests. Zero flags remain in data and are actively handled.
- test/fixtures/upstream_stages.npz: native,refilled,corrected,secondary_frequency,wavelength_dynamic and final wavelength/axis/arc-profile fields, each consumed by the first comparing test. Source numeric capture is independent of local math.
- evaluation/baseline_reference.npz: original nine κ values plus curvature-grid/observed-power pairs, consumed by standalone Judge and full-pipeline parity. No ground_truth.npz is fabricated.
- notebooks/reference_outputs/upstream_measurements.npz: independently reproduced originalκ,empirical errors,legacy errors and profiles, all consumed by notebook original/fresh comparisons. Its repeated κ/profile content has the explicitly identified notebook-review role, while the evaluator reference is its scoring boundary asset.
- notebooks/reference_outputs/fresh_measurements.npz: fresh nine κ values, retained empirical errors and corrected auxiliary covariance measurements, generated by the active notebook and indexed for measurement review. It is neither ground truth nor a cache used as reconstruction input.
- output/reconstruction.npz and judge_report.json: fresh generated submission/report. Review PNGs and temporary executed notebook were removed from the task output after private archiving. Required module inventory exactly follows the task template. No root wrappers, duplicate tests/reports or standalone metadata JSON exist.

## Mandatory gate ledger

| Gate | Status/evidence |
|---|---|
| Exact paper/source/data identity and mathematical traceability | PASS; original sources/patch/data hashes and mapping above |
| Original and clean methods on exact complete bound data | PASS; actual native logs, independent all-array source repeat and fresh CLI |
| Independent numerical/physics gates | PASS; source fixtures plus geometry/Fourier/L²/covariance checks, explicit DC handling |
| Constants/units/conventions | PASS; preserved coordinates, input counts, source rounding and README tests |
| Full CLI and standalone Judge | PASS; native0 and all9outputs/observed-power criteria |
| Judge malformed/poor/evaluator contracts | PASS; direct independent tests and native CLI errors |
| Solver-visible sufficiency and no leakage | PASS; public URL scan, numeric raw allowlist and self-contained measurement/output description |
| Minimal shared environment and no domain shortcuts | PASS; full shared CPU execution and AST call-role review |
| Required unit/parity/physics/pipeline tests | PASS;16tests in exact final native log |
| Active full notebook and scientific comparisons | PASS;4cells, native reconstruction, all9conditions, all profiles and original-error discrepancy retained |
| Metric/reference provenance and preselected thresholds | PASS; original full repeat before extraction, no actual-truth relabeling |
| README/plans actual interfaces and mathematics | PASS; manual semantic/sign/unit/default/call-flow review |
| Inventory/private archive/license | PASS; original MIT notices preserved, no supplied PDF/ASCII/archive in task |

## Private archive index

Paths below are relative to the worker archive outside the released task. The complete machine-readable list is scintools_archive_index.json.

| Archive file | Bytes | SHA256 |
|---|---|---|
| scintools_head.patch | 1785 | `e7ae9b1d9b09e48b1fa7097b6f5873558485412282cda5e099483329a68a4024` |
| scintools_arc_modelling.ipynb | 6007670 | `6cc9caa637934bc7e496e59f534c8b94f318e83e0b43fbdf4787eec177c99479` |
| scintools_paper_v2.pdf | 4237208 | `0a646c6632c090d1c24cc4ffdf008d1946ac17b0f4b6bcd8dd78b7e4b5192e19` |
| scintools_paper_v2.txt | 69426 | `5732fabf081555f8e8e5d2ae18369ff8f4d0358a2274b1e347c0aa16d00fdbc9` |
| oracle_scintools.py | 1328 | `670ac1e06ffb4d087e48316f7495b385fd6406ab987c79a34ad544803dcac73e` |
| oracle_scintools_full.py | 1919 | `9b94346b170b1a8a7bc7b064fc551ec6d0204f7ba71bc4d41e8ec0c6c0a16d83` |
| repeat_scintools.py | 1152 | `ce9b3362915eb5c2a279d85e18e1a646a941b44662730d667cea640ac8ba0e4f` |
| scintools_original.npz | 10988488 | `8b417f18e4c600d2e511b0bf792cfa59822f8a429212c8009bf8e6872b8e84b9` |
| scintools_original.json | 726 | `8c8b63d470f4247015aca1179f54cb38fb3fb3e14571e18ad7bc4c7f3e51afbd` |
| scintools_original.log | 2120 | `30eb1af99143099f88eaf8825e4a89b2427a204ba45f3b48037b80b3ce0be96a` |
| scintools_full_original.npz | 102312342 | `2bf8154f4b7e8ba4f8de1b5d5c8176a6983ca80c816a013587b30fad9f49e024` |
| scintools_full_original.json | 1839 | `311cfba99cc47291c54f17d6e42d0c4ade7ea1641f00f2d3898dd37e1262efe6` |
| scintools_full_original_first.json | 1840 | `060a3de4e12b6c365d40e764041b85f0b50435272cd1792d0d312c2da49b320c` |
| scintools_full_original.log | 6466 | `925d6c001983bee420af2c61e1eecbcbb88a52ff1bb029521c009f45fbf06828` |
| scintools_full_original_repeat.log | 2875 | `274e0ff0d8a6521cbb5b3f981476f528f3aee555fdd21a2ef7745e4c308c4ee7` |
| scintools_repeat_evidence.json | 39849 | `e4abd8e4aa80cd11c768ba870d838f6259551b90a53e2c471c6d9e06f3a0d603` |
| scintools_native_metadata.json | 1331 | `005bd993c1649fdac11105cf6b69515372d38b059105a05e25e11492203c76bd` |
| scintools_executed.ipynb | 2375699 | `5502d563059fbf04bf5a31b6442eb89f9e77c9ad361df8ada71525f0548f90aa` |
| scintools_notebook_final.log | 1296 | `9170d17051d6c402ce7cc8bdb7a635ff3332ab95b15e5d8732a90996d72d4e5c` |
| scintools_symbol_audit.json | 8623 | `1886fb62ca9b7df0bde15427a9124d1851d4506f37efb2881a3b22e859e21fb5` |
| scintools/LICENSE | 1071 | `a13cc9bb663810e74946c72734bbe05bde289fc599148c9b01deaf844f73484d` |
| scintools/p111220_074112.rf.pcm.dynspec | 3873017 | `cdf3da7e56c92d6ab6d83425c9c85cb9a816e917c0e75392248208c985648fc7` |
| scintools/p111220_084944.rf.pcm.dynspec | 3841273 | `ac32826130a4489625053fbb7e451fbd8a0378f5f92b95825153ea6844161e70` |
| scintools/p111220_095816.rf.pcm.dynspec | 3873017 | `16f1d4869d6ba92336cd0ef494462431eebc144153bf22feb0b5c6ac898b2a65` |
| scintools/p111220_110656.rf.pcm.dynspec | 3873017 | `59f89131c02ef5c2cc7d6031c95d4d6c916f8b0a3022f6fe85afb6971528b9d6` |
| scintools/p111220_121536.rf.pcm.dynspec | 3841273 | `fe78beb7cfbf265f188b51963f1ba076a58961b00a1d482c577d944996021c10` |
| scintools/p111220_132416.rf.pcm.dynspec | 3841273 | `4d410a7109f50150ca03f89ab03912ef7a17769aba699968fb76b5a5a690117c` |
| scintools/p111220_143248.rf.pcm.dynspec | 3841273 | `6cf6a2f2650ffcd5d3c2a339bcb7aa3ae42333a0d9f90a08b2797e54e675cdae` |
| scintools/p111220_154120.rf.pcm.dynspec | 3841273 | `70743b927bdb1fe35c3587c544b71a96c4b22071fb08be4e15544c7b861c1d00` |
| scintools/scintools/dynspec.py | 194277 | `56bd290d7eb689b47d1d57cc7178454d204ac940d3693182f6b3d6275bf47837` |
| scintools/scintools/scint_models.py | 41164 | `d09ed6611ae27af6b63aee54e8b28518ebb4fd96e1b8dbc4f2299577c3050ad4` |
| scintools/scintools/scint_sim.py | 64283 | `741a2f59691cf7876fcff9063a62af890b160bf2481265bdcfe8bc245a909073` |
| scintools/scintools/scint_utils.py | 49771 | `660c8f6cc122b621d004e12342b9907eda411d6cc10225ae6d5629341de5e1c6` |
| scintools/scintools/ththmod.py | 80951 | `c85df0b6fe4f2344ef955eb05b4ace75f311b8ed331b482937d1b5fe3f9ed77d` |
| scintools/scintools/__init__.py | 228 | `aae30a6fbe87f6d31b9abdc9eb59a6a6276a064ac91be2c707984e366efc29c9` |


Final parent review passed: source/equations/units/actualcallerflow/GT roles independently checked; every actual figure viewed; dual math engines passed. Current tests 17, actual figure groups 3. Parent corrected only notebook math delimiters from fenced blocks to$$, then complete active NB/CLI/Judge/fullsuite/dualrender actual0. Parent added actual-truth precedence (profiles remain acquisition-derived evaluator assets) and hidden-reference float64 validation with new meaningful test; all17tests pass. Known auxiliary covariance error corrected explicitly with independent derivative check; primary peak/noise-error unchanged. DC log divergence handled by complete physical linear power plus independent DCzero, not cropping a discrepant bin.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 4 executed code cells, 3 computed PNG outputs, 2 formulas. Private evidence: research_batch12_20261029/acceptance/j0437_interstellar_scintillation_arc/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## 2026-10-29 public benchmark semantic version v2

The v2 criterion retains the original0.5dB observed-arc-power deficit criterion on all nine curvatures, with positive public finite search support kappa/(2e6) in[2,400]. Out-of-profile predictions receive a finite failing deficit; interpolation never turns unsupported curvatures into plausible scores. Exact agreement with a finite optimizer is not required in the present method-reference task. When actual independent truth exists, its explicitly configured1percent maximum relative curvature error also applies. All15unit tests passed, including the truth-priority path. An independently selected observed-profile maximum passes and out-of-search values fail. No arrays changed. Previous validation entries describe their original dated execution and v1 method-consistency semantics; v2 does not retroactively convert them into blind-agent results. The old Judge/metrics and failure evidence remain in benchmark_readiness_20261029/semantic_v2/j0437_interstellar_scintillation_arc/. No fresh blind-agent reconstruction was run in this review.


2026-10-29 final inventory: generated output extras were copied byte-for-byte with verified SHA256 into maintainer benchmark_readiness_20261029/group2_task_output_archives/j0437_interstellar_scintillation_arc/ and removed from the release output directory. Archive index is group2_task_output_archives/index.json. Canonical reconstruction.npz and judge_report.json remain; source/actual comparison provenance is preserved. These archives are not runtime inputs.
