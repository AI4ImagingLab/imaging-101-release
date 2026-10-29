# Validation: Bayer raw colour reconstruction

All final mandatory gates below passed on the exact public observations. There is no independent full-colour ground truth; reference agreement establishes faithful method extraction, not missing-channel accuracy. The real camera sample was not generated for this benchmark.

## Source, paper and data

- Published method: H. S. Malvar, L.-w. He and R. Cutler, High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images, ICASSP 2004, III-485–III-488, DOI 10.1109/ICASSP.2004.1326587. Four-page original paper archived as Malvar_He_Cutler_ICASSP2004.pdf, SHA256 `3ff72638c049336f2fd0471810498e38de7c3ab88b3299f099033eb6ea7de24a`. Primary source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Demosaicing_ICASSP04.pdf . Figure 2 was rendered and visually compared with every shipped 5×5 coefficient, including 1/16 and −3/16 entries.
- Modern standard reference implementation: https://github.com/colour-science/colour-demosaicing , exact commit `7bff324983fb77b41444fda3bf922e354d386d1c`. The fetched remote HEAD was verified to match this checkout in root/source_inspection.json. This is a modern implementation, not code authored by the three paper authors. BSD-3-Clause copyright retained in task LICENSE; source/adaptation attribution in NOTICE.
- Real data: https://raw.pixls.us/getfile.php/1272/nice/ . Canon PowerShot A540 CHDK 10-bit DNG, 7,749,360 bytes, SHA256 `15acc0208610b7b78d2a8ae4da09db94d71ec2bbb7e36dc5a440c9295cda2975`. The repository row explicitly links CC0-1.0; archived data_provenance.json and root/raw_pixls_repository.json record source row, full URL, published matching digest and licensing. Source EXIF text is archived. Data attribution preserved in NOTICE.
- Public NPZ SHA256 `679a9799560eecb6e4dabd167a78547c92aba8c7c5a0c279e477c39b0b27fd03`. Native stored raw is 2136×2888; the camera-defined active rectangle starts at row 8, column 44 and has 2128×2840 pixels, RGGB. This is the full active sensor, not a selected content crop, resize or synthetic Bayer resampling. No rotation. Original rawpy decoder supplied uint16 ADUs, four independent acquisition black entries all31 ADU, sensor white1023 ADU and as-shot display gains. White balance is not applied to the submitted result. Fourth gain0 is an unused camera metadata entry; RGB display uses the first three positive gains.

## Independent original run, before extraction

Private command: root/oracle-env/Scripts/python.exe root/colour_demosaicing/capture_original.py, with the upstream checkout explicitly on sys.path; stdout and stderr archived in original_run.log and structured evidence in original_run.json. The unmodified original Malvar and bilinear functions ran on this exact normalized active mosaic. Both methods were independently repeated in the same capture; maximum absolute repeat difference0. Exit0, measured total 28.504455400s including raw decoding, both algorithms, reruns and compressed output capture. No training, randomness or GPU. Oracle Python3.11.11, NumPy2.4.6, SciPy1.17.1, rawpy0.27.1, private colour-demosaicing checkout and its Colour dependency. Cleaned environment versions/backend are recorded separately in cleaned_environment.json. A Colour optional-Matplotlib warning in the private import role did not affect the selected numerical functions; original outputs were finite and reproducible.

The source originals and capture scripts, original DNG, complete upstream checkout, numerical outputs, source paper and original logs live only under `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/root/colour_demosaicing` and the parent root archive. Private archive_index.json gives byte sizes and SHA256 hashes. Shared runtime dependencies were not upgraded. The raw decoder and domain reference packages are confined to the isolated oracle; they are absent from cleaned imports, requirements, tests, notebook and Judge.

Before local extraction, preimplementation_criteria.json fixed stage tolerances rtol1e-13/atol1e-14, NCC≥0.9999999999, dynamic-range NRMSE≤1e-10 and measured CFA sample maximum error≤1e-12. The independently repeated original is bit identical, but the nonzero tolerances allow harmless float64 reduction variation. These are numerical method tolerances, not relaxed physical accuracy thresholds. The Judge reference is a separately run original method result, never relabeled as truth.

## Equation, function and test traceability

| Paper stage / numerical convention | Original implementation | Local implementation | Independent validation |
| --- | --- | --- | --- |
| Bayer selection, disjoint R/G/B samples | bayer/masks.py:masks_CFA_Bayer | physics_model.cfa_masks, sample_rgb, sampling_adjoint | four Bayer phases, non-square adjoint dot identity and partition |
| DNG ADU normalization to black-to-white range | capture calls rawpy then (raw−black)/(white−black) before original API | preprocessing.load_observations, normalize_mosaic | known ADU values31,279,527,1023; captured full normalized input |
| Equations1–3 green interpolation plus same-colour curvature; alpha1/2 | bayer/demosaicing/malvar2004.py:demosaicing_CFA_Bayer_Malvar2004 | solvers.interpolation_kernels, reconstruct | original frozen 9×11 input, all4CFA phases, full native parity; independent affine reproduction |
| Equation4 green-site chroma interpolation; beta5/8, transposed directional pair | same original Malvar function | reconstruct with horizontal/vertical kernels and mask orientation | all4 original phase fixtures and exact retained observations |
| Equation5 opposite chroma; gamma3/4 | same original Malvar function | reconstruct opposite-colour kernel | Figure2 coefficient review, kernel sum1, original fixtures and full parity |
| Bilinear scientific comparator | bayer/demosaicing/bilinear.py:demosaicing_CFA_Bayer_bilinear | solvers.bilinear_comparison | separately frozen full original comparison; fresh full native notebook result |
| Evaluation and observation consistency | no original accuracy claim on this camera sample | independent evaluation/judge.py; visualization.compute_metrics separately | known hand-computable metric examples, undefined reference rejection, strict Judge and full pipeline tests |

## Paper/code discrepancies and limits

The paper describes averaging available neighbours with adjusted weights at boundaries. The selected modern original code instead uses SciPy ndimage reflect convolution. The local code preserves that actual original convention exactly. Interior kernels match paper Figure2; boundary handling is an explicitly identified implementation convention, not an unchanged reproduction of the paper's boundary prescription. Bilinear reflect can alter retained samples on the outermost border; the Malvar submission retains measured samples exactly. Neither discrepancy is hidden by cropping or scoring alignment.

An original function docstring accidentally lists One Microsoft Way (the address) as a fourth author. The public reference correctly gives the three actual authors. This bibliographic correction does not alter numerics. Real hot pixels are retained; signed gradient filters can spread sensor defects. Negative output values are valid, unclipped algorithm output. No missing-channel PSNR, physical RGB ground truth or superiority claim is invented. White balance, exposure, clipping and gamma are display-only and never change the Judge arrays. The notebook's synthetic affine/impulse diagnostics are independently known mathematical test inputs, not extra measured scenes or new benchmark acquisitions.

## Fresh execution and metrics

CLI `['D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe', '-m', 'src.main']`: exit0, 10.769176s; writes the single required batch-first float64 array `(1,2128,2840,3)` to output/reconstruction.npz. Standalone Judge: exit0, 1.254041s. Full unit/comparison suite: **22 passed**, exit0, 3.980182s. Latest complete active nbconvert execution: exit0, 31.095595s, timeout1800s justified by full native reconstruction and figure exports; all code cells executed without errors. Exact commands, environments, cwd, runtime and logs are in execution_records.json. Source method kernels themselves take 0.272355s for Malvar and 0.142739s for bilinear within the notebook; this excludes compressed I/O and export, so it is not conflated with CLI wall time.

Fresh Malvar vs independent original: NCC `0.9999999999999999`, DR-NRMSE `0.0`. Fresh bilinear vs its separate original: NCC `1.0`, DR-NRMSE `0.0`. Malvar measured sample error0. NCC is uncentered flattened cosine; NRMSE is RMS error divided by original array dynamic range, with no crop, rescale, phase, mean subtraction or gamma conversion. Negative submitted-value fraction `0.00022249726428747927` is measured, not removed. Reference precision, shape and units match. No truth file exists for this real scene.

Initial notebook execution failed because its kernel started inside notebooks/; the source now locates the actual task ancestor and imports cleaned src. The original failing execution remains archived. A later parent runner invocation mistakenly supplied a relative private directory and failed pytest temporary-directory setup (11 fixture setup errors, 11 test passes); no solver defect or failed test was hidden. That attempt was moved to relative_runner_attempt/ outside the release; rerunning with the absolute archive path gave22 passes. Extreme malformed-input stress deliberately causes intermediate overflow warnings; Judge returns an error with empty metrics and finite JSON, never NaN/Infinity. Final normal reconstruction/Judge/active notebook are successful.

## References, fixtures and figure consumers

- evaluation/baseline_reference.npz: independent original full Malvar method result; Judge and full-pipeline test consumer. Not actual RGB truth.
- notebooks/reference_outputs/upstream_bilinear.npz: independent original full bilinear method; notebook and full-pipeline comparator consumer. Not a duplicate of selected reference.
- notebooks/reference_outputs/upstream_measurements.json and fresh_measurements.json: labeled capture/fresh metric provenance, read during maintainer review. They are not solver-visible data.
- test/fixtures/upstream_cases.npz: original pre-extraction 9×11 input and all four original CFA outputs, consumed by the parameterized original parity tests. No fixture was produced by the cleaned solver.
- Four actively computed comparison groups are embedded in the executed canonical notebook and exported privately as PNG/SVG/PDF plus figure_source.npz and experiment_metrics.json. Figure1 shows the full observed scene, actual per-channel sensor distributions and fresh full-frame methods; Figure2 shows matched native crops and signed method differences; Figure3 shows measured-site residuals, defined sensor-row comparisons and all-native-pixel distributions; Figure4 shows signed kernels and independent unit-impulse response. Final figures1–4 were visually inspected for readability, truthful labels, common display scales and colour-map centering. Raw overview uses a4×4 block mean containing all CFA phases; reconstructed RGB overview uses display-only stride4. All metrics/histograms/quantiles use full native arrays. Initial stride4 raw preview selected only red sites and was corrected before acceptance; no reconstruction, threshold or raw observation changed.

## Plan/source semantic review

Reviewed every shipped callable and actual caller path against both plans. Forward sampling and its adjoint have matching masks, axis order and normalized units. The four gain-corrected filters and their transpose are explicit local arithmetic; no iterative optimizer, stopping rule, training or regularizer is falsely described. Direct convolution boundary mode matches the selected original. CLI reads only visible observations; notebook references are comparisons; Judge imports no src. Importing src.main preserves the existing Matplotlib backend (explicit unit test). Plan signatures, shapes, defaults, no-clipping convention and side effects match actual source. Final visualization source is archived separately because the historical initial document-builder script predates visual QA; source modules and executed notebook are authoritative.

## Completion gates

| Required gate | Status | Evidence |
| --- | --- | --- |
| Paper, verified exact source, data hashes, equation/function/case mapping | passed | above, source_inspection.json, data_provenance.json, archive_index.json |
| Actual original and local on exact bundled observations | passed | capture_original.py, original_run.json, CLI and active notebook logs |
| Independent original parity and applicable mathematical checks | passed | original frozen fixtures; four-phase/full pipeline; adjoint, affine, normalization |
| Constants, units, precision, coordinates, boundary conventions | passed | original decode metadata, README matching test, above boundary distinction |
| Fresh standard CLI and strict standalone Judge | passed | CLI/Judge execution records and finite report |
| Judge poor/invalid/missing/evaluator/report-write cases | passed | parameterized contract tests; exit0/1/2 and truth-precedence cases |
| Public allowlist suffices, no hidden-target calibration or GitHub leakage | passed | visible raw/black/acquisition gains; explicit output; source audit |
| Minimal validated shared dependencies; local mathematics and no cross-task/domain imports | passed | np/scipy/plt subset, source call-role audit, CPU tests |
| Full test suite | passed |22tests, latest absolute-path pytest log |
| Active notebook, preserved selected comparisons and fresh figures | passed | complete nbconvert log; two fresh full-frame methods, four inspected groups |
| Distinct historical/original/fresh results and justified metric thresholds | passed | labeled reference type, independently repeated original, nonzero numerical limits |
| README and plans match mathematics and signatures | passed | semantic review above; no proposed runtime described |
| Final inventory, notices, private provenance and cleanup | passed | final parent inventory audit; no PDFs/raw formats/temp notebooks/capture scripts in task |

Parent audit and final task hashes are recorded separately at the batch archive root. This task preserves modern-original numerical conventions with the explicitly identified paper-boundary difference; it makes no experimental accuracy claim beyond available real measurements.


## Parent acceptance review

Parent directly executed unchanged original twice, extracted all kernels, compared paper Figure2, ran22 tests, active full native reconstruction/notebook, strict independent foreign-cwd Judge0 and source/inventory audit errors[]. All4 actual final figure groups inspected. Real CC0 native sensor data; modern-original reflect boundary convention explicitly distinguished from paper edge prescription; no RGB truth claim.
