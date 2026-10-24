# Provenance and validation: Solar CRISP Stokes vector mapping

This task's scientific arrays are new to batch5. No preceding139 task was modified. The public observations are real SST/CRISP spectra, not generated data. The hidden baseline is a validated method reconstruction, never labeled ground truth.

## Fixed scope, source and private archive

Physical unknowns: nine depth-independent atmosphere maps across the entire contiguous native64x96 ROI y20:84,x240:336 from the author's120x350 cube. All30 wavelengths and four Stokes components are retained. Source observations, cavity and spectral transmission are acquired/calibrated inputs; no target-derived calibration or reference atmosphere enters the public NPZ.

Selected physical-method paper: Orozco Suarez & del Toro Iniesta2007, A&A462,1137-1145, DOI10.1051/0004-6361:20066201. The full10-page author manuscript arXiv1211.1502 was downloaded and read, including Eqs1-5 and the complete response appendix. The version has a2021 manuscript header despite its2012 arXiv identifier; no fresh publication date is inferred from that header. Additional source-family context: de la Cruz Rodriguez2019, A&A631,A153 DOI10.1051/0004-6361/201936635, full13-page arXiv1909.02604 read. The shipped ordinary per-pixel ME inversion does not claim reproduction of the2019 multi-resolution global method.

Original code: https://github.com/jaimedelacruz/pyMilne, exact remotely verified commit1db8549839aafb8d5e047861dfad3f595de1c042. GitHub API response and codeload archive are private; no remote freshness is inferred from an old local branch. Source MIT copyright J. de la Cruz Rodriguez (ISP-SU2020), retained in task LICENSE and local module notices. Raw author demo files are bundled in that repository under its MIT distribution. Private upstream checkouts, papers, archives, capture scripts and logs are outside this task, in maintainer research_batch5_20261021/astro_earth.

## Original run before local implementation

`capture_pymilne_oracle.py` ran the unmodified physical/inverse source on the exact bound ROI; only Windows build configuration was adapted. Command: oracle_env/Scripts/python.exe capture_pymilne_oracle.py. Source module MilneEddington, example_CRISP/invert_crisp.py grid, crisp.dual_fpi and imtools.fftconvol2d. One thread; deterministic nRandom1, nIter25, chi2_threshold1, mu0.93, two full inversion cycles, Gaussian45/FWHM15 initialization, cavity+0.45 velocity calibration. Initial atmosphere(1500,2.2,1,-0.5,0.035,50,0.1,0.24,0.7). The author's script can use4 randomized restarts; this explicitly selected configurable one-restart path avoids claiming that experiment was reproduced.

Original full run completed in79.1265003s; cycle1 meanchi2=138.9098265623882 and cycle2=133.272625612178. Full arrays, log and configuration: pymilne_oracle.npz, pymilne_oracle.log, pymilne_oracle.json. No runtime error occurred. Native C++ progress uses stderr; an early PowerShell redirection incorrectly surfaced NativeCommandError from that progress, so subsequent actual exit statuses were captured with Python subprocess. Successful completion is established by full arrays and terminal DONE, rather than the PowerShell progress diagnostic.

Oracle environment: Windows AMD64, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Astropy8.0.2, Cython3.3.0; private oracle_env. MSVC2019 toolset14.29.30133 and SDK10.0.19041.0, Eigen3.4.0 headers and official FFTW3.3.5 DLL64 are isolated privately. Build command setup_windows_oracle.py build_ext --inplace; flags /O2 /std:c++latest /openmp /D_USE_MATH_DEFINES /D__restrict__=__restrict /DNOMINMAX, FFTW import library, /MT consistent with source toolchain. OMP/BLAS/MKL threads1. Numerical computation is CPU; no GPU/checkpoint is needed. FITS readers/Eigen/FFTW/Cython are not cleaned runtime dependencies.

## Investigated source discrepancies and minimal corrections

1. FFT complex-copy byte count: src/fft_tools.hpp copy constructor copies ft and otf with nft*sizeof(double), but those arrays contain complex<double>. The latter half of each transfer function is therefore omitted after the source Region/Milne copy. Independent direct source FFT construction agrees with the local edge-extended correlation to6.7e-16; copied Milne initial spectra instead differed1.65e-4. Unconvolved spectra and responses matched to1e-13, isolating the issue from radiative transfer. Only both memcpy byte sizes were changed to nft*sizeof(std::complex<double>) in private pymilne_corrected_oracle. This FFT-only full run completed74.7134s with cycle2 chi2=133.19134214995586. Its complete separately labeled arrays are pymilne_corrected_oracle.npz, not the final baseline.

2. LM rejected residual: src/lm.hpp getCorrection overwrites r with residual at a trial model. On rejected trial the original continue retains bestModel and J(bestModel) but leaves r(trial), so the following equation combines different linearization points. A private diagnostic implementation retaining that original behavior reproduces six initially divergent source pixels with maximum scaled discrepancy2.98e-12 and chi2 discrepancies below4e-13 (pymilne_rejected_residual_diagnostic.npz). Thus the effect is independently isolated rather than assumed. Minimal fix: before rejection continue, call the original fx at unchanged bestModel to restore r. No optimizer, accepted update, objective, stopping rule or physical model is replaced. The full FFT+residual corrected oracle in pymilne_lm_corrected_oracle completed exit0 in62.875148s; mean cycle1 chi2=138.8934797003683 and cycle2=132.78200750761332. Both unchanged and each corrected full run remain privately archived and distinctly labeled in the notebook.

3. QR API semantics: Eigen ColPivHouseholderQR::solve invokes nonzeroPivots(), not rank(). setThreshold(1e-9) therefore does not truncate that solve. A first local extraction used GELSY1e-9, then a similarly truncated QR, exposing remaining low-rank thermodynamic discrepancies. Reading Eigen/src/QR/ColPivHouseholderQR.h lines511-527 and589-606 establishes the actual machine-precision nonzero rule. The final local pivoted QR reproduces it. After the residual fix, all84 original remaining diagnostic pixels agree within7e-11 once this call-role mismatch is removed. This is an extraction correction, not a third upstream bug.

The source wrapper describes dw_lines as km/s but the C++ actual window comparison uses angstrom; its default20 includes the entire two regions. The cleaned method evaluates both lines throughout both regions and documents actual units. The original azimuth wrap constants are float32 approximations3.1415926 and3.1416026; they are preserved rather than silently normalized to pi.

## Equation and algorithm traceability

| Required stage | Paper / mathematical role | Exact upstream file/function | Local executable | Independent validation |
|---|---|---|---|---|
| Polarized transfer matrix |2007Eq1 | src/Milne_internal.hpp compute_K | physics_model.synthesize_response | Full-source spectrum fixture; independent4x4 Judge solve |
| Linear source and emergent Stokes |2007Eqs2-5 | compute_Stokes / ME_Synth | synthesize_response explicit rational inverse | Source parity; unmagnetized, zero-opacity and azimuth symmetries |
| Zeeman strengths and splitting |2007appendix absorption and dispersion | src/line.hpp transition enumeration | zeeman_components | Group sum rules and exact original line-profile diagnostic |
| Voigt values and analytic responses |2007response appendix | source profile primitive / compute_K_der / compute_Stokes_RF | humlicek, profile, Dual | Original spectra+all9responses; independent finite differences |
| Spectral acquisition | Author CRISP example | invert_crisp.findgrid; fft_tools spectral degradation | regular_grid, degrade | Exact fine-grid/indices, original primitive numeric diagnostic, continuum conservation |
| Weighted objective and bounded LM |2019LM family, ordinary source execution | lm.hpp fx/fx_dx/getCorrection/fitData; wrapper_tools.fitOne | lm_block, pivoted_qr_step, clip_parameters | Rejection pixels, QR rank example, full-source final maps |
| Two-cycle initialization | Author ordinary CRISP example | imtools.fftconvol2d / gauss2d | smooth_model / reconstruct | Entire original cycle1 image through source smoothing fixture |
| Velocity calibration | Native instrument/example | invert_crisp final cavity+0.45 | reconstruct calibration | Public data/output contract; independent Judge subtraction |
| All output metrics | Public output definitions | Independent source arrays, no imported metric helper | standalone judge metrics and matrix forward | Known numeric cases, source RMS, bad outputs and actual lowered limit |

## Thresholds, result roles and numeric provenance

Parity contract frozen privately before the complete cleaned run: spectra rtol1e-9/atol1e-11; all9responses rtol1e-8/atol1e-9; smoothing atol1e-10; final physical parameter divided by source scale atol1e-5. No failure was bypassed by widening those limits. Local primitives and the actual solve semantics were corrected instead.

Final fresh local full reconstruction432.2859392s, cycle1 max scaled error7.4083184e-10; final maximum scaled error by parameter[7.298694981727749e-13,7.989164885202626e-13,7.834657367311593e-10,1.5287771049088406e-12,6.1624316760600095e-12,2.608686600069632e-11,2.9607427620703675e-11,4.3134940064248894e-13,3.822497873784414e-13]. Mean chi2=132.78200750761332. Private pymilne_cleaned_final_run.json and diagnostics retain full precision. This fresh run uses only the shared Python runtime and local core mathematics.

NCC>=0.99999 and dynamic-range NRMSE<=1e-5 apply separately to the eight non-azimuth parameter maps. Azimuth uses periodic RMS/pi<=1e-5 and mean cos(2delta)>=0.99999, where delta=0.5*atan2(sin(2 difference),cos(2 difference)); this cosine agreement is not NCC. The independently captured corrected reconstruction supplies the reference, never truth. Four observation RMS boundaries were frozen as1.1 times the corrected source Stokes RMS[5.4858908,11.93166126,13.39814124,13.38500291], divided by[.003,.0003,.0003,.003/3.5]. Full precision is in metrics.json. Independent SciPy Voigt/K-matrix/direct-correlation checks give[5.48590055,11.93166157,13.39814239,13.38499842], consistent with the measured rational-Voigt approximation. Source expected values are independently captured spectra, not metrics generated by cleaned forward code. Reference agreement is numerical reproducibility, not uniqueness or proof of photospheric physical accuracy.

The weighted RMS>1 is openly reported. It follows the deliberate10x QU and3.5x V emphasis and the limited depth-independent atmosphere. No claim of photon-noise fit is made. Actual unweighted spectral residuals, all Stokes comparisons, stage maps and thermodynamic distributions are shown. The source and fresh results agree and use all nine atmosphere quantities rather than scoring B alone.

## Assets and consumers

- Public raw_data.npz: real Stokes, wavelengths, independent instrument cavity/transmission. Source FITS arrays exactly sliced and cast float64; no synthetic generator.
- test/fixtures/upstream_stages.npz: independently captured corrected fine grids, indices, initial spectra/responses, full first-cycle model/chi2, smoothed model, second-cycle model/chi2, spatial PSF. Consumed by comparing tests.
- notebooks/reference_outputs/upstream_run.npz: unmodified full source measurements; upstream_fft_only_run.npz: FFT-only corrected source; upstream_corrected_run.npz: both minimal fixes. Consumed only by notebook scientific comparisons.
- evaluation/baseline_reference.npz: independently captured doubly corrected calibrated parameters. Used only by Judge/metric tests. No ground_truth.npz exists because actual truth is unavailable.
- Private figures/solar_crisp_stokes_vector_mapping: six PNG/SVG/PDF groups, numeric_sources.npz, metrics.json, manifest.json. Exported from actual fresh arrays and separately identified source runs. Six visually inspected groups cover all nine maps, all Stokes fit examples, source corrections, objective/iteration distributions, parameter distributions, and acquisition/forward/trade-off checks. Each export's SHA256 is in manifest.json.

## Validation commands, audit and gates

Shared runtime: Python 3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1.
Only these numerical/plotting dependencies appear in requirements. Standard library, FFT/QR/special functions and explicit task algebra were audited by symbol and call role. No imaging library, upstream checkout/import, optimizer wrapper, cross-task import or hidden baseline is used by solver runtime. Independent Judge uses separate matrix algebra and direct correlation; it imports no src.

README equations/signs/units and source functions were semantically reviewed. In particular the analytic inverse uses -h*(k cross r), the source-gradient term includes mu0.93, gf ratio is10^(-.442), width is angstrom, calibrated velocity is corrected only after inference, and fine-grid count/edge rules match original code. Source/plan callable signatures, defaults, shapes and caller flow agree. Notebook calls the same full reconstruction routine and evaluates its fresh submission. Library imports do not set Matplotlib backend.

Initial tests:27passed in21.08s, command shared-python -m pytest test/ -q -p no:cacheprovider --basetemp private/pymilne_tests_initial. Dedicated private pytest temp directory avoids inaccessible system pytest path; no package/environment upgrade was needed. Standalone Judge command --prediction output/reconstruction.npz --report output/judge_report.json exited0 on fresh output. Malformed missing file/key/shape/dtype/complex/NaN/Inf, invalid boundary, poor physical output, known numeric metrics, source forward RMS and genuinely lowered threshold are tested.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/revision/data/equation mapping | passed | Full papers/source and hash index below |
| Original exact-bound run | passed | Unchanged full original and separately corrected captures |
| Local exact-bound run and numerical/physical parity | passed | Fresh432.29s full run; all stage and physics tests |
| Constants/units/coordinates and precision | passed | README, exact preprocessing and physical checks |
| CLI python -m src.main | passed | Direct complete replay exit0 in331.2478s; final logs and record |
| Standalone independent Judge | passed | Fresh report;22finite metrics |
| Contract/error/poor-output tests | passed |27tests; standalone malformed-input subprocesses |
| Public allowlist and leakage audit | passed | All observations/calibration documented; no hidden data or public URL |
| Shared CPU runtime and explicit local mathematics | passed | NumPy/SciPy/Matplotlib; isolated C++ oracle |
| Executed active canonical notebook | passed | Full nbconvert exit0 in487.1542s; execution counts1,2,3; six inline PNG groups |
| Six scientific figure groups and numeric sources | passed | All six PNGs visually inspected; readable axes/legends and actual source arrays |
| README/plans/source consistency | passed | Semantic review above |
| Final template inventory/licenses | passed |25 canonical files; own generated caches moved to private archive; MIT notice retained |

## SHA256 archive index

- `pymilne2019.pdf`: `b244520319faeda8f7a672488c0dbbee12a7850c784df436e1eb9c835f452208`
- `analytic_response2007.pdf`: `25b48486f8e495a170dd032c49291bdb47565d50d85f15a601e22325546ca1fb`
- `raw_data.npz`: `17e0d1e77ec4cb05983661f6606ae3208c05eb7e6ea85a8f173a957e0b1465be`
- `upstream_stages.npz`: `dc01c6f334e5ef24fe795a31aac085fe85633f1260bc695967a210dee42d12ac`
- `baseline_reference.npz`: `70c7db73ea71f53bd609cead548873439626afc6ffd2fec716d7df880098799a`
- `crisp.6301_6302.2019-05-10_data.fits`: `93250ac47a9c949dfa2a2eac16c4fe89cb12bb5b1c8c07ae328e9244ae9ac6db`
- `crisp.6301_6302.2019-05-10_wave.fits`: `452693d29d3ba6380007557202bce0fb25b994628a268351668f509dbcaeab99`
- `crisp.6301_6302.2019-05-10_cmap.fits`: `44d54566ba2ebb8b48fc83e6009c8b77f7d3ff03b7cfcbc47a7f6528adb7c39e`
- `pymilne_oracle.npz`: `ba06de5896cda825308b2794e0848a794e46fe84f7bfca1b54c6c5433f583dec`
- `pymilne_corrected_oracle.npz`: `ff6b71fe6e53e567985975a35862c0d1d7d050ff7797888cb80b80658dce7ad9`
- `pymilne_lm_corrected_oracle.npz`: `10d0c064be9a070d742224c337aacb02d59a3ab7bf30bc5a1c4f7ef756199a44`
- `pymilne_cleaned_final_diagnostics.npz`: `4ae02920daa710a70f1350492d10bebd379bdb629c567bb8d8dabc3c5807e2df`
- `pymilne_rejected_residual_diagnostic.npz`: `e2ef88818492c9b9caabdf0d0ac5be07be67c29a7016fe3dfddb5ba3b024c9d4`


Final command replay completed: python -m src.main exit0/331.2477603s; standalone Judge exit0/2.7013592s; all30tests passed (pytest21.12s; outer process21.4475s), including same-point rejected-residual mathematical example, finite metric failures, report-write error and valid prediction from foreign working directory. Full canonical active notebook nbconvert exit0/487.1542249s and retains execution counts1,2,3 with six actual PNG outputs. Timeout1200s is justified by the measured complete CPU fitting runtime. Source remains unchanged after successful full execution; later evaluator changes only add strict hidden-reference schema validation. All scientific plots were visually inspected for legibility, units, actual method labels and parity scale. No notebook preview/cache substitutes for reconstruction. Exact commands, outputs and runtimes: private pymilne_final_validation.json, pymilne_final_cli.log, pymilne_final_judge.log, pymilne_final_tests.log, pymilne_notebook_record.json and pymilne_notebook.log.

Independent post-CLI and notebook outputs both pass all22 declared metrics; preselected parity limits were retained. The final rendered figures and numeric sources are indexed in the private figure manifest. No temporary executed notebook, source archive, PDF, duplicated report, bytecode or test cache remains in the task. All mandatory gates are now passed; parent independent acceptance is pending.


Parent review: azimuth evaluation now uses its intrinsic pi periodicity, delta=0.5 atan2(sin(2 difference),cos(2 difference)); periodic RMS/pi<=1e-5 and mean cos(2delta)>=0.99999. The other8maps retain unchanged NCC/NRMSE boundaries. This replaces an unphysical raw-angle discontinuity while retaining the preset error scale. An independent near0/pi analytic case verifies both metrics. The notebook dynamically prints standalone Judge metrics; full reexecution is being refreshed to retain actual new console output.


The independent Judge resolves ground_truth.npz first when that reserved actual-truth asset exists, otherwise baseline_reference.npz; this task supplies only the latter. A priority test uses distinct references to verify actual-truth precedence and fallback. No physical truth was invented.

Final parent-requested periodic/reference-priority verification:32tests passed, the genuinely lowered observation threshold remains consumed. Complete active canonical notebook reexecuted exit0/480.7671s and contains six inline PNG groups plus dynamically printed periodic metrics. Standalone new22metric Judge passed before and within the notebook. Exact private records: pymilne_periodic_validation.json, pymilne_parent_final_tests.json and corresponding UTF8 logs. No solver, source reference, raw observation or threshold changed beyond replacing the physically discontinuous azimuth metric with its fixed pi-periodic counterpart.


## Parent acceptance review

Parent independently ran32tests in24.80s, strict legal file/public-array/call-role audit errors[] and foreign-cwd standaloneJudge exit0. Read full README and both plans plus explicit nine-component differentiation, Zeeman enumeration/Voigt approximation, analytic polarized-transfer inverse, FFT degradation, bounded LM/pivotedQR state flow and two-cycle reconstruction; checked independent Judge's separate SciPyVoigt/4x4matrix/directPSF implementation and physics tests. Opened all6 actual scientific figure groups with all9atmospheric maps, every Stokes fit, unmodified/FFT-only/two-fix original comparison, both actual fitting cycles, all parameter distributions and independent acquisition/residual/parity diagnostics. Entire bound64x96realROI and all30wavelengths/4Stokes retained. Original complex-copy byte-count and rejected-residual state defects are independently demonstrated and minimally corrected with all original versions archived; local QR matches actual Eigen solve semantics, no widened1e-5parity tolerance. Complete native maps agree scaled7.83e-10. Parent required scientifically periodic azimuth metrics; final22metrics include8map NCC/NRMSE plus periodic-rms/pi and mean-cos2delta, and four independent forward residuals. Strict configuration, overflow, poor outputs, true-reference precedence, near0/pi equivalence and foreign/write-failure tests pass. Actual full CLI331.25s and final complete active notebook480.77s/6PNG/noerrors recorded. Weighting scales are not mislabeled photon noise; residual>1, weak-polarization ambiguity and depth-independent atmosphere limits remain visible. No actual atmosphere truth is invented or2019spatiallycoupled method falsely claimed. Exact source/data/paper-version/license provenance and semantic units/signs/factors/defaults/caller flow agree.

Final text audit also repaired literal question marks in notebook prose to unambiguous ASCII tau/mu/pi/lambda/delta and matrix inverse/transpose notation. Code cells and scientific results were unchanged; nevertheless the entire active notebook was executed again successfully, with six embedded figures. Evidence: pymilne_ascii_final_notebook.json/log.


## Explicit shared-kernel revalidation (parent)

The earlier generic `python -m jupyter nbconvert` dispatcher could select the host Anaconda converter/default kernel. Its successful historical output is not evidence of the pinned shared notebook environment. Original logs are retained; this correction does not retroactively infer an unmeasured historical version from PATH.

The entire active notebook was executed again with the explicit shared interpreter, direct `-m nbconvert`, and `--ExecutePreprocessor.kernel_name=imaging101-research`: exit 0, 440.143184200 seconds. All 3 code cells executed and all six scientific figure groups were freshly recomputed and exported as PNG/SVG/PDF with numerical sources. The independent actual-kernel probe prints sys.executable, Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1 and Matplotlib 3.11.1, using this exact kernelspec; the kernel argv names the shared interpreter directly. Exact command/cwd/runtime/logs are indexed in the maintainer-only `shared_kernel_revalidation/solar_crisp_stokes_vector_mapping/execution_records.json`; actual kernel probe/spec is `shared_kernel_revalidation/kernel_probe.json`. The parent reopened all six new figure groups and reran the standalone Judge/public-data/template/call-role audit without errors. Source mathematics, public observations, references and thresholds did not change. The complete current notebook and exported artifacts are SHA256 indexed in `shared_kernel_revalidation/final_evidence.json`.
