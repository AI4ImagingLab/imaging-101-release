# Provenance and validation: PMMA signed directional second moments

## Bound problem, paper, code and data

This is a new material-science observation-to-spatial-field task: measured reference/sample grid images are inverted into transmission and three directional second-moment maps. It is not scalar calibration or chemical concentration fitting. No existing task was modified. The supplied observations already existed; no new synthetic benchmark data were generated. The analytic Gaussian blur in tests is a mathematical unit check only.

Paper: Michelle K. Croughan, Ying Ying How, Allan Pennings, Kaye S. Morgan, Directional dark-field retrieval with single-grid x-ray imaging, Optics Express 31(7), 11578-11597, published version 2023, DOI https://doi.org/10.1364/OE.480031 . Author reference repository: https://github.com/Xray-grill/DiDaFi_MKCroughan , remotely verified revision ef998eaf39021a98fae2bb8ff0bddfdd1493a4e7 (archived didafi_commit.json). Its BSD-3-Clause license is preserved verbatim as task LICENSE. Original source copyright is retained. The author repository supplies the measured PMMA dataset under Diffusive_DF_data/Grid/PMMA with the repository-wide BSD license and no file-specific contrary license; its README explicitly identifies supplied example and associated dark-field data. This is an existing author-distributed real measurement dataset, not newly generated observations or a paper figure digitization. The associated diffusive dark-field acquisition is distinguished from claiming that every ROI is the directional paper's printed figure.

Data URLs at the fixed revision are https://github.com/Xray-grill/DiDaFi_MKCroughan/tree/ef998eaf39021a98fae2bb8ff0bddfdd1493a4e7/Diffusive_DF_data/Grid/PMMA . Original files are Flats_00.tif, GridA_00.tif, GridA+Sample_00.tif, float32 2136 by 2544. Fixed rows 900:1368 and columns 1000:1468 were selected before the oracle. Public counts and flat retain float32 exactly and gain a batch axis only. The period 12.68550106609808 was obtained using the original Fourier calibration on the full grid-only observation; no sample or hidden field enters this legitimate reference calibration. Dark=0 and unity exposure normalization are assumptions, not measured camera calibrations. Missing detector pitch and propagation distance prevent converting the moments into angular variance; units remain detector-pixel squared.

## Private archive and hashes

The maintainer archive is D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/material_physics. DiDaFi/ stores the exact source and original TIFF files; capture_pmma.py and capture_pmma_rerun.py are private oracle capture scripts. Original logs and numerical captures are pmma_oracle.log/json/npz and pmma_oracle_rerun.log/json/npz. pmma_tensor_reference.npz identifies the original signed tensor; pmma_fit_covariances.npz records original curve-fit conditional covariance for uncertainty investigation. pmma_physics_audit.* and pmma_uncertainty_audit.* retain the independent scientific investigation. report_catalog.json and group_manifest.json index group-level materials; only private state is updated.

| File relative to archive or task | SHA256 |
|---|---|
| data\raw_data.npz | 0638ac43bbb370f97a1e659d635e125ab86950665c1df374a87734a0bccd2a44 |
| pmma_oracle.npz | 09f447f3705eed966a8d328cd695878f267ebd901a4ddc035e66ea6debc6fd94 |
| pmma_oracle_rerun.npz | 09f447f3705eed966a8d328cd695878f267ebd901a4ddc035e66ea6debc6fd94 |
| pmma_tensor_reference.npz | 753fd157adb3c627656ac40cf2335237555016b413e7cc852432e4d7850e74fc |
| pmma_fit_covariances.npz | 592e1d40b111b43e0ed9e7cafcc97c4edb179f0eb8dd36136f2ca587193e3f9e |
| test\fixtures\upstream_coefficients.npz | 946764c4410dcefaf09f94de4738b105d8625602fe1c76d7e14a328dfc14b91d |
| DiDaFi\Diffusive_DF_data\Grid\PMMA\Flats_00.tif | e15ec53eb44b283c6af8a473eda4fa0bd14470dad6fc25ee688c7c1efd229ffd |
| DiDaFi\Diffusive_DF_data\Grid\PMMA\GridA+Sample_00.tif | edba4ce6e746ad06340626dd45c569afd9aba1fa2a492c688ca9c73b81cb00bc |
| DiDaFi\Diffusive_DF_data\Grid\PMMA\GridA_00.tif | 468e103aeaae5fb0a27f1873ad6082e4254b9077f3651ea07f6e5d5ad5063b5e |

## Original oracle before local extraction

Both original captures execute the upstream function definitions without modifying mathematical expressions or control flow. Upstream files contain incomplete blank assignments in their __main__ examples; capture compiles only the complete function-definition prefix before __main__, then calls the author's functions with the bound observations. TIFF imread/imsave are adapted to in-memory arrays to preserve the removed tifffile.imsave API and record outputs; these adapters do not alter numerical arrays. DASK=False selects the official serial branch. No source function is replaced by cleaned mathematics. Commands, from the private archive:

```
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
./oracle-env/Scripts/python.exe capture_pmma.py
./oracle-env/Scripts/python.exe capture_pmma_rerun.py
```

Windows, Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, tifffile 2026.3.3, dask 2026.8.0, isolated oracle environment; source image I/O and dask are absent from the cleaned runtime. First run 949.6941385000027 s; repeat 975.3120752 s; both completed and saved all outputs. Dense fit failures remain NaN as in source; approximately 4.83e-6 of grid coefficients and 9.66e-6 of sample coefficients are nonfinite. Block nanmean produces fully finite spatial outputs. Every source output array agrees bitwise on rerun, including matching NaN locations. No random seed is used by the method. Source optimizer warnings/failures are archived rather than omitted.

The original nested simultaneous_equations function was independently extracted from dark_field_retrieval.py via AST before cleaned implementation. Its original angle/eigenvalue outputs were converted into components using Cxx=sx2*cos(theta)^2+sy2*sin(theta)^2, Cyy=sx2*sin(theta)^2+sy2*cos(theta)^2, Cxy=(sy2-sx2)*sin(theta)*cos(theta). These are signed eigenvalues, not their source display's signed square roots. No clipped physical covariance is asserted. The standard-method tensor reference, transmission and identified observational amplitude ratios are stored in evaluation/baseline_reference.npz. There is no actual moment ground truth. notebooks/reference_outputs/upstream_reconstruction.npz identifies this original result separately from fresh measurements. test/fixtures/upstream_coefficients.npz supplies original dense grid/sample amplitudes, final signed components, transmission and five identified ratios for parity tests. Each fixture has an explicit consumer in test/comparing_test/test_parity.py. These processed ratios derive solely from public observations and are not privileged extra data.

## Equation / stage traceability

| Mathematical stage | Original function/file | Local function | Independent test |
|---|---|---|---|
| Flat normalization | flat_and_dark.py:flat_and_dark_images | preprocessing.load_data | raw keys/constants and original full fits |
| Valid 14 by 14 / 7 by 7 correlations | coefficient_images.py:coefficient_images | solvers.extract_coefficients | test_original_fits_and_complete_pipeline |
| Five amplitudes, two cosine phases | nested correlation_expression | physics_model.correlation_model | dense original coefficient fixture |
| 13 by 13 nanmean and mode ratios | dark_field_retrieval.py:retrieve_dark_field | average_coefficients | test_original_closed_form_tensor |
| Directional log equations and least-squares moments | nested simultaneous_equations | moments_from_coefficients | original tensor fixture and independent Gaussian Fourier blur |
| Gaussian attenuation and signed eigensystem | stated forward model / original tensor equations | mode_ratios, covariance_eigenvalues | independent periodic blur, zero blur, negative eig retained |
| Evaluation metrics | independent definitions | standalone Judge compare/evaluate | known numeric case, malformed/huge/zero/evaluator contract tests |

## Physical validity and limitations

The four modes correspond to vectors (1,0), (0,1), (1,-1), (1,1), with physical x right and y up. The independent periodic Gaussian Fourier blur test constructs a positive grid, convolves with a known C=(4,2,1), T=.7, obtains mode amplitudes directly by FFT, and recovers all three components to about 1e-12. This checks cross-term signs, factors of two, attenuation sign, and transmission without calling the source or building expected results from the cleaned inverse. A zero-blur example recovers zero moments; a negative eigenvalue example confirms no clipping.

There are 141 negative smaller eigenvalues among 1225 locations (11.51%), with 7 negative larger eigenvalues. These are weak-scattering exterior estimates; the scattering interior is positive. The smaller eigenvalue ranges to approximately -.648 detector-pixel squared; its negative-region median transmission is .836 versus .668 in the remaining region. The four-mode overdetermination reveals nonzero model closure: original normalized directional ratio residual .06614234195266891. Local stationarity/edge changes and non-Gaussian scattering are limitations, not silently treated as exact physics.

Original curve_fit returned conditional amplitude covariance. Delta-method propagation through absolute amplitudes, block means, ratios/logarithms, the tensor inverse and the smaller eigenvalue gives a conservative fit diagnostic. Since windows overlap and grid/sample fits can correlate, the standard-deviation bound for a block mean is the mean of individual standard deviations, and channel bounds are added. No division by sqrt(169) is justified or used. Among 141 negative locations, 130 lie within one conditional SD, 139 within 1.96 SD, and all within 2.761 SD. This supports noisy weak-signal interpretation but does not supply calibrated camera confidence intervals; systematic closure mismatch is shown separately. No negative region is masked, clipped, or excluded from scoring.

## Cleaned implementation and evaluation

The cleaned runtime uses only NumPy 2.4.6, SciPy 1.17.1 and Matplotlib 3.11.1 plus standard library. pytest and Jupyter are shared development tools. No upstream package, TIFF reader, dask, or domain reconstruction library is installed or imported by the task. Source, tests, notebook and Judge have no cross-task imports. scipy.signal.correlate and scipy.optimize.curve_fit are general computation primitives; the local periodic acquisition expression, windows, phases, amplitude treatment, averages and explicit tensor inverse are visible locally. No generator is needed.

Prespecified parity tolerances: dense source fits rtol/atol=1e-8 with equal NaN locations; final tensor rtol/atol=1e-6; linear extraction from frozen original amplitudes rtol/atol=1e-10. Source reruns are bitwise; linear tensor conversion differs only by float64 algebra order. Judge requires each of Cxx, Cyy, Cxy and T separately: NCC>=.999999, dynamic-range NRMSE<=1e-5. Its forward mode residual must be <=.0668037653721956, exactly 1.01 times the original .06614234195266891. This is a standard-method tolerance, not a claim that the measured field equals true covariance. The public output contract explicitly states signed estimates, pixel units, component and map axes, and no alignment/clipping.

CLI command `python -m src.main` completed in 947.755236 s, log pmma_cli.log, exit 0. Fresh Judge passed: Cxx NRMSE 7.519948873175182e-17, Cyy 5.927228873688505e-17, Cxy 5.623543252154953e-17, T zero error, NCC within floating-point rounding of one. Fresh mode residual .06614234195266892. Judge is independent of src and validates both required keys, shapes, real float32/64 and finiteness with allow_pickle=False; all fields are evaluated. Huge finite, zero, missing/shape/dtype/complex/NaN/Inf, missing hidden assets and report-write failure cases are covered.

First complete fit/parity plus physical tests: 5 passed in 913.04 s, pmma_fulltests.log, exit 0. Additional final unit/Judge suite: 14 passed in .20 s, pmma_unit_final.log, exit 0. Combined final collection has 16 tests (2 parity/pipeline plus 14 unit/contract tests). All use shared Python, single-thread environment, `-p no:cacheprovider` and private basetemp. Parent independently repeats final complete suite. No failed scientific test was bypassed.

## Notebook, figures and semantic review

The notebook actively reconstructs both complete 455 by 455 fit maps with conditional covariance recording, saves fresh outputs, calls the independent Judge, and compares original/fresh fields. Source outputs are explicitly labeled standard-method results. Actual transmission-only, isotropic, and signed-tensor forward predictions use the fresh reconstruction; they are constrained comparisons, not fabricated independently fitted algorithms. Four multi-panel figure groups cover measurements/fields, component oracle parity, directional closure and constrained comparisons, signed eigenvalues and conditional uncertainty. Private figures live in figures/pmma_grid_dark_field_tensor with PNG/SVG/PDF, numeric_sources.npz, measurements.json, and SHA256 figure_manifest.json. Active execution and visual inspection are recorded below when complete.

README and plans were semantically checked against all formulas, coordinate signs, units, default parameters and callable signatures. The map center is defined by the 7-pixel reference-patch center at window offset 4+3=7 plus averaging-start mean6, giving ROI start+13, not the 14-window geometric center6.5. Math is ASCII/raw LaTeX; no literal question-mark corruption or control-character escapes are accepted. Public README, raw observations, and requirements contain no GitHub links or hidden fields. No metadata JSON or extra task report directories exist. Temporary notebooks and private original source remain outside the task.

## Gate status

- Passed: paper/revision/data/license binding, oracle before extraction, repeated exact data run, frozen independent fixtures, complete local acquisition/inverse math, physical Gaussian mode test, signed weak-signal investigation, per-function and strict Judge contracts, CLI and fresh Judge, public contract/self-containment, dependency/call-role and plan semantic reviews.
- Passed: complete pipeline/upstream stage parity suite and final additional unit/Judge suite.
- Pending at this log entry: active notebook completion, all actual exported figures viewed, final inventory/hashes and foreign-CWD Judge. This task is not declared complete until the appended final evidence resolves them.

## Final active execution and handoff

Passed: full active nbconvert execution completed with no error cells; pmma_notebook.log and private pmma_executed.ipynb retain the capture, canonical notebook contains executed outputs. Command: shared Python -m nbconvert --to notebook --execute notebooks/pmma_grid_dark_field_tensor.ipynb --output pmma_executed.ipynb --output-dir private archive --ExecutePreprocessor.timeout=3600, JUPYTER_PATH=private/jupyter, kernel imaging101-research, single-thread variables as above. Active full reconstruction runtime 921.2972015000014 s, exit 0. Fresh per-component and forward metrics match the CLI to reported full precision. Fresh uncertainty reproduces the earlier independent source fit diagnostics: 141 negatives, 130 within one conditional SD, 139 within 1.96 SD, maximum absolute ratio2.7600336731539805.

All four actual PNG figure groups were opened and visually reviewed: material transmission/contrast, three signed components, source scatter and error maps, four directional observed/predicted scatter plots and spatial closure, constrained transmission-only/isotropic predictions, eigenvalues and conditional fit uncertainty. Scientific labels and units are readable; weak negative regions remain visible. SVG/PDF copies and exact numeric sources with hashes are indexed in figure_manifest.json. No historical published metrics are mislabeled as new measurements.

Foreign-current-directory standalone Judge completed exit0, pmma_foreign_judge.json. Final fast suite now has 15 passing tests, including an independent finite-difference amplitude-to-eigenvalue uncertainty propagation case; together with the two earlier passing long parity/pipeline cases, final collection is17 tests. pmma_unit_final.log records the final fast suite; pmma_fulltests.log records the completed dense original parity. Parent independently repeats the long suite.

Final semantic/inventory audit: pmma_final_audit.json reports errors=[] and per-file hashes. README/plans/notebook have no dropped backslashes, literal question marks or control-character formula corruption. Source and test AST parses pass; no forbidden domain imports or backend mutation exists. No cached original result is substituted for active reconstruction. All mandatory local gates are passed, including notebook/visual review and final inventory; ready for independent parent audit.

Private capture_pmma_tensor_reference.py reproduces the independently extracted original nested function and tensor conversion; final recapture verified exact existing frozen reference without overwriting it.


## Parent acceptance review

Parent independently ran17distinct tests: full16-case suite passed in1006.99s including complete455x455 double-channel coefficient fitting and full pipeline, plus newly added independent uncertainty finite-difference case passed in.32s. Foreign-cwd Judge0 and source/inventory audit errors[]. Read source/local cosine fits, flat normalization, block amplitudes and explicit directional Gaussian-to-tensor inverse; inspected all4 actual scientific groups. Real PMMA reference/sample/flat acquisition, signed noisy second-moment estimates and transmission all judged. All141 negative minor-eigenvalue pixels retained; conditional-fit covariance/Cauchy bound is explicitly not calibrated camera confidence, and systematic directional closure mismatch remains visible. Independent original runs and final original tensor-function capture match; no PSD clipping, hidden target or actual covariance truth invented. Active full921s notebook and conceptual comparisons actually executed.
