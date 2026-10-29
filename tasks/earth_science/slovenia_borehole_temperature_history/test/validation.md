# Provenance and validation

## Bound problem and selected method

Measured subsurface temperatures at all seven Slovenian sites of Xibalba version4 are inverted into21 successive30-year ground-surface-temperature anomalies per site. All237 native depth records are retained. The published standard rank-two SVD comparison is the selected method, fixed before local extraction; no historical temperature truth is available. Original PPI/BTI results are retained as explicitly labeled comparisons, with their independently demonstrated quantile defect. They are not the evaluated standard method.

The closest old binding is fire_wall_inverse_heatflux, which uses wall-temperature measurements to infer external fire heat flux in an engineering geometry. This task uses geological borehole depth temperatures to infer historical ground-surface temperature under half-space diffusion. The catalogue's DIRBE dust and exoplanet thermal tasks use radiative measurements and different unknowns. Complete prior catalogue and relevant old README review found no duplicate observation/unknown binding.

## Primary sources and remote identity

- Paper: Cuesta-Valero et al., Geoscientific Model Development15,7913–7932 (2022), DOI https://doi.org/10.5194/gmd-15-7913-2022, published28October2022. The paper's central standard SVD comparison is implemented; no claim is made that its proposed bootstrap estimator is the benchmark output.
- Original author repository: https://github.com/cuestavalero/CIBOR ; remote API tree revision5f736c1eeb453042308198976786dc6e58fb0c0c verified27October2026, archived in cibor_tree.json. Selective exact-revision source archive cibor_archive/. Code MIT license, retained as LICENSE.
- Measured archive: https://doi.org/10.6084/m9.figshare.13516487.v4 ; Figshare file31073704, xibalba_underground_temperature_database.zip,1145688bytes inspected before download, CC BY4.0. Private archive xibalba.zip and xibalba_archive/. Author README explicitly identifies measured borehole logs and lists its quality/depth-selection criteria. Each Slovenian file identifies site coordinates, logging year and original NOAA/Huang2008 source. Classification: author_real_processed. The benchmark adds no resampling, synthetic primary observations or hidden-target calibration.
- CIBOR's bundled demo profile was not used. The complete seven-site measured Slovenian subset was selected before extraction. Dataset grouping is geographic and includes every Slovenian file; no site was removed for poor fit.

## Private archive index and execution

Paths in this section are relative to the private astro_earth_physics work directory outside the task. cibor_binding.json contains each site, measurement year, count and original SHA256. cibor_prepare_oracle.py and cibor_oracle/capture.f03 define the input-only adapter and public-object capture. The capture calls unchanged original profile, SVD and forward-model modules. Original arrays include regression coefficients, all three anomalies, all three histories, complete forward predictions and the heat matrix obtained by invoking the original forward model on basis vectors.

The selected original run uses WSL Ubuntu24.04, GNU Fortran13.3.0, x86_64 Linux, double precision, CPU, OMP_NUM_THREADS=1. Build command from cibor_oracle: `gfortran ../cibor_archive/src/mod_kinds.f03 ../cibor_archive/src/mod_logger.f03 ../cibor_archive/src/mod_regression.f03 ../cibor_archive/src/mod_profile.f03 ../cibor_archive/src/mod_fm.f03 ../cibor_archive/src/mod_svd.f03 capture.f03 -o capture`. Execute `OMP_NUM_THREADS=1 ./capture`. Final independent rerun exit0,0.4822458999697119s; cibor_oracle_runtime.json and cibor_original_capture_final.log record command/result. Compiler/kernel are recorded in cibor_oracle_environment.txt. No stochastic seed is used by the standard method.

Supplementary original PPI uses100 diffusivities in[0.5e-6,1.5e-6), rank2 and all three anomalies at every site. Original regional BTI uses1000 replicates, seven sites,1000 uniformly sampled diffusivities, a fixed Fortran seed vector271828 and one OpenMP thread. Adapted input drivers and unchanged source compile commands are archived in cibor_prepare_comparisons.py and cibor_oracle/run_comparisons.sh. All completed with exit0; complete wall time was approximately6minutes, not instrumented individually. Original warning logs and output reports remain in cibor_oracle/. Repeated equal-endpoint integer-sampling warnings arise from the fixed rank choice. The precise run time claim above applies to the selected central SVD path only.

The clean runtime is shared CPython3.11 with NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1 on Windows, CPU, single-thread numerical execution. No upstream package or compiler is used by the task. Checked free physical RAM was12.0GiB and virtual memory26.5GiB; private archive downloads were small. Original and cleaned environments remain isolated.

## Pre-extraction freeze and independent tests

cibor_frozen_acceptance.json was created2026-10-27T05:24:48.869074+00:00 before local extraction. It hashes original stages and fixes kernel tolerance1e-13, anomaly tolerance1e-10, history tolerance1e-9, independent analytic matrix error6e-8 and adjoint tolerance1e-12. Original matrices versus analytic erfc differ by at most5.094117050408187e-8 due the preserved author approximation. Independent NumPy rank-two solves on the captured original matrices agree within3.8e-15K. Original full-profile temperature RMS ranges0.03246018880785645–0.08528435935456252K. Judge boundaries remain NCC>=.999999, dynamic-range NRMSE<=1e-6 and maximum observation RMS<=.095K. The nonzero agreement tolerance admits floating-point implementations; the physical residual margin is approximately11percent above the original maximum. No threshold was relaxed after local results.

| Paper/algorithm stage | Original function | Local function | Validation consumer |
|---|---|---|---|
| Conductive half-space response | fm_type.forward_model; svd.erfcc | heat_matrix; complementary_error | Frozen basis-vector operator; independent analytic erfc and telescoping identity |
| Quasi-equilibrium background | profile.anomaly; regression.linreg/invert | detrend; regression_inverse; ordered_product | Frozen coefficients and three anomaly arrays |
| Truncated spectral inverse | svd_type.inversion; svdcmp/svbksb | truncated_inverse | All seven original three-column histories; independent rank-two solve |
| Forward fit | fm_type.forward_model | forward | All three original forward profiles, plus independent Judge analytic response |
| Adjoint | Transpose of acquisition matrix | adjoint | Independent dot-product identity |
| Complete standard path | capture driver calling original modules | reconstruct; main | Full147-output parity and CLI |
| Metric definitions and CLI | Independent specification | standalone judge; visualization.metrics | Known analytic metrics and malformed/poor submission tests |
| Original uncertainty discrepancy | PPI.wtd_quantile; BTI.quantile | No corrected method silently substituted | Original two-point diagnostic and labeled notebook comparisons |

## Investigated discrepancies

The original regression reports Student-t95percent half-widths but the anomaly routine multiplies them by2. These are preserved as author perturbations and are not described as calibrated95percent intervals. Both original uncertainty quantile routines implement `(1-gamma)*x[k]+gamma*x[k]`, repeating the lower value. A private diagnostic copy changes only symbol visibility, exposing the unchanged original PPI routine; it returns0 for the median of[0,10], where linear interpolation gives5. Source, executable and log are archived as cibor_oracle/diagnostic*. This failed quantile validation is preserved in the notebook. Neither issue changes the selected central SVD reconstruction.

An initial local anomaly comparison differed by1.4e-9K because LAPACK inversion and BLAS summation changed a cancellation-sensitive regression covariance. Explicit original LU pivoting and accumulation order resolved this under the unchanged1e-10 tolerance. A first matrix-capture attempt read an uninitialized original `prof.temp` field instead of `prof.atemp_original`; capture was corrected before fixtures were frozen. No invalid capture is used by tests. An early extra analytic test evaluated a telescoped response at shallow unobserved depths10–15m and exceeded the matrix-domain6e-8 bound. The final checks separately test the frozen acquisition-matrix bound over the measured depth domain and the exact telescoping identity of the preserved approximation. The original approximation is not claimed to have a global6e-8 scalar-erfc bound.

## Artifact consumers and result provenance

data/raw_data.npz contains only native measured temperature/depth, site geometry, record counts and measurement years. evaluation/baseline_reference.npz contains the original central method result, not ground truth. test/fixtures/upstream_stages.npz contains original matrices, coefficients, three anomaly/history/forward arrays for all sites; every key is consumed by test_upstream_stages. test/fixtures/upstream_quantile_diagnostic.npz contains original diagnostic inputs/output and is consumed by the notebook. notebooks/reference_outputs/upstream_uncertainty.npz contains author-run PPI/BTI text-report curves, retaining original rounding and explicitly identified defects. notebooks/reference_outputs/fresh_measurements.npz contains freshly measured notebook runtime and Judge metrics; it is a recorded result, not a substitute reconstruction.

No published/historical metric is used as evidence of execution. The canonical notebook actively reconstructs all seven profiles and all21 bins, writes a fresh submission and evaluates it. Fresh rank1–4, diffusivity and geothermal-perturbation comparisons are computed using cleaned code. Original PPI/BTI curves are independently reproduced report comparisons and never labeled fresh. Seven figures were inspected; site-year colorbar and paired observation/fit colors were corrected and the affected figures re-inspected.

## Gates

- Passed: primary paper, verified remote source revision, measured-data identity, source license and traceability.
- Passed: unchanged original selected method on exact bound observations; independent frozen intermediates before extraction.
- Passed: all source, preprocessing, full-pipeline, metric, Judge and applicable physical checks;16 tests pass.
- Passed: standard CLI and standalone Judge exit0; known poor outputs fail and malformed/missing/evaluator/write errors produce error reports and exit2.
- Passed: full active six-section notebook executes natively through nbconvert; all seven figures inspected and all meaningful comparison groups retained with explicit provenance.
- Passed: self-contained README, complete batch-first NPZ/output contract, acquisition constants checked, no public GitHub/answer leak, no synthetic primary or invented truth.
- Passed: local explicit acquisition and inverse mathematics; permitted numerical dependencies only; no domain solver, cross-task imports or upstream runtime access.
- Passed: README and plans checked against equations, units, signs, normalization, callable signatures and execution flow.
- Passed: task inventory, license notices, canonical executed notebook and temporary-file removal.
- Explicit failure outside selected estimator: original PPI/BTI interpolated-quantile validation. Their uncertainty coverage is unverified and no such claim is made.

## Final commands and measurements

Fresh Judge metrics: {"ncc": 1.0000000000000002, "nrmse": 1.6909584299538166e-14, "max_temperature_rmse": 0.08528435353642987}.

- `cli`: `-m src.main`; exit0; wall seconds0.7004226000281051.
- `judge`: `evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; exit0; wall seconds0.2950112000107765.
- `tests`: `-m pytest test/ -v`; exit0; wall seconds2.0593424000544474.
- `notebook`: `-m nbconvert --to notebook --execute notebooks/slovenia_borehole_temperature_history.ipynb --output slovenia_borehole_temperature_history_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=600`; exit0; wall seconds4.671067399904132.

Exact commands and logs: cibor_execution_manifest.json and cibor_final_{cli,judge,tests,notebook}.log. Notebook timeout600s; observed complete execution4.67s. The Windows event-loop selector-thread and local kernel transport warnings are recorded and did not affect execution. Initial PowerShell redirection reported a NativeCommandError despite completed nbconvert output; a native Python subprocess rerun explicitly confirmed exit0.

## SHA256 identities

| Asset | SHA256 |
|---|---|
| xibalba.zip | 0f5103d41d87270f6e77cf7eed1bd3212f183293c7df2a16ea50462baddf33bf |
| cibor_frozen_acceptance.json | 22c9b105ef8ec0e203895b3e6c1ac86bbd5444b0aa37824f70a5d4103d35fe77 |
| staging/slovenia_borehole_temperature_history/data/raw_data.npz | 58f37d77781f8431a77766a3e613caf586f42e34f74900b96596cc210276e7f5 |
| staging/slovenia_borehole_temperature_history/evaluation/baseline_reference.npz | d6670bbf99a7e851baa8bb437ed8ca16fd61ac7b2aa8ec0b0e703f10348a04fe |
| staging/slovenia_borehole_temperature_history/test/fixtures/upstream_stages.npz | 7b4c35401906bb8490937e4487c1abef15ac7a08688793b7cff130c4500f89b2 |
| cibor_archive/src/mod_bti.f03 | 271255099eb5f0f6b2f2c5c0ee0badbcc463af2c6c161ac828a53d621383dba3 |
| cibor_archive/src/mod_ensembles.f03 | cd30bb3d812bc81aac30a56c5892112fbaac8fbd39380fad2ee7a4d2aac33f87 |
| cibor_archive/src/mod_fm.f03 | 6c402d995549974b29902f2c23f2cd0cf86fc6939a12d1e5c50d41c2c13ea595 |
| cibor_archive/src/mod_ghf.f03 | 45a72eb3e722ab32e58ff448b29792b74319be1a5e68dbfed10ea11613bda848 |
| cibor_archive/src/mod_kinds.f03 | f02f26034916110752672097986a330148be25e6278c4c5aea46ff70d2b81a36 |
| cibor_archive/src/mod_logger.f03 | e979373b74b317d54209db638523c72266baea98b03bfb7b2357f124198f6fa8 |
| cibor_archive/src/mod_ppi.f03 | 16879c31193867233f6a5f63184caa7527ea42455873ccb7086584fc782250b9 |
| cibor_archive/src/mod_profile.f03 | 9474d6e842d384afa44387820e19e2b7bdaa4b7f4880c444d4b8d03fc43a98e3 |
| cibor_archive/src/mod_regression.f03 | 22c70066d0bd219c33f304598214ac44abf2ff5d5a48091c782c5b1ddcc65da0 |
| cibor_archive/src/mod_svd.f03 | 4a8574f32d1d709dc28855bb4e1af3fd129778eb49a26f620e5cca15e8e8b69a |


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 8 executed code cells, 7 computed PNG outputs, 15 formulas. Private evidence: research_batch11_20261027/acceptance/slovenia_borehole_temperature_history/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
