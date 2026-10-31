# Validation: cosmic chronometer acceleration

All selected-task mandatory gates passed. No actual physical ground truth exists; the hidden reference is a validated standard-method result. This physical inverse recovers the expansion deceleration history q(z), H0 and supporting rate/derivative/covariance from derived differential-age observations. It is distinct from existing stellar spectra, orbital, brightness, and atmospheric tasks. Parent bound the full30-observation/four-kernel contract before extraction.

## Provenance and actual original executions

Source https://github.com/carlosandrepaes/GaPP at actual remote revision b42de3b8f081036f1d32fc93fae4a87d21ddeb4f. The remote master.patch first line is saved as gapp_head.patch, not inferred from a local tracking branch. GPL3-or-later source Copyright Marina Seikel2012/2013 is preserved in shipped LICENSE/NOTICE. Full source archive, documentation, source data and logs stay maintainer-only. Exact per-file hashes/bytes are in archive/gapp_archive_index.json.
Paper: Seikel,Clarkson,Smith,JCAP06(2012)036, DOI10.1088/1475-7516/2012/06/036; exact full reviewed arXiv1204.2832v2 dated2012-05-31, private gapp_paper_v2.pdf/text. Paper2.1 equations2.6,2.11,2.12,2.14 establish kernels/conditioning/likelihood;2.2 equations2.15-2.25 establish derivative covariance;2.3 equation2.26 establishes per-point joint uncertainty. The paper uses supernova distances in its science application; this selected author-support H0_measurement.py instead directly conditions on derived H chronometers. We do not claim reproduction of all original SN/dark-energy figures, arbitrary equation-of-state inference, or MCMC marginalization.
Data: all first30chronometer rows of author-support hz_cc_sdss.dat, physically checked against Moresco2016Table4/arXiv1601.01701. The18remaining source rows are BAO measurements with a different acquisition and covariance issue, excluded by acquisition type before fitting, never by residual. Their complete original48-point runs remain explicitly historical diagnostics. Thirty points spanz0.07..1.965; preserve the author rounded coordinates and errors. They are real published derived expansion rates, not raw spectra/ages or simulated H. No target calibration or hidden true cosmology enters raw.
Private original environment: Windowsx86_64 CPU, Python3.12.4/NumPy1.26.4/SciPy1.13.1 and MKL2023.1, single-thread numerical backend. Separate oracle source directory gapp_compat changes only ambiguous array==None/!=None comparisons to identity and provides legacy absolute-import search paths. Original untouched source is separately archived; Python2 print-only driver is represented by equivalent actual GP/DGP API calls, not claimed unmodified script execution. Original thetatrain string protocol is retained.
archive/oracle_gapp.py executes all4kernels on full30CC and original48mixed groups and records theta,H,Hprime,per-pointcovariance/runtime. archive/oracle_gapp_initialization.py independently repeats every30CCkernel at initial scales0.5,1,2, before local extraction (12complete fits). Original log files and original NPZ/JSON are indexed. Native original runs completed with exit0; TNC iteration diagnostics are algorithm output. No import-only or cached-source notebook is counted as original execution.
Original initialization sensitivity maxH0.0009838045 km/s/Mpc and maxq8.2463e-5, with raw covariance variation0.12495 in its product units. Frozen dimensionless covariance normalization uses diagonal standard-deviation products. These are finite numerical sensitivity observations, not physical uncertainty intervals.
Before local extraction, capture_gapp_stages.py ran actual original K, gradients, Cholesky,alpha, likelihood and gradient and all kernel mixed/input derivatives on8testpoints. Fixtures are independent of local code. Frozen criteria are recorded in frozen_gapp_case.json. The additional observation gate was computed only from original arrays before fresh metric assessment as1.1times the worst original standardized residual RMS.

## Equation/function/case traceability

| Mathematical stage | Original source/function | Local callable | Independent consumer/check |
|---|---|---|---|
| SE/Matérn covariance and hyperparameter/input derivatives | gapp/covfunctions/squex.py,mat52.py,mat72.py,mat92.py covfunc/gradcovfunc/dcovfunc/ddcovfunc | physics_model.covariance_operators | frozen8x8derivative arrays; independent central differences |
| A=K+diag(error²),Cholesky,alpha | gapp/gp.py input_covariance/alpha_L | solvers.factor_observations | original K/L/alpha fixtures |
| Marginal likelihood and analytic gradient | gapp/gp.py grad_nlog_likelihood | solvers.negative_log_likelihood | original likelihood/gradient plus independent finite differences |
| Positive TNC hyperparameter fit | gapp/gp.py train | solvers.fit_kernel | original full4kernel and12initialization runs |
| Conditional rate and derivative | gapp/gp.py prediction;gapp/dgp.py dprediction | solvers.condition | full original mean/derivative and error-normalized comparison |
| Pointwise2x2 covariance | gapp/dgp.py f_covariances([0,1]) | solvers.condition | full original covariance, symmetry/PD and no fullcrossz claim |
| Differential-age expansion identity and q | physical measurement relationship; paper derivative-ratio cosmological relation | physics_model.acceleration_parameter | independent EdS,deSitter,LambdaCDM exact cases |
| Nonlinear uncertainty | paper2.3 pointwise joint Gaussian sampling | q_delta_standard_deviation and active notebook pointwise MonteCarlo | exact Jacobian quadratic form;50000draws per selected point, negative-H draws retained |

No known erroneous source formula was copied silently. Paper supports MonteCarlo for nonlinear uncertainty; deterministic delta errors are explicitly diagnostic only, with actual pointwise MonteCarlo comparison in the notebook. Matérn comparisons come from the selected author-support source; all4are preserved. Original conditional covariance is per-point2x2, not across the full grid. Numerical optimization and Gaussian conditioning cannot establish physical accuracy. High-z q signs are weakly identified and kernel dependent.

## Fresh checks, notebook and contracts

- PASS cli: native exit0, 0.511744s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\gapp_cli_final.log, SHA256 c4fa5ff7d3999874d63394455b67d6c376367685908f4312478122d40b9b95ca.
- PASS judge: native exit0, 0.145296s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\gapp_judge_final.log, SHA256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.
- PASS tests: native exit0, 1.276986s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\gapp_tests_final.log, SHA256 9208f5d1d335766e2ffbdf2797824ceded0e51948c416d970c2cd5da340f1b04.
- PASS notebook: native exit0, 4.722703s; command D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/cosmic_chronometer_acceleration.ipynb --output gapp_executed.ipynb --output-dir D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive --ExecutePreprocessor.timeout=180; log D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\astro_earth_physics\archive\gapp_notebook_final.log, SHA256 2d8ed9a52dbbdebc8bd7626022d47b2d468e36dcdb28b0495d6307cd944e1c0d.

Fresh metrics: {"H0_identity_relative_error": 0.0, "maximum_H_relative_error": 1.278101601065748e-06, "maximum_Hprime_sigma_normalized_error": 2.3787344674457122e-05, "maximum_covariance_sigma_normalized_error": 2.7389640475437475e-05, "maximum_q_absolute_error": 7.38660304022698e-06, "observation_standardized_rms": 0.689033225185066, "q_identity_error": 0.0}.
22tests pass: four-kernel exact original stages, all analytic derivative/likelihood finite differences, independent expansion laws and delta variance, full pipeline, reference/poor scoring, required key/shape/float64/NaN/Inf/symmetry/PD handling, zero hidden H, nonfinite boundaries, missing file and report-write errors. Early tmp_path fixture hit a Windows Temp access denial (environmental, not algorithmic); replaced it with a scoped TemporaryDirectory under task output and reran22tests successfully. No shared filesystem deletion was performed.
Canonical notebook is fully executed:5active code cells; full primary4kernel reconstruction and Judge, both additional initialization fits and50000pointwise correlated-Gaussian uncertainty draws at3redshifts for all4kernels. It prints the actual shared kernel sys.executable. All original/fresh comparisons are identified; original arrays are comparison only. Archive/gapp_executed.ipynb is the complete nbconvert result; canonical equals it. Frontend Windows Proactor/ZeroMQ warnings are infrastructure warnings, not numerical failures.180second timeout is conservative for this few-second complete benchmark, not an imposed scientific runtime condition.
All3actual scientific figures were opened with view_image and inspected: rate/derivative with all30errors and4kernels, q/H0 with conditional uncertainties and original/fresh comparisons, full initialization-refit differences. Units/axes/legend/error bars checked; no missing output fabricated. Figures are embedded and privately archived; temporaryPNG removed from task output.
Full source/call-role audit is archive/gapp_symbol_audit.json. All forward kernels, derivative signs, likelihood, conditioning and physical inverse are explicit local NumPy/SciPy mathematics. SciPy TNC and linear algebra are permitted numerical primitives; no original/domain GP library, oracle path or cross-task import occurs in solver/tests/Judge/notebook. Shared runtime is Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,WindowsCPU, threads1. Public requirements list only actual runtime subset.
README/raw/requirements are sufficient solver inputs: all3raw keys/shapes/float64/units documented, all5required output arrays scored, source prior/grid/conventions clear. No truth, fitted hyperparameter, source code URL, privileged calibration or seed enters raw. No public GitHub URL was found. No synthetic generator exists because observations are reused.
README and both plans were semantically reviewed against the executable equations and signatures: kernel polynomial factors and derivative signs, covariance scale, likelihood signs/normalization, exact TNC initialization/lowerbounds/defaults/stopstatus, grid/order/units and pointwise uncertainty. Note that source fullyBayesian GP terminology does not imply fitted hyperparameter marginalization.

## Asset consumers

- data/raw_data.npz:30measured redshift/rate/error entries; preprocessing, observation consistency and notebook.
- test/fixtures/upstream_stages.npz:original data/testpoints, all4theta/K/L/alpha/NLL/gradient/derivative arrays; comparing_test/test_upstream.py consumes every field.
- evaluation/baseline_reference.npz:all5original output arrays; independent Judge. No invented ground_truth.npz.
- notebooks/reference_outputs/upstream_measurements.npz:all5original arrays, independently identified notebook comparisons; covariance/derivative support the original scientific result and equality checks. fresh_measurements.npz:all5newlocal arrays, generated actively, retained as identified review measurement rather than reconstruction cache.
- output/reconstruction.npz and judge_report.json are fresh generated outputs. No temporary executed notebook or reviewPNG remains in output. Archive index maps exact original files/logs/papers/capture scripts to SHA256.

## Mandatory gate ledger

| Gate | Status and evidence |
|---|---|
| Exact paper/revision/data/hash and equation mapping | PASS; corresponding evidence above and indexed native logs |
| Original and clean complete bound input runs | PASS; corresponding evidence above and indexed native logs |
| Independent original stages and physical checks | PASS; corresponding evidence above and indexed native logs |
| Constants/units/precision/coordinates preserved | PASS; corresponding evidence above and indexed native logs |
| Standard CLI and all-array Judge | PASS; corresponding evidence above and indexed native logs |
| Poor/malformed/evaluator/reportwrite handling | PASS; corresponding evidence above and indexed native logs |
| Public sufficiency and no leakage | PASS; corresponding evidence above and indexed native logs |
| Acquisition/output metadata and no standalone metadata | PASS; corresponding evidence above and indexed native logs |
| Minimal shared CPU runtime without domain packages | PASS; corresponding evidence above and indexed native logs |
| Explicit local math and call-role audit | PASS; corresponding evidence above and indexed native logs |
| All22unit/parity/physics/pipeline tests | PASS; corresponding evidence above and indexed native logs |
| Active full executed notebook and comparisons | PASS; corresponding evidence above and indexed native logs |
| Historical/original/fresh labels and justified metrics | PASS; corresponding evidence above and indexed native logs |
| README/plans executable semantic/signature match | PASS; corresponding evidence above and indexed native logs |
| Exact template inventory/license/archive cleanup | PASS; corresponding evidence above and indexed native logs |

No selected-task mandatory gate is failed or unverified. Physical truth and fully marginalized cosmic confidence intervals are unavailable and are not claimed.

Raw SHA256 8c7e50ee405001f998e526800e3a60998427b7045c9555e05b8b316ec09fe08a; hidden reference SHA256 1f5ddbff208128c12b18f0d55dc00a3729220989ab1947cef3a756e828eccb2c; canonical notebook SHA256 2c26b5ca7d55d1c33028c2303d3120682ac45eaf1084f0070aed3daa7b2ecf6d.

Final parent review passed: source/equations/units/actualcallerflow/GT roles independently checked; every actual figure viewed; dual math engines passed. Current tests 23, actual figure groups 3. Full original and fresh numerical/physics/stage/strictcontract checks pass. Parent independently ran CLI/Judge/all23tests/complete5cellactiveNB/dualmath. Added true-reference precedence test, moved zero-mean prior from acquisition table into conceptual MethodHints, corrected coordinate-docstring typo; no mathematical algorithm or frozen bound changed.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 3 computed PNG outputs, 14 formulas. Private evidence: research_batch12_20261029/acceptance/cosmic_chronometer_acceleration/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Generated figure inventory 2026-10-29

Output PNG exports were SHA256-verified into maintainer benchmark_readiness_20261029/output_artifact_archive; no runtime reader consumes them. The active notebook now exports computed figures to a tasks-ancestor-derived maintainer directory, with IMAGING_REVIEW_EXPORT_DIR/IMAGING_REVIEW_DIR override. Original embedded outputs and every scientific comparison retained. Current and planned domain-nested path resolution verified without rerunning unchanged reconstruction mathematics. Archive/hash index: group_1_output_artifact_archive.json.
