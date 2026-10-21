# Provenance and validation: full broadband Faraday spectrum

## Scope and actual observations

The bound task reuses the complete original 288-channel **simulated** spectrum `tests/RMsynth1D_testdata.dat` supplied by RM-Tools. It does not represent real sky data. The selected inverse is ordinary inverse variance RM-synthesis followed by the public single-threshold Hogbom CLEAN option, retaining absolute Q/U flux and the weighted reference-wavelength phase convention. No new synthetic observations or random scene were generated: `src/generate_data.py` is intentionally absent under the existing-observation exception. The all-channel source sample is not a reduced toy extracted from a larger scientific scene. Its underlying described screen has RM200rad/m², but the selected output is a resolution-limited restored spectrum with residual and sparse components. No independently saved, frequency-independent exact physical FDF matching this unnormalized spectral-index-bearing sample exists in the supplied observation. `baseline_reference.npz` is an independently executed method result, never relabeled true physical emission.

The public input contains only frequencies, Q+iU polarization and known component noise. StokesI and synthetic object descriptions are not shipped as privileged observations or calibration. No answer, hidden seed, target-derived calibration or original code URL is present in public inputs. Every original channel value is retained at float64 precision; only Q/U packing and one batch axis change the container.

## Exact sources and private archive

Private archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/faraday/`. This is never imported by the solver, Judge, tests or notebook.

- Repository: https://github.com/CIRADA-Tools/RM-Tools ; remotely verified master commit **bcfd15c1282d61a8be196d16e677b27516b9a680**, package1.4.11. `remote_commit.json` is the actual GitHub REST response; `tree.json`, `upstream.zip`, and exact `source/RM-Tools-bcfd15c1282d61a8be196d16e677b27516b9a680/` preserve complete source. This is verified remote evidence, not an assumed local tracking revision.
- Selected paper: Van Eck et al., *RM-Tools: Software for Analyzing Polarized Radio Spectra*, arXiv2601.20092v1, full HTML https://arxiv.org/html/2601.20092v1 ; subsequent journal DOI10.3847/1538-4365/ae3dea is identified separately, not substituted for the read version. Full `paper2601.20092v1.html` and `paper_source.json` retain the exact paper/version/hash. Selected scope is sectionsIII.2–III.3 and the no-Stokes-I-normalization option. Optional QU fitting, peak characterization/uncertainties, complexity moments and super resolution are outside this task; no missing method is shown as computed.
- Physical foundations: Brentjens and de Bruyn2005 DOI10.1051/0004-6361:20052990. The selected current source Gaussian-width convention is preserved, not silently replaced with an earlier approximation.
- The original 288x7 ASCII input SHA256 is **765f0d9cf9935203b88ef93e2dea046a4d1cec9d32746b7b3b54318ecd5cba7a**. Bound NPZ SHA256 is **669b32e0e7be055a983bec65e7167453a5e31d61c39dd70102643c8ffb7e0d31**. `bound_observations.npz` is the exact independently prepared oracle input; the public file is copied unchanged. Public extraction omits unused StokesI and uncertainty columns; Q/U values are neither rescaled nor normalized.
- Code and supplied synthetic observation are under the repository MIT license, copyright(c)2020 Canadian Initiative for Radio Astronomy Data Analysis. Full license is preserved in task `LICENSE`; adapted operator/inverse source carries attribution. The complete original source and paper remain outside the task.

Real data were sought before choosing this existing synthetic observation. SPICE-RACS has explicitly licensed CC-BY4.0 real Q/U spectra and a documented 2024 correction multiplying previously erroneous Q/U by2. Archive metadata includes `portal_62975.json`, `portal_62977.json`, `portal_versions.json`, API probes and original/corrected collection identities. The corrected full deposit has291203files/~9.94TB; the full-data listing and an attempted offset/limit listing timed out. It is not claimed that no real data exist. This task transparently chooses a directly runnable, licensed complete existing synthetic observation instead of silently using uncorrected or invented sky data. An initial guessed unrelated arXiv page was rejected by title and is not paper evidence. API access/provenance is separate from physical validation.

## Independent original execution before extraction

`capture_oracle.py` imports exact original `RMutils.util_RM` functions; it does not import cleaned code, edit upstream source, or replace an upstream inverse. The selected caller invokes `do_rmsynth_planes` on all288observations with nBits64/NUFFT eps1e-12, `get_rmsf_planes` with the complete extended response and fitRMSFFalse, and `do_rmclean_hogbom` with gain0.1,1000maximum updates, cutoff3sigma and windowNaN (the original public `window=None` option). A second original CLEAN execution on the same independent dirty input verifies deterministic equality. It is not presented as a different-noise or different-object experiment.

Actual command: `oracle-env/Scripts/python.exe capture_oracle.py`, private archive working directory. Process exit0, measured original method runtime **0.028315699892118573s**,50updates, repeated clean maximum absolute difference0. The complete capture precedes local production code. Original input, intermediates, final arrays, numerical criteria and environmental role are in `upstream_stages.npz`, `oracle_run.json`, `frozen_criteria.json`, `oracle-complete.log`. The initial direct-import attempt failed because uninstalled source lacked importlib package metadata; installing the exact source with `pip install --no-deps` in the private oracle fixed this. The failure log is retained as `oracle.log`, without relabeling it a successful run.

Oracle: Python3.11.11/Windowsbuild26200, NumPy1.26.4, SciPy/FINUFFT/Astropy/deprecation/tqdm and exact installed RM-Tools1.4.11. The isolated oracle dependencies do not enter the cleaned runtime. Original package installation and its stdout/stderr are archived; PowerShell's stderr-native-warning wrapper can report1 after pip's successful install, whereas the actual source-capture exit0 independently verifies the functioning environment. No pip upgrade was performed. Complete versions and NumPy configuration are indexed in the final archive manifest/environment records.

The clean shared runtime is Python3.11.11, NumPy2.4.6 and Matplotlib3.11.1 as pinned in `environment-research.yml`. It has no RM-Tools/Astropy/FINUFFT reconstruction dependency. CPU execution uses one OpenBLAS/OMP/MKL thread; neither GPU execution nor other platforms are claimed. Kernel `imaging-101-research` explicitly selects the shared runtime; the notebook prints its executable and NumPy version. General complex matrices, sums, norms and NumPy argmax implement all inverse mathematics locally. Matplotlib is used only for review plots, without selecting a backend in library modules.

## Frozen criteria and original measurements

Before local implementation, `frozen_criteria.json` set transform/pipeline parity rtol1e-9/atol2e-11, reconstruction and component complex relative-L2 boundaries1e-7, and forward consistency boundary**1.209597539639874**. Original transformed noise0.005892556509887896Jy/beam, cutoff0.017677669529663688Jy/beam, wavelength reference0.10327486263730823m², restoring FWHM58.90951914103005rad/m², sampled peak201rad/m²,50updates and forward statistic1.0996341269453398 are captured measurements. Forward boundary is1.1times that independently measured source statistic, frozen before cleaned results. The grid step3rad/m² and physical expected RM200 yield a sampled recovered peak within one bin, but that context is not a fabricated reference array.

The two array metrics are explicitly named complex relative L2: norm(estimate-reference)/norm(reference), without magnitude/phase alignment or NCC/NRMSE relabeling. The finite Fourier-sum implementation replaces a general NUFFT primitive while retaining the same mathematical operator, weights, phase reference and inverse updates. Finite64-bit direct sums differ from the original numerical NUFFT at approximately1e-13 relative. Nonzero1e-7 scoring tolerances admit numerical approximation rather than self-comparison zero tolerance. Repeated original CLEAN is deterministic; actual fresh errors are independently measured, not used to relax boundaries.

The third metric independently forward-models the required components using public observations, sums complex squared residual/noise² and divides by2*288. The denominator is the number of real channel observations, not an inferred degrees-of-freedom correction after data-dependent component selection. No physical uniqueness or absolute sky truth follows from any of these comparisons.

## Equation/function/test traceability

| Selected paper stage | Exact upstream function | Local function | Independent case |
|---|---|---|---|
| Wavelength, weighted reference and transformed noise; paperIII.2Eqs1–3 | util_RM.do_rmsynth_planes; corresponding source weighting caller | preprocessing.prepare | exact fixture lambda2/weights/reference; metadata test |
| Complex inverse transform/sign/normalization | util_RM.do_rmsynth_planes | physics_model.dirty_spectrum/adjoint | original NUFFT dirty spectrum; analytic thin screen and complex adjoint identity |
| Sampling response, full extended support and width | util_RM.get_rmsf_planes | physics_model.rotation_measure_response | entire803-bin independent response/grid/width fixture |
| Global complex peak selection and subtraction; paperIII.3 | util_RM.RMcleaner._cleanloop, first selected loop | solvers.hogbom_clean | complete original50-update final components/residual/restore; independent below-cutoff and residual-operator cases |
| Unit-peak Gaussian restore plus residual | util_RM.gauss1D; do_rmclean_hogbom | solvers.hogbom_clean | independent full original restored spectrum; full CLI fixture |
| Physical observation equation | paperIII.2; source thin-screen convention | physics_model.forward | analytic phase/sign/scale; adjoint; independently implemented Judge forward |
| Complex metrics/strict submission contract | benchmark-defined, no source metric relabeling | independent evaluation/judge.py; visualization.relative_l2 | known complex scalar examples, overflow/malformed/poor/missing/reference/write-error tests |
| Full bound acquisition and selected inverse | exact original function sequence in capture_oracle.py | main.run/main; notebook | new subprocessCLI and foreign-cwdJudge; active nbconvert |

The source contains an optional second masked CLEAN loop; its low-level default window0 can continue to the iteration cap. This task selects the explicitly supported public windowNone/low-levelNaN path, so the second loop condition is false. We do not claim to reproduce default-window0 while omitting its mathematics. Current source width3.8/delta(lambda²), weighted t0, float64 types, original response peak/index/crop/tie convention and50actual updates are preserved. None is silently changed to a preferred alternative. The paper's unselected complexity-moment expression is not used by runtime or evaluator.

## Actual clean commands, execution and scientific review

- `python -m src.main`: complete all288-channel computation; exit0. Independently timed process0.14846330019645393s (`fresh_cli.json`); internal method time is printed separately.
- `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`: exit0,0.138103999895975s (`fresh_judge.json`). Active notebook also invokes the exact standalone CLI in a new process on its newly computed submission.
- `python -m pytest test/ -v --basetemp <private>/pytest-final`: initial complete suite26passed in2.78s, process3.0764955999329686s exit0; one later metadata assertion was added, with final rerun recorded in the appended acceptance evidence. The first parity collection failed before src existed (`red.log`); Judge tests likewise failed before judge.py existed (`judge-red.log`). These are intended test-first failures, not hidden successful checks.
- `python -m nbconvert --to notebook --execute notebooks/faraday_depth_rmclean.ipynb --ExecutePreprocessor.kernel_name=imaging-101-research --ExecutePreprocessor.timeout=180 --output faraday_executed.ipynb --output-dir <private>`: exit0,3.024162500165403s (`nbconvert.json`). Timeout180s is justified by the measured full inverse/figures, not a runtime-based simplification. All four code cells execute, no errors, three embedded PNG scientific figures; the authoritative notebook contains those actual executed outputs. Temporary executed copy and extracted PNGs are private, never retained in task output.

Actual fresh notebook values: reconstruction complex relative error**4.244245764528543e-13**, components error**1.174829573676006e-13**, forward statistic**1.0996341269453034**; all frozen metrics pass. Original and fresh measurements are separately labeled. No historical paper metric is relabeled as this run.

Root actually viewed all three extracted scientific figures. The full-interval dirty sidelobes and restored peak are retained; original/fresh restored curves overlap, and signed real/imaginary discrepancies are visible at1e-13scale. All components and fixed-gauge complex spectra are shown. Both full288-channel Q/U observations, forward models, component-noise residuals and803-binRMSF are visible. The metric figure separately plots complex numerical error and physical data consistency against frozen boundaries. Axes carry correct flux/depth/frequency/noise units, legends are readable, and no missing method is replaced with zero. Source spectral-index/model limitations remain explicit in the notebook and plans.

Semantic review checks source signatures/defaults and actual caller flow, positive forward/negative adjoint signs, no hidden bin-width factor, per-real-component noise convention, weight/reference normalization, padding, shifted response, Gaussian peak normalization, stopping/gain/window selection, complex output convention and independent evaluator. README/plans do not claim optional routines as implemented. Inspection verifies no library backend selection and no cross-task/upstream/private archive imports. Public README and requirements have no GitHub URLs or hidden-reference keys/paths.

## Asset consumers and inventory

| Shipped asset | Provenance | Consumer |
|---|---|---|
| data/raw_data.npz | Unchanged bound oracle NPZ, original full288Q/U/noise/frequency values | Solver, notebook, metadata tests, independent forward metric |
| test/fixtures/upstream_stages.npz | Independent original outputs before extraction: preprocessing, dirty response, full CLEAN and model | Stage/full parity tests; not generated by functions under test |
| evaluation/baseline_reference.npz | Original complete restored spectrum and components only | Independent Judge and labeled notebook original comparison |
| notebooks/reference_outputs/upstream_measurements.npz | Independent original runtime/update/forward scalar measurements | Notebook provenance comparison |
| notebooks/reference_outputs/fresh_measurements.npz | Actual active notebook runtime/update/metric scalars | Maintainer provenance of fresh run |
| output/reconstruction.npz; output/judge_report.json | Actual active notebook computation and its standaloneJudge | Maintainer acceptance/review |
| LICENSE | Full source MIT notice preserved | Redistribution obligations |

Reference arrays retained in both hidden baseline and stage fixture have distinct necessary consumers (scoring versus independent complete-stage parity), rather than unused archival copies. There is no second evaluation report tree, metadataJSON, standalonePDF, root wrapper, legacyNPY copy or imported source checkout. Final private `archive_index.json` records exact hashes/byte counts for source archive, paper, input, capture scripts, environments, frozen criteria, logs and shipped artifacts. All final gates below rely on the specifically recorded selected-method scope.

## Completion gates

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper, verified revision, data hashes and stage mapping | passed | Exact remote REST/source/paperHTML and table above |
| Original and clean full runs on bound observations | passed | Original exit0; CLI exit0; active all288-channel notebook |
| Independent parity and applicable physics | passed | Frozen original stages; analytic phase/scale/sign/adjoint and operator residual |
| Constants, units, precision and coordinates | passed | Exact copied arrays/weights/reference plus metadata checks |
| StandardCLI and fresh standaloneJudge | passed | CLI/Judge actual exit0; all3metrics within unchanged boundaries |
| Judge malformed/poor/evaluator failure handling | passed | Full contract tests and actual foreign-cwd invocation |
| Public allowlist sufficient; no hidden leakage | passed | READMEfull acquisition/complex output contract; no publicGitHub/seed/answer |
| Minimal environment and local core mathematics | passed | NumPy/Matplotlibonly; noRM-Tools/Astropy/FINUFFTdomainruntime |
| Unit/parity/physics/fullpipeline tests | passed |26initial complete; final metadata-added suite recorded below |
| Active notebook and figure review | passed | Full selectedinverse, actual nbconvertexit0, three root-viewed figures |
| Metric provenance and frozen thresholds | passed | Distinct original/fresh; beforelocalfrozencriteria and actualrepeat |
| Plans/README/source semantic consistency | passed | Sign/scale/window/interfaces/caller review above |
| Final inventory, archive and license | passed | Requiredassets/consumers; MITnotice; indexedprivatearchive |

This accepts a fully validated **simulated-observation method-reproduction task**. Real-sky validity, a unique physical emission distribution, full paper experiment reproduction, GPU portability and unselected algorithms are not claimed.

### Final metadata and parent acceptance

The final complete command `python -m pytest test/ -v --basetemp <private>/pytest-metadata-final` passed **27 tests**, exit0; exact output is `final-metadata-tests.log`. It includes the newly added exact public noise/frequency/c/reference-grid check. Source mathematics, references and frozen boundaries did not change after the executed notebook. A separate parent invocation from the batch archive working directory passed the actual output through the standalone Judge; `<batch>/faraday_depth_rmclean-fresh-independent-judge.json` records the unchanged three passing metrics. Parent AST/import/call-role, complete public-array and file-inventory audit reports no errors, with the mandatory source MIT license explicitly allowed. Final `cleaned_environment.json`, `oracle-package-environment.txt` and `archive_index.json` make interpreter/backend/package and file-hash evidence concrete. The final notebook has four executed code cells and three actual root-inspected scientific figures. All mandatory selected-task gates are passed.
