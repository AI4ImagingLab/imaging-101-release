# Provenance, independent original runs and validation evidence

**Status: READY for independent parent review.** Complete unchanged original, full cleaned CLI and active four-inverse Notebook have actual successful child exits. All46 tests, independent physics/parity, both math parsers, all six actual figure inspections and final source/plan/interface review pass. Parent acceptance remains separate.

## Binding, source, paper and rights

The physical unknown is the complete31×31 complex density and its signed201×201 canonical Wigner field for the333 heralding condition. Acquisition is balanced single-shot homodyne with publicly supplied independent vacuum/unconditional variance calibration. Both outputs are graded. This is a quantum phase-space field reconstruction, distinct from coded-aperture gamma source imaging, proton Lorentz maps, temporal FROG fields and existing magnetic-source inversion. It is not a second count for a different quantum sample or a scalar fit.

Private archive P is `research_batch7_20261022/groups/physics_materials/xanadu/` outside the task. The complete raw/source archives, original logs, paper PDFs/JATS, private capture scripts and failed acquisition records remain there; final package contains no archive/checkout/PDF.

- Experimental paper: M.V.Larsen et al., *Integrated photonic source of Gottesman–Kitaev–Preskill qubits*, Nature642,587–591(2025), DOI https://doi.org/10.1038/s41586-025-09044-5. Primary complete EuropePMC JATS PMCIDPMC12176634 is archived as P/paper_fulltext.xml,92872bytes, SHA256`59e4a75fd5b6f93046f7843caa59d25ac7ed0a037ca7d273514380001b8de60e`. Task-relevant methods and complete supplementary state-heralding/tomography sectionD were read.
- Actual supplementary PDF P/supplement/41586_2025_9044_MOESM1_ESM.pdf was fetched through the EuropePMC supplementary archive. Its archive6785469bytes SHA256`2f64e3f1452d1352b3ea1881792d7456f3a2288edce729cf57b80f496f323c50`. Native manuscript/cached examples and bootstrap uncertainty context are distinguished from the bound original Notebook's100-update inverse.
- Mathematical method: A.I.Lvovsky, *Iterative maximum-likelihood reconstruction in quantum homodyne tomography*, J.Opt.B6,S556–S559(2004), DOI https://doi.org/10.1088/1464-4266/6/6/014. Actual archived arXiv quant-ph/0311097v2,25Nov2003,254890bytes SHA256`e759f67f24b975da269736cd6993a529261f8f7ec5a22dacb30e7e1b3b9f6ccd`. Its actual commutator i/2 and native2025 i convention are explicitly reconciled, rather than mixing Hermite scales.
- Original author repository: https://github.com/XanaduAI/xanadu-gkp-data, exact fetched remote revision `5a49ea80a3d3776042474a154e335246291ece8c`. Source ZIP SHA256`7fa1f96dbfde225d2171bb9999844c983fdf0d7ba5b89efd430da6dbd4a7b76e`. Exact remote source was actually retrieved; this is not an assertion that a cached tracking branch is currently freshest.
- Code license: Apache-2.0, preserved as task LICENSE and local adaptation notices. The displacement helper in upstream attributes an earlier MrMustard recurrence; local helper independently evaluates the explicit generalized-Laguerre closed formula with general special functions and separately checks actual upstream values. No quantum framework is vendored or imported.
- **Data type: real.** Original author README explicitly offers download/analysis of the public S3 measurements. Paper Data Availability points to the release. A separate raw-data redistribution license was not identified. The paper's CC-BY-NC-ND and the source Apache license are not silently applied to external raw data. This is a local research package, with no claim of unrestricted external redistribution.

## Actual full raw acquisition and lossless binding

Original public archive https://xanadu-gkp-data.s3.amazonaws.com/data.tar.gz was actually streamed and fully read:15,076,397,942compressedbytes, computed SHA256`292df1bbcbe6af0c04030845e1a3a88cca5f6ca7fd324dac09e0665efb057ee4`. No duplicate15GB compressed archive was retained. All100 original NPZ members are stored once in P/raw_original. Their exact names/hashes and headers are indexed in P/raw_stream_complete.json. Stream runtime1022.9033288seconds.

The first HTTP attempt ended early while reading member57. Its first56complete members and partial file are separately retained/indexed; they are not counted as the complete experiment. The successful full second stream validates HTTP Range206 recovery and compares reused member hashes. Tar member order is not chronological: its last five members are the earliest timestamps. Therefore the selection uses all100 files sorted by integer timestamp exactly as the author's Notebook does, never the first tar members, a shot cap or an easier subset. All100 contain333, so the original stop-at-first-missing-condition rule uses them all. Native raw shot count3,756,232; accepted count1,881,028.

P/binding_before_original.json freezes the exact native condition, file ordering,64phases/block,80% confidence rule,31-state Hilbert support,100iterations,hbar1 and201×201[-7,7] grid before local implementation. Stage relative tolerance1e−10/scaled absolute1e−12, update absolute1e−10, full density/Wigner scaled absolute2e−8 were fixed before local outputs.

P/package_binding.py performs only lossless conversion of existing measured arrays. Original trusted author object NPZ files are loaded privately during conversion; final public NPZ is numeric and runtime uses allow_pickle=False. Original ADC int16, distances uint16, phase/file indices int16 and calibration float64 are retained. Recomputed calibrated q and float64-cast distances are byte-equal to original capture. Public raw_data.npz27464872bytes SHA256`2dd0b825fde77cee78a28225bb07501d39626b200647e01ac48b5bfd0bd091d3`. P/binding_packaged.json contains every original file hash, final shape/dtype and full output contract. No observations are newly simulated; no generate_data.py is appropriate.

## Independent original full execution before extraction

Private command: research Python launches P/run_original.py, which runs the private oracle executable on P/capture_original.py. That driver executes original Notebook mathematical cells0,7,9,11,13,15,17,19,21 **verbatim**, importing unchanged analysis_functions.py. It excludes only download I/O cell3 because the complete exact archive was already acquired and verified. Original files are hardlinked into the native downloaded_data path without a second physical copy. No mathematical cell is filtered, replaced or implemented by the driver.

**Actual native child0,394.1942684999958seconds**, inner original driver393.7959195000003seconds. All100 files and all1,881,028accepted shots enter every native100-step update. Read-only tracing captures actual local R/amplitudes/Born/density stages at1/2/7/25/50/100, plus full preprocessing, final state, Wigner and marginals. Command, actual sys.executable, environment/backend, per-cell hashes, source hashes and runtime are in P/original/child_execution.json and execution.json; paired stdout/stderr and numpy_backend.txt are archived.

Original environment: Python3.11.11, NumPy2.4.6/SciPy1.17.1, CPU1thread, private oracle venv referring to general shared numerical primitives. Private ipywidgets supports the original Notebook's tqdm widget only; no quantum library is used. Source README historically suggests Python3.12; this execution is an actual unchanged-source modern backend reproduction, not a claim to recreate its historical environment. Main source SHA256`cf0f7d2800bccf3bb546f2e0219598c69972959753eb11e28712a5bf9832cc26`; original Notebook SHA256`88740a62b41cac59486fe3b59fddff78736a4cf43c35330ae21a7f6dd0d34af1`; both unchanged before/after.

Full original captured numeric SHA256`fb88b39ac8cef14690bb291a4f54b67cf5199a190d286f98b68f9b7c939376b5`. Its density has trace0.9999999999999999, Hermitian maximum error0, minimum eigenvalue−6.305006980114358e−18. Wigner minimum−0.024306356570102904, finite-window integral0.9990877567121863 and negative volume0.02417387617145962. The finite-window integral is not forced to one. The native log-likelihood increases on this actual dataset; minimum recorded step increment0.10255185049027205, last step fidelity0.9999999780449996. These are observed diagnostics, not a universal convergence theorem or exact state fidelity.

Only noninteractive Agg show warnings occur in original plot calls; they do not alter mathematics or output. No mathematical error is hidden by a shim. The cleaned runtime does not import any private source or oracle dependency.

## Independent stages and physical checks

P/freeze_original_stages.py calls the unchanged original module on383 real accepted samples spanning the full observation sequence, with original31-state initialization and seven updates. It freezes actual full R matrices, Born vectors, Hermite amplitudes, densities and histories. The fixture is not generated by cleaned code. Stage archive SHA256`6ac4475bd6ba87b8deaff7fafeb31fee733a61848ecfb5e8439e605cd11ecfb5`; capture/criteria file explicitly records no local implementation existed.

Independent general special-function Hermite amplitudes agree within4.833954429062804e−15. Number-state POVM norms through n30 integrate to one within1.7319479184152442e−14 on[-12,12]. An initial[-8,8] integration check failed because high photon-number tails extend beyond that analytic integration window. The integration domain was extended, preserving the1e−10 normalization criterion; no source, benchmark output window, observation, parity criterion or Judge boundary was changed.

Original Wigner recurrence is independently checked against analytic vacuum, single-photon and complex coherent states: maximum absolute differences0,5.551115123125783e−17 and5.551115123125783e−16, with full analytic-state integrals1. The complex coherent peak has positive Q/P consistent with the chosen amplitude, independently checking imaginary coherence and momentum sign. Actual nontrivial original displacement matrices include both native stabilizers and a rectangular complex-amplitude case.

P/freeze_likelihood_physics.py uses independently evaluated Hermite functions and a deterministic private trace-zero Hermitian test direction. At identity/31, finite-difference negative mean log-likelihood derivative−0.02041939506369772 agrees with the original R-gradient contraction−0.020419394976969117 to8.672860346559119e−11, versus a predeclared1e−8absolute criterion. The private seed selects only an analytic test direction, never a benchmark optical state or observation. The fixture has an identified test consumer.

## Equation/function/test traceability

| Paper or native stage | Actual original implementation | Local function | Independent evidence |
|---|---|---|---|
| SupplementD measured quadrature/vacuum scale | Notebook11/15, per-file calibration | preprocessing.calibrate | Full q byte equality, broad frozen raw samples; no Wigner-window shot rejection |
| SupplementD phase distortion | Notebook13 phase_distortion/fitting_function/curve_fit | phase_distortion/variance_model/calibrate | All100 original fit vectors and calibrated samples |
| SupplementD confidence selection | Notebook17 histogram-centre cumulative selection | calibrate native201bins/strict inequalities | All original cutoffs and1,881,028selection indices exact |
| Lvovsky Eq6–8 Born/Hermite POVM, mapped to hbar1 | maxlik normalized Hermite recurrence | hermite_amplitudes/born_probabilities | Actual complex amplitudes/Born fixtures; independent closed Hermite and vacuum/one-photon formulas |
| Lvovsky Eq9 likelihood | maxlik sum(log(denom)) | log_likelihood; maximum_likelihood history | Original7/full100history; independent likelihood directional difference |
| Lvovsky Eq10 R and Eq5 positive update | maxlik full R-rho-R/trace and symmetrizations | density_update/maximum_likelihood | Actual R/density seven-stage fixtures; Hermitian/PSD/trace independent checks |
| Native initialization/fixed budget/fidelity | maxlik identity31,100iterations,fidelity sqrtm | maximum_likelihood/step_fidelity | All captured density snapshots, LL/F history, full end-to-end bit equality |
| SupplementD canonical Wigner field | wigner_function coupled recurrence | physics_model.wigner_function | Independent analytic states and closed-Laguerre Judge; full original field |
| Native finite-grid marginal visualization | probability_distribution | quadrature_probability | Original analytic-state discrete masses; explicitly not likelihood PDF |
| Native stabilizer/displacement quantities | displacement/laguerre and Notebook26/28 | displacement direct closed formula; visualization.state_statistics | Actual original matrices, zero/adjoint physical checks; historical metrics separate |
| Photon support and phase-sensitivity controls | Explicit local-only additional experiments | maximum_likelihood with stated changed support/phase | Fresh active Notebook controls, no cached substitute |

This selected inverse has custom positive updates, not a generic optimizer; no adjoint solve or proximal regularizer is required. Finite-dimensional POVMs, acquisition normalization and gradient checks replace irrelevant generic-adjoint gates. R is an explicit task-specific operator.

## Historical comparison preservation

The original Notebook additionally displays cached analysis_results states111/133/335/444 and cached333 stabilizer/squeezing reports. These are not raw observations. P/capture_historical_context.py actually executes original cells0/24/26/28 verbatim, captures their numeric Wigner/marginals/matrices and cached stabilizers, and preserves unchanged source/cache hashes. Actual0/2.046356699996977seconds. Cached file SHA256`f8e4f8262310b4339dbc2d5a5eb850d733e122a12c6105c8c29033bb10f9e1a0`; final numeric context SHA256`da2429378eb719ddf748623788efcbfdf12c0efca6b7e3775390fc0c44228dc9`.

The active Notebook retains all four different-heralding cached patterns plus cached333 and native stabilizer context with explicit historical labels, alongside fresh bound333. It does not claim fresh raw reconstructions for those cached conditions, exact known state truth, or verified cached iteration/calibration parameters. Original measured calibration/phase/confidence/quadrature/ML/Wigner/density/marginal comparisons are preserved and augmented with genuinely computed sensitivity experiments.

## Hidden assets and consumers

| Packaged asset | Provenance and consumer |
|---|---|
| data/raw_data.npz | Lossless original real ADC/distance/index/calibration conversion; sole standard solver input |
| evaluation/baseline_reference.npz | Actual original full-data density/Wigner; independent standalone Judge; never labeled truth |
| notebooks/reference_outputs/upstream.npz | Actual original full density/Wigner,100-step diagnostics, phase calibration/cutoffs and snapshots; Notebook original comparison |
| notebooks/reference_outputs/historical_context.npz | Actual original cached-context cell run; explicitly labeled historical Notebook figure/table only |
| test/fixtures/upstream_stages.npz | Original383-real-shot seven-update matrices and analytic original transform checks; comparing_test/test_original_stages.py |
| test/fixtures/upstream_preprocessing.npz | Original confidence indices, broad actual samples and amplitude samples; full calibration parity tests |
| test/fixtures/likelihood_physics.npz | Independent Hermite directional derivative and original R contraction; likelihood physics test |
| notebooks/reference_outputs/fresh.npz | Active canonical Notebook primary full reconstruction and diagnostic arrays; complete-execution review tests |
| notebooks/reference_outputs/fresh_comparisons.npz | Active lower-support and uncorrected-phase inverses; actual scientific figure source |

No ground_truth.npz is fabricated. The standard solver reads none of these hidden references or fixtures. All NPZ files consumed in the cleaned package use allow_pickle=False. Original trusted object-bearing raw archives remain private only.

## Metrics and frozen boundaries

Pre-local independent complex64 serialization relative density error2.2112014269567773e−8; float32 Wigner serialization dynamic-range NRMSE3.971346495791132e−9. Quality limits were frozen as at least ten times independently observed valid precision error, with nonzero minimum tolerances1e−6 and1e−7. They are not relaxed after the cleaned output. Physical bounds require relative Hermitian error and trace error<=1e−6, minimum eigenvalue>=−1e−6 and independently closed-Laguerre rho-to-W maximum error<=1e−6. Both complete outputs must pass every metric. No phase/gain alignment, crop, sign fit or finite-window normalization exists.

## Actual commands and gate ledger

Research executable is the shared research-runtime Python3.11.11; CPU OMP/OPENBLAS/MKL/NUMEXPR1. No shared installation/upgrade. Clean requirements are NumPy2.4.6, SciPy1.17.1 and Matplotlib3.11.1 only. Private ipywidgets has original progress-display role; cleaned code does not import it. No quantum/computational-imaging domain library is present in cleaned source/generator/Judge/tests/Notebook.

- **PASS** actual complete original native mathematical path: P/run_original.py, child0,394.1942684999958s; exact data binding, all100files, original source hashes unchanged.
- **PASS** original-stage/analytic physics freezes before local extraction, including likelihood directional derivative and coordinate convention.
- **PASS** actual complete cleaned CLI: `PY -m src.main --diagnostics P/fresh_full.npz`, child0,332.6751925999997s. P/clean_cli_execution.json and paired stdout/stderr record actual process. Full density, Wigner, all100 LL/F values, all calibration parameters/cutoffs/indices/accepted q/phase and all six captured density snapshots are **bit-identical** to independently captured original arrays. P/clean_cli_parity.json records every comparison.
- **PASS** fresh standalone Judge actual0 from foreign cwd; rho/W reference errors0, Hermitian error0, trace error1.1102230246251565e−16, minEig−6.305006980114358e−18, independent Laguerre transform maximum discrepancy1.214306433183765e−16.
- **PASS** first12physics/stage tests0/10.50s and complete calibration test0/1.10s. Frozen tolerances unchanged.
- **PASS**24independent Judge/metric/contract cases0/26.55s, including valid complex64/float32, foreign cwd, zero/mixed/non-Hermitian/trace/phase/wrong-W controls, missing/corrupt/file/key/shape/type/object/NaN/Inf inputs, missing/invalid evaluator assets, thresholds and report-write failure; analytic Judge vacuum/single-photon field checks included.
- **PASS** complete active canonical Notebook primary inverse plus full-data cutoff10/20 and raw-phase controls: actual nbconvert child0,1007.221696099994seconds; five code cells executed1–5, no error outputs. Actual kernel sys.executable is the research-runtime Python recorded in P/nbconvert_execution.json. Canonical executed Notebook SHA256`4a069e526190c839250141cd940a06f00c8f6d95233c57bd364a8025d2e2dc1e`. Default active inverse never loads a cached reconstruction. Timeout7200seconds per cell was selected for full native measurements, not a seconds-only demo.
- **PASS** all six actual final PNGs opened and inspected; paired PNG/SVG/PDF, full-precision numeric NPZ, measured metrics JSON and per-panel provenance in P/figures. Scientific/visual findings are individually recorded in P/final_visual_semantic_review.json. Signed fields use neutral-zero diverging scales, comparisons use stated common scales, and cached context is explicit.
- **PASS** complete46tests after the independent truth-priority review fix, actual child0/46.32698710000841seconds, P/final_pytest_execution.json and stdout/stderr. Initial default Windows Temp directory access was denied during24fixture setups, while19scientific tests passed; explicit private --basetemp resolved that unrelated path permission without changing tests or thresholds.
- **PASS**38formulas parsed by actual MathJax3.2.2 and KaTeX0.16.22, zero failures. Final static AST audit covers all source, Judge, tests and active Notebook cells; no forbidden reconstruction imports, public GitHub URLs, hidden solver reference access or cross-task imports. Source/README/both plans were manually reviewed for actual mathematical signs, units, factors, signatures/defaults and call flow; P/static_audit.json and final_visual_semantic_review.json.
- **PASS** final canonical-only file inventory and required Apache notice. Task caches moved reversibly to P/final_cache_archive; no temporary executed Notebook, PDF, upstream checkout, original log, source archive, .npy or .mat remains in the task.

No unresolved mandatory gate is marked completed. Parent independent acceptance is separate from readiness.

## Parent review correction: truth priority

Parent review identified that the evaluator initially selected only baseline_reference.npz. It now selects ground_truth.npz whenever present, otherwise the method reference. Malformed actual truth is an evaluator error and never triggers fallback. This experimental task still supplies no actual truth. Three independent tests use an analytic vacuum density/Wigner field: truth wins over the different method reference, truth works without baseline, and malformed truth fails despite a valid baseline. All27 Judge tests pass0/29.53s; complete46tests subsequently pass0/46.32698710000841s. Numerical reconstruction, scientific metrics, thresholds, source formulas and canonical Notebook are unchanged. Parent explicitly permitted no redundant expensive Notebook rerun for this evaluator-only contract repair.


## 2026-10-29 public benchmark semantic version v2

The v2 public target is a physical density operator maximizing the mean log Born probability on every confidence-selected shot. Confidence selection uses the public strict cutoffs5554.373134328358,5811.4427860696524,7281.3880597014922 and retains1881028shots. The Judge independently evaluates the Hermite recurrence in4096-shot chunks and the public variance-fit phase/quadrature normalization; it does not import the estimator. Density Hermiticity, positivity, trace and density-to-Wigner consistency remain required. The accepted mean-log-likelihood deficit is at most1e-8nats/shot against the independent attained standard. Legacy density/Wigner identity is diagnostic only when no actual truth exists; actual truth, if provided, additionally has explicit strict truth errors. The existing35-test unit suite passed; the final actual-truth subset3tests passed after the final truth-priority change. The current CLI passes and a physical vacuum state fails. This certifies conditional likelihood quality, not physical state accuracy or a unique unconstrained estimator. Previous validation entries describe their original dated execution and v1 method-consistency semantics; v2 does not retroactively convert them into blind-agent results. The old Judge/metrics and failure evidence remain in benchmark_readiness_20261029/semantic_v2/gkp_homodyne_wigner_tomography/. No fresh blind-agent reconstruction was run in this review.
