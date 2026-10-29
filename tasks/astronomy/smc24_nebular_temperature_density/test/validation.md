# Provenance and validation

## Physical binding and source identity
Author publication: Luridiana,Morisset,Shaw,PyNeb:AnewtoolforanalyzingemissionlinesI,A&A573A42(2015),DOI10.1051/0004-6361/201323152; primary arXiv1410.6662v2(28October2014) full mathematical sections and AppendixA reviewed. Code:https://github.com/Morisset/PyNeb_devel, exact remotely verified revision7aab5be7cf460e5e063a005253b1d60cffb63ecb. The actual original run uses official PyPI PyNeb1.1.32 in private isolated pyneb-oracle venv. Its **all35installedPythonfiles** have exact matching gitblobSHA1 values at this remote revision, not merely matching version strings; archive/pyneb_source_verification.json records each file/hash. PyPI remote wheel SHA25696a63479536b4e53fdb36e1da20d72b9291cf920e042d765dd4a0c4ca3160cbe, size28751057bytes. Source code/license is GPL3; license and attribution preserved. Original atomic text files with scientific-source comments are privately archived under PyNeb/atomic_files.

Observed quantity: SMC24 processed integrated optical/IR line intensities from original author table/samplefile. Classification author_real_processed: these are published observational compilation values, not generated synthetic data or simulated truth. Exact observing instrument/epoch for each compiled line is not supplied by this source; do not claim raw instrument frames or independent metrological calibration. Source sample_scripts/smc24.dat numerical values match the task's current author binding. The paper appendix's SIII18.7micron entry10.4 differs from current samplefile10.0; preserve current selected file and distinguish modern-code/data reproduction from numeric recreation of the2015figure. The source extraSIII33.6micron line is invalid in the selected atom label table and not used by any of the seven original diagnostics. It emits a warning upstream. Local input preserves all23validobservations without silently substituting a33.5line or inventing flux. Original invalid entry remains private in source archive with recorded hash.

Atomic probabilities/levels/collision strengths are independent published model coefficients. Selected datasets:NIIFFT04/T11,OII Z82-WFD96/Kal09,OIIIFFT04-SZ00/SSB14,SIIRGJ19/TZ10,SIIIFFTI06/TG99,NeIIIGMZ97/McLB00. This explicitly modern atomic-data set differs from unspecified2015defaults. Default observational5percent errors are an assumption of the original review script, not supplied measured detector variance. No actual Te/Ne truth exists; baseline_reference.npz is original-method output, never ground truth.

## Original exact-observation run before extraction
Private oracle_pyneb.py imports only official isolated original package, reads the exact source table with original Observation.readData, sets original CCM89 correction and all seven diagnostics, then executes allfour original getCrossTemDen paths and saves corrected fluxes, outputs and six-ion atomic stages. Native exit0; initial original numerical runtime0.17729989998042583s. Capture was completed before extraction and frozen in pyneb_preextraction_original.npz/json; independent full rerun after extraction is additionally preserved with stdout in pyneb_original.log. The rerun is not used to generate expectations from cleaned code. Original environment: Windows11build26200,AnacondaPython3.12.4,NumPy1.26.4,SciPy1.13.1,PyNeb1.1.32; CPU, allBLASthreads1, noGPU. Shared cleaned runtime:Python3.11.11,NumPy2.4.6,Matplotlib3.11.1; no PyNeb installed/imported in cleaned runtime. Oracle dependencies were installed only in private isolated venv, never shared runtime. Seeds: not applicable, deterministic method.

## Independent fixtures and equation-stage traceability

| Published stage | Original file/function | Local function | Independent validation |
|---|---|---|---|
| Eq1-2 statistical balance/normalization | core/pynebcore.py Atom.getPopulations | AtomicIon.populations | six-ion3Tby3n frozen populations, balance residual,sum1,positivity |
| Eqcollision/detailed balance | Atom.getOmega/getCollRates | omega/rates | original collision/rate grids, independent detailed balance |
| Eq3emissivity | Atom.getEmissivity | emissivities | original full emissivity matrices at3Tby3n |
| Published extinction input handling | extinction/red_corr.py CCM89/getCorr; Observation.correctData | extinction/deredden | all23original corrected values |
| Temperature/density ratio search | Atom._getTemDen_1 nsect_recur | nsection/invert_ratio | fourfull crossed solutions |
| Coupled diagnostics | core/diags.py getCrossTemDen | crossed_solution | original4solutions, shared data,rtol1e-8 |
| OriginalAppendix7diagnosticcomparison | sample_scripts/smc24_diagnostics.py | diagnostic_figure | all7curves actively evaluated, reviewed |
| Full benchmark | original observations plusgetCrossTemDen | main.run | fresh nativeCLI/Judge and complete executednotebook |

Fixtures:test/fixtures/upstream_atomic.npz contains independently captured omega/rates/populations/emissivity matrices. notebooks/reference_outputs/upstream_original.npz contains corrected observations and originalfourconditions plus labeled atomic stage arrays; consumed by paritytests and notebook. All copied raw calibration arrays preserve native precision/constants and batch-first interface. No truth/hiddenseed/target-derived calibration is released.

## Numerical discrepancies investigated
Initial local emissivity missed inverseAngstrom-to-cm factor1e8; independent original fixtures detected it and the formula was corrected, with original strict rtol1e-10 preserved. SII collision table has8levels but radiative/energy model5: compare only the active upper-left5levels for Omega, while original populations/rates/emissivities are already5levels. This is the original min-level convention, not a changed model or relaxed tolerance. Absolute extinction (rather than Hbeta-renormalized extinction) matters especially for optical/IR diagnostic pairs. Original parser stores cHbeta and E(B-V)float32; recomputing E(B-V) under NumPy2 shifted intermediate precision. Preserve the actual original E(B-V)=0.0325583778321743011 in public acquisition metadata, as required for supplied constant fidelity. All corrected fluxes now pass1e-10relative; final4conditions pass1e-8relative. The original recursion tolerance maxError0.001 is fractional in code despite percent wording in upstreamdocstring; documented actual behavior. Crossed temperature update uses olddensity, explicitly preserved. No threshold was loosened to hide a discrepancy.

## Metrics and review
Separate maximum componentwise relative error across all4temperatures/densities, eachbound1e-4 frozen before extraction. Both must pass. Independent knownmetric cases verify direction/value; each row/quantity perturbed1percent must fail. Allzero fails; malformed key/shape/dtype/complex/NaN/Inf yields error with emptymetrics; foreignCWD works, unavailablehiddenassets and reportwritefailure nativeexit2. Referenceagreement does not imply physicalaccuracy or a uniform singlezone.

Complete notebook actively reconstructs observations with local source, computes all7physical curves, raw/corrected23linecomparison, original/freshpairedTeNe and actual omit-extinctionrefit. Full nbconvert nativeexit0 with explicit sharedkernelargv, finalwall5.436613s; actual fullfitruntime and ablation values in fresh_measurements.json. Private original full executednotebook archived as pyneb_executed.ipynb and canonical task notebook retains actualoutputs. Windowszmq selector/TCP warnings occurred; no cell or numericalerrors. All3figures were opened using view_image: readable units/legends, all7curves present, fourfreshsolutions labeled, source-original/freshbars match, actual omittedextinctionconditionchanges visible. Plotlegend headroom corrected and complete notebook reexecuted afterwards. Display assumptions are labeled, not treated as noise measurements.

## Gate status
Passed: exact paper/source revision association and data hashes; original exact observations; frozen atomic/preprocess/fullpipelineparity; independent equilibrium/detailedbalance/conservation; supplied units/dtypes/grid/constants; nativeCLI/Judge; strictmalformed/poor/CWD/assets/writechecks; self-contained publicobservation/calibration/README; no truth/privilegedseed/GitHubURL publicleak; local mathematics and minimalruntime without domainpackages; fullpytest; complete fresh notebook+figures; source/formulas/plans/signatures consistency; inventory/licenses. Parentindependentacceptancepending. Limitation: original linecompilation instrument/epoch not in author source, explicitly author_real_processed rather than rawreal; do not claim an independent physicalTeNetruth or2015defaultatomicdataset reproduction.

## Private archive index and SHA256

- `PyNeb/smc24.dat`: `699b565623be8f3b2cf6dec6503cbf960a8ce771950ab7b186090ba4e686ca6f`
- `pyneb_preextraction_original.npz`: `a8253bf6d3fa5acba9dc7c72979647e317eaea9f4809e946df555cdd5df8efbe`
- `pyneb_preextraction_original.json`: `5673233e4456aa09bf2f414e83e7674f3a4a9a62878941beb9b9b9110fc183d7`
- `pyneb_atomic_fixtures.npz`: `2021eef03cfe5fc6757313e1ea595874ba73976868339863be287a64c1ef03ab`
- `pyneb_calibration.npz`: `12780529f13ff4fc2f520980d3dd32b487f351d99f588a4ab4385f13ba8ee752`
- `pyneb_source_verification.json`: `9c1d5e22f39f8f5b71287016c2b33274fbd498eb4ba75382c93f2c0d7205bca2`
- `oracle_pyneb.py`: `3ffefcdf2a1098f23aaa6f6716099d26fd7e89fff8fefcdad0c190995eedfb9b`
- `export_pyneb_calibration.py`: `b9279372a4582fe878fe6f76c6b4719dd7cef71787a6fd7d24e5ea2af267f383`
- `pyneb_executed.ipynb`: `484ffae46326c1ebab946a7516596a3688531e0551d57bd96b9a5ea6f5eae07d`


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 3 computed PNG outputs, 4 formulas. Private evidence: research_batch12_20261029/acceptance/smc24_nebular_temperature_density/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Final public-estimator sufficiency amendment

Independent review identified that tight method-reference comparisons require the selected conditional/finite estimator to be defined publicly. README Method Hints now specifies the necessary mathematical prior, feasible set, or canonical finite mapping and constants, without source code, a full hyperparameter table or step-by-step pseudocode. Physical unknown/data, local mathematics, all Judge criteria and all executed notebook code/results are unchanged. The old README/hash, exact source correspondence and new hash are retained privately in acceptance/smc24_nebular_temperature_density/public_mapping_amendment.json and public_mapping_before_README.md. Parent semantically compared every new equation/convention to actual source and independently rerendered all documents with both math engines. No new numerical reconstruction is claimed from this documentation-only amendment.


## Benchmark public-contract semantic revision (2026-10-29)

Status: scientific acceptance contract v2, distinct from historical exact-method reproduction. Original README, Judge and metric configuration were retained in the maintainer archive `benchmark_readiness_20261029/semantic_v2/smc24_nebular_temperature_density/`. Public observations, hidden arrays and existing reconstruction bytes were unchanged. The reference implementation and its notebook retain their historical reconstruction role; they are not the definition of permitted solver behavior.

The Judge now independently assembles the atomic rate balance from public calibration and tests both extinction-corrected line ratios at every returned pair. The maximum fractional ratio tolerance 0.001 is the original scalar diagnostic tolerance, fixed before inspecting new solutions. It no longer compares to a five-step Jacobi approximation. Historical output residual: 0.00038851112025506396; independent log-parameter least-squares solution residual: 1.0180648912783141e-7. Both pass. Malformed files/keys/shapes/dtypes/nonfinite values and unavailable calibration return error. Per-quantity 50% perturbations fail. The complete task unit suite passed under the shared runtime (see maintainer semantic_v2/smc24_pytest.log).
