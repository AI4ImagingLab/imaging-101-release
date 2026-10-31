# Provenance and validation evidence

## Bound physical inverse and data

Recover three latent continuum/line flux histories, two top-hat transfer responses, and stochastic continuum amplitude/damping time from actual1989NGC5548IUE light curves. All60epochs per channel retained. Author processing averages measurements within a single observing shift; not raw detector data, not simulation. No dense true latent field or independent transfer GT exists. evaluation/baseline_reference.npz is the actually rerun standard method, never relabeled truth. No target-derived calibration is public.

Paper: Zu,Kochanek,Peterson2011ApJ73580 DOI10.1088/0004-637X/735/2/80; archived arXiv1008.0641v2 header2May2011(the PDF typesetting date says29May2018). Data paper Clavel etal1991ApJ36664 DOI10.1086/169540. Both complete primary PDFs inspected and stored privately. This task selects the published stochastic/top-hat model and current author's MAP code path, not the paper's complete MCMC posterior or a claim of unchanged2011binary execution.

Original repository https://github.com/nye17/JavelinTNG exact remotely verified commit01198c53a114cdf405c300b5944d62c56b4473cf. GitHubAPI commit response in remote.json, full archive and filehash inventory retained. GPL2 code permits extraction; retain LICENSE. Historical author's javelin repository directs to this current rewrite. Real measurements from https://www.asc.ohio-state.edu/astronomy/agnwatch/n5548/lcv/IUE1989/ : c1337.gex,lya.gex,c4.gex, sourced from official AGNWatch index. Public archive does not supply a blanket modern open-data license; no such license is invented. Raw spectra/instrument response not supplied; source-published fluxes and errors are the bound observation.

## Original run before local extraction

Private archive base imaging-101-maintainer/research_batch10_20261027/agn. run_original.py first demonstrated that two default Nelder-Mead starts exhaust1600evaluations. Their outputs were exploratory, not convergence evidence. run_original_converged.py establishes the declared restart protocol before local code: for each of3initials call original Rmap_Model.do_map, restart from returned point on warning, at most5calls. Starts0and1converge after one restart; start2converges directly. No original numerical function was modified or monkeypatched. Its actual warnings and per-round parameters/objectives are retained in original_run.json and native_repeat.stdout. Command: oracle_env/Scripts/python.exe -X utf8 run_original_converged.py, private agn working directory. Exit0 and complete covariance/likelihood/prediction arrays frozen into original_fixture.npz. Original prediction uses do_pred(dense=10), all600points and conditional SD. First unredirected successful capture and independently repeated subprocess run agree; the frozen tests use the latter. No paper/cached notebook output substitutes for this exact-data execution.

Isolated oracle virtualenv imports original source only through its own .pth. It shares existing numerical site-packages via another .pth (explicit environment exception to avoid duplicate installations), but the clean research-runtime path cannot import javelin. Python3.11.11, NumPy2.4.6,SciPy1.17.1,Windows CPU, OPENBLAS/OMP/MKL threads1. No GPU or randomness in MAP. NumPy backend and BLAS build details are in environment.json; Optional threadpoolctl is absent in the shared runtime; no new dependency was installed. Clean runtime NumPy/SciPy/Matplotlib. No per-task install/upgrade performed.

## Equation/function/test traceability

| Paper mathematical stage | Original code | Local implementation | Validation |
| --- | --- | --- | --- |
| Section2 continuum covariance and top-hat response, covariance appendix | spear_covfunc.py normal vectorized primitives | physics_model explicit exponential antiderivatives/covariance | Native initial/final180x180covariance; independent finite-width numerical integrals, OU limit and SPD |
| Sample centering and channel constants | zylc.LightCurve | preprocessing.prepare | Exact sorted time/channel/errors, shift invariance |
| Mean-marginalized Gaussian objective | lcmodel._lnlike_from_U,lnpostfn_spear_p | solvers.likelihood | Four objective terms and mean corrections at initial/finalstates |
| Original selected MAP search | Rmap_Model.do_map/scipy.fmin | solvers.reconstruct | Complete3start results, explicit restart warning evidence, end-to-end stage bounds |
| Conditional process reconstruction | Rmap_Model.do_pred,PredictRmap | solvers.predict | All1800means and conditional SD; batched solve adaptation only |

Each symbol, units, mean convention, initialization, constraints, simplex stopping and prediction formula is explained in plan/approach.md. plan/design.md signatures were checked against actual functions; no unsupported MCMC, posterior precision or mass-recovery claims. Original source adds finite1e300sentinels for multiple invalid bounds; local guard adds once when any hard bound fails, irrelevant to bound feasible stages and verified full numerical parity. All scientific objective terms for feasible parameters are unchanged.

## Frozen gates and numerical conventions

binding_before_local.json fixed input and fixture hashes, stage1e-9relative/1e-8absolute tolerances, scaled parameter.005and profile RMS.01before implementation. Tests were written first. Original and local use observer-frame days and identical native channel flux scales, diagonal measurement noise, no clipping of observation residuals, and source default argsort. The extrapolation grid is defined publicly. No hidden truth or source URL appears in visible material. Original fixture in test/fixtures/upstream.npz is consumed by parity tests; notebooks/reference_outputs/upstream.npz separately supplies review original comparisons. baseline_reference.npz supplies independent Judge. Fresh cleaned_measurements.npz and fresh_measurements.json are produced by active notebook for report/review consumption, never read by solver. Duplication is limited to these documented consumers.

NCC/NRMSE inappropriate for tightly identifiable time/transfer parameters and heterogeneous channel scales; declared separate channel RMS, transfer scaled error, positive continuum log ratio, and observed residual WRMS are explicit finite metrics. Thresholds frozen before cleaned results, supplemented by float32submission validation. Original observed WRMS approximately1.39884,.72758,.68442. Observation bounds are1.05*original+.01and are numerical consistency criteria, not independent physical truth. Full precision in metrics.json. Histories beyond acquisition range are model extrapolation. Fixed-parameter31x31lag slice is not a profile or posterior; three converged modes and boundary-width solution remain shown. This task is distinct from all239existing tasks: temporal echo transfer and latent UV flux, not stellar kinematics, lensing, image deblurring or purely forward simulation.

## Execution gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper, remote revision, data and mapping | passed | Archive/index and trace table |
| Original exact observations | passed | Complete native rerun exit0, both warning restarts explicitly resolved |
| Frozen parity and independent physics | passed | 3 comparing tests +5 independent mathematics tests |
| Unit/Judge contract | passed | Full23test run; missing/key/shape/dtype/complex/NaN/Inf/poor/alloutputs/foreigncwd/writefailure/truthprecedence |
| README/plan/source semantic review | passed | Operator, means, units, interfaces and local-only runtime inspected |
| Fresh CLI/Judge and final active nbconvert | passed | Independent parent commands exit0; complete notebook19.24s |
| Final math and figure inspection | passed |19formulas rendered; all6parent PNGs individually opened and reviewed |
| Final inventory/cleanup | passed | Parent inventory frozen after cache removal |

The appended parent evidence confirms all formerly pending execution/review gates.

## Private artifact hashes

- `original.zip`: SHA256 `7bc9dc6b39eff3fe1f5c3ddbe7974e350d091debf8aeae6df7124ea5b73989b8`, 15261133 bytes.
- `remote.json`: SHA256 `15f4c10b471470924af467f84558d7906805741c0abec497d763e0c7fc16f54e`, 6007 bytes.
- `source_inventory.json`: SHA256 `c108950b166c66834df48db2fa23c1a72c305b88794c1533a58cabf42bc617fb`, 75336 bytes.
- `method.pdf`: SHA256 `649676f4327aa54384f508e58223b397e3b2a0538387c8134104f47c49871229`, 646276 bytes.
- `data_paper.pdf`: SHA256 `3ec1b22f9cedf1212e539cbffd1312e28e306c0c40803c06bdccae2c23b98eaa`, 1776152 bytes.
- `run_original.py`: SHA256 `3c543be7e072ef81ac5ac8e50702ff9352ccf289919320de4c5eeb73659df10e`, 2350 bytes.
- `run_original_converged.py`: SHA256 `311ea61808d0e4450dcf407959de522a660bbd1541b439ee2a1e3264244cd42d`, 2732 bytes.
- `original_run.json`: SHA256 `0eea62ec90125c9299dd67d5cccbc59778e731043a48e5220e6dc445511ef173`, 2868 bytes.
- `native_repeat.stdout`: SHA256 `3f080b5c49cbaeca3a388eaf1a41a91df5ada16c3d81b04e42161c11aaa71d8b`, 2035 bytes.
- `native_repeat.stderr`: SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, 0 bytes.
- `original_fixture.npz`: SHA256 `5685ab74b9ce110fa817c97e58dec6851b4c30c1e8c5bae66bef09089549646b`, 489197 bytes.
- `binding_before_local.json`: SHA256 `c64dda81334957757ea53324d56a2bb4de22976f083a89b2238a227c706d529c`, 1275 bytes.
- `data/c1337.gex`: SHA256 `7cd436039a3014b7cde490c434c038d4ed7aa94f5c2976d1c0172ca1603ee2a5`, 1740 bytes.
- `data/lya.gex`: SHA256 `77aa7daa78b18ab8602f883e3d64ad43006d2c5084db7b3607748516db9b0acd`, 1740 bytes.
- `data/c4.gex`: SHA256 `c691ff3270713663bcc7937c1d942d6d99335bbcea3096a1734c4b04565c873d`, 1740 bytes.

Original repeated runtime 11.929455s; selected original log objective -431.72080871476.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 6 computed PNG outputs, 19 formulas. Private evidence: research_batch10_20261027/acceptance/agn_uv_reverberation_transfer/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.

## Encoding review amendment

Parent report audit identified Windows PowerShell stdin ASCII replacement in explanatory text and units. Restored unambiguous ASCII notation and units in README/plans/notebook before revalidation; equations already expressed in TeX and all numerical code/data/references were unchanged. Pre-amendment acceptance inventory and documents preserved privately in acceptance/agn_uv_reverberation_transfer/encoding_amendment_before. Complete parent gates rerun after repair.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 6 computed PNG outputs, 19 formulas. Private evidence: research_batch10_20261027/acceptance/agn_uv_reverberation_transfer/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Portable notebook kernel metadata

Final packaging replaces the maintainer-only kernelspec name shared-py with the standard python3 name. All code cells, Markdown cells, execution counts, outputs, scientific arrays and non-kernel metadata are unchanged; parent execution still used the explicitly recorded validated shared interpreter. This is a metadata-only portability amendment, not a new numerical execution. The shared environment must be selected when running the notebook. Private before/after hashes and cell equality evidence: acceptance/agn_uv_reverberation_transfer/portable_kernel_amendment.json.
