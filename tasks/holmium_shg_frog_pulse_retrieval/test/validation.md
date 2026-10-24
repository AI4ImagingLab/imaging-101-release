# Validation and provenance — Ho:CALGO Regen SHG-FROG

## Bound scientific task

Recover all512 temporal complex-envelope samples from the full author-processed512×512 SHG-FROG measurement of the Ho:CALGO regenerative-amplifier output after its Martinez compressor. This is a time-domain field phase-retrieval task, explicitly distinguished from a spatial image. The primary acquisition is **FROG_Regen/a.dat**. There is no independent actual field truth. `evaluation/baseline_reference.npz` is the normalized result of the first reproduced original COPRA1000 run with seed1234; it is not relabeled truth. Output unit energy is sum(abs(E)^2)*dt=1, with units s^-1/2. The entire complex array is evaluated; weak tails are never masked.

## Paper, source and rights

Method paper: Geib, Zilk, Pertsch and Eilenberger, Optica6(4),495–505(2019), DOI10.1364/OPTICA.6.000495. Exact actually read version: **arXiv1810.04780v1**, including all20 PDF pages and supplement. Private PDF `geib_copra_paper.pdf`, SHA256 `10d4131dc9c31e1a673c2f0963bfee62aca8ccfdc0c0d3739d6b18da74537880`. Download URL https://arxiv.org/pdf/1810.04780; the PDF itself identifies v1. Full extracted text and rendered Eq11–19 review pages are archived (`geib_copra_paper.txt`, `paper_review-03.png`, `paper_review-04.png`). The method equations, complex gradients, local/global stage transitions, gain projection, comparisons and ambiguity supplement were read rather than inferred from an abstract.

Experimental paper: Suzuki etal., *High peak-power2.1-µm femtosecond Holmium amplifier at100kHz*, exact **arXiv2412.09096v1**, https://arxiv.org/pdf/2412.09096v1. Full text read; Fig4 and the associated pulse/acquisition narrative were visually inspected on rendered page3. PDF SHA256 `1aa7b27d7fbfcfec97e8ca087360592fc04c164c3665a9932155058cce88dbdd`. Its published historical0.25% Regen FROG error and approximately750fs pulse are context, not results of this cleaned run.

Original author repository https://github.com/ncgeib/pypret at exact remotely verified commit **d47deb675640439df7c8b7c08d71f45ecea3c568**. Live API response is `remote_commit.json`; downloaded source archive and per-file inventory are private. This is a verified immutable revision, not a claim that a local branch is current. Source license MIT; full original license/copyright preserved in task `NOTICE`. Core source files are unchanged in the isolated upstream runs. The clean extraction removes framework dispatch, object persistence and domain dependencies, while retaining the mathematical conventions and selected updates.

Real experimental data https://zenodo.org/records/14390876, version1, DOI10.5281/zenodo.14390876, creator Anna Suzuki with named collaborators. Metadata explicitly identifies `a.dat` as measured trace and `arecon.dat`, `Ek.dat`, `Speck.dat` as reconstructed quantities. License metadata **cc-by-4.0**; attribution, link and changes are retained in `NOTICE`. The full response is `real_data_record.json`. The `FROG_Regen.zip` download SHA256 is c89da2a859b8b9dddd53057efc9f26e74be80554a2eacec9b23d16f0275e3820; original member `real_data/FROG_Regen/FROG_Regen/a.dat` SHA256 **a34f9e8ae9618987d13cdfc04a1be2d39b8447a7c312b014f3f4e929a9c61244**. `frog.dat` preprocessing log SHA256 `8ef403ade57d50ab0c0fdca8f1760114bc88d18cb1e8c50a8f4647e4bde032a9`.

The author's log identifies raw266 delay points,4094 wavelength points, approximately30.1fs scan spacing, constant-wavelength input bins, repeated background subtraction, detector-region extraction, centering and wavelength shifting. The distributed a.dat is already processed on its512×512 retrieval grid. It is **not raw CCD data**. Historical field/spectrum reconstructions are never initialization, calibration or truth. The author plotting notebook only reads these files and does not document the speed-of-light conversion; no unsupported approximate-c correction is fitted or asserted.

## Exact observation binding and preserved conventions

Read the512 wavelength coordinates, actual512 delays and512×512 intensity values. Reverse spectral order, transpose into delay/frequency order, and divide by the observed maximum. Preserve every processed value and zero. Fit the printed signal wavelengths converted using c=299792458m/s to an equidistant angular-frequency grid. The fitted step is407474458795.6243rad/s and fundamental carrier903760270468365.0rad/s. Maximum coordinate-print rounding residual is929623809.75rad/s or0.002281428417618363pixel. This is a coordinate interpretation, not interpolation of the trace.

The reciprocal Fourier time increment is30.116847910804345fs. Actual delay increment is30.096fs. Their maximum coordinate difference is5.335065165911059fs. **COPRA and PIE use the actual offgrid delays.** PCGPA alone is labeled a diagnostic on linear delay-regridded observations; its returned field is also evaluated against the original delays. The clean code explicitly guards its exact-grid requirement instead of relying on the original ineffective axis comparison. Source's flat effective spectral-response/instantaneous thin-crystal approximation is retained. Any wavelength-bin density conversion already performed by the author's processing is not independently recoverable; no extra guessed response or Jacobian is applied to change the distributed trace. This limitation and structured residuals are disclosed.

Public NPZ SHA256 `552e86f121d34ff1d44be76e465b97c9437848eaa3f3c9a8b09f903f95856937`; only trace,delay,time,angular_frequency are released. All keys are batch-first and documented in README. No hidden seed, target-derived response, historical reconstructed field, actual truth, or upstream URL appears in solver-visible inputs. This task is distinct from existing spatial coherent diffraction, exoplanet light curves and wavefront tasks: its acquisition is nonlinear delayed self-gating spectroscopy and its unknown is a temporal complex pulse.

## Original runs and frozen independent stages

Oracle executable: private `oracle-env/Scripts/python.exe`, Python3.11.11, NumPy2.4.6, SciPy1.17.1, original pypret Fourier backend `scipy.fftpack`; CPU, OMP/OPENBLAS/MKL threads1. No domain package is installed into the cleaned shared runtime. The private oracle uses the existing numerical libraries and adds only the author checkout to its import path. `clean_environment.json`, original run JSONs and stage-capture JSON identify actual executable/backend/source hashes. The cleaned interpreter's `find_spec('pypret')` is false; full source/CLI/tests/notebook use only local mathematics.

Actual original command sequence: shared Python runs `run_regen_rounding.py`, which invokes isolated Python on `run_regen_seed1234.py`,1235 and1236. Actual command/runtime/exit records are `regen_commands.json`; raw stdout/stderr are `original_regen_seed*.log`. These unchanged-source COPRA runs use1000 iterations, alpha0.25 and an independently specified zero-phase50fs Gaussian. First exact-bound original runtime100.98346669999592s, exit0, no captured warnings, R=0.002252565543858924. No author historical pulse is read. The .003 metadata-only rounding guard is documented: an initial MPC-specific .002 guard rejected Regen's0.002281pixel coordinate rounding before inversion; preserving the failed logs, the diagnostic guard was corrected without altering observations or method.

`run_regen_widths.py` executes original1000-step500fs and1000fs initializations; `regen_width_commands.json` records actual commands, times and exit0. `run_regen_long.py` performs three independent original continuations for5000 additional steps from their respective first1000-step best fields, with restart into the original local/global algorithm. These are explicitly continuations, not silently called cold6000-step trajectories. `command_regen_long_seed*.json` and full logs preserve the runs. Separate original PIE1000 and power-PCGPA1000 runs are indexed by `regen_comparison_commands.json` and JSON/log/NPZ files. PCGPA R on its altered comparison data differs from R on the original actual-delay acquisition, both retained.

`capture_regen_stages.py` runs unchanged author functions on the bound Regen data before full cleaned scoring. Read-only Python tracing captures actual local variables; a permutation-recording wrapper returns the original RNG result unchanged. Frozen fixture SHA256 `ee4abc1ff98fcc8e1052b6ca8f0b9fc2f121acce72eb67a5f072113b23beb453`; environment/source hashes in `regen_original_stages_capture.json`. It includes field/roundtrip, delayed signal/phase masks, projection, gradient, initialization maximum, complete local and forced-global COPRA steps, PIE step, order/beta and scalar step lengths, plus the original one-iteration old-input best-save behavior. Tests never import the oracle. Every fixture file has an explicit parity-test consumer.

Stage and physics criteria were fixed before local inverse extraction in `parity_criteria_frozen.json`: forward/projection relative L2<=1e-11; gradients<=1e-10; updates/scalars<=1e-9; full trace-error absolute difference<=1e-7; independent dense Fourier/Parseval/Gaussian checks<=1e-10; directional gradients<=1e-5. Before selecting Regen, the same unchanged source stages on MPC established extraction parity, then Regen stages froze the exact final binding. Measured local Fourier/SHG/projection/gradient/local/global/PIE stage relative differences were zero for the initial extraction. Regen tests confirm these fixed limits without source imports.

## Equation/function/test traceability

| Required mathematics | Original file/function | Local implementation | Independent/frozen validation |
|---|---|---|---|
| Fourier quadrature and centered phases, supplement discretization | pypret/fourier.py FourierTransform | physics_model.FourierGrid.forward/backward | Dense direct matrix, roundtrip, Parseval; original output_field/output_roundtrip |
| SHG delayed self-product, paper nonlinear signal model | pypret/pnps.py FROG._calculate | SHGFrog.signal/trace | Actual fractional-delay fixtures and independent analytic Gaussian delay/SHG scaling |
| Gain and intensity residual, paper Eq11–13 | retrieval/retriever.py _error_vector/_r/_Rr | scaled_residual; independent Judge trace_error | Original mu/R/r stage fixtures; full pipeline and explicit known cases |
| Intensity projection, Eq14/source convention | BaseRetriever._project | SHGFrog.project | Original projection; independent abs(F(P))^2=D/mu identity |
| Complex signal gradient, supplement SHG derivatives | FROG._gradient | SHGFrog.gradient | Frozen complex gradient, complex-direction finite differences |
| COPRA stochastic local sweep and maximum-gradient step | COPRARetriever._retrieve_begin/_retrieve_step | Retriever initialization/step local | Original permutation, old/current maxima, Z,gamma,target,gradient,new spectrum |
| Global residual gradient and two normalized steps, Eq17–19 | COPRARetriever._retrieve_step global | Retriever.step global | Frozen gradr,etar,target,gZ,Z,etaz and independent residual-gradient finite difference |
| Selection, switch and returned best | StepRetriever._retrieve | Retriever.step/run | Frozen maxiter1 old-input results and original complete1000 trajectory |
| PIE sweep | PIERetriever._retrieve_step | Retriever.step pie | Frozen beta/order/target/update and complete independently rerun PIE |
| PCGPA power iteration | PCGPARetriever._retrieve_step | Retriever.step pcgpa | Original comparison on identical explicitly regridded data; original-delay forward diagnostic |
| Physical phase/time/reversal ambiguities | Paper supplement ambiguity discussion | Standalone Judge align_field | Artificial fractional shift/phase/reversal, independent original seed/width/length ensemble |

Paper/code discrepancies: source switch counter5 vs paper body10; physically correct source projection sqrt(D/mu) vs inconsistent printed v1 gain prefactors; original old-input best-save and approximate local statistic retained; source global logs may lag one iterate. These are explicit in approach and notebook. The ineffective upstream PCGPA grid guard is corrected only to reject invalid comparison inputs, not to modify its update. The primary physical model is unchanged.

## Independent field uncertainty and frozen quality boundaries

The quality boundary was frozen at 2026-10-21T20:14:29.417902+00:00 in `quality_criteria_frozen.json`, SHA256 `6e0e4d95eed0ed5a1d236ea4e630a12c7625880770395e16a1cc61e3cf634839`, before any completed cleaned full-quality scoring. A first freeze-script invocation from the task cwd failed to import a private helper; it produced no thresholds or score. It was immediately rerun successfully from the private archive while the first clean CLI was still computing, before observing its result. No threshold was changed based on the clean output.

For normalized fields, compare all512 complex samples after global phase, continuous periodic translation and optional conjugate time reversal. Relative complex L2 is norm(Ealigned-Eref)/norm(Eref); coherent overlap is abs(vdot(Ealigned,Eref))/(norms), with no mean subtraction. These are explicitly task-specific complex metrics, not mislabeled range-normalized NRMSE. Source paper's peak-normalized pulse error is a different quantity and is not substituted.

| Independent original condition | Complete trace R | Aligned full complex relative L2 | Coherent overlap |
|---|---|---|---|
| regen_seed1235 | 0.002249421218146042 | 0.2610470882868271 | 0.9659272088484849 |
| regen_seed1236 | 0.002247777417441865 | 0.2263205469919756 | 0.9743895050046262 |
| regen_width500 | 0.002248573216378383 | 0.1962012975128845 | 0.9807525254271302 |
| regen_width1000 | 0.00224881063263886 | 0.1702589490211959 | 0.9855059451390991 |
| regen_long_seed1234 | 0.002245291202511506 | 0.1684865256367302 | 0.9858061453394316 |
| regen_long_seed1235 | 0.00224461110653339 | 0.2777881713605502 | 0.9614168659260807 |
| regen_long_seed1236 | 0.00224566293377204 | 0.2480371048695701 | 0.9692387973039606 |

The maximum independent complex discrepancy is0.2777881713605502. Bound1.1times this is **0.30556698849660524**, equivalent unit-energy overlap **0.9533144077705578**. Trace bound **0.0024743633399606463** is1.1times the largest independent nonbaseline source R=0.0022494212181460417. The baseline's own R=0.002252565543858924 also lies safely below this frozen bound. Energy relative error<=1e-6 enforces the explicit output normalization; original normalization roundoff is at machine precision, and no arbitrary magnitude is admitted. These moderate bounds represent method-reference uncertainty; they are not claims of high-precision physical phase truth. The complete field remains evaluated, rather than hiding weak samples.

Poor controls computed before cleaned scoring:750fs Gaussian has complex error0.4070655/R0.00771187; time-phase-erased0.5217161/0.0105393; spectral-phase-erased0.4324344/0.00689628; random phase1.29704/0.0282829; shuffled1.27529/0.0284270; uniform1.21020/0.0283781. All exceed both field and observation boundaries. Exact numbers/definitions are in `regen_poor_controls.json`; the notebook freshly recomputes normalized variants. Zero output and amplitude2x independently fail the Judge. Thus neither a generic pulse nor retaining amplitude while losing phase passes.

Earlier MPC exact-observation attempts, including precise-c offgrid runs and the parent's failed forced-dt attempts, remain private. Three1000+5000-step MPC trajectories retained incompatible basins (R0.00636895/0.00536342/0.00549904; complex differences approximately0.74 relative to the first basin), so that binding was rejected for this full-field task. It is not mixed with Regen results or described as successful. The existing author64×64 noiseless regression observation and independent pulse truth were also preflighted as a fallback; no new synthetic trace was generated and it is not the shipped acquisition.

## Clean execution and checks

Shared executable `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. Exact numerical/backend details are archived in `clean_environment.json`. The CLI sets only numerical thread counts when executed; library imports select no Matplotlib backend. Requirements include only NumPy/SciPy/Matplotlib and no test/Jupyter/oracle packages. Tests and notebook require the shared development tools, not per-task installations.

Actual `python -m src.main` exit0, first observed local retrieval61.69119130000763s and R0.002252565543858924. Its fresh `evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` exit0 measured:

```json
{
  "complex_relative_l2": 1.4507248258626062e-08,
  "coherent_overlap": 0.9999999999999998,
  "energy_relative_error": 0.0,
  "trace_R": 0.0022525655438589364
}
```

`python -m pytest test/ -v --basetemp=<private>/pytest_retry`: **17passed in58.62s**. The initial run had9passed and8environment fixture errors because an existing system `pytest-of-admin` temp directory denied access; its exact log remains `pytest_first.log`. The explicit private base fixed the environment without changing tests or quality limits. After strengthening fixture consumption and hidden-asset checks, the affected stage/contract suite passed16 tests,1full-pipeline test deselected, in2.07s (`pytest_contract_final.log`); the unchanged full solver pipeline had already passed in the17-test run. Malformed missing/key/shape/complex64/NaN/Inf submissions return2; finite zero/amplitude-scaled predictions return1; foreign cwd reference resolution, report creation/write failure and unavailable evaluator assets are tested. Standalone Judge imports no solver or submitted code.

One dense direct-Fourier test initially used an inappropriate fixed2e-26 absolute error despite the frozen criterion being relative1e-10. Dense exponent cancellation measured at about1e-13 relative; the test was corrected to apply the already frozen relative criterion. This did not relax a physical or parity boundary. The failed assertion is retained in session evidence.

## Notebook and final review status

The canonical notebook actively reconstructs the full binding, computes fresh PIE1000 and PCGPA1000 comparisons, and three fresh200-step alpha sensitivities. It loads only separately identified original spectra for independent stability comparisons. Six chapters explain problem/data, physics, method, results, sensitivity and conclusion. The private kernel spec explicitly selects the shared interpreter; the notebook prints actual executable, NumPy/SciPy/Matplotlib/backend and data hash. Timeout2400s is justified for the complete computed methods and plots. Original runs, cleaned runs and historical paper numbers are separately labeled.

An initial full nbconvert attempt was interrupted during extremely slow export of hundreds of thousands of vector heatmap cells; partial figures and log are archived under `archived_vector_export_attempt` and `notebook_vector_export_interrupted.log`. The sole change was rasterizing heatmap cells inside SVG/PDF while preserving vector axes/text and identical numerical arrays. Complete nbconvert was restarted; its final completion, actual outputs, figure inspection and inventory evidence are recorded below after execution finishes.

## Mandatory gate ledger

| Gate | Current result | Evidence |
|---|---|---|
| Paper/full text/exact remote source/data/license binding | passed | PDF/source/data hashes and archive records above |
| Original method on exact final observations | passed | Original3seeds,widths,long runs and exact command logs |
| Independent stages and physical mathematics | passed | Frozen fixtures, analytic/dense/finite-difference checks and17-test run |
| Local full reconstruction and standalone Judge | passed | CLI and measured fresh report |
| Adversarial NPZ/Judge/foreign-cwd/report failures | passed | Unit/contract tests |
| Self-contained public acquisition/output/no leaks | passed | final_static_audit.json errors[]; README plus public NPZ |
| Local mathematics/minimal dependencies/no domain runtime | passed | Explicit source, frozen parity, clean env has no pypret |
| Active full notebook and scientific comparisons | passed | Direct subprocess nbconvert returncode0; all5active cells and six actual figures |
| README/plans mathematical and signature consistency | passed | Source signatures and semantic Fourier/gain/gradient/update/units review; parent formula review |
| Final figures/inventory/archive hashes | passed |18PNG/SVG/PDF, numeric NPZ/JSON, plot_provenance, archive_index and final evidence |


## Final executed notebook, semantic review and inventory

**Passed**: direct shared-interpreter `subprocess.run([...],shell=False)` executed the entire canonical notebook via `python -m nbconvert --to notebook --execute ... --ExecutePreprocessor.timeout=2400 --ExecutePreprocessor.kernel_name=research_runtime`. Actual process returncode **0**, runtime **294.395674900s**. The full exact command, cwd and thread/kernel/export environment are `notebook_final_process.json`; stdout and stderr are separately preserved as `notebook_final.stdout.log` and `notebook_final.stderr.log`. The earlier PowerShell wrapper returned1 despite a complete notebook, because native warning stderr became a shell NativeCommandError; that ambiguity is archived and does not establish this final success. The required successful native exit was obtained in the separate final full execution, not inferred from an artifact. Proactor/ZMQ warnings are retained, not suppressed or represented as scientific errors.

All5code cells completed with execution counts1–5 and no error outputs; all six explanatory chapters remain present. `notebook_completion_evidence.json` stores actual cell timestamps, executable and backend. Actual kernel: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe`, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, backend`module://matplotlib_inline.backend_inline`. Notebook hash **8d7b0b5f88837220400c72470db774f98d13d5c2d1b214b6ac95af7a898b42ab**. The exported executed notebook stays private; canonical notebook retains full outputs.

Fresh primary runtime, complete method metrics and independently identified original sensitivity values are in `computed_measurements.json`. Actual fresh comparison measurements:

```json
{
  "COPRA": {
    "trace_R": 0.002252565543858924,
    "complex_relative_l2": 1.4507248258626062e-08,
    "coherent_overlap": 0.9999999999999998,
    "runtime_seconds": 68.22776809999777,
    "relative_spectrum_parity_error": 0.0
  },
  "PIE": {
    "trace_R": 0.0025876067409720547,
    "complex_relative_l2": 0.23690236172538665,
    "coherent_overlap": 0.9719386355044668,
    "runtime_seconds": 110.17141249999986,
    "relative_spectrum_parity_error": 0.0
  },
  "PCGPA regrid": {
    "trace_R": 0.0025949697892908965,
    "complex_relative_l2": 0.23444217176606694,
    "coherent_overlap": 0.9725184340488048,
    "runtime_seconds": 56.90625140001066,
    "relative_spectrum_parity_error": 0.0
  }
}
```

The final alpha experiment isolates the global stage: all variants start from the **same fresh primary best field**, force global mode, and execute200global/zero local steps. This avoids a vacuous early local-stage comparison, where alpha is inactive. Per-run mode arrays are saved in `plot_numeric_sources.npz`, and each count is asserted during active execution. Actual sensitivity measurements:

```json
{
  "alpha=0.125": {
    "actual_error": 0.0022480512267769315,
    "runtime_seconds": 12.725380099989707,
    "global_iterations": 200,
    "local_iterations": 0,
    "initialization": "Shared freshly reconstructed primary best; force global stage"
  },
  "alpha=0.25": {
    "actual_error": 0.002247363555365459,
    "runtime_seconds": 13.283553999994183,
    "global_iterations": 200,
    "local_iterations": 0,
    "initialization": "Shared freshly reconstructed primary best; force global stage"
  },
  "alpha=0.5": {
    "actual_error": 0.0022490564118470657,
    "runtime_seconds": 12.907094499998493,
    "global_iterations": 200,
    "local_iterations": 0,
    "initialization": "Shared freshly reconstructed primary best; force global stage"
  }
}
```

`notebooks/reference_outputs/upstream_*.npz` are independent original spectra/histories/forward traces; their consumer is the notebook's explicit original-vs-cleaned and independent stability comparison. `fresh_cleaned_measurements.npz` is collected from this final active notebook's saved numerical sources after execution, and separately identifies fresh field/spectra/histories for maintainer review and serialization comparison. It is never an active solver input or substituted for fresh reconstruction. `test/fixtures/original_stages.npz` is consumed by the stage tests; all associated source captures are private. No other unused fixture/reference files are shipped.

All six PNG figures were opened and visually inspected, including all panels, units, method groups, phase uncertainty, full-window tail comparison, source/fresh convergence, poor controls and alpha comparison. SVG/PDF counterparts and complete numeric NPZ/JSON sources are private under `figures/holmium_shg_frog_pulse_retrieval/`; `plot_provenance.json` links all18artifact hashes, numeric array shapes, actual canonical notebook and consumers. Figure1's caption explicitly states **white log pixels are exact recorded zero, not missing data**. Figure3's signed full-residual extrema are -0.10394133534935379 and+0.10040476967421164; its actual norm limits are symmetric±0.10394133534935379, with no clipping or saturation. Weak phase traces are faint only for visualization; all512complex samples remain scored.

Additional independent exact-binding mathematical checks: dense-quadrature finite difference of the original Regen global residual gradient has relative directional error4.4702789681305385e-10 (`regen_independent_global_gradient.json`). Actual complete-field Judge tests after global phase0.73rad, continuous time shift137.37fs and conjugate time reversal all passed; exact metrics are `physical_ambiguity_judge_cases.json`. Small forward changes under a fractional gauge shift reflect the finite discrete product's spectral aliasing; all remain below the independently frozen observation bound. No frequency shift or per-pixel phase freedom is admitted.

Semantic review checked every shipped core signature against `actual_signatures.json`, then independently reviewed Fourier2pi factors and sign, actual-vs-Fourier coordinates, both delayed-product derivatives, gain normalization, local maximum-gradient normalization, global residual gradient/sign/step denominators, original selection/logging state, output energy units and full-complex metric conventions. PCGPA's matrix update is explicitly written as **elementwise conjugation** of C followed by matrix multiplication, not a conjugate-transpose adjoint. The library import backend test preserved `tkagg` before/after importing`src.main`. `final_static_audit.json` reports errors[] for import/call roles, no domain dependencies, public URL/data allowlist, active notebook and inventory. Parent independent review reported17tests passed with actual exit0 in62.38s and94formulas passed both MathJax and KaTeX; final narrative additions do not alter core equations or the inverse method.

All original failure logs, MPC rejected-binding evidence, fallback preflight, papers, source checkout, capture scripts, oracle environment and incomplete vector exports remain outside the final task. `archive_index.json` enumerates private files with byte sizes and SHA256; `final_evidence.json` enumerates task file hashes and actual completion evidence. Task caches are safely moved into private `archived_task_caches` after verified source/destination containment. No existing accepted task was changed. Parent final acceptance remains a separate review.


Parent final pure-documentation clarification: the canonical notebook now explicitly defines N=512 frequency/time samples, M=512 actual delays, time/angular-frequency increments, the fundamental carrier, fitted gain, residual sum of squares, peak-normalized RMS residual and elementwise complex-conjugation notation. Only two Markdown cells changed. All code cells, their execution metadata and outputs, scientific figures/numeric arrays and the captured successful nbconvert process remain unchanged. The executed private archive documents the original numerical execution; this later symbol-definition clarification requires no numerical rerun. Canonical/provenance/completion/inventory hashes were updated consistently.
