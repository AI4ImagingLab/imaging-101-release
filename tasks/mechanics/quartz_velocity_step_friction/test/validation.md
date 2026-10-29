# Provenance and validation

## Binding and source
Selected paper Kim/Lapusta/Faulkner/Bedford, GJI247(2026)ggag362, DOI10.1093/gji/ggag362. Original author data/code Zenodo10.5281/zenodo.21401458, exact versioned record21401458, CC-BY4.0; this is a Zenodo original source rather than a GitHub revision. Verified public record and each downloaded file MD5/size are archived in friction_zenodo_record.json. Private archive is the sibling material_mechanics/archive directory outside the task. Full wet/dry CSV, notebook, metadata PDF and Julia files remain private in quartz_friction. Only the914 bound wet measurements enter raw. All source/data/code SHA256 and incident history are indexed by friction_archive_index.json; raw hash and frozen capture hashes are in friction_vector_frozen_criteria.json.

The author500-point drift calculation is applied on1.1..2.5mm and the complete1.49..1.7mm window is retained. No target model is used to generate observations. Manual JumpIdxs are absent upstream: bound full window and independent loadpoint transition24.824239s, snapped to24.8s, are explicit adaptations. CSV and manuscript show10Hz; the supplied metadata PDF's1Hz statement is inconsistent. M is mass/contact area, not total mass; stiffness90.5MPa/mm becomes90.5e9Pa/m. Effective40MPa is60MPa confinement minus20MPa pore pressure. Pressure-zero source state is a deviation. The source damping helper interprets the numerical velocity argument as damping fraction and evaluates before the later mass assignment; the actual resulting C is preserved and disclosed as a model coefficient. This is not independently measured damping.

## Source discrepancy and independent correction
All supplied localMinimizer variants return scalar chi-square to scipy least_squares. If r is the914-vector and s=rᵀr, the resulting objective is .5s²; its residual Jacobian is2rᵀJ and GN matrix4JᵀrrᵀJ has rank<=1. The manuscript NLS objective and two-parameter covariance require the full residual vector. This independently verified mathematical discrepancy is explicitly corrected; see friction_residual_vector_source.diff. This is a selected corrected/extended method, never unchanged source reproduction. Physical equations, all914 observations, analytic initial estimate, bounds and default TRF settings remain unchanged. Prior scalar-interface originals and failed local Radau/difference-step attempts are preserved in friction_scalar_source_error and separately named friction_*diagnosis logs, marked superseded/failed. No former failure is relabeled pass and no gate is relaxed.

## Independent original execution and pre-freeze
friction_vector_original_execution.json records original Julia full physics through the private interop bridge, Python SciPy standard residual-vector NLS, actual command/environment/native Julia exit0, runtime103.469s,10nfev/30residual calls. Independent new-process repeat friction_vector_repeat_execution.json took101.285s, native0, bit-identical arrays. Both fit chi-square4616.358747591697 and RMS.0008989523440455936; a=.0113103176736835 and Dc=3.0125351907023342e-6m. Full original native state trajectory is friction_vector_original_states.txt. No actual true parameters exist.

Julia1.10.10 runs --compiled-modules=no with private Project/Manifest and depot archive/jd (short U: alias points only to this archive). Precompilation/PyCall failures are retained; a persistent native Julia stdin bridge runs the original springSliderModel function bodies and SciPy supplies the optimizer, avoiding the unused PyCall integration. The import adapter changes package facade/imports only. The manuscript mentions Rodas4 while released source selects Rosenbrock23; this task selects and extracts actual released Rosenbrock23, not a silent claim of manuscript backend identity. Exact OrdinaryDiffEqRosenbrock2.7.5/Core versions and git-tree hashes are in the private Manifest. Their MIT source/notice is preserved.

friction_vector_original_jacobian.npz records residual Jacobian singular values964.8664892354 and234.6719610021, rank2. Independent central step1e-4 check differs by.0015541 relatively. Covariance inv(JᵀJ)*chiSquare/(914-2) has positive eigenvalues5.43713688e-6 and9.19141358e-5; it is conditional local NLS uncertainty, not a Bayesian posterior or true error. Frozen original arrays and nonzero gates were captured BEFORE the local residual-interface correction in friction_vector_frozen_original.npz and friction_vector_frozen_criteria.json. Independent repeats are identical, but original ODE/interpolation numerical error and physically meaningful precision motivate nonzero gates. Limits remain a/b2e-4, Dc5e-8m, theta.05s, velocity relativeL2.005, friction max1e-5 and observed RMS1.1*original. These are all required, none vacuous.

## Traceability and independent mathematics
| Stage/equation | Original role | Local role | Validation |
|---|---|---|---|
| drift and steady portions | DetrendedData notebook / localMinimizer | prepare, initial_guess | raw identity, frozen ba/mu0/initial |
| spring/mass/regularized contact law | springSliderModel.jl ODE | derivatives, regularized_friction | source initial/full forward; independent units/steady equilibrium |
| slip state and theta transform | springSliderModel.jl | state_rhs, trajectory | positive trajectory, known steady state, transform checks |
| ODE Jacobian | source automatic differentiation | state_jacobian | independent centered state perturbations |
| Rosenbrock23 stages | rosenbrock_perform_step.jl | ode_integrator.step | autonomous and nonautonomous original one-step cases |
| Hairer initial step / PI control | initdt.jl, controllers.jl, alg_utils.jl | ode_integrator.integrate | full original accepted count and observation forward |
| displacement interpolation | original LinearInterpolation | trajectory | original914 outputs; initial/full parity |
| corrected NLS | original driver residual interface diff | residual_vector,reconstruct | independent original repeat/full inverse gates |
| metrics/contracts | independent definitions | standalone judge | known numeric cases, poor/malformed/evaluator inputs |

Generic autonomous and explicitly time-dependent one-step fixtures come from full native original Rosenbrock23 fixed .01 step, captured by capture_rosenbrock_cases.jl with command/exit/runtime in rosenbrock_generic_execution.json. Their sole packaged consumer is test_rosenbrock.py. The full local adaptive forward at original independently fitted parameters gives maximum friction discrepancy1.38e-10 and14680 accepted/29 rejected steps; first step3.12379e-5s. Default parameter Jacobian relative difference.001186 demonstrates the critical numerical fidelity without changing optimizer finite differences. Earlier Radau tiny-difference Jacobians were dominated by interpolation mesh noise and failed: those diagnostics are retained, not used as fixtures.

## Runtime, interfaces and notebook
Shared CPU Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, one BLAS/OpenMP thread. Requirements list only used shared packages. No Julia/SciML/domain runtime library, original import, hidden initializer, cross-task import or fitted calibration occurs in source/tests/notebook/Judge. Call/import roles are archived in friction_import_call_role_audit.json. Plans were semantically reviewed against executable force/state signs, scales, all signatures, controller defaults, residual-vector interface and actual paths.

The handoff manifest and friction_final_execution_progress.json are the authoritative fresh execution evidence: independent CLI, standalone Judge, full pytest and complete active nbconvert notebook commands/native exits/runtimes/log hashes. The original, CLI and notebook arrays are separately identified by SHA256. Notebook uses fresh full914 inversion, original comparisons, real residuals, state/speed, parameter comparisons, fresh objective grid and apparatus sensitivity. It explicitly omits full MCMC/dry/aging refits from this selected NLS component, and preserves relevant state-law interpretation with an honest derivative comparison. Fresh figures must be visually inspected and marked passed in the manifest before worker completion.

## Gate status
Paper/source/data binding, original corrected full+repeat, independent frozen fixtures, physical checks, local executable inverse, original parity, standalone Judge contracts, public allowlist/no truth or GitHub URL leaks, minimal dependency/call-role audit, plans/interfaces and template inventory are implemented. Final CLI/Judge/tests/notebook/visual gate statuses are recorded from native execution in the handoff manifest; any nonzero or pending result prevents completion. No source PDFs, downloads, complete checkouts, duplicate report trees or temporary executed notebooks are in the task. Canonical notebook is the executed active review deliverable. No ground_truth.npz is invented; baseline_reference is labeled a corrected original-method inference.

The first final notebook attempt completed fresh inversion and all seven metrics, then failed in the new covariance comparison because its original NPZ key was written as jacobian instead of optimizer_jacobian. This display/input-key error and its native failure log/progress remain archived as friction_covariance_key_failure_*. The key was corrected and the entire notebook rerun from the beginning. Existing passed CLI and18-test native executions were retained because no runtime mathematics changed; the resumed finalizer executes only the complete notebook and preserves the three original command records. This is not a cached reconstruction preview or a claim that the failed attempt passed.


## Final measured gate results
- cli: passed, native exit0, 63.360357200028375s; friction_final_cli.log
- judge: passed, native exit0, 0.13692120008636266s; friction_final_judge.log
- tests: passed, native exit0, 65.86995490000118s; friction_final_tests.log
- notebook: passed, native exit0, 250.2294104999164s; friction_final_notebook.log

All notebook figures were viewed and passed scientific/label/legibility review. The fresh complete notebook and CLI independently pass all seven unchanged metrics. Unit/parity/physics/Judge/pipeline tests passed; full counts are in the native pytest log. Five-package static inventory, batch-first raw and public GitHub-URL audit passed (five_package_static_audit.json). All mandatory worker gates are passed; independent parent acceptance remains pending.


Final parent review passed: source/equations/units/actualcallerflow/GT roles independently checked; every actual figure viewed; dual math engines passed. Current tests 20, actual figure groups 6. Confirmedsource scalar chi-square-as-residual error, explicit correction to paper standardNLS; fullcorrectedoriginal Julia inverse and independentrepeat executed before localinterfacefreeze. Original14,680accepted/29rejectedODEsteps and autonomous/nonautonomous independentstep fixtures supportnumericalextraction. All originalstrictfinalbounds unchanged. Parent verified everyworker source/prediction/native-log/executedNB hash and actualkernelargv (Windowspathnormalization), reused actual63.36sCLI/250.23scompleteactiveNB honestly, rerancomplete tests. Added positivephysicaloutput and finiteoverflowmetricguards with2genuinecontractcases; all20currenttests rerun andindependentJudgepassed. Core solver/notebook numericalpathunchanged, noextraCLI/NBrerunclaimed.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

Parent independently reran all tests, standalone Judge, structural and both mathematical rendering checks. Full bound-data CLI and a separate complete active notebook reconstruction were verified from preserved native exit/runtime, kernel and scientific artifact identities; another complete reconstruction is not claimed. Notebook execution origin: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\material_mechanics\archive\friction_native_handoff.ipynb Reviewed artifact: 6 executed code cells, 6 actual PNG outputs, 5 formulas. Private evidence: research_batch12_20261029/acceptance/quartz_velocity_step_friction/: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch12_20261029\material_mechanics\friction_handoff_manifest.json. All final panels were inspected by parent. Canonical notebook preserves all executed code, counts and actual scientific outputs from the complete private shared-runtime nbconvert execution. No archive is substituted for a fresh fit.


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- 10.5281/zenodo.21401458: https://doi.org/10.5281/zenodo.21401458; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
