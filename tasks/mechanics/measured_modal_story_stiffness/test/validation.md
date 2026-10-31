# Binding and provenance

Paper Otsuki,Lander,Dong,Wang2022 Advances in Structural Engineering25(4)698–715, DOI10.1177/13694332211022066. Author repository https://github.com/ywang-structures/Structural-Model-Updating, remotely verified revision754b5b7413188164f108226981d5d0f2afbdedca. GPL3 retained in LICENSE. Private archive root research_batch11_20261027/material_mechanics/archive contains SMU-tree.json, SMU.zip (17,015,040bytes), selective exact source in SMU-short, and full author PDF supplied in repository. Size inspected before download. Default PowerShell Expand-Archive failed due long Windows paths; Python extraction into SMU-short preserved bytes. No source mathematical file was modified.

Bound dataset: Examples/FourStoryStructure/Experiment/FourStoryStructure.mat, SHA256e65e640781d138517b277663d318819e04726ced4d2c9bd3114aff322cff688d. Classification author_real_processed. Author README explicitly states laboratory four-story shear frame, shake-table base excitation, accelerometers at base/floors1–3, weighed floor masses and first2 extracted modes used for updating. The selected original script loads this exact MAT file and selects these rows/modes. Other stored modes/fourth row are not bundled, scored or relabeled independent truth. Raw acceleration histories are unavailable in this example. Paper establishes method; the repository establishes this experiment binding. No simulated eighteen-story data are used.

Novelty: measured partial eigenmodes→story stiffness and unobserved modal amplitudes. Existing experimental_modal_identification is FRF→poles/residues; sparse_acceleration_virtual_response uses known stiffness→motion; plane_stress_elasticity_tomography is static full-field displacement→element moduli. Parent reserved this distinct binding before extraction.

# Original execution and compatibility

Private prepare_smu.py creates smu_capture/driver.m and two environment adapters. Original mathematical SMU functions are unchanged. Adapter version.m returns2015b only for MATLAB-release compatibility selection; optimoptions.m translates options to Octave optimset, maps chosen supported lsqnonlin branch to lm_svd_feasible, disables progress display. This is a documented Octave backend port, not a claim to run unavailable MATLAB fmincon interior-point. Paper explicitly supports lsqnonlin and recommends bounded least squares for MDR/eigenvector formulations. No shared package was installed/upgraded.

Command: octave-cli.exe --quiet archive/smu_capture/driver.m (Octave10.3.0, optim1.6.2, Windows64 CPU, single-thread OMP/OpenBLAS). Original full3 formulations×12 fixed randomized starts executed twice,72.620702s/69.847796s, exit0. Exact numerical repeat maximum difference0. Logs smu_oracle.log and smu_oracle_repeat.log; original arrays capture_first.mat,capture.mat; frozen converted upstream.npz. Compatibility warnings report function shadowing and original N=size(M0) colon-vector use; they do not suppress failing solves. Original MAC convergence caps and two bad boundary basins are retained with actual exit statuses. Source includes its analytical objective/Jacobian/eigensensitivity and bound-handling pipeline. No cached original results were used as the oracle.

freeze_smu.py saved smu_frozen_acceptance.json BEFORE build_smu.py wrote local mathematics. Stage atol1e−9,rtol1e−8. Primary output max relative1e−5 and residual RMS≤.1613221901832689 are fixed. Best original MDR objective .17206511765637536, maximum gradient magnitude4.37e−10. Local primary output error<9e−12 relative; all original initial residuals/Jacobians agree within frozen stage tolerances. Nonconverged MAC original minima may be improved by the fresh generic optimizer; no numerical equality of different bad basins is claimed. Paper1,2,3 comparisons are preserved and their objective scales labeled separately.

# Stage traceability

| Theory/stage | Original function | Local function | Validation |
|---|---|---|---|
| Affine shear stiffness, Eq1 | FourStoryStructureExperiment/assemblyK | influence_matrices/stiffness_matrix | spring energy identity, frozen residuals |
| Modal observation normalization | StructModelUpdating | normalize_modes | exact raw observations and signed unit maximum |
| Dynamic residual, Eq6 | ObjFuncMDRLsqnonlin | residual_jacobian(form3) | all12 original initial/final-best cases |
| Dynamic Jacobian, Eqs25–26 | JacMDRLsqnonlin | residual_jacobian(form3) | original matrices and finite differences |
| MAC/eigenvector objectives, Eqs3,5 | ObjFuncMPDLsqnonlin | residual_jacobian(form1/2) | all original initial cases and best final |
| Eigenvalue/shape sensitivities | JacMPDLsqnonlin | reduced eigenvector derivative solve | original Jacobians, independent finite differences |
| Bounded multiple-start inverse | StructModelUpdating/lsqnonlin | solve/least_squares | best MDR quantities/objective; active all3 comparison notebook |

# Arrays and consumers

data/raw_data.npz contains only original first2 frequencies and first3 rows of those modes, all float64. Acquired masses and unit conversion are visible README constants. No raw-data truth, source-dependent seed/object, hidden answer or GitHub URL. evaluation/baseline_reference.npz holds original selected best MDR stiffness(1,4) and latent_modes under key unmeasured_modes(1,1,2); no physical truth exists. Notebook upstream_measurements stores these plus all36 original solution/objective/exit arrays for comparison. test/fixtures/upstream_stages.npz stores all original initial states, initial/final residuals/Jacobians, optimizer solutions/objectives/status and selected raw arrays for parity tests. Fixtures retain native algorithm stage shapes, not the public batch interface. Fresh notebook measurement arrays are separately identified outputs, never used as truth or initialization.

Source runtime is Python3.11.11, NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1 on Windows CPU. Permitted primitives only: array operations, generalized eigh, linear solve and generic least_squares with local analytic mathematics. No domain library/cross-task/upstream imports. Original fixed starts are source hyperparameters, generated before seeing cleaned outputs. Explicit relative bounds are meaningful for nonzero stiffness/latent reference values. Judge independently assembles physical residuals from public observations and honors actual-truth precedence if later supplied. Float16/complex/integer/shape/key/NaN/Inf and missing-assets contracts tested.

# Verification to date

15 tests pass: exact observation/stage parity, all3 analytical Jacobians, spring energy, eigen-equilibrium, normalization, metrics, malformed and poor submissions, GT precedence, foreign cwd CLI and report-write failure. Full primary reconstruction agrees at8.86e−12 stiffness and6.07e−12 latent relative error. Final active notebook and visual review evidence appended after execution. No threshold has been relaxed.

# Final active notebook and gate audit

Native `python -m nbconvert --to notebook --execute notebooks/measured_modal_story_stiffness.ipynb --output measured_modal_story_stiffness_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=imaging-batch11-material --ExecutePreprocessor.timeout=180` completed exit0 in4.27s. Private execution kernel uses shared Python; after execution only kernelspec metadata was restored to portable python3 and code-cell sources were checked unchanged. Windows ZMQ selector-thread/local TCP warnings did not prevent execution. Canonical notebook retains outputs; temporary notebook removed. Five PNGs archived in smu_figures were individually inspected: physical schematic/observations; paired original/fresh stiffness; all36 original/fresh optimization outcomes; three formulations' stiffness/frequency/mode agreement; modal residuals, Jacobian SVD and20 explicitly synthetic1% robustness perturbations. The form2 objective axis was widened and notebook rerun to avoid magnifying machine-precision differences. No computed methods were replaced by cached results. All formulas and physical scales semantically checked against source and README; no forbidden domain calls or cross-task imports.

Fresh reference_outputs/fresh_measurements.npz contains submission arrays plus runtime_seconds(1), mdr_residual(1,8), jacobian_singular_values(1,6), perturbation_stiffness(1,20,4), and objective_form1/2/3(1,12), explicitly identified audit outputs. It is not consumed by reconstruction or used as truth. Original comparison values, fresh values and supplementary perturbations are labeled separately.

Final main, independent Judge and full15-test suite pass; logs archive/smu_final_check_0..2.log. Judge measures parameter relative error8.853362487570848e−12 and RMS equilibrium residual matching the frozen original best within rounding. Original approximate MAC best1.6045850998600367e−7 improves to fresh1.6044187837741494e−7; this is a recorded finite-iteration/backend difference, not a threshold adjustment. Original bad MDR basin remains visible while fresh runs converge to the same selected best.

All mandatory gates PASS: exact paper/revision/data and mappings; original exact observations and repeated capture; independent parity/physics with prefrozen tolerances; constants/units; main/Judge; malformed/poor/evaluator error contracts; visible-input sufficiency; no privileged data/source URL leakage; README metadata/complete output; shared CPU dependencies; explicit local mathematics/call roles;15 tests; active six-story notebook and visual review; distinct original/fresh provenance; plan/signature/equation consistency; standard inventory/GPL notice. No physical GT is claimed. Archive index is smu_archive_index.json. Parent independent acceptance remains a separate review step.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 5 computed PNG outputs, 9 formulas. Private evidence: research_batch11_20261027/acceptance/measured_modal_story_stiffness/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
