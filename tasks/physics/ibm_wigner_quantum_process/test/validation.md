# Validation evidence: IBM Wigner quantum process

## Fixed binding, sources and licenses

Task: physics, auxiliary-qubit spherical interferometry -> unknown single-qubit unitary process matrix. This is distinct from existing bbo_photon_polarization_density and gkp_homodyne_wigner_tomography, whose unknown is a state density matrix/function. No change of sample or algorithm is being used to duplicate an old state-tomography task. Parent conditionally reserved the process binding before local extraction.

Paper: Devra et al., arXiv:2411.05404v2, 18 December2024, https://arxiv.org/html/2411.05404v2; published DOI10.1007/s11128-026-05269-z. Original code: https://github.com/amitQC/Wigner-Unknown-Process-Tomography, exact remote API tree SHA5d0046357ddd3bfb623b7cb028fdc8014f0059c8 verified2026-10-27. Archived API response is wigner_process_tree.json; source is wigner_process_archive/. This is a verified remote revision, not a local tracking branch. Author license LICENSE.xml is LGPL3; the task LICENSE includes its GPL3 incorporated terms, and NOTICE preserves author attribution. Data are supplied within this licensed repository, with no separate restrictive data notice identified.

The selected data are the complete author expec_H.mat and Leb_50.mat, not a simulated example. The MATLAB filenames/comments identify experimental Hadamard measurements. Independent numerical binding: applying the unmodified author initialization to this file yields [0.39368072,0.18679134,0.40349409,0.05813852], which matches all four experimental AppendixF.1 values [0.3937,0.1868,0.4035,0.0581]. The separate acquisition-example notebook defaults to Aer simulation and is explicitly not treated as evidence of measurement provenance. Its data cannot substitute for this experimental file. All8x50complexvalues are bundled losslessly; only shape packing and degree-to-radian geometry conversion occur. The full50weights sum to1.000000000000002. No actual physical truth is available. The ideal target Hadamard in author plotting code is neither truth nor input to the inverse algorithm.

The scalar0.00995 is an approximate analytic basis normalization, close to1/(32pi); it is not obtained from the unknown target. Tests prove global positive amplitude scaling leaves the normalized primary output unchanged. The solver accesses only measured observations and independent angular quadrature. No target gate, raw-shot fabrication, target-derived error correction or reconstructed reference appears in public arrays.

## Original execution and compatibility

Archive root is the maintainer-private directory research_batch11_20261027/astro_earth_physics, outside the released task. wigner_oracle/capture_iterative.m extracts the original numerical script blocks while omitting display/plot-only sections; original helper functions are executed unchanged. It retains the author target solely in unused plot/fidelity context, with no dependency from that target into any reconstructed quantity. Command: octave-cli.exe --no-gui --quiet wigner_oracle/capture_iterative.m. Exit0,1.78757s; wigner_oracle/iterative_original.log and iterative_original.mat archive all stages. Deterministic CPU, no random seed used. Original MATLAB helper functions and source hashes remain archived.

Original optimized comparison: capture_optimized.m, optimizer_X.m and optimoptions.m in wigner_oracle/. Octave10.3.0 lacks MATLAB optimoptions and supplies column callback vectors. The private compatibility adapter maps MaxFunEvals and options, transposes callback vectors into stored rows, and explicitly warns that Octave chooses its own finite-difference step rather than MATLAB's requested1e-8. This is a documented optimizer/environment adaptation, not an unmodified MATLAB trajectory claim. The initial attempt failed on callback shape; optimized_original.log retains that failure. Corrected compatible run exit0,0.22427s; optimized_original_adapted.log and optimized_original.mat retain outputs. The paper objective, measured data, coefficient normalization and initialization are unchanged. This comparison is not the primary benchmark method.

The original sign routine emits a fallback warning even when ideal branches exist, because its final largest-weight branch always executes. We preserve its effective behavior, documenting this code discrepancy instead of silently repairing it. Original optimizer uses the coefficient-vector norm, despite a Frobenius-norm comment, and a non-unit initial quaternion normalization; both formulas are preserved and tested. The historical paper optimized quaternion differs from this current author-code run. We do not claim a reproduction of the historical optimized numbers; the primary iterative method is validated against its actual original run and the optimization comparison against its actual original objective.

## Freeze and stage traceability

Before writing local implementation, wigner_frozen_acceptance.json froze all primary numerical tolerances, physical tolerances, objective-minimum tolerance and Judge boundaries. Frozen timestamp: 2026-10-27T05:56:34.374575+00:00. Droplets/basis/correlation absolute1e-12; quaternion history/matrix1e-10; independent harmonic/forward identities1e-12; directional objective derivative1e-6; optimized objective minimum agreement1e-7. Judge phase-aligned RMS and unitarity boundaries1e-6 permit complex64 representation error while remaining restrictive. No threshold was relaxed. The1e-5 matrix proximity check for the optional optimized comparison is supplementary; its mandatory frozen criterion is objective agreement, since trajectories differ.

| Paper stage | Original source | Local callable | Validation |
| --- | --- | --- | --- |
| Auxiliary acquisition, rank0/1 signals | paper sectionsIII-IV; Qiskit acquisition expectation formulas | physics_model.ideal_expectation | arbitrary non-target quaternion acquisition-to-projection identity |
| Scaled droplets, Eq27 | both main MATLAB files r1/r2/r3 | preprocessing.droplets | frozen all400complexmeasurements stage |
| Spherical basis | BasisGen_Lebedev, harmonicY | physics_model.basis | upstream all basis values; independent weighted orthogonality |
| Weighted overlaps | Overlap_Lebedev/New | overlap_coefficients; reconstruct correlation | original C, basis, coefficient-derived channel matrices |
| Relative signs/initialization | Quat_sign_predictor_New; main | correlation_signs; optimization_inputs | all four dominant-index cases; original initial Euler angles |
| Iterative weighted combination | Recons_DropletBased_Experimental_Main steps1-7 | reconstruct | every quaternion iterate and final matrix |
| Euler objective Eq31 | cost_fun_new; optimizer_X | euler_quaternion; objective; optimize_comparison | every original objective sample, minimum, independent directional derivative |
| Metrics | independently declared evaluation convention | independent Judge; visualization.process_metrics | analytic zero/scaled/phase matrix cases, malformed/poor contracts |

## Measured outcomes and reference identities

Fresh primary versus original: maximum complex matrix difference6.664589960562512e-16; quaternion trajectory max1.1102230246251565e-15; droplets6.99e-18; basis2.78e-17; correlation3.25e-19. Fresh optimized objective minimum differs from original by-1.7763568394002505e-15. Fresh BFGS terminates with precision-loss diagnostic; its warning is retained, and objective parity is the supported conclusion. It is not mislabeled as a successful-gradient termination. All mandatory primary stages pass frozen tolerances.

Judge fresh metrics: {"phase_aligned_process_rmse": 6.6629755435415765e-16, "unitarity_error": 3.1411987038228775e-16}. The hidden baseline_reference.npz is the reproduced original iterative process, not actual truth. test/fixtures/upstream.npz stores independent original droplets, basis, correlations, full quaternion history, primary matrix, normalized comparison channels, initial angles, original optimization trajectory/objectives and comparison matrix; consumers are parity tests. notebooks/reference_outputs/upstream.npz stores the two original methods and their histories, consumed by active notebook figures. fresh_measurements.npz stores only fresh local primary/comparison/ablation measurements from the executed notebook for audit; no solver reads it. Paper values are explicitly historical. No reconstruction is described as ground truth.

Notebook ablations relative to original primary: optimized0.07489840452, scalar rank removed0.0372892021, vector rank removed0.9635790585, alternating directions0.06010449228 (global-phase-aligned complex-entry RMS). These differences are informative method/sampling comparisons, not hidden truth errors. Ideal forward residual shows attenuation; diagnostic fitted gain about0.293+0.003i is fitted to visible measurements only, never used as calibration or input.

## Runtime, interfaces and review

Clean environment: shared research-runtime Python, NumPy/SciPy/Matplotlib pinned in requirements; full versions and numerical backend in wigner_clean_environment.txt. Original environment in wigner_octave_environment.txt. CPU-only, OMP/OPENBLAS/MKL threads1. No upstream/domain package appears in cleaned imports. Local functions explicitly evaluate harmonic/Pauli algebra, correlations, projections, sign rule, normalized inverse updates and the task-specific objective. SciPy minimize is used only for the documented general-optimizer comparison. No opaque domain reconstruction shortcut or cross-task import occurs.

Commands and logs: wigner_run_manifest.json; wigner_validate_0.log CLI exit0,0.424s; wigner_validate_1.log Judge exit0,0.123s; wigner_validate_2.log20tests pass0.94s; wigner_validate_3.log initial successful active notebook. Final notebook re-executed after adding original optimized panels and phase legends: wigner_notebook_final.log exit0,3.665s,timeout120s. Final canonical notebook contains executed outputs; temporary executed notebook and PNG inspections are private. Initial pytest attempt hit an unrelated inaccessible default TEMP directory; setting private TEMP/TMP resolved all10setup errors without code/numerical changes. Jupyter emits Windows event-loop/TCP transport warnings; no notebook cell errors.

Seven figures were visually inspected: all measured channels, four-method complex matrices, primary/optimized convergence, fresh rank/sampling ablations, ideal/gain-diagnostic residuals, primary and optimized spherical process surfaces. Added phase colorbars and re-inspected final panels2,6,7. Labels, axis units, color ranges and provenance are visible. No missing method is plotted as zero or archived result described as fresh. Notebook contains six active scientific sections. All17extracted formulas pass MathJax3.2.2 and KaTeX0.16.22 (wigner_math_render_check.json); no control/replacement characters. README uses math fences; notebook uses display-dollar math.

Semantic review checked the quaternion signs, basis factors, squared-norm projection denominator, full50point quadrature, ten normalized estimates/nine updates, complex matrix convention, original normalization quirks, all callable signatures and the main/notebook execution paths against both plans. Public README/raw data/requirements suffice without hidden method/reference assets. Every public NPZ key has its shape/dtype/units table. No public GitHub link, hidden gate, truth, reconstruction, seed or metadata file leaks. Judge is independent, uses allow_pickle=False, validates complex dtype/shape/finite values, resolves hidden files relative to itself, and handles poor/malformed/evaluator errors with0/1/2exitcodes. All required output entries are scored; there is only one required matrix quantity.

## Completion gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Paper/revision/data/equation mapping | passed | source/hash/archive index and stage table |
| Original and local exact-observation runs | passed | all50directions, both ranks, all4controls; archived logs |
| Upstream stage parity and independent physics | passed | frozen tolerances;20tests |
| Units/constants/coordinate conventions | passed | README, physical identities, full basis checks |
| CLI and standalone Judge | passed | fresh output/report; independent cwd test |
| Invalid/poor/evaluator contracts | passed | parametrized malformed input and unavailable-reference tests |
| Public input sufficient/no leak | passed | input keys and import/call-role review |
| Minimal environment/explicit local math | passed | requirements/backend and source audit |
| Active notebook/full comparisons | passed | final execution, seven inspected figures |
| Historical/upstream/fresh distinctions | passed | explicit provenance and comparison limitations above |
| Plans/signatures/math consistency | passed | semantic review and render audit |
| Inventory/license/temp cleanup | passed | canonical task layout and private inventory |

## Private archive hashes

- `wigner_paper.html` SHA256 `f7230bbce7517f38194c423d1fe1d2adfe4590c52d818398511c8fdf4deaaf7d`
- `wigner_process_tree.json` SHA256 `393cd35ca9bad29aecc52b298cb6e5ae3e2457ba26201a2e988e2946a77b1aee`
- `wigner_frozen_acceptance.json` SHA256 `3347211a054a17f8e0078d7e3edfa9ef92b5b394a874a205b49afed25a934996`
- `wigner_oracle/iterative_original.mat` SHA256 `9caea081801ccc61898986d0787c3c4ee2efb7edcda49716031c1b7392b1532d`
- `wigner_oracle/optimized_original.mat` SHA256 `15d019e8df04569ee41d9618e42da0c9c3b3e313132bfc2b15988d5e1f9c1376`
- `wigner_process_archive/Reconstruction_algorithm/Without optimization/expec_H.mat` SHA256 `73a088b8c169d5a62cdccc9b0123f22017eda6556a231ca94efd9cdc9b1736b5`
- `wigner_process_archive/Reconstruction_algorithm/Without optimization/Leb_50.mat` SHA256 `9e0798bfbda11bb7cf5ccb5f2dcd6323c8617b54e1e158092ed7f340dfbe0bd5`
- `staging/ibm_wigner_quantum_process/data/raw_data.npz` SHA256 `8830616f29504c858cdc05aba6c25b9486bb9018f170fb198734a62c0751ed09`
- `staging/ibm_wigner_quantum_process/test/fixtures/upstream.npz` SHA256 `8e489ef4c28801058c58d1929351cd6d408c7e5c903734aac90976d66979fb22`
- `staging/ibm_wigner_quantum_process/evaluation/baseline_reference.npz` SHA256 `8e2476d0570f72bd8378afbd1accb82f19b52d0ceec5efa641695f434b4d33ee`

Final expanded contract check: wigner_tests_final.log,20tests passed, including independent visualization/Judge metric identity and report-write failure.

Judge reference precedence: if ground_truth.npz exists beside the Judge it is selected; otherwise baseline_reference.npz is used. Invalid higher-priority truth is an evaluator error, never a silent fallback. The shipped task has no actual truth and only the validated baseline. A temporary independent contract fixture tests all three cases.

Parent-review correction verified:21 tests pass including truth priority, absent-truth baseline fallback, and malformed-priority-asset rejection; wigner_truth_precedence_tests.log. No numerical method or threshold changed.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 7 computed PNG outputs, 17 formulas. Private evidence: research_batch11_20261027/acceptance/ibm_wigner_quantum_process/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Current benchmark contract v2 (2026-10-29)

The v2 evaluator independently forms all eight complex expected channels on the 50-direction Lebedev grid, with the public quadrature weights. Weighted mean squared observable error may exceed the attainable fit by at most 1e-6; the original unitarity boundary 1e-6 remains. Global unitary phase is a gauge. An independently optimized quaternion with lower real-data loss passes; a wrong Pauli operation fails. The conditional ideal-unitary fit is not independent process truth. The cleaned reconstruction algorithm is unchanged.

Earlier method-identity metrics are historical v1 reports, superseded for current benchmark acceptance. Full command stdout, native exits, controls and immutable old assets are indexed in the external benchmark_readiness_20261029 maintainer archive.
