# Provenance and validation: BBO photon polarization density

Status: locally validated; pending coordinating maintainer acceptance. Domain physics. Bound physical inverse problem: two-photon complex polarization density and count scale from36 corrected analyzer counts. Nearest old task gkp_homodyne_wigner_tomography reconstructs a continuous-variable Wigner field from homodyne quadratures; this task uses discrete polarization projectors/four-detector coincidences and a4x4 density matrix. This is a distinct acquisition and unknown representation, not a changed object or regularizer.

## Sources and real-data evidence

Primary chapter: J.B.Altepeter,D.F.V.James,P.G.Kwiat, Quantum State Tomography,2004 author manuscript,33pages, https://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf . Primary PDF inspected through web on2026-10-27, pp29–30 including Figure7. The caption explicitly identifies experimental observations, accidental correction and detector-efficiency normalization. The apparatus labels351nm pump,702nm downconversion,10nm filter,two BBO crystals,QWP/HWP/PBS analyzers. All36 values independently transcribed and checked against Fig7b (including HH40230,HV150,VH220,VV39094); table orientation agrees with Eq6 circular states. Data classification **author_real_processed**. No raw acquisition timestamps, integration durations or correction covariance were published with this table; no such quantities are fabricated. Numerical facts transcribed with attribution; no figure or PDF redistributed inside task.

Original code: https://github.com/KwiatLab/Quantum-Tomography , remote API verified revision1760fbcfbb644037ff20297e2e6a273626e179e9. MIT copyright2020 University of Illinois Board of Trustees retained in LICENSE and adapted source notices. This is the original laboratory's modern Python implementation of its published method, not an assertion that the2004 run used this Python revision. Initial discovery through thresholdqst's experimental example led to the primary chapter; thresholdqst was not copied because no license was found. Its synthetic ideal state is not used as truth.

Archive index relative to batch worker directory: archive/qkw/ contains original modules, README, pyproject and LICENSE at the verified commit; qkw_tree.json pins remote file metadata. archive/capture_quantum.py executes original tomography_MLE without cleaned imports. archive/quantum_oracle.npz contains exact bound counts/projectors plus initial coordinates, initial residual, probe density, final rho, scale and objective. archive/quantum_preimplementation_protocol.md records initial parity/physics tolerances before extraction. archive/quantum_independent_reruns.npz contains independent optimizer initial-state sensitivity. archive/quantum_run.json records environment/runtime; archive/quantum_commands.json records fresh command timing/exits. Logs: archive/oracle_rerun_quantum.log, cli_quantum.log, judge_quantum.log, pytest_quantum.log, notebook_quantum.log. Private review images and numeric notebook measures: archive/review/quantum_comparisons.png, quantum_sensitivity.png, quantum_notebook_measurements.json. No paper PDF, raw source checkout or capture script is shipped.

Public NPZ SHA256: `bb2e06e8e50b9ff2d56fc6a03dface99e2f23939589c1a54a9c906a063d5b812`. All values float64 or complex128 without downcast, exact numeric transfer from published table. Projectors are known calibrated analyzer states, not target-derived. Source file SHA256 index:
```json
{
  "quantum_oracle.npz": "39e003ec07abbc80aed8c10b9037cd393994a02e288f121adff80ba19e97508c",
  "quantum_independent_reruns.npz": "901f08c494d0d6fc49452967c9e9a5a2d93a51aa6cebb293a4eb928929162a4e",
  "capture_quantum.py": "b6aa2a2a24a97433d2390fd4512a011cbb5b30aba641847dd5acbb864a9b57a2",
  "qkw/LICENSE": "923eef3ef5093cf567115c074063b434f642e0e9d05650b8ad29b1a40dfc0d91"
}
```

## Original execution and discrepancy investigation

Isolated oracle-env uses Python3.11.11 and the shared NumPy2.4.6/SciPy1.17.1 numerical primitives read-only via a local .pth; original package is imported only from private archive. tomli-w1.2.0 installed only into oracle-env for original config support. Cleaned runtime cannot import QuantumTomography and requires only NumPy/SciPy/Matplotlib. CPU Windows, single OMP/MKL/OpenBLAS threads. See original dependency pyproject for modern library version provenance; no task-level upgrades. Native original call: Tomography(2).tomography_MLE(I/4,counts,projectors,zeros(36)); initialization/stopping match source. Native runtime initially0.09794s, fresh log exits0. No warnings in original reconstruction. Notebook frontend reports Windows ZMQ event-loop fallback and transport advisory; kernel uses explicitly private batch10-worker specification pointing to shared research Python, not Anaconda default.

The original high-level linear initializer documents a transpose convention problem. This task calls the supported MLE routine with target-independent I/4 explicitly; it does not silently rely on the linear initializer or fix an upstream formula. Independent Born probabilities and phase checks establish the convention. Original current coordinate labels are permuted relative to paper Eq75 but the executable positive-matrix mapping is unchanged.

Native original output: scale79259.40841883626, Pearson-like objective501.3468431122622; rho agrees with the manuscript rounded matrix (HH.509,HH/VV coherence−.489+.017i). Investigation checked all36 values, analyzer order, normalization and circular convention against primary source. Per-setting corrected totals vary by roughly percent levels; the manuscript says counts have already been corrected. Therefore objective501 is retained and disclosed, not normalized away, threshold-relaxed or described as shot-noise accuracy. Missing correction covariance prevents assigning a validated reduced chi-square. The complete corrected acquisition is preserved. This is a fixed common-scale method estimate.

## Traceability and independent validation

| Paper stage | Original code | Local callable | Test/evidence |
|---|---|---|---|
| Polarization states Eq6 and Born projections | TomoClassHelpers.maxlike_fitness trace | physics_model.predict | analytic HH probabilities,orthogonal completeness,circular conjugation sensitivity |
| Positive state Eq75 and normalization | TomoFunctions.t_matrix,t_to_density | triangular,density | frozen probe density; PSD/unit trace/Hermitian tests |
| Pearson likelihood / nonlinear inverse | TomoClass.tomography_MLE | solvers.reconstruct,residual | frozen initial residual and complete native output parity |
| Initial count scale | density2t(I/4),.0001 offset | initial_coordinates | frozen original coordinates |
| Paper Figure7c characterization | published tangle/entropy/overlap | visualization.state_metrics | independent ideal Bell/mixed-state cases; fresh notebook table |
| Measurement completeness | quantum projector span | local linear design | rank16; no-circular rank9 and held-out residuals |
| Standalone benchmark metrics | separate local judge | evaluation/judge.py | known reference identity and manually known residual scalar; negative/invalid cases |

Before extraction, parity thresholds: forward/probe/initial residual1e-12; final density2e-5; intensity2e-5 relative; objective.01 absolute. Fresh cleaned final output agrees exactly with native original in this backend. Independent original reruns from slightly different positive mixed initial states differ by0,3.44113e-7,1.48913e-6 Frobenius; scale errors<=2.54322e-9. Float32 representation changes density1.06071e-8 and intensity2.73638e-8 relative. Final Judge boundaries tightened to2e-5 density and1e-6 intensity using those measurements; preimplementation conservative proposal.001 retained in archive. Physics tolerance1e-7 accommodates declared complex64 output. Observation normalized RMSE=.006664358465880508; fixed boundary.02 distinguishes grossly inconsistent measurements while numerical-reference/physicality bounds remain conjunctive. This criterion is not a pure noise test.

Hidden evaluation/baseline_reference.npz contains reconstruction(1,4,4),intensity(1,1), both original-method results. No ground_truth.npz because no independent measured state exists. test/fixtures/upstream.npz supplies parity tests. notebooks/reference_outputs/upstream.npz separately supplies notebook original-run comparisons. Consumers are explicit; no runtime depends on either. Historical manuscript metrics are separately labeled and never used as fresh numeric fixtures. The ideal Bell vector is a characterization target only, never hidden truth or solver initialization.

## Tests, notebook and review

Commands from task root: `python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test/ -v`; native `python -m jupyter nbconvert --to notebook --execute notebooks/bbo_photon_polarization_density.ipynb --output bbo_photon_polarization_density_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=batch10-worker`.

All21 tests pass, including full pipeline, original intermediates, independent physics, metric known cases, missing/invalid assets, shape/dtype/keys,NaN/Inf/object,poor/zero/mixed/conjugated/nonphysical/scaled output,foreign-working-directory CLI and failed report writing. Initial test attempt encountered preexisting Windows ACL denial at default pytest temp root; rerun uses private PYTEST_DEBUG_TEMPROOT without changing tests or scoring. This is an environment path exception, not a numerical failure.

Notebook native execution completes with six explicit story sections, all36 fresh MLE,16-outcome MLE,linear inversion,no-circular phase test and24 labeled synthetic count perturbations. Figures inspected: no cut labels or missing series; imaginary matrix uses separate±.025 color range to reveal phase structure; real matrix±.51; measured counts have correct analyzer axes; normalized residuals and eigenvalues show differences; synthetic uncertainty plot explicitly non-calibrated. Historical Figure7 characterization table retained as comparison to actual fresh metrics. Canonical notebook retains executed outputs; temporary executed duplicate removed. Timeout180s justified by actual full execution around8s, leaving startup margin. No zero-filled methods, hidden-result substitution or matplotlib backend mutation.

Plans reviewed against source signatures, all16-coordinate ordering, T T-dagger orientation, floors,objective normalization,complex conventions,initialization,stopping,paths and units. Runtime symbol audit: onlyNumPy/SciPy general optimizer/Matplotlib; no domain solver, cross-task imports,upstream imports or task-trained assets. Public README/acquisition NPZ/dependencies are sufficient. Public files checked for forbidden GitHub domains and privileged reference arrays. Minimal dependency versions match actually executed shared environment. No new synthetic benchmark observations generated; notebook perturbations explicitly supplementary.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Primary paper,verified remote revision,data hashes,traceability | passed | above/source archive |
| Exact bound original and cleaned execution | passed | native logs/intermediate fixtures |
| Independent parity and applicable physics | passed |21 tests |
| Constants,units,precision and conventions | passed | exact counts/projectors and analytic tests |
| Standard CLI and independent Judge | passed | output/judge_report.json |
| Negative/error Judge contracts | passed | test_contract.py |
| Public self-contained acquisition/output and no leaks | passed | README and NPZ audit |
| Minimal local executable math/no domain runtime | passed | source symbol review |
| All tests and active notebook | passed | logs/canonical executed notebook |
| Historical/original/fresh distinction and threshold rationale | passed | protocol/reruns/notebook |
| Plans/source semantic agreement | passed | reviewed above |
| Inventory/license/temp cleanup | passed | template inventory and retained MIT notice |
| Independent physical truth | unavailable (not mandatory for real data) | no measured truth claimed |
| Coordinating maintainer release acceptance | pending | task staged, not integrated |


Coordinating review follow-up: Judge now explicitly selects ground_truth.npz when present, otherwise baseline_reference.npz, and rejects nonphysical hidden references as evaluator errors. Focused precedence/invalid-reference test added; all20 unit tests pass plus prior2 pipeline/parity tests (22total). Current dataset still has no ground truth. Branch-only change did not require reexecuting notebook. Figure count is2 figure groups comprising9 panels.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 2 computed PNG outputs, 13 formulas. Private evidence: research_batch10_20261027/acceptance/bbo_photon_polarization_density/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
