# Provenance and validation

## Binding and source
Paper: Probst et al., RSI86 024706 (2015), DOI10.1063/1.4907935; arXiv1410.3365v2 (26 October2014). Full primary HTML was read; its regenerated header date is not the paper version date. Author repository https://github.com/sebastianprobst/resonator_tools, remote API-verified commit dc5fe575ed5b433a1d0d227cee00359f57ae6f0f. Native source and S21 measurement are selectively archived outside task at worker archive/resonator_tools; source_catalog/tree JSON indexes verified source file sizes before download. Git transport initially failed; PowerShell HTTPS raw/API download succeeded. No remote freshness is inferred from local tracking.

Data: examples/S21testdata.s2p, SHA256 f817432df65cc2724ed5c2ea4550bd4750471b520e7f7eb7009e398fdcd5f74b. Native file identifies RAW.DATA, 2012-08-08 15:06:23, NIST275mK and -90dBm. It is an author-provided real instrument record, not simulator output. All1001 frequency/S21 samples are retained; native GHz multiplied by exactly1e9, real/imaginary columns3/4 combined as complex128. Other S-parameters are placeholders, not bundled. No derived calibration or truth is visible. Repository LGPL3 license is preserved; no separate data-license exception is specified upstream. Maintain repository attribution for source and example measurement. No independent loss truth is claimed.

## Original before extraction
Private oracle_resonator.py inserts only archived original source, calls unmodified notch_port.add_froms2p then full autofit on the exact native file. It saves original arrays plus fit dictionary, and independently invokes calibration/circle stages for fixtures. Original capture and JSON/log are in archive. Original and cleaned environments share the already-installed numerical interpreter but upstream is isolated by private source path (no installation, no runtime imports of upstream, no dependency contamination); no separate oracle venv was needed since only the same general primitives are required. This is a modern author-code reproduction, not execution in the2015 package environment. CPU, Windows, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, BLAS threads1; noGPU. Original measured runtime0.0393s on first full run, native exit0, no numerical warnings. Original rerun was completed before fixture freezing/extraction; its final runtime is in resonator_original.json.

Frozen private frozen_resonator_case.json records criteria before local extraction. tests consume only frozen NPZ, never upstream imports. test/fixtures/upstream_stages.npz is consumed by full-stage parity; notebooks/reference_outputs/upstream_original.npz by active notebook original comparison. Fresh figures and fresh_measurements.json are active notebook review outputs consumed by maintainer visual review. evaluation/baseline_reference.npz contains only identified original-method reference; actual truth unavailable. No ground_truth file invented.

## Equation/stage traceability

| Paper stage | Original | Local | Validation |
|---|---|---|---|
| Eq1 complex notch | circuit.notch_port._S21_notch | solvers._S21_notch; independent physics_model.forward | analytic resonance/no coupling, source parity |
| Eq3-11 algebraic circle | circlefit._fit_circle | solvers._fit_circle | frozen circle center/radius |
| Environment delay | notch_port.get_delay/do_calibration | same standalone functions | frozen seven calibration outputs |
| Eq12 periodic phase | circlefit._phase_fit | solvers._phase_fit | final normalization and output parity |
| Eq13 off-resonant point | notch_port.do_calibration | do_calibration | frozen calibration/normalization |
| Diameter loss | notch_port.circlefit | circlefit; independent dissipation_quality | inverse-Q identity, known analytic case |
| Covariance propagation | _get_cov_fast_notch/circlefit | same explicit derivatives and covariance | original fit dictionary uncertainty in notebook |
| Complete pipeline | notch_port.autofit | reconstruct/main.run | fresh CLI/Judge/full active notebook |

## Ambiguities and numerical fidelity
Paper §IV expression for complexQc has a sign inconsistency with §I/Eq1. Code uses absQc exp(-i phi), matching Eq1. Real diameter correction uses cos(phi), so returned corrected Qc/Qi do not depend on that textual sign typo. Paper cable-delay residual formula and SNR display omit powers expected for squared-distance/statistical variance; local implementation explicitly preserves the actual source radial-distance least-squares objective. These are recorded paper/code discrepancies, not silently corrected algorithms. This task uses the author's modern scaled least-squares implementation and explicitly retains its baseline slope0, fsolve/SVD and periodic phase conventions. Source semantic review corrected an early documentation draft to the actual leastsq/curve_fit and fsolve implementations. No threshold was relaxed.

## Metrics and quality
All per-array metrics use explicit nonzero bounds in metrics.json. Original self-comparison gives zero, but bounds were frozen nonzero before extraction: resonance relative1e-7, eachQ max relative1e-4, complete complex relativeL2 1e-5. Original observation residual0.00879655899647674 gives bound1.1times that residual. Fresh output is numerically identical to original on this CPU; actual physical accuracy is not proven. Structured raw residual is visible and attributed to conditional isolated-resonance/constant-gain limitations, not an independently calibrated noise variance. Known independent metric vectors verify all metrics; zeros fail quality. Invalid inputs yield errors and empty metrics; missing hidden assets and report-write failure return2. Foreign-working-directory execution passes.

## Notebook and visual review
The complete notebook was actively executed twice (second fixes Markdown backslash escaping), native nbconvert exit0, shared-kernel argv explicitly points to research-runtime Python. Full second execution approximately5.5s wall including converter startup; fresh fit runtime is recorded in fresh_measurements.json. Native notebook preserved in archive/resonator_executed.ipynb. Converter produced Windows zmq selector-thread/TCP warnings; no numerical or cell errors. Both complete figures were opened using view_image: all axes/units/legends readable, no clipping, measured/fresh/original overlaps consistent, structured residuals visible, paired Q and uncertainty bars correctly labeled. No substitute cache/zero values. Original example other-instrument baseline/GUI notebooks are outside this bound dataset; comparison uses this task's full scientifically relevant channels and corrected/uncorrected dissipation. Paper synthetic SNR sweeps are contextual, not claimed reproduced by this real-data benchmark.

## Gate status
Passed: remote/paper/data binding; original exact-data run; frozen stage/full parity; independent forward and inverse-Q physics; native units/coordinates/dtypes; main and strict Judge; malformed/poor/CWD/assets/write tests; allowlist and no GitHub URL/public target leakage; minimal pinned dependencies and CPU execution; no domain shortcuts or cross-task/upstream runtime imports; complete active notebook and visual review; actual-method mathematical/signature plan consistency; standard file inventory and source license. Parent independent acceptance remains pending. No claim of independent ground truth or2015 environment compatibility.


Supplemental frequency-density experiment: independently ran original author full autofit on every1/2/4/8th native sample (1001/501/251/126 points), archived oracle_resonator_subsampling.py and frozen resonator_subsampling.npz. Notebook actively repeats complete calibration+fit on each subset, original/fresh all-parameter parity rtol1e-8; maximum corrected internal-Q drift0.1523%, versus full-sweep conditional1sigma1.3396%. The benchmark remains all1001points and unchanged thresholds. upstream_subsampling.npz is consumed by this notebook experiment; fresh_subsampling.npz and subsampling_review.png are fresh review records. Third figure actually viewed and readable; matches original/fresh curves, informative nonmonotonic Q sensitivity. Final full notebook native exit0 wall5.85635s, includes complete benchmark and all4subset refits.

Parent evaluator hardening: a zero or negative hidden reference quality factor makes relative-Q scoring undefined. Judge now rejects that asset explicitly with code2/error and empty metrics, and checks all measured metrics are finite before report generation. Added an independently corrupted-reference contract case; complete parent rerun passed15 tests, CLI, Judge and full active notebook. No scientific threshold or reconstruction changed. Design description was corrected from two to three figures and identifies the density experiment's direct active reconstruct calls.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 3 computed PNG outputs, 10 formulas. Private evidence: research_batch12_20261029/acceptance/microwave_notch_resonator_losses/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Final public-estimator sufficiency amendment

Independent review identified that tight method-reference comparisons require the selected conditional/finite estimator to be defined publicly. README Method Hints now specifies the necessary mathematical prior, feasible set, or canonical finite mapping and constants, without source code, a full hyperparameter table or step-by-step pseudocode. Physical unknown/data, local mathematics, all Judge criteria and all executed notebook code/results are unchanged. The old README/hash, exact source correspondence and new hash are retained privately in acceptance/microwave_notch_resonator_losses/public_mapping_amendment.json and public_mapping_before_README.md. Parent semantically compared every new equation/convention to actual source and independently rerendered all documents with both math engines. No new numerical reconstruction is claimed from this documentation-only amendment.
