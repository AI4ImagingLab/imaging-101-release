# Provenance and validation evidence

## Binding and realism

Paper: Tran etal., JPhysChemC126(2022)14003–14010, DOI10.1021/acs.jpcc.2c04254. Full text consulted through author dissertation chapter7, https://epub.uni-bayreuth.de/id/eprint/7210/1/DissertationEPub.pdf (54,299,202bytes). Original author repository https://github.com/thomas-tran-de/lockinthermography, remotely verified tree de9d20826571d3e23fa7ba6d82a3a49c6f04e659; BSD3 license retained. Remote tree response archived, not a freshness inference from local tracking. Dataset is author_real_processed: native IRBIS lock-in harmonic camera exports, not synthetic temperatures. Acquisition headers give 2021-10-12, frequencies7.547/6.154Hz, frame rate400Hz, frames per period53/65. Paper §7.3.2 establishes the infrared camera, modulated laser and1.3µm 8× lens; §7.3.3 explains amplitude/phase export and fiber analysis. Author program sets the identical pixel scale. QF01 is the author's label; quartz is consistent with the paper's quartz frequency experiments, but exact specimen diameter and figure correspondence are unverified and not claimed. Paper's500Hz frame rate and30% duty differ from these files (400Hz and approximately50%); use native header metadata, not generic paper settings. This is not a complete Fig2 frequency/power sweep.

New unknown is axial material diffusivity from harmonic spatial temperature decay. Existing heat-flux inversion reconstructs a source/boundary loading from thermal data; it does not infer this material transport coefficient. This binding was reserved with the parent before extraction.

## Original execution and independence

Private root is `research_batch11_20261027/material_mechanics`. Archive includes lockinthermography.zip, lockinthermography-tree.json, extracted original source, author dissertation/paper_text.txt, lockin_capture/upstream.npz and metadata.json, lockin_oracle.log, lockin_oracle_repeat.log, lockin_frozen_acceptance.json. Capture driver is capture_lockin.py; frozen acceptance driver freeze_lockin.py. Original command: `lockin-oracle-env/Scripts/python.exe capture_lockin.py`, run twice on all four exact bound files. Original Python3.12.4, NumPy1.26.4, SciPy1.13.1, Matplotlib3.8.4, Windows CPU, single-thread OMP/OpenBLAS/MKL. Runtime4.73s for recorded capture, exit0; repeat maximum difference0. Capture does not use randomization.

Original mathematical source is unchanged. Private adapter supplies the two GUI selections from measured images and forces Latin1 in np.loadtxt, matching the author's header decoder. GUI interaction is bypassed only after fixing its region/window inputs. Initial default-codec attempt failed on header bytes; initial horizontal selection after native90° rotation yielded an empty ROI. These were driver failures, not successful original runs; corrected input orientation completed the full original pipeline. NumPy2.4 rejects an original scalar-array conversion, so original execution uses its compatible NumPy1.26 environment. Cleaned runtime has no upstream dependency.

Fixtures and thresholds were frozen before build_lockin.py extraction. Stage comparisons rtol=atol=1e−8; output relative limits1e−5; original worst residuals .07901680999052618log-mK and .07679799642444958rad multiplied by1.1 for observation bounds. No threshold was relaxed. Fresh coefficients/diffusivities agree <9e−16 relative; explicit joint fit differs1.13e−11 from original generic curve fit. Original uncertainty variable StdThermalDiff stores propagated variance; no SD claim is made and it is not a judged unknown.

## Equation/function/test mapping

| Mathematical stage | Original implementation | Local implementation | Independent case |
|---|---|---|---|
| Harmonic axial heat equation, paper theory Eqs1–9 (thesis7.1–7.9) | LIT_FiberAnalyzer thermal slope formula | physics_model.thermal_profile / diffusivity_from_slopes | finite-difference harmonic PDE and arbitrary nonzero heat-loss cancellation |
| Header units, native orientation | LIT_Meas loading | preprocessing.native_to_vertical | exact original preprocessed arrays; analytic90°/degree case |
| Fiber detection/alignment | LIT_FiberAnalyzer image alignment | estimate_angle / orient_and_crop | frozen angle, center and complete rotated arrays |
| Distance profiles | LIT_FiberAnalyzer data extraction | centerline_profiles | all original finite and padded profile samples |
| Linear interval regression | LIT_Analyzer.LinFitRegion | linear_fit / reconstruct | original coefficients, analytic line |
| Slope product and frequency regression | LIT_FiberAnalyzer / LIT_SweepData | reconstruct | original per-branch and joint diffusivity |

## Hidden assets and metric conventions

No physical truth exists. baseline_reference.npz stores original reconstruction(1,2,2), fit_parameters(1,2,2,2,2), joint_diffusivity(1,1). observed_distance(1,308) and observed_profiles(1,2,2,308,2) are original extractions of the public observations, consumed solely by independent observation-consistency scoring; no privileged measurement is introduced. notebooks/reference_outputs/upstream_measurements.npz contains identified original output quantities for notebook comparisons. test/fixtures/upstream_stages.npz contains original native-preprocessed, rotated, profile, coefficient, diffusivity and uncertainty-stage measurements consumed by parity tests; scalar original frequencies and angles retain native scalar shape as fixtures, not observation/submission interface arrays. Original named StdThermalDiff is variance. No reconstruction is labeled ground truth. Float64 native observations are unchanged apart from lossless NPZ conversion.

Fresh runtime uses Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, Windows CPU, single-thread environment. No imaging/domain package is used. Allowed calls are array arithmetic, argpartition, order1 image resampling, closed-form regression and plotting. Source and Judge do not load each other's implementations. No target-derived calibration, GT or GitHub URL occurs in visible inputs.

## Current verification

Fourteen tests passed: full stage parity, analytic OLS, forward model, harmonic PDE, heat-loss cancellation, unit/rotation conventions, independent known metrics, missing keys/files, wrong shapes/dtypes, complex,NaN,Inf, poor result, unavailable assets, CLI evaluation from foreign cwd, report-write failure and plot smoke. Initial test collection caught a missing closing bracket in the Judge CLI return; corrected before scoring, no mathematical change. Notebook execution/visual review and final consistency audit will be appended below before handoff.

## Final notebook and completion gates

Native command `python -m nbconvert --to notebook --execute notebooks/quartz_fiber_thermal_diffusivity.ipynb --output quartz_fiber_thermal_diffusivity_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=imaging-batch11-material --ExecutePreprocessor.timeout=180` completed exit0 in4.34s. Private kernelspec points to shared Python; canonical notebook metadata restored to portable python3 after execution, with source cells verified unchanged. Windows ZMQ selector-thread and local TCP transport warnings were emitted, without execution errors. Timeout180s comfortably exceeds measured execution. Canonical notebook retains executed outputs; temporary output notebook removed. Five generated PNGs archived privately in lockin_figures and individually inspected: measured maps, complete profiles/fits, paired method comparisons and sensitivity, branch residuals, slope/parity summary. Labels, units and plotted groups were readable, and residual structure remained visible. No missing methods were replaced by cached or fabricated data.

Fresh notebook measurements include the three submission keys plus runtime_seconds(1) and residual_rms(1,2,2,2), saved in notebooks/reference_outputs/fresh_measurements.npz for identified audit measurements; notebook reconstruction never loads this file. Final CLI, Judge and14-test suite all exit0; logs archive/lockin_final_check_0..2.log. Judge gives diffusivity relative error1.1228462604151446e−11, coefficient error7.771561172376096e−16, log-amplitude RMS.0790168099905262, phase RMS.07679799642444958. The full arithmetic uses no domain library and remains independent of upstream installation.

Gate review: PASS paper/revision/data/source mapping; PASS original exact-data run and repeated capture; PASS pre-extraction frozen tests/physical checks; PASS constants and finite exact data; PASS main/Judge; PASS negative/error contracts; PASS self-contained public observations/output conventions; PASS absence of truth or source URL leakage; PASS README metadata and NPZ interface; PASS minimal shared dependency CPU runtime; PASS local core mathematics/call-role review; PASS14 tests; PASS active native notebook and all visual comparisons; PASS provenance and fixed metrics; PASS plans/signatures/units and closed-form derivation review; PASS template inventory and BSD notice. No mandatory gate remains unverified. Exact specimen diameter and published-figure identity remain expressly unclaimed, not fabricated completion evidence.

Traceability refinements: fiber stages map specifically to LIT_FiberAnalyzer.FocusOnFiber, FindCenterFromAmp, getVerticalLines, extractFiberPixels and AnalyzeFiberLITMeas. Archive hashes indexed in archive/lockin_archive_index.json. Source fixture consumer is the parity suite; notebook original measurements consumer is the original/fresh comparison; fresh_measurements is an explicitly identified audit output, not an oracle. Final review confirms signs, radians, mm²/s, unweighted objective, through-origin joint normalization and source/data flow agree with implementation.

Parent contract review corrections: restrict submission dtype to declared float32 or float64 (reject float16), and give ground_truth.npz precedence when present. No ground truth is supplied in this real task; scoring continues to use the same baseline. Fixed observed_distance/observed_profiles always come from baseline_reference as identified public-measurement extractions, independent of which quantity reference is selected. Added float16 rejection and conflicting-GT precedence tests. Full suite now16 passed in1.03s with private pytest-lockin-contract directory; fresh Judge remains pass with identical metrics and unchanged boundaries. These evaluator-only corrections do not alter reconstruction or any plotted notebook result.


## Exact source file hashes

{
  "measurements": [
    {
      "amplitude_path": "testing\\Fiber Testing\\ASCII\\QF01_FreqSweep_reverse_Results_0008_Period0001_Amp.txt",
      "sha256": "9528b6cd0893ee3567ba976a4f83a3fa6c2f7d84067e31b483d8761ab2a15964",
      "phase_sha256": "0965cfa590691e1b8665e6332e7d20b3c0ad43bbb4a8470e3a4cd19f0aae82d8",
      "frequency": 7.547,
      "angle": 0.1703736486770024,
      "center": [
        20,
        550
      ],
      "diffusivity": [
        0.7198615792605564,
        0.7576524238632905
      ]
    },
    {
      "amplitude_path": "testing\\Fiber Testing\\ASCII\\QF01_FreqSweep_reverse_Results_0009_Period0001_Amp.txt",
      "sha256": "85c4d95f8baa04c93b26ee12fc8fc316d73a426f6de671d33c9365badf720712",
      "phase_sha256": "0579a5eb014527b0d2ff5d8a473a2f5765fdfc46aed543c4200e2530ec918725",
      "frequency": 6.154,
      "angle": 0.1703736486770024,
      "center": [
        20,
        549
      ],
      "diffusivity": [
        0.7252878996942944,
        0.7402361379115219
      ]
    }
  ],
  "runtime": 4.732955000014044,
  "python": "3.12.4 | packaged by Anaconda, Inc. | (main, Jun 18 2024, 15:03:56) [MSC v.1929 64 bit (AMD64)]",
  "numpy": "1.26.4",
  "exit_status": 0
}


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 5 computed PNG outputs, 24 formulas. Private evidence: research_batch11_20261027/acceptance/quartz_fiber_thermal_diffusivity/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
