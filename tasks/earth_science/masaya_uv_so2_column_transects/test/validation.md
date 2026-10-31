# Masaya measured SO2 task: provenance and validation

## Binding and measured acquisition

Status: locally validated; parent integration review remains separate. This is Earth science: a volcanic SO2 column transect recovered from UV absorption spectra. It is not generic spectral denoising, instrument calibration, or an atmospheric column supplied as truth. No independent measured SO2 truth is available. evaluation/baseline_reference.npz contains the validated original method result.

Primary method: Esse et al., “iFit: A simple method for measuring volcanic SO2 without a measured Fraunhofer reference spectrum”, Journal of Volcanology and Geothermal Research (2020), DOI https://doi.org/10.1016/j.jvolgeores.2020.107000. The author accepted manuscript dated July 1, 2020 was inspected at https://pure.manchester.ac.uk/ws/portalfiles/portal/174067623/iFit_JVGR_Revision_2.pdf. Section 4.3.1 explicitly includes the no-flat-field comparison selected here.

Original repository: https://github.com/ManchesterVolcanology/iFit, remotely verified revision 7c523c71d0ae106bccd42cce65aaa9779559fb44, MIT license retained. This is the verified selected revision, not a claim of indefinite remote freshness. Original source is archived under archive/ifit outside this task.

Data: Ben Esse, iFit example data, Figshare 12174675 version 1, DOI https://doi.org/10.6084/m9.figshare.12174675.v1, CC BY 4.0. The 47,227,680-byte Spectra.zip is archived intact. Classification: verified_real instrument spectra. The paper identifies the Masaya traverse experiment and the actual files identify Ocean Optics FLMS02101, 2018-01-14, 100 ms integration, ten coadds. All 360 files spectrum_00316.txt through spectrum_00675.txt and the measured dark spectrum are retained; no row is removed. Each has 2048 wavelength/intensity samples. Data conversion only places existing numeric observations into batch-first NPZ. Calibration comes from independent solar/SO2/O3/Ring spectra, not the hidden SO2 solution. The Figshare metadata and original file hashes are preserved in ifit_figshare.json and archive/ifit_paper_run.json.

## Original execution and independent fixtures

Private command: oracle-env/Scripts/python.exe archive/capture_ifit.py, executed with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. Original iFit modules, not cleaned functions, perform preprocessing and all 360 curve fits. Native rerun exited 0 in 83.0925853 s (fit portion 81.1049588 s). Evidence: archive/ifit_native_rerun_command.json and ifit_native_rerun.log; source/data/environment metadata: ifit_paper_run.json; numeric capture: ifit_paper_oracle.npz. Environment: Windows 10 build 26200, Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, CPU numerical backend, single thread. The private environment adds only original verification dependencies and is never imported by released task modules.

The pre-extraction protocol archive/ifit_preimplementation_protocol.md fixes all stage and pipeline tolerances before local implementation. Preprocessing and sampled calibrations must match exactly; ILS absolute error <=1e-12, forward relative tolerance 1e-12; final SO2 relative norm and dynamic-range NRMSE <=1e-4, maximum absolute difference <=2e14 molecules/cm^2; fitted spectra relative norm <=1e-5. No tolerance was widened after a failure. The independent original rerun versus frozen fixtures produced: {"parameters": 0.0, "fit": 0.0, "processed": 0.0, "flags": 0.0, "initial_forward": 0.0, "ils": 0.0}.

Native fit flags are all 1 and all selected columns/spectra are finite. Native OptimizeWarning reports unavailable covariance for some nuisance parameters; only 61.1% of parameter-error entries are finite. These are neither suppressed as success evidence nor presented as reliable confidence intervals. SO2 ranges from -1.80909978e16 to 1.7423723384e18 molecules/cm^2.

## Discrepancies investigated

Paper Table 1 and present example config differ: initial O3 is 1e19 in the paper versus 1e9 in the example, O3 temperature is 243 K versus 223 K, and shift is -0.1 versus +0.1 nm. This binding explicitly selects the paper parameters and 243 K calibration. The current API saturation default of 100 counts is unsuitable for these recorded spectra; the invocation explicitly passes 65535 counts. The no-flat-field condition is an actual paper comparison, not an invented target-derived calibration. The first example-config capture remains archived separately and is not this task's oracle.

A first cleaned vectorized stray-light reduction differed by only 7.2759e-12 counts but changed SO2 by 4.83e15 molecules/cm^2 (NRMSE .0003028), failing the frozen tolerances. Restoring the original rowwise reduction order resolved this numerical conditioning issue: preprocessing, ILS, initial forward spectrum, final SO2 and fitted spectra all match exactly. Evidence: archive/ifit_clean_parity.json and ifit_clean_details.npz. No new model or optimizer was substituted. A unit assertion expecting cosine similarity to equal exactly 1 was corrected to an analytic 1e-14 floating-point tolerance after observing 0.9999999999999998; the metric and evaluation boundary did not change.

## Equation and function traceability

| Paper stage | Original implementation | Local implementation | Independent validation |
|---|---|---|---|
| Dark and stray-light correction | spectral_analysis.Analyser.pre_process | preprocessing.preprocess | frozen corrected intensities, exact |
| Beer-Lambert intensity (Eq. 10 model) | spectral_analysis.Analyser.fwd_model | physics_model.optical_depth, forward | frozen forward; doubled optical depth and absorption checks |
| Asymmetric super-Gaussian response | make_ils.make_ils | physics_model.instrument_lineshape | frozen profile, positive unit sum, constant interior |
| Wavelength shift/stretch, cubic sampling | Analyser.fwd_model | physics_model.forward | full native forward and pipeline parity |
| Nonlinear spectral least squares | Analyser.fit_spectrum / curve_fit | solvers.fit_spectrum, reconstruct | all 360 native final columns and fitted spectra |
| SO2 traverse/residual comparisons | paper traverse and flat-field comparisons | notebook and visualization.summary_figure | complete fresh notebook and actual figure inspection |

The notebook preserves the paper's comparison logic through measured-versus-fitted spectra, SO2 transects, optical-depth contributions and a traditional differential-absorption comparison. The additional linear DOAS diagnostic is explicitly not QDOAS parity: it fixes mean fitted registration/profile and uses the first 20 observed spectra as a measured reference. Original QDOAS results are not fabricated or claimed as rerun. Historical paper residual standard deviations (0.60% without flat correction and 0.46% with correction) are labeled historical acquisition/method context, not equivalent benchmark metrics.

## Fresh tests, notebook and metrics

Command: python -m pytest test/ -v. Final exit 0; runtime 84.758 s. All 19 tests pass, covering original intermediates, full CLI reconstruction/Judge, independent physics, known metrics, malformed and poor predictions, missing/evaluator assets, actual-truth precedence, foreign working directory and report-write failures. Full transcript: archive/ifit_final_pytest.log. The CLI is actively exercised by the full pipeline test, not replaced by fixture copying.

The native nbconvert run executes the complete notebook with a 600 s timeout and the explicit research-runtime kernel. It freshly reconstructs all 360 observations in 80.3413698 s; canonical notebook retains executed outputs. The temporary output notebook is archived outside the task. Three actual figure groups, 15 panels total, were visually inspected: the complete measured spectrogram and fit residuals, original/fresh SO2 profiles, representative spectra, all three optical-depth contributions, fresh DOAS comparison, no-SO2 ablation, initialization sensitivity and fitted profile. Labels and scales were readable; no archived array is presented as a fresh computation. Archive review/ifit_figure_1.png through _3.png and ifit_notebook_measurements.json record this review.

Fresh results: NCC=0.9999999999999999, dynamic-range NRMSE=0; mean spectral residual standard deviation 0.6283061179639226%; DOAS/iFit SO2 correlation 0.9990728923240646. Changing initial profile width in the supplemental experiment changes some columns by up to 4.889811686980056e15 molecules/cm^2. This is disclosed model/optimization sensitivity, not hidden by relaxing parity bounds. The fixed numerical benchmark must reproduce its declared initialization. Reference agreement does not establish physical accuracy, and native covariance warnings preclude claiming calibrated uncertainty.

Evaluation uses cosine similarity and RMS error divided by reference range, with fixed NCC>=.9999 and NRMSE<=1e-4. The independently repeated original run is stable under the fixed environment; the nonzero bounds allow numerical representation error without admitting wrong profiles. Zero, reversed, scaled and constant profiles are rejected by contract tests. The Judge independently validates exact shape, numeric real dtype, finite values and reference configuration, with actual truth taking precedence if supplied. This task ships no actual truth.

## Inventory, source-role and semantic review

Source implements acquisition and inverse mathematics explicitly using NumPy and SciPy primitives. scipy curve_fit is a general optimizer, not a domain inversion shortcut. No upstream, cross-task or computational-imaging package is imported by source, Judge, tests or notebook. requirements.txt contains only the cleaned shared-runtime subset; pytest/Jupyter remain development tools. The public README, NPZ and requirements are sufficient and contain no GitHub URL, hidden target, fitted nuisance calibration or secret seed. All public keys are documented with batch-first shapes and units. Plans were checked against actual signatures, units, signs, convolution alignment, interpolation, initialization and stopping defaults. No standalone metadata JSON or synthetic generator is included.

Consumers: test/fixtures/upstream.npz is the frozen original numerical test fixture; notebooks/reference_outputs/upstream.npz is the notebook's identified original comparison; evaluation/baseline_reference.npz is the method reference; output/reconstruction.npz is fresh CLI output. Their separate roles are intentional. Archive capture scripts, upstream source, original downloads and logs remain outside the task. LICENSE preserves iFit attribution; NOTICE preserves measured-data CC BY 4.0 attribution.

Array SHA256 values:
```json
{
  "data\\raw_data.npz": "122d16e17bd5a328b1bbefba5efbfca867814ca4f645dcb2e53ec6cf83823289",
  "evaluation\\baseline_reference.npz": "9057827e5e4185b4606edea1e974b5f8879d101ab323cc3626b6891e8637c96b",
  "notebooks\\reference_outputs\\upstream.npz": "f3fcae8cf9db2424cc8a4652668ce43f89d4663efd726be3c18bad9ce77e72e3",
  "output\\reconstruction.npz": "9057827e5e4185b4606edea1e974b5f8879d101ab323cc3626b6891e8637c96b",
  "test\\fixtures\\upstream.npz": "f3fcae8cf9db2424cc8a4652668ce43f89d4663efd726be3c18bad9ce77e72e3"
}
```

Mandatory local gates: paper/revision/data traceability PASS; exact-observation original run PASS; independent frozen parity PASS; independent physics PASS; fresh CLI/Judge PASS; malformed and poor-output contract PASS; public-only solvability PASS; no hidden-data/GitHub leak PASS; minimal runtime and explicit local mathematics PASS; full tests PASS; active executed notebook PASS; actual figure review PASS; provenance/metrics consistency PASS; README/plans semantic consistency PASS. Parent acceptance is pending and is not implied by these local checks.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 3 computed PNG outputs, 2 formulas. Private evidence: research_batch10_20261027/acceptance/masaya_uv_so2_column_transects/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
