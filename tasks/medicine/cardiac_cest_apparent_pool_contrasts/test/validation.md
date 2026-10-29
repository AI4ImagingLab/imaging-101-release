# Provenance and validation — 2026-10-27

## Binding, source and acquisition evidence

Verified real acquisition, privately reconstructed into magnitude observations: the author's complete 62-frame cardiac radial CEST scan 12, Bruker PharmaScan 70/16 7 T, study 20241011_141940_M1913_1_1. This is cardiac acquisition with a broad input-signal support, not a myocardium segmentation and not the initially considered liver binding. The inverse target is six apparent Lorentzian amplitudes, widths, centers and local water frequency; no concentration, exchange rate or independent physical truth is available. This differs from diffusion microstructure, dynamic contrast kinetics, relaxometry and linear fluorescence unmixing in the existing catalogue.

Primary paper: Pre-CAT, 2026, https://pmc.ncbi.nlm.nih.gov/articles/PMC13142558/ (full XML privately archived). Author source https://github.com/jweigandwhittier/Pre-CAT at remotely verified commit 5b0ca050dd81f333b999712c4a401296e0896b3d, MIT; source and notices retained privately, required LICENSE retained in task. Real data https://doi.org/10.6084/m9.figshare.30505685, author Jonah Weigand-Whittier, CC BY 4.0 verified in figshare_metadata.json, attribution in NOTICE. Native archive 402826835 bytes, SHA256 1b54f06416e83ab93ff6ac88a81bf8588c7e8218dddcc15cf1bc37b474749ca8. Native fid hash 72cddf5bb38f147a8eb6350b10d123fa7c1afd7d814edf02f2ad3ab1c5f74b3a; trajectory hash 2be3364d074ae80dc2459459ef8bba221b6706c96a7f541dbc0ea88aa2151623.

The distributed online 2dseq was investigated but NOT used as a substitute for the author's radial branch. Original Bruker reader extracts all native fid/trajectory samples; original BART v1.0.00 nufft -i and rss 8 reconstruct all 62 frames. BART source https://github.com/mrirecon/bart revision 731bfd3ec4f09ffa59b497ded1274a2c0357c8af, BSD3. Private WSL Ubuntu24.04 build uses existing gcc/make/FFTW/BLAS and privately extracted LAPACKE dependencies; PNG disabled. No shared package was upgraded. Optional PCA and respiratory motion correction are disabled, not implicitly reproduced. The task starts at these legitimately measured-derived magnitude images; MRI image reconstruction is acquisition conversion, not the target inverse method.

Acquisition parameters are copied from native method records, including 192x192 image, 25x25 mm FOV, 0.9 mm slice, 1000 radial projections, four coils, 53 saturated frames plus nine references, 300.324686903002 MHz working frequency, TE 2.12390670906575 ms, TR 8 ms, peak B1 1.1 microtesla, twelve 13.7 ms Gaussian pulses and 4.56 ms delay. Frequencies follow original rounding to 0.01 ppm. Public support is first reference greater than 15% of its maximum: all 8936 pixels retained, including noisy boundary pixels. No fitted result selects support.

## Independent original execution and freeze

Private archive relative to this batch worker: archive/Pre-CAT. Capture scripts radial_prepare.py, radial_oracle.sh, cest_oracle.py; logs radial_oracle.log, cest_oracle.log and cest_repeat.log. Native source scripts/load_study.py thermal_drift and scripts/cest_fitting.py two_step execute directly; capture wrapper only records curve_fit inputs/outputs. BART full conversion about 80 s, exit0. Original two_step all8936 fits:171.3592675 s, zero failures. Full independent repeat:158.2753226 s, zero failures, bitwise-identical saved results. Initial PowerShell stderr handling reported exit1 for harmless Streamlit bare-mode warnings despite completed files; repeat explicitly propagated process exit0. No scientific warning was hidden.

Oracle environment: private archive/ivim-oracle-env Python3.11.11, NumPy2.4.6, SciPy1.17.1, Streamlit1.66.1, scikit-learn1.9.1, pandas3.1.0; plotting dependencies accessed from validated shared environment. These GUI/parser dependencies are private only. Windows CPU, all BLAS/OpenMP thread counts1; deterministic bounded least-squares with no random seed. Original exposed maxfev set500 before full execution instead of source default50; xtol1e-10 and ftol1e-4 unchanged. Every supported spectrum completes. No fitted-value filtering or failures replaced with zeros.

freeze_cest.py created archive/Pre-CAT/frozen_acceptance.json BEFORE local src extraction: stage rtol1e-7/atol1e-8; individual amplitude RMSE1e-5, width RMSE0.001 ppm, center RMSE0.0001 ppm, shift RMSE1e-5 ppm. Full independent original repeat has exactly zero parameter differences; tolerances permit floating-point implementation differences without vacuous self-comparison. Whole normalized observation RMS0.07805802475666919 gives fixed auxiliary boundary0.08586382723233611. No threshold changed after local comparison. This criterion is reference-method agreement, not biological accuracy.

## Traceability and independent checks

| Mathematical stage | Original | Local | Validation |
|---|---|---|---|
| Temporal reference interpolation and normalization | load_study.thermal_drift | preprocessing.normalize | test_original_normalization, complete original normalization.npz |
| Lorentz line and sum | cest_fitting Lorentz/background functions | physics_model.lorentzian,background,pool_sum,predict | three original fit stages; analytic peak/FWHM and Jacobian finite differences |
| Initial water/MT field estimate | cest_fitting.two_step first curve_fit | solvers.fit_spectrum first stage | fit_stage_0,4468,8935 original fixtures |
| Shifted restricted background and residual pool fit | cest_fitting.two_step later curve_fit calls | solvers.fit_spectrum stages two/three | all stage parameters, abscissae, data and predictions |
| Complete native support reconstruction | original all-pixel capture loop | solvers.reconstruct/main.run | test_fresh_full_pipeline and standalone Judge |
| Per-quantity RMSE and observation reprojection | independent NumPy definitions | evaluation.judge | known metric cases, malformed/poor inputs and foreign cwd |
| Diagnostic bound fraction | analytic three-position case | visualization.bound_activity | unit test narrow water gauge |

All formulas, factors, signs, units, signatures, support and output shapes were reviewed against source and both plans. The underlying Lorentz denominator uses width squared/4 (FWHM), frequency correction subtracts field shift, and residual subtracts background attenuation. No hidden calibration is injected. No magnitude conversion, alignment, map rescaling or pixel selection is performed by Judge. Library modules do not choose a Matplotlib backend. Clean runtime imports only standard library, NumPy, SciPy, Matplotlib; no domain or cross-task imports, no generator because observations are real.

## Evidence assets and consumers

test/fixtures/normalization.npz is complete upstream temporal normalization and is consumed by stage parity. Three fit_stage NPZs record original selected support-index fits, used only by comparing tests. evaluation/baseline_reference.npz holds original amplitude,width,center,field_shift arrays; actual ground truth does not exist. notebooks/reference_outputs/upstream_fit.npz holds original full parameters,shift and original regional RMSE for notebook comparisons. fresh_fit.npz and fresh_measurements.json are freshly generated notebook results, clearly identified separately. Original regional RMSE excludes the water region and is never relabeled as all-offset RMS. None of these references is solver-visible.

## Fresh results, notebook and visual review

Standard main reconstructs all8936 spectra and writes the declared four arrays. All18 component parameter fields are bitwise identical to original; field-shift RMSE1.218512846681098e-16 arises from original averaging coordinate differences versus direct initial center. All20 Judge metrics pass. The active six-section notebook executes complete reconstruction (173.3167967000045 s), evaluates fresh output and computes genuine diagnostics. Native nbconvert used private research-bio kernel pointing at shared research-runtime; timeout1800 s, canonical executed outputs retained with portable python3 metadata, no duplicate executed notebook in output.

Six figures visually inspected: measured reference/support/drift; all six amplitudes; all widths and centers; three measured spectra with two-step and joint curves plus separate components; original/fresh scatter; all-pixel residual/field-zero forward ablation and field map. Labels, color scales, full support and independent reference identity are clear. Weak parameters often saturate bounds; this is reported, not hidden. Bound activity uses1e-6 of each box range, including narrow frequency gauge. No physical accuracy claim follows from parity.

Optional selected-pixel joint diagnostic originally raised curve_fit maxfev at the middle pixel. The same bounded least-squares objective and unchanged2000 evaluation cap now expose the actual last iterate with explicit convergence status, rather than invent a result. Pixels(86,14),(104,137),(120,88): two-step RMS .1472065638,.0432283743,.0345759542; joint RMS .1428930696,.0359092923,.0329888305. First/third converge in29/16 evaluations; middle reaches2000 and is labeled unconverged. This does not change primary reconstruction. Field=0 fixed-parameter forward ablation RMS .138800399463304 versus .07805802475666919 full, explicitly not a separately reconstructed method. No synthetic primary or invented truth.

## Commands and gates

Shared research-runtime/Scripts/python.exe; OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1. Commands: python -m src.main; python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; python -m pytest test/ -q with a new private --basetemp; python -m nbconvert --to notebook --execute --inplace notebooks/cardiac_cest_apparent_pool_contrasts.ipynb --ExecutePreprocessor.kernel_name=research-bio --ExecutePreprocessor.timeout=1800. Default Windows pytest temporary directory is inaccessible; no test code changed, a new private temporary directory resolves the environmental setup error. Earlier complete suite20passed163.42s; added bound-activity analytic test is included in final suite.

Passed: source/paper/data identity and mapping; exact-bound original and repeat; independent fixtures/physics; constants/coordinates; full clean run and Judge; Judge negative/error contracts; self-contained public README/all arrays/batch axes; no truth/calibration/GitHub leak; minimal clean runtime and symbol-role review; active notebook execution; all six figures inspected; historical/original/fresh distinctions; source/plans/README consistency; license and inventory. Final test outcome is appended below after process completion. Private cache directories are excluded by parent integration; no attempt to bypass earlier automatic cache-cleanup rejection.

## Hash manifest

```json
{
  "archive\\Pre-CAT\\paper.xml": "8e702f23849b0f8a6b44393703132386a4a5b10eb3af5ad1b1c305100467f2ea",
  "archive\\Pre-CAT\\remote_metadata.json": "1329d0b08f804eb6c5166e9010a7b1433971679baea7199857cc9ce8c3ebeb2f",
  "archive\\Pre-CAT\\bound_input.npz": "061eda970025730d64b8edec37e2b1d52862dfacaf6371a449e68be374afd45c",
  "archive\\Pre-CAT\\original_fit.npz": "bee5f32dd0a72c2615e927dcb18c34759b6bdb7c55e80d80275a94d0d761e072",
  "archive\\Pre-CAT\\repeat_fit.npz": "bee5f32dd0a72c2615e927dcb18c34759b6bdb7c55e80d80275a94d0d761e072",
  "archive\\Pre-CAT\\frozen_acceptance.json": "be8be040edb558089421cac2efbc45a88716ebd8fb8cb4202964744cfa5d17c3",
  "tasks\\cardiac_cest_apparent_pool_contrasts\\data\\raw_data.npz": "061eda970025730d64b8edec37e2b1d52862dfacaf6371a449e68be374afd45c"
}
```
`nFinal restored suite: 21 passed in 168.71 s, exit0, using private pytest-final-restore. Standalone CLI Judge rerun exit0 on fresh output.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 6 computed PNG outputs, 1 formulas. Private evidence: research_batch11_20261027/acceptance/cardiac_cest_apparent_pool_contrasts/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Current benchmark v2 evaluation, 2026-10-29

The v2 Judge independently evaluates every supported voxel and every53-frequency observation under the six-pool model. Each voxel MSE must be no higher than attained baseline MSE+1e-6; original whole-field RMS boundary and all physical ranges/support constraints remain. No ground truth exists. Independent bounded linear-amplitude fitting produces a different improved solution accepted by the Judge.19 unit tests passed. See cest_pytest.log.

This intentionally changes benchmark acceptance semantics while preserving the validated published reference algorithm and historical parity evidence. No observations, hidden reference NPZ files or stored reconstruction bytes changed. No full fresh blind Agent solve is claimed. Old public/evaluation files are archived privately under benchmark_readiness_20261029 before changes.
