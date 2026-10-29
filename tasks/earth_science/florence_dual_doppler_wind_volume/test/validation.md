# Provenance and completed validation gates

## Fixed acquisition, source and rights

This task recovers three spatial wind components on a9×21×21 Cartesian volume from two real NOAA NEXRAD observations during Hurricane Florence,14September2018. It is neither nowcasting nor scalar calibration. Author-supplied gridded Level-II radar files grid_ltx.nc and grid_mhx.nc are bound unchanged numerically. Original file metadata identifies radar A34.77590942N,76.87619019W,43m and radar B33.989151N,78.42910767W,44m. The filenames appear reversed relative to station geographical identifiers, so public documents use A/B and the actual coordinates. Scan reference times06:24:50 and06:24:17UTC differ; a simultaneous frozen-time field is an explicit approximation. No independent voxel noise covariance is supplied. Original gridding/dealiasing/calibration is retained, not inferred from hidden winds.

Original author repository https://github.com/openradar/PyDDA, remotely verified commit399627224ad4f2e72fdc83ae00e8ba2b641b1b67; live response openradar__PyDDA_remote.json and full source zip archived. BSD3source LICENSE retained. Original author data https://github.com/rcjackson/pydda-sample-data, exact downloads/registry SHA256 in florence_downloads.json. grid_ltx SHA256 d848a9cf33cd3b0008f8542680b35019f9aae80940d09327c2b6ded197096ffa; grid_mhx SHA256 b7ccaa9e09fdc03c02c5d254c04dfb1ecf737f170da0cc8510ef061f9adc3752. NOAA observations are US federal data. Official rights source https://data.noaa.gov/docucomp/xmlComponent/show/690447 states these NOAA-produced data are not subject to US copyright and provides CC0-1.0 dedication; captured noaa_data_rights.html. Third-party author gridding is attributed separately; no invented license is asserted for a repository lacking a separate data-license file. Local benchmark processing preserves author provenance and NOAA attribution. The earlier Darwin CPOL/Berrima working binding is archived privately, not shipped as another task, because its mixed-source redistribution terms were less explicit.

Paper Jackson et al.2020, PyDDA: A Pythonic Direct Data Assimilation Framework for Wind Retrievals, JORS8:20, DOI10.5334/jors.264. Actual nine-page published institutional NOAA PDF https://repository.library.noaa.gov/view/noaa/32197/noaa_32197_DS1.pdf, SHA256 a9a5437121a0cc7109c276e79c6ac63cec83cd339e64d134e8da4a6d9bcd037c; full extracted text jackson2020_pydda.txt read, particularly cost-function Eq1, adjustable constraints table, optimizer and Florence example. Source-supported configurable background-model term is used as a **zero-mean horizontal Gaussian prior**, not as measured/model wind calibration. The source applies it to u/v only; no artificial vertical prior is added. Required source-reference DOI10.1175/JTECH-D-11-00019.1 and10.1175/2009JTECHA1256.1 retained in README.

Full source volumes31×351×401 are sampled without interpolation at z0:17:2,y196:237:2,x155:196:2. Window center216,175 was selected only using valid measurements and the original30–150degree beam-crossing geometry. No source historical u/v/w reconstruction enters public data, oracle initialization, geometry or prior. All raw fields are promoted from original float32 to float64 without new precision; finite invalid placeholders become0and retain zero radial weight. Original and local calculations use the same valid numeric values and source coordinate projection. Source origin33.989151N,78.42910767W; original point lon/lat and beam angles are computed by original Py-ART/PyDDA functions before extraction. All public NPZ arrays and README units/axes are explicit. Public data SHA256 97a8d558b4f183d637ad9ee0ac785e9add26759bb9da2519fe8111bfddb223cc.

## Original execution before local extraction

Private capture_florence.py loads unchanged original wind_retrieve, cost_functions JAX, angles and read_grid functions. Importlib package shells avoid unrelated optional readers but actual scientific calls remain upstream. Py-ART transforms, original radial weighting, empirical terminal speed, full PyDDA JAX objective/autodifferentiation and JAXopt LBFGSB all execute. The selected backend is original JAX64 CPU, not a rewritten upstream adapter. No private oracle is imported by cleaned source/tests/notebook/Judge.

Original full parameters: Co1,Cm256,Cmod1e-4,zero_prior horizontal model fields, unit prior weights,Cx=Cy=Cz=Cb0,frz5000,low_pass_filterFalse,mask_outside_optFalse,mask_w_outside_optFalse,max_iterations500,upper_bc=lower_bc1,tolerance1e-8,parallelFalse,bounds±100m/s,zero initialization. Prior fields are mathematical regularizer assumptions, explicitly disclosed, not extra observations. Original first run 42.779180s, independent repeat actual exit0 in florence_original_command.json with log florence_original_rerun.log. Python3.11.11 NumPy2.4.6 SciPy1.17.1 JAX0.10.2 x64CPU. Thread limits1; original environments and selected source hashes in oracle JSON. Logs retain warnings including unsupported NumPy masked-array backend conversion, with numerical validity independently tested.

Frozen original fixture includes actual preprocessing rmsVr, terminal speed, a nontrivial arbitrary wind case, full objective and autodiff gradient, and final complete u/v/w. It was captured before cleaned task creation. Original valid radial count6576,rmsVr32.63715956937172m/s. Local explicit mathematics reaches objective 448.88186673635431, max projected gradient 1.8318539033216306e-09, 113iterations. Original JAXopt and local SciPy use different general L-BFGS-B line searches, so we compare converged minimizers, not artificial equal iteration paths.

## Mathematical discrepancy investigation and traceability

The original NumPy continuity-gradient implementation is not the exact discrete adjoint of its stated cost at boundaries and omits a density-gradient contribution. Independent directional derivatives identified this discrepancy. We select the already supplied original JAX automatic-differentiation backend, preserving the published objective, rather than silently correcting the NumPy oracle. Local gradients explicitly implement the transpose of one-sided-edge/centered-interior differences. A negative-gradient shortcut is not used. Both vertical analysis-plane boundaries remain exactly0from initialization and zero boundary gradients; this is a selected model assumption, not a measured impermeable storm boundary. Empirical fall-speed coefficient gaps exactly at60dBZ below freezing and49dBZ above freezing are preserved and disclosed; none of the valid selected observations lies at those exact gaps.

| Required equation/stage | Unchanged upstream function | Local implementation | Independent validation |
|---|---|---|---|
| Radar radial projection | Jax calculate_radial_vel_cost | physics_model.projection/forward | original arbitrary cost/gradient and physical zero-air-wind comparison |
| Empirical terminal velocity | calculate_fall_speed | physics_model.fall_speed | frozen original valid-voxel values, source constants and units |
| Radial RMS normalization/coverage | _get_dd_wind_field_scipy weighting | preprocessing.normalize_observations | exact frozen rms and public validity |
| Anelastic mass Eq1 constraint | calculate_mass_continuity_cost | physics_model.divergence | uniform horizontal flow zero divergence; full gradient finite difference |
| Discrete exact derivative | original JAX grad | physics_model.derivative_adjoint/objective_gradient | transpose identity and independent directional derivative |
| Source-supported model prior | calculate_model_cost | objective_gradient prior | explicit zero-mean u/v penalty; no prior on w |
| Full bounded inverse | _get_dd_wind_field_scipy JAXopt | solvers.reconstruct SciPy fmin_l_bfgs_b | original complete u/v/w and independent reruns |
| Metric/NPZ boundary | declared contract | standalone evaluation/judge.py | known vectors and malformed/poor/evaluator/foreign-cwd tests |

Stage tolerances fixed before task packaging: rms1e-14relative,fall5e-14relative,cost1e-13relative,gradient1e-12relative/1e-14absolute; full per-component dynamic-range NRMSE<5e-6. Exact-adjoint identity1e-13absolute; quadratic directional difference epsilon1e-3 avoids subtractive cancellation without truncation bias. All13tests pass. End-to-end radial residual RMS 2.28058715256m/s versus zero-air-wind 32.6295947456m/s, required improvement factor>.8. All three components meet independent original parity and finite optimizer convergence. This does not establish physical wind truth, especially for weakly constrained w.

## Reference tolerances, notebook and review

No independent wind truth exists. evaluation/baseline_reference.npz and notebooks/reference_outputs/upstream_measurements.npz identify original method result; no ground_truth file. Frozen upstream_stages fixture is consumed by comparing tests. Fresh cleaned_measurements NPZ is written independently by main/notebook. Evaluation checks all three exact float64(1,9,21,21)outputs with cosine NCC≥.99999 and dynamic-range NRMSE≤5e-5, no alignment/rescaling/cropping. The nonzero allowance covers backend convergence and serialization, based on actual independent original and cleaned repeats and float32 serialization measurement, not self-comparison. Actual measured values:

```json
{
  "reconstruction": {
    "original_independent_rerun_nrmse": 0.0,
    "clean_independent_rerun_nrmse": 0.0,
    "float32_serialization_nrmse": 1.3195777838304547e-08,
    "fresh_original_nrmse": 3.747437651693141e-08
  },
  "northward": {
    "original_independent_rerun_nrmse": 0.0,
    "clean_independent_rerun_nrmse": 0.0,
    "float32_serialization_nrmse": 1.3024932422980498e-08,
    "fresh_original_nrmse": 5.3363758800120513e-08
  },
  "vertical": {
    "original_independent_rerun_nrmse": 0.0,
    "clean_independent_rerun_nrmse": 0.0,
    "float32_serialization_nrmse": 2.0513414889349672e-09,
    "fresh_original_nrmse": 1.665758544832971e-07
  }
}
```

Zero, spatially uniform and reversed spatial fields all return valid-prediction fail exit1; actual reports archived florence_tolerance_reruns.json. Contract tests also exercise missing files/keys,wrongshape,dtype,NaN/Inf,complex,evaluator asset absence,report-write failure and foreign working directory. Judge imports no solver. Public allowlist has no hidden reference, seed, target-derived calibration or GitHub URL. Only NumPy/SciPy/Matplotlib are required; cleaned runtime has no PyDDA/Py-ART/JAX imports.

Six-story notebook actively reconstructs the complete raw bound volume; plots two measured radial acquisitions at three heights, beam vertical sensitivity/terminal speed/coverage, all three fresh wind components at three heights, matched original/fresh/difference maps, residual histograms and maps versus zero air wind, divergence and fixed-w boundary profile. Five actual horizontal-prior refits and three actual continuity-weight refits expose conditional w sensitivity. No missing method is replaced by zero; zero air wind is explicitly a physical comparison. Historical paper findings are context, not claimed reproduced wind truth. Actual kernel executable/backend/np/scipy/thread environment printed and archived. Full nbconvert execution exit0; canonical notebook retains outputs. Temporary notebook remains privately outside task. PNG/SVG/PDF and numeric_sources.npz for all six figure groups are private in D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth\figures\florence_dual_doppler_wind_volume.

README and both plans were reviewed against executable equations/signs/normalizations/units/boundaries, exact signatures/defaults, all3outputs and caller flow. Source/README/discretization constants agree by tests. General L-BFGS-B is documented, not a domain inverse shortcut. Source imports do not change plotting backend. Complete commands/runtime/exit evidence:

```json
[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\florence_dual_doppler_wind_volume",
    "runtime_seconds": 0.6418172999983653,
    "exit_code": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\florence_dual_doppler_wind_volume",
    "runtime_seconds": 0.1361528000124963,
    "exit_code": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\florence_pytest"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\florence_dual_doppler_wind_volume",
    "runtime_seconds": 1.4767752000043401,
    "exit_code": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "jupyter",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/florence_dual_doppler_wind_volume.ipynb",
      "--output",
      "florence_executed.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth",
      "--ExecutePreprocessor.timeout=180"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\florence_dual_doppler_wind_volume",
    "runtime_seconds": 13.010513000001083,
    "exit_code": 0
  }
]
```

Actual notebook environment:

```json
{
  "executable": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
  "os": "Windows-10-10.0.26200-SP0",
  "numpy": "2.4.6",
  "numpy_file": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Lib\\site-packages\\numpy\\__init__.py",
  "scipy": "1.17.1",
  "matplotlib": "3.11.1",
  "backend": "module://matplotlib_inline.backend_inline",
  "threads": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  }
}
```

All mandatory gates passed: exact paper/revision/data provenance; original and local exact-observation runs; frozen stage/full parity and independent physics; all component convergence; strict Judge fresh-pass and poor/malformed rejection; public self-contained metadata and dependency subset; local explicit forward/gradient/inverse objective;13tests; active executed notebook; justified nonzero rerun tolerance; plans/source consistency; final inventory and license. Parent scientific visual acceptance is pending. Archive root D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth; florence_archive_index.json indexes source/paper/data/capture/log/environment hashes. Canonical notebook SHA256 2fe1100dcbeda36bcd190663cd5612a967b71f9fcadf5b64cb13c443499b8d81.

Final binding semantic audit: both plans were rederived from final Florence arrays, source signatures and the original rerun log after identifying two stale Darwin values in the initial draft. Source dimensions31,351,401; original final iteration99/J448.8818667363615/gradient7.710967420138104e-9; local113/J448.8818667363543. README, notebook source and plans were scanned for obsolete date, shape, window and objective values; no stale acquisition remains. Both plans fully replaced, not supplemented with contradictory appendices.

Public sufficiency review: README Method Hints now defines the exact conditional objective, its256/1e-4 coefficients, RMS normalization, first-difference stencil, +/-100m/s bounds and fixed w endpoints. These quantities are part of the requested conditional inverse, not hidden solver tuning. Acquisition metadata remains instrument/geometry only. Observation blanks and neutral signed-wind pixels are explicitly distinguished. Numerical implementation, notebook and thresholds unchanged.
