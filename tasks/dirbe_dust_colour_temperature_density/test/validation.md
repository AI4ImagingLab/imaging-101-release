# Provenance and completed validation gates

## Fixed task and independent sources

Selected unknowns are32×48 apparent colour-temperature and apparent surface-density fields, not physical ground truth. Acquisition is real COBE DIRBE ZSMA100/140/240μm quoted brightness at native sky centres over Taurus. The standard method is the **unchanged original monochromatic PTS GreyBody.fit** with fixed beta2 and opacity.192m²/kg, applied independently to every sky sample. No synthetic observations are generated. PTS acquisition/model application to DIRBE is new; this does not claim reproduction of the paper's EAGLE sample or a published DIRBE-specific reconstruction.

Original author repository https://github.com/SKIRT/PTS, exact remotely verified commit ce9d0f8f57cc27e001072ac77bf8e7dd0f896ebb. Live GitHub response is privately archived in SKIRT__PTS_remote.json; full unmodified checkout and zip are outside this task. License AGPL-3.0 from source LICENSE.txt is retained as task LICENSE, and source attribution appears in local mathematics. NASA observations and calibration are publicly distributed scientific products; their exact download records and hashes are archived, not scraped from third-party reconstructions.

Paper: Camps et al., *Far-infrared and dust properties of present-day galaxies in the EAGLE simulations*, MNRAS4621057–1075, DOI10.1093/mnras/stw1735. Exact obtained19-page arXiv1607.04402 manuscript from https://arxiv.org/pdf/1607.04402 is private camps2016.pdf; SHA256 04e90018a61bca2f291417a874023ca8e206846ef28e739d999f77311c3138b5. Full extracted text camps2016.txt was read, especially Section3.1 Eq5, its Planck/opacity law, fixed beta2, Cortese opacity.192 versus Zubko.330, and the caveat on inferred dust masses. The paper's cited Cortese recipe and galaxy result numbers are not mislabeled as this task's results. PTS GreyBody is the selected original author's explicit source fitting implementation. The paper's flux law and source are consistent; the application to quoted broad-band DIRBE data requires an explicitly documented monochromatic approximation.

Private archive root: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth`. `dirbe_archive_index.json` records exact bytes and SHA256 for the paper, complete source archive, live revision, original capture script, independent outputs, data products, calibration, commands, logs and actual notebook environment. No PDFs/checkouts/capture scripts are shipped in the task. NASA source URLs are in private dirbe_downloads.json and dirbe_calibration_downloads.json. Bound public NPZ SHA256 d89060cfea98f0dbbc6b7d4cca9170ed3e8d33ca23bd7bccd098c3482ddf3e4e.

Original data are DIRBE_BAND08/09/10_ZSMA.FITS Resid and StdDev columns. Sky centres are DIRBE_SKYMAP_INFO.FITS GLON-CSC/GLAT-CSC. Private cKDTree nearest-centre selection is target-independent; maximum angular deviation0.225290638degrees. Source rows are preserved; no intensity interpolation, masking or positivity-based sample removal occurs. All chosen measured intensities and relative uncertainties are positive. Native float32 radiances are promoted without invented precision. Public instrument response is the November1996 Pass3B table from NASA; arrays contain the union of the three complete nonzero supports. Beam0.7°×0.7°, native quadrilateralized-spherical-cube grid393216pixels. Reference values never enter calibration selection.

## Original execution before local implementation

Private capture_dirbe.py loads **unmodified** source core/basics/greybody.py directly via importlib, bypassing unrelated toolkit imports. This is an isolated oracle role; no cleaned source/test/notebook imports PTS or private oracle scripts. It calls original GreyBody(1.,2.,.192).fit on every exact bound observation with Tinit20,Minit.01,sigma=weekly scatter,maxfev5000. Numerical one-pc coefficient is converted to surface density M*Msun/pc². This unit transformation assumes no actual target distance. Original exp overflow warnings occur during unconstrained LM trial steps; original and local final arrays are finite, positive and stable. Warnings are retained in logs rather than suppressed. `dirbe_oracle_command.json` records actual subprocess exit0; PowerShell warning wrapping was avoided by direct subprocess capture. Original environment is the shared standard numerical stack plus private astropy for FITS reading, not an installed PTS domain library. Source Python3.11.11, NumPy2.4.6,SciPy1.17.1, CPU1thread, no GPU. Original runtime 0.936503s. Stage arrays for six wavelengths and three temperatures/masses were independently frozen from original Bnu/greybody before local extraction.

Tolerances were fixed before judging local outputs: source Planck/flux rtol2e-14; complete fitted parameters/predictions rtol1e-8, with small absolute tolerances1e-10parameter and1e-8radiance. Maximum fresh T discrepancy 0K and density discrepancy 8.6736173798840355e-19kg/m². Standard-method hidden reference is original fitted fields; neither it nor the fresh local fit is actual physical truth.

## Equation/function/test traceability

| Required process | Paper and original source | Local executable function | Independent validation |
|---|---|---|---|
| Planck radiance and historical constants | Eq5, greybody.py Bnu | physics_model.planck | frozen original values; Rayleigh–Jeans limit |
| Opacity power law and thin emission | Eq5, greybody.py greybody | physics_model.flux/forward | frozen original flux; density linearity |
| Weighted nonlinear fit and positive guard | GreyBody.fit | solvers.fit_pixel/reconstruct | full original1536pixel pipeline; positive guard |
| Temperature-column Jacobian | analytic derivative of Eq5 | physics_model.jacobian | independent central finite differences |
| Quoted broad-band calibration | NASA DIRBE colour-correction definition | physics_model.bandpass_prediction | narrow-passband limit; measured response diagnostic |
| Absolute quantity scale, two-field scoring | output contract | standalone judge.compare/evaluate | independently known metric vectors; malformed/evaluator/foreign-cwd tests |

## Results, limitations and comparisons

Original results, fresh cleaned results and historical paper results are separately identified. Hidden evaluation/baseline_reference.npz stores original-method T and Sigma, and notebooks/reference_outputs/upstream_measurements.npz stores those identified outputs and original predicted radiance for notebook comparison. test/fixtures/upstream_pipeline.npz is consumed by full-pipeline parity; upstream_stages.npz is consumed by Planck/flux stage tests. No ground_truth.npz is created because no actual physical temperature/column truth is available. The standard source fit and all seven beta sensitivity reconstructions are computed actively in the notebook, with no cache substituted for fresh computation.

Fresh monochromatic residual RMS 1.86941749509MJy/sr, standardized RMS 0.702889505007. This standardized RMS averages three channel residuals per pixel, **not** reduced chi-square (only one nominal residual degree of freedom). Same fitted fields through the full measured passbands give standardized RMS 5.17233340386; colour-factor ranges [[0.9156380498202946, 0.9411088181989582], [0.9205226109228802, 0.9899298155155408], [0.9578477489474264, 1.0302941009201843]]. These nonzero, structured differences explicitly expose known broad-band bias. The diagnostic is not relabeled as the selected original method and is not used to claim physical truth. Correlated sky samples, possible temperature mixtures, opacity uncertainty and single-temperature modeling remain limitations.

Thresholds NCC>=.999999 and dynamic-range NRMSE<=1e-5 apply separately to both required fields. Nonzero tolerance reflects reproducibility rather than self-comparison0; measured local/source differences above are far smaller. Zero maps fail; malformed shape/dtype/key/complex/object/NaN/Inf inputs return evaluator error with empty metrics; missing hidden assets and report-write failure are checked. No hidden crop, rescaling, centering or coordinate alignment.

## Actual commands and environments

```json
[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\dirbe_dust_colour_temperature_density",
    "runtime_seconds": 1.4095386999979382,
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
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\dirbe_dust_colour_temperature_density",
    "runtime_seconds": 0.14082590000180062,
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
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\dirbe_pytest"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\dirbe_dust_colour_temperature_density",
    "runtime_seconds": 2.249465999993845,
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
      "notebooks/dirbe_dust_colour_temperature_density.ipynb",
      "--output",
      "dirbe_executed.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth",
      "--ExecutePreprocessor.timeout=180"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\dirbe_dust_colour_temperature_density",
    "runtime_seconds": 26.70701640000334,
    "exit_code": 0
  }
]
```

All four commands exited0: standard CLI, independent Judge,14tests, full active nbconvert. Notebook execution timeout180s is comfortably above the actual 26.707s including seven full-field sensitivity refits. Actual notebook kernel (not merely launcher) printed:

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

CPU single-thread execution validated. Runtime uses only NumPy,SciPy,Matplotlib; no PTS, astropy, HEALPix or other domain packages are imported by source/generator/Judge/tests/notebook. Private FITS extraction and original oracle dependencies are outside runtime. pytest uses private --basetemp due Windows default temporary-root permission behavior. No per-task environment installs/upgrades occurred.

## Review gates

Passed: exact paper/revision/data and equation mapping; original exact-bound-data execution before extraction; independent stage/full parity and applicable physics; constants/units/coordinates; fresh CLI and Judge; malformed/adversarial/evaluator contracts; solver-visible sufficiency with all data keys and complete metadata/output contract; no hidden target/calibration/GitHub URL leak; local explicit mathematics and no forbidden imports;14tests; active complete six-chapter notebook and computed comparisons; separate metric/result provenance; semantic review of both plans against equations, optimizer parameterization and callable defaults/shapes; inventory/license/archive cleanup. Scientific visual inspection is recorded separately in private dirbe_visual_review.json before final parent handoff.

Every figure has private PNG/SVG/PDF and numeric_sources.npz in figures/dirbe_dust_colour_temperature_density; source arrays have the figure name as prefix and are consumed by that figure. Canonical executed notebook retains outputs. Temporary executed notebook remains private. No redundant screenshots or unconsumed data are inside the task. Hidden plan/source/runtime is distinct from solver-visible README/raw observations/requirements.

## Independent rerun and numerical tolerance audit

Original unmodified capture and clean CLI were independently re-executed in new Python processes; both exit0. Full private commands/logs/arrays in dirbe_independent_rerun.log/.npz, dirbe_clean_independent_rerun.log/.npz and dirbe_tolerance_reruns.json. First and second local outputs each compared against the independently rerun original arrays, rather than self-comparing one saved method reference. Deterministic CPU reruns agree to the measured errors below. Since exact reproducibility does not imply a meaningful positive numerical budget, float32 serialization then promotion was separately measured as a precision stress (not an accepted dtype or a new standard result). The unchanged thresholds allow modest differences beyond this quantization floor while remaining orders of magnitude below spatially wrong/uniform/zero maps; all three adversarial submissions actually return exit1. No tolerance was relaxed. The absolute per-quantity scale is retained.

```json
{
  "reconstruction": {
    "first_clean": {
      "ncc": 0.9999999999999998,
      "nrmse": 0.0,
      "max_absolute_error": 0.0
    },
    "second_clean": {
      "ncc": 0.9999999999999998,
      "nrmse": 0.0,
      "max_absolute_error": 0.0
    },
    "float32_roundtrip": {
      "ncc": 0.9999999999999994,
      "nrmse": 6.566151967485231e-08,
      "max_absolute_error": 9.515425887229867e-07
    },
    "zero": {
      "ncc": 0.0,
      "nrmse": 2.1235007389153995,
      "max_absolute_error": 22.16882555773382
    },
    "uniform": {
      "ncc": 0.9988031867915024,
      "nrmse": 0.10386055041590955,
      "max_absolute_error": 5.439813208425154
    },
    "spatially_reversed": {
      "ncc": 0.9980816263237313,
      "nrmse": 0.1315327855421954,
      "max_absolute_error": 5.748512232784293
    },
    "nrmse_threshold_over_float32_roundtrip": 152.2962010248736
  },
  "surface_density": {
    "first_clean": {
      "ncc": 0.9999999999999999,
      "nrmse": 1.9087426452506302e-17,
      "max_absolute_error": 8.673617379884035e-19
    },
    "second_clean": {
      "ncc": 0.9999999999999999,
      "nrmse": 1.9087426452506302e-17,
      "max_absolute_error": 8.673617379884035e-19
    },
    "float32_roundtrip": {
      "ncc": 0.9999999999999999,
      "nrmse": 4.229880828174432e-09,
      "max_absolute_error": 2.2369140047218128e-10
    },
    "zero": {
      "ncc": 0.0,
      "nrmse": 0.17234571603979335,
      "max_absolute_error": 0.006151908229670973
    },
    "uniform": {
      "ncc": 0.8379543078790911,
      "nrmse": 0.09405601901739151,
      "max_absolute_error": 0.0052739008196538125
    },
    "spatially_reversed": {
      "ncc": 0.7832021903098576,
      "nrmse": 0.11348616901319591,
      "max_absolute_error": 0.0047727517616113135
    },
    "nrmse_threshold_over_float32_roundtrip": 2364.1327985866415
  }
}
```


Parent visual revision: signed temperature/density difference scales symmetric about0 with minimum fallback; three colour-factor maps share.90..1.10range soK=1 is neutral. Complete active notebook re-executed29.486182699998608s, all14tests and CLI/Judge rechecked. Figures components/colour_diagnostic re-exported and visually confirmed with actual unchanged numeric sources. Final canonical NB SHA256 4883ad6d118117811a49ab886044400f208022bb80af260fad3b9f583b62151f.
