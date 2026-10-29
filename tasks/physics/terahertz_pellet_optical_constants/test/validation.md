# Author provenance, independent verification and final gates

## Fixed physical problem and original sources

The unknowns are real refractive index n(f) and signed apparent absorption α(f) for each whole PE and lactose/PE pellet. Both independent real quantities, both actual materials and all selected1256 frequency samples are required. The acquisition is normal-incidence transmitted primary pulses measured relative to an empty path; source pulse and noise calibration are not the inverse unknowns. No independently measured true optical spectrum exists. evaluation/baseline_reference.npz is an original standard-method result, never ground truth.

Primary publication: Chi Ki Leung, Jasper N. Ward-Berry, Elena Wanvig i Dot, Jongmin Lee and J. Axel Zeitler, Tutorial: Accurate Determination of Refractive Index and Absorption Coefficient in Terahertz Time-Domain Spectroscopy, Journal of Infrared, Millimeter, and Terahertz Waves46,59(2025), published18August2025; https://doi.org/10.1007/s10762-025-01070-8 . The full primary paper was read via https://link.springer.com/article/10.1007/s10762-025-01070-8 and archived privately as thz_paper.html. Selected method is the paper's common-window and reference/sample versus baseline slab extraction (sections3,5 and whole-mixture optical extraction in6). It retains all defined output quantities; pure constituent effective-medium inference is a different inverse output and explicitly outside this task. Phase method primary source: Jepsen2019 https://doi.org/10.1007/s10762-019-00578-0 .

Original code: https://github.com/dotTHzTAG/thzpy/tree/17634ed0e4e39b0398c26b30445441c9f0b8a20a . Exact remote commit was verified with API+codeload archive before execution; sources.json records it. thzpy.zip and complete unmodified source remain private. Code LICENSE is LGPL-3.0 (not an earlier inferred MIT label); complete license and explicit THzPy adaptations are preserved in the task LICENSE and source headers. Author deposited full figure code abscoeff_thzpy_publ.py is separately archived and attributed; it is not imported by runtime.

Data: https://doi.org/10.17863/CAM.120263.2 , Apollo version2 https://www.repository.cam.ac.uk/items/e8f8d203-30a8-4658-8fce-85422f5a71c8 . Its rights field explicitly statesCC BY4.0, archived as thz_data_rights.html. Attribution is Leung and coauthors, Cambridge Apollo. This version replaces the author's erroneous older LMPE_publ.thz; only corrected LM05_PE95_publ.thz is bound. Direct original data download: https://api.repository.cam.ac.uk/server/api/core/bitstreams/5eb53df8-49c9-4026-8b38-b8169ec29583/content . Author analysis download: https://api.repository.cam.ac.uk/server/api/core/bitstreams/63d2db8f-f887-4c32-8e95-757d51d1b31b/content .

All3008 native float64 time/field samples from datasetsds1mixture,ds2PE,ds3air are preserved. NPZ packaging only separates time and field axes, with leadingbatch1. No crop, synthetic object, downcast, denoising, target normalization or hidden-target calibration occurs. Micrometer thicknesses2.65/2.56mm are actual acquisition information. Metadata from HDF5 records2025-05-22, TeraSmart/Menlo Systems transmission, mixed pelletmass299.8mg, reference285.1mg; nominal paper5wt% and13mmdiameter and1000averages are explicitly distinguishable from more precise file masses. Data slight time-coordinate differences remain public rather than assuming identical original scans.

## Independent original run before local extraction

Private root: D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\batch3\astro_earth_physics . Archive index: thzpy.zip; archives/thzpy-17634ed0e4e39b0398c26b30445441c9f0b8a20a/; LM05_PE95_publ.thz; abscoeff_thzpy_publ.py; capture_thz.py; thz_oracle.npz/json/log; thz_acceptance_before_extraction.json; thz_paper.html; thz_data_rights.html. The exact original private command was oracle-env/Scripts/python.exe capture_thz.py, working in that root. Actual original return0, no stderr, full intermediate/window/frequency/amplitude/phase and final optical values frozen. Original repeat independently executes uniform_slab again and max difference0. Original capture precedes build_thz.py/local math. The capture imports only original THzPy, HDF5 reader and NumPy; it never imports local task code. Raw .thz/HDF5 reading is maintainer-only; cleaned input is strictly NPZ and no h5py dependency is shipped.

Private oracle CPython3.11.11 Windowsbuild26200, NumPy2.4.6, h5py3.16, archived THzPy commit; oracle-env isolated from clean runtime. Shared clean runtime CPython3.11.11, NumPy2.4.6, Matplotlib3.11.1. CPU and numerical threads1. Exact timing, hashes and parameters are in thz_oracle.json below. No domain library or new dependency was installed in shared clean environment.

Original settings fixed before local extraction: Hann15ps half-width, FFT padding exponent3, inclusive frequencyband.2..2.5THz, phasefitcenter.3THz withwidth.2THz, thickness2.56mmPE/2.65mmmixture, na1. Upstream parity acceptance fixed in thz_acceptance_before_extraction.json: stagesrtol/atol1e-12, complete optical spectra rtol/atol1e-11; independent analytic roundtriprtol1e-12. Fixture expectations came from original code, not copied local calculations.

```json
{
  "code_url": "https://github.com/dotTHzTAG/thzpy/tree/17634ed0e4e39b0398c26b30445441c9f0b8a20a",
  "paper": "https://doi.org/10.1007/s10762-025-01070-8",
  "data": "https://doi.org/10.17863/CAM.120263.2",
  "code_license": "LGPL3",
  "data_license": "CC BY4.0",
  "raw_shape": [
    3,
    3008,
    2
  ],
  "raw_dtype": "float64",
  "window_shape": [
    3,
    2,
    1025
  ],
  "frequency_shape": [
    1256
  ],
  "frequency_first": 0.20141601708893048,
  "frequency_last": 2.4993896666035464,
  "frequency_spacing": 0.0018310547008084654,
  "sample_thickness_mm": 2.65,
  "reference_thickness_mm": 2.56,
  "settings": {
    "min_frequency": 0.2,
    "max_frequency": 2.5,
    "upsampling": 3,
    "peak_snr": 0.3,
    "fit_range": 0.2,
    "all_optical_constants": true
  },
  "negative_absorption_counts": [
    276,
    157
  ],
  "repeat_maxabs": 0.0,
  "runtime_seconds": 0.025229600025340915,
  "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
  "numpy": "2.4.6",
  "os": "Windows-10-10.0.26200-SP0",
  "hashes": {
    "LM05_PE95_publ.thz": "80a4bfb6c6b209dad342db63d20d32d2a4784c525a3d7ed27868c10127d36b98",
    "abscoeff_thzpy_publ.py": "989fe950cdd02725bc4839907f2bbd2b17db0db42ddfdb6e5c1f354dee4bd259",
    "thzpy.zip": "bf6e5423be5829392f28b0b2843bb7a399704740c5c013b431b9d075984dc09e",
    "thz_oracle.npz": "2cade8e633b922c4d89e1de1f78142b5ed474ef9b3230596671e0293d639b739"
  }
}
```

## Equation-to-code-to-case traceability

| Equation/stage | Original implementation | Local implementation | Validation |
|---|---|---|---|
| Paper section3 equal half-width and preserved delay | timedomain/timedomain.py window,common_window; timedomain/_timedomain.py _symmetric_window | preprocessing.common_window | original complete window fixture; independent relative-delay pulses |
| FFT ratio, frequency axis | transferfunctions/_transferfunctions.py _transform | solvers.transform | full1256frequency/amplitude fixtures; analytic delayed Gaussian |
| Referenced phase unwrap, fitted2πinteger offset | transferfunctions/_transferfunctions.py _unwrap | solvers.transform | full1256phase fixtures; independent known delay sign |
| Single-pass Fresnel transmission and optical inversion | transferfunctions/_transmission.py _uniform_slab; transferfunctions/transferfunctions.py uniform_slab | physics_model.slab_inverse andslab_transfer | original both-material/both-quantity full parity; analytic .96interface amplitude andunits roundtrip |
| Unit conversion, both optical outputs | frequencydomain/_frequencydomain.py _all_optical_constants | solvers.reconstruct, physics_model.slab_inverse | originaln/alphafullarrays; independentm^-1tocm^-1check |
| Standard evaluation | independently specified benchmark metric | Judge.score andvisualization.metrics | known numerical cases; each independent material/quantity poor prediction |

The model coefficients and phase conventions are semantically checked. Field attenuation usesαd/2, notαd; outputαcm^-1 is .01times internalm^-1. Physical n+iκ convention and negative-exponentialFFT delay sign are explicitly distinguished. The original intentionally uses largest absolute peak for windowing but positive maximum for phase ramps; both preserved and tested. Common-window interpolation is by original sample index, not silently replaced by a supposedly better interpolant. No upstream mathematical correction is claimed. NumPy least-squares rcond=None is an explicit equivalent convention tested on exact data. A divide-by-zero warning in an analytic unit test arises only at excluded far-frequency FFT bins of a Gaussian; all retained analytic and measured spectra are finite, and no epsilon hides the denominator. Baseline is not poor merely because measured apparent α is slightly negative: this is a declared model limitation, retained with sensitivity analysis rather than passive-material accuracy claims.

## Assets, measurements and metric definitions

Public input NPZ contains only time_ps andwaveforms, both(1,3,3008)float64. test/fixtures/upstream_stages.npz contains originaloutput_windowed,output_frequency,output_amplitude,output_phase, consumed by comparing_test/test_parity.py. evaluation/baseline_reference.npz contains both originalreal_index andabsorption_coefficient, consumed by full parity and standaloneJudge. notebooks/reference_outputs/upstream_measurements.json is consumed in the active notebook to identify original runtime/provenance. No actualtruth array is invented. Full observations remain public, and no fixture/reference is imported by solver.

NCC is uncentered flattened cosine. NRMSE is RMS difference divided by each reference spectrum's dynamic range, independently for each of2materials and2quantities. Every one of8metricconditions must pass. NCC>=.999999 andNRMSE<=1e-6 are finite nonzero numerical tolerances, justified by original repeated equality and generic precision; they are not a vacuous allowance for poor physical accuracy. The tiny dynamic range of n makes NRMSE necessary even though cosine alone is insensitive. No alignment, scale fitting, magnitude coercion or clipping is allowed. Fresh cleaned spectra agree exactly with the original in this backend; floatingNCC can exceed1by2e-16 due roundoff and is reported rather than altering the declared formula.

The canonical executed notebook actively reconstructs full bound data with cleaned modules, identifies original separately, evaluates fresh arrays, and actually computes window/boxcar/calibration sensitivity. All five finalfigures were exported asPNG/SVG/PDF and actually opened for inspection; titles,labels,units,legends andnegative apparent absorption are legible without overlap. Numerical_trace.npz contains every plotted raw/fresh/original/diagnostic/variant array; fresh_measurements.json retains full precision. Final visualcontract records the finalcomposition afterdraftlayout existed, beforefinalexports, rather than falsely claiming an earlier design date.

Scope departures from the larger tutorial are explicit: this task selects the two whole-slab optical properties, not pure-lactose Maxwell–Garnett constituent inversion, missing-baseline conventional approximations or alternate sample cells. The active review preserves relevant measured pulse comparison, PEversusmixture optical quantities, spectral phase/amplitude, apodisation and thickness dependence. No omittedmethod is replaced by zero, and no archived curve is called fresh. Forward residuals are algebraic checks using the same inferredquantities, not independent accuracy evidence. No historical paper metric is treated as this run.

Fresh measurements:

```json
{
  "fresh_runtime_seconds": 0.006975699914619327,
  "original_runtime_seconds": 0.025229600025340915,
  "agreement": {
    "real_index": [
      {
        "ncc": 1.0,
        "nrmse": 0.0
      },
      {
        "ncc": 1.0000000000000002,
        "nrmse": 0.0
      }
    ],
    "absorption_coefficient": [
      {
        "ncc": 1.0,
        "nrmse": 0.0
      },
      {
        "ncc": 1.0000000000000002,
        "nrmse": 0.0
      }
    ]
  },
  "negative_alpha_bins": [
    276,
    157
  ],
  "forward_max_amplitude_error": 2.220446049250313e-16,
  "forward_max_phase_error_rad": 2.842170943040401e-14,
  "thickness_one_percent_index_max_change": 0.004626423508197419
}
```

## Actual commands, contract tests and semantic review

The final standardCLI, standaloneJudge,14tests andcomplete nbconvert all returned0. Exactcommands/runtimes below; thz_final_0.log..thz_final_3.log contain actualsubprocessstdout/stderr. Timeout300sec exceeds measured4.5secnotebookruntime; privateJUPYTER_PATH and explicitimaging-researchkernel point to shared cleanPython. CanonicalNB retains executedoutputs with exactlyfive distinct embeddedfigures and no automatic duplicate. Only declaredreconstruction.npz andjudge_report.json remain underoutput.

```json
[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "runtime_seconds": 0.15498810005374253,
    "returncode": 0
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
    "runtime_seconds": 0.20962099987082183,
    "returncode": 0
  },
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "-p",
      "no:cacheprovider",
      "--basetemp",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\batch3\\astro_earth_physics\\tmp_thz_final"
    ],
    "runtime_seconds": 1.1514328001067042,
    "returncode": 0
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
      "notebooks/terahertz_pellet_optical_constants.ipynb",
      "--ExecutePreprocessor.timeout=300",
      "--output",
      "terahertz_pellet_optical_constants.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch3_20261020\\batch3\\astro_earth_physics\\executed"
    ],
    "runtime_seconds": 4.476824200013652,
    "returncode": 0
  }
]
```

14tests include exactoriginalstages/fullpipeline, analyticforward/inverseandunitconventions,relative delay,FFTphase,knownmetrics,finite/shape/dtype/key/filecontracts,poorcompleteandindividualquantity/materialoutputs,evaluatorassets/thresholds,truthpriority,foreignCWDandreportwritefailure. Tests use frozen numeric fixtures, not original imports. StrictNPZload usesallow_pickle=False andnofilenameguessing. JudgeJSON isUTF8,finiteandindependentofsolver. Headerconstants/READMEthicknessagreementtested. Allcoremathisliteral localexecutablecode; genericFFT,lstsq,unwrap,arrays are allowednumericprimitives. Matplotlib is visualizationonly. No forbiddendomain solver,checkpoint or cross-taskimport exists anywhere inruntime/tests/Judge/NB. No backendsetter exists inmodules/main import.

README andbothplans were reviewed semantically againstsource: inversionfactors,phasebranch,Fouriersign,windowrounding/padding/indexing,frequencyselection,thicknessunits,materialorder,datatypes,arraydimensionsandactualcallerflowagree. Closed-formmethod hasnohiddeniterativeoptimizerorunimplementedregularization. PublicREADMEfive-sectionstructure contains everyNPZkey/acquisitionparameter/output andnoGitHubdomain/hiddenreferencefile/targetseed. requirementscontainsactualpinnedruntimepackagesonly; pytest/Jupyter remain shareddevelopmenttools. Exactinventory follows template; noPDF,FITS,HDF5,upstreamcheckout,capturelogs,temporaryexecutednotebookorfinalfigureexport is shipped inside task. No synthetic generator is appropriate.

| Mandatory gate | Status |
|---|---|
| Primary paper, exact verified commit, licensed real data and hashes | passed |
| Independent original exact-observation run and deterministic repeat before extraction | passed |
| Upstream stage/full fixtures and independent physical checks | passed |
| Constants, units, signs, precision and coordinates | passed |
| Fresh CLI and complete strict Judge | passed |
| Poor/malformed/missing/nonfinite/evaluator/reportwrite/foreigncwd tests | passed |
| Public allowlist sufficient, no hidden-target or GitHub leakage | passed |
| Minimal clean CPU runtime and explicit local mathematics | passed |
| Full14test suite | passed, one documented excluded-frequency analytic warning |
| Full active nbconvert and computed scientific comparisons | passed |
| Metric/source distinction and thresholds | passed |
| README and mathematical/interfaceplans review | passed |
| Exact finalinventory, licenses and actualfivefigureinspection | passed |

## Private archive and task hashes

```json
{
  "task": {
    "LICENSE": "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118",
    "README.md": "2030a94066b4b21056ead3c56dbba3da6cfbc2c4294322a194a72dc5e5d967a0",
    "requirements.txt": "75618fbc20ccc5860a4243cda7dca3734342f9a46ac084a2d4c96164404ea787",
    "data\\raw_data.npz": "37c971b144e73a04b596e978c7a8dd2cf8d331cf5af8adebd7b4b59f3933012d",
    "evaluation\\baseline_reference.npz": "dc7cd357fff9e537812e60fa8931c9c370a55524cfd09f4c005d8c8bfa25b284",
    "evaluation\\judge.py": "67a949d1862dd177b23315ff91772081a6ec8f3786dc1d0527687cbc2b309173",
    "evaluation\\metrics.json": "f3524114fe5beb61419598be5eaf4df413138a4b85f161885147db80cba0d9fc",
    "notebooks\\terahertz_pellet_optical_constants.ipynb": "78ce7ece952422ffb08828d5a8ca6c274a8ce2bff008bbddddc7902dd4152e0f",
    "notebooks\\reference_outputs\\upstream_measurements.json": "33c8c2792ff7ed5b03c384a4f43b3968bb08ad99b854b31a955bdcefc49735d1",
    "output\\judge_report.json": "34223c8e4986bb183090b392177552cf08d04131277f34a874c4df82141ed789",
    "output\\reconstruction.npz": "dc7cd357fff9e537812e60fa8931c9c370a55524cfd09f4c005d8c8bfa25b284",
    "plan\\approach.md": "aaacb0a2cf646c5ea0460bc7d33cd0998fc8cf6cbfbd09f1d9bbb9f73777dd73",
    "plan\\design.md": "d64f15a19d4b4863f6814925e49950d2c702d58e51461908b90b518af9da3924",
    "src\\main.py": "2f48f814f7091aada03b59d38457b38e20810eac9a1babf6b2127b0d6faa4779",
    "src\\physics_model.py": "bcd39820c826b12070a4734328ffe3fed4dc989dadeb218064e907ed9dcf69d3",
    "src\\preprocessing.py": "96ac9c83f6ba89a7664b063eaf1b5e46d69938988766e94c6c25ab074b66e95e",
    "src\\solvers.py": "673d2bf7c79b95698844a181e11c9cef9d43b2706d3582662a3d630f980da84d",
    "src\\visualization.py": "60a5b641c2020f2c8837f5812ac31255e3314785c9146350acbf993a2b1dc3a5",
    "src\\__init__.py": "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
    "test\\comparing_test\\test_parity.py": "7b69f94c9b796dd5de8e04c0b34ada0072449c27c20258a97aff01669dde36d1",
    "test\\fixtures\\upstream_stages.npz": "4b8b68b8845d6800d48ee6b0f8e8c4caea902fe99846ac67c88580bbf9cb99f9",
    "test\\unit_test\\test_judge.py": "be9ba81a0abf58bce95ecd472a2587bab215b8f2203fe9c6a7ef1bc9dcc82e21",
    "test\\unit_test\\test_physics.py": "c9d13824d698d90bfd4756569ec8f9489985630b244da06b84f116ea8c28533c"
  },
  "private": {
    "thzpy.zip": "bf6e5423be5829392f28b0b2843bb7a399704740c5c013b431b9d075984dc09e",
    "LM05_PE95_publ.thz": "80a4bfb6c6b209dad342db63d20d32d2a4784c525a3d7ed27868c10127d36b98",
    "abscoeff_thzpy_publ.py": "989fe950cdd02725bc4839907f2bbd2b17db0db42ddfdb6e5c1f354dee4bd259",
    "capture_thz.py": "eb2272e4c238213661e06102d773857e2451320aaeb1044f28dc28824c2e6408",
    "thz_oracle.npz": "2cade8e633b922c4d89e1de1f78142b5ed474ef9b3230596671e0293d639b739",
    "thz_oracle.json": "33c8c2792ff7ed5b03c384a4f43b3968bb08ad99b854b31a955bdcefc49735d1",
    "thz_paper.html": "32ed63159c77e21ee19ca1b9aa3213ccf0218eb59539560b132a8e68ef0e18ea",
    "thz_data_rights.html": "bc66e7a891f082c6efffcfdb89d9c63032926f8f8e5611bbf4a06c0fcc6bdcda",
    "thz_acceptance_before_extraction.json": "54336184cbb834fb4bce6b7bba491eeb2b57c56f549302b9c5270165eae10ed8"
  }
}
```

Task writes frozen for parentreview; parent independent acceptance is a separate gate, not claimed here.


## Independent parent acceptance

Passed: parent independently ran 14 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed all native waveforms for both materials and empty path, window/Fourier/phase conventions, Fresnel single-pass inversion and units. Negative apparent absorption is retained and model exclusions are explicit. All five actual figures, independent synthetic limiting checks and per-material/per-quantity Judge contracts were reviewed. Near-zero inverse-consistency is explicitly algebraic, not proof of material accuracy. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
