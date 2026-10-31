# Exact provenance, qualification and validation

## Fixed inverse, deduplication and data classification

Recover a32×64spatial dayside thermal brightness field from126unresolved secondary-eclipse flux samples, using the selected published three-eigencurve posterior. This is a spatial inverse, not scalar fitting/calibration/forward-only modeling. Existing exoplanet_imaging recovers companion images from coronagraphic speckle/image diversity; existing lopeg_doppler_brightness_mapping uses rotationally Doppler-shifted stellar line profiles. Neither shares this occulting stellar-limb acquisition plus planetary thermal brightness unknown. The159-task catalog existing_tasks.md/full.json was screened for eclipse/phase/occultation counterparts; none exists. Parent screened/approved distinct acquisition/unknown.

**Existing author simulation**, not real JWST measurements: Eigenspectra data/output_lightcurves/practice1p01renorm/tseries_03310_nm.csv, SHA256372823f5d0e38ddfb5f2fdc37a03cdb986063a1f7378b780a4115cb54d86a533. Original input CSV values are converted ppm to ratio by1e-6; times/uncertainties unchanged, no new noise realization or synthetic object is created. Source add_noise.py preserveInput=True and paper Section2.1explicitly identify noiseless time series with simulated NIRCam error bars. Wavelength3.31micrometres,126samples, spacing136.77324s, constant sigma3.6395766103691654e-5. No known true spatial map is assigned because the practice directory does not establish a unique original generator target. Public grids are legitimate requested coordinates, not target-derived calibration. No hidden map, seed, harmonic coefficients or source intensity values enter raw_data. Raw NPZ SHA256 4fbe5e055a46db76dabc66360f0a8cfaa56c4024006ab09d40601be34f2e52a3.

This task cleans the published single-band eigencurve reconstruction stage, not the subsequent multiwavelength k-means eigenspectra grouping or atmospheric retrieval. The selected upstream justafew function is a complete single-band degree2/3PCinverse. Original optional plotting includes light-curve/model and map comparisons, preserved here with richer residual/posterior/model-order comparisons. Full multiband clustering is a different output and is not claimed reproduced.

## Exact papers, author sources and rights

Author Eigenspectra https://github.com/multidworlds/eigenspectra remotely verified commit88e11837bb17640f4f29ff9b2ed255d2ab085f4d. Author forward https://github.com/tomlouden/spiderman remotely verified commit379a5b6d2daa7b365a7ee65bdc643a6c92151c5d. Remote response JSON, complete original zip/checkouts, exact selected files and qualified copies archived privately. Eigenspectra MIT source/data notice and emcee MIT notice retained in NOTICE; SPIDERMAN GPL3fulllicense retained as LICENSE, local derived mathematical implementation under that compatible combined distribution. Existing simulation is supplied by author repository with its MIT license and no separate data exclusion identified. No upstream binary/domain package is shipped.

Actual PDFs read, not DOI-only assertions: Mansfield et al.2020Eigenspectra, DOI10.1093/mnras/staa3179, arXiv2010.02197v1,13-page6October2020preprint, SHA256900ff7d4e7a0036f30d05f052615ab2922c7bb2811324ad138b8bafeedf1cb3d; Rauscher/Suri/Cowan2018A More Informative Map, arXiv1806.05700v2,17pages1October2018revision, SHA2563e61d486a29a6193ba6db44dd8b1fa46c0322847972ca09a9d8ecf1b16d02c43; Louden/KreidbergSPIDERMAN, DOI10.1093/mnras/sty558, arXiv1711.00494v1,15-page3November2017preprint (its template says MNRAS2016and unfinished acceptance placeholders), SHA256a5f508367ec8dc69066d69d0d1f71e68c1c6fa63dfbefaad2512548ba7d49fe6. These exact archived preprints, not invented final journal versions, supplied the read content. Sources https://arxiv.org/pdf/2010.02197, https://arxiv.org/pdf/1806.05700, https://arxiv.org/pdf/1711.00494; text/PDF hashes and page counts in eclipse_papers.json. Read MansfieldSections2.1/2.2for simulated acquisition, degree2PCA, posterior means/uncertainties and50taucriterion; Rauscher eigencurve covariance/uniform term/transform derivation; LoudenSection2.1equal-area sectors and geometric overlap. The original data identify the selected acquisition; no paper figure is substituted for the run.

## Original-first gate and necessary qualifications

The exact author light-curve basis and full selected eigencurves_justafew inverse were first executed before cleaned task creation. The supplied original1000steps/200burn-in configuration produced independent successful chains but retained only20–30tauand was not accepted as adequate. These actual short runs and all failed runs remain in eclipse_before_memory_fix and eclipse_original_short_chains. The original main eigencurves.py contains floor-division in a model-selection diagnostic; we explicitly select the author's separate fixed3PCjustafew branch, whose Gaussian likelihood is correct. No silent repair of that diagnostic or substituted inverse is claimed.

Unmodified C source has identified memory errors: c_src/web.c declares an unused outer coords pointer, shadows it in the active loop, and frees the uninitialized outer pointer twice. The minimum private correction initializes this unused pointer toNULL; free(NULL)is defined and no operative geometry changes. _web.c call_map_model decrements borrowed bright_obj after separately decrementing its newly acquired converted array; removing the borrowed decrement corrects ownership. A temporary compiler-initialization flag and caller reference compensation were investigated, but final qualified build uses explicit corrections with ordinary-O2. Final original/qualified file hashes:

```json
[
  {
    "original": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\spiderman_memory_fix\\web.c",
    "qualified": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\spiderman-379a5b6d2daa7b365a7ee65bdc643a6c92151c5d\\c_src\\web.c",
    "original_sha256": "e51e13195ea5936093755d5b7003eaad69e8da90f0e544bb40af6c9db3d51019",
    "qualified_sha256": "97577ed625be8ff4edc9e6d9afc35e7896c12a89ac7a144642c363d97162fb01"
  },
  {
    "original": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\spiderman_memory_fix\\_web.c",
    "qualified": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\spiderman-379a5b6d2daa7b365a7ee65bdc643a6c92151c5d\\c_src\\_web.c",
    "original_sha256": "af1cef1d64699e652953d752b40b7b85fd3bd200d9246164b469d3f4b75fb2bd",
    "qualified_sha256": "163404e419f91ddb269e007f847c0cb67f44f98bc11bdd5341e4278ffe4eb228"
  },
  {
    "original": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\spiderman_memory_fix\\eigencurves_justafew_original.py",
    "qualified": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\eigenspectra-88e11837bb17640f4f29ff9b2ed255d2ab085f4d\\eigencurves_justafew.py",
    "original_sha256": "caf4577111db57c3a7a7491ac24c5a417bae56c62d4ddcf3bbb95833e664a38b",
    "qualified_sha256": "da08e0541e744f3dd7c1a58f23707ecf7b384ba4c34e415bb029e6acbe5a5b7a"
  }
]
```

The supplied setup.py unused numpy.distutils import is handled by import-only placeholders under Python3.12/NumPy1.26; unused legacy plotting/FITS/BATMAN imports are import-only placeholders and no scientific call uses them. The selected spherical model calls no BATMAN/stellar FITS routines. Build command, actual exit0, compiler warnings and runtime environment are eclipse_spider_build_command.json/log and eclipse_c_oracle_environment.log. C worker uses private WSL Python3.12.3,NumPy1.26.4,SciPy1.13.1,Matplotlib3.9.4; Windows Python inverse uses Python3.11.11,NumPy2.4.6,SciPy1.17.1,emcee3.1.6. NumPy2rejects assigning a one-element percentile array to a scalar, so the qualified Python source adds item()at that assignment. The chosen longer configuration is32walkers4000steps400burn-in; this changes sample count only. Failed WSL long translation runs with segmentation/illegal-instruction errors are retained, not hidden. The final inverse runs the original selected Python function in Windows, with every original sh_lcs call freshly delegated to the actual author C forward in a separate private WSL process. After sampling, a separate original C worker evaluates the posterior-mean spatial map and predicted light curve. No cleaned function supplies these oracle arrays, and no cached paper/notebook forward is substituted.

This is **qualified original reproduction**, not unmodified-source reproduction. Minimum memory/compatibility/configuration edits, process dispatch and all source hashes are explicit. Independent local angular integration and analytic harmonic values agree with actual original C outputs: all17basis curves max difference 6.3060667798708891e-14; original C map versus local analytic harmonic evaluation max 8.6736173798840355e-19. Independent two-circle area/rotation/visibility tests further check physics, so numerical agreement is not two copies of a known wrong formula. Normal tests consume frozen numeric arrays; no private oracle imports occur in source/Judge/tests/notebook.

Original long-chain commands are eclipse_windows_original_seed20261022/23/24_command.json, all actual exit0 with logs. Original numeric files are separate eclipse_oracle[_seed...].npz/json. Original basis/data/PCA/likelihood/least-squares/proposal/update/full-map fixtures were captured independently; cleaned code did not generate parity values. Every original chain was complete before final reference freezing. The source moves are emcee3.1.6default two-half StretchMove,a2,32walkers, Gaussian1e-5initial displacement, selected exact likelihood/priors. Local source explicitly implements the same transition rather than importing emcee.

## Traceability, stage parity and independent physics

| Paper/required stage | Author function | Local function | Evidence |
|---|---|---|---|
| Equal-area disk integration | SPIDERMAN web/lightcurve, generate_planet, blocked | cells,visible_areas,geometry,basis | all17original curves; independent circle overlap/partition/rotation |
| Real harmonic normalization/coordinates | SPIDERMAN spherical map | harmonics | original C posterior map at2048coordinates |
| Centered PCA/uniform separation | pca_eig.princomp | eigencurves/design_matrix | original eigenvalues/leading projector and native-design least-squares initializer |
| Gaussian likelihood/nonnegative c0,b | justafew.lnlike/lnprior/mpmodel | log_probability | original arbitrary nontrivial case; analytic Gaussian density/prior |
| Stretch proposal and Jacobian | emcee.StretchMove.get_proposal | stretch_proposal | frozen exact proposals/logJacobian |
| Red-blue ordering/acceptance | emcee.StretchMove.propose | ensemble_step | original complete accepted/rejected positions/logdensity/mask |
| Full selected posterior | eigencurves_justafew | reconstruct/sample_posterior | independent original/local long chains, analytic mean, map consistency |
| Spherical posterior transformation | justafew coefficient loop | spherical_coefficients | original posterior spatial map and low MC error |
| Contract/metrics/forward check | declared conditional model | standalone Judge | known vectors, actual poor maps, invalid/evaluator/foreign-cwd cases |

Stage tolerances fixed before scoring:17curveabsolute3e-12,leading3eigenvaluesrelative1e-9,leadingprojectorabsolute1e-8,original mapabsolute2e-17,native-design least-squares parameters1e-14,stretchproposalexact,fulltransitionpositions1e-16,logdensity1e-8. Covariance null modes are nonunique and not compared as arbitrary eigenvectors. Area integration preserves the selected400tilebrightnessdiscretization;64pointGausswithsine-squareendpointtransformationis independently verified against analytic circle intersection. The forward model is not silently replaced by continuous-surface/point-grid integration. An added initializer check investigated an8.45e-9flux stopping difference from matrix versus sequential summation and arbitrary PCA signs; fitted residual norms agreed within4e-13. Local initialize_parameters now preserves source summation and exactly replays the original frozen native design at1e-14parameter tolerance. Fresh-gauge initializer equality is not substituted for separate forward/PCA/full-posterior gates. Full statistical comparisons use separate seeds and analytic Gaussian posterior, not deterministic matching-seed equality.

## Measured convergence, nonzero tolerance and observation consistency

All original and three independent local long-chain measurements:

```json
[
  {
    "kind": "qualified_original",
    "seed": 20261022,
    "retained_steps": 3600,
    "maximum_autocorrelation": 51.85676067661182,
    "minimum_effective_samples": 2221.5039755068415,
    "length_over_max_tau": 69.4219992345888,
    "maximum_analytic_mean_error_mcse": 2.525382491703467,
    "fresh_reference_nrmse": 0.0010293467210332845
  },
  {
    "kind": "qualified_original",
    "seed": 20261023,
    "retained_steps": 3600,
    "maximum_autocorrelation": 53.96586859188943,
    "minimum_effective_samples": 2134.682587455166,
    "length_over_max_tau": 66.70883085797394,
    "maximum_analytic_mean_error_mcse": 0.8995372635593565,
    "fresh_reference_nrmse": 0.0007946925250821759
  },
  {
    "kind": "qualified_original",
    "seed": 20261024,
    "retained_steps": 3600,
    "maximum_autocorrelation": 56.87613022568663,
    "minimum_effective_samples": 2025.4542554650263,
    "length_over_max_tau": 63.29544548328207,
    "maximum_analytic_mean_error_mcse": 1.050139124044159,
    "fresh_reference_nrmse": 0.001065759683910108
  },
  {
    "kind": "fresh_local",
    "seed": 20261025,
    "retained_steps": 3600,
    "maximum_autocorrelation": 51.37650394673166,
    "minimum_effective_samples": 2242.270126426703,
    "length_over_max_tau": 70.07094145083447,
    "maximum_analytic_mean_error_mcse": 0.6979494429572974,
    "reference_nrmse": 0.0010293467210332845
  },
  {
    "kind": "fresh_local",
    "seed": 20261026,
    "retained_steps": 3600,
    "maximum_autocorrelation": 52.614298551830935,
    "minimum_effective_samples": 2189.5188792931485,
    "length_over_max_tau": 68.42246497791089,
    "maximum_analytic_mean_error_mcse": 0.8094502894321005,
    "reference_nrmse": 0.0006547349640056291
  },
  {
    "kind": "fresh_local",
    "seed": 20261027,
    "retained_steps": 3600,
    "maximum_autocorrelation": 49.93563419013842,
    "minimum_effective_samples": 2306.969799589536,
    "length_over_max_tau": 72.092806237173,
    "maximum_analytic_mean_error_mcse": 1.8147929213461884,
    "reference_nrmse": 0.001487533942780105
  }
]
```

Each meets retained length>50largesttau and minimum ESS>1000. Original-chain means agree with the independent analytic Gaussian solution within measured Monte Carlo standard errors. Independent-ensemble-mean split Rhat values [1.016622714627217, 1.0162550458025394, 1.0173344459084566, 1.0162780880429376, 1.0285737928291414] use three separately seeded ensembles, each split in time; correlated walkers are not mislabeled as independent chains. Maximum pairwise map NRMSE among all6independent means is 0.0017686981259046313. Standardized observation RMS 1.9500407983367776 remains~1.95, reflecting selected3PCmodel mismatch despite noiseless inputs. It is not presented as physical noise fitting or true-map accuracy.

The reference is original seed20261022posterior mean, not truth. Boundaries NCC.9999,NRMSE.005and standardized observationRMS2.05were chosen from independent-chain variation, actual adequate chain length and meaningful poor-map failures. Nonzero tolerance is not1.1times self-error0; measured float32serializationfloor 2.4613887444247167e-08 is much smaller than stochastic mean variability. Every zero,uniform,spatially reversed,double-scale map returns validfail exit1; actual reports in eclipse_chain_stability.json. No threshold was relaxed after a failure. Six independent chains and analytic checks show that the .005allowance is nonvacuous against spatially wrong maps.

Judge uses exact entirefloat64(1,32,64)map, uncenteredcosineNCC and dynamic-rangeNRMSE. Its separate forward check uses public coordinate/measurement/sigma arrays and a frozen original **known-geometry**9harmonic observationoperator stored in baseline_reference.observation_basis, not a target-derived calibration. It projects the submitted map onto declared degree2space and estimates only the nonnegative-near1physical stellar offset (here unconstrained optimum is positive); no map offset/gain/alignment is applied. The public output contract explains this nuisance convention. All hidden arrays have consumers: baseline reconstruction/operatorJudge; upstream_measurements originalmap/prediction/coefficientsreview; upstream_posterior all3chains/means/maps/covariances/taus/lengths/walkersnotebook; upstream_stages and proposal/transitionfixturescomparingtests; cleaned_measurements freshnotebookmeasurement. No ground_truthfile is invented.

## Executed notebook, local environment and final review

Six-story canonical notebook actively reconstructs rawdata with cleaned modules, evaluates freshfile, displays original3chainmeans and freshmap/differences, computes first3eigenmaps/curves/covariancespectrum, actualresiduals and computeduniformcomparison, conditionalposteriorSD/correlation/traces/analyticmeans/ESS, and five actual1..5PCrefits. Paper's multiband clustering is explicitly outside this selected single-band reconstruction stage. Historical paper numbers are not claimed reproduced; all plotted fresh maps/predictions are computed. No missing comparison is filled with zero; uniform is an explicitly solved baseline. Optional original plotting is preserved conceptually and expanded scientifically.

Canonical notebook executes completely via nbconvert; actual kernel is shared research-runtime,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,inlinebackend,CPUone thread. Full kernel environment:

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

Commands/runtimes/exit evidence:

```json
[
  {
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\exoplanet_thermal_eclipse_mapping",
    "runtime_seconds": 1.0694651999947382,
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
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\exoplanet_thermal_eclipse_mapping",
    "runtime_seconds": 0.1395094000035897,
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
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth\\eclipse_pytest"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\exoplanet_thermal_eclipse_mapping",
    "runtime_seconds": 2.177092800004175,
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
      "notebooks/exoplanet_thermal_eclipse_mapping.ipynb",
      "--output",
      "eclipse_executed.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\groups\\astro_earth",
      "--ExecutePreprocessor.timeout=180"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\exoplanet_thermal_eclipse_mapping",
    "runtime_seconds": 19.126299900002778,
    "exit_code": 0
  }
]
```

All13tests pass including independent source stages, physics, sampler, metrics and Judgeadversaries. Main/Judge freshpass and fullnbconvertexit0. Importbackendtestpassed; no library module changes backend. Requirements are only NumPy/SciPy/Matplotlib sharedpins; no emcee,SPIDERMAN,starry,HEALPix or other domainlibrary enters cleaned runtime/generator/Judge/tests/notebook. Source implements task-specific geometry/harmonics/covariance/likelihood/priors/stretchproposals/acceptance/posteriortransformation/initialization/stopping. No cross-task imports. Public README/raw/environment alone define forward orientation, normalization, harmonics, selected3PCposterior, priors andoutput; no GitHubURLs,truth,privilegedseed,target-derivedcalibration or standalone metadataJSON leak.

Both plans and README were semantically reviewed against actual functions/signatures/defaults/returnshapes/units/normalizations/coordinate rotations/sampler update ordering/metrics. Frozen original/source-qualified references remain identified. Private figures D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth\figures\exoplanet_thermal_eclipse_mapping include six PNG/SVG/PDF groups and numeric_sources.npz; actual visual review is recorded separately after inspection. Temporary executed notebook stays private,canonical outputsretained. Final required provenance/source licenses andinventory conform. Archive root D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth; eclipse_archive_index.json hashes papers,originalarchives/live revisions,qualifiedsource,commands/logs/environments/data/numerics. Canonical notebook SHA256 363bc6fe7b9e846684d332417a7ed3701f72b28f725d27a129e479309cc125fc. Mandatory gates passed with explicit original-qualification exception; parent scientific acceptance pending.

Final figure5 correction: all3 complete original retained32-walker chains are stored and actively replayed with the identical ACF/Sokal estimator; no walker thinning or different burn is used. The plotted diagnostic is now retained length/tau, with an explicit50thresholdline; the companion panel shows exact whole-ensemble ESS. eclipse_figure5_consistency.json asserts every plotted maximumtau/minimumratio/minimumESS equals the final stability table. Original maxima51.8567606766,53.9658685919,56.8761302257; fresh51.3765039467. Their minimum length/tau69.4219992346,66.7088308580,63.2954454833,70.0709414508; minimum ESS2221.5039755068,2134.6825874552,2025.4542554650,2242.2701264267. The two duplicate initializer investigation paragraphs were merged into one. Full13tests/main/Judge/NB reexecuted successfully and all figure formats/numerics reexported.
