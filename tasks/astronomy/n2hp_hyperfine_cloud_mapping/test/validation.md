# Provenance and validation evidence

## Scope and completion status

Bound task: `n2hp_hyperfine_cloud_mapping`, astronomy, CPU, real IRAM 30m N2H+(1-0) observational parameter mapping. The released inputs contain a contiguous 3x3 patch of existing OphA spectra and legitimate molecular/frequency calibration. No new synthetic observations, simulated truth, target-informed calibration, trained weights or physical truth arrays are supplied. Nine fitted sightlines sample overlapping telescope beams, so this is a compact beam-scale inverse parameter imaging benchmark, not nine independent spatial resolution elements.

All mandatory gates for this bound task pass. The selected published method is constrained weighted hyperfine radiative-transfer least squares. Original MPFIT and cleaned generic TRF use the same forward expression, objective, initialization and physical parameter conventions; their internal optimizer trajectories are explicitly different. We do not claim bitwise MPFIT algorithm-step reproduction or execution of the full 4x4 neighboring-guess example.

## Source, paper and actual data

- Original code: https://github.com/pyspeckit/pyspeckit ; exact archived commit `92c69bfdad2980a915a53007b03d8910e10b6bba`. A new shallow clone was acquired by the parent coordinator, then copied to the private archive; `git rev-parse HEAD` confirms this revision and `git status --short` was empty. A later `git ls-remote ... HEAD` attempt failed to connect; no claim is made that this revision remains the latest remote HEAD. The oracle imports the unchanged source directly, as recorded by its `__file__`, rather than an installed similarly named distribution.
- Optional exploratory test repository: https://github.com/pyspeckit/pyspeckit-tests ; newly cloned revision `c11ed9b3747a7b5bc18a48340ca3c2f207cd8c85`. Its test spectra were inspected but are NOT the bound data and are not redistributed. No license is inferred for its independent data.
- Method paper: Ginsburg et al., AJ 163, 291 (2022), DOI https://doi.org/10.3847/1538-3881/ac695a . Archived paper is arXiv `2205.04987v1`; exact arXiv metadata response and PDF are private. Task-relevant sections: spectral models and nonlinear fitting, hyperfine model, Appendix C/C1-C2, and background-subtracted transfer expression D6. Published historical CLASS table values are not asserted to be measurements of this run.
- Data paper: Andre, Belloche, Motte & Peretto, A&A 472, 519 (2007), catalogue `J/A+A/472/519`. Actual source: https://cdsarc.cds.unistra.fr/ftp/J/A+A/472/519/fits/opha_n2h.fit ; catalogue description: https://cdsarc.cds.unistra.fr/ftp/J/A+A/472/519/ReadMe . The description explicitly identifies IRAM 30m on-the-fly real N2H+ cubes of six Ophiuchus cores, approximately 26 arcsec resolution, OphA 20 kHz spectral resolution. The data paper PDF endpoint returned HTTP 403; the available observational paper identity and data description are the archived CDS record, not an invented successful paper download. The selected method paper PDF was obtained and read.
- Original FITS shape `(1,501,30,21)`, float32, BUNIT=K, OBJECT=VLA1623A, ORIGIN=GILDAS-(LAOG-IRAM). `DATE-OBS=2007-07-04` and DATE/HISTORY indicate file-writing metadata, and are NOT claimed to prove the actual observing date. Core and acquisition claims come from the catalogue description. Original FITS SHA-256: `9f9331ab9fb240276dfdb5d8a5938b7063c824f58490813b7f81d6a8ff4b5065`.
- Bound brightness: `original[0,:,25:28,12:15]`, preserving all 501 channels. Original float32 values are promoted exactly to float64. Velocity is `((index+1)-403.0960083008)*(-62.84131109715)/1000+2.5` km/s; frequency is `93176265000*(1-velocity/299792.458)` Hz. No smoothing, interpolation, baseline fit, noise injection or target selection from cleaned results occurs.
- Molecular calibration is selected upstream J=1-0 constants, derived from laboratory frequencies/relative strengths, not estimated from a hidden cloud target. Explicitly preserve the upstream sum of strengths and component-dependent Doppler convention. Input arrays were appended to the observation-only NPZ after the independently bound original run; observation/frequency values are identical. Final NPZ and individual array fingerprints below remove ambiguity about package versus numeric data hashes.

## Data rights and source notices

Source mathematical code is MIT, copyright (c) 2010 Adam Ginsburg & Jordan Mirocha, preserved in full in the license section below. No original imaging package or full checkout is shipped. VizieR's archived terms state that retrieved data are free of usage in a scientific context with explicit original author/publication/publisher citation; commercial usage depends on origin. Data are not falsely relicensed as MIT or CC. The catalogue ReadMe has no separate copyright clause. README References and the migrated notice below attribute Andre et al., A&A, IRAM and CDS/VizieR, link the catalogue and terms, and distinguish code from data rights. Scientific benchmark use is the selected use context; this is not a representation of unrestricted commercial rights.

## Binding and rejected exploratory pixels

Before local source extraction, execute the full upstream example's central `(y=25:29,x=12:16)` 4x4 patch using identical fixed initialization on each sightline. Most pixels converge. The lower-left relative pixel `(3,0)`, original `(28,12)`, runs to Te=1835.6771396452025 K, t0=0.0043129718866144915 and MPFIT status 5 (iteration limit), indicating thin-limit degeneracy. Its scaled finite-difference Jacobian has condition number 146795.84750765143 and temperature/opacity derivative cosine 0.9999999920745907. The original `(26,15)` pixel reaches Te=80.60695678 K, t0=0.20906583 and condition 3243.1010. The central retained examples have substantially lower conditioning (e.g. 476.6890 at relative `(0,0)`). The complete 4x4 spectra, original models, parameter arrays, statuses/logs and numerical condition evidence remain under private `exploratory_4x4/`, `exploratory-4x4.log` and `identifiability.json`. The initial exploratory stdout was replaced when the bound 3x3 capture was finalized; the complete private 4x4 archive is an explicitly repeated unchanged-original run made later to preserve the exploratory evidence. The bound 3x3 original fixtures and pre-extraction acceptance file were retained separately throughout. The condition/Jacobian analysis is additional later original-model analysis, not misdated pre-extraction evidence.

Fix the contiguous upper-left 3x3 patch **before implementing local equations** to evaluate all four parameters on finite-opacity, converged standard fits. This is a physical identifiability/data-curation decision, not a choice based on cleaned agreement. Bound source coordinates are y=25,26,27 and x=12,13,14, zero-based. All nine original fits terminate with MPFIT status 1. No method or output quantity is discarded. We explicitly do not claim full 4x4-example reproduction.

## Original executable run and frozen criteria

Private isolated environment: Python 3.11.15 on Windows 10.0.26200, NumPy 1.26.4, SciPy 1.12.0, Astropy 6.1.7, Matplotlib 3.8.4; full dependency versions and numerical backend in `oracle-environment.txt`/`oracle-backend.json`. Installed pyspeckit distribution 1.0.4 is present only in the isolated environment; the capture prepends the archived source and records its actual module path, so the executed method is the exact archived commit. Domain dependencies are absent from the cleaned runtime. Later PDF inspection utilities are recorded in the oracle environment, not public requirements.

Original command from repository root:

```text
../imaging-101-maintainer/batch10/task04_astronomy/oracle-env/Scripts/python.exe ../imaging-101-maintainer/batch10/task04_astronomy/capture_oracle.py
```

The capture invokes unchanged `n2hp.n2hp_vtau_fitter`/`SpectralModel.fitter` on every exact bound spectrum, using the unchanged original `n2hp_vtau`/`hyperfine` forward functions and MPFIT. Error is population standard deviation over original radio velocity 20..28 km/s. Initial p=(5,.5,3,1); maxiter=200; ftol=xtol=gtol=1e-10. There is no random seed or random data generation. All 501 channels are fitted, all nine standard results converge, process exit 0, no numerical warning. Bound original fit time: 1.28610409994144 seconds excluding imports/loading.

`test/fixtures/upstream_run.npz` was captured from this independent execution BEFORE local extraction. It contains original parameters `(1,4,3,3)`, model `(1,501,3,3)`, noise `(3,3)`, chi_square `(3,3)`, probe_parameters `(4,)`, probe forward brightness `(501,)` and probe opacity `(501,)`. Probe parameters are (8 K,2,3.4 km/s,0.3 km/s). A later independent original-forward finite-difference capture, `upstream_jacobian.npz`, stores the probe Jacobian and relative step 1e-5; it is additional stage coverage, not represented as the original pre-extraction fixture. Normal tests never import upstream code.

Independent original repetition was bit-identical. Original ftol sensitivity 1e-8 versus 1e-10 had maximum relative shifts (Te,t0,v,sigma): (1.1501855315525518e-5,1.3421499336145399e-5,1.5249127558666445e-6,5.603794601344549e-6). Recasting original brightness back to its source float32 changes no samples/results. `acceptance-before-extraction.json` freezes forward atol=rtol=1e-11; all-four parameter maximum relative tolerance 2e-4; fitted-model RMS difference <=1e-5 K. The original max reduced chi-square was 1.2160321401642157; observation-consistency boundary 1.3 was specified before judging fresh results. No thresholds were relaxed after local results.

## Equation / upstream / local / test traceability

| Required stage or equation | Unmodified upstream file/function | Local executable function | Independent validation consumer |
|---|---|---|---|
| Molecular component constants and degeneracy normalization | spectrum/models/n2hp.py dictionaries, n2hp_vtau | physics_model.py OFFSETS, STRENGTHS, FREQUENCIES, NORMALIZATION | public constant equality; frozen opacity/forward probe |
| Radio channel conversion and line-free noise | examples/n2hp_cube_example.py error-map stage; original FITS calibration | preprocessing.load_observations, estimate_noise | original noise fixture exact equality; README frequency-constant tests |
| Gaussian component opacity, physical sigma | hyperfine.py hyperfinemodel.hyperfine component loop; paper Appendix C model | physics_model.optical_depth | original opacity fixture; independent integrated Gaussian opacity; redshift sign test |
| C1-C2 Planck source-minus-background and transfer D6 | hyperfine.py hyperfine final T0/background/spec block | physics_model.brightness | original brightness fixture; zero-opacity, equilibrium and saturation checks |
| Weighted least-squares residual and chi-square | models/model.py mpfitfun and fitter; mpfit.py generic LM | solvers.fit_spectrum local residual, generic scipy TRF | original per-pixel parameter/model/loss fixtures; exact recomputation of residual sum; original-forward numerical Jacobian |
| Pixelwise spectral parameter maps | original n2hp_vtau_fitter called for every bound pixel | solvers.reconstruct | all-nine full-pipeline upstream parity; independent noiseless physical recovery |
| Generic optimizer steps | original MPFIT LM internals | scipy generic trust-region reflective backend | Deliberately different backend; no false update-trajectory parity claim; same objective/bounds/init and fixed converged-output acceptance |
| Standalone map metrics and observation consistency | independently defined benchmark contract, not copied upstream metrics | evaluation/judge.py | known numerical fractional-error case, baseline/fresh consistency, malformed and poor prediction tests |
| Maintainer comparisons | original cube example Tex/velocity map and spectrum inspection | visualization.compare_maps, active notebook | executed notebook with all four maps, representative spectra/residuals and paired all-nine chi-square bars |

## Paper/code discrepancies and independent interpretation

The selected paper v1 Appendix C2 has the correct positive Planck exponent. Appendix D7 has a visible negative exponent typo, verified by rendering page 14 to a private image. A negative exponent would yield negative source brightness and an incorrect high-temperature limit. Executable `hyperfine.py` uses the positive exponent, agreeing with C2 and independent thermal/saturation tests. Local code follows the validated executable expression; no erroneous D7 transcription is silently preserved. This is a paper typographical discrepancy, not an upstream-code correction.

Upstream's `lines=(1-voff/c)*freq_dict[name]` enters widths and the velocity offset while line centers remain `freq_dict[name]`. Preserve this slightly component-dependent convention explicitly as L in README/approach; do not silently simplify to a single rest frequency. The 15 J=1-0 strengths sum to 0.9999990000000001 and are renormalized; round-off is preserved. Upstream includes distant J=2-1/3-2 components; their Gaussian contributions are numerically zero at this bound band and fit domain, independently confirmed by full-forward parity. sigma is not FWHM.

Original cube example uses optional neighboring guesses and a multiprocessing traversal. This task selects independent sightline fitting with the original documented fixed initialization, preserving the underlying spectral inverse and avoiding traversal-dependent initialization. The cleaned solver is explicitly generic TRF rather than MPFIT LM. No custom published inverse update is removed, and this backend change is disclosed throughout README hints, plans and notebook. The lower positive width guard excludes only the undefined zero-width boundary; all converged widths lie above 0.15 km/s.

## Actual references, fresh measurements and metrics

Actual physical truth is unavailable, so `evaluation/ground_truth.npz` is deliberately absent. `evaluation/baseline_reference.npz` contains the original four standard-method maps, in K, dimensionless, km/s and km/s respectively. It is not relabeled truth. `notebooks/reference_outputs/upstream_measurements.npz` contains this task's reproduced original parameters, spectra and weighted losses, consumed by notebook comparisons. `cleaned_measurements.npz` contains newly computed local parameters, spectra, losses and measured runtime, refreshed on active notebook execution. Fixtures have separate parity-test consumers. Paper tables or cached historical values do not supply any metric gate.

The metrics are task-specific: for every map, maximum absolute fractional error against nonzero standard values; and maximum spatial reduced chi-square of model versus actual observed spectra. NCC and dynamic-range NRMSE are not used for disparate physical quantities and their narrow value ranges. Thresholds 2e-4 and 1.3 are non-vacuous; a 0.1 km/s perturbed prediction fails and invalid/missing arrays error. All four quantities must pass. No scale, translation, masking or phase alignment is permitted.

Fresh local Judge results (full precision):

```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "excitation_temperature_relative_max": 4.8828702486725936e-05,
    "optical_depth_relative_max": 7.101986760430604e-05,
    "velocity_relative_max": 4.02668660382813e-07,
    "dispersion_relative_max": 6.010691065585748e-06,
    "maximum_reduced_chi_square": 1.2160321401279806
  },
  "thresholds": {
    "excitation_temperature_relative_max": {
      "comparison": "<=",
      "boundary": 0.0002
    },
    "optical_depth_relative_max": {
      "comparison": "<=",
      "boundary": 0.0002
    },
    "velocity_relative_max": {
      "comparison": "<=",
      "boundary": 0.0002
    },
    "dispersion_relative_max": {
      "comparison": "<=",
      "boundary": 0.0002
    },
    "maximum_reduced_chi_square": {
      "comparison": "<=",
      "boundary": 1.3
    }
  },
  "errors": []
}
```

Notebook's fresh reconstruction time: 0.16084210004191846 seconds. Original runtime and fresh runtime are measured on their actual environments and are not a controlled speed benchmark. All figures were extracted and visually inspected: four quantity rows each show upstream/fresh/signed difference with matched paired scales; representative real spectra and residuals show proper component locations; parameter errors remain below the frozen line and all nine paired reduced-chi-square bars remain below 1.3. The notebook makes fresh reconstructions, not archive substitution.

## Commands, environment and checks

Cleaned shared environment: Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1, CPU Windows. Backend details in `cleaned-environment.json`; the actual shared environment has no pyspeckit, Astropy or spectral_cube installed. `requirements.txt` lists only runtime subset pins; pytest/Jupyter remain development tooling. No packages were installed/upgraded in the shared environment. All calculations are float64 CPU; GPU is unnecessary.

Final standard commands, task working directory:

| Command | Exit | Wall time seconds | Private log |
|---|---|---|---|
| `python -m src.main` | 0 | 0.595873 | main-final.log |
| `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.129243 | judge-final.log |
| `python -m pytest test/ -v -p no:cacheprovider` | 0 | 2.184270 | pytest-final.log |
| `python -m jupyter nbconvert --to notebook --execute notebooks/n2hp_hyperfine_cloud_mapping.ipynb --ExecutePreprocessor.timeout=120 --output n2hp_executed.ipynb --output-dir <private archive>` | 0 | 3.774611 | notebook-final.log |

Pytest: 28 tests passed, including original stage and pipeline parity, independent physical properties/recovery, every required map, metric known cases and Judge contracts. Judge tests cover missing files/keys, bad NPZ, shape, integer/complex/object dtype, NaN/Inf, nonphysical domain, valid poor predictions, float32/float64, hidden asset/configuration failure, actual-truth precedence, foreign working directory and report-write failure. Invalid input reports contain empty metrics; codes 0/1/2 and strict finite JSON are checked.

The default user TEMP directory refused pytest access; setting private writable TEMP/TMP resolves this environmental issue. The first noise-error unit test incorrectly selected two valid line-free channels; it was corrected to use channels outside that interval. Final tests pass; earlier failures are not counted as scientific evidence. Nbconvert emits the standard Windows Proactor/ZMQ selector and local unencrypted kernel warnings, but completes without notebook errors. A subprocess-based run records the actual nbconvert exit code 0; PowerShell's rendering of stderr alone is not interpreted as an execution failure. All four notebook code cells have execution counts and three inline figures; timeout 120 seconds is far above observed approximately 4-second complete kernel execution while remaining bounded. There is no temporary executed notebook in task output.

Manual source/plan review checked callable signatures, physical parameters, opacity normalization, Planck and Doppler signs, background subtraction, sigma convention, optimizer initialization/bounds/stops, path resolution and fresh notebook flow. AST/call-role audit found only NumPy/SciPy/Matplotlib general computation and standard library imports in source/Judge/tests/notebook; no domain reconstruction solver, cross-task import, forbidden imaging library or dynamic upstream source import. No matplotlib.use call occurs. CLI and notebook import of main preserve backend.

Public allowlist review checks README, raw NPZ and requirements only: every key has batch-first shape/dtype/unit table; raw contains only observations and molecular calibration; no GitHub URL/domain, hidden reference path, metadata JSON, hidden seed or target-derived calibration leaks. README contains physical forward equation, complete four-array output, acquisition metadata and conceptual hints. All source, evaluation, plans, tests and notebooks remain maintainer-only per benchmark release policy.

## Package hashes

| Artifact and documented consumer | SHA-256 |
|---|---|
| data/raw_data.npz | `8893e9b04cdf30551db61ce119d51906dbbf297318975b3b3ff1b994bf549054` |
| test/fixtures/upstream_run.npz | `68317e1a10776ac3e0200c606bd583868c85a9a095cdecd298ffdc4432514b7b` |
| test/fixtures/upstream_jacobian.npz | `c597c99946b87967e5309fc3a32c74d627c754515c037c2d897def3e88d72ab4` |
| evaluation/baseline_reference.npz | `1e80551e986188634b5b3ac13fde684dcf930f707bdc15cbcb2a37e626b22df9` |
| notebooks/reference_outputs/upstream_measurements.npz | `fe1028f984705bc0034130dd1373e552b953ebc328e34defda51db7036a92dc7` |
| notebooks/reference_outputs/cleaned_measurements.npz | `45324ed064f8b4793b25c54749aade8ec03ed3b460dfe87fca11441946dc65da` |
| output/reconstruction.npz | `8d01d90aa577c75e316c73b0657bb890e9301ae15598151e28c79a5c4db887b8` |
| evaluation/metrics.json | `736bd6b12e455a04a4207a70c434385dae11416daeb9ef96f2e826bfee68eabc` |

Individual bound-array byte fingerprints (C-order values; shapes/dtypes documented above):

```json
{
  "brightness": "d26a39a5d432322c2550337926f1c9901b209a5c162d3f3229f805b666b39bed",
  "frequency": "d57861395b876bb947ff3a05877907b4af21e6f09a73046dd6e7002644605b3b",
  "hyperfine_velocity_offsets": "c715ed9d6315074f10f8f10a6433ab3da04be76db395c6edbaaafced50104bbb",
  "hyperfine_relative_strengths": "ecceadf7db1ee18cfd8c7edfb3a6884d034a49ae94daacbbe684242b7160e57a"
}
```

## Maintainer archive index

Private archive root: `../imaging-101-maintainer/batch10/task04_astronomy/` outside the task. `upstream-code/` is the unchanged complete source checkout at the pinned commit; `upstream-data/` is unused exploratory test-data checkout at its separate pinned commit; `oracle-env/` is the isolated oracle environment. `exploratory_4x4/` preserves all candidate observations, original fixtures and outputs. Capture/write scripts, PDF, rendered paper page, original logs, extracted review figures and executed temporary notebook stay here, not the task release. The failed data-paper PDF download has no fabricated file. Shared environment lives separately at `../imaging-101-maintainer/shared-env/`.

| Archived source/evidence | SHA-256 |
|---|---|
| opha_n2h.fit | `9f9331ab9fb240276dfdb5d8a5938b7063c824f58490813b7f81d6a8ff4b5065` |
| ReadMe | `df266e264c7723b3db720f2b95b16fdf7f19a7d495009ac355cf5b16471a28b5` |
| cds-terms.html | `f6bab16f7060f41c98958f0044fa3a526eeeeacdd23cb2fa7a529016d2b29e9f` |
| pyspeckit-paper.pdf | `f2c43ea44a62f1ecf324052e0eb2653c38a0f7b2870880319f861da2076b6c91` |
| arxiv-query.xml | `1072db427d1211c601a9fe826572ce4ccc1c1ed0e1e042e51faced4831c9cb71` |
| capture_oracle.py | `75a42192745fc5cdfd29bc999493ac505be3bbb1f06d14c7474cded3d002d75d` |
| oracle-run.log | `d52efe834a88f1bd330650006cb02f57e1b1e5689930ef8509cc3c9eb897960c` |
| oracle-run.json | `a958263751047c68d227addb2819f25369885f0256e5faa57b458b7fad0ee94a` |
| oracle-environment.txt | `e1a718f48e947edc5706a75446678f855de7bbdea191d17350ed7c40e3733604` |
| oracle-backend.json | `1b9d9a816999dc5bbe636186a1cfb3b71d7f70a873ff2b39aec69654c029bec5` |
| capture_exploratory_4x4.py | `1b505adcd1603c566495acc1d7fe54f890e631387a76b78030419ab994f2fae7` |
| exploratory-4x4.log | `a45f2a6ed4323188374cf123920a5b48f9a5041a33fbf1b0f07151c412351250` |
| exploratory-4x4.json | `34ee456e0c475ff317286f0c6a8ab78d1af862b36c35b939f4a52559d78e64fd` |
| identifiability.py | `0e6cd96e54e11c415df3b4c18b6942b0770f5f49162000e703c316b48c1a6ea9` |
| identifiability.json | `6aa4266381b26da1f5b0299feac36b39a351ed9fb7d98a4647e934e79970a49f` |
| identifiability.log | `079d17b5e901251dc617b4861ee7e058c94315ec7df7e131801ef5537ca44ee9` |
| sensitivity.py | `0deaee4ade6a6fe6d348e32336d89150a91d4282dfb9bf73e7a2d2e429d95f3b` |
| sensitivity.json | `96cd1077e3cf69d33a905b1b2635218ebd8450458e30522aa6c473fa2760ae96` |
| sensitivity.log | `ccf6319ace15f3469964468da0410867c1bf4e1bdb3c354ed69f0da4319498e6` |
| acceptance-before-extraction.json | `b5d266a5197f7019d73efd7900fe319dd156a8405868b7814b3b32f03e183c52` |
| bound-array-hashes.json | `381dc81aefdaa83c51983b405af1034546f342f4af1b1dfe6bd2176887b0db4b` |
| final-commands.json | `fd7ca64b02246c19ce0bdb3076aa5a8f0d57bda7d8aa7628bf85aa8f354d2b88` |
| cleaned-environment.json | `76fe8c584482819351957b767b9d345e5b152d6a2311b79ebd92f1835326ebdf` |
| main-final.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| judge-final.log | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| pytest-final.log | `9f5d5fe020f1faad148d9e8d040db783b7c73cf2616d218408bc6aeb15d762b5` |
| notebook-final.log | `29725388e790a4f4b90eb62e9cf054040bf951cb20a6c072ff7da50296495263` |

`figure-1.png` through `figure-3.png` are private notebook visual-inspection exports; `paper-page14.png` is the independent sign-discrepancy visual check. `n2hp_executed.ipynb` is the private executed review copy; the shipped notebook itself contains executed outputs. Temporary script-generated __pycache__ directories are removed before handoff. No unconsumed fixtures, duplicate validation reports, wrapper entry points, standalone metadata JSON, raw original FITS/PDF or upstream checkout remains in the task.

## Completion gates

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper, exact source revision, data hashes and equation mapping | passed | Source/paper/data sections and traceability/hash tables; remote freshness limitation explicit |
| Original and cleaned run on exact observations | passed | capture_oracle.py, frozen brightness/frequency byte fingerprints, original and fresh commands/logs |
| Independent upstream parity and physics | passed | Stage/model/noise/loss/Jacobian/pipeline tests and independent Gaussian/thermal/noiseless checks |
| Constants, units, coordinates and precision | passed | Original header mapping, public molecular arrays and README constant checks |
| Standard solver output and standalone Judge | passed | main/judge final exit 0 and complete four-map report |
| Judge malformed/poor/evaluator-error contracts | passed | Parametrized unit and subprocess contract tests |
| Visible inputs suffice and no hidden leakage | passed | Self-contained public equations/calibration/acquisition/output, allowlist audit |
| Metadata/README/no standalone metadata JSON | passed | Required sections, all keys/units, complete acquisition table |
| Minimal shared runtime without upstream domains | passed | Cleaned environment records domain package absence and CPU execution |
| Explicit local forward/inverse mathematics | passed | Source/call-role audit; disclosed allowed generic optimizer backend |
| Complete test suite | passed | 28 tests, pytest final exit 0 |
| Active executed notebook and scientific comparisons | passed | nbconvert exit 0, four active code cells, three visually inspected figures |
| Metric/result provenance and fixed boundaries | passed | Pre-extraction acceptance, original sensitivity, reference/fresh separation |
| README/plans and signatures/mathematics consistency | passed | Manual equation/sign/unit/caller review and public-constant tests |
| Final inventory, archive and legal notices | passed | Task template, consumed reference/fixtures, full license/notice record below, private archive index |

## Independent coordinator review

On 2026-10-18 the coordinator independently reran the standard CLI, the
standalone Judge from another working directory, all 28 tests and the active
notebook through nbconvert. All four commands exited 0. The explicit radiative
transfer, component frequency convention, objective and optimizer substitution
were reviewed against the approach plan. The coordinator inspected the four
parameter-map comparisons and observed/fitted spectra with noise-normalized
residuals. Public-input and import scans passed. Additional logs, executed
review notebook and figures are privately archived under
`batch10/coordinator/n2hp_hyperfine_cloud_mapping`; the structured review is
the sibling `n2hp_hyperfine_cloud_mapping-review.json`. Original packaged
measurement snapshots are retained; changed runtime snapshots from the review
are archived separately.


## Preserved full license and attribution record (documentation revision)

The user requested removal of task-root LICENSE/NOTICE files. Their complete original texts and attributions are retained verbatim below, and original bytes are backed up in the external revision archive. This relocates notices without changing their terms, scientific data rights or source copyright. Historical upstream source paths/hashes remain historical evidence. Current source references point to this section.

### Original LICENSE (verbatim)

Original-byte SHA-256: `083c58fdf27b56946ebc6b370908a2bfcded6140bd73a58073d6d4fb2c9dd66c`.

```text
Copyright (c) 2010 Adam Ginsburg & Jordan Mirocha



Permission is hereby granted, free of charge, to any person obtaining a copy of

this software and associated documentation files (the "Software"), to deal in

the Software without restriction, including without limitation the rights to

use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies

of the Software, and to permit persons to whom the Software is furnished to do

so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in all

copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR

IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,

FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE

AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER

LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,

OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

SOFTWARE.

```

### Original NOTICE (verbatim)

Original-byte SHA-256: `690f1958c2299f38353b32278551f7e57fdefc056d1db899c610c7642916c5e7`.

```text
Code attribution: mathematical model adapted from pyspeckit, Copyright (c) 2010 Adam Ginsburg & Jordan Mirocha, MIT license; full notice preserved in LICENSE. The generic numerical optimizer is provided by SciPy. No upstream imaging package is included in this task runtime.



Observational data attribution: Andre, Belloche, Motte & Peretto (2007), Astronomy & Astrophysics 472, 519; IRAM 30m N2H+(1-0) maps of Ophiuchus, CDS catalogue J/A+A/472/519, OphA cube. Data are a contiguous crop promoted to float64 and repackaged without interpolation or synthesized observations. CDS/VizieR is acknowledged.



VizieR states that retrieved data are free of usage in a scientific context with explicit citation of the original authors, publication and publisher. Commercial usage depends on the originating publisher's terms. This task does not relicense the observations under the code's MIT license. The archival data README contains no separate dataset copyright statement. Source and terms: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/472/519 and https://cds.unistra.fr/vizier-org/licences_vizier.html .

```


## Completed documentation revision and revalidation (2026-10-18)

This revision changes explanatory documents, notebook organization and license
placement. Acquisition arrays, inverse mathematics, evaluation assets, thresholds
and packaged scientific reference snapshots are unchanged. The original documents,
notebooks, code and license bytes are retained under the private archive
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/revision_docs/task10/n2hp_hyperfine_cloud_mapping/before/`; removed root notices are additionally retained in
`removed-root-files/`. Full original license/attribution text and original hashes
are reproduced verbatim in the preceding preserved-license record. References to
the former root notice in source comments/module docstrings now identify this
validation record; executable ASTs match the pre-revision source after stripping
docstrings. Necessary README attribution references were updated accordingly.

The approach plan now defines the actual forward discretization, variables,
normalizations, units, initialization, inverse updates/options, termination,
evaluation and scientific limitations. The design plan records every shipped
runtime/Judge top-level signature, parameter/return contract, caller flow, file
consumer and side effect. Signatures were checked against source, and equations,
units, signs and caller roles were reviewed against the unchanged implementation.

The notebook has exactly six H2 sections: Problem, Data, Physics, Method, Results,
Conclusion, numbered 1 through 6. Data includes concrete original filenames,
repository/archive provenance, crop or processing status, visible array contracts
and acquisition metadata. Physics and Method are self-contained. All original
scientific comparison groups remain present, with upstream, fresh and initial or
alternative-condition results labeled explicitly. The default path actively
reconstructs observations with local modules and evaluates the fresh submission;
historical reference snapshots are never rewritten during review.

Revalidation gates: **passed**. The full `python -m pytest test/ -v
-p no:cacheprovider` run passed **28 tests**. Full `python -m jupyter nbconvert
--to notebook --execute notebooks/n2hp_hyperfine_cloud_mapping.ipynb --output review_executed.ipynb
--output-dir <private-task-archive> --ExecutePreprocessor.timeout=240` completed
with exit 0. Every one of **5 code cells** executed without error. All
**3 exported figures** were opened and visually inspected; comparison
panels, annotations, residuals and metric groups remain readable and scientifically
consistent. The shipped notebook contains these executed outputs. Exact absolute
commands, working directory, exit status and wall times are in `commands.json`;
stdout/stderr are in `pytest.log` and `notebook.log`. Fresh reconstruction runtimes
and metric streams are in `new-notebook-streams.json`; no historical runtime was
relabelled as a new measurement.

Byte-level checks passed for all public observations, evaluation files and every
file under `notebooks/reference_outputs/` against the pre-revision snapshot.
The private `before-hashes.json`, `reference-hashes-before.json` and
`reference-hashes-after.json` record this evidence. Figures are indexed by
`figures.json`; the executed review notebook and PNGs stay private. The private
`final-manifest.json` indexes the entire final task with SHA256 hashes, including
this validation record, and records the immutable-reference comparison. No
temporary executed notebook, screenshot, upstream source or capture script is
shipped in the task. No scientific completion gate is replaced by this
documentation review; the original numerical evidence remains authoritative.

Final reviewed document hashes (the validation record is hashed externally to
avoid a self-referential digest):

| File | SHA256 |
|---|---|
| `plan/approach.md` | `d931c4cfbaf63c3a1419062e3686e7029c1a161e3202c4af7fe4dc41821a7404` |
| `plan/design.md` | `59f10453ac883a7cf63ebb22fe0e5548722b69952ebe47b808a4d686eb1118ce` |
| `notebooks/n2hp_hyperfine_cloud_mapping.ipynb` | `28e68c68559e0c5bf89df06b0e462839ca06a13219eddfa622aa1f709365a06a` |
| `README.md` | `774970399868f63786d82954a701e05eba645ca81bc4367391592de62b8040f2` |

### Final notebook Markdown readability review

The final prose review corrected concatenated English words and spacing while
preserving scientific meanings and all fenced mathematical formulas. Repeated
source/crop descriptions in the phase-shifting and Spitzer Data sections were
consolidated into their detailed provenance paragraphs. Every code cell was
compared in full against the successfully executed private
`review_executed.ipynb`: source, execution count, outputs and cell metadata are
identical. This Markdown-only change therefore did not rerun unchanged code or
relabel old computations. `final-markdown-proof.json` records the equality
checks and final notebook SHA256; the document-hash table above and private
`final-manifest.json` are updated to the final bytes.
