# Validation and provenance

Task version1; review date2026-10-30. No independent physical particle truth is available. `evaluation/baseline_reference.npz` contains the independently run standard fitted parameters, used as an archived scientific comparison, never as an identity target. The Judge directly evaluates both particles' public forward objectives. A better fit passes. Local changes in fitted radius/index with low cost show that reference equality would be unjustified.

## Sources and private archive
- Published method: Lee et al., Optics Express15,18275–18282(2007), DOI10.1364/OE.15.018275; publisher-version PDF independently read from https://web.science.uu.nl/SCM/Articles/2007/Lee_OptExpress_2007.pdf . Paper Eq1: interference intensity; Eq2: empty-field normalization; Eq3: full Lorenz-Mie scattering fit. The paper's own632nm/135nm-per-pixel acquisition is historical context, not this later447nm/48nm-per-pixel author tutorial acquisition.
- Author source: https://github.com/davidgrier/pylorenzmie exact remote-cloned revision838349246b8ae4818a239e011b5c5831671b1f4b. Remote clone succeeded this session; no claim about remote revisions after that clone. GPL-3.0 LICENSE copied unchanged; NOTICE identifies adaptations.
- Maintainer archive: research_batch13_20261030/worker2_mechanics_earth_astro. `pylorenzmie/` is the original checkout. `oracle_holo_detect.py`, `oracle_holo_run.py`, `oracle_holo_freeze.py` capture unmodified upstream source execution and fixtures. `oracle_extra/` supplies isolated opencv5.0.0.93,trackpy0.7,looseversion1.3.0; shared environment is unchanged. Full pip/tool outputs remain in agent session; original numeric log files are oracle_holo_run.log and oracle_holo_run_corrected.log. `holo_hashes.json` indexes the original image/background/notebook, packaged raw and fixture SHA256 values.
- Original observations: author docs/tutorials/sample.png and independent background.png;1024x1280 nativeuint8 arrays. Public NPZ stores exact PNG-decoded counts with leading batch dimension, no downcasting, denoising or target-derived calibration. Background was recorded without sample according to the author's notebook; nominal author-provided dark count13 (not an independently measured dark frame) and instrument447nm/.048um-per-pixel/nmedium1.34 are notebook calibration constants.

## Reproduced upstream run
Oracle: existing research_batch11_20261027/astro_earth_physics/oracle-env, Python3.11.11 Windows AMD64; NumPy/SciPy/pandas/Numba plus private optional decoding/localization dependencies. AllOMP/MKL/OpenBLAS/Numba thread counts1, CPU only. Source NumPy model supplies frozen operator arrays; upstream accelerated Numba CPU model supplies inverse measurements. No GPU.
Commands: oracle python oracle_holo_detect.py; oracle python oracle_holo_run.py; oracle python oracle_holo_freeze.py. Each exited0. Raw detection located exactly two features; fixed observed-data ROI boxes(809,187,228,228) and(222,635,200,200). Upstream full-mask Cauchy fits took1.0663s and0.6376s. DEseed0;bounds4.8..16.8um for height,.5..2um radius,1.35..1.8index;local optimizer original default settings. Fit noise scale.05 is the source default, not independently calibrated shot noise. Excluded pixels are normalizedROI maxima exactly as source Feature.

### Discrepancy investigated
Literal current SingleFrame notebook gives DE z bounds100..350 and labels them pixels. Current DEEstimator source explicitly treats bounds as micrometres and divides by magnification. Literal execution converged to spurious large heights(~2679/7528pixels), low contrast and worse fits; this failure is retained in oracle_holo_run.log. We independently checked dimensional conversion100pixels*.048um/pixel=4.8um,350pixels*.048=16.8um, corrected only bound units in the capture configuration and reran original equations/estimators. Successful corrected run is oracle_holo_run_corrected.log. This is a documented modern-source unit adaptation, not an unchanged execution of an inconsistent tutorial. No acquisition constant was changed.

## Equation and stage traceability
| Required stage | Paper/source | Local callable | Independent case |
|---|---|---|---|
| Empty-field count normalization | LeeEq2;SingleFrame normalization | preprocessing.load | exact nativeuint8 and analytic count normalization test |
| Homogeneous sphere coefficients | Sphere.mie_coefficients Riccati recurrence | physics_model.Sphere.mie_coefficients | independently captured upstream coefficient array |
| Vector outgoing field,illumination phase | LeeEq3;LorenzMie.lorenzmie/field | physics_model.LorenzMie.lorenzmie/field | upstream complex field201pixel fixture |
| Interference intensity | LeeEq1/Eq3;LorenzMie.hologram | physics_model.LorenzMie.hologram |201pixel upstream hologram fixture; index-matched sphere analytic invisibility |
| Global fringe-order initialization | upstream DEEstimator | solvers.reconstruct differential_evolution stage | active pipeline; same family,bounds and objective; local sample realization differs |
| Robust full-pixel inverse refinement | upstream Optimizer robustCauchy | solvers.reconstruct least_squares stage | full exact bound observations;objective comparison accepts better solutions |
| Standalone objective | explicit public conditional target | evaluation/judge.score | fresh,quality-fail,per-parameter perturbations and malformed contract tests |

## Evidence and thresholds
Before final notebook validation the standalone thresholds were frozen at1.342/1.161, derived from independently run upstream full-pixel costs1.3408371701343458/1.1602416266942688. Differences from upstream printed redchi come from redchi's subtraction of5fitted degrees of freedom; notebook/Judge consistently report mean Cauchy objective per valid pixel. Numerical budgets are0.0011628298656542/0.0007583733057312 absolute objective, under0.09% of upstream values. These are small numerical optimization budgets, not uncertainty confidence regions. No threshold was relaxed after final validation. Fresh clean costs1.224126635862471/1.1602397172224104. Native Cauchy optimizer reached a better first-particle minimum than the upstream run; parameters therefore differ without being penalized. Full float precision is retained in JSON/NPZ and notebook outputs.
Independent parity uses rtol=atol=2e-12, chosen for float64 recurrences on201nonsingular pixels. Actual maximum hologram difference0; complex field and coefficient checks pass. Physical index-matched sphere has intensity1to2e-13absolute tolerance.

## Runtime and notebook
Clean runtime Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1; no pandas/Numba/OpenCV/domain package imports in the cleaned task. `python -m src.main` exited0 and wrote fresh NPZ in14.884s. `python -m pytest test/ -q --basetemp=<private archive>/pytest_holo_2` exited0:7passed in5.71s. Windows default pytest temp permissions failed initially; a private explicit basetemp resolved that environment-only issue. No test assertion was weakened.
`python -m nbconvert --to notebook --execute ... --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=holo-shared` exited0 with the explicit private shared-runtime kernelspec and one numerical thread. Fresh reconstruction runtime in notebook17.0196s; comparisons and ablations also actively ran. Initial `python -m jupyter nbconvert` dispatched through PATH to Anaconda; that run was stopped, not counted. Successful command uses nbconvert directly. Windows ZMQ Proactor/TCP kernel warnings did not interrupt execution. Executed notebook replaced the review notebook; temporary executed output notebook removed. Portable notebook metadata reset to standard python3 after execution, without changing cell outputs.
Five embedded figures were extracted privately to holo_figure1..5.png and all viewed individually: raw/background/normalized frame;fresh model/residuals;upstream model/residuals;Cauchy objective ablations;linear-loss model/residuals. Labels/colorbars/axes and every comparison are visible. Central structured residuals remain in real data; the conditional isolated-sphere model does not claim to explain dust, aberration or coherent neighbor effects. We do not interpret objective agreement as independent particle accuracy. Source localization-diameter/nfringe tuning illustrations are replaced by fixed declared observed-data ROIs; inverse robust/nonrobust/global-start comparisons and both physical particle groups are retained. No cached result is presented as fresh.

## Contract coverage and semantic review
Judge checks missing file/key,wrong shape,complex/integer dtype and NaN/Inf,each physical parameter independently changed,poor valid parameters,foreign working directory,missing metrics/raw assets and failed report writing. Error reports have empty metrics; quality failures exit1,errors exit2,passes exit0. Hidden reference parameter arrays are never used in scoring. `test/fixtures/upstream_forward.npz` consumer:test_upstream.py; `notebooks/reference_outputs/upstream_fit.npz` consumer:active notebook upstreamcomparison;Full upstream predictions/residuals/coordinates are retained privately as upstream_holo_1/2.npz, avoiding unused redundant task arrays. `evaluation/baseline_reference.npz` is the reserved standard result archive. No actual ground_truth file is invented.
Plans reviewed against shipped signatures,position/radius/wavelength units,vector components,negative-z incident phase,objective normalization,Cauchy sign and actualCLI/notebook flow. Local field is vector rather than scalar amplitude. Solver library modules never select matplotlib backend. AST/source import review found only stdlib/NumPy/SciPy/Matplotlib; no cross-task/domain solver/framework runtime shortcuts. README/publicNPZ/requirements contain noGitHubURLs or hidden arrays. Both raw arrays exactly equal source-decoded observations and independently acquired calibration.

## Completion gates
All mandatory gates for this selected modern author-method task are passed:source/paper/revision/data hashes;original andclean bound runs;independent operator fixtures/analyticphysics;units/nativeprecision;CLI/Judge;malformed/negative/evaluator contracts;public sufficiency;no leakage;minimaldependencies;local executable physics/inverse;pytest;active six-chapter notebook and five actual figure reviews;historical/upstream/fresh provenance;plans/README consistency;inventory/license. The documented source tutorial unit discrepancy is resolved by independent dimensional validation; published2007acquisition results are not claimed reproduced by this later author data.

## SHA256 inventory

```json
{
  "docs\\tutorials\\sample.png": "a16582e538afdf7561ff4b53dee8df6e49ea4593284c0282aa8ba91ba6baa6d5",
  "docs\\tutorials\\background.png": "50524e5bfeef9196cff0c65256cf80067eb5d39f41798dbfb3ee3adaffc8bc52",
  "docs\\tutorials\\SingleFrame.ipynb": "c4d388b7ff07482df6c8cf199419c3e14c0added38ed65933407702bdc46c9cc",
  "data/raw_data.npz": "8cd8bb338c256db3c82b05699357be0142b1ac533a8f644efca67b9e564a0f45",
  "test/fixtures/upstream_forward.npz": "b2b8b57820d68d6eeccb51dbae1195a7b0625148f22d4a63567bfdd5e03df68e"
}
```

Oracle environment independently inspected: Windows10 build26200,Python3.11.11,NumPy2.0.2,SciPy1.17.1,pandas3.1.0,Numba0.68.0; exact machine-readable record is private oracle_environment.json.

## Parent API/documentation review correction
Particle.ab now raises NotImplementedError; only Sphere supplies physical scattering coefficients. All extracted ndarray/dictionary/iterator annotations resolve explicitly. No Mie recurrence or thresholds changed. The excluded maximum is an observed-maximum convention, with no verified camera-saturation claim. Dark offset13 is nominal source calibration; the separate background remains independently acquired.

Type-hint introspection using typing.get_type_hints on src/physics_model.py: 47 functions/property accessors resolved without NameError, including Image alias.

Type-hint introspection using typing.get_type_hints on evaluation/judge.py: 49 functions/property accessors resolved without NameError, including Image alias.

## Enhanced complete notebook review
Re-executed full active reconstruction and all comparisons after adding self-contained displayed physics/objective equations, symbol definitions, physical-unit tables and actual source/fresh/ablation result tables. Explicit UTF8 subprocess/report decoding added. nbconvert exit0, 5figures retained; canonical executed outputs frozen, temporary executed copy removed. Numerical methods and thresholds unchanged.
All5enhanced-run figures were actually opened with view_image and inspected after execution; no missing groups or cut-off axes found.


## Independent parent acceptance, batch13

Parent reviewed explicit Riccati-Bessel sphere coefficients, outgoing vector-wave recurrence, all electric-field components, negative-z incident phase and full-pixel robust objective. The independently recorded background is distinct from nominal dark13counts. Source tutorial z bounds required an explicitly documented pixel-to-micrometre unit adaptation, not a silent physics change. The maximum exclusion is an observed-maximum convention, not verified camera saturation. Parent7tests, fullCLI, foreign Judge and math all exited0. Nonphysical unused base-coefficient placeholder was removed and type annotations resolved. Parent read the enhanced completely executed notebook with defined physical equations and actual parameter/unit tables, and viewed all five canonical figure groups with both particles, independent source, full residuals and fresh ablations. Better feasible coupled size/index/depth fits pass without matching hidden particle parameters; structured real residuals remain visible and no independent particle truth is claimed.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/colloidal_hologram_particle_geometry/ records commands, reports, actual extracted canonical figures and final hashes.

## Reopened evaluator-configuration guard
Parent audit identified vacuous all(empty-thresholds) pass and unsupported comparison interpretation. Judge now requires the exact known metric set, exact comparison directions, numeric nonboolean finite boundaries and appropriate signs. Empty/missing/extra metrics, unsupported comparison, NaN/Inf, nonnumeric, boolean, nonpositive upper bounds and malformed JSON are evaluator errors2 with empty metrics. Valid boundary values and scientific scoring were not changed. Only evaluator guards/tests changed; no solver or Notebook rerun is needed for this guard.


## Independent parent acceptance, batch13

Parent reviewed explicit Riccati-Bessel sphere coefficients, outgoing vector-wave recurrence, all electric-field components, negative-z incident phase and full-pixel robust objective. The independently recorded background is distinct from nominal dark13counts. Source tutorial z bounds required an explicitly documented pixel-to-micrometre unit adaptation, not a silent physics change. The maximum exclusion is an observed-maximum convention, not verified camera saturation. Parent7tests, fullCLI, foreign Judge and math all exited0. Nonphysical unused base-coefficient placeholder was removed and type annotations resolved. Parent read the enhanced completely executed notebook with defined physical equations and actual parameter/unit tables, and viewed all five canonical figure groups with both particles, independent source, full residuals and fresh ablations. Better feasible coupled size/index/depth fits pass without matching hidden particle parameters; structured real residuals remain visible and no independent particle truth is claimed. Follow-up evaluator guard review rejected empty/missing/extra metric sets, invalid comparison directions and nonnumeric/nonfinite boundaries. Relevant new tests and existing fresh scoring pass; valid scientific boundaries and reconstruction mathematics are unchanged. Previous parent acceptance/hash evidence remains archived rather than silently rewritten.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/colloidal_hologram_particle_geometry/ records commands, reports, actual extracted canonical figures and final hashes.
