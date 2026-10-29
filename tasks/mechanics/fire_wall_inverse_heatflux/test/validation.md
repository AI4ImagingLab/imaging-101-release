# Provenance and validation: real fire-wall inverse heat-flux field

Status: **ready for independent parent audit**. All local validation gates below have executed. This task reconstructs a spatial incident-radiation field from real thermal observations; it is not a scalar heat-transfer fit, calibration-only task, or forward boundary extrapolation. It is a closed-form inverse, so artificial iterative convergence/training is not introduced.

## Paper, source revision and rights

Published method: P. Dehghani and M. J. DiDomizio(2024), *HFITS: An analysis tool for calculating heat flux to planar surfaces using infrared thermography*, SoftwareX28,101934, final published version, DOI https://doi.org/10.1016/j.softx.2024.101934 . The article identifies original code version1.0.2 and the real dataset below. This task selects the maintained authors' original implementation at an exact newer revision, rather than claiming its HEAD is the historical paper-release tag.

Original code: https://github.com/ulfsri/HFITS , exact commit **62ae8d2d1998b700d9bc50af3f8dfbdf0f359161**, downloaded by network clone and independently verified with the actual remote commits/HEAD API in this session on2026-10-21. `verified_remote_commits.json` stores this verification. An attempted additional git ls-remote failed with a connection error; that failure is not described as a successful freshness check. The successful independent API response supplies the exact verified revision. Full original checkout remains private. Its actual license is **GPL-3.0** and is preserved in task LICENSE. NOTICE attributes the algorithm adaptations and the separate data license. No original source file is altered for the oracle.

Existing data: M. J. DiDomizio(2024), *Heat Transfer and Fire Damage Patterns on Walls for Fire Model Validation*, datasetv1.0.0 published2024-01-26, UL Research Institutes/FSRI, DOI https://doi.org/10.5281/zenodo.10543089 . Public record https://zenodo.org/records/10543089 ; actual Zenodo API metadata explicitly states **CC-BY-4.0** (`hfits_zenodo_metadata.json`). Published observations are real infrared wall-temperature fields already processed by the author, not new simulations and not a raw-camera calibration claim. The source recommends these processed MAT files for subsequent analyses.

Source member: `PROC/FLUXFIELD/HEPT_FLUXFIELD_proc.mat`, containing the heptane-fire processed thermal fields. The complete ZIP is13,564,747,218bytes; its advertised MD5 is21e30c1bdb6fcc1ad999cc1e3e7692a7. We **did not download or claim to verify that whole-archive MD5**. Instead, parse the complete ZIP64 central directory and all678entries, preserve its binary/index privately, then fetch only the selected member's exact compressed range. Selected compressed bytes508,847,822, uncompressed bytes509,375,101, local-header offset12,597,588,999, compressed-data offset12,597,589,067. Decompression verified **CRC32 d18640eb**, exact uncompressed byte count, and SHA256 **d40e78c9589b5f0482204f4cf1618885a6b638f1228b6c9baa7f6dd989166cd5**. `hfits_member_provenance.json` records URLs/ranges/license/CRC/SHA/download time313.2871642s. Range requests returned HTTP206. `hfits_zip_index.json` records complete member names/sizes/compressed sizes/CRC/header offsets; `hfits_archive_central_directory.bin` preserves raw directory bytes. Only about509MB of the13.6GB archive is transferred for the main data member. A separate ~0.8MB member `PROC/DATA/HEPT_STL_3_proc.mat` was fetched with its own size/CRC validation for an independent temperature-unit/ambient/time check.

Changes permitted by CC-BY4.0: extract a time/spatial window, preserve the original numeric temperature values, convert inch coordinate arrays to metres using exactly0.0254, and repackage observation arrays in NPZ. README and NOTICE provide attribution, DOI, license link, and modifications. No copyrighted paper PDF, whole upstream checkout, download archive, or original log is shipped inside the task.

## Physical unknown, bound observations and conventions

Unknown: incident radiative exposure q(t,y,x), **kW/m²**, at all25times and64-by48spatial target pixels (76,800values). It includes ambient incoming radiation; it is not net absorbed heat flux. Data are the author's `FLUXFIELD.zdata_temp_wall` field from HEPT_FLUXFIELD, shape(804,193,97)before selection. Source x/y positions are(97,)/(193,), half-inch spacing. Select time indices100:125, vertical64:130, horizontal24:74; exact unbatched temperature shape(25,66,50). Target is that window's interior pixels1:-1along both spatial axes. A measured collar is visible input, so every target finite-volume stencil has real neighbors. No extrapolated boundary, hidden seed, target-derived calibration, or privileged flux observation is provided to the solver. Gas/heat-flux arrays in the original MAT remain private and are not extra solver inputs.

The original file's `FLUXFIELD` does not include a separate time vector. The independently downloaded per-test DATA/time begins0,1,2,…seconds, and source documentation states all processed time series are shifted to ignitiont=0. Thus the selected field uses documented one-second samples100…124s. No resampling or float32 conversion occurs. Full public NPZ SHA256 `cb4cb9b1823c58df78dc6933d5b6f8e6eac9080fa868bd7c0307a5d95202f3e1`. Exact upstream-bound temperature array parity is tested, including source spacing/time-step constants. All release arrays are float64 with batch-first interfaces.

**Unit discrepancy investigated, not silently copied:** the original data README labels zdata_temp_wall asdegC, but the actual stored field at ignition has min292.198095238095, median292.558963133641, max293.038105478751. Those are Kelvin-scale room temperatures. Independent ambient DAQ measurements are about20.77°C (293.92K), consistent within~1.7Kwith the processed thermography field, rather than a292°C initial room. At100…124s the bound stored temperatures range360.2612596006144…551.4192857142857K. Select the unmodified HFITS **Kelvin** input branch and do not add273.15. `hfits_temperature_units_check.json`, source data README and independent ambient MAT are archived. This is interpretation of existing absolute observations supported by an independent measured ambient scale, not an invented observation or arbitrary post-fit offset.

Fixed model assumptions, publicly specified:304steel sheet,d=.000793m,epsilon=.94,rho8050kg/m³,k(T)=8.1119ln(T/1K)-30.377W/(mK),cp(T)=1000(.1414ln(T/1K)-.3534)J/(kgK),ambient293.15K,hf=hb=10W/(m²K),sigma5.67e-8W/(m²K^4),spatial spacing.0127m,dt1s. These are the original existing **steel(Engineering Toolbox)** and **constant convection** options selected in the oracle's GUI-control values. They are nominal constitutive/boundary assumptions, not estimated true exposure. Source default FSRI material option is not used; no coefficient from a different source option is silently mixed in.

The source uses central-cell k(T)times LaplacianT, not a full variable-conductivity divergence; that modeling approximation is explicit in README/plans. The sheet is assumed uniform through thickness. Full-thickness PDE inversion, fitted spatial convection, or specimen-specific property re-estimation is not claimed.

## Oracle-first implementation and independent fixtures

Before creating local mathematics, privately `prepare_hfits.py` extracts exact bound real temperatures. Then `capture_hfits.py` imports the **unchanged** original `HFITS/Scripts/IHT.py` in an isolated oracle environment. A trivial Control.getadapter supplies the same scalar values as GUI controls; it does not replace a mathematical function or inverse update. It calls original `inverse_heat_transfer` on exact66-by50-by25temperature values, square .0127mspacing,1sstep, Kelvin, nominal Engineering Toolbox steel, constant convection. It captures the full interior output, k/cp/rho stage values and a two-component conductive-gradient stage before local code is written. The original cropped result is an **explicitly defined subproblem target**, not an evaluator's implicit crop. The original outer-pixel implementation is irrelevant because those measured-collar pixels are not targets.

Original timed inverse runtime **0.554531500s**. Independent original rerun has maximum absolute difference **0.0**. All original values are finite. Full original window including collar has q range4.471562549743884…28.349332781214805kW/m²; the declared interior target has5.10636128…27.8541957kW/m². Separate ranges are not conflated. `hfits_frozen.npz` was captured before local extraction. A later identical original rerun adds complete stdout/stderr archive coverage; `hfits_capture_console.log`/`hfits_original.log`/`hfits_capture_command.json` contain actual original execution, progress output, options, and exit0. Latest complete capture command wall time3.701306600s includes import and two runs; it is distinct from timed inverse runtime above.

Local `thermal_properties`, second-order temporal difference, explicit five-point Laplacian, four-term energy decomposition and direct inverse are readable NumPy mathematics. No original/domain library is imported in source, Judge, tests, or notebook. Full-pipeline maximum discrepancy versus independently captured original is **7.1054273576010019e-15kW/m²**. Forward/material/stage/pipeline parity tolerances were set from independent original captures before accepting the local result: relative/absolute2e-12for full flux and1e-14-scalematerial comparisons. The precision/backend change does not hide a changed physical model.

## Environments, runtime and archives

Private group root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/astro_earth_mechanics`. Original HFITS checkout/selected MAT members/capture scripts, ZIP index/central-directory bytes, metadata, original logs and environment reports are outside the final task. `hfits_archive_index.json` records each item's size/SHA256 and documented role. The complete source revision is recorded above; repository LICENSE/NOTICE is preserved.

Initial import probe in private NumPy2.4environment fails because original upstream dependencyfdm0.4uses removed np.math. It is captured as a **rejected environment**, not numerical success or a source-code repair (`hfits_rejected_numpy2_environment.log`). No shared package was downgraded. The successful private original oracle uses Python3.11.11,NumPy1.26.4,SciPy1.13.1,pandas2.2.3,h5py3.11.0,fdm0.4.0, plus pandas' required pytz/tzdata; GUI tkinter comes from Python. fdm is imported by the original module even though the chosen time derivative is NumPy gradient. These are upstream-only import dependencies, absent from the cleaned runtime. `hfits_oracle_environment.txt` records exact versions and NumPy BLAS/compiler details. The numerical backend/thread environment is explicit; no unverifiable GPU compatibility claim occurs.

Clean runtime is the prevalidated shared Python3.11.11/NumPy2.4.6/Matplotlib3.11.1 environment, Windows10build26200x86_64, CPU only. Shared numerical environment was not upgraded or modified. All commands set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. pytest/nbconvert/ipykernel are shared development tooling, not runtime requirements. A memory observation during validation reported17,290,400KiBfreephysical and39,531,348KiBfreevirtual/commit, above the6GiBminimum. Main MAT loading peaks below1.5GiB; no >1GiBwholearchive download or shared-runtime domain dependency occurs.

## Traceability

| Paper/source stage | Original unmodified symbol | Local symbol | Validation evidence |
|---|---|---|---|
| Temperature/material constitutive laws | IHT.k_metal,cp_metal,rho_metal selected steel option | thermal_properties | Captured full-stage k/cp/rho, analytic formula case |
| Thermal storage | inverse_heat_transfer, numpy gradient edge_order2 | time_derivative, energy_terms.storage | Independent quadratic-in-time exact derivative; source full flux parity |
| In-plane thin-plate conduction | calculate_gradients, summed/area | laplacian, energy_terms.conduction | Captured two-component gradient; quadratic spatial field Laplacian; collar locality |
| Two-face convective loss | inverse_heat_transfer, constant option | energy_terms.convection | Original exact options and complete numerical parity |
| Radiation/emissivity convention | inverse_heat_transfer, two-face radiative term | energy_terms.radiation | Independent ambient-equilibrium irradiance identity |
| Closed-form inverse energy balance | inverse_heat_transfer | reconstruct | Full original output at all76,800target values |
| Forward balance/units | Same thin-plate governing balance | predicted_storage; independent Judge arithmetic | Storage identity and standalone balance metric |
| Pointwise versus spatial calorimeter interpretation | HFITS thin-plate selected case | active notebook omission diagnostics | Fresh no-storage/no-conduction fields, no synthetic data or cached substitution |

All variables, signs,1/epsilonand1/1000factors, Kelvin fourth power, square spacing, dt/endpoints, and target-collar relation are defined in plans/README. No optimizer, regularizer, initialization, step size, or stopping criterion is invented for the closed-form inverse.

## Every hidden array and consumer

- `evaluation/baseline_reference.npz`: keyreconstruction,float64(1,25,64,48),kW/m², original unmodified exact-data HFITS output from the frozen capture. No actual true flux is available, so no ground_truth.npzis invented. Consumed by independent Judge and contract tests.
- `notebooks/reference_outputs/upstream_hfits.npz`: same independently identified original method field, consumed by active notebook overlay/parity metrics.
- `notebooks/reference_outputs/historical_author_estimate.npz`: keyincident_flux,(25,64,48),from separately identified archived author `zdata_hf_rad_i`on the same selected field grid. It has different fitted convection/processing and is **historical method output, not truth and not fresh**. Consumed only by the explicitly labeled notebook comparison. Never used by solver or Judge.
- `test/fixtures/upstream_hfits.npz`: input_temperature_kelvin,param_dx_m,param_dt_s,output_incident_radiation,output_k,output_cp,output_rho,output_conductive_gradient. Each is consumed by `test/comparing_test/test_upstream.py`; raw-input values remain exactly equal. Fixture SHA256 `a9b199675ed49ed8f1472cba4f465dcb65a6dd9c48da496025b7026ddf1a36e1`. No cleaned function generated an expected parity array.

## Metrics and numerical tolerance

Fresh exact metrics: `{"ncc": 1.0, "nrmse": 8.14362535444289e-18, "balance_nrmse": 2.012687964880763e-15}`. NCC=flattened cosine without mean subtraction. NRMSE=RMS field error divided by the reference dynamic range. All25times and all64-by48target pixels are scored; no mask/translation/rescaling/implicit crop is applied. The independent Judge separately reconstructs temporal derivatives using explicit second-order endpoint formulas, spatial curvature using a separately written stencil, and the energy RHS inW/m². Its balance metric is RMS(epsilon1000q-RHS)/RMS(RHS). It imports no solver code.

A self-reference baseline has NCC1/NRMSE0; literal1.1timeszero is inappropriate. Independent source reruns have zero difference and local backend/NumPy-port differences are~7e-15kW/m². Choose nonzero reference NRMSE tolerance1e-7,NCC>=.999999999,and independent energy-balance normalized residual<=1e-7. A zero uniform valid prediction is correctly rejected with exit1. These numerical tolerances test reproducibility, not independently known physical accuracy. The physical-model sensitivities and historical estimate differences are separately reported. No NaN/Infinity is emitted as JSON.

## Actual final execution commands

Commands run by private verify_hfits.py capture subprocess exit code/stdout/stderr/runtime, not PowerShell's response to diagnostic stderr. Log root is the private group root above.

| Check | Exact command | Exit | Actual seconds | Log |
|---|---|---|---|---|
| cli | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main` | 0 | 0.149996800 | `final_hfits_cli.log` |
| judge | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.145727600 | `final_hfits_judge.log` |
| judge_foreign | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe D:\yan1\agent\AutoSkill\imaging-101-release\tasks\fire_wall_inverse_heatflux\evaluation\judge.py --prediction D:\yan1\agent\AutoSkill\imaging-101-release\tasks\fire_wall_inverse_heatflux\output\reconstruction.npz --report D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics\hfits_judge_foreign.json` | 0 | 0.140245900 | `final_hfits_judge_foreign.log` |
| pytest | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v -p no:cacheprovider --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics\pytest_hfits_final` | 0 | 1.300825200 | `final_hfits_pytest.log` |
| nbconvert | `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m nbconvert --to notebook --execute notebooks/fire_wall_inverse_heatflux.ipynb --output fire_wall_inverse_heatflux_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=imaging101-runtime --ExecutePreprocessor.timeout=180` | 0 | 11.503890500 | `final_hfits_nbconvert.log` |

Final full test suite: **21 passed**. Tests cover every core function/material/temporal/spatial inverse stage, independent original-stage/full parity, analytic half-space-equivalent thermal equilibrium, quadratic derivatives, measured-collar locality, independent metric cases, CLI pipeline, malformed NPZ/missing key/shape/dtype/complex/NaN/Inf/overflow, zero-quality output, invalid/missing evaluator assets/thresholds, foreign cwd, report parent creation/write failure, and import-backend preservation.

The active notebook executes the whole25-frame target path and all selected ablations/sensitivities from local source. Timeout180s is justified to include kernel launch, all actual comparisons, scientific figures and PNG/SVG/PDFexports; recorded actual wall time is in the table above. Successful executed outputs are copied into the canonical notebook and the temporary executed output notebook is removed. No cache-preview path satisfies this validation. Nonfatal Windows ZMQ selector-thread/local-TCP warnings are preserved in logs; no notebook cell errors occurred.

## Scientific figures, numerical sources and semantic review

Three figure groups with six panels each are exported privately as PNG/SVG/PDF. `hfits_figures/figure_sources.npz` contains exact public observation arrays, fresh q, original q, historical author q, four fresh energy terms, no-conduction/no-storage fields, and allh=5/10/20sensitivity results. `hfits_figures/manifest.json` records file sizes/SHA256. All three rendered groups were actually opened and reviewed. Overlong titles were shortened; omission RMS uses a log scale to make the small real conduction contribution visible; image extents include the correct half-pixel edges around supplied coordinate centers. Final notebook exports were re-executed after these display changes.

Group1pairs measured temperatures/fresh flux at100,112,124s. Group2shows storage/convection/radiation/conduction, temporal mean budget, and freshly computed physical-term omissions. Group3distinguishes original numerical reproduction from historical author's different model estimate and recomputes boundary-model sensitivity. No missing method is zero-filled and no archived reconstruction masquerades as fresh. There is no artificial convergence plot for an algebraic inverse. The historical field is not used for thresholds or truth. The notebook explains both thin-skin and local-conductivity limitations.

Source/plan semantic review checked Kelvin and radiation reference; factor2andambient contribution; epsilonabsorbed-versus-incident scaling; W→kWfactor1000; cp/kg/m³units; sign of conduction; time-axis and second-order endpoints; spatial target coordinates/collar; k(T)timesLaplacian approximation; selected material/constant convection; callable signatures/defaults/returns/file paths; and actual notebook/CLIflow. Source/generator/Judge/test/notebook call-role audit finds no domain inverse-library call, upstream import, cross-task import, hidden-reference use in the solver, or matplotlib.use. No generator is shipped because observations are reused.

## Completion gate ledger

| Gate | Status | Evidence |
|---|---|---|
| Paper/version/revision/data hashes/equation-stage mapping | PASS | Exact paper version, verified remote code/API, full member CRC/SHA and traceability |
| Original exact-data execution before local extraction | PASS | Unmodified source, exact temperature window, original stage/full outputs and rerun |
| Independent parity and applicable physics | PASS | Material/gradient/full parity; quadratic derivative/equilibrium/locality checks |
| Constants/units/precision/coordinates | PASS | Stored Kelvin discrepancy investigated, float64 preserved, square grid/collar/time conventions tested |
| CLI and standalone Judge | PASS | Fresh declared output passes from task and foreign cwd |
| Judge malformed/quality/evaluator/report contracts | PASS | All invalid/overflow/zero-quality/configuration/write cases tested |
| Solver-visible information sufficiency | PASS | Four visible arrays and full model/units/output contract/metadata in README |
| No truth/target calibration/extra observation/URL leak | PASS | Temperature and coordinates only; Github URLs confined to hidden provenance |
| Minimal runtime/domain independence | PASS | NumPy/Matplotlib only, upstream original import dependencies isolated privately |
| Full tests | PASS |21passed, actual command/log above |
| Active full notebook | PASS | Complete raw-data reconstruction, comparisons and successful nbconvert |
| Metric/history/original/fresh provenance | PASS | Historical estimate/source oracle/fresh measurements separately labeled |
| Plans and semantic interface/math review | PASS | Actual closed-form runtime documented, no proposed methods |
| Final inventory/licenses/private archive cleanup | PASS | Task template, GPL3+CCBY notices, no source MAT/PDF/log/archive/temp notebook inside task |

Parent independent acceptance is still required; no claim of actual true fire-flux accuracy is made from method agreement alone.


## Parent acceptance review

Parent independently ran21 tests (21 passed in0.98s), foreign-cwd Judge0, inventory/call-role audit errors[], read energy balance/material laws/collar derivatives and full provenance, and visually inspected all3 executed six-panel figures. Exact author-code run on real CC-BY4 thermal observations, documented Kelvin source-label discrepancy independently checked. Incident radiation includes front ambient; central-cell conductivity model and nominal convection are explicit approximations. Historical author irradiance remains separately labeled, never truth.
