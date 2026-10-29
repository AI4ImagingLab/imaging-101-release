# Provenance and validation

## Scope and reference role

This is a new physical inverse task: measured probe current versus bias to normalized electron energy distribution plus eight plasma parameters. It is not a spatial emission reconstruction, state tomography, or a resampled existing inverse. All 304 native Beckers2017 observations are retained. The native 64-point EEDF support is inferred using measured potentials. No true electron distribution, temperature, density or potential is supplied; `evaluation/baseline_reference.npz` is the independently reproduced standard method, never ground truth. The raw data contain only bias and current, with no fitted calibration, hidden object or privileged seed.

Public raw NPZ SHA256: `419b9ba490b6f844bf75e65071de5e7f12eec73e32b6cb5fd1528ea844493c51`. Source measurement `docs/notebooks/langmuir_samples/Beckers2017.npy` SHA256: `9377022c6625fcec961f69b360bc56b07f142de6e75240794649cb442fa6dd09`. It is a real helium laboratory characteristic already smoothed by the data provider. No claim is made about its unavailable raw precursor or smoothing settings. Probe area includes side surface and one end, matching the native notebook. No new synthetic benchmark observation was generated.

## Original source, papers, license and private archive

Private archive base is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch9_20261026/astro_earth_physics`. All archive/oracle paths below are relative to this private base; no upstream source or PDF is shipped inside the task.

Source: https://github.com/PlasmaPy/PlasmaPy, release tag v2025.8.0, exact remotely observed commit `b48a6d0861ce02f8c01c8dd91552d95b252e4c28`. Tag HTML and its currentOid are archived as `archives/PlasmaPy__PlasmaPy/tag_page.html`. This is a verified historical release, not a claim that the remote main branch is current. Complete source archive `archives/PlasmaPy__PlasmaPy/source.zip`, SHA256 `0ee4637e527ac8a00df21b57b4d99a92f6f8472c8cbb2421ba7dd4b68e012481`; extraction `archives/PlasmaPy__PlasmaPy/PlasmaPy-2025.8.0`. The installed oracle's diagnostics/langmuir.py SHA256 `dd17feda6e5eb931b2f7faca67f6d807ecca937166319b7ade7126be1edfa4de` equals the archived tagged source exactly. Installed package version is 2025.8.0. BSD 3-clause license with patent protections is preserved in LICENSE and attribution in NOTICE; the measurement is included in the licensed source distribution.

Physical references: Mott-Smith and Langmuir, Physical Review28,727–763(1926), DOI10.1103/PhysRev.28.727; Druyvesteyn, Zeitschrift für Physik64,781–798(1930), DOI10.1007/BF01773007. The full 37-page 1926 paper is archived privately as `archives/PlasmaPy__PlasmaPy/MottSmith_Langmuir_1926.pdf`, SHA256 `0f990fbbf01e3952c4dfb52969a560c8b4b4519d881099545e7f11315d511f20`, from https://ulcar.uml.edu/eeas/sp2012%28Burke%29/Mott-Smith_Langmuir_1926_PR.pdf. Its extraction is the adjacent TXT. The linear squared cylindrical current and logarithmic Maxwellian current sections were reviewed against the executable formulas. The 1930 publisher page confirms the double-differentiation inversion; its full text is subscription-only and was not claimed as downloaded. The exact differential expression is supplied by the native source docstring and independently derived/checked using the isotropic collection integral. Beckers' TU/e thesis URL was investigated but returned HTTP403; no claim of full-thesis retrieval is made. Data/method binding is established directly by the complete author-supplied PlasmaPy notebook `docs/notebooks/diagnostics/langmuir_analysis.ipynb`, which performs this first single-Maxwellian analysis and displays the same values.

## Exact original execution before local extraction

Capture driver `run_langmuir_original.py` imports the unchanged original PlasmaPy module and calls `swept_probe_analysis(Characteristic(bias*V,current*A),area,'He-4+',bimaxwellian=False,visualize=True,plot_EEDF=True)` on the exact full measurement. It additionally saves original ion fit, first/refined windows and fits, current curves, and native get_EEDF outputs by calling original functions. It neither replaces original formulas nor alters input observations. No cached notebook output substitutes for execution.

Original environment: `oracles/mttime_env/Scripts/python.exe`, Python3.11.11, Windows10.0.26200, NumPy2.4.6, SciPy1.17.1, Astropy8.0.2, PlasmaPy2025.8.0, Matplotlib3.11.2. This separate maintainer-only environment was extended for oracle verification only; the shared cleaned runtime was not installed into or upgraded. Source/dependency/backend metadata and original warning are in `oracles/langmuir/oracle_environment.log`. All CPU, OPENBLAS/OMP/MKL threads1, no GPU. The deterministic original method uses no seed.

The initial exact run completed in0.4354345s, exit0. Its original capture and metadata are frozen under `oracles/langmuir/first_capture/Beckers2017_original_outputs.npz` and `_original_run.json`. Before implementation, `oracles/langmuir/acceptance_before_extraction.json` fixed stage tolerances and all Judge thresholds. Original fixtures then populated `test/fixtures/upstream.npz` and `notebooks/reference_outputs/upstream.npz`. Both copies have explicit consumers: the former supports stage/physics/pipeline tests, the latter supports notebook comparison only. Their unused upstream hot_fraction=None object field was omitted, allowing every shipped NPZ to load with allow_pickle=False; immutable original captures retain the complete raw record. Original extrapolated electron current contains intentional NaNs above maximum measured current, solely a visualization mask. Tests compare its finite physical interval and compare unmasked mathematical predictions separately; no submission/reference output contains NaN.

A fresh unchanged original rerun was captured in `oracles/langmuir/original_repeat.log`, exit0, with every numerical capture identical (NaNs compared equal). `repeat_comparison.json` records per-array equality. This later repeat supplements the initial independent pre-extraction fixture; it is not presented as having occurred before extraction. The optional original second bimaxwellian dataset was investigated independently and archived under the same oracle directory, but it is not a second task or an undeclared observation. The selected task remains the complete first single-Maxwellian experiment; its current fit and EEDF comparisons are preserved.

## Equation/function/test traceability

| Paper or algorithm stage | Original implementation | Local executable mathematics | Validation |
|---|---|---|---|
| Isotropic retarding electron collection, Maxwellian logarithmic current | 1926 collection theory; _fit_func_lin_inverse | physics_model.distribution_current/electron_current | independent Maxwellian quadrature agrees analytic exponential; known-temperature fit |
| Cylindrical squared-current OML collection | get_ion_density_OML/extrapolate_ion_current_OML | solvers.reconstruct ion polyfit; physics_model.ion_current/densities | frozen original coefficients/current, analytic slope-density and SI scaling |
| Sorted unique characteristic | Characteristic.get_unique_bias/sort | preprocessing.unique_characteristic | exact measured equality; independent duplicate averaging case |
| Plasma/floating potential and saturation samples | get_plasma_potential/get_floating_potential/get_*_saturation_current | reconstruct full-gradient/argmin/min stages | all four scalar outputs exactly equal original; no cropped gradient |
| Two-stage Maxwellian temperature | extract_exponential_section/get_electron_temperature | fit_temperature and reconstruct strict window updates | original first/refined window arrays and separated current; fitted physical temperature |
| Thermal/Bohm density estimates | get_electron_density_LM/get_ion_density_LM | physics_model.densities | original densities, independent thermal-current inverse and area/current scaling |
| Druyvesteyn differentiation and normalization | get_EEDF | physics_model.normalized_eedf | frozen EEDF/energy exact; analytic quadratic curvature and signed normalization |
| Submission and all metric definitions | independently specified interface | standalone evaluation/judge.py | exact/double-scale known cases, every-quantity mutation, malformed/evaluator/report failure tests |

## Numerical and physical review

Fresh cleaned/reference relative differences: `{"V_P": 0.0, "V_F": 0.0, "I_es": 0.0, "I_is": 0.0, "n_e": 1.3475461117619807e-16, "n_i": 1.5360475015509095e-16, "T_e": 0.0, "n_i_OML": 1.8891314354156382e-16, "reconstruction": 0.0, "energy": 0.0}`. EEDF, energy, temperature, both potentials and both saturation proxies agree exactly; all density differences are below1.9e-16. Repeated original outputs are exact. Stage tolerance choices were frozen before extraction: EEDF rtol1e-12/atol1e-14, ion fit rtol1e-12/atol1e-12, physical fit temperature/densities rtol1e-7, raw/potential selections exact. Intercept x0/y0 have a gauge ambiguity in y+(V-x0)/T, so parity compares predicted current and temperature, not arbitrary redundant intercept coordinates. Local curve_fit preserves the native initialization, unbounded Levenberg–Marquardt defaults and two-stage stopping convention.

Independent physical checks verify the energy collection kernel against a known Maxwellian current, the full-sweep derivative sign and square-root energy weighting against a quadratic current, preservation of negative curvature, descending energy/absolute trapezoidal normalization, known exponential temperature, mA²-to-A² conversion, exact exposed-area convention, charge/mass scale, density/current scaling, and observed-only native window selection. No calibrated noise model was supplied. No target-derived calibration is used.

Important native approximations are retained and disclosed. Distribution differentiation uses total measured current, assuming ion curvature is negligible; it does not use the fitted electron-only current. The plasma potential is a discrete maximum-gradient proxy; the floating potential is a minimum-absolute-current sample. Electron saturation current is the measured total current at the selected plasma potential. The ion saturation proxy is the global observed current minimum. These approximations explain disagreement between electron, Bohm-ion and OML-ion densities; no hidden correction, forced quasineutral equality, or truth label is added. The notebook shows this discrepancy and compares the recovered EEDF to the ideal Maxwellian curve from its fitted temperature.

## Evaluation and negative controls

Judge requires every one of ten arrays: EEDF/energy `(1,64)` and eight scalar `(1,1)` float64 arrays, all finite and real. It uses only NumPy and standard library, no reference solver imports, hidden paths relative to its own file, and caller-relative prediction/report paths. Missing keys/files, wrong shape, float32, complex/object, NaN/Inf, unavailable hidden assets and report-write failures return exit2, with empty metrics when malformed/evaluator errors occur. Valid predictions that miss any metric return1. A zero or doubled EEDF, altered scale or any changed scalar fails; every required quantity has an independent mutation test. Known exact and doubled-scale metric values were checked independently.

Boundaries are nonzero numerical tolerances, not mechanical1.1×zero self-error: EEDF L2-relative1e-7, energy maxabs1e-10eV, maximum scalar relative1e-6, integral error1e-7. Native independent repeats and fresh discrepancies justify these tolerances. There is no implicit registration, interpolation, clipping or scale fit. Agreement to a standard-method reference is not evidence of physical truth.

## Commands, tests and active notebook

Final command logs `oracles/langmuir/final_command_0.log` through `_2.log` record shared-environment `python -m src.main`, standalone Judge, and `python -m pytest test/ -v -p no:cacheprovider --basetemp <private unique D: path>`. All exit0;21tests passed. The shared runtime contains no PlasmaPy/Astropy; AST/call-role audit `call_role_audit.json` shows only local mathematics with permitted general numerical primitives. Shared CPU/backend/environment details are in `cleaned_environment.json`.

Notebook command: `python -m jupyter nbconvert --to notebook --execute notebooks/helium_langmuir_electron_distribution.ipynb --ExecutePreprocessor.kernel_name=imaging101_shared --ExecutePreprocessor.timeout=180 --output helium_langmuir_public_contract_executed.ipynb --output-dir <private oracle directory>`. PASS, exit0. The private kernelspec's argv explicitly points to the shared Python; notebook prints that executable and NumPy version. The external nbconvert controller emits Windows zmq loop warnings, but all code executes in the declared shared kernel. The final executed notebook is private `oracles/langmuir/helium_langmuir_public_contract_executed.ipynb`, not a duplicate shipped notebook. Active fresh full inverse took 0.004374200s;180s timeout comfortably covers full CPU reconstruction and all plots without assuming every task must run instantly.

Eleven cells implement six story sections and seven computed figure groups: full observation/derivatives, OML and two temperature windows/current residual, EEDF/upstream/Maxwellian and density comparison, all-quantity metric table, fresh first/refined fit ablation,32 full noise inversions and3 sampling-stride inversions. Each plotted comparison is computed; no cached reference is substituted for a fresh result. Noise fractions0,1e-5,5e-5,1e-4 multiply observed current dynamic range with RNG20261026,8replicates each; these are explicit sensitivity studies rather than calibrated measurement errors. Perturbed EEDF comparison interpolates only common support for that figure, clearly distinguished from the strict Judge. No failures were hidden. Sampling ablations retain their own inferred support and do not replace the original benchmark.

Notebook original/fresh values are identified in `notebooks/reference_outputs/fresh_metrics.json` (fresh identified runtime, Judge metrics, ablations and perturbation records). Private extracted PNGs and contact sheet under `oracles/langmuir/figure_review` were visually inspected: readable units and labels, all observations present, meaningful real curves and residuals, no zero placeholders. README and notebook rendered previews were inspected in Chromium using a private setContent helper, with screenshots README_math.png/notebook_math.png. Browser session was closed.

## Documentation, privacy and final gates

README/plans and all notebook source/metadata were scanned for control characters: zero tabs, bells, backspace, formfeed or CR. Official math scanner found76formulas, zero delimiter/context issues after changing notebook display math to Jupyter-compatible delimiters. MathJax3.2.2 and KaTeX0.16.22 each report zero parser failures; full document previews render76formulas in4documents with zero failures. JSON/SVG/HTML evidence is `oracles/langmuir/document_math*` and `preview_final`. Equations were also semantically checked for signs, factors, energy ordering, temperature/J conversion and area/mA conversions; successful parsing alone was not used as correctness evidence.

README gives every raw key, acquisition parameter and complete output contract, with no hidden reference names/paths or GitHub URL. All constants match source and tests. Plans describe actual shipped signatures, defaults, returns, shapes, units, side effects, current mask convention, selected optimizer, and notebook-only comparisons. No standalone metadata JSON or generator is present. There are no upstream imports, checkpoints, opaque domain-library calls or cross-task dependencies. PDFs, complete source, notebooks/log captures, isolated environment and review PNGs are outside the task. LICENSE/NOTICE are retained. The private task_inventory.json indexes final files/hashes.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Exact source/paper/data provenance and mapping | passed | source tag/commit/hash, full1926paper, native notebook/source and stage table |
| Original and cleaned exact-observation execution | passed | pre-extraction capture, independent repeat, final CLI |
| Frozen upstream parity and independent physics | passed | stage fixtures, analytic integral/derivative/units cases |
| Constants/units/precision/window conventions | passed | complete native sweep and exact support, explicit full ion mass |
| Standard NPZ/Judge interface and fresh pass | passed | final_command logs and judge_report.json |
| Negative input, poor prediction and evaluator contracts | passed | every-output mutation and malformed/error tests |
| Public allowlist sufficiency and no leakage | passed | README/raw/key audit; no fitted inputs or hidden answer |
| Template/acquisition/output/no metadata JSON | passed | final inventory and README contract |
| Shared minimal runtime and no domain shortcut | passed | actual shared CLI/notebook and call-role audit |
| Unit/metric/parity/physics/pipeline tests | passed |21passed |
| Active notebook, scientific comparisons, visual QA | passed | full nbconvert,7figure groups,76rendered formulas |
| Result role/metric/threshold consistency | passed | reference explicitly not truth; original/fresh identifiers |
| README/both plans/source consistency | passed | semantic equations and signature review above |
| Final inventory/notices/private archive separation | passed | inventory and LICENSE/NOTICE; no release edits |


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 7 computed PNG outputs, 76 formulas. Private evidence: research_batch9_20261026/acceptance/helium_langmuir_electron_distribution/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Solver-visible URL boundary review (2026-10-29)

Software documentation, standard-toolbox archives and mixed code/data archive links were removed from public README references while preserving attribution and license notices. The sources remain here for maintainer provenance; they do not add solver calibration.

- source: https://docs.plasmapy.org/en/stable/notebooks/diagnostics/langmuir_analysis.html; role: software documentation/source or mixed code-data archive; moved to maintainer provenance
