# Provenance and validation evidence: lopeg_doppler_brightness_mapping

## Scope and physical novelty — passed

The spatial unknown is positive photospheric brightness on 4584 Roche surface elements, from Stokes I rotational line distortions. This is not scalar line-parameter fitting. It is distinct from the existing EHT uv-imaging, solar PFSS exterior extrapolation and wavefront tasks. The other LO Peg task reconstructs a different unknown from a different spectroscopic channel; this task does not import it or consume its inferred surface image.

## Paper and exact original source — passed

Original repository: https://github.com/folsomcp/ZDIpy . Exact checkout commit `86d9f98c426a709f3298bab95fda5f6f5e65e111`. The live remote API commit/HEAD response was actually obtained and archived in `verified_remote_commits.json`. Git ls-remote encountered transport failure; no successful ls-remote is claimed. Numerical oracle imports the private checkout at this commit, not an installed third-party reconstruction service.

Requested source method paper: Folsom et al. (2018), MNRAS 474,4956–4987, DOI 10.1093/mnras/stx3021; archived author manuscript is https://arxiv.org/pdf/1711.08636v1, submitted 2017-11-23. Oblate geometry reference: Cang et al. (2020), A&A 643,A39, DOI 10.1051/0004-6361/202037693; archived https://arxiv.org/pdf/2008.12120v1 (the served PDF carries a later compile date, explicitly recorded as the retrieved v1 URL rather than represented as the final publisher PDF). Both papers were web verified and their task-relevant forward/entropy/geometry sections reviewed. Source comments additionally identify Skilling & Bryan (1984), MNRAS211,111, DOI10.1093/mnras/211.1.111, and the positive/negative entropy of Hobson & Lasenby (1998).

The sample is explicitly the author-distributed **LO Peg August 2014 example**. Its input filenames and reduced-spectrum headers identify LOPEG. The method paper is the source's requested citation, not a claim that LO Peg or these exact observations appear in a particular paper figure. No published historical numerical score is substituted for this reproduction.

## Code and real-data rights — passed

The original root LICENSE is GNU GPL version 2 and is preserved byte-for-byte as task LICENSE, with source credit and adaptations in NOTICE and module headers. The sixteen normalized `.prof.norm` data files are distributed as part of the same licensed repository. No separate restriction or conflicting rights notice was found. We rely on that repository-wide distribution license for these exact bundled numeric observations and preserve attribution; we do not claim ownership of the underlying telescope exposures or redistribute unprovided echelle spectra. Hashes for each consumed original file are in `zdipy_observation_provenance.json`.

The raw observed profiles contain real mean-line intensity, circular polarization and diagnostic null columns. Only legitimate selected channel(s), reported errors, measured velocities and Julian dates are bundled. Stokes V and diagnostic null are omitted entirely from the brightness task. All arrays are float64 and batch-first. The 89 measured velocity samples are selected inclusively from the original Vr+[-80,80] interval, without interpolation. The source's already normalized profile files are reused; no raw-data renormalization, new noise, generated hidden object or target-derived calibration is introduced.

Bundled `data/raw_data.npz` SHA256: `e407c8dd0c24031d84d645af606d9fc5eb94044281e34c1a54a959cde019f1fa`. Original CSV-like observation filenames/dates and their hashes are separately indexed. `baseline_reference.npz` is a **method reconstruction**, never truth; `ground_truth.npz` is intentionally absent because actual surface truth is unavailable.

## Original exact-observation execution — passed

Private command: `oracle-runtime/Scripts/python.exe freeze_zdi.py brightness` with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. `freeze_zdi.py` loads the unmodified original `zdipy.py`, `core.mainFuncs`, `core.memSimple3` and source observation readers. It sets public source configuration flags to fitBri=1,fitMag=0, targetValue=1 and max iterations20. All fit initializations are the source's default unit brightness and zero field; init-from-file is off. Observer wrappers copy inputs/outputs at the original loop/update boundary but do not change any numerical function, parameter update or stopping condition. Cases are private `zdipy_brightness_frozen_case/`; source library remains in `ZDIpy/`. Source strength fit and full selected reconstruction execute, not merely import or a different demo.

The original unmodified runtime uses Python3.11.11, NumPy2.4.6, SciPy1.15.3, CPU single-thread; full NumPy/BLAS backend output is `zdipy_original_backend.txt`. Shared SciPy1.17 removed lpmn and cannot run the unchanged original spherical basis. This is a dependency API issue, not an upstream physics repair: no original source correction was made. The isolated oracle environment is private and never imported by cleaned modules, tests, Judge or notebook. Original stdout logs, numeric fixtures and source/environment hashes are in the private archive.

Original full selected run elapsed 2.576015200s, exit0, 3 iterations. Original summary [iteration,entropy,C/N,Test,area-mean brightness,area-mean absolute contrast,area-mean |B|] = `[3.0, -0.002037162278611254, 0.9999979536097119, 3.804149293747309e-09, 0.999842241916724, 0.010080316511419064, 0.0]`. Original input matrices, geometry, line calibration, first entropy/gradient update, complete current-image diagnostics and final physical maps/profiles were frozen **before local implementation**. Earlier independent original runs and the frozen capture agree to floating-point precision; initial brightness full-source comparison is elementwise exact. No cached paper score establishes this gate.

An additional independent original run `capture_zdi_calibration.py` records local unscaled intensity, weak-field derivative kernels, surface continuum weights and viewing vectors at initialization. These contain only known acquisition geometry and the line strength calibrated from visible measured I, with brightness one and B zero, so they are target-independent evaluator calibration. Their role is independent observation-consistency scoring, not a source of surface truth. The local runtime never reads them. Captures are frozen numeric data; normal tests do not execute or import the original source.

## Extraction, mathematical checks and discrepancies — passed

The local code retains the selected Roche construction, ring area quadrature, viewing projection, differential rotation, rational Humlicek line approximation, weak-field sign/units, normalized continuum derivatives, entropy weights and the explicit ten-direction Skilling--Bryan control/update. It does not call a computational-imaging package. SciPy contributes general quadrature, associated Legendre values, eigensolvers and scalar Brent optimization only. There is no ready-made stellar forward/inverse solver in the cleaned dependency list.

`extract_zdi.py` is a private, auditable source-extraction utility: it selects numerical geometry/profile/entropy functions, omits source data readers/writers and fitted-map initialization, rejects the unselected target-entropy branch, replaces removed lpmn by the explicit Pprime recurrence with general lpmv, and makes the already-disabled post-disk instrumental convolution an explicit documented no-op. Its output contains executable formulas, not an upstream import or opaque wrapper. Local `preprocessing.reconstruct` supplies the bound observations and explicitly controls every update and stopping decision. Both tasks carry their own independent files, with no cross-task import.

New Legendre identity: for x=cos(theta), `(x*x-1)*dP(l,m)/dx=l*x*P(l,m)-(l+m)*P(l-1,m)` and dP/dtheta=-sin(theta)*dP/dx, with P(l-1,m)=0 when m=l. It preserves Condon–Shortley sign and harmonic normalization. Analytic l=1 values and finite differences at x=-.83,-.2,.12,.91 pass; full Cartesian harmonic basis agrees with original at rtol1e-9/atol1e-11. The shared midpoint grid avoids x=+/-1 singularities. No physical model, coefficient truncation, sign or data is changed.

Source inconsistencies investigated and preserved explicitly:

- The source name radiusEq is inconsistent with the Roche formula's polar normalization; the same formula, constants and .72 adopted radius are preserved. The independently adopted velocity scale vsini/sin(i) is not forced to equal the period/radius-implied value. This conditional model is not a new radius inference.
- Brightness packing returns an alias, so its last proposed image is applied in place before the loop stops. Magnetic packing returns a copy and its final proposal is not unpacked after convergence. This original output convention is retained and documented. Archived final source spectra refer to the evaluated image. The Judge and fresh plots independently reproject the actual returned spatial output; no archived spectrum substitutes for its measurement fit.
- `C/N` means 1424 sample-normalized chi-square, not residual degrees of freedom. The LSD variances are marginal; diagonal weighting is an adopted assumption, not proof of independence.
- Source maximumBright1.01 is unused in positive traditional brightness mode; no upper clipping is imposed. The selected actual brightness map can exceed1.01. The filling-factor slice is empty. Field strength is unrestricted and signed through its harmonic coefficients.
- Weak-field, one effective spectral line, fixed atmosphere/brightness for magnetic inversion and adopted shear/geometry are limitations. Agreement with the standard method is not true-structure accuracy or a uniqueness claim.

### Equation/stage traceability

| Paper/process | Original file/function | Local function | Validation consumer |
| --- | --- | --- | --- |
| Roche shape, curved area, effective gravity (Cang oblate model) | geometryStellar.starGrid, GetSurfaceArea, calcRclatOblate | physics_model.starGrid numerical methods | original geometry fixtures; sphere4pi, normals |
| Sheared rotating view and projected velocities | geometryStellar.visibleGrid, getCyclesClat | physics_model.visibleGrid, getCyclesClat | original first view/projArea and full profiles |
| Continuum limb/gravity weighting | lineprofileVoigt.limbDarkening, diskIntProfAndDeriv | physics_model same selected formulas | original response; spatial reprojection; independent Judge |
| Local Voigt + instrument Gaussian | lineprofileVoigt.localProfileAndDeriv | physics_model.localProfileAndDeriv | original initial/final spectra and response |
| Weak-field polarization, Folsom Stokes model | lineprofileVoigt.updateProfDeriv | physics_model.updateProfDeriv and disk integrals | coefficient FD, zero B and B linearity when selected |
| Measured equivalent width and local depth | readObs.getObservedEW; mainFuncs.fitLineStrength | preprocessing.prepare, calibration.fitLineStrength | frozen original calibrated strength |
| Vector harmonic spatial representation | magneticGeom.initMagGeom/getAllMagVectors/Cart | physics_model.magSphHarmoics + legendre_values | original Cartesian basis; l1 analytic radial field; norm invariance |
| Weighted likelihood C, derivative | memSimple3.get_c_gradc | solvers.get_c_gradc | original first step and forward-response FD |
| Positive/signed entropy and curvature | memSimple3.get_s_grads | solvers.get_s_grads | original first step; independent scalar entropy FD |
| Skilling-Bryan search/whitening/control | memSimple3.searchDir,diagDir,control,updateImg | solvers corresponding explicit functions | frozen first proposal, full iterate diagnostics/final spatial map |
| Source stopping/proposal convention | mainFuncs.mainFittingLoop | preprocessing.reconstruct | original final profile, final physical output and C/N |
| Output/metrics | original actual numeric map, no source NCC report | standalone evaluation/judge.py | analytic metric cases; strict malformed/poor/result/report tests |

## Numerical parity and independent physics — passed

Source-comparison tolerances are stated in test source: geometry/line calibration rtol1e-13, profile1e-8 with atol1e-12, response1e-9 with atol1e-13, first update1e-9 with atol1e-8, spatial map1e-9 with atol1e-8; trajectory rtol1e-8/atol1e-7 accounts for tiny eigensystem/Legendre differences in nonlinear control. These are numerical tolerances, not relaxed image-quality boundaries. Full frozen inputs precede extraction; preliminary local probes diagnosed parity, then the explicit complete test contracts and final acceptance run were executed. We do not represent later notebook metrics as earlier upstream fixtures.

Fresh full-map maximum absolute difference from the original is 0 relative brightness. Measured output Judge metrics: `{"brightness_ncc": 1.0, "brightness_nrmse": 0.0, "observation_chi2_per_sample": 0.9999986664942541}`. Full iteration and forward/derivative comparisons pass. Independent tests verify spherical area4pi, outward unit normals, analytic l=1 harmonic pattern, spherical/Cartesian magnitude invariance, Legendre derivative recurrence FD, entropy first/second derivatives, selected likelihood response FD and physical spatial-output forward closure. The continuum-normalized brightness derivative includes its denominator and passes FD at separated surface cells. Pixel count is not claimed to equal independently resolved spatial degrees of freedom. The magnetic target0.8 sensitivity exhausts the20-step budget without meeting both convergence criteria; its plot is explicitly labeled budget reached and the actual residual is reported. The standard target1.0 converges.

## Reference and thresholds — passed

No actual truth exists. Hidden `evaluation/baseline_reference.npz` contains the reproduced original spatial output and independent acquisition kernels used for scoring; it is explicitly labeled standard inverse. Each required brightness cell is evaluated, with no crop, phase rotation, rescale, mean subtraction or sign alignment. NCC is flattened cosine similarity. NRMSE is RMS error divided by the reference component's max-min range. A nonzero numerical bound1e-7 and NCC>=.999999999 are used for this self-reference task rather than mechanically setting zero tolerance. Actual original/cleaned parity above is orders of magnitude tighter. The separate actual-spatial-output observation chi-square/sample must be <=1.002, above the method's1.001 target relaxation only by a small declared numerical/proposal allowance. Thresholds are not evidence of unique surface truth.

Zero-norm magnetic estimates receive NCC0 and a finite dynamic-range error, and fail as valid poor predictions; nonpositive brightness is malformed. Valid uniform, scaled, reversed-cell-order and wrong-sign/contrast fields fail. Every magnetic component has its own NCC/NRMSE thresholds so a weak component cannot hide behind another. Missing key/file, wrong shape, float32/complex/object dtype, NaN/Inf, unavailable evaluator assets and report-write failure are explicitly tested. Judge returns0/1/2 according to the required contract, uses allow_pickle=False, contains no solver import, and succeeds from a foreign cwd. Invalid inputs have empty metrics and descriptive errors.

## Actual final commands — passed

- exit 0; 2.776730700s; cwd `D:\yan1\agent\AutoSkill\imaging-101-release\tasks\lopeg_doppler_brightness_mapping`; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main`; log `final_brightness_cli.log`.
- exit 0; 0.347017200s; cwd `D:\yan1\agent\AutoSkill\imaging-101-release\tasks\lopeg_doppler_brightness_mapping`; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; log `final_brightness_judge.log`.
- exit 0; 0.326370900s; cwd `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics`; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe D:\yan1\agent\AutoSkill\imaging-101-release\tasks\lopeg_doppler_brightness_mapping\evaluation\judge.py --prediction D:\yan1\agent\AutoSkill\imaging-101-release\tasks\lopeg_doppler_brightness_mapping\output\reconstruction.npz --report D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics\brightness_judge_foreign.json`; log `final_brightness_judge_foreign.log`.
- exit 0; 9.976076800s; cwd `D:\yan1\agent\AutoSkill\imaging-101-release\tasks\lopeg_doppler_brightness_mapping`; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -v -p no:cacheprovider --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics\pytest_brightness_final`; log `final_brightness_pytest.log`.
- exit 0; 30.743940800s; cwd `D:\yan1\agent\AutoSkill\imaging-101-release\tasks\lopeg_doppler_brightness_mapping`; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m nbconvert --to notebook --execute notebooks/lopeg_doppler_brightness_mapping.ipynb --output lopeg_doppler_brightness_mapping_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=imaging101-runtime --ExecutePreprocessor.timeout=180`; log `final_brightness_nbconvert.log`.

All30 tests passed, covering unit, independent metric/Judge, original stage/update/parity, physical derivatives and full reconstruction. CLI wrote the contract file; independent Judge and foreign-cwd Judge passed. CPU execution used shared Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, with single-thread backend as captured in `zdipy_cleaned_backend.txt`. No original domain package is installed in the cleaned shared runtime. Original dependencies live in a private environment and are not public requirements. Notebook kernel is the same shared runtime (`imaging101-runtime`), timeout180s, full actual executed notebook copied back to the canonical path and temporary executed output removed.

## Notebook review and scientific figure sources — passed

The notebook actively reconstructs all1424 samples, evaluates a fresh physical reprojection with the standalone Judge, computes target0.8/1.0/1.2 sensitivities using actual new runs of the same method, and exports three scientific groups. Group1 (six panels) covers measured/predicted signal, standardized residual map, selected-epoch fits, source/fresh epoch misfit and residual distribution. Group2 (six panels: original/fresh/difference brightness, coverage, area-weighted contrast histogram and real phase coverage) displays the actual spatial output. Group3 (six panels) contains allthree fresh sensitivity maps, actual likelihood convergence, source/fresh Test convergence and spatial-change/actual-fit comparisons. Displayed source/fresh labels distinguish the reproduced original result and this run; no historical paper scores are plotted as measured values.

The figures' actual arrays are separately saved in each `_source.npz`. Private `brightness_figures/` contains PNG, SVG, PDF and numeric files with a SHA256/byte manifest. Every panel was opened and visually checked for signal/axes/units/legends, paired comparability and complete plots. The notebook embeds those actual executed plots; no exported bitmap replaces a computation. The identical-brightness difference is an exactly zero comparison, not a missing method filled with zeros. Source/stored spectra versus final-proposal reprojection are explicitly distinguished. Sensitivity runs expose regularization/noise-fit dependence and do not assert their targets are real noise measurements.

## File inventory, provenance and semantic audit — passed

Private root: `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch4_20261021\astro_earth_mechanics`. `brightness_archive_index.json` indexes consumed original numeric captures, scripts, exact source hashes, paper files, backend logs, final command logs and figure artifacts. The frozen case source/control/full upstream checkout and downloaded PDFs remain private. `test/fixtures/upstream.npz` is consumed by comparing tests; compact `notebooks/reference_outputs/upstream.npz` is consumed by source/fresh comparisons; `fresh.npz` is the separately identified actual notebook result, not upstream or truth. Evaluator line kernels are consumed only by standalone Judge. No fixture or reference array lacks a consumer.

README visible tables cover every raw NPZ key, all acquisition constants and physical output conventions; no hidden reference array key/path or forbidden repository URL is disclosed there. Public requirements are the shared numerical subset only. No standalone metadata JSON, root wrapper, PDF, raw MAT/archive, second test/report tree or imported original package is present in task. Plans were derived from executable selected mathematics and reviewed for signs, entropy factors, wavelength/Gauss units, Doppler direction, surface basis, calibration, proposal aliasing, signatures, default values and actual caller flow; no proposed method is represented as shipped.

All mandatory AGENTS gates are passed for parent review: exact source/paper/data provenance; full original and local bound runs; frozen independent parity; applicable physics; local acquisition/inverse mathematics; CLI and strict Judge; visible allowlist; minimal CPU runtime; complete30-test suite; actively executed notebook and required comparisons; separate numeric result provenance; correct README/plans; final template inventory, private archives, license preservation and figure hash manifest. Parent acceptance remains separate from this author validation.

## Post-review markdown correction — passed

Parent review identified escaped-control corruption in the initial README display formula. The README generator now uses a raw formatted string. An additional byte-preserving audit found the same issue in two notebook markdown blocks, which were replaced using literal-backslash construction (chr(92)); generator source is corrected too. Actual Notebook code cells, execution counts and computed outputs were retained without reexecuting numerical code. The audit removes only genuine CRLF pairs before checking, so embedded bare CR, TAB, backspace and form-feed are detected. All group README/plans/notebook markdown now have zero unexpected C0 controls and zero naked LaTeX command tokens in mathematical spans. Private evidence: markdown_control_audit.json and audit_markdown_controls.py. Frozen accepted MT/HFITS files were scanned read-only and were unchanged.


## Parent acceptance review

Parent independently ran30tests (30 passed in10.04s), foreign-cwd Judge0 and source/inventory audit errors[]. Read Roche geometry, explicit line profile and maximum-entropy updates; inspected all3 actually executed scientific groups, original-stage fixtures and physical checks. Real16-epoch LO Peg Stokes I line profiles reconstruct4584 spatial surface-brightness cells; source final proposal versus last evaluated state distinguished and final model independently reprojected. Original standard map is not stellar truth. README/plans/notebook formula escaping repaired and byte-audited without changing any executed numerical code or output.

## Final boundary-schema compatibility correction — passed

Authorized final schema-only correction on2026-10-21: evaluation/metrics.json now stores the sole canonical numeric boundary for each metric under its top-level <metric>_boundary key, with directions in metric_comparisons. Judge read_thresholds actually reads these named fields and derives its report thresholds; no unused compatibility alias or duplicate numeric boundary is retained. Boundary numbers, comparison directions, acquisition/inverse algorithms, observations, truth/reference arrays and output contracts are unchanged.

New test_boundary_schema.py proves active consumption by changing a canonical NCC boundary and observing a valid failure, and proves missing boundary, missing comparison and unexpected metrics return evaluator error. These complement the existing per-component, physical/parity and malformed-submission tests. The pre-change Hawaii schema test was run and failed as expected; the migrated schema then passed. This was a harness-field compatibility correction, not a new scientific method or relaxed acceptance.

Protected-file hashes prove all src numerical files, README, both plans, raw observations and hidden NPZ references are byte-identical to the pre-correction snapshot. All submission and notebook fresh numerical arrays except intentionally varying runtime_seconds are exactly array-equal to the previous accepted run; fresh measured metrics and Judge report threshold values are exactly unchanged. Previous README formula/Cartesian-coordinate corrections remain intact. Actual final reruns:

| Command family | Seconds | Exit |
| --- | --- | --- |
| cli | 2.766793000 | 0 |
| judge | 0.351157200 | 0 |
| judge_foreign | 0.341558000 | 0 |
| pytest | 10.094522100 | 0 |
| nbconvert | 28.576361000 | 0 |

Complete pytest result: 35 passed. Full active nbconvert completed with no error outputs; canonical notebook retains actual executed outputs and the temporary executed copy is absent. Exact commands/cwd/logs remain private final_brightness_commands.json and final_brightness_NAME.log. Private boundary_schema_verification.json records protected hashes and numerical equality; the pre-change snapshots are boundary_schema_before/lopeg_doppler_brightness_mapping. All figure exports/numeric sources were regenerated by the active notebook and manifests refreshed. Original oracle assets were not rerun or changed for this schema migration. Parent re-audit of this final schema remains separate from the earlier scientific acceptance record above.

Latest schema/fresh artifact hashes (the earlier tables above identify their earlier accepted runs):

- evaluation\metrics.json: 7f4ab547fed16516a04c0d1434843b1d7da74592abe3f99514651641ba5a153e
- evaluation\judge.py: 2c893f12805e2ce7403040240beda28e874a03bbd7fa45ffcc1b3234b614706f
- output\reconstruction.npz: e10ba688138f2745084ee80bf36d8d9a1d42d05fb0fae4a1e66ee66d73cf3ee6
- notebooks\reference_outputs\fresh.npz: 6f65f24654c501cc82c4ada1412cfbcf86f51812a39540a38f6d1a8bb056a70b
- notebooks\lopeg_doppler_brightness_mapping.ipynb: bc56bc75fd9b5f3b3e88eb64d871023521a4863536ec058619a52fd17fb36ec5


## Current benchmark v2 evaluation, 2026-10-29

The v2 Judge retains full measured-spectrum chi-square<=1.002 and independently integrates curved Roche cell areas for relative entropy to the unit photosphere. Entropy cost may not exceed attainable baseline cost by more than1e-7 per total area. No ground truth exists. A different higher-entropy data-consistent field passes, while uniform/scaled/shuffled fields fail.25 unit tests passed. See lopeg_pytest.log.

This intentionally changes benchmark acceptance semantics while preserving the validated published reference algorithm and historical parity evidence. No observations, hidden reference NPZ files or stored reconstruction bytes changed. No full fresh blind Agent solve is claimed. Old public/evaluation files are archived privately under benchmark_readiness_20261029 before changes.
