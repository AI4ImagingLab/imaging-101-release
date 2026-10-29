# Authoritative provenance and validation

## Fixed physical task and provenance

Recover the complete ordered402vertex/800facet three-dimensional Hertha surface,
three spin quantities, two2D AO offsets, two log gains, and one2D occultation
offset. All30relative curves/923photometric samples, two native150by150 image/PSF
windows, and18detected occultation chords are retained.135is the asteroid ID,
**not135lightcurves**. Early private process-description strings containing
“135lightcurves” are a preserved metadata mistake; numeric captures, source-only
freeze and all current public contracts use the actual30curves/923samples.
Data type: real author-reduced astronomical observations, not raw CCD frames and
not a newly generated simulation. `jittered` is an author reduction filename.

Paper: Viikinkoski/Kaasalainen/Durech, ADAM, A&A576,A8(2015),
doi10.1051/0004-6361/201425259; downloaded exact URL
https://arxiv.org/pdf/1501.05958v1 . Complete11pages/text were read, including
surface parameterizations, Fourier projection, data modalities and optimization.
PDF SHA256 `3cd6006f2bcf79d9c36a86d2c2312119113b37d56b7745db8f5f4696e058cc66`; 2374387bytes. The downloaded PDF's
page header prints **September10,2021 / copyright ESO2021**, whereas its arXiv
watermark identifies **1501.05958v1,23Jan2015**. Both facts are retained; the
reason for the differing typesetting/header date is unknown. Do not interpret
this as proof of the original2015 typesetting. The selected later author code
and explicit configuration define numerical details rather than a claim that
every later default equals the2015 paper. PDF/text/download record stay private.

Original repository https://github.com/matvii/ADAM exact remotely verified commit
`25810f767b9de349a356df520109f4705f87f7a7`; primary remote probe and source archive
are `../screen/matvii__ADAM/probe.json`, `remote_primary.html`, `source.zip`.
Archive SHA256 `ed2395ead5b06c94011068ae6e04e35560f3b07aa15b97c3e68160bafa8e3ca6`.
The original README declares CC BY4.0; NOTICE and mathematical source preserve
attribution. Separate observation redistribution rights remain **unresolved**;
no data CC grant is invented. This records authorized local research validation,
not an external publication/redistribution approval.

Maintainer archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch8_20261022/groups/astro_earth_mech/adam`. `archive_index.json` indexes paper,
source provenance, original data and run/capture evidence with hashes. Supplied
full source/PDF/archives/logs are outside the task. Public NPZ SHA256:
`fa26531b6fe7ccea1cdc0240af4d33ecfb61c507bdb43714df971046b73211ec`. `bound_data_provenance.json` lists each original
file hash/header, exact native window, every public array shape/type and NPZ hash.
Native AO file dimensions are257by258 and257by259; the author-selected window is
zero-based y/x50:200, corresponding to FITS one-based51..200. No quality-selected
new crop or withheld photometry/chord is used. Fixed grid/facets are representation
calibration independent of unknown radii; supplied PSFs/directions/distances were
available with acquisition, not estimated from hidden shape. The unused original
timing-error descriptive column is privately preserved, omitted publicly because
the selected estimator does not use it and its units are not explicitly supplied.

The selected `utils.c::read_lcurve` reader uses **supplied JD minus2443846.0**,
with no extra photometric light-time correction. Earlier “light-travel-corrected
photometry” prose was corrected in README/plans/notebook without changing values.
AO and occultation retain their separately documented source light-time correction.
No unverified claim is made about corrections before the author supplied LC file.

## Independent original execution before extraction

Native exact source runs in private Ubuntu24.04 WSL with GCC13, GSL/FFTW,
CFITSIO/WCS/LAPACKE/BLAS and private iniparser library. No upstream domain package
or these dependencies are installed into the cleaned Python runtime. Build/run
commands, dynamic libraries, flags, stderr/stdout and binary hashes are retained
in the corresponding process records. The O1original binary SHA256 is
`ef3ab9312c9608881ca33383c870b342e5c1501756b8650a00b7050634df24fe`.
Source C domain formulas are unchanged for oracle runs; only configuration input
filename aliases and output destinations changed. Seed/initialization is the
author deterministic ellipsoid43.56/43.56/35.64km, latitude52deg,longitude272deg,
period8.40060hours, phase0 and occultation shift(0,-10)km; no truth seed is input.
Source O2compiler ICE, initial missing-iniparser build and supplied tracked-binary
history are retained as separate failed/different paths, not silently called
successful exact-source runs. The source O1 and O0 native runs are successful.

| Actual execution | Native exit | Runtime seconds | Private record/log directory |
|---|---:|---:|---|
| Original exact-source O1 | 0 | 153.23650748100044 | `source_original/process.json` |
| Independent source O1 repeat | 0 | 152.0284428219993 | `source_independent_repeat/process.json` |
| Independent source O0 | 0 | 212.06172813599915 | `source_O0_repeat/run_process.json` |
| Passive original O1 capture | 0 | 154.49691585699838 | `passive_stages/process.json` |
| Passive original O0 capture | 0 | 217.00363968799866 | `passive_stages_O0/process.json` |
| Native FFT/input capture | 0 | 10.380327515998943 | `input_stage_capture/process.json` |
| Author subdivision configuration | 0 | 75.75073819300087 | `source_subdivision/process.json` |
| Historical supplied executable | 0 | 414.4101708000003 | `original_native_process.json` |
| First active NB, completed fit then plotting failure | 1 | 578.675107099989 | `notebook_execution/process.json` |
| Selected full pytest37 | 0 | 260.7729919999838 | `validation_selected_lpmv/pytest_process.json` |
| Final full active nbconvert | 0 | 635.9756927000126 | `notebook_final_execution/process.json` |

Original command example: `/tmp/imaging_b8_adam_source_build/adam source_original.ini`;
the full command/cwd appears in each JSON. Native selected config allows100trials,
not100guaranteed accepted updates; the original O1stops after35trials. Passive
captures add only numeric exports; original outputs remain byte-identical to
uninstrumented full runs. O0/O1 are independent full compiled reruns, not a
same-output self comparison. Input capture additionally records all native FFT
arrays. Original supplied executable reaches a different99trial historical state;
it is preserved but not conflated with the compiled exact source reference.

The source `fit_oct.c` writes outshape **before** remeshing final accepted
coefficients for its state/lightcurve exports. That old snapshot differs from
accepted mesh by max0.5172838523858019km. The target baseline uses the fullprecision
accepted mesh captured after the source's own final remesh, not a silently mixed
old mesh/new spin state. `independent_physics_checks.json` records both roles.

## Pre-local acceptance and independent physics

`source_only_acceptance_freeze.json` is frozen at
`2026-10-22T12:29:34.402147+00:00`, explicitly before any local solver existed. Baseline is
the actual original accepted O1state; no actual complete asteroid truth is known.
`evaluation/baseline_reference.npz` therefore must never be labeled ground truth.
Independent original O0/O1 differences set each field boundary to1.5times its
measured error. The complete objective gate is1.1times the larger independently
reproduced original objective. Same-seed/self comparison zero error did not set
these nonzero limits. No boundary changed after any local failure.

| Fixed metric (all <= boundary) | Boundary |
|---|---:|
| mesh_x_range_nrmse | 0.0014460159943030682 |
| mesh_y_range_nrmse | 0.003762820021147654 |
| mesh_z_range_nrmse | 0.0073342806756775958 |
| latitude_deg_absolute_error | 0.025692699873374636 |
| longitude_deg_absolute_error | 0.023868969684599506 |
| period_hours_absolute_error | 1.9895085312526817e-07 |
| ao_offsets_rms_arcsec | 8.1092826863431947e-05 |
| ao_log_gains_rms_dimensionless | 0.00023829220386925636 |
| occultation_offset_rms_km | 0.10617342130228428 |
| complete_weighted_objective | 32.059766710378398 |

The source-only initial ellipsoid has XYZrange-NRMSE
0.022040895435039894/0.04349559656326695/0.08003306466400616;
zero0.2919549431277298/0.25255589771676734/0.2772103206813305;
shuffled vertices0.4274219894596985/0.3633693586206282/0.39841721574248634.
These are rejected, not accommodated. All402vertices/allcoordinates and all
spin/nuisance outputs are graded; no spatial masking, alignment, rescaling or
cropping is fitted. Longitude alone wraps physically by360degrees. Coordinate
NRMSE is RMS/dynamic range of that coordinate, not a pooled range or L2 ratio.

Independent original triangle Fourier integrals versus barycentric quadrature
have regular-frequency errors<=8.32e-16 and centered derivative errors<=3.57e-11.
An explicitly nonsampled tiny-frequency case(u1e-7,v2e-7) has2.4434e-5integral
and large derivative cancellation; it is retained as a limitation, not silently
“fixed” or represented as a sampled acquisition point. Complete sampled FFT,
normal equations, Jacobians and full forward objectives are separately tested.
Real SH agrees with independently normalized general Legendre functions within
9.99e-16; octantoid centered derivatives within1.12e-9. Complete original finite
differences validate negative-residual Jacobian sign (axis relative3.355e-9,
latitude1.229e-7); original JTJ and JTr recomputation errors2.76e-15/1.60e-15,
first damped system relative residual1.593e-16. Captured damped matrix AFTER
LAPACKE factorization is not mistaken for a pristine normal matrix.

`test/fixtures` consumers are explicit: original_geometry.npz->test_geometry;
original_fourier_observations.npz->test_fourier_preparation;
original_joint_initial.npz->test_joint_stages (complete residual46595, selected
Jacobian rows, complete normal/rhs/firstupdate); original_chord_fallback.npz->
test_judge missing-chord regression. The latter is a post-extraction additional
independent unchanged-original primitive capture, not falsely pre-local evidence:
four square/offset/chord known cases GCC native0; helper TDD initially failed
missing implementation then passed original values within2e-15. The complete15
perturbation forward objectives agree with source-parity-validated local native
math within4.96e-9, including3states using the native missing-positive-chord
sigmoid penalty. Such fallback is an optimization heuristic, not a physical
endpoint; detailed missing endpoints are explicitly NaN and counts saved. Normal
bound fits have real intersections. The final scalar objective and JSON metrics
remain finite.

## Equation/function/test traceability

| Paper requirement / selected source convention | Upstream file/function | Local executable function | Independent/parity case |
|---|---|---|---|
| Real SH octantoid and analytic coordinate derivatives, paper surface model | octantoid.c; sph_harm.c | physics_model.harmonic_basis/octantoid | original_geometry; known lpmv/centered FD |
| Spin/ecliptic/camera geometry | rotate.c::rotate; utils.c::Calculate_Frame_Matrix | rotation_matrix; local kernel frame | original geometry and rotation FD |
| Facet visibility and normalized scattering | Calculate_LC.c; utils.c facet normals/visibility | lightcurve_operator; local analytic derivative kernel | complete LC residual/J rows; mean-normalization FD |
| Analytic facet Fourier projection, paper Fourier model | Calc_FTC.c; Calculate_AO.c; Calculate_AOs.c | triangle_fourier; ao_operator | direct quadrature/FD; complete AO model/J/normal |
| Native window FFT/DC/PSF preparation | read_ao.c; FFT preparation helpers | preprocessing.image_fourier/load_observations | all4native FFT groups complete <=6.34e-16 |
| All detected chord intersections and source missed-chord fallback | Fit_Occ.c::Fit_Occ/find_chord_dist_simple; find_chord.c | occultation_operator; independent Judge chord geometry | original OC residual/J;4fallback native cases;15forward states |
| Concavity/dihedral/center of mass penalties | convex_reg.c; dihedral_reg.c; center_of_mass.c | mesh_penalties | original penalty/J; centered FD |
| Degree weighted octantoid regularization | octantoid_reg.c | solvers.octantoid_penalty | closed formula/FD; original penalty row |
| Complete residual weighting and custom damped update/stopping | fit_oct.c | solvers.residual_and_jacobian/reconstruct | original full r,JTJ,JTr and first native lower update; full CLI quality |
| All output quantities and separate physical consistency | original final accepted remesh/state | main.run; independent evaluation.forward_objective | standalone/foreign cwd Judge; known original objective; poor controls |

Stage tests preserve strict residual atol1e-10/rtol2e-10, normalized sampled
Jacobian differences LC/OC/regularizer<1e-10, AO<3e-7 (C99 complex versus
std::complex cancellation at near-axis edges is independently exposed), normal
relative<1e-9, rhs<1e-10, first lower-triangle update<1e-8. These stage tolerances
do not replace the frozen completefield/output gates. Initial stages are far
more accurate than nonlinear stopping-state uncertainty; full quality can fail
despite passing initial math. Native multiplication order for sphere angular
steps is retained; harmonic formulas/Condon--Shortley normalization are explicit.

## Local runtime and scientific qualifications

Shared Windows x64 Python3.11.11/NumPy2.4.6/SciPy1.17.1/Matplotlib3.11.1,
CPU one numerical thread; actual kernel interpreter/backend/version/C++sourceSHA
and reconstruction time are saved in fresh_metrics.json and final nbconvert
process JSON. `environment.json` records NumPy/SciPy BLAS configuration and the
existing compiler; no shared installation/upgrade occurs. Core operators are
readable local source, not a domain API/framework wrapper. Python owns all46595
residuals/311unknowns, analytic chain rule, regularization, initialization,
additive custom LM, damping factor10, accepted/rejected decisions, native lower
symmetric solve and stopping; no upstream executable/checkpoint is invoked.

Explicit parent-authorized environment exception: selected facet/visibility/
intersection/analytic derivative math is embedded as readable C++17 `CPP_SOURCE`
in physics_model.py. Existing MSVC14.29.30133 x64/SDK10.0.19041.0 compiles /LD /O2
/fp:precise /EHsc /std:c++17 /MT /W3 into a source-hash temporary cache. Child-only
INCLUDE/LIB locate the installed compiler and SDK; no runtime download or private
library dependency. Compile/source/cache/log details are indexed, and cold-cache
math probe is recorded separately. Standard-library complex lifetime allocation
is explicit. The source COMearly return is moved after its already-present frees
as a **memory-only correction**; no numeric expression is changed. Unselected
albedo/Hapke branches are specialized to fixed false. `local_kernel_build/
extraction_map.json` maps every selected native function to original file/hash;
`symbol_call_role_audit.json` inventories Python/C++symbols and their roles.

Known source comment/code discrepancy: COMcomment mentions a0.01clip but the
selected D=NULLactual branch does not clip, and this behavior is retained. The
original nonsampled Fourier cancellation and stale export are disclosed above.
The selected Legendre primitive is generic SciPy lpmv, not a domain solver.
An explicit SHrecurrence path with native grid/lower solve completed tests but
failed the two fixed pole gates (latitude0.08373279762632535deg and
longitude0.1363500997476308deg); its full37suite was36pass+1fail/native1 in
272.7590881s. Original logs and source hashes remain in validation_run. The
failed default NPZ was later overwritten by actual standard CLI; no preserved
failed full NPZ is falsely claimed. Its actual recorded pole numbers are saved
as a labeled diagnostic. Switching to independently verified lpmv/nativegrid/
lower solve passed the unchanged gates; no per-angle arithmetic mixing or
post-hoc threshold widening is used. Earlier upper-triangle and old-grid/lower
diagnostics are also separately archived, never conflated with the selected run.

## Fresh measurements, tests and notebook

Full selected `python -m pytest test/ -v` actual native0/260.7729919999838s,
37cases including the actual standard default CLI and independent foreign-cwd
Judge. After adding the missing-chord regression, all37light cases (excluding
only the already executed full CLI case) passed in26.69s; the additional
broken-truth-link contract regression was red then green. Final current light
suite/process and parent full-suite acceptance are recorded below/in final_evidence.
The link regression mocks only Path.is_symlink because this Windows account
lacks link creation privilege; it prepares actual raw/reference/prediction files
so unrelated missing-data errors cannot produce a vacuous pass. Judge now uses
truth.exists()or truth.is_symlink(), then strict validation; invalid actual truth
never falls back. This pure malformed-reference branch was changed after final
NB startup; the normal existing-baseline numerical path is unchanged.

Judge independently validates allrequired keys/shapes, float32/64only, finite
values, typed finite nonnegative boundaries excluding bool/string, hidden paths,
actual-truth priority, zero/missing/wrongdtype/wrongshape/NaN/Inf/extreme-finite
overflow, report-write failure and CLI0/1/2. Invalid/evaluator errors have empty
metrics. It imports no solver or compiled kernel. Independent regularized forward
original objective29.145242464284717 differs from native29.14524246398036 by
3.04e-10. Field metrics are independent of nonlinear optimization.

| Current active-fit metric | Measured value |
|---|---:|
| mesh_x_range_nrmse | 0.00096476126630029384 |
| mesh_y_range_nrmse | 0.0025067309254862406 |
| mesh_z_range_nrmse | 0.0049043672601490167 |
| latitude_deg_absolute_error | 0.017151096523093656 |
| longitude_deg_absolute_error | 0.016193896797062735 |
| period_hours_absolute_error | 1.3293060163732662e-07 |
| ao_offsets_rms_arcsec | 5.3768139558947871e-05 |
| ao_log_gains_rms_dimensionless | 0.00015410974314812058 |
| occultation_offset_rms_km | 0.069608763049292199 |
| complete_weighted_objective | 28.494259934220199 |

First real active nbconvert computed the complete fresh fit in532.217999s and
passed all10gates, but then failed on a missing-positive-chord sensitivity case;
its actual exit1/578.6751071s is preserved, never counted as success. The missing
native penalty was independently tested and retained, without shrinking perturbations.
Diagnostic preflight reused explicitly labeled **already computed** fresh arrays
to exercise allremaining plot/sensitivity paths; it is not a reconstruction
validation. The final full nbconvert status is **passed**. It uses explicit
private kernel argv=[researchPY,-X,utf8,-m,ipykernel_launcher,...], JUPYTER_PATH,
%matplotlibinline, and subprocess.run(shell=False) with separate stdout/stderr,
actual exit/runtime and source/canonical hashes. The harmless Proactor/ZMQselector
and local TCP-kernel warnings are retained, not hidden as failures or “fixed”.
Final Notebook actively recomputes the entire selected inverse, no cache-preview
substitution, and retains alloutputs. Chapters: Problem,Data,Physics,Method,
Results,Conclusion. It compares initial ellipsoid/original O1/independent original
O0/fresh fields and all30LCfits, both AOresidual images and all18chords, actual LM
history, failed recursion-path measurements and15full-data forward sensitivities.
Author subdivision full native config (578vertices1152faces) has different
support/weights; it is honestly an archived original configuration comparison,
not an unimplemented fresh method or a same-weight controlled ablation. No
scientifically relevant group is replaced by zero or historical-as-fresh output.
No new synthetic observations are generated (unchanged observations with model
parameter sensitivity), so generate_data.py is intentionally absent.

Six actual figures are exported PNG/SVG/PDF plus numeric NPZ into private
`figures/`; plot_provenance.json links each consumer/source/hash. AllsixPNGwere
actually opened; signed residuals are zero-centered, the complete physical axes
and color units shown, and no evaluation support is selected by visualization.
Minor3Dview/tick density and missed-chord annotations were fixed before final
execution. Retained outputs/array roles are documented in
notebooks/reference_outputs/measurement_roles.json. Plans were semantically
checked against real equations/signs/units/normalization/native lower update,
actual signatures/defaults/output paths and alltrial branches, not merely names.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
|1Paper/revision/data/equation map|passed|This report, primary probe, paper/data hashes, traceability table|
|2Original and cleaned exact observations|passed|Independent source native0; full selected pytest CLI native0|
|3Independent parity and physics|passed|Original frozen fixtures, quadrature/FD/normal checks, stage suites|
|4Constants/units/precision/coordinates|passed|Public tables, full FFT/grid tests, rotation/offset conventions|
|5Standard CLI and fresh Judge|passed|Full pipeline test; current actual active fit all10gates|
|6Poor/malformed submissions and evaluator failures|passed|Independent Judge tests, truth priority, overflow/config/CLI cases|
|7Public allowlist sufficiency|passed|Allobservation/calibration keys and conditional model/output definitions in README|
|8No privileged input/GitHub leak|passed|Public NPZrole audit; README/requirements URL scan; no target radii/seed|
|9README metadata/output/NPZlayout|passed|Exactkey table, acquisition metadata, all5output arrays; no metadataJSON|
|10Minimal runtime/environment exception|passed|Explicit existing-MSVC exception, local math-only DLL, research Python CPU|
|11Local math and symbol/call-role audit|passed|Embedded readable operators/custom Python LM; independent Judge; role inventory|
|12Unit/parity/physics/pipeline tests|passed|Full37native0 plus final independent contract additions/light-suite evidence|
|13Active complete nbconvert|passed|Actual notebook_final_execution/process.json; no artifact-only success claim|
|14Result/metric provenance|passed|Separate original/O0/fresh/failed-recursion roles and fixed pre-local thresholds|
|15README/plans/source semantic match|passed|Explicit rotation/SH/scattering/objective/LM and signature review|
|16Final inventory/provenance/notices|passed|Final inventory/archive/figure hashes, NOTICE, temporary products outside task|

Parent independent acceptance remains pending until recorded by root; worker
completion does not claim parent approval. `final_evidence.json` and private
inventory contain final command/status/hash pointers. Wider data redistribution
permission remains unresolved as disclosed, not a claimed cleared license.

## Final current evidence and semantic review (24 October 2026)

Current full suite includes the two final malformed-truth regressions: **39 passed**;
actual native return code, command, elapsed seconds and one-thread environment
are recorded in `validation_final_current/pytest_process.json`, with full stdout
and stderr adjacent. This rerun includes active full standard CLI reconstruction
and independent foreign-working-directory Judge. No inverse kernel or frozen gate
changed after the successful complete notebook. All runtime/evaluation/test files
were hash-checked against this current suite's starting snapshot. A final Judge
docstring-only correction is recorded in judge_docstring_equivalence.json:
source k=4 enters exp(-2*k*d), giving steepness8/km. Removing docstrings leaves
identical executable AST; no numeric rerun is implied or needed for this wording.

`qa/formulas_render_check.json`: all76 extracted formulas across README, both
plans and all notebook markdown cells pass MathJax3.2.2 and KaTeX0.16.22 with
zero parser/scanner issues. Allsix final PNGs were opened individually during
this completion review; `plot_provenance.json` binds their PNG/SVG/PDF/NPZ hashes
to actual canonical-notebook and reference-output hashes. Independent inventory/
public-contract audit has errors=[] and its standalone Judge passes all10metrics.

`cold_compile_process.json` binds current embedded C++source SHA to successful
existing-MSVC compilation from an initially absent private cache. An initial
private probe wrongly selected zero frequency and returned NaN after successful
compilation. The selected acquisition explicitly discards DC; this unsupported
probe is retained, and no math was changed to accommodate it. Independent
nonzero-frequency triangle quadrature differs by1.6653345369377348e-16. This is
a kernel compile/physics check, not an additional full inverse run. The full
current CPU pipeline and earlier active notebook provide actual execution evidence.

Manual semantic review reconfirmed the coordinate/spin matrices, arcsec/km
conversion, photometry JD-only versus corrected AO/occultation timing,
per-curve normalization, signed residual/J=-dr/dp convention, regularizer
outer squaring, native lower-triangle solve, accepted/rejected damping and
stopping conditions. Both plans' callable/default/shape and actual source caller
flow agree. The malformed actual-truth branch is strictly an evaluator contract
change; existing-baseline normal numerical objective is unchanged.

Final packaging removes only task-local bytecode and pytest cache directories.
The canonical active executed notebook and its six embedded scientific figures
remain; temporary executed notebook/compiler/capture/PDF/log files stay private.
Private `READY.json` contains every final task-file SHA256 and evidence pointers.
Worker status is ready for independent parent review; no parent acceptance or
wider data redistribution authorization is claimed.


## Parent independent private acceptance,2026-10-24

Parent independently ran complete39tests native0/254.734995800s, including fresh full multimodal CLI and standalone foreign-cwd Judge. All76formulas passed KaTeX and MathJax. All6final scientific PNG groups were individually opened and reviewed: complete observations, full ordered surface, original numerical/support comparisons, all30photometric curves, both AO images/all18chords and full convergence/sensitivity. No failed recursion-path result is labeled accepted, and missing-chord sensitivity states are explicitly marked with native fallback costs. Canonical notebook SHA256dfc6189d4cff3c111ac19f83bc23c8c151671c65963f8f6818265e3317870e32 is byte-identical to actual complete native0/635.975692700s nbconvert execution with the shared numerical kernel. Parent verified all36worker file hashes before evidence addition, standalone10required gates, the independent cold math-kernel compilation and nonzero-frequency quadrature. Independent read-only semantic review SHA2566d03d53b9261832feb256fdb58e29027a1ccd61f443902abefbcd75467e0246f checked method, units, Jacobian signs, explicit LM and all nuisance fields; its minor4-vs8steepness docstring issue was corrected with executable-AST identity evidence, not a mathematical change. Public lightcurve times preserve supplied JD-minus-epoch only; AO and occultation corrections are separately identified. Exact original O0/O1variability fixes numerical limits; there is no independent shape truth or physical precision claim. Existing MSVC C++17environment exception, copyright, separate data-rights uncertainty and arXiv2015watermark/PDF2021layout-date ambiguity remain explicit. Private acceptance awaits complete20-task release.
