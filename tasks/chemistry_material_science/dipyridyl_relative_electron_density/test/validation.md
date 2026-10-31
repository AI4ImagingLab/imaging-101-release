# Original provenance and validation evidence

## Binding and scientific scope

Private archive P is `research_batch7_20261022/groups/root/gsas_readonly`
under the sibling maintainer repository. Capture/build/review scripts are in
its parent directory. They are maintainer-only and never imported by shipped
source, tests, generator, Judge or notebook. The archive index records files,
hashes and consumers. Complete upstream source, source binaries, downloaded
papers, logs and API responses remain outside the task directory.

This task recovers a full periodic real72x24x96 electron-density contrast from
4508 existing measured single-crystal X-ray reflections. It is distinct from
finite-crystal coherent-displacement imaging and orientation-averaged small-angle
scattering. No atomic coordinates are supplied or fitted. There is no independent
density truth, so the hidden baseline is an explicitly identified method result.
It is never relabeled ground truth.

## Source identities, paper access and license

| Source | Exact identity | Evidence |
|---|---|---|
| Original author software | https://github.com/AdvancedPhotonSource/GSAS-II commit cf739c09ddb965fd619b18d7ab49d6ac6c2c9d2e | Remote API verified; source.zip SHA25623ddb60aae04c334aed74b795e4a43ac1f5bacdf901c0d34b77dba73c0ad4e03; all529 original files audited unchanged |
| Original author data/tutorial | https://github.com/AdvancedPhotonSource/GSAS-II-tutorials commit924e9eb82dc6bdeebf55b2867ea30558ee530d7c | Exact remote version in tutorial_head.json; downloaded observations and tutorial individually hashed |
| Selected method paper | Toby and Von Dreele2013, J.Appl.Cryst.46:544–549, DOI10.1107/S0021889813003531 | Full author-uploaded reprint read, especially section4.6 and Appendix A; genesis_author_access.json |
| Primary author reprint | https://www.researchgate.net/publication/245022847_GSAS-II_the_genesis_of_a_modern_open-source_all_purpose_crystallography_software_package | Brian Toby upload17March2014; full text read via web access, not merely abstract/metadata |
| Charge-flipping precursor | Oszlanyi and Suto2004, ActaA60:134–141, DOI10.1107/S0108767303027569 | https://arxiv.org/pdf/cond-mat/0308129v1; actual17-page938906-byte PDF SHA256fd83a8518767cf38918a40bdbdcfda67d433558312e7d453630198328f1e6c37 |
| Full author tutorial | https://advancedphotonsource.github.io/GSAS-II-tutorials/CFXraySingleCrystal/CFSingleCrystal.htm | tutorial.html and full extracted text; actual P21 symmetry and initial teaching-model distinctions read |
| Bound real FCF | S2dipyridyl.fcf,230776 bytes | SHA25684b25b11c1bd95cf652b5f1a228a32e919e3b766fa5f572e87a370362849872d |

Publisher and direct reprint PDF access attempts returned403/401 or connection
closure; those are retained as failed downloads. Successful full author-text
access is distinguished from an unavailable byte-hashable2013 PDF. Metadata-only
OpenAlex/DOI access is not substituted for reading the full method. The actual
2003 arXiv precursor PDF was downloaded/read separately. The selected modern
double-threshold algorithm differs from the precursor's F000 treatment; it
is not falsely called an unchanged implementation of only the2004 algorithm.

Full UChicago Argonne software terms and copyright2010 are retained in LICENSE.
This is a custom Argonne license, not a guessed BSD label. The native GENHKL
legacy Regents of University of California/Los Alamos copyright notice is
retained verbatim in NOTICE. Data are offered in the author's repository;
no separate external-data redistribution grant was identified. Software terms
are not invented as a universal data license. This is local research packaging;
no external publication or author contact was performed.

## Input-leak investigation before implementation

The FCF contains measured Fo-squared, uncertainties, calculated Fc-squared and
phases. Only hkl, measured Fo-squared and reported sigma are converted. The
FCF scale was globally fitted to a structural model. Divide **both** all signed
Fo-squared and sigma by the measured RMS Fo-squared before binding; no model
scale remains as absolute-density calibration. Public intensity RMS is1.
All136 negative finite observations remain, and every one contributes to the
physical residual. No Fc, phase, F000, electron count or atom coordinate is input.

The author tutorial explicitly identifies actual P21 unique-axis b, whereas
P21/c in the introductory CIF/teaching setup is preliminary. Use actual P21.
Cell c=23.1175 follows exact FCF rather than rounded23.118 in other documents.
CIF metadata were read privately; its atoms were not imported. GPX was archived
but never opened. Initial raw-scale/P21c probes are labeled preliminary and do
not establish the final normalized/P21 bound task. The normalization, symmetry,
unknown, output scale, grid and all-observation contract were fixed before
local inverse implementation in binding_discrepancies_prelocal.json and
frozen_acceptance_criteria.json.

The public NPZ is a lossless conversion after this documented scalar
normalization; its exact hash is in task_binding_frozen.json and final inventory.
Private unbatched bound_existing_observations.npz has a different container
hash; never confuse it with the public batch-first NPZ. Public input keys are
exactly hkl(1,4508,3),intensity(1,4508),sigma_intensity(1,4508).

## Original runtime and genuine failed routes

The unmodified current author CIFhklReader rejects this ordinary FCF because
of an unconditional continue after its _shelx_hkl_file test. Its native exit1
is preserved. The successful verification uses generic PyCifRW to convert
only the three legitimate observation quantities into the documented native
RefList interface, then runs the actual unmodified GSASIImath.ChargeFlip,
geometry, compiled GENHKL and original origin routine. This is a declared
input-format adapter, not a patched inverse or successful claim for the failed
reader. Exact source and all529 original files remain ZIP-byte-identical.

Private original environment: Python3.11.11,NumPy1.26.4,SciPy1.15.3,
PyCifRW4.4.6 and requests2.32.5. No upstream package is installed into the
cleaned shared runtime. PyCifRW initially failed to build without vcruntime.h;
restoring existing VS2019 include/library paths allowed the private build.
Official author win_64_p3.11_n1.26 binary archive supplies native geometry.
Its libquadmath dependency required an isolated official conda-forge
m2w64-gcc-libs-core5.3.0-7 runtime and private DLL directories. Setup commands,
compiler, exact binary and DLL hashes are in oracle_setup.json,binary_record.json,
oracle_install2.json,dll_loader_record.json and logs. No shared library upgrade.

The normalized P21 original source commands use the private capture wrapper
run_gsas_relative_original.py and one numerical thread. Seed20261022;
original GUI cancellation callback represents fixed source budget and preserves
the native Rcf<5% stop. Read-only inspection captures states and intermediate
arrays; no source file was edited to insert captures.

| Final bound original run | Native result | Role |
|---|---|---|
| bound20 and independent_repeat20, GUI stop at Rcf<=20% | Both0,24 actual completed updates; complete repeated arrays bitwise | Independent source repeat and author-like cancellation comparison |
| extended200, seed20261022,stop0,budget200 | Native0,process19.123097500007134s | Selected original full200-update baseline |
| seed23extended200, seed20261023 | Native0,complete200 | Independent source-only seed stability |
| seed24extended200, seed20261024 | Native0,complete200 | Independent source-only seed stability |
|125 native GENHKL hkl cases | Native0 | Independent geometry/absence/multiplicity/phase fixtures |

Commands, exit codes, actual runtimes and separate stdout/stderr are in
relative_original_capture_runs.json and relative_*_process.json. Complete
numeric states/results are in relative_P21_original_runs/. Preliminary probe
and early FCF-reader failures remain separate. The result is finite but the
selected source budget ends near19% diagnostic; do not claim convergence or
actual electron-density accuracy merely because native execution succeeds.

## Pre-extraction criteria and independently frozen fixtures

frozen_acceptance_criteria.json was frozen2026-10-21T23:52:04.925658Z before
any cleaned inverse, SHA2566c3a8428fc953954277db1da48df3f19f51a323d636f36adbac780fd12f96d11.
It records all source identities, observation binding, exact seeds, whole-map
alignment and criteria. Full independent seed comparisons give NCC.95052997
and.94183072, rangeNRMSE.01214718 and.01317199. The24-update GUI run gives
NCC.96157828/NRMSE.01070515. Freeze minimum NCC.9 and maximum
rangeNRMSE1.1*.013171987765645874=.014489186542210463.

The signed all4508-intensity relative L2 residual is.00010239649895416404,
including the negative-measurement floor; maximum is1.1times that,
.00011263614884958044. Relative mean tolerance is1e-6. Source float32 output
serialization NRMSE9.606955311027432e-10 justifies nonzero finite handling.
Stage tolerances rtol5e-7,atol1e-9 are frozen independently, not selected after
observing local divergence. No threshold was relaxed during extraction.

A random-phase map exactly reproduces positive amplitudes yet has source-only
NCC.12136234/NRMSE.05119286, failing density criteria. This proves positive
modulus agreement alone is not an informative reconstruction test. Independently
known fractional shift(.37,-.29,.41) is recovered within1e-11 voxels with
rangeNRMSE2.15e-13. No fitted gain, global sign, crop or selected-voxel mask.

The original pre-extraction upstream_stages.npz contains initialization,
full source state1/2/7/25/100 intermediates and complete history. It does not
duplicate the baseline final map. source_geometry.npz contains original metric,
all spacings, grid and125 actual compiled GENHKL cases. Their consumers are
test/comparing_test/test_original_stages.py. Baseline_reference.npz stores the
unmodified original full200 method map. No actual truth file is fabricated.

## Extraction, backend divergence and qualification

The first shared NumPy2.4 extraction suite gave12passed/3failed: initial and
short stages agreed; later full free-running phases diverged. This was not
accepted or hidden. Independent execution of the same local functions inside
the private original NumPy1.26 backend then matched every recorded original
state and final map **bitwise**, recorded in local_trajectory_numpy1.26.4.json.
This isolates a backend-sensitive nonlinear trajectory rather than changing
the published formula to accommodate failed tests.

Private Python3.12.13 with NumPy2.4.6/SciPy1.17.1 and official author
win_64_p3.12_n2.2 native binary was then prepared. The unchanged source performs
all200 updates but fails in postprocessing because NumPy2.4 removed the
reshape keyword newshape. Its actual native exit1 is retained under
modern_source_runs/modern200 and backend_divergence_before_modern_run.json.

Only a **private compatibility copy** changes the first findOffset reshape
call from keyword newshape to positional shape. No inverse equation, FFT,
initialization, iteration or numeric convention changes. Exact before/after
hashes and the one-line diff are in modern_reshape_compatibility.json. Both
qualified original full200 runs return native0 and are bitwise equal.
Their entire initial state, every captured update, complete history, origin
roll and final map match shared local NumPy2.4 **bitwise**, despite the original
native ABI using Python3.12 while cleaned runtime uses3.11.

modern_source_local_parity.json and modern_source_migration.json record this
evidence. The original529-file source archive remains unchanged. The new
independent upstream_modern_stages.npz fixture is explicitly **post-extraction**,
SHA2564470a63b5cd8d6ca65153c2b1f3f950d956df30c25578ff729d6738b2eac8dde.
It is never called pre-extraction evidence. The old baseline, full old results,
old fixtures and original criteria are retained. Tests separately verify one
complete update at each frozen old-backend state and strict whole modern-source
trajectory parity. Notebook plots distinguish original-old, qualified-modern
and fresh-local fields and all histories rather than dropping a difficult
comparison. Cross-backend field quality still uses unchanged prelocal criteria.

## Equation, source, local function and validation mapping

| Required stage | Original source | Local executable function | Independent evidence |
|---|---|---|---|
| Cell Gram and reciprocal metric | GSASIIlattice cell2A,calc_rDsq | reciprocal_metric_coefficients,dspacings,cell_basis | Native full metric/spacings plus independent Cartesian reciprocal basis |
| P21 equivalent phases, duplicates, absences | Native sources/GENHKL and GSASIIspc.GenHKLf | equivalent_reflections |125 actual compiled source hkl cases; analytic odd0k0 screw absence |
| Seeded orbit/Friedel initialization | GSASIImath.ChargeFlip reflection initialization | initialize_structure_factors | Full independently captured initial complex grid, all4508 signed rows |
|2013 section4.6,AppendixA threshold maps | ChargeFlip FFT/rho/sigma/double where blocks | charge_flip_update,flip_density | Source r,sigma,flipped,temporary,phase at1/2/7/25/100; independent sign-flip norm |
| Fixed observed modulus replacement and Rcf | ChargeFlip reciprocal projection/residual | charge_flip_update | All source phases/residual history; negative-intensity noise-floor analytic case |
| Source GUI/intrinsic stops | ChargeFlip callback and Rcf<5 | reconstruct | Whole independent original200 run and24-step comparison; fixed budgets labeled |
| Heuristic symmetry-phase origin | GSASIImath.findOffset | source_origin_shift | Original integer roll,score,terms and final field; known global-shift scientific checks |
| Physical positive-exponent10/N acquisition | Source final FFT/10 convention | structure_factors | Independent known displaced Fourier point source and translation/inversion invariance |
| Full pipeline | Original selected ChargeFlip exact observations | reconstruct,main | Same-backend whole source/local parity, separate cross-backend quality and standalone Judge |
| Full evaluation | Independently defined no solver imports | evaluation/judge.py | Analytic shift/inversion/no-gain cases, malformed input/asset/CLI/adversarial contracts |

Source geometry, FFT normalization/sign, factor ten, source threshold ordering,
zero guard, masks used only inside the source diagnostic, seed/orbit write order,
half-grid padding and origin heuristic were semantically reviewed against source.
Both plans derive from the shipped functions, not copied opaque upstream docs.

## Judge independence and meaningful contracts

Judge loads real float32/64 shape(1,72,24,96), exact key reconstruction and finite
values. Missing/malformed/key/shape/dtype/object/NaN/Inf are errors with empty
metrics. Truth, if supplied, takes precedence; invalid truth never falls back.
No solver code is imported or executed. Assets resolve relative to Judge,
caller CLI paths remain caller-relative. JSON is finite UTF-8, nested parents
are created, report-write failure produces stderr and native2.

Full-map uncentered NCC/rangeNRMSE allow only periodic translation and whole
spatial inversion. A newly added independent sparse-spectrum analytic test
found single-start coarse-correlation refinement can settle at a wrong peak.
The fix uses the eight highest periodic local maxima plus zero starts, the
same analytic gradient and fixed optimizer settings, without changing any
quality threshold. This finite deterministic search is documented; no claim
of a general global-optimizer certificate is made. A default pytest temporary
directory initially failed with Windows permission denied; rerunning with a
private explicit basetemp fixes environment handling, not solver math. All13
new Judge cases then pass. Zero prediction is a valid fail with intensity
relative L2 exactly1. Tests cover no gain fit and true inversion/translation.

## Active notebook and scientific comparisons

The canonical notebook explicitly follows problem,data,physics,method,results,
conclusion. It loads public measured observations, executes full200-step primary,
saves fresh reconstruction and calls the independent standalone Judge. It then
actually recomputes full200-step seeds20261023/24, a20-step budget and upper
thresholds20/50. No cached source array is substituted for a fresh comparison.
The original GUI20%-stop is labeled as24 actual updates. Full old and qualified
modern source maps/history are identified independently.

Eight figure groups show all measured reciprocal locations/signed data/noise;
full-cell Cartesian density display and complete sections; distinct old/modern/
fresh source parity and complete error histogram; every actual trajectory;
fresh phase-seed controls; fresh budget/threshold controls; all signed physical
residuals and a freshly constructed poor random-phase map; actual primary
snapshots1/2/7/25/100/200. The3D top0.5% selection is explicitly display-only,
with full arrays scored/stored. Negative values, residual outliers and full
histories are retained. These are conditional method comparisons, not errors
against experimental truth.

Paired PNG/SVG/PDF and full-precision numeric sources are exported privately.
fresh_results.npz is consumed by maintainer/report review only, with runtime,
environment,data hash,method roles and metrics in fresh_run.json. It is never
loaded by the solver or by the active primary reconstruction. Original
source_comparisons.npz supplies the labeled source GUI/seed figures. Every
fixture/reference has a documented numerical/plot consumer.

The first notebook launch failed before the first cell because no globally
registered research-runtime kernel existed. A private kernel descriptor now
points to the exact shared research Python; JUPYTER_PATH is scoped only to the
runner. Actual commands, native results, logs, input/canonical hashes and1800s
timeout rationale are in notebook_process.json. Failed and successful runs
have distinct private log directories. Completion is not inferred from imports,
cached notebook outputs or a plot file existing.

## Final gates

Final native suite, CLI, Judge, notebook, visual review, math-rendering,
public sufficiency, forbidden-call roles, file inventory and archive hashes
are recorded in final_evidence.json and the final observed results appended
below. A mandatory failed or unverified gate blocks acceptance. No incomplete
gate is claimed passed by this provisional validation narrative.


## Final observed native results

Complete native parent suite: **43 passed**, exit0, process16.03442320000613s. This includes original stages, same-backend complete pipeline, independent physics, public constants, backend import and strict Judge contracts. Full standard CLI native0/4.652564700008952s; standalone Judge from foreign cwd native0/0.6875083000049926s.

Final full active native nbconvert: **exit0**, 64.4576143999875s, canonical SHA256`095c55cfc34354bdc493b50d7b48a8a6b2b176be037409488ff07a5c94890b4c`. It computes all six full-volume inverses and eight figure groups.32 formulas render without errors in both MathJax and KaTeX. All eight actual figure groups were inspected; 3D axis/title and source24-update labels were corrected followed by another complete successful execution.

Fresh standalone metrics: {"ncc": 0.9989629082253738, "nrmse": 0.001758786221731934, "intensity_relative_l2": 0.00010239649895416404, "mean_relative": 5.5514425609748885e-18}. Same-backend qualified author/full local output and every captured state are bitwise identical; original NumPy1.26 baseline remains separately identified. All43 tests pass without modifying pre-extraction quality criteria.

Gate status: **passed** for paper/version/data/equation mapping; exact-bound original and cleaned native executions; independent frozen source stages and physics; constants/units/precision; standard CLI and independent Judge; adversarial contracts; public self-containment/no target or GitHub leakage; minimal shared runtime and local core math; complete tests; active six-story native notebook and actual scientific comparisons; distinct source/backend/fresh provenance; semantic plans and source call roles; canonical file inventory, private archive and legal notices. The exact source qualifications remain explicit: FCF reader I/O adapter, private modern reshape API repair, backend-sensitive finite nonlinear trajectory, relative scale, method reference not physical truth and external-data rights unstated.

Private final_evidence.json, plot_provenance.json and archive_index.json identify process evidence, all final scientific assets, their numerical consumers and hashes. No temporary executed notebook or paper/source archive is shipped. Generated caches were moved reversibly to private archive.


## Public-contract scientific isolation 2026-10-29

Measured amplitudes alone admit distinct phase structures; a validated structural prior/equivalence criterion is unresolved. Benchmark eligible=false. Scientific review is completed; blind-agent execution is a separate unexecuted gate and is not the reason for isolation. Removed README recipe, where present, is archived privately under benchmark_readiness_20261029/private_readme_recipes. Existing arrays/Judge thresholds unchanged.
