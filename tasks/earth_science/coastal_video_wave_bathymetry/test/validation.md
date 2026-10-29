# UBathy extraction and validation record

Status: READY FOR PARENT ACCEPTANCE. Native qualification, full fresh CLI/Judge,
complete active notebook, independent parent50-test pipeline, all9 actual figures,
presentation replay and inventory checks passed. Root owns final acceptance
freezing; this worker does not self-freeze acceptance. This file is private.

## Fixed task and provenance

The unknown is positive-up seabed elevation at all 8963 native reconstruction
sites, with an explicit observation-supported validity mask. Inputs are complete
native green-channel samples from both selected recordings, their native mean
time axes, and independently calibrated camera/mesh coordinates. No mode phase,
wave number, reconstructed bed or target-derived prior is public input.

Upstream: https://github.com/Ulises-ICM-UPC/UBathy , verified revision
`9f4fbdcc7e0c5aec47a2f6bfaf3f567c8c00f251`, AGPL-3.0. Local explicit
mathematics preserves attribution; LICENSE contains upstream terms.
Source archive is under the maintainer batch `astro_earth_physics/archives/`.
`original_source_integrity.json` indexes exact source/PDF/archive hashes and
checks that both executed source copies are byte-identical to that archive.
Source ZIP SHA256 is
`4f57c40b29d1a22d85a9179eb0da033fbeee60f1235f7b70cc02c4dcaec66512`.

Paper: Simarro and Calvete, Remote Sensing 14(23), 6139 (2022),
https://doi.org/10.3390/rs14236139 . The actual archived PDF is the institutional
author manuscript, 19 pages, running footer **Version January 2, 2023 submitted
to Remote Sens.**, with first-page volume placeholders. It is not represented as
the publisher's final PDF. Retrieved from
https://upcommons.upc.edu/server/api/core/bitstreams/274cb5a4-fb23-410b-b67a-4d33da8257dc/content .
PDF SHA256 is
`f3ad5aa0e67a359ad7f7dc8caf90605542394ecf7ba9857455432ca42989f66f`.

Author model/data deposit: https://doi.org/10.5281/zenodo.7360216 , CC BY 4.0.
Existing real observations: Castelldefels Beach video, 1 August 2020. Independent
survey: Barcelona Port Authority, 31 July 2020. This one-day difference can
contain real bed evolution; the survey is an independently measured comparator,
not exact synchronous physical truth. Tide levels originate from the independent
Barcelona harbour tide-gauge record. No EPSG identifier was independently
verified; only the supplied common metric Cartesian frame is asserted.

Raw NPZ SHA256:
`decb32ba727501349f25cf63eacd64481beb649aee717fcae1e7c6306c0b62b9`.
The 3,777,837-byte public file contains exactly the nine README keys. Native
digital counts are uint8, converted to floating values by division by 255.
Oblique uses 5075 mode sites, 2479 wave sites and 301 frames; planview uses
26382 mode sites, 7555 wave sites and 300 frames. Both native time grids span
0--160 s, with distinct time steps 160/300 and 160/299 s. No interpolation to
nominal 3 Hz is performed. Native zero-count planview border rows are retained.
Known water levels are 0.177 and 0.183 m; gravity is 9.80665 m/s^2.

## Private archive and original execution

All paths below are relative to maintainer `astro_earth_physics/oracles/ubathy/`;
none is a cleaned-runtime import or solver-visible asset.

* `original_work/`: unmodified original source, observations and all products.
* `capture_observations.py`, `observation_capture.json`, `public_inputs.npz`:
  original-helper pixel sampling and independently calibrated meshes.
* `preflight_failed_missing_ffmpeg*`, `preflight_failed_empty_cached_directory*`:
  actual initial failures, retained. The source's overwrite=false treated a
  failed empty frame directory as cached; that empty directory was archived,
  then fresh extraction was run. No failure is counted as success.
* `bin/`: private original-only FFmpeg binary from its wheel. It is never a
  cleaned dependency. `preflight.log`, `preflight_execution.json` record the
  successful full frame/mesh preparation (6.84 s).
* `full_native.log`, `full_native_execution.json`, `full_native_resume.log`,
  `full_native_resume_execution.json`, `resume_input_manifest.json`:
  original full DMD/K/B/Kalman run. The first bounded run stopped after 13
  self-generated K fields; overwrite=false resumed its own products, not author
  cached answers. Seed was reset on continuation, explicitly preventing a claim
  of one uninterrupted random stream. All 20 fields and full bed were completed.
  DMD took 24.4047 s; resumed K 1172.2067 s; B 68.1368 s; Kalman 2.6886 s.
  Author `output_example` files were not substituted for fresh results.
* `original_reference.npz`: identified original fresh bed, spatial SD, validity,
  independently surveyed bed and mesh. Single-day Kalman initialization equals
  the B result; no temporal-history improvement is claimed.
* `stochastic_repeat_work/`, `stochastic_repeat*.json`, `stochastic_repeat.log`:
  independent unchanged original K+B repeat, seed 20261027, on frozen original
  DMD modes. K 826.9925 s, B 27.5721 s. This is not a second full-movie run.
* `native_sequential_qualification.py`, `native_sequential*.json/log`:
  uninterrupted unchanged original K+B, original fixed seed 20261026, same full
  20-mode order. This new run is required to qualify the faithful sequential
  sampler; initial interrupted execution is not mislabeled as stream-equivalent.
  Its first command completed all20 continuous K fields, then exited1 before
  B computation because the private work copy omitted calibrated mesh_B.txt.
  `native_sequential_b_retry.py` copies only the identical original mesh, asserts
  exact equality to public reconstruction_xy, and runs the deterministic original
  B stage on those unchanged K fields. The first failed log/exit is retained;
  `native_sequential_b_retry*` separately records repair, hash and stage exit.
  This is not represented as a successful first end-to-end command.

Original environment is the isolated maintainer `oracles/mttime_env`, Python
3.11.11, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.2, OpenCV original-only.
Cleaned runtime is the shared environment: Python 3.11.11, NumPy 2.4.6,
SciPy 1.17.1, Matplotlib 3.11.1. All runs use CPU with OPENBLAS/OMP/MKL threads
set to one. Guard drivers record memory every 0.5 s and stop below 6 GiB free
physical or committed memory. Numerical backends and file hashes are indexed in
the private archive inventory. No upstream domain library is installed in the
cleaned runtime. OpenCV and FFmpeg only prepare/capture existing original data.
`original_environment.json` and `cleaned_environment.json` record the actual
Windows/Python/package/backend configurations. Original OpenCV is5.0.0; private
FFmpeg reports7.1 essentials build (GCC14.2.0), not a cleaned dependency.
Original plotting emits retained equal-aspect/axis-limit warnings. No warning is
silently treated as numerical success; actual stage exits are logged separately.
`archive_index.json` indexes private scripts, logs, fixtures and outputs with
SHA256 hashes and enumerates the preserved work directories. Its generator is
`build_archive_index.py`; these files remain outside the task package.

## Independent fixtures and equation mapping

| Physical/algorithm stage | Original file/function | Local function | Independent case |
|---|---|---|---|
| Analytic signal and time contexts | ulises_ubathy / X2DMD_2506 preprocessing | physics_model.analytic_signal; preprocessing.window_specs | upstream_stages, all 8 windows |
| Snapshot DMD, log-eigenvalue periods | X2DMD_2506 | physics_model.dmd; solvers.extract_modes | all 20 modes, period/phase gauge/amplitude |
| Explicit dispersion approximation | Kappa2MuOneStep; WGH2KOneStep | kappa_to_mu; dispersion_wavenumber | frozen source values plus independent Brent root, shallow/deep limits |
| Adaptive phase-plane RANSAC | RANSACPlane; NForRANSAC | fit_phase_plane; ransac_trial_count | 4 real neighborhoods x64 independent draws; actual-draw replay32 cases |
| Multiscale local wave fitting | ubathy.ObtainK | estimate_wave_fields | complete fullraw path plus phase/noise diagnostics |
| Native observation-quality aggregation | ubathy.ObtainB | aggregate_wave_observations | known-order/filter unit tests; source B-stage full observations |
| Grid consensus and BFGS | ubathy.ObtainB | fit_bed; fit_bed_point | 8 source point cases and full frozen 94877-observation B stage |
| Spatial SD, NaN support erosion | ubathy.ObtainB | spatial_uncertainty | all8963 source SD values and validity; missing-neighbor negative case |
| Single-date Kalman initialization | PerformKalman | explicit identity initialization implicit in submitted B bed | original single-date B/Kalman agreement |

`test/fixtures/upstream_stages.npz` consumes37 source arrays in source-stage tests.
The original capture invokes unchanged author functions, with exact agreement to
the saved original20 mode files. It is not generated by local src.
`upstream_ransac_statistics.npz` contains four real observation neighborhoods,
64 original coefficients and inlier counts each. Local independent random draws
are compared by frozen five-standard-error statistical bounds, not seed equality.
`upstream_fixed_triplets.npz` captures32 actual original sample sequences and
final coefficients/inliers/draw counts. Recording the sampler return values does
not alter original arithmetic. Replay isolates local mathematics/stopping from
random stream changes. Source coefficients and inliers agree exactly.
Measured local/source discrepancies across all20 selected modes are at most
6.6141e-15 relative period error,9.6362e-14 rad weighted phase-gauge RMS,
and9.3132e-14 relative amplitude error (`local_stage_errors.json`). The explicit
FFT/DMD extraction is numerically equivalent, not bit-identical.
`upstream_bed_stage.npz` contains50 arrays: eight original BFGS cases, six full
wave-observation vectors, full rawbed/radii and final bed/SD. Full deterministic
local B stage has identical validity and zero maximum bed error against source.

All capture scripts and source environments remain private. Tests import only
cleaned src and frozen numerical NPZ fixtures, never archived source.

## Physics discrepancies investigated

The author approximation's quoted 0.047% accuracy describes mu/root relative
error, not the nonlinear dispersion-equation residual. Independent checks find
mu relative error 0.0004688177644 and equation residual 0.000923291281. Since
the positive dispersion elasticity is at most two, the corresponding equation
bound is 2e+e^2=0.00100025 for e=0.0005. The mislabeled first draft is retained
as `acceptance_first_physics_draft.json`; source formula is unchanged.

The three native trial depths are 6, 4.1666667 and 2.3333333 m; the lower
0.5 m search bound is not one of these initial wave-radius depths. BFGS is
unconstrained and can return depths beyond6 m. No clipping is introduced.
Native spatial SD uses np.std, not nanstd: any unsupported neighboring raw bed
causes support erosion. SD is local spatial variability, not posterior uncertainty.
BFGS may return a finite estimate with precision-loss status; the author returned
x is preserved, with actual optimizer-success diagnostics shown explicitly.

## Frozen evaluation and stochastic support

Actual survey has all8963 locations and is stored in ground_truth.npz.
baseline_reference.npz contains the original method's reconstruction, validity
and observation-only interior_validity. It is not physical truth. A missing
survey file is an evaluator error; no method output is relabeled as survey.

Original5615 valid sites versus independent survey: cosine NCC
0.9926581740375782, dynamic-range NRMSE0.06947820996167868, RMSE0.4272770956 m.
Unchanged all-valid field bounds are 0.9 times that NCC and 1.1 times that NRMSE.
An independent original repeat has5572 valid sites, required-site recall
0.9852181656, Jaccard0.9782493369, and shared-support bed RMS difference0.179690 m.
Its own-valid survey NCC0.9927281501 and NRMSE0.0706276102 pass those bounds.

An initial exact-all5615 support contract was falsified by that original repeat:
83 source-valid boundary sites disappeared and40 extras appeared. All83 lost
sites lie within17.321 m of original unsupported sites; these are observation
support changes, not points selected by survey error. The original-only analysis
is `coverage_stability_analysis.json` and `coverage_stability_locations.npz`.
Treating missing sentinel0 as physical bed also falsely failed the native repeat.

Before any cleaned full-field result was computed/judged, root froze a20 m
observation-only boundary band. Interior4410 sites are original-valid and lie
strictly farther than20 m from every original-invalid site. GT values/errors do
not participate in this membership. Both original executions cover all4410.
Interior metrics use their own fixed reference dynamic range: original NCC
0.9950154629160847, NRMSE0.08072550183769539; repeat NCC0.9951610814578645,
NRMSE0.07919450463895529. Distinct interior boundaries are their original values
times0.9/1.1. Required5615 recall>=0.98 and interior coverage=1 are mandatory.
All submitted-valid sites, including extras, are independently survey-scored.
Unsupported sites must have exact finite0 sentinel, explicitly not physical bed.
Root freeze: `acceptance/coastal_video_wave_bathymetry/coverage_qualification_frozen.json`,
UTC2026-10-26T15:31:51Z. Full hashes/time and selection rationale are in
`original_only_interior_metrics.json`. No field threshold or mask is relaxed
after cleaned results.

## Fresh local results and unresolved gates

Initial batched-sampler local CLI completed113.58 s, exit0. All-valid survey
NCC0.9933248338451487, NRMSE0.06773502986743632 and required recall0.9814781834
passed, but one of4410 interior points was unsupported: Judge correctly failed.
This run is not a completed benchmark. `first_batched_cli*`, `first_batched_reconstruction.npz`, `first_batched_judge_report.json` and `cleaned_diagnostics*`
preserve logs and actual arrays privately. Site6909 has an individually valid bed
estimate; neighboring6908 had only8 quality observations below native minimum10,
causing native SD NaN erosion. No point was filled, removed or reclassified.

Although batched draws preserved the sampling law and32 actual-triplet arithmetic
cases agreed exactly, prefetching64 proposals unnecessarily altered the native
random stream. It was removed after root review. The shipped sampler now uses
sequential RandomState.choice with the original fixed seed20261026, native draw
consumption, tie handling, singular skipping and stopping arithmetic. No seed
search is performed. The new uninterrupted native and fixed-seed local
qualification executions run concurrently with independent single-thread and
memory guards, as authorized by root. The evaluation/mask freeze predates both.
On local completion its output hash is frozen without inspecting GT/Judge;
scoring waits for completion of the independent uninterrupted native qualification.
No claim is made that native completion predates the local computation. Old
failed evidence remains an audit, not an active cache.

Component tests at initial extraction:12 source/physics/bed comparisons passed
16.58 s. Combined prior-sampler suite49 passed18.86 s; revised sequential source,
physics, distribution and actual-draw tests9 passed4.60 s. One initial test used
an inaccessible Windows C: temporary directory; rerun used a private D: TEMP.
This environment error is distinct from a numerical test failure.

The initial gates below were completed as separately recorded; presentation
updates and final inventory are tracked in the final checklist.

The faithful sequential CLI subsequently completed811.5311657 s, native exit0,
with5577 valid sites. Output SHA256 was frozen before evaluation as
`b917a86a624af4228f2598f789eb072a35fd671256b788e5c48ba1a0574c5a50` in
`sequential_local_output_frozen.json`. The uninterrupted source's original
K stage completed790.52 s; mesh-only B repair completed27.53 s with exit0.
Source qualification passed all unchanged metrics:5580 valid sites, required
coverage0.98753339269813 and interior coverage1. Its interior NCC/NRMSE are
0.9950781980276602/0.08024297063388498 and all-valid metrics
0.9930317097740343/0.06769220391401265.

Fresh sequential local Judge exited0: required coverage0.9877114870881567,
interior coverage1, interior NCC0.995112649237658 and
NRMSE0.07994553735864468, all-valid NCC0.9930284204194217 and
NRMSE0.067696538411297. No threshold, mask or seed was changed after inspection.
The long pipeline contains tiny measured FFT/DMD differences and stochastic
consensus decisions; roundoff-induced branch changes are a plausible explanation,
not an independently traced proof of every full-field discrepancy. Source and
local support are not asserted bit-identical. They share5573 sites, with7 source-only and4 local-only sites;
shared bed RMS difference0.0467477321 m (maximum3.05356 m). These discrepancies
are explicitly plotted; neither maximum nor unsupported sites are suppressed.
This source/local RMS is smaller than the independently measured original
repeat RMS0.17969 m. Both methods meet the independent survey/support contract.

Final sequential49-test suite passed18.99 s, exit0, logged under
`final_pytest.stdout/.stderr` and `final_pytest_execution.json`. A separate
full-pipeline comparing test was then added: it reconstructs every bound raw
observation afresh, writes a temporary NPZ and invokes the standalone Judge from
another working directory. Parent runs the resulting50-test suite concurrently
with the helper's one complete active notebook. This worker does not duplicate
that expensive additional path. The completed parent result is recorded below.

The one complete active notebook finished with nbconvert returncode0 in
865.0637814 s, UTC2026-10-26T16:01:19.508326Z--16:15:44.572821Z. Its complete
fresh reconstruction took810.1888201 s, with20 modes and95156 qualified wave
observations,5577 final valid sites and identical fresh Judge metrics to the
CLI. Canonical notebook contains22 cells,11 executed code cells,9 computed PNG
figures, zero error outputs and no unexecuted code cells. No source/evaluation/
raw asset hash changed during execution; all cell sources match the executed
input. The private evidence directory is maintainer batch
`bio_med_material/ubathy_review/`: `before_execution.json`, `execution.json`,
`notebook_execution.log`, `executed_archive.ipynb`, `notebook_measurements.json`,
`figures.json` and `figures/`. The temporary executed notebook and measurement
JSON were moved out of task/output and preserved privately. Figure QA and parent
50-test completion remain distinct final checks, not inferred from exit0.

The notebook includes all native observations and spectra, per-mode phase and
amplitude source comparisons, local wave fields, independent survey/native/fresh
bed and full coverage, same-support errors, actual gamma residuals/spatial SD,
source-only repeat differences and earlier failed-prefetch coverage. Controlled
phase/noise experiments use real captured neighborhoods. Fresh downstream bed
refits isolate60 s/90 s window fields, oblique-only/planview-only fields, and a
uniform+0.10 m known-water-level perturbation; these are explicitly downstream
refits on newly computed wave fields, not additional whole-movie reruns.

## Reference-output consumers

notebooks/reference_outputs/upstream.npz contains only six consumed arrays:
bed/standard_deviation/validity from the original full run and corresponding
repeat_bed/repeat_standard_deviation/repeat_validity from independent source
K+B repeat. The active notebook computes fresh results from raw observations;
these arrays are only labeled source comparisons. Survey comes from the actual
hidden survey asset and geometry from public input, without duplicated NPZ copies.
Source-only repeat differences describe stochastic stability, not posterior
uncertainty. All original full arrays remain archived privately.

`notebooks/reference_outputs/failed_prefetch_local.npz` is the byte-identical
copy of `first_batched_reconstruction.npz` with only reconstruction/validity.
Its explicit notebook consumer is the earlier FAILED local-prefetch coverage
comparison. It is neither upstream truth nor fresh current-method recovery.

`notebooks/reference_outputs/sequential_upstream.npz` contains bed,
standard_deviation and validity from the uninterrupted native20-field K run and
its mesh-repaired deterministic B stage. It supplies the primary same-config
source comparison in the notebook. The earlier original/repeat references retain
their labels and their separate coverage-qualification role.

## Independent parent checks and final checklist

Parent ran the complete50-test suite, including an additional actual fullraw
reconstruction and standalone Judge, in830.11 s (process830.575 s), exit0.
Minimum available physical memory was12.673 GiB. Evidence is maintainer batch
`acceptance/coastal_video_wave_bathymetry/`: `execution.json`, `pytest.stdout`,
`pytest.stderr` and `pytest_memory.json`. After the public metadata test gained
README-parsing assertions, parent separately reran its four unit tests:4 passed
0.71 s, evidence `metadata_contract_recheck.json`. These are repeat verification
of four existing tests, not54 distinct tests or a second full-pipeline claim.

The independent Judge contract suite passed32 tests, including malformed inputs,
truth absence as evaluator error, bad extra sites, missing boundary support,
any missing interior point, exact finite sentinel and independent direct-distance
verification of the4410 mask. Root's separate source/paper/GT review and all9
actual figures were inspected. MathJax and KaTeX checks both have zero failures
for20 formulas in root `document_math_render_check.json`; helper additionally
checked its4 notebook formulas. No check substitutes for the active inverse run.

Caches were moved, never deleted, to private `handoff_cache_archive/`. Task has
34 required/consumed files, including LICENSE/NOTICE and the three distinct
notebook comparison NPZs. No temporary executed notebook, measurement JSON,
source checkout, PDF, capture script or original log remains inside the task.
The static import/call-role audit records only NumPy primitives, generic SciPy
optimization/spatial lookup, Matplotlib and standard library. Local FFT/DMD,
dispersion, consensus/RANSAC, objectives and complete inverse are executable
mathematics. No upstream imports, cross-task imports or ready-made domain inverse
are used. Generic BFGS is the actual native optimizer on the local objective;
its defaults and termination/line search are explicit in approach.md.

- [x] Paper version, exact revision, raw hashes, archive index and stage mapping.
- [x] Actual original and cleaned runs on the bound complete observations.
- [x] Frozen upstream stage fixtures, actual-draw replay, independent statistics
  and mathematical/physical limits, including full deterministic bed stage.
- [x] Matching units, constants, coordinates and README acquisition metadata.
- [x] Standard CLI and standalone Judge pass; all required quantities evaluated.
- [x] Malformed submissions, poor estimates, coverage and evaluator errors tested.
- [x] Public input contains observations/calibration only; no GitHub/truth leak.
- [x] Minimal shared runtime, explicit local mathematics, no domain solver calls.
- [x] Full50-test parent suite including fresh end-to-end inverse passed.
- [x] Entire active notebook completed,9 actual computed figures retained.
- [x] Survey/source/fresh/failed historical roles and metric definitions separate.
- [x] README/plans/signatures/objectives/units/flow semantically reviewed.
- [x] Final presentation-only replay evidence and final34-file inventory handoff.

Root performs final acceptance and freezes its own accepted inventory. This
worker does not self-declare independent parent acceptance.

### Final presentation and computed experiments

Helper and root inspected all9 actual computed figures. Root approved a
presentation-only coordinate-axis locator revision in notebook cell11 and six
Markdown story-heading revisions. The map cell was separately executed through
private nbconvert, returncode0 in3.933 s, using the actual completed fresh
reconstruction; other code cells and their computed outputs were unchanged.
No source/evaluation/raw asset changed. This display replay does not replace or
claim to repeat the865.063781 s fullraw reconstruction. Canonical notebook keeps
the full executed recovery and the updated actually executed map output.
`bio_med_material/ubathy_review/display_revision.json` and
`display_cell11_executed.ipynb` document the exact difference and replay;
`executed_archive.ipynb` preserves the complete original full execution.
`figures.json` indexes the current9 PNGs. Thus the earlier all-cell identity
statement describes that original full execution, with this later isolated
presentation change explicitly distinguished.

Actual downstream ablations in `fresh_experiments.json` (own valid footprint,
not the fixed interior) are:60 s4917 valid/NCC0.991963/NRMSE0.0750594;
90 s5266/0.993127/0.0675728; oblique-only1991/0.992295/0.0681851;
planview-only4040/0.993144/0.0888932. Different coverage is explicitly shown and
these rows are not silently compared on different supports as one fixed-domain
accuracy claim. Full-precision measurements and paired coverage are archived.
The uniform+0.10 m known-water-level experiment has5577 common sites and
1.87363e-8 m RMS departure from the expected+0.10 m bed shift. This is a physical
calibration invariance experiment, not a new target-derived offset. The active
full case has6174 raw bed fits, all with actual BFGS success flags, followed by
native NaN/SD support erosion to5577 final sites. Success is measured for this
run and is not assumed by the algorithm or generalized to all noise cases.

`final_call_role_audit.json` records all24 source/test/notebook import units,
all9 public finite raw arrays and zero forbidden public GitHub-domain matches.
`semantic_plan_review.json` records actual signatures and physics/objective/
initialization/flow review; its final inventory hashes identify the delivered
version. `worker_inventory.json` lists the final34 task files for root review,
not an accepted/frozen parent inventory.


## Independent parent acceptance

Completed worker CLI811.53s passed; parent independently reran the complete raw-observation reconstruction inside the full pytest suite and separately invoked standalone Judge, all exit0. Final documentation passed both math renderers. Notebook execution origin: Complete worker nbconvert execution on the exact frozen source/raw/evaluation assets; parent independently verified input snapshot, all source hashes and mtimes, all code-cell sources/execution counts and actual outputs. No duplicate full parent notebook invocation is claimed. Reviewed artifact: 11 executed code cells, 9 actual PNG outputs, 20 formulas. Private evidence: research_batch9_20261026/acceptance/coastal_video_wave_bathymetry/: execution.json, notebook_input_snapshot.json, notebook_source_and_output_audit.json, semantic_review.json; original worker logs/archive in bio_med_material/ubathy_review/; plot_layout_amendment.json. All final panels were inspected by parent. Canonical notebook preserves the verified865.06s full numerical execution; only six Markdown headings and one locator-only plot cell were amended. The plot was actually reexecuted3.93s from the same fresh submission with unchanged source/evaluation/raw assets, then independently inspected. Other10codecell objects are identical to original execution. No duplicate complete parent nbconvert run is claimed.


### Public source-link separation, 2026-10-29
Removed the direct public URL https://doi.org/10.5281/zenodo.7360216 from the solver README. Role: maintainer provenance identifies model/data deposit; direct mixed archive link removed. The scientific citation, dataset attribution and license remain; complete source/data provenance stays in this maintainer-only validation record. The released NPZ supplies every observation/calibration input, so no software archive is required by the solver contract.
