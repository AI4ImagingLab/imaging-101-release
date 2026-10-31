# Validation and provenance

## Binding and permissions

The selected task is complete 432-fragment position plus C4 orientation recovery.
This is a new unknown/acquisition pair: unordered spatial fragments with unknown
permutation and orientation, rather than a pixelwise demosaicking, registration,
orthographic structure or spectral inverse. The author's real photograph is used
with its existing synthetic cut/scramble acquisition. It is explicitly not called
a measured physical broken-object dataset. All old299 tasks are preserved.

Source repository: https://github.com/vahanhuroyan/PuzzleDemoGCL . Exact remotely
verified commit: c957fe5a70a0e153e6ba93440dd13144f9945818 . Metadata and downloaded
file hashes are in W2/fallback_scout/puzzle_source_metadata.json and
puzzle_download_manifest.json, with lexical/GT-role audit in
gcl_selected_gt_role_audit.json and gcl_truth_lexical_inventory.json.
Paper: Vahan Huroyan, Gilad Lerman, Hau-Tieng Wu, Solving Jigsaw Puzzles by the
Graph Connection Laplacian, arXiv:1811.03188v5; SIAM Journal on Imaging Sciences
13(4),1717–1753 (2020). PDF is privately archived at
W2/fallback_scout/gcl_paper.pdf, SHA256 ae9a8eb2caff0c7eb4578c7be8ab6590501b97b975f5c4b1724ce586a2709456.

No repository-wide LICENSE/SPDX grant was found. Photo redistribution permission
is unverified. This is the explicitly authorized local-research artifact, with
source credit retained in NOTICE; no external publication/push/relicense is
authorized or implied. It is not labeled MIT merely because another project is.

Here W1/W2/W3 mean B/worker1_math_physics_astro,
B/worker2_mechanics_earth_astro and B/worker3_material_bio_med, respectively;
B is D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch13_20261030.
Cstate is C:/Users/admin/AppData/Local/imaging101_batch13_oracles/worker1_gcl_state_v1.
These private paths are provenance, never public runtime inputs.

## Data and actual truth

The public NPZ contains only uint16 tiles (1,432,28,28,3). SHA256 04609aea0476b71683ab7e51a744b22a9448f4f81a61e8e5aaa3c60cdfc8c84c.
All pixels exactly equal the bound original MATLAB ap cells; no float conversion,
GT, permutation seed, original photo URL or neighboring-boundary clue is public.
The frozen source observation MAT SHA is
d8fc7260801f6e62913da4daa7a67be15906a1558dbf68a3e87307622986a634.
Pixel stream SHA is ef79c0b1b3c51bff111c6221ee396f0a357647311b5288b83f74eb64fc89ec06.
Root independently reloaded only ap and checked exact identity in
acceptance/gcl_public_candidate_identity.json. Observation-only exact-C4 hashing
with array-equality confirmation found no duplicate or rotationally symmetric
tiles: acceptance/gcl_public_tile_symmetry_audit.json.

Actual acquisition truth is stored as positions and quarter_turns, batch-first
int64, in evaluation/ground_truth.npz, SHA256 562709c2857e361d7aced67ef24099359c5c2b2d8c3b9ceb8438e680d83dec27. It comes from the known
forward shuffle ppp and applied turns randscram, not a fitted reconstruction.
Original image IDs are column-major: row=(ID-1)%18, column=(ID-1)//18. The
correcting turn is (1-randscram)%4. Private export/source hashes and mapping are
W1/gcl_acquisition_truth_provenance.json. The public generator explicitly applies
negative correcting turns to extracted blocks. Exact real-bound pixel roundtrip
and an independent floating adjoint identity are tested.

## Original execution and compatibility versions

The selected experiment retains author ScramblePositions=1 and rotations=1,
all432 pieces, both global alternatives at initialization and every refinement,
five outer refinements, and the original inner maximum-five update policy.
Input adaptation only loads the exact already-bound ap. It does not reshuffle,
subsample, read old inverse states as initialization, or use GT to select a fit.
The original ppp/randscram arguments are restricted to score-report diagnostics;
show=0 and do_replicate=0 make global debug/replica branches inactive.

Successful execution is the original selected method with documented numerical
compatibility adaptations, not a byte-unchanged program. Full source copies and
diffs are W1/gcl_source_frozen_v1, gcl_source_author_rui_v1 and
gcl_source_stagewise_v1; adaptation ledgers/diffs are gcl_source_v1_preparation,
gcl_author_rui_backend_preparation, gcl_stagewise_source_preparation and matching
.diff files. No domain code is installed in the cleaned runtime.

Original scalar MGC completed on exact frozen attributes in an actual native
8.4 traced run (139.875s, exit0), yielding independent full float32 fixtures.
The same-commit author ComparePiecePairA_ROT_rui backend computes fresh scores;
all2,979,072 finite entries are bitwise equal and all6,912 positive-infinite
self-pair locations identical. The predeclared <=2ULP plus 2eps32 bound needed
no slack. Independent NumPy controls reproduce the full array. See
W1/gcl_rui_original_full_fixture_parity.json and
W3/gcl_generic_controls/rui_mgc_full_independent_audit.json.

Native8.4 and11.3 produced several original pure-M segfault, false-OOM and
index failures despite small/finite inputs and memory headroom. Root cause is
unresolved: no hardware or universal environment-fix claim is made. Original
logs, passive-instrumentation controls and numerical-DLL paths remain private.
The first wrapper's 0.03s no-op is explicitly invalid, documented in
W1/puzzle_first_scout_wrapper_noop_status.json; it is never counted as a run.

Validated generic compatibility covers legacy graph components (canonical
minimum-member labels, not a claimed exact MATLAB tie labeling), smallest-left
singular-mode SVD, original seven-argument LP and values-only integer set
intersection in three selected four-loop calls. Analytic/SciPy/gauge/LP
feasibility controls are W3/gcl_generic_controls, including strict same-input
octave113_revalidation. Generic intersection's83 original/NumPy cases and
malformed-domain checks are W2/intersection_compat/validation.json. No global
builtin shadow or changed GCL equation is used.

Because native errors recurred across long interpreter lifetimes, exact-state
cold-process continuation runs the original prefix once and each full outer body
in a fresh process. Cstate/checkpoint_stage0.mat through stage5.mat contain every
scientific source workspace variable, plus rand/randn states. Independent
recursive class/shape/real/complex/sparse/typed-bit roundtrip checks pass before
continuation. This includes NaN payload and signed-zero microcontrols. Globals
and capture-only counters are audited in
W3/gcl_generic_controls/CHECKPOINT_STATE_AND_GLOBAL_REVIEW.md. No scientific
persistent state is omitted. Fresh source0 recomputes attributes and MGC; cached
scores do not replace the forward computation.

| Completed stage | Actual exit | Wall seconds | W1 receipt/log |
|---|---:|---:|---|
| 0 | 0 | 22.2253256 | `gcl_cold_source_stage0.json/log` |
| 1 | 0 | 13.5174389 | `gcl_cold_source_stage1.json/log` |
| 2 | 0 | 13.4025239 | `gcl_cold_source_stage2.json/log` |
| 3 | 0 | 13.4062374 | `gcl_cold_source_stage3.json/log` |
| 4 | 0 | 13.4065782 | `gcl_cold_source_stage4.json/log` |
| 5 | 0 | 13.5023897 | `gcl_cold_source_stage5_retry1.json/log` |

Stage5 first attempt exited1 after4.3887s in original reorder_SCO; its distinct
receipt/log and input are retained. A single retry from the exact same stage4
checkpoint/RNG completed, without changing source math, selecting by GT or
repeating successful0..4. This is not called an uninterrupted single-process run.
Root independently checked all successful stage exits, bit-roundtrips, complete
432-ID grids and cumulative physical tile turns. Evidence:
acceptance/gcl_complete_source_independent_review.json. All six stages have joint
and oriented-neighbor accuracy1 and rendered dynamic-range NRMSE0.

Runtime: official GNU Octave11.3.0 portable, native Windows, builtin image2.20,
optim1.6.3/statistics1.8.3; CPU numerical threads1; per-child PATH starts its own
mingw64/bin. Exe SHA256
387ac823fb5b01b3407fb34edda6ab1cdba0c9619841a0f6dee323d05a850f2d.
Shared runtime is untouched. Old8.4 remains an independent archived environment.

## Preimplementation criteria and independent evidence

Root froze215 input/source/checkpoint/proof hashes and the full public pose
contract before local code/output, acceptance/gcl_preimplementation_gate.json.
Required joint_piece_accuracy and oriented_neighbor_accuracy are both >=0.95;
position/orientation/ordinary-neighbor rates and full-canvas dynamic-range NRMSE
are diagnostics. One jointly chosen global C4 acts on both position and turns.
No reflection or independent metric alignment is permitted. Rates rather than
zero reference error avoid forcing a numerical path; actual pose truth takes
precedence over source output. Criteria were not loosened after seeing local
results. Source perfect assembly does not by itself prove arbitrary puzzles
identifiable; the explicit bound exact-symmetry audit supports strict IDs here.

W3/gcl_local/fixture_freeze_before_code.json records the pre-code source captures,
tolerances and hashes. Pixel statistics are exact; covariance/graph computed
float64 parity uses rtol2e-12/atol1e-14. Source mode comparison uses an operator,
residual/subspace and common-C4 check, not arbitrary singular-vector phase.
Randomized optional graph bridges are tested by connectivity, symmetry, bridge
count and selected-edge properties, never matching random seeds or samples.
W2/gcl_local_position/fixture_provenance.json independently freezes LP fixtures
before code; sparse matrices are exact, feasibility/objective use2e-6 with
appropriate scaling, and arbitrary nonunique LP primal bases need not match.

## Traceability and consumed fixtures

| Paper/model stage | Upstream function | Local executable | Independent case |
|---|---|---|---|
| Fragment permutation/C4 forward | PuzzleFun | generate_data.acquire/adjoint | real-bound pixel roundtrip; analytic adjoint/norm |
| RGB gradient statistics | ComputePieceAttributes | preprocessing.compute_attributes | source_mgc.npz four complete attribute sets; analytic covariance/SPD |
| Mahalanobis compatibility | ComparePiecePairA_ROT / author _rui | preprocessing.compute_mgc | source_mgc.npz full16 score; all relative-turn physical controls |
| Nearest graph / Jaccard | pairwiseRotPosWeight_nn/_4, initial_nb_construct, use_jaccard_index | physics_model pairwise_nearest/four, initial_neighbors, jaccard_neighbors | source_graph_cases.npz initial operators |
| Four-cycle connection / minimum mode | finding_4_loops, calculate_rotations | four_loops, rotation_step | captured initial/refined graph and A, source mode subspace; analytic C4 |
| Score pose transform | reorder_SCO | reorder_scores | source_graph_cases.npz selected64pair full16 and fresh rotated MGC |
| LP weighted displacement | get_normSCO,get_candidate_and_weight,lp_sp | solvers.normalize_scores,candidate_matches,build_lp,solve_lp | position/source_scores,source_normalization,source_lp NPZ |
| Discrete component geometry | recover_connected_block,do_greedy_assembly,TrimPuzzle,joinPiecesR | recover_components,greedy_assemble,trim_block,join_blocks | position/source_geometry,source_controls NPZ |
| Fill operation | FillPuzzleHoles_V2_rui3 | solvers.fill_holes | source_fill_predelete.npz plus independent single-hole case |
| Mismatch/refinement | find_mismatches,fill_holes | physics_model.find_mismatches,fill_holes,refine_graph | source_graph_cases.npz real complete and deliberate2/6-hole captured cases |
| Both orientations / five refinements | solve_type_2_img | main.reconstruct | 12 original candidate parity; fresh full pipeline test |
| Pose quality | actual acquisition mapping | independent judge; visualization |40Judge cases and14 independent visualization tests |

Every fixture listed above is loaded by comparing tests. No uncited large MAT,
paper PDF, source checkout, original log or temporary screenshot is shipped.
Source fixture keys preserve original names; they are numeric, pickle-free.
The 135MB full original graph capture remains privately archived on C with
SHA-verification index acceptance/closed_worker3_gcl_fixture_archive.json;
only consumed compressed NPZ cases are shipped. Original compressed LP capture
CRC failure is preserved; validated uncompressed MAT-v6 capture generated the
NPZ. One single-hole original-control attempt reaches a native singleton-delete
exception after forming the valid filled b; it is explicitly not a successful
full-fill original run. Both predelete b and an independent analytic fill test
are used. The bound full original runtime itself has no such incomplete case.

## Source discrepancies retained honestly

The legacy candidate cost omits the final positive-row/column terms due to strict
less-than boundary tests; it is preserved and independently tested, not called
the paper's full-neighbor cost. The source90-branch res_all_sol index is a
diagnostic bookkeeping issue; actual real_goal and rotated pixels define source
results. Last rotsNum is an incremental action, not cumulative orientation;
source output rotations were identified independently by exact raw-pixel C4
matching. The source reported score ignores some physical rotations and is not
used as the task's quality metric. The selected LP path contains neither replica
GT selection nor truth-powered repair.

## Fresh local validation and review

Python3.11.11 shared maintainer runtime: NumPy2.4.6, SciPy1.17.1,
Matplotlib3.11.1, CPU1. All acquisition/inverse math is explicit local code;
SciPy SVD, sparse components and general linprog are general numerical
primitives. No domain reconstruction solver, Octave call, cross-task import,
source loading or fixture read appears in src runtime. Source, generator,
Judge, tests and active notebook are subject to the final call-role audit.

Actual first complete CLI: W1/gcl_local_first_full.json/log, exit0,6.0054921s.
Actual independent Judge: gcl_local_first_judge.json/log, exit0, all five rates1.
The complete initialization plus five refinements and all12 placement candidates
execute; each cumulative q is checked against actual rotated tile pixels.
W2's independent12 candidate placement comparison also passes, all12 feasible
and identical geometry moduloC4. W3 full six-stage numerical audit records96
operator comparisons, not96 pytest cases.

Initial worker full pytest: W1/gcl_full_suite.json/log,80passed in11.36s, exit0
(wall11.6492366s). This distinct count comprises40Judge,14visualization,
7LP,7graph/sourceparity,8independent graph/physics,3acquisition and1fullpipeline.
Repeated runs are not added to the count. First copied test import typo was
corrected before acceptance. Default pytest temporary-root access failed on this
machine; the same tests pass using a new private C --basetemp, with original
failure logs retained. This is not represented as a numerical failure or success.

Notebook first execution exited0 but used unintended default python3 and Agg,
with no embedded PNG; it does not satisfy the notebook gate. It is preserved as
gcl_active_notebook_initial.json/log and private executed artifact. The corrected
execution explicitly selects the isolated shared-runtime kernel and inline
plotting, while library modules never choose a backend. Seven actual figures
were extracted and individually viewed (observation/truth/precision, compatibility
and ambiguity, full source/fresh/error, six-stage quality and both candidate
costs, complete pose error/orientation maps, deliberate output perturbations,
and identical seam crops). Final baseline comparison and final notebook receipt
will be appended after their actual execution, before completion.

The README and approach/design plans were reviewed against actual operators,
units, signs, input/output shapes, budgets and signatures. Necessary refinements
include minimum LEFT rather than fifth-smallest mode, current nn+diag/2 weights,
the negative source-turn sign, output correcting q versus acquisition s,
float32-to-float64 statistics and legacy boundary cost. No arbitrary seed is
part of the public problem. README is self-contained and contains no source URL.

## Completed original comparison experiments

Both additional methods were actually rerun on these identical 432 uint16 tiles,
without reshuffling or a historical inverse warm start. Exact input identity,
commands, native environments, warnings, source diffs, resource monitoring and
hashes are frozen in W2/gcl_baselines/baseline_review_ledger.json and
baseline_files_sha256.json. The source is the same verified commit. These are
reproduced upstream comparisons, not fresh local-code claims.

| Method | Actual total wall time (s) | Joint pose accuracy | Oriented-neighbor accuracy | Complete/unique fragments |
|---|---:|---:|---:|---|
| Original Gallagher, fixed output gauge | 19.521017600083724 | 0.6435185185185185 | 0.8905109489051095 | 432/432 |
| Original Yu type 2, complete 1728 hypotheses | 89.090 (rounded) | 1 | 1 | 432/432 |
| Selected GCL, first fresh local CLI | 6.0054921000264585 | 1 | 1 | 432/432 |

Times have different language, backend and I/O scopes and do not establish an
algorithmic speedup. Gallagher's fresh attributes/MGC/core take respectively
0.3652470112, 7.4001650810 and 10.4312250614 s. Yu's take 1.4460630417,
47.5670092106 and 38.8262469769 s. Both baseline runtimes use native GNU Octave
8.4.0, CPU one thread, in their isolated original environments; cleaned GCL uses
the shared Python/NumPy/SciPy runtime. Neither comparison result is substituted
for the selected local reconstruction.

The original Gallagher demo driver reshuffles and contains a GT-score break;
it is not used. Its complete core receives the bound ap once. GT-dependent
global-orientation scoring is removed from the inverse and handled only in
private common-C4 evaluation. Fixed output gauge does not change relative
assembly. The actual complete arrangement has the displayed real errors; it is
not repaired or filled with truth to meet the selected GCL thresholds.

Yu retains all 1728 actual rotation hypotheses, fresh attributes and Rui MGC,
SCO_modify, original replica handling and full LP/boundary alternatives with
rigid=1, buddy=1, loop=1, loop2=0 and speed_up=1. The original PuzzleMain has
commented-out label recovery/update after inner LP iterations; that source path
is preserved rather than silently replaced with PuzzleMain_test. A first run
failed after 73.57 s at a native singleton-dimension deletion. Its failure is
preserved; an independently checked keep-index primitive is the only additional
compatibility change. All 72 generic shape cases were checked, including the
native singleton failures and independent NumPy agreement. The rerun is fresh,
not a cached reconstruction. The original source returned rotation annotation
can be logical and lose group numbers. Physical correcting turns are therefore
derived from the actually selected raw replica IDs; candidate 0 is selected by
the first maximum raw rank among 16 candidates, never a true score. Both final
poses render from public raw tiles exactly equal to the respective actual source
images (maximum code-value discrepancy 0). No duplicate or hole remains.

Only tiny pose arrays are shipped as notebooks/reference_outputs/gallagher_pose.npz
and yu_pose.npz. Their consumer is the active notebook's method comparison,
including real images, all pose metrics, dynamic-range NRMSE and paired bars.
Full source grids/images/masks are private, avoiding redundant image copies.
W1/gcl_baseline_exports.json records original capture hashes and exact physical
render checks. Neither baseline's poor/complete classification changes evaluation
boundaries. Actual truth remains exclusively the acquisition poses.

## Earlier notebook and boundary evidence (before parent reopen)

The earlier direct notebook was executed with nbconvert through the explicit
shared-runtime kernel: W1/gcl_active_notebook_portable.json/log, actual exit 0,
13.332749400055036 s. Its setup prints the real interpreter and Python/package
versions, and actively runs the entire selected local reconstruction. Canonical
kernelspec metadata is portable python3, without a private kernel requirement;
the exact execution override and kernel argv are preserved in
W1/gcl_notebook_freeze.json. Source cells are byte-identical to the executed
definition. Temporary executed artifacts remain on C, outside the task.

All nine actual figures were inspected individually: observed tiles/truth/native
precision; full MGC and ambiguity; source/fresh/full-image error; all six stage
qualities and both candidates; 432-pose error and cumulative turns; deliberate
orientation/position stress controls; seam crops; both reproduced upstream
images versus fresh selected GCL; and paired actual pose/seam bars. The first
seven final PNGs are byte-identical to the seven already individually viewed,
and the two baseline figures were additionally viewed. Full PNG SHA and review
locations are in gcl_notebook_freeze.json. Missing groups are not filled with
zeros or cached local outputs. True zero fresh image error is explicitly labeled.

Final actual dual rendering: acceptance/rotated_fragment_image_assembly/math,
49 formulas, MathJax 3.2.2 and KaTeX 0.16.22 both zero failures. Markdown plans
use inline dollar math and fenced display math; the notebook uses compatible
inline/display delimiters. Source-string control-character and semantic symbol
checks pass. Render success is supplemented by the independently reviewed
factors, signs and units described above.

Actual public-only harness prepare and trusted judge CLI both exit 0:
W1/gcl_harness_gate.json. The prepared workspace contains exactly README.md,
requirements.txt and data/raw_data.npz before submission; no hidden source,
truth, fixtures or encoded result is exported. An inert fresh CLI submission is
copied only for trusted out-of-workspace judging. This is a boundary validation,
not a claimed blind coding-agent experiment. The task code uses no hidden inputs
at reconstruction time. AST import and manual call-role audits are recorded in
W1/gcl_package_audit.json; the final inventory/SHA manifest is privately frozen.

## Parent numerical reopen and rejected owned-input experiment

The earlier worker packet was not parent-accepted. The parent's separate clone
first completed its CLI, then failed the first full-pipeline pytest case at the
strict negative covariance quadratic-form guard; its remaining 79 tests passed.
The original source/clone hashes and actual failure remain in
acceptance/rotated_fragment_image_assembly/parent_command_gates.json,
tests.stdout/stderr and numerical_clone_source_snapshot.json. This was before
any pipeline LP or graph SVD, so prior test optimization state is not an
established explanation. W1/gcl_parent_gate_reopen_review.json records the
actual test-order and environment comparison. No native or hardware root cause
is asserted. W3's independent instrumented clone also produced a native access
violation at the final NumPy sum; that failure is retained, not recast as a pass.

An owned/contiguous-input-only experiment passed its predeclared three complete
CLIs and three complete 80-test suites, plus bitwise full source MGC parity.
Nevertheless, after the two-line version was adopted, an additional stage CLI
failed the same strict guard (exit 1, 1.6623686 s). Thus this experiment was
rejected as the final backend. Its stage test and notebook successes remain
historical only. The subsequent Judge exit 0 read an old output and is expressly
NOT fresh-solve evidence. All receipts are W1/gcl_owned_solve_probe.json,
gcl_owned_input_stage_patch.json, gcl_owned_stage_cli.json/log and
gcl_owned_stage_judge.json/log. The old code/notebook are preserved on C via
gcl_spd_adoption.json; no failed evidence is overwritten.

## Shipped explicit-SPD numerical primitive, version explicit-spd-v1.2

The shipped preprocessing file is copied byte-for-byte from the independently
validated guarded candidate, SHA256
3da6a1d8d07711447992023ae7045f6bbe8d6eac710cb7d89c19db839ccf1227.
For C=LL-transpose, it evaluates z=L-inverse times u by explicit float64 scalar
three-by-three Cholesky factorization and forward substitution, then sums the
three squared whitened components. This is mathematically the same SPD
Mahalanobis form, not clipping, jitter, absolute-valuing a bad result or
weakening a tolerance. Real/finite/symmetric inputs, positive finite pivots and
finite nonnegative output are required; the original strict negative-quadratic
guard remains in compute_mgc. All source direction/mirror conventions, raw
precision, covariance normalization and final float32 scores are unchanged.

W3/gcl_mgc_debug/guarded_cholesky_validation.json records all four original
attribute sets bitwise identical and all 2,979,072 finite MGC scores bitwise
identical (maximum ULP 0, relative difference 0), with 6,912 positive-infinite
self-pair positions exact. The predeclared 2-ULP/2-eps32 bound was not changed.
Independent 80-digit decimal scalar Cholesky checks cover 80 vectors/covariances
including worst-conditioned real edge covariances; maximum relative error is
7.892004100034233e-14. Windows longdouble has only 52 mantissa bits here, so it
was not falsely described as an extended-precision oracle. Independent audit of
all 1,728 original edge covariances gives minimum eigenvalue 2683.21066 and
maximum condition about 6049.64. This establishes valid SPD physics without
explaining the native failure. Private fallback/source audits are archived in
W3/gcl_mgc_debug, not imported at reconstruction time.

Before stage adoption, W1/gcl_cholesky_probe.json fixes and records the complete
finite experiment sequence: source full-MGC parity, CLI1, full80pytest1, CLI2,
full80pytest2, CLI3, full80pytest3, and an additional CLI4. All executed, with no
pass picking or tolerance modification. CLI walls were 4.9545808, 4.9745154,
5.0628372 and 4.9999689 s; full 80-test walls were 9.8503019, 9.7824721 and
10.1331402 s. The actual before-experiment plan, commands, source SHA and each
stdout/stderr hash are in C:/Users/admin/AppData/Local/imaging101_batch13_oracles/
w1_gcl_cholesky_probe. These are finite stability observations, not a proof of
the original native root cause or a universal stability guarantee.

After exact adoption, the standard stage CLI was additionally run fresh:
W1/gcl_spd_stage_cli.json/log, exit 0, 4.9136491 s; its fresh standalone Judge
then exited 0 (gcl_spd_stage_judge.json/log). All five pose rates equal 1.
The final stage suite is now 92 distinct pytest cases: the earlier 80 plus
12 independent analytic/guard tests for the new primitive, actual 92 passed
in 9.30 s (wall 9.6105351), gcl_spd_stage_tests.json/log. The guard test source
SHA is fb913ad13e574d776b777729b87a747b9b7b0e69de6cb77039a90228d98c5839;
only its import was changed to src.preprocessing. Repeated runs and the 80
decimal vectors are not added to the distinct pytest count.

The evaluator is still pose-v1.0, SHA
267a9d6ae97ba9e1fb4ef86604a18b508b43ce2d6ae4585efbe2114b8d4080ba;
both 0.95 thresholds and metrics file are unchanged. Method/operator numerical
compatibility versioning does not silently version the scientific target.

## Final active notebook execution isolation

The first direct-kernel notebook with the SPD backend failed at refinement 5
inside four_loops at a scalar-index exception; gcl_active_notebook_spd.json/log
preserves exit 1 and full traceback. Its first five stage messages are not
called complete reconstruction. No four-loop mathematical rule was modified.
Two private exception-only probes subsequently completed (11.2963185 and
11.4629333 s) without reproducing or capturing that exception. Their handler
would save actual local array types, shapes, strides, flags and values only
after an exception. Earlier private-probe setup mistakes (C notebook relative
TASK lookup and source-list normalization) never entered reconstruction and are
retained separately, not counted as numerical trials.

The canonical notebook now isolates the full fresh reconstruction from the
plotting kernel in one same-interpreter inline child script. Its only inverse
input is data/raw_data.npz; no truth, seed, historical score, prior assembly or
saved stage state is supplied. The child freshly computes initialization and
all five refinements once, saving the normal two-key submission and a separate
temporary NPZ with every six-stage pose, cumulative turn, goal, candidate cost,
source turn, branch, timing and LP trace. The notebook actively loads these new
results, evaluates the independent Judge against the fresh temporary submission,
and deletes the temporary directory. It is neither a cache preview nor a
reduced-budget experiment. The standard CLI remains a direct complete solve.

Actual canonical full nbconvert: gcl_active_notebook_child.json/log, exit 0,
11.9557817 s. Its outputs prove ACTIVE_CLEANED_CHILD_SUCCESS with the same
interpreter, the bound raw SHA, all six completed stage records, fresh Judge
exit 0 and all five rates 1. Portable python3 metadata and executed-source-cell
identity are recorded in gcl_notebook_freeze_child.json. All nine current PNGs
are byte-identical to the nine individually viewed images, verified by full SHA
lists. Historical source and both original comparison methods remain explicitly
labeled, and no missing reconstruction was filled with zeros. Native exception
causes remain unclaimed; notebook isolation is a documented execution boundary.

Current dual rendering passes 64 formulas in both MathJax and KaTeX, including
the explicit Cholesky/whitening expressions, after semantic factor/unit/sign
review. README, public raw and requirements are unchanged. The prior public-only
prepare/trusted-Judge boundary evidence therefore still tests the same public
assets and evaluator; parent reacceptance separately reruns the final clone.
Current AST/call-role audit and file inventory are frozen in the new private
packet manifest. No diagnostic hooks, private probes, decimal oracle, failed
source copy or temporary child output is shipped.

## Current worker gate status

Passed for the shipped explicit-spd-v1.2 candidate: verified paper/source/data,
complete bound original method, frozen independent fixture/physics gates,
local explicit method, fresh full CLI/Judge, 92 current tests, all output
quantities and poor/malformed contracts, raw-only reconstruction, complete
active child-isolated notebook with all six stages/two original comparison
groups/nine actual figures, dual math rendering, plans/signatures/units,
minimal validated dependencies and template/source-credit inventory. Independent
parent reacceptance is pending; earlier failed parent/owned/direct-notebook
attempts remain failed rather than being erased by current successes.

NOE remains rejected_unreleased and uncounted. The original 299 tasks and
the other accepted 20 tasks were not modified for this candidate.


## Independent parent acceptance, batch13

Parent independently reviewed all432 native uint16 RGB observations, the exact block/permutation/C4 forward model, cumulative physical rotations, complete integer pose contract and one common C4 alignment. This is synthetic fragment acquisition of an author-provided real photograph, not measured physical fragments. The actual known acquisition poses are truth; no reconstruction is relabeled as truth or exposed as calibration. Independent observation-only audits found no exact duplicate or rotationally symmetric tiles. The complete original selected method actually ran initialization and all five refinements, including both position candidates at every stage, with explicitly documented author Rui MGC, generic numerical compatibility and exact all-workspace/RNG cold-process continuation. Failed original native runs remain archived and are not called unmodified-program successes. Parent independently reconstructed each of the five original physical image assemblies from raw pixels and actual captured patches: all five coupled pose/neighbor rates1 and full-image dynamic-range NRMSE0. The original source, raw observations, all checkpoints and scientific thresholds were frozen before local inverse implementation. The independent Judge uses actual acquisition truth and two predeclared rates at least0.95, allowing other inverse algorithms rather than enforcing source iterates. Parent-owned40 metric/contract cases pass, including all common C4 transforms, big-endian int64, independently maximized incompatible gauges, finite quality boundaries, malformed submissions, evaluator failures and standalone exit/report behavior. These40 are counted once within the complete92pytest cases. Explicit local code preserves full16-channel MGC, normalized graph operators, minimum left singular mode, four-cycle updates, weighted L1 free-coordinate LP, source anchor and greedy/trim/fill conventions; applicable independent pixel-adjoint, covariance positivity, analytic and physics controls supplement original-function fixtures. The original LU path and an owned-input-only candidate had actual negative-quadratic/access-violation failures, and a direct notebook had a later scalar-index exception; these remain undiagnosed, with no hardware or proven-root-cause claim. They were not accepted by clipping, jitter or threshold relaxation. The current version uses explicit guarded float64 3x3 SPD Cholesky whitening; independently all2979072 finite float32 source scores remain bitwise identical, all6912 infinity positions match, Decimal80 controls validate the mathematics, and a fixed seven-complete-run private verification succeeded. Parent then independently ran the final complete CLI5.022s, all92pytest9.53s, foreign-working-directory Judge and64dual-rendered formulas successfully. Parent read every final notebook cell and individually viewed all nine computed figure groups. The final active notebook launches one fresh local child with raw observations only, executes initialization plus all five refinements, receives all actual numerical histories and LP traces, independently judges its newly generated submission, then cleans its owned temporary files. This separates reconstruction from the plotting kernel without caching an inverse, changing the method or dropping iterations. All nine final PNGs are byte-identical to those actually viewed by the parent. Actual complete Gallagher19.521s and full1728-hypothesis Yu type2 89.090s comparisons retain their own provenance and are labeled reproduced upstream, distinct from fresh local GCL; physical rotations are independently checked against rendered pixels. Gallagher joint0.6435185/oriented-neighbor0.8905109 correctly fails the unchanged standard, while the different Yu algorithm passes, demonstrating implementation fairness. Different backend and I/O runtimes are not claimed as algorithmic speedups. Source diagnostic bugs, singleton-slice portability changes and all failed attempts remain identified. Minimal runtime imports, public raw-only solvability, equation/sign/unit/API consistency and the six-part scientific narrative were reviewed. Missing redistribution grants are not invented; this is authorized local research, with no publication or push implied. Original299 tasks and all20 earlier frozen packages remain retained. This is trusted reference-submission validation, not a blind Agent solve trial.

Parent public allowlist export contains README, raw NPZ and requirements only; the shared environment is provided separately for this private staging test. Trusted harness score_submission evaluated the inert submission with isolated imports and exited0. This is an export/contract test, not an OS-isolated Agent trial. Private acceptance/rotated_fragment_image_assembly/ records commands, reports, actual extracted canonical figures and final hashes.
