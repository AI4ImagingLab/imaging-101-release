# Validation and provenance: complete measured spectral material tomography

## Actual-operand power repair and bounded verification

Full fresh1000 parity, final Judge, the full notebook and complete tests remain
**unverified**. No reference, threshold or submission has been promoted.

The later actual-input diagnostic resolved the earlier Q163 uncertainty:
original Q163, lifted Y164, WA163 and the calibrated dictionary were bitwise
equal to the saved local operands. The old local R164 was reproduced bitwise,
establishing that this isolated replay applies to the observed divergence.
Eight source SPG stages matched before trial9 first differed only in f_new by
one ULP; its residual, norm, gradient and candidate weights were bitwise equal.
An independently executed original Octave evaluation of that same residual
returned norm780.4069459910742 in both implementations. Linux original scalar
power produced609035.0013511154; Windows power and multiplication produced
609035.0013511153. Original direct multiplication also produced the latter.
The original power objective210.95464767189657 matches its actual archived
objective bitwise; the old local value was210.95464767189654.

The runtime now evaluates the three norm-squared objective expressions through
generic Linux libm power in SourceBlas. This preserves the original expression,
with no value-specific branch or imaging logic in the arithmetic worker. An
independent original Octave batch captured412 scalar powers: three actual R164
residual norms, earlier independent norm cases, all384 actual cubic d1 values
from101..164, finite numerical boundaries and general non-square powers. Every
cleaned generic power matched the independent output bitwise. All384 actual
source cubic steps also matched the existing local formula bitwise; source
d1^2 and explicit d1*d1 agreed on all those actual inputs. This observation
does not assert equivalence on every float. The completed scalar-role audit
then retained the original explicit d1^2 operation through the same generic
power helper, instead of extrapolating multiplication equivalence to unseen
iterations. This is an operation-preserving port, not a claim of an observed
cubic discrepancy. Original explicit multiplications in the scaled-norm
algorithm remain unchanged. All384 actual cubic cases still match bitwise
inside the actual SourceBlas context. The scalar/cubic/checkpoint tests plus
the earlier independent scalar-role tests passed all5 cases in2.38s, exit0.

Frozen fixture `test/fixtures/upstream_scalar_power.npz` has SHA256
`8cbcaa8446855b441695c4183c38592fcdaa845afe37b83c47a851ef8fcc0fd2`;
its consumer is `test/comparing_test/test_scalar_power.py`. Original MAT SHA256
is `540bb0775a70af11176743b0b61a6a3ed69c0549c52f8e7d51d65131c96ace13`.
Original capture-script SHA256 is
`eafc8e8bc903ba9261ae3791d72171abb6b20ff6df34f8c430f96c4e144a0c50`.
These assets contain independent upstream outputs, never cleaned-generated
expected values. The subsequent repaired actual R164 replay matched all16
source SPG stages, every compared field and the final source R164 bitwise.
The final R164 F-byte SHA256 is
`fdcfa7a42a7bcc59f2292087dc8a042a96e1dc85848cb915e63dc37814ea163d`.

After completing the source cubic-power role, a final separate actual-input
R164 replay again matched all16 source SPG stages and final R bitwise. Driver
`validate_repaired_allpowerroles_R164.py` and report
`original_resume164/repaired_allpowerroles_R164_validation.json` preserve this
additional evidence without replacing the earlier repair report. It exited0
in41.235s including30s gating; peak Python private commit559,644,672bytes,
global physical/commit minima10,612,822,016 /11,089,907,712bytes. Both owned
Windows process IDs and Linux worker absence were checked after real exit.

Private archive additions under research10/task03: `probe_objective164_scalar.py`,
`verify_scalar_power_repair.py`, `validate_repaired_R164.py`, and
`original_resume164/{scalar_probe_original.m,same_x_objective164_scalar.json,
capture_scalar_power_cases.m,independent_scalar_power_cases.mat,
scalar_power_fixture_validation.json,repaired_R164_validation.json,
power_cases_original.log,power_cases_original_time.txt}`. Both bounded drivers
exited0, taking75.297s and41.093s including30s resource gates. They used no GPU,
reserved1.5GiB and required both free physical/commit>=7.5GiB stable30s with
6GiB floors. Their minimum available physical/commit were10,197,037,056 /
10,845,429,760bytes and10,610,941,952 /11,093,180,416bytes respectively;
Python peak private commit138,231,808 and559,095,808bytes. All owned Windows
Python/WSL launchers and Linux Octave/BLAS workers were checked gone afterward.
The selected Ubuntu24.04 Linux arithmetic environment and four-thread
OpenBLAS0.3.26 context are the same separately identified environment below.
This bounded validation fixes the demonstrated first cause; it does not
establish a complete fresh1000 run or accuracy of the material estimates.

The generic scalar worker uses Ubuntu24.04 Python3.12.3 / GCC13.3.0 and
glibc2.39 on WSL2 Linux6.6.87.2. `/usr/lib/x86_64-linux-gnu/libm.so.6` SHA256
is `1b87a1a50b496cfead2b0ad134c2ff536705c82608db240c7e8aa48d6c0e4217`;
libc.so.6 SHA256 is
`d8db8739a1633c972cec6a4fe0566bdcec6fd088f98723492ab0361f66238f75`.
Read-only environment/hash capture exited0 and is archived as
`original_resume164/linux_libm_environment.json`. This is a measured platform
exception for source arithmetic; arbitrary host libm equivalence is not claimed.

An additional independently executed original checkpoint100/300 regression
captured the actual A/R Frobenius norms and original alternating feasibility
projection of each R. The four norms and both projections match cleaned code
bitwise. This is a fixed-state regression, not a full300 trajectory replay;
Q300 and its complete SPG inputs were not captured. Frozen fixture
`test/fixtures/upstream_checkpoint100300.npz`, consumed by
`test_scalar_power.py`, has SHA256
`ced5b32504b59f8148540b324ccb4e647e697c8b224875237e167682909fe03a`.
Original MAT SHA256 is
`f00b2075d016064b96e734b79a45f356a0dc60839f85e0251c23322c08998013`.
Private capture/check drivers are `capture_checkpoint100300_original.m` and
`verify_checkpoint100300_regression.py`; output is
`original_resume164/checkpoint100300_regression.json`. Original and local
drivers exited0 in32.516s including the30s gate; no GPU was created. Observed
global physical/commit minima9,511,219,200 /10,103,091,200bytes exceeded6GiB.
Python peak private commit144,175,104bytes; both owned Windows process IDs
were verified gone. The same explicit numerical environment was used.

The scalar-repair test run passed36 mathematics/stage/generic-primitive tests
in122.34s but20 Judge-contract setups failed because the system temporary
directory was inaccessible (`WinError5`). Re-running the22 Judge-contract
tests with a new private `--basetemp` passed all22 in7.45s, exit0. No scientific
test tolerance or expected value was changed. The former run's20 errors are
environment failures, not passing tests. Complete full-output tests remain
pending fresh benchmark and notebook execution.

After confirming the original controller's actual source/relay exit0 and that
the obsolete conversion watcher had exited, the completed original MAT files
were converted with `convert_completed_original_results.py`. This converter
checks their fixed SHA256 values and reads the actual original best-return
arrays, not a later checkpoint. It exited0. Newly identified review assets:
`notebooks/reference_outputs/upstream_ordered_corrected.npz` SHA256
`86bceb7a897b33e5e91c5ad08aa6b47176528f9e6b6526c8270a21c0ae3e4be9`,
and `notebooks/reference_outputs/upstream_ordered_cjoint.npz` SHA256
`a42f96eba3d3a57bb782e0f5af2ed1eb5cabfda8e67cd8ee4bd13d7fc915fa3f`.
Their consumers are the notebook and complete-run comparing tests. Conversion
manifest: private `completed_source_conversion.json`. This step creates no
baseline and makes no fresh-output parity claim. The additional three scalar,
cubic and checkpoint regression tests subsequently passed in1.12s, exit0.

Static lifecycle review found that the ordered-GPU SIRT branch retained both
CPU CSR matrices after the two device copies were complete, although subsequent
GPU iterations never use those host arrays. The branch now deletes only these
two redundant references; the current stream is synchronized before release so
asynchronous host-to-device copies cannot outlive their host buffers. The CPU
path, dimensions, coefficients, ordering and updates are unchanged. A
coordinated single tiny-GPU check on an unsorted4x4
CSR with two channels and five SIRT updates was bitwise identical to the serial
CPU path, taking1.487s; the process exited0 and its actual PID was checked gone
before the other task's full GPU context could launch. Private evidence:
`sirt_gpu_host_lifecycle_check.json`. This is a lifecycle regression, not new
complete128-channel calibration evidence; that remains part of the full notebook.

The two superseded unordered-CUDA review arrays without active notebook/test
consumers were moved intact to private
`unused_historical_reference_assets/`, with a SHA-verified `archive_index.json`:
`upstream_cjoint.npz` SHA256
`ecb3ea986ef4dd7df07eb5f46074b3d6cf2e9728c0fc177b6c98be16f8492ced`,
`upstream_corrected.npz` SHA256
`6e902c200b49856a3831b98e2e2ebc6aff07223e45690b7440e37959f630c824`.
The actively displayed historical unmodified-original result remains identified
as `upstream_original.npz`; current independent comparisons use the two ordered
assets above. No current scientific comparison group was removed.

## Exact-state continuation evidence, 2026-10-19

The task remains **incomplete**: full fresh reconstruction parity, the active
notebook, final Judge measurements and complete test suite have not passed.
No baseline or submission is promoted by the following private diagnostics.

A private cleaned-code continuation from the original CURRENT state100 stopped
at its first history-word difference at164. All four history columns101:163
were bitwise equal to the genuine original full1000 run. At164 only iterate
progress differed: local0.02827847010367804 versus
original0.0282784701036781, an18ULP difference. Residual, residual progress and
rho were bitwise equal. The actual local state164 and preceding state163 were
saved privately; SHA256 of `private_continuation100_first_drift.npz` is
`ab39c7664de40e5a3561b397baee959f60b4d0c82eaab03763814143772939db`.
An independent original Octave evaluation of the norms of those actual saved
local deltas matched the cleaned norms bitwise. Therefore the norm calculation
on those particular deltas does not explain the observed progress difference.
Evidence: `first_drift164_norm_operands_proof.json/.log`.

An independently executed original CURRENT100-to164 continuation then restored
actual A/R/Q100, the preceding history and post-update100 rho0.01. It recomputed
the original norm of Y and preserved the original SPG, projection, cubic
interpolation and ordered float64 sparse products. The already verified original
CPU SciPy CSR oracle was used, with only its redundant same-dtype matrix cast
removed and dtype explicitly asserted. It retained the original transpose and
CSR traversal order. The random initialization was still executed before state
restoration. Best-return selection was reset at101 for diagnostic purposes;
it never enters the update equations, and these returned arrays must not become
a benchmark reference. Every101:164 A/R/F state, SPG stage scalar/hash and actual
cubic interpolation input/output was captured. Objective/SPG full operands and
Q/residual/previous states were additionally captured at163 and164.

The original continuation and relay exited0 after1388.25seconds of controller
wall time. **All four history columns101:164 are bitwise equal to the genuine
original full1000 source history**, whose MAT SHA256 remains
`e908542717c58f957f548a9cf268d0c1a78f87753db1fd16799a150d1765d496`.
Its independent small state checker also exited0. Actual process-exit checks
confirmed that the controller, relay launcher/worker, WSL launcher, Octave and
checker had all exited before any later numerical context was authorized.

The actual original163 A and R are **bitwise equal** to the saved local163 A
and R. At164, five of16 R entries differ, with maximum absolute difference
5.551115123125783e-17;15,846 A entries differ, with maximum absolute difference
1.1102230246251565e-16. Thus arithmetic disagreement is present during update164
despite identical preceding A/R. At that diagnostic stage Q163 and the first
differing SPG operation were unresolved. The later actual-operand tests and
generic scalar-power repair documented above resolve this first divergence;
no threshold was changed.

The CPU diagnostic used a3.5GiB conservative additional host reservation,
both free physical and commit memory at least9.5GiB for30seconds before load,
and a6GiB global floor checked every0.5seconds. It created no GPU context.
Observed global minima were8,385,609,728bytes physical and8,668,082,176bytes
commit, so the floor was not crossed. Relay native peak private commit was
1,280,876,544bytes and peak working set1,188,646,912bytes. Original Octave's
independently measured maximum resident set was1,064,740KiB. Its actual running
library mappings identified the selected generic OpenBLAS0.3.26 binaries;
requested numerical thread environment was4. The actual archive inventory has
1,624 capture files totaling3,886,840,806bytes;22 full-objective operand files
account for3,445,317,087bytes. This fits the initial4GiB disk reservation.

Private archive index additions: `prepare_source_resume164.py`,
`launch_original_resume164.py`, `oracle_sparse_server_resume164_cpu.py`,
`check_original_resume164.py`, `check_resume164_history_stdlib.py`,
`original_resume164/preparation_manifest.json`,
`original_resume164/controller_status.json`,
`original_resume164/history_and_states_proof.json`,
`original_resume164/actual_lifecycle_resource_completion.json`,
`original_resume164/relay_native_resource.json`,
`original_resume164/octave_native_time.txt`,
`original_resume164/actual_octave_runtime_mapping.json`,
`original_resume164/captures/`, and `original_resume164/completed.mat`.
The continuation MAT SHA256 is
`e1e3fbb65e22d1f8ec6f3aa35667c687c728dcea815dd8301cb2fc5936be382c`.
These private source instruments and arrays are not imported by the cleaned
task and do not satisfy fresh reconstruction validation.

## Full-trajectory failure and investigation, 2026-10-19

The complete independent original sequence has now exited successfully:
selected ADJUST1000 followed by cJoint500, source_exit0 and relay_exit0,
controllerwall10078.748230899917seconds. Original cJoint returned finite
float64 Aj(65025,4) and Fj(4,65); all four500-value history arrays,
resmin and runtime are finite. Its minimum and last residual coincide at
iteration500:0.13036023239035935; actual method runtime3490.8370950222015s.
The original cJoint MAT SHA256 is
`d0daf538b00189a2876d005d8d721760a636237f821148cb731355ed626e2461`.
`actual_original_cjoint500_completion_proof.json/.log` records shapes,
numeric-array hashes, finite checks, source/controller status and whole-process
exits. Its read-only checker uses standard-library MAT-v5 parsing only,
avoiding another numerical/GPU context during active private investigation.
This completes the original-run gate; cleaned full parity, comparisons,
Judge, notebook and final-suite gates remain failed or pending. No accepted
baseline was promoted.

The serialized CPU-only higher-thread probe completed exit0 at22:07:37UTC.
Actual automatic NumPy BLAS threads were24 with the current thread environment
unset; requested16,24,32,64 produced actual16,24,24,24 under OpenBLAS
0.3.31.188.0 Haswell MAX_THREADS24. On full captured original step10 operands,
all three spectra/prediction/spectral-adjoint products at all five conditions
were bitwise source: zero differences, maxabs0 and maxULP0. Native measured
peakcommit was1,100,980,224bytes, peakworking-set313,778,176bytes; child
walltime4.077999999979511seconds. Sampled freecommit/freephysical remained
above6GiB. Full operand/output/source hashes and exact configuration are
in `cpu_thread_probe_child_status.json`; lifecycle verification is
`cpu_thread_probe_completed_verification.json`. All CPU processes exited
before restoring the original strict continuation waiter. This bounded result
does not establish historical failed-run thread environment or parity of
later operands/line-search branches. No accepted baseline was generated.

The private original-state100 diagnostic remains resource-gated and cannot
satisfy a fresh reconstruction gate. At21:38UTC its verified stdlib-only waiter
was replaced with a guard requiring the existing9.5GiBfreecommit/freephysical
and GPU allocation conditions to remain qualified for at least30seconds before
any numerical imports. Failing polls reset this timer; polling is10seconds.
The existing6GiBpostload protection and scientific continuation are unchanged.
Prior state/script, command/PID/zero-child evidence and replacement rationale
are archived in `private_continuation100_stability_gate_replacement.json` and
its indexed before-state/script files. This touched no original scientific
process or benchmark/evaluation output.

At2026-10-18T21:14:15UTC the independent selected original ADJUST completed
all1000updates and wrote `oracle_ordered_corrected1000.mat` privately; SHA256
`e908542717c58f957f548a9cf268d0c1a78f87753db1fd16799a150d1765d496`.
Measured epoch-clock method runtime was6567.527808189392seconds, including
the observational capture overhead. Returned Ac(65025,4), Fc(4,65), Rc(4,4)
and T(4,65) are finite float64 arrays. All1000values of each residual,
residual-progress, iterate-progress and rho history are finite and match the
actual final checkpoint history bitwise. The selected returned best occurs
at795 with residual0.153078688318426; the last/current1000 residual is
0.15784104558113948. Returned arrays differ from last/current arrays, and
returned Fc equals Rc*T bitwise. This result is an original-method run;
it has not been promoted to an accepted hidden baseline. Full evidence,
source/log/checkpoint hashes and snapshot are indexed in
`actual_original1000_completion_proof.json/.log`. The same original process
naturally continued into full cJoint500; its whole-process exit status remains
pending and must not be recorded as zero until it exits.

**Mandatory full reconstruction parity is failed; cleaning is not complete.**
The actual fresh ordinary CLI and notebook attempt developed numerical drift
against the independent original full run. At iteration400 the source residual
was0.1903396330775972 and the fresh printed residual0.190339694; at500 these
were0.18008432559370652 and0.176151712. Matching rounded earlier residuals
does not establish bitwise trajectory agreement. The first microscopic divergence
has not yet been located.

After preserving process identity and logs, the coordinator authorized stopping
only verified owned fresh CLI PID37808 after iteration550; its notebook and BLAS
worker subsequently exited naturally. This was an unsuccessful reconstruction
and notebook execution. Neither a fresh submission nor an accepted hidden
baseline was produced or promoted. The independent original full source and
its relay continue unchanged. Thresholds remain frozen.

Private evidence is indexed under
`research10/task03/failed_ordered_full_attempt_20261019/`, including
`interruption_evidence.json`, before/after notebook logs, source log snapshot,
and read-only source/fresh-history comparison reports. The independent CPU100
versus controlled local orderedGPU100 entire four-column history is bitwise
equal (`prefix100_entire_history_bitwise_proof.json`); this prefix does not
establish full1000 agreement.

Bounded diagnostics have established: all actual128-channel float64 corrected
observations and selected65channels are bitwise source inputs
(`actual_full_float64_observation_proof.json`); actual source R*T products at
checkpoints100..600 match local explicit F-output products bitwise across tested
layouts; original projection outputs on actual R200/R300 match local outputs
bitwise (`small_actual_projection_diagnostic.json`). NumPy/source-order norms
can differ by one ULP on small differences, but no resulting projection branch
or output difference has been demonstrated. Fixed original dense spectra,
prediction and spectral-adjoint operands match source bitwise with actual
Windows NumPy OpenBLAS threadcounts1,4,8
(`small_windows_thread_product_diagnostic.json`). Historical Windows thread
environments of the prefix and failed CLI were not recorded and are unknown;
Linux source-compatible BLAS4 was independently recorded. These diagnostic
results do not identify or resolve the full-trajectory discrepancy.

The older running-status snapshot below is retained for chronology and is
superseded by this failed-attempt status.

## Authoritative selected-path status at 2026-10-18 19:29 UTC

The task is **not complete**: actual ordered original1000/cJoint500, fresh notebook/CLI1000 and its comparisons, full Judge, final suite and visual review are active or pending. The historical investigations below are dated evidence snapshots; their earlier pending statements and rejected backends do not override this current status.

At19:37UTC the actual independently executed ordered full-source run reached100updates. Every captured A/R checkpoint1..20 and30,40,...100 is bitwise equal to the independently executed original CPU100 run; `actual_ordered_full_source_cpu100_proof.json/.log` records this additional actual-source check. Its passive checker exited naturally. The selected fresh notebook has independently recomputed its complete128-channel100-update calibration and is actively executing its ordinary1000-update CLI. The ordered fast regression suite completed with54passed, one known historical PyTorch CSR beta warning,20.36s, exit0; log `pytest_ordered_fast.log`. This excludes the still-pending full-result parity tests and the final complete-suite run.

Resource evidence: two eager-import passive source/Judge observers each committed approximately0.75GiB while only waiting. Only those verified-owned idle helpers were canceled; their command lines and private-memory counts are preserved in `deferred_observer_cancellation.json`. Active source/relay, freshCLI, and notebook were not interrupted. Their replacement `deferred_source_observers.py` imports only Python standard library while waiting. It starts numerical conversion only after the actual source1000 result and freshCLI result both exist with stable timestamps and at least3GiB commit headroom, then performs standalone fresh Judge and poor-output checks. Numerical child thread counts are fixed to one. This deferral changes no mathematical computation, threshold or reference selection. Current wait state and logs are indexed in `deferred_source_observer_status.json` and `deferred_source_observer.log`.

| Gate | Status | Evidence |
|---|---|---|
| Independent actual original CPU100 | Passed | Original corrected ADJUST/MinConf in Octave, independent SciPy CSR relay; source1805.8656543000834s, controller1813.7499583000317s; source/relay exit0 |
| Cleaned orderedGPU100 vs independent original CPU100 | Passed | Genuine fresh public Y and previously freshly reconstructed exact T; every captured map/R at1..20 and30,40,...100 bitwise equal; unchanged elementwise rtol2e-10/atol2e-11;875.4808599000098s including0.3116755998926237s source waits |
| Full128-channel100-step local CPU SIRT | Passed | All8,323,200values,128quartz,260T values bitwise original;2420.6434092000127s |
| Full128-channel100-step shipped orderedGPU SIRT | Passed | Fresh public raw data/local fullW, all8,323,200values bitwise independent original and fresh CPU; quartz/T bitwise fresh CPU;134.6551714000525s calibration,143.0354508999735s total; process exit0 |
| Independent general ordered GPU primitive reruns | Passed | Float32/64 unsorted cancellation cases; actual checkpoint forward/transpose four repeats bitwise SciPy; independent full float32 SIRT repeated twice bitwise original |
| Complete selected original/local1000 and cJoint500 | Running | Actual source launch2026-10-18T19:24:28.404895Z; private launch_ordered_full.py; no iteration or acquisition reduction |
| Active six-section notebook and fresh ordinary CLI | Running | python -m jupyter nbconvert --execute; ordinary python -m src.main --backend cuda_ordered; all128channel calibration freshly recomputed |
| Final full-suite/Judge/visual/document consistency | Unverified | Must pass after active runs; no completion claim |

All four prefix100 components have nonzero dynamic ranges. Weakest spectral range is0.0007882960277534807. Independent source100 has weight-row equality error6.985868551412366e-7, nonnegativity0, abundance-sum excess2.220446049250313e-16 and weight-column excess2.220446049250313e-16. The approved boundaries were frozen before fresh full computation, at2026-10-18T19:24:14.900448Z: every component of all three outputs requires NCC>=0.999999 and dynamic-range NRMSE<=1e-6; only R-row equality has2e-6 tolerance from the four-entry Cauchy bound on source1e-6 feasibility discrepancy; all other feasibility quantities use1e-10. Fixed12-view observation relativeL2 must be<=0.17007280335223202, frozen as1.1times prior independently measured source0.15461163941112. This is method reproduction and public-data consistency, not chemical truth accuracy. Frozen metrics SHA256 is f8a2343a78964f5a27fb7a90b21bbe48d822c95c899035b026397ebd224c20c7; proposal SHA256 b8789d77b98c3f646e96001e16030b358e2bf05a966d9e4c82bf1df17a5b6373. No baseline is promoted before strict complete1000 parity.

Current environment: Windows10.0.26200, Python3.11.11, NumPy2.4.6, SciPy1.17.1, CuPy13.6.0, PyTorch2.9.1+cu128; selected CuPy CUDA runtime12090/driver API13030, NVIDIA RTX4080SUPER16GiB. CuPy supplies RawKernel/device buffers only, never a domain reconstruction or sparse solver API. Its explicit ordered scalar kernel disables fusion and uses rounded scalar products/additions. Selected float32 SIRT retains CPU source-order normalizers, pointwise float32 updates, and contiguous float32 ROI mean. StandardCLI default is cuda_ordered; PyTorch sparse modes are historical diagnostics whose nondeterminism was independently observed. Generic Linux BLAS remains the independently validated exception described below.

The prefix diagnostic's original default-pool counter was mistakenly labeled GPU peak allocation. Its unchanged raw status is archived privately; corrected status calls it default_pool_reserved_bytes_at_exit and explicitly states that OrderedCsr private pools are excluded. No zero-memory or precise peak claim is made from that counter.

Selected independent outputs use distinct upstream_ordered_corrected.npz/upstream_ordered_cjoint.npz identifiers. Private watch_ordered_source.py publishes complete source assets by atomic rename and checks fresh CLI shapes/dtypes/finiteness plus unchanged complete-array parity before promoting independently computed source arrays to baseline_reference.npz. The notebook waits only after its genuine fresh reconstruction, then actively computes all comparison groups and a1000-update seed11 stability repeat. Historical unmodified/control sparse CUDA uses a different nondeterministic backend, so control-vs-selected differences are not an isolated guard-correction ablation. Runtime of the new selected source is measured by epoch-clock differences, independent of nested tic/toc; historical external saved-file timings remain separately labeled.

Private archive index additions: cpu_source_prefix_status.json/oracle_cpu_prefix.log/oracle_cpu_states; deterministic_prefix_status.json/deterministic_prefix.log/local_deterministic_states; threshold_proposal.json/frozen_full_thresholds.json; ordered_actual_checkpoint_proof.json; root_csr_benchmark.json; root_sirt_float32_diagnostic.json; fresh_gpu_calibration_proof.json/fresh_gpu_calibration.log; ordered_environment.json; ordered_source_full_status.json/oracle_ordered_full.log/oracle_ordered_states; ordered_source_conversion_status.json; ordered_full_notebook.log. Private source/relay/controller/profiler scripts are not cleaned imports or public materials. Test complete_reconstruction consumes identified actual source/fresh arrays, including full cJoint parity, after mandatory run completion.

## Historical investigation chronology and configuration evidence

## Current four-material configuration: frozen, not complete

### Additional numerical and independent-evaluation investigations (19 October)

**Independent deterministic-CPU source pairing.** `launch_cpu_prefix.py` runs the actual corrected original ADJUST/MinConf in Octave with `oracle_sparse_server_cpu.py`, an independent SciPy CSR service importing no cleaned modules. The source reads its independently prepared measured Y, original full W and original CPU-SIRT dictionary. The local `cpu_sparse` mode separately builds W from public geometry, uses genuine fresh Y/T, and preserves C-input/F-return dense layouts. Every captured map and dictionary-weight checkpoint through iteration 50 is bitwise identical. At iteration 50 the private consumer first read the original MAT while its writer was still saving and raised MatReadError; this is an I/O synchronization error, not an algorithm divergence. The local iteration-50 arrays had already been saved. After the source file completed, `cpu_local_50_post_io_proof.json` confirms both saved local arrays remain bitwise exact. The original CPU 100-step process continues unchanged; the private consumer now retries incomplete MAT files and verifies their captured iteration/key completeness. No source state is substituted into local reconstruction. A complete fast ordered-CUDA replay will be compared with all completed original CPU 100-step checkpoints before any full run.

**CUDA sparse numerical-stability gate failed independently of extraction.** Private `probe_first_forward.py` compares the actual complete source archive with the newly built local W: all 46,901,466 float64 weights, all indices and pointers are bitwise equal, as are A0/R0. Both CUDA constructors yield identical int64 CSR indices and float64 values. Nevertheless, six repeated multiplications using the **same original CUDA CSR object and same dense tensor pointer** differ by `5.684341886080802e-14` to `7.105427357601002e-14`. Repeating with `torch.use_deterministic_algorithms(True)` gives the same nonzero range; the flag does not make this installed CSR operation reproducible. Source-constructor/local-constructor outputs and repeated local outputs likewise differ at this scale. These are independently observed primitive-reduction variations, not input/calibration/operator discrepancies. `first_forward_proof.json/.npz` and `first_forward_determinism.log` preserve the proof. They explain why matching isolated states does not establish reproducibility of a long sensitive trajectory. A deterministic source/local numerical configuration must be independently validated before any new full run or nonzero reference-tolerance proposal; no threshold has been loosened to absorb the previous failed full results.

**Initial-feasibility source arithmetic.** The first genuine fresh-calibration/preserved-CSR 100-prefix stopped naturally at its first unchanged elementwise failure, iteration 13: maps maxabs `5.996034224686753e-11`, relative L2 `2.790431262844376e-11`; weights still passed. Prefix wall time was `104.5837109000422` s and peak CUDA allocation `1530899968` bytes. This is failed strict parity evidence, not a completed 100-step run. Independent source capture `root_initial_projection.m/.mat` then identified an arithmetic association: the source precomputes `1/j` and multiplies cumulative offsets; local division by j differs one ULP for active count three. The independently executed corrected-source initial R projection takes 39 alternating passes. After explicit reciprocal multiplication, every row and column half-step and the final result are bitwise identical. Original MAT SHA256 is `fb0d8df325640099e5f5842a05aa985252f77f1f403c0606f8996b77dc76adda`; frozen `test/fixtures/upstream_initial_projection.npz` SHA256 is `8bb085d069e14bffde75ffc5879afd9bb78d7593daca22a79fb44cc2f6a46280`, consumer `test/comparing_test/test_initial_projection.py`. Twenty projection/scalar/mathematics tests passed in 13.44 s. The independent deep-SPG proof remains bitwise exact after this correction (2.6853886999888346 s, worker exit 0). Prefix must be rerun after the additional sparse numerical-stability investigation; no acceptance boundary changed.

**Full fresh 128-channel calibration passed exactly.** `research-runtime/Scripts/python.exe validate_fresh_calibration.py` freshly loaded the selected public NPZ, constructed the entire local acquisition matrix, ran all 128 channels for all 100 CPU source-ordered SIRT updates, and computed the quartz ROI and selected augmented dictionary locally. No captured array was used as reconstruction input. After computation, comparison against the independently executed original established bitwise equality of the corrected measurements, all 8,323,200 channel-image values, all 128 quartz values and all 260 dictionary values: maximum absolute and relative-L2 differences are zero. CPU SIRT took 2420.6434092000127 s; total validation took 2428.6214347999776 s; completion UTC was 2026-10-18T18:34:47.679431. Private `fresh_calibration_status.json`, `fresh_calibration.log` and `fresh_full128_calibration.npz` identify the run. The latter SHA256 is `f23bc7d1596c5a6a31acbc2e87b52cbc3852ab814c81c2351598f465d53dcf8b`; it is an investigation artifact, not a runtime input. The active notebook's CLI must still recompute this calibration.

**Rejected generated submission quarantined.** The former generated `output/reconstruction.npz` was archived privately as `pre_source_compatible_output_reconstruction.npz`, SHA256 `9c8ded2c2d79eb4c4711b44d48f352bee746d3e140920f84bbfb934446b17175`. It predates the integrated source-compatible arithmetic gates and is not an accepted submission or hidden reference. The task output directory will receive only the newly executed complete pipeline result once that gate runs.

**Integrated regression suite after extraction corrections.** `python -m pytest test/ -v --basetemp <private-unique-directory>` completed with **53 passed, one known PyTorch CSR beta warning, 109.39 s**, exit 0. The exact `assert_array_equal` comparison of both distinct original measured channels after all 100 CPU-SIRT updates is included and passed. The suite also exercises actual CPU/CUDA sparse-call ordering, scalar associations, generic-worker failures/cleanup, operator adjoints, objective finite differences, feasible constraints, original staged updates, independent regularized linear solves, metric conventions, and malformed Judge contracts. The private command output is `pytest_pre_prefix.log` under the task03 archive. This regression suite does not replace the still-pending full 128-channel calibration, 100-step source prefix, 1000-step reconstruction, and active notebook gates.

**Exact ASTRA source identity.** The original CPU-SIRT implementation was checked against the remotely verified ASTRA v2.5.0 annotated tag object `8676badda7eb2757e4218e36ca7c32eedb4aa089`, resolving to commit `934920a46781142c2b8f0177007c395eae601783`. Private API captures are `astra_v250_remote_tag.json` and `astra_v250_remote_commit.json`. The archived files and SHA256 values are `v250-SirtAlgorithm.cpp` (`ca8e2984562a88f5aacec3f6c62ed268815a9c9f0868b3c99a0dc6dce076ce31`), `v250-DataProjectorPolicies.inl` (`3c7d7c450b552913a58df228773a371bb03b5160cb88abff7f2b51b1504dff38`), and `v250-SparseMatrixProjector2D.inl` (`186ce63f11d2cf28ad9a40f78c0da379ad27af368a28ab453352acf0e3374f5f`). Their consumers are the local source-ordered SIRT extraction and the independent original first-step/normalizer fixtures, not a cleaned-runtime import. No ASTRA package is needed by the cleaned method.

**Integrated sparse-order correction.** A later actual-call audit found SciPy CSR astype changes precision by deduplicating/sorting entries. Earlier "order-corrected full" trials checked the entry matrix before ADJUST, then ADJUST's float64 conversion sorted it again. Consequently those runs are entrance-order-only diagnostics, not true source-order-preserving runs. The local cast_sparse_preserving_order explicitly reconstructs CSR using independently cast data and unchanged pointers/indices. test/comparing_test/test_runtime_sparse_order.py checks both an unsorted dtype regression and the actual adjust-to-LinearMap.multiply boundary against the independent nine-angle fixture; both pass, together with the cubic regression, three tests in0.44s. This correction applies to float32 SIRT and float64 ADJUST/cJoint. Existing historical order-only measurements below remain failed evidence and must not be interpreted as validation of the corrected integrated path.

**CPU SIRT extraction.** Original verification uses ASTRA2.5.0 sparse-matrix CPU SIRT, not GPU SIRT. Versioned source SirtAlgorithm.cpp, DataProjectorPolicies.inl and SparseMatrixProjector2D.inl are privately archived (v250 filenames). Source residual starts at the measured ray and subtracts every pixel product sequentially; first summing W-times-x and then subtracting changes float32 arithmetic. source_ordered_sirt implements the explicit augmented sparse operator [I,-W] applied to [y;x], so the observation is accumulated first and each source-ordered negative weight follows. The normalization uses source-ordered sums and 1e-6 zero guard; backprojection accumulates detector-fast native rays and applies column normalization before adding the correction. Full bound geometry is relabeled from MATLAB to original detector-fast rays/row-major pixels without sorting ray entries, then reconstructed maps return in MATLAB pixel order.

An independent original first-two-step capture on measured channel29 records forward/backward ones, reconstruction1 and reconstruction2. The local preserved-order prototype matches every normalizer and both iterates bitwise, then matches all65025values of the independent original100-step result bitwise (33.31798189994879s for two duplicate channels). Initial reordered-CSR attempts failed: row sums differed0.001007080078125 and100-step relativeL2 1.5150936860663933e-7; private sirt_augmented_residual_trial.json and original arrays retain the latest proof, while the chronology is explicitly described here. Shipped public-data reconstruction of the first two distinct channels passed its existing tolerance checks in40.68s; an exact-array assertion is now also present and will be executed with the complete suite. Full128-channel fresh SIRT, quartz ROI and dictionary exactness remain under validation. Quartz reduction explicitly materializes the source channel/row/column contiguous layout before the original float32 mean.

**Original source queue completed.** All three source methods now completed naturally with source exit0 and controller9534.452722699963s: unmodified ADJUST1000, corrected ADJUST1000, original cJoint500. cJoint saved-file boundary runtime1741.3778158s; corrected boundary3402.2312134s and original4385.1106532s are recomputed using integer nanosecond file times. Earlier floating-second subtraction3402.2312133312225 is the same boundary at lower timestamp precision. The controller subsequently reports error because its old local comparator rejected the intentionally interrupted hybrid run, exit1 in1.111s. That downstream local error is not a failed original run. The owned relay stopped naturally and released CUDA. convert_oracles.py creates identified upstream notebook outputs and hash manifest only; it does not promote a baseline or set metric thresholds.

**Generic environment independence.** Only the validated generic libblas.so.3 (SHA256 8ba4a98f44d763c2e648755ef452025919e28633339f23ff605044de3483a25d), libopenblasp-r0.3.26.so (bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e), its soname symlink, copyright and provenance were copied into the shared maintainer generic-blas environment. The package is Ubuntu24.04 libopenblas0-pthread0.3.26+ds-1ubuntu0.1, package hash7dc3b4384c02aecb87eb8b70fa26c5843a08af242f4638aa4b36922bdc4f5b04. Dependencies are system libm, libgfortran5, libc, libgcc_s and loader; no LAPACK/domain/Octave package is required by the cleaned worker. WSL's default reference libblas lacks the required OpenBLAS implementation. src/linear_algebra.py and src/blas_worker.py resolve source-relative worker paths and generic environment variables, contain no private oracle path, and implement binary no-pickle IPC with startup identity check, timeouts, descriptive process errors and explicit close. The shipped modules using this separate shared math-library environment reproduce the independent deep-state SPG bitwise, wall1.001278799958527s, worker exit0. Frozen upstream_scalar_roles.npz SHA256 79cc902b8a30ef752ce8afbfec57d8b911c6600bd6148a8da0881c97561a6c9b contains original independently captured s/y products and cubic scalars; consumer test/comparing_test/test_scalar_call_roles.py passed both original scalar cases in0.42s. SourceBlas is an explicit context; its eventual standard-entry integration remains pending full validation.

**Latest fixed-state generic BLAS proof; full gate still unresolved.** The sparse-only CUDA diagnostic was intentionally terminated after iteration100 failed the unchanged strict map tolerance (relativeL2 9.55077198007715e-10, maximum absolute 2.569112998729306e-9). The dedicated worker exited4294967295 after432.8524664s; private hybrid_interruption.json identifies this as an interrupted diagnostic, never a completed1000-step run. Previous all-GPU1000-step maps remain unacceptable: weakest-material NCC against corrected source is only0.4828 before and0.5133 after sparse ordering. Similar forward residuals do not justify broadening component thresholds.

An independent same-input dense capture, oracle_debug10_dense.m, separated every product in the real original step10 subproblem. Local F-output NumPy reproduces spectra, prediction and spectral adjoint bitwise. Its long-inner gradient differs6.366462912410498e-11 absolute. Windows OpenBLAS0.3.31/0.3.30 and a privately extracted, never installed0.3.26 wheel all show that discrepancy. Linux0.3.26 direct ctypes DGEMM, both NN and TN and1/2/4threads, reproduces the original gradient bitwise. Thus a matching version string alone is insufficient; platform/build and actual primitive roles matter. Evidence: dense_product_call_role_comparison.json, linux_blas_products.json, linux_windows_source_blas_comparison.json. The original environment is Ubuntu24.04 OpenBLAS0.3.26, Haswell, MAX_THREADS64, four threads. The private Windows wheel hash is fda16c88af8790a989d6c040f55baf4e1724cb38965f6da7b1afebafa54231cf; it remains rejected as a solution.

A private persistent Python stdlib/ctypes binary IPC prototype now delegates only generic DGEMM/DDOT/DSYRK primitives to this Linux BLAS, with all task objectives, projections and inverse updates still in local source. No upstream algorithm or private oracle relay is imported. A separate original scalar capture establishes that self product s-transpose-s uses DSYRK, not DDOT; the difference is one ULP in the BB numerator. More directly, the first deep-state backtrack exposed local multiplication/division association differing from the source cubic formula. The implementation now preserves step minus step times the parenthesized numerator/denominator ratio. Before that correction, the third trial/final weights differed2.220446049250313e-16 despite exact preceding evaluations; this deliberately failed exact-equality diagnostic is not silently counted as success.

After the explicit ratio-order correction and source primitive selection, all three original deep-state SPG trials, objectives, gradients and final weights match **bitwise**. Both a full generic-IPC prototype and a selective variant pass; the latter uses local F-output products for inner dimensions below4096 and generic Linux products for longer contractions, a dimension-only rule. Evidence: generic_blas_ipc_fixed_state_proof.json and selective_blas_ipc_fixed_state_proof.json; worker startup0.10s, long-gradient transfer/product about0.035s, explicit close exit0. These are fixed-state proofs, not full benchmark parity. No shared package was installed, no threshold changed and no further full run is accepted yet. Runtime integration, complete100-prefix and1000-step validation remain pending.

The corrected original full1000-step ADJUST completed: best observation residual0.153593345270713, independent12-view residual0.15461163941112, map nonnegativity violation0, map sum excess2.22e-16, dictionary row error9.160097510507725e-7 and column excess2.22e-16. The externally measured corrected-method boundary runtime is3402.2312133312225s; nested source tic makes its raw runtime field invalid. Original cJoint500 remains independently running and is not declared finished.

**Original numeric call-role extraction.** The complete dominant-axis-order-corrected all-GPU trial completed exit0 in497.76509859994985s; its intermediate map discrepancy remains0.00601724at300 and0.230883at400, so CSR ordering alone is insufficient. No full parity is accepted. Private ordered_status.json, local_ordered_exact_dictionary.npz, local_ordered_states and ordered_checkpoint_trajectory.json retain this separate diagnostic configuration.

The coordinator independently identified GNU Octave8.4's Frobenius implementation in release-8-4-0/liboctave/numeric/oct-norm.cc, SHA256 **8123ca24153519263a312ac77df293db46a800b21b462cb859870e6179b1c348**. It uses serial scaled accumulation, not BLAS or parallel sum-of-squares. The source is privately archived as root-octave-oct-norm.cc/.h. Local explicit extraction groups record maxima and uses sequential NumPy cumsum without fitting any numeric constant. A fresh independent Octave run oracle_norm_cases.m exited0 and captured ten zero, known, random, extreme-scale and SPG matrix cases; all ten local norms match bitwise. The complete154200x65bound observation norm independently matches **1443.5211740353877** bitwise when recomputed locally (0.0976614s). Native F-vector SPG dot products agree to at most1.14e-13absolute (2.2e-16relative); the source sequential L1 sum agrees bitwise. Private norm_convention_independent_proof.json identifies results. This is a mathematical numeric-convention extraction, never a captured-normalization constant in the solver.

Frozen fixture upstream_reduction_conventions.npz SHA256 **94ace5dd936fe6044a5f8c9fcf67a84f2570648e732ca86d1ace8bdd9f766495** contains only these original-case arrays/measurements; its documented consumer is test/comparing_test/test_reduction_conventions.py. The fixture was captured before implementation. Initial test collection from the wrong directory failed to find src, then the correct task-directory test failed because the new norm function did not yet exist. After extraction, five non-SIRT frozen stage/operator/reduction tests passed in0.37s. The additional small CUDA-sparse boundary/four-material two-step test passed with the unchanged rtol2e-10/atol2e-11 (three reduction/hybrid tests in1.61s); an initial fixture-key typo was corrected without altering arrays or tolerance. Known warning: PyTorch CSR support is marked beta.

The original sparse relay actually reads serialized F-data as a C-contiguous host matrix, runs only sparse W/A or W-transpose products on CUDA, serializes C output by its transpose, and returns F-contiguous matrices through Octave fread. Original dense factor arithmetic is CPU/OpenBLAS. The new local cuda_sparse mode explicitly preserves these roles and memory layouts without importing the private relay. Independent real common-state SPG10 now has a bitwise initial objective/normalization, relative gradient differences about1e-15, same three evaluations and accepted weights within2.22e-16; common_state_after_reduction_extraction.json preserves this proof. Full hybrid benchmark reproduction is still pending and the standard default has not yet been declared validated.

**CSR ordering correction under validation.** Later original checkpoints show map relativeL2 discrepancy growing from4.65e-10at100 to1.72e-7at200 and0.00585464at300; mixture weights differ0.00261777at300. These failures remain in full_checkpoint_trajectory.json. Before attributing this to unavoidable nonlinear sensitivity, the concrete sparse-reduction mismatch was corrected. Archived source FanFlatBeamLineKernelProjector2D.inl traverses rows then columns for vertical rays and columns then rows for horizontal rays. A first attempt to sort every ray by C-pixel index failed16.1%of frozen indices; source review corrected the rule to dominant-axis traversal. Local preserve_source_pixel_order now derives that order from geometry without loading any reference asset. Independent nine-angle/all257detector/full255image fixture pointers, indices and weights agree **exactly**; the frozen forward/adjoint test passes, exit0,0.30s. This changes storage/reduction order only, not weights or the physical model. Existing in-flight full local runs imported the earlier order and must be labeled pre-correction; full post-correction reproduction remains pending. No metric threshold has been relaxed.

The local controlled full 1000-iteration run using the original captured four-material dictionary completed with exit0 in515.8631747000618 wall seconds (private overlap_status.json and overlap_controlled_full.log). Its iteration1000 residual is0.153934695; this is not necessarily its selected best iterate. The actual standard CLI subsequently started independently, recomputing the full public-data calibration. Full corrected-source comparison remains pending; a local completion alone does not pass that gate.

The stricter100-iteration checkpoint diagnostic remains a recorded investigation, not relaxed acceptance. Relative map L2 discrepancy is4.6472765e-10 (maximum absolute1.2500e-9); relative mixture-weight discrepancy is1.8145e-11. The preselected elementwise rtol2e-10/atol2e-11 fails on some near-zero entries. Common-state SPG checks reproduce the original accepted step to about2.22e-16 and trace the first failure to accumulated earlier floating-point drift. Private four_prefix100_comparison.json and debug10_* preserve both successes and failures.

Private sparse_reduction_probe.json independently establishes that the complete original and cleaned matrices have exactly equal weights after canonical sorting. The original column permutation leaves CSR column indices unsorted; the cleaned COO-to-CSR path sorts them, changing floating-point reduction order. Original/local projection differences on an identical dense input are at most5.684e-14; C/F memory layout alone also changes products by5.684e-14; CPU SciPy versus GPU reaches2.132e-13. Selected-ray independent math.fsum dots bound GPU error by1.421e-14 and SciPy error by4.263e-14. Neither matrix nor normalization has been modified to hide the drift.

The independent standalone Judge now also checks public-data observation consistency on fixed views0,50,...,550, all257detectors and all65selected bins. Its upper boundary rule was fixed before measuring the full cleaned result as1.1times the independently validated source residual on those rays. The matched ray helper independently re-evaluates the selected float32 discretization and imports no solver. It matches the frozen compiled-source forward fixture at rtol2e-12/atol2e-11. An initial double-precision Siddon attempt failed the preselected rough-map comparison (maximum discrepancy about.8696): exact-double intersections differ from the source's float32/repeated-coordinate convention. The acceptance operator was corrected to the selected convention rather than relaxing tolerance; the separate Siddon helper remains only for independent analytic axis/diagonal tests. This is a discretization distinction, not a claim that the analytic Siddon method is incorrect.

Judge contract tests additionally reject a missing forward boundary and a reference-matching result with failed observation consistency. Command `python -m pytest test/unit_test/test_judge_contract.py -q` in the pinned research runtime completed **22passed in3.98s**, exit0. Its isolated contract fixtures monkeypatch the forward metric deliberately; the separate original-forward fixture validates the actual numerical projector. Complete benchmark forward measurements and final threshold JSON remain pending.

**IN PROGRESS.** The current benchmark follows the paper section6.6 four-material experiment: tungsten, gold, lead, and observation-derived quartz. The release demonstration's 43-atom basis was rejected after independent singular-value and rerun checks: numerical rank25/43 and condition5.1441533135717734e17 made its coefficient coordinates unsuitable for stable evaluation. This is an explicit scientific configuration change, not unchanged reproduction of the 43-atom demonstration. No views, pixels, channels, or iterations were reduced.

The current public dictionary contains original CSV zero-based rows31,36,39, in W74/Au79/Pb82 order, shape(1,3,128). All photon observations, dark/flat counts, geometry, and energy arrays are unchanged. Current raw_data.npz SHA256 is **3e5eda5c73d7c8d46b1cb8d25dbdddbad4fed6a7fad315a6bf4a353046ad8895**. Outputs are reconstruction(1,4,255,255), spectra(1,4,65), and dictionary_weights(1,4,4), all float64. The old public NPZ is retained only privately as legacy43_raw_data.npz.

Official NIST elemental tables from pml.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z74.html, z79.html, and z82.html were independently linearly interpolated onto20.90+1.10*c keV. Agreement with the three released rows is relative about4.1e-16. The source filename's20.35and161.15are bin edges, **not** an energy-grid mismatch. verify_nist_four_atoms.py, archived HTML, and four_dictionary_identifiability.json provide evidence. The four-row scaled basis has rank4 and condition368.3236925959782; singular values are1.048310111736249,.2684558241321183,.16301754297337387,.002846165296475136.

The quartz basis is defined publicly by the complete100-step float32 SIRT reconstruction of all128channels followed by the mean over zero-based image slices129:140 in both dimensions. This author-selected observation ROI defines coefficient coordinates; it is not independently known instrument calibration or material truth. No estimated spectrum is supplied as privileged calibration. There is no actual image/composition truth.

The configuration was frozen before a four-basis full run in private four_basis_frozen_contract.json. Detached driver detached_driver.py was released by root's gpu_approved.flag and is running oracle_four_full.m: independent unmodified1000-step ADJUST, corrected1000-step ADJUST, and original500-step cJoint, followed by local_full_validation.py. These full four-basis runs, full numerical/physical stability, reference/threshold selection, actual CLI/Judge execution, complete notebook execution/visual review, and final consistency checks remain **unverified**. No threshold has been relaxed and no pending gate is treated as passed.

Current-basis independent stage capture passed: capture_four_stages.py selects204real measured rays and625central pixel columns from the complete independently matched matrix, promotes its exact float32 weights to float64 as the full inverse does, and uses the four-row original dictionary. oracle_four_stages.m executes the unchanged original objective/gradient functions, original two-step ADJUST, and separately guard-corrected two-step ADJUST. WSL Octave exit0, runtime1.5627633000258356s; warnings are vec.m builtin shadowing and the WSL localhost proxy notice. Complete command/log/environment records are private oracle_four_stages_execution.json and oracle_four_stages.log. The initial capture accidentally left dense W in float32; Octave's mixed products then reduced precision, which was corrected before freezing without changing the preselected tolerance rtol2e-10/atol2e-11. test/comparing_test/test_upstream_parity.py::test_four_material_original_stage_fixture passed0.65s. Fixture upstream_four_stages.npz SHA256 **64b11cdf84589d6685d10ad0936b6144abee28ca3b58ed28bead474a553eea72**; its consumer checks identical initialization, both objectives and gradients, and corrected two-step maps/spectra/weights. This is stage evidence, not a full-data run.

The legacy43 original1000-step run completed and saved oracle_adjust_full.mat. Its queued corrected43 stage was intentionally stopped at that boundary to release GPU resources; queue exit1 records this interruption rather than failure of the completed original run. Existing42-atom stage fixtures validate generic operations only and do not establish bound four-basis full-pipeline parity. Previously validated full line-operator/SIRT checks retain unchanged observations and geometry.

The current four-basis unmodified1000-step run completed and saved oracle_adjust_four.mat at2026-10-18T16:26:10UTC, best observation residual0.15308189858787355. Its best R is nonnegative, has maximum row-sum error about6.5e-7 and column sums<=1; its maps are in[0,1] with pixel sum<=1. Thus current notebooks/plans do not claim this run returned the historical43-atom invalid mixture. The corrected1000-step run is now active. Raw captured runtime_seconds=4.399755001068115 is invalid as full-method runtime: source-internal tic calls overwrite the unhandled outer tic. Conversion uses externally observable method-boundary file timestamps, with original boundaries oracle_initial_four.mat -> oracle_adjust_four.mat, corrected boundaries oracle_adjust_four.mat -> oracle_corrected_four.mat, and cJoint boundaries oracle_corrected_four.mat -> oracle_cjoint_full.mat. This wall time includes setup/boundary serialization and is labeled accordingly; source cumulative timing lines and complete detached parent monotonic time are independent supporting evidence. Raw MAT fields are retained, never silently overwritten.

Full-geometry exact-T20-step diagnostics preserve a deliberately strict preselected elementwise rtol2e-10/atol2e-11 failure from the near-zero R coordinate at iteration10. At20, map relativeL2 is8.588878615836252e-11 and R relativeL2 is1.3618863407756037e-11, but near-zero map entries still fail that absolute tolerance; no tolerance was changed. Private four_prefix_checkpoint_comparison.json contains every coordinate-test result. Independent source MinConf tracing at step10, from the same original-state9 WA/B/T/R, gives the same three objective evaluations and accepted path; local CPU/CUDA final R differs from source by2.22e-16/1.67e-16. That source subproblem also reconstructs the actual originalR10 to2.22e-16. Reconstructing the actual local-state9 subproblem gives its localR10 to6.94e-18, also with three evaluations: the first strict-coordinate failure propagates prior inputs, including WA max difference4.788121010790292e-10 and lifted B max difference2.6629809468659005e-12. Original Q9 reconstructed forward from checkpoints1..9 and backward from original Q20 differs by relative1.9361325745205457e-16; lifted-data uncertainty is1.0658141036401502e-16. These checks isolate the iteration10 input propagation; they do not certify later stability.

Independent compensated summation with math.fsum of all10,023,000 squared observations gives norm1443.521174034858, identical to Torch; original Octave gives1443.5211740353877 (relative3.6700581419385377e-13), NumPy gives1443.5211740348764 (relative1.2758571223048134e-14). Fixed-input original/local objective-gradient discrepancies are about3.7e-13 relative, consistent with this backend norm reduction. Standard n-term dot-product error envelopes gamma(n)=nu/(1-nu), with float64 u=1.1102230246251565e-16, are recorded independently of agreement. No norm was changed to imitate another backend, and this roundoff analysis does not authorize broad task thresholds. Scripts/data/logical states are indexed in debug10_input_precision.json, debug10_trace_comparison.json, debug10_perturbation.json and their companion private scripts. Complete1000-step parity remains the decisive pending stability gate.

## Historical 43-atom investigation record

The following sections preserve the earlier investigation and its then-current status. Their old43-atom data hash, fixture dimensions, exploratory results, queued-run descriptions, and unresolved public-basis concerns are historical evidence, not assertions about the newly frozen four-material benchmark. Source/dataset provenance and license notices remain applicable; current configuration and gate status are defined above.

### Historical status

**IN PROGRESS: mandatory full-source parity, final Judge thresholds, fresh CLI verification, and executed notebook review remain unresolved. This task is not yet declared cleaned.** The full unmodified ADJUST run is currently executing; the separately corrected original and original cJoint are queued. No missing measurement is presented as a success and no source reconstruction is relabeled actual truth.

The benchmark is deliberately research scale: 600 measured views, 257 detector positions, 128 acquired energy bins, 255 by 255 pixels, 100 SIRT updates for every acquired channel, and 1000 outer/two inner ADJUST updates on all 65 selected bins. The bound observations have not been replaced with a small phantom or an easier model. Small fixture cases below validate individual stages only.

## Selection, source, and exact revisions

Selected method: M. T. Zeegers, A. Kadu, T. van Leeuwen, and K. J. Batenburg, ADJUST: A Dictionary-Based Joint Reconstruction and Unmixing Method for Spectral Tomography, Inverse Problems (2022), DOI https://doi.org/10.1088/1361-6420/ac932e; arXiv:2112.11406v2, 4 October2022. The downloaded paper version is archived as ADJUST_paper.pdf and searchable ADJUST_paper.txt. These are private maintainer material, not packaged solver input.

Original repository: https://github.com/mzeegers/ADJUST, remote revision **7751a223ef33aa3cf6e8583d286b331b1729f274**, verified through the remote service during this session. This is an actually verified remote revision, not an inferred local tracking branch. Download ZIP SHA256: 51e5ab5286c70182e5f9e2691d9262d705f1e3498e3878a94d35926286e9c055. Original ADJUST.m SHA256: 5e4a8ce2291a8db49f9b4c19cc0cac72dda779c55d920307a18c521c39b08908. Original example8_microCTData.m SHA256: 164dfa38848de9a9eb43294d0d30c9c802e403c2e196fc486ec86d108c2c71f1. Original cJoint.m SHA256: 47b7eb32998ab3866a232ab2f54323f038f97df8289b7c7c52fffd2c11703474.

Selected discretization is the paper's generic W interface instantiated by an exactly matched CPU line_fanflat matrix, not a numerical reproduction of the demonstration's CUDA projector/backprojector. ASTRA 2.5.0 supplies the isolated compiled operator/SIRT oracle only. Its line-kernel source is privately archived as FanFlatBeamLineKernelProjector2D.inl. The local line integrator evaluates the same pixel intersections and float32 conventions explicitly. Domain libraries are absent from the cleaned runtime.

MinConf source ZIP and authors' distribution page are archived from https://www.cs.ubc.ca/~schmidtm/Software/minConf.zip and the corresponding Software page. Their attribution is Mark Schmidt, Ewout van den Berg, Michael P. Friedlander, and Kevin Murphy. No explicit license file was found in this distribution; no assumption of MIT licensing for MinConf is made. The cleaned SPG code states the mathematics independently; the unchanged MinConf implementation is retained only in the private original-run environment.

The selected method and measurement contract were fixed in private selection_contract.md before extraction. The initial candidate XRF repository was rejected for inaccessible research data (SharePoint HTTP403) rather than replacing it with a synthetic demonstration; see candidate_screening.md.

## Real-data identity, legal attribution, and conversion

Measured dataset: Jonathan Sittner, Jose Ricardo Da Assuncao Godinho, Axel Renno, Veerle Cnudde, Marijn Boone, Thomas de Schryver, Denis van Loo, Margarita Merkulova, Antti Roine, and Jussi Liipo, **Data for: Spectral X-ray Computed Micro Tomography: 3-dimensional chemical imaging**, RODARE DOI https://doi.org/10.14278/rodare.1627. Metadata retrieved from https://rodare.hzdr.de/api/records/1627 is archived as rodare1627.json. License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/. Attribution and change description are provided here and in the README. Original creators do not endorse this cleaned benchmark.

Downloaded Spectral_CT_W_Au_Pb.zip size is 34,265,217 bytes. Published MD5 and measured MD5 agree: fe1a0c464ca5af472edce0f2b965696c. SHA256 is a2cf5883eb09325a4006bde7683fd869f1ccaa3a852b1856b51ddc7e60d2a04c. All 600 uint16 scan TIFFs and the original uint16 dark/flat TIFFs are retained privately. The packaged NPZ preserves their numeric count values, adds a leading batch axis, and adds supplied geometry/energy and the original dictionary table; it is not resampled, cropped, denoised, or synthesized. Original recon TIFFs are historical third-party reconstructions and are neither public inputs nor actual truth.

Original NISTequivalentofRealData_MatrixTXray128chan_42_MinEn20.35_MaxEn161.15.csv SHA256: 800f5042995a88454dfc6f5a088768ff3f0d46c17b87f87e3d735461139a63b5. Dark TIFF SHA256: da9d7a510b29d8befe722c10afcff8115b54331826d2ac6e1ce3a0e06cf320b9. Flat TIFF SHA256: e7b8b605fdf0e241a1477fdbbbccb1d49d63a72f5d3623134d422f20d16e9948. Energy-bin source SHA256: 197901d461fc32cd17a33bd9881dc2fe102496c1afac4896292b42d52ffe7d06. All other TIFF hashes are enumerated in the private download_manifest_sha256.json, rather than embedding 600 redundant rows here.

Current packaged data/raw_data.npz SHA256: **3e5eda5c73d7c8d46b1cb8d25dbdddbad4fed6a7fad315a6bf4a353046ad8895**. Solver-visible keys are exactly counts (1,600,128,257) uint16, dark/flat (1,128,257) uint16, dictionary (1,3,128) float64, angles (1,600) float64 radians, energies (1,128) float64 keV. The three dictionary rows are the fixed source W74, Au79 and Pb82 rows; the fourth quartz atom must be freshly estimated using the public ROI protocol. The earlier 42-row archive had SHA256 ac9efe8ae8e85dfc5651cec3cc63d3e7db4020176abe61acacf47cc56000794b and is historical, not the selected input. No reconstructed channel, quartz spectrum, hidden seed, object label, or target-derived calibration is released. There is no independently measured image or composition truth for this slice. A final validated original-method result must therefore be stored as baseline_reference.npz, never ground_truth.npz.

## Scientifically material discrepancies and selected resolution

1. **Paper versus release constraints.** The paper's dictionary-weight formulation has nonnegative row and column sums bounded by one. The supplied projectTwoSimplexConstr instead applies row sums equal to one and column sums <=one. The selected benchmark is explicitly the **reference-release row-equality variant**, not a silent claim of reproducing the paper's inequality variant. README, approach, design, and notebook must state this distinction.
2. **Projection is not orthogonal.** Independently solving the small constrained Euclidean QP using SciPy SLSQP shows that the source's alternating feasibility routine can differ from the nearest feasible point by 0.4000001568063687. For the frozen example, QP objective is 2.975000000000009 while alternating objective is 3.594999655026038. Therefore no exact proximal-map or global-convergence claim is made. independent_projection_checks.json and a mathematical unit test retain this counterexample. The alternating rule is preserved rather than silently replaced with a different optimizer.
3. **Feasibility guard bug requires an explicit correction.** Original projectTwoSimplexConstr skips all projection when the total proposed weight sum is <one, leaving negative entries and invalid row sums. This is independently disproved by an all-negative two-row example. A historical 43-atom faithful local 1000-step diagnostic enters this branch and returns row sums approximately [-.09545,-.003276,.109497,.980752], with negative R and best data residual .184813. The minimal selected correction removes only that guard; row/column projection operations, 1e-6 tolerance, and 10000-pass limit remain. Private correction.diff and oracle_corrected/ADJUST.m identify the change. Original unmodified and corrected full-source results are recorded separately; the corrected result cannot be labeled unchanged source reproduction.
4. **Operator convention.** The original CUDA demo uses approximate pixel-driven backprojection. A coordinator independent 32-square, 30-angle, 65-detector probe measured CUDA normalized dot errors .0113, .0303, and .00328 versus CPU line errors around 1e-9. This is a source convention, not a general ASTRA defect claim. The chosen CPU line W and its exact transpose provide a mathematically matched gradient while retaining full acquisition dimensions and physical geometry. Private coordinator/astra_adjoint_probe.py/.json record that probe.
5. **Geometry source/settings differences.** Original reconstruction source uses SOD=57.999001 mm, SDD=883.081116 mm, detector pitch=.80 mm, and center257/2-127.379570. Supplied settings report SDD=883.081305 and rotation axis127.379796. This task preserves source constants and identifies the discrepancy. Angles include both0 and2*pi over600samples, as in the source. The small difference is not silently optimized against a hidden target.
6. **Energy/channel counts.** MATLAB30:94 is65channels despite a source comment saying64. All128channels are first self-calibrated. An earlier suspicion of a grid mismatch was disproved by official NIST interpolation:20.35..161.15keV are bin edges, while the dictionary and measured bin midpoints agree. The .015 dictionary factor is a nominal basis scale and does not establish absolute material mass density.
7. **MATLAB comparison dependency.** RU/UR rely on proprietary nnmf upstream. This environment has GNU Octave, not MATLAB Statistics Toolbox. The local RU/UR groups are full-data mathematical comparison ports with explicit multiplicative/NNLS updates, not verified replicas of MATLAB's hidden nnmf. This limitation is labeled wherever their results are displayed. Core ADJUST and cJoint use independently executed original code rather than this exception.

8. **Source pseudo-truth is another reconstruction.** example8 sets T_gt=T1(:,30:94), exactly the same array as T, then invokes ADJUST again and names that reconstruction ADJUST-true/strProb.A,F. This second random run is not actual truth and supplies no known-material dictionary. The notebook intentionally relabels this group as a same-dictionary independent-initialization stability repeat, actively recomputing the complete corrected algorithm with seed11 as well as seed10. This preserves the source's repeated-run comparison while rejecting its misleading truth label. The original demo's global random stream after proprietary nnmf cannot be reconstructed here; the two explicit seeds are a declared reproducible convention, not a claim of matching that hidden state.

## Equation/stage/function/test traceability

| Mathematical or algorithm stage | Original source | Local executable implementation | Independent validation |
|---|---|---|---|
| Beer-Lambert log transmission, dark/flat subtraction, angle reversal | examples/example8_microCTData.m data preparation | preprocessing.correct_counts | unsigned/log/order tests; original oracle_preprocessing.npz |
| Generic fan-beam W with pixel-length weights | example8 geometry; isolated ASTRA CPU line projector | physics_model.geometry_vectors/build_system_matrix | complete46,901,466-weight equality; nine-angle/full-image frozen products; central-ray analytic test |
| Exact transpose and spatial gradient | ADJUST.m misfitA; paper squared-Frobenius factorization | physics_model.adjoint; solvers.spatial_objective | independent dot identity and finite differences; original objective/gradient fixture |
| 128-channel100-step SIRT calibration | example8 SIRT; ASTRA2.5.0 SirtAlgorithm.cpp/DataProjectorPolicies.inl | physics_model.sirt/source_ordered_sirt | bitwise equality for all8,323,200fresh128-channel100-step values; first-step/normalizer exact checks |
| Quartz ROI and augmented dictionary | example8 rows/cols130:140, .015*NIST | preprocessing.calibrated_dictionary | bitwise equality for fresh128quartz and all260selected dictionary values; ROI known-array test |
| ADJUST Y~WART, objective/||Y||, R gradient | ADJUST.m misfitR; factorized spectral model | solvers.spectral_objective | original scalar/gradient fixture and central finite differences |
| A nonnegative row-sum<=1 constraint | ADJUST.m projSimplexA | project_simplex_columns/project_material_maps | independent constrained QP and analytic cases |
| R row-equality/column-inequality feasibility | ADJUST.m projectTwoSimplexConstr, with disclosed guard correction | project_dictionary_weights | negative-guard counterexample; QP nonorthogonality evidence; corrected-original two-step fixture |
| BB1, nonmonotone SPG, cubic backtracking | original MinConf_SPG invoked by ADJUST | spectral_projected_gradient/_cubic_step | original/corrected stage runs; full parity pending |
| Lifted residual, dynamic rho, stopping, coherent best iterate | original ADJUST.m main loop | solvers.adjust | independent original two-step; full1000 pending |
| Unconstrained spectral joint baseline | original cJoint.m | solvers.classical_joint | complete original500-step exit0; active cleaned notebook comparison pending |
| RU/UR regularized reconstruction and factorization | original RU.m/UR.m plus nnmf | tikhonov_cg/factor_nonnegative/sequential_comparison | independent dense linear solution and scipy.optimize.nnls; mathematical port only |
| Joint component identity and output metrics | declared benchmark contract | independent evaluation/judge.py | known numerical cases, single-permutation consistency, poor/malformed/evaluator-error tests |

## Oracle environment, commands, and independent fixtures

Private archive root is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research10/task03`; all paths in this section are relative to it. It contains original source/downloads/paper/TIFFs, capture scripts, original logs, isolated dependencies, and full intermediate matrices. These files are excluded from the final task directory and are never imported by cleaned code/tests/Judge/notebook.

Original operator/SIRT: isolated oracle-env Python with ASTRA2.5.0 wheel; compatible numerical/DLL dependencies provided from the validated research runtime without installing ASTRA there. `oracle_prepare.py` records generation of the complete fan matrix and independent CPU SIRT, one energy bin at a time. Run command: `oracle-env/Scripts/python.exe oracle_prepare.py`; log oracle_prepare.log. All128channels100iterations completed, summed per-channel1265.599482599995s. oracle_sirt.npz stores independent channel images/quartz/times. Oracle SIRT is zero-init, no positivity, relaxation1, original float32 compiled arithmetic, exact same measured observations. This is a full measured-data run, not a cached upstream notebook.

Original algorithm interpreter: WSL Ubuntu24.04, GNU Octave8.4.0, private OpenBLAS0.3.26 pthread distribution with OPENBLAS_NUM_THREADS=4 and OMP_NUM_THREADS=4. It does not change global BLAS alternatives. MinConf and the original ADJUST/cJoint are private dependencies. `oracle_stages.m` executes unchanged original objectives, gradients, initial states, and two updates. `oracle_corrected_stages.m` independently executes the minimally corrected original function. Both exited0. Their outputs are stage_outputs.mat and corrected_stage_outputs.mat. Two-step local matching uses rtol2e-11/atol2e-12 fixed before final comparison.

Full-source queue command: `research-runtime/Scripts/python.exe launch_oracle_full.py`, invoking `wsl -d Ubuntu-24.04 -- env LD_LIBRARY_PATH=<private-openblas> OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 octave-cli --quiet <private>/oracle_full.m`. It executes original1000ADJUST, corrected-original1000ADJUST, then original500cJoint with rng(10,'twister'). All selected four-basis stages completed naturally with source exit0: original1000, corrected1000 and cJoint500. The original/corrected/cJoint externally observed boundary times are4385.1106532/3402.2312134/1741.3778158seconds. The detached controller took9534.452722699963seconds; its later error is the separate rejected old local comparator, not an original-code failure. detached_status.json, oracle_four_runtime_provenance.json and original logs are authoritative; raw nested-tic MAT runtime fields are invalid.

To accelerate only general sparse products, `oracle_sparse_server.py` and SparseOracle.m pass Wv/W'v to torch sparse float64. The service imports no cleaned code and performs no reconstruction step, projection, objective, or stopping logic. Independent CPU/GPU oracle comparison gave forward relative error about5e-16, transpose8e-16, and two-outer map error2.339e-13. These comparisons are oracle_gpu_cpu_stage.mat and oracle_gpu_cpu_stage_openblas4_rerun.log. An earlier cross-filesystem request.flag PermissionError is preserved in oracle_gpu_cpu_stage_openblas4.log; the service retries transient file access rather than treating that failed run as evidence. Service calls/device/runtime are recorded in oracle_sparse_service.json on shutdown.

Full original/local line-matrix parity is **exact**, shape(154200,65025),46,901,466nonzeros,0differing entries, maxabs0. See line_full_parity.json. The first attempted local rounding convention produced discrepancies near axial rays; initial_geometry_rounding_discrepancy.npz preserves the failure. Rounding the detector basis before its start point and repeated float32 accumulation resolved it; tolerances were not enlarged to hide the error.

Frozen fixtures and consumers:

| File under test/fixtures | Numeric origin and consumer | SHA256 |
|---|---|---|
| upstream_line_operator.npz | Nine angles of original compiled full255-square W and random forward/transpose products; comparing_test/test_upstream_parity.py | 598f84c53e14c6c0ca2f0618fd29368474cae33a2fc01933dec246942af93003 |
| upstream_sirt.npz | Original compiled first2SIRT images and full128quartz; calibration parity checks | 699a2d2edc3b7787d1bcf7c037c3447caf9eb5aed13e228babf8bf3ab0bc9326 |
| upstream_stages.npz | Independently executed original small-stage objective/gradients/initial states, measured-data subset; comparing_test/test_upstream_parity.py | 67ef042b4cae9fc1193c2624df98716fab6a3709429e049f32718a908c064caf |
| upstream_corrected_stages.npz | Independent corrected-original two-step A/F/R, same measured stage inputs; comparing_test/test_upstream_parity.py | b2ea387596e310c3d188a7d5c19ab2b6d1f61297c5a4ed30554b45c87cb79d36 |
| upstream_four_stages.npz | Independently executed current four-basis original/corrected objectives, gradients, initialization and two updates; comparing_test/test_upstream_parity.py | 64b11cdf84589d6685d10ad0936b6144abee28ca3b58ed28bead474a553eea72 |
| upstream_reduction_conventions.npz | Independently executed Octave zero, known, random, extreme-scale and actual-SPG reductions; comparing_test/test_reduction_conventions.py | 94ace5dd936fe6044a5f8c9fcf67a84f2570648e732ca86d1ace8bdd9f766495 |
| upstream_scalar_roles.npz | Original deep-state self/cross products, BB numerator/denominator and cubic scalars; comparing_test/test_scalar_call_roles.py | 79cc902b8a30ef752ce8afbfec57d8b911c6600bd6148a8da0881c97561a6c9b |

Upstream fixtures were captured from original functions, not generated by the cleaned functions under test. Stage tests were defined/captured before extracting their corresponding inverse implementation. Full original execution did not precede all local implementation: it remained running while extraction proceeded. This chronological deviation from the strict upstream-first workflow is explicit; completion still requires the pending independent full pipeline run rather than retroactively calling local fixtures upstream evidence.

## Cleaned environment and measured results so far

Validated research interpreter is Python3.11.11 on Windows10.0.26200, NumPy2.4.6, SciPy1.17.1, PyTorch2.9.1+cu128, Matplotlib3.11.1. CUDA runtime12.8, NVIDIA driver610.88, GeForceRTX4080SUPER with17170956288bytes nominal device memory. Task requirements list only runtime packages; pytest/Jupyter are shared development tooling. No task-specific package upgrade or domain solver is required.

Historical diagnostic full local SIRT took8.309983000042848s on CUDA; relativeL2 against the independent complete original run is5.494071615430585e-7, maxabs1.341104507446289e-7, full quartz maxabs9.313225746154785e-10. sirt_full_parity.json records unrounded measurements. CPU local stage/analytic operations pass; full cleaned CPU benchmark has not yet been timed/executed. The full compiled original SIRT did run on CPU.

Private local_full_first.npz/.log is the faithful release-guard exploration, not the selected corrected output. Private local_corrected_first.npz/.log is a complete1000-step corrected exploratory run, ADJUST time259.7s, best residual about.143068, final residual about.154845, weight rows1and minimum0. corrected_material_maps.png was inspected: three separated grain maps and broad matrix background, with residual streak/ring artifacts. This is qualitative consistency, not image truth. These exploratory runs preceded the added per-iteration feasibility diagnostic; the final standard CLI and notebook must be rerun after all source changes.

No historical paper metric is relabeled this run's result. No baseline metrics or thresholds are finalized until full original/corrected/local runs are available. Current metrics.json explicitly remains pending. The final reference will be the corrected-original method result, not a local self-comparison. Nonzero numerical tolerance must reflect independent repeated executions; all quantities and all components must pass. Baseline-reference agreement cannot establish physical accuracy of this measured scan.

Test command executed: `python -m pytest test/ -q` using the validated interpreter, with TEMP/TMP redirected to private task03/temp because the global pytest temp directory has a Windows ACL denial. Result: **35passed in7.22s**, including geometry/adjoint/gradient tests, NNLS/CG, original/corrected-stage parity, known metric cases, strict Judge contracts, valid zero failure, missing/shape/dtype/complex/NaN/Inf rejection, actual-truth precedence, missing hidden asset error, invalid threshold rejection, report-parent creation, and report-write failure. The prior ACL-blocked test invocation is an environment failure, not numerical evidence.

Additional complete-geometry CPU validation: comparing_test/test_upstream_parity.py::test_original_cpu_sirt_on_full_measured_geometry reconstructs the first two actual measured channels for100updates with the complete600-view matrix. It passed in41.75s against frozen original compiled outputs, using predeclared relativeL2<2e-6and maxabs<5e-7. This consumes upstream_sirt.npz in the shipped tests. Together with the35fasttests,36tests have now passed individually; the final combined full suite still awaits pipeline tests.

Numerical reduction investigation: the unmodified-guard controlled local run was repeated with exactly the original128-channelSIRT-derived T, rather than fresh local quartz. Its trajectory still differed substantially after about100updates (local200=.219916094 versus original200about.2780), so quartz rounding is not the sole explanation. The original scalar objective evaluates norm(res,'fro') followed by squaring, whereas an initial local implementation used sum(res*res). The local implementation now preserves the source norm-then-square operation. This does not change the mathematical objective, but its effect under the extremely strict1e-12Armijo test must be measured rather than assumed negligible. All35fasttests passed again in6.70s after the change. A controlled full norm-convention rerun and corrected-original factor checkpoints are pending; no thresholds have been relaxed. The earlier exploratory runs remain labeled as using the initial scalar reduction.

Public-contract issue under review: the three-output contract includes dictionary weights for a43rd quartz atom estimated from the observations, but the README currently treats that spectrum as unknown without exposing the source's specific ROI/basis-estimation convention. Coefficients can also be nonidentifiable under a correlated dictionary even when spectra/maps are similar. The public-sufficiency and all-quantity evaluation gates therefore remain unresolved, not implicitly passed. Hidden quartz will not be released as public calibration, and the contract will not be silently changed or coefficient thresholds loosened merely to accommodate a divergence. Full independent corrected runs must first establish whether this ambiguity affects the selected measurement and method.

## Notebook and review gates

The required active six-section notebook is under construction. It must reconstruct all128channels and full1000ADJUST updates from raw data, actively run RU/UR/cJoint comparison groups, compare to separately identified independent original/corrected outputs, show map/spectral/dictionary/residual/metric/runtime figures, and clearly disclose the selected row-equality variant and guard correction. Executed nbconvert status, runtime, final metric table, and visual review are **unverified**. Default python3 kernel points to a different interpreter on this machine; verification must select imaging-101-research through the research-runtime JUPYTER_PATH. An optional archived preview is not reconstruction validation.

Plan/API review: design.md was written against shipped functions/defaults and distinguishes CLI, comparisons, Judge, tests, and private oracle. RU/UR iteration counts are20and100respectively; factor_nonnegative's actual argument is als_iterations=30. Equations/signs/normalization and all runtime paths require final review after full parity. Signature matching alone is not asserted as scientific verification.

## Completion checklist

| Mandatory gate | State | Evidence or unresolved work |
|---|---|---|
| Paper, exact remote revision, data hashes, stage mapping | passed | Source/data identity and table above; complete hashes private |
| Original and cleaned full methods on exact bundled observations | unverified | Full SIRT passed; original/corrected1000andcJoint500queued; final CLI pending |
| Independent upstream parity and applicable physics | unverified |35stage/unit/contract tests pass; full inverse parity pending |
| Constants/units/coordinates/precision preserved and tested | passed for operator/stages | Fullmatrixexact; analytic+gradienttests; full ADJUST acceptance pending |
| Standard CLI writes declared file and fresh Judge passes | unverified | Source present; full final run/reference thresholds pending |
| Judge rejects poor/malformed/evaluator failures | passed | Independent known-case contract tests |
| Public allowlist suffices, no hidden information leakage | provisional | All raw/calibration present; final public URL/array audit pending |
| README metadata/output contract, no metadata JSON | passed | README documents all6keys and3outputs |
| Minimal dependencies and runtime independent of domain packages | passed for tested paths | NumPy/SciPy/Torch/Matplotlib only; final full run pending |
| Core local mathematics and forbidden-shortcut audit | provisional | Local explicit operators/updates; final symbol audit pending |
| Full pytest suite | provisional |35passed; full-pipeline integration tests still to add |
| Active notebook executes and preserves comparisons | unverified | Notebook completion/nbconvert/visual review pending |
| Historical/upstream/fresh provenance and thresholds agree | unverified | Sources distinguished; final metric measurements pending |
| README/plans match mathematics/signatures/flow | provisional | Detailed plans drafted; final semantic review pending |
| Final inventory and license notices preserved | unverified | No standalone LICENSE/NOTICE; full preserved notices below; final cleanup pending |

## Preserved upstream notices

The task has no standalone LICENSE or NOTICE file, as requested. Substantial adaptations of the ADJUST method retain its MIT terms below. The explicit CPU line-kernel extraction preserves ASTRA's copyright and GPLv3-or-later terms, including the complete GPL text appended below. This is not a blanket claim that the independent MinConf distribution is MIT licensed. The measured dataset remains CC-BY-4.0 with the attribution and conversion description above.

### ADJUST MIT notice

```text
MIT License

Copyright (c) 2021 mzeegers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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

### ASTRA extraction notice

```text
Copyright: 2010-2022, imec Vision Lab, University of Antwerp
           2014-2022, CWI, Amsterdam
Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/
The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
```

### GNU General Public License version 3

```text
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

                            Preamble

  The GNU General Public License is a free, copyleft license for
software and other kinds of works.

  The licenses for most software and other practical works are designed
to take away your freedom to share and change the works.  By contrast,
the GNU General Public License is intended to guarantee your freedom to
share and change all versions of a program--to make sure it remains free
software for all its users.  We, the Free Software Foundation, use the
GNU General Public License for most of our software; it applies also to
any other work released this way by its authors.  You can apply it to
your programs, too.

  When we speak of free software, we are referring to freedom, not
price.  Our General Public Licenses are designed to make sure that you
have the freedom to distribute copies of free software (and charge for
them if you wish), that you receive source code or can get it if you
want it, that you can change the software or use pieces of it in new
free programs, and that you know you can do these things.

  To protect your rights, we need to prevent others from denying you
these rights or asking you to surrender the rights.  Therefore, you have
certain responsibilities if you distribute copies of the software, or if
you modify it: responsibilities to respect the freedom of others.

  For example, if you distribute copies of such a program, whether
gratis or for a fee, you must pass on to the recipients the same
freedoms that you received.  You must make sure that they, too, receive
or can get the source code.  And you must show them these terms so they
know their rights.

  Developers that use the GNU GPL protect your rights with two steps:
(1) assert copyright on the software, and (2) offer you this License
giving you legal permission to copy, distribute and/or modify it.

  For the developers' and authors' protection, the GPL clearly explains
that there is no warranty for this free software.  For both users' and
authors' sake, the GPL requires that modified versions be marked as
changed, so that their problems will not be attributed erroneously to
authors of previous versions.

  Some devices are designed to deny users access to install or run
modified versions of the software inside them, although the manufacturer
can do so.  This is fundamentally incompatible with the aim of
protecting users' freedom to change the software.  The systematic
pattern of such abuse occurs in the area of products for individuals to
use, which is precisely where it is most unacceptable.  Therefore, we
have designed this version of the GPL to prohibit the practice for those
products.  If such problems arise substantially in other domains, we
stand ready to extend this provision to those domains in future versions
of the GPL, as needed to protect the freedom of users.

  Finally, every program is threatened constantly by software patents.
States should not allow patents to restrict development and use of
software on general-purpose computers, but in those that do, we wish to
avoid the special danger that patents applied to a free program could
make it effectively proprietary.  To prevent this, the GPL assures that
patents cannot be used to render the program non-free.

  The precise terms and conditions for copying, distribution and
modification follow.

                       TERMS AND CONDITIONS

  0. Definitions.

  "This License" refers to version 3 of the GNU General Public License.

  "Copyright" also means copyright-like laws that apply to other kinds of
works, such as semiconductor masks.

  "The Program" refers to any copyrightable work licensed under this
License.  Each licensee is addressed as "you".  "Licensees" and
"recipients" may be individuals or organizations.

  To "modify" a work means to copy from or adapt all or part of the work
in a fashion requiring copyright permission, other than the making of an
exact copy.  The resulting work is called a "modified version" of the
earlier work or a work "based on" the earlier work.

  A "covered work" means either the unmodified Program or a work based
on the Program.

  To "propagate" a work means to do anything with it that, without
permission, would make you directly or secondarily liable for
infringement under applicable copyright law, except executing it on a
computer or modifying a private copy.  Propagation includes copying,
distribution (with or without modification), making available to the
public, and in some countries other activities as well.

  To "convey" a work means any kind of propagation that enables other
parties to make or receive copies.  Mere interaction with a user through
a computer network, with no transfer of a copy, is not conveying.

  An interactive user interface displays "Appropriate Legal Notices"
to the extent that it includes a convenient and prominently visible
feature that (1) displays an appropriate copyright notice, and (2)
tells the user that there is no warranty for the work (except to the
extent that warranties are provided), that licensees may convey the
work under this License, and how to view a copy of this License.  If
the interface presents a list of user commands or options, such as a
menu, a prominent item in the list meets this criterion.

  1. Source Code.

  The "source code" for a work means the preferred form of the work
for making modifications to it.  "Object code" means any non-source
form of a work.

  A "Standard Interface" means an interface that either is an official
standard defined by a recognized standards body, or, in the case of
interfaces specified for a particular programming language, one that
is widely used among developers working in that language.

  The "System Libraries" of an executable work include anything, other
than the work as a whole, that (a) is included in the normal form of
packaging a Major Component, but which is not part of that Major
Component, and (b) serves only to enable use of the work with that
Major Component, or to implement a Standard Interface for which an
implementation is available to the public in source code form.  A
"Major Component", in this context, means a major essential component
(kernel, window system, and so on) of the specific operating system
(if any) on which the executable work runs, or a compiler used to
produce the work, or an object code interpreter used to run it.

  The "Corresponding Source" for a work in object code form means all
the source code needed to generate, install, and (for an executable
work) run the object code and to modify the work, including scripts to
control those activities.  However, it does not include the work's
System Libraries, or general-purpose tools or generally available free
programs which are used unmodified in performing those activities but
which are not part of the work.  For example, Corresponding Source
includes interface definition files associated with source files for
the work, and the source code for shared libraries and dynamically
linked subprograms that the work is specifically designed to require,
such as by intimate data communication or control flow between those
subprograms and other parts of the work.

  The Corresponding Source need not include anything that users
can regenerate automatically from other parts of the Corresponding
Source.

  The Corresponding Source for a work in source code form is that
same work.

  2. Basic Permissions.

  All rights granted under this License are granted for the term of
copyright on the Program, and are irrevocable provided the stated
conditions are met.  This License explicitly affirms your unlimited
permission to run the unmodified Program.  The output from running a
covered work is covered by this License only if the output, given its
content, constitutes a covered work.  This License acknowledges your
rights of fair use or other equivalent, as provided by copyright law.

  You may make, run and propagate covered works that you do not
convey, without conditions so long as your license otherwise remains
in force.  You may convey covered works to others for the sole purpose
of having them make modifications exclusively for you, or provide you
with facilities for running those works, provided that you comply with
the terms of this License in conveying all material for which you do
not control copyright.  Those thus making or running the covered works
for you must do so exclusively on your behalf, under your direction
and control, on terms that prohibit them from making any copies of
your copyrighted material outside their relationship with you.

  Conveying under any other circumstances is permitted solely under
the conditions stated below.  Sublicensing is not allowed; section 10
makes it unnecessary.

  3. Protecting Users' Legal Rights From Anti-Circumvention Law.

  No covered work shall be deemed part of an effective technological
measure under any applicable law fulfilling obligations under article
11 of the WIPO copyright treaty adopted on 20 December 1996, or
similar laws prohibiting or restricting circumvention of such
measures.

  When you convey a covered work, you waive any legal power to forbid
circumvention of technological measures to the extent such circumvention
is effected by exercising rights under this License with respect to
the covered work, and you disclaim any intention to limit operation or
modification of the work as a means of enforcing, against the work's
users, your or third parties' legal rights to forbid circumvention of
technological measures.

  4. Conveying Verbatim Copies.

  You may convey verbatim copies of the Program's source code as you
receive it, in any medium, provided that you conspicuously and
appropriately publish on each copy an appropriate copyright notice;
keep intact all notices stating that this License and any
non-permissive terms added in accord with section 7 apply to the code;
keep intact all notices of the absence of any warranty; and give all
recipients a copy of this License along with the Program.

  You may charge any price or no price for each copy that you convey,
and you may offer support or warranty protection for a fee.

  5. Conveying Modified Source Versions.

  You may convey a work based on the Program, or the modifications to
produce it from the Program, in the form of source code under the
terms of section 4, provided that you also meet all of these conditions:

    a) The work must carry prominent notices stating that you modified
    it, and giving a relevant date.

    b) The work must carry prominent notices stating that it is
    released under this License and any conditions added under section
    7.  This requirement modifies the requirement in section 4 to
    "keep intact all notices".

    c) You must license the entire work, as a whole, under this
    License to anyone who comes into possession of a copy.  This
    License will therefore apply, along with any applicable section 7
    additional terms, to the whole of the work, and all its parts,
    regardless of how they are packaged.  This License gives no
    permission to license the work in any other way, but it does not
    invalidate such permission if you have separately received it.

    d) If the work has interactive user interfaces, each must display
    Appropriate Legal Notices; however, if the Program has interactive
    interfaces that do not display Appropriate Legal Notices, your
    work need not make them do so.

  A compilation of a covered work with other separate and independent
works, which are not by their nature extensions of the covered work,
and which are not combined with it such as to form a larger program,
in or on a volume of a storage or distribution medium, is called an
"aggregate" if the compilation and its resulting copyright are not
used to limit the access or legal rights of the compilation's users
beyond what the individual works permit.  Inclusion of a covered work
in an aggregate does not cause this License to apply to the other
parts of the aggregate.

  6. Conveying Non-Source Forms.

  You may convey a covered work in object code form under the terms
of sections 4 and 5, provided that you also convey the
machine-readable Corresponding Source under the terms of this License,
in one of these ways:

    a) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by the
    Corresponding Source fixed on a durable physical medium
    customarily used for software interchange.

    b) Convey the object code in, or embodied in, a physical product
    (including a physical distribution medium), accompanied by a
    written offer, valid for at least three years and valid for as
    long as you offer spare parts or customer support for that product
    model, to give anyone who possesses the object code either (1) a
    copy of the Corresponding Source for all the software in the
    product that is covered by this License, on a durable physical
    medium customarily used for software interchange, for a price no
    more than your reasonable cost of physically performing this
    conveying of source, or (2) access to copy the
    Corresponding Source from a network server at no charge.

    c) Convey individual copies of the object code with a copy of the
    written offer to provide the Corresponding Source.  This
    alternative is allowed only occasionally and noncommercially, and
    only if you received the object code with such an offer, in accord
    with subsection 6b.

    d) Convey the object code by offering access from a designated
    place (gratis or for a charge), and offer equivalent access to the
    Corresponding Source in the same way through the same place at no
    further charge.  You need not require recipients to copy the
    Corresponding Source along with the object code.  If the place to
    copy the object code is a network server, the Corresponding Source
    may be on a different server (operated by you or a third party)
    that supports equivalent copying facilities, provided you maintain
    clear directions next to the object code saying where to find the
    Corresponding Source.  Regardless of what server hosts the
    Corresponding Source, you remain obligated to ensure that it is
    available for as long as needed to satisfy these requirements.

    e) Convey the object code using peer-to-peer transmission, provided
    you inform other peers where the object code and Corresponding
    Source of the work are being offered to the general public at no
    charge under subsection 6d.

  A separable portion of the object code, whose source code is excluded
from the Corresponding Source as a System Library, need not be
included in conveying the object code work.

  A "User Product" is either (1) a "consumer product", which means any
tangible personal property which is normally used for personal, family,
or household purposes, or (2) anything designed or sold for incorporation
into a dwelling.  In determining whether a product is a consumer product,
doubtful cases shall be resolved in favor of coverage.  For a particular
product received by a particular user, "normally used" refers to a
typical or common use of that class of product, regardless of the status
of the particular user or of the way in which the particular user
actually uses, or expects or is expected to use, the product.  A product
is a consumer product regardless of whether the product has substantial
commercial, industrial or non-consumer uses, unless such uses represent
the only significant mode of use of the product.

  "Installation Information" for a User Product means any methods,
procedures, authorization keys, or other information required to install
and execute modified versions of a covered work in that User Product from
a modified version of its Corresponding Source.  The information must
suffice to ensure that the continued functioning of the modified object
code is in no case prevented or interfered with solely because
modification has been made.

  If you convey an object code work under this section in, or with, or
specifically for use in, a User Product, and the conveying occurs as
part of a transaction in which the right of possession and use of the
User Product is transferred to the recipient in perpetuity or for a
fixed term (regardless of how the transaction is characterized), the
Corresponding Source conveyed under this section must be accompanied
by the Installation Information.  But this requirement does not apply
if neither you nor any third party retains the ability to install
modified object code on the User Product (for example, the work has
been installed in ROM).

  The requirement to provide Installation Information does not include a
requirement to continue to provide support service, warranty, or updates
for a work that has been modified or installed by the recipient, or for
the User Product in which it has been modified or installed.  Access to a
network may be denied when the modification itself materially and
adversely affects the operation of the network or violates the rules and
protocols for communication across the network.

  Corresponding Source conveyed, and Installation Information provided,
in accord with this section must be in a format that is publicly
documented (and with an implementation available to the public in
source code form), and must require no special password or key for
unpacking, reading or copying.

  7. Additional Terms.

  "Additional permissions" are terms that supplement the terms of this
License by making exceptions from one or more of its conditions.
Additional permissions that are applicable to the entire Program shall
be treated as though they were included in this License, to the extent
that they are valid under applicable law.  If additional permissions
apply only to part of the Program, that part may be used separately
under those permissions, but the entire Program remains governed by
this License without regard to the additional permissions.

  When you convey a copy of a covered work, you may at your option
remove any additional permissions from that copy, or from any part of
it.  (Additional permissions may be written to require their own
removal in certain cases when you modify the work.)  You may place
additional permissions on material, added by you to a covered work,
for which you have or can give appropriate copyright permission.

  Notwithstanding any other provision of this License, for material you
add to a covered work, you may (if authorized by the copyright holders of
that material) supplement the terms of this License with terms:

    a) Disclaiming warranty or limiting liability differently from the
    terms of sections 15 and 16 of this License; or

    b) Requiring preservation of specified reasonable legal notices or
    author attributions in that material or in the Appropriate Legal
    Notices displayed by works containing it; or

    c) Prohibiting misrepresentation of the origin of that material, or
    requiring that modified versions of such material be marked in
    reasonable ways as different from the original version; or

    d) Limiting the use for publicity purposes of names of licensors or
    authors of the material; or

    e) Declining to grant rights under trademark law for use of some
    trade names, trademarks, or service marks; or

    f) Requiring indemnification of licensors and authors of that
    material by anyone who conveys the material (or modified versions of
    it) with contractual assumptions of liability to the recipient, for
    any liability that these contractual assumptions directly impose on
    those licensors and authors.

  All other non-permissive additional terms are considered "further
restrictions" within the meaning of section 10.  If the Program as you
received it, or any part of it, contains a notice stating that it is
governed by this License along with a term that is a further
restriction, you may remove that term.  If a license document contains
a further restriction but permits relicensing or conveying under this
License, you may add to a covered work material governed by the terms
of that license document, provided that the further restriction does
not survive such relicensing or conveying.

  If you add terms to a covered work in accord with this section, you
must place, in the relevant source files, a statement of the
additional terms that apply to those files, or a notice indicating
where to find the applicable terms.

  Additional terms, permissive or non-permissive, may be stated in the
form of a separately written license, or stated as exceptions;
the above requirements apply either way.

  8. Termination.

  You may not propagate or modify a covered work except as expressly
provided under this License.  Any attempt otherwise to propagate or
modify it is void, and will automatically terminate your rights under
this License (including any patent licenses granted under the third
paragraph of section 11).

  However, if you cease all violation of this License, then your
license from a particular copyright holder is reinstated (a)
provisionally, unless and until the copyright holder explicitly and
finally terminates your license, and (b) permanently, if the copyright
holder fails to notify you of the violation by some reasonable means
prior to 60 days after the cessation.

  Moreover, your license from a particular copyright holder is
reinstated permanently if the copyright holder notifies you of the
violation by some reasonable means, this is the first time you have
received notice of violation of this License (for any work) from that
copyright holder, and you cure the violation prior to 30 days after
your receipt of the notice.

  Termination of your rights under this section does not terminate the
licenses of parties who have received copies or rights from you under
this License.  If your rights have been terminated and not permanently
reinstated, you do not qualify to receive new licenses for the same
material under section 10.

  9. Acceptance Not Required for Having Copies.

  You are not required to accept this License in order to receive or
run a copy of the Program.  Ancillary propagation of a covered work
occurring solely as a consequence of using peer-to-peer transmission
to receive a copy likewise does not require acceptance.  However,
nothing other than this License grants you permission to propagate or
modify any covered work.  These actions infringe copyright if you do
not accept this License.  Therefore, by modifying or propagating a
covered work, you indicate your acceptance of this License to do so.

  10. Automatic Licensing of Downstream Recipients.

  Each time you convey a covered work, the recipient automatically
receives a license from the original licensors, to run, modify and
propagate that work, subject to this License.  You are not responsible
for enforcing compliance by third parties with this License.

  An "entity transaction" is a transaction transferring control of an
organization, or substantially all assets of one, or subdividing an
organization, or merging organizations.  If propagation of a covered
work results from an entity transaction, each party to that
transaction who receives a copy of the work also receives whatever
licenses to the work the party's predecessor in interest had or could
give under the previous paragraph, plus a right to possession of the
Corresponding Source of the work from the predecessor in interest, if
the predecessor has it or can get it with reasonable efforts.

  You may not impose any further restrictions on the exercise of the
rights granted or affirmed under this License.  For example, you may
not impose a license fee, royalty, or other charge for exercise of
rights granted under this License, and you may not initiate litigation
(including a cross-claim or counterclaim in a lawsuit) alleging that
any patent claim is infringed by making, using, selling, offering for
sale, or importing the Program or any portion of it.

  11. Patents.

  A "contributor" is a copyright holder who authorizes use under this
License of the Program or a work on which the Program is based.  The
work thus licensed is called the contributor's "contributor version".

  A contributor's "essential patent claims" are all patent claims
owned or controlled by the contributor, whether already acquired or
hereafter acquired, that would be infringed by some manner, permitted
by this License, of making, using, or selling its contributor version,
but do not include claims that would be infringed only as a
consequence of further modification of the contributor version.  For
purposes of this definition, "control" includes the right to grant
patent sublicenses in a manner consistent with the requirements of
this License.

  Each contributor grants you a non-exclusive, worldwide, royalty-free
patent license under the contributor's essential patent claims, to
make, use, sell, offer for sale, import and otherwise run, modify and
propagate the contents of its contributor version.

  In the following three paragraphs, a "patent license" is any express
agreement or commitment, however denominated, not to enforce a patent
(such as an express permission to practice a patent or covenant not to
sue for patent infringement).  To "grant" such a patent license to a
party means to make such an agreement or commitment not to enforce a
patent against the party.

  If you convey a covered work, knowingly relying on a patent license,
and the Corresponding Source of the work is not available for anyone
to copy, free of charge and under the terms of this License, through a
publicly available network server or other readily accessible means,
then you must either (1) cause the Corresponding Source to be so
available, or (2) arrange to deprive yourself of the benefit of the
patent license for this particular work, or (3) arrange, in a manner
consistent with the requirements of this License, to extend the patent
license to downstream recipients.  "Knowingly relying" means you have
actual knowledge that, but for the patent license, your conveying the
covered work in a country, or your recipient's use of the covered work
in a country, would infringe one or more identifiable patents in that
country that you have reason to believe are valid.

  If, pursuant to or in connection with a single transaction or
arrangement, you convey, or propagate by procuring conveyance of, a
covered work, and grant a patent license to some of the parties
receiving the covered work authorizing them to use, propagate, modify
or convey a specific copy of the covered work, then the patent license
you grant is automatically extended to all recipients of the covered
work and works based on it.

  A patent license is "discriminatory" if it does not include within
the scope of its coverage, prohibits the exercise of, or is
conditioned on the non-exercise of one or more of the rights that are
specifically granted under this License.  You may not convey a covered
work if you are a party to an arrangement with a third party that is
in the business of distributing software, under which you make payment
to the third party based on the extent of your activity of conveying
the work, and under which the third party grants, to any of the
parties who would receive the covered work from you, a discriminatory
patent license (a) in connection with copies of the covered work
conveyed by you (or copies made from those copies), or (b) primarily
for and in connection with specific products or compilations that
contain the covered work, unless you entered into that arrangement,
or that patent license was granted, prior to 28 March 2007.

  Nothing in this License shall be construed as excluding or limiting
any implied license or other defenses to infringement that may
otherwise be available to you under applicable patent law.

  12. No Surrender of Others' Freedom.

  If conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot convey a
covered work so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you may
not convey it at all.  For example, if you agree to terms that obligate you
to collect a royalty for further conveying from those to whom you convey
the Program, the only way you could satisfy both those terms and this
License would be to refrain entirely from conveying the Program.

  13. Use with the GNU Affero General Public License.

  Notwithstanding any other provision of this License, you have
permission to link or combine any covered work with a work licensed
under version 3 of the GNU Affero General Public License into a single
combined work, and to convey the resulting work.  The terms of this
License will continue to apply to the part which is the covered work,
but the special requirements of the GNU Affero General Public License,
section 13, concerning interaction through a network will apply to the
combination as such.

  14. Revised Versions of this License.

  The Free Software Foundation may publish revised and/or new versions of
the GNU General Public License from time to time.  Such new versions will
be similar in spirit to the present version, but may differ in detail to
address new problems or concerns.

  Each version is given a distinguishing version number.  If the
Program specifies that a certain numbered version of the GNU General
Public License "or any later version" applies to it, you have the
option of following the terms and conditions either of that numbered
version or of any later version published by the Free Software
Foundation.  If the Program does not specify a version number of the
GNU General Public License, you may choose any version ever published
by the Free Software Foundation.

  If the Program specifies that a proxy can decide which future
versions of the GNU General Public License can be used, that proxy's
public statement of acceptance of a version permanently authorizes you
to choose that version for the Program.

  Later license versions may give you additional or different
permissions.  However, no additional obligations are imposed on any
author or copyright holder as a result of your choosing to follow a
later version.

  15. Disclaimer of Warranty.

  THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. Limitation of Liability.

  IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING
WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS
THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY
GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE
USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF
DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD
PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS),
EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

  17. Interpretation of Sections 15 and 16.

  If the disclaimer of warranty and limitation of liability provided
above cannot be given local legal effect according to their terms,
reviewing courts shall apply local law that most closely approximates
an absolute waiver of all civil liability in connection with the
Program, unless a warranty or assumption of liability accompanies a
copy of the Program in return for a fee.

                     END OF TERMS AND CONDITIONS

            How to Apply These Terms to Your New Programs

  If you develop a new program, and you want it to be of the greatest
possible use to the public, the best way to achieve this is to make it
free software which everyone can redistribute and change under these terms.

  To do so, attach the following notices to the program.  It is safest
to attach them to the start of each source file to most effectively
state the exclusion of warranty; and each file should have at least
the "copyright" line and a pointer to where the full notice is found.

    <one line to give the program's name and a brief idea of what it does.>
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

  If the program does terminal interaction, make it output a short
notice like this when it starts in an interactive mode:

    <program>  Copyright (C) <year>  <name of author>
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate
parts of the General Public License.  Of course, your program's commands
might be different; for a GUI interface, you would use an "about box".

  You should also get your employer (if you work as a programmer) or school,
if any, to sign a "copyright disclaimer" for the program, if necessary.
For more information on this, and how to apply and follow the GNU GPL, see
<https://www.gnu.org/licenses/>.

  The GNU General Public License does not permit incorporating your program
into proprietary programs.  If your program is a subroutine library, you
may consider it more useful to permit linking proprietary applications with
the library.  If this is what you want to do, use the GNU Lesser General
Public License instead of this License.  But first, please read
<https://www.gnu.org/licenses/why-not-lgpl.html>.

```


## 2026-10-29 current v2 public joint-factorization contract

The output remains abundance maps A(4,255,255),65-bin spectra F(4,65), and dictionary weights R(4,4), stored batch first as float64. Component permutation is a gauge: all three arrays must be permuted together. The independent standalone Judge does not import src, use initialization identity, or compare method arrays. It tests A,R,F nonnegative, per-pixel sum(A)<=1, each R row sum1, each column sum<=1, F=R T, and all600views/all65selected energies through the public fan-beam optical-depth model. The full squared residual may not exceed independently attained standard by more than factor1.00000001; the original twelve-view relative residual boundary.1700728033 is additionally retained. Dictionary discrepancy limit is1e-6, R row sum2e-6, other simplex/nonnegative limits1e-10.

T is not hidden pure-material truth or target-derived calibration: its first three atoms are the visible calibration and its fourth is the explicitly defined observation-only100step,all128energy SIRT ROI statistic. README defines X0=0, the exact100step recurrence, ray/voxel normalization cutoff1e-6, float32 arithmetic, and11by11ROI indices129..139. The independent original observation-derived T is cached in the hidden baseline only to avoid repeating that deterministic functional inside every Judge; new fresh T was bitwise identical. This cache is not solver input and no pure quartz truth is asserted.

The real full cuda_ordered pipeline completed100SIRT+1000ADJUST, exit0,2002.839s CLI runtime (2004.266s controller). Three required arrays, T and every entry of all four1000step trajectories were bitwise identical to the independently completed original: selected best iterate795, residual.153078688318426. The independent corrected-original ADJUST epoch wall time6567.527808189392s, independent original cJoint3490.8370950222015s, historical unmodified4385.1106532s, and this fresh run have distinct provenance. Independent Judge full fan-beam projection gave.15307868831848234, a5.6e-14difference from the source residual. The baseline was added only after these actual gates; it is the independent original method output, not physical truth and not the new fresh output.

Validated backend boundary: Windows10.0.26200,Python3.11.11,PyTorch2.9.1+cu128,CuPy13.6,RTX4080SUPER16GiB,driver610.88. Generic Linux OpenBLAS0.3.26 Haswell4threads and Linux libm are reached through the existing WSL worker, explicitly configured via IMAGING_BLAS_LIBRARY; native Windows NumPy/OpenMP use1thread. Default system libblas probe failed (no openblas_get_config); the configured existing library probe and real complete pipeline succeeded without installation. This is an explicit Windows/WSL numerical exception, not a claim of a wholly native Windows CPU run. Live notebook guard requires9.5GiB initial physical/commit headroom and stops its owned processes below6GiB; resource samples/logs are private.

Stage/mathematical/Judge tests62passed in32.87s on this exact configured backend. The active full notebook completed exit0 in5315.453s with a research-runtime kernel and10800s per-cell timeout. It genuinely computed RU/UR/cJoint500 and the second ADJUST1000 initialization, preserving every original comparison. All9rendered figures and every metric panel were actually reviewed; zero cell errors. Source/fresh cJoint maps and spectra were exact; the second-initialization difference is retained as model/optimizer sensitivity. Final full pytest test/ run passed64tests in61.37s, including complete ADJUST history/quartz parity and fresh cJoint parity; exact commands/stdout/exit are in semantic_v2/adjust_complete_pytest.json. Canonical notebook outputs were refreshed only from this actual executed artifact after strict per-cell source identity checks, retaining portable original kernel metadata. No further reconstruction was required. Initial ADJUST missing-output/missing-baseline evaluator failure and earlier divergence evidence remain historical; no cached result was substituted for the fresh reconstruction.

Private evidence: benchmark_readiness_20261029/semantic_v2/adjust_spectral_material_tomography/{fresh_full_pipeline_execution.json,fresh_original_full_parity.json,independent_original_physical_validation.json,backend_probe.json,backend_configured_probe.json,notebook_execution.json}; stage tests in semantic_v2/adjust_stage_pytest.json. Raw observations and independent actual-truth assets were unchanged. No fresh blind-agent solve is claimed.
