# Authoritative provenance and validation

Status: **READY for independent parent acceptance**, not yet independently accepted. All local mandatory gates below passed. No release task was edited. Qualification and task assets are in a private staging area. There is no actual constitutive ground truth.

## Exact binding, rights, and primary materials

Paper: Abbasi et al., Discovery of Hyperelastic Constitutive Laws from Experimental Data with EUCLID, arXiv2510.24747v1 (16Oct2025), https://arxiv.org/html/2510.24747v1 ; journal DOI10.1007/s11340-026-01290-6. Primary full HTML was actually read, including §2.3.1 Eq23 and fixed-point description. CC BY4.0 applies to paper text, not automatically software/data.

Original repository: https://github.com/EUCLID-code/EUCLID-hyperelasticity-PlaneStress-ExperimentalData ; exact remotely verified commit4e76e26bdeebb388b4221dc561748cb229627201, recorded in source_revision.json. Archived source has **no LICENSE grant**. No source code or comments were copied/adapted into this task: executable local math was independently authored from the paper equations and audited conventions. Upstream remains private only. This is a corrected fixed-lambda Eq23 stage, not unchanged software reproduction and not the full paper Pareto/refinement/FE pipeline.

Dataset DOI10.3929/ethz-c-000788267; selected complete author UT/dataset_600_steps, all600 frames (IDs10..6000),383nodes,696triangles. Original3,018,100,640-byte author ZIP remains remotely indexed; exact selected46,334,728-byte range and extraction hashes in selected_data_provenance.json, zip_range_source.json and archive_entries.json. Archived author README identifies real NR-40 rubber, thickness2.5mm, calibrated stereo-DIC,40mm/min displacement rate,2Hz sampling. Reaction records are already N/mm. All numeric CSV precision was retained as float64; no downcast. Stored element F/stress fields were NOT bundled or used as solver input; F is computed from observed displacement and legitimate author geometry gradients.

Official rights independently re-fetched from https://www.research-collection.ethz.ch/server/api/core/items/dfb788ca-3f77-4572-b118-53343622526a and archived as official_item_metadata.json on26Oct2026. Exact dc.rights.license: **In Copyright - Non-Commercial Use Permitted**; dc.rights.uri: http://rightsstatements.org/page/InC-NC/1.0/. Initial bundle/item fetch429 and web403 were not misrepresented as successful; direct correct item endpoint succeeded. README/NOTICE prominently retain the restriction. No broader commercial or external-publication permission is asserted.

Private archive root: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch9_20261026/math_mechanics/euclid`. Full original checkout, PDFs/download ranges, capture scripts and logs stay outside the task. Source hash/remote verification is distinguished from current freshness: exact captured revision verified; no claim it remains the latest branch.

## Original exact-observation run and failure qualification

Original command: sharedPY capture_full_original.py with A/oracle-deps prepended, source core/config_TPE1, CPU torch threads/interops1, OMP/MKL/OpenBLAS1. Python3.11.11, NumPy2.4.6, Torch2.9.1+cu128 CPU; original-only pandas2.3.3/seaborn0.13.2 and other import dependencies are isolated in oracle-deps, absent from cleaned imports. Full600 run finished exit0 in10530.14997s; full_original_run.log (UTF16), full_original_progress.json, original_first_stage.npz, original_full_normal_system.npz and original_full_solution.npz preserved. Original max200 seed0 was exactly replayed from the frozen normal system, validating sufficient-statistic inverse replay. This replay does not claim to repeat3h preprocessing.

Native algorithm **FAILED scientific reference qualification**. Permanent threshold pruning can trap supports; original winner has90 negative inactive KKT violations. max10000 improves iteration convergence but does not stabilize the native winner: refined seed1 vs seed0 full-domain stress difference25.80%. Native computeCostLp counts lambda inside each of600 dataset loops, whereas its fixed-point normal equations and paper Eq23 use one lambda. Both criteria and all15 higher-iteration native candidates are retained; source ranking differs from single-lambda ranking in2/3 seeds. Neither exact original replay nor source energy checks establish a valid optimum. Native output is retained only as an explicitly failed comparison.

## Corrected independent qualification BEFORE local extraction

assistant_audit/corrected_qualification_frozen_before_local.json froze the unchanged2189-query domain,3-start corrected-optimizer arrays, independent convex certificate, original-system hashes and thresholds BEFORE any cleaned implementation or full600 local assembly. Original operators were the immutable exact-observation H/b; corrected_irls_probe.py independently implemented the published weighted-LS fixed-point family with positivity-constrained subproblems, reentry and scaled floor continuation. This is a substantive algorithm correction, not mere tolerance refinement. No original unlicensed code was edited/copied. All3 original-operator starts were bitwise identical;12.73s combined. Full projected KKT1.5321e-10. Independent general Clarabel0.11.1 convex-QP diagnostic had full scaled active KKT2.81e-14/inactive min1.12e-6 and gap7.73e-14; no Hessian clipping or physical kernel change. Corrected weighted-LS energy/stress versus certificate relativeL2 were1.06e-12/1.40e-11. General QP is private independent certification, not cleaned inverse implementation.

Source H is PSD as a sum of Gram matrices; its scaled smallest eigenvalue-2.70e-14 is floating-point roundoff. Source feature dictionary is redundant (e.g. exponent2 recovers I1-3 and exponent-2 recovers I2-3); no full171rank/unique-coefficient claim is made. Actual query domain frozen before candidates:101 UT stretches1..3.823 plus every696 actual F at IDs1000,3000,6000. No post-fit masks, frame exclusions or baseline-dependent query selection.

Frozen boundaries: energy relativeL2<=1e-5; all-component stress relativeL2<=1e-5; full weak residual<=0.036310288113492045. First two comfortably exceed independently measured1e-11 law variation while reject native6%-26% failures; third is1% above independently certified sqrt(V/C)=0.03595078031. These are numerical conditional-estimator tolerances, not physical-accuracy or calibrated-noise guarantees. Full weights/dictionary/units/query definitions are public; optimizer implementation knobs remain maintainer-only. No threshold was relaxed after local results.

## Equation / source / local / validation traceability

| Published stage or convention | Original source | Independent local function | Validation evidence |
|---|---|---|---|
| Triangle deformation and quadrature | core/loadFemData.py, utilities.py | preprocessing.element_deformation | actual first-stage and all600 normal-system parity; calibrated partition unity |
| Incompressible plane stress, paper Eqs8/11/13 | core/utilities.py invariant/stretch derivatives | physics_model.features/law | analytical neo-Hookean3D pressure elimination; objectivity; central finite difference |
| Invariant/Gent–Thomas/Ogden dictionary, Eq7 | core/features.py; config_TPE1 N5/150alpha | physics_model.features | source small3F fixture, actual696-element first frame, all171derivatives |
| Weak force balance / Eq22 | models_weak.py computeWeakLHS, considerReactionGlobal | weak_operator/frame_system/assemble | independent one-triangle exact factor2; active all600 H/b parity |
| Nonnegative fixed-lambda LASSO Eq23 | models_Lp.py applyPenaltyLpThreshold, buggy computeCostLp ranking | solvers.solve/positive_quadratic/objective | corrected original-system3starts + independent QP; local full projected KKT |
| Full Pareto/refinement | larger paper pipeline/source driver | NOT selected/implemented | Explicitly outside task; no full-paper-reproduction claim |

Discrepancies: v1 Eq12 incorrectly leaves pressure in stored energy despite J3D=1; Eq8 gives W=Qtheta, matching audited source. v1 Eq24 uses lambda where its weak-loss description suggests beta; this task explicitly uses source/paper Eq23 beta20. Paper main study521features/-50..50 differs selected author config171features/-10..20; actual configured binding is explicit. Alpha nominal0 is retained at-3.55e-14. Final native applyThreshold can early-return without refitting; no always-refit claim is made. Native scientific energy checks were separately performed only on450 synthetic test deformations in the audit, not falsely called full600 reruns.

## Fresh local implementation, numerical results, and plans

Runtime only NumPy2.4.6, Matplotlib3.11.1 for notebook; no domain packages, Torch, Pandas or Clarabel are imported by cleaned solver/Judge/tests/notebook. Local_import_audit.json lists actual AST imports. Core task math and custom inverse updates are visible local functions; no hidden matrices/checkpoints are imported by reconstruction. Full600 assembly streams one frame (696×171×4 derivative tensor), not a2.3GB all-frame tensor. Resource headroom remained above6GiB; CPU single-thread environment was used.

Small source feature/Jacobian scaled differences3.46e-15/1.20e-14. Full600 local assembly13.10s; H/b relative errors1.95e-16/2.62e-16 against original normal system. Fresh corrected solve~4s, original-system certificate stress difference1.40e-11. Fresh local weakSSE34577.88329057, penalty5054.58880762, total39632.47209819, residual0.03595078030945. Source measurements and fresh cleaned measurements are separately identified; neither is true physical ground truth.

README and plans were semantically checked against actual signatures, ordering, factors2, beta/lambda placement, signs, physical units, source conventions and active call paths. MethodHints defines conditional estimator but gives no full optimizer recipe. Plan includes actual working-set/floor updates, stopping/failure checks and defaults. Five display formulas passed both KaTeX and MathJax in math_check/document_math_render_check.json; notebook uses standard display delimiters, not literal fenced-math cells.

## Actual commands and completion gates

Commands executed by final_execution.py with the validated shared Python, CPU thread environment and explicit private shared-py Jupyter kernel. Each log is private A/final_LABEL.log. All subprocess exit codes were captured directly, avoiding PowerShell stderr-redirection status ambiguity.

- cli: exit0, 17.067092s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m src.main`.
- judge: exit0, 0.202403s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`.
- tests: exit0, 18.175573s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m pytest test/ -q --basetemp=D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch9_20261026\math_mechanics\euclid\pytest_final`.
- notebook: exit0, 57.813249s; command `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/euclid_constitutive_law.ipynb --output euclid_constitutive_law_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=shared-py --ExecutePreprocessor.timeout=900`.

Tests:25passed, including upstream small math/actual first frame/full600 assembly, fresh corrected fit, finite-difference gradients/objectivity/plane-stress pressure elimination, objective factor2, boundary masks/calibration, all public keys, and strictJudge contracts. Invalid/missing/key/shape/integer/complex/nonfinite/negative inputs yield2 and empty metrics; zero law gives1; reference gives0; hidden-asset and report-write errors give2; external caller directory and report-parent creation tested. No solver import occurs in Judge.

Notebook: complete active reconstruction, strict independent evaluation,7computed scientific figures across8stories. Actual kernel interpreter/version is printed. All outputs/error cells inspected; no error outputs. All7PNG figures actually opened: complete geometry/acquisition; native/certificate/fresh laws plus2189query differences; full600 residuals and derived spatial fields;15native candidate criteria plus IRLS effort; invariance/FD/rank diagnostics; actually fitted penalties/restricted-model ablations; all600 force-calibration perturbations and fresh independent-start stability. Original scientific comparisons relevant to selected stage are retained; omitted full Pareto/unregularized/FE study is explicit rather than presented as completed. Temporary executed notebook was moved outside task; canonical notebook contains executed outputs.

Gate status: paper/revision/hashes/mapping PASS; exact original run PASS; native suitability FAIL (retained); separately corrected selected-stage qualification PASS; independent parity/physics PASS; constants/units PASS; active CLI/Judge PASS; strictnegativecases PASS; public self-sufficiency/leak audit PASS; minimal runtime/local math PASS;25tests PASS; active notebook/7visuals/math PASS; metric/provenance distinction PASS; plan consistency PASS; template/license/inventory PASS. Corrected variant acceptance by parent remains pending and is not implied by these local passes.

## Hidden assets and consumers

evaluation/baseline_reference.npz is qualified corrected-method theta/law plus original Q/G/H/b/C operators for independent Judge scoring. It is NOT ground_truth. All query entries are evaluated. test/fixtures/small_math_probe comes from original small3F/triangle execution, original_first_stage from exact first acquisition frame, original_full_normal_system from original all600 run, corrected_optimizer from frozen independent corrected-system run. Tests document their consumers. Notebook references native_failed_baseline (unmodified original), native_single_lambda_selected (max10000 source candidate), native_multiseed_comparison (15native candidate metrics), independent_certificate (privateQPtheta), each used in an explicit comparison. Raw inputs contain no H/b/Q/G, fitted coefficients, stresses, true labels, source fitted model or simulator seed.

Hash index below identifies private source/qualification inputs. selected_data_provenance.json contains every original selected CSV hash. Final readiness manifest additionally hashes all shipped files; generated CLI output is separately identified.

| Private archive path | SHA256 |
|---|---|
| `source_revision.json` | `09fdee6674fae351dd85bbb1a02b8ba6805f0d1966f5ea3de3671398b093106f` |
| `selected_data_provenance.json` | `f966496776ca95bddfe1484851bea3158a8bc6e09cbf321ef38bea48b9d8b1b6` |
| `official_item_metadata.json` | `5f5082e9ed863ca0ffcfa0f0bcd5ee64fd46337ce4e5231abff374a360eb771d` |
| `original_full_normal_system.npz` | `261a54a18e48aef1f221fcb80159dc586e1169d286b48a865f3fde40866d7cfb` |
| `original_full_solution.npz` | `e3067b8c4ebd606063d2fab80704ae90a41fbed9ffcb7a91d027a8563d248343` |
| `original_first_stage.npz` | `cd1aabc96bffc809b60d59057c2f49708e6e595d06a15cd232b5bc84336aef3f` |
| `assistant_audit\corrected_qualification_frozen_before_local.json` | `980169f99a2b702ffeb615e30d6cec27ccde4f5deaee5071e056b1cc0ea7704d` |
| `assistant_audit\corrected_irls_probe.py` | `5db7922b7b9670a1a2fb454ea1eb69447ca0eba4e6d14001f7a250a83eaa5ad5` |
| `assistant_audit\corrected_irls_probe.npz` | `51496b554672403fc4f5e21fdf73557e9609c380c1ab8d4203853d087ecaa133` |
| `assistant_audit\independent_convex_qp.json` | `bcace010b0f3a53a1ccc7b6fe473a0b92e85305147d3ae223e523c5b1d4b8dd6` |
| `assistant_audit\announced_query_gradients.npz` | `bca77e6edce850b47254eb86f00c91c0717f70e07d057942ba483c107c3f304f` |

Public raw_data.npz SHA256: `747a40b952a05f1c6aa97d05f70068783d65639da59a93b3e3e0ccefbe6fb05c`.


## Parent review corrections and final recheck

Independent peer audit is archived as euclid/parent_peer_review.json. Its direct four-active-column normal-equation certificate found positive coefficients, minimum active normalized Gram eigenvalue1.287248e-4, all-inactive scaled gradient minimum1.115877e-6 and active gradient maximum1.69e-16; energy/stress versus the frozen law differ1.72e-13/8.23e-13. Frozen qualification14:39:15UTC predates local source creation14:40:08UTC. The corrected IRLS family, all600 analytic assembly, physical normalization and conditional law contract passed scientific review.

The parent repaired README/approach/NOTICE encoding and missing mathematical backslashes (sum/outer-product); valid renderer syntax alone had not detected plain-letter semantic corruption. Numeric solver, observations, dictionary, regularization, thresholds and references did not change. Current documentation is checked again by both math renderers.

Judge hardening changed only evaluation/judge.py and its unit contract tests. Private pre-fix backups are parent_peer_before_judge.py and parent_peer_before_test_judge.py. The Judge now strictly accepts real float32/float64 coefficient arrays, validates exact float64 hidden Q/G/H/b/C and energy/stress schemas, checks finite/nondegenerate normalizations, and explicitly prioritizes actual truth if supplied. No actualtruth asset was added. Scale/log arithmetic handles extreme finite coefficients; only mathematically unrepresentable scores saturate at float64 maximum. Valid poor huge-finite predictions return fail1 with finite JSON, while malformed arrays/assets return error2 with empty metrics. Actual ordinary metrics and all frozen boundaries remain unchanged within floating rounding.

Thirty-two relevant Judge contract tests passed in2.37s, including known metrics, accurate float32, float16/object rejection, huge finite input, malformed hidden shapes/dtypes/values/norms, truth priority and malformed-truth nonfallback, foreign caller directory and report failure. A final parent full45-test and actual notebook recheck follow; their commands, direct native exit status, timing and old-log archive are recorded in acceptance/euclid_constitutive_law/execution.json. These fresh parent rechecks supersede earlier25-test counts without relabelling the old runs.


Final mechanical recheck completed: new standalone Judge exit0 in 0.213174s; complete45tests PASS (19.00s pytest-reported, 19.254666s subprocess); B/recheck_notebook.py actual full notebook exit0 in 55.205844s. All10code cells executed, zero error outputs,7PNG outputs extracted,5formulas passed both renderers. Fresh executed notebook and all7PNG paths/hashes are in acceptance/euclid_constitutive_law/mechanical_recheck_summary.json. Old Judge/test/execution logs and pre-update design/validation were retained under prior_judge_hardening; prior notebook logs under prior_notebook_attempts. Generated PNG byte signatures and inventory were verified. Independent parent visual review and final canonical synchronization/inventory freeze are still pending; this worker did not finalize or freeze.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 10 executed code cells, 7 computed PNG outputs, 5 formulas. Private evidence: research_batch9_20261026/acceptance/euclid_constitutive_law/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


Final packaging: output/notebook_measurements.json was SHA256-verified into the maintainer benchmark_readiness_20261029/root_task_output_archives/euclid_constitutive_law archive and removed from the task. It is an auxiliary maintainer diagnostic with no runtime reader; canonical notebook measurements/figures and the standard reconstruction/report remain. Archive index.json records its original filename and hash.
