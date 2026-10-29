# Provenance and validation

## Binding and distinctness
Measured resonance frequencies -> two isotropic stiffnesses of a free 4140 cylinder, with initially unknown mode correspondences. This inverse elastic-spectrum problem differs from image displacement/stress reconstruction, EUCLID constitutive identification and generic structural-modal deformation tasks: the acquisition is contact RUS, the unknowns are bulk c11,c44, and the forward operator is a three-dimensional traction-free continuum eigenproblem with missing and degenerate resonances. Root approved this binding and explicit manual-matching adaptation before extraction.

Paper: Torres, Flores-Betancourt and Hermann, JASA151,3547-3563(2022), DOI10.1121/10.0011397. Author accepted manuscript is privately archived as archive/ruscal_paper.pdf; Figure5 and accompanying text identify measured commercial4140 steel, cylinder dimensions and experimental acquisition using Balakirev etal., RSI90,121401(2019), DOI10.1063/1.5123165. Source: https://github.com/RUS-ORNL/RUScal, actual HTML-verified current revision b3446a804b38ee9e03c1fe9a10ecd2a1771240d3. Private archive/rus_archive_index.json records every source, paper and capture identity. BSD3 source/data snapshot notice is retained in LICENSE. No claim of later remote freshness.

## Exact observations and conventions
Source rusin_4140_fit.dat has76 hand-assigned rows, including six zero placeholders. Keep all70 positive frequencies and corresponding source fit weights. Zero placeholders are not measurements and are removed. Repeated frequency values encode author assignment of experimentally degenerate modes, not independent instrument repeats. Two positive entries have zero source weight, and one has weight0.5: these are author's analysis preferences, not inferred measurement standard deviations. All remain public and all are matched; weight0 entries do not affect paired LM. Original manual row indices exist only in test/fixtures/upstream.npz; no such labels, target stiffness estimate or target-derived calibration enter raw input.

Public data contains only frequencies (1,70) float64 MHz and analysis_weights (1,70) float64 dimensionless. Dimensions1.900094,1.900094,1.338456cm and mass29.7024g are independent geometry calibration. Cylinder axis z, free traction surface, homogeneous isotropic model. Native code scales density and stiffness internally; explicit local conversion is independently checked against cylinder moments and stiffness-frequency scaling.

## Original execution
archive/capture_ruscal.py runs the exact bundled observations using the author's bundled binary ruscal_author.exe, SHA2568958b0bcc06fbb92b985a9c82e51dfc117e29417c905f0820ba55da350ff5c8e, PE32/i386. The original full run took19.3437276s, completed three fitting iterations and printed Run ended. Its native exit is1 because original rus.c unconditionally returns1 at the end: this is recorded as1, never rewritten to0. The capture wrapper validates expected return1 plus complete output before returning0 itself. Detailed native commands, stdout/stderr, input/output hashes and nominal/perturbed forward runs are indexed in archive/ruscal_original_execution.json and archive/ruscal_oracle/. The complete original spectrum and moduli were captured before local extraction. Independent nominal (2.7,.8) and perturbed (2.8,.85) Mbar runs took about0.53s each.

A diagnostic x64 GCC14.2 build exposed source long-from-pointer truncation on Windows LLP64 and is not a trustworthy oracle. It is retained privately as diagnostic evidence, never used for fixtures. The bundled author32-bit executable avoids that incompatible pointer representation.

No true material moduli are available. evaluation/baseline_reference.npz and notebooks/reference_outputs/upstream_measurements.npz identify the original method fit: c11=275.4848GPa,c44=82.4368GPa as printed (six decimals Mbar); all76 theoretical frequencies are native printed outputs. Source weighted RMS=0.0774871percent. These are not labeled ground truth.

## Predeclared gates and disclosed adaptation
archive/ruscal_frozen_criteria.json predates extraction. Forward tolerances atol3e-6MHz,rtol2e-6; per-modulus absolute error<=0.002GPa; full spectrum maximum absolute error<=3e-6MHz; weighted observation RMS<=0.08523581percent. Tolerances exceed native output quantization yet remain small relative to measured mismatch. Two scalar moduli have no meaningful image NCC/range-NRMSE, so separately evaluate each. No post-fit threshold relaxation occurred.

The author source takes manual correspondences. This package explicitly adds monotone dynamic programming as an implementation of that manual stage. It minimizes squared relative matching costs, permits missing predicted modes and assigns every positive supplied entry. It uses only measured frequencies, geometry and coarse initial steel moduli. An initial fit to twelve low modes follows the paper's low-frequency initialization recommendation. It is not claimed to be unchanged author matching code. Source paired LM, polynomial order14, finite difference increments, damping, mixed residual conventions and stopping are preserved. Six independent coarse priors recover all70 original manual associations exactly; physically degenerate permutations would have been allowed by the predeclared gate but were not needed. archive/ruscal_multiple_priors.json records full results. Final unrounded local c11=275.4848274415856,c44=82.43676028426498GPa; fresh RMS=0.077487069797percent. Forward nominal error was4.9643526e-7MHz, within source six-decimal print precision. Original arrays are never imported by runtime reconstruction.

## Equation/function/test traceability
| Stage | Original source/stage | Local implementation | Independent validation |
|---|---|---|---|
| Cylinder moments | rus.c geometric monomial integrals | physics_model.moment_table | analytic volume, transverse and axial second moments, odd symmetry |
| Monomial symmetry | rus.c basis ordering and mirror decomposition | basis_blocks | complete basis count, two original spectra |
| Elastic energy/mass | rus.c matrix assembly | assemble_block/operators | six rigid modes, positive nonrigid spectrum, sqrt stiffness law |
| Generalized eigenvalues | source spectrum eigensolver | Cholesky whitening/eigvalsh | original nominal and perturbed76-frequency fixtures |
| Manual matching | source user-provided mode list; paper low-mode initialization | match_modes/reconstruct, explicitly adapted | exhaustive small known problem; all original labels over six independent priors |
| LM finite difference/curvature/damping | rus.c fitted-frequency iteration | paired_lm/objective | full original fitted spectrum, moduli and residual |
| All output metrics | native original fit plus independent formulas | independent evaluation/judge.py | known nonzero cases, bad quality and strict contract cases |

The original LM normal equations use unnormalized weighted frequency residuals while acceptance uses relative-frequency chi. This mismatch is faithfully retained and documented, not misrepresented as a single exact-gradient optimizer. Thirty repeated rejected trials raise a diagnostic; no such bound is reached in validated runs.

## Gates and execution evidence
PASSED: original exact-data run and independent frozen fixtures; local full reconstruction; all70 recovered correspondences across six priors; moment/rigid-mode/stiffness scaling physics; standalone fresh Judge; strict missing/key/shape/dtype/complex/NaN/Inf/quality failures; reference precedence; evaluator missing assets; foreign CWD and report failure; active full notebook; semantic plan/equation/interface and import/call-role review; public allowlist leak review. Full native CLI/Judge/pytest/notebook commands, exits, runtimes, input/source/output hashes and kernel argv are in rus_handoff_manifest.json outside this task. Parent acceptance remains pending.

Initial pytest run passed6 numerical tests but nine temporary-directory setup errors occurred because system pytest temp root was inaccessible. Tests now own context-managed temporary directories beneath test/, cleaned automatically; all15 pass without changing scientific tolerances. This is an environment/test harness correction. Initial full nbconvert execution succeeded. Its runtime and exact executed identity are archived separately from the final handoff execution.

Notebook actively recomputes all76 frequencies and both moduli, all observation residuals, original-vs-fresh comparison, order8/10/12/14 convergence, three coarse prior reconstructions and a fresh2D objective contour. The selected manuscript's convergence question is retained; the full order20 study is explicitly outside this selected order14 benchmark. No missing comparison is zero-filled. Displayed figures are inspected individually; figure hashes and review are in the manifest. cleaned_measurements.json is consumed as the notebook's full-precision fresh review record. Upstream reference arrays and independent fixtures have distinct named consumers. No cached reconstruction substitutes for the active fit.

Core dependency/call roles: NumPy array arithmetic, inverses/eigenvalues and NPZ; SciPy general Cholesky/triangular solve/symmetric eigenvalue primitives; Matplotlib display only. There is no imaging/inverse domain library, original import, cross-task import or hidden fixture access in src. Judge imports no solver code. Library imports do not change the Matplotlib backend. Requirements pin only the validated shared runtime subset. All known deviations and native warnings remain recorded above/private logs.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 4 computed PNG outputs, 2 formulas. Private evidence: research_batch12_20261029/acceptance/steel_resonant_elastic_moduli/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
