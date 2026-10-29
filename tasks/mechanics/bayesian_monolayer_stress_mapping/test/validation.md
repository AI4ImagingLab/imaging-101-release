# Validation and provenance: Bayesian monolayer stress mapping

All mandatory gates passed on2026-10-21; ready for independent parent audit. The problem is a full spatial internal-stress inverse from traction observations. This differs from the initial119-task displacement-to-substrate-traction inverse. Mechanics category/reservation was approved by parent. No old task was changed.

## Paper, original source and rights

- Nier, Jain, Lim, Ishihara, Ladoux and Marcq (2016), Inference of Internal Stress in a Cell Monolayer, Biophysical Journal110(7),1625–1635, DOI https://doi.org/10.1016/j.bpj.2016.03.002. Full author manuscript arXiv1603.03694v1 (11March2016) was downloaded and read: https://arxiv.org/pdf/1603.03694v1, private Nier2016_v1.pdf,38pages. Primary HTML https://arxiv.org/html/1603.03694v1 was also verified. We do not claim the fixed default branch reproduces the hierarchical parameter selection or exact numerical results of a paper figure.
- Original author source https://github.com/pmarcq/BISM, remotely verified main commit54d822d6e084a96e4e41c8c6cfe4b60a6c68dbb3. The actual GitHub API response bism_verified_revision.json agrees with cloned checkout HEAD. BISM.m identifies Vincent Nier and dates07/09/17; manual is Nier/Marcq9November2016. GPL-3.0 LICENSE applies to the distributed code/example repository; no separate restrictive data terms occur. Full license/attribution preserved in task LICENSE/NOTICE.
- Original data are Traction_field.mat and Stress_field.mat in that exact source repository. They are author-existing numerical data, not experimental/live-cell data. The original manual page3 explicitly identifies the traction example as a confined viscous finite-element simulation driven by active force dipoles with added5% relative random noise, and identifies Stress_field.mat as the exact stress from the same simulation. The supplied observation and truth arrays are reused unchanged in float64 NPZ. No new simulator, random seed, extra target-derived calibration or synthetic observations were introduced. Source MATs/manual/PDF remain private.

The manual says traction is kPa and height-integrated stress kPa micrometres. Some extra source scatter axes say Pa micrometres, inconsistent with that explicit manual and earlier source L-curve label. We preserve the actual numeric arrays without an unsupported1000factor and use the manual's internally consistent kPa/kPa-micrometre convention. Independent divergence scale checks below support sign and scale, while not proving an exact sampling round trip.

## Coordinate and source ambiguities resolved

The source script calls xmin=ymin1,xmax=ymax99, spacing2, giving the center sample grid1:2:99. The manual gives physical outer boundaries0and100micrometres. The staggered face coordinates extend half a cell, so these conventions are consistent. Numerical flattening is row-major (source reshape(matrix-transpose,N,1)).

The source y-grid comment says rows run top to bottom. However the executed quiver(meshgrid(x,y),mTx,mTy) and principal-stress plot use the numerical y=1:2:99 with default Cartesian upward-increasing axes; no YDir reverse occurs. This source comment/actual display distinction was independently raised in parent review. Final README, both plans and notebook explicitly use increasing row index as positive y and plot it upward with origin='lower', increasing ylim and quiver angles='xy'. No observation or tensor value/sign was flipped. The initial draft downward display was corrected and the entire notebook reexecuted, not merely its wording.

The paper names a reduced prior covariance B and uses B^-1 in the posterior. BISM.m actually builds prior precision I+alpha_xy²D-transpose-D+alpha_BC²E-transpose-E, while its comment misleadingly says covariance before explicitly stating S0^-1=B/s0². Local B consistently denotes the executed precision. No precision was inverted incorrectly to match a symbol name.

## Exact-data original oracle before local implementation

Private run_bism_oracle.py copied exact source data into bism_original_case. The source changes are only fplot=0 and read-only capture statements, recorded as actual unified bism_capture.diff. Graphics disabled skips plotting only; the complete original50² mean and full10200² posterior covariance paths execute unchanged. The capture records A,B,T,complete staggered mean, full covariance, central output, original diagonal-only errors and force-fit results. No mathematical equation, fixed parameter, dimensionality or noise value was patched. Source clear-all behavior is retained. The full original covariance is archived privately, not silently replaced by diagonal-only computation.

Original selected branch: BC1,meth_Lambda3,lambda1e-6,alpha_xy=alpha_BC1000,l2micrometres,noise_value.001kPa. The.001 is the source's chosen conditional uncertainty scale and is not the actual5% simulation noise standard deviation. It does not affect this fixed-lambda posterior mean.

GNU Octave10.3.0 official portable Windowsx86_64-w64-mingw32, BLAS_LIBS='-lblas -lxerbla',LAPACK_LIBS='-llapack'; no GPU. OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS were1. Original first full run16.0141189s and second16.0098508s both exit0 with no numerical warnings. Original reproducibility: maximum face and central differences0.0. First and second command/memory logs and console outputs are separately preserved. No import-only/demo-only gate is substituted for execution.

The original source full covariance reaches peak RSS3385122816bytes in the first run (about3.15GiB), explicitly coordinated with parent. Minimum free physical15011123200bytes and commit35279888384bytes, above required6GiB. Second peak3385081856bytes, minimum free physical14743441408bytes. Each stage was monitored every second; exact samples are in bism_original_command*.json. Original full sparse covariance MAT is1248521032bytes; it remains private, hash-indexed.

The source numeric fixtures bism_frozen.npz and predeclared bism_fixture_contract.json were written before local mathematical implementation. Freeze-capture code initially needed .toarray() for sparse original error fields; this capture type handling changed no original math or data. Frozen mean tolerance rtol1e-7,atol1e-8 and covariance rtol1e-7,atol1e-10 were not relaxed after local tests.

## Local mathematics and traceability

The paper's Gaussian posterior is executed in the original supplied fixed-lambda branch. Other hierarchical, MAP-hyperparameter and L-curve selection branches are not claimed as the shipped standard method. The notebook's explicit lambda sweep is a sensitivity experiment, not a replacement parameter-selection claim.

| Paper equation/stage | Exact upstream stage | Local function | Independent validation |
| --- | --- | --- | --- |
| Eq1–8 force balance and staggered grid | BISM.m Matrices A and B / Ax,Ay | physics_model.operators and force_balance | all original A entries; analytic affine-stress divergence; independent flux differences, adjoint |
| Eq10 prior symmetry and boundary | BISM.m Bdiff,Bbc,alpha_xy,alpha_BC | physics_model.operators | all original precision entries; positive quadratic and explicit D/E energy |
| Eq13–17 Gaussian mean/covariance | BISM.m Stress and covariance matrix estimations | solvers.reconstruct,objective_gradient | full face and center source parity; scalar finite-difference gradient; stationarity |
| Source interpolation and physical shear | BISM.m Data interpolation and stress representation | physics_model.center_stress,interpolation,unpack_faces | independent sparse interpolation and exact source central maps |
| Source original approximate error bars | diagonal Spost blocks/Ssigma pair sums | solvers.posterior_diagnostics diagonal_only_variance | original error-map parity and native covariance samples |
| Complete linear covariance propagation | full source Spost, paper Gaussian posterior | solvers.posterior_diagnostics exact_variance | independent small-grid dense inverse/LCL-transpose; complete original covariance identity/cross terms |
| Physical tensor truth comparison | source Additional figures / exact Stress_field.mat | standalone Judge metrics and notebook | exact true NPZ, analytic metrics, every component wrong-sign rejection |

Local A maps10200auxiliary stresses to5000traction observations. The four row-major blocks are xx verticalfaces,yy horizontalfaces,xy horizontalfaces,yx verticalfaces. Finite differences divide by2micrometres. D compares adjacent face sums without halving, E selects400free-boundary values; B=I+1e6D-transpose-D+1e6E-transpose-E. Objective is l²||A sigma−T||²+lambda sigma-transpose-B-sigma. A general sparse LU solve evaluates H sigma=l²A-transpose T with H=lambda B+l²A-transpose A. No domain reconstruction call, artificial iteration, clipping, hidden alignment or rheological substitution occurs.

Normal center stresses have two weights1/2; center physical shear has four weights1/4. Output is one real float64(1,3,50,50) array reconstruction, orderedxx,yy,xy, unitskPa micrometres. Auxiliary faces are numerical state, not invented physical truth or mixed-reference submission keys. The standalone Judge uses actual author center truth for all quantities.

All original A and B coefficients agree exactly. Local maximum face/original difference1.60403022e-8kPa micrometres and central difference1.20662960e-8 satisfy predeclared scale-aware tolerances; the source/local different sparse factorizations of a precision with weakly regularized nullspace explain small linear-solve roundoff, not a changed model. Local normal-equation gradient norm3.1031e-13. Additional parity covers traction prediction and native covariance samples.

## Independent physical and posterior checks

An independently coded finite-volume flux balance equals matrix A on the actual inferred full face field; an affine analytic stress gives exact known constant force for a rectangular4×7case, testing derivative directions, staggering, units and signs. Adjoint dot products, positive prior energy, central interpolation row sums and objective finite differences pass. Source truth is not merely trusted as a filename: its manual identifies the same FEM simulation and a separate central finite-difference divergence check validates meaningful sign/unit agreement with bundled tractions. Exact closure is not expected because FEM sampling, added noise and the finite-difference stencil differ.

Independent diagnostic values (original full covariance, before local):

{
  "posterior_identity_max_error": 2.220446049250313e-16,
  "max_cross_covariance": 0.15325173577938278,
  "proper_to_approx_variance_range": [
    1.0000064255429408,
    1.9999486277141842
  ],
  "truth_divergence_residual_rms_kpa": [
    0.03547563820250626,
    0.036160551185249266
  ],
  "traction_rms_kpa": [
    0.12396069579380484,
    0.14833307379391059
  ],
  "truth_divergence_cosine": [
    0.9597505464038826,
    0.9722813872112447
  ],
  "unit_scale_ls_fit": [
    1.0607659138440908,
    1.0763752301557679
  ]
}

The independent source H*C-s²l²I residual is2.22e-16 on16selected full columns; nonzero covariance reaches.15325(kPa micrometres)². Source approximate central errors use only diag(C); proper variance is diag(LCL-transpose). Their variance ratio spans1.0000064255 to1.9999486277, so the source approximation is not relabeled exact. Correct propagation is an explicitly additional diagnostic; original mean and source approximate bars remain separately preserved. Local native diagonals and proper variances are computed by batched linear solves without allocating the full dense covariance, validated against original and small-grid independent dense inversion. The conditional Gaussian model is not a calibration of actual FEM/model discrepancy or true5% noise; coverage claims are avoided.

## Assets, truth, metrics and boundaries

data/raw_data.npz contains unchanged traction only. evaluation/ground_truth.npz is the original exact simulation tensor, not inferred stress. There is no baseline_reference.npz. test/fixtures/upstream.npz has the captured original operators, staggered mean, center mean, force prediction, covariance diagonal/submatrix, exact propagated variance and independently calculated truth-divergence diagnostic; numeric comparing tests are its consumer. notebooks/reference_outputs/upstream.npz holds compact identified original outputs for plotting; fresh.npz is separately written by the actively executed local notebook. No MAT, PDF, source checkout, capture log or redundant standalone array is retained in the task.

NCC is per-component cosine similarity without centering; NRMSE is RMS error divided by that truth component's full dynamic range. All six metric boundaries must pass:0.9× original independently rerun NCC and1.1× its nonzero NRMSE. This is an actual-truth task, so source self-comparison does not set thresholds. Actual fresh result:

{
  "status": "pass",
  "passed": true,
  "metrics": {
    "xx_ncc": 0.9677131820553927,
    "xx_nrmse": 0.03219874256209875,
    "yy_ncc": 0.9892072481321321,
    "yy_nrmse": 0.01911766209011334,
    "xy_ncc": 0.9801887227753011,
    "xy_nrmse": 0.03815161942714859
  },
  "thresholds": {
    "xx_ncc": {
      "comparison": ">=",
      "boundary": 0.8709418638620124
    },
    "xx_nrmse": {
      "comparison": "<=",
      "boundary": 0.035418616810291466
    },
    "yy_ncc": {
      "comparison": ">=",
      "boundary": 0.8902865233201596
    },
    "yy_nrmse": {
      "comparison": "<=",
      "boundary": 0.02102942829789748
    },
    "xy_ncc": {
      "comparison": ">=",
      "boundary": 0.8821698505015243
    },
    "xy_nrmse": {
      "comparison": "<=",
      "boundary": 0.04196678136561255
    }
  },
  "errors": []
}

No crop, rescale, orientation/translation adjustment or mask is used. All3components are required and separately tested by wrong-sign submissions. No separate force threshold is imposed on cell-centered output because auxiliary face values are not submitted and native divergence cannot be recovered uniquely from a central tensor. Force consistency remains an explicit source/local stage test and actual notebook diagnostic. Zero and very poor arrays fail normally; missing/key/shape/dtype/complex/object/NaN/Inf inputs and arithmetic overflow produce descriptive error with empty metrics. Missing evaluator assets, invalid configurations and report-write failure return code2. Foreign-cwd paths work and no solver is imported by the Judge.

## Actual execution and notebook review

Shared cleaned runtime: Python3.11.11,NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1 WindowsCPU single-thread; full backend details bism_cleaned_environment.txt. No new shared package or upstream domain dependency was installed. requirements.txt declares only these used runtime packages; pytest/Jupyter are shared development tooling.

| Command family | Seconds | Exit |
| --- | --- | --- |
| cli | 0.351445900 | 0 |
| judge | 0.128573000 | 0 |
| judge_foreign | 0.114843600 | 0 |
| pytest | 1.236408200 | 0 |
| nbconvert | 32.798457800 | 0 |

Exact arrays/cwd/arguments are in final_bism_commands.json with corresponding final_bism_NAME.log. Commands are python -m src.main; evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json; independent foreign-cwd invocation; python -m pytest test/ -v -p no:cacheprovider --basetemp <private>/pytest_bism_final; python -m nbconvert --to notebook --execute notebooks/bayesian_monolayer_stress_mapping.ipynb --output bayesian_monolayer_stress_mapping_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=imaging101-runtime --ExecutePreprocessor.timeout=600. Timeout allows the native7500central covariance linear forms and10200face diagonal solves; measured path is about33s. All31tests pass. Final canonical notebook contains actual active executed outputs; temporary executed output notebook was removed.

The notebook computes the fresh full mean, six lambda reconstructions, no-boundary/no-symmetry ablations and full native covariance diagnostics using only local source. All4actual figure groups were visually inspected after final coordinate correction:1traction maps/vectors and exact/fresh principal stress plus actual force residual;2all three exact/fresh/error maps with common physical scales;3lambda L-curve and truth errors, constraint ablations and shared-bin residual distributions;4component truth scatters with both original approximate and proper conditional error bars, proper SD maps and exact/approximate variance ratios. L-curve annotations were separated and force residual distributions use common bins/log density for readable comparisons. Actual closed-form stationarity and objective endpoints are printed, with no invented iterative convergence.

Fresh/source uncertainty parity is printed in the notebook. Original paper historical metrics are not silently called this run's measured values. Private bism_figures exports PNG/SVG/PDF plus exact per-group source NPZ and SHA256 manifest; experiment_summary.json stores actual metrics/runtimes and is separately archive-hashed. Original extra principal-stress and component/error-bar comparison logic is retained; added uncertainty propagation is deliberately identified as an additional scientific check.

Notebook logs retain nonfatal notebook-cell-ID normalization, Windows Proactor/zmq selector-thread and kernel TCP transport warnings; they are not hidden or called numerical failures. The executed canonical notebook has normalized cell IDs.

README, approach and design were semantically checked against source equations, precision/covariance meanings, factors, face/grid axes, units, selected branch and actual caller signatures. Source/call-role audit bism_callrole_audit.json covers source,Judge,tests and notebook; general sparse LU and eigenanalysis are permitted mathematical primitives. Byte-level Markdown/NB audit does not normalize bare CR away and finds no C0 controls. Public README/requirements contain no GitHub links, actual truth keys/path or privileged observation. No cross-task imports, caches or temporary executed notebooks remain.

## Completion gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Verified paper/full manuscript, exact original author revision and rights | passed | Paper/manual, remote revision response and LICENSE |
| Exact-data complete original run before local, reproducible second run | passed | Two0-exit16s runs; complete covariance; frozen contract |
| Independent source stage/full fixtures and physical checks | passed |31tests, affine divergence/adjoint/gradient, covariance identities |
| Constants, units, precision, boundaries and coordinates | passed | Original arrays/operator equality, manual/plot ambiguity analysis |
| Local explicit acquisition, prior, objective and inverse | passed | A/B/D/E/L, direct Gaussian solve and batched covariance formulas |
| CLI and strict standalone/foreign Judge | passed | Actual commands/reports, every component assessed |
| Malformed/poor/asset/config/report-write/overflow handling | passed | Contract tests with correct codes0/1/2 |
| Public allowlist sufficient and no hidden/source leaks | passed | Self-contained physical README, single observation NPZ and audit |
| Minimal shared runtime and no domain shortcuts | passed | Imports/call roles and actual CPU execution |
| Complete tests and full active notebook | passed |31passed and actual nbconvert exit0 |
| Original/fresh/truth/conditional approximations identified | passed | Separate assets, actual component metrics and4rich groups |
| Plans/source/notebook consistent and figures reviewed | passed | Semantic/signature/coordinate review and actual final PNG inspection |
| Final layout, licenses and archived evidence | passed | Task manifest and private hash index |

## Private archive index and hashes

Private root D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/astro_earth_mechanics. All complete upstream source, raw MATs, papers, captures, original environment, full covariance and logs are private. bism_archive_index.json records the following evidence; bism_task_manifest.json indexes every shipped file. Neither index recursively includes its own hash.

| Archive path | Bytes | SHA256 |
| --- | --- | --- |
| BISM/BISM.m | 18601 | ece5f5e1c584dfec46075f4e31ca83d3469ec9226a3ad95c84f0d2fe6f3f24cf |
| BISM/BISM_manual.pdf | 177519 | 11326b7a495eaaadb4f3d01b7d43f2ba1bf16fed392b031ee74c92efaf779988 |
| BISM/LICENSE | 35823 | 230184f60bae2feaf244f10a8bac053c8ff33a183bcc365b4d8b876d2b7f4809 |
| BISM/Traction_field.mat | 38686 | 1ab0fcf47f16b01ee12dd633f7e0ffe40bf36ffe92eedc8ed5e5078977c6657e |
| BISM/Stress_field.mat | 57542 | a99cc50b51ccc0feaea37807ecb53f11c455bf44caa64bfb981f919054aa5538 |
| BISM/README.md | 332 | 36199495f52e07c27bcb78f2cf5c8601208e5f2c75396b15478159e995c81748 |
| Nier2016_v1.pdf | 7065431 | e843dda061b8fcc2c9d2bd80466e85ace72f870275d4796f1cb07808f806b53f |
| bism_verified_revision.json | 4809 | 69ecb72b2be74a58ffb756347fea125bbd7e76f94c0d72e00516776d665030a6 |
| run_bism_oracle.py | 3607 | dec01cac1827c0341d0be41c0a90c72dd944190efb367021a11446f54ed8843e |
| bism_capture.diff | 963 | db1c43f37576e166bcb27f4ff8e2be8c0a15762a2247010cb436306e8c5cb405 |
| bism_original_case/BISM_capture.m | 19081 | 33d6797534ca873259924ce2ff294b8470aab73ef189d93fa462d2bd354bd368 |
| bism_original_case/bism_stage.mat | 220721 | 971a9f9ff4d4bf642d6e5ce3bd51c1b4496329be2ce5065468865d3d4a1d919d |
| bism_original_case/bism_output.mat | 135799 | 5a059d9afb5dd1a3c6b484283ed5024fcd26334ca1ef307dd13d699cf629bebc |
| bism_original_case/bism_full_covariance.mat | 1248521032 | 3d562a48dbb08dcc2c8ad234fbf21dedaea0275b82fc2cb6950d05c798cc8399 |
| bism_original_command_first.json | 4024 | 6d1b69277d24704fab915435b0fd5307a533493f00046f113e35f2ca5593f5e7 |
| bism_original_command.json | 4022 | f80327b205e2461c7eaf510728f8e74e992623dc5abbeb08f7af7cfc49b92ed9 |
| bism_original_console_first.log | 45 | 6fbf92360248d4a69c575374c1ca033741e8d180a303f0e0082e55ffe58d7b7c |
| bism_original_console.log | 45 | 6fbf92360248d4a69c575374c1ca033741e8d180a303f0e0082e55ffe58d7b7c |
| bism_original_reproducibility.json | 67 | ffb4b576817cea686f2f726e0bc39300f8bfd526cc29420875905b2e85150724 |
| bism_original_environment.txt | 296 | db6bbaf9a73bdda73e105d890185796adf0387737b1d1551d4b7e5f038e6f30d |
| bism_cleaned_environment.txt | 2427 | 2055bd485d79c0a9abfe545d53f2350813c16ca713c197fffeb1356c4276efe8 |
| freeze_bism.py | 4599 | 78c2c887b1009f2ed9a42795217a1a5f0ad474acc519be5a94df3104887166ab |
| bism_fixture_contract.json | 566 | 6a7bb8e96e17e82b321a816fe75395807721e13b6909cf9b71724f5ddd259118 |
| bism_frozen.npz | 578348 | 73e9d01db54100947a986f6df570859276b85ba3f6ced1251f75e7125a6fa02a |
| bism_independent_diagnostics.json | 554 | 535c572865ee2a4cd52f0bbcf9f3b28ada80417b731ee943a967e8a2f65aa136 |
| build_bism.py | 9256 | 78654bdc162d839d507e13a3baeaaf695f59843e925b42217328adb5de9a9f99 |
| docs_bism.py | 18931 | 2a31d99e8c8e91da9aa57746edef16b4ccf0fc5bd144052fc308029b1f239bd9 |
| notebook_bism.py | 17325 | a8129803407297a1cf6bbba0ba0715cdca90924c30d5445a8b0f4bb188c35b8f |
| verify_bism.py | 1869 | 96c1a6cc333f8cb13cd05de677d7563e93defb389a4f6c3bf35182063afe29f9 |
| bism_judge.py | 3469 | 18b43b2de60d0b3cc1e2ba478486839bea35a7325f8d65ece10065ebe1e4becf |
| bism_tests.py | 5346 | 953efcfad9e5f7445085ec7d7a2a0a1f8fb42b17fa1ee1a56290198a1874935f |
| bism_contract.py | 4067 | 50d7d25ca87018ccfa3cf7b1d4ec010d2323aac4119a3eb9514be6416c94a84b |
| final_bism_commands.json | 2843 | 53f767a333878b8d2d1a73d38d0a496a376991ece9d107a1517edaa2b6292e38 |
| final_bism_cli.log | 117 | 879b16062d97f08f8b4595da9727bc6787c5623136ffbcbd04ff3f7a605f067d |
| final_bism_judge.log | 6 | e0ba8b72f361b6f4f71dc78a0b7a570475bdd44e176caad604ba134208475403 |
| final_bism_judge_foreign.log | 6 | e0ba8b72f361b6f4f71dc78a0b7a570475bdd44e176caad604ba134208475403 |
| final_bism_pytest.log | 3241 | 21f282ddcb598bb756b02db4f32d38043f7d313d6f09a40ea930a0bb3fc8ef99 |
| final_bism_nbconvert.log | 1373 | 46ada88b84f509eb3e4d2052d4389f8ea08cbfde570c2437d76ef420a105a3d4 |
| bism_figures/manifest.json | 2399 | 6dc5464c48c50381f1c8fb46954a77ca037c8d027b64e5ebe35fabb1f3303294 |
| bism_figures/experiment_summary.json | 2777 | 46368f4da887513773f86e8541f474ff5dc702842e035ffc5223b6c409af66a3 |

## Task array hashes

- data\raw_data.npz: 25535f34119cb63bfdf982331611c766f9db1308aaa41ed2ff1305a2ae4e6ae1
- evaluation\ground_truth.npz: cbf058073310a516b49cc1eaa0b432bae2881cf8f6f6a31827f64dcd47648c44
- test\fixtures\upstream.npz: 73e9d01db54100947a986f6df570859276b85ba3f6ced1251f75e7125a6fa02a
- notebooks\reference_outputs\fresh.npz: 6d6174edf21bcd261757ca6d51e5a7653ad88884ba340967361339b7bc89b547
- notebooks\reference_outputs\upstream.npz: 220b4993efd399922466b37c8cdcc9492882b6cbf392529f2133e7d63d51552a


## Parent acceptance review

Parent independently ran31tests (31 passed in0.97s), foreign-cwd Judge0 and source/inventory audit errors[]. Read explicit staggered divergence, four-face auxiliary fields, symmetry/boundary precision, Gaussian sparse inverse and center interpolation; inspected all4 actual scientific groups after coordinate review. Original complete10200-square posterior covariance and mean twice executed, independent H*C identity and small-grid dense calculations verified. Author-existing noisy FEM traction recovers all three center stresses against genuine supplied simulation truth, not a reconstructed reference. Correct full L*C*L-transpose uncertainty is separately labeled from original diagonal-only approximation and assumed.001kPa noise scale. Source row comment versus actual Cartesian y plotting ambiguity resolved with +y increasing row, origin lower, no data/component sign flip. Six lambda fits and boundary/shear ablations actively computed.

## Final boundary-schema compatibility correction — passed

Authorized final schema-only correction on2026-10-21: evaluation/metrics.json now stores the sole canonical numeric boundary for each metric under its top-level <metric>_boundary key, with directions in metric_comparisons. Judge read_thresholds actually reads these named fields and derives its report thresholds; no unused compatibility alias or duplicate numeric boundary is retained. Boundary numbers, comparison directions, acquisition/inverse algorithms, observations, truth/reference arrays and output contracts are unchanged.

New test_boundary_schema.py proves active consumption by changing a canonical NCC boundary and observing a valid failure, and proves missing boundary, missing comparison and unexpected metrics return evaluator error. These complement the existing per-component, physical/parity and malformed-submission tests. The pre-change Hawaii schema test was run and failed as expected; the migrated schema then passed. This was a harness-field compatibility correction, not a new scientific method or relaxed acceptance.

Protected-file hashes prove all src numerical files, README, both plans, raw observations and hidden NPZ references are byte-identical to the pre-correction snapshot. All submission and notebook fresh numerical arrays except intentionally varying runtime_seconds are exactly array-equal to the previous accepted run; fresh measured metrics and Judge report threshold values are exactly unchanged. Previous README formula/Cartesian-coordinate corrections remain intact. Actual final reruns:

| Command family | Seconds | Exit |
| --- | --- | --- |
| cli | 0.390230300 | 0 |
| judge | 0.160057300 | 0 |
| judge_foreign | 0.150961200 | 0 |
| pytest | 1.723807200 | 0 |
| nbconvert | 35.954125700 | 0 |

Complete pytest result: 36 passed. Full active nbconvert completed with no error outputs; canonical notebook retains actual executed outputs and the temporary executed copy is absent. Exact commands/cwd/logs remain private final_bism_commands.json and final_bism_NAME.log. Private boundary_schema_verification.json records protected hashes and numerical equality; the pre-change snapshots are boundary_schema_before/bayesian_monolayer_stress_mapping. All figure exports/numeric sources were regenerated by the active notebook and manifests refreshed. Original oracle assets were not rerun or changed for this schema migration. Parent re-audit of this final schema remains separate from the earlier scientific acceptance record above.

Latest schema/fresh artifact hashes (the earlier tables above identify their earlier accepted runs):

- evaluation\metrics.json: dd31cf8d4c88c2af8f8da0a812e3b7caf8b9d8906489d39fe9b34faf11f04d76
- evaluation\judge.py: 0422b9f1121b68bb8152bf6b2186e442b26218df044cce57682e6439cfe1d567
- output\reconstruction.npz: 2a20d91e6a22dfec1956f3f74b51522b20a9e60547764ff72ba0c64c9e6a94ff
- notebooks\reference_outputs\fresh.npz: 6d6174edf21bcd261757ca6d51e5a7653ad88884ba340967361339b7bc89b547
- notebooks\bayesian_monolayer_stress_mapping.ipynb: 9946df954bff5e90718110bea2c2eb0e48e688615a6f4ae0074f52200fb8f954
