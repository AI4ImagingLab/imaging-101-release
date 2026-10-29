# Provenance, scientific validation and gate evidence

Worker status: READY for independent parent acceptance, not parent-accepted. No release-directory insertion or acceptance-ledger update is performed by this worker. All numerical gates below are measured; limitations are not treated as truth.

## Binding and rights

Physical unknown: the complete nonnegative spatial fluorescence yield eta*muaf (mm^-1) on all 5,707 original MRI-derived tissue nodes. Observations: real, author-offered mouse-head fluorescence and excitation intensities for all 56 ordered active source-detector pairs. This is not simulated data, raw spectrometer spectra or an independently measured molecular map. Full 3D tutorial geometry contains 29,448 tetrahedra. The fixed geometry-only coarse basis has 3,323 nodes and 17,350 tetrahedra; it derives solely from the author20^3 grid/tissue geometry, without fitted values. Shipping its tessellation preserves ambiguous regular-grid triangulation choices. P1 interpolation, source loads, FEM physics, sensitivities and inverse updates are all computed explicitly locally.

Paper: Davis et al., Comparing implementations of magnetic-resonance-guided fluorescence molecular tomography for diagnostic classification of brain tumors, JBO15 051602 (2010), DOI10.1117/1.3483902, PMC2951993. Full author-institution PDF (10pages) was retrieved from https://www.ohsu.edu/sites/default/files/2018-10/magnetic-resonance-guided-%EF%AC%82uorescence-molecular-tomography-for-diagnostic-classi%EF%AC%81cation-brain-tumors.pdf and read, including methods2.4, all comparison groups and discussion/conclusion. The selected baseline is the paper-supported unguided fluorescence tomography family implemented by the2018 author tutorial, not an exact reproduction of its 2D study, original optical-property numbers, hard/soft-prior alternatives, animal-population statistics or ROC claims. The tutorial explicitly selects the3D20x20x20 reconstruction basis. Its complete13page PDF was inspected separately. The notebook states why population and anatomical-prior comparisons are outside this bound one-acquisition baseline.

Code: https://github.com/nirfast-admin/NIRFAST ; actual remotely verified commit9bbb66a4a63182ccef432bf8dfb8d68e4fa24a9b (saved live commit/repository/tree API records, not a claim based only on a local tracking branch). Complete source ZIP, full downloaded data and source license remain private. The extracted NIRFAST mathematics is BSD3, Copyright(c)2003 Dartmouth College NIRFAST Group. The exact applicable notice is retained in LICENSE. No Triangle/TetGen or other upstream meshing framework is shipped or imported. Code qualification below is explicitly disclosed rather than called unchanged reproduction.

Data: the public author NIRFAST sample collection, Kitware Girder collection571697a38d777f4fe2c91693, tutorial folder5ac049858d777f0685791d65, raw optical file5ac04e188d777f0685791d82 (Mouse_head_data.paa,5810bytes, publisher SHA512223f760c47bd36128afbe87f54c018de7e76bd31201d3392336961c3eb4d650b499759bfac917379cff9ef2004e9229f8da00446854db97435fd799bbb1062d7). Optical source, mesh ZIP and each unpacked mesh file have full SHA256 in nirfast_archive_index.json. The source file contains extra intrinsic/phase columns that are not inputs to the selected CW fluorescence pipeline; only its exact measured excitation and fluorescence columns and all active pair rows are bundled. No synthesized noise, synthetic target, output-dependent crop or interpolation of observations is used.

Data redistribution rights are separately unspecified. Inspected collection metadata has empty license metadata; the tutorial offers sample files for its walkthrough but no separate public redistribution license. NOTICE quotes the tutorial's sample-download statement and preserves this limitation. Software BSD and article rights are not asserted to license data. Work is within the authorized local research/benchmark scope; no external publication/upload is performed. This limitation must remain visible before any future external release.

## Original run, environment and chronology

The actual unmodified MATLAB/C source was executed on the exact bound real intensities and author mesh using isolated native GNU Octave8.4.0/WSL Ubuntu24.04 and privately compiled original C MEX files. No shared environment packages were installed or upgraded. Original command is recorded in nirfast_original_probe_run.json; native exit0,145.2463656seconds. The original run completes, but its boundary-physics gate fails as documented below; it is not accepted as physically correct merely because it executes. Original outputs and warnings remain in oracles/nirfast_original_run and nirfast_original_probe.stdout/.stderr.

An Octave/MATLAB reader compatibility adapter tokenizes importdata text headers only, retaining numeric parsing. The original header reader otherwise misclassified the source table because Octave returns one whole header string while MATLAB returns tokens. This is independently proved in nirfast_reader_compatibility.json; the adapter does not change coordinates, optical coefficients, interpolation or reconstruction. Failed attempts using the wrong header convention, already-initialized mesh or numeric rather than actual GUI struct lambda are archived under nirfast_failed_* and are not passed gates. Actual GUI lambda is Automatic/value10, maximum40iterations, no mean filtering. Read/write/header compatibility is distinct from the mathematical Robin qualification.

Native solver backend and compilation configuration are captured in nirfast_oracle_backend.json. General source backslash linear systems execute native Octave/SuiteSparse numerics, not a private Python imitation. CPU: Intel Core i9-14900KF; original and cleaned runs CPU1, OMP/OPENBLAS/MKL thread variables1. Cleaned environment: Windows, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1; exact platform/BLAS runtime data in nirfast_cleaned_backend.json. The compiled C emits original unused nonvoid-return warnings; independent matrices/fields verify actual numerical buffers. WSL networking/runtime-directory and Jupyter Windows ZMQ/transport warnings are archived; none is called a scientific failure or silently suppressed into a success claim.

Before qualification runs, measured free physical memory was8.30GiB, free commit21.74GiB and Dfree98.9GiB, above mandatory minima. Oracle runs were sequential CPU1 and modest-memory. No GPU is required. The cleaned task imports only standard library/general NumPy/SciPy/Matplotlib. Domain packages, oracle MEX files and upstream source are never imported by source, Judge, tests or notebook.

## Independently proved Robin correction

The original gen_matrices_3d.c boundary_int contains a duplicated last branch for vertices0,2,3, omits1,2,3, and selects only the first matching vertex-flag face. Actual topology contains3,624 exterior faces; original selections number3,587, including364 interior triangles and omitting401 true exterior faces. There are175 tetrahedra with boundary pattern0111 and192 with all4boundary vertices. Independent exact P1 volume plus the source-selected Robin triangles reproduces the original matrix at relative3.84e-16, identifying the precise cause rather than guessing a numerical backend issue. Original constant-field Robin flux differs by0.2194%; physical matrix differs by0.8315%.

The private qualified C change identifies faces with multiplicity exactly1 and integrates every such face using the unchanged P1 triangle mass/Fresnel coefficient. It changes no PDE, mesh, source/detector, bulk coefficients, calibration, gamma parameterization, inverse update, regularization or stopping rule. Full exactdiff is nirfast_robin_qualification.diff, with original/modified SHA256 in nirfast_robin_qualification.json. Qualified constant-field flux relative error1.22e-16 and independent full P1 matrix relative error3.84e-16 passed BEFORE its complete native runs. The original count13was never imposed: the qualified run independently happened to stop at13evaluations after12updates.

Qualified commands and native results:

| Run | Exit | Runtime seconds | Log |
|---|---|---|---|
| nirfast_qualified_probe | 0 | 148.97101639999892 | `nirfast_qualified_probe.stdout/.stderr` (same prefix in archive) |
| nirfast_qualified_quality | 0 | 11.829893100002664 | `nirfast_qualified_quality.stdout/.stderr` (same prefix in archive) |
| nirfast_qualified_stages | 0 | 12.870037200002116 | `nirfast_qualified_stages.stdout/.stderr` (same prefix in archive) |
| nirfast_qualified_repeat_probe | 0 | 146.25769499999296 | `nirfast_qualified_repeat_probe.stdout/.stderr` (same prefix in archive) |
| nirfast_qualified_repeat_quality | 0 | 12.10205830000632 | `nirfast_qualified_repeat_quality.stdout/.stderr` (same prefix in archive) |

All5,707 reconstructed yield values and the13-entry projection history are bitwise identical between the two independent complete qualified native runs; see nirfast_qualified_repeat_parity.json. The native quality capture explicitly enables fluorescence (mesh.fl=1), since the source reconstruction's final femdata call otherwise returns excitation only. Qualified all56 forward relativeL2=.18907149090940267 versus freshly input-fitted uniform=.45029970956931437 and zero1. Full log SSE decreases8.258469772411354→2.327039105194606, preserving every evaluation. Original incomplete-boundary result has forwardL2=.200961261963587 versus its own uniform=.462232015305044; this is labeled a deficient-source comparison, not biological ground truth.

## Independent inexact-Jacobian physics

Original gen_source_fl interpolates the nodal product gamma*phix and applies the P1 mass matrix. Original IntFG_tet4 instead integrates the continuum triple-P1 product; its adjoint additionally includes exp(-0.15i). Local code retains that published/source inexact scheme. Exact simplex moments reproduce the qualified native pre-Jacobian to relative4.4569933745e-11. The independent physical mass-chain derivative agrees with native central finite differences to relative6.2694614e-13. Its source phase-projected counterpart differs by.01122892206 (1-cos.15); the continuum-versus-mass-chain directional discrepancy is.01211934658. Constant fields agree; an affine single-element refinement from h1,.5,.25,.125 gives errors.07732,.04152,.02154,.01097, demonstrating first-order consistency of this element diagnostic, not a false claim of an exact discrete gradient. Detailed arrays/derivations: nirfast_qualified_independent_jacobian.json and independent_jacobian.npz in the qualified source archive. The notebook independently visualizes actual rowwise sensitivities; larger rowwise norm discrepancies do not contradict the specified directional check.

## Frozen criteria and local extraction

The complete qualified outputs, stages and criteria were frozen in nirfast_binding_frozen.json before any local src directory existed. Frozen source fixtures contain native matrices, excitation/emission/adjoint fields, log/pre sensitivities, calibrated intensities, geometric interpolation and projection history. Later packaged directional-physics fixture consists exclusively of previously captured native direction/central-FD/adjoint arrays, not values generated by cleaned code. Its source capture preceded local extraction.

Pre-extraction limits: NCC≥.999999; dynamic-range NRMSE≤1e-6; maximum absolute error≤1e-10mm^-1; full56 forward relativeL2≤.191. The actual reference is full-precision native MAT, not the six-significant-digit text solution. Native-format roundtrip measured max4.9798254264e-11mm^-1 and DR-NRMSE2.9717864256e-7, providing a nonzero observed representation envelope; chosen limits exceed it by factors2.0 and3.36. Bitwise native reruns alone are not a tolerance rationale, nor are input perturbations used to loosen fixed-input reproduction. Exact P1/flux and mass-chain checks independently verify mathematical scale. No tolerance was changed after local execution.

Local first complete result matched allnodes at maxabs1.62e-16mm^-1 and DR-NRMSE1.33e-12, with the same13-evaluation stopping and forward.1890714909079. Stage tolerances stayed fixed: matrix relativeL2≤1e-12, fields≤1e-10, continuum sensitivity≤1e-8, geometry interpolation maxabs≤1e-9, calibrated intensities relativeL2≤1e-10. Local P1 topology, source/detector barycentrics, source phase, mass source, cubic adjoint moments, finite-difference derivative bisection, sparse solves, relative-coordinate data-space damping and2%stop are explicit executable code. No opaque vendored domain framework or alternate easy inverse is used.

## Equation/function/test traceability

| Paper/stage | Upstream | Local | Independent/numeric consumer |
|---|---|---|---|
| Diffusion model, Robin weak form | gen_matrices_3d.c; femdata_fl | assemble_fem, exterior_faces | test_native_fem_operators; single-tetra mass/Robin conservation; independent boundary qualification |
| Source/detector physics | gen_source_point; get_boundary_data; gen_source_adjoint | geometry_operators; barycentric_locations | test_native_geometric_interpolation_and_forward_stages |
| Excitation-normalized measurements | calibrate_fl | reconstruct calibration | native calibrated intensity fixture, all56pair parity |
| Data-driven uniform initialization | calibrate_fl derivative bisection | initialize_uniform | full pipeline uniform_muaf fixture; active initialization figure |
| Continuum adjoint, tripleP1 moments | prejacobian_fl; IntFG_tet4 | continuum_prejacobian | native stage comparison, exact cubic moments, exact mass-chain/nativeFD distinction |
| Coarse spatial representation | pixel_basis; interpolatef2r_fl/interpolatep2f_fl | geometry_operators; C/B matrices | geometry-only native mapping fixture, known smooth test field |
| Regularized update, paper Eq1 family | reconstruct_fl JJt branch | reconstruct relative-coordinate update | complete field and every objective evaluation parity |
| Native positivity/stop | reconstruct_fl | separate gamma/yield; source2%rule | complete pipeline physical range and exact stopping checks |
| Full evaluation | no inherited solver scorer | independent judge.py | known analytic metrics, truth-priority/no-fallback, invalidNPZ/config/CLI/report-write tests |

## Numeric assets and consumers

| Packaged asset | SHA256 |
|---|---|
| `data/raw_data.npz` | `c0ad7de1ab6a7d24d4e33b1bfcb8c451976bfbd1e884bb23da23e648426c06c9` |
| `evaluation/baseline_reference.npz` | `03490672769df1c537619a6894685207c2f494500dd0a2278c458c373c55f8ca` |
| `test/fixtures/upstream_stages.npz` | `f0c6596df5fae68ad09611c2b7fe8716cc841aef1b1e1a2dd615287984c9d724` |
| `test/fixtures/native_directional_physics.npz` | `1d104e404df781ee00a62bacd5b15487c3455edb080f2400f3266f0ef71ca8aa` |
| `notebooks/reference_outputs/original_incomplete_robin.npz` | `5baafcfcc320944c045e2693ff45952f5313e95043db1aa467c51796efba51e4` |
| `notebooks/reference_outputs/qualified_upstream_measurements.npz` | `c2147c7336e4ec9f5029f177c59a4ae06e9bfdffd288a646c23379ae6cdf0847` |
| `notebooks/reference_outputs/fresh_cleaned_measurements.npz` | `e4c3369a0339cb5f9f89d9f8818dad283b6947c7e3b7b202b0cf800015ec538d` |
| `notebooks/mouse_head_fluorescence_yield_tomography.ipynb` | `0e02e5ec6c07c3c3d84109e060918277f82897d7eb1fdfd219f92a9c5be818e3` |

upstream_stages.npz is consumed only by comparing tests, with all arrays identified above. native_directional_physics.npz is consumed by the exact mass-chain/nativeFD test. baseline_reference.npz is consumed by Judge, full pipeline parity and source-comparison notebook; it is explicitly a method-reference, not truth. original_incomplete_robin.npz is consumed by the failed-source field/history comparison. qualified_upstream_measurements.npz contains native all56prediction/uniform/observation arrays and13objective entries for notebook comparisons. fresh_cleaned_measurements.npz is actively overwritten only after the notebook's fresh reconstruction, providing separately identified fresh prediction/objective/metrics to the review package; it is never used as reconstruction input or substituted for fresh computation. No unexplained duplicate full field is retained under reference_outputs.

## Final commands, interfaces and actual notebook

- pytest: native exit0, 10.201481399999466seconds; command/cwd in nirfast_final_checks.json, full stdout/stderr under nirfast_final_pytest.*
- main: native exit0, 2.201716599985957seconds; command/cwd in nirfast_final_checks.json, full stdout/stderr under nirfast_final_main.*
- foreign_judge: native exit0, 0.7330963000131305seconds; command/cwd in nirfast_final_checks.json, full stdout/stderr under nirfast_final_foreign_judge.*

The final full test run contains35passed tests, native0; JUnit and complete log retained. Standard python -m src.main writes exact float64(1,5707) reconstruction. Foreign-cwd standalone Judge exits0. Independent truth priority and invalid-truth-no-fallback are tested using analytic arrays without changing bound assets. Invalid missing/key/shape/dtype/complex/object/NaN/Inf inputs, poor valid predictions, one-node max-error failure, undefined/invalid thresholds and report-write failures are covered. JSON is UTF-8 with finite numbers and contract exit codes0/1/2. No solver imports occur in Judge.

Final canonical notebook is the actual native nbconvert output: exit0, runtime 41.918485199974384seconds, timeout1200, exact pinned shared-Python private kernelspec, CPU1, SHA256 `0e02e5ec6c07c3c3d84109e060918277f82897d7eb1fdfd219f92a9c5be818e3`. All code cells actively execute the bound reconstruction, independent Judge, full56 residuals, source comparisons, original boundary-defect comparison, five actual regularization settings, exact/inexact sensitivity comparison, matched uniform/zero nulls, all-node volume and orthogonal P1 tissue slices. All13actual figure groups were opened and inspected; figures have private PNG/SVG/PDF plus pairedNPZ/JSON and hashes at `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch8_20261022\groups\bio_med\nirfast_figures\figure_index.json`. Figure1 mesh visibility, Figure3 log scaling and Figure7 title spacing were corrected and the full notebook reexecuted. Prior visual revisions are superseded; arrays were compared elementwise and unchanged except adding explicitly native measurement markers. The final optical-coefficient wording identifies assumed model inputs without unsupported independent-measurement claims. Earlier markdown-only revision records are history, not the current canonical output hash.

README/approach/design and notebook mathematics were semantically reviewed against actual source: background coefficients are exact mesh values, source .15radian CW/adjoint conventions preserved, emission mass-chain distinct from continuum preJac, no forced13stop, no lambda decay/filter/prior, correct units and relative-coordinate update. All required public inputs are in the allowlist; no reference, hidden initialization, target mask, molecular truth or public GitHub URL appears in README/raw/requirements. README and plans contain no shell/PDF dependency. Final dual KaTeX/MathJax review is recorded privately; current count102formulas, zero failures.

## Gate ledger

| Mandatory gate | Result | Evidence |
|---|---|---|
| Primary paper/source revision/data/right notices | PASS | Full method/tutor source and indexed SHA/NOTICE; separate data-rights limit retained |
| Original exact-observation execution | PASS execution; FAILED original Robin physics | Original native0 retained; no unchanged-physics success claim |
| Qualified complete source/repeat | PASS | Minimal source diff, independently verified full topology/flux, full native0twice |
| Independent fixtures/physics frozen before local | PASS | Source MAT capture/freeze chronology and mass-chain/continuum/topology checks |
| Local explicit math and minimal runtime | PASS | All stages/fullfield parity; no source/domain imports |
| Standard CLI and strict independent Judge | PASS | Native0, foreigncwd0, full physical consistency |
| Unit/parity/physics/NPZ/metric/Judge contracts | PASS | 35tests native0 with archived JUnit |
| Active complete notebook/actual comparisons | PASS | Native0 and13actual figure groups with paired numerics |
| Public contract/math/plans/source agreement | PASS worker review | Complete interfaces and equations checked; parent independent review pending |
| Final layout/notices/inventory | PASS worker review | Task inventory and private cache archive; no original PDFs/checkouts in task |

Remaining scientific limits: no independent spatial truth, only56channels for a high-dimensional volume, assumed optical coefficients, diffusion-model mismatch, target-normalized intensity scale, conditional inexact solver and geometry-basis fallback. The model reference proves method reproduction, not tumor accuracy. Separate sample-data public redistribution rights remain unspecified. Parent acceptance is not asserted by this worker document.


## Parent independent private acceptance

Parent independently reran35tests: native0/9.077919700s. Both mathematical renderers passed102formulas. Parent reviewed the acquisition/FEM/continuum-sensitivity/inverse source, independent Judge, full provenance, both plans and six-part notebook, and actually opened all13scientific PNG groups individually. The canonical Notebook is byte-for-byte identical to the archived actual successful native execution (0/41.9184852s), SHA256 0e02e5ec6c07c3c3d84109e060918277f82897d7eb1fdfd219f92a9c5be818e3; all code execution counts are present and no error output exists. Original incomplete Robin boundary assembly and qualified full exterior-face integration remain distinguished. The preserved continuum sensitivity is explicitly inexact for the nodal-product source, with independent exact derivative/refinement evidence. All5707nodes and56channels are retained, with forward relativeL2.18907149 versus uniform.45029971. No molecular truth, tumor diagnostic accuracy, or independent optical-coefficient calibration is claimed. Separate data redistribution rights remain unspecified. This records private staging acceptance only, not batch8release.


### Public source-link separation, 2026-10-29
Removed the direct public URL https://www.nirfast.org/ from the solver README. Role: standard domain-software project/tutorial site. The scientific citation, dataset attribution and license remain; complete source/data provenance stays in this maintainer-only validation record. The released NPZ supplies every observation/calibration input, so no software archive is required by the solver contract.


## Current benchmark contract v2 (2026-10-29)

The task now asks for a nonnegative measurement-consistent conditional field, with no unique spatial truth certification. The independent P1 FEM evaluates all 56 calibrated observations; mean squared log-residual excess <=1e-6 and the original relative L2 <=0.191 remain. GN damping 10 is a numerical update setting, not a static Tikhonov prior. An independent complete 56x5707 adjoint sensitivity matrix and NNLS-derived improved feasible field pass; zero/scaled/spatially permuted fields fail. The source reconstruction is unchanged.

Historical pixel/parameter identity thresholds and prior notebook reports refer to v1. The external benchmark_readiness_20261029 archive preserves old README/Judge/metrics/tests and actual current CPU command logs, including failed diagnostics. Source algorithms and arrays were not changed.
