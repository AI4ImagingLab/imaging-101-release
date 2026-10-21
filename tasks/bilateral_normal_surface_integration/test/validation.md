# Authoritative provenance and validation: bilateral integration

## Fixed problem, original source and observation rights

This is a mathematical geometric-imaging task: reconstruct both complete native orthographic normal fields, vase168×168 and tent128×128, with the published bilateral integration. The two normal fields are **author-provided analytic examples**, not real optical measurements and not newly generated synthetic observations. The earlier DiLiGenT option was not bound: redistribution rights were unresolved and its supplied normals could be derived from geometric truth rather than independent optical measurements. Source-own provided examples are the explicit fallback allowed by the user's real-data-first preference; this task is not counted as a real-data task.

Paper: Cao, Santo, Shi, Okura and Matsushita, Bilateral Normal Integration, ECCV2022, official16-page proceedings version https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610545.pdf . Required orthographic tangent equations Eq3, one-sided discretization Eqs16–18, bilateral objective Eqs19–22, frozen-weight iteration Eq23 and normal equations Eq24 were read and checked against the source and local algebra. The perspective branch, depth prior and full six-external-method comparative study are explicitly outside this bound task; no complete paper-benchmark reproduction is claimed.

Repository: https://github.com/hoshino042/bilateral_normal_integration; remotely API-verified revision **93ad6d5b2fdc4d1eeafaaa4c4ae1251a039ec216**, full archive source.json/remote_commit.json/upstream.zip and original readable reference are private. A git transport timeout is not claimed successful or treated as freshness evidence. Selected source is `bilateral_normal_integration_simple.py`, modern readable version2.0, not the original2022 binary/environment or accelerated NumPy/Numba/CuPy alternatives. Copyright(C)2022XuCao and GPL3 notices remain in extracted modules and complete LICENSE. The selected `data/supp_vase` and `data/supp_tent` are source-own analytic fields under this distribution, avoiding separately credited third-party meshes/scans.

Observation decoding is original cv2 read-unchanged, BGR->RGB channel reversal, then16-bit q/65535*2-1 in float64; mask PNG values become bool. No renormalization, blur, target depth, random seed, analytic height formula, texture or synthetic-corruption pattern enters public observations. Both masks are complete grids. Existing observations are reused, so no generate_data.py is present. Full original PNGs/source/checkouts/PDF/capture scripts/logs are maintainer-only, outside the task. Heights have pixel-pitch units, not invented metric distance. The additive gauge is fixed by an observed-domain mean, never true depth.

| Asset | SHA256 |
|---|---|
| exact bound data/raw_data.npz | 9d0c655500b64269539b14af41549529aca77c325cc71d13e37b312d043e9dfb |
| independent original initial capture | 7b18be24252ad5f9c80c443b806a00559f7d143fcaf8abdd0622cbc385d052d9 |
| test/fixtures/upstream_stages.npz | 7b18be24252ad5f9c80c443b806a00559f7d143fcaf8abdd0622cbc385d052d9 |
| hidden standard-method reference | 652feb9458224af5a5b261afe516c56e09c04c219405ec5f3a8df6f6f39d9506 |

Private archive root `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\math_physics\bini`, indexed by archive_index.json with file sizes/SHA256. Root capture/assembly scripts are `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020/capture_bini_oracle.py`, prepare_bini_task.py, continue_root_oracles.py, finalize_bini_docs.py and execute_root_notebook.py; these are capture tools, never imported by the task. Exact source commit, original raw-array shapes and source/download hashes are preserved independently from final local source hashes.

## Original exact-bound execution before local extraction

Command: private oracle-env/Scripts/python.exe capture_bini_oracle.py, from the repository workspace, single CPU thread for OMP/OPENBLAS/MKL. Private original source imported unchanged; PyVista is used solely because the source returns a mesh, and cv2 decodes the original PNG. Both full normal fields are passed to the original routine before any local src implementation. Parameters are fixed before extraction in success_criteria.json: vasek4/tentk1, max100, relative-energy tolerance1e-4, CG rtol1e-3 and source max5000. Original code captures decoded inputs, encoded bytes, every CSR derivative's data/indices/indptr, raw depth/gauge-corrected depth, both directional weights, mesh vertices and complete energy trace. No cleaned-generated output supplies parity expectations.

First original run: 2.267504700s; vase43 iterations, energy111.51781323101733, tent70 iterations, energy84.943569608958256. Source exp can overflow at saturated0/1 logistic weights; actual arrays remain finite in support. The original source intentionally stores NaN outside support, but both bound masks are full. Mesh outputs remain private; no domain renderer is needed locally. Original warnings and progress on stderr initially caused PowerShell's native-command error presentation despite Python recording completed output. The original run was therefore independently repeated through a Python subprocess; oracle_rerun_exit.json records actual exit0,2.7495032s, stdout/stderr UTF8 log. Every captured array is exactly equal across those two independent original runs (`repeat_equality.json`). No source formula or threshold was changed to accommodate shell behavior.

Original and cleaned environments: Windows, Python3.11.11, NumPy2.4.6, SciPy1.17.1 and same numerical backend; environment.json records current unchanged installed backend and interpreter roles. Private oracle package/dependency installation logs are at math_physics/oracle-env and the batch root; none is an additional cleaned requirement. The shared cleaned runtime contains only pinned NumPy/SciPy/Matplotlib subset for this task. It runs CPU without PyVista/cv2/upstream normal-integration imports.

prepare_bini_task.py staged exact original fixtures, independent tests and final metrics.json **before** local extraction. red_before_local.log records the expected missing-src import failure. Initial test execution exposed a test decoder assumption255 rather than original16-bit65535; only that incorrect test expectation was corrected after checking captured encoded dtype, without altering bound inputs, algorithm or numerical tolerance. The local normalized inputs and four sparse matrices now match the original exactly.

## Mathematical traceability and independent validation

| Paper stage | Original file/function | Local function | Validation case |
|---|---|---|---|
| Orthographic tangent equations and channel/sign convention Eq3 | simple.bilateral_normal_integration nx/ny/nz | physics_model.system | hand-derived plane, directional signs and doubled pitch |
| One-sided differences and missing-neighbor rows Eqs16–17 | simple.build_derivative_matrices | physics_model.derivative_matrices | all original CSR entries; masked-hole adjoint and constant-nullspace tests |
| Sigmoid normal-scaled contrast Eqs18/22 | simple.bilateral_normal_integration grad/weights | physics_model.weights | both original directional weight arrays; complementary weights and flat case |
| Weighted objective Eqs19–21 | simple.bilateral_normal_integration energy | solvers.integrate | full original energy history; independent fixed-weight directional finite difference |
| Jacobi-preconditioned CG frozen-weight solve Eqs23–24 | simple.bilateral_normal_integration CG | solvers.integrate | final original depths and complete traces; independent analytic plane and true CG status |
| Gauge and units | source depth, then private fixed gauge | solvers.integrate | all original raw gauges; zero-mean plane and linear unit scaling |
| Evaluation | independently specified known vectors | judge.metrics, visualization.quality | same independently known NCC/NRMSE vectors, zero estimate and constant-reference rejection |
| Complete pipeline | source full-field call | solvers.reconstruct / main.run | both full native arrays, no easy-component-only scoring |

All source conventions are explicit locally: A stacks up/down/right/left; b repeats -green,-green,-red,-red; nz=-blue. Missing rows remain zero with their original RHS retained. This adds unavoidable boundary energy, not a fictitious height anchor. The sigmoid uses normal-scaled depth differences, consistent with paper Eq17, not unweighted gradients or residuals. Source w-order and swapped displayed horizontal/vertical weight naming are preserved. No generic optimization replaces the published custom alternating updates.

One independently discovered degenerate case is documented, not silently copied: a perfectly flat exact normal field has zero initial energy, so source relative-energy stopping becomes0/0. Local code explicitly terminates that already stationary zero residual after one solve; a dedicated independent test verifies this correction. Neither bound input has zero energy, so the original trajectories/results are unchanged. Inner CG failure is now reported rather than ignored; every bound and plotted solve reports status0. Logistic overflow is treated only as its exact finite saturation; nonfinite inputs/results are not hidden.

## References, metric definitions and fresh measurements

No actual depth truth is available. evaluation/baseline_reference.npz stores the independently executed standard method, never relabeled as ground truth. Its two keys are exactly the public output contract. The original stage fixture is consumed by comparing_test; notebooks/reference_outputs/upstream_traces.npz contains only the two original energy arrays consumed by the notebook, avoiding duplicate depth/weight arrays. upstream_measurements.json identifies original data/code/parameters/environment, fresh_measurements.json identifies the active local reconstruction. Original mesh/PNG/reference copies not consumed in the task remain private.

Both complete fields separately use cosine NCC without mean subtraction, and NRMSE=RMS(estimate-reference)/(max(reference)-min(reference)). The declared zero-mean gauge is part of solver output; Judge does not fit an offset, sign, scale, crop or alignment. Nonzero numeric boundaries NCC>=.999999/NRMSE<=1e-5 were fixed before extraction. Independent repeats are identical; the tolerance permits numerical backend variation rather than mechanically applying1.1×zero self-error. Undefined/constant references are evaluator errors. Both required maps must meet both metrics. Fresh actual metrics:

```json
{
  "reconstruction_ncc": 0.9999999999999998,
  "reconstruction_nrmse": 0.0,
  "tent_reconstruction_ncc": 1.0000000000000002,
  "tent_reconstruction_nrmse": 0.0
}
```

The standalone Judge validates exact shape, real float64, keys, finite values, reference precedence and valid boundaries before scoring. Malformed inputs/evaluator failures have empty metrics/statuserror and exit2; valid poor predictions have statusfail/exit1; pass exits0. It never imports solver code and resolves hidden assets from its own file. CLI report parents, foreign cwd and report-write failures are exercised independently.

## Complete active notebook and actual scientific comparisons

Final nbconvert exit0, runtime25.944150700s, timeout900s; exact command/interpreter are in `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020\bini_nbconvert.json`. The canonical notebook actively reconstructs both complete fields through cleaned source, evaluates fresh output, recomputes every noncanonical k in0/1/2/4/8 and every CG tolerance1e-2/1e-3/1e-5, and displays4 unique actual PNGs. There is no cache-preview substitution. Full-precision tables/maps/energies and PNG/SVG/PDF are private in final_figures. Original and fresh profiles/traces have separate labels. Equal-weight integration is this exact discretization's k0 ablation; external paper methods are explicitly omitted rather than fabricated or silently dropped.

Parent actually opened every scientific figure. The first map layout had long titles colliding with neighboring colorbars; titles were shortened and split over lines, fonts sized, and the complete notebook rerun. Both horizontal/vertical weight maps are retained. Surfaces are plotted with stride2 only to bound rendering, while reconstruction and all metrics use every native pixel. Signed parity maps use a zero-centered diverging scale; source-energy and all directional residual histograms show boundary/discontinuity mismatch, with logarithmic counts revealing small populations. Surface/profile maps do not imply metric or true depth. Sensitivity curves expose nonconvex reweighting/inner-tolerance dependence: a tighter CG solve can enter a different path and differ from the selected operational result, so default-reference error is never presented as true-surface error.

## Plans, interfaces, cleanup and mandatory gates

Manual semantic review checked paper equations, factors/signs, masked indexing, row/column/normal axes, weight ordering, objective normalization, CG preconditioner/stop, gauge/units and actual callable signatures against both plans and README. Main uses only public data and defaults. Notebook helper comparisons are separate actual calls; importing src.main does not select a backend. Source/generator/Judge/tests/notebook use no reconstruction/domain library or cross-task import. The private oracle exception is not used at runtime.

| Gate | State / evidence |
|---|---|
| Paper/version/exact source/data/hash/equation mapping | passed; primary proceedings, remote source and tables above |
| Original exact-bound run/repeat and fresh local run | passed; original captured exit/logs, repeat equality, final CLI and active notebook |
| Independent original parity and applicable physics | passed; exact input/operators, original full outputs/weights/traces; analytical signs/plane/adjoint/gradient/nullspace/pitch |
| Units/constants/precision/conventions | passed;16-bit decoding, untouched arrays and manual/readme/plan check |
| CLI output and fresh Judge | passed; both full arrays, finite metrics and generated report |
| Strict Judge contracts | passed; valid bad result, missing files/keys, shapes/dtypes/complex/NaN/Inf, truth precedence/evaluator/config/report/cwd |
| Public input sufficiency/no privileged leakage | passed; README/raw NPZ/pinned subset, no analytic depth/formula/seed or hidden-reference access |
| README/acquisition/output/template inventory | passed;5required sections, every public key batch-first and all outputs specified; no standalone metadata |
| Minimal CPU environment/local mathematics | passed; no PyVista/cv2/domain/upstream imports, allowed sparse/CG primitives only |
| Full suite | passed;10 distinct tests, with looped both-map adversarial contracts |
| Complete active notebook/comparisons/figures | passed; final nbconvert,4unique actual images and full actual sweep evidence |
| Honest result provenance/metrics/limits | passed; author-analytic data and method-reference status explicit, historical claims not substituted |
| Plans/source/README semantic agreement | passed; mathematics/signatures/defaults/flow reviewed above |
| Cleanup/archive/license/consumer inventory | passed; no original checkout/PDF/capture/log/screenshot/temp execution in task; only two canonical output files |

Cross-task acceptance and final source/file hashes are maintained in the parent archive after this note; no task result is claimed as independent physical depth accuracy.


## Independent parent acceptance

Passed: parent independently ran 10 distinct tests in the cleaned shared environment and the standalone Judge from a foreign working directory. Source/public-interface/dependency/call-role/executed-notebook/inventory audit has no errors. Reviewed paper orthographic tangent equations and normal-scaled bilateral contrast, every original sparse operator, all full depths/weights/energy updates, CG/gauge/units and independent plane/adjoint/gradient/nullspace/pitch checks. Original repeats are exactly equal. Four actual figures were opened; long labels were corrected and the complete notebook rerun. Analytic provided data and lack of metric depth truth are explicit, and the flat-field zero-energy stopping correction is separately documented. No figure or array was accepted on an unexecuted preview. Parent raw test/audit evidence and final file hashes are indexed under `D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch3_20261020`. Final source hashes are refreshed after this acceptance-only note.
