# Provenance and validation evidence

Status: ready for independent parent audit; not self-accepted. All mandatory gates below passed. This task reuses real experimental observations and has no independently measured volumetric truth.

## Source, papers, and licenses

Dataset: Levin et al., Nanomaterial datasets to advance tomography in scanning transmission electron microscopy, Scientific Data 3,160041 (2016), DOI https://doi.org/10.1038/sdata.2016.41, version of record. Primary paper https://www.physics.ucla.edu/research/imaging/Publications/pdf/tomo_sdata.pdf provides the cobalt phosphide HAADF acquisition and identifies two blank/backlash images. Data article https://doi.org/10.6084/m9.figshare.2075176, archived Figshare metadata figshare_2075176.json explicitly licenses the aligned data CC0. File3706987 tiltser_Co2P.tif is 203484297 bytes; MD5 2de7e7232f3ec3400b860e050b132e96 exactly matches publisher metadata. Private archive file tomviz/tiltser_Co2P_parallel.tif was assembled from HTTP byte ranges and hash-verified. This is the complete aligned acquisition, not newly simulated measurements.

Modern implementation: https://github.com/OpenChemistry/tomviz, remotely verified commit40a96c18c90633cb29279b2864bd893f9f866692, archived remote response tomviz_commit.json. Source tomviz/python/Recon_SIRT.py; BSD3 license preserved at task LICENSE with source notices. Modern software paper is Schwartz et al., Real-time 3D analysis during electron tomography using tomviz, Nature Communications13,4458 (2022), DOI https://doi.org/10.1038/s41467-022-32046-0. This later source revision is not claimed to be the code used in the2016 dataset paper. The selected Landweber option is one of the source's SIRT variants; it has no row/column normalization.

The full source/archive, download files, remote responses, capture scripts and original logs are private in D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/material_physics. tomo_archive_index.json records bytes and SHA256 for every bound source/data/capture asset. No upstream source package or paper PDF is bundled in the cleaned task.

## Binding the observations before oracle execution

Original stack shape76 by1157 by1157. Remove zero-based indices49 and50, retain rows/columns2:1154, and take float64 arithmetic means of nonoverlapping9 by9 detector blocks. Bound observations are74 by128 by128, stored batch-first float64. This is the same pixel-integration operation for the original oracle and local code; neither reconstruction receives the unbinned source. Angles are -75 to75 step2 degrees, omitting23 and25. Official Tomviz example https://openchemistry.org/files/temdata/TiltSeries_NanoParticle_doi_10.1021-nl103400a.emd has the identical74 retained-angle list; its spatial spacing was not substituted for the paper's nominal0.71nm source sampling. Output pitch6.39nm is explicitly nominal. The original data are already aligned.

Raw public arrays contain only measurements and angles. Detector horizontal coordinates are independent reconstruction slices. Each cross-section is row-major with depth row increasing opposite physical depth and column increasing detector y. Output axes are(batch,x,z,y). Both acquisition and source use pixel/ray width1 in nominal voxel units. No absolute mass-density calibration exists.

## Original run and independent fixtures

Isolated original environment: oracle-env/Scripts/python.exe, Python3.11.11 Windows CPU, NumPy2.4.6, SciPy1.17.1; single-thread OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1. Exact private command: oracle-env/Scripts/python.exe capture_tomo.py. Original source's CompletableOperator/progress/cancel GUI objects receive a minimal private progress adapter; source parallelRay, SIRT initialize/recon2, iteration control and numerical functions execute unchanged. The adapter does not replace any numerical operation. Raw file hash is checked before capture.

Original source operator.transform configuration: Niter100, stepSize0.0001, updateMethodIndex0 Landweber, Nupdates0; full128-slice volume. Matrix has1474124 nonzero entries. First run131.4683178s, exit0, tomo_oracle.log/json/npz. Independent identical rerun179.0323655s, exit0, capture_tomo_rerun.py, tomo_oracle_rerun.log/json/npz. All final entries are bitwise identical. Source volume dtypefloat32, range0 to4120.3955, shape128 cubed. Captures preceded local method implementation.

Frozen fixture consumers:

| Artifact | Independent producer | Consumer |
|---|---|---|
| test/fixtures/upstream_operator.npz | Unmodified original parallelRay rows/columns/values | test_original_ray_matrix exact sparse representation parity |
| test/fixtures/upstream_full.npz | Original full100-iteration reconstruction | test_original_full_pipeline full-field parity |
| notebooks/reference_outputs/upstream_reconstruction.npz | Original volume, required key reconstruction | Notebook labeled original-method comparison |
| evaluation/baseline_reference.npz | Original volume, reconstruction(1,128,128,128)float32 | Independent Judge; standard-method reference, never actual truth |

Fixture prefixes input/param/config/output identify roles. Full parity was fixed at rtol2e-6, atol0.0005 before local acceptance. Observed maximum local/original difference is zero, not merely below tolerance. Exact operator rows/columns/values agree. Original rerun equality motivates narrow nonzero reference tolerances; numerical agreement does not establish true density accuracy.

## Equation/function/test traceability

| Physical or algorithm stage | Upstream | Local | Validation |
|---|---|---|---|
| Detector block integration and retained geometry | Existing aligned Figshare frames and capture binding | load_data validates frozen integration output | Raw hashes, exact retained angle test, README constants |
| Ray/pixel line integrals | Recon_SIRT.parallelRay/rmepsilon | physics_model.parallel_ray/rmepsilon | Exact frozen COO parity; analytic8-voxel constant path length |
| Forward/adjoint ordering | SIRT recon2 A and A.T | physics_model.forward/adjoint | Independent segment-intersection matrix and adjoint inner product |
| Objective gradient A.T(Aq-b) | SIRT recon2 Landweber option | solvers.reconstruct | Central finite-difference directional gradient |
| Zero initialization, step0.0001,100 updates, positivity | ReconSirtOperator.transform and SIRT recon2 | solvers.reconstruct | Full original-pipeline parity; analytic residual descent and positivity |
| Dynamic-range NRMSE/cosine NCC | Evaluation conventions fixed for this benchmark | Independent Judge and visualization.metrics | Known scaling/self-reference values and full notebook comparison |
| Full observation consistency | Independent ray/pixel segment clipping | Judge ray_matrix and evaluate | Independent16-grid projector parity and all128-slice residual |

## Discrepancies, precision, and physical checks

The source adds0.001degrees to every angle if any exact zero angle is present; none of the bound odd-degree angles is zero. It shifts globally negative observations by their minimum; bound data do not require this shift. The source stores the volume float32, initializes a float32 work vector, then promotes this work vector after its first float64 gradient update and keeps that dtype across slices. Local code preserves that mixed-precision path. Changing to a homogeneous precision is not claimed as unchanged reproduction.

Source globally suppresses divide-by-zero/invalid warnings in geometry. Local geometry scopes that same suppression to the function using np.errstate; matrix values remain exact. A local adjoint wrapper axis error was caught before validation and corrected to iterate slices rather than angles; it does not affect original source fixtures. Independent ray-intersection agreement, adjoint identity and gradient finite differences now pass. The original selected model is a linear HAADF projection approximation, not a full multiple-scattering model.

Original full-volume projection residual0.14384113582003832. Judge independently constructs segment intersections and obtains0.1438411360390179 on the fresh result; the small difference is float32 matrix summation ordering, not a rescale. Fresh local forward residual0.14384113582004115. Inspection of angle-resolved residuals shows larger mismatch near extreme tilts. No noise variance is available, so no chi-square calibration or noiseless consistency claim is made. Missing-wedge and finite-domain mismatch remain explicit limitations. Analytic tests demonstrate path lengths, adjoint/gradient consistency and nonnegative convergence independent of the benchmark baseline.

Thresholds were fixed before Judge acceptance: NCC>=0.999999, standard dynamic-range NRMSE<=0.00001, projection residual<=0.1452795471782387 (1.01 times original). The two reference-error boundaries allow ordinary float32 backend rounding despite exact same-machine reruns. No target gain, crop, registration, positivity substitution or reference fallback is used by Judge. All128 cubed output entries are scored. The baseline volume is never labeled truth.

## Execution evidence

All cleaned commands use the shared research-runtime/Scripts/python.exe and run from the task directory unless stated otherwise. Shared runtime environment and numerical package versions are archived in tomo_environment.json. No package was installed or upgraded in the shared runtime; upstream-only GUI/Tomviz code is never imported by local source/tests/notebook/Judge. Cleaned imports are standard library, NumPy, SciPy general sparse primitives, and Matplotlib plotting.

| Command | Exit | Evidence |
|---|---|---|
| python -m src.main |0| tomo_cli.log,164.098s, full volume saved |
| python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json |0| tomo_judge.log and generated report; NCC0.9999999999999981, NRMSE0, projection residual0.1438411360390179 |
| python -m pytest test/ -v -p no:cacheprovider --basetemp PRIVATE/pytest-tomo-release |0| tomo_tests_release.log,19passed182.87s |
| python -m nbconvert --to notebook --execute notebooks/stem_cobalt_phosphide_tomography.ipynb --output tomo_executed_final.ipynb --output-dir PRIVATE --ExecutePreprocessor.timeout=1200 |0| tomo_notebook_final.log; explicit imaging101-research shared-runtime kernel, active full reconstruction168.3524247s |

Tests include exact sparse/full parity; analytic geometry; independent evaluator projector; adjoint/gradient; nonnegative descent; known metrics; input constants/dtype/allowlist; Judge missing file/key, shape/dtype/nonfinite errors; zero and1e308 finite bad predictions; foreign current directory; missing hidden asset; report-write failure. Judge emits finite strict JSON and correct0/1/2 codes. Early wrong-working-directory test invocations and an overstrict exact floating-point cosine assertion were corrected; no failed mandatory check is represented as passed. The corrected full suite is the authoritative result.

Notebook actively computes the entire selected benchmark through local source modules, invokes standalone Judge on its fresh output, and reconstructs three additional central-slice iteration variants10/30/100. It loads original arrays only for explicitly labeled comparison. Timeout1200s provides margin over measured~3min reconstruction plus plots. Canonical notebook contains successful executed outputs; temporary notebooks/logs remain private. Final Markdown-only encoding cleanup did not change executed code or numeric outputs.

## Figure and semantic review

Four actual figure groups, each PNG/SVG/PDF, plus exact numeric_sources.npz and SHA256 manifest, are private in figures/stem_cobalt_phosphide_tomography. Groups: measured projections/sinogram/angular coverage; three-axis original/fresh/error sections; forward projections/signed residuals; convergence/angle residual/central line/fresh iteration sensitivity. All displayed arrays were computed or explicitly identified as the original source result. No zero-filled absent method is plotted. Exactly zero source difference is correctly displayed on a nonnegative numerical-scale colorbar. Figures were visually inspected; text encoding and long titles were corrected before final export. Sinogram uses actual nonuniform retained-angle coordinates.

README formula, signs, angle units, depth orientation, nominal scale, complete public array keys and output contract were semantically reviewed against executable geometry. Plans describe actual call signatures, mixed precision, zero initialization, step/stopping, objective normalization and optional comparisons. Mathematical strings were scanned for lost LaTeX backslashes/control characters; explanatory formulas use raw strings and ASCII notation where appropriate. No public GitHub URL, truth, hidden reconstruction, target-derived calibration or upstream path appears in README/raw/requirements. A package-level import/backend audit confirms no domain solver or cross-task import. Importing main does not change Matplotlib backend.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper/revision/data hashes/licenses/traceability |passed| Above; archive index and preserved BSD license |
| Exact-bound original and cleaned execution/rerun |passed| Original logs, CLI, zero full-volume difference |
| Independent fixtures and physical checks |passed|19-test suite and explicit independent projector |
| Units/constants/coordinates/precision |passed| Geometry, constants, full parity, docs review |
| Standard CLI and standalone Judge |passed| Direct commands and generated report |
| Malformed/poor/overflow/evaluator/report failures |passed| Contract tests |
| Public allowlist sufficient, no leaks |passed| README/raw/requirements review |
| Local math/minimal dependencies |passed| NumPy/SciPy explicit intersections and inverse updates |
| Active full notebook and comparisons |passed| Executed canonical notebook, final log |
| Metric/reference/figure provenance |passed| Above and private figure manifest |
| README/plans/source semantic agreement |passed| Recorded semantic review and tests |
| Template/inventory/license cleanup |passed| No generator, PDF, upstream archive, temporary notebook or duplicate report inside task |

## Bound package SHA256

| Asset | SHA256 |
|---|---|
| data/raw_data.npz | 0bc6e518fb411c9334eee71c0f2f50b91bb361b834c2399f423fd2ea5905cc4f |
| evaluation/baseline_reference.npz | 3b10015af58580890bc021faf75287bfb8b5c1500a8d25eb532bff6fd4dcd37b |
| test/fixtures/upstream_operator.npz | 652228899674445fdfdc0a1d7d246cf1739e73854c735aa34c8785b145f2b1d0 |
| test/fixtures/upstream_full.npz | ac9f3cb7e088004b010b09c734fee1b6637f501eaf83f187a67adde09c1e6fa1 |


## Parent acceptance review

Parent independently ran19tests (19 passed in176.40s), foreign-cwd Judge0 and source/inventory audit errors[]. Read explicit ray/grid segment construction, mixed-precision positivity-constrained Landweber updates and source/plan conventions; inspected all4 actual scientific figure groups. Real CC0 aligned Co2P STEM tilt data, fixed9x9 detector averaging and nominal6.39nm pitch; missing angles23/25degrees and missing wedge retained. Independent segment-clipping projector and matched adjoint/gradient checks supplement exact original Tomviz parity. Standard method reference is explicitly not atomic composition or physical truth.
