# Provenance and validated mechanical inverse

## Fixed binding and evidence archive

Archive root `B/foam`, where B is the maintainer-only
`research_batch7_20261022/groups/astro_earth_mech` directory outside the release.
The archive index and final evidence JSON identify individual files by SHA256.
No external domain package is installed in or required by the cleaned runtime.

The physical unknown is the entire three-dimensional atlas of 357 relative
interface tensions and 65 region pressures, including the exterior gauge, on
the existing real 64-cell Phallusia microscopy-derived mesh. This is geometry
to force inference, distinct from traction-image to monolayer continuum stress.
The observations are already segmented/meshed; they are not raw fluorescence
images. No force truth, segmentation truth, absolute tension calibration or
physical voxel spacing is supplied. The reference is an independently executed
author-method result, never biological ground truth.

| Artifact | Exact identity | SHA256 / archive |
|---|---|---|
| Paper | Ichbiah et al., *Embryo mechanics cartography: inference of 3D force atlases from fluorescence microscopy*, Nature Methods 2023, DOI 10.1038/s41592-023-02084-7 | Complete article XML `primary_1.xml`, full extracted text `paper_fulltext.txt`, reading notes `paper_reading.txt`; final index records hashes |
| Primary full text | https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10703677/fullTextXML | Authoritative full article, methods, figures/captions and references were read; CC BY 4.0 article |
| Mathematical supplement | `supplementary_files/41592_2023_2084_MOESM1_ESM.pdf`, all 14 pages and extracted `supplement.txt` | From Europe PMC supplementaryFiles archive, ZIP SHA256 `7f6131d89b19f519a43d327a64d2b52a4e98d2296fb4d8fb59145b30f720b6f2` |
| Author code | https://github.com/VirtualEmbryo/foambryo | Remotely verified commit `54dc23e8020a032bf191eb7f8ba97b8b8b269086`, GPL-3.0-or-later |
| Source ZIP | `source.zip`, extracted `src/` | `014b9041cdcd923344fc24cb1f5bc2d7692c7c1c8cb6ee2dfa5abdb5b2654b1b` |
| Real author data | https://www.dropbox.com/scl/fo/fanim64m3dwy6in9dkq6v/AGnqONanjcZUMEtvFNFy9wo?rlkey=mx7ffzbea3v93rsqrkdr9ok1q&dl=1 | `author_data.zip`, SHA256 `8bf5e9d1e5058ec75f2b743c940263b8673bdd59c6ca5359e002a47069fb8bd0` |
| Bound existing mesh | `data/Meshes/Phallusia_Guignard_et_al.rec` | `becec7ae514b43392c8c522eae44037c2518a9ec82165b7ddcccc50af2f91a4c` |
| Public observations | `data/raw_data.npz` | `c54a51f98112b4dca8b60c8fec07a12ea101759b3c9f1b4e5d62c25b068e2369` |
| Pre-extraction upstream stages | `source_stage_capture.npz` | `cad50b8fe6a4451a680ee7dfe6fd927b8d2b7503e89bca02a85df7426ad87538` |

The DOI identifies the published version, while the selected source is the
explicit later author revision above. No claim of byte-identical 2023 software
is made. The main-paper PDF download returned HTML, and another endpoint denied
access: `paper.pdf` is a retained failed HTML download, **not a paper PDF**.
Complete primary article XML and the actual mathematical supplement establish
the paper-reading evidence. Failed downloads and an initial Windows long-path
extraction are preserved privately, not packaged or called successful.

Source copyright and full GPL license are retained in LICENSE and NOTICE.
The article's CC BY license and software's GPL license are not assigned to the
external mesh. The authors offer these example data for use; a separate mesh
redistribution license was not stated. This unresolved redistribution scope is
disclosed as a limitation for this local research package, rather than inventing
permission or describing the data as GPL/CC. Data were not generated here.

## Original execution before extraction

The private oracle venv `foam/oracle-env` has Python 3.11.11 and read-only access
to the shared NumPy 2.4.6 / SciPy 1.17.1 numerical libraries. Only that private
environment installs foambryo's upstream dependencies (dw3d, trimesh, edt,
meshio and image tooling). The exact installation report is `oracle_install.json`.
The source checkout is imported only by private capture scripts. Original
domain files remain byte-identical to their source ZIP; the final byte audit
records any mismatch and does not treat Python-generated caches as source.
All numerical runs use CPU, with OMP, OPENBLAS, MKL and NUMEXPR thread counts one;
there is no GPU computation or shared environment upgrade. Full backend/platform
and package information is saved in `environment_provenance.json`.

Every command below is an actual native subprocess argument list with
`shell=False`; stdout and stderr are separate private files. The process JSONs
record exact executable, command, return code and elapsed time. `OP` denotes
the private oracle Python; `PY` denotes the shared research Python.

| Command, from archive/task as appropriate | Native result | Evidence |
|---|---|---|
| `OP -X utf8 original_bound.py` | 0, 11.393723400004092 s | `original_process.json`, `original_stdout.log`, `original_stderr.log`, `original_bound.npz`, `original_bound_metrics.json` |
| `OP -X utf8 capture_stages.py` | 0, 16.181335000001127 s | `source_stage_process.json`, logs, `source_stage_capture.npz/json` |
| `OP -X utf8 capture_comparisons.py` | 0, 54.26667019999877 s | `source_comparisons_process.json`, logs, `source_comparisons.npz/json` |
| Original historical Gram-route capture | Successful independent direct source run | `capture_gram.py`, `source_gram_comparison.npz/json`; historical Gram is distinguished from later source Operator |
| `PY -m src.main` | 0, 9.672266299996409 s | `cli_process.json`, `cli_stdout.log`, `cli_stderr.log` |
| `PY evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0, 0.3161229000106687 s | `judge_process.json`, logs and generated report |

The first native original run computes scalar Young–Dupré tensions and
variational Laplace pressures on the exact complete mesh. The second independent
full run in stage capture is bitwise equal for both fields. Captures include
the actual 1708 by 357 tension system, all 1707 wedge keys/angles/line lengths,
all interface areas, selected analytic derivative columns and full pressure Gram
and cross blocks. No fixture is made by the cleaned function being tested.

The original warns that 128 malformed edges touching three regions cannot
supply valid triple angles. The warnings remain in original stderr. An independent
topological and geometric audit (`independent_mesh_audit.py`, JSON and NPZ)
finds those same 128 edges, 6874 valid triple edges, positive cell volumes,
no zero-area triangle, and edge-incidence counts 203404/6874/1331/22/12 for
valences 2/3/4/5/6. Minimum triangle area is 0.5 mesh-unit squared. All triangles
remain in area/volume gradients, including faces incident to excluded junctions.
The package preserves 70433 raw vertices, 144312 triangles and all region labels;
only unused vertices are removed during numerical assembly, leaving 67303 used
vertices and 201909 derivative columns. The 65 labels are nonconsecutive,
between 0 and 90; output indexing uses sorted observed labels.

## Frozen success criteria and independent physical checks

`freeze_criteria.py` ran before cleaned inverse implementation; its timestamp,
fixture hash, measurements, controls and decisions are in `frozen_criteria.json`.
Bitwise repeatability alone did not set zero tolerances. The captured original
system was independently solved with LAPACK gelsd, gelss and gelsy in float64
and float32, and source fields were separately serialized through float32.
Worst measured range-normalized RMS errors were
4.769238430128643e-7 for tension and 4.419089165790887e-7 for pressure. Five times
those measured spreads gives the frozen nonzero limits below. These limits
describe numerical reproduction of a chosen estimator, not biological accuracy.

| Quantity | NCC minimum | Range-normalized RMS maximum |
|---|---|---|
| Every one of 357 tensions | 0.9999999999794431 | 2.384619215064322e-6 |
| Every one of 65 pressures | 0.9999999999979622 | 2.2095445828954433e-6 |

NCC is uncentered cosine similarity. NRMSE is RMS error divided by the full
reference dynamic range, separately for the two quantities, with no crop,
rescaling, permutation or alignment. Both quantities must meet both limits.
Additional independent pressure moment residual is at most 1e-5, mean tension
error at most 1e-6 and exterior pressure magnitude at most 1e-10.

Pre-extraction controls were rejected: zero tension/pressure NRMSE
0.1600298781962364/0.37259471051659276; uniform
0.07892868328333776/0.19209266395864852; reversed
0.11525675776955407/0.2795045432531241. Tests additionally perturb each quantity
separately. These errors exceed frozen limits by many orders of magnitude.

Frozen stage tolerances: geometry rtol 2e-12, atol 2e-10; angles rtol/atol
2e-12; solver rtol 2e-10, atol 2e-11. Large-dimensional Gram blocks have an
explicit 2e-8 absolute accumulation tolerance. Formulation comparisons use
rtol 2e-9, atol 2e-10. No threshold was relaxed after a cleaned result failed.

Independent physical tests use a regular tetrahedron of edge length one:
area sqrt(3), enclosed volume sqrt(2)/12, uniform unit tension gives pressure
4 sqrt(6). Area and volume gradients pass centered directional differences;
translation invariance, label/winding equivalence, summed volume gradients,
length scaling and a 120-degree equal-tension junction are checked separately
from upstream parity. Actual full-mesh pressure Gram rank is 64 and condition
number 91.78183984946723. The standalone Judge independently re-derives triangle
gradients, not imports the solver. It checks pressure normal equations rather
than demanding zero total vertex force on noisy measured geometry.

## Equation, source and executable test traceability

The complete formula definitions, numerical gauges and API signatures are in
the two shipped plans, and the public conditional objective is in README.
The supplement's force-inference derivations and the article's geometry-method
sections were read as well as the primary figures and comparison captions.

| Required stage / paper derivation | Selected upstream file/function | Local executable implementation | Independent validation consumer |
|---|---|---|---|
| Surface-energy area and signed-volume derivatives | `geometry.py: compute_area_derivative_sparse`, `compute_volume_derivative_sparse` | `physics_model.area_volume_operators` | `test_source_stages.py`, tetrahedron directional derivatives/scaling tests |
| Contact angles and triple-line averaging | `wedge_refusal.py: line_averaged_angles`, DCEL half-edge angle reader | `physics_model.junction_geometry` | All 1707 captured keys/angles/lengths, independent topology audit |
| Scalar Young–Dupré / supplement scalar balance | `tension_inference.py: _build_matrix_tension_symmetrical_yd`, `_infer_tension_symmetrical_yd` | `tension_system`, `solve_tensions` | Full captured system and pipeline, ideal 120-degree junction |
| Tension scale ambiguity / exact selected-source gauge | Source least-squares/gauge utilities used by tension inference | `solvers.solve_tensions` | Frozen independent drivers/precision; mean-one and mean-two experiments |
| Variational Laplace / surface energy pressure normal equations | `pressure_inference.py: _infer_pressures_variational`, `_extract_submatrices` | `solvers.solve_pressures` | Source G/H blocks, full pressure parity, independent tetrahedron and Judge moments |
| Projection Young–Dupré | `_build_matrix_tension_projection_yd` | `tension_system(method='projection')` | `test_formulation_parity.py`, active full comparison |
| Lami / inverse Lami / Log Lami | `_build_matrix_tension_lami`, `_build_matrix_tension_inv_lami`, corresponding logarithmic source builder | Three local `tension_system` branches | Independent original full fields and native gauge checks |
| Historical variational Young–Dupré Gram elimination | `_infer_tension_variational_yd` with `VariationalRoute.Gram` | `solve_variational_tensions(route='gram')` | Separate source Gram capture and fresh full comparison |
| Later source direct operator route | Same source with Operator route | Explicit local Householder elimination and dense general least squares | Separate original Operator fields, not conflated with Gram |
| Laplace curvature pressure and area-weighted curvature alternative | `geometry.py` curvature reader, `pressure_inference.py: _infer_pressures_laplace`, `_build_matrix_laplace` | `interface_curvatures`, `solve_laplace_pressures` | Both original complete pressure arrays and sign/order diagnostic |
| Paper length-weighting comparison | Explicit line-length weights in source builders | `weighted=False` ablation | Active complete same-data weighted/unweighted comparison |

## Documented numerical and scientific qualifications

The selected source is a later revision with an exact final mean-tension gauge;
the paper's appended soft normalization is not called byte-identical. The
current source also offers a direct Operator variational tension estimator,
whereas the historical Gram method normalizes Gram rows before elimination.
Both are retained, computed and named separately: on noisy measured geometry
they optimize different weighted residuals and need not agree.

Chord wedge angles use arccos in [0,pi], retaining the source's reflected-reflex
convention. Mean wedge sums can depart from 360 degrees; figure 5 displays the
departure. Source half-edge accounting repeats each line length six times; this
factor cancels in relative length weights. Neither feature is silently corrected.
Log Lami's native gauge is geometric mean one; native arrays are preserved, and
only the explicitly labeled cross-method plot rescales its entire atlas to
arithmetic mean one. None of these plotting normalizations changes evaluation.

An actual full-test failure occurred during curvature extraction. Equivalent
division/einsum expressions changed a curvature sign at vertex 55954 whose
curvature vector is almost tangent to its normal; pressure disagreement was
about 8.016e-7 (weighted variant 2.794e-7). Original diagnostic captures are
`source_curvature_diagnostic.npz/json`; `diagnose_local_curvature.py` isolates
the formula, and failed full test logs remain in `prior_full_tests/`. Final code
preserves source Heron expression, reciprocal multiplication and three-term
sum order. Captured Laplacian vectors, normals and inverse incident areas then
match bitwise, and all final comparison pressures agree near machine precision.
This is faithful arithmetic extraction, not an upstream formula correction or
threshold relaxation. The diagnostic capture was made after extraction to
investigate the failure and is not mislabeled as the earlier frozen fixture.

The curvature comparison explicitly leaves NaN for cells not connected to the
exterior gauge through measurable interfaces. A small independent disconnected
graph test verifies this guard. Every cell is connected on the bound dataset;
the guard changes no bound output or notebook result. No NaN is allowed in a
submission or selected reconstruction. Earlier failures from the first Judge's
incorrect consecutive-label assumption remain privately in
`prior_cli_and_judge/`; the final evaluator uses actual sorted labels.

## Notebook, fresh results and actual comparisons

The canonical executed notebook is
`notebooks/ascidian_cell_interface_force_atlas.ipynb`. It reconstructs raw
observations through local `src`, writes a fresh submission, calls the independent
Judge, actively computes all paper families and both variational routes, both
curvature pressure variants, length ablation, full geometry perturbations,
two full mesh-scale runs, gauge change and six precision/backend cases.
No archived reconstruction is used as a fresh result. Historical simulated
force-truth performance in the paper is explicitly context, not an accuracy
measurement of this real specimen.

Actual final command is `PY -X utf8 -m nbconvert --to notebook --execute` with
the private `imaging_foam` kernel, timeout 2400, output outside the task. Native
return code is **0**, runtime **156.08088270000007 s**, independently captured
stdout/stderr and command in `notebook_process.json` and `notebook_execution/`.
An earlier successful execution is archived under `prior_notebooks/`; final
execution includes corrected 3D spacing, categorical cutaway and exploded-view
caption. The canonical executed notebook SHA256 is
`7483664b215065bfce49125415794e3738665d9de7da7ae5c5eb4c028132b00c`.
Actual kernel `sys.executable` is the shared research Python, Python 3.11.11,
NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1, inline backend; all numerical
threads one. These are saved from the running kernel in
`notebooks/reference_outputs/fresh_run.json`, not inferred from a parent shell.

The fresh selected result has tension/pressure NRMSE
8.015325328981584e-16 / 7.351816260942865e-16; NCC
1.0000000000000002 for each (floating point cosine roundoff around one).
Independent pressure-moment residual is 1.4073086993879775e-13, both gauges exact.
All separately captured original comparisons agree: largest tension absolute
difference across original/cleaned methods is 4.3720582709738665e-13 (Gram);
largest pressure difference is 4.9057979900624105e-15 (Gram). Curvature comparisons
have maximum pressure differences 1.6653345369377348e-16 and
1.734723475976807e-16. Full-precision experiment metrics are retained.

The six actual scientific figure groups are exported privately as matching
PNG/SVG/PDF plus numeric NPZ. They were each opened and inspected after final
execution; dense mesh polygons are intentionally rasterized inside vector
containers while axes and labels remain vector. `figures/plot_provenance.json`
associates canonical notebook, observation hash, data arrays, consumer and
each exported file hash.

1. Complete observed exterior/cutaway categorical geometry, contact-angle
   distribution and all-interface area distribution.
2. Tension cutaway and explicitly 75%-exploded cell surfaces, every cell pressure
   and the observed-geometry force graph. No exploded coordinate is used in fitting.
3. Every original/fresh tension and pressure, and symmetric signed spatial
   differences with zero at the neutral color.
4. All eight tension formulations/weighting variants and conditional pressure
   distributions, paired actual objective residuals, both curvature comparisons,
   spatial Gram discrepancy and length-weighting ablation.
5. Every malformed edge, chord-angle closure defect, three fresh full geometry
   perturbations with sigma .1/.25/.5 and seeds 210/211/212, complete pressure
   responses and measured positivity margins. Jitter is a stress test, not a
   calibrated uncertainty model or confidence interval.
6. Actual length scales .5/2, inverse-length pressure scaling, mean-two gauge,
   all three LAPACK drivers in both precisions with frozen limits, complete
   singular-value spectrum and absolute-scale ambiguity. Exact zeros remain
   zero numerically; log bars use an explicitly disclosed 1e-18 display floor.

## Asset consumers, contracts and completion

`test/fixtures/source_stages.npz` is the frozen original-stage fixture consumed
by comparing tests. `evaluation/baseline_reference.npz` contains the separately
identified original full fields consumed solely as the evaluator's method
reference. `notebooks/reference_outputs/original_formulations.npz/json` and
`original_gram.npz` supply original comparisons to tests/notebook;
`original_run.json` records their provenance. `fresh_run.json` and
`fresh_diagnostics.npz` are active notebook outputs for metric/mechanical review.
An unused duplicate of the full stage fixture was moved out of this task.

Judge tests cover missing/corrupt files, exact keys including extra-key rejection,
shape, integer/float16/complex/object dtypes, NaN/Inf, constant reference,
missing evaluator assets, undefined/reversed/nonfinite metric definitions,
zero/uniform/reversed/separately wrong quantities, supported float32 submission,
foreign working directory, nested report creation and report-write failure.
Invalid submissions/evaluator configurations produce error/empty metrics/exit2;
valid poor estimates produce fail/exit1; valid good estimates pass/exit0.
Both required quantities and every independent physical metric are evaluated.

Final native pytest, foreign-CWD Judge, public allowlist/import/call-role/inventory
audit, double mathematical parser results, environment/source byte audit,
final file inventory and hashes are recorded in the private completion evidence.
The complete plans were derived from the selected code and full paper rather
than copied as a proposal; signatures, constants, operators, signs, unit scale,
conditional objective, gauge, nonconsecutive labels and actual notebook caller
flow were semantically reviewed. No library module selects a plotting backend.
README plus public NPZ and pinned requirements suffice without hidden assets.

Final status is determined by `final_evidence.json`, with a separate entry for
each mandatory gate; a pending or failed gate blocks READY. Temporary notebooks,
source archives, paper files, numerical capture scripts, logs, and cache files
are outside the task; the canonical executed notebook and its outputs remain.


### Final observed gate results

Final complete native pytest: **33 passed**, exit **0**, runtime
56.94113429999561 s (`full_tests_process.json` and separate stdout/stderr).
Readonly package/import/call-role/public-data audit: **errors=[]**. Independent
foreign-working-directory Judge: **exit0**. All **52** extracted formulas pass
both MathJax and KaTeX with zero extraction issues or renderer failures
(`qa/summary.json`, `qa/math_render_check.json`). These checks complement the
physical derivations and actual six-figure review; they do not replace them.

The newly added disconnected-graph test initially expected the wrong pressure
sign; with p0-p1=-2 H gamma and H=-.5, p1=-1. Correcting that test expectation
required no solver or threshold change. The failed test log is archived.
Final guard/configuration tests were added after Notebook execution; they do
not change any valid bound numerical result. Final standard CLI and Judge
were independently rerun successfully after those changes.

All mandatory gates are **passed**: primary paper/revision/data/equation mapping;
exact-bound original and cleaned native runs; pre-extraction parity/physics;
constants/units/coordinates; standard CLI and independent Judge; malformed and
poor-output handling; public input sufficiency and no hidden/public URL leak;
minimal environment and local explicit mathematics; complete tests; active
Notebook/native nbconvert0; separated historical/original/fresh metrics;
semantic plans/signatures; final inventory and required notices. Source ZIP
byte audit has zero mismatches. Environment warning about optional PyYAML
formatting was retained; no package was installed to suppress it.

Required qualifications remain explicit: no independent real force truth,
no absolute force/length calibration, known malformed junction exclusions,
source chord-angle convention, estimator disagreement on noisy geometry and
unresolved separate redistribution license for the real example mesh. These
are disclosed scientific/rights limitations, not unexecuted mandatory gates.

`final_evidence.json`, `completion.json`, `archive_index.json` and
`figures/plot_provenance.json` index actual final scientific assets, canonical
Notebook, native process evidence and all task file hashes. Caches and the
unused stage-array duplicate were moved reversibly into the private archive.
The task contains no temporary executed Notebook, figure/PDF or upstream
checkout. The final canonical Notebook retains every actual output.
