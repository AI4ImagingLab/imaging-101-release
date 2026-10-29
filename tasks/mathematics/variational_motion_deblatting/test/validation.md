# Provenance and validation: variational motion deblatting

## Scope and selected scientific method

This task reconstructs a moving floorball's premultiplied color appearance,
opacity and normalized exposure occupancy from **real repository-provided
photographs**, not newly simulated observations. It packages the single-frame
variational deblatting subproblem, not the paper's entire multi-frame tracker or
high-speed benchmark. CPU execution is validated; GPU execution is not claimed.
There is no independent sharp truth for this selected repository example.

The standard method preserves the upstream acquisition operators, blind
alternation, custom ADMM/shrinkage/simplex updates, rotational mask penalty,
initialization, final trajectory fitting, truncated conjugate gradients and
stopping rules. It **explicitly corrects a verified upstream constraint-projection
defect**. The unchanged original source and an explicitly named local original
mode are separately reproduced. No corrected result is labeled unchanged parity.

Primary paper actually downloaded/read: Kotera et al., *Intra-Frame Object
Tracking by Deblatting*, ICCV Workshops 2019, final CVF conference PDF:
https://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kotera_Intra-Frame_Object_Tracking_by_Deblatting_ICCVW_2019_paper.pdf
DOI: 10.1109/ICCVW.2019.00283. Related preprint identifier 1905.03633v2 was
remotely verified as revised 2019-06-20. The source also cites the later journal
extension, *Tracking by Deblatting*, IJCV 2021,
https://doi.org/10.1007/s11263-021-01480-w. Its Springer PDF endpoint returned
3038 bytes of anti-bot HTML, **not a PDF**. It is archived as
springer_response.html; no full-text verification of that secondary version is
claimed. The available 2019 paper and exact executable source establish the
selected single-frame equations. Code-specific additions are identified below.

## Website search, revision and redistribution

The public source is https://github.com/rozumden/deblatting_python, remotely
resolved through the repository commits API on 2026-10-18 to
`2536898810cfd1d75b0bed6acac30533981c08da` (committer date
2022-01-12T12:06:46Z). This is an actually verified remote revision, not an
assumed local tracking branch. The exact revision zip was downloaded through the
API and archived. Search also screened public TV/inpainting and deconvolution
repositories; this candidate was selected because it supplies real affected
observations **and** a usable background and has explicit MIT licensing.

Relevant exact-revision assets:

- https://raw.githubusercontent.com/rozumden/deblatting_python/2536898810cfd1d75b0bed6acac30533981c08da/imgs/floorball1.png
- https://raw.githubusercontent.com/rozumden/deblatting_python/2536898810cfd1d75b0bed6acac30533981c08da/imgs/floorball_bgr.png
- https://raw.githubusercontent.com/rozumden/deblatting_python/2536898810cfd1d75b0bed6acac30533981c08da/deblatting.py
- https://raw.githubusercontent.com/rozumden/deblatting_python/2536898810cfd1d75b0bed6acac30533981c08da/run.py

The repository labels the selected path `test_real`, supplies both PNGs, and
describes the background as available scene information, for example from a
median of previous frames. The exact camera, exposure, radiometric response and
background-estimation history of these example PNGs are not documented. No such
metadata is invented. The independent background is legitimate scene input; it
is not estimated from a hidden sharp target. No target truth is present.

The original repository license is MIT, copyright (c) 2022 Denys Rozumnyi; its
complete text is preserved verbatim in the consolidated licensing section below. No separate image-specific license or
third-party notice is supplied with these example assets; redistribution follows
the repository's stated MIT terms, with this provenance limitation recorded.
The antialiased line primitive is translated to local readable Python from
scikit-image v0.22.0 `skimage/draw/_draw.pyx`, function `_line_aa`, retaining its
Zingl rasterization conventions. The scikit-image BSD license text is appended
to the consolidated licensing section of this validation record. This is a small general rasterization primitive, not a domain inverse
solver. No scikit-image or upstream reconstruction library is a cleaned dependency.

## Data conversion and immutable source hashes

The original upstream detector, using only the observed photograph and supplied
background, returns bounding box (333,736,466,837) and approximate minor-axis
diameter 36.11291853256522 pixels. The upstream example's rounded half-diameter
margin is 18 pixels. Apply its exact crop: rows [315,484), columns [718,855),
retaining every pixel, no interpolation. OpenCV's upstream BGR ordering is
preserved. Convert integer camera values to float64 by division by 255 and add
the leading batch axis. Both public keys have shape (1,169,137,3).

No new synthetic generation, noise injection, target-dependent calibration,
resampling, hidden seed or target initialization is used. `src/generate_data.py`
is correctly omitted. The public raw file contains only observation and
background; it does not contain reconstructed F,M,H or a sharp target.

| Asset | SHA-256 |
|---|---|
| Exact upstream zip | a3087a5e83a07e3273e59fc78c284712e49738cac26d4d90e77b41a12a4900fd |
| Primary conference PDF | 3b3bbd46ef4a511216a717346ddbfe606d0323d9bc3580f53344d3df9b731e6d |
| Original floorball1.png | 1a06e830af9226c0cc77cf850e66c78a3834d9db0be56a113ca1d5eac8297495 |
| Original floorball_bgr.png | e4c85eed9cf7980a0ee1886181b401aaa354a88c9bda888c7355881dd924216a |
| Bundled data/raw_data.npz | 2bbb015ddc2ec6b049ef0419b3210fe042d3ad6ca66a871c051645194b3127cb |

The final archive index and per-file task hashes are recorded in the external
archive's archive_index.json and task_inventory.json. Local source changes are
uncommitted and identified by their exact file hashes, not by a fabricated Git
revision. No task changes are committed by this cleaning operation.

## Maintainer-only archive and environments

All original sources, downloaded papers, capture scripts and raw logs reside
outside the task under:

`D:/yan1/agent/AutoSkill/imaging-101-maintainer/batch10/task01_mathematics/`

| Archive item | Role |
|---|---|
| revision.json, upstream.zip, upstream/ | Exact remote revision response and complete unmodified checkout. |
| paper_iccvw2019.pdf, paper.txt | Primary published method and extracted text. |
| springer_response.html | Failed secondary journal PDF retrieval, accurately identified as HTML. |
| skimage_draw_v0.22.0.pyx, skimage_LICENSE.txt | Source and notice for local line rasterizer. |
| capture.py, upstream.log, upstream_rerun.log | Original unchanged oracle invocation and independent rerun. |
| capture_stages.py, stages.log | Original first-update/operator capture. |
| corrected_oracle.py, corrected_oracle.log | Separately identified source run with corrected iterative Dykstra. |
| projection_audit.py, projection_audit.json | Independent convex projection audit against SLSQP. |
| audit_same_backend.py, same_backend.log, same_backend_audit.json, backend_comparison.json | Controlled numerical sensitivity isolation. |
| acceptance_before_extraction.txt | Numerical acceptance criteria frozen before the first local result was judged. |
| cleaned_environment.txt, oracle_environment.txt | Python, platform and NumPy numerical-backend reports. |
| cleaned_corrected.log, initial_tests.log, unit_tests.log, normal_tests.log, final_tests.log, release_tests.log | Actual commands' execution evidence, including early failures and corrected checks. |
| notebook.log, notebook_final.log, notebook_*figure_*.png | Complete notebook execution logs and inspected rendered figures. |
| bound_inputs.npz, upstream_result.npz, corrected_oracle_result.npz, stage_fixtures.npz, upstream_first_update.npz, same_backend_dykstra_result.npz | Independently identified numerical captures. |
| extract.py, build_notebook.py, tmp/ | Maintainer extraction/build tooling and development temporary files, never solver dependencies. |

Oracle environment is separate from the shared cleaned environment: Python
3.11.11, NumPy 1.26.4, SciPy 1.10.1, scikit-image 0.22.0, OpenCV-headless 4.8.1.78,
on Windows 10 build 26200 x86-64 CPU. NumPy reports OpenBLAS 0.3.23.dev ILP64.
These versions retain the original positional SciPy CG call and sparse `.A`
attribute, so the original numerical source is run without adaptation.

Cleaned environment: shared Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1,
Matplotlib 3.11.1, Windows 10 build 26200 x86-64 CPU, OpenBLAS
0.3.31.188.0 ILP64. pytest 9.1.1, nbformat 5.11.1, nbconvert 7.17.1 and the
Python kernel are shared development tools, not task runtime requirements.
Execution sets OPENBLAS_NUM_THREADS=1. Full numerical-backend reports are
archived. No packages are installed into or upgraded in the shared environment.

The cleaned source's only compatibility adaptations to the inverse code are
explicit `rtol=..., maxiter=...` CG keywords and `.toarray()` for sparse dense
conversion. Interactive GUI calls are removed. Upstream visualization and full
checkouts never enter the cleaned runtime. No CUDA/GPU claim is made.

## Independent original run and correction investigation

Original capture command, with its interpreter from the isolated oracle
environment:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
& oracle-env/Scripts/python.exe capture.py
```

The capture script imports the **unmodified** upstream `deblatting.py`, obtains
the exact original crop using its detector and executes `estimateFMH` on those
bound observations. Only verbose printing/GUI display is disabled. No seed is
needed for the deterministic inverse method. The first full run exits 0 in
186.58116519998293 s. Independent unchanged rerun exits 0 in
176.45612400001846 s and all three saved arrays are bitwise identical. The only
numerical-source warning is SciPy SparseEfficiencyWarning for construction of
the rotational matrix; it is not a failed inversion. PowerShell's redirected
stderr also wraps this warning in NativeCommandError diagnostic text even though
the Python exit status is 0. This is recorded without mistaking stderr for failure.

The original first-update fixture separately freezes kernel, appearance and
opacity after one warm-state alternating update on the same data. General
operator probes use a private RNG seed 94 solely for numerical test inputs;
this is not an observation-generation seed or public target specification.

### Verified source discrepancy

The source's `project2pyramid` assigns `mf_old = mf` and subsequently changes mf
in place before residual calculation. Independent SLSQP solves 50 four-dimensional
convex projection cases (audit RNG seed 25). Original projector versus the true
Euclidean projection has maximum coordinate discrepancy 0.6152191049822793.
Its maximum F-M constraint excess on these probes is 0.0051115824978453794.
A separate adversarial test gives exactly 0.0068359375 constraint excess. It is
therefore incorrect to call the source implementation exact Dykstra projection.

The standard method replaces only this operation by the direct sorted active-set
solution of the same convex projection. Against independent SLSQP the maximum
coordinate difference is 2.220446049250313e-15, with normal-cone/KKT and explicit
feasibility tests. The original approximation remains explicitly named
`upstream_cyclic` for unchanged-source parity; it is not the default method.

A **separately modified oracle** changes the source projector to correct Dykstra
with saved iterate copies, four half-space projections per cycle, maximum 200
cycles, and maximum iterate-change tolerance 1e-12. Its full pipeline exits 0 in
166.9816602000501 s. All other source numerical updates remain unchanged. This
oracle, not the defective original result, supplies the standard-method hidden
reference. It is never relabeled truth or unchanged upstream.

The correction changes the nonconvex estimate. Fresh corrected output compared
to the unmodified original has appearance NCC 0.9941132820024281, opacity NCC
0.9997423659780096 and kernel NCC 0.8005917783535947. Its observation RMSE changes
from the original 0.0047773183200332196 to 0.004833980023591831. We claim the proximal
operation now satisfies the declared mathematics; we do **not** claim that this
proves improved unknown sharp-target accuracy or a lower data residual.

### Numerical stability and predeclared tolerances

Before the first local reconstruction was judged, acceptance_before_extraction.txt
fixed operator atol=1e-12, rtol=1e-11 and output NCC>=0.99999,
dynamic-range NRMSE<=0.0001 for each required quantity, without alignment. These
bounds are retained. Original independent repeatability is bitwise exact; full
cross-environment and independent-projection runs quantify a nonzero allowance.

Local original-mode full output versus unmodified source has appearance NRMSE
1.0808871897072547e-05, opacity NRMSE 2.5428498385620535e-05 and kernel NRMSE 0.
Fresh exact-mode versus independent corrected oracle has appearance NRMSE
1.278903286868898e-05, opacity NRMSE 1.662658634008976e-05 and kernel NRMSE
1.3602484631913632e-17. Corresponding NCCs are approximately one; exact values
are retained in metrics.json and fresh_measurements.json. No boundary was
increased after inspecting these values.

An additional controlled same-backend run replaces the exact local projector by
the independent corrected Dykstra while retaining NumPy 2.4.6/SciPy 1.17.1.
Across 149 actual projection calls, the maximum instantaneous coordinate
disagreement is only 1.0651340920375674e-12. Nevertheless, final appearance and
opacity NRMSE relative to exact mode are 2.2910298091465953e-05 and
2.8851034846665252e-05, with identical fitted kernel. Running the same corrected
Dykstra across the old/new numerical environments gives final appearance/opacity
NRMSE 1.5372859455164038e-05 / 1.576058775695562e-05. Thus the full 1e-5 variations
cannot be attributed solely to either backend changes or raw proximal accuracy:
tiny arithmetic changes propagate through finite/truncated CG and the alternating
inverse iterations. These controls establish the scale but do not isolate every
CG versus nonconvex-path contribution. The original 1e-4 numerical criterion
comfortably covers the measured independent variations without being vacuous.

An early first-update assertion used 1e-10 elementwise tolerance on finite CG
solutions and failed with appearance maximum absolute difference
5.27184261e-07 across versions. Exact difference/embedding probes still matched.
The test now distinguishes primitive-operator parity from finite inverse-update
parity, applying the already predeclared output NRMSE criterion to the latter.
Independent direct-convolution matrices validate both solver normal operators to
2e-12, including the cross terms, adjoints and positive definiteness. The source
CG relative residual tolerance is 1e-5 with only 25 iterations. The initial failed
assertion and its evidence remain archived; no solver parameter or evaluation
boundary is relaxed to hide it.

## Equation/function/test traceability

| Paper stage / implemented addition | Upstream file/function | Local implementation | Independent validation |
|---|---|---|---|
| Exposure/occlusion equation (1) | deblatting.py estimateH/estimateFM FFT products; utils.py fmo_model for visualization uses a different finite support convolution convention | physics_model.forward; explicit FFT normal products in solvers | Direct circular-convolution matrix tests for both H and joint F/M; impulse/zero-foreground checks; independent Judge model. |
| Compact field origin / extraction | deblatting.py psfshift, psfshift_idx, vec3 | physics_model matching functions | Frozen upstream embedding; even-origin impulse at (18,18); index-map inverse checks. |
| Single-frame alternating inverse problems (3)-(7) | deblatting.py estimateFMH/estimateH/estimateFM | solvers explicit warm-state blind alternation and CG closures | Original first-update fixture and unchanged full pipeline; corrected independent oracle full pipeline. |
| Zero-support TV, vector shrinkage | createDerivatives0, lp_proximal_mapping | physics_model sparse operators; solvers custom auxiliary/multiplier updates | Independent padded differences, adjoint identities, soft-threshold cases and direct normal matrices. |
| Positive exposure regularization / selected code unit-mass variant | proj2simplex; Params.sum1 | physics_model.proj2simplex and explicit H split | Simplex KKT, nonnegative unit-mass final exposure; frozen upstream probe. |
| Convex F/M coupling and paper Dykstra discussion | project2pyramid | exact active-set projection by default; original cyclic approximation only in explicitly named diagnostic mode | Independent SLSQP, KKT normal-cone probes, regression for source defect, corrected-Dykstra oracle and actual-call sensitivity audit. |
| Source additions: opacity TV and rotational symmetry | estimateFM; createRnMatrix; lambda_R | Same local matrix construction and mask normal term | Original full pipeline parity; direct positive-semidefinite derivative normals; shared source signature/math review. These additions are code-specific, not claimed present in the 2019 objective verbatim. |
| Optional previous appearance/template terms | F_T/M_T and lambda_T | Retained callable options, absent in this single-frame runtime | Plans and caller inspection confirm no hidden template or target initialization is used. |
| Trajectory fitting family (8), selected code simplified line fit | psffit plus utils.renderTraj | physics_model.psffit plus preprocessing local Zingl rasterizer | Fitted kernel matches original/corrected oracle; horizontal/single-pixel primitive cases and full field comparisons. No RANSAC/tracker claim. |
| Stop/init/parameters | Params, estimateFMH, StateH, StateFM | solvers and main.reconstruct | Frozen first update, exact original full run and standard CLI full tests. |
| Numerical agreement / observation consistency | Newly explicit benchmark definitions, not upstream tracking IoU | Independent Judge; visualization.metrics; notebook | Known analytical metric cases, all three fields, poor outputs and full contract tests. |

The paper introduces a broader tracker and previous-frame appearance terms. The
selected repository's single-frame example has no previous sharp appearance and
therefore no active template penalty. The runtime FFT inverse model is circular;
the upstream optional visualization helper utils.fmo_model uses fftconvolve with
finite 'same' boundaries. We explicitly select and validate the **inverse code's**
convention, rather than silently mixing those two operators. The public contract,
Judge, local forward model and independent direct-convolution tests agree.

## Fresh standard execution, metrics and physical limits

Standard command is `python -m src.main` from this task directory. It exits 0 and
writes all declared batch-first float64 arrays in output/reconstruction.npz;
an independently logged run took 196.330 s. The independent Judge command is:

```text
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
```

It returns 0/pass. Fresh metrics against the corrected-Dykstra standard reference:

| Quantity | Cosine NCC | Dynamic-range NRMSE |
|---|---|---|
| Appearance | 0.9999999997377993 | 0.00001278903286868898 |
| Opacity | 0.9999999999554233 | 0.00001662658634008976 |
| Exposure kernel | 1.0 | 1.3602484631913632e-17 |

Observation RMSE is 0.004833980023591831 normalized intensity, versus
0.03626009717879427 for the background alone (about 7.50-fold residual reduction).
The fixed consistency boundary 0.0051 allows a small numerical/model allowance
above the independently reproduced source residuals; it remains far below the
background-only error. All three comparison quantities must also meet their
strict agreement bounds; observation fit alone is insufficient.

The final finite-ADMM primal appearance minimum is -0.017576679782593117;
opacity spans 0.0009769078918804137 to 0.9990532102718381; maximum F-M excess is
0.002243585764727274. These are explicitly measured numerical constraint
residuals of the returned primal linear solve; the proximal auxiliary itself
is physically feasible. We retain the original stopping policy and do not clip
or relabel the result. Physics tests require appearance >-0.02, opacity in [0,1]
and F-M excess <0.003 for this validated finite computation. Exposure occupancy
is nonnegative, with mass 0.9999999999999999. These finite residuals, the unknown
camera response and the fitted straight trajectory are practical scientific
limitations, not exact positivity/true-target claims.

No correlation or reconstruction metric is computed against a fabricated target.
NCC is dot-product cosine without mean subtraction; NRMSE is RMS error divided
by reference dynamic range. Zero estimates receive NCC=0; constant/zero reference
arrays are evaluator errors. No crop, shift, phase, reflection or scale alignment
is applied. All machine-readable numbers remain full precision and finite.

## Tests, notebook and review evidence

The final test command is `python -m pytest test/ -v`. The suite contains 32
checks including original frozen primitive/update/full-pipeline parity, corrected
oracle full pipeline, independent direct-convolution normal operators and
adjoints, constraint projection/KKT, impulse conventions, analytical metric
cases, public contracts and standalone Judge pass/fail/error behavior. Missing
files/keys, corrupt NPZ, wrong shape, float32/complex/object dtype, NaN/Inf,
missing evaluator assets, malformed thresholds, truth precedence, outside-CWD
evaluation and report-write failure are explicitly exercised.

Initial source-defect and overly strict finite-CG assertions failed and were
investigated as described above. An intermediate complete suite passed all 27
then-present checks in 374.41 s; additional independent operator/update/contract
checks were subsequently added. The final complete command exits 0 with
**32 passed in 399.36 s**, recorded in release_tests.log. Both unchanged-source
and corrected standard full reconstructions are actively rerun by this suite.

Notebook command:

```text
jupyter nbconvert --to notebook --execute notebooks/variational_motion_deblatting.ipynb --output variational_motion_deblatting_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=900
```

The complete active reconstruction executes successfully. Final fresh
reconstruction cell runtime is 183.2463253000751 s. Per-cell timeout 900 s is
justified by repeated approximately 170-200 s CPU runs, not an artificial
seconds-only requirement. NumPy/SciPy execute all reconstruction steps through
cleaned src modules. All code cells execute without errors; the three inline
figures include real observation/background/prediction/residual, every estimated
unknown across four identified measurements, and paired per-quantity NCC/NRMSE
bars plus full-precision numerical tables.

All three figure PNG outputs were extracted to the external archive and visually
inspected. An initial NCC display lower bound 0.9 hid the original exposure
kernel's approximately 0.80 NCC; it was corrected to show the full [0,1] range
and the **entire** notebook was executed again. The final metric figure was
inspected and retains that scientific difference. Labels, crop views, common
kernel scale and residual colorbar are legible. Display-only clipping is
explicitly labeled; scientific arrays/metrics are not clipped. Original run.py
comparisons of exposure kernel/appearance/opacity are retained and expanded;
no missing method is filled by zeros or disguised archived values.

The final executed notebook replaces the unexecuted source notebook at the
authoritative notebook path. Only that single notebook is retained. Temporary
executed copies, parity submission NPZ and Python/pytest cache directories are
removed from the task after checks. Original execution logs and inspected PNGs
remain outside the task. The notebook may emit the documented Windows ZMQ
selector/TCP kernel warnings; these are environment warnings, not cell failures.

## Array inventory and documented consumers

| Task asset | Origin / role | Consumer |
|---|---|---|
| data/raw_data.npz | Original paired real PNG crops, no synthetic observation | CLI, notebook, loader contract and Judge observation consistency. |
| evaluation/baseline_reference.npz | Explicitly modified independent corrected-Dykstra oracle F,M,H; no actual truth | Standalone Judge and corrected full-pipeline comparison. |
| test/fixtures/upstream_operators.npz | Unchanged source seeded numeric probes | Primitive stage parity test. |
| test/fixtures/upstream_first_update.npz | Unchanged source first H and joint F/M updates on bound observations | Alternating-update parity test. |
| test/fixtures/upstream_pipeline.npz | Unchanged source full F,M,H on bound observations | Explicit original-mode full pipeline parity test. |
| notebooks/reference_outputs/upstream_run.npz | Identified original source full result | Notebook original-versus-corrected comparisons. |
| notebooks/reference_outputs/cleaned_upstream_parity.npz | Fresh local full computation retaining original projection | Notebook separately identified local original-mode row. |
| notebooks/reference_outputs/corrected_dykstra_oracle.npz | Identified corrected independent oracle | Notebook scientific comparison and metric plots. |
| notebooks/reference_outputs/fresh_cleaned.npz | Actively computed final notebook reconstruction | Notebook Judge evaluation, numerical audit and measurement record. |
| notebooks/reference_outputs/fresh_measurements.json | Fresh runtime, Judge, constraint and comparison metrics | This validation record and machine-readable maintainer review. |
| output/reconstruction.npz, output/judge_report.json | Generated standard submission and report | Standard CLI/Judge handoff and full-pipeline tests. |

References duplicated across fixture/review/evaluator boundaries have distinct
documented consumers; no unused numerical file or independently invented truth
is retained. Shared source/runtime path inspections cover every .py module,
Judge, fixture test and notebook code cell. Only NumPy/SciPy/Matplotlib and
standard library are used. SciPy FFT/sparse/CG and test-only SLSQP are general
numerical primitives; every acquisition, objective, projection, ADMM step and
normal-operator product remains local executable mathematics.

## Plan/signature/units review and release gates

README was checked against loader shapes, intensity/channel conventions, compact
origin, physical units, output fields and Judge behavior. Both plans were derived
from actual source/paper equations and inspected for signs, factors, objective
normalization, zero/circular boundaries, options actually active at runtime,
initialization, defaults, callable signatures and stateful execution. In
particular, compact origin is (18,18), not (17,17); a dedicated independent
impulse test enforces the even-sized convention. Joint normal matrices were
compared against direct non-FFT acquisition matrices, verifying the physically
required positive quadratic cross-block assembly. Notebook/Judge/visualization
metrics have independent analytical cases and use identical definitions.

Public-surface inspection covers README, both raw NPZ keys and requirements.
No GitHub-domain URL, hidden reconstruction, target-derived calibration,
privileged seed/object, standalone metadata JSON or source dependency is exposed.
README gives a complete mathematical input/output contract, domain metadata,
acquisition table and conceptual hints. Hidden maintainer material is necessary
for reproducibility but is not needed to formulate or solve the public inverse
problem.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Primary paper, exact remote revision, data hashes and equation/function/test mapping | passed | Primary CVF PDF, API revision and traceability table. Secondary journal full text is not claimed verified. |
| Original and cleaned runs on exact bundled real observations | passed | Unchanged source run and bitwise rerun; local CLI and correction-labeled independent oracle. |
| Independent primitive/update/full parity and physics | passed | Frozen source fixtures; direct acquisition normal matrices; exact projection SLSQP/KKT; full original and corrected modes. |
| Constants, precision, units and coordinates preserved | passed | README/loader, even-origin impulse, upstream fixtures and operator review. |
| Standard CLI and independent Judge | passed | Fresh all-quantity output and report pass with observation consistency. |
| Judge rejects poor/malformed data and handles evaluator/report failures | passed | Contract tests including return codes 0/1/2 and actual-truth precedence. |
| Public allowlist sufficient and free of privileged data / GitHub URLs | passed | README, raw observation/background and minimal requirements inspection. |
| README metadata and complete output contract; no metadata JSON | passed | Public-contract tests and manual equation/units review. |
| Minimal shared runtime without upstream domain packages | passed | Shared environment full CPU runs, tests and notebook; isolated oracle-only dependencies. |
| Local executable core mathematics and symbol/call-role audit | passed | Explicit FFT/sparse acquisition, custom ADMM, shrinkage, simplex and corrected projection; no domain solver import. |
| Final complete pytest test/ command | passed | Actual release_tests.log: 32 passed in 399.36 s, exit 0. |
| Active notebook reconstruction, evaluation, complete comparisons and nbconvert | passed | Final full execution, 183.2463253000751 s reconstruction and three visually inspected figures. |
| Historical/original/corrected/fresh provenance and metrics | passed | Separate arrays, quantitative source-defect investigation and non-relaxed predeclared bounds. |
| README/plans/source equations/signatures/execution agree | passed | Semantic and interface review, independent operator tests and explicit limitations above. |
| Final inventory, cleanup and license notices | passed | Task Template inventory checked; generated caches, parity submission and temporary executed copy removed; MIT/BSD notices retained; archive/task hashes indexed. |


## Independent coordinator review (2026-10-18)

A separate coordinator reran the standard CLI, the standalone Judge from a different working directory, the complete pytest suite, and the active notebook through nbconvert in the shared CPU environment. All four commands returned exit code 0. Command arguments, timing, stdout/stderr, the executed review notebook, and extracted figures are archived at `batch10/coordinator/variational_motion_deblatting/`; the structured check report is `batch10/coordinator/variational_motion_deblatting-review.json`. The coordinator inspected all embedded comparison figures and reviewed the implemented source and mathematical plan. Packaged reference snapshots were preserved around the notebook rerun; any rerun snapshots are separately archived.


## Consolidated original licenses and attribution

On the user's documentation-layout request, standalone license/notice files were removed after their exact original bytes were archived privately. The complete original texts, including every copyright, condition and disclaimer, are reproduced verbatim below. This changes the notice location, not the governing license.

### Original LICENSE (verbatim)

```text
MIT License



Copyright (c) 2022 Denys Rozumnyi



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





Files: *

Copyright: 2009-2022 the scikit-image team

License: BSD-3-Clause



Files: doc/source/themes/scikit-image/layout.html

Copyright: 2007-2010 the Sphinx team

License: BSD-3-Clause



Files: skimage/feature/_canny.py

       skimage/filters/edges.py

       skimage/filters/_rank_order.py

       skimage/morphology/_skeletonize.py

       skimage/morphology/tests/test_watershed.py

       skimage/morphology/watershed.py

       skimage/segmentation/heap_general.pxi

       skimage/segmentation/heap_watershed.pxi

       skimage/segmentation/_watershed.py

       skimage/segmentation/_watershed_cy.pyx

Copyright: 2003-2009 Massachusetts Institute of Technology

           2009-2011 Broad Institute

           2003 Lee Kamentsky

           2003-2005 Peter J. Verveer

License: BSD-3-Clause



Files: skimage/filters/thresholding.py

       skimage/graph/_mcp.pyx

       skimage/graph/heap.pyx

Copyright: 2009-2015 Board of Regents of the University of

           Wisconsin-Madison, Broad Institute of MIT and Harvard,

           and Max Planck Institute of Molecular Cell Biology and

           Genetics

           2009 Zachary Pincus

           2009 Almar Klein

License: BSD-2-Clause



File: skimage/morphology/grayreconstruct.py

      skimage/morphology/tests/test_reconstruction.py

Copyright: 2003-2009 Massachusetts Institute of Technology

           2009-2011 Broad Institute

           2003 Lee Kamentsky

License: BSD-3-Clause



File: skimage/morphology/_grayreconstruct.pyx

Copyright: 2003-2009 Massachusetts Institute of Technology

           2009-2011 Broad Institute

           2003 Lee Kamentsky

           2022 Gregory Lee (added a 64-bit integer variant for large images)

License: BSD-3-Clause



File: skimage/segmentation/_expand_labels.py

Copyright: 2020 Broad Institute

           2020 CellProfiler team

License: BSD-3-Clause



File: skimage/exposure/_adapthist.py

Copyright: 1994 Karel Zuiderveld

License: BSD-3-Clause



Function: skimage/morphology/_skeletonize_cy.pyx:_skeletonize_loop

Copyright: 2003-2009 Massachusetts Institute of Technology

           2009-2011 Broad Institute

           2003 Lee Kamentsky

License: BSD-3-Clause



Function: skimage/_shared/version_requirements.py:_check_version

Copyright: 2013 The IPython Development Team

License: BSD-3-Clause



Function: skimage/_shared/version_requirements.py:is_installed

Copyright: 2009-2011 Pierre Raybaut

License: MIT



File: skimage/feature/_fisher_vector.py

Copyright: 2014 2014 Dan Oneata

License: MIT





License: BSD-2-Clause



Redistribution and use in source and binary forms, with or without

modification, are permitted provided that the following conditions

are met:

1. Redistributions of source code must retain the above copyright

   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright

   notice, this list of conditions and the following disclaimer in the

   documentation and/or other materials provided with the distribution.

.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS

``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT

LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR

A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR

CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,

EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,

PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR

PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF

LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING

NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS

SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



License: BSD-3-Clause



Redistribution and use in source and binary forms, with or without

modification, are permitted provided that the following conditions

are met:

1. Redistributions of source code must retain the above copyright

   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright

   notice, this list of conditions and the following disclaimer in the

   documentation and/or other materials provided with the distribution.

3. Neither the name of the University nor the names of its contributors

   may be used to endorse or promote products derived from this software

   without specific prior written permission.

.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS

``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT

LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR

A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR

CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,

EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,

PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR

PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF

LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING

NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS

SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



License: MIT



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


## Documentation revision: six-part active review and consolidated notices

The user requested the exact six ordered notebook parts Problem, Data, Physics,
Method, Results, Conclusion and deeper implemented plans, using the EHT task as
a depth reference. This revision read that reference and the actual source,
Judge, tests, public contract and prior provenance before rewriting. Data tables
now enumerate every visible key with its actual batch-first shape, dtype, units
and legitimate observation/geometry role, and the Data cell actually loads them.
Physics is separated from inverse method, with explicit coordinate and boundary
conventions. Method gives the actual initialization, updates, normalization,
parameters and stopping policy. Every original scientific plot, comparison group
and numeric table remains in Results, which actively reconstructs using src.
Conclusion retains task-specific scientific limitations rather than claiming
method agreement proves physical truth.

The approach plan now explains the implemented derivation, operators/transposes
where relevant, units, normalizations, algorithmic signs, inner/outer stopping,
evaluation and independent validation. The design plan maps every shipped
callable to its exact signature and actual input/output array shape, dtype, units,
defaults, exceptions and state/file side effects, and distinguishes CLI, review
notebook and independent evaluator flows. No executable algorithm, observation,
reference image or metric boundary was changed. Source copyright docstrings that
pointed at removed notice filenames were redirected only to this record.

Standalone license/notice files were first copied byte-for-byte to the private
revision archive; complete original notices and copyright attribution are retained
in the consolidated section above. Current explanatory references point here;
original filename headings inside verbatim records are historical labels.
The governing license is unchanged.

Not rerun for this documentation-only revision; the prior full source suite passed32 tests, and the full active reconstruction notebook plus independent Judge were rerun.

Private revision evidence: `../imaging-101-maintainer/batch10/revision_docs/task09/variational_motion_deblatting/` (repository-relative sibling archive). Its `before/` preserves the prior plans, notebook, references and notices; `before-hashes.json` freezes the old package. `verification.json` records exact commands, runtimes, return codes and preserved reference hashes. `notebook.log`, `judge.log` and, where rerun, `pytest.log` are actual command output.

The full selected reconstruction notebook was executed through shared-environment `python -m jupyter nbconvert --execute`, using a writable private TEMP/TMP, shared Scripts on PATH and CPU thread count1. Per-cell timeout900s covers the slow blind inverse as well as faster tasks. Notebook exit code: 0; wall time 206.30554239999037 seconds; independent Judge exit code 0. All code cells ran without errors. The 3 extracted figures were visually inspected for readable axes, labels, scientific comparison groups and numerical consistency. The executed output was saved back to the authoritative notebook path.

Before execution all `notebooks/reference_outputs/` bytes were snapshotted. New measurements were archived separately under `new-reference-measurements/`, then every original reference snapshot was restored byte-for-byte; `reference_hashes_preserved=true`. The shipped notebook and output submission are freshly computed, while restored snapshots retain their original measurement identity and hashes. This is an evidence-preservation procedure, not a cache substitute for reconstruction.

Current package hashes, including changed plans/README/notebook/source notice text, are indexed by private `after-hashes.json`; older source/task inventories remain historical records under their original archive rather than being misrepresented as hashes of this revision. Public raw NPZ, baseline and metric-boundary file SHA256 values were checked unchanged. No temporary executed notebook, figure PNG, source checkout or cache was added to the released task.


## 2026-10-29 public benchmark semantic version v2

V2 judges all three arrays using the published joint data/TV/rotational-opacity objective, with observation RMS<=.0051, physical violation<=1e-6 and joint objective ratio<=1.00000001 versus an independently captured and feasibly projected standard. F is premultiplied bounded encoded-camera intensity, 0<=F<=M<=1; no HDR physical bound is inferred. H is nonnegative with unit mass. The legacy source output violated the box/pyramid by.01757668; this was not rounding and is preserved as failed legacy evidence. New source reconstruct performs an exact final pyramid projection and kernel normalization, then a feasible objective-decreasing polish. The four scale variables form a convex quadratic with explicit TV-linear and opacity-rotation terms; SLSQP uses analytic gradient, ftol1e-13,maxiter200, positive RGB scales and opacity bound1/max(M), plus F<=M linear inequalities. Five subsequent explicit projected-gradient steps use the true joint objective/gradient, initial step.01, at most24halvings and strict1e-14decrease. The existing inverse core and upstream_cyclic diagnostic mode remain explicit and separately labeled. This is a declared semantic/source correction, not unchanged source parity. Original output/baseline/source are archived with hashes outside task. The new hidden method baseline is the exact feasible projection of independently captured original arrays, not truth or a relabeled fresh output. Independent feasible improvement ratio.9999960287568 passes; negative-F fails. Genuine main completed99.343s and active notebook103.532s, exit0; full32-test suite passed219.86s. All3computed notebook figures were actually reviewed, retaining original/parity/corrected-source/fresh comparisons. Raw arrays and real truth were unchanged. Previous validation entries describe their original dated execution and v1 method-consistency semantics; v2 does not retroactively convert them into blind-agent results. The old Judge/metrics and failure evidence remain in benchmark_readiness_20261029/semantic_v2/variational_motion_deblatting/. No fresh blind-agent reconstruction was run in this review.


2026-10-29 final inventory: generated output extras were copied byte-for-byte with verified SHA256 into maintainer benchmark_readiness_20261029/group2_task_output_archives/variational_motion_deblatting/ and removed from the release output directory. Archive index is group2_task_output_archives/index.json. Canonical reconstruction.npz and judge_report.json remain; source/actual comparison provenance is preserved. These archives are not runtime inputs.
