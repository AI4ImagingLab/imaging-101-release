# Provenance and validation: zebrafish indentation modulus

Status: local task gates passed; awaiting batch-root integration review.
Validation date: 2026-10-27 (Asia/Shanghai). Domain: mechanics.

## Scientific binding and real-observation evidence

Paper: Müller, Abuhattum, Möllmert, Ulbricht, Taubenberger and Guck,
*nanite: using machine learning to assess the quality of atomic force
microscopy-enabled nano-indentation data*, BMC Bioinformatics 20:465,
published 10 September 2019, DOI 10.1186/s12859-019-3010-3.
Primary full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC6734308/.
Selected method is its contact-mechanics fitting stage, not the trained
quality-classification experiment. Section Nanoindentation measurements
identifies CellHesion200/JPK, spherical polystyrene beads, radius 18.64 µm,
nu=0.5 and the corrected spherical-contact formula. The paper states rating 5
is just usable; scores 0–4 identify unusable curves. Its classifier/ROC training
experiment is outside this inverse-task scope and is not silently represented
as reproduced. Notebook retains quality-group comparisons and actual poor-curve
diagnostics while explicitly explaining this scope.

Data: https://doi.org/10.6084/m9.figshare.7297202,
API snapshot https://api.figshare.com/v2/articles/7297202.
Exact downloaded file: https://ndownloader.figshare.com/files/13481393,
zef18.h5, 529,200,018 bytes. Author metadata explicitly describes 1,132 raw AFM
force-indentation curves, preprocessing/fits and manual ratings. File contains
original binary JPK acquisitions, rather than just plotted results. Those files
record instrument `JPK00758-CellHesion-200`, calibrated cantilever stiffness,
sampling, dates and stage positions. Classification: **verified_real**.
The calibrated force/height arrays are author/instrument processed real
measurements, not simulated ideal curves. Figshare data license: CC0.

Downloaded MD5 0c63f427f4be25b1f27c33104be317f0 agrees with author API metadata.
SHA256 of zef18.h5:
`63d89a8aa911a255fb4597b2c1801e30ea14810feef1bb42c11ef10f02a1d055`.
No target-derived calibration is supplied. Spring constants and positions
come from original acquisition metadata. Manual ratings and historical fit
parameters remain hidden. No physical ground-truth modulus is available.

Final benchmark: every one of the 454 original acquisitions with pre-existing
manual quality >=5, including all samples of each approach/retract record.
Acquisitions span 172 source ROI folders and 79 instrument session IDs during
1–3 July 2015. These identifiers must not be interpreted as specimen counts.
The complete archive has 219 ROI folders and 100 session IDs. The public NPZ
preserves 6,355,919 measured samples with variable-length packing. No cropping,
downsampling, interpolation or dtype reduction occurs. The raw NPZ is
89,663,062 bytes. Equality with original parsed force, height, segment and
calibration arrays is exact by construction and independently checked during
the direct-final-NPZ oracle rerun.

Scope history is retained: initial author and cleaned exploratory runs processed
all 1,132 curves. Some preassigned quality-0 curves produced apparent moduli near
6.2e-13 Pa. Such converged fits are not meaningful tissue targets. The final
binding therefore uses the paper's existing human-quality criterion, rather
than selecting against cleaned error or inventing physical truth. All 454
accepted moduli range from 14.692 to 285.114 Pa. The final selection and original
indices are frozen in the private afm_selection.json. Five negative controls
are the first source-key-sorted acquisition at each score 0–4, selected by that
rule rather than to maximize a desired visual effect.

Nearest old tasks: afm_molecular_surface_height infers topographic height from
tip-convolved images; phantom_complex_shear_modulus infers complex shear from
harmonic waves; neither uses quasi-static spherical force indentation to
recover local Young modulus and contact. Acquisition and unknown are distinct.

## Source revision, licensing and archive

Author repository: https://github.com/AFM-analysis/nanite.
Remotely verified master revision:
`44fbc2454bf047e828a7d13c3323ea4bd6fb9426`.
The GitHub commits API response is archived in nanite_commit.json; a failed git
clone was replaced with the exact-revision API source archive. This is a current
library implementation of the selected paper formula, not a claim that the
2019 software environment was recreated. The native defaults now allow fitting
baseline; the oracle explicitly fixes it to zero to match the selected paper
configuration and historical metadata. No source numerical functions were
modified for the oracle. Core cleaned formula/preprocessing adaptations preserve
GPL-3.0; the upstream LICENSE is included and authors acknowledged in source.

Maintainer archive index, outside the task under
`research_batch10_20261027/material_mechanics/`:

| Artifact | Consumer / purpose |
|---|---|
| zef18.h5; figshare_metadata.json | Original data and acquisition/license evidence. |
| nanite.zip; nanite_source/AFM-analysis-nanite-44fbc24/; nanite_commit.json | Verified upstream source and revision evidence. |
| oracle-env/ | Isolated upstream-only environment; not a cleaned dependency. |
| inspect_afm.py; probe.jpk-force | Acquisition metadata/model exploratory probe. |
| capture_afm.py; afm_capture.log | Original all-curve capture command and log; captures original outputs before extraction. |
| afm_oracle/*.jpk-force | Extracted original acquisitions used by native upstream. |
| afm_oracle/raw.npz; oracle.npz; metadata.json; capture_report.json | Complete calibrated archive, independent stages/final outputs and recorded runtime. |
| afm_oracle/initial_all1132_cleaned.npz | Exploratory cleaned all-curve result; not final task reference. |
| afm_parity_policy.md; afm_selection.json; afm_groups.json | Frozen parity criteria, author-QC selection and grouping. |
| rerun_afm_selected.py; afm_oracle/selected_rerun.npz; selected_rerun.json | Independent native author rerun directly on final public NPZ. |
| afm_setup.py; select_afm.py; build_afm_notebook.py | Private conversion/packaging/notebook authoring scripts, never runtime imports. |
| jupyter/; afm_figures/ | Private kernel registration and visual-inspection exports; outside task. |

## Original execution and frozen parity

Windows, Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1, lmfit 1.3.4,
nanite built from the pinned source as version 4.2.3, afmformats 0.18.7,
h5py 3.16.0. CPU-only, OMP/MKL/OPENBLAS thread counts set to 1.
The isolated environment also holds nanite's unrelated rating dependencies;
they are not installed or imported in the cleaned runtime. Numerical fits are
deterministic, with no stochastic seed. The general MINPACK backend is preserved.

Commands, relative to the private archive:
`oracle-env/Scripts/python.exe capture_afm.py` (original files), followed by
`oracle-env/Scripts/python.exe rerun_afm_selected.py` (final public arrays).
Original full capture: 110.73272309999447 s including conversion/compression,
exit 0, 1,132 curves and 15,847,782 samples, no fit warnings/failures.
Final 454-curve native rerun: 3.368437300028745 s, exit 0, maximum differences
from the corresponding independent original-file fits: 0 Pa and 0 m.
An early attempt reused the same temporary filename and triggered reader-cache
failure; unique source-key filenames fixed this. It was not counted as success.

Before local numerical extraction, parity tolerances were frozen in
afm_parity_policy.md: preprocessed x atol 1e-18 m, force atol 1e-22 N;
model/residual rtol 2e-12 with atol 1e-22 N; final E rtol 2e-4, atol 2e-4 Pa;
contact atol 2e-9 m. These were not relaxed after observing cleaned results.
Seven independent complete stage cases retain original archive indices
0,1,12,100,500,900,1100. Some are preassigned poor curves; they remain valuable
numerical unit/parity cases but are not final task targets. Full-pipeline parity
uses only the final independently selected 454-curve binding.

| Paper/stage | Original file/function | Local function | Validation |
|---|---|---|---|
| Cantilever compliance z=h+f/k | preproc.preproc_compute_tip_position | preprocessing.preprocess | frozen preprocessing cases; analytic tip identity |
| Noncontact force offset / initial contact | poc.compute_poc, poc_deviation_from_baseline; preproc_correct_force_offset/tip_offset | preprocessing.contact_index/preprocess | seven frozen cases; analytic crossing |
| Spherical formula (paper Eq.3) | model_sneddon_spherical_approximation.hertz_sneddon_spherical_approx | physics_model.contact_force | original forward values; independent exact implicit Sneddon check |
| Contact weighting | model.residuals.compute_contact_point_weights/residual | physics_model.weights/residual | original weighted residuals; analytic weights |
| Joint constrained E/contact fit | fit.IndentationFitter._fit; lmfit MINPACK | solvers.fit_curve | seven fitted cases and full 454-curve parity |
| Full reconstruction | Indentation.apply_preprocessing/fit_model | solvers.reconstruct; main.main | native final-NPZ oracle, CLI, notebook |
| Quality-dependent comparisons | paper manual rating workflow | notebook author-label groups and fresh ablations | 454 real retained curves; historical 1,132 labeled separately |

Independent physics passed: noncontact zero force, modulus linearity, positive
monotonic force for the tested acquisition-scale indentation interval, exact
spherical law relative error <=1e-4 for d/R<=0.5. No adjoint is required for
this two-parameter nonlinear contact model. MINPACK uses finite differences
of the explicitly implemented residual. No opaque domain inverse is retained.

## Reference arrays, metrics and thresholds

Hidden evaluation/baseline_reference.npz contains reconstruction (1,454) in Pa
and contact_position (1,454) in m from fresh author-code execution. It is not
ground_truth.npz. Reference-output consumers:

| File | Arrays and provenance | Consumer |
|---|---|---|
| upstream_measurements.npz | Fresh upstream E/contact and existing manual quality for 454; historical author modulus/quality for all 1,132 | Notebook parity and labeled historical QC context |
| acquisition_groups.npz | Integer ROI and instrument-session grouping from source file provenance, 454 curves | Notebook separate-session stage plot and ROI distributions |
| rejected_examples.npz | Complete measured traces/calibration of five preassigned score 0–4 negative controls | Notebook actively fitted diagnostic examples only |
| test/fixtures/upstream_stages.npz | Original complete inputs and fresh original preprocessing/model/residual/final values for seven fixed indices | Stage parity tests |

Final hashes (SHA256):
- raw_data.npz: `259fca843b5d693de80bc62accb336aebf338f6910794694b0a1d76083e3f3cc`
- baseline_reference.npz: `8977b498263cede269849e4c4491a01522f701f05eb63c26466565d32ec2c296`
- upstream_stages.npz: `c7967faf9f6060fc5234029e2bca81e7450a8acb4d148c592c1fe12424e4cf13`

NCC is the flattened cosine without centering. NRMSE is RMS error divided by
the modulus reference range, without alignment. The additional maxima evaluate
every modulus and absolute contact position. Self-reference NCC=1/NRMSE=0 is not
used to set a zero tolerance. Independent final-NPZ/native rerun is identical;
cleaned/backend probes differ by at most 9.58e-9 Pa and below 5e-16 m. Thresholds
are conservative numerical reproducibility bounds: NCC>=0.999999,
NRMSE<=1e-5, E maximum error<=0.01 Pa, contact maximum error<=5e-9 m.
These allow float32 submissions and modest solver rounding, while staying far
below experimental geometric uncertainty (0.34/37.28 relative bead-diameter
uncertainty) and rejecting physically different solutions. They do not claim
independent accuracy. No thresholds were relaxed to make a poor baseline pass.

Fresh shared-runtime result: NCC 0.9999999999999999,
NRMSE 2.149407760111837e-12, maximum E error 9.575643389325705e-9 Pa,
maximum contact error 9.286191607338345e-17 m. Judge passes.
All-zero modulus is a valid but failing-quality submission; missing file/key,
bad shape, integer/complex dtype, NaN/Inf and missing evaluator assets produce
error/empty metrics. Report-write failure returns exit 2. Independent known
vectors check Judge and visualization metric definitions.

## Execution, notebook and review

Cleaned shared environment: Python 3.11.11, NumPy 2.4.6, SciPy 1.17.1,
Matplotlib 3.11.1. Requirements list only these runtime packages. No domain
library, h5py, sklearn, lmfit or oracle source is required by cleaned runtime.
`python -m src.main` passes; `python evaluation/judge.py --prediction
output/reconstruction.npz --report output/judge_report.json` returns 0.
`python -m pytest test/ -v`: **25 passed in 3.43 s**.
An initial test run had nine temporary-fixture setup errors because an existing
system pytest temp root was inaccessible; an owned test-only temporary fixture
resolved this without changing scientific code or production paths.

The complete notebook executes through native nbconvert with a 600 s timeout,
chosen conservatively for three full fits plus plotting. A private kernel spec
selects the shared runtime. Initial `python -m jupyter nbconvert` dispatched a
different system Anaconda executable/kernel and failed on its old Matplotlib
boxplot API. The successful command uses `python -m nbconvert --to notebook
--execute notebooks/zebrafish_indentation_modulus.ipynb --output
zebrafish_indentation_modulus_executed.ipynb --output-dir output
--ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=research-clean`
with JUPYTER_PATH pointing to the private archive kernel spec. No environment
packages were upgraded. Windows Jupyter emitted event-loop and local TCP kernel
warnings; no scientific cells failed in the validated run.

Notebook actively reconstructs all 454 curves and both complete ablations.
Observed standard fit runtime: 2.159 s; two ablations: 6.790 s. Median actual
unweighted force RMSE in nN: spherical weighted 0.04168941307105845,
parabolic weighted 0.04054641162931799, spherical unweighted
0.03740714764471818. Lower unweighted residual does not imply superior physical
elasticity inference when the selected method downweights contact artifacts.
Seven computed figures cover raw traces/calibration, contact physics, parity,
force residuals, matched-quality ablations and historical QC, rejected examples,
and real single-session geometry/ROI distributions. Every figure was opened and
visually inspected; long repeated x-axis labels were shortened before final
execution. No invented maps, synthetic primary data, fake methods or reference
arrays posed as fresh results are present. Canonical executed outputs are
retained in the notebook; the duplicate output notebook is removed at handoff.

Source/plan semantic review checked raw-force compliance before baseline
subtraction, decreasing tip-coordinate sign, absolute versus relative contact,
all polynomial factors, force units, variable-dependent weights, original
lower-bound transform, MINPACK settings and approach-only fitting. Function
signatures/defaults and notebook ablation caller flow match plan/design.md.
Public README formulas use fenced math; all symbols, packed-array conventions,
known acquisition parameters and output quantities are defined there.

## Gate ledger

| Mandatory gate | Status / evidence |
|---|---|
| Paper, revision, real-data hashes, stage mapping | Passed, sections above and private API snapshots |
| Original and local exact-observation execution | Passed, native selected-NPZ rerun and full pipeline |
| Independent fixtures and physical law checks | Passed, stage tests and exact Sneddon comparison |
| Constants, units, coordinate/precision conventions | Passed, preservation and analytic tests |
| Standard CLI and fresh Judge | Passed, output report |
| Judge malformed/negative/evaluator/write cases | Passed, unit tests |
| Self-contained public inputs without hidden answers | Passed, README/raw/environment review |
| No public GitHub links or target-derived calibration | Passed, solver-visible audit |
| Metadata and complete output contract | Passed, README tables; no separate metadata JSON |
| Minimal runtime and explicit local mathematics | Passed, NumPy/SciPy/Matplotlib-only local imports |
| Unit/parity/physics/pipeline tests | Passed, 25 tests |
| Active scientific notebook with six story sections | Passed, native execution and seven inspected figures |
| Historical/reproduced/fresh labels and metrics | Passed, notebook/provenance separation |
| Plans/README/source consistency | Passed, semantic review above |
| Inventory and licensing | Passed locally; canonical notebook only, GPL and CC0 attribution retained |

Limitations: apparent elasticity model, subjective original quality selection,
no independent measured modulus truth, and no spatial registration across
sessions. These limitations remain explicit rather than being interpreted as
evidence for a stronger physical claim.

Parent-review follow-up: notebook display math uses  with existing computed outputs retained. Judge rejects negative moduli and catches overflow from extremely large finite submissions as evaluator errors with empty metrics; new contract cases passed.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 7 executed code cells, 7 computed PNG outputs, 11 formulas. Private evidence: research_batch10_20261027/acceptance/zebrafish_indentation_modulus/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.


## Current benchmark contract v2 (2026-10-29)

The v2 evaluator independently profiles one additive force baseline for each approach curve under the public corrected spherical contact law. Its force-range-normalized weighted residual MSE may exceed the attainable fit by at most 1e-6 for each of all 454 curves. The baseline is an explicit measurement nuisance, not an output rescaling; modulus must be nonnegative and contact position must lie in the approach-tip interval. An analytic baseline-shifted contact control and independently improved linear modulus/baseline fit pass. No independent modulus truth is claimed. The cleaned reconstruction algorithm is unchanged.

Earlier method-identity metrics are historical v1 reports, superseded for current benchmark acceptance. Full command stdout, native exits, controls and immutable old assets are indexed in the external benchmark_readiness_20261029 maintainer archive.
