# Provenance and validation

## Fixed physical binding

This task reconstructs the complete spatial slip distribution of the author's
North Anatolian fault velocity tutorial: 1292 triangular shallow patches,
five deep rectangles constrained to a shared rate, and two remote rectangles.
It also evaluates the fixed basin-shape coefficient and five velocity datums.
The inputs are existing real, author-curated satellite LOS and horizontal GNSS
velocity products, not raw radar images or receiver waveforms. There is no
independent subsurface truth. `evaluation/baseline_reference.npz` is explicitly
the reproduced final conditional tutorial estimate.

The selected method is the author's public homogeneous-half-space tutorial,
not the article's final layered-medium Bayesian posterior. All three actual
tutorial inverse comparisons are retained: zero-prior GLS, shifted-prior NNLS,
and shifted-prior NNLS with tied deep slip. The source notebook cells executed
are 1,3,5,7,10,13,15,16,18,21,23,27,28,29,32,33,37. Plotting and the decorative
geometry replacement in cell25 are omitted; this replacement does not rebuild
the already assembled Green matrix. No observation, patch, estimated quantity,
or inverse comparison is removed.

## Primary sources and rights

Maintainer archive root (outside this task):
`D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch7_20261022/groups/astro_earth_mech/csi`.
Hereafter private paths are relative to that directory.

* CSI: https://github.com/jolivetr/csi, exact revision
  `074ca028697b488fdb036e67d799787b374ddbb9`, GPL-3.0. Remote HTML `currentOid`
  was verified at 2026-10-21T21:04:59Z; the group `remote_verification.json`
  retains the observation. API403 failures are not treated as remote verification.
  `source.zip` has 221380423 bytes and SHA256
  `4c300844921e190d7c72957ca778e30926967ec71624a07de5eded46a9a6730e`.
* Exact author inputs are `DataAndModels/tutoInversionCreep` in that revision;
  the source tutorial is `notebooks/tutoInversionCreep.ipynb`.
  `native_data_hashes.json` records every bound native input hash and
  `native_notebook_text.txt` records the inspected notebook text. There is no
  separate verified observation-data license. GPL source licensing and article
  licensing are not asserted to license these data. This limitation is retained
  in NOTICE and the private report catalog.
* Jolivet et al., **Daily to Centennial Behavior of Aseismic Slip Along the
  Central Section of the North Anatolian Fault**, JGR Solid Earth, published
  7 July2023, DOI https://doi.org/10.1029/2022JB026018. The author-uploaded
  published 17-page PDF was read in full through the PDF web parser, including
  sections, captions and references. Archived parsed responses are
  `paper_author_uploaded_pdf_tooltext000.json`, `275.json`, `440.json`,
  `600.json`, `755.json` (the latter filenames share the full prefix).
  Primary author-uploaded PDF URL:
  https://www.researchgate.net/profile/Romain-Jolivet/publication/372201076_Daily_to_centennial_behavior_of_aseismic_slip_along_the_central_section_of_the_North_Anatolian_Fault/links/64afaf98c41fb852dd6c9fc8/Daily-to-Centennial-Behavior-of-Aseismic-Slip-Along-the-Central-Section-of-the-North-Anatolian-Fault.pdf
  Direct-download TLS/HTTP failures are retained separately. No successful
  binary download or PDF byte hash is invented: the archive indexes the actual
  parsed primary PDF responses. The article states CC BY-NC; this does not
  establish rights to independently curated tutorial observations.
* Meade2007, Algorithms for the calculation of exact displacements, strains,
  and stresses for triangular dislocation elements in a uniform elastic half
  space, DOI https://doi.org/10.1016/j.cageo.2006.12.003. The complete main
  paper and task-relevant displacement/loop construction were read; the unused
  74-page strain expansion is not implemented. Primary PDF with appendices:
  https://www.ceri.memphis.edu/~smalley/ESCI7355/meadetriangulardislocations2007_apndx_a_b.pdf
  `meade2007_with_appendices.pdf`, 1207493 bytes, SHA256
  `75047b789c8dbfe9ae3fc8b9bcddb425c77b114e70c746da0b9b568f64c5f53c`.
  Its public download required one explicitly scoped certificate-verification
  exception after verified-TLS failures; `meade_download.json` records that
  transport limitation. Author/DOI metadata were independently cross-checked.
* Independent rectangle oracle: https://github.com/jolivetr/okada4py,
  revision `cc1381a40246bbdb3efff13400da0f3f7a52dc8a`, GPL-3.0;
  `okada4py_source.zip`, SHA256
  `8916fe6114159254a4661154ac2ec08f280a90ce69e5f5c231d2622594940964`.
  Remote `currentOid` evidence is `okada4py_primary.html` and its download JSON.
  This is private oracle-only C/Fortran, never imported by cleaned code.

The shipped LICENSE and NOTICE preserve CSI/triangular-dislocation attribution
and clarify data rights. No supplied PDF, upstream checkout, download archive,
or capture script is inside the released task directory.

## Original execution before extraction

`run_original.py` invokes the isolated author implementation on the exact native
observations. `original_process.json` records the complete argv, seed policy,
source revision, executed cells, actual exit and duration. Original native exit0,
192.65879190000123 seconds. Stage arrays are `original_stages/cell23.npz`,
`cell29.npz`, `cell33.npz`; stdout/stderr are retained. The method is deterministic
and has no hidden stochastic initializer.

Oracle environment: WSL Ubuntu, Python3.12.3, NumPy2.2.6, SciPy1.15.3,
Matplotlib Agg; OMP/OPENBLAS/MKL/NUMEXPR threads1. `oracle-wsl-env` is private.
Original CSI and original okada4py source are unchanged. Only notebook execution
and numeric capture are external scripts. `oracle_install_process.json` and
`okada_install_process.json` retain the actual isolated installation/build logs;
the shared cleaned environment was not modified.

The first independent capture attempt failed on a capture-only nonexistent
`.geometry` property: exit1/66.8152seconds. Its logs and script are preserved.
The corrected capture accesses the actual `.ellipshape` attribute. It completed
with exit0/162.6349508seconds, recorded in
`independent_repeat2_process.json`. All1305 parameters in all three inverse
groups are bitwise equal to the first original run. This capture correction
does not change source mathematics or observations.

Source-only `capture_numeric_precision.py` independently reran original
`equalizeParams` and `ConstrainedLeastSquareSoln` on the captured full system.
Native repeat exit0/43.412427953s, max difference0; serializing the system to
float32 and restoring float64 before the same original solve completed
exit0/42.805077170s. The latter is a numerical precision diagnostic, not another
physical measurement or altered benchmark binding. Full results and environment
are `upstream_precision_measurements.json/.npz`.

## Independent standards frozen before local mathematics

`freeze_criteria.py` produced `frozen_criteria.json` and
`frozen_method_reference.npz` before local acquisition/inverse extraction.
The independent serialization main-field range-NRMSE is
5.8648899380666354e-9; its preselected factor1.5 gives
8.797334907099953e-9. Cosine boundary is 1−96 machine epsilons,
0.9999999999999787. Every scalar/constant or remote quantity has its own
nonzero absolute tolerance from the independent precision measurement; these
are listed without rounding in `evaluation/metrics.json` and the frozen JSON.
No pooled dynamic range conceals small components. All1299 slips and all six
nuisance coefficients must pass. The forward weighted RMS reference is
1.1809396435215258; its independently fixed factor1.05 boundary is
1.2399866256976022.

Frozen bad controls include zero main/slips (main NRMSE0.3283038392,
weighted RMS2.9926571068), uniform main slip (0.2592596828,1.2775387998),
reversed main slip (0.3322,1.2591), wrong sign (0.6566,1.6797), and the incorrect
1000-fold basin-unit conversion (weighted RMS169.47). Earlier GLS and untied
NNLS also fail the selected conditional target. Each declared output group is
independently perturbed and rejected by contract tests. Reference agreement is
explicitly numerical/method reproducibility, not proof of geological accuracy.

Stage thresholds were also frozen: Green matrix atol2e-11/rtol1e-9;
covariances atol1e-11/rtol2e-12. Noncanonical original comparison-group parameter
atol is 4.500325418987927e-7, from1.5 times measured full-vector precision error.
There has been no threshold relaxation after cleaned results.

## Paper/code discrepancies and physical checks

1. Article Section4.2 uses triangular nodal interpolation, layered elasticity,
   additional nuisance motion, and 90000 AlTar posterior samples. The selected
   public tutorial instead uses constant-slip triangles, homogeneous elasticity,
   a fixed covariance prior and GLS/NNLS. Neither the paper posterior nor its
   21.5mm/year headline result is claimed reproduced. The tutorial shared deep
   rate is26.05956114460914mm/year.
2. The source `buildCmXY` docstring has different exponents/amplitudes from its
   executable product. The selected executable covariance, including the
   half-exponents and variances (not standard deviations), is retained and
   explicitly defined in README, approach and notebook.
3. The original basin Green function is m/Pa and is multiplied by−1e6 when
   fitting mm/year velocities. Dimensional closure is
   1000mm/m ×1000Pa/kPa=1e6, so its formal coefficient is kPa/year with a
   deflation sign. The source MPa print label is wrong by1000. We correct the
   label/definition only, never the matrix or coefficient. The radius6km
   exceeds depth3km; this is a fixed empirical subsidence shape, not validated
   physical cavity pressure. Independent on-axis magnitude, radial symmetry,
   sign and the precise unit conversion are tested.
4. Both NNLS groups shift the five datum priors to−100; GLS uses zero prior.
   Thus differences are not attributed solely to positivity. The final deep
   prior variance is100 for the one tied variable, not a sum of five variances.
5. Rectangles are locally evaluated as the exact sum of two triangular loops.
   Before extraction, original Okada and original Meade operators were compared
   at all3515 receiver positions for all seven rectangles: maximum absolute
   discrepancy7.883019324111951e-13/unit slip, maximum relative L2
   1.135884e-12. `source_rectangle_equivalence.json/.npz` and
   `check_rectangle_superposition.py` retain the independent case. This is exact
   common-edge cancellation, not a different meshing approximation.
6. All receivers are on the free surface with nonnegative source depths, so the
   original interior-prism correction is inactive. The local operator clearly
   implements this surface-only scope. Vertex reversal, translation invariance
   and slip linearity are independently tested. Source rounding/branch
   conventions, including `RotateXyVec` unit-spacing additions, are retained.

## Equation/function/test traceability

| Required mathematics or stage | Original file/function | Local function | Independent/parity consumer |
|---|---|---|---|
| Observations, component-first ordering, native geometry | tutorial and data readers | preprocessing.load_observations, observation_vector | comparing_test/test_operators.py |
| Half-space angular dislocation and closed triangle loop | csi/triangularDisp.py displacement, adv, RotateXyVec; Meade loop construction | physics_model.displacement, adv, RotateXyVec | full original Green matrix; symmetry/linearity/translation tests |
| Finite rectangular dislocations | csi/okadafull.py / okada4py | physics_model.build_green_matrix, sum of two exact triangles | source_rectangle_equivalence; full Green parity |
| LOS projection, GNSS selection, track/translation indicators | source buildGFs, assembleGFs | physics_model.build_green_matrix | full original G and d parity |
| Fixed basin response and unit scaling | csi/mogifull.py and tutorial −1e6 column | physics_model.mogi_deflation_basis | known on-axis dimensional and symmetry tests |
| Correlated radar and diagonal GNSS covariance | source buildCd | physics_model.build_data_covariance | full original Cd parity |
| Anisotropic spatial prior and scalar variances | source buildCmXY and tutorial | physics_model.build_model_covariance | full original Cm parity |
| Quadratic objective and gradient | source solve definitions | solvers.objective_gradient | independent central finite differences |
| GLS and shifted whitened NNLS | multisourcesolve.computeCmPost/ConstrainedLeastSquareSoln | solvers.solve_system | all three original full-parameter vectors, KKT diagnostics |
| Tied deep variable and variance100 | multisourcesolve.equalizeParams | solvers.tie_deep_parameters | independent known mapping and full final parity |
| All outputs and standalone scoring | frozen source measurements, specified metric definitions | solvers.output_arrays; independent evaluation/judge.py | known metrics, malformed assets, all bad controls, CLI0/1/2 |
| New correlated-noise diagnostic only | no historical source result claimed | generate_data.correlated_velocity_perturbation | known seed/units/covariance scaling and unchanged input tests |
| Plotting and metric presentation | fresh computed groups and separately identified source groups | visualization helpers | known cosine/range, neutral-zero color, complete vector extents |

## Numeric asset consumers

* `test/fixtures/original_system.npz`: original main model covariance used by
  full covariance and pipeline parity. Original G,Cd,d are stored once under
  reserved hidden reference keys, avoiding redundant giant array copies.
* `test/fixtures/prefrozen_criteria.json`: provenance assertions for the
  before-local tolerance selection.
* `notebooks/reference_outputs/original_comparisons.npz`: all three separately
  named original full1305-vectors; consumed by active notebook original/fresh
  comparisons and pipeline parity tests. They never supply a fresh result.
* `evaluation/baseline_reference.npz`: declared output reference plus original
  G,Cd,d calibration used by the independent forward criterion and parity.
  This hidden calibration is not solver-visible or derived from hidden slip.
* Private `figures/numeric_sources.npz` and `experiment_metrics.json`: full
  computed observations, operators, all method fields, residuals and diagnostic
  results underlying the six paired PNG/SVG/PDF groups. `plot_provenance.json`
  ties those bytes to canonical notebook and figure consumers.

## Clean runtime and execution evidence

The cleaned runtime uses only standard library, NumPy2.4.6, SciPy1.17.1 and
Matplotlib3.11.1 on Windows/Python3.11.11. Numerical physics, covariance,
whitening, mapping and inverse assembly are explicit local code. Generic
linear algebra and NNLS are SciPy primitives. There are no CSI, okada4py,
projection/domain-library or cross-task imports in source, tests, Judge,
generator or notebook. Shared requirements were not installed/upgraded.

First complete fresh local run (`run_first_local.py`) exit0/37.334721s;
main-field NRMSE1.884461669703111e-13, cosine1.0, weighted observation
RMS1.1809396435215331. Every separately measured nuisance/deep quantity passes
its frozen boundary. Objective25539.736070798703 and projected-gradient
infinity norm4.6071590986684896e-11. Full environment, optimization timing and
all12 metrics are in `first_local_result.json` and its process/log files.

The standard command `python -m src.main` completed exit0/35.6361202s;
`python evaluation/judge.py --prediction output/reconstruction.npz --report
output/judge_report.json` completed exit0/0.9566555s. Initial complete pytest
run had24passed, exit0/101.889199s. `core_validation_process.json` records
exact argv, working directories, return codes and runtimes; corresponding
stdout/stderr are retained. Final tests add independently known plotting
metrics, backend preservation and complete vector-tip extents; final actual
process evidence is indexed in the final-execution section below.

Judge loads allow_pickle=False, validates all required float64 shapes and
finite values, handles missing/corrupt files and references, and never imports
solver code. Actual truth, when supplied, takes priority; malformed actual
truth is an evaluator error with no fallback. Known-case regression verifies
this policy independently. Foreign-working-directory CLI scoring is also
captured. Reports contain finite JSON values and explicit exit0/1/2 behavior;
report-write failure returns2.

## Active notebook and scientific review

The canonical notebook has six narrative chapters: inverse problem, complete
data, explicit physics, selected conditional mathematics, three fresh method
comparisons/results, and sensitivity/limitations. It actively builds the full
operator, solves all three original groups, evaluates every quantity and the
forward criterion, and runs diagonal-noise, half/double-prior and a seeded
0.15-amplitude correlated-noise experiment. New observations are made only
through the documented/tested generator. There is no cache-preview route.

`execute_notebook.py` uses research Python and subprocess.run(shell=False) for
actual `python -m nbconvert --execute`, timeout1800, private kernel
`imaging_csi` pinned to the same executable. It records separate stdout/stderr
and actual returncode/runtime, then copies the exact successful executed
artifact to the canonical notebook without output substitution. The notebook
records actual sys.executable, package versions, backend, numerical
configuration and thread variables in its output and private metric JSON.

First execution was exit0/157.2169504s; second exit0/161.0046959s.
Both full attempts and figures are privately retained in notebook_attempt1/2.
Six actual figure groups were opened and inspected. Subsequent changes are
scientific presentation only: explicit color saturation windows/extend marks,
complete1292-patch geometry display, matched method colors and full GNSS vector
tails/tips with clear keys. Final complete execution is recorded below; no
cached computation substitutes for that execution.

The six figure groups show all data/geometry/noise, original/local unit-response
identity, all three full spatial slip comparisons, all LOS/GNSS fits and signed
residuals, paired numerical/metric and nuisance comparisons, and all fresh
sensitivity fields. Signed colors have symmetric zero-neutral normalization;
clipped display scales are explicitly labeled and never affect metrics or
saved numeric arrays. A single perturbed reconstruction is sensitivity, not a
posterior sample or independent geological truth.

README, approach and design were checked against implemented signatures,
complete constants, signs, units, coordinate/component ordering, covariance
normalization, priors, equality mapping and actual caller flow. Public raw data
contains legitimate known geometry/LOS/calibration and velocities only. No
truth, fitted target, hidden seed or GitHub URL is exposed in the public
allowlist. Plans describe actual shipped code rather than a proposed pipeline.

## Final execution and completion evidence

The maintainer finalizer appends actual final process results and hashes here
after successful execution. The independent original, frozen numerical
standards and first fresh measurements above remain separately identified.

Final complete notebook: actual exit0/160.599820300005s,
canonical SHA256 `b12aa597c33408f6c550bf21a104856a9908ccd8d16ba1e778335c303395d865`. Final complete pytest:27passed,
exit0/109.524394499997s. Foreign-working-directory Judge:
exit0/0.956781200002s. Exact commands and separate logs are
`final_checks_process.json`, `pytest_final_stdout.log`,
`pytest_final_stderr.log`, `judge_foreign_stdout.log`,
`judge_foreign_stderr.log`, and `notebook_process.json`/stdout/stderr.
All code cells executed without an error output; all six final PNGs were
actually opened and reviewed. `static_audit.json` reports errors[].

Final mandatory gates: PASS source/paper/data binding; PASS original exact-data
run; PASS independent reruns and pre-local tolerances; PASS operator/covariance
parity and independent physics/gradient tests; PASS all three complete inverse
comparisons; PASS standard CLI and fresh standalone Judge; PASS malformed and
poor-output/actual-truth priority contracts; PASS clean-runtime symbol/import
audit; PASS public allowlist sufficiency and no privileged/GitHub leakage;
PASS27tests; PASS full active notebook native0 and all scientific figure review;
PASS plan/source semantics and signatures; PASS template inventory and license
notices. No mandatory gate is failed or unverified. Geological truth and
separate data redistribution permission are explicitly unavailable rather than
claimed validated.

Private `archive_index.json` indexes hashes of exact source/download archives,
primary paper bytes or actual parsed-PDF responses, native data hashes, capture
scripts, process/environment records, logs, frozen arrays and comparison
measurements. Private `completion.json` and `final_evidence.json` identify the
final task inventory. Task caches were reversibly archived outside the task;
no temporary executed notebook is retained under output.

Parent review amendment: standalone Judge now rejects boolean/string/nonfinite
boundaries, NCC boundaries outside[-1,1], and negative error boundaries as
evaluator errors with empty metrics. Twelve independent malformed-configuration
cases plus all original15Judge-contract cases passed: actual native exit0,
27passed/13.93s (tool session55502). See
`parent_config_review_amendment.json`. This changes malformed-config handling
only; solver, valid thresholds, canonical notebook and prediction bytes remain
unchanged. The final complete suite before this amendment had27passed; with
12additional cases the collected suite now has39cases, and the changed Judge
subset was executed in full. No unsupported claim of a new whole-suite run
is made.
