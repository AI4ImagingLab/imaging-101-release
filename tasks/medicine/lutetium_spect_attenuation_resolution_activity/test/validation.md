# Provenance and completed validation

This private task binds all 96 full 128×128 real photon views and the full
128³ native reconstruction lattice of the author's lutetium-177 NEMA phantom.
No output-dependent support selection, image downsampling, hidden intensity
calibration or invented truth is used. The native original, independently
repeated native original, fresh local method and historical scanner result
are separately identified. All scientific comparisons operate on actual data.

## Primary sources, exact revision, rights

Paper: Polson and colleagues, *PyTomography: A Python library for medical image
reconstruction*, SoftwareX, DOI10.1016/j.softx.2024.102020. The full 28-page
author manuscript arXiv2309.01977v6 (31 December2024 version, PDF dated
3 January2025) was read, including Appendix A.2 and Figure A.6, which describe
this exact phantom and OSEM4×8 with attenuation/PSF/TEW. The manuscript and
the modern source are distinguished: this is a validated modern port, not
an assertion that the source matches the paper-era version byte for byte.

Author source: https://github.com/PyTomography/PyTomography redirects to the
canonical qurit project. Live API repository and commit responses were captured
on2026-10-22. Exact verified revision:
`69a7a0b3e79302fc7789d100946331658d3e22c7`, modern source version3.4.0.
MIT license is present and retained verbatim in NOTICE. Generic homogeneous
rotation conventions were extracted from Kornia0.8.2; Apache2.0 text and
attribution are also preserved. Neither package is a cleaned-runtime dependency.

Data: Lucas Polson, *PyTomography SPECT Tutorial data*, Zenodo15314460,
https://doi.org/10.5281/zenodo.15314460, published1 May2025, explicitly
CC-BY-4.0 in the live record. This is a **real physical phantom**, not a newly
generated object, author simulation or patient scan. Nominal hot-sphere to
background filling ratio10:1 is context only and supplies no voxel truth.
The independent CT attenuation is legitimate instrument/acquisition calibration.
The historical Siemens reconstruction is maintainer comparison only and never
appears in public input.

All upstream archives, DICOMs, capture scripts, original logs, paper and figure
exports live outside this package in maintainer archive
`research_batch8_20261022/groups/bio_med/spect/`, abbreviated S below.
`S/acceptance_evidence_index.json` records complete SHA256 and size for the
paper, source ZIP, API metadata, observations/reference/fixture files and all
critical execution evidence. `S/data_member_provenance.json` records every
original DICOM member's complete SHA256, byte size and ZIP CRC. There are
78 complete members: one NM projection DICOM, one historical scanner
reconstruction and76 CT slices. The publicly listed container SPECT.zip is
1,430,341,130 bytes, publisher MD5 da33319971ace40e5b68fef19bdb2d08.
Only the central directory and exact required complete members were retrieved
through ordinary HTTP206 requests. **The whole ZIP was not independently
hashed**; publisher checksum is not relabelled as an independent verification.
78 selected files total67,244,554 uncompressed bytes and10,337,557 transferred
member bytes; source observations are not spatially cropped. Capture script
S/fetch_native_members.py and ZIP inventory/tail are retained.

## Source execution before extraction

The unmodified original source ran in private `S/oracle_env` with Python3.11.11,
shared NumPy2.4.6/SciPy1.17.1/Torch2.9.1+cu128 exposed read-only and private
pydicom3.0.1, Kornia0.8.2, kornia-rs0.1.10, fft-conv-pytorch1.2.0,
rt-utils1.2.7, nibabel5.4.2, pandas2.3.3, torchrbf1 and uproot5.6.8.
These domain packages are absent from the cleaned runtime import graph. No
shared environment package was installed or upgraded for this verification.
OS is Windows build26200, GPU NVIDIA RTX4080SUPER16GiB, driver610.88,
CUDA runtime12.8. Torch/BLAS/OpenMP CPU thread count is one. The shared
cleaned interpreter is a separate environment without PyTomography/Kornia.

The passive capture wrapper `S/run_native.py` calls the unchanged author
reader, `SPECTAttenuationTransform`, `SPECTPSFTransform`, `SPECTSystemMatrix`,
`PoissonLogLikelihood`, and `OSEM(...)(n_iters=4,n_subsets=8)`. Its callback
only records each completed subset result. It reconstructs the exact complete
observations now bundled. First wrapper attempt used a nonexistent keyword
`energy_kev` when requesting scanner PSF metadata and failed before inverse
execution; `S/native_failed_keyword.log` preserves it. The wrapper was corrected
to the author's positional argument; no source mathematics was modified.

| Execution | Evidence and command | Native exit | Reconstruction time |
|---|---|---|---|
| Original CUDA | `python run_native.py`; S/native_run.json/.log | 0 | 7.13677749998169 s including reader/configuration |
| Independent original CUDA repeat | `python run_native_repeat.py`; S/native_repeat.json/.log | 0 | 7.13539 s |
| Full original CPU, one thread | `python run_native_cpu.py`; S/native_cpu.json/.log | 0 | 741.8369 s |
| First full cleaned CUDA | `python check_cleaned.py`; S/first_cleaned_run.json/.npz | 0 | 2.3187647000013385 s including stage probes |
| Full cleaned CPU, one thread | `python -m src.main --device cpu --output private/cleaned_full_cpu.npz`; S/cleaned_full_cpu_run.json/.log | 0 | 729.7772113999818 s |

Original first/repeat arrays are bitwise identical. Source and local default
CUDA FP, BP and complete32-update activity are bitwise identical. All activity
values are finite/nonnegative; maximum25.1272411346 native units. The count
arrays retain exact uint16 values; converting them to float32 for computation
is lossless. No target result enters calibration. The source CPU/GPU scatter
estimate differs at most3.8147e-6 counts due to float32 arithmetic.

CPU portability is numerically validated, not claimed bitwise. Fresh full CPU
versus original GPU has max error4.482269287109375e-5, maximum/range
1.783828659537779e-6, DR-NRMSE2.0057591327054133e-8 and cosine
NCC0.999999999998982. Versus independently original CPU: max
7.05718994140625e-5, max/range2.8085731924421447e-6,
DR-NRMSE1.6615912471438388e-8, NCC0.9999999999993112.
S/cleaned_cpu_comparison.json retains full values and environment.

## Frozen independent criteria and physical checks

`S/binding_frozen.json`, timestamp2026-10-22T18:22:13.563810, was written
**before the first local method implementation**. It binds the selected
unknowns, complete observations, source revision, dataset rights, output dtype,
independent fixture hashes and all numerical boundaries. No boundary has been
changed after examining local results. No actual activity truth exists, hence
evaluation/baseline_reference.npz is explicitly a standard-method result.
Ground truth, if actually present, takes precedence and invalid truth cannot
fall back to the method result.

Independent source CPU/GPU difference, not self-comparison alone, establishes
a nonzero numerical envelope: source CPU/GPU max7.2479248046875e-5,
DR-NRMSE2.3244323564369887e-8, cosine0.9999999999986487. Source GPU
nextafter float32 envelope has DR-NRMSE1.1611917177557498e-9. The fixed
NCC lower limit0.9999999, DR-NRMSE upper1e-6 and max/range upper5e-6
cover measured backend variation without permitting rescaled/constant outputs.
The max/range bound is about1.7 times the prelocal original backend maximum.
Numerical agreement never supplies a physical-accuracy claim.

The complete forward relative L2 upper boundary0.22 was fixed from original
full-data0.2192173698222792, independently reduced with NumPy float64. It
includes all96×128×128 bins and the fixed additive TEW term. The same original
model gives an optimised uniform-activity null0.7272666999 and a scatter-only
zero-activity null0.7343370328. Full original CPU forward error is
0.21921738701066465. Long CPU float32 Torch norm reduction has a small
accumulation bias, so no float32 scalar discrepancy is mislabeled as physics
failure: S/native_quality_float64.json explicitly uses common float64 norms.

Independent physics were performed before extraction, not generated using
cleaned functions:

- Beer–Lambert constant slab,11 depth planes, mu0.12/cm: expected survival
  exp[-mu*d*(10.5-i)], maximum source error8.27e-8. This tests half-voxel
  integration, sign, detector direction and centimetre units.
- Independent Gaussian discrete normalisation at every source depth/radius:
  maximum kernel difference1.35594e-7 from float64 analytic evaluation.
- Independent homogeneous inverse-coordinate matrices plus SciPy bilinear
  interpolation agree with original float64 rotation within2.298e-14 after
  retaining source float32 pi and normalisation constants. Using mathematical
  pi/full-precision normalisation instead yields2.18888e-6: an explicitly
  identified numerical convention, not an unexplained source error.
- Complete positive and signed probe FP/BP dot products diagnose inexact
  continuous backprojection. Positive norm-scaled gap4.0259e-6; signed gap
  1.20912e-5. A relative dot gap on a nearly zero signed product is0.039515;
  it is retained and not disguised by normalisation.
- Independent explicit bilinear matrices on fixed physical grids33/65/129/257
  have exact-transpose dot discrepancies0 to1e-15. The inverse-angle
  integrated gap decreases4.59138e-5 to1.11310e-10, but pointwise relative L2
  discrepancy remains7.1203% to6.8662%. This proves only weak-integrated
  consistency; no false exact discrete adjoint or strong convergence claim.
  This is the selected published numerical backprojector, not a silently
  corrected/new algorithm. Source fixes are not applied to the method.
- Independent first OSEM update equals original update0 exactly. Complete
  original full-data Poisson NLL after initial/each completed iteration:
  143174981.1, −12305061.4, −12332173.3, −12337809.7, −12340312.9.
  These are unnormalised, without count factorial, not a chi-square statistic.

The discrepancy is material: inverse-angle interpolation approximates the
continuous backprojector and does not make OSEM an exact discrete-gradient
method. Geometry/PSF/attenuation/TEW remain physically explicit and the actual
bound full-data quality materially improves the predeclared nulls.

## Equation/stage traceability

| Paper model / stage | Original source function | Local mathematics | Validation consumer |
|---|---|---|---|
| Poisson primary plus scatter, AppendixA.2 configuration | likelihoods/poisson_log_likelihood.py, SPECTSystemMatrix.forward | physics_model.Acquisition.forward; solvers.reconstruct | full_pipeline_all_32_states; standalone forward Judge |
| Author NM orientation, angle/radius parsing | io/SPECT/dicom.py get_metadata/get_projections | native reader captured once; preprocessing.load_data exact arrays | raw_shapes_and_constants; source full observation hashes |
| CT independent attenuation calibration | get_attenuation_map_from_CT_slices; scanner ct_table.csv | public registered mu; local probability/integration | constant slab; frozen full FP/BP |
| Centred padded bilinear rotations | utils/spatial.py; generic Kornia geometry | rotation_grid, rotate_volume | independent_scipy geometry; source stage parity |
| Centre-to-face absorption | attenuation.get_prob_of_detection_matrix | detection_probability | half_voxel_beer_lambert; zero_attenuation |
| Depth-dependent Gaussian CDR | psf._get_sigma, get_1D_PSF_layer_gaussian, Seperable1DBlurNet | gaussian_kernel, blur_volume | physical_width_and_normalization; symmetric blur adjoint; full stage parity |
| TEW lower/upper windows | dicom.get_energy_window_scatter_estimate; utils/scatter.py | preprocessing.scatter_estimate | TEW/first_update; independent full source observations |
| Inverse-angle continuous backprojection | SPECTSystemMatrix.backward | Acquisition.backward | norm_scaled weak dot checks; independent refinement fixture |
| OSEM sensitivity, preconditioner,32 updates | preconditioned_gradient_ascent.OSEM; PoissonLogLikelihood | solvers.reconstruct | first update and all32 frozen states |
| Native output scale / no voxel-volume factor | author DICOM tutorial t_dicomdata.ipynb | full-volume count-sum model | FP linearity; no fitted scaling comparisons |
| Unnormalised NLL / full L2 / agreement | independently captured source states, separate float64 quality reduction | visualization metrics; independent Judge metrics/forward | analytic cosine/RMS; native full-data controls; priority/error tests |

`plan/approach.md` and `plan/design.md` were derived from the actual extracted
source and checked against callable signatures, axes, centre convention, cm/mm,
Gaussian normalisation, TEW operation ordering, reciprocal multiplication,
projection signs, fixed updates, all output fields and no-backend-mutation
imports. The README defines every required physical symbol and all input/output
arrays. Public README/requirements/raw contain no source-site URL, hidden result,
target-derived calibration or source-only dependency.

## Fixture and reference consumers

| Packaged NPZ | Identity and provenance | Consumer |
|---|---|---|
| data/raw_data.npz | Three complete native windows losslessly stored uint16; independently registered CT mu; original angles/radii | main and active notebook; standalone Judge public forward calibration |
| test/fixtures/upstream_stages.npz | S/native_run/native_stages.npz, original positive input object/projections, full FP/BP, subset0 sensitivity/ratio/gradient/update; captured before local | comparing_test stage/first-update/weak-dot cases |
| test/fixtures/independent_adjoint_refinement.npz | S/independent_adjoint_refinement.py/.json explicit finite bilinear matrix proofs, four grids, no source or local fit | notebook actual limitation figure13 |
| evaluation/baseline_reference.npz | Independently original CUDA final activity, batch first | standalone Judge; explicitly not truth |
| notebooks/reference_outputs/upstream_measurements.npz | Identified original CUDA final activity/prediction/all32 states | stage/full-state tests and labelled upstream notebook comparisons |
| notebooks/reference_outputs/historical_vendor.npz | Complete actual Siemens DICOM, original reader transpose(2,1,0), divided by96 exactly as author tutorial | labelled historical vendor notebook figure6; never a reference truth or public calibration |
| notebooks/reference_outputs/cleaned_measurements.npz | Fresh full standard notebook result/prediction/32 states | maintainer numeric review; not loaded to replace active reconstruction |

No NPZ allows pickle, and no upstream import appears in runtime/tests/notebook.
The complete imported-symbol/call-role audit covers Torch sampling/convolution,
inverse/cumsum/exp as general numerical primitives; no ready-made domain solver,
forward projector or imaging reconstruction library is used. Source licenses
and attribution are retained; complete source/PDF/logs are outside the package.

## CLI, tests, notebook and actual science figures

`python -m src.main` actively completes and writes the declared file, native0.
Foreign-working-directory `python evaluation/judge.py --prediction <absolute
fresh output> --report <private report>` completes native0; S/foreign_judge_report.json:
NCC1.0000000000000002 (ordinary float64 roundoff), NRMSE0, max/range0,
full forward L2 0.21921736982219. All values/thresholds are finite JSON numbers.
Output report is copied to output/judge_report.json, not used as a substitute
for any independent test.

`python -m pytest test/ -q --basetemp <private temporary directory>`:
**29 passed, native0, 6.68 seconds**, S/pytest_initial.log/.xml and
S/run_tests.py. Tests cover analytic attenuation/rotation/PSF/metrics,
full-stage parity, every original subset state, full physical forward quality,
linearity, malformed/missing/wrong-shape/wrong-dtype/complex/object/NaN/Inf
NPZs, missing/invalid hidden assets, zero/constant poor outputs, truth priority,
invalid truth with no fallback, metric-boundary validation, nested CLI reports
and report-write failure. Generic Windows pytest temp permissions required an
explicit private --basetemp; no algorithm or evaluation criterion was altered.
An initial priority test mistakenly asserted an arbitrary NRMSE>.01 for this
sparse volume; it was replaced by the independently exact RMS/(2range)
expectation. This was a test correction, not a criterion or Judge change.

The prior pre-contract-fix canonical active six-chapter notebook was executed via pinned
`python -m nbconvert --to notebook --execute ...`, timeout1800 seconds:
**native0,58.21595849998994 seconds**, superseded SHA256
`f718e4a8e9903001adead7235426e916ed2274fed51a527a26ebf8a7e495c1f8`.
S/notebook_run.json/.log and S/spect_executed.ipynb record this execution;
the canonical packaged notebook is byte-identical. A prior complete visual
revision native0/60.2097 seconds/hash69797e... is retained privately as
superseded, not the current notebook. Final visual changes label real axes,
use the author's z61/y70/x64 display convention, and expand post-initialisation
convergence. They do not change the bound reconstruction or scientific arrays.

Thirteen final actual-result figure groups are in S/figures, each PNG/SVG/PDF
with paired numeric NPZ and hash JSON; figure_inventory.json connects each
artifact to its computed values. All13 PNGs were opened for actual visual review
including signed-zero colour convention, shared comparison scales, axes/units,
unfitted vendor comparison, residual panels, finite-iteration effects and
explicit inexact-adjoint limitation. Display slices do not change evaluation
support. The notebook fresh standard reconstruction remains bitwise equal to
original CUDA, with physical full L2 0.2192173698222733.

Actual full-data ablations (all freshly reconstructed from identical complete
observations, then all scored with the same complete calibrated forward model):

| Fresh inversion assumption | Full physical forward relative L2 | Unnormalised Poisson NLL |
|---|---:|---:|
| Complete attenuation+PSF+TEW | 0.2192173698222733 | −12340312.9301385 |
| No attenuation in inversion | 0.5217084645594788 | −11281900.749176303 |
| No PSF in inversion | 0.22079700948078063 | −12326266.598618574 |
| No TEW in inversion | 0.36079529208512534 | −12137234.60131333 |

Fresh eight-iteration and one-subset comparisons are actual diagnostic runs,
not replacements for the fixed standard. All four complete iteration states
are forward evaluated; no historical curve or zero-filled missing method is
plotted as fresh. S/figures/experiment_summary.json preserves full precision.
No new synthetic observation was generated; model-removal and iteration
comparisons reuse actual measurements, so no src/generate_data.py is required.

## Gate disposition

All mandatory gates are passed: primary paper/revision/data/rights and mappings;
exact-input original GPU/repeat/full CPU; independently frozen fixtures and
applicable physical checks with explicitly bounded approximate BP; local
explicit mathematics; full local CUDA/CPU execution; complete output/Judge/error
contract tests; self-contained public input and no leakage; plans consistent
with actual implementation; active native0 notebook and actual figure review;
minimal pinned dependencies, clean runtime without domain packages; documented
fixture/reference consumers and source licensing; final inventory audit.
Parent independent acceptance remains a separate gate and is not claimed here.
Physical limitations are retained rather than concealed: no voxel truth, no
absolute Bq calibration, fixed TEW/Gaussian response, approximate continuous
backprojection and finite-iteration method dependence.

Final mechanical verification: all70 mathematical expressions across README/plans/notebook pass both independent KaTeX and MathJax parsers (B8/qa/lutetium_spect_attenuation_resolution_activity_math_render_check.json). Private self-audit S/self_audit.json reports errors=[] including required structure, public-array tables/URL checks, NPZ finite checks, imported symbols/call roles, executed notebook outputs and standalone foreign-cwd Judge. Independent source operator probes use torch seed739 and rotation test NumPy seed1804; OSEM itself is deterministic and has no observation-generation seed.

## Final evaluator contract revision and current execution

Parent semantic review required three contract corrections, without changing any reconstruction mathematics or the four predeclared numerical boundaries. A shape/dtype-correct finite volume with any negative voxel is a valid quality failure through negative_count<=0, also declared in metrics.json. A one-voxel value -1e-20 is independently tested with the real complete forward evaluator. Negative evaluator reference is an evaluator error. NCC boundaries outside (0,1], negative error boundaries, booleans and any nonzero physical-negative-count boundary are rejected. Every measured metric must be finite before the report is written. A full volume of maximum finite float32 values causes forward arithmetic overflow and now returns descriptive error2 with empty metrics and valid finite JSON; no Inf/NaN is emitted.

The original29-test logs and executed notebook are preserved as S/before_judge_contract_fix_*; they are superseded evidence, not the final revision. The final complete test suite passes42 tests, native0 (S/pytest_initial.log/.xml; final runtime recorded there). Added regressions exercise tiny negative activity, invalid metric domains, negative hidden references, extreme finite inputs and nonfinite derived metrics.

The **current final canonical notebook** actually reconstructs again and actively invokes the standalone revised Judge CLI: native0, 60.31782709999243 seconds, SHA256 `f67626ee6c2446d26fda8ad5d9c3a21242df259e13786abb52611d203af87652`. It is byte-identical to S/spect_executed.ipynb. S/notebook_run.json/.log identify this current execution. All13 embedded figure PNGs are byte-identical to the previously fully opened final visual revision; S/judge_fix_visual_identity.json verifies every actual PNG rather than assuming plots were unchanged. The output report includes negative_count=0 and all original numerical criteria pass. The solver, calibration, source reference and frozen tolerances are unchanged.


## Parent independent private acceptance

Parent independently reran all42 tests: native0/9.157523000s; all70 formulas passed both mathematical renderers. Parent reviewed source operators, inverse updates, independent Judge, both plans, provenance and the active six-part notebook; all13 final scientific PNG groups were actually opened individually. Canonical notebook and archived native0/60.3178271s execution are byte-identical, SHA256 f67626ee6c2446d26fda8ad5d9c3a21242df259e13786abb52611d203af87652. All final READY file hashes and figure export hashes were independently verified. All96 views and128-cubed voxels remain; CPU full reconstruction evidence is retained. The inverse-angle backprojection remains explicitly approximate, with independent exact-transpose and refinement comparisons; no exact-adjoint or absolute Bq calibration claim is made. The hidden reference is a reproduced finite-iteration method result, not actual voxel truth. Tiny-negative predictions, invalid boundary domains and finite-input forward overflow are covered by final regression tests. This is private staging acceptance, pending batch completion and release.


## Current benchmark contract v2 (2026-10-29)

Current acceptance uses nonnegative activity, all 96 photopeak views, the independently assembled attenuation/depth-PSF operator and fixed adjacent-window scatter. Original forward relative L2 <=0.22 remains. Normalized mean Poisson deviance <=0.1295695136004951 is the independently measured attainable 0.1295685136004951 plus predeclared 1e-6; it is not measured noise. An independent scalar likelihood improvement yields 0.12956534506424106 and forward L2 0.21913413980961838; zero and tenfold activity fail. Stable log differences avoid overflow at zero prediction. This certifies a measurement-consistent conditional reconstruction, with no unique spatial truth certification. The source reconstruction is unchanged.

Historical pixel/parameter identity thresholds and prior notebook reports refer to v1. The external benchmark_readiness_20261029 archive preserves old README/Judge/metrics/tests and actual current CPU command logs, including failed diagnostics. Source algorithms and arrays were not changed.
