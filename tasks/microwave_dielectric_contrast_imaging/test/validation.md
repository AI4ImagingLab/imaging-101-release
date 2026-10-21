# Provenance and validation

This task reconstructs a **spatial real dielectric-contrast field** from indirect complex scattered fields. It reuses an author-existing simulation, not a new acquisition, real measurement or reconstruction relabeled as truth. The selected method is the modern EISPY2D implementation of Born iteration, not purported original 1989 software. The output contract, observations, method and acceptance boundaries were fixed before local extraction.

## Sources, versions, rights and private archive

The maintainer-only batch archive is `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021` (abbreviated B4). Original source/data and oracle evidence are in `B4/root/microwave_screening/eispy2d` (O); this extraction, experiment logs, paper copies and figure exports are in `B4/bio_med/microwave` (P). Neither archive is imported by the shipped runtime or tests.

| Source | Exact provenance |
|---|---|
| Foundational method | Wang and Chew, *An iterative solution of the two-dimensional electromagnetic inverse scattering problem*, 1989, DOI https://doi.org/10.1002/ima.1850010111. Nine-page academic-hosted paper https://www.ee.iitm.ac.in/uday/2015a-EEL766/chew_89.pdf is archived as P/wang_chew1989.pdf; extracted text is P/wang_chew1989.txt. |
| Modern author paper | Batista, Adriano and Batista, https://arxiv.org/abs/2111.02185v2, version2 dated12January2022, thirty-page O/eispy2d_arxiv211102185v2.pdf. Related final IEEE Access DOI https://doi.org/10.1109/ACCESS.2025.3573679; the archived 2022 PDF is explicitly not represented as the2025 final publication. |
| Author reference implementation | https://github.com/andre-batista/eispy2d exact commit `a72831db93d64c7b80282579f48de4c110627eeb`; parent remotely verified API history in O/remote_commits.json. O/source.zip SHA256 `188f291858fbf66d63f15ad7a48e094ebdcfdac5c8ec33947db550500ddc19dc`; O/provenance.json records4126697bytes and exact archive URL. O/upstream retains full source. |
| Existing author data | Committed `cst_basic` at the exact source revision, SHA256 `43063ec28a337b9a5dbe48748bc6ea6e75d68ef048c22ed2b3a1dc29a65a10d1`. Original simulated triangle dielectric object, lossless background relative permittivity4, object8, native60² material array. All9receivers and10plane waves retained. The target geometry is hidden, not a public solver hint. |
| Public binding | O/bound_observations.npz, copied byte-for-byte as data/raw_data.npz, SHA256 `88d24975377c1da3566f2fc1ebdf0e9d974398c4530d80cb61cdf7c897d322cd`. It contains only measurements and independent acquisition angles. |
| License | Author repository GPLv3, preserved in task LICENSE/NOTICE with adaptation attribution. No separate restriction was found for its committed author simulation. Paper copies are private scholarly source evidence, not redistributed in the task. |

The signed full inventory and per-file SHA256s are P/manifest.json. Six-task source/report navigation is B4/bio_med/report_catalog.json. All URLs above are maintainer-only; public README/requirements contain no GitHub-domain URLs.

## Oracle-first run and preimplementation decisions

Parent scripts `B4/root/microwave_probe.py`, `capture_microwave_original.py` and `freeze_microwave_original.py` execute the real selected author modules in the isolated `B4/root/oracle-env/Scripts/python.exe` environment. Its NumPy2.4.6/SciPy1.17.1 and original Numba dependencies are documented in the run archive; the shared runtime is unchanged. Top-level convenience `__init__` reexports reference an absent cgm.py, so the oracle uses package namespace shims only to bypass those broken reexports. The real Richmond, regularization, BIM and MoM-CG-FFT source is imported normally and unchanged. Nonselected stochastic archive classes are passive Holders only while loading metadata; selected test/configuration classes are real original classes. The inverse never reads stored reconstructed images.

Original alpha0.001,0.01,0.1 each execute15outer updates on these exact observations, recorded in O/original_bim_0p001,0p01,0p1.{json,log,npz}. The independent standard repeat in O/original_frozen.{json,log} takes135.12304319999748s, exit0, output **bit identical** to the first original standard run. O/original_stages.npz freezes geometry, matrices, incident fields, every complex contrast and field input, final field, legacy objectives and observations. The original physical REAL-contrast observation residual is0.023201646071690542; state residual9.443187945667526e-9. Original alpha0.01 takes125.75485010000193s; all original exact commands, warnings and logs remain privately indexed.

Before local source existed, P/capture_extra.py imported only these original selected modules and froze P/extra_original_stages.npz: extended kernel, A/AH actions, first linear kernel and three actual inner iterations. P/extra_capture.json records original execution and historical group sizes. This independent source fixture is copied into test/fixtures, never regenerated using cleaned functions.

Parent fixed native30² real cell-mean contrast rather than the original software's60² nearest-neighbor display. O/coarse_actual_truth.npz averages the actual author60² contrast over exact2×2 areas. `test_truth_contract.py` verifies this independently from test/fixtures/truth_averaging_case.npz. O/coarse_original_reference.npz is separately the last original complex contrast's real part. No fitted scale, clipping, crop, shift, rotation, interpolation or global phase adjustment is permitted.

Preimplementation O/preimplementation_criteria.json freezes stage rtol2e-8/atol2e-9 and full rtol1e-6/atol1e-7. Original true-reference baseline NCC0.8953621762316742 and dynamic-range NRMSE0.09552068704478095 imply fixed boundaries NCC≥0.8058259586085068 and NRMSE≤0.10507275574925905. Physical residual≤0.0255218106788596 is independently required. These were not relaxed after extraction. Negative ringing is preserved: minimum contrast near−0.105 still implies positive relative permittivity near3.58.

## Mathematical traceability and discrepancies

| Mathematical stage | Original source/function | Local function | Independent validation |
|---|---|---|---|
| TM scalar volume-integral state and observation equations; outgoing exp(+iωt) convention | Wang–Chew integral-equation formulation; discretization/richmond.py richmond_data/richmond_state | physics_model.green_matrices, scattered_field | Exact frozen GS/GD; dense FFT-equivalence and weak-scattering tests |
| Equal-area disk integration and analytic self term | discretization/richmond.py | green_matrices, extended_kernel | Exact original coefficients; no discarded Hankel singularity evaluated |
| np.arange half-cell coordinates, circular receivers and negative-sign incident phase | core/configuration.py, forward/mom_cg_fft.py incident_field | coordinates, incident_field | Exact original arrays; independent plane-wave sign test |
| Receiver-major/source-minor K[(m,s),n]=GS[m,n]Et[n,s] | discretization/collocation.py kernel_GSE | solvers.linear_kernel | Original first K and independent scalar indexing |
| Complex Tikhonov normal equation, no observation normalization | inverse/regularization.py tikhonov | solvers.tikhonov | Original first contrast and regularized stationarity |
| Alternating contrast/total-field updates | inverse/bim.py solve | solvers.reconstruct | Every15complex contrasts/field inputs, final field and legacy objective; full output |
| Conjugated Hankel-first-kind zero-extended FFT interaction | forward/mom_cg_fft.py __get_extended_matrix, __fft_A, __fft_AH | extended_kernel, prepare_fft, fft_A, fft_AH | Original kernel/actions, independent dense complex operator and adjoint identity |
| Shared all-source step denominator and per-source PR-like update | forward/mom_cg_fft.py CG_FFT | cg_fft, total_field | Original first3inner steps; physical state residual and zero contrast |
| Real εr roundtrip for physical propagation; complex intermediate diagnostic | inverse/bim.py and forward/mom_cg_fft.py solve | physical_contrast, reconstruct | Separate physical versus legacy curves; independent standalone dense Judge |
| Full-array true-reference NCC and DR-NRMSE | Declared benchmark definitions, not author PAD | visualization.ncc/nrmse and independent evaluation/judge.py | Analytic metric cases and crossimplementation tests |

Important scope/convention decisions: the original foundational paper uses vacuum contrast; this source generalizes to background4. Its historical illustrative stopping policy is not asserted to be the selected modern15updates. The author source solves **complex** linear contrast, then propagates a **real** material; its legacy objective uses complex intermediate contrast and therefore differs from a physical lossless-field residual. Both are retained, with distinct labels. The original custom multi-source CG formula has a shared Frobenius denominator, and is preserved rather than silently replaced by textbook CG. FFT kernel transforms are cached with the same tiled source shape; this removes redundant work without altering mathematics. Temporary singular diagonal distances are replaced before assigning the same analytic self term. Vectorized K assembly may differ from Numba scalar multiply by one ulp; fixed stage tests cover amplification through iteration. No physical correction or method replacement was necessary.

## Hidden assets and historical context

| File | Consumer and provenance |
|---|---|
| evaluation/ground_truth.npz | Actual author simulation contrast, exact2×2 physical cell means; Judge and notebook posthoc metrics only. |
| notebooks/reference_outputs/upstream.npz | Original15-step native30real result, not truth; comparing tests and notebook comparison. |
| test/fixtures/oracle_stages.npz | Original stage outputs, prefix output_ for clarity; comparing test consumes every retained key. Original60display and truth omitted here to avoid duplicate assets. |
| test/fixtures/extra_oracle_stages.npz | Prelocal original inputs/operators/three inner steps/first K, consumed by fast stage test. |
| test/fixtures/truth_averaging_case.npz | Actual native60contrast; unit test verifies exact cell averaging. |
| notebooks/reference_outputs/historical_author_results.npz | Original committed DE/PSO/GA3groups×30runs; explicitly cached/unreproduced historical context, consumed only by notebook/visualization. |
| notebooks/reference_outputs/cleaned_metrics.npz | Fresh active notebook scalar/table measurements, written then read in the notebook as a separately identified record, never a reconstruction input. |

Historical author `zeta_epad` is100*mean(abs((εtrue−εestimate)/εtrue)) on original60² relative-permittivity maps, from core/result.py compute_zeta_epad. It is not DR-NRMSE, not contrast error and not a fresh result. Figure5 retains all three30-run distributions and median-ranked representative maps, labeling these historical/different-grid data separately from the three fresh BIM runs. No claim of new DE/PSO/GA execution is made. The selected BIM scope and intentional distinction are explicit in README/notebook.

## Execution, metrics and review evidence

All cleaned execution uses `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`, Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1, Windows CPU. No upstream imaging package is installed/imported by cleaned code. Environment: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. P/memory_check.json records17,808,108KiB free physical and39,430,156KiB free commit before verification, exceeding6GiB; process arrays are well below1.5GiB. No shared package upgrade or large download occurred.

P/verify_final.py runs CLI, full tests and notebook with independent output destinations; commands, elapsed times and complete stdout/stderr are P/final_cli,final_judge,final_pytest,final_notebook.{json,log}. Standard task cwd is tasks/microwave_dielectric_contrast_imaging. Required commands are `python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test/ -v -p no:cacheprovider`; and `python -m nbconvert --to notebook --execute notebooks/microwave_dielectric_contrast_imaging.ipynb --ExecutePreprocessor.kernel_name=imaging-101-research --ExecutePreprocessor.timeout=1200` with private temporary output. Timeout1200s accommodates all three15-step CPU reconstructions and five figure exports with margin. All three run actively through local source, not cached reconstruction substitution. The first notebook code explicitly selects inline rendering. Optional IMAGING_REVIEW_EXPORT/PREDICTION/REPORT only route maintainer exports away from concurrent CLI writes; defaults remain self-contained task output paths.

Initial local nominal solve took77.6459664s: max final output difference7.156658390905513e-10, max all-stage complex contrast difference5.666497176603021e-9, field-input difference6.6296867957918145e-9, legacy-objective difference2.636870546132014e-10. All geometry/operators and independent3inner steps agree exactly. Fresh nominal NCC0.8953621762338124, DR-NRMSE0.09552068704215007; iterative physical residual0.023201646086564425 and state residual9.47384030928378e-9. Independent dense Judge residual0.023201636171553772 differs only at forward stopping tolerance. Actual coarse truth independently has residual0.014338320686576313, establishing a reasonable physical criterion despite coarse discretization and existing noise. It is never inserted into an inverse run.

Final exact command outcomes, experiment table, visual inspection and gate signoff are appended below after the complete verification. Machine-readable exact results and artifact hashes are authoritative over rounded prose.

## Scientific limits

These existing simulated observations establish a sparse9×10 single-frequency2D lossless scalar benchmark, not real-material validation or a3D/full-vector claim. Original60-cell synthesis and30-cell inverse pulse discretization differ. Phase calibration and normalized incident amplitude are assumed known, so this is not absolute radiometry. Sparse angular/frequency sampling and Tikhonov bias limit identifiable detail; a low residual is not uniqueness. The bound square30 implementation does not assert untested arbitrary rectangular-grid support. Author noise is fixed-amplitude random-phase complex perturbation, not Gaussian noise. Historical stochastic outcomes cannot be compared numerically to fresh BIM metrics without matching grid and metric definitions.

## Final completed verification (2026-10-21)

| Command | Exit | Wall seconds | Evidence |
|---|---:|---:|---|
| final_cli | 0 | 66.334305 | P/final_cli.json and .log |
| final_judge | 0 | 0.706097 | P/final_judge.json and .log |
| final_pytest_private | 0 | 73.020889 | P/final_pytest_private.json and .log |
| final_notebook | 0 | 215.042719 | P/final_notebook.json and .log |

Full suite: **25/25 passed** with explicit private `--basetemp P/pytest-final-private`. The first default-temp attempt had12passed/13setup errors caused solely by Windows permission denial in system pytest-of-admin; P/final_pytest.{log,json} preserves that unsuccessful command. No algorithm, test assertion or acceptance boundary was changed. The corrected command reran all25tests, including15-stage parity.

Canonical notebook: 3 executed code cells, zero error outputs, **5 nonempty image/png scientific groups**. All5private PNGs were visually inspected at full review resolution; PNG/SVG/PDF and exact numeric_sources.npz/metrics.json are in P/gallery. Scalar images preserve signed values, paired field plots use common color scales, dimensions/units and historical labels are legible. The source’s historical60-cell raw-error summaries differ from native30-cell benchmark metrics and are not compared as equal metrics.

| Fresh alpha | Native30 NCC | Native30 DR-NRMSE | Physical residual | Actual solve seconds | Original30 max absolute difference |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.8953621762338124 | 0.09552068704215007 | 0.02320164608656443 | 66.15651470000012 | 7.156659362350659e-10 |
| 0.01 | 0.8874720225395006 | 0.1041580642499178 | 0.148486859437215 | 65.18554589999985 | 2.55351295663786e-15 |
| 0.1 | 0.8194858029792061 | 0.1595606992433612 | 0.5503281012110363 | 64.64585749999969 | 2.775557561562891e-16 |

All three fresh alpha results independently satisfy the preselected full numerical parity tolerance versus their original native30solution. The original60display is verified to consist of exactly repeated2×2cells before extracting native values for this private check; **no evaluation prediction or truth is downsampled by the Judge**. Alpha0.01/0.1 are sensitivity comparisons and need not pass the selected nominal physical-fit threshold.

Final gate ledger:

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/revision/data/license and mathematical mapping | PASS | Sources table, original archives and traceability table |
| Original exact-bound runs and independent rerun | PASS | O/original_frozen.json; original3alpha archives |
| Independent stage/full/physics/adjoint checks | PASS | Full25tests; final_audit.json |
| Constants, coordinates, signs, units and output contract | PASS | Exact original geometry; public contract and physical tests |
| CLI and strict independent Judge | PASS | final_cli/final_judge records; fresh output report |
| Malformed, poor, overflow, foreign-cwd and evaluator-failure handling | PASS | 16Judge tests |
| Solver allowlist sufficiency/no hidden truth or URL leak | PASS | Raw allowlist only measurements/angles; public scan |
| Minimal shared runtime/local mathematics/no cross-task imports | PASS | AST import/call-role audit and full runtime execution |
| Active complete NB and required scientific comparisons | PASS | 3freshalphas;5inlinePNG; no errors |
| Metric/provenance/reference distinctions | PASS | Actual cell-mean truth; historical PAD distinct; independent dense residual |
| README/plans/interfaces semantic consistency | PASS | final_audit.json signatures and explicit equation/call review |
| Final inventory, private figures and license | PASS | manifest.json; LICENSE/NOTICE; no temporary notebook in output |

No failed or unverified mandatory gate remains. Completion here means ready for independent parent audit, not parent acceptance. No other frozen task was modified.

Numerical backend supplements: P/original_environment.json and P/cleaned_environment.json capture the actual executables, Python/NumPy/SciPy versions, CPU/thread settings and complete NumPy/SciPy compiler/BLAS/LAPACK configurations. NumPy uses scipy-openblas0.3.31.188.0 ILP64; SciPy uses scipy-openblas0.3.30 LP64. Captured original Numba0.68.0 is upstream-only and never imported by a shipped source/test/notebook/Judge. `capture_environment.py` ran under each respective interpreter without installation or upgrades; show_config emitted only the optional-PyYAML formatting warning, and no PyYAML dependency was added. Original exact standard capture command is `B4/root/oracle-env/Scripts/python.exe B4/root/freeze_microwave_original.py`; supplemental source capture command is that interpreter followed by `P/capture_extra.py`. The source's known discarded diagonal Hankel singularity warnings and convenience import issue are recorded rather than treated as physical failures. Final notebook emitted the standard Windows Proactor/ZeroMQ selector-thread warning and local-kernel TCP warning; execution completed with zero notebook error outputs.


## Parent acceptance review

Parent independently ran25tests (25 passed in71.24s), foreign-cwd Judge0 and source/inventory audit errors[]. Read explicit Hankel cell/self integrals, finite Toeplitz FFT/complex adjoint, source shared-denominator iterative state solver and complete Born updates; inspected all5 actual scientific groups. Original author-existing simulated9receiver/10illumination fields reconstruct native30-square real contrast; actual fine simulation truth is exact physical2x2 cell-area averaged, not reconstructed display interpolation. All15 source stages and independent dense physical state/observation checks pass predeclared tolerances; three alphas actually reconstructed. Historical DE/PSO/GA30-run groups are explicitly cached and unreproduced, with different grid/metric context kept separate. Standard truth NCC.89536, dynamic-range NRMSE.09552 and dense physical observation residual.02320; no sign/scale/phase alignment or clipping.
