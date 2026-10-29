# Provenance, physical corrections and validation

## Scientific contract and source versions

Bound problem: complete real MDCK-colony PIV displacement → two-component surface traction on a300µm bonded elastic layer. Unknown is traction, not displacement, bead-image registration, cell segmentation or monolayer stress. Public observations are legitimately measured displacements, without any true/reference traction or target-derived calibration. Actual force truth is unavailable; baseline is a standard-method result, explicitly labelled as such.

Code https://github.com/fabrylab/pyTFM , exact remote HEAD **757fbe789b18746d5bc6afabeef67918e044c3ad**, verified by network `git ls-remote ... HEAD` on2026-10-20; private complete checkout agrees. GPLv3 license is preserved in LICENSE and attribution in NOTICE. Selected inverse is the original `pyTFM/TFM_functions.py::ffttc_traction_finite_thickness`, reached through `TFM_tractions`; half-space and unsmoothed comparisons use original companion functions. No core upstream source was modified.

Paper: Bauer, Prechová, Fischer, Thievessen, Gregor, Fabry (2021), *pyTFM: A tool for traction force and monolayer stress microscopy*, PLOS Computational Biology17(6), e1008364, published2021-06-21, DOI **10.1371/journal.pcbi.1008364**. Published full text and mathematical discussion were read at https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008364 . Exact publisher PDF archived privately as tfm-paper.pdf. The paper includes both traction and monolayer stress; only its stated traction inverse is selected here, without claiming the separate cell-sheet FEM/stress pipeline.

Data https://github.com/AndreasBauerGit/pyTFM_data , tag R resolved remotely to commit **96e2933d75b5df1f4e49d25927494e0eb982c130**. Archival release DOI **10.5281/zenodo.4047040**, published2020-09-23, experimental data for paper Fig4/5. Files `raw_data_section_Analysis_MDCK_cell_colony/12before_shift.tif` and `12after_shift.tif`, each16182638 bytes. Their repository LICENSE is **CC BY-SA4.0**, not merely CC BY: derived public displacement remains CC BY-SA4.0 with source attribution and transformations in NOTICE/README.

| Asset | SHA-256 |
| --- | --- |
| 12before_shift.tif | 0af286cd604d5964e206752f949ec607a93a8183b6bb4b929ef4fc3c8c528b51 |
| 12after_shift.tif | ee6d44fdf98dfa18916bb28ffacbc7e9e044c891e2b69faeb3cad4cf68ffc8a0 |
| Original data LICENSE | 87a816969906840bf7af8d4d01cdfad4741b18946365e1f286007935509f2edb |
| Bundled raw_data.npz | 460fbd055f3bae288c3315b83407d0f536e8149acc8af1befd6f7bd98f5452ec |

No synthetic observations were generated, no generate_data.py is needed, and no original TIFFs/PDFs are retained in the task. The visible field preserves every measured PIV value in float64 after a reversible row-order change; no downcast, favourable spatial crop or traction-based masking.

## Maintainer-only archive index

Private base `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/earth_mechanics/`:

| Item | Purpose / consumer |
| --- | --- |
| pytfm-upstream/ | Complete exact-revision upstream source, docs, GPL license. |
| tfm-data/{before.tif,after.tif,LICENSE} | Exact bound original data and CC BY-SA terms. |
| tfm-paper.pdf | Published manuscript; not shipped task input. |
| oracle-env/ | Isolated upstream dependencies, never imported by cleaned task. |
| capture_tfm.py | Calls original PIV and elastic functions, applying explicitly identified physical-coordinate calibration before the elastic oracle. |
| tfm_acceptance.json | Acceptance thresholds and physics checks frozen after independent capture and before cleaned implementation; coordinate revision explicitly appended without threshold change. |
| tfm_uncorrected_tutorial_oracle.npz/.log | Separately labelled historical capture with tutorial orientation/approximate pitch, not bundled baseline. |
| tfm_oracle.npz/.log, tfm_oracle_stderr.log | Corrected-coordinate bound observation run and finite/unfiltered/half-space outputs. |
| tfm_oracle_repeat.log, tfm_oracle_repeat_stderr.log | Independent exact rerun verification. |
| openpiv_install*.log | Private upstream dependency installation/compatibility evidence. |
| oracle_numerical_backend.log, cleaned_numerical_backend.log | Numerical backend details. |
| tfm_tests.log, tfm_tests_stderr.log, tfm_notebook*.log | Unit/physics/parity and active notebook execution evidence. |
| tfm_fresh_evidence.json | Full-precision fresh metrics/timing/parity/environment and per-file hashes. |
| tfm_executed_review.ipynb, tfm_review_0.png..tfm_review_2.png | Final executed notebook and visually inspected figure extracts. |
| finish_tfm.py, validate_tfm.py | Private notebook/evidence authoring scripts; no runtime imports. |

## Independent upstream execution

Command `oracle-env/Scripts/python.exe capture_tfm.py` from private base. Exit **0**, corrected full PIV+all3 inverse comparisons compute **0.9823944000527263s**. Repeat same command exit **0**; all captured arrays exactly equal to frozen initial corrected run, max absolute difference0.0. Initial tutorial-convention capture also completed exit0 in1.1413911001291126s but is expressly not the benchmark reference.

Original calculate_deformation uses real full image pair, window100/overlap60, Gaussian subpixel correlation, peak-to-peak1.05 validation, extreme magnitude threshold mean+20std, local mean replacement10iterations/kernel2. Reorder rows to physical y upward, pass exact pitch40×.201 into original TFM_tractions. No reconstruction estimated by cleaned modules is fed to oracle; capture has no src import. The full canonicalized displacement stored in raw_data.npz is exactly equal to the frozen oracle inputs.

Oracle Python3.11.11 Windows build26200, NumPy2.4.6/SciPy1.17.1, OpenPIV0.23.8, tifffile2026.3.3, single-thread environment OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1. pyTFM source was imported directly from the exact checkout in its private environment. Original pyTFM's expected sig2noise_val and replace_outliers API is compatible with OpenPIV0.23.8;0.26.1 was initially inspected then replaced only in oracle environment. No shared package was upgraded. No stochastic algorithm/seed is used for bound-data extraction or inversion.

Warnings: the original finite-layer code evaluates divisions at k=0 before overwriting its DC coefficient with zero, so it emits RuntimeWarning invalid/divide-by-zero. Final arrays are finite. The local implementation handles DC analytically and avoids these warnings without replacing the method. No upstream cosh overflow occurs on this acquisition; local algebraic sech form additionally preserves stability for larger thicknesses.

## Physical discrepancies investigated, with independent evidence

1. **Coordinate orientation.** Original PIV returns v positive upward while image array rows increase downward. Independent known-translation check using a random intensity image shifted down3/right2 pixels gave original output median ux=2.0001318283875236, uy=-3.0008454256986425. Thus reversing output rows (without flipping vector signs) makes both components and spatial axes Cartesian. The exact signed measured values are retained. Using image-row coordinates unchanged in the Fourier tensor would reverse the physical meaning of off-diagonal derivatives. The public data, README and physics model now share one coordinate convention.
2. **Measurement pitch.** The tutorial calculates mean(image_shape/field_shape)*.201≈8.336244260204081µm. Actual interrogation-centre separation is (window-overlap)*.201=**8.040000000000001µm**. Its approximation differs by3.6846%. The selected acquisition passes the exact calibrated pitch into the original elastic inverse; it does not pretend an unchanged tutorial reproduction. The earlier approximate-pitch capture is retained privately, distinct from the benchmark's independent run. Both conventions were investigated before finalizing baseline/metric evidence; thresholds were not loosened.
3. **Double-Nyquist corner.** Original finite code retains the real double-Nyquist cross-coupling while real inverse FFT removes unmatched imaginary terms on single Nyquist lines. Original half-space code explicitly zeros the entire cross-coupling on both Nyquist lines, including their intersection. The first local attempt zeroed all Nyquist cross terms, producing finite-parity max error0.121Pa. Fourier conjugacy analysis identified the corner difference; local finite stiffness now retains it, and half-space retains the original explicit zero. The finite/infinite physics limit is tested away from that single discrete convention, with no changed parity tolerance. Final whole-array parity is≤2.28e-13Pa.
4. **Unknown scope and dataset result.** This is a physically calibrated application of the published finite-thickness kernel to its real experimental observations. It is not a claim to reproduce the paper's exact Fig5 values, cell-stress maps, contractility mask or boundary line tension. No omitted FEM or cell boundary is presented as reconstructed. The benchmark reconstructs all measured-field traction components and retains finite/half-space/unfiltered scientific comparisons.

## Equation/function/test traceability

| Paper requirement / stage | Original file/function | Local callable | Independent test |
| --- | --- | --- | --- |
| Experimental displacement observation | TFM_functions.calculate_deformation | frozen public measured displacement; preprocessing.load_data | source input arrays exact equality; shift/orientation investigation above |
| Pixel calibration, remove rigid translation, finite support | ffttc_traction_finite_thickness preparation | preprocessing.prepare_displacement | known input/crop/padding example; exact pitch test; rigid-translation invariance |
| Finite-layer Green tensor inversion | ffttc_traction_finite_thickness equations and factors | physics_model.stiffness | analytic longitudinal/transverse sinusoidal eigenmode responses; SPD/positive work; compliance roundtrip |
| FFT vector traction | same original finite function | physics_model.apply_spectral; solvers.reconstruct | complete finite/unfiltered upstream fixtures, linearity, zero input |
| Gaussian physical resolution | TFM_tractions fs conversion + scipy filter | solvers.reconstruct smoothing_um | full finite smoothed vs unfiltered reference parity |
| Half-space comparison | ffttc_traction | stiffness thickness=None; reconstruct mode=infinite | full half-space parity; analytic limit away from documented Nyquist corner |
| Evaluation metrics / field reporting | benchmark declared cosine/range definitions | independent judge.metrics; notebook calls same Judge | known numerical cases; signed reversal and zero rejected |
| Standard execution flow | selected API usage | src.main.main; active notebook | CLI/backend smoke, exact output contract, nbconvert full active inverse |

## Fresh result, parity and threshold provenance

Shared runtime Python3.11.11, NumPy2.4.6, SciPy1.17.1, Matplotlib3.11.1. Cleaned runtime imports only these documented packages and stdlib. No OpenPIV, pyTFM, tifffile, domain solver, vendored framework or cross-task import exists in source/Judge/tests/notebook. CPU complete inverse **0.0005930999759584665s**; notebook measured0.000611s. Small50×50 FFT fields require negligible memory; no GPU or large-memory exception.

Fresh whole-array max absolute differences against independent upstream: finite **1.1368683772161603e-13Pa**, unfiltered **2.2737367544323206e-13Pa**, infinite **2.2737367544323206e-13Pa**. Frozen parity acceptance rtol1e-10/atol1e-8Pa passes without modification. Independent rerun differences0.0, so strict positive Judge tolerances NCC≥.99999999 and dynamic-range NRMSE≤1e-7 safely exceed repeat/FFT arithmetic noise and remain non-vacuous. Reference self-comparison1/0 is reported only as baseline context; the fresh method is independently measured.

Fresh finite NCC **1.0**, dynamic-range NRMSE **2.1602394498225148e-17**, RMS component traction **95.54680367615819Pa**. Fresh half-space NCC versus finite .9999998922302057, NRMSE3.6809041316999165e-5; unfiltered finite NCC .9253356552008623, NRMSE .045953762408579506. The half-space is close because300µm thickness is large relative to most measured wavelengths; this is measured rather than invented. It is still distinct from the required finite result. Standard reference agreement is not proof of physical accuracy; no calibrated noise variance exists to claim chi-square≈1.

## Executed checks and review

From task directory using the shared interpreter:

* `python -m src.main`: exit **0**, actively computes both signed components into output/reconstruction.npz.
* `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`: exit **0**, status pass, fresh metrics above.
* `python -m pytest test/ -v`: final exit **0**, **17passed in1.25s**. Unit tests cover missing file/key, shape, float32/complex, NaN/Inf, poor zero/reversed result, evaluator-assets missing, non-task working directory, report-write errors, independent numeric NCC/NRMSE examples, CLI and unchanged plotting backend; parity/physics tests above. Initial red test failed import before any local method existed. Intermediate parity failure was investigated and corrected mathematically, not hidden or threshold-relaxed.
* `python -m nbconvert --to notebook --execute notebooks/mdck_finite_thickness_traction.ipynb --output mdck_finite_thickness_traction_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=research`: exit **0**. Private JUPYTER_PATH selects a kernel pointing to the shared validated runtime; TEMP/TMP use private writable temp.180s allows kernel/plot overhead despite submillisecond inverse. Notebook actively computes finite/infinite/unfiltered paths, Judge, signed fields, vector plot, standard-method residuals and paired numeric/bar comparisons. Final executed outputs embedded in the maintained notebook; temporary output execution file archived privately then removed.

Visually inspected all3extracted figures: measured signed components and units readable; fresh traction maps computed; common-scale upstream/fresh/half-space magnitudes; numerical-difference map ≈1e-13Pa; finite-versus-half-space≈0.17Pa; vector directions; RMS comparisons95.55Pa and130.90Pa before smoothing. None is labelled truth. No library calls matplotlib.use().

Semantic/signature review checked actual FFT normalization, µm/pixel scaling, physical pitch, array axes, mean subtraction, padding/crop, gamma factors, finite/infinite Nyquist corner, DC, stable hyperbolics, Gaussian sigma6µm→pixels, all output components and comparison paths. README acquisition constants agree with source and tests. Notebook source constants and Judge use consistent metrics; no disconnected visualization metric implementation exists.

## Completion gates

| Gate | State | Evidence |
| --- | --- | --- |
| Paper, remotely verified source/data revisions, hashes, mapping | passed | Source/archive sections and table |
| Original/cleaned exact bound-data execution | passed | Full canonicalized observation passed to unmodified inverse; both exit0 |
| Independent parity + mathematical physics | passed | Three whole-field upstream comparisons; analytic eigenmode/energy/roundtrip/DC checks |
| Constants/units/coordinates faithful and discrepancies explicit | passed | Exact physical calibration correction documented, tested and upstream rerun; no unchanged-tutorial claim |
| Standard CLI and required-array Judge | passed | Both components scored, strict NPZ and fresh pass |
| Invalid/poor/evaluator/report contracts | passed | Full17-test suite |
| Self-contained public README/raw/environment | passed | All model constants/axes/output keys public, no metadata sidecar |
| No hidden information or GitHub URL in solver allowlist | passed | Only measured displacement and independently known calibration; README/requirements no forbidden URLs |
| Minimal runtime/local inverse mathematics/symbol audit | passed | NumPy/SciPy explicit tensor, no upstream/cross-task/runtime dependencies |
| Active notebook, comparisons, visual inspection | passed | Executed and inspected full3-variant reconstruction |
| Provenance and nonzero reference tolerances | passed | Independent repeats, frozen acceptance, historical/unmodified-core/fresh distinctions |
| Plans/README execution/signature semantics | passed | Reviewed actual shipped call paths and mathematical conventions |
| Template inventory/licenses/cleanup | passed | Required files, GPL/CC BY-SA notices; no TIFF/PDF/temporary executed output in task |

No mandatory gate remains failed or unverified. Physical model accuracy for noisy measured displacements remains a stated limitation rather than a ground-truth claim.
# Independent parent acceptance, 2026-10-20

The coordinating parent reran all17tests in the shared research runtime: **17 passed in1.27s**, exit0. A separate foreign-working-directory standaloneJudge invocation on the actively reconstructed output exits0. Parent AST/import/call-role, public-array, README, executed-notebook and final artifact inventory checks report no errors; required GPL/CC-BY-SA legal notices are retained. The parent actually viewed all three executed scientific figures, including the complete signed displacement/traction fields, independent-original/fresh/half-space comparison and residual/direction panels, and smoothing-resolution/paired method statistics. The explicit stiffness/compliance, exact8.04µm measurement pitch, reversible row-coordinate change, DC removal, finite/infinite Nyquist conventions, Gaussian resolution and actual caller interfaces agree with the plans and contract. This accepts the physically calibrated selected elastic inverse application, not an unchanged tutorial image reproduction or independent true-force measurement. Private evidence: `<batch>/mdck_finite_thickness_traction-parent-audit.json`, `mdck_finite_thickness_traction-fresh-independent-judge.json`, and the parent pytest directory, under `research_batch_20261020`.

