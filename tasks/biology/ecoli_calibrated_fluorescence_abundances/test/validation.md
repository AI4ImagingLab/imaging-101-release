# Benchmark acceptance revision 2 — 2026-10-29

The reference algorithm remains historical local nonconvex SLPRU described below. Acceptance now assesses the separately declared global convex target: normalize interior Y/255 by its RMS to B, minimize generalized KL(B||MU)+0.1 nuclear(U)+0.1 sum row-L2(U), U>=0. The submitted max-one field has one missing common scale profiled analytically. This is an explicit semantic change, not equivalence to inverse-singular-value reweighting. Digital intensities are not calibrated Poisson counts. The independent Judge imports no reconstruction functions or hidden maps and retains the existing observation residual and physical/nonnegative/max-one/border constraints.

Objective-per-datum limit0.07310834324578482 equals the feasible independently reproduced standard field value0.07310833324578482 plus1e-8 numerical allowance. Independent per-pixel NNLS initialization followed by general L-BFGS-B convex fitting yields0.07229811631277298 and observation error0.3318427636271077, passing without matching SLPRU. This control reaches its500-iteration budget; no global convergence certificate is claimed or needed to verify its feasible quality. Constant and reversed-spectrum controls yield0.5196790091932176 and0.8628110268585083 and fail; zero also fails. This is an explicit attainable regularized quality bound, not a broadened image-agreement tolerance.

Private evidence under benchmark_readiness_20261029: ecoli_metrics_v1.json, ecoli_v2_controls.json/log, ecoli_pytest_v2.log, ecoli_v2_metriccell_report.json. Full tests:22 passed in25.52s. A scale-profiling unit control verifies invariance to common gain. Algorithm and original arrays are unchanged. Notebook diagnostics are labeled historical; its current evaluator call retains report/exit API and was exercised on the existing reconstruction. Fresh blind agent reconstruction was not run.

# Provenance and validation

## Bound experiment and independence

This biology task binds the entire 30 x 30 x 24 E. coli demonstration distributed with Wang et al., *Unmixing biological fluorescence image data with sparse and low-rank Poisson regression*, Bioinformatics 39(4), 2023, btad159, DOI 10.1093/bioinformatics/btad159. Primary full text is https://pmc.ncbi.nlm.nih.gov/articles/PMC10081874/ . Source is https://github.com/WANGRUOGU/SL-PRU at remotely verified commit `36a41ae5deb042aefa57bf6c977081cd0e8c4f67`; GitHub API revision/tree response is archived. Source and bundled data are distributed under the repository MIT license, reproduced in task LICENSE; no independent TIFF-specific restriction was found.

Classification: **author_real_processed**. The paper describes confocal spectral measurements of fixed stained E. coli using Zeiss LSM710/880 instruments and 488/561/633 nm excitation. The actual mixture TIFF retains ImageJ microscopy metadata (UTF-16 big endian tag50839): Plan-Apochromat63x/1.40 Oil DIC M27 objective; x/y pitch9.9521491860627815e-8m; spectral channel endpoints and bandwidth. Its stored array is only30x30x24 uint8, whereas retained parent-image metadata mentions1356x1356. We therefore call this the author demonstration ROI, never the complete parent field, raw photon counts, or unprocessed camera data. No resizing, additional crop, synthetic noise or channel selection was performed locally. Conversion reads every TIFF frame, stacks channels last, and adds a batch axis; uint8 values are preserved exactly.

`endmembers.mat` contains the supplied24x13 matrix E. The paper and author demo identify separately acquired single-stain controls; all13 control TIFFs are retained privately. E has independently calibrated, unit-maximum emission columns and is not inferred from the hidden abundance map. The author's PNMF control-extraction function actually uses Euclidean multiplicative updates despite its name; this task uses the supplied calibration, does not execute PNMF, and makes no claim of freshly reestimating it or of Poisson calibration. All calibration dimensions and full rank are checked.

Nearest existing bindings reviewed: mcr_hyperspectral and raman_cell_phenotyping estimate unknown spectral endmembers and mixture fractions from blind spectroscopy; this task estimates fluorescent brightness fields with independently measured, fixed label spectra, no sum-to-one constraint, and a spectral-confocal acquisition. A change of regularizer or organism alone is not the novelty claim.

No independent true abundance measurement exists. `evaluation/baseline_reference.npz` is the executed original SLPRU result, never ground truth. Observation agreement and numerical reproduction do not establish biological accuracy.

## Private archive and immutable identities

Maintainer archive relative to batch11: `bio_med/archive/SL-PRU/`. It contains `remote_metadata.json`, `paper.xml` (primary article BioC text), `ReadMe.txt`, `code/`, all native `data/` files, `instrument_metadata.txt`, `oracle_input.mat`, `oracle.m`, instrumented `SLPRU_capture.m`, `oracle.log`, `oracle_rerun.log`, four `original_*.mat` outputs, `rerun_SLPRU.mat`, `window.mat`, `stage_*.mat`, `frozen_acceptance.json`, and `final_evidence.json`. Figure PNGs are inspection artifacts outside the task. Private preparation/capture-conversion scripts are `bio_med/prepare_slpru.py`, `freeze_slpru.py`, `make_slpru_notebook.py`, and `finalize_slpru.py`. None is imported by the task.

| Asset | SHA256 |
|---|---|
| Native mixture.tif | 306df7b918d53e1b78210768a98d9666d19d294866c994c69075474a6797bd79 |
| Native endmembers.mat | 3dfb6b658fc7a64bf12f4905e619e8283c2c9f7098e767d73b99543a2083a091 |
| Public data/raw_data.npz | d827b84a47f6291553c6fdd64dd4579487e130b784d61753cedf8b14b45bb456 |
| Original SLPRU MAT | b616bc9be14b7e0ed5adc6697e343c52ec62bd444a25da4a8d4c95428637f7cf |
| Independent original rerun MAT | f16d988e260cde6f544302fdd059b4f3b0551d70d18a735a6788948de2986e2e |
| Criteria frozen before extraction | a2badbb9a2518c7d4abd9540c18b3866e76e21fcbf32696c988ba7278df44458 |

## Original execution before extraction

Date2026-10-27. Windows10 build26200, CPU single-thread (`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`), GNU Octave10.3.0 with image package. Existing portable executable is indexed in the batch archive; BLAS library identity is printed in oracle.log. No randomness or training is involved. Command from private source root: `octave-cli.exe --quiet oracle.m`. The harness supplies a MAT conversion of the exact TIFF and calibration, replacing only the MATLAB Tiff reader, and calls unmodified `UnmixSpIm` and `SLPRU`/`SLNLS` functions. It selects the source-exposed penalties0.1/0.1. Source arithmetic, tolerances, max1000 iterations, residual schedule and normalization are unchanged. Runtime seconds: SLPRU212.6359009743, NLS46.7234780788, SNLS40.3892941475, SLNLS45.5368778706. Exit0. A first harness attempt failed before reconstruction because Octave lacks `blas_version`; metadata capture was changed to `__octave_config_info__('BLAS_LIBS')` and the complete oracle rerun succeeded. This was a harness compatibility correction, not an algorithm correction.

Diagnostic capture inserts snapshot saves after updates at iterations1,11,101,501 in a separate copy of SLPRU. Its final window output equals unmodified SLPRU exactly (asserted by oracle). `window.npz` contains the original central pixel's complete3x3 window and final U/A; stage files contain scaled observations, iterates, split variables, duals, penalty and residuals. Stage fixtures are never generated by cleaned code. A second independent complete unmodified SLPRU run produced exactly the same image (maximum absolute difference0).

## Equations and stage traceability

| Mathematical stage | Original source | Local implementation | Independent validation |
|---|---|---|---|
| Calibrated linear emission M a | SLPRU.m matrix products | physics_model.forward/adjoint | Independent inner-product adjoint identity; stage MU-dependent outputs |
| uint8/255, column-major3x3 neighborhoods | UnmixSpIm.m | preprocessing.detector_units/window | Known integer/axis case and original window fixture |
| RMS scaling and nonnegative least-squares initialization | SLPRU.m/SLNLS.m | solvers.solve_window | Original Y and first-stage U/V fixtures |
| Four-split normal-equation U update | SLPRU.m | solve_window matrix inverse/update | Iterations1,11,101,501 fixtures |
| Positive quadratic Poisson proximal root | SLPRU.m | physics_model.poisson_prox | Analytic stationarity and finite-difference likelihood gradient |
| Reweighted singular-value shrinkage | SLPRU.m | solve_window SVD update | Original V2 snapshots |
| Reweighted fluorophore row shrinkage | SLPRU.m/VectorSoftRow.m | solvers.group_shrink | Original V3 snapshots, zero-penalty identity |
| Positivity, dual signs, penalty rescaling | SLPRU.m | solve_window V4/D/residual block | Original duals, residuals, mu and final window |
| Gaussian comparison fidelity and penalties | SLNLS.m, UnmixSpIm.m | solve_window(poisson=False), reconstruct(method) | All four complete original images in active notebook |
| Central vector and whole-image maximum normalization | UnmixSpIm.m | solvers.reconstruct | Complete fresh pipeline against independent original |
| Evaluation metrics | Explicit independent definitions | Standalone judge plus visualization | Independently known numeric vectors; malformed/poor input cases |

The source implements reweighted shrinkage, not a fixed convex nuclear/group penalty. The local plan records those exact updates and avoids claiming the simple convex objective uniquely describes every iterate. With group penalty0 the source expression has a removable0/0 at zero rows; local code evaluates its identity limit explicitly. Entirely zero windows return zero explicitly. These edge extensions do not change the bound selected-method output and have tests. No other scientific correction was made. Some windows reach the original iteration cap; the notebook reports the actual fractions instead of asserting convergence.

## Frozen acceptance and interpretation

Before local extraction, frozen_acceptance.json fixed full-image atol2e-6/rtol2e-5 and stage atol1e-8/rtol1e-7, Judge NCC>=0.999999 and dynamic-range NRMSE<=2e-5. These conservative float/SVD portability tolerances are nonzero although same-environment original reruns are identical. They were not enlarged after inspection of cleaned results. Central stage maximum discrepancy was approximately2.08e-10; complete local image maximum error is approximately1.2e-13, orders of magnitude smaller than the bound.

The original observation relative error is0.33461260886103705. Before cleaned result inspection its upper boundary was set to1.1 times that value,0.3680738697471408. This is a fixed auxiliary consistency guard, not a claim of an accurate noise model or threshold on a true abundance field. The single nonnegative global intensity gain is needed because the declared output deliberately discards global brightness. Only the fixed interior is used; reference image comparison uses the full declared array without gain, crop or alignment. The notebook compares residuals, allocation sparsity, cap fractions and the group-penalty ablation so the fairly large residual cannot be hidden by the near-perfect reference NCC. Baseline self-NRMSE0 is not used mechanically as a zero acceptance threshold.

The default NCC is cosine similarity with no mean subtraction; NRMSE is RMSE divided by reference dynamic range. Metric values in machine assets retain precision. No historical paper result is represented as a measurement of this bound ROI.

## Clean execution and review

Clean runtime: project research-runtime Python3.11.11, NumPy2.4.6, Matplotlib3.11.1, CPU Windows10 build26200; all numerical threads1. Requirements list only runtime packages. SciPy/PIL are private conversion tools, not cleaned dependencies. No domain library, external source import, cross-task import or submitted-code execution occurs. `python -m src.main` completed and produced the required finite float64 NPZ. `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` returned0.

Full tests: `python -m pytest test/ -v --basetemp=../../archive/SL-PRU/pytest-tmp` returned0, **21 passed in55.65s**. The default Windows temp root initially caused13 fixture setup PermissionErrors, while8 tests passed; the isolated writable basetemp resolved infrastructure errors without changing tests or thresholds. Cases include full pipeline, snapshots, independent physics/known metrics, public constants, backend import preservation, missing prediction/key, wrong shape/integer/complex/object dtype, NaN/Inf, zero/negative poor results, foreign caller directory, report-write failure and missing hidden assets. Judge paths resolve relative to its own file and malformed inputs have empty metrics. No reference is substituted for missing output.

Notebook validation and visual inspection are recorded in the completion addendum below. Plans were reviewed semantically against the actual normalization, signs, reweighting, stopping checks, units, signatures and caller flow; no proposed algorithm is described as shipped. Public README has every raw key, acquisition metadata, axis/dtype/units, exact output contract and no GitHub links. Hidden arrays are explicitly method outputs; no target or truth leaked into the observation NPZ. Final inventory contains only template assets and required MIT license, with private source/log/capture files outside the task.

## Completion addendum

Notebook executed successfully using `python -m nbconvert --to notebook --execute notebooks/ecoli_calibrated_fluorescence_abundances.ipynb --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=research-bio` with a private JUPYTER_PATH kernelspec explicitly pointing to the shared research Python. Exit0. The canonical notebook retains outputs; portable metadata was restored to python3 afterward without changing cells or outputs. An initial successful run had auto-selected Anaconda's kernel and was not accepted as shared-runtime evidence; an intervening private kernelspec BOM parsing error was fixed before launching the validated kernel. ZMQ emitted Windows selector-thread and local TCP transport warnings; no computational cell error occurred.

Fresh shared-runtime measured seconds: NLS9.645150099997409, SNLS13.160537100047804, SLNLS15.78812239994295, SLPRU64.18764010001905; no-group ablation58.57096529996488. The1800-second per-cell timeout comfortably accommodates complete comparisons without omitting any benchmark pixels. Full SLPRU maximum absolute parity error7.058329615228232e-14, RMSE2.305678883718358e-15. Fresh reference arrays and the identified JSON table are under notebooks/reference_outputs; they are consumed by maintainer review, not by the solver or Judge. Original arrays remain separately named.

All five executed figures were extracted privately and visually inspected: measured ROI/control spectra and overlap; all13 labels for all4 methods on a shared0-1 scale; four residual maps and same-pixel spectra; residual/active-label/iteration-cap comparison bars; and the complete no-group ablation. Labels and color bars are readable, panels show actual nonzero measured/reconstructed information, no method is fabricated, and the large SLPRU cap fraction0.5599489795918368 is conspicuously displayed. SLPRU mean active labels2.079081632653061 versus NLS2.318877551020408 illustrates the prior's effect, without claiming true biological superiority. Every plotted reconstruction was actively computed in the notebook.

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper/revision/data identity and mapping | Passed | Source/archive/hash and traceability sections |
| Original and local exact-bound runs | Passed | Four original MATs, repeat, CLI output, active notebook |
| Frozen parity and independent physics | Passed |21 tests, pre-extraction criteria |
| Constants/units/coordinates | Passed | Native metadata, README and tests |
| Entry point and standalone Judge | Passed | Both CLI exit0; notebook fresh pass |
| Judge poor/malformed/evaluator cases | Passed | Parameterized and independent metric contract tests |
| Public self-contained inputs, no truth/URL leak | Passed | README/raw/environment review |
| Minimal runtime and explicit local math | Passed | Import/call-role review, no domain package |
| Full unit/comparing suite | Passed |21 passed55.65s |
| Active six-section notebook and every figure | Passed | Shared-runtime execution and visual inspection above |
| Provenance/threshold distinction | Passed | Method reference only; original/fresh separated |
| Plans and implementation agreement | Passed | Semantic signature/equation review |
| Template inventory and licenses | Passed | MIT retained; private archive outside task |

Scientific limitations remain processed8-bit intensity, a small author ROI, substantial model residual, no independent abundance truth, and original iteration-cap behavior. These are disclosed limitations of the preserved method and binding, not unverified execution gates. Ready for parent acceptance review.


## Independent parent acceptance

Canonical kernel metadata is portable python3; all executed cells, counts and scientific outputs are preserved from the private shared-runtime execution.

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 6 executed code cells, 5 computed PNG outputs, 4 formulas. Private evidence: research_batch11_20261027/acceptance/ecoli_calibrated_fluorescence_abundances/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
