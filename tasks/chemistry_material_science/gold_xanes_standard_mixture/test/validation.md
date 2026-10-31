# Provenance and validation: gold XANES standard mixture

Status: all mandatory local gates passed; parent acceptance pending. Date2026-10-27. Classification **author_real_processed**: actual instrument XAS, author normalized/energy-aligned ASCII exports, no synthetic benchmark observations. No independent composition truth; native standard-method output is explicitly a baseline reference.

## Binding and novelty
Measured Au-L3 spectrum plus six measured independent chemical standards -> library spectral weights and modeled near-edge spectrum. Distinct from old copper EXAFS radial-pair inference (oscillatory fine structure/scattering paths), unknown-endmember MCR factorization, Raman phenotype classification, and XPS depth profiles. This is known-standard chemical speciation with fixed measured calibration. All327sample measurements and every point of all6standards retained; native fitting selects160points by original interval, not target agreement. No hidden observation/label/truth used for selection or calibration. Full raw context plotted.

## Primary paper and actual acquisition evidence
Lengke, Ravel, Fleet, Wanger, Gordon, Southam, Environmental Science & Technology40(20),6304–6309(2006), DOI https://doi.org/10.1021/es061040r. Original publisher PDF supplied by author repository: private `xanes_source/es061040r.pdf`; text `xanes_paper.txt`. Paper identifies in situ synchrotron XAS at Advanced Photon Source, cyanobacterium Plectonema boryanum UTEX485, gold-chloride exposure at25°C, three starting concentrations. The sample export lacks a unique time/concentration mapping; README does not assign one.

Author Bruce Ravel's primary worked example https://bruceravel.github.io/demeter/aug/examples/aucl.html explicitly links these measured gold/cyanobacteria data to that paper. It describes a Teflon fluid cell, added Au(III)-chloride and repeated overnight XAS scans; gold foil provided alignment. This is real acquisition evidence beyond merely being in an author repository. Source .dat headers identify `cyanobacteria.prj` and the exact Au foil, Au3 Cl aq, Au hydroxide, Au thiocyanide, Au sulphide, Au thiosulphate aq, au_cyano_data exports. Author Larch documentation `doc/fitting_examples.rst` links this exact example to the paper and Ravel's tutorial. https://xraypy.github.io/xraylarch/fitting_examples.html

Gold hydroxide is a spectral proxy: paper p6307 explains that it can compensate pH/hydration differences and is not expected literally in that form. Ravel's tutorial explains that several sulfur-ligand standards have similar fits, so the secure interpretation is a sulfur-containing intermediate rather than a unique named compound. Task, plots and plans preserve these limitations. Measured noise is unavailable: epsilon0.001 is an assumed common objective scale only. No independently verified mass fractions are asserted.

## Original source and license
Author repo https://github.com/xraypy/xraylarch. Run exact remote-verified revision `247278cefbb16eda04c91f79fc1b43df000fa498`; remote API snapshot `xanes_oracle_revision.json` fetched2026-10-27. Available latest remote snapshot `xanes_commit.json` is `8fe9fdada80f11dfa795fc78881d85d95b8bb84c`. All15files in examples/fitting/doc_example3 are byte-identical between current download and the selected archived revision. This is verified identity, not an unsupported freshness claim. Prior source path `../../research_batch_20261020/optics_astronomy/source/xraylarch-247278cefbb16eda04c91f79fc1b43df000fa498`, relative to batch directory's parent maintainer root as indexed below. MIT license copied intact into task; example data carry no separate restrictive notice. PDF is not redistributed in task. Larch method paper: Newville2013, DOI10.1088/1742-6596/430/1/012007.

Exact source ASCII SHA256:
- `s1.dat`: `51a27985fecb99f714c0ca75041a177f69c054b22992b968fcdc6aacd271b477`
- `s2.dat`: `98d78265bb92af3ed1d315a9fe8861600aceec6f70be68cd301c2ec264103248`
- `s3.dat`: `16bc82961a5ae14838578811d4669ccd5ba194c32e00e0019decfc9a8622e29b`
- `s4.dat`: `30fb5ee459b594c7237d016fc10e2a33c21f911e775bd3b9f55e0d52e2277d1b`
- `s5.dat`: `add5ff1759772b949825681b1e499784558174130fcd372548e01ccaa3ace7d6`
- `s6.dat`: `dcdb48d2cf1c85b5512413a6f1b189b0211a6a029542e0205cb59b346408a89f`
- `unknown.dat`: `0322e705fc31c4fc63e6461d27606f893388897a695dd387760cf21f30cd00fb`

## Original execution and independent capture
Private capture script `capture_xanes.py` runs original `ReadData.lar` and all numerical statements of `fit3.lar`, with only syntax compatibility: remove enddef/endif, convert Python2 print to print(), replace nested run by prior execution, skip GUI plotting after all fits. Original Larch `index_of`, `interp`, `param`, `param_group`, `minimize` and native lmfit remain unmodified. GUI omission changes no numerical statement. Copy uses deepcopy, matching independent parameter groups. Original `.lar` Interpreter startup was attempted and failed on unrelated XRD dependency `pymatgen`; this failed path is recorded in `probe_xanes.py` and session output, not claimed successful. Headless original numerical entrypoints are the successful verification path.

Exact binding was built from all ASCII double values as `xanes_raw_binding.npz`, then original read_ascii call was supplied the unchanged arrays from this NPZ as Larch Groups. Thus native preprocessing and inverse run on EXACT final bundled observations, no precomputed fit substituted. CLI command from maintainer: `research-runtime/Scripts/python.exe research_batch10_20261027/material_mechanics/capture_xanes.py`; output `xanes_native.log`, `xanes_oracle.json`, `xanes_oracle.npz`. Exit0, elapsed0.054322s including capture. Native warnings: 'No conda env active, defaulting to base' informational; no failed convergence. Two complete captures repeated unchanged final values. Prior probe numerical capture and corrected first-residual capture preserved same fitted results; initial fixture was corrected because dependent Larch expressions update on first residual, not object construction. This is capture correction, not changed numerical tolerance or target.

Oracle environment: Windows, Python3.11.11, NumPy2.4.6/SciPy1.17.1/lmfit1.3.4, CPU SciPy MINPACK. Private legacy `optics_astronomy/oracle-packages` supplies Larch-only dependencies via capture sys.path; `xanes_extra` contains silx3.1.3/fabio2026.6.0/larixite2026.1.0/gemmi0.7.5 from attempted interpreter support, never imported by task. No shared environment modifications. OMP/OPENBLAS/MKL thread counts1. Clean environment same NumPy/SciPy, Matplotlib3.11.1; `research-runtime` has no domain solver installed. No GPU.

## Frozen tests and discrepancies
`xanes_frozen_contract.md` was written after native capture and before cleaned source. All eleven models are mandatory. Stage tolerances: interpolation1e-13 absolute; initial/residual1e-10; weights1e-5; fitted spectra1e-6; first4statistics1e-3. Final clean weights, spectra and statistics equal native values bit-for-bit on this runtime. A first local BLAS matrix-product implementation gave1.6883e-6 discrepancy in model11: investigated MINPACK boundary-start sensitivity and restored native sequential additions/subtractions, resolving all discrepancies WITHOUT relaxing any threshold. Last dependent-weight expression is clipped to[0,1] natively; corner behavior can violate sum1. Preserved/documented and separately tested; selected result is interior and physically simplex-valid. No known equation error silently copied.

Independent checks: exact affine least-squares selected weights within2e-7; linearity; analytic physical-weight Jacobian finite differences1e-7; increasing energy and full interval support; positivity and sum1. Unit/Judge tests cover missing keys/files, exact shapes, real floating dtype, NaN/Inf, negative weights, finite1e308overflow, invalid/missing reference and boundaries, float32 submission, known RMS offset, bad but valid zero/wrong/shifted outputs, foreign working directory, CLI exit0/1/2, report-write failure. 26tests pass in1.57s. Initial test attempts failed because C:/Users/admin/AppData/Local/Temp/pytest-of-admin access was denied; set TMP/TEMP to private material_mechanics/tmp, then standard tests passed. One earlier traceback renderer access violation occurred while reporting those environmental errors; no solver crash or silent missing test.

## Equation/function/test traceability
| Required stage | Upstream | Local | Independent case |
|---|---|---|---|
| Floor energy bounds and inclusive slice | ReadData.lar, math.utils.index_of | preprocessing.prepare | native exact energies, metadata test |
| Independent-standard linear interpolation | ReadData.lar, math.utils.interp | preprocessing.prepare | native standards1e-13, no extrapolation |
| Weighted absorption sum | fit3.lar sum_standards | physics_model.forward | native all11curves, linearity |
| Standardized residual sign and scale | fit3.lar resid | physics_model.residual | native first residual, analytic derivative |
| Bounds and dependent expression | fit3.lar param_groups, lmfit Parameter | physics_model.bounded_weights | first-evaluation fixture, explicit corner test |
| LM updates and stopping | larch.fitting.minimize, lmfit.minimizer.leastsq | solvers.fit_model | eleven complete native fits, interior affine optimum |
| Eleven model comparisons and ranking | fit3.lar loop | solvers.solve | native statistics and selected index7 |
| Metric/output consistency | independent task contract | evaluation/judge.py | known offset, failures, CLI |

## Fresh result and numerical boundaries
Fresh selected weights equal native reference; max weight error0, spectrumRMSE0, forwardRMSE5.962248466967893e-17, observationRMSE0.003618693081344629, simplexerror0. Boundaries are nonzero:1e-4,1e-5,1e-6,0.0037,1e-6 respectively, frozen before cleaned scoring. Repeated native runs agree; float32 perturbs weights below1e-7. Numerical bounds allow serialization/backend roundoff but exclude even the nearest three-standard alternative; measurement bound is only~2.25%above native residual and rejects poor fits. These are method-reference tolerances, not physical accuracy confidence bounds. Standard NCC/NRMSE are unsuitable for short sparse weight vectors and omitted; exact RMS formulas are documented. Judge imports no solver, reloads public calibration independently and uses no alignment/crop/rescaling.

Historical paper and tutorial numbers are context only, not task results. Native captures and fresh cleaned notebook measurements are separately labeled. Fresh canonical notebook is not cache mode; all11fits, 3shifted-library reconstructions and affine6standard diagnostic computed. Full seven-figure review passed: raw spectrum, library features, multi-model fits/residuals/components/native difference, paired11model bars, identifiability, calibration sensitivity, unconstrained diagnostic. Two plots were improved after inspection: true nonuniform energy in pcolormesh and annotation margins. Canonical re-executed and all figures inspected; no clipping prevents reading. Notebook math uses$$; README usesmathfences. No fakezero comparison: zero parity is actually measured.

## Execution commands and review
From task root with clean runtime and OMP/OPENBLAS/MKL=1:
`python -m src.main`; `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`; `python -m pytest test/ -q --tb=short` (TMP/TEMP private writable directory). All exit0.
Notebook: `python -m nbconvert --to notebook --execute notebooks/gold_xanes_standard_mixture.ipynb --output gold_xanes_executed.ipynb --output-dir output --ExecutePreprocessor.kernel_name=research-clean --ExecutePreprocessor.timeout=600`. JUPYTER_PATH private material_mechanics/jupyter; kernel points clean research-runtime. Exit0, full runtime~4s including kernel startup. Windows selector-thread warning from zmq, unrelated to numerical output. Canonical output retained, temporary executed duplicate removed. `xanes_notebook_stream.txt` records actual solver timer and metrics. Signature/default/shape/caller-flow review passed against both plans. No matplotlib.use in libraries. Imports and call roles reviewed: NumPy linear algebra/interpolation, SciPy general leastsq, Matplotlib only; no task/domain solver, upstream imports, hardcoded outputs or cross-task imports.

## Final inventory and archive index
All paths below are maintainer-only except packaged allowlist. Private base: `research_batch10_20261027/material_mechanics/`. Archives: `xanes_source/`15files including originalPDF; `xanes_source_listing.json`; `xanes_commit.json`; `xanes_oracle_revision.json`; `xanes_paper.txt`; `probe_xanes.py`; `capture_xanes.py`; `xanes_native.log`; `xanes_oracle.npz/json`; `xanes_raw_binding.npz`; `xanes_frozen_contract.md`; `xanes_notebook_stream.txt`; `xanes_figures/1.png` through7.png. Source/dependency reuse is maintainer `research_batch_20261020/optics_astronomy/source/xraylarch-247278cefbb16eda04c91f79fc1b43df000fa498` and `oracle-packages/`; none copied/imported into task. Build/doc/notebook/finalize scripts remain private. Task fixtures native.npz consumed by comparing_test; upstream_comparisons.npz consumed by notebookplots; baseline_reference consumed byJudge; fresh output consumed byJudge. No unused NPZ, PDFs or source archives in task.

| Array asset | Bytes | SHA256 |
|---|---|---|
| data/raw_data.npz | 37956 | `3f2484b18e89c9840712f7eecadd01c629c86ac52b33af05701279c79e85dec3` |
| evaluation/baseline_reference.npz | 1856 | `377fedea286781ec4ea1389b092f74fdb3cfc357cbb574c50c911a234cddb495` |
| notebooks/reference_outputs/upstream_comparisons.npz | 15896 | `cd24f66aa86342dda6160fc7d3a62f380c6e6a870de897969e83e8c6f8fc4c97` |
| output/reconstruction.npz | 1856 | `377fedea286781ec4ea1389b092f74fdb3cfc357cbb574c50c911a234cddb495` |
| test/fixtures/native.npz | 27660 | `b9139f9f58ec61b0fbfb561fd033a84a5a1441b2cb5a33cf94eed1cef7d949c1` |

## Mandatory gates
PASS paper/revision/data hashes and traceability; PASS native exact-bound run and clean full run; PASS independent stage parity and physics; PASS constants/units/precision; PASS CLI/Judge; PASS malformed and poor-output handling; PASS public problem sufficient; PASS no truth/calibration/GitHub leakage; PASS metadata/output contract; PASS shared runtime minimal dependencies; PASS explicit math/call-role audit; PASS26tests; PASS active six-section notebook and seven visual reviews; PASS provenance/metric consistency; PASS plans/source semantic review; PASS template/license/inventory cleanup. Parent integration acceptance remains external to these local gates.

Parent-requested exact float32/64 validation added; float16 rejection tested. Calibration annotations now face inward and notebook re-executed.


## Independent parent acceptance

CLI, standalone Judge, pytest, active nbconvert and dual-engine math rendering were independently rerun: all exit0. Parent notebook: 5 executed code cells, 7 computed PNG outputs, 18 formulas. Private evidence: research_batch10_20261027/acceptance/gold_xanes_standard_mixture/execution.json and semantic_review.json. All panels actually inspected by parent; source/physics/GT roles and limitations semantically checked. Canonical notebook is this parent-executed artifact. No new source or notebook execution is inferred from cached upstream outputs.
