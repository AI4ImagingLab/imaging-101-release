# Provenance and validation evidence

## Frozen scientific scope

Real QDM rock-section upward-field acquisition → full published real-data dipole interpretation: continuation, gradients, multiscale detection, Euler centres, local vector moments, and above-surface rejection. Output is all 75 buried-source positions and signed moment vectors, not only the 46 quality-filtered sources. No independent physical truth is available. This task does not claim that the dipole approximation is accurate for every grain.

## Sources, revision, licensing and archive

Source repository: https://github.com/compgeolab/micromag-euler-dipole . `git ls-remote ... HEAD` verified remote HEAD **fea3f8fb01a3017e6055049be1b70ff083b0bf81** on 2026-10-20; the complete private checkout resolves to the same commit. This is an actually network-verified revision, not an assumed local tracking revision.

Paper: Souza-Junior et al., *Full vector inversion of magnetic microscopy images using Euler deconvolution as a priori information*, DOI **10.1029/2023GC011082**. The exact repository manuscript `paper/content.tex` at that commit was read and archived. The preprint is also available at https://eartharxiv.org/repository/object/5452/download/10758/ . Reproduction uses the repository-associated manuscript and code, not a claim of bitwise reproduction of every version of the journal article.

Code MIT; manuscript/figures/data CC BY 4.0, explicitly declared in upstream README and LICENSE files. Original source copyright and MIT license retained in local LICENSE/NOTICE. LoG peak/pruning conventions follow scikit-image BSD-3-Clause; its attribution and terms appear in NOTICE. No source paper PDFs, upstream checkouts, capture scripts or logs are in the final task.

Private archive base: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch_20261020/earth_mechanics/`.

| Archive item | Consumer / purpose |
| --- | --- |
| micromag-upstream/ | Complete immutable source checkout, original MATLAB acquisition, paper TeX, licenses, original real-data notebook. |
| mag-select/ | Earlier selective downloads at the same revision; retained private download evidence only. |
| oracle-env/ | Isolated upstream environment; never imported by cleaned code/tests/notebook/Judge. |
| capture_mag.py | Calls unmodified upstream functions on the exact full acquisition and freezes numeric outputs. |
| mag_oracle.npz | Full independent upstream numeric capture; first capture is frozen as test/fixtures/upstream.npz. |
| mag_oracle.log, mag_oracle_repeat.log, mag_oracle_warnings.log | First run and repeated-run stdout/warnings. |
| oracle-install.log, oracle_numerical_backend.log | Oracle dependency/backend evidence. |
| cleaned_numerical_backend.log, mag_tests.log, mag_notebook.log | Cleaned backend, tests, active nbconvert log. |
| mag_fresh_evidence.json | Full-precision fresh timing, metric/parity measurements, environment and per-file SHA-256 inventory. |
| mag_executed_review.ipynb, mag_review_0.png .. mag_review_3.png | Executed review and inspected image extracts, private retained evidence. |
| finish_mag.py, local_mag_review.py, validate_mag.py | Maintainer artifact-authoring/evidence scripts, no runtime imports. |
| jupyter/kernels/research/kernel.json | Private kernelspec pointing to the shared validated runtime. |

Original `data/Bz_uc0.mat` SHA-256:
`2f2fbb4dfb645e27af09c76199ebcdd42c0fa137588072d3748eaf3be61fa0a6`.
Bundled `data/raw_data.npz` SHA-256: `7db775b6d245e85168fce64f25a9421bd2205f002e6c7a4992d940f9c24d5f2e`.
The NPZ acquisition preserves every value of upstream Bz×1e9 and the exact arange-based micrometre axes. No downcasting, resampling, spatial subsetting, or synthetic data generation occurred. All bundled observation arrays and reference arrays are NPZ. Full packaged-file hashes are indexed in private `mag_fresh_evidence.json`; direct per-array equality to the frozen upstream capture is tested.

## Independent upstream run and acceptance

Command: `oracle-env/Scripts/python.exe capture_mag.py`, archive directory above. The capture imports the original `code/micromag.py` from the exact checkout; no upstream source modification or cleaned solver import. The procedure duplicates the executable scientific cells of `code/03_real_data.ipynb` on all 576,000 bound measurements. Seed: not applicable; no stochastic operations. First capture compute time 3.0351353001315147 s; explicit verified repeat exit **0**, 2.9148958998266608 s. First PowerShell merged-stderr invocation itself reported native-warning error status despite completed Python output; the explicit separated stdout/stderr repeat resolves the execution gate. All frozen arrays compare exactly between independent oracle reruns (max absolute difference 0.0).

Python 3.11.11, Windows 10 build 26200, CPU single-thread environment NUMBA_NUM_THREADS=OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1. Oracle NumPy 2.4.6, SciPy 1.17.1, Harmonica 0.6.0, Verde 1.9.0, Choclo 0.0.1, scikit-image 0.26.0, xarray 2026.9.0. Backend details saved separately. This is a validated modern Python port of the supplied method, not the original Python 3.10 environment. Warnings are deprecated xarray `drop` and xrft inverse-lag defaults; no invalid numerical values occurred. 100 windows detected, 75 buried-source solutions, 46 satisfying R2>=.85 and SNR>5 dB, matching the manuscript's reported group sizes.

Acceptance is upstream-stage/pipeline parity plus independent physics, not Judge self-comparison. Deterministic stage tolerance is rtol=1e-9 and 5e-9 nT/nT-per-µm absolute floor for FFT-related fields. Derived parameters use scale-aware 1e-9*max|reference|; no unit-insensitive floor for tiny moments/covariances. Windows must agree elementwise. Separate evaluator tolerances 1e-7 µm position RMSE and 1e-7 relative moment L2 are conservative nonzero bounds above the observed repeat/float64 transform errors. They measure reproducibility against a standard result; thresholds do not claim physical accuracy. Preliminary local comparisons were diagnostic; final reference acceptance retained the upstream algorithm and constants rather than weakening thresholds after numerical failures.

## Equation / algorithm traceability

| Paper stage / mathematical requirement | Upstream file/function | Local implementation | Independent validation |
| --- | --- | --- | --- |
| QDM SI→nT/µm coordinates | micromag.load_qdm | preprocessing.load_data; frozen NPZ conversion | unchanged Bz/axes exact equality; metadata test |
| Fourier potential-field continuation exp(-h|k|) | hm.upward_continuation used in notebook | preprocessing.upward | full upstream continued field; analytic Fourier harmonic attenuation and DC preservation |
| Cartesian gradients/TGA | micromag.data_gradients | preprocessing.prepare | fx/fy/fz/TGA/stretch upstream arrays; analytic Euler homogeneity |
| LoG anomaly windows | micromag.detect_anomalies / skimage.feature.blob_log | preprocessing.detect_windows | all 100 windows exact equality |
| Euler homogeneity SI=3 | micromag.euler_deconvolution | solvers.euler | all upstream centre/background outputs; independent finite-difference dipole Euler check |
| Dipole tensor / magnetic coefficient | micromag._dipole_jacobian_fast / Choclo kernels | physics_model.dipole_jacobian | analytic on-axis Bz, sign/units and superposition; vector-fit parity |
| Full-vector least squares and covariance | micromag.dipole_moment_inversion | solvers.invert_moment | all upstream moments/covariances/scores; analytic noiseless moment recovery |
| Position filtering and full pipeline | code/03_real_data.ipynb inversion loop | solvers.reconstruct, main.main | 100/75/46 counts; all-output parity and CLI smoke |
| Signed direction, filtered/all scientific comparisons | upstream real-data notebook | visualization.angles/review; active notebook | upstream/fresh overlay; numeric group counts; inspected executed maps and local windows |
| Position/moment scoring | explicit benchmark definition | independent evaluation/judge.py | known metric/permutation examples and malformed/poor submission tests |

## Discrepancies investigated and preserved conventions

1. Upstream vertical derivative comment claims forward difference, but code uses central upward/downward continuation. Local code preserves the executable central formula, independently checked against analytic Euler homogeneity.
2. Paper says SNR=5 means signal five times noise. Actual code reports **10log10(variance ratio)** in dB, so 5 means ratio sqrt(10), not 5. Local code and plots retain the executable dB definition. No relabelling of ratio metrics occurred.
3. Initial local use of nominal mu0/(4π)=1e-7 gave a covariance discrepancy just outside 1e-9 relative acceptance. Investigation identified Choclo's exact constant **mu0=1.25663706212e-6 H/m**, a relative 5.443757e-10 difference in the coefficient. The exact supplied constant is now implemented and documented; no tolerance was relaxed. Max covariance parity difference decreased from 1.2921e-35 to 7.131600248519005e-40 A²m⁴.
4. The supplied dataset includes very strong grains. Overview field plots use the upstream ±10000 nT display limits to reveal weak sources; no numerical clipping is applied to reconstruction. All-source residuals intentionally omit fitted local backgrounds, matching the source notebook. Their visible residual is a model limitation, not hidden numerical success evidence.
5. Direction scatter replaces the upstream stereographic projection, explicitly explained in the notebook. Both all/filtered groups, moment-amplitude colour, signed directions, histograms, field maps, source boxes/depths, paired count bars, and first-five local observed/predicted/residual comparisons remain. No fabricated truth or missing-method zeros.

## Cleaned execution and verification

Shared runtime: `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research-runtime/Scripts/python.exe`; Python 3.11.11; NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1. Only these three runtime packages are imported; no upstream domain packages are installed into or required by the cleaned runtime. CPU fresh complete reconstruction **1.1576945001725107 s**; no GPU, largest transform grids <100 MB, no large-memory exception. No new synthetic observations, simulator seed or target-derived calibration.

Commands from task directory:

* `python -m src.main`: exit **0**, writes (1,75,3) float64 positions and moments.
* `python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json`: exit **0**, status pass; position RMSE **8.778175159062668e-13 µm**, moment relative L2 **4.508011256644531e-14**.
* `python -m pytest test/ -v`: final exit **0**, **19 passed in 4.62 s** (includes signed-direction known cases). TEMP/TMP set to private writable archive temp due to pre-existing access denial at system pytest temp directory; the first run recorded that environmental error and the exact-constant discrepancy above.
* `python -m nbconvert --to notebook --execute notebooks/magnetic_microscopy_dipole_inversion.ipynb --output magnetic_microscopy_dipole_inversion_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180 --ExecutePreprocessor.kernel_name=research`: exit **0**, private JUPYTER_PATH research kernelspec explicitly selects shared runtime. An initial generic `python -m jupyter nbconvert` dispatched the global Anaconda executable/kernel; it was rerun correctly using nbconvert's module and explicit kernel. Timeout 180 s allows ample headroom over measured ≈1.2 s computation plus plotting/kernel startup. Final executed outputs embedded in the maintained notebook; temporary output notebook removed and execution evidence archived privately.

Fresh maximum parity errors: positions 5.229594535194337e-12 µm; moments 1.4152543292899458e-24 A m²; backgrounds 2.433608869978343e-11 nT; R2/SNR 5.5848659030743875e-12; windows exactly equal; covariance 7.131600248519005e-40 A²m⁴. Full preprocessing comparison passed. Judge tests include missing files/keys, wrong shape/float32/complex dtype, NaN/Inf, zero-moment poor output, unavailable evaluator assets, non-task cwd, report-write error, known metric/permutation cases. No undefined NCC/NRMSE is used for this source-parameter problem.

Visual review: inspected extracted `mag_review_0.png`, `mag_review_1.png`, `mag_review_3.png`; all science panels have readable axes, colour scales and labels; detection boxes, local dipole residual structure and both direction groups visible. Paired bars show upstream/fresh 75 and 46 counts. No library changes the Matplotlib backend. Method/plan semantic review checked exact permeability, SI conversion, FFT signs/DC, central derivative, LoG scales/border/pruning, Euler sign/factors, normal-equation fit, covariance N-3, dB SNR, all/filtered flows and signatures. Runtime, notebook and Judge do not import upstream code.

## Completion gates

| Gate | State | Evidence |
| --- | --- | --- |
| Paper, verified revision, data hashes, equation mapping | passed | Source/archive and traceability sections |
| Original and cleaned full bound-data runs | passed | Explicit exit-0 repeat, CLI and active notebook |
| Independent parity and physical tests | passed | Full fixture pipeline plus harmonic/dipole/Euler analytic checks |
| Constants/units/precision/coordinates | passed | Exact original arrays and corrected exact permeability |
| Standard NPZ CLI/Judge quality | passed | Output and report, both required quantities scored |
| Malformed/poor/evaluator failure contracts | passed | 18-test suite |
| Solver allowlist self-containment / leakage audit | passed | README has all acquisition metadata and output contract; only Bz/x/y public; no GitHub URLs in README/requirements |
| Shared dependencies and domain-symbol audit | passed | Local NumPy/SciPy mathematics; no domain-library/cross-task imports anywhere in task |
| Active notebook and scientific comparisons | passed | Executed full reconstruction and visual review above |
| Result provenance and thresholds | passed | Historical paper counts distinguished from reproduced/fresh arrays; nonzero reference tolerances |
| README and plans semantic/signature review | passed | Reviewed actual functions and mathematical conventions above |
| Inventory/licenses/archive cleanup | passed | Required template files, LICENSE/NOTICE; no temporary executed notebook, upstream source or separate metadata file |

No mandatory gate remains failed or unverified. The physical accuracy limitations of real-data dipole interpretation remain explicitly stated and are not replaced by reference agreement.
# Independent parent acceptance, 2026-10-20

The coordinating parent reran the complete task suite in the shared research runtime: **19 passed in4.85s**, exit0. A new standalone Judge invocation from the private batch archive working directory passed the actual actively reconstructed output. Parent AST/import/call-role, README/public-array, executed-notebook and file-inventory checks found no errors. Required LICENSE and NOTICE texts are retained under AGENTS.md. The parent actually viewed all four executed scientific figures: complete acquired/continued/gradient/window/dipole/residual maps; all-versus-quality-selected direction/depth/score groups and original/fresh overlay; paired count bars; and all five selected local observation/prediction/residual comparisons. The coordinate/units/constants, explicit local detection/Euler/dipole physics, fixed method-reference scope and residual/model limitations were reviewed. No physical truth is claimed from numerical agreement. Evidence: private `<batch>/magnetic_microscopy_dipole_inversion-parent-audit.json` and `magnetic_microscopy_dipole_inversion-fresh-independent-judge.json`. This accepts the selected real-data method-reproduction task without widening thresholds.



## Public-contract scientific isolation 2026-10-29

The method-selected source catalog is not identified by the public dipole forward model alone. Benchmark eligible=false. Scientific review is completed; blind-agent execution is a separate unexecuted gate and is not the reason for isolation. Removed README recipe, where present, is archived privately under benchmark_readiness_20261029/private_readme_recipes. Existing arrays/Judge thresholds unchanged.
