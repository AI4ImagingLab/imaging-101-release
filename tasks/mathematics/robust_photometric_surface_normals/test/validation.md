# Provenance and validation: robust photometric surface normals

Final local gates passed; independent parent inspection is recorded below/by the batch audit. The target is a dense signed normal field recovered from indirect lighting response. Data are **existing author renderings**, not real photography and not newly generated synthetic observations. No ground-truth normal or reference reconstruction is part of solver input.

## Paper, exact original source and licensing

Paper: Satoshi Ikehata, David P. Wipf, Yasuyuki Matsushita and Kiyoharu Aizawa, Robust Photometric Stereo using Sparse Regression, CVPR2012,318–325, DOI https://doi.org/10.1109/CVPR.2012.6247691 . The actual8-page author paper was downloaded from https://satoshi-ikehata.github.io/SBLPS_cvpr2012.pdf , archived as SBLPS_cvpr2012.pdf, SHA256 `f01684b3df182539a055cb9abb692c683ab2a4dba3e4af752bc4ce0deb330219`; full text extracted and model/SBL/L1 sections read. The university primary metadata https://hub.hku.hk/handle/10722/371141 independently confirms authors/title/pages/DOI. Low-rank comparator paper: Wu et al., ACCV2010,703–717, DOI https://doi.org/10.1007/978-3-642-19318-7_55 ; proceedings publication2011 is distinct from conference year2010. The source README's unrelated later PAMI issue/pages entry is not copied as the selected citation.

Original author code/data: https://github.com/yasumat/RobustPhotometricStereo , exact commit `f03aa95b57e746a7d31df76b1c0fa0a83584a3c1`. Actual remote git ls-remote HEAD was verified against the clone before extraction, recorded in root/source_inspection.json. Original source hashes are in original_run.json and the archive index. Selected functions are rps.RPS._solve_l1, _solve_sbl, _solve_l2 and _solve_rpca plus their small explicit numerical routines. This is code authored by the paper coauthor, not a third-party domain-wrapper substitute.

The actual source README states "This package is distributed under the GNU General Public License." It specifies no GPL version and no separate LICENSE file. The adaptation retains that unversioned grant and author/support attribution in NOTICE, chooses GPLversion3 for this adaptation, and includes the full standard GNU GPL3text in LICENSE; it does not falsely claim an upstream explicitly versioned GPL3file. The same source distributes the existing bunny images/lights/foreground mask/renderer normals with no separate restrictive asset notice. Complete source notice and source/data hashes are privately archived. Source capture/decode-only OpenCV/scikit-learn dependencies are isolated from the cleaned runtime.

## Bound data and output, fixed before implementation

The existing author directory data/bunny/bunny_specular contains50native256×256×3rendered RGB arrays. Original RPS.load_npyimages performs the arithmetic RGB mean; original load_lightnpy loads lights.npy with internal shape3×50; original load_mask reads author mask.png. Preserve all50lighting conditions, the full native spatial grid, intensity scale and Cartesian basis. No image crop, resize, tone map, gamma, hidden normal-driven calibration, new renderer scene or random seed is supplied. Public support is the author's separately provided silhouette/foreground ROI, not a function of hidden normals. Antialiased image fringe can contain nonzero samples outside the author ROI (919pixels differ from the any-nonzero-intensity mask); therefore it is not falsely described as thresholding all nonzero measurements. The ROI fixes the requested normal domain and the zero-background output convention.

Public raw NPZ SHA256 `097fbcda9cfa9d1b752f43e0cad7398c41564ba718562599f723e29c211b3fb1`. Arrays: measurements(1,50,256,256)float64, light_directions(1,50,3)float64, support(1,256,256)bool. Preprocessing reconstructs exactly the independently captured pixel-by-light intensity matrix and3×50light matrix; their byte SHA256 values and foreground count20317are frozen in test/fixtures/upstream_preprocessing.json. The hidden actual truth is the author's independent gt_normal.npy, repackaged without renormalizing into evaluation/ground_truth.npz, SHA256 `626f312856805c415a48aaea4ec763d5784a241e2fdce1be040519a7f5e2e34c`. Foreground norm deviation is at most2.22e−16. It is genuine synthetic truth, never a reconstruction renamed truth.

The output is float64(1,256,256,3), Cartesian x,y,z normals in the calibrated light basis, unit foreground and exactly zero outside the supplied support. No height-integration, metric depth, sign flip or orientation alignment is part of the contract. Absolute image pitch is unavailable and unnecessary for this orientation inverse.

## Oracle-first run and method-scope review

Command: root/oracle-env/Scripts/python.exe root/photometric/capture_original.py, cwd private root/photometric; the independent repeat adds argument independent_repeat. The script invokes unchanged original RPS methods and rpsnumerics routines on the exact public observations; no cleaned source exists when fixtures/criteria are frozen. Original total exit0 runtime `120.945481800s`, independent repeat exit0 `108.849210800s`. All frozen input/output arrays, including all4full normal fields, compare bitwise equal between the two captures. Original individual timings and foreground/all-pixel statistics are retained separately in original_run.json; partial checkpoint archives are privately indexed as historical capture outputs, not shipped task artifacts.

Original environment: Windows build26200, Python3.11.11, NumPy2.4.6, SciPy1.17.1, scikit-learn1.9.1; OpenCV5.0.0.93 serves only the original PNG mask loader. The first original import failed because cv2 was absent; it was installed in the private oracle only and the complete original method was then actually run. This is not falsely reported as an unmodified-runtime success before dependency resolution. Single-thread OMP/OPENBLAS/MKL settings and no randomness apply. CPU only. Cleaned runtime is shared Python3.11.11/NumPy2.4.6/Matplotlib3.11.1; exact numerical backend is recorded in cleaned_environment.json. No shared dependency was installed or upgraded.

Paper equations5–7give sparse-error regression; its convex L1alternative and SBL comparison are both scientifically relevant. Reading paper equations10–13and the actual code revealed the source-SBL mean normal matrix uses1e−6I while its variance update matrix usesI. Those cannot be presented as one identical Gaussian posterior covariance. **Before local implementation**, select the published L1IRLS alternative as the standard task method, retain all four original algorithms as active comparisons, explicitly label source-SBL as the preserved two-penalty heuristic, and freeze new L1truth-based acceptance values. initial_sbl_criteria.json preserves the earlier provisional scope; preimplementation_criteria.json and fix_photometric_method_scope.py document this pre-extraction decision. No failed local numerical parity or need for weaker error thresholds caused the selection. The source heuristic is not silently fixed or claimed to be the exact paper posterior; there is no monotonic SBL evidence claim.

Independent original Python function tracing on actual pixels8831,34686,53637captured every pre-stopping L1x/W/residual and SBLx/gamma/C/d. L1trace counts241/153/24, SBL85/19/24. The private freeze_stages.py runs unchanged original numerical functions, before local code; source-generated arrays are not cleaned-function fixtures. Relative/absolute parity tolerances1e−9/1e−10were fixed in advance. L1baseline foreground NCC`0.9959952770373275`, DR-NRMSE`0.02619656488364513`, mean angle4.597762735142496degrees. All four results remain in the comparison.

## Equation / source / local / validation mapping

| Required mathematics | Original file/function | Local function | Validation case |
|---|---|---|---|
| Lambertian lighting and sparse additive departures, paper equations1–7 | rps.py:RPS methods with L.T | physics_model.forward, adjoint | independent dot-product adjoint and known exact diffuse normals |
| Original arithmetic mean/layout and foreground domain | RPS.load_npyimages/load_mask/load_lightnpy | preprocessing.load_observations, measurement_matrix | independent original matrix/light byte hashes and source mask count |
| L1IRLS weighted linear fit, reciprocal-residual weights | rpsnumerics.L1_residual_min | solvers.l1_coefficients | full original iteration traces at3actual pixels; convex loss decrease and dimensionless stationarity |
| Albedo removal/unit normal/background | RPS._solve_l1; sklearn normalize | physics_model.normalize_rows; reconstruct | negligible-row and exact known normal cases; full pipeline allpixels |
| Ordinary L2comparison | RPS._solve_l2 | reconstruct(method='l2') | full independent original normal map |
| Identified two-penalty source-SBL comparator | rpsnumerics.sparse_bayesian_learning | source_sbl_coefficients | every original coefficient/gamma/C/d at3pixels; full independent original map |
| Nuclear/L1 split, sparse-first ALM, then light regression | rpca_inexact_alm; RPS._solve_rpca | low_rank_sparse; reconstruct(method='rpca') | independent constraint residual case and full original map |
| Signed angular/unit and component metrics | independent benchmark definitions | standalone Judge; separate visualization.metrics | analytically orthogonal normal pair, known background error, undefined reference rejection |

Extracted numerical routines are small local readable functions containing explicit weighted solves, reweighting, gamma updates, SVD thresholds, primal/dual updates and stop checks. No domain package or copied opaque framework is called. NumPy least squares/SVD/linear solve are allowed computation primitives. Core forward intensity equation, adjoint and all inverse updates are executable local mathematics. The Judge imports no solver.

## Actual fresh results and acceptance

| Method | Original foreground mean angle [deg] | Fresh mean angle [deg] | Max component difference vs original | Fresh measured solver seconds |
|---|---|---|---|---|
| l2 | 18.4703820190205 | 18.4703820190205 | 0 | 0.027045500 |
| l1 | 4.5977627351425 | 4.59776273514244 | 2.22044604925031e-16 | 98.105478200 |
| sbl | 3.97288183856026 | 3.97288183856018 | 2.22044604925031e-16 | 47.963117900 |
| rpca | 3.38375955930513 | 3.38375955930503 | 0 | 3.695056800 |

These are actual measurements on the same public arrays; original historical all-image averages (which include zero background) are retained in private metadata but are not relabeled as foreground accuracy. Fresh machine precision is preserved in notebooks/reference_outputs/fresh_measurements.json and private final_figures/experiment_metrics.json. Timings were observed under concurrent maintainer activity and are illustrative execution costs, not a controlled speedup claim.

L1joint Judge gates fixed before implementation: NCC≥0.8963957493335948; DR-NRMSE≤0.02881622137200965; foreground mean angular error≤5.057539008656746degrees; maximum unit-norm/background deviation≤1e−8. These follow0.9baseline NCC/1.1baseline error with an additional strict angular gate. The cosine threshold alone is not treated as sufficient. Current fresh CLI passes all metrics against actual synthetic truth. A zero foreground vector incurs180degreeangular error and fails; zero backgrounds do not dilute accuracy. There is no fitting, cropping, hidden sign correction or magnitude-only conversion.

CLI `python -m src.main`: exit0,99.215199800s. Standalone Judge: exit0,0.148577800s. Parent full current suite `python -m pytest test/ -v -p no:cacheprovider --basetemp <private>`: **22passed**, exit0,149.626033800s (robust_photometric_surface_normals-parent-tests.log/json). It includes all4fresh full-method reconstructions against independent original outputs. Latest complete active notebook `python -m nbconvert --to notebook --execute notebooks/robust_photometric_surface_normals.ipynb --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=imaging101-research --output robust_photometric_surface_normals_executed.ipynb --output-dir <private>`: exit0,408.106826900s; every code cell executed and all5figure groups are embedded. Explicit private kernel points at shared Python, inline backend set in execution environment; no library backend mutation. Temporary executed notebook stays private, not task/output. Exact commands/cwd/logs/runtimes are in execution_records.json and parent test records.

Independent mathematical tests check light-operator adjointness, known noise-free normal recovery, unit normalization, ALM equality residual, and all frozen original trajectories. An initial stationarity test used an absolute normal-gradient bound and failed at1.0598e−5because squared IRLS weights can reach1e16. This was investigated rather than changing solver or parity tolerance. Final test uses dimensionless least-squares backward error ||B^T(Bx−d)||/[||B||(||Bx||+||d||)]<1e−12, B=WA,d=Wb, and separately verifies actual L1loss decreases. Independent original trajectories remain exact. The malformed1e300submission stress deliberately overflows an intermediate norm; the Judge returns evaluator/input error2with empty metrics and finite JSON, not NaN/Infinity. Normal runs are finite.

## Notebook experiments, figures and review

The notebook actively computes full L2,L1,SBL,RPCA and deterministic10/25light L1subsets. The latter produce actual mean angular errors5.023536449115831and4.5835937942447975degrees, compared with4.597762735142442for50lights. A larger subset is not falsely forced to be monotonically better on this finite corrupted scene. Actual3pixel diffuse-fit predictions and IRLS loss trajectories use local coefficients, never hidden truth. Timeout1800s covers the measured~408s full scientific review path. The10light IRLS run can take longer than50lights because stopping/conditioning differ; there is no fabricated cost scaling assumption.

Five actual groups:6observed lighting images; renderer truth+all4normal/error maps; paired original/fresh angular bars+actual runtimes+error distributions+numerical differences; actual pixel diffuse fits+real convex loss; and10/25/50light field sensitivity. All5were opened and reviewed for real content, scientific labels, common scales, truthful source names, readable legends and no overlap. After initial visual review, observed images use a common square-root display norm to reveal diffuse detail beneath bright specularities; the physical colour-bar labels retain original intensity units. No observation or metric is gamma-transformed. Final figure1and updated runtime panel were reopened after full rerun. Figures2–5preserve their verified structure/arrays. All5PNG/SVG/PDFexports and exact figure_source.npz are private. No historical paper table is filled with made-up fresh values.

## Fixture/reference consumers and archive

Public observations are the only solver data. Hidden evaluation/ground_truth.npz contains actual renderer normals, consumed by Judge and accuracy review. notebooks/reference_outputs/original_methods.npz contains separately captured original4normal maps, consumed by notebook and full-pipeline parity test. It is not a duplicate labeled truth. upstream_measurements.json and fresh_measurements.json are identified numeric provenance records for maintainer review. test/fixtures/upstream_stages.npz contains only consumed original3pixel traces and inputs; upstream_preprocessing.json stores capture-time byte hashes/counts rather than duplicate full observations. No consumerless fixture, private PDF, PNG, upstream checkout, original log or capture script is packaged.

Private root `D:/yan1/agent/AutoSkill/imaging-101-maintainer/research_batch4_20261021/root/photometric` contains the complete original repository, author paper, original arrays, original/independent repeat logs, freeze/capture scripts, original source/data hashes, environment report, final figure sources and historical attempts. archive_index.json indexes byte lengths and SHA256. Original sourcefiles and all50array hashes are also in original_run.json. Historical builder scripts predate the final visual/test review; actual shipped code and final executed notebook are authoritative, and source hashes in fresh_measurements identify the final solver. No shared-runtime changes or cross-task runtime imports occurred.

## Semantic plan and contract review

Reviewed both plans against actual signatures, defaults, shape/dtype/units, caller/data flow and every update. Light transpose, pixel-major ordering, positive-albedo unit normalization, no clamp in the diffuse linear model, sparse-first RPCA order, original absolute stop condition, zero background, exact sign basis and foreground-only metrics agree. The SBL mean/covariance caveat is consistent across code, notebook and plans. CLI has no hidden references; notebook labels its independent originals and actual truth; Judge independently loads all assets relative to itself and uses caller-relative prediction/report paths. `src.main` import preserves the current Matplotlib backend. Public README follows all5required sections, batch-first data tables, acquisition metadata and full output contract, with no public GitHub URLs or standalone metadata file.

## Mandatory gates

| Gate | Status | Evidence |
|---|---|---|
| Exact verified source, paper version/data hashes and mapping | passed | Source/paper/data/archive sections above |
| Original and local exact-observation runs | passed | Independent unchanged original twice, fresh CLI and full active notebook |
| Independent stage/full parity and mathematical checks | passed | Frozen original traces/maps, adjoint/noiseless/convex/constraint checks |
| Constants/units/precision/coordinates | passed | Capture hashes, explicit source numerical order and signed normal basis |
| CLI and strict Judge | passed | Current output passes actual renderer truth |
| Judge malformed/poor/evaluator/reference/report cases | passed | Analytic metrics,9parameterized contracts, precedence and foreign cwd tests |
| Public inputs suffice and no privileged normal/calibration/URL leak | passed | Visible light/intensity/supplied silhouette domain only; no truth used for calibration |
| Minimal validated environment and explicit local mathematics | passed | NumPy/Matplotlib subset, source-call audit, no domain/cross-task imports |
| Full current tests | passed |22tests in parent full-suite record |
| Active complete notebook and preserved comparisons | passed | Full408s reconstruction, all4methods+subsets,5embedded groups |
| Labeled result provenance and justified joint metrics | passed | Actual truth, independent original, fresh identified separately |
| Plan/source semantic consistency | passed | Explicit review above and public-metadata test |
| Final inventory, notices, archive and cleanup | passed | Parent source/inventory audit plus archive index |


## Parent acceptance review

Parent directly executed unchanged author code twice on exact bound observations, froze all4 full methods and actual pixel iteration traces before extraction, read original8page paper, and ran final22test full suite (22 passed in149.28s). Fresh CLI/strict truth Judge/foreign-cwd Judge0, full408s active notebook and all5 scientific groups reviewed. Standard L1 method selected before extraction after identifying source-SBL posterior-convention discrepancy; all4 original algorithms preserved as actual comparisons with SBL labeled source heuristic. Genuine author-existing synthetic truth; no photographic-data claim. Final README indentation/formula and notebook rho text repaired after static review without changing executed numerical cells/results. Final source/inventory audit errors[].
