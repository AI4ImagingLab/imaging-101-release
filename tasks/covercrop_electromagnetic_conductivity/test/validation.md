# Validation and provenance

## Bound task and source

Published method: McLachlan et al., EMagPy, Computers & Geosciences146 (2021)104561, DOI10.1016/j.cageo.2020.104561. Cumulative sensitivity model: McNeill1980 TN6. Exact author repository https://github.com/hkexgroup/emagpy ; live GitHub API commit7002ddf36a32902c5b077e31112cbb8f9ce5d076 verified2026-10-22, not merely a local tracking revision. Complete API responses and source checkout/archive are maintainer-only at `imaging-101-maintainer/research_batch6_20261022/groups/astro_earth/`. The code is GPL3; task LICENSE preserves its terms. Sourcecopyright/adaptation is retained in local physics/solver docstrings. Acquisition source: `src/examples/cover-crop/coverCropTransect.csv`, SHA256 `171ee6c84fdd4cc1e54fb48b38f0bdf13b3247d6c85d3f8288955449c098c891`. All30 rows and all six channels are included, with observation float64 precision, ordering, geometry and original supplied layer depths preserved. No observation was synthesized; no generator is shipped. Bound raw NPZ SHA256 `60183a8ec8ec76587c76ea47f107c9315edd5711c1a168a7dbd5fc767118bdcf`.

Selected unknown is a30 by7 spatial soil conductivity section under local1D layering. The source data are calibrated apparent conductivities in mS/m. Supplied coil identifiers explicitly encode VCP/HCP, spacing,30kHz and ground-contact h0. Surface elevations are not silently applied as instrument heights. Hidden baseline array key reconstruction has shape(1,30,7),float64,mS/m and is independently reproduced author-method output, not physical truth. Frozen `test/fixtures/upstream_stages.npz` contains exact input observations/geometry plus upstream Jacobian, roughness operator, CSgn comparison, selected bounded output and forward prediction. Its consumer is test/comparing_test/test_upstream.py. Notebook reference outputs have separately identified upstream and fresh measurements; both are consumed by the notebook or the comparison/atlas export. No data is supplied twice under different public names.

## Original implementation gate and environment

Private capture `capture_em.py` imports the unmodified archived original EMagPy package; the cleaned task never imports it. It calls Problem.createSurvey on the exact bundled30-row transect, setInit(depths0=[.25,.5,.75,1,1.5,2]), then original Problem.invert with CSgn and the selected original CS/L-BFGS-B bounded L2 branch. Selected parameters: alpha=.07,bnds=[(0,300)]*7,options ftol1e-12,gtol1e-8,maxiter1000,njobs1; other regularizers zero, initial conductivities20. Original independent capture command: research-runtime/Scripts/python.exe capture_em.py, exit0, runtime0.3743061s, archived log em_oracle.log and detailed environment em_oracle.json. Private oracle-only pandas/joblib/cloudpickle are installed under oracle_deps; neither requirements nor cleaned imports contain them. NumPy2.4.6,SciPy1.17.1,Matplotlib3.11.1,Python3.11.11,Windows CPU, OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1. NumPy reports its actual BLAS configuration in `em_environment.txt`. No GPU is required or used. Deterministic algorithm, no random seed. Existing shared packages were not upgraded or installed into the shared runtime.

Original CSgn produced negative conductivities at sites8 and15. The selected published bounded L2 branch improves physical validity by enforcing conductivity nonnegativity; all observations remain included. No claim of unchanged CSgn reproduction is made. Its frozen result is retained as a separately labeled comparison. The strong site8 residual remains visible. Absence of known noise covariance and lateral/metallic/model discrepancies prevents a physical-accuracy claim or reduced-chi-square criterion. No threshold relaxation was used to disguise an operator mismatch.

## Traceability and fixed validation cases

| Stage / mathematical statement | Upstream source/function | Local implementation | Independent case |
|---|---|---|---|
| Cumulative HCP/VCP kernels, C(0)=1,C(infinity)=0 | invertHelper.emSens | physics_model.cumulative_sensitivity | analytic monotonicity/partition/half-space checks |
| Finite layer kernel differences and half-space tail | invertHelper.fCS,forward1d_full,buildJacobian | physics_model.forward,jacobian,adjoint | frozen upstream Jacobian/prediction; adjoint dot product; uniform half-space |
| sqrt(sum(data^2)/6+alpha sum(diff(sigma)^2)/7) | Problem.invert,objfunc L2 branch | solvers.objective | independent finite-difference gradient and analytic quadratic minimizer |
| Bounded inverse, sigma0=20,L-BFGS-B | Problem.invert,solve | solvers.reconstruct | all30 independent frozen solutions;30 successful local optimizers |
| Alternative original GN normalization | Problem.invertGN | notebook frozen original comparison | distinctly labeled, not standard runtime |
| Same-objective unbounded inverse | independently derived normal equations | solvers.quadratic_solution | objective equivalence; physical-bound comparison |
| Cosine NCC; dynamic-range NRMSE | benchmark definitions | independent Judge and visualization.compare | known vectors,zero,constant-reference rejection |

Predeclared comparison tolerances in comparing test: operator rtol2e-14,atol1e-15; computed solution/prediction rtol2e-6,atol1e-5mS/m. They reflect deterministic float64 kernels and finite-difference stopping tolerances, respectively. Actual full local-vs-upstream maximum difference=0mS/m; NRMSE=0; NCC=1. Conductivity units are conserved; depth/spacing common scaling leaves the dimensionless kernel unchanged. Independent gradient tolerance2e-9 absolute. No implicit crop,scale,translation,mask or magnitude conversion is used.

## Metrics and threshold rationale

Hidden standard-method reference is an independently captured reconstruction, so self-comparison is NCC1/NRMSE0 and is not physical accuracy. Selected numerical reproducibility tolerance is NCC>=.99999 and dynamic-range NRMSE<=.0005. The latter corresponds to much less than0.1mS/m at this reference's dynamic range and comfortably contains finite-difference termination and upstream/local arithmetic effects while rejecting zero and factor-two submissions. Two separately executed deterministic local paths (CLI and active notebook) reproduce the captured output. Observation RMS8.3589644081273686mS/m is diagnostic only: no noise covariance was measured, and station8 dominates model discrepancy. Judge rejects malformed inputs and unavailable assets as evaluator errors; full valid poor outputs return fail. JSON contains only finite measured values. Visualization metrics share independent known numerical vectors with Judge tests.

## Actual command evidence

| Command | Exit | Wall seconds | Private log |
|---|---|---|---|
| `-m src.main` | 0 | 0.7871092999994289 | em_validation_0.log |
| `evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json` | 0 | 0.12831770000047982 | em_validation_1.log |
| `-m pytest test/ -v --basetemp D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth\em_pytest_final` | 0 | 1.2507304000027943 | em_validation_2.log |
| `-m jupyter nbconvert --to notebook --execute notebooks/covercrop_electromagnetic_conductivity.ipynb --output em_executed.ipynb --output-dir D:\yan1\agent\AutoSkill\imaging-101-maintainer\research_batch6_20261022\groups\astro_earth --ExecutePreprocessor.timeout=180` | 0 | 15.212505399991642 | em_validation_3.log |

The default Windows pytest temporary root has an existing permission problem; the same complete suite passes with --basetemp set to a fresh private maintainer directory. This is an OS temp-directory exception, not a skipped scientific test. Sixteen tests pass, covering input preservation, forward/Jacobian parity, full-pipeline parity, adjoint identity, uniform half-space, depth/spacing scale invariance, finite-difference gradient, explicit constants, valid/poor predictions, missing key/file,wrong shape,float32,complex,NaN,Infinity,missing hidden assets,invalid thresholds,and report-writing errors. The CLI and Judge also run as separate processes.

## Notebook, visual review and plan consistency

Canonical notebook has six story chapters (physical question, observations, forward model, fresh inverse, scientific comparisons/sensitivity, limitations). It runs all30 local inversions, writes a new submission, runs the independent Judge, compares reproduced-author and fresh maps, shows every-channel residuals, preserves the original GN comparison, computes the distinct unbounded comparison, and actively evaluates nine regularization settings. Six scientific figures have matching private PNG/SVG/PDF artifacts plus numeric_sources.npz under figures/covercrop_electromagnetic_conductivity. Runtime timeout180s is justified by actual15s complete notebook duration including startup and comparisons. Nbconvert exit0; every code cell executed with no error outputs. Windows ZMQ selector-thread and local TCP warnings are captured; no numerical warnings or reconstruction errors occurred. The temporary executed notebook remains private and canonical executed outputs are retained. Visual QA is recorded separately in em_visual_review.json. README math uses fenced math; notebook uses display-dollar equations.

Semantic plan/source review confirms the exact objective square root and factors1/6,1/7; positive-down local depth; ground-contact height; far-half-space tail; initialization/bounds/stopping criteria; actual callable signatures and optional comparison paths. A shape/signature match alone was not used as mathematical evidence. Source imports are recorded in em_static_audit.json and include only standard library,NumPy,SciPy,Matplotlib; no upstream package,domain solver,cross-task import,target weights or privileged array appears in runtime. Raw arrays and README alone define geometry/output; no truth or hidden seed/object is exposed. Public README/requirements have no GitHub URLs. No standalone metadata file exists.

## Completion gates

| Gate | Status | Evidence |
|---|---|---|
| Paper, exact remote revision, hashes, equation/function/test mapping | passed | provenance and traceability above |
| Original and local exact-bound observations | passed | em_oracle and em_validation0 |
| Independent parity and physical checks | passed | complete16-test suite |
| Constants, units, precision and coordinates | passed | exact input and README tests; physical scaling |
| CLI submission and fresh Judge | passed | em_validation0/1 |
| Judge adversarial inputs and evaluator failures | passed | unit tests |
| Visible README/data/environment suffice | passed | self-contained forward model and output contract |
| No truth,seed,target-calibration or GitHub leakage | passed | static audit; public observations/geometry only |
| Minimal runtime without upstream dependencies | passed | clean shared-runtime commands |
| Explicit local mathematics, call-role audit | passed | source/operator/objective and import audit |
| Full tests | passed |16 passed |
| Active notebook execution | passed | em_validation3; six computed figures |
| Result provenance and metric definitions | passed | independent upstream vs fresh labels |
| README/plans semantically match implementation | passed | equations/signatures/execution review above |
| Final inventory/license/temp cleanup | passed | final evidence inventory; LICENSE preserved |

## Paper version, direct mathematical reading and archive index

The actual read paper is the32-page accepted author manuscript hosted by Lancaster EPrints (not a claim to have read the publisher typeset version): https://eprints.lancs.ac.uk/145774/1/Author_copy_accepted_.pdf . Private file em_paper_accepted.pdf has SHA256 af15299629fd0fd4f187d4cd7bb4b8784ecb1e1955d27f59629bdcfed3c30148. Landing page https://eprints.lancs.ac.uk/id/eprint/145774/?template=browse identifies this as the accepted version, CC-BY-NC-ND4.0; it is archived privately and not redistributed in the task. The complete extracted text is archived as em_paper_accepted.txt. Directly reviewed relevant sections: section2.2, manuscript pages8–9, equations3–4 give VCP/HCP cumulative kernels using depth normalized by coil separation; section2.6.1, pages14–15, equations10,12,14 give data and adjacent-layer L2 penalties with normalization by number of coils/layers; section2.6.2, pages15–17, discusses Gauss–Newton and bounded general optimization. The source code takes the square root of the paper's total quadratic objective. This is a monotone transformation that preserves the minimizer but changes numerical convergence criteria; the cleaned implementation preserves the selected source code's square root and stopping settings and explicitly documents this paper/code convention. The paper's caveats concerning high conductivity, heterogeneity, height and instrument calibration are retained as interpretation limitations. This task reproduces the selected cumulative-sensitivity branch, not the paper's full-Maxwell inverse. The paper's provided crop survey code is the bound task's observation source; paper case-study result numbers are not relabeled as measurements of this run.

Private `em_archive_index.json` lists exact filenames, byte counts and SHA256 hashes for the manuscript, extracted text, landing page, complete source archive, live remote revision response, oracle arrays, original run log, capture script and actual notebook environment. The index is under the same maintainer group directory identified above. The optional McNeill TN6 PDF request to https://geonics.com/pdfs/technicalnotes/tn6.pdf returned HTTP403; it was not obtained or hashed. The required accepted source paper was obtained, read and hashed as above. No completion gate depends on this unavailable supplementary document. Paper PDFs remain outside the final task.

The actual executed notebook kernel prints and records sys.executable, Python, OS, NumPy/SciPy versions and module paths, Matplotlib version/backend, and thread controls in its first executed code-cell output. The captured JSON is em_notebook_kernel_environment.json. Assertions verify the executable belongs to the shared research-runtime and NumPy2.4.6/SciPy1.17.1 actually ran in the kernel. Thus the nbconvert launcher alone is not used as environment evidence.

Actual notebook environment:

```json
{
  "executable": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
  "os": "Windows-10-10.0.26200-SP0",
  "numpy": "2.4.6",
  "numpy_file": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Lib\\site-packages\\numpy\\__init__.py",
  "scipy": "1.17.1",
  "scipy_file": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Lib\\site-packages\\scipy\\__init__.py",
  "matplotlib": "3.11.1",
  "backend": "module://matplotlib_inline.backend_inline",
  "threads": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1"
  }
}
```
