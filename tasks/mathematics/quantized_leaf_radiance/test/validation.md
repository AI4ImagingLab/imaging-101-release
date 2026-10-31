# Quantized leaf provenance and validation

## Scope, source, paper and exact data

Selected task: native RGB interval-quantization inversion with explicit multichannel TV/TGV primal-dual updates. This observation operator is distinct from existing masked-texture, deblurring, exposure and sparse-transform tasks. The original159-task frozen inventory and semantic catalog were inspected before selection. The use of a shared regularizer does not turn a different acquisition into an identical task.

Original author repository https://github.com/kbredies/tgv_pycuda, exact live-verified revision9cd20ec40b2874ba610b570f77be0c49e91581e2. Remote API response is private remote_commit.json; source_provenance.json records verified source ZIP SHA256ea2fcb3189b7df68690d0b884386ec51ce3e83e4f02166b89b355a5efad853e5. No remote freshness is inferred from a local tracking branch. Apache2 source license and copyright are preserved. Original file SHA256:

```json
{
  "dequantize_pycuda.py": "a022f749e0855ab4782d312e1e1688c127305e4f0921250c0141720eec5b81a0",
  "common_pycuda.py": "c599b81d9aaf334f15e3163b01405462eab1d30b755a25e91418147e38dcdf89",
  "test_dequantize.py": "3b5cad6939716294a468a5bb0defc2e4bd53f07c0f734587dcaf188223fa6092",
  "examples.ipynb": "6f5a0ea716c1c2de1c5a7c8f4256c15c6e627d70facfb5196643b06aa11442ca",
  "test_data/oak_leaf.png": "7aadf18f641583f58680360fc582d2c18552cdb454c76699cd2306d272f87b2f",
  "test_data/README.md": "e3218cc9cdebf0cad28f633d32a57520b47c2485bfbb15e2806e5ecd6bc19de4",
  "LICENSE": "79b4e2f56966cd4262c8ef4001b106cb425ede221826ee634db60f97e5eee49e"
}
```

Paper: Bredies, *Recovering Piecewise Smooth Multichannel Images by Minimization of Convex Functionals with Total Generalized Variation Penalty*, LNCS8293:44–77,2014, DOI10.1007/978-3-642-54774-4_3. The task-relevant author April2012 SFBReport2012-006 Section4.4,p28,Eq21 was actually read from primary indexed paper text returned by web search. It specifies a box indicator PLUS midpoint p-power fidelity, not merely pure constrained TGV. Publisher bibliographic identity and the author's source/examples confirm the link. Primary URL https://imsc.uni-graz.at/mobis/publications/SFB-Report-2012-006.pdf . Access record paper_reading.md SHA256774c84693c287f282c3b56d2d991dbeec09f8e4947e797ff20895bf4dcc51032. Direct whole-PDF download repeatedly timed out; no local PDF, whole-paper hash, complete-paper reading or paper visual review is claimed. The task requires a paper/version and equation mapping, not fabrication of an unavailable download. This access limitation does not affect original executable method/data verification.

Photo: Stephen Heron, Oak leaves and bokeh, Flickr6802942537,2012, CC-BY-SA2.0, supplied as native512×512RGB test_data/oak_leaf.png. Attribution and share-alike terms are preserved in NOTICE. The original pyplot.imread decodes float32[0,1] encoded camera intensity. The author's test applies floor(6*f)/6 and ceil(6*f)/6, no crop/downsample/noise/seed. Classification: **author_defined_new_simulated quantization of author-provided real photograph**. It is not real sensor quantization. Public NPZ preserves the exact original float32 observations. The hidden original photo is actual pre-quantization truth, not a method result. No privileged original pixels, object/seed or target-derived calibration leak into the public two-array NPZ.

Bound packaged asset SHA256:

```json
{
  "data/raw_data.npz": "69f32d263ec76678211b3de1753ca58c70f0851b0976a9e365d05e83afdcb318",
  "evaluation/ground_truth.npz": "d26e8a83efe41ac344489d6e121160cbd43550f1ecc22d62fb7a3501586747ab",
  "notebooks/reference_outputs/upstream_outputs.npz": "3077bbfa1debe5a05051986d1b9228d5f5b89a5348ed3bc995eb6d505e6791af",
  "test/fixtures/upstream_states.npz": "33e6eca6ec27c0e19888b211b5d530eab06d6817a910e3c0206962428253c06c"
}
```

## Original execution before extraction

Original command: private oracle-env/Scripts/python.exe run_original.py, followed by capture_stages.py for independent numerical stages. Both call unmodified author dequantize_pycuda TV/TGV functions on the EXACT native observations, alpha/alpha1=.45,fac=2,maxiter=2000,vis=-1. Original tests' eleven-weight scan is preserved in active notebook comparisons; examples.ipynb's separately selected TV.49 is included in that scan and distinguished from paired same-.45-weight panels. Original implementation code is byte-identical to the hashed revision. Private capture code only transfers inputs/outputs and reads return-state locals with sys.settrace; it does not replace mathematical functions.

The original GPU environment is isolated under this maintainer archive: Python3.11.11,NumPy2.4.6,PyCUDA2025.1.2,Matplotlib3.11.1,Pillow12.3.0,NVIDIA RTX4080SUPER,CUDA toolkit12.2,driver13030,MSVC2019 14.29.30133,WindowsSDK10.0.19041. The initial private PyCUDA build failed because compiler include paths omitted installed headers; retry with explicit private INCLUDE/LIB/PATH succeeded. Build logs/wheel hash are in pycuda_build_result.json and pycuda_install_retry.log. No shared-runtime dependency installation or upgrade occurred. Original compiler output includes a benign kernel.cu warning; numerical run exit0, not a successful import substituted for execution.

Original actual run and environment:

```json
{
  "tv": {
    "runtime_seconds": 2.1590788000030443,
    "shape": [
      512,
      512,
      3
    ],
    "dtype": "float32",
    "psnr": 29.586143493652344
  },
  "tgv": {
    "runtime_seconds": 0.14991040001041256,
    "shape": [
      512,
      512,
      3
    ],
    "dtype": "float32",
    "psnr": 29.864883422851562
  },
  "environment": {
    "device": "NVIDIA GeForce RTX 4080 SUPER",
    "cuda_driver_version": 13030,
    "python": "3.11.11 (main, Mar 17 2025, 21:01:29) [MSC v.1943 64 bit (AMD64)]",
    "numpy": "2.4.6",
    "source_sha256": "a022f749e0855ab4782d312e1e1688c127305e4f0921250c0141720eec5b81a0"
  }
}
```

Predeclared criteria are private predeclared_criteria.json, fixed before local implementation: stage rtol2e-5/atol3e-6; full native source NRMSE<=2e-5; independent weighted adjoints, proximal minimization, interval consistency and boundary checks. The initial corner capture was recognized as uniform/zero-gradient and explicitly rejected as uninformative. It was replaced BEFORE extraction by a nondegenerate8×9RGB region rows180:188,columns220:229 with six different quantization levels. Original state arrays at0,1,2,5,30updates cover every u,u_,p and TGVw,w_,q; none is generated by cleaned code. All45 state-array comparisons are frozen. Max CPU stage discrepancy1.1920928955078125e-07, tensor stage discrepancy1.7881393432617188e-07; differences are float32 arithmetic/fusion, not exact-equality claims.

## Equation/function/test traceability

| Paper/source stage | Original file/function | Local function | Independent evidence |
|---|---|---|---|
| Section4.4 interval model | test_dequantize.py floor/ceil | quantization_intervals,generate_observations | exact decoded-photo/public-bound equality |
| Multichannel gradient and symmetric tensor | common_pycuda.py gradient kernels | gradient,symmetric_gradient | original state fixtures; weighted adjoint/edge tests |
| Negative adjoints and signs | common_pycuda.py divergence kernels | divergence,symmetric_divergence | independent inner-product identities, outer-face test |
| Pixelwise joint-color dual balls | project kernels | project_vector,project_tensor | original dual fixtures; independently bounded norms |
| Eq21 midpoint quadratic+box prox | clip_shrink2 | midpoint_prox | scalar constrained quadratic minimization |
| TV primal-dual updates | tv_dequantize | reconstruct(method='tv'),reconstruct_tensor | every original stage, native full reference |
| TGV primal/auxiliary/extrapolated updates | tgv_dequantize | reconstruct(method='tgv'),reconstruct_tensor | every original stage and full2000-update output |
| Objective and source conventions | selected source constants and Eq21 | objective | interval indicator,constant case, no per-pixel normalization |
| Benchmark metrics | explicit benchmark definitions | independent Judge and visualization.metrics | known[1,1,1]vs[0,1,2] case,zero and malformed inputs |

The weighted symmetric tensor includes its off-diagonal twice; the forward symmetric derivative's half factor is retained. Outgoing derivative faces are zero; stored boundary duals are ignored by adjoints. Affine fields are null only in the interior under this convention; no incorrect global-affine claim is made. The source contains other proximal alternatives, but selected dequantization calls midpoint L2 clip_shrink2. Those unused branches are not silently substituted. Every mathematical step is local readable NumPy or general Torch computation; no imaging-domain library is imported.

## Fresh execution, metrics and Judge

The standard CPU command python -m src.main exited0 in204.854835s (standard_cli.json/log). An earlier independent full CPU evaluation took179.245363s and yielded maximum source discrepancy2.086162567e-7, source-NRMSE1.873393707e-8 (first_full_local.json). The fresh tensor full run independently yielded max2.384185791e-7/source-NRMSE1.926915516e-8 (tensor_full_parity.json). CPU support is measured, not inferred from imports. The standard CPU source is used even though a GPU is available.

Notebook fresh measurements and original-source measurements are separately named in private figures/measurements.json; original float32 PSNR reporting is preserved in original_run.json, while all comparison/benchmark metrics use explicit float64 diagnostics. Both formulas and reference/scale were checked before comparing them. The original true-reference baseline NRMSE is.03211854620544848 and NCC.9975063086444824; thresholds are.9*NCC,1.1*NRMSE and maximum interval violation1e-6. The NRMSE criterion excludes the uninformative midpoint(.0456258098761), and zero/wrong-bin predictions fail. No threshold was relaxed to accommodate a poor fit. Neither low simulated error nor source parity establishes real detector accuracy.

Final metrics, measured selected runtimes and parity:

```json
{
  "rows": {
    "midpoint": {
      "ncc": 0.9949644664144419,
      "nrmse": 0.045625809876119874,
      "psnr_db": 26.815788271374256,
      "interval_violation": 0.0
    },
    "original_tv": {
      "ncc": 0.997337369951077,
      "nrmse": 0.03316598437091984,
      "psnr_db": 29.586142165102338,
      "interval_violation": 0.0
    },
    "original_tgv": {
      "ncc": 0.9975063086444503,
      "nrmse": 0.03211854620544848,
      "psnr_db": 29.864882411852452,
      "interval_violation": 0.0
    },
    "fresh_tv": {
      "ncc": 0.997337369951282,
      "nrmse": 0.03316598437009082,
      "psnr_db": 29.586142165319448,
      "interval_violation": 0.0
    },
    "fresh_tgv": {
      "ncc": 0.9975063086307837,
      "nrmse": 0.03211854629657328,
      "psnr_db": 29.86488238720937,
      "interval_violation": 0.0
    },
    "independent_standard_cpu": {
      "ncc": 0.9975063086434226,
      "nrmse": 0.032118546214981857,
      "psnr_db": 29.86488240927432,
      "interval_violation": 0.0
    }
  },
  "runtime_seconds": {
    "tv": 1.6526398999994854,
    "tgv": 6.2952158999978565
  },
  "parity_max_abs": {
    "tv": 1.7881393432617188e-07,
    "tgv": 2.384185791015625e-07,
    "standard_cpu_vs_original_tgv": 2.086162567138672e-07
  },
  "python": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
  "device": "cuda"
}
```

The standalone Judge scores actual photo truth, explicit public-bound feasibility and every required pixel/channel. It validates key,batchshape,float32,finite,[0,1],returns0/1/2 and writes UTF-8 allow_nan=False JSON. Malformed inputs produce empty metrics. Missing hidden reference/config/assets and report-write errors are evaluator errors. Foreign-cwd invocation and independent known-case metric checks are tested. There is no crop,alignment,fitgain,gamma or magnitude conversion.

## Active notebook and visual review

The canonical notebook contains problem,data,physics,method,results,conclusion; active calls use only cleaned src modules. Complete selected TV/TGV reconstructions and all22 source-grid parameter cases execute at native512²RGB for2000updates. No cached output is plotted as fresh. True photograph, original-method outputs, newly computed tensor results and independently executed standard CPU result are explicitly distinguished.

Actual nbconvert command/environment:

```json
{
  "command": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
    "-m",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "notebooks/quantized_leaf_radiance.ipynb",
    "--ExecutePreprocessor.kernel_name=imaging101-batch6",
    "--ExecutePreprocessor.timeout=3600",
    "--output=quantized_leaf_radiance_executed.ipynb",
    "--output-dir=output"
  ],
  "exit_code": 0,
  "wall_seconds": 125.63446990000375,
  "kernel_path": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research_batch6_20261022\\jupyter\\share\\jupyter\\kernels\\imaging101-batch6\\kernel.json"
}
```

Actual kernel executableD:\yan1\agent\AutoSkill\imaging-101-maintainer\research-runtime\Scripts\python.exe,devicecuda. Explicit shared-runtime kernel avoids PATH resolving an unrelated Anaconda environment. The3600s per-cell timeout allows the full22-case CPU route; actual notebook host uses the GPU and full walltime above. Nonfatal ZMQ Windows selector/TCP notices remain in the private log; all code cells completed without errors. Initial notebook execution failed on an embedded newline string, which was corrected and the entire notebook rerun; that failure is not represented as a successful run.

All seven actual scientific figures were opened: full truth/midpoint/fresh; matched native detail crops; common-scale errors; full metrics/profile; original/fresh parity; complete22case sensitivity/interval checks; and actual TGV auxiliary decomposition. Text,axes,units,colorbars,legends and common scales are readable. Accuracy bars start at0; narrow sensitivity axes are visible and labeled; no uncomputed placeholders appear. Figures/export numeric arrays are private, and the canonical executed notebook holds rendered figures. Markdown display formulas use double-dollar delimiters in the notebook and math fences in README/plans for GitHub/Jupyter compatibility. Temporary executed output notebook is removed.

## Plans, archive consumers and completion gates

approach.md and design.md were semantically checked against actual signatures/defaults, midpoint fidelity,signs,weights,boundary,step sizes,precision and execution paths. No proposed method is described as current. Requirements are the validated shared NumPy,Matplotlib,Torch subset; PyCUDA and compiler dependencies belong only to the private original oracle.

Private archive index: ../imaging-101-maintainer/research_batch6_20261022/groups/root/tgv/archive_index.json. Original source/source.zip,oracle build/run/capture scripts,complete environments/logs,initial rejected uniform capture records,primary-paper access note,final exports and numeric experiment archive are outside the task. Frozen fixture consumers are test/comparing_test/test_original.py. Original full TV/TGV reference consumers are the notebook and full pipeline parity test. Actual photo truth is consumed by Judge,notebook and deterministic acquisition equality test. No redundant truth copy or unused reference is shipped.

| Required gate | Status | Evidence |
|---|---|---|
| Paper/version,exact revision,data hashes,mapping | passed | Primary indexed Eq21 reading and published identity; original files/data hashes and table above; full PDF download limitation disclosed |
| Exact-observation original and local runs | passed | Unmodified original CUDA, full CPU standard CLI and fresh tensor; private commands/logs |
| Independent parity and physics | passed | Original45 state arrays/full image; weighted adjoints,prox,projection,boundary/acquisition tests |
| Constants,units,precision | passed | Native float32 protocol and joint-color norm; explicit CPU/CUDA tolerances |
| CLI output and standalone Judge | passed | Fresh actual standard output; strict independent evaluator and poor-output checks |
| Public sufficiency/no leakage | passed | README/bounds/environment suffice; only public intervals, no truth/calibration leak |
| Local mathematics/minimal dependencies | passed | Explicit source extraction, general NumPy/Torch only; CPU path executed |
| Unit,metric,Judge,parity,pipeline tests | passed |38 passed; parent test command/log indexed |
| Active notebook/comparisons | passed | Full nbconvert exit0,actual kernel,22case scan,7 reviewed figures |
| Provenance/threshold consistency | passed | Original/fresh/photo clearly separated; nonzero true-reference baseline and adverse examples |
| README/plans semantic consistency | passed | Sign/weight/unit/call-flow review above |
| Formula parse/inventory/final static review | passed | Parent audit errors=[]; independent foreign-cwd Judge exit0;45 formulas pass MathJax/KaTeX;7 actual figures inspected |

Optional full PDF archival remains unavailable; no mandatory original execution or mathematical check is waived because of that limitation.
