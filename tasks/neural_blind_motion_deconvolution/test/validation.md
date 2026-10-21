# Provenance and validation

## Scope and task identity

Task: neural_blind_motion_deconvolution, Mathematics, Hard. This is a complete
full-raster real-exposure joint image/kernel inverse problem, not a simulated
small-image Wiener task. The reference-release image network, three nonlocal
blocks, 51x51 unknown kernel, both discrepancy stages and all 2500 joint updates
are retained. No earlier task is imported or modified. No skills were used.

This file is maintainer-only. Solver-visible materials are README, public raw
observations and environment information. No independent exposure, fitted kernel,
pretrained inverse weights, upstream code URL or private source PNG is in that
allowlist. There is no data-generating scene seed: the image was photographed.

## Verified source versions and scientific selection

Method repository: https://github.com/csdwren/SelfDeblur
Remotely verified HEAD: e1c79ee528eb333416b032e4bc002337c2d5c0f3, verified using
git ls-remote on 2026-10-18. The complete ZIP for that immutable revision is
archived privately. A git clone failed with an RPC/network reset; the subsequent
official codeload archive succeeded. This is not a claim that a stale local
tracking branch was remotely fresh.

Published method: Dongwei Ren, Kai Zhang, Qilong Wang, Qinghua Hu, Wangmeng Zuo,
Neural Blind Deconvolution Using Deep Priors, CVPR 2020, published CVF paper:
https://openaccess.thecvf.com/content_CVPR_2020/papers/Ren_Neural_Blind_Deconvolution_Using_Deep_Priors_CVPR_2020_paper.pdf
The downloaded published PDF, metadata HTML and extracted text are private.

The selected release `selfdeblur_ycbcr.py` differs from the paper's Eq.6: it removes
TV, adds dot-product nonlocal blocks, and switches from MSE to 1-SSIM after 500
steps. This task reproduces that declared release variant. Paper Tables1-4 and
their Levin/Lai data are historical context, not measurements for this RealBlur
observation. No competing paper method is displayed as freshly computed.

Dataset repository: https://github.com/rimchang/RealBlur
Exact remote commit API response: 4d65bb32daa6dd89fd629311a43e69c945e631e9.
Dataset: Jaesung Rim, Haeyun Lee, Jucheol Won, Sunghyun Cho, Real-World Blur Dataset
for Learning and Benchmarking Deblurring Algorithms, ECCV 2020. Official archive:
https://huggingface.co/datasets/rimchang/RealBlur/resolve/main/RealBlur.tar.gz
The official README explicitly releases the dataset under CC BY 4.0.

The upstream SelfDeblur `house.jpg` example was investigated but is not bound or
redistributed because its original image authorization/provenance is unclear.
RealBlur provides independently acquired camera exposures and explicit terms.
Among the available complete scene026 pairs, the selected blur_3 has the largest
observed/independent-exposure RMS discrepancy and about0.53 of the reference's
Laplacian contrast. It includes strong directional streaking and degraded sign
lettering. This selection uses development truth only to establish that the
problem is meaningfully blurred; no target-derived calibration reaches solvers.
The complete raster is retained. The selected physical model is an approximation
for real motion/noise, whose actual quality and residuals must pass the gates.

## Real observations, precision and hidden reference roles

Observation source member:
RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene026/blur/blur_3.png
Independent exposure source member:
RealBlur-J_ECC_IMCORR_centroid_itensity_ref/scene026/gt/gt_3.png
Both complete decoded images are753x668 RGBuint8. They were distributed after
the dataset's ECC/intensity processing. No task resize/crop/synthetic degradation
is applied. Convert RGB to full-range integer Y by
`(4899R+9617G+1868B+8192)>>14`, then float32/255. This conversion is independently
compared pixelwise with the upstream OpenCV convention. Quantization is preserved;
there is no float64-to-float32 downcast of a high-precision radiometric acquisition.

Visible data/raw_data.npz contains only observation(1,753,668)float32, dimensionless.
Hidden evaluation/ground_truth.npz contains reconstruction(1,753,668)float32 from
the independent short exposure. It is actual independently acquired processed-
image truth, not a reconstruction mislabeled as truth. Its noise and imperfect
registration limit interpretation as an ideal latent scene. No independent true
motion kernel is available. Kernel evaluation therefore uses nonnegativity,
unit-sum and its coupled unaligned ability to explain the public observation.

The private archive contains only a 67108864-byte HTTPRange206 prefix of the
13115712166-byte official archive. The official full-archive LFS expected hash
is source metadata, not a measured local full-archive hash. Complete PNG members
were individually extracted and decoded, and their independent SHA256 values
are recorded below. Neither a successful first64MiB download nor a truncated-gzip
EOF is misreported as complete archive verification. Extraction/member manifests
are archived privately; incomplete members are excluded.

## Mathematical and code traceability

| Paper/release stage | Original file/function | Independent local implementation | Independent evidence/consumer |
|---|---|---|---|
| Sec3.1 image formation; Eq5 joint unknowns | selfdeblur_ycbcr.py conv2d | physics_model.blur_image | scipy discrete correlation, adjoint identity and finite-difference test |
| Eq5/figure2 image range | networks/skip.py skip | ImagePrior.forward sigmoid | frozen complete original first latent and full output parity |
| Sec3.2/figure2 kernel generator | networks/fcn.py fcn | KernelPrior.forward | frozen first kernel; simplex/gradient tests; full kernel parity |
| Five scales/odd dimensions | networks/skip.py; common.Concat | Scale.encode/decode | complete803x718 neural fixture and initialization checks |
| Three nonlocal release blocks | networks/non_local_dot_product.py forward | SpatialInteraction.forward | all142 image parameter gradients, full neural fixture and full-run parity |
| Feature padding/normalization | networks/common.py conv,bn,act | conv,norm_act,Scale | complete original initialization and first-step gradient/update capture |
| Fixed optimization inputs | utils/common_utils.py get_noise; ycbcr loop | initialize_priors/reconstruct | original initial tensors and first perturbation frozen privately |
| Eq5 squared data discrepancy | selfdeblur_ycbcr.py MSELoss | loss_for_step t<500 | frozen MSE, both first-layer gradients, all image parameter gradients |
| Release structural discrepancy | SSIM.py gaussian/create_window/_ssim | structural_similarity | original loss/input gradient and independent central finite differences |
| Sec3.3 Algorithm2 joint Adam | selfdeblur_ycbcr.py optimizer/step | reconstruct; general torch.optim.Adam | first original Adam updates and fresh full integration test |
| Release rates/stopping | selfdeblur_ycbcr.py MultiStepLR/num_iter | learning_rates;2500-loop | exact milestone unit cases, complete upstream/cleaned histories |
| Release saved-before-update output | selfdeblur_ycbcr.py save block | ReconstructionResult center crop | full image/kernel parity, executed notebook |
| Translation gauge | independent finite-support derivation | metric align_translation only | discrete nonwrapping image/kernel shift identity test |
| Truth/measurement scoring | task-defined independent formulas | visualization metrics; standalone Judge | known numerical metric cases and cross-implementation agreement |

All physical model signs, boundary domains, MSE mean factor, SSIM constants,
sigmoid/softmax ranges, operator crop coordinates, optimizer initialization/order
and saved-before-update convention were reviewed against actual source. Design
describes actual callable signatures and side effects. The803x718 latent models
all753x668 observed pixels; output is its25-pixel center crop. Only the703x618
valid correlation of that submitted center is compared with observation25:-25.
No implicit optical padding, double crop or target boundary is introduced.

## Original execution, numerical backend and investigated divergence

Private oracle_run.py orchestrates unmodified original networks.skip.skip,
networks.fcn.fcn and SSIM.SSIM on the exact bound luminance. It retains the complete
2500-step acquisition/loss/Adam schedule from selfdeblur_ycbcr.py, removes only
file/UI iteration around other demo photographs, supplies the chosen support51,
and captures numeric tensors before lossy PNG saving. It does not load cleaned
functions. Private OpenCV is only a reader/channel-conversion oracle; torchvision
and other unused demo utilities are not needed. The original method, not merely
an import or a cached paper result, executes on the bound observations.

An initial original run completed2500 steps in362.9962659999728s with only
cuDNN deterministic flags. Its fixed-gauge NCC=.9918410391059359,
NRMSE=.029215276490004403, unaligned forward NRMSE=.008502817858528754;
blurred identity NRMSE=.041142400423377426. This historical run set the quality
boundaries before any cleaned full run. The first cleaned run passed quality
but failed strict full numeric parity and reached a different translation gauge.
It is not accepted as final reproduction evidence.

Investigation compared every initial parameter, complete forward arrays, and
all142 image parameter gradients for multiple steps. With cuDNN-only settings,
even independent executions of the original source had first-backward differences
around1e-8 to1e-7, amplified by nonconvex Adam optimization. CUDA feature-padding
backward and other operations were not covered by cuDNN's flag alone.
Enabling `torch.use_deterministic_algorithms(True)` together with the cuBLAS
workspace config makes original/local initial arrays, full forward tensors and
all142 image gradients exactly equal through the independently checked first
four iterations. This is recorded in diagnose_deterministic.log. The final
oracle, stage fixtures, cleaned CLI, active integration test and notebook all use
that stronger backend. The complete deterministic original run must finish and
pass the already frozen quality bounds before subsequent execution proceeds.

Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA matrix work; disable TF32 for both
cuBLAS and cuDNN; disable cuDNN benchmarking; select deterministic cuDNN algorithms;
enable full Torch deterministic algorithms. Torch seed42, CPUthreads4, float32
training/float64 metrics. Adam(beta.9,.999,epsilon1e-8,weight_decay0) remains joint.
The explicit epoch table replaces only the deprecated scheduler call ordering;
the reference's schedule/implicit-softmax warnings are recorded, not hidden as
scientific failures. Original-source license absence is handled below rather
than by copying the upstream framework into the task.

## Acceptance frozen before final cleaned execution

The fixed image gauge uses demeaned correlation, a +/-25 integer displacement
box and row-major ties; scoring uses the50-pixel reference interior, retaining
653x568=370904 pixels. It removes only the unavoidable kernel-origin ambiguity.
The margin is kernel radius plus maximum allowed displacement, not an
easy-region crop picked from cleaned errors. No rotation, intensity offset/gain,
subpixel interpolation or local registration is optimized.

NCC is cosine similarity without mean subtraction; NRMSE is RMS/reference
dynamic range. Forward NRMSE is unaligned valid correlation/image-observation RMS
divided by observed-region range. Kernel mass error is abs(sum(k)-1); negative
mass is sum(max(-k,0)); image values must lie[0,1]. Thresholds remain
NCC>=.8926569351953423; NRMSE<=.03213680413900485;
forward NRMSE<=.00935309964438163; mass error<=1e-5; negative mass<=1e-8.
They are not relaxed after the deterministic-backend investigation. A delta
kernel plus identity blur fails image quality despite near-zero forward residual.
Negative/nonunit/mismatched kernels are separately rejected. Numeric full-pipeline
parity uses rtol2e-5/atol2e-6 for both required arrays; neural stage tolerances
are rtol2e-6/atol2e-7, and independently validated CPUadjoints use float64.

## Commands and artifact consumers

Shared research environment: ../../environment-research.yml, fixed versions.
Runtime subset: requirements.txt. No per-task upgrade or domain library is used.
Original-only packages are installed with --target into the private oracle-deps
directory, never the cleaned environment. OpenCV4.13.0.92 is the image/color
reader oracle; pypdf6.11.0 only extracts the privately archived paper.

Commands from the task directory:

```text
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v --basetemp=<private task01 pytest-temp>
jupyter nbconvert --to notebook --execute notebooks/neural_blind_motion_deconvolution.ipynb --output neural_blind_motion_deconvolution_executed.ipynb --output-dir <private notebook-execution> --ExecutePreprocessor.timeout=5400
```

The integration test itself runs a complete new CLI process and Judge in its
temporary directory before comparison. It does not inspect stale output as proof
of pipeline correctness. The notebook likewise calls local reconstruct2500 and
evaluates its fresh image/kernel, preserving original/independent truth/blurred
identity comparisons, kernels, losses, residuals, metric bars and a common profile.
Generated figures must still be visually inspected; successful execution alone
does not pass that gate.

upstream_stages.npz is consumed by the SSIM and full neural/Adam comparing tests.
Its perturbation, first latent/kernel/exposure, losses/gradients/updates come from
the independent original run or an explicitly identified original first-step
replay. upstream.npz is consumed by full fresh pipeline comparison and notebook;
its history is a method measurement, not truth. No fixture is generated by a
cleaned function and relabeled upstream. Full checkpoints are only private oracle
artifacts; no checkpoint is used by the shipped solver, Judge or notebook.

## Completion gates

<!-- BEGIN EXECUTION EVIDENCE -->

### Automatically recorded final execution evidence

Driver state: **computations_passed_review_pending**. UTC: 2026-10-18T15:03:36.438574+00:00

- original_deterministic_2500: **passed**.
- original_complete: **passed**.
- previously_frozen_quality: **passed**.
- capture_all_original_gradients: **passed**.
- freeze_independent_evidence: **passed**.
- cleaned_complete_cli: **passed**.
- strict_full_parity: **passed**.
- fresh_standard_judge: **passed**.
- pytest_complete: **passed**.
- notebook_complete: **passed**.
- notebook_execution: **passed**.
- figure_visual_review: **unverified**.
- fresh_notebook_judge: **passed**.

Commands, native exit statuses and elapsed seconds:

```json
[
  {
    "stage": "original_deterministic_2500",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\oracle_run.py"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01",
    "exit_code": 0,
    "seconds": 810.1166273999261,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\original_deterministic_2500.log"
  },
  {
    "stage": "capture_all_original_gradients",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\capture_all_gradients.py"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 0,
    "seconds": 10.009522399981506,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\capture_all_original_gradients.log"
  },
  {
    "stage": "freeze_independent_evidence",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\freeze_evidence.py"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 0,
    "seconds": 10.010109400027432,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\freeze_independent_evidence.log"
  },
  {
    "stage": "cleaned_complete_cli",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "src.main"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 0,
    "seconds": 1410.2748094999697,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\cleaned_complete_cli.log"
  },
  {
    "stage": "fresh_standard_judge",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 0,
    "seconds": 10.00982420006767,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\fresh_standard_judge.log"
  },
  {
    "stage": "pytest_complete",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "--basetemp=D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\pytest-final-temp"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 1,
    "seconds_until_status_interruption": 320.0719783999957,
    "seconds_after_adoption": 320.04897580004763,
    "note": "Native live process handle adopted after private status writer failed; no reconstruction restarted.",
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\pytest_locale_failure.log"
  },
  {
    "stage": "pytest_complete",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "pytest",
      "test/",
      "-v",
      "--basetemp=D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\pytest-utf8-temp"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 0,
    "seconds": 780.1523446000647,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\pytest_complete.log"
  },
  {
    "stage": "notebook_complete",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "-m",
      "jupyter",
      "nbconvert",
      "--to",
      "notebook",
      "--execute",
      "notebooks/neural_blind_motion_deconvolution.ipynb",
      "--output",
      "neural_blind_motion_deconvolution_executed.ipynb",
      "--output-dir",
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\notebook-execution",
      "--ExecutePreprocessor.timeout=5400",
      "--ExecutePreprocessor.kernel_name=imaging-101-research"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 0,
    "seconds": 830.2457387000322,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\notebook_complete.log"
  },
  {
    "stage": "fresh_notebook_judge",
    "command": [
      "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research-runtime\\Scripts\\python.exe",
      "evaluation/judge.py",
      "--prediction",
      "output/reconstruction.npz",
      "--report",
      "output/judge_report.json"
    ],
    "cwd": "D:\\yan1\\agent\\AutoSkill\\imaging-101-release\\tasks\\neural_blind_motion_deconvolution",
    "exit_code": 0,
    "seconds": 10.00958620000165,
    "log": "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\fresh_notebook_judge.log"
  }
]
```

Final deterministic original measurement:
```json
{
  "seconds": 803.9632628000109,
  "torch": "2.9.1+cu128",
  "numpy": "2.4.6",
  "gpu": "NVIDIA GeForce RTX 4080 SUPER",
  "seed": 42,
  "iterations": 2500,
  "completed": true,
  "observation_sha256": "3909ad304d6eb239390e5c4dc07f43a30003e8a1c5a4166a5a2387339353e3bb"
}
```

Original deterministic quality under unchanged earlier boundaries:
```json
{
  "status": "pass",
  "passed": true,
  "metrics": {
    "ncc": 0.9918829592143225,
    "nrmse": 0.02933090843188145,
    "forward_nrmse": 0.008693088412780653,
    "kernel_mass_error": 1.338995616606553e-07,
    "kernel_negative_mass": 0.0,
    "image_min": 0.001306101679801941,
    "image_max": 0.9999985694885254,
    "shift_row": -10,
    "shift_column": 15
  },
  "thresholds": {
    "ncc": {
      "comparison": ">=",
      "boundary": 0.8926569351953423
    },
    "nrmse": {
      "comparison": "<=",
      "boundary": 0.03213680413900485
    },
    "forward_nrmse": {
      "comparison": "<=",
      "boundary": 0.00935309964438163
    },
    "kernel_mass_error": {
      "comparison": "<=",
      "boundary": 1e-05
    },
    "kernel_negative_mass": {
      "comparison": "<=",
      "boundary": 1e-08
    },
    "image_min": {
      "comparison": ">=",
      "boundary": 0.0
    },
    "image_max": {
      "comparison": "<=",
      "boundary": 1.0
    }
  },
  "errors": []
}
```

Fresh standard CLI vs independently frozen original:
```json
{
  "reconstruction": 0.0,
  "kernel": 0.0
}
```

Notebook execution/figure capture:
```json
{
  "executed_code_cells": 7,
  "headings": [
    "## 1. Problem",
    "## 2. Data",
    "## 3. Physics",
    "## 4. Method",
    "## 5. Results",
    "## 6. Conclusion"
  ],
  "figure_paths": [
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\figure-00.png",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\figure-01.png",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\figure-02.png",
    "D:\\yan1\\agent\\AutoSkill\\imaging-101-maintainer\\research10\\task01\\figure-03.png"
  ]
}
```

Visual figure inspection and final semantic/file-inventory review remain **unverified** until a reviewing agent records actual inspection. Successful computations are not a claim that every completion gate passed.

<!-- END EXECUTION EVIDENCE -->

## Archive index and redistribution attribution

Private root: imaging-101-maintainer/research10/task01/ (outside final task).
Source archive/selfdeblur.zip and source/ hold the exact original code; paper.pdf,
paper.html,paper.txt preserve the published version; realblur-prefix64MB.tar.gz,
member/extraction manifests and realblur-prefix-extracted/ preserve acquisition.
oracle_nondeterministic/ and oracle.log retain the investigated historical run.
oracle/, oracle_deterministic.log, capture_all_gradients.py and frozen_acceptance*
identify final original execution and acceptance. oracle_run.py is the original
capture orchestrator, not shipped runtime. Private diagnostic scripts/logs retain
the exact-gradient and nondeterminism investigation. complete_driver.py/status.json
record sequential commands, native exit statuses, runtimes and failures. Logs,
temporary test outputs, executed notebook and extracted figure PNGs stay private.

Method code permission: no explicit upstream license was present at the verified
SelfDeblur revision. Its source is kept only in the private verification archive.
The shipped implementation was independently written from the specified
mathematics and tensor architecture, with explicit method attribution; no original
framework checkout or source file is redistributed. This is not a claim of an
upstream license grant. Full copyright/attribution terms for the redistributed
RealBlur dataset are preserved below, and the public README credits the creators,
paper, official dataset and the luminance-format adaptation.

RealBlur creators: Jaesung Rim, Haeyun Lee, Jucheol Won and Sunghyun Cho.
Dataset: Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms,
ECCV2020. Source https://huggingface.co/datasets/rimchang/RealBlur ; official
repository README: https://github.com/rimchang/RealBlur states CC BY4.0.
Changes: full-raster integer RGB-to-Y conversion and NPZ packaging only; hidden
independent exposure receives the same conversion. No author endorsement implied.

### Original data license text

The complete CC BY4.0 legal code is retained here instead of a standalone
LICENSE/NOTICE file. Source https://creativecommons.org/licenses/by/4.0/legalcode.txt

```text

Attribution 4.0 International

=======================================================================

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
    wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More considerations
     for the public:
    wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution 4.0 International Public License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution 4.0 International Public License ("Public License"). To the
extent this Public License may be interpreted as a contract, You are
granted the Licensed Rights in consideration of Your acceptance of
these terms and conditions, and the Licensor grants You such rights in
consideration of benefits the Licensor receives from making the
Licensed Material available under these terms and conditions.


Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.

  d. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  e. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  f. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  g. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  h. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  i. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  j. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  k. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.


Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part; and

            b. produce, reproduce, and Share Adapted Material.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties.


Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.

       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

       4. If You Share Adapted Material You produce, the Adapter's
          License You apply must not prevent recipients of the Adapted
          Material from complying with this Public License.


Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material; and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.


Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.


Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.


Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.


Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.


=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.


```

### Initial cryptographic source inventory

- `selfdeblur.zip` SHA256 `b5c62da02f400e9903f86ccef78c59a47207dfd5e3daf19998ec23445980ff60`; 2967471 bytes.
- `paper.pdf` SHA256 `208580660ea83b6ab2e0c42c541c7c156232a4574e88af5a7e34a34faf70f9e7`; 4130626 bytes.
- `realblur-prefix64MB.tar.gz` SHA256 `6fa2f520a87f25f50faaf5e88d55abde8328ff0b021b461a14886a2fdc989e08`; 67108864 bytes.
- `blur_3.png` SHA256 `3909ad304d6eb239390e5c4dc07f43a30003e8a1c5a4166a5a2387339353e3bb`; 462794 bytes.
- `gt_3.png` SHA256 `52ab82cbccf960aa1a4348b68e0a2c4076f642908952811a55c5fdc899fcf588`; 548355 bytes.
- `raw_data.npz` SHA256 `63624f69f2d24636de9aa77771982aba1ca0b31f8b132282effce94a5c96f5e9`; 253005 bytes.
- `ground_truth.npz` SHA256 `3320cd0bc656b681684a4bfa276d310d208d78b6eed2097afe0f4faf574da84b`; 360052 bytes.

Environment snapshot: {"os": "Windows-10-10.0.26200-SP0", "python": "3.11.11", "torch": "2.9.1+cu128", "numpy": "2.4.6", "cuda": "12.8", "gpu": "NVIDIA GeForce RTX 4080 SUPER", "driver": "610.88", "device_memory_bytes": 17170956288}


## Independent final maintainer acceptance

This final review supersedes the explicitly historical review-pending state in the automated execution capture. All mandatory local-cleaning gates are **passed**. The full original 2500 updates, fresh CLI, fresh full CLI integration test (34 tests passed in773.37s), and complete active notebook have completed successfully. Fresh image and kernel match the deterministic independent original arrays with maximum absolute difference0. The notebook contains7executed code cells and4actual embedded scientific figures, all visually inspected by the coordinating maintainer: complete measured exposure, independent-exposure/original/fresh/error comparison, kernels/loss transition/unaligned forward residual, and metric bars/common profile. Labels, units, gauge and valid domains agree with the scientific contract.

The coordinating maintainer semantically reviewed the acquisition cross-correlation, latent/submission domains, SSIM factors, neural parameterizations, initialization/update/saved-before-update order, image gauge, coupled forward criterion, declared APIs and actual CLI/notebook flow against both plans and README. The structural audit found no errors, public GitHub leaks, temporary artifacts or disallowed imports. Required source/data attribution remains in this file/private archive; standalone LICENSE/NOTICE are absent. Numeric fixture consumers are identified above.

An independent standalone Judge invocation from the external coordinator working directory passed the fresh notebook submission without solver imports. Its exact report is privately archived as `../coordinator/blind-root-independent-judge.json`; the final package inventory/hashes are `../coordinator/blind-final-review.json`. The earlier UTF8 subprocess-decoding failure and status-writer interruption remain documented as historical failures; the successful full retry changed neither physics nor acceptance thresholds. Full CPU reconstruction is not claimed: the declared full runtime requires CUDA, while operator/metric CPU checks pass.
