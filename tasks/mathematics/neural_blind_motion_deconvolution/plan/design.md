# Shipped interfaces and execution design

## 1. Scope, ownership and actual caller graph

This package implements the full joint dual-prior reconstruction of the bound
RealBlur-J observation. The runtime imports only NumPy, Torch and optional tqdm;
review plotting/metrics additionally use Matplotlib and SciPy. It never imports
the original repository, OpenCV, a pretrained reconstruction checkpoint, another
task, or an imaging-domain library. `src/main.py` is the sole standard executable
entry point: `python -m src.main`. There is no root wrapper or metadata JSON.

The actual flow is:

```text
main(argv)
  -> preprocessing.load_observation(input NPZ)
  -> solvers.reconstruct(float32 image, iterations=2500, device='cuda')
       -> initialize_priors -> configure_backend
       -> image generator + kernel generator
       -> physics_model.blur_image -> loss_for_step -> structural_similarity
       -> backward graph + joint Adam (2500 iterations)
       -> ReconstructionResult(center image, kernel, trace, elapsed)
  -> np.savez_compressed(output, reconstruction=..., kernel=...)

notebook (active independent reconstruction)
  -> same load_observation and reconstruct
  -> write its fresh submission to output/reconstruction.npz
  -> load identified upstream result and actual independent short exposure
  -> visualization metrics, comparisons and figures
  -> standalone evaluation/judge.py subprocess

fresh pipeline integration test
  -> new CLI subprocess and temporary output (complete 2500 steps)
  -> standalone Judge subprocess
  -> independent frozen upstream image AND kernel parity
```

The CLI and solver do not read notebooks, plans, fixtures or evaluation assets.
The notebook reads them only after computing a fresh reconstruction. There is no
cache-preview mode in the shipped notebook and no code that silently substitutes
a reference result for an unsuccessful reconstruction.

## 2. File and array contracts

| Location | Required keys and representation | Producer and consumers |
|---|---|---|
| `data/raw_data.npz` | `observation`: (1,753,668), float32 normalized luminance | Maintainer packaging of full measured exposure; CLI/notebook/Judge |
| `output/reconstruction.npz` | `reconstruction`: (1,753,668), float32; `kernel`: (1,51,51), float32 | CLI or notebook fresh run; Judge |
| `output/judge_report.json` | status, passed, metrics, thresholds, errors | Standalone Judge; notebook/reviewer |
| `evaluation/ground_truth.npz` | `reconstruction`: (1,753,668), float32 | Independent short exposure; Judge and maintainer notebook only |
| `evaluation/metrics.json` | measured upstream/identity provenance and numeric boundaries | Frozen acceptance; standalone Judge and maintainer review |
| `notebooks/reference_outputs/upstream.npz` | `reconstruction` (1,753,668), `kernel` (1,51,51), `predicted` (1,753,668), `history` (2500,4) | Original numerical code on exact observation; plots and full parity test |
| `test/fixtures/upstream_stages.npz` | complete first neural pass, perturbation, both gradients/updates, separate SSIM input/value/gradient | Private independent oracle capture; comparing tests |

Observation, reconstructed image, latent and predicted intensity are dimensionless
fractions of processed 8-bit full scale. Kernel entries are dimensionless weights.
Rows/columns are pixel coordinates with downward/rightward positive directions.
No distance, angular, exposure-time or radiance units are inferred from missing
camera metadata. NPZ loading always uses `allow_pickle=False`. Raw and truth are
float32 stored values; metrics deliberately promote arrays to float64.

The internal latent has shape (1,1,803,718). Its 25-pixel center crop is the
submitted (753,668) image. The optimizer correlates the complete latent with the
51x51 kernel, yielding the complete (753,668) predicted measurement. By contrast,
the Judge can only correlate the submitted center image, yielding (703,618),
which matches the public observation's `[25:-25,25:-25]` region. This smaller
check is explicit and never feeds back into the optimization objective.

## 3. Preprocessing API and validation boundary

`load_observation(path='data/raw_data.npz') -> numpy.ndarray`

- Accepts string/path-like NPZ path, relative to the caller's working directory.
- Requires exactly the key `observation`, dtype float32 and shape (1,753,668).
- Rejects nonfinite values and values outside [0,1] with ValueError. Missing files
  and invalid archive/container errors propagate from the filesystem/NumPy.
- Removes exactly one declared batch dimension and returns a copied (753,668)
  float32 array. It does not reshape, cast, normalize, resize or mutate input.
- Constants are `IMAGE_SHAPE=(753,668)`, `KERNEL_SIZE=51`, and
  `QUANTIZATION_STEP=1/255`. The README constants are checked against source.

Only maintainer packaging converts the source PNG's RGB pixels to integer
luminance. OpenCV is confined to the archived oracle environment; its integer
conversion is independently compared with the explicit arithmetic used for
packaging. No RGB decoder is needed by this runtime. Public `raw_data.npz` is
self-sufficient when combined with README acquisition conventions.

## 4. Forward and discrepancy APIs

`blur_image(latent, kernel) -> torch.Tensor`

- Inputs are real floating BCHW Torch tensors on the same device and dtype.
  The validated path uses (1,1,803,718) and (1,1,51,51), float32 CUDA.
- Explicitly validates four dimensions and single image/kernel channel. Channel
  and batch dimensions must be (1,1); invalid layouts raise ValueError.
- Calls the permitted general tensor cross-correlation primitive with no padding,
  no bias, unit stride and unit dilation. Result is (1,1,753,668).
- It does not silently reflect or wrap the acquisition input. Kernel positivity,
  sum and finite-value conditions are generator guarantees/evaluation checks,
  not repeated hidden coercions inside this primitive. Invalid device/dtype or
  too-small dimensions produce Torch's ordinary exceptions.
- It preserves autograd connectivity and does not mutate either tensor.

`structural_similarity(a, b) -> scalar torch.Tensor`

- Requires equal BCHW shapes and one channel; rejects layout mismatch with
  ValueError. Benchmark inputs are (1,1,753,668), float32 CUDA. Tests also use
  small float64 CPU arrays for derivatives.
- Builds 11 float32 Gaussian samples on CPU using sigma 1.5, normalizes their
  sum, forms the outer-product window and transfers it to the input device/dtype.
- Computes five zero-padded window moments using general `conv2d`, then the
  explicit mean SSIM quotient with C1=.0001 and C2=.0009. There is no hidden
  `skimage` or other ready-made SSIM call. The operation does not clamp variances.
- Output is a scalar on the input device/dtype with gradients for both inputs.
  No image is saved and no caller tensor is modified. Extreme/nonfinite inputs
  are not silently sanitized; reconstruction checks objective finiteness.

`loss_for_step(prediction, observation, step) -> scalar torch.Tensor`

- `step` is the zero-based optimizer iteration. Negative values raise ValueError.
- For step<500 use mean squared error; for step>=500 return 1-SSIM. Both inputs
  retain the complete measurement domain. No step-dependent crop or scaling is
  introduced. Layout/device errors follow from the called mathematical routines.
- No state, regularizer or optimizer is stored in this function.

## 5. Network building blocks and exact signatures

These modules are explicitly written task-local tensor models. Their mathematical
roles are described in approach.md; there is no vendored opaque imaging framework.

`conv(cin, cout, size=3, stride=1) -> nn.Sequential`

Returns ReflectionPad2d(size//2) followed by a biased Conv2d(cin,cout,size,stride)
with zero internal padding. The selected architecture calls it with size 1 or 3,
and stride 1 or 2. General positive dimensions are preconditions; unsupported
arguments are rejected by Torch. It allocates new parameters and does not share
weights across calls. It is a feature-network helper, not the blur operator.

`norm_act(channels) -> nn.Sequential`

Returns BatchNorm2d(channels) with default epsilon 1e-5, momentum .1, affine=True,
track_running_stats=True, followed by LeakyReLU(.2,inplace=True). The activation
can mutate its intermediate feature tensor. Running BN buffers change during
training; external observations/noise are not mutated.

`SpatialInteraction()`; `forward(self, x) -> torch.Tensor`

Input/output are (B,128,h,w). The validated three calls have h,w large enough for
2x2 max pooling. Biased 1x1 maps create 64-channel values, queries and keys. Keys
and values are pooled; query-key dot products are divided by the pooled site
count and multiply the values. A 64->128 projection plus BN is added to x, with
the projection BN scale/bias initially zero. The four convolution parameter
groups are allocated values, projection, queries, keys, preserving initialization
draw order. `forward` never recenters the physical kernel. Shape/channel and small
pooling errors are Torch errors rather than undocumented broadcasting.

`Scale(cin, index)`; `encode(self, x) -> (side, deeper)`;
`decode(self, side, deeper) -> torch.Tensor`

- `index` is zero-based 0..4 in the shipped network, not a public batch dimension.
- `encode` accepts (B,cin,h,w); returns 16-channel skip features at (h,w) and
  128-channel deeper features at (ceil(h/2),ceil(w/2)). The nonlocal block is used
  only for index>1. Feature convolutions use reflection padding.
- `decode` bilinearly upsamples its 128-channel deeper input by 2 with
  align_corners=False, selects the common minimum spatial dimensions, and centers
  both branches using floor-divided offsets. It concatenates 16 skip channels
  before 128 deeper channels, applies 144-channel BN, then the two explicit
  decoder convolution/normalization/activation groups.
- The nested `centered(a)` helper is local to decode and slices a BCHW tensor;
  it is not an externally callable image-registration method. At odd surplus
  sizes, the bottom/right surplus is excluded according to integer floor rules.
- No file I/O occurs. BN buffers and intermediate activations behave as above.

`ImagePrior()`; `forward(self, z) -> torch.Tensor`;
`optimization_parameters(self) -> parameter iterator`

`z` has (1,8,803,718), float32 CUDA. Five Scale objects are allocated top-down,
using cin=8 at level 0 and 128 thereafter. `forward` actively encodes every level,
decodes all levels in reverse, and applies the final 128->1 sigmoid. Output is
(1,1,803,718) in (0,1). A list of feature skips exists only for that call. It does
not load observations or checkpoints. The parameter iterator yields side/down/
nonlocal/refinement groups top-down, fusion/decoder groups bottom-up, then
readout, matching the reference optimizer traversal despite independent class
names. This avoids backend-dependent optimizer grouping differences.

`KernelPrior(size=51)`; `forward(self, z) -> torch.Tensor`

The validated `z` is a real 200-vector; two biased Linear maps have dimensions
200->1000 and 1000->size^2. ReLU6 is applied between them. Softmax along the final
axis is reshaped row-major to (1,1,size,size). The validated size=51 produces
2601 nonnegative weights with sum one to float32 rounding. Other positive sizes
are possible for analytic tests but do not change the benchmark output contract.
Wrong feature length/device/dtype produces a Torch exception. No model weights
are pretrained or shared with another task.

## 6. Reconstruction API, state and scheduling

`configure_backend(seed=42) -> None`

Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` if absent, before first model CUDA work;
sets Torch CPU threads to 4 and seeds Torch; disables cuDNN benchmarking and both
CUDA-matmul/cuDNN TF32 modes; enables cuDNN deterministic selection **and**
`torch.use_deterministic_algorithms(True)`. These are global process-side effects
and are documented. CuDNN selection alone leaves some CUDA reflection-padding
backward paths nondeterministic. The discrepancy was independently reproduced
and archived before switching the final oracle/runtime to the stronger backend.
No plotting backend is changed. An incompatible preexisting cuBLAS workspace
configuration is not silently replaced and may produce a Torch runtime error.

`initialize_priors(shape, device='cuda', seed=42) -> (image, kernel, z, zk)`

`shape` is an (H,W) pair, validated by its caller for the benchmark. After backend
configuration, creates image input U[0,.1) at (1,8,H+50,W+50) on CPU then transfers
it, allocates/transfers ImagePrior, creates/transfers a 200-vector kernel input,
and allocates/transfers KernelPrior. Outputs are newly allocated models/tensors.
Noise tensors do not require gradients. This ordering is checked against every
independently captured initial parameter in the private diagnostic run. Calls
modify RNG state; no target or hidden seed is consumed.

`learning_rates(step) -> (float, float)`

Returns (.01,.0001)*.5^d, where d counts satisfied milestones 1600,1900,2200.
The standard caller supplies an integer in 0..2499. This pure function does not
advance an opaque scheduler state or mutate optimizer groups itself.

`reconstruct(observation, iterations=2500, seed=42, device='cuda', progress=True)`
`-> ReconstructionResult`

- `observation` must be a finite 2-D float32 NumPy array in [0,1]. Invalid dtype,
  dimensionality/range/finiteness or a nonpositive/nonintegral iteration count
  raises ValueError. The public loader enforces the exact benchmark shape.
- CUDA availability is checked for device strings beginning with `cuda`; absence
  raises RuntimeError. CPU execution of primitives is validated, but a complete
  CPU reconstruction is not claimed. Torch handles unsupported devices or sizes.
- Creates new priors, fixed inputs, target tensor and joint Adam optimizer. The
  optimizer uses defaults beta=(.9,.999), eps=1e-8, zero weight decay and no AMSGrad.
  The image group follows `optimization_parameters`; kernel parameters form the
  second group. No shared optimizer state survives a call.
- Each loop perturbs only the image input, assigns both explicit rates, clears
  gradients, generates the latent/kernel, valid-correlates, applies a centered
  measurement-shape slice (zero offset on this benchmark), computes the current
  objective, checks finiteness, backpropagates, updates and records history.
  A nonfinite objective raises FloatingPointError identifying the iteration.
- Optional tqdm progress writes to stderr; it does not change computations.
  Short iteration counts exist only to test numerical stages. Standard CLI,
  active notebook and integration test always run all 2500 iterations.
- The returned center image/kernel come from the last forward pass before its
  optimizer update, following the original save convention. There is no final
  unperturbed reevaluation, best-reference selection, phase correction or crop
  inferred from the target. CUDA is synchronized before timing completes.
- Input NumPy observation is not overwritten. The process RNG/backend flags,
  allocated networks and BN buffers change during computation. No file I/O occurs.

`ReconstructionResult` is a dataclass with fields:

| Field | Shape/type | Meaning |
|---|---|---|
| `reconstruction` | (H,W) float32 ndarray | Copied 25-pixel center crop of final latent |
| `kernel` | (51,51) float32 ndarray | Copied final cross-correlation weights |
| `history` | (iterations,4) float64 ndarray | Zero-based step, objective, image rate, kernel rate |
| `runtime_seconds` | Python float | Wall time including setup and synchronized device computation |

## 7. CLI and visualization APIs

`main(argv=None) -> int` in src/main.py parses optional `--input`, `--output`, and
`--device`, defaulting respectively to data/raw_data.npz,
output/reconstruction.npz and cuda. Paths are relative to the caller's CWD. It
loads the observation, runs complete default reconstruction, creates the output
parent, adds one batch dimension to each array and writes a compressed NPZ. It
prints elapsed time/path and returns zero. Solver/load/write exceptions propagate
as a nonzero process exit; there is no fallback output or swallowed failure.
Importing this module does not select a Matplotlib backend or start a run.

`align_translation(image, reference) -> (aligned, ref_interior, (dy,dx))`

Requires equal 2-D shapes larger than 100 pixels. Converts to float64, evaluates
full FFT cross-correlation of demeaned arrays, and searches the center +/-25
integer displacement box using row-major argmax. Returns two fixed 50-pixel
interior views (653,568 for the benchmark) and the selected offset. It does not
rescale or interpolate either image. Degenerate metric handling is in its caller.

`image_metrics(image, reference) -> dict`

Uses that alignment; returns `ncc`, `nrmse`, `shift_row`, `shift_column`. NCC is
unsubtracted cosine similarity; NRMSE uses the reference interior dynamic range.
Zero norm or zero reference range raises ValueError. These are explicit
maintainer metrics; the independent Judge separately implements the same formulas.

`forward_nrmse(image, kernel, observation) -> float`

Computes valid correlation as FFT convolution with the reversed kernel, promotes
to float64, and compares with observation[25:-25,25:-25]. It does not align either
submitted quantity. The validated caller supplies contracted shapes and
nondegenerate observation; these are preconditions rather than hidden coercions.

`comparison_figure(observation, reference, upstream, fresh) -> matplotlib.Figure`

Draws four full images and their four aligned fixed-interior residual panels,
with common full-scale display and fixed residual limits. Metric values are
computed, not copied from historical reports. Caller owns saving/closing.

`diagnostic_figure(result, upstream_kernel, observation) -> matplotlib.Figure`

Draws independently reproduced/fresh kernels, both distinctly labeled loss stages
and the unaligned valid forward residual. It explicitly labels the MSE/1-SSIM
switch; the two loss scales are not presented as one unchanging objective. It
does not write files or alter the plotting backend.

## 8. Independent evaluator and test/data flow

The Judge imports only standard library, NumPy and SciPy. `HERE` is its resolved
evaluation directory. Hidden assets resolve relative to HERE; CLI paths remain
relative to caller CWD. It never runs or imports submitted code or the reference
solver. `read_prediction(path)` requires exactly both declared keys, exact
batch-first shapes, float32 dtype and finite values; no complex conversion,
reshape, magnitude, center crop or filename guessing occurs.

`score_arrays(image,kernel,truth,observation)` uses separate float64 metric code:
one image-reference integer alignment, declared fixed image interior, original
unaligned valid forward residual, kernel unit-sum error/negative mass and image
extrema. `evaluate(prediction)` loads the boundary file and actual truth, validates
hidden array shape/dtype/finiteness, checks finite thresholds/scores and returns
(report,code). Every metric boundary must pass. Malformed inputs or evaluator
failures yield status error, empty metrics and descriptive errors; valid low
quality returns fail and code 1; pass returns code 0. JSON serialization disallows
NaN/Infinity. `main(argv=None)` creates report parents and returns code 2 plus a
stderr diagnostic on report-write failure.

Unit tests cover raw-contract validation, true discrete correlation signs and
adjoints, finite-difference image and SSIM derivatives, blind translation gauge,
kernel positivity/simplex, all learning-rate transitions, metric numerical cases
and every Judge error/fail path. Comparing tests consume frozen **original-run**
arrays; they do not import archived upstream code. The full pipeline fixture runs
one complete 2500-step CLI in a fresh temp directory, independently judges its
new file, and compares both quantities to frozen upstream values. Timeout is
5400 seconds, justified by the stronger deterministic backend's measured cost.

The notebook uses the same local APIs and complete benchmark, evaluates its own
new output and displays independent truth, identity blur, original upstream and
fresh cleaned results. No temporary executed notebook or figure PNG belongs in
the final task; private execution logs/figure captures are indexed in validation.
All fixtures and reference arrays have their producer, hash, keys and consumer
recorded there. Licensing/data provenance is in validation rather than standalone
LICENSE/NOTICE files. No solver-visible material exposes upstream repository URLs.
