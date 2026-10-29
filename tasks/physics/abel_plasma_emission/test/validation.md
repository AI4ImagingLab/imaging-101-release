# Validation and provenance: Abel plasma emission

Validation date: 2026-10-18 (Asia/Shanghai). **All mandatory gates listed below
passed for the bound synthetic acquisition on Windows CPU.** No GPU run or
other operating system has been tested. This ledger is maintainer-only.
The legacy evaluation harness has not been migrated or claimed compatible.

## 1. Website search, choice and fixed scope

Public website retrieval used Python urllib HTTPS requests, not an unavailable
browser/search connector. Search requests and returned repository metadata are
stored in `website-search.json` in the external archive. Queries were
`Abel inversion PyAbel`, `traction force microscopy pyTFM`, and
`quantitative susceptibility mapping reconstruction`. Direct verification of
the PyAbel repository, tag API, paper, license and implementation resolved the
first query's incomplete repository search results. Search metadata alone was
not used as scientific evidence.

| Candidate | Domain | Retrieved website | Selection status |
|---|---|---|---|
| Abel emission reconstruction / PyAbel | Physics | https://github.com/PyAbel/PyAbel ; https://arxiv.org/abs/1902.09007 | Selected: inspectable explicit recursion, MIT source, analytic independent physics checks and CPU execution |
| Traction force microscopy / pyTFM | Mechanics | https://github.com/fabrylab/pyTFM | Metadata screened only; data/paper/license not fully verified in this search; not selected |
| Quantitative susceptibility mapping / INR-QSM | Medicine | https://github.com/AMRI-Lab/INR-QSM | Metadata screened only; full training/data requirements not verified; not selected |

The instruction was interpreted as choosing **one** task from the eight named
domains. It did not require eight cleaned tasks. Repository task-directory
inspection found no existing Abel-emission task. No generic image enhancement
problem was substituted for the physical inverse problem.

Fixed before local implementation: unknown emission f(r,z), optically thin
axisymmetric line-integration physics, selected Hansen-Law first-order hold
method, the exact bound synthetic observations, float64 output of shape
(1,65,257) in radiance/mm, and upstream-derived metric boundaries. Synthetic
data are used because no experimental dataset with verified suitable
redistribution rights was established in this search. No experimental-fidelity
claim is made.

## 2. Upstream revision, paper and rights

- Repository: https://github.com/PyAbel/PyAbel .
- Remotely verified tag: `v0.9.1`, commit
  `f16723713e118a2eeef9bc7ba17efd57ad074093`. The live GitHub ref API returned
  this commit; `revision.json` preserves the response. This is a selected
  release, **not a claim that the repository's current HEAD is this revision**.
- Source file executed unchanged: `abel/hansenlaw.py` in the archived checkout.
  SHA256 `57a8bf4cc385ea772fc2c96f455eb98b52e99d08fac52899b114dfefb1bd6902`.
- Main inspected paper: D. D. Hickstein et al., *A direct comparison of
  high-speed methods for the numerical Abel transform*, arXiv
  **1902.09007v1**, submitted 2019-02-24; published RSI 90, 065115 (2019),
  DOI 10.1063/1.5092635. PDF, version page, machine-rendered HTML and extracted
  text are archived. A request for v2 returned HTTP 404; the version page lists
  only v1. The regenerated HTML's displayed date differs from the submission
  history; the identifier/version and PDF hash identify the inspected source.
- Historical algorithm references: Hansen and Law, JOSA A 2, 510-520 (1985),
  DOI 10.1364/JOSAA.2.000510; Hansen, IEEE TASSP 33, 666-671 (1985), DOI
  10.1109/TASSP.1985.1164579. The local equations follow the validated modern
  implementation and its explanatory documentation. No claim of running the
  original 1985 software or reproducing its original experiment is made.
- Code license: MIT, copyright (c) 2015 The PyAbel Team; original license is
  retained verbatim in the consolidated license section below, with attribution in the adapted modules.
  Vectorization and first-order hold history are preserved in module comments.
- Paper license: arXiv perpetual non-exclusive distribution license. Paper
  files remain in the maintainer archive and are not redistributed in the task.
  Bound observations and actual truth were generated for this task; they are
  not third-party experimental data or target-derived calibration.
- Cleaned repository base: `e2bdd145cc4335e34b019ba091b2f7e5f8a82856`, dirty
  user worktree. New task files are uncommitted. File hashes in the release
  inventory and `cleaned_measurements.json` identify the actual cleaned code;
  the base commit is not mislabeled as containing this implementation.

## 3. Bound data, generator and hidden truth

Public NPZ has **only** key `projection`, float64, shape (1,65,257), radiance
units, with axes (batch,z,y). `evaluation/ground_truth.npz` contains actual
analytic synthetic emission under key `reconstruction`, float64, same shape
but axes (batch,z,r), radiance/mm. It is not an upstream reconstruction.
No `baseline_reference.npz` is supplied because actual simulation truth exists.

Hidden generation recipe: r(j)=j*0.025 mm; z(i)=(i-32)*0.1 mm. For widths
w=(1.5,0.7,0.42) mm, emission is a sum over k of
A(k,z)*exp(-(r/w(k))^2), with
A(0,z)=0.9*exp(-(z/2.3)^2),
A(1,z)=-0.65*exp(-(z/2.3)^2), and
A(2,z)=0.5*exp(-((z-1.1)/0.65)^2).
The first two components form a nonnegative hollow profile; their sum plus
the localized core remains nonnegative. The independent analytic projection
replaces each component with sqrt(pi)*w(k)*A(k,z)*exp(-(y/w(k))^2).
Noise uses NumPy default_rng(1407), independent N(0,0.002^2) radiance noise.
Widths use the 1/e Gaussian convention, not standard deviations.

Private `capture_upstream.py` created these observations and truth using the
analytic formula before any cleaned reconstruction module existed, then
executed the original transform on precisely these observations. Local
`src/generate_data.py` is the executable, documented generation implementation.
Independent x-line quadrature, Gaussian pairs, physical scaling, and flux
conservation test the formula rather than relying on two matching copies.
Stochastic tests check moments, not equality of seeded noise samples.

| File | SHA256 |
|---|---|
| `data/raw_data.npz` | `ed834af32088018016348833b7ecff6025a9b151900a54a15ad2273e0606d78d` |
| `evaluation/ground_truth.npz` | `cbd8d0ffcfd85de2c6ab1340279e2253c291153633da8ee305e0d6380af2ddbd` |
| `test/fixtures/upstream_stages.npz` | `1a28834d7392d6c8a3f1138839eb9f29827d7428ee909f2fd96e9a69225a497f` |
| `notebooks/reference_outputs/upstream_reconstruction.npz` | `c0a77334817ccc1e05cba7f282c22b409c79e84d6d5e90a0cec0a5913997a705` |
| `notebooks/reference_outputs/cleaned_reconstruction.npz` | `85a17660c937a917e4f4bdd52d51bdc09afbfbd0435eada832b7c04ae91f8e4c` |
| `evaluation/metrics.json` | `2ddbc8210bdb27f0b8324961e3df3cf7ff94c2f881853dbaacfc8f2859f5d716` |

No dtype conversion was used to shrink data. Storage and computation remain
float64. No noiseless projection, seed, object coefficients, target, hidden
reference, privileged observation or target-derived calibration occurs in
the public NPZ/README/environment allowlist. The simulator's seed and object
definition are confined to hidden source/provenance. Acquisition constants
and units are copied literally into README and tested against code.

## 4. Original run and predeclared acceptance criteria

External archive root, relative to the repository root:
`../imaging-101-maintainer/abel_hansenlaw_20261018/`.
Its `archive-index.json` lists all retained verification artifacts with hashes
and consumers. It is outside the final task package.

Oracle command, from that archive:
`oracle-env/Scripts/python.exe capture_upstream.py`.
The script imports the **unmodified original file** by importlib and invokes
its `hansenlaw_transform`; it does not copy/reimplement the numerical method.
A return-event trace captures original derivative and coefficient arrays;
no upstream formula or function body is modified. The full package's optional
GUI/image/data imports are not needed for this single published transform.
All private capture code and checkout paths are absent from the cleaned runtime.

Oracle parameters: dr=0.025 mm, inverse first-order hold=1; zero-order inverse
and both forward orders also recorded. A signed, nonzero-edge (3,19) probe
uses seed 311 and dr=0.13 mm. Oracle runtime for the recorded transform set:
**0.009727800 s**, exit 0; transform warning list empty.
Independent repeat max absolute difference: 0.0.
The later NumPy backend-report helper emitted an advisory to install pyyaml
for prettier configuration output; it did not affect the transform and no
extra package was installed for it. Original numeric results are archived
and frozen as the stage fixture, separately from cleaned results.

Before local implementation, `upstream-run.json` fixed stage/pipeline parity
rtol=1e-12, atol=1e-12; Gaussian forward and inverse relative L2 limits 0.015;
independent line-quadrature absolute tolerance 1e-6 radiance units. Float64
well-conditioned scalar/power operations justify these parity tolerances.
The selected backend yields zero maximum local/upstream difference on the
bound acquisition; the tests still use numerical tolerances, not a universal
bitwise-parity claim. Noise-weighted error checks use statistical ranges.

## 5. Equation/stage -> upstream -> local -> validation traceability

| Required equation or stage | Upstream location | Local implementation | Independent/frozen validation |
|---|---|---|---|
| 2019 paper Eq. 1: line-of-sight Abel forward model | paper Eq. 1; `abel/hansenlaw.py:hansenlaw_transform(direction='forward')` | `physics_model.forward_transform`; analytic `generate_data.synthetic_fields` | bound forward outputs, signed probes, Gaussian analytic pair, x-line quadrature and flux conservation |
| 2019 paper Eq. 2: inverse Abel integral | paper Eq. 2; inverse branch of `hansenlaw_transform` | `solvers.inverse_transform`, `physics_model.propagate` | bound inverse outputs and analytic Gaussian inverse |
| Pixel-centered integer radial convention | paper section III.2 | `preprocessing.DR`, generator arange grid | Gaussian absolute-scale test, literal metadata test, no half-pixel offset |
| Projection derivative, first/zero-order options | `hansenlaw_transform`: gradient/forward difference drive | `solvers.inverse_drive` | frozen `output_drive`; both inverse orders and signed probes |
| Nine-state kernel constants | `hansenlaw_transform`: h, lam | `physics_model.H`, `LAMBDA` | coefficient arrays and signed-probe forward/inverse parity |
| Signed interval integral and Phi/B0/B1 construction | nested `I`; phi and gamma expressions | `state_integrals`, `state_coefficients` | traced frozen `output_phi`, `output_B0`, `output_B1`; full directional/order probes |
| First-order recurrence and zero-state initialization | state loop in `hansenlaw_transform` | `physics_model.propagate` | final original outputs, linearity and no-input-mutation semantics |
| Jacobian and mm unit factors | forward `-2*dr*pi`; inverse derivative `/dr` | `forward_transform`, `inverse_drive` | analytic absolute scale; doubling spacing doubles forward/halves inverse |
| Adjacent endpoint copies | final two upstream assignments | final `propagate` assignments | explicit endpoint identities and nonzero-edge signed oracle probe |
| Batch packaging / lossless input | task wrapper, no scientific upstream equivalent | `load_projection`, `reconstruct`, `main.run` | exact observation fixture equality, pipeline NPZ key/shape/dtype test |
| Actual simulation truth and noise | analytic task-specific generation, not an upstream cached report | `synthetic_fields`, `generate` | quadrature, positive truth, noise moments and chi-square |
| NCC / dynamic-range NRMSE | task metrics, not relabeled upstream error definitions | independent Judge and `visualization.comparison_metrics` | known vectors NCC=0.8, NRMSE=sqrt(10/3)/5; Judge/Notebook agreement |

The method requires no adjoint, optimization loss/gradient, proximal operator,
neural network, learning rate or stochastic inverse update. Those stages are
not applicable, rather than silently untested required mathematics.

### Investigated discrepancies and deliberate choices

1. Upstream defaults to hold_order=0; this task explicitly selects and passes
   hold_order=1. Upstream/local zero-order results remain an active notebook
   comparison. No default-option mismatch is hidden as a reproduction.
2. Archived `doc/transform_methods/hansenlaw.rst` says B1=0 in its zero-order
   prose, while the executable code sets B0=0 and B1=Gamma0. The local method
   retains the validated **code** convention. Direct inspection of the state
   integral shows that a constant interval drive removes the outer-drive
   coefficient B0 under the code's indexing; signed probes and analytic
   Gaussian checks independently support this convention. No numerical
   upstream error was copied solely because its documentation agreed.
3. The modern release copies both adjacent endpoint values; it neither uses
   the latest master's new background option nor sets the last output to zero.
   This pinned convention is preserved and tested. Axis copying produces a
   small approximation error visible in the analytic residual panel. No
   local correction or post-hoc crop was applied to conceal it.
4. This benchmark uses independently analytic synthetic projections, not a
   numerical forward/inverse roundtrip as its sole truth evidence. Original
   paper experimental comparisons and eight-way algorithm speed claims are
   outside the fixed selected-method scope and are not claimed reproduced.
   No supplied notebook comparison group was removed; the notebook is new.

## 6. Numerical results, definitions and fixed thresholds

Reference for all task-quality metrics: actual analytic simulation emission,
all samples, absolute scale, no mask, mean removal, crop, shift or gain fit.
NCC=dot(u,v)/(norm(u)*norm(v)); NRMSE=sqrt(mean((u-v)^2))/ptp(v).
Zero output has NCC=0; constant/zero-norm references are evaluator errors.

| Measurement provenance | NCC | Dynamic-range NRMSE |
|---|---:|---:|
| Historical paper/cached report | Not used as bound-data evidence | Not used |
| Independent reproduced upstream, first order | 0.99982660898346831 | 0.0041198016585959757 |
| Fresh cleaned run saved with source hashes | 0.99982660898346798 | 0.0041198016585959757 |
| Reproduced upstream, zero order | 0.99904135319667986 | 0.0096640783903557716 |
| Fresh Notebook zero order | 0.99904135319667986 | 0.0096640783903557716 |

NCC threshold = 0.9*baseline_NCC = **0.8998439480851215** (>=).
NRMSE threshold = 1.1*baseline_NRMSE = **0.004531781824455574** (<=).
These values were written by the oracle capture **before** cleaned source
implementation; no threshold was relaxed after inspecting cleaned results.
The baseline error is nonzero and small, so this is not degenerate
self-comparison or a poor-baseline accommodation. Both metrics must pass.
Machine-readable files retain full floating-point precision. Tiny differences
in the last NCC bits between the Judge's stable normalized formula and the
visualization's direct cosine formula are below 1e-12 and checked explicitly.

Independent physical diagnostics from the executed Notebook:

- Gaussian forward relative L2 error: 0.0014181075628600806.
- Gaussian inverse relative L2 error: 0.0018207107299165032.
- Analytic bound projection vs numerical forward relative L2:
  0.0014665804678705146.
- Bound observation noise chi-square/sample:
  0.99746078278220451, consistent with unity.
- Positive simulated emission and cylindrical flux conservation pass.
  No positivity is artificially imposed on noisy inverse results.

## 7. Environment and reproducible execution

No suitable activated shared environment was found: default PATH Python was
3.14 without pytest; Anaconda base was Python 3.12 with older numerical
packages. Existing environments were not upgraded. A maintainer shared
environment was created outside the repo at
`../imaging-101-maintainer/shared-env` using Python 3.11.11 and the exact
project `environment.yml` pins required for this task and its development
checks. `requirements.txt` is the runtime subset: NumPy 2.4.6, Matplotlib
3.11.1. SciPy 1.17.1 is used only for independent test quadrature; pytest and
Jupyter remain development tooling. Pillow 12.3.0 matches the shared pin as
a Matplotlib dependency. No domain-imaging package is installed in this
shared cleaned environment (`find_spec('abel') is None`).

Execution OS: Windows 10 API build 26200, x86_64 CPU, CPython 3.11.11,
NumPy 2.4.6 with scipy-openblas 0.3.31.188.0, ILP64, dynamic architecture.
Threads: OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1.
SciPy 1.17.1, Matplotlib 3.11.1, pytest 9.1.1, nbformat 5.11.1,
nbconvert 7.17.1, ipykernel 7.3.0. Full dependency locks and NumPy backend
information are archived in `cleaned-environment.txt`, `oracle-environment.txt`
and `upstream-run.json`. There is no runtime package-version exception or GPU
dependency. No claim about untested Linux/macOS or CUDA compatibility is made.

Oracle isolation: separate `oracle-env` at the archive root, same Python/NumPy
versions, SciPy and six/setuptools retained for optional upstream verification
only. The unmodified single-module oracle itself uses NumPy. Oracle checkout
and capture scripts are never imported by cleaned source, generator, Judge,
tests or Notebook. Frozen numeric fixtures are their only parity dependency.

PowerShell environment for the commands (all paths are maintainer-only):

```powershell
$env:PATH = '<archive-parent>/shared-env/Scripts;' + $env:PATH
$env:TEMP = '<archive>/tmp'
$env:TMP = $env:TEMP
$env:JUPYTER_RUNTIME_DIR = $env:TEMP
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
cd tasks/abel_plasma_emission
python -m src.main
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v
jupyter nbconvert --to notebook --execute notebooks/abel_plasma_emission.ipynb --output abel_plasma_emission_executed.ipynb --output-dir output --ExecutePreprocessor.timeout=180
```

The shared environment has a prefix-local python3 kernelspec pointing at its
Python, not Anaconda base. Public files contain no machine-specific paths.

## 8. Executed checks, errors encountered and resolution

| Final command/check | Exit / outcome | Evidence |
|---|---|---|
| Original pinned function on bound observations | 0 | upstream-run.json, upstream-run.log, frozen numeric fixture |
| `python -m src.main` | 0; active reconstruction | cleaned-cli.log; wall 0.131874 s, internal run time in log |
| Standalone Judge on fresh output | 0, pass | judge.log; output/judge_report.json |
| `python -m pytest test/ -v --tb=short` | 0; **52 passed** | pytest.log; wall 2.999752 s |
| `jupyter nbconvert --execute` full Notebook | 0; all 5 code cells executed, no error outputs | notebook.log, release-verification.json; wall 3.389288 s |
| Notebook first-order reconstruction | computed, 0.002010800 s | executed review Notebook stream output |
| Notebook zero-order reconstruction | computed, 0.001598200 s | executed review Notebook stream output |
| Import/call-role audit | passed; 15 source files/code cells | symbol-call-audit.json; review below |
| Public input leakage and raw-key audit | passed | release-verification.json; metadata unit test |
| Figure visual inspection | passed; four figures | external review-6-1.png and review-8-1/2/3.png; embedded in final Notebook |

The 180 s per-cell timeout leaves substantial startup/backend margin over the
measured few-second complete Notebook run; it is not used to skip computation.
No training or larger benchmark path is needed for this linear direct method.

Errors encountered and resolved, not omitted as successful first attempts:

- Before implementation, three selected parity tests failed because local
  modules did not yet exist. This established the red stage before extraction.
- First complete test run: 23 passed, 19 setup errors due to access denied at
  the machine's existing Temp/pytest-of-admin directory. No numerical tolerance
  changed. A private writable TEMP/TMP resolved it; the final 52 tests pass.
- The first Notebook launcher delegated to Anaconda's jupyter-nbconvert from
  PATH and could not find the custom kernel. Prefixing PATH with the shared
  environment's Scripts and using its prefix-local standard python3 kernel
  fixed the executable mismatch. The final command uses the shared nbconvert
  and the required shared Python, verified by environment evidence.
- A new truncated-ZIP prediction test initially exposed an uncaught BadZipFile
  and no report. The Judge now catches BadZipFile and EOFError alongside the
  other malformed-input cases; the test and full suite pass. It was not
  accommodated by dropping the malformed case.
- Successful nbconvert emits a Windows Proactor/ZeroMQ selector-thread fallback
  warning and an ipykernel local TCP transport advisory. They are archived
  verbatim in notebook.log; execution and numerical checks pass. No Notebook
  cell error is hidden or ignored. Library modules never call matplotlib.use.

### Judge adversarial and contract coverage

CLI tests run from a foreign working directory, so asset-relative resolution
is exercised. They cover absent file, missing key, wrong shape, wrong dtype,
complex dtype, NaN, Inf, random corrupt bytes, truncated ZIP, a valid poor
all-zero result and an actual passing reconstruction. Error reports have empty
metrics and exit 2; valid poor arrays have measured metrics and exit 1.
Tests also cover missing evaluator assets, nonconstant-reference requirement,
invalid threshold, wrong reference dtype/shape, actual-truth precedence,
baseline fallback, report directory creation and report-write failure/stderr.
UTF-8 reports contain finite JSON values, status/passed/metrics/thresholds/errors.

## 9. Import/call-role and documentation consistency review

Static import **and callable-symbol** inventories cover every source module,
generator, standalone Judge, test and all Notebook code cells. The full list
is in symbol-call-audit.json. Semantic review found:

- NumPy array construction, arithmetic, powers, logs, sums and broadcasting
  implement the local kernel and state updates; `np.gradient` is the explicitly
  selected generic finite-difference primitive. No ready-made inverse is called.
- NumPy random normal is used only for synthetic acquisition/probe/statistical
  tests. It neither supplies an answer to reconstruction nor enters evaluation.
- NumPy safe NPZ I/O serves declared input/output/reference/fixture roles.
  Runtime main/solver never loads hidden truth or references. Judge independently
  loads only submissions and evaluator assets; it does not import src.
- SciPy integrate.quad appears only in the independent physical test. It is
  a generic integrator, not a reconstruction shortcut. Matplotlib calls make
  figures only. Standard-library subprocess executes the local Judge and its
  contract tests, never downloaded code in the cleaned pipeline.
- There are no computational-imaging packages, vendored opaque framework,
  cross-task imports, dynamic eval/exec/compile, learned weights or oracle
  imports. The only extracted upstream numerical code is readable local
  arithmetic with preserved license and explicit state equations.
- README forward/inverse signs, derivative spacing, radiance versus radiance/mm,
  coordinates, no half-pixel offset, real float64 shapes, no alignment and
  both metrics were compared with source, Notebook, plans and tests. No
  objective/optimizer is falsely described for this direct method.
- Plan recurrence was checked at n=C-1 and n=2 against source indices j=n-1;
  B0 multiplies outer drive j+1, B1 inner drive j; endpoint copies and zero
  initialization agree. Interval signs and forward -2*pi*d are preserved.
- README has exactly the required six title/section headings, acquisition
  metadata, the complete output contract, a visible-key table and conceptual
  hints. No GitHub URLs, target parameters or hidden file paths enter public
  materials. References use DOI/arXiv links. Shape/dtype/key and literal scalar
  constants are checked by the metadata contract test.

These callable signatures were collected from the shipped code and reviewed
against `plan/design.md` (signature agreement does not replace the semantic
mathematical checks above):

| Callable | Actual signature |
|---|---|
| `src.physics_model.forward_transform` | `(emission, dr=0.025, hold_order=1)` |
| `src.physics_model.propagate` | `(drive, coefficients)` |
| `src.physics_model.state_coefficients` | `(cols, direction='inverse', hold_order=1)` |
| `src.physics_model.state_integrals` | `(n, a)` |
| `src.physics_model.validate_half_image` | `(image, dr, hold_order)` |
| `src.solvers.inverse_drive` | `(image, dr=0.025, hold_order=1)` |
| `src.solvers.inverse_transform` | `(projection, dr=0.025, hold_order=1)` |
| `src.solvers.reconstruct` | `(projection, dr=0.025, hold_order=1)` |
| `src.preprocessing.load_projection` | `(path: str | pathlib.Path) -> numpy.ndarray` |
| `src.main.main` | `()` |
| `src.main.run` | `(input_path='data/raw_data.npz', output_path='output/reconstruction.npz')` |
| `src.generate_data.generate` | `(destination, seed=1407)` |
| `src.generate_data.synthetic_fields` | `(seed=1407)` |
| `src.visualization.comparison_metrics` | `(estimate, reference)` |
| `src.visualization.plot_comparison` | `(projection, truth, upstream, fresh)` |
| `src.visualization.plot_metric_bars` | `(measurements)` |
| `src.visualization.plot_profiles` | `(truth, upstream, fresh, zero_order)` |

## 10. Fixture/reference consumers and final inventory

| Retained asset | Producer/provenance | Documented consumer |
|---|---|---|
| test/fixtures/upstream_stages.npz | original file, captured by private script; bound-data and signed-probe outputs | comparing_test/test_parity_physics.py for derivative, coefficients, both directions/orders and pipeline parity |
| notebooks/reference_outputs/upstream_reconstruction.npz | independent original bound-data run, first and zero orders | active review Notebook image/profile/metric comparisons and passing Judge contract-test input |
| notebooks/reference_outputs/cleaned_reconstruction.npz | fresh local reconstruct, source hashes and environment in adjacent measurements | active Notebook repeatability check only; never substituted for the fresh plotted result |
| notebooks/reference_outputs/cleaned_measurements.json | separate cleaned run with full precision metrics, parameters, data/reference/source hashes | Notebook provenance/runtime display and this ledger |
| evaluation/ground_truth.npz | analytic physical simulation emission | standalone Judge and hidden tests/Notebook; never a public calibration |
| evaluation/metrics.json | upstream baseline-derived boundaries, fixed before local code | standalone Judge and tests; machine-readable full precision |
| notebooks/abel_plasma_emission.ipynb | active local reconstruction, executed with nbconvert | primary self-contained maintainer review artifact, with outputs embedded |
| output/reconstruction.npz and output/judge_report.json | fresh CLI/Notebook output and independent Judge | benchmark output contract; generated files remain ignored by Git |

The stage fixture and Notebook references intentionally serve different
test/review consumers. No standalone NPY copies, screenshot files, PDFs,
downloads, original logs, capture scripts, full upstream tree or temporary
executed-notebook duplicate remains inside the task. The executed transfer
Notebook was promoted to the single primary Notebook; source/test bytecode
and pytest caches were removed. Both plans describe only shipped functions.
Task data have an explicit narrow root .gitignore exception so the public NPZ
is not accidentally omitted from a later commit; existing unrelated changes
were preserved. No commit or publish action was performed.

External archive index entries include source URLs/revision, ZIP, PDF, extracted
paper/documentation checkout, generation measurements, original capture,
dependency snapshots, final command logs, symbol audit, inspection figures,
release-verification JSON and authoring scripts. Environments and temporary
test directories are working support, not release assets. No task imports them.

## 11. Completion gate ledger

| Mandatory gate | Status | Evidence |
|---|---|---|
| Paper, exact upstream revision, hashes, equation/function/test mapping | PASSED | sections 2-5; archive index |
| Original and local method on exact bundled observations | PASSED | original/cleaned logs, exact loader fixture check |
| Independent upstream parity and physics checks | PASSED | 52-test suite; analytic diagnostics |
| Constants, units, precision, coordinates and conventions preserved | PASSED | metadata/units/Gaussian/probe tests; semantic review |
| Standard entry point and independent Judge pass fresh output | PASSED | cleaned-cli.log, judge.log, report |
| Judge poor/malformed/evaluator/report failure handling | PASSED | subprocess contract tests, including truncated ZIP |
| Public inputs suffice without hidden materials | PASSED | full model, sampling/noise metadata and output contract in README; runtime requires only observations |
| No truth, privileged observation, target calibration, hidden seed/object or GitHub URL leakage | PASSED | public audit; sole projection key |
| README metadata/output contract; no standalone metadata JSON | PASSED | metadata contract and inventory |
| Minimal shared dependencies; cleaned runtime without upstream library | PASSED | exact shared pins; abel unavailable; full pipeline executed |
| Explicit local mathematics and symbol/call-role checks | PASSED | sources, call inventory and section 9 semantic review |
| Unit, metric, Judge, parity, physics and pipeline tests | PASSED | 52 passed, exit 0 |
| Active Notebook reconstruction, fresh metrics/comparisons and nbconvert execution | PASSED | all 5 code cells, 4 inspected figures, no error outputs |
| Historical/upstream/fresh provenance, metric consistency and justified thresholds | PASSED | sections 4/6; independent known metric case |
| README and both plans match mathematics, signatures, defaults and flow | PASSED | section 9 and signature inventory |
| Final task inventory, cleanup, archive provenance and license | PASSED | section 10; archive-index.json; consolidated original MIT license text below |

No mandatory gate is failed or unverified. Untested extensions (GPU/other OS,
experimental plasma data, other Abel algorithms and legacy harness migration)
are explicitly outside this task's declared scope.


## Consolidated original license and attribution texts

Task-level license/notice files were consolidated here at user request. The complete original text is retained verbatim below, with its original-byte SHA256 for verification; archived originals are outside the task in revision_docs/task08. These terms continue to apply.

### Original LICENSE

SHA256 `fe613f7b9160b12707a1710458f3cd2df92563535baa0a9b1b0b839658d9c4fd`

```text
The MIT License (MIT)

Copyright (c) 2015 The PyAbel Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## Document and six-part notebook revision (2026-10-18)

At the user's explicit request, this revision consolidates task-level license/notice files into the full-text legal section in this validation record and removes the separate files. The terms, copyright attribution and data-source notices continue to apply. The full old documents, original notebook, license/notice bytes and all historical reference snapshots were archived before editing at `batch10/revision_docs/task08/abel_plasma_emission/original/`, outside this task. No skills were used. EHT original's complete approach/design and active comparison notebook were read as depth references; its science and data are not imported into this task.

Both plans now explain the concrete forward/inverse mathematics, dimensions, objective normalization/signs, parameter choices, actual numerical execution and scientific limitations. The design documents every shipped source/Judge callable's actual signature, shapes/dtypes/units, defaults, explicit versus assumed validation, errors, mutation/I/O behavior, module dependencies, distinct CLI/notebook flows and test mapping. No unimplemented optimizer, iterative update or convergence feature was invented for these direct methods.

The primary notebook now has exactly these six ordered H2 sections: `1. Problem`, `2. Data`, `3. Physics`, `4. Method`, `5. Results`, `6. Conclusion`. Data includes every visible array key/shape/dtype/unit, actual source and acquisition table followed by active loading. Physical acquisition, coordinate/unit/boundary conventions and mathematical reconstruction are explained separately. All earlier scientific plots, method groups, independent checks and metric tables are retained and freshly executed through local src functions. There is only one notebook inside the task.

Latest full suite: **52 passed in 2.55 s**, exit 0. Full active `python -m nbconvert --to notebook --execute notebooks/abel_plasma_emission.ipynb --output executed.ipynb --output-dir <private revision archive> --ExecutePreprocessor.timeout=180` returned exit 0, with 5 executed code cells and **4 image outputs**. Every image was extracted and visually inspected in this revision. Plot labels, comparisons and quantitative scales remain readable; no absent method/result was filled in. The 180 s timeout covers startup and plotting while the measured reconstruction remains short. Private `pytest.log`, `notebook.log`, `verification.json`, `executed.ipynb` and `figure*.png` record commands, timings, outputs and visual evidence; the executed notebook was promoted to the single primary notebook. Notebook warnings are the previously identified local Windows/Jupyter transport advisories; there are no cell errors.

Fresh standalone Judge report: `{"ncc": 0.9998266089834679, "nrmse": 0.004119801658595976}`, status `pass`. The fresh output and Judge report are separately archived as `fresh_reconstruction.npz` and `fresh_judge_report.json`; full new timing/diagnostic streams are in verification.json. Historical reference_outputs bytes were snapshotted and verified identical before/after execution, including any old runtime/source-hash measurement records. Those source hashes continue to describe their original measurements, not the current documentation revision. No observation, hidden reference, parity fixture or threshold was changed. Protected-asset before/after SHA256 maps are recorded in verification.json. Runtime algorithms and executable statements are unchanged.

License migration verification: LICENSE: `fe613f7b9160b12707a1710458f3cd2df92563535baa0a9b1b0b839658d9c4fd`; complete original bytes verified within the consolidated section. Historical mentions of an accompanying LICENSE inside the preserved original notice are quoted original text; they now resolve to the consolidated license immediately preceding it, rather than a standalone file.

### Current revised file hashes

| File | SHA256 |
|---|---|
| `README.md` | `29f4774bc4ed056f92d71171ee55e15ed44605f0980036e0fbbc6a1cd39fb897` |
| `plan/approach.md` | `20728586b47496c3df870e0fa231cab60cd5529403ee395fa8ecfb6269a40170` |
| `plan/design.md` | `21aa5ca2e510580b089581bbfa85b3fb328eb0a55f500bac40bd38a4900376ac` |
| `notebooks/abel_plasma_emission.ipynb` | `1658f8b86c9b39b767a535338f8efeb7c74fb6c4f074c551adb84a34dbfa9fe3` |
| `src/__init__.py` | `53bb6824a3382e51e104f9635c607358f559d2577cd5f77e59135c76c0c37540` |
| `src/generate_data.py` | `a856a3c3ecef9023a206a479de0dc1ac00859a149150b4ccf30568af30052481` |
| `src/main.py` | `9f27b7e1598f1ecb152a4a383d747c76f906a15a5209e10d862b3e901a25a68c` |
| `src/physics_model.py` | `59d320ad0f19466d4dd64ed0e6d12a500aa06f32035b45d026f0d219c2a8208f` |
| `src/preprocessing.py` | `bdf9daa347cfeb91c467cb7672644e75f3615e31d47806ce8985171fd4b6eea7` |
| `src/solvers.py` | `a964e8b158d265c76c3ecea6b7e595fab5c7dcc08c77d83aa77178abeda7efc8` |
| `src/visualization.py` | `b2348ca27a0e8186fdbb0e902713a4c23290f0f651be9fa726b9ad54c2ceaeef` |

Only two src module docstrings changed: physics_model.py and solvers.py now point their original copyright/MIT attribution to test/validation.md instead of the deleted ../LICENSE. No executable AST statement changed; current hashes above identify the documentation update. The retained Gaussian diagnostics, first/zero-order original/fresh four-way table, six-image panel, radial profiles and paired metric bars were all inspected.


### Final Markdown-only editorial polish

The final notebook wording merges repeated Data introductions into one source/selection paragraph followed by the complete tables. Problem, Physics, Method, Results and Conclusion were reviewed together to remove repeated provenance, acquisition or no-iteration statements while retaining task-specific mathematical definitions, processing caveats, comparisons and limitations. Only Markdown cells changed: every code cell, execution count and embedded result/output is byte-equivalent under deterministic JSON serialization to the already executed notebook (SHA256 `29d9e33d577759154ff063130b05566cbf56e259ebf5080059ca3200f5b6d6c3`). The successful numerical execution and 4 inspected image outputs therefore remain applicable; no identical numerical rerun was needed. `revision_docs/task08/abel_plasma_emission/markdown_final.ipynb` archives this final wording, and the notebook SHA256 in the current revised file table above is updated.
