# Computational Imaging Task Cleaning Specification 

This file provides guidance to Agent when working with code in this repository.

## Project Overview

**imaging-101** is a benchmark suite for evaluating coding agents on computational imaging tasks. It covers 8 domains (mathematics, physics, chemistry & material science, astronomy, earth science, mechanics, biology, medicine) with standardized task structure, reference implementations, and automated evaluation.

## Repository Structure

```
imaging-101/
├── skills/                    # Shared skill library (available to agents)
├── evaluation_harness/        # Benchmark evaluation framework
├── docs/                      # Documents
└── tasks/<task_name>/         # Individual imaging tasks

```

### Task Template

Every task follows this standardized layout:

```
tasks/<task_name>/
├── README.md                  # Problem definition: physics, data description, method hints
├── requirements.txt           # Sandbox environment dependencies
├── data/
│   └── raw_data.npz           # Visible observations and calibration arrays only
├── output/
│   ├── judge_report.json      # Metrics, passed or not.
│   └── reconstruction.npz     # Solver submission; contract documented in README
├── plan/
│   ├── approach.md            # Solution methodology and algorithm description
│   └── design.md              # Code architecture with function signatures
├── src/
│   ├── preprocessing.py       # Raw data → processed observations (y)
│   ├── physics_model.py       # Forward model: x → y
│   ├── solvers.py             # Inverse solvers: y → x_hat
│   ├── visualization.py       # Plotting utilities and metrics
│   ├── generate_data.py       # Optional: synthetic data generation
|   └── main.py                # standard implement code pipeline entry point
├── notebooks/
|	├──reference_outputs/      # Hidden standard code running results.
│   └── <task_name>.ipynb      # **Required** end-to-end tutorial notebook (user review interface)
└── evaluation/
	├── metrics.json		   # Hidden, the threshold for passing the task.
    ├── judge.py               # Hidden, standalone submission evaluator, reading metrics.json.
    ├── ground_truth.npz       # Hidden measured/simulated truth, when available
    ├── baseline_reference.npz # Hidden standard-method result if truth is unavailable
└── test/
│   ├── validation.md          # checklist and validation reports.
│   ├── unit_test/             # checklist and validation reports.
│   ├── comparing_test/        # checklist and validation reports.

```

## Evaluation Modes

The released solver input is an **allowlist**: `README.md`, `data/raw_data.npz`,
and environment information (the shared environment specification and task
dependency requirements). Metadata is included directly in the README; do not
create other files such as `meta_data.json`.

| Mode | Solver sees | Solver produces | Evaluator uses |
|------|-------------|-----------------|----------------|
| End-to-end | README + raw observations + environment information | `output/reconstruction.npz` | Hidden `evaluation/judge.py` and reference arrays |

standard implement code `src/`, `plan/`, `/notebooks`, and `evaluation/` directory are maintainer-only assets for users to learn, which is invisible for coding agents when solving problems end-to-end. 

## Commands

```bash
cd tasks/<task_name>
conda activate imaging-101               # Shared environment, prepared by maintainers
# requirements.txt documents the task subset; do not upgrade packages per task
python main.py                           # Maintainer reference run; writes output/reconstruction.npz
python evaluation/Judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest evaluation/tests/ -v    # Run all tests
```

## Task Cleaning Principles

Tasks are created by **cleaning** existing research code from published papers and their reference implementations. The goal is to restructure real computational imaging code into the standardized benchmark format.

### Code cleaning workflow

Cleaning follows a **test-driven** process. Never rewrite code first and hope it works — define success criteria from the original code, then clean.

1. **Run the original code** — execute the reference implementation on the task's dataset end-to-end. Save all intermediate and final outputs (preprocessed data, model checkpoints, loss history, reconstructed images/samples) privately. This run is mandatory on the exact bundled observations, not just a different upstream demo. Record the upstream URL and commit, data provenance/hash, seed, environment, command, runtime, exit status and log paths in `test/validation.md`. A paper result, accessible repository or successful import is not evidence of a working task.
2. **Define parity tests** — write test code in `test/` that compares original vs cleaned code at every pipeline stage: preprocessing, forward model, loss functions, solver/architecture, and full-pipeline outputs. These tests guarantee the cleaned code is functionally identical to the original.
3. **Clean the code** — restructure into our `src/` layout. Run parity tests after each change; fix any discrepancies immediately. Do not proceed to the next stage until all parity tests pass.
4. **Generate reference outputs** — from the validated original method, produce reference reconstructions and, when applicable, checkpoints (model weights, loss history) and posterior samples/statistics in `notebooks/reference_outputs/`. Store actual truth in `evaluation/ground_truth.npz`, or the standard-method reconstruction in `evaluation/baseline_reference.npz` when truth is unavailable. Never call a standard-method reconstruction ground truth. Save measured metrics and justified thresholds to `evaluation/metrics.json`; run the cleaned standard method through `evaluation/judge.py` and require it to pass.
5. **Write unit tests and fixtures** — per-function tests under  `test/unit_test/`.
6. **Write the notebook** (`notebooks/<task_name>.ipynb`) — this is the **primary maintainer review deliverable (hidden from solvers)** and must always be generated as the final step of cleaning. The notebook serves as the interface for users to review, validate, and critique the implementation. Requirements:
   - Loads precomputed results from `notebooks/reference_outputs/` so it **runs in seconds** without heavy computation.
   - Contains all key visualizations: ground truth, reconstructions, comparison figures, metrics tables/charts.
   - Includes **commented-out code** for running the full pipeline from scratch (so users can reproduce).
   - Uses the cleaned `src/` modules (not the original library) to demonstrate the implementation works.
   - Should be self-contained and tell a clear story: problem → data → method → results → conclusion. Contains necessary fomulas, implement details. 
   - **No opaque paper references** — never write "as in Eq. N" or cite a paper figure/section as a substitute for explanation. Every formula must appear explicitly with variables defined; every algorithm must be described in enough detail that the notebook is fully self-contained without the original paper.
   - **Verify the notebook executes without errors** by running `jupyter nbconvert --execute` before finishing.
   - **Never call `matplotlib.use()` in `src/` modules** — backend selection is `main.py`'s responsibility. Library code must stay backend-agnostic so notebooks can use `%matplotlib inline` without conflict.

### README requirements

Every `README.md` must follow this structure and these rules:

**Required sections and header format:**

```
# <Task Name>

> One-sentence summary: what is reconstructed, from what data, and what makes it hard.

> Domain: <domain> | Keywords: <kw1>, <kw2>, ... | Difficulty: Easy / Medium / Hard

## Background
## Problem Description
## Data Description
## Method Hints
## References
```

The summary line (first `>`) must be self-contained in one sentence. The domain/keywords line (second `>`) follows immediately after.

**Data tables** — every key in the visible `data/raw_data.npz` must appear in a table with exactly these columns:

| Key | Shape | Dtype | Description |
Shape must use batch-first convention `(1, ...)`; Description must include units.

**Ground truth handling** — README data tables must not describe hidden truth,
reference reconstructions, their keys, paths or sample values. Document these in
`test/validation.md` instead. Actual truth may come from simulation or an
independent validated reference measurement. Instrument calibration data (PSFs,
system matrices, sensitivity maps) belongs in `raw_data.npz` if legitimately
available to the solver at acquisition time. A calibration estimated using the
hidden target must not silently be treated as available input.

**Output contract** — in `Problem Description`, require the solver to save
`output/reconstruction.npz` (relative to its task workspace). Default key:
`reconstruction`. List every required output key, batch-first shape, numeric
dtype, units, axis order, physical scale and any real/complex convention. A
multi-quantity inverse problem may require multiple named arrays in this one
file; the Judge must evaluate all required quantities. Explain evaluation metrics
and any scientifically justified alignment convention without describing hidden
reference files. The output must not depend on a particular solver code layout.

**Keywords** — domain-method level only (e.g., `wave imaging`, `self-supervised learning`), not implementation level (e.g., `f-k migration`, `U-Net`, `recorruption`). Aim for 2–4 keywords.

**Background scope** — explain the physics motivation and real-world applications of the imaging modality (2–4 paragraphs). Do not include: paper citations or author names (those belong in References), solver or algorithm details (those belong in Method Hints), or sample-specific information (that belongs in Problem Description). A good Background answers: what physical effect is exploited, what makes the modality useful, and where it is applied in practice.

**Problem Description scope** — describe the physical forward model and the inverse problem only: the measurement equation with all variables defined, what makes the problem ill-posed or computationally hard, sample/acquisition context, and the input/output specification. Do not include reconstruction pipelines, training objectives, or algorithm-specific steps — those belong in Method Hints or `plan/approach.md`. Do not duplicate the acquisition parameter table here — those values are documented in Data Description.

**Data Description scope** — document visible observation/calibration array keys,
shapes, dtypes and units, followed by an **Acquisition metadata** table with columns
`Parameter | Value | Unit | Meaning`. Include all acquisition geometry, sensor
parameters, coordinate conventions, known noise parameters and boundary
conditions needed to formulate the forward model. Arrays stay in `raw_data.npz`;
small scalar/string parameters are written directly in this table. No standalone
metadata JSON is created or required. Reference code uses explicit matching
constants; it need not parse Markdown. Tests check that these constants agree
with the README. Solver hyperparameters belong in maintainer `plan/approach.md`
and reference implementation constants, not the acquisition table. Synthetic
generation details belong in the hidden `src/generate_data.py` docstring and
`evaluation/validation.md`; the solver README remains self-contained and must
not rely on access to these files.

**No public GitHub URLs** — solver-visible materials must not expose GitHub URLs,
including README references, data descriptions, notebook-independent public
documentation, task requirements and shared environment specifications. This
also covers GitHub raw-content, API, download and shortened-link domains
(`github.com`, `raw.githubusercontent.com`, `api.github.com`, `github.io`,
`githubusercontent.com`, `codeload.github.com`, `git.io`) and URLs hidden in
Markdown link targets, HTML, badges or comments. 

**Method Hints depth** — conceptual level only: name the algorithm family, the key decomposition or regularization, and why it suits the problem. Key iterative update formulas are appropriate here and help agents understand the algorithm structure. Do not give full hyperparameter tables or step-by-step pseudocode — that belongs in `plan/approach.md`. The hint should guide *what* to implement, not *how* to implement it.

**Formula self-containment** — all mathematical notation must be defined in the README itself. Never write "as in Eq. N of [Paper]" or reference a figure from the original paper as a substitute for explanation.

**GitHub math rendering pitfalls** — GitHub renders math via KaTeX but applies Markdown preprocessing first, which can silently break formulas:

- **Display formulas**: prefer ` ```math ` fenced blocks over `$$...$$` for complex expressions — they bypass Markdown preprocessing entirely and render more reliably.
- **Underscore pairing in bullet lines**: GitHub pairs `_` as italic markers across an *entire* bullet line before parsing `$...$` math spans. Count all `_` characters in a bullet line (including those inside separate `$...$` blocks); if there are ≥ 2, they will be paired as italic and break all math in that line. Fix by simplifying subscript notation or moving the formula to a display block.
- **Curly braces in `\left`/`\right`**: write `\left\lbrace`/`\right\rbrace` rather than `\left\{`/`\right\}` — GitHub preprocesses `\{` as an escaped literal brace, leaving KaTeX to see the invalid sequence `\left{`.
- **Roman text in inline math**: use `\mathrm{}` rather than `\text{}` for upright text inside `$...$` spans; `\text{}` can fail in some inline contexts.

**Data format** — all observation and ground-truth arrays must be stored as `.npz` (not `.mat`, `.npy`, or other formats). See the Data section below for the full interface specification.

### Evaluation metrics format

`evaluation/metrics.json` is the boundary file consumed by the evaluation harness.
It always follows this schema:

```json
{
  "baseline": [
    {"method": "<description>", "ncc_vs_ref": <value>, "nrmse_vs_ref": <value>}
  ],
  "ncc_boundary": <value>,
  "nrmse_boundary": <value>
}
```

The task Judge evaluates the exact declared output arrays against
`evaluation/ground_truth.npz`, or against `evaluation/baseline_reference.npz` if
truth is unavailable. If both exist, actual truth takes precedence. A run passes
only if every required metric meets its declared boundary. No implicit center
crop, rescaling, magnitude conversion or output/reference filename guessing is
allowed. Task-specific masks or phase/translation alignment must be scientifically
justified, fixed before evaluation and documented in the output contract.

For suitable non-degenerate true-reference tasks, starting thresholds remain
`ncc_boundary = 0.9 * baseline_ncc` and
`nrmse_boundary = 1.1 * baseline_nrmse`, subject to actual validation. Never use
these mechanically when the baseline error is zero or a metric is undefined.
When the standard method itself supplies the hidden reference, self-comparison
produces NCC=1 and NRMSE=0; determine a nonzero numerical tolerance from independent
reruns and, where meaningful, add a forward observation-consistency criterion.
Record the rationale and measurements in `evaluation/validation.md`. Reference
agreement is not proof of physical accuracy. Poor baselines must be improved or
the task rejected, not accommodated by vacuous thresholds.

For tasks where NCC/NRMSE are not meaningful (e.g. phase, localization or spectral
parameter inference), define explicit task-specific metrics, comparison directions
and `*_boundary` entries. Reject undefined metric configurations during task
validation; specify a meaningful alternative for zero norm or constant references.
Do not emit NaN/Infinity as JSON numbers.

**Standard NRMSE definition** — unless a task explicitly documents otherwise, NRMSE is
computed as RMS error normalised by the dynamic range of the reference:

$$\text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum_i (\hat{x}_i - x_{\text{ref},i})^2}}{\max(x_\text{ref}) - \min(x_\text{ref})}$$

where $\hat{x}$ is the reconstruction and $x_\text{ref}$ is the reference (ground truth or
reference output).  Tasks that deviate from this definition (e.g. phase tasks using
std-normalisation, flux tasks using L2-ratio) must document their formula explicitly in
`evaluation/metrics.json` under an `"nrmse_definition"` key.

**Standard NCC definition** — NCC is computed as cosine similarity between flattened
estimate and reference vectors (no mean subtraction):

$$\text{NCC} = \frac{\hat{x} \cdot x_\text{ref}}{\|\hat{x}\|\,\|x_\text{ref}\|}$$

where $\hat{x}$ and $x_\text{ref}$ are flattened to 1-D vectors.  This is equivalent to
the cosine of the angle between the two vectors.  Tasks must not add a per-task
`"ncc_definition"` field unless they deviate from this convention.

### Standalone Judge interface

Every new task must include `evaluation/Judge.py` with this command interface:

```bash
python evaluation/Judge.py --prediction output/reconstruction.npz --report output/judge_report.json
```

- Resolve hidden reference paths relative to `Path(__file__).resolve().parent`,
  not the caller's current directory. CLI prediction/report paths are relative to
  the caller's working directory unless absolute.
- Load NPZ with `allow_pickle=False`. Validate file existence, required keys,
  declared shapes/dtypes, and finite values before scoring. Do not silently crop,
  reshape, coerce complex results to real or substitute a reference for a missing
  submission. Never run or import submitted solver code.
- Write UTF-8 JSON with `status` (`pass`, `fail`, `error`), `passed` (boolean),
  `metrics` (measured numeric values), `thresholds` (each metric's comparison and
  boundary), and `errors` (list of messages). Invalid inputs return empty metrics
  and a descriptive error; unavailable hidden assets are evaluator errors.
- Return exit code `0` for pass, `1` for a valid prediction below thresholds, and
  `2` for malformed input or evaluator errors. Create the report parent directory
  when needed. Report-write failure also returns `2` with a stderr diagnostic.
- Keep the Judge independent of the reference solver and its checkpoints. Use
  deterministic evaluation routines and document any public calibration inputs
  required for observation-consistency metrics.

This NPZ/Judge contract is for subsequent harness integration. The existing
`evaluation_harness` includes legacy `.npy` scoring assumptions and is **not**
made compatible merely by following this specification.

### Shared environment

Prefer one project-level `environment.yml` for Linux/WSL2
and one NVIDIA GPU, including CPU execution paths where available. Start dependency
resolution with Python 3.11 and the common NumPy/SciPy, plotting, NPZ/HDF5, testing
and PyTorch stack; freeze exact mutually compatible versions only after validation.
Use one CUDA runtime family and matching GPU packages. Select the PyTorch backend
for multi-backend packages when available. Record OS, driver, GPU memory and
external compiler/runtime requirements, which conda alone cannot supply.

Each task's `requirements.txt` documents its subset of the shared pinned environment.
Do not perform per-task upgrades that break other tasks. Do not add domain libraries
to bypass the explicit-mathematics rules below. Prefer available weights
or small CPU baselines over training large models. If a genuine version/runtime
conflict remains, document the isolated exception and the reason; never claim all
tasks run in one environment without actually checking them. Environment information
is public but contains no private filesystem paths or credentials.

### Completion checklist

Before considering a task "done", verify every item:

- [ ] Original and cleaned standard methods run on the bundled data; `evaluation/validation.md` records reproducible evidence
- [ ] `python main.py` writes the declared `output/reconstruction.npz` and the Judge passes it
- [ ] Judge rejects poor output and reports missing file/key, shape/dtype errors and NaN/Inf correctly
- [ ] README + raw observations + environment information suffice to solve the task
- [ ] No truth, reference output, hidden generation seed/object or privileged extra observation leaks into visible inputs
- [ ] No GitHub URLs occur in any solver-visible material; source provenance is maintainer-only
- [ ] Acquisition metadata appears in README, with no standalone metadata JSON
- [ ] Shared environment or a documented exception has been validated
- [ ] `python -m pytest evaluation/tests/ -v` — all tests pass
- [ ] `notebooks/<task_name>.ipynb` exists and executes without errors
- [ ] Notebook actively reconstructs from raw data and evaluates the fresh result; commented solve commands or cached-only plots are insufficient
- [ ] `evaluation/metrics.json` exists with expected metrics
- [ ] `README.md`, `plan/approach.md`, `plan/design.md` exist
- [ ] `requirements.txt` lists all dependencies
- [ ] Core forward and inverse mathematics is executable local code, not a library solver call
- [ ] Runtime, generator, Judge, tests and notebooks pass dependency and forbidden-solver checks
- [ ] The cleaned task runs without the upstream domain packages installed
- [ ] Delete all intermediate products not in `Task Template`.

### Code extraction
- **Extract explicit mathematics** — copy and adapt the relevant published/reference algorithm into local, readable functions, preserving its equations, discretization and numerical conventions. Preserve license notices and record the source version and adaptations in private provenance. Do not substitute a different physical model or easier inverse problem merely to remove a dependency.
- **No domain-library shortcut** — even a package used by the original 59 tasks does not authorize hiding the task's core mathematics. Newly cleaned task code must explicitly implement acquisition operators, boundary/unit conventions, objective and regularizer, gradients/adjoints or Jacobians where required, and the inverse updates, initialization and stopping rules. Closed-form inverses should explicitly evaluate their formula; artificial iteration or padding code is not required. Equations in comments or a README do not substitute for executable mathematical steps.
- **Dependency ceiling and default allowlist** — no new scientific dependency outside those already used by the original 59 tasks may be introduced. The default implementation subset is Python standard library, NumPy, SciPy, PyTorch, scikit-image, Pillow, Matplotlib, imageio and tqdm; existing data-I/O tools such as h5py/tifffile and pytest/Jupyter notebook tooling may be used for their infrastructure roles. Other legacy packages require an explicit justification and must not implement the core task in a black box. The current 16-task rewrite uses NumPy/SciPy/Matplotlib plus testing/notebook tools.
- **Numerical primitives, not complete solvers** — FFT, convolution, interpolation, sparse matrix construction, direct linear-system solves, eigendecompositions and special functions are permitted building blocks. Core optimization loops, proximal/reweighting updates, line searches and stopping decisions must be local. Calls such as `skimage.restoration.richardson_lucy`, `scipy.optimize.minimize`/`least_squares`, `torch.optim`, PyLops FISTA, SimPEG inversion, MNE inverse operators, DIPY reconstruction models and SCICO ADMM are not permitted substitutes, even if their package is otherwise allowed. Use descriptive intermediate variables and mathematical docstrings, not one-line wrappers or opaque vendored frameworks.
- **Isolated upstream verification exception** — the unmodified upstream implementation may run in a separate maintainer-only oracle environment to produce independent parity fixtures. Private capture scripts and archived upstream source are not part of the cleaned task runtime and must never be imported by its solver, generator, Judge, tests or notebook. They may retain upstream dependencies solely for that verification role. Normal tests consume frozen numeric fixtures with recorded provenance, not upstream imports. Public requirements/environment files contain only cleaned-runtime dependencies; the original verification environment is documented privately.
- **No cross-task imports** — each task must run independently.
- **Version provenance** — preserve the published algorithm and record the upstream commit and cleaned dependencies. Maintained library versions may be used for the shared environment after parity validation; distinguish original-paper reproduction from a modern library port. Package release date alone does not define answer leakage. Never bundle task-specific trained weights or target-derived information as generic environment assets.

### Data
- If real observation data is unavailable, **generate synthetic data** via `src/generate_data.py`. Results should be qualitatively similar to the paper figures, not necessarily identical.
- **Coordinate convention consistency** — `generate_data.py` must use the exact same pixel coordinate convention (e.g., half-pixel offset, axis direction) as `physics_model.py`. A mismatch causes correct amplitudes but wrong phases, yielding χ²/DOF orders of magnitude too high. Always verify by round-tripping: forward-model the ground truth with `physics_model.py` and compare to generated observations. Use a justified noise-weighted residual (χ²/DOF near 1 when its assumptions apply); use numerical tolerance for noiseless data. Where feasible also compare an independent discretization or upstream forward model, since sharing the same erroneous generator and solver can conceal a model error.
- **Parameter unit convention consistency** — when extracting code from a reference library, verify that every parameter passed to internal formulas uses the expected unit convention (dimensionless vs physical). For example, a propagation model may expect a normalised pixel spacing `res = dx·n0/λ` (dimensionless); passing `dx` in μm instead causes a systematic scale error in all frequency-domain quantities (kz, phase advance, scattering strength) without any runtime error. Always check the reference library's convention explicitly before porting.
- **Standard data interface** — all tasks must follow this file layout:
  - `data/raw_data.npz`: all observation arrays **and instrument calibration parameters** (e.g., PSF kernels, system matrices) bundled by descriptive key names (e.g., `measurements`, `vis_cal`, `uv_coords`). Shape always includes a batch dimension as the first axis: `(1, ...)`.
  - `evaluation/ground_truth.npz`: true physical ground truth from simulation or independent validated reference measurements — e.g., ground-truth refractive index, true source image, simulated scatterer positions. Same batch-first convention. **Do not put instrument parameters (PSF, calibration data) here** — those belong in `raw_data.npz` because they are part of the measurement process, not the unknown to be recovered.
  - `evaluation/baseline_reference.npz` (optional): when no true ground truth exists (real observational data), store the reference algorithm's output here for NCC/NRMSE evaluation. Shape follows batch-first convention `(1, ...)`. The `ncc_vs_ref` / `nrmse_vs_ref` in `evaluation/metrics.json` are computed against this file. If both `ground_truth.npz` and `baseline_reference.npz` exist, `ground_truth.npz` takes precedence. Document hidden reference keys and provenance only in `evaluation/validation.md`.
  - README `Acquisition metadata`: imaging parameters only (sensor specs, physical quantities, acquisition geometry); arrays go in visible `raw_data.npz`. Do not create `meta_data.json`. Solver parameters are maintained in `plan/approach.md` and named reference-code constants, not in the acquisition table.
  - Exclude any hidden answer or privileged extra measurement from `raw_data.npz`, including clean fully sampled data for a sparse/noisy inverse problem. Simulator seeds and object definitions that reproduce the answer are maintainer-only. Calibration must be realistically available.
  - Use float32 for large volumes (>50 MB at float64) to stay within the GitHub 100 MB
    file size limit; load with `.astype(np.float64)` before computation.
  - Do not keep redundant `.npy` copies once npz files are in place.
- Keep all existing reference materials (PDFs, `reference_website_github.md`) in each task folder.

### Testing
- Every function in `src/` must have a corresponding test fixture (`evaluation/fixtures/`) and unit test (`evaluation/tests/`). Except for the visualization, we don't need to test the visualization.
- Fixture naming: `param_*` (constructor args), `input_*` (function inputs), `config_*` (settings), `output_*` (expected outputs).
- **Deterministic outputs** (e.g., forward model, gradient computation): use explicit numerical tolerances appropriate to dtype/backend (`rtol=1e-10` only where float64 conditioning permits it). Document justified GPU/float32 tolerances; do not describe a tolerance comparison as exact equality.
- **Stochastic outputs** (e.g., noise injection, MCMC samples): test statistical properties only (mean, variance, shape, dtype). Never rely on random seed matching for exact value comparison.
- **SSIM is unreliable for heavily underdetermined problems** — for limited-angle or sparse-measurement tasks (e.g., ODT, interferometry), SSIM can decrease even as NRMSE and NCC improve, because sparsity-promoting regularization or positivity constraints change local structural patterns independently of reconstruction accuracy. Use NRMSE and NCC as primary quality indicators; treat SSIM as secondary for such tasks.
