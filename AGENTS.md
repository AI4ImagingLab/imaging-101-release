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

Every newly cleaned task follows this layout. Paths and filename case are
authoritative throughout this specification; do not add root-level entry-point
wrappers or duplicate test/report directories.

```text
tasks/<task_name>/
|-- README.md                  # Public problem, data, metadata, and output contract
|-- requirements.txt           # Cleaned runtime subset of the shared environment
|-- data/
|   `-- raw_data.npz           # Public observations and legitimate calibration only
|-- output/
|   |-- judge_report.json      # Generated evaluation report
|   `-- reconstruction.npz     # Generated solver submission
|-- plan/
|   |-- approach.md            # Implemented mathematics and algorithm
|   `-- design.md              # Shipped interfaces and actual execution flow
|-- src/
|   |-- preprocessing.py
|   |-- physics_model.py
|   |-- solvers.py
|   |-- visualization.py
|   |-- generate_data.py       # Required when generating new synthetic observations
|   `-- main.py                # Standard entry point: python -m src.main
|-- notebooks/
|   |-- reference_outputs/     # Hidden reference arrays and identified measurements
|   `-- <task_name>.ipynb      # Required active reconstruction/review notebook
|-- evaluation/
|   |-- metrics.json           # Hidden metric definitions and pass thresholds
|   |-- judge.py               # Hidden standalone evaluator; lowercase filename
|   |-- ground_truth.npz       # Hidden actual truth, when available
|   `-- baseline_reference.npz # Hidden method result when actual truth is unavailable
`-- test/
    |-- validation.md          # Authoritative provenance and validation evidence
    |-- fixtures/              # Frozen upstream numerical cases and test inputs
    |-- unit_test/             # Function, metric, and Judge contract tests
    `-- comparing_test/        # Upstream parity, physics, and pipeline tests
```

`src/__init__.py` is permitted for package imports. Omit `src/generate_data.py`
when existing observations are reused and no new synthetic observations are
generated. A task-level `LICENSE` or `NOTICE` is permitted when redistribution
terms require it; preserve source copyright and license notices. Files inside
fixture and reference-output directories must have a documented consumer and
provenance. Both hidden reference filenames are reserved, but actual truth is
required only when available; do not invent truth to fill the template.

Use `test/` for all tests, fixtures, and validation evidence; do not create
parallel test/fixture directories or a second validation report under evaluation.
Keep supplied paper PDFs, complete upstream checkouts, download archives,
capture scripts, and original logs in a maintainer-only archive outside the final
task directory. Record their source, revision, hashes, license, and archive index
in `test/validation.md`; do not retain standalone PDFs or
`reference_website_github.md` in the task package. Remove temporary executed
notebooks, screenshots, unused scripts, and redundant array copies before handoff.

## Evaluation Modes

The released solver input is an **allowlist**: `README.md`, `data/raw_data.npz`,
and environment information (the shared environment specification and task
dependency requirements). Metadata is included directly in the README; do not
create other files such as `meta_data.json`.

| Mode | Solver sees | Solver produces | Evaluator uses |
|------|-------------|-----------------|----------------|
| End-to-end | README + raw observations + environment information | `output/reconstruction.npz` | Hidden `evaluation/judge.py` and reference arrays |

The reference implementation in `src/`, plans, notebooks, evaluation assets, and
`test/` are maintainer-only materials, hidden from agents solving end-to-end tasks.

## Commands

```bash
cd tasks/<task_name>
conda activate imaging-101               # Shared environment, prepared by maintainers
# requirements.txt documents the task subset; do not upgrade packages per task
python -m src.main                       # Standard run; writes output/reconstruction.npz
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
python -m pytest test/ -v                 # Unit, parity, physics, and pipeline tests
jupyter nbconvert --to notebook --execute notebooks/<task_name>.ipynb --output <task_name>_executed.ipynb --output-dir output
```

## Task Cleaning Principles

New tasks are cleaned **from the supplied GitHub repository, paper, and data**.
The deliverable is a validated local standard implementation of the selected
published method, with explicit mathematical and physical processes, packaged as
a benchmark task. Copying files, replacing API wrappers, or changing directory
format alone does not complete a new task. Format-only maintenance is a separate
scope and applies only when the user explicitly requests it; a legacy exception
does not authorize shortcuts in newly cleaned tasks.

Before implementation, fix the task's physical unknowns, acquisition model,
selected method, dataset, and output contract. Record the repository URL and exact
commit, paper version, and data hashes in `test/validation.md`. Distinguish an
actually verified remote revision from an available local tracking revision.
Do not claim remote freshness when network verification was unavailable.

Maintain a traceability table in `test/validation.md` mapping each required paper
equation or algorithm stage to its upstream file/function, local function, and
validation case. Record ambiguities and discrepancies between paper, code, and
data. Investigate suspected upstream errors with independent mathematical or
physical checks; do not silently copy a known error, change the method, or claim
unchanged reproduction after a correction.

### Code cleaning workflow

Cleaning follows a **test-driven** process. Establish success criteria from an
independent upstream run before implementing the cleaned method.

1. **Run the original code** -- execute the reference implementation on the exact observations to be bundled, using an isolated upstream environment when necessary. Save intermediate and final numeric outputs privately. Record source revision, data hashes, seed, environment including numerical backend, command, runtime, exit status, warnings, and log/archive locations in `test/validation.md`. A paper result, cached upstream notebook, successful import, or run on a different demo does not establish a working task. If this run cannot be completed, record the unresolved gate and do not declare cleaning complete.
2. **Define independent parity and physics tests** -- freeze numeric fixtures captured from the upstream run, then compare preprocessing, forward model, losses, gradients/adjoints, solver updates, and full-pipeline outputs under `test/comparing_test/`. Select tolerances and acceptance criteria before judging cleaned results. Add independent physical or mathematical checks where applicable; agreement between two copies of an erroneous formula is insufficient. Fixtures generated by the cleaned function itself are not upstream parity evidence.
3. **Extract and implement the method locally** -- write the required operators and inverse mathematics under `src/` using permitted numerical primitives. Preserve the selected method, discretization, constants, and numerical conventions. Run the relevant stage tests after each extraction; resolve discrepancies before proceeding. Draft `plan/approach.md` and `plan/design.md` from the actual method and update them as implementation decisions are validated.
4. **Generate and identify reference outputs** -- keep upstream measurements and fresh cleaned-method measurements separately identified in `notebooks/reference_outputs/`, with their data, method, parameter, environment, and metric provenance in `test/validation.md`. Store actual truth in `evaluation/ground_truth.npz`, or a validated standard-method result in `evaluation/baseline_reference.npz` when truth is unavailable. Never relabel reconstruction as truth. Save justified thresholds in `evaluation/metrics.json`; require the fresh cleaned output to pass `evaluation/judge.py`. Judge success does not replace parity or physical validation.
5. **Complete unit and contract tests** -- place per-function tests and metric/Judge checks in `test/unit_test/`, numeric fixtures in `test/fixtures/`, and full-pipeline tests in `test/comparing_test/`. Record passed, failed, and unverified checks without conflating them.
6. **Write and execute the notebook** (`notebooks/<task_name>.ipynb`) -- this is the primary maintainer review deliverable, hidden from solvers, and must be finalized after the implementation, references, tests, and plans agree:
   - By default, actively reconstruct the bound benchmark observations using cleaned `src/` modules and evaluate the fresh results. Load precomputed results from `notebooks/reference_outputs/` for upstream comparisons and an optional explicitly labeled cache-preview mode. That preview does not satisfy reconstruction validation. Record actual runtime and choose a justified execution timeout rather than requiring every task to run in seconds.
   - Execute the complete selected benchmark reconstruction path. Optional commented commands may document a larger upstream experiment or training run, but cannot replace active benchmark reconstruction.
   - Include key ground-truth/reference, reconstruction, comparison, and metric visualizations. Preserve the scientific comparisons and explanatory logic of any supplied upstream notebook; for a new notebook, cover the selected paper's task-relevant comparisons. Do not silently drop methods, comparison groups, metric tables, paired bars, or scientifically relevant annotations. Explain intentional changes.
   - A plotted fresh result must have been computed. Never fill a missing method with zero or substitute an archived reference as a fresh reconstruction. Identify historical reports, results of this task's upstream run, and fresh cleaned-code measurements separately, including method and acquisition condition.
   - Use the cleaned modules, not the upstream library. Tell a self-contained story: problem, data, physics, method, results, and conclusion. State formulas with variables defined and explain algorithms; paper equation/figure references cannot substitute for explanation.
   - Execute the notebook using `jupyter nbconvert --execute` and verify that it completes without errors. Inspect the generated comparisons and metrics, record evidence in `test/validation.md`, and remove the temporary executed notebook from `output/`.
   - Never call `matplotlib.use()` in library modules. If needed, select a backend only in the CLI execution path of `src/main.py` before importing plotting utilities. Importing `src.main` in a notebook must not change its backend.

### Plan and implementation consistency

`plan/approach.md` must explain the implemented forward model, discretization,
units, objective normalization and signs, regularizers, initialization, update
rules, step-size/line-search decisions, stopping conditions, parameters, and
evaluation conventions. Define every mathematical symbol and explain the method
without requiring the reader to look up an opaque paper equation reference.

`plan/design.md` must describe the shipped modules, callable signatures, defaults,
array shapes/dtypes/units, return values, side effects, file paths, and actual
caller/data flow. Distinguish the standard entry point from notebook comparisons
and optional helpers. Do not describe a proposed, archived, or unimplemented
method as part of the current runtime. Historical context, if retained, must be
explicitly labeled and must not substitute for the current implementation plan.

Before completion, check signatures against source and semantically review the
equations, factors, signs, units, and execution paths. Signature matching alone
does not prove algorithmic consistency. Derive new-task plans from the paper and
validated code; do not blindly copy upstream design documents or shorten them
until essential mathematical and architectural details disappear. Record this
review in `test/validation.md`.

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
`test/validation.md`; the solver README remains self-contained and must
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
For tasks using NCC and NRMSE, the default schema is shown below. Explicit metric-definition and provenance fields may extend it; task-specific metrics replace unsuitable NCC/NRMSE fields as documented below:

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
Record the rationale and measurements in `test/validation.md`. Reference
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

### Metric and result provenance

Distinguish paper or cached upstream reports, upstream results reproduced on the
bound observations, and fresh cleaned-code results. Associate each with its data
hash, code revision, environment, parameters, reference, and metric definition in
`test/validation.md`. Historical values are context, not evidence of this run.

Compare metrics only after checking formulas, normalization, reference arrays,
physical scale, masks, and alignment. Preserve upstream metric definitions under
explicit names; never silently relabel L2-normalized NRMSE as dynamic-range NRMSE.
If both are reported, compute and label both. Keep full numeric precision in
machine-readable results and round only presentation values.

Check Judge, notebook, and visualization metrics against the same independently
known numerical cases. Keep Judge implementation independent of solver imports.
Passing the Judge alone proves neither upstream parity nor physical accuracy.

### Standalone Judge interface

Every new task must include `evaluation/judge.py` with this command interface:

```bash
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
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

Use the validated project-level environment for cleaned tasks. The default runtime
subset is Python standard library, NumPy, SciPy, PyTorch, Matplotlib, and tqdm;
declare only packages actually needed by the task. PyTorch is an allowed general
computation framework, including tensors, GPU computation, automatic
differentiation, neural-network modules, and general-purpose optimizers. Image
readers, HDF5 tools, or other dependencies must have a documented role and
compatibility validation. Existing use elsewhere in this repository is not an
authorization to add a dependency or bypass local mathematics.

Each `requirements.txt` documents the task's subset of shared pinned versions.
Keep pytest and Jupyter tooling in the shared development environment. Inspect
available packages before installing anything; do not install unnecessary tools,
upgrade packages per task, or install upstream domain libraries into the cleaned
runtime. Use a separate maintainer-only environment for upstream verification.

Validate CPU execution where supported. When GPU computation is required, record
OS, Python, numerical backend, driver, GPU memory, and compatible runtime versions.
Document genuine environment exceptions and reasons in `test/validation.md`;
never claim compatibility without running the relevant checks. Public environment
files contain no private paths, credentials, or upstream-only dependencies.

### Completion checklist

Record each gate as passed, failed, or unverified in `test/validation.md`, with
commands and evidence. A failed or unverified mandatory gate prevents completion.

- [ ] Paper, exact upstream revision, data hashes, and equation/function/test mapping are recorded
- [ ] Original and cleaned methods run on the exact bundled observations, with reproducible environments and logs
- [ ] Independent upstream parity fixtures and applicable physics checks pass, with justified tolerances
- [ ] Supplied constants, units, precision, coordinates, and numerical conventions are preserved and tested
- [ ] `python -m src.main` writes the declared `output/reconstruction.npz`; `evaluation/judge.py` passes the fresh result
- [ ] Judge rejects poor outputs and handles missing files/keys, shape/dtype errors, NaN/Inf, and evaluator failures correctly
- [ ] README, raw observations, and environment information suffice to solve the task without hidden assets
- [ ] No truth, target-derived calibration, privileged observation, hidden seed/object, or GitHub URL leaks into solver-visible materials
- [ ] README contains acquisition metadata and the complete output contract; no standalone metadata JSON exists
- [ ] Minimal shared dependencies or a documented exception are validated; the entire cleaned runtime works without upstream domain packages installed
- [ ] Core forward and inverse mathematics is local executable code; source, generator, Judge, tests, and notebook pass symbol/call-role checks for forbidden solver shortcuts
- [ ] `python -m pytest test/ -v` passes unit, metric, Judge, parity, physics, and pipeline checks
- [ ] Notebook actively reconstructs raw data, evaluates fresh results, preserves required comparisons, and executes successfully via nbconvert
- [ ] Historical, reproduced upstream, and fresh cleaned results are labeled; metric formulas and thresholds are consistent and justified
- [ ] README and both plans match actual mathematics, signatures, defaults, units, and execution flow
- [ ] Final file inventory follows Task Template; temporary/duplicate files are removed, provenance is indexed, and required license notices are preserved

Do not bypass a gate by relaxing thresholds, dropping difficult methods or
comparisons, substituting cached reports, or claiming an unexecuted upstream or
notebook run succeeded. Investigate unresolved numerical divergence explicitly.

### Code extraction
- **Extract explicit mathematics** — copy and adapt the relevant published/reference algorithm into local, readable functions, preserving its equations, discretization and numerical conventions. Preserve license notices and record the source version and adaptations in private provenance. Do not substitute a different physical model or easier inverse problem merely to remove a dependency.
- **No domain-library shortcut** -- newly cleaned code must explicitly implement acquisition operators, boundary/unit conventions, objectives and regularizers, gradients/adjoints or Jacobians where required, and inverse updates, initialization, and stopping rules. Closed-form inverses must evaluate their formula directly; artificial iteration is unnecessary. Equations in documentation are not executable mathematical steps.
- **Minimal dependency policy** -- use the Shared environment subset by default. Additional libraries require a specific justified role and must not hide core mathematics. Audit actual imported symbols and call roles throughout source, generator, Judge, tests, and notebook; an allowed package name does not make all its APIs permissible.
- **General computation tools are allowed** -- NumPy, SciPy, and PyTorch may provide FFT, convolution, interpolation, sparse matrices, linear-system solves, eigendecompositions, special functions, automatic differentiation, neural-network layers, and general-purpose optimization such as `scipy.optimize.minimize`/`least_squares` and `torch.optim`. Implement the task-specific acquisition model, discretization, objective, regularization, and reconstruction/training pipeline locally. Document the selected optimizer, parameters, initialization, and stopping rules. If a published method requires custom proximal, reweighting, or other inverse updates, implement those steps explicitly rather than replacing the method with a generic optimizer.
- **No computational-imaging libraries** -- do not use `ehtim` or other computational-imaging/domain reconstruction libraries in the cleaned runtime, generator, Judge, tests, or notebook. Prohibited shortcuts include PyLops FISTA, SimPEG inversion, MNE inverse operators, DIPY reconstruction models, SCICO ADMM, and ready-made reconstruction functions such as `skimage.restoration.richardson_lucy`. Their forward/inverse mathematics must be extracted into readable local code using the permitted general computation tools. A wrapper around a domain solver or an opaque vendored framework does not satisfy this requirement. The isolated upstream verification exception below applies only to the oracle run.
- **Isolated upstream verification exception** — the unmodified upstream implementation may run in a separate maintainer-only oracle environment to produce independent parity fixtures. Private capture scripts and archived upstream source are not part of the cleaned task runtime and must never be imported by its solver, generator, Judge, tests or notebook. They may retain upstream dependencies solely for that verification role. Normal tests consume frozen numeric fixtures with recorded provenance, not upstream imports. Public requirements/environment files contain only cleaned-runtime dependencies; the original verification environment is documented privately.
- **No cross-task imports** — each task must run independently.
- **Version provenance** — preserve the published algorithm and record the upstream commit and cleaned dependencies. Maintained library versions may be used for the shared environment after parity validation; distinguish original-paper reproduction from a modern library port. Package release date alone does not define answer leakage. Never bundle task-specific trained weights or target-derived information as generic environment assets.

### Numerical fidelity

Preserve supplied acquisition constants verbatim, including their stored units,
dtypes, precision, axis order, and real/complex convention. Moving metadata must
not replace an archived value with a recomputed expression. If only a formula is
available, derive its value using upstream conventions and validate it explicitly.
Test copied constants and unchanged arrays for exact equality; use justified,
scale-aware tolerances for computed outputs. Absolute decimal-place checks must
not conceal changes in tiny physical quantities.

Check FFT signs and normalization, half-pixel grids, axis directions,
radians/degrees, per-component noise conventions, objective factors such as
`1/M` versus `1/(2M)`, regularizer signs, flux normalization, initialization,
line searches, and stopping rules. Record discrepancies and corrections.
Nonconvex results may be sensitive to small changes: investigate with controlled
runs varying one factor at a time. Do not attribute divergence to the environment
without evidence or relax acceptance thresholds to conceal it.

### Data
- If real observation data is unavailable, **generate synthetic data** via `src/generate_data.py`. Results should be qualitatively similar to the paper figures, not necessarily identical.
- **Coordinate convention consistency** — `generate_data.py` must use the exact same pixel coordinate convention (e.g., half-pixel offset, axis direction) as `physics_model.py`. A mismatch causes correct amplitudes but wrong phases, yielding χ²/DOF orders of magnitude too high. Always verify by round-tripping: forward-model the ground truth with `physics_model.py` and compare to generated observations. Use a justified noise-weighted residual (χ²/DOF near 1 when its assumptions apply); use numerical tolerance for noiseless data. Where feasible also compare an independent discretization or upstream forward model, since sharing the same erroneous generator and solver can conceal a model error.
- **Parameter unit convention consistency** — when extracting code from a reference library, verify that every parameter passed to internal formulas uses the expected unit convention (dimensionless vs physical). For example, a propagation model may expect a normalised pixel spacing `res = dx·n0/λ` (dimensionless); passing `dx` in μm instead causes a systematic scale error in all frequency-domain quantities (kz, phase advance, scattering strength) without any runtime error. Always check the reference library's convention explicitly before porting.
- **Standard data interface** — all tasks must follow this file layout:
  - `data/raw_data.npz`: all observation arrays **and instrument calibration parameters** (e.g., PSF kernels, system matrices) bundled by descriptive key names (e.g., `measurements`, `vis_cal`, `uv_coords`). Shape always includes a batch dimension as the first axis: `(1, ...)`.
  - `evaluation/ground_truth.npz`: true physical ground truth from simulation or independent validated reference measurements — e.g., ground-truth refractive index, true source image, simulated scatterer positions. Same batch-first convention. **Do not put instrument parameters (PSF, calibration data) here** — those belong in `raw_data.npz` because they are part of the measurement process, not the unknown to be recovered.
  - `evaluation/baseline_reference.npz` (optional): when no true ground truth exists (real observational data), store the reference algorithm's output here for NCC/NRMSE evaluation. Shape follows batch-first convention `(1, ...)`. The `ncc_vs_ref` / `nrmse_vs_ref` in `evaluation/metrics.json` are computed against this file. If both `ground_truth.npz` and `baseline_reference.npz` exist, `ground_truth.npz` takes precedence. Document hidden reference keys and provenance only in `test/validation.md`.
  - README `Acquisition metadata`: imaging parameters only (sensor specs, physical quantities, acquisition geometry); arrays go in visible `raw_data.npz`. Do not create `meta_data.json`. Solver parameters are maintained in `plan/approach.md` and named reference-code constants, not in the acquisition table.
  - Exclude any hidden answer or privileged extra measurement from `raw_data.npz`, including clean fully sampled data for a sparse/noisy inverse problem. Simulator seeds and object definitions that reproduce the answer are maintainer-only. Calibration must be realistically available.
  - Respect repository file-size limits. Convert large arrays to float32 only after validating the effect on physical fidelity and upstream parity. Record storage/computation dtypes and conversion error; do not silently downcast observations or change the validated computation precision.
  - Do not keep redundant `.npy` copies once npz files are in place.

### Testing
- Test each core numerical function using documented fixtures in `test/fixtures/` and tests in `test/unit_test/`; cover pipeline wrappers with smoke and contract tests. Parity and full-pipeline tests belong in `test/comparing_test/`.
- Expected parity outputs must come from independently captured upstream execution, with provenance, not from the cleaned function under test. Analytic cases provide additional independent checks.
- Where applicable, test adjoint identities, finite-difference gradients, conservation, positivity, units, boundaries, and observation consistency. Include malformed submission and failing-quality Judge cases.
- Plot layout needs smoke execution and visual inspection rather than pixel-perfect fixtures. Metric computations in visualization modules are numerical functions and are not exempt from tests.
- Fixture naming: `param_*` (constructor args), `input_*` (function inputs), `config_*` (settings), `output_*` (expected outputs).
- **Deterministic outputs** (e.g., forward model, gradient computation): use explicit numerical tolerances appropriate to dtype/backend (`rtol=1e-10` only where float64 conditioning permits it). Document justified GPU/float32 tolerances; do not describe a tolerance comparison as exact equality.
- **Stochastic outputs** (e.g., noise injection, MCMC samples): test statistical properties only (mean, variance, shape, dtype). Never rely on random seed matching for exact value comparison.
- **SSIM is unreliable for heavily underdetermined problems** — for limited-angle or sparse-measurement tasks (e.g., ODT, interferometry), SSIM can decrease even as NRMSE and NCC improve, because sparsity-promoting regularization or positivity constraints change local structural patterns independently of reconstruction accuracy. Use NRMSE and NCC as primary quality indicators; treat SSIM as secondary for such tasks.

### Pilot reference

The completed tasks `tasks/eht_black_hole_original/` serve as format references. When in doubt, follow their structure, code style, and test patterns.
