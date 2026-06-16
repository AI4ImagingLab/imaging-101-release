# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**imaging-101** is a benchmark suite for evaluating coding agents on computational imaging tasks. It covers 8 domains (physics, chemistry, biology, medicine, astronomy, earth science, mechanics, industrial/commercial) with standardized task structure, reference implementations, and automated evaluation.

## Repository Structure

```
imaging-101/
├── skills/                    # Shared skill library (available to agents)
├── evaluation_harness/        # Benchmark evaluation framework
└── tasks/<task_name>/         # Individual imaging tasks
```

### Task Template

Every task follows this standardized layout:

```
tasks/<task_name>/
├── README.md                  # Problem definition: physics, data description, method hints
├── requirements.txt           # Sandbox environment dependencies
├── main.py                    # Pipeline entry point
├── data/
│   ├── raw_data.npz           # Bundled observation arrays (keys documented in README)
│   ├── ground_truth.npz       # Bundled ground truth arrays (keys documented in README)
│   └── meta_data.json         # JSON: imaging parameters for forward modeling
├── plan/
│   ├── approach.md            # Solution methodology and algorithm description
│   └── design.md              # Code architecture with function signatures
├── src/
│   ├── preprocessing.py       # Raw data → processed observations (y)
│   ├── physics_model.py       # Forward model: x → y
│   ├── solvers.py             # Inverse solvers: y → x_hat
│   ├── visualization.py       # Plotting utilities and metrics
│   └── generate_data.py       # Optional: synthetic data generation
├── notebooks/
│   └── <task_name>.ipynb      # **Required** end-to-end tutorial notebook (user review interface)
└── evaluation/
    ├── metrics.json           # Quantitative results for this task
    ├── reference_outputs/     # Pretrained checkpoints, samples, intermediate arrays
    ├── fixtures/              # Per-function test fixtures
    └── tests/                 # Unit tests, parity tests, integration tests
```

## Evaluation Modes

| Mode              | Agent sees                         | Agent produces      | Evaluated against                |
|-------------------|------------------------------------|---------------------|----------------------------------|
| Plan generation   | README.md + data/meta_data         | plan/               | plan/approach.md + plan/design.md|
| Function-level    | README.md + plan/                  | Single function     | Unit tests + numerical precision |
| End-to-end        | README.md + data/                  | src/ + main.py      | evaluation/reference_outputs/    |
| + Skills          | Same + skills/                     | Same                | Same                             |

## Commands

```bash
cd tasks/<task_name>
pip install -r requirements.txt          # Install dependencies
python main.py                           # Run reconstruction pipeline
python -m pytest evaluation/tests/ -v    # Run all tests
```

## Task Cleaning Principles

Tasks are created by **cleaning** existing research code from published papers and their reference implementations. The goal is to restructure real computational imaging code into the standardized benchmark format.

### Code cleaning workflow

Cleaning follows a **test-driven** process. Never rewrite code first and hope it works — define success criteria from the original code, then clean.

1. **Run the original code** — execute the reference implementation on the task's dataset end-to-end. Save all intermediate and final outputs (preprocessed data, model checkpoints, loss history, reconstructed images/samples).
2. **Define parity tests** — write `test_parity.py` that compares original vs cleaned code at every pipeline stage: preprocessing, forward model, loss functions, solver/architecture, and full-pipeline outputs. These tests guarantee the cleaned code is functionally identical to the original.
3. **Clean the code** — restructure into our `src/` layout. Run parity tests after each change; fix any discrepancies immediately. Do not proceed to the next stage until all parity tests pass.
4. **Generate reference outputs** — from the trained original model, produce reference checkpoints (model weights, loss history), posterior samples/statistics, and ground truth in `evaluation/reference_outputs/`. Save quantitative metrics to `evaluation/metrics.json`.
5. **Write unit tests and fixtures** — per-function tests using `evaluation/fixtures/`.
6. **Write the notebook** (`notebooks/<task_name>.ipynb`) — this is the **primary user-facing deliverable** and must always be generated as the final step of cleaning. The notebook serves as the interface for users to review, validate, and critique the implementation. Requirements:
   - Loads precomputed results from `evaluation/reference_outputs/` so it **runs in seconds** without heavy computation.
   - Contains all key visualizations: ground truth, reconstructions, comparison figures, metrics tables/charts.
   - Includes **commented-out code** for running the full pipeline from scratch (so users can reproduce).
   - Uses the cleaned `src/` modules (not the original library) to demonstrate the implementation works.
   - Should be self-contained and tell a clear story: problem → data → method → results → conclusion.
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

**Data tables** — every key in `raw_data.npz` and `ground_truth.npz` must appear in a table with exactly these columns:

| Key | Shape | Dtype | Description |
Shape must use batch-first convention `(1, ...)`; Description must include units.

**Ground truth handling** — must be explicit either way:
- If `ground_truth.npz` exists → list all keys in a table as above.
- If it does not exist → write explicitly: "No `ground_truth.npz` is provided for this task; [one sentence explaining the reference solution or evaluation approach used instead]." If a `baseline_reference.npz` is provided, document its keys in a `### data/baseline_reference.npz` subsection with the same table format.
- `ground_truth.npz` is for quantities that would only be known from simulation (e.g., ground-truth refractive index, true source image). Instrument calibration data — PSF kernels, system matrices, sensitivity maps — belong in `raw_data.npz` even when they are precisely known, because they describe the measurement process rather than the unknown signal.

**Keywords** — domain-method level only (e.g., `wave imaging`, `self-supervised learning`), not implementation level (e.g., `f-k migration`, `U-Net`, `recorruption`). Aim for 2–4 keywords.

**Problem Description scope** — describe the physical forward model and the inverse problem only: the measurement equation, what makes the problem ill-posed or computationally hard, and the input/output specification. Do not include reconstruction pipelines, training objectives, or algorithm-specific steps — those belong in Method Hints or `plan/approach.md`.

**Method Hints depth** — conceptual level only: name the algorithm family, the key decomposition or regularization, and why it suits the problem. Do not give full hyperparameter tables or step-by-step pseudocode — that belongs in `plan/approach.md`. The hint should guide *what* to implement, not *how* to implement it.

**Formula self-containment** — all mathematical notation must be defined in the README itself. Never write "as in Eq. N of [Paper]" or reference a figure from the original paper as a substitute for explanation.

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

The harness evaluates an agent's reconstruction by computing NCC and NRMSE between
the normalised centre-crop of the agent's output and the reference reconstruction at
`evaluation/reference_outputs/`.  A run passes if `ncc >= ncc_boundary` and
`nrmse <= nrmse_boundary`.

Standard thresholds: `ncc_boundary = 0.9 * baseline_ncc`,
`nrmse_boundary = 1.1 * baseline_nrmse`.

For tasks where NCC/NRMSE are not meaningful (e.g. sharpness-based quality, spectral
metrics), add task-specific keys to `baseline` and define corresponding `*_boundary`
entries.  The harness will check every `*_boundary` key present in the file.

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

### Completion checklist

Before considering a task "done", verify every item:

- [ ] `python main.py` runs to completion
- [ ] `python -m pytest evaluation/tests/ -v` — all tests pass
- [ ] `notebooks/<task_name>.ipynb` exists and executes without errors
- [ ] `evaluation/metrics.json` exists with expected metrics
- [ ] `README.md`, `plan/approach.md`, `plan/design.md` exist
- [ ] `requirements.txt` lists all dependencies

### Code extraction
- **Copy and adapt from reference repos** — clone the reference repo, extract relevant functions, and fit them to our directory structure. Preserve original algorithm logic faithfully; do not rewrite algorithms from scratch.
- **Domain-standard libraries are allowed** — widely-used packages the original code depends on (e.g., `ehtim`, `astropy`, `torchkbnufft`, `scikit-image`) may be imported directly. Only task-specific research packages (e.g., `import dpi`, `import bhnerf`) should have their functions extracted into `src/`. List all dependencies in `requirements.txt`.
- **No cross-task imports** — each task must run independently.
- **No information leakage** — do not depend on packages published after the paper, since an agent solving the task would not have access to them.

### Data
- If real observation data is unavailable, **generate synthetic data** via `src/generate_data.py`. Results should be qualitatively similar to the paper figures, not necessarily identical.
- **Coordinate convention consistency** — `generate_data.py` must use the exact same pixel coordinate convention (e.g., half-pixel offset, axis direction) as `physics_model.py`. A mismatch causes correct amplitudes but wrong phases, yielding χ²/DOF orders of magnitude too high. Always verify by round-tripping: forward-model the ground truth with `physics_model.py`, compare to generated visibilities, and check χ²/DOF ≈ 1.
- **Parameter unit convention consistency** — when extracting code from a reference library, verify that every parameter passed to internal formulas uses the expected unit convention (dimensionless vs physical). For example, a propagation model may expect a normalised pixel spacing `res = dx·n0/λ` (dimensionless); passing `dx` in μm instead causes a systematic scale error in all frequency-domain quantities (kz, phase advance, scattering strength) without any runtime error. Always check the reference library's convention explicitly before porting.
- **Standard data interface** — all tasks must follow this file layout:
  - `data/raw_data.npz`: all observation arrays **and instrument calibration parameters** (e.g., PSF kernels, system matrices) bundled by descriptive key names (e.g., `measurements`, `vis_cal`, `uv_coords`). Shape always includes a batch dimension as the first axis: `(1, ...)`.
  - `data/ground_truth.npz`: true physical ground truth that would only be known from simulation — e.g., ground-truth refractive index, true source image, simulated scatterer positions. Same batch-first convention. **Do not put instrument parameters (PSF, calibration data) here** — those belong in `raw_data.npz` because they are part of the measurement process, not the unknown to be recovered.
  - `data/baseline_reference.npz` (optional): when no true ground truth exists (real observational data), store the reference algorithm's output here for NCC/NRMSE evaluation. Shape follows batch-first convention `(1, ...)`. The `ncc_vs_ref` / `nrmse_vs_ref` in `evaluation/metrics.json` are computed against this file. If both `ground_truth.npz` and `baseline_reference.npz` exist, `ground_truth.npz` takes precedence. Document all keys in a data table in `README.md` under a `### data/baseline_reference.npz` subsection.
  - `data/meta_data.json`: imaging parameters only (sensor specs, physical quantities, acquisition geometry); no data arrays and **no solver/algorithm parameters**. Iteration counts, learning rates, regularization weights, fitting thresholds, and optimizer settings must not appear here — they belong in `plan/approach.md` (documented) and hard-coded in `main.py` as named constants (e.g. `_ADMM_PARAMS = {...}`). Putting solver parameters in `meta_data.json` leaks algorithm information to an evaluation agent that reads this file at test time.
  - Use float32 for large volumes (>50 MB at float64) to stay within the GitHub 100 MB
    file size limit; load with `.astype(np.float64)` before computation.
  - Do not keep redundant `.npy` copies once npz files are in place.
- Keep all existing reference materials (PDFs, `reference_website_github.md`) in each task folder.

### Testing
- Every function in `src/` must have a corresponding test fixture (`evaluation/fixtures/`) and unit test (`evaluation/tests/`). Except for the visualization, we don't need to test the visualization.
- Fixture naming: `param_*` (constructor args), `input_*` (function inputs), `config_*` (settings), `output_*` (expected outputs).
- **Deterministic outputs** (e.g., forward model, gradient computation): test with exact numerical comparison (`rtol=1e-10`).
- **Stochastic outputs** (e.g., noise injection, MCMC samples): test statistical properties only (mean, variance, shape, dtype). Never rely on random seed matching for exact value comparison.
- **SSIM is unreliable for heavily underdetermined problems** — for limited-angle or sparse-measurement tasks (e.g., ODT, interferometry), SSIM can decrease even as NRMSE and NCC improve, because sparsity-promoting regularization or positivity constraints change local structural patterns independently of reconstruction accuracy. Use NRMSE and NCC as primary quality indicators; treat SSIM as secondary for such tasks.

### Pilot reference
The completed tasks `tasks/eht_black_hole_original/` serve as format references. When in doubt, follow their structure, code style, and test patterns.
