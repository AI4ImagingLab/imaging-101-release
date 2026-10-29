# Public-input inverse-problem benchmark

Tasks are organized as `tasks/<domain>/<task_name>/`. The eight domain folders
are `mathematics`, `physics`, `chemistry_material_science`, `astronomy`,
`earth_science`, `mechanics`, `biology`, and `medicine`. Bare task names and
`domain/task_name` both resolve through `evaluation_harness.task_paths`.

The original 59 task contents are preserved as legacy tasks. Their presence in
the catalogue does not imply that missing assets or old evaluation contracts
have been repaired. The later 240 tasks are the scope of the public-contract
audit; consult the per-task readiness registry rather than assuming that the
number of directories equals the number of independently certified tasks.

## What the evaluated agent receives

Only `README.md`, `data/raw_data.npz`, `requirements.txt`, and the shared
`environment.yml` are exported. README supplies the physical forward model,
all necessary acquisition constants and conventions, legitimate calibration,
the unknowns, and the complete output contract. Mathematical objectives and
regularization weights that define a conditional target are part of the problem
definition. Reference code paths, numerical implementation recipes and hidden
answers are not inputs to the benchmark.

`src/`, `plan/`, `notebooks/`, `test/`, `evaluation/` and existing `output/` remain
maintainer materials. An encoded or read-only ground-truth blob is still a leak;
no self-evaluation script containing one is supplied. No source-repository URL
may occur in exported task text or data.

## Export, solve, judge

Use the prepared project Python environment. No model API or LLM client packages
are needed to export inputs or evaluate a completed submission.

```powershell
python -m evaluation_harness.benchmark prepare `
  --task physics/phase_shifting_holography `
  --workspace D:/benchmark-work/phase_shifting_holography
```

Give the evaluated agent access to **only this export**, not the repository.
An ordinary host directory is not an OS security boundary: a process with host
permissions can navigate outside it. For benchmark runs, use an isolated VM or
container mounting only the export, without the private repository, host home,
credentials or network. Install the pinned environment in the image beforehand.
The Docker runner copies selected public files and never mounts a private
`/workspace_src` tree; networking is disabled during agent execution.

The agent chooses its own code layout and solver. It must compute all quantities
declared in README and save exactly `output/reconstruction.npz`. No generic 2-D
shape, magnitude conversion, implicit crop, fitted intensity scale or guessed
NPZ key overrides that task's contract.

After the agent finishes, invoke the trusted evaluator on the maintainer host:

```powershell
python -m evaluation_harness.benchmark judge `
  --task physics/phase_shifting_holography `
  --workspace D:/benchmark-work/phase_shifting_holography `
  --report D:/benchmark-results/phase_shifting_holography.json
```

This copies only the inert submission into a temporary private directory and
executes the task's trusted Judge with an isolated Python import path. Submitted
Python is never imported or executed. The output report contains `status`,
`passed`, `metrics`, `thresholds`, `errors`, and `judge_exit_code`. Exit codes
are 0 (pass), 1 (valid submission below quality requirements), and 2 (malformed
submission or evaluator error).

Direct evaluation remains available from the task workspace:

```powershell
python evaluation/judge.py --prediction output/reconstruction.npz --report output/judge_report.json
```

Keep the Judge and its reference assets private during solving. The direct
command is a maintainer command, not a reason to copy hidden assets to an agent.

## Scientific fairness

A method reference is not ground truth. Exact matching of a particular seed,
iteration count, approximate root or machine-precision tie can unfairly reject
another valid inverse solution. Where applicable, the revised contracts use
explicit mathematical objectives, calibrated observation consistency, physical
constraints and scientifically justified equivalence classes. All required
quantities are still evaluated. An actually measured ground truth remains truth.

Changes in evaluation semantics are versioned in each affected task's private
`test/validation.md`, with original evaluator backups, independent positive and
negative controls, and numerical evidence. A new semantic contract is not
presented as unchanged reproduction of an old algorithm. Unknown calibration
cannot be invented, and an unresolved target cannot be certified by a static
URL/shape scan or by substituting a reference output for a fresh solve.

The older `.npy` scoring and L2/L3 reference-plan research modes remain legacy
interfaces. Public NPZ tasks use the native standalone Judge and public-only L1.
For external or multi-agent solvers, use the export/judge interface above.
