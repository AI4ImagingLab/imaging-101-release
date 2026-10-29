# Inverse-101

A benchmark for Agents solving scientific inverse problems from observations.

The repository contains **299 tasks across eight domains** in
`tasks/<domain>/<task_name>/`. The original 59 task contents are preserved.
The later 240 have undergone the [public-input audit](docs/BENCHMARK_AUDIT.md):
**219 are ready for the local benchmark; 21 remain blocked**.
The [task registry](docs/task_registry.json) records each task's status and path.

The evaluated Agent receives only `README.md`, `data/raw_data.npz`, task
requirements and shared environment information. It chooses its implementation
and writes all declared quantities to `output/reconstruction.npz`.
Reference implementations, plans, notebooks, truth and evaluation assets remain
private. Run the Agent in an isolated container or VM with only this export.

```powershell
python -m evaluation_harness.benchmark prepare --task physics/phase_shifting_holography --workspace D:/benchmark-work/holography
python -m evaluation_harness.benchmark judge --task physics/phase_shifting_holography --workspace D:/benchmark-work/holography --report D:/benchmark-results/holography.json
```

See the [current benchmark guide](docs/PUBLIC_BENCHMARK.md) for input isolation,
NPZ contracts, trusted scoring, and exit codes. A method reference is not physical
ground truth. A stored submission passing a Judge is not a blind Agent trial.
Blocked tasks are retained and are refused by the audited benchmark exporter.

The eight domains are mathematics, physics, chemistry and materials, astronomy,
earth science, mechanics, biology, and medicine. The shipped task structure is
defined in [AGENTS.md](AGENTS.md). Existing `.npy` evaluation and reference-plan
research modes are historical interfaces, described in
[the legacy evaluation guide](docs/EVALUATION_GUIDE.md).

Task-specific licenses and data-use limitations remain applicable. Local
benchmark readiness does not establish permission for external redistribution.
See [LICENSE](LICENSE) and each task's notices and provenance.

Large observation and reference arrays use Git LFS. Install Git LFS before cloning
and run `git lfs pull` to materialize the arrays before running a task.
