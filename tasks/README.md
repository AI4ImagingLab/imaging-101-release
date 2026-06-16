---
license: mit
task_categories:
  - other
tags:
  - computational-imaging
  - benchmark
  - inverse-problems
  - test-fixtures
  - astronomy
  - eht
  - black-hole
pretty_name: imaging-101-fixtures
size_categories:
  - n<1K
---

# imaging-101-fixtures

Test fixtures for the **[Inverse-101](https://github.com/starpacker/inverse-101)** benchmark suite, a collection of standardized computational imaging tasks for evaluating coding agents.

## About imaging-101

**imaging-101** is a benchmark for evaluating coding agents on computational imaging inverse problems. It covers domains including physics, chemistry, biology, medicine, astronomy, earth science, mechanics, and industrial/commercial imaging, each with standardized task structure, reference implementations, and automated evaluation.

The full benchmark code, task definitions, and evaluation harness live on GitHub: **https://github.com/starpacker/inverse-101**

## What's in this dataset?

This dataset hosts the **evaluation fixtures** — pre-computed input/output pairs used by per-function unit tests to verify correctness of agent-generated code. Fixtures are stored as `.npz` (NumPy) and `.json` files.

### Current tasks with fixtures

| Task | Domain | Description |
|------|--------|-------------|
| `eht_black_hole_original` | Astronomy | EHT black hole imaging with closure quantities (gain-invariant observables) |
| `eht_black_hole` | Astronomy | EHT black hole imaging from sparse interferometric measurements |
| `eht_black_hole_UQ` | Astronomy | Probabilistic black hole imaging with uncertainty quantification (DPI) |

### Fixture structure

Each task's fixtures are organized by source module:

```
tasks/<task_name>/evaluation/fixtures/
├── preprocessing/       # Data loading and preprocessing fixtures
├── physics_model/       # Forward model fixtures (visibility, PSF, chi-squared, etc.)
├── solvers/             # Inverse solver and regularizer fixtures
└── visualization/       # Metrics computation fixtures
```

**Naming convention:**
- `param_*` — constructor arguments
- `input_*` — function inputs
- `config_*` — configuration/settings
- `output_*` — expected outputs

## How to use

These fixtures are designed to be used together with the [Inverse-101](https://github.com/starpacker/inverse-101) GitHub repository. Clone the repo and place fixtures under each task's `evaluation/fixtures/` directory, or download them directly:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="AI4Imaging/imaging-101",
    repo_type="dataset",
    local_dir="tasks/",
    token=<your hf token>,
)
```

Then run the tests:

```bash
cd tasks/<task_name>
pip install -r requirements.txt
python -m pytest evaluation/tests/ -v
```

## Citation

If you use this benchmark, please cite the imaging-101 project:

```
@misc{imaging101,
  title={imaging-101: A Benchmark Suite for Computational Imaging},
  url={https://github.com/starpacker/inverse-101},
}
```

## License

This dataset follows the same license as the [Inverse-101](https://github.com/starpacker/inverse-101) repository.
