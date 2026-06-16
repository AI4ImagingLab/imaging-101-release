# Inverse-101

**A benchmark and evaluation harness for LLM agents on scientific inverse problems in computational imaging.**

[Project Website](https://starpacker.github.io/agent-imaging-website/) | [Tasks](tasks/) | [Evaluation Harness](evaluation_harness/) | [Evaluation Guide](docs/EVALUATION_GUIDE.md)

---

## What Is This?

Inverse-101 packages 57 computational imaging tasks and a unified evaluation harness for testing LLM agents. Each task asks an agent to reason about a scientific inverse problem, implement part or all of the imaging pipeline, and produce outputs that can be scored automatically.

The benchmark covers astronomy, biology, chemistry and materials, earth science, medicine, and physics. Tasks follow a common layout with `README.md`, `plan/`, `src/`, `data/`, and `evaluation/` folders so agents can be compared consistently.

## Three Evaluation Mechanisms

| Mechanism | CLI value | What it measures | Main score |
| --- | --- | --- | --- |
| Planning | `planning` or `plan` | Can the model design the algorithm and module structure? | LLM-as-judge plan score |
| Function mode | `function` | Can the model implement one module against unit tests? | pytest pass rate |
| End-to-end | `end2end` or `end_to_end` | Can the model build a full reconstruction pipeline? | NCC / NRMSE |

## One-Command Evaluation

After cloning and configuring credentials, choose a model, task, and mechanism:

```bash
python -m evaluation_harness run \
  --model "$MODEL" \
  --task ct_fan_beam \
  --mechanism end2end \
  --level L1 \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --framework react \
  --output results/end_to_end
```

Use `--dry-run` to validate the command without calling an LLM:

```bash
python -m evaluation_harness run \
  --model demo-model \
  --task ct_fan_beam \
  --mechanism end2end \
  --dry-run
```

## Quick Start

```bash
git clone https://github.com/starpacker/inverse-101.git
cd inverse-101
python -m pip install -r evaluation_harness/requirements.txt
python scripts/download_assets.py --task ct_fan_beam
```

Set an OpenAI-compatible endpoint:

```bash
export MODEL="gpt-4o"
export BASE_URL="https://api.openai.com/v1"
export API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:MODEL="gpt-4o"
$env:BASE_URL="https://api.openai.com/v1"
$env:API_KEY="your-api-key"
```

Optional but recommended for isolated execution:

```bash
docker build -t imaging101-sandbox -f evaluation_harness/Dockerfile .
```

If Docker is unavailable, the harness falls back to a local temporary workspace.

## Environment Handling

You do not need to install every task package into your main Python environment. During a real `run`, the harness installs `tasks/<task>/requirements.txt` inside the Docker sandbox; without Docker, it creates an isolated local workspace and venv for that run.

For a quick preflight before spending tokens:

```bash
python -m evaluation_harness check-env --task ct_fan_beam
```

For machine-readable automation:

```bash
python -m evaluation_harness check-env --task ct_fan_beam --json
```

For manual debugging or repeated local experiments, create a reusable task venv:

```bash
python -m evaluation_harness setup-env --task ct_fan_beam --venv .venvs/ct_fan_beam
```

Use `--dry-run` to print the setup commands without installing packages.

Environment tiers:

| Tier | Meaning | What to do |
| --- | --- | --- |
| Tier 1 | Standard pip dependencies such as NumPy/SciPy/Matplotlib | Run normally |
| Tier 2 | Additional or specialized Python packages | Run `check-env`; use Docker or `setup-env` if needed |
| Tier 3 | Task-specific environment files such as `Dockerfile`, `environment.yml`, or `ENVIRONMENT.md` | Read the task notes before running |

## Examples

### Planning

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism planning \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --output results/planning
```

The model writes `plan/approach.md` and `plan/design.md`; the harness scores them against the task reference plan.

### Function Mode

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism function \
  --target-function physics_model \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --output results/function_mode
```

The model implements one module in `src/`; the harness runs the matching pytest file from `evaluation/tests/`.

### End-to-End

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism end2end \
  --level L1 \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --output results/end_to_end
```

The model builds the full pipeline and should produce `output/reconstruction.npy`. The harness scores reconstruction quality with NCC, NRMSE, PSNR, and SSIM where reference data are available.

End-to-end levels:

| Level | Agent receives |
| --- | --- |
| `L1` | Task README, data, requirements |
| `L2` | L1 plus `plan/approach.md` |
| `L3` | L2 plus `plan/design.md` |

## Data And Fixtures

The repository keeps code, task definitions, and lightweight metadata in GitHub. Large task arrays, fixtures, and reference outputs are stored on Hugging Face at [starpacker52/imaging-101](https://huggingface.co/datasets/starpacker52/imaging-101/tree/main/tasks) and tracked by `assets_manifest.json`.

Download assets for one task:

```bash
python scripts/download_assets.py --task ct_fan_beam
```

Download the full benchmark asset set:

```bash
python scripts/download_assets.py --all
```

The downloader restores files into the matching `tasks/<task_name>/data/`, `tasks/<task_name>/evaluation/fixtures/`, and `tasks/<task_name>/evaluation/reference_outputs/` paths, then verifies SHA-256 hashes from the manifest.

## Repository Layout

```text
inverse-101/
  tasks/                 57 benchmark tasks
  evaluation_harness/    CLI, agents, sandbox runners, scorers
  docs/                  evaluation and contribution guides
  scripts/               batch runners and analysis utilities
  assets_manifest.json   Hugging Face asset index with size and SHA-256 hashes
  config_llm.yaml        example model endpoint configuration
```

## Task Layout

```text
tasks/<task_name>/
  README.md              problem statement and data description
  requirements.txt       task-specific Python dependencies
  main.py                full pipeline entry point
  data/                  observations and metadata
  plan/                  reference approach and design
  src/                   implementation modules
  evaluation/            tests, metrics, fixtures, reference outputs
  notebooks/             tutorial or exploratory notebook
```

## Contributing

New tasks should preserve the standard layout and include a clear problem statement, runnable pipeline, tests, and scoring assets. See [docs/NEW_TASK_GUIDE.md](docs/NEW_TASK_GUIDE.md).

## Citation

```bibtex
@misc{inverse101,
  title = {Inverse-101: Benchmarking LLM Agents for Scientific Computational Imaging Problems},
  url = {https://github.com/starpacker/inverse-101},
  year = {2026}
}
```

## License

See [LICENSE](LICENSE) if present in the distribution.
