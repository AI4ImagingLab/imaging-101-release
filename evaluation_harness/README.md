# Evaluation Harness

The harness provides one CLI for three benchmark mechanisms:

```bash
python -m evaluation_harness run --model MODEL --task TASK --mechanism MECHANISM
```

`MECHANISM` can be `planning`, `function`, or `end2end`.

## Install

```bash
python -m pip install -r evaluation_harness/requirements.txt
python scripts/download_assets.py --task ct_fan_beam
docker build -t imaging101-sandbox -f evaluation_harness/Dockerfile .
```

Docker is recommended. Without Docker, the harness uses a local temporary workspace.

## Task Assets

Large task assets are stored on Hugging Face instead of GitHub. Restore one task before evaluation:

```bash
python scripts/download_assets.py --task ct_fan_beam
```

Restore all task assets:

```bash
python scripts/download_assets.py --all
```

`assets_manifest.json` records each asset path, size, and SHA-256 hash. The downloader preserves the original `tasks/...` paths and verifies downloaded files.

## Task Environments

Each benchmark task can declare its own dependencies in `tasks/<task>/requirements.txt`. Real evaluation runs install those dependencies inside the sandbox automatically:

| Runner | Dependency behavior |
| --- | --- |
| Docker | Installs task requirements inside the container |
| Local fallback | Creates a temporary venv in the run workspace and installs task requirements |

Use `check-env` when you want to inspect a task before running it:

```bash
python -m evaluation_harness check-env --task ct_fan_beam
python -m evaluation_harness check-env --task ct_fan_beam --json
```

Use `setup-env` when you want a reusable local venv for debugging outside the harness:

```bash
python -m evaluation_harness setup-env --task ct_fan_beam --venv .venvs/ct_fan_beam
```

Preview the exact commands without installing anything:

```bash
python -m evaluation_harness setup-env --task ct_fan_beam --dry-run
```

The environment report classifies tasks into three tiers: Tier 1 standard pip environments, Tier 2 specialized Python packages, and Tier 3 tasks with explicit environment files such as `Dockerfile`, `environment.yml`, or `ENVIRONMENT.md`.

## Common Arguments

| Argument | Required | Description |
| --- | --- | --- |
| `--task` | yes | Directory name under `tasks/` |
| `--model` | yes | Model identifier sent to the API |
| `--mechanism` / `--mode` | yes | `planning`, `function`, or `end2end` |
| `--base-url` | no | OpenAI-compatible API base URL |
| `--api-key` | no | API key; defaults to `OPENAI_API_KEY` |
| `--framework` | no | `react`, `multi_agent`, `copilot`, or `deepcode` |
| `--output` | no | Results directory |
| `--dry-run` | no | Print resolved config without model calls |

## Planning Evaluation

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism planning \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --output results/planning
```

The model generates `plan/approach.md` and `plan/design.md`. The scorer compares the generated plan with the task reference plan using pairwise LLM judging and rubric scoring.

## Function-Mode Evaluation

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

The model implements one module. The scorer runs `tasks/<task>/evaluation/tests/test_<module>.py` and records test pass rate.

Use the module stem for `--target-function`: `physics_model`, `preprocessing`, `solvers`, `visualization`, or another module available in that task.

## End-to-End Evaluation

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

The model builds the full pipeline and should produce `output/reconstruction.npy`. The scorer computes reconstruction quality metrics against reference data when available.

| Level | Visible guidance |
| --- | --- |
| `L1` | README, data, requirements |
| `L2` | L1 plus `plan/approach.md` |
| `L3` | L2 plus `plan/design.md` |

## External Agent Workflow

For tools that run outside this harness:

```bash
python -m evaluation_harness prepare --task ct_fan_beam --level L1
```

Run the external agent in the prepared workspace, then collect results:

```bash
python -m evaluation_harness collect \
  --task ct_fan_beam \
  --workspace-dir /path/to/workspace \
  --agent-name claude_code \
  --output results/external
```

## Output

Planning and end-to-end runs save JSON files under the selected output directory. Function-mode runs use:

```text
results/function_mode/<task>/<model>_<date>/<module>/result.json
results/function_mode/<task>/<model>_<date>/<module>/src/<module>.py
```

Aggregate a function-mode task:

```bash
python -m evaluation_harness summarize \
  --dir results/function_mode/ct_fan_beam/demo-model_20260613
```

## Development Checks

```bash
python -m evaluation_harness --help
python -m evaluation_harness run --task ct_fan_beam --mechanism end2end --model demo --dry-run
python -m pytest tests/test_cli.py -q
```
