# Evaluation Guide

This guide shows how to run one model on one task with one evaluation mechanism.

## Setup

```bash
git clone https://github.com/starpacker/inverse-101.git
cd inverse-101
python -m pip install -r evaluation_harness/requirements.txt
python scripts/download_assets.py --task ct_fan_beam
```

Set credentials for an OpenAI-compatible endpoint:

```bash
export MODEL="gpt-4o"
export BASE_URL="https://api.openai.com/v1"
export API_KEY="your-api-key"
```

PowerShell:

```powershell
$env:MODEL="gpt-4o"
$env:BASE_URL="https://api.openai.com/v1"
$env:API_KEY="your-api-key"
```

Validate a command without spending tokens:

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism end2end \
  --model demo \
  --dry-run
```

## Task Assets

GitHub stores the task code and lightweight metadata. Large arrays, fixtures, and reference outputs live on Hugging Face at `starpacker52/imaging-101` and are indexed by `assets_manifest.json`.

Download one task before running it:

```bash
python scripts/download_assets.py --task ct_fan_beam
```

Download every task asset:

```bash
python scripts/download_assets.py --all
```

The downloader writes files back into the expected `tasks/<task>/data/`, `tasks/<task>/evaluation/fixtures/`, and `tasks/<task>/evaluation/reference_outputs/` directories and verifies SHA-256 hashes.

## Environment Preflight

Task-specific packages live in `tasks/<task>/requirements.txt`. A real harness run installs them automatically in the sandbox, so users do not have to put every task dependency into the main Python environment.

Before a run, inspect the task environment:

```bash
python -m evaluation_harness check-env --task ct_fan_beam
```

For scripts and CI:

```bash
python -m evaluation_harness check-env --task ct_fan_beam --json
```

For repeated manual debugging:

```bash
python -m evaluation_harness setup-env --task ct_fan_beam --venv .venvs/ct_fan_beam
```

If `check-env` reports Tier 3, read the task-specific environment files before running. These tasks may need a custom Docker image, conda environment, GPU package, or external system dependency.

## Mechanism 1: Planning

Planning evaluation measures whether a model can write a scientifically correct algorithm plan and code design.

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism planning \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --framework react \
  --output results/planning
```

Output includes JSON scores plus a markdown comparison between generated and reference plans.

## Mechanism 2: Function Mode

Function mode measures whether a model can implement one module while seeing the task description, reference plan, and the relevant unit tests.

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism function \
  --target-function physics_model \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --framework react \
  --output results/function_mode
```

Common target modules are `physics_model`, `preprocessing`, `solvers`, and `visualization`.

Summarize a task after running several modules:

```bash
python -m evaluation_harness summarize \
  --dir results/function_mode/ct_fan_beam/<model>_<date>
```

## Mechanism 3: End-to-End

End-to-end evaluation measures whether a model can build the full pipeline and save `output/reconstruction.npy`.

```bash
python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism end2end \
  --level L1 \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --framework react \
  --output results/end_to_end
```

Difficulty levels:

| Level | Agent receives |
| --- | --- |
| `L1` | README, data, requirements |
| `L2` | L1 plus `plan/approach.md` |
| `L3` | L2 plus `plan/design.md` |

End-to-end scores include NCC, NRMSE, PSNR, and SSIM when reference arrays are present.

## Frameworks

| Framework | Use when |
| --- | --- |
| `react` | You want the built-in single-agent loop |
| `multi_agent` | You want the planner/architect/coder/judge pipeline |
| `copilot` | You want to prepare and collect an external-agent run |
| `deepcode` | You want the DeepCode integration |

## External Agents

Prepare a workspace:

```bash
python -m evaluation_harness prepare --task ct_fan_beam --level L1
```

Run your external agent in that workspace, then collect:

```bash
python -m evaluation_harness collect \
  --task ct_fan_beam \
  --workspace-dir /path/to/workspace \
  --agent-name claude_code \
  --output results/external
```

## Result Files

| Mechanism | Output |
| --- | --- |
| Planning | JSON result plus plan comparison markdown |
| Function | `result.json`, generated module source, optional `summary.json` |
| End-to-end | JSON result with quality metrics |

Run logs are written under `logs/interactions/` unless `--log-file` is provided.
