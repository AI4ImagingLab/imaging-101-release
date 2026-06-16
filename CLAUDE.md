# Agent Notes

This repository contains the Inverse-101 benchmark: standardized computational imaging tasks plus an evaluation harness for LLM agents.

## Repository Map

```text
evaluation_harness/   CLI, runners, agents, scorers
tasks/                benchmark tasks
docs/                 user-facing guides
scripts/              batch and analysis utilities
```

## Task Contract

Each task should be self-contained:

```text
tasks/<task_name>/
  README.md
  requirements.txt
  main.py
  data/
  plan/
  src/
  evaluation/
  notebooks/
```

Do not introduce cross-task imports. Keep generated outputs under ignored result/output directories.

## Evaluation Commands

```bash
python -m evaluation_harness run --task TASK --mechanism planning --model MODEL
python -m evaluation_harness run --task TASK --mechanism function --target-function physics_model --model MODEL
python -m evaluation_harness run --task TASK --mechanism end2end --level L1 --model MODEL
```

Use `--dry-run` to validate a command without calling a model.

## Development Checks

```bash
python -m evaluation_harness --help
python -m evaluation_harness run --task ct_fan_beam --mechanism end2end --model demo --dry-run
python -m pytest tests/test_cli.py -q
```
