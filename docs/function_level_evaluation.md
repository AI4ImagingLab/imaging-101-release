# Function-Level Evaluation — How-To

This guide explains how to run the **function-level** benchmark for a model and
how to compute the pass-rate metrics from the outputs.

## 1. What function-level evaluation is

For each task, the agent must implement **one target module** from scratch
(`src/<module>.py`) given the task's `README.md`, `plan/`, and the module's unit
tests. It is scored by running those tests. Three modules are evaluated per task:

| Module | File the agent must write | Test file it is scored against |
|---|---|---|
| `preprocessing` | `src/preprocessing.py` | `evaluation/tests/test_preprocessing.py` |
| `physics_model` | `src/physics_model.py` | `evaluation/tests/test_physics_model.py` |
| `solvers` | `src/solvers.py` | `evaluation/tests/test_solvers.py` |

With 57 tasks × 3 modules there are **171 (task, module) pairs**.

The agent is a **single ReAct loop** (`evaluation_harness/frameworks/react/agent.py`):
it iterates `RUN` / `READ_FILE` / `WRITE_FILE` / `DONE` actions up to
`--max-iterations`, stopping on `DONE` or the cap. The harness copies the task
into a temp workspace, builds an **isolated per-task venv** (`pip install -r
requirements.txt`), lets the agent write the target module, then runs pytest on
the module's test file. The non-target modules are seeded from the reference
`src/` so cross-module imports work; the **target module is withheld** and must
be produced by the agent.

## 2. Prerequisites

- **Harness interpreter** with `requests` + `pyyaml` (numpy/scipy/torch are
  inherited from the base env). The LLM client is plain `requests` against any
  OpenAI-compatible `/chat/completions` endpoint.
  ```bash
  BASEPY=/sw/user/python/miniforge3-pytorch-24.11.3/bin/python   # any Python 3.10+ with numpy
  "$BASEPY" -m venv --system-site-packages .venv_harness
  .venv_harness/bin/python -m pip install -q requests pyyaml
  ```
- **Network egress** on the machine that runs it (needed for both the model API
  and the per-task `pip install`).
- **Model endpoint**: `--base-url` + `--api-key` for an OpenAI-compatible gateway,
  and the exact `--model` id (see `model_config.txt`).

## 3. Run one (task, module)

```bash
.venv_harness/bin/python -m evaluation_harness run \
  --task ct_fan_beam \
  --mechanism function \
  --target-function preprocessing \
  --model "Vendor2/Claude-4.6-opus" \
  --base-url "https://api.gpugeek.com/v1" \
  --api-key "$API_KEY" \
  --max-iterations 10 \
  --output results
```
Notes:
- `--mechanism function` (alias `--mode`) selects function-level.
- `--max-iterations` default is 20; we use **10** for the model comparison.
- `--max-tokens` default 32768 (lower it for verbose reasoning models if they
  self-truncate).
- Add `--dry-run` to validate arguments without calling the model.

## 4. Run the full matrix for one model

`scripts/run_function_matrix.py` enumerates all 171 pairs, runs each via the
harness, and is **idempotent** (skips pairs that already have a `result.json`).
It is configured by environment variables:

| Env var | Meaning | Typical value |
|---|---|---|
| `MODEL` | model id | `Vendor2/GPT-5.5` |
| `OUTPUT` | results root (isolate each run/model here) | `results_gpt55` |
| `MAX_ITER` | `--max-iterations` passed through | `10` |
| `MAX_CONCURRENCY` | parallel pairs | `4`–`6` |
| `FORCE` | `1` = re-run even if an older dated result exists | `0`/`1` |
| `RUN_DATE` | with `FORCE=1`, only results stamped this date count as "done" (resume-safe fresh re-run) | `$(date +%Y%m%d)` |

The API endpoint/key are constants near the top of the script — set them there.

```bash
OUTPUT=results_gpt55 MODEL="Vendor2/GPT-5.5" MAX_ITER=10 MAX_CONCURRENCY=6 \
  .venv_harness/bin/python scripts/run_function_matrix.py
```

**Idempotency / re-runs:**
- Default (`FORCE=0`): skips any pair with an existing result in `OUTPUT` — use to
  *resume* an interrupted run.
- Fresh full re-run: `FORCE=1 RUN_DATE=$(date +%Y%m%d)` — re-runs everything but
  only treats *today's* results as done, so it is still resumable and never
  overwrites older dated runs.
- **Give each model (and each replication) its own `OUTPUT` root.** The runner's
  logs (`_matrix_logs/`, `_summary.tsv`) are not model-namespaced, so concurrent
  runs sharing a root would clobber each other's logs.

## 5. Run on the cluster (SLURM)

`scripts/run_function_matrix_generic.sbatch` is a parametrized batch job
(DeltaAI `ghx4`, account `bgkc-dtai-gh`). It builds a **node-local, job-private
venv** so the interpreter can't be disturbed by other processes, then runs the
matrix.

```bash
mkdir -p results_gpt55/function_mode/_matrix_logs
sbatch -J gpt55_fn \
  --export=ALL,MODEL=Vendor2/GPT-5.5,OUTROOT=results_gpt55,MAXIT=10 \
  scripts/run_function_matrix_generic.sbatch
```
Run several models in parallel by submitting once per model with distinct
`OUTROOT`. Recommended resources: `--mem=180g`, `--gpus=1`, `MAX_CONCURRENCY=6`
(96 GB / conc-6 has OOM-killed; 180 GB is safe). Wall time ~1–7 h per model
depending on how many iterations the model uses.

## 6. Output layout

Per pair, the harness writes:
```
<OUTPUT>/function_mode/<task>/<SafeModel>_<YYYYMMDD>/<module>/
    result.json         # scores + metadata
    pytest_output.txt   # raw pytest log
    src/<module>.py      # the agent's code (for modules that ran)
```
`<SafeModel>` is the model id with `/`→`_` (e.g. `Vendor2_GPT-5.5`). A run may
span two date stamps if it crosses midnight; aggregate across both.

Key `result.json` fields:

| Field | Meaning |
|---|---|
| `tests_total` | # tests pytest **collected** (see caveat in §7) |
| `tests_passed`, `tests_failed` | pass/fail counts |
| `test_pass_rate` | `tests_passed / tests_total` |
| `iterations` | ReAct iterations used |
| `stopped_reason` | `done` \| `max_iterations` \| `error` |
| `total_tokens`, `llm_calls`, `wall_time_seconds` | cost/latency |

## 7. Computing the results

A **module passes** iff `tests_total > 0` and `tests_passed == tests_total`
(all tests in `test_<module>.py` pass). Define, per model:

- **Strict (module) pass rate** — fraction of the 171 pairs that fully pass.
  Best cross-model metric (denominator fixed at 171).
- **Per-module strict** — fully-passing modules out of 57, per module.
- **Fully-solved tasks** — tasks where *all three* modules pass (out of 57).
- **Micro** — total tests passed ÷ total tests collected (pooled).
- **Macro** — mean over pairs of `tests_passed / tests_total`.

Aggregation script:

```python
import json, glob, collections
ROOT, MODEL = "results_gpt55", "Vendor2_GPT-5.5"   # SafeModel form

best = {}
for f in glob.glob(f"{ROOT}/function_mode/*/{MODEL}_*/*/result.json"):
    p = f.split("/"); r = json.load(open(f))
    best[(p[-4], p[-2])] = (r["tests_passed"], r["tests_total"])   # (task, module)

valid  = [(a, b) for a, b in best.values() if b > 0]
strict = sum(1 for a, b in valid if a == b)
tp = sum(a for a, b in valid); tt = sum(b for a, b in valid)
macro  = sum(a / b for a, b in valid) / len(valid)

print(f"strict pairs : {strict}/171 = {100*strict/171:.1f}%")
print(f"micro        : {tp}/{tt} = {100*tp/tt:.1f}%")
print(f"macro        : {100*macro:.1f}%")

for m in ("preprocessing", "physics_model", "solvers"):
    sp = sum(1 for (t, mm), (a, b) in best.items()
             if mm == m and b > 0 and a == b)
    print(f"  {m:14s} strict {sp}/57")

# fully-solved tasks (all 3 modules pass)
bytask = collections.defaultdict(dict)
for (t, m), (a, b) in best.items():
    bytask[t][m] = (b > 0 and a == b)
full = [t for t, d in bytask.items() if len(d) == 3 and all(d.values())]
print(f"fully-solved tasks: {len(full)}/57")
```

**Treating unfinished pairs as failures.** If a model never produced a module
(job killed / iteration cap), it has no `result.json`. For a complete /171
denominator, count each missing pair as a failure (0 passed). Strict-over-171 is
unaffected; for micro/macro, add `0 / <reference-test-count>` where the reference
count is the max `tests_total` observed for that pair across models.

## 8. Caveats

- **`tests_total` is per-run, not fixed.** If the agent's module fails to import,
  pytest errors at collection and reports **1** test instead of the true N. So
  micro/macro **denominators shift between runs**, and the micro rate is biased
  optimistically (a `0/21` collapse shows up as `0/1`). Prefer the **strict**
  metric for cross-model comparison, or normalize micro against a fixed reference
  test count per pair.
- **Reasoning models (`--max-tokens`).** Some models (e.g. GLM, Kimi) put chain
  of thought in a hidden `reasoning_content` channel and/or generate very long
  replies. With too small a budget they return empty `content`; the default
  32768 is usually enough. Very verbose models can exhaust the token budget or
  overflow the context window and stall — a known failure mode, not an API issue.
- **Iteration budget matters.** Fast models finish in 2–4 iterations; weaker ones
  sit at the cap. Compare models only at the **same `--max-iterations`**
  (we standardized on 10; a 20-iter run is not directly comparable).
- **OOM.** Concurrency × heavy tasks can trip the node OOM killer; use ≥180 GB or
  lower `MAX_CONCURRENCY`.
- **Broken shared venv.** Prefer a job-private venv (as the sbatch does) so a
  long run's interpreter can't be repointed/removed mid-flight.
```
