# Plan (Planning) Evaluation — How-To

This guide explains how to run the **planning** benchmark for a model and how to
compute the plan-quality metrics. It mirrors `function_level_evaluation.md`.

> **Readiness note.** The planning *code* (generation + LLM-as-judge scoring) is
> complete and every task ships a golden reference plan. However, the batch
> planning runs so far used `--no-plan-score` (they saved plan docs but skipped
> scoring), so most existing `results/planning/*.json` have `plan_scores: null`.
> To get scores you must run *with* scoring enabled (see §5). See also the judge
> caveat in §8.

## 1. What planning evaluation is

Given `README.md` + `data/meta_data.json`, the agent produces two design
documents (no code, no tests):

| Output | Prompt phase |
|---|---|
| `plan/approach.md` | Phase 1 — solution methodology / algorithm |
| `plan/design.md` | Phase 2 — code architecture with function signatures |

Unlike function mode this is **not** a ReAct loop — it is a **two-phase
generation** (`runner.py::_run_plan_mode`): generate `approach.md`, then feed it
back to generate `design.md`.

The plan is scored by an **LLM-as-judge** against the task's **golden reference
plan** (`tasks/<task>/plan/approach.md` + `design.md`), combining:

1. **Pairwise** — the judge compares the generated plan vs. the reference over
   `n_pairwise_rounds` rounds (default 3), with **A/B position swapping** to
   cancel position bias → `pairwise_win_rate ∈ [0,1]` (fraction of rounds the
   generated plan is judged ≥ the reference).
2. **Rubric** — the judge scores the plan 1–5 on **6 dimensions**:
   `problem_understanding`, `mathematical_formulation`, `algorithm_design`,
   `code_architecture`, `completeness`, `scientific_correctness` →
   `rubric_weighted_avg ∈ [1,5]`.
3. **Combined**:
   ```
   overall_score = 0.5 * pairwise_win_rate + 0.5 * (rubric_weighted_avg - 1) / 4
   ```
   (rubric mapped 1→0, 5→1; `overall_score ∈ [0,1]`).

### Difficulty levels (what the agent is given)

| Level | Files seeded into the workspace |
|---|---|
| L1 (default) | `README.md` + `data` (plan from scratch) |
| L2 | + `plan/approach.md` (approach given, must produce design) |
| L3 | + full `plan/` (approach + design given) |

## 2. Prerequisites

Same harness interpreter as function mode (`requests` + `pyyaml`; see
`function_level_evaluation.md` §2) and an OpenAI-compatible model endpoint
(`--base-url`, `--api-key`, `--model`). Scoring needs the **same endpoint** (it
reuses the run's LLM client as the judge — see §8). No per-task pip/venv is
needed for planning (no code is executed), so planning is much lighter than
function mode and can run without a GPU.

## 3. Run one task (with scoring)

```bash
.venv_harness/bin/python -m evaluation_harness run \
  --task ct_sparse_view \
  --mechanism planning \
  --level L1 \
  --model "Vendor2/Claude-4.6-opus" \
  --base-url "https://api.gpugeek.com/v1" \
  --api-key "$API_KEY" \
  --output results/planning
```
- Omit `--no-plan-score` to **enable** LLM-judge scoring (it runs automatically
  after generation).
- Add `--level L2` / `--level L3` to change what is seeded.
- `--dry-run` validates arguments without calling the model.

To generate **without** scoring (docs only, e.g. to score later or in bulk):
```bash
... --mechanism planning --no-plan-score --output results/planning
```

## 4. Batch generation

`scripts/run_planning_batch.sh` runs a list of `model|task` pairs, is idempotent
(skips a pair whose `approach.md` **and** `design.md` already exist), and uses
bounded concurrency. **Note:** as written it passes `--no-plan-score`, so it only
produces the plan docs — scoring must be done separately (re-run the pair without
`--no-plan-score`, or add a scoring pass). Edit the `PAIRS=(...)` list and the
`BASE_URL`/`API_KEY` constants at the top, then:
```bash
MAX_CONCURRENCY=8 bash scripts/run_planning_batch.sh
```
Planning has no GPU or heavy-install cost, so it can run on a login node or a
small CPU allocation; it does not need the `ghx4` SLURM setup used for function
mode.

## 5. Enabling scoring

Scoring is part of `run` and fires automatically **unless** `--no-plan-score` is
set and **iff** an LLM client is configured (`scorer.py:127`). It:
- reads the generated `plan/approach.md` + `plan/design.md`,
- reads the golden reference from `tasks/<task>/plan/`,
- runs pairwise (3 rounds) + rubric via `plan_scorer.evaluate_plan`,
- writes the `plan_scores` block into the result JSON.

If `approach.md` or `design.md` is missing, scoring returns an error/empty score
rather than crashing.

## 6. Output layout

Plan mode writes a **flat JSON** per run under `--output`:
```
results/planning/<task>_plan_<framework>_<SafeModel>_<YYYYMMDD_HHMMSS>.json
results/planning/<SafeModel>/<task>/approach.md      # the generated docs
results/planning/<SafeModel>/<task>/design.md
```
The result JSON contains `generated_approach`, `generated_design`, and (when
scored) a `plan_scores` object:

| `plan_scores` field | Meaning |
|---|---|
| `pairwise_win_rate` | fraction of rounds generated ≥ reference (0–1) |
| `pairwise_verdicts` | per-round raw verdicts |
| `pairwise_n_rounds` | rounds run (default 3) |
| `rubric_scores` | dict of the 6 dimensions, each 1–5 |
| `rubric_feedback` | 1–2 sentence justification per dimension |
| `rubric_weighted_avg` | weighted mean of the 6 (1–5) |
| `overall_score` | `0.5*win_rate + 0.5*(avg-1)/4` (0–1) |
| `overall_feedback` | judge's summary |
| `judge_model`, `n_judge_calls` | judge id + call count |

`plan_scores` is `null` when the run was done with `--no-plan-score`.

## 7. Computing the results

Choose a binary **acceptance** criterion (a plan "passes"), then aggregate.
Common criteria:
- **Beats reference**: `pairwise_win_rate >= 0.5` (the plan is judged at least as
  good as the golden plan) — the most natural pass/fail.
- **Quality bar**: `rubric_weighted_avg >= 4.0`, or `overall_score >= 0.5`.

Report continuous means (`pairwise_win_rate`, `rubric_weighted_avg`,
`overall_score`) alongside the pass rate.

```python
import json, glob, collections
MODEL = "Vendor2_Claude-4.6-opus"   # SafeModel form (/ -> _)
WIN_THRESH = 0.5

rows = []
for f in glob.glob(f"results/planning/*_{MODEL}_*.json"):
    d = json.load(open(f))
    ps = d.get("plan_scores")
    if not ps:                      # unscored run
        continue
    task = d["task_name"]
    rows.append((task, ps["pairwise_win_rate"],
                 ps["rubric_weighted_avg"], ps["overall_score"]))

n = len(rows)
if n:
    passed = sum(1 for _, w, _, _ in rows if w >= WIN_THRESH)
    print(f"scored tasks      : {n}")
    print(f"pass (win>= {WIN_THRESH}) : {passed}/{n} = {100*passed/n:.1f}%")
    print(f"mean win_rate     : {sum(r[1] for r in rows)/n:.3f}")
    print(f"mean rubric (1-5) : {sum(r[2] for r in rows)/n:.2f}")
    print(f"mean overall      : {sum(r[3] for r in rows)/n:.3f}")
    # per-dimension rubric means
    dims = collections.defaultdict(list)
    for f in glob.glob(f"results/planning/*_{MODEL}_*.json"):
        ps = json.load(open(f)).get("plan_scores")
        if ps:
            for k, v in ps["rubric_scores"].items():
                dims[k].append(v)
    for k, v in dims.items():
        print(f"  {k:26s} {sum(v)/len(v):.2f}")
```

## 8. Caveats

- **Judge = model under test (self-judging).** There is **no `--judge-model`
  flag**; `evaluate_plan` uses the run's own `llm_client`, so `judge_model`
  equals the model being evaluated. For a fair cross-model comparison, run
  scoring with a single fixed strong judge — e.g., generate all plans with
  `--no-plan-score`, then score them all through one judge model/endpoint (this
  requires a small scoring driver, or temporarily pointing the run's client at
  the judge). Do not compare `overall_score` across models that self-judged.
- **LLM-judge non-determinism.** Verdicts vary run to run; the 3-round pairwise
  with position swapping reduces but does not remove variance. Average over
  rounds/replicates for stable numbers.
- **Scores are mostly not computed yet.** Most existing `results/planning`
  outputs have `plan_scores: null` (batch used `--no-plan-score`); re-run with
  scoring before reporting a plan leaderboard.
- **Reference-plan dependency.** Pairwise needs `tasks/<task>/plan/approach.md` +
  `design.md`. All 57 tasks currently have them; a missing reference disables the
  pairwise half (rubric still runs).
- **Not the same as the multi-agent "critic".** The end-to-end multi-agent
  pipeline has a separate PASS/REJECT `CriticAgent` checklist
  (`agents/planner_agent.py`); that is an internal gate, distinct from this
  LLM-judge plan *quality* score.
```
