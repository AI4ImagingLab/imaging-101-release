# Prefill Recovery for "Thought-only" Reasoning Models

## The problem

Some reasoning models (observed with **kimi-k3**, and earlier during plan
generation) reply with only a `Thought:` line and then **stop**
(`finish_reason: stop`, ~50–110 completion tokens), never emitting an `Action:`.

Our agent uses a **text ReAct protocol**: one assistant response must contain a
`Thought:` *and* exactly one action block (`Action: RUN` / `READ_FILE` /
`WRITE_FILE` / `DONE`). A response with no parseable action is scored as
`FORMAT_ERROR (no actions found)`, and the iteration is wasted. For kimi-k3 this
happened ~6 times per function-mode run (and 15–29 times per plan-generation
run), pinning every run at the iteration cap and leaving many modules never
written (function-mode strict ≈ 6%; 15/57 plan tasks produced no docs).

## Root cause (diagnosed, not guessed)

- **Not a parser bug** and **not** a hidden-`reasoning_content` issue — `content`
  *was* populated; it simply held only the `Thought:`.
- It is a **tool-calling instinct**: the model states its intent and yields,
  expecting a tool round-trip, because our harness uses a *text* action protocol
  rather than native tool-calls. It stalls mainly when the next step is
  exploratory ("let me read the data first").
- Two recovery strategies were tested: a stronger *text nudge* in the observation
  failed; an **assistant prefill** succeeded.

## The fix

Force the model past its stop point by **prefilling the assistant turn** so the
completion must continue into the action. Two small, shared changes (used by both
plan and function modes via the same `Agent.run()`):

### 1. `evaluation_harness/llm_client.py` — `chat(..., prefill=...)`

`chat()` gains an optional `prefill` argument. When set, an assistant turn
holding `prefill` is appended to the request messages, so the model continues
*from* it instead of starting a fresh turn:

```python
send_messages = messages
if prefill is not None:
    send_messages = list(messages) + [{"role": "assistant", "content": prefill}]
```
(Everything else — retries, token accounting — is unchanged.)

### 2. `evaluation_harness/frameworks/react/agent.py` — recovery loop

In the ReAct loop, immediately after parsing a response: if **no action was
found but the response contains `Thought:`**, re-request with the model's own
response plus a trailing `\nAction:` prefilled, up to `self.prefill_retries`
(= 3) times:

```python
if not actions and "Thought:" in response_text:
    for _ in range(self.prefill_retries):
        prefill = response_text.rstrip() + "\nAction:"
        cont, _ = self.client.chat(windowed, prefill=prefill)
        # gateway may echo a full response (already has "Action:") or just the tail
        combined = cont if "Action:" in cont else ("Action:" + cont)
        parsed = self._parse_all_actions(combined)
        good = [a for a in parsed if a[0] != "FORMAT_ERROR"]
        if good:                       # accept only a real, parseable action
            response_text = combined
            actions = parsed
            break
```

Key robustness details:
- **Normalization** — the gateway sometimes returns a full fresh response (which
  already contains `Action:`) and sometimes only the continuation; `combined`
  handles both.
- **Guarded acceptance** — the recovery is accepted only if it yields a
  non-`FORMAT_ERROR` action, so a continuation that is *another* bare `Thought:`
  does not get mistaken for success.
- **`self.prefill_retries = 3`** — set in `Agent.__init__`.

## Worked example: how one format error is removed

A format error occurs when a turn contains no parseable `Action:`. kimi-k3
produces exactly that — a bare `Thought:` — so without the fix the turn is
counted as `FORMAT_ERROR` and burned.

**Turn (as sent):** the conversation so far, ending with the user's last
`Observation:`. kimi-k3 replies and stops early:
```
Thought: Let me start by exploring the workspace structure to understand the setup.
```
`_parse_all_actions("Thought: …")` → `[]` (no `Action:`) → would be `FORMAT_ERROR`.

**Recovery fires** (`not actions and "Thought:" in response_text`). We re-send the
*same* conversation but append an assistant turn whose content is the model's own
reply plus a trailing `\nAction:`:
```
role=assistant, content=
    "Thought: Let me start by exploring the workspace structure to understand the setup.\nAction:"
```
Because the assistant message ends mid-sentence at `Action:`, the API must
**continue** it rather than begin a fresh turn, so the completion is forced to be
whatever comes after `Action:` — e.g. ` RUN\nCommand: ls -la /workspace`.

**Normalize + re-parse:** `combined = "Action:" + cont` (the leading `Action:`
we prefilled is prepended back), giving:
```
Action: RUN
Command: ls -la /workspace
```
`_parse_all_actions(combined)` now returns a real `RUN` action → the guard
`good = [a for a in parsed if a[0] != "FORMAT_ERROR"]` is non-empty → accepted.
The log records `Recovered action via prefill continuation`, and the turn
executes normally instead of being wasted. If the continuation were *another*
bare `Thought:` (no `Action:`), the guard rejects it and we retry (≤3), so a
format error is only emitted when the model resists all retries.

Net effect on the format-error count: each successful recovery converts one
would-be `FORMAT_ERROR` into a productive action **within the same iteration**,
which is why the per-run format-error total collapses (see below).

## Properties

- **Targeted** — fires only on the "Thought-only, no Action" failure mode; a
  no-op for models that already emit `Thought + Action`.
- **Model-agnostic** — keeps the existing text protocol; no per-model branching
  (aside from the pre-existing rule that omits `temperature` for `kimi*` models).
- **Bounded** — at most `prefill_retries` (3) extra LLM calls per stalled turn.
- **Iteration-preserving** — a recovered turn costs extra *LLM calls*, not extra
  *iterations*; at `--max-iterations 10` kimi-k3 keeps all 10 productive
  iterations, so comparability with the other iter-10 model runs is retained.

## Empirical results

**Format-error removal (kimi-k3):**

| Run | Format errors | Prefill recoveries fired |
|---|---|---|
| Full 171-pair, **no fix** | **1003** | — |
| Full 171-pair, **with fix** | **205** (−80%) | **492** |

492 stalled turns were converted into real actions; the remaining 205 are turns
where the model kept returning a bare `Thought:` through all 3 retries.

**Downstream effect on scores:**
- **Plan generation:** 42/57 → **57/57** tasks produced docs; residual format
  errors 15–29 → 0–5 per run.
- **Function mode, full 171 (kimi-k3, iter 10):** strict module passes
  **11 (6.4%) → 34 (19.9%)**; micro 21.7% → 50.6%. Modules that were previously
  unreachable now write and pass (e.g. `exoplanet_imaging` solvers 0→14/14).
- **Function mode, 5-task probe:** strict **1/15 → 7/15**; format errors
  ~6/pair → ~1/pair.

## Caveats

- **Not a full fix.** The recovery has a 3-retry budget; if the model resists and
  keeps returning bare `Thought:` turns, the module is still never written. In
  the 5-task test, 2/15 pairs still stalled this way. The remaining failures are
  now mostly **genuine capability** (code written but wrong), which is the
  desired benchmark signal.
- **Extra API cost.** Each stalled turn issues up to 3 additional requests, so
  heavy-stalling models do more API work. For plan generation this was paired
  with `--max-iterations 30` + generous timeouts; function mode stalls less and
  runs fine at `--max-iterations 10`.
- **Don't over-tune.** Raising `prefill_retries` or the iteration cap recovers a
  little more format-loss but starts compensating for a model that genuinely
  doesn't follow the ReAct contract — and breaks comparability with the other
  models' iter-10 runs.
```
