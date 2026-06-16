"""Plan evaluation with ELO-inspired pairwise judging + rubric scoring.

Adapted from inverse_planning_eval's ELO mechanism:
- Pairwise comparison of generated plan vs. reference (golden) plan
- Multiple judge rounds with position swapping to mitigate bias
- Rubric-based dimension scoring (absolute 1-5 per dimension)
- Combined scoring: 0.5 * pairwise_win_rate + 0.5 * rubric_normalized

Since imaging-101 evaluates one model at a time (not a multi-model
tournament), we compare the generated plan against the reference
plan/approach.md and plan/design.md that ship with each task.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from .llm_client import LLMClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pairwise judge prompt (adapted from inverse_planning_eval/evaluation/elo.py)
# ---------------------------------------------------------------------------

PAIRWISE_JUDGE_PROMPT = """\
You are an expert impartial judge evaluating AI-generated solution plans \
for a computational imaging task.

Your goal is to determine which of two plans is better, or if they are tied.

**Task Description:**
{task_description}

**Plan A:**
{plan_a}

**Plan B:**
{plan_b}

**Evaluation Criteria:**
1. **Correctness**: Does the plan correctly describe the physics, math, and algorithms?
2. **Completeness**: Does it cover all pipeline stages (data loading, forward model, \
solver, output)?
3. **Mathematical Precision**: Are equations explicit and correct (not vague)?
4. **Code Architecture**: Are function signatures, module structure, and data flow clear?
5. **Implementability**: Could a developer implement the full pipeline from this plan alone?

**Verdict:**
Compare Plan A against Plan B.
If Plan A is better, output "[[A is Better]]".
If Plan B is better, output "[[B is Better]]".
If they are functionally equivalent, output "[[Tie]]".

**Reasoning:**
Provide a brief explanation (2-3 sentences) for your decision.

Output Format:
[[Verdict]]
Reasoning...
"""


# ---------------------------------------------------------------------------
# Rubric scoring prompt (absolute 1-5 per dimension)
# ---------------------------------------------------------------------------

RUBRIC_JUDGE_PROMPT = """\
You are an expert scientific computing reviewer evaluating an AI-generated \
solution plan for a computational imaging problem.

You will be given:
1. The original problem description (README)
2. A generated approach document (approach.md)
3. A generated code design document (design.md)

Score each dimension on a 1-5 scale:
  1 = Missing or completely wrong
  2 = Major gaps or significant errors
  3 = Adequate but with notable omissions
  4 = Good with minor gaps
  5 = Excellent, thorough and correct

You MUST respond with ONLY a JSON object:
{{
  "problem_understanding": {{"score": <1-5>, "feedback": "<1-2 sentences>"}},
  "mathematical_formulation": {{"score": <1-5>, "feedback": "<1-2 sentences>"}},
  "algorithm_design": {{"score": <1-5>, "feedback": "<1-2 sentences>"}},
  "code_architecture": {{"score": <1-5>, "feedback": "<1-2 sentences>"}},
  "completeness": {{"score": <1-5>, "feedback": "<1-2 sentences>"}},
  "scientific_correctness": {{"score": <1-5>, "feedback": "<1-2 sentences>"}},
  "overall_feedback": "<2-3 sentence summary>"
}}
"""

RUBRIC_WEIGHTS = {
    "problem_understanding": 0.15,
    "mathematical_formulation": 0.25,
    "algorithm_design": 0.20,
    "code_architecture": 0.20,
    "completeness": 0.10,
    "scientific_correctness": 0.10,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlanScore:
    """Complete plan evaluation result."""

    # Pairwise comparison (ELO-inspired)
    pairwise_win_rate: float = 0.0        # 1.0 = always beats reference
    pairwise_verdicts: list[dict] = field(default_factory=list)
    pairwise_n_rounds: int = 0

    # Rubric scoring (absolute 1-5)
    rubric_scores: dict[str, float] = field(default_factory=dict)
    rubric_feedback: dict[str, str] = field(default_factory=dict)
    rubric_weighted_avg: float = 0.0

    # Combined
    overall_score: float = 0.0  # 0.5 * win_rate_normalized + 0.5 * rubric_normalized
    overall_feedback: str = ""

    # Metadata
    judge_model: str = ""
    n_judge_calls: int = 0


# ---------------------------------------------------------------------------
# Pairwise comparison with position swapping
# ---------------------------------------------------------------------------

def _parse_verdict(content: str) -> float:
    """Parse judge verdict -> score for Plan A. 1.0=A wins, 0.0=B wins, 0.5=tie."""
    if not content:
        return 0.5
    if "[[A is Better]]" in content:
        return 1.0
    if "[[B is Better]]" in content:
        return 0.0
    if "[[Tie]]" in content:
        return 0.5
    # Fallback: case-insensitive search
    lower = content.lower()
    if "a is better" in lower:
        return 1.0
    if "b is better" in lower:
        return 0.0
    return 0.5


def _run_pairwise(
    client: LLMClient,
    task_desc: str,
    generated_plan: str,
    reference_plan: str,
    n_rounds: int = 3,
) -> tuple[float, list[dict]]:
    """Run pairwise comparison with position swapping.

    To mitigate position bias, we alternate which plan is A vs B:
      Round 1: generated=A, reference=B
      Round 2: reference=A, generated=B
      Round 3: generated=A, reference=B

    Returns (win_rate_for_generated, list_of_verdicts).
    """
    verdicts = []
    gen_scores = []

    for i in range(n_rounds):
        if i % 2 == 0:
            plan_a, plan_b = generated_plan, reference_plan
            swap = False
        else:
            plan_a, plan_b = reference_plan, generated_plan
            swap = True

        prompt = PAIRWISE_JUDGE_PROMPT.format(
            task_description=task_desc,
            plan_a=plan_a,
            plan_b=plan_b,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response, _ = client.chat(messages)
            raw_score = _parse_verdict(response)
            gen_score = (1.0 - raw_score) if swap else raw_score
        except Exception as e:
            log.error("Pairwise judge call %d failed: %s", i + 1, e)
            gen_score = 0.5
            response = f"Error: {e}"

        verdicts.append({
            "round": i + 1,
            "generated_is_plan": "A" if not swap else "B",
            "raw_verdict_for_A": raw_score,
            "score_for_generated": gen_score,
            "reasoning": response[:500] if response else "",
        })
        gen_scores.append(gen_score)

    win_rate = sum(gen_scores) / len(gen_scores) if gen_scores else 0.5
    return win_rate, verdicts


# ---------------------------------------------------------------------------
# Rubric scoring
# ---------------------------------------------------------------------------

def _run_rubric(
    client: LLMClient,
    readme: str,
    approach: str,
    design: str,
) -> tuple[dict, dict, float]:
    """Score plan on 6 rubric dimensions. Returns (scores, feedback, weighted_avg)."""
    prompt = f"""\
== Problem Description (README.md) ==
{readme}

== Generated Approach (approach.md) ==
{approach}

== Generated Code Design (design.md) ==
{design}

Evaluate the quality of these plan documents. Respond with ONLY the JSON object.
"""
    messages = [
        {"role": "system", "content": RUBRIC_JUDGE_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        response, _ = client.chat(messages)
    except Exception as e:
        log.error("Rubric judge call failed: %s", e)
        return {}, {}, 0.0

    # Parse JSON
    json_text = response.strip()
    if json_text.startswith("```"):
        json_text = re.sub(r"^```(?:json)?\s*\n?", "", json_text)
        json_text = re.sub(r"\n?```\s*$", "", json_text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\\s\\S]*\}", response)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return {}, {}, 0.0
        else:
            return {}, {}, 0.0

    scores = {}
    feedback = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for dim, weight in RUBRIC_WEIGHTS.items():
        if dim in data and isinstance(data[dim], dict):
            s = float(data[dim].get("score", 0))
            scores[dim] = s
            feedback[dim] = data[dim].get("feedback", "")
            weighted_sum += s * weight
            total_weight += weight

    avg = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0
    overall_fb = data.get("overall_feedback", "")
    feedback["overall"] = overall_fb

    return scores, feedback, avg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_plan(
    client: LLMClient,
    readme: str,
    generated_approach: str,
    generated_design: str,
    reference_approach: str = "",
    reference_design: str = "",
    n_pairwise_rounds: int = 3,
) -> PlanScore:
    """Evaluate plan quality using ELO-inspired pairwise + rubric scoring.

    Parameters
    ----------
    client : LLMClient
        LLM client for judge calls.
    readme : str
        Task README.md.
    generated_approach, generated_design : str
        The model-generated plan documents.
    reference_approach, reference_design : str
        The reference (golden) plan documents from the task.
    n_pairwise_rounds : int
        Number of pairwise comparison rounds (with position swapping).

    Returns
    -------
    PlanScore
    """
    result = PlanScore(judge_model=client.config.model)

    if not generated_approach or generated_approach.startswith("[File not found"):
        result.overall_feedback = "approach.md was not generated"
        return result
    if not generated_design or generated_design.startswith("[File not found"):
        result.overall_feedback = "design.md was not generated"
        return result

    generated_plan = (
        f"## Approach\n{generated_approach}\n\n## Code Design\n{generated_design}"
    )

    # --- 1. Pairwise comparison vs. reference ---
    if reference_approach and reference_design:
        reference_plan = (
            f"## Approach\n{reference_approach}\n\n## Code Design\n{reference_design}"
        )
        task_desc = readme[:3000]

        log.info("Running pairwise comparison (%d rounds)...", n_pairwise_rounds)
        win_rate, verdicts = _run_pairwise(
            client, task_desc, generated_plan, reference_plan, n_pairwise_rounds
        )
        result.pairwise_win_rate = round(win_rate, 3)
        result.pairwise_verdicts = verdicts
        result.pairwise_n_rounds = n_pairwise_rounds
        result.n_judge_calls += n_pairwise_rounds
    else:
        log.warning("No reference plan available — skipping pairwise comparison")
        result.pairwise_win_rate = -1

    # --- 2. Rubric scoring ---
    log.info("Running rubric scoring...")
    scores, feedback, avg = _run_rubric(client, readme, generated_approach, generated_design)
    result.rubric_scores = scores
    result.rubric_feedback = feedback
    result.rubric_weighted_avg = avg
    result.n_judge_calls += 1

    # --- 3. Combined score ---
    if result.pairwise_win_rate >= 0:
        rubric_norm = (avg - 1.0) / 4.0  # maps 1->0, 5->1
        result.overall_score = round(
            0.5 * result.pairwise_win_rate + 0.5 * rubric_norm, 3
        )
    else:
        result.overall_score = round((avg - 1.0) / 4.0, 3)

    result.overall_feedback = feedback.get("overall", "")
    return result
