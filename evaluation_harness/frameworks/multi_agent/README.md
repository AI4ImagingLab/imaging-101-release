# Multi-Agent Pipeline Framework — Imaging-101 Evaluation

Multi-agent **Plan→Architect→Code→Judge** pipeline framework.

## Architecture
```
User prompt
    ↓
┌──────────────────────────────────────────┐
│          Multi-Agent Pipeline            │
│                                          │
│  ┌──────────┐    ┌─────────┐            │
│  │ Planner  │───→│ Critic  │ (PASS/REJECT)
│  └──────────┘    └─────────┘            │
│       ↓                                  │
│  ┌──────────┐                            │
│  │Architect │ (code skeleton + design)   │
│  └──────────┘                            │
│       ↓                                  │
│  ┌──────────┐                            │
│  │  Coder   │ (implement each file)      │
│  └──────────┘                            │
│       ↓                                  │
│  ┌──────────┐    ┌─────────┐            │
│  │ Execute  │───→│  Judge  │ (fix loop)  │
│  └──────────┘    └─────────┘            │
└──────────────────────────────────────────┘
    ↓
output/reconstruction.npy
```

## Agents
| Agent       | Role                                      |
|-------------|-------------------------------------------|
| `Planner`   | Analyze task, produce algorithmic plan     |
| `Critic`    | Review plan (PASS/REJECT loop)             |
| `Architect` | Design code skeleton (files + signatures)  |
| `Coder`     | Implement each file from plan + skeleton   |
| `Judge`     | Diagnose failures, route tickets back      |

## Usage
```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --framework multi_agent \
    --level L1 \
    --model gemini-2.5-pro \
    --api-key $API_KEY
```

## Key Files
- `evaluation_harness/multi_agent.py` — Pipeline orchestrator
- `evaluation_harness/agents/` — Individual agent implementations
  - `planner_agent.py`, `architect_agent.py`, `coder_agent.py`, `judge_agent.py`
