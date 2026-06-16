# ReAct Framework — Imaging-101 Evaluation

Single-agent **ReAct** (Reasoning + Acting) framework.

## Architecture
```
User prompt
    ↓
┌─────────────────────┐
│   ReAct Agent Loop   │
│  Thought → Action    │
│  → Observation       │
│  → repeat            │
└─────────────────────┘
    ↓
output/reconstruction.npy
```

## Actions
| Action       | Description                        |
|--------------|------------------------------------|
| `WRITE_FILE` | Create/overwrite a file in sandbox |
| `RUN`        | Execute a shell command            |
| `READ_FILE`  | Read a file from the sandbox       |
| `DONE`       | Signal task completion             |

## Usage
```bash
python -m evaluation_harness run \
    --task eht_black_hole_original \
    --mode end_to_end \
    --framework react \
    --level L1 \
    --model gpt-4o \
    --api-key $OPENAI_API_KEY
```

## Key Files
- `evaluation_harness/agent.py` — ReAct agent loop
- `evaluation_harness/prompts.py` — System and task prompts
- `evaluation_harness/llm_client.py` — LLM API client
