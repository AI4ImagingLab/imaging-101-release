# Claude Code / Third-Party Agent Framework — Imaging-101 Evaluation

Evaluation framework for **black-box coding agents** (Claude Code, GitHub Copilot, Cursor, Windsurf, etc.).

## Architecture
```
┌─────────────────────────────────────────────┐
│              prepare_copilot_evaluation       │
│                                               │
│  1. Copy visible files (README, data, reqs)  │
│  2. Generate .prompt.md                       │
│  3. Embed obfuscated self_eval.py            │
│  4. Create tmp_L{1,2,3}/ sandbox workspace   │
└─────────────────────────────────────────────┘
              ↓
    Operator pastes prompt into Claude Code
              ↓
    Agent works: writes code, runs tests
              ↓
    output/reconstruction.npy
              ↓
┌─────────────────────────────────────────────┐
│         collect + score_copilot_results       │
│                                               │
│  Compare reconstruction vs ground truth      │
│  Metrics: NRMSE, NCC, PSNR, SSIM            │
└─────────────────────────────────────────────┘
```

## Anti-Cheat Design
- Ground truth is **obfuscated** (compress → XOR → base64)
- `self_eval.py` prints only aggregate metrics, never raw arrays
- Reference implementation (`src/`) is NEVER copied to sandbox
- Test files and fixtures are NEVER copied

## Difficulty Levels
| Level | What's Given                              |
|-------|-------------------------------------------|
| L1    | Task description + data only              |
| L2    | + `plan/approach.md` (algorithmic approach)|
| L3    | + `plan/design.md` (code architecture)    |

## Usage

### Step 1: Prepare sandbox
```bash
python -m evaluation_harness prepare \
    --task eht_black_hole_original \
    --level L1
```

### Step 2: Run agent (manually)
Open the generated `tmp_L1/` folder in Claude Code, paste `.prompt.md`.

### Step 3: Collect & score
```bash
python -m evaluation_harness collect \
    --task eht_black_hole_original \
    --workspace-dir ./tmp_L1 \
    --level L1 \
    --agent-name claude_code
```

## Key Files
- `evaluation_harness/copilot_runner.py` — Sandbox prep and result collection
- `evaluation_harness/copilot_scorer.py` — Scoring for third-party agents
- `evaluation_harness/copilot_prompt.py` — Prompt generation (L1/L2/L3)
