"""Score every task's shipped reference reconstruction through the live
task_scoring.score_task and report whether it reproduces metrics.json baseline.
"""
import os, json
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
from evaluation_harness import task_scoring as ts

REPO = Path(__file__).resolve().parent.parent
ALL = sorted(d for d in os.listdir(REPO / "tasks") if (REPO / "tasks" / d).is_dir())

def kind(t):
    if ts._custom and ts._custom.is_custom(t): return "custom"
    if t in ts.TASK_CONFIG: return "special"
    if t in ts.GENERIC_RECIPES: return "generic"
    return "NONE"

supported = ok = err = 0
rows = []
for t in ALL:
    k = kind(t)
    if k == "NONE":
        rows.append((t, k, "NOT REGISTERED", "")); continue
    supported += 1
    ref = REPO / "tasks" / t / "evaluation" / "reference_outputs"
    res = ts.score_task(t, REPO / "tasks" / t, ref)
    if "error" in res:
        rows.append((t, k, "ERROR", res["error"])); err += 1; continue
    metrics = {kk: v for kk, v in res.items() if kk not in ("boundaries", "passed", "scorer")}
    rows.append((t, k, f"passed={res.get('passed')}", json.dumps(metrics))); ok += 1

print(f"{'TASK':<44}{'KIND':<9}{'STATUS':<16} METRICS")
print("=" * 130)
for t, k, st, m in rows:
    print(f"{t:<44}{k:<9}{st:<16} {m[:62]}")
print("=" * 130)
print(f"registered: {supported}/{len(ALL)}   scored-without-error: {ok}   errors: {err}   "
      f"unregistered: {len(ALL)-supported}")
print(f"breakdown: special={len(ts.TASK_CONFIG)} generic={len(ts.GENERIC_RECIPES)} "
      f"custom={len(ts._custom.CUSTOM_SCORERS) if ts._custom else 0}")
