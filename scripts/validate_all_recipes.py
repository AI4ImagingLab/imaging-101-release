"""Validate every registered task-aware recipe (special + generic) reproduces
its metrics.json baseline, via the live task_scoring.validate_against_baseline.
"""
from pathlib import Path
from evaluation_harness import task_scoring as ts

REPO = Path(__file__).resolve().parent.parent
registered = sorted(set(ts.TASK_CONFIG) | set(ts.GENERIC_RECIPES))

ok = bad = 0
print(f"Registered task-aware recipes: {len(registered)}  "
      f"(special={len(ts.TASK_CONFIG)}, generic={len(ts.GENERIC_RECIPES)})")
print("=" * 90)
for t in registered:
    kind = "special" if t in ts.TASK_CONFIG else "generic"
    res = ts.validate_against_baseline(t, REPO / "tasks" / t)
    if "error" in res:
        print(f"  [{kind:7}] {t:<44} ERROR: {res['error']}"); bad += 1; continue
    tag = "OK" if res["match"] else "MISMATCH"
    diffs = {k: f"{d['computed']}~{d['baseline']}" for k, d in res["diffs"].items()}
    print(f"  [{kind:7}] {t:<44} {tag:<9} {diffs}")
    ok += res["match"]; bad += (not res["match"])
print("=" * 90)
print(f"reproduce baseline: {ok}/{len(registered)}   (mismatch/error: {bad})")
