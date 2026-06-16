#!/usr/bin/env python3
"""Run the function-mode evaluation matrix for one model.

Enumerates (task, module) pairs where src/<module>.py and
evaluation/tests/test_<module>.py both exist, restricted to a module
allow-list, runs each as `evaluation_harness run --mechanism function`,
and is idempotent (skips pairs that already have a result.json for this model).
"""
from __future__ import annotations

import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Cap BLAS/OpenMP threads to 1 per child process. On a 144-core host, numpy's
# OpenBLAS otherwise spawns ~64 threads per process; with several concurrent
# runs (each spawning venv/pip/pytest/agent subprocesses) this exhausts the
# thread/process limit and runs die with "pthread_create failed". Function-mode
# work is agent + light pytest, not BLAS-bound, so 1 thread is fine.
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
CHILD_ENV = {**os.environ, **THREAD_ENV}

REPO = Path(__file__).resolve().parents[1]
MODEL = "Vendor3/DeepSeek-V4-Pro"
BASE_URL = "https://api.gpugeek.com/v1"
API_KEY = "YOUR_API_KEY"
OUTPUT = "results"
ALLOW = {"solvers", "physics_model", "preprocessing"}
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))

SAFE_MODEL = MODEL.replace("/", "_").replace("\\", "_")
LOGDIR = REPO / "results" / "function_mode" / "_matrix_logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
SUMMARY = LOGDIR / "_summary.tsv"


def enumerate_pairs():
    pairs = []
    tasks_dir = REPO / "tasks"
    for task in sorted(p.name for p in tasks_dir.iterdir() if p.is_dir()):
        src_dir = tasks_dir / task / "src"
        test_dir = tasks_dir / task / "evaluation" / "tests"
        if not src_dir.is_dir() or not test_dir.is_dir():
            continue
        src = {p.stem for p in src_dir.glob("*.py")} - {"__init__"}
        tests = {p.stem[5:] for p in test_dir.glob("test_*.py")}
        for m in sorted(src & tests & ALLOW):
            pairs.append((task, m))
    return pairs


def already_done(task, module):
    return bool(list((REPO / "results" / "function_mode").glob(
        f"{task}/{SAFE_MODEL}_*/{module}/result.json")))


def run_pair(task, module):
    log = LOGDIR / f"{task}__{module}.log"
    cmd = [
        sys.executable, "-m", "evaluation_harness", "run",
        "--task", task, "--mechanism", "function",
        "--target-function", module, "--model", MODEL,
        "--base-url", BASE_URL, "--api-key", API_KEY,
        "--output", OUTPUT, "-v",
    ]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=str(REPO), stdout=fh,
                            stderr=subprocess.STDOUT, env=CHILD_ENV).returncode
    # read back the result
    passed = total = -1
    stopped = "?"
    rjs = sorted((REPO / "results" / "function_mode").glob(
        f"{task}/{SAFE_MODEL}_*/{module}/result.json"))
    if rjs:
        import json
        try:
            d = json.loads(rjs[-1].read_text())
            passed, total, stopped = d.get("tests_passed", -1), d.get("tests_total", -1), d.get("stopped_reason", "?")
        except Exception:
            pass
    status = "OK" if rc == 0 and total > 0 and passed == total else (
        "FAIL_RC" if rc != 0 else f"PART({passed}/{total})" if total > 0 else "NO_TESTS")
    line = f"{status}\t{task}\t{module}\t{passed}/{total}\tstopped={stopped}\trc={rc}"
    with open(SUMMARY, "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)
    return status


def main():
    pairs = [(t, m) for (t, m) in enumerate_pairs() if not already_done(t, m)]
    skipped = [(t, m) for (t, m) in enumerate_pairs() if already_done(t, m)]
    SUMMARY.write_text("")
    print(f"Function matrix: model={MODEL}  allow={sorted(ALLOW)}", flush=True)
    print(f"to_run={len(pairs)}  skipped(existing)={len(skipped)}  concurrency={MAX_CONCURRENCY}", flush=True)
    t0 = time.time()
    counts = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        futs = {ex.submit(run_pair, t, m): (t, m) for (t, m) in pairs}
        done = 0
        for fut in as_completed(futs):
            status = fut.result()
            counts[status.split("(")[0]] = counts.get(status.split("(")[0], 0) + 1
            done += 1
            if done % 10 == 0:
                print(f"  ... {done}/{len(pairs)} done, {time.time()-t0:.0f}s elapsed", flush=True)
    print(f"\nMatrix complete in {time.time()-t0:.0f}s. status counts: {counts}", flush=True)

    # Per-task summaries
    print("\nGenerating per-task summaries ...", flush=True)
    tasks = sorted({t for t, _ in enumerate_pairs()})
    for task in tasks:
        for run_dir in (REPO / "results" / "function_mode" / task).glob(f"{SAFE_MODEL}_*"):
            subprocess.run([sys.executable, "-m", "evaluation_harness", "summarize",
                            "--dir", str(run_dir)], cwd=str(REPO),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
