"""Public-only NPZ benchmark preparation and trusted, out-of-workspace judging.

This interface deliberately does not import a submitted solver, guess an array
key, or place hidden references/an encoded self-evaluation answer in a workspace.
Host filesystem workspaces are exports, not an OS security boundary. Run agents
in an isolated container with only that export mounted and networking disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .task_paths import resolve_task

PUBLIC_FILES = ("README.md", "data/raw_data.npz", "requirements.txt")


def uses_npz_contract(task_dir: Path) -> bool:
    readme = Path(task_dir) / "README.md"
    return ((Path(task_dir) / "evaluation/judge.py").is_file()
            and readme.is_file()
            and "output/reconstruction.npz" in readme.read_text(encoding="utf-8-sig"))


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_files(task_dir: Path) -> list[Path]:
    task_dir = task_dir.resolve()
    paths = [task_dir / rel for rel in PUBLIC_FILES]
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(task_dir):
            raise ValueError(f"Missing/unsafe public asset: {path.name}")
    return paths


def ensure_benchmark_ready(task_dir: Path) -> None:
    """Do not silently export a task explicitly quarantined by the audit."""
    for parent in Path(task_dir).resolve().parents:
        registry = parent / "docs" / "task_registry.json"
        if not registry.is_file():
            continue
        data = json.loads(registry.read_text(encoding="utf-8-sig"))
        record = next((x for x in data["tasks"] if x["task"] == Path(task_dir).name), None)
        # Legacy tasks were explicitly excluded from this repair scope. Keep
        # their existing interfaces available without certifying them anew.
        if record is not None and record.get("benchmark_status") in ("blocked", "unverified"):
            raise ValueError(f"Task is not certified for this public benchmark: {record['task']}: "
                             + "; ".join(record.get("blocking_reasons", [record['benchmark_status']])))
        return


def prepare_public_workspace(task_dir: Path, workspace: Path) -> dict:
    task_dir, workspace = Path(task_dir).resolve(), Path(workspace).resolve()
    ensure_benchmark_ready(task_dir)
    if workspace.is_relative_to(task_dir) or task_dir.is_relative_to(workspace):
        raise ValueError("Export workspace and private task must be disjoint")
    files = public_files(task_dir)
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("Workspace must be new or empty; stale hidden files are forbidden")
    workspace.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for source in files:
        rel = source.relative_to(task_dir)
        dest = workspace / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        inventory[rel.as_posix()] = _sha256(dest)
    root_environment = next((p / "environment.yml" for p in task_dir.parents if (p / "environment.yml").is_file()), None)
    if root_environment is not None:
        dest = workspace / "environment.yml"
        shutil.copyfile(root_environment, dest)
        inventory[dest.name] = _sha256(dest)
    (workspace / "output").mkdir(exist_ok=True)
    # Inventory is returned to the maintainer, never written into the public export.
    return {"task": task_dir.name, "workspace": str(workspace), "files": inventory,
            "output": "output/reconstruction.npz", "hidden_assets_exported": False}


def score_submission(task_dir: Path, prediction: Path, report: Path | None = None,
                     *, timeout: float = 300, python: str | None = None) -> dict:
    """Execute only the trusted task Judge, against one inert copied submission."""
    task_dir, prediction = Path(task_dir).resolve(), Path(prediction).absolute()
    judge = task_dir / "evaluation" / "judge.py"
    if not judge.is_file() or prediction.is_symlink():
        result = {"status": "error", "passed": False, "metrics": {}, "thresholds": {},
                  "errors": ["Missing standalone NPZ Judge" if not judge.is_file()
                             else "Prediction must not be a symbolic link"], "judge_exit_code": 2}
        if report is not None:
            if Path(report).absolute() == prediction:
                raise ValueError("Report must not overwrite a submission")
            _json_write(Path(report), result)
        return result
    with tempfile.TemporaryDirectory(prefix="imaging101-judge-") as temp:
        private = Path(temp)
        submitted = private / "reconstruction.npz"
        if prediction.is_file():
            shutil.copyfile(prediction, submitted)
        destination = private / "judge_report.json"
        cmd = [python or sys.executable, "-I", str(judge), "--prediction", str(submitted), "--report", str(destination)]
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME"):
            env.pop(key, None)
        env.update(PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            run = subprocess.run(cmd, cwd=task_dir, env=env, capture_output=True,
                                 encoding="utf-8", errors="replace", timeout=timeout)
            if not destination.is_file():
                raise ValueError(f"Judge produced no JSON report (exit {run.returncode}): {run.stderr[-1000:]}")
            def reject_constant(value):
                raise ValueError(f"Non-finite JSON constant {value}")
            result = json.loads(destination.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
            required = {"status", "passed", "metrics", "thresholds", "errors"}
            if not required <= result.keys():
                raise ValueError("Judge report lacks required contract fields")
            expected = {"pass": 0, "fail": 1, "error": 2}.get(result["status"])
            if expected is None or run.returncode != expected or result["passed"] is not (expected == 0):
                raise ValueError("Judge exit/status/passed fields disagree")
            result["judge_exit_code"] = run.returncode
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            result = {"status": "error", "passed": False, "metrics": {}, "thresholds": {},
                      "errors": [str(exc)], "judge_exit_code": 2}
    if report is not None:
        report = Path(report).absolute()
        if report == prediction:
            raise ValueError("Report must not overwrite a submission")
        _json_write(report, result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Export README, raw observations and environment only")
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    score = commands.add_parser("judge", help="Score a completed export with the hidden trusted Judge")
    score.add_argument("--task", required=True)
    score.add_argument("--workspace", type=Path, required=True)
    score.add_argument("--report", type=Path, required=True)
    score.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args(argv)
    try:
        task = resolve_task(args.repo, args.task)
        if args.command == "prepare":
            result = prepare_public_workspace(task, args.workspace)
            code = 0
        else:
            result = score_submission(task, args.workspace / "output" / "reconstruction.npz", timeout=args.timeout)
            code = result["judge_exit_code"]
    except (ValueError, OSError) as exc:
        result = {"status": "error", "passed": False, "metrics": {}, "thresholds": {},
                  "errors": [str(exc)], "judge_exit_code": 2}
        code = 2
    if args.command == "judge":
        try:
            if args.report.absolute() == (args.workspace / "output/reconstruction.npz").absolute():
                raise ValueError("Report must not overwrite a submission")
            _json_write(args.report, result)
        except (ValueError, OSError) as exc:
            print(f"Report-write failure: {exc}", file=sys.stderr)
            result = {"status": "error", "passed": False, "metrics": {}, "thresholds": {},
                      "errors": [str(exc)], "judge_exit_code": 2}
            code = 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
