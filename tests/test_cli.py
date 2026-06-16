import subprocess
import sys
import json


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "evaluation_harness", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_dry_run_accepts_mechanism_alias_for_end2end():
    result = run_cli(
        "run",
        "--task",
        "ct_fan_beam",
        "--mechanism",
        "end2end",
        "--model",
        "demo-model",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "mode: end_to_end" in result.stdout
    assert "task: ct_fan_beam" in result.stdout
    assert "model: demo-model" in result.stdout


def test_dry_run_normalizes_planning_alias():
    result = run_cli(
        "run",
        "--task",
        "ct_fan_beam",
        "--mode",
        "planning",
        "--model",
        "demo-model",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "mode: plan" in result.stdout


def test_check_env_json_reports_task_requirements():
    result = run_cli("check-env", "--task", "ct_fan_beam", "--json")

    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["task"] == "ct_fan_beam"
    assert data["requirements_file"].endswith("tasks/ct_fan_beam/requirements.txt")
    assert "numpy" in [pkg["name"] for pkg in data["packages"]]
    assert data["setup_command"].startswith("python -m evaluation_harness setup-env")


def test_setup_env_dry_run_prints_commands_without_installing():
    result = run_cli(
        "setup-env",
        "--task",
        "ct_fan_beam",
        "--venv",
        ".venvs/test-ct-fan-beam",
        "--dry-run",
    )

    assert result.returncode == 0
    assert "Dry run: environment setup commands" in result.stdout
    assert "python -m venv .venvs/test-ct-fan-beam" in result.stdout
    assert " install -r tasks/ct_fan_beam/requirements.txt" in result.stdout
