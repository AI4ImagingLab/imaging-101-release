"""Environment preflight and setup helpers for benchmark tasks."""

from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


SPECIAL_ENV_FILES = ("Dockerfile", "environment.yml", "environment.yaml", "ENVIRONMENT.md")

SPECIAL_PACKAGES = {
    "astra-toolbox",
    "cupy",
    "ehtim",
    "finufft",
    "jax",
    "matlabengine",
    "odl",
    "sigpy",
    "svmbir",
    "tike",
    "torch",
    "torchkbnufft",
}

COMMON_PACKAGES = {
    "ipykernel",
    "jupyter",
    "matplotlib",
    "nbconvert",
    "numpy",
    "pillow",
    "pytest",
    "requests",
    "scikit-image",
    "scipy",
    "tqdm",
}


def normalize_package_name(name: str) -> str:
    """Normalize a package name using the PEP 503 comparison form."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(requirements_path: Path) -> list[dict[str, str]]:
    """Parse install requirement names from a requirements.txt file.

    The parser is intentionally conservative: it extracts package names from
    ordinary requirement specifiers and marks direct URLs/options as unchecked.
    """
    if not requirements_path.exists():
        return []

    packages: list[dict[str, str]] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--", "-f", "-i", "--extra-index-url")):
            packages.append({"name": line, "raw": raw_line, "kind": "option"})
            continue
        if line.startswith(("git+", "http://", "https://")):
            packages.append({"name": line, "raw": raw_line, "kind": "direct"})
            continue

        line_without_marker = line.split(";", 1)[0].strip()
        match = re.match(r"([A-Za-z0-9_.-]+)", line_without_marker)
        if match:
            packages.append(
                {
                    "name": normalize_package_name(match.group(1)),
                    "raw": raw_line,
                    "kind": "package",
                }
            )
        else:
            packages.append({"name": line, "raw": raw_line, "kind": "unknown"})
    return packages


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _classify_tier(packages: list[dict[str, str]], env_files: list[str]) -> str:
    package_names = {pkg["name"] for pkg in packages if pkg["kind"] == "package"}
    if "ENVIRONMENT.md" in env_files or "Dockerfile" in env_files:
        return "Tier 3 - custom environment notes"
    if package_names & SPECIAL_PACKAGES:
        return "Tier 2 - specialized Python packages"
    if package_names - COMMON_PACKAGES:
        return "Tier 2 - additional Python packages"
    return "Tier 1 - standard pip environment"


def check_task_environment(
    repo_root: Path,
    task_name: str,
    python_executable: str = sys.executable,
) -> dict:
    """Return a structured environment report for one task."""
    task_dir = (repo_root / "tasks" / task_name).resolve()
    requirements_path = task_dir / "requirements.txt"

    if not task_dir.exists():
        return {
            "task": task_name,
            "status": "error",
            "error": f"Task directory not found: {task_dir}",
        }

    packages = parse_requirements(requirements_path)
    package_reports = []
    missing = []
    unchecked = []

    for pkg in packages:
        report = dict(pkg)
        if pkg["kind"] == "package":
            version = _distribution_version(pkg["name"])
            report["installed"] = version is not None
            report["installed_version"] = version or ""
            if version is None:
                missing.append(pkg["name"])
        else:
            report["installed"] = None
            report["installed_version"] = ""
            unchecked.append(pkg["name"])
        package_reports.append(report)

    env_files = [name for name in SPECIAL_ENV_FILES if (task_dir / name).exists()]
    tier = _classify_tier(packages, env_files)
    setup_command = f"python -m evaluation_harness setup-env --task {task_name}"

    recommendations = [setup_command]
    if env_files:
        recommendations.append(
            "Review task-specific environment files: " + ", ".join(env_files)
        )
    if missing:
        recommendations.append(
            "Missing packages can usually be installed from the task requirements file."
        )
    if unchecked:
        recommendations.append(
            "Some requirement lines are direct URLs or pip options and cannot be checked locally."
        )

    return {
        "task": task_name,
        "status": "ok",
        "task_dir": task_dir.as_posix(),
        "python": python_executable,
        "requirements_file": requirements_path.as_posix() if requirements_path.exists() else "",
        "requirements_present": requirements_path.exists(),
        "tier": tier,
        "environment_files": env_files,
        "packages": package_reports,
        "missing_packages": sorted(set(missing)),
        "unchecked_requirements": unchecked,
        "setup_command": setup_command,
        "recommendations": recommendations,
    }


def environment_commands(
    task_dir: Path,
    venv_dir: Path,
    python_executable: str,
    *,
    display_relative_to: Path | None = None,
) -> list[str]:
    """Return shell-like setup commands for display and dry-runs."""
    task_display_dir = _display_path(task_dir, display_relative_to)
    venv_display_dir = _display_path(venv_dir, display_relative_to)
    requirements_path = task_dir / "requirements.txt"
    commands = [f"{python_executable} -m venv {venv_display_dir}"]
    pip_cmd = _display_path(_venv_executable(venv_dir, "pip"), display_relative_to)
    commands.append(f"{pip_cmd} install --upgrade pip")
    if requirements_path.exists():
        commands.append(f"{pip_cmd} install -r {task_display_dir}/requirements.txt")
    commands.append(f"{pip_cmd} install pytest")
    return commands


def setup_task_environment(
    task_dir: Path,
    venv_dir: Path,
    python_executable: str = sys.executable,
    *,
    dry_run: bool = False,
    force: bool = False,
    timeout: int = 1800,
) -> dict:
    """Create a venv and install task requirements."""
    repo_root = task_dir.parent.parent.resolve()
    task_dir = task_dir.resolve()
    actual_venv_dir = venv_dir if venv_dir.is_absolute() else repo_root / venv_dir
    commands = environment_commands(
        task_dir,
        venv_dir,
        python_executable,
        display_relative_to=repo_root,
    )

    if dry_run:
        return {"status": "dry-run", "venv": venv_dir.as_posix(), "commands": commands}

    if actual_venv_dir.exists():
        if not force:
            return {
                "status": "error",
                "error": f"Virtual environment already exists: {actual_venv_dir}. Use --force to recreate it.",
                "commands": commands,
            }
        shutil.rmtree(actual_venv_dir)

    completed = []
    for command in commands:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        completed.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout[-4000:],
                "stderr": result.stderr[-4000:],
            }
        )
        if result.returncode != 0:
            return {
                "status": "error",
                "error": f"Command failed: {command}",
                "venv": actual_venv_dir.as_posix(),
                "completed": completed,
            }

    return {"status": "ok", "venv": actual_venv_dir.as_posix(), "completed": completed}


def _venv_executable(venv_dir: Path, name: str) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _display_path(path: Path, base: Path | None) -> str:
    if base is not None:
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            pass
    return path.as_posix()
