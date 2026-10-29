"""Discover task directories without relying on the historical flat layout."""
from pathlib import Path

DOMAIN_FOLDERS = (
    "mathematics", "physics", "chemistry_material_science", "astronomy",
    "earth_science", "mechanics", "biology", "medicine",
)


def repository_root(task_dir: Path) -> Path:
    for parent in Path(task_dir).resolve().parents:
        if (parent / "AGENTS.md").is_file() and (parent / "evaluation_harness").is_dir():
            return parent
    raise ValueError("Cannot locate benchmark repository")


def iter_task_dirs(repo_root: Path):
    base = Path(repo_root).resolve() / "tasks"
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name in DOMAIN_FOLDERS:
            yield from (p for p in sorted(child.iterdir()) if p.is_dir())
        else:
            yield child


def resolve_task(repo_root: Path, name: str) -> Path:
    base = (Path(repo_root).resolve() / "tasks").resolve()
    parts = name.replace("\\", "/").split("/")
    if len(parts) not in (1, 2) or any(p in ("", ".", "..") for p in parts):
        raise ValueError("Use a task name or domain/task; traversal is forbidden")
    if len(parts) == 2:
        if parts[0] not in DOMAIN_FOLDERS:
            raise ValueError("Unknown task domain")
        matches = [base.joinpath(*parts)]
    else:
        matches = [p for p in iter_task_dirs(repo_root) if p.name == name]
    matches = [p.resolve() for p in matches if p.is_dir() and p.resolve().is_relative_to(base)]
    if len(matches) != 1:
        raise ValueError(f"Task must resolve uniquely: {name!r} ({len(matches)} matches)")
    return matches[0]
