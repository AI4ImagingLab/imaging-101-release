"""Build and validate the Hugging Face asset manifest for task fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "starpacker52/imaging-101"
ASSET_EXTENSIONS = {
    ".csv",
    ".fits",
    ".h5",
    ".hdf5",
    ".jpeg",
    ".jpg",
    ".json",
    ".mat",
    ".npy",
    ".npz",
    ".png",
    ".pt",
    ".pth",
    ".tif",
    ".tiff",
    ".uvfits",
}


@dataclass(frozen=True)
class AssetRef:
    local_path: Path
    relative_path: str
    task: str


def posix_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def task_name_from_relative(relative_path: str) -> str:
    parts = relative_path.split("/")
    if len(parts) < 2 or parts[0] != "tasks":
        raise ValueError(f"asset path is not under tasks/: {relative_path}")
    return parts[1]


def is_reproducibility_asset(relative_path: str, suffix: str) -> bool:
    if "/evaluation/fixtures/" in relative_path:
        return True
    if "/evaluation/reference_outputs/" in relative_path:
        return True
    return "/data/" in relative_path and suffix.lower() in ASSET_EXTENSIONS


def discover_assets(root: Path | str, task: str | None = None) -> list[AssetRef]:
    root = Path(root).resolve()
    tasks_root = root / "tasks"
    if not tasks_root.exists():
        raise FileNotFoundError(f"tasks directory not found: {tasks_root}")

    assets: list[AssetRef] = []
    for path in tasks_root.rglob("*"):
        if not path.is_file():
            continue
        relative_path = posix_relative(path, root)
        asset_task = task_name_from_relative(relative_path)
        if task and asset_task != task:
            continue
        if is_reproducibility_asset(relative_path, path.suffix):
            assets.append(AssetRef(path, relative_path, asset_task))
    return sorted(assets, key=lambda asset: asset.relative_path)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_asset(path: Path, *, allow_pickle: bool = False) -> None:
    with path.open("rb") as handle:
        if handle.read(48).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Git LFS pointer found; real asset content is missing")

    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        import numpy as np

        loaded = np.load(path, allow_pickle=allow_pickle)
        if hasattr(loaded, "close"):
            loaded.close()
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        return
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            next(csv.reader(handle), None)
        return

    with path.open("rb") as handle:
        handle.read(16)


def validate_assets(
    assets: Iterable[AssetRef],
    *,
    allow_pickle: bool = False,
) -> list[dict[str, str]]:
    errors = []
    for asset in assets:
        try:
            validate_asset(asset.local_path, allow_pickle=allow_pickle)
        except Exception as exc:  # pragma: no cover - exact dependency errors vary
            errors.append({"path": asset.relative_path, "error": str(exc)})
    return errors


def build_manifest(
    root: Path | str,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    task: str | None = None,
) -> dict:
    assets = discover_assets(root, task=task)
    entries = []
    total_bytes = 0
    for asset in assets:
        size = asset.local_path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": asset.relative_path,
                "hf_path": asset.relative_path,
                "task": asset.task,
                "size": size,
                "sha256": file_sha256(asset.local_path),
            }
        )

    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "asset_count": len(entries),
        "total_bytes": total_bytes,
        "assets": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root containing tasks/.")
    parser.add_argument("--output", default="assets_manifest.json")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--task", help="Only include one task.")
    parser.add_argument("--validate", action="store_true", help="Validate readable asset files before writing.")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Allow pickle-backed NumPy files during maintainer validation.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if validation finds unreadable files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    assets = discover_assets(root, task=args.task)
    if args.validate:
        errors = validate_assets(assets, allow_pickle=args.allow_pickle)
        if errors:
            print(json.dumps({"validation_errors": errors}, indent=2), flush=True)
            if args.strict:
                return 1

    manifest = build_manifest(root, repo_id=args.repo_id, task=args.task)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {manifest['asset_count']} assets "
        f"({manifest['total_bytes'] / 1024 / 1024:.2f} MB) to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
