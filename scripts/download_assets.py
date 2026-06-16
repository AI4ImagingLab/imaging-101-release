"""Download task data, fixtures, and reference outputs from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.asset_manifest import DEFAULT_REPO_ID, file_sha256


Downloader = Callable[..., str]


@dataclass(frozen=True)
class DownloadResult:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


def load_manifest(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def filter_assets(assets: Iterable[dict], task: str | None = None) -> list[dict]:
    selected = [asset for asset in assets if not task or asset["task"] == task]
    return sorted(selected, key=lambda asset: asset["path"])


def existing_file_matches(path: Path, expected_sha256: str) -> bool:
    return path.exists() and path.is_file() and file_sha256(path) == expected_sha256


def download_assets(
    manifest: dict,
    *,
    root: Path | str = ".",
    task: str | None = None,
    repo_id: str | None = None,
    downloader: Downloader | None = None,
    verify: bool = True,
    force: bool = False,
) -> DownloadResult:
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    root = Path(root).resolve()
    repo_id = repo_id or manifest.get("repo_id") or DEFAULT_REPO_ID
    repo_type = manifest.get("repo_type", "dataset")
    downloaded = 0
    skipped = 0
    failed = 0

    for asset in filter_assets(manifest.get("assets", []), task=task):
        local_path = root / asset["path"]
        expected_sha256 = asset["sha256"]
        if existing_file_matches(local_path, expected_sha256):
            skipped += 1
            continue
        if local_path.exists() and not force:
            failed += 1
            print(f"hash mismatch, use --force to replace: {asset['path']}")
            continue
        if local_path.exists() and force:
            local_path.unlink()

        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloader(
            repo_id=repo_id,
            repo_type=repo_type,
            filename=asset.get("hf_path", asset["path"]),
            local_dir=str(root),
        )
        if verify and not existing_file_matches(local_path, expected_sha256):
            failed += 1
            print(f"downloaded file failed sha256 verification: {asset['path']}")
            continue
        downloaded += 1

    return DownloadResult(downloaded=downloaded, skipped=skipped, failed=failed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="assets_manifest.json")
    parser.add_argument("--root", default=".", help="Repository root to populate.")
    parser.add_argument("--repo-id", help="Override Hugging Face dataset repo id.")
    parser.add_argument("--task", help="Download assets for one task.")
    parser.add_argument("--all", action="store_true", help="Download all assets.")
    parser.add_argument("--force", action="store_true", help="Replace existing files with mismatched hashes.")
    parser.add_argument("--no-verify", action="store_true", help="Skip sha256 verification after download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all and not args.task:
        raise SystemExit("Choose --all or --task <task_name>.")

    manifest = load_manifest(args.manifest)
    result = download_assets(
        manifest,
        root=args.root,
        task=None if args.all else args.task,
        repo_id=args.repo_id,
        verify=not args.no_verify,
        force=args.force,
    )
    print(
        f"downloaded={result.downloaded} skipped={result.skipped} failed={result.failed}"
    )
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
