from pathlib import Path
import subprocess
import sys

import pytest

from scripts.asset_manifest import build_manifest, discover_assets, file_sha256, validate_asset
from scripts.download_assets import download_assets


def write_file(path: Path, data: bytes = b"data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_discover_assets_includes_reproducibility_files(tmp_path):
    write_file(tmp_path / "tasks/demo/data/raw_data.npz")
    write_file(tmp_path / "tasks/demo/data/raw_data.txt")
    write_file(tmp_path / "tasks/demo/evaluation/fixtures/input.npy")
    write_file(tmp_path / "tasks/demo/evaluation/reference_outputs/expected.npy")
    write_file(tmp_path / "tasks/demo/output/reconstruction.npy")
    write_file(tmp_path / "tasks/demo/src/solver.py")

    assets = [asset.relative_path for asset in discover_assets(tmp_path)]

    assert assets == [
        "tasks/demo/data/raw_data.npz",
        "tasks/demo/evaluation/fixtures/input.npy",
        "tasks/demo/evaluation/reference_outputs/expected.npy",
    ]


def test_build_manifest_records_task_size_hash_and_hf_path(tmp_path):
    write_file(tmp_path / "tasks/demo/evaluation/fixtures/input.npy", b"abc")

    manifest = build_manifest(tmp_path, repo_id="owner/dataset")

    assert manifest["schema_version"] == 1
    assert manifest["repo_id"] == "owner/dataset"
    assert manifest["asset_count"] == 1
    assert manifest["total_bytes"] == 3
    assert manifest["assets"][0] == {
        "path": "tasks/demo/evaluation/fixtures/input.npy",
        "hf_path": "tasks/demo/evaluation/fixtures/input.npy",
        "task": "demo",
        "size": 3,
        "sha256": file_sha256(tmp_path / "tasks/demo/evaluation/fixtures/input.npy"),
    }


def test_build_manifest_filters_to_one_task(tmp_path):
    write_file(tmp_path / "tasks/demo/evaluation/fixtures/input.npy")
    write_file(tmp_path / "tasks/other/evaluation/fixtures/input.npy")

    manifest = build_manifest(tmp_path, task="other")

    assert [asset["task"] for asset in manifest["assets"]] == ["other"]


def test_download_assets_skips_verified_files_and_downloads_missing(tmp_path):
    existing = tmp_path / "tasks/demo/evaluation/fixtures/existing.npy"
    missing = tmp_path / "tasks/demo/evaluation/fixtures/missing.npy"
    write_file(existing, b"already here")
    existing_hash = file_sha256(existing)

    manifest = {
        "repo_id": "owner/dataset",
        "assets": [
            {
                "path": "tasks/demo/evaluation/fixtures/existing.npy",
                "hf_path": "tasks/demo/evaluation/fixtures/existing.npy",
                "task": "demo",
                "size": existing.stat().st_size,
                "sha256": existing_hash,
            },
            {
                "path": "tasks/demo/evaluation/fixtures/missing.npy",
                "hf_path": "tasks/demo/evaluation/fixtures/missing.npy",
                "task": "demo",
                "size": 10,
                "sha256": file_sha256(existing),
            },
        ],
    }
    calls = []

    def fake_downloader(repo_id, repo_type, filename, local_dir):
        calls.append((repo_id, repo_type, filename, local_dir))
        write_file(Path(local_dir) / filename, b"downloaded")

    result = download_assets(
        manifest,
        root=tmp_path,
        task="demo",
        downloader=fake_downloader,
        verify=False,
    )

    assert result.downloaded == 1
    assert result.skipped == 1
    assert calls == [
        (
            "owner/dataset",
            "dataset",
            "tasks/demo/evaluation/fixtures/missing.npy",
            str(tmp_path),
        )
    ]
    assert missing.read_bytes() == b"downloaded"


def test_validate_asset_can_explicitly_allow_numpy_pickle(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "object_array.npy"
    np.save(path, np.array([{"ok": True}], dtype=object))

    with pytest.raises(ValueError):
        validate_asset(path)

    validate_asset(path, allow_pickle=True)


def test_validate_asset_rejects_git_lfs_pointer(tmp_path):
    path = tmp_path / "sample.tiff"
    write_file(
        path,
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:abc\n"
        b"size 123\n",
    )

    with pytest.raises(ValueError, match="Git LFS pointer"):
        validate_asset(path)


def test_download_assets_script_can_run_directly():
    result = subprocess.run(
        [sys.executable, "scripts/download_assets.py", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert result.returncode == 0
    assert "--task" in result.stdout
