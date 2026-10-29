"""Boundary regressions: hidden data stays private, native Judge stays authoritative."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation_harness.benchmark import prepare_public_workspace, score_submission
from evaluation_harness.task_paths import resolve_task


@pytest.fixture
def task(tmp_path):
    root = tmp_path / "repo"
    path = root / "tasks" / "physics" / "control"
    for folder in ("data", "evaluation", "src", "output"):
        (path / folder).mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("Save output/reconstruction.npz with keys x and phase.")
    (path / "requirements.txt").write_text("numpy\n")
    (root / "environment.yml").write_text("name: test\n")
    np.savez(path / "data/raw_data.npz", observation=np.array([[1., 2.]]))
    np.savez(path / "data/leaked_gt.npz", truth=np.array([[123.]]))
    np.savez(path / "evaluation/ground_truth.npz", truth=np.array([[123.]]))
    (path / "src/solvers.py").write_text("raise RuntimeError('private solution')")
    (path / "evaluation/judge.py").write_text("# trusted standalone evaluator\n")
    return path


def test_public_export_excludes_sibling_data_answers_and_all_private_assets(task, tmp_path):
    workspace = tmp_path / "export"
    manifest = prepare_public_workspace(task, workspace)
    assert set(manifest["files"]) == {"README.md", "data/raw_data.npz", "requirements.txt", "environment.yml"}
    assert {p.relative_to(workspace).as_posix() for p in workspace.rglob('*') if p.is_file()} == set(manifest["files"])
    assert (workspace / "output").is_dir()


def test_stale_workspace_cannot_retain_encoded_answers(task, tmp_path):
    workspace = tmp_path / "export"
    workspace.mkdir()
    (workspace / "self_eval.py").write_text("secret = 1")
    with pytest.raises(ValueError, match="new or empty"):
        prepare_public_workspace(task, workspace)


def test_quarantined_task_is_not_silently_exported_as_a_valid_benchmark(task, tmp_path):
    docs = task.parents[2] / 'docs'
    docs.mkdir()
    (docs / 'task_registry.json').write_text(json.dumps({'tasks': [
        {'task': 'control', 'benchmark_status': 'blocked', 'blocking_reasons': ['Independent calibration absent']}
    ]}))
    with pytest.raises(ValueError, match='Independent calibration absent'):
        prepare_public_workspace(task, tmp_path / 'should-not-exist')
    assert not (tmp_path / 'should-not-exist').exists()


@pytest.mark.parametrize("name", ["../control", "physics/../control", "/physics/control", "physics/control/extra"])
def test_resolver_forbids_traversal(task, name):
    with pytest.raises(ValueError):
        resolve_task(task.parents[2], name)


def test_resolver_bare_and_qualified_names(task):
    assert resolve_task(task.parents[2], "control") == task
    assert resolve_task(task.parents[2], "physics/control") == task


def install_control_judge(task):
    (task / "evaluation/judge.py").write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prediction');p.add_argument('--report');a=p.parse_args()
good=Path(a.prediction).read_bytes()==b'valid' if Path(a.prediction).exists() else False
d=dict(status='pass' if good else 'error',passed=good,metrics={'two_quantities':7} if good else {},thresholds={},errors=[])
Path(a.report).write_text(json.dumps(d));raise SystemExit(0 if good else 2)
''')


def test_trusted_judge_runs_outside_submission_and_never_imports_it(task, tmp_path):
    install_control_judge(task)
    untrusted = tmp_path / "untrusted"
    untrusted.mkdir()
    (untrusted / "sitecustomize.py").write_text("raise RuntimeError('submitted Python must never run')")
    prediction = untrusted / "reconstruction.npz"
    prediction.write_bytes(b'valid')
    result = score_submission(task, prediction)
    assert result['status'] == 'pass'
    assert result['metrics'] == {'two_quantities': 7}


def test_missing_prediction_returns_machine_readable_error(task, tmp_path):
    install_control_judge(task)
    result = score_submission(task, tmp_path / "missing.npz", tmp_path / "report.json")
    assert result['status'] == 'error' and result['judge_exit_code'] == 2


def test_native_npz_scoring_dispatch_does_not_guess_reference_keys(task, tmp_path):
    from evaluation_harness.reference_scoring import score_reconstruction
    install_control_judge(task)
    workspace = tmp_path / "export"
    (workspace / "output").mkdir(parents=True)
    (workspace / "output/reconstruction.npz").write_bytes(b'valid')
    result = score_reconstruction(task, workspace)
    assert result.get('status') == 'pass' and result['metrics'] == {'two_quantities': 7}


def test_docker_never_mounts_private_task_tree(task, monkeypatch):
    from evaluation_harness.docker_runner import DockerRunner
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    runner = DockerRunner('image', task)
    runner.start(visible_paths=['README.md', 'data/raw_data.npz', 'requirements.txt'])
    docker_run = next(c for c in calls if c[:2] == ['docker', 'run'])
    assert not any('/workspace_src' in str(c) for c in docker_run)
    assert '--network=none' in docker_run


def test_copilot_npz_export_has_no_obfuscated_gt(task, tmp_path):
    from evaluation_harness.frameworks.claude_code.copilot_runner import CopilotSandbox
    workspace = tmp_path / 'copilot'
    CopilotSandbox(task, workspace).prepare()
    assert not (workspace / 'self_eval.py').exists()
    assert not (workspace / 'data/leaked_gt.npz').exists()


def test_copilot_collection_uses_native_judge_without_llm_dependencies(task, tmp_path):
    from evaluation_harness.frameworks.claude_code.copilot_scorer import _compute_quality_metrics
    install_control_judge(task)
    workspace = tmp_path / 'finished'
    (workspace / 'output').mkdir(parents=True)
    (workspace / 'output/reconstruction.npz').write_bytes(b'valid')
    result = _compute_quality_metrics(workspace, task)
    assert result['status'] == 'pass' and result['metrics']['two_quantities'] == 7


def test_docker_copies_only_allowlisted_observation_not_data_siblings(task, monkeypatch):
    from evaluation_harness.docker_runner import DockerRunner
    exported = []
    def run(cmd, **kwargs):
        if cmd[:2] == ['docker', 'cp']:
            source = Path(cmd[2])
            exported.extend(p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file())
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    DockerRunner('image', task).start(['README.md', 'data', 'requirements.txt'])
    assert set(exported) == {'README.md', 'data/raw_data.npz', 'requirements.txt'}


def test_local_export_never_seeds_private_assets_or_installs_packages(task, tmp_path, monkeypatch):
    from evaluation_harness.local_runner import LocalRunner
    workspace = tmp_path / 'local-export'
    workspace.mkdir()
    monkeypatch.setattr('evaluation_harness.local_runner.tempfile.mkdtemp', lambda **kw: str(workspace))
    runner = LocalRunner('unused', task)
    monkeypatch.setattr(runner, 'exec', lambda cmd: pytest.fail('Prepared benchmark must not install packages'))
    runner.start()
    assert not (workspace / 'src').exists()
    assert not (workspace / 'evaluation').exists()
    assert not (workspace / 'data/leaked_gt.npz').exists()


def test_judge_cli_unknown_task_returns_error_report_not_traceback(task, tmp_path):
    report = tmp_path / 'unknown.json'
    command = [sys.executable, '-m', 'evaluation_harness.benchmark', '--repo',
               str(task.parents[2]), 'judge', '--task', 'unknown', '--workspace',
               str(tmp_path / 'workspace'), '--report', str(report)]
    run = subprocess.run(command, capture_output=True, encoding='utf-8')
    assert run.returncode == 2
    assert json.loads(report.read_text(encoding='utf-8'))['status'] == 'error'
    assert 'Traceback' not in run.stderr


def test_judge_cli_report_write_failure_returns_exit2(task, tmp_path):
    install_control_judge(task)
    workspace = tmp_path / 'workspace'
    (workspace / 'output').mkdir(parents=True)
    (workspace / 'output/reconstruction.npz').write_bytes(b'valid')
    report = tmp_path / 'report-directory'
    report.mkdir()
    command = [sys.executable, '-m', 'evaluation_harness.benchmark', '--repo',
               str(task.parents[2]), 'judge', '--task', 'physics/control', '--workspace',
               str(workspace), '--report', str(report)]
    run = subprocess.run(command, capture_output=True, encoding='utf-8')
    assert run.returncode == 2
    assert 'Report-write failure' in run.stderr
