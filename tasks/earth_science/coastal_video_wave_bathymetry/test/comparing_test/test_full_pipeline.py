"""Active full-raw inverse and independent standalone Judge, never cached output."""
import json
import subprocess
import sys
import numpy as np
from src.preprocessing import ROOT, load_data
from src.solvers import reconstruct


def test_complete_bound_observation_pipeline(tmp_path):
    result = reconstruct(load_data(), seed=20261026)
    prediction = tmp_path/'reconstruction.npz'
    report = tmp_path/'judge_report.json'
    np.savez_compressed(prediction, **result)
    command = [sys.executable, str(ROOT/'evaluation/judge.py'),
               '--prediction', str(prediction), '--report', str(report)]
    process = subprocess.run(command, cwd=tmp_path, capture_output=True)
    assert process.returncode == 0, (process.stdout + process.stderr).decode('utf8', errors='replace')
    measured = json.loads(report.read_text(encoding='utf8'))
    assert measured['status'] == 'pass' and measured['passed'] is True
    assert measured['metrics']['interior_coverage'] == 1.0
