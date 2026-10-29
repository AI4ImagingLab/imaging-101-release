from pathlib import Path
import subprocess
import sys
import numpy as np
from src.preprocessing import load_observations
from src.solvers import reconstruct

ROOT = Path(__file__).resolve().parents[2]


def test_cli_fresh_output_and_calibrated_sideband():
    run = subprocess.run([sys.executable, '-m', 'src.main'], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    data = load_observations(ROOT / 'data/raw_data.npz')
    with np.load(ROOT / 'output/reconstruction.npz', allow_pickle=False) as f:
        np.testing.assert_array_equal(f['reconstruction'], reconstruct(data['hologram'], data['vacuum']))
