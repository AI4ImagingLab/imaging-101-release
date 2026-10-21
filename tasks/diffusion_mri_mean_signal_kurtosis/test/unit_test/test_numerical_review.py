from pathlib import Path
import importlib.util,json,numpy as np
from src.visualization import relative_l2
ROOT=Path(__file__).resolve().parents[2]
def test_visualization_metric_known_cases():
    assert relative_l2(np.array([2.,0]),np.array([1.,0]))==1

def test_active_notebook_archive_consistency():
    with np.load(ROOT/'notebooks/reference_outputs/cleaned.npz',allow_pickle=False) as a,np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as b:
        for key in a.files:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-12)
    m=json.loads((ROOT/'notebooks/reference_outputs/measurement.json').read_text())
    assert m['reconstruction_runtime_s']>0 and m['experiment_runtime_s']>0
    with np.load(ROOT/'notebooks/reference_outputs/experiments.npz',allow_pickle=False) as e:
        assert len(e.files)>3
