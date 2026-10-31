from pathlib import Path
import numpy as np
import pytest
from src.visualization import metrics
from src.preprocessing import CRACK_TIP,CRACK_ANGLE_DEG,MIN_RADIUS,MAX_RADIUS,GRID_INCREMENT,ANGLE_GAP_DEG
from src.physics_model import YOUNG_MODULUS,POISSON_RATIO

def test_visualization_metrics_independent_known_numbers():
    result=metrics(np.array([2.,4.]),np.array([1.,2.]))
    assert result['ncc']==pytest.approx(1)
    assert result['nrmse']==pytest.approx(np.sqrt(2.5))

def test_readme_constants_and_no_leak():
    root=Path(__file__).resolve().parents[2]
    readme=(root/'README.md').read_text(encoding='utf-8')
    assert CRACK_TIP.tolist()==[15.39,.4] and CRACK_ANGLE_DEG==-.27
    assert (MIN_RADIUS,MAX_RADIUS,GRID_INCREMENT,ANGLE_GAP_DEG)==pytest.approx((.7695,3.078,.01,20.),abs=1e-14)
    assert (YOUNG_MODULUS,POISSON_RATIO)==(72000.,.33)
    for value in ['15.39','0.40','-0.27','0.7695','3.078','0.01','72000','0.33','44581','231','559']:
        assert value in readme
    for banned in ['github.com','github.io','githubusercontent.com','git.io','baseline_reference.npz','ground_truth.npz']:
        assert banned not in readme
    assert not any(ord(c)<32 and c not in '\n\r\t' for c in readme)
