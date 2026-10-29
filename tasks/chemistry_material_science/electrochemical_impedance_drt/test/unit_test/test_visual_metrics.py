import numpy as np
from src.visualization import relative_error
def test_visual_metric_known_case():
    assert relative_error(np.array([2.,4.]),np.array([1.,2.]))==1.
