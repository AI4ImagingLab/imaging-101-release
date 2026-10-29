import numpy as np
from src.visualization import relative_error
def test_known_real_and_complex_error_case():
    assert relative_error(np.array([0.,2.]),np.array([0.,1.]))==1
    assert relative_error(np.array([0.,2j]),np.array([0.,1j]))==1
