import numpy as np
from src.visualization import error
def test_known_real_and_complex_error_case():
    assert error(np.array([0.,2.]),np.array([0.,1.]))==1
    assert error(np.array([0.,2j]),np.array([0.,1j]))==1
