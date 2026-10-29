import numpy as np
from src.visualization import roi_residual

def test_forward_residual_known_zero_drive():
    # Zero drive predicts zero for any kernel: RMS(error)/RMS(nonzero data)=1.
    result = {'neural_drive': np.zeros((1,2,152)),
              'hrf': np.ones((1,2,37)),
              'filtered_bold': np.tile(np.arange(1.,153.)[:,None], (1,2))}
    np.testing.assert_array_equal(roi_residual(result), [1.,1.])
