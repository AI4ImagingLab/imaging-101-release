import numpy as np
from src.visualization import relative_l2,preview,alternate_dc_weights
def test_known_metric_and_bayer_preview_units():
    assert relative_l2(np.array([0,2]),np.array([0,1]))==1
    np.testing.assert_array_equal(preview(np.full((4,4),164,dtype='uint16')),100)
def test_equal_gain_identical_tiles_have_unit_alternate_dc_weight():
    tiles=np.full((2,2,2,32,32),164,dtype='uint16')
    np.testing.assert_array_equal(alternate_dc_weights(tiles,1),1)
