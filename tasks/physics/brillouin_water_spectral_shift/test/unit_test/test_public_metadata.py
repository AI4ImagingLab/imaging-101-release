from pathlib import Path
import numpy as np
from src.physics_model import setup
from src.solvers import REGIONS
ROOT=Path(__file__).resolve().parents[2]

def test_public_metadata_matches_executed_constants():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    for s in ['780.24','6.5','0.006743','1.45367','0.8','295.15','3.78e9','5.066e9','299792458','500','-2,-1,0,1,2']:assert s in text,s
    instrument=setup();assert instrument.pixel_size==6.5e-6;assert instrument.calibration.num_brillouin_samples==2
    np.testing.assert_array_equal(REGIONS,[[4e9,6e9],[9e9,11e9],[-2e9,2e9],[13e9,17e9]])
    raw=np.load(ROOT/'data/raw_data.npz',allow_pickle=False)
    for key in raw.files:assert '`'+key+'`' in text
    assert raw['images'].shape==(1,10,2,400,400);assert raw['calibration_images'].shape==(1,2,10,400,400)
    assert raw['images'].dtype==np.uint16;assert raw['calibration_images'].dtype==np.uint16

def test_visual_metric_known_case():
    from src.visualization import relative_error
    assert relative_error(np.array([2.,4.]),np.array([1.,2.]))==1
