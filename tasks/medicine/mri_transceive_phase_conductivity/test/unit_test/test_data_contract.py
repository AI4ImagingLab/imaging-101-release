from pathlib import Path
import numpy as np
from src.preprocessing import load_observations
from src.physics_model import FREQUENCY_HZ, SPACING_XYZ, MU0

ROOT=Path(__file__).resolve().parents[2]


def test_public_geometry_constants_and_missing_observations():
    phase,magnitude,mask=load_observations(ROOT/'data/raw_data.npz')
    assert mask.sum()==11513 and phase.shape==(25,96,96,80)
    assert np.isfinite(phase[:,mask]).all() and np.isnan(phase[:,~mask]).all()
    assert np.isfinite(magnitude[:,mask]).all() and np.isnan(magnitude[:,~mask]).all()
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert '| Frequency f | 128000000 | Hz |' in text and FREQUENCY_HZ==128000000
    assert '| Voxel spacing x,y,z | 0.002,0.002,0.002 | m |' in text and SPACING_XYZ==(.002,.002,.002)
    assert '4*pi*10^-7' in text and MU0==4*np.pi*1e-7


def test_import_main_has_no_output_or_plot_backend_side_effect():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
