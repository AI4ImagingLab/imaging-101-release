from pathlib import Path
import numpy as np
from src.preprocessing import load_observations,reconstruction_grid
from src.physics_model import KERNELS
def test_public_metadata_matches_actual_inputs():
    root=Path(__file__).resolve().parents[2];readme=(root/'README.md').read_text(encoding='utf8')
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z.files)=={'redshift','hubble_observation','hubble_error'}
        for key in z.files:
            assert f'| {key} | (1, 30) | float64 |' in readme
            assert z[key].shape==(1,30) and z[key].dtype==np.dtype('float64')
    z,y,e=load_observations(root/'data/raw_data.npz');grid=reconstruction_grid()
    assert len(z)==30 and min(z)==.07 and max(z)==1.965 and np.all(e>0)
    assert len(grid)==200 and grid[0]==0 and grid[-1]==1.965
    assert KERNELS==('SquaredExponential','Matern52','Matern72','Matern92')
    assert '200 equally spaced redshifts from 0 to 1.965 inclusive' in readme
