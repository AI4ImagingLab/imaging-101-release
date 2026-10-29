from pathlib import Path
import numpy as np
from src.preprocessing import KEYS,load_observations,prepare_forcing
def test_all_native_observations_and_nominal_metadata():
    root=Path(__file__).resolve().parents[2];readme=(root/'README.md').read_text(encoding='utf8')
    with np.load(root/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z.files)==set(KEYS)|{'pixel_index'}
        for key in KEYS:
            assert z[key].shape==(1,110935) and z[key].dtype==np.dtype('float32')
            assert f'| {key} | (1, 110935) | float32 |' in readme
        assert z['pixel_index'].dtype==np.dtype('int64') and np.all(np.diff(z['pixel_index'][0])>0)
        assert z['pixel_index'].min()>=0 and z['pixel_index'].max()<336*335
    inputs=load_observations(root/'data/raw_data.npz');f=prepare_forcing(inputs)
    assert f['canopy_mask'].sum()==110929 and np.all(f['height']==.3)
    np.testing.assert_allclose(f['z0'][f['canopy_mask']],.0375)
    np.testing.assert_allclose(f['d0'][f['canopy_mask']],.195)
    for text in ['111;10.98','30.25;38.25','| 45 | degree east','0.3;0.1','0.98;0.95','5.670373e-8','1013.25']:assert text in readme
