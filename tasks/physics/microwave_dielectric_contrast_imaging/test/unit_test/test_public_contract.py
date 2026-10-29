from pathlib import Path
import numpy as np
from src.physics_model import WAVENUMBER,LENGTH,GRID,BACKGROUND_EPSILON
ROOT=Path(__file__).resolve().parents[2]
def test_public_npz_keys_shapes_dtypes_and_metadata():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:
        assert set(f.files)=={'measurements','illumination_angles','receiver_angles'}
        for key,shape,dtype in [('measurements',(1,9,10),'complex128'),('illumination_angles',(1,10),'float64'),('receiver_angles',(1,9),'float64')]:
            assert f[key].shape==shape and f[key].dtype==np.dtype(dtype)
            assert '| '+key+' | '+str(shape)+' | '+dtype+' |' in text
        np.testing.assert_array_equal(f['illumination_angles'][0],np.arange(0,2*np.pi,2*np.pi/10));np.testing.assert_array_equal(f['receiver_angles'][0],np.arange(0,2*np.pi,2*np.pi/9))
    assert WAVENUMBER==12.575070131717595+0j and LENGTH==.8 and GRID==30 and BACKGROUND_EPSILON==4.
    for word in ['12.575070131717595','300000000','0.4996540966663685','0.8/30','1.0','random-phase','exp(+i omega t)']:assert word in text
    assert 'github' not in text.lower()
