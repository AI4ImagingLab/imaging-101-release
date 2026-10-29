from pathlib import Path
import importlib
import numpy as np
from src import preprocessing as pre

ROOT=Path(__file__).resolve().parents[2]

def test_public_constants_and_all_keys_are_self_contained():
    readme=(ROOT/'README.md').read_text(encoding='utf8')
    for value in ['15.8489','5.5008','23.1175','96.916','4508','72','24','96','P21','(0,1/2,0)']:
        assert value in readme
    np.testing.assert_array_equal(pre.CELL,[15.8489,5.5008,23.1175,90.,96.916,90.])
    np.testing.assert_array_equal(pre.ROTATIONS,[np.eye(3,dtype=int),np.diag([-1,1,-1])])
    np.testing.assert_array_equal(pre.TRANSLATIONS,[[0,0,0],[0,.5,0]])
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        for k in z.files:assert '| `'+k+'` |' in readme
        assert (z['intensity']<0).sum()==136
        assert np.sqrt(np.mean(z['intensity']**2))==1
    assert 'github' not in readme.lower()

def test_importing_main_does_not_change_backend():
    import matplotlib
    before=matplotlib.get_backend();importlib.import_module('src.main')
    assert matplotlib.get_backend()==before
