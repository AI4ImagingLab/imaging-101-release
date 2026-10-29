from pathlib import Path
import numpy as np
import pytest
from src.physics_model import cfa_masks,sample_rgb,sampling_adjoint,WHITE_LEVEL,SHAPE
from src.preprocessing import normalize_mosaic
from src.solvers import reconstruct,interpolation_kernels
T=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('pattern',['RGGB','BGGR','GRBG','GBRG'])
def test_upstream_small_cases_and_observation_identity(pattern):
    with np.load(T/'test/fixtures/upstream_cases.npz',allow_pickle=False) as z:
        a=z['input_small_case'];result=reconstruct(a,pattern)
        np.testing.assert_allclose(result,z['output_small_'+pattern],rtol=1e-13,atol=1e-14)
        np.testing.assert_array_equal(sample_rgb(result,pattern),a)

def test_adjoint_non_square_and_all_masks():
    r=np.random.default_rng(18);x=r.normal(size=(9,12,3));y=r.normal(size=(9,12))
    for pattern in ['RGGB','BGGR','GRBG','GBRG']:
        masks=cfa_masks(y.shape,pattern)
        assert np.all(sum(m.astype(int) for m in masks)==1)
        np.testing.assert_allclose(np.vdot(sample_rgb(x,pattern),y),np.vdot(x,sampling_adjoint(y,pattern)),rtol=1e-14,atol=1e-14)

def test_polynomial_reproduction_interior_and_kernel_normalization():
    yy,xx=np.indices((12,14));rgb=np.stack([.1+.01*xx,.3+.005*yy,.7+.003*(xx+yy)],axis=-1)
    result=reconstruct(sample_rgb(rgb))
    np.testing.assert_allclose(result[2:-2,2:-2],rgb[2:-2,2:-2],rtol=1e-13,atol=1e-14)
    for k in interpolation_kernels():np.testing.assert_allclose(k.sum(),1,rtol=0,atol=1e-15)

def test_normalization_exact_units_and_errors():
    a=np.array([[31,1023],[527,279]],dtype=np.uint16)
    np.testing.assert_array_equal(normalize_mosaic(a,np.array([31.]*4)),[[0,1],[.5,.25]])
    assert WHITE_LEVEL==1023 and SHAPE==(2128,2840)
    text=(T/'README.md').read_text(encoding='utf8')
    for literal in ['1023','2128','2840','31','RGGB']:assert literal in text
    with pytest.raises(ValueError):normalize_mosaic(a,np.array([1023.]*4))
    with pytest.raises(ValueError):reconstruct(np.full((3,3),np.nan))
    with pytest.raises(ValueError):cfa_masks((4,4),'RGB')


def test_main_import_preserves_selected_notebook_backend():
    import importlib
    import matplotlib
    before=matplotlib.get_backend()
    importlib.import_module('src.main')
    assert matplotlib.get_backend()==before
