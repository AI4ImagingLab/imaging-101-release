from pathlib import Path
import copy,numpy as np,pytest
from src.physics_model import SpectralModel,FixedCubic,finite_convolution,rotation_kernel,normalize_continuum
from src.preprocessing import load_observations,mask_flux,select_ranges

ROOT=Path(__file__).resolve().parents[2]

def test_unit_continuum_and_finite_fill():
    np.testing.assert_allclose(finite_convolution(np.ones(11),np.array([.2,.5,.3])),1.,atol=1e-15)
    with pytest.raises(ValueError):finite_convolution(np.ones(11),np.ones(4))
    x=np.linspace(6000.,6004.,21);synthetic=np.linspace(.8,1.1,21)
    actual=normalize_continuum(synthetic,7.*synthetic,x)
    np.testing.assert_allclose(actual,7.*synthetic,rtol=1e-14,atol=1e-14)

def test_cubic_reproduces_known_polynomial():
    x=np.linspace(6000.,6004.,12);query=np.linspace(5999.9,6004.1,29);centered=x-6002
    coefficients=np.array([1.,.3,-.2,.04]);y=sum(c*centered**i for i,c in enumerate(coefficients))
    expected=sum(c*(query-6002)**i for i,c in enumerate(coefficients))
    np.testing.assert_allclose(FixedCubic(x).operator(query)@y,expected,rtol=1e-11,atol=1e-11)

def test_dynamic_rotation_is_normalized_positive():
    x=np.linspace(-1,1,301)
    for velocity in [1.,3.8,12.,50.]:
        kernel=rotation_kernel(x,6000.,velocity)
        assert np.all(kernel>=0);np.testing.assert_allclose(kernel.sum(),1.,atol=1e-15)

def test_barycentric_affine_field_and_bounds():
    raw=load_observations(ROOT/'data/raw_data.npz');model=SpectralModel(raw);test=copy.copy(model)
    points=raw['parameter_grid'];test.coefficients=(points[:,0]+2*points[:,1]-3*points[:,2])[:,None]
    p=np.array([5.777,4.44,-.1,3.8]);weight=test.interpolate_weights(p)
    np.testing.assert_allclose(weight,[p[0]+2*p[1]-3*p[2]],rtol=1e-13,atol=1e-13)
    with pytest.raises(ValueError):model.forward(np.array([10.,4.,0.,3.8]))

def test_observation_only_preprocessing_and_mask_order():
    raw=load_observations(ROOT/'data/raw_data.npz');ranked=select_ranges(raw)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as z:
        for i,key in enumerate(['wavelength','flux','uncertainty']):np.testing.assert_array_equal(ranked[key],z[f'read_spectrum_0_{i}'])
        arrays=mask_flux(ranked['wavelength'],ranked['flux'],ranked['uncertainty'],raw['masks'])
        for i,value in enumerate(arrays):np.testing.assert_array_equal(value,z[f'mask_flux_0_{i}'])
