from pathlib import Path
import numpy as np
import pytest
from src.preprocessing import load_data
from src.physics_model import PlaneStressModel

ROOT=Path(__file__).resolve().parents[2]

def model():return PlaneStressModel(load_data(ROOT/'data/raw_data.npz'))

def test_linear_displacement_affine_strain_in_both_triangle_orientations():
    m=model();g=m.acquisition.nodes
    displacement=np.c_[.001*g[:,0]+.0002*g[:,1],-.0001*g[:,0]+.002*g[:,1]]
    actual=m.strain(displacement)
    expected=np.broadcast_to(np.array([[.001,.00005],[.00005,.002]]),actual.shape)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-17)

def test_stiffness_symmetry_energy_and_equilibrium():
    m=model();E=200+50*np.sin(np.arange(800)*.1);K=m.assemble(E)
    assert np.max(abs((K-K.T).data))<1e-13
    direction=np.sin(np.arange(882)*.2)
    assert direction@(K@direction)>0
    u=m.forward(E)
    assert np.linalg.norm(K@u-m.force)/np.linalg.norm(m.force)<1e-11

def test_homogeneous_scale_includes_penalty_replacement():
    m=model();a=m.forward(np.full(800,200.));b=m.forward(np.full(800,400.))
    np.testing.assert_allclose(a,2*b,rtol=1e-12,atol=1e-17)

def test_final_acquisition_is_in_small_strain_regime():
    m=model()
    with np.load(ROOT/'test/fixtures/upstream_initial_stages.npz',allow_pickle=False) as fixture:
        principal=np.linalg.eigvalsh(m.strain(fixture['output_displacement']))
        assert np.max(abs(principal))<.01
        assert abs(np.max(abs(principal))-.002439117202456555)<1e-14

@pytest.mark.parametrize('value',[0.,-1.,np.nan])
def test_nonphysical_modulus_rejected(value):
    E=np.full(800,200.);E[5]=value
    with pytest.raises(ValueError):model().forward(E)
