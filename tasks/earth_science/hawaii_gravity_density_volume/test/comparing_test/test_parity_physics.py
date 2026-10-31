from pathlib import Path
import numpy as np
from scipy import sparse
import pytest
from src.preprocessing import load_data,prepare_observations,project_trend
from src.physics_model import mesh_geometry,gravity_matrix,G
from src.solvers import reconstruct,regularization_matrix,objective_gradient
ROOT=Path(__file__).resolve().parents[2]
@pytest.fixture(scope='module')
def oracle():
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as f:return dict(f)
@pytest.fixture(scope='module')
def run():return reconstruct(load_data(ROOT/'data/raw_data.npz'))
def test_original_preprocessing_and_coordinates(oracle):
    locations,anomaly,stages=prepare_observations(load_data(ROOT/'data/raw_data.npz'))
    np.testing.assert_allclose(locations,oracle['locations'],rtol=1e-14,atol=1e-9)
    np.testing.assert_allclose(anomaly,oracle['anomaly'],rtol=1e-13,atol=1e-8)
    np.testing.assert_allclose(stages['normal_gravity'],oracle['normal_gravity'],rtol=1e-14,atol=1e-8)
    np.testing.assert_allclose(stages['simple_bouguer'],oracle['simple_bouguer'],rtol=1e-13,atol=1e-8)
def test_original_all_prism_coefficients(run,oracle):
    np.testing.assert_allclose(run[1]['kernel'],oracle['kernel'],rtol=1e-9,atol=2e-10)
    np.testing.assert_allclose(run[1]['projected_kernel'],oracle['projected_kernel'],rtol=1e-9,atol=2e-10)
def test_original_regularization_gram_and_objective(oracle):
    original=sparse.csr_matrix((oracle['reg_data'],oracle['reg_indices'],oracle['reg_indptr']),shape=tuple(oracle['reg_shape']))
    local=regularization_matrix()
    difference=local.T@local-original.T@original
    assert not difference.nnz or np.max(np.abs(difference.data))<1e-10
    model=oracle['model'];assert np.linalg.norm(local@model)**2==pytest.approx(float(oracle['final_regularizer']),rel=1e-13)
def test_original_full_inverse(run,oracle):
    np.testing.assert_allclose(run[0].ravel(),oracle['model'],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(run[1]['prediction'],oracle['prediction'],rtol=1e-9,atol=1e-8)
    assert len(run[1]['history'])==2 and run[1]['steps'].tolist()==[1.]
    assert run[1]['history'][-1,1]==pytest.approx(float(oracle['final_misfit']),rel=1e-10)
    assert run[1]['history'][-1,3]<1e-5
def test_fixed_mesh_units_and_axis_order(oracle):
    center,bounds=mesh_geometry()
    np.testing.assert_allclose(center,oracle['cell_centers'],atol=1e-12)
    np.testing.assert_array_equal(bounds[:, :, 1]-bounds[:, :, 0],np.tile([3000.,3000.,2500.],(2048,1)))
    assert np.max(bounds[:,2])==0 and np.min(bounds[:,2])==-20000
    assert center[1,0]-center[0,0]==3000 and center[16,1]-center[0,1]==3000
def test_independent_newtonian_volume_quadrature():
    center,bounds=mesh_geometry();selected=bounds[[0,1023,2047]]
    receiver=np.array([[10000.,-5000.,50000.]])
    actual=gravity_matrix(receiver,selected)[0]
    points,weights=np.polynomial.legendre.leggauss(8)
    z,y,x=np.meshgrid(points,points,points,indexing='ij');wz,wy,wx=np.meshgrid(weights,weights,weights,indexing='ij')
    independent=[]
    for bound in selected:
        middle=bound.mean(axis=1);half=np.diff(bound,axis=1).ravel()/2
        source=np.stack([middle[0]+half[0]*x,middle[1]+half[1]*y,middle[2]+half[2]*z],axis=-1)
        displacement=source-receiver[0];radius=np.linalg.norm(displacement,axis=-1)
        independent.append(G*1e8*np.prod(half)*np.sum(wx*wy*wz*displacement[...,2]/radius**3))
    np.testing.assert_allclose(actual,independent,rtol=1e-8,atol=1e-12)
    assert np.all(actual<0)
def test_prism_symmetry_and_far_field_mass():
    bound=np.array([[[-1.,1.],[-1.,1.],[-1.,1.]]])
    receivers=np.array([[5.,3.,20.],[-5.,3.,20.],[5.,-3.,20.],[0.,0.,1000.]])
    coefficients=gravity_matrix(receivers,bound)[:,0]
    np.testing.assert_allclose(coefficients[:3],coefficients[0],rtol=1e-10,atol=1e-13)
    assert coefficients[-1]==pytest.approx(-G*1e8*8/1000**2,rel=1e-4)
def test_trend_projector_consistency(run):
    s=run[1];trend=s['trend'];matrix=s['projected_kernel']
    np.testing.assert_allclose(trend.T@s['anomaly'],0,atol=1e-7)
    np.testing.assert_allclose(trend.T@matrix,0,atol=1e-8)
    np.testing.assert_allclose(project_trend(matrix,trend),matrix,atol=1e-12)
    nuisance=trend@np.array([2.,-4.,1.])
    np.testing.assert_allclose(project_trend(s['anomaly']+nuisance,trend),s['anomaly'],atol=1e-10)
def test_likelihood_regularizer_gradient_finite_difference(run):
    s=run[1];model=np.linspace(-.1,.1,2048);args=(s['projected_kernel'],s['anomaly'],s['regularizer'])
    objective,gradient,*_=objective_gradient(model,*args)
    for index in [0,1032,2047]:
        h=1e-5;p=model.copy();m=model.copy();p[index]+=h;m[index]-=h
        fd=(objective_gradient(p,*args)[0]-objective_gradient(m,*args)[0])/(2*h)
        assert fd==pytest.approx(gradient[index],rel=1e-7,abs=1e-6)
def test_uniform_regularizer_only_smallness():
    r=regularization_matrix();result=r@np.ones(2048)
    np.testing.assert_allclose(result[:2048],50.,atol=0)
    np.testing.assert_allclose(result[2048:],0.,atol=0)
def test_normal_gravity_equatorial_and_polar_values():
    d=dict(latitude=np.array([0.,90.]),longitude=np.array([-155.275,-155.275]),elevation=np.zeros(2),observed_gravity=np.zeros(2))
    _,_,s=prepare_observations(d)
    assert s['normal_gravity'][0]==978032.53359
    assert s['normal_gravity'][1]==pytest.approx(983218.493786308,rel=1e-9)
def test_main_import_backend():
    import matplotlib
    before=matplotlib.get_backend();import src.main
    assert before==matplotlib.get_backend()
