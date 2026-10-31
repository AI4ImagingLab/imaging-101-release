from pathlib import Path
import numpy as np
import pytest
from src.cuda_runtime import project
from src.preprocessing import load_data,weights_and_directions
from src.physics_model import TensorAcquisition,radial_basis
ROOT=Path(__file__).resolve().parents[2]

def test_independent_axis_aligned_ray_integrals_and_constant_mean():
    image=np.arange(80,dtype='float32').reshape(8,10)
    p=project(image[None],np.array([0.]),10)[0,0]
    np.testing.assert_array_equal(p,image.sum(axis=0))
    ones=project(np.ones((1,8,10),dtype='float32'),np.array([0.]),10)[0,0]
    np.testing.assert_array_equal(ones,np.full(10,8,dtype='float32'))
    np.testing.assert_array_equal(p/ones,image.mean(axis=0))

def test_independent_tensor_contractions_and_weight_scaling():
    data={'noise_std':np.array([[1.,2.,0.]]),'diffraction_vectors':np.array([[[1.,0.,1.]],[[0.,1.,1.]],[[0.,0.,0.]]])}
    w,s=weights_and_directions(data);np.testing.assert_array_equal(w,[[1.,.5,0.]])
    expected=np.array([[1,0,.5],[0,1,.5],[0,0,0],[0,0,1],[0,0,0],[0,0,0]])
    np.testing.assert_allclose(s[:,0],expected,atol=3e-12,rtol=0)
    tensor=np.array([2.,3.,4.,.7,.5,.3]);direction=np.array([1.,1.,0.])/np.sqrt(2);matrix=np.array([[2,.7,.5],[.7,3,.3],[.5,.3,4]])
    np.testing.assert_allclose(tensor@s[:,0,2],direction@matrix@direction,atol=1e-11)

def test_matched_jacobian_adjoint_and_bound_objective_gradient():
    data=load_data(ROOT/'data/raw_data.npz');model=TensorAcquisition(data);x=np.sin(np.arange(6*model.count)).reshape(6,-1)*.001;y=np.cos(np.arange(96000)).reshape(500,192)
    lhs=np.vdot(model.jacobian@x.ravel(),y.ravel());rhs=np.vdot(x,model.adjoint(y));np.testing.assert_allclose(lhs,rhs,rtol=1e-12,atol=1e-12)
    flat=x.ravel();_,gradient=model.objective(flat)
    for phase in [0.,.7,1.9]:
        direction=np.cos(np.arange(flat.size)*.37+phase);direction/=np.linalg.norm(direction);eps=1e-3
        numeric=(model.objective(flat+eps*direction)[0]-model.objective(flat-eps*direction)[0])/(2*eps);analytic=gradient@direction
        assert abs(numeric-analytic)/max(abs(numeric),abs(analytic))<1e-4

def test_rbf_support_bias_and_constants():
    data=load_data(ROOT/'data/raw_data.npz');b=radial_basis(data['support']);assert b.shape==(10176,65)
    np.testing.assert_array_equal(b.toarray()[~data['support'].ravel()],0);np.testing.assert_array_equal(b.toarray()[:,-1],data['support'].ravel())
    with np.load(ROOT/'data/raw_data.npz') as z:assert set(z.files)=={'strain_measurements','diffraction_vectors','noise_std','angles','support'}

def test_invalid_projection_arrays():
    with pytest.raises(ValueError):project(np.ones((8,10)),np.zeros(5))

def test_cpu_ray_operator_independent_axis_sums_and_adjoint():
    from src.cpu_diagnostic import joseph_matrix
    matrix=joseph_matrix((8,10),np.array([0.]),10)
    image=np.arange(80,dtype='float32').reshape(8,10)
    np.testing.assert_array_equal(matrix@image.ravel(),image.sum(axis=0))
    np.testing.assert_array_equal(matrix@np.ones(80,dtype='float32'),np.full(10,8,dtype='float32'))
    x=np.sin(np.arange(80));y=np.cos(np.arange(10))
    np.testing.assert_allclose(np.vdot(matrix@x,y),np.vdot(x,matrix.T@y),rtol=1e-13)
