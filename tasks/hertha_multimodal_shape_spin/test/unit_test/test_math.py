import numpy as np
import pytest
from scipy.integrate import quad_vec
from src.physics_model import triangle_fourier,octantoid,rotation_matrix,mesh_penalties
from src.preprocessing import image_fourier
from src.solvers import initial_parameters,octantoid_penalty

def test_triangle_integral_against_independent_barycentric_quadrature():
    vertices=np.array([[-.4,.2],[1.2,-.1],[.1,1.5]]);freq=np.array([[.11,.23],[0.,.19],[.21,0.],[.3,.3]])
    result,derivative=triangle_fourier(freq,vertices)
    determinant=np.linalg.det((vertices[1:]-vertices[0]).T)
    expected=np.array([determinant*quad_vec(lambda a:quad_vec(lambda b:np.exp(-2j*np.pi*f@(vertices[0]+a*(vertices[1]-vertices[0])+b*(vertices[2]-vertices[0]))),0,1-a,epsabs=1e-12)[0],0,1,epsabs=1e-12)[0] for f in freq])
    np.testing.assert_allclose(result,expected,atol=1e-12,rtol=1e-12)
    for j in range(6):
        delta=np.zeros(6);delta[j]=1e-5
        plus=triangle_fourier(freq,vertices.ravel()+delta)[0];minus=triangle_fourier(freq,vertices.ravel()-delta)[0]
        np.testing.assert_allclose(derivative[:,j],(plus-minus)/2e-5,rtol=2e-8,atol=2e-10)

def test_fourier_phase_center_and_dc_normalization_known_image():
    image=np.zeros((8,8));image[4,4]=17
    signal,frequency=image_fourier(image,pixel_scale=.5)
    np.testing.assert_allclose(signal,np.ones(39),atol=1e-15)
    image[4,4]=0;image[4,5]=17
    shifted,_=image_fourier(image,pixel_scale=.5)
    np.testing.assert_allclose(shifted,np.exp(-2j*np.pi*.5*frequency[:,0]),atol=1e-15)

@pytest.mark.parametrize('image',[np.zeros((4,4)),np.zeros((3,4)),np.full((4,4),np.nan)])
def test_fourier_invalid_input(image):
    with pytest.raises(ValueError):image_fourier(image)

def test_harmonic_octantoid_penalty_gradient():
    a=initial_parameters()[:300];a[151]=.02;a[299]=-.03
    value,gradient=octantoid_penalty(a)
    assert value>0
    for j in [0,100,151,200,299]:
        d=np.zeros(300);d[j]=1e-6
        fd=(octantoid_penalty(a+d)[0]-octantoid_penalty(a-d)[0])/2e-6
        assert abs(fd-gradient[j])<1e-11

def test_all_mesh_regularizer_gradients_at_nonconvex_known_mesh():
    p=initial_parameters();p[13]=.03;p[155]=-.02;p[299]=.01
    mesh,_=octantoid(p[:300]);values,gradient=mesh_penalties(mesh)
    for axis,vertex in [(0,33),(1,77),(2,151)]:
        delta=np.zeros_like(mesh);delta[vertex,axis]=1e-5
        fd=(mesh_penalties(mesh+delta)[0]-mesh_penalties(mesh-delta)[0])/2e-5
        np.testing.assert_allclose(gradient[:,axis*402+vertex],fd,atol=2e-8,rtol=3e-5)

def test_reconstruction_import_does_not_change_notebook_backend():
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert matplotlib.get_backend()==before
