import numpy as np,torch,pytest
from src.physics_model import rotation_matrix,normalized_grid,resample_volume,rotate_volume,binary_projection,project,objective
from src.network import build_network
from src.generate_data import read_binvox

def test_rotation_orientation_orthogonality_and_inverse():
    a=torch.tensor([.2,-.3,.4],dtype=torch.float64);R=rotation_matrix(a)
    np.testing.assert_allclose(R@R.T,np.eye(3),atol=3e-16);assert abs(torch.det(R)-1)<1e-15
    np.testing.assert_allclose(rotation_matrix(torch.tensor([np.pi/2,0,0],dtype=torch.float64))[1],[0,0,1],atol=1e-15)

def test_inclusive_grid_and_native_integer_knot_artifact():
    grid=normalized_grid(3,dtype=torch.float64);np.testing.assert_array_equal(grid[0],[-1,-1,-1]);np.testing.assert_array_equal(grid[-1],[1,1,1])
    q=torch.full((27,3),.5,dtype=torch.float64);v=torch.ones(3,3,3,dtype=torch.float64)
    np.testing.assert_allclose(resample_volume(v,q),1,atol=0)
    q[:,0]=1.;np.testing.assert_array_equal(resample_volume(v,q),0)
    q[:,0]=-1.;np.testing.assert_array_equal(resample_volume(v,q),0)

def test_trilinear_affine_reproduction_and_volume_adjoint():
    grid=normalized_grid(3,dtype=torch.float64);v=torch.arange(27,dtype=torch.float64).reshape(3,3,3).requires_grad_();q=(grid+1)*.55+.2
    y=resample_volume(v,q);expected=(9*q[:,0]+3*q[:,1]+q[:,2]).reshape(3,3,3)
    torch.testing.assert_close(y,expected,atol=1e-14,rtol=1e-14)
    weight=torch.cos(torch.arange(27,dtype=torch.float64)).reshape(3,3,3);g=torch.autograd.grad((y*weight).sum(),v)[0]
    torch.testing.assert_close((y*weight).sum(),(v*g).sum(),atol=1e-13,rtol=1e-13)

def test_forward_volume_and_angle_gradients_cpu_finite_difference():
    torch.manual_seed(8);v=torch.rand(5,5,5,dtype=torch.float64,requires_grad=True);a=torch.tensor([[.234,.427,-.358]],dtype=torch.float64,requires_grad=True)
    weight=torch.linspace(-1,1,25,dtype=torch.float64).reshape(5,5,1)
    loss=(project(v,a)*weight).sum();gv,ga=torch.autograd.grad(loss,(v,a));eps=1e-6
    direction=torch.sin(torch.arange(125,dtype=torch.float64)).reshape(5,5,5)
    fd=((project(v+eps*direction,a)-project(v-eps*direction,a))*weight).sum()/(2*eps)
    torch.testing.assert_close(fd,(gv*direction).sum(),rtol=2e-7,atol=1e-8)
    for j in range(3):
        delta=torch.zeros_like(a);delta[0,j]=eps
        fd=((project(v,a+delta)-project(v,a-delta))*weight).sum()/(2*eps)
        torch.testing.assert_close(fd,ga[0,j],rtol=2e-7,atol=1e-8)

def test_opacity_units_monotonicity_and_prior_gradient():
    v=torch.full((4,4,4),.2,dtype=torch.float64);np.testing.assert_allclose(binary_projection(v),1-np.exp(-.5*.8))
    assert torch.all(binary_projection(v*2)>binary_projection(v))
    a=torch.tensor([[.2,.3,.4]],dtype=torch.float64,requires_grad=True);prior=torch.zeros_like(a);z=torch.zeros(2,2,1,dtype=torch.float64)
    loss,data=objective(z,z,a,prior);g=torch.autograd.grad(loss,a)[0]
    torch.testing.assert_close(g,.1*torch.sin(a)/3);assert data==0

def test_complete_network_cpu_forward_backward():
    torch.manual_seed(3);net=build_network();x=torch.randn(1,3,64,64,64);y=net(x)
    assert y.shape==(1,1,64,64,64) and torch.all((y>0)&(y<1));y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())

def test_binvox_rle_axis_convention(tmp_path):
    p=tmp_path/'test.binvox';p.write_bytes(b'#binvox 1\ndim 2 2 2\ntranslate 0 0 0\nscale 1\ndata\n'+bytes([0,3,1,2,0,3]))
    expected=np.array([0,0,0,1,1,0,0,0],dtype=np.float32).reshape(2,2,2).transpose(0,2,1)
    np.testing.assert_array_equal(read_binvox(p),expected)
