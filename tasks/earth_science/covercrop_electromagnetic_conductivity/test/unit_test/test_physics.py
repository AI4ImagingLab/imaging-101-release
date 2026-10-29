import numpy as np
from src.physics_model import cumulative_sensitivity,jacobian,forward,adjoint
from src.solvers import objective
from src.preprocessing import load_data
def test_halfspace_and_depth_partition():
    d=load_data();J=jacobian(d['layer_interfaces'],d['coil_spacing'])
    np.testing.assert_allclose(J.sum(1),1.,atol=1e-15)
    assert (J>0).all()
    np.testing.assert_allclose(forward(np.full(7,25.),d['layer_interfaces'],d['coil_spacing']),25.,atol=1e-13)
    for o in ('hcp','vcp'):
        c=cumulative_sensitivity(np.linspace(0,10,301),1.,o)
        assert c[0]==1 and np.all(np.diff(c)<0)
def test_adjoint_gradient_units_and_discretization():
    d=load_data();z=d['layer_interfaces'];s=d['coil_spacing'];J=jacobian(z,s)
    x=np.arange(7.)+4;y=np.arange(6.)-2
    np.testing.assert_allclose(forward(x,z,s)@y,x@adjoint(y,z,s),rtol=1e-13)
    np.testing.assert_allclose(jacobian(z*100,s*100),J,rtol=1e-13)
    a=d['apparent_conductivity'][0];D=np.diff(np.eye(7),axis=0);r=J@x-a
    g=(J.T@r/6+.07*D.T@D@x/7)/objective(x,a,z,s)
    eps=1e-4;num=np.array([(objective(x+eps*np.eye(7)[i],a,z,s)-objective(x-eps*np.eye(7)[i],a,z,s))/(2*eps) for i in range(7)])
    np.testing.assert_allclose(g,num,atol=2e-9)
def test_readme_constants():
    from pathlib import Path
    text=Path('README.md').read_text(encoding='utf8');d=load_data()
    assert '30000' in text and '0.25, 0.5, 0.75, 1, 1.5, 2' in text
    np.testing.assert_array_equal(d['layer_interfaces'],[.25,.5,.75,1,1.5,2])
    np.testing.assert_array_equal(d['coil_spacing'],[.32,.71,1.18,.32,.71,1.18])
