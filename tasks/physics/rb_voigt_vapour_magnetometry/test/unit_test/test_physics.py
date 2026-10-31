import numpy as np
import pytest
from src.physics_model import jx,jy,jz,Hhfs,Hamiltonian,Rb85,Rb87,Rb85_D2,Rb87_D2,transmission
@pytest.mark.parametrize('spin',[.5,1.,1.5,2.5])
def test_angular_momentum_commutator(spin):
    x,y,z=jx(spin),jy(spin),jz(spin)
    np.testing.assert_allclose(x@y-y@x,1j*z,atol=2e-15)
    np.testing.assert_allclose(x@x+y@y+z@z,spin*(spin+1)*np.eye(len(x)),atol=3e-15)
@pytest.mark.parametrize('name,atom,line',[('Rb85',Rb85,Rb85_D2),('Rb87',Rb87,Rb87_D2)])
def test_zero_field_ground_hyperfine_splitting(name,atom,line):
    eigen=np.linalg.eigvalsh(Hhfs(0.,.5,atom.I))
    expected=[]
    for f in [atom.I-.5,atom.I+.5]:expected.extend([.5*(f*(f+1)-atom.I*(atom.I+1)-.75)]*int(2*f+1))
    np.testing.assert_allclose(eigen,expected,atol=2e-15)
    model=Hamiltonian(name,'D2',1.,0.)
    np.testing.assert_allclose(model.groundEnergies,np.array(expected)*atom.As+line.IsotopeShift,atol=1e-9)
def test_transverse_mode_energy_and_normalization():
    x=np.linspace(-10000,10000,101);a=transmission(x,100.,4200.,.7,.4)
    b=transmission(x,100.,4200.,7.,4.)
    np.testing.assert_allclose(a,b,rtol=1e-14,atol=1e-14)
    parallel=transmission(x,100.,4200.,1.,0.);perpendicular=transmission(x,100.,4200.,0.,1.)
    np.testing.assert_allclose(a,(.7**2*parallel+.4**2*perpendicular)/(.7**2+.4**2),atol=1e-14)
    assert np.all(a>=0) and np.all(a<=1+1e-12)
