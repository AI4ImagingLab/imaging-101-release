"""Independent public-ODE check, including threshold passages and reversals."""
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from src.physics_model import internal_states


def test_full_path_against_independent_continuous_odes():
    fixture=Path(__file__).resolve().parents[1]/'fixtures/upstream_states.npz'
    with np.load(fixture,allow_pickle=False) as z:
        q=z['parameters'];e=z['plastic_strain']
    directions=np.sign(np.diff(e))
    assert np.all(directions!=0)
    boundaries=np.r_[0,np.flatnonzero(directions[1:]!=directions[:-1])+1,len(e)-1]
    numerical=np.zeros((len(e),5))
    state=np.r_[np.zeros(4),q[0]];numerical[0]=state
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        direction=float(directions[start]);p=np.abs(e[start:stop+1]-e[start])
        def derivative(_,z):
            out=np.empty(5)
            out[:3]=q[3:6]*(direction*np.sqrt(2/3)*q[7:10]-z[:3])
            out[3]=q[6]*(direction*np.sqrt(2/3)*q[10]-np.sign(z[3])*max(abs(z[3])-np.sqrt(2/3)*q[11],0))
            out[4]=np.sqrt(3/2)*q[1]*(1-q[2]*z[4])
            return out
        solution=solve_ivp(derivative,(0,float(p[-1])),state,method='DOP853',t_eval=p[1:],rtol=1e-11,atol=1e-9)
        assert solution.success
        numerical[start+1:stop+1]=solution.y.T
        state=solution.y[:,-1]
    alpha,isotropic,stress=internal_states(q,e)
    oriented=np.r_[0,directions]*numerical[:,4]
    np.testing.assert_allclose(alpha,numerical[:,:4],rtol=0,atol=2e-5)
    np.testing.assert_allclose(isotropic,oriented,rtol=0,atol=2e-5)
    np.testing.assert_allclose(stress,np.sqrt(1.5)*numerical[:,:4].sum(1)+oriented,rtol=0,atol=4e-5)
