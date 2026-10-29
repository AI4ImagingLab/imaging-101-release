from pathlib import Path
import numpy as np
from src.physics_model import load_ions,BOLTZMANN_ANGK
from src.preprocessing import extinction

def test_equilibrium_and_detailed_balance():
    ions=load_ions(Path(__file__).parents[2]/'data/raw_data.npz')
    for ion in ions.values():
        t=12000.;density=6000.;p=ion.populations(t,density);q=ion.rates(t)
        flow=density*q+ion.A
        residual=flow.T@p-p*flow.sum(axis=1)
        np.testing.assert_allclose(residual,0,atol=1e-14)
        for lower in range(ion.n-1):
            for upper in range(lower+1,ion.n):
                lhs=ion.weight[lower]*q[lower,upper]
                rhs=ion.weight[upper]*q[upper,lower]*np.exp((ion.energy[lower]-ion.energy[upper])/(BOLTZMANN_ANGK*t))
                np.testing.assert_allclose(lhs,rhs,rtol=1e-13,atol=0)

def test_extinction_limits_and_no_backend_side_effect():
    assert extinction(np.array(200000.))==0
    assert extinction(np.array(4000.))>extinction(np.array(7000.))>0
    import matplotlib
    before=matplotlib.get_backend()
    import src.main
    assert before==matplotlib.get_backend()
