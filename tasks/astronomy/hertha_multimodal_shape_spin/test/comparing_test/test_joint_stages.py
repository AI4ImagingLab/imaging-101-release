from pathlib import Path
import numpy as np
from scipy.linalg import solve
from src.preprocessing import load_observations
from src.solvers import initial_parameters,residual_and_jacobian

ROOT=Path(__file__).resolve().parents[2]

def test_complete_initial_linearization_and_first_native_lower_update():
    residual,jacobian=residual_and_jacobian(initial_parameters(),load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'test/fixtures/original_joint_initial.npz',allow_pickle=False) as ref:
        np.testing.assert_allclose(residual,ref['residual'],atol=1e-10,rtol=2e-10)
        rows=ref['jacobian_rows'];actual=jacobian[rows];expected=ref['jacobian']
        # Edge-integral analytic derivatives near axis-aligned edges are
        # cancellation-sensitive across C99 and std::complex implementations.
        # Independent directional-FD and direct-integral tests separately
        # check mathematics; full normal/update tests check inverse relevance.
        for label,selection,tolerance in [('LC',rows<923,1e-10),('AO',(rows>=923)&(rows<46519),3e-7),('OC',(rows>=46519)&(rows<46591),1e-10),('regularizers',rows>=46591,1e-10)]:
            relative=np.linalg.norm(actual[selection]-expected[selection])/max(np.linalg.norm(expected[selection]),1e-30)
            assert relative<tolerance,(label,relative)
        normal=jacobian.T@jacobian;rhs=jacobian.T@residual
        assert np.linalg.norm(normal-ref['normal'])/np.linalg.norm(ref['normal'])<1e-9
        assert np.linalg.norm(rhs-ref['rhs'])/np.linalg.norm(ref['rhs'])<1e-10
        damped=normal+.1*np.diag(normal.diagonal());update=solve(damped,rhs,assume_a='sym',lower=True)
        assert np.linalg.norm(update-ref['first_update'])/np.linalg.norm(ref['first_update'])<1e-8
        assert np.linalg.norm(damped@update-rhs)/np.linalg.norm(rhs)<1e-12
