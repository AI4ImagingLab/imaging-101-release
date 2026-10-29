from pathlib import Path
import numpy as np
from src.solvers import ecb
from src.physics_model import rate,jacobian
ROOT=Path(__file__).resolve().parents[2]
def test_independent_R_nlsModel_stages():
    with np.load(ROOT/'test/fixtures/original_stages.npz',allow_pickle=False) as archive:
        for a,b in zip(archive['stage'],archive['summary']):
            s,v=a[:,0],a[:,1];seed=ecb(s,v)
            np.testing.assert_allclose(seed,b[1:3],rtol=1e-10,atol=1e-10)
            forward=rate(s,seed);r=v-forward
            np.testing.assert_allclose(forward,a[:,2],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(r,a[:,3],rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(r@r,b[3],rtol=1e-10,atol=1e-10)
            j=jacobian(s,seed)
            np.testing.assert_allclose(j,a[:,4:6],rtol=1e-5,atol=1e-7)
            q,rr=np.linalg.qr(j,mode='reduced');delta=np.linalg.solve(rr,q.T@r)
            np.testing.assert_allclose(delta,b[4:6],rtol=1e-5,atol=1e-6)
