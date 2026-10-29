from pathlib import Path
import numpy as np
from src.ode_integrator import step

def test_original_autonomous_and_nonautonomous_steps():
    with np.load(Path(__file__).parents[1]/'fixtures/rosenbrock_original.npz') as cases:
        y=np.array([1.,.2,3.]);diagonal=np.array([-1.,-2.,-3.])
        for name in ['autonomous','nonautonomous']:
            forcing=(lambda t:np.zeros(3)) if name=='autonomous' else (lambda t:np.array([np.sin(t),t,np.exp(-t)]))
            derivative=(lambda t,y:np.zeros(3)) if name=='autonomous' else (lambda t,y:np.array([np.cos(t),1.,-np.exp(-t)]))
            result,_,_=step(lambda t,y:diagonal*y+forcing(t),lambda t,y:np.diag(diagonal),derivative,0.,y,.01)
            np.testing.assert_allclose(result,cases[name],rtol=1e-13,atol=1e-14)

def test_residual_vector_full_observation_contract():
    from src.solvers import residual_vector
    from src.preprocessing import prepare
    root=Path(__file__).parents[2]
    data=prepare(root/'data/raw_data.npz')
    with np.load(root/'test/fixtures/upstream.npz') as f:
        x=f['initial_log_parameters']
    residual=residual_vector(x,data)
    assert residual.shape==(914,) and np.all(np.isfinite(residual))
