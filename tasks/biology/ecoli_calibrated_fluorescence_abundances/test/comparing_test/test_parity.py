from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import solve_window,reconstruct

def test_upstream_window_and_stages():
    _,m=load_data()
    with np.load('test/fixtures/window.npz') as d:
        u,a,diag=solve_window(d['input_Y'],m,capture=True)
        np.testing.assert_allclose(u,d['output_U'],rtol=1e-7,atol=1e-8)
        np.testing.assert_allclose(a,d['output_A'].ravel(),rtol=1e-7,atol=1e-8)
    for i,stage in diag['snapshots'].items():
        with np.load(f'test/fixtures/stage_{i}.npz') as d:
            for k,v in stage.items():np.testing.assert_allclose(v,np.squeeze(d[k]) if np.ndim(v)==0 else d[k],rtol=1e-7,atol=1e-8,err_msg=f'{i}:{k}')

def test_fresh_complete_pipeline_parity(tmp_path):
    from src.main import run
    x,_=run(output_path=tmp_path/'submission.npz')
    with np.load('notebooks/reference_outputs/upstream_methods.npz') as d:np.testing.assert_allclose(x,d['SLPRU'][0],rtol=2e-5,atol=2e-6)
    assert x.shape==(30,30,13) and np.all(x>=0) and x.max()==1

def test_import_does_not_change_notebook_backend():
    import matplotlib,importlib
    before=matplotlib.get_backend();importlib.import_module('src.main');assert before==matplotlib.get_backend()
