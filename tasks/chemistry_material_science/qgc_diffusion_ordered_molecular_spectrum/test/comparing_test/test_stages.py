from pathlib import Path
import numpy as np
from src.preprocessing import prepare
from src.physics_model import kernel
from src.solvers import ppxa,entropy_l1_prox,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_all_support_preprocessing_and_proximals():
    d=prepare(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as f:
        np.testing.assert_array_equal(d['observations'],f['Y']);np.testing.assert_array_equal(d['eta'],f['eta'])
        assert len(d['groups'])==29 and sum(map(len,d['groups']))==2277
        np.testing.assert_allclose(kernel(d['b_values']),f['kernel'],rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(entropy_l1_prox(f['prox_input']),f['initial_prox'],rtol=1e-10,atol=1e-12)
        for count in [1,2,5]:
            x,_=ppxa(kernel(d['b_values']),f['inverse'],d['observations'][5:6],d['eta'][5:6],nbiter=count)
            np.testing.assert_allclose(x[0],f['step_'+str(count)][:,0],rtol=1e-9,atol=1e-10)

def test_full_selected_pipeline():
    d=prepare(ROOT/'data/raw_data.npz');fresh,detail=reconstruct(d)
    with np.load(ROOT/'test/fixtures/upstream_full.npz') as f:ref=f['reconstruction'][0]
    x=fresh['reconstruction'][0]
    relative=np.linalg.norm(x-ref,axis=1)/np.linalg.norm(ref,axis=1)
    cosine=np.sum(x*ref,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(ref,axis=1))
    assert relative.max()<1e-4 and cosine.min()>.99999
    assert np.max(detail['residual']/d['eta'])<=1.01
    assert np.all(x.min(axis=1)>=-1e-8*x.max(axis=1))
