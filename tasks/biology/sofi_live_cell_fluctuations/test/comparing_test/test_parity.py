import numpy as np
from pathlib import Path
from src.preprocessing import load_data,temporal_mean
from src.physics_model import moments,cumulants
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_original_stages_and_pipeline():
    raw=load_data(ROOT/'data/raw_data.npz');f=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    for i in range(3):
        block=raw[0,i,:,:24,:24];mean=temporal_mean(block);m=moments(block,mean);c=cumulants(m)
        np.testing.assert_array_equal(mean,f['output_mean'][i,:24,:24])
        for k in range(1,5):np.testing.assert_array_equal(m[k],f['output_moments'][i,k-1,:24,:24])
        np.testing.assert_allclose(c[4],f['output_reconstruction'][i,1,:24,:24],rtol=2e-12,atol=1e-5)
    out=reconstruct(raw);ref=np.load(ROOT/'evaluation/baseline_reference.npz')
    for k in out:np.testing.assert_allclose(out[k],ref[k],rtol=2e-12,atol=1e-5)
