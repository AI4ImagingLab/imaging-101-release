from pathlib import Path
import numpy as np
from src.preprocessing import load_data,select_lags
from src.solvers import reconstruct,reconstruct_one,estimate_intercept
from src.main import run
ROOT=Path(__file__).resolve().parents[2]
def test_independent_upstream_all_stages():
    t,g=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:
        for i in range(3):
            j=i+1;np.testing.assert_array_equal(t[i],z[f'input_time_{j}']);np.testing.assert_array_equal(g[i],z[f'input_correlation_{j}'])
            r=reconstruct_one(t[i],g[i])
            for local,up in [('time','time'),('g1','g1'),('msd','msd'),('exponent','alpha'),('omega','omega'),('storage','storage'),('loss','loss'),('parameters','params'),('window','window')]:
                np.testing.assert_allclose(r[local],z[f'output_{up}_{j}'],rtol=1e-8,atol=1e-9)
            np.testing.assert_allclose(r['intercept'],z[f'output_intercept_{j}'][0],rtol=1e-8,atol=1e-9)
def test_fresh_full_pipeline(tmp_path):
    r=run(ROOT/'data/raw_data.npz',tmp_path/'fresh.npz')
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:
        for k in z.files:np.testing.assert_allclose(r[k],z[k],rtol=1e-8,atol=1e-9)
