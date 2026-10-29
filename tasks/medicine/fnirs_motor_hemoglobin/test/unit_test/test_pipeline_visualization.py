from pathlib import Path
import numpy as np
import pytest,subprocess,sys,importlib
ROOT=Path(__file__).resolve().parents[2]

def test_pipeline_and_metrics(tmp_path):
    from src.main import run
    from src.visualization import metrics
    c,elapsed=run(ROOT/'data/raw_data.npz',tmp_path/'out.npz')
    assert elapsed>0 and c.shape==(1,68,2,4860)
    np.testing.assert_array_equal(c,np.load(tmp_path/'out.npz')['reconstruction'])
    y=np.array([[[[0.,1.,2.],[0.,1.,2.]]]])
    m=metrics(y,y);assert m['hbo_ncc']==pytest.approx(1.) and m['hbr_nrmse']==0.
    j=importlib.import_module('src.main');assert callable(j.run)

def test_all_public_array_bytes_preserved():
    import hashlib,json
    expected=json.loads((ROOT/'test/fixtures/input_hashes.json').read_text())
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        assert set(z)==set(expected)
        for k in z:
            assert hashlib.sha256(z[k].tobytes()).hexdigest()==expected[k]
