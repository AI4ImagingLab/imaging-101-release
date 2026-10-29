from pathlib import Path
import numpy as np,json
import numpy as np
import pytest,json,hashlib
from src.main import run
from src.visualization import table_metrics
ROOT=Path(__file__).resolve().parents[2]

def test_pipeline_and_metric_independence(tmp_path):
    table,seconds=run(ROOT/'data/raw_data.npz',tmp_path/'out.npz')
    assert seconds>0
    np.testing.assert_array_equal(table,np.load(tmp_path/'out.npz')['reconstruction'])
    ref=np.array([[[0,1,2,10,2,.1],[0,4,5,20,3,.2]]],float)
    x=ref.copy();x[0,:,1]+=.1;x[0,:,3]*=1.2
    m=table_metrics(x,ref)
    assert m['detection_f1']==1. and m['position_rmse_pixels']==pytest.approx(.1) and m['mass_relative_rmse']==pytest.approx(.2)

def test_native_bytes_and_public_constants():
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as z:
        expected=json.loads((ROOT/'test/fixtures/input_hashes.json').read_text())
        assert hashlib.sha256(z['frames'].tobytes()).hexdigest()==expected['frames']
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    for literal in ('128','uint16','300','5000','CC0'):assert literal in text
