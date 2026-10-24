from pathlib import Path
import numpy as np
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_fresh_full100frame_pipeline(data,judge,tmp_path):
    submission,diagnostics=reconstruct(data);path=tmp_path/'fresh.npz';np.savez_compressed(path,**submission);report=judge.evaluate(path);assert report['passed'],report
    original=np.load(ROOT/'notebooks/reference_outputs/qualified_upstream_tracks.npz')['rows'];actual=np.c_[submission['track_id'][0],submission['frame_index'][0],submission['positions'][0]]
    assert actual.shape==original.shape;np.testing.assert_array_equal(actual[:,:2],original[:,:2]);np.testing.assert_allclose(actual[:,2:],original[:,2:],rtol=0,atol=1e-6)
    assert diagnostics['statistics'].shape==(100,4)
