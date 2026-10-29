from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import carriers,phases,displacement,integrate_gradient
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_upstream_stages():
    r,d=load_data(ROOT/'data/raw_data.npz');z=np.load(ROOT/'test/fixtures/upstream_stages.npz');c=carriers(r)
    np.testing.assert_array_equal(c['peaks'],z['output_peaks']);assert c['calibration']==z['output_calibration']
    np.testing.assert_array_equal(c['masks'],z['output_masks']);np.testing.assert_allclose(c['reference_carriers'],z['output_reference_carriers'],rtol=0,atol=0)
    wrapped=phases(d,c,unwrap=False);np.testing.assert_array_equal(wrapped,z['output_wrapped'])
    phi=phases(d,c);delta=phi-z['output_phases'];delta-=np.median(delta,axis=(1,2),keepdims=True);np.testing.assert_allclose(delta,0,atol=1e-12)
def test_upstream_full_pipeline():
    r,d=load_data(ROOT/'data/raw_data.npz');v,diagnostics=reconstruct(r,d);z=np.load(ROOT/'test/fixtures/upstream_stages.npz')
    np.testing.assert_allclose(v,z['output_height'],rtol=1e-8,atol=1e-12)
