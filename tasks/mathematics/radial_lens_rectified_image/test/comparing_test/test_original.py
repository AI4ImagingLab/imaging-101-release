from pathlib import Path
import sys,numpy as np
T=Path(__file__).resolve().parents[2];sys.path.insert(0,str(T))
from src.preprocessing import load_data,normalize_points
from src.physics_model import determinant_polynomial,transform_points,rectify
from src.solvers import infer_distortion
def test_original_stages_and_geometry():
    d=load_data();p,s=normalize_points(d['line_points'])
    with np.load(T/'test/fixtures/upstream_stages.npz') as f:
        np.testing.assert_allclose(s,f['normalization'],rtol=1e-14)
        np.testing.assert_allclose(determinant_polynomial(p,d['line_ids']),f['polynomial'],rtol=1e-10,atol=1e-10)
        a=infer_distortion(d['line_points'],d['line_ids'])
        np.testing.assert_allclose(a,f['distortion'],rtol=1e-5,atol=1e-16)
        np.testing.assert_allclose(transform_points(d['line_points'],a),f['mapped_points'],rtol=0,atol=.003)

def test_complete_original_image():
    d=load_data();a=infer_distortion(d['line_points'],d['line_ids'])
    result=rectify(d['image'],a)
    with np.load(T/'test/fixtures/upstream_stages.npz') as f:
        assert np.sqrt(np.mean((result.astype(float)-f['reconstruction'])**2))<=.2
