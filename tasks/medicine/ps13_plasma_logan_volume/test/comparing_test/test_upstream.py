from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import cumulative_integral,transformed_data
from src.solvers import fit_tacs,reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_integrals_and_normal_equations():
    z=np.load(ROOT/"test/fixtures/upstream_stages.npz");a=z["input_activity"];cp=z["input_cp"];s=z["input_start"];d=z["input_duration"]
    assert np.array_equal(cumulative_integral(a,s,d),z["output_integrated_activity"])
    assert np.array_equal(cumulative_integral(cp,s,d),z["output_integrated_cp"])
    x,y,w=transformed_data(a,cp,s,d)
    for i in range(len(a)):
        assert np.array_equal(x[i].T@np.diag(w)@x[i],z["output_normal_matrix"][i])
        assert np.array_equal(x[i].T@np.diag(w)@y[i,:,None],z["output_normal_rhs"][i])
    v,b=fit_tacs(a,cp,s,d)
    assert np.array_equal(v,z["output_distribution_volume"])
    assert np.array_equal(np.stack([b,v],axis=-1),z["output_fit_coefficients"][:,:,0])

def test_full_native_pipeline_original_parity():
    a,m,cp,s,d,_=load_data();v,b=reconstruct(a,m,cp,s,d)
    with np.load(ROOT/"notebooks/reference_outputs/upstream_measurements.npz") as z:assert np.array_equal(v,z["reconstruction"][0])
    assert (v[~m]==0).all()

def test_standard_main_fresh_submission():
    import subprocess,sys
    r=subprocess.run([sys.executable,"-m","src.main"],cwd=ROOT,capture_output=True);assert r.returncode==0,r.stderr
    with np.load(ROOT/"output/reconstruction.npz") as p,np.load(ROOT/"evaluation/baseline_reference.npz") as ref:assert np.array_equal(p["reconstruction"],ref["reconstruction"])
