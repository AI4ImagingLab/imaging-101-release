from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.physics_model import polynomial_kernels,relative_residual
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_kernels_and_all_stages():
    z=np.load(ROOT/'test/fixtures/upstream_stages.npz');kernels=polynomial_kernels()
    for name,derivative in [('kernel_0',(0,0)),('kernel_xx',(2,0)),('kernel_yy',(0,2))]:assert np.array_equal(kernels[derivative],z[name])
    wave,mask,f,c=load_data();v,u,lu=reconstruct(wave,f,c,mask)
    assert np.array_equal(u,z['output_filtered_wave']);assert np.array_equal(lu,z['output_laplacian'])
    assert np.array_equal(v,np.where(mask,z['output_modulus'],0))
    expected=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction']
    assert np.array_equal(v.transpose(0,3,2,1)[None],expected)

def test_fresh_pipeline_and_standalone_judge():
    import subprocess,sys
    r=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True);assert r.returncode==0,r.stderr
    p=ROOT/'output/reconstruction.npz';a=np.load(p)['reconstruction'];b=np.load(ROOT/'evaluation/baseline_reference.npz')['reconstruction'];assert np.array_equal(a,b)
    r=subprocess.run([sys.executable,'evaluation/judge.py','--prediction',str(p),'--report','output/judge_report.json'],cwd=ROOT,capture_output=True);assert r.returncode==0,r.stderr
