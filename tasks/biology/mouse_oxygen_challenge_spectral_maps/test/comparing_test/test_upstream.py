from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct,unmix
from src.physics_model import forward,oxygenation
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_original_stages_and_all76_cycle_maps():
    z=np.load(ROOT/'test/fixtures/upstream_stages.npz');images,spectra,labels,_,_=load_data()
    assert np.array_equal(spectra,z['input_spectra']);assert np.array_equal(np.linalg.pinv(spectra),z['pseudoinverse'])
    a=unmix(images,spectra);assert np.array_equal(a,z['original_coefficients'])
    s=oxygenation(a);assert np.array_equal(s,z['original_saturation'][:,0])
    assert np.array_equal(forward(a,spectra),z['original_spectral_projection'])
    masked,ms=reconstruct(images,spectra,labels);b=np.load(ROOT/'evaluation/baseline_reference.npz')
    assert np.array_equal(masked[None],b['reconstruction']);assert np.array_equal(ms[None],b['saturation'])

def test_main_fresh_and_standalone_judge():
    import subprocess,sys
    p=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True);assert p.returncode==0,p.stderr
    a=np.load(ROOT/'output/reconstruction.npz');b=np.load(ROOT/'evaluation/baseline_reference.npz')
    for k in a.files:assert np.array_equal(a[k],b[k])
    p=subprocess.run([sys.executable,'evaluation/judge.py','--prediction','output/reconstruction.npz','--report','output/judge_report.json'],cwd=ROOT,capture_output=True);assert p.returncode==0,p.stderr

def test_original_oxygen_enhancement_comparison():
    from src.physics_model import enhancement_summary
    z=np.load(ROOT/'test/fixtures/upstream_enhancement.npz');images,spectra,labels,_,_=load_data();_,s=reconstruct(images,spectra,labels)
    steps,mean,std,delta=enhancement_summary(s,labels==1)
    assert np.array_equal(steps,z['steps'])
    for a,name in [(mean,'baseline'),(std,'baseline_std'),(delta,'delta')]:assert np.array_equal(a[labels>0],z[name][labels>0])
