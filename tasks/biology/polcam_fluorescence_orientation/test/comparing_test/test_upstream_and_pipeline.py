from pathlib import Path
import sys,subprocess,numpy as np
from src.preprocessing import load_observations,split_channels
from src.solvers import estimate_stokes,reconstruct
from src.physics_model import polarization
ROOT=Path(__file__).resolve().parents[2]

def test_frozen_upstream_stages():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        ch=split_channels(f['input_raw']);np.testing.assert_array_equal(ch,f['output_channels'])
        s=estimate_stokes(ch);np.testing.assert_array_equal(s,f['output_stokes'])
        a,d=polarization(s);np.testing.assert_allclose(a,f['output_orientation'],rtol=0,atol=1e-14);np.testing.assert_allclose(d,f['output_degree'],rtol=0,atol=1e-14)

def test_complete_bound_upstream_parity():
    result=reconstruct(load_observations(ROOT/'data/raw_data.npz'))
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(result['reconstruction'],f['reconstruction'])
        for k in ['orientation','degree']:np.testing.assert_allclose(result[k],f[k],rtol=0,atol=1e-14)

def test_cli_and_judge_from_unrelated_directory(tmp_path):
    out=tmp_path/'fresh.npz';report=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,'-m','src.main','--input',str(ROOT/'data/raw_data.npz'),'--output',str(out)],cwd=ROOT,capture_output=True,text=True);assert p.returncode==0,p.stderr
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(out),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr
    import json
    assert json.loads(report.read_text())['status']=='pass'
