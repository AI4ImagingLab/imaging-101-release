from pathlib import Path
import json,subprocess,sys
import numpy as np
from src.preprocessing import load_raw
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_complete_fresh_pipeline_against_native_and_standalone_judge(tmp_path):
    result=reconstruct(load_raw(ROOT/'data/raw_data.npz'))
    x=result['reconstruction']
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:
        y=z['reconstruction']
    assert np.sqrt(np.mean((x-y)**2))/np.ptp(y)<=1e-6
    assert np.max(abs(x-y))<=.001
    assert result['free_dof_force_relative']<1e-8
    assert result['strain_energy_n_mm']>0
    with np.load(ROOT/'test/fixtures/native_full_stress.npz',allow_pickle=False) as z:
        y=z['native_stress']
    assert np.linalg.norm(result['gauss_stress']-y)/np.linalg.norm(y)<1e-5
    prediction=tmp_path/'fresh.npz';report=tmp_path/'reports/judge.json'
    np.savez_compressed(prediction,reconstruction=x)
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(prediction),'--report',str(report)],cwd=tmp_path,capture_output=True)
    assert run.returncode==0,run.stderr
    assert json.loads(report.read_text(encoding='utf8'))['status']=='pass'
