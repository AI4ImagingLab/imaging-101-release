from pathlib import Path
import subprocess,sys
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]
def test_full_original_parity():
    xy,v=load_data(ROOT/'data/raw_data.npz');actual=reconstruct(xy,v)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as z:expected=z['reconstruction'][0]
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
def test_cli_and_judge():
    for args in [['-m','src.main'],['evaluation/judge.py','--prediction','output/reconstruction.npz','--report','output/judge_report.json']]:
        result=subprocess.run([sys.executable]+args,cwd=ROOT,capture_output=True,text=True);assert result.returncode==0,result.stderr
