from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import compare,reconstruct,direct
ROOT=Path(__file__).resolve().parents[2]
def test_public_original_conversion():
    s,v=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/original.npz') as a:
        np.testing.assert_allclose(s,a['substrate'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(v,a['rates'],rtol=0,atol=1e-12)
def test_all_96_original_results():
    s,v=load_data(ROOT/'data/raw_data.npz')
    fresh=np.array([compare(s,v),compare(s,v,True)])
    with np.load(ROOT/'test/fixtures/original.npz') as a:
        np.testing.assert_allclose(fresh[:,0],a['parameters'][:,0],rtol=0,atol=.002)
        np.testing.assert_allclose(fresh[:,1:],a['parameters'][:,1:],rtol=0,atol=1e-8)
def test_complete_primary_predictions():
    s,v=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/original.npz') as a:
        np.testing.assert_allclose(reconstruct(s,v)['fitted_rate'][0],a['fitted'],rtol=0,atol=.002)
def test_gauss_newton_descent_and_stop():
    s,v=load_data(ROOT/'data/raw_data.npz')
    for observed in v:
        p,h=direct(s,observed,return_history=True)
        assert np.all(np.diff(h[:,2])<=1e-9)
        assert h[-1,3]<=1e-5 and len(h)<=51 and np.all(p>0)
def test_standard_cli_and_judge(tmp_path):
    import subprocess,sys
    subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'report.json')],cwd=tmp_path,check=True)
