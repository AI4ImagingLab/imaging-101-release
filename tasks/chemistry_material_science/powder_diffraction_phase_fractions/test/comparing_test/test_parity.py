from pathlib import Path
import numpy as np
from src.preprocessing import load_data
from src.solvers import reconstruct,bfgs
from src.physics_model import rwp,finite_gradient
ROOT=Path(__file__).resolve().parents[2]
def test_all_eight_original_stages_and_both_methods():
    fixture=np.load(ROOT/'test/fixtures/upstream.npz'); data=load_data(ROOT/'data/raw_data.npz')
    for method in ['BFGS','NNLS']:
        result,diagnostics=reconstruct(data,method)
        for i,d in enumerate(diagnostics,1):
            np.testing.assert_allclose(result['shifts'][0,i-1],fixture[f'shift{i}'][0],atol=2e-7,rtol=0)
            np.testing.assert_allclose(np.c_[d['grid'],d['aligned']],fixture[f'aligned{i}'],atol=.02,rtol=0)
            np.testing.assert_allclose(d['matrix'],fixture[f'matrix{i}'],atol=.02,rtol=0)
            np.testing.assert_allclose(d['coefficients'],fixture[f'{method}{i}_coefficients'],atol=5e-5,rtol=0)
            np.testing.assert_allclose(result['reconstruction'][0,i-1],fixture[f'{method}{i}_fractions'],atol=.01,rtol=0)
            np.testing.assert_allclose(result['fitted'][0,i-1],fixture[f'{method}{i}_full'],atol=.05,rtol=0)
def test_independent_initial_loss_gradient_update():
    f=np.load(ROOT/'test/fixtures/upstream.npz')
    for i in range(1,9):
        a=f[f'matrix{i}']; y=f[f'aligned{i}'][:,1]; fun=lambda x:rwp(x,a,y)
        zero=np.zeros(7)
        np.testing.assert_allclose(fun(zero),f[f'loss0_{i}'][0],atol=1e-12)
        np.testing.assert_allclose(finite_gradient(fun,zero),f[f'gradient0_{i}'],atol=1e-7,rtol=0)
        first,_=bfgs(fun,zero,maxiter=1)
        np.testing.assert_allclose(first,f[f'update1_{i}'],atol=1e-7,rtol=0)
def test_cli_pipeline():
    import subprocess,sys
    p=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True)
    assert p.returncode==0,p.stderr
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(ROOT/'output/judge_report.json')],cwd=ROOT,capture_output=True)
    assert p.returncode==0,p.stderr
