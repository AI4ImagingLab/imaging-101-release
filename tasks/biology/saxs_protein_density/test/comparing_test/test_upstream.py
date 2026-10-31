from pathlib import Path
import numpy as np,subprocess,sys,json
from src.physics_model import shell_grid,scattering_target,reciprocal_projection,real_space_update
from src.preprocessing import load_observations
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_original_independent_frozen_stages():
    data=load_observations(ROOT/'data/raw_data.npz');grid=shell_grid();target,error,selected=scattering_target(data,grid)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        np.testing.assert_allclose(grid['q'],f['qbinsc'],rtol=1e-12,atol=1e-14);np.testing.assert_array_equal(grid['labels'],f['qbin_labels'])
        np.testing.assert_allclose(target,f['Idata'],rtol=1e-12);np.testing.assert_allclose(error,f['sigqdata'],rtol=1e-12);np.testing.assert_array_equal(selected,f['qba'])
        for j in [0,1,101,201,301]:
            projected,info=reciprocal_projection(f[f'step{j}_input_rho'],target,selected,grid)
            for key,original in [('fourier','F'),('intensity','Imean'),('factors','factors')]:np.testing.assert_allclose(info[key],f[f'step{j}_{original}'],rtol=1e-10,atol=1e-12)
            np.testing.assert_allclose(projected,f[f'step{j}_rhoprime'],rtol=1e-10,atol=1e-12)
            rho,support,sigma=real_space_update(projected,f[f'step{j}_input_support'],j,float(f[f'step{j}_input_sigma']))
            np.testing.assert_allclose(rho,f[f'step{j}_newrho'],rtol=1e-10,atol=1e-12);np.testing.assert_array_equal(support,f[f'step{j}_support'])

def test_complete_independent_original_density():
    fresh=reconstruct(load_observations(ROOT/'data/raw_data.npz'),return_history=True)
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:
        b=f['reconstruction'];error=np.sqrt(np.mean((fresh['reconstruction']-b)**2))/np.ptp(b)
        assert error<1e-6
        length=len(fresh['history']['chi']);np.testing.assert_allclose(fresh['history']['chi'],f['chi_seed41'][:length],rtol=1e-10,atol=1e-10)

def test_full_cli_and_foreign_directory_judge(tmp_path):
    pred=tmp_path/'fresh.npz';report=tmp_path/'nested/report.json'
    r=subprocess.run([sys.executable,'-m','src.main','--input',str(ROOT/'data/raw_data.npz'),'--output',str(pred)],cwd=ROOT,capture_output=True,text=True);assert r.returncode==0,r.stderr
    r=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert r.returncode==0,r.stderr;assert json.loads(report.read_text())['passed']
