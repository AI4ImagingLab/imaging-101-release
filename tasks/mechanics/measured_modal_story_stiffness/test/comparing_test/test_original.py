from pathlib import Path
import subprocess,sys
import numpy as np
from src.physics_model import residual_jacobian
from src.preprocessing import load_data
from src.solvers import solve,reconstruct
from src.main import run
from src.visualization import stiffness_figure
ROOT=Path(__file__).resolve().parents[2]
def test_all_original_residual_jacobian_stages():
    f,v=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:
        np.testing.assert_array_equal(f,z['input_frequency']);np.testing.assert_array_equal(v,z['input_modes'])
        for form in [1,2,3]:
            for i,x in enumerate(z[f'input_initial_{form}']):
                r,J=residual_jacobian(x,f,v,form);np.testing.assert_allclose(r,z[f'output_initial_residual_{form}'][i],rtol=1e-8,atol=1e-9);np.testing.assert_allclose(J,z[f'output_initial_jacobian_{form}'][i],rtol=1e-8,atol=1e-9)
            # Compare final analytic stages at the best original point; boundary local minima can have singular eigenvectors.
            i=np.argmin(z[f'output_objective_{form}']);r,J=residual_jacobian(z[f'output_solution_{form}'][i],f,v,form);np.testing.assert_allclose(r,z[f'output_final_residual_{form}'][i],rtol=1e-8,atol=1e-9);np.testing.assert_allclose(J,z[f'output_final_jacobian_{form}'][i],rtol=1e-8,atol=1e-9)
def test_complete_mdr_pipeline(tmp_path):
    p=tmp_path/'result.npz';fresh=run(ROOT/'data/raw_data.npz',p)
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:
        for k,v in fresh.items():np.testing.assert_allclose(v,z[k],rtol=1e-5,atol=0)
        fig=stiffness_figure(fresh['reconstruction'],z['reconstruction']);assert len(fig.axes)==1
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(tmp_path/'sub/report.json')]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert subprocess.run(cmd[:-1]+[str(tmp_path)],capture_output=True).returncode==2
