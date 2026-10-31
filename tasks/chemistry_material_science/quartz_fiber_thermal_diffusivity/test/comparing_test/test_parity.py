from pathlib import Path
import subprocess,sys
import numpy as np
from src.preprocessing import load_data,native_to_vertical
from src.solvers import reconstruct
from src.main import run
from src.visualization import profile_figure
ROOT=Path(__file__).resolve().parents[2]
def test_all_original_stages():
    a,p=load_data(ROOT/'data/raw_data.npz');out,rows=reconstruct(a,p)
    with np.load(ROOT/'test/fixtures/upstream_stages.npz') as z:
        for i,r in enumerate(rows):
            av,pv=native_to_vertical(a[i],p[i]);np.testing.assert_array_equal(av,z[f'input_amplitude_{i}']);np.testing.assert_array_equal(pv,z[f'input_phase_{i}'])
            mapping={'angle':'FibRotAngle','center':'Center','distance':'LinDist','amplitude':'rotated_amplitude','phase':'rotated_phase','diffusivity':'ThermalDiff'}
            for k,v in mapping.items():np.testing.assert_allclose(r[k],z[f'output_{v}_{i}'],rtol=1e-8,atol=1e-8,equal_nan=True)
            for s,side in enumerate(('Top','Bot')):
                for m,mode in enumerate(('Ampl','Phase')):
                    np.testing.assert_allclose(r['profiles'][s,:,m],z[f'output_Lin{mode}Data{side}_{i}'],rtol=1e-8,atol=1e-8,equal_nan=True)
                    np.testing.assert_allclose(r['coefficients'][s,m],z[f'output_{mode}FitData{side}_{i}'][:2],rtol=1e-8,atol=1e-8)
    with np.load(ROOT/'evaluation/baseline_reference.npz') as z:
        for k,v in out.items():np.testing.assert_allclose(v,z[k],rtol=1e-8,atol=1e-8)
    fig=profile_figure(rows);assert len(fig.axes)==4
def test_full_cli_and_judge(tmp_path):
    p=tmp_path/'prediction.npz';run(ROOT/'data/raw_data.npz',p)
    report=tmp_path/'nested/report.json'
    cmd=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(p),'--report',str(report)]
    assert subprocess.run(cmd,cwd=tmp_path,capture_output=True).returncode==0
    assert subprocess.run(cmd[:-1]+[str(tmp_path)],capture_output=True).returncode==2
