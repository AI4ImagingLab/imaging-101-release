from pathlib import Path
import numpy as np,subprocess,sys,json
from src.preprocessing import load_observations,filter_deflection
from src.physics_model import kinematics,hermite_virtual_fields
from src.solvers import reconstruct
ROOT=Path(__file__).resolve().parents[2]

def test_independent_frozen_stage_parity():
    with np.load(ROOT/'test/fixtures/upstream_stages.npz',allow_pickle=False) as f:
        a=filter_deflection(f['input_deflection'][None])[0];np.testing.assert_allclose(a,f['output_filtered_deflection'],rtol=0,atol=1e-16)
        fields=kinematics(a);ids=f['param_frame_indices'];np.testing.assert_array_equal(fields['time'],f['output_times'])
        for k in ['slope_x','slope_y','curv_xx','curv_yy','curv_xy']:np.testing.assert_allclose(fields[k][ids],f['output_'+k],rtol=1e-11,atol=1e-11)
        np.testing.assert_allclose(fields['acceleration'][ids],f['output_acceleration'],rtol=1e-11,atol=1e-9)
        for k,v in hermite_virtual_fields().items():np.testing.assert_allclose(v,f['output_virtual_'+k],rtol=1e-12,atol=1e-9)
        np.testing.assert_allclose(reconstruct(f['input_deflection'][None])['reconstruction'][0,ids],f['output_pressure'],rtol=1e-10,atol=1e-5)

def test_fresh_full_bound_original_pressure():
    a=reconstruct(load_observations(ROOT/'data/raw_data.npz'))['reconstruction']
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:np.testing.assert_allclose(a,f['reconstruction'],rtol=1e-10,atol=1e-5)

def test_standard_entrypoint_and_foreign_cwd_judge(tmp_path):
    pred=tmp_path/'fresh.npz';report=tmp_path/'nested/report.json'
    p=subprocess.run([sys.executable,'-m','src.main','--input',str(ROOT/'data/raw_data.npz'),'--output',str(pred)],cwd=ROOT,capture_output=True,text=True);assert p.returncode==0,p.stderr
    p=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(pred),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr;assert json.loads(report.read_text())['passed']
