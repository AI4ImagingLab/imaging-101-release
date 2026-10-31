from pathlib import Path
import sys,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.preprocessing import load_data,prepare_band
from src.physics_model import green_observations
from src.solvers import reconstruct,design_system,weighted_solve
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_all_original_preprocessing_green_and_pipeline():
    data=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as oracle:
        for e in range(5):
            for f in range(5):
                pairs=prepare_band(data,e,f);independent=judge.prepare_band(data,e,f)
                assert [p['station_slot'] for p in pairs]==list(np.flatnonzero(data['fit_valid'][e,f]))
                for p,q in zip(pairs,independent):
                    j=p['station_slot']
                    for key in ['energy','unsmoothed','tcoda','Ecoda']:
                        np.testing.assert_allclose(p[key],oracle[f'output_{key}_{e}_{f}_{j}'],rtol=1e-10,atol=2e-6 if key=='tcoda' else 0)
                        np.testing.assert_array_equal(q[key],p[key])
                for key in ['tbulk','Ebulk']:
                    np.testing.assert_allclose([p[key] for p in pairs],oracle[f'output_{key}_{e}_{f}'],rtol=1e-10,atol=0)
                g=oracle['output_reconstruction'][0,e,f,0];gc,gb=green_observations(pairs,g)
                jgc,jgb=judge.green_observations(independent,g)
                for p,a,b in zip(pairs,gc,jgc):
                    np.testing.assert_allclose(a,oracle[f"output_Gcoda_{e}_{f}_{p['station_slot']}"],rtol=2e-6,atol=0)
                    np.testing.assert_array_equal(a,b)
                np.testing.assert_allclose(gb,oracle[f'output_Gbulk_{e}_{f}'],rtol=2e-6,atol=0)
                np.testing.assert_array_equal(gb,jgb)
        out,details=reconstruct(data)
        for key in out:np.testing.assert_allclose(out[key],oracle['output_'+key],rtol=.005,atol=0)
        expected=np.array([[r['error'] for r in row] for row in details])
        np.testing.assert_allclose(judge.forward_errors(out,data),expected,rtol=1e-12,atol=1e-12)
        for row in details:
            for r in row:
                a,energy,w=design_system(r['pairs']);g=np.r_[np.concatenate(r['Gcoda']),r['Gbulk']]
                target=np.log(energy/g);coef,error=weighted_solve(a,target,w)
                # Independent normal equation optimum and station/source gauge.
                residual=target-a@coef
                np.testing.assert_allclose(a.T@(w*residual),0,atol=1e-6)
                assert abs(np.log(r['sites']).mean())<1e-12

def test_actual_cli(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz');assert code==0,report
