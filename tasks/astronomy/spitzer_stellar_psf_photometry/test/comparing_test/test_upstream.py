from pathlib import Path
import numpy as np
from src.preprocessing import load_data,prepare_psf,initialize
from src.physics_model import make_interpolator,model
from src.solvers import reconstruct

ROOT=Path(__file__).resolve().parents[2]

def test_frozen_stage_and_pipeline_parity():
    images,flight=load_data(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'test/fixtures/upstream.npz',allow_pickle=False) as upstream:
        np.testing.assert_allclose(prepare_psf(flight),upstream['calibration'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(np.stack([initialize(p) for p in images]),upstream['initial'],rtol=0,atol=1e-10)
        fitted=make_interpolator(prepare_psf(flight))
        np.testing.assert_allclose(np.stack([model(p,fitted) for p in upstream['parameters']]),upstream['model'],rtol=1e-6,atol=1e-10)
        result,models=reconstruct(images,flight)
        parameters=np.column_stack((result['flux'][0],result['positions'][0],result['background'][0]))
        np.testing.assert_allclose(parameters,upstream['parameters'],rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose(models,upstream['model'],rtol=1e-6,atol=1e-10)

def test_standard_cli_and_foreign_cwd_judge(tmp_path):
    import subprocess,sys,json,os
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    run=subprocess.run([sys.executable,'-m','src.main'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',env=env)
    assert run.returncode==0,run.stderr
    report=tmp_path/'new/report.json'
    run=subprocess.run([sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)],cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',env=env)
    assert run.returncode==0,run.stderr
    assert json.loads(report.read_text(encoding='utf-8'))['status']=='pass'
