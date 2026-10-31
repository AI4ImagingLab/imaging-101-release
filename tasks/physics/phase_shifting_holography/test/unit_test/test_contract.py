from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest
from src.visualization import metrics
from src.preprocessing import load_observations
from src.physics_model import WAVELENGTH,DX,DY

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('judge',ROOT/'evaluation/judge.py')
judge=importlib.util.module_from_spec(spec); spec.loader.exec_module(judge)

def test_known_metrics():
    a=np.array([0.,2.]); b=np.array([0.,1.])
    assert judge.score_arrays(a,b)==pytest.approx(dict(ncc_vs_ref=1.,nrmse_vs_ref=np.sqrt(.5)))
    assert metrics(a,b)==pytest.approx(judge.score_arrays(a,b))
    assert judge.score_arrays(np.zeros(2),b)['ncc_vs_ref']==0.
    with pytest.raises(ValueError): judge.score_arrays(a,np.ones(2))

@pytest.mark.parametrize('case',['missing','key','shape','integer','complex','nan','inf','poor','good'])
def test_judge_contract(case,tmp_path):
    path=tmp_path/'prediction.npz'
    if case!='missing':
        if case=='shape': arr=np.zeros((1,2,2,2))
        elif case=='good':
            with np.load(ROOT/'evaluation/baseline_reference.npz') as f: arr=f['reconstruction']
        else:
            arr=np.zeros(judge.SHAPE,dtype='int32' if case=='integer' else 'complex128' if case=='complex' else 'float64')
            if case=='nan': arr.flat[0]=np.nan
            if case=='inf': arr.flat[0]=np.inf
        np.savez_compressed(path,**({'other':arr} if case=='key' else {'reconstruction':arr}))
    result,code=judge.evaluate(path)
    assert code==(0 if case=='good' else 1 if case=='poor' else 2)
    assert result['status']==('pass' if code==0 else 'fail' if code==1 else 'error')
    if code==2: assert result['metrics']=={} and result['errors']
    json.dumps(result,allow_nan=False)

def test_evaluator_missing_and_bad_threshold(tmp_path):
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path)
    assert code==2 and report['errors']
    (tmp_path/'metrics.json').write_text('{"ncc_boundary":null,"nrmse_boundary":0}')
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path)
    assert code==2 and report['metrics']=={}

@pytest.mark.parametrize('case',['missing','shape','nonfinite','constant'])
def test_evaluator_reference_failures(case,tmp_path):
    (tmp_path/'metrics.json').write_text((ROOT/'evaluation/metrics.json').read_text())
    if case!='missing':
        arr=np.ones((1,2,2,2) if case=='shape' else judge.SHAPE)
        if case=='nonfinite': arr.flat[0]=np.nan
        np.savez_compressed(tmp_path/'baseline_reference.npz',reconstruction=arr)
    report,code=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path)
    assert code==2 and report['metrics']=={} and report['errors']

def test_cli_foreign_cwd_and_report_error(tmp_path):
    report=tmp_path/'nested/report.json'
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(report)]
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True,encoding='utf8',errors='replace')
    assert result.returncode==0,result.stderr
    assert json.loads(report.read_text())['passed']
    result=subprocess.run(command[:-1]+[str(tmp_path)],cwd=tmp_path,capture_output=True,text=True,encoding='utf8',errors='replace')
    assert result.returncode==2 and 'Cannot write report' in result.stderr

def test_readme_constants_and_public_contract():
    text=(ROOT/'README.md').read_text(encoding='utf8')
    assert '632.8e-9' in text and WAVELENGTH==632.8e-9
    assert '6.9e-6' in text and DX==DY==6.9e-6
    assert '(1, 4, 1024, 1024)' in text and '(1, 2, 1024, 1024)' in text
    for domain in ['github.com','github.io','githubusercontent.com','git.io']:
        assert domain not in text.lower()
    assert load_observations(ROOT/'data/raw_data.npz').dtype==np.float64

def test_bad_load(tmp_path):
    path=tmp_path/'x.npz'; np.savez(path,holograms=np.zeros((4,4)))
    with pytest.raises(ValueError): load_observations(path)


@pytest.mark.parametrize('slope',[(0.,0.),(.123,-.321),(np.pi-.001,-np.pi+.002)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_distinct_reference_tilt_gauges_pass(slope,dtype,tmp_path):
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:h=f['holograms'][0].astype(float)
    h-=h.mean(axis=(-2,-1),keepdims=True)
    q=h[0]+1j*h[1]-h[2]-1j*h[3];yy,xx=np.indices(q.shape)
    c=q*np.exp(1j*(slope[0]*(xx-512)+slope[1]*(yy-512)))
    a=np.stack((c.real,c.imag))[None].astype(dtype)
    path=tmp_path/'gauge.npz';np.savez(path,reconstruction=a)
    report,code=judge.evaluate(path)
    assert code==0,report

@pytest.mark.parametrize('corruption',['nonlinear_phase','magnitude_only','wrong_scale','global_phase'])
def test_preserved_magnitude_cannot_hide_wrong_complex_field(corruption,tmp_path):
    with np.load(ROOT/'data/raw_data.npz',allow_pickle=False) as f:h=f['holograms'][0].astype(float)
    h-=h.mean(axis=(-2,-1),keepdims=True)
    q=h[0]+1j*h[1]-h[2]-1j*h[3];yy,xx=np.indices(q.shape)
    if corruption=='nonlinear_phase':c=q*np.exp(1j*.3*np.sin(xx/31))
    elif corruption=='magnitude_only':c=np.abs(q).astype(complex)
    elif corruption=='wrong_scale':c=q/4
    else:c=q*np.exp(1j*.3)
    path=tmp_path/'bad.npz';np.savez(path,reconstruction=np.stack((c.real,c.imag))[None])
    report,code=judge.evaluate(path)
    assert code==1,report
