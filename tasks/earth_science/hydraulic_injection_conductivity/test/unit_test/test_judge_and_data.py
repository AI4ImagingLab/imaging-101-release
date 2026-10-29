from pathlib import Path
import json,importlib.util,subprocess,sys
import numpy as np
import pytest
from src.preprocessing import prepare
from src.generate_data import perturb_heads
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_hydraulic_judge',ROOT/'evaluation/judge.py');judge=importlib.util.module_from_spec(spec);spec.loader.exec_module(judge)

def test_known_metrics():
    y=np.array([0.,1.,2.]);np.testing.assert_allclose(judge.field_metrics(y,y),(1.,0.),rtol=0,atol=1e-15)
    cosine,error=judge.field_metrics(np.zeros(3),y);assert cosine==0;np.testing.assert_allclose(error,np.sqrt(5/3)/2)
    with pytest.raises(ValueError):judge.field_metrics(y,np.ones(3))
    with pytest.raises((ValueError,FloatingPointError)):judge.field_metrics(np.full(3,1e308),y)

@pytest.mark.parametrize('kind',['missing','key','shape','complex','integer','float16','nan','inf'])
def test_invalid_prediction(tmp_path,kind):
    p=tmp_path/'prediction.npz';x=np.zeros((1,75,100))
    if kind=='missing':pass
    elif kind=='key':np.savez(p,wrong=x)
    else:
        if kind=='shape':x=x[0]
        if kind=='complex':x=x.astype(complex)
        if kind=='integer':x=x.astype(int)
        if kind=='float16':x=x.astype(np.float16)
        if kind in ['nan','inf']:x.flat[0]=np.nan if kind=='nan' else np.inf
        np.savez(p,reconstruction=x)
    result=judge.evaluate(p);assert result['status']=='error' and result['metrics']=={} and result['errors']

@pytest.mark.parametrize('key,value',[('ncc_boundary',True),('ncc_boundary','0.9'),('ncc_boundary',1.1),('ncc_boundary',-1.1),('nrmse_boundary',-1),('nrmse_boundary',float('nan')),('forward_head_rms_boundary',float('inf')),('forward_head_rms_boundary',False)])
def test_invalid_config(tmp_path,key,value):
    c=json.loads((ROOT/'evaluation/metrics.json').read_text());c[key]=value;(tmp_path/'metrics.json').write_text(json.dumps(c))
    result=judge.evaluate(ROOT/'output/reconstruction.npz',tmp_path);assert result['status']=='error' and result['metrics']=={}

def test_actual_truth_precedence_invalid_truth_no_fallback(tmp_path,monkeypatch):
    c=dict(ncc_boundary=.99,nrmse_boundary=.01,forward_head_rms_boundary=1.)
    (tmp_path/'metrics.json').write_text(json.dumps(c));y=np.linspace(1,4,7500).reshape(1,75,100)
    np.savez(tmp_path/'ground_truth.npz',reconstruction=y);np.savez(tmp_path/'baseline_reference.npz',reconstruction=-y)
    p=tmp_path/'prediction.npz';np.savez(p,reconstruction=y);monkeypatch.setattr(judge,'hydraulic_rms',lambda *a:0.)
    assert judge.evaluate(p,tmp_path)['status']=='pass'
    np.savez(tmp_path/'ground_truth.npz',wrong=y)
    assert judge.evaluate(p,tmp_path)['status']=='error'

@pytest.mark.parametrize('control',['zero','constant','shuffle','transpose_order'])
def test_poor_fields(control):
    y=np.load(ROOT/'evaluation/ground_truth.npz')['reconstruction'];x=y.copy()
    if control=='zero':x*=0
    if control=='constant':x.fill(y.mean())
    if control=='shuffle':x=np.random.default_rng(24).permutation(x.ravel()).reshape(x.shape)
    if control=='transpose_order':x=x[0].T.ravel().reshape(x.shape)
    ncc,error=judge.field_metrics(x,y);assert ncc<.8971711355979166 or error>.052777390088457304

def test_generator_seed_units_geometry_and_no_mutation():
    raw=dict(np.load(ROOT/'data/raw_data.npz'));original=raw['hydraulic_head'].copy();a=perturb_heads(raw,.2,42);b=perturb_heads(raw,.2,42)
    np.testing.assert_array_equal(a['hydraulic_head'],b['hydraulic_head']);np.testing.assert_array_equal(raw['hydraulic_head'],original);np.testing.assert_array_equal(a['well_indices'],raw['well_indices'])
    assert abs(np.std(a['hydraulic_head']-original)/.2-1)<.1
    for sigma in [-1,True,float('nan')]:
        with pytest.raises(ValueError):perturb_heads(raw,sigma)

def test_public_input_validation():
    raw=dict(np.load(ROOT/'data/raw_data.npz'));obs,wells=prepare(raw);assert obs.shape==(870,) and wells.shape==(30,2)
    bad={k:v.copy() for k,v in raw.items()};bad['well_indices'][0,0]=bad['well_indices'][0,1]
    with pytest.raises(ValueError):prepare(bad)

def test_cli_foreign_directory_pass_fail_error_and_report_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report',str(tmp_path/'nested/report.json')]
    p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert p.returncode==0,p.stderr
    assert json.loads((tmp_path/'nested/report.json').read_text())['passed']
    bad=tmp_path/'zero.npz';np.savez(bad,reconstruction=np.zeros((1,75,100)));command[3]=str(bad)
    assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==1
    command[3]=str(tmp_path/'missing.npz');assert subprocess.run(command,cwd=tmp_path,capture_output=True).returncode==2
    command[5]=str(tmp_path);p=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True);assert p.returncode==2 and 'Report write failure' in p.stderr
