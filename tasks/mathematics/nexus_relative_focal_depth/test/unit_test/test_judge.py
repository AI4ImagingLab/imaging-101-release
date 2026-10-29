from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('standalone_judge',ROOT/'evaluation/judge.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)


def test_known_metric_case():
    a=np.array([0.,1.,2.]);b=np.array([0.,2.,2.])
    m=j.metrics(a,b)
    assert m['ncc']==pytest.approx(6/np.sqrt(40))
    assert m['nrmse']==pytest.approx(np.sqrt(1/3)/2)


@pytest.mark.parametrize('kind', ['zero','constant','reversed','wrong_shape','integer','complex','nan','inf','missing_key','object'])
def test_invalid_and_poor_submissions(tmp_path,kind):
    with np.load(ROOT/'evaluation/baseline_reference.npz',allow_pickle=False) as f:x=f['reconstruction']
    if kind=='zero':x=np.zeros_like(x)
    elif kind=='constant':x=np.full_like(x,.1)
    elif kind=='reversed':x=x[:,:,::-1]
    elif kind=='wrong_shape':x=x[0]
    elif kind=='integer':x=x.astype(np.int16)
    elif kind=='complex':x=x.astype(np.complex64)
    elif kind in ['nan','inf']:
        x=x.copy();x.flat[42]=np.nan if kind=='nan' else np.inf
    elif kind=='object':x=np.array([None],dtype=object)
    path=tmp_path/'submission.npz';np.savez(path,**{('other' if kind=='missing_key' else 'reconstruction'):x})
    report,code=j.evaluate(path)
    assert code==(1 if kind in ['zero','constant','reversed'] else 2)
    assert not report['passed']
    if code==2:assert report['metrics']=={} and report['errors']
    json.dumps(report,allow_nan=False)


def test_missing_prediction(tmp_path):
    report,code=j.evaluate(tmp_path/'absent.npz')
    assert code==2 and report['metrics']=={} and report['errors']


@pytest.mark.parametrize('kind',['missing_asset','malformed_config','invalid_boundary','constant_reference'])
def test_evaluator_failures(tmp_path,monkeypatch,kind):
    (tmp_path/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    if kind!='missing_asset':(tmp_path/'baseline_reference.npz').write_bytes((ROOT/'evaluation/baseline_reference.npz').read_bytes())
    if kind=='malformed_config':(tmp_path/'metrics.json').write_text('{')
    if kind=='invalid_boundary':
        c=json.loads((tmp_path/'metrics.json').read_text());c['mean_energy_excess_boundary']=float('nan')
        (tmp_path/'metrics.json').write_text(json.dumps(c))
    if kind=='constant_reference':np.savez(tmp_path/'baseline_reference.npz',reconstruction=np.ones(j.SHAPE,np.float32))
    monkeypatch.setattr(j,'HERE',tmp_path)
    report,code=j.evaluate(ROOT/'output/reconstruction.npz')
    assert code==2 and report['metrics']=={}


def test_missing_public_observations_are_evaluator_error(tmp_path,monkeypatch):
    (tmp_path/'metrics.json').write_bytes((ROOT/'evaluation/metrics.json').read_bytes())
    (tmp_path/'baseline_reference.npz').write_bytes((ROOT/'evaluation/baseline_reference.npz').read_bytes())
    np.savez(tmp_path/'ground_truth.npz',reconstruction=np.ones(j.SHAPE,np.float32))
    monkeypatch.setattr(j,'HERE',tmp_path)
    report,code=j.evaluate(ROOT/'output/reconstruction.npz')
    assert code==2 and report['errors']


def test_foreign_cwd_and_report_write_failure(tmp_path):
    command=[sys.executable,str(ROOT/'evaluation/judge.py'),'--prediction',str(ROOT/'output/reconstruction.npz'),'--report']
    p=subprocess.run([*command,str(tmp_path/'nested/report.json')],cwd=tmp_path,capture_output=True)
    assert p.returncode==0 and json.loads((tmp_path/'nested/report.json').read_text())['passed']
    blocker=tmp_path/'file';blocker.write_text('file')
    p=subprocess.run([*command,str(blocker/'report.json')],cwd=tmp_path,capture_output=True)
    assert p.returncode==2 and b'Cannot write report' in p.stderr


def test_float32_and_independent_descent_acceptance(tmp_path):
    coefficients=j.contrast_polynomials(ROOT/'data/raw_data.npz')
    with np.load(ROOT/'output/reconstruction.npz',allow_pickle=False) as z:base=z['reconstruction'].astype(float)
    before=j.field_energy(base,coefficients);u=base[0]
    derivative=np.zeros_like(u)
    for coefficient,power in zip(coefficients[:-1],range(8,0,-1)):derivative=derivative*(u/10)+power*coefficient/10
    gx=np.zeros_like(u);gy=np.zeros_like(u);gx[:,:-1]=np.diff(u,axis=1);gy[:-1]=np.diff(u,axis=0)
    norm=np.sqrt(gx*gx+gy*gy+1e-8);px=gx/norm;py=gy/norm
    gradient=-6*derivative;gradient[:,:-1]-=px[:,:-1];gradient[:,1:]+=px[:,:-1];gradient[:-1]-=py[:-1];gradient[1:]+=py[:-1]
    safe=(norm>.01)&(np.roll(norm,1,axis=0)>.01)&(np.roll(norm,1,axis=1)>.01)&(np.abs(u)<9.5)
    sparse=np.zeros_like(gradient);sparse[4:-4:8,4:-4:8]=gradient[4:-4:8,4:-4:8];sparse*=safe
    candidates=[np.clip(u-step*sparse,-10,10)[None] for step in (.1,.01,.001,.0001,.00001,.000001,.0000001,.00000001)]
    values=[j.field_energy(a,coefficients) for a in candidates]
    best=candidates[int(np.argmin(values))]
    assert min(values)<before-1e-10,(before,values)
    for candidate in (base.astype(np.float32),best):
        path=tmp_path/'alternative.npz';np.savez(path,reconstruction=candidate)
        report,code=j.evaluate(path)
        assert code==0,report
